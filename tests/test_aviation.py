import httpx
import pytest

from flightops import aviation


@pytest.mark.parametrize(
    ("raw", "expected"),
    [("10+", 10.0), (6, 6.0), ("1 1/2", 1.5), ("3/4", 0.75), (None, None), ("bad", None)],
)
def test_parse_visibility(raw, expected):
    assert aviation.parse_visibility(raw) == expected


def metar(visib="10+", clouds=None):
    return {"visib": visib, "clouds": clouds or []}


def statuses(result):
    return {f["rule"]: f["status"] for f in result["findings"]}


def test_clear_weather_passes():
    assert aviation.evaluate_weather(metar(), 300)["overall"] == "pass"


def test_visibility_exactly_at_minimum_passes():
    assert aviation.evaluate_weather(metar(visib=3), 300)["overall"] == "pass"


def test_low_visibility_fails():
    result = aviation.evaluate_weather(metar(visib="2 1/2"), 300)
    assert result["overall"] == "fail"
    assert statuses(result)["14 CFR 107.51(c)"] == "fail"


def test_cloud_clearance_boundary():
    at_limit = metar(clouds=[{"cover": "BKN", "base": 900}])
    below_limit = metar(clouds=[{"cover": "BKN", "base": 899}])
    assert aviation.evaluate_weather(at_limit, 400)["overall"] == "pass"
    assert aviation.evaluate_weather(below_limit, 400)["overall"] == "fail"


def test_lowest_cloud_layer_is_used():
    layers = [{"cover": "OVC", "base": 5000}, {"cover": "FEW", "base": 700}]
    assert aviation.evaluate_weather(metar(clouds=layers), 400)["overall"] == "fail"


def test_altitude_over_400_fails():
    result = aviation.evaluate_weather(metar(), 500)
    assert statuses(result)["14 CFR 107.51(b)"] == "fail"


def test_missing_visibility_is_unknown_not_pass():
    result = aviation.evaluate_weather({"clouds": []}, 300)
    assert result["overall"] == "unknown"


def test_caveats_state_what_a_metar_cannot_show():
    caveats = " ".join(aviation.evaluate_weather(metar(), 300)["caveats"])
    assert "2,000 ft" in caveats


def test_class_b_at_surface_requires_authorization():
    features = [
        {
            "attributes": {
                "CLASS": "B",
                "NAME": "PHOENIX CLASS B",
                "LOWER_VAL": 0,
                "UPPER_VAL": 9000,
            }
        },
        {"attributes": {"CLASS": "E", "NAME": "E5", "LOWER_VAL": 1200, "UPPER_VAL": -9998}},
    ]
    result = aviation.summarize_airspace(features)
    assert result["authorization_required"] is True
    assert [layer["name"] for layer in result["surface_controlled_layers"]] == ["PHOENIX CLASS B"]


def test_class_e_at_1200_ft_and_mode_c_do_not_require_authorization():
    features = [
        {"attributes": {"CLASS": "E", "NAME": "E5", "LOWER_VAL": 1200, "UPPER_VAL": -9998}},
        {"attributes": {"CLASS": None, "NAME": "MODE C", "LOWER_VAL": 0, "UPPER_VAL": 10000}},
    ]
    assert aviation.summarize_airspace(features)["authorization_required"] is False


def test_no_airspace_layers():
    assert aviation.summarize_airspace([])["authorization_required"] is False


def client_returning(response: httpx.Response) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(lambda request: response))


def test_get_metar_maps_fields():
    body = [{"icaoId": "KPHX", "rawOb": "METAR KPHX", "visib": "10+", "wspd": 4, "clouds": []}]
    result = aviation.get_metar("kphx", client_returning(httpx.Response(200, json=body)))
    assert result["station"] == "KPHX"
    assert result["wind_speed_kt"] == 4


def test_get_metar_rejects_bad_station_without_a_request():
    with pytest.raises(aviation.LookupError_):
        aviation.get_metar("not a station; drop")


@pytest.mark.parametrize("response", [httpx.Response(204), httpx.Response(200, json=[])])
def test_get_metar_with_no_data_raises(response):
    with pytest.raises(aviation.LookupError_):
        aviation.get_metar("KZZZ", client_returning(response))


def test_airspace_rejects_bad_coordinates():
    with pytest.raises(aviation.LookupError_):
        aviation.lookup_airspace(120, 0)


def test_airspace_service_error_raises():
    response = httpx.Response(200, json={"error": {"code": 500}})
    with pytest.raises(aviation.LookupError_):
        aviation.lookup_airspace(33.4, -111.9, client_returning(response))
