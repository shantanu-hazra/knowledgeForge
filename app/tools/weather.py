import openmeteo_requests
import pandas as pd
import requests_cache
from retry_requests import retry

BASE_URL = "https://api.open-meteo.com/v1/forecast"

# Session/client setup has no per-request state (caching is just an HTTP
# optimization keyed by URL+params), so it's safe to build once at module
# load and reuse across calls.
_cache_session = requests_cache.CachedSession(".cache", expire_after=3600)
_retry_session = retry(_cache_session, retries=5, backoff_factor=0.2)
_client = openmeteo_requests.Client(session=_retry_session)


def get_weather(
    latitude: float,
    longitude: float,
    hourly: list[str] | str = "temperature_2m",
    **extra_params,
) -> dict:
    """
    Fetch forecast data for a single location. Fully stateless — every
    call is independent and returns everything the caller needs.

    Args:
        latitude, longitude: location coordinates.
        hourly: one variable name or a list of them, e.g.
            ["temperature_2m", "relative_humidity_2m"].
        **extra_params: passed straight through to the API, e.g.
            daily="temperature_2m_max", timezone="auto", forecast_days=3.

    Returns:
        {
            "latitude": float,
            "longitude": float,
            "elevation": float,
            "utc_offset_seconds": int,
            "hourly": [
                {"date": <iso timestamp>, "temperature_2m": <value>, ...},
                ...
            ],
        }
    """
    variable_names = [hourly] if isinstance(hourly, str) else list(hourly)

    params = {
        "latitude": latitude,
        "longitude": longitude,
        "hourly": variable_names,
        **extra_params,
    }
    response = _client.weather_api(BASE_URL, params=params)[0]

    hourly_block = response.Hourly()
    dates = pd.date_range(
        start=pd.to_datetime(hourly_block.Time(), unit="s", utc=True),
        end=pd.to_datetime(hourly_block.TimeEnd(), unit="s", utc=True),
        freq=pd.Timedelta(seconds=hourly_block.Interval()),
        inclusive="left",
    )

    data = {"date": dates}
    for i, name in enumerate(variable_names):
        data[name] = hourly_block.Variables(i).ValuesAsNumpy()

    df = pd.DataFrame(data)
    df["date"] = df["date"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")  # JSON-friendly

    return {
        "latitude": response.Latitude(),
        "longitude": response.Longitude(),
        "elevation": response.Elevation(),
        "utc_offset_seconds": response.UtcOffsetSeconds(),
        "hourly": df.to_dict(orient="records"),
    }


if __name__ == "__main__":
    result = get_weather(latitude=52.52, longitude=13.41, hourly="temperature_2m")
    print(f"Coordinates: {result['latitude']}°N {result['longitude']}°E")
    print(f"Elevation: {result['elevation']} m asl")
    print(f"Hourly points: {len(result['hourly'])}")
    print(result["hourly"][:3])