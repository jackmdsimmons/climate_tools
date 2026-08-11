"""
A recorded World Bank Indicators API v2 response shape.

This fixture is the contract between the live API and `parse_worldbank`. It was
written from the API's published documentation rather than captured from a live
call, so if the real response ever differs, this file is the one place to
correct -- the adapter and its tests follow from it.

Deliberately includes the awkward rows a real response contains: a regional
aggregate, a null value for a year with no data, and a row whose ISO3 code is
blank.
"""

from __future__ import annotations

GDP_PAYLOAD = [
    {
        "page": 1,
        "pages": 1,
        "per_page": 20000,
        "total": 6,
        "sourceid": "2",
        "lastupdated": "2025-07-01",
    },
    [
        {
            "indicator": {"id": "NY.GDP.MKTP.CD", "value": "GDP (current US$)"},
            "country": {"id": "US", "value": "United States"},
            "countryiso3code": "USA",
            "date": "2023",
            "value": 27360935000000,
            "unit": "",
            "obs_status": "",
            "decimal": 0,
        },
        {
            "indicator": {"id": "NY.GDP.MKTP.CD", "value": "GDP (current US$)"},
            "country": {"id": "US", "value": "United States"},
            "countryiso3code": "USA",
            "date": "2020",
            "value": 21322949280000,
            "unit": "",
            "obs_status": "",
            "decimal": 0,
        },
        {
            "indicator": {"id": "NY.GDP.MKTP.CD", "value": "GDP (current US$)"},
            "country": {"id": "DE", "value": "Germany"},
            "countryiso3code": "DEU",
            "date": "2023",
            "value": 4456081000000,
            "unit": "",
            "obs_status": "",
            "decimal": 0,
        },
        # A year with no reported figure -- must be skipped, not zero-filled.
        {
            "indicator": {"id": "NY.GDP.MKTP.CD", "value": "GDP (current US$)"},
            "country": {"id": "DE", "value": "Germany"},
            "countryiso3code": "DEU",
            "date": "2024",
            "value": None,
            "unit": "",
            "obs_status": "",
            "decimal": 0,
        },
        # A regional aggregate, not a sovereign.
        {
            "indicator": {"id": "NY.GDP.MKTP.CD", "value": "GDP (current US$)"},
            "country": {"id": "Z4", "value": "East Asia & Pacific"},
            "countryiso3code": "EAS",
            "date": "2023",
            "value": 30500000000000,
            "unit": "",
            "obs_status": "",
            "decimal": 0,
        },
        # Some rows carry no ISO3 code at all.
        {
            "indicator": {"id": "NY.GDP.MKTP.CD", "value": "GDP (current US$)"},
            "country": {"id": "XX", "value": "Not classified"},
            "countryiso3code": "",
            "date": "2023",
            "value": 123456.0,
            "unit": "",
            "obs_status": "",
            "decimal": 0,
        },
    ],
]

ERROR_PAYLOAD = [
    {
        "message": [
            {
                "id": "120",
                "key": "Invalid value",
                "value": "The provided parameter value is not valid",
            }
        ]
    }
]

EMPTY_PAYLOAD = [
    {"page": 0, "pages": 0, "per_page": 0, "total": 0, "lastupdated": "2025-07-01"},
    None,
]
