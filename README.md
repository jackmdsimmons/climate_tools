# climate_tools

Scripts to source and analyze climate and biodiversity datasets.

## Setup

```
pip install -r requirements.txt
```

## Analysis

| Package | Description |
|---|---|
| [`climate_attribution/`](climate_attribution/README.md) | LMDI attribution of changes in portfolio climate KPIs — how much of a move in WACI came from reallocation vs. real decarbonisation |

```
pytest tests
```

## Datasources

| Source | Folder | Description |
|---|---|---|
| [SBTi](https://sciencebasedtargets.org/target-dashboard) | `sources/sbti/` | Science Based Targets initiative — corporate emissions targets |
| [World Bank](https://datahelpdesk.worldbank.org/knowledgebase/articles/889392) | `sources/worldbank/` | Sovereign indicators — territorial emissions and GDP |
| [Trase](https://trase.earth/open-data) | `sources/trase/` | Supply chain and deforestation data — 186 datasets across 10 countries and 10 commodities |

## Usage

Each script in `sources/` can be run independently. Downloaded data is saved to `data/` (excluded from git).

**SBTi**
```
python sources/sbti/fetch_sbti.py
```
Downloads the latest company and target datasets as dated Excel files. Updated by SBTi every Thursday.

**World Bank**
```
python sources/worldbank/fetch_worldbank.py --list
python sources/worldbank/fetch_worldbank.py --indicator NY.GDP.MKTP.CD
```
Sovereign emissions and GDP series, feeding `climate_attribution`. Not yet run
against the live API — see `climate_attribution/README.md`.

**Trase**
```
python sources/trase/fetch_trase.py
```
Downloads all 186 datasets as CSV. Supports optional filters:
```
python sources/trase/fetch_trase.py --country brazil --commodity soy
python sources/trase/fetch_trase.py --dry-run
```

---

Planned sources: GBIF, CDP, IUCN Red List, Global Carbon Project
