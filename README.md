# climate_tools

Scripts to source and analyze climate and biodiversity datasets.

## Setup

```
pip install -r requirements.txt
```

## Datasources

| Source | Folder | Description |
|---|---|---|
| [SBTi](https://sciencebasedtargets.org/target-dashboard) | `sources/sbti/` | Science Based Targets initiative — corporate emissions targets |
| [Trase](https://trase.earth/open-data) | `sources/trase/` | Supply chain and deforestation data — 186 datasets across 10 countries and 10 commodities |

## Usage

Each script in `sources/` can be run independently. Downloaded data is saved to `data/` (excluded from git).

**SBTi**
```
python sources/sbti/fetch_sbti.py
```
Downloads the latest company and target datasets as dated Excel files. Updated by SBTi every Thursday.

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
