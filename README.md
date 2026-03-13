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

## Usage

Each script in `sources/` can be run independently. Downloaded data is saved to `data/` (excluded from git).

**SBTi**
```
python sources/sbti/fetch_sbti.py
```
Downloads the latest company and target datasets as dated Excel files. Updated by SBTi every Thursday.

---

Planned sources: GBIF, CDP, IUCN Red List, Global Carbon Project
