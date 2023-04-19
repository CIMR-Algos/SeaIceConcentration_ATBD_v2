# Algorithm Input and Output Data Definition (IODD)


### Input data

| Field | Description | Shape/Amount |
| ---   | ----------- | ------------ |
| L1B TB | L1B Brightness Temperature at C, KU and KA-bands (both H and V polarization) | full swath or section of it (Nscans, Npos) |
| L1B NeΔT | Random radiometric uncertainty of the channels | full swath or section of it (Nscans, Npos) |

### Output data

| Field | Description | Shape/Amount |
| ---   | ----------- | ------------ |
| sea-ice concentration | The SIC between 0 and 1 (0: {term}`Open Water`, 1: {term}`Consolidated Ice`) | (Nscans, Npos) |
| raw sea-ice concentration value | SIC before Open Water Filter and thresholding (can be < 0 and > 1) | (Nscans, Npos) |
| sea-ice concentration total uncertainty | The retrieval uncertainty as 1-sigma | (Nscans, Npos) |
| sea-ice concentration uncertainty components | Several components of the retrieval uncertainty (TBD, e.g. radiometric uncertainty and algorithm uncertainty) | (Nscans, Npos) |
| status flags | indicates the reasons for missing, bad, or nominal SIC values. |  (Nscans, Npos) |

Tie-points: tie-points are generated offline (e.g. once a day) as part of the Level-2 SIC chain. They are thus
by-product of the chain, that are stored and available for running the SIC chain on each CIMR orbit.

### Auxiliary data

Auxiliary data will be required if RTM correction of the brightness temperature is activated. In that case,
fields of T2m, T0m, Wind Speed, Total Column Water Vapour, Total Cloud Liquid Water, remapped to the dimensions
of the swaths and collocated in time, will be required. They can for example come from ECMWF operational
analysis and forecast system.

### Ancillary data

| Field | Description | Shape/Amount |
| ---   | ----------- | ------------ |
| land mask | A land mask | Remapped to dimensions of the swaths (can also be available in Level-1b files). |
| maximum ice | Climatological maximum ice coverage | Remapped to dimensions of the swaths. |


