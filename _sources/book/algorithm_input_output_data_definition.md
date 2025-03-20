# Algorithm Input and Output Data Definition (IODD)

## Input Data

### Input L1 data

| Field | Description | Shape/Amount |
| ---   | ----------- | ------------ |
| L1B TB | L1B Brightness Temperature at C, K and KA-bands (both H and V polarization) | full swath or section of it (Nscans, Npos) |
| L1B TB OZA adjustment | L1B Brightness Temperature at C, K and KA-bands (both H and V polarization) | full swath or section of it (Nscans, Npos) |
| L1B NeΔT | Random radiometric uncertainty of the channels | full swath or section of it (Nscans, Npos) |
| L1B Geolocation | Latitude / Longitude | full swath or section of it (Nscans, Npos) |
| L1B OZA | Observation Zenith Angle | full swath or section of it (Nscans, Npos) |

### Auxiliary data

| Field | Description | Shape/Amount |
| ---   | ----------- | ------------ |
| land mask | A land mask | Remapped to dimensions of the swaths (can also be available in Level-1b files). |
| maximum ice | Climatological maximum ice coverage | Remapped to dimensions of the swaths. |
| tie-points | Algorithm Tie-points (algorithm coefficients) are generated offline (e.g. once a day) as part of the Level-2 SIC chain. | a few text files per day |

Additional auxiliary data will be required if RTM correction of the brightness temperature is activated. In that case,
fields of T2m, T0m, Wind Speed, Total Column Water Vapor, Total Cloud Liquid Water, remapped to the dimensions
of the swaths and collocated in time, will be required. They can for example come from ECMWF operational
analysis and forecast system.

### Input L2 data

The SIC and SIED algorithm does not require fields prepared by other L2 processors as input.

## Output Data


| Field | Description | Shape/Amount |
| ---   | ----------- | ------------ |
| sea-ice concentration | The SIC between 0 and 100% (0%: {term}`Open Water`, 1%: {term}`Consolidated Ice`) | EASE2 grid (nx,ny) |
| sea-ice edge | The SIED binary mask: 0 or 1 (0: Open Water, 1: Sea Ice) | EASE2 grid (nx,ny) |
| raw sea-ice concentration value | SIC before Open Water Filter and thresholding (can be < 0% and > 100%) | EASE2 grid (nx,ny) |
| sea-ice concentration total uncertainty | The retrieval uncertainty as 1-sigma | EASE2 grid (nx,ny) |
| sea-ice concentration uncertainty components | Several components of the retrieval uncertainty (TBD, e.g. radiometric uncertainty and algorithm uncertainty) | EASE2 grid (nx,ny) |
| status flags | indicates the reasons for missing, bad, or nominal SIC values. |  EASE2 grid (nx,ny) |

```{important}
In the table above, we used *ESA2 grid (nx,ny)* for the shape of the output fields, thus a *gridded* SIC and SIED L2 product. This corresponds
to the current implementation in the later sections of this ATBD. However, we note that most of the SIC (and SIED) algorithm is applied in *swath* projection.
Later versions of the algorithm could implement a *swath* projection as it output Shape/Amount.
```
