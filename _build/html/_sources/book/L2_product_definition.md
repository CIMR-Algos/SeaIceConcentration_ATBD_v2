# Level-2 product definition

```{important}
In the table below, we use *ESA2 grid (nx,ny)* for the shape of the L2 product fields, thus a *gridded* SIC and SIED L2 product. This corresponds
to the current implementation, in the later sections of this ATBD. However, most of the SIC (and SIED) algorithm is applied in *swath* projection.
Later versions of the algorithm could thus prepare *swath*-based L2 products.
```
Because the SIED algorithm runs from the final SIC field, and to minimize the number of product files, we store the SIED fields
in the same file as the SIC fields.

## Combined Sea Ice Concentration and Edge L2 product (NRT3H)

{numref}`l2_sic_variables` define the content of the Level-2 {term}`NRT3H` combined sea-ice concentration and sea-ice edge product file.
The structure is inspired by that of the EUMETSAT OSI SAF and ESA CCI products.

```{table} NetCDF Group: Processed data (TBC) for SIC and SIED NRT3H
:name: l2_sic_variables
|  name                  | description | units | dimensions |
|  --------------------- | ----- | ---- | ---- |
|  ice_conc              | The main SIC variable, with all filters applied | % [0 - 100] | EASE2 grid (nx,ny) |
|  ice_edge              | The main SIED variable (binary field)  | n/a [0 or 1] | EASE2 grid (nx,ny) |
|  raw_ice_conc_values   | "raw" SIC values before the {term}`OWF` and thresholds are applied | % [-20 - 120]| EASE2 grid (nx,ny) |
|  total_standard_uncertainty | total uncertainty (1 sigma) of the SIC field | % [0 - 100] | EASE2 grid (nx,ny)|
|  probability_correct   | Probability ($P \in [0.5;1]$) of correct classification | n/a | EASE2 grid (nx,ny) |
|  status_flag   | A flag indicating status of retrieval, e.g. "nominal", "over land", \... | n/a | EASE2 grid (nx,ny) |
```

```{note}
At this stage, an hypothesis is that the SIC3H product files could hold several SIC estimates (from different algorithms). Each
SIC would then possibly have its own group, similar to that in {numref}`l2_sic_variables`.
```

## Combined Sea Ice Concentration and Edge L2 product (NRT1H)

The SIC1H product files will have variables similar to those in {numref}`l2_sic_variables`, possibly less of them (e.g. no uncertainty variable)
if these have to be skipped for timeliness requirements. By the same token, we expect only one SIC algorithm to run within the {term}`NRT1H`
chain and thus only one SIC group in the data file. Finally, depending on the configuration of the {term}`NRT1H` chain, selected brightness
temperature channels might also be included in the {term}`SIC1H` product file, to help downstream applications
within sea-ice navigation safety (e.g. regional sea-ice drift and type monitoring). Candidate microwave channels are K (H- and V-pol) and KA (H- and V-pol).

```{table} NetCDF Group: Processed data (TBC) for SIC and SIED NRT1H
:name: l2_sic1h_variables
|  name                  | description | units | dimensions |
|  --------------------- | ----- | ---- | ---- |
|  ice_conc              | The main SIC variable, with all filters applied | % [0 - 100] | EASE2 grid (nx,ny) |
|  ice_edge              | The main SIED variable (binary field)  | n/a [0 or 1] | EASE2 grid (nx,ny) |
|  raw_ice_conc_values   | "raw" SIC values before the {term}`OWF` and thresholds are applied | % [-20 - 120]| EASE2 grid (nx,ny) |
|  status_flag   | A flag indicating status of retrieval, e.g. "nominal", "over land", \... | n/a | EASE2 grid (nx,ny) |
|  brightness_temperature   | brightness temperature (NRT1H) for selected channels, remapped at the location of ice_conc | K | EASE2 grid (nx,ny,nBands) |

```

