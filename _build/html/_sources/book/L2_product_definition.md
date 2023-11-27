# Level-2 product definition

The CIMR Level-2 SIC, SIC1H, and SIED products are on instrument grid, thus with a ({term}`Nscanl`,{term}`Nscanp`) structure. We
describe below the main variables for each type of products. Each product file will in addition have geolocation information
(latitude and longiture) and attributes, that are not described here at this stage.

## Sea Ice Concentration (SIC3H)

{numref}`l2_sic_variables` define the content (TBC) of the Level-2 sea-ice concentration ({term}`SIC3H`) product files.
Their structure is inspired by that of the EUMETSAT OSI SAF and ESA CCI products.

```{table} NetCDF Group: Processed data (TBC) for SIC3H
:name: l2_sic_variables
|  name                  | description | units | dimensions |
|  --------------------- | ----- | ---- | ---- |
|  ice_conc              | The main SIC variable, with all filters applied | n/a [0-1] | (Nscanl,Nscanp) |
|  raw_ice_conc_values   | "raw" SIC values before the {term}`OWF` and thresholds are applied | n/a [0-1] | (Nscanl,Nscanp) |
|  total_standard_uncertainty | total uncertainty (1 sigma) of the SIC field | n/a [0-1] | (Nscanl,Nscanp) |
|  algorithm_standard_uncertainty | uncertainty (1 sigma) contribution due to the tie-point variability | n/a [0-1] | (Nscanl,Nscanp) |
|  smearing_standard_uncertainty | uncertainty (1 sigma) contribution due to renapping and pan-sharpening | n/a [0-1] | (Nscanl,Nscanp) |
|  radiometric_standard_uncertainty | uncertainty (1 sigma) contribution due to $Ne\Delta T$ | n/a [0-1] | (Nscanl,Nscanp) |
|  status_flag   | A flag indicating status of retrieval, e.g. "nominal", "over land", \... | n/a | (Nscanl,Nscanp) |
```

At this stage, it is foreseen that the SIC3H product files can have several SIC estimates, from different channel combinations. Each
SIC will have its own group, similar to that in {numref}`l2_sic_variables`.

## Sea Ice Concentration 1H (SIC1H)

The SIC1H product files will have variables similar to those in {numref}`l2_sic_variables`, possibly less of them (e.g. no uncertainty variable)
if these have to be skipped for timeliness requirements. By the same token, we expect only one SIC algorithm to run within the {term}`NRT1H`
chain and thus only one SIC group in the data file. Finally, depending on the configuration of the {term}`NRT1H` chain, selected brightness
temperature channels might also be shipped in the {term}`SIC1H` product file, to help downstream applications
within sea-ice navigation safety (e.g. regional sea-ice drift and type monitoring). Candidate microwave channels are Ku (H- and V-pol) and Ka (H- and V-pol).

```{table} NetCDF Group: Processed data (TBC) for SIC1H
:name: l2_sic1h_variables
|  name                  | description | units | dimensions |
|  --------------------- | ----- | ---- | ---- |
|  ice_conc              | The main SIC variable, with all filters applied | n/a [0-1] | (Nscanl,Nscanp) |
|  raw_ice_conc_values   | "raw" SIC values before the {term}`OWF` and thresholds are applied | n/a [0-1] | (Nscanl,Nscanp) |
|  status_flag   | A flag indicating status of retrieval, e.g. "nominal", "over land", \... | n/a | (Nscanl,Nscanp) |
|  brightness_temperature   | brightness temperature (NRT1H) for selected channels, remapped at the location of ice_conc | K | (Nband,Nscanl,Nscanp) |

```

## Sea Ice Edge (SIED)

{numref}`l2_sic_variables` define the content (TBC) of the Level-2 sea-ice edge ({term}`SIED`) product files.

```{table} NetCDF Group: Processed data (TBC) for SIED
:name: l2_sie_variables
|  name                  | description | units | dimensions |
|  --------------------- | ----- | ---- | ---- |
|  ice_edge              | The main SIED variable: 0 for open water, 1 for sea ice. | n/a | (Nscanl,Nscanp) |
|  probability_correct   | Probability ($P \in [0.5;1]$) of correct classification | n/a | (Nscanl,Nscanp) |
|  status_flag   | A flag indicating status of retrieval, e.g. "nominal", "over land", \... | n/a | (Nscanl,Nscanp) |
```

