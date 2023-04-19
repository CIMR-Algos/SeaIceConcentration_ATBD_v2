# Abstract

Sea ice forms from seawater at the interface between the ocean and the atmosphere. Its formation plays a key role for vertical exchange
of salt and heat within the upper ocean and for the global thermohaline circulation, and its melt influences near-surface stratification
of the polar and surrounding seas. The presence or absence of sea ice has large implications for human-related activities in the 
polar regions (especially the Arctic), including shipping, resource exploitation (incl. fishing) and the lifestyle of indigeneous 
communities.

The area fraction of sea ice is also commonly named the {term}`Sea Ice Concentration` (SIC). It is a unitless number (the ratio of two
areas) and is reported either as a number in the range [0;1] or as a percentage in the range [0-100%]. SIC is by far the most important
variable when it comes to weather, ocean, and sea ice forecasting in the polar regions, and virually all operational centers assimilate
satellite-based products of {term}`SIC` to guide there forecasts. Polar {term}`SIC` is a key requirement for the {term}`Copernicus Imaging Microwave Radiometer` (CIMR)
mission and a prototype {term}`SIC3H` Level-2 algorithm is detailed in the following pages.

Timely information about SIC is also critical for safe navigation in ice-infested areas, which is why a dedicated {term}`SIC1H` Level-2
algorithm is needed for the {term}`CIMR` mission. In addition, Sea Ice Edge ({term}`SIED`) is a binary product derived from SIC to quickly
indicate where there is sea ice and where the ocean is free for ice. It is typically derived from a simple binary test involving a SIC
threshold, classically 15% SIC (all regions with SIC larger than the threshold are considere
ice-covered). From the binary product, one can derive the contour polygon of the ice cover. A prototype Level-2 {term}`SIED` algorithm is
thus described in the following pages.

