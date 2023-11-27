# Introduction, purpose and scope

{term}`Sea Ice Concentration` (SIC), aka sea-ice area fraction, measures the fraction of a pre-defined ocean area (sensor footprint, grid cell,...) that is covered by sea ice.
SIC is unitless and can equally be reported in the range [0,1] as well as [0%,100%].

The retrieval of SIC from space-borne passive microwave radiometer (PMR) data has a long and successful history {cite:p}`comiso:1986:sic,cavalieri:1984:sic`. The retrieval is based
on the contrast in emissivity (ε), and thus brightness temperature ({term}`TB`), between open water (low ε, low TB) and sea ice (high ε, high TB). This contrast depends on frequency,
polarization, and type of sea-ice observed, as illustrated in {numref}`fig_emis`.

```{figure} static_imgs/sea_ice_emis.png
---
name: fig_emis
---
Average top of atmosphere brightness temperatures (TB) and standard deviations of Arctic open water ({term}`OW`), first-year ({term}`FYI`) and multiyear ({term}`MYI`) sea ice.
Data from 6.9 to 89 GHz based on AMSR-E/2 observations with incidence angle 55°, at 1.4 GHz on SMOS observations averaged over 50° to 55° incidence angle, collected for footprints
with pure surface types in the Round Robin data package of the ESA Climate Change Initiative project on sea ice {cite:p}`pedersen:2021:rrdp`. Solid lines denote vertically
polarized TBs, and H horizontally. Note that lines connecting the TBs are meant for easy reading, but not for interpolation between the observation frequencies. Source: {cite:t}`lu:2018:emis`.
```

On {numref}`fig_emis` it appears clearly that better contrast between 0% SIC (Open Water, {term}`OW`) and 100% SIC (split here in two Arctic sea-ice types sea-ice  First Year Ice, {term}`FYI`,
and Multiyear Ice, {term}`MYI`) is achieved with horizontal polarization (H-pol) radiation, and at low frequencies, e.g. L-band (1.4 GHz), C-band (6.9 GHz), and X-band (10.7 GHz).

However, with real-aperture space-borne PMR sensors like {term}`SSMIS`, {term}`AMSR2`, and also {term}`CIMR`, the low frequency channels are those with coarsest spatial resolution. Even in
the case of CIMR, the Level-2 SIC product cannot rely only on the L- or C-band imagery as the resulting spatial resolution would be too coarse.

```{important}
The crux of designing SIC algorithms for passive microwave satellite missions is to achieve high retrieval accuracy (which calls for using the low-frequency channels) and high spatial
resolution (which calls for using the higher-frequency channels) at the same time.
```

This is further illustrated on {numref}`fig_crux`.

```{figure} static_imgs/sic_crux_diagram.png
---
name: fig_crux
---
Illustration how spatial resolution (left y-axis) improves with increasing frequencies (x-axis) for the main three classes of passive microwave radiometer missions
(SSM/I and {term}`MWI` in purple, AMSR-E/2/3 in orange, and CIMR in yellow). The SIC retrieval uncertainties (right y-axis) get larger with frequency (blue line).
The dashed horizontal line represents the objective for CIMR mission in terms of SIC: less than 5 km and less than 5% uncertainty.
```

{numref}`fig_crux` illustrates how the SSMIS and MWI do not meet the CIMR objectives in terms of resolution. AMSR-E/2/3 can meet the objective of spatial resolution,
but not accuracy, by using its 89 GHz imagery channels. CIMR can meet both resolution and accuracy requirements, pending appropriate algorithms are designed and adopted.

Key assets of {term}`CIMR` in terms of SIC monitoring are:
1. high resolution (4-5 km) capabilities at Ku- and Ka-band, two microwave frequencies that are at the core of most state-of-the-art sea-ice concentration algorithms today;
2. medium resolution (15 km) capabilities at C-band, a channel where the the atmosphere is mostly transparent and the sea-ice emissivities do not vary too much with sea-ice type;
3. the swath width and especially the "no hole at the pole" capability to improve coverage of sea ice monitoring in the Arctic;
4. the availability of a forward and a backward scan to improve the accuracy of retrievals.

CIMR has a requirement to provide some Level-2 products with 1 hour latency (see {term}`Near Real Time 1H`), including SIC, to
support safety at sea in Arctic regions. This requirement is translated in a dedicated {term}`SIC1H` Level-2 product that has to achieve
high resolution within a short latency. This might lead to reduced accuracy wrt to the nominal {term}`SIC3H` product. Typically, it means
the SIC1H algorithm should focus on Ku- and Ka-band imagery and potential skip some parts of the {term}`SIC3H` algorithm, e.g. atmospheric correction
of the brightness temperatures or deriving uncertainty estimates.

{term}`Sea Ice Edge` is a binary surface classification product derived from SIC. Users of the {term}`SIED` product typically do not need
to know how consolidated the sea-ice cover is, just that sea ice significantly covers an area. {term}`SIED` can for example be used to mask
FoVs and regions where other algorithms are not to be processed.

This document presents prototype algorithms for three CIMR Level-2 products:
1. the nominal Sea Ice Concentration (SIC) Level-2 product ({term}`SIC3H`);
2. the nominal Sea Ice Edge (SIED) Level-2 product ({term}`SIED`);
3. the Sea Ice Concentration (SIC) Level-2 product with {term}`Near Real Time 1H` latency requirement.

