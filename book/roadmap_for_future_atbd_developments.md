# Roadmap for future ATBD development

Sevearl algorithmic steps that were not implemented during DEVALGO have the potential to improve the accuracy of CIMR L2 SIC and SIED products. We highlight some of them in this section.

## RTM-based correction of the atmosphere and wind-roughening contribution to the TB signal

The contrast between open water and sea ice is largest at low frequency. One of the reasons is that wind roughening of the ocean surface, as well as emissivity and scattering
in the atmosphere contribute to the brightness temperature recorded by the sensor. Given auxiliary information (analysis and forecast fields of from ECMWF) and a radiative transfer
model, this contribution can be estimated and substracted from the TB signal to yield *no-wind surface* TBs. Using these TBs in the SIC algorithm has been shown to yield
more accurate SICs.

The atmospheric correction of TB was first introduced by {cite:t}`andersen:2006:nwp` and later refined by {cite:t}`tonboe:2016:sicv1` and {cite:t}`lavergne:2019:sicv2` (Sect. 3.4.1). These authors used a {term}`RTM`
(typically those of {cite:t}`wentz:1997:rtm`) and auxiliary fields from Numerical Weather Prediction models (e.g. T2m, wind speed, total columnar water vapour, etc...) to correct TBs for the
contribution of the atmosphere and ocean surfaces to the TOA signal. The effect in reducing SIC uncertainty is noticeable for algorithms using only K-, Ka- or W-band imagery (Fig. 6, {cite:t}`ivanova:2015:sicci1`)
but not so much for algorithms using C-band imagery. The RTM-based correction step has most impact over low SIC areas, and no effect over consolidated 100% SIC areas.
While mature, the technique requires a more complex flow-diagram and e.g. an internal iteration loop to implement the RTM-based correction.

Because CIMR uses several feeds to image the Earth surface, and because these feeds have different Observation Zenith Angle (OZA), the TB images of the open ocean will look "stripy", unless the TBs are
first adjusted to a reference OZA $\theta_{ref}$. Over the ocean, the OZA adjustment step will also require auxiliary data fields and radiative transfer models, so that it can be handled at the
same time as the RTM-based correction introduced above, before running the SIC algorithms described in this ATBD.

## Along-arc collocation and feed-specific `KKA` SIC algorithms

As introduced in {ref}`l1b_resampling`, along-arc resampling allows to collocate K and KA TB samples along the scanning arc of each feed (not mixing samples of different feeds together). This allows
keeping the same OZA during the collocation process. Because the Open Water tie-point will be different for each OZA, this opens the possibility to tune several SIC algorithms, one for each feed/OZA
instead of adjusting TBs to a reference OZA and tune one SIC algorithm. Initial developments during the DEVALGO project have demonstrated that it is feasible to train feed-specific `KKA` SIC algorithms,
and bypass the OZA adjustement step.

```{hint}
Along-arc collocation followed by feed-specific `KKA` algorithm is a strong candidate for the CIMR L2 NRT1H SIC and SIED product.
```

## Better and multi-scale pan-sharpening

A central step in the proposed `CKA@KA` algorithm is the pan-sharpening of the CKA SICs (as base image) with the KA SICs (as sharpener image). At present, the pan-sharpening equation
we implemented only uses the sharpener image for its improved resolution (its $\Delta_{ediges}$ in eq. {eq}`eq_pansharpen`), not its SIC values themselves. Other implementation
of the pan-sharpening method {cite:p}`kilic:2020:sic` also enter the sharpener SICs in the base image, which might have some advantages when the base SIC from `CKA` is more sensitive
to e.g. thin ice biases than the sharpening SICs from a `KA` algorithm.

Another potential improvement to the pan-sharpening step would be to combine the three SICs (from `CKA`, `KKA`, and `KA`) at once in a final SIC. These three SICs have different noise
characteristics and spatial resolutions, that could potential be optimally combined in a multi-scale pan-sharpening, rather than the two-scale pan-sharpening we used in this version.

## Better and more robust uncertainties

The uncertainty algorithms presented in this ATBD have the potential to be improved, in particular in terms of spatial / temporal correlation length scales, and the propagation of the SIC
uncertainty into the SIED uncertainty.


