#!/usr/bin/env python
# coding: utf-8

# # A Level-2 Sea Ice Concentration (SIC3H) algorithm for CIMR
# 
# This notebook implements a prototype for a Level-2 SIC3H algorithm for the CIMR mission.
# 
# We refer to the corresponding [ATBD](https://cimr-algos.github.io/SeaIceConcentration_ATBD/intro.html) and especially the [Baseline Algorithm Definition](https://cimr-algos.github.io/SeaIceConcentration_ATBD/baseline_algorithm_definition.html#baseline-algorithm-definition).
# 
# In particular, the figure below illustrates the overall concept of the processing:
# <img src="https://cimr-algos.github.io/SeaIceConcentration_ATBD/_images/SIC_concept_diagram.png" width="75%"/>

# In[1]:


from importlib import reload

import sys
import os
import numpy as np
import xarray as xr

from scipy.ndimage import gaussian_filter

from matplotlib import pylab as plt
import cmocean

import cartopy.crs as ccrs

# local modules contain software code that implement the SIC algorithm
from sirrdp import rrdp_file
from pmr_sic import tiepoints as tp
from pmr_sic import algo as sic_algo
from pmr_sic import hybrid_algo

# prototype re-gridding toolbox to handle the L1B input
if '/home/thomasl/Work/DEVALGO/Tools/' not in sys.path:
    sys.path.insert(0, '/home/thomasl/Work/DEVALGO/Tools/')
from tools import io_handler as io
from tools import collocation as coll
from tools import l2_format as l2

# top-level configuration for all plots
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 10})


# ## Parametrize the run
# 
# Here we decide on some values that yield for the whole notebook

# In[2]:


rrdp_dir = './sirrdp'
area = 'nh'
algos = dict()
algos['CKA'] = {'channels':('tb06v', 'tb37v', 'tb37h'), 'target_band':'C'}
algos['KUKA'] = {'channels':('tb19v', 'tb37v', 'tb37h'), 'target_band':'KU'}
algos['KA'] = {'channels':('tb37v', 'tb37h'),  'target_band':'KA'}

input_test_card = 'sceps_polar_1'
#input_test_card = 'radiometric'
#input_test_card = 'geometric'
l2_grid = 'tc'

# Change location where you stored the L1B file

if input_test_card == 'geometric':    
    # DEVALGO's simulated geometric test card
    l1b_path = '/home/thomasl/Documents/DEVALGO/Simul_L1B_20230421/'
    l1b_fn = 'W_PT-DME-Lisbon-SAT-CIMR-1B_C_DME_20230417T105425_LD_20280110T114800_20280110T115700_TN.nc'
    tc_path = '/home/thomasl/Downloads/'
    tc_fn = 'test_scene_2_compressed_lowres.nc'
elif input_test_card == 'radiometric':
    # DEVALGO's simulated radiometric test card
    l1b_path = '/home/thomasl/Documents/DEVALGO/Simul_L1B_20230421/'
    l1b_fn = 'W_PT-DME-Lisbon-SAT-CIMR-1B_C_DME_20230420T103323_LD_20280110T114800_20280110T115700_TN.nc'
    tc_path = '/home/thomasl/Downloads/'
    tc_fn = 'test_scene_1_compressed_lowres.nc'
elif input_test_card == 'sceps_polar_1':
    # SCEP's simulated radiometric test card
    l1b_path = '/home/thomasl/Documents/DEVALGO/From_SCEPS_Aug2023/'
    l1b_fn = 'Ref_scen_L1B_TDS/SCEPS/SCEPS_L1B_SCEPS_Geo_Polar_Scene_1/SCEPS_l1b_sceps_geo_polar_scene_1_unfiltered_tot_minimal_no_nedt.nc'
    tc_path = '/home/thomasl/Downloads/'
    tc_fn = 'test_scene_1_compressed_lowres.nc'
else:
    raise ValueError("Unknown test_card {}".format(input_test_card))
   
proj = 'nh'
l1b_fn = os.path.join(l1b_path,l1b_fn)
print(l1b_fn)


# ## Off-line preparation : Tune the SIC algorithms
# 
# The SIC algorithms `CKA`, `KUKA`, and `KA` require tuning before they can be run to compute SIC. The tuning step involves the preparation of the OW (Open Water, 0% SIC) and CI (Consolidated Ice, 100% SIC) tie-points, as well as the optimization of the algorithm parameters, to fit the training data best.
# 
# In this implementation of the algorithm, the training data are taken from the ESA CCI Sea Ice Concentration Round Robin Data Package of Pedersen et al. (2019). The relevant data files as well as routines to parse the files are stored in module `siddrp/`.
# 
# Pedersen, Leif Toudal; Saldo, Roberto; Ivanova, Natalia; Kern, Stefan; Heygster, Georg; Tonboe, Rasmus; et al. (2019): Reference dataset for sea ice concentration. figshare. Dataset. https://doi.org/10.6084/m9.figshare.6626549.v7
# 
# In an operational setup, this tuning should happen approximately every day and be based on dynamically loaded TB data (e.g. over the last 7 days). This does not need to run again for each new incoming L1B file.

# ### Load the OW and CI training data (from ESA CCI RRDP)

# In[3]:


reload(rrdp_file)
ow_files, ci_files = rrdp_file.find_rrdp_files(rrdp_dir, area=area, years=(2007, 2013, 2018))

channels_needed = []
for alg in algos.keys():
    channels_needed += algos[alg]['channels']
channels_needed = set(channels_needed)

rrdp_pos = rrdp_file.read_RRDP_files(area, ow_files, ci_files, channels=channels_needed)


# Transfer the training data in a tie-point object. This involves some processing, like computing the tie-points and their covariance matrices. 

# In[4]:


ow_tp = tp.OWTiepoint(source='rrdp', tbs=rrdp_pos['ow'])
ci_tp = tp.CICETiepoint(source='rrdp', tbs=rrdp_pos['ci'])
ow_tp.instr = 'CIMR'
ci_tp.instr = 'CIMR'


# ### Tune the CKA, KUKA, and KA algorithms on the training data

# In[5]:


for alg in algos.keys():
    print("Tune {}".format(alg,))
    algos[alg]['algo'] = hybrid_algo.HybridSICAlgo(algos[alg]['channels'], ow_tp, ci_tp)


# ## Step 1: Load the L1B files and prepare remapped TB arrays for each algo
# 
# The three algorithms `CKA`, `KUKA`, and `KA` combine different microwave channels (e.g. `CKA` combines C-Vpol, KA-Vpol, and KA-Hpol). Since each frequency channel is not sampled at the same location nor resolution, we must prepare location-matched, resolution-matched TB arrays for each algorithm.
# 
# In this version of the algorithm, we prepare TB arrays at the spatial resolution of the coarsest channel (e.g. we prepare C-Vpol, KA-Vpol, and KA-Hpol to the resolution of C-Vpol as input to the `CKA` algorithm).
# 
# This remapping is handled by software in the `Tools/` repository (a prototype CIMR Regridding Toolbox developed in the CIMR DEVALGO study).

# In[6]:


# global definitions
tb_dict = {'tb01':'L','tb06':'C','tb10':'X','tb19':'KU','tb37':'KA',}
rev_tb_dict = {v:k for k,v in tb_dict.items()}
bands_needed = []
for alg in algos.keys():
    bands_needed += algos[alg]['channels']
bands_needed = list(set([tb_dict[b[:-1]] for b in bands_needed]))


# In[7]:


reload(io)
# read L1B. We only read the bands needed for the three algorithms
print(l1b_fn)
ker = 5
full_l1b = io.CIMR_L1B(l1b_fn, selected_bands=bands_needed, keep_calibration_view=True,)

# align scanlines using the scan angle offset
full_l1b.align_arrays_to_start_at_zero_scan_angle()

# coarsen l1b samples along the scanlines with a kernel of 5 (horns are *not* combined)
ker = 5
coarsen_l1b = full_l1b.coarsen_along_scanlines(kernel=ker)

# split into forward / backward scan
fwd_l1b, bck_l1b = coarsen_l1b.split_forward_backward_scans(method='horn_scan_angle')

# reshape by interleaving the feeds 
reshaped_fwd_l1b = fwd_l1b.reshape_interleave_feed()
reshaped_bck_l1b = bck_l1b.reshape_interleave_feed()

# Collocate the channels with a nearest neighbour approach. This step covers both the
#   definition of the target grid, and the remapping. Use the correct 'target_band' for
#   each algorithm.
fwd_l1x = dict()
bck_l1x = dict()
for alg in algos.keys():
    fwd_l1x[alg] = coll.collocate_channels(reshaped_fwd_l1b.data, algos[alg]['target_band'], method='nn')
    bck_l1x[alg] = coll.collocate_channels(reshaped_bck_l1b.data, algos[alg]['target_band'], method='nn')

# prepare TBs in the structure expected as input to the algorithm
fwd_tbs = dict()
bck_tbs = dict()
for alg in algos.keys():
    fwd_tbs[alg] = dict()
    bck_tbs[alg] = dict()
    for ch in algos[alg]['channels']:
        band = tb_dict[ch[:-1]] + '_BAND'
        varn = 'brightness_temperature_'+ch[-1]
        fwd_tbs[alg][ch] = fwd_l1x[alg][band][varn].values
        bck_tbs[alg][ch] = bck_l1x[alg][band][varn].values
            
# extract the lat/lon arrays for later use in the pan-sharpening
fwd_geo = dict()
bck_geo = dict()
for geo in algos.keys():
    fwd_geo[geo] = dict()
    bck_geo[geo] = dict()
    for ll in ('lat','lon',):
        fwd_geo[geo][ll] = fwd_l1x[geo]['geolocation'][ll].to_numpy()
        bck_geo[geo][ll] = bck_l1x[geo]['geolocation'][ll].to_numpy()


# ## Step 2: Apply the 3 algos to compute intermediate SICs
# 
# This applies the three algorithms on their respective TB arrays (obtained from the L1B remapping of the TBs). The SIC algorithms are applied separately for the forward and backward scan of the swath.

# In[8]:


fwd_res = dict()
bck_res = dict()
for alg in algos.keys():
    
    # run the algorithm to compute SIC
    fwd_res[alg] = algos[alg]['algo'].compute_sic(fwd_tbs[alg])
    bck_res[alg] = algos[alg]['algo'].compute_sic(bck_tbs[alg])
    
    # Simple visualization in swath L1X geometry

    cmap = cmocean.cm.ice
    ucmap = cmocean.cm.thermal
    vmin, vmax = (0, 1)
    umin, umax = (0, 10)
    
    fig = plt.figure(figsize=(30,16))
    axF = fig.add_subplot(1,4,1)
    cF = axF.imshow(fwd_res[alg].sic, vmin=vmin, vmax=vmax, interpolation = 'none', origin='lower', cmap=cmap)
    axF.set_title(alg + " " + "FWD" + " " + "SIC")
    plt.colorbar(cF,orientation='horizontal', pad=0.05)
    #ax.set_xticks([]); ax.set_yticks([])
    ax = fig.add_subplot(1,4,2, sharex=axF, sharey=axF)
    uF = ax.imshow(fwd_res[alg].sdev, vmin=umin, vmax=umax, interpolation = 'none', origin='lower', cmap=ucmap)
    ax.set_title(alg + " " + "FWD" + " " + "SDEV")
    plt.colorbar(uF,orientation='horizontal', pad=0.05)
    #ax.set_xticks([]); ax.set_yticks([])
    axB = fig.add_subplot(1,4,3)
    cB = axB.imshow(bck_res[alg].sic, vmin=vmin, vmax=vmax, interpolation = 'none', origin='lower', cmap=cmap)
    axB.set_title(alg + " " + "BCK" + " " + "SIC")
    plt.colorbar(cB,orientation='horizontal', pad=0.05)
    #ax.set_xticks([]); ax.set_yticks([])
    ax = fig.add_subplot(1,4,4, sharex=axB, sharey=axB)
    uB=ax.imshow(bck_res[alg].sdev, vmin=umin, vmax=umax, interpolation = 'none', origin='lower', cmap=ucmap)
    ax.set_title(alg + " " + "BCK" + " " + "SDEV")
    plt.colorbar(uB,orientation='horizontal', pad=0.05)
    #ax.set_xticks([]); ax.set_yticks([])


# From the images above, it is clear that the three algorithms see high and low concentrations in the same locations in the Test Card, which means they are returning related SIC fields that we can later combine. The uncertainties (SDEV) are as expected lower for `CKA` than for `KUKA` that are lower than those of `KA`.

# ## Step 3 : Pan-sharpening and computation of "final" Level-2 SIC
# 
# The third step deploys a pan-sharpening methodology to combine pairs of intermediate SICs into the final Level-2 SICs. In our case, the ‘base’ image can be the high-accuracy / coarse-resolution intermediate SIC from `CKA` and the ‘sharpener’ image can be the low-accuracy / fine-resolution intermediate SIC from `KA`. This results in a final SIC named `CKA@KA`.
# 
# The ATBD calls for more such pan-sharpening to happen, but for the time being we focus on `CKA@KA`.
# 
# The pansharpening equation is simply
# $$
# \begin{array}{lc}
# C_{ER} &=& \textrm{Remap}_{HR}(C_{LR}) +  \Delta_{edges} \\
#        &=& \textrm{Remap}_{HR}(C_{LR}) + ( C_{HR} - C_{HR, blurred} ) \\ 
# \end{array}
# $$
# 
# where suffix "ER" refers to enhanced resolution (the final SIC), "LR" to "low resolution" (the 'base' SIC to be pan-sharpened),
# and "HR" to "high resolution" (the 'sharpener' SIC). The equation also involves $C_{HR, blurred}$ which is C_{HR} blurred to
# the spatial resolution of $C_{LR}$. The quantity $( C_{HR} - C_{HR, blurred})$ is sometimes referred to as a $\Delta_{edges}$ as it takes
# small values everywhere but in the regions where $C_{HR}$ exhibits sharp gradients (e.g. in the Marginal Ice Zone). The $\textrm{Remap}_{HR}$ operator
# remaps the location (only the location, not the resolution) of $C_{LR}$ to those of $C_{HR}$ to enable adding the two fields together. The resulting SIC field, $C_{ER}$ is
# thus at the locations of $C_{HR}$, with the spatial resolution of $C_{HR}$ and the accuracy of $C_{LR}$ (if the pan-sharpening works perfectly).
# 
# There are thus 3 steps for building the pan-sharpened SIC $C_{ER}$:
# 1. Regrid 'base' SIC (coarse resolution) to 'sharpener' SIC (high resolution) grid
# 2. Prepare the 'blurred' sharpener SIC field
# 3. Compute $\Delta_{edges}$ and finally $C_{ER}$
# 
# 
# ### Step 3.1 Regrid 'base' SIC (coarse resolution) to 'sharpener' SIC (high resolution) grid 
# A first intermediate step is to have the base SIC field (e.g. `CKA`) on the same grid as the sharpener grid `KA`. For this we use some of our collocation tools in the toolbox. 

# In[9]:


algo = 'CKA@KA'
base, sharpener = algo.split('@')


# In[10]:


reload(coll)
# extract target and source geometries
fwd_trg_lon = fwd_geo[sharpener]['lon']
fwd_trg_lat = fwd_geo[sharpener]['lat']
fwd_src_lon = fwd_geo[base]['lon']
fwd_src_lat = fwd_geo[base]['lat']
bck_trg_lon = bck_geo[sharpener]['lon']
bck_trg_lat = bck_geo[sharpener]['lat']
bck_src_lon = bck_geo[base]['lon']
bck_src_lat = bck_geo[base]['lat']

# Prepare a stack of the data to be regridded (e.g. SIC and SDEV)
what = ('sic','sdev','dal','owf')
fwd_stack_shape = tuple(list(fwd_src_lat.shape) + [len(what),])
bck_stack_shape = tuple(list(bck_src_lat.shape) + [len(what),])
fwd_src_stack = np.empty(fwd_stack_shape)
bck_src_stack = np.empty(bck_stack_shape)
for iw, w in enumerate(what):
    fwd_src_stack[...,iw] = fwd_res[base].get(w)
    bck_src_stack[...,iw] = bck_res[base].get(w)

# regrid and get _bAs (base @ sharpener grid), 
_fwd_bAs = coll._regrid_fields(fwd_trg_lon, fwd_trg_lat, fwd_src_lon, fwd_src_lat, fwd_src_stack)
_bck_bAs = coll._regrid_fields(bck_trg_lon, bck_trg_lat, bck_src_lon, bck_src_lat, bck_src_stack)

# store in an object
fwd_res[algo + '(bAs)'] = sic_algo.SICAlgoResult(_fwd_bAs[...,0], _fwd_bAs[...,1], _fwd_bAs[...,2], _fwd_bAs[...,3])
bck_res[algo + '(bAs)'] = sic_algo.SICAlgoResult(_bck_bAs[...,0], _bck_bAs[...,1], _bck_bAs[...,2], _bck_bAs[...,3])


# In[11]:


# The bAs now has the same shape as the sharpener
assert(fwd_res[sharpener].sic.shape == fwd_res[algo+'(bAs)'].sic.shape)
assert(bck_res[sharpener].sic.shape == bck_res[algo+'(bAs)'].sic.shape)


# ### Step 3.2 Prepare the 'blurred' sharpener SIC field 
# 
# Second intermediate step is to prepare a blurred version of the sharpener SIC field, keeping it in its own grid (each grid point in the blurred SIC field is computed from the surrounding pixels in the field). The aim is to blur the 'sharpener' SIC until a resolution similar to that of the 'base' SIC (but to stay in the 'sharpener' grid).
# 

# In[12]:


reload(coll)

# extract target and source geometries (the same: we stay in the sharpener's grid)
fwd_trg_lon = fwd_geo[sharpener]['lon']
fwd_trg_lat = fwd_geo[sharpener]['lat']
fwd_src_lon = fwd_geo[sharpener]['lon']
fwd_src_lat = fwd_geo[sharpener]['lat']
bck_trg_lon = bck_geo[sharpener]['lon']
bck_trg_lat = bck_geo[sharpener]['lat']
bck_src_lon = bck_geo[sharpener]['lon']
bck_src_lat = bck_geo[sharpener]['lat']
# Prepare a stack of the data to be regridded (only the SIC)
what = ('sic',)
fwd_stack_shape = tuple(list(fwd_src_lat.shape) + [len(what),])
bck_stack_shape = tuple(list(bck_src_lat.shape) + [len(what),])
fwd_src_stack = np.empty(fwd_stack_shape)
bck_src_stack = np.empty(bck_stack_shape)
for iw, w in enumerate(what):
    fwd_src_stack[...,iw] = fwd_res[sharpener].get(w)
    bck_src_stack[...,iw] = bck_res[sharpener].get(w)

# regrid and get _sbl (sharpener blurred)
params = {'method':'gauss', 'sigmas':25000, 'neighbours':55}
_fwd_sbl = coll._regrid_fields(fwd_trg_lon, fwd_trg_lat, fwd_src_lon, fwd_src_lat, fwd_src_stack, params=params)
_bck_sbl = coll._regrid_fields(bck_trg_lon, bck_trg_lat, bck_src_lon, bck_src_lat, bck_src_stack, params=params)

# store in an object
fwd_res[algo + '(blur)'] = sic_algo.SICAlgoResult(_fwd_sbl[...,0], fwd_res[sharpener].sdev, fwd_res[sharpener].dal, fwd_res[sharpener].owf)
bck_res[algo + '(blur)'] = sic_algo.SICAlgoResult(_bck_sbl[...,0], bck_res[sharpener].sdev, bck_res[sharpener].dal, bck_res[sharpener].owf)


# ### Step 3.3 Compute $\Delta_{edges}$ and finally $C_{ER}$
# 
# This is the final step of the pan-sharpening. We first compute $\Delta_{edges} = C_{HR} - C_{HR, blurred}$, then the final enhance-resolution SIC $C_{ER}$.

# In[13]:


# compute the delta edges
_fwd_delta = fwd_res[sharpener].sic - fwd_res[algo + '(blur)'].sic
_bck_delta = bck_res[sharpener].sic - bck_res[algo + '(blur)'].sic

# store in an object (intermediate result, we store it only for visualization in the notebook)
_fwd_zeros = np.zeros_like(_fwd_delta)
_bck_zeros = np.zeros_like(_bck_delta)
fwd_res[algo + '(delta)'] = sic_algo.SICAlgoResult(_fwd_delta, _fwd_zeros, _fwd_zeros, _fwd_zeros)
bck_res[algo + '(delta)'] = sic_algo.SICAlgoResult(_bck_delta, _bck_zeros, _bck_zeros, _bck_zeros)


# In[14]:


# compute final pan-sharpened SIC
_fwd_er = fwd_res[algo+'(bAs)'].sic + _fwd_delta
_bck_er = bck_res[algo+'(bAs)'].sic + _bck_delta

# store in an object
_fwd_zeros = np.zeros_like(_fwd_delta)
_bck_zeros = np.zeros_like(_bck_delta)
fwd_res[algo] = sic_algo.SICAlgoResult(_fwd_er, fwd_res[algo+'(bAs)'].sdev, _fwd_zeros, _fwd_zeros)
bck_res[algo] = sic_algo.SICAlgoResult(_bck_er, bck_res[algo+'(bAs)'].sdev, _bck_zeros, _bck_zeros)


# In[15]:


# Simple visualization in swath L1X geometry
for alg in (algo + '(bAs)', sharpener, algo + '(blur)', algo + '(delta)', algo):
    cmap = cmocean.cm.ice
    ucmap = cmocean.cm.thermal
    vmin, vmax = (0, 100)
    umin, umax = (0, 10)
    if 'delta' in alg:
        cmap = cmocean.cm.balance
        ucmap = cmap
        vmin, vmax = (-30, 30)
        umin, umax = (vmin, vmax)
    fig = plt.figure(figsize=(30,16))
    axF = fig.add_subplot(1,4,1)
    cF = axF.imshow(100*fwd_res[alg].sic, vmin=vmin, vmax=vmax, interpolation = 'none', origin='lower', cmap=cmap)
    axF.set_title(alg + " " + "FWD" + " " + "SIC")
    plt.colorbar(cF,orientation='horizontal', pad=0.05)
    #ax.set_xticks([]); ax.set_yticks([])
    ax = fig.add_subplot(1,4,2, sharex=axF, sharey=axF)
    uF = ax.imshow(fwd_res[alg].sdev, vmin=umin, vmax=umax, interpolation = 'none', origin='lower', cmap=ucmap)
    ax.set_title(alg + " " + "FWD" + " " + "SDEV")
    plt.colorbar(uF,orientation='horizontal', pad=0.05)
    #ax.set_xticks([]); ax.set_yticks([])
    axB = fig.add_subplot(1,4,3)
    cB = axB.imshow(100*bck_res[alg].sic, vmin=vmin, vmax=vmax, interpolation = 'none', origin='lower', cmap=cmap)
    axB.set_title(alg + " " + "BCK" + " " + "SIC")
    plt.colorbar(cB,orientation='horizontal', pad=0.05)
    #ax.set_xticks([]); ax.set_yticks([])
    ax = fig.add_subplot(1,4,4, sharex=axB, sharey=axB)
    uB=ax.imshow(bck_res[alg].sdev, vmin=umin, vmax=umax, interpolation = 'none', origin='lower', cmap=ucmap)
    ax.set_title(alg + " " + "BCK" + " " + "SDEV")
    plt.colorbar(uB,orientation='horizontal', pad=0.05)
    #ax.set_xticks([]); ax.set_yticks([])
    
    axF.set_xlim(400,600)
    axF.set_ylim(25,200)
    axB.set_xlim(350,550)
    axB.set_ylim(325,500)
    
    plt.show()


# ## Step 4 : Combine forward and backward scans
# 
# For the time being, we resample the SICs from the forward and the backward scans (separately) to an EASE2 grid, then combine them on the grid. We define a status_flag to record if a particular grid cell is from forward+backward scans, only forward, or only backward.
# 
# Note: in this implementation we only regrid to the Northern Hemisphere grid because we know that the simulated L1B orbit does not extend over the Southern Hemisphere. In general, orbits can cover both hemisphere and we will have to grid each orbit both to an NH and and SH grid.
# 
# ### Step 4.1 : Load the grid definition from parameter file

# In[16]:


import pyresample as pr
from pyresample import parse_area_file

algo = 'CKA@KA'
base, sharpener = algo.split('@')
#algo = 'KA'
#base, sharpener = 'KA', 'KA'

# this is the mean spacing of the SIC field, which is about 4000m for @KA
src_spacing = 4000

if l2_grid == 'ease2':
    
    grid_def_file = os.path.join('grids_py.def')

    grid_n = 'nh-ease2-025'
    adef  = parse_area_file(grid_def_file, grid_n)[0]
    trg_lon, trg_lat = adef.get_lonlats()
    trg_spacing = 2500

else:
    tc_fn = os.path.join(tc_path, tc_fn)
    ds = xr.open_dataset(tc_fn)
    
    adef_tc, _ = pr.utils.load_cf_area(ds, variable='L_band_H', y='y', x='x')
    trg_spacing = 1000
    trg_lon, trg_lat = adef_tc.get_lonlats()
    
    file_lon = ds['Longitude'].data
    file_lat = ds['Latitude'].data
    trg_spacing = 1000
    print('max difference in lon:', abs(file_lon-trg_lon).max())
    print('min difference in lat:', abs(file_lat-trg_lat).max())
    
    print("pyresample AreaDefinition for the TestCard:")
    print(adef_tc)
    adef = adef_tc


# ### Step 4.2 : Grid the forward and backward fields separately

# In[17]:


fwd_grd = dict()
bck_grd = dict()

# extract source geometries (those of the 'sharpener' field)
fwd_src_lon = fwd_geo[sharpener]['lon']
fwd_src_lat = fwd_geo[sharpener]['lat']
bck_src_lon = bck_geo[sharpener]['lon']
bck_src_lat = bck_geo[sharpener]['lat']


# Prepare a stack of the data to be regridded (SIC and sdev)
what = ('sic','sdev')
fwd_stack_shape = tuple(list(fwd_src_lat.shape) + [len(what),])
bck_stack_shape = tuple(list(bck_src_lat.shape) + [len(what),])
fwd_src_stack = np.empty(fwd_stack_shape)
bck_src_stack = np.empty(bck_stack_shape)
for iw, w in enumerate(what):
    fwd_src_stack[...,iw] = fwd_res[algo].get(w)
    bck_src_stack[...,iw] = bck_res[algo].get(w)

# regrid to the EASE2 grid
params = {'method':'gauss', 'sigmas':max(trg_spacing,src_spacing/2.), 'neighbours':8}
_fwd_grid = coll._regrid_fields(trg_lon, trg_lat, fwd_src_lon, fwd_src_lat, fwd_src_stack, params=params)
_bck_grid = coll._regrid_fields(trg_lon, trg_lat, bck_src_lon, bck_src_lat, bck_src_stack, params=params)

# store in an object
_grd_zeros = np.zeros_like(trg_lon)
fwd_grd[algo] = sic_algo.SICAlgoResult(np.ma.masked_invalid(_fwd_grid[:,:,0]),
                                       np.ma.masked_invalid(_fwd_grid[:,:,1]), _grd_zeros, _grd_zeros)
bck_grd[algo] = sic_algo.SICAlgoResult(np.ma.masked_invalid(_bck_grid[:,:,0]),
                                       np.ma.masked_invalid(_bck_grid[:,:,1]), _grd_zeros, _grd_zeros)


# In[18]:


# visualization of the gridded fields
cmap = cmocean.cm.ice
ucmap = cmocean.cm.thermal
vmin, vmax = (0, 100)
umin, umax = (0, 6)

cart_crs = adef.to_cartopy_crs()
#cart_crs=ccrs.LambertAzimuthalEqualArea(central_longitude=0, central_latitude=90)
fig, ax = plt.subplots(nrows=2, ncols=2, sharex=True, sharey=True, figsize=(10,10),
                       subplot_kw=dict(projection=cart_crs))
# first row : SICs
c = ax[0,0].imshow(100*fwd_grd[algo].sic, transform=cart_crs, extent=cart_crs.bounds, origin='upper',
              cmap=cmap,vmin=vmin,vmax=vmax)
ax[0,0].coastlines(color='red')
ax[0,0].set_title('FORWARD', fontsize='small')
ax[0,0].text(0.01,0.99,'SIC',va='top',transform=ax[0,0].transAxes)
plt.colorbar(c,orientation='horizontal', pad=0.05, shrink=0.75)

c = ax[0,1].imshow(100*bck_grd[algo].sic, transform=cart_crs, extent=cart_crs.bounds, origin='upper',
              cmap=cmap,vmin=vmin,vmax=vmax)
ax[0,1].coastlines(color='red')
ax[0,1].set_title('BACKWARD', fontsize='small')
ax[0,1].text(0.01,0.99,'SIC',va='top',transform=ax[0,1].transAxes)
plt.colorbar(c,orientation='horizontal', pad=0.05, shrink=0.75)

# second row : SDEVs
c = ax[1,0].imshow(fwd_grd[algo].sdev, transform=cart_crs, extent=cart_crs.bounds, origin='upper',
              cmap=ucmap,vmin=umin,vmax=umax)
ax[1,0].coastlines(color='red')
ax[1,0].text(0.01,0.99,'SDEV',va='top',transform=ax[1,0].transAxes)
plt.colorbar(c,orientation='horizontal', pad=0.05, shrink=0.75)
c=ax[1,1].imshow(bck_grd[algo].sdev, transform=cart_crs, extent=cart_crs.bounds, origin='upper',
              cmap=ucmap,vmin=umin,vmax=umax)
ax[1,1].coastlines(color='red')
ax[1,1].text(0.01,0.99,'SDEV',va='top',transform=ax[1,1].transAxes)
plt.colorbar(c,orientation='horizontal', pad=0.05, shrink=0.75)

plt.show()


# ### Step 4.3 : Combine the forward and backward fields

# In[19]:


# we call the result, the "merged" sic:
mrg_grd = dict()

# have a flag to indicate where the forward and backward scans contributed
mrg_grd_flag = np.zeros(fwd_grd[algo].sic.shape, dtype='i8')
_fwd_valid = ~(fwd_grd[algo].sic.mask)
_bck_valid = ~(bck_grd[algo].sic.mask)
mrg_grd_flag[_fwd_valid *  _bck_valid] = 3
mrg_grd_flag[_bck_valid * ~_fwd_valid] = 2
mrg_grd_flag[_fwd_valid * ~_bck_valid] = 1

# we do a naive merge (arithmetic mean) for now because the uncertainties
#   in forward and backward fields will be quite similar.
_mrg_grd_sic = 0.5 * (fwd_grd[algo].sic + bck_grd[algo].sic)
_mrg_grd_sic[_fwd_valid * ~_bck_valid] = fwd_grd[algo].sic[_fwd_valid * ~_bck_valid]
_mrg_grd_sic[_bck_valid * ~_fwd_valid] = bck_grd[algo].sic[_bck_valid * ~_fwd_valid]

# the uncertainties in the forward and backward SIC fields are highly correlated,
#   thus at the first order the uncertainties are the mean uncertainty (in variance).
#   This should be revisited e.g. to reduce the uncertainty due to NeDT (which is small
#   wrt that due to tie-points).
_mrg_grd_var = 0.5 * (fwd_grd[algo].sdev**2 + bck_grd[algo].sdev**2)
_mrg_grd_var[_fwd_valid * ~_bck_valid] = fwd_grd[algo].sdev[_fwd_valid * ~_bck_valid]**2
_mrg_grd_var[_bck_valid * ~_fwd_valid] = bck_grd[algo].sdev[_bck_valid * ~_fwd_valid]**2
_mrg_grd_sdev = _mrg_grd_var**0.5

# store in an object
mrg_grd[algo] = sic_algo.SICAlgoResult(_mrg_grd_sic, _mrg_grd_sdev, _grd_zeros, _grd_zeros)


# In[20]:


# visualize / plot
fig, ax = plt.subplots(nrows=1, ncols=3, sharex=True, sharey=True, figsize=(11,6), subplot_kw=dict(projection=cart_crs))
# first row : SICs
c = ax[0].imshow(100*mrg_grd[algo].sic, transform=cart_crs, extent=cart_crs.bounds, origin='upper',
              cmap=cmap,vmin=vmin,vmax=vmax)
#c = ax[0].imshow(100*mrg_grd[algo].sic, transform=cart_crs, extent=cart_crs.bounds, origin='upper',
#              cmap=cmocean.cm.balance,vmin=-10,vmax=10)
ax[0].coastlines(color='red')
ax[0].text(0.01,0.99,'SIC',va='top',fontsize=14,transform=ax[0].transAxes)
plt.colorbar(c,orientation='horizontal', pad=0.05, shrink=0.8)

# second row : SDEVs
c = ax[1].imshow(mrg_grd[algo].sdev, transform=cart_crs, extent=cart_crs.bounds, origin='upper',
              cmap=ucmap,vmin=umin,vmax=umax)
ax[1].coastlines(color='red')
ax[1].text(0.01,0.99,'SDEV',va='top',fontsize=14,transform=ax[1].transAxes)
plt.colorbar(c,orientation='horizontal', pad=0.05, shrink=0.8)

# second row : FLAG
c = ax[2].imshow(mrg_grd_flag, transform=cart_crs, extent=cart_crs.bounds, origin='upper',vmin=0,vmax=3)
ax[2].coastlines(color='red')
ax[2].text(0.01,0.99,'FLAG',va='top',fontsize=14,transform=ax[2].transAxes,color='w')
plt.colorbar(c,orientation='horizontal', pad=0.05, shrink=0.8,)

plt.show()


# ## Step 5: Write Level-2 product file
# 
# We write a netCDF/CF files containing the merged SIC on the EASE2 2.5km grid. This requires a pre-processing of the SIC fields to apply the Open Water Filter, and separate the off-range SICs into specific variables.
# 
# Writing the netCDF/CF file is done via xarray.

# ### Step 5.1: Pre-processing of the SIC fields
# 
# TODO

# In[21]:


# change SIC from [0-1] to [0-100%] range for writing to file (the SDEV is already in %)
mrg_grd[algo].sic *= 100


# ### Step 5.2: Format L2 SIC file and write to disk

# In[22]:


reload(l2)
# get a template L2 format (netCDF/CF) from the Tools module
ds_l2 = l2.to_cf_template(adef, skip_lonlat=False)

# create a DataArray for SIC from the template
da_sic = xr.DataArray(mrg_grd[algo].sic, coords=ds_l2['template'].coords, dims=ds_l2['template'].dims,
                       attrs=ds_l2['template'].attrs, name='ice_conc')
da_sic.attrs['long_name'] = 'Sea Ice Concentration from the {} algorithm'.format(algo)
da_sic.attrs['standard_name'] = 'sea_ice_area_fraction'
da_sic.attrs['units'] = 1
da_sic.attrs['auxiliary_variables'] = 'ice_conc_total_uncertainty,status_flag'

# create a DataArray for SDEV from the template
da_sdev = xr.DataArray(mrg_grd[algo].sdev, coords=ds_l2['template'].coords, dims=ds_l2['template'].dims,
                       attrs=ds_l2['template'].attrs, name='ice_conc_total_uncertainty')
da_sdev.attrs['long_name'] = 'Total Uncertainty for the Sea Ice Concentraion from the {} algorithm'.format(algo)
da_sdev.attrs['standard_name'] = 'sea_ice_area_fraction standard_error'
da_sdev.attrs['units'] = 1

# create a DataArray for FLAG from the template
da_flg = xr.DataArray(mrg_grd_flag, coords=ds_l2['template'].coords, dims=ds_l2['template'].dims,
                       attrs=ds_l2['template'].attrs, name='status_flag')
da_flg.attrs['long_name'] = 'Processing and status flags for the Sea Ice Concentraion from the {} algorithm'.format(algo)

# add the data arrays to the ds_l2 object
ds_l2 = ds_l2.merge(da_sic)
ds_l2 = ds_l2.merge(da_sdev)
ds_l2 = ds_l2.merge(da_flg)

# customize the global attributes
ds_l2.attrs['title'] = 'Example CIMR L2 NRT3H Sea Ice Concentration for the DEVALGO Radiometric Test Card'
ds_l2.attrs['l1b_file'] = os.path.basename(l1b_fn)

# remove the 'template' variable (we don't need it anymore)
ds_l2 = ds_l2.drop('template')

# write to file
l2_n = './CIMR_L2_SIC_{}_{}.nc'.format(l2_grid.upper(), input_test_card.upper())
ds_l2.to_netcdf(l2_n, format='NETCDF4_CLASSIC')
print(l2_n)


# In[23]:


# We can stop the notebook here if the target grid was not that of the TestCard
if l2_grid != 'tc':
    raise Exception


# # Performance Assessment
# 
# ## Compare to the "ground-truth" SICs
# 
# Access the ground truth SIC from the Test Card and compare pixel-by-pixel to the SIC retrieved by the algorithm.

# In[24]:


owci_flg_cice = 2
owci_flg_ow   = 1

tc_tbs = dict()
if input_test_card == 'sceps_polar_1':
    # load SIC truth from the GEO file
    geo_file = '/home/thomasl/Documents/DEVALGO/From_SCEPS_Aug2023/Ref_scen_GEO_TDS/SCEPS/Cards/SCEPS_Geo_Polar_Scene_1/cimr_sceps_geo_card_polarscene_1_20161217_v1p0_part1.nc'
    tc_ds = xr.open_dataset(geo_file)
    tc_sic = tc_ds['asi_sea_ice_concentration_nh'][0,:,:].to_masked_array().transpose()
    
    # load TOA Tbs from the TOA file
    toa_file = '/home/thomasl/Documents/DEVALGO/From_SCEPS_Aug2023/Ref_scen_TOA_TDS/SCEPS/SCEPS_TOA_Geo_Polar_Scene_1/cimr_sceps_toa_card_polarscene_1_20161217_v1p0_aa_000.nc'
    tc_l1b = xr.open_dataset(toa_file)
    toa_band_name = 'toa_tbs_{b:}_{p:}po'
    
    tc_tbs['tb06v'] = np.rot90(tc_l1b[toa_band_name.format(b='C', p='V')].isel(time=0).sel(incidence_angle=55),-1)
    tc_tbs['tb19v'] = np.rot90(tc_l1b[toa_band_name.format(b='Ka', p='V')].isel(time=0).sel(incidence_angle=55),-1)
    tc_tbs['tb37v'] = np.rot90(tc_l1b[toa_band_name.format(b='Ku', p='V')].isel(time=0).sel(incidence_angle=55),-1)
    tc_tbs['tb37h'] = np.rot90(tc_l1b[toa_band_name.format(b='Ku', p='H')].isel(time=0).sel(incidence_angle=55),-1)
    tc_owci = np.zeros_like(tc_tbs['tb06v']).astype('int')
    tc_owci[-500:-20,20:500] = owci_flg_cice
    tc_owci[20:500,-500:-20] = owci_flg_ow
    
    # how many pixels to mask around the test card
    margin = 3
    
else:
    print(tc_fn)
    tc_fn = os.path.join(tc_path, tc_fn)
    tc_l1b = xr.open_dataset(tc_fn,)
    tc_surf = tc_l1b['surfaces'].data
    
    # how many pixels to mask around the test card
    margin = 15
    
    # use the surfaces variable to find where is ice and water
    tc_ice_mask = (tc_surf == 1)+(tc_surf == 2)
    tc_ocean_mask = (tc_surf == 5)+(tc_surf == 6)+(tc_surf == 7)+(tc_surf == 8)
    tc_oceanice_mask = tc_ice_mask + tc_ocean_mask
        
    tc_sic = np.ma.array(np.zeros_like(tc_surf))
    tc_sic[tc_ice_mask] = 100.
    tc_sic[~tc_oceanice_mask] = np.ma.masked
    
    # load TOA Tbs
    toa_band_name = '{b:}_band_{p:}'
    
    tc_tbs['tb06v'] = tc_l1b[toa_band_name.format(b='C', p='V')].to_masked_array()
    tc_tbs['tb19v'] = tc_l1b[toa_band_name.format(b='Ku', p='V')].to_masked_array()
    tc_tbs['tb37v'] = tc_l1b[toa_band_name.format(b='Ka', p='V')].to_masked_array()
    tc_tbs['tb37h'] = tc_l1b[toa_band_name.format(b='Ka', p='V')].to_masked_array()
    print(tc_tbs['tb06v'].shape, tc_tbs['tb06v'].min(), tc_tbs['tb06v'].max())
    
    tc_owci = np.zeros_like(tc_tbs['tb06v']).astype('int')
    tc_owci[tc_sic==1] = owci_flg_cice
    tc_owci[tc_sic==0] = owci_flg_ow
    
    if input_test_card == 'radiometric':    
        # Remove the top-most and right-most areas of the Radiometric TestCard
        #   Because we focus on the cells for the time being.
        _margin = 200
        tc_geom = np.ones_like(tc_surf).astype('bool')
        tc_geom[:,:_margin] = False
        tc_geom[-_margin:,:] = False
        tc_sic[~tc_geom] = np.ma.masked
        for ch in tc_tbs.keys():
            tc_tbs[ch][~tc_geom] = np.ma.masked
    

# Remove the border of the TestCard (because of the spill-over from the 0K from outside the scene):
tc_border = np.ones_like(tc_sic).astype('bool')
tc_border[-margin:,:] = False
tc_border[:margin,:] = False
tc_border[:,-margin:] = False
tc_border[:,:margin] = False
tc_sic[~tc_border] = np.ma.masked
for ch in tc_tbs.keys():
    tc_tbs[ch][~tc_border] = np.ma.masked


# In[25]:


cmap_sic = cmocean.cm.ice
cmap_sic.set_bad('grey')
vmin = 0
vmax = 100

cmap_dif = cmocean.cm.balance
cmap_dif.set_bad('grey')
dmin = -25
dmax = -dmin

sic_diff = mrg_grd[algo].sic - tc_sic

# visualize / plot
fig, ax = plt.subplots(nrows=1, ncols=3, sharex=True, sharey=True, figsize=(12,6), subplot_kw=dict(projection=cart_crs))
# first col : SIC truth
c = ax[0].imshow(tc_sic, transform=cart_crs, extent=cart_crs.bounds, origin='upper',
              cmap=cmap_sic,vmin=vmin,vmax=vmax)
ax[0].coastlines(color='red')
ax[0].text(0.01,1.01,'SIC (Truth)',va='bottom',fontsize=14,transform=ax[0].transAxes)
plt.colorbar(c,orientation='horizontal', pad=0.05, shrink=0.8)
# second col : SIC algorithm
c = ax[1].imshow(mrg_grd[algo].sic, transform=cart_crs, extent=cart_crs.bounds, origin='upper',
              cmap=cmap_sic,vmin=vmin,vmax=vmax)
ax[1].coastlines(color='red')
ax[1].text(0.01,1.01,'SIC (DEVALGO)',va='bottom',fontsize=14,transform=ax[1].transAxes)
plt.colorbar(c,orientation='horizontal', pad=0.05, shrink=0.8)
# third col : SIC diff
c = ax[2].imshow(sic_diff, transform=cart_crs, extent=cart_crs.bounds, origin='upper',
              cmap=cmap_dif,vmin=dmin,vmax=dmax)
ax[2].coastlines(color='red')
ax[2].text(0.01,1.01,'SIC (DEVALGO - Truth)',va='bottom',fontsize=14,transform=ax[2].transAxes)
plt.colorbar(c,orientation='horizontal', pad=0.05, shrink=0.8)
plt.show()


# In[26]:


sic_diff_1d = sic_diff.compressed()
fig, ax = plt.subplots(figsize=(6,6))
hist_range = (dmin, dmax)
ax.hist(sic_diff_1d, bins=50, range=hist_range, density=True)
ax.set_title("SIC (DEVALGO - Truth)\n{}".format(input_test_card), fontsize='small')
ax.text(0.99,0.98, 'BIAS: {:.2f} [%]'.format(sic_diff_1d.mean()),
        transform=ax.transAxes, ha='right', va='top', fontsize='small')
ax.text(0.99,0.93, 'RMSE: {:.2f} [%]'.format(sic_diff_1d.std()),
        transform=ax.transAxes, ha='right', va='top', fontsize='small')
plt.show()


# The differences in True SIC can stem from the difference in calibration between the forward model used to prepare the TestCards and the SIC0 and SIC1 datasets used to tune the SIC algorithms. We explore this below.

# In[27]:


def plot_2d_tiepoints(cice,ow,chx,chy,ax=None,leg=False,title=True,samples=('ow','cice')):
    
    if ax is None:
        fig = plt.figure(figsize=(10.,10.))
        ax = fig.add_subplot(111,aspect=1)
    
    try:
        ow_x = getattr(ow,chx)
        ow_y = getattr(ow,chy)
        ci_x = getattr(cice,chx)
        ci_y = getattr(cice,chy)
    except AttributeError as ae:
        raise ValueError('{} is not in {}'.format(ae,ow.channels))
    
    ichx = ow.channels.index(chx)
    ichy = ow.channels.index(chy)
    
    # plot the scatter
    if 'ow' in samples:
        ax.scatter(ow_x, ow_y, marker='d', color='dodgerblue', s=10, label='{} OW'.format(ow.source.upper()))
    if 'cice' in samples:
        ax.scatter(ci_x, ci_y, marker='s', color='gray', s=10, label='{} CICE'.format(cice.source.upper()))
    
    # place the tie-point signatures
    s = 200
    
    # draw ice line
    ax.plot([cice.myi_tp[ichx],cice.fyi_tp[ichx]],[cice.myi_tp[ichy],cice.fyi_tp[ichy]],'r-',lw=2)
    
    # label
    ax.set_xlabel(chx + ' [K]')
    ax.set_ylabel(chy + ' [K]')
    
    if title:
        ax.set_title('{} {}'.format(ow.source.upper(), ow.area))
    
    if leg:
        ax.legend()
    
    return ax


fig, ax = plt.subplots(nrows=3, ncols=2, figsize=(9,12))
c=ax[0,0].imshow(tc_tbs['tb06v'], interpolation='none', origin='lower'); ax[0,0].axis('off'); ax[0,0].set_title('06v');plt.colorbar(c,shrink=0.5)
c=ax[0,1].imshow(tc_tbs['tb19v'], interpolation='none', origin='lower'); ax[0,1].axis('off'); ax[0,1].set_title('19v');plt.colorbar(c,shrink=0.5)
c=ax[1,0].imshow(tc_tbs['tb37v'], interpolation='none', origin='lower'); ax[1,0].axis('off'); ax[1,0].set_title('37v');plt.colorbar(c,shrink=0.5)
c=ax[1,1].imshow(tc_tbs['tb37h'], interpolation='none', origin='lower'); ax[1,1].axis('off'); ax[1,1].set_title('37h');plt.colorbar(c,shrink=0.5)
c=ax[2,0].imshow(tc_sic, interpolation='none', origin='lower'); ax[2,0].axis('off'); ax[2,0].set_title('SIC');plt.colorbar(c,shrink=0.5)
c=ax[2,1].imshow(tc_owci, interpolation='none', origin='lower'); ax[2,1].axis('off'); ax[2,1].set_title('OW/CI');plt.colorbar(c,shrink=0.5)
plt.show()


# In[28]:


fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(16,8))
ax[0] = plot_2d_tiepoints(ci_tp, ow_tp, 'tb37v', 'tb06v', leg=True, ax=ax[0])
ax[0].scatter(tc_tbs['tb37v'][tc_owci==2], tc_tbs['tb06v'][tc_owci==2],
              s=10, marker='d', color='coral', label='TestCard CICE')
ax[0].scatter(tc_tbs['tb37v'][tc_owci==1], tc_tbs['tb06v'][tc_owci==1],
              s=10, marker='s', color='darkblue', label='TestCard OW')
ax[0].legend()
ax[1] = plot_2d_tiepoints(ci_tp, ow_tp, 'tb37v', 'tb19v', leg=False, ax=ax[1])
ax[1].scatter(tc_tbs['tb37v'][tc_owci==2], tc_tbs['tb19v'][tc_owci==2],
              s=10, marker='d', color='coral', )
ax[1].scatter(tc_tbs['tb37v'][tc_owci==1], tc_tbs['tb19v'][tc_owci==1],
              s=10, marker='s', color='darkblue',)
plt.show()


# ## Compare two algo runs, one from L1B and one from TestCard TBs
# 
# In order to assess the SICs obtained from the simulated L1B files, we want to compare them to the SICs that are obtained directly from applying the algorithm onto the TBs from the TestCard. This is much simpler because the data structure is simpler (no forward and backward scans) and we do not need the pan-sharpening step since all TBs are with the same 1x1 km resolution.

# In[29]:


# tc = TestCard
tc_fn = os.path.join(tc_path, tc_fn)
tc_l1b = xr.open_dataset(tc_fn,)

# prepare TBs in the structure expected as input to the algorithm.
tc_tbs = dict()
for alg in algos.keys():
    tc_tbs[alg] = dict()
    for ch in algos[alg]['channels']:
        varn = tb_dict[ch[:-1]].capitalize() + '_band_' + ch[-1:].upper()
        tc_tbs[alg][ch] = tc_l1b[varn].values
    
# extract the lat/lon arrays for later use in the pan-sharpening
tc_geo = dict()
geo_list = list(algos.keys()) + ['KA',]
for geo in geo_list:
    tc_geo[geo] = dict()
    tc_geo[geo]['lat'] = tc_l1b['Latitude'].to_numpy()
    tc_geo[geo]['lon'] = tc_l1b['Longitude'].to_numpy()


# In[30]:


tc_res = dict()
for alg in algos.keys():
    
    # run the algorithm to compute SIC
    tc_res[alg] = algos[alg]['algo'].compute_sic(tc_tbs[alg])
    
    # Simple visualization 
    cmap = cmocean.cm.ice
    ucmap = cmocean.cm.thermal
    vmin, vmax = (0., 1)
    umin, umax = (0, 10)
    
    fig, (ax0,ax1) = plt.subplots(ncols=2,figsize=(6,5))
    cF = ax0.imshow(tc_res[alg].sic, vmin=vmin, vmax=vmax, interpolation = 'none', origin='upper', cmap=cmap)
    ax0.set_title(alg + " " + "TC" + " " + "SIC", fontsize='small')
    plt.colorbar(cF,orientation='horizontal', pad=0.05)
    ax0.set_xticks([]); ax0.set_yticks([])
    uF = ax1.imshow(tc_res[alg].sdev, vmin=umin, vmax=umax, interpolation = 'none', origin='upper', cmap=ucmap)
    ax1.set_title(alg + " " + "TC" + " " + "SDEV", fontsize='small')
    plt.colorbar(uF,orientation='horizontal', pad=0.05)
    ax1.set_xticks([]); ax1.set_yticks([])
    
    # post-processing to align with the product from L1B
    tc_res[alg].sic *= 100.


# Also plot the CKA@KA results from the simulated L1B file (remapped onto the TC grid)
alg = 'CKA@KA'
fig, (ax0,ax1) = plt.subplots(ncols=2,figsize=(6,5))
cF = ax0.imshow(mrg_grd[alg].sic/100., vmin=vmin, vmax=vmax, interpolation = 'none', origin='upper', cmap=cmap)
ax0.set_title(alg + " " + "L1B" + " " + "SIC", fontsize='small')
plt.colorbar(cF,orientation='horizontal', pad=0.05)
ax0.set_xticks([]); ax0.set_yticks([])
uF = ax1.imshow(mrg_grd[alg].sdev, vmin=umin, vmax=umax, interpolation = 'none', origin='upper', cmap=ucmap)
ax1.set_title(alg + " " + "L1B" + " " + "SDEV", fontsize='small')
plt.colorbar(uF,orientation='horizontal', pad=0.05)
ax1.set_xticks([]); ax1.set_yticks([])
plt.show()


# ## Compare the SICs from the simulated TBs and from the TestCard TBs

# In[31]:


tc_surf = tc_l1b['surfaces'].data

if input_test_card == 'radiometric':
    # Remove pixels from the TestCard that are neither water nor sea-ice
    tc_ice_mask = (tc_surf == 1)+(tc_surf == 2)
    tc_ocean_mask = (tc_surf == 5)+(tc_surf == 6)+(tc_surf == 7)+(tc_surf == 8)
    tc_oceanice_mask = tc_ice_mask + tc_ocean_mask
else:
    tc_ice_mask = np.ones_like(tc_surf).astype('bool')
    tc_ocean_mask = np.ones_like(tc_surf).astype('bool')
    tc_oceanice_mask = np.ones_like(tc_surf).astype('bool')

# Remove the border of the Test Scene (because of the spill-over from the 0K from outside the scene):
margin = 15
tc_border = np.ones_like(tc_surf).astype('bool')
tc_border[-margin:,:] = False
tc_border[:margin,:] = False
tc_border[:,-margin:] = False
tc_border[:,:margin] = False
tc_ice_mask *= tc_border
tc_ocean_mask *= tc_border
tc_oceanice_mask *= tc_border

# Remove the top-most and right-most areas of the Radiometric TestCard
#   Because we focus on the cells for the time being.
if input_test_card == 'radiometric':
    margin = 200
    tc_geom = np.ones_like(tc_surf).astype('bool')
    tc_geom[:,:margin] = False
    tc_geom[-margin:,:] = False

    tc_ice_mask *= tc_geom
    tc_ocean_mask *= tc_geom
    tc_oceanice_mask *= tc_geom
    
fig, ax = plt.subplots(figsize=(8,8))
ax.imshow(tc_oceanice_mask, interpolation='none', vmin=0, vmax=1)
ax.set_xticks([]); ax.set_yticks([])
ax.set_title("Mask for ocean and sea ice surfaces")
plt.show()


# ### Explore the differences in SIC
# 
# #### Maps of differences

# In[32]:


cmap_dif = cmocean.cm.balance
cmap_dif.set_bad('grey')


# Plot difference map (Ocean + Sea Ice).
sic_diff = mrg_grd['CKA@KA'].sic - tc_res['CKA'].sic
sic_diff[~tc_oceanice_mask] = np.nan

fig, ax = plt.subplots(figsize=(8,8))
c = ax.imshow(sic_diff,
          interpolation='none', cmap=cmap_dif, vmin=-100, vmax=+100, origin='lower')
plt.colorbar(c,orientation='horizontal', pad=0.05, shrink=0.7)
ax.set_xticks([]); ax.set_yticks([])
ax.set_title("Difference SIC (L1B) - SIC (TestCard)\nOcean and Sea Ice cells", fontsize='small')
plt.show()

# Plot difference map (Ocean only)
sic_diff = mrg_grd['CKA@KA'].sic - tc_res['CKA'].sic
sic_diff[~tc_ocean_mask] = np.nan

fig, ax = plt.subplots(figsize=(8,8))
c = ax.imshow(sic_diff,
          interpolation='none', cmap=cmap_dif, vmin=-100, vmax=+100, origin='lower')
plt.colorbar(c,orientation='horizontal', pad=0.05, shrink=0.7)
ax.set_xticks([]); ax.set_yticks([])
ax.set_title("Difference SIC (L1B) - SIC (TestCard)\nOcean cells", fontsize='small')
plt.show()

# Plot difference map (SeaIce only)
sic_diff = mrg_grd['CKA@KA'].sic - tc_res['CKA'].sic
sic_diff[~tc_ice_mask] = np.nan

fig, ax = plt.subplots(figsize=(8,8))
c = ax.imshow(sic_diff,
          interpolation='none', cmap=cmap_dif, vmin=-100, vmax=+100, origin='lower')
plt.colorbar(c,orientation='horizontal', pad=0.05, shrink=0.7)
ax.set_xticks([]); ax.set_yticks([])
ax.set_title("Difference SIC (L1B) - SIC (TestCard)\nSea Ice cells", fontsize='small')
plt.show()


# #### Histogram of differences

# In[33]:


# Plot histogram (Ocean + Sea Ice).
sic_diff = mrg_grd['CKA@KA'].sic - tc_res['CKA'].sic
sic_diff[~tc_oceanice_mask] = np.nan
sic_diff = sic_diff[~np.isnan(sic_diff)]

hist_range = (-50,+50)
fig, ax = plt.subplots(figsize=(8,8))
ax.hist(sic_diff, bins=50, range=hist_range, density=True)
ax.set_title("Difference SIC (L1B) - SIC (TestCard)\nOcean and Sea Ice cells", fontsize='small')
ax.text(0.01,0.98, 'BIAS: {:.2f} [%]'.format(sic_diff.mean()),
        transform=ax.transAxes, ha='left', va='top', fontsize='small')
ax.text(0.01,0.93, 'RMSE: {:.2f} [%]'.format(sic_diff.std()),
        transform=ax.transAxes, ha='left', va='top', fontsize='small')
plt.show()

# Plot histogram (Sea Ice).
sic_diff = mrg_grd['CKA@KA'].sic - tc_res['CKA'].sic
sic_diff[~tc_ice_mask] = np.nan
sic_diff = sic_diff[~np.isnan(sic_diff)]

fig, ax = plt.subplots(figsize=(8,8))
ax.hist(sic_diff, bins=50, range=hist_range, density=True)
ax.set_title("Difference SIC (L1B) - SIC (TestCard)\nSea Ice only", fontsize='small', )

plt.show()

# Plot histogram (Ocean).
sic_diff = mrg_grd['CKA@KA'].sic - tc_res['CKA'].sic
sic_diff[~tc_ocean_mask] = np.nan
sic_diff = sic_diff[~np.isnan(sic_diff)]

fig, ax = plt.subplots(figsize=(8,8))
ax.hist(sic_diff, bins=50, range=hist_range, density=True)
ax.set_title("Difference SIC (L1B) - SIC (TestCard)\nOcean only", fontsize='small')
plt.show()


# In[ ]:




