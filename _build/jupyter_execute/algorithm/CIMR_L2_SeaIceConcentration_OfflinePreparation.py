#!/usr/bin/env python
# coding: utf-8

# # Offline preparations for L2 Sea Ice Concentration (SIC) and Sea Ice Edge (SIED) algorithms for CIMR
# 
# This notebook implements a series of "offline" preparation steps for the Level-2 SIC3H and SIED3H algorithm for the CIMR mission.
# 
# We refer to the corresponding [ATBD](https://cimr-algos.github.io/SeaIceConcentration_ATBD/intro.html) and especially the [Baseline Algorithm Definition](https://cimr-algos.github.io/SeaIceConcentration_ATBD/baseline_algorithm_definition.html#baseline-algorithm-definition).
# 
# Generally, *offline preparations* are steps that are run on a regular basis (e.g. every hours, every day, etc...). They aim at preparing everything needed for the L2 algorithms to run efficiently at once a new L1B file is available. For example, external Auxiliary Data Files (ADFs) can be fetched and pre-processed on an hourly basis by an *offline* chain, to minimize the waiting time when a new L1B product arrives.
# 
# Specifically, this notebook implements the dynamic tuning of the SIC algorithm against a rolling archive of L1B files, as is the case in the EUMETSAT OSI SAF and ESA CCI processing chains. In an operational, the rolling archive gives access to typically 5-10 days of L1B data (full L1B files or subsets of TBs). In this demonstration prototype, we have to use the few simulated L1B files we have.

# In[1]:


import sys, os
from glob import glob
from importlib import reload


# modules in the L2 Sea Ice Concentration ATBD (v2) contain software code that implement the SIC algorithm
from sirrdp import rrdp_file
from pmr_sic import tiepoints as tp
from pmr_sic import hybrid_algo

# modules to read CIMR L1B files (for the dynamic tuning against an archive of past L1B files)
if '/Tools/' not in sys.path:
    sys.path.insert(0, os.path.abspath('../..') + '/Tools/')
from tools import io_handler as io
from tools import collocation as coll

import numpy as np
import json

# encoder class for writing JSON files
class MyEncoder(json.JSONEncoder):
    # https://stackoverflow.com/questions/27050108/convert-numpy-type-to-python/27050186#27050186
    def default(self, obj):
        if isinstance(obj, int):
            return int(obj)
        elif isinstance(obj, float):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(MyEncoder, self).default(obj)


# In[2]:


# This cell has a tag 'parameters' and is used for the CLI with papermill
l1b_archive_dir = '../data/input'
out_dir = '../data/aux'
use_oza_adjust = True


# In[3]:


algos = dict()
algos['KA'] = {'channels':('tb37v', 'tb37h'),  'target_band':'KA'}
algos['CKA'] = {'channels':('tb06v', 'tb37v', 'tb37h'), 'target_band':'C'}
algos['KKA'] = {'channels':('tb19v', 'tb37v', 'tb37h'), 'target_band':'KU'}


tb_dict = {'tb01':'L','tb06':'C','tb10':'X','tb19':'KU','tb37':'KA',}
rev_tb_dict = {v:k for k,v in tb_dict.items()}
bands_needed = []
for alg in algos.keys():
    bands_needed += algos[alg]['channels']
bands_needed = list(set([tb_dict[b[:-1]] for b in bands_needed]))


# In[4]:


# check the input parameters
if not os.path.isdir(out_dir):
    raise ValueError("The output directory {} does not exist.".format(out_dir))
if not os.path.isdir(l1b_archive_dir):
    raise ValueError("The l1b archive directory {} does not exist.".format(l1b_archive_dir))


# **An hybrid dynamic / fixed tuning for the SCEPS Polar Scene 1** The SCEPS Polar Scene 1 (v2.1, early 2024) does not reach to high-enough SICs, its peak is around 92%-94% and there are very few values above 95%. This means we cannot extract TB samples for near-100% SIC conditions with lat-lon box. On the other hand, we know the sea-ice emissivity model used by SCEPS for this version of the scene is tuned against the CCI RRDP (AMSR-E and AMSR2).
# 
# We thus go for a hybrid approach (dynamic / fixed) for the tuning of our algorithms:
# * The OW TB samples are selected from the CIMR L1B file, in a lat-lon box over open water conditions.
# * The CI TB samples are taken from the CCI RRDP (AMSR-E and AMSR2).
# 

# In[5]:


# locate and read the RRDP files
rrdp_dir = os.path.abspath('.') + '/sirrdp'
area = 'nh'
ow_files, ci_files = rrdp_file.find_rrdp_files(rrdp_dir, area=area, years=(2007, 2013, 2018))

channels_needed = []
for alg in algos.keys():
    channels_needed += algos[alg]['channels']
channels_needed = set(channels_needed)

rrdp_pos = rrdp_file.read_RRDP_files(area, ow_files, ci_files, channels=channels_needed)


# In[6]:


# boxes of lat/lon: (lat_min, lat_max, lon_min, lon_max)
# To simplify, we use lat/lon boxes here, but in an operational
# mode the region masks would be defined with more advanced geomasks.
OW_box = (73., 76.2, -2., 40)
# CI_box = (82.5, 88., 5, 50)


# Locate CIMR L1B files to use in the training (for the time being we only have 1 realistic swath, so we must use it. But in an operational context the offline training would access a rolling archive of past L1B data, typically 5-10 days.
# 
# To simplify, we only extract OW and CI samples from one swath file from the archive.

# In[7]:


arch_l1b_fn_patt = 'SCEPS_l1b_sceps_geo_polar_scene_1_unfiltered_tot_minimal_nom_nedt_apc_tot_v2p1.nc'
arch_l1b_fn_patt = os.path.join(l1b_archive_dir, arch_l1b_fn_patt)
arch_l1b_files = glob(arch_l1b_fn_patt)
if len(arch_l1b_files) == 0:
    print("WARNING : did not find any good L1B file in the archive ({})".format(arch_l1b_fn_patt))


# Do the training sample extraction and the storing into tie-point objects for each algorithm in turn (`CKA`, `KKA`, and `KA`).

# In[8]:


reload(coll)
for algo in algos.keys():
    b = algos[algo]['target_band']
    bands_needed = list(set([tb_dict[b[:-1]] for b in algos[algo]['channels']]))
    print(algo, bands_needed, )
    arch_l1b = io.CIMR_L1B(arch_l1b_files[0], keep_calibration_view=True, selected_bands=bands_needed)
    
    # if asked by the user, apply the (pre-computed) OZA adjustment fields for all bands
    if use_oza_adjust:
        arch_l1b.apply_OZA_adjustment()
    
    # coarsen l1b samples along the scanlines with a kernel of 5 (horns are *not* combined)
    arch_l1b = arch_l1b.coarsen_along_scanlines(kernel=5)
    # collocate TBs to the target band of the algorithm.
    if algo == 'KA':
        # we use only one frequency: no need for collocation
        pass
    elif algo == 'KKA':
        # the collocation is done along scanlines (to not mix feeds with different OZAs)
        arch_l1b = arch_l1b.collocate_along_scan(target_band=b)
    elif algo == 'CKA':
        # the collocation is done across scanlines and mixes feeds with different OZAs
        #    note : here we also mix forward and aft- scans which is maybe not optimal.
        arch_l1b = coll.collocate_channels(arch_l1b.data, b, method='nn')
    else:
        raise NotImplementedError("Collocation step missing for {}".format(algo))
    
    # how to access the lat and lon depends on the collocation type
    if isinstance(arch_l1b, io.CIMR_L1B):
        _lat = arch_l1b.data[b].lat
        _lon = arch_l1b.data[b].lon
    elif str(arch_l1b['_type']) == 'L1C swath':
        _lat = arch_l1b['geolocation']['lat']
        _lon = arch_l1b['geolocation']['lon']
    else:
        raise ValueError("Unrecognized L1 data type.")
    
    # Create the masks for OW and CI from the lat-lon boxes
    _mask = dict()
    _mask['ow'] = (_lat > OW_box[0])*(_lat < OW_box[1])*(_lon > OW_box[2])*(_lon < OW_box[3]).astype('int')
    #_mask['ci'] = (_lat > CI_box[0])*(_lat < CI_box[1])*(_lon > CI_box[2])*(_lon < CI_box[3]).astype('int')
    
    # Prepare for extracting the training samples.We also store the feed number (e.g. 0-7 for KU/KA)
    #   to be able to later train algorithm for specific feeds.
    dyn_tbs = dict()
    dyn_tbs['ow'] = dict()
    #dyn_tbs['ci'] = dict()
    dyn_tbs_feed = dict()
    dyn_tbs_feed['ow'] = None
    #dyn_tbs_feed['ci'] = None
    
    # Extract the brightness temperatures in the OW areas and store in sample dictionaries
    for ch in algos[algo]['channels']:
        # for each input channel to the algorithm (e.g. tb19v), deduce the
        #   name of the variable to be read in the L1B (or L1C) data structure.
        bn = tb_dict[ch[:-1]]
        pol = ch[-1:]
        
        if isinstance(arch_l1b, io.CIMR_L1B):
            bnstr = ''
            if bn != b:
                bnstr = '{}_'.format(bn)
            tb_n = '{}brightness_temperature_{}'.format(bnstr, pol)
            # read and apply the OW or CI mask
            for w in ('ow',):
                dyn_tbs[w][ch] = arch_l1b.data[b][tb_n].where(_mask[w]).to_masked_array().compressed()
                # keep the feed number information only for the target band
                if bn == b and dyn_tbs_feed[w] is None:
                    dyn_tbs_feed[w] = arch_l1b.data[b]['orig_horn'].where(_mask[w]).to_masked_array().compressed()
        elif str(arch_l1b['_type']) == 'L1C swath':
            tb_n = 'brightness_temperature_{}'.format(pol)
            # read and apply the OW or CI mask
            for w in ('ow',):
                dyn_tbs[w][ch] = arch_l1b[bn+'_BAND'][tb_n].where(_mask[w]).to_masked_array().compressed()
                # keep the feed number information only for the target band
                if bn == b and dyn_tbs_feed[w] is None:
                    dyn_tbs_feed[w] = arch_l1b[bn+'_BAND']['orig_horn'].where(_mask[w]).to_masked_array().compressed()
        else:
            raise ValueError("Unrecognized L1 data type.")
            
    #
    # TUNING COMBINING ALL FEEDS INTO ONE ALGORITHM
    #
    
    # Transfer the training data in a tie-point object. This involves some processing,
    #  like computing the tie-points and their covariance matrices. 
    ow_tp = tp.OWTiepoint(source='CIMRL1B-ALLFEEDS', tbs=dyn_tbs['ow'])
    ci_tp = tp.CICETiepoint(source='rrdp', tbs=rrdp_pos['ci'])
    ow_tp.instr = ci_tp.instr = 'CIMR'
    print("OW tie-point for {} ALLFEEDS: {}".format(algo, ow_tp.tp))

    # Tune the SIC algorithm on these tie-points
    tuned_algo = hybrid_algo.HybridSICAlgo(algos[algo]['channels'], ow_tp, ci_tp)

    # Store the algorithm on disk (JSON file) for later re-use
    json_fn = os.path.join(out_dir,'{}_sic_{}.json'.format(algo.upper(), ow_tp.source.upper()))
    with open(json_fn, 'w') as fp_out:
        json.dump(tuned_algo.strip().to_dict(), fp_out, indent=4, sort_keys=True, cls=MyEncoder)

    print("{} SIC configuration file is in {}".format(algo, json_fn))
    
    #
    # TUNING SEPARATE ALGORITHMS, ONE FOR EACH FEED
    #
    for feed in range(0, io.n_horns[b]):
        FEEDNB = 'FEED{}'.format(feed)
        
        dyn_tbs_f = dict()
        dyn_tbs_f['ow'] = dict()
        #dyn_tbs_f['ci'] = dict()
        for ch in algos[algo]['channels']:
            for w in ('ow',):
                dyn_tbs_f[w][ch] = dyn_tbs[w][ch][dyn_tbs_feed[w]==feed]
        
        ow_tp = tp.OWTiepoint(source='CIMRL1B-{}'.format(FEEDNB), tbs=dyn_tbs_f['ow'])
        ci_tp = tp.CICETiepoint(source='CCI-RRDP', tbs=rrdp_pos['ci'])
        ow_tp.instr = ci_tp.instr = 'CIMR'
        print("OW tie-point for {} {}: {}".format(algo, FEEDNB, ow_tp.tp))
        
        # Tune the SIC algorithm on these tie-points
        tuned_algo = hybrid_algo.HybridSICAlgo(algos[algo]['channels'], ow_tp, ci_tp)

        # Store the algorithm on disk (JSON file) for later re-use
        json_fn = os.path.join(out_dir,'{}_sic_{}.json'.format(algo.upper(), ow_tp.source.upper()))
        with open(json_fn, 'w') as fp_out:
            json.dump(tuned_algo.strip().to_dict(), fp_out, indent=4, sort_keys=True, cls=MyEncoder)

        print("{} SIC configuration file is in {}".format(algo, json_fn))


# At the end, we also run a tuning of each algorithm against the static set of SIC0 and SIC1 data points
# from the ESA CCI Round Robin Data Package (AMSR-E + AMSR2 TBs). This would not be used in the operational processing, but is used now to demonstrate the impact of dynamic tuning of the algorithms in the Performance Assessment chapter.
# 
# the training data are taken from the ESA CCI Sea Ice Concentration Round Robin Data Package of Pedersen et al. (2019). The relevant data files as well as routines to parse the files are stored in module siddrp/.
# 
# Pedersen, Leif Toudal; Saldo, Roberto; Ivanova, Natalia; Kern, Stefan; Heygster, Georg; Tonboe, Rasmus; et al. (2019): Reference dataset for sea ice concentration. figshare. Dataset. https://doi.org/10.6084/m9.figshare.6626549.v7

# In[9]:


# run the tuning for each algorithm in turn
for algo in algos.keys():
    
    # Transfer the training data in a tie-point object. This involves some processing,
    #  like computing the tie-points and their covariance matrices. 
    ow_tp = tp.OWTiepoint(source='rrdp', tbs=rrdp_pos['ow'])
    ci_tp = tp.CICETiepoint(source='rrdp', tbs=rrdp_pos['ci'])
    ow_tp.instr = ci_tp.instr = 'AMSRs'
    ow_tp.source = ci_tp.source = 'CCI-RRDP'
    
    # Tune the SIC algorithm on these tie-points
    tuned_algo = hybrid_algo.HybridSICAlgo(algos[algo]['channels'], ow_tp, ci_tp)
    
    # Store the algorithm on disk (JSON file) for later re-use
    json_fn = os.path.join(out_dir,'{}_sic_{}.json'.format(algo.upper(), ow_tp.source.upper()))
    with open(json_fn, 'w') as fp_out:
        json.dump(tuned_algo.strip().to_dict(), fp_out, indent=4, sort_keys=True, cls=MyEncoder)

    print("{} SIC configuration file is in {}".format(algo, json_fn))


# In[ ]:




