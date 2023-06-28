from __future__ import print_function

import os
import glob
import numpy as np
from math import ceil, floor

from . import readdata

#import readdata

rrdp_file_sep = ','
rrdp_file_dtype_ref_crrel = [['ref_lat','f4'],
                 ['ref_lon','f4'],
                 ['ref_timestamp',"DT/%Y-%m-%dT%H:%M:%SZ"],
                 ['ref_id','S20'],
                 ['ref_tdiff','int'],
                 ['ref_timestamp2','DT/%m/%d/%Y %H:%M'],
                 ['ref_lat2','f4'],
                 ['ref_lon2','f4'],
                 ['ref_qual','S12'],
                 ['ref_t2m','f4'],
                 ['ref_mslp','f4'],
                 ['ref_ssp','f4'],
                 ['ref_it','f4'],
                 ['ref_isp','f4'],
                 ['ref_ibp','f4'],
                 ['ret_T1','f4',],
                 ['ret_T2','f4',],
                 ['ret_T3','f4',],
                 ['ret_T4','f4',],
                 ['ret_T5','f4',],
                 ['ret_T6','f4',],
                 ['ret_T7','f4',],
                 ['ret_T8','f4',],
                 ['ret_T9','f4',],
                 ['ret_T10','f4',],
                 ['ret_T11','f4',],
                 ['ret_T12','f4',],
                 ['ret_T13','f4',],
                 ['ret_T14','f4',],
                 ['ret_T15','f4',]]
rrdp_file_dtype_ref_sic0 = [['ref_lat','f4'],
                 ['ref_lon','f4'],
                 ['ref_timestamp',"DT/%Y-%m-%dT%H:%M:%SZ"],
                 ['ref_id','S12'],
                 ['ref_sic','f4']]

rrdp_file_dtype_ref_sic1 = rrdp_file_dtype_ref_sic0 + [['ref_areachange','f4']]
rrdp_file_dtype_ref_thinice = rrdp_file_dtype_ref_sic0 + [['ref_sit','f4']]


rrdp_file_dtype_era = [['era_lat','f4'],
                       ['era_lon','f4'],
                       ['era_timestamp',"DT/%Y-%m-%dT%H:%M:%SZ"],
                       ['era_id','S12'],
                       ['era_ufile','S50'],
                       ['era_msl','f4'],
                       ['era_u10','f4'],
                       ['era_v10','f4'],
                       ['era_ws','f4'],
                       ['era_t2m','f4'],
                       ['era_skt','f4'],
                       ['era_istl1','f4'],
                       ['era_istl2','f4'],
                       ['era_istl3','f4'],
                       ['era_istl4','f4'],
                       ['era_sst','f4'],
                       ['era_d2m','f4'],
                       ['era_tcwv','f4'],
                       ['era_tclw','f4'],
                       ['era_tciw','f4'],
                       ['era_ssrd','f4'],
                       ['era_strd','f4'],
                       ['era_e','f4'],
                       ['era_tp','f4'],
                       ['era_sf','f4'],
                       ['era_fal','f4'],
                       ['era_ci','f4']]



rrdp_file_dtype_pmr1 = [['pmr_lat','f4'],
                 ['pmr_lon','f4'],
                 ['pmr_timestamp',"DT/%Y-%m-%dT%H:%M:%SZ"],
                 ['pmr_id','S12']]
rrdp_file_dtype_pmr2 = [['pmr_eia','f4'],
                        ['pmr_eaa','f4'],
                        ['pmr_scanpos','int'],
                        ['pmr_ufile','S50'],
                        ['pmr_tdiff','int']]

rrdp_file_dtype_scat = [['scat_lat','f4'],
                        ['scat_lon','f4'],
                        ['scat_timestamp','DT/%Y-%m-%dT%H:%M:%SZ'],
                        ['scat_id','S32'],
                        ['scat_ufile','S50'],
                        ['scat_s0','f4'],
                        ['scat_s0_mask','f4'],
                        ['scat_nbs','int'],
                        ['scat_flag','int'],
                        ['scat_std','f4']]

rrdp_file_dtype_smos = [['smos_lat','f4'],
                        ['smos_lon','f4'],
                        ['smos_timestamp','DT/%Y-%m-%dT%H:%M:%SZ'],
                        ['smos_id','S32'],
                        ['smos_ufile','S50'],
                        ['smos_tbv','f4'],
                        ['smos_tbh','f4'],
                        ['smos_rmse_v','f4'],
                        ['smos_rmse_h','f4'],
                        ['smos_nmp','int'],
                        ['smos_dataloss','int'],]

rrdp_file_dtype_smap = [['smap_lat','f4'],
                        ['smap_lon','f4'],
                        ['smap_timestamp','DT/%Y-%m-%dT%H:%M:%SZ'],
                        ['smap_id','S32'],
                        ['smap_ufile','S50'],
                        ['smap_tbv','f4'],
                        ['smap_tbh','f4'],
                        ['smap_std_v','f4'],
                        ['smap_std_h','f4'],
                        ['smap_nmp','int'],]

# latitude,longitude,time,reference-id,upstreamfile,sigma0_inner,sigma0_mask_inner,nb_inner,std_inner,sigma0_outer,sigma0_mask_outer,nb_outer,std_outer
rrdp_file_dtype_qscat = [['qscat_lat','f4'],
                        ['qscat_lon','f4'],
                        ['qscat_timestamp','DT/%Y-%m-%dT%H:%M:%SZ'],
                        ['qscat_id','S32'],
                        ['qscat_ufile','S32'],
                        ['qscat_s0_inn','S50'],
                        ['qscat_mask_inn','f4'],
                        ['qscat_nb_inn','f4'],
                        ['qscat_std_inn','f4'],
                        ['qscat_s0_out','S50'],
                        ['qscat_mask_out','f4'],
                        ['qscat_nb_out','f4'],
                        ['qscat_std_out','f4'],]


def get_channels(instr):
    if 'SSMI' in instr:
        channels = ['tb19h','tb19v','tb22v','tb37h','tb37v','tb90h','tb90v']
    elif 'AMSR' in instr:
        channels = ['tb06h','tb06v','tb07h','tb07v','tb10h','tb10v','tb19h','tb19v','tb22h','tb22v','tb37h','tb37v','tb90h','tb90v']
    else:
        raise ValueError('Instrument %s is not supported for RRDP files' % (instr,))
    return channels

def load_rrdp_file(fn, max_n_lines=None):

    if not isinstance(fn,type([])):
        fn = [fn]

    fn_bn = os.path.basename(fn[0])

    instr = 'AMSR'
    if 'SIC1' in fn_bn:
        what = 'SIC1'
    elif 'SIC0' in fn_bn:
        what = 'SIC0'
    else:
        raise NotImplementedError('un-supported RRDP file type {}'.format(fn_bn))

    channels = get_channels(instr)
    rrdp_file_dtype_tbs = [[ch,'f4'] for ch in channels]

    rrdp_file_dtype = []
    if 'SIC0' in what:
        rrdp_file_dtype.extend(rrdp_file_dtype_ref_sic0)
    elif 'SIC1' in what:
        rrdp_file_dtype.extend(rrdp_file_dtype_ref_sic1)
    else:
        raise ValueError('Not a supported RRDP2 file (what = {})'.format(what,))

    if 'ERA' in fn_bn: 
        rrdp_file_dtype.extend(rrdp_file_dtype_era)

    rrdp_file_dtype.extend(rrdp_file_dtype_pmr1)
    rrdp_file_dtype.extend(rrdp_file_dtype_tbs)
    rrdp_file_dtype.extend(rrdp_file_dtype_pmr2)

    if 'ASCAT' in fn_bn:
        rrdp_file_dtype.extend(rrdp_file_dtype_scat)
    if 'SMOS' in fn_bn:
        rrdp_file_dtype.extend(rrdp_file_dtype_smos)
    if 'SMAP' in fn_bn:
        rrdp_file_dtype.extend(rrdp_file_dtype_smap)
    if 'QSCAT' in fn_bn:
        rrdp_file_dtype.extend(rrdp_file_dtype_qscat)

    header=','.join([i for i,_ in rrdp_file_dtype])
    types=','.join([t for _,t in rrdp_file_dtype])

    ##print header
    ##print types

    fc = readdata.readdata(fn,rrdp_file_dtype,sep=rrdp_file_sep,max_n_lines=max_n_lines)

    # we are only interested in lines that have both era and AMSR data...
    mask = np.zeros((len(fc['ref_lat']),)).astype('bool')
    for var in ('era_ws','era_t2m','tb19h','tb90h',):
        mask = np.logical_or(mask,fc[var]==-999)


    # we are only interested on lines that have polar latitude
    mask = np.logical_or(mask,abs(fc['pmr_lat']) < 45.)
    fc.mask = mask


    return fc

def find_rrdp_files(rrdp_dir, area, years=range(2016,2020)):
    """ Search for SIC0 and SIC1 files in the RRDP dataset, looking both for v2 and v3 files """
    ow_files = []
    ci_files = []
    for year in years:
        if year <= 2011:
            instr = 'AMSR'
            rrdp = 'rrdp2'
        elif year < 2016:
            instr = 'AMSR2'
            rrdp = 'rrdp2'
        else:
            instr = 'AMSR2'
            rrdp = 'rrdp3'
        
        rrdpdir = os.path.join(rrdp_dir,{'rrdp2':'RRDP_v2.0','rrdp3':'RRDP_v3.0'}[rrdp])
        
        ow_patt = '*-{i:}-*-???{b:}-{y:}-{h:}.text'.format(i=instr, b='SIC0', y=year, h=area[0].upper())
        for sdir in ('','*'):
            f = glob.glob(os.path.join(rrdpdir, sdir, ow_patt))
            if len(f) == 1:
                ow_files.append(f[0])
                break
            elif len(f) > 1:
                raise ValueError("WARNING!!! pattern {} corresponds to several files in {}".format(ow_patt, rrdpdir))
        
        ci_patt = '*-{i:}-*-???{b:}-{y:}-{h:}.text'.format(i=instr, b='SIC1', y=year, h=area[0].upper())
        for sdir in ('','*'):
            f = glob.glob(os.path.join(rrdpdir, sdir, ci_patt))
            if len(f) == 1:
                ci_files.append(f[0])
                break
            elif len(f) > 1:
                raise ValueError("WARNING!!! pattern {} corresponds to several files in {}".format(ci_patt, rrdpdir))

    return ow_files, ci_files
   
def read_RRDP_pos(f, max_n_lines=None, months=list(range(1,13)), channels=('tb19v','tb37v','tb37h',), with_atm=False):
    
    # read the file
    rrdp_data = load_rrdp_file(f, max_n_lines=max_n_lines)
    
    # screen rrdp data for only winter data
    a_time  = rrdp_data['ref_timestamp'].compressed()
    a_mon   = np.array([dt.month for dt in a_time])
    a_year  = np.array([dt.year for dt in a_time])
    a_mon_winter = np.zeros(a_time.size).astype('bool')
    for wmon in months:
        a_mon_winter = np.logical_or(a_mon_winter,a_mon==wmon)
    a_mon  = a_mon[a_mon_winter]
    a_year = a_year[a_mon_winter]
    
    # screen SIC1 rrdp data on area convergence 
    try:
        a_achange = rrdp_data['ref_areachange'].compressed()
        # as in CCI+ Sea Ice Phase 1 Year 2 PVIR
        ok_achange = (a_achange >= 0.985) * (a_achange <= 0.996)
    except ValueError:
        # an OW file
        ok_achange = None
        pass
    
    # transfer to a dict(), keeping only the variables we are interested in
    ret   = dict()
    ret_ids   = ['ref_lat','ref_lon', 'ref_timestamp','pmr_eia',]
    if with_atm:
        ret_ids.append(['era_ws', 'era_t2m', 'era_skt', 'era_tcwv', 'era_tclw',])
    ret_ids.extend(channels)

    for i in ret_ids:
        ret[i] = rrdp_data[i].compressed()
        ret[i] = (ret[i])[a_mon_winter]
        if ok_achange is not None:
            ret[i] = (ret[i][ok_achange])
        
    return ret

def concat_RRDP_pos(pos, fc):

    keys = fc.keys()
    try:
        for k in keys:
            pos[k] = np.append(pos[k], fc[k])
    except KeyError as k:
        pos = {}
        for k in keys:
            pos[k] = fc[k]
    
    return pos

def read_RRDP_files(area, ow_files, ci_files, channels=('tb19v','tb37v','tb37h'), with_atm=False): 
    max_n_lines = None
    pos = {}
    pos['ow'] = {}
    pos['ci'] = {}
    
    # OW : we keep all months
    for f in ow_files:
        fc = read_RRDP_pos(f, max_n_lines = max_n_lines, channels=channels, with_atm=with_atm)
        pos['ow'] = concat_RRDP_pos(pos['ow'], fc)
    
    print("Number of OW samples: ",len(pos['ow']['ref_lat']))
        
    # CICE : we keep only the winter months
    #   as in CCI+ Sea Ice Phase 1 Year 2 PVIR
    if area == 'nh':
        winter_months = [11, 12, 1, 2, 3, 4]
    elif area == 'sh':
        winter_months = [5, 6, 7, 8, 9, 10]

    for f in ci_files:
        fc = read_RRDP_pos(f, max_n_lines = max_n_lines, channels=channels, with_atm=with_atm)
        months = np.array([dt.month for dt in fc['ref_timestamp']])
        select_month = np.zeros(len(fc['ref_lat'])).astype('bool')
        for m in winter_months:
            select_month[m == months] = True
        for k in fc.keys():
            fc[k] = fc[k][select_month]
        pos['ci'] = concat_RRDP_pos(pos['ci'], fc)
    
    print("Number of CI samples: ",len(pos['ci']['ref_lat']))

    return pos


def load_rrdp_data(rrdpdir,instr,area,cice_years,cice_months,max_sel=1000,
                   ow_winter=False,max_n_lines=None, with_nwp=False):

    all_chns = ['tb06v','tb06h','tb10v','tb10h','tb19v','tb19h','tb22v','tb37v','tb37h','tb90v','tb90h',]
    if with_nwp:
        all_chns.extend(['era_ws','era_t2m','era_tcwv','pmr_eia'])

    Area = area.upper()[0]

    patt = os.path.join(rrdpdir,'DMI_SIC0_'+Area,'QSCAT-vs-SMAP-vs-SMOS-vs-ASCAT-vs-{}-vs-ERA-vs-DMISIC0-*-{}.text'.format(instr,Area))
    all_ow_files = glob.glob(patt)
    if len(all_ow_files) == 0:
        raise ValueError("WARNING!!!! No DMISIC0 files for {} {} ({})".format(instr,area,patt))


    all_cice_files = []
    for y in cice_years:
        patt = os.path.join(rrdpdir,'DTU_SIC1_'+Area,'QSCAT-vs-SMAP-vs-SMOS-vs-ASCAT-vs-{}-vs-ERA-vs-DTUSIC1-{}-{}.text'.format(instr,y,Area,))
        new_cice_files = glob.glob(patt)
        all_cice_files.extend(new_cice_files)

    if len(all_cice_files) == 0:
        raise ValueError("WARNING!!!! No DTUSIC1 files for {} {} (years {})".format(instr,area,cice_years))

    # read RRDP
    rrdp_cice = load_rrdp_file(all_cice_files,max_n_lines=max_n_lines)
    rrdp_ow   = load_rrdp_file(all_ow_files,max_n_lines=max_n_lines)

    # screen rrdp data for only winter data
    cice_time  = rrdp_cice['ref_timestamp'].compressed()
    cice_mon   = np.array([dt.month for dt in cice_time])
    cice_year  = np.array([dt.year for dt in cice_time])
    cice_mon_winter = np.zeros(cice_time.size).astype('bool')
    for wmon in cice_months:
        cice_mon_winter = np.logical_or(cice_mon_winter,cice_mon==wmon)
    cice_mon  = cice_mon[cice_mon_winter]
    cice_year = cice_year[cice_mon_winter]

    ow_time  = rrdp_ow['ref_timestamp'].compressed()
    ow_mon   = np.array([dt.month for dt in ow_time])
    ow_year  = np.array([dt.year for dt in ow_time])
    ow_mon_winter = np.zeros(ow_time.size).astype('bool')
    for wmon in cice_months:
        ow_mon_winter = np.logical_or(ow_mon_winter,ow_mon==wmon)
    if ow_winter:
        ow_mon = ow_mon[ow_mon_winter]
    ow_year = ow_year[ow_mon_winter]

    ow_tbs   = dict()
    cice_tbs = dict()
    ow_tb_ids   = ['pmr_lat','pmr_lon',] + all_chns
    cice_tb_ids = ['pmr_lat','pmr_lon',] + all_chns

    for ch in ow_tb_ids:
        ow_tbs[ch]   = rrdp_ow[ch].compressed()
        if ow_winter:
            ow_tbs[ch] = (ow_tbs[ch])[ow_mon_winter]

    for ch in cice_tb_ids:
        cice_tbs[ch] = rrdp_cice[ch].compressed()
        cice_tbs[ch] = (cice_tbs[ch])[cice_mon_winter]


    # remove some points as not not to clutter the graph
    one_ch = ow_tbs.channels[0] 
    ow_ev   = int(ceil(len(ow_tbs[one_ch]) / float(max_sel)))
    cice_ev = int(ceil(len(cice_tbs[one_ch]) / float(max_sel)))
    #print("Select every {} OW points".format(ow_ev,))
    #print("Select every {} CICE points".format(cice_ev,))
    for ch in ow_tbs.keys():
        ow_tbs[ch]   = ow_tbs[ch][::ow_ev]
    for ch in cice_tbs.keys():
        cice_tbs[ch] = cice_tbs[ch][::cice_ev]
    cice_mon  = cice_mon[::cice_ev]
    cice_year = cice_year[::cice_ev]
    ow_mon    = ow_mon[::ow_ev]
    ow_year   = ow_year[::ow_ev]

    ow_tbs['mon']    = ow_mon
    ow_tbs['year']   = ow_year
    cice_tbs['mon']  = cice_mon
    cice_tbs['year'] = cice_year

    return ow_tbs,cice_tbs

def load_rrdp_samples(rrdpdir,instr,area,what,years,months,max_sel=1000,max_n_lines=None, with_nwp=False,channels='all'):

    if channels == 'all':
        all_chns = ['tb06v','tb06h','tb10v','tb10h','tb19v','tb19h','tb22v','tb22h','tb37v','tb37h','tb90v','tb90h',]
        if with_nwp:
            all_chns.extend(['era_ws','era_t2m','era_tcwv','pmr_eia'])
    else:
        all_chns = list(channels)

    Area = area.upper()[0]

    if what not in ('ow','cice'):
        raise ValueError('parameter what should be either "ow" or "cice"')

    if what == 'ow':
        spec = ('DMI','SIC0')
    else:
        spec = ('DTU','SIC1')

    # find the files
    all_files = []
    for y in years:
        patt = '*-{i:}-*-???{b:}-{y:}-{h:}.text'.format(i=instr,b=spec[1],y=y,h=Area)
        for sdir in ('','*'):
            f = glob.glob(os.path.join(rrdpdir, sdir, patt))
            if len(f) == 1:
                all_files.append(f[0])
                break
            elif len(f) > 1:
                raise ValueError("WARNING!!! pattern {} corresponds to several files in {}".format())

    if len(all_files) == 0:
        raise ValueError("WARNING!!!! No {} files for {} {}\n\t{}".format(what.upper(),instr,area,rrdpdir))

    # read RRDP
    rrdp_data = load_rrdp_file(all_files,max_n_lines=max_n_lines)

    # screen rrdp data for only winter data
    a_time  = rrdp_data['ref_timestamp'].compressed()
    a_mon   = np.array([dt.month for dt in a_time])
    a_year  = np.array([dt.year for dt in a_time])
    a_mon_winter = np.zeros(a_time.size).astype('bool')
    for wmon in months:
        a_mon_winter = np.logical_or(a_mon_winter,a_mon==wmon)
    a_mon  = a_mon[a_mon_winter]
    a_year = a_year[a_mon_winter]

    # transfer to a dict()
    tbs   = dict()
    tb_ids   = ['pmr_lat','pmr_lon',] + all_chns
    for ch in tb_ids:
        tbs[ch]   = rrdp_data[ch].compressed()
        tbs[ch] = (tbs[ch])[a_mon_winter]

    # remove some points
    one_ch = all_chns[0]
    ev   = int(ceil(len(tbs[one_ch]) / float(max_sel)))
    if ev == 0: ev = 1
    print("Select every {} {} points".format(ev, what.upper(),))
    for ch in tbs.keys():
        tbs[ch]   = tbs[ch][::ev]

    tbs['mon']    = a_mon[::ev]
    tbs['year']   = a_year[::ev]

    return tbs

if __name__ == '__main__':
    """ Small main routine to demonstrate the loading the RRDP files """
    import sys

    rrdp2_dir = './RRDP_v2.0/'
    rrdp3_dir = './RRDP_v3.0/'

    max_lines = 1000

    for h in ('N','S'):
        # test loading a RRDP v3 file
        rrdp_files = glob.glob(os.path.join(rrdp3_dir,'*-DTUSIC1-*-{}.text'.format(h)))
        d = load_rrdp_file(rrdp_files[0],max_n_lines=max_lines)
        rrdp_files = glob.glob(os.path.join(rrdp3_dir,'*-DTUSIC0-*-{}.text'.format(h)))
        d = load_rrdp_file(rrdp_files[0],max_n_lines=max_lines)

        # test loading a RRDP v2 file
        rrdp_files = glob.glob(os.path.join(rrdp2_dir,'*','*-DTUSIC1-*-{}.text'.format(h)))
        d = load_rrdp_file(rrdp_files[0],max_n_lines=max_lines)
        rrdp_files = glob.glob(os.path.join(rrdp2_dir,'*','*-DMISIC0-*-{}.text'.format(h)))
        d = load_rrdp_file(rrdp_files[0],max_n_lines=max_lines)


