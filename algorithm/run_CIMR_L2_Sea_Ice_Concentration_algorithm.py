#!/usr/bin/env python3
"""
   Python script to run CIMR L2 SIC algorithm notebook from the command-line, using papermill
"""

import sys
import papermill as pm
from datetime import date
from dateutil.relativedelta import *

default_l2_dir = '.'
default_l2_grid = 'ease2-2.5km-arctic'

def run_notebook(l1b_path, l2_dir = default_l2_dir, l2_grid = default_l2_grid):
    
    notebook = 'CIMR_L2_Sea_Ice_Concentration_algorithm.ipynb'
    notebook_out = notebook.replace('.ipynb','_out.ipynb')

    notebook_par = {'l1b_path': l1b_path, 'l2_dir': l2_dir, 'l2_grid': l2_grid,}

    print("Call {} with params\n\t{}".format(notebook, notebook_par))

    _ = pm.execute_notebook(notebook,notebook_out,parameters=notebook_par)


if __name__ == '__main__':
    import argparse

    # prepare and parse parameters to the script
    parser = argparse.ArgumentParser(prog='run_CIMR_L2_Sea_Ice_Concentration_algorithm.py',
                                     description='run CIMR L2 SIC algorithm prepared in CIMR DEVALGO')
    parser.add_argument('L1B_PATH', help='Path to the CIMR L1B file.')
    parser.add_argument('-o', help='Directory where to write the L2 SIC file', default=default_l2_dir)
    parser.add_argument('--no-oza-adjust', help='Do NOT do OZA adjustement.', dest='use_oza_adjust', action='store_false')
    parser.add_argument('--aux-dir', help='Where to read the.', dest='use_oza_adjust', action='store_false')
    parser.add_argument('-g', help='Target grid of the L2 SIC algorithm.', default=default_l2_grid)
    args = parser.parse_args()

    # run the L2 SIC algorithm via the notebook
    print('++++++++ START L2 Sea Ice Concentration ++++')
    sic_notebook = './CIMR_L2_Sea_Ice_Concentration_algorithm.ipynb'
    sic_notebook_out = os.path.join(args.t, os.path.basename(sic_notebook).replace('.ipynb','_out.ipynb'))
    sic_notebook_par = {'l1b_path': oza_l1b_file, 'l2_dir': out_dir, 'l2_grid': args.g, 'use_oza_adjust': args.use_oza_adjust,
                        'aux_dir': tmp_dir, 'tuning_method':'CIMRL1B-PERFEED', 'l1b_archive_dir': os.path.dirname(oza_l1b_file)}
    _ = pm.execute_notebook(sic_notebook,sic_notebook_out,parameters=sic_notebook_par,cwd=os.path.dirname(sic_notebook))
    print('++++++++ END L2 Sea Ice Concentration ++++')

