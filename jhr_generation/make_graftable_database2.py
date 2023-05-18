#!/usr/bin/env python

import os
import sys
import argparse
import math
import json

# import npose

from importlib import reload
import npose_util
reload(npose_util)
from npose_util import *


# This program takes a helix_list and a turn_list and tries to splice them
# It then generates a database of splices that were successful



parser = argparse.ArgumentParser()
parser.add_argument("helix_list", type=str )
parser.add_argument("turn_list", type=str )
parser.add_argument("-turn_rmsd_res", type=int, default=2)
parser.add_argument("-helix_rmsd_res", type=int, default=1)
parser.add_argument("-max_rmsd", type=float, default=0.3)
parser.add_argument("-min_added_len", type=int, default=3)
parser.add_argument("-out_folder", type=str, default="graftable_databases")


args = parser.parse_args(sys.argv[1:])

turn_rmsd_res = args.turn_rmsd_res
helix_rmsd_res = args.helix_rmsd_res
max_rmsd = args.max_rmsd
min_added_len = args.min_added_len

if (not os.path.exists(args.out_folder)):
    os.makedirs(args.out_folder)


# Make sure we don't just add the last res
#  Because then we'd have to graft onto it from the other direction
assert(min_added_len >= 2)

# The last res has a weird O, we don't want it
assert(helix_rmsd_res >= 1)
assert(turn_rmsd_res >= 1)

# You can't rmsd residues you don't have!!
assert(turn_rmsd_res <= min_added_len)


# Get the names of the pdbs

helix_list = []
with open(args.helix_list) as f:
    for line in f:
        line = line.strip()
        if (len(line) == 0):
            continue
        helix_list.append(line)

turn_list = []
with open(args.turn_list) as f:
    for line in f:
        line = line.strip()
        if (len(line) == 0):
            continue
        turn_list.append(line)


# Load all the pdbs at once because we can

print("Loading helices")
helices = []
helix_names = []
ithelices = []
ncac_helices = []
for fname in helix_list:
    npose = npose_from_file_fast(fname)
    helices.append(npose)
    ithelices.append(itpose_from_tpose(tpose_from_npose(npose)))
    ncac_helices.append(extract_N_CA_C(npose))
    helix_names.append(get_tag(fname))

print("Loading turns")
turns = []
turn_names = []
for fname in turn_list:
    turns.append(npose_from_file_fast(fname))
    turn_names.append(get_tag(fname))


print(len(turns), len(helices))

print("Moving all helices to origin")



ncac_helices_at_orig_prepend = []
ncac_helices_at_orig_postpend = []
for ihelix in range(len(helices)):
    ncac = ncac_helices[ihelix]
    ithelix = ithelices[ihelix]


    start_helix_res = min_added_len
    after_end_helix_res = len(ithelix) - helix_rmsd_res

    with_i_at_orig = []
    for i in range(len(ithelix)):
        if ( i < start_helix_res or i >= after_end_helix_res ):
            # with_i_at_orig.append(None)
            continue

        first_res = i - turn_rmsd_res
        after_last_res = i + 1 + helix_rmsd_res

        subhelix = ncac[first_res*3:after_last_res*3]

        xformed = xform_npose(ithelix[i], subhelix )

        with_i_at_orig.append(xformed[:,:3])

    ncac_helices_at_orig_prepend.append(np.array(with_i_at_orig))


    start_helix_res = helix_rmsd_res
    after_end_helix_res = len(ithelix) - min_added_len

    with_i_at_orig = []
    for i in range(len(ithelix)):
        if ( i < start_helix_res or i >= after_end_helix_res ):
            # with_i_at_orig.append(None)
            continue

        first_res = i - helix_rmsd_res
        after_last_res = i + 1 + turn_rmsd_res

        subhelix = ncac[first_res*3:after_last_res*3]

        xformed = xform_npose(ithelix[i], subhelix )

        with_i_at_orig.append(xformed[:,:3])

    ncac_helices_at_orig_postpend.append(np.array(with_i_at_orig))


literally_a_range = np.arange(0, 100)



db = {}



for iturn, turn in enumerate(turns):
    if ( iturn % 1000 == 0 ):
        print(iturn, "/", len(turns))
    turn_name = turn_names[iturn]

    tturn = tpose_from_npose(turn)

    prepend_db = {}
    postpend_db = {}


    prepend_resnum = turn_rmsd_res
    prepend_xform = tturn[prepend_resnum]

    turn_for_rmsd = turn[(prepend_resnum - turn_rmsd_res)*R:(prepend_resnum+1+helix_rmsd_res)*R]
    turn_rmsd_atoms = extract_N_CA_C( turn_for_rmsd )

    turn_rmsd_atoms = xform_npose( np.linalg.inv( prepend_xform ), turn_rmsd_atoms )[:,:3]

    for ihelix, helix in enumerate(helices):
        ithelix = ithelices[ihelix]
        ncac_helix = ncac_helices[ihelix]

        # Let min_added_len = 3
        # Let helix_rmsd_res = 1
        # Let turn_rmsd_res = 2
        #
        #           |    # bounded by helix_rmsd_res
        # HHHHHHHHHHHH
        #         TTTTTT
        #    |            # bounded by min_added_len
        # HHHHHHHHHHHH
        #  TTTTTTT  


        # python style bounds
        start_helix_res = min_added_len
        after_end_helix_res = nsize(helix) - helix_rmsd_res

        ok_pos = []

        with_i_at_orig = ncac_helices_at_orig_prepend[ihelix]

        # this happens in principle
        # with_i_at_orig = with_i_at_orig[start_helix_res:after_end_helix_res]
        # helix_posses = literally_a_range[start_helix_res:after_end_helix_res]

        rmsd2 = np.sum( np.square( with_i_at_orig - turn_rmsd_atoms ), axis=(-2,-1) ) / len(turn_rmsd_atoms)

        # ok_pos = helix_posses[rmsd2 < max_rmsd*max_rmsd]
        ok_pos = np.where( rmsd2 < max_rmsd*max_rmsd )[0] + start_helix_res


        # ok_pos = []

        # for helix_pos in range(start_helix_res, after_end_helix_res):

        #     # We need turn_rmsd_res + 1 + helix_rmsd_res
        #     # first_res = helix_pos - turn_rmsd_res
        #     # after_last_res = helix_pos + 1 + helix_rmsd_res

        #     # subhelix = ncac_helix[first_res*3:after_last_res*3]

        #     # xformed = xform_npose(prepend_xform @ ithelix[helix_pos], subhelix )

        #     xformed = with_i_at_orig[helix_pos-start_helix_res]

        #     rmsd = calc_rmsd( xformed, turn_rmsd_atoms)
        #     # print(rmsd)

        #     # test = xform_npose(prepend_xform @ ithelix[helix_pos], helix )
        #     # dump_npdb(np.concatenate((test[:(helix_pos+1)*R], turn[(turn_rmsd_res+1)*R:])), "rmsd_%.3f.pdb"%(rmsd))

        #     if (rmsd < max_rmsd):

        #         ok_pos.append(helix_pos)


        if (len(ok_pos) > 0):
            prepend_db[helix_names[ihelix]] = list(ok_pos)



    postpend_resnum = nsize(turn) - turn_rmsd_res - 1
    postpend_xform = tturn[postpend_resnum]

    turn_for_rmsd = turn[(postpend_resnum - helix_rmsd_res)*R:(postpend_resnum+1+turn_rmsd_res)*R]
    turn_rmsd_atoms = extract_N_CA_C( turn_for_rmsd )

    turn_rmsd_atoms = xform_npose( np.linalg.inv( postpend_xform ), turn_rmsd_atoms )[:,:3]

    for ihelix, helix in enumerate(helices):
        ithelix = ithelices[ihelix]
        ncac_helix = ncac_helices[ihelix]

        # Let min_added_len = 3
        # Let helix_rmsd_res = 1
        # Let turn_rmsd_res = 2
        #
        #       |             # bounded by helix_rmsd_res
        #      HHHHHHHHHHHH
        #    TTTTTT
        #         |            # bounded by min_added_len
        # HHHHHHHHHHHH
        #     TTTTTTT  


        # python style bounds
        start_helix_res = helix_rmsd_res
        after_end_helix_res = nsize(helix) - min_added_len

        ok_pos = []

        with_i_at_orig = ncac_helices_at_orig_postpend[ihelix]


        # this happens in principle
        # with_i_at_orig = with_i_at_orig[start_helix_res:after_end_helix_res]
        helix_posses = literally_a_range[start_helix_res:after_end_helix_res]

        rmsd2 = np.mean( np.sum( np.square( with_i_at_orig - turn_rmsd_atoms ), axis=-1 ), axis=-1 )

        # ok_pos = helix_posses[rmsd2 < max_rmsd*max_rmsd]
        ok_pos = np.where( rmsd2 < max_rmsd*max_rmsd )[0] + start_helix_res



        # for helix_pos in range(start_helix_res, after_end_helix_res):

        #     # We need helix_rmsd_res + 1 + turn_rmsd_res
        #     # first_res = helix_pos - helix_rmsd_res
        #     # after_last_res = helix_pos + 1 + turn_rmsd_res

        #     # subhelix = ncac_helix[first_res*3:after_last_res*3]

        #     # xformed = xform_npose(postpend_xform @ ithelix[helix_pos], subhelix )

        #     xformed = with_i_at_orig[helix_pos-start_helix_res]

        #     rmsd = calc_rmsd( xformed, turn_rmsd_atoms)

        #     # test = xform_npose(postpend_xform @ ithelix[helix_pos], helix )
        #     # dump_npdb(np.concatenate((turn[:(-turn_rmsd_res-1)*R], test[(helix_pos)*R:])), "rmsd_%.3f.pdb"%(rmsd))

        #     if (rmsd < max_rmsd):

        #         ok_pos.append(helix_pos)


        if (len(ok_pos) > 0):
            postpend_db[helix_names[ihelix]] = list(ok_pos)



        
    db[turn_name] = {"pre":prepend_db, "post":postpend_db}



    
db_name = os.path.basename(args.turn_list) + "_" + os.path.basename(args.helix_list) + \
            "_turnres_%i_helixres_%i_minadd_%i_rmsd_%.3f.json"%(turn_rmsd_res, helix_rmsd_res, min_added_len, max_rmsd)

def convert(o):
    if isinstance(o, np.int64): return int(o)  
    raise TypeError

with open(os.path.join(args.out_folder, db_name), "w") as f:
    f.write(json.dumps(db, indent=2, default=convert))




















