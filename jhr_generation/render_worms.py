#!/usr/bin/env python

import sys
import os


# import npose_util

import hashlib

from importlib import reload
import npose_util
reload(npose_util)
from npose_util import *
import npose_util as nu

import argparse

# import motif_stuff
import motif_stuff2



import helical_worms
reload(helical_worms)
import helical_worms
from helical_worms import *


parser = argparse.ArgumentParser()
parser.add_argument("helix_list", type=str )
parser.add_argument("turn_list", type=str )
parser.add_argument("render_list", type=str )
parser.add_argument("-allowable_db", type=str)
parser.add_argument("-silent", action="store_true")
parser.add_argument("-no_score", action="store_true")
parser.add_argument("-repeats", type=int, default=4)
parser.add_argument("-add_n_repeats", type=int, default=0)


args = parser.parse_args(sys.argv[1:])

db_helix, db_turn, allowable_db = load_dbs( args.helix_list, args.turn_list, args.allowable_db )




to_evaluate = []
with open(args.render_list) as f:
    for line in f:
        line = line.strip()
        if (len(line) == 0):
            continue
        try:
            to_evaluate.append(eval(line))
        except:
            pass

filv_fl = motif_stuff2.fmh.get_search_mask("FILV", "FL")

if ( args.silent ):
    open_silent = open("out.silent", "w")


null_care_mask = np.zeros((3000), np.bool)
null_care_mask.fill(True)

work_care_mask = np.zeros((3000), np.bool)

first = True
for segments in to_evaluate:


    use_repeats = args.repeats

    if ( args.add_n_repeats ):
        assert( args.repeats > 0 )
        assert( len(segments) % args.repeats == 1 )
        repeat_size = len(segments)//args.repeats
        repeat_region = segments[-repeat_size:]

        for i in range(args.add_n_repeats):
            segments += repeat_region

        use_repeats += args.add_n_repeats

    npose, parts, out_name = result_to_pdb(segments, db_helix, db_turn, dump_pdb=False)

    m = hashlib.md5()
    m.update("_".join(out_name).encode('utf-8'))
    hash_name = m.hexdigest()

    if ( not args.no_score ):

        care_mask = null_care_mask[:nu.nsize(npose)]
        if ( use_repeats > 0 ):
            lb = 1
            ub = use_repeats - 1

            assert( (nu.nsize(npose) - 1) % use_repeats == 0)

            asu_size = (nu.nsize(npose)-1) // use_repeats

            work_care_mask.fill(False)
            work_care_mask[asu_size*lb:asu_size*ub] = True

            care_mask = work_care_mask[:nu.nsize(npose)]


            asu1 = npose[1*nu.R:(1+asu_size)*nu.R]
            asu2 = npose[(1+asu_size)*nu.R:(1+asu_size*2)*nu.R]

            tpose1 = nu.tpose_from_npose(asu1)
            tpose2 = nu.tpose_from_npose(asu2)

            com1 = nu.center_of_mass(asu1)
            com2 = nu.center_of_mass(asu2)

            xform1 = tpose1[0]
            xform2 = tpose2[0]

            xform1[:3,3] = com1[:3]
            xform2[:3,3] = com2[:3]

            angle, rise, r = get_helical_params(xform1, xform2)#, hash_name)

            # if ( r < 8 ):
            #     continue
            # if ( np.abs(rise) > 5 ):
            #     continue
            # if ( angle < 30 ):
            #     continue
            # if ( angle)

            # print("Angle: %.1f rise: %.1f r: %.1f"%(angle, rise, r))

        ca_cb = nu.extract_atoms( npose, [nu.CA, nu.CB]).reshape(-1, 2, 4)[...,:3]

        is_helix = get_is_helix(segments, nu.nsize(npose))
        is_segment = get_is_segment(segments, nu.nsize(npose))

        # my_is_helix = nu.npose_dssp_helix(npose)

        # import IPython
        # IPython.embed()

        too_close_misses = get_cb_cb_too_close(ca_cb[:,1], is_helix, 4.5*4.5, care_mask, 1000)

        actual_clashes = get_cb_cb_too_close(ca_cb[:,1], care_mask, 3.5*3.5, care_mask, 1000)

        absurd_clashes = get_cb_cb_too_close(ca_cb[:,1], care_mask, 2.5*2.5, care_mask, 1000)


        # if ( too_close_misses > 12 ):
        #     # fail[0] = True
        #     continue


        #if ( actual_clashes > 3 ):
        #    # fail[0] = True
        #    continue

        #if ( absurd_clashes > 0 ):
        #    # fail[0] = True
        #    continue


        sc_neighbors, percent_core, percent_surf, median_neigh, neighs = get_avg_sc_neighbors( ca_cb, care_mask )

        # if ( percent_core < 0.32 ):
        #     # fail[1] = True
        #     continue

        is_core_boundary = neighs > 2
        is_core = neighs > 5.2


        context = extract_CA(npose)
        pts = context[:,:3]

        max_dia = 0
        for pt in pts:
            max_dia = max(max_dia, np.max(np.linalg.norm( pt - pts, axis=1 ) ))

        com = np.sum( pts, axis=0 ) / len(pts)

        dist_from_com = np.linalg.norm( pts - com, axis=1)

        wacky_dia = np.max(dist_from_com)*2
        rg = np.sum(dist_from_com)/len(pts)


        dssp = npose_dssp(segments, 4)
        # hits, _ = motif_stuff.motif_score_npose( npose, dssp )

        hits2, froms, tos, _ = motif_stuff2.motif_score_npose( npose )




        sc_neighbors_mask, percent_core_mask, percent_surf_mask, median_neigh_mask, neighs_mask = get_avg_sc_neighbors_mask( ca_cb,  is_helix )

        window = worst_motif_hits_in_window(froms, tos, len(context), 13)
        window = worst_core_in_window(segments, neighs_mask, len(context), 6)

        who_else5, position5, _, _, _ = worst_second_worst_motif_interaction(is_segment, is_helix, froms, tos, len(context), is_core_boundary, care_mask, 5)
        who_else7, position7, _, _, _ = worst_second_worst_motif_interaction(is_segment, is_helix, froms, tos, len(context), is_core_boundary, care_mask, 7)
        who_else9, position9, _, _, _ = worst_second_worst_motif_interaction(is_segment, is_helix, froms, tos, len(context), is_core_boundary, care_mask, 9)

        avg5, avg5_2 = avg_second_worst_motif_interaction(is_segment, is_helix, froms, tos, len(context), is_core_boundary, care_mask, 5)
        avg7, avg7_2 = avg_second_worst_motif_interaction(is_segment, is_helix, froms, tos, len(context), is_core_boundary, care_mask, 7)
        avg9, avg9_2 = avg_second_worst_motif_interaction(is_segment, is_helix, froms, tos, len(context), is_core_boundary, care_mask, 9)

        motif_score = (is_core_boundary[froms] & is_core_boundary[tos] & care_mask[froms] & care_mask[tos]).sum()

        # motif_score = (is_core[froms] & is_core[tos]).sum()

        hits_per_res = motif_score / care_mask.sum()


        core_doesnt_have_motif = is_core.copy()

        good_motif_mask = is_core[froms] & is_core[tos]
        good_froms = froms[good_motif_mask]
        good_tos = tos[good_motif_mask]

        core_doesnt_have_motif[good_froms] = False
        core_doesnt_have_motif[good_tos] = False

        num_core_no_motif = core_doesnt_have_motif.sum()


        # all_hits = motif_stuff2.motif_score_npose_all( npose, filv_fl )

        # cb_cb_distances2 = np.zeros((len(context), len(context)), np.float)

        # for i in range(len(ca_cb)):
        #     cb_cb_distances2[i,:] = np.sum( np.square(ca_cb[:,1] - ca_cb[i,1]), axis=-1)


        # if ( hits_per_res < 3 ):
        #     continue



        connected_scores = ss_core( segments, neighs )
        worst_connect = np.min(connected_scores)

        fail = np.zeros(10, np.bool)

        # if ( too_close_misses > 10 ):
        #     fail[0] = True
        #     # continue


        # if ( too_close_misses > 12 ):
        #     fail[0] = True
        #     continue

        if ( percent_core < 0.28 ):
            fail[0] = True
            # continue

        if ( avg5 < 0.75 ):
            fail[1] = True
            # continue

        if ( avg9 < 0.85 ):
            fail[2] = True
            # continue

        if ( avg9_2 < 0.85 ):
            fail[3] = True
            # continue


        # if ( who_else7 < 4 ):
        #     fail[3] = True
        #     continue

        # if ( who_else5 < 1 ):
        #     fail[4] = True
        #     # continue

        #if ( np.any(fail) ):
        #    print(" ".join("X" if x else " " for x in fail))
        #    continue

        # print(segments)
        # continue

        if ( args.repeats > 0 ):

            print("Angle: %.1f rise: %.1f r: %.1f"%(angle, rise, r))


        print(hash_name, "%5.1f %5.2f %5.2f %5.2f %5.2f %5.2f  %5i %5i %5i  %5.1f - %5.2f %5.2f %5.2f  %5.2f %5.2f %5.2f   - %5i %5i %5i"%(
            sc_neighbors, percent_core, percent_surf, median_neigh, max_dia, rg, hits2, hits2, motif_score, hits_per_res,
            avg5, avg7, avg9,  avg5_2, avg7_2, avg9_2, 
            absurd_clashes, actual_clashes, too_close_misses))

    else:
        print(hash_name)


    if ( args.silent ):
        nu.add_to_silent_file_open(npose, hash_name, open_silent, first)
    else:
        dump_npdb(npose, hash_name + ".pdb")

    # break

    first = False



if ( args.silent ):
    open_silent.close()











