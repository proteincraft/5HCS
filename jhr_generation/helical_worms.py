#!/usr/bin/env python


import copy

import os
import sys
import argparse
import math
import json
import hashlib
from numba import njit
import random

# import npose

from importlib import reload
import npose_util
reload(npose_util)
from npose_util import *
import npose_util as nu

import motif_stuff2

video = False
video_num = 0


the_log = []
do_log = False
show_log = False
def log(message):
    global the_log
    if (show_log):
        print(message)
    the_log.append(message)


wrap_it_up = False
fruitless_count = 0


import functools

og_print = print
print = functools.partial(print, flush=True)


class DBBase:
    def __init__(self, npose, tag):
        self.tag = tag
        self.npose = npose
        self.tpose = tpose_from_npose(npose)
        self.itpose = itpose_from_tpose(self.tpose)
        # self.CAs = extract_CA(npose)
        self.CAs = extract_atoms(npose, [CB])
        self.size = nsize(npose)

        self.npose_ca_cb = nu.extract_atoms(npose, [nu.CA, nu.CB])

        self.dists_matrix = []
        for i in range(self.size):
            pt = self.CAs[i]
            row = np.linalg.norm( self.CAs - pt, axis=1)
            self.dists_matrix.append(row)

class DBHelix(DBBase):
    def __init__(self, npose, tag, allowed_db):
        super(DBHelix, self).__init__(npose, tag)

        self.allowed_turns_by_pos = []
        for i in range(self.size):
            self.allowed_turns_by_pos.append([])

        for turn_tag in allowed_db:
            post = allowed_db[turn_tag]['pre']
            if ( tag not in post):
                continue
            positions = post[tag]
            for pos in positions:
                self.allowed_turns_by_pos[pos].append(turn_tag)

class DBTurn(DBBase):
    def __init__(self, npose, tag, allowed_db):
        super(DBTurn, self).__init__(npose, tag)

        self.allowed_post_helices = allowed_db[tag]['post']

class DBOrigin(DBBase):
    def __init__(self, npose, tag, db_helix, options):
        super(DBOrigin, self).__init__(npose, tag)

        helix_spare_res = options['helix_spare_res']
        min_helix_len = options['min_helix_len']

        self.allowed_post_helices = {}
        for helix_tag in db_helix:
            low = helix_spare_res
            # where high is 1 after last res
            high = db_helix[helix_tag].size - min_helix_len - helix_spare_res
            self.allowed_post_helices[helix_tag] = list(range(low, high))


class SpareData:
    def __init__(self, dia_pts, num_helix_clashes):
        self.dia_pts = dia_pts
        self.num_helix_clashes = num_helix_clashes

    def clone(self):
        return SpareData(list(self.dia_pts), self.num_helix_clashes)

# db_helix = dict of helices -> entry (npose, tpose, itpose, dists_matrix, allowed_turns_by_pos )
# db_turn = dict of turns -> entry (npose, tpose, itpose, dists_matrix, allowed_post_helices)
# 


# worm segments
# pieces from start to finish
# ( tag, entry_res, exit_res )


null_care_mask = np.zeros((1000), bool)
null_care_mask.fill(True)

work_care_mask = np.zeros((1000), bool)


def add_a_turn(context, splice_xform, worm_segments, worm_length, spare_data, db_helix, db_turn, options, results ):
    if (wrap_it_up):
        return 1

    is_helix = get_is_helix( worm_segments, len(context) )
    clashes = get_cb_cb_too_close(context, is_helix, 4.5*4.5, null_care_mask[:len(context)], options['max_actual_helix_clashes'])

    if ( clashes > options['max_actual_helix_clashes'] ):
        return 0

    clashes = get_cb_cb_too_close(context, is_helix, 3.5*3.5, null_care_mask[:len(context)], options['true_clashes'])

    if ( clashes > options['true_clashes'] ):
        return 0

    clashes = get_cb_cb_too_close(context, is_helix, 2.5*2.5, null_care_mask[:len(context)], options['absurd_clashes'])

    if ( clashes > options['absurd_clashes'] ):
        return 0


    global fruitless_count
    fruitless_count += 1
    if ( fruitless_count >= options['fruitless_rollback']):
        if (do_log):
            log("Fruitless rollback!!!! " + str(worm_segments))
        rollback = len(worm_segments) -1  # extra -1 because we're rolling back this layer
        rollback  = options['max_segments'] / 2
        fruitless_count = 0
        return rollback

    splice_pt = splice_xform[:,-1]

    # here my refers to the most recently added helix
    my_segment = worm_segments[-1]
    my_tag = my_segment[0]
    my_db = db_helix[my_tag]


    context_by_dist, context_dist_limits = prepare_context_by_dist_and_limits(context, splice_pt,
                                                options['helix_max_dist'])

    max_turn_clashes = options['max_turn_clashes']
    clash_dist = options['clash_dist']
    extra_res = options['turn_spare_res']

    dia_pts = context[spare_data.dia_pts]
    max_dia = options['max_diameter']
    max_dia2 = max_dia*max_dia

    allowed_turns = my_db.allowed_turns_by_pos[my_segment[2]]
    turn_indices = np.arange(len(allowed_turns))

    rollback = 0
    dia_fail = 0
    num_clashed = 0
    total = 0

    turn_tries = min( len(turn_indices), options['max_turn_tries'])

    if ( options['randomize'] ):
        turn_indices = random.sample(list(turn_indices), turn_tries)


    for iturn in range(turn_tries):
        turn_tag = allowed_turns[turn_indices[iturn]]
        turn_db = db_turn[turn_tag]

        turn_size = turn_db.size
        turn_CAs = turn_db.CAs
        turn_tpose = turn_db.tpose
        turn_itpose = turn_db.itpose
        turn_dists_matrix = turn_db.dists_matrix

        start = extra_res
        # here end refers to one after the last res
        end = turn_size - extra_res

        turn_xform = splice_xform @ turn_itpose[start]
        xformed_CAs = xform_npose( turn_xform, turn_CAs)
        residue_dists = turn_dists_matrix[start]

        if ( video ):
            dump_video_frame(list(worm_segments) + [(turn_tag, start, end-1)], db_helix, db_turn)

        if ( np.max( np.sum( np.square(xformed_CAs[end-1,:3] - dia_pts), axis=1 )) > max_dia2 ):
            dia_fail += 1
            continue

        total += 1

        these_forward_dists = residue_dists[start+1:end]
        these_CAs = xformed_CAs[start+1:end]

        clashes = clash_check_points_context(these_CAs, these_forward_dists, context_by_dist, 
                                            context_dist_limits, clash_dist, max_turn_clashes)

        if ( clashes >= max_turn_clashes ):
            num_clashed += 1
            continue

        new_segments = list(worm_segments)
        new_segments.append( (turn_tag, start, end-1) )

        new_context = np.concatenate( (context, these_CAs[:,:3]) )
        new_splice_xform = turn_xform @ turn_tpose[end-1]

        new_worm_length = worm_length + end - start - 1

        new_spare_data = spare_data.clone()
        new_spare_data.dia_pts.append(new_worm_length-1)

        pass_filter, new_options = filter_after_turn(new_context, new_segments, new_worm_length, new_spare_data, options)
        if ( not pass_filter ):
            # filtered += 1
            continue

        if ( not options['turn_end_func'] is None ):

            finished, rollback = options['turn_end_func']( new_context, new_splice_xform, new_segments, new_worm_length, null_care_mask[:len(context)], options, results, db_helix, db_turn )
            if ( rollback > 0):
                break
            if ( finished ):
                # if ( rollback == 0):
                    # rg_fail += 1
                continue


        rollback = add_a_helix(new_context, new_splice_xform, new_segments, new_worm_length, new_spare_data, db_helix, db_turn, options, results )

        if ( rollback > 0):
            break

    if (do_log):
        log("%sT LEVEL%i: Total: %i Clashed: %i Dia_failed: %i"%("  "*len(worm_segments), len(worm_segments), total, num_clashed, dia_fail))
    rollback = max( 0, rollback-1 )
    return rollback

@njit(fastmath=True, cache=True)
def get_good_ends( low_end, ok_ends, end_size, xformed_CAs, dia_pts, max_dia2 ):

    is_any = False
    for iend in range(end_size):

        # Check RG
        # print(these_CAs[-1,:3])
        if ( np.max( np.sum( np.square(xformed_CAs[iend + low_end-1,:3] - dia_pts), axis=1 ) ) > max_dia2 ):
            # filtered += 1
            ok_ends[iend] = False
            continue
        ok_ends[iend] = True
        is_any = True
    return is_any

# @njit(fastmath=True)
# def get_good_ends2( low_end, ok_ends, end_size, xformed_CAs, dia_pts, max_dia2 ):

#     total = len(ok_ends)
#     subset = xformed_CAs[low_end-1:low_end-1+end_size,:3]

#     any_too_far = np.sum( np.square( subset[:,None,:] - dia_pts ), axis=-1 ) > max_dia2

#     reshaped = any_too_far.reshape(-1, len(dia_pts))

#     # is_any = False
#     # for i in range(end_size):
#     #     val = ~np.any(reshaped[i])
#     #     ok_ends[i] = val
#     #     is_any |= val


#     ok_ends[:] = ~np.any( reshaped, axis=-1)


#     return np.any(ok_ends) #is_any


@njit(fastmath=True, cache=True)
def rg_sq( context ):
    size = len(context)
    com = np.sum( context, axis=0 ) / size
    dists_sq = np.sum(np.square( context - com), axis=1 )
    rg_sq = np.sum( dists_sq ) / size
    return rg_sq

def add_a_helix(context, splice_xform, worm_segments, worm_length, spare_data, db_helix, db_turn, options, results ):
    if (wrap_it_up):
        return 1

    is_helix = get_is_helix( worm_segments, len(context) )
    clashes = get_cb_cb_too_close(context, is_helix, 4.5*4.5, null_care_mask[:len(context)], options['max_actual_helix_clashes'])

    if ( clashes > options['max_actual_helix_clashes'] ):
        return 0

    clashes = get_cb_cb_too_close(context, is_helix, 3.5*3.5, null_care_mask[:len(context)], options['true_clashes'])

    if ( clashes > options['true_clashes'] ):
        return 0

    clashes = get_cb_cb_too_close(context, is_helix, 2.5*2.5, null_care_mask[:len(context)], options['absurd_clashes'])

    if ( clashes > options['absurd_clashes'] ):
        return 0

    global fruitless_count
    fruitless_count += 1
    if ( fruitless_count >= options['fruitless_rollback']):
        if (do_log):
            log("Fruitless rollback!!!! " + str(worm_segments))
        rollback = len(worm_segments)-1  # extra -1 because we're rolling back this layer
        rollback  = options['max_segments'] / 2
        fruitless_count = 0
        return rollback


    splice_pt = splice_xform[:,-1]

    # here my refers to the most recently added turn
    my_segment = worm_segments[-1]
    my_tag = my_segment[0]
    my_db = db_turn[my_tag]

    # splice on the test helix to get the unit vector
    test_helix_splice_res = 1
    test_helix_db = db_helix[options['test_helix']]
    last_test_helix_CA = test_helix_db.CAs[-1]
    test_helix_xform = splice_xform @ test_helix_db.tpose[test_helix_splice_res]
    xformed_CA = test_helix_xform @ last_test_helix_CA
    unit_vector = xformed_CA[:3] - splice_pt[:3]


    context_by_dist, context_dist_limits = None, None

    min_helix_len = options['min_helix_len']
    max_helix_len = options['max_helix_len']
    max_total_helix_clashes = options['max_total_helix_clashes']
    clash_dist = options['clash_dist']
    helix_spare_res = options['helix_spare_res']

    remaining_helix_clashes = max_total_helix_clashes - spare_data.num_helix_clashes

    max_rg = options['max_rg']
    max_rg2 = max_rg*max_rg

    # Don't be stupid on the last segment
    if ( len(worm_segments) == options['max_segments'] -1 ):
        min_worm_length = options['min_worm_length']

        # make sure we don't come up short
        min_helix_len = max( min_helix_len, min_worm_length - worm_length )
        # we can't make it long enough
        if ( min_helix_len > max_helix_len ):
            return 0

    dia_pts = context[spare_data.dia_pts]
    max_dia = options['max_diameter']
    max_dia2 = max_dia*max_dia

    allowed_post_helices = my_db.allowed_post_helices
    helix_keys = list(allowed_post_helices.keys())
    helix_indices = np.arange(len(helix_keys))

    num_clashed = 0
    dia_fail = 0
    rg_fail = 0
    total = 0

    rollback = 0
    # ok_ends = np.zeros( options['max_db_helix_size'] + 1 ).astype('int8')
    ok_ends = np.zeros( options['max_db_helix_size'] + 1, bool )#.astype('bool')

    helix_tries = min( len(helix_keys), options['max_helix_tries'])

    if ( options['randomize'] ):
        helix_indices = random.sample(list(helix_indices), helix_tries)

    for ihelix in range(helix_tries):
        helix_tag = helix_keys[helix_indices[ihelix]]
        if (rollback > 0):
            break
        helix_starts = allowed_post_helices[helix_tag]
        helix_db = db_helix[helix_tag]

        helix_size = helix_db.size
        helix_CAs = helix_db.CAs
        helix_tpose = helix_db.tpose
        helix_itpose = helix_db.itpose
        helix_dists_matrix = helix_db.dists_matrix

        num_starts = min( options['max_helix_starts'], len(helix_starts) )

        for start in random.sample(helix_starts, num_starts):
            if (rollback > 0):
                break

            helix_xform = splice_xform @ helix_itpose[start]
            xformed_CAs = xform_npose( helix_xform, helix_CAs)
            residue_dists = helix_dists_matrix[start]

            # here end refers to one after the last res
            # +1 because the start residue gets consumed
            low_end = start + min_helix_len + 1
            high_end = min( start + max_helix_len + 1, helix_size - helix_spare_res )

            if ( high_end <= low_end ):
                continue

## This replaces #####################
            end_size = high_end+1-low_end
            # ok_ends = np.zeros(high_end+1-low_end).astype(bool)

            # is_any = get_good_ends( low_end, ok_ends, end_size, xformed_CAs, dia_pts, max_dia2 )
            is_any = get_good_ends( low_end, ok_ends, end_size, xformed_CAs, dia_pts, max_dia2 )
            # print(ok_ends)
            # print(ok_ends2)
            # assert( is_any == is_any2 )
            # for i in range(end_size):
            #     assert(bool(ok_ends[i]) == bool(ok_ends2[i]))
            if ( not is_any ):
                dia_fail += end_size
                total += end_size
                if (do_log):
                    log("Total dia fail!")
                rollback = 1 # If this happens, quit early because nothing else is going to work
                return rollback         
                # continue

            num_ends = min( options['max_helix_ends'], end_size )

            for iend in random.sample(list(range(end_size)), num_ends):
                total += 1
                if ( not ok_ends[iend] ):
                    dia_fail += 1
                    continue
                end = iend + low_end

### This ############################
            # for end in range(low_end, high_end+1):
            #     total += 1

            #     # Check RG
            #     # print(these_CAs[-1,:3])
            #     if ( np.max( np.sum( np.square(xformed_CAs[end-1,:3] - dia_pts), axis=1 )) > max_dia2 ):
            #         filtered += 1
            #         continue
####################################

                if ( video ):
                    dump_video_frame(list(worm_segments) + [(helix_tag, start, end-1)], db_helix, db_turn)


                these_CAs = xformed_CAs[1+start:end]
                these_forward_dists = residue_dists[start+1:end]

                # check clashes
                if ( context_by_dist is None ):
                    context_by_dist, context_dist_limits = prepare_context_by_dist_and_limits(context, splice_pt,
                                                                            options['helix_max_dist'], unit_vector)

                clashes = clash_check_points_context(these_CAs, these_forward_dists, context_by_dist, 
                                                    context_dist_limits, clash_dist, remaining_helix_clashes+1, tol=1.5)

                if ( clashes > remaining_helix_clashes ):
                    num_clashed += 1
                    continue


                new_segments = list(worm_segments)
                new_segments.append( (helix_tag, start, end-1) )

                new_context = np.concatenate( (context, these_CAs[:,:3]) )
                new_splice_xform = helix_xform @ helix_tpose[end-1]

                new_worm_length = worm_length + end - start - 1

                new_spare_data = spare_data.clone()
                new_spare_data.num_helix_clashes += clashes
                new_spare_data.dia_pts.append(new_worm_length-1)

                rg2 = rg_sq( new_context )
                if ( rg2 > max_rg2 * 1.2 ):
                    return 0
                if ( rg2 > max_rg2 ):
                    rg_fail += 1
                    break

                pass_filter, new_options = filter_after_helix(new_context, new_segments, new_worm_length, new_spare_data, options)
                if ( not pass_filter ):
                    # filtered += 1
                    continue

                if ( not options['helix_end_func'] is None ):

                    finished, rollback = options['helix_end_func']( new_context, new_splice_xform, new_segments, new_worm_length, null_care_mask[:len(context)], options, results, db_helix, db_turn )
                    if ( rollback > 0):
                        break
                    if ( finished ):
                        if ( rollback == 0):
                            rg_fail += 1
                        continue

                rollback = add_a_turn(new_context, new_splice_xform, new_segments, new_worm_length, new_spare_data, db_helix, db_turn, options, results )
                if ( rollback > 0):
                    break
                

    if (do_log):
        log("%sH LEVEL%i: Total: %i Clashed: %i Dia_fail: %i Rg_fail: %i"%("  "*len(worm_segments), len(worm_segments), total, num_clashed, dia_fail, rg_fail))

    rollback = max(0, rollback - 1)
    return rollback




@njit(fastmath=True, cache=True)
def get_cb_cb_too_close(cbs, is_helix, close2, care_mask, limit):
    clashes = 0
    for i in range(len(cbs)):
        if ( not is_helix[i] ):
            continue
        x = cbs[i, 0]
        y = cbs[i, 1]
        z = cbs[i, 2]
        for j in range(i+1, len(cbs)):
            if ( not is_helix[j] ):
                continue

            if ( not care_mask[i] and not care_mask[j] ):
                continue
            
            dx = x - cbs[j, 0]
            dy = y - cbs[j, 1]
            dz = z - cbs[j, 2]

            dist2 = dx*dx + dy*dy + dz*dz

            if ( dist2 < close2 ):
                clashes += 1
                if ( clashes > limit ):
                    return clashes
    return clashes


angle_rise_r_mask = np.array(
    [
    [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
    [[1, 1, 1], [1, 1, 0], [0, 0, 0]],
    [[1, 1, 0], [0, 0, 0], [0, 1, 0]]
    ]
)

def jhr_finish( context, splice_xform, worm_segments, worm_length, care_mask, options, results, db_helix, db_turn ):
    if ( worm_length > options['max_worm_length']):
        return True, 0
    if ( worm_length < options['min_worm_length'] ):
        return False, 0
    if ( len(worm_segments) > options['max_segments']):
        return True, 0



    xform1 = np.identity(4)
    xform2 = splice_xform

    # since xform1 is identity we don't need to do this
    two_from_one = xform2 #@ np.linalg.inv(xform1)

    com1 = np.ones(4, float)
    com1[:3] = context[1:].mean(axis=-2)

    com2 = two_from_one @ com1

    xform1[:3,3] = com1[:3]
    xform2[:3,3] = com2[:3]

    # angle, rise, r = get_helical_params(xform1, xform2, x1_no_rot=False)

    angle, rise, r = get_helical_params_numba(xform1, xform2, x1_no_rot=True)

    arise = abs(rise)

    min_angle, max_angle = options['helical_angle_range']
    min_rise, max_rise = options['helical_rise_range']
    min_radius, max_radius = options['helical_radius_range']

    if ( angle < min_angle or angle > max_angle ):
        return True, 0
    if ( arise < min_rise or arise > max_rise ):
        return True, 0
    if ( r < min_radius or r > max_radius ):
        return True, 0


    # angle_bin_ = np.searchsorted([0, 30, 60], angle) - 1
    # rise_bin_ = np.searchsorted([0, 2.5, 5], np.abs(rise)) - 1
    # r_bin_ = np.searchsorted([0, 12, 20], r) - 1

    if ( angle < 30 ):
        angle_bin = 0
    elif ( angle < 60 ):
        angle_bin = 1
    else:
        angle_bin = 2

    if ( arise < 2.5 ):
        rise_bin = 0
    elif ( arise < 5 ):
        rise_bin = 1
    else:
        rise_bin = 2

    if ( r < 12 ):
        r_bin = 0
    elif ( r < 20 ):
        r_bin = 1
    else:
        r_bin = 2

    # assert(angle_bin == angle_bin_)
    # assert(rise_bin == rise_bin_)
    # assert(r_bin == r_bin_)



    if ( not angle_rise_r_mask[angle_bin, rise_bin, r_bin] ):
        return True, 0




    worm_segments = list(worm_segments)
    our_segments = worm_segments[1:]
    for i in range(options['repeats']-1):
        worm_segments += our_segments

    asu_size = len(context) - 1

    work_care_mask.fill(False)
    work_care_mask[asu_size:asu_size*3] = True

    # worm, parts, out_name = result_to_pdb( worm_segments, db_helix, db_turn, dump_pdb=False, trim_excess=True)
    # angle2, rise2, r2 = get_helical_params_helper(worm, options['repeats'])
    # print(angle, angle2, rise, rise2, r, r2)

    # import IPython
    # IPython.embed()

    # context = nu.extract_CA( worm )

    care_mask = work_care_mask[:asu_size*options['repeats']+1]

    if ( video ):
        dump_video_frame(worm_segments, db_helix, db_turn)

    return check_finished(None, splice_xform, worm_segments, worm_length, care_mask, options, results, db_helix, db_turn, (angle, rise, r))
    

def check_finished( context, splice_xform, worm_segments, worm_length, care_mask, options, results,  db_helix, db_turn, temp ):
    if ( worm_length >= options['min_worm_length'] ):
        if ( worm_length > options['max_worm_length']):
            return True, 0

        # rg2 = rg_sq( context )
        # max_rg = options['max_rg']
        # max_rg2 = max_rg*max_rg
        # if ( rg2 > max_rg2 ):
        #     return True, 0


        worm_ca_cb = result_to_ca_cb_fast( worm_segments, db_helix, db_turn )

        # print("ok")

        # ca_cb = nu.extract_atoms( worm, [nu.CA, nu.CB]).reshape(-1, 2, 4)[...,:3]
        ca_cb = worm_ca_cb.reshape(-1, 2, 4)[...,:3]

        if ( context is None ):
            context = ca_cb[:,0]

        is_helix = get_is_helix( worm_segments, len(context) )
        clashes = get_cb_cb_too_close(ca_cb[:,1], is_helix, options['clash_dist']**2, care_mask, options['max_actual_helix_clashes'])

        if ( clashes > options['max_actual_helix_clashes'] ):
            return True, 0


        true_clashes = get_cb_cb_too_close(ca_cb[:,1], care_mask, 3.5*3.5, care_mask, options['true_clashes'])

        if ( true_clashes > options['true_clashes'] ):
            return True, 0

        absurd_clashes = get_cb_cb_too_close(ca_cb[:,1], care_mask, 2.5*2.5, care_mask, options['absurd_clashes'])

        if ( absurd_clashes > options['absurd_clashes'] ):
            return True, 0


        sc_neighbors, percent_core, percent_surf, median_neigh, neighs = get_avg_sc_neighbors( ca_cb, care_mask )

        # print("here")

        # if ( sc_neighbors < options['min_sc_neigh'] ):
        #     return True, 0

        global fruitless_count

        if ( median_neigh < options['median_sc_neigh'] * 0.75 ):
            # print("T")
            fruitless_count += 1
            return True, 0

        if ( percent_core < options['perc_core_scn'] * 0.75 ):
            # print("T")
            fruitless_count += 1
            return True, 0

        if ( median_neigh < options['median_sc_neigh'] * 0.9 ):
            # print("T")
            fruitless_count += 0
            return True, 0

        if ( percent_core < options['perc_core_scn'] * 0.9 ):
            # print("T")
            fruitless_count += 0
            return True, 0


        if ( median_neigh < options['median_sc_neigh'] ):
            return True, 0

        if ( percent_core < options['perc_core_scn'] ):
            return True, 0


        worm, parts, out_name = result_to_pdb( worm_segments, db_helix, db_turn, dump_pdb=False, trim_excess=True)

        is_core = neighs > 5.2
        is_core_boundary = neighs > 2

        hits, froms, tos, misses = motif_stuff2.motif_score_npose( worm, care_mask, is_core_boundary )




        motif_hash_core = (is_core_boundary[froms] & is_core_boundary[tos]).sum()
        assert( motif_hash_core == len(froms) )

        motif_hash_core_per_res = motif_hash_core / care_mask.sum()

        # motif_hash_per_res = hits / len(context)

        if ( motif_hash_core_per_res < options['motif_hash_core_per_res'] ):
            return True, 0


        is_helix = get_is_helix(worm_segments, len(context))
        is_segment = get_is_segment(worm_segments, len(context))

        worst_others9, position, avg9, avg9_2, _ = worst_second_worst_motif_interaction(is_segment, is_helix, froms, tos, len(context), is_core_boundary, care_mask, 9)
        worst_others7, position, avg7, avg7_2, _ = worst_second_worst_motif_interaction(is_segment, is_helix, froms, tos, len(context), is_core_boundary, care_mask, 7)
        worst_others5, position, avg5, avg5_2, _ = worst_second_worst_motif_interaction(is_segment, is_helix, froms, tos, len(context), is_core_boundary, care_mask, 5)

        if ( worst_others5 < options['min_worst_others5'] ):
            return True, 0

        if ( worst_others7 < options['min_worst_others7'] ):
            return True, 0

        if ( worst_others9 < options['min_worst_others9'] ):
            return True, 0

        if ( avg5 < options['avg_worst_others5'] ):
            return True, 0

        if ( avg7 < options['avg_worst_others7'] ):
            return True, 0

        if ( avg9 < options['avg_worst_others9'] ):
            return True, 0

        if ( avg5_2 < options['avg_worst_others5_2'] ):
            return True, 0

        if ( avg7_2 < options['avg_worst_others7_2'] ):
            return True, 0

        if ( avg9_2 < options['avg_worst_others9_2'] ):
            return True, 0


        # m = hashlib.md5()
        # m.update("_".join(out_name).encode('utf-8'))
        # hash_name = m.hexdigest()
        # print("%s %6.2f %6.2f %6.2f %6.2f %6.2f %6.2f %6.2f "%(hash_name, clashes, true_clashes, percent_core, motif_hash_core, worst_others5, worst_others7, worst_others9 ))

        extra_print = ""
        if ( options['repeats'] > 0 ):
            angle, rise, r = get_helical_params_helper(worm, options['repeats'])#, hash_name)

            extra_print = "# %.1f %.1f %.1f"%(angle, rise, r)# + " " + str(temp)

        fruitless_count = 0
        results.append( worm_segments )
        print(worm_segments, extra_print)
        # if ( len(results) % 10000 == 0 ):
            # f = open("results.txt", "w")
            # for result in results:
            #     f.write(str(result) + "\n")
            # f.close()
        # print("Success")
        if (do_log):
            log("%sWORM: Success: length: %i"%("  "*len(worm_segments), worm_length))
        if ( len(results) > options['max_results']):
            global wrap_it_up
            wrap_it_up = True
        rollback = 1
        if ( options['randomize'] ):
            if ( random.random() < options['chance_of_rollback'] ):
                rollback = random.randint(2, len(worm_segments)-1) # This lets it at most go back to the first helix
                if (do_log):
                    log("Rollback!!! %i"%rollback)
        return True, rollback
    if ( len(worm_segments) >= options['max_segments']):
        if (do_log):
            log("%sWORM: Fail: length: %i"%("  "*len(worm_segments), worm_length))
        return True, 0
    return False, 0

def common_filter(context, worm_segments, worm_length, spare_data, options):
    # diameter
    dia_pts = context[spare_data.dia_pts]
    max_dia = options['max_diameter']
    for ipt, pt in enumerate(dia_pts):
        max_dist = np.max(np.linalg.norm(pt - dia_pts, axis=1))
        if ( max_dist > max_dia):
            return False
    return True

def filter_after_helix(context, worm_segments, worm_length, spare_data, options):
    # if ( not common_filter(context, worm_segments, worm_length, spare_data, options)):
    #     assert(False)
    #     return False, options
    return True, options

def filter_after_turn(context, worm_segments, worm_length, spare_data, options):
    # if ( not common_filter(context, worm_segments, worm_length, spare_data, options)):
    #     assert(False)
    #     return False, options
    return True, options


def slow_evaluator( worm_segments, db_helix, db_turn, dump_pdb=False):
    npose, parts, name = result_to_pdb(worm_segments, db_helix, db_turn, dump_pdb=dump_pdb)
    context = extract_CA(npose)
    pts = context[:,:3]

    max_dia = 0
    for pt in pts:
        max_dia = max(max_dia, np.max(np.linalg.norm( pt - pts, axis=1 ) ))

    com = np.sum( pts, axis=0 ) / len(pts)

    dist_from_com = np.linalg.norm( pts - com, axis=1)

    wacky_dia = np.max(dist_from_com)*2
    rg = np.sum(dist_from_com)/len(pts)
    moment = np.sum(np.square(dist_from_com))/len(pts)
    third_moment = np.sum(np.power(dist_from_com, 3))/len(pts)

    out_dict = {}
    out_dict['dia'] = max_dia
    out_dict['wacky_dia'] = wacky_dia
    out_dict['rg'] = rg
    out_dict['moment'] = moment
    out_dict['third_moment'] = moment
    out_dict['name'] = name

    return out_dict


def dump_video_frame(segments, db_helix, db_turn ):
    global video_num

    name = "video.pdb"

    if ( video_num == 0 ):
        if ( os.path.exists(name) ):
            os.remove(name)

    npose, parts, _ = result_to_pdb(segments, db_helix, db_turn, False)

    with open(name, "a") as f:
        f.write("MODEL %5i\n"%video_num)
        nu.dump_npdb(npose, "", out_file=f)
        f.write("ENDMDL\n")


    video_num += 1
    if ( video_num == 2000 ):
        sys.exit()

def dump_these_results(segments, db_helix, db_turn):
    for segment in segments:
        result_to_pdb(segment, db_helix, db_turn)

def get_name(segments):
    name = []
    for tag, start, end in segments:
        name.append("%s-%i-%i"%(tag, start, end))
    return "_".join(name)

ca_cb_scratch = np.zeros((1000, 4), float)
t_scratch = np.zeros((100, 4, 4), float)
it_scratch = np.zeros((100, 4, 4), float)
lb_scratch = np.zeros(100, int)
ub_scratch = np.zeros(100, int)

def result_to_ca_cb_fast( segments, db_helix, db_turn ):

    dbs = [db_turn, db_helix]

    cur = 0
    for idx, segment in enumerate(segments):
        if (idx == 0):
            continue
        tag, start, end = segment

        db = dbs[idx%2][tag]

        npose = db.npose_ca_cb

        real_start = start + 1
        if ( idx == 1 ):
            real_start -=  1

        npose = npose[real_start*2:(end+1)*2]

        ca_cb_scratch[cur:cur+len(npose)] = npose


        it_scratch[idx-1] = db.itpose[start]
        t_scratch[idx-1] = db.tpose[end]

        lb_scratch[idx-1] = cur
        ub_scratch[idx-1] = cur + len(npose)

        cur += len(npose)

    inner_result_to_ca_cb_fast( ca_cb_scratch[:cur], t_scratch[:idx], it_scratch[:idx], lb_scratch[:idx], ub_scratch[:idx] )

    return ca_cb_scratch[:cur]

@njit(fastmath=True, cache=True)
def inner_result_to_ca_cb_fast( ca_cb, ts, its, lbs, ubs ):

    splice_xform = np.identity(4)

    ca_cb = ca_cb.reshape(-1, 4, 1)

    for i in range(len(ts)):
        t = ts[i]
        it = its[i]
        lb = lbs[i]
        ub = ubs[i]

        xform = splice_xform @ it

        for j in range(lb, ub):
            ca_cb[j] = xform @ ca_cb[j]

        # ca_cb[lb:ub] = xform @ ca_cb[lb:ub]

        # ca_cb[lb:ub] = (xform @ ca_cb[lb:ub,:,None]).reshape(-1, 4)

        splice_xform = xform @ t

def check_result_to_pdb(segments, db_helix, db_turn, dump_pdb=True, trim_excess=True, CA_CB_only=False):
    return result_to_pdb(segments, db_helix, db_turn, dump_pdb, trim_excess, CA_CB_only)

def result_to_pdb(segments, db_helix, db_turn, dump_pdb=True, trim_excess=True, CA_CB_only=False):

    dbs = [db_turn, db_helix]

    name = []

    worm = None
    parts = []

    splice_xform = np.identity(4)
    for idx, segment in enumerate(segments):
        if (idx == 0):
            continue
        tag, start, end = segment

        name.append("%s-%i-%i"%(tag, start, end))

        db = dbs[idx%2][tag]

        use_R = nu.R

        npose = db.npose

        if ( CA_CB_only ):
            use_R = 2
            npose = db.npose_ca_cb

        if ( trim_excess ):
            real_start = start + 1
            if ( idx == 1 ):
                real_start -=  1
            npose = npose[real_start*use_R:(end+1)*use_R]

        # if ( CA_CB_only ):
        #     npose = nu.extract_atoms(npose, [nu.CA, nu.CB])

        xform = splice_xform @ db.itpose[start]
        npose = xform_npose( xform, npose )


        parts.append(npose)
        if (worm is None):
            worm = npose
        else:
            worm = np.concatenate((worm, npose))

        splice_xform = xform @ db.tpose[end]

    # m = hashlib.md5()
    # m.update("_".join(name).encode('utf-8'))
    # out_name = m.hexdigest()

    out_name = "_".join(name)


    if (dump_pdb):
        dump_npdb(worm, out_name + ".pdb")

    return worm, parts, out_name


# distance, CA->CB dot CA->CB, CA->CB dot distance, CA->N dot CA->N
@njit(fastmath=True, cache=True)
def get_dot_representation(npose, out_data, offset):

    size = int(len(npose)/R)

    for i in range(size):
        for j in range(size):
            res1 = npose[R*i:R*(i+1),:3]
            res2 = npose[R*j:R*(j+1),:3]

            distance_vect = res2[CA] - res1[CA]
            CA_CB1 = res1[CB] - res1[CA]
            CA_CB2 = res2[CB] - res2[CA]
            CA_N1 = res1[N] - res1[CA]
            CA_N2 = res2[N] - res2[CA]

            distance = np.linalg.norm(distance_vect)
            mag_CA_CB1 = np.linalg.norm(CA_CB1)
            mag_CA_CB2 = np.linalg.norm(CA_CB2)
            mag_CA_N1 = np.linalg.norm(CA_N1)
            mag_CA_N2 = np.linalg.norm(CA_N2)

            out_data[offset,i,j,0] = distance

            out_data[offset,i,j,1] = 0 if (mag_CA_CB1 == 0 or mag_CA_CB2 == 0) else np.dot(CA_CB1, CA_CB2) / mag_CA_CB1 / mag_CA_CB2

            out_data[offset,i,j,2] = 0 if (mag_CA_CB1 == 0 or distance == 0) else np.dot(CA_CB1, distance_vect) / mag_CA_CB1 / distance

            out_data[offset,i,j,3] = 0 if (mag_CA_N1 == 0 or mag_CA_N2 == 0) else np.dot(CA_N1, CA_N2) / mag_CA_N1 / mag_CA_N2


@njit(fastmath=True, cache=True)
def get_slow_clash(npose):
    any_clash = False
    clash_dist2 = 1.8*1.8
    for iatom in range(len(npose)):
        atom = npose[iatom]
        dists2 =  np.sum( np.square(atom - npose), axis=1) 

        indices = np.where( dists2 < clash_dist2 )[0]
        if ( np.any( np.abs(indices - iatom) > R*6) ):
            return True
    return False

@njit(fastmath=True, cache=True)
def get_avg_sc_neighbors(ca_cb, care_mask):
    conevect = (ca_cb[:,1] - ca_cb[:,0] )
    # conevect_lens = np.sqrt( np.sum( np.square( conevect ), axis=-1 ) )

    # for i in range(len(conevect)):
    #     conevect[i] /= conevect_lens[i]

    conevect /= 1.5

    maxx = 11.3
    max2 = maxx*maxx

    neighs = np.zeros(len(ca_cb))

    core = 0
    surf = 0

    summ = 0
    for i in range(len(ca_cb)):

        if ( not care_mask[i] ):
            continue

        vect = ca_cb[:,0] - ca_cb[i,1]
        vect_length2 = np.sum( np.square( vect ), axis=-1 )

        vect = vect[(vect_length2 < max2) & (vect_length2 > 4)]
        vect_length2 = vect_length2[(vect_length2 < max2) & (vect_length2 > 4)]

        vect_length = np.sqrt(vect_length2)

        # x1 = vect_length2
        # x2 = vect_length2 * vect_length2
        # x3 = vect_length2 * x2
        # x4 = x2 * x2

        # vect_length = -7.41077e-8 * x4 + 2.309905e-5 * x3 - 0.0027321 * x2 + 0.2036471 * x1 + 1.307829

        for j in range(len(vect)):
            vect[j] /= vect_length[j]
            # print(vect_length2[j], np.linalg.norm(vect[j]))
        # vect_normed = vect / vect_length

        # dist_term = 1 / ( 1 + np.exp( vect_length - 9  ) )

        # linear fit to the above sigmoid
        dist_term = np.zeros(len(vect))
        for j in range(len(vect)):
            if ( vect_length[j] < 7 ):
                dist_term[j] = 1
            elif (vect_length[j] > maxx ):
                dist_term[j] = 0
            else:
                dist_term[j] = -0.23 * vect_length[j] + 2.6


        angle_term = ( np.dot(vect, conevect[i] ) + 0.5 ) / 1.5

        for j in range(len(angle_term)):
            if ( angle_term[j] < 0 ):
                angle_term[j] = 0

        sc_neigh = np.sum( dist_term * np.square( angle_term ) )

        # if ( sc_neigh > 5.2 ):
        #     core += 1

        # if ( sc_neigh < 2 ):
        #     surf += 1
        # print(sc_neigh)

        neighs[i] = sc_neigh

        summ += sc_neigh

    care_sum = care_mask.sum()

    percent_core = (neighs[care_mask] > 5.2).sum() / care_sum
    percent_surf = (neighs[care_mask] < 2).sum() / care_sum
    avg_scn = np.mean(neighs[care_mask])
    median_scn = np.median(neighs[care_mask])

    return avg_scn, percent_core, percent_surf, median_scn, neighs


@njit(fastmath=True, cache=True)
def get_avg_sc_neighbors_mask(ca_cb, mask):
    conevect = (ca_cb[:,1] - ca_cb[:,0] )
    # conevect_lens = np.sqrt( np.sum( np.square( conevect ), axis=-1 ) )

    # for i in range(len(conevect)):
    #     conevect[i] /= conevect_lens[i]

    conevect /= 1.5

    maxx = 11.3
    max2 = maxx*maxx

    neighs = np.zeros(len(ca_cb))

    core = 0
    surf = 0

    summ = 0
    for i in range(len(ca_cb)):

        vect = ca_cb[:,0] - ca_cb[i,1]
        vect_length2 = np.sum( np.square( vect ), axis=-1 )

        vect = vect[(vect_length2 < max2) & (vect_length2 > 4) & mask]
        vect_length2 = vect_length2[(vect_length2 < max2) & (vect_length2 > 4) & mask]

        vect_length = np.sqrt(vect_length2)

        # x1 = vect_length2
        # x2 = vect_length2 * vect_length2
        # x3 = vect_length2 * x2
        # x4 = x2 * x2

        # vect_length = -7.41077e-8 * x4 + 2.309905e-5 * x3 - 0.0027321 * x2 + 0.2036471 * x1 + 1.307829

        for j in range(len(vect)):
            vect[j] /= vect_length[j]
            # print(vect_length2[j], np.linalg.norm(vect[j]))
        # vect_normed = vect / vect_length

        # dist_term = 1 / ( 1 + np.exp( vect_length - 9  ) )

        # linear fit to the above sigmoid
        dist_term = np.zeros(len(vect))
        for j in range(len(vect)):
            if ( vect_length[j] < 7 ):
                dist_term[j] = 1
            elif (vect_length[j] > maxx ):
                dist_term[j] = 0
            else:
                dist_term[j] = -0.23 * vect_length[j] + 2.6


        angle_term = ( np.dot(vect, conevect[i] ) + 0.5 ) / 1.5

        for j in range(len(angle_term)):
            if ( angle_term[j] < 0 ):
                angle_term[j] = 0

        sc_neigh = np.sum( dist_term * np.square( angle_term ) )

        if ( sc_neigh > 5.2 ):
            core += 1

        if ( sc_neigh < 2 ):
            surf += 1
        # print(sc_neigh)

        neighs[i] = sc_neigh

        summ += sc_neigh

    before_mask = neighs
    neighs = neighs[mask]

    return np.sum(neighs) / len(neighs), np.sum(neighs > 5.2) / len(neighs), np.sum( neighs < 2 ) / len(neighs), np.median(neighs), before_mask

import scipy.spatial.transform

def get_helical_params_helper(npose, repeats):

    assert( (nu.nsize(npose) - 1) % repeats == 0)
    asu_size = (nu.nsize(npose)-1) // repeats

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

    return get_helical_params(xform1, xform2)

tiny_rotation = scipy.spatial.transform.Rotation.from_rotvec(np.array([0.01, 0, 0])).as_matrix()

# if you cache this it segfaults...
@njit(fastmath=False, cache=False)
def rot_vec_from_rot(rot):

    trace = rot[0,0] + rot[1,1] + rot[2,2]

    # this indicates there is no rotation
    if ( trace >= 3 ):
        return 0, np.array([1, 0, 0], np.float_)

    # Finding the axis for angle = pi is really really hard
    # instead just return something approximate
    if ( trace <= -1 ):
        fake = tiny_rotation @ rot
        _, vec = rot_vec_from_rot(fake)
        return np.pi, vec


    theta = math.acos((trace - 1)/2)
    twosin = math.sin(theta)*2

    # theta = np.arccos((np.trace(rot)-1)/2)
    # twosin = np.sin(theta)*2

    vec = np.zeros(3, np.float_)
    vec[0] = (rot[2,1] - rot[1,2]) / twosin
    vec[1] = (rot[0,2] - rot[2,0]) / twosin
    vec[2] = (rot[1,0] - rot[0,1]) / twosin

    return theta, vec

def scipy_version(rot):
    rot_vec = scipy.spatial.transform.Rotation.from_dcm(rot).as_rotvec()

    angle = np.linalg.norm(rot_vec)
    unit = rot_vec / angle 

    return angle, unit

@njit(fastmath=True, cache=True)
def dot_numba(a, b):
    return a[0]*b[0] + a[1]*b[1] + a[2]*b[2]

@njit(fastmath=True, cache=True)
def norm_numba(a):
    return math.sqrt( dot_numba( a, a ) )

@njit(fastmath=True, cache=False)
def get_helical_params_numba(xform1, xform2, x1_no_rot=False):

    if ( x1_no_rot ):
        inv = xform1.copy()
        inv[:3,3] = -xform1[:3,3]
        rt = inv @ xform2
    else:
        rt = np.linalg.inv(xform1) @ xform2

    rot = rt[:3,:3]
    trans = rt[:3,3]

    # angle, unit = scipy_version(rot)
    angle, unit = rot_vec_from_rot(rot)

    rise = dot_numba( unit, trans )

    # r is harder
    # we first get the part of trans that is in the plane where unit is the normal
    #
    # this is the base of an isosceles triangle where the other 2 sides are r
    #
    # then sin(1/2 angle ) * r = 1/2 plane_trans

    trans_on_plane = trans - dot_numba( unit, trans ) * unit

    base = norm_numba(trans_on_plane)

    r = base / 2 / math.sin( angle / 2 )

    return math.degrees(angle), rise, r

def get_helical_params(xform1, xform2, debug="", x1_no_rot=False):

    if ( x1_no_rot ):
        inv = xform1.copy()
        inv[:3,3] = -xform1[:3,3]
        rt = inv @ xform2
    else:
        rt = np.linalg.inv(xform1) @ xform2

    rot = rt[:3,:3]
    trans = rt[:3,3]

    # angle, unit = scipy_version(rot)
    angle, unit = rot_vec_from_rot(rot)


    rise = np.dot( unit, trans )

    # r is harder
    # we first get the part of trans that is in the plane where unit is the normal
    #
    # this is the base of an isosceles triangle where the other 2 sides are r
    #
    # then sin(1/2 angle ) * r = 1/2 plane_trans

    trans_on_plane = trans - np.dot( unit, trans ) * unit

    base = np.linalg.norm(trans_on_plane)

    r = base / 2 / np.sin( angle / 2 )

    if ( len(debug) == 0 ):
        return np.degrees(angle), rise, r

    real_unit = xform1[:3,:3] @ unit
    real_trans_on_plane = xform1[:3,:3] @ trans_on_plane

    midpoint = xform1[:3,3] + real_trans_on_plane / 2
    dist_to_center = np.cos( angle / 2 ) * r

    to_center_unit = np.cross( real_unit, real_trans_on_plane )
    to_center_unit /= np.linalg.norm(to_center_unit)

    center = to_center_unit * dist_to_center + midpoint

    nu.dump_pts([midpoint, center], "%s_midpoint_center.pdb"%debug)
    nu.dump_line(center, real_unit, 10, "%s_helix_vector.pdb"%debug)
    nu.dump_line(midpoint, to_center_unit, r, "%s_to_center.pdb"%debug)
    nu.dump_line(xform1[:3,3], (xform2 - xform1)[:3,3], 1, "%s_center_to_center.pdb"%debug)
    nu.dump_line(xform1[:3,3], real_trans_on_plane, 1, "%s_trans_on_plane.pdb"%debug)

    return np.degrees(angle), rise, r





def ss_core(segments, sc_neighs ):

    # print(sc_neighs)

    scores = []

    seqpos = 0
    for idx, segment in enumerate(segments):

        tag, start, end = segment
        length = end - start

        elem_start = seqpos
        elem_end = seqpos + length

        seqpos = elem_end

        if (idx == 0):
            continue
        if ( idx % 2 != 1 ):    # is turn
            continue

        if ( idx == 1 ):
            elem_start -= 1

        # print(sc_neighs[elem_start:elem_end])

        percent_core = np.sum(sc_neighs[elem_start:elem_end] > 5.2) / length

        scores.append(percent_core )


    return scores


def ss_connectedness(segments, froms, tos ):

    scores = []

    seqpos = 0
    for idx, segment in enumerate(segments):

        tag, start, end = segment
        length = end - start

        elem_start = seqpos
        elem_end = seqpos + length

        seqpos = elem_end

        if (idx == 0):
            continue
        if ( idx % 2 != 1 ):    # is turn
            continue

        if ( idx == 1 ):
            elem_start -= 1

        from_us = (froms >= elem_start) & (froms < elem_end)
        to_us = (tos >= elem_start) & (tos < elem_end)

        our_connected = from_us ^ to_us

        scores.append(np.sum(our_connected) / length )


    return scores

def get_is_helix(segments, nsize):

    is_helix = np.zeros(nsize, bool)

    dssp = npose_dssp(segments, 4)

    for i in range(len(is_helix)):
        is_helix[i] = dssp[i] == "H"

    return is_helix

    # seqpos = 0
    # for idx, segment in enumerate(segments):

    #     tag, start, end = segment
    #     length = end - start

    #     elem_start = seqpos
    #     elem_end = seqpos + length

    #     seqpos = elem_end

    #     if (idx == 0):
    #         continue
    #     if ( idx % 2 != 1 ):    # is turn
    #         continue

    #     if ( idx == 1 ):
    #         elem_start -= 1

    #     is_helix[elem_start:elem_end] = True

    # return is_helix

def worst_motif_hits_in_window(froms, tos, nsize, window_size):

    worst = 1000

    for start in range(0, nsize):
        end = start + window_size
        if ( end >= nsize ):
            continue

        from_us = (froms >= start) & (froms < end)
        to_us = (tos >= start) & (tos < end)

        involves_us = from_us | to_us

        score = np.sum(involves_us)

        worst = min(score, worst)

    return worst

def worst_core_in_window(segments, neighs, nsize, window_size):

    is_helix = get_is_helix(segments, nsize)
    # print(is_helix)

    worst = 1000
    idx = 0

    for start in range(0, nsize):
        end = start + window_size
        if ( end >= nsize ):
            continue

        if ( not np.all(is_helix[start:end] ) ):
            continue

        score = np.percentile( neighs[start:end], 50)
        # score = np.sum(neighs[start:end] < 2)/window_size

        if (score < worst):
            idx = start
        worst = min(score, worst)

    # print(list(zip(list(range(1, 1+len(is_helix))), is_helix, neighs )))
    # print(idx)

    return worst


def get_is_segment(segments, nsize):
    is_segment = np.zeros(nsize, int)

    is_helix = get_is_helix(segments, nsize)

    in_helix = True
    helixno = 0
    for i in range(nsize):
        if ( is_helix[i] ):
            if ( not in_helix ):
                helixno += 1
                in_helix = True
            is_segment[i] = helixno
        else:
            in_helix = False
            is_segment[i] = -1

    return is_segment

import scipy.stats


import numba

@njit(fastmath=True, cache=True)
def int_mode(array):
    counts = numba.typed.Dict.empty(key_type=numba.types.int64, value_type=numba.types.int64)

    for elem in array:
        if ( not elem in counts ):
            counts[elem] = 1
            continue
        counts[elem] += 1

    max_elem = 0
    max_value = 0
    for elem in array:
        count = counts[elem]
        if ( count > max_value ):
            max_value = count
            max_elem = elem

    return elem


# _that_array = np.ones(500, np.uint64)

# import getpy
# def getpy_mode(array):
#     that_array = _that_array[:len(array)] #np.ones(len(array), np.uint64)

#     d = getpy.Dict(np.uint64, np.uint64, default_value=0)

#     d.iadd( array, that_array )

#     return d.keys()[ d.values().argmax() ]

@njit(fastmath=True, cache=True)
def worst_second_worst_motif_interaction(is_segment, is_helix, froms, tos, nsize, mask, care_mask, window_size):

    # print(list(zip(froms, tos)))
    # is_segment = get_is_segment(segments, nsize)
    # is_helix = get_is_helix(segments, nsize)
    # print(is_segment)
    # print(is_helix)

    worst = 100
    idx = 0

    scores = []
    for start in range(0, nsize):
        end = start + window_size
        if ( end >= nsize ):
            continue

        if ( not np.all(is_helix[start:end] ) ):
            continue

        if ( not np.all( care_mask[start:end] ) ):
            continue

        our_segment = is_segment[start]

        # from_us = is_segment[froms] == our_segment #(froms >= start) & (froms < end)
        # to_us = is_segment[tos] == our_segment #(tos >= start) & (tos < end)

        from_us = (froms >= start) & (froms < end) & mask[froms] & mask[tos]
        to_us = (tos >= start) & (tos < end) & mask[froms] & mask[tos]

        interesting = from_us ^ to_us

        # from_us = from_us[interesting]
        # to_us = to_us[interesting]

        who = np.zeros(len(from_us), np.int_)
        who.fill(-2)
        who[~from_us] = is_segment[froms[~from_us]]
        who[~to_us] = is_segment[tos[~to_us]]

        # print(our_segment)
        # print(who)
        who = who[interesting]
        who = who[who != -1]
        who = who[who != our_segment]
        assert( not np.any(who == -2 ) )

        # print(who)

        score = 0
        if ( len(who) != 0 ):
            most_connected = int_mode(who)


            # print(most_connected, my_mode)
            # print(start, most_connected)

            who_else = who[who != most_connected]

            score = len(who_else)

        scores.append(score)

        if (score < worst):
            idx = start
        worst = min(score, worst)

    # print(list(zip(list(range(1, 1+len(is_helix))), is_helix, neighs )))
    # print(idx)
    # print(scores)

    scorez = np.array(scores, np.float_)

    return worst, idx, np.mean(scorez > 1), np.mean(scorez > 2), scorez



def avg_second_worst_motif_interaction(is_segment, is_helix, froms, tos, nsize, mask, care_mask, window_size):

    # print(list(zip(froms, tos)))
    # is_segment = get_is_segment(segments, nsize)
    # is_helix = get_is_helix(segments, nsize)
    # print(is_segment)
    # print(is_helix)

    worst = 100
    idx = 0

    scores = []
    for start in range(0, nsize):
        end = start + window_size
        if ( end >= nsize ):
            continue

        if ( not np.all(is_helix[start:end] ) ):
            continue

        if ( not np.all( care_mask[start:end] ) ):
            continue

        our_segment = is_segment[start]

        # from_us = is_segment[froms] == our_segment #(froms >= start) & (froms < end)
        # to_us = is_segment[tos] == our_segment #(tos >= start) & (tos < end)

        from_us = (froms >= start) & (froms < end) & mask[froms] & mask[tos]
        to_us = (tos >= start) & (tos < end) & mask[froms] & mask[tos]

        interesting = from_us ^ to_us

        # from_us = from_us[interesting]
        # to_us = to_us[interesting]

        who = np.zeros(len(from_us), int)
        who.fill(-2)
        who[~from_us] = is_segment[froms[~from_us]]
        who[~to_us] = is_segment[tos[~to_us]]

        # print(our_segment)
        # print(who)
        who = who[interesting]
        who = who[who != -1]
        who = who[who != our_segment]
        assert( not np.any(who == -2 ) )

        # print(who)

        score = 0
        if ( len(who) != 0 ):
            most_connected = int_mode(who)


            # print(most_connected, my_mode)
            # print(start, most_connected)

            who_else = who[who != most_connected]

            score = len(who_else)

        scores.append(score)

        if (score < worst):
            idx = start
        worst = min(score, worst)

    # print(list(zip(list(range(1, 1+len(is_helix))), is_helix, neighs )))
    # print(idx)
    # print(scores)

    scores = np.array(scores)

    return np.mean(np.clip(scores, 0, 1)), np.mean(scores >= 2)




def npose_dssp(segments, helix_res_on_turn):
    segments = segments[1:]
    dssp = "H"
    last_one = "L"
    for segment in segments:
        if ( last_one == "L"):
            dssp += "H"*(segment[2] - segment[1])
            last_one = "H"
        else:
            dssp += "H"*(helix_res_on_turn-1)
            dssp += "L"*(segment[2] - segment[1] - helix_res_on_turn - (helix_res_on_turn-1))
            dssp += "H"*(helix_res_on_turn)
            last_one = "L"
    return dssp



def load_dbs( helix_list_name, turn_list_name, allowable_db_name, options = {} ):


        if ('helix_spare_res' not in options):
            options['helix_spare_res'] = 1
        if ('min_helix_len' not in options):
            options['min_helix_len'] = 1

        helix_list = []
        with open(helix_list_name) as f:
            for line in f:
                line = line.strip()
                if (len(line) == 0):
                    continue
                helix_list.append(line)

        turn_list = []
        with open(turn_list_name) as f:
            for line in f:
                line = line.strip()
                if (len(line) == 0):
                    continue
                turn_list.append(line)

        print("Loading database")
        with open(allowable_db_name) as f:
            allowable_db = json.loads(f.read())


        print("Loading helices")
        db_helix = {}
        for fname in helix_list:
            npose = npose_from_file_fast(fname)
            tag = get_tag(fname)

            db_helix[tag] = DBHelix(npose, tag, allowable_db)

            if ("ideal" in tag):
                options['test_helix'] = tag

        max_db_helix_size = 0
        max_helix_dist = 0
        for tag in db_helix:
           max_helix_dist = max( max_helix_dist, np.max(np.array(db_helix[tag].dists_matrix)))
           max_db_helix_size = max( max_db_helix_size, nsize( db_helix[tag].npose ) )
        options['helix_max_dist'] = max_helix_dist + 1
        options['max_db_helix_size'] = max_db_helix_size



        print("Loading turns")
        db_turn = {}
        for fname in turn_list:
            npose = npose_from_file_fast(fname)
            tag = get_tag(fname)

            db_turn[tag] = DBTurn(npose, tag, allowable_db)

        db_turn['origin'] = DBOrigin(npose[:1*R], 'origin', db_helix, options)

        return db_helix, db_turn, allowable_db


########## START OF MAIN ####################


def main():

    # parser = argparse.ArgumentParser()
    # parser.add_argument("helix_list", type=str )
    # parser.add_argument("turn_list", type=str )
    # parser.add_argument("-allowable_db", type=str)


    # args = parser.parse_args(sys.argv[1:])

    options = {}
    options['min_worm_length'] = 48
    options['max_worm_length'] = 65
    options['max_segments'] = 3*2 # because of origin
    options['max_results'] = 100000
    options['min_helix_len'] = 8
    options['max_helix_len'] = 1000
    options['max_turn_clashes'] = 4
    options['clash_dist'] = 4.5
    options['helix_spare_res'] = 1
    options['turn_spare_res'] = 2
    options['max_diameter'] = 25
    options['max_rg'] = 10.5    # 12.5 is the correct number fo 30% scn 6helical
    options['min_sc_neigh'] = 0  # 2.65 works decent
    options['perc_core_scn'] = 0.25
    options['randomize'] = True
    options['chance_of_rollback'] = 1/10
    options['fruitless_rollback'] = 20
    options['max_helix_tries'] = 10

    options['true_clashes'] = 10
    options['motif_hash_core_per_res'] = 0
    options['min_worst_others5'] = 0
    options['repeats'] = 0


    # 3 helical

    # topology
    options['min_worm_length'] = 48
    options['max_worm_length'] = 65
    options['max_segments'] = 3*2 # because of origin
    options['chance_of_rollback'] = 1/10

    # scoring
    options['min_worst_others7'] = 1
    options['min_worst_others9'] = 2
    options['perc_core_scn'] = 0.21
    options['min_sc_neigh'] = 0 
    options['median_sc_neigh'] = 0.2
    options['max_actual_helix_clashes'] = 5


    # performance
    options['max_total_helix_clashes'] = 4
    options['max_rg'] = 10.7  
    options['fruitless_rollback'] = 70  
    options['max_diameter'] = 30
    options['min_helix_len'] = 10
    options['max_helix_tries'] = 6
    options['max_helix_starts'] = 6
    options['max_helix_ends'] = 4
    options['max_turn_tries'] = 9

    # 6 helical

    # options['min_worm_length'] = 100
    # options['max_worm_length'] = 119
    # options['max_segments'] = 6*2 # because of origin
    # options['chance_of_rollback'] = 1/10
    # options['fruitless_rollback'] = 30
    # options['max_diameter'] = 25
    # options['max_rg'] = 12.5    # 12.5 is the correct number fo 30% scn 6helical
    # options['min_sc_neigh'] = 0  # 2.65 works decent
    # options['perc_core_scn'] = 0.28
    # options['max_helix_tries'] = 3
    # options['max_turn_tries'] = 30

    jhr = True


    parser = argparse.ArgumentParser()
    parser.add_argument("helix_list", type=str )
    parser.add_argument("turn_list", type=str )
    parser.add_argument("-allowable_db", type=str)


    for option in options:
        parser.add_argument("-" + option, type=type(options[option]), default=options[option])

    args = parser.parse_args(sys.argv[1:])

    for option in options:
        options[option] = args.__getattribute__(option)

    options['turn_end_func'] = None
    options['helix_end_func'] = check_finished

    if ( jhr ):
        options['min_worm_length'] = 30
        options['max_worm_length'] = 50
        options['min_helix_len'] = 5

        options['clash_dist'] = 4.5

        options['turn_end_func'] = jhr_finish
        options['helix_end_func'] = None

        options['max_segments'] = 2*2+1

        options['repeats'] = 4

        #scoring

        # options['min_worst_others5'] = 1
        # options['min_worst_others7'] = 6
        # options['min_worst_others9'] = 7
        # options['perc_core_scn'] = 0.29
        # options['min_sc_neigh'] = 0 
        # options['median_sc_neigh'] = 0.2
        # options['max_actual_helix_clashes'] = 12
        # options['true_clashes'] = 4
        # options['motif_hash_core_per_res'] = 1.4

        # from halfroids

        options['min_worst_others5'] = 0
        options['min_worst_others7'] = 0
        options['min_worst_others9'] = 0

        options['avg_worst_others5'] = 0.75
        options['avg_worst_others7'] = 0
        options['avg_worst_others9'] = 0.85

        options['avg_worst_others5_2'] = 0
        options['avg_worst_others7_2'] = 0
        options['avg_worst_others9_2'] = 0.85

        options['perc_core_scn'] = 0.28
        options['min_sc_neigh'] = 0 
        options['median_sc_neigh'] = 0.2
        options['max_actual_helix_clashes'] = 16
        options['true_clashes'] = 3
        options['absurd_clashes'] = 0
        options['motif_hash_core_per_res'] = 2.3


        options['helical_angle_range'] = (25, 90)
        options['helical_rise_range'] = (0, 8)
        options['helical_radius_range'] = (10, 30)

        # options['helical_angle_range'] = (0, 5)
        # options['helical_rise_range'] = (0, 1000)
        # options['helical_radius_range'] = (100, 1000000000000000)


        # performance

        options['max_total_helix_clashes'] = 8
        options['max_rg'] = 10.7  
        options['fruitless_rollback'] = 70  
        options['max_diameter'] = 30
        options['min_helix_len'] = 10
        options['max_helix_tries'] = 6
        options['max_helix_starts'] = 6
        options['max_helix_ends'] = 4
        options['max_turn_tries'] = 9



    db_helix, db_turn, allowable_db = load_dbs( args.helix_list, args.turn_list, args.allowable_db, options )


    npose = db_turn[list(db_turn.keys())[0]].npose
    get_slow_clash(npose)
    out = np.zeros((1, nsize(npose), nsize(npose), 4))
    get_dot_representation(npose, out, 0)
    print("Loaded")

    # It seems weird to start this way. But the convention is
    # to use the last residue from the previous part
    # So the origin must be the first residue
    start_xform = np.identity(4)
    start_segments = [('origin', 0, 1)]
    start_context = np.array([[0, 0, 0]])
    start_length = 1

    spare_data = SpareData([0], 0)

    results = []

    # If it rolls back enough times it can break of the bottom loop
    while ( not wrap_it_up ):
        add_a_helix(start_context, start_xform, start_segments, start_length, spare_data, db_helix, db_turn, options, results )

import signal

def handler(signal_received, frame):
    # Handle any cleanup here
    global wrap_it_up
    wrap_it_up = True
    # print('SIGINT or CTRL-C detected. Exiting gracefully')
    # exit(0)

if (__name__ == "__main__"):
    signal.signal(signal.SIGINT, handler)
    main()

# print("\n".join(the_log))










