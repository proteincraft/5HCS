#!/usr/bin/env python

import sys
import os

# import npose

import hashlib

from numba import njit

from importlib import reload
import npose_util as nu
reload(nu)

import argparse

# import motif_stuff
# import motif_stuff2
import re
import time
import itertools

import atomic_depth

import numpy as np

import helical_worms
reload(helical_worms)
import motif_stuff2

if ( len(sys.argv) > 2 ):
    files = sys.argv[1:]
else:
    files = []
    with open(sys.argv[1]) as f:
        for line in f:
            line = line.strip()
            if ( len(line) == 0 ):
                continue
            files.append(line)


parser = argparse.ArgumentParser()
parser.add_argument("pdbs", type=str, nargs="*")
parser.add_argument("-in:file:silent", type=str, default="")
parser.add_argument("-worm_segments", type=str, default="")
parser.add_argument("-helix_list", type=str, default="")
parser.add_argument("-turn_list", type=str, default="")
parser.add_argument("-allowable_db", type=str, default="")
parser.add_argument("-force_silent", action="store_true")
parser.add_argument("-debug", action="store_true")
parser.add_argument("-allow_nonstandard", action="store_true")
parser.add_argument("-pcscn_cut", type=float, default=0.19)
parser.add_argument("-avgfive_cut", type=float, default=0.70)
parser.add_argument("-avgnine_cut", type=float, default=0.80)
parser.add_argument("-avgnine_two_cut", type=float, default=0.80)

parser.add_argument("-min_scaff_length", type=float, default=120)
parser.add_argument("-max_scaff_length", type=float, default=165)
parser.add_argument("-min_curvature_angle", type=float, default=90)
parser.add_argument("-helix_worst_gap_cut", type=float, default=100000)


args = parser.parse_args(sys.argv[1:])

pdbs = args.pdbs
silent = args.__getattribute__("in:file:silent")
segments_file = args.worm_segments



if ( silent != "" ):
    print("Loading silent")
    nposes, pdbs = nu.nposes_from_silent( silent )

all_segments = None
if ( segments_file != "" ):
    db_helix, db_turn, allowable_db = helical_worms.load_dbs( args.helix_list, args.turn_list, args.allowable_db )
    all_segments = []
    with open(segments_file) as f:
        for line in f:
            line = line.strip()
            if (len(line) == 0):
                continue
            try:
                all_segments.append(eval(line))
            except:
                pass

    pdbs = range(len(all_segments))


def get_ipdb(i):
    if ( silent != "" ):
        return nposes[i], pdbs[i]

    if ( not all_segments is None ):
        print(all_segments[i])
        npose, parts, out_name = helical_worms.result_to_pdb(all_segments[i], db_helix, db_turn, dump_pdb=False)

        m = hashlib.md5()
        m.update("_".join(out_name).encode('utf-8'))
        hash_name = m.hexdigest()

        return npose, hash_name

    npose = nu.npose_from_file_fast(pdbs[i])
    return npose, nu.get_tag(pdbs[i])


def get_concavity(npose, out_score_map, ignore_loops=True):

    resl_scale = 1

    # print("Atomic depth")
    resl = 1*resl_scale

    cas = nu.extract_atoms( npose, [nu.CA] )

    radii = np.repeat(2, len(cas))
    surf = atomic_depth.AtomicDepth(cas[:,:3].reshape(-1), radii, 8, 1, True, 1)
    # print("Done")

    # verts = surf.get_surface_vertex_bases().reshape(-1, 3)
    # vert_normals = surf.get_surface_vertex_normals().reshape(-1, 3)
    face_centers = surf.get_surface_face_centers().reshape(-1, 3)
    face_normals = surf.get_surface_face_normals().reshape(-1, 3)

    clustered, _ = nu.cluster_points(face_centers, 1)
    # print("clustered")
    face_centers = face_centers[clustered]
    face_normals = face_normals[clustered]

    if ( args.debug ):
        nu.dump_pts(face_centers, "surface.pdb")

    # compatible = np.zeros((len(face_centers), len(face_centers)), np.bool)

    # remove parallel rays
    crit = np.cos(np.radians(20))
    # for i in range(len(face_centers)):
    #     dot_values = np.sum(face_normals*face_normals[i], axis=-1)
    #     compatible[i,:] = dot_values < crit

    compatible = np.sum( face_normals[:,None] * face_normals[None,:], axis=-1) < crit

    c2 = compatible.copy()

    # next only allow faces that are facing each other
    # Also, disallow points closer than 6 A
    # check this by ensuring the normal vector and the facing vector are in the same direction
    # for i in range(len(face_centers)):
    #     facings = face_centers[i] - face_centers
    #     dots = np.sum(facings*face_normals, axis=-1)
    #     compatible[i,:] &= dots > 0
    #     compatible[i,:] &= np.sum( np.square(facings), axis=-1) > 6*6
    # compatible &= compatible.T

    facings = face_centers[:,None] - face_centers[None,:]
    dots = np.sum(facings*face_normals[None,:], axis=-1)

    compatible &= dots > 0
    compatible &= np.sum( np.square( facings ), axis=-1) > 6*6
    compatible &= compatible.T



    if ( ignore_loops ):
        is_helix = nu.npose_dssp_helix(npose)
        ncac = nu.extract_atoms(npose, [nu.N, nu.CA, nu.C])[:,:3]

        is_helix[:2] = False
        is_helix[-2:] = False

        loop_atoms = ncac.reshape(-1, 3, 3)[~is_helix].reshape(-1, 3)

        # print("loops")
        loopclash = nu.clashgrid_from_points(loop_atoms, 3.8, 1*resl_scale)

        clashes = loopclash.arr[tuple(loopclash.floats_to_indices(face_centers).T)]

        if ( args.debug ):
            nu.dump_pts(face_centers[clashes > 0], "excluded.pdb")

        compatible[ clashes >= 1] = 0
        compatible &= compatible.T



    # print("clashes")
    clashgrid = nu.clashgrid_from_points(face_centers - face_normals*0.9, 0.6, 1*resl_scale)
    # print("Clashgrid done")

    if ( args.debug ):
        clashgrid.dump_grids_true("clashgrid.pdb", lambda x:x)


    ray_pairs = np.array( np.where( np.triu(compatible) ) ).T

    starts = face_centers[ray_pairs[:,0]]
    ends = face_centers[ray_pairs[:,1]]

    norm_vecs = ends - starts
    norm_vecs /= np.linalg.norm(norm_vecs, axis=-1)[...,None]

    pad_size = 1

    padded_starts = starts + norm_vecs * pad_size
    padded_ends = ends - norm_vecs * pad_size

    # print("Preparing to ray trace %i pairs"%(len(padded_starts)))
    clashes = clashgrid.ray_trace_many(padded_starts, padded_ends, 1)
    # print("Done")

    good_indices = np.where(clashes == 0)[0]

    if ( len(good_indices) == 0 ):
        print("Baddd")
        out_score_map['angle'] = 0
        out_score_map['diameter'] = 0
        out_score_map['count'] = 0
        # sys.exit()
        return False


    good_ray_pairs = ray_pairs[good_indices]

    good_ray_dots = np.sum( face_normals[good_ray_pairs[:, 0]] * face_normals[good_ray_pairs[:,1]], axis=-1)
    good_ray_vecs = face_centers[good_ray_pairs[:, 0]] - face_centers[good_ray_pairs[:, 1]]
    good_ray_lengths = np.linalg.norm(good_ray_vecs, axis=-1)


    # Define axis H as the cross product of the two normals
    # At some point the two normals cross paths with respect to H
    # Define r1 as the distance from V1_base to the crossing point
    # Define r2 as the distance from V2_base to the crossing point
    # Define h as the distance along H between the two crosses
    # Let K = H_hat X V1_hat

    # V1_base + r1*V1_hat + H_hat*h = V2_base + r2*V2_hat
    #
    #
    # End up with this equation after dotting with  H, V1, and K as basis: (hats ommited)
    # [ V1H  -V2H  HH  ][ r1 ]   [ V2_base*H  - V1_base*H ]
    # [ V1V1 -V2V1 HV1 ][-r2 ] = [ V2_base*V1 - V1_base*V1]
    # [ V1K  -V2K  HK  ][  h ]   [ V2_base*K  - V1_base*K ]

    V1_hat = face_normals[good_ray_pairs[:, 0]]
    V2_hat = face_normals[good_ray_pairs[:, 1]]
    V1_base = face_centers[good_ray_pairs[:, 0]]
    V2_base = face_centers[good_ray_pairs[:, 1]]
    H_hat = np.cross( V1_hat, V2_hat )
    H_hat /= np.linalg.norm(H_hat, axis=-1)[...,None]
    K_hat = np.cross( H_hat, V1_hat )


    # Make a set of 3x3 where every row has, H, V, K, and every column has V1, -V2, H dotted together
    # Make the 3x3 part of A*x=B. The first row is [V1*H, -V2*H, H*H] where * is dot

    # make a list of things that look like this: Don't forget that all these are vectors!!!
    # [[ H, H, H], [V1, V1, V1], [K, K, K]]

    # Step one is interleave so that we get, H V K H V K H V K
    interleaved = np.zeros((len(H_hat)*3, 3))
    interleaved[0::3,:] = H_hat
    interleaved[1::3,:] = V1_hat
    interleaved[2::3,:] = K_hat

    # Step two is perform repeat 3 on axis=-2 to get HHH VVV KKK HHH VVV KKK HHH VVV KKK
    repeated = np.repeat(interleaved, 3, axis=-2)

    # Finish it up with a reshape to get [[HHH], [VVV], [KKK]], ...

    three_by_three_by_three = repeated.reshape(-1, 3, 3, 3)

    # We made the part with H V1 and K, now make the part with V1, -V2, and H

    # First, we have to make another giant 3x3x3, but this time with a different repeat patter
    # V1a V2a Ha V1b V2b Hb V1c V2c Hc...
    interleaved2 = np.zeros((len(V1_hat)*3, 3))
    interleaved2[0::3,:] = V1_hat
    interleaved2[1::3,:] = -V2_hat
    interleaved2[2::3,:] = H_hat

    # group this into 3x3s and repeat those
    # [V1a V2a Ha] [V1b V2b Hb] [V1c V2c Hc]
    grouped = interleaved2.reshape(-1, 3, 3)

    # [V1a V2a Ha] [V1a V2a Ha] [V1a V2a Ha] [V1b ...
    repeated2 = np.repeat(grouped, 3, axis=-3)

    # [[V1a V2a Ha] [V1a V2a Ha] [V1a V2a Ha]] [[V1b ...
    other_three_by_three_by_three = repeated2.reshape(-1, 3, 3, 3)

    # Time to calculate the dot product
    A = np.sum( three_by_three_by_three * other_three_by_three_by_three, axis=-1)

    inv_A = np.linalg.inv(A)


    B = np.stack(( 

        np.sum(V2_base*H_hat - V1_base*H_hat, axis=-1),
        np.sum(V2_base*V1_hat - V1_base*V1_hat, axis=-1),
        np.sum(V2_base*K_hat - V1_base*K_hat, axis=-1)

        )).T

    # A*x=B
    # A-1*A*x=A-1*B
    # x = A-1*B

    x = (inv_A @ B[...,None]).reshape(-1, 3)

    r1 = x[:,0]
    r2 = x[:,1]
    h = x[:,2]
    all_h =  h



    most_anti = np.min(good_ray_dots)
    cutoff = most_anti + 0.1

    mask = np.zeros(1)

    while (mask.sum() < 120 and cutoff < 1 ):
        mask = (good_ray_dots <= cutoff) & ((r1 + r2)/2 > np.abs(all_h))

        cutoff += 0.01

    if ( cutoff > 1 ):
        print("Bad?!?!")
        out_score_map['angle'] = 0
        out_score_map['diameter'] = 0
        out_score_map['count'] = mask.sum()
        return False
        # sys.exit()


    def draw_two_atoms(name, atom1, atom2):
        with open(name, "w") as f:
            f.write(format_atom(1,resi=1, x=atom1[0], y=atom1[1], z=atom1[2]))
            f.write(format_atom(1,resi=1, x=atom2[0], y=atom2[1], z=atom2[2]))


    def draw_two_atoms_w_rays(name, atom1, ray1, atom2, ray2):
        with open(name, "w") as f:
            f.write(format_atom(1,resi=1, x=atom1[0], y=atom1[1], z=atom1[2]))
            for i in range(1, 10):
                f.write(format_atom(1,resi=1, x=atom1[0]+i*ray1[0], y=atom1[1]+i*ray1[1], z=atom1[2]+i*ray1[2]))
            f.write(format_atom(1,resi=1, x=atom2[0], y=atom2[1], z=atom2[2]))
            for i in range(1, 10):
                f.write(format_atom(1,resi=1, x=atom2[0]+i*ray2[0], y=atom2[1]+i*ray2[1], z=atom2[2]+i*ray2[2]))

    def draw_atoms_w_rays(name, atoms, rays):
        with open(name, "w") as f:
            for k in range(len(atoms)):
                atom1 = atoms[k]
                ray1 = rays[k]*2
                f.write(format_atom(1,resi=1, x=atom1[0], y=atom1[1], z=atom1[2]))
                for i in range(1, 10):
                    f.write(format_atom(1,resi=1, x=atom1[0]+i*ray1[0], y=atom1[1]+i*ray1[1], z=atom1[2]+i*ray1[2]))
            # f.write(format_atom(1,resi=1, x=atom2[0], y=atom2[1], z=atom2[2]))
            # for i in range(1, 10):
            #     f.write(format_atom(1,resi=1, x=atom2[0]+i*ray2[0], y=atom2[1]+i*ray2[1], z=atom2[2]+i*ray2[2]))





    if ( args.debug ):
        with open("concave.pdb", "w") as f:
            concave = face_centers[np.unique(good_ray_pairs[mask].reshape(-1))]
            nu.dump_pts(concave, "concave.pdb")


    # draw_two_atoms("test.pdb", face_centers[good_pair[0]], face_centers[good_pair[1]])



    lengths = good_ray_lengths[mask]
    dots = good_ray_dots[mask]

    assert(len(r1) == len(good_ray_dots))

    r1s = r1[mask]
    r2s = r2[mask]
    hs = h[mask]

    r = np.median((r1s + r2s)/2)
    h = np.median(np.abs(hs))


    def visualize_pair( offset):
        base1 = V1_base[offset]
        ray1 = V1_hat[offset]
        base2 = V2_base[offset]
        ray2 = V2_hat[offset]
        angle = np.degrees(np.arccos(np.dot(ray1, ray2)))
        diameter = np.linalg.norm(base1 - base2)
        radius1 = r1[offset]
        radius2 = r2[offset]

        h_hat = H_hat[offset]
        h_base = base1 + ray1*radius1
        h_ = all_h[offset]

        h_base2 = base2 + ray2*radius2

        print("Angle: %.1f  Size: %.1f  Radius1: %.1f Radius2: %.1f Rise: %.1f"%
                        (angle, diameter, radius1, radius2, h_))
        draw_atoms_w_rays("test.pdb", [base1, base2, h_base, h_base2], [ray1, ray2, h_hat, h_hat])




    length = np.median(lengths)
    dot = np.median(dots)

    angle = np.degrees(np.arccos(dot))

    print("Angle: %.1f  Size: %.1f  Radius: %.1f  Rise: %.1f"%(angle, length, r, h))

    out_score_map['angle'] = angle
    out_score_map['size'] = length
    out_score_map['radius'] = r
    out_score_map['rise'] = h
    # out_score_map['count'] = mask.sum()

    return True
    # sys.exit()


def get_extraneous(npose):
    # nu.dump_npdb(npose, "test.pdb")
    is_helix = nu.npose_dssp_helix(npose)
    ss_elems = nu.npose_helix_elements(is_helix)

    # print(is_helix)

    assert(is_helix[0] and is_helix[-1])

    is_segment = np.zeros(nu.nsize(npose), np.int)
    is_segment.fill(-1)

    num_helices = 0
    for i, elem in enumerate(ss_elems):
        helix, start, end = elem
        if ( not helix ):
            continue
        is_segment[start:end+1] = i
        num_helices += 1

    care_mask = np.ones(nu.nsize(npose), np.bool)

    ca_cb = nu.extract_atoms(npose, [nu.CA, nu.CB]).reshape(-1, 2, 4)[...,:3]

    return is_helix, is_segment, ss_elems, care_mask, ca_cb

def trim_this_pose(npose, froms, tos, min_length, max_length):

    min_helix_length = 12

    # prepare info
    is_helix, is_segment, ss_elems, care_mask, ca_cb = get_extraneous(npose)

    save_froms = froms
    save_tos = tos


    first_bundle_mask = (is_segment == 0) | (is_segment == 2) | (is_segment == 4)

    k = len(ss_elems)-1
    last_bundle_mask = (is_segment == k-0) | (is_segment == k-2) | (is_segment == k-4)

    # get info about first and last helices
    _, _, last_n_res = ss_elems[0]
    _, first_c_res, _ = ss_elems[-1]

    # trim residues will be removed
    max_n_trim_res = last_n_res - min_helix_length
    max_c_trim_res = first_c_res + min_helix_length

    # cant trim one of the helices
    if ( max_n_trim_res < -1 or max_c_trim_res > nu.nsize(npose) ):
        return None, None, None

    # all possible trimmings
    ntrims = range(-1, max_n_trim_res+1)
    ctrims = range(max_c_trim_res, nu.nsize(npose))

    ntrim_ctrim = list(itertools.product(ntrims, ctrims))
    scores = np.zeros(len(ntrim_ctrim))
    scores.fill(-1)

    for i, (ntrim, ctrim) in enumerate(ntrim_ctrim):

        size = ctrim - ntrim - 1
        if ( size < min_length or size > max_length ):
            continue


        froms = save_froms - ntrim - 1
        tos = save_tos - ntrim - 1

        oob = (tos < 0) | (tos >= size) | (froms < 0) | (froms >= size)

        tos = tos[~oob]
        froms = froms[~oob]

        # a = trim_npose(npose, ntrim+1, ctrim-1)
        # nu.dump_npdb(a, "test.pdb")


        score_map = {}
        fail = score_seg2(is_segment[ntrim+1:ctrim], is_helix[ntrim+1:ctrim], ca_cb[ntrim+1:ctrim],
            froms, tos, first_bundle_mask[ntrim+1:ctrim], score_map, "N")
        if (not fail ):
            fail = score_seg2(is_segment[ntrim+1:ctrim], is_helix[ntrim+1:ctrim], ca_cb[ntrim+1:ctrim],
                        froms, tos, last_bundle_mask[ntrim+1:ctrim], score_map, "C")

        if ( fail ):
            continue
        # _, _, _, _, neighs = helical_worms.get_avg_sc_neighbors( ca_cb[ntrim+1:ctrim], care_mask[ntrim+1:ctrim] )
        # is_core_boundary = neighs > 2

        # _, _, avg5_n, avg5_2_n, _ = helical_worms.worst_second_worst_motif_interaction(is_segment[ntrim+1:ctrim], is_helix[ntrim+1:ctrim],
        #                                         froms, tos, size, is_core_boundary&first_bundle_mask[ntrim+1:ctrim], care_mask[ntrim+1:ctrim], 5)
        # _, _, avg9_n, avg9_2_n, _ = helical_worms.worst_second_worst_motif_interaction(is_segment[ntrim+1:ctrim], is_helix[ntrim+1:ctrim],
        #                                         froms, tos, size, is_core_boundary&first_bundle_mask[ntrim+1:ctrim], care_mask[ntrim+1:ctrim], 9)

        # _, _, avg5_c, avg5_2_c, _ = helical_worms.worst_second_worst_motif_interaction(is_segment[ntrim+1:ctrim], is_helix[ntrim+1:ctrim],
        #                                         froms, tos, size, is_core_boundary&last_bundle_mask[ntrim+1:ctrim], care_mask[ntrim+1:ctrim], 5)
        # _, _, avg9_c, avg9_2_c, _ = helical_worms.worst_second_worst_motif_interaction(is_segment[ntrim+1:ctrim], is_helix[ntrim+1:ctrim],
        #                                         froms, tos, size, is_core_boundary&last_bundle_mask[ntrim+1:ctrim], care_mask[ntrim+1:ctrim], 9)

        avg5_n = score_map["N_avg5"]
        avg9_n = score_map["N_avg9"]
        avg9_2_n = score_map["N_avg9_2"]
        pcscn_n = score_map["N_pcscn"]

        avg5_c = score_map["C_avg5"]
        avg9_c = score_map["C_avg9"]
        avg9_2_c = score_map["C_avg9_2"]
        pcscn_c = score_map["C_pcscn"]

        # this is the important line
        score = (avg5_n + avg9_n + 2*avg9_2_n)**0.5 + (avg5_c + avg9_c + 2*avg9_2_c)**0.5

        scores[i] = score


        # print("%5i %5i %7.2f %7.2f %7.2f"%(ntrim+1, ctrim+1, score, pcscn_n, pcscn_c))


    if ( scores.max() == -1 ):
        return None, None, None

    argmax = np.argmax(scores)

    ntrim, ctrim = ntrim_ctrim[argmax]

    npose = trim_npose(npose, ntrim+1, ctrim-1)

    return npose, ntrim, ctrim

def score_seg2(is_segment, is_helix, ca_cb, froms, tos, mask, score_map, prefix):

    start = np.min(np.where(mask)[0])
    end = np.max(np.where(mask)[0])

    # npose = trim_npose(npose, start, end)
    # nu.dump_npdb(npose, prefix + ".pdb")



    is_segment = is_segment[start:end+1]
    is_helix = is_helix[start:end+1]
    ca_cb = ca_cb[start:end+1]
    mask = mask[start:end+1]

    froms = froms - start
    tos = tos - start
    oob = (tos < 0) | (tos >= len(ca_cb)) | (froms < 0) | (froms >= len(ca_cb))
    froms = froms[~oob]
    tos = tos[~oob]

    _, _, _, _, neighs = helical_worms.get_avg_sc_neighbors( ca_cb, mask )
    is_core_boundary = neighs > 2

    percent_core = (neighs[mask] > 5.2).mean()

    if ( percent_core < args.pcscn_cut ):
        return True

    if ( is_core_boundary.sum() == 0 ):
        print("None")

    _, _, avg9, avg9_2, _ = helical_worms.worst_second_worst_motif_interaction(is_segment, is_helix,
                                            froms, tos, len(ca_cb), is_core_boundary, mask, 9)

    if ( avg9 < args.avgnine_cut ):
        return True
    if ( avg9_2 < args.avgnine_two_cut ):
        return True

    _, _, avg5, avg5_2, _ = helical_worms.worst_second_worst_motif_interaction(is_segment, is_helix,
                                            froms, tos, len(ca_cb), is_core_boundary, mask, 5)

    if ( avg5 < args.avgfive_cut ):
        return True

    score_map[prefix + "_avg5"] = avg5
    score_map[prefix + "_avg9"] = avg9
    score_map[prefix + "_avg9_2"] = avg9_2
    score_map[prefix + "_pcscn"] = percent_core

    return False


def score_seg(npose, is_segment, is_helix, froms, tos, is_core_boundary, mask, score_map, prefix):


    _, _, avg5, avg5_2, _ = helical_worms.worst_second_worst_motif_interaction(is_segment, is_helix,
                                            froms, tos, nu.nsize(npose), is_core_boundary&mask, mask, 5)
    _, _, avg9, avg9_2, _ = helical_worms.worst_second_worst_motif_interaction(is_segment, is_helix,
                                            froms, tos, nu.nsize(npose), is_core_boundary&mask, mask, 9)

    score_map[prefix + "_avg5"] = avg5
    score_map[prefix + "_avg9"] = avg9
    score_map[prefix + "_avg9_2"] = avg9_2

def get_helix_stats(cas, params1, params2):

    _, start1, end1 = params1
    _, start2, end2 = params2

    cas1 = cas[start1:end1+1]
    cas2 = cas[start2:end2+1]

    # output shape should be (cas1.shape[-2], cas2.shape[-2])
    pair_dists = np.linalg.norm(cas1[:,None,:] - cas2, axis=-1)

    closest = np.min(pair_dists)

    # terrible greedy method here

    needed = np.min([len(cas1), len(cas2), 4])

    dists = []

    close_pairs = np.dstack(np.unravel_index(np.argsort(pair_dists.ravel()), pair_dists.shape))[0]
    used1 = set()
    used2 = set()

    for i1, i2 in close_pairs:
        if ( i1 in used1 ):
            continue
        if ( i2 in used2 ):
            continue
        used1.add(i1)
        used2.add(i2)

        dists.append(pair_dists[i1, i2])

    return closest, np.mean(dists)




def get_five_cross(npose, score_map):

    is_helix, is_segment, ss_elems, care_mask, ca_cb = get_extraneous(npose)

    num_fives = score_map['helices'] - 4

    cas = nu.extract_atoms(npose, [nu.CA])

    whole_closests = []
    whole_avgs = []

    for ifive in range(num_fives):
        these_helices = ss_elems[ifive*2:(ifive+5)*2:2]

        closests = []
        avgs = []

        closest, avg = get_helix_stats(cas, these_helices[0], these_helices[3])
        closests.append(closest)
        avgs.append(avg)
        closest, avg = get_helix_stats(cas, these_helices[0], these_helices[4])
        closests.append(closest)
        avgs.append(avg)
        closest, avg = get_helix_stats(cas, these_helices[1], these_helices[3])
        closests.append(closest)
        avgs.append(avg)
        closest, avg = get_helix_stats(cas, these_helices[1], these_helices[4])
        closests.append(closest)
        avgs.append(avg)

        closest = np.min(closests)
        avg = np.min(avgs)

        whole_closests.append(closest)
        whole_avgs.append(avg)

    score_map['helix_worst_gap'] = np.max(whole_closests)
    score_map['helix_worst_gap_avg'] = np.max(whole_avgs)




def score_npose(npose, froms, tos, score_map):

    is_helix, is_segment, ss_elems, care_mask, ca_cb = get_extraneous(npose)

    score_map['helices'] = len(ss_elems)//2 + 1

    # here we make sure that the first and last helical bundles look ok

    first_bundle_mask = (is_segment == 0) | (is_segment == 2) | (is_segment == 4)

    k = len(ss_elems)-1

    last_bundle_mask = (is_segment == k-0) | (is_segment == k-2) | (is_segment == k-4)

    assert(np.all(is_helix[first_bundle_mask]))
    assert(np.all(is_helix[last_bundle_mask]))


    _, _, _, _, neighs = helical_worms.get_avg_sc_neighbors( ca_cb, care_mask )
    is_core_boundary = neighs > 2

    score_seg2(is_segment, is_helix, ca_cb, froms, tos, first_bundle_mask, score_map, "N")
    score_seg2(is_segment, is_helix, ca_cb, froms, tos, last_bundle_mask, score_map, "C")

    any_fail = False
    if ( score_map["N_pcscn"] < args.pcscn_cut or score_map["C_pcscn"] < args.pcscn_cut ):
        # print("Fail pcscn: %6.2f %6.2f"%(score_map["N_pcscn"], score_map["C_pcscn"]))
        any_fail = True

    if ( score_map["N_avg5"] < args.avgfive_cut or score_map["C_avg5"] < args.avgfive_cut ):
        # print("Fail avg5: %6.2f %6.2f"%(score_map["N_avg5"], score_map["C_avg5"]))
        any_fail = True

    if ( score_map["N_avg9"] < args.avgnine_cut or score_map["C_avg9"] < args.avgnine_cut ):
        # print("Fail avg9: %6.2f %6.2f"%(score_map["N_avg9"], score_map["C_avg9"]))
        any_fail = True

    if ( score_map["N_avg9_2"] < args.avgnine_two_cut or score_map["C_avg9_2"] < args.avgnine_two_cut ):
        # print("Fail avg9_2: %6.2f %6.2f"%(score_map["N_avg9_2"], score_map["C_avg9_2"]))
        any_fail = True

    # print("")

    if ( any_fail ):
        return False

    # score_seg(npose, is_segment, is_helix, froms, tos, is_core_boundary&first_bundle_mask, first_bundle_mask, score_map, "N_")
    # score_seg(npose, is_segment, is_helix, froms, tos, is_core_boundary&last_bundle_mask, last_bundle_mask, score_map, "C_")

    get_concavity(npose, score_map)

    if ( score_map['angle'] <  args.min_curvature_angle ):
        return False

    get_five_cross(npose, score_map)

    if ( score_map['helix_worst_gap'] > args.helix_worst_gap_cut ):
        return False

    return True


# make this a helper to avoid mistakes
def trim_npose(npose, save_start, save_end):
    return npose[save_start*nu.R:(save_end+1)*nu.R]


def trim_jhr(npose, tag, score_map, string_map):

    is_helix = nu.npose_dssp_helix( npose )
    ss_elements = nu.npose_helix_elements( is_helix )

    if ( not args.allow_nonstandard ):
        if ( len(ss_elements) != 16 ):
            print("Bad secondary structure")
            return None

    if ( not ss_elements[0][0] ):
        print("Doesnt start with helix")
        return None

    if ( not ss_elements[-1][0] ):

        _, last_loop_start, last_loop_end = ss_elements[-1]

        assert(last_loop_end == nu.nsize(npose)-1)

        npose = trim_npose(npose, 0, last_loop_start-1)
        ss_elements = ss_elements[:-1]
    else:
        assert( args.allow_nonstandard)

    if ( len(ss_elements) % 2 != 1 ):
        print("Weird secondary structure, how is this possible?")
    num_input_helices = (len(ss_elements)+1)//2
    max_trim = num_input_helices - 5

    care_mask = np.ones(nu.nsize(npose), np.bool)

    ca_cb = nu.extract_atoms( npose, [nu.CA, nu.CB]).reshape(-1, 2, 4)[...,:3]

    _, _, _, _, neighs = helical_worms.get_avg_sc_neighbors( ca_cb, care_mask )


    # remember, core and boundary positions will change later
    is_core_boundary = neighs > 2

    _, froms, tos, _ = motif_stuff2.motif_score_npose( npose, care_mask, is_core_boundary )

    min_length = args.min_scaff_length
    max_length = args.max_scaff_length

    ncuts = [0, 1, 2]
    ccuts = [0, 1, 2]

    to_ret = []

    save_npose = npose
    save_froms = froms
    save_tos = tos

    icut = 0
    for ncut, ccut in itertools.product(ncuts, ccuts):
        if ( ncut + ccut > max_trim ):
            continue
        icut += 1
        npose = save_npose.copy()
        froms = save_froms.copy()
        tos = save_tos.copy()

        is_false, first_remove, end = ss_elements[-ccut*2]
        c_removed = 0
        if ( ccut > 0 ):
            assert( not is_false )
            c_removed = nu.nsize(npose) - first_remove
            npose = trim_npose(npose, 0, first_remove-1)

        removed = 0
        if ( ncut > 0 ):
            is_false, start, last_remove = ss_elements[-1+ncut*2]
            assert( not is_false )
            npose = trim_npose(npose, last_remove+1, nu.nsize(npose)-1)
            removed = last_remove+1

        froms -= removed
        tos -= removed

        oob = (tos < 0) | (tos >= nu.nsize(npose)) | (froms < 0) | (froms >= nu.nsize(npose))

        froms = froms[~oob]
        tos = tos[~oob]


        ca_cb = nu.extract_atoms(npose, [nu.CA, nu.CB]).reshape(-1, 2, 4)[...,:3]
        # nu.dump_npdb(npose, "%i.pdb"%(icut))

        # print("icut: %i "%icut + "_n%i_c%i"%(ncut, ccut))
        
        old_size = nu.nsize(npose)
        npose, ntrim, ctrim = trim_this_pose(npose, froms, tos, min_length, max_length)

        if ( npose is None ):
            continue

        # this is what happened
        # npose = trim_npose(npose, ntrim+1, ctrim-1)
        removed += ntrim + 1
        c_removed += old_size - ctrim

        # args.debug = ncut==1 and ccut==2

        froms = froms - ntrim - 1
        tos = tos - ntrim - 1
        oob = (tos < 0) | (tos >= nu.nsize(npose)) | (froms < 0) | (froms >= nu.nsize(npose))
        tos = tos[~oob]
        froms = froms[~oob]

        score_map = {}
        passes = score_npose(npose, froms, tos, score_map)

        score_map['trimmed_N'] = removed
        score_map['trimmed_C'] = c_removed

        if ( passes ):
            to_ret.append([npose, tag + "_n%i_c%i"%(ncut, ccut), score_map, {}])



    return to_ret












if ( silent != "" or args.force_silent ):
    open_silent = open("out.silent", "w")

work_care_mask = np.zeros((400), np.bool)

first_score = True
first_silent = True

start = 0
if ( os.path.exists("ckpt")):
    with open("ckpt") as f:
        try:
            start = int(f.read())
            print("Starting at checkpoint %i"%start)
        except:
            pass

for ipdb in range(start, len(pdbs)):
    with open("ckpt", "w") as f:
        f.write(str(ipdb))
    t0 = time.time()
        # try:
    for k in [1]:

        npose, tag = get_ipdb(ipdb)

        score_map = {}
        string_map = {}

        out_stuff = trim_jhr(npose, tag, score_map, string_map)

        if ( not out_stuff is None):

            to_iterate = [(out_stuff, tag, score_map, string_map)]

            if ( isinstance(out_stuff, list) ):
                to_iterate = out_stuff

            for npose_out, tag_out, score_map_out, string_map_out in to_iterate:

                open_score = open("score.sc", "a")
                nu.add_to_score_file_open(tag_out, open_score, first_score, score_map_out, string_map_out )
                open_score.close()
                first_score = False


                if ( silent == "" and not args.force_silent):
                    nu.dump_npdb(npose_out, tag_out + ".pdb")
                else:
                    nu.add_to_silent_file_open( npose_out, tag_out, open_silent, first_silent, score_map_out, string_map_out)
                    first_silent = False


        seconds = int(time.time() - t0)

        print("protocols.jd2.JobDistributor: " + tag + " reported success in %i seconds"%seconds)


    # except Exception as e:
    #     print("Error!!!")
    #     print(e)



if ( silent != "" or args.force_silent ):
    open_silent.close()

















