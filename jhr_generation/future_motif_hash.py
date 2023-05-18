#!/usr/bin/env python

import os
import sys

import getpy
import numpy as np
import itertools
import xbin


from numba import njit

data_dir = os.path.dirname(os.path.realpath(__file__)) + "/../jhr_generation_dbs/"


motif_dict = getpy.Dict(np.dtype('uint64'), np.dtype('uint64'), 0)
# motif_dict.load("/home/bcov/from/derrick/getpy_motif/motif_dict_1.0_15_-1.0_H-H.dump")
keys = np.load(open(data_dir + "motif_dict_1.0_15_-1.0_H-H_keys.dump", 'rb'), allow_pickle=False)
values = np.load(open(data_dir + "motif_dict_1.0_15_-1.0_H-H_values.dump", 'rb'), allow_pickle=False)
motif_dict[keys] = values
del keys
del values

motif_masks = np.load(open(data_dir + "motif_array_1.0_15_-1.0_H-H.dump", 'rb'), allow_pickle=False)

binner = xbin.XformBinner(1.0, 15, 512)

def pair2mask( a, b ):

    a = a.astype(np.uint64)
    b = b.astype(np.uint64)

    value = a*20+b

    # least_signicant_encoding
    mask = np.zeros((len(a), 7), np.uint64)

    bit = value % 63

    value //= 63
    byte = value

    bit_mask = np.left_shift( np.uint64(1), bit.astype(np.uint64), dtype=np.uint64 )

    pairs = np.zeros((len(a), 2), np.uint64)
    pairs[:,0] = np.arange(0, len(a), dtype=np.uint64).astype(np.uint64)

    pairs[:,1] = byte

    mask[tuple(pairs.T)] = bit_mask

    return mask


def pairs2mask( a, b , both_directions=False ):

    mask = np.array([0,0,0,0,0,0,0], dtype=np.uint64)
    for i, j in zip(a, b):
        forward_mask = pair2mask(np.array([i], dtype=np.uint64), np.array([j], dtype=np.uint64))
        mask = np.bitwise_or(forward_mask, mask)

        if both_directions:
            reverse_mask = pair2mask(np.array([j], dtype=np.uint64), np.array([i], dtype=np.uint64))
            mask = np.bitwise_or(reverse_mask, mask)

    return mask


def mask2matrix( masks ):

    out = np.zeros((len(masks), 20, 20), np.bool)

    count = 0
    for byte in range(7):
        bit_mask = np.uint64(1)
        for bit in range(63):

            large_ind = count // 20
            small_ind = count % 20

            out[:,large_ind,small_ind] = np.bitwise_and( masks[:,byte], bit_mask, dtype=np.uint64 ) > 0

            bit_mask <<= np.uint64(1)
            count += 1

            if ( count >= 400 ):
                break
        if ( count >= 400 ):
            break

    return out

@njit (fastmath = True)
def fill_masks(motif_masks, indexes, masks):
    for i in range(len(indexes)):
        index = indexes[i]
        motif_masks[index] = np.bitwise_or(motif_masks[index], masks[i])


aa = "ACDEFGHIKLMNPQRSTVWY"
aa2num = {}
for i, a in enumerate(aa):
    aa2num[a] = i


# takes 2 strings
def get_search_mask(these, by_these):
    num_these = [aa2num[x] for x in these]
    num_by_these = [aa2num[x] for x in by_these]
    combos = list(itertools.product(num_these, num_by_these))

    masks = pairs2mask(*list(zip(*combos)))

    return np.bitwise_or.reduce(masks, axis=-2)


def get_raw_hits(xforms):

    keys = binner.get_bin_index(xforms.reshape(-1, 4, 4)).astype(np.uint64)

    raw_hits = motif_masks[motif_dict[keys]]
    
    return raw_hits

def get_masked_hits(xforms, search_mask):
    raw_hits = get_raw_hits(xforms)

    return np.any( np.bitwise_and(raw_hits, search_mask), axis=-1 )



npose_dir = "/home/bcov/sc/random/npose"

def dump_n_x_vs_x_hashes(n, x, by_x, name, random_size=0):

    reverse = getpy.Dict(np.dtype('uint64'), np.dtype('uint64'), 0)
    keys = np.array(list(motif_dict))
    values = motif_dict[keys]
    reverse[values] = keys

    search_mask = get_search_mask(x, by_x)

    indices = np.where(np.any(np.bitwise_and(motif_masks, search_mask), axis=-1))[0]

    xbin_hashes = reverse[indices.astype(np.uint64)]

    assert(not np.any(xbin_hashes == 0))

    xforms = binner.get_bin_center(xbin_hashes)
    xforms[:,:3,3] += (np.random.random((len(xforms), 3)) - 0.5) * random_size

    pts = np.r_[ np.array([[0, 0, 0]]), xforms[:n, :3, 3]]


    if (npose_dir not in sys.path):
        sys.path.append(npose_dir)

    import npose_util as nu

    nu.dump_pts(pts, name)



















# import xarray as xr

# sys.path.append("/home/bcov/from/derrick/getpy_motif/")
# db = pickle.load(open("/home/sheffler/debug/derp_learning/datafiles/pdb_res_pair_data.pickle", 'rb'))
# db = xr.open_zarr("/home/bcov/from/derrick/getpy_motif/pdb_res_pair_data.zarr")

# which_pairs = (db['p_etot'] < -1.5) & (db['ssid'][db['p_resi']] == 1) & (db['ssid'][db['p_resj']] == 1)

# res_j = db['p_resj'][which_pairs]
# res_i = db['p_resi'][which_pairs]
# aa_i = db['aaid'][db['p_resi']][which_pairs]
# aa_j = db['aaid'][db['p_resj']][which_pairs]
# xijbin = np.array(db["xijbin_0.5_7.5"][which_pairs])
# xjibin = np.array(db["xjibin_0.5_7.5"][which_pairs])
# masks = fmh.pair2mask(aa_i, aa_j)
# motif_dict = getpy.Dict(np.dtype('uint64'), np.dtype('uint64'))
# all_keys = np.unique(np.concatenate((xijbin, xjibin))).astype(np.uint64)
# all_indx = np.arange(1, len(all_keys)+1,dtype=np.uint64)
# motif_dict[all_keys] = all_indx
# motif_masks = np.zeros((len(all_keys), 7), dtype=np.uint64)
# xijidx = motif_dict[xijbin.astype(np.uint64)]
# xjiidx = motif_dict[xjibin.astype(np.uint64)]
# fmh.fill_masks(motif_masks, xijidx, masks)
# fmh.fill_masks(motif_masks, xjiidx, masks)


# # test #
# for matrix in mask2matrix(motif_masks[motif_dict[all_keys[:100]]]):
#     print()
#     print(np.array(np.where(matrix)).T)
# # test #

# keys = all_keys[:100]
# key_hits = motif_dict.__contains__(keys)
# value_hits = motif_masks[motif_dict[keys[key_hits]]]

# filter_mask = pairs2mask([0, 9, 17], [9, 9, 9], both_directions=True)

# np.sum(np.any(np.bitwise_and(value_hits, filter_mask), axis=1))

"""
motif_dict.dump("/home/drhicks1/motif_dict.dump")
motif_dict = getpy.Dict(np.dtype('uint64'), np.dtype('uint64'))
motif_dict.load("/home/drhicks1/motif_dict.dump")
"""

"""
np.save(open("/home/drhicks1/motif_array.dump", 'wb'), motif_masks, allow_pickle=False)
motif_masks = np.load(open("/home/drhicks1/motif_array.dump", 'rb'), allow_pickle=False)
"""


















