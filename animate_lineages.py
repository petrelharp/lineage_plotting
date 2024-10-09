#!/usr/bin/env python3

import sys
import argparse
import tskit
import numpy as np
import spatial_slim as sps

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

usage = """
Makes an animation of *lineages* moving back
at a randomly chosen set of positions on the genome
from the chromosomes of the given number of diploid individuals.
"""

parser = argparse.ArgumentParser(prog=sys.argv[0], description=usage)
parser.add_argument("treefile")
parser.add_argument("-o", "--outfile", type=str)
parser.add_argument("-n", "--num_indivs", default=3, type=int)
parser.add_argument("-p", "--num_positions", default=1, type=int)
parser.add_argument("-s", "--seed", type=int)
parser.add_argument("-t", "--dt", type=float, help="conversion factor from ticks to time")
parser.add_argument("-x", "--xmin", type=float, help="minimum x coordinate of sampled lineages")
args = parser.parse_args()

if args.outfile is None:
    outbase = ".".join(args.treefile.split(".")[:-1])
    args.outfile = outbase + ".lineages.mp4"

ts = sps.SpatialSlimTreeSequence(tskit.load(args.treefile), dim=2)

params = ts.metadata['SLiM']['user_metadata']
width = params['WIDTH'][0]
height = params['HEIGHT'][0]

if args.dt is None:
    try:
        args.dt = params['DT']
    except KeyError:
        args.dt = 1.0

if args.xmin is None:
    args.xmin = 0

rng = np.random.default_rng(args.seed)

num_positions = min(args.num_positions, int(ts.sequence_length))
positions = rng.choice(int(ts.sequence_length), args.num_positions)

today,  = np.where(np.logical_and(
        ts.individual_times == 0,
        ts.individual_locations[:,0] >= args.xmin
))
children = np.random.choice(today, args.num_indivs)

size = (8 * max(1.0, width/height), 8 * max(1.0, height/width))
fig, ax = plt.subplots(figsize=size)

animation = sps.animate_lineage(
    fig,
    ts, 
    children=children,
    positions=positions,
    time_ago_interval=(0.0, np.max(ts.individual_times)),
    dt=args.dt,
)
animation.save(args.outfile, writer='ffmpeg')

