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
Makes an animation of the population, forwards in time.
"""

parser = argparse.ArgumentParser(prog=sys.argv[0], description=usage)
parser.add_argument("treefile")
parser.add_argument("-o", "--outfile", type=str)
args = parser.parse_args()

if args.outfile is None:
    outbase = ".".join(args.treefile.split(".")[:-1])
    args.outfile = outbase + ".pop.mp4"

ts = tskit.load(args.treefile)

params = ts.metadata['SLiM']['user_metadata']
width = params['WIDTH'][0]
height = params['HEIGHT'][0]
size = (8 * max(1.0, width/height), 8 * max(1.0, height/width))

fig, ax = plt.subplots(figsize=size)
animation = sps.animate_individuals(fig, ts)
animation.save(args.outfile, writer='ffmpeg')
