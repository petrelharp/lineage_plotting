#!/usr/bin/env python3

import sys
import pyslim, tskit
import numpy as np
import spatial_slim as sps

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

usage = """
Usage:
    {} (treefile) [duration in seconds]
""".format(sys.argv[0])

if len(sys.argv) != 2:
    raise ValueError(usage)

treefile = sys.argv[1]
outbase = ".".join(treefile.split(".")[:-1])

ts = pyslim.load(treefile)

params = ts.metadata['SLiM']['user_metadata']
width = params['WIDTH'][0]
height = params['HEIGHT'][0]
size = (8 * max(1.0, width/height), 8 * max(1.0, height/width))

fig, ax = plt.subplots(figsize=size)
animation = sps.animate_individuals(fig, ts)
animation.save(outbase + ".pop.mp4", writer='ffmpeg')
