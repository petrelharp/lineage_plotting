#!/usr/bin/env python3

import sys
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
Usage:
    {} (treefile) [num indivs=3] [num positions = 1]
where the KEY=VALUE pairs get passed to SLiM.
""".format(sys.argv[0])

num_indivs = 3
num_positions = 1
if len(sys.argv) < 2:
    raise ValueError(usage)
elif len(sys.argv) >= 3:
    num_indivs = int(sys.argv[2])
elif len(sys.argv) >= 4:
    num_positions = int(sys.argv[3])
elif len(sys.argv) > 4:
    raise ValueError(usage)

treefile = sys.argv[1]
outbase = ".".join(treefile.split(".")[:-1])

ts = sps.SpatialSlimTreeSequence(tskit.load(treefile), dim=2)

params = ts.metadata['SLiM']['user_metadata']
dt = params['DT'][0]
width = params['WIDTH'][0]
height = params['HEIGHT'][0]
size = (8 * max(1.0, width/height), 8 * max(1.0, height/width))
fig, ax = plt.subplots(figsize=size)

today = np.where(ts.individual_times == 0)[0]
animation = sps.animate_lineage(
    fig,
    ts, 
    children=np.random.choice(today, num_indivs), 
    positions=np.random.randint(0, ts.sequence_length - 1, num_positions),
    time_ago_interval=(0.0, np.max(ts.individual_times)),
    dt=dt,
)
animation.save(outbase + ".lineages.mp4", writer='ffmpeg')

