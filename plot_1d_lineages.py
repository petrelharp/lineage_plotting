#!/usr/bin/env python3

import sys
import pyslim, tskit
import numpy as np
import spatial_slim as sps

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

usage = """
Makes a plot of *lineages* moving back
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

ts = pyslim.load(treefile)

num_gens = ts.metadata['SLiM']['generation']
params = ts.metadata['SLiM']['user_metadata']
dt = params['DT'][0]
width = params['WIDTH'][0]
height = params['HEIGHT'][0]

popsize = pyslim.population_size(
        ts,
        x_bins = np.linspace(0, width, 101),
        y_bins = (0, height),
        time_bins = np.arange(num_gens),
)[:,0,:]

today = ts.individuals_alive_at(0)
has_parents = ts.has_individual_parents()
max_time = np.max(ts.individual_times[has_parents])
if len(today) < num_indivs:
    raise ValueError(f"Not enough individuals: only {len(today)} alive today!")

size = (12, 8)
fig, ax = plt.subplots(figsize=size)
ax.set_xlabel("time ago")
ax.set_ylabel("geographic position")

ax.imshow(
    popsize[:,:int(max_time)],
    extent=(0, max_time, 0, width),
    interpolation='none',
    aspect=(size[1] / size[0]) * (max_time / width),
    cmap="Oranges",
)

children = np.random.choice(today, num_indivs, replace=False)
ax.scatter(
        np.repeat(0.0, num_indivs),
        [ts.individual(i).location[0] for i in children]
)
lc = sps.lineage_paths(ax, ts,
              children=children,
              positions=np.random.randint(0, ts.sequence_length - 1, num_positions),
              max_time=max_time,
              time_first=True,
)
ax.add_collection(lc)

fig.savefig(outbase + ".lineages.png")
plt.close(fig)
