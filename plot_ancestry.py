#!/usr/bin/env python3

import sys
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import tskit

usage = """
Makes plots of the locations of ancestors of the most north- and southward
individuals at each point in the past for which we have recorded individuals.

Usage:
    {} (treefile)
""".format(sys.argv[0])

if len(sys.argv) < 2 or len(sys.argv) > 3:
    raise ValueError(usage)

treefile = sys.argv[1]
outbase = ".".join(treefile.split(".")[:-1])

ts = tskit.load(treefile)
params = ts.metadata['SLiM']['user_metadata']

indiv_times = ts.individual_times
indiv_pops = ts.individual_populations
indiv_locs = ts.individual_locations

modern = np.logical_and(
        indiv_times < 100,
        indiv_pops == 1
)
target_indivs = [
    np.where(np.logical_and(
        modern,
        indiv_locs[:, 1] == np.max(indiv_locs[modern, 1]),
    ))[0][0],
    np.where(np.logical_and(
        modern,
        indiv_locs[:, 1] == np.min(indiv_locs[modern, 1]),
    ))[0][0],
]

node_anc = ts.sample_count_stat(
        [ts.individual(n).nodes for n in target_indivs],
        lambda x: x/2, # 2 for diploidy
        2,
        polarised=True,
        strict=False,
        mode='node'
)

indiv_anc = np.zeros((ts.num_individuals, len(target_indivs)))
for n in ts.nodes():
    if n.individual >= 0:
        indiv_anc[n.individual] += node_anc[n.id]


xlim = (min(indiv_locs[:,0]), max(indiv_locs[:,0]))
ylim = (min(indiv_locs[:,1]), max(indiv_locs[:,1]))
tts = list(set(indiv_times))
tts.sort()

for k, ind in enumerate(target_indivs):
    fig, axs = plt.subplots(int(np.ceil(len(tts)/3)), 3, figsize=(6, 6 * (ylim[1] - ylim[0]) / (xlim[1] - xlim[0])))
    j = 0
    for axlist in axs:
        for ax in axlist:
            if j >= len(tts):
                break
            target_time = tts[j]
            anc_indivs = np.where(np.logical_and(
                indiv_times == target_time,
                indiv_pops == 1 # exclude dummy indiv in pop 2
            ))[0]
            assert len(anc_indivs) > 0
            ax.set_title(f"time ago = {target_time}")
            ax.set_aspect(1)
            ax.set_xlim(*xlim)
            ax.set_ylim(*ylim)
            # sps.plot_density(ts, time, ax, scatter=False)
            anc_props = indiv_anc[anc_indivs, k]
            ax.scatter(
                    indiv_locs[anc_indivs, 0],
                    indiv_locs[anc_indivs, 1], 
                    sizes=200 * anc_props / np.max(anc_props),
                    facecolor='black',
                    edgecolor=None,
                    alpha=0.75,
           )
            j += 1
    plt.tight_layout()
    fig.savefig(outbase + f".ancestry_locations.{ind}.png", bbox_inches='tight')
