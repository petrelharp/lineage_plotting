#!/usr/bin/env python3

import sys
import pyslim, tskit
import numpy as np
import spatial_slim as sps

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.animation as ani
import matplotlib.collections as cs

usage = """
Makes an animation of *lineages* moving back for the given number of
time steps at a randomly chosen set of positions on the genome
from the chromosomes of the given number of diploid individuals.
Usage:
    {} (num times) (num indivs) (num positions) (script name) [KEY=VALUE [KEY=VALUE]]
where the KEY=VALUE pairs get passed to SLiM.
""".format(sys.argv[0])

if len(sys.argv) < 5:
    raise ValueError(usage)

try:
    num_steps = int(sys.argv[1])
except ValueError:
    raise ValueError("First argument should be numeric (a number of time steps).")
try:
    num_indivs = int(sys.argv[2])
except ValueError:
    raise ValueError("Second argument should be numeric (a number of individuals' lineages to plot).")
try:
    num_positions = int(sys.argv[3])
except ValueError:
    raise ValueError("Third argument should be numeric (a number of positions to plot lineages at).")
script = sys.argv[4]
kwargs = {}
for kv in sys.argv[5:]:
    k, v = kv.split("=")
    kwargs[k] = v

if 'seed' not in kwargs:
    kwargs['seed'] = np.random.randint(10000)

def animate_lineage(ts, children, positions, num_steps):
    """
    An animation of the lineages ancestral to the given children
    at the given positions.
    """
    fig = plt.figure(figsize=(9,9))
    ax = fig.add_subplot(111)
    locs = ts.individual_locations
    xmax = np.ceil(max(locs[:,0]))
    ymax = np.ceil(max(locs[:,1]))
    ax.set_xlim(0, xmax)
    ax.set_ylim(0, ymax)
    colormap = lambda x: plt.get_cmap("cool")(x/max(ts.individual_ages))
    treecolors = [plt.get_cmap("viridis")(x) for x in np.linspace(0, 1, len(positions))]

    inds = ts.individuals_alive_at(0)
    circles = ax.scatter(locs[inds, 0], locs[inds, 1], s=10, 
                         edgecolors=colormap([0 for _ in inds]),
                         facecolors='none')
    # will record here tuples of the form (time, x, y)
    # and will later add the given lines at the corresponding times
    nodes = ts.individual_nodes(children)
    node_times = ts.tables.nodes.time
    node_indivs = ts.tables.nodes.individual
    paths = []
    for p in positions:
        tree = ts.at(p)
        for u in nodes:
            out = [np.concatenate([[node_times[u]], locs[node_indivs[u], :2]])]
            u = tree.parent(u)
            while u is not tskit.NULL:
                uind = node_indivs[u]
                if uind is tskit.NULL:
                    break
                out.append(np.concatenate([[node_times[u]], locs[uind, :2]]))
                u = tree.parent(u)
            paths.append(np.row_stack(out))
    pathcolors = []
    for c in treecolors:
        pathcolors.extend([c] * num_positions)
    lc = cs.LineCollection([], linewidths=0.5, colors=pathcolors)
    ax.add_collection(lc)

    def update(frame):
        inds = ts.individuals_alive_at(frame)
        circles.set_offsets(locs[inds,:2])
        # color based on age so far
        circles.set_color(colormap(ts.individual_ages_at(frame)[inds]))
        show_paths = []
        for path in paths:
            dothese = (path[:, 0] <= frame)
            show_paths.append(path[dothese, 1:])
        lc.set_paths(show_paths)
        return circles, lc

    animation = ani.FuncAnimation(fig, update, 
                                  frames=np.linspace(0, num_steps, num_steps + 1))
    return animation

treefile = sps.run_slim(script = script, **kwargs)
outbase = ".".join(treefile.split(".")[:-1])

ts = sps.SpatialSlimTreeSequence(pyslim.load(treefile), dim=2)

today = np.where(ts.individual_times == 0)[0]
animation = animate_lineage(ts, 
                            np.random.choice(today, num_indivs), 
                            np.random.randint(0, ts.sequence_length - 1, num_positions),
                            num_steps)
animation.save(outbase + ".lineages.mp4", writer='ffmpeg')

