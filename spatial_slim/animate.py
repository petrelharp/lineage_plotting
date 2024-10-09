import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as ani
import matplotlib.collections as cs
import pyslim, tskit

def animate_individuals(fig, ts, times=None, ax=None, duration=20):
    """
    Make an animation of the individuals.

    `times` should be in units of forwards (SLiM) time.
    `duration` is in seconds
    """
    if ax is None:
        ax = fig.axes[0]

    num_gens = ts.metadata['SLiM']["tick"]
    if times is None:
        times = num_gens - np.arange(num_gens)

    # don't need too many frames
    while duration / len(times) < 0.1:
        times = times[::2]

    locs = ts.individuals_location.reshape((ts.num_individuals,3))
    xmax = np.ceil(max(locs[:,0]))
    ymax = np.ceil(max(locs[:,1]))
    ax.set_xlim(0, xmax)
    ax.set_ylim(0, ymax)
    colormap = lambda x: plt.get_cmap("cool")(x/max(pyslim.individual_ages(ts)))
    frame = np.max(times)
    inds = pyslim.individuals_alive_at(ts, frame)
    next_inds = pyslim.individuals_alive_at(ts, frame - 1)
    circles = ax.scatter(locs[inds, 0], locs[inds, 1], s=10, 
                         edgecolors=colormap([0 for _ in inds]),
                         facecolors='none')
    filled = ax.scatter(locs[next_inds, 0], locs[next_inds, 1], s=10, 
                        facecolors=colormap([0 for _ in next_inds]),
                        edgecolors='none')

    def update(frame):
        inds = pyslim.individuals_alive_at(ts, frame)
        next_inds = pyslim.individuals_alive_at(ts, frame - 1)
        circles.set_offsets(locs[inds,:2])
        filled.set_offsets(locs[next_inds,:2])
        # color based on age so far
        circles.set_color(colormap(pyslim.individual_ages_at(ts, frame)[inds]))
        return circles, filled

    # interval is an integer number of milliseconds
    interval = int(1000 * duration / (len(times) - 1))
    animation = ani.FuncAnimation(fig, update, frames=times, interval=interval)
    return animation


def animate_lineage(fig, ts, children, positions, time_ago_interval=None, ax=None, duration=20, dt=1.0):
    """
    An animation of the lineages ancestral to the given children
    at the given positions along the genome.
    Here `time_ago_interval` should be in units of SLiM time units *ago*.
    """
    if ax is None:
        ax = fig.axes[0]
    num_gens = ts.metadata["SLiM"]["tick"]
    if time_ago_interval is None:
        time_ago_interval = (0.0, num_gens - 1)

    # don't need too many frames
    times = np.arange(time_ago_interval[0], time_ago_interval[1])
    while duration / len(times) < 0.1:
        times = times[::2]

    locs = ts.individuals_location.reshape((ts.num_individuals, 3))
    xmax = np.ceil(max(locs[:,0]))
    ymax = np.ceil(max(locs[:,1]))
    ax.set_xlim(0, xmax)
    ax.set_ylim(0, ymax)
    colormap = lambda x: plt.get_cmap("cool")(x/max(pyslim.individual_ages(ts)))
    treecolors = [plt.get_cmap("viridis")(x) for x in np.linspace(0, 1, len(positions))]

    inds = pyslim.individuals_alive_at(ts, time_ago_interval[0])
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
        pathcolors.extend([c] * len(positions))
    lc = cs.LineCollection([], linewidths=0.5, colors=pathcolors)
    ax.add_collection(lc)
    ax.set_title(f"t = {(num_gens - time_ago_interval[0])*dt:.2f} (time {time_ago_interval[0]*dt:.2f} ago)")

    def update(frame):
        inds = pyslim.individuals_alive_at(ts, frame)
        circles.set_offsets(locs[inds,:2])
        # color based on age so far
        circles.set_color(colormap(pyslim.individual_ages_at(ts, frame)[inds]))
        show_paths = []
        for path in paths:
            dothese = (path[:, 0] <= frame)
            show_paths.append(path[dothese, 1:])
        lc.set_paths(show_paths)
        ax.set_title(f"t = {(num_gens - frame)*dt:.2f} (time {frame*dt:.2f} ago)")
        return circles, lc

    animation = ani.FuncAnimation(fig, update, frames=times)
    return animation

