import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as ani
import matplotlib.collections as cs

def animate_individuals(fig, ts, times=None, ax=None, duration=20):
    """
    Make an animation of the individuals.

    `times` should be in units of forwards (SLiM) time.
    `duration` is in seconds
    """
    if ax is None:
        ax = fig.axes[0]

    num_gens = ts.metadata['SLiM']["generation"]
    if times is None:
        times = num_gens - np.arange(num_gens)

    # don't need too many frames
    while duration / len(times) < 0.1:
        times = times[::2]

    locs = ts.individual_locations
    xmax = np.ceil(max(locs[:,0]))
    ymax = np.ceil(max(locs[:,1]))
    ax.set_xlim(0, xmax)
    ax.set_ylim(0, ymax)
    colormap = lambda x: plt.get_cmap("cool")(x/max(ts.individual_ages))
    frame = np.max(times)
    inds = ts.individuals_alive_at(frame)
    next_inds = ts.individuals_alive_at(frame - 1)
    circles = ax.scatter(locs[inds, 0], locs[inds, 1], s=10, 
                         edgecolors=colormap([0 for _ in inds]),
                         facecolors='none')
    filled = ax.scatter(locs[next_inds, 0], locs[next_inds, 1], s=10, 
                        facecolors=colormap([0 for _ in next_inds]),
                        edgecolors='none')

    def update(frame):
        inds = ts.individuals_alive_at(frame)
        next_inds = ts.individuals_alive_at(frame - 1)
        circles.set_offsets(locs[inds,:2])
        filled.set_offsets(locs[next_inds,:2])
        # color based on age so far
        circles.set_color(colormap(ts.individual_ages_at(frame)[inds]))
        return circles, filled

    # interval is an integer number of milliseconds
    interval = int(1000 * duration / (len(times) - 1))
    animation = ani.FuncAnimation(fig, update, frames=times, interval=interval)
    return animation


