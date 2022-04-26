import numpy as np
import tskit
import matplotlib

def _break_path(path, W):
    """
    Split up the path any time it jumps by more than W/2
    (as it would if it went around the end of a periodic region).
    """
    diffs = np.diff(path[:, 0])
    db = 1 + np.where(np.abs(diffs) > W/2)[0]
    breaks = list(set([0] + list(db) + [path.shape[0]]))
    breaks.sort()
    for j in range(len(breaks) - 1):
        x = path[breaks[j]:breaks[j+1], :]
        if j > 0:
            a = path[breaks[j] - 1, :].copy()
            b = path[breaks[j], :].copy()
            a[0] = 0 if b[0] < W/2 else W
            a[1] = (a[1] + b[1]) / 2
            x = np.row_stack([a, x])
        if j < len(breaks) - 2:
            a = path[breaks[j+1] - 1, :].copy()
            b = path[breaks[j+1], :].copy()
            b[0] = 0 if a[0] < W/2 else W
            b[1] = (a[1] + b[1]) / 2
            x = np.row_stack([x, b])
        yield x


def get_lineages(ts, children, positions, max_time_ago):
    """
    A dictionary of the lineages ancestral to the given children
    at the given positions.
    Keys are (node id, position on the genome).
    """
    locs = ts.individual_locations
    # will record here tuples of the form (time, x)
    nodes = np.concatenate([ts.individual(i).nodes for i in children])
    node_times = ts.tables.nodes.time
    # careful: some are tskit.NULL
    node_indivs = ts.tables.nodes.individual
    has_parents = ts.has_individual_parents()
    paths = {}
    for p in positions:
        tree = ts.at(p)
        for n in nodes:
            out = [np.array([locs[node_indivs[n], 0], node_times[n]])]
            u = tree.parent(n)
            while u is not tskit.NULL:
                uind = node_indivs[u]
                if (uind is tskit.NULL
                        or ts.node(u).time > max_time_ago
                        or not has_parents[uind]):
                    break
                out.append(np.array([locs[uind, 0], node_times[u]]))
                u = tree.parent(u)
            out = np.row_stack(out)
            paths[(n, p)] = out
    return paths


def lineage_paths(ax, ts, children, positions, max_time_ago, periodic=False, width=None, time_on_x=True, dt=None):
    """
    Returns a collection of lines tracing the lineages ancestral to the given
    children at the given positions, up until max_time ago. Times are in *forwards* time, 
    from max_time_ago (times dt, if present).
    """
    path_dict = get_lineages(ts, children, positions, max_time_ago)
    locs = ts.individual_locations
    treecolors = {p : matplotlib.pyplot.get_cmap("viridis")(x)
                  for p, x in zip(positions, np.linspace(0, 1, len(positions)))}
    paths = []
    pathcolors = []
    for u, p in path_dict:
        birth_time = ts.node(u).time
        x0 = ts.individual(ts.node(u).individual).location[0]
        this_paths = [
                np.array([
                     [x0, 0.0],
                     [x0, birth_time]
                 ])
        ]
        if periodic:
            this_paths.extend(_break_path(path_dict[(u, p)], width))
        else:
            this_paths.extend([path_dict[(u, p)]])
        for this_path in this_paths:
            this_path[:, 1] = max_time_ago - this_path[:, 1]
            if dt is not None:
                this_path[:, 1] *= dt
            if time_on_x:
                this_path = this_path[:, (1,0)]
            paths.append(this_path)
            pathcolors.append(treecolors[p])
    lc = matplotlib.collections.LineCollection(paths, linewidths=1.5, colors=pathcolors)
    return lc
