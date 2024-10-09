import pyslim, tskit
import numpy as np
import tqdm
import scipy.sparse as sparse

class SpatialSlimTreeSequence(tskit.TreeSequence):

    def __init__(self, ts, dim=3):
        super(SpatialSlimTreeSequence, self).__init__(ts._ll_tree_sequence)
        self.dim = dim

    def individual_distance_to_point(self, point):
        """
        Returns the array of distances of each individual's location to the point
        whose coordinates are given by `point`.

        :param float point: The coordinates of the point. If this has length less than
            three, remaining coordinates will be set to 0.0.
        """
        point = np.concatenate((np.array(point), np.zeros(self.dim - len(point))))
        if point.shape != (self.dim,):
            raise ValueError("point not of the correct shape: must be coercible to"
                             + "a vector of length self.dim or less.")

        return np.sqrt(np.sum((self.individual_locations.reshape((self.num_individuals,3)) - point) ** 2, axis=1))

    def individuals_in_circle(self, center, radius, time=None):
        """
        Returns the IDs of individuals at distance less than or equal to `radius` of `center.

        :param float center: The coordinates of the center.
        :param float radius: The radius of the circle.
        :param float time: The time to look at; defaults to no constraint.
        """
        dists = self.individual_distance_to_point(center)
        if time is None:
            out, = np.where(dists <= radius)
        else:
            alive = pyslim.individuals_alive_at(self, time)
            out = alive[dists <= radius]
        return out

    def node_children_dict(self, left=0.0, right=None):
        """
        Return a dict whose keys are nodes and whose values are lists of node IDs
        of their children.

        :param float left: The left end of the portion of genome considered.
        :param float right: The right end of the portion of genome considered.
            Defaults to the sequence length.
        """
        if right is None:
            right = self.sequence_length
        if left < 0 or right > self.sequence_length or left > right:
            raise ValueError("Illegal left, right bounds.")
        edges = self.tables.edges
        out = {n:[] for n in range(self.num_nodes)}
        for edge in ts.edges():
            if edges.left < right and edges.right >= left:
                out[e.parent].append(e.child)
        return out

    def node_parents_dict(self, left=0.0, right=None):
        """
        Return a dict whose keys are nodes and whose values are lists of node IDs
        of their parents.

        :param float left: The left end of the portion of genome considered.
        :param float right: The right end of the portion of genome considered.
            Defaults to the sequence length.
        """
        if right is None:
            right = self.sequence_length
        if left < 0 or right > self.sequence_length or left > right:
            raise ValueError("Illegal left, right bounds.")
        edges = self.tables.edges
        out = {n:[] for n in range(self.num_nodes)}
        for edge in ts.edges():
            if edges.left < right and edges.right >= left:
                out[e.child].append(e.parent)
        return out

    def individual_children_dict(self, left=0.0, right=None):
        """
        Return a dict whose keys are individuals and whose values are lists of
        individual IDs of their children.

        :param float left: The left end of the portion of genome considered.
        :param float right: The right end of the portion of genome considered.
            Defaults to the sequence length.
        """
        if right is None:
            right = self.sequence_length
        if left < 0 or right > self.sequence_length or left > right:
            raise ValueError("Illegal left, right bounds.")
        edges = self.tables.edges
        out = {n:[] for n in range(self.num_individuals)}
        for edge in ts.edges():
            if edges.left < right and edges.right >= left:
                parent = ts.node(e.parent).individual
                child = ts.node(e.child).individual
                if (parent is not tskit.NULL
                     and child is not tskit.NULL):
                    out[parent].append(child)
        return out

    def individual_parents_dict(self, left=0.0, right=None):
        """
        Return a dict whose keys are individuals and whose values are lists of
        individual IDs of their parents.

        :param float left: The left end of the portion of genome considered.
        :param float right: The right end of the portion of genome considered.
            Defaults to the sequence length.
        """
        if right is None:
            right = self.sequence_length
        if left < 0 or right > self.sequence_length or left > right:
            raise ValueError("Illegal left, right bounds.")
        edges = self.tables.edges
        out = {n:[] for n in range(self.num_individuals)}
        for edge in self.edges():
            if edge.left < right and edge.right >= left:
                parent = self.node(edge.parent).individual
                child = self.node(edge.child).individual
                if (parent is not tskit.NULL
                     and child is not tskit.NULL):
                    out[child].append(parent)
        return out

    def node_children(self, parent_nodes, left=0.0, right=None):
        """
        Returns a list of all (parent, child) pairs of node IDs for the given 
        parents, such that child inherited from parent somewhere in the region
        [left, right).

        :param int parent_nodes: The node IDs of the parents.
        :param float left: The left end of the portion of genome considered.
        :param float right: The right end of the portion of genome considered.
            Defaults to the sequence length.
        """
        if len(parent_nodes) == 0:
            return []
        if max(parent_nodes) >= self.num_nodes or min(parent_nodes) < 0:
            raise ValueError("Parent node ID out of bounds.")
        if right is None:
            right = self.sequence_length
        if left < 0 or right > self.sequence_length or left > right:
            raise ValueError("Illegal left, right bounds.")
        edges = self.tables.edges
        yesthese = np.logical_and(np.isin(edges.parent, parent_nodes),
                                          edges.left < right,
                                          edges.right >= left)
        return zip(edges.parent[yesthese], edges.child[yesthese])

    def node_parents(self, children, left=0.0, right=None):
        """
        Returns an array whose rows are all (parent, child) pairs of node IDs
        for the given children, such that child inherited from parent somewhere
        in the region [left, right).

        :param int children: The node IDs of the children.
        :param float left: The left end of the portion of genome considered.
        :param float right: The right end of the portion of genome considered.
            Defaults to the sequence length.
        """
        if len(children) == 0:
            return []
        if max(children) >= self.num_nodes or min(children) < 0:
            raise ValueError("Node child index out of bounds.")
        if right is None:
            right = self.sequence_length
        if left < 0 or right > self.sequence_length or left > right:
            raise ValueError("Illegal left, right bounds.")
        edges = self.tables.edges
        yesthese = np.logical_and(np.isin(edges.child, children),
                                          edges.left < right,
                                          edges.right >= left)
        return np.column_stack((edges.parent[yesthese], edges.child[yesthese]))

    def individual_parents(self, children, time=None, left=0.0, right=None):
        """
        Returns an array whose rows are all (parent, child) pairs of individual
        IDs for the given children, such that child inherited from parent
        somewhere in the region [left, right), and the parent is alive at the
        given time.

        :param int children: The individual IDs of the children.
        :param float time: The time ago the parent should be alive. Defaults to
            no constraint.
        :param float left: The left end of the portion of genome considered.
            Defaults to zero.
        :param float right: The right end of the portion of genome considered.
            Defaults to the sequence length.
        """
        if len(children) == 0:
            return []
        if max(children) >= self.num_individuals or min(children) < 0:
            raise ValueError("Individual child index out of bounds.")
        child_nodes = self.individual_nodes(children, flatten=True)
        node_parents = self.node_parents(child_nodes, left=left, right=right)
        indiv_parents = np.column_stack((self.nodes_individual[node_parents[:, 0]],
                                         self.nodes_individual[node_parents[:, 1]]))
        yesthese = np.logical_and(indiv_parents[:,0] != tskit.NULL,
                                  indiv_parents[:,1] != tskit.NULL)
        if time is not None:
             yesthese = np.logical_and(yesthese,
                                       np.isin(indiv_parents[:,1],
                                               pyslim.individuals_alive_at(self, time)))
        return indiv_parents[yesthese, :]

    def individual_nodes(self, individuals, flatten=True):
        """
        Return the list of nodes associated with these individuals,
        either as a list of lists, or as a single flat list.

        :param iterable individuals: The list of individual IDs.
        """
        nodes = [self.individual(i).nodes for i in individuals]
        if flatten:
            nodes = [x for y in nodes for x in y]
        return nodes

    def relatedness_matrix(self, left=0.0, right=None):
        """
        Constructs the sparse matrix whose [i,j]th entry gives the amount that node j
        inherited *directly* from node i, i.e., the sum of the length of all edges
        that have i as a parent and j as a child.
        """
        if right is None:
            right = self.sequence_length
        edges = self.tables.edges
        R = sparse.coo_matrix((np.fmin(right, edges.right) - np.fmax(left, edges.left), 
                               (edges.parent, edges.child)), 
                               shape = (self.num_nodes, self.num_nodes), dtype = 'float')
        return R.tocsc()


    def relatedness(self, focal_nodes, max_hops):
        """
        For each node, find the smallest number of genealogical hops to one of focal_nodes.
        """
        X = (self.relatedness_matrix() > 0)
        Xt = X.transpose()
        out = np.repeat(np.inf, self.num_nodes)
        out[focal_nodes] = 0
        x = np.repeat(0.0, self.num_nodes)
        x[focal_nodes] = 1.0
        for n in range(1, max_hops + 1):
            # n is the number of up-hops
            x = X.dot(x)
            y = x.copy()
            out[y > 0] = np.fmin(out[y > 0], n)
            for k in range(1, max_hops + 1 - n):
                # k is the number of down-hops
                y = Xt.dot(y)
                # now y[j] is the number of paths of length n + k 
                #  that go from any focal node to j.
                out[y > 0] = np.fmin(out[y > 0], n + k)
        return out

    def proportion_ancestry_nodes(self, sample_sets, show_progress=False):
        """
        Computes for each node the proportion of the genomes in each of sample
        sets inheriting from that node. In other words, if the output is A,
        then A[i,j] is the proportion of sample_sets[i]'s ancestry contributed
        by node j.

        NOTE: instead, use e.g.,
        ts.sample_count_stat([[0], [1]], lambda x: x, 2, polarised=True, strict=False, mode='node')
        to do this!
        """
        for sample_set in sample_sets:
            U = set(sample_set)
            if len(U) != len(sample_set):
                raise ValueError("Cannot have duplicate values within set")

        K = len(sample_sets)
        A = np.zeros((K, self.num_nodes))
        parent = np.zeros(self.num_nodes, dtype=int) - 1
        sample_count = np.zeros((K, self.num_nodes), dtype=int)
        last_update = np.zeros(self.num_nodes)

        def update_counts(edge, sign):
            # Update the counts and statistics for a given node. Before we change the
            # node counts in the given direction, check to see if we need to update
            # statistics for that node. When a node count changes, we add the
            # accumulated statistic value for the span since that node was last updated.
            v = edge.parent
            while v != -1:
                if last_update[v] != left:
                    length = left - last_update[v]
                    for j in range(K):
                        A[j, v] += length * sample_count[j, v]
                    last_update[v] = left
                for j in range(K):
                    sample_count[j, v] += sign * sample_count[j, edge.child]
                v = parent[v]

        # Set the intitial conditions.
        for j in range(K):
            for u in sample_sets[j]:
                sample_count[j][u] = 1

        progress_iter = tqdm.tqdm(
            self.edge_diffs(), total=self.num_trees, disable=not show_progress)
        for (left, right), edges_out, edges_in in progress_iter:
            for edge in edges_out:
                parent[edge.child] = -1
                update_counts(edge, -1)
            for edge in edges_in:
                parent[edge.child] = edge.parent
                update_counts(edge, +1)

        # Finally, add the stats for the last tree
        for v in range(self.num_nodes):
            length = self.sequence_length - last_update[v]
            for j in range(K):
                A[j, v] += length * sample_count[j, v]
        # renormalize
        for j in range(K):
            A[j,:] /= len(sample_sets[j]) * self.sequence_length
        return A

    def proportion_ancestry_nodes_slow(self, sample_set, left=0.0, right=None):
        """
        Computes an array of values that gives, for each node, the total amount of
        genome in `sample_set` that descent from that node.

        IS VERY SLOOOOOW

        :param int sample_set: The children.
        :param float time: The time ago the parent should be alive. Defaults to
            no constraint.
        :param float left: The left end of the portion of genome considered.
            Defaults to zero.
        :param float right: The right end of the portion of genome considered.
            Defaults to the sequence length.
        """
        n = len(sample_set)
        if right is None:
            right = self.sequence_length
        if left < 0 or right > self.sequence_length or left > right:
            raise ValueError("Illegal left, right bounds.")
        trees = self.trees(sample_counts=True, tracked_leaves=sample_set)
        # first count up how much shared ancestry each node has
        ancestry = np.zeros(self.num_nodes)
        for tree in trees:
            if tree.interval[0] > right:
                break
            while tree.interval[1] < left:
                next
            tree_span = min(right, tree.interval[1]) - max(left, tree.interval[0])
            for n in tree.nodes():
                ancestry[n] += tree.num_tracked_samples(n) * tree_span
        ancestry /= len(sample_set) * ts.sequence_length
        return ancestry
