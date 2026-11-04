'''
Build one split of the eleven functional datasets.

    python3 -m plug.split --lohi       pooled: nothing crosses anywhere, nodes dropped
    python3 -m plug.split --datasail   per dataset: sequence leakage minimised inside one

That flag is the only argument. Everything else that defines the experiment is a
constant below, in a tracked file, where a change shows up in a diff.

The two answer different questions. datasail minimises sequence leakage WITHIN
a benchmark and forbids ligand leakage, solving each benchmark on its own subgraph, so it is the honest
per-benchmark comparison and is not set up for training across benchmarks. lohi
is solved over the pooled graph, so a test set is clear of every benchmark's
training set, which is what transfer and fine-tuning need; it pays in retention.

The random controls read the written files and live in plug-analysis.
'''

import csv
import gzip
import hashlib
import io
import json
import platform
import resource
import sys
import time
from collections import Counter
from pathlib import Path

import networkx as nx

from . import config as c
from . import graph as G
from .benchmarks import ELEVEN, REGISTRY
from .external import datasail_cluster_1d
from .external import min_vertex_k_cut as lohi

# -----------------------------------------------------------------------------
# the Lo-Hi balanced vertex minimum k-cut
# -----------------------------------------------------------------------------

PARTS = ("train", "val", "test")
EXCLUDED = "excluded"           # the fourth state, and the only one that is not a part

def build_graph(nodes, edgelist):
    '''the neighbourhood graph: an edge means too alike to be separated'''
    graph = nx.Graph()
    graph.add_nodes_from(nodes)
    for node1, node2 in edgelist:
        if node1 != node2 and node1 in graph and node2 in graph:
            graph.add_edge(node1, node2)
    return graph

def split_components(graph):
    '''
    the component to cut, and the rest sorted largest first.

    Lo-Hi's get_main_component picks the one to cut. the others are sorted here
    rather than left in the order networkx happens to return: the greedy pass
    below depends on the order it sees them in, and that order should come from
    the data and not from a dict's insertion history.
    '''
    giant, rest = lohi.get_main_component(graph)
    return giant, sorted((sorted(component) for component in rest),
                         key=lambda component: (-len(component), component[0]))

def coarsen(graph):
    '''
    Lo-Hi's coarse_graph: cluster each node with its unclustered neighbours,
    biggest neighbourhood first. sound because two neighbours can only be
    separated by dropping one of them, so a coarse solution is always a valid
    fine one.

    two adjustments, neither of them to the method. their function indexes its
    cluster array by node id, so nodes are relabelled 0..n-1 first -- in sorted
    order, which is also the order their stable sort falls back on for a tie, so
    the clustering is a function of the graph and not of its insertion history.
    and their quotient keeps a self-loop per cluster; a self-loop says a cluster
    may not be in two parts at once, which sum_k x_i,k <= 1 already says, so they
    are dropped to keep the model honest about its size.

    they coarsen on a similarity threshold, counting only neighbours above it.
    here an edge exists only above the bar already, so every neighbour counts --
    weight 1.0 against a threshold of 0.5 is that same test, already applied.
    '''
    order = sorted(graph.nodes())
    index_of = {node: i for i, node in enumerate(order)}
    relabelled = nx.Graph()
    relabelled.add_nodes_from(range(len(order)))
    for node1, node2 in graph.edges():
        relabelled.add_edge(index_of[node1], index_of[node2], weight=1.0)

    quotient, cluster_of_index = lohi.coarse_graph(relabelled, 0.5)
    quotient.remove_edges_from(nx.selfloop_edges(quotient))
    return quotient, {node: int(cluster_of_index[index_of[node]]) for node in order}

def coarsen_rounds(graph, rounds):
    '''
    coarsen repeatedly, each round over the last quotient graph.

    one round is Lo-Hi. more is the dial that makes this solvable at all: the
    two-sided band is what makes the model hard, and the search gets
    dramatically easier as the blocks get fewer. it is a trade against RETENTION
    only, never against the guarantee. two rounds is what the pool needs -- at
    one the model sits at a 99% gap, at three the clustering has swallowed the
    component into one block no band can be met with.
    '''
    quotient, cluster_of = coarsen(graph)
    for _ in range(max(rounds - 1, 0)):
        if quotient.number_of_edges() == 0: break
        quotient, merged_into = coarsen(quotient)
        cluster_of = {node: merged_into[cluster] for node, cluster in cluster_of.items()}
    return quotient, cluster_of

def per_part(weight):
    '''
    a weight read as one weight per part.

    a protein is worth the same wherever it goes, so one number says it. a
    MOLECULE is not: a protein-ligand row survives only if the molecule lands in
    the part its protein did, so its weight arrives as one number per part.
    '''
    return (list(weight) if isinstance(weight, (list, tuple))
            else [weight] * len(PARTS))

def fracs(frac, datasets):
    '''the target shares per dataset, from one triple or a dict of them'''
    if isinstance(frac, dict): return [tuple(frac[dataset]) for dataset in datasets]
    return [tuple(frac)] * len(datasets)

def pair_frac(frac, arity):
    '''
    the PROTEIN shares that land these ROW shares, for a dataset whose rows name
    `arity` proteins. a pair survives only when both ends are in one part, so a
    part with a tenth of the proteins gets a hundredth of the pairs; the rule is
    on rows, so an interaction dataset has to aim at the square roots. aiming at
    a tenth is what left ppi's test set at 2%.
    '''
    if arity <= 1: return tuple(frac)
    root = [f ** (1.0 / arity) for f in frac]
    return tuple(r / sum(root) for r in root)

def cluster_weights(members, weights, datasets):
    '''cluster -> dataset -> weight per part, summed over the cluster's nodes'''
    out = []
    for cluster_nodes in members:
        per_dataset = [[0] * len(PARTS) for _ in datasets]
        for node in cluster_nodes:
            for d, dataset in enumerate(datasets):
                weight = (weights.get(node) or {}).get(dataset)
                if weight:
                    for p, one in enumerate(per_part(weight)): per_dataset[d][p] += one
        out.append(per_dataset)
    return out

def tolerance_of(strata_tolerance, key):
    '''one tolerance for every class, or one per key when the keys differ in kind'''
    if isinstance(strata_tolerance, dict): return strata_tolerance[key]
    return strata_tolerance

def solve(quotient, members, weights, datasets, frac, tolerance, standing, solver,
          verbose, strata=None, strata_frac=None, strata_tolerance=0.15, floors=None):
    '''
    the MIP, for cluster i and part k:

        max  sum_i,k  w_i,k x_i,k                      keep as much as possible
        s.t. sum_k x_i,k <= 1                          one part, or dropped
             x_i,k + sum_{k'!=k} x_j,k' <= 1   edge (i,j)
                                                       no edge joins two parts
             lo_d,k <= kept_d,k <= hi_d,k              every dataset banded

    the edge constraint never forbids an edge, it forbids an edge from being
    CUT, so the only way to keep two neighbours apart is to drop one. that is
    why the guarantee is structural rather than audited.

    the band is two-sided and written against each dataset's RETAINED weight,
    which stays linear because a fraction of a sum of variables is a sum of
    variables. the weight already placed outside this solve is added as a
    constant, so the band holds over the finished split and not just this
    component.

    strata are the same constraint written once more per class -- dms's five
    assay types, shs27k's interaction modes -- so that a split cannot be
    leakage-free and the right size while still asking the test set a question
    the training set never covered. they are banded and NOT counted in the
    objective: an shs27k pair is often in several classes at once, so adding them
    to the loss would weigh it by how many classes a row happens to belong to.
    their band is looser than the dataset's, because a class with forty rows in
    it cannot be cut to five points.

    a dataset named in `floors` is balanced on something other than its rows --
    its proteins, passed in as one more stratum -- so its rows lose the two-sided
    band and keep only a floor per part, which stops one huge assay from leaving
    a part with almost nothing to score.
    '''
    import mip

    cluster_weight = cluster_weights(members, weights, datasets)
    want = fracs(frac, datasets)
    strata_keys = sorted(strata_frac or ())
    cluster_strata = cluster_weights(members, strata or {}, strata_keys)
    started = time.time()
    n_clusters, n_parts = len(quotient), len(PARTS)

    model = mip.Model(sense=mip.MAXIMIZE, solver_name=solver)
    in_part = [[model.add_var(var_type=mip.BINARY) for _ in PARTS]
               for _ in range(n_clusters)]
    model.objective = mip.xsum(
        sum(cluster_weight[cluster][d][part] for d in range(len(datasets)))
        * in_part[cluster][part]
        for cluster in range(n_clusters) for part in range(n_parts))
    for one_cluster in in_part: model += mip.xsum(one_cluster) <= 1

    # Lo-Hi writes this as one constraint per ordered pair of parts, six per edge.
    # aggregating over the far end gives three and a tighter relaxation: if x[a][k]
    # is 1 then every other part of b is 0, so b is in k or dropped
    for cluster1, cluster2 in sorted(quotient.edges()):
        for part in range(n_parts):
            others = [other for other in range(n_parts) if other != part]
            model += in_part[cluster1][part] + mip.xsum(
                in_part[cluster2][other] for other in others) <= 1
            model += in_part[cluster2][part] + mip.xsum(
                in_part[cluster1][other] for other in others) <= 1

    for d, dataset in enumerate(datasets):
        placed = standing.get(dataset, [0] * n_parts)
        kept = [mip.xsum(cluster_weight[cluster][d][part] * in_part[cluster][part]
                         for cluster in range(n_clusters)) + placed[part]
                for part in range(n_parts)]
        total = mip.xsum(kept)
        if dataset in (floors or {}):
            for part in range(n_parts):
                model += kept[part] >= floors[dataset][part] * total
            continue
        for part in range(n_parts):
            model += kept[part] >= (want[d][part] - tolerance) * total
            model += kept[part] <= (want[d][part] + tolerance) * total

    # once more per class, looser, and never in the objective
    for k, key in enumerate(strata_keys):
        placed = standing.get(key, [0] * n_parts)
        kept = [mip.xsum(cluster_strata[cluster][k][part] * in_part[cluster][part]
                         for cluster in range(n_clusters)) + placed[part]
                for part in range(n_parts)]
        total = mip.xsum(kept)
        slack = tolerance_of(strata_tolerance, key)
        for part in range(n_parts):
            model += kept[part] >= (strata_frac[key][part] - slack) * total
            model += kept[part] <= (strata_frac[key][part] + slack) * total

    # two parts that want the same share AND start from the same place are
    # interchangeable, and ordering them drops only the mirror image. both halves
    # matter: for molecules the parts pay differently, and if validation starts with
    # more than test then the swapped solution need not even be feasible
    same_everywhere = all(min(per_part_weight) == max(per_part_weight)
                          for per_dataset in cluster_weight
                          for per_part_weight in per_dataset)
    pooled = [mip.xsum(sum(cluster_weight[cluster][d][part] for d in range(len(datasets)))
                       * in_part[cluster][part] for cluster in range(n_clusters))
              for part in range(n_parts)]
    for part in range(n_parts):
        for later in range(part + 1, n_parts):
            if (same_everywhere
                    and all(abs(f[part] - f[later]) < 1e-12 for f in want)
                    and all(placed[part] == placed[later]
                            for placed in standing.values())):
                model += pooled[part] >= pooled[later]

    # one thread, because a parallel branch and bound returns whichever incumbent a
    # race produced. no time limit and no gap tolerance either: an answer that
    # stopped early is an answer whose value depends on how fast the machine was.
    #
    # the gap has to be set, not left alone. python-mip inherits max_mip_gap = 1e-4,
    # which on this objective is some 80 datapoints of retention that no one proved
    # were unreachable. zero is what the sentence above is actually claiming
    model.threads = 1
    model.max_mip_gap = 0.0
    model.verbose = 1 if verbose else 0
    status = model.optimize()

    if status not in (mip.OptimizationStatus.OPTIMAL, mip.OptimizationStatus.FEASIBLE):
        raise RuntimeError(
            f"the k-cut did not solve: the solver says {status.name}. INFEASIBLE means "
            f"no assignment of this component meets the band at tolerance {tolerance} "
            "-- the honest moves are a looser tolerance or a stricter threshold, not a "
            "time limit.")
    # the band is written against retained weight, so dropping everything satisfies it
    # trivially. the objective normally rules that out, so landing there means no
    # non-empty split exists at these shares -- a refusal, not an empty test set
    if model.objective_value is not None and model.objective_value < 1 and any(
            max(per_part_weight) for per_dataset in cluster_weight
            for per_part_weight in per_dataset):
        raise RuntimeError(
            f"the k-cut solved to zero retention: {status.name}. the only way this "
            f"component meets its share +/- {tolerance} is by retaining nothing.")

    where = {}
    for cluster in range(n_clusters):
        chosen = [part for part in range(n_parts)
                  if in_part[cluster][part].x is not None
                  and in_part[cluster][part].x >= 0.5]
        for node in members[cluster]:
            where[node] = PARTS[chosen[0]] if chosen else EXCLUDED
    return where, {"status": status.name, "objective": model.objective_value,
                   "gap": model.gap, "seconds": round(time.time() - started, 1),
                   "clusters": n_clusters, "variables": model.num_cols,
                   "constraints": model.num_rows}

def place(components, weights, datasets, frac, tolerance, strata=None,
          strata_frac=None, strata_tolerance=0.15, floors=None):
    '''
    place whole components, largest first, into the part that wants them most.

    this is Lo-Hi's pass over the small components. a component has no edge out
    of it, so wherever it goes it takes nothing with it and nothing here can
    break the guarantee.

    two questions in order, not added together. where is this component worth
    most? for a protein that is "everywhere equally" and decides nothing, which
    is what lets the balance decide; for a molecule it is the part its proteins
    are in, and no amount of wanting balance should park it where none of its
    rows can be retained. then: which part is furthest below its share. the
    first question is capped, because worth alone would send everything to
    train, where four rows in five are, and leave the two small parts empty.
    '''
    # the stratum keys ride along as extra rows of the same accounting: they get a
    # ceiling and a deficit like a dataset does, but they never enter `worth`, so what
    # a component is worth is still counted once
    strata_keys = sorted(strata_frac or ())
    keys = list(datasets) + strata_keys
    index_of = {key: i for i, key in enumerate(keys)}
    want = fracs(frac, datasets) + [tuple(strata_frac[key]) for key in strata_keys]
    # a dataset with only a floor on its rows has no ceiling on them to respect here
    slack = ([1.0 if dataset in (floors or {}) else tolerance for dataset in datasets]
             + [tolerance_of(strata_tolerance, key) for key in strata_keys])
    weights = {node: {**(weights.get(node) or {}), **((strata or {}).get(node) or {})}
               for node in set(weights) | set(strata or {})}
    n_real = len(datasets)
    standing = {key: [0] * len(PARTS) for key in keys}

    # the most each dataset could still end up with, counted before anything is
    # placed: measured against the running total instead, the first component placed
    # is always 100% of its part and every part looks full
    budget = [0] * len(keys)
    for component in components:
        for node in component:
            for dataset, weight in (weights.get(node) or {}).items():
                budget[index_of[dataset]] += max(per_part(weight))

    where = {}
    for component in components:
        gain = [[0] * len(PARTS) for _ in keys]
        for node in component:
            for dataset, weight in (weights.get(node) or {}).items():
                for part, one in enumerate(per_part(weight)):
                    gain[index_of[dataset]][part] += one
        if not any(any(per_dataset) for per_dataset in gain):
            # no row names it, so it cannot move any balance: train, where it is
            # harmless and still available to a model that reads the pool whole
            for node in component: where[node] = PARTS[0]
            continue
        worth, deficit, has_room = [], [], []
        for part in range(len(PARTS)):
            short, fits = 0.0, True
            for d in range(len(keys)):
                if not any(gain[d]): continue
                placed = standing[keys[d]]
                short += want[d][part] * (sum(placed) + gain[d][part]) - placed[part]
                if ((placed[part] + gain[d][part]) / (budget[d] or 1)
                        > want[d][part] + slack[d]):
                    fits = False
            # only the real datasets say what a component is worth; a class is a
            # constraint on where it may go, not a second count of its rows
            worth.append(sum(gain[d][part] for d in range(n_real)))
            deficit.append(short)
            has_room.append(fits)
        best = max(range(len(PARTS)),
                   key=lambda part: (worth[part] if has_room[part] else -1,
                                     deficit[part], -part))
        for node in component: where[node] = PARTS[best]
        for d, key in enumerate(keys): standing[key][best] += gain[d][best]
    return where, standing

def split(nodes, edgelist, weights, datasets, frac, tolerance=0.05, rounds=2,
          solver="HIGHS", verbose=True, say=print, strata=None, strata_frac=None,
          strata_tolerance=0.15, floors=None):
    '''
    the whole of it: build the graph, place what can be placed whole, solve the
    one component that cannot.

    the order is the one difference from Lo-Hi, and it is deliberate. Lo-Hi
    solves the main component first and spreads the rest afterwards, which means
    the band the MIP enforced is not the band of the finished split. Placing the
    small components first and passing their weight in as a constant makes the
    band hold over what is written out. Both passes are deterministic, so this
    is still one shot.
    '''
    graph = build_graph(nodes, edgelist)
    giant, rest = split_components(graph)
    n_components = len(rest) + 1
    # nothing to cut: with no component bigger than a node every one of them goes
    # somewhere whole, and the greedy pass is the whole answer
    if giant.number_of_nodes() == 1:
        where, _ = place([sorted(giant.nodes())] + rest, weights, datasets, frac,
                         tolerance, strata, strata_frac, strata_tolerance, floors)
        counts = Counter(where.values())
        return where, {"status": "no cut needed", "seconds": 0.0,
                       "components": n_components, "largest_component": 1,
                       "by_part": {part: counts.get(part, 0)
                                   for part in (*PARTS, EXCLUDED)}}
    say(f"  {len(nodes):,} nodes, {graph.number_of_edges():,} edges, "
        f"{n_components:,} components; solving the largest "
        f"({giant.number_of_nodes():,} nodes), placing {len(rest):,} whole")

    where, standing = place(rest, weights, datasets, frac, tolerance, strata,
                            strata_frac, strata_tolerance, floors)
    quotient, cluster_of = coarsen_rounds(giant, rounds)
    members = [[] for _ in range(quotient.number_of_nodes())]
    for node in sorted(cluster_of): members[cluster_of[node]].append(node)
    say(f"  coarsened {giant.number_of_nodes():,} nodes -> {len(quotient):,} clusters, "
        f"{quotient.number_of_edges():,} cluster edges")

    cut, info = solve(quotient, members, weights, datasets, frac, tolerance, standing,
                      solver, verbose, strata, strata_frac, strata_tolerance, floors)
    where.update(cut)
    say(f"    {info['status']} in {info['seconds']}s, {info['variables']:,} vars / "
        f"{info['constraints']:,} constraints")

    counts = Counter(where.values())
    info.update(components=n_components, largest_component=giant.number_of_nodes(),
                coarsen_rounds=rounds, tolerance=tolerance,
                targets={dataset: list(shares) for dataset, shares
                         in zip(datasets, fracs(frac, datasets))},
                strata=len(strata_frac or ()), strata_tolerance=strata_tolerance,
                floors=floors or {},
                by_part={part: counts.get(part, 0) for part in (*PARTS, EXCLUDED)})
    return where, info

# -----------------------------------------------------------------------------
# the two strategies
# -----------------------------------------------------------------------------

# how far under its share a split may fall, past DATASAIL_EPSILON. when a dataset cannot
# be packed to the asked-for floor the SIZE is what gives way, one rung at a time, and
# which rung each dataset needed is recorded
EPSILON_LADDER = (0.20, 0.35, 0.50, 0.70)

def ladder_from(epsilon):
    '''the asked-for floor first, then the rungs looser than it'''
    return (epsilon,) + tuple(one for one in EPSILON_LADDER if one > epsilon)

def everywhere(where, datasets):
    '''one pooled assignment, read as the same one for every dataset'''
    return {name: where for name in datasets}

def spectral_reduce(members, graded, target, seed=42):
    """
    DataSAIL's own reduction: cluster the clusters down to `target`, seeded.

    This is `additional_clustering` in their clustering.py, which runs
    SpectralClustering on the precomputed cluster similarity when there are more
    clusters than the model wants. Seeded it is reproducible -- two runs at
    random_state=42 over 6,000 molecules and 175,091 scored pairs gave identical
    labels -- because sklearn feeds the seed to ARPACK's start vector.

    It is handed an already-clustered axis, as theirs is: their cluster() runs
    ecfp or mmseqs first and only then reduces. That ordering is not cosmetic
    here. On raw entities the molecule axis is 41,026 nodes whose similarity
    graph is disconnected, sklearn warns about exactly that, and ARPACK did not
    finish in 32 minutes. On the 3,337 components it takes 112 seconds.
    """
    if len(members) <= target: return members
    from scipy.sparse import coo_matrix
    from sklearn.cluster import SpectralClustering
    pairs, similarity = sparse_similarity(members, graded)
    size = len(members)
    rows = [a for a, b in pairs] + [b for a, b in pairs] + list(range(size))
    cols = [b for a, b in pairs] + [a for a, b in pairs] + list(range(size))
    scores = [similarity[pair] for pair in pairs] * 2 + [1.0] * size
    affinity = coo_matrix((scores, (rows, cols)), shape=(size, size)).tocsr()
    labels = SpectralClustering(n_clusters=target, affinity="precomputed",
                                random_state=seed).fit_predict(affinity)
    grouped = {}
    for member, label in zip(members, labels):
        grouped.setdefault(int(label), []).append(member)
    merged = []
    for label in sorted(grouped):
        flat = []
        for member in grouped[label]:
            flat.extend([member] if isinstance(member, str) else member)
        merged.append(sorted(flat))
    return sorted(merged, key=lambda group: (-len(group), group[0]))

def sparse_similarity(members, graded):
    '''
    the pairs of this axis that carry a score, as indices into members.

    one cvxpy term per pair that HAS a similarity rather than one per pair: every
    pair left out is multiplied by a similarity of zero, so the objective is the
    same number as the dense matrix DataSAIL would build, and the dense matrix
    for 12,718 proteins is 1.3 billion floats nobody needs.
    '''
    index = {}
    for i, member in enumerate(members):
        if isinstance(member, str): index[member] = i
        else:
            for node in member: index[node] = i
    similarity = {}
    for node1, node2, score in graded:
        first, second = index.get(node1), index.get(node2)
        if first is None or second is None or first == second: continue
        key = (first, second) if first < second else (second, first)
        if key in similarity:
            similarity[key] = max(similarity[key], score)
        else:
            similarity[key] = score
    pairs = sorted(similarity)
    return pairs, similarity

def dataset_strata(strata, dataset, members, frac, delta):
    '''
    DataSAIL's stratification matrix for one dataset: a row per cluster, a
    column per label class of that dataset.

    their constraint is a FLOOR per (split, class) -- each class must put at
    least share * (1 - delta) of its rows in every split -- which asks more of a
    class than a two-sided band does. a cluster here is a whole component, so a
    class with most of its rows in one component cannot give its share to
    validation however the solve goes, and the model comes back infeasible
    rather than stratified. so a class is passed only when no single cluster
    holds more of it than it can spare.
    '''
    per_node, keys = strata[:2]
    mine = [key for key in keys if key.split("/", 1)[0] == dataset]
    if not mine: return None, []
    import numpy as np
    rows_in = {key: [0.0] * len(members) for key in mine}
    for cluster, cluster_nodes in enumerate(members):
        for node in cluster_nodes:
            for key, rows in (per_node.get(node) or {}).items():
                if key in rows_in: rows_in[key][cluster] += rows
    room = 1.0 - 2 * min(frac[1:]) * (1 - delta)
    spread = [key for key in mine
              if sum(rows_in[key]) and max(rows_in[key]) / sum(rows_in[key]) <= room]
    if not spread: return None, []
    s_matrix = np.zeros((len(members), len(spread)))
    for column, key in enumerate(spread):
        for cluster, rows in enumerate(rows_in[key]):
            s_matrix[cluster, column] = rows
    return s_matrix, spread

def solve_one(dataset, members, cluster_weights, frac, s_matrix, delta, solver, ladder,
              pairs, similarity, floors=None):
    '''
    one C1 solve for one dataset, loosening only the size floors.

    the ladder is walked with the class floors on first and off after at each
    rung, because a class floor that cannot be met is a fact about where that
    class sits rather than something a looser size rule fixes. what it settled
    on is returned rather than printed, so the provenance says which dataset
    needed what.
    '''
    def attempt(matrix, epsilon):
        return datasail_cluster_1d.solve_c1(
            clusters=list(range(len(members))), weights=cluster_weights,
            s_matrix=matrix, similarities=similarity, distances=None, delta=delta,
            epsilon=epsilon, splits=list(frac), names=list(PARTS), max_sec=0,
            solver=solver, log_file=None, pairs=pairs, floors=floors)

    for epsilon in ladder:
        for matrix in ([s_matrix, None] if s_matrix is not None else [None]):
            assignment = attempt(matrix, epsilon)
            if assignment is not None:
                return assignment, {"epsilon": epsilon, "mode": "C1",
                                    "stratified": matrix is not None}
    raise RuntimeError(
        f"{dataset}: DataSAIL's C1 is infeasible at every epsilon up to {ladder[-1]} "
        f"even unstratified, over {len(members):,} entities. its size floors cannot be "
        "met at all, which is a fact about the weights rather than about the solver.")

# C2, DataSAIL's double-cold split, is NOT used on the protein-ligand datasets and the
# reason is arithmetic. It assigns the two axes independently, aimed at the shares whose
# product is frac -- 58.6/20.7/20.7 for 80/10/10 -- and an interaction survives only when
# both of its ends agree. Measured on the molecule axis of bindingdb, ONE cluster carries
# 83.8% of the row weight (lp-pdbbind: 68.5%, and its protein axis 59.4%, already past the
# 58.6% train target). A cluster that big has to land whole in one part, so validation and
# test cannot both reach their floors however epsilon or the cluster count is set:
# bindingdb came out 99.4/0.2/0.4 and lp-pdbbind 93.2/3.6/3.1. DataSAIL's own PDBBind
# example runs on the core set, 182 entities, where no such blob exists, and their paper
# says only that assigning every interaction "cannot be guaranteed" -- not that the
# survivors land on the asked-for ratios. So the molecules here follow their rows instead,
# in assign(), and the chemistry rule is bought back in dropped rows rather than in the
# band. plug/external/SOURCE pins the upstream file that measurement was made with.

def datasail_split(nodes, edgelist, weights, datasets, frac, tolerance, say, strata, arity,
                   solver, delta, graded, target, by_protein):
    '''
    DataSAIL once per dataset, on real similarity scores, one cluster per entity.

    Pooled, this does not work and the reason is arithmetic rather than
    computational. C1 takes ONE weight vector, so its only size rule is a floor
    on each split's share of ALL the rows; the two largest components of the
    pooled graph hold 65.9% and 16.1% of the rows, and moving those two about
    satisfies a pooled 80/10/10 while a small dataset sits entirely inside one
    of them. Three datasets came out of that with no validation set at all.

    Per dataset, with one cluster per entity and the measured scores between
    them, the model is the one DataSAIL is for. `graded` carries mmseqs identity
    for every pair over the coverage bar -- including the pairs BELOW the split's
    own threshold, which are exactly the ones that make the objective
    non-trivial.

    Only the PROTEIN axis is solved here, for every dataset including the two
    protein-ligand ones: see the note above on why C2 cannot hold the band on
    those. Each of their rows names one protein, so the protein axis aims
    straight at frac and the molecules are placed afterwards by assign().

    What this gives up, deliberately, is the hard guarantee. Entities are not
    held together in components, so two proteins over the threshold CAN
    be separated: the leakage is MINIMISED, not forbidden, and what is left is
    measured afterwards rather than asserted here. lohi is the arm that
    forbids it. That is the bargain, and having both is the point.
    '''
    mine_of = {}
    for node in nodes:
        for name in (weights.get(node) or {}):
            mine_of.setdefault(name, []).append(node)

    def axis_weights(members, dataset):
        '''
        the rows each cluster carries for this dataset, one number per cluster -- or
        its proteins, for a dataset in `by_protein`
        '''
        return row_weights(members, dataset, by_protein=dataset in by_protein)

    def row_weights(members, dataset, by_protein=False):
        def one(node):
            weight = (weights.get(node) or {}).get(dataset)
            if weight is None: return 0.0
            if by_protein: return 1.0
            return sum(per_part(weight)) / len(PARTS)
        return [one(member) if isinstance(member, str) else sum(map(one, member))
                for member in members]

    def crossing(where, edges):
        return sum(1 for node1, node2 in edges if node1 in where and node2 in where
                   and where[node1] != where[node2])

    ladder = ladder_from(tolerance)
    where_of, info = {}, {}
    for dataset in datasets:
        # DataSAIL's order: cluster the axis, then reduce it. one cluster per entity to
        # start, because handed a similarity matrix that is what their own `cluster()`
        # does. their mmseqs path is the one to avoid: its similarity matrix is "filled
        # with 1s", which throws the identities the objective needs away
        members = [[node] for node in sorted(mine_of.get(dataset, ()))]
        mine = spectral_reduce(members, graded, target)
        where_of[dataset] = {}
        if not mine:
            info[dataset] = {"nodes": 0, "status": "no nodes"}
            continue
        cluster_weights = axis_weights(mine, dataset)
        pairs, similarity = sparse_similarity(mine, graded)

        # a dataset whose rows name TWO proteins aims its floors at the square roots
        # of its row shares: a pair survives only when both ends land in one part, so
        # a part with a tenth of the proteins would get a hundredth of the pairs. a
        # protein-ligand row names one protein, so it aims straight at frac
        want = pair_frac(frac, arity[dataset])
        s_matrix, banded = dataset_strata(strata, dataset, mine, want, delta)
        # a protein-balanced dataset keeps a share of its rows in val and test as well
        floors = None
        if dataset in by_protein:
            rows = row_weights(mine, dataset)
            floors = [(rows, [share * sum(rows) for share in row_floors()])]
        assignment, how = solve_one(dataset, mine, cluster_weights, want, s_matrix,
                                    delta, solver, ladder, pairs, similarity, floors)

        for cluster, part in assignment.items():
            for node in mine[int(cluster)]: where_of[dataset][node] = part
        # what the objective could not prevent, counted rather than claimed away
        crossed = crossing(where_of[dataset], edgelist)
        info[dataset] = {"nodes": len(mine_of.get(dataset, ())),
                         "clusters": len(mine), "reduced_to": target,
                         "biggest_cluster": max((len(one) for one in mine), default=0),
                         "similar_pairs": len(pairs),
                         "crossing_edges": crossed,
                         "label_classes_banded": len(banded),
                         "status": "optimal", **how}
        why = ("" if how["stratified"] else
               " (no bandable classes)" if not banded else
               " (class floors unreachable)")
        say(f"  {dataset}: {how['mode']}, {len(mine):,} protein clusters, "
            f"{len(pairs):,} similar pairs — {crossed:,} edges crossing, "
            f"epsilon {how['epsilon']}, {len(banded)} classes banded{why}")

    return where_of, {
        "status": "minimised", "solved_per_dataset": True,
        "epsilon_asked": tolerance,
        "loosened_epsilon": sorted(name for name, one in info.items()
                                   if one.get("epsilon", tolerance) > tolerance),
        "per_dataset": info}

STRATEGIES = ("datasail", "lohi")

# -----------------------------------------------------------------------------
# building, stamping and writing one split
# -----------------------------------------------------------------------------

TASKS = ELEVEN
FRAC = (0.8, 0.1, 0.1)          # train / val / test, of each dataset's RETAINED rows
TOLERANCE = 0.05                # how far from FRAC a dataset may land. the RULE
# and what the solve aims at, which is tighter. the band is enforced on the weight
# proxy -- a row shared out over the nodes it rides on -- while the rule is counted off
# the written manifests, and the two differ. the gap is worst for a small interaction
# dataset: shs27k's rows are halved over two nodes each and survive only when both ends
# agree, so its proxy share and its written share come apart by over a point on 2,093
# rows. 0.05 let it land at 85.4% and 0.045 at 85.7%, both outside the rule they were
# meant to satisfy. 0.035 leaves room for the drift
SOLVE_TOLERANCE = 0.035
ROUNDS = 2                      # coarsening rounds before the Lo-Hi solve
# a class gets a looser band than a dataset does. a dataset has a hundred thousand rows
# to divide and can land within five points; a class with forty rows in it cannot, and
# insisting it does only drops the class
STRATA_TOLERANCE = 0.15
# a protein-balanced benchmark still keeps shares of its rows, the same in lohi and in
# DataSAIL. val and test keep ROW_FLOOR, so one huge assay cannot leave a part with almost
# nothing to score: without it DataSAIL left fireprot's test set at 9 proteins and 72 rows
ROW_FLOOR = 0.05
# and train keeps TRAIN_ROW_FLOOR, so the training set stays a training set. DataSAIL's
# C1 minimises leakage and nothing else, so where no split leaks (dms has 0 crossing
# edges) every assignment ties and the first feasible one is returned: with only the val
# and test floors, dms trained on 42% of its rows. lohi maximises retained rows and never
# went near this, but it is held to the same floor so the two methods share one rule
TRAIN_ROW_FLOOR = 0.60
# DataSAIL's epsilon: how far under its share a split may fall. its own default. where a
# dataset cannot be packed to it, EPSILON_LADDER loosens its floor and records that it did
DATASAIL_EPSILON = 0.05
# how many clusters DataSAIL's seeded spectral step reduces each axis to. their own
# additional_clustering, run in their order -- cluster the axis, then reduce it. on raw
# entities it does not finish: bindingdb's 41,026 molecules ran 32 minutes without
# converging, because their similarity graph is disconnected and ARPACK grinds on the
# null space. on an already-clustered axis it is a couple of minutes
DATASAIL_CLUSTERS = 1000
# datasail gets SCIP: the symmetry workaround in datasail_support is written for it and
# DataSAIL's own default is a MIP solver of this kind. lohi takes c.solver
SOLVER_OF = {"datasail": "SCIP"}

def open_gz(path):
    '''
    a gzip writer with the timestamp zeroed.

    gzip stamps the current time into its header, so without this two identical
    runs produce different bytes and "exactly reproducible" would be a claim
    about the contents only.
    '''
    raw = gzip.GzipFile(filename=str(path), mode="wb", mtime=0)
    return io.TextIOWrapper(raw, newline="", write_through=True)

def row_split(cls, row, protein_graph, molecule_graph, where_of, molecule_where_of):
    '''where one row goes under this assignment, or why it goes nowhere'''
    where = where_of.get(cls.name) or {}
    molecule_where = molecule_where_of.get(cls.name) or {}
    nodes = G.row_nodes(cls, row, protein_graph["of_entity"])
    if not nodes: return "", "no_sequence"
    parts = {where.get(node) for node in nodes}
    if None in parts: return "", "unplaced_protein"
    if EXCLUDED in parts: return "", "excluded_protein"
    # a row whose two proteins disagree belongs to neither: keeping it would put the
    # same datapoint on both sides of the split
    if len(parts) > 1: return "", "proteins_disagree"
    part = parts.pop()
    if molecule_graph and "ligand" in cls.views:
        smiles = str(row.get(cls.ligand) or "").strip()
        if smiles:
            molecule_part = molecule_where.get(G.ligand_node(smiles))
            if molecule_part is None: return "", "unplaced_ligand"
            if molecule_part == EXCLUDED: return "", "excluded_ligand"
            if molecule_part != part: return "", "ligand_disagrees"
    return part, ""

def molecule_weights(classes, protein_graph, molecule_graph, where_of):
    '''
    what each molecule is worth IN EACH PART: the rows whose protein is already
    there. a molecule put in train is worth the rows whose proteins are in train
    and nothing for the rest, which is why this runs after the protein split.
    '''
    per_molecule = {}
    for cls in classes:
        if "ligand" not in cls.views: continue
        where = where_of.get(cls.name) or {}
        for row in cls.rows():
            smiles = str(row.get(cls.ligand) or "").strip()
            if not smiles: continue
            node = G.ligand_node(smiles)
            if node not in molecule_graph["comp_of"]: continue
            parts = {where.get(n) for n in G.row_nodes(cls, row,
                                                       protein_graph["of_entity"])}
            if len(parts) != 1: continue
            part = parts.pop()
            if part not in PARTS: continue
            counts = per_molecule.setdefault(node, {}).setdefault(
                cls.name, [0] * len(PARTS))
            counts[PARTS.index(part)] += 1
    return per_molecule

def molecules_follow_rows(molecule_graph, weights, dataset):
    '''
    put each molecule where most of its rows already are.

    datasail does not solve its molecules, so they need placing without giving
    rows up, and a row survives only if its molecule sits with its protein. so each molecule goes where it is worth most; what that gives up is
    the chemistry rule, since two near-identical molecules can then straddle.

    only the named dataset's rows count, which is what the per-dataset arm needs: the same molecule is in bindingdb and in lp-pdbbind,
    and once those two are split independently it has to follow each one's
    proteins separately.
    '''
    where = {}
    for node in molecule_graph["nodes"]:
        totals = [0] * len(PARTS)
        for name, per_part in (weights.get(node) or {}).items():
            if name != dataset: continue
            for i, one in enumerate(per_part): totals[i] += one
        best = max(range(len(PARTS)), key=lambda i: (totals[i], -i))
        where[node] = PARTS[best]
    return where

def drop_crossing_chemistry(where, edges, weights, dataset, frac):
    '''
    exclude every molecule whose chemistry would otherwise straddle the split.

    molecules placed by their rows keep almost every row, which is what makes
    the band reachable, but on its own it gives the chemistry rule up: two
    molecules over the tanimoto bar can sit on opposite sides. so each crossing
    pair has one end EXCLUDED here and row_split drops the rows naming it. the
    rule is bought back in retention rather than in balance, which is the trade
    this arm can afford and C2 could not.

    WHICH end pays is chosen by the band, and the obvious rule is the one that
    does not work. Dropping the end carrying fewer rows costs the fewest rows
    per pair, but the lighter end of a crossing pair is almost always the
    validation or test one -- so the chemistry rule gets paid for out of the two
    splits with nothing to spare. Measured on lp-pdbbind that is train -8%, val
    -58%, test -41%, landing 87.9/5.1/7.0 from a solve that had hit 80/10/10.
    Always dropping the train end instead fails the other way: a popular
    scaffold similar to one test molecule would cost thousands of rows to save
    two.

    So each pair is settled by the thing actually at stake. Both ends are
    costed against the running row totals and the one leaving the three shares
    closer to frac is excluded, the node id breaking a tie. While train is over
    its share that takes rows from train, and it stops as soon as train is not,
    which is what neither fixed rule can do. One pass is enough: excluding an
    end settles that pair for good and can never make another pair cross. A
    pair with an end this dataset never measures is left alone, since no row of
    this dataset can leak through it.
    '''
    def counts_of(node):
        return (weights.get(node) or {}).get(dataset) or ()

    def keeps(node):
        '''the rows this molecule holds on to where it sits: the rest already disagree'''
        counts, part = counts_of(node), where.get(node)
        return counts[PARTS.index(part)] if counts and part in PARTS else 0

    where = dict(where)
    kept = [0.0] * len(PARTS)
    for node in where:
        if counts_of(node) and where[node] in PARTS:
            kept[PARTS.index(where[node])] += keeps(node)

    def drift(node):
        '''the worst of the three shares' distances from frac, with this one gone'''
        totals = list(kept)
        totals[PARTS.index(where[node])] -= keeps(node)
        whole = sum(totals) or 1.0
        return max(abs(one / whole - want) for one, want in zip(totals, frac))

    for node1, node2 in sorted(edges):
        if not (sum(counts_of(node1)) and sum(counts_of(node2))): continue
        here, there = where.get(node1), where.get(node2)
        if here == there or here not in PARTS or there not in PARTS: continue
        loser = min((node1, node2), key=lambda node: (drift(node), node))
        kept[PARTS.index(where[loser])] -= keeps(loser)
        where[loser] = EXCLUDED
    return where

def protein_targets(classes, frac):
    '''
    the share each dataset's PROTEINS should get so that its ROWS come out at
    frac, and how many proteins each dataset's rows name.

    for a dataset whose rows name one protein the two are the same question. for
    an interaction dataset they are not: a pair survives only when both of its
    ends land in the same part, so a part with a tenth of the proteins gets a
    hundredth of the pairs, and the quotas have to aim at the square roots.
    '''
    arity = {cls.name: len(cls.keyfields()) for cls in classes}
    return ({name: pair_frac(frac, n) for name, n in arity.items()}, arity)

def row_floors():
    '''the least share of its rows a protein-balanced benchmark keeps in each part'''
    return (TRAIN_ROW_FLOOR, ROW_FLOOR, ROW_FLOOR)

def single_protein(classes):
    '''
    the benchmarks balanced on proteins: each row names one protein and no ligand.

    the claim is generalisation to new proteins, and a held-out protein with fifty
    thousand variants is still one protein the model has not seen, so balancing rows
    would fill the test set with the proteins that carry the fewest measurements. pair
    and protein-ligand benchmarks keep their rows, since the row is the datapoint there
    and a tenth of the proteins holds only a hundredth of the pairs.
    '''
    return {cls.name for cls in classes
            if len(cls.keyfields()) == 1 and "ligand" not in cls.views}

def classes_in_proteins(per_node, single):
    '''a single-protein benchmark's label classes, recounted as one per protein'''
    return {node: {key: (1 if key.split("/", 1)[0] in single else weight)
                   for key, weight in per_dataset.items()
                   if weight > 0 or key.split("/", 1)[0] not in single}
            for node, per_dataset in per_node.items()}

def balance_on_proteins(classes, weights, per_node, strata_frac, strata_tolerance):
    '''
    move each single-protein benchmark's band from its rows to its proteins.

    the proteins go in as one more banded key per benchmark, `<name>/#proteins`, worth
    one per node, and the rows keep only their floors, row_floors(). the objective is
    untouched, so the solve still keeps as many rows as the bands allow. the
    benchmark's label classes are recounted in proteins too, so a class is balanced
    in the same currency as the benchmark it belongs to. strata_frac and
    strata_tolerance gain the new keys in place.
    '''
    single = single_protein(classes)
    if not single: return per_node, None
    counted = classes_in_proteins(per_node, single)
    for node, per_dataset in weights.items():
        ones = {f"{name}/#proteins": 1 for name in per_dataset if name in single}
        if ones: counted[node] = {**counted.get(node, {}), **ones}
    for name in sorted(single):
        strata_frac[f"{name}/#proteins"] = tuple(FRAC)
        strata_tolerance[f"{name}/#proteins"] = SOLVE_TOLERANCE
    floors = {name: row_floors() for name in sorted(single)}
    return counted, floors

def assign(strategy, classes, protein_graph, molecule_graph, weights, strata, say):
    '''
    run one strategy over the proteins and then over the molecules.

    hands back a split per dataset per node. lohi gives every dataset the same
    assignment, because it is solved over the pool and a protein two datasets
    measure carries one label; datasail gives each dataset its own,
    because it is solved per dataset and cannot do otherwise.
    '''
    datasets = sorted(cls.name for cls in classes)
    want, arity = protein_targets(classes, FRAC)
    edgelist = [edge for kind in G.KINDS for edge in protein_graph["edges"][kind]]
    per_node, keys = strata[0], strata[1]

    if strategy == "lohi":
        # the only strategy that can take a target per dataset and per class. a class
        # of an interaction dataset needs the same square roots its dataset does
        strata_frac = {key: pair_frac(FRAC, arity[key.split("/", 1)[0]])
                       for key in keys}
        strata_tolerance = {key: STRATA_TOLERANCE for key in keys}
        per_node, floors = balance_on_proteins(classes, weights, per_node, strata_frac,
                                               strata_tolerance)
        where, info = split(
            protein_graph["nodes"], edgelist, weights, datasets, want, SOLVE_TOLERANCE,
            rounds=ROUNDS, solver=c.solver, say=say, strata=per_node or None,
            strata_frac=strata_frac or None, strata_tolerance=strata_tolerance,
            floors=floors)
        where_of = everywhere(where, datasets)
    else:
        # only the proteins are solved: a C2 double-cold solve cannot hold the band on
        # the two ligand datasets, for the reason noted above datasail_split, so their
        # molecules are placed below by their rows instead. C1 takes one weight vector,
        # so a protein-balanced benchmark is weighted by its proteins alone
        single = single_protein(classes)
        where_of, info = datasail_split(
            protein_graph["nodes"], edgelist, weights, datasets, FRAC, DATASAIL_EPSILON, say,
            (classes_in_proteins(per_node, single), *strata[1:]), arity,
            SOLVER_OF["datasail"], STRATA_TOLERANCE, protein_graph["graded"],
            DATASAIL_CLUSTERS, single)

    molecule_where_of = everywhere({}, datasets)
    if molecule_graph:
        per_molecule = molecule_weights(classes, protein_graph, molecule_graph, where_of)
        if strategy == "lohi":
            # the k-cut again, so no chemistry crosses either. it costs rows
            molecule_where, _ = split(
                molecule_graph["nodes"], molecule_graph["edges"]["ligand"], per_molecule,
                sorted({d for v in per_molecule.values() for d in v}), FRAC,
                SOLVE_TOLERANCE, rounds=ROUNDS, solver=c.solver, say=say)
            molecule_where_of = everywhere(molecule_where, datasets)
        else:
            # solved per dataset, so each dataset's molecules follow its own proteins:
            # the same ligand is in bindingdb and in lp-pdbbind and the two splits are
            # independent. then the chemistry rule is enforced by dropping rows, which
            # is what keeps this arm's per-benchmark guarantee true of the ligands too
            molecule_where_of = {
                name: drop_crossing_chemistry(
                    molecules_follow_rows(molecule_graph, per_molecule, name),
                    molecule_graph["edges"]["ligand"], per_molecule, name, FRAC)
                for name in datasets}
    return where_of, molecule_where_of, info

def stamp(classes, protein_graph, molecule_graph, where_of, molecule_where_of, out, say):
    '''
    one file per dataset, a line per datapoint in rows() order. that order is
    what lets a reader label their own copy by zipping, and a length mismatch is
    then an error rather than a silent mislabelling.
    '''
    (out / "splits").mkdir(parents=True, exist_ok=True)
    counted = {}
    for cls in classes:
        rows = Counter()
        with open_gz(out / "splits" / f"{cls.name}.csv.gz") as f:
            writer = csv.writer(f)
            writer.writerow(["row_index", "split", "dropped_because"])
            for i, row in enumerate(cls.rows()):
                part, why = row_split(cls, row, protein_graph, molecule_graph,
                                      where_of, molecule_where_of)
                writer.writerow([i, part, why])
                rows[part or f"dropped:{why}"] += 1
        counted[cls.name] = dict(rows)
        kept, total = sum(rows[p] for p in PARTS), sum(rows.values())
        say(f"  {cls.name}: {kept:,}/{total:,} rows ({100 * kept / max(total, 1):.0f}%) "
            f"— {rows['train']:,}/{rows['val']:,}/{rows['test']:,}")
    return counted

def write_nodes(protein_graph, molecule_graph, where_of, molecule_where_of, weights, out):
    '''
    the assignment itself, so anything downstream can re-derive the rest.

    a line per (node, dataset) rather than per node, because that is what a
    split solved per dataset actually is: the same protein takes one label from
    each dataset that measures it, and under the per-dataset arm those labels
    need not agree. the pooled arm writes the same label on every line of a
    node, which is the thing that makes it usable across benchmarks, and
    writing it this way means that can be read off the file instead of
    taken on trust. a node no dataset measures gets one line with an empty
    dataset.
    '''
    (out / "nodes").mkdir(parents=True, exist_ok=True)
    def lines(node, assignment, extra):
        '''one line per dataset measuring this node, or one with no dataset'''
        names = sorted(weights.get(node) or ())
        if not names:
            label = next((w.get(node, "") for w in assignment.values() if node in w), "")
            return [[node, "", label, *extra("")]]
        return [[node, name, (assignment.get(name) or {}).get(node, ""), *extra(name)]
                for name in names]

    with open_gz(out / "nodes" / "proteins.csv.gz") as f:
        writer = csv.writer(f)
        writer.writerow(["node", "dataset", "split", "length", "component", "entities"])
        for node in protein_graph["nodes"]:
            aliases = protein_graph["aliases"].get(node, ())
            writer.writerows(lines(node, where_of, lambda name: [
                protein_graph["lengths"].get(node, ""),
                protein_graph["comp_of"][node],
                " ".join(one for one in aliases
                         if not name or one.split("::", 1)[0] == name)]))
    if molecule_graph:
        with open_gz(out / "nodes" / "molecules.csv.gz") as f:
            writer = csv.writer(f)
            writer.writerow(["node", "dataset", "split", "component"])
            for node in molecule_graph["nodes"]:
                writer.writerows(lines(node, molecule_where_of,
                                       lambda name: [molecule_graph["comp_of"][node]]))

def written_digests(out):
    '''
    every written file, fingerprinted.

    the node files are not the payload: a user downloads `splits/<dataset>.csv.gz`
    and labels their own copy by zipping it to rows(), so that is the file whose
    digest has to be on record. keyed by the path relative to the split, because
    two directories here hold a `molecules.csv.gz` and a bare name would collide
    '''
    return {str(path.relative_to(out)):
            hashlib.sha256(path.read_bytes()).hexdigest()[:16]
            for path in sorted(out.rglob("*.csv.gz"))}

def searches_used():
    '''the cached search tables this build actually read, with their digests'''
    return {kind: {"table": path.name, "bytes": path.stat().st_size,
                   "sha256": hashlib.sha256(path.read_bytes()).hexdigest()[:16]}
            for kind, path in sorted(G.USED.items()) if path.exists()}

def run(strategy, tasks=TASKS, root=None, say=print):
    '''build one split and write it'''
    classes = [REGISTRY[name] if isinstance(name, str) else name for name in tasks]
    seconds, started = {}, time.time()

    say(f"graph over {len(classes)} datasets ...")
    protein_graph = G.build(classes, say)
    molecule_graph = G.build_ligands(classes, say)
    weights, totals, unretainable = G.weights(classes, protein_graph, molecule_graph, say)
    strata = G.strata_weights(classes, protein_graph, molecule_graph, say)
    seconds["graph"] = time.time() - started

    started = time.time()
    say(f"{strategy} ...")
    where_of, molecule_where_of, info = assign(strategy, classes, protein_graph,
                                               molecule_graph, weights, strata, say)
    seconds[strategy] = time.time() - started

    out = Path(root or c.splits) / protein_graph["tag"] / strategy
    started = time.time()
    say(f"writing {out} ...")
    write_nodes(protein_graph, molecule_graph, where_of, molecule_where_of,
                weights, out)
    rows = stamp(classes, protein_graph, molecule_graph, where_of, molecule_where_of,
                 out, say)
    digests = written_digests(out)
    seconds["write"] = time.time() - started

    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    (out / "provenance.json").write_text(json.dumps({
        "tag": protein_graph["tag"], "strategy": strategy,
        "datasets": protein_graph["tasks"], "thresholds": c.THRESHOLDS,
        "structure_views": protein_graph["views"],
        "structure_coverage": protein_graph["coverage"],
        "frac": list(FRAC), "tolerance": TOLERANCE,
        "solve_tolerance": SOLVE_TOLERANCE, "rounds": ROUNDS,
        "stratified": True, "strata_tolerance": STRATA_TOLERANCE,
        "balance_on_proteins": True,
        "row_floors": dict(zip(PARTS, row_floors())),
        "datasail_epsilon": DATASAIL_EPSILON,
        "graph": {"proteins": len(protein_graph["nodes"]),
                  "components": len(protein_graph["groups"]),
                  "edges": {kind: len(protein_graph["edges"][kind]) for kind in G.KINDS},
                  "molecules": len(molecule_graph["nodes"]) if molecule_graph else 0,
                  "molecule_edges": (len(molecule_graph["edges"]["ligand"])
                                     if molecule_graph else 0),
                  "label_classes": len(strata[1]),
                  "label_classes_too_concentrated_to_band": len(strata[2]),
                  "rows": dict(sorted(totals.items())),
                  "unretainable_rows": dict(sorted(unretainable.items()))},
        "rows": rows, "solve": info, "files": digests,
        "searches": searches_used(),
        # ru_maxrss is bytes on darwin and kilobytes on linux
        "cost": {"seconds": {k: round(v, 1) for k, v in seconds.items()},
                 "peak_memory_mb": round(peak / (1 << (20 if sys.platform == "darwin"
                                                       else 10)), 1),
                 "platform": f"{platform.system().lower()} {platform.machine()}",
                 "python": platform.python_version(),
                 "solver": SOLVER_OF.get(strategy, c.solver)},
    }, indent=2, sort_keys=True) + "\n")
    return out

def main(argv=None):
    '''one flag, naming the strategy'''
    argv = list(sys.argv[1:] if argv is None else argv)
    asked = [name for name in STRATEGIES if f"--{name}" in argv]
    if len(argv) != 1 or len(asked) != 1:
        print("\n".join(__doc__.strip().splitlines()[2:4]))
        return 2
    out = run(asked[0])
    print(f"\nwrote {out}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
