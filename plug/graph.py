'''
What is too alike to separate, as one graph over all eleven datasets, and the
searches that find it.

Two proteins get an edge at mmseqs 30% identity over 80% coverage, or foldseek
TM 0.7 both ways round; two molecules at ecfp4 tanimoto 0.5. Nothing joins a
protein to a molecule. A node is a hash of the sequence, so one protein is one
node across every dataset that measures it. Each search caches pairs WITH THEIR
SCORE down to a floor far below the threshold, so the threshold is a filter over
a cached table rather than another search.
'''

import gzip
import hashlib
import pickle
import re
import subprocess
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path

from . import config as c

# no e-value cutoff on either search. an e-value moves with how big the searched set
# is, so a borderline hit dropped here turns up later as a shared sequence across two
# splits. identity, coverage and TM do not move with the size of the search
EVALUE = "inf"

# every cached artefact a build actually consumed, so the split that came out of it can
# record their digests. the names are content-keyed, but a shared scratch directory is
# still shared: a table rebuilt under the same name by something else would change the
# splits silently, and a recorded digest turns that into something a rerun can catch.
#
# the pooled pickles belong here as much as the raw tables do, and matter more: on a
# warm build graph.py returns a pickle and the mmseqs table is never opened at all
USED = {}

def node_of(seq):
    '''the node one sequence is -- identical sequences collapse before any search'''
    return "p" + hashlib.sha1(seq.strip().upper().encode()).hexdigest()[:16]

def ligand_node(smiles):
    '''the node one molecule is'''
    return "l" + hashlib.sha1(smiles.strip().encode()).hexdigest()[:16]

def alias(bench, key):
    '''a dataset's own entity id, namespaced so two datasets cannot collide'''
    return f"{bench}::{key}"

def row_nodes(cls, row, of_entity):
    '''the protein nodes one row rides on: one for a measurement, two for a pair'''
    nodes = set()
    for field in cls.keyfields():
        nodes.update(of_entity.get(alias(cls.name, row[field]), ()))
    return sorted(nodes)

def _seq_key(seqs):
    '''name a pooled search by the sequences in it, the way _struct_key does'''
    digest = hashlib.sha256()
    for node in sorted(seqs):
        digest.update(node.encode())
        digest.update(b"\0")
        digest.update(seqs[node].encode())
    return digest.hexdigest()[:16]

def sequence_hits(seqs, say):
    '''
    every sequence pair worth keeping, searched once and cached WITH ITS IDENTITY.

    both coverage modes are run and either one counts: mode 1 wants the found
    protein covered and mode 2 the searched one, and a short protein that is one
    domain of a longer one is leakage whichever way round it was searched. A pair
    found by both keeps the higher identity.

    the table is kept down to `seq_floor`, not down to the split's `min_id`, and
    for the same two reasons the foldseek table is kept down to `struct_floor`.
    Every threshold at or above the floor is then a filter over one table rather
    than another hour of mmseqs. And the hits BELOW min_id are the ones a graded
    objective needs: two proteins at 25% identity over 80% of their length are
    not joined by the hard threshold, so they sit in different components, and
    nothing but their recorded identity can tell a solver to keep them apart.
    Without them every between-component similarity is zero and DataSAIL's
    objective is the constant zero.
    '''
    numbered = sorted(seqs)
    c.work.mkdir(parents=True, exist_ok=True)
    cache = (c.work / f"mmseqs_{len(numbered)}_{_seq_key(seqs)}"
             f"_ge{int(c.seq_floor * 100)}_cov{int(float(c.cov) * 100)}.tsv.gz")
    if not cache.exists() and c.reuse_only:
        raise FileNotFoundError(
            f"config.reuse_only is set and there is no cached sequence table at {cache}. that "
            f"is mmseqs over {len(numbered):,} sequences in two coverage modes, an hour "
            "or more, so the miss is reported rather than started. the name is built "
            "from the sequences, so a miss means the data moved.")
    if not cache.exists():
        say(f"    mmseqs over {len(numbered):,} sequences -> {cache.name}")
        t0 = time.time()
        best = {}
        with tempfile.TemporaryDirectory(dir=c.work) as tmp:
            tmp = Path(tmp)
            fasta, hits = tmp / "in.fasta", tmp / "hits.m8"
            # numbered, because dataset ids are not always safe in a fasta header
            with open(fasta, "w") as out:
                for i, node in enumerate(numbered):
                    out.write(f">s{i}\n{seqs[node]}\n")
            for cov_mode in ("1", "2"):
                # --max-seqs uncapped: in a pool full of near-copies a cap fills with
                # those and hides the distant hits, and a hidden hit is a leak
                subprocess.run([str(c.mmseqs), "easy-search", str(fasta), str(fasta),
                                str(hits), str(tmp / f"s{cov_mode}"), "-s", c.sens,
                                "-e", EVALUE, "-c", c.cov, "--cov-mode", cov_mode,
                                "--max-seqs", str(len(numbered)), "--threads", c.threads,
                                "-v", "1", "--format-output", "query,target,fident"],
                               check=True)
                for line in open(hits):
                    query, target, identity = line.split()[:3]
                    identity = float(identity)
                    if identity < c.seq_floor: continue
                    node1 = numbered[int(query[1:])]
                    node2 = numbered[int(target[1:])]
                    if node1 == node2: continue
                    pair = (node1, node2) if node1 < node2 else (node2, node1)
                    if identity > best.get(pair, -1.0): best[pair] = identity
        # written to a neighbour and moved, so a killed search leaves no half table
        partial = cache.with_suffix(".part")
        with gzip.open(partial, "wt") as out:
            for (node1, node2), identity in sorted(best.items()):
                out.write(f"{node1}\t{node2}\t{identity:.4f}\n")
        partial.replace(cache)
        say(f"    {len(best):,} pairs at identity >= {c.seq_floor} over coverage "
            f"{c.cov} in {time.time() - t0:.0f}s")

    USED["sequence"] = cache
    with gzip.open(cache, "rt") as f:
        for line in f:
            node1, node2, identity = line.rstrip("\n").split("\t")
            yield node1, node2, float(identity)

def _struct_key(paths):
    '''name a set of structures by its entity names and the bytes of its files'''
    digest = hashlib.sha256()
    for entity in sorted(paths):
        if "\t" in entity or "\n" in entity:
            raise ValueError(f"entity name {entity!r} holds a tab or newline, which the "
                             "cached hit table uses as separators")
        digest.update(entity.encode())
        digest.update(b"\0")
        digest.update(Path(paths[entity]).read_bytes())
    return digest.hexdigest()[:16]

def structure_hits(paths, say):
    '''
    every structural pair worth keeping, searched once and cached.

    the TM score is read both ways round, which is the shape version of the two
    coverage modes. foldseek's own alntmscore measures over the matched stretch
    instead, which scores one domain inside a much longer chain as a whole-chain
    match -- six times too many hits.

    the table is cached under a name built from the chains themselves, and every
    threshold at or above struct_floor is then a filter over that one table
    rather than another day of tm-align.
    '''
    if c.split_tm < c.struct_floor:
        raise ValueError(f"split_tm {c.split_tm} is below the cached table's floor "
                         f"{c.struct_floor}, so the table does not hold every pair the "
                         "split would need. lower struct_floor and search again.")
    entities = list(paths)
    c.work.mkdir(parents=True, exist_ok=True)
    cache = (c.work / f"foldseek_{len(paths)}_{_struct_key(paths)}"
             f"_ge{int(c.struct_floor * 100)}.tsv.gz")
    if not cache.exists() and c.reuse_only:
        raise FileNotFoundError(
            f"config.reuse_only is set and there is no cached structure table at {cache}. that "
            f"is foldseek over {len(entities):,} chains, hours to days, so the miss is "
            "reported rather than started. the name is built from the chain set, so a "
            "miss means the structures moved.")
    if not cache.exists():
        say(f"    foldseek over {len(entities):,} structures -> {cache.name}")
        t0 = time.time()
        with tempfile.TemporaryDirectory(dir=c.work) as tmp:
            tmp = Path(tmp)
            pool = tmp / "pool"; pool.mkdir()
            for i, entity in enumerate(entities):
                (pool / f"s{i}.pdb").write_bytes(Path(paths[entity]).read_bytes())
            database, result, hits = tmp / "db", tmp / "res", tmp / "hits.m8"
            # foldseek chatters on stdout, which a caller may be writing data to
            run = lambda *args: subprocess.run([str(c.foldseek), *args, "-v", "1"],
                                               check=True, stdout=sys.stderr)
            run("createdb", str(pool), str(database))
            run("search", str(database), str(database), str(result), str(tmp / "ss"),
                "-e", EVALUE, "--alignment-type", "1", "-a",
                "--max-seqs", str(len(entities)), "--threads", c.threads,
                *(("--gpu", "1") if c.gpu else ()))
            run("convertalis", str(database), str(database), str(result), str(hits),
                "--format-output", "query,target,qtmscore,ttmscore",
                "--threads", c.threads)
            # written to a neighbour and moved, so a killed search leaves no half table
            partial, kept = cache.with_suffix(".part"), 0
            with open(hits) as source, gzip.open(partial, "wt") as out:
                for line in source:
                    query, target, query_tm, target_tm = line.split("\t")[:4]
                    if max(float(query_tm), float(target_tm)) < c.struct_floor: continue
                    entity1 = entities[int(re.match(r"s(\d+)", query).group(1))]
                    entity2 = entities[int(re.match(r"s(\d+)", target).group(1))]
                    out.write(f"{entity1}\t{entity2}\t{float(query_tm):.4f}"
                              f"\t{float(target_tm):.4f}\n")
                    kept += 1
            partial.replace(cache)
            say(f"    {kept:,} pairs at tm >= {c.struct_floor} in {time.time() - t0:.0f}s")

    USED["structure"] = cache
    with gzip.open(cache, "rt") as f:
        for line in f:
            entity1, entity2, query_tm, target_tm = line.rstrip("\n").split("\t")
            if max(float(query_tm), float(target_tm)) > c.split_tm:
                yield entity1, entity2

def ligand_links(smiles):
    '''
    molecules over the tanimoto bar, in blocks because the pairwise matrix is the
    only part of this that does not fit in memory. two entries sharing a molecule
    are linked too. stereochemistry is ignored, so two enantiomers score 1.0 and
    cannot be separated -- the safe way round.
    '''
    import numpy as np
    from rdkit import Chem, RDLogger
    from rdkit.Chem import rdFingerprintGenerator
    RDLogger.DisableLog("rdApp.*")

    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    owners, fingerprints, seen, unreadable = [], [], {}, {}
    for name, molecules in smiles.items():
        for molecule in molecules:
            if molecule in seen: owners[seen[molecule]].append(name); continue
            if molecule in unreadable: unreadable[molecule].append(name); continue
            parsed = Chem.MolFromSmiles(molecule)
            # a few rows are not readable chemistry. they still cannot straddle a
            # split, so they are kept by name and simply compared to nothing
            if parsed is None: unreadable[molecule] = [name]; continue
            seen[molecule] = len(fingerprints); owners.append([name])
            fingerprints.append(generator.GetFingerprintAsNumPy(parsed).astype(np.float32))

    # two rows naming the same molecule are the same molecule, which is a similarity
    # of one and not a score to be minimised
    for sharing in (*owners, *unreadable.values()):
        for other in sharing[1:]: yield sharing[0], other, 1.0
    if not fingerprints: return

    bitmatrix = np.asarray(fingerprints, dtype=np.float32)
    bits_set = bitmatrix.sum(1)
    for start in range(0, len(bitmatrix), 1500):
        stop = min(start + 1500, len(bitmatrix))
        shared = bitmatrix[start:stop] @ bitmatrix.T
        tanimoto = shared / (bits_set[start:stop, None] + bits_set[None, :]
                             - shared + 1e-9)
        for row in range(stop - start):
            tanimoto[row, : start + row + 1] = 0                 # each pair once
        # kept down to tan_floor rather than to the split's own tanimoto, for the same
        # reason the sequence and structure tables are: the pairs below the threshold
        # are not edges, they are what a graded objective has to minimise
        for first, second in zip(*np.nonzero(tanimoto >= c.tan_floor)):
            yield (owners[start + int(first)][0], owners[int(second)][0],
                   float(tanimoto[first, second]))


# what counts as too alike to separate: each kind is a measurement of similarity between
# two proteins. an id one dataset files several sequences under is not one, since
# allobench's target_id groups by the biology studied (ASD11990000_1 is T4 lysozyme and
# three unrelated receptor constructs). a row still RIDES on every node its entity names,
# through row_nodes, so a row whose sequences land in different splits is dropped and
# says so
KINDS = ("sequence", "structure")

# a label class is banded only if it reaches at least this many separate components and
# holds at least this many rows. below either, it cannot be spread across three splits
# however the solver tries, and asking makes the model infeasible instead of balanced
MIN_CLASS_COMPONENTS = 3
MIN_CLASS_ROWS = 30
# and a dataset is stratified only if it has at most this many classes. cafa5 labels
# 8,630 proteins with 1,462 go terms and each protein carries about twenty of them at
# once, so a band per term is not a balance condition -- it is a thousand interacting
# constraints on one small dataset, and the model comes out infeasible rather than
# stratified. a handful of classes per dataset, like dms's five assay types, is the
# case this is for; the rest is reported instead of required
MAX_CLASSES_PER_DATASET = 20

def merge(names, *links):
    '''union-find: alike something alike lands in one component, so it cannot split'''
    index_of = {name: i for i, name in enumerate(names)}
    parent = list(range(len(names)))

    def root(index):
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    for edgelist in links:
        for node1, node2 in edgelist:
            root1, root2 = root(index_of[node1]), root(index_of[node2])
            if root1 != root2: parent[root1] = root2
    groups = {}
    for name in names: groups.setdefault(root(index_of[name]), []).append(name)
    return list(groups.values())

def collect(tasks, say):
    '''read every dataset's rows once for the sequences they name'''
    seqs, of_entity, aliases = {}, {}, {}
    for cls in tasks:
        n_rows = 0
        for row in cls.rows():
            n_rows += 1
            for key_field, seq_field in zip(cls.keyfields(), cls.seqfields()):
                sequence = str(row[seq_field]).strip().upper()
                # a row with no sequence names no protein; it is carried to the
                # manifest as dropped rather than quietly given a node
                if not sequence: continue
                node = node_of(sequence)
                seqs.setdefault(node, sequence)
                entity = alias(cls.name, row[key_field])
                nodes = of_entity.setdefault(entity, [])
                if node not in nodes: nodes.append(node)
                entities = aliases.setdefault(node, [])
                if entity not in entities: entities.append(entity)
        say(f"  {cls.name}: {n_rows:,} rows")
    if not seqs:
        raise FileNotFoundError("no sequences in any dataset — download them first, "
                                "see data/download_*.sh")
    say(f"  pooled: {len(seqs):,} distinct sequences over {len(of_entity):,} entities")
    return seqs, of_entity, aliases

def structure_paths(tasks, views, of_entity, say):
    '''
    one structure file per entity, pooled so foldseek runs once over everything.

    an entity with no structure would be checked on sequence alone while its
    neighbours are checked on both, which is a hole in the exact guarantee the
    structure view is for -- so a shortfall stops the run.
    '''
    paths, coverage = {}, {}
    for cls in tasks:
        if cls.name not in views: continue
        wanted = [entity for entity in of_entity
                  if entity.startswith(f"{cls.name}::")]
        found = {alias(cls.name, entity): path
                 for entity, path in cls.structures().items()}
        missing = sorted(entity for entity in wanted if entity not in found)
        resolved = (len(wanted) - len(missing)) / max(len(wanted), 1)
        coverage[cls.name] = round(resolved, 6)
        say(f"  {cls.name}: {len(wanted) - len(missing):,}/{len(wanted):,} entities "
            f"resolve to a structure ({resolved:.2%})")
        if resolved < c.rcsb_min_cov:
            raise FileNotFoundError(
                f"{cls.name}: {len(missing):,} of {len(wanted):,} entities have no "
                f"structure ({resolved:.2%}), so a graph built now would check those on "
                f"sequence alone and the structural guarantee would not be what its "
                f"name says. the first few are {', '.join(missing[:5])}")
        paths.update({entity: found[entity] for entity in wanted if entity in found})
    return paths, coverage

def seq_cache(names):
    '''
    where the sequence half lives. its name carries the sequence thresholds and
    the dataset list and nothing else, because nothing else moves an mmseqs hit.
    '''
    thresholds = (f"id{int(c.min_id * 100)}_cov{int(float(c.cov) * 100)}"
                  f"_ge{int(c.seq_floor * 100)}")
    listed = "|".join(sorted(names))
    return c.work / (f"pooled_seq_{thresholds}_{len(names)}tasks_"
                     f"{hashlib.sha1(listed.encode()).hexdigest()[:8]}.pkl")

def sequence_half(tasks, say):
    '''the pooled nodes and their sequence edges, searched once and then cached'''
    path = seq_cache([t.name for t in tasks])
    c.work.mkdir(parents=True, exist_ok=True)
    USED["sequence_half"] = path
    if path.exists():
        say(f"  reusing the pooled sequence search in {path.name}")
        return pickle.loads(path.read_bytes())
    if c.reuse_only:
        raise FileNotFoundError(
            f"config.reuse_only is set and there is no cached sequence search at {path}. that "
            "is an hour or more of mmseqs, so the miss is reported rather than started.")

    seqs, of_entity, aliases = collect(tasks, say)
    say(f"  sequence search over {len(seqs):,} pooled sequences ...")
    started = time.time()
    # the table holds every pair down to seq_floor with its identity; the EDGES are
    # the ones over min_id, and the rest are kept for the graded objective
    graded = [(node1, node2, identity)
              for node1, node2, identity in sequence_hits(seqs, say)]
    hits = [(node1, node2) for node1, node2, identity in graded
            if identity > c.min_id]
    say(f"    {len(graded):,} pairs in the table, {len(hits):,} of them over "
        f"id {c.min_id}, in {time.time() - started:.0f}s")
    half = {"nodes": sorted(seqs),
            "lengths": {node: len(sequence) for node, sequence in seqs.items()},
            "aliases": {node: sorted(entities) for node, entities in aliases.items()},
            "of_entity": {entity: sorted(nodes)
                          for entity, nodes in of_entity.items()},
            "sequence": hits, "graded": graded}
    path.write_bytes(pickle.dumps(half))
    return half

def tag(names, views):
    '''a name for one build, carrying every threshold and dataset that shaped it'''
    thresholds = (f"id{int(c.min_id * 100)}_cov{int(float(c.cov) * 100)}"
                  f"_ge{int(c.seq_floor * 100)}")
    if views: thresholds += f"_tm{int(c.split_tm * 100)}"
    # the edge kinds are in the name because they decide the components, and a build
    # that drops one is a different graph under the same thresholds
    listed = ("|".join(sorted(names)) + "//" + "|".join(sorted(views))
              + "//" + "|".join(KINDS))
    return (f"{thresholds}_{len(names)}tasks_"
            f"{hashlib.sha1(listed.encode()).hexdigest()[:8]}")

def build(tasks, say=print):
    '''the pooled protein graph: nodes, typed edges, and the components they imply'''
    views = {cls.name for cls in tasks if "structure" in cls.views}
    half = sequence_half(tasks, say)
    nodes, of_entity = half["nodes"], half["of_entity"]
    known = set(nodes)

    edges = {kind: [] for kind in KINDS}
    edges["sequence"] = sorted({tuple(sorted((node1, node2)))
                                for node1, node2 in half["sequence"]
                                if node1 != node2 and node1 in known and node2 in known})
    say(f"  {len(edges['sequence']):,} sequence pairs at id {c.min_id} / cov {c.cov}")

    coverage = {}
    if views:
        paths, coverage = structure_paths(tasks, views, of_entity, say)
        # a structural hit is between two ENTITIES, since that is what has a
        # structure; it becomes an edge between every node each of them carries
        pairs = set()
        for entity1, entity2 in structure_hits(paths, say):
            for node1 in of_entity.get(entity1, ()):
                for node2 in of_entity.get(entity2, ()):
                    if node1 != node2: pairs.add(tuple(sorted((node1, node2))))
        edges["structure"] = sorted(pairs)
        say(f"  {len(edges['structure']):,} structure pairs at tm {c.split_tm} over "
            f"{len(paths):,} chains")

    groups = merge(nodes, *(edges[kind] for kind in KINDS))
    groups = sorted((sorted(group) for group in groups),
                    key=lambda group: (-len(group), group[0]))
    comp_of = {node: i for i, group in enumerate(groups) for node in group}
    say(f"  {len(nodes):,} proteins, {sum(len(edges[kind]) for kind in KINDS):,} edges, "
        f"{len(groups):,} components, largest {len(groups[0]):,} "
        f"({100 * len(groups[0]) / len(nodes):.1f}%)")

    return {"tasks": sorted(cls.name for cls in tasks), "views": sorted(views),
            "nodes": nodes, "lengths": half["lengths"], "aliases": half["aliases"],
            "of_entity": of_entity, "edges": edges, "groups": groups,
            # every searched pair WITH its identity, including the ones under the
            # threshold. the edges above are the thresholded view of this
            "graded": half["graded"], "comp_of": comp_of, "coverage": coverage,
            "tag": tag([cls.name for cls in tasks], views)}

def build_ligands(tasks, say=print):
    '''the molecule graph, which is its own graph and not part of the protein one'''
    names = sorted(cls.name for cls in tasks if "ligand" in cls.views)
    if not names: return None
    key = (f"tan{int(c.tanimoto * 100)}_ge{int(c.tan_floor * 100)}_{len(names)}tasks_"
           f"{hashlib.sha1('|'.join(names).encode()).hexdigest()[:8]}")
    cache = c.work / f"pooled_ligands_{key}.pkl"
    USED["ligands"] = cache
    if cache.exists():
        say(f"  reusing {cache.name}")
        return pickle.loads(cache.read_bytes())

    # read from the rows rather than from ligands(): the rows are what retention is
    # counted over, so collecting anywhere else risks a row with no molecule node
    smiles = {}
    for cls in tasks:
        if "ligand" not in cls.views: continue
        for row in cls.rows():
            molecule = str(row.get(cls.ligand) or "").strip()
            if molecule: smiles.setdefault(ligand_node(molecule), molecule)
    if not smiles: return None

    say(f"  tanimoto over {len(smiles):,} molecules ...")
    started = time.time()
    node_of_molecule = {molecule: node for node, molecule in smiles.items()}
    best = {}
    for one, other, score in ligand_links({molecule: [molecule]
                                           for molecule in smiles.values()}):
        node1, node2 = node_of_molecule[one], node_of_molecule[other]
        if node1 == node2: continue
        pair = (node1, node2) if node1 < node2 else (node2, node1)
        if score > best.get(pair, -1.0): best[pair] = score
    graded = sorted((node1, node2, score) for (node1, node2), score in best.items())
    edges = sorted(pair for pair, score in best.items() if score >= c.tanimoto)
    say(f"    {len(graded):,} pairs over tanimoto {c.tan_floor}, {len(edges):,} of them "
        f"over {c.tanimoto}, in {time.time() - started:.0f}s")
    groups = sorted((sorted(group) for group in merge(sorted(smiles), edges)),
                    key=lambda group: group[0])
    molecule_graph = {"nodes": sorted(smiles), "edges": {"ligand": edges},
                      "graded": graded, "groups": groups,
                      "comp_of": {node: i for i, group in enumerate(groups)
                                  for node in group}}
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_bytes(pickle.dumps(molecule_graph))
    return molecule_graph

def strata_weights(tasks, protein_graph, molecule_graph, say=print):
    '''
    the same weights again, counted one label class at a time, for the datasets
    whose label names a real class.

    a class is a (dataset, class) key, so a stratum of one dataset never
    constrains another. the shares are counted the same way the dataset weights
    are -- a row shared out over the nodes it rides on -- so the two bands are in
    the same currency. a row can belong to several classes at once (an shs27k
    pair is often both binding and reaction) and then it is counted under each,
    which is why these are only ever banded and never added to the objective.
    '''
    per_node = {}
    molecules = set(molecule_graph["nodes"]) if molecule_graph else set()
    components_of = {}
    for cls in tasks:
        # only the datasets whose label names a real class. the rest have their label
        # cut into quantile bins, which are even by construction and not a class a
        # datapoint belongs to, so there is nothing there to keep in proportion
        if not cls.stratify: continue
        rows = list(cls.rows())
        for row, names in zip(rows, cls.strata(rows)):
            if not names: continue
            nodes = row_nodes(cls, row, protein_graph["of_entity"])
            if not nodes or any(node not in protein_graph["comp_of"] for node in nodes):
                continue
            carried_by = list(nodes)
            if "ligand" in cls.views and molecule_graph:
                molecule = str(row.get(cls.ligand) or "").strip()
                if molecule:
                    node = ligand_node(molecule)
                    if node not in molecules: continue
                    carried_by.append(node)
            share = 1.0 / len(carried_by)
            for name in names:
                key = f"{cls.name}/{name}"
                for node in carried_by:
                    per_node.setdefault(node, Counter())[key] += share
                components_of.setdefault(key, set()).update(
                    protein_graph["comp_of"][node] for node in nodes)

    # a class can only be banded if it CAN be spread. one whose rows all ride on a
    # single component goes wherever that component goes, so a ceiling on it makes the
    # whole model infeasible rather than balanced. so a class is banded only where it
    # reaches enough separate components to be divided at all, and the rest are reported
    seen = sorted({key for counts in per_node.values() for key in counts})
    per_dataset = Counter(key.split("/", 1)[0] for key in seen)
    keys, unbandable = [], {}
    for key in seen:
        dataset = key.split("/", 1)[0]
        spread = len(components_of.get(key, ()))
        rows = sum(counts.get(key, 0) for counts in per_node.values())
        why = ("too many classes in the dataset"
               if per_dataset[dataset] > MAX_CLASSES_PER_DATASET else
               "in too few components" if spread < MIN_CLASS_COMPONENTS else
               "too few rows" if rows < MIN_CLASS_ROWS else None)
        if why is None: keys.append(key)
        else: unbandable[key] = {"components": spread, "rows": round(rows, 1),
                                 "why": why}
    for node in per_node:
        per_node[node] = Counter({key: v for key, v in per_node[node].items()
                                  if key in set(keys)})
    why = Counter(one["why"] for one in unbandable.values())
    say(f"  {len(keys):,} label classes banded over "
        f"{len({key.split('/', 1)[0] for key in keys})} datasets, {len(unbandable):,} "
        f"left out (" + ", ".join(f"{count:,} {reason}"
                                  for reason, count in sorted(why.items())) + ")")
    return per_node, keys, unbandable

def weights(tasks, protein_graph, molecule_graph, say=print):
    '''
    what each node is worth to each dataset, as the rows naming it.

    a row is SHARED OUT between its nodes, 1/n each, rather than counted once
    against every one of them. the model weights nodes and a row is only linear
    in the assignment when it rides on one node, so counting it whole on each
    end overcounts every pair, every affinity, and every entity whose several
    chains are several nodes -- and then the band holds on the proxy while the
    real row fractions drift off it. shared out, the weights sum back to exactly
    the rows wherever the nodes end up together, which an entity's chains always
    do. split.py measures the true counts from the manifests regardless.
    '''
    per_node, totals, unretainable = {}, Counter(), Counter()
    molecules = set(molecule_graph["nodes"]) if molecule_graph else set()
    for cls in tasks:
        for row in cls.rows():
            totals[cls.name] += 1
            nodes = row_nodes(cls, row, protein_graph["of_entity"])
            if not nodes or any(node not in protein_graph["comp_of"] for node in nodes):
                unretainable[cls.name] += 1
                continue
            carried_by = list(nodes)
            if "ligand" in cls.views and molecule_graph:
                molecule = str(row.get(cls.ligand) or "").strip()
                if molecule:
                    node = ligand_node(molecule)
                    if node not in molecules:
                        unretainable[cls.name] += 1
                        continue
                    carried_by.append(node)
            share = 1.0 / len(carried_by)
            for node in carried_by:
                per_node.setdefault(node, Counter())[cls.name] += share
    say(f"  {sum(totals.values()):,} rows over {len(per_node):,} weighted nodes "
        f"({sum(unretainable.values()):,} can never be retained)")
    return per_node, totals, unretainable
