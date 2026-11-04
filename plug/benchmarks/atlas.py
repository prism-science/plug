# atlas md flexibility, one item per simulated pdb chain
'''
One simulated pdb chain per file, with a per residue RMSF from three replica
runs. It ships no split of its own. ATLAS is about how a shape moves, so two
chains that fold the same way can't sit in different splits however far their
sequences have drifted — this is the one benchmark that declares the structure
view, and its split checks sequence and fold both. There is no fallback if the
structures are missing: a graph built without the structure half is a different
graph and must not be cached under the same name. RMSF is per residue but atlas
is scored one chain at a time, so the class a chain belongs to is how mobile it
is on average.
'''

import csv

from . import Benchmark
from .. import config as c

class Atlas(Benchmark):
    '''per residue rmsf for simulated pdb chains.'''

    name = "atlas"
    # every item is about one pdb chain
    key = "pdb_chain"
    published = ()
    # shape leaks too, not just sequence
    views = ("structure",)
    # what it is scored on, and so what the split is stratified on
    label = "rmsf"

    @classmethod
    def strata(cls, rows):
        '''bin chains by mean mobility: the label is per residue, the score per chain'''
        summarised = [{**row, "rmsf": sum(row["rmsf"]) / max(len(row["rmsf"]), 1)}
                      for row in rows]
        return Benchmark.strata.__func__(cls, summarised)

    @classmethod
    def rows(cls):
        '''
        one item per chain, with the mean rmsf over the three runs and the three runs
        themselves.
        '''
        for path in sorted((c.data / "atlas" / "rmsf").glob("*.tsv")):
            seq, replicas = [], ([], [], [])
            with open(path, newline="") as f:
                for r in csv.DictReader(f, delimiter="\t"):
                    aa = (r.get("seq") or "").strip()
                    if not aa: continue
                    seq.append(aa)
                    for k in range(3): replicas[k].append(float(r[f"RMSF_R{k + 1}"]))
            if not seq: continue
            yield {"sequence": "".join(seq).upper(),
                   "pdb_chain": path.stem,
                   # the mean rmsf over the three runs, and the three runs themselves
                   "rmsf": [sum(v) / 3 for v in zip(*replicas)],
                   "rmsf_replicas": [list(v) for v in replicas]}


    @classmethod
    def structures(cls):
        '''
        the simulated structures, so two chains that fold the same way can't sit in
        different splits however far their sequences have drifted.
        '''
        d = c.data / "atlas" / "pdb"
        found = {p.stem: p for p in sorted(d.glob("*.pdb"))} if d.exists() else {}
        if not found:
            raise FileNotFoundError(f"atlas has no structures in {d} — "
                                    "rerun data/download_atlas.sh to fetch them")
        return found
