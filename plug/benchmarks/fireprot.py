# fireprot ddG, one item per substitution on the uniprot sequence
'''
FireProtDB, curated by ThermoMPNN. It ships train/val/test split by sequence
and its positions are numbered against the uniprot sequence rather than the pdb
one. The benchmark also carries the pdb sequence of each protein and the mutant
of every measured substitution. FireProt
spells no mutant out, so each one is applied here — and only where the residue
it names really is at that position, which is the same check the scoring makes
before it trusts a row.
'''

import csv

from . import Benchmark
from .. import config as c
from .. import rcsb

class Fireprot(Benchmark):
    '''ddG for substitutions, numbered against uniprot and split by sequence.'''

    name = "fireprot"
    # every item is about one wild type protein
    key = "pdb"
    published = ("train", "val", "test")
    # shape leaks too: the entries are fetched from rcsb, one chain each
    views = ("structure",)
    # what it is scored on, and so what the split is stratified on
    label = "ddg"

    @classmethod
    def rows(cls):
        '''one item per substitution with a measured ddg.'''
        for split in ("train", "val", "test"):
            path = c.data / "fireprot" / f"{split}.csv"
            if not path.exists(): continue
            with open(path, newline="") as f:
                for r in csv.DictReader(f):
                    if not (r.get("ddG") or "").strip(): continue
                    yield {"sequence": r["sequence"].strip().upper(),
                           "mutation": f"{r['wild_type']}{r['position']}{r['mutation']}",
                           "ddg": float(r["ddG"]),
                           "pdb": r.get("pdb_id_corrected", ""),
                           "split": split}


    @classmethod
    def structures(cls):
        '''
        the one chain of each wild type's entry, fetched from rcsb and cached.
        fireprot is keyed on the entry and numbers its positions against the uniprot
        sequence, so the chain kept is the one whose observed sequence best matches
        the sequence the benchmark carries rather than whichever chain the entry
        happens to list first.
        '''
        return rcsb.structures(cls)
