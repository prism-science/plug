# allobench allosteric and active sites, one item per entry
'''
Proteins with a known allosteric site and a known active site. It ships no
split of its own. The site residues arrive numbered the way the pdb file writes
them, as labels like "B-THR-7", so they are renumbered against the uniprot
sequence through sifts and then checked against the residue they name; a
position that does not convert or does not match is dropped rather than shipped
wrong. Scoring is one auroc per protein over its residues.
'''

import ast
import csv
import re

from . import Benchmark
from .. import config as c
from .. import rcsb
from ..sifts import THREE, sites

def terms(v):
    '''a residue column arrives either as a list or as a string that looks like one.'''
    if isinstance(v, str): return ast.literal_eval(v) if v.startswith("[") else []
    return list(v) if hasattr(v, "__len__") else []

def labels(v):
    '''
    parse site residue labels into the shape sifts renumbers, so "B-THR-7" becomes
    ("B", "T", 7).
    '''
    out = []
    for lab in terms(v):
        m = re.match(r"^([A-Za-z0-9]+)-([A-Z]{3})-(-?\d+)", str(lab).strip())
        if m and m.group(2) in THREE: out.append((m.group(1), THREE[m.group(2)], int(m.group(3))))
    return out

class Allobench(Benchmark):
    '''allosteric and active site residues, numbered against uniprot.'''

    name = "allobench"
    # every item is about one target protein
    key = "target_id"
    published = ()
    # shape leaks too: the entries are fetched from rcsb, one chain each
    views = ("structure",)
    # what it is scored on, and so what the split is stratified on
    label = "allosteric_site"



    @classmethod
    def strata(cls, rows):
        '''bin proteins by how much of them is site: the label is per residue'''
        summarised = [{**row, "allosteric_site": len(row["allosteric_site"])
                       / max(len(row["sequence"]), 1)} for row in rows]
        return Benchmark.strata.__func__(cls, summarised)

    @classmethod
    def rows(cls):
        '''
        one item per entry, with both sites counted along the sequence shipped with
        it.
        '''
        with open(c.data / "allobench" / "AlloBench.csv", newline="") as f:
            for r in csv.DictReader(f):
                s = (r.get("sequence") or "").strip().upper()
                if not s: continue
                pdb, acc = r.get("allosteric_pdb", ""), r.get("pdb_uniprot", "")
                yield {"sequence": s,
                       "target_id": r.get("target_id", ""),
                       "gene": r.get("target_gene", ""),
                       "organism": r.get("organism", ""),
                       "uniprot": acc,
                       "pdb": pdb,
                       # the residues making up each site, counted along the sequence above
                       "allosteric_site": sites(labels(r.get("allosteric_site_residue", "")),
                                                pdb, acc, s),
                       "active_site": [p for p in terms(r.get("active_site_residue", ""))
                                       if isinstance(p, int) and 1 <= p <= len(s)]}

    @classmethod
    def structures(cls):
        '''
        the one chain of each entry that is the protein this benchmark is about,
        fetched from rcsb and cached. an allobench entry is a complex and its
        allosteric site sits on one chain of it, so the chain whose observed sequence
        best matches the sequence the benchmark carries is the one kept — the same
        chain the site residues were renumbered against.
        '''
        return rcsb.structures(cls)
