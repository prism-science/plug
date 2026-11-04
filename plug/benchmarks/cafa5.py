# cafa5 gene ontology function prediction, one item per protein
'''
The CAFA5 function test set as packaged by bioreason-pro. The labels are gene
ontology terms across the three branches — biological process, molecular
function and cellular component — and the proteins were held back by date
rather than by similarity, which is exactly the kind of split a homology split
is meant to replace. It ships no split of its own.
'''

import ast

import pandas as pd

from . import Benchmark
from .. import config as c

def terms(v):
    '''a go column arrives either as a list or as a string that looks like one.'''
    if isinstance(v, str): return ast.literal_eval(v) if v.startswith("[") else []
    return list(v) if hasattr(v, "__len__") else []

class Cafa5(Benchmark):
    '''go term prediction on a temporal holdout.'''

    name = "cafa5"
    # every item is about one protein
    key = "protein_id"
    published = ()
    # what it is scored on, and so what the split is stratified on
    label = "go"



    # a term with fewer positives than this is too rare to constrain a split with
    min_positives = 20

    @classmethod
    def strata(cls, rows):
        '''one class per go term a protein carries, keeping the terms with enough positives'''
        from collections import Counter
        seen = Counter(term for row in rows for branch in ("bp", "mf", "cc")
                       for term in row[f"go_{branch}"])
        keep = {term for term, count in seen.items() if count >= cls.min_positives}
        return [tuple(sorted({term for branch in ("bp", "mf", "cc")
                              for term in row[f"go_{branch}"]} & keep))
                for row in rows]

    @classmethod
    def rows(cls):
        '''one item per protein, with its go terms split by branch.'''
        path = c.data / "cafa5" / "test" / "data" / "test-00000-of-00001.parquet"
        for r in pd.read_parquet(path).itertuples():
            yield {"sequence": str(r.sequence).strip().upper(),
                   "protein_id": r.protein_id,
                   "organism": r.organism,
                   # the three go branches: biological process, molecular function,
                   # cellular component
                   "go_bp": terms(r.go_bp),
                   "go_mf": terms(r.go_mf),
                   "go_cc": terms(r.go_cc)}
