# proteingym deep mutational scanning, one item per measured variant
'''
ProteinGym's substitution benchmark, one row per measured variant and grouped
by assay. Every assay targets one protein, so the assay id is what names the
protein here rather than the sequence — two assays on the same protein still
measure different things, and the split has to keep them together because they
are the same protein. It ships no split of its own.
'''

import csv
import sys

from . import Benchmark
from .. import config as c

# target sequences are long enough to trip the default csv field limit
csv.field_size_limit(sys.maxsize)

class Dms(Benchmark):
    '''deep mutational scanning scores, grouped by assay.'''

    name = "dms"
    # every item is about the protein its assay targets
    key = "dms_id"
    published = ()
    # what it is scored on, and so what the split is stratified on
    label = "assay"
    # a real class per datapoint, so the split is banded on it
    stratify = True

    @classmethod
    def rows(cls):
        '''
        one item per measured variant, with the assay's target sequence and what that
        assay measured.
        '''
        folder = c.data / "proteingym"
        # one row per assay: its target sequence, what it measured, and the protein
        with open(folder / "DMS_substitutions.csv", newline="") as f:
            meta = {r["DMS_id"]: (r["target_seq"].strip().upper(),
                                  r["coarse_selection_type"], r["UniProt_ID"])
                    for r in csv.DictReader(f)}
        for dms_id, (seq, assay, uniprot) in meta.items():
            path = folder / "DMS_ProteinGym_substitutions" / f"{dms_id}.csv"
            if not path.exists(): continue
            with open(path, newline="") as f:
                for r in csv.DictReader(f):
                    if not (r.get("DMS_score") or "").strip(): continue
                    yield {"sequence": seq, "mutation": r["mutant"],
                           "score": float(r["DMS_score"]), "assay": assay,
                           "dms_id": dms_id, "uniprot": uniprot}

