'''
One file per dataset, each saying only how to read its own rows.

A dataset declares the field naming the protein a row is about and the field
holding its sequence, as tuples when a row is about a pair, plus any further
view it can leak through: "structure", "ligand".
'''

import gzip

def iter_fasta(path):
    '''a fasta, gzipped or not, as (id, sequence) -- the id is the first word'''
    op = gzip.open if str(path).endswith(".gz") else open
    with op(path, "rt") as f:
        acc, seq = None, []
        for line in f:
            if line.startswith(">"):
                if acc is not None: yield acc, "".join(seq)
                acc, seq = line[1:].strip().split()[0], []
            else: seq.append(line.strip())
        if acc is not None: yield acc, "".join(seq)

class Benchmark:
    '''one functional dataset: its rows, and the ways it can leak'''

    name = ""
    key = "sequence"        # the field naming the protein (a tuple of two for a pair)
    seq = "sequence"        # the field holding its sequence (likewise)
    views = ()              # "structure", "ligand" — what to search beyond sequence
    ligand = "smiles"       # for a protein-ligand dataset, the field holding it
    published = ()          # the splits it shipped with. recorded, never used
    label = None            # the field it is scored on, which strata() turns into classes
    bins = 4                # how many quantile bins a numeric label is cut into
    # whether the split is BANDED on those classes. true only where the label names a
    # class a datapoint belongs to -- dms's assay type, shs27k's interaction mode. a
    # continuous label cut into quantile bins is not a class: the bins are an artefact
    # of the cut and even by construction, so banding them constrains the solve without
    # answering what rules.md asks. every dataset's gaps are still reported
    stratify = False

    @classmethod
    def rows(cls):
        '''each dataset writes this one and nothing else: one dict per datapoint'''
        raise NotImplementedError

    @classmethod
    def structures(cls):
        '''entity -> a structure file, for a dataset declaring the structure view'''
        return {}

    @classmethod
    def ligands(cls):
        '''entity -> its smiles'''
        return {}

    @classmethod
    def strata(cls, rows):
        '''
        the class each row belongs to, in rows() order, read off `label`: a
        number is cut into `bins` quantile bins over the whole dataset, a list
        or set puts the row in one class per entry, and anything else is a class
        already. a dataset whose label is per-residue overrides this and
        summarises it first.

        these are what the split is stratified on: a split can be leakage-free
        and the right size and still ask the test set a question the training set
        never covered, which is what happens to dms when one of its five assay
        types lands mostly on one side.
        '''
        if not cls.label: return [() for _ in rows]
        values = [row.get(cls.label) for row in rows]
        if all(isinstance(v, (list, set, tuple)) for v in values):
            return [tuple(f"{cls.label}={one}" for one in sorted(set(v))) for v in values]
        try:
            numbers = [float(v) for v in values]
        except (TypeError, ValueError):
            return [(f"{cls.label}={v}",) for v in values]
        import numpy as np
        edges = np.quantile(numbers, [i / cls.bins for i in range(1, cls.bins)])
        return [(f"{cls.label}~q{int(np.searchsorted(edges, v))}",) for v in numbers]

    @classmethod
    def keyfields(cls):
        '''key as a tuple, so nothing downstream asks whether a row names a pair'''
        return (cls.key,) if isinstance(cls.key, str) else tuple(cls.key)

    @classmethod
    def seqfields(cls):
        '''the same for seq'''
        return (cls.seq,) if isinstance(cls.seq, str) else tuple(cls.seq)

from .megascale import Megascale
from .fireprot import Fireprot
from .dms import Dms
from .cafa5 import Cafa5
from .allobench import Allobench
from .ppi import Ppi
from .shs27k import Shs27k
from .passerrank import Passerrank
from .lp_pdbbind import LpPdbbind
from .bindingdb import Bindingdb
from .atlas import Atlas

REGISTRY = {b.name: b for b in Benchmark.__subclasses__()}
# every dataset a split or screen runs over, in a fixed order so the tag over them is stable
DATASETS = ("allobench", "atlas", "bindingdb", "cafa5", "dms", "fireprot", "lp-pdbbind",
            "megascale", "passerrank", "ppi", "shs27k")
