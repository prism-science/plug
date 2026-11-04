'''
Renumber pdb residue numbers into uniprot sequence positions, so a site named
the way a pdb file numbers it lines up with the uniprot sequence.
'''

import csv
import gzip
import re
from functools import lru_cache

from . import config as c

THREE = {"ALA":"A","ARG":"R","ASN":"N","ASP":"D","CYS":"C","GLN":"Q","GLU":"E","GLY":"G",
         "HIS":"H","ILE":"I","LEU":"L","LYS":"K","MET":"M","PHE":"F","PRO":"P","SER":"S",
         "THR":"T","TRP":"W","TYR":"Y","VAL":"V"}

def number(cell):
    '''the number off a pdb residue number, which may carry a letter ("27A")'''
    match = re.match(r"^(-?\d+)", (cell or "").strip())
    return int(match.group(1)) if match else None

@lru_cache(maxsize=1)
def segments():
    '''(pdb, chain, accession) -> the stretches sifts maps, as (first, last, shift)'''
    if not c.sifts.exists():
        raise FileNotFoundError(f"{c.sifts} is missing — run data/download_sifts.sh")
    stretches = {}
    with gzip.open(c.sifts, "rt") as f:
        next(f)                                  # a dated comment sits above the header
        for row in csv.DictReader(f):
            first = number(row["PDB_BEG"])
            last = number(row["PDB_END"])
            uniprot_first = number(row["SP_BEG"])
            if first is None or last is None or uniprot_first is None: continue
            key = (row["PDB"].lower(), row["CHAIN"], row["SP_PRIMARY"])
            stretches.setdefault(key, []).append((first, last, uniprot_first - first))
    return stretches

def to_uniprot(pdb, chain, position, accession):
    '''one pdb residue number on the uniprot sequence, or None if unmapped'''
    for first, last, shift in segments().get((pdb.lower(), chain, accession), ()):
        if first <= position <= last: return position + shift
    return None

def sites(labels, pdb, accession, sequence):
    '''
    renumber site residues, keeping the ones that convert AND then land on the
    residue they name -- the same promise the mutation datasets make.
    '''
    positions = set()
    for chain, residue, position in labels:
        moved = to_uniprot(pdb, chain, position, accession)
        if moved and 1 <= moved <= len(sequence) and sequence[moved - 1] == residue:
            positions.add(moved)
    return sorted(positions)
