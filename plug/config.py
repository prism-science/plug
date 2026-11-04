'''
Every setting plug uses, in one tracked file: the paths, the search binaries and
the thresholds that define leakage. Nothing is read from the environment, so a
change to any of them shows up in a diff. Point `data` and `work` somewhere else
with a symlink, or by editing them here.
'''

from pathlib import Path

repo = Path(__file__).resolve().parent.parent

# the eleven datasets, the caches, and where a solved split is written
data = repo / "data"
work = repo / "work"
splits = repo / "results"

# the search binaries, and how they run
mmseqs = "mmseqs"
foldseek = "foldseek"
threads = "8"
gpu = False                     # True, with a gpu build of foldseek, searches on cuda

# the reservoirs an unlabeled training set is sampled from, and where it is written
uniref_fa = data / "uniref" / "uniref90.fasta.gz"
pdb_dir = data / "pdb"
trainsets = repo / "out"

# where pdb entries are fetched and the single chains pulled out of them kept
rcsb_dir = data / "rcsb"
rcsb_workers = 12               # parallel downloads; past ~16 rcsb throttles
rcsb_min_cov = 1.0              # a dataset with a missing structure is a hole, not a split
sifts = data / "sifts" / "uniprot_segments_observed.csv.gz"

# refuse to START a search that is not cached. mmseqs over the pool is an hour and
# foldseek over 22k chains is a day, and the caches are keyed on their inputs, so a
# miss means an input moved -- better an error than a job noticed tomorrow
reuse_only = False

min_id = 0.3                    # mmseqs identity: two proteins are the same family
cov = "0.8"                     # ..over this much of either one (cov-mode 1 and 2)
sens = "7.5"
split_tm = 0.7                  # foldseek TM both ways round: the same fold, confidently
tanimoto = 0.5                  # ecfp4 (morgan r=2, 2048 bits, no stereo): one chemotype
struct_floor = 0.4              # the cached foldseek table is kept down to this TM
# and the same for sequence: the cached mmseqs table is kept down to this identity, far
# below min_id. the pairs between it and min_id are not edges -- they are what a graded
# objective minimises
seq_floor = 0.0
tan_floor = 0.3                 # and the cached tanimoto pairs down to this

# HiGHS, not the CBC Lo-Hi used: python-mip's bundled CBC segfaults on macos arm64 on a
# two-variable model, and a split nobody can rerun on their laptop is not reproducible.
# the split is reproducible for one solver, so this is part of the specification
solver = "HIGHS"

THRESHOLDS = {"min_id": min_id, "coverage": float(cov), "cov_modes": "1 union 2",
              "sensitivity": sens, "split_tm": split_tm, "tanimoto": tanimoto,
              "fingerprint": "ecfp4 (morgan r=2, 2048 bits, no stereochemistry)"}
