'''
Sample a reservoir into an unlabeled training set that leaks into no benchmark.

    python3 -m plug.sample_unlabeled_trainset --seq      uniref90, dropped on mmseqs identity
    python3 -m plug.sample_unlabeled_trainset --struct   the pdb, dropped on foldseek TM
    python3 -m plug.sample_unlabeled_trainset --both     the pdb, dropped on either

That flag is the only argument. It is the same leakage rule split.py uses, with
the same thresholds from config.py, pointed the other way: split.py carves the
eleven datasets into splits, this holds a whole reservoir clear of every one of
them at once, in every split, since a model being tested should never have seen
any of it. A candidate is thrown out if it matches any benchmark protein.

The reservoir is never held in memory. It is read once per round in a stream and
reservoir-sampled, the sample is searched against the benchmarks, the matches
are thrown out, and sampling repeats until the quota is met. A hundred million
sequences cost one pass per round and not one database.

Writes out/trainset_<reservoir>.fasta. The paths are in config.py.
'''

import gzip
import random
import subprocess
import sys
import tempfile
from pathlib import Path

from . import config as c
from . import graph as G
from .benchmarks import ELEVEN, REGISTRY

RESERVOIRS = ("seq", "struct", "both")
QUOTA = 1_000_000           # how many clean sequences to end up with
OVERSAMPLE = 1.5            # draw this much more than the shortfall each round
MIN_LEN, MAX_LEN = 20, 1000 # no fragments, no giant non-physiological sequences
SEED = 0
# one match is enough to drop a candidate, so mmseqs need not collect them all
MAX_SEQS = "50"
# structure file endings foldseek can read
STRUCTS = (".pdb", ".cif", ".ent", ".mmcif", ".pdb.gz", ".cif.gz", ".ent.gz", ".mmcif.gz")

class Unlabeled:
    '''
    the sequences that survived, keyed by uniprot code. indexes like a list of
    {sequence, id, name}, so it can be handed straight to a torch DataLoader.
    '''

    def __init__(self, seqs):
        '''seqs: "<code> <what it is>" -> sequence'''
        self.seqs = seqs
        self.items = [{"sequence": sequence, "id": name.split(None, 1)[0], "name": name}
                      for name, sequence in seqs.items()]

    @classmethod
    def from_fasta(cls, path):
        '''read a training set written earlier, so it does not have to be sampled again'''
        return cls({f"{accession} {note}".rstrip(): sequence
                    for accession, note, sequence in read_fasta(path)})

    def __len__(self): return len(self.items)

    def __getitem__(self, i): return self.items[i]

    def write_fasta(self, path):
        '''write it out, one record per sequence'''
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as out:
            for name, sequence in self.seqs.items(): out.write(f">{name}\n{sequence}\n")
        return path

def read_fasta(path):
    '''
    a fasta, gzipped or not, as (id, description, sequence). the description is
    kept so the written training set still says what each uniref sequence is.
    '''
    op = gzip.open if str(path).endswith(".gz") else open
    with op(path, "rt") as f:
        accession, note, seq = None, "", []
        for line in f:
            if line.startswith(">"):
                if accession is not None: yield accession, note, "".join(seq)
                head = line[1:].strip().split(None, 1)
                accession, note, seq = head[0], (head[1] if len(head) > 1 else ""), []
            else: seq.append(line.strip())
        if accession is not None: yield accession, note, "".join(seq)

def mmseqs(*args):
    '''an mmseqs command, quietly'''
    subprocess.run([str(c.mmseqs), *args, "--threads", c.threads, "-v", "1"], check=True)

def foldseek(*args):
    '''a foldseek command, quietly. it chatters on stdout, so that goes to stderr'''
    subprocess.run([str(c.foldseek), *args, "-v", "1"], check=True, stdout=sys.stderr)

def sequence_target(classes, tmp, say):
    '''
    every benchmark sequence as one mmseqs database. read from the rows the way
    graph.py reads them, so the sequences held clear are exactly the pooled nodes.
    '''
    seqs, of_entity, _ = G.collect(classes, say)
    fasta, database = tmp / "benchmarks.fasta", tmp / "benchmarks_db"
    with open(fasta, "w") as out:
        for node in sorted(seqs): out.write(f">{node}\n{seqs[node]}\n")
    mmseqs("createdb", str(fasta), str(database), "--dbtype", "1")
    return database, of_entity

def structure_target(classes, of_entity, tmp, say):
    '''
    every benchmark structure as one foldseek database: the datasets declaring
    the structure view, one file per entity, as graph.py finds them. a shortfall
    stops the run there for the same reason it does there.
    '''
    views = {cls.name for cls in classes if "structure" in cls.views}
    paths, _ = G.structure_paths(classes, views, of_entity, say)
    pool = tmp / "benchmark_structs"
    pool.mkdir()
    # numbered, as graph.py does, because entity names are not always safe file names
    for i, entity in enumerate(sorted(paths)):
        (pool / f"s{i}.pdb").write_bytes(Path(paths[entity]).read_bytes())
    database = tmp / "benchmark_structs_db"
    foldseek("createdb", str(pool), str(database))
    return database

def sequence_leakers(pool, target, tmp):
    '''
    the candidates over min_id identity to a benchmark sequence, over cov of
    either one. both coverage modes, either one counts: mode 1 catches a
    candidate holding a whole benchmark protein, mode 2 one sitting inside one.
    '''
    fasta, query = tmp / "pool.fasta", tmp / "pool_db"
    # the accession alone is the query name: mmseqs cuts a header at the first space
    with open(fasta, "w") as out:
        for accession, _, sequence in pool: out.write(f">{accession}\n{sequence}\n")
    mmseqs("createdb", str(fasta), str(query), "--dbtype", "1")
    leakers = set()
    for cov_mode in ("1", "2"):
        result, hits = tmp / f"seq_res{cov_mode}", tmp / f"seq{cov_mode}.m8"
        mmseqs("search", str(query), str(target), str(result), str(tmp / f"ss{cov_mode}"),
               "-s", c.sens, "-e", G.EVALUE, "-c", c.cov, "--cov-mode", cov_mode,
               "--max-seqs", MAX_SEQS)
        mmseqs("convertalis", str(query), str(target), str(result), str(hits),
               "--format-output", "query,target,fident")
        for line in open(hits):
            candidate, _, identity = line.split("\t")[:3]
            if float(identity) > c.min_id: leakers.add(candidate)
    return leakers

def structure_leakers(query, target, tmp):
    '''
    the candidates over split_tm to a benchmark structure, read both ways round
    as graph.py reads it: the shape version of the two coverage modes.
    '''
    result, hits = tmp / "struct_res", tmp / "struct.m8"
    foldseek("search", str(query), str(target), str(result), str(tmp / "fs"),
             "-e", G.EVALUE, "--alignment-type", "1", "-a", "--threads", c.threads,
             *(("--gpu", "1") if c.gpu else ()))
    foldseek("convertalis", str(query), str(target), str(result), str(hits),
             "--format-output", "query,target,qtmscore,ttmscore", "--threads", c.threads)
    leakers = set()
    for line in open(hits):
        candidate, _, query_tm, target_tm = line.split("\t")[:4]
        if max(float(query_tm), float(target_tm)) > c.split_tm: leakers.add(candidate)
    return leakers

def sample_stream(items, n, rng):
    '''reservoir sampling: n items from a stream without loading it, each equally likely'''
    picked, seen = [], 0
    for item in items:
        seen += 1
        if len(picked) < n:
            picked.append(item)
        else:
            j = rng.randrange(seen)
            if j < n: picked[j] = item
        if seen % 20_000_000 == 0: print(f"    ...read {seen:,}", flush=True)
    return picked

def sequence_sample(n, used, rng):
    '''n uniref sequences not drawn before, within the length bounds'''
    items = ((accession, note, sequence)
             for accession, note, sequence in read_fasta(c.uniref_fa)
             if accession not in used and MIN_LEN <= len(sequence) <= MAX_LEN)
    pool = sample_stream(items, n, rng)
    used.update(accession for accession, _, _ in pool)
    return pool

def structure_sample(n, used, rng, tmp):
    '''
    n structure files not drawn before, as foldseek's database of them at
    tmp/candidates_db and the chains' sequences. the files are sorted before
    sampling, so the draw does not depend on the order the disk lists them in.
    '''
    files = (path for path in sorted(c.pdb_dir.rglob("*"))
             if path.name.endswith(STRUCTS) and path.name not in used)
    picks = sample_stream(files, n, rng)
    if not picks: return []
    used.update(path.name for path in picks)
    pool = tmp / "candidates"
    pool.mkdir()
    for path in picks: (pool / path.name).symlink_to(path.resolve())
    database, fasta = tmp / "candidates_db", tmp / "candidates.fasta"
    foldseek("createdb", str(pool), str(database))
    foldseek("convert2fasta", str(database), str(fasta))
    # a structure file's header is just its chain's name, so there is nothing to describe
    return [(name, "", sequence) for name, _, sequence in read_fasta(fasta)]

def code(name):
    '''a uniref cluster name -> the uniprot code it stands for'''
    return name.split("_", 1)[1] if name.startswith("UniRef90_") else name

def sample(reservoir, quota=QUOTA, tasks=ELEVEN, say=print):
    '''sample until there are `quota` sequences that match no benchmark protein'''
    classes = [REGISTRY[name] for name in tasks]
    by_seq, by_struct = reservoir in ("seq", "both"), reservoir in ("struct", "both")
    rng = random.Random(SEED)
    c.work.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(dir=c.work) as tmp:
        tmp = Path(tmp)
        say(f"benchmarks over {len(classes)} datasets ...")
        seq_target, of_entity = sequence_target(classes, tmp, say)
        struct_target = (structure_target(classes, of_entity, tmp, say) if by_struct
                         else None)
        kept, used, round_number = {}, set(), 0

        while len(kept) < quota:
            round_number += 1
            n = int((quota - len(kept)) * OVERSAMPLE)
            say(f"round {round_number}: drawing {n:,} candidates "
                f"(have {len(kept):,}/{quota:,})")
            # each round its own directory, so no search finds the last one's files
            scratch = tmp / f"round{round_number}"
            scratch.mkdir()
            pool = (structure_sample(n, used, rng, scratch) if by_struct
                    else sequence_sample(n, used, rng))
            if not pool:
                say("the reservoir is used up")
                break

            leakers = set()
            if by_seq: leakers |= sequence_leakers(pool, seq_target, scratch)
            if by_struct:
                leakers |= structure_leakers(scratch / "candidates_db", struct_target,
                                             scratch)
            say(f"  threw out {len(leakers):,} of {len(pool):,}")

            # the header reads ">A0ACG4HM60 LacI family ... Tax=Nocardia ...", so the code
            # names the sequence and the description says what it is
            for accession, note, sequence in pool:
                if accession not in leakers and len(kept) < quota:
                    kept[f"{code(accession)} {note}".rstrip()] = sequence

    return Unlabeled(kept)

def main(argv=None):
    '''one flag, naming the reservoir'''
    argv = list(sys.argv[1:] if argv is None else argv)
    asked = [name for name in RESERVOIRS if f"--{name}" in argv]
    if len(argv) != 1 or len(asked) != 1:
        print("\n".join(__doc__.strip().splitlines()[2:5]))
        return 2
    data = sample(asked[0])
    path = data.write_fasta(c.trainsets / f"trainset_{asked[0]}.fasta")
    print(f"\nwrote {path} -- {len(data):,} sequences that match no benchmark protein")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
