[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24120499.svg)](https://doi.org/10.5281/zenodo.24120499)
# PLUG 

**Protein Leakage-free evaluation for Unbiased Generators of function.**

## Motivation

Data leakage is ubiquitous in protein function prediction. A model tested on homologs of what it trained on can score well by recalling a label rather than generalising, and each benchmark defines its own split, so scores cannot be compared across functions. PLUG applies one definition of leakage to eleven functional datasets at once: two proteins are too alike to separate at mmseqs identity > 0.3 over 80% coverage or foldseek TM > 0.7, and two ligands at ecfp4 tanimoto > 0.5. It offers two kinds of split and a screen for pretraining data:

- **lohi** solves [Lo-Hi](https://github.com/SteshinSS/lohi_neurips2023)'s balanced vertex minimum k-cut over the pooled graph. Nothing similar crosses a split in any dataset, so it supports transfer between benchmarks, and it pays for that in dropped data.
- **datasail** runs [DataSAIL](https://github.com/kalininalab/DataSAIL) (leakage minimization) once per benchmark. It keeps more data and minimises sequence leakage inside a benchmark rather than forbidding it, so it suits one benchmark at a time and shouldn't be used for transfer.
- **sample_unlabeled_trainset** holds a training set from a reservoir clear of every benchmark.

## Usage

Needs [uv](https://docs.astral.sh/uv/), [mmseqs2](https://github.com/soedinglab/MMseqs2) and [foldseek](https://github.com/steineggerlab/foldseek) on your `PATH`. Every setting is a constant in `plug/config.py` or at the top of the script that uses it.

```bash
uv sync                         # the pinned solver is part of what makes a split reproducible

# download the data
sh data/download_benchmarks.sh  # the eleven datasets
sh data/download_uniref.sh      # the sequence reservoir, only for --seq
sh data/download_pdb.sh         # the structure reservoir, only for --struct and --both

# generate splits
uv run python3 -m plug.split --lohi       # pooled, nothing crosses, drops data
uv run python3 -m plug.split --datasail   # per benchmark, leakage minimised

# hold out training data from all datasets
uv run python3 -m plug.sample_unlabeled_trainset --seq      # uniref90, screened on sequence
uv run python3 -m plug.sample_unlabeled_trainset --struct   # the pdb, screened on structure
uv run python3 -m plug.sample_unlabeled_trainset --both     # the pdb, screened on either
```

A split is written to `results/<tag>/<strategy>/`, and `splits/<dataset>.csv.gz` has one line per datapoint in the order of that dataset's `rows()`, so labelling a copy is a zip. Check the two lengths match first: a dataset whose rows have changed would otherwise be mislabelled without any error. A training set is written to `out/trainset_<reservoir>.fasta`.

The downloads fetch whatever each source currently publishes, so a split you build covers the data as it stands today. The splits in `results/` are the paper's, built on the data of that time; their labels only line up with a download whose rows have not changed, and BindingDB's already have.

The first build runs mmseqs over the pool (an hour or more) and foldseek over 22k chains (a day), then caches both in `work/`. The sequence and ligand caches are named for the thresholds and the dataset list, not their contents, so delete `work/` after downloading newer data.

## Adding a dataset

A dataset is one file in `plug/benchmarks/` that says how to read its rows:

```python
# plug/benchmarks/mydata.py
import csv
from . import Benchmark
from .. import config as c

class Mydata(Benchmark):
    name = "mydata"
    key = "uniprot"           # the field naming the protein; a tuple of two for a pair
    seq = "sequence"          # the field holding its sequence
    views = ()                # add "structure" and/or "ligand" if it can leak through them
    label = "fitness"         # what it is scored on

    @classmethod
    def rows(cls):            # one dict per datapoint, in a fixed order
        with open(c.data / "mydata" / "mydata.csv", newline="") as f:
            yield from csv.DictReader(f)
```

Import it in `plug/benchmarks/__init__.py` and add its name to `DATASETS`, the tuple of datasets every split and screen runs over. With the `ligand` view, set `ligand` to the field holding each row's smiles (default `"smiles"`). With `structure`, give each row a `pdb` field and return `rcsb.structures(cls)` from `structures()`, as `allobench.py` does; every protein needs a structure or the build stops. Add a `data/download_mydata.sh` so others can fetch it.

Then rerun whichever you need, exactly as above:

```bash
uv run python3 -m plug.split --lohi                     # or --datasail
uv run python3 -m plug.sample_unlabeled_trainset --seq  # or --struct, --both
```

Adding a dataset changes the dataset list, so the build gets a new tag in `results/` and the searches run again over the larger pool. lohi is solved over that pool, so every dataset's lohi split can change, not only the new one's; datasail solves each dataset on its own.

## Main results

- **Leakage inflates scores.** Against random controls dealt to the same sizes and matched on zero-shot difficulty, lohi lowers frozen ESM3 probe performance by 25.9% (linear) and 26.3% (MLP) on average over the eleven datasets, with AUROC measured above chance. The drop is largest for ligand binding (LP-PDBBind 82%, BindingDB 66%, linear) and smallest for residue-level readouts such as ATLAS, where the linear probe does not drop at all.
- **Overlap with pretraining data is large.** Of one million UniRef90 sequences, 38.5% have a homolog in a PLUG dataset at 20% identity, 21.5% at 30% and 8.0% at 50%.
- **Datasets overlap each other.** A protein appearing in more than one dataset (2,051 of them) carries one split label under lohi; 1,097 carry two under datasail, which is why only lohi may be trained across benchmarks (transfer learning)

## Licenses

PLUG does not redistribute data. Each source keeps its own terms, and two of them restrict reuse: ATLAS is CC BY-NC 4.0, and LP-PDBBind is licensed for educational, research and not-for-profit use only, with affinities that come from PDBbind, whose license forbids redistribution. This repo ships code, download scripts that fetch each dataset from its source, and split labels, and every user accepts each source's terms directly. Terms, full texts and citations for every dataset are in [`licenses/README.md`](licenses/README.md).

PLUG's own code is MIT ([`licenses/PLUG-MIT-LICENSE`](licenses/PLUG-MIT-LICENSE)). `plug/external/` vendors Lo-Hi and DataSAIL, both MIT, with every edit recorded in [`plug/external/SOURCE`](plug/external/SOURCE).



