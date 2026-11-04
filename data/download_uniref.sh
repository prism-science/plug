#!/bin/sh
# download uniref90 into data/uniref — the sequence reservoir that
# plug.sample_unlabeled_trainset samples with --seq. ~40 GB, and uniprot replaces it with each
# release, so a sample drawn from a later release is a different sample.
set -e
d="$(dirname "$0")/uniref"
mkdir -p "$d"
curl -fL https://ftp.uniprot.org/pub/databases/uniprot/uniref/uniref90/uniref90.fasta.gz -o "$d/uniref90.fasta.gz"
