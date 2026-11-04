#!/bin/sh
# mirror the whole pdb (mmcif) into data/pdb — the structure reservoir that
# plug.sample_unlabeled_trainset samples with --struct and --both. ~250k gzipped structures,
# so this is a big pull; needs rsync.
set -e
d="$(dirname "$0")/pdb"
mkdir -p "$d"
rsync -rlpt -z --delete --port=33444 \
  rsync.rcsb.org::ftp_data/structures/divided/mmCIF/ "$d/"
