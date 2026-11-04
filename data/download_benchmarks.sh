#!/bin/sh
# download all eleven benchmarks and the sifts table two of them need. the structures that
# atlas does not ship are fetched from rcsb by plug itself, the first time they are read.
set -e
here="$(dirname "$0")"
for name in allobench atlas bindb cafa5 fireprot lp_pdbbind megascale passerrank ppi proteingym shs27k; do
  echo "== $name"
  sh "$here/download_$name.sh"
done
