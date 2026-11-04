#!/bin/sh
# download the newest BindingDB release (curated protein-ligand binding affinities, articles
# subset) into data/bindingdb. the tsv's target sequence column holds each protein; affinity
# columns hold Ki/Kd/IC50/EC50. releases are named by month and only the latest stay online.
set -e
d="$(dirname "$0")/bindingdb"
mkdir -p "$d"
base="https://www.bindingdb.org/rwd/bind/downloads/BindingDB_BindingDB_Articles"
found=""
for release in $(python3 -c '
import datetime
month = datetime.date.today().replace(day=1)
for _ in range(4):
    print(month.strftime("%Y%m"))
    month = (month - datetime.timedelta(days=1)).replace(day=1)'); do
  if curl -fsSL "${base}_${release}_tsv.zip" -o "$d/bindingdb.zip" 2>/dev/null; then found=$release; break; fi
done
[ -n "$found" ] || { echo "no BindingDB release could be downloaded" >&2; exit 1; }
echo "BindingDB release $found"
cd "$d" && unzip -q -o bindingdb.zip && rm bindingdb.zip
