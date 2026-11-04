# the splits

One directory per build, named for the thresholds and the datasets that made it. This one
is `id30_cov80_ge0_tm70_11tasks_3500e7c0`:

```
<strategy>/provenance.json          thresholds, targets, graph, solve, row counts, cost, digests
<strategy>/splits/<dataset>.csv.gz  row_index, split, dropped_because   <- a line per datapoint
<strategy>/nodes/proteins.csv.gz    node, dataset, split, length, component, entities
<strategy>/nodes/molecules.csv.gz   node, dataset, split, component
```

`<strategy>` is `datasail` or `lohi`. These are committed, not gitignored: the
whole set is 19 MB for 3.1 million datapoints under two strategies, small enough that
anyone training on it can clone rather than rebuild.

These are the paper's splits, built on the data of that time, and plug-analysis is their
archive. `python3 -m plug.split` writes over them in place from whatever `data/` now holds, and
the downloads fetch each source's current data, so a rebuild on fresh downloads is a new split.

The splits balance the seven single-protein benchmarks on
proteins with row floors, and under that rule lohi and datasail are in band on all eleven.
