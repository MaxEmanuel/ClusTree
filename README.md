# Data Structures Make Clustering Algorithms Obsolete

Demonstration system: the nodes of a spatial index (R*-tree and others) serve as the
initial clustering of a data set; k-means and DBSCAN then run on the tree instead of on
the raw points.

## Build and run

    cmake -S . -B build -G Ninja && cmake --build build
    ./build/treeclust-test                       # correctness tests
    ./build/treeclust-server --port 8080         # then open http://127.0.0.1:8080
    ./build/treeclust-bench --help               # benchmark for the paper (CSV output)

Requirements: C++17 compiler, CMake, nlohmann-json (fetched automatically if missing).
The server binds to 127.0.0.1 by default; use `--host 0.0.0.0` to expose it.

## Layout

| Path | Content |
|---|---|
| `src/index.{hpp,cpp}` | R*-tree, R-tree (quadratic / linear split), STR bulk loading, k-d tree, quadtree; range, eps-ball and kNN queries |
| `src/cluster.{hpp,cpp}` | tree level as clustering, k-means on tree nodes, DBSCAN variants, baselines, adjusted Rand index |
| `src/dataset.{hpp,cpp}` | synthetic generators, CSV import |
| `src/http.{hpp,cpp}`, `src/server.cpp` | dependency-free HTTP server and JSON/binary API |
| `web/` | browser front end (plain JavaScript, canvas) |
| `bench/bench.cpp` | conventional k-means / DBSCAN versus index construction + clustering on the index |
| `tests/test_main.cpp` | invariants, queries and DBSCAN checked against brute force |

## Clustering methods

| Method | Description |
|---|---|
| `index-partition` | every node at the chosen depth is a cluster; no algorithm runs |
| `kmeans-tree` | weighted k-means over the node centroids; points inherit their node's cluster |
| `kmeans-tree-seeded` | as above, then Lloyd iterations over the points from those centres |
| `kmeans-baseline` | k-means++ and Lloyd on the raw points |
| `dbscan-tree-dense` | exact DBSCAN; subtrees with diagonal <= eps and >= minPts points are core points of one cluster and issue no neighbourhood query |
| `dbscan-nodes` | approximate DBSCAN over the nodes, no point-to-point distances |
| `dbscan-index` | exact DBSCAN, one index-supported eps-query per point |
| `dbscan-baseline` | exact DBSCAN without index (quadratic) |

## HTTP API

`GET /api/state`, `POST /api/dataset/generate`, `POST /api/dataset/upload` (CSV body),
`POST /api/index`, `GET /api/points` (float32), `GET /api/tree?depth=`,
`GET /api/partition?depth=` (int32), `POST /api/cluster`, `GET /api/labels?id=` (int32),
`GET /api/ari?ref=`, `POST /api/query` (range / knn / radius), `POST /api/insert`,
`POST /api/runs/clear`. Request and response fields are documented in `src/server.cpp`.

## Benchmark

`treeclust-bench` writes one CSV row per data set size, structure and method with
`build_ms`, `cluster_ms`, `total_ms` (= build + clustering), cluster count, noise, SSE and
the adjusted Rand index against the ground truth and against the conventional algorithm.
Results of the runs used so far are in `results/`.

### Results (`results/mixed_2d.csv`, `results/mixed_4d.csv`)

Generator `mixed` (8 Gaussian blobs of varying density, 5 % uniform noise), node capacity
16, median of 3 repetitions, single-threaded. Commands:

    treeclust-bench --n 10000,100000,1000000 --eps 0.005 --minpts 10 --k 8 --reps 3 \
        --brute-limit 100000 --structures rstar,str,kdtree --levels-up 0,1 --csv results/mixed_2d.csv
    treeclust-bench --n 100000 --d 4 --eps 0.02 --minpts 10 --k 8 --reps 3 \
        --brute-limit 100000 --structures rstar,str,kdtree --csv results/mixed_4d.csv

1,000,000 points, 2-D, node-based methods at the leaf level. Times in ms; "vs. conv." is
the adjusted Rand index against the conventional algorithm.

| Method | Index | Build | Clustering | Total | vs. conv. |
|---|---|---:|---:|---:|---:|
| k-means, conventional | none | – | 957 | 957 | – |
| k-means on tree nodes | R*-tree | 3,947 | 100 | 4,047 | 0.68 |
| k-means on tree nodes | STR | 203 | 170 | 373 | 0.60 |
| k-means on tree nodes | k-d tree | 256 | 81 | 337 | 0.80 |
| DBSCAN, conventional with index | R*-tree | 3,947 | 63,033 | 66,980 | 1.00 |
| DBSCAN, conventional with index | k-d tree | 256 | 52,592 | 52,848 | 1.00 |
| DBSCAN, dense subtrees (exact) | R*-tree | 3,947 | 12,169 | 16,116 | 1.00 |
| DBSCAN, dense subtrees (exact) | STR | 203 | 1,441 | 1,644 | 1.00 |
| DBSCAN, dense subtrees (exact) | k-d tree | 256 | 276 | 532 | 1.00 |
| DBSCAN on tree nodes (approx.) | R*-tree | 3,947 | 1,139 | 5,086 | 0.05 |
| DBSCAN on tree nodes (approx.) | k-d tree | 256 | 758 | 1,014 | 1.00 |

Index-free DBSCAN (quadratic) needs 26.5 s for 100,000 points.

Observations:

* A tree level on its own is not a clustering of the kind k-means or DBSCAN produce: the
  leaf level has ARI ~ 0 against the ground truth (tens of thousands of tiny clusters).
  It is a compression of the data on which the algorithms run cheaply.
* k-means on the nodes costs 4-18 % of conventional k-means once an index exists. Counting
  construction, the insertion-built R*-tree is about 4x slower end to end than conventional
  k-means at 1M points; bulk-loaded STR and the k-d tree are about 2.5-3x faster.
  The result is a different local optimum (ARI 0.6-0.8 against the baseline run); the SSE
  is better or worse depending on the seed, so averages over seeds are needed.
* The dense-subtree DBSCAN returns the DBSCAN result (ARI 1 up to border-point ties) and is
  5x (R*-tree) to 190x (k-d tree) faster than index-supported DBSCAN in 2-D. In 4-D the
  gain shrinks to 1.04x (R*-tree) - 3x (k-d tree), because few subtrees fit into an eps-ball.
* DBSCAN on nodes alone is unreliable: ARI between 0.05 and 0.997 against exact DBSCAN,
  depending on structure, level and data size.

## Paper

`paper/paper.tex` (EDBT demonstration paper); build with `cd paper && latexmk -pdf paper.tex`.
The k-means seed study it cites is in `results/kmeans_seeds_1m.csv` (first column: seed).
