// Clustering on top of a spatial index.
//
// The central idea: the nodes of the index at some depth already partition the data into
// spatially compact groups ("micro-clusters"). This partition is used
//   * directly, as an initial clustering (initialClusters),
//   * as a compressed input for k-means (kmeansTree), and
//   * to skip neighbourhood queries in DBSCAN (dbscanIndex with denseShortcut) or to run
//     DBSCAN on the nodes instead of the points (dbscanNodes).
// kmeansBaseline and dbscanIndex without the shortcut are the conventional baselines.
#pragma once

#include <cstdint>
#include <map>
#include <string>
#include <vector>

#include "index.hpp"

namespace tc {

struct ClusterResult {
    std::string method;
    std::vector<int32_t> labels;  // per point; -1 = noise
    int clusters = 0;             // number of non-empty clusters
    size_t noise = 0;             // number of noise points
    double sse = 0.0;             // sum of squared distances to the cluster means (no noise)
    double ms = 0.0;              // run time of the algorithm (without index construction)
    int iterations = 0;           // Lloyd iterations over the points (k-means)
    std::map<std::string, double> extra;
};

/// Every node of the cut at `depth` becomes one cluster.
ClusterResult initialClusters(const SpatialIndex& idx, int depth);

/// k-means++ seeding and Lloyd's algorithm on the raw points.
ClusterResult kmeansBaseline(const Dataset& ds, int k, uint64_t seed, int maxIter);

/// Weighted k-means over the centroids of the micro-clusters at `depth`. With refine, the
/// resulting centres seed Lloyd's algorithm on the raw points; without, each point simply
/// inherits the cluster of its micro-cluster.
ClusterResult kmeansTree(const SpatialIndex& idx, int depth, int k, uint64_t seed, int maxIter,
                         bool refine);

/// Exact DBSCAN (minPts counts the point itself) with index-supported eps-queries. With
/// denseShortcut, subtrees whose MBR diagonal is <= eps and that hold >= minPts points are
/// known to consist of core points of one cluster, so no query is issued for their points.
ClusterResult dbscanIndex(const SpatialIndex& idx, double eps, int minPts, bool denseShortcut);

/// Exact DBSCAN without any index: every eps-neighbourhood is found by a linear scan,
/// i.e. quadratic run time. The index-free reference point.
ClusterResult dbscanBaseline(const Dataset& ds, double eps, int minPts);

/// Approximate DBSCAN on the nodes of the cut at `depth`, without any distance computation
/// between points: a node is core if the estimated size of an eps-neighbourhood in its
/// region (uniformity assumption inside each MBR) is >= minPts; core nodes whose MBRs are
/// at most eps apart form one cluster.
ClusterResult dbscanNodes(const SpatialIndex& idx, int depth, double eps, int minPts);

/// Adjusted Rand index of two labelings (noise is treated as one cluster).
double adjustedRand(const std::vector<int32_t>& a, const std::vector<int32_t>& b);

}  // namespace tc
