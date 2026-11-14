#include "cluster.hpp"

#include <chrono>
#include <cmath>
#include <random>
#include <stdexcept>
#include <unordered_map>

namespace tc {

namespace {

using Clock = std::chrono::steady_clock;

double msSince(Clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
}

/// Fills clusters, noise and sse from the labels.
void finalize(const Dataset& ds, ClusterResult& r) {
    const int d = ds.d;
    const size_t n = ds.size();
    int32_t maxLabel = -1;
    for (int32_t l : r.labels) maxLabel = std::max(maxLabel, l);
    const size_t k = static_cast<size_t>(maxLabel + 1);
    std::vector<double> sum(k * d, 0.0);
    std::vector<size_t> cnt(k, 0);
    r.noise = 0;
    for (size_t i = 0; i < n; ++i) {
        int32_t l = r.labels[i];
        if (l < 0) { ++r.noise; continue; }
        ++cnt[l];
        const double* p = ds.pt(i);
        for (int j = 0; j < d; ++j) sum[l * d + j] += p[j];
    }
    r.clusters = 0;
    for (size_t c = 0; c < k; ++c) {
        if (!cnt[c]) continue;
        ++r.clusters;
        for (int j = 0; j < d; ++j) sum[c * d + j] /= static_cast<double>(cnt[c]);
    }
    r.sse = 0.0;
    for (size_t i = 0; i < n; ++i) {
        int32_t l = r.labels[i];
        if (l >= 0) r.sse += dist2(ds.pt(i), &sum[l * d], d);
    }
}

/// The micro-clusters induced by a cut through the index.
struct Micro {
    std::vector<const Node*> nodes;
    std::vector<uint32_t> ids;       // point ids, grouped by micro-cluster
    std::vector<size_t> offset;      // nodes.size() + 1 offsets into ids
    std::vector<double> centroid;    // nodes.size() * d
    std::vector<double> weight;      // point counts
    size_t size() const { return nodes.size(); }
};

Micro extractMicro(const SpatialIndex& idx, int depth) {
    const Dataset& ds = idx.data();
    const int d = ds.d;
    Micro m;
    m.nodes = idx.cut(depth);
    m.ids.reserve(idx.size());
    m.offset.push_back(0);
    m.centroid.assign(m.nodes.size() * d, 0.0);
    for (size_t c = 0; c < m.nodes.size(); ++c) {
        SpatialIndex::collect(m.nodes[c], m.ids);
        m.offset.push_back(m.ids.size());
        const size_t cnt = m.offset[c + 1] - m.offset[c];
        double* ctr = &m.centroid[c * d];
        for (size_t i = m.offset[c]; i < m.offset[c + 1]; ++i) {
            const double* p = ds.pt(m.ids[i]);
            for (int j = 0; j < d; ++j) ctr[j] += p[j];
        }
        for (int j = 0; j < d; ++j) ctr[j] /= static_cast<double>(cnt);
        m.weight.push_back(static_cast<double>(cnt));
    }
    return m;
}

/// k-means++ seeding; w may be null (unit weights).
std::vector<double> seedPlusPlus(const double* X, const double* w, size_t n, int d, int k,
                                 std::mt19937_64& rng) {
    std::vector<double> C(static_cast<size_t>(k) * d);
    std::vector<double> D(n, std::numeric_limits<double>::infinity());
    std::vector<double> cum(n);
    auto sample = [&](bool useDist) {
        double total = 0.0;
        for (size_t i = 0; i < n; ++i) {
            total += (w ? w[i] : 1.0) * (useDist ? D[i] : 1.0);
            cum[i] = total;
        }
        if (total <= 0.0) return std::uniform_int_distribution<size_t>(0, n - 1)(rng);
        double u = std::uniform_real_distribution<double>(0.0, total)(rng);
        size_t i = std::lower_bound(cum.begin(), cum.end(), u) - cum.begin();
        return std::min(i, n - 1);
    };
    for (int c = 0; c < k; ++c) {
        size_t pick = sample(c > 0);
        std::copy(X + pick * d, X + pick * d + d, &C[static_cast<size_t>(c) * d]);
        for (size_t i = 0; i < n; ++i)
            D[i] = std::min(D[i], dist2(X + i * d, &C[static_cast<size_t>(c) * d], d));
    }
    return C;
}

/// Lloyd's algorithm; w may be null. Returns the number of iterations.
int lloyd(const double* X, const double* w, size_t n, int d, std::vector<double>& C, int k,
          int maxIter, std::vector<int32_t>& assign) {
    assign.assign(n, -1);
    std::vector<double> sum(static_cast<size_t>(k) * d), wsum(k);
    int it = 0;
    while (it < maxIter) {
        ++it;
        size_t changed = 0;
        for (size_t i = 0; i < n; ++i) {
            const double* p = X + i * d;
            int32_t best = 0;
            double bd = std::numeric_limits<double>::infinity();
            for (int c = 0; c < k; ++c) {
                double dd = dist2(p, &C[static_cast<size_t>(c) * d], d);
                if (dd < bd) { bd = dd; best = c; }
            }
            if (assign[i] != best) { assign[i] = best; ++changed; }
        }
        if (!changed) break;
        std::fill(sum.begin(), sum.end(), 0.0);
        std::fill(wsum.begin(), wsum.end(), 0.0);
        for (size_t i = 0; i < n; ++i) {
            const double wi = w ? w[i] : 1.0;
            const double* p = X + i * d;
            double* s = &sum[static_cast<size_t>(assign[i]) * d];
            for (int j = 0; j < d; ++j) s[j] += wi * p[j];
            wsum[assign[i]] += wi;
        }
        for (int c = 0; c < k; ++c)
            if (wsum[c] > 0)  // an empty cluster keeps its centre
                for (int j = 0; j < d; ++j) C[static_cast<size_t>(c) * d + j] = sum[static_cast<size_t>(c) * d + j] / wsum[c];
    }
    return it;
}

struct UnionFind {
    std::vector<uint32_t> p;
    explicit UnionFind(size_t n) : p(n) {
        for (size_t i = 0; i < n; ++i) p[i] = static_cast<uint32_t>(i);
    }
    uint32_t find(uint32_t x) {
        while (p[x] != x) { p[x] = p[p[x]]; x = p[x]; }
        return x;
    }
    void unite(uint32_t a, uint32_t b) {
        a = find(a); b = find(b);
        if (a != b) p[std::max(a, b)] = std::min(a, b);
    }
};

void checkDbscanParams(double eps, int minPts) {
    if (!(eps > 0)) throw std::runtime_error("eps must be positive");
    if (minPts < 1) throw std::runtime_error("minPts must be >= 1");
}

}  // namespace

// ---------------------------------------------------------------------------------------

ClusterResult initialClusters(const SpatialIndex& idx, int depth) {
    auto t0 = Clock::now();
    ClusterResult r;
    r.method = "index-partition";
    r.labels.assign(idx.data().size(), -1);
    std::vector<const Node*> nodes = idx.cut(depth);
    std::vector<uint32_t> ids;
    for (size_t c = 0; c < nodes.size(); ++c) {
        ids.clear();
        SpatialIndex::collect(nodes[c], ids);
        for (uint32_t p : ids) r.labels[p] = static_cast<int32_t>(c);
    }
    r.ms = msSince(t0);
    finalize(idx.data(), r);
    return r;
}

ClusterResult kmeansBaseline(const Dataset& ds, int k, uint64_t seed, int maxIter) {
    const size_t n = ds.size();
    if (k < 1 || static_cast<size_t>(k) > n) throw std::runtime_error("k must be in [1, n]");
    auto t0 = Clock::now();
    ClusterResult r;
    r.method = "kmeans-baseline";
    std::mt19937_64 rng(seed);
    std::vector<double> C = seedPlusPlus(ds.x.data(), nullptr, n, ds.d, k, rng);
    r.iterations = lloyd(ds.x.data(), nullptr, n, ds.d, C, k, maxIter, r.labels);
    r.ms = msSince(t0);
    finalize(ds, r);
    return r;
}

ClusterResult kmeansTree(const SpatialIndex& idx, int depth, int k, uint64_t seed, int maxIter,
                         bool refine) {
    const Dataset& ds = idx.data();
    const int d = ds.d;
    if (k < 1) throw std::runtime_error("k must be >= 1");
    auto t0 = Clock::now();
    ClusterResult r;
    r.method = refine ? "kmeans-tree-seeded" : "kmeans-tree";
    Micro m = extractMicro(idx, depth);
    if (m.size() < static_cast<size_t>(k))
        throw std::runtime_error("the index has only " + std::to_string(m.size()) +
                                 " nodes at this depth, fewer than k=" + std::to_string(k) +
                                 "; choose a deeper level");
    std::mt19937_64 rng(seed);
    std::vector<double> C =
        seedPlusPlus(m.centroid.data(), m.weight.data(), m.size(), d, k, rng);
    std::vector<int32_t> microAssign;
    int microIters = lloyd(m.centroid.data(), m.weight.data(), m.size(), d, C, k, maxIter,
                           microAssign);
    r.extra["micro_clusters"] = static_cast<double>(m.size());
    r.extra["micro_iterations"] = microIters;
    r.extra["micro_ms"] = msSince(t0);
    if (refine) {
        r.iterations = lloyd(ds.x.data(), nullptr, ds.size(), d, C, k, maxIter, r.labels);
    } else {
        r.labels.assign(ds.size(), -1);
        for (size_t c = 0; c < m.size(); ++c)
            for (size_t i = m.offset[c]; i < m.offset[c + 1]; ++i)
                r.labels[m.ids[i]] = microAssign[c];
    }
    r.ms = msSince(t0);
    finalize(ds, r);
    return r;
}

// ---------------------------------------------------------------------------------------

namespace {

/// Shared DBSCAN driver. Without an index (idx == nullptr) every eps-neighbourhood is
/// computed by a scan over all points.
ClusterResult dbscanImpl(const Dataset& ds, const SpatialIndex* idx, double eps, int minPts,
                         bool denseShortcut) {
    checkDbscanParams(eps, minPts);
    const int d = ds.d;
    const size_t n = ds.size();
    const double eps2 = eps * eps;
    auto t0 = Clock::now();
    ClusterResult r;
    r.method = !idx ? "dbscan-baseline" : denseShortcut ? "dbscan-tree-dense" : "dbscan-index";

    enum : uint8_t { UNSEEN = 0, CORE = 1, NONCORE = 2 };
    constexpr uint32_t NONE = 0xffffffffu;
    std::vector<uint8_t> state(n, UNSEEN);
    std::vector<uint32_t> borderOf(n, NONE);
    UnionFind uf(n);

    // Phase 1 (shortcut only): maximal subtrees with diagonal <= eps and >= minPts points.
    // All their points are pairwise within eps, hence core points of one cluster.
    struct Dense { const Node* node; std::vector<uint32_t> pts; };
    std::vector<Dense> dense;
    std::unordered_map<const Node*, uint32_t> denseId;
    size_t densePoints = 0;
    if (denseShortcut && idx && n > 0) {
        std::vector<const Node*> stack{idx->root()};
        std::vector<uint32_t> pts;
        while (!stack.empty()) {
            const Node* nd = stack.back();
            stack.pop_back();
            if (nd->mbr.diag2(d) <= eps2) {
                pts.clear();
                SpatialIndex::collect(nd, pts);
                if (pts.size() >= static_cast<size_t>(minPts)) {
                    for (uint32_t p : pts) { state[p] = CORE; uf.unite(pts[0], p); }
                    densePoints += pts.size();
                    denseId[nd] = static_cast<uint32_t>(dense.size());
                    dense.push_back({nd, pts});
                }
                continue;
            }
            for (const Node* c : nd->children) stack.push_back(c);
        }
        // Phase 2: connect dense subtrees that contain a pair of points within eps.
        std::vector<uint32_t> ca, cb;
        for (uint32_t a = 0; a < dense.size(); ++a) {
            const Rect& ra = dense[a].node->mbr;
            stack.assign(1, idx->root());
            while (!stack.empty()) {
                const Node* nd = stack.back();
                stack.pop_back();
                if (nd->mbr.minDist2(ra, d) > eps2) continue;
                auto it = denseId.find(nd);
                if (it == denseId.end()) {
                    for (const Node* c : nd->children) stack.push_back(c);
                    continue;
                }
                const uint32_t b = it->second;
                if (b <= a || uf.find(dense[a].pts[0]) == uf.find(dense[b].pts[0])) continue;
                const Rect& rb = nd->mbr;
                ca.clear();
                cb.clear();
                for (uint32_t p : dense[a].pts)
                    if (rb.minDist2(ds.pt(p), d) <= eps2) ca.push_back(p);
                for (uint32_t p : dense[b].pts)
                    if (ra.minDist2(ds.pt(p), d) <= eps2) cb.push_back(p);
                bool linked = false;
                for (size_t i = 0; i < ca.size() && !linked; ++i)
                    for (size_t j = 0; j < cb.size(); ++j)
                        if (dist2(ds.pt(ca[i]), ds.pt(cb[j]), d) <= eps2) { linked = true; break; }
                if (linked) uf.unite(dense[a].pts[0], dense[b].pts[0]);
            }
        }
    }

    // Phase 3: one eps-query for every remaining point. Links to points that are processed
    // later are established when those are processed (the neighbourhood is symmetric).
    std::vector<uint32_t> nb;
    size_t queries = 0;
    for (size_t i = 0; i < n; ++i) {
        if (state[i] != UNSEEN) continue;
        const uint32_t p = static_cast<uint32_t>(i);
        nb.clear();
        if (idx) {
            idx->radius(ds.pt(p), eps, nb);
        } else {
            const double* pp = ds.pt(p);
            for (size_t q = 0; q < n; ++q)
                if (dist2(pp, ds.pt(q), d) <= eps2) nb.push_back(static_cast<uint32_t>(q));
        }
        ++queries;
        if (nb.size() >= static_cast<size_t>(minPts)) {
            state[p] = CORE;
            for (uint32_t q : nb) {
                if (state[q] == CORE) uf.unite(p, q);
                else if (state[q] == NONCORE && borderOf[q] == NONE) borderOf[q] = p;
            }
        } else {
            state[p] = NONCORE;
            for (uint32_t q : nb)
                if (state[q] == CORE) { borderOf[p] = q; break; }
        }
    }

    // Labels: connected components of core points; border points join a neighbouring core.
    r.labels.assign(n, -1);
    std::vector<int32_t> rootLabel(n, -1);
    int32_t next = 0;
    for (size_t i = 0; i < n; ++i) {
        if (state[i] != CORE) continue;
        uint32_t root = uf.find(static_cast<uint32_t>(i));
        if (rootLabel[root] < 0) rootLabel[root] = next++;
        r.labels[i] = rootLabel[root];
    }
    for (size_t i = 0; i < n; ++i)
        if (state[i] == NONCORE && borderOf[i] != NONE)
            r.labels[i] = rootLabel[uf.find(borderOf[i])];

    r.ms = msSince(t0);
    r.extra["range_queries"] = static_cast<double>(queries);
    if (denseShortcut) {
        r.extra["dense_nodes"] = static_cast<double>(dense.size());
        r.extra["dense_points"] = static_cast<double>(densePoints);
    }
    finalize(ds, r);
    return r;
}

}  // namespace

ClusterResult dbscanIndex(const SpatialIndex& idx, double eps, int minPts, bool denseShortcut) {
    return dbscanImpl(idx.data(), &idx, eps, minPts, denseShortcut);
}

ClusterResult dbscanBaseline(const Dataset& ds, double eps, int minPts) {
    return dbscanImpl(ds, nullptr, eps, minPts, false);
}

ClusterResult dbscanNodes(const SpatialIndex& idx, int depth, double eps, int minPts) {
    checkDbscanParams(eps, minPts);
    const Dataset& ds = idx.data();
    const int d = ds.d;
    const double eps2 = eps * eps;
    auto t0 = Clock::now();
    ClusterResult r;
    r.method = "dbscan-nodes";
    Micro m = extractMicro(idx, depth);
    const size_t k = m.size();
    std::unordered_map<const Node*, uint32_t> nodeId;
    for (size_t i = 0; i < k; ++i) nodeId[m.nodes[i]] = static_cast<uint32_t>(i);

    // Density estimate per node, assuming the points of a node are spread uniformly over
    // its MBR: the mass inside the node's MBR grown by eps, scaled to the volume of an
    // eps-ball, estimates the size of an eps-neighbourhood in that region. Neighbouring
    // nodes are found by a traversal that stops at the cut.
    const double ballVolume = std::pow(M_PI, 0.5 * d) / std::tgamma(0.5 * d + 1.0) * std::pow(eps, d);
    std::vector<std::vector<uint32_t>> nbs(k);
    std::vector<char> core(k, 0);
    std::vector<const Node*> stack;
    for (size_t i = 0; i < k; ++i) {
        const Rect& ri = m.nodes[i]->mbr;
        Rect grown = ri;
        double grownVolume = 1.0;
        for (int j = 0; j < d; ++j) {
            grown.lo[j] -= eps;
            grown.hi[j] += eps;
            grownVolume *= grown.hi[j] - grown.lo[j];
        }
        double mass = 0.0;
        stack.assign(1, idx.root());
        while (!stack.empty()) {
            const Node* nd = stack.back();
            stack.pop_back();
            if (!nd->mbr.intersects(grown, d)) continue;
            auto it = nodeId.find(nd);
            if (it == nodeId.end()) {
                for (const Node* c : nd->children) stack.push_back(c);
                continue;
            }
            double frac = 1.0;  // share of the neighbour's MBR that lies in the grown MBR
            for (int j = 0; j < d; ++j) {
                const double len = nd->mbr.hi[j] - nd->mbr.lo[j];
                if (len <= 0) continue;
                frac *= (std::min(nd->mbr.hi[j], grown.hi[j]) - std::max(nd->mbr.lo[j], grown.lo[j])) / len;
            }
            mass += frac * m.weight[it->second];
            if (nd->mbr.minDist2(ri, d) <= eps2) nbs[i].push_back(it->second);
        }
        core[i] = mass * ballVolume / grownVolume >= minPts;
    }
    // Core nodes whose MBRs are within eps of each other belong to one cluster.
    UnionFind uf(k);
    for (size_t i = 0; i < k; ++i)
        if (core[i])
            for (uint32_t j : nbs[i])
                if (core[j]) uf.unite(static_cast<uint32_t>(i), j);
    std::vector<int32_t> rootLabel(k, -1), nodeLabel(k, -1);
    int32_t next = 0;
    for (size_t i = 0; i < k; ++i) {
        if (!core[i]) continue;
        uint32_t root = uf.find(static_cast<uint32_t>(i));
        if (rootLabel[root] < 0) rootLabel[root] = next++;
        nodeLabel[i] = rootLabel[root];
    }
    // Points of core nodes inherit the node's cluster. Points of the other nodes are
    // border points if they lie within eps of a core node's MBR, and noise otherwise.
    r.labels.assign(ds.size(), -1);
    for (size_t c = 0; c < k; ++c) {
        for (size_t i = m.offset[c]; i < m.offset[c + 1]; ++i) {
            const uint32_t p = m.ids[i];
            if (core[c]) { r.labels[p] = nodeLabel[c]; continue; }
            for (uint32_t j : nbs[c]) {
                if (core[j] && m.nodes[j]->mbr.minDist2(ds.pt(p), d) <= eps2) {
                    r.labels[p] = nodeLabel[j];
                    break;
                }
            }
        }
    }
    r.ms = msSince(t0);
    r.extra["micro_clusters"] = static_cast<double>(k);
    finalize(ds, r);
    return r;
}

// ---------------------------------------------------------------------------------------

double adjustedRand(const std::vector<int32_t>& a, const std::vector<int32_t>& b) {
    if (a.size() != b.size()) throw std::runtime_error("labelings differ in length");
    const size_t n = a.size();
    if (n < 2) return 1.0;
    std::unordered_map<uint64_t, uint64_t> pairs;
    std::unordered_map<int32_t, uint64_t> ca, cb;
    for (size_t i = 0; i < n; ++i) {
        uint64_t key = (static_cast<uint64_t>(static_cast<uint32_t>(a[i])) << 32) |
                       static_cast<uint32_t>(b[i]);
        ++pairs[key];
        ++ca[a[i]];
        ++cb[b[i]];
    }
    auto comb2 = [](uint64_t x) { return 0.5 * static_cast<double>(x) * static_cast<double>(x - 1); };
    double sumPairs = 0, sumA = 0, sumB = 0;
    for (auto& kv : pairs) sumPairs += comb2(kv.second);
    for (auto& kv : ca) sumA += comb2(kv.second);
    for (auto& kv : cb) sumB += comb2(kv.second);
    const double expected = sumA * sumB / comb2(n);
    const double maxIndex = 0.5 * (sumA + sumB);
    if (maxIndex == expected) return 1.0;
    return (sumPairs - expected) / (maxIndex - expected);
}

}  // namespace tc
