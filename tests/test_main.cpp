// Correctness tests: structural invariants, queries against brute force, DBSCAN against a
// quadratic reference implementation, and sanity checks for the k-means variants.
#include <algorithm>
#include <cstdio>
#include <random>
#include <string>

#include "cluster.hpp"
#include "index.hpp"

using namespace tc;

static int failures = 0;
static int checks = 0;

#define CHECK(cond, ...)                                  \
    do {                                                  \
        ++checks;                                         \
        if (!(cond)) {                                    \
            ++failures;                                   \
            std::printf("FAIL %s:%d: ", __FILE__, __LINE__); \
            std::printf(__VA_ARGS__);                     \
            std::printf("\n");                            \
        }                                                 \
    } while (0)

static const Structure kAll[] = {Structure::RStar, Structure::RQuadratic, Structure::RLinear,
                                 Structure::STR, Structure::KdTree, Structure::QuadTree};

static void testQueries(const Dataset& ds, Structure s, int cap, uint64_t seed) {
    IndexParams ip;
    ip.structure = s;
    ip.capacity = cap;
    SpatialIndex idx(ds, ip);
    idx.build();
    std::string err, tag = std::string(structureName(s)) + "/" + ds.name + "/d" +
                           std::to_string(ds.d) + "/M" + std::to_string(cap);
    CHECK(idx.validate(err), "%s: %s", tag.c_str(), err.c_str());
    CHECK(idx.size() == ds.size(), "%s: size", tag.c_str());

    // The cut at every depth is a partition of the data.
    for (int depth = 0; depth <= idx.height(); ++depth) {
        ClusterResult r = initialClusters(idx, depth);
        CHECK(r.noise == 0, "%s: cut at depth %d misses points", tag.c_str(), depth);
        CHECK(static_cast<size_t>(r.clusters) == idx.cut(depth).size(), "%s: cut size", tag.c_str());
    }
    CHECK(idx.cut(idx.height()).size() == idx.leafCount(), "%s: deepest cut = leaves", tag.c_str());

    std::mt19937_64 rng(seed);
    Rect b = ds.bounds();
    const int d = ds.d;
    const size_t n = ds.size();
    for (int q = 0; q < 30; ++q) {
        double c[MAX_DIM];
        Rect box = Rect::empty(d);
        for (int j = 0; j < d; ++j) {
            double w = b.hi[j] - b.lo[j];
            c[j] = b.lo[j] + w * std::uniform_real_distribution<double>(-0.1, 1.1)(rng);
            double h = w * std::uniform_real_distribution<double>(0.01, 0.3)(rng);
            box.lo[j] = c[j] - h;
            box.hi[j] = c[j] + h;
        }
        std::vector<uint32_t> got, want;
        idx.range(box, got);
        for (size_t i = 0; i < n; ++i)
            if (box.contains(ds.pt(i), d)) want.push_back(static_cast<uint32_t>(i));
        std::sort(got.begin(), got.end());
        CHECK(got == want, "%s: range query %d (%zu vs %zu)", tag.c_str(), q, got.size(), want.size());

        double eps = 0.05 * (b.hi[0] - b.lo[0]);
        got.clear();
        want.clear();
        idx.radius(c, eps, got);
        for (size_t i = 0; i < n; ++i)
            if (dist2(c, ds.pt(i), d) <= eps * eps) want.push_back(static_cast<uint32_t>(i));
        std::sort(got.begin(), got.end());
        CHECK(got == want, "%s: radius query %d", tag.c_str(), q);

        const size_t k = 1 + q;
        auto nn = idx.knn(c, k);
        std::vector<double> dists(n);
        for (size_t i = 0; i < n; ++i) dists[i] = dist2(c, ds.pt(i), d);
        std::sort(dists.begin(), dists.end());
        CHECK(nn.size() == std::min(k, n), "%s: knn size", tag.c_str());
        bool same = true;
        for (size_t i = 0; i < nn.size(); ++i) same = same && nn[i].first == dists[i];
        CHECK(same, "%s: knn query %d", tag.c_str(), q);
    }
}

static void testIncremental(Structure s, int d) {
    Dataset ds;
    ds.d = d;
    ds.name = "incremental";
    IndexParams ip;
    ip.structure = s;
    ip.capacity = 8;
    SpatialIndex idx(ds, ip);
    idx.build();
    std::mt19937_64 rng(7);
    std::normal_distribution<double> g(0.0, 1.0);
    std::string err;
    for (int i = 0; i < 3000; ++i) {
        double p[MAX_DIM];
        // drifting clusters and exact duplicates stress splits and region growth
        for (int j = 0; j < d; ++j) p[j] = (i % 7) * 0.5 + g(rng) * (0.1 + i * 1e-4);
        if (i % 50 < 12) for (int j = 0; j < d; ++j) p[j] = 3.25;
        idx.insert(ds.add(p));
        if (i % 500 == 499 || i < 40)
            CHECK(idx.validate(err), "%s incremental d=%d after %d: %s", structureName(s), d, i + 1, err.c_str());
    }
    std::vector<uint32_t> all;
    idx.range(ds.bounds(), all);
    CHECK(all.size() == ds.size(), "%s incremental: full range query", structureName(s));
}

/// Textbook DBSCAN with quadratic neighbourhood computation.
static void bruteDbscan(const Dataset& ds, double eps, int minPts, std::vector<int32_t>& labels,
                        std::vector<char>& core) {
    const size_t n = ds.size();
    std::vector<std::vector<uint32_t>> nb(n);
    for (size_t i = 0; i < n; ++i)
        for (size_t j = 0; j < n; ++j)
            if (dist2(ds.pt(i), ds.pt(j), ds.d) <= eps * eps) nb[i].push_back(static_cast<uint32_t>(j));
    core.assign(n, 0);
    for (size_t i = 0; i < n; ++i) core[i] = nb[i].size() >= static_cast<size_t>(minPts);
    labels.assign(n, -1);
    int32_t next = 0;
    for (size_t i = 0; i < n; ++i) {
        if (!core[i] || labels[i] >= 0) continue;
        std::vector<uint32_t> stack{static_cast<uint32_t>(i)};
        labels[i] = next;
        while (!stack.empty()) {
            uint32_t p = stack.back();
            stack.pop_back();
            for (uint32_t q : nb[p]) {
                if (labels[q] >= 0) continue;
                labels[q] = next;
                if (core[q]) stack.push_back(q);
            }
        }
        ++next;
    }
}

static void testDbscan(const Dataset& ds, Structure s, double eps, int minPts) {
    IndexParams ip;
    ip.structure = s;
    ip.capacity = 8;
    SpatialIndex idx(ds, ip);
    idx.build();
    std::vector<int32_t> ref;
    std::vector<char> core;
    bruteDbscan(ds, eps, minPts, ref, core);
    int32_t refClusters = 0;
    for (int32_t l : ref) refClusters = std::max(refClusters, l + 1);

    for (int variant = 0; variant < 3; ++variant) {
        const bool shortcut = variant == 1;
        if (variant == 2 && s != Structure::RStar) continue;  // index-free: test once
        ClusterResult r = variant == 2 ? dbscanBaseline(ds, eps, minPts)
                                       : dbscanIndex(idx, eps, minPts, shortcut);
        std::string tag = std::string(structureName(s)) + "/" + ds.name +
                          (variant == 2 ? "/baseline" : shortcut ? "/dense" : "/plain");
        // Core points and noise are uniquely determined; border points may legitimately
        // be assigned to any adjacent cluster.
        std::vector<int32_t> a, b;
        bool noiseSame = true, borderOk = true;
        for (size_t i = 0; i < ds.size(); ++i) {
            if ((ref[i] < 0) != (r.labels[i] < 0)) noiseSame = false;
            if (core[i]) { a.push_back(ref[i]); b.push_back(r.labels[i]); }
        }
        CHECK(noiseSame, "%s: noise set differs", tag.c_str());
        CHECK(a.empty() || adjustedRand(a, b) == 1.0, "%s: core partition differs", tag.c_str());
        CHECK(r.clusters == refClusters, "%s: %d clusters, expected %d", tag.c_str(), r.clusters, refClusters);
        // every border point must be within eps of a core point with the same label
        for (size_t i = 0; i < ds.size() && borderOk; ++i) {
            if (core[i] || r.labels[i] < 0) continue;
            bool found = false;
            for (size_t j = 0; j < ds.size() && !found; ++j)
                found = core[j] && r.labels[j] == r.labels[i] &&
                        dist2(ds.pt(i), ds.pt(j), ds.d) <= eps * eps;
            borderOk = found;
        }
        CHECK(borderOk, "%s: border point attached to a non-adjacent cluster", tag.c_str());
        if (shortcut)
            CHECK(r.extra.at("range_queries") + r.extra.at("dense_points") == static_cast<double>(ds.size()),
                  "%s: query accounting", tag.c_str());
    }
}

static void testKmeans() {
    GenParams g;
    g.kind = "blobs";
    g.n = 20000;
    g.k = 6;
    g.spread = 0.02;
    g.seed = 3;
    Dataset ds = generate(g);
    ClusterResult base = kmeansBaseline(ds, 6, 1, 100);
    CHECK(base.clusters == 6, "baseline cluster count");
    for (Structure s : kAll) {
        IndexParams ip;
        ip.structure = s;
        SpatialIndex idx(ds, ip);
        idx.build();
        const int depth = idx.height();
        ClusterResult micro = kmeansTree(idx, depth, 6, 1, 100, false);
        ClusterResult seeded = kmeansTree(idx, depth, 6, 1, 100, true);
        CHECK(micro.noise == 0 && seeded.noise == 0, "%s: unlabelled points", structureName(s));
        // Lloyd on the points can only improve on the centres found on the micro-clusters.
        CHECK(seeded.sse <= micro.sse * (1 + 1e-9), "%s: refinement increased SSE", structureName(s));
        // On well separated blobs every variant should essentially find the same optimum
        // unless seeding is unlucky; allow generous slack.
        CHECK(micro.sse < 3.0 * base.sse, "%s: micro SSE %g vs baseline %g", structureName(s), micro.sse, base.sse);
        bool threw = false;
        try { kmeansTree(idx, 0, 6, 1, 100, false); } catch (const std::exception&) { threw = true; }
        CHECK(threw, "%s: k larger than number of nodes must be rejected", structureName(s));
    }
    std::vector<int32_t> a{0, 0, 1, 1, 2, 2}, b{5, 5, 3, 3, 9, 9}, c{0, 1, 0, 1, 0, 1};
    CHECK(adjustedRand(a, b) == 1.0, "ARI of identical partitions");
    CHECK(adjustedRand(a, c) < 0.01, "ARI of unrelated partitions");
}

int main() {
    for (const char* kind : {"blobs", "uniform", "mixed", "rings"}) {
        for (int d : {2, 3, 5}) {
            GenParams g;
            g.kind = kind;
            g.n = 4000;
            g.d = d;
            g.k = 4;
            g.noise = std::string(kind) == "mixed" ? 0.1 : 0.0;
            g.seed = 11 + d;
            Dataset ds = generate(g);
            for (Structure s : kAll)
                for (int cap : {4, 16, 50}) testQueries(ds, s, cap, 99);
        }
    }
    {   // tiny and degenerate inputs
        for (size_t n : {size_t(1), size_t(3), size_t(17)}) {
            GenParams g;
            g.kind = "uniform";
            g.n = n;
            Dataset ds = generate(g);
            for (Structure s : kAll) testQueries(ds, s, 4, 5);
        }
        Dataset dup;
        dup.d = 2;
        dup.name = "duplicates";
        double p[2] = {1.0, 2.0}, q[2] = {1.0, 3.0};
        for (int i = 0; i < 300; ++i) dup.add(i % 3 ? p : q);
        for (Structure s : kAll) testQueries(dup, s, 4, 5);
    }
    for (Structure s : kAll)
        for (int d : {2, 4}) testIncremental(s, d);

    for (const char* kind : {"blobs", "moons", "mixed"}) {
        GenParams g;
        g.kind = kind;
        g.n = 2500;
        g.k = 4;
        g.noise = 0.08;
        g.seed = 21;
        Dataset ds = generate(g);
        for (Structure s : kAll) {
            testDbscan(ds, s, 0.02, 5);
            testDbscan(ds, s, 0.05, 10);
            testDbscan(ds, s, 0.15, 4);
        }
    }
    testKmeans();

    std::printf("%d checks, %d failures\n", checks, failures);
    return failures ? 1 : 0;
}
