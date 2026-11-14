// Benchmark for the paper: conventional k-means and DBSCAN versus building an index
// (R*-tree and others) and clustering on top of it. Writes one CSV row per
// (data set size, structure, method); times are medians over the repetitions.
//
//   build_ms    index construction (0 for the index-free baselines)
//   cluster_ms  the clustering step alone
//   total_ms    build_ms + cluster_ms: the end-to-end cost when no index exists yet
//   ari_truth   adjusted Rand index against the generator's labels
//   ari_ref     adjusted Rand index against the conventional algorithm
//               (k-means baseline resp. exact DBSCAN)
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "cluster.hpp"
#include "index.hpp"

using namespace tc;
using Clock = std::chrono::steady_clock;

namespace {

std::vector<std::string> split(const std::string& s) {
    std::vector<std::string> out;
    std::istringstream in(s);
    std::string tok;
    while (std::getline(in, tok, ',')) if (!tok.empty()) out.push_back(tok);
    return out;
}

double median(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

struct Options {
    GenParams gen;
    std::vector<size_t> sizes{10000, 100000, 1000000};
    std::vector<Structure> structures{Structure::RStar, Structure::STR, Structure::KdTree};
    std::vector<int> levelsUp{0};  // cut: this many levels above the deepest level
    int capacity = 16;
    int k = 8;
    double eps = 0.01;
    int minPts = 10;
    int reps = 3;
    int maxIter = 100;
    uint64_t kmSeed = 1;
    size_t bruteLimit = 50000;
    std::string csv;
};

void usage() {
    std::puts(
        "usage: treeclust-bench [options]\n"
        "  --kind blobs|mixed|moons|rings|uniform   generator (default mixed)\n"
        "  --n 10000,100000,1000000                 data set sizes\n"
        "  --d 2            dimensionality          --gen-k 8        generated clusters\n"
        "  --spread 0.03    cluster std. deviation  --noise 0.05     noise fraction\n"
        "  --seed 42\n"
        "  --structures rstar,str,kdtree            (also: rquadratic, rlinear, quadtree)\n"
        "  --capacity 16    node capacity M\n"
        "  --levels-up 0    cut(s) for the node-based methods, in levels above the leaves\n"
        "  --k 8            k for k-means           --max-iter 100\n"
        "  --km-seed 1      seed of the k-means++ initialisation\n"
        "  --eps 0.01       --minpts 10             DBSCAN parameters\n"
        "  --reps 3         repetitions (median)\n"
        "  --brute-limit 50000   largest n for the index-free (quadratic) DBSCAN\n"
        "  --csv FILE       write the CSV to FILE instead of stdout");
}

}  // namespace

int main(int argc, char** argv) {
    Options o;
    o.gen.kind = "mixed";
    o.gen.k = 8;
    o.gen.noise = 0.05;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) { usage(); std::exit(1); }
            return argv[++i];
        };
        try {
            if (a == "--kind") o.gen.kind = next();
            else if (a == "--n") { o.sizes.clear(); for (auto& t : split(next())) o.sizes.push_back(std::stoull(t)); }
            else if (a == "--d") o.gen.d = std::stoi(next());
            else if (a == "--gen-k") o.gen.k = std::stoi(next());
            else if (a == "--spread") o.gen.spread = std::stod(next());
            else if (a == "--noise") o.gen.noise = std::stod(next());
            else if (a == "--seed") o.gen.seed = std::stoull(next());
            else if (a == "--structures") { o.structures.clear(); for (auto& t : split(next())) o.structures.push_back(parseStructure(t)); }
            else if (a == "--capacity") o.capacity = std::stoi(next());
            else if (a == "--levels-up") { o.levelsUp.clear(); for (auto& t : split(next())) o.levelsUp.push_back(std::stoi(t)); }
            else if (a == "--k") o.k = std::stoi(next());
            else if (a == "--km-seed") o.kmSeed = std::stoull(next());
            else if (a == "--max-iter") o.maxIter = std::stoi(next());
            else if (a == "--eps") o.eps = std::stod(next());
            else if (a == "--minpts") o.minPts = std::stoi(next());
            else if (a == "--reps") o.reps = std::max(1, std::stoi(next()));
            else if (a == "--brute-limit") o.bruteLimit = std::stoull(next());
            else if (a == "--csv") o.csv = next();
            else { usage(); return a == "--help" ? 0 : 1; }
        } catch (const std::exception& e) {
            std::fprintf(stderr, "bad argument for %s: %s\n", a.c_str(), e.what());
            return 1;
        }
    }

    FILE* out = o.csv.empty() ? stdout : std::fopen(o.csv.c_str(), "w");
    if (!out) { std::perror("csv"); return 1; }
    std::fprintf(out, "dataset,n,d,structure,capacity,method,depth,nodes_in_cut,build_ms,cluster_ms,"
                      "total_ms,clusters,noise,sse,ari_truth,ari_ref,iterations,range_queries\n");

    try {
        for (size_t n : o.sizes) {
            GenParams g = o.gen;
            g.n = n;
            const Dataset ds = generate(g);
            std::fprintf(stderr, "== %s n=%zu d=%d\n", ds.name.c_str(), ds.size(), ds.d);

            // Runs `f` reps times; returns the last result with the median time.
            auto timed = [&](auto&& f) {
                std::vector<double> ms;
                ClusterResult r;
                for (int rep = 0; rep < o.reps; ++rep) { r = f(); ms.push_back(r.ms); }
                r.ms = median(ms);
                return r;
            };
            auto emit = [&](const char* structure, const ClusterResult& r, int depth, size_t cutNodes,
                            double buildMs, const ClusterResult* ref) {
                std::string ariTruth = ds.hasTruth() ? std::to_string(adjustedRand(r.labels, ds.truth)) : "";
                std::string ariRef = ref ? std::to_string(adjustedRand(r.labels, ref->labels)) : "";
                auto rq = r.extra.find("range_queries");
                std::fprintf(out, "%s,%zu,%d,%s,%d,%s,%s,%s,%.3f,%.3f,%.3f,%d,%zu,%.6g,%s,%s,%d,%s\n",
                             ds.name.c_str(), ds.size(), ds.d, structure, *structure ? o.capacity : 0,
                             r.method.c_str(), depth >= 0 ? std::to_string(depth).c_str() : "",
                             cutNodes ? std::to_string(cutNodes).c_str() : "", buildMs, r.ms,
                             buildMs + r.ms, r.clusters, r.noise, r.sse, ariTruth.c_str(),
                             ariRef.c_str(), r.iterations,
                             rq == r.extra.end() ? "" : std::to_string(static_cast<size_t>(rq->second)).c_str());
                std::fflush(out);
                std::fprintf(stderr, "   %-10s %-20s build %10.1f ms  cluster %10.1f ms\n",
                             *structure ? structure : "-", r.method.c_str(), buildMs, r.ms);
            };

            // Conventional algorithms without any index.
            ClusterResult kmBase = timed([&] { return kmeansBaseline(ds, o.k, o.kmSeed, o.maxIter); });
            emit("", kmBase, -1, 0, 0.0, nullptr);
            ClusterResult dbRef;
            bool haveDbRef = false;
            if (n <= o.bruteLimit) {
                dbRef = timed([&] { return dbscanBaseline(ds, o.eps, o.minPts); });
                haveDbRef = true;
                emit("", dbRef, -1, 0, 0.0, nullptr);
            }

            for (Structure s : o.structures) {
                IndexParams ip;
                ip.structure = s;
                ip.capacity = o.capacity;
                std::vector<double> buildTimes;
                std::unique_ptr<SpatialIndex> idx;
                for (int rep = 0; rep < o.reps; ++rep) {
                    idx = std::make_unique<SpatialIndex>(ds, ip);
                    auto t0 = Clock::now();
                    idx->build();
                    buildTimes.push_back(std::chrono::duration<double, std::milli>(Clock::now() - t0).count());
                }
                const double buildMs = median(buildTimes);
                const char* name = structureName(s);

                // Exact DBSCAN with index-supported queries: the usual way to run DBSCAN.
                ClusterResult dbIdx = timed([&] { return dbscanIndex(*idx, o.eps, o.minPts, false); });
                if (!haveDbRef) { dbRef = dbIdx; haveDbRef = true; }
                emit(name, dbIdx, -1, 0, buildMs, &dbRef);
                emit(name, timed([&] { return dbscanIndex(*idx, o.eps, o.minPts, true); }), -1, 0, buildMs, &dbRef);

                for (int up : o.levelsUp) {
                    const int depth = std::max(0, idx->height() - up);
                    const size_t cutNodes = idx->cut(depth).size();
                    emit(name, timed([&] { return initialClusters(*idx, depth); }), depth, cutNodes, buildMs, nullptr);
                    if (cutNodes >= static_cast<size_t>(o.k)) {
                        emit(name, timed([&] { return kmeansTree(*idx, depth, o.k, o.kmSeed, o.maxIter, false); }),
                             depth, cutNodes, buildMs, &kmBase);
                        emit(name, timed([&] { return kmeansTree(*idx, depth, o.k, o.kmSeed, o.maxIter, true); }),
                             depth, cutNodes, buildMs, &kmBase);
                    }
                    emit(name, timed([&] { return dbscanNodes(*idx, depth, o.eps, o.minPts); }),
                         depth, cutNodes, buildMs, &dbRef);
                }
            }
        }
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
    if (out != stdout) std::fclose(out);
    return 0;
}
