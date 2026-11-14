// HTTP back end of the demonstration: holds one data set, one index and the clustering
// runs computed so far, and exposes them through a small JSON/binary API (see README.md).
#include <chrono>
#include <cstdio>
#include <cstring>
#include <memory>
#include <mutex>

#include <nlohmann/json.hpp>

#include "cluster.hpp"
#include "http.hpp"
#include "index.hpp"

using json = nlohmann::json;
using namespace tc;

namespace {

#ifndef TREECLUST_WEB_DIR
#define TREECLUST_WEB_DIR "web"
#endif

using Clock = std::chrono::steady_clock;

struct Run {
    int id;
    json params;
    ClusterResult result;
    json summary;
};

struct State {
    std::mutex mu;
    Dataset ds;
    IndexParams indexParams;
    std::unique_ptr<SpatialIndex> idx;
    double buildMs = 0;
    std::vector<size_t> cutSizes;  // number of nodes in the cut at each depth
    std::vector<Run> runs;
    int nextRunId = 1;
} G;

template <class T>
std::string toBytes(const std::vector<T>& v) {
    return std::string(reinterpret_cast<const char*>(v.data()), v.size() * sizeof(T));
}

void refreshCutSizes() {
    G.cutSizes.clear();
    for (int depth = 0, h = G.idx->height(); depth <= h; ++depth)
        G.cutSizes.push_back(G.idx->cut(depth).size());
}

void rebuildIndex() {
    G.runs.clear();
    G.idx = std::make_unique<SpatialIndex>(G.ds, G.indexParams);
    auto t0 = Clock::now();
    G.idx->build();
    G.buildMs = std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
    refreshCutSizes();
}

json stateJson() {
    json j;
    Rect b = G.ds.bounds();
    json lo = json::array(), hi = json::array();
    for (int i = 0; i < G.ds.d; ++i) {
        lo.push_back(b.lo[i]);
        hi.push_back(b.hi[i]);
    }
    j["dataset"] = {{"name", G.ds.name}, {"n", G.ds.size()}, {"d", G.ds.d},
                    {"hasTruth", G.ds.hasTruth()}, {"lo", lo}, {"hi", hi}};
    j["index"] = {{"structure", structureName(G.indexParams.structure)},
                  {"capacity", G.indexParams.capacity},
                  {"minFill", G.indexParams.minFill},
                  {"reinsertFrac", G.indexParams.reinsertFrac},
                  {"height", G.idx->height()},
                  {"nodes", G.idx->nodeCount()},
                  {"leaves", G.idx->leafCount()},
                  {"buildMs", G.buildMs},
                  {"cutSizes", G.cutSizes}};
    json runs = json::array();
    for (const Run& r : G.runs) runs.push_back(r.summary);
    j["runs"] = runs;
    return j;
}

json body(const HttpRequest& req) {
    if (req.body.empty()) return json::object();
    json j = json::parse(req.body, nullptr, false);
    if (j.is_discarded() || !j.is_object()) throw std::runtime_error("request body is not a JSON object");
    return j;
}

/// The two displayed dimensions, validated.
std::pair<int, int> dims(int dx, int dy) {
    if (dx < 0 || dy < 0 || dx >= G.ds.d || dy >= G.ds.d)
        throw std::runtime_error("projection dimension out of range");
    return {dx, dy};
}

json nodeBox(const Node* n, int dx, int dy) {
    return json::array({n->mbr.lo[dx], n->mbr.lo[dy], n->mbr.hi[dx], n->mbr.hi[dy]});
}

json statsJson(const QueryStats& st, int dx, int dy) {
    json visited = json::array();
    for (const Node* n : st.visited) {
        json b = nodeBox(n, dx, dy);
        b.push_back(n->leaf ? 1 : 0);
        visited.push_back(b);
    }
    return {{"nodesVisited", st.nodesVisited}, {"leavesVisited", st.leavesVisited},
            {"pointsChecked", st.pointsChecked}, {"visited", visited}};
}

// --- handlers --------------------------------------------------------------------------

void hState(const HttpRequest&, HttpResponse& res) {
    std::lock_guard<std::mutex> lock(G.mu);
    res.body = stateJson().dump();
}

void hGenerate(const HttpRequest& req, HttpResponse& res) {
    json j = body(req);
    GenParams g;
    g.kind = j.value("kind", g.kind);
    g.n = j.value("n", g.n);
    g.d = j.value("d", g.d);
    g.k = j.value("k", g.k);
    g.spread = j.value("spread", g.spread);
    g.noise = j.value("noise", g.noise);
    g.seed = j.value("seed", g.seed);
    Dataset ds = generate(g);
    std::lock_guard<std::mutex> lock(G.mu);
    G.idx.reset();
    G.ds = std::move(ds);
    rebuildIndex();
    res.body = stateJson().dump();
}

void hUpload(const HttpRequest& req, HttpResponse& res) {
    Dataset ds = parseCsv(req.body, req.param("label") == "1", req.param("name", "upload"));
    if (ds.size() > 5'000'000) throw std::runtime_error("at most 5,000,000 points are supported");
    std::lock_guard<std::mutex> lock(G.mu);
    G.idx.reset();
    G.ds = std::move(ds);
    rebuildIndex();
    res.body = stateJson().dump();
}

void hIndex(const HttpRequest& req, HttpResponse& res) {
    json j = body(req);
    std::lock_guard<std::mutex> lock(G.mu);
    IndexParams p = G.indexParams;
    if (j.contains("structure")) p.structure = parseStructure(j["structure"].get<std::string>());
    p.capacity = j.value("capacity", p.capacity);
    p.minFill = j.value("minFill", p.minFill);
    p.reinsertFrac = j.value("reinsertFrac", p.reinsertFrac);
    if (p.capacity < 4 || p.capacity > 1024) throw std::runtime_error("capacity must be in [4, 1024]");
    if (p.minFill < 0.1 || p.minFill > 0.5) throw std::runtime_error("minFill must be in [0.1, 0.5]");
    if (p.reinsertFrac < 0 || p.reinsertFrac > 0.5) throw std::runtime_error("reinsertFrac must be in [0, 0.5]");
    G.indexParams = p;
    rebuildIndex();
    res.body = stateJson().dump();
}

void hPoints(const HttpRequest& req, HttpResponse& res) {
    std::lock_guard<std::mutex> lock(G.mu);
    auto [dx, dy] = dims(std::atoi(req.param("dx", "0").c_str()), std::atoi(req.param("dy", "1").c_str()));
    const size_t n = G.ds.size();
    std::vector<float> xy(2 * n);
    for (size_t i = 0; i < n; ++i) {
        xy[2 * i] = static_cast<float>(G.ds.pt(i)[dx]);
        xy[2 * i + 1] = static_cast<float>(G.ds.pt(i)[dy]);
    }
    res.contentType = "application/octet-stream";
    res.body = toBytes(xy);
}

void hTruth(const HttpRequest&, HttpResponse& res) {
    std::lock_guard<std::mutex> lock(G.mu);
    res.contentType = "application/octet-stream";
    res.body = toBytes(G.ds.truth);
}

void hLabels(const HttpRequest& req, HttpResponse& res) {
    std::lock_guard<std::mutex> lock(G.mu);
    const int id = std::atoi(req.param("id").c_str());
    for (const Run& r : G.runs) {
        if (r.id != id) continue;
        res.contentType = "application/octet-stream";
        res.body = toBytes(r.result.labels);
        return;
    }
    throw std::runtime_error("unknown run id");
}

/// The partition induced by the index at the given depth as labels, without recording a run.
void hPartition(const HttpRequest& req, HttpResponse& res) {
    std::lock_guard<std::mutex> lock(G.mu);
    const int depth = std::max(0, std::atoi(req.param("depth", "0").c_str()));
    res.contentType = "application/octet-stream";
    res.body = toBytes(initialClusters(*G.idx, depth).labels);
}

/// Bounding rectangles of all nodes down to the requested depth, projected to (dx, dy).
/// Each node: [depth, isLeaf, lo_x, lo_y, hi_x, hi_y, fanout].
void hTree(const HttpRequest& req, HttpResponse& res) {
    std::lock_guard<std::mutex> lock(G.mu);
    auto [dx, dy] = dims(std::atoi(req.param("dx", "0").c_str()), std::atoi(req.param("dy", "1").c_str()));
    const int maxDepth = std::atoi(req.param("depth", "0").c_str());
    const size_t limit = 200000;
    json nodes = json::array();
    bool truncated = false;
    if (G.idx->size() > 0) {
        std::vector<std::pair<const Node*, int>> queue{{G.idx->root(), 0}};
        for (size_t head = 0; head < queue.size(); ++head) {  // breadth-first: shallow levels first
            auto [n, depth] = queue[head];
            if (nodes.size() >= limit) { truncated = true; break; }
            nodes.push_back(json::array({depth, n->leaf ? 1 : 0, n->mbr.lo[dx], n->mbr.lo[dy],
                                         n->mbr.hi[dx], n->mbr.hi[dy],
                                         n->leaf ? n->pts.size() : n->children.size()}));
            if (depth < maxDepth)
                for (const Node* c : n->children) queue.push_back({c, depth + 1});
        }
    }
    res.body = json{{"nodes", nodes}, {"truncated", truncated}}.dump();
}

void hCluster(const HttpRequest& req, HttpResponse& res) {
    json j = body(req);
    std::lock_guard<std::mutex> lock(G.mu);
    const std::string method = j.value("method", "");
    const int depth = j.value("depth", G.idx->height());
    const int k = j.value("k", 5);
    const uint64_t seed = j.value("seed", uint64_t{1});
    const int maxIter = std::clamp(j.value("maxIter", 100), 1, 10000);
    const double eps = j.value("eps", 0.02);
    const int minPts = j.value("minPts", 5);
    if (G.ds.size() == 0) throw std::runtime_error("the data set is empty");
    if (depth < 0) throw std::runtime_error("depth must be >= 0");

    Run run;
    json params = {{"method", method}};
    if (method == "index-partition") {
        run.result = initialClusters(*G.idx, depth);
        params["depth"] = depth;
    } else if (method == "kmeans-baseline") {
        run.result = kmeansBaseline(G.ds, k, seed, maxIter);
        params.update({{"k", k}, {"seed", seed}});
    } else if (method == "kmeans-tree" || method == "kmeans-tree-seeded") {
        run.result = kmeansTree(*G.idx, depth, k, seed, maxIter, method == "kmeans-tree-seeded");
        params.update({{"k", k}, {"seed", seed}, {"depth", depth}});
    } else if (method == "dbscan-index" || method == "dbscan-tree-dense") {
        run.result = dbscanIndex(*G.idx, eps, minPts, method == "dbscan-tree-dense");
        params.update({{"eps", eps}, {"minPts", minPts}});
    } else if (method == "dbscan-baseline") {
        if (G.ds.size() > 200000)
            throw std::runtime_error("index-free DBSCAN is quadratic; limited to 200,000 points here");
        run.result = dbscanBaseline(G.ds, eps, minPts);
        params.update({{"eps", eps}, {"minPts", minPts}});
    } else if (method == "dbscan-nodes") {
        run.result = dbscanNodes(*G.idx, depth, eps, minPts);
        params.update({{"eps", eps}, {"minPts", minPts}, {"depth", depth}});
    } else {
        throw std::runtime_error("unknown clustering method: " + method);
    }
    const ClusterResult& r = run.result;
    const bool usesIndex = method != "kmeans-baseline" && method != "dbscan-baseline";
    run.id = G.nextRunId++;
    run.params = params;
    run.summary = {{"id", run.id},
                   {"method", method},
                   {"params", params},
                   {"structure", usesIndex ? structureName(G.indexParams.structure) : ""},
                   {"capacity", G.indexParams.capacity},
                   {"clusters", r.clusters},
                   {"noise", r.noise},
                   {"sse", r.sse},
                   {"ms", r.ms},
                   {"buildMs", usesIndex ? G.buildMs : 0.0},
                   {"iterations", r.iterations},
                   {"extra", r.extra},
                   {"ariTruth", nullptr}};
    if (G.ds.hasTruth()) run.summary["ariTruth"] = adjustedRand(r.labels, G.ds.truth);
    res.body = run.summary.dump();
    G.runs.push_back(std::move(run));
    if (G.runs.size() > 200) G.runs.erase(G.runs.begin());
}

/// Adjusted Rand index of every stored run against the reference run.
void hAri(const HttpRequest& req, HttpResponse& res) {
    std::lock_guard<std::mutex> lock(G.mu);
    const int ref = std::atoi(req.param("ref").c_str());
    const Run* refRun = nullptr;
    for (const Run& r : G.runs)
        if (r.id == ref) refRun = &r;
    if (!refRun) throw std::runtime_error("unknown run id");
    json out = json::object();
    for (const Run& r : G.runs)
        out[std::to_string(r.id)] = adjustedRand(r.result.labels, refRun->result.labels);
    res.body = out.dump();
}

void hClearRuns(const HttpRequest&, HttpResponse& res) {
    std::lock_guard<std::mutex> lock(G.mu);
    G.runs.clear();
    res.body = stateJson().dump();
}

/// Queries are posed in the displayed plane (dx, dy). In higher-dimensional data a range
/// query is unbounded in the other dimensions; kNN and radius queries use the centre of
/// the data space for them.
void hQuery(const HttpRequest& req, HttpResponse& res) {
    json j = body(req);
    std::lock_guard<std::mutex> lock(G.mu);
    auto [dx, dy] = dims(j.value("dx", 0), j.value("dy", 1));
    const int d = G.ds.d;
    const std::string type = j.value("type", "");
    QueryStats st;
    st.trace = true;
    json out;
    auto t0 = Clock::now();
    if (type == "range") {
        Rect q;
        q.lo.fill(0);
        q.hi.fill(0);
        for (int i = 0; i < d; ++i) {
            q.lo[i] = -std::numeric_limits<double>::infinity();
            q.hi[i] = std::numeric_limits<double>::infinity();
        }
        q.lo[dx] = std::min(j.at("x0").get<double>(), j.at("x1").get<double>());
        q.hi[dx] = std::max(j.at("x0").get<double>(), j.at("x1").get<double>());
        q.lo[dy] = std::min(j.at("y0").get<double>(), j.at("y1").get<double>());
        q.hi[dy] = std::max(j.at("y0").get<double>(), j.at("y1").get<double>());
        std::vector<uint32_t> ids;
        G.idx->range(q, ids, &st);
        out["ids"] = ids;
    } else if (type == "knn" || type == "radius") {
        double p[MAX_DIM];
        Rect b = G.ds.bounds();
        for (int i = 0; i < d; ++i) p[i] = G.ds.size() ? b.center(i) : 0.0;
        p[dx] = j.at("x").get<double>();
        p[dy] = j.at("y").get<double>();
        std::vector<uint32_t> ids;
        if (type == "knn") {
            const int k = std::clamp(j.value("k", 10), 1, 100000);
            auto nn = G.idx->knn(p, static_cast<size_t>(k), &st);
            for (auto& [dist, id] : nn) ids.push_back(id);
            out["radius"] = nn.empty() ? 0.0 : std::sqrt(nn.back().first);
        } else {
            const double eps = j.at("eps").get<double>();
            if (!(eps > 0)) throw std::runtime_error("eps must be positive");
            G.idx->radius(p, eps, ids, &st);
            out["radius"] = eps;
        }
        out["ids"] = ids;
    } else {
        throw std::runtime_error("unknown query type: " + type);
    }
    out["ms"] = std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
    out["stats"] = statsJson(st, dx, dy);
    out["totalNodes"] = G.idx->nodeCount();
    res.body = out.dump();
}

/// Inserts points given in the displayed plane; other coordinates are set to the centre.
void hInsert(const HttpRequest& req, HttpResponse& res) {
    json j = body(req);
    std::lock_guard<std::mutex> lock(G.mu);
    auto [dx, dy] = dims(j.value("dx", 0), j.value("dy", 1));
    const json& pts = j.at("points");
    if (!pts.is_array() || pts.size() > 100000) throw std::runtime_error("points: array of up to 100000 [x, y] pairs");
    Rect b = G.ds.bounds();
    auto t0 = Clock::now();
    for (const json& pt : pts) {
        double p[MAX_DIM];
        for (int i = 0; i < G.ds.d; ++i) p[i] = G.ds.size() ? b.center(i) : 0.0;
        p[dx] = pt.at(0).get<double>();
        p[dy] = pt.at(1).get<double>();
        if (!std::isfinite(p[dx]) || !std::isfinite(p[dy])) throw std::runtime_error("non-finite coordinate");
        G.idx->insert(G.ds.add(p));
    }
    const double ms = std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
    G.runs.clear();  // stored labelings no longer cover all points
    refreshCutSizes();
    json out = stateJson();
    out["insertMs"] = ms;
    res.body = out.dump();
}

}  // namespace

int main(int argc, char** argv) {
    std::string host = "127.0.0.1", web = TREECLUST_WEB_DIR;
    int port = 8080;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--port" && i + 1 < argc) port = std::atoi(argv[++i]);
        else if (a == "--host" && i + 1 < argc) host = argv[++i];
        else if (a == "--web" && i + 1 < argc) web = argv[++i];
        else {
            std::printf("usage: %s [--host 127.0.0.1] [--port 8080] [--web DIR]\n", argv[0]);
            return a == "--help" ? 0 : 1;
        }
    }

    GenParams g;
    g.kind = "mixed";
    g.n = 20000;
    g.k = 7;
    g.noise = 0.05;
    G.ds = generate(g);
    rebuildIndex();

    HttpServer srv;
    srv.serveStatic(web);
    srv.route("GET", "/api/state", hState);
    srv.route("POST", "/api/dataset/generate", hGenerate);
    srv.route("POST", "/api/dataset/upload", hUpload);
    srv.route("POST", "/api/index", hIndex);
    srv.route("GET", "/api/points", hPoints);
    srv.route("GET", "/api/truth", hTruth);
    srv.route("GET", "/api/labels", hLabels);
    srv.route("GET", "/api/tree", hTree);
    srv.route("GET", "/api/partition", hPartition);
    srv.route("POST", "/api/cluster", hCluster);
    srv.route("GET", "/api/ari", hAri);
    srv.route("POST", "/api/runs/clear", hClearRuns);
    srv.route("POST", "/api/query", hQuery);
    srv.route("POST", "/api/insert", hInsert);

    std::printf("treeclust: serving %s on http://%s:%d\n", web.c_str(), host.c_str(), port);
    std::fflush(stdout);
    return srv.listen(host, port) ? 0 : 1;
}
