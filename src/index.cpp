#include "index.hpp"

#include <cassert>
#include <numeric>
#include <queue>
#include <stdexcept>

namespace tc {

namespace {
constexpr double INF = std::numeric_limits<double>::infinity();
constexpr int QUAD_MAX_DEPTH = 20;
constexpr size_t RSTAR_CANDIDATES = 32;
}  // namespace

const char* structureName(Structure s) {
    switch (s) {
        case Structure::RStar: return "rstar";
        case Structure::RQuadratic: return "rquadratic";
        case Structure::RLinear: return "rlinear";
        case Structure::STR: return "str";
        case Structure::KdTree: return "kdtree";
        case Structure::QuadTree: return "quadtree";
    }
    return "?";
}

Structure parseStructure(const std::string& s) {
    for (Structure k : {Structure::RStar, Structure::RQuadratic, Structure::RLinear,
                        Structure::STR, Structure::KdTree, Structure::QuadTree})
        if (s == structureName(k)) return k;
    throw std::runtime_error("unknown index structure: " + s);
}

SpatialIndex::SpatialIndex(const Dataset& ds, IndexParams params)
    : ds_(ds), params_(params), d_(ds.d) {
    if (d_ < 1 || d_ > MAX_DIM) throw std::runtime_error("unsupported dimensionality");
    if (params_.capacity < 4 || params_.capacity > 1024)
        throw std::runtime_error("node capacity must be in [4, 1024]");
    M_ = params_.capacity;
    m_ = std::clamp(static_cast<int>(params_.minFill * M_), 2, M_ / 2);
    root_ = new Node;
    root_->mbr = Rect::empty(d_);
    root_->cell = Rect::empty(d_);
    rootCell_ = Rect::empty(d_);
}

SpatialIndex::~SpatialIndex() { destroy(root_); }

void SpatialIndex::destroy(Node* n) {
    if (!n) return;
    for (Node* c : n->children) destroy(c);
    delete n;
}

void SpatialIndex::recomputeMbr(Node* n) const {
    n->mbr = Rect::empty(d_);
    if (n->leaf)
        for (uint32_t p : n->pts) n->mbr.expand(ds_.pt(p), d_);
    else
        for (Node* c : n->children) n->mbr.expand(c->mbr, d_);
}

void SpatialIndex::tightenUp(Node* n) const {
    for (; n; n = n->parent) recomputeMbr(n);
}

// ---------------------------------------------------------------------------------------
// Construction
// ---------------------------------------------------------------------------------------

void SpatialIndex::build() {
    destroy(root_);
    root_ = new Node;
    root_->mbr = Rect::empty(d_);
    root_->cell = Rect::empty(d_);
    size_ = 0;
    const size_t n = ds_.size();
    switch (params_.structure) {
        case Structure::RStar:
        case Structure::RQuadratic:
        case Structure::RLinear:
            for (size_t i = 0; i < n; ++i) insert(static_cast<uint32_t>(i));
            break;
        case Structure::STR:
            buildSTR();
            size_ = n;
            break;
        case Structure::KdTree: {
            std::vector<uint32_t> ids(n);
            std::iota(ids.begin(), ids.end(), 0u);
            kdBuildInto(root_, ids.data(), ids.data() + n);
            size_ = n;
            break;
        }
        case Structure::QuadTree: {
            std::vector<uint32_t> ids(n);
            std::iota(ids.begin(), ids.end(), 0u);
            rootCell_ = ds_.bounds();
            quadBuildInto(root_, ids.data(), ids.data() + n, rootCell_, 0);
            size_ = n;
            break;
        }
    }
}

void SpatialIndex::insert(uint32_t pid) {
    switch (params_.structure) {
        case Structure::KdTree: kdInsert(pid); break;
        case Structure::QuadTree: quadInsert(pid); break;
        default: {
            std::vector<char> reinserted(64, 0);
            insertEntry(Entry{Rect::point(ds_.pt(pid), d_), nullptr, pid}, 0, reinserted);
        }
    }
    ++size_;
}

// ---------------------------------------------------------------------------------------
// R-tree family: insertion
// ---------------------------------------------------------------------------------------

Node* SpatialIndex::chooseSubtree(const Rect& r, int level) const {
    Node* n = root_;
    static thread_local std::vector<std::pair<double, size_t>> cand;
    while (n->level > level) {
        const auto& ch = n->children;
        size_t best = 0;
        if (rstarLike() && n->level == 1) {
            // R*: children are leaves -> minimise the overlap enlargement. For large nodes
            // only the RSTAR_CANDIDATES entries with the least area enlargement are examined.
            cand.clear();
            for (size_t i = 0; i < ch.size(); ++i) {
                double a = ch[i]->mbr.area(d_);
                cand.emplace_back(ch[i]->mbr.united(r, d_).area(d_) - a, i);
            }
            if (cand.size() > RSTAR_CANDIDATES) {
                std::partial_sort(cand.begin(), cand.begin() + RSTAR_CANDIDATES, cand.end());
                cand.resize(RSTAR_CANDIDATES);
            }
            double bestOv = INF, bestEnl = INF, bestArea = INF;
            for (auto [enl, i] : cand) {
                const Rect& ri = ch[i]->mbr;
                Rect u = ri.united(r, d_);
                double ov = 0.0;
                for (size_t j = 0; j < ch.size(); ++j) {
                    if (j == i) continue;
                    ov += u.overlap(ch[j]->mbr, d_) - ri.overlap(ch[j]->mbr, d_);
                }
                double area = ri.area(d_);
                if (ov < bestOv || (ov == bestOv && (enl < bestEnl ||
                                    (enl == bestEnl && area < bestArea)))) {
                    bestOv = ov; bestEnl = enl; bestArea = area; best = i;
                }
            }
        } else {
            // Guttman / R* above the leaf level: least area enlargement, ties by area.
            double bestEnl = INF, bestArea = INF;
            for (size_t i = 0; i < ch.size(); ++i) {
                double area = ch[i]->mbr.area(d_);
                double enl = ch[i]->mbr.united(r, d_).area(d_) - area;
                if (enl < bestEnl || (enl == bestEnl && area < bestArea)) {
                    bestEnl = enl; bestArea = area; best = i;
                }
            }
        }
        n = ch[best];
    }
    return n;
}

void SpatialIndex::insertEntry(const Entry& e, int level, std::vector<char>& reinserted) {
    Node* n = chooseSubtree(e.r, level);
    if (level == 0) {
        n->pts.push_back(e.pid);
    } else {
        n->children.push_back(e.child);
        e.child->parent = n;
    }
    for (Node* a = n; a; a = a->parent) a->mbr.expand(e.r, d_);

    while (n && fanout(n) > static_cast<size_t>(M_)) {
        // R* overflow treatment: on the first overflow per level during one insertion,
        // reinsert a part of the entries instead of splitting (not for the root).
        if (params_.structure == Structure::RStar && n != root_ && params_.reinsertFrac > 0 &&
            !reinserted[n->level]) {
            reinserted[n->level] = 1;
            reinsert(n, reinserted);
            return;
        }
        Node* nn = splitNode(n);
        if (n == root_) {
            Node* r = new Node;
            r->leaf = false;
            r->level = n->level + 1;
            r->children = {n, nn};
            n->parent = nn->parent = r;
            recomputeMbr(r);
            root_ = r;
            return;
        }
        Node* p = n->parent;
        p->children.push_back(nn);
        nn->parent = p;
        tightenUp(p);
        n = p;
    }
}

void SpatialIndex::reinsert(Node* n, std::vector<char>& reinserted) {
    const int level = n->level;
    double c[MAX_DIM];
    for (int i = 0; i < d_; ++i) c[i] = n->mbr.center(i);

    std::vector<std::pair<double, Entry>> es;
    es.reserve(fanout(n));
    auto centerDist = [&](const Rect& r) {
        double s = 0.0;
        for (int i = 0; i < d_; ++i) s += (r.center(i) - c[i]) * (r.center(i) - c[i]);
        return s;
    };
    if (n->leaf) {
        for (uint32_t p : n->pts) {
            Rect r = Rect::point(ds_.pt(p), d_);
            es.push_back({centerDist(r), Entry{r, nullptr, p}});
        }
    } else {
        for (Node* ch : n->children) es.push_back({centerDist(ch->mbr), Entry{ch->mbr, ch, 0}});
    }
    // Remove the p entries farthest from the node centre ...
    std::sort(es.begin(), es.end(),
              [](const auto& a, const auto& b) { return a.first > b.first; });
    size_t p = static_cast<size_t>(std::lround(params_.reinsertFrac * M_));
    p = std::clamp<size_t>(p, 1, es.size() - m_);
    n->pts.clear();
    n->children.clear();
    for (size_t i = p; i < es.size(); ++i) {
        if (n->leaf) n->pts.push_back(es[i].second.pid);
        else n->children.push_back(es[i].second.child);
    }
    tightenUp(n);
    // ... and reinsert them, closest first ("close reinsert").
    for (size_t i = p; i-- > 0;) insertEntry(es[i].second, level, reinserted);
}

// ---------------------------------------------------------------------------------------
// R-tree family: node splitting
// ---------------------------------------------------------------------------------------

Node* SpatialIndex::splitNode(Node* n) {
    std::vector<Entry> es, g1, g2;
    es.reserve(fanout(n));
    if (n->leaf)
        for (uint32_t p : n->pts) es.push_back(Entry{Rect::point(ds_.pt(p), d_), nullptr, p});
    else
        for (Node* c : n->children) es.push_back(Entry{c->mbr, c, 0});

    switch (params_.structure) {
        case Structure::RQuadratic: splitQuadratic(es, g1, g2); break;
        case Structure::RLinear: splitLinear(es, g1, g2); break;
        default: splitRStar(es, g1, g2); break;
    }

    Node* nn = new Node;
    nn->level = n->level;
    nn->leaf = n->leaf;
    n->pts.clear();
    n->children.clear();
    auto fill = [](Node* dst, const std::vector<Entry>& g) {
        for (const Entry& e : g) {
            if (dst->leaf) dst->pts.push_back(e.pid);
            else { dst->children.push_back(e.child); e.child->parent = dst; }
        }
    };
    fill(n, g1);
    fill(nn, g2);
    recomputeMbr(n);
    recomputeMbr(nn);
    return nn;
}

void SpatialIndex::splitRStar(std::vector<Entry>& es, std::vector<Entry>& g1,
                              std::vector<Entry>& g2) const {
    const int n = static_cast<int>(es.size());
    const bool pointEntries = es[0].child == nullptr;
    std::vector<Rect> prefix(n), suffix(n);

    struct Dist { double overlap = INF, area = INF; int k = 0; int axis = 0; bool byHi = false; };
    Dist best;
    double bestMargin = INF;

    auto sortBy = [&](int axis, bool byHi) {
        std::sort(es.begin(), es.end(), [axis, byHi](const Entry& a, const Entry& b) {
            return byHi ? a.r.hi[axis] < b.r.hi[axis] : a.r.lo[axis] < b.r.lo[axis];
        });
    };

    // ChooseSplitAxis: the axis with the least sum of margins over all distributions.
    // ChooseSplitIndex: on that axis the distribution with least overlap, ties by area.
    for (int axis = 0; axis < d_; ++axis) {
        double marginSum = 0.0;
        Dist axisBest;
        for (int byHi = 0; byHi < (pointEntries ? 1 : 2); ++byHi) {
            sortBy(axis, byHi);
            prefix[0] = es[0].r;
            for (int i = 1; i < n; ++i) prefix[i] = prefix[i - 1].united(es[i].r, d_);
            suffix[n - 1] = es[n - 1].r;
            for (int i = n - 2; i >= 0; --i) suffix[i] = suffix[i + 1].united(es[i].r, d_);
            for (int k = m_; k <= n - m_; ++k) {  // first group: es[0..k)
                const Rect& a = prefix[k - 1];
                const Rect& b = suffix[k];
                marginSum += a.margin(d_) + b.margin(d_);
                double ov = a.overlap(b, d_);
                double ar = a.area(d_) + b.area(d_);
                if (ov < axisBest.overlap || (ov == axisBest.overlap && ar < axisBest.area))
                    axisBest = Dist{ov, ar, k, axis, byHi != 0};
            }
        }
        if (marginSum < bestMargin) {
            bestMargin = marginSum;
            best = axisBest;
        }
    }
    sortBy(best.axis, best.byHi);
    g1.assign(es.begin(), es.begin() + best.k);
    g2.assign(es.begin() + best.k, es.end());
}

void SpatialIndex::distributeRest(std::vector<Entry>& rest, std::vector<Entry>& g1,
                                  std::vector<Entry>& g2, Rect r1, Rect r2,
                                  bool pickNext) const {
    const size_t m = static_cast<size_t>(m_);
    while (!rest.empty()) {
        // Honour the minimum fill: if one group needs all remaining entries, it gets them.
        if (g1.size() + rest.size() <= m) {
            g1.insert(g1.end(), rest.begin(), rest.end());
            break;
        }
        if (g2.size() + rest.size() <= m) {
            g2.insert(g2.end(), rest.begin(), rest.end());
            break;
        }
        size_t pick = rest.size() - 1;
        const double a1 = r1.area(d_), a2 = r2.area(d_);
        if (pickNext) {
            // Quadratic PickNext: the entry with the greatest preference for one group.
            double bestDiff = -1.0;
            for (size_t i = 0; i < rest.size(); ++i) {
                double d1 = r1.united(rest[i].r, d_).area(d_) - a1;
                double d2 = r2.united(rest[i].r, d_).area(d_) - a2;
                if (std::abs(d1 - d2) > bestDiff) { bestDiff = std::abs(d1 - d2); pick = i; }
            }
        }
        Entry e = rest[pick];
        rest[pick] = rest.back();
        rest.pop_back();
        double d1 = r1.united(e.r, d_).area(d_) - a1;
        double d2 = r2.united(e.r, d_).area(d_) - a2;
        bool first;
        if (d1 != d2) first = d1 < d2;
        else if (a1 != a2) first = a1 < a2;
        else first = g1.size() <= g2.size();
        if (first) { g1.push_back(e); r1.expand(e.r, d_); }
        else { g2.push_back(e); r2.expand(e.r, d_); }
    }
}

void SpatialIndex::splitQuadratic(std::vector<Entry>& es, std::vector<Entry>& g1,
                                  std::vector<Entry>& g2) const {
    // PickSeeds: the pair that would waste the most area if put into one node.
    size_t s1 = 0, s2 = 1;
    double worst = -INF;
    for (size_t i = 0; i < es.size(); ++i) {
        double ai = es[i].r.area(d_);
        for (size_t j = i + 1; j < es.size(); ++j) {
            double waste = es[i].r.united(es[j].r, d_).area(d_) - ai - es[j].r.area(d_);
            if (waste > worst) { worst = waste; s1 = i; s2 = j; }
        }
    }
    g1.push_back(es[s1]);
    g2.push_back(es[s2]);
    Rect r1 = es[s1].r, r2 = es[s2].r;
    es.erase(es.begin() + s2);  // s2 > s1
    es.erase(es.begin() + s1);
    distributeRest(es, g1, g2, r1, r2, true);
}

void SpatialIndex::splitLinear(std::vector<Entry>& es, std::vector<Entry>& g1,
                               std::vector<Entry>& g2) const {
    // LinearPickSeeds: the pair with the greatest normalised separation along any axis.
    size_t s1 = 0, s2 = 1;
    double bestSep = -INF;
    for (int axis = 0; axis < d_; ++axis) {
        size_t hiLo = 0, loHi = 0;
        double minLo = INF, maxHi = -INF;
        for (size_t i = 0; i < es.size(); ++i) {
            if (es[i].r.lo[axis] > es[hiLo].r.lo[axis]) hiLo = i;
            if (es[i].r.hi[axis] < es[loHi].r.hi[axis]) loHi = i;
            minLo = std::min(minLo, es[i].r.lo[axis]);
            maxHi = std::max(maxHi, es[i].r.hi[axis]);
        }
        double width = maxHi - minLo;
        if (hiLo == loHi || width <= 0) continue;
        double sep = (es[hiLo].r.lo[axis] - es[loHi].r.hi[axis]) / width;
        if (sep > bestSep) { bestSep = sep; s1 = loHi; s2 = hiLo; }
    }
    if (s1 > s2) std::swap(s1, s2);
    g1.push_back(es[s1]);
    g2.push_back(es[s2]);
    Rect r1 = es[s1].r, r2 = es[s2].r;
    es.erase(es.begin() + s2);
    es.erase(es.begin() + s1);
    distributeRest(es, g1, g2, r1, r2, false);
}

// ---------------------------------------------------------------------------------------
// Sort-Tile-Recursive bulk loading
// ---------------------------------------------------------------------------------------

namespace {
// Orders idx[b, e) such that consecutive runs of `cap` items form spatially compact tiles.
template <class Coord>
void strOrder(std::vector<uint32_t>& idx, size_t b, size_t e, int dim, int d, size_t cap,
              const Coord& coord) {
    std::sort(idx.begin() + b, idx.begin() + e,
              [&](uint32_t x, uint32_t y) { return coord(x, dim) < coord(y, dim); });
    if (dim == d - 1) return;
    const size_t n = e - b;
    const double pages = std::ceil(static_cast<double>(n) / cap);
    const int k = d - dim;  // remaining dimensions
    const size_t slab =
        cap * static_cast<size_t>(std::ceil(std::pow(pages, (k - 1.0) / k) - 1e-9));
    for (size_t s = b; s < e; s += slab)
        strOrder(idx, s, std::min(e, s + slab), dim + 1, d, cap, coord);
}
}  // namespace

void SpatialIndex::buildSTR() {
    const size_t n = ds_.size();
    if (n == 0) return;
    const size_t cap = static_cast<size_t>(M_);
    std::vector<uint32_t> idx(n);
    std::iota(idx.begin(), idx.end(), 0u);
    strOrder(idx, 0, n, 0, d_, cap,
             [&](uint32_t i, int dim) { return ds_.pt(i)[dim]; });

    std::vector<Node*> nodes;
    for (size_t s = 0; s < n; s += cap) {
        Node* leaf = new Node;
        leaf->pts.assign(idx.begin() + s, idx.begin() + std::min(n, s + cap));
        recomputeMbr(leaf);
        nodes.push_back(leaf);
    }
    int level = 0;
    while (nodes.size() > 1) {
        ++level;
        idx.resize(nodes.size());
        std::iota(idx.begin(), idx.end(), 0u);
        strOrder(idx, 0, idx.size(), 0, d_, cap,
                 [&](uint32_t i, int dim) { return nodes[i]->mbr.center(dim); });
        std::vector<Node*> parents;
        for (size_t s = 0; s < idx.size(); s += cap) {
            Node* p = new Node;
            p->leaf = false;
            p->level = level;
            for (size_t i = s; i < std::min(idx.size(), s + cap); ++i) {
                p->children.push_back(nodes[idx[i]]);
                nodes[idx[i]]->parent = p;
            }
            recomputeMbr(p);
            parents.push_back(p);
        }
        nodes.swap(parents);
    }
    delete root_;
    root_ = nodes[0];
}

// ---------------------------------------------------------------------------------------
// Bucket k-d tree
// ---------------------------------------------------------------------------------------

void SpatialIndex::kdBuildInto(Node* n, uint32_t* b, uint32_t* e) {
    n->mbr = Rect::empty(d_);
    for (uint32_t* p = b; p != e; ++p) n->mbr.expand(ds_.pt(*p), d_);
    const size_t cnt = static_cast<size_t>(e - b);
    int dim = 0;
    double ext = 0.0;
    for (int i = 0; i < d_ && cnt; ++i)
        if (n->mbr.hi[i] - n->mbr.lo[i] > ext) { ext = n->mbr.hi[i] - n->mbr.lo[i]; dim = i; }
    if (cnt <= static_cast<size_t>(M_) || ext <= 0.0) {
        n->leaf = true;
        n->pts.assign(b, e);
        return;
    }
    n->leaf = false;
    n->pts.clear();
    n->pts.shrink_to_fit();
    uint32_t* mid = b + cnt / 2;
    std::nth_element(b, mid, e, [&](uint32_t x, uint32_t y) {
        return ds_.pt(x)[dim] < ds_.pt(y)[dim];
    });
    n->splitDim = dim;
    n->splitVal = ds_.pt(*mid)[dim];
    for (int side = 0; side < 2; ++side) {
        Node* c = new Node;
        c->parent = n;
        n->children.push_back(c);
        kdBuildInto(c, side == 0 ? b : mid, side == 0 ? mid : e);
    }
}

void SpatialIndex::kdInsert(uint32_t pid) {
    const double* p = ds_.pt(pid);
    Node* n = root_;
    for (;;) {
        n->mbr.expand(p, d_);
        if (n->leaf) break;
        n = n->children[p[n->splitDim] < n->splitVal ? 0 : 1];
    }
    n->pts.push_back(pid);
    if (n->pts.size() > static_cast<size_t>(M_)) {
        std::vector<uint32_t> tmp = std::move(n->pts);
        n->pts.clear();
        kdBuildInto(n, tmp.data(), tmp.data() + tmp.size());
    }
}

// ---------------------------------------------------------------------------------------
// Bucket quadtree (2^d-ary)
// ---------------------------------------------------------------------------------------

uint32_t SpatialIndex::quadCode(const Rect& cell, const double* p) const {
    uint32_t code = 0;
    for (int i = 0; i < d_; ++i)
        if (p[i] >= cell.center(i)) code |= (1u << i);
    return code;
}

Rect SpatialIndex::quadSubCell(const Rect& cell, uint32_t code) const {
    Rect r = cell;
    for (int i = 0; i < d_; ++i) {
        if (code & (1u << i)) r.lo[i] = cell.center(i);
        else r.hi[i] = cell.center(i);
    }
    return r;
}

void SpatialIndex::quadBuildInto(Node* n, uint32_t* b, uint32_t* e, const Rect& cell,
                                 int depth) {
    n->cell = cell;
    n->mbr = Rect::empty(d_);
    for (uint32_t* p = b; p != e; ++p) n->mbr.expand(ds_.pt(*p), d_);
    const size_t cnt = static_cast<size_t>(e - b);
    if (cnt <= static_cast<size_t>(M_) || depth >= QUAD_MAX_DEPTH || n->mbr.diag2(d_) <= 0.0) {
        n->leaf = true;
        n->pts.assign(b, e);
        return;
    }
    n->leaf = false;
    n->pts.clear();
    n->pts.shrink_to_fit();
    std::vector<std::pair<uint32_t, uint32_t>> coded(cnt);
    for (size_t i = 0; i < cnt; ++i) coded[i] = {quadCode(cell, ds_.pt(b[i])), b[i]};
    std::sort(coded.begin(), coded.end());
    for (size_t i = 0; i < cnt; ++i) b[i] = coded[i].second;
    for (size_t s = 0; s < cnt;) {
        size_t t = s;
        while (t < cnt && coded[t].first == coded[s].first) ++t;
        Node* c = new Node;
        c->parent = n;
        c->code = coded[s].first;
        n->children.push_back(c);
        quadBuildInto(c, b + s, b + t, quadSubCell(cell, c->code), depth + 1);
        s = t;
    }
}

void SpatialIndex::quadInsert(uint32_t pid) {
    const double* p = ds_.pt(pid);
    if (size_ == 0 || !rootCell_.contains(p, d_)) {
        // The point lies outside the indexed region: rebuild over a region with headroom.
        std::vector<uint32_t> ids;
        collect(root_, ids);
        ids.push_back(pid);
        Rect b = Rect::empty(d_);
        for (uint32_t i : ids) b.expand(ds_.pt(i), d_);
        if (size_ > 0) {
            for (int i = 0; i < d_; ++i) {
                double w = 0.25 * (b.hi[i] - b.lo[i]);
                b.lo[i] -= w;
                b.hi[i] += w;
            }
        }
        rootCell_ = b;
        for (Node* c : root_->children) destroy(c);
        root_->children.clear();
        quadBuildInto(root_, ids.data(), ids.data() + ids.size(), rootCell_, 0);
        return;
    }
    Node* n = root_;
    int depth = 0;
    for (;;) {
        n->mbr.expand(p, d_);
        if (n->leaf) break;
        uint32_t code = quadCode(n->cell, p);
        Node* next = nullptr;
        for (Node* c : n->children)
            if (c->code == code) { next = c; break; }
        if (!next) {
            next = new Node;
            next->parent = n;
            next->code = code;
            next->cell = quadSubCell(n->cell, code);
            next->mbr = Rect::empty(d_);
            n->children.push_back(next);
        }
        n = next;
        ++depth;
    }
    n->pts.push_back(pid);
    if (n->pts.size() > static_cast<size_t>(M_) && depth < QUAD_MAX_DEPTH &&
        n->mbr.diag2(d_) > 0.0) {
        std::vector<uint32_t> tmp = std::move(n->pts);
        n->pts.clear();
        Rect cell = n->cell;
        quadBuildInto(n, tmp.data(), tmp.data() + tmp.size(), cell, depth);
    }
}

// ---------------------------------------------------------------------------------------
// Queries (generic over the MBR hierarchy)
// ---------------------------------------------------------------------------------------

void SpatialIndex::range(const Rect& q, std::vector<uint32_t>& out, QueryStats* st) const {
    if (size_ == 0) return;
    std::vector<const Node*> stack{root_};
    while (!stack.empty()) {
        const Node* n = stack.back();
        stack.pop_back();
        if (!n->mbr.intersects(q, d_)) continue;
        if (st) {
            ++st->nodesVisited;
            if (n->leaf) ++st->leavesVisited;
            if (st->trace) st->visited.push_back(n);
        }
        if (n->leaf) {
            if (st) st->pointsChecked += n->pts.size();
            for (uint32_t p : n->pts)
                if (q.contains(ds_.pt(p), d_)) out.push_back(p);
        } else {
            for (const Node* c : n->children) stack.push_back(c);
        }
    }
}

void SpatialIndex::radius(const double* p, double eps, std::vector<uint32_t>& out,
                          QueryStats* st) const {
    if (size_ == 0) return;
    const double e2 = eps * eps;
    static thread_local std::vector<const Node*> stack;
    stack.clear();
    stack.push_back(root_);
    while (!stack.empty()) {
        const Node* n = stack.back();
        stack.pop_back();
        if (st) {
            ++st->nodesVisited;
            if (n->leaf) ++st->leavesVisited;
            if (st->trace) st->visited.push_back(n);
        }
        if (n->leaf) {
            if (st) st->pointsChecked += n->pts.size();
            for (uint32_t q : n->pts)
                if (dist2(p, ds_.pt(q), d_) <= e2) out.push_back(q);
        } else {
            for (const Node* c : n->children)
                if (c->mbr.minDist2(p, d_) <= e2) stack.push_back(c);
        }
    }
}

std::vector<std::pair<double, uint32_t>> SpatialIndex::knn(const double* p, size_t k,
                                                           QueryStats* st) const {
    // Best-first search (Hjaltason & Samet): nodes and points share one priority queue.
    struct Item {
        double dist;
        const Node* node;  // nullptr for points
        uint32_t pid;
        bool operator>(const Item& o) const { return dist > o.dist; }
    };
    std::vector<std::pair<double, uint32_t>> res;
    if (size_ == 0 || k == 0) return res;
    std::priority_queue<Item, std::vector<Item>, std::greater<Item>> pq;
    pq.push({root_->mbr.minDist2(p, d_), root_, 0});
    while (!pq.empty() && res.size() < k) {
        Item it = pq.top();
        pq.pop();
        if (!it.node) {
            res.emplace_back(it.dist, it.pid);
            continue;
        }
        const Node* n = it.node;
        if (st) {
            ++st->nodesVisited;
            if (n->leaf) ++st->leavesVisited;
            if (st->trace) st->visited.push_back(n);
        }
        if (n->leaf) {
            if (st) st->pointsChecked += n->pts.size();
            for (uint32_t q : n->pts) pq.push({dist2(p, ds_.pt(q), d_), nullptr, q});
        } else {
            for (const Node* c : n->children) pq.push({c->mbr.minDist2(p, d_), c, 0});
        }
    }
    return res;
}

// ---------------------------------------------------------------------------------------
// Introspection
// ---------------------------------------------------------------------------------------

namespace {
template <class F>
void walk(const Node* n, int depth, const F& f) {
    f(n, depth);
    for (const Node* c : n->children) walk(c, depth + 1, f);
}
}  // namespace

int SpatialIndex::height() const {
    int h = 0;
    walk(root_, 0, [&](const Node*, int depth) { h = std::max(h, depth); });
    return h;
}

size_t SpatialIndex::nodeCount() const {
    size_t c = 0;
    walk(root_, 0, [&](const Node*, int) { ++c; });
    return c;
}

size_t SpatialIndex::leafCount() const {
    size_t c = 0;
    walk(root_, 0, [&](const Node* n, int) { c += n->leaf; });
    return c;
}

std::vector<const Node*> SpatialIndex::cut(int depth) const {
    std::vector<const Node*> out;
    if (size_ == 0) return out;
    std::vector<std::pair<const Node*, int>> stack{{root_, 0}};
    while (!stack.empty()) {
        auto [n, dep] = stack.back();
        stack.pop_back();
        if (dep >= depth || n->leaf) {
            if (!n->leaf || !n->pts.empty()) out.push_back(n);
            continue;
        }
        for (const Node* c : n->children) stack.push_back({c, dep + 1});
    }
    return out;
}

void SpatialIndex::collect(const Node* n, std::vector<uint32_t>& out) {
    if (n->leaf) {
        out.insert(out.end(), n->pts.begin(), n->pts.end());
        return;
    }
    for (const Node* c : n->children) collect(c, out);
}

bool SpatialIndex::validate(std::string& err) const {
    const Structure s = params_.structure;
    const bool rtree = s != Structure::KdTree && s != Structure::QuadTree;
    const bool minFillHolds =
        s == Structure::RStar || s == Structure::RQuadratic || s == Structure::RLinear;
    std::vector<char> seen(ds_.size(), 0);
    size_t total = 0;
    int leafDepth = -1;
    bool ok = true;
    auto fail = [&](const std::string& msg) { if (ok) { ok = false; err = msg; } };

    walk(root_, 0, [&](const Node* n, int depth) {
        if (!ok) return;
        if (n->leaf != n->children.empty() && !(n->leaf && n->children.empty()))
            return fail("leaf flag inconsistent");
        if (!n->leaf && !n->pts.empty()) return fail("inner node holds points");
        for (const Node* c : n->children)
            if (c->parent != n) return fail("broken parent pointer");
        if (size_ > 0) {
            Rect r = Rect::empty(d_);
            if (n->leaf) for (uint32_t p : n->pts) r.expand(ds_.pt(p), d_);
            else for (const Node* c : n->children) r.expand(c->mbr, d_);
            if (!(n->leaf && n->pts.empty()) && !r.equals(n->mbr, d_))
                return fail("MBR not tight at depth " + std::to_string(depth));
        }
        const size_t f = fanout(n);
        if (rtree) {
            if (f > static_cast<size_t>(M_)) return fail("node overflow");
            if (minFillHolds && n != root_ && f < static_cast<size_t>(m_))
                return fail("node underflow");
            if (n == root_ && !n->leaf && f < 2) return fail("inner root with < 2 children");
            if (n->leaf) {
                if (leafDepth < 0) leafDepth = depth;
                if (depth != leafDepth) return fail("tree not balanced");
                if (n->level != 0) return fail("leaf level != 0");
            } else {
                for (const Node* c : n->children)
                    if (c->level != n->level - 1) return fail("level numbering broken");
            }
        } else if (s == Structure::KdTree) {
            if (n->leaf && f > static_cast<size_t>(M_) && n->mbr.diag2(d_) > 0)
                return fail("k-d leaf overflow");
            if (!n->leaf && f != 2) return fail("k-d inner node without 2 children");
        } else {
            if (!n->leaf)
                for (const Node* c : n->children)
                    if (!n->cell.contains(c->cell, d_)) return fail("quadtree cell nesting");
            if (n->leaf)
                for (uint32_t p : n->pts)
                    if (!n->cell.contains(ds_.pt(p), d_)) return fail("point outside cell");
        }
        for (uint32_t p : n->pts) {
            if (p >= seen.size()) return fail("point id out of range");
            if (seen[p]) return fail("duplicate point id");
            seen[p] = 1;
            ++total;
        }
    });
    if (ok && total != size_) fail("point count mismatch");
    return ok;
}

}  // namespace tc
