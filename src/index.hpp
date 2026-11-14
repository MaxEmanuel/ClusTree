// Hierarchical spatial index structures over point data.
//
// All structures share one node layout (a hierarchy of minimum bounding rectangles), so
// queries and the clustering algorithms work on any of them:
//
//   RStar       R*-tree (Beckmann et al., SIGMOD 1990): overlap-minimising ChooseSubtree,
//               margin-based split, forced reinsertion
//   RQuadratic  R-tree with Guttman's quadratic split (SIGMOD 1984)
//   RLinear     R-tree with Guttman's linear split
//   STR         R-tree bulk-loaded with Sort-Tile-Recursive (Leutenegger et al., ICDE 1997);
//               later insertions use the R*-tree algorithms
//   KdTree      bucket k-d tree, median split along the widest dimension
//   QuadTree    bucket quadtree (2^d-ary region tree), regular midpoint decomposition
#pragma once

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "dataset.hpp"
#include "geometry.hpp"

namespace tc {

enum class Structure { RStar, RQuadratic, RLinear, STR, KdTree, QuadTree };

const char* structureName(Structure s);
/// Throws std::runtime_error for unknown names.
Structure parseStructure(const std::string& s);

struct Node {
    Rect mbr;                     // tight bounding rectangle of all points below
    Node* parent = nullptr;
    int level = 0;                // R-trees: height above the leaf level (leaf = 0)
    bool leaf = true;
    std::vector<Node*> children;  // inner nodes
    std::vector<uint32_t> pts;    // leaves: point ids
    int splitDim = -1;            // k-d tree
    double splitVal = 0.0;        // k-d tree
    uint32_t code = 0;            // quadtree: orthant within the parent cell
    Rect cell;                    // quadtree: region covered by this node
};

struct QueryStats {
    size_t nodesVisited = 0;
    size_t leavesVisited = 0;
    size_t pointsChecked = 0;
    bool trace = false;                  // record the visited nodes
    std::vector<const Node*> visited;
};

struct IndexParams {
    Structure structure = Structure::RStar;
    int capacity = 16;           // M: max. entries per node (leaf bucket size for kd/quad)
    double minFill = 0.4;        // m = minFill * M (R-tree variants)
    double reinsertFrac = 0.3;   // p = reinsertFrac * M (R*-tree forced reinsertion)
};

class SpatialIndex {
public:
    /// The index references the data set; insert() expects the point to be in the data set.
    SpatialIndex(const Dataset& ds, IndexParams params);
    ~SpatialIndex();
    SpatialIndex(const SpatialIndex&) = delete;
    SpatialIndex& operator=(const SpatialIndex&) = delete;

    /// Builds the index over all points currently in the data set.
    void build();
    /// Inserts the (already stored) point with the given id.
    void insert(uint32_t pid);

    void range(const Rect& q, std::vector<uint32_t>& out, QueryStats* st = nullptr) const;
    void radius(const double* p, double eps, std::vector<uint32_t>& out,
                QueryStats* st = nullptr) const;
    /// k nearest neighbours as (squared distance, id), ascending.
    std::vector<std::pair<double, uint32_t>> knn(const double* p, size_t k,
                                                 QueryStats* st = nullptr) const;

    const Node* root() const { return root_; }
    const Dataset& data() const { return ds_; }
    const IndexParams& params() const { return params_; }
    int dim() const { return d_; }
    size_t size() const { return size_; }
    /// Depth of the deepest leaf (root = depth 0).
    int height() const;
    size_t nodeCount() const;
    size_t leafCount() const;
    /// The nodes at the given depth plus all leaves above it: a partition of the data.
    std::vector<const Node*> cut(int depth) const;
    /// Appends all point ids below n to out.
    static void collect(const Node* n, std::vector<uint32_t>& out);
    /// Checks the structural invariants; returns false and a message on violation.
    bool validate(std::string& err) const;

private:
    struct Entry {
        Rect r;
        Node* child;
        uint32_t pid;
    };

    const Dataset& ds_;
    IndexParams params_;
    int d_;
    int M_;
    int m_;
    Node* root_ = nullptr;
    size_t size_ = 0;
    Rect rootCell_;  // quadtree

    static void destroy(Node* n);
    void recomputeMbr(Node* n) const;
    void tightenUp(Node* n) const;
    static size_t fanout(const Node* n) { return n->leaf ? n->pts.size() : n->children.size(); }

    // R-tree family
    bool rstarLike() const {
        return params_.structure == Structure::RStar || params_.structure == Structure::STR;
    }
    Node* chooseSubtree(const Rect& r, int level) const;
    void insertEntry(const Entry& e, int level, std::vector<char>& reinserted);
    void reinsert(Node* n, std::vector<char>& reinserted);
    Node* splitNode(Node* n);
    void splitRStar(std::vector<Entry>& es, std::vector<Entry>& g1, std::vector<Entry>& g2) const;
    void splitQuadratic(std::vector<Entry>& es, std::vector<Entry>& g1, std::vector<Entry>& g2) const;
    void splitLinear(std::vector<Entry>& es, std::vector<Entry>& g1, std::vector<Entry>& g2) const;
    void distributeRest(std::vector<Entry>& rest, std::vector<Entry>& g1, std::vector<Entry>& g2,
                        Rect r1, Rect r2, bool pickNext) const;
    void buildSTR();

    // k-d tree / quadtree
    void kdBuildInto(Node* n, uint32_t* b, uint32_t* e);
    void kdInsert(uint32_t pid);
    void quadBuildInto(Node* n, uint32_t* b, uint32_t* e, const Rect& cell, int depth);
    void quadInsert(uint32_t pid);
    uint32_t quadCode(const Rect& cell, const double* p) const;
    Rect quadSubCell(const Rect& cell, uint32_t code) const;
};

}  // namespace tc
