// Point data sets: storage, synthetic generators and CSV import.
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "geometry.hpp"

namespace tc {

struct Dataset {
    int d = 2;
    std::string name;
    std::vector<double> x;       // row-major, n * d
    std::vector<int32_t> truth;  // optional ground-truth labels (-1 = noise); empty if unknown

    size_t size() const { return d ? x.size() / d : 0; }
    const double* pt(size_t i) const { return &x[i * d]; }
    bool hasTruth() const { return !truth.empty(); }
    Rect bounds() const;
    /// Appends a point and returns its id.
    uint32_t add(const double* p, int32_t label = -1);
};

struct GenParams {
    std::string kind = "blobs";  // blobs | mixed | moons | rings | uniform
    size_t n = 10000;
    int d = 2;
    int k = 5;            // number of generated clusters (blobs, mixed, rings)
    double spread = 0.03; // standard deviation / jitter
    double noise = 0.0;   // fraction of uniformly distributed noise points
    uint64_t seed = 42;
};

/// Throws std::runtime_error for invalid parameters.
Dataset generate(const GenParams& p);

/// Parses numeric CSV (separators: comma, semicolon, tab, blank). A non-numeric first
/// line is treated as header. If lastColumnIsLabel, the last column becomes ground truth.
Dataset parseCsv(const std::string& text, bool lastColumnIsLabel, const std::string& name);

}  // namespace tc
