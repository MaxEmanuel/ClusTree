// Basic geometry for the spatial index structures: axis-aligned rectangles
// (minimum bounding rectangles, MBRs) in up to MAX_DIM dimensions.
#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>

namespace tc {

constexpr int MAX_DIM = 8;

struct Rect {
    std::array<double, MAX_DIM> lo;
    std::array<double, MAX_DIM> hi;

    /// The empty rectangle: neutral element of expand().
    static Rect empty(int d) {
        Rect r;
        r.lo.fill(0.0);
        r.hi.fill(0.0);
        for (int i = 0; i < d; ++i) {
            r.lo[i] = std::numeric_limits<double>::infinity();
            r.hi[i] = -std::numeric_limits<double>::infinity();
        }
        return r;
    }

    static Rect point(const double* p, int d) {
        Rect r;
        r.lo.fill(0.0);
        r.hi.fill(0.0);
        for (int i = 0; i < d; ++i) r.lo[i] = r.hi[i] = p[i];
        return r;
    }

    bool isEmpty(int d) const { return d > 0 && lo[0] > hi[0]; }

    void expand(const double* p, int d) {
        for (int i = 0; i < d; ++i) {
            lo[i] = std::min(lo[i], p[i]);
            hi[i] = std::max(hi[i], p[i]);
        }
    }

    void expand(const Rect& o, int d) {
        for (int i = 0; i < d; ++i) {
            lo[i] = std::min(lo[i], o.lo[i]);
            hi[i] = std::max(hi[i], o.hi[i]);
        }
    }

    Rect united(const Rect& o, int d) const {
        Rect r = *this;
        r.expand(o, d);
        return r;
    }

    double area(int d) const {
        double a = 1.0;
        for (int i = 0; i < d; ++i) a *= (hi[i] - lo[i]);
        return a;
    }

    /// Sum of the edge lengths (the "margin" of the R*-tree paper, up to a constant factor).
    double margin(int d) const {
        double m = 0.0;
        for (int i = 0; i < d; ++i) m += (hi[i] - lo[i]);
        return m;
    }

    /// Squared length of the diagonal.
    double diag2(int d) const {
        double s = 0.0;
        for (int i = 0; i < d; ++i) s += (hi[i] - lo[i]) * (hi[i] - lo[i]);
        return s;
    }

    double overlap(const Rect& o, int d) const {
        double a = 1.0;
        for (int i = 0; i < d; ++i) {
            double w = std::min(hi[i], o.hi[i]) - std::max(lo[i], o.lo[i]);
            if (w <= 0.0) return 0.0;
            a *= w;
        }
        return a;
    }

    bool intersects(const Rect& o, int d) const {
        for (int i = 0; i < d; ++i)
            if (lo[i] > o.hi[i] || hi[i] < o.lo[i]) return false;
        return true;
    }

    bool contains(const double* p, int d) const {
        for (int i = 0; i < d; ++i)
            if (p[i] < lo[i] || p[i] > hi[i]) return false;
        return true;
    }

    bool contains(const Rect& o, int d) const {
        for (int i = 0; i < d; ++i)
            if (o.lo[i] < lo[i] || o.hi[i] > hi[i]) return false;
        return true;
    }

    double center(int i) const { return 0.5 * (lo[i] + hi[i]); }

    /// Squared minimum distance between the rectangle and a point.
    double minDist2(const double* p, int d) const {
        double s = 0.0;
        for (int i = 0; i < d; ++i) {
            double v = 0.0;
            if (p[i] < lo[i]) v = lo[i] - p[i];
            else if (p[i] > hi[i]) v = p[i] - hi[i];
            s += v * v;
        }
        return s;
    }

    /// Squared minimum distance between two rectangles.
    double minDist2(const Rect& o, int d) const {
        double s = 0.0;
        for (int i = 0; i < d; ++i) {
            double v = 0.0;
            if (o.hi[i] < lo[i]) v = lo[i] - o.hi[i];
            else if (o.lo[i] > hi[i]) v = o.lo[i] - hi[i];
            s += v * v;
        }
        return s;
    }

    bool equals(const Rect& o, int d) const {
        for (int i = 0; i < d; ++i)
            if (lo[i] != o.lo[i] || hi[i] != o.hi[i]) return false;
        return true;
    }
};

inline double dist2(const double* a, const double* b, int d) {
    double s = 0.0;
    for (int i = 0; i < d; ++i) s += (a[i] - b[i]) * (a[i] - b[i]);
    return s;
}

}  // namespace tc
