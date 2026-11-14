#include "dataset.hpp"

#include <cstdlib>
#include <map>
#include <random>
#include <stdexcept>

namespace tc {

Rect Dataset::bounds() const {
    Rect r = Rect::empty(d);
    for (size_t i = 0, n = size(); i < n; ++i) r.expand(pt(i), d);
    return r;
}

uint32_t Dataset::add(const double* p, int32_t label) {
    uint32_t id = static_cast<uint32_t>(size());
    x.insert(x.end(), p, p + d);
    if (hasTruth()) truth.push_back(label);
    return id;
}

Dataset generate(const GenParams& g) {
    if (g.n == 0 || g.n > 5'000'000) throw std::runtime_error("n must be in [1, 5000000]");
    if (g.d < 2 || g.d > MAX_DIM) throw std::runtime_error("d must be in [2, 8]");
    if (g.k < 1 || g.k > 10000) throw std::runtime_error("k must be in [1, 10000]");
    if (g.noise < 0 || g.noise >= 1) throw std::runtime_error("noise must be in [0, 1)");

    std::mt19937_64 rng(g.seed);
    std::uniform_real_distribution<double> uni(0.0, 1.0);
    std::normal_distribution<double> gauss(0.0, 1.0);

    Dataset ds;
    ds.name = g.kind;
    ds.d = (g.kind == "moons" || g.kind == "rings") ? 2 : g.d;
    const int d = ds.d;
    const size_t nNoise = static_cast<size_t>(g.noise * g.n);
    const size_t nData = g.n - nNoise;
    ds.x.reserve(g.n * d);
    ds.truth.reserve(g.n);
    double p[MAX_DIM];

    if (g.kind == "blobs" || g.kind == "mixed") {
        // Gaussian blobs; "mixed" varies size and density per blob.
        std::vector<double> centers(static_cast<size_t>(g.k) * d), sd(g.k), weight(g.k);
        double wsum = 0;
        for (int c = 0; c < g.k; ++c) {
            for (int j = 0; j < d; ++j) centers[c * d + j] = 0.1 + 0.8 * uni(rng);
            bool mixed = g.kind == "mixed";
            sd[c] = mixed ? g.spread * (0.4 + 1.6 * uni(rng)) : g.spread;
            weight[c] = mixed ? 0.3 + uni(rng) : 1.0;
            wsum += weight[c];
        }
        size_t made = 0;
        for (int c = 0; c < g.k; ++c) {
            size_t cnt = (c == g.k - 1) ? nData - made
                                        : static_cast<size_t>(nData * weight[c] / wsum);
            for (size_t i = 0; i < cnt; ++i) {
                for (int j = 0; j < d; ++j) p[j] = centers[c * d + j] + sd[c] * gauss(rng);
                ds.x.insert(ds.x.end(), p, p + d);
                ds.truth.push_back(c);
            }
            made += cnt;
        }
    } else if (g.kind == "moons") {
        for (size_t i = 0; i < nData; ++i) {
            int c = static_cast<int>(i % 2);
            double t = M_PI * uni(rng);
            if (c == 0) { p[0] = std::cos(t); p[1] = std::sin(t); }
            else { p[0] = 1.0 - std::cos(t); p[1] = 0.5 - std::sin(t); }
            p[0] = (p[0] + 1.0) / 3.0 + g.spread * gauss(rng);
            p[1] = (p[1] + 0.75) / 3.0 + g.spread * gauss(rng);
            ds.x.insert(ds.x.end(), p, p + 2);
            ds.truth.push_back(c);
        }
    } else if (g.kind == "rings") {
        // Concentric rings; point counts proportional to the circumference.
        double rsum = 0;
        for (int c = 0; c < g.k; ++c) rsum += (c + 1);
        size_t made = 0;
        for (int c = 0; c < g.k; ++c) {
            size_t cnt = (c == g.k - 1) ? nData - made
                                        : static_cast<size_t>(nData * (c + 1) / rsum);
            double r = 0.45 * (c + 1) / g.k;
            for (size_t i = 0; i < cnt; ++i) {
                double t = 2 * M_PI * uni(rng);
                p[0] = 0.5 + r * std::cos(t) + g.spread * gauss(rng);
                p[1] = 0.5 + r * std::sin(t) + g.spread * gauss(rng);
                ds.x.insert(ds.x.end(), p, p + 2);
                ds.truth.push_back(c);
            }
            made += cnt;
        }
    } else if (g.kind == "uniform") {
        for (size_t i = 0; i < nData; ++i) {
            for (int j = 0; j < d; ++j) p[j] = uni(rng);
            ds.x.insert(ds.x.end(), p, p + d);
        }
    } else {
        throw std::runtime_error("unknown generator: " + g.kind);
    }

    if (g.kind == "uniform") {
        for (size_t i = 0; i < nNoise; ++i) {
            for (int j = 0; j < d; ++j) p[j] = uni(rng);
            ds.x.insert(ds.x.end(), p, p + d);
        }
        return ds;  // no ground truth
    }
    for (size_t i = 0; i < nNoise; ++i) {
        for (int j = 0; j < d; ++j) p[j] = uni(rng);
        ds.x.insert(ds.x.end(), p, p + d);
        ds.truth.push_back(-1);
    }

    // Shuffle so that the insertion order carries no cluster information.
    const size_t n = ds.size();
    for (size_t i = n - 1; i > 0; --i) {
        size_t j = std::uniform_int_distribution<size_t>(0, i)(rng);
        if (i == j) continue;
        for (int c = 0; c < d; ++c) std::swap(ds.x[i * d + c], ds.x[j * d + c]);
        std::swap(ds.truth[i], ds.truth[j]);
    }
    return ds;
}

Dataset parseCsv(const std::string& text, bool lastColumnIsLabel, const std::string& name) {
    Dataset ds;
    ds.name = name;
    ds.d = 0;
    std::map<std::string, int32_t> labelIds;
    std::vector<double> row;
    std::string lastTok;
    size_t pos = 0, lineNo = 0;
    while (pos < text.size()) {
        size_t eol = text.find('\n', pos);
        if (eol == std::string::npos) eol = text.size();
        std::string line = text.substr(pos, eol - pos);
        pos = eol + 1;
        ++lineNo;
        row.clear();
        lastTok.clear();
        bool ok = true;
        size_t i = 0;
        auto isSep = [](char c) { return c == ',' || c == ';' || c == '\t' || c == ' ' || c == '\r'; };
        std::vector<std::string> toks;
        while (i < line.size()) {
            while (i < line.size() && isSep(line[i])) ++i;
            size_t j = i;
            while (j < line.size() && !isSep(line[j])) ++j;
            if (j > i) toks.push_back(line.substr(i, j - i));
            i = j;
        }
        if (toks.empty()) continue;
        size_t numeric = toks.size() - (lastColumnIsLabel ? 1 : 0);
        for (size_t t = 0; t < numeric; ++t) {
            char* end = nullptr;
            double v = std::strtod(toks[t].c_str(), &end);
            if (end == toks[t].c_str() || *end != '\0' || !std::isfinite(v)) { ok = false; break; }
            row.push_back(v);
        }
        if (!ok) {
            if (ds.size() == 0 && ds.d == 0) continue;  // header line
            throw std::runtime_error("CSV: non-numeric value in line " + std::to_string(lineNo));
        }
        if (row.size() < 2) throw std::runtime_error("CSV: need at least two numeric columns");
        int cols = static_cast<int>(std::min<size_t>(row.size(), MAX_DIM));
        if (ds.d == 0) ds.d = cols;
        if (cols != ds.d)
            throw std::runtime_error("CSV: inconsistent column count in line " + std::to_string(lineNo));
        ds.x.insert(ds.x.end(), row.begin(), row.begin() + cols);
        if (lastColumnIsLabel) {
            auto it = labelIds.find(toks.back());
            if (it == labelIds.end())
                it = labelIds.emplace(toks.back(), static_cast<int32_t>(labelIds.size())).first;
            ds.truth.push_back(it->second);
        }
    }
    if (ds.size() == 0) throw std::runtime_error("CSV: no data rows found");
    return ds;
}

}  // namespace tc
