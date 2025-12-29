#include "mtx_to_csr.h"
#include <vector>
#include <algorithm>
#include <cassert>
#include <cmath>

constexpr double MIN_DENSITY = 0.5;
constexpr double GAMMA = 1.5; // Modify to bias block score
constexpr int MIN_AREA = 2500;

struct BestSubmatrix {
    int r0 = -1, r1 = -1;
    int c0 = -1, c1 = -1;
    int area = 0;
    double density = 0.0;
    double score = 0.0;
    
    BestSubmatrix() = default;
    BestSubmatrix(int r0_, int r1_, int c0_, int c1_, int area_, double density_)
        : r0(r0_), r1(r1_), c0(c0_), c1(c1_), area(area_), density(density_) {}
};

template<typename T>
int row_nnz(
    const SpMatrixCompressed<T>& csr,
    int r,
    int c0,
    int c1
) {
    int cnt = 0;
    for (int k = csr.indptr[r]; k < csr.indptr[r + 1]; ++k) {
        int c = csr.indices[k];
        if (c >= c0 && c < c1)
            cnt++;
    }
    return cnt;
}

template<typename T>
int col_nnz(
    const SpMatrixCompressed<T>& csc,
    int c,
    int r0,
    int r1
) {
    int cnt = 0;
    for (int k = csc.indptr[c]; k < csc.indptr[c + 1]; ++k) {
        int r = csc.indices[k];
        if (r >= r0 && r < r1)
            cnt++;
    }
    return cnt;
}

struct Region {
    int r0, r1;
    int c0, c1;

    int height() const { return r1 - r0; }
    int width()  const { return c1 - c0; }
    int area()   const { return height() * width(); }
};

// Remove trailing rows and columns that do not improve density
template<typename T>
void trim_block(
    BestSubmatrix& b,
    const SpMatrixCompressed<T>& csr,
    const SpMatrixCompressed<T>& csc
) {
    bool improved = true;

    while (improved) {
        improved = false;

        int height = b.r1 - b.r0;
        int width  = b.c1 - b.c0;
        int area   = height * width;
        double cur_density = b.density;

        struct Candidate {
            char type; // 'T','B','L','R'
            double density;
        };

        Candidate best = {'X', cur_density};

        auto try_remove = [&](char t, int nnz_removed) {
            int new_area = area - (t == 'T' || t == 'B' ? width : height);
            double new_density =
                double(b.density * area - nnz_removed) / new_area;

            if (new_density > best.density) {
                best = {t, new_density};
            }
        };

        try_remove('T', row_nnz(csr, b.r0,     b.c0, b.c1));
        try_remove('B', row_nnz(csr, b.r1 - 1, b.c0, b.c1));
        try_remove('L', col_nnz(csc, b.c0,     b.r0, b.r1));
        try_remove('R', col_nnz(csc, b.c1 - 1, b.r0, b.r1));

        if (best.type != 'X') {
            improved = true;
            switch (best.type) {
                case 'T': b.r0++; break;
                case 'B': b.r1--; break;
                case 'L': b.c0++; break;
                case 'R': b.c1--; break;
            }
            b.density = best.density;
            b.area = (b.r1 - b.r0) * (b.c1 - b.c0);
        }
    }
}

inline double block_score(const BestSubmatrix& b) {
    if (b.area <= 0 || b.density <= MIN_DENSITY)
        return -1.0;  // invalid / reject

    return b.area * std::pow(b.density - MIN_DENSITY, GAMMA);
}

inline void merge_best(BestSubmatrix& dst, const BestSubmatrix& src) {
    if (src.score > dst.score) {
        dst = src;
    }
}

// Find blocks where row or col is >= MIN_SPAN
template<typename T>
void dense_pass_generic(
    const SpMatrixCompressed<T>& mat,
    const Region& region,
    BestSubmatrix& global_best
) {
    constexpr int MIN_SPAN = MIN_AREA; // If MIN_SPAN == MIN_AREA, then we don't need to special case vectors

    const bool row_major = (mat.direction == CompressionDirection::ROW);

    const int OUTER_BEGIN = row_major ? region.r0 : region.c0;
    const int OUTER_END   = row_major ? region.r1 : region.c1;

    const int INNER_BEGIN = row_major ? region.c0 : region.r0;
    const int INNER_END   = row_major ? region.c1 : region.r1;

    const int OUTER_LEN = OUTER_END - OUTER_BEGIN;
    const int INNER_LEN = INNER_END - INNER_BEGIN;

    if (OUTER_LEN < MIN_SPAN)
        return;

    #pragma omp parallel
    {
        std::vector<int> hist(INNER_LEN, 0);
        BestSubmatrix local_best;
        local_best.score = -1.0;

        #pragma omp for schedule(dynamic,1)
        for (int o0 = OUTER_BEGIN; o0 <= OUTER_END - MIN_SPAN; ++o0) {
            std::fill(hist.begin(), hist.end(), 0);

            for (int o1 = o0; o1 < OUTER_END; ++o1) {
                // Add outer slice
                for (int k = mat.indptr[o1]; k < mat.indptr[o1 + 1]; ++k) {
                    int idx = mat.indices[k];
                    if (idx >= INNER_BEGIN && idx < INNER_END) {
                        hist[idx - INNER_BEGIN]++;
                    }
                }

                int span = o1 - o0 + 1;
                if (span < MIN_SPAN) continue;

                int nnz = 0;
                int i0 = 0;

                for (int i1 = 0; i1 < INNER_LEN; ++i1) {
                    nnz += hist[i1];

                    while (i0 <= i1) {
                        int width = i1 - i0 + 1;
                        int area  = span * width;
                        double density = double(nnz) / area;

                        double candidate_score = area * std::pow(density - MIN_DENSITY, GAMMA);
                        if (candidate_score <= local_best.score)
                            break;

                        if (density >= MIN_DENSITY) {
                            if (row_major) {
                                local_best = {
                                    o0, o1 + 1,
                                    INNER_BEGIN + i0,
                                    INNER_BEGIN + i1 + 1,
                                    area,
                                    density
                                };
                                local_best.score = block_score(local_best);
                            } else {
                                local_best = {
                                    INNER_BEGIN + i0,
                                    INNER_BEGIN + i1 + 1,
                                    o0, o1 + 1,
                                    area,
                                    density
                                };
                                local_best.score = block_score(local_best);
                            }
                            break;
                        }

                        nnz -= hist[i0++];
                    }
                }
            }
        }

        #pragma omp critical
        merge_best(global_best, local_best);
    }
}

// When both row and col <= MIN_SPAN, we can still find blocks whose area is >= MIN_AREA
template<typename T>
void general_rectangle_pass(
    const SpMatrixCompressed<T>& mat,
    const Region& region,
    BestSubmatrix& global_best
) {

    const bool row_major = (mat.direction == CompressionDirection::ROW);

    const int OUTER_BEGIN = row_major ? region.r0 : region.c0;
    const int OUTER_END   = row_major ? region.r1 : region.c1;

    const int INNER_BEGIN = row_major ? region.c0 : region.r0;
    const int INNER_END   = row_major ? region.c1 : region.r1;

    const int INNER_LEN = INNER_END - INNER_BEGIN;

    #pragma omp parallel
    {
        std::vector<int> hist(INNER_LEN, 0);
        BestSubmatrix local_best;
        local_best.score = -1.0;

        #pragma omp for schedule(dynamic,1)
        for (int o0 = OUTER_BEGIN; o0 < OUTER_END; ++o0) {
            std::fill(hist.begin(), hist.end(), 0);

            for (int o1 = o0; o1 < OUTER_END; ++o1) {
                for (int k = mat.indptr[o1]; k < mat.indptr[o1 + 1]; ++k) {
                    int idx = mat.indices[k];
                    if (idx >= INNER_BEGIN && idx < INNER_END) {
                        hist[idx - INNER_BEGIN]++;
                    }
                }

                int height = o1 - o0 + 1;
                if (height * INNER_LEN < MIN_AREA)
                    continue;

                int nnz = 0;
                int i0 = 0;

                for (int i1 = 0; i1 < INNER_LEN; ++i1) {
                    nnz += hist[i1];

                    while (i0 <= i1) {
                        int width = i1 - i0 + 1;
                        int area  = height * width;

                        if (area < MIN_AREA)
                            break;
                        if (area <= local_best.area)
                            break;

                        double density = double(nnz) / area;
                        if (density >= MIN_DENSITY) {
                            if (row_major) {
                                local_best = {
                                    o0, o1 + 1,
                                    INNER_BEGIN + i0,
                                    INNER_BEGIN + i1 + 1,
                                    area,
                                    density
                                };
                                local_best.score = block_score(local_best);
                            } else {
                                local_best = {
                                    INNER_BEGIN + i0,
                                    INNER_BEGIN + i1 + 1,
                                    o0, o1 + 1,
                                    area,
                                    density
                                };
                            }
                            local_best.score = block_score(local_best);
                            break;
                        }

                        nnz -= hist[i0++];
                    }
                }
            }
        }

        #pragma omp critical
        merge_best(global_best, local_best);
    }
}


template<typename T>
void find_best_in_region(
    const SpMatrixCompressed<T>& csr,
    const SpMatrixCompressed<T>& csc,
    const Region& region,
    BestSubmatrix& best
) {
    if (region.area() < MIN_AREA)
        return;
    best = BestSubmatrix{};

    dense_pass_generic(csr, region, best);
    dense_pass_generic(csc, region, best);

    if (best.area == 0) {
        general_rectangle_pass(csr, region, best);
    }

    // Sanity
    if (best.area > 0) {
        assert(best.r0 >= region.r0 && best.r1 <= region.r1);
        assert(best.c0 >= region.c0 && best.c1 <= region.c1);
    }
}

template<typename T>
void decompose_region(
    const SpMatrixCompressed<T>& csr,
    const SpMatrixCompressed<T>& csc,
    const Region& region,
    std::vector<BestSubmatrix>& result,
    std::vector<bool>& row_used,
    std::vector<bool>& col_used
) {
    if (region.area() < MIN_AREA)
        return;

    BestSubmatrix best;
    find_best_in_region(csr, csc, region, best);

    if (best.area < MIN_AREA)
        return;

    // ---- Strong exclusion guard ----
    for (int r = best.r0; r < best.r1; ++r)
        if (row_used[r]) return;

    for (int c = best.c0; c < best.c1; ++c)
        if (col_used[c]) return;

    trim_block(best, csr, csc);
    best.score = block_score(best);

    if (best.area < MIN_AREA || best.density < MIN_DENSITY)
        return;

    // ---- Accept block ----
    result.push_back(best);

    for (int r = best.r0; r < best.r1; ++r)
        row_used[r] = true;

    for (int c = best.c0; c < best.c1; ++c)
        col_used[c] = true;

    // ---- Recurse (safe even if regions overlap) ----
    if (region.r0 < best.r0)
        decompose_region(csr, csc,
            {region.r0, best.r0, region.c0, region.c1},
            result, row_used, col_used);

    if (best.r1 < region.r1)
        decompose_region(csr, csc,
            {best.r1, region.r1, region.c0, region.c1},
            result, row_used, col_used);

    if (region.c0 < best.c0)
        decompose_region(csr, csc,
            {best.r0, best.r1, region.c0, best.c0},
            result, row_used, col_used);

    if (best.c1 < region.c1)
        decompose_region(csr, csc,
            {best.r0, best.r1, best.c1, region.c1},
            result, row_used, col_used);
}
