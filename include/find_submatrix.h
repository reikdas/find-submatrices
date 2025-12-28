#include "mtx_to_csr.h"
#include <vector>
#include <algorithm>
#include <cassert>

struct BestSubmatrix {
    int r0 = -1, r1 = -1;
    int c0 = -1, c1 = -1;
    int area = 0;
    double density = 0.0;
    
    BestSubmatrix() = default;
    BestSubmatrix(int r0_, int r1_, int c0_, int c1_, int area_, double density_)
        : r0(r0_), r1(r1_), c0(c0_), c1(c1_), area(area_), density(density_) {}
};

struct Region {
    int r0, r1;
    int c0, c1;

    int height() const { return r1 - r0; }
    int width()  const { return c1 - c0; }
    int area()   const { return height() * width(); }
};

inline void merge_best(BestSubmatrix& dst, const BestSubmatrix& src) {
    if (src.area > dst.area) {
        dst = src;
    }
}

template<typename T>
void dense_pass_generic(
    const SpMatrixCompressed<T>& mat,
    const Region& region,
    BestSubmatrix& global_best
) {
    constexpr int MIN_SPAN = 2500;
    constexpr double MIN_DENSITY = 0.5;

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
                            } else {
                                local_best = {
                                    INNER_BEGIN + i0,
                                    INNER_BEGIN + i1 + 1,
                                    o0, o1 + 1,
                                    area,
                                    density
                                };
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


template<typename T>
void general_rectangle_pass(
    const SpMatrixCompressed<T>& mat,
    const Region& region,
    BestSubmatrix& global_best
) {
    constexpr int MIN_AREA = 2500;
    constexpr double MIN_DENSITY = 0.5;

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
                            } else {
                                local_best = {
                                    INNER_BEGIN + i0,
                                    INNER_BEGIN + i1 + 1,
                                    o0, o1 + 1,
                                    area,
                                    density
                                };
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


template<typename T>
void find_best_in_region(
    const SpMatrixCompressed<T>& csr,
    const SpMatrixCompressed<T>& csc,
    const Region& region,
    BestSubmatrix& best
) {
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
    constexpr int MIN_AREA = 2500;
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
