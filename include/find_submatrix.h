#ifndef FIND_SUBMATRIX_H
#define FIND_SUBMATRIX_H

#include "mtx_to_csr.h"
#include <vector>
#include <algorithm>

struct BestSubmatrix {
    int r0 = -1, r1 = -1;
    int c0 = -1, c1 = -1;
    int area = 0;
    
    BestSubmatrix() = default;
    BestSubmatrix(int r0_, int r1_, int c0_, int c1_, int area_)
        : r0(r0_), r1(r1_), c0(c0_), c1(c1_), area(area_) {}
};

inline void merge_best(BestSubmatrix& dst, const BestSubmatrix& src) {
    if (src.area > dst.area) {
        dst = src;
    }
}

template<typename T>
void dense_pass_generic(
    const SpMatrixCompressed<T>& mat,
    BestSubmatrix& global_best
) {
    const bool is_row_compressed = (mat.direction == CompressionDirection::ROW);
    const int OUTER = is_row_compressed ? mat.num_rows : mat.num_cols;
    const int INNER = is_row_compressed ? mat.num_cols : mat.num_rows;
    const int MIN_SPAN = 2500;

    #pragma omp parallel
    {
        BestSubmatrix local_best;
        std::vector<int> hist(INNER);

        #pragma omp for schedule(dynamic,1)
        for (int o0 = 0; o0 <= OUTER - MIN_SPAN; ++o0) {
            std::fill(hist.begin(), hist.end(), 0);

            for (int o1 = o0; o1 < OUTER; ++o1) {
                // Add slice o1
                for (int k = mat.indptr[o1]; k < mat.indptr[o1 + 1]; ++k) {
                    hist[mat.indices[k]]++;
                }

                int span = o1 - o0 + 1;
                if (span < MIN_SPAN) continue;

                // Sliding window on inner dimension
                int nnz = 0;
                int i0 = 0;

                for (int i1 = 0; i1 < INNER; ++i1) {
                    nnz += hist[i1];

                    while (i0 <= i1) {
                        int width  = i1 - i0 + 1;
                        int height = span;

                        int area = is_row_compressed
                            ? height * width
                            : width * height;

                        if (area <= local_best.area) break;

                        double density = double(nnz) / area;
                        if (density > 0.5) {
                            if (is_row_compressed) {
                                local_best = BestSubmatrix{o0, o1 + 1, i0, i1 + 1, area};
                            } else {
                                local_best = BestSubmatrix{i0, i1 + 1, o0, o1 + 1, area};
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

#endif // FIND_SUBMATRIX_H