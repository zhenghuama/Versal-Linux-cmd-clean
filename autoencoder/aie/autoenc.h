
#ifndef AUTOENC_H
#define AUTOENC_H

#define DENSE_FN(IDX) void f##IDX(input_window_int8 * __restrict matA, output_window_int8 * __restrict matC);

DENSE_FN(0)
DENSE_FN(1)
DENSE_FN(2)
DENSE_FN(3)
DENSE_FN(4)
DENSE_FN(5)
DENSE_FN(6)
DENSE_FN(7)
DENSE_FN(8)

#undef DENSE_FN

#define PARALLEL_MMUL(IDX, PART) void p##IDX##PART(input_window_int8 * __restrict matA, output_window_int8 * __restrict matC);


PARALLEL_MMUL(0,0)
PARALLEL_MMUL(1,0)
PARALLEL_MMUL(2,0)
// PARALLEL_MMUL(3,0)
// PARALLEL_MMUL(4,0)
// PARALLEL_MMUL(5,0)
PARALLEL_MMUL(6,0)
PARALLEL_MMUL(7,0)
PARALLEL_MMUL(8,0)

PARALLEL_MMUL(0,1)
PARALLEL_MMUL(1,1)
PARALLEL_MMUL(2,1)
// PARALLEL_MMUL(3,1)
// PARALLEL_MMUL(4,1)
// PARALLEL_MMUL(5,1)
PARALLEL_MMUL(6,1)
PARALLEL_MMUL(7,1)
PARALLEL_MMUL(8,1)

#undef PARALLEL_MMUL


#define SUM(IDX) void s##IDX(input_window_int8 * __restrict matA, \
    input_window_int8 * __restrict matB, \
    output_window_int8 * __restrict matC);


// SUM(0)
// SUM(1)
// SUM(2)
// SUM(3)
// SUM(4)
// SUM(5)
// SUM(6)
// SUM(7)
SUM(8)

#undef SUM

#define PARALLEL_SUM(IDX, PART) void s##IDX##PART(input_window_int8 * __restrict matA, \
    input_window_int8 * __restrict matB, \
    output_window_int8 * __restrict matC);

PARALLEL_SUM(0,0)
PARALLEL_SUM(1,0)
PARALLEL_SUM(2,0)
// PARALLEL_SUM(3,0)
// PARALLEL_SUM(4,0)
// PARALLEL_SUM(5,0)
PARALLEL_SUM(6,0)
PARALLEL_SUM(7,0)
PARALLEL_SUM(8,0)

PARALLEL_SUM(0,1)
PARALLEL_SUM(1,1)
PARALLEL_SUM(2,1)
// PARALLEL_SUM(3,1)
// PARALLEL_SUM(4,1)
// PARALLEL_SUM(5,1)
PARALLEL_SUM(6,1)
PARALLEL_SUM(7,1)
PARALLEL_SUM(8,1)

#undef PARALLEL_SUM


#endif
