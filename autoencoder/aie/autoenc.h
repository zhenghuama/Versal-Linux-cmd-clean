
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

#endif
