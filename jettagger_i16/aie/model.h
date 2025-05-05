
#ifndef AUTOENC_H
#define AUTOENC_H

#define DENSE_FN(IDX) void f##IDX(input_window_int16 * __restrict matA, output_window_int16 * __restrict matC);

DENSE_FN(0)
DENSE_FN(1)
DENSE_FN(2)
DENSE_FN(3)

#undef DENSE_FN

#endif
