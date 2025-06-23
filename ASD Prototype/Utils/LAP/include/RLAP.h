#pragma once

#ifdef __cplusplus
extern "C" {
#endif

int solve_rlapf(int numrows, int numcols,
               const float* cost, bool maximize,
               int* rowsol, int* colsol) noexcept;

int solve_rlap(int numrows, int numcols,
              const double* cost, bool maximize,
              int* rowsol, int* colsol) noexcept;

#ifdef __cplusplus
}
#endif
