#pragma once

#ifdef __cplusplus
extern "C" {
#endif

float solve_lapf(int inputDims, int outputDims, const float* cost, int* solutionRows, int* solutionCols) noexcept;
double solve_lap(int inputDims, int outputDims, const double* cost, int* solutionRows, int* solutionCols) noexcept;

#ifdef __cplusplus
}
#endif
