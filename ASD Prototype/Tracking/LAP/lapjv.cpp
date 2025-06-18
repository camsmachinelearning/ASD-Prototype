//
//  lapjv.cpp
//  ASD Prototype
//
//  Created by Benjamin Lee on 6/17/25.
//

#include <memory>
#include "lapjv.h"
#include "lapjv-core.h"


float solve_lapf(int inputDims, int outputDims, const float* cost, int* solutionRows, int* solutionCols) noexcept {
    std::unique_ptr<float[]> u(new float[inputDims]);
    std::unique_ptr<float[]> v(new float[outputDims]);

    return rlap<int, float>(
        inputDims,
        outputDims,
        cost,
        false,
        solutionRows,
        solutionCols,
        u.get(),
        v.get()
    );
}

double solve_lap(int inputDims, int outputDims, const double* cost, int* solutionRows, int* solutionCols) noexcept {
    std::unique_ptr<double[]> u(new double[inputDims]);
    std::unique_ptr<double[]> v(new double[outputDims]);

    return rlap<int, double>(
        inputDims,
        outputDims,
        cost,
        false,
        solutionRows,
        solutionCols,
        u.get(),
        v.get()
    );
}
