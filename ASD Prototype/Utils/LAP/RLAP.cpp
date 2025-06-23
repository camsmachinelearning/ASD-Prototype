//
//  lapjv.cpp
//  ASD Prototype
//
//  Created by Benjamin Lee on 6/17/25.
//

#include <memory>
#include <cmath>
#include "RLAP.h"
#include "CrouseSAP.h"


int solve_rlapf(int numrows, int numcols,
                const float* cost, bool maximize,
                int* rowsol, int* colsol) noexcept {
    return solve_rectangular_linear_assignment<int, float>(numrows,
                                                           numcols,
                                                           cost,
                                                           maximize,
                                                           rowsol,
                                                           colsol);
}

int solve_rlap(int numrows, int numcols,
                const double* cost, bool maximize,
                int* rowsol, int* colsol) noexcept {
    return solve_rectangular_linear_assignment<int, double>(numrows,
                                                            numcols,
                                                            cost,
                                                            maximize,
                                                            rowsol,
                                                            colsol);
}
