/**
 From https://github.com/ClementLF/rlapjv
 
 */

#include <cassert>
#include <cstdio>
#include <limits>
#include <memory>

#ifdef __GNUC__
#define always_inline __attribute__((always_inline)) inline
#define restrict __restrict__
#endif

template <typename idx, typename cost>
always_inline std::tuple<cost, cost, idx, idx>
find_umins(
    idx dim, idx i, const cost *restrict assign_cost,
    const cost *restrict v)
{
    const cost *local_cost = &assign_cost[i * dim];
    cost umin = local_cost[0] - v[0];
    idx j1 = 0;
    idx j2 = -1;
    cost usubmin = std::numeric_limits<cost>::max();
    for (idx j = 1; j < dim; j++)
    {
        cost h = local_cost[j] - v[j];
        if (h < usubmin)
        {
            if (h >= umin)
            {
                usubmin = h;
                j2 = j;
            }
            else
            {
                usubmin = umin;
                umin = h;
                j2 = j1;
                j1 = j;
            }
        }
    }
    return std::make_tuple(umin, usubmin, j1, j2);
}


/// @brief Jonker-Volgenant algorithm.
/// @param dim0 in problem size
/// @param dim1 out problem size
/// @param assign_cost in cost matrix
/// @param verbose in indicates whether to report the progress to stdout
/// @param rowsol out column assigned to row in solution / size dim
/// @param colsol out row assigned to column in solution / size dim
/// @param u out dual variables, row reduction numbers / size dim
/// @param v out dual variables, column reduction numbers / size dim
/// @return achieved minimum assignment cost
template <typename idx, typename cost>
cost rlap(int dim0, int dim1,
         const cost *restrict assign_cost, bool verbose,
         idx *restrict rowsol, idx *restrict colsol,
         cost *restrict u, cost *restrict v)
{
    auto free = std::unique_ptr<idx[]>(new idx[dim0]);    // list of unassigned rows.
    auto collist = std::unique_ptr<idx[]>(new idx[dim1]); // list of columns to be scanned in various ways.
    auto d = std::unique_ptr<cost[]>(new cost[dim1]);     // 'cost-distance' in augmenting path calculation.
    auto pred = std::unique_ptr<idx[]>(new idx[dim1]);    // row-predecessor of column in augmenting/alternating path.
    idx numfree = 0;

#if _OPENMP >= 201307
#pragma omp simd
#endif


    for (idx i = 0; i < dim0; i++)
    {
        free[i] = i;
        rowsol[i] = -1;
        numfree++;
    }


    for (idx j = 0; j < dim1; j++)
    {
        colsol[j] = -1;
    }

    
    // AUGMENTING ROW REDUCTION
    for (int loopcnt = 0; loopcnt < 2; loopcnt++)
    { // loop to be done twice.
        // scan all free rows.
        // in some cases, a free row may be replaced with another one to be scanned next.
        idx k = 0;
        idx prevnumfree = numfree;
        numfree = 0; // start list of rows still free after augmenting row reduction.
        while (k < prevnumfree)
        {
            idx i = free[k++];

            // find minimum and second minimum reduced cost over columns.
            cost umin, usubmin;
            idx j1, j2;
            std::tie(umin, usubmin, j1, j2) = find_umins(dim1, i, assign_cost, v);

            idx i0 = colsol[j1];
            cost vj1_new = v[j1] - (usubmin - umin);
            bool vj1_lowers = vj1_new < v[j1]; // the trick to eliminate the epsilon bug
            if (vj1_lowers)
            {
                // change the reduction of the minimum column to increase the minimum
                // reduced cost in the row to the subminimum.
                v[j1] = vj1_new;
            }
            else if (i0 >= 0)
            { // minimum and subminimum equal.
                // minimum column j1 is assigned.
                // swap columns j1 and j2, as j2 may be unassigned.
                j1 = j2;
                i0 = colsol[j2];
            }

            // (re-)assign i to j1, possibly de-assigning an i0.
            rowsol[i] = j1;
            colsol[j1] = i;

            if (i0 >= 0)
            { // minimum column j1 assigned earlier.
                if (vj1_lowers)
                {
                    // put in current k, and go back to that k.
                    // continue augmenting path i - j1 with i0.
                    free[--k] = i0;
                }
                else
                {
                    // no further augmenting reduction possible.
                    // store i0 in list of free rows for next phase.
                    free[numfree++] = i0;
                }
            }
        }
        if (verbose)
        {
            printf("lapjv: AUGMENTING ROW REDUCTION %d / %d\n", loopcnt + 1, 2);
        }
    } // for loopcnt

    // AUGMENT SOLUTION for each free row.
    for (idx f = 0; f < numfree; f++)
    {
        idx endofpath;
        idx freerow = free[f]; // start row of augmenting path.
        if (verbose)
        {
            printf("lapjv: AUGMENT SOLUTION row %d [%d / %d]\n",
                   freerow, f + 1, numfree);
        }

// Dijkstra shortest path algorithm.
// runs until unassigned column added to shortest path tree.
#if _OPENMP >= 201307
#pragma omp simd
#endif
        for (idx j = 0; j < dim1; j++)
        {
            d[j] = assign_cost[freerow * dim1 + j] - v[j];
            pred[j] = freerow;
            collist[j] = j; // init column list.
        }

        idx low = 0; // columns in 0..low-1 are ready, now none.
        idx up = 0;  // columns in low..up-1 are to be scanned for current minimum, now none.
                     // columns in up..dim-1 are to be considered later to find new minimum,
                     // at this stage the list simply contains all columns
        bool unassigned_found = false;
        // initialized in the first iteration: low == up == 0
        idx last = 0;
        cost min = 0;
        do
        {
            if (up == low)
            { // no more columns to be scanned for current minimum.
                last = low - 1;
                // scan columns for up..dim-1 to find all indices for which new minimum occurs.
                // store these indices between low..up-1 (increasing up).
                min = d[collist[up++]];
                for (idx k = up; k < dim1; k++)
                {
                    idx j = collist[k];
                    cost h = d[j];
                    if (h <= min)
                    {
                        if (h < min)
                        {             // new minimum.
                            up = low; // restart list at index low.
                            min = h;
                        }
                        // new index with same minimum, put on undex up, and extend list.
                        collist[k] = collist[up];
                        collist[up++] = j;
                    }
                }

                // check if any of the minimum columns happens to be unassigned.
                // if so, we have an augmenting path right away.
                for (idx k = low; k < up; k++)
                {
                    if (colsol[collist[k]] < 0)
                    {
                        endofpath = collist[k];
                        unassigned_found = true;
                        break;
                    }
                }
            }

            if (!unassigned_found)
            {
                // update 'distances' between freerow and all unscanned columns, via next scanned column.
                idx j1 = collist[low];
                low++;
                idx i = colsol[j1];
                const cost *local_cost = &assign_cost[i * dim1];
                cost h = local_cost[j1] - v[j1] - min;
                for (idx k = up; k < dim1; k++)
                {
                    idx j = collist[k];
                    cost v2 = local_cost[j] - v[j] - h;
                    if (v2 < d[j])
                    {
                        pred[j] = i;
                        if (v2 == min)
                        { // new column found at same minimum value
                            if (colsol[j] < 0)
                            {
                                // if unassigned, shortest augmenting path is complete.
                                endofpath = j;
                                unassigned_found = true;
                                break;
                            }
                            else
                            { // else add to list to be scanned right away.
                                collist[k] = collist[up];
                                collist[up++] = j;
                            }
                        }
                        d[j] = v2;
                    }
                }
            }
        } while (!unassigned_found);

// update column prices.
#if _OPENMP >= 201307
#pragma omp simd
#endif
        for (idx k = 0; k <= last; k++)
        {
            idx j1 = collist[k];
            v[j1] = v[j1] + d[j1] - min;
        }

        // reset row and column assignments along the alternating path.
        {
            idx i;
            do
            {
                i = pred[endofpath];
                colsol[endofpath] = i;
                idx j1 = endofpath;
                endofpath = rowsol[i];
                rowsol[i] = j1;
            } while (i != freerow);
        }
    }
    if (verbose)
    {
        printf("lapjv: AUGMENT SOLUTION finished\n");
    }

    // calculate optimal cost.
    cost rlapcost = 0;
#if _OPENMP >= 201307
#pragma omp simd reduction(+ \
                           : rlapcost)
#endif

    for (idx i = 0; i < dim0; i++)
    {
        const cost *local_cost = &assign_cost[i * dim1];
        idx j = rowsol[i];
        u[i] = local_cost[j] - v[j];
        rlapcost += local_cost[j];
    }
    if (verbose)
    {
        printf("rlapjv: optimal cost calculated\n");
    }

    return rlapcost;
}
