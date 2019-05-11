
#ifndef TEST_ID_HPP_
#define TEST_ID_HPP_

#include "id.hpp"
//#include "gaussian_kernel.hpp"
#include <cmath>
#include <cstring>
#include "kernel_inputs.hpp"

namespace askit {

double reconstruct_mat(int num_rows, int num_cols, int k, 
  const double* A, std::vector<lapack_int>& skeleton, std::vector<double>& proj);

double compute_error(int num_rows, int num_cols, int k,
    std::vector<double>& A_nocol, std::vector<double>& A_approx, const double* A);

double test_id_error(const double* A, int num_rows, int num_cols, int rank);

double test_adaptive_id(const double* A, int num_rows, int num_cols, double epsilon, int max_rank);

}

#endif
