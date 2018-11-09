/** @file dense_matrix_inverse.h
 *
 *  @brief A utility for updating a dense matrix when its number of
 *         rows and number of columns grow by one.
 *
 *  @author Dongryeol Lee (dongryel@cc.gatech.edu)
 */

#ifndef MLPACK_SERIES_EXPANSION_DENSE_MATRIX_INVERSE_H
#define MLPACK_SERIES_EXPANSION_DENSE_MATRIX_INVERSE_H

#include <armadillo>
#include "core/table/dense_matrix.h"
#include "core/table/dense_point.h"

namespace mlpack {
namespace series_expansion {

/** @brief The class for updating a inverse of a kernel matrix.
 */
class DenseMatrixInverse {
  public:

    static arma::mat *Update(
      const arma::mat &previous_inverse,
      const arma::vec &inverse_times_new_column,
      double projection_error) {

      arma::mat *new_matrix_inverse =
        new arma::mat();
      new_matrix_inverse->set_size(
        previous_inverse.n_rows + 1, previous_inverse.n_cols + 1);

      for(unsigned int j = 0; j < previous_inverse.n_cols; j++) {
        for(unsigned int i = 0; i < previous_inverse.n_rows; i++) {
          new_matrix_inverse->at(i, j) =
            previous_inverse.at(i, j) +
            inverse_times_new_column[i] *
            inverse_times_new_column[j] / projection_error;
        }
      }

      for(unsigned int j = 0; j < previous_inverse.n_cols; j++) {
        new_matrix_inverse->at(j, previous_inverse.n_cols) =
          - inverse_times_new_column[j] / projection_error;
        new_matrix_inverse->at(previous_inverse.n_rows, j) =
          - inverse_times_new_column[j] / projection_error;
      }
      new_matrix_inverse->at(
        new_matrix_inverse->n_rows - 1,
        new_matrix_inverse->n_cols - 1) = 1.0 / projection_error;

      // Return the computed inverse.
      return new_matrix_inverse;
    }
};
}
}

#endif
