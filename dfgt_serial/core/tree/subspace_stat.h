/** @file subspace_stat.h
 *
 *
 *  @author Dongryeol Lee (dongryel@cc.gatech.edu)
 */

#ifndef CORE_TREE_SUBSPACE_STAT_H
#define CORE_TREE_SUBSPACE_STAT_H

#include <armadillo>

namespace core {
namespace tree {

class SubspaceStat {

  private:

    static const double epsilon_ = 0.2;

    static void ComputeResidualBasis_(
      const arma::mat &first_basis, const arma::mat &second_basis,
      const arma::vec &mean_diff, arma::mat *projection,
      arma::mat *residual_basis) {

      // First project second basis and the difference of the two means
      // onto the first basis...
      arma::mat projection_first_basis;
      arma::vec projection_mean_diff;
      projection->Init(first_basis.n_cols(), second_basis.n_cols() + 1);
      projection_first_basis.Alias(projection->GetColumnPtr(0),
                                   first_basis.n_cols(), second_basis.n_cols());
      projection_mean_diff.Alias(projection->GetColumnPtr(second_basis.n_cols()),
                                 projection->n_rows());
      la::MulTransAOverwrite(first_basis, second_basis, &projection_first_basis);
      la::MulOverwrite(mean_diff, first_basis, &projection_mean_diff);

      // Reconstruct and compute the reconstruction error...
      residual_basis->Init(second_basis.n_rows(), second_basis.n_cols() + 1);
      arma::mat residual_basis_second_basis;
      arma::vec residual_basis_mean_diff;
      residual_basis_second_basis.Alias(residual_basis->GetColumnPtr(0),
                                        first_basis.n_rows(),
                                        second_basis.n_cols());
      residual_basis_mean_diff.Alias(residual_basis->GetColumnPtr
                                     (second_basis.n_cols()),
                                     first_basis.n_rows());
      residual_basis_second_basis.CopyValues(second_basis);
      residual_basis_mean_diff.CopyValues(mean_diff);

      la::MulExpert(-1, false, first_basis, false, *projection, 1,
                    residual_basis);

      // Loop over each residual basis...
      for(index_t i = 0; i < residual_basis->n_cols(); i++) {

        double *column_vector = residual_basis->GetColumnPtr(i);

        for(index_t j = 0; j < i; j++) {
          double dot_product = la::Dot(residual_basis->n_rows(), column_vector,
                                       residual_basis->GetColumnPtr(j));
          la::AddExpert(residual_basis->n_rows(), -dot_product,
                        residual_basis->GetColumnPtr(j), column_vector);
        }

        // Normalize the vector if done...
        double length = la::LengthEuclidean(residual_basis->n_rows(),
                                            column_vector);

        if(length > epsilon_) {
          la::Scale(residual_basis->n_rows(), 1.0 / length, column_vector);
        }
        else {
          la::Scale(residual_basis->n_rows(), 0, column_vector);
        }
      } // end of looping over each residual basis...
    }

    static void ComputeColumnMean_(
      const arma::mat &data, index_t start,
      index_t count, arma::vec *mean) {

      mean->Init(data.n_rows());
      mean->SetZero();

      for(index_t i = start; i < start + count; i++) {
        arma::vec data_col;
        data.MakeColumnarma::vec(i, &data_col);
        la::AddTo(data_col, mean);
      }
      la::Scale(1.0 / ((double) count), mean);
    }

    static void ColumnMeanCenter_(
      const arma::mat &data, index_t start,
      index_t count, const arma::vec &mean, arma::mat *data_copy) {

      data_copy->Init(data.n_rows(), count);

      // Subtract the mean vector from each column of the matrix.
      for(index_t i = start; i < start + count; i++) {
        arma::vec data_copy_col, data_col;
        data_copy->MakeColumnarma::vec(i - start, &data_copy_col);
        data.MakeColumnarma::vec(i, &data_col);

        la::SubOverwrite(data_col, mean, &data_copy_col);
      }
    }

    static void ColumnMeanCenterTranspose_(
      const arma::mat &data, index_t start,
      index_t count, const arma::vec &mean, arma::mat *data_copy) {

      data_copy->Init(count, data.n_rows());

      // Subtract the mean vector from each column of the matrix.
      for(index_t i = start; i < start + count; i++) {
        for(index_t j = 0; j < data.n_rows(); j++) {
          data_copy->set(i - start, j, data.get(j, i) - mean[j]);
        }
      }
    }

    static void ComputeNormalizedCumulativeDistribution_(
      const arma::vec &src, arma::vec *dest, double *total) {
      *total = 0;
      dest->Init(src.length());
      (*dest)[0] = src[0];
      (*total) += src[0];

      for(index_t i = 1; i < src.length(); i++) {
        (*dest)[i] = (*dest)[i - 1] + src[i];
        (*total) += src[i];
      }
    }

    static index_t FindBinNumber_(
      const arma::vec &cumulative_distribution, double random_number) {

      for(index_t i = 0; i < cumulative_distribution.length(); i++) {
        if(random_number < cumulative_distribution[i]) {
          return i;
        }
      }
      return cumulative_distribution.length() - 1;
    }

    void FastSvdByColumnSampling_(
      const arma::mat& mean_centered, bool transposed,
      arma::vec &singular_values_arg,
      arma::mat &left_singular_vectors_arg,
      arma::mat &right_singular_vectors_arg) {

      // First determine the column length-squared distribution...
      arma::vec squared_lengths;
      squared_lengths.Init(mean_centered.n_cols());
      for(index_t i = 0; i < mean_centered.n_cols(); i++) {
        squared_lengths[i] =
          la::Dot(mean_centered.n_rows(), mean_centered.GetColumnPtr(i),
                  mean_centered.GetColumnPtr(i));
      }

      // Compute the normalized cumulative distribution on the squared lengths.
      arma::vec normalized_cumulative_squared_lengths;
      double total_squared_lengths;
      ComputeNormalizedCumulativeDistribution_
      (squared_lengths, &normalized_cumulative_squared_lengths,
       &total_squared_lengths);

      // The number of samples...
      int num_samples = std::max((int) sqrt(mean_centered.n_cols()), 1);
      arma::mat sampled_columns;
      sampled_columns.Init(mean_centered.n_rows(), num_samples);

      // Commence sampling...
      for(index_t s = 0; s < num_samples; s++) {

        // Generate random number between 0 and total_squared_lengths
        // and find out which column is picked.
        double random_number = math::Random(0, total_squared_lengths);
        index_t sample_number =
          FindBinNumber_(normalized_cumulative_squared_lengths, random_number);

        // Normalize proportion to squared length and the number of
        // samples.
        double probability =
          squared_lengths[sample_number] / total_squared_lengths;
        for(index_t j = 0; j < mean_centered.n_rows(); j++) {
          sampled_columns.set(j, s, mean_centered.get(j, sample_number) /
                              sqrt(num_samples * probability));
        }
      }

      // Let C = sampled columns. Then here we compute C^T C and
      // computes its eigenvector.
      arma::mat sampled_product, tmp_right_singular_vectors, tmp_vectors;
      arma::vec tmp_eigen_values;
      la::MulTransAInit(sampled_columns, sampled_columns, &sampled_product);
      la::SVDInit(sampled_product, &tmp_eigen_values,
                  &tmp_right_singular_vectors, &tmp_vectors);

      // Cut off small eigen values...
      int eigen_count = 0;
      for(index_t i = 0; i < tmp_eigen_values.length(); i++) {
        if(tmp_eigen_values[i] >= 0 &&
            tmp_eigen_values[i] >= epsilon_ * tmp_eigen_values[0]) {
          eigen_count++;
        }
      }
      arma::mat aliased_right_singular_vectors;
      aliased_right_singular_vectors.Alias
      (tmp_right_singular_vectors.GetColumnPtr(0),
       tmp_right_singular_vectors.n_rows(), eigen_count);

      if(transposed) {
        // Now exploit the relationship between the right and the left
        // singular vectors. Normalize and retrieve the singular values.
        la::MulInit(sampled_columns, aliased_right_singular_vectors,
                    &right_singular_vectors_arg);
        singular_values_arg.Init(eigen_count);
        for(index_t i = 0; i < eigen_count; i++) {
          singular_values_arg[i] = sqrt(tmp_eigen_values[i]);
          la::Scale(mean_centered.n_rows(), 1.0 / singular_values_arg[i],
                    right_singular_vectors_arg.GetColumnPtr(i));
        }

        // Now compute the right singular vectors from the left singular
        // vectors.
        la::MulTransAInit(mean_centered, right_singular_vectors_arg,
                          &left_singular_vectors_arg);
        for(index_t i = 0; i < left_singular_vectors_arg.n_cols(); i++) {
          double length =
            la::LengthEuclidean(left_singular_vectors_arg.n_rows(),
                                left_singular_vectors_arg.GetColumnPtr(i));
          la::Scale(left_singular_vectors_arg.n_rows(), 1.0 / length,
                    left_singular_vectors_arg.GetColumnPtr(i));
        }
      }
      else {
        // Now exploit the relationship between the right and the left
        // singular vectors. Normalize and retrieve the singular values.
        la::MulInit(sampled_columns, aliased_right_singular_vectors,
                    &left_singular_vectors_arg);
        singular_values_arg.Init(eigen_count);
        for(index_t i = 0; i < eigen_count; i++) {
          singular_values_arg[i] = sqrt(tmp_eigen_values[i]);
          la::Scale(mean_centered.n_rows(), 1.0 / singular_values_arg[i],
                    left_singular_vectors_arg.GetColumnPtr(i));
        }

        // Now compute the right singular vectors from the left singular
        // vectors.
        la::MulTransAInit(mean_centered, left_singular_vectors_arg,
                          &right_singular_vectors_arg);
        for(index_t i = 0; i < right_singular_vectors_arg.n_cols(); i++) {
          double length =
            la::LengthEuclidean(right_singular_vectors_arg.n_rows(),
                                right_singular_vectors_arg.GetColumnPtr(i));
          la::Scale(right_singular_vectors_arg.n_rows(), 1.0 / length,
                    right_singular_vectors_arg.GetColumnPtr(i));
        }
      }
    }

  public:

    // Member variables

    /** @brief The starting index of the points owned by the current
     *         statistics object.
     */
    int start_;

    /** @brief The number of points owned by the current statistics
     *         object.
     */
    int count_;

    /** @brief The mean of the points owned by the current statistics
     *         object.
     */
    arma::vec mean_vector_;

    /** @brief The left singular vectors that comprise the subspace for
     *         the current statistics object.
     */
    arma::mat left_singular_vectors_;

    /** @brief The singular values.
     */
    arma::vec singular_values_;

    /** @brief The maximum L2 reconstruction error of the points owned
     *         by the node using the left singular vectors as the basis
     *         set.
     */
    double max_l2_norm_reconstruction_error_;

    /** @brief Dimension-reduced data with repsect to the mean vector.
     */
    arma::mat dimension_reduced_data_;

    // Member functions

    void ComputeMaxL2NormReconstructionError(const arma::mat &dataset) {

      arma::vec diff_vector, reconstructed_vector;
      diff_vector.Init(dataset.n_rows());
      reconstructed_vector.Init(dataset.n_rows());
      dimension_reduced_data_.Init(left_singular_vectors_.n_cols(), count_);

      max_l2_norm_reconstruction_error_ = 0;
      for(index_t i = start_; i < start_ + count_; i++) {

        // Get the reference to the projected vector to be computed.
        arma::vec proj_vector;
        dimension_reduced_data_.MakeColumnarma::vec(i - start_, &proj_vector);

        // Compute the projection of each point and its reconstruction
        // error.
        la::SubOverwrite(dataset.n_rows(), mean_vector_.ptr(),
                         dataset.GetColumnPtr(i), diff_vector.ptr());
        la::MulOverwrite(diff_vector, left_singular_vectors_, &proj_vector);

        la::MulOverwrite(left_singular_vectors_, proj_vector,
                         &reconstructed_vector);

        la::SubFrom(diff_vector, &reconstructed_vector);
        max_l2_norm_reconstruction_error_ =
          std::max(max_l2_norm_reconstruction_error_,
                   la::Dot(reconstructed_vector, reconstructed_vector));
      }
    }

    /** @brief Compute PCA exhaustively for leaf nodes.
     */
    void Init(const arma::mat& dataset, index_t &start, index_t &count) {

      // Set the start and count info before anything else...
      start_ = start;
      count_ = count;

      if(count == 1) {
        mean_vector_.Init(dataset.n_rows());
        mean_vector_.CopyValues(dataset.GetColumnPtr(start));
        left_singular_vectors_.Init(dataset.n_rows(), 1);
        left_singular_vectors_.SetZero();
        singular_values_.Init(1);
        singular_values_.SetZero();
        ComputeMaxL2NormReconstructionError(dataset);
        return;
      }

      arma::mat mean_centered, tmp_left_singular_vectors;
      arma::vec tmp_singular_values;

      // Compute the mean vector owned by this node.
      ComputeColumnMean_(dataset, start, count, &mean_vector_);
      ColumnMeanCenter_(dataset, start, count, mean_vector_, &mean_centered);
      arma::mat right_singular_vectors;

      // Compute the SVD of the covariance matrix.
      la::SVDInit(mean_centered, &tmp_singular_values,
                  &tmp_left_singular_vectors, &right_singular_vectors);

      // Take square root of the returned values so that the proper
      // singular values can be computed.
      int subspace_count = 0;
      for(index_t i = 0; i < tmp_singular_values.length(); i++) {
        tmp_singular_values[i] = sqrt(tmp_singular_values[i]);
        if(tmp_singular_values[i] >= epsilon_ * tmp_singular_values[0]) {
          subspace_count++;
        }
      }
      singular_values_.Init(subspace_count);
      for(index_t i = 0; i < subspace_count; i++) {
        singular_values_[i] = tmp_singular_values[i];
      }
      left_singular_vectors_.Copy(tmp_left_singular_vectors.GetColumnPtr(0),
                                  tmp_left_singular_vectors.n_rows(),
                                  subspace_count);

      /*
      // Determine which dimension is longer: the row or the column...
      // If there are more columns than rows, then we do
      // column-sampling.
      if(dataset.n_rows() <= count) {

        // Compute the mean centered dataset.
        ColumnMeanCenter_(dataset, start, count, mean_vector_, &mean_centered);

        FastSvdByColumnSampling_(mean_centered, false, singular_values_,
      	       left_singular_vectors_,
      	       right_singular_vectors_tmp);
      }
      else {

        // Compute the mean centered dataset.
        ColumnMeanCenterTranspose_(dataset, start, count, mean_vector_,
      		 &mean_centered);

        FastSvdByColumnSampling_(mean_centered, true, singular_values_,
      	       left_singular_vectors_,
      	       right_singular_vectors_tmp);
      }
      */

      ComputeMaxL2NormReconstructionError(dataset);
    }

    /** @brief Merge two eigenspaces into one.
     */
    void Init(
      const arma::mat& dataset, index_t &start, index_t &count,
      const SubspaceStat& left_stat, const SubspaceStat& right_stat) {

      // Set the starting index and the count...
      start_ = start;
      count_ = count;

      // Compute the weighted mean of the two means...
      mean_vector_.Copy(left_stat.mean_vector_);
      la::Scale(left_stat.count_, &mean_vector_);
      la::AddExpert(right_stat.count_, right_stat.mean_vector_, &mean_vector_);
      la::Scale(1.0 / ((double) count), &mean_vector_);

      // Compute the difference between the two PCA models...
      arma::vec diff_mean;
      la::SubInit(left_stat.mean_vector_, right_stat.mean_vector_, &diff_mean);

      // Compute the residual of the projection of the right basis and
      // the mean difference onto the left basis.
      arma::mat subspace_projection_reconstruction_error, subspace_projection;
      ComputeResidualBasis_(left_stat.left_singular_vectors_,
                            right_stat.left_singular_vectors_, diff_mean,
                            &subspace_projection,
                            &subspace_projection_reconstruction_error);

      // Now we setup the eigenproblem to be solved for stitching two
      // PCA models together.
      arma::mat merging_problem;
      int dimension_merging_problem = left_stat.singular_values_.length() +
                                      right_stat.singular_values_.length();
      merging_problem.Init(dimension_merging_problem, dimension_merging_problem);
      merging_problem.SetZero();

      // Compute the multiplicative factors.
      double factor1 = ((double) left_stat.count_) / ((double) count);
      double factor2 = ((double) right_stat.count_) / ((double) count);
      double factor3 = ((double) left_stat.count_ * right_stat.count_) /
                       ((double) count * count);

      // Setup the top left block...using the outer-product formulation
      // of the matrix-matrix product.
      //
      // Remember that eigenvalues are squared singular values!!
      for(index_t j = 0; j < left_stat.singular_values_.length(); j++) {
        merging_problem.set(j, j, factor1 * left_stat.singular_values_[j] *
                            left_stat.singular_values_[j] /
                            ((double) left_stat.count_));
      }

      for(index_t j = 0; j < right_stat.singular_values_.length() + 1; j++) {
        const double *column_vector = subspace_projection.GetColumnPtr(j);

        // Now loop over each component of the upper left submatrix.
        for(index_t i = 0; i < left_stat.singular_values_.length(); i++) {
          for(index_t k = 0; k < left_stat.singular_values_.length(); k++) {

            if(j < right_stat.singular_values_.length()) {
              merging_problem.set(k, i, merging_problem.get(k, i) + factor2 *
                                  column_vector[k] * column_vector[i] *
                                  right_stat.singular_values_[j] *
                                  right_stat.singular_values_[j] /
                                  ((double) right_stat.count_));
            }
            else {
              merging_problem.set(k, i, merging_problem.get(k, i) + factor3 *
                                  column_vector[k] * column_vector[i]);
            }
          }
        }
      }

      // Compute the projection of the right basis and the mean
      // difference onto the residual basis.
      arma::mat projection_right_basis;
      la::MulTransAInit(subspace_projection_reconstruction_error,
                        right_stat.left_singular_vectors_,
                        &projection_right_basis);
      arma::vec projection_mean_diff;
      la::MulInit(diff_mean, subspace_projection_reconstruction_error,
                  &projection_mean_diff);

      // Set up the top right block...also using the outer-product
      // formulation of the matrix-matrix product.
      for(index_t j = 0; j < right_stat.singular_values_.length() + 1; j++) {
        const double *column_vector = subspace_projection.GetColumnPtr(j);
        const double *column_vector2 =
          (j < right_stat.singular_values_.length()) ?
          projection_right_basis.GetColumnPtr(j) : projection_mean_diff.ptr();

        // Now loop over each component of the upper right submatrix.
        for(index_t i = left_stat.singular_values_.length();
            i < dimension_merging_problem; i++) {
          for(index_t k = 0; k < left_stat.singular_values_.length(); k++) {

            if(j < right_stat.singular_values_.length()) {
              merging_problem.set
              (k, i, merging_problem.get(k, i) + factor2 *
               column_vector[k] *
               column_vector2[i - left_stat.singular_values_.length()] *
               right_stat.singular_values_[j] *
               right_stat.singular_values_[j] / ((double) right_stat.count_));
            }
            else {
              merging_problem.set
              (k, i, merging_problem.get(k, i) + factor3 *
               column_vector[k] *
               column_vector2[i - left_stat.singular_values_.length()]);
            }
          } // end of iterating over each row...
        } // end of iterating over each column...
      } // end of iterating over each column...

      // Set up the lower left block... This is basically a tranpose of
      // the upper right block.
      for(index_t i = 0; i < left_stat.singular_values_.length(); i++) {
        for(index_t j = left_stat.singular_values_.length();
            j < dimension_merging_problem; j++) {
          merging_problem.set(j, i, merging_problem.get(i, j));
        }
      }
      // Set up the lower right block... also using the outer-product
      // formulation of the matrix-matrix product.
      for(index_t j = 0; j < right_stat.singular_values_.length() + 1; j++) {
        const double *column_vector =
          (j < right_stat.singular_values_.length()) ?
          projection_right_basis.GetColumnPtr(j) : projection_mean_diff.ptr();

        // Now loop over each component of the upper left submatrix.
        for(index_t i = left_stat.singular_values_.length();
            i < dimension_merging_problem; i++) {
          for(index_t k = left_stat.singular_values_.length();
              k < dimension_merging_problem; k++) {

            if(j < right_stat.singular_values_.length()) {
              merging_problem.set
              (k, i, merging_problem.get(k, i) + factor2 *
               column_vector[k - left_stat.singular_values_.length()] *
               column_vector[i - left_stat.singular_values_.length()] *
               right_stat.singular_values_[j] *
               right_stat.singular_values_[j] / ((double) right_stat.count_));
            }
            else {
              merging_problem.set
              (k, i, merging_problem.get(k, i) + factor3 *
               column_vector[k - left_stat.singular_values_.length()] *
               column_vector[i - left_stat.singular_values_.length()]);
            }
          }
        }
      }

      // Compute the eigenvector of the system and rotate...
      arma::vec tmp_singular_values;
      arma::mat tmp_left_singular_vectors, tmp_right_singular_vectors;

      la::SVDInit(merging_problem, &tmp_singular_values,
                  &tmp_left_singular_vectors, &tmp_right_singular_vectors);
      int eigen_count = 0;
      for(index_t i = 0; i < tmp_singular_values.length(); i++) {
        if(tmp_singular_values[i] >= epsilon_ * tmp_singular_values[0]) {
          eigen_count++;
        }
      }

      // Rotation...
      left_singular_vectors_.Init(dataset.n_rows(), eigen_count);
      left_singular_vectors_.SetZero();

      for(index_t i = 0; i < tmp_left_singular_vectors.n_cols(); i++) {
        const double *column_vector =
          (i < left_stat.left_singular_vectors_.n_cols()) ?
          left_stat.left_singular_vectors_.GetColumnPtr(i) :
          subspace_projection_reconstruction_error.GetColumnPtr
          (i - left_stat.left_singular_vectors_.n_cols());

        for(index_t j = 0; j < eigen_count; j++) {
          for(index_t k = 0; k < dataset.n_rows(); k++) {
            left_singular_vectors_.set
            (k, j, left_singular_vectors_.get(k, j) +
             column_vector[k] * tmp_left_singular_vectors.get(i, j));
          }
        }
      }

      // Copy over the singular values...
      singular_values_.Init(eigen_count);
      for(index_t i = 0; i < eigen_count; i++) {
        singular_values_[i] = sqrt(tmp_singular_values[i] * count_);
      }

      ComputeMaxL2NormReconstructionError(dataset);
    }
};
}
}

#endif
