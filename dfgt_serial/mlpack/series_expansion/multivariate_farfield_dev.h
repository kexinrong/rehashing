/** @file multivariate_farfield_dev.h
 *
 *  This file contains an implementation of $O(D^p)$ expansion for
 *  computing the coefficients for a far-field expansion for an
 *  arbitrary kernel function. This is a template specialization of
 *  CartesianFarField class.
 *
 *  @author Dongryeol Lee (dongryel@cc.gatech.edu)
 */

#ifndef MLPACK_SERIES_EXPANSION_MULTIVARIATE_FARFIELD_DEV_H
#define MLPACK_SERIES_EXPANSION_MULTIVARIATE_FARFIELD_DEV_H

#include "mlpack/series_expansion/cartesian_expansion_global.h"
#include "mlpack/series_expansion/cartesian_farfield.h"

namespace mlpack {
namespace series_expansion {

template<>
template<typename KernelAuxType, typename TreeIteratorType>
void CartesianFarField <
mlpack::series_expansion::MULTIVARIATE >::AccumulateCoeffs(
  const KernelAuxType &kernel_aux_in,
  TreeIteratorType &it, int order) {

  int dim = kernel_aux_in.global().get_dimension();
  int total_num_coeffs = kernel_aux_in.global().get_total_num_coeffs(order);
  arma::vec tmp;
  int r, i, j, k, t, tail;
  std::vector<short int> heads(dim + 1, 0);
  arma::vec x_r;
  double bandwidth_factor =
    kernel_aux_in.BandwidthFactor(kernel_aux_in.kernel().bandwidth_sq());

  // initialize temporary variables
  tmp.set_size(total_num_coeffs);
  x_r.set_size(dim);
  arma::vec pos_coeffs;
  arma::vec neg_coeffs;
  pos_coeffs.zeros(total_num_coeffs);
  neg_coeffs.zeros(total_num_coeffs);

  // set to new order if greater
  if(order_ < order) {
    order_ = order;
  }
  arma::vec C_k;

  // Repeat for each reference point in this reference node.
  arma::vec point;
  arma::vec weight;
  while(it.HasNext()) {
    it.Next(&point, &weight);

    // Calculate the coordinate difference between the ref point and
    // the centroid.
    for(i = 0; i < dim; i++) {
      x_r[i] = (point[i] - center_[i]) / bandwidth_factor;
    }

    // initialize heads
    heads[dim] = std::numeric_limits<short int>::max();

    tmp[0] = 1.0;

    for(k = 1, t = 1, tail = 1; k <= order; k++, tail = t) {
      for(i = 0; i < dim; i++) {
        short int head = heads[i];
        heads[i] = t;

        for(j = head; j < tail; j++, t++) {
          tmp[t] = tmp[j] * x_r[i];
        }
      }
    }

    // Tally up the result in A_k.
    for(i = 0; i < total_num_coeffs; i++) {

      // Replace with the following case for non-uniform weights.
      double prod = weight[0] * tmp[i];
      if(prod > 0) {
        pos_coeffs[i] += prod;
      }
      else {
        neg_coeffs[i] += prod;
      }
    }

  } // End of looping through each reference point

  // get multiindex factors
  core::table::Alias(
    kernel_aux_in.global().get_inv_multiindex_factorials(), &C_k);

  for(r = 0; r < total_num_coeffs; r++) {
    coeffs_[r] += (pos_coeffs[r] + neg_coeffs[r]) * C_k[r];
  }
}

template<>
template<typename KernelAuxType>
double CartesianFarField<mlpack::series_expansion::MULTIVARIATE>::EvaluateField(
  const KernelAuxType &kernel_aux_in,
  const arma::vec &x_q, int order) const {

  // dimension
  int dim = kernel_aux_in.global().get_dimension();

  // total number of coefficients
  int total_num_coeffs = kernel_aux_in.global().get_total_num_coeffs(order);

  // square root times bandwidth
  double bandwidth_factor =
    kernel_aux_in.BandwidthFactor(kernel_aux_in.kernel().bandwidth_sq());

  // the evaluated sum
  double pos_multipole_sum = 0;
  double neg_multipole_sum = 0;
  double multipole_sum = 0;

  // computed derivative map
  arma::mat derivative_map;
  kernel_aux_in.AllocateDerivativeMap(dim, order, &derivative_map);

  // temporary variable
  arma::vec arrtmp;
  arrtmp.set_size(total_num_coeffs);

  // (x_q - x_R) scaled by bandwidth
  arma::vec x_q_minus_x_R;
  x_q_minus_x_R.set_size(dim);

  // compute (x_q - x_R) / (sqrt(2h^2))
  for(int d = 0; d < dim; d++) {
    x_q_minus_x_R[d] = (x_q[d] - center_[d]) / bandwidth_factor;
  }

  // compute deriative maps based on coordinate difference.
  kernel_aux_in.ComputeDirectionalDerivatives(
    x_q_minus_x_R, &derivative_map, order);

  // compute h_{\alpha}((x_q - x_R)/sqrt(2h^2)) ((x_r - x_R)/h)^{\alpha}
  for(int j = 0; j < total_num_coeffs; j++) {
    const std::vector<short int> &mapping =
      kernel_aux_in.global().get_multiindex(j);
    double arrtmp =
      kernel_aux_in.ComputePartialDerivative(derivative_map, mapping);
    double prod = coeffs_[j] * arrtmp;

    if(prod > 0) {
      pos_multipole_sum += prod;
    }
    else {
      neg_multipole_sum += prod;
    }
  }

  multipole_sum = pos_multipole_sum + neg_multipole_sum;
  return multipole_sum;
}

template<>
template<typename KernelAuxType>
void CartesianFarField<mlpack::series_expansion::MULTIVARIATE>::Print(
  const KernelAuxType &kernel_aux_in, const char *name, FILE *stream) const {

  int dim = kernel_aux_in.global().get_dimension();
  int total_num_coeffs = kernel_aux_in.global().get_total_num_coeffs(order_);

  fprintf(stream, "----- SERIESEXPANSION %s ------\n", name);
  fprintf(stream, "Far field expansion\n");
  fprintf(stream, "Center: ");

  for(unsigned int i = 0; i < center_.n_elem; i++) {
    fprintf(stream, "%g ", center_[i]);
  }
  fprintf(stream, "\n");

  fprintf(stream, "f(");
  for(int d = 0; d < dim; d++) {
    fprintf(stream, "x_q%d", d);
    if(d < dim - 1)
      fprintf(stream, ",");
  }
  fprintf(stream, ") = \\sum\\limits_{x_r \\in R} K(||x_q - x_r||) = ");

  for(int i = 0; i < total_num_coeffs; i++) {
    const std::vector<short int> &mapping =
      kernel_aux_in.global().get_multiindex(i);
    fprintf(stream, "%g ", coeffs_[i]);

    fprintf(stream, "(-1)^(");
    for(int d = 0; d < dim; d++) {
      fprintf(stream, "%d", mapping[d]);
      if(d < dim - 1)
        fprintf(stream, " + ");
    }
    fprintf(stream, ") D^((");
    for(int d = 0; d < dim; d++) {
      fprintf(stream, "%d", mapping[d]);

      if(d < dim - 1)
        fprintf(stream, ",");
    }
    fprintf(stream, ")) f(x_q - x_R)");
    if(i < total_num_coeffs - 1) {
      fprintf(stream, " + ");
    }
  }
  fprintf(stream, "\n");
}

template<>
template<typename KernelAuxType>
void CartesianFarField <
mlpack::series_expansion::MULTIVARIATE >::TranslateFromFarField(
  const KernelAuxType &kernel_aux_in, const CartesianFarField &se) {

  double bandwidth_factor =
    kernel_aux_in.BandwidthFactor(kernel_aux_in.kernel().bandwidth_sq());
  int dim = kernel_aux_in.global().get_dimension();
  int order = se.get_order();
  int total_num_coeffs = kernel_aux_in.global().get_total_num_coeffs(order);
  arma::vec prev_coeffs;
  arma::vec prev_center;
  const std::vector < std::vector<short int> > &multiindex_mapping =
    kernel_aux_in.global().get_multiindex_mapping();
  const std::vector < std::vector<short int> > &lower_mapping_index =
    kernel_aux_in.global().get_lower_mapping_index();

  std::vector <short int> tmp_storage;
  arma::vec center_diff;
  arma::vec inv_multiindex_factorials;

  center_diff.set_size(dim);

  // retrieve coefficients to be translated and helper mappings
  core::table::Alias(se.get_coeffs(), &prev_coeffs);
  core::table::Alias(se.get_center(), &prev_center);
  tmp_storage.resize(kernel_aux_in.global().get_dimension());
  core::table::Alias(
    kernel_aux_in.global().get_inv_multiindex_factorials(),
    &inv_multiindex_factorials);

  // no coefficients can be translated
  if(order == -1)
    return;
  else
    order_ = order;

  // compute center difference
  for(int j = 0; j < dim; j++) {
    center_diff[j] = prev_center[j] - center_[j];
  }

  for(int j = 0; j < total_num_coeffs; j++) {

    const std::vector <short int> &gamma_mapping = multiindex_mapping[j];
    const std::vector <short int> &lower_mappings_for_gamma =
      lower_mapping_index[j];
    double pos_coeff = 0;
    double neg_coeff = 0;

    for(unsigned int k = 0; k < lower_mappings_for_gamma.size(); k++) {

      const std::vector <short int> &inner_mapping =
        multiindex_mapping[lower_mappings_for_gamma[k]];

      int flag = 0;
      double diff1;

      // compute gamma minus alpha
      for(int l = 0; l < dim; l++) {
        tmp_storage[l] = gamma_mapping[l] - inner_mapping[l];

        if(tmp_storage[l] < 0) {
          flag = 1;
          break;
        }
      }

      if(flag) {
        continue;
      }

      diff1 = 1.0;

      for(int l = 0; l < dim; l++) {
        diff1 *= pow(center_diff[l] / bandwidth_factor, tmp_storage[l]);
      }

      double prod =
        prev_coeffs[lower_mappings_for_gamma[k]] * diff1 *
        inv_multiindex_factorials[
          kernel_aux_in.global().ComputeMultiindexPosition(tmp_storage)];

      if(prod > 0) {
        pos_coeff += prod;
      }
      else {
        neg_coeff += prod;
      }

    } // end of k-loop

    coeffs_[j] += pos_coeff + neg_coeff;

  } // end of j-loop
}

template<>
template<typename KernelAuxType, typename CartesianLocalType>
void CartesianFarField <
mlpack::series_expansion::MULTIVARIATE >::TranslateToLocal(
  const KernelAuxType &kernel_aux_in, int truncation_order,
  CartesianLocalType *se) const {

  arma::vec pos_arrtmp, neg_arrtmp;
  arma::mat derivative_map;
  kernel_aux_in.AllocateDerivativeMap(
    kernel_aux_in.global().get_dimension(), 2 * truncation_order,
    &derivative_map);
  arma::vec local_center;
  arma::vec cent_diff;
  arma::vec local_coeffs;
  int local_order = se->get_order();
  int dimension = kernel_aux_in.global().get_dimension();
  int total_num_coeffs =
    kernel_aux_in.global().get_total_num_coeffs(truncation_order);
  double bandwidth_factor =
    kernel_aux_in.BandwidthFactor(kernel_aux_in.kernel().bandwidth_sq());

  // get center and coefficients for local expansion
  core::table::Alias(se->get_center(), &local_center);
  core::table::Alias(se->get_coeffs(), &local_coeffs);
  cent_diff.set_size(dimension);

  // if the order of the far field expansion is greater than the
  // local one we are adding onto, then increase the order.
  if(local_order < truncation_order) {
    se->set_order(truncation_order);
  }

  // Compute derivatives.
  pos_arrtmp.set_size(total_num_coeffs);
  neg_arrtmp.set_size(total_num_coeffs);

  // Compute center difference divided by the bandwidth factor.
  for(int j = 0; j < dimension; j++) {
    cent_diff[j] = (local_center[j] - center_[j]) / bandwidth_factor;
  }

  // Compute required partial derivatives.
  kernel_aux_in.ComputeDirectionalDerivatives(cent_diff, &derivative_map,
      2 * truncation_order);
  std::vector<short int> beta_plus_alpha;
  beta_plus_alpha.resize(dimension);

  for(int j = 0; j < total_num_coeffs; j++) {

    const std::vector<short int> &beta_mapping =
      kernel_aux_in.global().get_multiindex(j);
    pos_arrtmp[j] = neg_arrtmp[j] = 0;

    for(int k = 0; k < total_num_coeffs; k++) {

      const std::vector<short int> &alpha_mapping =
        kernel_aux_in.global().get_multiindex(k);
      for(int d = 0; d < dimension; d++) {
        beta_plus_alpha[d] = beta_mapping[d] + alpha_mapping[d];
      }
      double derivative_factor =
        kernel_aux_in.ComputePartialDerivative(derivative_map, beta_plus_alpha);

      double prod = coeffs_[k] * derivative_factor;

      if(prod > 0) {
        pos_arrtmp[j] += prod;
      }
      else {
        neg_arrtmp[j] += prod;
      }
    } // end of k-loop
  } // end of j-loop

  const arma::vec &C_k_neg =
    kernel_aux_in.global().get_neg_inv_multiindex_factorials();
  for(int j = 0; j < total_num_coeffs; j++) {
    local_coeffs[j] += (pos_arrtmp[j] + neg_arrtmp[j]) * C_k_neg[j];
  }
}
}
}

#endif
