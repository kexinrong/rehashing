/** @file lmetric.h
 *
 *  An implementation of general L_p metric.
 *
 *  @author Dongryeol Lee (dongryel@cc.gatech.edu)
 */

#ifndef CORE_METRIC_KERNELS_LMETRIC_H
#define CORE_METRIC_KERNELS_LMETRIC_H

#include <armadillo>
#include <boost/serialization/serialization.hpp>
#include "core/table/dense_point.h"
#include "core/math/math_lib.h"

namespace core {
namespace metric_kernels {

/** @brief A trait class for computing a squared distance.
 */
template<int t_pow>
class LMetricDistanceSqTrait {
  public:
    template<typename LMetricType>
    static double Compute(
      const LMetricType &metric_in,
      const arma::vec &a, const arma::vec &b) {
      return core::math::Pow<2, t_pow>(metric_in.DistanceIneq(a, b));
    }
};

/** @brief Template specialization for computing a squared distance
 *         under L2 metric, which avoids a square root operation.
 */
template<>
class LMetricDistanceSqTrait<2> {
  public:
    template<typename LMetricType>
    static double Compute(
      const LMetricType &metric_in,
      const arma::vec &a, const arma::vec &b) {
      return metric_in.DistanceIneq(a, b);
    }
};

/** @brief An L_p metric for vector spaces.
 *
 * A generic Metric class should simply compute the distance between
 * two points.  An LMetric operates for integer powers on arma::vec spaces.
 */
template<int t_pow>
class LMetric {

  private:

    // For boost serialization.
    friend class boost::serialization::access;

  public:

    template<class Archive>
    void serialize(Archive &ar, const unsigned int version) {
    }

    /** @brief Returns the identifier.
     */
    std::string name() const {
      return std::string("lmetric");
    }

    /** @brief Sets the scaling factor for each dimension. This does
     *         not do anything.
     */
    template<typename TableType>
    void set_scales(const TableType &scales_in) {
    }

    /** @brief Computes the distance metric between two points.
     */
    double Distance(
      const arma::vec &a, const arma::vec &b) const {
      return core::math::Pow<1, t_pow>(DistanceIneq(a, b));
    }

    double DistanceIneq(
      const arma::vec &a, const arma::vec &b) const {
      double distance_ineq = 0;
      int length = a.n_elem;
      for(int i = 0; i < length; i++) {
        distance_ineq += core::math::Pow<t_pow, 1>(a[i] - b[i]);
      }
      return distance_ineq;
    }

    /** @brief Computes the distance metric between two points, raised
     *         to a particular power.
     *
     * This might be faster so that you could get, for instance, squared
     * L2 distance.
     */
    double DistanceSq(
      const arma::vec &a, const arma::vec &b) const {

      return core::metric_kernels::LMetricDistanceSqTrait<t_pow>::Compute(
               *this, a, b);
    }
};
};
};

#endif
