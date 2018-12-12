/** @file gen_metric.h
 *
 *  The generic metric tree builder.
 *
 *  @author Dongryeol Lee (dongryel@cc.gatech.edu)
 */

#ifndef CORE_TREE_GEN_METRIC_TREE_H
#define CORE_TREE_GEN_METRIC_TREE_H

#include <armadillo>
#include <vector>
#include <boost/serialization/serialization.hpp>
#include <boost/tuple/tuple.hpp>

#include "core/math/math_lib.h"
#include "core/tree/ball_bound.h"
#include "core/tree/general_spacetree.h"
#include "core/table/dense_matrix.h"
#include "core/table/memory_mapped_file.h"

namespace boost {
namespace serialization {

template <class Archive, typename MetricType>
void serialize(
  Archive & ar,
  ::boost::tuple< MetricType, ::core::tree::BallBound, int > &c,
  const unsigned int version) {
  ar & ::boost::tuples::get<0>(c);
  ar & ::boost::tuples::get<1>(c);
  ar & ::boost::tuples::get<2>(c);
}

template <class Archive, typename T>
void serialize(
  Archive & ar, std::pair<arma::vec, T> &c, const unsigned int version) {
  ar & c.first;
  ar & c.second;
}
}
}

namespace core {
namespace tree {

/** @brief The general metric tree specification.
 */
template<typename IncomingStatisticType>
class GenMetricTree {
  public:

    typedef core::tree::BallBound BoundType;

    typedef IncomingStatisticType StatisticType;

  private:

    /** @brief Computes the furthest point from the given pivot and
     *         finds out the index.
     */
    template<typename MetricType>
    static int FurthestColumnIndex_(
      const MetricType &metric_in,
      const arma::vec &pivot,
      const arma::mat &matrix,
      int begin, int count,
      double *furthest_distance) {

      int furthest_index = -1;
      int end = begin + count;
      *furthest_distance = -1.0;

      {
        // Local variables used for reduction.
        int local_furthest_index = -1;
        double local_furthest_distance = -1.0;

        for(int i = begin; i < end; i++) {
          arma::vec point;
          core::table::MakeColumnVector(matrix, i, &point);
          double distance_between_center_and_point =
            metric_in.Distance(pivot, point);

          if(local_furthest_distance < distance_between_center_and_point) {
            local_furthest_distance = distance_between_center_and_point;
            local_furthest_index = i;
          }
        } // end of for-loop.

        // Reduction.
        {
          if(local_furthest_distance > (*furthest_distance)) {
            *furthest_distance = local_furthest_distance;
            furthest_index = local_furthest_index;
          }
        }
      }
      return furthest_index;
    }

  public:

    template<typename MetricType>
    static void FindBoundFromMatrix(
      const MetricType &metric_in,
      const arma::mat &matrix,
      int first, int count, BoundType *bounds) {

      MakeLeafNode(metric_in, matrix, first, count, bounds);
    }

    /** @brief Makes a leaf node in the metric tree.
     */
    template<typename MetricType>
    static void MakeLeafNode(
      const MetricType &metric_in,
      const arma::mat &matrix,
      int begin, int count, BoundType *bounds) {

      // Clear the bound to zero.
      bounds->center().zeros();

      int end = begin + count;
      arma::vec &bound_ref = bounds->center();

      {
        arma::vec local_sum;
        local_sum.zeros(bound_ref.n_elem);

        for(int i = begin; i < end; i++) {
          arma::vec col_point;
          core::table::MakeColumnVector(matrix, i, &col_point);
          local_sum += col_point;
        }

        // Final reduction.
        {
          bound_ref += local_sum;
        }
      }

      // Divide by the number of points.
      bound_ref = (1.0 / static_cast<double>(count)) * bound_ref;

      double furthest_distance;
      FurthestColumnIndex_(
        metric_in, bounds->center(), matrix, begin, count, &furthest_distance);
      bounds->set_radius(furthest_distance);
    }

    template<typename MetricType, typename TreeType>
    static void CombineBounds(
      const MetricType &metric_in,
      arma::mat &matrix,
      TreeType *node, TreeType *left, TreeType *right) {

      // Compute the weighted sum of the two pivots
      arma::vec &bound_ref = node->bound().center();
      arma::vec &left_bound_ref = left->bound().center();
      arma::vec &right_bound_ref = right->bound().center();
      bound_ref = left->count() * left_bound_ref +
                  right->count() * right_bound_ref;
      bound_ref =
        (1.0 / static_cast<double>(node->count())) * bound_ref;

      double left_max_dist, right_max_dist;
      FurthestColumnIndex_(
        metric_in, node->bound().center(), matrix, left->begin(),
        left->count(), &left_max_dist);
      FurthestColumnIndex_(
        metric_in, node->bound().center(), matrix, right->begin(),
        right->count(), &right_max_dist);
      node->bound().set_radius(std::max(left_max_dist, right_max_dist));
    }

    template<typename MetricType>
    static void ComputeMemberships(
      const MetricType &metric_in,
      const arma::mat &matrix,
      int first, int end,
      BoundType &left_bound, BoundType &right_bound,
      int *left_count, std::deque<bool> *left_membership,
      bool favor_balance = false) {

      left_membership->resize(end - first);
      *left_count = 0;

      {
        int local_left_count = 0;

        for(int left = first; left < end; left++) {

          // Make alias of the current point.
          arma::vec point;
          core::table::MakeColumnVector(matrix, left, &point);

          // Compute the distances from the two pivots.
          double distance_from_left_pivot =
            metric_in.Distance(point, left_bound.center());
          double distance_from_right_pivot =
            metric_in.Distance(point, right_bound.center());

          // We swap if the point is further away from the left pivot.
          if(distance_from_left_pivot > distance_from_right_pivot) {
            (*left_membership)[left - first] = false;
          }
          else {
            (*left_membership)[left - first] = true;
            local_left_count++;
          }
        }

        // Final reduction.
        {
          (*left_count) += local_left_count;
        }
      }

      if(favor_balance) {
        int count = end - first;
        if((*left_count) <= count / 4) {

          // Steal from the right.
          for(int i = 0; i < count; i++) {
            if(!((*left_membership)[i])) {
              (*left_count)++;
              (*left_membership)[i] = true;
            }
            int current_right_count = count - (*left_count);
            if(abs((*left_count) - current_right_count) < count / 8) {
              break;
            }
          }
        }
        else if((*left_count) >= count / 4 * 3) {

          // Steal from the left.
          for(int i = 0; i < count; i++) {
            if((*left_membership)[i]) {
              (*left_count)--;
              (*left_membership)[i] = false;
            }
            int current_right_count = count - (*left_count);
            if(abs((*left_count) - current_right_count) < count / 8) {
              break;
            }
          }
        }
      }
    }

    template<typename MetricType, typename TreeType, typename IndexType>
    static bool AttemptSplitting(
      const MetricType &metric_in,
      arma::mat &matrix,
      arma::mat &weights,
      TreeType *node, TreeType **left,
      TreeType **right, int leaf_size, IndexType *old_from_new,
      core::table::MemoryMappedFile *m_file_in) {

      // Pick a random row.
      int random_row = core::math::RandInt(
                         node->begin(), node->begin() + node->count());
      arma::vec random_row_vec;
      core::table::MakeColumnVector(matrix, random_row, & random_row_vec);

      // Now figure out the furthest point from the random row picked
      // above.
      double furthest_distance;
      int furthest_from_random_row =
        FurthestColumnIndex_(
          metric_in, random_row_vec, matrix, node->begin(), node->count(),
          &furthest_distance);
      arma::vec furthest_from_random_row_vec;
      core::table::MakeColumnVector(
        matrix, furthest_from_random_row, &furthest_from_random_row_vec);

      // Then figure out the furthest point from the furthest point.
      double furthest_from_furthest_distance;
      int furthest_from_furthest_random_row =
        FurthestColumnIndex_(
          metric_in, furthest_from_random_row_vec, matrix, node->begin(),
          node->count(), &furthest_from_furthest_distance);
      arma::vec furthest_from_furthest_random_row_vec;
      core::table::MakeColumnVector(
        matrix, furthest_from_furthest_random_row,
        &furthest_from_furthest_random_row_vec);

      // Allocate the left and the right.
      *left = (m_file_in) ?
              m_file_in->Construct<TreeType>() : new TreeType();
      *right = (m_file_in) ?
               m_file_in->Construct<TreeType>() : new TreeType();
      (*left)->bound().center() = furthest_from_random_row_vec;
      (*right)->bound().center() = furthest_from_furthest_random_row_vec;
      int left_count = 0;
      if(furthest_from_furthest_distance <
          std::numeric_limits<double>::epsilon()) {
        return false;
      }
      else {
        left_count = TreeType::MatrixPartition(
                       metric_in, matrix, weights, node->begin(), node->count(),
                       (*left)->bound(), (*right)->bound(), old_from_new);
      }
      (*left)->Init(node->begin(), left_count);
      (*right)->Init(node->begin() + left_count, node->count() - left_count);
      return true;
    }
};
}
}

#endif
