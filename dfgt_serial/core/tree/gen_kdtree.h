/** @file tree/kdtree.h
 *
 *  The generic kd-tree builder.
 *
 *  @author Dongryeol Lee (dongryel@cc.gatech.edu)
 */

#ifndef CORE_TREE_GEN_KDTREE_H
#define CORE_TREE_GEN_KDTREE_H

#include "core/tree/general_spacetree.h"
#include "core/tree/hrect_bound.h"

namespace core {
namespace tree {

/** @brief The generic midpoint splitting specification for kd-tree.
 */
class GenKdTreeMidpointSplitter {
  public:

    /** @brief Computes the widest dimension and its width of a
     *         bounding box.
     */
    template<typename BoundType>
    static void ComputeWidestDimension(
      const BoundType &bound, int *split_dim, double *max_width) {

      *split_dim = -1;
      *max_width = -1.0;
      for(int d = 0; d < bound.dim(); d++) {
        double w = bound.get(d).width();
        if(w > *max_width) {
          *max_width = w;
          *split_dim = d;
        }
      }
    }

    /** @brief The splitter that simply returns the mid point of the
     *         splitting dimension.
     */
    template<typename BoundType>
    static double ChooseKdTreeSplitValue(
      const BoundType &bound, int split_dim) {
      return bound.get(split_dim).mid();
    }
};

/** @brief The specification of the kd-tree.
 */
template< typename IncomingStatisticType >
class GenKdTree {

  public:

    /** @brief The bounding primitive used in kd-tree.
     */
    typedef core::tree::HrectBound BoundType;

    /** @brief The statistics type used in the tree.
     */
    typedef IncomingStatisticType StatisticType;

  public:

    /** @brief Computes two bounding primitives and membership vectors
     *         for a given consecutive column points in the data
     *         matrix.
     */
    template<typename MetricType>
    static void ComputeMemberships(
      const MetricType &metric_in,
      const arma::mat &matrix,
      int first, int end,
      BoundType &left_bound, BoundType &right_bound,
      int *left_count, std::deque<bool> *left_membership) {

      // Get the split dimension and the split value.
      int split_dim = static_cast<int>(left_bound.get(0).lo);
      double split_val = left_bound.get(0).hi;

      // Reset the left bound and the right bound.
      left_bound.Reset();
      right_bound.Reset();
      *left_count = 0;
      left_membership->resize(end - first);

      // Build the bounds for the kd-tree.
      {
        // The local accumulants.
        int local_left_count = 0;
        BoundType local_left_bound;
        BoundType local_right_bound;
        local_left_bound.Init(left_bound.dim());
        local_right_bound.Init(right_bound.dim());

        for(int left = first; left < end; left++) {

          // Make alias of the current point.
          arma::vec point;
          core::table::MakeColumnVector(matrix, left, &point);

          // We swap if the point is further away from the left pivot.
          if(point[split_dim] > split_val) {
            (*left_membership)[left - first] = false;
            local_right_bound |= point;
          }
          else {
            (*left_membership)[left - first] = true;
            local_left_bound |= point;
            local_left_count++;
          }
        } // end of for-loop.

        // Final reduction.
        {
          (*left_count) += local_left_count;
          left_bound |= local_left_bound;
          right_bound |= local_right_bound;
        }
      }
    }

    template<typename MetricType>
    static void FindBoundFromMatrix(
      const MetricType &metric_in,
      const arma::mat &matrix,
      int first, int count, BoundType *bounds) {

      int end = first + count;
      {
        // Local variable for accumulating the bound information for a
        // thread.
        BoundType local_bound;
        local_bound.Init(bounds->dim());

        for(int i = first; i < end; i++) {
          arma::vec col;
          core::table::MakeColumnVector(matrix, i, &col);
          local_bound |= col;
        }

        // The final reduction.
        {
          (*bounds) |= local_bound;
        }
      }
    }

    /** @brief Makes a leaf node by constructing its bound.
     */
    template<typename MetricType>
    static void MakeLeafNode(
      const MetricType &metric_in,
      const arma::mat& matrix,
      int begin, int count, BoundType *bounds) {

      FindBoundFromMatrix(metric_in, matrix, begin, count, bounds);
    }

    /** @brief Combines the bounding primitives of the children node
     *         to form the bound for the self.
     */
    template<typename MetricType, typename TreeType>
    static void CombineBounds(
      const MetricType &metric_in,
      arma::mat &matrix,
      TreeType *node, TreeType *left, TreeType *right) {

      // Do nothing.
    }

    /** @brief Attempts to split a kd-tree node and reshuffles the
     *         data accordingly and creates two child nodes.
     */
    template<typename MetricType, typename TreeType, typename IndexType>
    static bool AttemptSplitting(
      const MetricType &metric_in,
      arma::mat &matrix,
      arma::mat &weights,
      TreeType *node, TreeType **left,
      TreeType **right, int leaf_size,
      IndexType *old_from_new,
      core::table::MemoryMappedFile *m_file_in) {

      int split_dim = -1;
      double max_width = -1;

      // Find the splitting dimension.
      core::tree::GenKdTreeMidpointSplitter::ComputeWidestDimension(
        node->bound(), &split_dim, &max_width);

      // Choose the split value along the dimension to be splitted.
      double split_val =
        core::tree::GenKdTreeMidpointSplitter::ChooseKdTreeSplitValue(
          node->bound(), split_dim);

      // Allocate the children and its bound.
      *left = (m_file_in) ?
              m_file_in->Construct<TreeType>() : new TreeType();
      *right = (m_file_in) ?
               m_file_in->Construct<TreeType>() : new TreeType();
      (*left)->bound().Init(matrix.n_rows);
      (*right)->bound().Init(matrix.n_rows);

      int left_count = 0;
      if(max_width < std::numeric_limits<double>::epsilon()) {
        return false;
      }
      else {

        // Copy the split dimension and split value.
        (*left)->bound().get(0).lo = split_dim;
        (*left)->bound().get(0).hi = split_val;

        left_count = TreeType::MatrixPartition(
                       metric_in, matrix, weights, node->begin(), node->count(),
                       (*left)->bound(), (*right)->bound(), old_from_new);
      }
      (*left)->Init(node->begin(), left_count);
      (*right)->Init(
        node->begin() + left_count, node->count() - left_count);

      return true;
    }
};
}
}

#endif
