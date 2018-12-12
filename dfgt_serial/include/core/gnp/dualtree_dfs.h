/** @file dualtree_dfs.h
 *
 *  A template generator for performing a depth first search dual-tree
 *  algorithm.
 *
 *  @author Dongryeol Lee (dongryel@cc.gatech.edu)
 */

#ifndef CORE_GNP_DUALTREE_DFS_H
#define CORE_GNP_DUALTREE_DFS_H

#include <boost/tuple/tuple.hpp>
#include <map>
#include "core/math/range.h"

namespace core {
namespace gnp {

/** @brief The dualtree algorithm template generator.
 */
template<typename ProblemType>
class DualtreeDfs {

  public:

    /** @brief The table type.
     */
    typedef typename ProblemType::TableType TableType;

    /** @brief The tree type.
     */
    typedef typename TableType::TreeType TreeType;

    /** @brief Global constants type for the problem.
     */
    typedef typename ProblemType::GlobalType GlobalType;

    /** @brief The type of result computed by the engine.
     */
    typedef typename ProblemType::ResultType ResultType;

  private:

    /** @brief The number of deterministic prunes.
     */
    int num_deterministic_prunes_;

    /** @brief The number of probabilistic prunes.
     */
    int num_probabilistic_prunes_;

    /** @brief The pointer to the problem.
     */
    ProblemType *problem_;

    /** @brief The starting query node.
     */
    TreeType *query_start_node_;

    /** @brief The query table.
     */
    TableType *query_table_;

    /** @brief Starting reference node.
     */
    TreeType *reference_start_node_;

    /** @brief The reference table.
     */
    TableType *reference_table_;

  private:

    /** @brief Performs the base case for a given node pair.
     */
    template<typename MetricType>
    void DualtreeBase_(
      const MetricType &metric,
      TreeType *qnode,
      TreeType *rnode,
      bool qnode_and_rnode_are_equal,
      ResultType *result);

    /** @brief Determines whether a pair of query/reference pair can
     *         be pruned deterministically.
     */
    bool CanSummarize_(
      TreeType *qnode,
      TreeType *rnode,
      bool qnode_and_rnode_are_equal,
      typename ProblemType::DeltaType &delta,
      const core::math::Range &squared_distance_range,
      typename ProblemType::ResultType *query_results);

    /** @brief Summarize a given pair of query/reference using a
     *         deterministic approximation.
     */
    void Summarize_(
      TreeType *qnode, TreeType *rnode,
      const typename ProblemType::DeltaType &delta,
      typename ProblemType::ResultType *query_results);

    template<typename MetricType>
    bool CanProbabilisticSummarize_(
      const MetricType &metric,
      TreeType *qnode,
      TreeType *rnode,
      bool qnode_and_rnode_are_equal,
      double failure_probability,
      typename ProblemType::DeltaType &delta,
      const core::math::Range &squared_distance_range,
      typename ProblemType::ResultType *query_results);

    /** @brief Employ a probabilistic summarization with the given
     *         probability level.
     */
    template<typename MetricType>
    void ProbabilisticSummarize_(
      const MetricType &metric,
      GlobalType &global,
      TreeType *qnode,
      double failure_probability,
      const typename ProblemType::DeltaType &delta,
      typename ProblemType::ResultType *query_results);

    /** @brief The canonical recursive case for dualtree depth-first
     *         algorithm.
     */
    template<typename MetricType>
    bool DualtreeCanonical_(
      const MetricType &metric,
      TreeType *qnode,
      TreeType *rnode,
      double failure_probability,
      const core::math::Range &squared_distance_range,
      ResultType *query_results);

    /** @brief Postprocess unaccounted contributions.
     */
    template<typename MetricType>
    void PostProcess_(
      const MetricType &metric,
      TreeType *qnode, ResultType *query_results,
      bool do_query_results_postprocess);

  public:

    /** @brief The heuristic for choosing one node over the other.
     */
    template<typename MetricType>
    static void Heuristic(
      const MetricType &metric,
      TreeType *node,
      TreeType *first_candidate,
      TreeType *second_candidate,
      TreeType **first_partner,
      core::math::Range &first_squared_distance_range,
      TreeType **second_partner,
      core::math::Range &second_squared_distance_range);

    static void PreProcess(
      const GlobalType &global,
      TableType *query_table_in,
      TreeType *qnode,
      typename ProblemType::ResultType *query_results,
      unsigned long int initial_pruned_in);

    /** @brief Preprocesses the reference tree.
     */
    static void PreProcessReferenceTree(
      GlobalType &global_in, TreeType *rnode);

    /** @brief The constructor.
     */
    DualtreeDfs();

    TreeType *query_start_node() const;

    /** @brief Sets the starting query node for the dual-tree
     *         computation.
     */
    void set_query_start_node(TreeType *query_start_node_in);

    TreeType *reference_start_node() const;

    /** @brief Sets the starting reference node for the dual-tree
     *         computation.
     */
    void set_reference_start_node(TreeType *reference_start_node_in);

    /** @brief Returns the number of deterministic prunes so far.
     */
    int num_deterministic_prunes() const;

    /** @brief Returns the number of probabilistic prunes so far.
     */
    int num_probabilistic_prunes() const;

    /** @brief Returns the pointer to the problem spec.
     */
    ProblemType *problem();

    /** @brief Returns the query table.
     */
    TableType *query_table();

    /** @brief Returns the reference table.
     */
    TableType *reference_table();

    /** @brief Initializes the dual-tree engine with a problem spec.
     */
    void Init(ProblemType &problem_in);

    /** @brief Computes the result.
     */
    template<typename MetricType>
    void Compute(
      const MetricType &metric,
      typename ProblemType::ResultType *query_results,
      bool do_initializations = true);
};
}
}

#endif
