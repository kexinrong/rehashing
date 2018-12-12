/** @file kde_result.h
 *
 *  The computed results for kde dual-tree algorithm.
 *
 *  @author Dongryeol Lee (dongryel@cc.gatech.edu)
 */

#ifndef MLPACK_KDE_KDE_RESULT_H
#define MLPACK_KDE_KDE_RESULT_H

namespace mlpack {
namespace kde {

/** @brief Represents the storage of KDE computation results.
 */
class KdeResult {
  private:

    // For BOOST serialization.
    friend class ::boost::serialization::access;

    typedef std::vector<double> ContainerType;

  public:

    /** @brief The lower bound on the density sum.
     */
    ContainerType densities_l_;

    /** @brief The approximate density sum per query.
     */
    ContainerType densities_;

    /** @brief The upper bound on the density sum.
     */
    ContainerType densities_u_;

    /** @brief The number of points pruned per each query.
     */
    std::vector<unsigned long int> pruned_;

    /** @brief The amount of maximum error incurred per each query.
     */
    ContainerType used_error_;

    /** @brief The number of far-to-local translations.
     */
    int num_farfield_to_local_prunes_;

    /** @brief The number of far-field evaluations.
     */
    int num_farfield_prunes_;

    /** @brief The number of direct local accumulations.
     */
    int num_local_prunes_;

    /** @brief Accumulates the given query result to this query
     *         result.
     */
    void Accumulate(const KdeResult &result_in) {

      // Do nothing.
    }

    /** @brief Copies the given result back onto the result.
     */
    void Copy(const KdeResult & result_in) {
      for(unsigned int i = 0; i < result_in.densities_l_.size(); i++) {
        densities_l_[i] = result_in.densities_l_[i];
        densities_[i] = result_in.densities_[i];
        densities_u_[i] = result_in.densities_u_[i];
        pruned_[i] = result_in.pruned_[i];
        used_error_[i] = result_in.used_error_[i];
      }
      num_farfield_to_local_prunes_ += result_in.num_farfield_to_local_prunes_;
      num_farfield_prunes_ += result_in.num_farfield_prunes_;
      num_local_prunes_ += result_in.num_local_prunes_;
    }

    /** @brief The assignment operator that defaults back to copying
     *         values.
     */
    void operator=(const KdeResult &result_in) {
      this->Copy(result_in);
    }

    /** @brief The copy constructor that copies.
     */
    KdeResult(const KdeResult &result_in) {
      this->operator=(result_in);
    }

    /** @brief Serialize the KDE result object.
     */
    template<class Archive>
    void serialize(Archive &ar, const unsigned int version) {
      ar & densities_l_;
      ar & densities_;
      ar & densities_u_;
      ar & pruned_;
      ar & used_error_;
    }

    /** @brief Seeds the given query reuslt with the initial number of
     *         pruned reference points.
     */
    void Seed(int qpoint_index, unsigned long int initial_pruned_in) {
      pruned_[qpoint_index] += initial_pruned_in;
    }

    /** @brief The default constructor.
     */
    KdeResult() {
      SetZero();
    }

    /** @brief Normalizes the density of each query.
     */
    template<typename GlobalType>
    void Normalize(const GlobalType &global) {
      typename std::vector<double>::iterator densities_l_it =
        densities_l_.begin();
      typename std::vector<double>::iterator densities_it =
        densities_.begin();
      typename std::vector<double>::iterator densities_u_it =
        densities_u_.begin();
      for(; densities_l_it != densities_l_.end() ;
          densities_l_it++, densities_it++, densities_u_it++) {
        (*densities_l_it) *= global.get_mult_const();
        (*densities_it) *= global.get_mult_const();
        (*densities_u_it) *= global.get_mult_const();
      }
    }

    template<typename MetricType, typename GlobalType>
    void PostProcess(
      const MetricType &metric,
      const arma::vec &qpoint,
      int q_index,
      const arma::vec &q_weight,
      const GlobalType &global,
      const bool is_monochromatic) {
      if(global.normalize_densities()) {
        densities_l_[q_index] *= global.get_mult_const();
        densities_[q_index] *= global.get_mult_const();
        densities_u_[q_index] *= global.get_mult_const();
      }
    }

    /** @brief Prints the KDE results to a file.
     */
    void Print(const std::string &file_name) const {
      FILE *file_output = fopen(file_name.c_str(), "w+");
      std::vector<double>::const_iterator densities_l_it =
        densities_l_.begin();
      std::vector<double>::const_iterator densities_it =
        densities_.begin();
      std::vector<double>::const_iterator densities_u_it =
        densities_u_.begin();
      std::vector<unsigned long int>::const_iterator pruned_it =
        pruned_.begin();
      for(; densities_it != densities_.end(); densities_l_it++, densities_it++,
          densities_u_it++, pruned_it++) {
        fprintf(file_output, "%g\n", *densities_it);
      }
      fclose(file_output);
    }

    template<typename GlobalType, typename TreeType, typename DeltaType>
    void ApplyProbabilisticDelta(
      GlobalType &global, TreeType *qnode, double failure_probability,
      const DeltaType &delta_in) {

      // Get the iterator for the query node.
      typename GlobalType::TableType::TreeIterator qnode_it =
        global.query_table()->get_node_iterator(qnode);
      int qpoint_index;

      // Look up the number of standard deviations.
      double num_standard_deviations = global.compute_quantile(
                                         failure_probability);

      do {
        // Get each query point.
        qnode_it.Next(&qpoint_index);
        core::math::Range contribution;
        (*delta_in.mean_variance_pair_)[qpoint_index].scaled_interval(
          delta_in.pruned_, num_standard_deviations, &contribution);
        contribution.lo = std::max(contribution.lo, 0.0);
        contribution.hi = std::min(
                            contribution.hi, static_cast<double>(delta_in.pruned_));
        densities_l_[qpoint_index] += contribution.lo;
        densities_[qpoint_index] += contribution.mid();
        densities_u_[qpoint_index] += contribution.hi;
        pruned_[qpoint_index] += delta_in.pruned_;
      }
      while(qnode_it.HasNext());
    }

    template<typename GlobalType>
    void Init(const GlobalType &global_in, int num_points) {
      this->Init(num_points);
    }

    int size() const {
      return static_cast<int>(densities_l_.size());
    }

    void Init(int num_points) {
      densities_l_.resize(num_points);
      densities_.resize(num_points);
      densities_u_.resize(num_points);
      pruned_.resize(num_points);
      used_error_.resize(num_points);
      SetZero();
    }

    void SetZero() {
      std::vector<double>::iterator densities_l_it =
        densities_l_.begin();
      std::vector<double>::iterator densities_it =
        densities_.begin();
      std::vector<double>::iterator densities_u_it =
        densities_u_.begin();
      std::vector<unsigned long int>::iterator pruned_it =
        pruned_.begin();
      std::vector<double>::iterator used_error_it =
        used_error_.begin();
      for(; used_error_it != used_error_.end() ;
          densities_l_it++, densities_it++, densities_u_it++,
          pruned_it++, used_error_it++) {
        (*densities_l_it) = 0;
        (*densities_it) = 0;
        (*densities_u_it) = 0;
        (*pruned_it) = 0;
        (*used_error_it) = 0;
      }
      num_farfield_to_local_prunes_ = 0;
      num_farfield_prunes_ = 0;
      num_local_prunes_ = 0;
    }

    template<typename KdePostponedType>
    void ApplyPostponed(
      int q_index, const KdePostponedType &postponed_in) {
      densities_l_[q_index] = densities_l_[q_index] + postponed_in.densities_l_;
      densities_[q_index] = densities_[q_index] + postponed_in.densities_e_;
      densities_u_[q_index] = densities_u_[q_index] + postponed_in.densities_u_;
      pruned_[q_index] = pruned_[q_index] + postponed_in.pruned_;
      used_error_[q_index] = used_error_[q_index] + postponed_in.used_error_;
    }

    template<typename GlobalType, typename KdePostponedType>
    void FinalApplyPostponed(
      const GlobalType &global,
      const arma::vec &qpoint,
      int q_index,
      const KdePostponedType &postponed_in) {

      // Evaluate the local expansion.
      densities_[q_index] +=
        postponed_in.local_expansion_.EvaluateField(
          global.kernel_aux(), qpoint);

      // Apply postponed.
      ApplyPostponed(q_index, postponed_in);
    }
};
}
}

#endif
