/** @file kde_arguments.h
 *
 *  @author Dongryeol Lee (dongryel@cc.gatech.edu)
 */

#ifndef MLPACK_KDE_KDE_ARGUMENTS_H
#define MLPACK_KDE_KDE_ARGUMENTS_H

#include "core/table/table.h"
#include "core/metric_kernels/lmetric.h"

namespace mlpack {
namespace kde {

/** @brief The class that manages the arguments necessary for KDE
 *         computation.
 */
template<typename TableType>
class KdeArguments {
  public:

    std::string densities_out_;

    int leaf_size_;

    TableType *reference_table_;

    TableType *query_table_;

    unsigned long int effective_num_reference_points_;

    double bandwidth_;

    double absolute_error_;

    /** @brief Whether to use the measurement error mode or not.
     */
    bool measurement_error_mode_;

    double relative_error_;

    double probability_;

    std::string kernel_;

    std::string series_expansion_type_;

    core::metric_kernels::LMetric<2> *metric_;

    bool tables_are_aliased_;

    bool normalize_densities_;

  public:

    template<typename GlobalType>
    void Init(
      TableType *reference_table_in, TableType *query_table_in,
      GlobalType &global_in) {
      reference_table_ = reference_table_in;
      query_table_ = query_table_in;
      effective_num_reference_points_ =
        global_in.effective_num_reference_points();
      bandwidth_ = global_in.bandwidth();
      absolute_error_ = global_in.absolute_error();
      relative_error_ = global_in.relative_error();
      measurement_error_mode_ = global_in.measurement_error_mode();
      probability_ = global_in.probability();
      kernel_ = global_in.kernel().name();
      series_expansion_type_ = global_in.series_expansion_type();
      tables_are_aliased_ = true;
      normalize_densities_ = global_in.normalize_densities();
    }

    template<typename GlobalType>
    void Init(GlobalType &global_in) {
      reference_table_ = global_in.reference_table()->local_table();
      if(reference_table_ != query_table_) {
        query_table_ = global_in.query_table()->local_table();
      }
      effective_num_reference_points_ =
        global_in.effective_num_reference_points();
      bandwidth_ = global_in.bandwidth();
      absolute_error_ = global_in.absolute_error();
      relative_error_ = global_in.relative_error();
      measurement_error_mode_ = global_in.measurement_error_mode();
      probability_ = global_in.probability();
      kernel_ = global_in.kernel().name();
      series_expansion_type_ = global_in.series_expansion_type();
      tables_are_aliased_ = true;
      normalize_densities_ = global_in.normalize_densities();
    }

    /** @brief The default constructor.
     */
    KdeArguments() {
      leaf_size_ = 0;
      reference_table_ = NULL;
      query_table_ = NULL;
      effective_num_reference_points_ = 0;
      bandwidth_ = 0.0;
      absolute_error_ = 0.0;
      measurement_error_mode_ = false;
      relative_error_ = 0.0;
      probability_ = 0.0;
      kernel_ = "";
      series_expansion_type_ = "";
      metric_ = NULL;
      tables_are_aliased_ = false;
      normalize_densities_ = true;
    }

    /** @brief The destructor.
     */
    ~KdeArguments() {

      // If the tables are not aliased,
      if(tables_are_aliased_ == false) {

        // If monchromatic, destroy only one of the tables.
        if(reference_table_ == query_table_) {
          if(reference_table_ != NULL) {
            delete reference_table_;
          }
        }
        else {

          // Otherwise, we have to check and destroy both.
          if(reference_table_ != NULL) {
            delete reference_table_;
          }
          if(query_table_ != NULL) {
            delete query_table_;
          }
        }
      }
      reference_table_ = NULL;
      query_table_ = NULL;

      // Destroy the metric.
      if(metric_ != NULL) {
        delete metric_;
        metric_ = NULL;
      }
    }
};
}
}

#endif
