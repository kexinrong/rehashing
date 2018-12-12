/** @file kde_dev.h
 *
 *  The kernel density estimator object that processes user inputs and
 *  to produce the computation results.
 *
 *  @author Dongryeol Lee (dongryel@cc.gatech.edu)
 */

#ifndef MLPACK_KDE_KDE_DEV_H
#define MLPACK_KDE_KDE_DEV_H

#include "core/gnp/dualtree_dfs_dev.h"
#include "core/metric_kernels/lmetric.h"
#include "core/table/transform.h"
#include "mlpack/kde/kde.h"

namespace mlpack {
namespace kde {

template<typename TableType>
void KdeArgumentParser::PrescaleTable_(
  const std::string &prescale_option, TableType *table) {

  if(prescale_option == "hypercube") {
    core::table::UnitHypercube::Transform(table);
  }
  else if(prescale_option == "standardize") {
    core::table::Standardize::Transform(table);
  }
}

template<typename TableType, typename KernelAuxType>
TableType *Kde<TableType, KernelAuxType>::query_table() {
  return query_table_;
}

template<typename TableType, typename KernelAuxType>
TableType *Kde<TableType, KernelAuxType>::reference_table() {
  return reference_table_;
}

template<typename TableType, typename KernelAuxType>
typename Kde <
TableType, KernelAuxType >::GlobalType &
Kde<TableType, KernelAuxType>::global() {
  return global_;
}

template<typename TableType, typename KernelAuxType>
bool Kde<TableType, KernelAuxType>::is_monochromatic() const {
  return is_monochromatic_;
}

template<typename TableType, typename KernelAuxType>
void Kde<TableType, KernelAuxType>::Compute(
  const KdeArguments<TableType> &arguments_in,
  ResultType *result_out) {

  // Instantiate a dual-tree algorithm of the KDE.
  typedef Kde<TableType, KernelAuxType> ProblemType;
  core::gnp::DualtreeDfs< ProblemType > dualtree_dfs;
  dualtree_dfs.Init(*this);

  // Compute the result.
  dualtree_dfs.Compute(* arguments_in.metric_, result_out);
  printf("Number of prunes: %d\n", dualtree_dfs.num_deterministic_prunes());
  printf("Number of probabilistic prunes: %d\n",
         dualtree_dfs.num_probabilistic_prunes());
  printf(
    "Number of far-to-local prunes: %d\n",
    result_out->num_farfield_to_local_prunes_);
  printf("Number of farfield prunes: %d\n", result_out->num_farfield_prunes_);
  printf("Number of local prunes: %d\n", result_out->num_local_prunes_);
}

template<typename TableType, typename KernelAuxType>
template<typename IncomingGlobalType>
void Kde<TableType, KernelAuxType>::Init(
  KdeArguments<TableType> &arguments_in, IncomingGlobalType *global_in) {

  reference_table_ = arguments_in.reference_table_;
  if(global_in != NULL) {
    is_monochromatic_ = global_in->is_monochromatic();
    query_table_ = arguments_in.query_table_;
    reference_table_ = arguments_in.reference_table_;
  }
  else {
    if(arguments_in.query_table_ == arguments_in.reference_table_) {
      is_monochromatic_ = true;
      query_table_ = reference_table_;
    }
    else {
      is_monochromatic_ = false;
      query_table_ = arguments_in.query_table_;
    }
  }

  // Declare the global constants.
  KernelAuxType *kernel_aux_ptr =
    (global_in) ?
    const_cast<KernelAuxType *>(& global_in->kernel_aux()) : NULL;
  typename IncomingGlobalType::MeanVariancePairListType *
  mean_variance_pair_ptr =
    (global_in) ?
    const_cast <
    typename IncomingGlobalType::MeanVariancePairListType * >(
      global_in->mean_variance_pair()) : NULL;
  global_.Init(
    reference_table_, query_table_,
    arguments_in.effective_num_reference_points_, kernel_aux_ptr,
    arguments_in.bandwidth_, mean_variance_pair_ptr, is_monochromatic_,
    arguments_in.relative_error_, arguments_in.absolute_error_,
    arguments_in.probability_, arguments_in.normalize_densities_);
}

template<typename TableType, typename KernelAuxType>
void Kde<TableType, KernelAuxType>::set_bandwidth(double bandwidth_in) {
  global_.set_bandwidth(bandwidth_in);
}

bool KdeArgumentParser::ConstructBoostVariableMap(
  int argc,
  char *argv[],
  ::boost::program_options::variables_map *vm) {

  // Convert C input to C++; skip executable name for Boost.
  std::vector<std::string> args(argv + 1, argv + argc);

  // Call the other function.
  return ConstructBoostVariableMap(args, vm);
}

bool KdeArgumentParser::ConstructBoostVariableMap(
  const std::vector<std::string> &args,
  ::boost::program_options::variables_map *vm) {

  ::boost::program_options::options_description desc("Available options");
  desc.add_options()(
    "help", "Print this information."
  )(
    "absolute_error",
    ::boost::program_options::value<double>()->default_value(0.0),
    "Absolute error for the approximation of KDE per each query point."
  )(
    "bandwidth",
    ::boost::program_options::value<double>(),
    "OPTIONAL kernel bandwidth, if you set --bandwidth_selection flag, "
    "then the --bandwidth will be ignored."
  )(
    "densities_out",
    ::boost::program_options::value<std::string>()->default_value(
      "densities_out.csv"),
    "OPTIONAL file to store computed densities."
  )(
    "kernel",
    ::boost::program_options::value<std::string>()->default_value("gaussian"),
    "Kernel function used by KDE.  One of:\n"
    "  epan, gaussian"
  )(
    "queries_in",
    ::boost::program_options::value<std::string>(),
    "OPTIONAL file containing query positions.  If omitted, KDE computes "
    "the leave-one-out density at each reference point."
  )(
    "references_in",
    ::boost::program_options::value<std::string>(),
    "REQUIRED file containing reference data."
  )(
    "series_expansion_type",
    ::boost::program_options::value<std::string>()->default_value("multivariate"),
    "Series expansion type used to compress the kernel interaction. One of:\n"
    "  hypercube, multivariate"
  )(
    "probability",
    ::boost::program_options::value<double>()->default_value(1.0),
    "Probability guarantee for the approximation of KDE."
  )(
    "relative_error",
    ::boost::program_options::value<double>()->default_value(0.1),
    "Relative error for the approximation of KDE."
  )(
    "leaf_size",
    ::boost::program_options::value<int>()->default_value(20),
    "Maximum number of points at a leaf of the tree."
  )(
    "prescale",
    ::boost::program_options::value<std::string>()->default_value("none"),
    "OPTIONAL scaling option. One of:\n"
    "  none, hypercube, standardize"
  );

  ::boost::program_options::command_line_parser clp(args);
  clp.style(::boost::program_options::command_line_style::default_style
            ^ ::boost::program_options::command_line_style::allow_guessing);
  try {
    ::boost::program_options::store(clp.options(desc).run(), *vm);
  }
  catch(const ::boost::program_options::invalid_option_value &e) {
    std::cerr << "Invalid Argument: " << e.what() << "\n";
    exit(0);
  }
  catch(const ::boost::program_options::invalid_command_line_syntax &e) {
    std::cerr << "Invalid command line syntax: " << e.what() << "\n";
    exit(0);
  }
  catch(const ::boost::program_options::unknown_option &e) {
    std::cerr << "Unknown option: " << e.what() << "\n";
    exit(0);
  }

  ::boost::program_options::notify(*vm);
  if(vm->count("help")) {
    std::cout << desc << "\n";
    return true;
  }

  // Validate the arguments. Only immediate termination is allowed
  // here, the parsing is done later.
  if(vm->count("references_in") == 0) {
    std::cerr << "Missing required --references_in.\n";
    exit(0);
  }
  if((*vm)["kernel"].as<std::string>() != "gaussian" &&
      (*vm)["kernel"].as<std::string>() != "epan") {
    std::cerr << "We support only epan or gaussian for the kernel.\n";
    exit(0);
  }
  if((*vm)["series_expansion_type"].as<std::string>() != "hypercube" &&
      (*vm)["series_expansion_type"].as<std::string>() != "multivariate") {
    std::cerr << "We support only hypercube or multivariate for the "
              "series expansion type.\n";
    exit(0);
  }
  if(vm->count("bandwidth") > 0 && (*vm)["bandwidth"].as<double>() <= 0) {
    std::cerr << "The --bandwidth requires a positive real number.\n";
    exit(0);
  }
  if(vm->count("bandwidth") == 0) {
    std::cerr << "Missing required --bandwidth.\n";
    exit(0);
  }
  if((*vm)["probability"].as<double>() <= 0 ||
      (*vm)["probability"].as<double>() > 1) {
    std::cerr << "The --probability requires a real number $0 < p <= 1$.\n";
    exit(0);
  }
  if((*vm)["relative_error"].as<double>() < 0) {
    std::cerr << "The --relative_error requires a real number $r >= 0$.\n";
    exit(0);
  }
  if((*vm)["leaf_size"].as<int>() <= 0) {
    std::cerr << "The --leaf_size needs to be a positive integer.\n";
    exit(0);
  }
  if(vm->count("prescale") > 0) {
    if((*vm)["prescale"].as<std::string>() != "hypercube" &&
        (*vm)["prescale"].as<std::string>() != "standardize" &&
        (*vm)["prescale"].as<std::string>() != "none") {
      std::cerr << "The --prescale needs to be: none or hypercube or " <<
                "standardize.\n";
      exit(0);
    }
  }
  return false;
}

template<typename TableType>
bool KdeArgumentParser::ParseArguments(
  ::boost::program_options::variables_map &vm,
  KdeArguments<TableType> *arguments_out) {

  // A L2 metric to index the table to use.
  arguments_out->metric_ = new core::metric_kernels::LMetric<2>();

  // Given the constructed boost variable map, parse each argument.

  // Parse the densities out file.
  arguments_out->densities_out_ = vm["densities_out"].as<std::string>();

  // Parse the leaf size.
  arguments_out->leaf_size_ = vm["leaf_size"].as<int>();
  std::cout << "Using the leaf size of " << arguments_out->leaf_size_ << "\n";

  // Parse the measurement error mode flag.
  arguments_out->measurement_error_mode_ =
    (vm.count("measurement_error_mode") > 0);
  if(arguments_out->measurement_error_mode_) {
    std::cerr << "Measurement error mode is on.\n";
  }
  else {
    std::cerr << "Measurement error mode is off.\n";
  }

  // Parse the reference set and index the tree.
  std::cout << "Reading in the reference set: " <<
            vm["references_in"].as<std::string>() << "\n";
  arguments_out->reference_table_ = new TableType();
  arguments_out->reference_table_->Init(
    vm["references_in"].as<std::string>(), 0,
    (vm.count("noise_scales_in") > 0) ?
    & (vm["noise_scales_in"].as<std::string>()) :
    (const std::string *) NULL);
  std::cout << "Finished reading in the reference set.\n";

  // Verify that the noise scales are properly read in for the
  // measurement error mode.
  if(arguments_out->measurement_error_mode_) {
    if(static_cast<int>(
          arguments_out->reference_table_->weights().n_rows) !=
        arguments_out->reference_table_->n_attributes()) {
      std::cerr << "--measurement_error_mode requires noise scales per" <<
                " each dimension per each reference point.\n";
      exit(0);
    }
  }

  // Scale the dataset.
  PrescaleTable_(
    vm["prescale"].as<std::string>(), arguments_out->reference_table_);
  std::cout << "Scaled the dataset with the option: " <<
            vm["prescale"].as<std::string>() << "\n";

  std::cout << "Building the reference tree.\n";
  arguments_out->reference_table_->IndexData(
    *(arguments_out->metric_), arguments_out->leaf_size_, 0);
  std::cout << "Finished building the reference tree.\n";

  // Parse the query set and index the tree.
  if(vm.count("queries_in") > 0) {
    std::cout << "Reading in the query set: " <<
              vm["queries_in"].as<std::string>() << "\n";
    arguments_out->query_table_ = new TableType();
    arguments_out->query_table_->Init(vm["queries_in"].as<std::string>());
    std::cout << "Finished reading in the query set.\n";

    // Scale the dataset.
    PrescaleTable_(
      vm["prescale"].as<std::string>(), arguments_out->query_table_);
    std::cout << "Scaled the dataset with the option: " <<
              vm["prescale"].as<std::string>() << "\n";

    std::cout << "Building the query tree.\n";
    arguments_out->query_table_->IndexData(
      *(arguments_out->metric_), arguments_out->leaf_size_, 1);
    std::cout << "Finished building the query tree.\n";
    arguments_out->effective_num_reference_points_ =
      arguments_out->reference_table_->n_entries();
  }
  else {
    arguments_out->query_table_ = arguments_out->reference_table_;
    arguments_out->effective_num_reference_points_ =
      arguments_out->reference_table_->n_entries() - 1;
  }

  // Parse the bandwidth.
  arguments_out->bandwidth_ = vm["bandwidth"].as<double>();
  std::cout << "Bandwidth of " << arguments_out->bandwidth_ << "\n";

  // Parse the absolute error.
  arguments_out->absolute_error_ = vm["absolute_error"].as<double>();

  // Parse the relative error.
  arguments_out->relative_error_ = vm["relative_error"].as<double>();
  std::cout << "For each query point $q in \\mathcal{Q}$, " <<
            "we will guarantee: " <<
            "$| widetilde{G}(q) - G(q) | \\leq "
            << arguments_out->relative_error_ <<
            " \\cdot G(q) + " << arguments_out->absolute_error_ <<
            " | \\mathcal{R} |$ \n";

  // Parse the probability.
  arguments_out->probability_ = vm["probability"].as<double>();
  std::cout << "Probability of " << arguments_out->probability_ << "\n";

  // Parse the kernel type.
  arguments_out->kernel_ = vm["kernel"].as< std::string >();
  std::cout << "Using the kernel: " << arguments_out->kernel_ << "\n";

  // Parse the series expansion type.
  arguments_out->series_expansion_type_ =
    vm["series_expansion_type"].as<std::string>();
  std::cout << "Using the series expansion type: " <<
            arguments_out->series_expansion_type_ << "\n";


  return false;
}
}
}

#endif
