/** @file test_kde.h
 *
 *  @author Dongryeol Lee (dongryel@cc.gatech.edu)
 */

#ifndef MLPACK_KDE_TEST_KDE_H
#define MLPACK_KDE_TEST_KDE_H

#include <boost/test/unit_test.hpp>
#include "core/gnp/dualtree_dfs_dev.h"
#include "core/table/random_dataset_generator.h"
#include "core/tree/gen_metric_tree.h"
#include "core/math/math_lib.h"
#include "mlpack/kde/kde_dev.h"
#include "mlpack/series_expansion/kernel_aux.h"

namespace mlpack {
namespace kde {
namespace test_kde {
extern int num_dimensions_;
extern int num_points_;
}

class TestKde {

  private:

    template<typename QueryResultType>
    bool CheckAccuracy_(
      const QueryResultType &query_results,
      const std::vector<double> &naive_query_results,
      double relative_error) {

      // Compute the collective L1 norm of the products.
      double achieved_error = 0;
      for(unsigned int j = 0; j < query_results.size(); j++) {
        double per_relative_error =
          fabs(naive_query_results[j] - query_results[j]) /
          fabs(naive_query_results[j]);
        achieved_error = std::max(achieved_error, per_relative_error);
        if(relative_error < per_relative_error) {
          std::cout << query_results[j] << " against " <<
                    naive_query_results[j] << ": " <<
                    per_relative_error << "\n";
        }
      }
      std::cout <<
                "Achieved a relative error of " << achieved_error << "\n";
      return achieved_error <= relative_error;
    }

  public:

    template<typename MetricType, typename TableType, typename KernelType>
    static void UltraNaive(
      const MetricType &metric_in,
      TableType &query_table, TableType &reference_table,
      const KernelType &kernel,
      std::vector<double> &ultra_naive_query_results) {

      ultra_naive_query_results.resize(query_table.n_entries());
      for(int i = 0; i < query_table.n_entries(); i++) {
        arma::vec query_point;
        query_table.get(i, &query_point);
        ultra_naive_query_results[i] = 0;

        for(int j = 0; j < reference_table.n_entries(); j++) {
          arma::vec reference_point;
          arma::vec reference_weight;
          reference_table.get(
            j, &reference_point, &reference_weight);

          // By default, monochromaticity is assumed in the test -
          // this will be addressed later for general bichromatic
          // test.
          if(i == j) {
            continue;
          }

          double kernel_value =
            kernel.Evaluate(
              metric_in, query_point, reference_point, reference_weight);

          ultra_naive_query_results[i] += kernel_value;
        }

        // Divide by N - 1 for LOO. May have to be adjusted later.
        ultra_naive_query_results[i] *=
          (1.0 / (kernel.CalcNormConstant(query_table.n_attributes()) *
                  ((double)
                   reference_table.n_entries() - 1)));
      }
    }

    int StressTestMain() {
      for(int i = 0; i < 20; i++) {
        for(int k = 0; k < 4; k++) {
          // Randomly choose the number of dimensions and the points.
          mlpack::kde::test_kde::num_dimensions_ = core::math::RandInt(2, 5);
          mlpack::kde::test_kde::num_points_ = core::math::RandInt(500, 1001);

          switch(k) {
            case 0:
              StressTest <
              mlpack::series_expansion::GaussianKernelHypercubeAux > ();
              break;
            case 1:
              StressTest <
              mlpack::series_expansion::GaussianKernelMultivariateAux > ();
              break;
            case 2:
              StressTest <
              mlpack::series_expansion::EpanKernelMultivariateAux > ();
              break;
            case 3:
              StressTest <
              mlpack::series_expansion::DeconvGaussianKernelMultivariateAux > ();
              break;
          }
        }
      }
      return 0;
    }

    template<typename KernelAuxType>
    int StressTest() {

      static const enum mlpack::series_expansion::CartesianExpansionType
      ExpansionType = KernelAuxType::ExpansionType;
      typedef core::table::Table <
      core::tree::GenMetricTree <
      mlpack::kde::KdeStatistic<ExpansionType> > ,
             mlpack::kde::KdeResult > TableType;

      // The list of arguments.
      std::vector< std::string > args;

      // Push in the reference dataset name.
      std::string references_in("random.csv");
      args.push_back(std::string("--references_in=") + references_in);

      // Push in the densities output file name.
      args.push_back(std::string("--densities_out=densities.txt"));

      // Push in the kernel type.
      std::cout << "\n==================\n";
      std::cout << "Test trial begin\n";
      std::cout << "Number of dimensions: " <<
                mlpack::kde::test_kde::num_dimensions_ << "\n";
      std::cout << "Number of points: " <<
                mlpack::kde::test_kde::num_points_ << "\n";

      KernelAuxType dummy_kernel_aux;
      if(dummy_kernel_aux.kernel().name() == "epan") {
        std::cout << "Epan kernel, \n";
        args.push_back(std::string("--kernel=epan"));
      }
      else if(dummy_kernel_aux.kernel().name() == "gaussian") {
        std::cout << "Gaussian kernel, \n";
        args.push_back(std::string("--kernel=gaussian"));
      }
      else if(dummy_kernel_aux.kernel().name() == "deconv_gaussian") {
        std::cout << "Deconvolution Gaussian kernel, \n";
        args.push_back(std::string("--kernel=gaussian"));
        args.push_back(std::string("--measurement_error_mode"));
        args.push_back(
          std::string("--noise_scales_in=random_noise_scales.csv"));
        TableType random_table;
        core::table::RandomDatasetGenerator::Generate(
          mlpack::kde::test_kde::num_dimensions_,
          mlpack::kde::test_kde::num_points_, 0, std::string("none"),
          std::string("uniform"), false, &random_table);
        random_table.Save(std::string("random_noise_scales.csv"));
      }
      if(dummy_kernel_aux.series_expansion_type() == "hypercube") {
        args.push_back(std::string("--series_expansion_type=hypercube"));
      }
      else if(dummy_kernel_aux.series_expansion_type() ==
              "multivariate") {
        args.push_back(std::string("--series_expansion_type=multivariate"));
      }

      // Push in the leaf size.
      int leaf_size = 20;
      std::stringstream leaf_size_sstr;
      leaf_size_sstr << "--leaf_size=" << leaf_size;
      args.push_back(leaf_size_sstr.str());

      // Push in the relative error argument.
      double relative_error = 0.1;
      std::stringstream relative_error_sstr;
      relative_error_sstr << "--relative_error=" << relative_error;
      args.push_back(relative_error_sstr.str());

      // Push in the randomly generated bandwidth.
      double bandwidth =
        core::math::Random(
          0.05 * sqrt(mlpack::kde::test_kde::num_dimensions_),
          0.1 * sqrt(mlpack::kde::test_kde::num_dimensions_));
      std::stringstream bandwidth_sstr;
      bandwidth_sstr << "--bandwidth=" << bandwidth;
      args.push_back(bandwidth_sstr.str());

      // Generate the random dataset and save it.
      TableType random_table;
      core::table::RandomDatasetGenerator::Generate(
        mlpack::kde::test_kde::num_dimensions_,
        mlpack::kde::test_kde::num_points_, 0, std::string("none"),
        std::string("uniform"), false, &random_table);
      random_table.Save(references_in);

      // Parse the KDE arguments.
      mlpack::kde::KdeArguments<TableType> kde_arguments;
      boost::program_options::variables_map vm;
      mlpack::kde::KdeArgumentParser::ConstructBoostVariableMap(args, &vm);
      mlpack::kde::KdeArgumentParser::ParseArguments(vm, &kde_arguments);

      std::cout << "Bandwidth value " << bandwidth << "\n";

      // Call the KDE driver.
      mlpack::kde::Kde<TableType, KernelAuxType> kde_instance;
      kde_instance.Init(
        kde_arguments,
        (typename mlpack::kde::Kde <
         TableType, KernelAuxType >::GlobalType *) NULL);

      // Compute the result.
      mlpack::kde::KdeResult kde_result;
      kde_instance.Compute(kde_arguments, &kde_result);

      // Call the ultra-naive.
      std::vector<double> ultra_naive_kde_result;

      UltraNaive(
        *(kde_arguments.metric_), *(kde_arguments.reference_table_),
        *(kde_arguments.reference_table_),
        kde_instance.global().kernel(),
        ultra_naive_kde_result);
      if(CheckAccuracy_(
            kde_result.densities_,
            ultra_naive_kde_result,
            kde_arguments.relative_error_) == false) {
        std::cerr << "There is a problem!\n";
      }

      return 0;
    };
};
}
}
#endif
