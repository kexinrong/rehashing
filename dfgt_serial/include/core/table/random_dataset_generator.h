/** @file random_dataset_generator.h
 *
 *  @author Dongryeol Lee (dongryel@cc.gatech.edu)
 */

#ifndef CORE_PARALLEL_RANDOM_DATASET_GENERATOR_H
#define CORE_PARALLEL_RANDOM_DATASET_GENERATOR_H

#include <armadillo>
#include <iostream>
#include "core/math/math_lib.h"
#include "core/table/transform.h"

namespace core {
namespace table {
class RandomDatasetGenerator {
  public:

    enum DatasetType { UNIFORM, ANNULUS, ARITH, BALL, CLUSNORM, CUBEDIAM, CUBEEDGE, CORNERS, GRID, NORMAL, SPOKES  };

  private:

    static DatasetType DetectType_(const std::string &option) {

      // By default, uniform type.
      DatasetType return_type = UNIFORM;
      if(option == "annulus") {
        return_type = ANNULUS;
        std::cerr << "Using the annulus distribution.\n";
      }
      else if(option == "arith") {
        return_type = ARITH;
        std::cerr << "Using the arith distribution.\n";
      }
      else if(option == "ball") {
        return_type = BALL;
        std::cerr << "Using the ball distribution.\n";
      }
      else if(option == "clusnorm") {
        return_type = CLUSNORM;
        std::cerr << "Using the clusnorm distribution.\n";
      }
      else if(option == "cubediam") {
        return_type = CUBEDIAM;
        std::cerr << "Using the cubediam distribution.\n";
      }
      else if(option == "cubeedge") {
        return_type = CUBEEDGE;
        std::cerr << "Using the cubeedge distribution.\n";
      }
      else if(option == "corners") {
        return_type = CORNERS;
        std::cerr << "Using the corners distribution.\n";
      }
      else if(option == "grid") {
        return_type = GRID;
        std::cerr << "Using the grid distribution.\n";
      }
      else if(option == "normal") {
        return_type = NORMAL;
        std::cerr << "Using the normal distribution.\n";
      }
      else if(option == "spokes") {
        return_type = SPOKES;
        std::cerr << "Using the spokes distribution.\n";
      }
      else {
        std::cerr << "Defaulting to uniform distribution.\n";
      }
      return return_type;
    }

  public:
    template<typename TableType>
    static void Generate(
      int num_dimensions, int num_points,
      int rank, const std::string &prescale_option,
      const std::string &dataset_option,
      bool generate_weights, TableType *random_dataset,
      int num_weights_per_point = 1) {

      // Allocate table.
      random_dataset->Init(
        num_dimensions, num_points, rank, num_weights_per_point);
      DatasetType random_type = DetectType_(dataset_option);

      // For the CLUSNORM distribution.
      arma::mat random_point;
      random_point.zeros(num_dimensions, 10);
      if(random_type == CLUSNORM) {
        for(int j = 0; j < 10; j++) {
          for(int i = 0; i < num_dimensions; i++) {
            random_point.at(i, j) = core::math::Random<double>();
          }
        }
      }

      // For the GRID distribution,
      double gridsize =
        pow(
          1.3 * num_points, 1.0 / static_cast<double>(num_dimensions));

      // Generate random number.
      for(int j = 0; j < num_points; j++) {
        arma::vec point;
        double theta = 0.0;
        double distance = 0.0;
        double magnitude = 0.0;
        random_dataset->get(j, &point);

        switch(random_type) {
          case UNIFORM:
            for(int i = 0; i < num_dimensions; i++) {
              point[i] = core::math::Random<double>();
            }
            break;
          case ANNULUS:

            // First two dimensions are on a circle.
            theta = core::math::Random<double>() *
                    2.0 * ::boost::math::constants::pi<double>();
            point[0] = sin(theta);
            point[1] = cos(theta);

            // The rest of the dimensions are uniformly random.
            for(int i = 2; i < num_dimensions; i++) {
              point[i] = core::math::Random<double>();
            }
            break;
          case ARITH:
            point[0] = j * j;
            for(int i = 1; i < num_dimensions; i++) {
              point[i] = 0.0;
            }
            break;
          case BALL:
            distance = pow(
                         core::math::Random<double>(),
                         1.0 / static_cast<double>(num_dimensions));
            for(int i = 0; i < num_dimensions; i++) {
              point[i] = core::math::RandGaussian(1.0);
              magnitude = magnitude + core::math::Sqr(point[i]);
            }
            magnitude = sqrt(magnitude);
            for(int i = 0; i < num_dimensions; i++) {
              point[i] = point[i] / magnitude * distance;
            }
            break;
          case CLUSNORM:
            for(int i = 0; i < num_dimensions; i++) {
              point[i] = random_point.at(i, j % 10)  +
                         core::math::RandGaussian(0.05);
            }
            break;
          case CUBEDIAM:
            point[0] = core::math::Random<double>();
            for(int i = 1; i < num_dimensions; i++) {
              point[i] = point[0];
            }
            break;
          case CUBEEDGE:
            point[0] = core::math::Random<double>();
            for(int i = 1; i < num_dimensions; i++) {
              point[i] = 0.0;
            }
            break;
          case CORNERS:
            for(int i = 0; i < 2; i++) {
              if(core::math::Random<double>() < 0.5) {
                point[i] = core::math::Random<double>() - 0.5;
              }
              else {
                point[i] = core::math::Random<double>() + 1.5;
              }
            }
            for(int i = 2; i < num_dimensions; i++) {
              point[i] = core::math::Random<double>();
            }
            break;
          case GRID:
            for(int i = 0; i < num_dimensions; i++) {
              point[i] = floor(gridsize * core::math::Random<double>()) -
                         gridsize / 2.0;
            }
            break;
          case NORMAL:
            for(int i = 0; i < num_dimensions; i++) {
              point[i] = core::math::RandGaussian(1.0);
            }
            break;
          case SPOKES:
            for(int i = 0; i < num_dimensions; i++) {
              point[i] = 0.5;
            }
            point[j % num_dimensions] = core::math::Random<double>();
            break;
        }

        // Set the weight to the random one.
        if(generate_weights) {
          random_dataset->weights().at(0, j) = core::math::Random(1.0, 5.0);
          for(int k = 1; k < num_weights_per_point; k++) {
            random_dataset->weights().at(k, j) =
              core::math::Random(0.1, 0.5);
          }
        }
      }

      // Scale the dataset.
      if(prescale_option == "hypercube") {
        core::table::UnitHypercube::Transform(random_dataset);
      }
      else if(prescale_option == "standardize") {
        core::table::Standardize::Transform(random_dataset);
      }

      // Now, make sure that all coordinates are non-negative.
      if(prescale_option != "hypercube") {
        core::table::TranslateToNonnegative::Transform(
          random_dataset);
      }

      std::cout << "Scaled the dataset with the option: " <<
                prescale_option << "\n";
    }
};
}
}

#endif
