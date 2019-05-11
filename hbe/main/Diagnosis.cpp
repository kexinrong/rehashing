/*
 *  Diagnosis:
 *      Output the estimated relative variance of HBE and RS given dataset and hashing scheme.
 *      The lower the variance, the more sample efficient the estimator is.
 *      Take the median of the estimated variance for robustness.
 *
 *  Example usage:
 *      ./hbe conf/shuttle.cfg gaussian
 */

#include <chrono>
#include "../utils/DataIngest.h"
#include "../alg/RS.h"
#include "../alg/AdaptiveRSDiag.h"
#include "parseConfig.h"

int main(int argc, char *argv[]) {
    if (argc < 2) {
        std::cout << "Need config file" << std::endl;
        exit(1);
    }

    char *scope = argv[2];
    parseConfig cfg(argv[1], scope);
    DataIngest data(cfg, false);

    AdaptiveRSDiag rs(data.X_ptr, data.kernel, data.tau, 0.6);
    rs.setMedians(5);

    std::random_device rd;
    std::mt19937 rng(rd());
    std::uniform_int_distribution<int> distribution(0, data.M - 1);

    auto t1 = std::chrono::high_resolution_clock::now();
    for (int iter = 0; iter < 3; iter ++) {
        vector<double> rs_cost;
        vector<double> hbe_cost;
        int j = 0;
        while (j < 30) {
            int idx = distribution(rng);
            VectorXd q = data.X_ptr->row(idx);
            if (data.hasQuery != 0) {
                q = data.Y_ptr->row(j);
            }
            rs.clearSamples();
            vector<double> rs_est = rs.query(q);
            if (rs_est[0] < data.tau) { continue; }
            double r2 = max(rs_est[0], data.tau);
            r2 *= r2;

            int actual = rs.findActualLevel(q, rs_est[0], data.eps);
            rs.getConstants();
            rs.findRings(1, 0.5, q, actual);
            // Uncomment to output rs.lambda, rs.l for visualization
            // std::cout << rs.lambda << "," << rs.l << std::endl;

            // Estimate relative variance; not necessary for visualization
            rs_cost.push_back(rs.RSDirect() / r2);
            hbe_cost.push_back(rs.HBEDirect() / r2);

            j ++;
        }
        std::cout << "rs:" << dataUtils::getAvg(rs_cost) << ", hbe: " <<  dataUtils::getAvg(hbe_cost) << std::endl;
    }
    auto t2 = std::chrono::high_resolution_clock::now();
    std::cout << "Diagnosis took: " <<
              std::chrono::duration_cast<std::chrono::milliseconds>(t2-t1).count() << "ms" << std::endl;

}