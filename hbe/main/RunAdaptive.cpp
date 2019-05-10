/*
 *  Adaptive Sampling:
 *     Run adaptive sampling given 1) dataset 2) epsilon 3) RS or HBE
 *
 *
 *  Example usage:
 *      ./hbe conf/shuttle.cfg 0.2 true
 *          => Run adaptive sampling with RS, with eps=0.2
 *
 *      ./hbe conf/shuttle.cfg 0.9
 *          => Run adaptive sampling with HBE, with eps=0.9
 *
 */

#include <chrono>
#include "../alg/RS.h"
#include "../alg/AdaptiveRS.h"
#include "../alg/AdaptiveHBE.h"
#include "../utils/DataIngest.h"
#include "parseConfig.h"

void update(vector<double>& results, vector<double> est, double exact) {
    results[0] += fabs(est[0] - exact) / exact;
    results[1] += est[1];
}

int main(int argc, char *argv[]) {
    if (argc < 2) {
        std::cout << "Need config file" << std::endl;
        exit(1);
    }

    char *scope = argv[2];
    double eps = atof(argv[3]);
    bool random = (argc > 4);

    parseConfig cfg(argv[1], scope);
    DataIngest data(cfg, true);

    shared_ptr<AdaptiveEstimator> est;
    std::cout << "eps = " << eps << std::endl;
    if (random) {
        std::cout << "RS" << std::endl;
        est = make_shared<AdaptiveRS>(data.X_ptr, data.kernel, data.tau, eps);
    } else {
        std::cout << "HBE" << std::endl;
        auto t1 = std::chrono::high_resolution_clock::now();
        est = make_shared<AdaptiveHBE>(data.X_ptr, data.kernel, data.tau, eps, true);
        auto t2 = std::chrono::high_resolution_clock::now();
        std::cout << "Adaptive Table Init: " <<
                  std::chrono::duration_cast<std::chrono::milliseconds>(t2-t1).count() / 1000.0 << std::endl;
    }

    est->totalTime = 0;
    vector<double> results(2,0);

    for (int j = 0; j < data.M; j++) {
        int idx = j * 2;
        VectorXd q = data.X_ptr->row(j);
        if (data.hasQuery != 0) {
            q = data.Y_ptr->row(j);
        } else {
            if (!data.sequential) {
                q = data.X_ptr->row(data.exact[idx + 1]);
            }
        }
        vector<double> estimates = est->query(q);
        update(results, estimates, data.exact[idx]);
    }

    std::cout << "Sampling total time: " << est->totalTime / 1e9 << std::endl;
    std::cout << "Average Samples: " << results[1] / data.M << std::endl;
    std::cout << "Relative Error: " << results[0] / data.M << std::endl;
}