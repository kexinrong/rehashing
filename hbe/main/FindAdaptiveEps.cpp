/*
 *  Parameter search for adaptive sampling:
 *      This program finds the smallest epsilon for random sampling and HBE that achieves a true relative error < 0.1.
 *      Epsilon is a parameter that controls error in the adaptive sampling procedure.
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

void findEps(bool isRandom, DataIngest& data) {
    double eps = 0.6;
    bool stop = false;
    bool times = false;

    double head = 0;
    double tail = 1;

    shared_ptr<AdaptiveEstimator> est;
    while (!stop) {
        std::cout << "------------------" << std::endl;
        std::cout << "eps = " << eps << std::endl;
        if (isRandom) {
            est = make_shared<AdaptiveRS>(data.X_ptr, data.kernel, data.tau, eps);
        } else {
            auto t1 = std::chrono::high_resolution_clock::now();
            est = make_shared<AdaptiveHBE>(data.X_ptr,  data.kernel, data.tau, eps, true);
            auto t2 = std::chrono::high_resolution_clock::now();
            std::cout << "Adaptive Table Init: " <<
                      std::chrono::duration_cast<std::chrono::seconds>(t2-t1).count() << std::endl;
        }
        est->totalTime = 0;
        vector<double> results(2,0);

        for (int j = 0; j < data.M; j++) {
            int idx = j * 2;
            VectorXd q = data.X_ptr->row(j);
            if (data.hasQuery != 0) {
                q = data.Y_ptr->row(j);
                idx = j;
            } else {
                if (!data.sequential) {
                    q = data.X_ptr->row(int(data.exact[idx + 1]));
                }
            }

            vector<double> vals = est->query(q);
            update(results, vals, data.exact[idx]);
        }

        std::cout << "Sampling total time: " << est->totalTime / 1e9 << std::endl;
        std::cout << "Average Samples: " << results[1] / data.M << std::endl;
        std::cout << "Relative Error: " << results[0] / data.M << std::endl;

        double err = results[0] / data.M;
        if (eps == 0.6) {
            if (err < 0.1) {
                times = true;
                eps *= 2;
            } else {
                eps /= 2;
            }
        } else {
            if (times) {
                if (err > 0.1) {
                    stop = true;
                    head = eps / 2;
                    tail = eps;
                    break;
                }
                eps *= 2;
            } else {
                if (err < 0.1) {
                    stop = true;
                    head = eps;
                    tail = eps * 2;
                    break;
                }
                eps /= 2;
            }
        }
    }

    std::cout << "Binary search: ["<< head << "," << tail << "]\n";
    while (true) {
        eps = (head + tail) / 2;

        std::cout << "------------------" << std::endl;
        std::cout << "eps = " << eps << std::endl;
        if (isRandom) {
            est = make_shared<AdaptiveRS>(data.X_ptr, data.kernel, data.tau, eps);
        } else {
            auto t1 = std::chrono::high_resolution_clock::now();
            est = make_shared<AdaptiveHBE>(data.X_ptr, data.kernel, data.tau, eps, true);
            auto t2 = std::chrono::high_resolution_clock::now();
            std::cout << "Adaptive Table Init: " <<
                      std::chrono::duration_cast<std::chrono::seconds>(t2-t1).count() << std::endl;
        }
        est->totalTime = 0;
        vector<double> results(2,0);

        for (int j = 0; j < data.M; j++) {
            int idx = j * 2;
            VectorXd q = data.X_ptr->row(j);
            if (data.hasQuery != 0) {
                q = data.Y_ptr->row(j);
                idx = j;
            } else {
                if (!data.sequential) {
                    q = data.X_ptr->row(int(data.exact[idx + 1]));
                }
            }
            vector<double> vals = est->query(q);
            update(results, vals, data.exact[idx]);
        }

        std::cout << "Sampling total time: " << est->totalTime / 1e9 << std::endl;
        std::cout << "Average Samples: " << results[1] / data.M << std::endl;
        std::cout << "Relative Error: " << results[0] / data.M << std::endl;

        double err = results[0] / data.M;
        if (err < 0.11 && err > 0.09) {
            break;
        } else if (err > 0.1) {
            tail = eps;
        } else {
            head = eps;
        }
    }

}

int main(int argc, char *argv[]) {
    if (argc < 2) {
        std::cout << "Need config file" << std::endl;
        exit(1);
    }

    char *scope = argv[2];
    parseConfig cfg(argv[1], scope);
    DataIngest data(cfg, true);

    std::cout << "RS\n";
    findEps(true, data);

    std::cout << "======================================\n";

    std::cout << "HBE\n";
    findEps(false, data);

}


