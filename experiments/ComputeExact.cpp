//
// Created by Kexin Rong on 9/21/18.
//
#include <iostream>
#include <sstream>
#include <fstream>
#include "dataUtils.h"
#include "parseConfig.h"
#include <chrono>

int main(int argc, char *argv[]) {
    if (argc < 2) {
        std::cout << "Need config file" << std::endl;
        exit(1);
    }

    char* scope = argv[2];
    parseConfig cfg(argv[1], scope);
    // The dimensionality of each sample vector.
    int d = cfg.getDim();
    // The number of sources which will be used for the gauss transform.
    int N = cfg.getN();
    int M = cfg.getM();

    // The bandwidth.
    double h = cfg.getH();
    if (!cfg.isConst()) {
        if (strcmp(scope, "exp") == 0) {
            h *= pow(N, -1.0/(d+4));
        } else {
            h *= sqrt(2);
        }
    }


    // Read input
    double *x = new double[N * d];
    dataUtils::readFile(cfg.getDataFile(), cfg.ignoreHeader(), N,
            cfg.getStartCol(), cfg.getEndCol(), &x[0]);

    int hasQuery = strcmp(cfg.getDataFile(), cfg.getQueryFile());

    double *y;
    if (hasQuery != 0) {
        y = new double[M * d];
        dataUtils::readFile(cfg.getQueryFile(), cfg.ignoreHeader(), M,
                            cfg.getStartCol(), cfg.getEndCol(), &y[0]);
    }

    double * g = new double[M];

    // Random init
    std::random_device rd;  //Will be used to obtain a seed for the random number engine
    std::mt19937_64 rng = std::mt19937_64(rd());
    std::uniform_int_distribution<int> distribution(0, N - 1);

    auto t1 = std::chrono::high_resolution_clock::now();
    std::ofstream outfile(cfg.getExactPath());
    double hSquare = h * h;
    for(int j = 0; j < M; j++) {
        int idx = j;
        if (M < N && hasQuery == 0) {
            idx = distribution(rng);
        }
        g[j] = 0.0;
        for(int i = 0; i < N; i++) {
            double norm = 0.0;
            for (int k = 0; k < d; k++) {
                double temp;
                if (hasQuery != 0) {
                    temp = x[(d*i) + k] - y[(d*idx) + k];
                } else {
                    temp = x[(d*i) + k] - x[(d*idx) + k];
                }
                norm = norm + (temp*temp);
            }
            if (strcmp(scope, "gaussian") == 0) {
                g[j] = g[j] + exp(-norm/hSquare);
            } else { // exp kernel
                g[j] = g[j] + exp(-sqrt(norm/hSquare));
            }
        }
        g[j] /= N;
        outfile << g[j] << "," << idx << "\n";
    }
    outfile.close();
    auto t2 = std::chrono::high_resolution_clock::now();

    std::cout << M << " queries: " << std::chrono::duration_cast<std::chrono::milliseconds>(t2-t1).count() << std::endl;

}