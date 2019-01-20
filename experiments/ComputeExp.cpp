//
// Created by Kexin Rong on 1/14/19.
//
#include <iostream>
#include <sstream>
#include <fstream>
#include "dataUtils.h"
#include "parseConfig.h"
#include "expkernel.h"
#include "gaussiankernel.h"
#include "mathUtils.h"
#include "dataUtils.h"
#include "bandwidth.h"
#include <chrono>

int main(int argc, char *argv[]) {
    if (argc < 2) {
        std::cout << "Need config file" << std::endl;
        exit(1);
    }

    char array[] = "gaussian";
    char* scope = array;
    parseConfig cfg(argv[1], scope);
    // The dimensionality of each sample vector.
    int d = cfg.getDim();
    // The number of sources which will be used for the gauss transform.
    int N = cfg.getN();
    int M = min(cfg.getM(), 10000);

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

    // Get Gaussian Exact
    double *exact = new double[M * 2];
    dataUtils::readFile(cfg.getExactPath(), false, M, 0, 1, &exact[0]);
    std::cout << exact[0] << "," << exact[1] << std::endl;

    cfg = parseConfig(argv[1], "exp");

    double *g = new double[M];

    // The bandwidth.
    double h = cfg.getH();
    if (!cfg.isConst()) {
        h *= pow(N, -1.0/(d+4));
    }
    std::cout << "Bandwidth: " << h << std::endl;

    auto t1 = std::chrono::high_resolution_clock::now();
    std::ofstream outfile(cfg.getExactPath());
    double hSquare = h * h;
    for(int j = 0; j < M; j++) {
        int idx = exact[j*2+1];
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
            // exp kernel
            g[j] = g[j] + exp(-sqrt(norm/hSquare));
        }
        g[j] /= N;
        if (j < 20) {
            std::cout << exact[j*2] << "," << g[j] << std::endl;
        }
        outfile << g[j] << "," << idx << "\n";
    }
    outfile.close();
    auto t2 = std::chrono::high_resolution_clock::now();

    std::cout << M << " queries: " << std::chrono::duration_cast<std::chrono::milliseconds>(t2-t1).count() << std::endl;

}