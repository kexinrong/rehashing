//
// Created by Kexin Rong on 9/21/18.
//
#include <iostream>
#include <sstream>
#include <fstream>
#include "dataUtils.h"
#include "parseConfig.h"

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
    int M = 100000;
    // The bandwidth.
    double h = cfg.getH();
    if (strcmp(scope, "exp") == 0) {
        h *= pow(N, -1.0/(d+4));
    } else {
        h *= sqrt(2);
    }

    // Read input
    double *x = new double[N * d];
    dataUtils::readFile(cfg.getDataFile(), cfg.ignoreHeader(), N,
            cfg.getStartCol(), cfg.getEndCol(), &x[0]);
    double *y = new double[M * d];
    for (int i = 0; i < M * d; i ++) { y[i] = x[i]; }
    double * g = new double[M];

    std::ofstream outfile(cfg.getExactPath());
    double hSquare = h * h;
    for(int j = 0; j < M; j++) {
        g[j] = 0.0;
        for(int i = 0; i < N; i++) {
            double norm = 0.0;
            for (int k = 0; k < d; k++) {
                double temp = x[(d*i) + k] - y[(d*j) + k];
                norm = norm + (temp*temp);
            }
            if (strcmp(scope, "gaussian") == 0) {
                g[j] = g[j] + exp(-norm/hSquare);
            } else { // exp kernel
                g[j] = g[j] + exp(-sqrt(norm/hSquare));
            }
        }
        g[j] /= N;
        outfile << g[j] << "\n";
    }
    outfile.close();
}