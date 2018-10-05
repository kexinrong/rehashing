//
// Created by Kexin Rong on 10/3/18.
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

    std::string scope = "exp";
    parseConfig cfg(argv[1], scope.c_str());
    // The dimensionality of each sample vector.
    int d = cfg.getDim();
    // The number of sources which will be used for the gauss transform.
    int N = cfg.getN();
    int samples = N;
    int M = 1000;

    // Read input
    double *x = new double[N * d];
    dataUtils::readFile(cfg.getDataFile(), cfg.ignoreHeader(), N,
                        cfg.getStartCol(), cfg.getEndCol(), &x[0]);

    std::ofstream outfile("resources/" + std::string(argv[2]));
    srand(time(NULL));
    for(int j = 0; j < M; j++) {
        for(int i = 0; i < samples; i++) {
            //int idx = rand() % N;
            int idx = i;
            double norm = 0.0;
            for (int k = 0; k < d; k++) {
                double temp = x[(d*idx) + k] - x[(d*j) + k];
                norm = norm + (temp*temp);
            }
            outfile << sqrt(norm);
            if (i < samples - 1) {
                outfile << ",";
            }
        }
        outfile << "\n";

    }
    outfile.close();

}