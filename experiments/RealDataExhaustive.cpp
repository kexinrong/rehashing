//
// Created by Kexin Rong on 9/21/18.
//

#include "gaussiankernel.h"
#include "mathUtils.h"
#include "dataUtils.h"
#include "bandwidth.h"
#include "math.h"
#include <iostream>
#include <fstream>
#include "../alg/naiveKDE.h"
#include <chrono>

int main() {
    MatrixXd X = dataUtils::readFile("resources/shuttle_normed.csv", true, 43500, 9);
    int n = X.rows();
    int dim = X.cols();
    std::cout << "N=" << n << ", d=" << dim << std::endl;

    // Algorithms init
    shared_ptr<Kernel> simpleKernel = make_shared<Gaussiankernel>(dim);
    shared_ptr<MatrixXd> X_ptr = make_shared<MatrixXd>(X);
    naiveKDE naive(X_ptr, simpleKernel);

    ofstream outfile;
    outfile.open ("shuttle_gaussian.txt");
    double t = 0;
    for (int idx = 0; idx < n; idx++) {
        VectorXd q = X.row(idx);
        // Naive
        auto t1 = std::chrono::high_resolution_clock::now();
        double kde = naive.query(q);
        auto t2 = std::chrono::high_resolution_clock::now();
        t += std::chrono::duration_cast<std::chrono::milliseconds>(t2-t1).count();
        outfile << kde << "\n";
    }
    outfile.close();
    std::cout << "Total time: " << t / 1e6 << " sec";
}