//
// Created by Kexin Rong on 10/12/18.
//


#include "expkernel.h"
#include "mathUtils.h"
#include "dataUtils.h"
#include "math.h"
#include "kernel.h"
#include "expkernel.h"
#include "gaussiankernel.h"
#include "bandwidth.h"
#include "../data/SyntheticData.h"
#include <chrono>

using Eigen::IOFormat;

const int uN = 50;
const int uC = 4000;
const int cN = 50000;
const int cC = 4;
const int scales = 4;
const int dim = 3;
const double spread = 0.01;
const double mu = 1.0 / 1000;

const int iterations = 100000;

int dims[] = {3, 5, 8, 10, 15, 20, 30, 50, 100};

int main() {
    for (int i = 0; i < sizeof(dims)/sizeof(dims[0]); i ++) {
        std::cout << "-------------------------------------------------------" << std::endl;
        int dim = dims[i];
        shared_ptr<Kernel> kernel = make_shared<Gaussiankernel>(dim);
        GenericInstance data = SyntheticData::genMixed(uN, cN, uC, cC, dim, mu, scales, spread, kernel);
//        data.output("resources/data/generic_gaussian" + std::to_string(dim) + ".txt");
//        int n = data.points.rows();
//        std::cout <<  "N=" << n << ", d=" << dim << std::endl;
        std::ofstream outfile("resources/data/query"+ std::to_string(dim) + ".txt");
        for (int i = 0; i < iterations; i ++) {
            VectorXd q = data.query(0.5, false);
            for (int j = 0; j < dim - 1; j ++) {
                outfile << q[j] << ",";
            }
            outfile << q[dim-1] << "\n";
        }
        outfile.close();
    }

}