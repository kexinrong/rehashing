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
const int dim = 10;
const double spread = 0.01;
const double mu = 1.0 / 1000;

const int nqueries = 100000;
const int N = 500000;

int dims[] = {3, 5, 8, 10, 15, 20, 30, 50, 100};
int clusters[] = {1, 10, 100, 1000, 10000, 100000, 500000};


void genQuery(std::ofstream &outfile, GenericInstance &data, int dim) {
    for (int i = 0; i < nqueries; i ++) {
        VectorXd q = data.query(0.5, false);
        for (int j = 0; j < dim - 1; j ++) {
            outfile << q[j] << ",";
        }
        outfile << q[dim-1] << "\n";
    }
}

int main() {
    //for (size_t i = 0; i < sizeof(dims)/sizeof(dims[0]); i ++) {
    for (size_t i = 0; i < sizeof(clusters)/sizeof(clusters[0]); i ++) {

        std::cout << "-------------------------------------------------------" << std::endl;
//        int dim = dims[i];
        int k = clusters[i];
        int n = N / k;
        std::cout << "n = " << n << ", k = " << k << std::endl;
        shared_ptr<Kernel> kernel = make_shared<Gaussiankernel>(dim);
        // Output instance
        GenericInstance data = SyntheticData::genSingle(n, k, dim, mu, scales, spread, kernel);
//        GenericInstance data = SyntheticData::genMixed(uN, cN, uC, cC, dim, mu, scales, spread, kernel);
        data.output("resources/generic_gaussian/data_" +
            std::to_string(k) + "," + std::to_string(n) + ".txt");
        std::cout <<  "N=" << data.points.rows() << std::endl;

        // Output query
        std::ofstream outfile("resources/generic_gaussian/query" +
            std::to_string(k) + "," + std::to_string(n) + ".txt");
        genQuery(outfile, data, dim);
        outfile.close();
    }

}