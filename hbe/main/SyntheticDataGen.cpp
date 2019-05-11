/*
 *  Main program for generating "worst-case" and "D-strucutured" instances.
 *
 */


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

const int uN = 100;         // #pts (per cluster) for "uncorrelated"
const int uC = 5000;        // #clusters for "uncorrelated"
const int cN = 50000;       // #pts (per cluster) for "correlated"
const int cC = 10;          // #clusters for "correlated"
const int scales = 4;
const int dim = 10;
const double spread = 0.001;
const double mu = 1.0 / 1000;

const int nqueries = 10000;
const int N = 1000000;

//int dims[] = {5, 10, 20, 50, 100, 200, 500};
int clusters[] = {1, 10, 100, 1000, 10000, 100000};

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
//    for (size_t i = 0; i < sizeof(dims)/sizeof(dims[0]); i ++) {
    for (size_t i = 0; i < sizeof(clusters)/sizeof(clusters[0]); i ++) {

        std::cout << "-------------------------------------------------------" << std::endl;
        int dim = 100;
        //int dim = dims[i];
        std::cout << "dim = " << dim << std::endl;
        int k = clusters[i];
        int n = N / k;
        std::cout << "#pts (per cluster) = " << n << ", #clusters = " << k << std::endl;
        shared_ptr<Kernel> kernel = make_shared<Gaussiankernel>(dim);
        // Generate an instance
        GenericInstance data = SyntheticData::genSingle(n, k, dim, mu, scales, spread, kernel);
        // Generate a mixed instance
        //GenericInstance data = SyntheticData::genMixed(uN, cN, uC, cC, dim, mu, scales, spread, kernel);
        data.output("../resources/data/generic_" +
            std::to_string(k) + ".txt");
        std::cout <<  "N=" << data.points.rows() << std::endl;

        // Output query
        std::ofstream outfile("../resources/data/query_" +
            std::to_string(k) + ".txt");
        genQuery(outfile, data, dim);
        outfile.close();
    }

}