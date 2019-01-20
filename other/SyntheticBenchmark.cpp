//
// Created by Kexin Rong on 10/14/18.
//

#include "../data/SyntheticData.h"
#include "expkernel.h"
#include "gaussiankernel.h"
#include "mathUtils.h"
#include "dataUtils.h"
#include "bandwidth.h"
#include "math.h"
#include "../alg/RS.h"
#include "../alg/naiveKDE.h"
#include "../alg/BaseLSH.h"
#include <chrono>

const double eps = 0.5;
const double beta = 0.5;
const double tau = 1e-3;

int dims[] = {3, 5, 8, 10, 15, 20, 30, 50, 100};
//int N = 428816;
int N = 408935;
int M = 100000;
//int N = 434188;

int main() {
    for (size_t i = 0; i < sizeof(dims)/sizeof(dims[0]); i ++) {
        int dim = dims[i];
        std::cout << "-------------------------------------------------------" << std::endl;
        std::string fname = "resources/data/generic_gaussian" + std::to_string(dim) + ".txt";
        MatrixXd X = dataUtils::readFile(fname, false, N, 0, dim - 1);
        std::cout << "d=" << dim << std::endl;

        // Bandwidth
        auto band = make_unique<Bandwidth>(N, dim);
        //band->useConstant(pow(N, -1.0/(dim+4)) * 2);
        shared_ptr<Kernel> simpleKernel;
        simpleKernel = make_shared<Gaussiankernel>(dim);
        X = dataUtils::normalizeBandwidth(X, band->bw);
        shared_ptr<MatrixXd> X_ptr = make_shared<MatrixXd>(X);

        RS rs(X_ptr, simpleKernel);

        double means = ceil(1.5 * mathUtils::gaussRelVar(tau) / eps / eps);
        int M = (int)(means);
        double w = 3 * sqrt(log(1/tau));
        int k = dataUtils::getPowerW(w, beta);
//        double diam = dataUtils::estimateDiameter(X, tau);
//        int k = dataUtils::getPower(diam, beta);
//        double w = dataUtils::getWidth(k, beta);
        std::cout << "M=" << M << ",w=" << w << ",k=" << k << std::endl;
        auto t1 = std::chrono::high_resolution_clock::now();
        BaseLSH hbe(X_ptr, M, w, k, 1, simpleKernel, 1);
        auto t2 = std::chrono::high_resolution_clock::now();
        std::cout << "HBE init took: " << std::chrono::duration_cast<std::chrono::seconds>(t2-t1).count() << std::endl;

        vector<double> error = vector<double>(2, 0);
        int m1 = (int) (1 / eps / eps * mathUtils::randomRelVar(tau));
        int m2 = (int) (1 / eps / eps * mathUtils::gaussRelVar(tau));
        std::cout << "RS samples:" << m1 << ", HBE samples:" << m2 << std::endl;

        fname = "resources/exact/generic" + std::to_string(dims[i]) + "_query10k.txt";
        MatrixXd exact = dataUtils::readFile(fname, false, M, 0, 0);

        naiveKDE naive(X_ptr, simpleKernel);
        fname = "resources/data/query" + std::to_string(dim) + ".txt";
        MatrixXd queries = dataUtils::readFile(fname, false, M, 0, dim-1);
        for (int j = 0; j < M; j++) {
            VectorXd q = queries.row(j);
            double kde = exact(j, 0);

            // RS
            double rsKDE = rs.query(q, tau, m1);
            error[0] += fabs(kde - rsKDE) / kde;

            // HBE
            double hbeKDE = hbe.query(q, tau, m2);
            error[1] += fabs(kde - hbeKDE) / kde;
        }
        std::cout << "RS total time: " << rs.totalTime / 1e9 << std::endl;
        std::cout << "RS average error: " << error[0] / M << std::endl;
        std::cout << "HBE total time: " << hbe.totalTime / 1e9 << std::endl;
        std::cout << "HBE average error: " << error[1] / M << std::endl;
        std::cout << "-------------------------------------------------------" << std::endl;
    }
}
