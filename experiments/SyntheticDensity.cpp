//
// Created by Kexin Rong on 10/14/18.
//

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
const double tau = 5e-5;

int dims[] = {3, 5, 8, 10, 15, 20, 30, 50, 100};
int N = 408935;
int M = 100000;

int main(int argc, char *argv[]) {
    int dim = atoi(argv[1]);
    std::cout << "-------------------------------------------------------" << std::endl;
    std::string fname = "resources/data/generic_gaussian" + std::to_string(dim) + ".txt";
    MatrixXd X = dataUtils::readFile(fname, false, N, 0, dim - 1);
    std::cout << "d=" << dim << std::endl;

    // Bandwidth
    auto band = make_unique<Bandwidth>(N, dim);
    //band->useConstant(pow(N, -1.0/(dim+4)) * 2);
    std::cout << band->bw[0] << std::endl;
    shared_ptr<Kernel> simpleKernel;
    simpleKernel = make_shared<Gaussiankernel>(dim);
    X = dataUtils::normalizeBandwidth(X, band->bw);
    shared_ptr<MatrixXd> X_ptr = make_shared<MatrixXd>(X);

    fname = "resources/data/query" + std::to_string(dim) + ".txt";
    MatrixXd queries = dataUtils::readFile(fname, false, M, 0, dim-1);

    naiveKDE naive(X_ptr, simpleKernel);
    fname = "resources/exact/generic" + std::to_string(dim) + "_query10k.txt";
    std::ofstream myfile(fname);
    for (int j = 0; j < M; j++) {
        VectorXd q = queries.row(j);
        double kde = naive.query(q);
        myfile<<kde << "\n";
    }
    myfile.close();
    std::cout << "-------------------------------------------------------" << std::endl;
}
