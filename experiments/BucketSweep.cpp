//
// Created by Kexin Rong on 10/3/18.
//

#include <iostream>
#include <sstream>
#include <fstream>
#include "bandwidth.h"
#include "expkernel.h"
#include "gaussiankernel.h"
#include "dataUtils.h"
#include "parseConfig.h"
#include "../alg/BaseLSH.h"
#include "../alg/SketchLSH.h"

int main(int argc, char *argv[]) {
    if (argc < 2) {
        std::cout << "Need config file" << std::endl;
        exit(1);
    }

    char* scope = argv[2];
    parseConfig cfg(argv[1], scope);
    const double eps = cfg.getEps();
    const double tau = cfg.getTau();
    const double beta = cfg.getBeta();
    // The dimensionality of each sample vector.
    int dim = cfg.getDim();
    // The number of sources which will be used for the gauss transform.
    int N = cfg.getN();
    // The bandwidth.  NOTE: this is not the same as standard deviation since
    // the Gauss Transform sums terms exp( -||x_i - y_j||^2 / h^2 ) as opposed
    // to  exp( -||x_i - y_j||^2 / (2*sigma^2) ).  Thus, if sigma is known,
    // bandwidth can be set to h = sqrt(2)*sigma.
    double h = cfg.getH();
    if (!cfg.isConst()) {
        if (strcmp(scope, "exp") == 0) {
            h *= pow(N, -1.0/(dim+4));
        } else {
            h *= sqrt(2);
        }
    }

    MatrixXd X = dataUtils::readFile(
            cfg.getDataFile(), cfg.ignoreHeader(), N, cfg.getStartCol(), cfg.getEndCol());

    auto band = make_unique<Bandwidth>(N, dim);
    band->useConstant(h);
    shared_ptr<Kernel> kernel;
    double means = 0;
    shared_ptr<Kernel> simpleKernel;
    if (strcmp(scope, "gaussian") == 0) {
        kernel = make_shared<Gaussiankernel>(dim);
        simpleKernel = make_shared<Gaussiankernel>(dim);
        means = ceil(6 * mathUtils::gaussRelVar(tau) / eps / eps);
    } else {
        kernel = make_shared<Expkernel>(dim);
        simpleKernel = make_shared<Expkernel>(dim);
        means = ceil(6 * mathUtils::expRelVar(tau) / eps / eps);
    }
    kernel->initialize(band->bw);
//    dataUtils::checkBandwidthSamples(X, eps, kernel);
    // Normalized by bandwidth
    X = dataUtils::normalizeBandwidth(X, band->bw);
    shared_ptr<MatrixXd> X_ptr = make_shared<MatrixXd>(X);

    // Estimate parameters
    int tables = (int)(means * 1.1);
    tables = 1000;
    double w = 3 * log(1/tau);
//    double w = 3;
    int k = dataUtils::getPowerW(w, beta);
//    double diam = dataUtils::estimateDiameter(X, tau);
//    int k = dataUtils::getPower(diam, beta);
//    double w = dataUtils::getWidth(k, beta);

    // Algorithms init
    std::cout << "M=" << tables << ",w=" << w << ",k=" << k << std::endl;
    auto t1 = std::chrono::high_resolution_clock::now();
    int subsample = int(sqrt(N));
//    int subsample = 1;
//    BaseLSH hbe(X_ptr, tables, w, k, 1, simpleKernel, subsample);
    SketchLSH hbe(X_ptr, tables, w, k, 1, simpleKernel);
    auto t2 = std::chrono::high_resolution_clock::now();
    std::cout << "HBE Table Init: " << std::chrono::duration_cast<std::chrono::seconds>(t2-t1).count() << std::endl;

    std::ofstream outfile(argv[3]);
    std::map<int,int> cluster;
    std::map<int, int>::iterator iter;
    int count = 0;
    for (auto &t : hbe.tables) {
        int max_bucket = 0;
        for (auto &it : t.table) {
            max_bucket = std::max(max_bucket, it.second.count);
            outfile << it.second.count << ",";
        }
        outfile << " \n";
//        for (int i = 0; i < N; i ++) {
//            vector<HashBucket> buckets = t.sample(X.row(i));
//            if (buckets[0].count == max_bucket) {
//                if (count == 0) {
//                    cluster[i] = 1;
//                } else {
//                    iter = cluster.find(i);
//                    if (iter != cluster.end()) {
//                        iter->second += 1;
//                    } else {
//                        cluster[i] = 1;
//                    }
//                }
//            }
//        }
//        count += 1;
    }

//    for (auto &it : cluster) {
//        if (it.second > 5) {
//            outfile << it.first << "," << it.second << "\n";
//        }
//    }

    outfile.close();

}