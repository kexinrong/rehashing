#include "HybridAdaptive.h"
#include "dataUtils.h"

void HybridAdaptive::buildLevels(shared_ptr<MatrixXd> X, shared_ptr<Kernel> k, double tau, double eps) {
    double tmp = log(1/ tau);
    // Effective diameter
    r = sqrt(tmp);
    // # guess
    I = (int) ceil(tmp / LOG2);
    mui = vector<double>(I);
    Mi = vector<int>(I);
    ti = vector<double>(I);
    ki = vector<int>(I);
    wi = vector<double>(I);
    int n = X->rows();
    double diam = dataUtils::estimateDiameter(X, tau);
    double exp_k = dataUtils::getPower(diam, 0.5);
    double exp_w = dataUtils::getWidth(exp_k, 0.5);

    for (int i = 0; i < I; i ++) {
        if (i == 0) {
            mui[i] = (1 - gamma);
        } else {
            mui[i] = (1 - gamma) * mui[i - 1];
        }

        // Exponential Kernel
        if (k->getName() == EXP_STR) {
            ki[i] = exp_k;
            wi[i] = exp_w;
        } else {         // Gaussian Kernel
            ti[i] = sqrt(log(1 / mui[i]));
            ki[i] = (int) (3 * ceil(r * ti[i]));
            wi[i] = ki[i] / ti[i] * SQRT_2PI;
        }

        if (1 / sqrt(mui[i]) < 10 && i > 0) {         // Random layer
            random.push_back(true);
            Mi[i] = (int) (ceil(mathUtils::randomRelVar(mui[i]) / eps / eps));
            levels.push_back(BaseLSH());
//            std::cout << "Level " << i << ", samples " << Mi[i] <<
//                      ", target: "<< mui[i] << " (RS)" << std::endl;
        } else { // HBE layer
            random.push_back(false);
            Mi[i] = (int) (ceil(k->RelVar(mui[i]) / eps / eps));
            int t = int(Mi[i] * L * 1.1);
            int samples = int(sqrt(n));
            levels.push_back(BaseLSH(X, t, wi[i], ki[i], k, samples));
//            std::cout << "Level " << i << ", samples " << Mi[i] <<
//                  ", target: "<< mui[i] << " (HBE)" << std::endl;
        }

    }
}

HybridAdaptive::HybridAdaptive(shared_ptr<MatrixXd> data, shared_ptr<Kernel> k, double lb, double eps) {
    std::random_device rd;  //Will be used to obtain a seed for the random number engine
    rng = std::mt19937_64(rd());

    tau = lb;
    X = data;
    kernel = k;
    numPoints = data->rows();
    buildLevels(data, k, tau, eps);
}


std::vector<double> HybridAdaptive::evaluateQuery(VectorXd q, int l) {
    // MoM
    std::vector<double> results = std::vector<double>(2, 0);
    results[1] =  Mi[l];

    std::vector<double> Z = std::vector<double>(L, 0);
    std::uniform_int_distribution<int> distribution(0, numPoints - 1);

    for (int i = 0; i < L; i ++) {
        if (random[l]) {
            std::vector<int> indices(results[1]);
            for (int j = 0; j < results[1]; j ++) {
                indices[j] = distribution(rng);
            }
            std::sort(indices.begin(), indices.end());
            for (int j = 0; j < results[1]; j ++) {
                int idx = indices[j];
                double d = kernel->density(q, X->row(idx));
                Z[i] += d;
            }
            Z[i] /= results[1];

        } else {
            Z[i] =  levels[l].query(q, tau, results[1]);
        }
    }

    results[0] = mathUtils::median(Z);
    return results;
}

