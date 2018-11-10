//
// Created by Kexin Rong on 2018-11-09.
//

#include "AdaptiveRS.h"

void AdaptiveRS::buildLevels(double tau, double eps) {
    double tmp = log(1/ tau);
    // Effective diameter
    r = sqrt(tmp);
    // # guess
    I = (int) ceil(tmp / LOG2);
    mui = vector<double>(I);
    ti = vector<double>(I);
    ki = vector<int>(I);
    wi = vector<double>(I);
    Mi = vector<int>(I);

    for (int i = 0; i < I; i ++) {
        if (i == 0) {
            mui[i] = (1 - gamma);
        } else {
            mui[i] = (1 - gamma) * mui[i - 1];
        }
        ti[i] = sqrt(log(1 / mui[i]));
        ki[i] = (int) (3 * ceil(r * ti[i]));
        wi[i] = ki[i] / ti[i] * SQRT_2PI;
        Mi[i] = (int) (3 * ceil(mathUtils::randomRelVar(mui[i]) / eps / eps));
        std::cout << "Level " << i << ", samples " << Mi[i] <<
            ", target: "<< mui[i] << std::endl;
    }
}

AdaptiveRS::AdaptiveRS(shared_ptr<MatrixXd> data, shared_ptr<Kernel> k, double tau, double eps) {
    X = data;
    kernel = k;
    numPoints = data->rows();
    buildLevels(tau, eps);
}


AdaptiveRS::AdaptiveRS(shared_ptr<MatrixXd> data, shared_ptr<Kernel> k, int samples,
        double tau, double eps) {
    int n = data->rows();
    if (samples >= n) {
        X = data;
        kernel = k;
        numPoints = data->rows();
        buildLevels(tau, eps);
        return;
    }
    // Sample input matrix
    std::random_device rd;
    std::mt19937 gen(rd());
    auto indices = mathUtils::pickSet(n, samples, gen);
    X = make_shared<MatrixXd>(samples, data->cols());
    int i = 0;
    for (auto idx: indices) {
        X->row(i) = data->row(idx);
        i ++;
    }
    kernel = k;
    numPoints = samples;
    buildLevels(tau, eps);
}


std::vector<double> AdaptiveRS::evaluateQuery(VectorXd q, int level, int maxSamples) {
    std::random_device rd;  //Will be used to obtain a seed for the random number engine
    std::mt19937_64 rng = std::mt19937_64(rd());
    std::uniform_int_distribution<int> distribution(0, numPoints - 1);

    std::vector<double> results = std::vector<double>(2, 0);
    results[1] = min(maxSamples, Mi[level]);

    std::vector<int> indices(results[1]);
    for (int j = 0; j < results[1]; j ++) {
        indices[j] = distribution(rng);
    }
    std::sort(indices.begin(), indices.end());
    for (int j = 0; j < results[1]; j ++) {
        int idx = indices[j];
        results[0] += kernel->density(q, X->row(idx));
    }
    return results;
}