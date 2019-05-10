#include "AdaptiveHBE.h"
#include "dataUtils.h"
#include "SketchTable.h"

void AdaptiveHBE::buildLevels(shared_ptr<MatrixXd> X, shared_ptr<Kernel> k, double tau, double eps,
        bool sketch) {
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
    int N_SKETCHES = 5;
    int samples = int(sqrt(n));
    double diam = dataUtils::estimateDiameter(X, tau);
    double exp_k = dataUtils::getPower(diam, 0.5);
    double exp_w = dataUtils::getWidth(exp_k, 0.5);

    //Will be used to obtain a seed for the random number engine
    std::random_device rd;
    std::mt19937_64 rng(rd());

    int ntables = 0;
    for (int i = 0; i < I; i ++) {
        if (i == 0) {
            mui[i] = (1 - gamma);
//            mui[i] = 0.4;
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
        Mi[i] = (int) (ceil(k->RelVar(mui[i]) / eps / eps));
        ntables += Mi[i];
    }

    if (sketch) { // HBS
        int m = min(ntables * samples, 2*n);
        vector<SketchTable> sketches;
        vector<vector<int>> indices;
        for (int i = 0; i < N_SKETCHES; i ++ ){
            std::vector<int> idx;
            shared_ptr<MatrixXd> X1 = dataUtils::downSample(X, idx, m/N_SKETCHES, rng);
            sketches.push_back(SketchTable(X1, wi[I-1], ki[I-1], rng));
            indices.push_back(idx);
        }

        for (int i = 0; i < I; i ++) {
            int t = int(Mi[i] * L * 1.1);
            s_levels.push_back(SketchLSH(X, sketches, indices, t, wi[i], ki[i], k, rng));
        }
    } else { // Uniform
        for (int i = 0; i < I; i ++) {
            int t = int(Mi[i] * L * 1.1);
            b_levels.push_back(BaseLSH(X, t, wi[i], ki[i], k, samples));
        }
    }
}

AdaptiveHBE::AdaptiveHBE(shared_ptr<MatrixXd> data, shared_ptr<Kernel> k, double lb,
        double eps, bool sketch) {
    numPoints = data->rows();
    tau = lb;
    use_sketch = sketch;
    buildLevels(data, k, tau, eps, sketch);
}


std::vector<double> AdaptiveHBE::evaluateQuery(VectorXd q, int l) {
    // MoM
    std::vector<double> results = std::vector<double>(2, 0);
    results[1] = Mi[l];

    std::vector<double> Z = std::vector<double>(L, 0);
    for (int i = 0; i < L; i ++) {
        if (use_sketch) {
            Z[i] = s_levels[l].query(q, tau, results[1]);
        } else {
            Z[i] = b_levels[l].query(q, tau, results[1]);
        }
    }

    results[0] = mathUtils::median(Z);
    results[1] *= L;
    return results;
}

