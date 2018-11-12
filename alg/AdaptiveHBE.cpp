//
// Created by Kexin Rong on 2018-11-11.
//

#include "AdaptiveHBE.h"

void AdaptiveHBE::buildLevels(shared_ptr<MatrixXd> X, shared_ptr<Kernel> k, double tau, double eps) {
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

    for (int i = 0; i < I; i ++) {
        if (i == 0) {
            mui[i] = (1 - gamma);
        } else {
            mui[i] = (1 - gamma) * mui[i - 1];
        }
        ti[i] = sqrt(log(1 / mui[i]));
        ki[i] = (int) (3 * ceil(r * ti[i]));
        wi[i] = ki[i] / ti[i] * SQRT_2PI;
        Mi[i] = (int) (ceil(k->RelVar(mui[i]) / eps / eps));

        int t = int(Mi[i] * 1.1);
        levels.push_back(SketchLSH(X, t, wi[i], ki[i], k, max(1, t/200)));
//        std::cout << "Level " << i << ", samples " << Mi[i] <<
//                  ", target: "<< mui[i] << ", k:" << ki[i] << ", w:" << wi[i] << std::endl;
    }
}

AdaptiveHBE::AdaptiveHBE(shared_ptr<MatrixXd> data, shared_ptr<Kernel> k, double lb, double eps) {
    tau = lb;
    buildLevels(data, k, tau, eps);
}


std::vector<double> AdaptiveHBE::evaluateQuery(VectorXd q, int l, int maxSamples) {
    std::vector<double> results = std::vector<double>(2, 0);
    results[1] = min(maxSamples, Mi[l]);
    results[0] = levels[l].query(q, tau, results[1]);
    return results;
}
