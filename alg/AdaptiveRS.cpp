#include "AdaptiveRS.h"
#include "dataUtils.h"
#include "mathUtils.h"
#include <math.h>

void AdaptiveRS::buildLevels(double tau, double eps) {
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
            //mui[i] = 0.4;
        } else {
            mui[i] = (1 - gamma) * mui[i - 1];
        }
        Mi[i] = (int) (ceil(mathUtils::randomRelVar(mui[i]) / eps / eps));

        // Gaussian Kernel
        ti[i] = sqrt(log(1 / mui[i]));
        ki[i] = (int) (3 * ceil(r * ti[i]));
        wi[i] = ki[i] / ti[i] * SQRT_2PI;
//        std::cout << "Level " << i << ", samples " << Mi[i] <<
//            ", target: "<< mui[i] << std::endl;
    }
}

AdaptiveRS::AdaptiveRS(shared_ptr<MatrixXd> data, shared_ptr<Kernel> k, double tau, double eps) {
    std::random_device rd;  //Will be used to obtain a seed for the random number engine
    rng = std::mt19937_64(rd());

    X = data;
    kernel = k;
    numPoints = data->rows();
    if (kernel->getName() == EXP_STR) {
        double diam = dataUtils::estimateDiameter(data, tau);
        exp_k = dataUtils::getPower(diam, 0.5);
        exp_w = dataUtils::getWidth(exp_k, 0.5);
    }

    lb = tau;
    buildLevels(tau, eps);
}


AdaptiveRS::AdaptiveRS(shared_ptr<MatrixXd> data, shared_ptr<Kernel> k, int samples,
        double tau, double eps) {
    std::random_device rd;  //Will be used to obtain a seed for the random number engine
    rng = std::mt19937_64(rd());

    int n = data->rows();
    if (samples >= n) {
        X = data;
        kernel = k;
        numPoints = data->rows();
        buildLevels(tau, eps);
        return;
    }
    // Sample input matrix
    std::mt19937_64 gen(rd());
    auto indices = mathUtils::pickSet(n, samples, gen);
    X = make_shared<MatrixXd>(samples, data->cols());
    int i = 0;
    for (auto idx: indices) {
        X->row(i) = data->row(idx);
        i ++;
    }
    kernel = k;
    numPoints = samples;

    if (kernel->getName() == EXP_STR) {
        double diam = dataUtils::estimateDiameter(data, tau);
        exp_k = dataUtils::getPower(diam, 0.5);
        exp_w = dataUtils::getWidth(exp_k, 0.5);
    }
    lb = tau;
    buildLevels(tau, eps);
}


std::vector<double> AdaptiveRS::evaluateQuery(VectorXd q, int level) {
    contrib.clear();
    samples.clear();

    std::uniform_int_distribution<int> distribution(0, numPoints - 1);

    std::vector<double> results = std::vector<double>(2, 0);
    results[1] =  Mi[level];

    std::vector<double> Z = std::vector<double>(L, 0);
    for (int i = 0; i < L; i ++) {
        std::vector<int> indices(results[1]);
        for (int j = 0; j < results[1]; j ++) {
            indices[j] = distribution(rng);
        }
        std::sort(indices.begin(), indices.end());
        for (int j = 0; j < results[1]; j ++) {
            int idx = indices[j];
            samples.push_back(idx);
            double d = kernel->density(q, X->row(idx));
            contrib.push_back(d);
            Z[i] += d;
        }
    }

    results[0] = mathUtils::median(Z) / results[1];
    results[1] *= L;
    return results;
}


int AdaptiveRS::findActualLevel(VectorXd &q, double truth, double eps) {
    int i = 0;
    while (i < I) {
        std::vector<double> results = evaluateQuery(q, i);
        if (fabs(results[0] - truth) / truth < eps) {
            return i;
        }
        i ++;
    }
    return I - 1;
}
