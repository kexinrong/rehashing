//
// Created by Kexin Rong on 2018-11-09.
//

#include "AdaptiveRS.h"
#include "dataUtils.h"
#include "mathUtils.h"

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
        } else {
            mui[i] = (1 - gamma) * mui[i - 1];
        }
        Mi[i] = (int) (ceil(mathUtils::randomRelVar(mui[i]) / eps / eps));
//        std::cout << "Level " << i << ", samples " << Mi[i] <<
//            ", target: "<< mui[i] << std::endl;
    }
}

AdaptiveRS::AdaptiveRS(shared_ptr<MatrixXd> data, shared_ptr<Kernel> k, double tau, double eps) {
    X = data;
    kernel = k;
    numPoints = data->rows();
    if (kernel->getName() == EXP_STR) {
        double diam = dataUtils::estimateDiameter(data, tau);
        exp_k = dataUtils::getPower(diam, 0.5);
        exp_w = dataUtils::getWidth(exp_k, 0.5);
    }

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

    if (kernel->getName() == EXP_STR) {
        double diam = dataUtils::estimateDiameter(data, tau);
        exp_k = dataUtils::getPower(diam, 0.5);
        exp_w = dataUtils::getWidth(exp_k, 0.5);
    }
    buildLevels(tau, eps);
}


std::vector<double> AdaptiveRS::evaluateQuery(VectorXd q, int level, int maxSamples) {
   contrib.clear();
   samples.clear();

    std::random_device rd;  //Will be used to obtain a seed for the random number engine
    std::mt19937_64 rng = std::mt19937_64(rd());
    std::uniform_int_distribution<int> distribution(0, numPoints - 1);

    std::vector<double> results = std::vector<double>(2, 0);
    results[1] = min(maxSamples, Mi[level]);

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
    return results;
}

double AdaptiveRS::findRSRatio(double est, double eps) {
    double thresh = min(est * eps / 10, 1.0 /numPoints);
  //  std::cout << "thresh: " << thresh << std::endl;
    double mmin = contrib[0];
    double mmax = contrib[0];
    for (size_t i = 1; i < contrib.size(); i ++) {
        mmin = min(max(contrib[i], thresh), mmin);
        mmax = max(contrib[i], mmax);
    }
    return mmax / mmin;
}

double AdaptiveRS::findHBERatio(VectorXd &q, int level, double est, double eps) {
    double thresh = min(est * eps / 10, 1.0 /numPoints);
    double mmin = 1;
    double mmax = 0;
    for (size_t i = 0; i < samples.size(); i ++) {
        if (contrib[i] < thresh) {
            continue;
        }
        int idx = samples[i];
        VectorXd delta = X->row(idx) - q.transpose();
        double c = delta.norm() / exp_w;
        double p = mathUtils::collisionProb(c, exp_k);
        double k_i = contrib[i] / p / p;
        double k_j = contrib[i] / p;
        if (mmax == 0) {
            mmin = k_j;
            mmax = k_i;
        } else {
            mmax = max(mmax, k_i);
            mmin = min(mmin, k_j);
        }
    }
    return mmax / mmin;
}


int AdaptiveRS::findTargetLevel(double est) {
    int i = 0;
    while (i < I) {
        if (est > mui[i]) {
            return i;
        }
        i ++;
    }
    return I - 1;
}

int AdaptiveRS::findActualLevel(VectorXd &q, double truth, double eps) {
    int i = 0;
    while (i < I) {
        std::vector<double> results = evaluateQuery(q, i, Mi[i]);
        if (fabs(results[0] - truth) / truth < eps) {
            return i;
        }
        i ++;
    }
    return I - 1;
}
