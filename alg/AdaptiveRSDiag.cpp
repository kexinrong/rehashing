//
// Created by Kexin Rong on 1/2/19.
//

#include "AdaptiveRSDiag.h"

//
// Created by Kexin Rong on 2018-11-09.
//

#include "AdaptiveRS.h"
#include "dataUtils.h"
#include "mathUtils.h"
#include <math.h>

void AdaptiveRSDiag::buildLevels(double tau, double eps) {
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

        // Gaussian Kernel
        ti[i] = sqrt(log(1 / mui[i]));
        ki[i] = (int) (3 * ceil(r * ti[i]));
        wi[i] = ki[i] / ti[i] * SQRT_2PI;
//        std::cout << "Level " << i << ", samples " << Mi[i] <<
//            ", target: "<< mui[i] << std::endl;
    }
}

AdaptiveRSDiag::AdaptiveRSDiag(shared_ptr<MatrixXd> data, shared_ptr<Kernel> k, double tau, double eps) {
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


AdaptiveRSDiag::AdaptiveRSDiag(shared_ptr<MatrixXd> data, shared_ptr<Kernel> k, int samples,
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
    lb = tau;
    buildLevels(tau, eps);
}


std::vector<double> AdaptiveRSDiag::evaluateQuery(VectorXd q, int level) {
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
            double d = kernel->density(q, X->row(idx));
            samples.push_back(idx);
            contrib.push_back(d);
            Z[i] += d;
        }
    }

    results[0] = mathUtils::median(Z) / results[1];
    return results;
}

void AdaptiveRSDiag::clearSamples() {
    contrib.clear();
    samples.clear();
}

void AdaptiveRSDiag::getConstants(double est, double eps) {
//    thresh = eps * lb / 10;
    thresh = 1e-10;
    u = vector<double>(4, 0);
    s = vector<int>(4, 0);
    u_global = 0;
    sample_count = 0;
    for (size_t i = 0; i < contrib.size(); i ++) {
        if (contrib[i] < thresh) {
            continue;
        }
        sample_count += 1;
        u_global += contrib[i];
    }
    u_global /= sample_count;

    w1Max = 0;
    w1Min = 1;
    for (size_t i = 0; i < contrib.size(); i ++) {
        if (contrib[i] < thresh) {
            continue;
        }
        if (contrib[i] >= u_global) { // S1
            u[0] += contrib[i];
            s[0] += 1;
            w1Min = min(contrib[i], w1Min);
            w1Max = max(contrib[i], w1Max);
        } else {
            u[3] += contrib[i];
            s[3] += 1;
        }
    }
    u[0] /= s[0];
    u[3] /= s[3];
}

//double getAvg(double lower, double upper) {
//    int cnt = 0;
//    int s = 0;
//    for (size_t i = 0; i < contrib.size(); i ++) {
//        if (contrib[i] >= lower && contrib[i] <= upper) {
//            s += contrib[i];
//            cnt += 1;
//        }
//    }
//    return s / cnt;
//}
//
//void findRings(int strategy, double eps) {
//    if (strategy == 0) { // Trivial
//        lambda = u_global;
//        l = u_global;
//    } else if (strategy == 1)  {
//
//    }
//}

double AdaptiveRSDiag::RSTriv() {
//    std::cout << u[0] << "," << u[3] << "," << w1Max << "," << w1Min << "," << u_global << std::endl;
    return (w1Max / w1Min) * u[0] * u[0] + s[3] * w1Max * u[0] / sample_count + u_global * u[3];
}

double AdaptiveRSDiag::HBETriv(VectorXd &q, int level) {
    // Gaussian Kernel
    if (kernel->getName() != EXP_STR) {
        exp_w = wi[level];
        exp_k = ki[level];
    }

    double pmin = 1;
    vector<double> prob;
    for (size_t i = 0; i < samples.size(); i ++) {
        if (contrib[i] < thresh) {
            prob.push_back(0);
            continue;
        }
        int idx = samples[i];
        VectorXd delta = X->row(idx) - q.transpose();
        double c = delta.norm() / exp_w;
        double p = mathUtils::collisionProb(c, exp_k);
        prob.push_back(p);
        pmin = min(pmin, p);
    }

    double vij_wwmax = 0;
    double vij_wmax = 0;
    int pi = -1;
    int pj = -1;
    int pi_ = -1;
    int pj_ = -1;
    double kjmin = std::numeric_limits<double>::max();
    double kimax = 0;
    double kjmax = 0;
    for (size_t i = 0; i < samples.size(); i ++) {
        if (contrib[i] < thresh) {
            continue;
        }

        double k_i = contrib[i] / prob[i] / prob[i];
        double k_j = contrib[i] / prob[i];

        if (contrib[i] > u_global) { // S1
            double tmp = min(pmin, prob[i]) / prob[i] / prob[i] * contrib[i];
            vij_wmax = max(vij_wmax, tmp);
//            if ((w1Max - contrib[i]) < 1e-5) {
//                pj_ = i;
//            }

            if (k_i > kimax) {pi = i;}
            if (k_j < kjmin) {pj = i;}
            if (k_j > kjmax) {pi_ = i;}
            kimax = max(kimax, k_i);
            kjmin = min(kjmin, k_j);
            kjmax = max(kjmax, k_j);
        }
    }
    double vmax = 1 / pmin;

    vij_wwmax = kjmax * w1Max;
    if (prob[pi] > prob[pj]) {
//        std::cout << ">: pi: " << prob[pi] << ",  wi: " << contrib[pi] << ", pj:" << prob[pj] << ", wj: " << contrib[pj]
//                  << std::endl;
        vij_wwmax = max(vij_wwmax, kimax / kjmin);
    }
//    } else {
//        std::cout << "<: pi: " << prob[pi_] << ",  wi: " << contrib[pi_] <<", pj:" << prob[pj_] << ", wj: " << contrib[pj_] << ", w1max:"
//            << w1Max << std::endl;
//        vij_wwmax = kjmax * w1Max;
//    }
//    std::cout << vij_wwmax << "," << vij_wwmax * u[0] * u[0] << "," << s[3] * vij_wmax * u[0] / sample_count << std::endl;

    return vij_wwmax * u[0] * u[0] + s[3] * vij_wmax * u[0] / sample_count + vmax * u_global * u[3];
}


int AdaptiveRSDiag::findTargetLevel(double est) {
    int i = 0;
    while (i < I) {
        if (est > mui[i]) {
            return i;
        }
        i ++;
    }
    return I - 1;
}

int AdaptiveRSDiag::findActualLevel(VectorXd &q, double truth, double eps) {
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
