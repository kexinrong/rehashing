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

template <typename T>
vector<size_t> sort_indexes(const vector<T> &v) {

    // initialize original index locations
    vector<size_t> idx(v.size());
    iota(idx.begin(), idx.end(), 0);

    // sort indexes based on comparing values in v
    sort(idx.begin(), idx.end(),
         [&v](size_t i1, size_t i2) {return v[i1] < v[i2];});

    return idx;
}

void AdaptiveRSDiag::getConstants() {
    // Sort samples by contribution
    vector<int> tmp_samples;
    vector<double> tmp_weights;
    for (auto i: sort_indexes(contrib)) {
        tmp_samples.push_back(samples[i]);
        tmp_weights.push_back(contrib[i]);
    }
    samples = tmp_samples;
    contrib = tmp_weights;

    thresh = 1e-10;
    u = vector<double>(4, 0);
    s4 = 0;
    set_mins = vector<double>(4, 1);
    set_maxs = vector<double>(4, 0);
    set_start.clear();
    u_global = 0;
    sample_count = 0;
    size_t idx = 0;
    while (contrib[idx] < thresh) {idx ++; }
    // Start of S4
    set_start.push_back(idx);

    for (size_t i = idx; i < contrib.size(); i ++) {
        sample_count += 1;
        u_global += contrib[i];
    }
    u_global /= sample_count;

    // Calcuate S4 stats
    size_t i = idx;
    while (set_start.size() == 1) {
        if (contrib[i] < u_global) {
            u[3] += contrib[i];
            s4 += 1;
            set_maxs[3] = max(contrib[i], set_maxs[3]);
        } else if (contrib[i - 1] < u_global) {
            // End of S4
            set_start.push_back(i);
        }
        i ++;
    }
    u[3] /= sample_count;
}


void AdaptiveRSDiag::findRings(int strategy, double eps) {
    if (strategy == 0 || sample_count < 3) { // Trivial
        lambda = u_global;
        l = u_global;
        set_start.push_back(set_start[1]);
        set_start.push_back(set_start[1]);
    } else if (strategy == 1)  { // Direct
        //double min_u = (1 - eps / 2) * u_global;
        double min_u = (eps * u_global - u[3]) / 2;

        // Find lambda (S3)
        double s = 0;
        size_t i = set_start[1];
        while (s < min_u && i < contrib.size()) {
            s += contrib[i] / sample_count;
            i ++;
        }
        lambda = contrib[i-1];
        set_start.push_back(i);

        // Find L (S1)
        s = 0;
        i = contrib.size() - 1;
        while (s < min_u && i >= set_start[2]) {
            s += contrib[i] / sample_count;
            i --;
        }
        i = min(contrib.size()-2, i);
        l = contrib[i+1];
        set_start.push_back(i+1);
    }
    set_start.push_back(contrib.size());

    std::cout << lambda << ", " << l << std::endl;

    // Calculate set stats
    for (size_t i = 0; i < 3; i ++) {
        for (int j = set_start[3-i]; j < set_start[4-i]; j ++) {
            u[i] += contrib[j];
            set_maxs[i] = max(contrib[j], set_maxs[i]);
            set_mins[i] = min(contrib[j], set_mins[i]);

        }
        u[i] /= sample_count;
    }
}

double AdaptiveRSDiag::RSTriv() {
    return (set_maxs[0] / set_mins[0]) * u[0] * u[0] + s4 * set_maxs[0] * u[0] / sample_count + set_maxs[3] * u[3];
}

double AdaptiveRSDiag::HBETriv(VectorXd &q, int level) {
    // Gaussian Kernel
    if (kernel->getName() != EXP_STR) {
        exp_w = wi[level];
        exp_k = ki[level];
    }

    double p4max = 0;
    double p1min = 1;
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
        if (contrib[i] < u_global) { // S4
            p4max = max(p4max, p);
        } else { // S1
            p1min = min(p1min, p);
        }
        pmin = min(pmin, p);
    }

    double v4_wmax = 0;
    int pi = -1;
    int pj = -1;
    double kjmin = std::numeric_limits<double>::max();
    double v1_wmax = 0;
    for (size_t i = 0; i < samples.size(); i ++) {
        if (contrib[i] < thresh) {
            continue;
        }

        double k_i = contrib[i] / prob[i] / prob[i];
        double k_j = contrib[i] / prob[i];

        if (contrib[i] > u_global) { // S1
            if (k_i > v1_wmax) {pi = i;}
            if (k_j < kjmin) {pj = i;}
            v1_wmax = max(v1_wmax, k_i);
            kjmin = min(kjmin, k_j);
        } else { // S4
            v4_wmax = max(v4_wmax, k_i);
        }
    }
    double sup3 = v4_wmax * p4max;
    double sup2 = v1_wmax * p4max;

    double sup1 = 1 / p1min;
    if (prob[pi] > prob[pj]) {
//        std::cout << ">: pi: " << prob[pi] << ",  wi: " << contrib[pi] << ", pj:" << prob[pj] << ", wj: " << contrib[pj]
//                  << std::endl;
        sup1 = max(sup1, v1_wmax / kjmin);
    }

//    std::cout << sup1 << "," << sup2 << "," << sup3 << std::endl;
    return sup1 * u[0] * u[0] + s4 * sup2 * u[0] / sample_count + sup3 * u[3];
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
