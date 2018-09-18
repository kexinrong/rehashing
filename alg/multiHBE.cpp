//
// Created by Kexin Rong on 9/17/18.
//

#include "multiHBE.h"
#include "dataUtils.h"

multiHBE::multiHBE(shared_ptr<MatrixXd> X, shared_ptr<Kernel> ker, int batch,
        double tau, double eps, double del, int threads) {
    numPoints = X->rows();
    kernel = ker;
    batchSize = batch;
    delta = del;
    maxLevel = ceil(log(1/tau) / log(4));
    numTables = round(1 * kernel->RelVar(tau, delta) / eps / eps / batch);

    double density = 1;
    double beta = 0.5;
    for (int t = 0; t < maxLevel; t ++) {
        density /= 4;
        targets.push_back(density);
        int M = 1 * ceil(kernel->RelVar(density, delta) / eps / eps);
        numSamples.push_back(M);
        int k = dataUtils::getPowerMu(density, beta);
        double w = dataUtils::getWidth(k, beta);
        // Initialize tables for the current level
        vector<HashTable> level;
        for (int i = 0; i < numTables; i++) {
            level.push_back(HashTable(X, w, k, batch));
        }
        tables.push_back(level);
        numHash.push_back(k);
        binWidth.push_back(w);
        cout << "Level: " << targets.size()-1 << ", M=" << M
            << ", w=" << w << ", k=" << k << endl;
    }
}

double multiHBE::normalizeConst(double dist, double mu, double delta) {
    double z = 0;
    int t = kernel->findLevel(mu, maxLevel);
    double d = kernel->density(dist);
    double invD = kernel->invDensity(mu);
    for (int i = 0; i < t; i ++) {
        double w = binWidth[i];
        int k = numHash[i];
        if (!kernel->shouldReject(d, mathUtils::collisionProb(dist, w, k),
                mathUtils::collisionProb(invD, w, k), mu, delta)) {
            double p = mathUtils::collisionProb(dist, w, k);
            z += p * p;
        }
    }
    return z;
}

vector<double> multiHBE::query(VectorXd query) {
    //vector<VectorXd> samples;
    vector<double> est(2, 0);
    for (int t = 0; t < maxLevel; t ++) {
        double mu = targets[t];
        double Z = 0;
        int j = 0;
        int m = numSamples[t];
        while (j < m) {
            vector<double> results = evaluateQuery(query, mu, m - j);
            Z += results[0];
            j += (int)results[1];
        }
        est[1] += m;
        //cout << "Level: " << t << ", target: " << targets[t] << ", estimate: " << Z/m << endl;
        if (Z / m >= mu) {
            est[0] = Z / m;
            return est;
        }
    }
    est[0] = targets.back();
    return est;
};


vector<double> multiHBE::evaluateQuery(VectorXd query, double mu, int maxSamples) {
    int t = kernel->findLevel(mu, maxLevel);
    double invD = kernel->invDensity(mu);
    vector<double> results(2, 0);
    idx = (idx + 1) % numTables;
    size_t n = min(maxSamples, batchSize);
    results[1] = n;
    for (int i = 0; i < t; i ++) {
        double w = binWidth[i];
        int k = numHash[i];

        vector<HashBucket> buckets = tables[i][idx].sample(query);
        for (size_t j = 0; j < n; j ++) {
            HashBucket bucket = buckets[j];
            int cnt = bucket.count;
            if (cnt > 0) {
                VectorXd diff = bucket.sample - query;
                double dist = diff.norm();
                double weight = kernel->density(dist);
                double p = mathUtils::collisionProb(dist, w, k);
                double p_mu = mathUtils::collisionProb(invD, w, k);
                if (!kernel->shouldReject(weight, p, p_mu, mu, delta)) {
                    results[0] += weight * p / normalizeConst(dist, mu, delta) * cnt / numPoints;
                }
            }
        }
    }
    return results;
}


//double multiHBE::reweightSample(VectorXd query, double mu, int maxSamples, vector<VectorXd>& samples) {
//    int t = kernel->findLevel(mu, maxLevels);
//    double invD = kernel->invDensity(mu);
//    for (int i = 0; i < t; i ++) {
//        idx[i] = (idx[i] + 1) % numTables[i];
//        double w = binWidth[i];
//        int k = numHash[i];
//        for (auto s : samples) {
//            vector<HashBucket> buckets = tables[i][idx[i]].sample(s);
//        }
//
//        vector<HashBucket> buckets = tables[i][idx[i]].sample(query);
//        size_t n = min(maxSamples, batchSize);
//        results[1] = n;
//        for (size_t j = 0; j < n; j ++) {
//            HashBucket bucket = buckets[j];
//            int cnt = bucket.count;
//            if (cnt > 0) {
//                VectorXd delta = bucket.sample - query;
//                double dist = delta.norm();
//                double weight = kernel->density(dist);
//                double p = mathUtils::collisionProb(dist, w, k);
//                double p_mu = mathUtils::collisionProb(invD, w, k);
//                if (!kernel->shouldReject(weight, p, p_mu, mu, delta)) {
//                    results[0] += weight * p / normalizeConst(dist, mu, delta) * cnt / numPoints;
//                }
//            }
//        }
//    }
//}