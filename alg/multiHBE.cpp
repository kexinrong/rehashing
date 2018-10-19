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
        std::random_device rd;  //Will be used to obtain a seed for the random number engine
        std::mt19937_64 rng(rd());
        for (int i = 0; i < numTables; i++) {
            level.push_back(HashTable(X, w, k, batch, rng));
        }
        tables.push_back(level);
        numHash.push_back(k);
        binWidth.push_back(w);

        cout << "Level: " << targets.size()-1 << ", M=" << M
            << ", w=" << w << ", k=" << k << endl;
    }

    // Precompute p_mu for each level
    for (double mu : targets) {
        double invD = kernel->invDensity(mu);
        vector<double> tmp;
        for (int t = 0; t < maxLevel; t ++) {
            // Precompute
            double w = binWidth[t];
            double k = numHash[t];
            tmp.push_back(mathUtils::collisionProb(invD, w, k));
        }
        p_mus.push_back(tmp);
    }
}


void multiHBE::clear() {
    counts.resize(0);
    tableIdx.resize(0);
    p_mus.resize(0);
    dists.resize(0);
    kernels.resize(0);
    probs.resize(0);
}

vector<double> multiHBE::query(VectorXd query) {
    vector<double> est(2, 0);
    for (int t = 0; t < maxLevel; t ++) {
        double mu = targets[t];
        // Reweight old samples
        double Z = reweightSample(query, mu, t);
        // New samples
        int j = 0;
        int m = numSamples[t] - est[1];
        while (j < m) {
            vector<double> results = getNewSamples(query, mu, t, m - j);
            Z += results[0];
            j += (int)results[1];
        }
        est[1] += m;
        //cout << "Level: " << t << ", target: " << targets[t] << ", estimate: " << Z/est[1] << endl;
        if (Z / est[1] >= mu) {
            est[0] = Z / est[1];
            return est;
        }
    }
    est[0] = targets.back();
    return est;
};

vector<double> multiHBE::normalizeConst(double dist, double weight, double log_mu,
                                        int skipLevel, int targetLevel, double* z) {
    if (targetLevel == 0) { return vector<double>(1); }

    vector<double> ps(targetLevel + 1);
    for (int i = 0; i < targetLevel; i ++) {
        if (i == skipLevel) { continue; }
        double p = mathUtils::collisionProb(dist, binWidth[i], numHash[i]);
        double p_mu = p_mus[targetLevel][i];
        ps[i] = p;
        if (!kernel->shouldReject(weight, p, p_mu, log_mu, delta)) {
            *z += p * p;
        }
    }
    return ps;
}

double multiHBE::sumNormalizeConst(double dist, double weight, double log_mu, int skipLevel,
        int targetLevel, vector<double> &probs) {
    double z = 0;
    for (int i = 0; i <= targetLevel; i ++) {
        if (i == skipLevel) { continue; }
        if (probs[i] < 0) {
            probs[i] = mathUtils::collisionProb(dist, binWidth[i], numHash[i]);
        }
        double p = probs[i];
        if (p < 0) { cout << "i" << i << endl; }
        double p_mu = p_mus[targetLevel][i];
        if (!kernel->shouldReject(weight, p, p_mu, log_mu, delta)) {
            z += p * p;
        }
    }
    return z;
}


vector<double> multiHBE::getNewSamples(VectorXd query, double mu, int t, int maxSamples) {
    vector<double> results(2, 0);
    idx = (idx + 1) % numTables;
    size_t n = min(maxSamples, batchSize);
    results[1] = n;
    double log_mu = log(mu);
    tableIdx.push_back(idx);
    vector<int> bucket_count(t + 1, 0);
    vector<double> curr_dist(t + 1, 0);
    vector<double> curr_kernel(t + 1, 0);
    vector<vector<double>> curr_probs(t + 1);
    for (int i = 0; i <= t; i ++) {
        double w = binWidth[i];
        int k = numHash[i];
        double p_mu = p_mus[t][i];

        HashBucket bucket = tables[i][idx].sample(query)[0];
        int cnt = bucket.count;
        bucket_count[i] = cnt;
        if (cnt > 0) {
            double dist = (bucket.sample - query).norm();
            double weight = kernel->density(dist);
            curr_dist[i] = dist;
            curr_kernel[i] = weight;
            double p = mathUtils::collisionProb(dist, w, k);
            if (!kernel->shouldReject(weight, p, p_mu, log_mu, delta)) {
                double z = p * p;
                curr_probs[i] = normalizeConst(dist, weight, log_mu, i, t, &z);
                results[0] += weight * p / z * cnt / numPoints;
            } else {
                curr_probs[i] = vector<double>(t+1, -1);
            }
            curr_probs[i][i] = p;
        }
    }
    counts.push_back(bucket_count);
    probs.push_back(curr_probs);
    dists.push_back(curr_dist);
    kernels.push_back(curr_kernel);
    return results;
}

double multiHBE::reweightSample(VectorXd query, double mu, int t) {
    if (t == 0) { return 0; }
    double log_mu = log(mu);

    double Z = 0;
    for (size_t i = 0; i < tableIdx.size(); i++) {
        for (int l = 0; l <= t; l ++) {
            double p_mu = p_mus[t][l];
            double w = binWidth[l];
            int k = numHash[l];

            if (l == t) {
                HashBucket bucket = tables[l][tableIdx[i]].sample(query)[0];
                int cnt = bucket.count;
                counts[i].push_back(cnt);
                if (cnt > 0) {
                    double dist = (bucket.sample - query).norm();
                    double weight = kernel->density(dist);
                    dists[i].push_back(dist);
                    kernels[i].push_back(weight);
                    double p = mathUtils::collisionProb(dist, w, k);
                    if (!kernel->shouldReject(weight, p, p_mu, log_mu, delta)) {
                        double z = p * p;
                        probs[i].push_back(normalizeConst(dist, weight, log_mu, l, t, &z));
                        Z += weight * p / z * cnt / numPoints;
                    } else {
                        probs[i].push_back(vector<double>(t+1, -1));
                    }
                    probs[i][l][l] = p;
                }
            } else {
                int cnt = counts[i][l];
                if (cnt > 0) {
                    double dist = dists[i][l];
                    double weight = kernels[i][l];
                    double p = probs[i][l][l];
                    // New: collision probability at the new level
                    double new_p = mathUtils::collisionProb(dist, binWidth[t], numHash[t]);
                    probs[i][l].push_back(new_p);
                    if (!kernel->shouldReject(weight, p, p_mu, log_mu, delta)) {
                        double z = p * p;
                        z += sumNormalizeConst(dist, weight, log_mu, l, t, probs[i][l]);
                        Z += weight * p / z * cnt / numPoints;
                    }
                }
            }
        }
    }
    return Z;
}