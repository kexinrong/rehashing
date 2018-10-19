//
// Created by Kexin Rong on 9/5/18.
//

#include "BaseLSH.h"


BaseLSH::BaseLSH(shared_ptr<MatrixXd> X, int M, double w, int k, int batch,
                 shared_ptr<Kernel> ker, int threads) {
    batchSize = batch;
    numTables = round(M / batch);
    binWidth = w;
    numHash = k;
    numPoints = X->rows();
    kernel = ker;

    std::random_device rd;  //Will be used to obtain a seed for the random number engine
    rng = std::mt19937_64(rd());
    for (int i = 0; i < numTables; i++) {
        tables.push_back(HashTable(X, w, k, batch, rng));
    }
}



vector<double> BaseLSH::MoM(VectorXd query, int L, int m) {
    std::vector<double> Z = std::vector<double>(L, 0);
    for (int i = 0; i < L; i ++) {
        int j = 0;
        while (j < m) {
            vector<double> results = evaluateQuery(query, m - j);
            Z[i] += results[0];
            j += (int)results[1];
        }
    }
    return Z;
}

vector<double> BaseLSH::evaluateQuery(VectorXd query, int maxSamples) {
    idx = (idx + 1) % numTables;
    vector<HashBucket> buckets = tables[idx].sample(query);
    vector<double> results = vector<double>(2, 0);
    size_t n = min(maxSamples, batchSize);
    results[1] = n;
    for (size_t i = 0; i < n; i ++) {
        HashBucket bucket = buckets[i];
        int cnt = bucket.count;
        if (cnt > 0) {
            VectorXd delta = bucket.sample - query;
            double c = delta.norm() / binWidth;
            double p = mathUtils::collisionProb(c, numHash);
            results[0] += kernel->density(delta) / p * cnt / numPoints;
        }
    }
    return results;
}
