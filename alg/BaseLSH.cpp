//
// Created by Kexin Rong on 9/5/18.
//

#include "BaseLSH.h"

BaseLSH::BaseLSH(MatrixXd X, int M, double w, int k, int batch) {
    batchSize = batch;
    numTables = M / batch + 1;
    binWidth = w;
    numHash = k;
    numPoints = X.rows();
}

double* BaseLSH::MoM(VectorXd query, int L, int m) {
    double* Z = new double[L];
    std::memset(Z, 0, L);
    for (int i = 0; i < L; i ++) {
        int j = 0;
        while (j < m) {
            double* results = evaluateQuery(query, m - j);
            Z[i] += results[0];
            j += (int)results[1];
        }
    }
    return Z;
}

double* BaseLSH::evaluate(std::vector<HashBucket> buckets, VectorXd query, int maxSamples) {
    double *results = new double[2]{0, 0};
    for (HashBucket bucket : buckets) {
        if (results[1] >= maxSamples) { break; }
        int cnt = bucket.count;
        if (cnt == 0) {
            // Reweight 0 points according to #samples per hash bucket
            results[1] += 1;
        } else {
            VectorXd delta = bucket.sample - query;
            double c = delta.norm() / binWidth;
            double p = mathUtils::collisionProb(c, numHash);
            results[0] += kernel->density(delta) / p * cnt / numPoints;
            results[1] += 1;
        }
    }
    return results;
}