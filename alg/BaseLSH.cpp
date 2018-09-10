//
// Created by Kexin Rong on 9/5/18.
//

#include "BaseLSH.h"

BaseLSH::BaseLSH(MatrixXd X, int M, double w, int k, int batch,
                 shared_ptr<Kernel> ker, int threads) {
    batchSize = batch;
    numTables = M / batch + 1;
    binWidth = w;
    numHash = k;
    numPoints = X.rows();
    kernel = ker;

    for (int i = 0; i < numTables; i ++ ) {
        tables.push_back(HashTable(X, w, k, batchSize));
    }
}


vector<double> BaseLSH::MoM(VectorXd query, int L, int m) {
    std::vector<double> Z = std::vector<double>(L, 0);
    for (int i = 0; i < L; i ++) {
        for (int j = 0; j < m; j ++ ){
            Z[i] += evaluateQuery(query);
        }
    }
    return Z;
}

double BaseLSH::evaluateQuery(VectorXd query) {
    idx = (idx + 1) % numTables;
    HashBucket bucket = tables[idx].sample(query);
    double results = 0;
    int cnt = bucket.count;
    if (cnt > 0) {
        VectorXd delta = bucket.sample - query;
        double c = delta.norm() / binWidth;
        double p = mathUtils::collisionProb(c, numHash);
        results = kernel->density(delta) / p * cnt / numPoints;
    }
    return results;
}