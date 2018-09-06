//
// Created by Kexin Rong on 9/5/18.
//

#include "E2LSH.h"

E2LSH::E2LSH(MatrixXd X, int M, double w, int k, int batchSize,
             shared_ptr<Kernel> kernel, int threads) : BaseLSH(X, M, w, k, batchSize, kernel) {
    for (int i = 0; i < numTables; i ++ ) {
        tables.push_back(HashTable(X, w, k, batchSize));
    }
}

vector<double> E2LSH::evaluateQuery(VectorXd query, int maxSamples) {
    idx = (idx + 1) % numTables;
    HashTable t = tables.at(idx);
    std::vector<HashBucket> buckets = t.sample(query);
    vector<double> results = evaluate(buckets, query,maxSamples);
    return results;
}