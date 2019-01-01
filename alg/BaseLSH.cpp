//
// Created by Kexin Rong on 9/5/18.
//

#include "BaseLSH.h"

using Eigen::PermutationMatrix;
using Eigen::Dynamic;

BaseLSH::BaseLSH() {}

BaseLSH::BaseLSH(shared_ptr<MatrixXd> X, int M, double w, int k,
                 shared_ptr<Kernel> ker, int subsample) {
    batchSize = 1;
    numTables = M;
    binWidth = w;
    numHash = k;
    numPoints = X->rows();
    kernel = ker;
    int dim = X->cols();

    std::random_device rd;  //Will be used to obtain a seed for the random number engine
    rng = std::mt19937_64(rd());

    if (subsample > 1 && subsample < numPoints) {
        // Shuffle Matrix
        PermutationMatrix<Dynamic,Dynamic> perm(numPoints);
        perm.setIdentity();
        std::random_shuffle(perm.indices().data(), perm.indices().data()+perm.indices().size());
        MatrixXd X_perm = perm * (*X); // permute rows
        srand (time(NULL));
        for (int i = 0; i < numTables; i ++) {
            int idx = rand() % (numPoints - subsample);
            shared_ptr<MatrixXd> X_sample = make_shared<MatrixXd>(X_perm.block(idx, 0, subsample, dim));
            tables.push_back(HashTable(X_sample, w, k, rng));
        }
        numPoints = subsample;
    } else {
        for (int i = 0; i < numTables; i++) {
            tables.push_back(HashTable(X, w, k, rng));
        }
    }

}



vector<double> BaseLSH::MoM(VectorXd query, int L, int m) {
    std::vector<double> Z = std::vector<double>(L, 0);
    for (int i = 0; i < L; i ++) {
        for (int j = 0; j < m; j ++){
            Z[i] += evaluateQuery(query);
        }
    }
    return Z;
}

double BaseLSH::evaluateQuery(VectorXd query) {
    idx = (idx + 1) % numTables;
    auto buckets = tables[idx].sample(query);
    double results = 0;
    auto& bucket = buckets[0];
    for (int j = 0; j < bucket.SCALES; j ++) {
        int cnt = bucket.count[j];
        if (cnt > 0) {
            VectorXd delta = bucket.sample[j] - query;
            double c = delta.norm() / binWidth;
            double p = mathUtils::collisionProb(c, numHash);
            results += kernel->density(delta) / p * cnt;
        }
    }
    results /= numPoints;
    return results;
}
