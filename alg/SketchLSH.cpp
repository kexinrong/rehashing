//
// Created by Kexin Rong on 2018-10-26.
//

#include "SketchLSH.h"
#include "SketchTable.h"

SketchLSH::SketchLSH(shared_ptr<MatrixXd> X, int M, double w, int k, int batch,
                 shared_ptr<Kernel> ker) {
    batchSize = batch;
    numTables = round(M / batch);
    binWidth = w;
    numHash = k;
    int N = X->rows();
    numPoints = int(sqrt(N));
    kernel = ker;

    //Will be used to obtain a seed for the random number engine
    std::random_device rd;
    rng = std::mt19937_64(rd());

    for (int i = 0; i < N_SKETCHES; i++) {
        SketchTable t = SketchTable(X, w, k, batch, rng);
        std::cout << t.gamma << std::endl;
        for (int j = 0; j < numTables / N_SKETCHES; j ++) {
            vector<pair<int, double>> samples = t.sample(numPoints, rng);
            tables.push_back(HashTable(X, w, k, batch, samples, rng));
        }
    }
    // Shuffle to mix data sampled from separate base tables
    std::random_shuffle ( tables.begin(), tables.end() );
}

vector<double> SketchLSH::MoM(VectorXd query, int L, int m) {
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

vector<double> SketchLSH::evaluateQuery(VectorXd query, int maxSamples) {
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
            results[0] += bucket.weight * kernel->density(delta) / p * cnt / numPoints;
        }
    }
    return results;
}