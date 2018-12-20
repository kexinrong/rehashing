//
// Created by Kexin Rong on 2018-10-26.
//

#include "SketchLSH.h"
#include "SketchTable.h"

SketchLSH::SketchLSH(shared_ptr<MatrixXd> X, int M, double w, int k, shared_ptr<Kernel> ker) {
    numTables = M;
    binWidth = w;
    numHash = k;
    int N = X->rows();
    numPoints = int(sqrt(N));
    kernel = ker;
    numTables = numTables / N_SKETCHES * N_SKETCHES;

    //Will be used to obtain a seed for the random number engine
    std::random_device rd;
    rng = std::mt19937_64(rd());

    for (int i = 0; i < N_SKETCHES; i++) {
        SketchTable t = SketchTable(X, w, k, rng);
        for (int j = 0; j < numTables / N_SKETCHES; j ++) {
            vector<pair<int, double>> samples = t.sample(numPoints, rng);
            tables.push_back(HashTable(X, w, k, samples, rng, true));
        }
    }
    // Shuffle to mix data sampled from separate base tables
    std::random_shuffle ( tables.begin(), tables.end() );
}

SketchLSH::SketchLSH(shared_ptr<MatrixXd> X, int M, double w, int k,
                     shared_ptr<Kernel> ker, int sketches) {
    binWidth = w;
    numHash = k;
    int N = X->rows();
    numPoints = int(sqrt(N));
    kernel = ker;
    N_SKETCHES = sketches;
    numTables = M / N_SKETCHES * N_SKETCHES;

    //Will be used to obtain a seed for the random number engine
    std::random_device rd;
    rng = std::mt19937_64(rd());

    for (int i = 0; i < N_SKETCHES; i++) {
        SketchTable t = SketchTable(X, w, k, rng);
        for (int j = 0; j < numTables / N_SKETCHES; j ++) {
            vector<pair<int, double>> samples = t.sample(numPoints, rng);
            tables.push_back(HashTable(X, w, k, samples, rng, true));
        }
    }
    // Shuffle to mix data sampled from separate base tables
    if (sketches > 1) {
        std::random_shuffle ( tables.begin(), tables.end() );
    }
}

//
//SketchLSH::SketchLSH(const SketchLSH& other, int nbuckets) {
//    numTables = other.numTables;
//    binWidth = other.binWidth;
//    numHash = other.numHash;
//    numPoints = other.numPoints;
//    kernel = other.kernel;
//
//    for (const auto& t : other.tables) {
//        tables.push_back(HashTable(t, nbuckets));
//    }
//}

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
    size_t n = maxSamples;
    results[1] = n;
    for (size_t i = 0; i < n; i ++) {
        HashBucket bucket = buckets[i];
        for (int j = 0; j < bucket.SCALES; j ++) {
            int cnt = bucket.count[j];
            if (cnt > 0) {
                VectorXd delta = bucket.sample[j] - query;
                double c = delta.norm() / binWidth;
                double p = mathUtils::collisionProb(c, numHash);
                results[0] += kernel->density(delta) / p * bucket.wSum[j] / numPoints / bucket.SCALES;
            }
        }
    }
    return results;
}