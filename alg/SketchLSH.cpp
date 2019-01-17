//
// Created by Kexin Rong on 2018-10-26.
//

#include "SketchLSH.h"

SketchLSH::SketchLSH(shared_ptr<MatrixXd> X, int M, double w, int k, shared_ptr<Kernel> ker) {
    numTables = M;
    binWidth = w;
    numHash = k;
    kernel = ker;
    numTables = numTables / N_SKETCHES * N_SKETCHES;

    //Will be used to obtain a seed for the random number engine
    std::random_device rd;
    rng = std::mt19937_64(rd());

    for (int i = 0; i < N_SKETCHES; i++) {
        SketchTable t = SketchTable(X, w, k, rng);
        for (int j = 0; j < numTables / N_SKETCHES; j ++) {
            numPoints.push_back(t.samples);
            vector<pair<int, double>> samples = t.sample(t.samples, rng);
            tables.push_back(HashTable(X, w, k, samples, rng));
        }
    }
    // Shuffle to mix data sampled from separate base tables
    std::random_shuffle ( tables.begin(), tables.end() );
}

SketchLSH::SketchLSH(shared_ptr<MatrixXd> X, int M, double w, int k, int scales, shared_ptr<Kernel> ker) {
    numTables = M;
    binWidth = w;
    numHash = k;
    kernel = ker;
    numTables = numTables / N_SKETCHES * N_SKETCHES;

    //Will be used to obtain a seed for the random number engine
    std::random_device rd;
    rng = std::mt19937_64(rd());

    for (int i = 0; i < N_SKETCHES; i++) {
        SketchTable t = SketchTable(X, w, k, rng);
        for (int j = 0; j < numTables / N_SKETCHES; j ++) {
            numPoints.push_back(t.samples);
            vector<pair<int, double>> samples = t.sample(t.samples, rng);
            tables.push_back(HashTable(X, w, k, samples, rng, scales));
        }
    }
    // Shuffle to mix data sampled from separate base tables
    std::random_shuffle ( tables.begin(), tables.end() );
}

SketchLSH::SketchLSH(shared_ptr<MatrixXd> X, vector<SketchTable> &sketches, vector<vector<int>> &indices,
        int M, double w, int k, shared_ptr<Kernel> ker, std::mt19937_64& rng) {
    binWidth = w;
    numHash = k;
    kernel = ker;
    int N_SKETCHES = sketches.size();
    numTables = M  / N_SKETCHES * N_SKETCHES;

    for (int i = 0; i < N_SKETCHES; i++) {
        auto& t = sketches[i];
        for (int j = 0; j < numTables / N_SKETCHES; j ++) {
            numPoints.push_back(t.samples);
            vector<pair<int, double>> samples = t.sample(t.samples, rng);
            for (size_t s = 0; s < samples.size(); s ++) {
                samples[s].first = indices[i][samples[s].first];
            }
            tables.push_back(HashTable(X, w, k, samples, rng));
        }
    }
    // Shuffle to mix data sampled from separate base tables
    if (N_SKETCHES > 1) {
        std::random_shuffle ( tables.begin(), tables.end() );
    }
}


vector<double> SketchLSH::MoM(VectorXd query, int L, int m) {
    std::vector<double> Z = std::vector<double>(L, 0);
    for (int i = 0; i < L; i ++) {
        for (int j = 0; j < m; j ++){
            Z[i] += evaluateQuery(query);
        }
    }
    return Z;
}

double SketchLSH::evaluateQuery(VectorXd query) {
    idx = (idx + 1) % numTables;
    auto buckets = tables[idx].sample(query);
    double results = 0;
    auto& bucket = buckets[0];
    for (int j = 0; j < bucket.SCALES; j ++) {
        if (bucket.count[j] > 0) {
            VectorXd delta = bucket.sample[j] - query;
            double c = delta.norm() / binWidth;
            double p = mathUtils::collisionProb(c, numHash);
            results += kernel->density(delta) / p * bucket.wSum[j];
        }
    }
    results /= numPoints[idx];
    return results;
}