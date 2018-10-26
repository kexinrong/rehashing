//
// Created by Kexin Rong on 2018-10-26.
//

#ifndef HBE_SKETCHTABLE_H
#define HBE_SKETCHTABLE_H

#include "mathUtils.h"
#include <unordered_map>
#include <vector>
#include <exception>
#include <iostream>
#include <sstream>
#include <chrono>
#include <random>
#include <boost/functional/hash.hpp>

using namespace std;

class SketchTable {
public:
    unordered_map<size_t, vector<int>> table;
    vector<size_t> bucket_keys;
    vector<size_t> bucket_size;
    vector<double> weights;
    discrete_distribution<> sampling_distribution;
    MatrixXd G;
    VectorXd b;
    double binWidth;
    int numHash;

    SketchTable() {}

    SketchTable(shared_ptr<MatrixXd> X, double w, int k, int batch, std::mt19937_64 &rng) {
        binWidth = w;
        numHash = k;

        int d = X->cols();
        G = mathUtils::randNormal(k, d, rng) / binWidth;
        b = mathUtils::randUniform(k, rng);

        int n = X->rows();
        MatrixXd project(numHash, n);
        for (int i = 0; i < n; i ++) { project.col(i) = b; }

        project += G * X->transpose();
        int t = 0;
        for (int i = 0; i < n; i ++) {
            VectorXd x = X->row(i);
            size_t key = getkey(project.col(i));
            auto it = table.find(key);
            if (it == table.end()) {
                vector<int> vec = {i};
                table[key] = vec;
                t += 1;
            } else {
                it->second.push_back(i);
            }
        }

        vector<double> sampling_probabilities;
        double gamma = log(t) / log(n);
        double sum = 0;
        for (unordered_map<size_t, vector<int>>::iterator it=table.begin(); it!=table.end(); ++it) {
            bucket_keys.push_back(it->first);
            bucket_size.push_back(it->second.size());
            double pi = pow(it->second.size(), gamma);
            sum += pi;
            sampling_probabilities.push_back(pi);
        }
        sum /= n;
        for (int i = 0; i < t; i ++) {
            weights.push_back(pow(bucket_size[i], 1-gamma) * sum);
        }
        sampling_distribution = discrete_distribution<>(sampling_probabilities.begin(),
                sampling_probabilities.end());
    }

    size_t getkey(VectorXd v) {
        size_t key = 0;
        for (int i = 0; i < numHash; i ++) {
            boost::hash_combine(key, (int)ceil(v(i)));
        }
        return key;
    }

    vector<pair<int, double>> sample(int n, std::mt19937_64 &rng) {
        vector<pair<int, double>> samples;
        for (int i = 0; i < n; i ++) {
            // Sampling bucket
            int bucket_idx = sampling_distribution(rng);
            std::uniform_int_distribution<int> uni(0, bucket_size[bucket_idx] - 1);
            // Sampling point from bucket
            int sample_idx = uni(rng);
            samples.push_back(make_pair<>(table[bucket_keys[bucket_idx]][sample_idx],
                    weights[bucket_idx]));
        }
        return samples;
    }
};


#endif //HBE_SKETCHTABLE_H
