//
// Created by Kexin Rong on 9/5/18.
//

#ifndef HBE_HASHTABLE_H
#define HBE_HASHTABLE_H

#include "HashBucket.h"
#include "mathUtils.h"
#include <unordered_map>
#include <vector>
#include <exception>
#include <iostream>
#include <sstream>
#include <chrono>
#include <boost/functional/hash.hpp>

using namespace std;

class HashTable {
public:
    unordered_map<size_t, HashBucket> table;
    MatrixXd G;
    VectorXd b;
    double binWidth;
    int numHash;
    int batchSize;
    int bucket_count = 0;

    int W_SCALE = 3;
    double min_weight;
    double max_weight;
    double weight_step;

    HashTable() {}

    HashTable(shared_ptr<MatrixXd> X, double w, int k, std::mt19937_64 &rng) {
        binWidth = w;
        numHash = k;
        batchSize = 1;

        int d = X->cols();
        G = mathUtils::randNormal(batchSize * k, d, rng) / binWidth;
        b = mathUtils::randUniform(batchSize * k, rng);

        int n = X->rows();
        MatrixXd project(batchSize * numHash, n);
        for (int i = 0; i < n; i ++) { project.col(i) = b; }

        project += G * X->transpose();
        for (int i = 0; i < n; i ++) {
            VectorXd x = X->row(i);
            vector<size_t> keys = getkey(project.col(i));
            for (auto key: keys) {
                auto it = table.find(key);
                if (it == table.end()) {
                    table[key] = HashBucket(x);
                    bucket_count ++;
                } else {
                    it->second.update(x, rng);
                }
            }
        }
    }

    HashTable(shared_ptr<MatrixXd> X, double w, int k,
            vector<pair<int, double>>& samples, std::mt19937_64 &rng) {
        binWidth = w;
        numHash = k;
        batchSize = 1;

        int d = X->cols();
        G = mathUtils::randNormal(batchSize * k, d, rng) / binWidth;
        b = mathUtils::randUniform(batchSize * k, rng);

        int n = samples.size();
        MatrixXd project(batchSize * numHash, n);
        for (int i = 0; i < n; i ++) { project.col(i) = b; }

        // Get samples from matrix
        MatrixXd X_sample(n, d);
        sort(samples.begin(), samples.end());
        for (int i = 0; i < n; i ++) {
            X_sample.row(i) = X->row(samples[i].first);
        }

        project += G * X_sample.transpose();
        for (int i = 0; i < n; i ++) {
            VectorXd x = X_sample.row(i);
            vector<size_t> keys = getkey(project.col(i));
            for (auto key: keys) {
                auto it = table.find(key);
                if (it == table.end()) {
                    table[key] = HashBucket(x, samples[i].second);
                    bucket_count ++;
                } else {
                    it->second.update(x, samples[i].second, rng);
                }
            }
        }
    }

    int getWeightBucket(double weight) {
        return int(floor(log(weight/min_weight) / log(weight_step)));
    }

    HashTable(shared_ptr<MatrixXd> X, double w, int k, vector<pair<int, double>>& samples,
            std::mt19937_64 &rng, int scales) {
        binWidth = w;
        numHash = k;
        batchSize = 1;
        W_SCALE = scales;

        int d = X->cols();
        G = mathUtils::randNormal(batchSize * k, d, rng) / binWidth;
        b = mathUtils::randUniform(batchSize * k, rng);

        int n = samples.size();
        MatrixXd project(batchSize * numHash, n);
        for (int i = 0; i < n; i ++) { project.col(i) = b; }

        // Get samples from matrix
        MatrixXd X_sample(n, d);
        sort(samples.begin(), samples.end());
        for (int i = 0; i < n; i ++) {
            X_sample.row(i) = X->row(samples[i].first);
        }

        min_weight = n;
        max_weight = 0;
        for (int i = 0; i < n; i ++) {
            min_weight = min(min_weight, samples[i].second);
            max_weight = max(max_weight, samples[i].second);
        }
        weight_step = pow(max_weight/min_weight, 1.0/W_SCALE) * 1.1;

        project += G * X_sample.transpose();
        for (int i = 0; i < n; i ++) {
            VectorXd x = X_sample.row(i);
            vector<size_t> keys = getkey(project.col(i));
            for (auto key: keys) {
                auto it = table.find(key);
                double weight = samples[i].second;
                int idx = getWeightBucket(weight);
                if (it == table.end()) {
                    table[key] = HashBucket(x, weight, idx, W_SCALE);
                    bucket_count ++;
                } else {
                    it->second.update(x, weight, idx, rng);
                }
            }
        }
    }



    vector<size_t> hashfunction(VectorXd x) {
        VectorXd v = G * x + b;
        return getkey(v);
    }

    vector<size_t> getkey(VectorXd v) {
        vector<size_t> keys(batchSize);
        for (int b = 0; b < batchSize; b++) {
            size_t key = b;
            for (int i = 0; i < numHash; i ++) {
                boost::hash_combine(key, (int)ceil(v(i)));
            }
            keys[b] = key;
        }
        return keys;
    }

    vector<HashBucket> sample(VectorXd query) {
        vector<size_t> keys = hashfunction(query);
        vector<HashBucket> buckets;
        for (auto key : keys) {
            auto it = table.find(key);
            if (it == table.end()) {
                buckets.push_back(HashBucket());
            } else {
                buckets.push_back(it->second);
            }
        }
        return buckets;
    }

};


#endif //HBE_HASHTABLE_H
