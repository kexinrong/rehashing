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
    unordered_set<size_t> used_keys;
    MatrixXd G;
    VectorXd b;
    double binWidth;
    int numHash;
    int batchSize;
    int bucket_count = 0;

    HashTable() {}

    HashTable(shared_ptr<MatrixXd> X, double w, int k, int batch, std::mt19937_64 &rng) {
        binWidth = w;
        numHash = k;
        batchSize = batch;

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

    HashTable(shared_ptr<MatrixXd> X, double w, int k, int batch,
            vector<pair<int, double>>& samples, std::mt19937_64 &rng) {
        binWidth = w;
        numHash = k;
        batchSize = batch;

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

    HashTable(const HashTable &other, size_t nbuckets) {
        binWidth = other.binWidth;
        numHash = other.numHash;
        batchSize = other.batchSize;

        G = other.G;
        b = other.b;

        for (auto it=other.table.begin(); it!=other.table.end(); it++) {
            table[it->first] = HashBucket(it->second);
        }

        double wSum = 0;
        vector<double> weights;
        for (auto it=table.begin(); it!=table.end(); it ++) {
            weights.push_back(it->second.wSum);
            wSum += it->second.wSum;
        }
        if (weights.size() <= nbuckets) { return; }

        // Keep top N buckets
        std::sort(weights.begin(), weights.end(), std::greater<double>());
        double thresh = weights[nbuckets];
        double s = 0.0;
        for (auto it=table.begin(); it!=table.end();) {
            double w = it->second.wSum;
            if (w > thresh) {
                s += w;
                ++it;
            } else {
                table.erase(it++);
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
                used_keys.insert(key);
            }
        }
        return buckets;
    }

};


#endif //HBE_HASHTABLE_H
