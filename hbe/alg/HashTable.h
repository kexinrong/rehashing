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

///
/// Hash table for LSH
///
class HashTable {
public:
    ///
    /// Table: map from hash keys to HashBucket objects
    ///
    unordered_map<size_t, HashBucket> table;

    ///
    /// Number of buckets in the hash table
    ///
    int bucket_count = 0;

    HashTable() {}

    /// Construct an LSH table
    /// \param X dataset
    /// \param w bin wdith
    /// \param k number of hash functions
    /// \param rng
    HashTable(shared_ptr<MatrixXd> X, double w, int k, std::mt19937_64 &rng) {
        binWidth = w;
        numHash = k;

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

    /// Construct an LSH table on from the specified set of weighted samples.
    /// \param X dataset
    /// \param w bin width
    /// \param k number of hash functions
    /// \param samples a collection of weighted samples; each pair contains the weight and index of the sample
    /// \param rng random number generator
    HashTable(shared_ptr<MatrixXd> X, double w, int k,
            vector<pair<int, double>>& samples, std::mt19937_64 &rng) {
        binWidth = w;
        numHash = k;

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

    /// Construct an LSH table on from the specified set of weighted samples.
    /// \param X dataset
    /// \param w bin width
    /// \param k number of hash functions
    /// \param samples a collection of weighted samples; each pair contains the weight and index of the sample
    /// \param rng random number generator
    /// \param scales number of weight scales for each bucket
    HashTable(shared_ptr<MatrixXd> X, double w, int k, vector<pair<int, double>>& samples,
            std::mt19937_64 &rng, int scales) {
        binWidth = w;
        numHash = k;
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


    /// Find hash buckets that the query falls in.
    /// \param query query point
    /// \return
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

private:
    ///
    /// Minimum weight in the bucket. Used to determine the weight scale.
    ///
    double min_weight;
    ///
    /// Maximum weight in the bucket. Used to determine the weight scale.
    ///
    double max_weight;
    ///
    /// Used to determine the weight scale.
    ///
    double weight_step;

    ///
    /// Hash function parameter: h(x) = (Gx + b) / binWidth
    ///
    MatrixXd G;

    ///
    /// Hash function parameter: h(x) = (Gx + b) / binWidth
    ///
    VectorXd b;

    ///
    /// Hash function parameter: h(x) = (Gx + b) / binWidth
    ///
    double binWidth;

    ///
    /// Concatenate numHash hash functions to make one hash key
    ///
    int numHash;

    ///
    /// Constructor and store batchSize number of hash tables together.
    /// One call of sample() on the combined hash table returns batchSize * W_SCALE samples.
    ///
    int batchSize = 1;

    ///
    /// Number of weight scales for hash buckets.
    ///
    int W_SCALE = 3;


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

    int getWeightBucket(double weight) {
        return int(floor(log(weight/min_weight) / log(weight_step)));
    }

};


#endif //HBE_HASHTABLE_H
