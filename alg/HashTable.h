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

    HashTable(MatrixXd X, double w, int k, int batch) {
        binWidth = w;
        numHash = k;
        batchSize = batch;

        int n = X.rows();
        int d = X.cols();
        G = mathUtils::randNormal(batchSize * k, d) / binWidth;
        b = mathUtils::randUniform(batchSize * k);

        MatrixXd project(batchSize * k, n);
        for (int i = 0; i < n; i ++) { project.col(i) = b; }

        project += G * X.transpose();
        for (int i = 0; i < n; i ++) {
            VectorXd x = X.row(i);
            vector<size_t> keys = getkey(project.col(i));
            for (int j = 0; j < batchSize; j ++) {
                try {
                    table.at(keys.at(j)).update(x);
                } catch (exception& e) {
                    table[keys.at(j)] = HashBucket(x);
                }
            }
        }
    }

    vector<size_t> hashfunction(VectorXd x) {
        VectorXd v = G * x + b;
        return getkey(v);
    }

    vector<size_t> getkey(VectorXd v) {
        assert (v.rows() == batchSize * numHash && v.cols() == 1);
        vector<size_t> keys(batchSize);
        for (int s = 0; s < batchSize; s ++) {
            size_t key = s;
            for (int i = numHash * s; i < numHash * (s + 1); i ++) {
                boost::hash_combine(key, (int)ceil(v(i)));
            }
            keys[s] = key;
        }
        return keys;
    }

    vector<HashBucket> sample(VectorXd query) {
        vector<size_t> keys = hashfunction(query);
        vector<HashBucket> buckets(batchSize);
        for (int s = 0; s < batchSize; s ++) {
            try {
                buckets[s] = table.at(keys.at(s));
            } catch (exception& e) {
                buckets[s] = HashBucket();
            }
        }
        return buckets;
    }

};


#endif //HBE_HASHTABLE_H
