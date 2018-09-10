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
#include <tsl/robin_map.h>

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
            size_t key = getkey(project.col(i));
            auto it = table.find(key);
            if (it == table.end()) {
                table[key] = HashBucket(x);
            } else {
                //table[key].update(x);
                it->second.update(x);
            }
        }
    }

    size_t hashfunction(VectorXd x) {
        VectorXd v = G * x + b;
        return getkey(v);
    }

    size_t getkey(VectorXd v) {
        size_t key = 0;
        for (int i = 0; i < numHash; i ++) {
            boost::hash_combine(key, (int)ceil(v(i)));
        }
        return key;
    }

    HashBucket sample(VectorXd query) {
        size_t key = hashfunction(query);
        auto it = table.find(key);
        if (it == table.end()) {
            return HashBucket();
        } else {
            return it->second;
        }
    }
};


#endif //HBE_HASHTABLE_H
