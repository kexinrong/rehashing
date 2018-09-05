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

using Eigen::MatrixXd;
using Eigen::VectorXd;

using namespace std;

class HashTable {
public:
    unordered_map<string, HashBucket> table;
    MatrixXd G;
    VectorXd b;
    double binWidth;
    int numHash;
    int batchSize;

    HashTable(MatrixXd X, double w, int k, int batchSize) {
        binWidth = w;
        numHash = k;
        batchSize = batchSize;

        int n = X.rows();
        int d = X.cols();
        G = mathUtils::randNormal(batchSize * k, d);
        b = mathUtils::randNormal(batchSize * k);

        MatrixXd shift(batchSize * k, n);
        for (int i = 0; i < n; i ++) { shift.col(i) = b; }

        MatrixXd project = G * X.transpose() / binWidth + shift;
        for (int i = 0; i < n; i ++) {
            VectorXd x = X.row(i);
            vector<string> keys = getkey(project.col(i));
            for (int j = 0; j < batchSize; j ++) {
                try {
                    table.at(keys.at(j)).update(x);
                } catch (exception& e) {
                    table[keys.at(j)] = HashBucket(x);
                }
            }
        }
    }

    vector<string> hashfunction(VectorXd x) {
        VectorXd v = G * x / binWidth + b;
        return getkey(v);
    }

    vector<string> getkey(VectorXd v) {
        assert (v.rows() == batchSize * numHash && v.cols() == 1);
        vector<string> keys(batchSize);
       for (int s = 0; s < batchSize; s ++) {
            ostringstream os;
            for (int i = numHash * s; i < numHash * (s + 1); i ++) {
                os << (int)ceil(v(i));
                os << ',';
            }
            os << s;
            keys.push_back(os.str());
        }
        return keys;
    }

    vector<HashBucket> sample(VectorXd query) {
        vector<string> keys = hashfunction(query);
        vector<HashBucket> buckets(batchSize);
        for (int s = 0; s < batchSize; s ++) {
            try {
                buckets.push_back(table.at(keys.at(s)));
            } catch (exception& e) {
                buckets.push_back(HashBucket());
            }
        }
        return buckets;
    }

};


#endif //HBE_HASHTABLE_H
