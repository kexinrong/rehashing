//
// Created by Kexin Rong on 9/5/18.
//

#ifndef HBE_HASHBUCKET_H
#define HBE_HASHBUCKET_H


#include <Eigen/Dense>
#include <random>
#include "kernel.h"

using Eigen::MatrixXd;
using Eigen::VectorXd;

class HashBucket {
public:
    int SCALES = 1;
    vector<int> count;
    vector<VectorXd> sample;
    vector<double> wSum;
    std::uniform_real_distribution<> unif;

    HashBucket() {
        SCALES = 1;
        count.push_back(0);
    }

    HashBucket(VectorXd p) {
        unif = std::uniform_real_distribution<>(0, 1);
        sample.push_back(p);
        count.push_back(1);
    }

    HashBucket(VectorXd p, double wi) {
        unif = std::uniform_real_distribution<>(0, 1);
        sample.push_back(p);
        count.push_back(1);
        wSum.push_back(wi);
    }

    HashBucket(VectorXd p, double wi, int idx, int scales) {
        unif = std::uniform_real_distribution<>(0, 1);
        SCALES = scales;
        count = vector<int>(scales, 0);
        sample = vector<VectorXd>(scales);
        wSum = vector<double>(scales, 0);

        sample[idx] = p;
        count[idx] = 1;
        wSum[idx] = wi;
    }

//    HashBucket(const HashBucket &other) {
//        unif = std::uniform_real_distribution<>(0, 1);
//        SCALES = other.SCALES;
//        sample = other.sample;
//        count = other.count;
//        wSum = other.wSum;
//    }

    void update(VectorXd p, std::mt19937_64 &rng) {
        count[0] += 1;
        // Reservoir sampling
        float r = unif(rng);
        if (r <= 1.0 / count[0]) {
            sample[0] = p;
        }
    }

    // A-Chao
    void update(VectorXd p, double wi, std::mt19937_64 &rng) {
        count[0] += 1;

        wSum[0] += wi;
        double pi = wi / wSum[0];
        float r = unif(rng);
        if (r <= pi) {
            sample[0] = p;
        }
    }

    // A-Chao with bucket
    void update(VectorXd p, double wi, int idx, std::mt19937_64 &rng) {
        count[idx] += 1;

        wSum[idx] += wi;
        double pi = wi / wSum[idx];
        float r = unif(rng);
        if (r <= pi) {
            sample[idx] = p;
        }
    }
};


#endif //HBE_HASHBUCKET_H
