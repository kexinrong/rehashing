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
    int N;
    int count;
    VectorXd sample;
    double wSum;
    std::uniform_real_distribution<> unif;

    HashBucket() { count = 0; }

    HashBucket(VectorXd p) {
        unif = std::uniform_real_distribution<>(0, 1);
        sample = p;
        count = 1;
    }

    HashBucket(VectorXd p, double wi) {
        unif = std::uniform_real_distribution<>(0, 1);
        sample = p;
        count = 1;
        wSum = wi;
    }

    void update(VectorXd p, std::mt19937_64 &rng) {
        count += 1;
        // Reservoir sampling
        float r = unif(rng);
        if (r <= 1.0 / count) {
            sample = p;
        }
    }

    // A-Chao
    void update(VectorXd p, double wi, std::mt19937_64 &rng) {
        count += 1;

        wSum += wi;
        double pi = wi / wSum;
        float r = unif(rng);
        if (r <= pi) {
            sample = p;
        }
    }
};


#endif //HBE_HASHBUCKET_H
