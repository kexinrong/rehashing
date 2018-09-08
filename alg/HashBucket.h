//
// Created by Kexin Rong on 9/5/18.
//

#ifndef HBE_HASHBUCKET_H
#define HBE_HASHBUCKET_H


#include <Eigen/Dense>
#include "kernel.h"

using Eigen::MatrixXd;
using Eigen::VectorXd;

class HashBucket {
public:
    int N;
    int count;
    VectorXd sample;

    HashBucket() { count = 0; }

    HashBucket(VectorXd p) {
        sample = p;
        count = 1;
        srand (time(NULL));
    }

    void update(VectorXd p) {
        count += 1;
        // Reservoir sampling
        float r = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
        if (r <= 1.0 / count) {
            sample = p;
        }
    }
};


#endif //HBE_HASHBUCKET_H
