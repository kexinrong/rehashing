#ifndef HBE_HASHBUCKET_H
#define HBE_HASHBUCKET_H


#include <Eigen/Dense>
#include <random>
#include "kernel.h"

using Eigen::MatrixXd;
using Eigen::VectorXd;

///
/// Hash bucket for LSH tables
///
class HashBucket {
public:
    ///
    /// Store SCALES data samples per bucket, one from each weight scale.
    /// Comparing to storing 1 sample per bucket (default),
    /// storing more samples slightly longer but is more accurate.
    /// We do not recommend setting this to more than 3.
    ///
    int SCALES = 1;

    ///
    /// Count of data points that falls into the bucket
    ///
    vector<int> count;

    ///
    /// Samples of data points that falls into the bucket
    ///
    vector<VectorXd> sample;

    ///
    /// Sum of weights of all data points that falls into the bucket.
    ///
    vector<double> wSum;
    std::uniform_real_distribution<> unif;


    ///
    /// Default constructor.
    ///
    HashBucket() {
        SCALES = 1;
        count.push_back(0);
    }

    /// Build a new hash bucket for data point p.
    /// \param p a data point that belongs to the bucket
    HashBucket(VectorXd p) {
        unif = std::uniform_real_distribution<>(0, 1);
        sample.push_back(p);
        count.push_back(1);
    }

    /// Build a new hash bucket for data point p and weight wi.
    /// \param p a data point that belongs to the bucket
    /// \param wi weight of the data point
    HashBucket(VectorXd p, double wi) {
        unif = std::uniform_real_distribution<>(0, 1);
        sample.push_back(p);
        count.push_back(1);
        wSum.push_back(wi);
    }

    /// Build a new hash bucket for data point p, weight wi, scale idx out of a total of nscales scales
    /// \param p data
    /// \param wi weight
    /// \param idx index of scale that p falls into in the bucket
    /// \param nscales total number of scales
    HashBucket(VectorXd p, double wi, int idx, int nscales) {
        unif = std::uniform_real_distribution<>(0, 1);
        SCALES = nscales;
        count = vector<int>(nscales, 0);
        sample = vector<VectorXd>(nscales);
        wSum = vector<double>(nscales, 0);

        sample[idx] = p;
        count[idx] = 1;
        wSum[idx] = wi;
    }


    /// Insert point into bucket: update count and replace bucket sample with reservoir sampling.
    /// \param p data point
    /// \param rng random number generator
    void update(VectorXd p, std::mt19937_64 &rng) {
        count[0] += 1;
        // Reservoir sampling
        float r = unif(rng);
        if (r <= 1.0 / count[0]) {
            sample[0] = p;
        }
    }


    /// Insert point p with weight wi into bucket:
    /// update count and replace bucket sample with weighted reservoir sampling (A-Chao).
    /// \param p data point
    /// \param wi weight of data point
    /// \param rng random number generator
    ///
    void update(VectorXd p, double wi, std::mt19937_64 &rng) {
        count[0] += 1;

        wSum[0] += wi;
        double pi = wi / wSum[0];
        float r = unif(rng);
        if (r <= pi) {
            sample[0] = p;
        }
    }

    /// Insert point p with weight wi, belonging to scale idx into bucket:
    /// update count and replace bucket sample with weighted reservoir sampling (A-Chao).
    /// Note that samples for each scale is stored and updated separately.
    /// \param p data point
    /// \param wi weight of data point
    /// \param idx scale idx of data point
    /// \param rng random number generator
    ///
    void update(VectorXd p, double wi, int idx, std::mt19937_64 &rng) {
        count[idx] += 1;

        wSum[idx] += wi;
        double pi = wi / wSum[idx];
        float r = unif(rng);
        if (r <= pi) {
            sample[idx] = p;
        }
    }

private:

};


#endif //HBE_HASHBUCKET_H
