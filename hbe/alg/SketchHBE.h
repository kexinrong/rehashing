#ifndef HBE_SKETCHLSH_H
#define HBE_SKETCHLSH_H

#include "HashBucket.h"
#include "HashTable.h"
#include "SketchTable.h"
#include "MoMEstimator.h"


///
/// HBE on HBS
///
class SketchHBE : public MoMEstimator {

public:
    ///
    /// A collection of hash tables for HBE.
    ///
    vector<HashTable> tables;

    /// Build N_SKETCHES number of sketches (HBS); sample from sketches to build hash tables for HBE
    /// \param X full dataset
    /// \param M number of samples
    /// \param w bin width
    /// \param k number of hash functions
    /// \param ker kernel function
    SketchHBE(shared_ptr<MatrixXd> X, int M, double w, int k, shared_ptr<Kernel> ker);

    /// Build N_SKETCHES number of sketches (HBS); sample from sketches to build hash tables for HBE
    /// \param X full dataset
    /// \param M number of samples
    /// \param w bin width
    /// \param k number of hash functions
    /// \param scales number of weight scales in hash buckets
    /// \param ker kernel function
    SketchHBE(shared_ptr<MatrixXd> X, int M, double w, int k, int scales, shared_ptr<Kernel> ker);

    /// Build HBE from points sampled from given sketches
    /// \param X full dataset
    /// \param sketches HBS to sample points from
    /// \param indices maps a sample (index) from HBS to an index in the whole dataset
    /// \param M number of samples
    /// \param w bin width
    /// \param k number of hash functions
    /// \param ker kernel function
    /// \param rng random number generator
    SketchHBE(shared_ptr<MatrixXd> X, vector<SketchTable> &sketches, vector<vector<int>> &indices,
            int M, double w, int k, shared_ptr<Kernel> ker, std::mt19937_64& rng);

protected:
    ///
    /// Take a biased sample from a hash table via HBE.
    /// \param query query point
    /// \return normalized contribution of the biased sample
    double evaluateQuery(VectorXd query);
    vector<double> MoM(VectorXd query, int L, int m);

private:
    ///
    /// Number of hash tables
    ///
    int numTables;
    double binWidth;
    int numHash;
    vector<int> numPoints;
    shared_ptr<Kernel> kernel;
    ///
    /// Index of hash table that we should sample from next.
    ///
    int idx = 0;
    ///
    /// Number of sketches to sample from
    ///
    int N_SKETCHES = 5;

    std::mt19937_64 rng;
};


#endif //HBE_SKETCHLSH_H
