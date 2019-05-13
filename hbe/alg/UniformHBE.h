#ifndef HBE_BASELSH_H
#define HBE_BASELSH_H

#include "HashBucket.h"
#include "HashTable.h"
#include "MoMEstimator.h"

///
/// HBE on Uniform samples
///
class UniformHBE : public MoMEstimator {
public:
    ///
    /// A collection of hash tables for HBE.
    ///
    vector<HashTable> tables;

    UniformHBE();

    ///
    /// \param X dataset
    /// \param M number of samples
    /// \param w bin width
    /// \param k number of hash functions
    /// \param ker kernel function
    /// \param subsample build table on a random subsample number of points from the original dataset
    UniformHBE(shared_ptr<MatrixXd> X, int M, double w, int k, shared_ptr<Kernel> ker, int subsample);

protected:
    ///
    /// Take a biased sample from a hash table via HBE.
    /// \param query query point
    /// \return normalized contribution of the biased sample
    double evaluateQuery(VectorXd query);
    std::vector<double> MoM(VectorXd query, int L, int m);

private:
    ///
    /// Number of hash tables
    ///
    int numTables;
    double binWidth;
    int numHash;
    int numPoints;
    shared_ptr<Kernel> kernel;
    ///
    /// Index of hash table that we should sample from next.
    ///
    int idx = 0;
    std::mt19937_64 rng;

};


#endif //HBE_BASELSH_H
