#ifndef HBE_BASELSH_H
#define HBE_BASELSH_H

#include "HashBucket.h"
#include "HashTable.h"
#include "MoMEstimator.h"


class BaseLSH : public MoMEstimator {
public:
    vector<HashTable> tables;
    int numTables;
    double binWidth;
    int numHash;
    int numPoints;
    shared_ptr<Kernel> kernel;
    int batchSize = 1;
    int idx = 0;
    std::mt19937_64 rng;

    BaseLSH();
    BaseLSH(shared_ptr<MatrixXd> X, int M, double w, int k, shared_ptr<Kernel> ker, int subsample);

protected:
    double evaluateQuery(VectorXd query);
    std::vector<double> MoM(VectorXd query, int L, int m);

};


#endif //HBE_BASELSH_H
