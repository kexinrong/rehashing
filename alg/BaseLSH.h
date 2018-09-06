//
// Created by Kexin Rong on 9/5/18.
//

#ifndef HBE_BASELSH_H
#define HBE_BASELSH_H

#include "HashBucket.h"
#include "HashTable.h"
#include "MoMEstimator.h"


class BaseLSH : public MoMEstimator {
public:
    int numTables;
    double binWidth;
    int numHash;
    int numPoints;
    shared_ptr<Kernel> kernel;
    int batchSize = 100;
    int idx = 0;

    BaseLSH(MatrixXd X, int M, double w, int k, int batch, shared_ptr<Kernel> ker);

protected:
    virtual std::vector<double> evaluateQuery(VectorXd query, int maxSamples) = 0;

    std::vector<double> evaluate(std::vector<HashBucket> buckets, VectorXd query, int maxSamples);
    std::vector<double> MoM(VectorXd query, int L, int m);

};


#endif //HBE_BASELSH_H
