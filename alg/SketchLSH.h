//
// Created by Kexin Rong on 2018-10-26.
//

#ifndef HBE_SKETCHLSH_H
#define HBE_SKETCHLSH_H

#include "HashBucket.h"
#include "HashTable.h"
#include "MoMEstimator.h"


class SketchLSH : public MoMEstimator {

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
    int N_SKETCHES = 5;

    SketchLSH(shared_ptr<MatrixXd> X, int M, double w, int k, int batch, shared_ptr<Kernel> ker);

protected:
    vector<double> evaluateQuery(VectorXd query, int maxSamples);
    std::vector<double> MoM(VectorXd query, int L, int m);
};


#endif //HBE_SKETCHLSH_H
