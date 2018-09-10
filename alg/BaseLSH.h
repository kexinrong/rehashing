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
    vector<HashTable> tables;
    int numTables;
    double binWidth;
    int numHash;
    int numPoints;
    shared_ptr<Kernel> kernel;
    int batchSize = 100;
    int idx = 0;

    BaseLSH(MatrixXd X, int M, double w, int k, int batch, shared_ptr<Kernel> ker, int threads);
    void precomputeErf(double tau, double epsilon);
    double collision(double c, int k);

protected:
    vector<double> erfs;
    double step;
    double r;

    double evaluateQuery(VectorXd query);
    double evaluate(HashBucket buckets, VectorXd query);
    std::vector<double> MoM(VectorXd query, int L, int m);

private:
    double erfLookup(double x);

};


#endif //HBE_BASELSH_H
