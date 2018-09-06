//
// Created by Kexin Rong on 9/5/18.
//

#ifndef HBE_E2LSH_H
#define HBE_E2LSH_H


#include "BaseLSH.h"

class E2LSH : public BaseLSH {
public:
    std::vector<HashTable> tables;

    E2LSH(MatrixXd X, int M, double w, int k, int batchSize, shared_ptr<Kernel> ker, int threads);

protected:
    vector<double> evaluateQuery(VectorXd query, int maxSamples);
};


#endif //HBE_E2LSH_H
