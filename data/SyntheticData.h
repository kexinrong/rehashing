//
// Created by Kexin Rong on 9/5/18.
//

#ifndef HBE_SYNTHETICDATA_H
#define HBE_SYNTHETICDATA_H

#include "GenericInstance.h"

class SyntheticData {
public:
    static GenericInstance genMixed(int n, int dim, double density, int uncorrelatedClusters,
            int correlatedClusters, int numScales, double spread);

    static GenericInstance genUncorrelated(int n);
};


#endif //HBE_SYNTHETICDATA_H
