//
// Created by Kexin Rong on 9/5/18.
//

#ifndef HBE_SYNTHETICDATA_H
#define HBE_SYNTHETICDATA_H

#include "GenericInstance.h"

class SyntheticData {
public:
    static GenericInstance genMixed(int uN, int cN, int uC, int cC, int dim, double density,
                                    int numScales, double spread, shared_ptr<Kernel> kernel);

    static GenericInstance genSingle(int pts, int clusters, int dim, double density,
            int numScales, double spread, shared_ptr<Kernel> kernel);

    static GenericInstance genUncorrelated(int n);
};


#endif //HBE_SYNTHETICDATA_H
