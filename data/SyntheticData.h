//
// Created by Kexin Rong on 9/5/18.
//

#ifndef HBE_SYNTHETICDATA_H
#define HBE_SYNTHETICDATA_H

#include "GenericInstance.h"

class SyntheticData {
public:
    static GenericInstance genMixed(int uN, int cN, int uC, int cC, int dim, double density,
                                    int numScales, double spread);

    static GenericInstance genUncorrelated(int n);
};


#endif //HBE_SYNTHETICDATA_H
