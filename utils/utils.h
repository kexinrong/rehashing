//
// Created by Kexin Rong on 9/3/18.
//

#ifndef HBE_UTILS_H
#define HBE_UTILS_H

#include <math.h>

class Utils {

public:
    static const double E1;
    static const double E2;

    static double expRelVar(double mu) { return E1 / sqrt(mu); }
    static double gaussRelVar(double mu) { return E2 / sqrt(mu); }
    static double randomRelVar(double mu) { return 1 / mu; }


};

const double Utils::E1 = exp(1.5);
const double Utils::E2 = exp(1.854);


#endif //HBE_UTILS_H
