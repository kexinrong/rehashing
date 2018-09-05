//
// Created by Kexin Rong on 9/3/18.
//

#ifndef HBE_UTILS_H
#define HBE_UTILS_H

#include <Eigen/Dense>
#include <math.h>
#include "utils/CSVparser.h"

using namespace parser::csv;
using Eigen::MatrixXd;

class Utils {

public:
    static const double E1;
    static const double E2;

    static double expRelVar(double mu) { return E1 / sqrt(mu); }
    static double gaussRelVar(double mu) { return E2 / sqrt(mu); }
    static double randomRelVar(double mu) { return 1 / mu; }

    static MatrixXd readFile(string filename, bool ignoreHeader, int n, int d) {
        std::ifstream f(filename);
        CsvParser parser(f);

        MatrixXd data(n, d);
        bool firstRow = true;
        int i = 0;
        for (auto& row : parser) {
            if (ignoreHeader && firstRow) {
                firstRow = false;
                continue;
            }
            int j = 0;
            for (auto& field : row) {
                data(i, j) = std::stof(field);
                j += 1;
                if (j == d) { break; }
            }
            i += 1;
            if (i == n) { break; }
        }
        return data;
    }


};

const double Utils::E1 = exp(1.5);
const double Utils::E2 = exp(1.854);


#endif //HBE_UTILS_H
