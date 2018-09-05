//
// Created by Kexin Rong on 9/4/18.
//

#ifndef HBE_DATAUTILS_H
#define HBE_DATAUTILS_H

#include "CSVparser.h"
#include <Eigen/Dense>

using namespace parser::csv;
using Eigen::MatrixXd;


class dataUtils {

public:
    static MatrixXd readFile(std::string filename, bool ignoreHeader, int n, int d) {
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

#endif //HBE_DATAUTILS_H
