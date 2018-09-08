//
// Created by Kexin Rong on 9/4/18.
//

#ifndef HBE_DATAUTILS_H
#define HBE_DATAUTILS_H



#include "CSVparser.h"
#include <Eigen/Dense>

using namespace parser::csv;
using Eigen::MatrixXd;

const double SQRT_PI2 = sqrt(M_PI/2);


class dataUtils {

public:
    static double estimateDiameter(MatrixXd X, double tau) {
        int n = X.rows();
        double radius = 0;
        for (int i = 0; i < n; i ++) {
            radius = max(radius, X.row(i).norm());
        }
        return min(2 * radius, log(n / tau));
    }

    static int getPower(double diameter, double beta) {
        return min((int) ceil(diameter * diameter) * 3,
                (int)ceil(beta * diameter * SQRT_PI2));
    }

    static int getPowerW(double width, double beta) {
        return (int) ceil(beta *  SQRT_PI2 * width);
    }

    static double getWidth(int power, double beta) {
        return 1 / beta / SQRT_PI2 * power;
    }

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
