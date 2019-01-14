//
// Created by Kexin Rong on 9/4/18.
//

#ifndef HBE_DATAUTILS_H
#define HBE_DATAUTILS_H



#include "CSVparser.h"
#include "kernel.h"
#include "math.h"
#include <Eigen/Dense>

using namespace parser::csv;
using Eigen::MatrixXd;

const double SQRT_PI2 = sqrt(M_PI/2);
const double SQRT_2_PI = sqrt(2 * M_PI);


class dataUtils {

public:
    static double estimateDiameter(MatrixXd &X, double tau) {
        int n = X.rows();
        double radius = 0;
        for (int i = 0; i < n; i ++) {
            radius = max(radius, X.row(i).norm());
        }
        return min(2 * radius, log(n / tau));
    }

    static double estimateDiameter(shared_ptr<MatrixXd> X, double tau) {
        int n = X->rows();
        double radius = 0;
        for (int i = 0; i < n; i ++) {
            radius = max(radius, X->row(i).norm());
        }
        return min(2 * radius, log(n / tau));
    }

    static int getPowerMu(double mu, double beta) {
        return ceil(SQRT_2_PI * beta * log(1 / mu));
    }

    static int getPower(double diameter, double beta) {
        return min((int) ceil(beta * beta * diameter * diameter) * 3,
                (int)ceil(beta * diameter * SQRT_PI2));
    }

    static int getPowerW(double width, double beta) {
        return (int) ceil(beta *  SQRT_PI2 * width);
    }

    static double getWidth(int power, double beta) {
        return 1 / beta / SQRT_PI2 * power;
    }

    static MatrixXd normalizeBandwidth(MatrixXd &X, std::vector<double> bw) {
        int d = X.cols();
        for (int i = 0; i < d; i ++) {
            X.col(i) /= bw[i];
        }
        return X;
    }

    static void checkBandwidthSamples(MatrixXd &X, double eps, shared_ptr<Kernel>& kernel) {
        int n = X.rows();
        int half = n / 2;
        size_t samples = round(n / eps / eps);
        vector<double> densities(2, 0);
        for (size_t i = 0; i < samples; i ++) {
            int x = rand() % half;
            int y = rand() % half;
            densities[0] += kernel->density(X.row(x), X.row(y));
            x = rand() % half;
            y = rand() % half;
            densities[1] += kernel->density(X.row(x + half), X.row(y + half));
        }
        densities[0] /= samples;
        densities[1] /= samples;
        std::cout << "u1=" << densities[0] << ", u2=" << densities[1] << std::endl;
        std::cout << "diff=" << densities[0] - densities[1] << ", expected=" << 1/sqrt(n) << std::endl;
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
                if (field.length() < 1) {
                    data(i, j) = 0;
                } else {
                    data(i, j) = std::stof(field);
                }
                j += 1;
                if (j == d) { break; }
            }
            i += 1;
            if (i == n) { break; }
        }
        return data;
    }

    static MatrixXd readFile(std::string filename, bool ignoreHeader, int n, int startCol, int endCol) {
        std::ifstream f(filename);
        CsvParser parser(f);

        MatrixXd data(n, endCol - startCol + 1);
        bool firstRow = true;
        int i = 0;
        for (auto& row : parser) {
            if (ignoreHeader && firstRow) {
                firstRow = false;
                continue;
            }
            int j = 0;
            for (auto& field : row) {
                if (j < startCol) {
                    j += 1;
                    continue;
                }
                if (field.length() < 1) {
                    data(i, j - startCol) = 0;
                } else {
                    data(i, j - startCol) = std::stof(field);
                }
                if (j == endCol) { break; }
                j += 1;
            }
            i += 1;
            if (i == n) { break; }
        }
        return data;
    }

    static void readFile(std::string filename, bool ignoreHeader, int n, int startCol, int endCol, double *data) {
        std::ifstream infile(filename.c_str());

        int dim = endCol - startCol + 1;
        int i = 0;

        std::string line;
        std::string delim = ",";
        while (std::getline(infile, line)) {
            if (ignoreHeader && i == 0) {
                i += 1;
                continue;
            }

            size_t start = 0;
            size_t end = line.find(delim);
            if (endCol == 0) {
                end = line.length();
                if (ignoreHeader) {
                    data[(i-1) * dim] = atof(line.substr(start, end - start).c_str());
                } else {
                    data[i * dim] = atof(line.substr(start, end - start).c_str());
                }
            } else {
                int j = 0;
                while (end != std::string::npos && j <= endCol) {
                    if (j >= startCol) {
                        if (ignoreHeader) {
                            data[(i-1) * dim + j - startCol] = atof(line.substr(start, end - start).c_str());
                        } else {
                            data[i * dim + j - startCol] = atof(line.substr(start, end - start).c_str());
                        }
                    }
                    start = end + delim.length();
                    end = line.find(delim, start);
                    j += 1;
                }
                if (j == endCol && end == std::string::npos) {
                    end = line.length();
                    if (ignoreHeader) {
                        data[(i-1) * dim + j - startCol] = atof(line.substr(start, end - start).c_str());
                    } else {
                        data[i * dim + j - startCol] = atof(line.substr(start, end - start).c_str());
                    }
                }
            }

            i += 1;
            if (ignoreHeader && i == n + 1) {
                break;
            } else if (!ignoreHeader && i == n) {
                break;
            }
        }
        infile.close();
    }

    static shared_ptr<MatrixXd> downSample(shared_ptr<MatrixXd> data, vector<int>& indices, int samples, std::mt19937_64 &rng) {
        std::unordered_set<int> elems = mathUtils::pickSet(data->rows(), samples, rng);

        indices.clear();
        for (int e : elems) { indices.push_back(e); }
        std::sort(indices.begin(), indices.end());

        shared_ptr<MatrixXd> X_sample = make_shared<MatrixXd>(MatrixXd::Zero(samples, data->cols()));
        for (int k = 0; k < samples; k ++) {
            X_sample->row(k) = data->row(indices[k]);
        }
        return X_sample;
    }

    static double getAvg(vector<double>& results) {
        double sum = 0;
        for (auto& n : results) { sum += n; }
        return sum / results.size();
    }

    static double getStd(vector<double>& results) {
        double avg = getAvg(results);
        double var = 0;
        for (auto& n : results) {
            var += (n - avg) * (n - avg);
        }
        return var / results.size();
    }

};

#endif //HBE_DATAUTILS_H
