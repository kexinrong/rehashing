//
// Created by Kexin Rong on 9/21/18.
//
#include <stdio.h>
#include <stdlib.h>     /* atof */
#include <iostream>
#include <sstream>
#include <fstream>
#include <string.h> // for memset
#include <math.h>   // for abs
#include <algorithm>    // std::max
#include <chrono>

void readFile(std::string filename, bool ignoreHeader, int n, int startCol, int endCol, double *data) {
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

int main() {
    // The dimensionality of each sample vector.
    int d = 9;

    // The number of targets (vectors at which gauss transform is evaluated).
    int M = 43500;

    // The number of sources which will be used for the gauss transform.
    int N = 43500;

    int samples = 50;

    // The bandwidth.  NOTE: this is not the same as standard deviation since
    // the Gauss Transform sums terms exp( -||x_i - y_j||^2 / h^2 ) as opposed
    // to  exp( -||x_i - y_j||^2 / (2*sigma^2) ).  Thus, if sigma is known,
    // bandwidth can be set to h = sqrt(2)*sigma.
    double h = sqrt(2);

    double x[N * d];
    readFile("resources/shuttle_normed.csv", true, N, 0, 8, &x[0]);

    // The target array.  It is a contiguous array, where
    // ( y[j*d], y[j*d+1], ..., y[j*d+d-1] ) is the jth d-dimensional sample.
    // For example, below M = 10 and d = 7, so there are 10 rows, each
    // a 7-dimensional sample.
    double y[M * d];
    for (int i = 0; i < M * d; i ++) { y[i] = x[i]; }

    double * g = new double[M];

    srand(time(NULL));
    auto t1 = std::chrono::high_resolution_clock::now();
    double hSquare = h * h;
    for(int j = 0; j < M; j++) {
        g[j] = 0.0;
        for(int i = 0; i < samples; i++) {
            int idx = rand() % N;
            double norm = 0.0;
            for (int k = 0; k < d; k++) {
                double temp = x[(d*idx) + k] - y[(d*j) + k];
                norm = norm + (temp*temp);
            }
            g[j] = g[j] + exp(-norm/hSquare);
        }
        g[j] /= samples;
    }
    auto t2 = std::chrono::high_resolution_clock::now();

    std::cout << "Random Sampling total time: " <<
        std::chrono::duration_cast<std::chrono::milliseconds>(t2-t1).count() * 1.0 / 1e3 << std::endl;


    double exact[M];
    readFile("resources/shuttle_gaussian.txt", false, M, 0, 0, &exact[0]);
    double avg_relerror = 0;
    for (int i = 0; i < M; i ++) {
        avg_relerror +=  fabs(g[i] - exact[i]) / exact[i];
    }
    avg_relerror /= M;
    printf("Average relative error: %f\n", avg_relerror);
}