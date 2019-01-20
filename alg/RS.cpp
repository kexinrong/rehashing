#include "RS.h"
#include "mathUtils.h"
#include <iostream>
#include <unordered_set>

RS::RS(shared_ptr<MatrixXd> data, shared_ptr<Kernel> k) {
    X = data;
    kernel = k;
    numPoints = data->rows();
}


RS::RS(shared_ptr<MatrixXd> data, shared_ptr<Kernel> k, int samples) {
    int n = data->rows();
    if (samples >= n) {
        X = data;
        kernel = k;
        numPoints = data->rows();
        return;
    }
    // Sample input matrix
    std::random_device rd;
    std::mt19937_64 gen(rd());
    auto indices = mathUtils::pickSet(n, samples, gen);
    X = make_shared<MatrixXd>(samples, data->cols());
    int i = 0;
    for (auto idx: indices) {
        X->row(i) = data->row(idx);
        i ++;

    }
    kernel = k;
    numPoints = samples;
}

std::vector<double> RS::MoM(VectorXd q, int L, int m) {
    std::random_device rd;  //Will be used to obtain a seed for the random number engine
    std::mt19937_64 rng = std::mt19937_64(rd());
    std::uniform_int_distribution<int> distribution(0, numPoints - 1);

    std::vector<double> Z = std::vector<double>(L, 0);
    for (int i = 0; i < L; i ++) {
        std::vector<int> indices(m);
        for (int j = 0; j < m; j ++) {
            indices[j] = distribution(rng);
        }
        std::sort(indices.begin(), indices.end());
        for (int j = 0; j < m; j ++) {
            int idx = indices[j];
            if (m == numPoints) { idx = j; }
            Z[i] += kernel->density(q, X->row(idx));
        }
    }
    return Z;
}
