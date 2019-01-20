#ifndef EPS_HERDING_H
#define EPS_HERDING_H

#include <vector>
#include "kernel.h"
#include "dataUtils.h"

using namespace std;

class Herding {
public:
    vector<int> X_indices;
    vector<vector<int>> rs_indices;
    vector<pair<int, double>> samples;
    vector<double> densities;

    shared_ptr<MatrixXd> X;
    shared_ptr<Kernel> ker;
    int N;

    Herding(shared_ptr<MatrixXd> data, shared_ptr<Kernel> kernel, int m, std::mt19937_64 & rng) {
        ker = kernel;

        N = data->rows() / m;
        // Downsample X to (n / k)
        X_indices.clear();
        X = dataUtils::downSample(data, X_indices, N, rng);

        rs_indices.clear();
        densities = vector<double> (N, 0);
        std::uniform_int_distribution<int> uniform(0, N - 1);
        // Estimate density via samples
        for (int i = 0; i < N; i ++) {
            std::vector<int> indices(m);
            for (int j = 0; j < m; j ++) {
                indices[j] = uniform(rng);
            }
            std::sort(indices.begin(), indices.end());
            rs_indices.push_back(indices);

            auto q = X->row(i);
            for (int j = 0; j < m; j ++) {
                int idx = indices[j];
                if (m == N) { idx = j; }
                densities[i] += kernel->density(q, X->row(idx));
            }
            densities[i] /= m;
        }

        // Get m samples
        samples.clear();
        vector<double> new_densities(N, 0);
        for (int i = 0; i < m; i++) {
            int idx = 0;
            double max_diff = -1;
            for (int j = 0; j < N; j ++) {
                double diff = densities[j] - new_densities[j];
                if (diff > max_diff) {
                    max_diff = diff;
                    idx = j;
                }
            }
            samples.push_back(make_pair<>(X_indices[idx], 1.0 / m));
            for (int j = 0; j < N; j ++) {
                new_densities[j] = (new_densities[j] * i + ker->density(X->row(j), X->row(idx))) / (i + 1);
            }
        }
    }
};


#endif //EPS_HERDING_H
