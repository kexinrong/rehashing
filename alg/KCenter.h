#ifndef EPS_KCENTER_H
#define EPS_KCENTER_H

using namespace std;

class KCenter {
public:
    int N;
    vector<int> centers;
    vector<pair<int, double>> center_samples;
    vector<pair<int, double>> rs_samples;

    shared_ptr<MatrixXd> X;
    vector<int> c_indices;

    int kc;
    int n_center;
    int kr;
    int n_rs;

    KCenter(shared_ptr<MatrixXd> data, shared_ptr<Kernel> kernel, int k, int seed, std::mt19937_64 &rng) {
        int n = data->rows();
        std::uniform_int_distribution<int> uniform(0, n - 1);

        kc = k;
        n_center = n;
        kr = 0;
        n_rs = 0;
        if (pow(k, 3) > n) {
            kc = int(pow(n, 1.0/3));
            n_center = int(pow(n, 2.0/3));
            kr = k - kc;
            n_rs = n - n_center;

            X = dataUtils::downSample(data, c_indices, n_center, rng);

            // Get random samples
            vector<int> rs;
            for (int i = 0; i < n_center; i ++) {
                if (i == 0) {
                    for (int j = 0; j < c_indices[i]; j++) {
                        rs.push_back(j);
                    }
                } else {
                    for (int j = c_indices[i-1] + 1; j < c_indices[i]; j++) {
                        rs.push_back(j);
                    }
                    if (i == n_center - 1) {
                        for (int j = c_indices[i] + 1; j < n; j++) {
                            rs.push_back(j);
                        }
                    }
                }
            }

            std::unordered_set<int> elems = mathUtils::pickSet(n_rs, kr, rng);
            for (int e : elems) {
                rs_samples.push_back(make_pair<>(rs[e], 1.0 / kr));
            }
        } else {
            for (int i = 0; i < n_center; i ++) {
                c_indices.push_back(i);
            }
            X = data;
        }
        N = X->rows();

        // Greedily find K centers
        centers.clear();
        centers.push_back(seed);
        double max_dist;
        for (int i = 0; i < kc - 1; i ++) {
            int max_idx = 0;
            max_dist = 0;
            for (int j = 0; j < N; j ++) {
                double min_dist = INT_MAX;
                for (size_t c = 0; c < centers.size(); c ++) {
                    auto delta = X->row(j) - X->row(centers[c]);
                    double dist =  delta.norm();
                    min_dist = min(min_dist, dist);
                }
                if (min_dist > max_dist) {
                    max_dist = min_dist;
                    max_idx = j;
                }
            }
            centers.push_back(max_idx);
        }

        // Find vector of weights
        VectorXd y(kc);
        for (int i = 0; i < kc; i ++) {
            for (int j = 0; j < N; j ++) {
                y(i) += kernel->density(X->row(j), X->row(centers[i]));
            }
            y(i) /= N;
        }

        MatrixXd K(kc, kc);
        for (int i = 0; i < kc; i ++) {
            for (int j = 0; j < kc; j ++) {
                K(i, j) = kernel->density(X->row(centers[j]), X->row(centers[i]));
            }
        }

        MatrixXd pinv = K.completeOrthogonalDecomposition().pseudoInverse();
        VectorXd w = pinv * y;


        for (int i = 0; i < kc; i ++) {
            center_samples.push_back(make_pair<>(c_indices[centers[i]], w(i)));
        }

    }
};

#endif //EPS_KCENTER_H
