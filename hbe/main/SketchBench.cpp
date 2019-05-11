/*
 *  Sketching experiments:
 *      Compare the relative error of Uniform, HBS, Herding and SKA under varying sketch sizes.
 *      For each sketch, we output the average relative error and stanfard error of the mean.
 *
 *  Example usage:
 *      ./hbe conf/shuttle.cfg gaussian
 */

#include <chrono>
#include "../alg/RS.h"
#include "../alg/MRSketch.h"
#include "../alg/Herding.h"
#include "../alg/KCenter.h"
#include "../utils/DataIngest.h"
#include "parseConfig.h"

int main(int argc, char *argv[]) {
    if (argc < 2) {
        std::cout << "Need config file" << std::endl;
        exit(1);
    }

    char* scope = argv[2];
    parseConfig cfg(argv[1], scope);
    DataIngest data(cfg, true);
    data.estimateHashParams();

    //Will be used to obtain a seed for the random number engine
    std::random_device rd;
    std::mt19937_64 rng(rd());

    vector<int> nsamples;
    int upper = 2000;
    int s = 50;
    int interval = 200;
    while (s < upper) {
        nsamples.push_back(s);
        s += interval;
    }
    nsamples.push_back(upper);

    for (size_t idx = 0; idx < nsamples.size(); idx ++) {
        int m = nsamples[idx];
        std::cout << "----------------------------" << std::endl;
        std::cout << "sketch size=" << m << std::endl;
        std::unordered_set<int> elems = mathUtils::pickSet(data.M, 10000, rng);
        for (size_t iter = 0; iter < 5; iter ++) {
            RS rs(data.X_ptr, data.kernel, m);
            MRSketch hbs_simple = MRSketch(data.X_ptr, m, data.w, data.k, 5, rng);
            Herding herding = Herding(data.X_ptr, data.kernel, m, rng);
            KCenter kcenter = KCenter(data.X_ptr, data.kernel, m, 1, rng);

            // Evaluate errors on random samples
            auto& hbs_samples = hbs_simple.final_samples;
            auto& h_samples = herding.samples;
            auto& hc_samples = kcenter.center_samples;
            auto& hc_rs_samples = kcenter.rs_samples;

            vector<vector<double>> err;
            for (int i = 0; i < 4; i ++) {
                std::vector<double> tmp;
                err.push_back(tmp);
            }
            for (int idx : elems) {
            // for (int idx = 0; idx < M; idx ++) {
                double exact_val = data.exact[idx * 2];
                // Uncomment the for loop and following line to focus on low-density queries
//                if (exact_val > 5 * tau) {continue; }
                double query_idx = data.exact[idx * 2 + 1];
                VectorXd q = data.X_ptr->row(query_idx);

                // HBS
                double est = 0;
                for (size_t j = 0; j < hbs_samples.size(); j ++) {
                    est += hbs_samples[j].second * data.kernel->density(q, data.X_ptr->row(hbs_samples[j].first));
                }
                est /= hbs_samples.size();
                err[0].push_back(fabs(est - exact_val) / max(data.tau, exact_val));

                // Uniform
                double rs_est = rs.query(q, data.tau, m);
                err[1].push_back(fabs(rs_est - exact_val) / max(data.tau, exact_val));

                // Herding
                est = 0;
                for (size_t j = 0; j < h_samples.size(); j ++) {
                    est += h_samples[j].second * data.kernel->density(q, data.X_ptr->row(h_samples[j].first));
                }
                err[2].push_back(fabs(est - exact_val) / max(data.tau, exact_val));

                // KCenter
                est = 0;
                for (size_t j = 0; j < hc_samples.size(); j ++) {
                    est += hc_samples[j].second * data.kernel->density(q, data.X_ptr->row(hc_samples[j].first));
                }
                if (hc_rs_samples.size() > 0) {
                    double est1 = 0;
                    for (size_t j = 0; j < hc_rs_samples.size(); j ++) {
                        est1 += hc_rs_samples[j].second * data.kernel->density(q,
                                data.X_ptr->row(hc_rs_samples[j].first));
                    }
                    est = est / kcenter.kc + est1 * (1 - 1/kcenter.kc);
                }
                err[3].push_back(fabs(est - exact_val) / max(data.tau, exact_val));

            }

            std::cout << "# queries: " << err[0].size() << std::endl;
            std::cout << "HBS: " << dataUtils::getAvg(err[0]) << "," << dataUtils::getSE(err[0]) << std::endl;
            std::cout << "RS: " << dataUtils::getAvg(err[1]) << "," << dataUtils::getSE(err[1]) << std::endl;
            std::cout << "Herding: " << dataUtils::getAvg(err[2]) << "," << dataUtils::getSE(err[2]) << std::endl;
            std::cout << "KCenter: " << dataUtils::getAvg(err[3]) << "," << dataUtils::getSE(err[3]) << std::endl;
            std::cout << std::endl;
        }
    }




}