/*
 *  Runtime/accuracy benchmark:
 *      Compare runtime and accuracy of HBE and RS. Candidates are:
 *          RS: RS on a reservoir of random samples
 *          Uniform HBE: HBE on a reservoir of random samples
 *          Sketch HBE: HBE on a sketch produced by HBS
 *          Sketch (3 scales) HBE: HBE on HBS, where in each hash bucket of the HBS,
 *                                 we store 3 data samples (one from each weight scale).
 *                                 Comparing to storing 1 sample per bucket (default),
 *                                 this option takes slightly longer but is more accurate.
 *
 *
 *      The relative runtime of HBE and RS can be controlled by changing the "sample_ratio"
 *      parameter in the config file.
 *
 *  Example usage:
 *      ./hbe conf/shuttle.cfg gaussian
 */



#include <algorithm>    // std::max
#include <chrono>
#include "parseConfig.h"
#include "../utils/DataIngest.h"
#include "../alg/RS.h"
#include "../alg/naiveKDE.h"
#include "../alg/BaseLSH.h"
#include "../alg/SketchLSH.h"

double relErr(double est, double exact) {
    return fabs(est - exact) / exact;
}

int main(int argc, char *argv[]) {
    if (argc < 2) {
        std::cout << "Need config file" << std::endl;
        exit(1);
    }

    char* scope = argv[2];
    parseConfig cfg(argv[1], scope);
    DataIngest data(cfg, true);
    data.estimateHashParams();

    double means = ceil(6 * data.kernel->RelVar(data.tau) / data.eps / data.eps);

    // Estimate parameters
    int tables = min((int)(means * 1.1), 1100);

    // Algorithms init
    int subsample = int(sqrt(data.N));
    std::cout << "M=" << tables << ",w=" << data.w << ",k=" << data.k << ",samples=" << subsample << std::endl;
    auto t1 = std::chrono::high_resolution_clock::now();
    BaseLSH hbe(data.X_ptr, tables, data.w, data.k, data.kernel, subsample);
    auto t2 = std::chrono::high_resolution_clock::now();
    std::cout << "Uniform Sample Table Init: " <<
        std::chrono::duration_cast<std::chrono::seconds>(t2-t1).count() << std::endl;

    t1 = std::chrono::high_resolution_clock::now();
    SketchLSH sketch(data.X_ptr, tables, data.w, data.k, data.kernel);
    t2 = std::chrono::high_resolution_clock::now();
    std::cout << "Sketch Table Init: " <<
        std::chrono::duration_cast<std::chrono::seconds>(t2-t1).count() << std::endl;

    t1 = std::chrono::high_resolution_clock::now();
    SketchLSH sketch4(data.X_ptr, tables, data.w, data.k, 3, data.kernel);
    t2 = std::chrono::high_resolution_clock::now();
    std::cout << "Sketch Table Init (3 scales): " <<
        std::chrono::duration_cast<std::chrono::seconds>(t2-t1).count() << std::endl;

    double cnt = 0;
    for (auto& t : hbe.tables) {
        cnt += t.bucket_count;
    }
    std::cout << "Average table size: " << cnt / tables << std::endl;

    int rs_size = min(int(cnt), data.N);
    std::cout << "RS reservoir size: " << rs_size << std::endl;
    RS rs(data.X_ptr, data.kernel, rs_size);


    for (int i = 0; i < 10; i ++) {
        int samples = 100 * (i + 1);
        std::cout << "------------------" << std::endl;
        std::cout << "HBE samples: " << samples << ", RS samples: " << int(samples * data.sample_ratio) << std::endl;
        hbe.totalTime = 0;
        rs.totalTime = 0;
        sketch.totalTime = 0;
        sketch4.totalTime = 0;
        vector<double> hbe_error;
        vector<double> sketch_error;
        vector<double> rs_error;
        vector<double> sketch_scale_error;

        for(int j = 0; j < data.M; j++) {
            int idx = j * 2;
            VectorXd q = data.X_ptr->row(j);
            if (data.hasQuery != 0) {
                q = data.Y_ptr->row(j);
            } else {
                if (!data.sequential) {
                    q = data.X_ptr->row(data.exact[idx + 1]);
                }
            }
            double exact_val = data.exact[idx];
            if (exact_val < data.tau) { continue; }

            double hbe_est = hbe.query(q, data.tau, samples);
            double sketch_est = sketch.query(q, data.tau, samples);
            double sketch_scale_est = sketch4.query(q, data.tau, samples);
            double rs_est = rs.query(q, data.tau, int(samples * data.sample_ratio));
            hbe_error.push_back(relErr(hbe_est, exact_val));
            rs_error.push_back(relErr(rs_est, exact_val));
            sketch_error.push_back(relErr(sketch_est,exact_val));
            sketch_scale_error.push_back(relErr(sketch_scale_est, exact_val));
        }

        std::cout << "Uniform HBE total time: " << hbe.totalTime / 1e9 << std::endl;
        std::cout << "Sketch HBE total time: " << sketch.totalTime / 1e9 << std::endl;
        std::cout << "Sketch (3 scales) HBE total time: " << sketch4.totalTime / 1e9 << std::endl;
        std::cout << "RS Sampling total time: " << rs.totalTime / 1e9 << std::endl;
        printf("Uniform HBE relative error: %f, %f, %f\n",
                dataUtils::getAvg(hbe_error), dataUtils::getStd(hbe_error), dataUtils::getMax(hbe_error));
        printf("Sketch HBE relative error:  %f, %f, %f\n",
               dataUtils::getAvg(sketch_error), dataUtils::getStd(sketch_error), dataUtils::getMax(sketch_error));
        printf("Sketch (3 scales)  HBE relative error:  %f, %f, %f\n",
               dataUtils::getAvg(sketch_scale_error),dataUtils:: getStd(sketch_scale_error), dataUtils::getMax(sketch_scale_error));
        printf("RS relative error:  %f, %f, %f\n",
               dataUtils::getAvg(rs_error), dataUtils::getStd(rs_error), dataUtils::getMax(rs_error));
    }
}