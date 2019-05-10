#ifndef EPS_MRSKETCH_H
#define EPS_MRSKETCH_H

#include "SketchTable.h"
#include "dataUtils.h"

class MRSketch {
public:
    vector<pair<int, double>> final_samples;

    // Single Resolution
    MRSketch(shared_ptr<MatrixXd> X, int m, double w, int k, int ntbls, std::mt19937_64 & rng) {
        final_samples.clear();
        int N = X->rows();

        //Will be used to obtain a seed for the random number engine
        int remain = m % ntbls;

        for (int i = 0; i < ntbls; i ++) {
            int nsamples = m / ntbls;
            if (i < remain) {nsamples ++;}

            // Subsample dataset
            int subsample = N * 2 / ntbls;
            std::vector<int> indices;
            shared_ptr<MatrixXd> X_sample = dataUtils::downSample(X, indices, subsample, rng);

            SketchTable t = SketchTable(X_sample, w, k, rng);
            vector<pair<int, double>> samples = t.sample(nsamples, rng);

            // Output samples and normalize weights
            for (size_t j = 0; j < samples.size(); j ++) {
                samples[j].first = indices[samples[j].first];
                final_samples.push_back(samples[j]);
            }
        }

    }
};


#endif //EPS_MRSKETCH_H