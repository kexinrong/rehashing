//
// Created by Kexin Rong on 1/10/19.
//

#ifndef EPS_MRSKETCH_H
#define EPS_MRSKETCH_H

#include "SketchTable.h"
#include "dataUtils.h"

class MRSketch {
public:
    vector<pair<int, double>> final_samples;

    // Multi Resolution
    MRSketch(shared_ptr<MatrixXd> X, int m, double w, int k, double tau) {
        final_samples.clear();
        int N = X->rows();
        int L = log(N * tau) / log(8);
        std::uniform_int_distribution<int> distribution(0, N - 1);

        //Will be used to obtain a seed for the random number engine
        std::random_device rd;
        std::mt19937_64 rng(rd());

        double norm_cst = 0;
        for (int i = 1; i < L + 1; i ++) {
            norm_cst += pow(4, -i);
        }
        norm_cst += pow(4, -L);

        // S0, w0
        for (size_t i = 0; i < 1; i ++) {
            int nsamples = m / 2;

            SketchTable t = SketchTable(X, w, k, rng);
            vector<pair<int, double>> samples = t.sample(nsamples, rng);
            // Output samples and normalize weights
            for (size_t j = 0; j < samples.size(); j ++) {
                samples[j].second /= nsamples;
                samples[j].second *= (1 - norm_cst);
                final_samples.push_back(samples[j]);
            }
        }

        // S1, ..., SL
        for (int l = 1; l < L + 1; l ++) {
            int tables = pow(4, l);
            double l_cst = pow(8, l);
            int nsamples =  m / 2 / l_cst;
            if (l == L) { tables *= 2;}
            for (int j = 0; j < tables; j ++) {
                // Subsample dataset
                int subsample = N / l_cst;
                std::vector<int> indices;
                shared_ptr<MatrixXd> X_sample = dataUtils::downSample(X, indices, subsample, rng);

                // HBS
                SketchTable t = SketchTable(X_sample, w, k, rng);
                vector<pair<int, double>> samples = t.sample(nsamples, rng);
                // Output samples and normalize weights
                for (size_t i = 0; i < samples.size(); i ++) {
                    samples[i].second /= nsamples;
                    samples[i].second /= tables;
                    samples[i].second *= pow(4, -l);
                    samples[i].first = indices[samples[i].first];
                    final_samples.push_back(samples[i]);
                }
            }
        }
    }

    // Single Resolution
    MRSketch(shared_ptr<MatrixXd> X, int m, double w, int k, int ntbls) {
        final_samples.clear();
        int N = X->rows();

        //Will be used to obtain a seed for the random number engine
        std::random_device rd;
        std::mt19937_64 rng(rd());

        for (int i = 0; i < ntbls; i ++) {
            int nsamples = m / ntbls;

            // Subsample dataset
            int subsample = N * 2 / ntbls;
            std::vector<int> indices;
            shared_ptr<MatrixXd> X_sample = dataUtils::downSample(X, indices, subsample, rng);

            SketchTable t = SketchTable(X_sample, w, k, rng);
            vector<pair<int, double>> samples = t.sample(nsamples, rng);
            // Output samples and normalize weights
            for (size_t j = 0; j < samples.size(); j ++) {
                samples[j].second /= nsamples;
                samples[j].second /= ntbls;
                samples[j].first = indices[samples[j].first];
                final_samples.push_back(samples[j]);
            }
        }

    }

};


#endif //EPS_MRSKETCH_H
