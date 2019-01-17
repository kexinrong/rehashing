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
    MRSketch(shared_ptr<MatrixXd> X, int m, double w, int k, double tau, std::mt19937_64 & rng) {
        final_samples.clear();
        int T = 4;
        int N = X->rows();
        int L = pow(1/tau, 1.0/4);

        // S1, ..., SL
        double wi = w;
        int ki = k;
        double norm_cst = 0;
        for (int l = 1; l < L + 1; l ++) {
            double layer_cst = (l + 1) * (l + 1);
            norm_cst += 1.0 / layer_cst;
            wi = w * layer_cst;
            ki = int(k * layer_cst);
            int nsamples =  m / T / layer_cst;
            for (int j = 0; j < T; j ++) {
                // Subsample dataset
                int subsample = N / T / layer_cst;
//                std::cout << "layer " << l << "," << subsample << "\n";
                std::vector<int> indices;
                shared_ptr<MatrixXd> X_sample = dataUtils::downSample(X, indices, subsample, rng);

                // HBS
                SketchTable t = SketchTable(X_sample, wi, ki, rng);
                vector<pair<int, double>> samples = t.sample(nsamples, rng);
                // Output samples and normalize weights
                for (size_t i = 0; i < samples.size(); i ++) {
                    //samples[i].second /= nsamples;
                    //samples[i].second /= T;
                    //samples[i].second /= layer_cst;
                    samples[i].first = indices[samples[i].first];
                    final_samples.push_back(samples[i]);
                }
            }
        }

        // S0, w0
        int remain = m - final_samples.size();
        int extra = remain % T;
        for (size_t i = 0; i < T; i ++) {
            int nsamples = remain / T;
            if (i < extra) {nsamples ++;}
            int subsample = N / T;
            std::vector<int> indices;
            shared_ptr<MatrixXd> X_sample = dataUtils::downSample(X, indices, subsample, rng);

            SketchTable t = SketchTable(X_sample, w, k, rng);
            vector<pair<int, double>> samples = t.sample(nsamples, rng);

            // Output samples and normalize weights
            for (size_t j = 0; j < samples.size(); j ++) {
                //samples[j].second /= nsamples;
                //samples[j].second /= T;
                //samples[j].second *= (1-norm_cst);
                samples[j].first = indices[samples[j].first];
                final_samples.push_back(samples[j]);
            }
        }
    }

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
                //samples[j].second /= nsamples;
                //samples[j].second /= ntbls;
                samples[j].first = indices[samples[j].first];
                final_samples.push_back(samples[j]);
            }
        }

    }

};


#endif //EPS_MRSKETCH_H
