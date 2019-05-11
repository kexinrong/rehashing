/*
 *  Computes the ground truth KDE give dataset, bandwidth and kernel.
 *  By default, we assume that each column of the dataset has been Z-NORMALIZED.
 *  If not and if the values of the column is spread across a large range,
 *  the performance of HBE might get impacted.
 *
 *  The output file contains density of M random points in the dataset.
 *  Each line in the output file contains a random query point's KDE and index (row number in the dataset).
 *
 *  Example usage:
 *      ./hbe conf/shuttle.cfg gaussian
 *
 */

#include "../utils/DataIngest.h"
#include "../alg/naiveKDE.h"
#include "parseConfig.h"
#include <chrono>


int main(int argc, char *argv[]) {
    if (argc < 2) {
        std::cout << "Need config file" << std::endl;
        exit(1);
    }

    char* scope = argv[2];
    parseConfig cfg(argv[1], scope);
    DataIngest data(cfg, false);

    // Random init
    std::random_device rd;  //Will be used to obtain a seed for the random number engine
    std::mt19937_64 rng = std::mt19937_64(rd());
    std::uniform_int_distribution<int> distribution(0, data.N - 1);

    auto t1 = std::chrono::high_resolution_clock::now();
    std::ofstream outfile(cfg.getExactPath());

    naiveKDE kde(data.X_ptr, data.kernel);


    for (int j = 0; j < data.M; j ++) {
        int idx = j;
        // Get random query
        if (data.M < data.N && data.hasQuery == 0) {
            idx = distribution(rng);
        }
        outfile << kde.query(data.X_ptr->row(idx)) << "," << idx << "\n";
    }
    outfile.close();
    auto t2 = std::chrono::high_resolution_clock::now();

    std::cout << data.M << " queries: " <<
        std::chrono::duration_cast<std::chrono::seconds>(t2-t1).count() << " sec" << std::endl;

}