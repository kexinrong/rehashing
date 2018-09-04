#include "library.h"

#include <iostream>
#include <Eigen/Dense>
#include "alg/naiveKDE.h"
#include "utils/kernel.h"
#include "utils/expkernel.h"


int main() {
    // code
    std::cout << "Naive KDE" << std::endl;
    MatrixXd data = MatrixXd::Random(10,3);
    Expkernel kernel;
    kernel.initialize(new double[3]{1,1,1}, 3);
    naiveKDE naive(data, &kernel);
    double result = naive.query(data.row(0));
    std::cout << result << std::endl;
    return 0;
}