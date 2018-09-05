#include "library.h"

#include <iostream>
#include <Eigen/Dense>
#include "alg/naiveKDE.h"
#include "utils/utils.h"
#include "utils/expkernel.h"


int main() {
    // code
    std::cout << "Naive KDE" << std::endl;
    MatrixXd data = Utils::readFile("resources/test.csv", false, 4, 3);
    std::cout << data << std::endl;
    Expkernel kernel;
    kernel.initialize(new double[3]{1,1,1}, 3);
    naiveKDE naive(data, &kernel);
    double result = naive.query(data.row(0));
    std::cout << result << std::endl;

    return 0;
}