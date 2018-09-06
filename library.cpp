#include "library.h"

#include <iostream>
#include <Eigen/Dense>
#include "alg/naiveKDE.h"
#include "alg/RS.h"
#include "mathUtils.h"
#include "dataUtils.h"
#include "expkernel.h"


int main() {
    // Data
    MatrixXd data = dataUtils::readFile("resources/test.csv", false, 4, 3);
    std::cout << data << std::endl;
    Expkernel kernel(3);

    // Naive
    std::cout << "Naive KDE" << std::endl;
    naive naive(data, &kernel);
    double result = naive.query(data.row(0));
    std::cout << result << std::endl;

    // RS
    std::cout << "RS" << std::endl;
    RS rs(data, &kernel);
    result = rs.query(data.row(0), 0.01, 3);
    std::cout << result << std::endl;

    return 0;
}