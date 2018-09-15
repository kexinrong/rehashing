
#include <iostream>
#include <fstream>
#include "test_id.hpp"

int main(int argc, char* argv[])
{

  // Everything should be column major
 
  double A[32] = {1, 1, 1, 1, 1, 1, 1, 1,
    2, 2, 2, 2, 2, 2, 2, 2, 
    4, 4, 4, 4, 4, 4, 4, 4,
    3, 3, 3, 3, 3, 3, 3, 3};
  int num_rows = 8;
  int num_cols = 4;
  
  // the specified rank for the ID
  int k = 1;

  double error = test_id_error(A, num_rows, num_cols, k);

  std::cout << "F-norm error for rank " << k << " ID approx: " << error << "(should be 0)\n\n";
  

  /////////////////////////////
  double C[32] = {1, 1, 1, 1, 1, 1, 1, 1,
    2, 0, 0, 0, 0, 0, 0, 0,
    4, 4, 4, 4, 4, 4, 4, 4,
    3, 3, 3, 3, 3, 3, 3, 3};
  // the specified rank for the ID
  k = 2;

  double C_error = test_id_error(C, num_rows, num_cols, k);

  std::cout << "F-norm error for rank " << k << " ID approx: " << C_error << "(should be 0)\n\n";
  

  num_cols = 5;

  double D[40] = {1, 1, 1, 1, 1, 1, 1, 1,
      2, 0, 0, 0, 0, 0, 2, 0, 
      4, 4, 4, 4, 4, 4, 4, 4,
      3, 3, 3, 3, 3, 3, 3, 3, 
      -1, -1, -1, -1, -1, -1, -1, -1};

  // the specified rank for the ID
  k = 2;

  double D_error = test_id_error(D, num_rows, num_cols, k);

  std::cout << "F-norm error for rank " << k << " ID approx: " << D_error << "(should be 0)\n\n";
  
  // This is a test matrix randomly generated in matlab
  k = 10;
  num_rows = 1000;
  num_cols = 100;
  std::ifstream in("id_test_mat.out", std::ifstream::in);


  double* B = new double[num_rows * num_cols];

  // IMPORTANT: this stores it backwards 
  /*
  for (int j = 0; j < num_cols; j++) {
    for (int i = 0; i < num_rows; i++) {
        in >> B[j + i * num_cols];
    }
  }
  */
  for (int i = 0; i < num_rows; i++) {
    for (int j = 0; j < num_cols; j++) {
        in >> B[i + j * num_rows];
    }
  }
  
  in.close();

  double b_error = test_id_error(B, num_rows, num_cols, k);

  double matlab_b_error = 0.004274;
  
  std::cout << "F-norm error for rank " << k << " ID approx: " << b_error << "  (should be " << matlab_b_error << ")\n\n";
  
  delete B;  
  
  
  return 0;
  
}




