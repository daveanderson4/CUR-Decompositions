#include <iostream>
#include <fstream>
#include <vector>
#include <math.h>
#include <cassert>
#include "/usr/common/software/eigen3/3.2.6/hsw/include/eigen3/Eigen/Dense"
//#include "/usr/common/usg/eigen3/3.2.0/include/eigen3/Eigen/Dense"

/*/////////////////////////////////////////////////////////////////

CUR Matrix Approximation
Relative error in the Frobenius norm is output

David G. Anderson
2016

Linear Algebra subroutines are performed with Eigen

To run: number of rows and columns must be specified

/////////////////////////////////////////////////////////////////*/

using namespace Eigen;

void read_data(MatrixXd &A) {

  // read in data matrix
  int m = A.rows();
  int n = A.cols();
  std::ifstream fin;
  fin.open("A.txt");
  assert(fin);
  for (int row = 0; row < m; ++row) {
    for (int col = 0; col < n; ++col) {
      fin >> A(row, col);
    }
  }
  fin.close();
}

void read_data(VectorXd &br, VectorXd &bc) {

  // read in data vector
  int l = br.rows();
  std::ifstream fin;
  fin.open("r.txt");
  assert(fin);
  for (int row = 0; row < l; ++row) {
    fin >> br[row];
  }
  fin.close();

  // read in second data vector
  l = bc.rows();
  fin.open("c.txt");
  assert(fin);
  for (int row = 0; row < l; ++row) {
    fin >> bc[row];
  }
  fin.close();
}

void pinv(MatrixXd A, MatrixXd &Ainv) {
  // returns pseudo-inverse of a matrix
  JacobiSVD<MatrixXd> svd(A, ComputeThinU | ComputeThinV);
  double tol = 1e-6;
  VectorXd sing_inv = svd.singularValues();
  for (long i=0; i<svd.nonzeroSingularValues(); ++i) {
    sing_inv(i) = 1 / svd.singularValues()[i];
  }
  Ainv = svd.matrixV() * sing_inv.asDiagonal() * svd.matrixU().transpose();
}

int main(int argc, char* argv[]) {

  // input parameters
  assert (argc > 2);
  int m = atoi(argv[1]);
  int n = atoi(argv[2]);

  // orginial data matrix
  MatrixXd A (m, n);
  read_data(A);

  // indices picked by column selection algorithm
  VectorXd br (m), bc (n);
  read_data(br, bc);

  // truncation rank
  int k = 20;

  // row and col oversampling parameter
  int oversamp[] = {20, 25, 30, 35, 40};
  std::vector<int> rc (oversamp, oversamp + sizeof(oversamp) / sizeof(oversamp[0]));

  // svd of A - this is not necessary to form the approximation
  // and is included only to measure performance
  JacobiSVD<MatrixXd> svd(A, ComputeThinU | ComputeThinV);

  // norms for calculating relative errors
  int  norm_A = svd.singularValues()[0]; 
  int fnorm_A = A.norm();

  // rank-k best approximation errors
  double  normk_error = svd.singularValues()[k];
  double fnormk_error = 0;
  for (int i = k; i < svd.rank(); ++i) {
    fnormk_error += svd.singularValues()[i] * svd.singularValues()[i];
  }
  fnormk_error = sqrt(fnormk_error);

  // build R and C matrices throughout iteration
  int prev_size = 0;
  MatrixXd R (0, A.cols());
  MatrixXd C (A.rows(), 0);

  // track error
  std::vector<double> err;

  // test each level of oversampling
  for (auto rc_i : rc) {

    // resize R and C
    int size_increase = rc_i - prev_size;
    R.conservativeResize(R.rows() + size_increase, R.cols());
    C.conservativeResize(C.rows(), C.cols() + size_increase);

    // add new rows and cols
    for ( ; prev_size < rc_i; ++prev_size) {
      R.block(prev_size, 0, 1, R.cols()) = A.block(br(prev_size), 0, 1, R.cols());
      C.block(0, prev_size, C.rows(), 1) = A.block(0, bc(prev_size), C.rows(), 1);
    }

    // QR decomposition for stability
    HouseholderQR<MatrixXd> qr(C);
    MatrixXd thinQ (MatrixXd::Identity(C.rows(), rc_i));
    thinQ = qr.householderQ() * thinQ;

    // form approximation with careful order of multiplication
    MatrixXd Rinv (R.cols(), R.rows()); 
    pinv(R, Rinv);
    MatrixXd U = (thinQ.transpose() * A) * Rinv;
    MatrixXd Approx = thinQ * U * R; 

    // relative error
    err.push_back((A - Approx).norm() / fnormk_error);

  }

  // print results
  for (int i = 0; i < rc.size(); ++i) {
    std::cout << "Rel. Fro. error for " << rc[i] << " oversampling: " << err[i] << std::endl;
  }
  return 0;
}



