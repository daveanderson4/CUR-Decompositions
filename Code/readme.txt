CUR Matrix Decompositions

Summary:
Reconstruct a data matrix using the CUR decomposition
after a column selection algorithm is performed.
Here, column selection is performed using unweighted
spectral graph sparsification (see Graph-Sparsification)

David G. Anderson
2016

Notes:
Compilation is performed on NERSC’s Cori using MKL.
Linear algebra subroutines executed using Eigen library

files:
 - CUR.cpp
     Code to test the quality of approximation while
     varying the amount of oversampling
 - makefile
     standard makefile
 - A.txt
     Data matrix.  Dimensions: 150-by-150.  The data is
     a Gaussian RBF kernel matrix (with sigma = 0.05)
     of the Iris data set from the UCI Machine Learning
     Repository.  See:
     http://archive.ics.uci.edu/ml/datasets/Iris
 - r.txt
     List of columns selected from UCS column selection
     algorithm.  See Graph-Sparsification
 - c.txt
     Identical to r.txt for this example because A is
     symmetric.