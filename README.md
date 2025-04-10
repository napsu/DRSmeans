# DRS-means - Distributed Random Swap for Minimum Sum-of-Squares Clustering

This repository contains a Python implementation of the DRS-means algorithm, based on its description in [1] by _Olzhas Kozbagarov_ and _Rustam Mussabayev_.
This implementation, created by _Tapio Pahikkala_, utilizes [Random swap](https://github.com/uef-machine-learning/RandomSwap)-algorithm by _Pasi Fränti_ and _Juha Kivijärvi_ [2].

This implementation is used in our paper: 

 * N. Karmitsa, V.-P. Eronen, M.M. Mäkelä, T. Pahikkala, A. Airola, "[Stochastic limited memory bundle algorithm for clustering in big data](https://www.sciencedirect.com/science/article/pii/S0031320325003140?via%3Dihub)", _Pattern Recognition_, Vol. 165,  111654, 2025.

## Files included
* DRSmeans.py
  - Main program for DRS-means
     
* RandomSwapAlt.py
  - Accelerated Random Swap algorithm. The original algorithm is available at https://github.com/uef-machine-learning/RandomSwap


## Installation and usage

To use the code:

  1. Download DRSmeans.py and RandomSwapAlt.py.
  2. Define the data-file, no. clusters, and no. random swap iterations at the end of the DRSmeans.py file.
  3. Finally, just type "python DRSmeans.py".

The algorithm returns the MSSC-function value and used time, as well as cluster centers and distributions.

## References:

  [1] O. Kozbagarov, R. Mussabayev, "[Distributed random swap: An efficient algorithm for minimum sum-of-squares clustering](https://www.sciencedirect.com/science/article/pii/S0020025524011186)", _Information Sciences_ 681 (2024) 121204.
  
  [2] P. Fränti, J. Kivijärvi. "[Randomized local search algorithm for the clustering problem](www.cs.joensuu.fi/pub/franti/papers/Rls.ps)". _Pattern Analysis and Applications_, 3 (4), 358-369, 2000.


