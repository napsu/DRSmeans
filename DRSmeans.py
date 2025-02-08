"""
Implementation of the DRS-means algorithm, based on its description 
in [1]. This implementation, created by Tapio Pahikkala, modifies 
and utilizes Random swap-algorithm [2] by Pasi Fränti and Juha Kivijärvi.

This implementation is used in our paper:

N. Karmitsa, V.-P. Eronen, M.M. Mäkelä, T. Pahikkala, A. Airola, 
"Stochastic limited memory bundle algorithm for clustering in big data", 
2025.


References:
[1] O. Kozbagarov, R. Mussabayev, "Distributed random swap: An efficient 
algorithm for minimum sum-of-squares clustering", Information Sciences 681 
(2024) 121204.

[2] P. Fränti, J. Kivijärvi. "Randomized local search algorithm for the 
clustering problem". Pattern Analysis and Applications, 3 (4), 358-369, 2000.
"""

import copy
import cProfile as profile
import pstats
import random

import numpy as np
from scipy.spatial import distance

from sklearn.neighbors import NearestNeighbors

#import random_swap as rs
import RandomSwapAlt as rs_alt
import time 


def ModelCentroidsDistribution(Clist):
    C = np.array(Clist)
    clen = C.shape[0]
    #Compute all pairwise distances between cluster centers
    cdistances = distance.cdist(C, C, 'euclidean')
    constant = np.sum(cdistances)
    cinds = np.arange(clen)
    cdistances[cinds, cinds] = 1 #Set diagonal to 1 to avoid zero division
    Pclusterpair = constant / cdistances
    Pclusterpair[cinds, cinds] = 0 #Set diagonal to zero
    Pclusterpair = Pclusterpair / np.sum(Pclusterpair)
    Pcluster = np.sum(Pclusterpair, axis = 0) #Just use the marginal distribution for simplicity
    return Pcluster

def ModelObjectsDistribution(X, Pdense, C, j):
    dm = np.square(distance.cdist(X, C, 'euclidean'))
    dm[:, j] = np.inf
    squareddists = np.min(dm, axis = 1)
    pdist = squareddists / np.sum(squareddists)
    px = pdist * Pdense / np.sum(pdist * Pdense)
    return px


def PerformDRS(X,iterationsRS,iterationKmean,clusters):
    
    """
    ----------
    Performs Distribuded Random Swap -algorithm for given parameters.

    Uses the k_means function implemented in the file.

    Parameters:
    ----------
    X : N*V dimensional array with N datapoints
        The actual coordinates of the datapoints

    iterationsRS : int
        Stops random swap after the amount of iterations

    clusters : int
        Initializes random_swap with given amount of clusters

    iterationKmean : int
        Stops k-means after the amount of  iterations
    ----------

    Output:
    ----------
    clustering function value (multiplied by N*V)

    centroids : V dimensional array with C datapoints
        Predefined coordinates of centroids

    partition : scalar array with N datapoints
        Information about which datapoint belongs to which centroid
    ----------
    ----------

    """
    rng = np.random.default_rng()
    #
    h=int(X.shape[0] / 100) + 2
    
    nbrs = NearestNeighbors(n_neighbors=h, algorithm='ball_tree').fit(X)

    distances, indices = nbrs.kneighbors(X)
    hdists = distances[:, h - 1]
    constant = np.sum(hdists)
    invhdists = constant / hdists
    pdense = invhdists / np.sum(invhdists)
    
    #/* initial solution */
    # option 1.  select points ramdomly
    C = rs_alt.SelectRandomRepresentatives(X,clusters)
    P = rs_alt.OptimalPartition(C,X)
    
    #soption 2. elect points from centroid by k-means
    # kmeans = KMeans(n_clusters=clusters, random_state=0).fit(X)
    # P=kmeans.labels_
    # C=kmeans.cluster_centers_
    
    xinds = np.arange(X.shape[0])
    cinds = np.arange(len(C))
    
    err=rs_alt.ObjectiveFunction(P,C,X)
    # print("Initial MSE:",err*len(X[0])*len(X)) # multiplied by N*V
    # print("Initial MSE:",err)
    it=0
    while it <iterationsRS:
        pc = ModelCentroidsDistribution(C)
        j = rng.choice(cinds, p=pc)
        px = ModelObjectsDistribution(X, pdense, C, j)
        xi = rng.choice(xinds, p=px)
        C_new = copy.deepcopy(C)
        C_new[j] = X[xi]
        P_new = rs_alt.LocalRepartition(copy.deepcopy(P),C_new,X,j)
        P_new,C_new = rs_alt.K_means(P_new,C_new,X,iterationKmean)
        new_err=rs_alt.ObjectiveFunction(P_new,C_new,X)
        if  new_err<err :
           P=copy.deepcopy(P_new)
           C=copy.deepcopy(C_new)
           print("Iteration:",it,"MSE=",new_err*len(X[0])*len(X)) # multiplied by N*V
           # print("Iteration:",it,"MSE=",new_err)
           err=new_err
        it+=1
    return P,C

#profiler = profile.Profile()
#profiler.enable()

start = time.time() 

# Give your data here
X = np.loadtxt('/data/drift.txt') 
end = time.time() 
timedata = end-start
print(X.shape)

# Give the numbers of RSiterations k-means iterations and clusters
iterationsRS,iterationKmean,clusters = 1000, 2, 25 
print(iterationsRS,iterationKmean,clusters)

for x in range(1): # loop for random runs
    start = time.time()

    P, C = PerformDRS(X,iterationsRS,iterationKmean,clusters)
    objective_value = rs_alt.ObjectiveFunction(P,C,X)*len(X[0])*len(X) # multiplied by N*V
    end = time.time() 

    print("DRSmeans   :",objective_value,end-start,timedata)
    print(" ")


#stats = pstats.Stats(profiler).strip_dirs().sort_stats("cumtime")
#stats.print_stats(50)


#print(P) #Final cluster indices for each data
#print(C) #Cluster center vectors of n_dim

