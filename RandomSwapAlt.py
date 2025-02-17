"""
Author: Jiawei Yang (22 Nov, 2021)

Modified by Sami Sieranoja (September 2022)
Version 1.3

Accelerated version by Tapio Pahikkala (February 2025)



The method is based on the Random Swap algorithm in:

P. Franti, "Efficiency of random swap clustering", 
Journal of Big Data, 5:13, 1-29, 2018. 

The original version with recommended T=5000 iteration:

P. Franti and J. Kivijarvi, "Randomized local search algorithm for the 
clustering problem", Pattern Analysis and Applications, 3 (4), 358-369, 2000. 

"""
import random

import numpy as np
from numpy import genfromtxt

from scipy.spatial import distance
from scipy.sparse import coo_array

import copy
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
import collections



############### start random swap algorithm ###############

def PerformRS(X,iterationsRS,iterationKmean,clusters):
    
    """
    ----------
    Performs Random Swap -algorithm for given parameters.

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
    centroids : V dimensional array with C datapoints
        Predefined coordinates of centroids

    partition : scalar array with N datapoints
        Information about which datapoint belongs to which centroid
    ----------
    ----------

    """
    
    
    #/* initial solution */
    # option 1.  select points ramdomly
    C = SelectRandomRepresentatives(X,clusters)
    P = OptimalPartition(C,X)
    
    #soption 2. select points from centroid by k-means
    # kmeans = KMeans(n_clusters=clusters, random_state=0).fit(X)
    # P=kmeans.labels_
    # C=kmeans.cluster_centers_

    err=ObjectiveFunction(P,C,X)
    #print("Initial MSE:",err)
    it=0
    while it <iterationsRS:
        C_new,j= RandomSwap(C.copy(),X,clusters)
        P_new= LocalRepartition(copy.deepcopy(P),C_new,X,j)
        P_new,C_new= K_means(P_new,C_new,X,iterationKmean)
        new_err=ObjectiveFunction(P_new,C_new,X)
        if  new_err<err :
           P=copy.deepcopy(P_new)
           C=copy.deepcopy(C_new)
           #print("Iteration:",it,"MSE=",new_err)
           err=new_err
        it+=1
    return P,C


def K_means(P,C,X,T):
    #/* performs two K-means iterations */
    for i in range(T):
        #/* OptimalRepresentatives-operation should be before
        #OptimalPartition-operation, because we have previously tuned
        #partition with LocalRepartition-operation */
        C = OptimalRepresentatives(P,X,len(C))
        P = OptimalPartition(C,X)

    return P,C


def OptimalPartition(C,X):
    dm = distance.cdist(X, C, 'euclidean')
    P = np.argmin(dm, axis = 1)
    return P


def OptimalRepresentatives(P,X,clusters):
    unique, counts = np.unique(P, return_counts=True)
    row = P
    col = np.arange(len(X))
    data = (1 / counts)[P]
    coo = coo_array((data, (row, col)), shape=(clusters, len(X))).tocsr()
    avgs = coo @ X
    
    return avgs


def SelectRandomDataObject(C,X,m):
    N=len(X)
    ok = False
    while(not ok):
        i = Random(0,N)
        ok = True
        #/* eliminate duplicates */
        for j in range(m):
            if np.array_equal (C[j],X[i]):
                ok = False
    return X[i]


def SelectRandomRepresentatives(X,clusters):
    C = np.zeros((clusters, X.shape[1]))
    for i in range (clusters):
        C[i] = SelectRandomDataObject(C,X,i);
    return C


def RandomSwap(C,X,clusters):
    j = Random(0,len(C))
    C[j] = SelectRandomDataObject(C,X,clusters)
    return C,j


def LocalRepartition(P,C,X,j):
    #/* object rejection */
    dm = distance.cdist(X, C, 'euclidean')
    P = np.argmin(dm, axis = 1)
    return P



#/* this (example) objective function is sum of squared distances
#   of the data object to their cluster representatives */

def ObjectiveFunction(P,C,X):
    
    #(MSE=TSE/(N*V)
    summ = np.sum((X-C[P])**2)
    N=len(X)
    return summ/(N*len(X[0])) #calculates nMSE =(TSE/(N*V))



def Random(a,b):     #returns random number between a..b
    re=random.randint(a,b-1)
    return re

############### End of random swap algorithm ###############
