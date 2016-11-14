
"""  The support Point algorithm implementation in python 

"""
#  algorithm model for Support points ... 

from __future__ import division 

import numpy as np 
import scipy.stats as stats
from math import exp, sqrt, pi
import matplotlib.pyplot as plt 
import pylab as pl
import time

# import seaborn as sns

# setting seed for random ...
np.random.seed(seed=111)


#  get Y samples from a normal distribution .... 
mu = 0.0
sigma = 1.

# setting parameters 
N = 5000
n = 50
p = 2
eps_init = 0.3
psi = 1.
tolerance1 = 0.05
distribution = 'gaussian'
# distribution = 'exponential'


def resampleY(N, p, distr='gaussian'):
	if distr == 'gaussian':
		Y = (sigma**2)*np.random.randn(N,p) + mu
	elif distr== 'exponential':
		Y = np.random.exponential(1/100, (N,p))
	
	return Y

#  tau ~ U(0,1)
# sample one point from Uniform distribution .
# tau = np.random.random_sample()


# for diagnostics ....
tau = 0.1

# samples from normal dist.
Y = resampleY(N, p, distribution)

# if Y is two dimensional - plotting trick 
if Y.shape[1] == 2:
	Y1 = Y[:,0]
	Y2 = Y[:,1]

# sample D from F, the generating function directly.

# if distribution == 'gaussian':
# elif distribution == 'exponential':
# 	D = np.random.exponential(1/100, (n,p))

# 	D = resampleY(n,p)

D = resampleY(n,p, distribution)

def calcDistance(x_pt, y_pt, esp):
	"""
	This function caclulates the l2 norm
	or in other words the euclidean distance 
	between two points. A small positive value 
	epsilon is also added to avoid being too
	close to zero, this is called the 
	smoothed l2 norm.
	"""
	val = np.linalg.norm((x_pt - y_pt))
	dist = np.sqrt(val** 2 + esp)
	return   dist 


#calculate Energy hat E' for sp.bcd algorithm 
#  y is an array of points, x is just one point.
def calcEnergyHat(y_array, x, Xrest, esp, n=100):
	"""
	This function calculates the E_hat term given in 
	the paper. It is basically the energy function,
	calculated by contribution of all other points in 
	the reduced dataset with the point x to be optimised.
	this is the simplified function to be optimised for 
	just a point. E hat is ised in sp.bcd algorithm.
	"""

	#  no of points in distribution. 
	N = y_array.shape[0]
	inter_dist = np.zeros(N)
	intra_dist = np.zeros(n)
	for i in range(N):
		inter_dist[i] = calcDistance(x, y[i], esp)

	for j in range(n):
		intra_dist[j] = calcDistance(x, Xrest[j], esp)

	dist1 = np.sum(inter_dist)
	dist2 = np.sum(intra_dist)
	dist1_avg = dist1 / N
	dist2_avg = dist2 / n

	final_dist = dist1_avg - dist2_avg
	return final_dist


#  calculates ENergy epsilon where X is a vector of points, Y is the posterior samples.
def calcEnergyEps(X, Y, esp, n=100):
	'''
	This function calculates the energy between X and Y
	the original energy-distance function between X vector and Y vector
	and which measures the similarity/representability of Y by X.
	Y- original distribution
	X- sample distribution

	'''

	N = Y.shape[0]
	n = X.shape[0]
	dist1 = np.zeros((n,N))
	dist2 = np.zeros((n,n))
	for i in range(n):
		for j in range(N):
			dist1[i,j] = calcDistance(X[i], Y[j], esp)

	dist1_sum = 2 * np.sum(dist1) / (N*n)
	for i in range(n):
		for j in range(n):
			dist2[i,j] = calcDistance(X[i], X[j], esp)

	dist2_sum = np.sum(dist2) / (n*n)

	energy = dist1_sum - dist2_sum
	return energy


# calculates gradient of E. 

#  using numerical diff, but could just be anything else actually better,
 # but i dont know how to do it by autodiff/symbolic diff
def calcEnergyGradient(D, Y, eps):
	'''
	function to calculate the gradient of Energy function 
	using numerical differentiation: delE/delx =  (E(x+ delx) - E(x)) / delx 
	delx = 1e-5
	'''
	D_delta = np.copy(D)
	delta = 1e-6
	D_delta = D_delta + delta
	eng = calcEnergyEps(D, Y, eps)
	eng_incr = calcEnergyEps(D_delta, Y, eps)
	gradient = (eng_incr - eng) / delta
	return gradient
	# return 0.0001


def Mmstep(x, X_rest, Y, eps):
	'''
	This function calculates the update for xi given the rest of x:
	(x1, x2, x3, x4, .. , xn)

	'''
	n = X_rest.shape[0]
	N = Y.shape[0]
	val1 = np.zeros(N)
	inv_val1 = np.zeros(N)
	y_term = np.zeros((N,p))
	xdiff = np.zeros((n,p))
	
	xdiff_sum = 0.
	y_sum = 0.

	for i in range(N):
		val1[i] = calcDistance(x, Y[i], eps)
		inv_val1[i] = 1. / val1[i] 
		y_term[i] = Y[i] / val1[i]

	sum_val1 = np.sum(val1)
	sum_inv_val1 = np.sum(inv_val1)
	y_sum = np.sum(y_term, axis=0)

	for j in range(n):
		xdiff[j] = (x - X_rest[j]) / calcDistance(x, X_rest[j], eps)

	xdiff_sum = np.sum(xdiff, axis=0)
	xdiff_sum = (xdiff_sum * N)/ n
	x_new = (xdiff_sum + y_sum) * ( 1 / sum_inv_val1)  
	# print "term1:", y_sum
	# print "term2:", xdiff_sum
	# print "term3:", sum_inv_val1
	# print "new x:", x_new
	return  x_new


start_time = time.time()
k = 0
eps_k = eps_init
D_iter = np.copy(D)
counter = 0
Y_iter = np.copy(Y)

#  main algorithm body. 
print "Outer Loop started now.."
while calcEnergyGradient(D_iter, Y, eps_k) < psi*eps_k and k < 35: 
	# print "energy gradient:", calcEnergyGradient(D_iter, Y, eps_k)
	print "Outer Loop Iteration:", k
	for i in range(n):
		# print "point number:", i
		x_init = D_iter[i]
		l = 0
		x_new = x_init 
		x_old = x_init
		
		# ridiculous small quantity for the first iter
		old_update = 0.0000001
		indices = [x for x in range(n) if x != i]
 
		# using l1 norm for convergence in the case where p is 1.
		while np.linalg.norm(x_new - x_old) > 0.01  or l == 0:
			# print "number of iters innner loop:", l
			# print "x old:", x_old
			# print "x new:", x_new
			old_update = np.linalg.norm(x_new - x_old)
			Y_iter = resampleY(N, p, distribution)
			# print "previous update:", old_update
			x_old = x_new
			x_new =  Mmstep(x_old, D_iter[indices], Y_iter, eps_k)
			l = l+1
			# too many iterations in inside loop
			if l ==145:
				continue				
		
		D_iter[i] = x_new
	eps_k = eps_k * tau
	k = k+1

print "total iters in outer 1loop:", k

# plotting for 2-D disribution
if p == 2:
	D1 = D_iter[:,0]
	D2 = D_iter[:,1]
	D_old_1 = D[:,0]
	D_old_2 = D[:,1]


# print "old D:", D
# print "new D:", D_iter

# find out time taken for execution ..
end_time = time.time()
time_taken = end_time - start_time

print "time taken for algorithm in seconds:", time_taken
####################################################
####plotting code
####################################################

#  plot for Y points ...
if p ==1 :
	blue_points, = plt.plot(Y, 'bo', label='Posterior')
elif p ==2:
	blue_points, = plt.plot(Y1, Y2, 'bo', label='Posterior')

if p==1:
	red_points, = plt.plot(D_iter,'ro', label='SP')
	green_points, = plt.plot(D, 'go', label='init sample')
elif p ==2:
	# red_cross, = plt.plot(D_old_1, D_old_2,'ro', label='SP')
	red_points, = plt.plot(D1, D2,'ro', label='SP')
	green_points, = plt.plot(D_old_1, D_old_2, 'go', label='init sample')
plt.legend([blue_points, red_points, green_points], ["Posterior", "SupportPts", 'Init'])

# plt.legend([red_cross, green_cross], [ "SupportPts", 'Init'])

# plt.savefig('gaussian-SP-1-1000-30.pdf')
plt.show()  
