#!/usr/bin/env python
# coding: utf-8

# Libraries needed for all questions

import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
import numpy as np
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D

import math
from sklearn.metrics import confusion_matrix
from sklearn.datasets import load_iris

# procedure to generate random samples according to a normal distribution N(mu,var) in d dimensions

def random_samples(m, v, s):

	if len(m) != 1:
		n = len(v)
		cov = [[0] * n] * n
		for i in range(0,n):
			cov[i][i] = v[i]
		return multivariate_normal(m, cov, s)
	else:
		return np.random.normal(m, v, s)

# Example of generation random samples according to multivariate normal distribution
m = [0, 5]
v = [10, 30]
rv = random_samples(m, v, 50)

if(len(m) == 1):
	sns.kdeplot(rv)
	plt.show()
else:
	x = np.linspace(-10,10,500)
	y = np.linspace(-10,10,500)
	X, Y = np.meshgrid(x,y)
	pos = np.empty(X.shape + (2,))
	pos[:, :, 0] = X; pos[:, :, 1] = Y
	fig = plt.figure()
	ax = fig.gca(projection='3d')
	ax.plot_surface(X, Y, rv.pdf(pos),cmap='viridis',linewidth=0)
	ax.set_xlabel('X axis')
	ax.set_ylabel('Y axis')
	ax.set_zlabel('Z axis')
	plt.show()

# Read Synthetic data and Iris dataset

# Syntetic Data
filename = 'data_dhs_chap2.csv'
x = np.loadtxt(filename, delimiter=',', skiprows=1, usecols=[0,1,2])
y = np.loadtxt(filename, delimiter=',', skiprows=1, usecols=[3])

# Iris dataset
iris_data = load_iris()

# Mean and Covariance for the Synthetic Data
m1 = np.mean(x[y==0], axis=0)
m2 = np.mean(x[y==1], axis=0)
m3 = np.mean(x[y==2], axis=0)

cov1 = np.cov(x[y==0].T)
cov2 = np.cov(x[y==1].T)
cov3 = np.cov(x[y==2].T)

means = [m1, m2, m3]
cov = [cov1, cov2, cov3]
for i in range(3):
    print("\nCLASS [w{}]:\nMean:{}\nCovarience Matrix:\n{}".format(i+1, means[i], cov[i]))


# General procedures that will be usefull in exercise

# procedure for calculate univariate discriminant function
def uni_dis(x, m, sigma, pw):
	return -(0.5/sigma)*(x - m)*(x - m) - (0.5)*math.log(2*np.pi) - math.log(math.sqrt(sigma)) + math.log(pw)

# procedure for calculate multivariate discriminant function
def disc_fun(x, m, cov, pw):
	dim = x.shape[0]
	pw = np.array(pw)
	return -0.5*(np.dot(np.dot((x-m).T, np.linalg.inv(cov)), (x-m))) - (dim/2)*math.log(2*math.pi) - 0.5*math.log(np.linalg.det(cov)) + math.log(pw)

# procedure for calculate error percentage in training data
def error(pred, y):
	return ((pred!=y).astype(int).sum()/pred.shape[0])*100

# Procedure for calculate Euledian Distance
def euledian(x1, x2):
	return math.sqrt(np.sum((x1 - x2)**2, axis=0))

# Procedure for calculate Mahalanobis distance
def mahalanobis(x, mu, cov):
	x = np.array(x)
	mu = np.array(mu)
	return math.sqrt(np.dot(np.dot((x-mu).T, np.linalg.inv(cov)), (x-mu)))

# Examples of Eucledian and Mahalanobis Distance

print('\nEucledian Distance between [1, 2, 1] and {}:'.format(m1), euledian(np.array([1,2,1]), m1))

print('\nMahalanobis Distance between [1, 2, 1] and {}:'.format(m1), mahalanobis(np.array([1,2,1]), m1, cov1))

# procedure of compute the Decotomozer of 1 D feature space with 2 classes

def univariate(x, y, pw):

	x1 = x[:20,0]

	m = [np.mean(x1[:10]), np.mean(x1[10:20])]
	sigma = [np.cov(x1[0:10]), np.cov(x1[10:20])]

	g1x = uni_dis(x1, m[0], sigma[0], pw[0])
	g2x = uni_dis(x1, m[1], sigma[1], pw[1])

	pred = (g1x < g2x)

	return pred

''' As the rule of Discriminant Function Classification, points belong to that class whose discriminant function in Maximum'''

pred = univariate(x, y, [0.5, 0.5])
cm = confusion_matrix(y[:20], pred)
err = error(pred, y[:20])
print('\n1.) Error of dicotomozer of 1 D feature Space: {}%'.format(err))
print('Confusion Matrix: \n', cm)


# procedure of compute the Decotomozer of 2 D and 3 D feature space with 2 classes

def multivariate(x, y, pw):
	x = np.array(x)

	dim = x.shape[1]
	samples = x.shape[0]
	classes = len(pw)

	g1x = np.array([])
	g2x = np.array([])
	g3x = np.array([])

	m1 = np.array([np.mean(x[:10,0]), np.mean(x[0:10, 1])])
	m2 = np.array([np.mean(x[10:20,0]), np.mean(x[10:20, 1])])

	if dim == 3:
		m1  = np.concatenate((m1, np.array([ np.mean(x[0:10, 2]) ]) ), axis=None)
		m2  = np.concatenate((m2, np.array([ np.mean(x[10:20, 2]) ]) ), axis=None)

	sigma1 = np.cov(x[0:10, 0:dim].T)
	sigma2 = np.cov(x[10:20, 0:dim].T)

	if classes == 3:
		sigma3 = np.cov(x[20:30, 0:dim].T)
		m3 = np.array([np.mean(x[20:30,0]), np.mean(x[20:30, 1]), np.mean(x[20:30, 2]) ])

	for i in range(0,samples):
		g1 = disc_fun(np.array(x[i, 0:dim]).T, m1.T, sigma1, pw[0])
		g1x = np.append(g1x, g1)

		g2 = disc_fun(np.array(x[i, 0:dim]).T, m2.T, sigma2, pw[1])
		g2x = np.append(g2x, g2)

		if classes == 3:
			g3 = disc_fun(np.array(x[i, 0:dim]).T, m3.T, sigma3, pw[2])
			g3x = np.append(g3x, g3)
	
	g = np.concatenate((g1x.reshape(samples, 1), g2x.reshape(samples, 1)), axis = 1)
	if classes == 3:
		g = np.concatenate((g, g3x.reshape(samples, 1)), axis = 1)
    
	pred = np.argmax(g, axis=1)
	return g, pred

g, pred = multivariate(x[0:20, 0:2], y, [0.5, 0.5])
print('By the rule of maximum discriminant function, the classification results are \n', )
cm = confusion_matrix(y[:20], pred)
err = error(pred, y[:20])
print('\n2.) Error of dicotomozer of 2 D feature Space: {}%'.format(err))
print('Confusion Matrix: \n', cm)
print('\n   g1x          g2x        predictions')
print(np.concatenate((g, pred.reshape(20,1)), axis=1))

g, pred = multivariate(x[0:20, 0:3], y, [0.5, 0.5])
cm = confusion_matrix(y[:20], pred)
err = error(pred, y[:20])
print('\n3.) Error of dicotomozer of 3 D feature Space: {}%'.format(err))
print('Confusion Matrix: \n', cm)
print('\n   g1x          g2x        predictions')
print(np.concatenate((g, pred.reshape(20,1)), axis=1))


# Prediction of all the three categories data with different prior probabilities

g, pred = multivariate(x, y, [0.333, 0.333, 0.333])
cm = confusion_matrix(y, pred)
err = error(pred, y)
print('\nError of 3 class classification with prior probabilities {}: {}%'.format([0.333, 0.333, 0.333], err))
print('Confusion Matrix: \n', cm)
print('\n   g1x          g2x          g3x           predictions')
print(np.concatenate((g, pred.reshape(30,1)), axis=1))

g, pred = multivariate(x, y, [0.8, 0.1, 0.1])
cm = confusion_matrix(y, pred)
err = error(pred, y)
print('\nError of 3 class classification with prior probabilities{}: {}%'.format([0.8, 0.1, 0.1], err))
print('Confusion Matrix: \n', cm)
print('\n   g1x          g2x          g3x           predictions')
print(np.concatenate((g, pred.reshape(30,1)), axis=1))


# prediction on Test data with priors [0.333, 0.333, 0.333] and [0.8, 0.8, 0.8] using each of the category means

test_data = [[1, 2 ,1],
			 [5, 3, 2],
			 [0, 0, 0],
			 [1, 0, 0]]
test_samples = len(test_data)
test_data = np.array(test_data)

priors = [[0.333, 0.333, 0.333],
			[0.8, 0.1, 0.1]]

for i in range(len(priors)):
	g1x = np.array([])
	g2x = np.array([])
	g3x = np.array([])
	print('\nTest data Classification with prior probabilities {} are: '.format(priors[i]))
	for j in range(test_samples):
		g1 = disc_fun(test_data[j], m1, cov1, priors[i][0])
		g1x = np.append(g1x, g1)

		g2 = disc_fun(test_data[j], m2, cov2, priors[i][1])
		g2x = np.append(g2x, g2)

		g3 = disc_fun(test_data[j], m3, cov3, priors[i][2])
		g3x = np.append(g3x, g3)
	
	g_test = np.concatenate((g1x.reshape(test_samples, 1), g2x.reshape(test_samples, 1), g3x.reshape(test_samples, 1)), axis = 1)
	pred = np.argmin(g_test, axis=1)
		
	for k in range(test_samples):
		print('{} belongs to class {}'.format(test_data[k], pred[k]))


# prediction on Test data with priors [0.333, 0.333, 0.333] and [0.8, 0.8, 0.8] using Mahalanobis Distance

d1x = np.array([])
d2x = np.array([])
d3x = np.array([])
for i in range(test_samples):
	d1 = mahalanobis(test_data[i], m1, cov1)
	d1x = np.append(d1x, d1)

	d2 = mahalanobis(test_data[i], m2, cov2)
	d2x = np.append(d2x, d2)

	d3 = mahalanobis(test_data[i], m2, cov3)
	d3x = np.append(d3x, d3)

d = np.concatenate((d1x.reshape(test_samples, 1), d2x.reshape(test_samples, 1), d3x.reshape(test_samples, 1)), axis=1)
pred = np.argmin(d, axis=1)
for k in range(test_samples):
		print('{} belongs to class {}'.format(test_data[k], pred[k]))


# procedure for Discriminant Function for all three cases on IRIS Dataset

def pred_iris(x, y, pw, case):
	x = np.array(x)

	dim = x.shape[1]
	samples = x.shape[0]
	classes = len(pw)

	g1x = np.array([])
	g2x = np.array([])
	g3x = np.array([])

	m1 = np.mean(x[y==0], axis=0)
	m2 = np.mean(x[y==1], axis=0)
	m3 = np.mean(x[y==2], axis=0)

	sigma1 = np.cov(x[y==0].T)
	sigma2 = np.cov(x[y==1].T)
	sigma3 = np.cov(x[y==2].T)

	if case == 1:
		sigma = ((np.diag(((sigma1+sigma2+sigma3)/3)*np.eye(dim)).sum())/dim)*np.eye(dim)
		sigma1=sigma2=sigma3=sigma

	if case == 2:
		sigma1=sigma2=sigma3=((sigma1+sigma2+sigma3)/3)

	for i in range(samples):
		g1 = disc_fun(x[i], m1, sigma1, pw[0])
		g1x = np.append(g1x, g1)

		g2 = disc_fun(x[i], m2, sigma2, pw[1])
		g2x = np.append(g2x, g2)

		g3 = disc_fun(x[i], m3, sigma3, pw[2])
		g3x = np.append(g3x, g3)

	g = np.concatenate((g1x.reshape(samples, 1), g2x.reshape(samples, 1), g3x.reshape(samples, 1)), axis = 1)
	pred = np.argmax(g, axis=1)
	return pred

for case in range(1,4):
	pred = pred_iris(iris_data['data'], iris_data['target'], [0.333, 0.333, 0.333], case)
	cm = confusion_matrix(iris_data['target'], pred)
	print('\nConfusion Matrix for case',case, ' \n',cm)
