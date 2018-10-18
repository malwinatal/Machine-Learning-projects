# -*- coding: utf-8 -*-
"""
Created on Tue Oct 11 15:02:30 2016

@author: Beatriz
@author: Malwina
@author: Raquel
"""
#numbers are the four features (variance, skewness 
#and curtosis of Wavelet Transformed image and the 
#entropy of the bank note image

#the class label, an integer with values 0 or 1, 
#to distinguish between real bank notes and fake bank 
#notes

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import StratifiedKFold
from sklearn.utils import shuffle
from sklearn.cross_validation import train_test_split


"""get the score in form of accuracy of the result
1 - result of score == error"""
from sklearn.cross_validation import cross_val_score

#equivalent to the linear regression but for kneighbours
from sklearn.neighbors import KNeighborsClassifier

#a way to specify different kernels for the Naive-Bayes classifier
from sklearn.neighbors.kde	import KernelDensity


#LOAD, SHUFFLE, STANDARDIZE
mat     = np.loadtxt('TP1-data.csv', delimiter=',')

#random generation
data    = shuffle(mat)
ys      = data[:,-1]
xs      = data[:,:-1]

means   = np.mean(xs, axis = 0) 
stdevs  = np.std(xs, axis = 0)
xs      = (xs-means)/stdevs



#train_test_split(...)	#test_size = 0.33; stratification (get the training and test set)
#split data into training and test sets
X_r,X_t,Y_r,Y_t = train_test_split(xs, ys, test_size=0.33, stratify = ys)    

#BUILD THE MODEL with the training set!! WITH LOGISTIC REGRESSION TO 
#FIND THE BEST NUMBER OF FEATURES
    
#estimate the classes with validation
#calculate and return 2 mean (estimated) errors -> training error and validation error

#LOGISTIC REGRESSION
################################################################################

errs = []

folds = 10 #or 5, it depends
kf = StratifiedKFold(Y_r, n_folds=folds)
c = 1
best_err = 10000

for i in range (20):
    
    reg= LogisticRegression(C=c, tol=1e-10)
    scores=cross_val_score(reg, X_r, Y_r, cv=kf)
    error=1-np.mean(scores)
    if error<best_err:
        best_c=c
        best_err=error        
            
    logC = np.log(c)  
    #print logC,':',tr_err/folds, va_err/folds
    errs.append((logC,error))
    c = 2*c
    

errs = np.array(errs)
fig = plt.figure(figsize=(8,8), frameon=False)
plt.plot(errs[:,0], errs[:,1], '-b', linewidth=3) #validation a azul
plt.show()
print('LOGISTIC REGRESSION')
print'Best C is: ', best_c
print'Best validation error is: ', best_err

reg= LogisticRegression(C=best_c, tol=1e-10)
reg.fit(X_r, Y_r)
lr_test_err=1-reg.score(X_t, Y_t)
print'Test error: ', lr_test_err
print("########################################################################")

#to crerate 1/0 matrix (1 - classified correctly, 0 - classified incorrectly)
predLR = reg.predict(X_t) #prediction
LR01 = []

for i in range(predLR.shape[0]):
    if predLR[i] == Y_t[i]:
        LR01.append(1)      
    else:
        LR01.append(0)    



#################################################################################
#K-NEAREST NEIGHBOURS
###############################################################################

errs_knn = []

lowest = 10000
for k in range (1, 39, 2):
       
    reg= KNeighborsClassifier(n_neighbors=k)
    scores=cross_val_score(reg, X_r, Y_r, cv=kf)
    error=1-np.mean(scores)
    if error<lowest:
        best_k=k
        lowest=error
        
    errs_knn.append((k, error))

errs_knn = np.array(errs_knn)
fig = plt.figure(figsize=(8,8), frameon=False)
plt.plot(errs_knn[:,0], errs_knn[:,1], '-b', linewidth=3) #validation a azul
plt.show()
print('K-NEAREST NEIGHBOURS')
print'Best K is: ', best_k
print'Best validation error is: ', lowest

knn = KNeighborsClassifier(n_neighbors=best_k)
knn.fit(X_r, Y_r)
knn_test_err=1-knn.score(X_t, Y_t)
print'Test error: ', knn_test_err
print("########################################################################")  

predKNN = knn.predict(X_t) #prediction
KNN01 = []

for i in range(predKNN.shape[0]):
    if predKNN[i] == Y_t[i]:
        KNN01.append(1)      
    else:
        KNN01.append(0)    
 

################################################################################


class KDENB:
    kdes=[]
    
    def __init__(self, bw):         
         self.bw = bw
         self.kdes=[]
           
       
    def fit(self, X, Y):
    
        #split original data X by binary class
        X=np.array(X)
        Y=np.array(Y)
        X0 = X[Y[:]==0]
        X1 = X[Y[:]==1]
       
        #calculte priori probability for each class value in log scale
        self.base0 = np.log(float(X0.shape[0])/X.shape[0])
        self.base1 = np.log(float(X1.shape[0])/X.shape[0])
        
        for ix in range(X0.shape[1]):
            
            kde0 = KernelDensity(kernel='gaussian', bandwidth = self.bw)
            kde0.fit(X0[:, [ix]])
            kde1 = KernelDensity(kernel='gaussian', bandwidth = self.bw)
            kde1.fit(X1[:, [ix]])
            
            self.kdes.append([kde0, kde1]) #should return the number of features*2 (kde0 e kde1)
    
    
    def score(self, X, Y):
      
        pred=self.predict(X, Y)       
        score=0
        for sample in range(X.shape[0]):
            if Y[sample]==pred[sample]:
                score+=1
      
        return float(score)/X.shape[0]
    
    def predict(self, X, Y):
    
        #init column vector with priori probability P0, P1
        #uma coluna com n linhas -> number of entities
        p0 = np.ones(X.shape[0])*self.base0
        p1 = np.ones(X.shape[0])*self.base1
        self.kdes=np.array(self.kdes)
    
        for ix in range(X.shape[1]):
            #evaluate density model on the feature data ix
            p0 = p0 + self.kdes[ix][0].score_samples(X[:,[ix]])
            p1 = p1 + self.kdes[ix][1].score_samples(X[:,[ix]])

        predict=np.zeros(X.shape[0])
        
        for sample in range(X.shape[0]):
            if p0[sample]<p1[sample]:
                predict[sample]=1
                            
        return np.array(predict)
        
    #gets the parameters for the estimator
    def get_params(self, deep = True): #bw == bandwith
        
        return {"bw":self.bw}

    
    #set the parameters of the estimator 
    def set_params(self, **parameters):

        for parameter, value in parameters.items():
            self.setattr(parameter, value)
        return self

################################################################################
#NAIVE BAYES
################################################################################
lowest  = 100000
errs_nb = []
bandw=0.01
for bw in range (50):     
           
    reg     = KDENB(bandw)
    scores  = cross_val_score(reg, X_r, Y_r, cv=kf)
    error   = 1 - np.mean(scores)
    
    if error < lowest:
        best_bandwidth  = bandw
        lowest          = error
       
    errs_nb.append((bandw,error))
    bandw+=0.02

errs_nb = np.array(errs_nb)
fig     = plt.figure(figsize=(8,8), frameon=False)
plt.plot(errs_nb[:,0], errs_nb[:,1], '-b', linewidth=3)
plt.show()
print('NAIVE BAYES')
print'Best Bandwidth is: ', best_bandwidth
print'Best validation error is: ', lowest

nb          = KDENB(best_bandwidth)
nb.fit(X_r, Y_r)
nb_test_err    = 1 - nb.score( X_t, Y_t)
print 'Test error: ', nb_test_err
print("########################################################################")

predNB = nb.predict(X_t, Y_t) #prediction
NB01 = []

for i in range(predNB.shape[0]):
    if predNB[i] == Y_t[i]:
        NB01.append(1)      
    else:
        NB01.append(0)  
        
    
################################################################################
#COMPARISON
################################################################################

def McNemar(test1, test2):
    E01 = []
    E10 = []
    for i in range(Y_t.shape[0]):
        if test1[i] == 0 and test2[i] == 1:
            E01.append(i)          
        if test1[i] == 1 and test2[i] == 0:
            E10.append(i)
    
    E01 = np.array(E01)
    E10 = np.array(E10)
    e01 = E01.shape[0]
    e10 = E10.shape[0]
    
    
    chi_sq = (float(((np.absolute(e01 - e10)) - 1)))**2 / (e01 + e10)
    
    return chi_sq

print"########################################################################"
print"COMPARISON"
#Logistic Regression and K-Nearest Neighbors
LR_KNN = McNemar(LR01, KNN01)
print "For LOGISTIC REGRESSION and K-NEAREST NEIGHBOURS: ", LR_KNN

#Logistic Regression and Naive Bayes
LR_NB = McNemar(LR01, NB01)
print "For LOGISTIC REGRESSION and NAIVE BAYES: ", LR_NB

#Naive Bayes and K-Nearest Neighbors
NB_KNN = McNemar(NB01, KNN01)
print "For NAIVE BAYES and K-NEAREST NEIGHBOURS: ", NB_KNN

if LR_KNN >3.84:
    if lr_test_err<knn_test_err:
        print 'Logistic Regression is better than K-Nearest Neighbours'
    else:
        print 'K-Nearest Neighbours is better than Logistic Regression'
        
if LR_NB>3.84:
    if lr_test_err<nb_test_err:
        print 'Logistic Regression is better than Naive Bayes'
    else:
        print 'Naive Bayes is better than Logistic Regression'
        
if NB_KNN>3.84:
    if knn_test_err<nb_test_err:
        print 'K-Nearest Neighbours is better than Naive Bayes'
    else:
        print 'Naive Bayes is better than K-Nearest Neighbours'
