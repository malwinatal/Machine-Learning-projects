# -*- coding: utf-8 -*-
"""
Created on Tue Nov 22 14:34:41 2016

@author: Beatriz
@author: Malwina
@author: Raquel
"""

from pandas import read_csv
import numpy as np
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import axes3d
from mpl_toolkits.basemap import Basemap
from sklearn.metrics import silhouette_score
from sklearn.neighbors import KNeighborsClassifier

from sklearn import cluster #for DBSCAN AND KMEANS
from skimage.io import imsave,imread

data    = read_csv('century6_5.csv')
lat     = data.latitude.values
#print lat

lon     = data.longitude.values
#print lon

plt.figure(figsize=(10,5))
plt.plot(lon, lat, '.')
plt.show()

radius=6371
x = radius * np.cos(lat * np.pi/180) * np.cos(lon * np.pi/180)
y = radius * np.cos(lat * np.pi/180) * np.sin(lon * np.pi/180)
z = radius * np.sin(lat * np.pi/180)

data3d = np.zeros((len(x),3))
data3d[:,0] = x
data3d[:,1] = y
data3d[:,2] = z

fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection = '3d')
ax.scatter(x, y, z, s=10)
plt.show()


def plot_classes_mw(labels,lon,lat):
    """Plot seismic events using Mollweide projection.
    Arguments are the cluster labels and the longitude and latitude
    vectors of the events"""
    img = imread("Mollweide_projection_SW.jpg")        
    plt.figure(figsize=(10,5))    
    plt.subplot(111, projection="mollweide")
    plt.imshow(img,zorder=0,extent=[-np.pi,np.pi,-np.pi/2,np.pi/2],aspect=0.5)        
    nots = np.zeros(len(labels)).astype(bool)
    diffs = np.unique(labels)    
    ix = 0
    x = lon/180*np.pi
    y = lat/180*np.pi
    for lab in diffs[diffs>=0]:        
        mask = labels==lab
        nots = np.logical_or(nots,mask)        
        plt.plot(x[mask], y[mask],'o', markersize=4, mew=1,zorder=1,alpha=0.5)
        ix = ix+1                    
    mask = np.logical_not(nots)
    if((mask==True).sum()>0):
        plt.plot(x[mask], y[mask], 'k.', markersize=1, mew=1,markerfacecolor='w',zorder=1)
    plt.axis('off')
    plt.show()
    
    

'''for k in range(3, 8):
    kmeans=cluster.KMeans(n_clusters=k, random_state=0).fit(data3d)
    labels=kmeans.predict(data3d)
    plot_classes_mw(labels, lon, lat)
    score=silhouette_score(data3d, labels)
    print "for ", k, " clusters the silhouette score is:", score'''


k=4
y=np.zeros(data3d.shape[0])
knn=KNeighborsClassifier().fit(data3d, y)
dist, ind=knn.kneighbors(X=data3d, n_neighbors=k, return_distance=True)
dist= dist[:, -1]
dist=np.sort(dist)

eps_values = []
'''for x in range(1, 11):
    for i in range(dist.shape[0]-1):
        diff = dist[i+1]-dist[i]
        if diff>x:
            #print "diffrence is ", x
            #print "distance is ", dist[i]
            eps_values.append(dist[i])
            break'''

deltax = float(1)/dist.shape[0]
distScale = dist/dist[-1]
devs = []

"""for i in range(1, distScale.shape[0]):
    dev = (distScale[i]-distScale[i-1])/deltax
    devs.append(dev)
    if dev > 1:
        eps_values.append(distScale[i])"""
        
for i in range(1, distScale.shape[0]):
    dev = (distScale[i]-distScale[i-1])/deltax
    devs.append(dev)

distDev = np.zeros((distScale[:-1].shape[0],2))
distDev[:,0] = dist[:-1]
distDev[:,1] = devs
distDevs = np.sort([distDev], axis = 1)

print "dusifhsduifhdsiufhudhfudshdfuishhfudhs gowno"

'''eps = []
print distDevs

hsh = distDevs[0]
print hsh
for i in range(distDevs.shape[0]):
    if distDevs[i]>1:
        eps.append(distDevs[i])

print "dev is equal= ", eps

#print eps_values.shape[0]
epsilon = eps[1]
print 'epsilon:', epsilon'''

'''plt.figure(figsize=(10,5))
plt.plot(dist)
plt.show()

eps_values = np.array(eps_values)
'''
'''for i in range(0, eps_values.shape[0]):
    e = eps_values[i]
    db=cluster.DBSCAN(eps=e, min_samples=k)
    labels=db.fit_predict(data3d)
    plot_classes_mw(labels, lon, lat)
    score=silhouette_score(data3d, labels)
    print "score", score, "epsilon", e, "diff of distance", i+1
'''
#e = 400
#db=cluster.DBSCAN(eps=e, min_samples=4)
#labels=db.fit_predict(data3d)
#plot_classes_mw(labels, lon, lat)
#score=silhouette_score(data3d, labels)
#print score


#best score: 0.305253105938, for epsilon 515.370685783 and min_samples=13

#k = number of features*2
