#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import matplotlib.pyplot as plt

corpus = []
for line in open('file_path','r').readlines():
    corpus.append(line.strip())
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(corpus)
word = vectorizer.get_feature_names()

transformer = TfidfTransformer()

tfidf = transformer.fit_transform(X)

weight = tfidf.toarray()
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
import numpy as np
def j(x):
    cost_list = []
    for i in range(1,x+1):

        clf = KMeans(n_clusters = i)
        s = clf.fit(weight)
        centers = clf.cluster_centers_
        labels = clf.labels_
        cost =clf.inertia_ **2
        cost_list.append(cost)
    return cost_list
cost1 = j(10)
k1 = [i for i in range(1,10+1)]

plt.plot(k1,cost1,marker ="*",color = 'red')
plt.xlabel("$k$")
plt.ylabel("$J(k)$")
plt.title("Within-cluster variance $J(k)$",fontsize=14)
plt.xticks(k1)
plt.grid()
plt.show()

def j1(y1):
    '''
    input = distance_array
    retrun = eq(10)
    '''
    list1 = []
    list2 = []
    for i in range(len(y1)-1):
        detk = y1[i]-y1[i+1]
        list1.append(detk)
    for j in range(len(list1)-1):
        rk = list1[j]/list1[j+1]
        list2.append(rk)
    return list2


cost2 = j1(cost1)
x = [i for i in range(2,10)]

plt.plot(x,cost2,marker ="*",color = 'blue')
plt.xlabel("$k$")
plt.ylabel("$ {{J''(k)}}/{{J'(k)}} $")
plt.title("Maximum ratio of two consecutive decreasing amounts",fontsize=14)
plt.xticks(x)
plt.show()

'''

from sklearn.decomposition import PCA
pca = PCA(n_components = 2)
newData = pca.fit_transform(weight)
print(newData)
x = [n[0] for n in newData]
y = [n[1] for n in newData]

import matplotlib.pyplot as plt
from numpy.random import rand

fig, ax = plt.subplots()
for color in ['blue', 'pink', 'yellow']:
    plt.scatter(x,y,c= y_pred,s=100,marker='s')
    ax.scatter(x, y, c=color,
               alpha=0.3, edgecolors='none')
ax.legend()
ax.grid(True)
plt.title("KMeans")
plt.xlabel("x")
plt.ylabel("y")
plt.show()
'''
