# -*- coding: utf-8 -*-
"""
Created on Sun Apr  3 20:07:58 2022

@author: thinka
"""
import json
import numpy as np
import matplotlib.pyplot as plt

file_path = "D:\\复旦文件\\研一下\\神经网络\\model_final.json"
with open(file_path,'r') as load_f:
    load_dict = json.load(load_f)
    
w1 = np.matrix(load_dict['w1'])
w1 = w1.T


U,sigma,VT = np.linalg.svd(w1)
total = sum(np.square(sigma))

s = np.diag(sigma[:3])
new = np.dot(U[:,:3],s)
new = np.dot(new,VT[:3,])
new = new[:,:3]

fig = plt.figure()
ax = fig.gca(projection='3d')
for i in new:
    x,y,z = i[0,0],i[0,1],i[0,2]
    ax.quiver(0,0,0,x,y,z,arrow_length_ratio=0.1)
ax.set_xlim(np.min(new[:,0]),np.max(new[:,0]))
ax.set_ylim(np.min(new[:,1]),np.max(new[:,1]))
ax.set_zlim(np.min(new[:,2]),np.max(new[:,2]))

plt.show()

w2 = np.matrix(load_dict['w2'])
w2 = w2.T
U,sigma,VT = np.linalg.svd(w2)
s = np.diag(sigma[:3])
new = np.dot(U[:,:3],s)
new = np.dot(new,VT[:3,])
new = new[:,:3]

fig = plt.figure()
ax = fig.gca(projection='3d')
for i in new:
    x,y,z = i[0,0],i[0,1],i[0,2]
    ax.quiver(0,0,0,x,y,z,arrow_length_ratio=0.1)
ax.set_xlim(np.min(new[:,0]),np.max(new[:,0]))
ax.set_ylim(np.min(new[:,1]),np.max(new[:,1]))
ax.set_zlim(np.min(new[:,2]),np.max(new[:,2]))

plt.show()
