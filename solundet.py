# -*- coding: utf-8 -*-
"""
Created on Thu Apr 12 20:52:37 2018

@author: Sugarkhuu
"""

import numpy as np
import math

delta=0.025
theta=.36
beta=.99
gamma=.95
kbar=12.6698
ybar=1.2353
rbar=0.0351
cbar=0.9186
hbar=1/3



A=np.transpose((0, -kbar, 0, 0))
B=np.transpose((0, (1-delta)*kbar, theta, -1))
C=((1, -1, -1, 0),
   (ybar, -cbar, 0, 0),
   (-1, 0, 1-theta, 0),
   (1,0,0,-1))
D=np.transpose((0, 0, 1, 0))
F=[0]
G=F
H=F
J=(0, -1, 0, beta*rbar)
K=(0, 1, 0, 0)
L=F
M=F
N=[.95]
Cinv = np.linalg.inv(C)
a=F-np.dot(np.dot(J,Cinv),A)
b=-(np.dot(np.dot(J,Cinv),B)-G+np.dot(np.dot(K,Cinv),A))
c=-np.dot(np.dot(K,Cinv),B)+H
P1=(-b+math.sqrt(b**2-4*a*c))/(2*a)
P2=(-b-math.sqrt(b**2-4*a*c))/(2*a)

if abs(P1)<1:
    P=P1
else:
    P=P2

R=-np.dot(Cinv,A*P+B)
Q=np.dot(np.dot(np.dot(J,Cinv),D)-L,N)+np.dot(np.dot(K,Cinv),D)-M
QD=np.kron(np.transpose(N),F-np.dot(np.dot(J,Cinv),A))+(np.dot(J,R)+np.dot(F,P)+G-np.dot(np.dot(K,Cinv),A))
Q=Q/QD
S=-np.dot(Cinv,A*Q+D)
print(P,Q,R,S)
