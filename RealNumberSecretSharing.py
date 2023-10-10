# -*- coding: utf-8 -*-
"""
Created on Wed Dec  9 10:02:02 2020

Implementation of the Real Number Secret Sharing scheme. 
The implementation consists of the sharing and recon algorithm of the scheme as well as multiplication and division operations.
@author: kst
"""

import random
import numpy as np
from scipy.interpolate import lagrange
import matplotlib.pyplot as plt


class RNS():
    def __init__(self, x, n, t):
        self.x = x
        self.n = n
        self.t = t
    
    def sharing(self, secret):
        xt = np.insert(np.sort(random.sample(list(self.x),self.t)), 0, 0)
        y = [np.random.normal(0, 1000) for i in range(self.t)]
        y.insert(0,secret)
        poly = lagrange(xt,y)
        shares = np.polyval(poly.coef, self.x)
        return shares.reshape(self.n,1)
    
    def rec(self,y):
        s = 0
        for j in range(self.n):
            p = 1
            for i in [k for k in range(self.n) if k != j]:
                p *= -self.x[i]/(self.x[j]-self.x[i])
            s+=y[j]*p
        return s
    
    def triplet(self):
        a = np.random.normal(0, 1000)
        b = np.random.normal(0, 1000)
        c = a*b
        return self.sharing(a), self.sharing(b), self.sharing(c)
    
    def mult(self,s1,s2):
        a,b,c = self.triplet()
        d = self.rec(s1-a)
        e = self.rec(s2-b)
        return d*e + d*b + a*e + c
        
    def div(self, s1):
        a,b,c = self.triplet()
        d = self.rec(self.mult(s1,a))
        return 1/d * a
    
