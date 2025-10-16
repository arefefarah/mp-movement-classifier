#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  3 18:20:04 2018

@author: dom
"""

import torch
import numpy


def hessian(loss, params, ):
    """Evaluate hessian of loss w.r.t params. Code by paul_c"""
    
    loss_grad=torch.autograd.grad(loss,params,create_graph=True)
    
    cnt = 0
    for g in loss_grad:
        g_vector = g.contiguous().view(-1) if cnt == 0 else torch.cat([g_vector, g.contiguous().view(-1)])
        cnt = 1
    l = g_vector.size(0)
    hessian = torch.zeros(l, l)
    for idx in range(l):
        grad2rd = torch.autograd.grad(g_vector[idx], params, create_graph=True)
        cnt = 0
        for g in grad2rd:
            g2 = g.contiguous().view(-1) if cnt == 0 else torch.cat([g2, g.contiguous().view(-1)])
            cnt = 1
        hessian[idx] = g2
    return hessian.cpu().data.numpy()


if __name__=="__main__":
    
    x=numpy.arange(5)
    K= torch.tensor(0.4**2 * numpy.exp( -(numpy.subtract.outer(x,x))**2/(10.0**2) ) + 1e-6*numpy.eye(len(x)),dtype=torch.float64)
    
    y=torch.tensor([2.0,3.0,-7.0,9.0,3.3],dtype=torch.float64,requires_grad=True)
    
    loss=0.5*torch.matmul(y,torch.matmul(K,y))
    
    hess=hessian(loss,y)
    
    print(numpy.linalg.norm(hess-K.detach().numpy()))
    
    