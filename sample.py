#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import tensorflow as tf


def sample_gaussian2d(mu1, mu2, s1, s2, rho):
    mean = [mu1, mu2]
    cov = [[s1*s1, rho*s1*s2], [rho*s1*s2, s2*s2]]
    x = np.random.multivariate_normal(mean, cov, 1)
    return x[0][0], x[0][1]

def get_style_states(model, args):
    c0, c1, c2 = model.istate_cell0.c.eval(), model.istate_cell1.c.eval(), model.istate_cell2.c.eval()
    h0, h1, h2 = model.istate_cell0.h.eval(), model.istate_cell1.h.eval(), model.istate_cell2.h.eval()
    return [c0, c1, c2, h0, h1, h2] 


def sample(model, args):
    # initialize some parameters
    [c0, c1, c2, h0, h1, h2] = get_style_states(model, args) # get numpy zeros states for all three LSTMs
    prev_x = np.asarray([[[0, 0, 1]]], dtype=np.float32)     # start with a pen stroke at (0,0)

    strokes, pis, windows, phis, kappas, char_to_plot = [], [], [], [], [], [] # the data we're going to generate will go here
    args['tsteps'] = 1000
    finished = False ; i = 0
    while not finished:
        feed = {model.input: prev_x,                 model.istate_cell0.c: c0, model.istate_cell1.c: c1, model.istate_cell2.c: c2,                 model.istate_cell0.h: h0, model.istate_cell1.h: h1, model.istate_cell2.h: h2}
        fetch = [model.pi_hat, model.mu1, model.mu2, model.sigma1_hat, model.sigma2_hat, model.rho, model.eos,                  model.cell0_final_state.c, model.cell1_final_state.c, model.cell2_final_state.c,                 model.cell0_final_state.h, model.cell1_final_state.h, model.cell1_final_state.h]
        [pi_hat, mu1, mu2, sigma1_hat, sigma2_hat, rho, eos,                  c0, c1, c2, h0, h1, h2] = model.sess.run(fetch, feed)
        
        #bias stuff:
        sigma1 = np.exp(sigma1_hat - args['biases']) ; sigma2 = np.exp(sigma2_hat - args['biases']) # eqn 61 and 62
        pi_hat *= 1 + args['biases'] 
        pi = np.zeros_like(pi_hat) 
        pi[0] = np.exp(pi_hat[0]) / np.sum(np.exp(pi_hat[0]), axis=0) # softmax
        
        # choose a component from the MDN
        idx = np.random.choice(pi.shape[1], p=pi[0])
        a = eos[0][0]
        eos = 1 if 0.35 < eos[0][0] else 0 # use 0.5 as arbitrary boundary
        x1, x2 = sample_gaussian2d(mu1[0][idx], mu2[0][idx], sigma1[0][idx], sigma2[0][idx], rho[0][idx])
            
        # store the info at this time step
#         windows.append(window)
#         phis.append(phi[0])
#         kappas.append(kappa[0].T)
        pis.append(pi[0])
        print([mu1[0][idx], mu2[0][idx], sigma1[0][idx], sigma2[0][idx], rho[0][idx], a])
#         strokes.append([x1, x2, eos])
        strokes.append([mu1[0][idx], mu2[0][idx], sigma1[0][idx], sigma2[0][idx], rho[0][idx], eos])
        char_to_plot.append([x1, x2, eos])
        
        # test if finished (has the read head seen the whole ascii sequence?)
        # main_kappa_idx = np.where(alpha[0]==np.max(alpha[0]));
        # finished = True if kappa[0][main_kappa_idx] > len(input_text) else False
        finished = True if i > args['tsteps'] else False
        
        # new input is previous output
        prev_x[0][0] = np.array([x1, x2, eos], dtype=np.float32)
        i+=1

    strokes = np.vstack(strokes)
    char_to_plot = np.vstack(char_to_plot)

    # the network predicts the displacements between pen points, so do a running sum over the time dimension
    strokes[:,:2] = np.cumsum(strokes[:,:2], axis=0)
    char_to_plot[:,:2] = np.cumsum(char_to_plot[:,:2], axis=0)
    return strokes, char_to_plot

def bivariate_normal(X, Y, sigmax=1.0, sigmay=1.0,
                     mux=0.0, muy=0.0, sigmaxy=0.0):
    """
    Bivariate Gaussian distribution for equal shape *X*, *Y*.
    See `bivariate normal
    <http://mathworld.wolfram.com/BivariateNormalDistribution.html>`_
    at mathworld.
    """
    Xmu = X-mux
    Ymu = Y-muy

    rho = sigmaxy/(sigmax*sigmay)
    z = Xmu**2/sigmax**2 + Ymu**2/sigmay**2 - 2*rho*Xmu*Ymu/(sigmax*sigmay)
    denom = 2*np.pi*sigmax*sigmay*np.sqrt(1-rho**2)
    return np.exp(-z/(2*(1-rho**2))) / denom

# a heatmap for the probabilities of each pen point in the sequence
def gauss_plot(strokes, title, figsize = (20,2), save_path='.'):
    import matplotlib.mlab as mlab
    import matplotlib.cm as cm
    import matplotlib as mpl
    mpl.use('Agg')
    import matplotlib.pyplot as plt
    plt.figure(figsize=figsize) #
    buff = 1 ; epsilon = 1e-4
    minx, maxx = np.min(strokes[:,0])-buff, np.max(strokes[:,0])+buff
    miny, maxy = np.min(strokes[:,1])-buff, np.max(strokes[:,1])+buff
    delta = abs(maxx-minx)/400. ;

    x = np.arange(minx, maxx, delta)
    y = np.arange(miny, maxy, delta)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X)
    for i in range(strokes.shape[0]):
        gauss = bivariate_normal(X, Y, mux=strokes[i,0], muy=strokes[i,1],             sigmax=strokes[i,2], sigmay=strokes[i,3], sigmaxy=0) # sigmaxy=strokes[i,4] gives error
        Z += gauss/(np.max(gauss) + epsilon)

    plt.title(title, fontsize=20)
    plt.imshow(Z)
    plt.savefig(save_path)
    plt.clf() ; plt.cla()

# plots the stroke data (handwriting!)
def line_plot(strokes, title, figsize = (20,2), save_path='.'):
    import matplotlib as mpl
    mpl.use('Agg')
    import matplotlib.pyplot as plt
    plt.figure(figsize=figsize)
    eos_preds = np.where(strokes[:,-1] == 1)
    eos_preds = [0] + list(eos_preds[0]) + [-1] #add start and end indices
    for i in range(len(eos_preds)-1):
        start = eos_preds[i]+1
        stop = eos_preds[i+1]
        plt.plot(strokes[start:stop,0], strokes[start:stop,1],'b-', linewidth=2.0) #draw a stroke
    plt.title(title,  fontsize=20)
    plt.gca().invert_yaxis()
    plt.savefig(save_path)
    plt.clf() ; plt.cla()

def line_plot3(strokes, char_to_plot, title, figsize = (20,2), save_path='.'):
    import matplotlib as mpl
    mpl.use('Agg')
    import matplotlib.pyplot as plt
    plt.figure(figsize=figsize)
    eos_preds = np.where(strokes[:,-1] == 1)
    eos_preds = [0] + list(eos_preds[0]) + [-1] #add start and end indices
    for i in range(len(eos_preds)-1):
        start = eos_preds[i]+1
        stop = eos_preds[i+1]
        plt.plot(char_to_plot[start:stop,0], char_to_plot[start:stop,1],'b-', linewidth=2.0) #draw a stroke
    plt.title(title,  fontsize=20)
    plt.gca().invert_yaxis()
    plt.savefig(save_path)
    plt.clf() ; plt.cla()
    
def get_bounds(data, factor):
    min_x = 0
    max_x = 0
    min_y = 0
    max_y = 0

    abs_x = 0
    abs_y = 0
    for i in range(len(data)):
        x = float(data[i, 0]) / factor
        y = float(data[i, 1]) / factor
        abs_x += x
        abs_y += y
        min_x = min(min_x, abs_x)
        min_y = min(min_y, abs_y)
        max_x = max(max_x, abs_x)
        max_y = max(max_y, abs_y)

    return (min_x, max_x, min_y, max_y)

def line_plot2(data, title, figsize = (20,2), save_path='.', factor = 50):
    import matplotlib as mpl
    mpl.use('Agg')
    import matplotlib.pyplot as plt
    min_x, max_x, min_y, max_y = get_bounds(data, factor)
    dims = (50 + max_x - min_x, 50 + max_y - min_y)
    plt.figure(figsize=dims)
    eos_preds = np.where(data[:,-1] == 1)
    eos_preds = [0] + list(eos_preds[0]) + [-1] #add start and end indices
    for i in range(len(eos_preds)-1):
        start = eos_preds[i]+1
        stop = eos_preds[i+1]
        plt.plot(data[start:stop,0], data[start:stop,1],'b-', linewidth=2.0) #draw a stroke
    plt.title(title,  fontsize=20)
    plt.gca().invert_yaxis()
    plt.savefig(save_path)
    plt.clf() ; plt.cla()


