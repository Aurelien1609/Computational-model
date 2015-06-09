# -*- coding: utf-8 -*-
from dana import *
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from math import *
import matplotlib.animation as animation


# ------------------- Functions ------------------------- #

def gauss(X, width = 0.1, center = 0.0) :
  """ Transform distance in gaussian value
  
  :param X: distance between two points
  :param width: distance dispersion
  :param center: center of dispersion
  
  """
  
  return np.exp(- (X - center) ** 2 / (2 * width ** 2)) #* (1 / (width * sqrt(2 * pi)))
  
# ------------------- Parameters ------------------------- #
time = 20.0 * second
dt = 10.0 * millisecond

Vrest = 0.0 #-65 * 10 ** -3
tau = 0.1
   
STR_size = 1024
STR = np.zeros(STR_size, dtype = [("V", float),("P", float, 2)])
X = np.random.uniform(-1.0, 1.0, STR_size)
STR["P"][:,0] = X

distance = cdist(STR["P"], STR["P"])

W_STR = (1.0 * gauss(distance, 0.1) - 0.75 * gauss(distance, 1.0)) / 4.0 # division par 4 car 256 = 1024 / 4
np.fill_diagonal(W_STR, 0)

STR_I = 1.0 * gauss(X, width = 0.1, center = -0.5) + 1.0 * gauss(X, width = 0.1, center = 0.5)


# ------------------- Model ------------------------- #

_STR = zeros(STR_size, ''' dU/dt = (-V + I + 0.1 * L + Vrest) / tau ; V = np.maximum(U, 0); I; L;''')

DenseConnection(_STR('V'), _STR('L'), W_STR)
SparseConnection(STR_I, _STR('I'), 1)

save_V = np.zeros(((time / dt) + 1, STR_size))
indent = 0
@before(clock.tick)
def times(t) :
  
  global indent
  
  save_V[indent] = _STR.V
  indent += 1
    
run(time, dt)

# ------------------- Histogram ------------------------- #

fig_2 = plt.figure(figsize=(12,9))

bins = 100
hist_nb_neurons, edges_1 = np.histogram(STR["P"][:,0], bins = bins)
hist_nb_neurons = np.maximum(hist_nb_neurons, 1)
hist_potential, edges_2 = np.histogram(STR["P"][:,0], bins = bins, weights = save_V[0])
hist = hist_potential / hist_nb_neurons
x = np.linspace(-1, 1, bins)

plt.subplot(211)
V = plt.plot(x, hist)
plt.ylim(0, np.max(hist))
plt.title("Frequence de decharge des neurones")

plt.subplot(212)
hist_I, a = np.histogram(STR["P"][:,0], bins = bins, weights = _STR.I)
hist_I = hist_I / hist_nb_neurons
plt.title("Courant de Stimulation")
plt.plot(x, hist_I, linewidth=1, color='blue', label='I(x)')
plt.legend(['I(x)'], 'upper right')

def updatefig_2(i) : 
  
  if i < len(save_V) - 1 :
    plt.subplot(211)
    hist_potential, edges_2 = np.histogram(STR["P"][:,0], bins = bins, weights = save_V[i])
    hist = hist_potential / hist_nb_neurons
    V[0].set_ydata(hist)
    plt.ylim(0, np.max(hist))
    plt.subplot(212)
    
ani = animation.FuncAnimation(fig_2, updatefig_2, interval = 100)

plt.show()











  
