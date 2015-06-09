# -*- coding: utf-8 -*-
from brian import *
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from math import *
import matplotlib.animation as animation

#np.random.seed(1) # Initialization of the random generator (same randoms values)

# ------------------- Functions ------------------------- #

def gauss(X, width = 0.1, center = 0.0) :
  """ Transform distance in gaussian value
  
  :param X: distance between two points
  :param width: distance dispersion
  :param center: center of dispersion
  
  """
  
  return np.exp(- (X - center) ** 2 / (2 * width ** 2)) #* (1 / (width * sqrt(2 * pi)))
  
def noise(V, level) :
  """ Initial level of the noisy neurons """
  
  V += np.random.uniform(-level, level, V.shape)
  
  return V
  
def lim_potential(X, reset, threshold) :
  """ To avoid membrane potential overcome reset value """
  
  if X.shape : 
    
    X = np.maximum(X, reset)
    X = np.minimum(X, threshold)
      
  return X
      
# ------------------- Parameters ------------------------- #

# times parameters
time = 1000 * ms
dt = 0.1 * ms
clock = Clock(dt = dt)

# Noise
k = 1e-4 * volt * second ** (.5)

# Potential time constants
STR_tau = 2.0 * ms

# Synaptics time constants
STR_tau_ge = 0.1 * ms
STR_tau_gi = 0.1 * ms

# Spike threshold
STR_threshold = -30.0 * mV

# Reset value after spikes
STR_reset = -70.0 * mV

# Resting Potential
STR_rest = -65.0 * mV
   
STR_size = 256
STR = np.zeros(STR_size, dtype = [("V", float),("P", float, 2)])
X = np.random.uniform(-1.0, 1.0, STR_size)
X = np.sort(X) # sort neurons in order to position
STR["P"][:,0] = X

distance = cdist(STR["P"], STR["P"])

W_STR = (1.0 * gauss(distance, 0.1) - 0.75 * gauss(distance, 1.0)) / (int(STR_size / 256.0)) # division par 4 car 256 = 1024 / 4
np.fill_diagonal(W_STR, 0) # avoid connections between neurons themselves
W_STR *= 10.0 # need power connection

W_STR_excitatory = (W_STR >= 0.0) * W_STR
W_STR_inhibitory = (W_STR <= 0.0) * W_STR

STR_I = 1.0 * gauss(X, width = 0.1, center = -0.5) + 1.0 * gauss(X, width = 0.1, center = 0.5)
STR_I *= 40.0


# ------------------- Model ------------------------- #
  
eqs = '''dV/dt = (-V + 0.1 * (ge + gi) + I + STR_rest + k * xi)/ STR_tau : volt
         dge/dt = -ge / STR_tau_ge : volt
         dgi/dt = -gi / STR_tau_gi : volt
         I : volt
         U = lim_potential(V, STR_reset, STR_threshold) : volt
      '''
         
_STR = NeuronGroup(STR_size, eqs, threshold = STR_threshold, reset = STR_reset, clock = clock, refractory = 5.0 * ms)

# Init values of differents variables
_STR.V = -65.0 * mV
_STR.ge = 0.0 * mV
_STR.gi = 0.0 * mV
_STR.I = 0.0 * mV

# Network connections with weights
delay_propagation = 50 * ms
STR_delay = (distance * delay_propagation) / 2.0 # max distance in [-1, 1] : abs(x - y) = 2
Ci = Connection(_STR, _STR, 'gi', delay = STR_delay, weight = W_STR_inhibitory)
Ce = Connection(_STR, _STR, 'ge', delay = STR_delay, weight = W_STR_excitatory)

d = 50 * ms
#Ci = Connection(_STR, _STR, 'gi', delay = d, weight = W_STR_inhibitory)
#Ce = Connection(_STR, _STR, 'ge', delay = d, weight = W_STR_excitatory)

# Record data
M = SpikeMonitor(_STR)
MV = StateMonitor(_STR, 'V', record = True) # if not True ==> Histogram doesn't work
Mge = StateMonitor(_STR, 'ge', record = True)
Mgi = StateMonitor(_STR, 'gi', record = True)
MI = StateMonitor(_STR, 'I', record = True)
MU = StateMonitor(_STR, 'U', record = True)

# time input
time_start = 10.0 * ms
time_end = 950.0 * ms # warning : bug with 30.0 * ms ...

@network_operation(Clock1 = EventClock(t = time_start, dt = dt))
def update(time) : 

  if time.t >= time_start and time.t < time_end : 
    _STR.I = noise(STR_I * mV, 2.5 * mV) # multiply for enough current  
    _STR.I = np.maximum(_STR.I, 0.0)
  
  if time.t == time_end :
    _STR.I = 0.0 * mV 

# Run simulation
run(time)

# ------------------- Displays ------------------------- #

# Born of displays
x0 = 0
x1 = int(time / ms)

# figure 1
figure(figsize = (14,9))
subplot(211)
raster_plot(M, title= 'Number of spikes for each neurons in Striatum')
xlim(0, time / ms)
ylim(0, STR_size)

plt.subplot(212)
bins = 50
hist_nb_neurons, edges_1 = np.histogram(STR["P"][:,0], bins = bins)
hist_nb_neurons = np.maximum(hist_nb_neurons, 1)
hist_potential, edges_2 = np.histogram(STR["P"][:,0], bins = bins, weights = MV[:, 0])
hist = hist_potential / hist_nb_neurons
x = np.linspace(-1, 1, bins)

hist_I, a = np.histogram(STR["P"][:,0], bins = bins, weights = MI[:, 200])
hist_I = hist_I / hist_nb_neurons
I = plt.plot(x, hist_I, linewidth=1, color='blue', label='I(x)')
plt.legend(['I(x)'], 'upper right')
plt.title('Input current (mV)')
plt.ylim(0, np.max(hist_I))

plt.show()

# figure : different membrane potential
figure(figsize = (14, 9))

subplot(511)
plot(MU.times / ms, MU[50] / mV)
title('Membrane potential of neurons ' + str(50))
xlabel('Time (ms)')
ylabel('U (mV)')
xlim(x0, x1)

subplot(512)
plot(MU.times / ms, MU[60] / mV)
title('Membrane potential of neurons ' + str(60))
xlabel('Time (ms)')
ylabel('U (mV)')
xlim(x0, x1)

subplot(513)
plot(MU.times / ms, MU[200] / mV)
title('Membrane potential of neurons ' + str(200))
xlabel('Time (ms)')
ylabel('U (mV)')
xlim(x0, x1)

subplot(514)
plot(MU.times / ms, MU[210] / mV)
title('Membrane potential of neurons ' + str(210))
xlabel('Time (ms)')
ylabel('U (mV)')
xlim(x0, x1)

subplot(515)
plot(MU.times / ms, MU[0] / mV)
title('Membrane potential of neurons ' + str(0))
xlabel('Time (ms)')
ylabel('U (mV)')
xlim(x0, x1)

plt.show()

# ------------------- Histograms ------------------------- #

step = 10 * ms
nb_cols = int(time / step)
nb_lines = STR_size
frequence = np.zeros((nb_lines, nb_cols))

for i in range(0, nb_cols) :
  for j in range(0, nb_lines) :
    
    frequence[j, i] = (len(np.intersect1d(np.where(M[j] < (i + 1) * step)[0], np.where(M[j] > (i) * step)[0]))) / step

fig_frequence = plt.figure(figsize = (12,9))

bins = 50
hist_nb_neurons, edges_1 = np.histogram(STR["P"][:,0], bins = bins)
hist_nb_neurons = np.maximum(hist_nb_neurons, 1)
hist_potential, edges_2 = np.histogram(STR["P"][:,0], bins = bins, weights = frequence[:, 0])
hist = hist_potential / hist_nb_neurons
x = np.linspace(-1, 1, bins)

F = plt.plot(x, hist)
ylabel('Frequency HZ')
plt.title("Frequency activiy of network")
plt.ylim(np.min(hist), np.max(hist))

def updatefig_frequence(i) : 
  
  if i < nb_cols - 1 :
    
    plt.title("Time = " + str(i))
    hist_potential, edges_2 = np.histogram(STR["P"][:,0], bins = bins, weights = frequence[:, i])
    hist = hist_potential / hist_nb_neurons
    F[0].set_ydata(hist)
    plt.ylim(np.min(hist), np.max(hist))
    
ani = animation.FuncAnimation(fig_frequence, updatefig_frequence, interval = 300)

plt.show()  














  
