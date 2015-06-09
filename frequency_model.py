#! /usr/bin/env python
# -*- coding: utf-8 -*-
import sys
import numpy as np
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
from math import sqrt
import matplotlib.animation as animation
from dana import *

''' Spikes model in computational neuroscience with Brian library. '''

# ----------------------------------------------- Parameters --------------------------------------------------------------------- #

#np.random.seed(1) # Initialization of the random generator (same randoms values)

STR_density = 1000
STR_size    = 1.00,1.00
STR_count   = STR_size[0] * STR_size[1] * STR_density

GP_density  = 1000
GP_size     = 0.85,0.85
GP_count    = GP_size[0] * GP_size[1] * GP_density


# Striatum
STR = np.zeros(STR_count, dtype = [("V", float),     # Membrane potential
                                   ("P", float, 2)]) # Spatial position
STR["P"][:,0] = (1.0-STR_size[0])/2 + np.random.uniform(0.0, STR_size[0], len(STR))
STR["P"][:,1] = (1.0-STR_size[1])/2 + np.random.uniform(0.0, STR_size[1], len(STR))

# Globus Pallidus
GP = np.zeros(GP_count, dtype = [("V", float),     # Membrane potential
                                 ("P", float, 2)]) # Spatial position
GP["P"][:,0] = (1.0-GP_size[0])/2 + np.random.uniform(0.0, GP_size[0], len(GP))
GP["P"][:,1] = (1.0-GP_size[1])/2 + np.random.uniform(0.0, GP_size[1], len(GP))

# Striatum -> Striatum
D = cdist(STR["P"], STR["P"])
W = np.abs(np.random.normal(0.0, 0.1,(len(STR),len(STR)))) 
S = np.sign(np.random.uniform(-1, 3, (len(STR), len(STR)))) # 60% connections positive / 40% connections negative
W_STR_STR = ((W > D)) * S
#W_STR_STR = (W > D)
np.fill_diagonal(W_STR_STR, 0) # neuron can not connect himself

# Globus Pallidus -> Globus Pallidus
D = cdist(GP["P"], GP["P"])
W = np.abs(np.random.normal(0.0,0.1,(len(GP),len(GP))))
W_GP_GP = D * (W > D)

# Striatum -> Globus Pallidus
D = cdist(STR["P"], GP["P"])
W = np.abs(np.random.normal(0.0,0.1,(len(STR),len(GP))))
W_STR_GP = D * (W > D)

# Globus Pallidus -> Striatum
D = cdist(GP["P"], STR["P"])
W = np.abs(np.random.normal(0.0,0.1,(len(GP),len(STR))))
W_GP_STR = D * (W > D)

def save_connections() : np.savez("connections.npz", W_GP_STR, W_STR_GP, W_GP_GP, W_STR_STR)
def load_connections() : W_GP_STR, W_STR_GP, W_GP_GP, W_STR_STR = np.load("connections.npz")

# ----------------------------------------------- Model --------------------------------------------------------------------- #

duration = 200 * millisecond # Default trial duration
dt = 1.0 / 1024 * second # Default Time resolution : number of power 2 to avoid approximations
#dt = 1 * millisecond

STR_H = -65.0 # Resting potentiel
Threshold = -30.0 # Maximal Voltage for Spikes activity 
STR_tau = 0.1 # Time constants 
STR_N = 1 * 10 ** -3 # Noise level
V_init = -65.0 # Init potential V

# Sigmoid parameter
fmin = 0.0
fmax = 1.0 # frequency PA
slope = 1.0 # steepness of the slope
mean_freq = -40.0

# Functions

def noise(V, level) :
  """ Initial level of the noisy neurons """
  
  V *= (1 + np.random.uniform(-level, level, V.shape))
  
  return V
  
def sigmoid(V, Fmin = 0, Fmax = 1, mean_freq = 0, slope = 1) :
  """ Boltzmann sigmoid
      Returns values between [fmin, fmax] """
  
  V = Fmin + ((Fmax - Fmin) / (1.0 + np.exp((mean_freq - V) / slope)))
  
  return V
  
# Populations # 

G_STR = zeros(len(STR), 'dV/dt = (-V + I_int + I_ext + Input + STR_H)/ STR_tau; U = sigmoid(noise(V, STR_N), fmin, fmax, mean_freq, slope); I_int; I_ext; Input;')

G_STR.V = V_init + np.random.uniform(-STR_N, STR_N, G_STR.V.shape)
G_STR.U = sigmoid(G_STR.V, fmin, fmax, mean_freq, slope)

# Connectivity #

SparseConnection(G_STR('U'), G_STR('I_int'), W_STR_STR * 10) # faster computation with sparse matrix

# Electrode Stimulation # 

def input_current(voltage = 100.0, position = [0.5, 0.5], tau_Stim = 0.15) :

  ''' Add current in dv/dt equation '''

  Stim = np.array([(voltage, position)], dtype=[('V', '<f8'), ('P', '<f8', 2)])
  Distance_Stim_Neurons = cdist(Stim['P'], STR['P']) # Compute distance between electrode and neurons in STR
  Stim_Voltage = np.exp(-Distance_Stim_Neurons / tau_Stim) * voltage # Value of stim voltage
  #Stim_Voltage = voltage * (Distance_Stim_Neurons < 0.3)
    
  return Distance_Stim_Neurons, Stim_Voltage

Input = 500.0 # here, we add current
Position = [0.5, 0.5] # electrode position
tau = 0.15

dist_Stim_STR, input_STR = input_current(Input, Position, tau) 

@clock.at(1 * millisecond) # avoid 0 ms
def data(times) : 
  G_STR.Input = input_STR

@clock.at(20 * millisecond)
def data(times) :  
  G_STR.Input = 0.0

# Trial setup # 

time = int(duration / dt) + 1

STR_record_V = np.zeros((time, len(STR)))
STR_record_U = np.zeros((time, len(STR)))
STR_record_Input = np.zeros((time, len(STR)))
STR_record_I_int = np.zeros((time, len(STR)))


record_index = 0

@before(clock.tick)
def save_data(times):     
  global record_index 

  STR_record_V[record_index] = G_STR.V
  STR_record_U[record_index] = G_STR.U  
  STR_record_Input[record_index] = G_STR.Input  
  STR_record_I_int[record_index] = G_STR.I_int
  record_index += 1
        
# Simulation #
run(time = duration, dt = dt)

# ----------------------------------------------- Displays --------------------------------------------------------------------- #

def infos() :

    '''Some information about the neuronal populations for each structures'''
    print "Striatal populations: %d" % len(STR)
    print "Pallidal populations: %d" % len(GP)
    print

    C = (W_STR_STR > 0).sum(axis=1) + (W_STR_STR < 0).sum(axis=1)
    L = W_STR_STR[np.where(W_STR_STR != 0)]
    print "Collateral striatal connections"
    print "Mean number:  %g (+/- %g)" % (C.mean(), C.std())
    print "Mean length:  %g (+/- %g)" % (L.mean(), L.std())
    print

    C = (W_GP_GP > 0).sum(axis=1) + (W_GP_GP < 0).sum(axis=1)
    L = W_GP_GP[np.where(W_GP_GP != 0)]
    print "Collateral pallidal connections"
    print "Mean number:  %g (+/- %g)" % (C.mean(), C.std())
    print "Mean length:  %g (+/- %g)" % (L.mean(), L.std())
    print

    C = (W_STR_GP > 0).sum(axis=1) + (W_STR_GP < 0).sum(axis=1)
    L = W_STR_GP[np.where(W_STR_GP != 0)]
    print "Striato-pallidal connections"
    print "Mean number:  %g (+/- %g)" % (C.mean(), C.std())
    print "Mean length:  %g (+/- %g)" % (L.mean(), L.std())
    print

    print "Mean # collateral striato-pallidal connections:  %g (+/- %g)" % (C.mean(), C.std())
    C = (W_GP_STR > 0).sum(axis=1) + (W_GP_STR < 0).sum(axis=1) 
    L = W_GP_STR[np.where(W_GP_STR != 0)]
    print "Pallido-striatal connections"
    print "Mean number:  %g (+/- %g)" % (C.mean(), C.std())
    print "Mean length:  %g (+/- %g)" % (L.mean(), L.std())
    print
    
infos()

def connections() :
  
  ''' The graph of connections of afferent and efferent neurons '''
  
  def on_pick(event):
      
      '''Show neuronals afferants with a click and efferents with control and click '''
      button = event.mouseevent.button
      index = event.ind[0]

      # Clear previous selection/connections
      STR_selection_plot.set_data([],[])
      STR_connections_plot.set_data([],[])
      GP_selection_plot.set_data([],[])
      GP_connections_plot.set_data([],[])

      # --- Output connections ---
      if button == 1:
          STR_connections_plot.set_color('red')
          GP_connections_plot.set_color('red')

          if event.artist == STR_plot:
              x,y = STR[index]['P']
              STR_selection_plot.set_data([x],[y])

              I = W_STR_STR[:,index].nonzero()
              STR_connections_plot.set_data(STR['P'][I,0], STR['P'][I,1])

              I = W_GP_STR[:,index].nonzero()
              GP_connections_plot.set_data(GP['P'][I,0], GP['P'][I,1])
          elif event.artist == GP_plot:
              x,y = GP[index]['P']
              GP_selection_plot.set_data([x],[y])

              I = W_GP_GP[:,index].nonzero()
              GP_connections_plot.set_data(GP['P'][I,0], GP['P'][I,1])

              I = W_STR_GP[:,index].nonzero()
              STR_connections_plot.set_data(STR['P'][I,0], STR['P'][I,1])

      # --- Input connections ---
      elif button == 3:
          STR_connections_plot.set_color('blue')
          GP_connections_plot.set_color('blue')

          if event.artist == STR_plot:
              x,y = STR[index]['P']
              STR_selection_plot.set_data([x],[y])

              I = W_STR_STR[index,:].nonzero()
              STR_connections_plot.set_data(STR['P'][I,0], STR['P'][I,1])

              I = W_STR_GP[index].nonzero()
              GP_connections_plot.set_data(GP['P'][I,0], GP['P'][I,1])

          elif event.artist == GP_plot:
              x,y = GP[index]['P']
              GP_selection_plot.set_data([x],[y])

              I = W_GP_GP[index,:].nonzero()
              GP_connections_plot.set_data(GP['P'][I,0], GP['P'][I,1])

              I = W_GP_STR[index,:].nonzero()
              STR_connections_plot.set_data(STR['P'][I,0], STR['P'][I,1])

      plt.draw()
  
  # Figure
  fig = plt.figure(figsize=(16,7), facecolor='white')
  fig.canvas.mpl_connect('pick_event', on_pick)

  # Striatum plot
  STR_ax = plt.subplot(121, aspect=1)
  STR_ax.set_title("Striatum")
  STR_plot, = STR_ax.plot(STR['P'][:,0], STR['P'][:,1], 'o', color='k', alpha=0.1, picker=5)
  STR_ax.set_xlim(0,1)
  STR_ax.set_xticks([])
  STR_ax.set_ylim(0,1)
  STR_ax.set_yticks([])
  STR_selection_plot,   = STR_ax.plot([],[], 'o', color='black', alpha=1.0, zorder=10)
  STR_connections_plot, = STR_ax.plot([],[], 'o', color='red',   alpha=0.5, zorder=10)


  # GP plot
  GP_ax = plt.subplot(122, aspect=1)
  GP_ax.set_title("Globus Pallidus")
  GP_plot, = GP_ax.plot(GP['P'][:,0], GP['P'][:,1], 'o', color='k', alpha=0.1, picker=5)
  GP_ax.set_xlim(0,1)
  GP_ax.set_xticks([])
  GP_ax.set_ylim(0,1)
  GP_ax.set_yticks([])
  GP_selection_plot,   = GP_ax.plot([],[], 'o', color='black', alpha=1.0, zorder=10)
  GP_connections_plot, = GP_ax.plot([],[], 'o', color='red',   alpha=0.5, zorder=10)

  plt.show()
  
connections()

frequence = 1
if frequence : 
  
  ''' Histogram 2D of record activity '''

  x = STR["P"][:,0]
  y = STR["P"][:,1]

  pause = False
  step = 1
  #step = min(int(duration / dt + 1), step) # step in ms
  times = 0

  bins = 18

  hist_cumulate, xa, ya = np.histogram2d(x,y, bins = bins, weights = STR_record_U[0])  
  hist_counts_neurons, xb, yb = np.histogram2d(x,y, bins = bins)
  hist_counts_neurons = np.maximum(hist_counts_neurons, 1)

  mean_activity = hist_cumulate / hist_counts_neurons

  def onClick(event):

      ''' Capture a click to turn the histogram paused '''
      global pause
      pause ^= True

  def updatefig(i) : 

    ''' Updated of potential activity'''
    
    global pause, times
      
    if not pause and times < len(STR_record_U) :

      hist_cumulate, xa, ya = np.histogram2d(x,y, bins = bins, weights = STR_record_U[times])  
      mean_activity = hist_cumulate / hist_counts_neurons
    
      plt.title("Mean of frequency networks = " + str("{0:.2f}".format(np.mean(STR_record_U[times]))) 
            + "\n Dispersion of frequency networks = " + str("{0:.2f}".format(np.std(STR_record_U[times]))) 
            + "\n Time = " + str(times) + " ms")  
          
      im.set_array(mean_activity)  
      times += step # acceleration of the visualization
        
    return im

  fig = plt.figure(figsize=(12, 8))  
  im = plt.imshow(mean_activity, interpolation='nearest', origin='low', extent=[0, 1, 0, 1], vmin = fmin, vmax = fmax) 
  # vmin = Vmin, vmax = Vmax : fix values potential V, cmap = 'hot'
  
  plt.xlabel('x')
  plt.ylabel('y')
  cbar = plt.colorbar()
  cbar.ax.set_ylabel('Frequency in HZ')

  fig.canvas.mpl_connect('button_press_event', onClick)
  ani = animation.FuncAnimation(fig, updatefig)

  plt.show()

  fig = plt.figure(figsize=(14, 9))    
  plt.subplot(222)
  H = plt.hist(STR_record_U[-1], color='.5', edgecolor='w')
  plt.title('Distribution of frequency at the end')
  #plt.xlabel('Frequency HZ')
  plt.ylabel('Number of Neurons')

  plt.subplot(221)
  H = plt.hist(STR_record_U[0], color='.5', edgecolor='w')
  plt.title('Distribution of frequency at start')
  plt.ylabel('Number of Neurons')
  
  plt.subplot(223)
  number_neuron = 0
  #M = np.mean(STR_record_U[:, 0])
  title = "STR Neuron " + str(number_neuron) + ", I_int " #+ str("{0:.2f}".format(M))
  time_step = np.arange(0, len(STR_record_I_int))
  #mean_step = np.zeros(len(STR_record_I_int)) + M  
  plt.plot(time_step, STR_record_I_int[:, number_neuron], c='b', label= title)
  #plt.plot(time_step, mean_step, c='r', label= 'Mean')
  plt.title(title)
  plt.xlabel("Time (mseconds)")
  plt.ylabel("Intensity (mV)")
  plt.xlim(0, len(STR_record_U) - 1) 
  #plt.xlim(0, len(STR_record_U) - 1)
  
  plt.subplot(224)
  number_neuron = 0
  M = np.mean(STR_record_U[:, 0])
  title = "STR Neuron " + str(number_neuron) + ", Mean Frequency = " + str("{0:.2f}".format(M))
  time_step = np.arange(0, len(STR_record_U))
  mean_step = np.zeros(len(STR_record_U)) + M  
  plt.plot(time_step, STR_record_U[:, number_neuron], c='b', label= title)
  plt.plot(time_step, mean_step, c='r', label= 'Mean')
  plt.title(title)
  plt.xlabel("Time (mseconds)")
  plt.ylabel("Frequency (HZ)") 
  plt.xlim(0, len(STR_record_U) - 1)

  plt.show()
  
