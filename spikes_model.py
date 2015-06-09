#! /usr/bin/env python
# -*- coding: utf-8 -*-
import sys
import numpy as np
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
from math import sqrt
import matplotlib.animation as animation
from brian import *


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
D_STR_STR = cdist(STR["P"], STR["P"]) # compute distance between all neurons in STR
W = np.abs(np.random.normal(0.0, 0.1,(len(STR),len(STR)))) # Weight = normal law ==> center = 0, dispersion = 0.1
W_STR_STR = 1 * (W > D_STR_STR) # Connections if weight > distance, here : most neurons with distance < dispersion(=0.1) will be connected; Connect = 1 ;no connect = 0
np.fill_diagonal(W_STR_STR, 0) # neurons cannot connect themselves
STR_ge, STR_gi = 1000, 0 # number of neurons excitatory and inhibitory

W_STR_STR_ge = W_STR_STR[0 : STR_ge, :]
W_STR_STR_gi = W_STR_STR[STR_ge : len(STR), :] * -1

delay_propagation = 5 * ms

STR_delay_ge = ((D_STR_STR[0 : STR_ge, :] * delay_propagation) / sqrt(2)) #+ delay_synaptic # delay transmission between 0.5 ms and 5 ms
STR_delay_gi = ((D_STR_STR[STR_ge : len(STR), :] * delay_propagation) / sqrt(2)) #+ delay_synaptic

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

# -----------------------------------------------  Model --------------------------------------------------------------------- #

duration = 200 * ms # Default trial duration

x0, x1 = 0, int(duration / ms) # Born of displays
dt = 0.1 * ms  # Default Time resolution
step_time = Clock(dt = dt)

STR_Rest = -65.0 * mV # Resting Potential
STR_N = 0.01 * mV # Noise
STR_tau = 1 * ms # Potential time constants
STR_tau_ge, STR_tau_gi = 0.01 * ms, 0.01 * ms # Synaptics Time constants
STR_TH = -30.0 * mV # Spike threshold
STR_R = -70.0 * mV # Reset value after spikes

k = 1e-4 * volt * second ** (.5) # Noise

eqs = '''dV/dt = (-V + k * xi + ge + gi + Input + STR_Rest)/ STR_tau : volt
         dge/dt = -ge / STR_tau_ge : volt
         dgi/dt = -gi / STR_tau_gi : volt
         Input : volt '''

G_STR = NeuronGroup(len(STR), eqs, threshold = STR_TH, reset = STR_R, clock = step_time, refractory = 5 * ms)

# Init values of differents variables
G_STR.V = -65.0 * mV + rand(len(STR)) * STR_N
G_STR.ge, G_STR.gi, G_STR.Input = 0.0 * mV, 0.0 * mV, 0.0 * mV

# Network connections with weights
G_STR_ge = G_STR[0 : STR_ge]
G_STR_gi = G_STR[STR_ge : len(STR)]

Ce = DelayConnection(G_STR_ge, G_STR, 'ge')
Ci = DelayConnection(G_STR_gi, G_STR, 'gi')

Ce.set_delays(G_STR_ge, G_STR, STR_delay_ge * second)
Ci.set_delays(G_STR_gi, G_STR, STR_delay_gi * second)

Ce.connect(G_STR, G_STR, W_STR_STR_ge)
Ci.connect(G_STR, G_STR, W_STR_STR_gi)

# Save data
M = SpikeMonitor(G_STR)
MV = StateMonitor(G_STR, 'V', record = True) # if not True ==> Histogram doesn't work
Mge = StateMonitor(G_STR, 'ge', record = 0)
Mgi = StateMonitor(G_STR, 'gi', record = 0)
MInput = StateMonitor(G_STR, 'Input', record = 0)

# Electrode Stimulation
time_start = 0.0 * ms
time_end = 35.0 * ms # warning : bug with 30.0 * ms ...
input_stim = 40.0 * mV
pos_stim = [0.5, 0.5] # electrode position
tau_stim = 0.5 # exponential decrease of voltage electrode

def input_current(voltage, position, tau) :

  ''' Add current in dv/dt equation '''

  Stim = np.array([(voltage, position)], dtype=[('V', '<f8'), ('P', '<f8', 2)]) 
  Distance_Stim_Neurons = cdist(Stim['P'], STR['P']) # Compute distance between electrode and neurons in STR
    
  #Stim_Voltage = (Distance_Stim_Neurons < 0.2) * voltage 
  Stim_Voltage = np.exp(-Distance_Stim_Neurons / tau) * voltage
    
  return Stim_Voltage[0]

input_dist = input_current(input_stim, pos_stim, tau_stim)

@network_operation(Clock = EventClock(t = time_start, dt = dt)) # use EventClock to avoid clock ambiguous
def update(time) : 

  if time.t == time_start : 
    G_STR.Input = input_dist
  
  if time.t == time_end :
    G_STR.Input = 0.0 * mV 
  
run(duration) # Run simulation

# -----------------------------------------------  Displays --------------------------------------------------------------------- #

# Save spikes every time dt (1 = spike & 0 = none)
record_time_spikes = zeros((int(duration / dt), len(STR)))
for i in range(len(M.spikes)) :
  
  neuron_number = M.spikes[i][0]
  time_spikes = int((M.spikes[i][1] / ms) * 10)
  record_time_spikes[time_spikes][neuron_number] += 1  
  
time_step = 1 * ms # histogram time step
bins = 32 # shape histogram = 13 X 13

frames = int(time_step / dt)
nb_save = int((duration / dt) / frames)

# compute spikes numbers (with time_step) and create associate list of histogram
list_histogram = []
inc = 0

for i in range(0, nb_save) :
  
  sum_hist, xa, ya = histogram2d(STR["P"][:,0], STR["P"][:,1], bins = bins, weights = np.zeros((1000,))) 
  for j in range(inc, inc + frames) : 
    
    histogram, xa, ya = histogram2d(STR["P"][:,0], STR["P"][:,1], bins = bins, weights = record_time_spikes[j ,:])
    sum_hist += histogram
    
  list_histogram.append(sum_hist)  
  inc += frames   
  
# compute maximal number of spikes (all bins and times)
v_max = 0
for i in range(len(list_histogram)) :
   
  temp_max = np.max(list_histogram[i])    
  if temp_max > v_max : v_max = temp_max


# if 1 : print display
connections = 1
graphics = 1
histogram = 1
delay = 0

if connections :
  
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


if histogram :
  
  ''' Animation : Histogram of spikes activity '''

  pause = False

  fig = figure(figsize = (10,7))

  im = plt.imshow(list_histogram[0], interpolation='nearest', origin='low', extent=[0, 1, 0, 1], cmap = 'hot', vmin = 0, vmax = v_max) 

  xlabel('x')
  ylabel('y')

  title('Histogram of spikes activity')
  cbar = plt.colorbar()
  cbar.ax.set_ylabel('Number of Spikes')

  run = 1
  time_step = time_step / ms
  times = time_step
  
  def onClick(event):
      
      ''' Capture a click to turn the histogram in pause '''
      global pause
      if pause : pause = False
      else : pause = True
      

  def updatefig(i) : 

    ''' "Updated of spikes activity" '''

    global run, title, times, time_step, pause
    if  run < len(list_histogram) and pause == False :
        
      times += time_step
      title('Histogram of spikes activity' + "\n" + str(times) + " ms")
      im.set_array(list_histogram[run])
      run += 1
          
    return im

  fig.canvas.mpl_connect('button_press_event', onClick)
  ani = animation.FuncAnimation(fig, updatefig)

  plt.show()
  
if graphics :
  
  ''' Additional graphics '''

  figure(figsize = (14,9))
  subplot(221)
  raster_plot(M, title= 'Number of spikes for each neurons in Striatum')
  xlim(0, duration / ms)
  ylim(0, len(STR))

  subplot(222)
  plot(MV.times / ms, MV[0] / mV)
  title('Membrane potential of neurons 0')
  xlabel('Time (ms)')
  ylabel('V (mV)')
  xlim(x0, x1)

  subplot(223)
  plot(Mge.times / ms, Mge[0] / mV, label = 'ge')
  plot(Mgi.times / ms, Mgi[0] / mV, 'r', label = 'gi')
  legend(loc=4)
  title('Synaptics current in neurons 0')
  xlabel('Time (ms)')
  ylabel('V (mV)')
  xlim(x0, x1)

  subplot(224)
  plot(MInput.times / ms, MInput[0] / mV)
  title('Input current in neurons 0')
  xlabel('Time (ms)')
  ylabel('V (mV)')
  xlim(x0, x1)
  
  plt.show()

if delay :
  
  ''' Print delay repartition depending on the distance '''
  figure(figsize = (14,9))
  
  scatter(D_STR_STR[0 : STR_ge, :], STR_delay_ge / ms)
  xlabel('Distance (no unit)')
  ylabel('Delay (ms)')
  xlim(0, np.max(D_STR_STR[0 : STR_ge, :]))
  ylim(0, 5.0)
  
  show()  
  






