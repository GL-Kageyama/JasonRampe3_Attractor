#======================================================================================
#------------------------    Jason Rampe 3 Attractor    -------------------------------
#======================================================================================

#------------------     new_X = sin(Y * b) + c * cos(X * b)     -----------------------
#------------------     new_Y = cos(X * a) + d * sin(Y * a)     -----------------------

#======================================================================================

import numpy as np
import pandas as pd
import panel as pn
import datashader as ds
from numba import jit
from datashader import transfer_functions as tf
from colorcet import palette_n

#--------------------------------------------------------------------------------------

ps = {k:p[::-1] for k, p in palette_n.items()}

pn.extension()

#--------------------------------------------------------------------------------------

@jit(nopython=True)
def JasonRampe3_trajectory(a, b, c, d, x0, y0, n):
    x, y = np.zeros(n), np.zeros(n)
    x[0], y[0] = x0, y0
    
    for i in np.arange(n-1):
        x[i+1] = np.sin(y[i]*b) + c*np.cos(x[i]*b)
        y[i+1] = np.cos(x[i]*a) + d*np.sin(y[i]*a)
        
    return x, y

#--------------------------------------------------------------------------------------

def JasonRampe3_plot(a=-2.76, b=-1.82, c=2.85, d=-0.87, n=1000000, colormap=ps['bmy']):
    
    cvs = ds.Canvas(plot_width=900, plot_height=500)
    x, y = JasonRampe3_trajectory(a, b, c, d, 0, 0, n)
    agg = cvs.points(pd.DataFrame({'x':x, 'y':y}), 'x', 'y')
    
    return tf.shade(agg, cmap=colormap)

#--------------------------------------------------------------------------------------

pn.interact(JasonRampe3_plot, n=(1,10000000))

#--------------------------------------------------------------------------------------

# The value of this attractor can be changed freely.
# Try it in the jupyter notebook.
