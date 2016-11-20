import matplotlib
#import matplotlib_init
import numpy as np
import os

def load_style(style):
    styles_dir = os.path.abspath(os.path.dirname(__file__))

    if np.isscalar(style):
        style = [style]

    for s in style:
        # !!! specifying multiple styles does not work (as matplotlib_init.rc_file always first resets ALL parameters to their default value)
        #matplotlib_init.rc_file('%s/%s.rc' % (styles_dir, s))
        matplotlib.rc_file('%s/%s.rc' % (styles_dir, s))
        matplotlib.rcParams.update(matplotlib.rcParams)
