from __future__ import division
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import corner
import pdb


def plotmaker(outdir,post,params, npar, inj):


    if int(inj['doInj']):
        #truevals = []
        #key = inj.keys()

        #for ii in range(0, len(key)):

        #    if key[ii] != "doInj":
        #        truevals.append(inj[key[ii]])
        truevals = np.array([inj['alpha'], inj['ln_omega0'], inj['log_Np'], inj['log_Na']])


    if outdir[-1] != '/':
        outdir = outdir + '/'
    fig = corner.corner(post, labels=params, quantiles=(0.16, 0.84),
                        smooth=None, smooth1d=None, show_titles=True,
                        title_kwargs={"fontsize": 12},label_kwargs={"fontsize": 14},
                        fill_contours=True, use_math_text=True, )


    # Put correct values
    # Extract the axes
    axes = np.array(fig.axes).reshape((npar, npar))

    for ii in range(npar):
        ax = axes[ii, ii]
        if int(inj['doInj']):
            ax.axvline(truevals[ii], color="g", label='true value')
    plt.savefig(outdir + 'corners.png', dpi=150)
    print "Posteriors plots printed in " + outdir + "corners.png"