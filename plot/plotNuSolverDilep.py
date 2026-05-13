import ROOT as r
import numpy as np
import pylab as plt
from ellipsetools import Ellipse as ET
# Trick to include the parent directory, so that we can import nuSolutions
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import nuSolutions as nuS

plt.rc("font", family="sans", size=14)

b  =r.TLorentzVector()
b_ =r.TLorentzVector()
mu =r.TLorentzVector()
mu_=r.TLorentzVector()

b.SetPtEtaPhiM(52.21875, 0.8309326171875, -2.24462890625 ,8.6796875)
b_.SetPtEtaPhiM(31.953125, 1.043701171875 ,-2.9091796875 ,5.44921875 ) 
mu.SetPtEtaPhiM(89.28206634521484, 0.35272216796875, -1.107666015625, 0.105712890625)
mu_.SetPtEtaPhiM(70.50473022460938 ,0.548095703125, 1.710693359375, 0.105712890625  )
met= [52.236774776655956, -35.27841514367014]

print("SOLVING")
mysol=nuS.doubleNeutrinoSolutions(b,b_, mu, mu_, met[0], met[1])
if mysol.N is not None:
    sols = mysol.nunu_s
#    print('this is', len(sols), sols)
    for i, v in enumerate(sols):
        print('solution pair',i, ":", sols[i])

    eN =ET(np.matrix(mysol.N))
    eN_=ET(np.matrix(mysol.N_))
    en_=ET(np.matrix(mysol.n_))

    fig = plt.figure()
    ax = plt.gca()
    eN.plot_ellipse(ax, edgecolor='k')
    eN_.plot_ellipse(ax, edgecolor='grey')
    en_.plot_ellipse(ax, edgecolor='black', ls='--')

    plt.plot(met[0], met[1],'kx')
    # [which intersection][px or py]
    for i, sol in enumerate(sols):
        if i==0: shape='o'
        elif i==1: shape='s'
        elif i==2: shape='^'
        else: shape='D'
            
        plt.plot(sol[0][0], sol[0][1], 'k'+shape, label='_nolegend_')
        plt.plot(sol[1][0], sol[1][1], shape, color='grey',label='_nolegend_')

    mettext  =r'p$_{{x,y}}^{{miss}}$ = {0:.0f}, {1:.0f} GeV'.format(met[0],met[1])
    nutext  ='neutrino'
    context ='MET constraint'

    ax.set(xlim=[-400, 400], ylim=[-475, 375]) # center ax ranges on ellipse

    plt.legend((mettext, 'neutrino', 'antineutrino','MET contstraint' ) , loc='center left', frameon=False)
    plt.grid()
    plt.xlabel(r'p$_{{x}}^{{miss}}$ [GeV]')
    plt.ylabel(r'p$_{{y}}^{{miss}}$ [GeV]')
    plt.tight_layout()
    plt.savefig('ellipses31.png')
    
else:
    print('failed')
