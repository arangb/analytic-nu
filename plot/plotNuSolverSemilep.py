import ROOT as r
import numpy as np
import pylab as plt
from ellipsetools import Ellipse as ET
# Trick to include the parent directory, so that we can import nuSolutions
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import nuSolutions as nuS


plt.rc("font", family="sans", size=12)

def plot_NuSolverLJ(bjet,lep,metx,mety,met_covxx,met_covyy,met_covxy,fname='NuSolver_ellipse.png'):
    # Check if bjet and lep have been created as r.Math.LorentzVector instances:
    if not (hasattr(bjet, 'Px') and hasattr(lep, 'Px')):
        print("Error: bjet and lep must be instances of ROOT.Math.LorentzVector")
        return

    met = [metx, mety]
    Sigm2 = np.matrix([[ met_covxx, met_covxy],[met_covxy, met_covyy]])

    mysol=nuS.singleNeutrinoSolution(bjet,mu,met[0],met[1],Sigm2)
    Dnu=np.sqrt(mysol.chi2[0,0])
    print('Dnu = ',  Dnu)
    #print('X matrix: ', mysol.X)
    #print('Hperp matrix: ',mysol.H_perp)
    #print('N matrix: ',mysol.N) #
    print('nu momentum = ',mysol.nu) # neutrino momentum

    e=ET(np.matrix(mysol.N))

    fig = plt.figure()
    ax = plt.gca()

    plt.plot(met[0],met[1],'kx')
    e.plot_ellipse(ax,color='blue')
    plt.arrow(0,0, mysol.nu[0], mysol.nu[1], ls=(5, (3,3)), head_width=7,  ec='black',fc='grey', label=r'$\nu$ soln', )
    plt.arrow(0,0, bjet.Px(), bjet.Py(), color='grey',  head_width=7,)
    plt.arrow(0,0, mu.Px(), mu.Py(), color='black', head_width=7,)

    #ax.set(xlim=[-200, 200], ylim=[-200, 200]) # center ax ranges on ellipse
    x0, y0 = e.center_of_ellipse
    a, b = e.axes_semimaj_semimin
    ax.set(xlim=[x0-1.4*2*a,x0+1.4*2*a],
           ylim=[y0-1.4*2*b,y0+1.4*2*b]) # center ax ranges on ellipse

    dnutext=r'$D_{{\nu}}$ = {0:.2f} GeV'.format(Dnu)
    mettext=r'p$^{{miss}}_{{x,y}}$ = {0:.0f}, {1:.0f} GeV'.format(met[0],met[1])
    nutext =r'$\nu$ soln'
    btext=  r'b jet $p_T$'
    mutext= r'lepton $p_T$'

    plt.legend( (mettext,dnutext, nutext,btext,mutext) , loc='upper left', frameon=False)
    plt.grid()
    plt.xlabel(r'p$^{{miss}}_{{x}}$ [GeV]')
    plt.ylabel(r'p$^{{miss}}_{{y}}$ [GeV]')
    plt.tight_layout()
    plt.savefig(fname)
    print("Plot saved in ", fname)

'''
#Can also check Otto's compiled C++. They match with the python version, but apparenty can run faster in columnar analysis
See: https://gitlab.cern.ch/hohare/ttbareft
import compiled.pynusolver as nusolver # to test the compiled version in cpp.
import awkward as ak
print('AGB',mu.px(),mu.py(),bjet.pz(),bjet.E())

lep_inputs=np.stack((ak.to_numpy(mu.px()),
                     ak.to_numpy(mu.py()),
                     ak.to_numpy(mu.pz()),
                     ak.to_numpy(mu.E())), axis=1).astype('float64') # one row has (px, py, pyz, E)
jets_inputs = np.stack((
                        ak.to_numpy(bjet.px()),
                        ak.to_numpy(bjet.py()),
                        ak.to_numpy(bjet.pz()),
                        ak.to_numpy(bjet.E()),
                        ak.to_numpy(False)),
                        axis=1).astype('float64') # one row has (px, py, pyz, E)

met_inputs = np.stack((
                        ak.to_numpy(met[0]),
                        ak.to_numpy(met[1]),
                        ak.to_numpy(sigma2[0,0]),
                        ak.to_numpy(sigma2[1,1]),
                        ak.to_numpy(sigma2[0,1])), axis=1).astype('float64') # one row has (px, py, covXX, covY
nu_array=np.zeros(4)
nusolver.run_nu_solver(lep_inputs,jets_inputs,met_inputs,nu_array)
print("Compiled",nu_array)
chi2=np.sqrt(nu_array[3]) # run_nu_solver returns (chi2)^2 for some reason
print(chi2)
'''

import csv
from itertools import islice

def get_event_from_csv_nanoaod(csvfname='data.csv',n=1):
    '''
    The csv file contains one row per event, and the columns have these headers:
    MET_pt,MET_phi,MET_covXX,MET_covYY,MET_covXY,Muon_pt,Muon_eta,Muon_phi,Jet_pt,Jet_eta,Jet_phi,Jet_mass,Jet_btagDeepB
    '''
    with open(csvfname, mode='r', newline='') as f:
        reader = csv.reader(f)
        # islice(iterator, start, stop) gets the nth item (0-indexed)
        nth_row = next(islice(reader, n, n + 1), None)
        print(nth_row)
        MET_pt,MET_phi,MET_covXX,MET_covYY,MET_covXY,Muon_pt,Muon_eta,Muon_phi,Jet_pt,Jet_eta,Jet_phi,Jet_mass,Jet_btagDeepB = map(float, nth_row)

    mu=r.Math.PtEtaPhiMVector(Muon_pt,Muon_eta,Muon_phi,0.105658)
    bjet=r.Math.PtEtaPhiMVector(Jet_pt,Jet_eta,Jet_phi,Jet_mass)
    return bjet,mu,MET_pt*np.cos(MET_phi),MET_pt*np.sin(MET_phi),MET_covXX,MET_covYY,MET_covXY

if __name__ == "__main__":
    # Better to use Math.LorentzVector and not TLorentzVector (which is deprecated)
    # You may also define you own LorentzVector class, and avoid having to import ROOT
    mu=r.Math.PtEtaPhiMVector(45,-1.22,0.54,0.105658)
    bjet=r.Math.PtEtaPhiMVector(65.8,-0.2434,-0.93,13.0078)
    metpt,metphi,met_covxx,met_covyy,met_covxy=103.87653,1.4360352,622.0,526.0,-14.3125
    plot_NuSolverLJ(bjet,mu,metpt*np.cos(metphi),metpt*np.sin(metphi),met_covxx,met_covyy,met_covxy,fname='ellipse23.png')

    # now read one row from nanoaod file:
    # (the * tells python to take that tuple and treat every item inside it as a separate input):
    plot_NuSolverLJ(*get_event_from_csv_nanoaod('data/ttljets_nano_aod.csv',10),'ellipse_10.png')
