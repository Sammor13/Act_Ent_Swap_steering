#!/usr/bin/python3.6
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 28 14:40:09 2021

@author: Samuel Morales
TODO: global cost function for density matrix; J1=J2, alpha=0 case, klist saving and plot
"""

import matplotlib as mpl  
import numpy as np
import qutip as qt
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.ticker import (MultipleLocator)

import scipy.special
from sympy import LeviCivita
from scipy import stats

mpl.font_manager._rebuild()

mpl.rcParams['text.usetex'] = True
mpl.rcParams['font.family'] = 'serif'

mpl.rcParams["font.serif"] = "STIX"
mpl.rcParams["mathtext.fontset"] = "stix"

plt.style.use('seaborn-dark-palette')

##Pauli matrices
s0 = qt.qeye(2)
sx = qt.sigmax()
sy = qt.sigmay()
sz = qt.sigmaz()

s=[s0,sx,sy,sz]

def main():
    ##Parameters
    Nqb = 2
    DeltaT = 0.2                                        ##time step length, default: 0.000001
    J = [1, 0.99, 1.01]                                 ##coupling strength, default: 100000
    N = 200                                             ##nr of time steps
    M = 100                                             ##number of trajectories
    Nst = 1                                             ##every Nst´th step is saved
    #plist = [0.9, 0.1]#, 0.01]#, 0.001, 0.0001, 0.00001]##cost fct probabilities
    plist =[9*10**(-i) for i in range(1,Nqb)]
    #plist.append(10**(-Nqb+1))
    plist.append(1-sum(plist))
    K = 9                                              ##nr of couplings
    Fstar = 0.99                                       ##stop fidelity
    epsilon = 1-Fstar**2                                ##stop threshold for cost function
    params = [N, Nst, DeltaT, J[:Nqb], epsilon, K]
    
    ##Target state
    #ph = np.exp(1j*2*np.pi*rng.random())
    #psiTarg = qt.qstate(Nqb*'d')+ph*qt.qstate(Nqb*'u')
    #psiTarg = qt.qstate(Nqb*'u')
    #psiTarg = qt.w_state(Nqb)       #W state
    psiTarg = qt.ghz_state(Nqb)     #GHZ state
    psiTarg = psiTarg.unit()
    
    ##Initial state
    psi0 = qt.qstate(Nqb*'d')
    psi0 = psi0.unit()
    #psi0 = qt.qeye([2]*Nqb).unit()                                 #maximally mixed
    #psi0 = qt.rand_ket(2**Nqb, dims=[[2]*Nqb,[1]*Nqb])             ##random state
    #psi0 = qt.rand_dm(2**Nqb, dims=[[2]*Nqb,[2]*Nqb])              ##random density matrix
    
    print('psiTarg={0}, psi0={1}'.format(psiTarg, psi0))
    #print(r'J={0}, deltaT={1}, N={2}, Nst={3}, M={4}, pList={5}, $\varepsilon$={6}, K={7}'.format(J, DeltaT, N, Nst, M, plist, epsilon, K))    
    print(r'[N, Nst, DeltaT, J, epsilon, K]={0}, M={1}, pList={2}'.format(params, M, plist))    
    
    ##parallel running M trajectories
    results = qt.parallel_map(trajec, [Nqb]*M, task_kwargs=dict(psi0=psi0, psiTarg=psiTarg, N=N, DeltaT=DeltaT, J=J, pList=plist, eps=epsilon, K=K, Nst=Nst), progress_bar=True)
    #results = qt.parallel_map(trajec, [Nqb]*M, task_kwargs=dict(psi0=psi0, psiTarg=psiTarg, param=params, pList=plist), progress_bar=True)
    result_array = np.array(results, dtype=object)
    #klist, xiList, Plist, FidList, costFlist, psiList, successStep = result_array.T
    klist, xiList, FidList, costFlist, psiList, successStep = result_array.T
    
    ##rearrange Distributions
    FidDistr = np.reshape(np.concatenate(FidList), (M, int(N/Nst)+1))
    costFDistr = np.reshape(np.concatenate(costFlist), (M, int(N/Nst)+1))
    #kDistr = np.reshape(np.concatenate(klist), (M, N))
    psiDistr = np.reshape(np.concatenate(psiList), (M, int(N/Nst)+1))
    if Nqb ==2:
        SDistr = np.reshape([qt.entropy_vn(psi.ptrace((0))) for psi in np.concatenate(psiList)], (M, int(N/Nst)+1))
    #SADistr = np.reshape(np.concatenate(SAlist), (len(LambdaList), M, int(N/Nst+1)))
    #SBDistr = np.reshape(np.concatenate(SBlist), (len(LambdaList), M, int(N/Nst+1)))
    #SCDistr = np.reshape(np.concatenate(SClist), (len(LambdaList), M, int(N/Nst+1)))
    
############################################################Plots
   
    ##coupling histogram
    fig, ax = plt.subplots(figsize=(8,6))
#    plt.subplot(1, 2, 1)
   # for k, lamb in enumerate(LambdaList):
    #plt.hist(successStep[k*M:(k+1)*M], bins=[50*j for j in range(int(N/50))], alpha=0.5, label=r'$[p_1,p_2]$={0}'.format(beta))
    plt.hist(np.concatenate(klist), bins=np.arange(0,K**2), alpha=0.5, label=r'$[p_r]$={0}'.format(plist), edgecolor='grey')       #edgecolor=
    #plt.axvline(meanN, color='k', ls='--', linewidth=2, label=r'$\overline{{t}}$')
    #plt.annotate(text='', xy=(meanN+stdN,M/300), xytext=(meanN-stdN,M/300), arrowprops=dict(arrowstyle='|-|'))
#    plt.hist(successStep2, bins=np.arange(0,N2+1,25*Nst2), alpha=0.5, label=r'$[p_r]$={0}'.format(beta), edgecolor='grey')       #edgecolor=
#    plt.axvline(meanN2, color='darkred', ls='--', linewidth=2, label=r'$\overline{{t}}$')
#    plt.annotate(text='', xy=(meanN2+stdN2,M2/300), xytext=(meanN2-stdN2,M2/300), arrowprops=dict(arrowstyle='|-|', color='darkred'))
    
    plt.xlim(left=0, right=K**2)
    #plt.ylim(bottom=0.95)#, top=850)
    #plt.ylim(bottom=10)
    plt.minorticks_on()
    
    plt.xlabel(r'$K$', fontsize=25)
    plt.ylabel(r'Number of trajectories', fontsize=25)
    #plt.title('N={0}, $F_{{fin}}={1}$'.format(Nqb, Ffin))
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    #plt.yscale('log')
    
    #plt.locator_params(axis='x', nbins=4)
    plt.tight_layout()    
    plt.savefig('coupl hist.pdf')
    plt.savefig('coupl hist.svg')
    
    #print(kDistr)
    print(FidDistr)
    print(costFDistr)
    print(successStep)
    
    ##purity of average
    fig, ax = plt.subplots(figsize=(8,6))#, dpi=600)
    #plt.title(r'N={0}, $F^{{*}}={1}, |00\rangle\rightarrow|\Phi_{{0,+}}\rangle$'.format(Nqb, Fstar), fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.ylim(bottom=0, top=1.01)
    plt.xlim(left=0, right=N)
    if psi0.type == 'ket':
        avgPur = [(sum([qt.ket2dm(psi) for psi in psiDistr[:,i]])**2).tr()/M**2 for i in range(int(N/Nst)+1)]
    elif psi0.type =='oper':
        avgPur = [(sum([psi for psi in psiDistr[:,i]])**2).tr()/M**2 for i in range(int(N/Nst)+1)]
        #avgPur = [np.mean([(psi**2).tr() for psi in psiDistr[:,i]]) for i in range(int(N/Nst)+1)]
    ax.plot(np.arange(0,N+1,Nst), avgPur, 'k', label=r'average C', linewidth=3)
    ax.plot(np.arange(0,N+1,Nst), np.ones(int(N/Nst)+1)*(1/2**Nqb), 'k:', label=r'minimum', linewidth=2)
    #ax.plot(np.arange(0,N+1,Nst), np.ones(int(N/Nst)+1)*(psiTarg**2).tr(), 'r:', label=r'target cost', linewidth=2)
    if psi0.type =='oper':
        ax.plot(np.arange(0,N+1,Nst), [(psi**2).tr() for psi in psiDistr[7,:]], '--', label=r'single traj 1', linewidth=2, alpha=0.7)
  #      ax.plot(np.arange(0,N+1,Nst), [(psi**2).tr() for psi in psiDistr[25,:]], '--', label=r'single traj 2', linewidth=2, alpha=0.7)
  #      ax.plot(np.arange(0,N+1,Nst), [(psi**2).tr() for psi in psiDistr[33,:]], '--', label=r'single traj 3', linewidth=2, alpha=0.7)
    ax.set_xlabel(r'$n_t$',fontsize=25)
    ax.set_ylabel(r'$\mathrm{{tr}}(\overline{{\rho}}^2)$',fontsize=25)
    ax.minorticks_on()
    ax.tick_params(length=8)            ##width=
    ax.tick_params(which='minor', length=4)            ##width=
    ax.locator_params(axis='x', nbins=4)
    ax.locator_params(axis='y', nbins=6)
    ax.xaxis.set_minor_locator(MultipleLocator(N/20))
    
    plt.tight_layout()
    #plt.subplots_adjust(left=, bottom, right=, top=, wspace=, hspace=)
    
    plt.savefig('avg pur.pdf', format='pdf')
    plt.savefig('avg pur.svg', format='svg')
    
    ##Average fidelity plot
    fig, ax = plt.subplots(figsize=(8,6))#, dpi=600)
    #plt.title(r'N={0}, $F^{{*}}={1}, |00\rangle\rightarrow|\Phi_{{0,+}}\rangle$'.format(Nqb, Fstar), fontsize=20, loc='right')
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.ylim(bottom=0, top=1.01)
    plt.xlim(left=0, right=N)
    avgC = [np.mean(FidDistr[:,i]) for i in range(int(N/Nst)+1)]
    ax.plot(np.arange(0,N+1,Nst), avgC, 'k', label=r'average C', linewidth=3)
    ax.plot(np.arange(0,N+1,Nst), FidDistr[7,:], '--', label=r'single traj 1', linewidth=2, alpha=0.7)
  #  ax.plot(np.arange(0,N+1,Nst), FidDistr[25,:], '--', label=r'single traj 2', linewidth=2, alpha=0.7)
  #  ax.plot(np.arange(0,N+1,Nst), FidDistr[33,:], '--', label=r'single traj 3', linewidth=2, alpha=0.7)
    ax.plot(np.arange(0,N+1,Nst), np.ones(int(N/Nst)+1)*(1-Fstar**2), 'k:', label=r'target cost', linewidth=2)
    ax.set_xlabel(r'$n_t$',fontsize=25)
    ax.set_ylabel(r'$C_{0}(t)$'.format(Nqb),fontsize=25)
    ax.minorticks_on()
    ax.tick_params(length=8)            ##width=
    ax.tick_params(which='minor', length=4)            ##width=
    ax.locator_params(axis='x', nbins=4)
    ax.locator_params(axis='y', nbins=6)
    ax.xaxis.set_minor_locator(MultipleLocator(N/20))
    #ax.text(1300, 0.3, r'$|\mathrm{{W}}\rangle$', fontsize=25)
    #plt.rcParams['axes.titlepad'] = -120  
        
    #plt.locator_params(which='minor', nbins=4)
    #ax.yaxis.set_minor_locator(MultipleLocator(0.05))
    
    axins = inset_axes(ax, width='45%', height='45%', borderpad=0.5)
    plt.xlim(left=0, right=N)
    plt.ylim(bottom=0, top=0.3)
    axins.set_xlabel(r'$n_t$',fontsize=20)
    axins.xaxis.set_label_coords(0.5,-0.09)
    #axins.set_ylabel(r'$C(t)$',fontsize=20)
    axins.minorticks_on()
    axins.xaxis.set_minor_locator(MultipleLocator(N/20))
    axins.yaxis.set_minor_locator(MultipleLocator(0.01))
    axins.tick_params(length=4)            
    axins.tick_params(which='minor', length=3)
    axins.locator_params(axis='x', nbins=4)
    axins.locator_params(axis='y', nbins=3)
    #axins.text(50, 0.2, r'$p_1=0.9$', fontsize=20)
    
    avgCloc = [np.mean(costFDistr[:,i]) for i in range(int(N/Nst)+1)]
    axins.plot(np.arange(0,N+1,Nst), avgCloc, 'k', label=r'average', linewidth=2)
    axins.plot(np.arange(0,N+1,Nst), costFDistr[7,:], '--', label=r'single traj 1', linewidth=1.5, alpha=0.7)
  #  axins.plot(np.arange(0,N+1,Nst), costFDistr[25,:], '--', label=r'single traj 2', linewidth=1.5, alpha=0.7)
  #  axins.plot(np.arange(0,N+1,Nst), costFDistr[33,:], '--', label=r'single traj 3', linewidth=1.5, alpha=0.7)
    
    plt.tight_layout()
    #plt.subplots_adjust(left=, bottom, right=, top=, wspace=, hspace=)
    
    plt.savefig('avg fid.pdf', format='pdf')
    plt.savefig('avg fid.svg', format='svg')

    ##Success statistics plot
    print(np.count_nonzero(successStep==N+1))
    print(np.count_nonzero(successStep==N+2))
    successStep = np.delete(successStep, np.where(successStep==N+1))
    successStep = np.delete(successStep, np.where(successStep==N+2))
    
    if successStep.any():
        meanN = np.mean(successStep)
        stdN = np.std(successStep)
        modeN = stats.mode(successStep)
    else:
        meanN, stdN, modeN = 0, 0, 0
    #successStep2 = np.delete(successStep2, np.where(successStep==0))
    
    #meanN2 = np.mean(successStep2)
    #stdN2 = np.std(successStep2)
    #modeN2 = stats.mode(successStep2)
    
    print(r'$[p_1,p_2]$={0}: $N_{{suc}}=${1}'.format(plist, meanN))
    print(r'$[p_1,p_2]$={0}: $N^{{med}}_{{suc}}=${1}'.format(plist, np.median(successStep)))
    print(r'$[p_1,p_2]$={0}: $STDN^{{med}}_{{suc}}=${1}'.format(plist, stdN))
    print(r'$[p_1,p_2]$={0}: $ModeN^{{med}}_{{suc}}=${1}'.format(plist, modeN))
    
    #successStep2 = np.delete(successStep2, np.where(successStep2==0))
    #meanN2 = np.mean(successStep2)
    #print(r'$[p_1,p_2]$={0}: $N2_{{suc}}=${1}'.format(beta, meanN2))
    
    successStep -=1
    #successStep2 -=1
    
    fig, ax = plt.subplots(figsize=(8,6))
    plt.hist(successStep, bins=np.arange(0,N+1,2*Nst), alpha=0.5, label=r'$[p_r]$={0}'.format(plist), edgecolor='grey')
    plt.axvline(meanN, color='k', ls='--', linewidth=2, label=r'$\overline{{t}}$')
    plt.annotate(text='', xy=(meanN+stdN,M/300), xytext=(meanN-stdN,M/300), arrowprops=dict(arrowstyle='|-|'))
#    plt.hist(successStep2, bins=np.arange(0,N2+1,25*Nst2), alpha=0.5, label=r'$[p_r]$={0}'.format(beta), edgecolor='grey')
#    plt.axvline(meanN2, color='darkred', ls='--', linewidth=2, label=r'$\overline{{t}}$')
#    plt.annotate(text='', xy=(meanN2+stdN2,M2/300), xytext=(meanN2-stdN2,M2/300), arrowprops=dict(arrowstyle='|-|', color='darkred'))
    
    plt.xlim(left=0, right=N)
    #plt.ylim(bottom=0.95)
    plt.minorticks_on()
    
    plt.xlabel(r'$n_t$', fontsize=25)
    plt.ylabel(r'Number of trajectories', fontsize=25)
    #plt.title('N={0}, $F_{{fin}}={1}$'.format(Nqb, Ffin))
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.yscale('log')
    
    ax.tick_params(length=8)            
    ax.tick_params(which='minor', length=4)
    plt.locator_params(axis='x', nbins=4)
    ax.xaxis.set_minor_locator(MultipleLocator(N/20))
    
    #axins = inset_axes(ax, width='30%', height='30%', borderpad=0.2)
    #plt.hist(successStep2, bins=np.arange(0,N2+1,2*Nst2), alpha=0.5, label=r'$[p_r]$={0}'.format(beta), edgecolor='grey')       #edgecolor=
    #plt.axvline(meanN2, color='k', ls='--', linewidth=1.5, label=r'$\overline{{t}}$')
    #plt.annotate(text='', xy=(meanN2+stdN2,M2/300), xytext=(meanN2-stdN2,M2/300), arrowprops=dict(arrowstyle='|-|'))
    
    #plt.xlim(left=0, right=130)
    #plt.ylim(bottom=0.95)
    #axins.set_xlabel(r'$n_t$',fontsize=20)
    #axins.set_ylabel(r'$C(t)$',fontsize=20)
    #axins.minorticks_on()
    #axins.xaxis.set_minor_locator(MultipleLocator(5))
    #axins.yaxis.set_minor_locator(MultipleLocator(0.05))
    #axins.tick_params(length=4)            ##width=
    #axins.tick_params(which='minor', length=3)            ##width=
    #axins.locator_params(axis='x', nbins=4)
    #axins.locator_params(axis='y', nbins=3)
    #plt.yscale('log')
    
    #plt.legend(fontsize=12, loc=1)
    plt.tight_layout()    
    plt.savefig('success step.pdf')
    plt.savefig('success step.svg') 

    ##entanglement plots
    if Nqb==2:        
        ##Average entanglement entropy plot    
        fig, ax = plt.subplots(figsize=(8,6))
        avgS = [np.mean(SDistr[:,i]) for i in range(int(N/Nst)+1)]
        ax.plot(np.arange(0,N+1,Nst), avgS[:], 'k', linewidth=3, label=r'average')
        ax.plot(np.arange(0,N+1,Nst), SDistr[7,:], '--', label=r'single traj 1', linewidth=2, alpha=0.7)
     #   ax.plot(np.arange(0,N+1,Nst), SDistr[25,:], '--', label=r'single traj 2', linewidth=2, alpha=0.7)
     #   ax.plot(np.arange(0,N+1,Nst), SDistr[33,:], '--', label=r'single traj 3', linewidth=2, alpha=0.7)
        ax.plot(np.arange(0,N+1,Nst), np.ones(int(N/Nst)+1)*np.log(2), 'k:', label=r'maximally entangled')
        #ax.legend(fontsize=15, loc=4)
        ax.set_xlabel(r'$n_t$',fontsize=25)
        ax.set_ylabel('$S(t)$',fontsize=25)
        plt.xlim(0,70)
        plt.ylim(0,0.73)
        plt.minorticks_on()
        ax.tick_params(length=8)            ##width=
        ax.tick_params(which='minor', length=4)            ##width=
        ax.locator_params(axis='x', nbins=4)
        ax.locator_params(axis='y', nbins=4)
        ax.xaxis.set_minor_locator(MultipleLocator(5))
        ax.yaxis.set_minor_locator(MultipleLocator(0.05))
    
        #plt.title(r'N={0}, $F^{{*}}={1}, |00\rangle\rightarrow|\Phi_{{0,+}}\rangle$'.format(Nqb, Fstar), fontsize=20)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        
   #     axins = inset_axes(ax, width='40%', height='40%', borderpad=0.5, loc=4, bbox_to_anchor=(.06, .3, 0.94, 0.94), bbox_transform=ax.transAxes)
   #     avgS2 = [np.mean(SDistr2[:,i]) for i in range(int(N2/Nst2)+1)]
   #     plt.xlim(left=0, right=7000)
   #     plt.ylim(0,0.73)
   #     axins.set_xlabel(r'$n_t$',fontsize=20)
   #     axins.set_ylabel(r'$S(t)$',fontsize=20)
   #     axins.minorticks_on()
   #     axins.xaxis.set_minor_locator(MultipleLocator(500))
   #     axins.yaxis.set_minor_locator(MultipleLocator(0.05))
   #     axins.tick_params(length=4)            ##width=
   #     axins.tick_params(which='minor', length=3)            ##width=
   #     axins.locator_params(axis='x', nbins=4)
   #     axins.locator_params(axis='y', nbins=4)
        #plt.title(r'$|\mathrm{{EPR}}\rangle \to |00\rangle$')
   #     axins.text(2000, 0.18, r'$|\mathrm{{Bell}}\rangle \to |00\rangle$', fontsize=20)
        
        #plt.rcParams['axes.titlepad'] = 10
        
   #     axins.plot(np.arange(0,N2+1,Nst2), avgS2, 'k', label=r'average', linewidth=3)
   #     axins.plot(np.arange(0,N2+1,Nst2), SDistr2[1,:], '--', label=r'single traj 1', linewidth=2, alpha=0.7)
   #     axins.plot(np.arange(0,N2+1,Nst2), SDistr2[418,:], '--', label=r'single traj 2', linewidth=2, alpha=0.7)
   #     axins.plot(np.arange(0,N2+1,Nst2), SDistr2[264,:], '--', label=r'single traj 3', linewidth=2, alpha=0.7)
   #     axins.plot(np.arange(0,N2+1,Nst2), np.ones(int(N2/Nst2)+1)*np.log(2), 'k:', label=r'maximally entangled')
        
        plt.tight_layout()
        
        plt.savefig('avg EE.pdf')
        plt.savefig('avg EE.svg')
       
    
##Trajectory simulator
def trajec(Nqb, psi0, psiTarg, N, DeltaT, J, pList, eps, K, Nst):
#def trajec(Nqb, psi0, psiTarg, param, pList):
    successStep = N+1
    #N, Nst, DeltaT, J, eps, K = param
    Gamma = [j**2*DeltaT for j in J]        ##jump rate, default: J*J*DeltaT
    
    #random number generator
    rng = np.random.default_rng() 
    
    ##Measured quantities
    #deltaFid = np.zeros(K**2)
    klist = np.zeros(N*int(Nqb/2))
    FidList = np.zeros(int(N/Nst)+1)
    costF = np.zeros(int(N/Nst)+1)
    #SAlist = np.zeros(int(N/Nst)+1)
    #SBlist = np.zeros(int(N/Nst)+1)
    #SClist = np.zeros(int(N/Nst)+1)
    xiList = np.zeros(N)
    #PList = np.zeros(N)
    psiList = np.zeros(int(N/Nst+1), dtype=object)
    
    ##coupling list
    if K == 9:
        slist = [1,1,1,-1,-1,-1,1,1,1]
        aList = [1,2,3,1,2,3,1,2,3]
        bList = [3,3,3,3,3,3,1,1,1]
    elif K == 10:
        slist = [1,1,1,-1,-1,-1,1,1,1,1]
        aList = [1,2,3,1,2,3,0,1,2,3]
        bList = [3,3,3,3,3,3,1,1,1,1]
    elif K == 12:
        slist = [1,1,1,-1,-1,-1,1,1,1,1,1,1]
        aList = [1,2,3,1,2,3,1,2,3,1,2,3]
        bList = [3,3,3,3,3,3,1,1,1,2,2,2]
    elif K == 14:
        slist = [1,1,1,-1,-1,-1,1,1,1,1,1,1,1,1]
        aList = [1,2,3,1,2,3,0,1,2,3,0,1,2,3]
        bList = [3,3,3,3,3,3,1,1,1,1,2,2,2,2]
        
    elif K == 3:
        slist = [1,1,1]
        aList = [1,2,3]
        bList = [1,1,1]
    elif K == 6:
        slist = [1,1,1,1,1,1]
        aList = [1,2,3,1,2,3]
        bList = [1,1,1,2,2,2]
        ##incl a=0
    elif K == 4:
        slist = [1,1,1,1]
        aList = [0,1,2,3]
        bList = [1,1,1,1]
    elif K == 8:
        slist = [1,1,1,1,1,1,1,1]
        aList = [0,1,2,3,0,1,2,3]
        bList = [1,1,1,1,2,2,2,2]
        
    ##operator list
    pauliPlaqList = plaqSlist(Nqb)
    
    ##initial state
    psiList[0] = psi0
    psi = psi0
    Spsi = qt.expect(pauliPlaqList,psi0)
    Starg = qt.expect(pauliPlaqList,psiTarg)
    
    ##subset list
    subset = iniSubset(Nqb)
    
    ##Initial cost function values
    costF[0] = sum([pList[j]*costFunc(Spsi, Starg, j, Nqb, subset[j]) for j in range(Nqb-1)])
    FidList[0] = np.linalg.norm(Spsi-Starg)**2/2**(Nqb+1)
    Fid = FidList[0]
    costF[0] += pList[-1]*Fid
    
    #SAlist[0] = qt.entropy_vn(psi0.ptrace((0)))
    #SBlist[0] = qt.entropy_vn(psi0.ptrace((1)))
    #SClist[0] = qt.entropy_vn(psi0.ptrace((2)))
    
    finish=False
    
    ##Time steps
    for i in range(1, N+1):
        ##starting steering pair
        nStart1 = int(N*rng.random())
        nStart2 = (nStart1+1)%Nqb
        
        ##steering neighbouring pairs
        for nPair in range(int(Nqb/2)):
            n1 = (nStart1+2*nPair)%Nqb
            n2 = (nStart2+2*nPair)%Nqb
            
            ##ordering, necessary?
            if n1>n2:
                ntemp = n1
                n1=n2
                n2=ntemp
            
            ##Coupling strengths
            J1, J2 = J[n1], J[n2]
            G1, G2 = Gamma[n1], Gamma[n2]
            
            ##Stoppage criterion
            if finish:
                break
        
        ##move out of loop?
            if Fid < eps:
                FidList[int((i-1)/Nst)+1:] = Fid
                costF[int((i-1)/Nst)+1:] = sum([pList[j]*costFunc(Spsi, Starg, j, Nqb, subset[j]) for j in range(Nqb-1)])
                costF[int((i-1)/Nst)+1:] += pList[-1]*Fid
                #Slist[i:] = qt.entropy_vn(psiList[i-1].ptrace((0)))
                #SAlist[int((i-1)/Nst)+1:] = qt.entropy_vn(psi.ptrace((0)))
                #SBlist[int((i-1)/Nst)+1:] = qt.entropy_vn(psi.ptrace((1)))
                #SClist[int((i-1)/Nst)+1:] = qt.entropy_vn(psi.ptrace((2)))
        
                klist[(i-1)*int(Nqb/2)+nPair:] = -1#np.nan
                xiList[i-1:] = np.nan
                successStep = i
                psiList[int((i-1)/Nst)+1:] = (int(N/Nst)-int((i-1)/Nst))*[1]*psi
                
                finish = True
                break
            
            ##Active decision making
            deltaFid = expCostF(Spsi, Starg, J, Gamma, DeltaT, pList, n1, n2, Nqb, K)
        
            #klist[i-1], PList[i-1] = metrop(deltaFid, betaEff, K)
            klis = int(rng.choice(np.where(deltaFid==np.nanmin(deltaFid))[0]))
            klist[(i-1)*int(Nqb/2)+nPair] = klis
            #klis = rng.choice(np.where(deltaFid==np.nanmin(deltaFid))[0])
            #klist[(i-1)*int(Nqb/2)+nPair] = klis
            
            ##chosen couplings
            s1 = slist[int(klis%K)]
            s2 = slist[int(klis/K)]
            alpha1 = aList[int(klis%K)]
            alpha2 = aList[int(klis/K)]
            beta1 = bList[int(klis%K)]
            beta2 = bList[int(klis/K)]
            
            ##non-improvement criterion
            dcost = 0.001
            if np.nanmin(deltaFid) > dcost:
                successStep=N+2
                break
            
            ##chosen Pauli operators
            sig1 = pauliPlaqList[alpha1*4**n1]
            sig2 = pauliPlaqList[alpha2*4**n2]
            
            ##Time step
            #beta1=beta2=z
            if beta1==3 and beta2==3:
                H = s1*J1*sig1+s2*J2*sig2
                psi = schroesol(psi, DeltaT, H)
            
            #beta1=z, beta2!=z
            elif beta1==3:
                c_op, H = np.sqrt(G2)*sig2, s1*J1*sig1
                psi, xiList[i-1] = unitsol(psi, DeltaT, H, c_op, G2*DeltaT)
            
            #beta1!=z, beta2=z
            elif beta2==3:
                c_op, H = np.sqrt(G1)*sig1, s2*J2*sig2
                psi, xiList[i-1] = unitsol(psi, DeltaT, H, c_op, G1*DeltaT)
            
            #beta1=beta2=x/y  
            elif beta1==beta2:
                c_op = [np.sqrt(G1/2)*sig1+np.sqrt(G2/2)*sig2, np.sqrt(G1/2)*sig1-np.sqrt(G2/2)*sig2]
                psi, xiList[i-1] = montecsol(psi, DeltaT, c_op)
               # if beta1==1:
               #     psi, xiList[i-1] = montecsol(psi, DeltaT, c_op)
               # else:
               #     psi, xiList[i-1] = montecsol2(psi, DeltaT, c_op)
            
            #(beta1=x, beta2=y) or (beta1=y, beta2=x)
            else:
                sRan = (-1)**int(2*rng.random())
                if beta1==1:
                    c_op, H = np.sqrt(G1)*sig1+sRan*1j*np.sqrt(G2)*sig2, sRan*np.sqrt(G1*G2)*sig1*sig2
                elif beta1==2:
                    c_op, H = np.sqrt(G1)*sig1-sRan*1j*np.sqrt(G2)*sig2, sRan*np.sqrt(G1*G2)*sig1*sig2
                psi, xiList[i-1] = unitsol(psi, DeltaT, H, c_op, (G1+G2)*DeltaT)
      
    #######TODO: move this part out of loop?     
        Spsi = qt.expect(pauliPlaqList,psi)    
        Fid = np.linalg.norm(Spsi-Starg)**2/2**(Nqb+1)
        
        if finish:
            break
        
        ##saving data every Nst step
        if i%Nst == 0:
            psiList[int(i/Nst)] = psi
            FidList[int(i/Nst)] = Fid
            costF[int(i/Nst)] = sum([pList[j]*costFunc(Spsi, Starg, j, Nqb, subset[j]) for j in range(Nqb-1)])
            costF[int(i/Nst)] += pList[-1]*Fid
                #SAlist[int(i/Nst)] = qt.entropy_vn(psi.ptrace((0)))
                #SBlist[int(i/Nst)] = qt.entropy_vn(psi.ptrace((1)))
                #SClist[int(i/Nst)] = qt.entropy_vn(psi.ptrace((2)))
        #Slist[i] = qt.entropy_vn(psiList[i].ptrace((0)))
            
    #return klist, xiList, PList, FidList, costF, psiList, successStep
    return klist, xiList, FidList, costF, psiList, successStep
    #return klist, xiList, PList, FidList, psiList, successStep

##Time evolution solver    
#Monte-Carlo solver
def montecsol(psi, deltaT, c_op):
    #random number generator
    rng = np.random.default_rng() 
    
    P = np.zeros(4)
    P[0:2] = qt.expect([c.dag()*c for c in c_op], psi)*deltaT
    P[2:4] = (1-2*P[0:2])/2
    while 1==1:
        k = int(4*rng.random())
        if rng.random() < P[k]:
            break
    if k < 2:
        if psi.type == 'ket':
            return (c_op[k]*psi).unit(), k+2
        elif psi.type == 'oper':
            return (c_op[k]*psi*c_op[k].dag()).unit(), k+2
    else:
        if psi.type == 'ket':
            return ((1-deltaT*c_op[k-2].dag()*c_op[k-2])*psi).unit(), k-2
        elif psi.type == 'oper':
            return (psi-deltaT*(c_op[k-2].dag()*c_op[k-2]*psi+psi*c_op[k-2].dag()*c_op[k-2])).unit(), k-2
       
##Unitary dynamics
def unitsol(psi, deltaT, H, c_op, P):
    #random number generator
    rng = np.random.default_rng() 
    
    #P = qt.expect(c_op.dag()*c_op, psi)*deltaT
    if rng.random() >= P:
        #return qt.sesolve(H, psi, np.array([0,deltaT]), []).states[1], 0
    #    return schroesol(psi, deltaT, H), 0
        if psi.type == 'ket':
            return ((1-1j*deltaT*H-P/2)*psi).unit(), 0
        elif psi.type == 'oper':
            return ((1-P)*psi-1j*deltaT*(H*psi-psi*H)).unit(), 0
    else:
        if psi.type == 'ket':
            return (c_op*psi).unit(), 1
        elif psi.type == 'oper':
            return (c_op*psi*c_op.dag()).unit(), 1
        #return (c_op*psi).unit(), 1

def schroesol(psi, deltaT, H):
    if psi.type == 'ket':
        return ((1-1j*deltaT*H)*psi).unit()
    elif psi.type == 'oper':
        return (psi-1j*deltaT*(H*psi-psi*H)).unit()   


#Fidelity calculation
##RDM cost fct
def RdmFidCalc(s, S, r, sTarg, STarg, rTarg, deltaT, J, Gamma, p1, p2, qb):
    deltaF = np.zeros(81)
    
    if qb==0:
        #indS = [(slice(None),i) for i in range(3)]
        indR1 = [[(slice(None),i,j) for j in range(3)] for i in range(3)]
        indR2 = [[(i,slice(None),j) for j in range(3)] for i in range(3)]
        indR3 = [[(i,j,slice(None)) for j in range(3)] for i in range(3)]
        #indR1 = ind1
        #indR2 = ind2
        #indR3 = ind3
    elif qb==1:
        #indS = [(slice(None),i) for i in range(3)]
        indR3 = [[(slice(None),i,j) for j in range(3)] for i in range(3)]
        indR1 = [[(j,slice(None),i) for j in range(3)] for i in range(3)]
        indR2 = [[(j,i,slice(None)) for j in range(3)] for i in range(3)]
        
        #indR1 = ind2
        #indR2 = ind3
        #indR3 = ind1  
    elif qb==2:
        #indS = [(slice(None),i) for i in range(3)]
        indR2 = [[(slice(None),j,i) for j in range(3)] for i in range(3)]
        indR3 = [[(j,slice(None),i) for j in range(3)] for i in range(3)]
        indR1 = [[(i,j,slice(None)) for j in range(3)] for i in range(3)]
        
        #indR1 = ind3
        #indR2 = ind1
        #indR3 = ind2
    
    cross1 = (3+p1+p2)/12*np.cross(s[qb], sTarg[qb])+(3-p2-3*p1)/12*sum([np.cross(S[qb][:,i], STarg[qb][:,i])+np.cross(S[(qb+2)%3][i,:], STarg[(qb+2)%3][i,:]) for i in range(3)])+(
            (1-p2-p1)/4*sum([np.cross(r[indR1[i][j]], rTarg[indR1[i][j]]) for i in range(3) for j in range(3)]))
    
    cross2 = (3+p1+p2)/12*np.cross(s[(qb+1)%3], sTarg[(qb+1)%3])+(3-p2-3*p1)/12*sum([np.cross(S[qb][i,:], STarg[qb][i,:])+np.cross(S[(qb+1)%3][:,i], STarg[(qb+1)%3][:,i]) for i in range(3)])+(
            (1-p2-p1)/4*sum([np.cross(r[indR2[i][j]], rTarg[indR2[i][j]]) for i in range(3) for j in range(3)]))
    
    dot1 = (3+p1+p2)/12*(np.dot(s[qb], sTarg[qb])-s[qb]*sTarg[qb])+(3-p2-3*p1)/12*sum([np.dot(S[qb][:,i], STarg[qb][:,i])-S[qb][:,i]*STarg[qb][:,i]+np.dot(S[(qb+2)%3][i,:], STarg[(qb+2)%3][i,:])-S[(qb+2)%3][i,:]*STarg[(qb+2)%3][i,:] for i in range(3)])+(
            (1-p2-p1)/4*sum([np.dot(r[indR1[i][j]], rTarg[indR1[i][j]])-r[indR1[i][j]]*rTarg[indR1[i][j]] for i in range(3) for j in range(3)]))
    
    dot2 = (3+p1+p2)/12*(np.dot(s[(qb+1)%3], s[(qb+1)%3])-s[(qb+1)%3]*sTarg[(qb+1)%3])+(3-p2-3*p1)/12*sum([np.dot(S[qb][i,:], STarg[qb][i,:])-S[qb][i,:]*STarg[qb][i,:]+np.dot(S[(qb+1)%3][:,i], STarg[(qb+1)%3][:,i])-S[(qb+1)%3][:,i]*STarg[(qb+1)%3][:,i] for i in range(3)])+(
            (1-p2-p1)/4*sum([np.dot(r[indR2[i][j]], rTarg[indR2[i][j]])-r[indR2[i][j]]*rTarg[indR2[i][j]] for i in range(3) for j in range(3)]))
    
    ###b1, b2=z,z
    ##s1=s2=+
    deltaF[:9] = -J/2*deltaT*(np.kron(cross1,[1,1,1])+np.kron([1,1,1],cross2))
    #a1=x, a2=x,y,z
    #deltaF[0:3] = -J/2*deltaT*(cross1[0]+cross2)
    #a1=y, a2=x,y,z
    #deltaF[3:6] = -J/2*deltaT*(cross1[1]+cross2)
    #a1=z, a2=x,y,z
    #deltaF[6:9] = -J/2*deltaT*(cross1[2]+cross2)
    ##s1=+,s2=-
    deltaF[9:18] = -J/2*deltaT*(np.kron(cross1,[1,1,1])-np.kron([1,1,1],cross2))
    #a1=x, a2=x,y,z
    #deltaF[9:12] = -J/2*deltaT*(cross1[0]-cross2)
    #a1=y, a2=x,y,z
    #deltaF[12:15] = -J/2*deltaT*(cross1[1]-cross2)
    #a1=z, a2=x,y,z
    #deltaF[15:18] = -J/2*deltaT*(cross1[2]-cross2)
    
    ##s1=-, s2=+,-
    deltaF[18:36] = -deltaF[:18]
    
    ###b1=z, b2=x
    ##s1=+
    deltaF[36:45] = -J/2*deltaT*np.kron(cross1,[1,1,1])+Gamma/2*deltaT*np.kron([1,1,1],dot2)
    #a1=x, a2=x,y,z
    #deltaF[36:39] = -J/2*deltaT*cross1[0]+Gamma/2*deltaT*dot2
    #a1=y, a2=x,y,z
    #deltaF[39:42] = -J/2*deltaT*cross1[1]+Gamma/2*deltaT*dot2
    #a1=z, a2=x,y,z
    #deltaF[42:45] = -J/2*deltaT*cross1[2]+Gamma/2*deltaT*dot2
    ##s2=-
    deltaF[45:54] = J/2*deltaT*np.kron(cross1,[1,1,1])+Gamma/2*deltaT*np.kron([1,1,1],dot2)
    #a1=x, a2=x,y,z
    #deltaF[45:48] = J/2*deltaT*cross1[0]+Gamma/2*deltaT*dot2
    #a1=y, a2=x,y,z
    #deltaF[48:51] = J/2*deltaT*cross1[1]+Gamma/2*deltaT*dot2
    #a1=z, a2=x,y,z
    #deltaF[51:54] = J/2*deltaT*cross1[2]+Gamma/2*deltaT*dot2
    
    ###b1=x, b2=z
    ##s2=+
    deltaF[54:63] = -J/2*deltaT*np.kron(cross2,[1,1,1])+Gamma/2*deltaT*np.kron([1,1,1],dot1)
    #a2=x, a1=x,y,z
    #deltaF[54:57] = -J/2*deltaT*cross2[0]+Gamma/2*deltaT*dot1
    #a2=y, a1=x,y,z
    #deltaF[57:60] = -J/2*deltaT*cross2[1]+Gamma/2*deltaT*dot1
    #a2=z, a1=x,y,z
    #deltaF[60:63] = -J/2*deltaT*cross2[2]+Gamma/2*deltaT*dot1
    ##s2=-
    deltaF[63:72] = J/2*deltaT*np.kron(cross2,[1,1,1])+Gamma/2*deltaT*np.kron([1,1,1],dot1)
    #a2=x, a1=x,y,z
    #deltaF[63:66] = J/2*deltaT*cross2[0]+Gamma/2*deltaT*dot1
    #a2=y, a1=x,y,z
    #deltaF[66:69] = J/2*deltaT*cross2[1]+Gamma/2*deltaT*dot1
    #a2=z, a1=x,y,z
    #deltaF[69:72] = J/2*deltaT*cross2[2]+Gamma/2*deltaT*dot1
    
    ###b1=x, b2=x
    #a1=x, a2=x,y,z
    #deltaF[72:75] = Gamma/2*deltaT*(dot1[0]+dot2)
    deltaF[72:81] = Gamma/2*deltaT*(np.kron(dot1,[1,1,1])+np.kron([1,1,1],dot2))
    
    for j in range(3):
        for i in range(3):
            #deltaF[72+3*j+i]+= -p/2*Gamma*deltaT/2*(np.dot(s1,s1)-s1[j]*s1[j]+np.dot(s2,s2)-s2[i]*s2[i])
            if S[qb][j,i]**2==1:
                #deltaF[72+3*j+i]+= p/2*Gamma*deltaT/8*(2*(s2[i]-S[j,i]*s1[j])**2)
                deltaF[72+3*j+i]+= (p1+p2)/3*Gamma*deltaT/8*(2*(s[(qb+1)%3][i]-S[qb][j,i]*s[qb][j])**2-4*(np.dot(s[qb],s[qb])-s[qb][j]*s[qb][j])+
                                                                         -4*(np.dot(s[(qb+1)%3],s[(qb+1)%3])-s[(qb+1)%3][i]*s[(qb+1)%3][i])+np.dot(r[indR3[j][i]]-S[qb][j,i]*s[(qb+2)%3],r[indR3[j][i]]-S[qb][j,i]*s[(qb+2)%3]))
                deltaF[72+3*j+i]+= p2/6*Gamma*deltaT/8*(-4*(np.dot(S[qb][:,i], S[qb][:,i])-S[qb][j,i]*S[qb][j,i]+sum([np.dot(S[(qb+2)%3][k,:], S[(qb+2)%3][k,:])-S[(qb+2)%3][k,j]*S[(qb+2)%3][k,j] for k in range(3)])
                                                           +np.dot(S[qb][j,:], S[qb][j,:])-S[qb][j,i]*S[qb][j,i]+sum([np.dot(S[(qb+1)%3][:,k], S[(qb+1)%3][:,k])-S[(qb+1)%3][i,k]*S[(qb+1)%3][i,k] for k in range(3)]))
                                                                    +np.dot(S[(qb+1)%3][i,:]-S[qb][j,i]*S[(qb+2)%3][:,j],S[(qb+1)%3][i,:]-S[qb][j,i]*S[(qb+2)%3][:,j])+np.dot(S[(qb+2)%3][:,j]-S[qb][j,i]*S[(qb+1)%3][i,:],S[(qb+2)%3][:,j]-S[qb][j,i]*S[(qb+1)%3][i,:])
                                                                    +(1+S[qb][j,i])**2*(S[qb][(j+1)%3,(i+1)%3]**2+S[qb][(j+2)%3,(i+2)%3]**2)+(-1+S[qb][j,i])**2*(S[qb][(j+1)%3,(i+2)%3]**2+S[qb][(j+2)%3,(i+1)%3]**2))
            else:
                #deltaF[72+3*j+i]+= p/2*Gamma*deltaT/2/(1-S[j,i]**2)*((s2[i]-S[j,i]*s1[j])**2+(s1[j]-S[j,i]*s2[i])**2)
                deltaF[72+3*j+i]+= (p1+p2)/3*Gamma*deltaT/2/(1-S[qb][j,i]**2)*((s[(qb+1)%3][i]-S[qb][j,i]*s[qb][j])**2-(1-S[qb][j,i]**2)*(np.dot(s[qb],s[qb])-s[qb][j]*s[qb][j])+
                                                                         (s[qb][j]-S[qb][j,i]*s[(qb+1)%3][i])**2-(1-S[qb][j,i]**2)*(np.dot(s[(qb+1)%3],s[(qb+1)%3])-s[(qb+1)%3][i]*s[(qb+1)%3][i])+np.dot(r[indR3[j][i]]-S[qb][j,i]*s[(qb+2)%3],r[indR3[j][i]]-S[qb][j,i]*s[(qb+2)%3]))
                deltaF[72+3*j+i]+= p2/6*Gamma*deltaT/2/(1-S[qb][j,i]**2)*((1-S[qb][j,i]**2)**2-(1-S[qb][j,i]**2)*(np.dot(S[qb][:,i], S[qb][:,i])-S[qb][j,i]*S[qb][j,i]+sum([np.dot(S[(qb+2)%3][k,:], S[(qb+2)%3][k,:])-S[(qb+2)%3][k,j]*S[(qb+2)%3][k,j] for k in range(3)])
                                                                                                                 +np.dot(S[qb][j,:], S[qb][j,:])-S[qb][j,i]*S[qb][j,i]+sum([np.dot(S[(qb+1)%3][:,k], S[(qb+1)%3][:,k])-S[(qb+1)%3][i,k]*S[(qb+1)%3][i,k] for k in range(3)]))
                                                                    +np.dot(S[(qb+1)%3][i,:]-S[qb][j,i]*S[(qb+2)%3][:,j],S[(qb+1)%3][i,:]-S[qb][j,i]*S[(qb+2)%3][:,j])+np.dot(S[(qb+2)%3][:,j]-S[qb][j,i]*S[(qb+1)%3][i,:],S[(qb+2)%3][:,j]-S[qb][j,i]*S[(qb+1)%3][i,:])
                                                                    +(1+S[qb][j,i])**2*(S[qb][(j+1)%3,(i+1)%3]**2+S[qb][(j+2)%3,(i+2)%3]**2)+(-1+S[qb][j,i])**2*(S[qb][(j+1)%3,(i+2)%3]**2+S[qb][(j+2)%3,(i+1)%3]**2))
            
    #if S[qb][0,0]**2==1:
    #    #deltaF[72]+= p1/2*Gamma*deltaT/8*(2*(s[(qb+1)%3][0]-S[qb][0,0]*s[qb][0])**2-4*(np.dot(s[qb],s[qb])-s[qb][0]*s[qb][0])-4*(np.dot(s[(qb+1)%3],s[(qb+1)%3])-s[(qb+1)%3][0]*s[(qb+1)%3][0]))
    #    deltaF[72]+= (p1+p2)/3*Gamma*deltaT/8*(2*(s[(qb+1)%3][0]-S[qb][0,0]*s[qb][0])**2-4*(np.dot(s[qb],s[qb])-s[qb][0]*s[qb][0])+
    #                                                             -4*(np.dot(s[(qb+1)%3],s[(qb+1)%3])-s[(qb+1)%3][0]*s[(qb+1)%3][0])+np.dot(r[indR3[0][0]]-S[qb][0,0]*s[(qb+2)%3],r[indR3[0][0]]-S[qb][0,0]*s[(qb+2)%3]))
    #    deltaF[72]+= p2/6*Gamma*deltaT/8*(-4*(np.dot(S[qb][:,0], S[qb][:,0])-S[qb][0,0]*S[qb][0,0]+sum([np.dot(S[(qb+2)%3][i,:], S[(qb+2)%3][i,:])-S[(qb+2)%3][i,0]*S[(qb+2)%3][i,0] for i in range(3)])
    #                                         +np.dot(S[qb][0,:], S[qb][0,:])-S[qb][0,0]*S[qb][0,0]+sum([np.dot(S[(qb+1)%3][:,i], S[(qb+1)%3][:,i])-S[(qb+1)%3][0,i]*S[(qb+1)%3][0,i] for i in range(3)]))
    #                                                        +np.dot(S[(qb+1)%3][0,:]-S[qb][0,0]*S[(qb+2)%3][:,0],S[(qb+1)%3][0,:]-S[qb][0,0]*S[(qb+2)%3][:,0])+np.dot(S[(qb+2)%3][:,0]-S[qb][0,0]*S[(qb+1)%3][0,:],S[(qb+2)%3][:,0]-S[qb][0,0]*S[(qb+1)%3][0,:])
    #                                                        +(1+S[qb][0,0])**2*(S[qb][1,1]**2+S[qb][2,2]**2)+(-1+S[qb][0,0])**2*(S[qb][1,2]**2+S[qb][2,1]**2))
    #else:
    #    deltaF[72]+= (p1+p2)/3*Gamma*deltaT/2/(1-S[qb][0,0]**2)*((s[(qb+1)%3][0]-S[qb][0,0]*s[qb][0])**2-(1-S[qb][0,0]**2)*(np.dot(s[qb],s[qb])-s[qb][0]*s[qb][0])+
    #                                                             (s[qb][0]-S[qb][0,0]*s[(qb+1)%3][0])**2-(1-S[qb][0,0]**2)*(np.dot(s[(qb+1)%3],s[(qb+1)%3])-s[(qb+1)%3][0]*s[(qb+1)%3][0])+np.dot(r[indR3[0][0]]-S[qb][0,0]*s[(qb+2)%3],r[indR3[0][0]]-S[qb][0,0]*s[(qb+2)%3]))
    #    deltaF[72]+= p2/6*Gamma*deltaT/2/(1-S[qb][0,0]**2)*((1-S[qb][0,0]**2)**2-(1-S[qb][0,0]**2)*(np.dot(S[qb][:,0], S[qb][:,0])-S[qb][0,0]*S[qb][0,0]+sum([np.dot(S[(qb+2)%3][i,:], S[(qb+2)%3][i,:])-S[(qb+2)%3][i,0]*S[(qb+2)%3][i,0] for i in range(3)])
    #                                                                                               +np.dot(S[qb][0,:], S[qb][0,:])-S[qb][0,0]*S[qb][0,0]+sum([np.dot(S[(qb+1)%3][:,i], S[(qb+1)%3][:,i])-S[(qb+1)%3][0,i]*S[(qb+1)%3][0,i] for i in range(3)]))
    #                                                        +np.dot(S[(qb+1)%3][0,:]-S[qb][0,0]*S[(qb+2)%3][:,0],S[(qb+1)%3][0,:]-S[qb][0,0]*S[(qb+2)%3][:,0])+np.dot(S[(qb+2)%3][:,0]-S[qb][0,0]*S[(qb+1)%3][0,:],S[(qb+2)%3][:,0]-S[qb][0,0]*S[(qb+1)%3][0,:])
    #                                                        +(1+S[qb][0,0])**2*(S[qb][1,1]**2+S[qb][2,2]**2)+(-1+S[qb][0,0])**2*(S[qb][1,2]**2+S[qb][2,1]**2))
    
    #if S[qb][0,1]**2==1:
    #    deltaF[73]+= p1/2*Gamma*deltaT/8*(2*(s[(qb+1)%3][1]-S[qb][0,1]*s[qb][0])**2-4*(np.dot(s[qb],s[qb])-s[qb][0]*s[qb][0])-4*(np.dot(s[(qb+1)%3],s[(qb+1)%3])-s[(qb+1)%3][1]*s[(qb+1)%3][1]))
    #else:
    #    deltaF[73]+= (p1+p2)/3*Gamma*deltaT/2/(1-S[qb][0,1]**2)*((s[(qb+1)%3][1]-S[qb][0,1]*s[qb][0])**2-(1-S[qb][0,1]**2)*(np.dot(s[qb],s[qb])-s[qb][0]*s[qb][0])+
    #                                                             (s[qb][0]-S[qb][0,1]*s[(qb+1)%3][1])**2-(1-S[qb][0,1]**2)*(np.dot(s[(qb+1)%3],s[(qb+1)%3])-s[(qb+1)%3][1]*s[(qb+1)%3][1])+np.dot(r[indR3[0][1]]-S[qb][0,1]*s[(qb+2)%3],r[indR3[0][1]]-S[qb][0,1]*s[(qb+2)%3]))
    #    deltaF[73]+= p2/6*Gamma*deltaT/2/(1-S[qb][0,1]**2)*((1-S[qb][0,1]**2)**2-(1-S[qb][0,1]**2)*(np.dot(S[qb][:,1], S[qb][:,1])-S[qb][0,1]*S[qb][0,1]+sum([np.dot(S[(qb+2)%3][i,:], S[(qb+2)%3][i,:])-S[(qb+2)%3][i,0]*S[(qb+2)%3][i,0] for i in range(3)])
    #                                                                                               +np.dot(S[qb][0,:], S[qb][0,:])-S[qb][0,1]*S[qb][0,1]+sum([np.dot(S[(qb+1)%3][:,i], S[(qb+1)%3][:,i])-S[(qb+1)%3][1,i]*S[(qb+1)%3][1,i] for i in range(3)]))
    #                                                        +np.dot(S[(qb+1)%3][1,:]-S[qb][0,1]*S[(qb+2)%3][:,0],S[(qb+1)%3][1,:]-S[qb][0,1]*S[(qb+2)%3][:,0])+np.dot(S[(qb+2)%3][:,0]-S[qb][0,1]*S[(qb+1)%3][1,:],S[(qb+2)%3][:,0]-S[qb][0,1]*S[(qb+1)%3][1,:])
    #                                                        +(1+S[qb][0,1])**2*(S[qb][1,2]**2+S[qb][2,0]**2)+(-1+S[qb][0,1])**2*(S[qb][1,0]**2+S[qb][2,2]**2))
    
    #if S[qb][0,2]**2==1:
    #    deltaF[74]+= p1/2*Gamma*deltaT/8*(2*(s[(qb+1)%3][2]-S[qb][0,2]*s[qb][0])**2-4*(np.dot(s[qb],s[qb])-s[qb][0]*s[qb][0])-4*(np.dot(s[(qb+1)%3],s[(qb+1)%3])-s[(qb+1)%3][2]*s[(qb+1)%3][2]))
    #else:
    #    deltaF[74]+= p1/2*Gamma*deltaT/2/(1-S[qb][0,2]**2)*((s[(qb+1)%3][2]-S[qb][0,2]*s[qb][0])**2-(1-S[qb][0,2]**2)*(np.dot(s[qb],s[qb])-s[qb][0]*s[qb][0])+
    #                                            (s[qb][0]-S[qb][0,2]*s[(qb+1)%3][2])**2-(1-S[qb][0,2]**2)*(np.dot(s[(qb+1)%3],s[(qb+1)%3])-s[(qb+1)%3][2]*s[(qb+1)%3][2]))
    
    #a1=y, a2=x,y,z
    #deltaF[75:78] = Gamma/2*deltaT*(dot1[1]+dot2)
    
    #if S[qb][1,0]**2==1:
    #    deltaF[75]+= p1/2*Gamma*deltaT/8*(2*(s[(qb+1)%3][0]-S[qb][1,0]*s[qb][1])**2-4*(np.dot(s[qb],s[qb])-s[qb][1]*s[qb][1])-4*(np.dot(s[(qb+1)%3],s[(qb+1)%3])-s[(qb+1)%3][0]*s[(qb+1)%3][0]))
    #else:
    #    deltaF[75]+= p1/2*Gamma*deltaT/2/(1-S[qb][1,0]**2)*((s[(qb+1)%3][0]-S[qb][1,0]*s[qb][1])**2-(1-S[qb][1,0]**2)*(np.dot(s[qb],s[qb])-s[qb][1]*s[qb][1])+
    #                                            (s[qb][1]-S[qb][1,0]*s[(qb+1)%3][0])**2-(1-S[qb][1,0]**2)*(np.dot(s[(qb+1)%3],s[(qb+1)%3])-s[(qb+1)%3][0]*s[(qb+1)%3][0]))
    
    #if S[qb][1,1]**2==1:
    #    deltaF[76]+= p1/2*Gamma*deltaT/8*(2*(s[(qb+1)%3][1]-S[qb][1,1]*s[qb][1])**2-4*(np.dot(s[qb],s[qb])-s[qb][1]*s[qb][1])-4*(np.dot(s[(qb+1)%3],s[(qb+1)%3])-s[(qb+1)%3][1]*s[(qb+1)%3][1]))
    #else:
    #    deltaF[76]+= p1/2*Gamma*deltaT/2/(1-S[qb][1,1]**2)*((s[(qb+1)%3][1]-S[qb][1,1]*s[qb][1])**2-(1-S[qb][1,1]**2)*(np.dot(s[qb],s[qb])-s[qb][1]*s[qb][1])+
    #                                            (s[qb][1]-S[qb][1,1]*s[(qb+1)%3][1])**2-(1-S[qb][1,1]**2)*(np.dot(s[(qb+1)%3],s[(qb+1)%3])-s[(qb+1)%3][1]*s[(qb+1)%3][1]))
    
    #if S[qb][1,2]**2==1:
    #    deltaF[77]+= p1/2*Gamma*deltaT/8*(2*(s[(qb+1)%3][2]-S[qb][1,2]*s[qb][1])**2-4*(np.dot(s[qb],s[qb])-s[qb][1]*s[qb][1])-4*(np.dot(s[(qb+1)%3],s[(qb+1)%3])-s[(qb+1)%3][2]*s[(qb+1)%3][2]))
    #else:
    #    deltaF[77]+= p1/2*Gamma*deltaT/2/(1-S[qb][1,2]**2)*((s[(qb+1)%3][2]-S[qb][1,2]*s[qb][1])**2-(1-S[qb][1,2]**2)*(np.dot(s[qb],s[qb])-s[qb][1]*s[qb][1])+
    #                                            (s[qb][1]-S[qb][1,2]*s[(qb+1)%3][2])**2-(1-S[qb][1,2]**2)*(np.dot(s[(qb+1)%3],s[(qb+1)%3])-s[(qb+1)%3][2]*s[(qb+1)%3][2]))
    
    #a1=z, a2=x,y,z
    #deltaF[78:81] = Gamma/2*deltaT*(dot1[2]+dot2)
    
    #if S[qb][2,0]**2==1:
    #    deltaF[78]+= p1/2*Gamma*deltaT/8*(2*(s[(qb+1)%3][0]-S[qb][2,0]*s[qb][2])**2-4*(np.dot(s[qb],s[qb])-s[qb][2]*s[qb][2])-4*(np.dot(s[(qb+1)%3],s[(qb+1)%3])-s[(qb+1)%3][0]*s[(qb+1)%3][0]))
    #else:
    #    deltaF[78]+= p1/2*Gamma*deltaT/2/(1-S[qb][2,0]**2)*((s[(qb+1)%3][0]-S[qb][2,0]*s[qb][2])**2-(1-S[qb][2,0]**2)*(np.dot(s[qb],s[qb])-s[qb][2]*s[qb][2])+
    #                                            (s[qb][2]-S[qb][2,0]*s[(qb+1)%3][0])**2-(1-S[qb][2,0]**2)*(np.dot(s[(qb+1)%3],s[(qb+1)%3])-s[(qb+1)%3][0]*s[(qb+1)%3][0]))
    
    #if S[qb][2,1]**2==1:
    #    deltaF[79]+= p1/2*Gamma*deltaT/8*(2*(s[(qb+1)%3][1]-S[qb][2,1]*s[qb][2])**2-4*(np.dot(s[qb],s[qb])-s[qb][2]*s[qb][2])-4*(np.dot(s[(qb+1)%3],s[(qb+1)%3])-s[(qb+1)%3][1]*s[(qb+1)%3][1]))
    #else:
    #    deltaF[79]+= p1/2*Gamma*deltaT/2/(1-S[qb][2,1]**2)*((s[(qb+1)%3][1]-S[qb][2,1]*s[qb][2])**2-(1-S[qb][2,1]**2)*(np.dot(s[qb],s[qb])-s[qb][2]*s[qb][2])+
    #                                            (s[qb][2]-S[qb][2,1]*s[(qb+1)%3][1])**2-(1-S[qb][2,1]**2)*(np.dot(s[(qb+1)%3],s[(qb+1)%3])-s[(qb+1)%3][1]*s[(qb+1)%3][1]))
    
    #if S[qb][2,2]**2==1:
    #    deltaF[80]+= p1/2*Gamma*deltaT/8*(2*(s[(qb+1)%3][2]-S[qb][2,2]*s[qb][2])**2-4*(np.dot(s[qb],s[qb])-s[qb][2]*s[qb][2])-4*(np.dot(s[(qb+1)%3],s[(qb+1)%3])-s[(qb+1)%3][2]*s[(qb+1)%3][2]))
    #else:
    #    deltaF[80]+= p1/2*Gamma*deltaT/2/(1-S[qb][2,2]**2)*((s[(qb+1)%3][2]-S[qb][2,2]*s[qb][2])**2-(1-S[qb][2,2]**2)*(np.dot(s[qb],s[qb])-s[qb][2]*s[qb][2])+
    #                                            (s[qb][2]-S[qb][2,2]*s[(qb+1)%3][2])**2-(1-S[qb][2,2]**2)*(np.dot(s[(qb+1)%3],s[(qb+1)%3])-s[(qb+1)%3][2]*s[(qb+1)%3][2]))
        
    #a1=0, a2=x,y,z
    #deltaF[81:84] = Gamma/2*deltaT*dot2
    
    #if s[(qb+1)%3][0]**2==1:
    #    deltaF[81] += p1/2*Gamma*deltaT/8*(4*(1-np.dot(s[(qb+1)%3],s[(qb+1)%3]))+np.dot(S[qb][:,0]-s[(qb+1)%3][0]*s[qb], S[qb][:,0]-s[(qb+1)%3][0]*s[qb]))
    #else:
    #    deltaF[81] += p1/2*Gamma*deltaT/2/(1-s[(qb+1)%3][0]**2)*(np.dot(S[qb][:,0]-s[(qb+1)%3][0]*s[qb], S[qb][:,0]-s[(qb+1)%3][0]*s[qb])+(1-s[(qb+1)%3][0]**2)*(1-np.dot(s[(qb+1)%3],s[(qb+1)%3])))
    
    #if s[(qb+1)%3][1]**2==1:
    #    deltaF[82] += p1/2*Gamma*deltaT/8*(4*(1-np.dot(s[(qb+1)%3],s[(qb+1)%3]))+np.dot(S[qb][:,1]-s[(qb+1)%3][1]*s[qb], S[qb][:,1]-s[(qb+1)%3][1]*s[qb]))
    #else:
    #    deltaF[82] += p1/2*Gamma*deltaT/2/(1-s[(qb+1)%3][1]**2)*(np.dot(S[qb][:,1]-s[(qb+1)%3][1]*s[qb], S[qb][:,1]-s[(qb+1)%3][1]*s[qb])+(1-s[(qb+1)%3][1]**2)*(1-np.dot(s[(qb+1)%3],s[(qb+1)%3])))
    
    #if s[(qb+1)%3][2]**2==1:
    #    deltaF[83] += p1/2*Gamma*deltaT/8*(4*(1-np.dot(s[(qb+1)%3],s[(qb+1)%3]))+np.dot(S[qb][:,2]-s[(qb+1)%3][2]*s[qb], S[qb][:,2]-s[(qb+1)%3][2]*s[qb]))
    #else:
    #    deltaF[83] += p1/2*Gamma*deltaT/2/(1-s[(qb+1)%3][2]**2)*(np.dot(S[qb][:,2]-s[(qb+1)%3][2]*s[qb], S[qb][:,2]-s[(qb+1)%3][2]*s[qb])+(1-s[(qb+1)%3][2]**2)*(1-np.dot(s[(qb+1)%3],s[(qb+1)%3])))
    
    #a2=0, a1=x,y,z
    #deltaF[84:87] = Gamma/2*deltaT*dot1
    
    #if s[qb][0]**2==1:
    #    deltaF[84] += p1/2*Gamma*deltaT/8*(4*(1-np.dot(s[qb],s[qb]))+np.dot(S[qb][0,:]-s[qb][0]*s[(qb+1)%3], S[qb][0,:]-s[qb][0]*s[(qb+1)%3]))
    #else:
    #    deltaF[84] += p1/2*Gamma*deltaT/2/(1-s[qb][0]**2)*(np.dot(S[qb][0,:]-s[qb][0]*s[(qb+1)%3], S[qb][0,:]-s[qb][0]*s[(qb+1)%3])+(1-s[qb][0]**2)*(1-np.dot(s[qb],s[qb])))
    
    #if s[qb][1]**2==1:
    #    deltaF[85] += p1/2*Gamma*deltaT/8*(4*(1-np.dot(s[qb],s[qb]))+np.dot(S[qb][1,:]-s[qb][1]*s[(qb+1)%3], S[qb][1,:]-s[qb][1]*s[(qb+1)%3]))
    #else:
    #    deltaF[85] += p1/2*Gamma*deltaT/2/(1-s[qb][1]**2)*(np.dot(S[qb][1,:]-s[qb][1]*s[(qb+1)%3], S[qb][1,:]-s[qb][1]*s[(qb+1)%3])+(1-s[qb][1]**2)*(1-np.dot(s[qb],s[qb])))
    
    #if s[qb][2]**2==1:
    #    deltaF[86] += p1/2*Gamma*deltaT/8*(4*(1-np.dot(s[qb],s[qb]))+np.dot(S[qb][2,:]-s[qb][2]*s[(qb+1)%3], S[qb][2,:]-s[qb][2]*s[(qb+1)%3]))
    #else:
    #    deltaF[86] += p1/2*Gamma*deltaT/2/(1-s[qb][2]**2)*(np.dot(S[qb][2,:]-s[qb][2]*s[(qb+1)%3], S[qb][2,:]-s[qb][2]*s[(qb+1)%3])+(1-s[qb][2]**2)*(1-np.dot(s[qb],s[qb])))
    
    ###b1=x, b2=y
    #a1=x, a2=x,y,z
    #deltaF[87:90] = Gamma/2*deltaT*(dot1[0]+dot2)
    #deltaF[81:90] = Gamma/2*deltaT*(np.kron(dot1,[1,1,1])+np.kron([1,1,1],dot2)+
    #                       -(p1/3+p2/6)*(np.dot(s[qb], s[qb])-np.kron([1,1,1],np.diag(S[qb].T@S[qb])+s[(qb+1)%3]*s[(qb+1)%3])+2*np.concatenate(S[qb]*S[qb])+
    #                            np.dot(s[(qb+1)%3], s[(qb+1)%3])-np.kron(np.diag(S[qb].T@S[qb])+s[qb]*s[qb],[1,1,1])))
    
    ###vielleicht + im zweiten Teil oder - im ersten?
    #for j in range(3):
    #    for i in range(3):
    #        deltaF[81+3*j+i] += p2/6*Gamma*deltaT/2*(sum([np.dot(r[indR1[i][k]], r[indR1[i][k]])-(r[indR1[i][k]]*r[indR1[i][k]])[j] for k in range(3)])+
    #                                                 sum([np.dot(r[indR2[j][k]], r[indR2[j][k]])-(r[indR2[j][k]]*r[indR2[j][k]])[i] for k in range(3)])+
    #                                                 -sum([np.dot(S[(qb+2)%3][k,:], S[(qb+2)%3][k,:])-(S[(qb+2)%3][k,:]*S[(qb+2)%3][k,:])[j] for k in range(3)])
    #                                                 -sum([np.dot(S[(qb+1)%3][:,k], S[(qb+1)%3][:,k])-(S[(qb+1)%3][:,k]*S[(qb+1)%3][:,k])[i] for k in range(3)]))
            
    #deltaF[81] += p2/6*Gamma*deltaT/2*(sum([np.dot(r[indR1[0][j]], r[indR1[0][j]])-(r[indR1[0][j]]*r[indR1[0][j]])[0] for j in range(3)])+
    #                                   sum([np.dot(r[indR2[0][j]], r[indR2[0][j]])-(r[indR2[0][j]]*r[indR2[0][j]])[0] for j in range(3)])+
    #                                   -sum([np.dot(S[(qb+2)%3][j,:], S[(qb+2)%3][j,:])-(S[(qb+2)%3][j,:]*S[(qb+2)%3][j,:])[0] for j in range(3)])
    #                                   -sum([np.dot(S[(qb+1)%3][:,j], S[(qb+1)%3][:,j])-(S[(qb+1)%3][:,j]*S[(qb+1)%3][:,j])[0] for j in range(3)]))
    #deltaF[82] += p2/6*Gamma*deltaT/2*(sum([np.dot(r[indR1[1][j]], r[indR1[1][j]])-(r[indR1[1][j]]*r[indR1[1][j]])[0] for j in range(3)])+
    #                                   sum([np.dot(r[indR2[0][j]], r[indR2[0][j]])-(r[indR2[0][j]]*r[indR2[0][j]])[1] for j in range(3)])+
    #                                   -sum([np.dot(S[(qb+2)%3][j,:], S[(qb+2)%3][j,:])-(S[(qb+2)%3][j,:]*S[(qb+2)%3][j,:])[0] for j in range(3)])
    #                                   -sum([np.dot(S[(qb+1)%3][:,j], S[(qb+1)%3][:,j])-(S[(qb+1)%3][:,j]*S[(qb+1)%3][:,j])[1] for j in range(3)]))
    #deltaF[83] += p2/6*Gamma*deltaT/2*(sum([np.dot(r[indR1[2][j]], r[indR1[2][j]])-(r[indR1[2][j]]*r[indR1[2][j]])[0] for j in range(3)])+
    #                                   sum([np.dot(r[indR2[0][j]], r[indR2[0][j]])-(r[indR2[0][j]]*r[indR2[0][j]])[2] for j in range(3)])+
    #                                   -sum([np.dot(S[(qb+2)%3][j,:], S[(qb+2)%3][j,:])-(S[(qb+2)%3][j,:]*S[(qb+2)%3][j,:])[0] for j in range(3)])
    #                                   -sum([np.dot(S[(qb+1)%3][:,j], S[(qb+1)%3][:,j])-(S[(qb+1)%3][:,j]*S[(qb+1)%3][:,j])[2] for j in range(3)]))
    
    #deltaF[84] += p2/6*Gamma*deltaT/2*(sum([np.dot(r[indR1[0][j]], r[indR1[0][j]])-(r[indR1[0][j]]*r[indR1[0][j]])[1] for j in range(3)])+
    #                                   sum([np.dot(r[indR2[1][j]], r[indR2[1][j]])-(r[indR2[1][j]]*r[indR2[1][j]])[0] for j in range(3)])+
    #                                   -sum([np.dot(S[(qb+2)%3][j,:], S[(qb+2)%3][j,:])-(S[(qb+2)%3][j,:]*S[(qb+2)%3][j,:])[1] for j in range(3)])
    #                                   -sum([np.dot(S[(qb+1)%3][:,j], S[(qb+1)%3][:,j])-(S[(qb+1)%3][:,j]*S[(qb+1)%3][:,j])[0] for j in range(3)]))
    #deltaF[85] += p2/6*Gamma*deltaT/2*(sum([np.dot(r[indR1[1][j]], r[indR1[1][j]])-(r[indR1[1][j]]*r[indR1[1][j]])[1] for j in range(3)])+
    #                                   sum([np.dot(r[indR2[1][j]], r[indR2[1][j]])-(r[indR2[1][j]]*r[indR2[1][j]])[1] for j in range(3)])+
    #                                   -sum([np.dot(S[(qb+2)%3][j,:], S[(qb+2)%3][j,:])-(S[(qb+2)%3][j,:]*S[(qb+2)%3][j,:])[1] for j in range(3)])
    #                                   -sum([np.dot(S[(qb+1)%3][:,j], S[(qb+1)%3][:,j])-(S[(qb+1)%3][:,j]*S[(qb+1)%3][:,j])[1] for j in range(3)]))
    #deltaF[86] += p2/6*Gamma*deltaT/2*(sum([np.dot(r[indR1[2][j]], r[indR1[2][j]])-(r[indR1[2][j]]*r[indR1[2][j]])[1] for j in range(3)])+
    #                                   sum([np.dot(r[indR2[1][j]], r[indR2[1][j]])-(r[indR2[1][j]]*r[indR2[1][j]])[2] for j in range(3)])+
    #                                   -sum([np.dot(S[(qb+2)%3][j,:], S[(qb+2)%3][j,:])-(S[(qb+2)%3][j,:]*S[(qb+2)%3][j,:])[1] for j in range(3)])
    #                                   -sum([np.dot(S[(qb+1)%3][:,j], S[(qb+1)%3][:,j])-(S[(qb+1)%3][:,j]*S[(qb+1)%3][:,j])[2] for j in range(3)]))
    
    #deltaF[87] += p2/6*Gamma*deltaT/2*(sum([np.dot(r[indR1[0][j]], r[indR1[0][j]])-(r[indR1[0][j]]*r[indR1[0][j]])[2] for j in range(3)])+
    #                                   sum([np.dot(r[indR2[2][j]], r[indR2[2][j]])-(r[indR2[2][j]]*r[indR2[2][j]])[0] for j in range(3)])+
    #                                   -sum([np.dot(S[(qb+2)%3][j,:], S[(qb+2)%3][j,:])-(S[(qb+2)%3][j,:]*S[(qb+2)%3][j,:])[2] for j in range(3)])
    #                                   -sum([np.dot(S[(qb+1)%3][:,j], S[(qb+1)%3][:,j])-(S[(qb+1)%3][:,j]*S[(qb+1)%3][:,j])[0] for j in range(3)]))
    #deltaF[88] += p2/6*Gamma*deltaT/2*(sum([np.dot(r[indR1[1][j]], r[indR1[1][j]])-(r[indR1[1][j]]*r[indR1[1][j]])[2] for j in range(3)])+
    #                                   sum([np.dot(r[indR2[2][j]], r[indR2[2][j]])-(r[indR2[2][j]]*r[indR2[2][j]])[1] for j in range(3)])+
    #                                   -sum([np.dot(S[(qb+2)%3][j,:], S[(qb+2)%3][j,:])-(S[(qb+2)%3][j,:]*S[(qb+2)%3][j,:])[2] for j in range(3)])
    #                                   -sum([np.dot(S[(qb+1)%3][:,j], S[(qb+1)%3][:,j])-(S[(qb+1)%3][:,j]*S[(qb+1)%3][:,j])[1] for j in range(3)]))
    #deltaF[89] += p2/6*Gamma*deltaT/2*(sum([np.dot(r[indR1[2][j]], r[indR1[2][j]])-(r[indR1[2][j]]*r[indR1[2][j]])[2] for j in range(3)])+
    #                                   sum([np.dot(r[indR2[2][j]], r[indR2[2][j]])-(r[indR2[2][j]]*r[indR2[2][j]])[2] for j in range(3)])+
    #                                   -sum([np.dot(S[(qb+2)%3][j,:], S[(qb+2)%3][j,:])-(S[(qb+2)%3][j,:]*S[(qb+2)%3][j,:])[2] for j in range(3)])
    #                                   -sum([np.dot(S[(qb+1)%3][:,j], S[(qb+1)%3][:,j])-(S[(qb+1)%3][:,j]*S[(qb+1)%3][:,j])[2] for j in range(3)]))
    
    #deltaF[87] -= (p1/3+p2/6)*Gamma*deltaT/2*(np.dot(s1, s1)-np.dot(S[:,0], S[:,0])-s1[0]*s1[0]+2*S[0,0]*S[0,0]+
    #                           np.dot(s2, s2)-np.dot(S[0,:], S[0,:])-s2[0]*s2[0])
    #deltaF[88] += p/2*Gamma*deltaT/2*(np.dot(s1, s1)-np.dot(S[:,1], S[:,1])-s1[0]*s1[0]+2*S[0,1]*S[0,1]+
    #                           np.dot(s2, s2)-np.dot(S[0,:], S[0,:])-s2[1]*s2[1])
    #deltaF[89] += p/2*Gamma*deltaT/2*(np.dot(s1, s1)-np.dot(S[:,2], S[:,2])-s1[0]*s1[0]+2*S[0,2]*S[0,2]+
    #                           np.dot(s2, s2)-np.dot(S[0,:], S[0,:])-s2[2]*s2[2])
    
    #a1=y, a2=x,y,z
    #deltaF[90:93] = Gamma/2*deltaT*(dot1[1]+dot2)
    
    #deltaF[90] += p/2*Gamma*deltaT/2*(np.dot(s1, s1)-np.dot(S[:,0], S[:,0])-s1[1]*s1[1]+2*S[1,0]*S[1,0]+
    #                           np.dot(s2, s2)-np.dot(S[1,:], S[1,:])-s2[0]*s2[0])
    #deltaF[91] += p/2*Gamma*deltaT/2*(np.dot(s1, s1)-np.dot(S[:,1], S[:,1])-s1[1]*s1[1]+2*S[1,1]*S[1,1]+
    #                           np.dot(s2, s2)-np.dot(S[1,:], S[1,:])-s2[1]*s2[1])
    #deltaF[92] += p/2*Gamma*deltaT/2*(np.dot(s1, s1)-np.dot(S[:,2], S[:,2])-s1[1]*s1[1]+2*S[1,2]*S[1,2]+
    #                           np.dot(s2, s2)-np.dot(S[1,:], S[1,:])-s2[2]*s2[2])
    
    #a1=z, a2=x,y,z
    #deltaF[93:96] = Gamma/2*deltaT*(dot1[2]+dot2)
    
    #deltaF[93] += p/2*Gamma*deltaT/2*(np.dot(s1, s1)-np.dot(S[:,0], S[:,0])-s1[2]*s1[2]+2*S[2,0]*S[2,0]+
    #                           np.dot(s2, s2)-np.dot(S[2,:], S[2,:])-s2[0]*s2[0])
    #deltaF[94] += p/2*Gamma*deltaT/2*(np.dot(s1, s1)-np.dot(S[:,1], S[:,1])-s1[2]*s1[2]+2*S[2,1]*S[2,1]+
    #                           np.dot(s2, s2)-np.dot(S[2,:], S[2,:])-s2[1]*s2[1])
    #deltaF[95] += p/2*Gamma*deltaT/2*(np.dot(s1, s1)-np.dot(S[:,2], S[:,2])-s1[2]*s1[2]+2*S[2,2]*S[2,2]+
    #                           np.dot(s2, s2)-np.dot(S[2,:], S[2,:])-s2[2]*s2[2])
    
    #a1=0, a2=x,y,z
    #deltaF[96:99] = Gamma/2*deltaT*dot2
    
    #a2=0, a1=x,y,z
    #deltaF[99:102] = Gamma/2*deltaT*dot1
    
    return deltaF

###plaquette operators
def plaqS(ind):
    #pauliLs = [s[i] for i in ind]
    pauliS = s[ind[0]]
    for i in ind[1:]:
        pauliS = qt.tensor(pauliS, s[i])
    return pauliS
    #return qt.tensor(pauliLs)
    
def plaqSlist(Nqb):
    pauliS=[]
    for i in range(4**Nqb):
        ind = [int(i/(4**j)%4) for j in range(Nqb)]
        pauliS.append(plaqS(ind))
    return pauliS

##Bloch tensor
#def blochS(psi, Nqb):
#    return qt.expect(plaqSlist(Nqb),psi)

##cost function for all levels
def csFct(S, Starg, Nqb, subsets):
    costf = np.zeros(Nqb)
    #for i in range(Nqb-1):
    #    costf[i] = costFunc(S, Starg, i+1, Nqb, subsets[i])
    #costf[Nqb-1] = np.linalg.norm(S-Starg)**2/2**(Nqb+1)
    
    for j in range(4**Nqb):
        cost = (S[j]-Starg[j])**2
        for i in range(Nqb-1):
            for s in subsets[i]:
                chk = 0
                for l in range(Nqb):
                    if l not in s and int(j/(4**l)%4)>0:
                        chk +=1
                #chk = sum(np.logical_and(np.logical_not(np.isin(range(Nqb),s)), (j/4**np.array(range(Nqb))).astype(int)%4>0))
                if chk == 0:
                    costf[i] += cost
        costf[Nqb-1] += cost
                    
    for i in range(Nqb-1):
        costf[i] *= 1/2**(i+2)/scipy.special.binom(Nqb,i+1)
    costf[Nqb-1] *= 1/2**(Nqb+1)
    
    return costf

##cost function of level i #TODO: parallel?
def costFunc(S, Starg, i, Nqb, subset):
    costf = 0
    for s in subset:
        for j in range(4**Nqb):
            chk = 0
            for l in range(Nqb):
                if l not in s and int(j/(4**l)%4)>0:
                    chk +=1
            #chk = sum(np.logical_and(np.logical_not(np.isin(range(Nqb),s)), (j/4**np.array(range(Nqb))).astype(int)%4>0))
            if chk == 0:
                costf+=(S[j]-Starg[j])**2
    
    costf *= 1/2**(i+2)/scipy.special.binom(Nqb,i+1)
    return costf

##expected cost function change
def expCostF(S, Starg, J, Gamma, deltaT, p, nA, nB, Nqb, K):
    ##coupling list
    if K == 9:
        slist = [1,1,1,-1,-1,-1,1,1,1]
        aList = [1,2,3,1,2,3,1,2,3]
        bList = [3,3,3,3,3,3,1,1,1]
    elif K == 10:
        slist = [1,1,1,-1,-1,-1,1,1,1,1]
        aList = [1,2,3,1,2,3,0,1,2,3]
        bList = [3,3,3,3,3,3,1,1,1,1]
    elif K == 12:
        slist = [1,1,1,-1,-1,-1,1,1,1,1,1,1]
        aList = [1,2,3,1,2,3,1,2,3,1,2,3]
        bList = [3,3,3,3,3,3,1,1,1,2,2,2]
    elif K == 14:
        slist = [1,1,1,-1,-1,-1,1,1,1,1,1,1,1,1]
        aList = [1,2,3,1,2,3,0,1,2,3,0,1,2,3]
        bList = [3,3,3,3,3,3,1,1,1,1,2,2,2,2]
    
    elif K == 3:
        slist = [1,1,1]
        aList = [1,2,3]
        bList = [1,1,1]
    elif K == 6:
        slist = [1,1,1,1,1,1]
        aList = [1,2,3,1,2,3]
        bList = [1,1,1,2,2,2]
        ##incl a=0
    elif K == 4:
        slist = [1,1,1,1]
        aList = [0,1,2,3]
        bList = [1,1,1,1]
    elif K == 8:
        slist = [1,1,1,1,1,1,1,1]
        aList = [0,1,2,3,0,1,2,3]
        bList = [1,1,1,1,2,2,2,2]
    
    ##subset list
    subset = iniSubset(Nqb)
    
    ##expected cost func change
    costf=np.zeros(K**2)    
    
    ##loop over couplings
    for j in range(K**2):
        ##coupling
        sA = slist[j%K]
        sB = slist[int(j/K)]
        aA = aList[j%K]
        aB = aList[int(j/K)]
        bA = bList[j%K]
        bB = bList[int(j/K)]
        
        ##correlator
        Q = S[aA*4**nA+aB*4**nB]
        
        ##F tensor
        F = np.zeros(4**Nqb)
        if aA==aB and aA==0:
            F = S
        else:
            for l in range(4**Nqb):
                muA = int(l/(4**nA)%4)
                muB = int(l/(4**nB)%4)
                if muA == 0 and muB == 0:
                    F[l] = S[l+aA*4**nA+aB*4**nB]
                elif muA == 0 and muB == aB:
                    F[l] = S[l+aA*4**nA-aB*4**nB]
                elif muA == aA and muB == 0:
                    F[l] = S[l-aA*4**nA+aB*4**nB]
                elif muA == aA and muB == aB:
                    F[l] = S[l-aA*4**nA-aB*4**nB]
                
                elif muA != 0 and muA != aA and muB != 0 and muB != aB:# and aA!=0 and aB!=0:
                    rtm1 = 0
                    for k1 in range(1,4):
                        if k1 != aA and k1 != muA:
                            for k2 in range(1,4):
                                if k2 != aB and k2 != muB:
                                    rtm1 += LeviCivita(aA,muA,k1)*LeviCivita(aB,muB,k2)*S[l+(k1-muA)*4**nA+(k2-muB)*4**nB]
                    F[l] = rtm1
                    
                elif muA != 0 and muA != aA and muB != 0 and muB != aB and aA==0 and aB!=0:
                    rtm1 = 0
                    for k2 in range(1,4):
                        if k2 != aB and k2 != muB:
                            rtm1 += LeviCivita(aB,muB,k2)*S[l+(k2-muB)*4**nB]
                    F[l] = rtm1
                elif muA != 0 and muA != aA and muB != 0 and muB != aB and aA!=0 and aB==0:
                    rtm1 = 0
                    for k1 in range(1,4):
                        if k1 != aA and k1 != muA:
                            rtm1 += LeviCivita(aA,muA,k1)*S[l+(k1-muA)*4**nA]
                    F[l] = rtm1
             #   else:                
             #       F[l] = 0
                
        ##c_eta
        avcp=0
        if bA != 3:
            avcp += Gamma[nA]
        if bB != 3:
            avcp += Gamma[nB]
        avcm=avcp
        if bA == bB and bA != 3:
            rtm1 = 2*np.sqrt(Gamma[nA]*Gamma[nB])*Q
            avcp += rtm1
            avcm -= rtm1
        
        dR, dR2 = np.zeros(4**Nqb), np.zeros(4**Nqb)
        ##<<dR>> terms
        for l in range(4**Nqb):
            muA = int(l/(4**nA)%4)
            muB = int(l/(4**nB)%4)
            
            rtm1 = 0
            rtm2 = 0
            ##A terms
            if muA != aA and muA != 0:
                if bA !=3:
                    rtm1 -= Gamma[nA]*S[l]
                    rtm2 -= Gamma[nA]*S[l]
                else:
                    rtm4 = 0
                    for k in range(1,4):
                        if k != muA and k != aA:
                            rtm4 += LeviCivita(aA,k,muA)*S[l+(k-muA)*4**nA]
                    rtm1 += sA*J[nA]*rtm4
            
            ##B terms
            if muB != aB and muB !=0:
                if bB !=3:
                    rtm1 -= Gamma[nB]*S[l]
                    rtm2 -= Gamma[nB]*S[l]
                else:
                    rtm4 = 0
                    for k in range(1,4):
                        if k != muB and k != aB:
                            rtm4 += LeviCivita(aB,k,muB)*S[l+(k-muB)*4**nB]
                    rtm1 += sB*J[nB]*rtm4
                    
            dR[l] = 2*deltaT*rtm1
        
        ##<<dR^2>> terms
            #dR2[l] = 0
            if bA != 3 or bB != 3:
                rtm3 = 0
                if bA == bB:
                    rtm3 = np.sqrt(Gamma[nA]*Gamma[nB])*(F[l]-Q*S[l])
                #rtm3 = (bA==bB)*np.sqrt(Gamma[nA]*Gamma[nB])*(F[l]-Q*S[l])
                dR2[l] = deltaT*((rtm2+rtm3)**2/avcp+(rtm2-rtm3)**2/avcm)       
                ##implement J1=J2
               # if avcp == 0:
               #     dR2[l] = deltaT*(rtm2-rtm3)**2
               # elif avcm == 0:
               #     dR2[l] = deltaT*(rtm2+rtm3)**2
               # else:
               #     dR2[l] = deltaT*((rtm2+rtm3)**2/avcp+(rtm2-rtm3)**2/avcm)
        
        ##assemble
       # deltaf = np.zeros(Nqb-1)
       # for m in range(4**Nqb):
           # cost = (S[m]-Starg[m])*dR[m]+dR2[m]
           # for l in range(Nqb-1):
           #     for s in subset[l]:
           #        chk = 0
           #        for i in range(Nqb):
           #            if i not in s and int(m/(4**i)%4)>0:
           #                chk +=1
           #        if chk == 0:
           #            deltaf[l]+=cost
       # for l in range(Nqb-1):
           # costf[j] += p[l]*deltaf[l]/scipy.special.binom(Nqb,l+1)/2**(l+1)
           # costf[j] += sum([p[l]*deltaf[l]/scipy.special.binom(Nqb,l+1)/2**(l+1) for l in range(Nqb-1)])
           
        for l in range(Nqb-1):
            deltaf = 0       
            for s in subset[l]:
                for m in range(4**Nqb):
                    chk = 0
                    for i in range(Nqb):
                        if i not in s and int(m/(4**i)%4)>0:
                            chk +=1
                    #chk = sum(np.logical_and(np.logical_not(np.isin(range(Nqb),s)), (m/4**np.array(range(Nqb))).astype(int)%4>0))
                    if chk == 0:
                        deltaf+=(S[m]-Starg[m])*dR[m]+dR2[m]
            costf[j] += p[l]*deltaf/scipy.special.binom(Nqb,l+1)/2**(l+1)
        costf[j] += sum(-p[-1]*Starg*dR/2**Nqb)
        
    return costf

#produce subset list
def iniSubset(N):
    subset = [[[i] for i in range(N)]]
    
    if N>2:
        subset.append([[i,j] for j in range(N-1) for i in range(j+1,N)])
    if N>3:
        subset.append([[i,j,k] for j in range(N-2) for i in range(j+1,N-1) for k in range(i+1,N)])
    if N>4:
        subset.append([[i,j,k,l] for j in range(N-3) for i in range(j+1,N-2) for k in range(i+1,N-1)
                        for l in range(k+1,N)])
    if N>5:
        subset.append([[i,j,k,l,m] for j in range(N-4) for i in range(j+1,N-3) for k in range(i+1,N-2)
                        for l in range(k+1,N-1) for m in range(l+1,N)])
    if N>6:
        subset.append([[i,j,k,l,m,n] for j in range(N-5) for i in range(j+1,N-4) for k in range(i+1,N-3)
                        for l in range(k+1,N-2) for m in range(l+1,N-1) for n in range(m+1,N)])
    if N>7:
        subset.append([[i,j,k,l,m,n,o] for j in range(N-6) for i in range(j+1,N-5) for k in range(i+1,N-4)
                        for l in range(k+1,N-3) for m in range(l+1,N-2) for n in range(m+1,N-1) for o in range(n+1,N)])
        
    return subset

##Multicore compatibility in windows
if __name__ == '__main__':
    main()