#!/usr/bin/python3.9
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 28 14:40:09 2021

@author: Samuel Morales
TODO: save traj
"""
 
import numpy as np
import qutip as qt
import matplotlib.pyplot as plt

from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.ticker import (MultipleLocator)
import itertools
import random                                                 ##for fully connected chain
import scipy.special
from sympy import LeviCivita
from scipy import stats

##Pauli matrices
s0 = qt.qeye(2)
sx = qt.sigmax()
sy = qt.sigmay()
sz = qt.sigmaz()

s=[s0,sx,sy,sz]

##main program runs simulations with fixed parameters and plots results
def main():
    #random number generator
    rng = np.random.default_rng() 
    
    ##Parameters
    Nqb = 4                                             ##Number of qubits
    DeltaT = 0.2                                        ##time step length, default: 0.2
    
    ##coupling strength, default: [1, 0.99, 1.01, 1.005, 0.995, 1.003]
    #J = Nqb*[1]                                            ##equal weak coupling                                         
    J = [1, 0.99, 1.01, 1.005, 0.995, 1.003, 0.997, 1.007]  ##default values from original paper
    #J = Nqb*[np.pi/4/DeltaT]                               ##equal strong coupling
    #J = [np.pi/4/DeltaT-0.05+0.1*rng.random() for i in range(Nqb)] ##random offset strong coupling, -0.1(5) works best
    
    N = 130                                             ##nr of time steps
    M = 100                                             ##number of trajectories
    Nst = 1                                             ##every Nst´th step is saved
    
    ##cost fct probabilities
    plist = [9*10**(-i) for i in range(1,Nqb)]
    plist.append(1-sum(plist))
    
    ##couplings, target fidelity Fstar
    K = 9                                               ##nr of couplings: 3, 4, 6, 7, 8, 9, 10, 11, 12, 14
    Fstar = 0.975                                         ##target fidelity
    epsilon = 1-Fstar**2                                ##stop threshold for cost function
    params = [N, Nst, DeltaT, J[:Nqb], epsilon, K]
    
    ##Target state
    #psiTarg = qt.w_state(Nqb)          #W state
    psiTarg = qt.ghz_state(Nqb)         #GHZ state
    psiTarg = psiTarg.unit()
    
    ##Initial state
    psi0 = qt.qstate(Nqb*'d')               ##starting in 00...0 state
    psi0 = psi0.unit()
    #psi0 = qt.rand_ket(2**Nqb, dims=[[2]*Nqb,[1]*Nqb])             ##random state
    
    print('psiTarg={0}, psi0={1}'.format(psiTarg, psi0))
    print(r'[N, Nst, DeltaT, J, epsilon, K]={0}, M={1}, pList={2}, F*={3}'.format(params, M, plist, Fstar))    
    
    ##parallel running M trajectories
    results = qt.parallel_map(trajec, [Nqb]*M, task_kwargs=dict(psi0=psi0, psiTarg=psiTarg, param=params, pList=plist), progress_bar=True)
    if Nqb == 2:
        k_List, xi_eta_List, global_C_List, local_C_List, psi_List, success_Step, S_List = np.array(results, dtype=object).T
    else:
        k_List, xi_eta_List, global_C_List, local_C_List, psi_List, success_Step = np.array(results, dtype=object).T
    
    ##rearrange Distributions
    global_C_Distr = np.reshape(np.concatenate(global_C_List), (M, int(N/Nst)+1))
    local_C_Distr = np.reshape(np.concatenate(local_C_List), (M, int(N/Nst)+1))
    psi_Distr = np.reshape(np.concatenate(psi_List), (M, int(N/Nst)+1))
    if Nqb ==2:
        S_Distr = np.reshape(np.concatenate(S_List), (M, int(N/Nst)+1))
    
    ##save data to files
    np.savetxt('coupl_list', np.concatenate(k_List))
    np.savetxt('success_steps', success_Step)
    np.savetxt('xi_eta_List', np.concatenate(xi_eta_List), fmt='%s')
    np.savetxt('psi_List', np.concatenate(psi_List), fmt='%s')
    
    ##choose trajectories for plots
    traj = [7,25,33]
    
############################################################Plots
   
    ##coupling histogram
    fig, ax = plt.subplots(figsize=(8,6))
    plt.hist(np.concatenate(k_List), bins=np.arange(0,K**2+1), alpha=0.5, edgecolor='grey')
    
    plt.xlim(left=0, right=K**2)
    plt.minorticks_on()
    
    plt.xlabel(r'$K$', fontsize=25)
    plt.ylabel(r'Number of trajectories', fontsize=25)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    
    plt.tight_layout()    
    plt.savefig('coupl hist.pdf')
    plt.savefig('coupl hist.svg')
    
    ##purity of average
    fig, ax = plt.subplots(figsize=(8,6))
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.ylim(bottom=0, top=1.01)
    plt.xlim(left=0, right=N)
    avg_Pur = [(sum([qt.ket2dm(psi) for psi in psi_Distr[:,i]])**2).tr()/M**2 for i in range(int(N/Nst)+1)]
    np.savetxt('avgPur', avg_Pur)
    
    ax.plot(np.arange(0,N+1,Nst), avg_Pur, 'k', label=r'average Purity', linewidth=3)
    ax.plot(np.arange(0,N+1,Nst), np.ones(int(N/Nst)+1)*(1/2**Nqb), 'k:', label=r'minimum', linewidth=2)
    
    ax.set_xlabel(r'$n_t$',fontsize=25)
    ax.set_ylabel(r'$\mathrm{{tr}}(\overline{{\rho}}^2)$',fontsize=25)
    ax.minorticks_on()
    ax.tick_params(length=8)
    ax.tick_params(which='minor', length=4)
    ax.locator_params(axis='x', nbins=4)
    ax.locator_params(axis='y', nbins=6)
    ax.xaxis.set_minor_locator(MultipleLocator(N/20))
    
    plt.tight_layout()
    plt.savefig('avg pur.pdf', format='pdf')
    plt.savefig('avg pur.svg', format='svg')
    
    ##Average fidelity/global cost function plot
    fig, ax = plt.subplots(figsize=(8,6))
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.ylim(bottom=0, top=1.01)
    plt.xlim(left=0, right=N)
    avg_global_C = np.mean(global_C_Distr, axis=0)
    np.savetxt('avg_global_cost', avg_global_C)
    ax.plot(np.arange(0,N+1,Nst), avg_global_C, 'k', label=r'average C', linewidth=3)
    ax.plot(np.arange(0,N+1,Nst), global_C_Distr[traj[0]], '--', label=r'single traj 1', linewidth=2, alpha=0.7)
    ax.plot(np.arange(0,N+1,Nst), global_C_Distr[traj[1]], '--', label=r'single traj 2', linewidth=2, alpha=0.7)
    ax.plot(np.arange(0,N+1,Nst), global_C_Distr[traj[2]], '--', label=r'single traj 3', linewidth=2, alpha=0.7)
    ax.plot(np.arange(0,N+1,Nst), np.ones(int(N/Nst)+1)*(1-Fstar**2), 'k:', label=r'target cost', linewidth=2)
    ax.set_xlabel(r'$n_t$',fontsize=25)
    ax.set_ylabel(r'$C_{0}(t)$'.format(Nqb),fontsize=25)
    ax.minorticks_on()
    ax.tick_params(length=8)
    ax.tick_params(which='minor', length=4)
    ax.locator_params(axis='x', nbins=4)
    ax.locator_params(axis='y', nbins=6)
    ax.xaxis.set_minor_locator(MultipleLocator(N/20))
    plt.tight_layout()
    
        #inset with local cost function
    axins = inset_axes(ax, width='45%', height='45%', borderpad=0.5)
    plt.xlim(left=0, right=N)
    plt.ylim(bottom=0, top=0.3)
    axins.set_xlabel(r'$n_t$',fontsize=20)
    axins.xaxis.set_label_coords(0.5,-0.09)
    axins.minorticks_on()
    axins.xaxis.set_minor_locator(MultipleLocator(N/20))
    axins.yaxis.set_minor_locator(MultipleLocator(0.01))
    axins.tick_params(length=4)            
    axins.tick_params(which='minor', length=3)
    axins.locator_params(axis='x', nbins=4)
    axins.locator_params(axis='y', nbins=3)
    
    avg_local_C = np.mean(local_C_Distr, axis=0)
    np.savetxt('avg_local_cost', avg_local_C)
    axins.plot(np.arange(0,N+1,Nst), avg_local_C, 'k', label=r'average Cloc', linewidth=2)
    axins.plot(np.arange(0,N+1,Nst), local_C_Distr[traj[0]], '--', label=r'single traj 1', linewidth=1.5, alpha=0.7)
    axins.plot(np.arange(0,N+1,Nst), local_C_Distr[traj[1]], '--', label=r'single traj 2', linewidth=1.5, alpha=0.7)
    axins.plot(np.arange(0,N+1,Nst), local_C_Distr[traj[2]], '--', label=r'single traj 3', linewidth=1.5, alpha=0.7)
    
    plt.savefig('avg fid.pdf', format='pdf')
    plt.savefig('avg fid.svg', format='svg')

    ##Success statistics plot
    print(np.count_nonzero(success_Step==N+1))
    print(np.count_nonzero(success_Step==N+2))
    success_Step = np.delete(success_Step, np.where(success_Step==N+1))            ##delete unconverged trajectories from step analysis
    success_Step = np.delete(success_Step, np.where(success_Step==N+2))            ##delete unconverged trajectories from step analysis
    
    if success_Step.any():
        meanN = np.mean(success_Step)
        stdN = np.std(success_Step)
        modeN = stats.mode(success_Step)
    else:
        meanN, stdN, modeN = 0, 0, 0
    
    print(r'$N_{{suc}}=${0}, $N^{{med}}_{{suc}}=${1}, $STDN^{{med}}_{{suc}}=${2}, $ModeN^{{med}}_{{suc}}=${3}'
          .format(meanN, np.median(success_Step), stdN, modeN))
    
    success_Step -=1         ##for correct representation because bins are defined as [n,n+1)
    
    fig, ax = plt.subplots(figsize=(8,6))
    plt.hist(success_Step, bins=np.arange(0,N+1,2*Nst), alpha=0.5, edgecolor='grey')
    plt.axvline(meanN, color='k', ls='--', linewidth=2, label=r'$\overline{{t}}$')
    plt.annotate(text='', xy=(meanN+stdN,M/300), xytext=(meanN-stdN,M/300), arrowprops=dict(arrowstyle='|-|'))
    
    plt.xlim(left=0, right=N)
    plt.minorticks_on()
    plt.xlabel(r'$n_t$', fontsize=25)
    plt.ylabel(r'Number of trajectories', fontsize=25)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.yscale('log')
    ax.tick_params(length=8)            
    ax.tick_params(which='minor', length=4)
    plt.locator_params(axis='x', nbins=4)
    ax.xaxis.set_minor_locator(MultipleLocator(N/20))
    
    plt.tight_layout()    
    plt.savefig('success step.pdf', format='pdf')
    plt.savefig('success step.svg', format='svg') 

    ##Average entanglement plot for 2 qubits
    if Nqb==2: 
        fig, ax = plt.subplots(figsize=(8,6))
        avg_S = np.mean(S_Distr, axis=0)
        np.savetxt('avg_S', avg_S)
        ax.plot(np.arange(0,N+1,Nst), avg_S, 'k', linewidth=3, label=r'average S')
        ax.plot(np.arange(0,N+1,Nst), S_Distr[traj[0]], '--', label=r'single traj 1', linewidth=2, alpha=0.7)
        ax.plot(np.arange(0,N+1,Nst), S_Distr[traj[1]], '--', label=r'single traj 2', linewidth=2, alpha=0.7)
        ax.plot(np.arange(0,N+1,Nst), S_Distr[traj[2]], '--', label=r'single traj 3', linewidth=2, alpha=0.7)
        ax.plot(np.arange(0,N+1,Nst), np.ones(int(N/Nst)+1)*np.log(2), 'k:', label=r'maximally entangled')
        ax.set_xlabel(r'$n_t$',fontsize=25)
        ax.set_ylabel('$S(t)$',fontsize=25)
        plt.xlim(0,70)
        plt.ylim(0,0.73)
        plt.minorticks_on()
        ax.tick_params(length=8)
        ax.tick_params(which='minor', length=4)
        ax.locator_params(axis='x', nbins=4)
        ax.locator_params(axis='y', nbins=4)
        ax.xaxis.set_minor_locator(MultipleLocator(5))
        ax.yaxis.set_minor_locator(MultipleLocator(0.05))
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        
        plt.tight_layout()
        plt.savefig('avg EE.pdf', format='pdf')
        plt.savefig('avg EE.svg', format='svg')
       
####################################################################################    
##Trajectory simulator
def trajec(Nqb, psi0, psiTarg, param, pList):
    N, Nst, DeltaT, J, eps, K = param       ##unpack parameters
    success_Step = N+1                      ##success_Step value for unsuccessful steering
    Gamma = [j**2*DeltaT for j in J]        ##jump rate, default: J*J*DeltaT
    
    #random number generator
    rng = np.random.default_rng() 
    
    ##Track chosen coupling, global cost, local cost, measurement outcomes, state
    k_List = np.zeros(N*int(Nqb/2))
    global_C = np.zeros(int(N/Nst)+1)
    local_C = np.zeros(int(N/Nst)+1)
    xi_eta_List = np.zeros(N, dtype=tuple)
    psi_List = np.zeros(int(N/Nst+1), dtype=object)
    
    ##entanglement entropy for 2 qubits:
    if Nqb == 2:
        S_list = np.zeros(int(N/Nst)+1)
        S_list[0] = qt.entropy_vn(psi0.ptrace((0)))
    
    ##coupling lists
    slist, aList, bList = coupl_list(K)
        
    ##operator list
    pauliPlaq_tensor = plaqS_tensor(Nqb)
    
    ##initial state
    psi_List[0] = psi0
    psi = psi0
    
    ##initialize Pauli tensors
    Spsi = np.reshape(qt.expect(pauliPlaq_tensor.flatten(),psi0), [4]*Nqb)
    Starg = np.reshape(qt.expect(pauliPlaq_tensor.flatten(),psiTarg), [4]*Nqb)
    
    ##subset list
    subset = iniSubset(Nqb)
    #subset = iniLocalSubset(Nqb)
    #subset = iniLocalSubsetPeriodic(Nqb)
    
    ##Initial cost function values
    global_C[0] = np.sum((Spsi-Starg)**2)/2**(Nqb+1)
    Fid = global_C[0]
    
    cost = csFct_tensor(Spsi, Starg, Nqb, subset)
    local_C[0] = sum([pList[j]*cost[j] for j in range(Nqb)])
    
    #qbList = np.arange(0,Nqb)                  ##needed for fully coupled chain
    
    ##Time step loop
    for i in range(1, N+1):
        ##starting steering pair
        nStart1 = int(N*rng.random())
        nStart2 = (nStart1+1)%Nqb
        
        #random.shuffle(qbList)                 ##needed for fully coupled chain
        
        ##steering coupled qubit pairs
        for nPair in range(int(Nqb/2)):
            ##Nearest neighbor coupling
            n1 = (nStart1+2*nPair)%Nqb
            n2 = (nStart2+2*nPair)%Nqb
            
            ##for fully coupled chain
            #n1 = int(qbList[2*nPair])          
            #n2 = int(qbList[2*nPair+1])
            
            ##closed boundary
            #if n1>n2:
            #    #break
            
            ##Coupling strengths
            J1, J2 = J[n1], J[n2]
            G1, G2 = Gamma[n1], Gamma[n2]
            
            ##Active decision making
            deltaC = expCostF_tensorized(Spsi, Starg, J, Gamma, DeltaT, pList, n1, n2, Nqb, K, subset)
            #deltaC = expCostF_tensorized_exact(Spsi, Starg, J, DeltaT, pList, n1, n2, Nqb, K, subset)
        
            k1, k2 = rng.choice(np.argwhere(deltaC == np.nanmin(deltaC)))
            k_List[(i-1)*int(Nqb/2)+nPair] = k1+K*k2
            
            ##chosen couplings
            s1 = slist[k1]
            s2 = slist[k2]
            alpha1 = aList[k1]
            alpha2 = aList[k2]
            beta1 = bList[k1]
            beta2 = bList[k2]
            
            ##non-improvement criterion
            dcost = 0.001
            if np.nanmin(deltaC) > dcost:
                success_Step = N+2
                break
            
            ##chosen Pauli operators
            sig1index = [0]*Nqb
            sig1index[n1] = alpha1
            sig1 = pauliPlaq_tensor[tuple(sig1index)]
            sig2index = [0]*Nqb
            sig2index[n2] = alpha2
            sig2 = pauliPlaq_tensor[tuple(sig2index)]
            
            ##Time step
            #beta1=z or beta2=z or (beta1=x, beta2=y) or (beta1=y, beta2=x)
            if beta1==3 or beta2==3 or beta1!=beta2:
                eta = (-1)**int(2*rng.random())
                H = (beta1==3)*s1*J1*sig1+(beta2==3)*s2*J2*sig2+(beta1!=3)*(beta2!=3)*eta*np.sqrt(G1*G2)*sig1*sig2
                c_op = (beta1!=3)*np.sqrt(G1)*sig1+(1-2*(beta1==2))*eta*1j*(beta2!=3)*np.sqrt(G2)*sig2
                psi, xi = unitsol(psi, DeltaT, H, c_op, ((beta1!=3)*G1+(beta2!=3)*G2)*DeltaT)
                xi_eta_List[i-1] = (xi, eta)
                
            #beta1=beta2=x/y  
            elif beta1==beta2:
                c_op = [np.sqrt(G1/2)*sig1+np.sqrt(G2/2)*sig2, np.sqrt(G1/2)*sig1-np.sqrt(G2/2)*sig2]
                psi, xi_eta_List[i-1] = ent_swap_sol(psi, DeltaT, c_op)
            
            ##Kraus operator construction
            #if beta1==3:
            #    A0 = (np.cos(J1*DeltaT)-1j*s1*np.sin(J1*DeltaT)*sig1)*(np.cos(J2*DeltaT)-(beta2==3)*1j*s2*np.sin(J2*DeltaT)*sig2)##Identity?no
            #    A1 = (np.cos(J1*DeltaT)-1j*s1*np.sin(J1*DeltaT)*sig1)*(beta2!=3)*1j*s2*np.sin(J2*DeltaT)*sig2
            #    P1 = (beta2!=3)*np.sin(J2*DeltaT)**2
            #    P0 = 1-P1
            #    Plist = [P0, P1]
            #    Alist = [A0, A1]
            #    psi, xi = krausSol(psi, Alist, Plist)
            #    xi_eta_List[i-1] = (xi, (-1)**int(2*rng.random()))
            #elif beta2==3:
            #    A0 = (np.cos(J2*DeltaT)-1j*s2*np.sin(J2*DeltaT)*sig2)*(np.cos(J1*DeltaT)-(beta1==3)*1j*s1*np.sin(J1*DeltaT)*sig1)
            #    A1 = (np.cos(J2*DeltaT)-1j*s2*np.sin(J2*DeltaT)*sig2)*(beta1!=3)*1j*s1*np.sin(J1*DeltaT)*sig1
            #    P1 = (beta1!=3)*np.sin(J1*DeltaT)**2
            #    P0 = 1-P1
            #    
            #    Plist = [P0, P1]
            #    Alist = [A0, A1]
            #    psi, xi = krausSol(psi, Alist, Plist)
            #    xi_eta_List[i-1] = (xi, (-1)**int(2*rng.random()))
            #else:
            #    A0p = (np.cos(J1*DeltaT)*np.cos(J2*DeltaT)
            #           -s1*s2*np.sin(J1*DeltaT)*np.sin(J2*DeltaT)*((beta1==1)+1j*(beta1==2))*((beta2==1)+1j*(beta2==2))*sig1*sig2)/np.sqrt(2)
            #    A0m = (np.cos(J1*DeltaT)*np.cos(J2*DeltaT)
            #           +s1*s2*np.sin(J1*DeltaT)*np.sin(J2*DeltaT)*((beta1==1)+1j*(beta1==2))*((beta2==1)+1j*(beta2==2))*sig1*sig2)/np.sqrt(2)
            #    A1p = (np.sin(J1*DeltaT)*np.cos(J2*DeltaT)*sig1+((beta1==beta2)+1j*(beta1!=beta2))*np.cos(J1*DeltaT)*np.sin(J2*DeltaT)*sig2)/np.sqrt(2)
            #    A1m = (np.sin(J1*DeltaT)*np.cos(J2*DeltaT)*sig1-((beta1==beta2)+1j*(beta1!=beta2))*np.cos(J1*DeltaT)*np.sin(J2*DeltaT)*sig2)/np.sqrt(2)
            #    
            #    Q = qt.expect(sig1*sig2, psi)
            #    P0p = (1-np.sin(J1*DeltaT)**2*np.cos(J2*DeltaT)**2-np.cos(J1*DeltaT)**2*np.sin(J2*DeltaT)**2
            #          +(beta1==beta2)*(-1)**beta1*0.5*s1*s2*np.sin(2*J1*DeltaT)*np.sin(2*J2*DeltaT)*Q)/2
            #    P0m = (1-np.sin(J1*DeltaT)**2*np.cos(J2*DeltaT)**2-np.cos(J1*DeltaT)**2*np.sin(J2*DeltaT)**2
            #         -(beta1==beta2)*(-1)**beta1*0.5*s1*s2*np.sin(2*J1*DeltaT)*np.sin(2*J2*DeltaT)*Q)/2
            #    P1p = (np.sin(J1*DeltaT)**2*np.cos(J2*DeltaT)**2+np.cos(J1*DeltaT)**2*np.sin(J2*DeltaT)**2
            #          +(beta1==beta2)*0.5*s1*s2*np.sin(2*J1*DeltaT)*np.sin(2*J2*DeltaT)*Q)/2
            #    P1m = (np.sin(J1*DeltaT)**2*np.cos(J2*DeltaT)**2+np.cos(J1*DeltaT)**2*np.sin(J2*DeltaT)**2
            #          -(beta1==beta2)*0.5*s1*s2*np.sin(2*J1*DeltaT)*np.sin(2*J2*DeltaT)*Q)/2
            #    
            #    Plist = [P0p, P0m, P1p, P1m]
            #    Alist = [A0p, A0m, A1p, A1m]
            #    psi, xi = krausSol(psi, Alist, Plist)
            #    xi_eta_List[i-1] = (int(xi/2), (-1)**(xi%2))
                
        ##Update values     
        Spsi = np.reshape(qt.expect(pauliPlaq_tensor.flatten(),psi), [4]*Nqb)
        Fid = np.sum((Spsi-Starg)**2)/2**(Nqb+1)
        
        ##stoppage criterion at global cost value eps, corresponding to F=F*
        if Fid < eps:
            ##final values
            psi_List[int((i-1)/Nst)+1:] = (int(N/Nst)-int((i-1)/Nst))*[1]*psi
            global_C[int((i-1)/Nst)+1:] = Fid
            cost = csFct_tensor(Spsi, Starg, Nqb, subset)
            local_C[int((i-1)/Nst)+1:] = sum([pList[j]*cost[j] for j in range(Nqb)])
            
            ##entanglement entropy for 2 qubits:
            if Nqb == 2:
                S_list[int((i-1)/Nst)+1:] = qt.entropy_vn(psi.ptrace((0)))
    
            k_List[i*int(Nqb/2):] = -1
            xi_eta_List[i:] = [(np.nan, np.nan)]*(N-i)
            success_Step = i
            break
        
        ##saving data every Nst step
        if i%Nst == 0:
            psi_List[int(i/Nst)] = psi
            global_C[int(i/Nst)] = Fid
            cost = csFct_tensor(Spsi, Starg, Nqb, subset)
            local_C[int(i/Nst)] = sum([pList[j]*cost[j] for j in range(Nqb)])
            
            ##entanglement entropy for 2 qubits:
            if Nqb == 2:
                S_list[int(i/Nst)] = qt.entropy_vn(psi.ptrace((0)))
    
    if Nqb == 2:
        return k_List, xi_eta_List, global_C, local_C, psi_List, success_Step, S_list
    else:
        return k_List, xi_eta_List, global_C, local_C, psi_List, success_Step

#produce subset lists
def iniSubset(N):
    subset = [list(itertools.combinations(range(N),i)) for i in range(1,N)]
    return subset

def iniLocalSubset(N):  ##open boundary
    subset = [[range(j,j+i) for j in range(N-i+1)] for i in range(1,N)]
    return subset

def iniLocalSubsetPeriodic(N):  ##periodic boundary
    subset = [[np.arange(j,j+i)%N for j in range(N)] for i in range(1,N)]
    return subset

##coupling list
def coupl_list(K):
    if K == 3:          ##beta=x; alpha=x,y,z; s=+
        slist = [1,1,1]
        aList = [1,2,3]
        bList = [1,1,1]
    elif K == 4:        ##beta=x; alpha=0,x,y,z; s=+
        slist = [1,1,1,1]
        aList = [0,1,2,3]
        bList = [1,1,1,1]
    elif K == 6:        ##beta=x,y; alpha=x,y,z; s=+
        slist = [1,1,1,1,1,1]
        aList = [1,2,3,1,2,3]
        bList = [1,1,1,2,2,2]
        
        ##beta=x,z; alpha=x,y,z; s=+
        #slist = [1,1,1,1,1,1]
        #aList = [1,2,3,1,2,3]
        #bList = [1,1,1,3,3,3]
    elif K == 7:        ##beta=x,z; alpha=0,x,y,z; s=+
        slist = [1,1,1,1,1,1,1]
        aList = [0,1,2,3,1,2,3]
        bList = [1,1,1,1,3,3,3]
    elif K == 8:        ##beta=x,y; alpha=0,x,y,z; s=+
        slist = [1,1,1,1,1,1,1,1]
        aList = [0,1,2,3,0,1,2,3]
        bList = [1,1,1,1,2,2,2,2]
    elif K == 9:        ##beta=x,z; alpha=x,y,z; s=+/-
        slist = [1,1,1,-1,-1,-1,1,1,1]
        aList = [1,2,3,1,2,3,1,2,3]
        bList = [3,3,3,3,3,3,1,1,1]
        
        ##beta=x,y,z; alpha=x,y,z; s=+
        #slist = [1,1,1,1,1,1,1,1,1]
        #aList = [1,2,3,1,2,3,1,2,3]
        #bList = [1,1,1,2,2,2,3,3,3]
    elif K == 10:        ##beta=x,z; alpha=0,x,y,z; s=+/-
        slist = [1,1,1,-1,-1,-1,1,1,1,1]
        aList = [1,2,3,1,2,3,0,1,2,3]
        bList = [3,3,3,3,3,3,1,1,1,1]
    elif K == 11:        ##beta=x,y,z; alpha=0,x,y,z; s=+
        slist = [1,1,1,1,1,1,1,1,1,1,1]
        aList = [1,2,3,0,1,2,3,0,1,2,3]
        bList = [3,3,3,1,1,1,1,2,2,2,2]
    elif K == 12:        ##beta=x,y,z; alpha=x,y,z; s=+/-
        slist = [1,1,1,-1,-1,-1,1,1,1,1,1,1]
        aList = [1,2,3,1,2,3,1,2,3,1,2,3]
        bList = [3,3,3,3,3,3,1,1,1,2,2,2]
    elif K == 14:        ##beta=x,y,z; alpha=0,x,y,z; s=+/-
        slist = [1,1,1,-1,-1,-1,1,1,1,1,1,1,1,1]
        aList = [1,2,3,1,2,3,0,1,2,3,0,1,2,3]
        bList = [3,3,3,3,3,3,1,1,1,1,2,2,2,2]
    
    return slist, aList, bList

##Time evolution solver  
##Entanglement swapping for beta1=beta2=x/y
def ent_swap_sol(psi, deltaT, c_op):
    '''return normalized state conditioned on measurement outcome
    Bell state measurement
    '''
    #random number generator
    rng = np.random.default_rng() 
    #probabilities
    P = np.zeros(4)
    P[0:2] = qt.expect([c.dag()*c for c in c_op], psi)*deltaT
    P[2:4] = (1-2*P[0:2])/2
    #stochastic measurement outcome
    while 1==1:
        k = int(4*rng.random())
        if rng.random() < P[k]:
            break
    ##time evolution
    if k < 2:
        return (c_op[k]*psi).unit(), (1,(-1)**k)
    else:
        return ((1-deltaT*c_op[k-2].dag()*c_op[k-2])*psi).unit(), (0,(-1)**k)
      
##Unitary dynamics for (beta1,beta2)=(x,y) or (x/y,z) and vice-versa
def unitsol(psi, deltaT, H, c_op, P):
    '''State evolution for single, unitary jump operator
    Parameters
    ----------

    H: :class:`qutip.Qobj`
        Hamiltonian

    psi: :class:`qutip.Qobj`
        initial state vector (ket)
    
    c_op: :class:`qutip.Qobj`
        unitary jump operator
        
    deltaT: (float)
        time step length
        
    Returns
    -------

    result: :class:`qutip.Qobj`

        Normalized time-evolved state (ket)
    '''
    #random number generator
    rng = np.random.default_rng()
    if rng.random() >= P:
        return ((1-1j*deltaT*H-P/2)*psi).unit(), 0
    else:
        return (c_op*psi).unit(), 1

##Exact Kraus operator evolution
def krausSol(psi, A, P):
    '''State evolution for multiple Kraus operators
    Parameters
    ----------

    psi: :class:`qutip.Qobj`
        initial state (ket or oper)
    
    A: List of :class:`qutip.Qobj`
        Kraus operators
        
    P: List of (float)
        Probabilities of corresponding Kraus operators
        
    Returns
    -------

    result: :class:`qutip.Qobj`

        Normalized time-evolved state (ket or oper)
    '''
    #random number generator
    rng = np.random.default_rng() 
    #stochastic measurement outcome
    while 1==1:
        k = int(len(P)*rng.random())
        if rng.random() < P[k]:
            break
    ##time evolution
    if psi.type == 'ket':
        return (A[k]*psi).unit(), k
    elif psi.type == 'oper':
        return (A[k]*psi*A[k].dag()).unit(), k

###plaquette operators
def plaqS(ind):
    return qt.tensor([s[i] for i in ind])
    
def plaqS_tensor(Nqb):
    S_tensor=np.zeros([4]*Nqb, dtype=object)
    for index in np.ndindex(*([4]*Nqb)):
        S_tensor[index] = plaqS(index)
    return  S_tensor

##cost function for all levels 1 - Nqb
def csFct_tensor(S, Starg, Nqb, subsets):
    costf = np.zeros(Nqb)
    for i in range(Nqb-1):
       costf[i] = costFunc_tensor(S, Starg, i+1, Nqb, subsets[i])
    costf[Nqb-1] = np.sum((S-Starg)**2)/2**(Nqb+1)
    return costf

##cost function of level i
def costFunc_tensor(S, Starg, i, Nqb, subset):
    costf = 0
    C = (S-Starg)**2
    for s in subset:
        ind = [slice(0,1+3*i) for i in np.isin(range(Nqb),s)]
        costf+=np.sum(C[tuple(ind)])
    
    costf *= 1/2**(i+1)/scipy.special.binom(Nqb,i)
    return costf

##expected cost function change
def expCostF_tensorized(S, Starg, J, Gamma, deltaT, p, nA, nB, Nqb, K, subset):
    JA = J[nA]
    JB = J[nB]
    GA = Gamma[nA]
    GB = Gamma[nB]
    
    ##coupling lists
    slist, aList, bList = coupl_list(K)
    
    ##expected cost func change
    costf=np.zeros([K]*2)    
    
    ##loop over couplings
    for coupl in np.ndindex(*([K]*2)):
        ##coupling
        sA = slist[coupl[0]]
        sB = slist[coupl[1]]
        aA = aList[coupl[0]]
        aB = aList[coupl[1]]
        bA = bList[coupl[0]]
        bB = bList[coupl[1]]
        
        ##different couplings with same dC for bA=y
        if bA == 2:
            ##bA,bB = y,x:
            if bB == 1:
                for couplOld in np.ndindex(*([K]*2)):
                    if bList[couplOld[0]] == 1 and bList[couplOld[1]] == 2 and aA == aList[couplOld[0]] and aB == aList[couplOld[1]]:
                        costf[coupl] = costf[couplOld]
                        break
                continue
            ##bA,bB = y,y:
            if bB == 2:
                for couplOld in np.ndindex(*([K]*2)):
                    if bList[couplOld[0]] == 1 and bList[couplOld[1]] == 1 and aA == aList[couplOld[0]] and aB == aList[couplOld[1]]:
                        costf[coupl] = costf[couplOld]
                        break
                continue
            ##bA,bB = y,z:
            if bB == 3:
                for couplOld in np.ndindex(*([K]*2)):
                    if bList[couplOld[0]] == 1 and bList[couplOld[1]] == 3 and aA == aList[couplOld[0]] and aB == aList[couplOld[1]]:
                        costf[coupl] = costf[couplOld]
                        break
                continue
        ##bA,bB = z,y:
        elif bA == 3 and bB == 2:
            for couplOld in np.ndindex(*([K]*2)):
                if bList[couplOld[0]] == 3 and bList[couplOld[1]] == 1 and aA == aList[couplOld[0]] and aB == aList[couplOld[1]]:
                    costf[coupl] = costf[couplOld]
                    break
            continue
        
        dC = dC_tensorized(S, Starg, deltaT, (sA, JA, GA), (sB, JB, GB), (aA, bA, nA), (aB, bB, nB), Nqb)
        
        ##assemble
        for l in range(Nqb-1):
            if p[l] == 0:
                continue
            deltaf = 0
            for s in subset[l]:
                ind = [slice(0,1+3*i) for i in np.isin(range(Nqb),s)]
                deltaf += np.sum(dC[tuple(ind)])
            costf[coupl] += p[l]*deltaf/scipy.special.binom(Nqb,l+1)/2**(l+1)
            
        costf[coupl] += p[-1]*np.sum(dC)/2**Nqb
        
    return costf

def expCostF_tensorized_exact(S, Starg, J, deltaT, p, nA, nB, Nqb, K, subset):
    JA = J[nA]
    JB = J[nB]
    
    ##coupling lists
    slist, aList, bList = coupl_list(K)
    
    ##expected cost func change
    costf=np.zeros([K]*2)    
    
    ##loop over couplings
    for coupl in np.ndindex(*([K]*2)):
        ##coupling
        sA = slist[coupl[0]]
        sB = slist[coupl[1]]
        aA = aList[coupl[0]]
        aB = aList[coupl[1]]
        bA = bList[coupl[0]]
        bB = bList[coupl[1]]
        
        ##different couplings with same dC for bA=y
        if bA == 2:
            ##bA,bB = y,x:
            if bB == 1:
                for couplOld in np.ndindex(*([K]*2)):
                    if bList[couplOld[0]] == 1 and bList[couplOld[1]] == 2 and aA == aList[couplOld[0]] and aB == aList[couplOld[1]]:
                        costf[coupl] = costf[couplOld]
                        break
                continue
            ##bA,bB = y,y:
            if bB == 2:
                for couplOld in np.ndindex(*([K]*2)):
                    if bList[couplOld[0]] == 1 and bList[couplOld[1]] == 1 and aA == aList[couplOld[0]] and aB == aList[couplOld[1]]:
                        costf[coupl] = costf[couplOld]
                        break
                continue
            ##bA,bB = y,z:
            if bB == 3:
                for couplOld in np.ndindex(*([K]*2)):
                    if bList[couplOld[0]] == 1 and bList[couplOld[1]] == 3 and aA == aList[couplOld[0]] and aB == aList[couplOld[1]]:
                        costf[coupl] = costf[couplOld]
                        break
                continue
        ##bA,bB = z,y:
        elif bA == 3 and bB == 2:
            for couplOld in np.ndindex(*([K]*2)):
                if bList[couplOld[0]] == 3 and bList[couplOld[1]] == 1 and aA == aList[couplOld[0]] and aB == aList[couplOld[1]]:
                    costf[coupl] = costf[couplOld]
                    break
            continue
        
        dC = dC_tensorized_exact(S, Starg, deltaT, (sA, JA), (sB, JB), (aA, bA, nA), (aB, bB, nB), Nqb)
        
        ##assemble
        for l in range(Nqb-1):
            if p[l] == 0:
                continue
            deltaf = 0
            for s in subset[l]:
                ind = [slice(0,1+3*i) for i in np.isin(range(Nqb),s)]
                deltaf += np.sum(dC[tuple(ind)])
            costf[coupl] += p[l]*deltaf/scipy.special.binom(Nqb,l+1)/2**(l+1)
            
        costf[coupl] += p[-1]*np.sum(dC)/2**Nqb
        
    return costf

##dC tensor:
def dC_tensorized(S, Starg, deltaT, A, B, kA, kB, Nqb):
    sA, JA, GA = A
    sB, JB, GB = B
    
    aA, bA, nA = kA
    aB, bB, nB = kB
    
    Qindex = [0]*Nqb
    Qindex[nA] = aA
    Qindex[nB] = aB
    Q = S[tuple(Qindex)]
    
    ##F tensor
    F = F_tensorized(S, Starg, (aA, bA, nA), (aB, bB, nB), Nqb)
    
    ##<c_eta^\dagger c_eta>
    avcp = (bA != 3)*GA+(bB != 3)*GB
    rtm1 = (bA == bB)*(bA != 3)*2*np.sqrt(GA*GB)*Q
    avcm = avcp-rtm1
    avcp += rtm1
    
    dC = np.zeros([4]*Nqb)
    
    dR = np.zeros([4]*Nqb)
    dR2 = np.zeros([4]*Nqb)
    rtm2 = np.zeros([4]*Nqb)
    
    indA = [slice(None)]*Nqb
    indB = [slice(None)]*Nqb
    for i in range(1,4):
        indA[nA] = i
        ##A terms
        if i != aA and aA!=0:
            if bA !=3:
                dR[tuple(indA)] -= 2*deltaT*GA*S[tuple(indA)]
                rtm2[tuple(indA)] -= GA*S[tuple(indA)]
            else:
                for k in range(1,4):
                    if k != i and k != aA:
                        Sindex = [slice(None)]*Nqb
                        Sindex[nA] = k
                        dR[tuple(indA)] += 2*deltaT*sA*JA*int(LeviCivita(aA,k,i))*S[tuple(Sindex)]
        indB[nB] = i
        ##B terms
        if i != aB and aB!=0:
            if bB !=3:
                dR[tuple(indB)] -= 2*deltaT*GB*S[tuple(indB)]
                rtm2[tuple(indB)] -= GB*S[tuple(indB)]
            else:
                for k in range(1,4):
                    if k != i and k != aB:
                        Sindex = [slice(None)]*Nqb
                        Sindex[nB] = k
                        dR[tuple(indB)] += 2*deltaT*sB*JB*int(LeviCivita(aB,k,i))*S[tuple(Sindex)]
    
    rtm3 = (bA == bB)*np.sqrt(GA*GB)*(F-Q*S)
    if avcp == 0 and avcm !=0:
        dR2 = deltaT*(rtm2-rtm3)**2/avcm
    elif avcm == 0 and avcp !=0:
        dR2 = deltaT*(rtm2+rtm3)**2/avcp
    elif avcp !=0 and avcm !=0:
        dR2 = deltaT*((rtm2+rtm3)**2/avcp+(rtm2-rtm3)**2/avcm)
    
    if bA != 3 and bB != 3:
        dC = (S-Starg)*dR+dR2
    else:
        dC = -Starg*dR
        
    return dC

##dC tensor:
def dC_tensorized_exact(S, Starg, deltaT, A, B, kA, kB, Nqb):
    sA, JA = A
    sB, JB = B
    
    aA, bA, nA = kA
    aB, bB, nB = kB
    
    Qindex = [0]*Nqb
    Qindex[nA] = aA
    Qindex[nB] = aB
    Q = S[tuple(Qindex)]
    
    ##F tensor
    F = F_tensorized(S, Starg, (aA, bA, nA), (aB, bB, nB), Nqb)
    
    ##H tensor
    H = F
    
    if aA!=0 and aB!=0:
        ind = [slice(None)]*Nqb
        for muA, muB in itertools.product(range(1,4), repeat=2):
            ind[nA] = muA
            ind[nB] = muB
            if muA != aA and muB != aB:
                H[tuple(ind)] *= -1
    
    ##measurement probabilities
    P0p = (1-np.sin(JA*deltaT)**2*np.cos(JB*deltaT)**2-np.cos(JA*deltaT)**2*np.sin(JB*deltaT)**2
          +(bA==bB)*(-1)**bA*0.5*sA*sB*np.sin(2*JA*deltaT)*np.sin(2*JB*deltaT)*Q)/2
    P0m = (1-np.sin(JA*deltaT)**2*np.cos(JB*deltaT)**2-np.cos(JA*deltaT)**2*np.sin(JB*deltaT)**2
          -(bA==bB)*(-1)**bA*0.5*sA*sB*np.sin(2*JA*deltaT)*np.sin(2*JB*deltaT)*Q)/2
    P1p = (np.sin(JA*deltaT)**2*np.cos(JB*deltaT)**2+np.cos(JA*deltaT)**2*np.sin(JB*deltaT)**2
          +(bA==bB)*0.5*sA*sB*np.sin(2*JA*deltaT)*np.sin(2*JB*deltaT)*Q)/2
    P1m = (np.sin(JA*deltaT)**2*np.cos(JB*deltaT)**2+np.cos(JA*deltaT)**2*np.sin(JB*deltaT)**2
          -(bA==bB)*0.5*sA*sB*np.sin(2*JA*deltaT)*np.sin(2*JB*deltaT)*Q)/2
        
    rtm02 = np.zeros([4]*Nqb)
    rtm12 = np.zeros([4]*Nqb)
    
    ###dR calculation
    dR = np.zeros([4]*Nqb)
    
    indAB = [slice(None)]*Nqb
    
    for i,j in itertools.product(range(4), repeat=2):
        indAB[nA] = i
        indAB[nB] = j
        if (Starg[tuple(indAB)] == 0).all() and bA != 3 and bB != 3:
            continue
  
        ##A terms
        if i != aA and aA!=0 and i !=0:
            if j == 0 or j == aB:
                dR[tuple(indAB)] -= 2*np.sin(deltaT*JA)**2*S[tuple(indAB)]
            else:
                dR[tuple(indAB)] -= 2*np.sin(deltaT*JA)**2*S[tuple(indAB)]*(1-np.sin(deltaT*JB)**2)
            if bA == 3:
                for k in range(1,4):
                    if k != i and k != aA:
                        Sindex = [slice(None)]*Nqb
                        Sindex[nA] = k
                        Sindex[nB] = j
                        
                        if j == 0 or j == aB:
                            dR[tuple(indAB)] += sA*np.sin(2*deltaT*JA)*int(LeviCivita(aA,k,i))*S[tuple(Sindex)]
                        else:
                            dR[tuple(indAB)] += sA*np.sin(2*deltaT*JA)*int(LeviCivita(aA,k,i))*S[tuple(Sindex)]*(1-2*np.sin(deltaT*JB)**2)
                        
        ##B terms
        if j != aB and aB!=0 and j !=0:
            if i == 0 or i == aA:
                dR[tuple(indAB)] -= 2*np.sin(deltaT*JB)**2*S[tuple(indAB)]
            else:
                dR[tuple(indAB)] -= 2*np.sin(deltaT*JB)**2*S[tuple(indAB)]*(1-np.sin(deltaT*JA)**2)
            if bB == 3:
                for k in range(1,4):
                    if k != j and k != aB:
                        Sindex = [slice(None)]*Nqb
                        Sindex[nB] = k
                        Sindex[nA] = i
                        
                        if i == 0 or i == aA:
                            dR[tuple(indAB)] += sB*np.sin(2*deltaT*JB)*int(LeviCivita(aB,k,j))*S[tuple(Sindex)]
                        else:
                            dR[tuple(indAB)] += sB*np.sin(2*deltaT*JB)*int(LeviCivita(aB,k,j))*S[tuple(Sindex)]*(1-2*np.sin(deltaT*JA)**2)
                            
        ##AB terms
        if bA == 3 and bB == 3 and i != aA and aA != 0 and j != aB and aB != 0 and i != 0 and j != 0:
            for k,l in itertools.product(range(1,4), repeat=2):
                if k != i and k != aA and l != j and l != aB:
                    Sindex = [slice(None)]*Nqb
                    Sindex[nA] = k
                    Sindex[nB] = l
                    dR[tuple(indAB)] += sA*sB*np.sin(2*JA*deltaT)*np.sin(2*JB*deltaT)*int(LeviCivita(aA,k,i))*int(LeviCivita(aB,l,j))*S[tuple(Sindex)]
                
    ##dR2 terms
        if bA != 3 and bB != 3:#and bA == bB
            ##A terms
            if i != aA and i!=0 and aA!=0:
                rtm12[tuple(indAB)] -= np.sin(JA*deltaT)**2*np.cos(JB*deltaT)**2*S[tuple(indAB)]
            ##B terms
            if j != aB and j!=0 and aB!=0:
                rtm12[tuple(indAB)] -= np.sin(JB*deltaT)**2*np.cos(JA*deltaT)**2*S[tuple(indAB)]
            
            ##AB terms
            if i != aA and i!=0 and aA!=0 and (j==aB or j==0 or aB==0):
                rtm02[tuple(indAB)] -= np.sin(JA*deltaT)**2*np.sin(JB*deltaT)**2*S[tuple(indAB)]
            elif j != aB and j!=0 and aB!=0 and (i==aA or i==0 or aA==0):
                rtm02[tuple(indAB)] -= np.sin(JA*deltaT)**2*np.sin(JB*deltaT)**2*S[tuple(indAB)]
            
    rtm03 = 1/4*(bA == bB)*(-1)**bA*sA*sB*np.sin(2*JA*deltaT)*np.sin(2*JB*deltaT)*(H-Q*S)
    rtm13 = 1/4*(bA == bB)*sA*sB*np.sin(2*JA*deltaT)*np.sin(2*JB*deltaT)*(F-Q*S)
            
    if P1p == 0 and P1m !=0:
        dR2 = (rtm12-rtm13)**2/P1m+(rtm02-rtm03)**2/P0m
    elif P1m == 0 and P1p !=0:
        dR2 = (rtm12+rtm13)**2/P1p+(rtm02+rtm03)**2/P0p
    elif P1p !=0 and P1m !=0:
        dR2 = ((rtm12+rtm13)**2/P1p+(rtm02+rtm03)**2/P0p+
                (rtm12-rtm13)**2/P1m+(rtm02-rtm03)**2/P0m)
    
    ##assemble
    if bA != 3 and bB != 3:#and bA == bB
        dC = (S-Starg)*dR+dR2/2
    else:
        dC = -Starg*dR
        
    return dC

##F tensor
def F_tensorized(S, Starg, A, B, Nqb):
    aA, bA, nA = A
    aB, bB, nB = B
    
    F = np.zeros([4]*Nqb)
    if aA==aB and aA==0:
        F = S
    else:
        ind = [slice(None)]*Nqb
        for muA in range(4):
            for muB in range(4):
                ind[nA] = muA
                ind[nB] = muB
                
                if (muA==aA or muA==0) and (muB==aB or muB==0):
                    Findex = [slice(None)]*Nqb
                    Findex[nA] = (muA==0)*aA
                    Findex[nB] = (muB==0)*aB
                    F[tuple(ind)] = S[tuple(Findex)]
                
                elif muA != 0 and muA != aA and muB != 0 and muB != aB and aA!=0 and aB!=0:
                    for k1 in range(1,4):
                        if k1 != aA and k1 != muA:
                            for k2 in range(1,4):
                                if k2 != aB and k2 != muB:
                                    Sindex = [slice(None)]*Nqb
                                    Sindex[nA] = k1
                                    Sindex[nB] = k2
                                    F[tuple(ind)] += int(LeviCivita(aA,muA,k1)*LeviCivita(aB,muB,k2))*S[tuple(Sindex)]
                    
                elif muA != 0 and muA != aA and muB != 0 and muB != aB and aA==0 and aB!=0:
                    for k2 in range(1,4):
                        if k2 != aB and k2 != muB:
                            Sindex = [slice(None)]*Nqb
                            Sindex[nA] = muA
                            Sindex[nB] = k2
                            F[tuple(ind)] += int(LeviCivita(aB,muB,k2))*S[tuple(Sindex)]
                elif muA != 0 and muA != aA and muB != 0 and muB != aB and aA!=0 and aB==0:
                    for k1 in range(1,4):
                        if k1 != aA and k1 != muA:
                            Sindex = [slice(None)]*Nqb
                            Sindex[nA] = k1
                            Sindex[nB] = muB
                            F[tuple(ind)] += int(LeviCivita(aA,muA,k1))*S[tuple(Sindex)]
                            
    return F

##run main program
if __name__ == '__main__':
    main()