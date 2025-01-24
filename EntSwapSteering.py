#!/usr/bin/python3.9
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 28 14:40:09 2021

@author: Samuel Morales
TODO: save traj, vectorize cost function calculation
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
    ##Parameters
    Nqb = 3                                             ##Number of qubits
    DeltaT = 0.2                                        ##time step length, default: 0.2
    J = Nqb*[1]                                         ##coupling strength, default: [1, 0.99, 1.01, 1.005, 0.995, 1.003]
    #J = [1, 0.99, 1.01, 1.005, 0.995, 1.003, 0.997, 1.007]  
    N = 130                                             ##nr of time steps
    M = 100                                             ##number of trajectories
    Nst = 1                                             ##every Nst´th step is saved
    
    ##cost fct probabilities
    plist =[9*10**(-i) for i in range(1,Nqb)]
    plist.append(1-sum(plist))
    
    ##couplings, target fidelity Fstar
    K = 9                                               ##nr of couplings: 3, 4, 6, 8, 9, 10, 12, 14
    Fstar = 0.8                                         ##target fidelity
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
    plt.hist(np.concatenate(k_List), bins=np.arange(0,K**2), alpha=0.5, edgecolor='grey')
    
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
    
    #avgCloc = [np.mean(costFDistr[:,i]) for i in range(int(N/Nst)+1)]
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
    elif K == 8:        ##beta=x,y; alpha=0,x,y,z; s=+
        slist = [1,1,1,1,1,1,1,1]
        aList = [0,1,2,3,0,1,2,3]
        bList = [1,1,1,1,2,2,2,2]
    elif K == 9:        ##beta=x,z; alpha=x,y,z; s=+/-
        slist = [1,1,1,-1,-1,-1,1,1,1]
        aList = [1,2,3,1,2,3,1,2,3]
        bList = [3,3,3,3,3,3,1,1,1]
    elif K == 10:        ##beta=x,z; alpha=0,x,y,z; s=+/-
        slist = [1,1,1,-1,-1,-1,1,1,1,1]
        aList = [1,2,3,1,2,3,0,1,2,3]
        bList = [3,3,3,3,3,3,1,1,1,1]
    elif K == 12:        ##beta=x,y,z; alpha=x,y,z; s=+/-
        slist = [1,1,1,-1,-1,-1,1,1,1,1,1,1]
        aList = [1,2,3,1,2,3,1,2,3,1,2,3]
        bList = [3,3,3,3,3,3,1,1,1,2,2,2]
    elif K == 14:        ##beta=x,y,z; alpha=0,x,y,z; s=+/-
        slist = [1,1,1,-1,-1,-1,1,1,1,1,1,1,1,1]
        aList = [1,2,3,1,2,3,0,1,2,3,0,1,2,3]
        bList = [3,3,3,3,3,3,1,1,1,1,2,2,2,2]
        
    ##operator list
    pauliPlaqList = plaqSlist(Nqb)
    
    ##initial state
    psi_List[0] = psi0
    psi = psi0
    
    ##initialize Pauli tensors
    Spsi = qt.expect(pauliPlaqList, psi0)
    Starg = qt.expect(pauliPlaqList, psiTarg)
    
    ##subset list
    subset = iniSubset(Nqb)
    
    ##Initial cost function values
    global_C[0] = np.linalg.norm(Spsi-Starg)**2/2**(Nqb+1)
    Fid = global_C[0]
    
    cost = csFct(Spsi, Starg, Nqb, subset)
    local_C[0] = sum([pList[j]*cost[j] for j in range(Nqb)])
    
    #qbList = np.arange(0,Nqb)                  ##needed for fully coupled chain
    
    ##Time step loop
    for i in range(1, N+1):
        ##starting steering pair
        nStart1 = int(N*rng.random())
        nStart2 = (nStart1+1)%Nqb
        
        #random.shuffle(qbList)                 ##needed for fully coupled chain
        
        ##steering neighbouring pairs
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
            deltaC = expCostF(Spsi, Starg, J, Gamma, DeltaT, pList, n1, n2, Nqb, K)
        
            klis = int(rng.choice(np.where(deltaC == np.nanmin(deltaC))[0]))
            k_List[(i-1)*int(Nqb/2)+nPair] = klis
            
            ##chosen couplings
            s1 = slist[int(klis%K)]
            s2 = slist[int(klis/K)]
            alpha1 = aList[int(klis%K)]
            alpha2 = aList[int(klis/K)]
            beta1 = bList[int(klis%K)]
            beta2 = bList[int(klis/K)]
            
            ##non-improvement criterion
            dcost = 0.001
            if np.nanmin(deltaC) > dcost:
                success_Step = N+2
                break
            
            ##chosen Pauli operators
            sig1 = pauliPlaqList[alpha1*4**n1]
            sig2 = pauliPlaqList[alpha2*4**n2]
            
            ##Time step
            #beta1=beta2=z
            if beta1==3 and beta2==3:
                xi_eta_List[i-1] = (0,(-1)**int(2*rng.random()))
                H = s1*J1*sig1+s2*J2*sig2
                psi = schroesol(psi, DeltaT, H)
            
            #beta1=z, beta2!=z
            elif beta1==3:
                c_op, H = np.sqrt(G2)*sig2, s1*J1*sig1
                psi, xi = unitsol(psi, DeltaT, H, c_op, G2*DeltaT)
                xi_eta_List[i-1] = (xi,(-1)**int(2*rng.random()))
            
            #beta1!=z, beta2=z
            elif beta2==3:
                c_op, H = np.sqrt(G1)*sig1, s2*J2*sig2
                psi, xi = unitsol(psi, DeltaT, H, c_op, G1*DeltaT)
                xi_eta_List[i-1] = (xi,(-1)**int(2*rng.random()))
            
            #beta1=beta2=x/y  
            elif beta1==beta2:
                c_op = [np.sqrt(G1/2)*sig1+np.sqrt(G2/2)*sig2, np.sqrt(G1/2)*sig1-np.sqrt(G2/2)*sig2]
                psi, xi_eta_List[i-1] = ent_swap_sol(psi, DeltaT, c_op)               
            
            #(beta1=x, beta2=y) or (beta1=y, beta2=x)
            else:
                eta = (-1)**int(2*rng.random())
                if beta1==1:
                    c_op, H = np.sqrt(G1)*sig1+eta*1j*np.sqrt(G2)*sig2, eta*np.sqrt(G1*G2)*sig1*sig2
                elif beta1==2:
                    c_op, H = np.sqrt(G1)*sig1-eta*1j*np.sqrt(G2)*sig2, eta*np.sqrt(G1*G2)*sig1*sig2
                psi, xi = unitsol(psi, DeltaT, H, c_op, (G1+G2)*DeltaT)
                xi_eta_List[i-1] = (xi, eta)
                
        ##Update values   
        Spsi = qt.expect(pauliPlaqList,psi)    
        Fid = np.linalg.norm(Spsi-Starg)**2/2**(Nqb+1)
        
        ##stoppage criterion at global cost value eps, corresponding to F=F*
        if Fid < eps:
            ##final values
            psi_List[int((i-1)/Nst)+1:] = (int(N/Nst)-int((i-1)/Nst))*[1]*psi
            global_C[int((i-1)/Nst)+1:] = Fid
            cost = csFct(Spsi, Starg, Nqb, subset)
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
            cost = csFct(Spsi, Starg, Nqb, subset)
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
    subset = []
    for i in range(1,N):
        subset.append(list(itertools.combinations(range(N),i)))
    return subset

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
    '''return normalized state conditioned on measurement outcome
    for single, unitary jump operator
    '''
    #random number generator
    rng = np.random.default_rng()
    if rng.random() >= P:
        return ((1-1j*deltaT*H-P/2)*psi).unit(), 0
    else:
        return (c_op*psi).unit(), 1

def schroesol(psi, deltaT, H):
    '''Single step Hamiltonian evolution
    Parameters
    ----------

    H: :class:`qutip.Qobj`
        Hamiltonian

    psi: :class:`qutip.Qobj`
        initial state vector (ket)
        
    deltaT: (float)
        time step length
        
    Returns
    -------

    result: :class:`qutip.Qobj`

        Normalized time-evolved state (ket)
    '''
    
    return ((1-1j*deltaT*H)*psi).unit()

###plaquette operators
def plaqS(ind):
    pauliLs = [s[i] for i in ind]
    return qt.tensor(pauliLs)
    
def plaqSlist(Nqb):
    return  [plaqS([int(i/(4**j)%4) for j in range(Nqb)]) for i in range(4**Nqb)]

##cost function for all levels 1 - Nqb
def csFct(S, Starg, Nqb, subsets):
    costf = np.zeros(Nqb)
    for i in range(Nqb-1):
       costf[i] = costFunc(S, Starg, i+1, Nqb, subsets[i])
    costf[Nqb-1] = np.linalg.norm(S-Starg)**2/2**(Nqb+1)
    return costf

##cost function of level i
def costFunc(S, Starg, i, Nqb, subset):
    costf = 0
    for s in subset:
        for j in range(4**Nqb):
            chk = 0
            for l in range(Nqb):
                if l not in s and int(j/(4**l)%4)>0:
                    chk +=1
            if chk == 0:
                costf+=(S[j]-Starg[j])**2
    
    costf *= 1/2**(i+1)/scipy.special.binom(Nqb,i)
    return costf

##expected cost function change
def expCostF(S, Starg, J, Gamma, deltaT, p, nA, nB, Nqb, K):
    JA = J[nA]
    JB = J[nB]
    GA = Gamma[nA]
    GB = Gamma[nB]
    
    ##coupling lists
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
    elif K == 8:        ##beta=x,y; alpha=0,x,y,z; s=+
        slist = [1,1,1,1,1,1,1,1]
        aList = [0,1,2,3,0,1,2,3]
        bList = [1,1,1,1,2,2,2,2]
    elif K == 9:        ##beta=x,z; alpha=x,y,z; s=+/-
        slist = [1,1,1,-1,-1,-1,1,1,1]
        aList = [1,2,3,1,2,3,1,2,3]
        bList = [3,3,3,3,3,3,1,1,1]
    elif K == 10:        ##beta=x,z; alpha=0,x,y,z; s=+/-
        slist = [1,1,1,-1,-1,-1,1,1,1,1]
        aList = [1,2,3,1,2,3,0,1,2,3]
        bList = [3,3,3,3,3,3,1,1,1,1]
    elif K == 12:        ##beta=x,y,z; alpha=x,y,z; s=+/-
        slist = [1,1,1,-1,-1,-1,1,1,1,1,1,1]
        aList = [1,2,3,1,2,3,1,2,3,1,2,3]
        bList = [3,3,3,3,3,3,1,1,1,2,2,2]
    elif K == 14:        ##beta=x,y,z; alpha=0,x,y,z; s=+/-
        slist = [1,1,1,-1,-1,-1,1,1,1,1,1,1,1,1]
        aList = [1,2,3,1,2,3,0,1,2,3,0,1,2,3]
        bList = [3,3,3,3,3,3,1,1,1,1,2,2,2,2]
        
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
                
                elif muA != 0 and muA != aA and muB != 0 and muB != aB and aA!=0 and aB!=0:
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
                
        ##<c_eta^\dagger c_eta>
        avcp = (bA != 3)*GA+(bB != 3)*GB
        rtm1 = (bA == bB)*(bA != 3)*2*np.sqrt(GA*GB)*Q
        avcm = avcp-rtm1
        avcp += rtm1
            
        dR, dR2 = np.zeros(4**Nqb), np.zeros(4**Nqb)
        
        ##calculate <<dR>> and <<dR^2>>
        for l in range(4**Nqb):
            ##<<dR>> terms
            muA = int(l/(4**nA)%4)
            muB = int(l/(4**nB)%4)
            
            rtm1 = 0
            rtm2 = 0
            ##A terms
            if muA != aA and muA != 0 and aA!=0:
                if bA !=3:
                    rtm1 -= GA*S[l]
                    rtm2 -= GA*S[l]
                else:
                    rtm4 = 0
                    for k in range(1,4):
                        if k != muA and k != aA:
                            rtm4 += LeviCivita(aA,k,muA)*S[l+(k-muA)*4**nA]
                    rtm1 += sA*JA*rtm4
            
            ##B terms
            if muB != aB and muB !=0 and aB!=0:
                if bB !=3:
                    rtm1 -= GB*S[l]
                    rtm2 -= GB*S[l]
                else:
                    rtm4 = 0
                    for k in range(1,4):
                        if k != muB and k != aB:
                            rtm4 += LeviCivita(aB,k,muB)*S[l+(k-muB)*4**nB]
                    rtm1 += sB*JB*rtm4
                    
            dR[l] = 2*deltaT*rtm1
        
            ##<<dR^2>>/2 terms
            if bA != 3 or bB != 3:
                rtm3 = 0
                if bA == bB:
                    rtm3 = np.sqrt(GA*GB)*(F[l]-Q*S[l])
                
                if avcp == 0:
                    dR2[l] = deltaT*(rtm2-rtm3)**2/avcm
                elif avcm == 0:
                    dR2[l] = deltaT*(rtm2+rtm3)**2/avcp
                else:
                    dR2[l] = deltaT*((rtm2+rtm3)**2/avcp+(rtm2-rtm3)**2/avcm)
        
        ##assemble
        for l in range(Nqb-1):
            deltaf = 0       
            for s in subset[l]:
                for m in range(4**Nqb):
                    chk = 0
                    for i in range(Nqb):
                        if i not in s and int(m/(4**i)%4)>0:
                            chk +=1
                    if chk == 0:
                        deltaf+=(S[m]-Starg[m])*dR[m]+dR2[m]
            costf[j] += p[l]*deltaf/scipy.special.binom(Nqb,l+1)/2**(l+1)
        costf[j] += sum(-p[-1]*Starg*dR/2**Nqb)
        
    return costf

##run main program
if __name__ == '__main__':
    main()