#
# June 3, 2022: (1) clean up local files for github upload
#               (2) get DRS-ST (using LASSO) and DRS-ST-FT
#

from timeit import default_timer as timer

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LassoCV
from scipy.optimize import nnls, least_squares


# LassoCV throws lots of  ConvergenceWarnings; so suppressing them here.
import warnings
warnings.filterwarnings("ignore")


#--------------------------
# G* functions for Maxwell
#--------------------------
def maxwellSAOS(w, g, tau, G0=0.0):
    """Takes DRS g, tau [from fitting LVE] and predicts G* using DRS
       Gp = first n elements of Gst"""
    
    n     = len(w)    
    S, W  = np.meshgrid(tau, w)
    ws    = S*W
    ws2   = ws**2

    K1    = ws2/(1+ws2)
    K2    = ws/(1+ws2)

    K     = np.vstack((K1, K2))   # 2n * nmodes
    Gst   = np.dot(K, g)

    Gst[:n] += G0                 # add plateau
         
    return Gst

#--------------------
# Main Lasso engine
#--------------------
def lassoFit(wexp, Gexp, isPlateau, decade_density=5, verbose=True):
    
    """The core LASSO engine: outputs DRS-ST and KKR-compliance
    Inputs:
        wexp: an array of size n containing the experimental frequencies
        Gexp: an array of size 2n containing G' stacked over G"
        isPlateau : is there a nonzero equilibrium modulus G0 to fit?
        decade_density: density of modes per decade; 
                        the number of modes N is inferred using this quantity
        verbose: if True does onscreen printing
        
    Outputs:
        g, tau : arrays of length <N for Maxwell modes. only nonzero elements are returned
        alpha  : the optimal value of alpha used in LASSO
        score  : coefficient of determination R2; used to quantitatively assess quality of fit

    """
    # odd number of modes, puts a mode at the center of data range
    wmin = min(wexp); wmax = max(wexp)
    n    = len(wexp)
    G0   = 0.0
    N    = decade_density * int(np.log10(wmax/wmin) + 2)
    tau = np.geomspace(0.1/wmax, 10/wmin, N)

    # Set up regression problem so that Gexp = K * g, 
    # where K is 2n*N, g = N*1, and Gst = 2n*1
    S, W    = np.meshgrid(tau, wexp)
    ws      = S*W
    ws2     = ws**2
    K       = np.vstack((ws2/(1+ws2), ws/(1+ws2)))   # 2n * N

    # then K is 2n*(N+1), where last element of g is G0
    if isPlateau:
        K   = np.hstack(( K, np.ones((len(Gexp), 1)) ))  # G' needs some G0
        K[n:, N] = 0.
   
       
    # actual fitting

    WtMat   = np.diag(np.sqrt(1./np.abs(Gexp))) # 2n*2n diagonal matrix with 1/sqrt(Gexp) on diagonal
    K_trans = np.dot(WtMat, K)
    G_trans = np.dot(WtMat, Gexp)


    model   = LassoCV(cv=3, positive=True, fit_intercept=False)
    
    tstart = timer()
    model.fit(K_trans, G_trans)

    score = model.score(K_trans, G_trans)  # R2
    alpha = model.alpha_
        
    if isPlateau:
        G0 = model.coef_[N]
        model.coef_ = model.coef_[:N]
    
    cnd   = np.abs(model.coef_) > 1.0e-16
    g     = model.coef_[cnd]
    tau   = tau[cnd]
    
    if verbose:
        print('time elapsed', timer() - tstart)
        print('condition number (unreg)', np.linalg.cond(K_trans))

        print("alpha", alpha)
        print("nmodes", len(g), N)
        print("G0", G0)
        #~ print(np.c_[tau, g])
        print("score", score)
        
    return g, tau, G0, alpha, score
 
   
def readGstData(fname):
    """Function: GetExpData(input)
       Reads in the experimental data from the input file
       Input:  fname = name of file that contains w, Gp, Gpp in 3 columns [w Gp Gpp]
       Output: A n*1 vector wexp, a 2n*1 vector Gexp"""

    w, Gp, Gpp = np.loadtxt(fname, unpack=True)  # 3 columns,  w - Gp - Gpp
    
    return w, np.append(Gp, Gpp)


def plotGst(g, tau, G0, ds, markTau=False, plotFile='none'):
    """Co-ordinator of all Gst plots"""
    
    wexp, Gexp = readGstData(ds)

    wmin = min(wexp)
    wmax = max(wexp)
    n    = len(wexp)
    
    N    = 200
    w    = np.geomspace(wmin, wmax, N)        
    Gst  = maxwellSAOS(w, g, tau,G0)

    plt.plot(w, Gst[:N], c=clr[0], label='$G^{\prime}$')
    plt.plot(wexp, Gexp[:n], 'o', c=clr[0], alpha=0.5)

    plt.plot(w, Gst[N:], c=clr[1], label="$G^{\prime\prime}$")
    plt.plot(wexp, Gexp[n:], 's', c=clr[1], alpha=0.5)

    if markTau:
        gmax = max(np.abs(g)); gmin = min(np.abs(g))
        for i, tau_i in enumerate(tau):
            plt.axvline(x=2.0 * np.pi/tau_i, c='gray', lw=1, alpha = 0.3)

    plt.xscale('log')
    plt.yscale('log')
    ylo = np.sort(np.abs(Gexp))[1]/2
    plt.ylim(ylo, None)
    plt.xlim(min(w), max(w))


    plt.xlabel(r'$\omega$ [rad/s]')
    plt.ylabel(r'$G^{*}$ [Pa]')

    plt.legend(loc='upper left')
    plt.tight_layout()
    if plotFile != 'none':
        plt.savefig(plotFile)
    plt.show()

    return

def calc_LogError(wexp, Gexp, g, tau, G0):
    """Log error used in Supplementary Info of paper"""
    
    Gpredict = maxwellSAOS(wexp, g, tau, G0)
        
    err = 1.0/(4*len(wexp)) * np.sum((np.log(Gpredict) - np.log(Gexp))**2)
    return err
    
#================== FINE-TUNING DRS-ST -> DRS-ST-FT ===============#

def MaxwellModes(z, w, Gexp, isPlateau):
    """
    
     Function: MaxwellModes(input)
    
     Solves the linear least squares problem to obtain the DRS
    
     Input: z = points distributed according to the density,
            t    = n*1 vector contains times,
            Gexp = 2n*1 vector contains Gp and Gpp
            isPlateau = True if G0 \neq 0    
    
     Output: g, tau = spectrum  (array)
             error = relative error between the input data and the G(t) inferred from the DRS
             condKp = condition number
    
    """

    N      = len(z)
    tau    = np.exp(z)
    n      = len(w)

    #
    # Prune small -ve weights g(i)
    #
    g, error, condKp = nnLLS(w, tau, Gexp, isPlateau)

    # first remove runaway modes outside window with potentially large weight
    izero = np.where(np.logical_or(max(w)*min(tau) < 0.02, min(w)*max(tau) > 50.))
    tau   = np.delete(tau, izero);
    g     = np.delete(g, izero)

    # search for small weights (gi) 
    if isPlateau:
        izero = np.where(g[:-1]/np.max(g[:-1]) < 1e-8)
    else:
        izero = np.where(g/np.max(g) < 1e-8)

    tau   = np.delete(tau, izero);
    g     = np.delete(g, izero)
        
    return g, tau, error, condKp

def nnLLS(w, tau, Gexp, isPlateau):
    """
    #
    # Helper subfunction which does the actual LLS problem
    # helps MaxwellModes
    #
    """    
    n       = int(len(Gexp)/2)
    ntau    = len(tau)
    S, W    = np.meshgrid(tau, w)
    ws      = S*W
    ws2     = ws**2
    K       = np.vstack((ws2/(1+ws2), ws/(1+ws2)))   # 2n * nmodes

    # K is n*ns [or ns+1]
    if isPlateau:
        K   = np.hstack(( K, np.ones((len(Gexp), 1)) ))  # G' needs some G0
        K[n:, ntau] = 0.                                 # G" doesn't have G0 contrib
    #
    # gets (Gst/GstE - 1)^2, instead of  (Gst -  GstE)^2
    #
    Kp      = np.dot(np.diag((1./Gexp)), K)
    condKp  = np.linalg.cond(Kp)
    g       = nnls(Kp, np.ones(len(Gexp)), maxiter=10*Kp.shape[1])[0]
        
    GstM       = np.dot(K, g)
    error     = np.sum((GstM/Gexp - 1.)**2)

    return g, error, condKp      

def FineTuneSolution(tau, w, Gexp, isPlateau):
    """Given a spacing of modes tau, tries to do NLLS to fine tune it further
       If it fails, then it returns the old tau back       
       Uses helper function: res_wG which computes residuals
       """       

    success   = False
    initError = np.linalg.norm(res_wG(tau, w, Gexp, isPlateau))

    try:
        res     = least_squares(res_wG, tau, bounds=(0.02/max(w), 50/min(w)),
                             args=(w, Gexp, isPlateau))        
        tau     = res.x        
        tau0    = tau.copy()
        success = True
    except:
        pass

    g, tau, _, _ = MaxwellModes(np.log(tau), w, Gexp, isPlateau)   # Get g_i, taui
    finalError   = np.linalg.norm(res_wG(tau, w, Gexp, isPlateau))

    # keep fine tuned solution, only if it improves things
    if finalError > initError:
        success = False

    return success, g, tau

def res_wG(tau, wexp, Gexp, isPlateau):
    """
        Helper function for final optimization problem
    """
    g, _, _ = nnLLS(wexp, tau, Gexp, isPlateau)
    Gmodel  = np.zeros(len(Gexp))

        
    S, W    = np.meshgrid(tau, wexp)
    ws      = S*W
    ws2     = ws**2
    K       = np.vstack((ws2/(1+ws2), ws/(1+ws2)))   # 2n * nmodes

    # add G0
    if isPlateau:
        Gmodel     = np.dot(K, g[:-1])
        n = int(len(Gexp)/2)
        Gmodel[:n] += g[-1]    
    else:
        Gmodel     = np.dot(K, g)
                
    residual = Gmodel/Gexp - 1.
        
    return residual



def get_DRS_ST_FT(wexp, Gexp, isPlateau):
    """Function invokes Lasso regression, and calls fine-tuning function"""
        g, tau, G0, alpha, score = lassoFit(wexp, Gexp, isPlateau, decade_density=2, verbose=False)
                
        succ, gf, tauf  = FineTuneSolution(tau, wexp, Gexp, isPlateau)        
        if succ:
            g   = gf.copy(); tau = tauf.copy()
            if isPlateau:
                G0  = g[-1]; g = g[:-1]
            else:
                G0 = 0.
                
        return g, tau, G0
    

#====================================================================
#                       main                                        
#====================================================================
if __name__ == '__main__':
    
    # input whether G0 = 0 (isPlateau = False), and file containing dataset
    isPlateau = False
    wexp, Gexp = readGstData('test3.dat')

    # extract DRS-ST using regular LASSO, and plot it
    g, tau, G0, alpha, score = lassoFit(wexp, Gexp, isPlateau, decade_density=5, verbose=False)
    plt.loglog(tau,g, '^-', c='#1f77b4', alpha=0.4, lw=1, label='DRS-ST')
    
    # DRS-ST-FT - uses additional fine-tuning steps
    g, tau, G0 = get_DRS_ST_FT(wexp, Gexp, isPlateau)
    plt.loglog(tau,g, 's-', c='#2ca02c', lw=1, label='DRS-ST-FT')

    # mark reliable range
    wmin = min(1./wexp)*np.exp(np.pi/2)
    wmax = max(1./wexp)*np.exp(-np.pi/2)
    plt.axvline(x=wmin, ls=':', c='gray', alpha=0.6)
    plt.axvline(x=wmax, ls=':', c='gray', alpha=0.6)

    plt.xlim(min(1/wexp), max(1/wexp))
    # ~ plt.ylim(1, 1e6)

    plt.xlabel(r'$\tau$')
    plt.ylabel(r'$g$')
    plt.legend()
    plt.tight_layout()
    plt.show()
