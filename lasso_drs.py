#
# Feb 1, 2022:  (1) Test LASSO to get DRS from G*
#               (2) try tests 3, 4, 5, 6, 7 from https://github.com/shane5ul/pyReSpect-freq/tree/master/tests
#
from timeit import default_timer as timer

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LassoCV, LassoLars

# LassoCV throws lots of  ConvergenceWarnings; so suppressing them here.
import warnings
warnings.filterwarnings("ignore")

import matplotlib as mpl
from cycler import cycler
mpl.rcParams['axes.prop_cycle'] = cycler('color', ['k', '#d62728', '#1f77b4', '#2ca02c', '#9467bd', '#7f7f7f', '#17becf'])

plt.style.use(['myjournal', 'seaborn-ticks'])
clr = [p['color'] for p in plt.rcParams['axes.prop_cycle']]
symb = ['o', 'v', 's', 'd', '^', '<']


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
def lassoFit(wexp, Gexp, isPlateau, decade_density=10, verbose=True):
    
    """The core LASSO engine:
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
   
    #########older#############
        
    #~ # actual fitting
    #~ WtMat   = np.diag(np.sqrt(1./np.abs(Gexp))) # 2n*2n diagonal matrix with 1/sqrt(Gexp) on diagonal

    #~ K_trans = np.dot(WtMat, K)
    #~ G_trans = np.dot(WtMat, Gexp)
    #~ model   = LassoCV(cv=3, fit_intercept=False)
    
    #~ tstart = timer()
    #~ model.fit(K_trans, G_trans)

    #~ score = model.score(K_trans, G_trans)  # R2
    #~ alpha = model.alpha_
    
    #########older#############
    

    #########newer#############

    # actual fitting
    WtMat   = np.diag(np.sqrt(1./np.abs(Gexp))) # 2n*2n diagonal matrix with 1/sqrt(Gexp) on diagonal
    
    K_trans = np.dot(WtMat, K)
    G_trans = np.dot(WtMat, Gexp)
    model   = LassoCV(cv=3, fit_intercept=False, positive=True)

    
    tstart = timer()
    model.fit(K_trans, G_trans)

    score = model.score(K_trans, G_trans)  # R2
    alpha = model.alpha_
    

    #########newer#############
    
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
        print("nmodes", len(g))
        print("G0", G0)
        #~ print(np.c_[tau, g])
        print("score", score)
        
    return g, tau, G0, alpha, score


def getSmoothWeights(x, y, isPlot=True):
    """Apply Savitzy-Golay filter to potentially noisy experimental data and
       figure out the smoothed weights to be later passed to LASSO"""

    # smooth to get weights
    from scipy.signal import savgol_filter
    from scipy.interpolate import interp1d


    # 1. Get y on uniform x-grid
    fint  =  interp1d(x, y, fill_value="extrapolate")
    xS    =  np.linspace(np.min(x), np.max(x), len(x))

    # 2. Apply the filter to get the smoothed values
    yS    = savgol_filter(fint(xS), 11, 3) # window size 11, polynomial order 3


    # 3. Return filtered value yF original x
    fint  = interp1d(xS, yS, fill_value="extrapolate")
    yF    = fint(x)  

    if isPlot:

        plt.plot(x, y, 'o', c=clr[0])
        plt.plot(xS, yS, c=clr[0])

        #~ plt.plot(x, y*weight, 'o-')
        
        plt.xlim(min(x), max(x))

        plt.xlabel(r'log $\omega$')
        plt.ylabel(r'log $G_{*}$')

        plt.legend(loc='upper left')
        plt.tight_layout()
        plt.show()

    return yF
 
   
def readGstData(fname, getWeights=False):
    """Function: GetExpData(input)
       Reads in the experimental data from the input file
       Input:  fname = name of file that contains w, Gp, Gpp in 3 columns [w Gp Gpp]
       Output: A n*1 vector wexp, a 2n*1 vector Gexp, and a 2n*1 vector wt = 1/smoothed Gexp"""

    w, Gp, Gpp = np.loadtxt(fname, unpack=True)  # 3 columns,  w - Gp - Gpp
    
    if getWeights:
        yF  = getSmoothWeights(np.log(w), np.log(Gp), isPlot=False)
        Gp = np.exp(yF) 

        yF = getSmoothWeights(np.log(w), np.log(Gpp), isPlot=False)
        Gpp = np.exp(yF)

    return w, np.append(Gp, Gpp)


def plotGst(g, tau, G0, ds, markTau=True):
    """Co-ordinator of all G3st plots"""
    
    wexp, Gexp = readGstData(ds)

    wmin = min(wexp)
    wmax = max(wexp)
    n    = len(wexp)
    
    N    = 200
    w    = np.geomspace(wmin, wmax, N)        
    Gst  = maxwellSAOS(w, g, tau,G0)

    plt.plot(w, Gst[:N], c=clr[0], label='$G^{\prime}$')
    plt.plot(wexp, Gexp[:n], 'o', c=clr[0])

    plt.plot(w, Gst[N:], c=clr[1], label="$G^{\prime\prime}$")
    plt.plot(wexp, Gexp[n:], 's', c=clr[1])

    if markTau:
        gmax = max(np.abs(g)); gmin = min(np.abs(g))
        for i, tau_i in enumerate(tau):
            alpha_i = 0.3 + np.sqrt((np.abs(g[i]) - gmin)/(gmax - gmin)) * 0.7
            plt.axvline(x=1.0/tau_i, c='gray', lw=1, alpha = alpha_i)

    plt.xscale('log')
    plt.yscale('log')
    ylo = np.sort(np.abs(Gexp))[1]/2
    plt.ylim(ylo, None)
    plt.xlim(min(w), max(w))


    plt.xlabel(r'$\omega$')
    plt.ylabel(r'$G_{*}$')

    plt.legend(loc='upper left')
    plt.tight_layout()
    #~ plt.savefig('Gst.pdf')
    plt.show()

    return


#============================
#           main
#============================
if __name__ == '__main__':

    isPlateau = False
    ds = 'test3.dat'
 
    wexp, Gexp = readGstData(ds, getWeights=True)
       
    g, tau, G0, alpha, score = lassoFit(wexp, Gexp, isPlateau, decade_density=5)
    plotGst(g, tau, G0, ds, markTau=True)
 
 
