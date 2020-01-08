#from sys import path
#path.append(r'path to folder with PFG.py')
#import PFG

################################
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import lmfit as lf
import re
from io import StringIO
from copy import deepcopy

def get_params(path):
    file = open(path+'/pdata/1/ct1t2.txt').read()
    params = re.findall('[^=^\n]*=\s*[0-9]+[^\n]*', file)
    dic = {}
    for i, n in enumerate(params): 
        par, val = n.split('=')
        par = par.strip()
        val = re.findall('\s*[0-9]+.[0-9]+e?[\+\-]?[0-9]*',val)[0]
        val = float(val)
        dic[par] = val
    return dic

def calc_k(data, params):
    gamma = params['Gamma']
    Gi = data['Gradient']
    LD = params['Little Delta']*1e-3
    BD = params['Big Delta']*1e-3
    data['k'] = (2*np.pi*gamma*Gi*LD)**2*(BD-LD/3)*1e4
    return data

def calc_q(data, params):
    gamma = params['Gamma']
    Gi = data['Gradient']
    LD = params['Little Delta']*1e-3
    BD = params['Big Delta']*1e-3
    data['q'] = (((gamma*Gi*LD)*100)/(2*np.pi))
    return data

def make_exp_decay(n=1):
    s = 'lambda k'
    f = '0'
    for i in range(n):
        s += ', A%i, D%i'%(i, i)
        f += '+A%i*np.exp(-k*D%i)'%(i, i)
    s += ': '+f
    func = eval(s)
    func.__name__ = 'Exponential Decay'
    return func

def import_ct1t2(path):
    file = open(path).read()
    splitted = re.split('d*\n=+\nd*',file)
    
    peaks = np.array([])
    for s in splitted:
        try:
            params = re.findall('[^=^\n]*=\s*[0-9]+[^\n]*', s)
            dic = {}
            for i, n in enumerate(params): 
                par, val = n.split('=')
                par = par.strip()
                val = re.findall('\s*[0-9]+.[0-9]+e?[\+\-]?[0-9]*',val)[0]
                val = float(val)
                dic[par] = val    
            table = re.findall('\nPoint[\w\s\S\W]*', s)[0]
            df = pd.read_table(StringIO(table), delimiter='[\s]+', engine='python')
            dic['data'] = df
            peaks = np.append(peaks, dic)
        except:
            pass
    return peaks

def plot_peak_decays(meas):
    '''
    takes a single or a list of evaluated Diff_Topspin_MultiPeak objects
    and returns a figure of the plotted decays
    '''

    fig, ax = plt.subplots()
    
    #if type(meas) == dict:
        
    
    if type(meas) == list:
        pass
    elif type(meas) != dict:
        meas = [meas]

    for m in meas:
        for p in range(len(m.peaks)):
            ax.plot(m.peaks[p]['data']['k'], m.peaks[p]['data']['Expt'],
                    'x', label='peak %i: $\Delta$ = %.1f ms'%(p, m.peaks[p]['Big Delta']))
    ax.set_yscale('log')
    ax.set_ylabel('$I\cdot I^{-1}$')
    ax.set_xlabel('$k$')
    ax.legend()
    return fig

class Diff_Topspin_MultiPeak(object):
    def __init__(self, path, normalize_intensity=False):
        self.path = path
        self.results = {}
        try:
            self.title = open(self.path+'/pdata/1/title').read()
            print(self.title)
        except:
            self.title = None
            print('no title file found')
        try:
            self.peaks = import_ct1t2(self.path)
            print('directly imported ct1t2 file')
        except:
            self.peaks = import_ct1t2(self.path+'/pdata/1/ct1t2.txt')
            print('imported normally')
        
        for n in self.peaks:
            calc_k(n['data'], n)
            calc_q(n['data'], n)
        print('%i peaks imported'%len(self.peaks))
        
        self.porig = deepcopy(self.peaks)
        
        if normalize_intensity:
            for p in self.peaks:
                p['data']['Expt'] /= p['data']['Expt'].iloc[0]
            for p in self.porig:
                p['data']['Expt'] /= p['data']['Expt'].iloc[0]
    
    def plot_decay(self, x='k', peaks=0, **kwargs):
        fig, ax = plt.subplots()
        if peaks == 0:
            ax.errorbar(x, 'Expt', data=self.peaks[0]['data'], label='peak 0', **kwargs)
        elif type(peaks) == list:
            for p in peaks:
                ax.errorbar(x, 'Expt', data=self.peaks[p]['data'], label='peak %i'%p, **kwargs)
            ax.legend()
        ax.set_yscale('log')
        ax.set_xlabel(x)
        ax.set_ylabel('$I \cdot I_0^{-1}$')
        return fig
    
    def set_datarange_k(self, kmin=None, kmax=None, peaks='all'):
        if peaks == 'all':
            for i, p in enumerate(self.peaks):
                if kmin is None:
                    kmin = self.porig[i]['data']['k'].min()
                if kmax is None:
                    kmax = self.porig[i]['data']['k'].max()

                p['data'] = self.porig[i]['data'][(self.porig[i]['data']['k']<=kmax) & (self.porig[i]['data']['k']>=kmin)]
        else:
            if kmin is None:
                kmin = self.peaks[peaks]['data']['k'].min()
            if kmax is None:
                kmax = self.peaks[peaks]['data']['k'].max()

            self.peaks[peaks]['data'] = self.porig[peaks]['data'][(self.porig[peaks]['data']['k']<=kmax)
                                                                  & (self.porig[peaks]['data']['k']>=kmin)]
    
    def set_datarange_points(self, plim=None, peaks='all'):
        
        pmin, pmax = plim
        
        if peaks == 'all':
            for i, p in enumerate(self.peaks):
                p['data'] = self.porig[i]['data'][pmin:pmax]
        else:
            self.peaks[peaks]['data'] = self.porig[peaks]['data'][pmin:pmax]
    
    def make_fitmodel(self, n_components=1, verbose=False):
        self.fitmodel = lf.Model(make_exp_decay(n_components))
        self.params = self.fitmodel.make_params()
        if verbose:
            self.params.pretty_print()
        
    def set_param(self, par, *args):
        self.params[par].set(*args)
        print(self.params[par])
    
    def fit_diffusion(self, peak=0, weights=None, report=True):
        '''
        standard weight function: [1/n for n in data]
        '''
        p = self.peaks[peak]
        if weights == '1/n':
            weights = np.array([1/n for n in p['data']['Expt']])
        if weights is None:
            weights = None
        
        self.results['peak %i'%peak] = self.fitmodel.fit(p['data']['Expt'], k=p['data']['k'], params=self.params, weights=weights)
        
        if report:
            print(self.results['peak %i'%peak].fit_report())
            
    def plot_fit(self, peak=0, ylim=None, report=False, xtext=0.5, ytext=0.5):
        
        k = self.peaks[peak]['data']['k']
        ydata = self.peaks[peak]['data']['Expt']
        yfit = self.results['peak %i'%peak].best_fit
        yinit = self.results['peak %i'%peak].init_fit
        
        #Plotten der Daten
        fig = plt.figure()#figsize=(8,12))
        ax = fig.add_subplot(111)
        ax.plot(k, ydata, 'x', label='data')
        ax.plot(k, yfit, label='best fit')
        ax.plot(k, yinit, '--', label='initial guess')
        #Beschriftungen
        ax.set_ylabel('$I \cdot I_0^{-1}$')
        ax.set_xlabel('$k$ / (s·m$^{-2}$)')
        ax.set_yscale('log')
        if self.title is not None:
            ax.set_title(self.title)

        #skalierung
        if ylim is None:
            ax.set_ylim(min(ydata)-abs(min(ydata))/2,2)
        else:
            ax.set_ylim(ylim)
        
        if report:
            ax.text(xtext,ytext,self.results['peak %i'%peak].fit_report(), transform=ax.transAxes)
        
        ax.legend()
        
        return fig
