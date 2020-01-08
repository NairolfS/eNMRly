## coding: utf-8
# This is the eNMR class. THE library for the evaluation of bruker-eNMR-spectra based on the VC-PowerSource of the
# Schönhoff working group.
# It works with the volt-increment method which calculates the respective voltage with the VC-list
# Further Implementation can be asked for at f_schm52@wwu.de or nairolf.schmidt@gmail.com
 
import matplotlib as mpl
# .use("pgf")
# pgf_with_custom_preamble = {
#     "pgf.rcfonts": False,    # don't setup fonts from rc parameters
#     "font.family" : "serif",
#     'text.latex.unicode' : True,
#     'text.usetex' : True,
#     'pgf.preamble' : r'\usepackage{unicode-math}'
# }
# mpl.rcParams.update(pgf_with_custom_preamble)

import matplotlib.pyplot as plt
import nmrglue as ng
import numpy as np
import pandas as pd
import scipy.optimize
from re import findall
from IPython.display import clear_output
from sklearn.linear_model import huber as hub


"""
This is the eNMR-module. It is created to eNMR data-evaltuation from Bruker spectrometers.

Tutorial: I will need to add some content at this point.

"""

def relegend(fig, new_labels, **kwargs):
    '''
    Takes a figure with a legend, gives them new labels (as a list) 
    
    **kwargs: ncol, loc etc.
        ncol: number of columns
        loc: location of the legend (see matplotlib documentation)
    '''
    ax = fig.gca()
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, new_labels, **kwargs)

def calibrate_x_axis(fig, val, majorticks=1, new_xlim=None):
    """
    this function takes a pyplot figure and shifts the x-axis values by val.
    majorticks defines the distance between the major ticklabels.
    """
    ax = fig.gca()
    xlim = ax.get_xlim()
    
    if xlim[0] > xlim[-1]:
        x1, x2 = ax.get_xlim()[::-1]
    else:
        x1, x2 = ax.get_xlim()
        
    ax.set_xticks(np.arange(x1, x2+1, majorticks)-val%1)
    ax.set_xticklabels(['%.1f'%f for f in ax.get_xticks()+val])
    if new_xlim is None:
        ax.set_xlim(xlim)
    else:
        ax.set_xlim(new_xlim)
        
    return spec

def phc_normalized(phc0, phc1):
    """
    nifty function to correct the zero order phase correction
    when phasing 1st order distortions. 
    phc0 -= phc1/(np.pi/2)
    
    returns: (phc0, phc1)
    
    best use in conjunctoin with self.proc()
   
    """
    
    #phc0 -= phc1/(np.pi/2)
    phc0 -= phc1/(np.pi)
    return phc0, phc1

class Measurement(object):
    """
    This is the main class for the analysis of eNMR raw spectra obtained on Bruker spectrometers with given settings.
    if creating an object from an eNMR measurement creates an error, Uink might have failed to be set and needs to be set manually instead.
    
    path:
        relative or absolute path to the measurements folder
    measurement:
        the to the experiment corresponding EXPNO
    alias:
        Here you can place an individual name relevant for plotting. If None, the path is taken instead.
    """

    def __init__ (self, path, measurement, alias=None,  linebroadening=5, n_zf_F2=2**14):
        self.path = path
        self.expno = str(measurement)
        self.dateipfad = self.path+self.expno
        self.alias = alias
        self.name = self.dateipfad.split(sep='/')[-2]+'/'+self.expno
        self.lineb = linebroadening
        
        ## correction factor U_0 for normalized spectra
        #self.U_0 = 0
        
        # takes the title page to extract the volt increment
        try:
            title = open(self.dateipfad+"/pdata/1/title").read()
            # reformatting the title_page removing excess of newlines (\n)

        except UnicodeDecodeError:
            title = open(self.dateipfad+"/pdata/1/title", encoding='ISO-8859-15').read()
            
        title = findall('.+', title)              
        self.title_page = self.name+'\n'

        for n in title:
            self.title_page += n+'\n'        

        # read in the bruker formatted data
        self.dic, self.data = ng.bruker.read(self.dateipfad)
        self.pdata = ng.bruker.read_procs_file(self.dateipfad+"/pdata/1")
        # original dataset
        self.data_orig = self.data

        # Berechnung der limits der ppm-skala
        self._ppm_r = self.pdata["procs"]["OFFSET"]
        self._ppm_l = -(self.dic["acqus"]["SW"]-self.pdata["procs"]["OFFSET"])
        self._ppmscale = np.linspace(self._ppm_l, self._ppm_r, n_zf_F2)  # np.size(self.data[0,:]))
        self.ppm = self._ppmscale

        # Bestimmung der dictionaries für die NMR-Daten aus den Messdateien
        # --> wichtig für die Entfernung des digitalen Filters
        self.udic = ng.bruker.guess_udic(self.dic, self.data)
        self._uc = ng.fileiobase.uc_from_udic(self.udic)

        # getting some other variables
        self.TD = self.dic["acqus"]["TD"]
        self.fid = self.data
        self.n_zf_F2 = n_zf_F2

        # the gamma_values in rad/Ts
        gamma_values = {'1H':26.7513e7,
                        '7Li': 10.3962e7,
                        '19F': 25.1662e7}

        self.gamma = gamma_values[self.dic["acqus"]["NUC1"]]
        # conversion from rad in °
        self.gamma = self.gamma/2/np.pi*360
        
        self.nuc = self.dic["acqus"]["NUC1"]
        
        # initialize dictionary for linear regression results
        self.lin_res_dic = {}
        #self.dependency = None

        # END __INIT__

    def calibrate_ppm(self, ppmshift):
        """
        calibrate ppm-scale by adding the desired value.

        :param ppmshift: value added to ppm-scale
        :return: nothing
        """
        self.ppm = self._ppmscale + ppmshift

    def proc(self, linebroadening=None, phc0=0, phc1=0, zfpoints=None, xmin=None, xmax=None, cropmode="percent"):
        """
        processes the spectral data by:
            - removing digital filter
            - create separate fid data
            - linebroadening on spectral data
            - zero filling
            - fourier transformation
            
        crops the data after fft set to xmin and xmax on the x-axis and returns the value
        when xmin and xmax or not both None
        
        xmin, xmax:
            min and maximum x-values to crop the data for processing (and display)
            can take relative or absolute values depending on the cropmode.
        
        cropmode: changes the x-scale unit
            "percent": value from 0% to 100% of the respective x-axis-length --> does not fail
            "absolute": takes the absolute values --> may fail            
        
        :return: nothing
        """
        
        
        if linebroadening is None:
            linebroadening = self.lineb
        
        if zfpoints is not None:
            zfp = zfpoints
        else:
            zfp = self.n_zf_F2
        _lineb = linebroadening
        
        # remove the digital filter
        self.data = ng.bruker.remove_digital_filter(self.dic, self.data)

        # process the spectrum
        # linebroadening
        self.data = ng.proc_base.em(self.data, lb=_lineb/self.dic["acqus"]["SW_h"])

        # zero fill to 32768 points
        try:
            self.data = ng.proc_base.zf_size(self.data, zfp)
        except ValueError:
            zfp = 2**15
            self.data = ng.proc_base.zf_size(self.data, zfp)
        # Fourier transform
        self.data = ng.proc_base.fft(self.data)
        
        # Phasecorrection
        self.data = ng.proc_autophase.ps(self.data, phc0, phc1)
        #correct ppm_scale
        self._ppmscale = np.linspace(self._ppm_l, self._ppm_r, zfp)  # np.size(self.data[0,:]))
        self.ppm = self._ppmscale

        self.data_orig = self.data

        if (xmin is not None) or (xmax is not None):
            self.data = self.set_spectral_region(xmin=xmin, xmax=xmax, mode=cropmode)

    def plot_fid(self, xmax=None, xmin=0, step=1):
        """
        plots every n-th(step) fid and scales the time axis with xmax and xmin
        
        :returns: figure
        """
        
        _xmax = self.TD/2 if xmax is None else xmax
        _xmin = xmin

        fig, ax = plt.subplots()
        
        for n in range(len(self.data[::step, 0])):
            ax.plot(self.fid[n, ::1].real)

        ax.set_xlim((_xmin, _xmax))
        ax.set_xlabel("data points")
        ax.ticklabel_format(style='sci', scilimits=(0,0), axis='y')
        ax.legend([x*step for x in range(len(self.data[::step, 0]))],
                   ncol=3,
                   title="row",
                   loc=1)
        #plt.show()
        return fig

    def plot_spec(self, row, xlim=None, figsize=None, invert_xaxis=True, sharey=True):#, ppm=True):
        """
        plots row 0 and row n in the range of xmax and xmin
        
        :returns: figure
        """
        

        _max = None if xlim is None else xlim[0]
        _min = None if xlim is None else xlim[1]
        
        if type(xlim) is not list:
            fig = plt.figure(figsize=figsize)
            ax = fig.add_subplot(111)
        elif type(xlim) == list:
            fig, ax = plt.subplots(ncols=len(xlim), nrows=1, figsize=figsize, sharey=sharey)
        
        _min = self.ppm[0] if xlim is None else xlim[1]
        _max = self.ppm[-1] if xlim is None else xlim[0]
        
        if type(xlim) is not list:
            if type(row) ==list:
                for r in row:
                    ax.plot(self.ppm, self.data[r, ::1].real, label='row %i'%r)
            else:
                ax.plot(self.ppm, self.data[row, ::1].real, label='row %i'%row)
            ax.legend()
            ax.set_xlim(xlim)
            #ax.set_title("this is row %i at %.0f V" % (row, self.eNMRraw.loc[row, "U / [V]"]))
            ax.set_xlabel("$\delta$ / ppm")
            ax.set_ylabel("intensity / a.u.")

            ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
            if invert_xaxis:
                xlimits = ax.get_xlim()
                ax.set_xlim(xlimits[::-1])
        
        elif type(xlim) == list:
            for axis, xlim in zip(ax,xlim):
                if type(row) ==list:
                    for r in row:
                        axis.plot(self.ppm, self.data[r, ::1].real, label='row %i'%r)
                else:
                    axis.plot(self.ppm, self.data[row, ::1].real, label='row %i'%row)
                
                axis.set_xlim(xlim)
                #ax.set_title("this is row %i at %.0f V" % (row, self.eNMRraw.loc[row, "U / [V]"]))
                axis.set_xlabel("$\delta$ / ppm")
                axis.set_ylabel("intensity / a.u.")

                axis.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
                if invert_xaxis:
                    xlimits = axis.get_xlim()
                    axis.set_xlim(xlimits[::-1])
            #fig.legend()
        return fig
    
    def plot_spec_1d(self, xlim=None, figsize=None, invert_xaxis=True):#, ppm=True):
        """
        plots row 0 and row n in the range of xmax and xmin
        
        :returns: figure
        """
        _max = None if xlim is None else xlim[0]
        _min = None if xlim is None else xlim[1]

        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)
        
        _min = self.ppm[0] if xlim is None else xlim[1]
        _max = self.ppm[-1] if xlim is None else xlim[0]
        
        ax.plot(self.ppm, self.data[::1].real)
        
        #ax.legend()
        ax.set_xlim(xlim)
        #ax.set_title("this is row %i at %.0f V" % (row, self.eNMRraw.loc[row, "U / [V]"]))
        ax.set_xlabel("$\delta$ / ppm")
        ax.set_ylabel("intensity / a.u.")

        ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
        if invert_xaxis:
            xlimits = ax.get_xlim()
            ax.set_xlim(xlimits[::-1])
        return fig

    def set_spectral_region(self, xmin, xmax, mode='absolute', ppm=True, original_data=True):
        """
        crops the data of the object set to xmin and xmax on the x-axis and returns the value
        
        mode: changes the x-scale unit
            "percent": value from 0% to 100% of the respective x-axis-length --> does not fail
            "absolute": takes the absolute values --> may fail
        
        :returns: cropped_data, cropped_ppmscale
        """
        
        if (xmin is not None) or (xmax is not None):

            if mode == "percent":
                ppmmin = xmin/100*self.data[0, :].size
                ppmmax = xmax/100*self.data[0, :].size
                if original_data:
                    self.data = self.data_orig[:, int(ppmmin):int(ppmmax)]
                else:
                    self.data = self.data[:, int(ppmmin):int(ppmmax)]
                self.ppm = self._ppmscale[int(ppmmin):int(ppmmax)]

            elif mode == "absolute":
                if ppm:
                    print('xmin: %i' % xmin)
                    print('xmax: %i' % xmax)
                    xmax = np.where(self._ppmscale <= xmax)[0][-1]  # if xmax is not None else -1
                    xmin = np.where(self._ppmscale >= xmin)[0][0]  # if xmin is not None else 0
                    print('xmin: %i' % xmin)
                    print('xmax: %i' % xmax)
                    print(self._ppmscale)
                
                if original_data:
                    self.data = self.data_orig[:, xmin:xmax]
                else:
                    self.data = self.data[:, xmin:xmax]
                    
                self.ppm = self._ppmscale[xmin:xmax]
            else:
                print('oops, you mistyped the mode')
        return self.data, self._ppmscale[xmin:xmax]

########################################################################################################################

class eNMR_Methods(Measurement):
    """
    This is the subclass of Masurement() containing all methods
    
    path:
        relative or absolute path to the measurements folder
    measurement:
        the to the experiment corresponding EXPNO
    alias:
        Here you can place an individual name relevant for plotting. If None, the path is taken instead.
    """
    # def get_xaxis(self):
    #
    #     if self.dependency is not None:
    #         return {
    #             "G": self.eNMRraw["g in T/m"],
    #             "U": self.eNMRraw["U / [V]"]
    #         }[self.dependency.upper()]
    #     else:
    #         print("an error occured, the dependency is None")
    def __init__(self, path, measurement, alias=None,  linebroadening=5):
        super().__init__(path, measurement, alias=None,  linebroadening=5)
        
        self._x_axis = {"U": "U / [V]",
                   "G": "g in T/m",
                   "I": "I / mA",
                   "RI": "RI / V"
                   }[self.dependency.upper()]
    
    def __repr__(self):
        return '''%s, expno %s, Delta = %.1fms, ppm range: %.1f to %.1f
    delta= %.1fms, g= %.3f T/m, e-distance=%.0fmm'''%(
        self.nuc, self.expno, self.Delta*1000, 
        self.ppm[0], self.ppm[-1],
        self.delta*1000, self.g, self.d*1000
        )
    
    def __getitem__(self, key):
        if type(key) == str:
            return self.eNMRraw[key]
        elif type(key) == int:
            return (self.ppm, self.data[key])
        
    #def __add__(self, other):
        
    
    def spam_and_eggs(self):
        """
        For Class inheritance test purposes
        """
        #self.x_var = {None:None,
                #"U": "U / [V]",
                #"G": "g in T/m"
        #}
        print(self.x_var)
        print("SPAM and EGGS!")
        

    def autophase(self,
                  returns=False,
                  method="acme",
                  progress=False,
                  period_compensation=True,
                  normalize=True):
        """
        analyzes the phase of the spectral data and returns phased data

        at this point no data cropping --> full spectrum is analyzed

        returns: if True, returns the raw phased Data. If false, returns nothing

        method: chooses the method for the phase correction
            "acme": standard method using entropy minimization
            "difference": --> _ps_abs_difference_score()
                minimizes the linear difference to the first spectrum by fitting p0
            "sqdifference": --> _ps_sq_difference_score()
                minimizes the square difference to the first spectrum by fitting p0

        progress:
            prints the progress. In case you are dealing with large datasets

        period_compensation:
            corrects for the 360 degree error that occasionaly occurs since a phase of 0 and 360
            is equivalent and indistinguishable

        normalize:
            normalizes all phase data to 0 degrees for 0V. This compensates for different reference 
            phases during the aquisition and phase analysis.
        """
        if self.dependency.upper() == 'G':
            normalize = False

        data_ph = np.array([])
        for n in range(len(self.data[:, 0])):
            if progress:
                print("row %i / %i" % (n+1, len(self.data[:, 0])))

            #####################################
            # using the data with the algorithm
            _val = self._autops(self.data[n, :], _fnc=method)
            #####################################

            data_ph = np.append(data_ph, _val)
                
            if method == 'acme':
                self.eNMRraw.loc[n, "ph0acme"] = _val[0]  # *-1
                if progress:
                    clear_output() # keeps the output clean. remove for more info on iteration steps etc.
            elif method == 'difference':
                self.eNMRraw.loc[n, "ph0diff"] = _val[0]  # *-1
                if progress:
                    clear_output() # keeps the output clean. remove for more info on iteration steps etc.
            else:
                self.eNMRraw.loc[n, "ph0sqdiff"] = _val[0]  # *-1
                if progress:
                    clear_output() # keeps the output clean. remove for more info on iteration steps etc.
       
        #corr_enmr = self.eNMRraw.sort_values("U / [V]")
        corr_enmr = self.eNMRraw.sort_values(self._x_axis)
        
        if method == 'acme':
            if period_compensation:
                for m in range(corr_enmr["ph0acme"].size):
                    if corr_enmr["ph0acme"].iloc[m]-corr_enmr["ph0acme"].iloc[abs(m-1)] < -300:
                        corr_enmr["ph0acme"].iloc[m] += 360
                    elif corr_enmr["ph0acme"].iloc[m]-corr_enmr["ph0acme"].iloc[abs(m-1)] > 300:
                        corr_enmr["ph0acme"].iloc[m] -= 360
                self.eNMRraw = corr_enmr
        
            if normalize:
                #self.U_0 = corr_enmr[corr_enmr["U / [V]"] == 0]["ph0acme"][0]
                self.U_0 = corr_enmr[corr_enmr[self._x_axis] == 0]["ph0acme"].iloc[0]
                for m in range(corr_enmr["ph0acme"].size):
                    corr_enmr["ph0acme"].iloc[m] -= self.U_0
                self.eNMRraw = corr_enmr

            #self.eNMRraw["ph0acmereduced"] = corr_enmr['ph0acme']/(self.eNMRraw['g in T/m'][0]*self.Delta*self.delta*self.gamma*self.d)
            self.eNMRraw["ph0acmereduced"] = corr_enmr['ph0acme']*self.d/(self.eNMRraw['g in T/m'][0]*self.Delta*self.delta*self.gamma)
        elif method == 'difference':
            if period_compensation:
                for m in range(corr_enmr["ph0diff"].size):
                    if corr_enmr["ph0diff"].iloc[m]-corr_enmr["ph0diff"].iloc[abs(m-1)] < -300:
                        corr_enmr["ph0diff"].iloc[m] += 360
                    elif corr_enmr["ph0diff"].iloc[m]-corr_enmr["ph0diff"].iloc[abs(m-1)] > 300:
                        corr_enmr["ph0diff"].iloc[m] -= 360
                self.eNMRraw = corr_enmr
        
            if normalize:
                self.U_0 = corr_enmr[corr_enmr[self._x_axis] == 0]["ph0diff"][0]
                for m in range(corr_enmr["ph0diff"].size):
                    corr_enmr["ph0diff"].iloc[m] -= self.U_0
                self.eNMRraw = corr_enmr

            #self.eNMRraw["ph0diffreduced"] = corr_enmr['ph0diff']/(self.eNMRraw['g in T/m'][0]*self.Delta*self.delta*self.gamma*self.d)
            self.eNMRraw["ph0diffreduced"] = corr_enmr['ph0diff']*self.d/(self.eNMRraw['g in T/m'][0]*self.Delta*self.delta*self.gamma)
        else:
            if period_compensation:
                for m in range(corr_enmr["ph0sqdiff"].size):
                    if corr_enmr["ph0sqdiff"].iloc[m]-corr_enmr["ph0sqdiff"].iloc[abs(m-1)] < -300:
                        corr_enmr["ph0sqdiff"].iloc[m] += 360
                    elif corr_enmr["ph0sqdiff"].iloc[m]-corr_enmr["ph0sqdiff"].iloc[abs(m-1)] > 300:
                        corr_enmr["ph0sqdiff"].iloc[m] -= 360
                self.eNMRraw = corr_enmr
        
            if normalize:
                self.U_0 = corr_enmr[corr_enmr[self._x_axis] == 0]["ph0sqdiff"][0]
                for m in range(corr_enmr["ph0sqdiff"].size):
                    corr_enmr["ph0sqdiff"].iloc[m] -= self.U_0
                self.eNMRraw = corr_enmr

            self.eNMRraw["ph0sqdiffreduced"] = corr_enmr['ph0sqdiff']*self.d/(self.eNMRraw['g in T/m'][0]*self.Delta*self.delta/1000*self.gamma)

        print("done, all went well")
        
        if returns is True:
            return data_ph  # , data_spec

    def _autops(self, data, _fnc="acme", p0=0.0):  # , p1=0.0):
        """
        FS: modified for eNMR purpose. optimizes only p0
        ----------
        Automatic linear phase correction

        Parameters
        ----------
        data : ndarray
            Array of NMR data.
        _fnc : str or function
            Algorithm to use for phase scoring. Built in functions can be
            specified by one of the following strings: "acme", "peak_minima"
        p0 : float
            Initial zero order phase in degrees.
        p1 : float
            Initial first order phase in degrees.

        Returns
        -------
        ndata : ndarray
            Phased NMR data.

        """
        self._fnc = _fnc
        self._p0 = p0
        # self.p1 = p1
        
        if not callable(_fnc):
            self._fnc = {
                # 'peak_minima': _ps_peak_minima_score,
                'acme': self._ps_acme_score,
                'difference': self._ps_abs_difference_score,
                'sqdifference': self._ps_sq_difference_score
            }[self._fnc]

        # self.opt ist die optimierte Phase für das Spektrum.
        self.opt = self._p0  # [self._p0, self.p1]
        self.opt = scipy.optimize.fmin(self._fnc, x0=self.opt, args=(data, ), disp=0)

        # self.phasedspc = ng.proc_base.ps(data, p0=self.opt[0], p1=self.opt[1])

        return self.opt  # self.phasedspc, self.opt

    @staticmethod
    def _ps_acme_score(ph, data):
        """
        FS: modified for eNMR purpose. optimizes only p0
        ------------
        Phase correction using ACME algorithm by Chen Li et al.
        Journal of Magnetic Resonance 158 (2002) 164-168

        using only the first derivative for the entropy

        Parameters
        ----------
        pd : tuple
            Current p0 and p1 values
        data : ndarray
            Array of NMR data.

        Returns
        -------
        score : float
            Value of the objective function (phase score)

        """
        stepsize = 1
        
        # s0 --> initial spectrum?
        s0 = ng.proc_base.ps(data, p0=-ph, p1=0)  # , p1=phc1 --> p1=0 lets the algorithm optimize only p0
        data = np.real(s0)

        # Calculation of first derivatives --> hier wird das absolute Spektrum erzeugt
        ds1 = np.abs((data[1:]-data[:-1]) / (stepsize*2))
        p1 = ds1 / np.sum(ds1)

        # Calculation of entropy
        p1[p1 == 0] = 1  # was macht diese Zeile? das Verstehe ich noch nicht richtig

        h1 = -p1 * np.log(p1)
        h1s = np.sum(h1)

        # Calculation of penalty
        pfun = 0.0
        as_ = data - np.abs(data)
        sumas = np.sum(as_)

        if sumas < 0:
            pfun = pfun + np.sum((as_/2) ** 2)

        p = 1000 * pfun

        return h1s + p
    
    def _ps_abs_difference_score(self, ph, data):
        """
        FS: modified for eNMR purpose. optimizes only p0
        ------------
        Parameters
        ----------
        pd : tuple
            Current p0 and p1 values
        data : ndarray
            Array of NMR data.

        Returns
        -------
        score : float
            Value of the objective function (phase score)
        """
       
        s0 = ng.proc_base.ps(data, p0=-ph, p1=0)  # , p1=phc1 --> p1=0 lets the algorithm optimize only p0
        phasedspec = np.real(s0)

        penalty = np.sum(np.abs(self.data[0, :].real - phasedspec))
        
        return penalty
    
    def _ps_sq_difference_score(self, ph, data):
        """
        FS: modified for eNMR purpose. optimizes only p0
        ------------
        Parameters
        ----------
        pd : tuple
            Current p0 and p1 values
        data : ndarray
            Array of NMR data.

        Returns
        -------
        score : float
            Value of the objective function (phase score)
        """
       
        s0 = ng.proc_base.ps(data, p0=-ph, p1=0)  # , p1=phc1 --> p1=0 lets the algorithm optimize only p0
        phasedspec = np.real(s0)

        penalty = np.sum(np.square(self.data[0,:].real - phasedspec))  # *1/n-1 wäre die Varianz,
        # ist hier egal, da alle Spektren die gleiche Anzahl an Punkten haben.
        
        return penalty
    
    def analyze_intensity(self, data='cropped', ph_var='ph0acme', normalize=True, ylim=None):
        """
        uses the phase data information to rephase and integrate all spectra and plot a comparison
        stores the intensity information in the measurement folder
        
        data:
            'orig': self.data_orig
            'cropped': self.data

        return: fig
        """
        _data = {'orig': self.data_orig,
                'cropped': self.data}[data]

        # x_axis = {}[self.dependency]

        # the correction factor for the normalization is added again.
        self.phased = [ng.proc_base.ps(_data[n, :], p0=-(self.eNMRraw.loc[n, ph_var]))# + self.U_0))
                  for n in range(len(_data[:, 0]))]

        intensity = np.array([self.phased[n].real.sum() for n in range(len(_data[:, 0]))])
        
        if normalize:
            intensity /= intensity[0]
                
        #u = [self.eNMRraw.loc[i, 'U / [V]'] for i, n in enumerate(self.eNMRraw['U / [V]'])]
        u = [self.eNMRraw.loc[i, self._x_axis] for i, n in enumerate(self.eNMRraw[self._x_axis])]

        intensity_data = pd.DataFrame()
        intensity_data["U"] = u
        intensity_data['intensity'] = intensity
        intensity_data['ph'] = self.eNMRraw[ph_var]# + self.U_0

        self.intensity_data = intensity_data

        fig = plt.figure(figsize=(8, 6))
        _ax = plt.subplot(221)
        _ax.scatter(intensity_data['U'], intensity_data['intensity'], c='k')
        _ax.set_ylabel('intensity / a.u.')
        if normalize and (ylim is None):
            _ax.set_ylim(0,1.05)
        elif normalize:
            _ax.set_ylim(*ylim)
        _bx = plt.subplot(222, sharey=_ax)
        _bx.plot(intensity_data['intensity'], 'ok')

        _cx = plt.subplot(223, sharex=_ax)
        _cx.scatter(intensity_data['U'], intensity_data['ph'], c='k')
        _cx.set_xlabel('$U$ / V')
        _cx.set_ylabel('$\t{\Delta}\phi$ / °')

        _dx = plt.subplot(224, sharex=_bx)
        _dx.plot(intensity_data['ph'], 'ok')
        _dx.set_xlabel('vc')

        fig.savefig(self.path+'intensity_plot_'+self.expno+".pdf")

        intensity_data.to_csv(self.path+'intensity_data_'+self.expno+".csv")

        return fig#, intensity_data

    def lin_huber(self, epsilon=1.35, ulim=None, y_column='ph0acme'):
        """
        robust linear regression method from scikit-learn module based on the least-square method with an additional threshhold (epsilon) for outlying datapoints
        outlying datapoints are marked as red datapoints
        
        epsilon:
            threshhold > 1
        ulim: 
            tuple defining the voltage limits for the regression e.g. ulim = (-100, 100)
        y_column:
            column(keyword) to be analyzed from the eNMRraw dataset
        
        stores results in lin_res_dic[y_column]
        :returns: nothing
        """

        # select x-axis
        #self._x_axis = {"U": "U / [V]",
                   #"G": "g in T/m"
                   #}[self.dependency.upper()]

        # convert data
        self._eNMRreg = self.eNMRraw[[self._x_axis, y_column]].sort_values(self._x_axis)
        
        # setting the axis for regression
        if ulim is None:
            self.umin = min(self.eNMRraw[self._x_axis])
        else:
            self.umin = ulim[0]
            
        if ulim is None:
            self.umax = max(self.eNMRraw[self._x_axis])
        else:
            self.umax = ulim[1]

        self._npMatrix = np.matrix(self._eNMRreg[(self.eNMRraw[self._x_axis] <= self.umax)
                                                 == (self.eNMRraw[self._x_axis] >= self.umin)])

        self._X_train, self._Y_train = self._npMatrix[:, 0], self._npMatrix[:, 1]
        
        # regression object
        self.huber = hub.HuberRegressor(epsilon=epsilon)
        self.huber.fit(self._X_train, self._Y_train)
        
        # linear parameters
        self.m = self.huber.coef_  # slope
        self.b = self.huber.intercept_  # y(0)
        self._y_pred = self.huber.predict(self._X_train)
        self._y_pred = self._y_pred.reshape(np.size(self._X_train), 1)
        
        # drop the outliers
        self._outX_train = np.array(self._X_train[[n == False for n in self.huber.outliers_]])
        self._outY_train = np.array(self._Y_train[[n == False for n in self.huber.outliers_]])
        self._outY_pred = np.array(self._y_pred[[n == False for n in self.huber.outliers_]])
        
        # mark outliers in dataset
        # self._inliers = [n is not True for n in self.huber.outliers_]

        self.eNMRraw["outlier"] = True

        for n in range(len(self._npMatrix[:, 0])):
            self.eNMRraw.loc[self.eNMRraw[self._x_axis] == self._npMatrix[n, 0], "outlier"] = self.huber.outliers_[n]

        # calculation of the slope deviation
        _sig_m_a = np.sqrt(np.sum((self._outY_train-self._outY_pred)**2)/(np.size(self._outY_train)-2))
        _sig_m_b = np.sqrt(np.sum((self._outX_train-self._outX_train.mean())**2))
        self.sig_m = _sig_m_a/_sig_m_b

        # debug
        #print(self.sig_m)

        # R^2
        self.r_square = self.huber.score(self._outX_train, self._outY_train)
        
        self.lin_res_dic[y_column] = {'b': self.b,
                                   'm': self.m,
                                   'r^2': self.r_square,
                                   'x': np.array(self._X_train.tolist()).ravel(),
                                   'y': self._y_pred.ravel(),
                                   'sig_m': self.sig_m}

    def lin_display(self, ylim=None, show_slope_deviation=True, n_sigma_displayed=1, dpi=500, y_column='ph0acme', textpos=(0.5,0.15), extra_note=''):
        """
        displays the linear huber regression
        If there is an alias available from measurement object, it will replace the path in the title
        
        ylim:
            set limits of the y-axis
        
        show_slope_deviation:
            display the standard deviation
        n_sigma_displayed:
            multiplicator of the displayed standard deviation
        y_column:
            column(keyword) to be analyzed from the eNMRraw dataset
        textpos:
            tuple for the textposition
        extra_note:
            added to the text
        dpi:
            adjusts the output dpi to the required value
        
        :returns: figure
        """
        
        textx, texty = textpos
        #_x_axis = {"U":"U / [V]", "G":"g in T/m"}
        #self._x_axis = {"U":"U / [V]", "G":"g in T/m"}[self.dependency]

        print("formula: y = {0}x + {1}".format(self.m,self.b))
        
        # create figure
        fig_enmr = plt.figure()

        # sublot phase data
        _ax = fig_enmr.add_subplot(111)
        
        # color format for outliers
        colors = ["r" if n else "k" for n in self.eNMRraw.sort_values(self._x_axis)['outlier']]
        
        _ax.scatter(x=np.ravel(self._eNMRreg[self._x_axis]),
                    y=np.ravel(self._eNMRreg[y_column]),
                    marker="o",
                    c=colors)
        _ax.set_ylim(ylim)

        # format the data for plotting
        _xdata = np.ravel(self._X_train)
        _ydata = np.ravel(self._y_pred)
        
        # Plot the regression
        _ax.plot(_xdata, _ydata, "r-")

        if show_slope_deviation:
            _ax.fill_between(_xdata, _xdata*(self.m+n_sigma_displayed*self.sig_m)+self.b,
                                     _xdata*(self.m-n_sigma_displayed*self.sig_m)+self.b,
                                     alpha=0.5,
                                     facecolor="blue")

        # make title
        if self.alias is None:
            title_printed = r'%s'%((self.path+self.expno).split("/")[-2]+", EXPNO: "+self.expno+extra_note)
#             plt.title('LiTFSIDOLDME') # debugging
        else:
            title_printed = self.alias+", EXPNO: "+self.expno+extra_note
#             plt.title(self.alias+", EXPNO: "+self.expno+extra_note)
#             plt.title('test2') # for debugging purposes
        plt.title(title_printed.replace('_',r' '))
    
        if self.dependency.upper() == "U":
            plt.xlabel("$U$ / V")
        elif self.dependency.upper() == "G":
            plt.xlabel("$g$ / $($T$\cdot$m$^{-1})$")
        elif self.dependency.upper() == 'I':
            plt.xlabel("$I$ / mA")
        elif self.dependency.upper() == 'RI':
            plt.xlabel("$(R \cdot I)$ / V")

        plt.ylabel("$\Delta\phi$ / °")
        
        # plotting the Textbox
        plt.text(textx, texty,
                 "y = %.4f $\cdot$ x + %4.2f\n$R^2$=%4.3f; $\sigma_m=$%4.4f"%(self.m,
                                                                              self.b,
                                                                              self.r_square,
                                                                              self.sig_m),
                 fontsize=14,
                 bbox={'facecolor':'white', 'alpha':0.7,'pad':10},
                 horizontalalignment='center',
                 verticalalignment='center',
                 transform=_ax.transAxes)

        return fig_enmr

    def lin_results_display(self, cols, regression=True, normalize=True, colors=None, markers=None, fig=None, x_legend=1.1, y_legend=1.0, ncol_legend=2, ymin=None, ymax=None, figsize=None):
        '''
        displays the phase results including the regression. Can take a previous figure from a different eNMR measurement in order to compare results
        the relegend() function can be used on these graphs to reprint the legend() for publishing
        
        cols:
            list of keywords for dataselection from the eNMRraw table
        colors:
            list of colors or a colormap --> see matplotlib documentation
        
        :returns: figure
        '''
        
        #self._x_axis = {"U":"U / [V]", "G":"g in T/m"}[self.dependency.upper()]
        
        if fig is None:
            fig = plt.figure(figsize=figsize)
        else:
            fig = fig

        ax = fig.add_subplot(111)

        if colors is None:
            prop_cycle = plt.rcParams['axes.prop_cycle']
            colors = prop_cycle.by_key()['color']
        else:
            colors = colors
        
        if markers is None:
            markers = ['o', '^', 's', '+', '*', 'D', 'v', 'x', '<']
        else:
            markers = markers

        message = {
            'ph0acme': 'Autophase with entropy minimization',
            'ph0diff': 'Autophase with linear difference minimization',
            'ph0sqdiff': 'Autophase with square difference minimization'
            }
        
        if type(cols) == list:
            for i, col in enumerate(cols):
                if normalize:
                    corr = self.eNMRraw.loc[self.eNMRraw['U / [V]']==0, col].iloc[0]
                else:
                    corr = 0
                    
                try:
                    ax.errorbar(self.eNMRraw[self._x_axis], self.eNMRraw[col]-corr, yerr=self.eNMRraw['%s_err'%col], fmt='o', label=message[col], c=colors[i], marker=markers[i])
                except KeyError:
                    if col == 'ph0acme':
                        ax.errorbar(self.eNMRraw[self._x_axis], self.eNMRraw[col]-corr, fmt='o', label=message[col], c=colors[i], marker=markers[i])
                    else:
                        ax.errorbar(self.eNMRraw[self._x_axis], self.eNMRraw[col]-corr, yerr=self.eNMRraw['%s_err'%col], fmt='o', label='fitted data %s'%col, c=colors[i], marker=markers[i])
                except ValueError:
                    ax.errorbar(self.eNMRraw[self._x_axis], self.eNMRraw[col]-corr, fmt='o', label=message[col], c=colors[i], marker=markers[i])
                if regression:
                    ax.plot(self.lin_res_dic[col]['x'], self.lin_res_dic[col]['y']-corr, '--', label='%s lin regression'%col, c=colors[i], marker=None)
        #if type(cols) == list:
            #for i, col in enumerate(cols):
                #try:
                    #ax.errorbar(self._x_axis, col, yerr='%s_err'%col, fmt='o', data=self.eNMRraw, label=message[col], c=colors[i], marker=markers[i])
                #except KeyError:
                    #ax.errorbar(self._x_axis, col, yerr='%s_err'%col, fmt='o', data=self.eNMRraw, label='fitted data %s'%col, c=colors[i], marker=markers[i])
                #except ValueError:
                    #ax.errorbar(self._x_axis, col, fmt='o', data=self.eNMRraw, label=message[col], c=colors[i], marker=markers[i])
                #if regression:
                    #ax.plot(self.lin_res_dic[col]['x'], self.lin_res_dic[col]['y'], '--', label='%s lin regression'%col, c=colors[i], marker=None)
        else:
            if normalize:
                corr = self.eNMRraw.loc[self.eNMRraw['U / [V]']==0, cols][0]
            else:
                corr = 0
                    
            try:
                ax.errorbar(self.eNMRraw[self._x_axis], self.eNMRraw[cols]-corr, yerr=self.eNMRraw['%s_err'%cols], fmt='o', label='Phasefitting of Peak %s' %cols, c=colors[0], marker=markers)
                if regression:
                    ax.plot(self.lin_res_dic[cols]['x'], self.lin_res_dic[cols]['y']-corr, '--', label='%s lin regression'%cols, c=colors[0], marker=markers)
            except KeyError:
                if cols=='ph0acme':
                    ax.errorbar(self.eNMRraw[self._x_axis], self.eNMRraw[cols]-corr, fmt='o', label=message[cols], c=colors[0], marker=markers)
                else:
                    ax.errorbar(self.eNMRraw[self._x_axis], self.eNMRraw[col]-corr, yerr=self.eNMRraw['%s_err'%col], fmt='o', label='fitted data %s'%cols, c=colors[i], marker=markers)
            except ValueError:
                ax.errorbar(self.eNMRraw[self._x_axis], self.eNMRraw[cols]-corr, fmt='o', label=message[cols], c=colors[0], marker=markers)
                if regression:
                    ax.plot(self.lin_res_dic[cols]['x'], self.lin_res_dic[cols]['y']-corr, '--', label='%s lin regression'%cols, c=colors[0], marker=None)
        #else:
            #try:
                #ax.errorbar(self._x_axis, cols, yerr='%s_err'%cols, fmt='o', data=self.eNMRraw, label='Phasefitting of Peak %s' %cols, c=colors[0], marker=markers)
                #if regression:
                    #ax.plot(self.lin_res_dic[cols]['x'], self.lin_res_dic[cols]['y'], '--', label='%s lin regression'%cols, c=colors[0], marker=markers)
            #except KeyError:
                    #ax.errorbar(self._x_axis, col, yerr='%s_err'%col, fmt='o', data=self.eNMRraw, label='fitted data %s'%cols, c=colors[i], marker=markers)
            #except ValueError:
                #ax.errorbar(self._x_axis, cols, fmt='o', data=self.eNMRraw, label=message[cols], c=colors[0], marker=markers)
                #if regression:
                    #ax.plot(self.lin_res_dic[cols]['x'], self.lin_res_dic[cols]['y'], '--', label='%s lin regression'%cols, c=colors[0], marker=None)
        ax.legend(bbox_to_anchor=(x_legend, y_legend), ncol=ncol_legend)
        
        xlabel = {'U': r'$U$ / V',
                  'G': r'$g$ / (T\,m$^{-1}$)',
                  'I': '$I$ / mA',
                  'RI': '$(R \cdot I)$ / V'
                      }[self.dependency.upper()]
        
        ax.set_xlabel(xlabel)
        if normalize:
            ax.set_ylabel(r'$\phi-\phi_0$ / °')
        else:
            ax.set_ylabel(r'$\phi$ / °')
        return fig

    def plot_spec_phcorr(self, phasekey='ph0acme', xlim=None, save=False, savepath=None, show=True, ppm=True, orig_data=True, x_legend=1.1, y_legend= 1.0, ncol_legend=2):
        """
        plots phase corrected rows

        ppm: When True, plots the x-axis with ppm scale, xmin and xmax then are percentages
            if False, plots only the number of datapoints, xmin and xmax then are absolute values

        save: saves the spectrum in the data folder

        show: displays the spectrum {plt.show()} or not
        
        returns:
            phased data within given range
        """
        if savepath is None:
            path = self.path+"layered_spectra_"+self.expno+".png"
        else:
            path = savepath
        if orig_data:
            phasedspc = [ng.proc_base.ps(self.data_orig[n, :], p0=-(self.eNMRraw[phasekey][n]))# obsolet: +self.U_0))  # the correction factor for the normalization is added again.
                          for n in range(len(self.data_orig[:, 0]))]  # p1=self.opt[1])
        else:
            phasedspc = [ng.proc_base.ps(self.data[n, :], p0=-(self.eNMRraw[phasekey][n]))# obsolet: +self.U_0))  # the correction factor for the normalization is added again.
                          for n in range(len(self.data[:, 0]))]  # p1=self.opt[1])
            
        #xmin = 0 if xmin is None else xmin
        fig, ax = plt.subplots()
        
        if not ppm:
            #xmax = self.TD if xmax is None else xmax
            #xmax = len(self.data[0,:]) if xmax is None else xmax
            for n in range(len(self.data[:, 0])):
                ax.plot(phasedspc[n].real)
            ax.set_xlim(xlim)
            ax.set_xlabel("data points")
            ax.set_ylabel("intensity (a.u.)")

        else:
            #xmax = self.ppm[0] if xmax is None else xmax
            #xmin = self.ppm[-1] if xmin is None else xmin
            #ixmax, ixmin = np.where(self.ppm >= xmin)[0][0], np.where(self.ppm >= xmax)[0][1]
#             irange = np.where(self._ppmscale <= xmin)[0][0], np.where(self._ppmscale <= xmax)[0][1]

            if orig_data:
                for n in range(len(self.data_orig[:, 0])):
                    # plt.plot(self._ppmscale[ixmin:ixmax], phasedspc[n].real)
                    ax.plot(self.ppm, phasedspc[n].real, label="row %i" %n)
            else:
                for n in range(len(self.data[:, 0])):
                    # plt.plot(self._ppmscale[ixmin:ixmax], phasedspc[n].real)
                    ax.plot(self.ppm, phasedspc[n].real, label="row %i" %n)
            # plt.axis(xmin=self._ppm_l-self.dic["acqus"]["SW"]*xmin/100,
            #          xmax=self._ppm_r+self.dic["acqus"]["SW"]*(1-xmax/100))
            ax.set_xlim(xlim)
            ax.set_xlabel("$\delta$ / ppm")
            ax.set_ylabel("intensity / a.u.")
            ax.legend(bbox_to_anchor=(x_legend, y_legend), ncol=ncol_legend)

        #ax = plt.gca()
        #ax.set_xlim(ax.get_xlim()[::-1])  # inverts the x-axis
        ax.ticklabel_format(style='sci', scilimits=(0,0), axis='y')

        if save:
            fig.savefig(path, dpi=500)
        if show:
            plt.show()
        #print("ixmax:", ixmax)
        #print("ixmin:", ixmin)
#         return ixmax, ixmin
        phasedspc = np.asarray(phasedspc)
        
        return fig, ax
        #if orig_data:
            #return phasedspc[:,ixmin:ixmax]
        #else:
            #return phasedspc
        # ps = ng.proc_base.ps

    def plot_spec_comparison_to_0(self, row, xmax=None, xmin=None, ppm=True):
        """
        plots row 0 and row n in the range of xmax and xmin
        """
        _max = None if xmax is None else xmax
        _min = None if xmin is None else xmin

        fig = plt.figure()
        if ppm:
            _max = self.ppm[0] if xmax is None else xmax
            _min = self.ppm[-1] if xmin is None else xmin
            
            plt.plot(self.ppm, self.data[0, ::1].real, label='row ' + str(0))
            plt.plot(self.ppm, self.data[row, ::1].real, label='row ' + str(row))
            plt.legend()
            plt.axis(xmax=_max, xmin=_min)
            plt.title("this is row %i at %.0f V" % (row, self.eNMRraw.loc[row, self._x_axis]))
            plt.xlabel("ppm")
            plt.ylabel("intensity / a.u.")
        if not ppm:
            plt.plot(self.data[0, ::1].real, label='row '+str(0))
            plt.plot(self.data[row, ::1].real, label='row '+str(row))
            plt.legend()
            plt.axis(xmax=_max, xmin=_min)
            plt.title("this is row %i at %.0f V" % (row, self.eNMRraw.loc[row, self._x_axis]))
            plt.xlabel("datapoints")#.sum()
            plt.ylabel("intensity / a.u.")

        plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))

        return fig

    def output_data(self):
        """
        saves the raw phase data in the measurement-folder as a csv
        """
        self.eNMRraw.to_csv(self.path+"phase_data_"+self.expno+".csv", sep=" ")


    def output_results(self, path=None):
        """
        saves the mobility result data in the measurement-folder similar to obj.output_data()
        """
        results_output = pd.Series([self.nuc,
                                    self.mu[0],
                                    self.sig_m*self.mu[0],
                                    self.d,
                                    self.g,
                                    self.delta,
                                    self.Delta,
                                    self.uInk],
                                   index=["nucleus",
                                          "µ",
                                          "sig_µ",
                                          "d / m",
                                          "g / (T/m)",
                                          "delta / s",
                                          "Delta / s",
                                          "Uink / V"],
                                   name=self.dateipfad)
        if path is None:
            results_output.to_csv(self.path+"mobility_data_"+self.expno+".csv")
        elif path is not None:
            results_output.to_csv(path+"mobility_data_"+self.expno+".csv")
        else:
            print('ooops!')
    
    def output_all_results(self, path=None, data=False):
        """
        saves the mobility result data in the measurement-folder similar to obj.output_data()
        """
        from pandas import ExcelWriter
        from openpyxl import load_workbook
        
        try:
            book = load_workbook(path)
            writer = pd.ExcelWriter(path, engine='openpyxl') 
            writer.book = book
            writer.sheets = dict((ws.title, ws) for ws in book.worksheets)
        except:
            writer = ExcelWriter(path, engine='xlsxwriter')

        for key in self.lin_res_dic.keys():
            #self.mobility(key)
            results_output = pd.Series([self.nuc,
                                        self.expno,
                                        self.lin_res_dic[key]['mu'][0],
                                        self.lin_res_dic[key]['mu_err'][0],
                                        self.d,
                                        self.g,
                                        self.delta,
                                        self.Delta,
                                        self.uInk],
                                    index=["nucleus",
                                           "expno",
                                            "µ",
                                            "sig_µ",
                                            "d / m",
                                            "g / (T/m)",
                                            "delta / s",
                                            "Delta / s",
                                            "Uink / V"],
                                    name=self.dateipfad)
            results_output.to_excel(writer, sheet_name=self.nuc+'expno'+str(self.expno)+'_'+key)
        
        if data:
            try:
                self.eNMRraw.drop(['data', 'fid'], 1).to_excel(writer, sheet_name=self.nuc+'expno'+str(self.expno)+'_data')
            except KeyError:
                self.eNMRraw.to_excel(writer, sheet_name=self.nuc+'expno'+str(self.expno)+'_data')
            except:
                print('oops, an unexpected error occured')
            writer.save()
        return
    
    def mobility(self, y_column='ph0acme', electrode_distance=None, verbose=True):
        """
        calculates and returns (mobility, deviation) from the regression data
        """
                #self.lin_res_dic = {y_column: {'b': self.b,
                                   #'m': self.m,
                                   #'r^2': self.r_square,
                                   #'y_reg': self._y_pred,
                                   #'sig_m': self.sig_m}}
        if electrode_distance is None:
            d = self.d
        else:
            d = electrode_distance
        
        if self.dependency.upper() == "G":
            g = self.uInk
        elif self.dependency.upper() == "U":
            g = self.g
        elif self.dependency.upper() == "I":
            g = self.g
        elif self.dependency.upper() == "RI":
            g = self.g
        else:
            print("no dependency was set")
        
        if y_column is None:
            self.mu = (self.m*d)/(self.gamma*self.delta*self.Delta*g)
            #return self.mu, self.mu*(self.sig_m/self.m)

        else:
            m = self.lin_res_dic[y_column]['m']
            sig_m = self.lin_res_dic[y_column]['sig_m']
            self.mu = (m*d)/(self.gamma*self.delta*self.Delta*g)
            self.lin_res_dic[y_column]['mu']=self.mu
            self.lin_res_dic[y_column]['mu_err']=self.mu*(sig_m/m)
            #return self.mu, self.mu*(sig_m/m)
            self.sig_m, self.m = sig_m, m
        
        if verbose:
            print ('%.2E (m^2/Vs)'%self.mu[0],'+- %.2E'%(self.mu*(self.sig_m/self.m)))
        
        return self.mu, self.mu*(self.sig_m/self.m)


    def analyzePhasecorrection(self, linebroadening=10, lin_threshhold=2.5,
                           graph_save=False, savepath=None, method="acme", xmin=None,
                           xmax=None, cropmode="absolute", progress=True, umin=None, umax=None, output_path=None):

        """
        standard phase correction analyzing routine for eNMR
        
        linebroadening:
            sets the linebroadening of the fourier transformation
        lin_threshhold:
            sets the threshhold for outlier detection of the linear regression
        graph_save:
            True: saves the graph as a png in the measurment folder
        savepath:
            None: Takes the standard-values from the measurement
            'string': full path to customized graph
                    works with .png, .pdf and .eps suffixes
        method: chooses between phase correction algorithms
			"acme": standard method using entropy minimization
			"difference": --> _ps_abs_difference_score()
                minimizes the linear difference to the first spectrum by fitting p0
            "sqdifference": --> _ps_sq_difference_score()
                minimizes the square difference to the first spectrum by fitting p0
        xmin, xmax:
            min and maximum x-values to crop the data for processing (and display)
            can take relative or absolute values depending on the cropmode.
        
        cropmode: changes the x-scale unit
            "percent": value from 0% to 100% of the respective x-axis-length --> does not fail
            "absolute": takes the absolute values --> may fail
        progress: This shows the spectral row that is processed in this moment. May be disabled in order to be able to stop clearing the output.
        """
        if output_path is None:
            output_path = 'expno_%i_results.xlsx'%self.expno
        
        self.proc(linebroadening=linebroadening, xmin=xmin, xmax=xmax, cropmode=cropmode)
        self.autophase(analyze_only=False, method=method, progress=progress)
        self.lin_huber(epsilon=lin_threshhold, umin=umin, umax=umax)
        # obj.lin() #---> dies ist die standardn, least squares method.
        self.lin_display(save=graph_save, dpi=330, savepath=savepath)
        self.mobility()
        self.output_all_results(output_path)


class eNMR_Measurement(eNMR_Methods):
    '''
    This is the subsubclass of Masurement() and subclass of eNMR_Methods specialised to process data obtained from the experimental Schönhoff set-up
    
    path:
        relative or absolute path to the measurements folder
    measurement:
        the to the experiment corresponding EXPNO
    alias:
        Here you can place an individual name relevant for plotting. If None, the path is taken instead.    
    Uink:
        voltage increment. Usually extracted from the title file if defined with e.g. "Uink = 10V"
        If Uink cannot be found or is wrong it can be entered manually during the data import.
        The voltage list is calculated from the voltage increment and the vc list when the incrementation loop is used in the pulse program
    dependency:
        'U': voltage dependent eNMR measurement
        'G': fieldgradient dependent eNMR measurement

    linebroadening:
        setting a standard-value for the linebroadening.
    '''
    def __init__(self, path, expno, Uink=None, dependency="U", alias=None, linebroadening=0.5, electrode_distance=2.2e-2):
        Measurement.__init__(self, path, expno, linebroadening=linebroadening, alias=alias)
        self.dependency = dependency.upper()
        
        self._x_axis = {"U": "U / [V]",
                   "G": "g in T/m",
                   "I": "I / mA",
                   'RI': 'RI / V'
                   }[self.dependency.upper()]
        
        #self._x_axis = {"G": "g in T/m",
                        #"U": "U / [V]"}[self.dependency.upper()]
        
        if dependency.upper() == "U":
            try:
                # takes the title page to extract the volt increment
                title = open(self.dateipfad+"/pdata/1/title").read()
                # gets the voltage increment using a regular expression
                #uimport = findall('[U|u]in[k|c]\s*=?\s*\d+', title)[0]
                uimport = findall('[U|u]in[k|c]\s*=+\s*\d+', title)[0]
                self.uInk = int(findall('\d+', uimport)[0])
            except ValueError:
                print('no volt increment found\nyou may want to put it in manually')
                self.uInk = Uink
            except IndexError:
                print('No Uink found! May not be an eNMR experiment.')
                self.uInk = Uink
                
        elif dependency.upper() == "G":
            try:
                # takes the title page to extract the volt increment
                title = open(self.dateipfad+"/pdata/1/title").read()
                # gets the voltage increment using a regular expression
                uimport = findall('[U|u]\s*=?\s*\d+', title)[0]
                self.uInk = int(findall('\d+', uimport)[0])

            except ValueError:
                print('no volt increment found\nyou may want to put it in manually')
                self.uInk = Uink
            except IndexError:
                print('No Uink found! May not be an eNMR experiment.')
                self.uInk = Uink # Uinktext

        if self.dependency.upper() == "U":
            try:
                self.vcList = pd.read_csv(self.dateipfad+"/vclist",
                                          names=["vc"]).loc[:len(self.data[:, 0])-1]
            except:
                print("There is a Problem with the VC-list or you performed a gradient dependent measurement")
        elif self.dependency.upper() == "G":
            self.vcList = pd.DataFrame(np.ones((len(self.data[:, 0]), 1)),
                                       columns=["vc"])
        else:
            print("The dependency is not properly selected, try again!")

        self.difflist = pd.read_csv(self.dateipfad+"/difflist",
                                    names=["g in T/m"])*0.01
        
        if Uink is not None:
            self.uInk = Uink
            
        self.vcList["U / [V]"] = [self.vcList["vc"][n]/2*self.uInk if self.vcList["vc"][n] % 2 == 0
                                  else (self.vcList["vc"][n]+1)/2*self.uInk*-1
                                  for n in range(len(self.data[:, 0]))]
        
        # try to open phase data, otherwise create new
        try:
            self.eNMRraw = pd.read_csv(self.path+"phase_data_"+self.expno+".csv",
                                       index_col=0, sep=" ")
            # --> update voltage list
            self.eNMRraw["U / [V]"] = self.vcList["U / [V]"]
        except:
            print("eNMRraw was missing and is generated")
            self.vcList["ph0"] = np.zeros(len(self.data.real[:, 0]))
            self.eNMRraw = self.vcList
        finally:
            self.eNMRraw["g in T/m"] = self.difflist
        
        self.p1 = self.dic["acqus"]["P"][1]
        self.d1 = self.dic["acqus"]["D"][1]
        
        try:
            # import of diffusion parameters for newer Spectrometers
            import xml.etree.ElementTree as etree
            diffpar = etree.parse(self.dateipfad+'/diff.xml')
            root = diffpar.getroot()
            self.Delta = float(root.findall('DELTA')[0].text)*1e-3
            self.delta = float(root.findall('delta')[0].text)*1e-3  # it should be read as in microseconds at this point due to bruker syntax
            print('The diffusion parameters were read from the respectie .XML!')
        except:
            # determination of the diffusion parameters for Emma
            self._d2 = self.dic["acqus"]["D"][2]
            self._d5 = self.dic["acqus"]["D"][5]
            self._d9 = self.dic["acqus"]["D"][9]
            self._d11 = self.dic["acqus"]["D"][11]
            self._p19, self._p18, self._p17 = self.dic["acqus"]["P"][19],\
                                            self.dic["acqus"]["P"][18],\
                                            self.dic["acqus"]["P"][17]
            print('That did not work. Your data is from an old spectrometer!')
            # calculating usable parameters
            self.delta = self._p17+self._p18
            self._Delta_1 = 0.001*(self._p17*2+self._p18)+(self._d2+self._d9+self._d5+self._d11)*1000+0.001*self.p1+self._d11
            self._Delta_2 = 0.001*(self._p17*2+self._p18)+(self._d2+self._d9+self._d5+self._d11)*1000+0.001*self.p1*2
            self._spoiler = (self._d11+self._p17+self._p19+self._p17)*0.001+self._d2*1000
            self.Delta = self._Delta_1+self._Delta_2+2*self._spoiler
            self.Delta *=1e-3
            self.delta *=1e-6
            

        # Elektrodenabstand in m
        self.d = electrode_distance
        self.g = self.eNMRraw["g in T/m"][0]
    
    def plot_spec(self, row, xlim=None, figsize=None, invert_xaxis=True, sharey=True):#, ppm=True):
        """
        plots row 0 and row n in the range of xmax and xmin
        
        :returns: figure
        """

        _max = None if xlim is None else xlim[0]
        _min = None if xlim is None else xlim[1]
        
        if type(xlim) is not list:
            fig = plt.figure(figsize=figsize)
            ax = fig.add_subplot(111)
        elif type(xlim) == list:
            fig, ax = plt.subplots(ncols=len(xlim), nrows=1, figsize=figsize, sharey=sharey)
        
        _min = self.ppm[0] if xlim is None else xlim[1]
        _max = self.ppm[-1] if xlim is None else xlim[0]
        
        def plot(r, axes=ax):
            
            unit = {'U': 'V', 'G': 'T/m', 'I': 'mA', 'RI': 'V'}[self.dependency.upper()]
            
            axes.plot(self.ppm, self.data[r, ::1].real, label='row %i, %i %s'%(r, self.eNMRraw[self._x_axis].iloc[r], unit))
        
        if type(xlim) is not list:
            if type(row) ==list:
                for r in row:
                    plot(r)
            else:
                plot(row)
        
            ax.set_xlim(xlim)
            #ax.set_title("this is row %i at %.0f V" % (row, self.eNMRraw.loc[row, "U / [V]"]))
            ax.set_xlabel("$\delta$ / ppm")
            ax.set_ylabel("intensity / a.u.")

            ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
            if invert_xaxis:
                xlimits = ax.get_xlim()
                ax.set_xlim(xlimits[::-1])
            
            ax.legend()
        
        elif type(xlim) == list:
            for axis, xlim in zip(ax,xlim):
                if type(row) ==list:
                    for r in row:
                        plot(r, axis)
                else:
                    plot(row, axis)
                
                axis.set_xlim(xlim)
                #ax.set_title("this is row %i at %.0f V" % (row, self.eNMRraw.loc[row, "U / [V]"]))
                axis.set_xlabel("$\delta$ / ppm")
                axis.set_ylabel("intensity / a.u.")
                axis.legend()
                axis.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
                if invert_xaxis:
                    xlimits = axis.get_xlim()
                    axis.set_xlim(xlimits[::-1])
            ax[-1].legend()
            #fig.legend()
        return fig
    


class Pavel(eNMR_Methods):
    '''
    This is the subsubclass of Masurement() and subclass of eNMR_Methods specialised to process data obtained from the experimental Swedish from Pavel set-up
    the voltage list is valculated from the vd-values

    path:
        relative or absolute path to the measurements folder
    expno:
        the to the experiment number corresponding EXPNO
    dependency:
        'U': voltage dependent eNMR measurement
        'G': fieldgradient dependent eNMR measurement

    alias:
        Here you can place an individual name relevant for plotting. If None, the path is taken instead.
    linebroadening:
        setting a standard-value for the linebroadening.
    '''
    def __init__(self, path, expno, dependency='U', alias=None, linebroadening=5, electrode_distance=2.2e-2, cell_resistance=None):
        
        self.dependency = dependency
        self.cell_resistance = cell_resistance
        
        super().__init__(path, expno, linebroadening=linebroadening, alias=alias)
        
        #self._x_axis = {"G": 'g in T/m', "U": 'U / [V]'}[dependency.upper()]
        # import the diffusion parameters
        import xml.etree.ElementTree as etree
        diffpar = etree.parse(self.dateipfad+'/diff.xml')
        root = diffpar.getroot()
        self.Delta = float(root.findall('DELTA')[0].text)*1e-3
        self.delta = float(root.findall('delta')[0].text)*1e-3  #in Seconds
        print('The diffusion parameters were read from the respectie .XML!')
        
        try:
            self.vdList = pd.read_csv(self.dateipfad+"/vdlist",
                                      names=["vd"]).loc[:len(self.data[:, 0])-1]
        except:
            print('no vdList found, generated ones list instead')
            self.vdList = pd.DataFrame(np.ones((len(self.data[:, 0]), 1)),
                                       columns=["vd"])
        self.eNMRraw = self.vdList

        #self.vdList["U / [V]"] = hier die Konversion von vdlist zu Spannungsliste
        try:
            self.difflist = pd.read_csv(self.dateipfad+"/gradlist",
                                    names=["g in T/m"])*0.01
        except:
            print('gradlist not found. difflist imported instead')
            self.difflist = pd.read_csv(self.dateipfad+"/difflist",
                                    names=["g in T/m"])*0.01
        self.eNMRraw["g in T/m"] = self.difflist
        
        self.d = electrode_distance
        self.g = self.eNMRraw["g in T/m"][0]
        

        # converts the vd-List
        for i, n in enumerate(self.eNMRraw['vd']):
            self.eNMRraw.loc[i, 'vd_temp'] = float(n[:-1])
        # calculates the applied Voltages

        if self.dependency.upper() == "U":
            self.eNMRraw[self._x_axis] = [
                0 if (self.eNMRraw.loc[i,'vd_temp'] <= 0.6)
                else 
                n if i%2==0 
                else 
                n*-1
                for i, n in enumerate(self.eNMRraw['vd_temp']*5)]
        
            self.uInk = self.eNMRraw['U / [V]'][0] - self.eNMRraw['U / [V]'][1] 
            if self.uInk == 0:
                self.uInk = self.eNMRraw['U / [V]'][0] - self.eNMRraw['U / [V]'][2] 
            if self.uInk < 0:
                self.uInk *= -1
        
        elif self.dependency.upper() == "I":
            self.uInk = None
            self.eNMRraw[self._x_axis] = [
                    0 if (self.eNMRraw.loc[i,'vd_temp'] <= 0.6)
                    else 
                    n if i%2==0 
                    else 
                    n*-1
                    for i, n in enumerate(self.eNMRraw['vd_temp'])
                ]
        
        elif self.dependency.upper() == "RI":
            self.uInk = None
            self.eNMRraw[self._x_axis] = [
                    0 if (self.eNMRraw.loc[i,'vd_temp'] <= 0.6)
                    else 
                    n if i%2==0 
                    else 
                    n*-1
                    for i, n in enumerate(self.eNMRraw['vd_temp'])
                ]
            self.uInk = self.eNMRraw['RI / V'][0] - self.eNMRraw['RI / V'][1] 
            if self.uInk == 0:
                self.uInk = self.eNMRraw['RI / V'][0] - self.eNMRraw['RI / V'][2] 
            if self.uInk < 0:
                self.uInk *= -1
            # calculation of the Voltage from cell resistance and Current /1000 because of mA
            self.eNMRraw[self._x_axis] *= self.cell_resistance/1000
            
    def plot_spec(self, row, xlim=None, figsize=None, invert_xaxis=True, sharey=True):#, ppm=True):
        return eNMR_Measurement.plot_spec(self, row, xlim, figsize, invert_xaxis, sharey)#, ppm=True):

#class Idependent(Pavel):
    #def __init__(self, path, expno, dependency='U', alias=None, linebroadening=5, electrode_distance=2.2e-2):
        #Pavel.__init__(path, expno, dependency='U', alias=None, linebroadening=5, electrode_distance=2.2e-2)
        
        #self.eNMR['I'] = None#vd=Funktion
        
        #self.eNMR.drop('U / [V]', inplace=True)

class Simulated(eNMR_Methods):
    '''
    sublass of eNMR_Methods and Measurement for the import of simulated data via the SpecSim class
    '''

    def __init__(self, sim, alias=None):
        self.g = sim.params.spec_par['g']
        self.d = sim.params.spec_par['d']
        self.Delta = sim.params.spec_par['Delta']
        self.delta = sim.params.spec_par['delta']
        self.dependency = 'U'
        self.eNMRraw = sim.eNMRraw
        self.data = sim.data
        self.data_orig = sim.data
        self.ppm = sim.ppm
        self.eNMRraw['g in T/m'] = self.g

        # the gamma_values in rad/Ts
        gamma_values = {'1H':26.7513e7,
                        '7Li': 10.3962e7,
                        '19F': 25.1662e7}
        self.gamma = gamma_values[sim.params.spec_par['NUC']]
        # Umrechnung von rad in °
        self.gamma = self.gamma/2/np.pi*360
        
        self.lin_res_dic = {}
        self.alias = alias
        self.path = 'simulated data/'
        self.expno = '0'
        self.uInk = self.eNMRraw.loc[1, 'U / [V]'] - self.eNMRraw.loc[0, 'U / [V]']
        self._x_axis = 'U / [V]'
        #self._x_axis = {"U": "U / [V]",
            #"G": "g in T/m"
            #}[self.dependency.upper()]
##################################

class MOSY(object):
    """
    takes an eNMR Measurement obj and yields as MOSY object for MOSY processing and depiction.
    the Measurement object needs to be processed (fouriertransformed in F2) before passing it to this class
    """
    def __init__(self, obj):
        self.obj = obj
        self.eNMRraw = obj.eNMRraw
        self.eNMRraw['data'] = None
        self.eNMRraw['fid'] = None
        for row in range(len(obj.data[:,0])):
            self.eNMRraw.set_value(row, 'data', pd.Series(obj.data[row,:]))
            #self.eNMRraw.set_value(row, 'fid', pd.Series(obj.fid[row,:]))
        data_sorted = np.array(obj.eNMRraw['data'].tolist())
        self.data = data_sorted.astype('complex').copy()
        self.TD1 = len(self.data[:,0])
        self.mscale = None
        self.mX = None
        self.mY = None
#         self.mZ = None
        
    def zerofilling_old(self, n=128, dimension=0):
        """
        n: number of total datapoints along the F1-dimension (Voltage)
        dimension 0 = F1
        dimension 1 = F2
        """
        if dimension == 0:
            self.data = np.concatenate((self.data, 
                                [[0 for x in range(len(self.data[0,:]))] for y in range(n-len(self.data[:,0]))]))
        if dimension == 1:
            self.data = np.concatenate((self.data, 
                                [[0 for x in range(n-len(self.data[0,:]))] for y in range(len(self.data[:,0]))]), axis=1)
        print('got it!')
        
    def zerofilling(self, n=128, dimension=0):
        """
        n: number of total datapoints along the F1-dimension (Voltage)
        dimension 0 = F1
        dimension 1 = F2
        """
        print('zero filling started!')
        
        if dimension == 0:
            old_shape = self.data.shape
            new_shape = old_shape
            new_shape = n, old_shape[1]
            print(old_shape, new_shape)
            data_new = np.zeros(new_shape, dtype=complex)
            for i in range(old_shape[0]):
                data_new[i,:] = self.data[i,:]
            self.data = data_new
            del(data_new)
            
        if dimension == 1:
        
            old_shape = self.data.shape
            new_shape = old_shape
            new_shape = old_shape[0], n
            print(old_shape, new_shape)
            data_new = np.zeros(new_shape, dtype=complex)
            for i in range(old_shape[1]):
                data_new[:,i] = self.data[:,i]
            self.data = data_new
            del(data_new)
        
        print('zero filling finished!')

    def fft_F1 (self):
        """
        Fourier Transformation from nmrglue.proc_base.fft)
        along the F1 dimension
        """
        #data_temp = np.zeros(self.data.shape)
        for n in range(len(self.data[0,:])):
            self.data[:,n] = ng.proc_base.fft(self.data[:,n])
        print("done")
    
    def calc_MOSY(self, u_max = None, n_zf=2**12, mobility_scale=True, include_0V=True, electrode_distance=2.2e-2, old_zf=False):
        
        '''
        Calculates the MOSY according to the States-Haberkorn-Ruben method described for MOSY applications by AUTOR NENNEN!
        
        u_max:
            maximum Voltage to be used for processing.
        n_zf:
            number of zerofilling for the F1 dimension
        mobility_scale:
            prints the converted mobilty y-scale if True
        include_0V:
            include the 0V spectrum or not.
        '''
        x = self.obj._x_axis
        
        if u_max is None:
            u_max = 10000
        
        if include_0V:
            positive = self.eNMRraw[(self.eNMRraw[x] >= 0)
                                    &(self.eNMRraw[x] <= u_max)].sort_values(x)
            negative = self.eNMRraw[(self.eNMRraw[x] <= 0)
                                    &(self.eNMRraw[x] >= -u_max)].sort_values(x, ascending=False)
        else:
            positive = self.eNMRraw[self.eNMRraw[x] >0].sort_values(x)
            negative = self.eNMRraw[self.eNMRraw[x] <0].sort_values(x, ascending=False)

        
        #dataset conversion for the SHR-method
        #if include_0V:
            #positive = self.eNMRraw[(self.eNMRraw['U / [V]'] >= 0)
                                    #&(self.eNMRraw['U / [V]'] <= u_max)].sort_values('U / [V]')
            #negative = self.eNMRraw[(self.eNMRraw['U / [V]'] <= 0)
                                    #&(self.eNMRraw['U / [V]'] >= -u_max)].sort_values('U / [V]', ascending=False)
        #else:
            #positive = self.eNMRraw[self.eNMRraw['U / [V]'] >0].sort_values('U / [V]')
            #negative = self.eNMRraw[self.eNMRraw['U / [V]'] <0].sort_values('U / [V]', ascending=False)

        SHR_real = np.zeros((len(positive['data']), len(positive['data'].iloc[0])))
        SHR_imag = np.zeros((len(positive['data']), len(positive['data'].iloc[0])))

        for n in range(1, len(positive['data'])):
            SHR_real[n,:] = positive['data'].iloc[n].real + negative['data'].iloc[n].real
            SHR_imag[n,:] = positive['data'].iloc[n].imag - negative['data'].iloc[n].imag

        SHR = SHR_real + SHR_imag*1j
        del(SHR_real)
        del(SHR_imag)
        self.data = SHR
        del(SHR)
        if old_zf:
            self.zerofilling_old(n=n_zf)
        else:
            self.zerofilling(n=n_zf)
        self.fft_F1()
     
        X = np.array([self.obj.ppm for y in range(len(self.data[:,0]))])
        Y = np.array([[y for x in range(len(self.data[0,:]))] for y in range(len(self.data[:,0]))])
        
        Y = ((0.5*n_zf-Y)*self.TD1/n_zf*360)/self.TD1 # conversion to ° phasechange per Uink
        Y = Y/self.obj.uInk  # ° phasechange per Volt
        
        if mobility_scale:
            #Y = (Y*self.obj.d)/(self.obj.gamma*self.obj.delta*self.obj.Delta*self.obj.g) # old version with self.d
            Y = (Y*electrode_distance)/(self.obj.gamma*self.obj.delta*self.obj.Delta*self.obj.g)
        
        self.mX = X
        del(X)
        self.mY = -Y
        del(Y)
    
    #def plot_MOSY_old(self, xlim=None, ylim=(-1e-8, 1e-8), yscale='linear',
                  #mobility_scale = True,
                  #save=False, savepath='',
                  #tight_layout=True, dpi=300,
                  #**kwargs):

        #"""
        #k:
            #correction factor withdrawn from the y-axis
        #yscale:
            #.. ACCEPTS: [ 'linear' | 'log' | 'symlog' | 'logit' | ... ]
        #**kwargs:
            #are passed to the .contour()
            #for example:
                #levels: list of values to be drawn as height lines
        #""" 
        #print('this function will be depreceated')
        
        #fig, ax = plt.subplots(2, sharex=True, figsize=(5,5),gridspec_kw={'height_ratios':[3,1]})

        #ax,bx = ax
        
        #n_zf = len(self.data[:,0]) # after zerofilling
        
        #X = np.array([self.obj.ppm for y in range(len(self.data[:,0]))])
        #Y = np.array([[y for x in range(len(self.data[0,:]))] for y in range(len(self.data[:,0]))])
        
        
        ## WICHTIG!!!! MUSS ICH HIER EINS ABZIEHEN ODER NICHT?!
        #Y = ((0.5*n_zf-k-Y)*self.TD1/n_zf*360)/self.TD1 # conversion to ° phasechange per Uink
        #Y = Y/self.obj.uInk  # ° phasechange per Volt
        
        #if mobility_scale:
            #Y = (Y*self.obj.d)/(self.obj.gamma*self.obj.delta*1e-6*self.obj.Delta*0.001*self.obj.g)
        
        
        #ax.contour(self.mX, self.mY, self.data.real, **kwargs)
        #ax.set_xlim(xlim)
        #ax.set_ylim(ylim)
        #bx.plot(self.mX[0], self.obj.data.real[0], 'r-')
        #ax.grid()
        
        #ax.set_ylabel(r'$\mu\;/\;(\textrm{m}^2 \textrm{V}^{-1} \textrm{s}^{-1})$')
        #bx.set_ylabel(r'intensity / a.u.')
        #bx.set_xlabel(r'$\delta$ / ppm')
        
        #ax.set_yscale(yscale)
        
        #bx.ticklabel_format(style='sci', scilimits=(0,2))
        
        #def autoscale_y(bx, margin=0.1):
            #"""This function rescales the y-axis based on the data that is visible given the current xlim of the axis.
            #ax -- a matplotlib axes object
            #margin -- the fraction of the total height of the y-data to pad the upper and lower ylims"""

            #def get_bottom_top(line):
                #xd = line.get_xdata()
                #yd = line.get_ydata()
                #lo,hi = bx.get_xlim()
                #try:
                    #y_displayed = yd[((xd>lo) & (xd<hi))]
                    #h = np.max(y_displayed) - np.min(y_displayed)
                #except:
                    #y_displayed = yd[((xd<lo) & (xd>hi))]
                    #h = np.max(y_displayed) - np.min(y_displayed)
                
                #bot = np.min(y_displayed)#-margin*h
                #top = np.max(y_displayed)#+margin*h
                #print(bot, top, h)
                #return bot,top

            #lines = bx.get_lines()
            #bot,top = np.inf, -np.inf

            #for line in lines:
                #new_bot, new_top = get_bottom_top(line)
                #if new_bot < bot: bot = new_bot
                #if new_top > top: top = new_top

            #bx.set_ylim(bot,top)
        
        #autoscale_y(bx, xlim)
        
        #if tight_layout:
            #fig.tight_layout()
        
        #if save:
            #fig.savefig(savepath, dpi=dpi)
        
        #return fig
    
    def plot_MOSY(self, xlim=None, ylim=(-1e-8, 1e-8), yscale='linear',
                  save=False, savepath='',
                  tight_layout=False, dpi=300, y_autoscale=True,
                  h_ratio=7, w_ratio=8, figsize=(5,5),
                  latex_backend=False, hspace=0, wspace=0,
                  **kwargs):
        
        from matplotlib import gridspec
        """
        yscale:
            .. ACCEPTS: [ 'linear' | 'log' | 'symlog' | 'logit' | ... ]
        **kwargs:
            are passed to the .contour()
            for example:
                levels: list of values to be drawn as height lines
                
        :returns: figure
        """
        fig= plt.figure(figsize=figsize)
        
        gs = gridspec.GridSpec(2, 2, width_ratios=[1, w_ratio], height_ratios=[1,h_ratio]) 

        mitte = fig.add_subplot(gs[3])
        oben = fig.add_subplot(gs[1], sharex=mitte, frameon=False)
        links = fig.add_subplot(gs[2], sharey=mitte, frameon=False)

        #fig.subplots_adjust(hspace=0, wspace=0)

        mitte.yaxis.tick_right()
        mitte.contour(self.mX, self.mY, self.data.real, **kwargs)
        mitte.set_ylim(ylim)
        mitte.set_xlim(xlim)
        links.set_xticks([])

        # calculation and plotting of the projections taking the maxima
        links.plot(np.amax(self.data, axis=1), self.mY[:,0],  'k')
        links.set_xlim(links.get_xlim()[::-1])
        oben.plot(self.mX[0], np.amax(self.data, axis=0), 'k')
        
        #axis format
        oben.tick_params(bottom='off')
        links.tick_params(left='off')
        oben.set_yticks([])
        plt.setp(oben.get_xticklabels(), visible=False);
        plt.setp(links.get_yticklabels(), visible=False);
        plt.setp(links.get_yaxis(), visible=False);
        if latex_backend:
            mitte.set_ylabel(r'$\mu\;/\;(\textrm{m}^2 \textrm{V}^{-1} \textrm{s}^{-1})$')
        else:
            mitte.set_ylabel(r'$\mu$ / (m$^2$V$^{-1}$s$^{-1})$')
        mitte.yaxis.set_label_position("right")
        mitte.set_xlabel(r'$\delta$ / ppm')
        mitte.set_yscale(yscale)
        mitte.ticklabel_format(axis='y', style='sci', scilimits=(0,2))
        mitte.grid()
        
        fig.subplots_adjust(hspace=hspace, wspace=wspace)
        
        def autoscale_y(bx, margin=0.1):
            """This function rescales the y-axis based on the data that is visible given the current xlim of the axis.
            ax -- a matplotlib axes object
            margin -- the fraction of the total height of the y-data to pad the upper and lower ylims"""

            def get_bottom_top(line):
                xd = line.get_xdata()
                yd = line.get_ydata()
                lo,hi = bx.get_xlim()
                try:
                    y_displayed = yd[((xd>lo) & (xd<hi))]
                    h = np.max(y_displayed) - np.min(y_displayed)
                except:
                    y_displayed = yd[((xd<lo) & (xd>hi))]
                    h = np.max(y_displayed) - np.min(y_displayed)
                
                bot = np.min(y_displayed)#-margin*h
                top = np.max(y_displayed)#+margin*h
                print(bot, top, h)
                return bot,top

            lines = bx.get_lines()
            bot,top = np.inf, -np.inf

            for line in lines:
                new_bot, new_top = get_bottom_top(line)
                if new_bot < bot: bot = new_bot
                if new_top > top: top = new_top

            bx.set_ylim(bot,top)
            
        if y_autoscale:
            autoscale_y(oben, xlim)
        
        if tight_layout:
            fig.tight_layout()
        
        if save:
            fig.savefig(savepath, dpi=dpi)

        return fig
    
    def plot_slices_F1(self, ppm, xlim=None, scaling=None, normalize=False, vline_0=False, annotate=False, annotate_pos=None, legend_loc=None, latex_backend=False, colors=None, figsize=None):
        """
        plots slices in F1-direction

        ppm:
            list of chemical shift-slices to be plotted
        xmin, xmax:
            limits for the x-axis
        scaling:
            list of scaling factors for spectra to adjust intensities
        normalize:
            normalization of each peak to 1
        vline_0:
            prints a vertical line at µ = 0
        annotate:
            automatically annotates the maxima of the slices and displays the respective mobility
        colors:
            list of colors/colorcodes as strings
        
        :returns: figure
        """

        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)
        # getting the standard color cycle
        prop_cycle = plt.rcParams['axes.prop_cycle']
        if colors is None:
            colors = prop_cycle.by_key()['color']

        if scaling is None:
            scaling = np.ones((len(ppm)))
        i=0
        # setting the initial textposition depending on the number of slices
        xpostext = 0.5-((len(ppm)-1)*0.1-0.1)
        if xpostext < 0.2:
            xpostext = 0.2
        ypostext = 0.2
        
        for n in ppm:
            pos = np.searchsorted(self.obj.ppm, n)
            x = self.mY[:,pos]
            if normalize:
                y = scaling[i]*self.data[:,pos].real/self.data[:,pos].max()
                ax.plot(x, y, label=r'$\delta=%.2f$'%n, c=colors[i])
                ax.set_ylabel('normalized intensity')
            else:
                y = scaling[i]*self.data[:,pos].real
                ax.plot(x, y, label=r'$\delta=%.2f$'%n, c=colors[i])
                ax.set_ylabel('intensity / a.u.')
            
            if annotate:
                xmax = x[np.argmax(y)]
                ymax = y.max()
                text = "$\mu$=%.2E"%(xmax)
                bbox_props = dict(boxstyle="square,pad=0.3", fc="w", ec="k", lw=0.72)
                arrowprops=dict(arrowstyle="->", color=colors[i])#, connectionstyle="angle,angleA=0,angleB=60")
                kw = dict(xycoords='data',textcoords="axes fraction",
                arrowprops=arrowprops, bbox=bbox_props, ha="right", va="top")
                if type(annotate_pos) == list:
                    xpostext, ypostext = annotate_pos[i]
                elif type(annotate_pos) == tuple:
                    xpostext += annotate[0]
                    ypostext += annotate[1]
                ax.annotate(text, xy=(xmax, ymax), xytext=(xpostext,ypostext), **kw)
                #placing of the textboxes
                if xpostext >= 1:
                    xpostext = 0.2
                    ypostext += 0.1
                else:
                    xpostext += 0.2
            i += 1

        if vline_0:
            ax.vlines(0, *ax.get_ylim(), linestyles='dotted')
        if latex_backend:
            ax.set_xlabel('$\mu$\;/\;(m$^2$V$^{-1}$s$^{-1})$')
        else:
            ax.set_xlabel('$\mu$ / (m$^2$V$^{-1}$s$^{-1})$')
        ax.set_xlim(xlim)
        ax.legend(loc=legend_loc)
        ax.ticklabel_format(style='sci', scilimits=(0,3))
        return fig
    
    def export_data(self, path):
        """
        saves the X, Y and Z-data to 3 textfiles in order to be used with other programs
        """
        pX = path+'_X'
        pY = path+'_Y'
        pZ = path+'_Z'
        np.savetxt(pX, self.mX, delimiter=',')
        np.savetxt(pY, self.mY, delimiter=',')
        np.savetxt(pZ, self.data, delimiter=',')
        

class eNMR_old_pre_VC(eNMR_Measurement):
    #'''
    #This is the subsubclass of Masurement() and subclass of eNMR_Methods specialised to process data obtained from the experimental Schönhoff set-up
    
    #path:
        #relative or absolute path to the measurements folder
    #measurement:
        #the to the experiment corresponding EXPNO
    #alias:
        #Here you can place an individual name relevant for plotting. If None, the path is taken instead.    
    #Uink:
        #voltage increment. Usually extracted from the title file if defined with e.g. "Uink = 10V"
        #If Uink cannot be found or is wrong it can be entered manually during the data import.
        #The voltage list is calculated from the voltage increment and the vc list when the incrementation loop is used in the pulse program
    #dependency:
        #'U': voltage dependent eNMR measurement
        #'G': fieldgradient dependent eNMR measurement

    #linebroadening:
        #setting a standard-value for the linebroadening.
    #'''
    def __init__(self, path, expno, Uink=None, dependency="U", alias=None, linebroadening=0.5, electrode_distance=2.2e-2):
        Measurement.__init__(self, path, expno, linebroadening=linebroadening, alias=alias)
        self.dependency = dependency.upper()
        
        self._x_axis = {"U": "U / [V]",
                   "G": "g in T/m",
                   "I": "I / mA",
                   'RI': 'RI / V'
                   }[self.dependency.upper()]
        
        #self._x_axis = {"G": "g in T/m",
                        #"U": "U / [V]"}[self.dependency.upper()]
        
        self.difflist = pd.read_csv(self.dateipfad+"/difflist",
                            names=["g in T/m"])*0.01
        
        self.vcList = pd.DataFrame()
        
        if self.dic['acqus']['PULPROG'][-3:] == 'var':
            polarity = 1
            print('this is a regular measurement! (non-_pol)')
        elif self.dic['acqus']['PULPROG'][-3:] == 'pol':
            polarity = -1
            print('this is a _pol-Measurement!')
        else:
            print("no var or pol PULPROG")
    
        if dependency.upper() == "U":
            try:
                # takes the title page to extract the volt increment
                title = open(self.dateipfad+"/pdata/1/title").read()
                # gets the voltage increment using a regular expression
                #uimport = findall('[U|u]in[k|c]\s*=?\s*\d+', title)[0]
                uimport = findall('[U|u]in[k|c]\s*=+\s*\d+', title)[0]
                self.uInk = int(findall('\d+', uimport)[0])
            except ValueError:
                print('no volt increment found\nyou may want to put it in manually')
                self.uInk = Uink
            except IndexError:
                print('No Uink found! May not be an eNMR experiment.')
                self.uInk = Uink
            
            self.vcList["U / [V]"] = [i*self.uInk*polarity for i in range(len(self.difflist))]
                
        elif dependency.upper() == "G":
            try:
                # takes the title page to extract the volt increment
                title = open(self.dateipfad+"/pdata/1/title").read()
                # gets the voltage increment using a regular expression
                uimport = findall('[U|u]\s*=?\s*\d+', title)[0]
                self.uInk = int(findall('\d+', uimport)[0])

            except ValueError:
                print('no volt increment found\nyou may want to put it in manually')
                self.uInk = Uink
            except IndexError:
                print('No Uink found! May not be an eNMR experiment.')
                self.uInk = Uink # Uinktext
                
            #if Uink is not None:
                #self.uInk = Uink
            
            self.vcList["U / [V]"] = [self.uInk*polarity for i in range(len(self.difflist))]
            #self.vcList["U / [V]"] = [self.vcList["vc"][n]/2*self.uInk if self.vcList["vc"][n] % 2 == 0
                                    #else (self.vcList["vc"][n]+1)/2*self.uInk*-1
                                    #for n in range(len(self.data[:, 0]))]

        #if self.dependency.upper() == "U":
            #try:
                #self.vcList = pd.read_csv(self.dateipfad+"/vclist",
                                          #names=["vc"]).loc[:len(self.data[:, 0])-1]
                
            #except:
                #print("There is a Problem with the VC-list or you performed a gradient dependent measurement")
        #elif self.dependency.upper() == "G":
            #self.vcList = pd.DataFrame(np.ones((len(self.data[:, 0]), 1)),
                                       #columns=["vc"])
        #else:
            #print("The dependency is not properly selected, try again!")

        #self.difflist = pd.read_csv(self.dateipfad+"/difflist",
                                    #names=["g in T/m"])*0.01
        
            #if Uink is not None:
                #self.uInk = Uink
                
            #self.vcList["U / [V]"] = [self.vcList["vc"][n]/2*self.uInk if self.vcList["vc"][n] % 2 == 0
                                    #else (self.vcList["vc"][n]+1)/2*self.uInk*-1
                                    #for n in range(len(self.data[:, 0]))]
            
        # try to open phase data, otherwise create new
        try:
            self.eNMRraw = pd.read_csv(self.path+"phase_data_"+self.expno+".csv",
                                       index_col=0, sep=" ")
            # --> update voltage list
            self.eNMRraw["U / [V]"] = self.vcList["U / [V]"]
        except:
            print("eNMRraw was missing and is generated")
            self.vcList["ph0"] = np.zeros(len(self.data.real[:, 0]))
            self.eNMRraw = self.vcList
        finally:
            self.eNMRraw["g in T/m"] = self.difflist
        
        self.p1 = self.dic["acqus"]["P"][1]
        self.d1 = self.dic["acqus"]["D"][1]
        
        try:
            # import of diffusion parameters for newer Spectrometers
            import xml.etree.ElementTree as etree
            diffpar = etree.parse(self.dateipfad+'/diff.xml')
            root = diffpar.getroot()
            self.Delta = float(root.findall('DELTA')[0].text)*1e-3
            self.delta = float(root.findall('delta')[0].text)*1e-3  # it should be read as in microseconds at this point due to bruker syntax
            print('The diffusion parameters were read from the respectie .XML!')
        except:
            # determination of the diffusion parameters for Emma
            self._d2 = self.dic["acqus"]["D"][2]
            self._d5 = self.dic["acqus"]["D"][5]
            self._d9 = self.dic["acqus"]["D"][9]
            self._d11 = self.dic["acqus"]["D"][11]
            self._p19, self._p18, self._p17 = self.dic["acqus"]["P"][19],\
                                            self.dic["acqus"]["P"][18],\
                                            self.dic["acqus"]["P"][17]
            print('That did not work. Your data is from an old spectrometer!')
            # calculating usable parameters
            self.delta = self._p17+self._p18
            self._Delta_1 = 0.001*(self._p17*2+self._p18)+(self._d2+self._d9+self._d5+self._d11)*1000+0.001*self.p1+self._d11
            self._Delta_2 = 0.001*(self._p17*2+self._p18)+(self._d2+self._d9+self._d5+self._d11)*1000+0.001*self.p1*2
            self._spoiler = (self._d11+self._p17+self._p19+self._p17)*0.001+self._d2*1000
            self.Delta = self._Delta_1+self._Delta_2+2*self._spoiler
            self.Delta *=1e-3
            self.delta *=1e-6
            

        # Elektrodenabstand in m
        self.d = electrode_distance
        self.g = self.eNMRraw["g in T/m"][0]
    
    def __add__(self, other):
        
        for obj in [self, other]:
            for k in obj.eNMRraw.columns:
                if k[:2] == 'ph':
                    obj.eNMRraw[k] -= obj.eNMRraw.loc[0, k]
                    print('%s normalized to 0V'%k)
                else:
                    pass
        self.eNMRraw = self.eNMRraw.append(other.eNMRraw)
        self.eNMRraw.sort_values('U / [V]', inplace=True)
        return self
