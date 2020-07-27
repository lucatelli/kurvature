#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Codiname:  Pelicoto
Version :  alpha v2 (07/2020)
by Geferson Lucatelli

This is the alpha version of the curvature code for a
1D galaxy light profile.



 $$\   $$\                                        $$\
$$ | $$  |                                       $$ |
$$ |$$  /$$\   $$\  $$$$$$\ $$\    $$\ $$$$$$\ $$$$$$\   $$\   $$\  $$$$$$\   $$$$$$\
$$$$$  / $$ |  $$ |$$  __$$\\$$\  $$  |\____$$\\_$$  _|  $$ |  $$ |$$  __$$\ $$  __$$\
$$  $$<  $$ |  $$ |$$ |  \__|\$$\$$  / $$$$$$$ | $$ |    $$ |  $$ |$$ |  \__|$$$$$$$$ |
$$ |\$$\ $$ |  $$ |$$ |       \$$$  / $$  __$$ | $$ |$$\ $$ |  $$ |$$ |      $$   ____|
$$ | \$$\\$$$$$$  |$$ |        \$  /  \$$$$$$$ | \$$$$  |\$$$$$$  |$$ |      \$$$$$$$\
\__|  \__|\______/ \__|         \_/    \_______|  \____/  \______/ \__|       \_______|

"""
########################################
from __future__ import division

import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import scipy.integrate as si
import scipy.signal
import argparse
import os
import pandas as pd
from matplotlib import gridspec
import pylab as pl
from matplotlib import cm
from scipy import signal
import statistics as stat
from scipy.stats import mode
import astropy.io.fits as pf
import scipy.ndimage as snd
from medpy.filter.smoothing import anisotropic_diffusion as AD
from scipy.interpolate import CubicSpline
from scipy.interpolate import splev,splrep
from scipy.optimize  import   leastsq, fmin, curve_fit
import scipy.ndimage as nd
from scipy.stats   import spearmanr
import kurvature_libs as klibs
import sys
import warnings
warnings.filterwarnings("ignore")
import matplotlib
# matplotlib.use('Qt5Agg')
# matplotlib.use('WebAgg')
matplotlib.use('GTK3Agg')

from matplotlib import use as mpluse
from scipy.interpolate import InterpolatedUnivariateSpline
# mpluse('Agg')



class config():
    """
    Configuration section.

        make_plots:       make or not the plots.

        Show: 		      If True, it will display the main plots.

        Plot_filter:      If True, displays the original profile and the
                            filtered one.

        normalize:	      Normalize (by maximum peak) or not the curvature.

        mean_over_PA:     Average K(R) for all possible PA's.
                          Note: Takes more time to process.

        filter_prop_Smax: Percentage of the radii range to use for
                            \sigma_max of the adaptive gaussian filter.

        filter_prop_Smin: Percentage of the radii range to use for
                            \sigma_min of the adaptive gaussian filter.

    """

    try:
        from jupyterthemes import jtplot
        jtplot.style('monokai')

        jtplot.style(grid=False, ticks=True, spines=True,gridlines='--')
        from matplotlib import rcParams,rc
        rcParams['font.family'] = 'sans-serif'
        rcParams['font.sans-serif'] = ['Cormorant Infant']
        # rcParams['font.sans-serif'] = ['Cormorant']
        plt.rc('xtick', labelsize=20)#,color="magenta")
        plt.rc('ytick', labelsize=20)#,color="magenta")
        # rc('text', usetex=True)
    except:
        pass
    # Show = True #show the plots
    if "--show" in sys.argv:
        Show = True #show the plots
    else:
        Show = False
        mpluse('Agg')

    PLOT_filter = True
    PLOT_extrapolation = False
    make_plots = False
    normalize = False
    mean_over_PA = True
    SAVE_data = True

    #these are the percentage of the length of R to be used for
    # sigma_max (Smax) and sigma_min (Smin).
    #That is, Smax ~ len(R) * filter_prop_Smax.
    filter_prop_Smin = 0.001
    filter_prop_Smax = 0.1

    plot_in_ARCSEC = True
    if plot_in_ARCSEC is True:
        # define the pixel scale manually of the instrument where the
        # image fits were taken.
        pxsc = 0.396#SDSS DR4-7
        # pxsc = 0.13#HST
        # pxsc = 0.75#Spitzer
        # pxsc = 0.25#PS
        # pxsc = 0.432#CTIO
        # pxsc = 0.55 #T80Cam/S-PLUS DR1
        # pxsc = 0.2267 #JPAS

    def __init__(self):
        print("Initializing kurvature")


class data_handle():
    """
    A class to deal with data reading and selection.
    """
    def __init__(self,profile=None,err=None,radius=None,petro=None,multiple_profiles=None):

        #get the data from the input files
        # print("Importing profile from input file...")

        # profile is the only required variable, but if --mpr is given, this
        # priority/requirement will be override to all tha is inside 'if' bellow:


        # else:
        self.filename = profile
        self.rootname = self.filename.replace('.*', '')
        # self.error   = klibs.get_data(param="IRerr",File = self.filename)
        self.profile = klibs.get_data(param="IR",File = self.filename)
        if radius is not None:
            self.radius_filename=radius
            try:
                self.radius = klibs.get_data(param="# Raios",File = \
                    self.radius_filename)
                print("    # Importing radius from input file...")
            except:
                self.radius = np.linspace(1.0,2*self.profile.shape[0],self.profile.shape[0])
                print("    # Creating standard radii...")
        else:
            try:
                self.radius = klibs.get_data(param="# Raios",File = self.filename)
                print("    # Using radius from profile input file....")
            except:
                self.radius = np.linspace(1.0,2*self.profile.shape[0],self.profile.shape[0])
                print("    # Creating standard radii...")

        if petro is None:
            self.RP = int(self.radius[-1]/2)
        else:
            self.RP = petro


        # self.profile = self.profile[:-2]
        # self.radius = self.radius[:-2]

        if "--mpr" in sys.argv:
            self.filename_mpr = multiple_profiles
            self.rootname_mpr = self.filename_mpr.replace('.*', '')
            # self.multiple_profiles = multiple_profiles
            hd = klibs.get_str(File = self.filename_mpr,HEADER=1)
            dic_profiles = {}
            for k in range(len(hd)):
                dic_profiles[hd[k]] = klibs.get_data(File = self.filename_mpr,param=hd[k])

            self.dic_profiles =  dic_profiles
            # print(dic_profiles)


            if radius is not None:
                self.radius_filename=radius
                hdR = klibs.get_str(File = self.radius_filename,HEADER=1)
                dic_R = {}
                for k in range(len(hdR)):
                    dic_R[hdR[k]] = klibs.get_data(File = self.filename_mpr,param=hdR[k])
                self.dic_R = dic_R
                if petro is None:
                    self.RP = int(self.dic_R[hdR[-1]][-1]/2)
                else:
                    self.RP = petro

            else:
                dic_R = {}
                for k in range(len(hd)):
                    # create a radii data with the same lengh of the profile (in
                    # the same order), with spacing units of 2.
                    # dic_R["R"+str(k)] = np.arange(1.0,self.dic_profiles[hd[k]].shape[0]+1,2)
                    dic_R["R"+str(k)] = np.linspace(1.0,2*self.dic_profiles[hd[k]].shape[0],self.dic_profiles[hd[k]].shape[0])
                self.dic_R = dic_R

                if petro is None:
                    self.RP = np.asarray(self.dic_R["R"+str(k)][-1]/2).astype(int)
                else:
                    self.RP = petro

        # print(self.radius)
        # print(self.profile)

        # print("sdfjsçkja fsdfjsdoiçfjdsifasdçl")
        # print()
        # self.radius = self.dic_R[list(dic_R.keys())[-1]]
        # print(self.radius)


        # self.profile = signal.resample(self.profile,3*len(self.profile),window=1)
        # self.radius   = signal.resample(self.radius ,3*len(self.radius),window=1)
        # print(self.profile)

        def savitzky_golay(y, window_size, order, deriv=0, rate=1):
            from math import factorial

            try:
                window_size = np.abs(np.int(window_size))
                order = np.abs(np.int(order))
                if (window_size % 2) == 0:
                   window_size += 1
            except ValueError as msg:
                raise ValueError("window_size and order have to be of type int")
            if window_size % 2 != 1 or window_size < 1:
                raise TypeError("window_size size must be a positive odd number")
            if window_size < order + 2:
                raise TypeError("window_size is too small for the polynomials order")
            order_range = list(range(order+1))
            half_window = (window_size -1) // 2
            # precompute coefficients
            b = np.mat([[k**i for i in order_range] for k in range(-half_window, half_window+1)])
            m = np.linalg.pinv(b).A[deriv] * rate**deriv * factorial(deriv)
            # pad the signal at the extremes with
            # values taken from the signal itself
            firstvals = y[0] - np.abs( y[1:half_window+1][::-1] - y[0] )
            lastvals = y[-1] + np.abs(y[-half_window-1:-1][::-1] - y[-1])
            y = np.concatenate((firstvals, y, lastvals))
            return np.convolve( m[::-1], y, mode='valid')

        # plt.plot(self.radius,np.log(savitzky_golay(self.profile, 10,3)))
        # plt.plot(self.radius,np.log(self.profile),'.')
        # self.profile = savitzky_golay(self.profile, 10,3)
        # data = InterpolatedUnivariateSpline(self.radius[:-3],self.profile[:-3],k=5)
        # R_sampled = np.linspace(self.radius[:-3][0],self.radius[:-3][-1],100)
        # data_s = data(R_sampled)
        # print(data_s)
        # plt.plot(self.radius,np.log(savitzky_golay(self.profile, 10,3)))
        # plt.plot(R_sampled,np.log(data_s),'.')
        # plt.plot(self.radius[:-3],np.log(self.profile[:-3]),'.')
        # plt.show()

        # self.profile = data(R_sampled)
        # self.profile = savitzky_golay(self.profile, 7,3)
        # self.radius = R_sampled
        fwhm_min = 1    #drop some points at inner regions (reduce psf or
                         #oversampling effects)
        self.Rmin = fwhm_min
        self.index_min = self.Rmin

        #Define the minimum signal to noise you want to work.
        if "--err" in sys.argv:
            try:
                self.error = klibs.get_data(param="IRerr",File = \
                    self.filename)
                self.signal_to_noise1 = 1.0
                self.snrr = np.where((self.profile[self.index_min:]/self.error[self.index_min:])>\
                    self.signal_to_noise1)[0]
                self.index_max= int(self.snrr[-1])
                self.MUR =-2.5*np.log10(self.profile[self.Rmin:])
                self.MURerr = -2.5/(np.log(10) * self.profile[self.index_min:]) * self.error[self.index_min:]

            except:
                # print("Error in loading the profile error data.")
                self.index_max = int(klibs.find_nearest(self.radius/self.RP,1.5)[1]+1)
        else:
            # print("Not using error data.")
            self.index_max = int(klibs.find_nearest(self.radius/self.RP,1.5)[1]+1)
        # print((self.profile[self.Rmin:]/self.error[self.Rmin:]))

        #drop points in outer regions.
        # self.index_max = int(klibs.find_nearest(self.radius/self.RP,1.3)[1]+1)

class kurvature():
    """
    The main class of the code. It perform any kurvature related
    calculations.
    """
    def __init__(self,kurv):
        self.K = kurv
        # self.normalise_in_log()
        self.filter()


    def filter(self):
        "Apply the filter routine to the input profile."
        # self.normalise_in_log()
        def extrapolate_data(profile,R,Npts):
            """
            Smooth extrapolation at the inner edge.
            This is done to avoid edge effects in the filter procedure or
            psf effects. However, this function must be used carefully since in
            some cases there is no convergence of a 'smooth extrapolation.'

            Further improvements still are needed.
            """

            r_m = -R[::-1][-Npts-1:-1]
            radius_extr = np.append(r_m,R)

            #This extrapolation method also smooth the data a little.
            #But, it does not remove information of the signal.
            #The actual filter is applied next.

            cs=CubicSpline(R,profile,bc_type='natural')
            profile_extr = cs(radius_extr)

            if config.PLOT_extrapolation==True:
                fig = plt.figure()
                fig.set_size_inches(6.0, 5.0)
                plt.plot(R,profile,label="original",lw=7,alpha=0.6)
                plt.plot(radius_extr,profile_extr,label="(inter + extra)polation",\
                    lw=4,alpha=0.9)
                plt.legend()
                plt.grid()
                # plt.show()
                plt.clf()
                plt.close()
            return profile_extr,radius_extr
        # print(normalise_in_log(self)

        def apply_filter(mu_0,nu_0,radius_norm):
            def gaussian(x, x0, sigma):
                """
                Gaussian function.
                """
                return 1./(sigma*np.sqrt(2.*np.pi)) * np.exp(-(x-x0)**2./(2.*sigma**2.))

            def adaptive_gaussian(self,profile,R):
                """
                sigma-variable adaptive gaussian filter.
                """
                I = profile.copy()
                Smax = config.filter_prop_Smax * len(profile)
                Smin = config.filter_prop_Smin * len(profile)
                # print("Smin = ",Smin," Smax = ", Smax)

                #Linear sigma
                Sigma = ((Smax*0.75-Smin)/R[-1])*R + Smin
                Inew = np.zeros_like(I)
                #apply the adaptive filter
                for i in range(len(I)):
                    G = gaussian(np.arange(len(I)),  float(i), Sigma[i] )
                    G=G/G.sum()
                    Inew[i] =  (I * G).sum()
                    # print Sigma[i]
                return Inew

            def lfilt(data):
                """
                scipy.signal.lfilter
                https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.lfilter.html
                """
                # b,a  = scipy.signal.butter(1,0.1)
                # b,a  = scipy.signal.bessel(3,0.2)

                # b,a  = scipy.signal.ellip(3,5,40,0.3)
                # b,a = scipy.signal.cheby2(3,3.,[0.01,0.5],btype="bandpass")
                # b,a  = scipy.signal.iirnotch(0.4,9)
                # b,a  = scipy.signal.cheby1(5,0.5,[0.15,0.9],btype="bandstop")
                # b,a  = scipy.signal.ellip(3,4,5,[0.1,0.9],btype="bandstop")
                # b,a  = scipy.signal.bessel(3,[0.2,0.99],btype="bandstop")
                # b,a = scipy.signal.cheby2(3,15.,[0.15,0.9],btype="bandstop")
                b,a  = scipy.signal.iirfilter(3,[0.10,0.9],btype="bandstop")
                # b,a  = scipy.signal.butter(3,[0.15,0.9],btype="bandstop")
                zi   = scipy.signal.lfilter_zi(b,a)
                z, _ = scipy.signal.lfilter(b,a,data,zi = zi*data[0])
                z2,_ = scipy.signal.lfilter(b,a,z,zi=zi*z[0])
                y    = scipy.signal.filtfilt(b,a,data,padlen = 5 )
                #padlen may be have the same results in t

                # plt.figure()
                # plt.plot(raio, xn, 'b', alpha=0.75)
                # plt.plot(raio, z, 'r--', raio, z2, 'r', raio, y, 'k')
                # plt.legend(('noisy signal', 'lfilter, once', 'lfilter, twice',
                #             'filtfilt'), loc='best')
                # plt.grid(True)
                # plt.show()
                return y
            def _extrapolate(x,y,Npts):
                from scipy.interpolate import CubicSpline
                xnew = np.linspace(x[0],x[-1]+Npts/(2*self.K.RP),len(x)+Npts)
                # print(x,xnew)
                cs =CubicSpline(x,y,bc_type ="natural")
                ynew = cs(xnew)
                return xnew,ynew

            def extrapola_IR_after(R,I,Npts):
                """
                Interpolate some artificial data points before IR data at R=0.
                We need to do this in order to avoid edge effects at the inermost
                region.
                """

                IrCut = I[-Npts:-1]
                RaiosCut = R[-Npts:-1]
                cs =CubicSpline(R,I,bc_type ="natural")
                Raios_more = np.arange(R[-1],R[-1]+Npts/(2*self.K.RP),1/(2*self.K.RP))
                # print(Raios_more)
                Raios_new = np.append(R,Raios_more)
                IR_new = cs(Raios_new)

                return Raios_new,IR_new

            # Select the number of points to extrapolate.
            Npts=2 #at the iner region
            Nptsf = 2# at the outer region
            #First, apply a smooth adaptive gaussian filter.
            mu_0_gauss = adaptive_gaussian(self,mu_0,radius_norm)
            #Second, extrapolate the data profile_gauss (log of the profile)
            mu_0_extr,radius_extr = extrapolate_data(mu_0_gauss,radius_norm,Npts)
            #call IIR or FIR filter to the extrapolated signal.
            mu_extr = lfilt(mu_0_extr)
            #after filtering, drop extrapolated points.
            mu = mu_extr[Npts:]

            #repeat the task, but fot the normalised-log profile.
            nu_0_G = adaptive_gaussian(self,nu_0,radius_norm)
            nu_0_extr, radius_extr = extrapolate_data(nu_0_G,radius_norm,Npts)

            radius_extr2,nu_0_extr2 = extrapola_IR_after(radius_extr[-Npts:],nu_0_extr[-Npts:],Nptsf)
            nu_0_extr2 = np.append(nu_0_extr[:-Npts],nu_0_extr2)
            radius_extr2 = np.append(radius_extr[:-Npts],radius_extr2)
            # plt.plot(self.radius_extr,self.nu_0_extr,label="extr1")
            # plt.plot(radius_extr2,nu_0_extr2,label="extr2")
            # plt.legend()
            # plt.show()

            log_normalised_filtered_signal = lfilt(nu_0_extr)
            #drop extrapolated points
            # log_normalised_signal = log_normalised_signal[Npts:-Nptsf]
            log_normalised_filtered_signal = log_normalised_filtered_signal[Npts:]
            # log_normalised_signal = log_normalised_signal[:-Npts]
            return log_normalised_filtered_signal,nu_0_extr,radius_extr

        def normalise_in_log(profile,radius):
            """
            Normalise profile and radius.

            See the definition for why this is needed.
            (Lucatelli & Ferrari, 2019).
            """

            y = profile.copy()
            x = radius.copy()
            mu_0 = np.log(y)#mu behaves like magnitude (e.g. log(I))
            nu_0 = klibs.normalise(np.log(y))
            radius_norm = klibs.normalise(x)
            return mu_0,nu_0,radius_norm
            # plt.plot(self.radius_norm,self.nu)
            # plt.show()

        # now, I need that the code calculates the curvature for every profile,
        # in the case --mpr arg is given.
        def k_R(f,r):
            """
            Call kurvature maths.
            """
            #First derivatites/differentials
            df  =  np.gradient(f,1.0)
            dr  =  np.gradient(r,1.0)
            #second derivatives/differentials
            d2f =  np.gradient(df,1.0)
            d2r =  np.gradient(dr,1.0)
            #curvature
            k   =  (dr * d2f  - df * d2r ) / np.power(dr**2.0 + df**2.0,1.5)
            return k,df, dr, d2f, d2r

        if "--mpr" in sys.argv:
            # print(self.K.dic_profiles,self.K.dic_R)
            dic_kurv = {}
            dic_nu = {}
            kurv_list = []
            nu_list = []
            radius_norm_list = []
            nu_0_list = []
            radius_extr_list = []
            nu_0_extr_list = []
            profile_list = []
            nu_list = []
            for k in range(len(self.K.dic_profiles.keys())):
                profile = np.asarray(list(self.K.dic_profiles.values())[k])
                profile_list.append(profile)
                radius  = np.asarray(list(self.K.dic_R.values())[k])
                mu_0,nu_0,radius_norm = normalise_in_log(profile,radius)
                # print(list(self.K.dic_profiles.values())[k])
                # plt.plot(list(self.K.dic_R.values())[k],list(self.K.dic_profiles.values())[k])
                nu,nu_0_extr,radius_extr = apply_filter(mu_0,nu_0,radius_norm)

                kurvature_filter, dnu_s, dR_extr,d2nu_s,\
                    d2R_extr  = k_R(nu,radius_norm)

                dic_nu["nu"+str(k)]   = nu
                dic_kurv["kR"+str(k)] = kurvature_filter
                kurv_list.append(kurvature_filter)
                nu_list.append(nu)
                radius_norm_list.append(radius_norm)
                nu_0_list.append(nu_0)
                radius_extr_list.append(radius_extr)
                nu_0_extr_list.append(nu_0_extr)

            self.K.nu = nu_list[0]
            self.K.radius_norm = radius_norm_list[0]
            self.K.nu_0 = nu_0_list[0]
            self.K.radius_extr = radius_extr_list[0]
            self.K.nu_0_extr = nu_0_extr_list[0]
            self.K.kurvature_filter = kurv_list[0]
            self.K.kurv_list = kurv_list
            self.K.nu_list = nu_list
            self.K.nu_0_list = nu_0_list
            self.K.radius_norm_list = radius_norm_list

            #drop positions 0 and -1 (0 is the main and -1 is the mean)
            self.K.mean_kurv = np.mean(np.asarray(kurv_list)[1:-1],axis=0)
            self.K.mean_profile = np.asarray(profile_list[-1])
            self.K.std_kurv = np.mean(np.asarray(kurv_list)[1:-1],axis=0)
            self.K.mean_radius_norm = np.mean(np.asarray(radius_norm_list)[1:-1],axis=0)
            KVAR  = []
            KMEAN = []
            KSTD  = []
            PVAR  = []
            PMEAN = []
            PSTD  = []
            nuVAR  = []
            nuMEAN = []
            nuSTD  = []


            # KMEAN,KSTD,KVAR = np.asarray(KMEAN), np.asarray(KSTD), np.asarray(KVAR)
            K_sum_std  = np.sum(np.std(kurv_list[1:-1][:self.K.index_max],axis=0))
            K_sum_var  = np.sum(np.var(kurv_list[1:-1][:self.K.index_max],axis=0))
            K_sum_mean = np.sum(np.mean(kurv_list[1:-1][:self.K.index_max],axis=0))/len(kurv_list[0])


            K_mean_std  = np.mean(np.std(kurv_list[1:-1][:self.K.index_max],axis=0))
            K_mean_var  = np.mean(np.var(kurv_list[1:-1][:self.K.index_max],axis=0))
            K_mean_mean = np.mean(np.mean(kurv_list[1:-1][:self.K.index_max],axis=0))/len(kurv_list[0])

            K_var_std  = np.var(np.std(kurv_list[1:-1][:self.K.index_max],axis=0))
            K_var_var  = np.var(np.var(kurv_list[1:-1][:self.K.index_max],axis=0))
            K_var_mean = np.var(np.mean(kurv_list[1:-1][:self.K.index_max],axis=0))/len(kurv_list[0])

            IR_sum_std  = np.sum(np.std(profile_list[1:-1][:self.K.index_max],axis=0))
            IR_sum_var  = np.sum(np.var(profile_list[1:-1][:self.K.index_max],axis=0))
            IR_sum_mean = np.sum(np.mean(profile_list[1:-1][:self.K.index_max],axis=0))/len(profile_list[0])

            IR_mean_std  = np.mean(np.std(profile_list[1:-1][:self.K.index_max],axis=0))
            IR_mean_var  = np.mean(np.var(profile_list[1:-1][:self.K.index_max],axis=0))
            IR_mean_mean = np.mean(np.mean(profile_list[1:-1][:self.K.index_max],axis=0))/len(profile_list[0])

            IR_var_std  = np.var(np.std(profile_list[1:-1][:self.K.index_max],axis=0))
            IR_var_var  = np.var(np.var(profile_list[1:-1][:self.K.index_max],axis=0))
            IR_var_mean = np.var(np.mean(profile_list[1:-1][:self.K.index_max],axis=0))/len(profile_list[0])

            nu_sum_std  = np.sum(np.std(nu_list[1:-1][:self.K.index_max],axis=0))
            nu_sum_var  = np.sum(np.var(nu_list[1:-1][:self.K.index_max],axis=0))
            nu_sum_mean = np.sum(np.mean(nu_list[1:-1][:self.K.index_max],axis=0))/len(nu_list[0])

            nu_mean_std  = np.mean(np.std(nu_list[1:-1][:self.K.index_max],axis=0))
            nu_mean_var  = np.mean(np.var(nu_list[1:-1][:self.K.index_max],axis=0))
            nu_mean_mean = np.mean(np.mean(nu_list[1:-1][:self.K.index_max],axis=0))/len(nu_list[0])

            nu_var_std  = np.var(np.std(nu_list[1:-1][:self.K.index_max],axis=0))
            nu_var_var  = np.var(np.var(nu_list[1:-1][:self.K.index_max],axis=0))
            nu_var_mean = np.var(np.mean(nu_list[1:-1][:self.K.index_max],axis=0))/len(nu_list[0])
            # print("asdsadsadsa ", self.K.index_max, len(nu_list[1]))
            self.K.K_sum_mean = K_sum_mean
            self.K.IR_sum_mean = IR_sum_mean
            self.K.nu_sum_mean = nu_sum_mean

            data_type = "multiple_profiles_data"
            #
            # data_to_save = np.asarray([Kmean_of_mean,Kmean_of_var,Kmean_of_std,KPmean_of_mean,KPmean_of_var,KPmean_of_std,numean_of_mean,numean_of_var,numean_of_std])
            # labes_do_save = ["Kmean_of_mean","Kmean_of_var","Kmean_of_std","KPmean_of_mean","KPmean_of_var","KPmean_of_std","numean_of_mean","numean_of_var","numean_of_std"]

            base = os.path.splitext(os.path.basename(self.K.filename.replace('_IR.csv', '')))[0]
            f = open(self.K.rootname+"_"+data_type+".csv",'w')
            f.write("#gal_name,K_sum_std,K_sum_var,K_sum_mean,K_mean_std,K_mean_var,K_mean_mean,IR_sum_std,IR_sum_var,IR_sum_mean,IR_mean_std,IR_mean_var,IR_mean_mean,nu_sum_std,nu_sum_var,nu_sum_mean,nu_mean_std,nu_mean_var,nu_mean_mean,K_var_std,K_var_var,K_var_mean,IR_var_std,IR_var_var,IR_var_mean,nu_var_std,nu_var_var,nu_var_mean,"+'\n')
            # f.write("#gal_name,Kmean_of_mean,Kmean_of_var,Kmean_of_std,KPmean_of_mean,KPmean_of_var,KPmean_of_std,numean_of_mean,numean_of_var,numean_of_std,"+'\n')
            f.write(base+','+str(K_sum_std)+','+str(K_sum_var)+','+str(K_sum_mean)+','+str(K_mean_std)+','+str(K_mean_var)+','+str(K_mean_mean)+','+str(IR_sum_std)+','+str(IR_sum_var)+','+\
                str(IR_sum_mean)+','+str(IR_mean_std)+','+str(IR_mean_var)+','+str(IR_mean_mean)+','+str(nu_sum_std)+','+str(nu_sum_var)+','+str(nu_sum_mean)+','+str(nu_mean_std)+','+\
                str(nu_mean_var)+','+str(nu_mean_mean)+','+str(K_var_std)+','+str(K_var_var)+','+str(K_var_mean)+','+str(IR_var_std)+','+str(IR_var_var)+','+str(IR_var_mean)+','+str(nu_var_std)+','+\
                str(nu_var_var)+','+str(nu_var_mean)+','+'\n')
            # f.write(base+','+str(Kmean_of_mean)+","+str(Kmean_of_var)+","+str(Kmean_of_std)+","+str(KPmean_of_mean)+","+str(KPmean_of_var)+","+str(KPmean_of_std)+","+str(numean_of_mean)+","+str(numean_of_var)+","+str(numean_of_std)+','+'\n')
            f.close()

            # print(data_to_save)
            # df = pd.DataFrame(data_to_save.T)
            #
            # df.to_csv(self.K.rootname+"_"+data_type+".csv",index=False,header=labes_do_save)#,header=labes_do_save)

            # plt.plot(mean_radius_norm,mean_kurv)

        else:
            self.K.mu_0,self.K.nu_0,self.K.radius_norm = normalise_in_log(self.K.profile.copy(),self.K.radius.copy())
            self.K.nu,self.K.nu_0_extr,self.K.radius_extr = apply_filter(self.K.mu_0,self.K.nu_0,self.K.radius_norm)
        # self.mu = apply_filter(self.nu_0_extr,self.radius_extr)

            #call k_R for filtered/smoothed data
            self.K.kurvature_filter, self.K.dnu_s, self.K.dR_extr,self.K.d2nu_s,\
                self.K.d2R_extr  = k_R(self.K.nu,self.K.radius_norm)

            #call k_R for original data
            # please, do not use this data (only for comparison)
            self.K.kurvature,self.K.dnu,self.K.dR,self.K.d2nu,self.K.d2R = \
                k_R(self.K.nu_0,self.K.radius_norm)

class kur_2D():
    def __init__(self,kurv,image_name):
        self.K = kurv
        self.K.image_name = image_name
        self.compute()
        # self.load_image()
        # self.log_contrast_enhance()

    def compute(self):
        def load_image(image_name):
            import astropy.io.fits as pf
            self.K.gal_image_0,self.K.hdr = pf.getdata(image_name, header=1)
            self.K.Mo,self.K.No = self.K.gal_image_0.shape
            try:
                self.K.x0header = self.hdr['OBJXPIX']
                self.K.y0header = self.hdr['OBJYPIX']
            except:
                pass

        def log_contrast_enhance(image):
            log_image = (np.log((image)))
            # print log_image
            where_are_NaNs = np.isnan(log_image)
            where_are_infs = np.isinf(log_image)
            log_image[where_are_NaNs] = 100 		 #gambiarra
            log_image[where_are_infs] = 100			 #gambiarra
            where_are_100 = np.where(log_image==100) #gambiarra
            log_image[where_are_100] = log_image.min()
            nu_0 = klibs.normalise(log_image)
            return nu_0 #the enhanced image in log scale.

        load_image(self.K.image_name)
        # self.K.nu0_2D = log_contrast_enhance(self.K.gal_image_0)
        self.K.nu0_2D  = self.K.gal_image_0
        # self.K.nu2D   = klibs.lfilt(self.K.nu0_2D)

        # self.K.nu2D = scipy.ndimage.percentile_filter(self.K.nu0_2D,-25,15)
        self.K.nu2D = AD(self.K.nu0_2D, niter=60, gamma=0.1)
        # self.K.nu2D = scipy.ndimage.minimum_filter(self.K.nu0_2D,5)#good for segmentation
        # self.K.nu2D = scipy.ndimage.uniform_filter(self.K.nu0_2D,5)
        # self.K.nu2D = self.K.nu0_2D
        dx_filt, dy_filt = np.gradient(self.K.nu2D)
        # dx,dy = np.gradient(renormaliza(np.log(abs(self.gal_image))),1.0)

        #first 2D derivatives (filter)
        dI_2D_filt = np.sqrt(dx_filt**2.+dy_filt**2.)

        #first differentials (filter and no filter)
        d2x_filt, d2y_filt =np.gradient(dI_2D_filt,1)

        #second 2D derivatives (filter)
        d2I_2D_filt = np.sqrt(d2x_filt**2.0+d2y_filt**2.0)

        #Kurvature 2D (filter and no filter)
        self.K.Kurvature_2D_filter = ( d2I_2D_filt )/ (  1. + (dI_2D_filt)**2. )**(3/2.)

        self.K.dx_filt = dx_filt
        self.K.dy_filt = dy_filt

        self.K.d2x_filt = d2x_filt
        self.K.d2y_filt = d2y_filt



class do_statistics():
    """
    Do some statistics in the curvature profile.

    Notice that you need to chose what curvature profile will be used for
    evaluation of statistics: 1) The averaged over all PA's; 2) the single
    one, from the PA obtained by mfmtk. Select with the function
    select_what_curvature() bellow.
    """
    def __init__(self,kurv):
        self.K = kurv
        self.select_curvature()
        self.find_probable_peaks()
        self.create_sub_regions()
        self.entropies()
        self.kurv_area_and_stats()
        self.kurv_shape_analysis()

    def select_curvature(self):
        if "--mpr" in sys.argv:
            self.K.sel_kurvature = self.K.mean_kurv.copy()[self.K.index_min:self.K.index_max]
        else:
            self.K.sel_kurvature = self.K.kurvature_filter.copy()[self.K.index_min:self.K.index_max]

    def find_probable_peaks(self):
        from scipy.signal import find_peaks as fp
        from scipy.signal import peak_prominences as pkp
        from scipy.signal import argrelextrema as _argrelextrema
        from scipy.signal import argrelmin as _argrelmin

        # vlines_peaks = (fp(self.K.sel_kurvature[self.K.index_min:self.K.index_max],\
            # prominence=prominence,width=width)[0])/self.K.RP + (self.K.index_min)/self.K.RP
        def detect_peaks(method='argrelextrema'):

            self.K.Knorm = klibs.normalise(self.K.sel_kurvature.copy())
            # data = self.K.sel_kurvature.copy()[:self.K.index_max]
            data = self.K.sel_kurvature.copy()

            # plt.plot(data)
            # plt.show()
            if method=='fp':

                # prominence=[(abs(data.max())-abs(data.min())),data.max()]#-data.min()]
                # prominence=[(1.0*data.max()-15*np.mean(data)),2*data.max()]#-data.min()]
                prominence=[(0.4*data.max()-0*np.mean(data)),2*data.max()]#-data.min()]
                # prominence=[20.0,20.5]#-data.min()]
                width = [0.05*len(data),0.75*len(data)]
                height = [0.1*np.mean(data),data.max()]
                threshold = None#[0.0001,2.2]
                plateau_size =  None#[0.75,1.00]
                distance = None#0.2*len(data)
                rel_height = 0.5


                peaks,peaks_properties = fp(data, height=height, prominence=prominence,\
                    width=width,distance = distance,threshold = threshold,plateau_size = plateau_size, rel_height=rel_height)
                # print("peaks",self.K.radius[peaks])
                vlines_peaks = 2*peaks/(self.K.RP) + 2*(self.K.index_min)/self.K.RP

                remove_indexes = np.where(vlines_peaks>=1.5)[0]#remove biased peaks > after 1.25 Rp
                # print("Peaks",peaks)
                for to_remove in remove_indexes:
                    peaks_properties["peak_heights"] = np.delete(peaks_properties["peak_heights"],to_remove)
                    peaks_properties["prominences"] = np.delete(peaks_properties["prominences"],to_remove)
                    peaks_properties["left_bases"] = np.delete(peaks_properties["left_bases"],to_remove)
                    peaks_properties["right_bases"] = np.delete(peaks_properties["right_bases"],to_remove)
                    peaks_properties["widths"] = np.delete(peaks_properties["widths"],to_remove)
                    peaks_properties["width_heights"] = np.delete(peaks_properties["width_heights"],to_remove)
                    peaks_properties["left_ips"] = np.delete(peaks_properties["left_ips"],to_remove)
                    peaks_properties["right_ips"] = np.delete(peaks_properties["right_ips"],to_remove)
                    vlines_peaks = np.delete(vlines_peaks,to_remove)
                    peaks = np.delete(peaks,to_remove)

                # print(peaks)
                self.K.peaks_flag = 0

                if peaks.size==0:
                    # print("No peaks")
                    self.K.peaks_flag = 1
                    peaks_properties["peak_heights"] = np.asarray([1.0])
                    peaks_properties["prominences"] = np.asarray([1.0])
                    peaks_properties["left_bases"] = np.asarray([len(data)*0.5])
                    peaks_properties["right_bases"] = np.asarray([0.0])
                    peaks_properties["widths"] = np.asarray([1.0])
                    peaks_properties["width_heights"] =np.asarray([0.5])
                    peaks_properties["left_ips"] = np.asarray([(len(data)*0.3)])
                    peaks_properties["right_ips"] = np.asarray([(len(data)*0.4)])
                    peaks  = np.asarray([(len(data)*0.4)]).astype(int)
                    vlines_peaks = 2*peaks/len(self.K.sel_kurvature) + 0*(self.K.index_min)/self.K.RP


                #
                # print("peak_heights",peaks_properties["peak_heights"])
                # print("prominences",peaks_properties["prominences"])
                # print("left_bases",peaks_properties["left_bases"])
                # print("right_bases",peaks_properties["right_bases"])
                # print("widths",peaks_properties["widths"])
                # print("width_heights",peaks_properties["width_heights"])
                # print("left_ips",peaks_properties["left_ips"])
                # print("right_ips",peaks_properties["right_ips"])

                downs = _argrelmin(data,order=40)
                vlines_downs = 2*downs[0]/len(self.K.sel_kurvature) + 0*(self.K.index_min)/len(self.K.sel_kurvature)
                self.K.peaks_properties = peaks_properties
                self.K.downs = downs
                self.K.Peaks = peaks
                self.K.peaks_properties = peaks_properties
                self.K.peaks_properties["right_ips"] = (self.K.peaks_properties["right_ips"]+(self.K.index_min))
                self.K.peaks_properties["left_ips"] = (self.K.peaks_properties["left_ips"]+(self.K.index_min))
                # self.K.peaks_properties["right_ips"]+(self.K.index_min)/self.K.RP
                # print(peaks_properties["right_ips"]+1*(self.K.index_min))
                # peak_properties = peaks[1:]
                # print(peaks_properties)
                # print(peaks_properties["peak_heights"])

            if method=='argrelextrema':
                peaks = _argrelextrema(self.K.Knorm,comparator=np.greater,order=5,mode='wrap')
                vlines_peaks = 2*peaks[0]/len(self.K.Knorm) + 0*(self.K.index_min)/len(self.K.Knorm)
                self.K.Peaks = peaks[0]
                downs = _argrelmin(self.K.Knorm,order=40)
                vlines_downs = 2*downs[0]/len(self.K.Knorm) + 0*(self.K.index_min)/len(self.K.Knorm)
                # print(vlines_downs,vlines_peaks)
            if method=='peakdetect_sixtenbe':
                """
                NOT WORKING
                """
                import peakdetect as pk_six
                _peaks = pk_six.peakdetect(np.array(self.K.Knorm[:self.K.index_max]), lookahead = 3, delta=10)
                peaks = []
                for posOrNegPeaks in _peaks:
                    for peak in posOrNegPeaks:
                        peaks.append(peak[0])
                # print(_peaks,peaks)
                vlines_peaks = 2*peaks/len(self.K.Knorm) + 0*(self.K.index_min)/len(self.K.Knorm)
                self.K.Peaks = peaks

            if method=='peakutils':
                import peakutils.peak
                peaks = peakutils.peak.indexes(data,thres=0.01/max(data),min_dist = 10)
                # print(peaks)
                vlines_peaks = 2*peaks/len(self.K.Knorm) + 0*(self.K.index_min)/len(self.K.Knorm)
                downs = _argrelmin(self.K.Knorm,order=40)
                vlines_downs = 2*downs[0]/len(self.K.Knorm) + 0*(self.K.index_min)/len(self.K.Knorm)
                self.K.Peaks = peaks
                self.K.Downs = downs

            if method=="janko_slavic":
                import findpeaks
                peaks = findpeaks.findpeaks(data,spacing=5,limit=0.4)
                vlines_peaks = 2*peaks/len(self.K.Knorm) + 0*(self.K.index_min)/len(self.K.Knorm)
                # print(peaks)
                downs = _argrelmin(self.K.Knorm,order=40)
                vlines_downs = 2*downs[0]/len(self.K.Knorm) + 0*(self.K.index_min)/len(self.K.Knorm)
                self.K.Peaks = peaks
                self.K.Downs = downs

            if method=="tony_belt":
                from tony_beltramelli_detect_peaks import detect_peaks as dpk
                peaks=np.asarray(dpk(data,1))
                # print(peaks)
                vlines_peaks = 2*peaks/len(self.K.Knorm) + 0*(self.K.index_min)/len(self.K.Knorm)
                downs = _argrelmin(self.K.Knorm,order=40)
                vlines_downs = 2*downs[0]/len(self.K.Knorm) + 0*(self.K.index_min)/len(self.K.Knorm)
                self.K.Peaks = peaks
                self.K.Downs = downs


            if method=="cwt":
                peaks=scipy.signal.find_peaks_cwt(data,np.arange(1,100))
                # print(peaks)
                vlines_peaks = 2*peaks/len(self.K.Knorm) + 0*(self.K.index_min)/len(self.K.Knorm)
                self.K.Peaks = peaks

            return vlines_peaks,vlines_downs
        vlines_peaks,vlines_downs = detect_peaks(method="fp")

        try:
            self.K.NKurPeaks = len(vlines_peaks)
            self.K.KurPeaks  = vlines_peaks
        except:
            self.K.NKurPeaks = 0
            self.K.KurPeaks  = 0
        try:
            self.K.NKurdowns = len(vlines_downs)
            self.K.Kurdowns  = vlines_downs
        except:
            self.K.NKurdowns = 0
            self.K.Kurdowns  = 0

        vlines_peaks = np.append(0.0,vlines_peaks)#why I added 0?
        # vlines_peaks = np.append(vlines_peaks,vlines_downs)
        # vlines_downs = np.append(0.0,vlines_downs)
        self.K.vlines_peaks = vlines_peaks
        self.K.vlines_downs = vlines_downs


    def entropies(self):
        """
        Estimate entropic quantities of the curvature.

        This is purelly experimental. I did not implemented the
        calculation of entropies properly yet, because this is
        not trivial to do so, since we need to considerade each
        pixel dependence of the image (image pixels are long range
        interacted).
        """
        def Bins(n,meth="knuth"):
            """
            Select the optimized bins number for a given histogram.
            """
            if meth=="scott":
                from astropy.stats import scott_bin_width
                bins=len(scott_bin_width(n,True)[1])
            if meth=="rice":
                N=len(n)
                bins=int(2.*N**(1./3.))
            if meth == "knuth":
                from astropy.stats import knuth_bin_width
                bins=len(knuth_bin_width(n,True)[1])
            return bins
        def Hentropy(data, bins=10,b=0,normed=1,meth="knuth"):
            """
            Shannon entropy (Ferrari, 2015).
            """
            if b==1:
                #create the histogram with the optimized bin width.
                bins=Bins(data,meth=meth)
                # print("OPT bins", bins)
            if b==0:
                bins=bins
            h = pl.histogram(data, bins=bins, normed=True)[0]
            pimg = h/float(h.sum())
            S =  pl.sum([ -p*pl.log(p) for p in pimg if p!=0 ])
            if normed==1:
                N = pl.log(bins)
            else:
                N = 1.0
            # print "ENtropy", S/N
            return S/N

        def new_entropy(x,y,b=0,meth="knuth"):
            """
            Entropy calculation based on (Ramos, 2016).
            """
            if b==1:
                x_bins=Bins(x.ravel(),meth="knuth")
                y_bins=Bins(y.ravel(),meth="knuth")
                # print("OPT bins", x_bins)
            if b==0:
                y_bins = 10
                x_bins = 10

            h = np.histogram2d(y,x,[y_bins,x_bins])
            bins = h[0]
            size = y_bins# * r_bins#len(bins[0])
            divided_bins = bins/size
            p = bins #divided_bins
            entropy=0
            entropy_matrix=[]
            p = p/p.sum()
            HM = np.zeros((len(p[:,0]),len(p[0,:])))
            for i in range(len(p[:,0])):
                for j in range(len(p[0,:])):
                    if p[i,j] !=0:
                        entropy_column= -p[i,j]*np.log((p[i,j]/(p[i,:].sum()) ))
                        Hsum = -(p[i,j])*np.log((p[i,j]/(p[i,:].sum()) ))
                        entropy +=Hsum
                        HM[i,j] = Hsum
                    else:
                        entropy +=0
                entropy_matrix.append(entropy_column)
            entropy_matrix = np.asarray(entropy_matrix)
            entropy_matrix[np.isnan(entropy_matrix)] = 0
            # print("COND ENTROPY",entropy)
            return entropy, entropy_matrix, HM.sum()

        def q_entropy(img, bins=10,b=0,normed=1,meth="knuth"):
            """
            Tsallis entropy.

            Return the Tsallis entropy for an array of q's.
            Return the maximum Tsallis entropy for an array of q's.
            Return que q that minimize the entropy.
            """
            Q=np.linspace(-3.001,6.001,500)
            SQ=[]
            SQM=[]
            if b==1:
                bins=Bins(img.ravel(),meth=meth)
                # print("OPT bins", bins)
            if b==0:
                bins=bins
            h = pl.histogram(img.ravel(), bins=bins, density=True)[0]
            pimg = h/float(h.sum())
            for q in Q:
                # fp.value += 1
                Sq= (1./(q-1.))*(1-pl.sum([p**q for p in pimg if p!=0 ]))
                if normed==1:
                    N = (1.-bins**(1.-q))/(q-1.)
                    SQM.append(N)
                else:
                    N = 1.0
                    SQM.append(N)
                SQ.append(Sq)
            SQ=np.asarray(SQ)
            # plt.plot(Q,SQ/np.asarray(SQM))
            # plt.show()
            SQM=np.asarray(SQM)
            ind=np.where((np.asarray(SQ)/np.asarray(SQM))==(np.asarray(SQ)/np.asarray(SQM)).min())[0][0]
            q_max=Q[ind]
            Sq_max=SQ[ind]/SQM[ind]
            # print("q  ->", q_max)
            # print("Sq ->", Sq_max)
            # if b==1:
            return q_max, Sq_max

        self.K.q_kur,self.K.Sq_kur = q_entropy(self.K.sel_kurvature)
        self.K.q_profile,self.K.Sq_profile = q_entropy(self.K.profile[self.K.index_min:self.K.index_max])
        self.K.q_nu,self.K.Sq_nu = q_entropy(self.K.nu_0[self.K.index_min:self.K.index_max])
        self.K.HC_K_new,_,self.K.HCM_K_new = new_entropy(self.K.radius_norm[self.K.index_min:self.K.index_max],self.K.sel_kurvature)
        self.K.HC_nu_new,_,self.K.HCM_nu_new = new_entropy(self.K.radius_norm[self.K.index_min:self.K.index_max],self.K.nu_0[self.K.index_min:self.K.index_max])
        self.K.Hkur = Hentropy(self.K.sel_kurvature)
        self.K.Hnu = Hentropy(self.K.nu_0[self.K.index_min:self.K.index_max])

        self.K.HC_inner_K_new,_,self.K.HCM_inner_K_new = new_entropy(self.K.r_s_inner,self.K.inner_data_s)
        self.K.H_inner_kur = Hentropy(self.K.inner_data_s)

        try:
            self.K.HC_outer_K_new,_,self.K.HCM_outer_K_new = new_entropy(self.K.r_s_outer,self.K.outer_data_s)
            self.K.H_outer_kur = Hentropy(self.K.outer_data_s)
            self.K.k_outer_q, self.K.k_outer_Sq = q_entropy(self.K.outer_data_s)

        except:
            self.K.HC_outer_K_new,self.K.HCM_outer_K_new = 0.,0.
            self.K.H_outer_kur = 0.0
            self.K.k_outer_q, self.K.k_outer_Sq = 0.0,0.0

        self.K.k_inner_q, self.K.k_inner_Sq = q_entropy(self.K.inner_data_s)

        # print(self.K.HCM_nu_new)


    def create_sub_regions(self):
        RRR = self.K.radius_norm.copy()[self.K.index_min:self.K.index_max]
        data = InterpolatedUnivariateSpline(RRR,self.K.sel_kurvature)
        R_sampled = np.linspace(RRR[0],RRR[-1],500)
        data_s = data(R_sampled)
        data_s_norm = data_s/data_s.max()

        positive_kur = np.zeros(len(data_s))
        negative_kur = np.zeros(len(data_s))
        positive_kur[data_s>0] = data_s[data_s>0]
        negative_kur[data_s<0] = data_s[data_s<0]

        Rf = len(RRR)
        scale_factor = len(data_s)#/len(self.K.sel_kurvature)

        # inner_stats = (self.K.peaks_properties["right_ips"][0]+\
            # 0.0*self.K.peaks_properties["widths"][0])/Rf
        inner_stats = 0.5
        # print(inner_stats)
        # print(self.K.radius_norm[self.K.Peaks]*2)
        self.K.inner_data_s = data_s.copy()[:int(inner_stats*data_s.size)]
        self.K.outer_data_s = data_s.copy()[int(inner_stats*data_s.size):]
        self.K.data_s_r1 = data_s.copy()[:int(0.20*scale_factor)]
        self.K.data_s_r2 = data_s.copy()[int(0.20*scale_factor):int(0.60*scale_factor)]
        self.K.data_s_r3 = data_s.copy()[int(0.60*scale_factor):]
        self.K.r_s_inner =  R_sampled[:int(inner_stats*scale_factor)]
        self.K.r_s_outer = R_sampled[int(inner_stats*scale_factor):]
        # print("Inner",self.K.r_s_inner*2)
        # print("Outer",self.K.r_s_outer*2)
        self.K.r_s_r1 = R_sampled[:int(0.20*scale_factor)]
        self.K.r_s_r2 = R_sampled[int(0.20*scale_factor):int(0.60*scale_factor)]
        self.K.r_s_r3 = R_sampled[int(0.60*scale_factor):]
        self.K.data_s = data_s
        self.K.R_sampled = R_sampled
        self.K.data_s_norm = data_s_norm
        self.K.data = data
        self.K.RRR = RRR
        self.K.inner_stats = inner_stats
        self.K.positive_kur = positive_kur
        self.K.negative_kur = negative_kur

    def kurv_area_and_stats(self):
        """
        Estimate the area under curvature, shape statistics of curvature
        and many other related quantities.

        NOTE: There are a lot of stuff bellow, and may of them may not work for
        morphology. I am testing now...
        """
        #FOR STANDARD CURVATURE
        #Separate the positive values from negative values of the curvature.
        RRR = self.K.RRR
        data_s = self.K.data_s
        data  = self.K.data
        R_sampled = self.K.R_sampled
        data_s_norm = self.K.data_s_norm
        positive_kur = self.K.positive_kur
        negative_kur = self.K.negative_kur

        #Total Area unde K(R).
        self.K.area = si.quadrature(data,R_sampled[0],R_sampled[-1])[0]

        #Positive Area under K(R).
        self.K.area_pos = si.simps(positive_kur,R_sampled,dx = 1./len(R_sampled))
        #Negative Area under K(R).
        self.K.area_neg = si.simps(negative_kur,R_sampled,dx = 1./len(R_sampled))

        self.K.k_sum_mean = np.sum(data_s)/np.mean(data_s)

        self.K.n_area =  si.simps(data_s_norm,R_sampled,dx = 1./len(R_sampled))

        self.K.k_mean = np.mean(data_s)

        self.K.n_k_mean = np.mean(data_s_norm)

        self.K.k_median = np.median(data_s)

        self.K.n_k_median = np.median(data_s_norm)

        self.K.k_std = np.std(data_s)

        self.K.n_k_std = np.std(data_s_norm)

        self.K.k_variance = stat.pvariance(data_s)

        self.K.n_k_variance = stat.pvariance(data_s_norm)

        self.K.k_mode = stat.mode(data_s)

        self.K.n_k_mode = stat.mode(data_s_norm)

        self.K.k_pstdev = stat.pstdev(data_s)

        self.K.n_k_pstdev = stat.pstdev(data_s_norm)


        self.K.Ar1 = si.simps(self.K.r_s_r1,self.K.data_s_r1)
        self.K.Ar2 = si.simps(self.K.r_s_r2,self.K.data_s_r2)
        self.K.Ar3 = si.simps(self.K.r_s_r3,self.K.data_s_r3)


        self.K.k_area_inner = si.simps(self.K.inner_data_s,self.K.r_s_inner)#/abs(self.K.Area)

        try:
            self.K.k_area_outer = si.simps(self.K.outer_data_s,self.K.r_s_outer)#/abs(self.K.Area)
        except:
            self.K.k_area_outer = 0.0

        self.K.k_mean_inner = np.mean(self.K.inner_data_s)
        self.K.k_mean_outer = np.mean(self.K.outer_data_s)

        self.K.k_inner_sum_mean = np.sum(self.K.inner_data_s)/np.mean(self.K.inner_data_s)
        self.K.k_outer_sum_mean = np.sum(self.K.outer_data_s)/np.mean(self.K.outer_data_s)


        self.K.n_inner_area =  si.simps(self.K.inner_data_s/data_s.max(),self.K.r_s_inner)
        try:
            self.K.n_outer_area =  si.simps(self.K.outer_data_s/data_s.max(),self.K.r_s_outer)
        except:
            self.K.n_outer_area =  0.0

        self.K.k_inner_mean = np.mean(self.K.inner_data_s)
        self.K.k_inner_mean = np.mean(self.K.outer_data_s)

        self.K.n_inner_k_mean = np.mean(self.K.inner_data_s/data_s.max())
        self.K.n_outer_k_mean = np.mean(self.K.outer_data_s/data_s.max())

        self.K.k_inner_median = np.median(self.K.inner_data_s)
        self.K.k_outer_median = np.median(self.K.outer_data_s)


        self.K.n_inner_k_median = np.median(self.K.inner_data_s/data_s.max())
        self.K.n_outer_k_median = np.median(self.K.outer_data_s/data_s.max())

        self.K.k_inner_std = np.std(self.K.inner_data_s)
        self.K.k_outer_std = np.std(self.K.outer_data_s)

        self.K.n_inner_k_std = np.std(self.K.inner_data_s/data_s.max())
        self.K.n_outer_k_std = np.std(self.K.outer_data_s/data_s.max())

        self.K.k_inner_variance = stat.pvariance(self.K.inner_data_s)
        try:
            self.K.k_outer_variance = stat.pvariance(self.K.outer_data_s)
        except:
            self.K.k_outer_variance = 0.0

        self.K.n_inner_k_variance = stat.pvariance(self.K.inner_data_s/data_s.max())
        try:
            self.K.n_outer_k_variance = stat.pvariance(self.K.outer_data_s/data_s.max())
        except:
            self.K.n_outer_k_variance = 0.0

        self.K.k_inner_mode = stat.mode(self.K.inner_data_s)
        try:
            self.K.k_outer_mode = stat.mode(self.K.outer_data_s)
        except:
            self.K.k_outer_mode = 0.0

        self.K.n_inner_k_mode = stat.mode(self.K.inner_data_s/data_s.max())
        try:
            self.K.n_outer_k_mode = stat.mode(self.K.outer_data_s/data_s.max())
        except:
            self.K.n_outer_k_mode = 0.0

        self.K.k_inner_pstdev = stat.pstdev(self.K.inner_data_s)
        try:
            self.K.k_outer_pstdev = stat.pstdev(self.K.outer_data_s)
        except:
            self.K.k_outer_pstdev = 0.0

        self.K.n_inner_k_pstdev = stat.pstdev(self.K.inner_data_s/data_s.max())

        try:
            self.K.n_outer_k_pstdev = stat.pstdev(self.K.outer_data_s/data_s.max())
        except:
            self.K.n_outer_k_pstdev = 0.0


    def kurv_shape_analysis(self):
        """
        Estimate statistics of the curvature's shape and related quantities.

        Some estimated quantities are experimental.
        At the moment, I am studying the best ways to do that.
        """
        import scipy.stats as Ss

        KRn = klibs.normalise(self.K.sel_kurvature.copy())
        KRn_inner = klibs.normalise(self.K.inner_data_s)
        self.K.kur_skew = Ss.skew(KRn)#second order mommentum
        self.K.kur_kurt = Ss.kurtosis(KRn)#third order mommentum
        self.K.kur_inner_kurt = Ss.kurtosis(KRn_inner)
        self.K.kur_inner_skew = Ss.skew(KRn_inner)
        try:
            KRn_outer = klibs.normalise(self.K.outer_data_s)
            self.K.kur_outer_kurt = Ss.kurtosis(KRn_outer)
            self.K.kur_outer_skew = Ss.skew(KRn_outer)
        except:
            KRn_outer = 0.0
            self.K.kur_outer_kurt = 0.0
            self.K.kur_outer_skew = 0.0




class kmorph():
    """
    Do morphometry in each part found of the galaxy profile.
    """
    def __init__(self,kurv):
        self.K = kurv
        self.k_C()

    def k_C(self):

        def find_Ri(frac,R,LR,LT):
            try:
                LRspl = scipy.interpolate.splrep(R, (LR - frac * LT), s=0.01)
                RR = scipy.interpolate.sproot(LRspl)[0]
            except:
                RR = 1.0
            return RR

        def Concentration_split_improved(R,LR,index_peaks,i):
            """
            This function will split each part of the profile, according
            the peaks found in curvature, and will calculate
            the fractional radius R20, R50, R50 and R80 for each
            component. Note that R50 will be the approximate effective
            radius of the component.
            """
            # print("***asd**asd*as*",self.K.peaks_properties["right_ips"].astype(int), index_peaks,len(LR))

            LTR_component    =   LR[index_peaks[i]:index_peaks[i+1]+1]-LR[index_peaks[i]]
            LT_component     =   LTR_component.max()
            R_component      =   R[index_peaks[i]:index_peaks[i+1]+1] - R[index_peaks[i]]
            Rmax_component   =   R_component.max()#full lenght of the component
            # print("Component Properties", Rmax_component,LT_component)
            # print("LT",LT_,LR_)
            # print("Total Luminosity of the component >>", LT_component)
            # print("Component lenght >>", Rmax_component/self.K.RP)

            R20  = find_Ri(0.2,R_component,LTR_component,LT_component)
            R50  = find_Ri(0.5,R_component,LTR_component,LT_component)
            R80  = find_Ri(0.8,R_component,LTR_component,LT_component)
            R90  = find_Ri(0.9,R_component,LTR_component,LT_component)
            # print(index_peaks[i],R[index_peaks[i]])
            R20  = R20+R[index_peaks[i]]
            R50  = R50+R[index_peaks[i]]
            R80  = R80+R[index_peaks[i]]
            R90  = R90+R[index_peaks[i]]

            C1 = np.log10(R80/R20)
            C2 = np.log10(R90/R50)
            # print("C1:",C1)
            # print("C2:",C2)
            # print("R50:",R50)
            return C1,C2,R50

        def model_sersic_area(R,n=4,L_T=None,Rn=None):
            """
            Creates a Sersic model with n=4 (classical elliptical)
            with the same luminosity of the input galaxy and with
            its same effective redii (R50 by the way).
            """

            if L_T == None:
                L_T = 1.0e6#arbitrary???
            if Rn==None:
                #see Ferrari, et al 2015.
                alpha = 0.78
                a = 1.91
                n0 = -1.04
                Rpmax = 5.81
                return ((Rp/Rpmax))*((a)/(n-n0))**np.exp(((n-n0)/a)**alpha)

            def IR_sersic(n,Rn,L_T,R):
                def bn(n):
                    return 2.*n - 1./3.+(4./405.*n) + (46./25515.*n**2.0)
                return L_T/(bn(n)*Rn**2.0)*np.exp(-bn(n)*((R/Rn)**(1./n)))

            def k_R(y,x):
                """
                Call kurvature maths.
                """
                def normalise(x):
                    return (x- x.min())/(x.max() - x.min())
                r = normalise(x)
                f = normalise(np.log(y))
                #First derivatites/differentials
                df  =  np.gradient(f,1.0)
                dr  =  np.gradient(r,1.0)
                #second derivatives/differentials
                d2f =  np.gradient(df,1.0)
                d2r =  np.gradient(dr,1.0)
                #curvature
                k   =  (dr * d2f  - df * d2r ) / np.power(dr**2.0 + df**2.0,1.5)
                return k,r

            Isersic = IR_sersic(n,Rn,L_T,R)
            KR,RR = k_R(Isersic,R)

            AreaModel = si.simps(KR,RR)
            return AreaModel


        try:
            self.K.LR = klibs.get_data(param="LR",File = self.K.filename)
            self.K.LT = self.K.LR.max()
        except:
            self.K.LR = scipy.integrate.cumtrapz(self.K.profile, x=self.K.radius, axis=-1, initial=None)
            self.K.LT = self.K.LR.max()

        R50  = find_Ri(0.5,self.K.radius,self.K.LR,self.K.LT)
        self.K.R50Gal = R50
        indexR50 = int(klibs.find_nearest(self.K.radius,self.K.R50Gal)[0]/2)
        self.K.I50Gal = self.K.profile[indexR50]
        self.K.Kurv50 = self.K.sel_kurvature[indexR50]

        print("Galaxy's R50=",self.K.R50Gal)
        print("Galaxy's IR at R50=",self.K.I50Gal)
        print("Galaxy's Kurv at R50=",self.K.Kurv50)
        # print(self.K.radius,len(self.K.radius))

        kur_area_model = model_sersic_area(self.K.radius,L_T=self.K.LT,Rn = R50)
        # print("AreaModel",kur_area_model)
        self.K.kur_area_model = kur_area_model

        # Concentration_split(self.K.radius,self.K.LR)
        # Concentration_split(self.K.radius[self.K.index_min:self.K.index_max],self.K.LR[self.K.index_min:self.K.index_max])

        #Possible concentrations
        CC1 = []
        CC2 = []
        RR50 = []
        # print("+++++++++++",self.K.peaks_properties["right_ips"])
        # index_peaks = np.append(0,self.K.peaks_properties["right_ips"].astype(int))
        index_peaks = np.append(0,(self.K.vlines_peaks[1:]*self.K.RP/2).astype(int))
        # print(int(self.K.vlines_peaks[1]*self.K.RP/2))
        # print(self.K.peaks_properties["right_ips"].astype(int))
        # print(self.K.peaks_properties)
        # print(self.K.peaks_properties["right_ips"])
        # print(self.K.Peaks/self.K.RP)

        index_peaks = np.append(index_peaks,self.K.index_max)

        # print(self.K.peaks_properties["right_ips"].astype(int),self.K.radius[self.K.peaks_properties["right_ips"].astype(int)]*2)
        # print(self.K.vlines_peaks*self.K.RP*2,self.K.radius[-1],self.K.radius[int(self.K.vlines_peaks[1]*self.K.RP)])
        plt.figure()
        for i in range(0,len(index_peaks)-1):
            C1i,C2i,R50i = Concentration_split_improved(self.K.radius,self.K.LR,index_peaks,i)

            plt.plot(self.K.radius[index_peaks[i]:index_peaks[i+1]+1]/self.K.RP,self.K.LR[index_peaks[i]:index_peaks[i+1]+1]/self.K.LR[-1],label="Component $"+str(i+1)+"$")
            plt.axvline(R50i/self.K.RP,color="darkmagenta")
            if i==len(index_peaks)-2:
                plt.axvline(R50i/self.K.RP,label=r"$R_{eff}$ -- Half light radii",color="darkmagenta")

            CC1.append(C1i)
            CC2.append(C2i)
            RR50.append(R50i)

        plt.xlabel(r"$R/R_p$")
        plt.ylabel(r"$L_T/L_{\rm total}$")
        plt.grid(which='major', axis='both', linestyle='--',color="gray",alpha=0.5)
        plt.legend(loc="best")
        plt.savefig(self.K.rootname+"_components.svg",dpi=300,bbox_inches='tight')
        plt.show()
        CC1= np.asarray(CC1)
        CC2= np.asarray(CC2)
        RR50= np.asarray(RR50)
        self.K.CC1 = CC1
        self.K.CC2 = CC2
        self.K.RR50 = RR50
        print("Effective Radius (1st_component,2nd_component)=(",RR50[0],',',RR50[1],')')
        # print("Slit concentrations", CC1,CC2,RR50)

class kSersic():
    """


    """
    def __init__(self,kurv):
        self.K = kurv
        self.model2D()
        self.run_imfit()

    def model2D(self):
        print(">Init of FIT2D")
        def rotation(PA,x0,y0,x, y):
            """Rotate an input image array. It can be used to modified
            the position angle (PA).
            """
            # gal_center = (x0+0.01,y0+0.01)
            x0 = x0
            y0 = y0
            #convert to radians
            t=PA*np.pi/18
            # t=PA
            return ((x-x0)*np.cos(t) + (y-y0)*np.sin(t),-(x-x0)*np.sin(t) + \
                    (y-y0)*np.cos(t))

        def S2D(pars,gal_size):
            "Single Sérsic Model > 2D"
            from numpy import meshgrid
            In,Rn,n,q,c,PA,x0,y0 = pars
            def bn(n):
                return 2.*n - 1./3.+((4./405.)*n) + ((46./25515.)*n**2.0)

            # if (gal_size[1]/2)%2 == 1:
            #     x,y=np.meshgrid(np.arange(-int(gal_size[1]/2),\
            #         int((gal_size[1])/2),1), np.arange(-int(gal_size[0]/2),\
            #         int(gal_size[0]/2)+1,1))
            # if (gal_size[0]/2)%2 == 1:
            #     x,y=np.meshgrid(np.arange(-int(gal_size[1]/2),\
            #         int((gal_size[1])/2)+1,1), np.arange(-int(gal_size[0]/2),\
            #         int(gal_size[0]/2),1))
            # else:
            #     x,y=np.meshgrid(np.arange(-int(gal_size[1]/2),\
            #         int((gal_size[1])/2)+1,1), np.arange(-int(gal_size[0]/2),\
            #         int(gal_size[0]/2)+1,1))
            x,y=np.meshgrid(np.arange((gal_size[1])),np.arange((gal_size[0])))

            x,y=rotation(PA,y0,x0,y,x)
            r=(abs(x)**(c+2.0)+((abs(y))/(q))**(c+2.0))**(1.0/(c+2.0))

            sersic2D=In*np.exp(	-bn(n)*(	(r/(Rn)	)**(1.0/n) -1.	))
            psf = "examples/psf_efigi_s13.fits"
            sersic2D = signal.fftconvolve(sersic2D,psf,'same')
            return(sersic2D)

        def MS2D(pars,gal_size):
            "Multiple Sérsic Model (B+D) > 2D"
            from numpy import meshgrid
            InB,RnB,nB,qB,cB,PAB,x0B,y0B,InD,RnD,nD,qD,cD,PAD,x0D,y0D = pars
            def bn(n):
                return 2.*n - 1./3.+0*((4./405.)*n) + ((46./25515.)*n**2.0)

            def create_sersic_model(PA,x0,y0,In,Rn,n,q,c):
                x,y=np.meshgrid(np.arange((gal_size[1])),np.arange((gal_size[0])))
                x,y=rotation(PA,x0,y0,x,y)
                r=(abs(x)**(c+2.0)+((abs(y))/(q))**(c+2.0))**(1.0/(c+2.0))
                model=In*np.exp(	-bn(n)*(	(r/(Rn)	)**(1.0/n) -1.	))
                return(model)

            Bulge = create_sersic_model(PAB,y0B,x0B,InB,RnB,nB,qB,cB)
            Disk = create_sersic_model(PAD,y0D,x0D,InD,RnD,nD,qD,cD)
            model2D = Bulge+Disk
            psfn = "examples/psf_efigi_s13.fits"
            psf = pf.getdata(psfn)
            model2D = signal.fftconvolve(model2D,psf,'same')
            return(model2D)

        def cal_PA_q():
            from fitEllipse2018 import main_test2
            #mean Inner q,  mean outer q , mean Inner PA, mean Outer PA
            self.K.qmi   ,  self.K.qmo   , self.K.PAmi  , self.K.PAmo = \
                main_test2(self.K.gal_image_0)

            #global PA , global q
            self.K.PA  , self.K.q = klibs.q_PA(self.K.gal_image_0)

            print("Initial PA and q = ", self.K.PA, self.K.q)
            print("Inner-Mean PA and q = ", self.K.PAmi, self.K.qmi)
            print("Outer-Mean PA and q = ", self.K.PAmo, self.K.qmo)


        def set_constraints():

            self.K.RR50_1 = self.K.RR50[0]
            self.K.RR50_2 = self.K.RR50[1]
            index_IR50_1 = int(klibs.find_nearest(self.K.radius,self.K.RR50_1)[0]/2)
            index_IR50_2 = int(klibs.find_nearest(self.K.radius,self.K.RR50_2)[0]/2)
            self.K.IR50_1 = self.K.profile[index_IR50_1]
            self.K.IR50_2 = self.K.profile[index_IR50_2]

            qmi,qmo,PAmi,PAmo = self.K.qmi,self.K.qmo,self.K.PAmi,self.K.PAmo
            PA,q = self.K.PA,self.K.q

            x0 = self.K.gal_image_0.shape[0]/2.0
            y0 = self.K.gal_image_0.shape[1]/2.0

            # print("Effective Radius", self.K.RR50_1,self.K.RR50_2)
            # print("Effective Intensity", self.K.IR50_1,self.K.IR50_2)
            print("Effective Intensity (1st_component,2nd_component)=(",self.K.IR50_1,',',self.K.IR50_2,')')

            dQ = 0.1

            InBi,RnBi,nBi,qBi,cBi,PABi,x0Bi,y0Bi = self.K.IR50_1*0.5,self.K.RR50_1*0.7,1.3,    qmi-dQ,-0.1,PAmi*0.5,x0-5,y0-5
            InDi,RnDi,nDi,qDi,cDi,PADi,x0Di,y0Di = self.K.IR50_2*0.3,self.K.RR50_2*0.7,0.8,    qmo-dQ,-0.1,PAmo*0.5,x0-5,y0-5
            if (qmi+dQ)<=1:
                InBf,RnBf,nBf,qBf,cBf,PABf,x0Bf,y0Bf = self.K.IR50_1*1.7,self.K.RR50_1*1.5,6.0,qmi+dQ*0.5, 0.1,     180,x0+5,y0+5
            else:
                InBf,RnBf,nBf,qBf,cBf,PABf,x0Bf,y0Bf = self.K.IR50_1*1.5,self.K.RR50_1*1.5,6.0,0.91  , 0.1,     180,x0+5,y0+5
            if (qmo+dQ)<=1:
                InDf,RnDf,nDf,qDf,cDf,PADf,x0Df,y0Df = self.K.IR50_2*1.7,self.K.RR50_2*2.0,1.2,qmo+dQ*0.5, 0.1,     180,x0+5,y0+5
            else:
                InDf,RnDf,nDf,qDf,cDf,PADf,x0Df,y0Df = self.K.IR50_2*1.5,self.K.RR50_2*2.0,1.2,0.91,   0.1,      180,x0+5,y0+5


            bounds_i = [InBi,RnBi,nBi,qBi,cBi,PABi,x0Bi,y0Bi,InDi,RnDi,nDi,qDi,cDi,PADi,x0Di,y0Di]
            bounds_f = [InBf,RnBf,nBf,qBf,cBf,PABf,x0Bf,y0Bf,InDf,RnDf,nDf,qDf,cDf,PADf,x0Df,y0Df]

            pars0 = InBi,RnBi*1.5,nBi*2.0,qBi,cBi,PAmi,x0Bi,y0Bi,InDi,\
                RnDf/2,1.0,qDi,cDi,PAmo,x0Di,y0Di

            # print(pars0)
            return pars0,bounds_i,bounds_f

        def residual2D(pars):
            # func = np.ravel(S2D(pars,self.K.gal_image_0.shape)-self.K.gal_image_0)
            func = np.ravel(MS2D(pars,self.K.gal_image_0.shape)-self.K.gal_image_0)
            # print(func)
            return(func)

        def do_fit_improved():
            pars0 = set_constraints()[0]
            InBi,RnBi,nBi,qBi,cBi,PABi,x0Bi,y0Bi,InDi,RnDi,nDi,qDi,cDi,PADi,x0Di,y0Di = set_constraints()[1]
            InBf,RnBf,nBf,qBf,cBf,PABf,x0Bf,y0Bf,InDf,RnDf,nDf,qDf,cDf,PADf,x0Df,y0Df = set_constraints()[2]
            peso = 1000.

            vinculos = [[InBi,InBf], [RnBi,RnBf], [nBi,nBf], [qBi,qBf], \
                        [cBi,cBf],[PABi,PABf], [x0Bi,x0Bf],[y0Bi,y0Bf], \
                        [InDi,InDf],[RnDi,RnDf],[nDi,nDf],[qDi,qDf],\
                        [cDi,cDf],[PADi,PADf],[x0Di,x0Df],[y0Di,y0Df]]
            fitResult = leastsq_bounds(residual2D,pars0, vinculos, boundsweight=peso,  full_output=1)
            parFit2 = fitResult[0]
            return parFit2

        def leastsq_bounds(func, x0, bounds, boundsweight=1000, **kwargs ):
            """
            From MFMTK.
            """
            from scipy.optimize import leastsq as LSQ
            # Example: test_leastsq_bounds.py

            def _inbox(X, box, weight=1 ):
                """ -> [tub( Xj, loj, hij ) ... ]
                    all 0  <=>  X in box, lo <= X <= hi
                """
                assert len(X) == len(box), \
                    "len X %d != len box %d" % (len(X), len(box))
                return weight * np.array([
                    np.fmax( lo - x, 0 ) + np.fmax( 0, x - hi )
                        for x, (lo,hi) in zip( X, box )])
            if bounds is not None  and  boundsweight > 0:
                funcbox = lambda p: \
                    np.hstack(( func(p),_inbox( p, bounds, boundsweight )))
            else:
                funcbox = func
            fitResult = LSQ( funcbox, x0,maxfev=1000, **kwargs )
            return fitResult


        def do_fit2D():
            methodFit="non_linear_LS_bound"#"LSQ"

            if methodFit=="non_linear_LS_bound":
                print("    # Calling Least Squares Bound Method")
                from scipy.optimize import least_squares as LS
                #            #Ini Rni  ni  qi   ci   PAi  x0i y0i  Inf Rnf  nf  qf  cf  PAf x0f y0f
                # bounds = ([0.05,20.0,2.0,0.6,-0.1,-180,-10,-10],[5.0,40.0,4.0,1.0,0.1,180,10, 10])
                pars0,bounds_i,bounds_f = set_constraints()

                bounds= (bounds_i,bounds_f)

                # bounds = ([Inf,Rni,ni,0.75,-0.01,-180,-5,-5],[Ini,Rnf,nf,1.0,0.01,180,5, 5])
                fitResult = LS(residual2D,pars0,bounds=bounds,method='trf',tr_solver="lsmr",f_scale=0.5,loss="soft_l1",max_nfev=1000,verbose=2)#tr_solver="exact",
                # fitResult = LS(residual2D,pars0,method='trf',tr_solver="lsmr",verbose=2)
                parFit2  = fitResult.x
                # print(fitResult.x[3])

            if methodFit=="LSQ":
                from scipy.optimize import leastsq as LSQ
                pars0,_,_ = set_constraints()
                fitResult,_ = LSQ(residual2D,pars0)
                parFit2 = fitResult

            if methodFit=="differential_evolution":
                from scipy.optimize import differential_evolution as DE
                "not working"
                # from yabox import DE
                # from scipy.optimize import basinhopping as BH
                # from scipy.optimize import Bounds
                # bounds = Bounds([0.05,5.0],[20.0,40.0],[2.0,4.0],[0.6,1.0],[-0.1,0.1],[-180,180],[-10,10],[-10,10])
                # de = DE(residual2D,bounds)
                # parFit2=de.solve(show_progress=True)
                # parFit2 = DE(func=residual2D,bounds=bounds)
                # pars0 = 0.1,30.0,3.0,0.9,0.0,45,0,0
                # bounds = [(0.05,5.0),(20.0,40.0),(2.0,4.0),(0.6,1.0),(-0.1,0.1),(-180,180),(-10,10),(-10,10)]
            if methodFit=="brute_force":
                "Not Working -- to much memory required."
                from scipy.optimize import brute as BB
                bounds = slice(0.1,5.0,0.5),slice(20.0,40.0,2),slice(2.0,4.0,0.5),slice(0.6,1.0,0.1),slice(-0.1,0.1,0.1),slice(-180,180,20),slice(-10,10,4),slice(-10,10,4)
                # bounds = [0.05,20.0,2.0,0.6,-0.1,-180,-10,-10],[5.0,40.0,4.0,1.0,0.1,180,10,10]
                fitResult=BB(residual2D,ranges=bounds,args=pars0)
                parFit2 = fitResult
            if methodFit=="shgo":
                "not working"
                from scipy.optimize import shgo as SHGO
                bounds = [(0.05,5.0),(20.0,40.0),(2.0,4.0),(0.6,1.0),(-0.1,0.1),(-180,180),(-10,10),(-10,10)]
                fitResult=SHGO(residual2D,bounds=bounds)
                parFit2 = fitResult
            if methodFit=="dualA":
                "not working"
                from scipy.optimize import dual_annealing as DA
                bounds = [(0.05,5.0),(20.0,40.0),(2.0,4.0),(0.6,1.0),(-0.1,0.1),(-180,180),(-10,10),(-10,10)]
                fitResult=DA(residual2D,bounds=bounds)
                parFit2 = fitResult


            # print(parFit2)
            return(parFit2)


        # parsBD = 0.1,30.0,3.0,0.9,0.0,45,0.5,0.5,0.1,30.0,1.0,0.9,0.0,45,0.5,0.5
        # model = MS2D(parsBD,(256,256))
        # plt.imshow(np.log(model))
        cal_PA_q()
        pars=do_fit2D()
        # pars=do_fit_improved()
        gal_size = self.K.gal_image_0.shape
        # fitr = S2D(pars,gal_size)
        fitr = MS2D(pars,gal_size)


        fig = plt.figure(constrained_layout=False)
        gs = gridspec.GridSpec(1,4,hspace=0.01,wspace=0.01,figure=fig)
        fig.set_size_inches(12.0, 4.0)
        axI = fig.add_subplot(gs[0,0:1])
        klibs.imshow(self.K.gal_image_0,sigma=3.0,bar=None)
        plt.title(r"Original",fontsize=10)
        plt.axis("off")

        axM = fig.add_subplot(gs[0,1:2])
        klibs.imshow(fitr,sigma=3.0,bar=None)
        plt.title(r"Model",fontsize=10)
        plt.axis("off")

        axR = fig.add_subplot(gs[0,2:3])
        noise = 0.3*np.random.rand(fitr.shape[0],fitr.shape[1])
        klibs.imshow(fitr-self.K.gal_image_0+noise,sigma=1.5,bar=None)
        plt.title(r"Residual Frac = "+str(format(np.sum(self.K.gal_image_0-\
            fitr)/np.sum(self.K.gal_image_0),'.2f')),fontsize=10)
        plt.axis("off")

        axP = fig.add_subplot(gs[0,3:4])
        plt.axis('off')
        # In,Rn,n,q,c,PA,x0,y0 = pars

        pf.writeto(self.K.image_name.replace(".fits","")+"_model_BD_constraint.fits",fitr,overwrite=True)
        pf.writeto(self.K.image_name.replace(".fits","")+"_residual_BD_constraint.fits",self.K.gal_image_0-fitr,overwrite=True)

        InB,RnB,nB,qB,cB,PAB,x0B,y0B,InD,RnD,nD,qD,cD,PAD,x0D,y0D = pars

        # print(fitr)

        InBi,RnBi,nBi,qBi,cBi,PABi,x0Bi,y0Bi,InDi,RnDi,nDi,qDi,cDi,PADi,x0Di,y0Di = set_constraints()[1]
        InBf,RnBf,nBf,qBf,cBf,PABf,x0Bf,y0Bf,InDf,RnDf,nDf,qDf,cDf,PADf,x0Df,y0Df = set_constraints()[2]


        plt.annotate(r"Initial Constraints Conditions",(  0.75,0.97),xycoords='figure fraction',fontsize=10)
        plt.annotate(r"$nB \in$["+str(format(nBi ,  '.2f'))+","+str(format(nBf ,'.2f'))+"]",(    0.75,0.96-0.05),xycoords='figure fraction',fontsize=10)
        plt.annotate(r"$R_nB \in$["+str(format(RnBi ,'.2f'))+","+str(format(RnBf ,'.2f'))+"]"+"  $\leftarrow\kappa$",(  0.75,0.93-0.05),xycoords='figure fraction',fontsize=10)
        plt.annotate(r"$I_nB \in$["+str(format(InBi ,'.2f'))+","+str(format(InBf ,'.2f'))+"]"+"  $\leftarrow\kappa$",(  0.75,0.90-0.05),xycoords='figure fraction',fontsize=10)
        plt.annotate(r"$nD \in$["+str(format(nDi    ,'.2f'))+","+str(format(nDf ,'.2f'))+"]",(   0.75,0.87-0.05),xycoords='figure fraction',fontsize=10)
        plt.annotate(r"$R_nD \in$["+str(format(RnDi ,'.2f'))+","+str(format(RnDf ,'.2f'))+"]",(  0.75,0.84-0.05),xycoords='figure fraction',fontsize=10)
        plt.annotate(r"$I_nD \in$["+str(format(InDi ,'.2f'))+","+str(format(InDf ,'.2f'))+"]",(  0.75,0.81-0.05),xycoords='figure fraction',fontsize=10)
        # plt.annotate(r"$qB \in$["+str(format(qBi ,'.2f'))+","+str(format(qBf ,'.2f'))+"]"   +" $\leftarrow$ ellipse",(  0.75,0.78-0.05),xycoords='figure fraction',fontsize=10)
        # plt.annotate(r"$qD \in$["+str(format(qDi ,'.2f'))+","+str(format(qDf ,'.2f'))+"]"   +" $\leftarrow$ ellipse",(  0.75,0.75-0.05),xycoords='figure fraction',fontsize=10)
        plt.annotate(r"$qB \sim $"+str(format(self.K.qmi ,'.2f'))+" $\leftarrow$ ellipse",(  0.75,0.78-0.05),xycoords='figure fraction',fontsize=10)
        plt.annotate(r"$qD \sim $"+str(format(self.K.qmo ,'.2f'))+" $\leftarrow$ ellipse",(  0.75,0.75-0.05),xycoords='figure fraction',fontsize=10)
        plt.annotate(r"$PAB \sim $"+str(format(self.K.PAmi ,'.2f'))+" $\leftarrow$ ellipse",(  0.75,0.72-0.05),xycoords='figure fraction',fontsize=10)
        plt.annotate(r"$PAD \sim $"+str(format(self.K.PAmo ,'.2f'))+" $\leftarrow$ ellipse",(  0.75,0.69-0.05),xycoords='figure fraction',fontsize=10)
        #
        #
        plt.annotate(r"Fit Results",( 0.75,0.66-0.1),xycoords='figure fraction',fontsize=10)
        plt.annotate(r"$R_nB=$"+str(format(RnB,'2f')),(    0.75,0.63-0.1),xycoords='figure fraction',fontsize=10)
        plt.annotate(r"$nB = $"+str(format( nB,'.2f')),(   0.75,0.60-0.1),xycoords='figure fraction',fontsize=10)
        plt.annotate(r"$I_nB=$"+str(format(InB,'2f')),(    0.75,0.57-0.1),xycoords='figure fraction',fontsize=10)
        plt.annotate(r"$qB = $"+str(format( qB,'.2f')),(   0.75,0.54-0.1),xycoords='figure fraction',fontsize=10)
        plt.annotate(r"$PAB= $"+str(format(PAB,'.2f')),(   0.75,0.51-0.1),xycoords='figure fraction',fontsize=10)
        plt.annotate(r"$cB = $"+str(format( cB,'.2f')),(   0.75,0.48-0.1),xycoords='figure fraction',fontsize=10)
        plt.annotate(r"$x_0B = $"+str(format( x0B,'.2f')),(0.75,0.45-0.1),xycoords='figure fraction',fontsize=10)
        plt.annotate(r"$y_0B = $"+str(format( y0B,'.2f')),(0.75,0.42-0.1),xycoords='figure fraction',fontsize=10)


        plt.annotate(r"$R_nD=$"+str(format(RnD,'.2f')),(   0.85,0.63-0.1),xycoords='figure fraction',fontsize=10)
        plt.annotate(r"$nD = $"+str(format( nD,'.2f')),(   0.85,0.60-0.1),xycoords='figure fraction',fontsize=10)
        plt.annotate(r"$I_nD=$"+str(format(InD,'.2f')),(   0.85,0.57-0.1),xycoords='figure fraction',fontsize=10)
        plt.annotate(r"$qD = $"+str(format( qD,'.2f')),(   0.85,0.54-0.1),xycoords='figure fraction',fontsize=10)
        plt.annotate(r"$PAD= $"+str(format(PAD,'.2f')),(   0.85,0.51-0.1),xycoords='figure fraction',fontsize=10)
        plt.annotate(r"$cD = $"+str(format( cD,'.2f')),(   0.85,0.48-0.1),xycoords='figure fraction',fontsize=10)
        plt.annotate(r"$x_0D = $"+str(format( x0D,'.2f')),(0.85,0.45-0.1),xycoords='figure fraction',fontsize=10)
        plt.annotate(r"$y_0D = $"+str(format( y0D,'.2f')),(0.85,0.42-0.1),xycoords='figure fraction',fontsize=10)



        plt.savefig(self.K.rootname+"_fit.svg",dpi=600,bbox_inches='tight')
        plt.show()

        plt.clf()
        plt.close()

        # pars0 = 0.1,30.0,3.0,0.9,0.0,45,128,128
        # residual2D(self,pars0)



    def run_imfit(self):
        """

        IMFIT Integration.
        Call imfit code with constrains from curvature and image mommenta.
        OBS: Create a symbolic link of your imfit code at /usr/bin/ or give
        the path to it bellow.

        """
        path_imfit = ""

        x0,y0 = self.K.gal_image_0.shape[0]/2.,self.K.gal_image_0.shape[1]/2.

        # PA,q = klibs.q_PA(self.K.gal_image_0)


        # InBi,RnBi,nBi,qBi,cBi,PABi,x0Bi,y0Bi,InDi,RnDi,nDi,qDi,cDi,PADi,x0Di,y0Di = set_constraints()[1]
        # InBf,RnBf,nBf,qBf,cBf,PABf,x0Bf,y0Bf,InDf,RnDf,nDf,qDf,cDf,PADf,x0Df,y0Df = set_constraints()[2]
        qmi,qmo,PAmi,PAmo = self.K.qmi,self.K.qmo,self.K.PAmi,self.K.PAmo

        dQ = 0.1

        InBi,RnBi,nBi,qBi,cBi,PABi,x0Bi,y0Bi = self.K.IR50_1*0.5,self.K.RR50_1*0.7,1.3,    qmi-dQ,-0.1,PAmi*0.5,x0-5,y0-5
        InDi,RnDi,nDi,qDi,cDi,PADi,x0Di,y0Di = self.K.IR50_2*0.3,self.K.RR50_2*0.7,0.8,    qmo-dQ,-0.1,PAmo*0.5,x0-5,y0-5
        if (qmi+dQ)<=1:
            InBf,RnBf,nBf,qBf,cBf,PABf,x0Bf,y0Bf = self.K.IR50_1*1.7,self.K.RR50_1*1.5,6.0,qmi+dQ*0.5, 0.1,     180,x0+5,y0+5
        else:
            InBf,RnBf,nBf,qBf,cBf,PABf,x0Bf,y0Bf = self.K.IR50_1*1.5,self.K.RR50_1*1.5,6.0,0.91  , 0.1,     180,x0+5,y0+5
        if (qmo+dQ)<=1:
            InDf,RnDf,nDf,qDf,cDf,PADf,x0Df,y0Df = self.K.IR50_2*1.7,self.K.RR50_2*2.0,1.2,qmo+dQ*0.5, 0.1,     180,x0+5,y0+5
        else:
            InDf,RnDf,nDf,qDf,cDf,PADf,x0Df,y0Df = self.K.IR50_2*1.5,self.K.RR50_2*2.0,1.2,0.91,   0.1,      180,x0+5,y0+5




        # self.K.image_name

        def do_double_fit():
            f = open(self.K.image_name.replace(".fits","")+"_imfit.conf","w")
            f.write("X0"  +  "   "+str(x0)+"   "+str(x0Bi)+    ","  +str(x0Bf)+'\n')
            f.write("Y0"  +  "   "+str(y0)+"   "+str(y0Bi)+    ","  +str(y0Bf)+'\n')
            f.write("FUNCTION Sersic"+'\n')
            f.write("PA"  +  "   "+str(PABi)+"   "+str(PABi)+    ","  +str(PABf)+'\n')
            f.write("ell" +  "   "+str(1-qBi)+"  "+str(1-qBf)+   ","  +str(1-qBi)+'\n')
            f.write("n"   +  "    "+str(nBi*1.2)+"   "+str(nBi)+     ","  +str(nBf)+'\n')
            f.write("I_e" +  "    "+str(InBi*1.2)+"  "+str(InBi)+     ","  +str(InBf)+'\n')
            f.write("R_e"  +  "   "+str(RnBi*1.2)+"  "+str(RnBi)+     ","  +str(RnBf)+'\n')

            # f.close()

            f.write("X0"  +  "   "+str(x0)+"   "+str(x0Di)+    ","  +str(x0Df)+'\n')
            f.write("Y0"  +  "   "+str(y0)+"   "+str(y0Di)+    ","  +str(y0Df)+'\n')
            f.write("FUNCTION Sersic"+'\n')
            f.write("PA"  +  "   "+str(PADi)+"   "+str(PADi)+    ","  +str(PADf)+'\n')
            f.write("ell" +  "   "+str(1-qDi)+"  "+str(1-qDf)+   ","  +str(1-qDi)+'\n')
            f.write("n"   +  "    "+str(nDi*1.2)+"   "+str(nDi)+     ","  +str(nDf)+'\n')
            f.write("I_e" +  "    "+str(InDi*1.2)+"  "+str(InDi)+     ","  +str(InDf)+'\n')
            f.write("R_e"  +  "   "+str(RnDi*1.2)+"  "+str(RnDi)+     ","  +str(RnDf)+'\n')
            f.close()
            imfit_meth = "--nm"#the method to call in imfit to find the solution.
            psf = "examples/psf_efigi_s13.fits"
            os.system(path_imfit+"imfit -c "+self.K.image_name.replace(".fits","")+"_imfit.conf "+ self.K.image_name+" --psf "+psf+\
                " --gain 4.5 --readnoise 0.7 --max-threads 6 --save-params "+\
                self.K.image_name.replace(".fits","")+"_params_imfit"+imfit_meth+".csv --save-model "+\
                self.K.image_name.replace(".fits","")+"_model_imfit"+imfit_meth+".fits --save-residual "+\
                self.K.image_name.replace(".fits","")+"_residual_imfit"+imfit_meth+".fits --max-threads 6 "+imfit_meth+" --loud")
        print("    # Calling IMFIT Code")
        do_double_fit()
    # def Fit_leastsq():
    #     """
    #     Run leastsq for kurvature constrained Sérsic Model Fit.
    #     """
    #
    # def Fit_Differential_Evolution():
    #     """
    #     Run scipy.optimize.differential_evolution for kurvature constrained
    #     Sérsic Fit.
    #     """









class make_plots():
    def __init__(self,kurv):
        self.K = kurv
        self.plot_cur_profile()
        self.plot_interpolations()
        # print(args.image)
        if args.image is not None:
            self.plot_2D()


    def plot_cur_profile(self):

        def make_format(current, other):
            """
            Format data/coordinates of two shared axis.
            https://stackoverflow.com/questions/21583965/matplotlib-cursor-value-with-two-axes
            Answered by 'unutbu'.
            """
            # current and other are axes
            def format_coord(x, y):
                # x, y are data coordinates
                # convert to display coords
                display_coord = current.transData.transform((x,y))
                inv = other.transData.inverted()
                # convert back to data coords with respect to ax
                ax_coord = inv.transform(display_coord)
                coords = [ax_coord, (x, y)]
                return ('Left: {:<40}    Right: {:<}'
                        .format(*['({:.3f}, {:.3f})'.format(x, y) for x,y in coords]))
            return format_coord

        fig = plt.figure(constrained_layout=True)
        gs = gridspec.GridSpec(1,1,hspace=0.7,wspace=0.4,figure=fig)
        fig.set_size_inches(7.0, 4.0)

        axK = fig.add_subplot(gs[0:2,0])


        vcounter=0
        # print(self.K.peaks_properties)

        for V in self.K.vlines_peaks:
            axK.axvline(V,color="white",ls=":",lw=2.0)
            try:
                # axK.axvline(self.K.peaks_properties["left_bases"][vcounter]/(0.5*self.K.RP),color="blue",ls="-",lw=1.5)
                # axK.axvline(self.K.peaks_properties["right_bases"][vcounter]/(0.5*self.K.RP),color="blue",ls="-",lw=1.5)
                axK.axvline(self.K.peaks_properties["left_ips"][vcounter]/(0.5*self.K.RP),color="pink",ls="-",lw=1.0,alpha=0.75)
                axK.axvline(self.K.peaks_properties["right_ips"][vcounter]/(0.5*self.K.RP),color="pink",ls="-",lw=1.0,alpha=0.75)
            except:
                pass
            # axK.text(V*2+0.04,self.K.Kurvature_filter[self.K.index_min:self.K.index_max].min()*0.4-0.95,"$"+str(vcounter+1)+"$",color="black",fontsize=20)
            vcounter = vcounter+1

        # plt.plot(self.K.kurvature_filter)
        from scipy.signal import chirp, peak_widths
        ampl = abs(self.K.kurvature_filter[self.K.index_min:self.K.index_max].max()-self.K.kurvature_filter[self.K.index_min:self.K.index_max].min())
        axK.plot(self.K.vlines_peaks[1:],self.K.kurvature_filter[self.K.index_min:self.K.index_max][self.K.Peaks],"x",linewidth=10, markersize=20)
        contour_heights = self.K.kurvature_filter[self.K.index_min:self.K.index_max][self.K.Peaks] - self.K.peaks_properties["prominences"]
        axK.vlines(x=self.K.vlines_peaks[1:], ymin = contour_heights,ymax = self.K.kurvature_filter[self.K.index_min:self.K.index_max][self.K.Peaks], color="C3")
        # axK.hlines(y=self.K.peaks_properties["prominences"],xmin = self.K.peaks_properties["left_ips"]/self.K.RP, xmax = self.K.peaks_properties["right_ips"]//self.K.RP)

        axK.hlines(self.K.peaks_properties["width_heights"],\
            xmin = self.K.peaks_properties["left_ips"]/(0.5*self.K.RP),\
            xmax = self.K.peaks_properties["right_ips"]/(0.5*self.K.RP), \
            color="purple")





        a = axK.plot((self.K.radius_norm*2)[self.K.index_min:self.K.index_max],\
            (self.K.kurvature_filter[self.K.index_min:self.K.index_max]),\
            '.-g',ms=10,lw=3.5,label = r"$\widetilde{\kappa}(R)$",alpha = 0.6)
        lns = a

        if "--mpr" in sys.argv:
        #     for i in range(len(self.K.radius_norm_list)):
        #         axK.plot((self.K.radius_norm_list[i]*2)[self.K.index_min:self.K.index_max],\
        #             (self.K.kurv_list[i][self.K.index_min:self.K.index_max]),\
        #             ms=5,lw=1.5,label = r"$\widetilde{\kappa}(R)$",alpha = 0.3)
            plt.annotate(r"$\sum (<\kappa>)=$"+str(format(self.K.K_sum_mean,'.2f')),(0.5,0.93),xycoords='figure fraction',fontsize=10)
            plt.annotate(r"$\sum (<IR>)=$"+str(format(self.K.IR_sum_mean,'.2f')),(0.5,0.86),xycoords='figure fraction',fontsize=10)
            plt.annotate(r"$\sum (<\nu>)=$"+str(format(self.K.nu_sum_mean,'.2f')),(0.5,0.80),xycoords='figure fraction',fontsize=10)

            _a = axK.plot((self.K.mean_radius_norm*2)[self.K.index_min:self.K.index_max],\
                (self.K.mean_kurv[self.K.index_min:self.K.index_max]),'--', marker='o',\
                color='magenta',ms=5,lw=2.5,label = r"$\langle \widetilde{\kappa}(R)\rangle$",antialiased=True)

            lns = lns + _a

        axK.set_ylabel("$\widetilde{\kappa}(R)$",fontsize = 15)
        axK.set_xlabel("$R/R_p$   , [$R_p="+str(self.K.RP)+"$px]",fontsize = 15)

        axK.set_ylim(self.K.kurvature_filter[self.K.index_min:self.K.index_max].min()*1.50-0.9,\
                self.K.kurvature_filter[self.K.index_min:self.K.index_max].max()*1.5)
        # axK.set_ylim(-13,13)
        axK.axhline(0, color='white',lw=0.5)
        axK.set_xlim(0.0,2.0)

        # Rf = int(len(self.K.what_Raios)*0.5)
        Rf = self.K.index_max
        Ri = self.K.index_min


        RRR = self.K.radius_norm.copy()*2

        try:
            turn_dow = np.where(self.K.kurvature_filter[self.K.index_min:]<0)[0][0]
            turn_dow = int(turn_dow + self.K.index_min)
        # print(turn_dow)
            xr1 = RRR[self.K.index_min:int(turn_dow)]
            xr2 = RRR[int(turn_dow):int(Rf)]

            x1 = self.K.kurvature_filter.copy()[Ri:int(turn_dow)]
            x2 = self.K.kurvature_filter.copy()[int(turn_dow):int(Rf)]

            # xr1 = RRR[self.K.index_min:int(0.5*Rf+1)]
            # xr2 = RRR[int(0.5*Rf):int(Rf)]
            # x1 = self.K.kurvature_filter.copy()[Ri:int(0.5*Rf+1)]
            # x2 = self.K.kurvature_filter.copy()[int(0.5*Rf):int(Rf)]
            #
            # axK.fill_between(xr1,x1,0,color='rebeccapurple')
            # axK.fill_between(xr2,x2,0,color='orange')

        except:
            xr1 = RRR[Ri:Rf]
            x1 = self.K.kurvature_filter.copy()[Ri:Rf]
            # axK.fill_between(xr1,x1,0,color='rebeccapurple')




        # axK.fill_between((self.K.radius_norm*2)[self.K.index_min:self.K.index_max]\

        #     ,self.K.kurvature_filter[self.K.index_min:self.K.index_max],0)

        axIr = axK.twinx()
        axIr.format_coord = make_format(axIr, axK)
        # axr = axK.twiny()
        #Plot IR or MUR.
        axK.grid(which='major', axis='both', linestyle='--',color="gray",alpha=0.2)
        b = axIr.plot(self.K.radius_norm[:]*2,self.K.nu_0[:],'.-r', \
            lw=3.5,ms=10, label = r"$\nu(R)$",alpha = 1.0)
        # print(self.K.radius)
        # b_ =axIr.plot(self.K.Raios_new[self.K.Rmin:]/(self.K.RP),self.K.nu[self.K.Rmin:],'.-', color='cyan', lw=1.0,ms=2, label=r"Perona-Malik [$\nu(R)$]")
        lns = lns + b #+ b_

        if "--mpr" in sys.argv:
            # for i in range(len(self.K.nu_0_list)):
            #     axIr.plot((self.K.radius_norm_list[i]*2)[:self.K.index_max],\
            #         (self.K.nu_0_list[i][:self.K.index_max]),\
            #         ms=5,lw=1.5,label = r"$\widetilde{\kappa}(R)$",alpha = 0.3)

            _b = axIr.plot(self.K.radius_norm_list[-1]*2,self.K.nu_0_list[-1],'.-', \
                color='white',lw=1.5,ms=10, label = r"$\langle \nu(R)\rangle $",alpha=0.6)
            lns = lns + _b

        # plt.figure()
        # plt.plot(self.K.radius_norm,self.K.nu_0)

        ###############
        try:
            #Join all the plot labels.
            c=axIr.plot([0,0],[0,0],label = r"$S/N ="+str(self.K.signal_to_noise1)+"$",color="darkmagenta",\
                ls = '-.',lw=0.5)
            axIr.axvline(2*(self.K.radius_norm[self.K.index_min:])[ int(self.K.snrr[-1]-self.K.index_min) ],\
                color="darkmagenta",ls = '-.',lw=0.5)
            lns = lns +c
        except:
            lns = lns

        axIr.set_ylabel(r"$\nu(R)$",fontsize = 12)
        axIr.set_xlim(0.0,2.0)
        axIr.set_ylim(-0.1,1.1)
        if "--err" in sys.argv:
            axIr.errorbar(self.K.radius_norm[self.K.index_min:]*2, self.K.nu_0[self.K.index_min:],(-0.1*self.K.MURerr),ms=0, lw=1, alpha=0.3, fmt='ok', ecolor='black')
        labs = [l.get_label() for l in lns]
        axIr.legend(lns, labs, loc = "upper right",prop={"size":10})

        # plt.plot(self.K.radius_norm*2,self.K.kurvature)
        # axIr.set_ylim(-15,15)
        # plt.savefig(self.K.rootname+"_kurvature.png",dpi=96,bbox_inches='tight')
        plt.savefig(self.K.rootname+"_kurvature.svg",dpi=600,bbox_inches='tight')
        # plt.show()
        # plt.clf()
        # plt.close()

    def plot_interpolations(self):
        # plt.plot(self.radius_norm,self.nu_0)
        # plt.plot(self.radius_norm,self.nu)
        # plt.plot(self.K.radius_norm,self.K.mu_0)
        # plt.show()
        fig = plt.figure(constrained_layout=True)
        plt.plot(self.K.radius_norm,self.K.nu_0,lw=2,color='red',label=r"$\nu_0$")
        plt.plot(self.K.radius_extr[0:10],self.K.nu_0_extr[0:10],label=r"$\nu_{0 extr}$")
        plt.plot(self.K.radius_norm,self.K.nu,lw=2,color='green',label=r"$\nu$")
        plt.plot(self.K.radius_norm,self.K.nu-self.K.nu_0,lw=2,label=r"residual")
        frac_err = np.sum(((self.K.nu-self.K.nu_0)/self.K.nu))
        plt.title(r"Information lost = $"+str(format(frac_err,'.2f'))+'\%$')
        plt.legend()
        # plt.plot(self.radius_extr,self.mu_0_extr)
        plt.savefig(self.K.rootname+"_profile_interpolation.svg",dpi=300,bbox_inches='tight')
        # plt.show()
        # plt.clf()
        # plt.close()

        # plt.figure()
        # plt.plot(self.K.radius,np.log(self.K.profile),'.')
        plt.show()


    def plot_2D(self):
        "MAKE 2D PLOTS OF DERIVATIVES AND CURVATURE"
        def rot180(imagem,x0,y0):
            R = np.matrix([[-1,0],[0,-1]])
            img180 = nd.affine_transform(imagem, R, offset=(2.*(y0-0.5),2.*(x0-0.5)) )
            return img180

        def background_asymmetry(pos, img):
            """"
            Reconstruct a sample of the sky with the same area
            as the Petrosian region of the galaxy and measure
            its asymmetry.
            """
            x, y = pos
            seed = int(100*x*y)
            np.random.seed(seed)

            num_pix = (self.Ss.petromaskFit).sum()
            size = int(np.sqrt(num_pix))
            bg_mask = ((~(self.Ss.petromaskFit).astype(bool)) * img)
            bg_pixels = bg_mask[np.where((bg_mask != 0))].ravel()
            pixels = np.random.choice(bg_pixels, size*size)

            bg_reconstructed = pixels.reshape(size, size)

            bgsub = bg_reconstructed - rot180(bg_reconstructed, size//2, size//2)
            bgasy = np.sum(abs(bgsub))/np.sum(abs(img))

            return bgasy

        def assimetria0(pos, img):
            (x0, y0) = pos
            psfmask = self.psfmask(self.S.psfsigma, *img.shape, x0, y0)
            imgorig = self.Ss.petromaskFit * psfmask * img
            imgsub  = self.Ss.petromaskFit * psfmask * (img- rot180(img,x0,y0))

            return (np.sum(abs(imgsub))/np.sum(abs(imgorig)))

        def assimetria1(pos, img):

            def mad(x):
                "median absolute deviation"
                return np.median( np.abs( x - np.median(x)) )

            x0, y0 = pos
            A1img  = np.abs(img - rot180(img, x0,y0))/(np.sum(np.abs(img)) )
            A1mask =  A1img > np.median(A1img) + 5 * mad(A1img)
            return np.sum( A1mask * A1img)

        def assimetria3(pos, img):
            x0, y0 = pos
            #psfmask = self.psfmask(self.S.psfsigma, *img.shape, x0,y0)
            # segementation
            #return -spearmanr(img[self.P.petroRegionIdx].ravel(), \
            #                  (rot180(img,x0,y0))[self.P.petroRegionIdx].ravel())[0]
            #sersic fit
            return -spearmanr(img.ravel(), \
                           (rot180(img,x0,y0)).ravel())[0]

        def cont_lines(data):
            data_min = np.median(data) + 0.1*np.std(data)
            data_max = data.max()
            data_steps = 15
            data_sequence = np.linspace(data_max,data_min,data_steps)
            # for data_iso in data_sequence:
                # delta = 0.05*data_iso
                # y,x = where((data>data_iso-delta) & (data<data_iso+delta))
            return data_sequence

        x0A = self.K.Kurvature_2D_filter.shape[0]/2
        y0A = self.K.Kurvature_2D_filter.shape[1]/2


        """
        fig = plt.figure(constrained_layout=True)
        gs = gridspec.GridSpec(2, 2,hspace=0.7, wspace=0.4,figure=fig)
        fig.set_size_inches(14.0, 10.0)

        # cut_i = int(self.K.Kurvature_2D_filter.shape[0]/2 - 2.0*self.K.RP)
        # cut_f = int(self.K.Kurvature_2D_filter.shape[0]/2 + 2.0*self.K.RP)
        cut_i = 0
        cut_f = -1

        ax_dxdy_root = fig.add_subplot(gs[0:1,0])
        ax_dxdy_root.set_title(r"Square root $\delta\nu(x,y)$")
        DS = np.sqrt(self.K.dx_filt**2.0+self.K.dy_filt**2.0)
        plt.gray()
        # WK = mfmlib.wlet2D(self.P.galnostars, scales=(1,2))
        klibs.imshow((DS)[cut_i:cut_f,cut_i:cut_f],sigma=2.5)
        ax_dxdy_root.contour((DS)[cut_i:cut_f,cut_i:cut_f],cont_lines((DS)[cut_i:cut_f,cut_i:cut_f])[::-1], colors='g')
        x0A1fit, y0A1fit  = fmin(assimetria1, (x0A,y0A), args=(DS[cut_i:cut_f,cut_i:cut_f],), disp=0)
        A1_dnu    = assimetria1((x0A1fit,y0A1fit), DS[cut_i:cut_f,cut_i:cut_f])

        x0A3fit, y0A3fit  = fmin(assimetria3, (x0A,y0A), args=(DS[cut_i:cut_f,cut_i:cut_f],), disp=0)
        A3_dnu    = 1+assimetria3((x0A3fit,y0A3fit), DS[cut_i:cut_f,cut_i:cut_f])
        ax_dxdy_root.set_title(r"Square root $\delta\nu(x,y)$  ,  "+"$A_1(\\delta \\nu)=$"+str(format(A3_dnu,'.2f')))
        # plt.annotate(r"$A_3(\delta \nu)=$"+str(format(A3_dnu,'.2f')),(0.1,0.93),xycoords='figure fraction',fontsize=12,color='blue')
        # plt.annotate(r"$A_1(\delta \nu)=$"+str(format(A1_dnu,'.2f')),(0.1,0.93),xycoords='figure fraction',fontsize=12,color='blue')


        # x0BGfit, y0BGfit = fmin(background_asymmetry, (x0A,y0A), args=(DS,))
        # x0A0fit, y0A0fit  = fmin(assimetria0, (x0A,y0A), args=(DS,), disp=0)


        # ABG   = background_asymmetry((x0BGfit, y0BGfit), DS)
        # A0    = assimetria0((x0A0fit, y0A0fit), DS) - ABG




        # ax_dxdy_root.contour(np.arcsin(DS)[cut_i:cut_f,cut_i:cut_f],cont_lines(DS[cut_i:cut_f,cut_i:cut_f])[::-1], colors='g',alpha=0.3)
        # ax_dxdy_root.contour(np.arcsin(DS)[cut_i:cut_f,cut_i:cut_f], 10, colors='g')
        # ax_dxdy_root.imshow(np.log(DS)[cut_i:cut_f,cut_i:cut_f])


        ax_d2xd2y = fig.add_subplot(gs[0:1,1])

        D2 = self.K.d2x_filt+self.K.d2y_filt
        klibs.imshow((D2)[cut_i:cut_f,cut_i:cut_f],sigma=2.5)
        ax_d2xd2y.contour((D2)[cut_i:cut_f,cut_i:cut_f],cont_lines((D2)[cut_i:cut_f,cut_i:cut_f])[::-1], colors='g')
        # ax_d2xd2y.contour(np.arcsin(D2)[cut_i:cut_f,cut_i:cut_f], 10, colors='g')
        # ax_d2xd2y.contour(np.arcsin(D2)[cut_i:cut_f,cut_i:cut_f],cont_lines(D2[cut_i:cut_f,cut_i:cut_f])[::-1], colors='g',alpha=0.3)
        # ax_d2xd2y.imshow(np.log(D2)[cut_i:cut_f,cut_i:cut_f])
        x0A1fit, y0A1fit  = fmin(assimetria1, (x0A,y0A), args=(D2[cut_i:cut_f,cut_i:cut_f],), disp=0)
        A1_d2nu    = assimetria1((x0A1fit,y0A1fit), D2[cut_i:cut_f,cut_i:cut_f])
        x0A3fit, y0A3fit  = fmin(assimetria3, (x0A,y0A), args=(D2[cut_i:cut_f,cut_i:cut_f],), disp=0)
        A3_d2nu    = 1+assimetria3((x0A3fit,y0A3fit), D2[cut_i:cut_f,cut_i:cut_f])
        ax_d2xd2y.set_title(r"$\sum \delta ^2 \nu (x,y)$  ,  "+"$A_1(\\delta^2 \\nu)=$"+str(format(A3_d2nu,'.2f')))
        # plt.annotate(r"$A_3(\delta^2 \nu)=$"+str(format(A3_d2nu,'.2f')),(0.6,0.93),xycoords='figure fraction',fontsize=12,color='blue')
        # plt.annotate(r"$A_1(\delta^2 \nu)=$"+str(format(A1_d2nu,'.2f')),(0.6,0.93),xycoords='figure fraction',fontsize=12,color='blue')


        ax_kur = fig.add_subplot(gs[1:2,0:2])
        K2D = self.K.Kurvature_2D_filter
        x0A1fit, y0A1fit  = fmin(assimetria1, (x0A,y0A), args=(K2D[cut_i:cut_f,cut_i:cut_f],), disp=0)
        x0A3fit, y0A3fit  = fmin(assimetria3, (x0A,y0A), args=(K2D[cut_i:cut_f,cut_i:cut_f],), disp=0)
        A3_K  = 1+assimetria3((x0A3fit,y0A3fit), K2D[cut_i:cut_f,cut_i:cut_f])
        A1_K    = assimetria1((x0A1fit,y0A1fit), K2D[cut_i:cut_f,cut_i:cut_f])
        ax_kur.set_title(r"$\kappa 2D \ $  ,  "+"$A_1(\kappa)=$"+str(format(A3_K,'.2f')))

        klibs.imshow((K2D)[cut_i:cut_f,cut_i:cut_f],sigma=2.5)


        # ax_kur.annotate(r"$A_3(\kappa)=$"+str(format(A3_K,'.2f')),(0.5,0.6),xycoords='figure fraction',fontsize=12,color='blue')


        # ax_kur.contour(np.arcsin(K2D)[cut_i:cut_f,cut_i:cut_f], 10, colors='g')
        # print("sdjtgdfgjdspofkap",cont_lines(K2D[cut_i:cut_f,cut_i:cut_f]))
        ax_kur.contour((K2D)[cut_i:cut_f,cut_i:cut_f],cont_lines((K2D)[cut_i:cut_f,cut_i:cut_f])[::-1], colors='g')
        # print("std k", np.std(K2D))
        # cont_lines
        # ax_kur.imshow(np.log(K2D)[cut_i:cut_f,cut_i:cut_f])

        np.save(self.K.rootname+'_kurvature.npy',K2D)
        np.save(self.K.rootname+'_2nd_der.npy',D2)
        np.save(self.K.rootname+'_1st_der.npy',DS)
        plt.savefig(self.K.rootname+"_kur2D.svg",dpi=200,bbox_inches='tight')
        plt.clf()
        plt.close()
        # plt.show()
        """

class save_data():
    def __init__(self,kurv):
        self.K = kurv
        self.save_csv_npy()
    def save_csv_npy(self):
        # dir_name = os.getcwd()
        self.K.dir_name = os.path.dirname(self.K.filename)
        self.K.savefile = os.path.splitext(self.K.filename)[0]
        # rootname2 = self.K.rootname.replace('_IR', '')
        self.K.base = os.path.splitext(os.path.basename(self.K.filename.replace('_IR.csv', '')))[0]

        """Saving the data"""

        np.savetxt(self.K.rootname+"_kurR.csv", list(zip(self.K.radius,\
            self.K.radius_norm,self.K.kurvature_filter,self.K.nu)),\
            delimiter=',',header ="#R,Rnorm,kurR,nu")

        try:
            f = open(self.K.rootname+".kur",'w')
            f.write("#gal_name,area,area_pos,area_neg,k_sum_mean,n_area,k_mean,n_k_mean,k_median,n_k_median,k_std,n_k_std,k_variance,n_k_variance,k_mode,n_k_mode,k_pstdev,n_k_pstdev,Ar1,Ar2,Ar3,k_area_inner,k_area_outer,k_mean_inner,k_mean_outer,k_inner_sum_mean,k_outer_sum_mean,n_inner_area,n_outer_area,k_inner_mean,k_inner_mean,n_inner_k_mean,n_outer_k_mean,k_inner_median,k_outer_median,n_inner_k_median,n_outer_k_median,k_inner_std,k_outer_std,n_inner_k_std,n_outer_k_std,k_inner_variance,k_outer_variance,n_inner_k_variance,n_outer_k_variance,k_inner_mode,k_outer_mode,n_inner_k_mode,n_outer_k_mode,k_inner_pstdev,k_outer_pstdev,n_inner_k_pstdev,n_outer_k_pstdev,kur_skew,kur_kurt,kur_inner_kurt,kur_inner_skew,kur_outer_kurt,kur_outer_skew,q_kur,Sq_kur,q_profile,Sq_profile,q_nu,Sq_nu, HC_K_new,HCM_K_new,HC_nu_new,HCM_nu_new,Hkur,Hnu,k_inner_q,k_outer_q,k_inner_Sq,k_outer_Sq,HC_inner_K_new,H_inner_kur,HC_outer_K_new,H_outer_kur,HCM_inner_K_new,HCM_outer_K_new,C1i,C1o,C2i,C2o,R50i,R50o,right_ips,peaks_flag,kur_area_model,"+'\n')
            f.write(self.K.base+','+str(self.K.area)+','+str(self.K.area_pos)+','+str(self.K.area_neg)+','+str(self.K.k_sum_mean)+','+str(self.K.n_area)+','+str(self.K.k_mean)+','+str(self.K.n_k_mean)+','+str(self.K.k_median)+','+str(self.K.n_k_median)+','+str(self.K.k_std)+','+str(self.K.n_k_std)+','+str(self.K.k_variance)+','+str(self.K.n_k_variance)+','+str(self.K.k_mode)+','+str(self.K.n_k_mode)+','+\
                str(self.K.k_pstdev)+','+str(self.K.n_k_pstdev)+','+str(self.K.Ar1)+','+str(self.K.Ar2)+','+str(self.K.Ar3)+','+str(self.K.k_area_inner)+','+str(self.K.k_area_outer)+','+str(self.K.k_mean_inner)+','+str(self.K.k_mean_outer)+','+str(self.K.k_inner_sum_mean)+','+str(self.K.k_outer_sum_mean)+','+str(self.K.n_inner_area)+','+str(self.K.n_outer_area)+','+str(self.K.k_inner_mean)+','+\
                str(self.K.k_inner_mean)+','+str(self.K.n_inner_k_mean)+','+str(self.K.n_outer_k_mean)+','+str(self.K.k_inner_median)+','+str(self.K.k_outer_median)+','+str(self.K.n_inner_k_median)+','+str(self.K.n_outer_k_median)+','+str(self.K.k_inner_std)+','+str(self.K.k_outer_std)+','+str(self.K.n_inner_k_std)+','+str(self.K.n_outer_k_std)+','+str(self.K.k_inner_variance)+','+str(self.K.k_outer_variance)+','+\
                str(self.K.n_inner_k_variance)+','+str(self.K.n_outer_k_variance)+','+str(self.K.k_inner_mode)+','+str(self.K.k_outer_mode)+','+str(self.K.n_inner_k_mode)+','+str(self.K.n_outer_k_mode)+','+str(self.K.k_inner_pstdev)+','+str(self.K.k_outer_pstdev)+','+str(self.K.n_inner_k_pstdev)+','+str(self.K.n_outer_k_pstdev)+','+str(self.K.kur_skew)+','+str(self.K.kur_kurt)+','+\
                str(self.K.kur_inner_kurt)+','+str(self.K.kur_inner_skew)+','+str(self.K.kur_outer_kurt)+','+str(self.K.kur_outer_skew)+','+str(self.K.q_kur)+','+str(self.K.Sq_kur)+','+str(self.K.q_profile)+','+str(self.K.Sq_profile)+','+str(self.K.q_nu)+','+str(self.K.Sq_nu)+','+str(self.K.HC_K_new)+','+str(self.K.HCM_K_new)+','+str(self.K.HC_nu_new)+','+str(self.K.HCM_nu_new)+','+str(self.K.Hkur)+','+str(self.K.Hnu)+','+str(self.K.k_inner_q)+','+str(self.K.k_outer_q)+','+\
                str(self.K.k_inner_Sq)+','+str(self.K.k_outer_Sq)+','+str(self.K.HC_inner_K_new)+','+str(self.K.H_inner_kur)+','+str(self.K.HC_outer_K_new)+','+str(self.K.H_outer_kur)+','+str(self.K.HCM_inner_K_new)+','+str(self.K.HCM_outer_K_new)+','+\
                str(self.K.CC1[0])+','+str(self.K.CC1[1])+','+str(self.K.CC2[0])+','+str(self.K.CC2[1])+','+str(self.K.RR50[0])+','+str(self.K.RR50[1])+','+\
                str(self.K.peaks_properties["right_ips"][0])+','+str(self.K.peaks_flag)+','+str(self.K.kur_area_model)+','+'\n')

        except:
            f = open(self.K.rootname+".kur",'w')
            f.write("#gal_name,area,area_pos,area_neg,k_sum_mean,n_area,k_mean,n_k_mean,k_median,n_k_median,k_std,n_k_std,k_variance,n_k_variance,k_mode,n_k_mode,k_pstdev,n_k_pstdev,Ar1,Ar2,Ar3,k_area_inner,k_area_outer,k_mean_inner,k_mean_outer,k_inner_sum_mean,k_outer_sum_mean,n_inner_area,n_outer_area,k_inner_mean,k_inner_mean,n_inner_k_mean,n_outer_k_mean,k_inner_median,k_outer_median,n_inner_k_median,n_outer_k_median,k_inner_std,k_outer_std,n_inner_k_std,n_outer_k_std,k_inner_variance,k_outer_variance,n_inner_k_variance,n_outer_k_variance,k_inner_mode,k_outer_mode,n_inner_k_mode,n_outer_k_mode,k_inner_pstdev,k_outer_pstdev,n_inner_k_pstdev,n_outer_k_pstdev,kur_skew,kur_kurt,kur_inner_kurt,kur_inner_skew,kur_outer_kurt,kur_outer_skew,q_kur,Sq_kur,q_profile,Sq_profile,q_nu,Sq_nu, HC_K_new,HCM_K_new,HC_nu_new,HCM_nu_new,Hkur,Hnu,k_inner_q,k_outer_q,k_inner_Sq,k_outer_Sq,HC_inner_K_new,H_inner_kur,HC_outer_K_new,H_outer_kur,HCM_inner_K_new,HCM_outer_K_new,right_ips,peaks_flag,"+'\n')
            f.write(self.K.base+','+str(self.K.area)+','+str(self.K.area_pos)+','+str(self.K.area_neg)+','+str(self.K.k_sum_mean)+','+str(self.K.n_area)+','+str(self.K.k_mean)+','+str(self.K.n_k_mean)+','+str(self.K.k_median)+','+str(self.K.n_k_median)+','+str(self.K.k_std)+','+str(self.K.n_k_std)+','+str(self.K.k_variance)+','+str(self.K.n_k_variance)+','+str(self.K.k_mode)+','+str(self.K.n_k_mode)+','+\
                str(self.K.k_pstdev)+','+str(self.K.n_k_pstdev)+','+str(self.K.Ar1)+','+str(self.K.Ar2)+','+str(self.K.Ar3)+','+str(self.K.k_area_inner)+','+str(self.K.k_area_outer)+','+str(self.K.k_mean_inner)+','+str(self.K.k_mean_outer)+','+str(self.K.k_inner_sum_mean)+','+str(self.K.k_outer_sum_mean)+','+str(self.K.n_inner_area)+','+str(self.K.n_outer_area)+','+str(self.K.k_inner_mean)+','+\
                str(self.K.k_inner_mean)+','+str(self.K.n_inner_k_mean)+','+str(self.K.n_outer_k_mean)+','+str(self.K.k_inner_median)+','+str(self.K.k_outer_median)+','+str(self.K.n_inner_k_median)+','+str(self.K.n_outer_k_median)+','+str(self.K.k_inner_std)+','+str(self.K.k_outer_std)+','+str(self.K.n_inner_k_std)+','+str(self.K.n_outer_k_std)+','+str(self.K.k_inner_variance)+','+str(self.K.k_outer_variance)+','+\
                str(self.K.n_inner_k_variance)+','+str(self.K.n_outer_k_variance)+','+str(self.K.k_inner_mode)+','+str(self.K.k_outer_mode)+','+str(self.K.n_inner_k_mode)+','+str(self.K.n_outer_k_mode)+','+str(self.K.k_inner_pstdev)+','+str(self.K.k_outer_pstdev)+','+str(self.K.n_inner_k_pstdev)+','+str(self.K.n_outer_k_pstdev)+','+str(self.K.kur_skew)+','+str(self.K.kur_kurt)+','+\
                str(self.K.kur_inner_kurt)+','+str(self.K.kur_inner_skew)+','+str(self.K.kur_outer_kurt)+','+str(self.K.kur_outer_skew)+','+str(self.K.q_kur)+','+str(self.K.Sq_kur)+','+str(self.K.q_profile)+','+str(self.K.Sq_profile)+','+str(self.K.q_nu)+','+str(self.K.Sq_nu)+','+str(self.K.HC_K_new)+','+str(self.K.HCM_K_new)+','+str(self.K.HC_nu_new)+','+str(self.K.HCM_nu_new)+','+str(self.K.Hkur)+','+str(self.K.Hnu)+','+str(self.K.k_inner_q)+','+str(self.K.k_outer_q)+','+\
                str(self.K.k_inner_Sq)+','+str(self.K.k_outer_Sq)+','+str(self.K.HC_inner_K_new)+','+str(self.K.H_inner_kur)+','+str(self.K.HC_outer_K_new)+','+str(self.K.H_outer_kur)+','+str(self.K.HCM_inner_K_new)+','+str(self.K.HCM_outer_K_new)+','+\
                str(self.K.peaks_properties["right_ips"][0])+','+str(self.K.peaks_flag)+','+'\n')


        # f.write("#gal_name,Area,abs_area,Area_pos,Area_neg,ArNinner,ArNouter,ksum,Nksum,kint,Nkint,sumInner,sumOuter,NsumInner,NsumOuter,mean_Nkurv,std_Nkurv,var_Nkurv,cumsum_Nkurv,positive_mean_Nkurv,negative_mean_Nkurv,positive_Nkurv_std,negative_kurv_std,positive_Nkurv_var,negative_Nkurv_var,NArea,NArea_pos,NArea_neg,NKmeanInner,NKmeanOuter,NKstdInner,NKstdOuter,NAreaInner,NAreaOuter,kur_var,kur_skew,kur_kurt,ArNpos,ArNneg,ArInnerPos,ArInnerNeg,ArOuterPos,ArOuterNeg,Ar1,Ar2,Ar3,C1i,C1o,C2i,C2o,right_ips,"+'\n')
        # f.write(self.K.base+','+str(self.K.Area)+','+str(self.K.absolute_Area)+','+str(self.K.Area_pos)+','+str(self.K.Area_neg)+','+str(self.K.ArNinner)+','+\
        # str(self.K.ArNouter)+','+str(self.K.ksum)+','+str(self.K.Nksum)+','+str(self.K.kint)+','+\
        # str(self.K.Nkint)+','+str(self.K.sumInner)+','+str(self.K.sumOuter)+','+str(self.K.NsumInner)+','+\
        # str(self.K.NsumOuter)+','+str(self.K.mean_Nkurv)+','+str(self.K.std_Nkurv)+','+str(self.K.var_Nkurv)+','+\
        # str(self.K.cumsum_Nkurv)+','+str(self.K.positive_mean_Nkurv)+','+str(self.K.negative_mean_Nkurv)+','+\
        # str(self.K.positive_Nkurv_std)+','+str(self.K.negative_kurv_std)+','+str(self.K.positive_Nkurv_var)+','+\
        # str(self.K.negative_Nkurv_var)+','+str(self.K.NArea)+','+str(self.K.NArea_pos)+','+str(self.K.NArea_neg)+','+\
        # str(self.K.NKmeanInner)+','+str(self.K.NKmeanOuter)+','+str(self.K.NKstdInner)+','+str(self.K.NKstdOuter)+','+\
        # str(self.K.NAreaInner)+','+str(self.K.NAreaOuter)+','+str(self.K.kur_var)+','+str(self.K.kur_skew)+','+str(self.K.kur_kurt)+','+\
        # str(self.K.ArNpos)+','+str(self.K.ArNneg)+','+str(self.K.ArInnerPos)+','+str(self.K.ArInnerNeg)+','+str(self.K.ArOuterPos)+','+\
        # str(self.K.ArOuterNeg)+','+str(self.K.Ar1)+','+str(self.K.Ar2)+','+str(self.K.Ar3)+','+\
        # str(self.K.CC1[0])+','+str(self.K.CC1[1])+','+\
        # str(self.K.CC2[0])+','+str(self.K.CC2[1])+','+str(self.K.peaks_properties["right_ips"][0])+','+'\n')
        #
        #
        f.close()




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Calculates the curvature\
                        of a given galaxy image.')
    parser.add_argument('-profile', '--profile', required=True,\
                        help='The input array of the light profile of the galaxy.')
    parser.add_argument('-r', '--radius', required=False,\
                        help='The radius array of the galaxy.')
    parser.add_argument('-err', '--error', required=False,\
                        help='Error of the profile.')
    parser.add_argument('-rp', '--petro', required=False,
                        help='The Petrosian radius.')
    parser.add_argument('-mpr', '--mpr', required=False,
                        help='Multiple Profiles (e.g. as a function of \
                        wavelength, PAs, etc).')
    parser.add_argument('-i', '--image', required=False,
                        help='FITS image. Please, for better results, you sould \
                        give a clean image, e.g. masked stars.')
    parser.add_argument('-show', '--show',required=False, nargs='?', default=False, const=True,
                        help='Show results.')
    # parser.add_argument('-s', '--save-plots', required=False,
    #                     help='Save plots (need to implement):\
    #                           0: only curvature plot.\
    #                           1: minimal plots.\
    #                           2: all plots. \
    #                           3: only kur 2D plot.')

    # parser.add_argument('-df', '--do_filter', required=False,
    #                     help='Use filter routine?.')

    args = parser.parse_args()
    #Run routines.
    print(">Starting Code")
    print("    # Reading input data")
    # print("    ")
    if args.mpr is not None:
        K = data_handle(args.profile,multiple_profiles = args.mpr)
    else:
        K = data_handle(args.profile)

    print(">Starting Kurvature Calculations")
    kurvature(K)


    print(">Performing Statistics")
    do_statistics(K)

    kmorph(K)
    if args.image is not None:
        # print(args.image)
        _k2D = kur_2D(K,args.image)
        fit  = kSersic(K)
    make_plots(K)

    save_data(K)


    exit()
