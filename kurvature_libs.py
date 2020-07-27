import numpy as np
import pylab as pl
import scipy
from scipy import signal

def get_str(File,string=None,HEADER=0):
    """
    Get string variable.
    """
    infile = open(File, 'r')
    firstLine = infile.readline()
    header=firstLine.split(',')
    if HEADER==1:
        # print(header)
        return(header)
    else:
        ind=header.index(string)
        return np.loadtxt(File,dtype='str',usecols=(ind),comments="#", \
            delimiter=",", unpack=False)

def get_data(File,param=None,HEADER=0):
    """
    Get a numerical variable from a table.

    HEADER: if == 1, display the file's header.
    """
    infile = open(File, 'r')
    firstLine = infile.readline()
    header=firstLine.split(',')
    if HEADER==1:
        # print(header)
        return(header)
    else:
        ind=header.index(param)
        return np.loadtxt(File,usecols=(ind),comments="#", delimiter=",", unpack=False)

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx],idx

def normalise(x):
    return (x- x.min())/(x.max() - x.min())


def lfilt(data):
    """
    scipy.signal.lfilter
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.lfilter.html
    """
    b,a  = scipy.signal.bessel(3,[0.2,0.5],btype="bandstop")
    zi   = scipy.signal.lfilter_zi(b,a)
    z, _ = scipy.signal.lfilter(b,a,data,zi = zi*data[0])
    z2,_ = scipy.signal.lfilter(b,a,z,zi=zi*z[0])
    y    = scipy.signal.filtfilt(b,a,data,padlen = 10 )
    return y


def resample_data(x_minor, y_minor,x_major):
    """
    We need to resample the data in order to comparre
    two arrays with different length.
    I select the array with major shape to resample the data.
    """
    y_major = signal.resample(y_minor,len(x_major),window=10)
    return y_major

def imshow(img, sigma=3, contours=0, bar=None, aspect='equal', extent=None, vmin=None, vmax=None, use_median=False):
    """
    improved version of pl.imshow,

    shows image with limits sigma above and below mean.

    optionally show contours and colorbar
    """

    def mad(x):
       return np.median( np.abs( x - np.median(x)) )

    # deals with NaN and Infinity
    img[np.where(np.isnan(img))]=0.0
    img[np.where(np.isinf(img))]=0.0


    # wether to use median/mad or mean/stdev.
    # note that mad ~ 1.5 stdev
    if use_median==False:
      if vmin==None:
         vmin = img.mean() - sigma * img.std()
      if vmax==None:
         vmax = img.mean() + sigma * img.std()
    else:
      if vmin==None:
         vmin = np.median(img) - 1.5*sigma * mad(img)
      if vmax==None:
         vmax = np.median(img) + 1.5*sigma * mad(img)


    pl.imshow(img, vmin=vmin, vmax=vmax, origin='lower', aspect=aspect, extent=extent, interpolation=None)

    if bar != None:
        pl.colorbar(pad=0)

    if contours >0:
        pl.contour(img, contours, colors='k', linestyles='solid', aspect=aspect, extent=extent)


def geo_mom(p,q,I, centered=True, normed=True, complex=False, verbose=False):
    """return the central moment M_{p,q} of image I
    http://en.wikipedia.org/wiki/Image_moment
    F.Ferrari 2012, prior to 4th JPAS
    From MFMTK.
    """

    M,N = I.shape
    x,y = np.meshgrid(np.arange(N), np.arange(M))

    M_00 = I.sum()

    if centered:
        # centroids
        x_c = (1/M_00) * np.sum(x * I )
        y_c = (1/M_00) * np.sum(y * I )

        x = x - x_c
        y = y - y_c

        if verbose:
            print('centroid  at', x_c,y_c)

    if normed:
        NORM = M_00**(1+(p+q)/2.)
    else:
        NORM = 1.0

    if complex:
        XX = (x+y*1j)
        YY = (x-y*1j)
    else:
        XX = x
        YY = y

    M_pq = (1/NORM) * np.sum( XX**p * YY**q * I)

    return M_pq

def q_PA(image):
    """
    From Morfometryka.
    """
    m00 = geo_mom(0,0,image,centered=0,normed=0)
    m10 = geo_mom(1,0,image,centered=0,normed=0)
    m01 = geo_mom(0,1,image,centered=0,normed=0)
    m11 = geo_mom(1,1,image,centered=0,normed=0)
    m20 = geo_mom(2,0,image,centered=0,normed=0)
    m02 = geo_mom(0,2,image,centered=0,normed=0)

    mu20 = geo_mom(2,0,image,centered=1,normed=0)
    mu02 = geo_mom(0,2,image,centered=1,normed=0)
    mu11 = geo_mom(1,1,image,centered=1,normed=0)

    # centroids
    x0col  = m10/m00
    y0col  = m01/m00

    # manor, minor and axis ratio
    lam1 = np.sqrt(abs(  (1/2.) * (mu20+mu02 + np.sqrt((mu20-mu02)**2 + 4*mu11**2))   )/m00)
    lam2 = np.sqrt(abs(  (1/2.) * (mu20+mu02 - np.sqrt((mu20-mu02)**2 + 4*mu11**2))   )/m00)
    a = max(lam1,lam2)
    b = min(lam1,lam2)

    PA = (1/2.) * np.arctan2(2*mu11,(mu20-mu02))
    if PA < 0:
        PA = PA+np.pi
    PAdeg = np.rad2deg(PA)
    a  = a
    b  = b
    q = b/a
    return PAdeg,b/a


"""
Useful commands.
rsvg-convert -f pdf -o .pdf .svg
cat *.mfmtk | sed '/#/d' | sed '/\\\//g'  > mfmkt.mfmtk
cat *.kur | sed '/#/d' | sed '/\\\//g'  > kur.csv
cat *multiple_profiles_data.csv | sed '/#/d' | sed '/\\\//g'  > multiple_profile_data.csv
cat *shape_PAs.csv | sed '/#/d' | sed '/\\\//g'  > kur_shape_PAs.csv
cat *.mfmtk | sed '/#/d' | sed '/\\\//g'  > mfmtk.csv
java -jar /home/geferson/Downloads/topcat-full.jar -f csv mfmtk.csv kur.csv
"""
"""To convert all the pdf plots (files) to jpg format, just do
$ for i in *.pdf; do convert -density 100 $i $i.jpg; done
on terminal
"""
