#!/usr/bin/python
"""
Original code from
http://nicky.vanforeest.com/misc/fitEllipse/fitEllipse.html

but notation changed to that of  www.wikipedia.com


"""
from __future__ import division
from pylab import *
import astropy.io.fits as pyfits
import numpy as np
from time import sleep
from numpy.linalg import eig, inv
from sys import argv
# import pyff

'''
def fitEllipse(x,y):
    x = x[:,np.newaxis]
    y = y[:,np.newaxis]
    # design matrix
    D =  np.hstack((x*x, x*y, y*y, x, y, np.ones_like(x)))
    # scatter matrix
    S = np.dot(D.T,D)
    # constrain matrix
    C = np.zeros([6,6])
    C[0,2] = C[2,0] = 2; C[1,1] = -1
    E, V =  eig(np.dot(inv(S), C))
    n = np.argmax(np.abs(E))
    a = V[:,n]
    return (a)
'''


def fitEllipse2(x,y):
    ''' cf.
        NUMERICALLY STABLE DIRECT LEAST SQUARES FITTING OF ELLIPSES,
        Halir, Flusser.
    '''
    x = x[:,np.newaxis]
    y = y[:,np.newaxis]
    D =  np.hstack((x*x, x*y, y*y, x, y, np.ones_like(x)))
    D1 = mat(D[:,:3])
    D2 = mat(D[:,3:])
    S1 = D1.T * D1
    S2 = D1.T * D2
    S3 = D2.T * D2
    C1= zeros((3,3))
    C1[2,0] = 2
    C1[1,1] =-1
    C1[0,2] = 2
    C1 = mat(C1)
    M = C1.I * (S1 - S2 * S3.I * S2.T)
    E,V = eig(M)
    VV = asarray(V)
    #print ( VV[:,1]**2 -  4*VV[:,0] * VV[:,2] < 0)

    idx =  where(E>0)[0]
    if len(idx)==0:
        idx = 0
    a1 = V[:,idx]
    a2 = - S3.I * S2.T * a1

    fit = ravel((a1,a2))
    A,B,C,D,E,F = fit
    Delta = ((A*C-B**2)*F + B*E*D/4 - C*D**2/4 - A*E**2/4)
    if C*Delta >=0:
        print('non real ellipse')
    return fit


#
# http://en.wikipedia.org/wiki/Matrix_representation_of_conic_sections
#
# also see http://mathworld.wolfram.com/Ellipse.html

def ellipse_center(fit):
    A,B,C,D,E,F = fit
    num1 = B**2-4*A*C
    x0=(2*C*D-B*E)/num1
    y0=(2*A*E-B*D)/num1
    return np.array([x0,y0])

def ellipse_angle( fit ):
    A,B,C,D,E,F = fit


    PA = (0.5*arctan2(B,A-C))
    # if PA<0: PA=PA+pi/2.
    return PA



    '''
    a = A
    b = B/2
    c = C

    if b < 1e-10:
        if a<c:
            PA = 0
        if a>c:
            PA = pi/2
    elif b>1e-10:
        if a<c:
            PA = 0.5*np.arctan2(2*b, (a-c))
        if a>c:
            PA = pi/2. + 0.5*np.arctan2(2*b, (a-c))
    return PA
    '''

def ellipse_axis_length( fit ):
    A,B,C,D,E,F = fit

    Aq = mat([[A, B/2., D/2.], [B/2., C, E/2.], [D/2., E/2., F]])
    A33 = Aq[:2,:2]

    lam1, lam2 = eigvals(A33)
    a = sqrt(abs(det(Aq)/(lam1*det(A33))))
    b = sqrt(abs(det(Aq)/(lam2*det(A33))))
    return array([max(a,b), min(a,b)])
    '''
    b,c,d,f,g,a = fit[1]/2, fit[2], fit[3]/2, fit[4]/2, fit[5], fit[0]

    print b**2-4*a*c < 0
    up = 2*( a*f**2 + c*d**2 + g*b**2 - 2*b*d*f - a*c*g)
    #down1=(b*b-a*c)*( (c-a)*np.sqrt(1+4*b*b/((a-c)*(a-c)))-(c+a))
    #down2=(b*b-a*c)*( (a-c)*np.sqrt(1+4*b*b/((a-c)*(a-c)))-(c+a))
    down1=(b**2-a*c)*( (-1)*np.sqrt((a-c)**2 + 4*b**2 ) - (a+c))
    down2=(b**2-a*c)*(      np.sqrt((a-c)**2 + 4*b**2 ) - (a+c))
    res1=np.sqrt(up/down1)
    res2=np.sqrt(up/down2)
    return np.array([res1, res2])
    #return np.array([max(res1,res2), min(res1,res2)])
    '''

def ellipse_axis_length_orig(fit):
    b,c,d,f,g,a = fit[1]/2, fit[2], fit[3]/2, fit[4]/2, fit[5], fit[0]
    up = 2*(a*f*f+c*d*d+g*b*b-2*b*d*f-a*c*g)
    down1=(b*b-a*c)*( (c-a)*np.sqrt(1+4*b*b/((a-c)*(a-c)))-(c+a))
    down2=(b*b-a*c)*( (a-c)*np.sqrt(1+4*b*b/((a-c)*(a-c)))-(c+a))
    res1=np.sqrt(up/down1)
    res2=np.sqrt(up/down2)
    return np.array([res1, res2])


def fitted_params(fit):
    'returns x0, y0, PA, a, b'
    x0,y0 = ellipse_center(fit)
    PA    = ellipse_angle(fit)
    a,b   = ellipse_axis_length(fit)
    return x0, y0, PA, a, b

def fitted_ellipse_points(fit, N=100):
    ''' returns N points in the fitted ellipse
    Useful for plotting purposes '''

    R = linspace(0,2*np.pi, N)

    x0, y0, PA, a, b = fitted_params(fit)

    xe = x0 + a*cos(R)*cos(PA) - b*sin(R)*sin(PA)
    ye = y0 + a*cos(R)*sin(PA) + b*sin(R)*cos(PA)
    return xe,ye


def main_test2(g):

    # g = pyfits.getdata(name)

    dxc = int(g.shape[0]*0.25*0.5)
    dyc = int(g.shape[1]*0.25*0.5)
    dc  = int(g.shape[0]/2)
    print(dxc,dc,dc-dxc)
    gal= g#[dc-dyc:dc+dyc,dc-dxc:dc+dxc]

    M,N = gal.shape

    #levels are almost equally spaced in log(log(I)) for a Sersic profile
    R = linspace(0,2*np.pi, 500)
    Imin = (np.log10(  np.median(gal) + 0.8*np.std(gal) ))
    Imax = 0.7*(np.log10(  gal.max() ))
    Isteps = 50
    Isequence = (10**(linspace(Imax,Imin,Isteps)))

    # print(Imin, Imax, Isequence)

    Iellipses = zeros_like(gal)

    gal0 = gal.copy()
    II   = []
    AA   = []
    BB   = []
    PPAA = []
    PHI  = []
    for I in Isequence:
        try:
            delta = 0.1*I
            y,x = where((gal>I-delta) & (gal<I+delta))

            # remove outliers
            remove_indexes = np.where(np.logical_or(abs(y - y.mean())>abs(y - y.mean()).mean()*2.0,\
                                                    abs(x - x.mean())>abs(x - x.mean()).mean()*2.0)==True)[0]
            x = np.delete(x,remove_indexes)
            y = np.delete(y,remove_indexes)

            if len(x)<6:
                continue
            try:
                fit   = fitEllipse2(x,y)
            except:
                continue
            x0,y0 = ellipse_center(fit)
            phi   = ellipse_angle(fit)
            if phi<0:
                phi = phi + np.pi
            a,b   = ellipse_axis_length(fit)

            PA = np.rad2deg(phi)

            II.append(I)#Note that II is not on the same size as Isequence.
            AA.append(a)
            BB.append(b)
            PPAA.append(PA)
            PHI.append(phi)

            xe = x0 + a*cos(R)*cos(phi) - b*sin(R)*sin(phi)
            ye = y0 + a*cos(R)*sin(phi) + b*sin(R)*cos(phi)

            Ifit = mean(gal[y,x])
            Iellipses[y,x] = Ifit
            # rr = np.sqrt(xe**2.0+ye**2.0)


            '''
            #mostra ellipses sendo ajustadas...
            gal.fill(0.0)
            gal[y,x] = 1.0
            pyff.imshow((gal)+(gal0)/gal0.mean())
            plot(x,y,'.g')
            plot (xe/ZOOM,ye/ZOOM, '-r', lw=1)
            gray()
            draw()
            #raw_input('........')
            sleep(0.05)
            clf()
            gal = gal0.copy()
            '''
            loopc = 0
            fit2 = array([0,0,0,0,0,0])
            dpoints = 1

            # plot (xe,ye, '-b', lw=2)

            while dpoints !=1  and loopc <50 :
                #print mean(fit-fit2)


                ### Distance matrix
                '''
                xx = x[:,np.newaxis]
                yy = y[:,np.newaxis]
                # design matrix
                DD =  mat(np.hstack((xx*xx, xx*yy, yy*yy, xx, yy, np.ones_like(xx))))
                A,B,C,D,E,F = fit
                AA = mat([A, B, C, D, E, F])
                Q = sqrt( abs(DD * AA.T ))
                qq = ravel(Q)
                #hist (qq,100)
                qqmean = qq.mean()
                qqstd  = qq.std()
                '''

                m = sqrt((x-x0)**2/a**2 + (y-y0)**2/b**2)
                mmean = m.mean()
                mstd  = m.std()

                # Q based
                #idxin = where( qq<(qqmean+1.0*qqstd) )[0]
                # m based
                idxin = where( (m>(mmean-2.0*mstd))  & (m<(mmean+2.0*mstd)))[0]
                if  len(idxin)<6:
                    break

                # plot(x,y, '.r',ms=4)
                # plot(x[idxin],y[idxin], '+b', ms=4)
                # plot (xe,ye, '-k', lw=2)
                # imshow(arcsinh(gal))


                fit2 = fitEllipse2(x[idxin],y[idxin])
                xe2, ye2 = fitted_ellipse_points(fit2)
                # plot (xe2,ye2, '-b', lw=2)

                dpoints = abs( len(x)- len(x[idxin]))

                x = x[idxin]
                y = y[idxin]

                xlim(0,N)
                ylim(0,M)
                draw()
                loopc += 1


            print('ctr=(%8.2f %8.2f)     I=%8.2f     q=%8.2f     PA=%8.2f' % ( x0, y0, Ifit, b/a, rad2deg(phi) ))
        except:
            pass

    II = np.asarray(II)
    AA = np.asarray(AA)
    BB = np.asarray(BB)
    PPAA = np.asarray(PPAA)
    PHI = np.asarray(PHI)
    split = int(len(II)/2)
    qmi  = np.median((BB/AA)[:split])
    qmo  = np.median((BB/AA)[split:])
    PAmi = np.median( PPAA[:split])
    PAmo = np.median( PPAA[split:])

    # plt.xlabel('$X$')
    # plt.ylabel('$Y$')
    #
    # plt.annotate(r"$q$-mi= "+str(format(qmi, '2f')),  (0.76,0.90),xycoords='figure fraction',fontsize=10)
    # plt.annotate(r"$q$-mo= "+str(format(qmo, '2f')),  (0.76,0.87),xycoords='figure fraction',fontsize=10)
    # plt.annotate(r"$PA$-mi="+str(format(PAmi,'2f')),  (0.76,0.84),xycoords='figure fraction',fontsize=10)
    # plt.annotate(r"$PA$-mo="+str(format(PAmo,'2f')),  (0.76,0.81),xycoords='figure fraction',fontsize=10)
    #
    # # print(Isequence)
    # imshow(arcsinh(gal))
    # contour(gal, Isequence[::-1], colors='k')
    # gray()
    # plt.show()
    # # figure()
    #
    # plot(gradient(gradient(Isequence)))
    # plot(rr,log(Isequence))
    # plot(m)
    # show()

    return(qmi,qmo,PAmi,PAmo)


# if __name__ == '__main__':
#     main_test2()
