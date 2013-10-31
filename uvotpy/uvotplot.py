'''
  These are function for making plots of UVOT grism stuff.
  
  binplot(*args, **kwargs)
      Bin up the arrays with the keyword bin=<number>
  
  waveAccPlot( wave_obs,pix_obs, wave_zmx, pix_zmx, disp_coef, acc, order=None) 
                display a figure of the accuracy of the wavelength solution
  
  contourpk(x,y,f, levels=None,xb=None,xe=None,yb=None,ye=None):
     contour plot with 1-D array inputs for X, Y and F , limits [xb,xe], [yb,ye]
     
     
'''
try:
  from uvotpy import uvotplot,uvotmisc,uvotwcs,rationalfit,mpfit,uvotio
except:
  pass  
import uvotgetspec as uvotgrism

import numpy as N
import pyfits
from pylab import ioff,ion,arange, plot, subplot, xlim, ylim, title, xlabel, \
     ylabel, polyval, figure, contour, plt, legend, polyval, polyfit, savefig, \
     text , grid, clf, gca
import os
import uvotmisc, uvotgetspec

def binplot(*args, **kwargs):
   '''Bin up the arrays with the keyword bin=<number> 
      Same parameters as used by plot (pyplot)
   
   '''
   if 'bin' in kwargs.keys():
      nbin = kwargs['bin']
      del kwargs['bin']
      print 'uvotplot nbin = ', nbin
      nargs = len(args)
      print 'uvotplot nargs = ',nargs
      if nargs == 1:
         x = args[0]
	 m = int( (len(x)+0.5*nbin)/nbin)+1
	 print 'uvotplot m = ', m
	 xx = 0.0*x[0:m].copy()
	 for i in range(len(x)): 
	    j = int(i/nbin)
	    xx[j] = xx[j] + x[i]
	 if xx[m-1] == 0.0: xx = xx[0:m-1]   
	 args = (xx)
      else:
         x = args[0]
         y = args[1]
	 m = int( (len(x)+0.5*nbin)/nbin)+1
	 print 'uvotplot m = ', m,' len(x) = ',len(x)
	 xx = 0.0*x[0:m].copy()
	 yy = xx.copy()
	 for i in range(len(x)): 
	    j = int(i/nbin)
	    xx[j] = xx[j] + x[i]
	    yy[j] = yy[j] + y[i]
	 if xx[m-1] == 0.0:
	    xx = xx[0:m-1]
	    yy = yy[0:m-1]   
	 xx = xx/nbin   
	 if nargs == 2: 
	    args = xx, yy
         elif nargs == 3:
	    z1 = args[2] 
	    args = xx, yy, z1
	 elif nargs == 4: 
	    z1 = args[2]
	    z2 = args[3]
	    args = xx, yy, z1, z2
	 else:
	    print 'cannot handle more than 4 arguments'
	    args = xx,yy   
      plot(*args, **kwargs)
   else:
      plot(*args, **kwargs)
      return

def zmxCoefOnAxis():
   ''' 
   These are the dispersion coefficients from the 
   ReportZemaxUVGrism.v1.3, (UV nominal) in reverse order 
   so they can be used with polyval
   '''
   return N.array([8.14e-10, -1.634e-6, 1.366e-3, 3.206, 2597.9])
   
def plot_ellipsoid_regions(Xim,Yim,Xa,Yb,Thet,b2mag,matched,ondetector,
      img_pivot,img_pivot_ori,img_size,limitMag,img_angle=0.0,lmap=False,
      makeplot=True,color='k',annulusmag=13.0,ax=None,chatter=1):
   '''  
   This routine is to plot ellipsoid regions on the grism image/graph, which 
   may be a rotated, cropped part of the detector image
   
   Parameters ellipse
   ------------------
   Xim, Yim : ndarray
       center ellipse: Xim,Yim, 
   
   Xa, Xb : ndarray
       length X-axis Xa, length Y-axis Xb,
   
   Thet : float   
       angle ellipse orientation major axis Thet
       
   b2mag : ndarray    
       B magnitude b2mag
       
   matched : ndarray, bool    
       indicating a match between USNO-B1 and uvotdetected 
       
   ondetector : ndarray, bool
       indicating center is on the detector image
       
   Parameters image
   ----------------  
   img_angle : float
         rotation of the detector image (to left) 
	  
   img_pivot_ori : list,ndarray[2]
         the original X,Y detector coordinate of the center of rotation 
	
   img_pivot : list, ndarray[2]
         the coordinate of the center of rotation in the rotated image
	 
   img_size : list, ndarray[2]
         the size of the image
	  
    Parameters map
    --------------
    lmap : bool 
        if lmap=True, produce a truth map excluding the selected ellipses   
   
   Returns
   -------
   None or boolean map image, plots an ellipse on the current figure     
   '''   
   from uvotmisc import uvotrotvec
   from numpy import where, sin, cos, ones, asarray, outer
   
   ann_size = 49.0
   
   # validate the input (TBD)
   if (chatter > 1) & makeplot: print "plotting ellipsoid regions on image for zeroth orders"
   if chatter > 2:
      print 'plot_ellipsoid_regions input data: shape Xim, etc ', Xim.shape
      print 'Yim ',Yim.shape,'  Xa ',Xa.shape,'  Yb ',Yb.shape,'  Thet ',Thet.shape
      print 'img_pivot = ',img_pivot
      print 'omg_pivot_ori = ',img_pivot_ori
      print 'img_size = ',img_size
      print 'limitMag = ',limitMag
      print 'img_angle = ',img_angle
      print 'lmap = ',lmap
      print 'annulusmag = ',annulusmag
   if chatter > 3:   
      print 'B2mag :',b2mag
   img_size = asarray(img_size)   
   if len(img_size) != 2:
      print "error img_size must be the x and y dimensions of the image"   
      return   
      
   # rotate the ellipse data and place on the coordinate system of the image
   if img_angle != 0.0:
      X,Y = uvotrotvec( Xim -img_pivot_ori[0], Yim -img_pivot_ori[1], -img_angle )
      #X += img_pivot[0]
      Y = Y + img_pivot[1]
   else:
      X = Xim -img_pivot_ori[0] + img_pivot[0]
      Y = Yim -img_pivot_ori[1] + img_pivot[1]

   # select items on the image with B2mag > limitMag
   if ax == None:
      xmin = 0  #-img_pivot[0]
      xmax = img_size[0] # img_size[0] + xmin
      ymin = 0  # -img_pivot[1]
      ymax = img_size[1] # img_size[1] + ymin
   else:  # select the image slice
      xlimits = ax.get_xlim()
      ylimits = ax.get_ylim()
      xmin = xlimits[0] -img_pivot[0]
      xmax = xlimits[1] #  img_size[0] + xmin
      ymin = ylimits[0] # -img_pivot[1]
      ymax = ylimits[1] #  img_size[1] + ymin
   if chatter > 2:
      print "Plot_ellipsoid_regions center limits to  X:", xmin, xmax,"   Y:",ymin,ymax   
   
   q = where((b2mag < limitMag) & (X > xmin) & (X < xmax) & (Y > ymin) & (Y < ymax))
   nq = len(q[0])

   # saturated source with annulus
   qsat = where((b2mag < annulusmag) & (X > xmin) & (X < xmax) & (Y > ymin) & (Y < ymax))
   nqsat = len(qsat[0])
   
   if chatter > 4: 
      print 'xmin, xmax, ymin, ymax = ', xmin,xmax, ymin, ymax
      print 'normal selection q = ',q
      print 'len(q[0]) ', nq
      print 'saturated selection qsat = ',qsat
      print 'len(qsat[0]) ', nqsat
   
   if nq == 0: 
      if chatter > 2: print "no zeroth order regions within magnitude bounds found "
      makeplot = False
      
   if chatter > 1:
      print "found ",nqsat," bright source(s) which may have a bright annulus on the image"   
   
   # scale the ellipse axes according to Bmag 
   # calibrate to some function of Bmag, limitMag, 
   # probably length depends on sptype brightness, width is limited.
   Xa1 = 14.0 + 0.* Xa.copy() + 1.5*(19-b2mag)
   Yb1 = 5.5  + 0.* Yb.copy()
   
   # plot the ellipses on the current image
   if makeplot:
     for i in range(nq):
        qq = q[0][i]
        ang = Thet[qq]-img_angle
        if chatter>4: 
           print 'plotting ellipse number ',qq
	   print 'angle = ',ang
      
        Ellipse( (X[qq],Y[qq]), (Xa1[qq], Yb1[qq]), ang, lw=1, color=color )
      
   # plot saturated annulus on the current image
     if nqsat > 0:      
        for i in range(nqsat):
           qq = qsat[0][i]
           ang = Thet[qq]-img_angle
           if chatter>4: 
              print 'plotting annulus number ',qq
	      print 'angle = ',ang 
           Ellipse( (X[qq],Y[qq]), (ann_size, ann_size), ang, lw=1, color=color )
	 
   if lmap: 
   # create a truth map for the image excluding the ellipses
      mapimg = ones(img_size, dtype=bool) 
      
      if nq == 0:
         if chatter > 1:
	    print 'no zeroth orders to put on map. mapimg.shape = ',mapimg.shape
         return mapimg

      else:
         for i in range(nq):
            qq = q[0][i]
            x,y,a,b,th = X[qq],Y[qq], Xa1[qq], Yb1[qq], Thet[qq]-img_angle 
            maskEllipse(mapimg, x,y,a,b,th)
      
      if nqsat > 0:
         # update the truth map for bright annulus excluding a circular region
         for i in range(nqsat): 
            qq = qsat[0][i]
            x,y,a,b,th = X[qq],Y[qq], ann_size, ann_size, Thet[qq]-img_angle 
            maskEllipse(mapimg, x,y,a,b,th)
	    if chatter > 1: print "masked bright source annulus at position [",x,",",y,"]"
	 		
      return mapimg    


def maskEllipse(maskimg, x,y,a,b,theta, test=0, chatter=1):
   '''update a mask excluding ellipse region
   
   Parameters
   ----------
   maskimg : ndarray, 2D, bool 
      boolean array to aplly mask to (i.e., numpy.ones( array([200,400]),dtype=bool) )
      
   x,y : int, float   
      ellipse center coordinate x,y
      
   a,b : float   
      ellipse major axis a; minor axis b; 
    
   theta : float   
      rotation angle theta counterclockwise in deg.
      
   Returns
   -------
   maskimg with all pixels inside the ellipse are set to False
   
   note
   ----
    x and y , a and b are interchanged 	  
   '''
   from numpy import sin, cos, abs, arange, ones, where, outer, asarray, pi
   
   maskimg = asarray(maskimg)
   ca = 1./(a*a)
   cb = 1./(b*b)
   th = theta / 180. * pi
   m11 = cos(th)
   m12 = sin(th)
   m21 = -sin(th)
   m22 = cos(th)
	  # locate coordinates (xmin, ymin) (xmax, ymax)
	  # and operate on the subset 
   xmin, xmax = x-abs(a), x+abs(a)+1
   ymin, ymax = y-abs(a), y+abs(a)+1
   x8,x9 = xmin, xmax
   y8,y9 = ymin, ymax
   
   # if ellipse (x,y,a,b,theta) outside maskimg, then return
   if not ( (xmin < x) & (x < xmax) & (ymin < y) & (y < ymax) ):
      return maskimg

   subimsize=maskimg[x8:x9,y8:y9].shape
   x7 = outer(arange(subimsize[0]) - abs(a),ones(subimsize[1]))
   y7 = outer(ones(subimsize[0]),arange(subimsize[1]) - abs(a))
   zx6 = m11*x7+m12*y7
   zy6 = m21*x7+m22*y7
   if test == 1: 
      maskimg[x8:x9,y8:y9][where(ca*zx6*zx6+cb*zy6*zy6 <= 1.0)] = False
   else:   
          img_size = maskimg.shape
          x1 = outer(arange(img_size[0]) - x,ones(img_size[1]))
          y1 = outer(ones(img_size[0]),arange(img_size[1]) - y)
          zx = m11*x1+m12*y1
          zy = m21*x1+m22*y1
          maskimg[where(ca*zx*zx+cb*zy*zy <= 1.0)] = False
   if chatter > 2:	    
      print 'center (',x,',',y,')'
      print 'ellipse a = ',a,'  b = ',b,'    theta = ',theta
      print ca,cb,m11,m12,m21,m22
      print xmin,xmax,ymin,ymax
      print x8,x9,y8,y9
      print subimsize
      print x7
      print y7
      print x1,y1
      print maskimg.shape
   return maskimg
	  
def Ellipse((x,y), (rx, ry), angle=0.0, resolution=200,  **kwargs):
    ''' 
    plot an ellipse using an N-sided polygon 
    
    Parameters
    ----------
    (x,y) : float
      centre ellipse
      
    (rx,ry) : float
      half axis ellipse
      
    angle : float
      angle in units of degrees
    
    resolution : int
      determines number of points to use
    
    and additional kwargs for pyplot.plot()
    
    Note
    ----
    Can only plot one ellipse at a time.
    '''
    from numpy import arange, cos, sin, pi
    from matplotlib.pylab import plot
    from uvotmisc import uvotrotvec
    
    # check x is a single value etc.
    
    theta = 2.0*pi/resolution*arange(resolution)
    xs = rx * cos(theta)
    ys = ry * sin(theta)
    if angle != 0.0:
        xs, ys = uvotrotvec(xs,ys,angle)
    xs += x
    ys += y	
    return plot(xs,ys,'-', **kwargs)
   

def contourpk(x,y,f, levels=None,xb=None,xe=None,yb=None,ye=None,s=60,kx=1,ky=1,dolabels=True, **kwargs):
   '''Make a contour plot with 1-D array inputs for X, Y and F. This is a  
   wrapper to convert lists of points (X,Y,Z) in 2-D arrays, then calls contour()
   
   Parameters
   ----------
   X, Y: ndarrays[:], 1D on a 2D plane
     coordinates X, Y
   Z : ndarray[:], 1D function on X,Y
   
   kwargs : dict
   -------------
    - **xb,xe,yb,ye** : float
      limits x,y for bispline interpolation valid region
    - **s** : float
      smoothing parameter for bisplrep
    - **kx, ky** : int
      order for the interpolation 
    - **dolabels** : bool
      labels on the contours if true
    - **levels** : list
      contour levels 
             
   Note
   ----
   warning: X, Y axis may have been interchanged
   ''' 
   import numpy
   from scipy import interpolate
   from pylab import contour, plt
   x1, x2, y1, y2 = min(x), max(x), min(y), max(y)
   xx = numpy.linspace(x1, x2)
   yy = numpy.linspace(y1, y2)
   X, Y = numpy.meshgrid(xx, yy)
   shp = X.shape
   task = 0
   tck = interpolate.bisplrep(x,y,f,kx=kx,ky=ky,s=s,xb=xb,xe=xe,yb=yb,ye=ye)     
   Z = interpolate.bisplev(xx, yy, tck)
   if levels == None:
      C = contour(Y, X, Z,**kwargs)
   else:
      C = contour(Y, X, Z, levels=levels,**kwargs)  
   if dolabels:     
      plt.clabel(C, inline=1,fontsize=10)    
   return Y,X,Z,tck, C
   

def waveAccPlot(wave_obs,pix_obs, wave_zmx, pix_zmx, disp_coef, obsid=None, 
    acc=None, order=None, wheelpos=200, figureno=1,legloc=[1,2]):
   '''Plots of the accuracy of the wavelength solution from zemax compared to
   the observed wavelengths.
     
   Parameters
   ----------  
   wave_obs,  pix_obs : ndarray 
      observed wavelengths points (green circles)
      
   wave_zmx ,pix_zmx : ndarray
      calculated zemax points (or the interpolated solution (red crosses) 
      
   disp_coef : ndarray
      dispersion coefficients
      
   disp_coef : list
      coefficients in reverse order: if p is of length N, this the polynomial
     is as follows for coeff named p:
     
          y(x) = p[0]*(x**N-1) + p[1]*(x**N-2) + ... + p[N-2]*x + p[N-1]
   
   kwargs : dict
   
    - **acc** : accuracy in wavelength 
      
    - **order** : order of polynomial disp_coef (default len(coef) )
    
    - **obsid** : if given, append to  title                      
   
   Notes
   -----
   
   **Figure description**
   
   x-axis :  pix - pixel number referenced to [260nm in first order]

   *Top panel only*
    
   y-axis: lambda - lambda_linear
   
   *linear term in the dispersion*
   a linear term is fit to the wavelengths 
   
      $\lambda_{lin}$ = coef[0]+coef[1]*pix
   
   *Bottom panel only*
   
   y-axis: residuals
     
      wave_obs, pix_obs - wave(pix_obs)  (green circles)
      wave_zmx, pix_zmx - wave(pix_zmx)  (red crosses)
      
   '''
   if wheelpos < 500:
      ref_wave = 2600.
      titl = 'Wavelength accuracy UV grism - '
      textstart = 1600
   else:
      ref_wave = 4200.
      titl = 'Wavelength accuracy V grism - '
      textstart = 2700   
   # zero of pix_obs forced to ref_wave (2600.or 4200.) for initial plot
   if order == None:
      order = len(disp_coef)
   dcoef = polyfit(wave_obs,pix_obs,order)
   doff=polyval(dcoef,ref_wave)
   pix_obs = pix_obs - doff
   print "fit through observations pixel position of anchor = ",doff
   
   n1, n2 = len(pix_obs), len(pix_zmx)
   pix1 = N.zeros( (n1+n2) )
   pix1[0:n1,] = pix_obs
   pix1[n1:(n1+n2),] = pix_zmx
   pix2 = pix1.min()
   pix3 = pix1.max()
   pix = N.arange(pix2,pix3)
   wav = polyval(disp_coef, pix)
   #           wavlin = disp_coef[-1]+disp_coef[-2]*xxx_pix  
   #           linear term in dispersion:
   w_obs  = wave_obs - (disp_coef[-1]+disp_coef[-2]*pix_obs)
   w_zmx  = wave_zmx - (disp_coef[-1]+disp_coef[-2]*pix_zmx)
   wavlin = wav - (disp_coef[-1]+disp_coef[-2]*pix)
   zero_offset = (wave_obs-polyval(disp_coef, pix_obs+doff) ).mean()
   zo = zero_offset
   if acc == None:
      wave_off = (wave_obs-polyval(disp_coef, pix_obs+doff) )
      acc = wave_off.std()
      print ' initial acc (all points) = ',acc 
      # remove outlyers
      q_in = N.where(abs(wave_off-zo) < 3.* acc)
      acc = (wave_off[q_in]).std()
      print ' after removing outliers: acc = ', acc 
      print 'accuracy of the fit = ',acc, ' angstrom'
   stracc =    str(((10*acc+0.5).__int__())/10.) +'$\AA$'
   zero_offset = ((10*zero_offset+0.5).__int__())/10.
   txt = '<$\Delta\lambda$> = '+str(zero_offset)+'$\AA\ \ \ \sigma_{observed-model}$ = '+stracc

   figure( num=figureno )

   subplot(211)
   plot(pix, wavlin, '-')
   plot(pix_obs,w_obs,'ob')
   plot(pix_zmx,w_zmx,'+r')
   ylabel('$\lambda$ - $\lambda_{linear}$  ($\AA$)')
   xlabel('pixels')
   if order == 4: 
     sord = 'fourth '
   elif order == 3:
     sord = 'third '
   elif order == 2:
     sord = 'second '
   elif order == 1: 
     sord = 'first '
   else: 
     sord = 'unknown '        
   legend((sord+'order fit','observed data','model'),loc=legloc[1])
   if obsid == None: obsid=''
   title(titl+obsid)
   # a = getp( gca )
   # setp(a, xlim=(pix1,pix2), xticks=[])

   subplot(212)
   w1 = wave_obs-polyval(disp_coef, pix_obs+doff)
   w2 = wave_zmx-polyval(disp_coef, pix_zmx)
   plot(wave_obs,w1, 'ob',label='_nolegend_')
   plot(wave_zmx,w2, '+r',label='_nolegend_')
   p0 = pix*0.
   p1 = p0 - acc+zo
   p2 = p0 + acc+zo
   plot(wav,p0,'-r',label='_nolegend_')
   plot(wav, p1,'--b',label='1-$\sigma$ limits')
   plot(wav, p2,'--b',label='_nolegend_' )
   ylabel('$\Delta\lambda$ ($\AA$)')
   xlabel('$\lambda$ ($\AA$)')
   a = gca()
   ylim = a.get_ylim()
   #if (ylim[0] > -16.0): ylim=(-16.0,ylim[1])
   ylim=(zo-2.1*acc,zo+2.1*acc)
   a.set_ylim(ylim)
   legend(loc=legloc[0])
   text(textstart,ylim[0]*0.9,txt)
   #a = getp( gca )
   #lim1, lim2 = 1.1*max(w1), 1.1*min(w1)
   #setp(a, xlim=(lim1,lim2)) #, xticks=[])
   savefig('accuracy.png')
   return acc, zero_offset

def make_spec_plot(nspec=10, parmfile='plotparm.par',wheelpos=160):
   '''
   Reads parameters from a comma delimited file 
   Each line is for one plot
   nspec is the number of plots on a page. 
   
   Note: this program has not been used since 2010, so probably needs updating
   '''
   # read plot parameter file in list
   f = open(parmfile,"r")
   plines = f.readlines()
   f.close()
   nfig = len(plines)
   nplot = (nfig+1)/nspec
   pwd = os.getcwd()
   NN = nspec*3000
   if wheelpos == 160: clocked = True
   if wheelpos == 200: clocked = False
   for kp in range(nplot):       
      xwa = N.zeros(NN).reshape(3000,nspec)
      xsp = N.zeros(NN).reshape(3000,nspec)
      speclen = N.zeros(nspec)
      texp= N.zeros(nspec)
      id  = N.empty(nspec,dtype='|S40')
      nsubplot = 0
      for kf in range(nspec):   # process the data
         print 'length list ' , len(xwa), len(xsp)
         nfig -= 1  
	 if nfig < 0: break
	 nsubplot += 1
         k = kf+nspec*kp  # specific plot
	 dir_,filestub,ra,dec,ext1,lfilt1,ext2,filt2,wpixscale,spextwid = (plines[k]).split(',')
	 ra = float(ra)
	 dec = float(dec)
	 print "procesing: ",dir_,filestub,ra,dec,ext1,lfilt1,ext2,filt2,wpixscale,spextwid
	 print "filestub = ", filestub
	 print "extension= ", ext1
	 print "width spectral extraction = ",spextwid
	 print "changing directory . . ."
	 os.chdir(dir_)
	 print "new directory = ", os.getcwd()
	 print "processing figure ",k," . . . "
	 if filt2 == "None" : filt2 = None
	 out = uvotgetspec.getSpec(ra,dec,filestub,int(ext1),lfilter=lfilt1, lfilt2=filt2,chatter=1,lfilt2_ext=int(ext2), spextwidth=int(spextwid), clocked=clocked)    
	 ( (dis, spnet, angle, anker, anker2, anker_field, ank_c), \
            (bg, bg1, bg2, extimg, spimg, spnetimg, offset) , \
            (C_1,C_2, img, H_lines, WC_lines), hdr ) = out
	 exposure = hdr['EXPOSURE']   
	 pos = ank_c[1]   # anchor in extracted image
	 ll = max( (pos-350,0) )  
	 #ul = min( (pos+1900, len(dis)-pos) ) 
	 print 'ul' , (pos+1900, len(dis))
	 ul = pos+1900        
	 print "exposure time = ", exposure
	 print "spectrum pixel range = ",ll," -- ",ul
	 print "saving spectrum . . . number ", kf 
	 wav = (polyval(C_1,dis[ll:ul]))
	 spe = (spnet[ll:ul])
	 speclen[kf] = len(wav)
	 figure(4+kp); plot(wav,spe/exposure); xlim(1700,6500)
	 xsp[:speclen[kf],kf]  = spe/exposure 
	 xwa[:speclen[kf],kf]  = wav
	 texp[kf] = exposure
	 id[kf] = filestub+'['+str(ext1)+']'
         #
      # calculate # plots left
      grid
      xlim(1750,6500)
      ylim(0,8)	         
      savefig('/Volumes/users/Users/kuin/caldata/specplot_sums'+str(kp)+'.png')
      spmax = 0.7*xsp.max()
      print "plotting spectra . . ."
      clf()
      for kf in range(nsubplot):   # make the plots
         subplot(nspec,1,kf)
         #k = kf+nspec*kp  ; specific plot
	 #
	 wl = xwa[:speclen[kf],kf]
	 sp = xsp[:speclen[kf],kf]
	 texpo = texp[kf]
	 plot(wl, sp, 'k', ls='steps') 
	 xlim(1700,6000)
	 ylim(0,spmax)
	 #
         text(1800,0.85*spmax,id[kf]+' '+str(texp[kf])+'s')
         grid
      xlabel('wavelength')
      ylabel('countrate')	 
      savefig('/Volumes/users/Users/kuin/caldata/specplot_'+str(kp)+'.png')
      # perhaps make here a summed spectrum
   os.chdir(pwd)
   return None
       
#####################################################################################
