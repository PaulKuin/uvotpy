#!/usr/bin/env python
# -*- coding: iso-8859-15 -*-
'''
   These are some general purpose routines
'''
from __future__ import division
from __future__ import print_function
# Developed by N.P.M. Kuin (MSSL/UCL)
from builtins import range
from past.utils import old_div
__version__ = '20140501-0.5.0'

import numpy as N

def rdTab(file, symb=' ', commentsymb='#',get_comments=False):
   '''RdTab will read in a table of numerical values
   provided every record has the same number of fields.
   Comment lines start by default with a hash mark, but
   that can be changed by passing another symbol in commentsymb
   comments in data records are not supported.
   
   Parameters
   ----------
   file : str
     file name ascii table
   symb : str
     character used to separate the columns  
   commentsymb : str
     character used in first position of line for comments
   get_comments : bool
     if True, return comments only      
    
    Returns
    -------
    table : ndarray 
      a table of values
      
    Notes
    -----
    The table must have equal length columns with only numbers.
    
    Use rdList to read a table with character data  
    
    NPMK (MSSL) 2010
     '''
   f = open(file)
   l = f.readlines()
   f.close()
   n = len(l)
   if get_comments:
      comments = []
      for line in l:
         if line[0] == commentsymb:
            comments.append(line)
      return comments  
   ni = list(l)
   k = 0
   for i in range(n): 
      ni[i] = l[i][0]
   ni = N.array(ni)   
   q = N.where(ni != commentsymb)
   q = q[0]
   l = N.array(l)
   l = l[q]
   if (symb != ' '): 
     nc = len(l[0].split(symb)) 
   else:   
     nc = len(l[0].split())
   k = len(l)
   data = N.zeros( (k,nc) )
   j = 0
   for i in range(k):
      #print i,j,l[i]
      if symb != ' ': 
         xx = l[i].split(symb)
      else: 
         xx = l[i].split()        
      if len(xx) == nc: 
         data[j,:] = N.array(xx)
         j += 1
   return data     

def rdList(file, symb=' ',chatter=0,line1=None,line2=None,skip='#'):
   '''Put data in list: chatter>4 gives detailed output 
   restrict lines in file with line1, line2
   skip lines with the skip char in first position

   Parameters
   ----------
   file : str
     file name ascii table
   symb : str
     character used to split out the columns
   line1,line2 : int
     sub-select records[line1:line2] 
   chatter : int
    
   Returns
   -------
    table : ndarray 
      a table of values
      
   Notes
   -----
   The table must have equal length columns and the same number 
   of fields on each row/record.
    
   Use rdTab to read a table with numerical only data  
       ''' 
   f = open(file,'U')
   l = f.readlines()
   f.close()
   # choose simple filter on selected lines?
   if line1 != None:
      if line2 == None: l=l[line1:]
      else: l=l[line1:line2] 
   n = len(l)
   ni = list(l)
   k = 0
   for i in range(n): 
      ni[i] = l[i][0]
   ni = N.array(ni)   
   q = N.where(ni != skip)
   q = q[0]
   ngood = len(q)
   out=list()
   for i in range(ngood): 
     if symb == ' ':
        k = l[q[i]].split()
     else:      
        k = l[q[i]].split(symb)
     out.append( k )
     if chatter > 4: print('rdList: ',i,' - ',k) 
   return out

def uvotrotvec(X, Y, theta):
   '''rotate vectors X, Y over angle theta (deg) with origen [0,0]
   
   Parameters
   ----------
   X, Y : arrays
     coordinates
   theta : float
     angle in degrees
   
   Returns
   -------
   rx,ry : arrays
     rotated coordinates
     
   '''
   import math
   import numpy

   angle = theta*math.pi/180.
   m11 = math.cos(angle)
   m12 = math.sin(angle)
   m21 = -math.sin(angle)
   m22 = math.cos(angle)
   #matrix = numpy.array([[m11, m12],
   #        [m21, m22]], dtype = numpy.float64)
   RX = m11 * X + m12 * Y
   RY = m21 * X + m22 * Y
   return RX, RY
   

def bilinear1(x1,x2,x1a,x2a,f,chatter=0):
   '''Bilinear interpolation
   
   
   
   Notes
   -----
   Given function f(i,j) given as a 2d array of function values at
   points x1a[i],x2a[j], derive the function value y=f(x1,x2) 
   by bilinear interpolation. 
   
   requirement: x1a[i] is increasing with i 
                x2a[j] is increasing with j
                
                **BROKEN**: need to search for 4 points around x1,y1, then 
                interpolate just in those 4 points. Interp2d should do 
                the same. Now the indexing doesnot work right.

   No special treatment to handle points near the borders 
     (see uvotgrism.bilinear)           
                
   20080826 NPMK                
   '''
   import numpy as np
   # check that the arrays are numpy arrays 
   x1a = np.asarray(x1a)
   x2a = np.asarray(x2a)
   # find indices i,j for the square containing (x1, y1)
   ki = x1a.searchsorted(x1)-1
   kj = x2a.searchsorted(x2)-1
   if ((ki+1 >= len(x1a)) ^ (kj+1 >= len(x2a)) ^ (ki < 0) ^ (kj < 0) ):
      print('bilinear. point outside grid x - nearest neighbor ')
      if ki + 1 > len(x1a) : ki = len(x1a) - 1
      if ki < 0 : ki = 0
      if kj + 1 > len(x2a) : kj = len(x2a) - 1
      if kj < 0 : kj = 0
      return f[ki, kj]
   # 
   y1 = f[ki  ,kj]
   y2 = f[ki+1,kj]
   y3 = f[ki+1,kj+1]
   y4 = f[ki  ,kj+1]
   # 
   t = old_div((x1 - x1a[ki]),(x1a[ki+1]-x1a[ki]))
   u = old_div((x2 - x2a[kj]),(x2a[kj+1]-x2a[kj]))
   #
   y = (1.-t)*(1.-u)*y1 + t*(1.-u)*y2 + t*u*y3 + (1.-t)*u*y4
   if chatter > 1: 
      print('bilinear.                   x         y          f[x,y]    ')
      print('bilinear.   first  point ',x1a[ki  ],x2a[kj],  f[ki,kj])
      print('bilinear.   second point ',x1a[ki+1],x2a[kj],  f[ki+1,kj])
      print('bilinear.   third  point ',x1a[ki+1],x2a[kj+1],  f[ki+1,kj+1])
      print('bilinear.   fourth point ',x1a[ki  ],x2a[kj+1],  f[ki,kj+1])
      print('bilinear. fractions t, u ', t, u)
      print('bilinear. interpolate at ', x1, x2, y)
   return y    
   

def interpgrid(x,y, xlist,ylist, xmap, ymap, kx=3, ky=3, s=50):
   ''' for position x,y and a 2-D mapping map(list),
       i.e., xmap[xlist,ylist],ymap[xlist,ylist] given on a grid xlist,ylist; 
       the nearest xlist, ylist positions to each x,y pair are found and 
       interpolated to yield  mapx(x,y),mapy(x,y)
         
   x,y : rank-1 arrays of data points
   xlist, ylist, xmap, ymap: rank-1 arrays of data points
   
   +
   2008-08-24 NPMK (MSSL)
   '''
   from scipy import interpolate
   # check if the input is right data type
   # ... TBD
   
   # compute the Bivariate-spline coefficients
   # kx = ky =  3 # cubic splines (smoothing)
   task = 0 # find spline for given smoothing factor
   # s = 50 # spline goes through the given points
   # eps = 1.0e-6  (0 < eps < 1)
   
   #(tck_x, ems1) 
   tck_x = interpolate.bisplrep(xlist,ylist,xmap,kx=kx,ky=ky,s=s)
   #(fp1, ier1, msg1) = ems1
   #if ier1 in [1,2,3]: 
   #   print 'an error occurred computing the bivariate spline (xmap) '
   #   print ier1, msg1
   #   # raise error
   #   return None
   tck_y = interpolate.bisplrep(xlist,ylist,ymap,kx=kx,ky=ky,s=s) 
   #(fp2, ier2, msg2) = ems2
   #if ier2 in [1,2,3]: 
   #   print 'an error occurred computing the bivariate spline (ymap) '
   #   print ier2, msg2
   #   # raise error
   #   return None
   # compute the spline    
   
   xval = interpolate.bisplev(x, y, tck_x)
   yval = interpolate.bisplev(x, y, tck_y)
   
   return xval,yval

def hydrogenlines(dis_zmx,wav_zmx,xpix,ypix,wave, 
      wpixscale=0.960, 
      cpixscale = 1.0,
      lines = (1723,1908,2297,2405,2530,2595,2699,2733,2906,3070,4649),
      c_offset = (49.0,-12.6), 
      order = 1, wheelpos=200):
      H_lines=( 6563.35460346,  4861.74415071,  4340.84299171,  4102.09662716, 
        3970.42438975,  3889.39532057,  3835.72671631,  3798.23761775,      
        3770.96821946,  3750.48834484,  3734.70346123)
      return WC_zemaxlines(dis_zmx,wav_zmx,xpix,ypix,wave, 
      wpixscale=wpixscale, 
      cpixscale = cpixscale,
      lines = H_lines,
      c_offset = c_offset, 
      order = order, wheelpos=wheelpos)
            
def WC_zemaxlines(dis_zmx,wav_zmx,xpix,ypix,wave, 
      wpixscale=0.960, 
      cpixscale = 1.0,
      lines = (1723,1908,2297,2405,2530,2595,2699,2733,2906,3070,4649),
      c_offset = (49.0,-12.6), 
      order = 1,
      wheelpos = 200):
    '''
   returns the predicted positions of spectral lines predicted by zemax 
   the default lines is specifically for WR86. 
   
   The following scaling is applied. First, the cpixscale is applied to the 
   coordinate position to find the anker point at 2600A, first order. Second, 
   the pixel positions relative to the anker position are scaled with wpixscale. 
   
   input dis_zmx, wav_zmx is for selected order only [default 1]
   xpix, ypix for all orders (assumed unscaled), the(interpolated 
   raw output from zemax - for the anker location)
   
   wave[k] is the k-th wavelength that goes with xpix[:,k],ypix[:,k]
   
   use:
   (xplines, yplines, dislines, lines) = wr86zemaxlines(zmxdis[1],zmxwav[1],xpix,ypix,wave)
    
    '''
    import zemax
    import numpy as np
    
    polyorder=4
    wav_zmx = np.asarray(wav_zmx)
    if len(wav_zmx) < 7 : polyorder = 2
    coef = N.polyfit(wav_zmx,dis_zmx,polyorder)
    lines = N.array(lines)         
    # q = where(lines < wav_zmx[len(wav_zmx)])
    # lines = lines[q]
    dislines = N.polyval(coef, lines)
    if (wave.mean() < 10) : wave = wave*1e4
    
    # scale coordinates 
    xp1 = xpix[order,:]*cpixscale+c_offset[0]
    yp1 = ypix[order,:]*cpixscale+c_offset[1]
    
    if wheelpos < 300:
       # find anker position in the given order at 2600A
       k = N.where(wave == 2600.)
       xa = xpix[order,k]*cpixscale+c_offset[0] ; xa=xa.squeeze()
       ya = ypix[order,k]*cpixscale+c_offset[1] ; ya=ya.squeeze()
       xa, ya = zemax.correctAnkPos(xa, ya)
    else:
       # find anker position in the given order at 4200A
       k = N.where(wave == 4200.)[0]
       xa = xpix[order,k]*cpixscale+c_offset[0] ; xa=xa.squeeze()
       ya = ypix[order,k]*cpixscale+c_offset[1] ; ya=ya.squeeze()
       xa, ya = zemax.correctAnkPos(xa, ya)
     
    # fit polynomial to the dispersion-scaled distance to anker
    cx = N.polyfit(wave,(xp1-xa)*wpixscale,polyorder)
    cy = N.polyfit(wave,(yp1-ya)*wpixscale,polyorder)
    
    # find scaled position (add position anker back in)
    xplines = N.polyval(cx,lines) + xa
    yplines = N.polyval(cy,lines)       + ya
    
    return (xplines, yplines, dislines, lines)    
           
def get_keyword_from_history(hist,key):
   '''Utility to get the keyword from the history list.
   
   Parameters
   ----------
   hist : list
   key : str
   
   Returns
   -------
   value belonging to key or `None`.
   
   Notes
   -----
   The history records are written while processing getSpec() and 
   added to the FITS header of the output file. 
   
   These can be read from the header by just getting *all* the history records. 
   '''
   n = len(key)+1
   key1 = key+'='
   for h in hist:
      if h[0:n] == key1:
         return h.split('=')[1]
   else: return None    
   
def GaussianHalfIntegralFraction(x):
   ''' 
   Computes the normalised integrated gaussian from 
   -x to +x. For x=inf, the result equals 1.
   
   x is in units of sigma
   
   Abramowitz & Stegun, par. 26.2 
   '''
   import numpy as np    
   Z_x = old_div(np.exp( -x*x/2.), np.sqrt(2.* np.pi))
   p = .33267
   a1 =  .4361836
   a2 = -.1201676
   a3 =  .9372980
   t = old_div(1.,(1.+p*x))
   P_x = 1. - Z_x * t* (a1 + t*(a2 + a3*t) ) 
   A_x = 2.*P_x - 1
   return  A_x  

def uniq(list):
   ''' preserves order '''
   set = {}
   return [ set.setdefault(x,x) for x in list if x not in set ]


def swtime2JD(TSTART,useFtool=True,):
   '''Time converter to JD from swift time 
   
   Parameter
   ---------
   TSTART : float
     swift time in seconds
   useFtool: if False the time is converted without correction from 
   spacecraft time to UTC (about a second per year).   
     
   Returns
   -------
   JD : float
     Julian Date
   MJD : float
     Modified Julian Date
   gregorian : str 
     `normal` date and time
   outdate : datetime   
     python datetime object
   
   Notes
   -----
   example (input TSTART as a string) 
   for 2001-01-01T00:00:00.000 
       TSTART=0.000 
       MJD= 51910.00000000 
       JD=2451910.5
   '''
   import datetime
   month2number={'JAN':'01','FEB':'02','MAR':'03','APR':'04','MAY':'05','JUN':'06',
                 'JUL':'07','AUG':'08','SEP':'09','OCT':'10','NOV':'11','DEC':'12'}
   if type(TSTART) != float: 
       raise IOError('input TSTART must be float %s'%(type(TSTART)))
   if useFtool:
      import os
      from numpy.random import rand
      delt,status = swclockcorr(TSTART)
      if not status: 
         print("no time correction for SC clock drift")
         return swtime2JD(TSTART,useFtool=False)
      else:   
         xt = float(TSTART+delt)
         return swtime2JD(xt,useFtool=False)
   else:
      import numpy as np
      deltime = datetime.timedelta(0,TSTART,0)
      # delt[0] # days;   delt[1] # seconds;  delt[2] # microseconds
      swzero_datetime = datetime.datetime(2001,1,1,0,0,0)
      gregorian = swzero_datetime + deltime
      MJD = np.double(51910.0) + old_div(TSTART,(24.*3600))
      JD = np.double(2451910.5) + old_div(TSTART,(24.*3600))
      outdate = gregorian.isoformat()
   return JD, MJD, gregorian, outdate

def JD2swift(JD):
   import numpy as np
   delt, status = swclockcorr(swifttime) 
   if status: 
      return (JD - np.double(2451910.5))*(86400.0)-delt
   else:   
      print ('WARNING: no correction made for UT-> SC clock time')
      return (JD - np.double(2451910.5))*(86400.0)

def swift2JD(tswift):
   delt, status = swclockcorr(swifttime) 
   if status: 
      return old_div(tswift+delt,86400.0)  + 2451910.5
   else:   
      print ('WARNING: no correction made for UT-> SC clock time')
      return old_div(tswift,86400.0)  + 2451910.5

   
def MJD2swift(MJD):   
   import numpy as np
   delt, status = swclockcorr(swifttime) 
   if status: 
      return (MJD - np.double(51910.0))*(86400.0) - delt
   else:   
      print ('WARNING: no correction made for UT-> SC clock time')
      return (MJD - np.double(51910.0))*(86400.0)
   
def swift2MJD(tswift):
   delt, status = swclockcorr(swifttime) 
   if status: 
      return old_div(tswift+delt,86400.0)  + 51910.0 
   else:   
      print ('WARNING: no correction made for UT-> SC clock time')
      return old_div(tswift,86400.0)  + 51910.0
   
def UT2swift(year,month,day,hour,minute,second,millisecond,chatter=0):
   '''Convert time in UT to swift time in seconds.
   
   Parameters
   ---------- 
   
   year : int
     e.g., 2012
   month : str or int
     e.g., 'JAN' 
   day : int
     e.g., 21 
   hour : int
   minute : int
   second : int
   millisecond : int
   
   Returns
   ------- 
   swifttime : float
     in seconds (see Heasarc for more conversions) - needs clock correction 
   '''
   import datetime
   import numpy as np
   if chatter > 1: print(year, type(year), month, type(month), day,type(day), hour,type(hour), minute, type(minute), second, type(second), millisecond, type(millisecond))
   swzero_datetime = datetime.datetime(2001,1,1,0,0,0)
   if type(month) == str:
      defmonths={'JAN':1,'FEB':2,'MAR':3,'APR':4,'MAY':5,'JUN':6,'JUL':7,'AUG':8,'SEP':9,'OCT':10,'NOV':11,'DEC':12} 
      imon = defmonths[month.upper()]   
      if chatter > 1: print(imon, type(imon))
   else: imon = month   
   xx = datetime.datetime(year,imon,day,hour,minute,second,millisecond*1000)  
   xdiff = xx-swzero_datetime
   swifttime = xdiff.total_seconds() 
   # convert from UT to SC time
   delt, status = swclockcorr(swifttime) 
   if status: 
      return swifttime-delt
   else:   
      print ('WARNING: no correction made for UT-> SC clock time')
      return swifttime
  
def swclockcorr(met,met_tolerance=50):
    """ 
    Swift MET correction for clock drift etc. 
    
    Parameters:
    
    met: float
       the swift mission elapsed time
    met_tolerance: float
       tolerance in days past last CALDB entry   
       
    output parameters:
    
    tcorr : float 
       time correction in seconds
    success: bool      
       True: time corr computed from the CALDB file
       False: rough estimate (to ~2 sec)
       
    For a mission time of T, the correction in seconds is computed
    with the following:
    T1 = (T-TSTART)/86400
    TCORR = TOFFSET + (C0 + C1*T1 + C2*T1*T1)*1E-6
    """
    import os
    import numpy as np
    from astropy.io import fits
    # uncorrected date and time
    times=swtime2JD(met,useFtool=False)
    date = times[3][:10]
    time = times[3][11:19]
    # get file with corrections
    caldb = os.getenv("CALDB")
    command="quzcif swift sc - -  clock  "+date+" "+time+" - > quzcif.out"
    os.system(command)
    f = open("quzcif.out")
    result = True
    try:
       tcorfile, ext = f.read().split()
       ext = int(ext)
       f.close()
    except:
       f.close()
       f = open("quzcif.out")
       print (f.readlines())
       f.close()
       os.system("rm -f quzcif.out")
       raise RuntimeError('Swift SC clock file not found')
       #return np.polyval(np.array([4.92294757e-08,  -8.36992570]),met), result   
    os.system("rm -f quzcif.out")
    xx = fits.open(tcorfile)
    x = xx[ext].data
    k = (met >= x['tstart']) & (met < x['tstop'])
    if np.sum(k) != 1:
        if met > x['tstart'][-1]:
           k = (met >= x['tstart'])
           k = np.max(np.where(k))
           if (met - x['tstart'][k]) > met_tolerance*86400.0:
               print (met, x['tstart'][k], met_tolerance)
               print ("WARNING: update the Swift SC CALDB - it is out of date")
               result=False
           else:
               result=True    
        else:   
           raise IOError('input MET not found in Swift SC clock file')
           #return np.polyval(np.array([4.92294757e-08,  -8.36992570]),met), result
    t1 = old_div((met - x['tstart'][k]),86400.0)
    tcorr = x['toffset'][k] + ( x['C0'][k] + 
       x['C1'][k]*t1 + x['C2'][k]*t1*t1)*1.0e-6
    tcorr = -tcorr   # add tcorr to MET to get time in UT
    xx.close()
    return tcorr, result   
   
def get_dispersion_from_header(header,order=1):
   """retrieve the dispersion coefficients from the FITS header """ 
   import numpy as np
   hist = header['history']
   n = "%1s"%(order)   
   C = [get_keyword_from_history(hist,'DISP'+n+'_0')]
   if C == [None]: 
       raise RuntimeError("header history does not contain the DISP keyword")
   try:
      coef = get_keyword_from_history(hist,'DISP'+n+'_1')
      if coef != None: C.append(coef)
      try:
         coef = get_keyword_from_history(hist,'DISP'+n+'_2')
         if coef != None: C.append(coef)
         try:
            coef = get_keyword_from_history(hist,'DISP'+n+'_3')
            if coef != None: C.append(coef)
            try:
               coef=get_keyword_from_history(hist,'DISP'+n+'_4')
               if coef != None: C.append(coef)
            except:
               pass
         except:
            pass   
      except:
         pass    
   except:
      pass   
   return np.array(C,dtype=float)

def get_sigCoef(header,order=1):
   '''retrieve the sigma coefficients from the FITS header '''  
   import numpy as np
   hist = header['history'] 
   n = "%1s"%(order)   
   SIG1 = [get_keyword_from_history(hist,'SIGCOEF'+n+'_0')]
   try:
      coef=get_keyword_from_history(hist,'SIGCOEF'+n+'_1')
      if coef != None: SIG1.append(coef)
      try:
         coef=get_keyword_from_history(hist,'SIGCOEF'+n+'_2')
         if coef != None: SIG1.append(coef)
         try:
            coef=get_keyword_from_history(hist,'SIGCOEF'+n+'_3')
            if coef != None: SIG1.append(coef)
            try:
               coef = get_keyword_from_history(hist,'SIGCOEF'+n+'_4')
               if coef != None: SIG1.append(coef)
            except:
               pass
         except:
            pass   
      except:
         pass    
   except:
      pass   
   return np.array(SIG1,dtype=float)
  
def get_curvatureCoef(header,order=1):
   '''retrieve the sigma coefficients from the FITS header '''  
   import numpy as np
   hist = header['history'] 
   n = "%1s"%(order)   
   CURVE = [get_keyword_from_history(hist,'COEF'+n+'_0')]
   try:
      coef=get_keyword_from_history(hist,'COEF'+n+'_1')
      if coef != None: CURVE.append(coef)
      try:
         coef=get_keyword_from_history(hist,'COEF'+n+'_2')
         if coef != None: CURVE.append(coef)
         try:
            coef=get_keyword_from_history(hist,'COEF'+n+'_3')
            if coef != None: CURVE.append(coef)
            try:
               coef = get_keyword_from_history(hist,'COEF'+n+'_4')
               if coef != None: CURVE.append(coef)
            except:
               pass
         except:
            pass   
      except:
         pass    
   except:
      pass   
   return np.array(CURVE,dtype=float)
  
def parse_DS9regionfile(file,chatter=0):
   '''
   parse the region file
   
   Note
   ----
   return structure with data
   so far only for circle() 
   does not grab colour or annotation metadata
   '''
   F = open(file)
   f = F.readlines()
   F.close()
   
   signs = []
   position = []
   box = []
   boxtype = []
   
   try:
      if f[0].split(":")[1].split()[0] == "DS9":
         version=f[0].split(":")[1].split()[-1]
      else:
         version="0"
      filename = f[1].split(":")[1].split("\n")[0]
      epoch = f[3].split("\n")[0].split(":")[0] 
      wcs = 'wcs'
      r = f[3].split("\n")[0].split(":")
      if len(r) > 1:  # other coordinate system definition 
         wcs = r[1]                              
   except:
     print("Error reading region file : ",file)
   try: 
     r = f[4].split("\n")[0]
     if chatter > 3: 
        print("line# 4",r)
     if r[0:4].upper() == "WCS":
        wcs = r
     elif len(r) == 0:
        do_nothing = True       
     elif r.split("(")[0] == "circle" :
        signs.append("+")
        boxtype.append("circle")
        values = r.split("(")[0].split(")")[0].split(",")
        position.append(values[0:2])
        box.append(values[2:])
     elif r.split("(")[0] == "-circle" :
        signs.append("-")
        boxtype.append("circle")
        values = r.split("(")[0].split(")")[0].split(",")
        position.append(values[0:2])
        box.append(values[2:])  
     else:
        print("problem with unknown region type - update _parse_DS9regionfile() ")
                
   except:
     print("problem reading end header region file ")   
   
   for k in range(5,len(f)):
     try:
        r = f[k].split("\n")[0]
        if chatter > 3:
           print("line# ",k,' line=',r)
        elif r == "\n":
           continue
        elif r.split("(")[0] == "circle" :
           signs.append("+")
           boxtype.append("circle")
           values = r.split("(")[1].split(")")[0].split(",")
           position.append(values[0:2])
           box.append(values[2:])
        elif r.split("(")[0] == "-circle" :
           signs.append("-")
           boxtype.append("circle")
           values = r.split("(")[1].split(")")[0].split(",")
           position.append(values[0:2])
           box.append(values[2:])       
        else:
           print("problem with unknown region type - update _parse_DS9regionfile() ")
                
     except:
        print("problem reading region record number = ",k)
        
   return (version,filename,epoch,wcs),(signs,boxtype,position,box)


def encircled_energy(uvotfilter, areapix):
   """ 
   Compute the encircled energy in a uvotfilter
   as compared that in the default 5" radius.
   
   Parameters
   ===========
   uvotfilter : one of ["wh","v","b","u","uvw1","uvm2","uvw2"]
      filer name
   areapix : float   
      constant describing the number of sub-pixels
      for computing the cps rate
      
   Output
   ======  
   ratio : float
      a number that the count rate needs to be *divided* by
      which represents the fraction of encircled energy in
      the circular area extended by areapix pixels.
      
   Notes
   =====
   This applies solely for point sources.    
   """
   import os
   caldb = os.getenv("CALDB")
   if uvotfilter == 'wh': uvotfilter = 'white'
   command="quzcif swift uvota - "+uvotfilter.upper()+\
           " REEF 2009-10-30 12:00:00 - > quzcif.out"
   print(command)          
   if not os.system(command):
      print("not " +command)       
   f = open("quzcif.out")
   reeffile, ext = f.read().split()
   ext = int(ext)
   f.close()
   os.system("rm -f quzcif.out")
   print(reeffile, ext)
   f = fits.getdata(reeffile,ext)
   r = f['radius'] # in arc sec
   E = f['reef'] 
   x = sqrt(old_div(areapix,pi))*0.502 # lookup radius
   f = interp1d(r,E)
   return f(x)
   
def polyfit_with_fixed_points(n, x, y, xf, yf): 
    """  compute a polynomial fit with fixed points 
    
    parameters
    ----------
    n : int
       order of polynomial
    x,y : array like
       data point coordinates
    xf,yf : array like
       fixed data point coordinates
       
    reference
    ---------
       Stackoverflow.com/users/110026/jaime
       with added typing x,y,xf,yf
    
    Example
    -------
    n, d, f = 50, 8, 3
    x = np.random.rand(n)
    xf = np.random.rand(f)
    poly = np.polynomial.Polynomial(np.random.rand(d + 1))
    y = poly(x) + np.random.rand(n) - 0.5
    yf = np.random.uniform(np.min(y), np.max(y), size=(f,))
    params = polyfit_with_fixed_points(d, x , y, xf, yf)
    poly = np.polynomial.Polynomial(params)
    xx = np.linspace(0, 1, 1000)
    plt.plot(x, y, 'bo')
    plt.plot(xf, yf, 'ro')
    plt.plot(xx, poly(xx), '-')
    plt.show()
    """
    import numpy as np
    x = np.asarray(x)
    y = np.asarray(y)
    xf = np.asarray(xf,dtype=float)
    yf = np.asarray(yf,dtype=float)
    mat = np.empty((n + 1 + len(xf),) * 2)
    vec = np.empty((n + 1 + len(xf),))
    x_n = x**np.arange(2 * n + 1)[:, None]
    yx_n = np.sum(x_n[:n + 1] * y, axis=1)
    x_n = np.sum(x_n, axis=1)
    idx = np.arange(n + 1) + np.arange(n + 1)[:, None]
    mat[:n + 1, :n + 1] = np.take(x_n, idx)
    xf_n = xf**np.arange(n + 1)[:, None]
    mat[:n + 1, n + 1:] = old_div(xf_n, 2)
    mat[n + 1:, :n + 1] = xf_n.T
    mat[n + 1:, n + 1:] = 0
    vec[:n + 1] = yx_n
    vec[n + 1:] = yf
    params = np.linalg.solve(mat, vec)
    return params[:n + 1]
    
    
