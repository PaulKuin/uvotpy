#!/sciencesoft/Ureka/variants/common/bin/python
# -*- coding: iso-8859-15 -*-
#
# This software was written by N.P.M. Kuin (Paul Kuin) 
# Copyright N.P.M. Kuin 
# All rights reserved
# This software is licenced under a 3-clause BSD style license
# 
#Redistribution and use in source and binary forms, with or without 
#modification, are permitted provided that the following conditions are met:
#
#Redistributions of source code must retain the above copyright notice, 
#this list of conditions and the following disclaimer.
#
#Redistributions in binary form must reproduce the above copyright notice, 
#this list of conditions and the following disclaimer in the documentation 
#and/or other materials provided with the distribution.
#
#Neither the name of the University College London nor the names 
#of the code contributors may be used to endorse or promote products 
#derived from this software without specific prior written permission.
#
#THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" 
#AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, 
#THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR 
#PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR 
#CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, 
#EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, 
#PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; 
#OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, 
#WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR 
#OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF 
#ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
'''
This software does the conversion from detector to raw coordinates 
for the Swift UVOT.   

Input parameters for the script are explained in the help:

   convert_raw2det.py --help 
   
To make a call from a python module, 
import convert_raw2det
rawx,rawy = convert_raw2det.get_raw_from_radec(ra,dec,skyfile,ext,chatter):  

used as a script, output to stdout consists of the original position, 
and the position corrected for the distortion.

The initialisation is slow, so using a parameter file for larger numbers
of data is faster.

History
--------
2014-09-01 NPMK based on calibration routines
2014-09-03 initial tests passed 
2014-09-09 improved Bspline coefficients 
2014-09-11 added a final 0.5 deg rotation to get better DET->RAW positions.
'''

import numpy as np
import os
import sys
import optparse
try:
   from scipy.interpolate import bisplev
except: 
   raise RuntimeError("ERROR: You need to install the scipy package")  

try:
   import astropy
   ver = astropy.__version__.split('.')
   if float(ver[0]) < 1: 
      sys.stderr.write( "WARNING: software was developed with Astropy version 1.0.dev9483\n")
except:
   raise RuntimeError("ERROR: You need to install the Astropy package")      

from astropy.io import ascii,fits
import astropy.coordinates as coord
import astropy.wcs as wcs
from astropy import units
import datetime

__version__ = "0.4"
status = 0

today_ = datetime.date.today()   
datestring = today_.isoformat()[0:4]+today_.isoformat()[5:7]+today_.isoformat()[8:10]

def get_tck(invert=True): 
    """ bi-cubic spline coefficients
    When invert = True  : det -> raw
    When invert = False : raw -> det 
    
    References to used Bisplines in Scipy
    -------------------------------------
.. [1] Dierckx P.:An algorithm for surface fitting with spline functions
   Ima J. Numer. Anal. 1 (1981) 267-283.
.. [2] Dierckx P.:An algorithm for surface fitting with spline functions
   report tw50, Dept. Computer Science,K.U.Leuven, 1980.
.. [3] Dierckx P.:Curve and surface fitting with splines, Monographs on
   Numerical Analysis, Oxford University Press, 1993.
    """
    # to re-derive the coefficients, use distortfit.py
    # a somewhat smoother set of coefficents smo=200 in x,y 
    #�has a slightly worse fit to the input in places
    # using the optimum smoothing of the Dierkx Bspline algorithm 
    # gives errors of about 4 pixels. That may be the error 
    # as out fit has been forced to follow the input data 
    # more closely.  

    from numpy import array
    if invert:
        # tck_dx smooth-x = 60 
        # fit error x1[200:1800]< 0.77 pix
        # fit error x1[:] < 2.53
        tck_dx = [array([
            0.        ,     0.        ,     0.        ,     0.        ,
          307.93368307,   995.54486058,  1418.97609776,  1742.67666618,
         1969.7368227 ,  2048.        ,  2048.        ,  2048.        ,
         2048.        ]),
 array([    0.        ,     0.        ,     0.        ,     0.        ,
          336.03384424,   610.49538798,   844.43993309,  1553.25572659,
         2048.        ,  2048.        ,  2048.        ,  2048.        ]),
 array([-53.99988424, -62.53304605, -49.08373291, -44.04315159,
        -34.37695901, -31.95582208, -47.63560897, -42.04184033,
        -53.64076712, -52.33376051, -41.81555221, -32.78551535,
        -27.1825518 , -24.41081553, -33.17671004, -42.72242535,
        -30.53081004, -23.29908644, -22.99792199, -14.75100644,
        -14.30185881,  -7.97697641, -10.7075572 , -10.15588028,
         -9.66383095,  -9.66469105,  -7.01397018,  -6.4893321 ,
         -4.95868711,  -0.3288418 ,   1.03690717,   0.82486819,
         11.67869925,  12.76240807,   8.9726494 ,  10.60400216,
         12.05516025,  10.11306941,  13.32625997,  15.11272389,
         31.40835434,  24.54425806,  25.60597645,  18.20693871,
         19.08001675,  16.37283689,  24.21539633,  26.58202983,
         50.83500738,  48.19139455,  34.83661537,  33.23971153,
         27.20671697,  25.62628621,  34.57384977,  38.99563895,
         47.60711536,  50.67148291,  48.8341474 ,  38.43286435,
         33.53609309,  29.48529047,  43.69719316,  42.3087273 ,
         45.76294249,  50.21210482,  53.49688212,  39.74521371,
         35.56821708,  30.58147343,  46.47254837,  42.89338064]),
 3,
 3]
        # tck_dy for smooth-y = 110
        # fit error y1[200:1800]< 1.13 pix
        # fit error y1[:] < 2.42
        tck_dy = [array([
            0.        ,     0.        ,     0.        ,     0.        ,
           97.30644532,   265.33943177,  1009.06725415,  1210.09980588,
         1393.74935126,  1729.3104187 ,  1966.61407844,  2048.        ,
         2048.        ,  2048.        ,  2048.        ]),
 array([    0.        ,     0.        ,     0.        ,     0.        ,
           55.10917959,   162.42875247,   377.49605285,   643.85467619,
          843.95541281,  1038.44724269,  1616.03498239,  1935.62643749,
         2048.        ,  2048.        ,  2048.        ,  2048.        ]),
 array([-45.97700902, -45.74102971, -47.04184945, -44.24858855,
        -27.10040671, -26.62917777, -19.0147577 , -21.01977369,
        -17.66908113,  -1.76933437,   0.94385892,   3.73385127,
        -42.14303251, -43.84922859, -49.09422608, -39.56086895,
        -26.40624213, -22.51340695, -19.48272878, -18.99002591,
        -16.34968953,  -4.66088356,   5.54973488,   8.0734774 ,
        -48.67876955, -48.48301367, -45.18354648, -31.98283962,
        -22.91794266, -18.30358866, -17.10117093, -18.03785869,
        -17.95929332,  -8.06337956,   3.66101392,   6.89482148,
        -27.74379438, -25.28917271, -19.6090298 , -14.14044173,
         -7.82409232,  -7.09725768,  -9.89836937, -11.73458943,
        -18.88953912, -12.95032335,  -8.11112903,  -5.72649427,
        -13.38439839, -12.02988661,  -9.31923843,  -2.93916958,
         -0.51978031,  -0.67929548,  -2.78867163,  -7.41670504,
        -10.11611551, -12.01180683,  -4.88263833,  -3.65077579,
         -2.20175539,  -1.11689409,   2.41897797,   7.19323204,
         11.30792466,   9.03060188,   7.49510165,   2.96210456,
         -2.30253635,  -2.67357085,   1.75357014,   3.20954877,
          1.08273093,   2.23826251,   6.86753088,  12.60319653,
         16.00500999,  16.36665918,  12.94252923,   8.53724293,
          2.13695114,   3.63973881,   9.47538706,  11.68509374,
          0.26759859,   2.94277993,   7.33093771,  14.95888297,
         20.33639267,  20.06766536,  19.63872472,  15.98708667,
          9.08508634,  12.93766481,  17.87835544,  19.73685612,
         -6.369083  ,  -4.16195834,   0.93263292,  13.58767956,
         19.28223925,  24.86026542,  21.86847669,  20.02040924,
         15.23441686,  22.79743445,  33.42367808,  35.73423013,
          1.87216059,   2.1774752 ,   2.97356246,   9.37623167,
         20.51050556,  23.32768811,  26.00695053,  22.74797168,
         19.84403337,  28.88990506,  39.70829573,  32.27731387,
          1.0289263 ,   2.43442255,   6.66728611,   8.29820233,
         20.00973831,  26.17904134,  30.46449401,  22.82461837,
         20.28428986,  32.1369143 ,  40.69826268,  34.55031575]),
 3,
 3]
    else:           
        tck_dx = [array([
            0.        ,     0.        ,     0.        ,     0.        ,
          1004.06710107,  1766.79890847,  2048.        ,  2048.        ,
          2048.        ,  2048.        ]),
  array([    0.        ,     0.        ,     0.        ,     0.        ,
           925.18677639,  2048.        ,  2048.        ,  2048.        ,
          2048.        ]),
  array([ 56.37661168,  45.55016725,  25.20939559,  33.92135879,
          44.94178639,  32.62101859,  22.65039897,  11.64014483,
          10.3972444 ,  15.26712107,   8.65403988,   4.56547882,
           7.80487865,  -2.62304408,  -2.52646789, -21.0803441 ,
         -14.10291407, -17.672413  , -13.95999177, -19.92596245,
         -47.55495646, -36.28498758, -23.89443912, -26.82738664,
         -38.29283108, -55.45900039, -45.69098767, -27.25588566,
         -33.41833297, -44.16436878]),
  3,
  3 ]
        tck_dy = [array([
            0.        ,     0.        ,     0.        ,     0.        ,
           982.16844804,  2048.        ,  2048.        ,  2048.        ,
          2048.        ]),
  array([    0.        ,     0.        ,     0.        ,     0.        ,
           675.91744082,  1695.83135879,  2048.        ,  2048.        ,
          2048.        ,  2048.        ]),
  array([ 49.61916793,  29.76662813,  14.2039948 ,  22.90222091,
           0.6163669 ,  -7.1411107 ,  30.69953324,  12.55115397,
           5.97645784,  24.13247827,  11.38267282,   2.75589496,
           0.24028354,  -6.27796237,  -2.192261  ,   8.96084647,
          11.03785524,   8.12426686,  -3.89623282, -23.42469887,
         -25.90449261,  -5.74819222, -11.26748564, -17.19066226,
           4.42176369, -13.70711787, -33.28765076, -13.75987181,
         -30.16803282, -37.61980086]),
  3,
  3 ]
    return tck_dx, tck_dy
    
def from_det_to_raw(detx,dety,invert=True):
    # if invert == False, then the detx,dety coordinates apply 
    # to RAW and rawx,rawy to DET
    if type(detx) == int: detx=[detx]
    if type(dety) == int: dety=[dety]
    if type(detx) == float: detx=list(detx)
    if type(dety) == float: dety=list(dety)
    detx = np.asarray(detx,dtype=float)
    dety = np.asarray(dety,dtype=float)
    if detx.ndim == 0:
       detx = np.asarray([detx])
       dety = np.asarray([dety])
    if invert: 
    # shift det coordinates before applying inverse distortion
       detx -= 77
       dety -= 77
    # if raw, they are in the right system     
    tck_dx, tck_dy =  get_tck(invert=invert)
    rawx = []
    rawy = []
    for _x,_y in zip(detx,dety):
       (_rawx, _rawy) = (_x + bisplev(_x, _y, tck_dx), 
                    _y + bisplev(_x, _y, tck_dy) )
       rawx.append(_rawx)
       rawy.append(_rawy)
    rawx = np.asarray(rawx,dtype=float)
    rawy = np.asarray(rawy,dtype=float)   
    if not invert:
    # (rawx,rawy are distortion corrected RAW on physical image system)
    #  convert to det coordinates so they can be used for further processing
       rawx += 77
       rawy += 77                   
    else:
       rawx,rawy = _rotvec(rawx-1023.5,rawy-1023.5,0.5)
       rawx = rawx+1023.5
       rawy = rawy+1023.5                   
    return rawx, rawy        

def radec2det(posJ2000,skyfile,ext,chatter, det_as_mm=False):                      
   if chatter > 0: sys.stderr.write( "from sky to det, ra,dec= %s\n"%(posJ2000))
   try:
       hdr = fits.getheader(skyfile,ext)
   except:
       raise RuntimeError("\nThe FITS header could not be found for %s[%s]\n"%(skyfile,ext))    
   w1 =wcs.WCS(header=hdr,)
   wD =wcs.WCS(header=hdr,key='D',)
   phys = w1.wcs_world2pix([[posJ2000.ra.deg,posJ2000.dec.deg]],0)
   xyposd = wD.wcs_pix2world(phys,0)
   xypos = xyposd/0.009075+np.array((1100.5,1100.5))
   if chatter > 0: 
      sys.stderr.write( "detector coordinates %s mm = %s pix\n"%(xyposd,xypos) )
   xypos = np.array(xypos)   # computation is for binx=1 not /hdr['binx']   
   xpos, ypos = xypos[:,0],xypos[:,1]
   if det_as_mm:
       return  xpos, ypos, xyposd[:,0], xyposd[:,1], 
   else:         
       return xpos, ypos

def radec2pos(ra,dec,chatter):
    try: 
        # default usage is to give the position in degrees 
        posJ2000 = coord.SkyCoord(ra=float(ra) * units.degree, 
                               dec=float(dec) * units.degree,
                               frame='icrs')
    except:
        try: 
            # if the position was given in sexagesimal units, this may work
            posJ2000 = coord.SkyCoord(ra,dec,'icrs')
        except:
            sys.stderr.write("\nError with input values RA=%s, DEC=%s \n"%(ra,dec))
            raise IOError("provide the coordinates in degrees, or ra as '00h42m00s', dec as '+41d12m00s'\n")     
    if chatter > 0: sys.stderr.write( "position used: %s\n"%(posJ2000)) 
    return posJ2000
    
def get_ext(file,chatter):      
    # we need the world coordinate transformation which are in the UVOT sky file headers
    # find the fits file extension 
    s1 = file.rsplit('+')
    s2 = file.rsplit('[')
    if (len(s1) == 1) & (len(s2) == 1): 
        ext = 1
        if chatter > 0: 
            sys.stderr.write( "assuming the first extension contains the sky image concerned.\n")
        skyfile = s1[0]
    elif (len(s1) == 1) & (len(s2) == 2):
        ext = s2[1].split(']')[1]
        skyfile = s2[0]
    elif (len(s1) == 2) & (len(s2) == 1):
        ext = s1[1]
        skyfile = s1[0]
    else:
       raise IOError("file extension could not be determined from skyfile")
    try:
        ext = int(ext)
    except:
        pass
    if chatter > 1: 
        sys.stderr.write("get_ext: %s + %s\n"%(skyfile,ext))    
    return skyfile, ext                  

def _rotvec(X, Y, theta):
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

def finish(fromsky,invert,chatter,posJ2000=None,ext=None,xpos=None,ypos=None,
       det_as_mm=False,skyfile=None,returns=False):
    if chatter > 1:
        sys.stderr.write("convert_sky2det2raw version="+__version__+'\n')
                      
    if fromsky & invert:
        if det_as_mm: 
            xpos, ypos, x_mm,y_mm = radec2det(posJ2000,skyfile,ext,chatter,det_as_mm=det_as_mm,)
        else:
            xpos, ypos = radec2det(posJ2000,skyfile,ext,chatter)
    
    try:  # needed for det2raw is only initialised when module called from the command line
       if (not det_as_mm) & det2raw:
           x_mm, y_mm = (xpos-1100.5)*0.009075, (ypos-1100.5)*0.009075         
    except:
       pass
       
    if chatter > 1: 
        sys.stderr.write( "xpos %s, "%(xpos))
        sys.stderr.write( "ypos %s\n"%(ypos))
    for x,y in zip(xpos,ypos):
        rx,ry= from_det_to_raw(x,y,invert=invert)
        if returns: 
           return rx,ry
        elif det2raw:
           sys.stdout.write( "%8.2f,%8.2f,  %8.2f,%8.2f, %9.5f, %9.5f \n"%(x, y, rx,ry, x_mm, y_mm) )
        else:   
           sys.stdout.write( "%8.2f,%8.2f,  %8.2f,%8.2f\n"%(x, y, rx,ry) )

def get_raw_from_radec(ra,dec,skyfile,ext,chatter):
    """
    Provide the position on the RAW uvot image given sky position.
    
    parameters
    ----------
    ra,dec : float, array
       The sky position RA and Dec in units of degrees. 
    skyfile : path
       The full path of the sky file 
    ext : fits extension ID
       The fits extension, either a number of the 
       value of the EXTNAME fits header keyword
       
    output
    ------
    rawx,rawy : float
       pixel position in RAW image (physical image coordinates)
        
    """
    posJ2000 = radec2pos(ra, dec, chatter)    
    skyfile, ext = get_ext(skyfile+"+"+str(ext),chatter)
    if os.access(skyfile,os.F_OK):
        sys.stderr.write(
    "using world coordinate transformation from "+skyfile+"["+str(ext)+"]\n")
    else:
       raise IOError("file not found: "+skyfile)
    return finish(True,True,chatter,posJ2000=posJ2000,ext=ext,
                  skyfile=skyfile,returns=True)                  
    
if __name__ == '__main__':
   #in case of called from the OS

   if status == 0:
      usage = "usage: %prog [options] -d ra dec"

      epilog = '''
Either give the position in just pixel coordinates, 
or give the sky position in ra,dec (J2000, degrees) and also 
supply the file path and fits extension to read the WCS 
coordinate transformations from.  
      
The default is to convert from detector to raw coordinates. 
If the other route is desired (raw->det), set -det2raw 
      ''' 
      parser = optparse.OptionParser(usage=usage,epilog=epilog)
      parser.disable_interspersed_args()
      
      # main options

      parser.add_option("", "--det2raw", dest = "det2raw", action="store_true",
                  help = "convert detector coordinate to raw coordinate [default]",default=True)

      parser.add_option("", "--raw2det", dest = "det2raw", action="store_false",
                  help = "convert raw coordinate to detector coordinate",)

      parser.add_option("", "--xypos", dest = "radec", action="store_false",
                  help = "input arguments are the X Y position in pixels",
                  metavar="[X,Y]",)

      parser.add_option("", "--radec", dest = "radec", action="store_true",
                  help = "the input arguments are ra and dec (J2000) positions in deg [default]",
                  default = True)

      parser.add_option("-p", "--par", dest = "parameterfile",
                  help = "filename containing comma-separated records of input [ra, dec, skyfilepath, ext, x, y, ]",
                  default = None)
                  
      parser.add_option("-f", "--file", dest = "skyfile",
                  help = "full filename path plus extension of the sky file",
                  default = None)

      parser.add_option("-d", "--data", dest = "data", nargs=2,
                  help = "the positional data",
                  default = None)

      parser.add_option("", "--chatter", dest = "chatter",
                  help = "verbosity [default: %default]",
                  default = 0)
                  
                  
   (options, args) = parser.parse_args()
   
   chatter = options.chatter
   if options.chatter > 0: 
       sys.stderr.write( "options: %s\n"%( options ))
       sys.stderr.write( "other args: %s\n"%(args))
            

   if options.parameterfile != None:
       pf = open(options.parameterfile)
       pr = pf.readlines()
       pf.close()
       for line in pr:
          pi = line.split(',')
          n = len(pi)
          # from sky
          if n < 4: 
             sys.stderr.write("the parameter file data record :\n%s\n cannot be used\n"%
                 (line))
          elif (n >= 4) & options.det2raw:               
             if (pi[0] != "") & (pi[1] != "") & (pi[2] != "") & (pi[3] != ""):
                ra, dec, skyfile,ext = pi[0], pi[1], pi[2], pi[3]
                if chatter > 1:
                    sys.stderr.write("input record data ra=%s  dec=%s file=%s ext=%s\n"%(ra,dec,skyfile,ext))
                posJ2000 = radec2pos(ra, dec, chatter)    
                skyfile, ext = get_ext(skyfile+"+"+ext,chatter)
                if os.access(skyfile,os.F_OK):
                    sys.stderr.write(
                    "using world coordinate transformation from "+skyfile+"["+str(ext)+"]\n")
                else:
                    raise IOError("file not found: "+skyfile)
                options.radec = True    
                finish(True,True,chatter,posJ2000=posJ2000,ext=ext)              
          elif n >= 6:
             if (pi[4] != "") & (pi[5] != ""):
                 xpos, ypos = [float(pi[4])], [float(pi[5])]
                 if chatter > 0:
                     sys.stderr.write("positions read = %s, %s "%(xpos,ypos))
                 options.radec = False
                 finish(False,options.det2raw,chatter,xpos=xpos,ypos=ypos)  
          else: 
              sys.stderr.write("error, in reading parameter file record: %s"%(pi))                 
       exit(code=0)
              
   if len(options.data) == 0:
       sys.stderr.write( "no values for ra and dec found\n")
       parser.print_help()
       parser.exit       
   elif len(options.data) == 2:    
       ra,dec = np.array(options.data,dtype=float)
   elif len(options.data)/2*2 == len(args):
       data = np.array(options.data,dtype=float).reshape(len(options.data)/2,2)
       ra  = data[:,0]
       dec = data[:,1] 
                         
   if not options.radec: 
       xpos,ypos = ra,dec
   else: xpos,ypos = None, None       

   if options.radec:
      posJ2000 = radec2pos(ra, dec, chatter)
      skyfile, ext = get_ext(options.skyfile,chatter)     
      # check the file is actually present
      if os.access(skyfile,os.F_OK):
         sys.stderr.write("using world coordinate transformation from "+skyfile+"["+str(ext)+"]\n")
      else:
         raise IOError("file not found: "+skyfile)
         
         
   finish(options.radec,options.det2raw,options.chatter,
          posJ2000=posJ2000,skyfile=skyfile,ext=ext,xpos=xpos,ypos=ypos)                 

########## (c) 2014             
