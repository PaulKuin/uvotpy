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

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
# Developed by N.P.M. Kuin (MSSL/UCL) 
# uvotpy module
# (c) 2009-2015, see Licence  

# 2015-08-04 start with copy of "getCalData()"
# 2019-02-26 continue coding/testing 
 
from builtins import str
from builtins import range
from builtins import object
import os, sys
import numpy as np
from astropy.io import ascii,fits
from astropy import units


__version__ = '0.2 20200905'  

typeNone = type(None)


class Caldb(object):

    def __init__(self, grismmode=None, wheelpos=None, msg="", 
         use_caldb=False,
         chatter=0 ):

        self.msg = msg
        self.phipos = None
        self.use_caldb = use_caldb
        self.calfilepath = None
        self.ea_extname = None
        self.ea_model = None
        if grismmode == None: 
           if wheelpos == None: 
              raise IOError ("Give either grism mode or filter wheel position.")
           else: 
              self.wheelpos = wheelpos
        else: 
            try:
               self.wheelpos = {"uv nominal":200,"uv clocked":160,
              "visible nominal":1000,"visible clocked":955}[grismmode.lower()]  
            except:
               raise IOError ("the grismmode parameter needs to be one of: "\
               "'uv nominal','uv clocked','visible nominal','visible clocked' ")        
        self.chatter=chatter        
                        
    def locate_caldata(self,what="wave",spectralorder=1):
        """
        if self.use_caldb and the calfile needed is present, then 
        use the caldb wavecal file. If not, then use the uvotpy/calfiles
        directory for retrieving the wavecal file.
        
        Parameters
        ==========
        what: str ["wave","flux"]
           "wave": returns the wavecal file path
           "flux": returns effective area file path, 
                   extension name 
                   model ID
                   
        spectralorder: int [1,2],optional           
           For retrieval of path to effective area file.
           
        Returns
        =======
        calibration file information
        
        WARNING: the earliest calibration files were incomplete and the methods 
        in WaveCal and FluxCal may not work on those.         
        """
        dtyp = {'wave':"WAVECAL",'flux':"SPECRESP"}[what.lower()]
        grismname = {160:"UGRISM",200:"UGRISM",955:"VGRISM",1000:"VGRISM"}[self.wheelpos]
        if self.use_caldb:  # test and update 
            # first try to find the calfile in the CALDB
            CALDB = os.getenv('CALDB')
            self.use_caldb = CALDB != '' 
            
        if self.use_caldb:            
            command="quzcif swift uvota - %s %s now now  wheelpos.eq.%s > quzcif.out"% \
                 (grismname,dtyp,str(self.wheelpos))
            status = os.system(command)
            f = open("quzcif.out")
            records = f.readlines()
            f.close()
            os.system("rm -f quzcif.out")
            arf, extens = records[0].split()  
            if what.lower() == "wave":
               self.calfilepath = CALDB + "/data/swift/uvota/bcf/grism/"+arf   
            elif what.lower() == "flux":
               self.calfilepath = CALDB + "/data/swift/uvota/cpf/arf/"+arf
            else: 
                raise IOError("Argument 'what' is not 'wave' nor 'flux'!")
                
        else: # go read from the uvotpy distribution
        
            UVOTPY = os.getenv("UVOTPY")
            if UVOTPY == "":
               raise RuntimeError("Fatal error: environment variable UVOTPY was not found.")
            if what.lower() == "wave":   
                arf = {160:"swugu0160wcal20041120v002.fits",
                       200:"swugu0200wcal20041120v001.fits",
                       955:"swugv0955wcal20041120v001.fits",
                      1000:"swugv1000wcal20041120v001.fits"}   
                self.calfilepath= UVOTPY+"/calfiles/"+arf[self.wheelpos]
                
            elif what.lower() == "flux":
               # find here the "latest" version of the calibration files has been hardcoded    
               # latest update:   
               if spectralorder == 1: 
                    if self.wheelpos == 200:          
                         calfile = 'swugu0200_20041120v105.arf'
                         self.ea_extname = "SPECRESPUGRISM200"
                         self.ea_model   = "ZEMAXMODEL_200"
                    elif self.wheelpos == 160:
                         calfile = 'swugu0160_20041120v105.arf'
                         self.ea_extname = "SPECRESPUGRISM160"
                         self.ea_model   = "ZEMAXMODEL_160"
                    elif self.wheelpos == 955: 
                         calfile = 'swugv0955_20041120v104.arf'
                         self.ea_extname = "SPECRESPVGRISM0955"
                         self.ea_model   = "ZEMAXMODEL_955"
                    elif self.wheelpos == 1000: 
                         calfile = 'swugv1000_20041120v105.arf'
                         self.ea_extname = "SPECRESPVGRISM1000"
                         self.ea_model   = "ZEMAXMODEL_1000"
                    else:   
                         raise RuntimeError( "FATAL: [uvotio.readFluxCalFile] invalid filterwheel position encoded" )
             
               elif spectralorder == 2:          
                    # HACK: force second order to 'nearest' option 2015-06-30 
                    if self.wheelpos == 200:          
                        calfile =  'swugu0200_2_20041120v999.arf' #'swugu0200_20041120v105.arf'
                        self.ea_extname = "SPECRESP0160GRISM2NDORDER"
                        self.ea_model   = ""
                    elif self.wheelpos == 160:
                        calfile = 'swugu0160_2_20041120v999.arf'  #'swugu0160_20041120v105.arf'
                        self.ea_extname = "SPECRESP0160GRISM2NDORDER"
                        self.ea_model   = ""
                    elif self.wheelpos == 955: 
                        calfile = 'swugv0955_2_20041120v999.arf' #'swugv0955_20041120v104.arf'
                        self.ea_extname = "SPECRESPVGRISM955"
                        self.ea_model   = ""
                    elif self.wheelpos == 1000: 
                        calfile = 'swugv1000_2_20041120v999.arf'  #swugv1000_20041120v105.arf'
                        self.ea_extname = "SPECRESPVGRISM1000"
                        self.ea_model   = ""
                    else:   
                        raise RuntimeError( "FATAL: [uvotio.readFluxCalFile] invalid filterwheel position encoded" )

               else:     
                   raise RuntimeError("spectral order not 1 or 2 - no effective area available")
                
               self.calfilepath= UVOTPY+"/calfiles/"+calfile
                
        if what.lower() == "wave":                    
            self.msg += "wavecal file : %s\n"%(self.calfilepath.split('/')[-1])                       
            return self.calfilepath
        elif what.lower() == "flux":
            self.msg += "effective area file : %s\n"%(self.calfilepath.split('/')[-1])                       
            return self.calfilepath, self.ea_extname, self.ea_model
            
class WaveCal(Caldb):

    """
    Wavelength calibration data provides: 
    angle of spectrum, 
    anchor position spectrum, 
    dispersion coefficients first, second order
    
    Parameters
    ==========
    
    offsetpos : list of length 2, optional
       X, Y coordinate for offset
    wheelpos : int [160,200,955,1000]
       filter wheel position or give grismmode, optionally
    grismmode : str ["uv nominal","uv clocked","visible nominal","visible clocked"]
       the grism id and clocking 
    offsetdelta : list of length 2, optional
       dX, dY for offset from the boresight in pix
    use_caldb : bool, optional
       if False, use the calibration file from the UVOTPY distribution, 
       otherwise, use the Swift CALDB   
    chatter : int , optional
       verbosity
    
    mode: str['interp2d','bisplines','bilinear']
       interpolation method. Default is cubic bisplines
    
    Methods
    =======
    anchor() returns anchor positions of first and second order
    theta() returns the angle of the spectrum on the detector at the 1st order anchor
    disp() 
         
    """

    def __init__(self, grismmode=None, wheelpos=None, msg="", 
         use_caldb=False,
         mode = 'bilinear', #'bisplines',
         _flimit=0.19, # do not change unless you know what you do
         _fail_or_report=True,
         chatter=0 ):

        self.N1 = 0
        self.niter = 0
        self.msg = msg
        self._flimit = _flimit
        #if fail has been unset, then returns "None" values when no solution can be found.
        self.fail = _fail_or_report == True 
        self.phipos = None
        self.use_caldb = use_caldb
        self.status = 0 # good
        self.calfilepath = None
        self.pixelscale = 0.54 # arcsec per pixel on grism (used in my calibration; Wayne used 0.56)
        self.subpixsize = 0.0001394444 # 0.502 arcsec/pix on lenticular/ 3600 to degrees 
        if grismmode == None: 
           if wheelpos == None: 
              raise IOError ("Give either grism mode or filter wheel position.")
           else: 
              self.wheelpos = wheelpos
        else: 
            try:
               self.wheelpos = {"uv nominal":200,"uv clocked":160,
              "visible nominal":1000,"visible clocked":955}[grismmode.lower()]  
            except:
               raise IOError ("the grismmode parameter needs to be one of: "\
               "'uv nominal','uv clocked','visible nominal','visible clocked' ")        
        self.bsangle={  # define the angle for the spectrum at the boresight location
            955:{'theta':142.5,'name':'V grism Clocked'},
           1000:{'theta':150.1,'name':'V grism Nominal'},
            160:{'theta':147.4,'name':'U grism Clocked'},
            200:{'theta':153.4,'name':'U grism Nominal'}
                }[self.wheelpos]
        self.field = None        
        self.xp1 = None  # xdet coordinate first order anchor
        self.yp1 = None  # ydet ...
        self.thetalist = None # angle spectrum on detector at anchor (counter clockwise) grid      
        self.xp2 = None  # xdet coordinate second order anchor
        self.yp2 = None  # ydet ...
        self.coef1list = None # 1st order list of dispersion coefficient and length polynomial grid
        self.coef2list = None # 2nd order ...
        self.coef1 = None # 1st order list of dispersion polynomial coefficients at offset position  
        self.coef2 = None # 2nd order ...
        self.anchor1 = None # anchor 1st order interpolated for offset position
        self.ancho2 = None # anchor 2nd order interpolated for offset position
        self.thetavalue = None # angle spectrum on detector at anchor (counter clockwise)
        self.mode = mode       
        self.offsetdelta = [0.,0.]
        self.chatter=chatter        
        self._locate_calfilepath()
        self._read_wavecalfile()

    def _locate_calfilepath(self,):
        x = Caldb(wheelpos=self.wheelpos, msg=self.msg, 
              use_caldb=self.use_caldb, chatter=self.chatter) 
        self.calfilepath = x.locate_caldata(what='wave')

    def anchor(self,sporder=None, offsetdelta=[0,0],):
        """
        returns first and second order anchor position in detector coordinates
        
        Parameters
        ==========
        sporder: int [1,2]
           returns for the first or second order only
           if fail unset, then returns None values when no solution can be found.
        offsetdelta : astropy Quantity, list of length 2, optional
           delta-X, delta-Y coordinate for offset (in pixels)
           
        """
        if hasattr(offsetdelta,'to'):
            self.offsetdelta = offsetdelta.to(units.arcsec).value*self.pixelscale
        else:    
            self.offsetdelta = offsetdelta  # offset from boresight 
        if not np.isscalar(self.offsetdelta[0]):
            if len(np.asarray(self.offsetdelta[0]).shape) > 1:
                raise IOError("calfiles.WaveCal.disp can only be called for one offset at a time.")    
        #       [delta X, delta Y] (subpixels) in detector coordinates  
        self._interpolate_anchor()
        #if status == 0:
        if sporder == 1: return self.anchor1
        elif sporder == 2: return self.anchor2
        else: return [self.anchor1],[self.anchor2] 
        

    def theta(self,offsetdelta=[0,0],):
        """
        returns the angle of the spectrum on the detector at the anchor1 position
        (astropy quantity)
        
        Parameters
        ==========
        offsetdelta : astropy Quantity, list of length 2, optional
           delta-X, delta-Y coordinate for offset from anchor/pointing (in pixels)
        """
        if hasattr(offsetdelta,'to'):
            self.offsetdelta = offsetdelta.to(units.arcsec).value*self.pixelscale
        else:    
            self.offsetdelta = offsetdelta  # offset from boresight (value)
        if not np.isscalar(self.offsetdelta[0]):
            if len(np.asarray(self.offsetdelta[0]).shape)  > 1:
                raise IOError("calfiles.WaveCal.theta can only be called for one offset at a time.")    
        #       [delta X, delta Y] (subpixels) in lent detector coordinates  
        self._interpolate_theta()
        return self.thetavalue * units.deg
        
    def disp(self,sporder=None,offsetdelta=[0,0]):
        """
        returns the lists with polynomial coefficients of the dispersion for  
        the first and second order
        
        Parameters
        ==========
        sporder: int [1,2]
           returns for the first or second order only
        offsetdelta : astropy Quantity, list of length 2, optional
           delta-X, delta-Y coordinate for offset (in pixels)
        
        """
        if hasattr(offsetdelta,'to'):
            self.offsetdelta = offsetdelta.to(unitd.arcsec).value*self.pixelscale
        else:    
            self.offsetdelta = offsetdelta  # offset from boresight 
        if not np.isscalar(self.offsetdelta[0]):
            if len(np.asarray(self.offsetdelta[0]).shape)  > 1:
                raise IOError("calfiles.WaveCal.disp can only be called for one offset at a time.")    
        #       [delta X, delta Y] (subpixels) in detector coordinates  
        self._interpolate_dispersion()
        #if status == 0:
        if sporder == 1: return self.coef1
        elif sporder == 2: return self.coef2
        else: return self.coef1, self.coef2
   

    def _read_wavecalfile(self,): 
         _flimit = self._flimit
         cal = fits.open(self.calfilepath)
         if self.chatter > 0: 
             print("opening the wavelength calibration file: %s"%(self.calfilepath))
         if self.chatter > 1: 
             print(cal.info())
         # put the data into more readily usable 2-D variables
         hdr0 = cal[0].header
         hdr1 = cal[1].header
         data = cal[1].data
         xf = data['PHI_X']
         self.N1 = N1 = int(np.sqrt( len(xf) )) 
         if N1**2 != len(xf): 
             raise RuntimeError("waveCal data: calfile array not square" )
         if self.chatter > 2: 
             print("GetCalData: input array size on detector is %i in x, %i in y"%(N1,N1))   
         xf = data['PHI_X'].reshape(N1,N1)
         yf = data['PHI_Y'].reshape(N1,N1)
         self.field = [xf,yf]
         #  first order anchor positions and theta angle grid
         self.xp1 = data['DETX1ANK'].reshape(N1,N1)
         self.yp1 = data['DETY1ANK'].reshape(N1,N1)
         self.thetalist  = 180.0 - data['SP1SLOPE'].reshape(N1,N1)
         c10 = data['DISP1_0'].reshape(N1,N1)
         c11 = data['DISP1_1'].reshape(N1,N1)
         c12 = data['DISP1_2'].reshape(N1,N1)
         c13 = data['DISP1_3'].reshape(N1,N1)
         c1n = data['DISP1_N'].reshape(N1,N1)
         #  second order 
         self.xp2 = data['DETX2ANK'].reshape(N1,N1)
         self.yp2 = data['DETY2ANK'].reshape(N1,N1) 
         c20 = data['DISP2_0'].reshape(N1,N1)
         c21 = data['DISP2_1'].reshape(N1,N1)
         c22 = data['DISP2_2'].reshape(N1,N1)
         c2n = data['DISP2_N'].reshape(N1,N1)
         if self.wheelpos == 955:
         #  first  order dispersion
            c14 = np.zeros(N1*N1).reshape(N1,N1)
         else:
         #  first  order dispersion
            c14 = data['disp1_4'].reshape(N1,N1)
         self.coef1list = [c10,c11,c12,c13,c14,c1n]
         self.coef2list = [c20,c21,c22,c2n]
         cal.close()
         
    def message(self,):
        return self.msg 
        
    def caldata(self,):  
        # give the binary table data       
        cal = fits.open(self.calfilepath)
        return cal[1].data

    def _offset2phi(self,):
        # input is the offset of the target from the anchor in the lenticular filter
        # offset in number of subpixels ; 
        # phipos in  units of degrees
        dx,dy = self.offsetdelta
        self.phipos = dx*self.subpixsize, dy*self.subpixsize 
        return self.phipos 
        
    def _phi2lfilter_offset(self,): 
        # iinnverse of _offset2phi
        x,y = self.field    
        x = x/self.subpixsize
        y = y/self.subpixsize
        return x, y
        
    def _interpolate_dispersion(self,): 
        """ 
        given the offset in pixels, get the field coordinate 
        then check if the offset is not too large
        if outside wavecal area: extrapolate; select the useful values only 
        
        """
        from scipy import interpolate
        import numpy as np
        #
        _flimit = self._flimit
        msg = self.msg
        N1 = self.N1
        rx, ry = self._offset2phi() # assume rx as well as ry are floats.   
        xf, yf = self.field   # array of field coordinates
        xp1, yp1 = self.xp1, self.yp1
        #
        #  test if offset is within the field array boundaries
        #
        xfp = xf[0,:]
        yfp = yf[:,0]
        if ((rx < np.min(xfp)) ^ (rx > max(xfp))):
           inXfp = False
        else:
           inXfp = True
        if ((ry < np.min(yfp)) ^ (ry > max(yfp))):
           inYfp = False
        else:
           inYfp = True         
        #
        #    lower corner (ix,iy)
        # 
        if inXfp :
           ix  = np.max( np.where( rx >= xf[0,:] )[0] ) 
           ix_ = np.min( np.where( rx <= xf[0,:] )[0] ) 
        else:
           if rx < np.min(xfp): 
               ix = ix_ = 0
               if self.chatter > 0: 
                   print("WARNING: point has xfield lower than calfile provides")
           if rx > np.max(xfp): 
               ix = ix_ = N1-1   
               if self.chatter > 0:
                   print("WARNING: point has xfield higher than calfile provides")
        if inYfp :   
            iy  = np.max( np.where( ry >= yf[:,0] )[0] ) 
            iy_ = np.min( np.where( ry <= yf[:,0] )[0] ) 
        else:
            if ry < np.min(yfp): 
                iy = iy_ = 0
                if self.chatter > 0:
                    print ("WARNING: point has yfield lower than calfile provides")
            if ry > np.max(yfp): 
                iy = iy_ = 27   
                if self.chatter > 0:
                    print("WARNING: point has yfield higher than calfile provides")
        if inYfp & inXfp & (self.chatter > 3): 
           print('getCalData.                             rx,         ry,     Xank,        Yank ')
           print(ix, ix_, iy, iy_)
           print('getCalData. gridpoint 1 position: ', xf[iy_,ix_], yf[iy_,ix_], xp1[iy_,ix_], yp1[iy_,ix_])
           print('getCalData. gridpoint 2 position: ', xf[iy ,ix_], yf[iy ,ix_], xp1[iy ,ix_], yp1[iy ,ix_])
           print('getCalData. gridpoint 3 position: ', xf[iy ,ix ], yf[iy ,ix ], xp1[iy ,ix ], yp1[iy ,ix ])
           print('getCalData. gridpoint 4 position: ', xf[iy_,ix ], yf[iy_,ix ], xp1[iy_,ix ], yp1[iy_,ix ])  
        # retrieve the coeeficients that were read from the file    
        [c10,c11,c12,c13,c14,c1n] = self.coef1list
        [c20,c21,c22,c2n] = self.coef2list
        #
        #  first block deals with offset point outside wavecal area
        #  i.e., exception at outer grid edges: 
        #
        if ((ix == N1-1) ^ (iy == N1-1) ^ (ix_ == 0) ^ (iy_ == 0)):
           
          # select only coefficient with order 4 (or 3 for wheelpos=955)
          if self.chatter > 0:
              print("IMPORTANT:")
              print("\nanchor point is outside the calibration array: extrapolating dispersion") 
              msg += "WARNING: anchor point is outside the wavecal area: extrapolating dispersion"
          try: 
              if self.wheelpos == 955 :
              # first order solution
                  q4 = np.where( c1n.flatten() == 3 )
                  xf = xf.flatten()[q4]
                  yf = yf.flatten()[q4]
                  c10 = c10.flatten()[q4]
                  c11 = c11.flatten()[q4]
                  c12 = c12.flatten()[q4]
                  c13 = c13.flatten()[q4]
                  c14 = np.zeros(len(q4[0]))
                  c1n = c1n.flatten()[q4]
                  # second order solution only when at lower or right boundary
                  if (ix == N1-1) ^ (iy == 0):
                      q2 = np.where( c2n.flatten() == 2 )[0]
                      c20 = c20.flatten()[q2]
                      c21 = c21.flatten()[q2]
                      c22 = c22.flatten()[q2]
                      c2n = c2n.flatten()[q2]
                  else:
                      N2 = N1/2
                      c20 = np.zeros(N2)
                      c21 = np.zeros(N2)
                      c22 = np.zeros(N2)
                      c2n = np.zeros(N2)
      
              else: 
                  q4 = np.where( c1n.flatten() == 4 )
                  xf = xf.flatten()[q4]
                  yf = yf.flatten()[q4]
                  c10 = c10.flatten()[q4]
                  c11 = c11.flatten()[q4]
                  c12 = c12.flatten()[q4]
                  c13 = c13.flatten()[q4]
                  c14 = np.zeros(len(q4[0]))
                  c1n = c1n.flatten()[q4]
                  c20 = c20.flatten()[q4]
                  c21 = c21.flatten()[q4]
                  c22 = c22.flatten()[q4]
                  c2n = c2n.flatten()[q4]
     
              # find the dispersion
      
              tck = interpolate.bisplrep(xf, yf, c10,xb=-_flimit,xe=+_flimit,yb=-_flimit,ye=_flimit, kx=3,ky=3,s=None) 
              c10i = interpolate.bisplev(rx,ry, tck)
              tck = interpolate.bisplrep(xf, yf, c11,xb=-_flimit,xe=+_flimit,yb=-_flimit,ye=_flimit, kx=3,ky=3,s=None) 
              c11i = interpolate.bisplev(rx,ry, tck)
              tck = interpolate.bisplrep(xf, yf, c12,xb=-_flimit,xe=+_flimit,yb=-_flimit,ye=_flimit, kx=3,ky=3,s=None) 
              c12i = interpolate.bisplev(rx,ry, tck)
              tck = interpolate.bisplrep(xf, yf, c13,xb=-_flimit,xe=+_flimit,yb=-_flimit,ye=_flimit, kx=3,ky=3,s=None) 
              c13i = interpolate.bisplev(rx,ry, tck)
              tck = interpolate.bisplrep(xf, yf, c14,xb=-_flimit,xe=+_flimit,yb=-_flimit,ye=_flimit, kx=3,ky=3,s=None) 
              c14i = interpolate.bisplev(rx,ry, tck)
      
              if ((ix == N1-1) ^ (iy == 0)):
                  tck = interpolate.bisplrep(xf, yf, c20,xb=-_flimit,xe=+_flimit,yb=-_flimit,ye=_flimit, kx=3,ky=3,s=None) 
                  c20i = interpolate.bisplev(rx,ry, tck)
                  tck = interpolate.bisplrep(xf, yf, c21,xb=-_flimit,xe=+_flimit,yb=-_flimit,ye=_flimit, kx=3,ky=3,s=None) 
                  c21i = interpolate.bisplev(rx,ry, tck)
                  tck = interpolate.bisplrep(xf, yf, c22,xb=-_flimit,xe=+_flimit,yb=-_flimit,ye=_flimit, kx=3,ky=3,s=None) 
                  c22i = interpolate.bisplev(rx,ry, tck)
              else:
                  c20i = c21i = c22i = np.NaN 
              if self.chatter > 3: 
                  print('getCalData. bicubic extrapolation  ') 
                  print('getCalData. dispersion first  order = ',c10i,c11i,c12i,c13i,c14i)
              if c20i == NaN:
                  print(" no second order extracted ")
              else:   
                      print('getCalData. dispersion second order = ', c20i,c21i, c22i)
          except: 
              if self.fail:  
                  raise RuntimeError("interpolation of wavecal data failed - ABORTING") 
              else: 
                  self.status = -1      
                  c14i,c13i,c12i,c11i,c10i = None, None, None, None, None
                  c22i,c21i,c20i = None, None, None
                
        else: 
        # 
        #  offset point is within the wavecal area (common case)
        #
        #  reduce arrays to section surrounding point
        #  get interpolated quantities and pass them onto self
           if self.mode == "interp2d":
               kx = ky = 1
               s = None
               self.mode = 'bisplines'
           else: 
               kx = ky = 3
               s = None    
           if self.mode == 'bisplines':
               # compute the Bivariate-spline coefficients
               # kx = ky =  3 # cubic splines (smoothing) and =1 is linear
               task = 0 # find spline for given smoothing factor
               #  s = 0 # 0=spline goes through the given points
               # eps = 1.0e-6  (0 < eps < 1)
               m = N1*N1
               if self.chatter > 3: print('\n getCalData. splines ') 
               qx = qy = np.where( (np.isfinite(xf.reshape(m))) & (np.isfinite(yf.reshape(m)) ) )
             
               tck  = interpolate.bisplrep(xf.reshape(m)[qx], yf.reshape(m)[qy], c10.reshape(m),xb=-_flimit,xe=+_flimit,yb=-_flimit,ye=_flimit, kx=kx,ky=ky,s=s)
               c10i = interpolate.bisplev(rx,ry, tck)
               tck  = interpolate.bisplrep(xf.reshape(m)[qx], yf.reshape(m)[qy], c11.reshape(m),xb=-_flimit,xe=+_flimit,yb=-_flimit,ye=_flimit, kx=kx,ky=ky,s=s)
               c11i = interpolate.bisplev(rx,ry, tck)
               tck  = interpolate.bisplrep(xf.reshape(m)[qx], yf.reshape(m)[qy], c12.reshape(m),xb=-_flimit,xe=+_flimit,yb=-_flimit,ye=_flimit, kx=kx,ky=ky,s=s)
               c12i = interpolate.bisplev(rx,ry, tck)
               tck  = interpolate.bisplrep(xf.reshape(m)[qx], yf.reshape(m)[qy], c13.reshape(m),xb=-_flimit,xe=+_flimit,yb=-_flimit,ye=_flimit, kx=kx,ky=ky,s=s)
               c13i = interpolate.bisplev(rx,ry, tck)
               tck  = interpolate.bisplrep(xf.reshape(m)[qx], yf.reshape(m)[qy], c14.reshape(m),xb=-_flimit,xe=+_flimit,yb=-_flimit,ye=_flimit, kx=kx,ky=ky,s=s)
               c14i = interpolate.bisplev(rx,ry, tck)
               if self.chatter > 2: print('getCalData. dispersion first order = ',c10i,c11i,c12i,c13i,c14i)
               tck  = interpolate.bisplrep(xf.reshape(m)[qx], yf.reshape(m)[qy], c20.reshape(m),xb=-_flimit,xe=+_flimit,yb=-_flimit,ye=_flimit, kx=kx,ky=ky,s=s)
               c20i = interpolate.bisplev(rx,ry, tck)
               tck  = interpolate.bisplrep(xf.reshape(m)[qx], yf.reshape(m)[qy], c21.reshape(m),xb=-_flimit,xe=+_flimit,yb=-_flimit,ye=_flimit, kx=kx,ky=ky,s=s)
               c21i = interpolate.bisplev(rx,ry, tck)
               tck  = interpolate.bisplrep(xf.reshape(m)[qx], yf.reshape(m)[qy], c22.reshape(m),xb=-_flimit,xe=+_flimit,yb=-_flimit,ye=_flimit, kx=kx,ky=ky,s=s)
               c22i = interpolate.bisplev(rx,ry, tck)
               if self.chatter > 2: print('getCalData. dispersion second order = ', c20i,c21i, c22i)
           #
           elif self.mode == 'bilinear':
               c10i = self.bilinear( rx, ry, xf[0,:].squeeze(), yf[:,0].squeeze(), c10 )
               c11i = self.bilinear( rx, ry, xf[0,:].squeeze(), yf[:,0].squeeze(), c11 )
               c12i = self.bilinear( rx, ry, xf[0,:].squeeze(), yf[:,0].squeeze(), c12 )
               c13i = self.bilinear( rx, ry, xf[0,:].squeeze(), yf[:,0].squeeze(), c13 )
               c14i = self.bilinear( rx, ry, xf[0,:].squeeze(), yf[:,0].squeeze(), c14 )
               c20i = self.bilinear( rx, ry, xf[0,:].squeeze(), yf[:,0].squeeze(), c20 )
               c21i = self.bilinear( rx, ry, xf[0,:].squeeze(), yf[:,0].squeeze(), c21 )
               c22i = self.bilinear( rx, ry, xf[0,:].squeeze(), yf[:,0].squeeze(), c22 )
               if self.chatter > 1: 
                   print('getCalData. bilinear interpolation') 
                   print('getCalData. dispersion first  order = ',c10i,c11i,c12i,c13i,c14i)
                   print('getCalData. dispersion second order = ', c20i,c21i, c22i)
            
        self.coef1 = C_1 = np.array([c14i,c13i,c12i,c11i,c10i])
        self.coef2 = C_2 = np.array([c22i,c21i,c20i])
        self.msg = msg
        

    def _interpolate_theta(self,): 
        """ 
        given the offset in pixels, get the field coordinate 
        then check if the offset is not too large
        if outside wavecal area: extrapolate; select the useful values only         
        """
        from scipy import interpolate
        import numpy as np
        #
        _flimit = self._flimit
        N1 = self.N1
        rx, ry = self._offset2phi() # assume rx as well as ry are floats.  
        xf, yf = np.asarray(self.field[0]),  np.asarray(self.field[1]) # array of field coordinates
        xp1, yp1 = np.asarray(self.xp1), np.asarray(self.yp1)
        c1n = self.coef1list[5]
        th = self.thetalist
        msg = self.msg
        #
        #  test if offset is within the field array boundaries
        #
        xfp = xf[0,:]
        yfp = yf[:,0]
        if ((rx < np.min(xfp)) or (rx > np.max(xfp))):
           inXfp = False
        else:
           inXfp = True
        if ((ry < np.min(yfp)) or (ry > np.max(yfp))):
           inYfp = False
        else:
           inYfp = True         
        #
        #    lower corner (ix,iy)
        # 
        if inXfp :
           ix  = np.max( np.where( rx >= xf[0,:] )[0] ) 
           ix_ = np.min( np.where( rx <= xf[0,:] )[0] ) 
        else:
           if rx < np.min(xfp): 
               ix = ix_ = 0
               if self.chatter > 0: 
                   print(f"WARNING: point has xfield rx={rx} lower than calfile provides")
           if rx > np.max(xfp): 
               ix = ix_ = N1-1   
               if self.chatter > 0:
                   print(f"WARNING: point has xfield rx={rx} higher than calfile provides")
        if inYfp :   
            iy  = np.max( np.where( ry >= yf[:,0] )[0] ) 
            iy_ = np.min( np.where( ry <= yf[:,0] )[0] ) 
        else:
            if ry < np.min(yfp): 
                iy = iy_ = 0
                if self.chatter > 0:
                    print(f"WARNING: point has yfield ry={ry} lower than calfile provides")
            if ry > np.max(yfp): 
                iy = iy_ = 27   
                if self.chatter > 0:
                    print(f"WARNING: point has yfield ry={ry} higher than calfile provides")
        if inYfp & inXfp & (self.chatter > 3): 
           print('waveCal.   extrapolate               rx,         ry,     Xank,        Yank ')
           print(ix, ix_, iy, iy_)
           print('waveCal. gridpoint 1 position: ', xf[iy_,ix_], yf[iy_,ix_], xp1[iy_,ix_], yp1[iy_,ix_])
           print('waveCal. gridpoint 2 position: ', xf[iy ,ix_], yf[iy ,ix_], xp1[iy ,ix_], yp1[iy ,ix_])
           print('waveCal. gridpoint 3 position: ', xf[iy ,ix ], yf[iy ,ix ], xp1[iy ,ix ], yp1[iy ,ix ])
           print('waveCal. gridpoint 4 position: ', xf[iy_,ix ], yf[iy_,ix ], xp1[iy_,ix ], yp1[iy_,ix ])              
        #
        #  first block deals with offset point outside wavecal area
        #  i.e., exception at outer grid edges: 
        #
        if ((ix == N1-1) or (iy == N1-1) or (ix_ == 0) or (iy_ == 0)):
           
          # select only coefficient with order 4 (or 3 for wheelpos=955)
          if self.chatter > 0:
              print("WARNING: anchor point is outside the calibration array: extrapolating theta") 
          msg += "WARNING: anchor point is outside the wavecal area: extrapolating theta\n"
          try: 
              if self.wheelpos == 955 :
                  q4 = np.where( c1n.flatten() == 3 )
                  th  = self.thetalist.flatten()[q4]
              else: 
                  q4 = np.where( c1n.flatten() == 4 )
                  th  = self.thetalist.flatten()[q4]
      
              # find the angle  
              # 21/02/16 problem: bi-spline since xf, yf dim = 28x28, but th flattened. 
              # Does not work
              # change to nearest grid point
              #tck = interpolate.bisplrep(xf, yf, th,xb=-_flimit,xe=+_flimit,yb=-_flimit,ye=_flimit, kx=3,ky=3,s=None) 
              #thi = interpolate.bisplev(rx,ry, tck)
              xx9 = np.swapaxes(np.array([xf.flatten(),yf.flatten()]),1,0)
              yy9 = self.thetalist.flatten()
              thi = interpolate.griddata(xx9,yy9,np.array([rx,ry]),
                           method='nearest',fill_value=np.nan,rescale=False)
              if not np.isscalar(thi):
                  thi=thi[0]
              if self.chatter > 2: 
                  print('waveCal. nearest grid point ') 
                  print('waveCal. angle theta = %7.1f ' % (thi ))
          except:   
              raise RuntimeError(f"interpolation of wavecal data at ({rx},{ry}) failed - ABORTING")   
                
        else: 
        # 
        #  offset point is within the wavecal area (common case)
        #
        #  reduce arrays to section surrounding point
        #  get interpolated quantities and pass them onto self
           if self.mode == "interp2d":
               kx = ky = 1
               s = None
               self.mode = 'bisplines'
           else: 
               kx = ky = 3
               s = None   
           if self.mode == 'bisplines':
           # compute the Bivariate-spline coefficients
              m = N1*N1
              qx = qy = np.where( (np.isfinite(xf.reshape(m))) & (np.isfinite(yf.reshape(m)) ) )
              tck3 = interpolate.bisplrep(xf.reshape(m)[qx], yf.reshape(m)[qy], th.reshape(m),
                  xb=-_flimit,xe=+_flimit,yb=-_flimit,ye=_flimit, kx=kx,ky=ky,s=s)
              thi  = interpolate.bisplev(rx,ry, tck3)
           #
           elif self.mode == 'bilinear':
               #  reduce arrays to section surrounding point and interpolate
               thi  = self.bilinear( rx, ry, xf[0,:].squeeze(), yf[:,0].squeeze(), th  )
               if self.chatter > 2: 
                   print('waveCal. bilinear interpolation') 
                   print('waveCal. angle theta = %7.1f ' % (thi ))
        # only theta for the first order is available         
        self.thetavalue = thi
        self.msg = msg


    def _interpolate_anchor(self,): 
        """ anchor at offset
        given the offset in pixels, get the field coordinate 
        then check if the offset is not too large
        if outside wavecal area: extrapolate; select the useful values only 
                
        """
        from scipy import interpolate
        import numpy as np
        #
        _flimit = self._flimit
        msg = self.msg
        N1 = self.N1
        rx, ry = self._offset2phi() # assume rx as well as ry are floats.   
        xf, yf = self.field   # array of field coordinates
        xp1, yp1 = self.xp1, self.yp1
        xp2, yp2 = self.xp2, self.yp2
        if self.chatter > 3:
           print (f"interpolation mode = {self.mode}\n")
        #
        #  test if offset is within the field array boundaries
        #
        xfp = xf[0,:]
        yfp = yf[:,0]
        if ((rx < np.min(xfp)) ^ (rx > np.max(xfp))):
           inXfp = False
        else:
           inXfp = True
        if ((ry < np.min(yfp)) ^ (ry > np.max(yfp))):
           inYfp = False
        else:
           inYfp = True         
        #
        #    lower corner (ix,iy)
        # 
        if inXfp :
           ix  = np.max( np.where( rx >= xf[0,:] )[0] ) 
           ix_ = np.min( np.where( rx <= xf[0,:] )[0] ) 
        else:
           if rx < np.min(xfp): 
               ix = ix_ = 0
               if self.chatter > 0: 
                   print("WARNING: point has xfield lower than calfile provides")
           if rx > np.max(xfp): 
               ix = ix_ = N1-1   
               if self.chatter > 0:
                   print("WARNING: point has xfield higher than calfile provides")
        if inYfp :   
            iy  = np.max( np.where( ry >= yf[:,0] )[0] ) 
            iy_ = np.min( np.where( ry <= yf[:,0] )[0] ) 
        else:
            if ry < np.min(yfp): 
                iy = iy_ = 0
                if self.chatter > 0:
                    print("WARNING: point has yfield lower than calfile provides")
            if ry > np.max(yfp): 
                iy = iy_ = 27   
                if self.chatter > 1:
                     print("WARNING: point has yfield higher than calfile provides")
        if inYfp & inXfp & (self.chatter > 3): 
           print('getCalData.                             rx,         ry,     Xank,        Yank ')
           print(ix, ix_, iy, iy_)
           print('waveCal. gridpoint 1 position: ', xf[iy_,ix_], yf[iy_,ix_], xp1[iy_,ix_], yp1[iy_,ix_])
           print('waveCal. gridpoint 2 position: ', xf[iy ,ix_], yf[iy ,ix_], xp1[iy ,ix_], yp1[iy ,ix_])
           print('waveCal. gridpoint 3 position: ', xf[iy ,ix ], yf[iy ,ix ], xp1[iy ,ix ], yp1[iy ,ix ])
           print('waveCal. gridpoint 4 position: ', xf[iy_,ix ], yf[iy_,ix ], xp1[iy_,ix ], yp1[iy_,ix ])  
            
        #
        #  first block deals with offset point outside wavecal area
        #  i.e., exception at outer grid edges: 
        #
        if ((ix == N1-1) ^ (iy == N1-1) ^ (ix_ == 0) ^ (iy_ == 0)):
           
          # select only coefficient with order 4 (or 3 for wheelpos=955)
          if self.chatter > 0:
              print("IMPORTANT:")
              print("\nanchor point is outside the calibration array: extrapolating all data") 
          msg += "WARNING: anchor point is outside the wavecal area: extrapolating all data"
          try: 
              if self.wheelpos == 955 :
              # first order solution
                  q4 = np.where( c1n.flatten() == 3 )
                  xf = xf.flatten()[q4]
                  yf = yf.flatten()[q4]
                  xp1 = xp1.flatten()[q4]
                  yp1 = yp1.flatten()[q4]
                  # second order solution only when at lower or right boundary
                  if (ix == N1-1) ^ (iy == 0):
                      q2 = np.where( c2n.flatten() == 2 )[0]
                      xp2 = xp2.flatten()[q2]
                      yp2 = yp2.flatten()[q2] 
                  else:
                      N2 = N1/2
                      xp2 = np.zeros(N2) 
                      yp2 = np.zeros(N2) 
              else: 
                  q4 = np.where( c1n.flatten() == 4 )
                  xf = xf.flatten()[q4]
                  yf = yf.flatten()[q4]
                  xp1 = xp1.flatten()[q4]
                  yp1 = yp1.flatten()[q4]
                  xp2 = xp2.flatten()[q4]
                  yp2 = yp2.flatten()[q4] 
     
              # find the anchor positions by extrapolation
              # 2020-09-05 npmk changed arguments tck2? from ?p1 to ?p2
              
              anker  = np.zeros(2)
              anker2 = np.zeros(2)
              tck1x = interpolate.bisplrep(xf, yf, xp1, xb=-_flimit,xe=+_flimit,yb=-_flimit,ye=_flimit,kx=3,ky=3,s=None) 
              tck1y = interpolate.bisplrep(xf, yf, yp1, xb=-_flimit,xe=+_flimit,yb=-_flimit,ye=_flimit,kx=3,ky=3,s=None) 
              tck2x = interpolate.bisplrep(xf, yf, xp2, xb=-_flimit,xe=+_flimit,yb=-_flimit,ye=_flimit,kx=3,ky=3,s=None) 
              tck2y = interpolate.bisplrep(xf, yf, yp2, xb=-_flimit,xe=+_flimit,yb=-_flimit,ye=_flimit,kx=3,ky=3,s=None) 
     
              anker[0]  = xp1i = interpolate.bisplev(rx,ry, tck1x) 
              anker[1]  = yp1i = interpolate.bisplev(rx,ry, tck1y)  
              anker2[0] = xp2i = interpolate.bisplev(rx,ry, tck2x) 
              anker2[1] = yp2i = interpolate.bisplev(rx,ry, tck2y) 
      
              if self.chatter > 2: 
                  print('waveCal. bicubic extrapolation  ') 
                  print('waveCal. first order anchor position = (%8.1f,%8.1f)' % (xp1i,yp1i))
              if c20i == NaN:
                  print(" no second order extracted ")
              else:  
                if self.chatter > 2: 
                  print('waveCal. second order anchor position = (%8.1f,%8.1f) ' % (xp2i,yp2i))
          except: 
              if self.fail:  
                 raise RuntimeError("interpolation of wavecal data failed - ABORTING") 
              else:
                 xp1i,yp1i,xp2i,yp2i = None, None, None, None  
                
        else: 
        # 
        #  offset point is within the wavecal area (common case)
        #
        #  reduce arrays to section surrounding point
        #  get interpolated quantities and pass them onto self
           if self.mode == "interp2d":
               kx = ky = 1
               s = None
               self.mode = 'bisplines'
           else: 
               kx = ky = 3
               s = None    
           if self.mode == 'bisplines':
              # compute the Bivariate-spline coefficients
              # kx = ky =  3 # cubic splines (smoothing) and =1 is linear
              task = 0 # find spline for given smoothing factor
              #  s = 0 # 0=spline goes through the given points
              # eps = 1.0e-6  (0 < eps < 1)
              m = N1*N1
              qx = qy = np.where( (np.isfinite(xf.reshape(m))) & (np.isfinite(yf.reshape(m)) ) )
              tck1 = interpolate.bisplrep(xf.reshape(m)[qx], yf.reshape(m)[qy], xp1.reshape(m)[qx],xb=-_flimit,xe=+_flimit,yb=-_flimit,ye=_flimit, kx=kx,ky=ky,s=s) 
              tck2 = interpolate.bisplrep(xf.reshape(m)[qx], yf.reshape(m)[qy], yp1.reshape(m)[qx],xb=-_flimit,xe=+_flimit,yb=-_flimit,ye=_flimit, kx=kx,ky=ky,s=s) 
              xp1i = interpolate.bisplev(rx,ry, tck1)
              yp1i = interpolate.bisplev(rx,ry, tck2)
              xp2i = 0
              yp2i = 0
              tck1 = interpolate.bisplrep(xf.reshape(m)[qx], yf.reshape(m)[qy], xp2.reshape(m)[qx],xb=-_flimit,xe=+_flimit,yb=-_flimit,ye=_flimit, kx=kx,ky=ky,s=s) 
              tck2 = interpolate.bisplrep(xf.reshape(m)[qx], yf.reshape(m)[qy], yp2.reshape(m)[qx],xb=-_flimit,xe=+_flimit,yb=-_flimit,ye=_flimit, kx=kx,ky=ky,s=s) 
              xp2i = interpolate.bisplev(rx,ry, tck1)
              yp2i = interpolate.bisplev(rx,ry, tck2)
             
              if self.chatter > 3: print('getCalData. x,y, = ',xp1i,yp1i, ' second order ', xp2i, yp2i)
           #
           elif self.mode == 'bilinear':
               xp1i = self.bilinear( rx, ry, xf[0,:].squeeze(), yf[:,0].squeeze(), xp1 )
               yp1i = self.bilinear( rx, ry, xf[0,:].squeeze(), yf[:,0].squeeze(), yp1 )
               xp2i = self.bilinear( rx, ry, xf[0,:].squeeze(), yf[:,0].squeeze(), xp2 )
               yp2i = self.bilinear( rx, ry, xf[0,:].squeeze(), yf[:,0].squeeze(), yp2 )
               if self.chatter > 2: 
                   print('waveCal. bilinear interpolation') 
                   print('waveCal. first order anchor position = (%8.1f,%8.1f)' % (xp1i,yp1i))
                   print('waveCal. second order anchor position = (%8.1f,%8.1f) ' % (xp2i,yp2i))            
        self.anchor1 = np.array([xp1i,yp1i]) 
        self.anchor2 = np.array([xp2i,yp2i]) 
        if self.chatter > 1: 
            print(f'waveCal. anker [DET-pix]   = {self.anchor1}')
        if self.chatter > 2:    
            print(f'waveCal. anker [DET-img]   = {self.anchor1 - [77+27,77+1]}')
            print(f'waveCal. second order anker [DET-pix] = {self.anchor2}  [DET-pix] ') 
            print(f'waveCal. second order anker [DET-img] = {self.anchor2 - [77+27,77+1]}  [DET-img] ') 
        self.msg = msg

    def bilinear(self,x1,x2,x1a,x2a,f,):
       '''
       Given function f(i,j) given as a 2d array of function values at
       points x1a[i],x2a[j], derive the function value y=f(x1,x2) 
       by bilinear interpolation. 
   
       requirement: x1a[i] is increasing with i 
                x2a[j] is increasing with j
       20080303 NPMK        
       '''
       import numpy as np
   
       # check that the arrays are numpy arrays
       x1a = np.asarray(x1a)
       x2a = np.asarray(x2a)
      
       #  find the index for sorting the arrays
       n1 = len(x1a)
       n2 = len(x2a)
       x1a_ind = x1a.argsort()
       x2a_ind = x2a.argsort()
   
       #  make a sorted copy
       x1as = x1a.copy()[x1a_ind]
       x2as = x2a.copy()[x2a_ind]
   
       # find indices i,j for the square containing (x1, x2)
       k1s = x1as.searchsorted(x1)-1
       k2s = x2as.searchsorted(x2)-1
   
       #  find the indices of the four points in the original array
       ki = x1a_ind[k1s]
       kip1 = x1a_ind[k1s+1]
       kj = x2a_ind[k2s]
       kjp1 = x2a_ind[k2s+1]
       if self.chatter > 3:
           print('FIND solution in (x,y) = (',x1,x2,')')
           print('array x1a[k-5 .. k+5] ',x1a[ki-5:ki+5])
           print('array x2a[k-5 .. k+5] ',x2a[kj-5:kj+5])
           print('length x1a=',n1,'   x2a=',n2)
           print('indices in sorted arrays = (',k1s,',',k2s,')')
           print('indices in array x1a: ',ki, kip1)
           print('indices in array x2a: ',kj, kjp1)
      
       #  exception at border:
       if ((k1s+1 >= n1) ^ (k2s+1 >= n2) ^ (k1s < 0) ^ (k2s < 0) ):
           if self.chatter > 3: print('bilinear. point outside grid x - use nearest neighbor ')
           if ki + 1 > len(x1a) : ki = len(x1a) - 1
           if ki < 0 : ki = 0
           if kj + 1 > len(x2a) : kj = len(x2a) - 1
           if kj < 0 : kj = 0
           return f[ki, kj]
  
       # Find interpolated solution
       y1 = f[kj  ,ki  ]
       y2 = f[kj  ,kip1]
       y3 = f[kjp1,kip1]
       y4 = f[kjp1,ki  ]
    
       t = (x1 - x1a[ki])/(x1a[kip1]-x1a[ki])
       u = (x2 - x2a[kj])/(x2a[kjp1]-x2a[kj])
   
       y = (1.-t)*(1.-u)*y1 + t*(1.-u)*y2 + t*u*y3 + (1.-t)*u*y4
       if self.chatter > 4: 
           print('bilinear.                   x         y          f[x,y]    ')
           print('bilinear.   first  point ',x1a[ki  ],x2a[kj],  f[ki,kj])
           print('bilinear.   second point ',x1a[kip1],x2a[kj],  f[kip1,kj])
           print('bilinear.   third  point ',x1a[kip1],x2a[kjp1],  f[kip1,kjp1])
           print('bilinear.   fourth point ',x1a[ki  ],x2a[kjp1],  f[ki,kjp1])
           print('bilinear. fractions t, u ', t, u)
           print('bilinear. interpolate at ', x1, x2, y)
       return y    
 
 
class FluxCal(Caldb):

    def __init__(self, grismmode=None, wheelpos=None, msg="", use_caldb=False, 
                 chatter=0):
    
        self.msg = msg
        self.chatter = chatter
        self.use_caldb = use_caldb
        self.phipos = None
        self.calfilepath = None
        self.ea_extname = None
        self.ea_model = None

        if grismmode == None: 
           if wheelpos == None: 
              raise IOError ("Give either grism mode or filter wheel position.")
           else: 
              self.wheelpos = wheelpos
        else: 
            try:
               self.wheelpos = {"uv nominal":200,"uv clocked":160,
              "visible nominal":1000,"visible clocked":955}[grismmode.lower()]  
            except:
               raise IOError ("the grismmode parameter needs to be one of: "\
               "'uv nominal','uv clocked','visible nominal','visible clocked' ")        

        
    def getZmxFlux(self, x = 0 ,y = 0 ,modelhdu=None ,ip=1):
       '''Interpolate model to get normalized flux. 
       
       parameters
       ----------
       x, y : float
         anchor coordinate x,y to find an interpolated solution to the model
         
       modelhdu : fits hdu 
         binary table extension (header + data)
         fields are wave, xpix, ypix, flux
     
       ip : int
         The order of the interpolation (1=linear, 2=quadratic, 3=cubic)  
     
       returns
       -------
       flux interpolated at (x,y) in (xpix, ypix) as function of wave
     
       '''   
       import numpy as np
       from scipy import interpolate
   
       # test input 
       if not ((type(modelhdu) != 'astropy.io.fits.hdu.table.BinTableHDU') | \
              (type(modelhdu) != 'pyfits.hdu.table.BinTableHDU') ):
              raise IOError("getZmxFlux modelhdu parameter is not a proper FITS HDU bintable type")
          
       n3     = 28*28
       print (modelhdu.header['NAXIS2'])
       n2     = np.int( modelhdu.header['NAXIS2'] /n3)
       if not ((n2 == 12) | (n2 == 16)):
          raise IOError("getZmxFlux: size of array in MODEL not correct; perhaps file corrupt?") 
   
       zmxwav = modelhdu.data['WAVE']
       xp     = modelhdu.data['XPIX']
       yp     = modelhdu.data['YPIX']
       zmxflux = modelhdu.data['FLUX']
   
       zmxwav = zmxwav.reshape(n2,n3)[:,0]
       xp     = xp.reshape(n2,n3)
       yp     = yp.reshape(n2,n3)
       zmxflux = zmxflux.reshape(n2,n3)
   
       flux   = np.zeros(n2,dtype=float)
      
       dminx =  0
       dminy = -100
       dmax  =  2100
         
       for j2 in range(n2):      # loop over wavelengths
          # filter out outliers
          q = (xp[j2,:] > dminx) & (xp[j2,:] < dmax) & (yp[j2,:] > dminy) & (yp[j2,:] < dmax) 
          if len(np.where(q)[0]) < 17:
             print("getZmxFlux warning: at wavelength=",zmxwav[j2]," not sufficient valid points found")
            
          fx = xp[j2,q]
          fy = yp[j2,q]     
          ff = zmxflux[j2,q]
            
          try:
             tck = interpolate.bisplrep(fx,fy,ff,xb=dminx,xe=dmax,yb=dminy,ye=dmax,kx=ip,ky=ip,)
             flux[j2]  = interpolate.bisplev(x, y, tck)
          except:
             raise RuntimeError ("getZmxFlux ERROR in interpolation") 
            
       return flux                         
               
    
    def readFluxCalFile(self, anchor=None, option="default", spectralorder=1,
                        arf=None,):
       """Read the new flux calibration file, or return None.
   
       Parameters
       ----------
       **anchor** : list, optional
          coordinate of the anchor
      
       **option** : str
          option for output selection: 
            option=="default" + anchor==None: old flux calibration
            option=="default" + anchor : nearest flux calibration + model extrapolation
        option=="nearest" : return nearest flux calibration
        option=="model" : model 
    
       **spectralorder** : int
            spectral order (1, or 2)
    
       **arf**: path | "CALDB" | "UVOTPY"
            fully qualified path to a selected response file
            or 
            "CALDB" : use caldb 
    
        - **msg**: str
            buffer message list (to add to)    

       Returns
       -------
       None if not (yet) supported
       option == 'model' returns the HDU (header+data) of the model 
       option == 'nearest'
          returns the fits HDU of the nearest calfile
       option == 'default' and anchor == None:
          returns the fits HDU of the nearest calfile 
       option == 'default' and anchor position given (in detector coordinates) 
          returns the fits HDU and an 
          interpolating function fnorm(wave in A) for the flux correction
       msg : string comments  separated by \n 
   
       Notes
       -----    
                 
       2013-05-05 NPMKuin
       """
       from astropy.io import fits
       import os 
       import sys
       import numpy as np
       from scipy import interpolate
       #from . import getZmxFlux

       msg = self.msg
       if arf == None: 
           if self.use_caldb: arf = 'CALDB'
           else: arf = 'UVOTPY'

       if (type(anchor) != typeNone):
          if (len(anchor) != 2):
             sys.stderr.write("input parameter named anchor is not of length 2")
          elif type(anchor) == str: 
             anchor = np.array(anchor, dtype=float)  

       check_extension = False
       if spectralorder == 2:
           option == "nearest"
           check_extension = True
         
       if self.chatter > 1:
           print("FluxCal: readFluxCalFile attempt to read effective area file: ")

       if (arf.upper() == "CALDB") | (arf.upper() == "UVOTPY"):
               # try to get the file from the CALDB or UVOTPY
               x = Caldb(wheelpos=self.wheelpos, msg=self.msg, 
                    use_caldb = arf.upper() == "CALDB",
                    chatter=self.chatter )
               self.calfile, self.ea_extname, self.ea_model= x.locate_caldata(
                    what="flux",spectralorder=spectralorder)
               hdu = fits.open(self.calfile)
               if self.chatter > 3: print ("accessing %s"%(self.calfile))
               if self.ea_extname == "": 
                   raise RunTimeError("Version flux cal file is too old.")
       else:
               # path to arf is explicitly supplied
               # the format must give the full path (starting with "/" plus FITS extension
               # if no extension was supplied and there is only one, assume that's it.
               # check version==2, using presence of CBD70001 keyword and see if spectral order is right
               if self.chatter > 3: print(arf)
               try:  # get extension from path 
                   if len(arf.split("+") ) == 2: 
                       file, extens = arf.split("+")
                   elif len(arf.split("[") ) == 2:
                       file = arf.split("[")[0]
                       extens = arf.split("[")[1].split("]")[0] 
                   else:
                       check_extension = True
                   arf = file
                   self.extname = extens # use given extension 
               except: 
                   raise IOError("The supplied effective area file name "+arf+" cannot be understood.")            
       
               hdu = fits.open(arf)
               if check_extension:  # old version file 
                   if hdu[1].header['CBD60001'].split("(")[1].split(")")[0] != spectralorder: 
                       raise IOError("The supplied effective area file is not correct spectral order.")
                   if ("CBD70001" not in hdu[extens].header) :  # old version
                       print("Spectral order = %i. \t"%(spectralorder))
                       print("Using the oldest version of the effective area. \n"+\
                         "Flux, COI correction will be wrong.")
                       self.msg = msg  
                       return hdu[self.ea_extname],msg
   
      
       if self.chatter > 0: print("Spectral order = %i: using flux calibration file: %s"%(spectralorder,arf))   
       if self.chatter > 2: hdu.info()
       msg += "Flux calibration file: %s\n"%(arf.split('/')[-1])
   
       if (option == "default") | (option == "nearest"):
      
          if type(anchor) == typeNone:  # assume centre of detector
               anchor = [1000,1000]
          else:
              if (option == "default"): 
                  modelhdu = hdu[self.ea_model]
              if self.wheelpos < 500:
                  n2 = 16
              else: 
                  n2 = 12   
              names = []      # names extensions
              calanchors = [] # anchor positions for each calibration extension
              dist = []       # distances 
              for h in range(1,len(hdu)):
                  N = hdu[h].header["EXTNAME"].upper()
                  NL = N.split("_")
                  if (len(NL) == 3):
                      if( int(NL[2][1]) == spectralorder): 
                          names.append(N)
                          root, ankxy, ord = NL
                          ankcal = ankxy.split("AY")
                          ankcal[0] = float(ankcal[0].split("AX")[1])
                          ankcal[1] = float(ankcal[1])
                          calanchors.append([ankcal[0],ankcal[1]])
                          dist.append( (ankcal[0]-anchor[0])**2+(ankcal[1]-anchor[1])**2 )
              # now we have a list of valid extnames, and anchor positions 
              # for the calibration file fits-extensions
              dist = np.array(dist) 
              k = np.where( dist == np.min(dist) )[0][0]  
              cal = hdu[names[k]]
              if self.chatter > 0:
                  print("Nearest effective area is %s  - selected"%(names[k]))
              msg += "Selected nearest effective area FITS extension \n\t%s\n"%(names[k])
              if (option == "nearest"): 
                  self.msg = msg
                  return cal, msg
                  
              try:  
                  if self.chatter > 4: 
                      print("ReadFluxCalFile:      calanchor ", calanchors[k]) 
                      print("ReadFluxCalFile:         anchor ", anchor)
                      print("ReadFluxCalFile: MODEL  extname ", modelhdu.header['extname']) 

                  modelcalflux = self.getZmxFlux (x=calanchors[k][0],y=calanchors[k][1],
                        modelhdu=modelhdu, )
                  modelobsflux = self.getZmxFlux (x=anchor[0],y=anchor[1],modelhdu=modelhdu, )
                  
                  q = np.isfinite(modelcalflux) & np.isfinite(modelobsflux) 
                  w = 10.0*modelhdu.data['WAVE']
                  if self.chatter > 4: 
                      print("ReadFluxCalFile:         check:  ")
                      print("ReadFluxCalFile:         w.shape ",w.shape)
                      print("ReadFluxCalFile:            =784*",n2," ?")
                  w = w.reshape(n2,784)[q,0]
                  fn = modelobsflux[q]/modelcalflux[q]
                  w1 = 1650.0
                  f1 = 1.0      # was f1 = (fn[1]-fn[0])/(w[1]-w[0])*(w1-w[0]) + fn[0]
                  n = len(w)+2
                  x = np.zeros(n,dtype=float)
                  y = np.zeros(n,dtype=float)
                  x[0] = w1
                  y[0] = f1
                  x[1:-1] = w
                  y[1:-1] = fn
                  x[-1] = 7000.
                  y[-1] = y[-2]
                  y[ y < 0 ] = 0.0
                  fnorm = interpolate.interp1d(x,y,bounds_error=False, fill_value=0.)        
                  msg += "Flux corrected for variation over detector using model\n"
                  self.msg = msg
                  return cal, fnorm, msg
              except RuntimeError:
                  pass
                  print("WARNING: Failure to use the model for inter/extrapolation of the calibrated locations.")
                  print("         Using Nearest Eaafective Area File for the Flux calibration.")
                  fnorm = interpolate.interp1d([1600,7000],[1.,1.],)
                  self.msg = msg
                  return cal, fnorm, msg 
       elif option == "model":
           return hdu[self.ea_model]
       else:
           raise RuntimeError( "invalid option passed to readFluxCalFile") 
    

    def get_model(self, spectralorder=1, arf=None):
        #from . import readFluxCalFile
        msg = self.msg
        chatter = self.chatter
        anchor = None
        x = self.readFluxCalFile(option="model",
             spectralorder=spectralorder,
             arf=arf, msg=msg,
             chatter=chatter)
        return x     

