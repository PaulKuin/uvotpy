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
import numpy as np



def maketime(t,format=None):
    from astropy.time import Time
    time = Time(t,format=format,)
    print (time)
    return time



class SimGrism():

   def __init__(self,target_position_or_name, wheelpos=160, roll=None, 
        offset=[0,0], datetime=None, blim=16.0, figno=16,
        timeformat=None, chatter=0):
       """ 
       input parameters:
          target: name, astropy coordinate, or (ra,dec) in degrees
          wheelpos: one of 160,200,955,1000
             uv clocked : 160, uv nominal: 200, visible clocked: 955, visible nominal: 1000
          roll : int or float
             value roll in degrees  
          offset: list of dimension 2
             offset on detector from center in arcminutes! 
          datetime: datetime object, time string formatted as "2000-01-01"
          blim: float 
             limiting magnitude (typically 16.5 plus minus 1.5)
          figno: int
             number for plot of grism detector image at roll extremes for given date.
          timeformat: string
             if datetime is given in an alternative format besides above 
             (from astropy.time.Time)
          chatter: 0..5 
             verbosity   
       old plan was: draw the plot:
           - the rotated DSS image 
           - the source positions [zeroth order]
           - the position of the spectra
           - the regions of avoidance around bright stars
             incl. readout streak   

       """
       from pylab import figure
       import warnings
       #import matplotlib
       
       warnings.filterwarnings('ignore') # ignore all
       #with warnings.catch_warnings():
       #   warnings.filterwarnings('ignore', r'posx and posy should be finite values')
       #   warnings.filterwarnings('ignore','UserWarning: Warning: converting a masked element to nan.\
       # return array(a, dtype, copy=False, order=order)' )
  
       # for certain sky position, find valid dates for given roll .... TBD
       
       # plot given date+time or roll 
       
       if type(datetime) != type(None):
          t0 = maketime(datetime,format=timeformat)
       else: t0 = None   
       
       if type(roll) == type(None): extra_plot = False 
       else: extra_plot = True
       
       X1 = SimGrism_sub1(target_position_or_name,wheelpos=wheelpos,roll=roll,
          offset=[offset[0],offset[1]],
          datetime=t0,blim=blim,storeDSS=None,chatter=chatter,)
       # now we have the range of roll ; plot extreme cases
       roll_range = X1.rolldata
       min_roll = float(roll_range["min_roll"])
       max_roll = float(roll_range["max_roll"])
       X0 = SimGrism_sub1(target_position_or_name,wheelpos=wheelpos,roll=min_roll,
          offset=[offset[0],offset[1]],
          datetime=t0,blim=blim,storeDSS=None,chatter=chatter,)
       X2 = SimGrism_sub1(target_position_or_name,wheelpos=wheelpos,roll=max_roll,
          offset=[offset[0],offset[1]],
          datetime=t0,blim=blim,storeDSS=None,chatter=chatter,)

       fig0 = figure(figno,figsize=(12.5,4.5))
       fig0.clf()
       ax0 = fig0.add_subplot(121)
       ax1 = fig0.add_subplot(122)
       R = X0.plot_catalog_on_det(ax0,title2="minimum roll")
       R = X2.plot_catalog_on_det(ax1,title2="maximum roll")
       fig0.colorbar(R,fraction=0.05,pad=0.05,label="blue=hot      yellow=cool")
       #ax0.text(-50,750,f"{target_name_or_position}",rotation='vertical',va='center')
       ax1.set_ylabel("")
       
       # print out details
       # use ranew,decnew = self.decsex(raoff.value,decoff.value) to get sexagesimal 
       targ_ra_hms, targ_dec_dms  = X1.decsex(X1.target.ra.deg,X1.target.dec.deg)
       point_ra_hms,point_dec_dms = X1.decsex(X1.pointing.ra.deg,X1.pointing.dec.deg)
       print (80*"=")
       print (f"\ntarget=({target_position_or_name}) -- report for roll:{X1.roll} has\n")
       print (f"   target position: {X1.target} = {targ_ra_hms}, {targ_dec_dms}")
       print (f"   at offset: {X1.offset} \n")
       print (f"   plan pointing position: {X1.pointing}  = {point_ra_hms}, {point_dec_dms}")
       print (f"   with roll = {X1.roll}\n ")
       print (80*"=")
       # 
       if extra_plot:
           fig1 = figure(figno+1)
           fig1.clf()
           X1.plot_catalog_on_det(fig1,title2="optimum roll")
     
        
   def update_roll():
       x=1
   
   def update_offset():
       x=1

detector_to_sc = [-118.774,-118.607,-118.607,-118.607] # teldef[0] grism 160,200,955,1000 2009 Wayne Landsman   
# => 241.393

class SimGrism_sub1(SimGrism):

   """
   code that can be used to display a star field and determine where 
   the grism spectra should fall relative to the sources on sky image

   SimGrism instance

   Parameters
   ==========
   target: str, list, astropy.coordinates.sky_coordinates.SkyCoord
      target name or initial position  
      
   offset : list or astropy.units.quantity.Quantity  of length 2
      offset of target from DET pointing position in arcmin on the grism image
      If type list: Xoffset, Yoffset 
      If type Quantity, Xoffset, Yoffset
      
   =>if there is an offset, the sky position of the pointing will be computed 
   using the target sky position and the offset on the grism image.
      
   roll: float, or astropy.units.quantity.Quantity 
      the roll angle in deg
   [savedVeil] : path | None (optional)
      file name of saved Veil data (to speed up program)
   [storeDSS] : path | None (optional)
      file name to store the DSS image   
   chatter : int [0..5]
      verbosity

   methods
   =======
   to match to sky rotate detector image,objects over the position angle, 
   i.e.,an angle of   240.65 - roll_angle (roll angle=pa_pnt keyword). 
   This was the value used for the grism calibration.[??]
   [ why not 360.-118.607 = 241.393 deg ? ]
   
   start: give roll, ra,dec of target and offset
   
   grab DSS image sky + metadata  

   Planned capabilities:     
   pick target (can be different RA,DEC:
     get angle spectrum
     get curvature parameters
     get distance zeroth order - first order - second order
     
   plot image
   plot rotated curved spectral track
   plot boundaries of detector
   plot extent of PSF of very bright sources
   plot first orders of very bright sources on/off detector
   
   update roll angle interactively
   update offset target
   update plot
   
   output solution
   
   Note
   ====
   This version : adjusts the angle if the spectrum is offset, 
   uses the curvature, interactively allows changes to offset and 
   roll
   """

   def __init__(self,target, wheelpos=160, roll=None, offset=[0,0],
       datetime=None, # must be an astropy.time.Time object
       savedVeil=None, 
       storeDSS=None,  
       blim=17.5, chatter=0):
       """
       Parameters:
          target : astropy.coordinates.sky_coordinate.SkyCoord
          wheelpos: int
             filter wheel filter selection
          roll: float (optional)
             S/C roll angle in degrees [0,360] 
             if roll is not given, the optimal roll for given date is computed 
             from the datetime, or zero  
          offset: offset  (optional)
             offset of target on detector [x,y coordinate in arcmin] 
             if the offset is zero, then the target is at the boresight 
             which is the pointing of the UVOT on the sky   
          datetime: Time (optional)
             must be an astropy.time.Time object 
       """
       import os, sys
       import astropy
       import numpy as np
       from astropy import coordinates     
       from astropy import units
       from astropy.io import fits, ascii
       from astropy import wcs
       from astroquery.vo_conesearch import conesearch
       from uvotpy import calfiles
       from astropy.time import Time 
       from uvotpy import roll as swiftroll
       from uvotpy import generate_USNOB1_cat
       from grism_utilities import query_DSS
       from scipy import ndimage 
       
       self.pixelscale = 0.54 
           # arcsec per pixel (used in my calibration); 
           # Wayne used 0.56 for the zeroth orders.
       self.wheelpos = wheelpos
       self.chatter = chatter
       self.chatter2 = 0
       self.dtor = np.pi/180.0
       self.targetin = target
       
       # spacecraft Roll angle (optional input parameter)
       if hasattr(roll,'to'):
           self.roll = roll.to(units.degrees)
           self.set_roll = False
       elif type(roll) == float:
           self.roll = roll*units.deg
           self.set_roll = False
       elif type(roll) == type(None):
           if type(datetime) == type(None): 
               self.roll = 0.0 *units.deg
               self.set_roll = False
           else:
               self.set_roll = True
               self.roll = 0.0*units.deg
       else:   
           raise IOError("Parameter roll_angle is not correct.")  
       
       # sky position target (input parameter)
       if hasattr(target, 'ra'):
           self.ra =  target.ra
           self.dec = target.dec
       elif type(target) == list:
           self.ra,self.dec = target 
           pos = coordinates.Skycoord(target[0],target[1],unit=[units.deg,units.deg])
           if chatter > 1 : print ("WARNING assuming input position target %s"%(pos))
           target = pos
       elif type(target) == str:
           pos = coordinates.SkyCoord.from_name(target,frame=coordinates.ICRS)
           self.ra =  pos.ra
           self.dec = pos.dec
           target = pos
       else:
           raise IOError("Parameter target is not a valid option.")    
       self.target = target # target sky position
           
       # offset from boresight "center image"
       if hasattr(offset,'to'):
           self.offset = offset.to(units.arcmin)
       elif type(offset) == list:
           self.offset = np.array([offset[0],offset[1]])*units.arcmin
       else:   
           raise IOError("Parameter offset needs units.") 
           
       self.pixoffset = self.offset.to(units.arcsec)/(self.pixelscale
               *units.arcsec/units.pix)
       
       # get first pointing (centre of image) from the offset and target position,
       # however, this is a simple linear shift and rotation, does not include the 
       # distortion of the grism yet, so the final target position in detector 
       # coordinates may differ a bit. So self.pixoffset is approximate. 
       
       self.pointing = None     
       self.offset_pointing() # loads -> self.pointing using target and offset

       # initialise the anchor - angle - dispersion data  
       # extend the extrapolation range of the calibration from 0.19 to 0.22
       self.cal = calfiles.WaveCal(wheelpos=wheelpos, use_caldb=False, 
                  chatter=self.chatter2,_fail_or_report=False, _flimit=0.22)
       
       # calibrated angle of the spectrum at the offset position 
       # first order anchor position on the detector *** in det coordinates ***
       self.theta = self.cal.theta(offsetdelta=self.offset) 
       self.anker = self.cal.anchor(offsetdelta=self.offset,sporder=1)
    
       # get veil (cached?)
       #self.veil = self.make_fademask( xaxis=1987, yaxis=2046, )
       
       # get initial DSS image at pointing position (not target, unless no offset)
       if chatter > 4: 
           print ("storeDSS:",type(storeDSS),' = ',storeDSS)
       self.dsshdr, self.dssimg = query_DSS(self.pointing, ImSize=27.0, server="STSCI",
                      version = "3", output=storeDSS, chatter=0)
       #self.rotimg = self.dssimg # initialise rotated dss                 
       
       # get UB1 _source list_ of stars 
       self.blim = blim 
       
       if self.chatter > 4: 
          print (f" input types to generate_USNOB1_cat:", type(self.ra.deg), type(self.dec.deg) )
       if self.wheelpos == 160: radius = 24.*units.arcmin
       elif self.wheelpos == 200 : radius = 30.*units.arcmin
       else: radius = 26*units.arcmin 
       
       tab = generate_USNOB1_cat.get_usnob1_cat( self.ra.deg, self.dec.deg, self.blim, radius=radius ,tableout=True, chatter=self.chatter)
       usnob1 = tab
       
       self.ub1ra    = tab['_RAJ2000']
       self.ub1dec   = tab['_DEJ2000']   # these are astropy.table.column.Column objects
           
       # sort by distance to target (to force numbering of nearby sources for plot)
       dist2target = []
       for p1,p2 in zip(self.ub1ra, self.ub1dec):
           dist2target.append( 
              (p1*np.cos(p2*self.dtor)-self.target.ra.deg*np.cos(self.target.dec.deg*self.dtor))**2 + 
              (p2 - self.target.dec.deg)**2
              ) 
       tab.add_column(dist2target,name="dist2t")
       tab.sort(["dist2t"])
       self.ub1ra    = tab['_RAJ2000']
       self.ub1dec   = tab['_DEJ2000']   # these are astropy.table.column.Column objects
       self.ub1b2mag = tab['B2mag']
       self.ub1r2mag = tab['R2mag']
       if self.chatter >3: print ("read in the usnob1 catalogue ")

       # WCS: 
       #    get wcs of the DSS image 
       #    like, Wcss = wcs.WCS (header=hdr,key='S',relax=True,)                 
       self.dssWcs = wcs.WCS (header=self.dsshdr,)    # ,key='S',relax=True,)  DSS file 
       
       # other WCS instances
       # sky to grism zeroth order anchor coordinates
       self.detWcs = wcs.WCS (header=self._det_header(),key='S') # IMG coordinate grisms
       
       # first order anchor is more complicated and mimics 
       #    the uvotgetspec.findInputAngle():
       # first convert from sky to an intermediary between RAW and DET (lenticular) 
       
       self.lenticularWcs = wcs.WCS (header=self._lenticular_header()) # IMG coordinate lenticular 
       # convert from the intermediate system to GRISM DET (mm) coordinates using x,y
       #wcsd={"CRVAL1":0,"CRVAL2":0,"CRPIX1":993.0,"CRPIX2":1022.5,"CTYPE1":'DETX',
       #   "CTYPE2":'DETY',"CUNIT1":'mm',"CUNIT2":'mm',"CDELT1":0.009075,
       #   "CDELT2":0.009075,"NAXIS1":1987,"NAXIS2":2046} 
       #wcsd={"CRVAL1":0,"CRVAL2":0,"CRPIX1":996.0,"CRPIX2":1022.5,"CTYPE1":'DETX', 
       #   "CTYPE2":'DETY',"CUNIT1":'mm',"CUNIT2":'mm',
       # shift from uvw2 boresight to lenticular DET center (in mm) 
       wcsd={"CRVAL1":0,"CRVAL2":0,"CRPIX1":953.23,"CRPIX2":1044.9,"CTYPE1":'DETX', 
          "CTYPE2":'DETY',"CUNIT1":'mm',"CUNIT2":'mm',
       "CDELT1":0.009075,"CDELT2":0.009075,"NAXIS1":1987,"NAXIS2":2046}
       self.DETWCS = wcs.WCS(wcsd)
       
       # after this, convert the DET (mm) to DET(pix) offset to be supplied to calfiles:
       # DX,DY = (xD,yD)/0.009075 + (1100.5,1100.5)
       # using x,y in a combined offssetwcs to get DX,DY would probably be:
       # shift from uvw2 boresight to lenticular DET center (pixels)
       wcsoffset={"CRVAL1":1100.5,"CRVAL2":1100.5,"CRPIX1":953.23,"CRPIX2":1044.9,
           "CTYPE1":'DETX',"CTYPE2":'DETY',"CUNIT1":'pix',"CUNIT2":'pix',
           "CDELT1":1,"CDELT2":1,"NAXIS1":1987,"NAXIS2":2046}
       self.wcsoffset = wcs.WCS(wcsoffset)
       # use this offset in the instance from "calfiles.py" to get the slope of the 1st order 
       # on the detector, the anchor, the dispersion.  

       # compute the source positions in img coordinates on the DSS image  
       #self.ub1_dssx, self.ub1_dssy = self.dssWcs.wcs_world2pix(self.ub1ra, self.ub1dec,0)   
       if self.chatter > 3: print(" done WCS conversions")

       
       if type(datetime) == type(None):
          datetime = Time.now()
          self.datetime = datetime
       if type(datetime) == astropy.time.core.Time:  
          self.yearday = datetime.yday
          self.year = int(self.yearday[:4])
          self.doy = int(self.yearday[5:8])
          #utime = swiftroll.ICSdateconv("%s-%03d-00:00:00"%(self.year,self.doy))
          #utime = datetime.unix  # <- wrong time
          #self.roll= swiftroll.optimum_roll(self.dtor*self.ra.deg, self.dtor*self.dec.deg, 
          #      utime )/self.dtor * units.deg
          if self.chatter > 3 : print (" times done ")
          
          self.rolldata = swiftroll.forday(self.ra.deg,self.dec.deg,
                day=self.doy,year=self.year) 
          if self.set_roll:      
              self.roll = self.rolldata['roll'] * units.deg 
          print ("Roll angle: roll = %.1f for %s - range=[%.1f,%.1f]"%(self.roll.value, 
                datetime.iso,self.rolldata['min_roll'],self.rolldata['max_roll']))
       else: 
          raise RunTimeError("Invalid datetime parameter entered.re")   
#
#    probably should initialise these 
#
       self.ax1 = None
       self.ax2 = None
       self.sp_ang = None
       self.anchor = None
       self.dss_footprint = None
       self.rotated_img = None
       if self.chatter > 3: print ("init done")
       
# # # # # # # # # 
       
   def plot_catalog_on_sky(self,fighandle):
       """
       Plot catalog on sky coordinates (RA, Dec) 
       
       Plot catalog stars with size according to brightness, and color according 
       to color scale in axes 
       includes colorbar
       
       no rotation - useful mainly for development

       """
       import numpy as np
       from astropy import coordinates, units     
       ax = fighandle.add_subplot(111)
       B2mR2 = [-0.8,+2.4]
       for a,b,c,d in zip(self.ub1ra,self.ub1dec,self.ub1b2mag,self.ub1r2mag): 
           B2mR2.append(c-d)
       B2mR2 = np.array( B2mR2 )   
       R = ax.scatter(self.ub1ra,self.ub1dec,
                  s=(20.-self.ub1b2mag)*2.,
                  c=B2mR2[2:],norm=None,cmap='plasma')
       ax.plot(self.ra.value,self.dec.value,'+',markersize=20,color='purple',lw=2,label='source') 
       fighandle.colorbar(R,fraction=0.05,pad=0.05,label="blue=hot      yellow=cool")
       ax.set_xlabel('RA (deg)')
       ax.set_ylabel('Dec (deg)') 
       # plot det frame
       #detwcs = wcs.WCS(header=self._det_header(),key='S')
       z = self.det_frame(WCS=self.detWcs)
       ax.plot(z[0][:],z[1][:],'k',lw=1)
       ax.plot(z[0][0],z[1][0],'*k',ms=7,label='det origin') # origin
       ax.plot(z[0][:2],z[1][:2],'or--',lw=2.7) # bottom detector image
       # slit (uvgrism) for source at anchor
       x,y,z,c = self.slit()
       xd,yd = self.rotate_slit(x,y)
       z1 =  self.detWcs.pixel_to_world(xd,yd)
       ax.plot(z1.ra,z1.dec,'b-',label='first order')
       ax.legend()
       ax.invert_xaxis() # RA runs opposite to longitude on ground
       
   def plot_catalog_on_det(self,fighandle,annotate=True,title2=""):
       """
       Similar to plot_catalog_on_sky, but now scaled to the detector coordinates
       """
       import numpy as np
       from astropy import coordinates, units  
       import matplotlib
       
       if self.chatter > 2: print(f"defining figure axis ")
       
       if type(fighandle) == matplotlib.figure.Figure :
          ax = fighandle.add_subplot(111)
          do_colorbar = True
       else: # just assume fighandle is an matplotlib.axes._subplots.AxesSubplot instance
          ax = fighandle
          do_colorbar = False
       # colour of stars (using UB1 catalog R2 and B2 magnitudes)
       B2mR2 = [-0.8,+2.4]
       for a,b,c,d in zip(self.ub1ra,self.ub1dec,self.ub1b2mag,self.ub1r2mag): 
           B2mR2.append(c-d)
       B2mR2 = np.array( B2mR2 )   
       
       #  det wcs positions of zeroth orders [IMG coord]
       detcat = np.array(self.detWcs.all_world2pix(self.ub1ra,self.ub1dec,0,adaptive=True))
       if self.chatter > 3: print(f"504 plot_catalog_on_det zeroth orders done")
       
       # zeroth order target position (image coordinate) [IMG coord] using 
       #   distortion correction for zeroth orders in WCS-S
       detsrc = self.detWcs.all_world2pix(self.ra,self.dec,0)
       
       # physical detector frame (grism image coordinate)
       z = np.array([0,1986,1986,0,0]), np.array([0,0,2045,2045,0])
       
       # find stars that can make a first order on detector and 
       # trim stars that are in dark region of clocked image 
       q1, q2 = self._screen_stars(detcat[0],detcat[1])
       
       # write table stars in field
       src_table = self.make_source_table(self.ub1ra[q1], self.ub1dec[q1],
           detcat[0][q1], detcat[1][q1], self.ub1b2mag[q1], B2mR2[2:][q1])
       if self.chatter > 3: print(f"521 plot_catalog_on_det field stars done")
       
       # treat target
       chatter = self.chatter
       self.chatter = 0
       xank,yank,theta = self.sky2det(self.ra.deg,self.dec.deg)
       
       self.chatter = chatter
       # slit points first order [input target coordinate in DET coord]
       if type(xank) != type(None):
           x,y,wid,sig = self.slit(xdet=xank,ydet=yank, sporder=1)
           xd,yd = self.rotate_slit(x,y,pivot=[xank-104,yank-78],theta=theta)  
       else: 
           print (f"WARNING: Target not on detector ? ")    
       if self.chatter > 3:
           print (f"536 plot_catalog_on_det target at {xank-104},{yank-78} with first order slope of {theta}.")
           
       # plot zeroth orders! 
       ax.scatter(detcat[0],detcat[1],   # plot all zeroth order near field sources
                 s=(20.-self.ub1b2mag)*2.,
                 c=B2mR2[2:],norm=None,cmap='plasma',marker='*',alpha=0.5)
       R = ax.scatter(detcat[0][q1],detcat[1][q1],  # plot sources in FOV 
                 s=(20.-self.ub1b2mag[q1])*2.3,
                 c=B2mR2[2:][q1],norm=None,cmap='plasma',marker='o')

       # plot target zeroth order         
       ax.plot(detsrc[0],detsrc[1], '+',markersize=15,color='darkgreen',lw=1.5,label='0th-target')
       # plot target first order
       target_xank_det, target_yank_det, target_angle = self.sky2det(self.target.ra.deg,self.target.dec.deg,)
       xank = target_xank_det-104
       yank = target_yank_det-78
       ax.plot(xank,yank, '+',markersize=15,color='purple',lw=1.5,label='1th-target')
       xxm,yym,xxp,yyp = self.slit_at_offset(xank,yank, target_angle) 
       q = (xxm > 0) & (xxm < 1970) & (yym < 2050) & (yym > 0)
       if xank < 1750:
           ax.plot(xxm[q],yym[q],'m-',lw=1)
           ax.plot(xxp[q],yyp[q],'m-',lw=1)
       if self.chatter > 3: print(f"558 plot_catalog_on_det plotted zeroth orders")
    
       # colorbar
       if annotate:
          if do_colorbar:
              fighandle.colorbar(R,fraction=0.05,pad=0.05,label="blue=hot      yellow=cool")
          ax.set_xlabel('IMG-X (pix)')
          ax.set_ylabel(f'{self.targetin}\nIMG-Y (pix)') 
          ax.plot(z[0][:],z[1][:],'k',lw=1)  # plot IMG frame
          ax.plot(z[0][0],z[1][0],'*k',ms=7,label='IMG origin') # origin
       # delineation of clocked aperture zeroth orders
       if self.wheelpos == 160:
          xcentre,ycentre = (2630.,54.) # clocking center of aperture
          radius1 = 2025. # shadowed zeroth order = radius aperture
          phi = np.arange(0,0.9,0.01)*np.pi/2 + np.pi/2+0.25
          ax.plot(radius1*np.cos(phi)+xcentre,radius1*np.sin(phi)+ycentre,'k',lw=0.5) 
       if self.wheelpos == 955:   # needs updating with V clocked parameters see line 540
          xcentre,ycentre = (2630.,54.) # clocking center of aperture
          radius1 = 2025. # shadowed zeroth order = radius aperture
          phi = np.arange(0,0.9,0.01)*np.pi/2 + np.pi/2+0.25
          ax.plot(radius1*np.cos(phi)+xcentre,radius1*np.sin(phi)+ycentre,'k',lw=0.5) 
          
       # first orders of field sources 
       for k in src_table[:]:
           xank = k["FO-Ximg"]
           if (type(xank) == type(None)) or (xank == -99) or (xank > 1800):
                if self.chatter > 3: print(f"584 plot_catalog_on_det problem type xank {type(xank)}")
                continue
           yank = k["FO-Yimg"]
           theta = k["theta"]
           zde1 = k["ZO-Ximg"]
           zde2 = k["ZO-Yimg"]
           B = k["B2mag"]
           lw=0.5  # adapt line width to the source brightness > 13
           if B < 13: lw += 0.70*(13.-B)
           nn = k["nr"]
           if self.chatter == 5: print (f"594 src table = \n{k}")
           xxm,yym,xxp,yyp = self.slit_at_offset(xank,yank, theta*units.deg) 
           q = (xxm > 0) & (xxm < 1970) & (yym < 2050) & (yym > 0)
           if xank < 1750:
             ax.plot(xxm[q],yym[q],'c-',lw=lw)
             ax.plot(xxp[q],yyp[q],'c-',lw=lw)
           if (annotate and (type(xank) != type(None)) and 
                np.isfinite(xank) and np.isfinite(yank)):
               ax.text(xank,yank,str(nn),fontsize=8,color='k',ma='center')
               ax.text(zde1,zde2,str(nn),fontsize=8,color='g',ma='center') 
       ax.set_xlim(-100,2450)
       ax.set_ylim(-500,2150)
       grism = {160:"uv-clocked ",200:"uv-nominal ",955:"vis-clocked",1000:"vis-nominal"}
       if annotate: ax.legend(title=
           f"{grism[self.wheelpos]}  roll={int(self.roll.value)}  B < {self.blim}",
           fontsize=8,
           bbox_to_anchor=(0.05,1.02,1.3,0.15),loc=3,ncol=3) 
       if self.chatter > 3: 
           print(f"612 plot_catalog_on_det done")
       if not do_colorbar: return R


   def slit_at_offset(self,xank,yank,theta):
       """
       parameters:
         xank, yank : float
            the x, and y position of the anchor (IMG)
         theta : float
            the slope of the spectrum on the detector   
       """
       import numpy as np
       from astropy import units
       # get array of slit 
       x,y,wid,sig = self.slit(xank+104,yank+78,sporder=1) 
       # Projection effect on width, use rotation 
       proj = np.abs(1./np.cos(theta.to(units.rad).value))
       # slit boundaries (shifted in y only)
       yyp = y+wid*proj
       yym = y-wid*proj
       # in grism IMG coordinates
       xdm = xank+x*np.cos(theta.to(units.rad).value)+yym*np.sin(theta.to(units.rad).value)
       ydm = yank+x*np.sin(theta.to(units.rad).value)-yym*np.cos(theta.to(units.rad).value)
       #
       xdp = xank+x*np.cos(theta.to(units.rad).value)+yyp*np.sin(theta.to(units.rad).value)
       ydp = yank+x*np.sin(theta.to(units.rad).value)-yyp*np.cos(theta.to(units.rad).value)
       return xdm, ydm, xdp, ydp

   def sky2det(self, ra,dec,):
       """ use calibration of anchors from offset scale on lenticular image
          given the sky coordinate (RA,Dec) of a source
          return the anchor position in the grism DET coordinate. 
          input ra,dec values in deg.
       """
       import numpy as np
       from astropy import units
       
       #theta = self.cal.theta(offsetdelta=[0,0]) 
       if not np.isscalar(ra):
          raise RuntimeError(f"651 sky2det parameter ra is not a scalar: ra={ra}\n")
       # position x,y on uvw2 image:   
       x, y = self.lenticularWcs.all_world2pix(ra*units.deg, dec*units.deg,1, )
       
       xd,yd = self.wcsoffset.all_pix2world(x,y,0)
       xbs,ybs = boresight(filter='uvw2',) # det - same as in _lenticular_header()
       dx,dy = xd-xbs, yd - ybs # this is the offset used in calibration getSpec -- feed to cal.*
       if self.chatter > 4: 
          print (f"sky2det\nx,y = ({x},{y})\n -- xd,yd = ({xd},{yd})\n -- w2 boresight=({xbs},{ybs})")
          print (f"det offset: ({dx},{dy})")
          
       chatter = self.chatter
       self.chatter = 0
       xank_det, yank_det = self.cal.anchor(offsetdelta=[dx,dy],sporder=1)
       self.chatter =  chatter
       theta = self.cal.theta(offsetdelta=[dx,dy])
       # if no solution, anxk_det and yank_det are None's
       if theta == np.nan: 
          print (f"669 sky2det ERROR x={xank_det} y={yank_det} theta={theta} ")
       return xank_det,yank_det, theta

   def _screen_stars(self,x,y,order=0,f=1.0):
       """
        input parameters:
        x, y : float, arrays
        (x, y) items are the det coordinates of points to select
        output parameters:
        detselect1: find stars that can make a first order and 
        detselect2: trim stars that are in dark region of clocked image 
       """
       if self.wheelpos == 160:
           xcentre,ycentre = (2630.,54.) # clocking center of aperture
           radius1 = 2025. # shadowed zeroth order = radius aperture
           radius2 = 2900. # no transmission outside 
           detselect1 = (x > 542*f) & (x < 2500*f) & (y > -440*f) & (y < 1860*f) &\
            ((x - xcentre)**2 + (y - ycentre)**2 < 2500**2) 
           detselect2 = (x > 542*f) & (x < 2500*f) & (y > -440*f) & (y < 1860*f) &\
            ((x - xcentre)**2 + (y - ycentre)**2 < 2500**2) # nothing outside
       elif (self.wheelpos == 200) ^ (self.wheelpos == 1000):
           detselect1 = (x > 542*f) & (x < 2500*f) & (y > -440*f) & (y < 1860*f)
           detselect2 = detselect1
       elif self.wheelpos == 955:
           xcentre,ycentre = (2030.,54.) # UPDATE provisional value copied from 160
           radius1 = 2025. # shadowed zeroth order
           radius2 = 2900. # no transmission outside 
           detselect1 = (x > 500*f) & (x < 2500*f) & (y > -440*f) & (y < 1860*f) &\
            ((x - xcentre)**2 + (y - ycentre)**2 < 2500**2) # 
           detselect2 = (x > 500*f) & (x < 2500*f) & (y > -440*f) & (y < 1860*f) &\
            ((x - xcentre)**2 + (y - ycentre)**2 < 2500**2) # nothing outside
       else: raise RuntimeError("invalid wheelpos value encountered _screen_stars")
       return detselect1,detselect2
       

   def make_source_table(self,ra,dec,zox,zoy,b2mag,BmR):
       from astropy.io import ascii
       from astropy.table import Table, Column, MaskedColumn
       BmR= np.round(BmR,decimals=3)
       ankx = []
       anky = []
       mask = []
       ang = []
       for s1,s2 in zip(ra,dec):
           xank1,yank1,theta = self.sky2det(s1,s2) # these are DET coordinates on grism
           if type(xank1) == type(None):
               ankx.append(np.ma.masked)
               anky.append(np.ma.masked)  
               mask.append(True)
               theta = theta.value
           else:
               ankx.append(xank1-104) # convert grism-det to grism-img coord
               anky.append(yank1-78)
               mask.append(False)
               theta = theta.value
           ang.append(theta)  
       tnum = Column(  np.arange(len(ra))+1 ,name='nr', dtype='i4')
       tra = Column( ra, name="RA", dtype='float64')
       tdec = Column( dec, name="Dec", dtype='float64')   
       tzox = Column( zox, name="ZO-Ximg", dtype='i8')
       tzoy = Column( zoy, name="ZO-Yimg", dtype='i8')
       tankx = MaskedColumn( np.round(ankx, decimals=3), name="FO-Ximg")#, dtype='>i8',mask=mask)
       tanky = MaskedColumn( np.round(anky, decimals=3), name="FO-Yimg")#, dtype='>i8',mask=mask)
       tang = Column( np.round(ang,decimals=3), name="theta",)# dtype='float')
       tb2 = Column( b2mag, name="B2mag", dtype='float')
       tbmr = Column( BmR, name="B-R", dtype='float')
       t = Table([tnum,tra,tdec,tzox,tzoy,tankx,tanky,tang,tb2,tbmr])
       t.pprint
       t["FO-Ximg"].fill_value = -99
       t["FO-Yimg"].fill_value = -99
       #t.write(output="field_stars.dat", format='ascii', overwrite=True)
       ascii.write([np.arange(len(ra))+1, ra, dec, zox, zoy, ankx, anky, ang, b2mag, BmR], 
            output="field_stars.dat",names=["nr","RA","Dec","ZO-Ximg","ZO-Yimg",
            "FO-Ximg","FO-Yimg","theta","B2mag","B-R"],overwrite=True) 
       #t = ascii.read( "field_stars.dat")    
       return t

   def offset_pointing(self,):
       """
       Given a target offset, determine the pointing 
       
       Parameters
       ==========
       roll_angle: float, or astropy.units.quantity.Quantity 
         the roll angle in deg
       offset : astropy.units.quantity.Quantity  of length 2
         Xoffset, Yoffset are the positions of the target 
         on the detector relative to the centre 
        
       returns
       =======
       pointing: astropy.coordinates.sky_coordinate.SkyCoord
         the sky position of the center of the image when 
         the target is at the offset position on the detector.
       """
       import numpy as np
       from astropy import coordinates     
       from astropy import units
       
       # position angle         
       pa = self.PA(self.roll.to(units.deg)) # 240.64*units.deg-self.roll.to(units.deg)
       # compute the new pointing sky position which places the source at the 
       # desired offset 
       raoff = self.ra.to(units.deg) + self.offset[1].to(units.deg) * \
            np.sin(pa.to(units.rad))/np.cos(self.dec.to(units.rad))+\
            self.offset[0].to(units.deg) * \
            np.cos(pa.to(units.rad))/np.cos(self.dec.to(units.rad))
       decoff= self.dec.to(units.deg) - \
            self.offset[1].to(units.deg) * np.cos(pa.to(units.rad))+\
            self.offset[0].to(units.deg) * np.sin(pa.to(units.rad))
       self.pointing = coordinates.SkyCoord(raoff,decoff,frame=coordinates.ICRS,)  
       if self.chatter > 2: print (self.pointing, raoff.deg, decoff.deg)     
       if self.chatter > 0:
           print ('Decimal RA/DEC of pointing: %12.7f  %12.7f' % (raoff.deg,decoff.deg))
           ranew,decnew = self.decsex(raoff.value,decoff.value)
           print ('Sexigesimal RA/DEC of pointing '+\
             'with offset %s: %s, %s\n' % (self.offset,ranew,decnew))
    
   def decsex(self,ra,dec):
       ''' 
   simple code to convert RA, Dec from decimal degrees to sexigesimal
   
   input ra, dec in decimal degrees
   output ra,dec in sexagesimal strings
       '''
       import numpy as np
   
       ra = np.mod(ra,360.0) # 0<= ra < 360, positive
       rah=int(ra/15.)
       ram=int( (ra/15.-rah)*60.)
       ras=(ra/15.-rah-ram/60.)*3600
       newra="%02i:%02i:%05.2f" % (rah,np.abs(ram),np.abs(ras)) 
   
       sdec = np.sign(dec)
       ded=int(dec)
       dem=int((dec-ded)*60.)
       des=np.abs((dec-ded-dem/60.)*3600)
       ded=sdec*np.abs(ded)
       dem=np.abs(dem)
       newdec="%0i:%02i:%04.1f"%(ded,dem,des) 
       return newra,newdec


   def find_pointing_position(self,sky_position,detector_position,):
       """
       given the detector position and sky position of a source, compute the 
       sky position of the pointing

       Parameters
       ==========
       sky_position: astropy SkyCoord
           RA,Dec sky coordinates as an astropy coordinate object
           which matches the detector_position
        
       detector_position : [float,float] * units.pix
           (x,y) detector position of the (sky_position) source in subpixels
        
       roll_angle: float, or astropy.units.quantity.Quantity 
           the roll angle in deg
        
       returns
       =======
       offset : astropy.units.quantity.Quantity  of length 2
           Xoffset, Yoffset are the positions of the source 
           on the detector relative to the centre 
        
       pointing: astropy SkyCoord
           the sky position of the boresight of the image (the pointing)
        
       PROBLEM: this does not take the distortion matrix into account. Should 
       use calls to those programs.   
        
       """
       from astropy import units, coordinates
       
       pixscale = self.pixelscale*units.arcsec/units.pix
       ra = sky_position.ra
       dec = sky_position.dec
       # position of target on the detector in pixels
       #if hasattr(detector_position, 'to'):
       #    pixdetx, pixdety =  detposition.to(units.pix)
       #else:
       pixdetx,pixdety = detector_position 
       anchor = self.grism_boresight() # ... get_boresight(wheelpos)
       pixoffset = (pixdetx - anchor[0]*units.pix, 
                    pixdety - anchor[1]*units.pix)  
       print (f"pixoffset={pixoffset} pix")              
       offset = xoff, yoff = (pixoffset[0]*pixscale, pixoffset[1]*pixscale)
       print (f"offset={offset} arcsec")
       pa = self.PA(self.roll.to(units.deg))
       # compute the sky-offset from the offset position to the boresight 
       # this is the opposite direction from going from the boresight to the 
       # offset and is used to calculate the new pointing 
       raoff = sky_position.ra + \
            yoff.to(units.deg)*np.sin(pa.to(units.rad))/np.cos(dec.to(units.rad))+\
            xoff.to(units.deg)*np.cos(pa.to(units.rad))/np.cos(dec.to(units.rad))
       decoff= sky_position.dec - \
            yoff.to(units.deg)*np.cos(pa.to(units.rad))+\
            xoff.to(units.deg)*np.sin(pa.to(units.rad))
       pointing = coordinates.SkyCoord(raoff,decoff,frame=coordinates.ICRS,)
       return offset, pointing    

   def detposition2sky(self,detector_position, ):
       """
       given the detector position, compute the (source) sky position

       Parameters
       ==========
       detector_position : [float,float] * units.pix
           (x,y) detector position of the sky_position source in subpixels
        
       roll_angle: float, or astropy.units.quantity.Quantity 
           the roll angle in deg
        
       returns
       =======
        
       pointing: astropy SkyCoord
           the sky position 
       
       The conversion RAW->DET must have been done already
       
       the boresight of the detector  --- > pointing ra,dec  == pivot
       
       offset position ---> offset ra,dec (map with rotation included) 
       
       """
       from astropy import units, coordinates
       
       wheelpos = self.wheelpos
       
       pixscale = self.pixelscale*units.arcsec/units.pix

       # position of target on the detector in pixels
       #if hasattr(detector_position, 'to'):
       #    pixdetx, pixdety =  detector_position.to(units.pix)
       #else:
       #    pixdetx,pixdety = detector_position 
       pixdetx,pixdety = detector_position 
       anchor = self.grism_boresight() # ... get_boresight(wheelpos)
       
       pixoffset = (pixdetx - anchor[0]*units.pix, 
                    pixdety - anchor[1]*units.pix)  
       print ("offset in pixels: %5.1i,%5.1i "%pixoffset)              
       offset = xoff, yoff = (pixoffset[0]*pixscale, pixoffset[1]*pixscale)
       print (f"offset = {offset} arcsec\n")
       pa = self.PA(self.roll.to(units.deg))  # in deg
       # compute the sky-offset from the offset position to the boresight 
       # this is the opposite direction from going from the boresight to the 
       # offset  
       raoff = sky_position.ra + \
            yoff.to(units.deg)*np.sin(pa.to(units.rad))/np.cos(dec.to(units.rad))+\
            xoff.to(units.deg)*np.cos(pa.to(units.rad))/np.cos(dec.to(units.rad))
       decoff= sky_position.dec - \
            yoff.to(units.deg)*np.cos(pa.to(units.rad))+\
            xoff.to(units.deg)*np.sin(pa.to(units.rad))
       pointing = coordinates.SkyCoord(raoff,decoff,frame=coordinates.ICRS,)
       return pixoffset, offset     

   def PA(self,roll):
       """
       Position angle 
          
         Q: does this change over time ?
         
       """
       from astropy import units
       if hasattr(roll,'to'):
           return 240.64*units.deg - roll
       else:
           return (240.64 - roll) * units.deg
       # for this value, 360-240.64=119.36 should be in teldef 
       # value taken from Wayne Landsman's tools. 
       # note according to 2011 header inspection, was 241.06
       # while ugrism160 teldef lists 360-118.607/118.8/118.774 at various 
       # times, so this angle could be off by ~0.5 deg 
               
    
   def grism_boresight(self,order=1):
       # this is the first order boresight from the 2015 calibration
       # the teldef file lists the zeroth order boresight 
       if order == 1:
          return {200: [ 928.53,1002.69],
                  160: [1013.4 , 948.8 ],  # DET-IMG 
                 1000: [ 969.3 ,1021.3 ],
                  955: [1063.7 , 952.6 ]}[self.wheelpos]  
       elif order == 0:
           return {200: [1449.22, 707.7],
                   160: [1494.9 , 605.8], #[1501.4 , 593.7], # ?[1494.9, 605.8],
                  1000: [1506.8 , 664.3],
                   955: [1542.5 , 556.4]}[self.wheelpos]
   
   def grism_dist_0to1(self):
       x0,y0 = self.grism_boresight(order=0)
       x1,y1 = self.grism_boresight(order=1)
       return np.sqrt( (x1-x0)**2 + (y1-y0)**2 )

   def spec_curvature(self,xdet=None, ydet=None, order=1,):
       '''
     Find the coefficients of the polynomial for the curvature.   
   
     Parameters
     ----------
     xdet,ydet : DET coordinates (pix) of source, 
        if None then use anchor 
     wheelpos : int, {160,200,955,1000}
        grism filter position in filter wheel
     anchor : list, array
        anchor position in detector coordinates (pixels)
     order : int
        the desired spectral order  
         
     Returns
     -------
        Provides the polynomial coefficients for y(x). 
   
     Notes
     -----
     The curvature is defined with argument the pixel coordinate in the dispersion 
     direction with reference to the the anchor coordinates in det-img 
     coordinates. The polynomial returns the offset normal to the dispersion.
     
     - 2011-03-07 Paul Kuin, initial version 
     - 2011-08-02 fixed nominal coefficients order=1
       '''
       from scipy import interpolate
       from numpy import array
       
       wheelpos = self.wheelpos
       if order != 2: ox = 1
       else: ox = 2
       #offset = self.offset / self.pixelscale
       anchor = self.cal.anchor(sporder=ox, offsetdelta=self.offset)
       
       if self.chatter > 1: 
          print (type(anchor), anchor, self.offset)
       # convert det position to image coordinates
       if type(xdet) == type(None):
          xin = anchor[0] -104
          yin = anchor[1]  -78
       else:
          xin = xdet - 104
          yin = xdet -78
       if ((wheelpos == 1000) ^ (wheelpos == 955)):
          # return y = 0 + 0.0*x coefficient
          return array([0.,0.])

       elif wheelpos == 160:

           if order == 1:

                tck_c1= [array([0.,0.,0.,0.,2048.,  2048.,  2048.,  2048.]), \
          array([0.,0.,0.,0.,  2048.,  2048.,  2048.,  2048.]), \
          array([ 0.1329227 , -0.28774943,  0.13672294, -0.18436127, -0.19086855,\
          0.23071908, -0.21803703,  0.11983982,  0.16678715, -0.2004285 ,\
          0.12813155, -0.13855324, -0.1356009 ,  0.11504641, -0.10732287,\
          0.03374111]),3,3]
          
                tck_c2 = [array([0.,0.,0.,0.,  2048.,  2048.,  2048.,  2048.]),\
         array([0.,0.,0.,0.,  2048.,  2048.,  2048.,  2048.]),\
         array([ -3.17463632e-04,   2.53197376e-04,  -3.44611897e-04,\
         4.81594388e-04,   2.63206764e-04,  -3.03314305e-04,\
         3.25032065e-04,  -2.97050826e-04,  -3.06358032e-04,\
         3.32952612e-04,  -2.79473410e-04,   3.95150704e-04,\
         2.56203495e-04,  -2.34524716e-04,   2.75320861e-04,\
        -6.64416547e-05]),3,3]
       
                tck_c3 = [array([ 0.,0.,0.,0.,2048.,  2048.,  2048.,  2048.]),\
          array([ 0.,0.,0.,0.,2048.,  2048.,  2048.,  2048.]),\
          array([ -4.14989592e-07,   5.09851884e-07,  -4.86551197e-07,\
          1.33727326e-07,   4.87557866e-07,  -5.51120320e-07,\
          5.76975007e-07,  -3.29793632e-07,  -3.42589204e-07,\
          3.00002959e-07,  -2.90718693e-07,   5.57782883e-08,\
          2.20540397e-07,  -1.62674045e-07,   8.70230076e-08,\
         -1.13489556e-07]),3,3]
                     
                coef = array([interpolate.bisplev(xin,yin,tck_c3),
                     interpolate.bisplev(xin,yin,tck_c2),
                     interpolate.bisplev(xin,yin,tck_c1), 0.])
                return coef
       
           elif order == 2: 
               tck_c0 = [array([ 0., 0., 0., 0., 1134.78683, 2048., 2048., 2048., 2048.]), \
        array([ 0., 0., 0., 0., 871.080060, 2048., 2048., 2048., 2048.]), \
        array([-110.94246902,   15.02796289,  -56.20252149,  -12.04954456,\
        311.31851187,  -31.09148174,  -48.44676102,   85.82835905,\
        -73.06964994,   99.58445164,   46.47352776,   11.29231744,\
        -68.32631894,   88.68570087,  -34.78582366,  -33.71033771,\
          6.89774103,   25.59082616,   23.37354026,   49.61868235,\
       -438.17511696,  -31.63936231,   28.8779241 ,   51.03055925,\
         16.46852299]), 3, 3]

               tck_c1 = [array([    0.,     0.,     0.,     0.,  2048.,  2048.,  2048.,  2048.]),\
        array([    0.,     0.,     0.,     0.,  2048.,  2048.,  2048.,  2048.]),\
        array([ 0.52932582, -0.76118033,  0.38401924, -0.189221  , -0.45446129,\
        0.73092481, -0.53433133,  0.12702548,  0.21033591, -0.45067611,\
        0.32032545, -0.25744487, -0.06022942,  0.22532666, -0.27174491,\
        0.03352306]), 3, 3]

               tck_c2 = [array([    0.,     0.,     0.,     0.,  2048.,  2048.,  2048.,  2048.]),\
        array([    0.,     0.,     0.,     0.,  2048.,  2048.,  2048.,  2048.]),\
        array([ -4.46331730e-04,   3.94044533e-04,  -1.77072490e-04,\
         2.09823843e-04,   3.02872440e-04,  -6.23869655e-04,\
         5.44400661e-04,  -3.70038727e-04,  -1.60398389e-04,\
         4.90085648e-04,  -4.91436626e-04,   4.62904236e-04,\
         4.05692472e-05,  -2.34521165e-04,   3.04866621e-04,\
        -1.25811263e-04]), 3, 3]

               coef = array([interpolate.bisplev(xin,yin,tck_c2),
                     interpolate.bisplev(xin,yin,tck_c1),
                     interpolate.bisplev(xin,yin,tck_c0)])
               return coef
       
           elif order == 3: 
       # not a particularly good fit.
               tck_c0 =   [array([0.,     0.,  1101.24169141,  2048.,2048.]), 
         array([0.,     0.,   952.39879838,  2048.,2048.]), 
         array([ -74.75453915,    7.63095536, -131.36395787,   11.14709189,
         -5.52089337,   73.59327202,  -57.25048374,   37.8898465 ,
         65.90098406]), 1, 1]   
                
               tck_c1 = [array([    0.,     0.,  2048.,  2048.]), 
         array([    0.,     0.,  2048.,  2048.]), 
         array([-0.04768498, -0.02044308,  0.02984554, -0.04408517]), 1, 1]
 
               coef = array([interpolate.bisplev(xin,yin,tck_c1),
                      interpolate.bisplev(xin,yin,tck_c0)])             
               return coef
       
           elif order == 0:
               tck_c0 =         [array([    0.,     0.,  1075.07521348,  2048. ,2048.]),
                  array([    0.,     0.,  1013.70915889,  2048. ,2048.]),
                  array([ 130.89087966,   25.49195385,    5.7585513 ,  -34.68684878,
                          -52.13229007, -168.75159696,  711.84382717, -364.9631271 ,
                          374.9961278 ]),1,1]
               tck_c1 =         [array([    0.,     0.,  2048.,  2048.]),
                  array([    0.,     0.,  2048.,  2048.]),
                  array([ 0.08258587, -0.06696916, -0.09968132, -0.31579981]),1,1]
                  
               coef = array([interpolate.bisplev(xin,yin,tck_c1),
                      interpolate.bisplev(xin,yin,tck_c0)])            
               return  coef
           else: 
               raise (ValueError)    

       elif wheelpos == 200:
   
           if order == 1:
               tck_c1 = [array([    0.,     0.,     0.,     0.,  2048.,  2048.,  2048.,  2048.]),\
          array([    0.,     0.,     0.,     0.,  2048.,  2048.,  2048.,  2048.]),\
          array([-0.00820665, -0.06820851,  0.04475057, -0.06496112,  0.062989  , \
        -0.05069771, -0.01397332,  0.03530437, -0.17563673,  0.12602437,\
        -0.10312421, -0.02404978,  0.06091811, -0.02879142, -0.06533121,\
         0.07355998]), 3, 3]
        
               tck_c2 = [array([    0.,     0.,     0.,     0.,  2048.,  2048.,  2048.,  2048.]),\
          array([    0.,     0.,     0.,     0.,  2048.,  2048.,  2048.,  2048.]),\
          array([  1.69259046e-04,  -1.67036380e-04,  -9.95915869e-05, \
         2.87449321e-04,  -4.90398133e-04,   3.27190710e-04, \
         2.12389405e-04,  -3.55245720e-04,   7.41048332e-04, \
        -4.68649092e-04,  -1.11124841e-04,   6.72174552e-04, \
        -3.26167775e-04,   1.15602175e-04,   5.78187743e-04, \
        -8.79488201e-04]), 3, 3]

               tck_c3 = [array([    0.,     0.,     0.,     0.,  2048.,  2048.,  2048.,  2048.]),\
          array([    0.,     0.,     0.,     0.,  2048.,  2048.,  2048.,  2048.]),\
          array([  1.11106098e-07,   2.72305072e-07,  -7.24832745e-07,\
         4.65025511e-07,  -2.35416547e-07,  -3.87761080e-07,\
         1.05955881e-06,  -6.46388216e-07,   3.15103869e-07,\
         5.48402086e-07,  -1.44488974e-06,   6.52867676e-07,\
         1.14004672e-08,  -9.48879026e-07,   1.64082320e-06,\
        -8.07897628e-07]), 3, 3]

               coef = array([interpolate.bisplev(xin,yin,tck_c3),
                      interpolate.bisplev(xin,yin,tck_c2),
                      interpolate.bisplev(xin,yin,tck_c1), 0.])
               return coef
       
           elif order == 2: 
     
               tck_c0 = [array([0.,0.,   956.25596245,  2048.,2048.]),
        array([0.,0.,  1067.40622524,  2048.,2048.]),
        array([ 17.82135471,  -4.93884392,  20.55439437, -18.22869669,
        13.11429182,  41.2680039 ,   9.8050793 ,  32.72362507,  -6.56524782]), 1, 1]
        
               tck_c1 =  [array([    0.,     0.,  2048.,  2048.]),
        array([    0.,     0.,  2048.,  2048.]),
        array([ 0.02362119, -0.03992572,  0.0177935 , -0.10163929]),1, 1]
           
               tck_c2 =  [array([    0.,     0.,  2048.,  2048.]),
        array([    0.,     0.,  2048.,  2048.]),
        array([ -6.32035759e-05,   5.28407967e-05,  -8.87338917e-06, 8.58873870e-05]),1,1]

               coef = array([interpolate.bisplev(xin,yin,tck_c2),
                    interpolate.bisplev(xin,yin,tck_c1),
                    interpolate.bisplev(xin,yin,tck_c0)])
               return coef
       
           elif order == 3:  
     
               tck_c0 = [array([    0.        ,     0.        ,   807.44415249,  2048.,2048.]),
                  array([    0.        ,     0.        ,  1189.77686531,  2048.,2048.]),
                  array([-5436.10353688,   218.93823252,  -254.71035527,   -24.35684969,
                   23.26131493,    51.66273635,    37.89898456,    46.77095978,
                   63.22039872]), 1, 1]

               tck_c1 = [array([    0.,     0.,  2048.,  2048.]),
                 array([    0.,     0.,  2048.,  2048.]),
                 array([-0.02591263, -0.03092398,  0.00352404, -0.01171369]), 1, 1]
               coef = array([interpolate.bisplev(xin,yin,tck_c1),
                    interpolate.bisplev(xin,yin,tck_c0)])             
               return coef
       
           elif order == 0:
               tck_c0 = [array([0.,0.,   798.6983833,  2048.,  2048.]),
                   array([0.,0.,  1308.9171309,  2048.,  2048.]),
                   array([ 1244.05322027,    24.35223956,  -191.8634177 ,  -170.68236661,
            -4.57013926, 20.35393124, -365.28237355,  -235.44828185, -2455.96232688]), 1, 1]
               tck_c1 =  [array([    0.,     0.,  2048.,  2048.]),
                    array([    0.,     0.,  2048.,  2048.]),
                    array([ 0.54398146, -0.04547362, -0.63454342, -0.49417562]),1,1]

               coef = array([interpolate.bisplev(xin,yin,tck_c1),
                    interpolate.bisplev(xin,yin,tck_c0)])            
               return  coef

           else: 
               raise (ValueError)    
         
       else:
           print('spec_curvature: illegal wheelpos value')
           raise (ValueError)   


   def slit(self, xdet=None,ydet=None, sporder=1):
       """
       computes the coordinates of the slit in pixel coordinates (unrotated) 
       To position, rotate by the slope of the spectrum (counter clock wise)
          pivot over first order anchor
       
       Only correct for the UV grism, since the slit is shorter for the V grism.
       
       """
       #import calfiles
       # read the table of coefficients/get the coeeficients of the Y(dis) offsets and limits[]
       # stored with array of angles used. 
       
       # for the UV grism !
       
       #                  ZEROTH ORDER CURVATURE
       if sporder == 0:
           c = self.spec_curvature(xdet=xdet,ydet=ydet,order=0)
           x0= -820
           x1= -570
           sig = np.array([4.7])
       #                    FIRST ORDER CURVATURE
       if sporder == 1:
           c = self.spec_curvature(xdet=xdet,ydet=ydet,order=1)
           x0=-400
           x1=1150
           sig = np.array([-8.22e-09, 6.773e-04, 3.338])
       #                  SECOND ORDER CURVATURE
       if sporder == 2:
           c = self.spec_curvature(xdet=xdet,ydet=ydet,order=2)
           x0=25
           x1=3000
           sig = np.array([-5.44e-07, 2.132e-03, 3.662])
       #                  THIRD ORDER CURVATURE
       if sporder == 3:
           c =  self.spec_curvature(xdet=xdet,ydet=ydet,order=3)
           x0=425
           x1=3000
           sig = array([0.0059,1.5])

       x = np.array(np.arange(x0,x1+10,10))
       nx = len(x)
       y = np.zeros(nx) - np.polyval(c,0)+ np.polyval(c,x)
       z = np.zeros(nx) + 2.5 * np.polyval(sig,x)
       return x, y, z, c 

   def rotate_slit(self,x,y,pivot=None, theta=None):
      '''
      for example, this rotates the slit position to the detector position
      to get the slit width, compute (x,y+/-z) from self.slit
      
      The pivot needs to be given in the detector coordinates (deg) value. 
      Default is the target. 
      
      '''
      import numpy as np
      from astropy import units
      
      # require the source anchor/pivot point (pixels):
      if type(pivot) == type(None):
      # pixel offset of the anchor position from the detector 
      # coordinate center at 1100.5,1100.5 
         raise IOError("rotate_slit needs the anchor pivot input")
      # require the angle of the spectrum (quantity in deg)
      if type(theta) == type(None):
         raise IOError("the angle of the spectrum is required")
      xx = x # subtract not pivot since this is anchor-zero based already
      yy = y 
      # rotate slit 
      yd = pivot[0]+xx*np.sin(theta.to(units.rad).value)-yy*np.cos(theta.to(units.rad).value)
      xd = pivot[1]+xx*np.cos(theta.to(units.rad).value)+yy*np.sin(theta.to(units.rad).value)
      return xd, yd  

      
   def rotate_xy(self,x,y,angle, orig1):
      ''' rotate arrays of (x,y) over angle with respect to origin [x0,y0] (pix)
          angle value in deg
          origin x0,y0 in pix
          anti-clockwise rotation 
      '''
      import numpy as np
      theta = angle /180.0 * np.pi
      xp = x - orig1[0] 
      yp = y - orig1[1]
      xout = orig1[0]+xp*np.cos(theta)-yp*np.sin(theta)
      yout = orig1[1]+xp*np.sin(theta)+yp*np.cos(theta)
      return xout, yout
      
   def rotate_wcs(self,dsshdr,dssimg, ang, orig=None):
      """ 
      rotate the image array and  wcs independently over the same angle.
          get dss WCS cdelt and rotation
          given ra,dec convert x = ra/cos(dec), y = dec (intermediate coord system)
          apply rotation ang x',y' = A dot (x y) 
          convert to new x'', y'' = x'*cos(dec), y' 
          
          rewritten Feb 1 2021
      """
      import numpy as np
      from scipy import ndimage 
      from astropy import units
      from astropy import wcs
      from astropy.io import fits
      # first find center image: in pixels, in sky
      dssWcs = wcs.WCS (header=dsshdr,) 
      xcen,ycen = (dsshdr['NAXIS1'])*0.5-1, (dsshdr['NAXIS2'])*0.5-1
      poscen = dssWcs.pixel_to_world(xcen,ycen) 
      # these are the new reference CRPIX1,CRPIX2 and CRVAL1,CRVAL2      
      # the header of the input dsssimg has a CD rotation matrix but not CDELT* 
      # to align the image to the WCS axes
      a21 = dsshdr['cd2_1']
      a22 = dsshdr['cd2_2']
      a11 = dsshdr['cd1_1']
      a12 = dsshdr['cd1_2']
      det = a11*a22 - a12*a21
      sgn = det/np.abs(det)
      rot1 = sgn * np.arctan2(sgn*a12,sgn*a11)
      rot2 =       np.arctan2(-a21,a22)      # compare rot1, rot2 same to within 2 deg?
      angdss = rot1 / self.dtor * units.deg  # 
      cdelt1 = sgn*np.sqrt(a11*a11 + a12*a12)  # positive: right-handed, negative: left-handed
      cdelt2 =     np.sqrt(a21*a21 + a22*a22)  # always positive
      # assuming angtot and ang are for the same handedness add the angles (in radians)
      rottot = rot1 + ang.to(units.rad).value 
      cd11 =  cdelt1 * np.cos(rottot) 
      cd12 = -cdelt1 * np.sin(rottot)
      cd21 =  cdelt2 * np.sin(rottot)
      cd22 =  cdelt2 * np.cos(rottot)
      # the CDELT* keywords & the rotation angle theta can be found from CD matrix
      # there is no LONPOLE keyword, so it is the default
      # once we rotate the image over aan angle, these need to be updated. 
 
      # rotate the dssimage around center (positive angle gives a left-handed rotation)
      # pivoting around the image center (for dsssimg, [xcen,ycen] )
      imgout = ndimage.rotate(dssimg,-ang.to(units.deg).value,reshape = False,
             order = 1,mode = 'constant',cval = 0.1,prefilter=False)
      # make simple header 
      h = fits.PrimaryHDU(imgout) 
      hdr = h.header
      hdr['cd1_1'] = cd11
      hdr['cd1_2'] = cd12
      hdr['cd2_1'] = cd21
      hdr['cd2_2'] = cd22
      #hdr['lonpole'] = ang.to(units.deg).value ??
      hdr['crpix1'] = xcen
      hdr['crpix2'] = ycen
      hdr['crval1'] = poscen.ra.deg
      hdr['crval2'] = poscen.dec.deg
      hdr['RADESYS'] = ('ICRS              ' ,'GetImage: GSC-II calibration using ICRS system')
      hdr['CTYPE1']  = ('RA---TAN          ' ,'GetImage: RA-Gnomic projection')
      hdr['CUNIT1']  = ('deg               ' ,'GetImage: degrees')   
      hdr['CTYPE2']  = ('DEC--TAN          ' ,'GetImage: Dec-Gnomic projection')
      hdr['CUNIT2']  = ('deg               ' ,'Getimage: degrees')
      return hdr, imgout

      
   def det_frame(self,WCS=None):
       """
       compute the sky corner coordinates of the det frame [img coord]
       given the appropriate det WCS (i.e., the 'S' wcs in UVOT). 
       
       BE CAREFUL to apply the pixel image coordinate WCS
       
       """
       from astropy import coordinates, units
       if WCS == None: 
           raise RuntimeError(f"det_frame needs a valid wcs instance\n")
       x,y = WCS.all_pix2world(np.array([0,1986,1986,0,0]),
            np.array([0,0,2045,2045,0]),0)
       pos_ = coordinates.SkyCoord(x,y,frame='icrs',unit=(units.deg, units.deg))     
       return x,y,pos_
       

   def _det_header(self,):
      """
      this is the grism det file header (simulated) *_dt.img
      """
      from astropy.io import fits
      from astropy import units
      coef = """XTENSION= 'IMAGE   '           / IMAGE extension                                
BITPIX  =                  -32 / number of bits per data pixel                  
NAXIS   =                    2 / number of data axes                            
NAXIS1  =                 1987 / length of data axis 1                          
NAXIS2  =                 2046 / length of data axis 2                          
PCOUNT  =                    0 / required keyword; must = 0                     
GCOUNT  =                    1 / required keyword; must = 1                     
CRPIX1S =          1448.000000                                                  
CRPIX2S =           703.000000                                                  
CRVAL1S =     136.204166175583                                                  
CRVAL2S =    -32.4930169210235                                                  
CDELT1S = -0.000156666785871793                                                 
CDELT2S = 0.000156666785871793                                                  
PC1_1S  =    0.755670245086613                                                  
PC1_2S  =   -0.654951085758962                                                  
PC2_1S  =    0.654952042271387                                                  
PC2_2S  =    0.755671475100696                                                  
CTYPE1S = 'RA---TAN-SIP'                                                        
CTYPE2S = 'DEC--TAN-SIP'                                                        
CUNIT1S = 'deg     '           / X coordinate units                             
CUNIT2S = 'deg     '           / Y coordinate units                             
CRPIX1  =                996.5                                                  
CRPIX2  =               1021.5                                                  
CRVAL1  =                   0.                                                  
CRVAL2  =                   0.                                                  
CDELT1  =             0.009075                                                  
CDELT2  =             0.009075                                                  
CTYPE1  = 'DETX    '           / X coordinate type                              
CTYPE2  = 'DETY    '           / Y coordinate type                              
CUNIT1  = 'mm      '           / X coordinate units                             
CUNIT2  = 'mm      '           / Y coordinate units                            
A_ORDER =                    3                                                  
B_ORDER =                    3                                                  
A_1_0   =    -0.00125153527908                                                  
A_2_0   =   -1.21308092203E-05                                                  
A_1_1   =    3.57697489791E-06                                                  
A_0_2   =   -4.98655501953E-06                                                  
A_3_0   =   -2.23440999701E-10                                                  
A_2_1   =    2.81157465077E-10                                                  
A_1_2   =    1.07794901513E-09                                                  
A_0_3   =    1.81850672672E-09                                                  
B_0_1   =     -0.0119355520972                                                  
B_2_0   =    1.29190114841E-06                                                  
B_1_1   =   -6.22446958796E-06                                                  
B_0_2   =    6.50166571708E-06                                                  
B_3_0   =     1.5607230673E-09                                                  
B_2_1   =    3.10676603198E-09                                                  
B_1_2   =    1.83793386146E-09                                                  
B_0_3   =     3.0412214095E-12                                                  
AP_ORDER=                    3 / Polynomial order, axis 1, detector to sky      
BP_ORDER=                    3 / Polynomial order, axis 2, detector to sky      
AP_1_0  =     0.00125480395117                                                  
AP_0_1  =   -1.36411236372E-07                                                  
AP_2_0  =     1.2138698679E-05                                                  
AP_1_1  =   -3.57720222046E-06                                                  
AP_0_2  =    5.12067402118E-06                                                  
AP_3_0  =    5.04857662962E-10                                                  
AP_2_1  =   -4.41525720641E-10                                                  
AP_1_2  =   -8.91001063794E-10                                                  
AP_0_3  =   -2.06470726234E-09                                                  
BP_1_0  =    4.40624953378E-07                                                  
BP_0_1  =      0.0121093187715                                                  
BP_2_0  =   -1.42450854484E-06                                                  
BP_1_1  =    6.34534204537E-06                                                  
BP_0_2  =   -6.67738246399E-06                                                  
BP_3_0  =     -1.675660935E-09                                                  
BP_2_1  =   -3.07108005097E-09                                                  
BP_1_2  =   -2.02039013787E-09                                                  
BP_0_3  =    8.68667185361E-11                                                  
   """
      hdr = fits.Header.fromstring(coef,'\n') 
      hdr['CRVAL1S'] = self.pointing.ra.deg
      hdr['CRVAL2S'] = self.pointing.dec.deg
      hdr['CRPIX1S'], hdr['CRPIX2S'] = self.grism_boresight(order=0) # this is in IMG coordinate
      x = self.PA(self.roll.to(units.deg)).to(units.rad).value
      hdr['PC1_1S'] = np.cos(x)
      hdr['PC1_2S'] = np.sin(x)
      hdr['PC2_1S'] = -np.sin(x)
      hdr['PC2_2S'] = np.cos(x)
      return hdr
 
   def _lenticular_header(self,):
      """The cal.anchor input is the offset on the lenticular filter dx,dy 
      which, given the delta RA and delta Dec from the pointing/boresight 
      and lenticular filter WCS (which requires the roll angle) can be computed.
      pixel scale 0.502"/pix  
      """
      from astropy.io import fits
      from astropy import units
      coef = """XTENSION= 'IMAGE   '           / IMAGE extension                                
BITPIX  =                  -32 / number of bits per data pixel                  
NAXIS   =                    2 / number of data axes                            
NAXIS1  =                 2048 / length of data axis 1                          
NAXIS2  =                 2048 / length of data axis 2                          
PCOUNT  =                    0 / required keyword; must = 0                     
GCOUNT  =                    1 / required keyword; must = 1                     
CRPIX1  =          1023.500000                                                  
CRPIX2  =          1023.500000                                                  
CRVAL1  =     1.0       /placeholder                                            
CRVAL2  =    -1.0       /placeholder                                            
CDELT1  = -0.000139444444                                                       
CDELT2  = 0.000139444444                                                        
PC1_1   =    0.7556       /placeholder                                          
PC1_2   =   -0.6549       /placeholder                                          
PC2_1   =    0.6549       /placeholder                                          
PC2_2   =    0.7556       /placeholder                                          
CTYPE1  = 'RA---TAN'                                                            
CTYPE2  = 'DEC--TAN'                                                            
CUNIT1  = 'deg     '           / X coordinate units                             
CUNIT2  = 'deg     '           / Y coordinate units                             
   """
      hdr = fits.Header.fromstring(coef,'\n') 
      hdr['CRVAL1'] = self.pointing.ra.deg
      hdr['CRVAL2'] = self.pointing.dec.deg
      crpix1,crpix2 = boresight(filter='uvw2',r2d=0) # IMG coordinate 1030.23,1121.9
      hdr['CRPIX1'] = crpix1  
      hdr['CRPIX2'] = crpix2  
      x = -self.PA(self.roll.to(units.deg)).value/180.0*np.pi
      hdr['PC1_1'] = np.cos(x)
      hdr['PC1_2'] = -np.sin(x)
      hdr['PC2_1'] = np.sin(x)
      hdr['PC2_2'] = np.cos(x)
      return hdr
      
   def _scale_dss_header(self,):
      from astropy.io import fits
      from astropy import units
      coef = """XTENSION= 'IMAGE   '           / IMAGE extension                                
BITPIX  =                  -32 / number of bits per data pixel                  
NAXIS   =                    2 / number of data axes                            
NAXIS1  =                 1987 / length of data axis 1                          
NAXIS2  =                 2046 / length of data axis 2                          
PCOUNT  =                    0 / required keyword; must = 0                     
GCOUNT  =                    1 / required keyword; must = 1                     
CRPIX1  =          1448.000000                                                  
CRPIX2  =           703.000000                                                  
CRVAL1  =     136.204166175583                                                  
CRVAL2  =    -32.4930169210235                                                  
CDELT1  = -0.000156666785871793                                                 
CDELT2  = 0.000156666785871793                                                  
PC1_1   =    0.755670245086613                                                  
PC1_2   =   -0.654951085758962                                                  
PC2_1   =    0.654952042271387                                                  
PC2_2   =    0.755671475100696                                                  
CTYPE1  = 'RA---TAN'                                                        
CTYPE2  = 'DEC--TAN'                                                        
CUNIT1  = 'deg     '           / X coordinate units                             
CUNIT2  = 'deg     '           / Y coordinate units                             
   """
      hdr = fits.Header.fromstring(coef,'\n') 
      hdr['CRVAL1S'] = self.pointing.ra.deg
      hdr['CRVAL2S'] = self.pointing.dec.deg
      x = self.PA(self.roll.to(units.deg)).value/180.0*np.pi
      hdr['PC1_1'] = np.cos(x)
      hdr['PC1_2'] = np.sin(x)
      hdr['PC2_1'] = -np.sin(x)
      hdr['PC2_2'] = np.cos(x)
      return hdr

   def _kernel(self, ):
       """ 
    zeroth order kernel for smearing (DSS) sky image - one angle for all 
    positions, but different for uv and visible. 
    Needs refinement. Check scipy for normalisation, etc..
       """
       if self.wheelpos < 5000:
           return [[2,1,0],[1,2,1],[0,1,2]]
       elif self.wheelpos > 500:
           return [[2,1,0],[1,2,1],[0,1,2]]
       else: 
           return None

   def get_fademask(self, xaxis=1987, yaxis=2046, savedFile=None):
       """
    The grism image size is 2048x2048 large, but in the clocked modes
    part of the image does not receive zeroth order light due to the 
    aperture offset. After the distortion correction, the image is 
    no longer square.
    cp
    Parameters
    ==========
    wheelpos : int
       filter wheel position 
    xaxis,yaxis: int
       size of the image   
    savedFile : path
       this is the path to a previously computed veil. 
       If it is not found, it is computed and save to the given 
       path for future use. 
       
    Returns
    =======
    for nominal grism (wheelpos = 200, 1000): 1.0
    for clocked grism : an array of the same size as the grism detector image, 
    which  gives the fraction of the nominal sensitivity, so multiplying this 
    with a sky field image of the same size shows how the sensitivity changes 
    due to clocking.
    .    
    
    To some degree the mask can be used to rectify the flux in a clocked 
    grism image by dividing the image by the mask. 
    
       """
       from scipy.interpolate import splev
       import os
       from astropy.io import fits
    
       wheelpos = self.wheelpos
       if type(savedFile) != type(None):
           if os.access(savedFile,os.F_OK): 
               veilfile = fits.open(path)
               if wheelpos == 160: veil=veilfile['veil160'].data
               if wheelpos == 955: veil=veilfile['veil955'].data
               else: veil=1
               veilfile.close()
               return veil
       else:
               if wheelpos == 160: 
                   veil160 = make_fademask(160,)
               elif wheelpos == 955:     
                   veil955 = make_fademask(955,)
               f0 = fits.PrimaryHDU( )
               if wheelpos == 160: 
                   f1 = fits.ImageHDU(data=veil160 )
                   f1.header['extname'] = 'veil160'  
                   w = fits.HDUlist([f0,f1])
               if wheelpos == 955:     
                   f2 = fits.ImageHDU(data=veil955 )
                   f2.header['extname'] = 'veil955'   
                   w = fits.HDUlist([f0,f2])
               w.writeto(savedFile)
               if wheelpos == 160: return veil160
               if wheelpos == 955: return veil955
               else: return 1
            
   def make_fademask(self, xaxis=1987,yaxis=2046,savedFile=None):
       """
    Parameters
    ==========
    wheelpos : int
       filter wheel position 
    xaxis,yaxis: int
       size of the image   
       """
       from scipy.interpolate import splev

       if (self.wheelpos == 200) | (self.wheelpos == 1000):
           self.veil = 1.0
           return

       # spline fit to the radial drop in intensity - clocked grism
       tck160 = (np.array([  800.,   800.,   800.,   800.,   850.,   900.,   925.,   950.,
         1000.,  1050.,  1100.,  1125.,  1150.,  1200.,  1250.,  1300.,
         1400.,  1500.,  1600.,  1700.,  1800.,  1900.,  2000.,  2100.,
         2150.,  2175.,  2188.,  2200.,  2300.,  2399.,  2399.,  2399.,
         2399.]),
         np.array([ 0.99550037,  0.96029605,  0.978109  ,  0.9623139 ,  0.99642261,
         0.98371725,  1.01057958,  0.96852573,  0.96487821,  0.94143026,
         0.97711066,  0.97005785,  0.88281779,  0.85129301,  0.75284767,
         0.56998446,  0.34816299,  0.39798629,  0.38544318,  0.35581461,
         0.33544096,  0.29537848,  0.26501543,  0.24820651,  0.24447688,
         0.25430225,  0.29443737,  0.05526821,  0.07713518,  0.        ,
         0.        ,  0.        ,  0.        ]),
         3)
       # same, nominal grism  
       tck200 = (np.array([  800.,   800.,   800.,   800.,   900.,  1000.,  1100.,  1150.,
         1200.,  1300.,  1400.,  1600.,  1800.,  2000.,  2100.,  2150.,
         2175.,  2200.,  2399.,  2399.,  2399.,  2399.]),
         np.array([ 1.00845546,  0.98420151,  1.00602679,  1.02206876,  0.97150274,
         0.9653038 ,  0.95257463,  0.87784561,  0.76576549,  0.45008588,
         0.43946584,  0.28037786,  0.21283117,  0.1953306 ,  0.19120614,
         0.18514519,  0.16155574,  0.02569637,  0.        ,  0.        ,
         0.        ,  0.        ]),
         3)
    
    
       if  self.wheelpos == 160:
           i = np.arange(xaxis,dtype=int)  # 0..1986
           j = np.arange(yaxis,dtype=int)  # 0..2045
           r = np.sqrt((np.outer(i,np.ones(yaxis))-2200)**2+(np.outer(np.ones(xaxis),j)-400)**2)
           r = r.transpose()
           veil = np.ones([yaxis,xaxis],dtype=float)
        
           veil[r > 2300] = 1e-2
           for i1 in i:
              for j1 in j:
                  if (r[j1,i1] > 1000) & (r[j1,i1] < 2400):
                      veil[j1,i1] = splev(r[j1,i1],tck160,)
           veil[veil > 1] = 1           
           self.veil = veil
       elif self.wheelpos == 955:
           i = np.arange(xaxis)
           j = np.arange(yaxis)
           r = np.sqrt((np.outer(i,np.ones(yaxis))-2100)**2+(np.outer(np.ones(xaxis),j)-500)**2)
           r = r.transpose()
           veil = np.ones([yaxis,xaxis],dtype=float)
        
           veil[r > 2300] = 1e-2
           for i1 in i:
              for j1 in j:
                  if (r[j1,i1] > 700) & (r[j1,i1] < 2400):
                      veil[j1,i1] = splev(r[j1,i1],tck200,)
           veil[veil > 1] = 1           
           self.veil = veil
       else:
           raise IOError("{0} parameter error")
       
   def find_best_roll_and_offset(self,):
       """
       Optimising result (coding in progress )
       parameters...
       """ 
       import os
       from astropy import coordinates     
       from astropy import units
       from astropy.io import fits
       from astropy import wcs
       from astropy.vo.client import conesearch
       import calfiles
       from scipy import ndimage
        
       ra = self.ra
       dec = self.dec
           
       # image rotation needed from North (sky) to observed (i.e., DSS, DET pixel positions)                 
       PA = self.PA(self.roll) #240.65 * unit.deg - roll_angle.to(unit.deg) 

       # get first pointing (centre of image) from the offset and target position:     
       self.offset_pointing()
    
       hdr, dssimg = self.dsshdr, self.dssimg    
                 
       # get wcs of the DSS image 
       #  like, Wcss = wcs.WCS (header=hdr,key='S',relax=True,)                 
       Wcs_ = self.dssWcs 
                  
       # the following inherits from some draggable class
       # and some image manipulating class 
    
       # calibrated angle of the spectrum at an offset for the 
       # first order anchor position on the detector *** in det coordinates ***
       theta = self.cal.theta(offsetdelta=self.offset) 
       anker = self.cal.anchor(offsetdelta=self.offset,sporder=1)
    
       #       > update offset/roll > new pointing    
       # rotate image around pointing position
       a = dssimg # pivot around the pointing position, not the target position
       self.rotim = ndimage.rotate(a,PA - theta,reshape = False,order = 1,mode = 'constant',cval = cval)
       # crop to [yaxis,xaxis]

    # multiply by smearing kernel 
    # multiply by veil 
    # imshow, contour
    # select stars from list (including those off the image)
    #   who fall in a band around the target's dispersion.
    # sort star list by brightness
    # for target and each star near the target
    #   -highlight target [blink]
    #   -plot location zeroth order of the star (halo for bright ones)
    #   -retrieve first order curvature 
    #   -retrieve distance zeroth-first order
    #   -compute location of the first order 
    #   -plot the location of the first order
    # enable change of roll angle image, and of offset
    #   - grab target > offset change
    #   - grab corner image > roll angle change
       

def boresight(filter='uvw1',order=1,wave=260,
              r2d=77.0,date=0,chatter=0):
   ''' provide reference positions on the 
       UVOT filters for mapping and as function of 
       time for grisms. 
       
       This function name is for historical reasons, 
       and provides a key mapping function for the 
       spectral extraction.  
   
       The correct boresight of the (lenticular) filters 
       should be gotten from the Swift UVOT CALDB 
       as maintained by HEASARC. The positions here 
       are in some cases substantially different from
       the boresight in the CALDB. They are reference 
       positions for the spectral extraction algorithms 
       rather than boresight. 
       
       The grism boresight positions at 260nm (uv grism)
       and 420nm (visible grism) in first order are served
       in an uncommon format (in DET pixels) 
       by adding (77,77) to the lenticular filter 
       RAW coordinate.(see TELDEF file) the grism 
       boresight was measured in DET coordinates, 
       not RAW. (offset correction should be 104,78)

       Parameters
       ----------
       filter : str 
          one of {'ug200','uc160','vg1000','vc955',
          'wh','v','b','u','uvw1','uvm2','uvw2'}
       
       order : {0,1,2}
          order for which the anchor is needed

       wave : float
          anchor wavelength in nm

       r2d : float 
          additive factor in x,y to anchor position 

       date: long
          format in swift time (s)
          if 0 then provide the first order anchor 
          coordinates of the boresight for mapping 
          from the lenticular filter position 

       chatter : int 
          verbosity 

       Returns
       ------- 
       When *date* = 0:
       
       For translation: The boresight for a filter 
       (in DET pixels) by adding (77,77) to the 
       lenticular filter RAW coordinate (see TELDEF file)
       the grism boresight was measured in DET 
       (The default r2d=77 returns the correct 
       boresight for the grisms in detector 
       coordinates. To get the grism boresight in 
       detector image coordinates, subtract (104,78) 
       typically. The difference is due to the distortion
       correction from RAW to DET)
       
       When *date* is non-zero, and *order*=0:
       The zeroth order boresight  
      
          
       NOTE: 
       -----
       THE TRANSLATION OF LENTICULAR IMAGE TO GRISM 
       IMAGE IS ALWAYS THE SAME, INDEPENDENT OF THE 
       BORESIGHT.
       THEREFORE THE BORESIGHT DRIFT DOES NOT AFFECT 
       THE GRISM ANCHOR POSITIONS AS LONG AS THE DEFAULT 
       BORESIGHT POSITIONS ARE USED. 
       [Becase those were used for the calibration].

       However, the zeroth order "reference" position 
       drift affects the "uvotgraspcorr" - derived 
       WCS-S. The positions used 

       History: 
         2014-01-04 NPMK : rewrite to inter/extrapolate 
         the boresight positions
       
   '''
   from scipy.interpolate import interp1d
   import numpy as np
   
   filterlist = ['ug200','uc160','vg1000','vc955',
           'wh','v','b','u','uvw1','uvm2','uvw2']
   if filter == 'list': return filterlist
   grismfilters = ['ug200','uc160','vg1000','vc955']
   lenticular = ['v','b','u','uvw1','uvm2','uvw2']
   
   #old pixel offset anchor based on pre-2010 data
   # dates in swift time, drift [x.y] in pixels 
   #dates=[209952000,179971200,154483349,139968000,121838400]
   #drift=[ [0,0], [+2.4,-2.0], [+3.4,-3.0], [+6.4,-10], [+6.4,-10]]
   
   # data from Frank's plot (email 2 dec 2013, uvw1 filter)
   # original plot was in arcsec, but the drift converted 
   # to pixels. uvw1 seems representative (except for white)
   swtime = np.array([  
         1.25000000e+08,   1.39985684e+08,   1.60529672e+08,
         1.89248438e+08,   2.23489068e+08,   2.46907209e+08,
         2.66126366e+08,   2.79601770e+08,   2.89763794e+08,
         3.01251301e+08,   3.13180634e+08,   3.28423998e+08,
         3.43445470e+08,   3.59351249e+08,   3.75257678e+08,
         4.50000000e+08])
   boredx = (np.array([-1.6, -0.870,0.546,1.174,2.328,2.47,
        2.813,3.076,3.400,3.805,4.149,4.656,
        5.081,5.607,6.072,8.56 ])-1.9)/0.502
   boredy = (np.array([ -0.75,-2.197,-4.857,-6.527,
        -7.098,-7.252,-7.142,-7.560,
        -7.670,-8.000,-8.043,-8.395,
        -8.637,-9.142,-9.670,-11.9])+6.8)/0.502
   # I assume the same overall drift for the grism 
   # boresight (in pixels). Perhaps a scale factor for the 
   # grism would be closer to 0.56 pix/arcsec 
   # the range has been extrapolated for better interpolation
   # and also to support the near future. The early
   # time extrapolation is different from the nearly constant
   # boresight in the teldef but within about a pixel.
   # I think the extrapolation is more accurate.
   fx = interp1d(swtime,boredx,bounds_error=False,fill_value="extrapolate")
   fy = interp1d(swtime,boredy,bounds_error=False,fill_value="extrapolate")
   
   # reference anchor positions          
   reference0 = {'ug200': [1449.22, 707.7],
                 'uc160': [1494.9 , 605.8], #[1501.4 , 593.7], # ?[1494.9, 605.8],
                 'vg1000':[1506.8 , 664.3],
                 'vc955': [1542.5 , 556.4]} 
                  
   # DO NOT CHANGE THE FOLLOWING VALUES AS THE WAVECAL DEPENDS ON THEM !!!
   reference1 = {'ug200': [ 928.53,1002.69],
                 'uc160': [1025.1 , 945.3 ], 
                 'vg1000':[ 969.3 ,1021.3 ],
                 'vc955': [1063.7 , 952.6 ]}            
                          
   if (filter in grismfilters):
      if (date > 125000000) and (order == 0):
          anchor = reference0[filter]
          anchor[0] += r2d-fx(date)
          anchor[1] += r2d-fy(date)
          return anchor
      elif (date > 125000000) and (order == 1):   
          anchor = reference1[filter]
          anchor[0] += r2d-fx(date)
          anchor[1] += r2d-fy(date)
          return anchor
      elif order == 1:    
          anchor = reference1[filter]
          anchor[0] += r2d
          anchor[1] += r2d
          return anchor
      elif order == 0:  
          raise RuntimeError(
          "The zeroth order reference position needs a date")  
      else:
          return reference1[filter]       
                  
   elif (date > 125000000) and (filter in lenticular):
      ref_lent = {'v':[951.74,1049.89],
                  'b':[951.87,1049.67],
                  'u':[956.98,1047.84],
                  'uvw1':[951.20,1049.36],
                  'uvm2':[949.75,1049.30],
                  'uvw2':[951.11,1050.18]}
      anchor = ref_lent[filter]
      anchor[0] += r2d-fx(date)
      anchor[1] += r2d-fy(date)
      return anchor
      
   elif (date > 122000000) and (filter == 'wh'):
      print("approximate static white filter boresight")
      if date > 209952000:
         return 949.902+r2d, 1048.837+r2d         
      elif date > 179971200:
         return 953.315+r2d, 1048.014+r2d        
      elif date > 154483349:
         return 954.506+r2d, 1043.486+r2d
      elif date > 139968000:
         return 956.000+r2d, 1039.775+r2d
      elif date >  121838400:
         return 956.000+r2d, 1039.775+r2d      
      else: return filterlist

   else:
      # this is the version used initially *(changed 2 june 2009)
      # DO NOT CHANGE THESE VALUES AS THE WAVECAL DEPENDS ON THEM !!!
      if   filter == 'uvw1': return 954.61+r2d, 1044.66+r2d
      elif filter == 'wh'  : return 954.51+r2d, 1043.49+r2d
      elif filter == 'v'   : return 955.06+r2d, 1045.98+r2d 
      elif filter == 'b'   : return 955.28+r2d, 1045.08+r2d 
      elif filter == 'u'   : return 960.06+r2d, 1043.33+r2d
      elif filter == 'uvm2': return 953.23+r2d, 1044.90+r2d 
      elif filter == 'uvw2': return 953.23+r2d, 1044.90+r2d
      elif filter == 'w1'  : return 954.61+r2d, 1044.66+r2d
      elif filter == 'm2'  : return 953.23+r2d, 1044.90+r2d 
      elif filter == 'w2'  : return 953.23+r2d, 1044.90+r2d
      elif filter == 'ug200':       
          if order == 1:
             if wave == 260: return 928.53+r2d,1002.69+r2d
      elif filter == 'uc160':       
          if order == 1:
             if wave == 260: return 1025.1+27+r2d,945.3+r2d
      elif filter == 'vg1000': 
          #elif order == 1: return 948.4+r2d, 1025.9+r2d
          if order == 1: return 969.3+r2d, 1021.3+r2d
      elif filter == 'vc955':
          if order == 1: return 1063.7+r2d, 952.6+r2d
         
   raise IOError("valid filter values are 'wh','v',"\
        "'b','u','uvw1','uvm2','uvw2','ug200',"\
        "'uc160','vg1000','vc955'\n")    
       
"""
probably do the following: 

Starting from dssWcs find the reference pixel position from the detector boresight sky position
Make the WCS for the detector with the borresight as the reference poiunt which now will match
a modified dssWcs#2 with the reference point matching the det. boresight. 
Now use that as the projection wcs in the add_axes() 

no, just transform the x,y to ra,dec using the transform self.detposition2sky ?

In [128]: S.dssWcs 
Out[128]: 
WCS Keywords

Number of WCS axes: 2
CTYPE : 'RA---TAN'  'DEC--TAN'  
CRVAL : 70.594625  -9.906138888888888  
CRPIX : 2695.863409896894  1149.8938874191672  
PC1_1 PC1_2  : 0.02528383053472523  -9.83906748150961e-05  
PC2_1 PC2_2  : 9.852028659179484e-05  0.02528478609657265  
CDELT : -0.018675847380390118  0.018675847380390118  
NAXIS : 953  953

corresponding WCS of detector image needed


# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
#       PLOT --- first attempt that failed
# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
   def plot_dss(self,):
        import numpy as np
        import matplotlib.pyplot as plt
        from astropy.io import fits, ascii
        from astropy import wcs, coordinates, units
        #from reproject import reproject_interp
        #from grism_utilities import query_DSS
        #from uvotpy import generate_USNOB1_cat
        #from scipy import ndimage   # rotate the image array and  wcs independently.
        from matplotlib import cm

        circang = np.arange(0,2.*np.pi,0.2)
        circle1 = np.sin(circang)
        circle2 = np.cos(circang)
        
        self.anker = self.cal.anchor(offsetdelta=self.offset,sporder=1) - np.array([77,77])
        self.anker2 = self.cal.anchor(offsetdelta=self.offset,sporder=2) - np.array([77,77])
        ank_dist12 = np.sqrt((self.anker[0]-self.anker2[0])**2+(self.anker[1]-self.anker2[1])**2)
        ank_dist = 610. # zero-first order !approximate - this varies by [-20,+20] pix over detector
        
        dssimg = self.dssimg
        wcs1 = self.dssWcs
        wcs2 = self.detWcs
        nx,ny = dssimg.shape
# scale intensity DSS image
        xr = int(nx*0.3), int(nx*0.7)
        yr = int(ny*0.3), int(ny*0.7)
        subimg = dssimg[xr[0]:xr[1],yr[0]:yr[1]]
        top = dssimg[xr[0]:xr[1],yr[0]:yr[1]].mean()+5.*dssimg[xr[0]:xr[1],yr[0]:yr[1]].std()
        # clip
        subimg[subimg > subimg.mean()+3.*subimg.std() ] = subimg.mean()+3.*subimg.std()
        hot = dssimg > top
        dssimg[hot] = top +3
        bg = subimg.mean()+4.*subimg.std() # source must be 4 sigma above background
        dssimg = dssimg - bg
        dssimg[dssimg <= 0] =  np.min(dssimg[dssimg > 0])+0.1
        # use reproject to scale to right size for det:
        #img, dssfootprint = reproject_interp( (dssimg, self.dsshdr), 
        #      self._scale_dss_header() )

        rot_dsshdr, img = self.rotate_wcs(self.dsshdr,dssimg,self.PA(self.roll), orig=None) # rotate wcs independently from image.
        self.rotated_img = img 
        self.rotated_hdr = rot_dsshdr
        self.wcs3 = wcs.WCS(header=rot_dsshdr)
        scale2dss = self.dsshdr['pltscale']/ (self.dsshdr['XPIXELSZ']) # "/pix =[arcsec/mm]/[pix in ?microns]

        print (f"anti-clockwise rotated DSS sky image by angle of {self.PA(self.roll)} deg\n")
        #self.dss_footprint = dssfootprint     
        x, y, z, c = self.slit()  # first order
        x0, y0, z0, c0 = self.slit(sporder=0)  # zeroth orderr
        xd,yd = self.rotate_slit(x,y) 
        xd,ydm = self.rotate_slit(x,y-z) 
        xd,ydp = self.rotate_slit(x,y+z)  
        x0d,y0d = self.rotate_slit(x0,y0) 
        
# UB1 source positions and magnitudes self.ub1ra, self,ub1dec, self.ub1b2mag, self.ub1r2mag
# image coordinates on dss image self.ub1_dssx, self.ub1_dssy       
       # get initial DSS image for pointing 
       #self.dsshdr, self.dssimg = query_DSS(self.pointing, ImSize=23.0, server="STSCI",
       #              version = "3", output=storeDSS, chatter=0)
        fig1 = plt.figure()
        self.ax1 = fig1.add_axes([0.09,0.10,0.90,0.89],projection=self.wcs3) # rotate axes by PA ???
        self.ax1.imshow( np.log(img), aspect='auto',  cmap=cm.inferno, origin='lower')
        self.ax1.coords.grid(True, color='white', ls='solid')
        #self.ax1.coords[0].set_axislabel('RA')
        #self.ax1.coords[1].set_axislabel('Dec')
#        ax1.text(0.15, 0.15, self.idstring, horizontalalignment='center',
#           verticalalignment='center', transform=ax.transAxes,color='c')
        x9,y9,pos1 = self.det_frame(self.detWcs)
        self.ax1.plot_coord(pos1,'k',lw=2,linestyle='dashed')
        
        px0, py0 = self.rotate_xy(x0d,y0d,self.PA(self.roll),[0,0]) # rotate from DET to Sky-pix
        x9,y9 = self.detWcs.all_pix2world(np.array(px0),np.array(py0),0)
        pos2 = coordinates.SkyCoord(x9,y9,frame='icrs',unit=(units.deg, units.deg))
        self.ax1.plot_coord(pos2,'*k',markersize=5)

        fig2 = plt.figure()
        self.ax2 = fig2.add_axes([0.09,0.10,0.90,0.89],)#projection=wcs2) 
#       2.5 sigma wide slit in detector coordinates: 
        #self.ax2.plot(xd,yd,'b--')
        self.ax2.plot(xd,ydm,'b',lw=1)
        self.ax2.plot(xd,ydp,'b',lw=1)
        self.ax2.plot(x0d,y0d,'b',lw=3)
        # get now the zeroth order positions of field stars
        px, py = self.detWcs.all_world2pix(self.ub1ra, self.ub1dec,0,
             unit=[units.deg,units.deg],frame=coordinates.ICRS) # unrotated in pix coord.
        #orig = self.grism_boresight(order=0) # it is centered on CRPIX?S 
        #(perhaps it is more accurate to use 
        # boresight(filter= xxx, order=0, wave = xxx, r2d=77.0, date=xxxx)
        # not [self._det_header()['CRPIX1S'],self._det_header()['CRPIX2S']], or without 'S'
        rpx,rpy = px,py #self.rotate_xy(px,py,self.PA(self.roll),orig)
        ub1ra,ub1dec=self.ub1ra,self.ub1dec
        q1 = (rpx < 2150) & (rpy > -150) & (self.ub1b2mag < 9.5) & (rpx > -150) & (rpy < 2150)
        q2 = (rpx < 2150) & (rpy > -100) & (self.ub1b2mag >= 9.5) & (rpx > -150) & (rpy < 2150) 
        for a,b,b2 in zip(rpx[q1],rpy[q1],self.ub1b2mag[q1]):
            circr = 30.-b2
            self.ax2.plot(a+circr*circle1,b+circr*circle2,lw=1,color='darkred',alpha=1)
        for n1,m1,p1,p2 in zip(rpx[q2],rpy[q2],ub1ra[q2],ub1dec[q2]):
           self.ax2.text(n1,m1,"%.3f,%.3f"%(p1,p2),fontsize=8)
        for a,b in zip(rpx[q1],rpy[q1]):
           circr = 32. # radius of bright source (16") exclusion zone in pixels
           self.ax2.plot(a+circr*circle1,b+circr*circle2,'k',lw=1)
        # other sources: PSF = 3" = 6 pixels  [ignore trailed shape in disp direction] 
        for a,b,in zip(rpx[q2],rpy[q2]):
           circr = 6. # radius of bright source exclusion zone in pixels
           self.ax2.plot(a+circr*circle1,b+circr*circle2,color='darkred',lw=1)
        self.ax2.plot([0,1987,1987,0,0],[0,0,2046,2046,0],'k--',lw=1)
        self.ax2.plot(x0d,y0d,'m',lw=1)
        

# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = 
#        end plot
# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = 
"""