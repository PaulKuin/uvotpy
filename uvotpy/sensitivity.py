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
# =====================================================
import numpy as np
import os, sys
from uvotpy import uvotmisc,uvotio
from astropy.io import fits

def get(uvotfilter, time, timekind='swifttime', 
        wave=None, sens_rate=0.01, wheelpos=0, 
        chatter=0):

   """
   The Swift UVOT sensitivity over time 
   
   Photometry: the Swift zeropoints are fixed at the start of the mission in 2005, so 
       the count rates have to be corrected for sensitivity loss in the detector 
       before converting to fluxes/magnitudes.
       
   Spectroscopy: The grism effective area has been specified also at the start of the 
       mission in 2005 and need to be corrected for sensitiivity loss in the detector.  
   
   The  sensitivity is computed only for one time, uvotfilter
         
   Input parameters: 
      uvotfilter: one of set below
      time : float 
      timekind: one of ['UT','swifttime','MJD']
      wave: array [grism]
      sens_rate : float 
         basic rate of decay for whole grism
      wheelpos: int 
         position in filter wheel to identify filter/mode (grism only)  
         
   Requires Swift CALDB configured       
         
   2020-08-07 NPMK fecit
   """
   bands = {'v':'v','b':'b','u':'u','uvw1':'uvw1','uvm2':'uvm2','uvw2':'uvw2',
     'w1':'uvw1','m2':'uvm2','w2':'uvw2', 'wh':'white', 'white':'white',
     'ugc':'ug160','vgc':'vgc990','ug160':'ug160','vg990':'vg990',
     'ugn':'ug200','vgn':'vg1000','un200':'ug200','vg1000':'vg1000',     
   }
   if uvotfilter.lower() in bands:
      current = bands[uvotfilter.lower()]
   else:
      raise IOError(f"invalid filter band name {uvotfilter}\n")    
      
   caldb = os.getenv("CALDB")
   
   # convert time (accuracy better than 1 minute)
   UT = None
   if timekind == 'UT':
      UT = time 
   if timekind == 'MJD':
      approx_swifttime = (time-51910.0)*86400. 
      UT = uvotmisc.swtime2JD(approx_swifttime,useFtool=False)[3]
   if timekind == 'swifttime':
      UT = uvotmisc.swtime2JD(time)[3]
   UTd,UTt = UT.split('T')
   yr,mon,day = UTd.split('-')
   hr,min,s = UTt.split(':')   
   yr, mon, day = np.int(yr), np.int(mon), np.int(day)
   hr, min, s = np.int(hr), np.int(min), np.int(float(s))
   swtime = uvotmisc.UT2swift(yr,mon,day,hr,min,np.int(s),0)   
      
   # photometry:
   if uvotfilter.lower() in ['v','b','u','white','uvw1','uvm2','uvw2']: 
       
      command = "quzcif swift uvota - "+current.upper()+" SENSCORR "+\
         UT.split('T')[0]+"  "+\
         UT.split('T')[1]+" - > sfile_.tmp"
      if chatter > 0:
          sys.stderr.write("shell command: "+command+"\n") 
      try:
         os.system(command)
      except:
         sys.stderr.write("sensitivity.get: quzcif ERROR reading cal file\n...") 
         raise IOError(f"ERROR reading calibration \n==>{command}\n")
         
      f = open('sfile_.tmp')
      sfile = f.readline()
      f.close()
      os.system("rm sfile_.tmp")
      if len(sfile) > 0: 
         try:
            tab = fits.getdata(sfile.split()[0],ext=int(sfile.split()[1]))
         except:
            raise IOError(f"cannot open file: {sfile} at time {time,UT}\n")   
         
         # columns in SENSCORR file are named TIME. OFFSET, SLOPE   
         # multiply count rate C as follows:
         #      C_new = C * (1 + OFFSET) * (1 + SLOPE)**DT
         k = np.where(swtime > tab['TIME'])[-1]
         dt = (swtime - tab['TIME'][k])/(86400.*365.26) # (in s/yr)
         senscorr = (1.0 + tab['OFFSET'][k]) * (1.0 + tab['SLOPE'][k])** dt 
         if chatter > 4: 
            print (f"selected indices {k} \n")
            print (f"dt = {dt}\n senscorr = {senscorr}\n")
            print (f"senscorr[-1] = {senscorr[-1]}")
         return senscorr[-1]
      else:
          if chatter > 0:
             print (f"WARNING - no sensitivity file found.\n=>Assuming no sensitivity loss.\n")
          return 1.0
   elif uvotfilter.lower() in list(bands.keys())[10:]:  # grism modes 
      return uvotio.sensitivityCorrection(swtime,wave=None,sens_rate=0.01,wheelpos=0)
   else:
      raise IOError(f"invalid filter band name {uvotfilter}\n")    
   #
