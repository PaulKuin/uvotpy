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
import os, sys
import optparse
import numpy as np
from astropy.io import fits

__version__ = "0.9.0"



"""
  search tree of data and compile basic information of all images/event files

  Items extracted are:

  date-obs tstart exposure filter aspcorr filename+ext datamode tossloss stalloss  frametime
"""
out = []  # global output file buffer
      
if __name__ == '__main__':

   status = 0
   
   if status == 0:
      usage = "usage: %prog [options] <base_directory>"

      epilog = '''
      The base directory assumes the basic Swift directory structure
      starting with on obsid like:
      
      00012345001/uvot/image/sw00012345001u*
                      /event/  
      
      unless the option --nodir is given                
      ''' 

      anchor_preset = list([None,None])
      bg_pix_limits = list([-100,-70,70,100])
      bg_lower_ = list([None,None])  # (offset, width) in pix, e.g., [20,30], default [50,50]
      bg_upper_ = list([None,None])  # (offset, width) in pix, e.g., [20,30], default [50,50]

      parser = optparse.OptionParser(usage=usage,epilog=epilog)
      
      # options

      parser.add_option("", "--nodir", dest = "nodir",
                  help = "If True only process files in the local directory [default: %default]",
                  default = False)

      #parser.add_option("", "--ra", dest = "ra",
      #            help = "RA (deg) [default: %default]",
      #            default = -1.0)
   
   (options, args) = parser.parse_args()

   if (len(sys.argv) == 1):
      # no arguments given 
      program_name = sys.argv[0]
      print (program_name," processing all directories")
      status = 1


   if (len(args) > 0):
       #        parser.print_help()
       parser.error("Incorrect argument(s) found on command line: "+str(args))

   nodir = options.nodir
   
   main(nodir)
   
   
def main(nodir=False,chatter=0,sortkey='time',filename=None,
         include_zip=False,mjdtimezero=None):   

   if include_zip:
      outheader = "%17s %16s %6s %6s %8s %31s %1s %7s %7s %4s %8s\n"%("date-obs","MJD","exposr","filter", 
      "aspcorr", "filename+ext", "M", "%tossloss", "%stalloss","roll","Tframe")
      fmt = "%17s %16.5f %6.1f %6s %8s %31s %1s %7.2f %7.2f %7.1f %8.6f\n"  
   else:
      outheader = "%17s %16s %6s %6s %8s %27s %1s %7s %7s %4s %8s\n"%("date-obs","MJD","exposr","filter", 
      "aspcorr", "filename+ext", "M", "%tossloss", "%stalloss","roll","Tframe")
      fmt = "%17s %16.5f %6.1f %6s %8s %27s %1s %7.2f %7.2f %7.1f %8.6f\n"  
   out = []
   valid = ['vv','bb','uu','w1','m2','w2','wh']
   valtyp = ['sk']
   validg = ['gu','gv']
   valtypg = ['dt']

   if nodir:
      files = os.listdir('.')
      if chatter > 1: print ("nodir:", files)
      for xx in files: 
          if len(xx) > 23:
              if (xx[:2] == 'sw') & (xx[14:16] in valid) & (xx[17:19] in valtyp):
                  _write(out,fmt,xx)
   else:
      base = os.listdir('.')
      if chatter > 1: print ("base :",base)
      for b in base:
         if chatter > 1: print ("b:",b)
         if (len(b) == 11) & os.path.isdir(b) :
            if chatter > 1: print ("len=11, isdir:")
            if b.isdigit(): 
               if chatter > 1: print ("isdigit")
               b2 = os.listdir(b)
               if chatter > 1: print ('b2: ',b2)
               if 'uvot' in b2:
                  b3 = os.listdir(b+'/uvot')
                  if chatter > 1: print (b3)
                  if 'image' in b3: 
                      files = os.listdir(b+'/uvot/image/')
                      if chatter > 1: print ("files= ",files)
                      for xx in files: 
                          if len(xx) > 22:
                              if (xx[:2] == 'sw') & (xx[14:16] in valid) & (xx[17:19] in valtyp):
                                  if chatter> 3: print ("_write call: ", b+'/uvot/image/'+xx)
                                  try:
                                      _write(out,fmt,b+'/uvot/image/'+xx,include_zip=include_zip) 
                                  except:
                                     sys.stderr.write("problem with %s\n"%(b+'/uvot/image/'+xx))
                                     pass   
                              if (xx[:2] == 'sw') & (xx[14:16] in validg) & (xx[17:19] in valtypg):
                                  if chatter> 3: print ("_write call: ", b+'/uvot/image/'+xx)
                                  try:
                                     _write(out,fmt,b+'/uvot/image/'+xx,include_zip=include_zip )
                                  except:
                                     sys.stderr.write("problem with %s\n"%(b+'/uvot/image/'+xx))
                                     pass   
                  #elif 'event' in b3:
                  #    files = os.listdir(b+'/uvot/event/')
                  #    for xx in files: 
                  #        if len(xx) > 23:
                  #            if (xx[:2] == 'sw') & (xx[14:16] in valid):
                  #                _write_event(out,fmt,path)
   
   if chatter > 2: print ('out: ',out)
   # assign 'record number ' index
   tind = np.arange(len(out)) 
   if sortkey == 'time': 
      times = np.empty(len(out),dtype=[('tind',int),('time',float)])   
      times['tind'] = tind        
      for k in tind:
         times['time'][k] = out[k][18:35]
      # sort on times   
      times = np.sort(times,order='time')
      if chatter > 2: print ('times: ',times)
   elif (sortkey == 'frametime') | (sortkey == 'framtime') :
      times = np.empty(len(out),dtype=[('tind',int),('ftime',float)])   
      times['tind'] = tind        
      for k in tind:
         if include_zip:
             times['ftime'][k] = float(out[k][116:124])
         else: 
             times['ftime'][k] = float(out[k][112:120])
      # sort on times   
      times = np.sort(times,order='ftime')
      if chatter > 2: print ('frame times: ',times)
         
   else:
      if chatter > 0: print ('unsorted data' )  
                  
   if filename == None:               
       sys.stdout.write(outheader)
       for k in times['tind']:
           sys.stdout.write(out[k])  
   else:                     
       f = open(filename,'w')
       f.write(outheader)
       for k in times['tind']: f.write(out[k]) 
                              
def swtime2JD(TSTART,useFtool=False):
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
   if useFtool:
      import os
      from numpy.random import rand
      delt,status = swclockcorr(TSTART)
      if not status: print("approximate time correction ")
      return swtime2JD(TSTART+delt,useFtool=False)
   else:
      import numpy as np
      delt = datetime.timedelta(0,TSTART,0)
      # delt[0] # days;   delt[1] # seconds;  delt[2] # microseconds
      swzero_datetime = datetime.datetime(2001,1,1,0,0,0)
      gregorian = swzero_datetime + delt
      MJD = np.double(51910.0) + TSTART/(24.*3600)
      JD = np.double(2451910.5) + TSTART/(24.*3600)
      outdate = gregorian.isoformat()
   return JD, MJD, gregorian, outdate



def _write(out,fmt,path,include_zip=False,chatter=0):
   # assumes an uvot image file header 
   f = fits.open(path)
   N = len(f)
   for ext in np.arange(1,N): 
      filen = _formatFilename(path,ext,include_zip=include_zip)
      x = _getHdrStuff(f[ext].header)
      tstart = x['tstart']
      xx = swtime2JD(tstart)
      mjd = xx[1]
      if chatter > 3 : print ("xx mjd ",xx, mjd)
      out.append( fmt%( x['dateobs'],      
         mjd,x['expo'],x['filt'], 
         x['aspcorr'],filen,x['datamode'],
         x['tossloss'],x['stalloss'],x['roll'],x['ftime'])     
      )
      
def _getHdrStuff(hdr):
   return  dict(   
      datamode = hdr['extname'][-1],
      dateobs  = hdr['date-obs'][:16],
      tstart   = hdr['tstart'],
      expo     = hdr['exposure'],
      filt     = hdr['filter'],
      roll     = hdr['PA_PNT'],
      aspcorr  = hdr['aspcorr'][:6],
      tossloss = 100.*hdr['tossloss']/hdr['exposure'],
      stalloss = 100.*hdr['stalloss']/hdr['exposure'],
      ftime    = hdr['framtime'],)
      
def _formatFilename(path,ext,include_zip=False):
   if include_zip:
     return path.rsplit('/')[-1]+"+"+str(ext)
   else:
     return path.rsplit('/')[-1].split('.gz')[0]+"+"+str(ext)

# copyright N.P.M. Kuin 2015   
