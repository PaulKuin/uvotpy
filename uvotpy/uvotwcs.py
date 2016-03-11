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
''' code to help with coordinate transformations. '''

try:
  from uvotpy import uvotplot,uvotmisc,uvotwcs,rationalfit,mpfit,uvotio
except:
  pass  
#import uvotgetspec as uvotgrism

def makewcshdr(filestub, ext, attfile, 
     indir="./", 
     teldef=None, 
     wheelpos=None,
     continue_when_graspcorr_fails=True, 
     catspec=None, 
     uvotgraspcorr_on=True,
     update_pnt=True, 
     chatter=1):
   '''make the header of a lenticular filter for a grism image
      to use in uvotapplywcs
      writes a file with a bogus image
      returns the name of the file
      
      Parameters
      ----------
      filestub : str
        identifying part of filename being "sw"+`obsid`
      ext : int 
        extension of fits file to process
      attfile : str
        attitude file name. Needs to be set when update_pnt set.
      indir : str
        path, directory of data files
      teldef : str
        filename `teldef` file for epoch of anchor calibration
      wheelpos : int, {160,200,955,1000}
        filterwheel position for grism
      continue_when_graspcorr_fails : float
        this would supply a solution - though quite bad
      uvotgraspcorr_on: bool
        if not set, then the original pointing is used 
	after optional update when update_pnt was set 
      update_pnt : bool	
	allows updating the header RA_PNT,DEC_PNT,PA_PNT keywords 
	using the atttitude file (which is required)
      catspec : path
        path to catalog spec file other than default	
      chatter : int
        verbosity
	
      Returns
      -------
      creates a fake sky file with appropriate header to run `findInputAngle`	
      
      Notes
      -----
      need to update the tstart and tstop of the primary header (not a showstopper)

   '''
   #
   # 2013-10-02 npmk correct error of assuming reference point graspcorr is boresight
   # 2013-06-25 npmk rewrite: base faked lenticular file on RA,DEC,ROLL from the aspect corrected grism image only 
   # 2010-12-01 npmk modified call to use the teldef file for the observation time (the default)
   #                 added time keywords to header copy
   # 2010-05-27 npmk added environment variable UVOTPY for finding input file
   # 2009-08-20 npmk changed to fixed boresight of calibration 
   # 2009-07-14 npmk changed swiftxform call to use the *_PNT keywords from the file.
   # 2009-07-23 npmk changed call swiftxform to use attitude file if given or approximate 
   #                 when attfile=None; apply if graspcorr='yes'
   from os import getenv,system
   import numpy as np
   try:
      from astropy.io import fits
      from astropy import wcs
   except:   
      import pyfits as fits
      import pywcs as wcs
   from uvotgetspec import boresight   
   
   msg = ""   
   if chatter > 0:
     print "makewcshdr(",filestub,',',ext,',',attfile,',indir=',indir,')'

   try:
      caldb = getenv('CALDB')
      distfiledir = caldb+'/data/swift/uvota/bcf/grism/'  
   except:
      raise IOError("CALDB environment variable has not been defined"+
         " - aborting at uvotwcs.makewcsheader() " )
     
   try:
      pydata = getenv('UVOTPY')
      uvw1dummy = pydata+'/calfiles/uvw1_dummy.img' 
      catspecdir = pydata+'/calfiles/'
   except: 
      print "UVOTPY environment variable has not been defined"   
      home = getenv('HOME') 
      uvw1dummy = home+'/pydata/uvw1_dummy.img'
      if not os.access(uvw1file,os.F_OK):
         raise RuntimeError("The UVOTPY environment variable has not been set or install error.")
      catspecdir=home+"/dev/uvotpy.latest/calfiles/"
      distfiledir=caldb+"/data/swift/uvota/bcf/grism/"
              
   # define which teldef file was used in the anchor calibration 
   # as needed by swiftxform to create similar uvw1 wcs file
   
   #   need to convert to caldb query like: 
   # quzcif swift uvota - VGRISM GRISMDISTORTION 2009-10-30 12:00:00 -
   # this teldef was that used for the time of calibration and is needed for the fake file
   # for getting the pointing from the grism graspcorr, the correct dated boresight is needed. 
   if teldef == None:
      if wheelpos == 160:
         teldef = '/data/swift/uvota/bcf/teldef/swuw120070911v002.teldef'
	 bore = np.array([1501.4, 593.7]) # [1503.5,596.5]) # 2009-02-19
	 band='uc160'
      if wheelpos == 200:
         teldef = '/data/swift/uvota/bcf/teldef/swuw120070911v001.teldef'
	 bore = np.array([1449.0, 703.5]) # [1446.0,710.0]) 
	 band='ug200'
      if wheelpos == 955:
         teldef = '/data/swift/uvota/bcf/teldef/swuw120070911v001.teldef'   
	 bore = np.array([1567.0, 534.7]) # [1560.0, 543.0] ) # 2009-09-28 comparing w1-gr-w1
	 band='vc955'  
      if wheelpos == 1000:
         teldef = '/data/swift/uvota/bcf/teldef/swuw120070911v001.teldef'
	 bore = np.array([1506.8, 664.3]) # [1504.5,670.0] ) # 2009-09-29 from comparing w1-gr-w1 
	 band='vg1000'
   if chatter > 2:
       print "teldef: "+teldef
       print "bore=",bore
       print "band="+band 	 
   
   if ((wheelpos == 160) ^ (wheelpos == 200)):
       grismfile = indir+'/'+filestub+'ugu_dt.img'
   elif ((wheelpos == 955) ^ (wheelpos == 1000)):
       grismfile = indir+'/'+filestub+'ugv_dt.img'
   if catspec == None: 
       catspec=catspecdir+"/usnob1.spec"
   uvw1file = indir+'/'+filestub+'ufk_rw.img'
   uvw1filestub = indir+'/'+filestub+'ufk'
   wcsfile  = indir+'/'+filestub+'ufk_sk.img'
   ranstr = ''
   command = 'cp '+uvw1dummy+' '+uvw1file
   if chatter > 2: 
      print "command: ",command
   if system( command ) != 0:
      print "uvotwcs: cannot create a dummy lenticular file "
      print "perhaps missing ?: "+uvw1dummy
      raise RuntimeError("Aborting: Cannot create dummy file")
   #
   #  Update the grism file header (RA,DEC_PA)_PNT parameters using the attitude file (option)
   #
   if update_pnt & (attfile != None):
       hdr_upd = get_distortion_keywords(wheelpos)
       fh = fits.open(grismfile ,mode="update")
       hdr=fh[int(ext)].header
       hdr.update(hdr_upd)
       fh[int(ext)].header['CTYPE1S']='RA---TAN-SIP'
       fh[int(ext)].header['CTYPE2S']='DEC--TAN-SIP'
       tstart = fh[int(ext)].header['tstart']
       tstop = fh[int(ext)].header['tstop']
       if chatter > 2: 
          print "initial header update grism file: "
	  print " "
	  fh[int(ext)].header
       roll=hdr['pa_pnt']
       if attfile != None:
           status, ra_pnt, dec_pnt, roll = get_pointing_from_attfile(tstart,tstop,attfile)
           if status == 0: 
               fh[int(ext)].header['ra_pnt'] = ra_pnt
               fh[int(ext)].header['dec_pnt']= dec_pnt
               fh[int(ext)].header['pa_pnt'] = roll	
	       msg += "updated header RA_PNT=%10.5f,DEC_PNT=%10.5f,roll=%8.1f"%(ra_pnt,dec_pnt,roll)    
	   if chatter > 2: 
	       print "further header updates"       
       fh.close()          
   #
   #  Always run uvotgraspcorr to get the corrected RA, DEC, ROLL. 
   #   
   if uvotgraspcorr_on:
      try:
          msg += "applying uvotgraspcorr\n "
          # reset 
          fh = fits.open(grismfile,mode="update")
          fh[int(ext)].header['aspcorr'] = 'NONE'
          fh.close()
	  if chatter > 2: print 'ASPCORR keyword reset to NONE'
          # find aspcorr
          command="uvotgraspcorr infile="+grismfile+" catspec="+catspec+\
          " distfile="+distfiledir+"/swugrdist20041120v001.fits "+\
          " outfile=attcorr.asp  clobber=yes chatter="+str(chatter) 
          #"  distfile="+distfiledir+"/swugrdist20041120v001.fits \
          if chatter > 0: print "command: ",command
          system(command)

          # find the zero order boresight reference point on the detector
	  # corresponding to the date of observation. Note that these 
	  # are dirrerent from those in the CALDB as used by uvotimgrism
          newhead = fits.getheader(grismfile,ext)
          roll = newhead['PA_PNT']
	  wS =wcs.WCS(header=newhead,key='S',relax=True,)
	  bore = boresight(filter=band,order=0,r2d=0,date=newhead['tstart'])
	  world = wS.all_pix2world([bore],0)[0]
	  
	  if chatter>0:
	      print "WCS pointing  "
	      print "filter band = "+band
	      print "boresight = ", bore
	      print "sky world coordinate pointing = ",world             
          if newhead["ASPCORR"].upper() != "GRASPCORR": 
              print "UVOTGRASPCORR did not find a valid solution ***********************************"
              print "UVOTGRASPCORR did not find a valid solution * wavelength scale offset warning *"
              print "UVOTGRASPCORR did not find a valid solution ***********************************"
	      msg += "UVOTGRASPCORR did not find a valid solution"
	      if not  continue_when_graspcorr_fails: 
	          raise RuntimeError (
		  "uvotgraspcorr failed to find a solution in call uvotwcs.makewcshdr")
	      # copy the unmodified  values : it is not smart to update the attitude 
	      #      with a bad attitude correction
	      ra_pnt = newhead['RA_PNT']
	      dec_pnt = newhead['DEC_PNT']
          else:
	      # use the new values found after success with uvotgraspcorr
	      if len(world) > 0: 
	          ra_pnt,dec_pnt = world
                  roll = newhead['PA_PNT']  # perhaps update from attcorr.asp ? 
		  msg += "updated pointing using corrected WCS-S keywords"
		  print "updated pointing using corrected WCS-S keywords"
		  attfile = None #  		  
	      else:	  
                  # get the updated ra,dec,roll values from the attcorr.asp table
		  # though I am not sure how good these values actually are 
                  # => check multi-extension files
                  system("ftlist attcorr.asp t colheader=no rownum=no columns=ra_pnt,dec_pnt,pa_pnt > attcorr.txt") 
                  f = open("attcorr.txt")
                  rec = f.readlines()
                  if len(rec) < (ext-1) : 
                      print "makewcsheader: not enough records in attcorr.txt to account for number of extensions."
                  ra_pnt, dec_pnt, roll = rec[ext-1].split()
                  f.close()
                  if chatter > 2: 
                     print "records from attcorr.asp:",rec
	             print "extracted from record: ",ra_pnt, dec_pnt, roll
                  else:  	   
                     system("rm attcorr.txt")
                  # now apply the aspect correction and replace the attitude file
                  if attfile == None:
                     if chatter > 2: 
  	                 print "no attitude file correction applied; using pa_pnt, dec_pnt, ra_pnt from attcorr.asp"	 
                  else:	 
                      command="uvotattcorr attfile="+attfile+" corrfile=attcorr.asp  outfile="+\
	                  filestub+".gat.fits chatter=5 clobber=yes"
                      if chatter > 0: print command
                      if system(command) == 0:
	                  # replace the attitude file for the following
                          attfile=filestub+".gat.fits"
      except:
          print ":-( uvotwcs.makewcshdr: perhaps uvotgraspcorr failed"
	  if continue_when_graspcorr_fails: 
              pass
	  else: 
	      raise RuntimeError ("uvotgraspcorr probably failed in call uvotwcs.makewcshdr")   
   else:
       g_hdr   = fits.getheader(grismfile,ext)   	 
       ra_pnt  = g_hdr['RA_PNT']
       dec_pnt = g_hdr['DEC_PNT']
       roll    = g_hdr['PA_PNT']
   #
   # copy header keywords from grism to wcsfile precursor to use for transforms.
   #
   g_hdr  = fits.getheader(grismfile, ext)
   d_list = fits.open(uvw1file,mode='update')
   m_hdr = d_list[0].header
   d_hdr = d_list[1].header
   _ukw(g_hdr,d_hdr,'TSTART')
   _ukw(g_hdr,d_hdr,'TSTOP')
   _ukw(g_hdr,d_hdr,'TIMESYS')
   _ukw(g_hdr,d_hdr,'MJDREFI')
   _ukw(g_hdr,d_hdr,'MJDREFF')
   _ukw(g_hdr,d_hdr,'TELAPSE')
   _ukw(g_hdr,d_hdr,'ONTIME')
   _ukw(g_hdr,d_hdr,'LIVETIME')
   _ukw(g_hdr,d_hdr,'EXPOSURE')
   _ukw(g_hdr,d_hdr,'OBS_ID')
   _ukw(g_hdr,d_hdr,'TARG_ID')
   _ukw(g_hdr,d_hdr,'PA_PNT')
   _ukw(g_hdr,d_hdr,'OBJECT')
   _ukw(g_hdr,d_hdr,'RA_OBJ')
   _ukw(g_hdr,d_hdr,'DEC_OBJ')
   d_hdr[ 'RA_PNT' ] = ra_pnt
   d_hdr['DEC_PNT' ] = dec_pnt
   d_hdr[ 'PA_PNT' ] = roll
   m_hdr[ 'RA_PNT' ] = ra_pnt
   m_hdr['DEC_PNT' ] = dec_pnt
   m_hdr[ 'PA_PNT' ] = roll
   d_hdr[ 'EXTNAME' ] = 'W1999999999I'
   d_hdr[ 'CREATOR'] = 'uvotwcs.uvotwcshdr.py version of 2013-10-26'
   d_hdr[ 'ORIGIN' ] = 'uvotwcs MSSL/Calibration intermediate file ' 
   #d_list[0].header.update(m_hdr)
   d_list[1].header.update(d_hdr)
   d_list.close()   

   if (attfile == None) | (update_pnt):  
   # perform transform based on header and pointing from running uvotgraspcorr.
       if chatter > 1: print "makewcsheader: using pointing provided by uvotgraspcorr"
       command="swiftxform infile="+uvw1file+" outfile="+wcsfile+" attfile=CONST:KEY " \
         +" alignfile=CALDB method=AREA to=SKY "\
	 +" ra="+str(ra_pnt)+" dec="+str(dec_pnt)+" roll="+str(roll)+" teldeffile=CALDB " \
         +" bitpix=-32 zeronulls=NO aberration=NO seed=-1956 copyall=NO "\
	 +" extempty=YES allempty=NO clobber=yes"
   else:  
   # attitude file given and not update_pnt set (nt sure about accuracy results)
       if chatter > 1: print "makewcsheader : using uvotgraspcorr + (updated) attitude file "
       command="swiftxform infile="+uvw1file+" outfile="+wcsfile+" attfile="+attfile \
         +" alignfile=CALDB method=AREA to=SKY ra=-1 dec=-1 roll=-1 teldeffile="+caldb+teldef \
         +" bitpix=-32 zeronulls=NO aberration=NO seed=-1956 copyall=NO "\
	 +" extempty=YES allempty=NO clobber=yes"

   if chatter > 0: print command  
   if system(command) == 0: 
       return wcsfile
   else:  
       raise RuntimeError( "uvotwcs.makewcsheader: error creating corresponding sky file - aborting " ) 


def _ukw(inhrd,outhdr,keyword):
   '''updates the outhdr keyword with the value from the inhdr keyword '''
   tmp =  inhrd[ keyword ]
   outhdr[ keyword ] = tmp  

def _WCCS_imxy2radec(header,Ximg,Yimg):
   ''' convert image positions to ra and dec using WCS-S keywords in header 
   
   INPUT
       header = fits header to get WCS keywords from
       Ximg = numpy array of x positions
       Yimg = numpy array of y positions
   
   OUTPUT numpy array with [RA,DEC] positions corresponding to [X,Y]
   '''
   return []
     
def _WCSSsky2xy(file,ext,RA,DEC):
   '''Find the image coordinates [X,Y] of a source with known 
   sky position (RA, DEC) in degrees. 
   '''
   return []
     
def _WCSS_radec2imxy(header,RA,DEC):
   ''' convert Sky positions (ra,dec) in degrees to image coordinates using 
       WCS-S keywords in header 
   
   INPUT
       header = fits header to get WCS keywords from
       RA  = numpy array of RA  values
       DEC = numpy array of DEC values
   
   OUTPUT numpy array with [X,Y] positions corresponding to [RA,DEC]
   '''
   return []
   
def correct_image_distortion(x,y,header):
   ''' This routine applies to the (x,y) position on the image the 
       grism distortion correction
       input header must be from the grism image (position) to be corrected '''  
   import numpy as N
   
   good = ('AP_ORDER' in header) & ('BP_ORDER' in header)
   if not good :
      #print "WARNING uvotwcs.correct_image_distortion found no distortion keywords in header"
      return x,y
   
   # distance to zeroth order anchor 
   xdif = x - header['crpix1s']
   ydif = y - header['crpix2s']
    
   AP_ORDER = N.int(header['AP_ORDER'])       #   3 
   BP_ORDER = N.int(header['BP_ORDER'])       #   3  
   
   AP = N.zeros([AP_ORDER+1,AP_ORDER+1],dtype=N.float64)                                               
   BP = N.zeros([BP_ORDER+1,BP_ORDER+1],dtype=N.float64)  
                                                
   AP[0,1] = AP_1_0  = header['AP_1_0']       #    -0.0119355520972  
   AP[1,0] = AP_0_1  = header['AP_0_1']       #    -0.0119355520972  
   AP[0,2] = AP_2_0  = header['AP_2_0']       #  -1.42450854484E-06 
   AP[1,1] = AP_1_1  = header['AP_1_1']       #   6.34534204537E-06 
   AP[2,0] = AP_0_2  = header['AP_0_2']       #  -6.67738246399E-06 
   AP[0,3] = AP_3_0  = header['AP_3_0']       #    -1.675660935E-09 
   AP[1,2] = AP_2_1  = header['AP_2_1']       # -3.07108005097E-09  
   AP[2,1] = AP_1_2  = header['AP_1_2']       #  -2.02039013787E-09 
   AP[3,0] = AP_0_3  = header['AP_0_3']       #   8.68667185361E-11 
   
   BP[0,1] = BP_1_0  = header['BP_1_0']    
   BP[1,0] = BP_0_1  = header['BP_0_1']       #    -0.0119355520972  
   BP[0,2] = BP_2_0  = header['BP_2_0']       #  -1.42450854484E-06 
   BP[1,1] = BP_1_1  = header['BP_1_1']       #   6.34534204537E-06 
   BP[2,0] = BP_0_2  = header['BP_0_2']       #  -6.67738246399E-06 
   BP[0,3] = BP_3_0  = header['BP_3_0']       #    -1.675660935E-09 
   BP[1,2] = BP_2_1  = header['BP_2_1']       # -3.07108005097E-09  
   BP[2,1] = BP_1_2  = header['BP_1_2']       #  -2.02039013787E-09 
   BP[3,0] = BP_0_3  = header['BP_0_3']       #   8.68667185361E-11 

   # perform the transforms from RA, DEC to image X,Y   
   
   xdif1 = 0. ; ydif1 = 0
   
   for i in range(4):
      for j in range(4):
         if AP[i,j] != 0.0:
            xdif1 = xdif1 + xdif**i*ydif**j*AP[j,i]            
         if BP[i,j] != 0.0:
            ydif1 = ydif1 + xdif**i*ydif**j*BP[j,i]

   #print 'The distortion correction x and y differences are : ',xdif1,ydif1

   x = xdif + xdif1 + header['crpix1s']
   y = ydif + ydif1 + header['crpix2s'] 
   
   return x,y 

def get_pointing_from_attfile(tstart,tstop,attfile):
   try:
      from astropy.io import fits
   except:
      import pyfits as fits   
   from numpy import abs
   
   status = 0 # success
   att = fits.open(attfile)
   try:
      t    = att[('ATTITUDE',1)].data['time']
      ra   = att[('ATTITUDE',1)].data['pointing'][:,0]
      dec  = att[('ATTITUDE',1)].data['pointing'][:,1]
      roll = att[('ATTITUDE',1)].data['pointing'][:,2]
   except:
      status = 1 # fail
      return status,-1,-1,-1   
   q = (t > tstart) & (t < tstop)
   t = t[q]
   ra = ra[q]
   dec = dec[q]
   roll = roll[q]
   if len(t) == 0:
      status = -1 # failure
      return status, -1, -1, -1
   else:
      q = abs(ra - ra.mean())  - 3.0*ra.std()  < 0
      ra_pnt = ra[q].mean()
      q = abs(dec - dec.mean()) - 3.0*dec.std()  < 0   
      dec_pnt = dec[q].mean()
      q = abs(roll - roll.mean()) - 3.0*roll.std()  < 0   
      roll_pnt = roll[q].mean()
      return status, ra_pnt, dec_pnt, roll_pnt

def get_distortion_keywords(wheelpos):
   '''provide the grism header with distortion keywords '''
   import os
   try:
      from astropy.io import fits
   except:
      import pyfits as fits
   
   if wheelpos < 500:
       command="quzcif swift uvota - VGRISM GRISMDISTORTION 2009-10-30 12:00:00 - > quzcif.out"
       name = 'UGRISM_%04d_DISTORTION'%(wheelpos)
   else:    
       command="quzcif swift uvota - UGRISM GRISMDISTORTION 2009-10-30 12:00:00 - > quzcif.out"
       name = 'VGRISM_%04d_DISTORTION'%(wheelpos)
   print name   
   os.system(command)
   f = open("quzcif.out")
   distfile = f.read().split()[0]
   f.close()
   os.system("rm -f quzcif.out")
   fdist = fits.open(distfile)
   head = fdist[name].header
   hdr = head['?_ORDER']
   hdr.update(head['A_?_?'])
   hdr.update(head['B_?_?'])
   hdr.update(head['?P_ORDER'])
   hdr.update(head['AP_?_?'])
   hdr.update(head['BP_?_?'])
   return hdr



