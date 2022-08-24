#!python
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
# To Do:
# 2020-07-14 Add the Shift-and-Add correction to the positioning of smaller windows
#            Add the shift caused by the pointing offset before settling 
#            Change code to fix HWWindow keywords in raw files
#            Add correct sensitivity loss (now 1%/yr)
# 2020-08-05 Get sensitivity loss from CALDB
# 2020-08-07 Add calling from the OS using optparse; encapsulating print/write with chatter
#

from __future__ import division
import numpy as np
from astropy.io import ascii,fits
from astropy import units
from astropy import coordinates as coord
import astropy
import sys,os
from uvotpy import sensitivity,uvotmisc
from uvotpy.convert_raw2det import radec2pos,radec2det,from_det_to_raw
from matplotlib import cm
import optparse

bands =['uwh','uw2','um2','uw1','uuu','ubb','uvv']
filts = ['wh','uvw2','uvm2','uvw1','u','b','v'] 
maxmag = {"wh":9.}

def photometry(obsid,
       target='target',
       radec=None,
       interactive=False,
       outfile='streak_photometry',
       do_only_band=None,
       rerun_readout_streak=False,
       snthresh=6.0, 
       timezero=None,
       datadir='.',
       figno=20,
       use_python_only=False,
       chatter=1):
   '''
   do the readout streak extraction in Swift UVOT images 
   
   input parameter
   ===============
   ***obsid*** : str
      e.g., '00032911013' for the observation
   ***target*** : str
      name target
   ***interactive*** : bool
      if True, select the row for computing an LSS correction      
      if False, no LSS correction will be done
   ***datadir***: path
      the path of the directory containing the image files   
   ***outfile*** : path
      the path of the file to append a summary of the selected sources to.
      When choosing an absolute path, multiple observations can be 
      done this way.  
   ***radec*** : list , optional
      give the sky position ra,dec (J2000/ICRS) in units of 
      decimal degrees, e.g., [305.87791,+20.767536]
   ***snthresh*** : float
      Signal to noise thresholt for readout streak detection above 
      background. Default=6.   
   ***do_only_band*** : string
      one of ['uvw2','uvm2','uvw1','u','b','v']
      select only this filter (multiple values not allowed) 
      The white filter cannot be done.
   ***timezero*** : long int, string, or astropy.time???
         the time to subtract either in swift time 
         (seconds since 2005-01-01T00:00:00.0 as a floating point number),
         or in the format "2014-02-01T01:12:13.0" as a string   
   ***figno*** : int or None    
         default = 20 
   ***use_python_only** : bool
      if the c-code readout_streak from Mat Page is not available, set to True      
         
   output
   ======
   - command line output of all readout streaks
   - writes a file with the python dictionary
   - appends magnitude and error with time and obsid+extension number to 'magfile'   
      
   Notes
   =====
   Required: - download the swift uvot auxil, uvot/hk, and uvot/image/ data. 
             - unzip the *_rw.img files if they are compressed, prior to running the code.
             - install the c-program readout_streak which can be downloaded from 
               http://www.mssl.ucl.ac.uk/www_astro/uvot/ 
   
   Sources which are over the brightness limit for which the UVOT readout 
   streak can be used for photometry can be recognized by posession of the 
   one or more of the following characteristics:     
      
      - very large halo of scattered light (> 60" radius)
      - cross of scattered light due to the mounting of the secondary
      - one-sided bright half-circle offset from the halo centre of the 
        source
      - multiple bright readout streaks right next to one another but due to 
        that one source (weak, bright, weak is ok)
   Of course, if the exposure time is small, these features may be 
   unrecognisable as such.
   
   When the most noticeable feature around the source is a round halo with 
   one bright readout streak, possibly weak streaks alongside, and a small 
   "smoke ring" of scattered light somewhere nearby, the source is usable        
   for this method.     
   
   Sources which are too faint can be reduced using the "uvotsource" "ftool". 
   
   How to use:
     - unpack the uvot data (auxil, uvot/hk, uvot/image)
     - unzip the uvot/image files
     - run the code like:
       > obsid = '00032911090'
       > ra,dec=305.87791,20.767536
       > indir = '../00033024012/uvot/image/' 
       
       I like to run from a "results" directory, which explains the indir parameter.
       > Z = readout_streak.readout_streak(obsid,radec=[ra,dec],target='V339 Del',
           magfile='/projects/V339_del/readout_photometry.txt')
       First the program will crop small-frame images that are in a large frame.           
       The program will reprocess the data to apply the MOD8 correction to the 
       raw image. Then run the c-program readout_streak written by Mat Page 
       which should have been installed. This program extracts the readout 
       streak positions and count rate.
       Then the program will pop up a window with the raw image, readout streaks 
       overlaid, The location of the source is indicated by a purple circle 
       but gives two possible locations [this is a bug that still needs investigation].
       Select the correct location. If the locations do not have a readout streak 
       (when the source has gotten too weak,) then click anywhere but type "S[kip]"
       and this image will not be used. Otherwise, type "Y[es]", to approve, 
       or anything else to have another go (the image does not update though). 
       
       If there are many images, this may take a while. 
       
       At the end, the results are written to the file specified in the input 
       parameter.
   
   History:
     2014-09-03 t0 06 npmk Added support for converting from Sky to RAW coordinates.
          Added Fits output. Changed informative print out to stderr, added 
          verbosity parameter. Change LSS to not be interactive when told. 
          Require RA, Dec or specidy "without". 
     2020-07-25 npmk rewrite: always both fits and ascii output; will use ra,dec provided, or 
     override if interactive; new ascii output format; works on full images frametime=11ms. 
     2020-08-07 NPMK upgrade to Python3 (drop Python 2 support)
   Bugs/desired upgrades:
    - the code only works correctly if run inside the directory/folder with the images
    - include the readout_streak c code in the distribution (requires fitsio and
      wcstools or cython.) 
     2020-08-19 NPMK wrote (slow) pure python implementation of Mat Page's readout_streak
      code, module ros.py. 
   
   '''
   import os
   import sys
   import numpy as np
   from astropy.io import fits
   import astropy
   
   magfile = outfile
   syserr = 0.1
   # test for presence CALDB and Swift Ftools
   CALDB = os.getenv('CALDB')
   if CALDB == '': 
      sys.stderr.write( 'WARNING: The CALDB environment variable has not been set\n')   
   HEADAS = os.getenv('HEADAS')
   if HEADAS == '': 
      sys.stderr.write('WARNING: The HEADAS environment variable has not been set\n'+\
           'which is needed for the calls to Swift Ftools\n')
   
   # require ra,dec, or go for interactive pick of source
   if (radec == None):
      raise IOError("The sky position is required for accurate magnitudes.\n"+
      "set radec = 'without' to do an interactive pick.")
   if radec != 'without':   
       if type(radec) == astropy.coordinates.sky_coordinate.SkyCoord:
            ra,dec = radec.ra.deg, radec.dec.deg
            sys.stderr.write("astropy.coord position ra=%f, dec=%f\n"%(ra,dec))
       elif type(radec) != list: 
             radec = 'without'
             sys.stderr.write( 
            "Since radec parameter is not a list [ra,dec], nor an astropy coordinate.\n"\
                "    Interactive pick of source required.\n")
       else:   
            ra,dec = radec
   else:
       radec = 'without' 
       
   # prepare to write both a fits and text output: strip expension off 
   if len(magfile) == 0:
      magfile = './streak_photometry'   
   if len(magfile.rsplit('.',1)) > 1:
      if magfile.rsplit('.',1)[1] == 'fit' or magfile.rsplit('.',1)[1] == 'fits': 
         magfile = magfile.rsplit('.',1)[0]
      elif magfile.rsplit('.',1)[1] == 'txt':    
         magfile = magfile.rsplit('.',1)[0]
   fitsout = magfile+'.fit'
   textout = magfile+'.txt'   
   # 
   # open or create fits file 
   if os.access(fitsout,os.F_OK):  
      # exists: append the new photometry
      magff = fits.open(fitsout,mode="update")
   else:
      # new table 
      magff = _init_fitsoutput(fitsout,chatter=chatter)

   # open or create text output file 
   if not os.access(textout, os.F_OK): #write header first time opening
       magfh = open(textout,'w')
       magfh.write("MJD mag(Vega) magerr syserr filter tstart date-obs obsid+ext lss\n") 
   else: 
       magfh = open(textout,'a')

   # run through all bands and locate the available raw image files 
   bands =['uwh','uw2','um2','uw1','uuu','ubb','uvv']
   filts = ['wh','uvw2','uvm2','uvw1','u','b','v'] 
   rawfiles = []  # list of raw files to process 
   result = []

   if do_only_band != None: 
       k_band = np.where( np.array(filts) == do_only_band )[0][0]
       bands = [bands[k_band]]
   for b in bands:
     filename = datadir+'/sw'+obsid+b+'_rw.img'
     skfilename = datadir+'/sw'+obsid+b+'_sk.img'
     sys.stderr.write("looking for "+filename+"\n")
     if os.access(filename, os.F_OK):
        rawfiles.append(filename)
     elif os.access(filename+'.gz', os.F_OK):
        os.system('gunzip '+filename+'.gz')
        rawfiles.append(filename)
     if os.access(skfilename+'.gz', os.F_OK):   
        os.system('gunzip '+skfilename+'.gz')
   if len(rawfiles) == 0:
      sys.stderr.write(f"No raw files for <obsid> found in the current directory.")
      return
   sys.stderr.write(f"Processing raw files found: {rawfiles}")

   # check the small mode image size is consistent with naxis; update header
   # 
   # >>> NEED ADD POINTING DECENTER AND SHIFT AND ADD CORRECTION TO WINDOW POSITION
   #  OR, FIND ACTUAL POSITION WINDOW IN ARRAY

   for rf in rawfiles:
     
     obses=[]
     #
     os.system("cp "+rf+" "+rf+".new") # work starting from a copy of the2 file
     #   This needs (1) nothing to do for full window, (2) check where window is, 
     #   (3) fix hwwindow keywords in header 
     #
     fix_hwwindow_header(rf, fiximage_extension=False, chatter=chatter)  
        
     # define file name variables for bp, md, sk 
     
     rf1 = rf.split('rw')[0]
     band = rf1[-4:-1]  # filter id     
     md = rf1+'md.img'
     bp = rf1+'bp.img'
     sk = rf1+'sk.img'
     if chatter > 0:
        sys.stderr.write(
        "filenames:\n   raw %s\n   bad points %s\n   mod8 corrected %s\n   sky %s\n"%
        ( rf, bp,md,sk))
     
     # reprocess to obtain the MOD8 corrected raw file. 
     # We use the mod8 file to measure readout streak.
     
     if not os.access(bp, os.F_OK):
        command = "uvotbadpix infile="+rf+".new"+" badpixlist=CALDB outfile="+\
        bp+" compress=YES clobber=yes history=yes chatter="+str(np.min([chatter,5]))
        if chatter > 0:
           sys.stderr.write("executing in shell: " +command )
        os.system(command)   
              
     if not os.access(md, os.F_OK):
        command = "uvotmodmap infile="+rf+".new"+" badpixfile="+bp+" outfile="+md+\
        " mod8prod=NO mod8file=CALDB nsig=3 ncell=16 subimage=NO "+\
        " xmin=0 xmax=2047 ymin=0 ymax=2047 clobber=yes history=yes chatter="\
        +str(np.min([chatter,5])) 
        if chatter > 0:
            sys.stderr.write("executing in shell: " +command )
        os.system(command)          

     # run Mat's readout_streak c program on the mod8 corrected file
     #  try the c-code implementation first
     resfile = "results."+band+"_.txt"
     if (not os.access(resfile, os.F_OK)) or rerun_readout_streak: 
         command = 'readout_streak infile='+md+' snthresh='+str(snthresh)+' > '+resfile
         if chatter > 0:     
             sys.stderr.write( "calling readout_streak main c program:\n"+command)
         # if already present only rerun if parameter is set
         if not use_python_only:
            command = 'readout_streak infile='+md+' snthresh='+str(snthresh)+' > '+resfile
            if chatter > 0:     
               sys.stderr.write( "calling readout_streak main c program:\n"+command)
            os.system(command)
         else: 
            import ros
            matros = ros.ros(infile=md,outfile=resfile,chatter=chatter)
            matresult = matros.process()  
         
     # process the output from readout_streak ; obtain raw coordinates for each extension
     hdu = fits.open(md)
     datesobs = []
     tstart_ = []
     extname_ = []
     rawxy_ = []
     for m in range(1,len(hdu)): 
         binx = hdu[m].header['binx']
         datesobs.append(hdu[m].header['date-obs'])
         tstart_.append( hdu[m].header['tstart'] ) 
         extname_.append(hdu[m].header['extname'] )  
         rawxy_.append([1024/binx,1024/binx])

     # convert the output to an obs dict (unless already done)
     obses, details = _read_readout_streak_output(
         obses,
         inp=resfile,
         band=band,        # md.split('_')[0][-3:],
         dateobs=datesobs,
         tstart=tstart_,
         extname=extname_,
         infile=md,
         rawxy=rawxy_,
         chatter=chatter,
         )
         
     # run readout_streak_mag on each image 
     for obs in obses:
        if chatter > 0:
            sys.stderr.write( "\n============== LSS and magnitudes for : "+
            obs["infile"]+"+"+str(obs["extension"])+"\n" )

        # find source position in raw image
        rawxy = obs['rawxy']  # default center of image
        if radec != 'without':
               ext = obs['extension']
               sys.stderr.write("  skyfile = %s, ext=%s\n"%(sk,ext))
               posJ2000 = radec2pos(ra, dec, chatter)
               detx, dety = radec2det(posJ2000,sk,ext,chatter, det_as_mm=False)
               obs.update( {'det_coord':(detx,dety)})       
               rawx, rawy = from_det_to_raw(detx,dety,invert=True)
               obs.update( {'rawxy': [rawx,rawy]})
               obs.update( {'img_coord':(rawx,rawy)})
               if (rawx < 0) | (rawx > 2047) | (rawy < 0) | (rawy > 2047):
                   if chatter > 0:
                      sys.stderr.write(
                      "WARNING: Source falls outside RAW image. "+
                      "RAW position = (%10.1f,%10.1f)\n "%(rawx,rawy))
                      sys.stderr.write("Continuing with interactive pick of source\n")
                   radec = 'without'
                   # and here a call to a subroutine for processing a single image 
               else: 
                   binning = fits.getval(sk, 'binx', ext=ext)
                   rawxy = [rawx/binning,rawy/binning]       

        # do the LSS correction   
        imode = interactive | (radec == 'without')
        lss, img_coord, det_coord  = _lss_corr(obs, figno=figno, 
           interactive=imode,rawxy=rawxy,chatter=chatter) 
        if chatter > 3: 
            print (f"\nlss corr result: \nlss={lss}, img_coord={img_coord}, det_coord={det_coord}\n")   
        obs.update( {'img_coord':img_coord})  
        obs.update( {'lss':lss})     

        # corrected magnitudes  
        if (lss != 0.0) & (len(obs['streak_col_SN_CR_ERR']) > 0):
            obs.update( {'det_coord':det_coord})       
            #if lss != 1.0:  # LSS could not be determined 
            obs = _decide_column(obs,target) # update obs 
            data = _readout_streak_mag(obs, target=target,lss=lss,
               subimg_coord=img_coord,det_coord=det_coord ,chatter=chatter)
            obs.update( {'magnitudes': data})
            result.append(obs)
        else:
           if chatter > 0:
               print ("\nskipping this image\nconsider running uvotsource\n" ) 

   # mag output : cycle through filters

   if chatter > 0: print (f"\n * * * Outputting results * * * \n***************************************")
   if (do_only_band != None):
      filts = ['uvw2','uvm2','uvw1','u','b','v']
   for fi in filts:
       for obj in result:
           datobs = obj['dateobs']    # INSERTED TO PREVENT FAILED WRITE 2022-08-23
           ext = obj['extension']
           tsta = obj['tstart']
           extnam = obj['extname']
           lss = obj['lss']
           MJD = dateobs2MJD(datobs)  #
           no_streaks_found = False
           if (do_only_band != None) & (fi != do_only_band): 
                break
           #sys.stderr.write("obj band= %s searching band = %s\n"%(obj['band'],fi))
           if obj['band'] == fi:
               mag = -99.  # initialise
               err = 99.
               try:
                  x = obj['img_coord'][0]
                  matched_only = obj['magnitudes'][0] == "matched_only"
                  # matched_only means there was a single streak with 'true' 
                  if chatter > 3: print (f"selected column img_coord={x}")
                  if len(obj['streak_col_SN_CR_ERR']) < 1: # no streak found
                      mag = -1. 
                      err = -1.
                      ext = -1
                      if chatter > 0:
                         print ("check no streak case :",obj)
                  else:    
                    if chatter > 3: print (f"line 443 OBJ = {obj}\n")
                    streaks=[]
                    for s in  obj['streak_col_SN_CR_ERR']: 
                        streaks.append(s[0])
                    streaks = np.array(streaks)
                    no_streaks_found = len(streaks) == 0 # should not happen 
                    if chatter > 3: print(f"446found the following streaks:\n\t {streaks}.")
                    # first try to pick the brightest streak within 16 subpixels
                    # if that did not work, just use the closest one in distance
                    try:   # 1
                      if chatter > 3: print(f"450trying to pick the brightest streak within 16 subpixels.")
                      
                      kandi = np.abs(streaks-x) < 16  # this can match more than one
                      kanmag = []
                      kanerr = []
                      if not matched_only: # this depends on _readout_streak_mag returning a single mag or for all streaks
                         for k in obj['magnitudes']:
                            kanmag.append( k[2] )
                            kanerr.append( k[3] )
                         if chatter > 3: print(f"466 brightest search: kandi={kandi} mags={kanmag}")   
                         kanmag = np.array(kanmag)[kandi]
                         kanerr = np.array(kanerr)[kandi]
                      else:  # single mag ... 
                         objmag = obj['magnitudes'][1]   
                         kanmag=[objmag[2]]
                         kanerr=[objmag[3]]
                         mag = kanmag[0]
                         err = kanerr[0]
                      if chatter > 3: 
                          print (f"474 brightest mag and err:\n\t mag = {mag}\n\t err = {err}\n")
                      if (len(kanmag) == 0) and (not no_streaks_found): 
                          if chatter > 3: print (f"476 since no brightest, try find closest streak to {x}. ")
                          k = np.where(np.abs((streaks-x)) == np.min(np.abs(streaks - x)))
                          k = k[0][0]
                          mag = obj['magnitudes'][k][2]  # wrong magnitudes is different array
                          err = obj['magnitudes'][k][3]
                      elif no_streaks_found:  # put failed numbers in mag,err
                          mag = -99.
                          err = 99.    
                    except:  
                       # 1 -- there is not a brightest ro streak nor a single one flagged true
                       # this will run if there are multiple streaks 'true' and not brightest
                       if chatter > 3: 
                          print(f"trying to find the closest readout streak to {x}. streaks={streaks}, {type(streaks)}")
                          
                       k = np.where(np.abs((streaks-x)) == np.min(np.abs(streaks - x)))
                       if chattter > 3: 
                           print (f"streak index is k={k}\n")
                       k = k[0][0]
                       if matched_only:
                          k_mag = obj['magnitudes'][1][0]
                          if np.abs(k_mag - k) > 16: 
                              print (f"\nWARNING\nWarning: matching problem with nearest to selected streak.\n\t Check results!\n")
                              mag = obj['magnitudes'][1][2]
                              err = obj['magnitudes'][1][3]
                          if chatter > 3: print (f"mag={mag}, err={err}")
                       else:
                          mag = obj['magnitudes'][k][2]
                          err = obj['magnitudes'][k][3]
                          if chatter > 3: print (f"mag={mag}, err={err}")
                       if chatter > 3: 
                          print ("*** problem line 499\n")
                       pass
                  # overlimit = obj['streak_col_SN_CR_ERR'][k][0] when, set errors to .9999
                  if chatter > 3: print ("line 515")
                  datobs = obj['dateobs']
                  ext = obj['extension']
                  tsta = obj['tstart']
                  extnam = obj['extname']
                  MJD = dateobs2MJD(datobs)
                  lss = obj['lss']
                  if chatter > 3: print ( f"{MJD}  {fi}={mag}+/-{err} {tsta} {datobs} {extnam}+{ext}" )
                  if chatter > 0:
                     sys.stderr.write( f"{MJD}  {fi}={mag}+/-{err} {tsta} {datobs} {extnam}+{ext}" )
                     sys.stderr.write("\n")
               except:
                  if chatter > 0:
                     sys.stderr.write( "there seems to be a problem writing: %s\n"%( obj ))
                  pass
               if chatter > 0 : print (f"... outputting result for {obsid}+{ext}\n") 
               if not no_streaks_found:
                  try:  
                      magfh.write("%12.5f %7.3f %5.3f %5.3f %6s %11.1f %11s %11s %2i %7.4f\n"%
                      (MJD,mag,err,syserr,fi,tsta,datobs[0:16],obsid,ext,lss))
                  except:
                      raise RuntimeError("Failed to write result to file.\n")      
                  magff = _mag_to_fitsout(magff, obj['band'], mag, err,tsta, datobs[0:16],\
                      obsid,ext,extnam,MJD,lss,syserr,chatter=chatter)

   magfh.close()
   #try:
   f = open('readout_streak.results.py','w')
   f.write("%s"%(result))
   f.close()
   if fitsout: 
       magff.writeto(fitsout,checksum=True,overwrite=True)  
       magff.close() 
   return result

def dateobs2MJD(dateobs):
   from astropy.time import Time
   t = Time(dateobs,format='fits')
   return t.mjd

def _lss_corr(obs,interactive=False,maxcr=False,figno=None,
         rawxy=None,
         target='target',
         chatter=0):
   '''determine the LSS correction for the readout streak source 
      for one particular image/extension
   
   parameters
   ---------
   obs : structure
      readout streak data
   interactive : bool
      pick source position on detector from image
   maxcr : bool
      
   figno : int
      figure number to use in display
   rawxy : list
      the target position on the raw image
   target : string (optional)
      name of target 
   chatter : int
      verbosity
   
   '''
   import os
   import numpy as np
   from astropy.io import fits
   from pylab import figure,imshow,ginput,axvspan,\
        axhspan,plot,autumn,title,clf,legend,text

   file = obs['infile']
   ext = obs['extension']
   cols = obs['streak_col_SN_CR_ERR']
   
   kol = []
   SN = []
   countrates=[]
   circle=[np.sin(np.arange(0,2*np.pi,0.05)),np.cos(np.arange(0,2*np.pi,0.05))]
   for k in cols:
      SN.append(k[1]) # S/N
      kol.append(k[0]) # column number relative to bottom rh corner subimage
      countrates.append(k[2])
      
   SN = np.array(SN)   
   if (len(SN) == 0) | (len(kol) == 0):   
      if chatter > 0:
          sys.stderr.write( "no readout streak columns found!\n"+
           "LSS correction cannot be determined.\n") 
      return 1.0, (0,0), (1100.5,1100.5)
      
   k_sn_max = np.where(np.max(kol) == kol) # maximum s/n column
   if chatter > 1:
       sys.stderr.write( "k S/N max=%i\n"%(k_sn_max[0][0]))

   hdr=fits.getheader(file, ext=ext)
   binx = hdr['BINX']
   coord = [[-1,-1]]
   if interactive:
      im=fits.getdata(file, ext=ext)   
      mn = im.mean()
      sig = im.std()
      #   plot
      fig = figure(figno)
      fig.clf()
      ax = fig.add_subplot(111)
      ax.imshow(im,vmin=mn-0.25*sig,vmax=mn+1.8*sig,cmap=cm.inferno)
      if rawxy != None:
          rawx,rawy = rawxy
          R = hdr['windowdx']/15./hdr['binx']
          if chatter > 1:
             if rawxy == [1024/binx,1024/binx]: print (f"Using the default values for the ")
             print ("\ncenter position rawxy+window0 = ",rawx-hdr['windowx0'],rawy-hdr['windowy0'])
             print (f"rawxy = ({rawx},{rawy})")
             print (f"windowx0,--y0 = ({hdr['windowx0']},{hdr['windowy0']})")
          ax.plot(R*circle[0]+rawx-hdr['windowx0'],R*circle[1]+rawy-hdr['windowy0'],
             '-',color='cyan',alpha=0.3,lw=1.5)
      title(u"PUT CURSOR on your OBJECT",fontsize=16)
      ax.text(1.1,0.5,obs['extname']+f"+{ext}",rotation='vertical',ha='center',
         verticalalignment='center',transform=ax.transAxes,color='m')
      if not maxcr:
          count = 0
          for k in kol:
              ax.axvspan(k-6,k+6,0.01,0.99, facecolor='w',alpha=0.3-0.02*count)
              count += 1
      else:
          k = k_sn_max[0][0]    
          ax.axvspan(k-10,k+10,0.01,0.99, facecolor='w',alpha=0.2)
      happy = False
      skip = False
      count = 0
      fig.canvas.draw()
      while not happy :  
         sys.stdout.write( "put the cursor on the location of your source\n" )
         count += 1
         coord = ginput(n=1,timeout=0)
         sys.stdout.write( "selected position in image: %s\n"%(coord))
         if len(coord[0]) == 2: 
            xpick,ypick = coord[0]
            xloc = coord[0][0]*hdr['binx']+hdr['windowx0']  # this is position in the 2048,2048 pixels
            yloc = coord[0][1]*hdr['binx']+hdr['windowy0']
            if chatter > 0:
               sys.stdout.write( "window corner: (%7.1f,%7.1f)\n"%( hdr['windowx0'],hdr['windowy0'] ))
               sys.stdout.write( "     in image: (%7.1f,%7.1f)\n"%(xpick,ypick))
               print(f" total = ({hdr['windowx0']+xpick},{hdr['windowy0']+ypick})\n")
               sys.stdout.write( "on detector (full raw image)  should be : (%7.1f,%7.1f)\n"%(xloc,yloc))
            text(xpick,ypick,'x',fontsize=14,color='k',ha='center',va='center') 
            img_coord = xpick, ypick 
            ax.axhspan(ypick-6,ypick+6,0,1,facecolor='w',alpha=0.2)
            fig.canvas.draw()
            if rawxy != None:
               rawx,rawy = rawxy
               R = hdr['windowdx']/20./hdr['binx']
               ax.plot(R*circle[0]+xpick,R*circle[1]+ypick, '-',
                  color='lawngreen',alpha=0.7,lw=1,label='adopted position')
               ax.legend() 
               
               #print (f"computed position with offset: windowxy0 = ({hdr['windowx0']},{hdr['windowy0']}) \n")  
               #plot(R*circle[0]+rawx-hdr['windowx0'],R*circle[1]+rawy-hdr['windowy0'], '-',color='k',alpha=0.7,lw=1)
               #plot(rawx-hdr['windowx0'],rawy-hdr['windowy0'],'o',markersize=25,color='w',alpha=0.3)
            ans = input("happy (yes,skip,no): ")
            if len(ans) > 0:
               if ans.upper()[0] == 'Y': 
                   happy = True
               if ans.upper()[0] == 'S': 
                   #skip this image 
                   return 0.0, coord[0], (yloc+104,xloc+78) 
         else:
            sys.stdout.write( "no position found\n")     
         if count > 10:
            sys.stdout.write( "Too many tries: aborting\n") 
            happy = True
      im = ''
   else: 
      # not interactive: a position on the (binned) image has been given 
      xloc, yloc = rawxy 
      img_coord = xloc, yloc # used for locating the source on the image array
      xloc = np.int(xloc)*binx #+hdr['windowx0']
      yloc = np.int(yloc)*binx #+hdr['windowy0']  
   
   # now do the correction 
   if chatter > 0:
      sys.stderr.write(f"getting the LSS correction at location {yloc},{xloc}\n")
   lss = 1.0
   band = obs['band']
   caldb = os.getenv('CALDB')
   command = "quzcif swift uvota - "+band.upper()+" SKYFLAT "+\
         obs['dateobs'].split('T')[0]+"  "+\
         obs['dateobs'].split('T')[1]+" - > lssfile_.tmp"
   if chatter > 0:
          sys.stderr.write("shell command: "+command+"\n") 
   try:
      os.system(command)
   except:
      sys.stderr.write("readoutstreak._lsscorr.quzcif error\n...setting to 1.0") 
      return 1.0, (0,0), (1100.5,1100.5) 
         
   f = open('lssfile_.tmp')
   lssfile = f.readline()
   f.close()
   try:
      f = fits.getdata(lssfile.split()[0],ext=int(lssfile.split()[1]))
      lss = f[np.int(yloc),np.int(xloc)]
   except:
      sys.stderr.write("cannot open lss file:  "+lssfile+" at position "+str(xloc)+','+str(yloc))   
   if chatter > 1:
         sys.stderr.write(f"lss correction = {lss}  at position {yloc+104}, {xloc+78}")
         sys.stderr.write("\n")
   det_coord = yloc+104,xloc+78  
   if chatter > 3:
      print ("_lss_corr output = ",lss,img_coord,det_coord)    
   return lss, img_coord, det_coord # lss, position of source on image, detector coordinate 


def _read_readout_streak_output(obses,inp='results.txt',
     band=None,
     dateobs=None,
     tstart=None,
     infile=None,
     extname=None,
     rawxy=None,
     chatter=0):
   '''convert output from <results.txt> to list of obs dict 
   and append them to obses list
   
   lower limits are not given ?
   ''' 
   import numpy as np
   if chatter > 5: 
      print (band, dateobs, tstart, infile, extname)
      print (rawxy,inp,obses)
   
   obs = []
   bandname=np.array(['white','uvw2','uvm2','uvw1','u','b','v'])
   bandits =np.array(['uwh','uw2','um2','uw1','uuu','ubb','uvv'])
   # example output of readout_streat infile=sw00032911033uw1_rw.img > result.txt
#
# Running readout_streak V2.0 
#No output fixedfile requested
#No output maskedfile requested
#Adopting SN limit of   6.0
#
# Extension 1, exposure 9.59203, frametime 0.0110322
#Ext001  Streak at column 0992, S/N =  16.1, CR = 0.455829 +- 0.028318
#Ext001  Streak at column 1905, S/N =   9.9, CR = 0.173861 +- 0.017612
#
# Extension 2, exposure 374.047, frametime 0.003603
#Ext002  Streak at column 0163, S/N =  67.2, CR = 1.166632 +- 0.017354
#Ext002  Streak at column 0147, S/N =  18.0, CR = 0.219290 +- 0.012194
   # read the file 
   f = open(inp)
   recs = f.readlines()
   f.close()
   n_ext = 0
   n_roc = 0 # count readout columns per extension
   ext_meta=[]
   ext_data=[]
   ext = 0
   if (chatter > 4): print("\nreading results file\n")
   for r in recs:
      if (chatter > 4): print (r)
      if r[1:10] == "Extension": 
         n_ext += 1
         if n_roc == 0:
             ext_data.append( dict(ext=ext, column=-1, SN=-1,cr=-1,err=-1))         
         n_roc = 0
         ext=r.split(',')[0].split()[-1]
         ext_meta.append(dict(
           ext=ext,
           exposure=r.split(',')[1].split()[1],
           frametime=r.split(',')[2].split()[1],
           ))
      if r[0:3] == 'Ext':
         if int(r[3:6]) != int(ext):
            if chatter > 1:
               sys.stderr.write("%s/n"%( r ))
               sys.stderr.write(
                 "check extension failed : %i  not equal to %i\n"%
                 (int(r[3:6]),int(ext) ) )
         ext_data.append(dict(
            ext = ext,
            column = r.split(',')[0].split()[4],
            SN = r.split(',')[1].split()[2],
            cr = r.split(',')[2].split()[2],
            err = r.split(',')[2].split()[4],
            ))
         n_roc += 1   
   if n_roc == 0:
      # last one was empty
      ext_data.append( dict(ext=ext, column=-1, SN=-1,cr=-1,err=-1))         
   if chatter > 3: 
      print ("\n  ext_meta: ",ext_meta,"\n  ext_data : ",ext_data,"\n")         
   for m in range(n_ext): 
      streak_col_SN_CR_ERR=[]
      streak_id=[]
      goodstreak=[] 
      extension=ext_meta[m]['ext']
      for mm in range(len(ext_data)):
         try:
            if ext_data[mm]['ext'] == extension:
               if chatter > 2:
                   sys.stderr.write( "_read_readout_streak_output: %s %s %s %s \n"%
                      (m, mm, extension, ext_data[mm]))
               streak_col_SN_CR_ERR.append([
                  float(ext_data[mm]['column']),
                  float(ext_data[mm]['SN']),
                  float(ext_data[mm]['cr']),
                  float(ext_data[mm]['err']),    ])
               streak_id.append(f"{mm+1}") # target match added when made 
               goodstreak.append(False)
         except: pass      
      obs=dict(
         band=bandname[band == bandits][0],
         dateobs=dateobs[m],
         tstart=tstart[m],
         extname=extname[m],
         infile=infile,
         extension=int(extension),
         exposure =float(ext_meta[m]['exposure']),
         frametime=float(ext_meta[m]['frametime']),
         streak_col_SN_CR_ERR=streak_col_SN_CR_ERR,
         streak_id=np.array(streak_id),
         goodstreak=np.array(goodstreak),
         rawxy=rawxy[m], 
         )
      obses.append(obs)
   return obses,(n_ext,ext_meta,ext_data)

def _readout_streak_mag(obs, target='target',lss=1.0,subimg_coord=None, 
       det_coord=None, instrument='UVOT', chatter=0):
   '''
   get the input parameters 
   
   input parameters
   ================
   obs: dict
   provides the measurements 
   
   output
   ======
   output  magnitudes on Vega system
   
   '''
   import numpy as np
   import datetime
   
   if instrument == 'UVOT':
      # table 1 from Mat Page et al. UVOT parameters
      frametimes = [11.0329e-3, 5.417e-3, 3.600e-3] # available UVOT CCD frametimes
      S          = [     9049.,    4369.,    2855.] # 
      max_cr     = [      0.30,     0.62,     0.95] # maximum coi-corrected CR 
      # table 2 
      zp = { # provides list of Vega-system zeropoints for  
      # full frame, large window, small window 
      # frame times 0.011, 0.078, 0.036  *** check ft for 8x8 frame
      'v':[8.00,8.79,9.25],
      'b':[9.22,10.01,10.47],
      'u':[8.45,9.24,9.70],
      'uvw1':[7.55,8.34,8.80],
      'uvm2':[6.96,7.75,8.21],
      'uvw2':[7.49,8.28,8.74],
      'white':[10.40,11.19,11.65]
      }
      t_MCP=2.36e-4
   elif instrument == "OM":
      raise IOError("OM not yet implemented")
   else:
      raise IOError("instrument must be UVOT or OM")
      
   overlimit = False
   band = obs['band'].lower()
   systematic_err = 0.1
   if chatter > 0:
      sys.stderr.write( "\n%s Readout Streak for %s in the %s filter. "%
         (target,obs['dateobs'], band))
   
   # approximate correction for sensitivity loss (not calibrated foor readout streak)
   date=obs['dateobs']
   senscorr = sensitivity.get(band, date, timekind='UT')
   obs.update({"senscorr":senscorr})
   if len(date) < 11:
       xseconds = datetime.datetime(int(date[:4]),int(date[5:7]),
         int(date[8:10]))- datetime.datetime(2005,1,1,0,0,0)
   else:
       xseconds = datetime.datetime(int(date[:4]),int(date[5:7]),
         int(date[8:10]),int(date[11:13]), int(date[14:16]),
         int(date[17:20]),0) - datetime.datetime(2005,1,1,0,0,0)
   xyear = xseconds.days/365.26
   maxsenscorr = 4
   if senscorr < 1.0 or senscorr > maxsenscorr : 
      if chatter > 0: 
         print (f"senscorr = {senscorr} is out of range: setting to 1 percent a year\n")
      senscorr = 1.+0.01*xyear
      obs["senscorr"]= senscorr
   if chatter > 0:
      sys.stderr.write( "sensitivity correction = %7.3f (~ %7.3f)"%(senscorr,1+0.01*xyear))

   # index for the proper frametime - search within 0.015
   try:
      k = np.where(abs(1-np.array(frametimes)/obs['frametime']) < 1.5e-2)[0][0]
   except:
      raise RuntimeError("Frame time is not in our list or differs by more than 1.5 percent")
   
   # if not target in 'streak_id' field, set first
   xx = obs['goodstreak'] #!= "" 
   if chatter > 3:
      print ("\nobs[...] ", obs)
      print ("target",target)
   if xx.sum() == 1:
      streak = obs['streak_col_SN_CR_ERR'][ np.where(np.array(obs['goodstreak']))[0][0] ]
      if chatter > 3:
         print ("in sub _readout_streak_mag: streak",streak)
      column,SN,rate,err = streak
      #err += systematic_err
      # note : only a return for the matched column 
      if chatter > 0:
         sys.stderr.write(f"The column matched is at {column}\n")
      return ['matched_only',_readout_streak_mag_sub(k,S,rate,t_MCP,err,obs,xyear,
              lss,zp,band,max_cr,overlimit)]
   else:
      # here no columnn chosen - results for all columns 
      result = []
      for streak in obs['streak_col_SN_CR_ERR']:
         column,SN,rate,err = streak 
         #err += systematic_err
         if chatter > 0:
             sys.stderr.write( "column,S/N,rate,err= "%(streak))
             if (subimg_coord != None):
                sys.stderr.write( "distance column to target: "%
                   ( column-subimg_coord[0] ))
 
         result.append(
           _readout_streak_mag_sub(k,S,rate,t_MCP,err,obs,xyear,lss,zp,
           band,max_cr,overlimit,chatter=chatter)
           )
      return result          

def _readout_streak_mag_sub(k,S,rate,t_MCP,err,obs,xyear,lss,zp,band,max_cr,
        overlimit,chatter=0): 
   import numpy as np
   senscorr = obs['senscorr']
   # correcting for the MCP recharge time - assume done
   if (1.0-(S[k]*rate*t_MCP)) <= 0. :
       overlimit=True
       r_coi = np.NaN
       r_coi_u = np.NaN
       r_coi_d = np.NaN
   else:    
       r_coi= -(np.log(1.0-(S[k]*rate*t_MCP)))/(S[k] * t_MCP)
       r_coi_u = -(np.log(1.0-(S[k]*(rate-err)*t_MCP)))/(S[k] * t_MCP)
       r_coi_d = -(np.log(1.0-(S[k]*(rate+err)*t_MCP)))/(S[k] * t_MCP)
       if chatter > 0:
          sys.stderr.write( "observed CR =%10.5f,  MCP-loss corrected CR =%10.5f"%(rate,r_coi) )
   # now correct for LSS
   
   r_coi = r_coi * senscorr/lss
   err = err * senscorr/lss
   r_coi_u = r_coi_u * senscorr/lss
   r_coi_d = r_coi_d * senscorr/lss
   
   if r_coi > max_cr[k]:  overlimit=True
   mag = zp[band][k] - 2.5*np.log10(r_coi)
   mag_u = zp[band][k] - 2.5*np.log10(r_coi_u)
   mag_d = zp[band][k] - 2.5*np.log10(r_coi_d)
   if overlimit: 
      if chatter > 0:
         sys.stderr.write( "WARNING: count rate is over the recommended limit !\n")
         sys.stderr.write( "%s magnitude <%7.3f (+%7.3f -%7.3f)\n"%(band,mag,mag_u-mag,mag-mag_d) )
      return overlimit,band,mag,-mag_u+mag,-mag+mag_d
   else:   
      sys.stderr.write( "%s magnitude = %7.3f +%7.3f -%7.3f\n"%(band,mag,mag_u-mag,mag-mag_d) )
   return overlimit,band,mag,mag_u-mag,mag-mag_d

def read_the_old_readout_streak_table(infile,comment='#',chatter=0): #obsolete
    """read the photometry table output from readout_streak into a structure"""
    # if exposure is added as a field, that needs to be an option
    f = open(infile)
    
    # just do a plain extraction for each filter - nothing fancy
    rec= 'notnull'
    w2data = []
    m2data = []
    w1data = []
    uudata = []
    bbdata = []
    vvdata = []
    while rec != "":
       rec=f.readline()[:-1]  # remove the newline character
       if rec == "" : continue
       elif rec[0] == '#': continue
       else: 
          this = rec.split()
          if len(this) != 15:
              print ("expected 15 items here, but this is what was found: ", this )
          else:
              if this[0] != '-1': # uvw2
                 obsid, ext = this[14].split('+')
                 w2data.append({"mag":float(this[0]),"err":float(this[6]),"tstart":np.float64(this[12]),
                     "date":this[13],"obsid":obsid,"ext":int(ext)})    
              if this[1] != '-1': # uvm2
                 obsid, ext = this[14].split('+')
                 m2data.append({"mag":float(this[1]),"err":float(this[7]),"tstart":np.float64(this[12]),
                     "date":this[13],"obsid":obsid,"ext":int(ext)})    
              if this[2] != '-1': # uvw1
                 obsid, ext = this[14].split('+')
                 w1data.append({"mag":float(this[2]),"err":float(this[8]),"tstart":np.float64(this[12]),
                     "date":this[13],"obsid":obsid,"ext":int(ext)})    
              if this[3] != '-1': # u
                 obsid, ext = this[14].split('+')
                 uudata.append({"mag":float(this[3]),"err":float(this[9]),"tstart":np.float64(this[12]),
                     "date":this[13],"obsid":obsid,"ext":int(ext)})    
              if this[4] != '-1': # b
                 obsid, ext = this[14].split('+')
                 bbdata.append({"mag":float(this[4]),"err":float(this[10]),"tstart":np.float64(this[12]),
                     "date":this[13],"obsid":obsid,"ext":int(ext)})                     
              if this[5] != '-1': # v
                 obsid, ext = this[14].split('+')
                 vvdata.append({"mag":float(this[5]),"err":float(this[11]),"tstart":np.float64(this[12]),
                     "date":this[13],"obsid":obsid,"ext":int(ext)})
    f.close()
    if chatter > 0:
       print ("read in %i w2 values"%(len(w2data)) )
       print ("read in %i m2 values"%(len(m2data)) )
       print ("read in %i w1 values"%(len(w1data)) )
       print ("read in %i u  values"%(len(uudata)) )
       print ("read in %i b  values"%(len(bbdata)) )
       print ("read in %i v  values"%(len(vvdata)) )
    if len(w2data) > 1: 
        w2= sorted(w2data,key=lambda item : item['tstart'])     
    else:
        w2 = None             
    if len(m2data) > 1: 
        m2 = sorted(m2data,key=lambda item : item['tstart'])     
    else:
        m2 = None             
    if len(w1data) > 1: 
        w1 = sorted(w1data,key=lambda item : item['tstart'])     
    else:
        w1 = None             
    if len(uudata) > 1: 
        uu = sorted(uudata,key=lambda item : item['tstart'])     
    else:
        uu = None             
    if len(bbdata) > 1: 
        bb = sorted(bbdata,key=lambda item : item['tstart'])     
    else:
        bb = None             
    if len(vvdata) > 1: 
        vv = sorted(vvdata,key=lambda item : item['tstart'])     
    else:
        vv = None                     
    return {"w2":w2,"m2":m2,"w1":w1,"u":uu,"b":bb,"v":vv}

def _init_fitsoutput(file,nrow=60,chatter=0):
   p = fits.PrimaryHDU()
   hdr = fits.Header()
   hdr['EXTNAME'] = ("MAG_READOUTSTREAK","extension contains readout streak magnitudes")
   hdr['REFERNC'] = ("2013MNRAS.436.1684P","Page M.J. et al. 2014, use & calibration of...")
   hdr['AUTHOR'] = ("N.P.M.KUIN","MSSL-UCL, source at http://github.com/PaulKuin")
   hdr['TELESCOP']=('Swift','Telescope (mission) name')
   hdr['INSTRUME']=('UVOTA','Instrument name')
   hdr['ORIGIN']=('UCL/MSSL','source of FITS file')
   hdr['CREATOR']=('readout_streak.py','uvotpy python library')
   hdr['COLSUSED'] = (0,"number of columns with valid data")
   hdr['COMMENT'] = ("negative mag and error indicates no readout streak found")
   col = [
   fits.Column(name="MJDstart", format= "D", unit="d" ,disp="F14.5", ascii=False, array=np.ones(nrow,dtype=np.double)),
   fits.Column(name="filter", format= "A5", disp="A5", ascii=False, array=np.empty(nrow,dtype='S5')),
   fits.Column(name="mag", format= "E", unit="mag" ,disp="F7.3", ascii=False, array=np.zeros(nrow,dtype=np.float)),
   fits.Column(name="mag_err", format= "E", unit="mag" ,disp="F7.3", ascii=False, array=np.zeros(nrow,dtype=np.float)),
   fits.Column(name="sys_err", format= "E", unit="mag" ,disp="F7.3", ascii=False, array=np.ones(nrow,dtype=np.float)),
   fits.Column(name="tstart", format= "D", unit="s" ,null="",disp="F15.3", ascii=False, array=np.ones(nrow,dtype=np.double)),
   fits.Column(name="date-obs", format= "A30", unit="UT" ,ascii=False, array=np.empty(nrow,dtype='S30')),
   fits.Column(name="obsid", format= "A11", ascii=False, array=np.empty(nrow,dtype='S11')),
   fits.Column(name="ext", format= "I",disp="I3", ascii=False, array=np.zeros(nrow,dtype=int)),
   fits.Column(name="extname", format= "A30",  ascii=False, array=np.empty(nrow,dtype='S30')),
   fits.Column(name="maglim", format= "E",unit='mag',ascii=False,array=np.zeros(nrow,dtype=np.float)),
   fits.Column(name="lss", format= "E", disp="F5.3", ascii=False,array=np.ones(nrow,dtype=np.float)),
   #fits.Column(name="", format= "", unit="" ,null="",disp="", ascii=False)
   ]
   b = fits.BinTableHDU.from_columns(columns=col,header=hdr,)
   fh = fits.HDUList(hdus=[p,b],)
   fh.writeto(file)
   fh.close()
   fh = fits.open(file, mode="update")
   if chatter > 0:
      fh.info()
   return fh

def fitsBinTable_add_nrows(BinTableHDU,nrows=60):
    "return the BinTableHDU with extra row"
    hdr = BinTableHDU.header
    tab = BinTableHDU.data
    cols = tab.columns
    ncol = hdr['tfields']
    nrec = hdr['naxis2']
    newdata = []
    for i in range(ncol):
        data = np.empty(nrec+nrows,dtype=cols.dtype[i])
        data[:nrec] = tab[tab.names[i]]
        c = cols.columns[i]
        c.array = data
        newdata.append(c)
    return fits.BintableHDU.from_columns(columns=newcata,header=hdr)    
  
def _mag_to_fitsout(magff,band,mag,err,tstart,dateobs,obsid,ext,extname,MJD,lss,syserr,chatter=0):
    # 
    if not (magff.fileinfo(1)['filemode'] == 'update'):
       raise CodeError("the file should be opened with mode update")
    if magff[1].header['COLSUSED'] == magff[1].header['naxis2']: 
        fitsBinTable_add_nrows(magff[1])   
    # fill a row of data, update COLSUSED record +1
    t = magff[1].data
    n = magff[1].header['COLSUSED']
    if chatter > 3:
       print (f"inputs _mag_to_fitsout : {band} {mag} {err} \n "+
       f"tstart {tstart} {dateobs} \n {obsid} {ext} {extname} MJD {MJD} {lss} \n")
       print (f"colused parameter = {n}\n")
    t['tstart'][n] = tstart
    t['MJDstart'][n] = MJD
    t['date-obs'][n] = dateobs
    t['obsid'][n] = obsid
    t['ext'][n] = ext
    t['extname'][n] = extname
    t['filter'] = band
    t['lss'] = lss
    t['sys_err'] = syserr
    if err > 0 :
       # no a limit
       t["mag"] = mag
       t["mag_err"] = err
       t['maglim'] = 0
    else:
       # limit   
       t["mag"] = mag
       t["mag_err"] = err
       t['maglim'] = 0
    magff[1].data = t
    magff[1].header['COLUSED'] = n+1
    if chatter > 3:
       print (f"<colused> parameter set to {n+1}\n")
    magff.flush()
    return magff    


def _decide_column(obs,target='target'): # decide on correct column
   # update the obj struct with target 
   xxid = obs['streak_id']
   xxgs = obs['goodstreak']
   try:
      x = obs['img_coord'][0]
      streaks=[]
      for s in obs['streak_col_SN_CR_ERR']: 
         streaks.append(s[0])  # column positions 
         # first try to pick the brightest streak within 16 subpixels
         # if that did not work, just use the closest one in distance
      try:  
            kandi = np.abs(streaks-x) < 16  # candidate streaks within distance 16
            kanmag = []
            kanerr = []
            for k in obj['magnitudes']:
               kanmag.append( k[2] )
               kanerr.append( k[3] )
            kanmag = np.array(kanmag)[kandi]
            kanerr = np.array(kanerr)[kandi]
            if len(kanmag) == 0: 
                # no match in 16 pix, find nearest
                k = np.where(np.abs((streaks-x)) == np.min(np.abs(streaks - x)))
                kk = k[0][0]
            else:
                kk = (kanmag == np.min(kanmag))  # select brightest
      except:  
         # if failed, select on distance 
             k = np.where(np.abs((streaks-x)) == np.min(np.abs(streaks - x)))
             kk = k[0][0]
             pass
   except: 
      raise RuntimeError("problem with finding the matching column")          
   xxid[kk] = target
   obs['streak_id'] = xxid
   xxgs[kk] = True
   obs['goodstreak'][kk] = True
   return obs

   
def fix_hwwindow_header(file, fiximage_extension=True, chatter=0):  
    """
    Fix the WINDOW?? parameters in the raw image file header
    and crop the image for hardware modes with fast frame times.
    
    parameters
    ----------
    file : path 
       to raw image file
    chatter : int
       verbosity
       
    Notes
    -----    
    some RAW image headers that were taken in a hardware mode (event) 
    have the parameters WINDOWX0,WINDOWY0, WINDOWDX, WINDOWDY set
    to 0,0,2048,2048, which is the event window size in the 
    house keeping files. However the actual data is in only a small
    part of the window, which is decided on-board depending on the 
    actual pointing. Further processing, for example for the 
    read-out streak needs just the good data. This program actually
    limits the range slightly to avoid empty columns at the edge. 
    
    Other image have the exposed part of the image not in the expected position. 
    
    Although the planned positions of windows in hardware mode are 
    positioned in the center of the detector, in practice some exposures 
    are started before the slew ends and the exposed image can appear 
    shifted by a considerable amount. 
    
    2020-07-25 Modifying approach to reduce complete reliance on HK files 
    2020-07-26 The WINDOWX0 and WINDOWY0 are not consistent across all type 
    of files, being transposing X and Y in the short timeframe images where 
    the data is embedded in a 2048x2048 data window (outside=0).  Changing 
    approach to always detect where the window is located. 
    """
    # The window location is encoded in the hk/sw00*uct.hk file. 
    # The parameters: DW_X0, DW_Y0, DW_XSIZ,DW_YSIZ are in 2 physical pixel units
    # Therefore the image coordinates are 
    # DW_X0*16:(DW_X0+DW_XSIZ)*16 ,  DW_Y0*16,(DW_Y0+DW_YSIZ)*16 
    # though the first left columns are looking blank. 
    # Is only needed if the the window is not properly put in the header as
    #can be seen when NAXIS1,NAXIS2=2048,2048 in the raw image header, while 
    # the frame time is faster than the default.
    #
    # a better fit excluding most of the unreliable edge regions seems to be:
    #In [714]: xx0 = x0*16-1+8
    #In [715]: xx1 = xx0+16*dx-8
    #In [716]: yy0 = y0*16-1+8
    #In [717]: yy1 = yy0+dx*16-8
    status = 0 
    if chatter > 0: 
        sys.stderr.write( "\nfix_hwwindow_header: examining "+file+"\n" )
    hdu = fits.open(file,'update')
    n_ext= len(hdu)
    obsid = hdu[1].header['obs_id']
    rootdir = file.rsplit('/',1)
    if len(rootdir) == 1:
       rootdir = './'
    elif len(rootdir) == 2:
       rootdir = rootdir[0]+'/'
    else: raise RuntimeError("this should not happen\n")      

    #if chatter > 2:
    #   sys.stderr.write("converting image size - checking presence HK file.\n")
    #if os.access(rootdir+'../hk/sw'+obsid+'uct.hk.gz', os.F_OK): 
    #   os.system('gunzip -f'+'../hk/sw'+obsid+'uct.hk.gz')
    #if os.access(rootdir+'../hk/sw'+obsid+'uct.hk', os.F_OK):
    #   hk = fits.open(rootdir+'../hk/sw'+obsid+'uct.hk')
    #else:
    #    status = 1
    #    sys.stderr.write("ERROR: houskeeping file %s not found\n")

    for k in range(1,n_ext):
        if chatter > 0:
           sys.stderr.write( "examining HDU number : %i\n"%(k))   
        ft = hdu[k].header['framtime'] 
        ax1 = hdu[k].header['naxis1']
        binx = hdu[k].header['binx']
        tstart=hdu[k].header['tstart']
        expid = hdu[k].header['expid']
        windowx0 = hdu[k].header['WINDOWX0']
        windowy0 = hdu[k].header['WINDOWY0']
        windowdx = hdu[k].header['WINDOWDX']
        windowdy = hdu[k].header['WINDOWDY']
        crval1 = hdu[k].header['CRVAL1']
        crval2 = hdu[k].header['CRVAL2']
        band = hdu[k].header['FILTER']
        # determine part of image with data > 0. img[y0img:y1img,x0img:x1img]
        x0img,x1img,y0img,y1img = _scan_image(hdu[k].data)
        if chatter > 3:  # report 
            sys.stderr.write(f"frametime = {ft}\n")
            sys.stderr.write(f"before change HDR[window**] x0={windowx0},y0={windowy0},"+
            f" dx={windowdx}, dy={windowdy}\n")
            sys.stderr.write("... scanning position of image\n")
            sys.stderr.write(f"scanned: x={x0img}--{x1img}, y={y0img}--{y1img}\n")
            sys.stderr.write( "TSTART = %f; exposure=%s\n"%(tstart,expid))
        
        if (windowdx*binx < 2048) & (not fiximage_extension):
              # we'll not fix this extension header 
              if chatter > 3:
                 sys.stderr.write("...not fixing header of extension %i\n"%(k))
              #break -- this cause further extensions to be skipped

        else:  # compute expected window size: 
           # for the hardware mode with smaller frame time, the window 
           # size gets smaller in proportion; use 0.1 tolerance
           expected = int(8*256/binx*(ft/0.0110329) + 0.1)
           if (np.abs(expected - ax1) > 5) & (ax1 == 2048/binx) : 
              # the expected image size is smaller than the image axis1,axis2 indicate
              # --> make a sub image for the extension
              #  information on the window position is in the housekeeping files
              """
             sys.stderr.write("converting image size.\n")
             xx = np.array(expid - hk[1].data['expid'],dtype=int)
             nnn= np.where( np.abs (xx) == np.min(np.abs(xx)))[0][0]
             DW_X0 = hk[1].data['DW_X0'][nnn]
             DW_Y0 = hk[1].data['DW_Y0'][nnn]
             DW_XSIZ = hk[1].data['dw_xsiz'][nnn]
             DW_YSIZ = hk[1].data['dw_ysiz'][nnn]
             x0=DW_X0*16+7
             x1=x0+DW_XSIZ*16-8
             y0=DW_Y0*16+7
             y1=y0+DW_YSIZ*16-8 
              """
              x0 = x0img
              y0 = y0img
              x1 = x1img
              y1 = y1img
              DW_X0=x0img/16
              DW_Y0=y0img/16
              DW_XSIZ = np.int((x1-x0)/16)
              DW_YSIZ = np.int((y1-y0)/16)
              #  now we need to compare the values found from scan_img with those from the header
              #
              if chatter > 0:
                 sys.stderr.write( "cropping image and updating image header window size"+
                  "    to x: %i:%i  y: %i:%i\n"%(y0,y1,x0,x1))
              hdu[k].data = hdu[k].data[y0:y1,x0:x1]
              hdu[k].header['DW_X0'] = DW_X0
              hdu[k].header['DW_Y0'] = DW_Y0
              hdu[k].header['DW_XSIZ'] = DW_XSIZ
              hdu[k].header['DW_YSIZ'] = DW_YSIZ
              hdu[k].header['WINDOWX0'] = x0
              hdu[k].header['WINDOWY0'] = y0
              hdu[k].header['WINDOWDX'] = x1-x0
              hdu[k].header['WINDOWDY'] = y1-y0
              hdu[k].header['CRVAL1'] = x0
              hdu[k].header['CRVAL2'] = y0
           elif (ax1 == 2048/binx):
              if chatter > 0:
                 sys.stderr.write("HDU[%i] the expected image size is %i\n"%(k, expected)) 
                 sys.stderr.write("while  naxis1 = %i - Problem?? binx=%i, frame time=%f\n"%
               (ax1,binx,ft))   
               
        #if (crval1 == 0) & (crval2 == 0) & (windowx0 !=0) & (windowy0 != 0): # &(band=='UVM2')  
        #   # this is for 3.6ms windows with wrong keywords in x,y (found for UVM2)
        #   hdu[k].header['CRVAL1'] = windowy0
        #   hdu[k].header['CRVAL2'] = windowx0         
        #   hdu[k].header['WINDOWX0'] = windowy0
        #   hdu[k].header['WINDOWY0'] = windowx0
           
   # hk.close()    
    hdu.writeto(file+".new",output_verify='fix',overwrite=True)
    hdu.close()
     
def _scan_image(img):
    a = img.shape
    x0,x1,y0,y1=0,a[0],0,a[1]
    x = np.where(img.sum(1) != 0)
    x0 = np.min(x)
    x1 = np.max(x)
    y = np.where(img.sum(0) != 0)
    y0 = np.min(y)
    y1 = np.max(y)
    return y0,y1,x0,x1
 
def read_a_maghist_file(infile,chatter=0):
    """read the fits output from running uvotsource or uvotmaghist into a structure """
    f = fits.open(infile)
    band = f[1].data['filter']
    tstart = np.array(f[1].data['tstart'],dtype=np.float64)
    tstop = f[1].data['tstop']
    mag = f[1].data['mag']
    err = f[1].data['mag_err']
    extname = f[1].data['extname']
    n = len(band)
    f.close()
    w2data = []
    m2data = []
    w1data = []
    uudata = []
    bbdata = []
    vvdata = []
    whdata = []    
    for i in range(n):
        date = uvotmisc.swtime2JD(tstart[i])[3][:16]
        if band[i].upper() == 'UVW2':
              w2data.append({"mag":mag[i],"err":err[i],"tstart":tstart[i],
                     "date":date,"tstop":tstop[i],"extname":extname[i]})    
        elif band[i].upper() == 'UVM2':
              m2data.append({"mag":mag[i],"err":err[i],"tstart":tstart[i],
                     "date":date,"tstop":tstop[i],"extname":extname[i]})    
        elif band[i].upper() == 'UVW1':
              w1data.append({"mag":mag[i],"err":err[i],"tstart":tstart[i],
                     "date":date,"tstop":tstop[i],"extname":extname[i]})    
        elif band[i].upper() == 'U':
              uudata.append({"mag":mag[i],"err":err[i],"tstart":tstart[i],
                     "date":date,"tstop":tstop[i],"extname":extname[i]})    
        elif band[i].upper() == 'B':
              bbdata.append({"mag":mag[i],"err":err[i],"tstart":tstart[i],
                     "date":date,"tstop":tstop[i],"extname":extname[i]})    
        elif band[i].upper() == 'V':
              vvdata.append({"mag":mag[i],"err":err[i],"tstart":tstart[i],
                     "date":date,"tstop":tstop[i],"extname":extname[i]})    
        elif band[i].upper() == 'WHITE':
              whdata.append({"mag":mag[i],"err":err[i],"tstart":tstart[i],
                     "date":date,"tstop":tstop[i],"extname":extname[i]})    
    if len(w2data) > 1: 
        w2= sorted(w2data,key=lambda item : item['tstart'])     
    else:
        w2 = None             
    if len(m2data) > 1: 
        m2 = sorted(m2data,key=lambda item : item['tstart'])     
    else:
        m2 = None             
    if len(w1data) > 1: 
        w1 = sorted(w1data,key=lambda item : item['tstart'])     
    else:
        w1 = None             
    if len(uudata) > 1: 
        uu = sorted(uudata,key=lambda item : item['tstart'])     
    else:
        uu = None             
    if len(bbdata) > 1: 
        bb = sorted(bbdata,key=lambda item : item['tstart'])     
    else:
        bb = None             
    if len(vvdata) > 1: 
        vv = sorted(vvdata,key=lambda item : item['tstart'])     
    else:
        vv = None                     
    if len(whdata) > 1: 
        wh = sorted(vvdata,key=lambda item : item['tstart'])     
    else:
        wh = None                     
    return {"w2":w2,"m2":m2,"w1":w1,"u":uu,"b":bb,"v":vv,'wh':wh}
     
def merge_data(from_readout1,from_readout2=None,from_maghist=None,
       timezero=None,maxerror=0.9,chatter=0):
    """ merge the photometry as prepared by calling 
         read_the_readout_streak_table 
         and
         read_a_maghist_file
         
     parameters
     ----------
     from_readout1: list
        data structure of photometry 
     from_readout2: list (optional)
        data structure of photometry 
     from_maghist: list
        data structure of photometry             
     timezero: float
        time origin in swift time
        default is to convert to MJD

     returns
     -------
     list of photometry 
     
     caveat: only uv filters implemented now
        
    """
    w2 = None
    m2 = None
    w1 = None
    w2_ = []
    m2_ = []
    w1_ = []
    if timezero == None:
        timezero = np.float64(-51910.0)*86400.
    if from_readout2 != None: 
         if from_readout1["w2"] != None:
             xx = from_readout1["w2"]
             for x in xx: 
                 if float(x['err']) < maxerror: w2_.append(x)            
         if (chatter > 0) & (w2_ != None): print ("%i w2 values read in"%(len(w2_)))
         if from_readout2["w2"] != None:
             for x in from_readout2["w2"]:
                 if float(x['err']) < maxerror: w2_.append(x)
         if (chatter > 0) & (w2_ != None): print ("%i w2 values read in"%(len(w2_)))
                 
         if from_readout1["m2"] != None:
             xx = from_readout1["m2"]
             for x in xx: 
                 if float(x['err']) < maxerror: m2_.append(x)            
         if (chatter > 0) & (m2_ != None): print ("%i m2 values read in"%(len(m2_)))
         if from_readout2["m2"] != None:
             for x in from_readout2["m2"]:
                 if float(x['err']) < maxerror: m2_.append(x)
         if (chatter > 0) & (m2_ != None): print ("%i m2 values read in"%(len(m2_)))
          
         if from_readout1["w1"] != None:
             xx = from_readout1["w1"]
             for x in xx: 
                 if float(x['err']) < maxerror: w1_.append(x)            
         if (chatter > 0) & (w1_ != None): print ("%i w1 values read in"%(len(w1_)))
         if from_readout2["w1"] != None:
             for x in from_readout2["w1"]:
                 if float(x['err']) < maxerror: w1_.append(x)
         if (chatter > 0) & (w1_ != None): print ("%i w1 values read in"%(len(w1_)))
         if type(w1_) == list: w1 = sorted(w1_, key=lambda x: x['tstart'])      
          
    if from_maghist != None:
         if from_maghist["w2"] != None:
             if w2_ == None: w2_ = []
             for x in from_maghist["w2"]:
                 if float(x['err']) < maxerror: w2_.append(x)
             if (chatter > 0) & (w2_ != None): print ("%i w2 values read in"%(len(w2_))) 
         if from_maghist["m2"] != None:
             if m2_ == None: m2_ = []
             for x in from_maghist["m2"]:
                 if float(x['err']) < maxerror: m2_.append(x)
             if (chatter > 0) & (m2_ != None): print ("%i m2 values read in"%(len(m2_)))
         if from_maghist["w1"] != None:  
             if w1_ == None: w1_ = []
             for x in from_maghist["w1"]:
                 if float(x['err']) < maxerror: w1_.append(x)
             if (chatter > 0) & (w1_ != None): print ("%i w1 values read in"%(len(w1_)))
    # sort data and add day                  
    if w2_ == []:
        w2 = None
    else:
        w2 = sorted(w2_, key=lambda x: x['tstart'])
        for x in w2:
             day = (x['tstart']-timezero)/86400.0
             x.update({'day':day})      
    if m2_ == []:
        m2 = None
    else:
        m2 = sorted(m2_, key=lambda x: x['tstart'])
        for x in m2:
             day = (x['tstart']-timezero)/86400.0
             x.update({'day':day})      
    if w1_ == []:
        w1 = None
    else:    
        w1 = sorted(w1_, key=lambda x: x['tstart'])
        for x in w1:
             day = (x['tstart']-timezero)/86400.0
             x.update({'day':day})                   
    return {"w2":w2,"m2":m2,"w1":w1}                 
       
def write_qdp_all(from_merge,outfile="photom.qdp",bin=True,timedel=None,mode='log',chatter=0):
    """given the output from merge_data() write a QDP file 
     
       parameters
       ----------
       from_merge: structure
           output from merge_date()
       outfile: path
           qdp output file name
       bin: bool
           binning 
       timedel: float
           timestep for binning (when mode:log then in log(time))
       mode: ['log','linear']
           bin up logarithmic in time or 
           linear in time
  
       results
       -------
       two qdp files are made, one with the full data, one with the binned data
       existing files are always overwritten!
    """
    if timedel == None:
       if mode == 'linear': timedel = 1.0
       if mode == 'log': timedel = 0.003
    #lists of all data
    w2 = from_merge["w2"]
    m2 = from_merge['m2']
    w1 = from_merge['w1']
    #open qdp file
    f = open(outfile.split('.')[0]+"_full_"+mode+"."+outfile.split('.')[1],'w')
    if bin: fb = open(outfile.split('.')[0]+"_bin_"+mode+"."+outfile.split('.')[1],'w')
    f.write("Skip Single\nRead Serr 1 2\nlog x on\ntime off\nlabel f\nlabel X Time (days)\n")
    f.write("label Y mag\nmarker 0 on\nLwidth 4\nCSize 1.37\n")
    if bin: fb.write("Skip Single\nRead Serr 1 2\nlog x on\ntime off\nlabel f\nlabel X Time (days)\n")
    if bin: fb.write("label Y mag\nmarker 0 on\nLwidth 4\nCSize 1.37\n")
    if w2 != None:
       f.write("!FILTER UVW2\n")
       if bin: 
          fb.write("!FILTER UVW2\n")
          t = []
          d = []
          m = []
          e = []
       for x in w2:
           if "tstop" in x:
               duration = x["tstop"]-x["tstart"]
           else: duration = 750.0   
           duration /= 86400. 
           f.write("%f  %f  %f   %f \n"%(x['day'],duration,x['mag'],x['err']) )
           if bin: 
               t.append(x['day'])
               d.append(duration)
               m.append(x['mag'])
               e.append(x['err'])
       if bin:
           out=_binit(t,d,m,e,timedel,mode=mode) 
           for x in out:
               fb.write("%f  %f  %f   %f \n"%(x['t'],x['d'],x['m'],x['e'] )  )
       f.write("NO NO NO NO \n")           
       if bin: fb.write("NO NO NO NO \n")          
    if m2 != None:
       f.write("!FILTER UVM2\n")
       if bin: 
          fb.write("!FILTER UVM2\n")
          t = []
          d = []
          m = []
          e = []
       for x in m2:
           if "tstop" in x:
               duration = x["tstop"]-x["tstart"]
           else: duration = 750.0    
           duration /= 86400. 
           f.write("%f  %f  %f   %f \n"%(x['day'],duration,x['mag'],x['err']) )
           if bin: 
               t.append(x['day'])
               d.append(duration)
               m.append(x['mag'])
               e.append(x['err'])
       if bin:
           out=_binit(t,d,m,e,timedel,mode=mode) 
           for x in out:
               fb.write("%f  %f  %f   %f \n"%(x['t'],x['d'],x['m'],x['e'] )  )
       f.write("NO NO NO NO \n")           
       if bin: fb.write("NO NO NO NO \n")          
    if w1 != None:
       f.write("!FILTER UVW1\n")
       if bin: 
          fb.write("!FILTER UVM2\n")
          t = []
          d = []
          m = []
          e = []
       for x in w1:
           if "tstop" in x:
               duration = x["tstop"]-x["tstart"]
           else: duration = 750.0    
           duration /= 86400. 
           f.write("%f  %f  %f   %f \n"%(x['day'],duration,x['mag'],x['err']) )
           if bin: 
               t.append(x['day'])
               d.append(duration)
               m.append(x['mag'])
               e.append(x['err'])
       if bin:
           out=_binit(t,d,m,e,timedel,mode=mode) 
           for x in out:
               fb.write("%f  %f  %f   %f \n"%(x['t'],x['d'],x['m'],x['e'] )  )
       f.write("NO NO NO NO \nr y 20 7\n")         
       if bin: fb.write("NO NO NO NO \n")          
    f.close()       
    fb.close()      

def _binit(time,dur,mag,err,timedel,mode='log',chatter=0):
    """bin the data 
       mode : ['linear','log']
          way the time bins are measured
    """
    if mode == 'log':
       ltime = np.log10(time)
    else: ltime = time   
    out = []
    lt0 = ltime[0]
    n = 0
    stime = 0.
    sdur = []
    wmag = 0.
    sweights = 0.
    for lt,t,d,m,e in zip(ltime,time,dur,mag,err):
        if chatter > 0:
            print (lt,t,d,m,e  )      
        if np.abs(lt-lt0) < timedel:
            if chatter > 0:
                print ('adding')
            stime += t
            sdur.append(t-d)
            sdur.append(t+d)
            sweights += 1./e**2    # by variance  
            wmag  += m/e**2
            n += 1  
        else:
            if chatter > 0:
                print ('writing')
            sdur = np.array(sdur)
            if chatter > 0:
                print ("sdur:",sdur)
            dd = sdur.max()-sdur.min()
            if dd < d: dd = d
            mm = wmag/sweights
            ee = np.sqrt(1./(sweights/n))
            out.append({'t':stime/n,'d':dd,'m':mm,'e':ee})
            n = 1
            stime = t
            lt0 = lt
            sdur = [t-d]
            wmag = m/e**2
            sweights = 1./e**2
    if chatter > 0:
        print ('wrapping up'  )  
    sdur = np.array(sdur)
    dd = sdur.max()-sdur.min()
    mm = wmag/sweights
    ee = np.sqrt(1./(sweights/n))
    ee
    if out[-1]['t'] < stime/n:
        out.append({'t':stime/n,'d':dd,'m':mm,'e':ee})
    return out    
####################### end readout streak subs

if __name__ == '__main__':
   #in case of called from the OS

   if status == 0:
      usage = "usage: %prog [options] -d obsid "

      epilog = '''
      get the readout streak photometry of the files of an obsid
      requires the raw and sky files, uncompressed
               ''' 
      parser = optparse.OptionParser(usage=usage,epilog=epilog)
      parser.disable_interspersed_args()
      
      # main options

      parser.add_option("", "--obsid", dest = "obsid", action="store_false",
                  help = "Swift OBSID of observation as a string, i.e. '00034380004'",
                  default="obsid",)
      parser.add_option("-t", "--target", dest = "target", action="store_true",
                  help = "name of target",
                  default="target",)
      parser.add_option("-s", "--radec", dest = "radec", action="store_true",
                  help = "the sky position as list [ra,dec] (J2000) positions in deg [default]",
                  default = "without")
      parser.add_option("-i", "--interactive", dest = "interactive", action="store_true",
                  help = "interactive? True/False",default=False)
      parser.add_option("-o", "--outfile", dest = "outfile", action="store_true",
                  help = "output file name",default="streak_photometry")
      parser.add_option("-d", "--datadir", dest = "datadir",
                  help = "path to directory with obsid data files",
                  default = '.')
      parser.add_option("-z", "--timezero", dest = "timezero",
                  help = "swift time reference (s since 2005-01-01T00:00:00)",
                  default = None)
      parser.add_option("-b", "--do_only_band", dest = "do_only_band",
                  help = "only process given uvot filter",
                  default = None)
      parser.add_option("-c", "--chatter", dest = "chatter",
                  help = "verbosity [default: %default]",
                  default = 0)
                  
   (options, args) = parser.parse_args()
   
   chatter = options.chatter
   if options.chatter > 0: 
       sys.stderr.write( "options: %s\n"%( options ))
       sys.stderr.write( "other args: %s\n"%(args))
   if options.obsid == "obsid":
       sys.stderr.write( f"The OBSID is a required argument.\n")
       parser.print_help()
       parser.exit       
   elif type(options.radec) != list:
       sys.stderr.write( "ra and dec must be given as a list \n")
       parser.print_help()
   elif len(options.radec) != 2:
       sys.stderr.write( "Provide a value for only one ra and dec\n")
       parser.print_help()

   photometry(options.obsid,
       target=options.target,
       radec=options.radec,
       interactive=options.interactive,
       outfile=options.outfile,
       do_only_band=options.do_only_band,
       rerun_readout_streak=False,
       snthresh=6.0, 
       timezero=options.timezero,
       datadir=options.datadir,
       figno=None,
       chatter=chatter)     
