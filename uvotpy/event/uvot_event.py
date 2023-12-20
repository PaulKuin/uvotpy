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
# 2019-04-04-
# --- change the events used from rawx,rawy to x,y (sky coord) which are 
#     corrected individually for the pointing
# 2023-11-12 checked photometry pipeline-- moved grism event stuff to bottom
#

from __future__ import division
import numpy as np
import os

CALDB = os.getenv('CALDB')

''' UVOT EVENT DATA PROCESSING 

    processing of event grism data developed for Toutatis (14 Dec 2012) 
    
    added processing of lenticular filter event data (8 May 2013)
    
    Paul Kuin
    
    '''
__version__ = "0.6 2023-11-14"   

print ("uvot_event version",__version__)
    

def degrees2sexagesimal(ra,dec,as_string=False):
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
       if as_string: 
          return newra,newdec
       else: 
          return rah,ram,ras,ded,dem,des


def split_eventlist_to_images(eventfile,times=[],dt=10.0, specfile='usnob1.spec',
   attfile=None,mode='default',ra_src=0.,dec_src=0.,
   catspecdir='/Volumes/users/Users/kuin/specfiles/', 
   preshift=[],cleanup=True,clobber=True,chatter=0):
   """
   Split image event file into snapshots, using times, find the aspect correction, 
   update the attitude file,  
   
   Parameters
   ----------
   eventfile : str
      path of the event file
   times : list
      list of [start,end] swift times (s) as float values
      if times not supplied, the times are taken from the 
      STDGTI extension of the event file, or from tstart and tstop
      note that in the last case there may be a VERY large number 
      of extensions.
   dt : float
      time step in seconds   
   specfile : str
      path of the spec file for the catalog server (e.g., usnob1.spec or GSC.spec)
   attfile : str
      path of the attitude file
   mode : str
      determines operation of this program
   ra_src,dec_src: float,float
      sky position of the/a src (uvot_shift is using that)   
   preshift : list
      shifts in ra,dec in arcsec to apply to the WCS prior to aspect 
      correction (for very large excursions)
   cleanup :  bool
      cleanup parameter value used in uvot/headas/ftools calls  
   chatter : int
      verbosity used in uvot/headas/ftools calls 
      
   Notes
   -----
   
   Prior to processing (after running coordinator) the event file should be screened with 
   uvotscreen aoexpr="ANG_DIST < 100. && ELV > 10. && SAA==0 && SAFEHOLD==0" \
              evexpr="QUALITY.eq.0.or.QUALITY.eq.256"
   (use old version 1.18 uvotscreen) 
   This screening does not screen for bad attitude, which would remove too much data.
   
   I want to be able to determine a better aspect solution using the catalog of choice, 
   updating the attitude file, and split the observation in different images. 
   
   So the first mode 'default" is to split up the event file and determine 
   a better aspect correction, then apply that correction to the attitude 
   file. Product: a better attitude file.
   
   steps 
     - extract raw image event slice with xselect based on times[i]
     - update header 
     - append as extension to raw file
     - run pipe3 to get sky image, retain attcorr file
     - run attcorr to update attitude 
     - append sky image to output sky file
   
   The second mode 'split' is to just believe the attitude file, and split the event 
   list into different sky images.
     -  
   """
   import os
   import numpy as np
   from astropy.io import fits
   from astropy import wcs
   
   teldef = ""
   
   if cleanup: 
      clean = 'yes' 
   else: 
      clean = 'no'
      
   os.system("rm -f timeslice.*")
   evt = fits.open(eventfile)
   evt_hdr = evt['EVENTS'].header
   evt_data = evt['EVENTS'].data
   primary = evt['PRIMARY']
   tstart_all = float(primary.header['tstart'])
   tstop_all = float(primary.header['tstop'])
   filtername = evt_hdr['FILTER'].strip()
   detnam = evt_hdr['DETNAM'].strip()
   # need to check what the values actually are for 'detnam'
   if detnam == '0160' : 
      filtername='UGRISM0160'; filtervv = 'gu' 
   if detnam == '0200' : 
      filtername='UGRISM0200'; filtervv = 'gu'
   if detnam == '0955' : 
      filtername='VGRISM0955'; filtervv = 'gv' 
   if detnam == '1000' : 
      filtername='VGRISM1000'; filtervv = 'gv'
   if filtername.lower() == 'b': filtervv = 'bb'
   if filtername.lower() == 'v': filtervv = 'vv'
   if filtername.lower() == 'u': filtervv = 'uu'
   if filtername.lower() == 'w1': filtervv = 'w1'    
   if filtername.lower() == 'm2': filtervv = 'm2'    
   if filtername.lower() == 'w2': filtervv = 'w2'    
   if filtername.lower() == 'uvw1': filtervv = 'w1'    
   if filtername.lower() == 'uvm2': filtervv = 'm2'    
   if filtername.lower() == 'uvw2': filtervv = 'w2'    
   if filtername.lower() == 'white': filtervv = 'wh'   
    
   teldef = _make_hdr_patch(tstart_all, filtername=filtername,outfile='hdr.patch')
   
   # assuming filename is close to convention do an autopsy 
   eventfilename = eventfile.split('/')[-1]
   eventdir = eventfile.split(eventfilename)[0]
   file1 = eventfilename.split('.')
   if (file1[1] == 'evt') & (file1[0][0:2] == 'sw'):
      obsid = file1[0][2:13]
      filtername = file1[0][14:16] 
   else:  # substitute some typical values
      obsid = '0099909001'
      filtername='zz'
   if attfile == None: attfile="../../auxil/sw"+obsid+"pat.fits"
   skyfile = "sw"+obsid+"u"+filtervv+"_sk.img"
   rawfile = "sw"+obsid+"u"+filtervv+"_rw.img"
   
   ntimes = len(times)
   if ntimes < 1: 
      print ("no time periods found: assuming whole list in one")
      if mode == 'split':
         # extract sky image here ? 
         return
      elif mode == 'default' : 
         times = _make_times(evt,dt)
         ntimes = len(times)
         if ntimes < 1:
            print ("trying some times")
            dt = int((tstart-tstop)/dt)
            t1 = np.arange(tstart_all,tstop_all,dt)
            t2 = t1+dt
            ntimes = len(t1)
            for k in range(ntimes):
                times.append([t1[k],t2[k]])      
   else:
      for i in range(ntimes): 
         if not ((times[i][0] >= tstart_all) & (times[i][1] <= tstop_all) ):
             print ("for times[',i,'] the range is outside that of the event list" )
            
                   
   if mode == 'default':
      # write output raw file main ()
      primary.writeto(rawfile,clobber=clobber)
      
      nblocks = int(ntimes/20)+1
      for kb in range(nblocks):
         nstart = 20*kb
         deltan = np.min([20,ntimes-nstart])
         for i in range(nstart,nstart+deltan):
         
            [tstart,tstop] = times[i] 
            print ("*** time range to do next: %12.1f,%12.1f ")
            make_img_slice(eventdir,eventfilename,tstart,tstop,outfile="./timeslice.evt",chatter=chatter)
            
            teldef = _make_hdr_patch(tstart, filtername=filtername,outfile='hdr.patch')

            command = "uvotrawevtimg eventfile=timeslice.evt attfile="+attfile+" outfile=rawfile.img "+\
                   " x0=0 y0=0 dx=2048 dy=2048 trel="+str(tstart).split('.')[0]+" t1="+str(tstart)+\
                   " t2="+str(tstop)+" teldeffile="+teldef+" clobber=yes chatter="+str(chatter)
            status = os.system(command)
            print (command, status)
         
            # apply header patch
            command = "fthedit rawfile.img @hdr.patch"
            status = os.system(command)
            print (status)
         
            # append timesslice `rawfile` file to rawfile
            command = "fappend rawfile.img[0] "+rawfile  
            status = os.system(command)
            print (status)
         
            status = os.system("mv -f timeslice.evt timeslice.old.evt")
         
         # process from raw to sky (after loop)
         command = "pipe3.pl sw"+obsid+"u"+filtervv+" "+attfile
         print (command)
         status = os.system(command)
         print ("done pipe3 ", status)
         
         if preshift != []: _preshift_skyfile(skyfile,preshift)
         
         # get aspect corrections        
         command = "uvotskycorr what=ID skyfile="+skyfile+\
            " catspec="+catspecdir+specfile+" attfile="+attfile+\
            " corrfile=attcorr2.asp clobber=yes outfile=attcorr2.asp starid='poscorr=120 rotcorr=60' "
         print (command  ) 
         status = os.system(command)
         print ("done uvotskycorr ID ", status)
                
         # and, finally, correct the attitude file      
         command="uvotattcorr attfile="+attfile+" corrfile=attcorr2.asp  "+\
                 "outfile=sw"+obsid+"gat.fits chatter=5 clobber=yes"
         print (command)
         status = os.system(command)
         print ("done uvotattcorr ", status)
      
         # apply uvotskycorr (no rotation correction)
         command = "uvotskycorr what=SKY skyfile="+skyfile+" corrfile=attcorr2.asp "+\
            " catspec="+catspecdir+specfile+" attfile="+attfile+\
            " clobber=yes options=INTERPOLATE outfile=sw"+obsid+"u"+filtervv+"_sk.corrected.img"
         print (command, status)
         status = os.system(command)
         print ("done uvotskycorr SKY")
      
         command = "mv "+skyfile+"  "+skyfile.split('.')[0]+"_"+str(kb)+".img"
         print (command)
         os.system(command)
         command = "mv "+rawfile+"  "+rawfile.split('.')[0]+"_"+str(kb)+".img"
         print (command)
         os.system(command)
         command = "mv attcorr2.asp attcorr"+"_"+str(kb)+"_"+filtervv+".asp"
         print (command)
         os.system(command)
         # write new output raw file main ()
         primary.writeto(rawfile,clobber=clobber)
         
      return
      
   elif mode == "split":            
      # expect to use "sw<obsid>gat.fits attitude file
      #
      #  change this to add GTI 
      #
      primary.writeto(rawfile,clobber=clobber)
      #primary.writeto(skyfile,clobber=clobber)      
      
      for i in range(ntimes):
         [tstart,tstop] = times[i] 
         
         make_img_slice(eventdir,eventfilename,tstart,tstop,outfile="./timeslice.evt",chatter=chatter)
         
      command = ""

     
def _make_times(fitsext,dt):
   """make list of time slices based on GTI in fits using step dt """
   import  numpy as np
   # try to read from stdgti
   extnames = []
   for i in range(1,len(fitsext)): extnames.append(fitsext[i].header['extname'])
   print ("extnames:")
   print (extnames)
   stdgti_present = False
   try:
     n = extnames.index('STDGTI')
     if n > 0: stdgti_present = True
   except: pass
   
   if stdgti_present:
      times = []
      tab = fitsext['STDGTI'].data
      for t in tab: 
         tstart = t[0]
         tstop = t[1]
         print ("times :",tstart," - ",tstop)
         tbegin = np.arange(tstart,tstop,dt)
         print ("begin times : (",len(tbegin),")",tbegin)
         tend = tbegin+dt
         tend[-1]=tstop
         print (tbegin)
         print (tend)
         for k in range(len(tbegin)): times.append([tbegin[k],tend[k]])  
   else:
      print ("=========>>>> supply the time ranges manually"  )
      times = []
   return times     
     
   
def _position_shift_event_list(evtfile, time, delta_ra, delta_dec, 
     delta_t=10, chatter=0, clobber=True):
   ''' for version 1.0 & is under development 2023-12-13
   
   Parameters
      evtfile: path
         the event file to correct
      time, delta_ra,delta_dec: float   
         the amount to correct RA and Dec at the given times
      delta_t : float
         the time step for each bin   
         
   Notes:
     Call the Simon/Mat/Sam code first to get the shift to apply in ra,dec 
     
     This is also done in the program
     
     WARNING: This does not work the same way as making the sub-images and 
     aligning those with uvot_shift. The resulting eventfile is not corrected 
     properly! The difference may be the use of attjumpcorr or coordinator? 
     
     For now this has been diabled (2023-12-18 npmk)
   
   '''
   import os
   from astropy.io import fits
   
   if evtfile.split('.')[-1] == 'gz':
       gzipped = True
       evtfn = evtfile.rsplit(".gz")[0]
       os.system(f"gunzip {evtfile}")
   else: 
       gzipped = False 
       evtfn = evtfile   

   # update the event file
   with fits.open(evtfn,mode='update') as evt:
       # extract the time, RA, DEC
       T = evt['EVENTS'].data['TIME']
       X = evt['EVENTS'].data['X']
       Y = evt['EVENTS'].data['Y']
       # select the items that need changing
       if np.isscalar(delta_t):
          dt = delta_t * np.ones( len(time) )
       for t_,dt_,dra,dde in zip(time,dt,delta_ra,delta_dec):
           if chatter > 1: print (f"event updates for {t_} +- {dt_} , deltas = {dra},{dde}")
           qt = (T >= t_ - dt_) & (T <= t_ + dt_) 
           X[qt] -= np.int(dra+0.5)
           Y[qt] -= np.int(dde+0.5)
           # unsure of sign correction..

       evt['EVENTS'].data['Y'] = Y
       evt['EVENTS'].header['COMMENT'] = f"updated X,Y for TIME={time[0]-dt[0]}-{time[-1]+dt[-1]}"
       evt.flush()
       evt.close()
       print (f"tried to update new event file {evtfn}")
   if gzipped: 
       if clobber:
          os.system(f"gzip -f {evtfn}") 
          print (f"gzipped {evtfn}")
       else:   
          os.system(f"gzip {evtfn}") 
          print (f"gzipped {evtfn}")
   if chatter > 1: return (X,Y)       


def process_image(evtfile, dt, 
    attfile=None, drada=15, drb=30,
    skip_attcorr=False, do_skycorr=True, 
    ra_src=0.,dec_src=0., ds9=True,
    logfile='process_image.log',
    cleanup=True,clobber=True,
    chatter=1):
    """

    take an event file with GTI's and replace the GTI's with smaller 
    time slices according to the input, but retaining the original 
    times as outer boundaries. 
    
    Parameters
    ==========
    file : path
       event file
    dt : float, list
       if one value, split GTIs into smaller slices using dt value, e.g., 20 
       if list, all elements of dt must be a list of start,stop times
          in swifttime (seconds since Jan 1 2005, 00:00:00.0) 
    attfile : path (optional)
       path to attitude file to create an image file with the new 
       GTIs.      
    drada, drb : float (optional)
       radius in arcsec for uvot_shift to search for fit 
       this will run when the skyfile has been produced.  
       Here drb is used for the rough initial shift, 
       drada for the fine adjustment
    ra_src,dec_src: float,float
      sky position of the/a src (uvot_shift is using that)   
    ds9 : bool
       plot the images in ds9 ?
    
    output
    ======
    file_new_uf.evt : path 
       output file name based on the event file name with _new_ added
       and placed in local directory
    fileuxx_rw.img, fileuxx_sk.img : path (optional)
       output raw image and sky image files when attitude file is given.
       if this option is not used, the likely command needed is printed on 
       output.      
       
    method
    =======
    grab the GTI's. There should be one summary and then for each time 
    slice a new one. Copy the old GTIs to a file; remove them from the 
    event file. Create a new summary GTI and extra GTIs for each time 
    slice. Add to the event file.   
    
    WARNING
    ======= 
    Issue 1:
       the toss/stall/bloc times were copied from the original gti header for
       the whole stretch rather than determined by examining the event list. 
       Depending on how the correction is being made for any non-zero loss, 
       (or not) this may give inaccurate results for the photometry. 
    Issue 2:
       Generated image file size can be something like 9GB, which means the file 
       cannot be kept in memory which slows things down. 
       We can either process in chunks, or use a subsection of the image, both 
       require a bit of work. Plan to do it by original GTIs. 
    Issue 3: 
       The initial aspect correction made using uvot_shift should be applied to
       each GTI instead of the whole file. <<< done, but sometimes last one not updated?
    Issue 4: 
       The updated event list (with moved X=ra, Y=dec values) is smaller and more 
       valuable. Do we need to keep a history record of the changes? <<< no
    Issue 5:
       We need a method to compare symmetry of the source in the selected aperture 
       so that we have an independent check on the source position accuracy.
    Issue 6:
       Bad sensitivity patches: can screen perhaps in advance but we will not do that 
       and rely on uvotsource
    Issue 7:
       There may be some remaining peculiarities in the img headers that I did not
       fix, like USTALL ...
    Issue 8:
       need to clean up intermediate products <<< flag cleanup added
    Issue 9: 
       For doing Skycorr, it would be good to have a better catalogue than USNO-B1, 
       perhaps the Gaia catalogue, or create a new catalogue with predicted uv bright 
       sources ? 
    Tested for a constant delta time of 10.0s. Checked result with fverify. 
    
    Example
    =======
    In [3]: evtfile='sw00034849012um2w1po_uf.evt'

    In [4]: uvot_event.process_image(evtfile,dt=50.0,\
            attfile='../../auxil/sw00034849012pat.fits',\
            drb=15, drada=12., skip_attcorr=False, chatter=2)
    
    History: 
    2015-05-02 NPMKuin initial version   
    2017-05/06 NPMK complete overhaul 
    2017-06-09 NPMK rewrite parts to avoid skipping over data gaps  
    2023-11-12 NPMK review - was there an offset in the output positions? 
       fixed print statements
    """
    import os,sys
    import numpy as np
    from astropy.io import fits, ascii as ioascii
    from astropy.coordinates import SkyCoord  # High-level coordinates
    from astropy.coordinates import ICRS, Galactic, FK4, FK5  # Low-level frames
    from astropy.coordinates import Angle, Latitude, Longitude  # Angles
    import astropy.units as units
    from uvotpy import uvotmisc
    from astropy.table import Table
    
    __version__ = "0.2 20170611"

    if clobber: sclobber = 'yes'
    logf = open(logfile,'w')
    
    # check if evt file has .gz 
    if evtfile.rsplit(".")[-1] == "gz":
       evtgzip = True
    else: evtgzip = False    
    
    # run coordinator on whole event file, do basic uvotscreen, run uvot_shift to 
    # determine overall shift needed, apply shift to input coordinator (and keep using that)
    #
    evtpath = evtfile.split('sw0')[0]
    outfile_ = "evtfile.tmp"
    outfroot = evtfile.split('/')[-1].split("_uf.evt")[0]
    outfile = evtfile.split('/')[-1].split("_uf.evt")[0]+"_new_uf.evt"
    #if evtgzip: outfile = outfile+".gz"
    shiftedf = outfroot[:16]+"_sk.img"
    
    if not os.access(evtfile,os.F_OK): raise IOError(
        "process_image: event file %s not found"%(evtfile)) 

    # copy the event list 
    if os.system("cp "+evtfile+" "+outfile): 
        raise IOError("copy event file failed")
        
    evt = fits.open(outfile,)
    if not 'EVENTS' in evt:  raise IOError(
      "when recreate_newGTItable: EVENTS extension not found in evtfile %s"%(evtfile)) 
    if not 'STDGTI' in evt:  raise IOError(
      "when recreate_newGTItable: STDGTI extension not found in evtfile %s"%(evtfile)) 
    if not 'WINDOW' in evt:  raise IOError(
      "when recreate_newGTItable: WINDOW extension not found in evtfile %s"%(evtfile)) 
    band = evt[0].header['filter']
    fkey = {'UVW2':'w2','UVM2':'m2','UVW1':'w1','U':'uu','B':'bb','V':'vv','WHITE':'wh','WH':'wh'}
    UVOTPY = os.getenv('UVOTPY')+"/calfiles/"
    minmatch = {'UVW2':'3','UVM2':'3','UVW1':'3','U':'10','B':'10','V':'10','WHITE':'10','WH':'10'}
    noatt = False
    uat1 = attfile 
          
    # the first time, do not skip the attitude correction, nex time use the `outfile`
    
    print (f"PASS 1 #############  PASS 1  ###########")
    
    ra_ = str(evt['window'].header['ra_pnt'])
    dec_ = str(evt['window'].header['dec_pnt'])
    if not skip_attcorr:  
        # pointing position 
        ra_ = str(evt['window'].header['ra_pnt'])
        dec_ = str(evt['window'].header['dec_pnt'])
        
        # convert  position to sexagesimal         
        c = SkyCoord(ra_,dec_, unit="deg")  # defaults to ICRS frame
        ras = c.ra.to_string(sep=':',unit='hour')+'  '
        decs = c.dec.to_string(sep=':')
        if (ra_src == 0.) & (dec_src == 0): 
            ra_src_ = ras
            dec_src_ = decs
        else:  
            ra_src_,dec_src_ =  degrees2sexagesimal(ra_src,dec_src,as_string=True)
            #use source position for uvot_event ; needs to be decimal notation
        
        evt.close()
        
        # here we should first run the ftools to update the attitude file and correct for
        # rotation as well as offset. However, for now we just jump to the uvot_shift
        # method and then do the split in small time slices by changing the GTI table
        # before generating an image file with many extensions.
        uat1 = attfile
        
        # try attjumpcorr first to correct settling jumps in attitude file data
        command=f"attjumpcorr {attfile} {attfile}.2"
        if not os.system(command):
            os.system(f"mv {attfile} {attfile}.ori;mv {attfile}.2 {attfile}" )
            logf.write( f"Ran attjumpcoor ftool on attitude file; original in {attfile}.2\n")
        if do_skycorr:
            # first get an initial att correction and correct the event list:
            command = "uvotimage infile="+outfile+ " prefix=sky1 attfile="+attfile +\
                " teldeffile='CALDB' alignfile='CALDB' mod8corr=no "+\
                " ra="+str(evt['window'].header['ra_pnt'])+\
                " dec="+str(evt['window'].header['dec_pnt'])+\
                " roll="+str(evt['window'].header['pa_pnt'])+\
                "  clobber="+sclobber+" chatter="+str(chatter)
            if chatter > 1: logf.write(command+'\n')    
            if os.system(command): logf.write( "could not create sky1 image file\n")
            
            command = "uvotskycorr what=ID skyfile=sky1u"+fkey[band.upper()]+"_sk.img "+\
                " attfile="+attfile+" outfile=uac1.hk catspec="+UVOTPY+"usnob1.spec"+\
                " starid='mag.min=11 mag.max=19 min.matches="+minmatch[band]+"'  "+\
                " corrfile=none  clobber="+sclobber+" chatter="+str(chatter)
                #" starid='mag.min=13 mag.max=19 min.matches=4'  options=FORCE  "
            if chatter > 1: logf.write (command+'\n' )   
            if os.system(command): 
                logf.write("could not create uac1.hk attitude correction file\n")
            
            command = "uvotattcorr attfile="+attfile+" corrfile=uac1.hk "+\
                      " outfile=uat1.fits chatter="+str(chatter)
            if chatter > 1: logf.write (command+'\n')    
            if os.system(command): 
               logf.write( "could not update attitude file with uac1.hk correction\n")
            noatt=False 
            uat1 = 'uat1.fits'  
            if not os.access('uat1.fits',os.F_OK): 
                noatt=True
                uat1 = attfile    
            
            # Here we can make use of the success or failure of aspcorr - perhaps use 
            # only sub-image centered aound source to achieve speedup ? (screen as such)
            
        command =  "coordinator eventfile="+outfile+" eventext=EVENTS attfile="+uat1+\
               " aberration=n randomize=y seed=1411 ra="+ra_+\
               " dec="+dec_+" teldef=CALDB"
        if chatter > 1: logf.write(command +'\n')   
        if os.system(command): logf.write("problem with coordinator (initial)\n")
            
        command = "uvotscreen infile="+outfile+" attorbfile="+attfile+\
                ' outfile='+outfile_+' badpixfile="CALDB"'+\
                ' aoexpr="ELV > 10. && SAA == 0"'+\
                ' evexpr="QUALITY.eq.0.or.QUALITY.eq.256" clobber='+sclobber
        #       ' aoexpr="ANG_DIST < 100. && ELV > 10. && SAA == 0"'+\ uvotscreen failed since ANG_DIST was not there
        if chatter > 1: logf.write(command+'\n' )   
        if os.system(command): logf.write( "uvotscreen failed (initial)\n")
        if cleanup:
           if os.system("mv "+outfile_+" "+outfile): print ("move of screened outfile failed")
        else:
           if os.system("cp "+outfile_+" "+outfile): print ("copy of screened outfile failed")
        
        # make an image file from the events
        evt = fits.open(evtfile,)
        stdgti = evt['STDGTI']
        window = evt['WINDOW']
        
        command = "uvotimage infile="+outfile+ " prefix=sky1 attfile="+attfile +\
                " teldeffile='CALDB' alignfile='CALDB' mod8corr=no "+\
                " ra="+ra_+" dec="+dec_+\
                " roll="+str(evt['window'].header['pa_pnt'])+\
                "  clobber="+sclobber+" chatter="+str(chatter)
        if chatter > 1: logf.write (command+'\n' )   
        if os.system(command): logf.write("could not create sky1 image file\n")
        
        skyfile = "sky1u"+fkey[band.upper()]+"_sk.img"
           
        evt.close()
        
        logf.write ("running uvot_shift to get better attitude\n ")
        
        command = 'uvot_shift '+ra_src_+"  "+dec_src_+"  "+str(drb)+"  "+skyfile
        if chatter > 1: logf.write (command +'\n')
        if os.system(command): print (f"ERROR: something wrong with the uvot_shift execution?")
        t = Table.read( "shifts.txt",format='ascii',fast_reader={'exponent_style': 'D'})
        if chatter > 0:
            sys.stdout.write("uvot_shift results on whole original event file:\n%s\n"%t)
        x1 = t['col3']
        x2 = t['col5']
        e1 = t['col4']
        e2 = t['col6']
        d_ra = x1.mean()/3600.
        d_dec = x2.mean()/3600.
        if (len(t) < 4.):         
           #  need to convert deltas from sexagesimal to deg. 
           ra  = float(ra_) - d_ra
           dec = float(dec_) - d_dec
        else: 
           # use the mean shift after filtering out outliers 
           # 20231112 - change to median filtering ? 
           q = (abs(e1) > abs(e1.mean()) - 2*e1.std()) & (abs(e1) > abs(e1.mean()) + 2*e1.std())
           if len(x1[q]) > 0: d_ra = x1[q].mean()/3600.
           ra = float(ra_)  - d_ra  
           q = (abs(e2) > abs(e2.mean()) - 2*e2.std()) & (abs(e2) > abs(e2.mean()) + 2*e2.std())
           if len(x2[q]) > 0: d_dec = x2[q].mean()/3600.
           dec = float(dec_) - d_dec
        ra_  = str(ra)
        dec_ = str(dec)   
        if not cleanup: os.system("mv shifts.txt shifts_original_evtfile.txt")
        if cleanup: 
            os.system("rm sky1u*")
        else:
            os.system("mv "+skyfile+"  original_"+skyfile)   

# edit the X, Y columns in the EVENT list extension using ftool fcopy with a filter

        n_gti = len(t) # some gti may have been screened
        for k in range(0,n_gti,1):
           
           d_ra_  = x1[k]
           d_dec_ = x2[k]
           command = 'fcopy "'+outfile+\
           '[EVENTS][gtifilter(`'+outfile+'[GTI'+str(k+1)+\
            ']`)][col TIME, RAW*, DET*, EXPREF, QUALITY, X = X - '+\
            str(d_ra_)+', Y = Y - '+str(d_dec_)+']"  '+outfile+'.upd'
           if chatter > 1: sys.stdout.write( command +'\n')   
           if os.system(command): logf.write( "problem updating the event file RA, DEC\n")
           if cleanup:
              os.system("mv "+outfile+".upd "+outfile)
           else:   
              os.system("cp "+outfile+".upd "+outfile)
           logf.write("updated event sky positions in file :"+outfile+'\n')
        
        else:
           command = 'fcopy "'+outfile+\
           '[EVENTS][col TIME, RAW*, DET*, EXPREF, QUALITY, X = X - '+str(d_ra)+\
           ', Y = Y - '+str(d_dec)+']"  '+outfile+'.upd'
           if chatter > 1: sys.stdout.write( command +'\n' )  
           if os.system(command): logf.write("problem updating the event file RA, DEC\n")
           if cleanup:
              os.system("mv "+outfile+".upd "+outfile)
           else:   
              os.system("cp "+outfile+".upd "+outfile)
           logf.write("updated event sky positions in file :"+outfile+'\n')
    
    command = "ls sky1*"
    logf.write(command+'\n')
    sys.stdout.write(command+'\n')
    os.system(command)
    
    
    ################ second pass: use selected dt to update positions 

    print (f"PASS 2 #############  PASS 2  ###########")
    evt = fits.open(outfile)
    
    stdgti = evt['STDGTI']
    window = evt['WINDOW']
    framtime = evt['EVENTS'].header['framtime']
    # init last GTIn number (starts at 1)
    gti_n = 0
    # number of original GTIs
    gti_number = evt['window'].header['naxis2']
    
    # iterate over each existing GTI and create new WINDOW extension with the desired gti
    new_tstart = []
    new_tstop  = []
    deadtimecorr = []
    windx0=[]
    windy0=[]
    winddx=[]
    winddy=[]
    tossloss = []
    blocloss= []
    stalloss = []
    
    # set up times for the new GTI's
    if type(dt) == list: 
        if chatter > 4: logf.write("dt is list\n")
        ts1 = []   # start time bins
        ts2 = []   # stop  time bins
        for r in dt: 
            # check range is within the existing GTIs
            in_range = False
            for rec in stdgti.data:
               if (r[0] >= rec[0]) & (r[1] <= rec[1]): in_range = True
            if in_range: 
               ts1.append(r[0])
               ts2.append(r[1])  
            else:
               sys.stdout.write("The following requested period is not allowed: %s\n",r)    
        ts1 = np.asarray(ts1)
        ts2 = np.asarray(ts2)    
    elif np.isscalar(dt):   # assume dt is a number 
        if chatter > 4: logf.write( "dt is a scalar\n")
        ts1 = []
        ts2 = []
        for utstart,utstop in stdgti.data:
           tt = np.arange(utstart,utstop,dt)
           for t in tt:
               ts1.append(t)
               if t+dt < utstop:  # 2023-11-12 modified. 
                  ts2.append(t+dt)
               else: 
                  ts2.append(utstop)
    if chatter > 3: 
        for a,b in zip(ts1,ts2):
            logf.write("new gti: %f, %f\n"%(a,b))

    wstart = []
    wstop = []
    wrec = []
    # grab the old start/stop times from the WINDOW data
    for rec in evt['window'].data:
        wstart.append(rec[8])
        wstop.append(rec[9])
        wrec.append(rec)
        
    # now create an update to the WINDOW extension including 
    # each new gti
    # loop over the new gti
    # filter out times without original GTI
    
    for t1,t2 in zip(ts1,ts2): 
        
        # find index wrec:
        i_wrec = -1
        for k in range(len(wrec)):
            if (t1 >= wstart[k]) & (t1 <= wstop[k]):
                i_wrec = k
        if i_wrec > -1: 
            rec = wrec[i_wrec]
            new_tstart.append(t1)
            new_tstop.append(t2)
            deadtimecorr.append(rec[14] )
            windx0.append(rec[4] )
            windy0.append(rec[5] )
            winddx.append(rec[6] )
            winddy.append(rec[7] )
            tossloss.append(rec[15] )  # this needs to be replaced by a HK value for times t1 - t2
            blocloss.append(rec[16] )
            stalloss.append(rec[17] )
    # change list to numpy arrays        
    new_tstart = np.asarray(new_tstart)
    new_tstop = np.asarray(new_tstop) 
     
    # now I have lists for the start, stop and dead time correction and can make new 
    # extensions ; write a new file
    hdulist = [evt[0]]
    # (update keywords: creator, checksum
    events_hdu = evt['events'] 
    hdulist.append(events_hdu)
    
    stdgti_hdr = evt['stdgti'].header 
    gti_hdu= evt['gti1']
    gti_hdu.add_checksum()
    # needs update of extname,  tstart,tstop date-obs date-end telapse ontime 
    #     origin creator date expid ??attflag?? seqpnum checksum utcfinit datasum 
            
    # create new and write stdgti to hdulist
    # update the STDGTI #
    cola = fits.Column(name='START',format='D',array=new_tstart,unit='s')
    colb = fits.Column(name='STOP',format='D',array=new_tstop,unit='s')
    cols = fits.ColDefs([cola,colb])
    stdgti_hdu= fits.BinTableHDU.from_columns(cols,header=stdgti_hdr)
    stdgti_hdu.add_checksum()
    hdulist.append(stdgti_hdu)
    if chatter > 3: logf.write ("new stdgti data %s\n"%stdgti_hdu.data)
    if chatter > 3: logf.write ("added stdgti hdu\n" )
    
    # write gti(n) to hdulist
    for s1,s2 in zip(new_tstart,new_tstop):
        if chatter > 3: logf.write ("gti times %f - %f\n"%(s1,s2))
        gti_n += 1
        gti_hdu.header['TSTART'] = s1
        gti_hdu.header['TSTOP'] = s2
        gti_hdu.header['EXTNAME'] = 'GTI'+str(gti_n)
        gti_hdu.header['TELAPSE'] = s2-s1
        gti_hdu.header['ONTIME'] = s2-s1
        gti_hdu.data['START'] = s1
        gti_hdu.data['STOP'] = s2 
        gti_hdu.header['DATE-OBS'] = uvotmisc.swtime2JD(s1)[3]
        gti_hdu.header['DATE-END'] = uvotmisc.swtime2JD(s2)[3]
        #gti_hdu.header['UTCFINIT'] = "" -- keep the value 
        gti_hdu.header['EXPID'] = np.int(s1) 
        gti_hdu.add_checksum()
        hdulist.append(gti_hdu.copy())
        if chatter > 3: logf.write ("gti data%s\n"%gti_hdu.data)
        if chatter > 3: logf.write ("added gti hdu number GTI"+str(gti_n)+'\n')
        
    # write new WINDOW GTI to hdulist  
    window_hdr = evt['window'].header
    windx0 = np.array(windx0)
    windy0 = np.array(windy0)
    winddx = np.array(winddx)
    winddy = np.array(winddy)
    deadtimecorr = np.array(deadtimecorr)
    tossloss = np.asarray(tossloss)
    blocloss = np.asarray(blocloss)
    stalloss = np.asarray(stalloss)
    col1 = fits.Column(name='START',   format='D',array=new_tstart,unit='s')
    col2 = fits.Column(name='EXPID',   format='J',array=np.array(new_tstart,dtype=np.long),)
    col3 = fits.Column(name='EXPREF',  format='I',array=np.arange(1,gti_n+1,1),)
    col4 = fits.Column(name='IMAGEEXT',format='I',array=np.arange(1,gti_n+1,1))
    col5 = fits.Column(name='WINDOWX0',format='J',array=windx0,unit='pixel')
    col6 = fits.Column(name='WINDOWY0',format='J',array=windy0,unit='pixel')
    col7 = fits.Column(name='WINDOWDX',format='J',array=winddx,unit='pixel')
    col8 = fits.Column(name='WINDOWDY',format='J',array=winddy,unit='pixel')
    col9 = fits.Column(name='UTSTART', format='D',array=new_tstart,unit='s')
    col10= fits.Column(name='UTSTOP',  format='D',array=new_tstop,unit='s')
    col11= fits.Column(name='UELAPSE', format='D',array=new_tstop-new_tstart,unit='s')
    col12= fits.Column(name='UONTIME', format='D',array=new_tstop-new_tstart,unit='s')
    col13= fits.Column(name='ULIVETIM',format='D',array=(new_tstop-new_tstart)*deadtimecorr,unit='s')
    col14= fits.Column(name='UEXPOSUR',format='D',array=(new_tstop-new_tstart)*deadtimecorr,unit='s')
    col15= fits.Column(name='UDEADC',  format='D',array=deadtimecorr,)
    col16= fits.Column(name='UTOSSL',  format='D',array=tossloss,)
    col17= fits.Column(name='UBLOCL',  format='D',array=blocloss,)
    col18= fits.Column(name='USTALL',  format='D',array=stalloss,)
    cols = fits.ColDefs([col1,col2,col3,col4,col5,col6,col7,col8,col9,
             col10,col11,col12,col13,col14,col15,col16,col17,col18])
    window_hdu= fits.BinTableHDU.from_columns(cols,header=window_hdr) 
    hdulist.append(window_hdu)
    if chatter > 3: logf.write("hdu list complete\n")
    
    # create the new event mode file with the GTIs for small timesteps
    fitsout = fits.HDUList(hdulist)
    fitsout.verify()
    
    if chatter > 0: logf.write ("writing event file with updated GTI list, named "+outfile+'\n')
    fitsout.writeto(outfile, clobber=clobber,checksum=True)       
    evt.close()
    fitsout.close()
    
    logf.write("event file %s with new GTIs created \n"%(outfile))
    
    # now that the event file has been updated with GTIs, create the image file from it
    
    # define the files that will be created 
    rawfile = outfroot+"_rw.img"
    skyfile = outfroot+"_sk.img"
            
    # make image file (chunck this so that the image file is no larger than 2GB ?
    # could also have same script only do the event file shifts in chuncks of 1 to 10 img ?
            
    command = "uvotimage infile="+outfile+" prefix=sky2  attfile="+uat1+\
                " teldeffile='CALDB' alignfile='CALDB' mod8corr=yes "+\
                " ra="+ra_+" dec="+dec_+\
                " roll="+str(evt['window'].header['pa_pnt'])+\
                "  clobber="+sclobber+" chatter="+str(chatter)
    if chatter > 1: logf.write (command +'\n' )  
    if os.system(command): logf.write("could not create image file (second go)\n")
     
     # rename files
    command = "mv sky2u"+fkey[band.upper()]+"_rw.img "+rawfile 
    if chatter > 1: logf.write(command+'\n' )   
    os.system(command)  
                          
    command = "mv sky2u"+fkey[band.upper()]+"_sk.img "+skyfile 
    if chatter > 1: logf.write (command +'\n')   
    if os.system(command): logf.write("could not update final sky file name\n")
                                       
    # convert pointing position to sexagesimal         
    c = SkyCoord(evt['window'].header['ra_pnt'],evt['window'].header['dec_pnt'] , unit="deg")  # defaults to ICRS frame
    ras = c.ra.to_string(sep=':',unit='hour')+'  '
    decs = c.dec.to_string(sep=':')
        
    #
    # we need to add the source position here for uvot_shift, not the pointing 
    if (ra_src == 0.) & (dec_src == 0): 
            ra_src_ = ras
            dec_src_ = decs
    else:  
            ra_src_,dec_src_ =  degrees2sexagesimal(ra_src,dec_src,as_string=True)
        
    logf.write ("running uvot_shift to get better attitude & pop open in ds9\n")
        
    command = 'uvot_shift '+ra_src_+"  "+dec_src_+"  "+str(drada)+"  "+skyfile   
    
    if chatter > 1: logf.write (command+'\n' )   
    if not os.system(command):
        logf.write ("uvot_shift completed; now inspect the images in ds9 \n")
        command = f"uvotimsum {skyfile}_shifted {shiftedf} exclude=none clobber={clobber}"
        if os.system(command): print (f"unsuccessful in creating summed image {shiftedf}")
        else: print (f"created summed image {shiftedf}")
        
        # try applying shifts found
        t = Table.read( "shifts.txt",format='ascii',fast_reader={'exponent_style': 'D'})
        if chatter > 0:
            sys.stdout.write("uvot_shift results on whole original event file:\n%s\n"%t)
        time1 = t['col1']   # start time
        time2 = t['col2']   # stop time
        x1 = t['col3']      # ra offset in deg
        x2 = t['col5']      # dec offset in deg
        e1 = t['col4']      # rms error in ra offset
        e2 = t['col6']      # rms error in dec offset

        os.system("mv shifts.txt "+outfroot+"_shifts.txt")
        command ="ds9 -scale log -memf "+skyfile+\
        "_shifted -regions load all src.reg &" 
        # -crop "+ras+" "+decs+" 90 90 wcs FK5 arcsec&"
        # -saveimage ds9.jpg&"
        logf.write(command+'\n')
        if ds9: os.system(command)

        # sft = open(  outfroot+"_shifts.txt")
        # read which extensions of the file are good, and update the aspcorr ? 
        # sft.close()
        
        # WARNING the following does not work as expected.  
        # times = (time2.data + time1.data)*0.5
        # delta_ra = x1.data
        # delta_dec = x2.data
        # Z = _position_shift_event_list(outfile, times, delta_ra, delta_dec,chatter=chatter,clobber=clobber)
              
        if cleanup:
            command ="rm "+rawfile+" "+skyfile
            logf.write(command+'\n')
            os.system(command)
            
    logf.close() 
    if evtgzip: 
        print (f"applying gzip {outfile}:")
        os.system(f"gzip {outfile}")   
    print (f"for light curve next, run:\nuvotmaghist infile={skyfile}_shifted srcreg=src.reg bkgreg=bg.reg outfile=phot.fits exclude=NONE,1")      
    #print (f"alternatively use the updated eventlist file with the uvotevtlc ftool.")
    print (f"for photometry, use {shiftedf}")
    if chatter > 1: return Z



''' 
NOTES: (update 2017-05-21) 

What to run? 


    # provide template records
    #stdgti_rec = evt['stdgti'].data[0]
    #stdgti_dtype=[('START', '>f8'), ('STOP', '>f8')]
    #window_rec = evt['WINDOW'].data[0] 
    #window_dtype=[('START', '>f8'), ('EXPID', '>i4'), ('EXPREF', '>i2'), 
    #  ('IMAGEXT', '>i2'), ('WINDOWX0', '>i4'), ('WINDOWY0', '>i4'), 
    #  ('WINDOWDX', '>i4'), ('WINDOWDY', '>i4'), ('UTSTART', '>f8'), 
    #  ('UTSTOP', '>f8'), ('UTELAPSE', '>f8'), ('UONTIME', '>f8'), 
    #  ('ULIVETIM', '>f8'), ('UEXPOSUR', '>f8'), ('UDEADC', '>f8'), 
    #  ('UTOSSL', '>f8'), ('UBLOCL', '>f8'), ('USTALL', '>f8')]
 STDGTI columns  (one for each GTI)
 ['START', 256490530.20506001
 'STOP']   256490646.97270074
 
 WINDOW columns (one each for each GTI)   
 
 ['START',     / Matching GTI extension start time  in sec    256490530.20506001
 'EXPID',      / Exposure ID                        in sec    256490530
 'EXPREF',     / Exposure GTI reference                               1
 'IMAGEXT',    / Name of image extension containing this data         0
 'WINDOWX0',   / Lower left hand RAWX               pixel             0
 'WINDOWY0',   / Lower left hand RAWY               pixel             0
 'WINDOWDX',   / RAW Window width                   pixel          2048
 'WINDOWDY',   / RAW Window height                  in pixel       2048
 'UTSTART',    / GTI TSTART value                             256490530.20506001
 'UTSTOP',     / GTI TSTOP value                              256490646.97270074
 'UTELAPSE',   / GTI TELAPSE value                                  116.76764073967934
 'UONTIME',    / GTI ONTIME value                                   116.76764073967934
 'ULIVETIM',   / GTI LIVETIME value                                 111.12857066232596  = udeadc*uontime
 'UEXPOSUR',   / GTI EXPOSURE value                                 111.12857066232596
 'UDEADC',     / GTI DEADC value                                      0.95170691090757698 depends on frametime 
 'UTOSSL',     / Image mode TOSSLOSS value                            0.0
 'UBLOCL',     / Image mode BLOCLOSS value                            0.0      
 'USTALL']     / Image mode STALLOSS value                            0.0

What next? 

iPython>uvot_event.recreate_evt_with_new_GTI_table('sw00035009034um2wupo_uf.evt', 30.0, )

Shell>uvotimage infile=sw00035009034um2wupo_new_uf.evt prefix="QQ" 
attfile=sw00035009034pat.fits.gz teldeffile=CALDB alignfile=CALDB 
mod8corr=yes ra=110.4727 dec=71.343434 roll=319.76 clobber=yes

uvot_shift 7:21:53.45 71:20:36.36 10 QQum2_sk.img

Note: making dt too small: not enough stars in image for uvot_shift  dt = 10 fail, dr = 30 fine
      search radius too large: uvot_shift fails on some 60" fail 10" success
      running attjumpcorr ahead of uvotimage may be a good idea. 

      the shifts.txt list tstart, tstop, xpix_shift?, ypix_shift? 
      
      NOT (need to update the att file, then uvotimage without correcting attitude data
           any further. then   *** Update att file from shifts.txt ? how? )
           
      problem with uvm2 image: bright source generates in wings counts that can mimic 
      a weak source and throw off uvot_image. Only solution seems to be to 
      make dt larger. 
         
      then run uvotmaghist with 3" and 5" apertures to get magnitudes/rates for lc. 
      *** Use an exposure map *** to make sure background is always taken from image.

   or

      uvotevtlc infile=sw00035009034um2wupo_uf.evt outfile=orig_m2_30.lc 
         srcreg=src.reg bkgreg=bg.reg timebinalg="u" timedel=30 clobber=yes
         
   or use Sam Oates code to make a light curve.
   
*********

Sergio: problem with screening takes too many data away. Solution include Q=256.
Sergio:  I process the data with commands like these

coordinator eventfile=sw00033000021ubbpo_uf.evt
    eventext=EVENTS 
    teldef=CALDB 
    attfile=sw00033000021uat.fits.gz 
    aberration=n 
    randomize=y seed=836 
    ra=137.3901 dec=33.123

uvotscreen infile=sw00033000021um2w1po_uf.evt 
    attorbfile=sw00033000021sao.fits.gz 
    outfile=sw00033000021um2w1po_cl.evt 
    badpixfile=CALDB 
    aoexpr="ANG_DIST < 100. && ELV > 10. && SAA == 0" 
    evexpr="QUALITY.eq.0.or.QUALITY.eq.256"
    
    (QUALITY=256 <= loss function not in good range)
    
    Then, check that you don't have trailed image slices.
    
*********
 
Sam: Using Mat's uvot_shift to update the event file position (ra,dec)?. - Check her 
perl program. I would expect some kind of interpolation after vetting the output 
from uvot_shift. (I do not have Sam's program) :/

This makes more sense then updating the attitude file, because the 
attitude should be dominated by the startrackers.  However, if we 
would get a better attitude which would also help the XRT, then it 
would make sense to update the attitude file. However, I think that 
the spatial resolution of the XRT is worse and it doesn't matter.

'''

#
#  Processing horizons positions    
#
def _edit_horizons_file(file):
   ''' 
   For solar system objects:
   
   After downloading the output from the Horizons website -> the orbital parameters, etc. 
   put hash marks in front of the header (->comments) and 
   post header to get clean data table to read with rdList
   '''
   
   fin = open(file,'U')
   fout = open(file+'.data','w')
   header = True
   print ("header = ",header)
   try: 
     # first line
     r = fin.readline()
     if len(r) > 4:
        if header:        
           fout.write("#%s"%(r))
        else:
           fout.write("%s"%(r))
     # the rest
     keep_going = True
     while keep_going:
        r = fin.readline()
        keep_going = len(r) > 0
        if r[0:5] == "$$EOE": 
           header = not header
           print (r)
           print (header)
        if header:        
           fout.write("#%s"%(r))
        else:
           fout.write("%s"%(r))
        if r[0:5] == "$$SOE": 
           header = not header
           print (r)
           print (header)
   except:
      pass
   fout.close()
   fin.close()
      
      
def get_target_position(file):      
   '''For solar system objects:
   
   read the RA,DEC from the horizons file
   
   returns a tuple interpolating function for (ra,dec)= returned_function(swifttime)
    
   '''                   
   from scipy import interpolate
   from uvotmisc import rdList, UT2swift
   
   t = rdList(file)
   
   N = len(t)
   #date = []; time  = []; 
   swifttime = []
   ra   = []; dec = []
   #apmag =[]; sbrt  = []; delta = []; deldot=[] ; sot = [], sto = []
   
   for i in range(N):
      #date.append(  t[i][ 0])
      #time.append(  t[i][ 1])
      # convert positions to decimal degrees
      ra.append( 15*(float(t[i][ 2]) + (float(t[i][3]) + float(t[i][4])/60.)/60.) )
      dec.append(    float(t[i][ 5]) + (float(t[i][6]) + float(t[i][7])/60.)/60.  ) 
      #apmag.append( t[i][ 8])
      #sbrt.append(  t[i][ 9])
      #delta.append( t[i][10])
      #deldot.append(t[i][11])
      #sot.append(   t[i][12])
      #sto.append(   t[i][13])
      # convert time to swifttime
      year =  int(t[i][0][0:4])
      month = t[i][0][5:8]
      day =   int(t[i][0][9:11])
      hms=t[i][1].split(':')
      if len(hms) == 3: 
         hour,minute,second=hms
      elif len(hms) == 2:
         hour,minute=hms         
         second = '0.0'
      else:
         hour = hms
         minute = '0'
         second = '0.0'  
      hour = int(hour)
      minute = int(minute)
      isec, msec = second.split('.')
      isec = int(isec)
      msec = int(msec)
      swifttime.append(UT2swift(year,month,day,hour,minute,isec,msec))

   return interpolate.interp1d(swifttime,ra,kind='linear'), interpolate.interp1d(swifttime,dec,kind='linear')

#
#  processing event data
#
def make_img_slice(eventdir,eventfile,tstart,tstop,outfile="./timeslice.evt",
       type='event', chatter=0): 
   ''' run xselect to create an event/image slice '''
   import os
   f = open("./times.lis","w")
   f.write("%f,%f\n"%(tstart,tstop))
   f.close()
   f = open("./xselect_driver.xco","w")
   f.write("uvot\n")
   f.write("read event %s\n"%eventfile)
   if eventdir == "": eventdir = "./"
   f.write(eventdir+"\n")
   f.write("yes\n")
   f.write("set device /NULL \n")
   f.write("filter time file times.lis\n")
   #f.write("filter time scc %f, %f\n"%(tstart,tstop))
   f.write("extract events\n")
   #f.write("yes\n")
   if type == 'event':
      f.write("save events %s\n"%(outfile))
   else: 
      f.write("set image sky\nextract image\nsave image %s\n"%(outfile))
   f.write("yes\n")
   f.write("quit\n")
   f.write("no\n")
   f.close()
   
   command = "xselect @xselect_driver.xco"
   os.system(command)
   
def _make_hdr_patch(tstart,filtername=None,outfile='../work/patch.hdr',chatter=1):
   """
   Write the text file to patch the header of the xselect filtered file
   
   Parameters
   ----------
   tstart : int
      start time (swift time) s
   filtername: {"UGRISM0160"|"UGRISM0200"|"VGRISM0955"|"VGRISM1000"|"WHITE"|"V"|"B"|"U"|"W1"|"M2"|"W2"}
       
   Note
   ----
   the teldef section needs to be replaced with a call to the CALDB    
       
   """
   import os
   from uvotpy import uvotmisc
   
   CALDB = os.getenv("CALDB")
   tim = uvotmisc.swtime2JD(tstart)[3].split('T')
   f = open(outfile,"w")
   filtername = filtername.upper()
   if filtername == 'VV': filtername = 'V'
   if filtername == 'BB': filtername = 'B'
   if filtername == 'UU': filtername = 'U'
   if filtername == "UGRISM0160":
      command = "quzcif swift uvota - UGRISM teldef "+tim[0]+" "+tim[1]+" wheelpos.eq.160"
   elif  filtername == 'UGRISM0200':  
      command = "quzcif swift uvota - UGRISM teldef "+tim[0]+" "+tim[1]+" wheelpos.eq.200"
   elif filtername == 'VGRISM0955':
      command = "quzcif swift uvota - VGRISM teldef "+tim[0]+" "+tim[1]+" wheelpos.eq.955"
   elif filtername == 'VGRISM1000':
      command = "quzcif swift uvota - VGRISM teldef "+tim[0]+" "+tim[1]+" wheelpos.eq.1000"
   elif filtername == 'WHITE' or filtername == 'WH':
      command = "quzcif swift uvota - WHITE teldef "+tim[0]+" "+tim[1]+" -"
   elif (filtername == 'V') | (filtername == 'B') | (filtername == 'U'):
      command = "quzcif swift uvota - "+filtername+" teldef "+tim[0]+" "+tim[1]+" - "
   elif (filtername == 'W1') | (filtername == 'UVW1'):
      command = "quzcif swift uvota - UVW1 teldef "+tim[0]+" "+tim[1]+" -"
   elif (filtername == 'M2') | (filtername == 'UVM2'):
      command = "quzcif swift uvota - UVM2 teldef "+tim[0]+" "+tim[1]+" -"
   elif (filtername == 'W2') | (filtername == 'UVW2'):
      command = "quzcif swift uvota - UVW2 teldef "+tim[0]+" "+tim[1]+" -"

   try: 
      os.system(command + " > teldef.txt")   
      f1 = open('teldef.txt')
      line = f1.readline()
      f1.close()
      teldef = line.split()[0]
      UTELDEF = teldef.split('/')[-1]
      DTELDEF = teldef.split(UTELDEF)[0]
      if chatter > 0:
          print ("TELDEF found ",teldef)
          print ("split: ",UTELDEF,"  ; ",DTELDEF)
      f.write("EXTNAME = 'gu%09iE'\nUTELDEF = %s\n"%(int(tstart),UTELDEF ))
   except: 
      print ("ERROR: the correct teldef file was not found in the CALDB query")
      print ("CONTINUING WITH 20041120v105 teldef version")
      print ("   **** THIS CAN CAUSE PROBLEMS ***    " )
      print (" filtername = "+filtername)
      pass
      if filtername == 'UGRISM0160':
          f.write("EXTNAME = 'gu%09iE'\nUTELDEF = 'swugu0160_20041120v105.teldef'\n"%(int(tstart)))
          UTELDEF = 'swugu0160_20041120v105.teldef'
      elif filtername == 'UGRISM0200':
          f.write("EXTNAME = 'gu%09iE'\nUTELDEF = 'swugu0200_20041120v105.teldef'\n"%(int(tstart)))
          UTELDEF = 'swugu0200_20041120v105.teldef'
      elif filtername == 'VGRISM0955':
          f.write("EXTNAME = 'gu%09iE'\nUTELDEF = 'swugv0955_20041120v105.teldef'\n"%(int(tstart)))
          UTELDEF = 'swugv0955_20041120v105.teldef'
      elif filtername == 'VGRISM1000':
          f.write("EXTNAME = 'gu%09iE'\nUTELDEF = 'swugv1000_20041120v105.teldef'\n"%(int(tstart)))
          UTELDEF = 'swugv1000_20041120v105.teldef'
      elif filtername == 'WHITE' or filtername == 'WH':
          f.write("EXTNAME = 'gu%09iE'\nUTELDEF = 'swuwh20041120v104.teldef'\n"%(int(tstart)))
          UTELDEF = 'swuwh20041120v104.teldef'
      elif filtername == 'V':
         f.write("EXTNAME = 'gu%09iE'\nUTELDEF = 'swuvv20041120v104.teldef'\n"%(int(tstart)))
         UTELDEF = 'swuvv20041120v104.teldef'
      elif filtername == 'B':
          f.write("EXTNAME = 'gu%09iE'\nUTELDEF = 'swubb20041120v104.teldef'\n"%(int(tstart)))
          UTELDEF = 'swubb20041120v104.teldef'
      elif filtername == 'U':
          f.write("EXTNAME = 'gu%09iE'\nUTELDEF = 'swuuu20041120v104.teldef'\n"%(int(tstart)))
          UTELDEF = 'swuuu20041120v104.teldef'
      elif filtername == 'W1':
          f.write("EXTNAME = 'gu%09iE'\nUTELDEF = 'swuw120041120v101.teldef'\n"%(int(tstart)))
          UTELDEF = 'swuw120041120v101.teldef'
      elif filtername == 'M2':
          f.write("EXTNAME = 'gu%09iE'\nUTELDEF = 'swum220041120v104.teldef'\n"%(int(tstart)))
          UTELDEF = 'swum220041120v104.teldef'
      elif filtername == 'W2':
          f.write("EXTNAME = 'gu%09iE'\nUTELDEF = 'swuw220041120v104.teldef'\n"%(int(tstart)))
          UTELDEF = 'swuw220041120v104.teldef'
      DTELDEF = "%s/data/swift/uvota/bcf/teldef/"%(CALDB)

   # write the FITS header lines needed
   f.write("DTELDEF = '%s'\n"%(DTELDEF))
   f.write("CTYPE1  = 'RAWX    '\nCTYPE2  = 'RAWY    '\nCUNIT1  = 'pixel   '\nCUNIT2  = 'pixel   '\n")
   f.write("EXPID   =           %010i\nBLOCLOSS=                    0\n"%(tstart))
   f.write("STALLOSS=                    0\nTOSSLOSS=                    0\n")
   f.write("CRPIX1  =                    1\nCRPIX2  =                    1\n")
   f.write("CRVAL1  =                    0\nCRVAL2  =                    0\n")
   f.write("CDELT1  =                    1\nCDELT2  =                    1\n")
   f.write("WINDOWX0=                    0\nWINDOWY0=                    0\n")
   f.write("WINDOWDX=                 2048\nWINDOWDY=                 2048\n")
   f.write("BINX    =                    1\nBINY    =                    1\n")
   f.write("ASPCORR = 'NONE    '\nMOD8CORR=                    F\n")
   f.write("FLATCORR=                    F\nSWROUND = 1.000000000000000E-02\n")
   f.close() 
   return DTELDEF+UTELDEF
   
def process_grism_events(eventfile,horizonsfile,delta_t=30,phafiledir="../spectra",chatter=1):   
   ''' For processing a grism event file:
   
   outline:
   make work and spectra dirs under the event dir 
   (mkdir ../work ../spectra, so that auxil dir is at ../../auxil)
   
   - get event file name
   - determine a rootname from the event file name
   - read GTI times from event file
   - get time step (interactive?)
   
   for each time step:
      - make image: 
        make_img_slice(eventdir,eventfile,tstart,tstop,outfile="./timeslice.evt",chatter=0)
      - get tstart, tstop
      - get ra,dec (begin, end) 
      - check ra,dec does not vary too much or WARN
      
      - run uvotrawevtimg eventfile=timeslice.evt \
            attfile=../../auxil/sw00091596001pat.fits \
            outfile=../work/sw00012345001ugu_rw \
            teldeffile="%s/data/swift/uvota/bcf/teldef/swugu0160_20041120v105.teldef"%(CALDB)
      - fix the FITS header: 
         - make a copy into event dir: 
          e.g., cp ../image/sw00091596001ugu_rw.img .
         - fappend 'test_raw.img[0]' sw00091596001ugu_rw.img
         - fdelhdu sw00091596001ugu_rw.img+1
      # make patch.hdr file with missing keywords
      _make_hdr_patch(tstart)
      #mv select_153844-153874.raw select_153844-153874ugu_rw.img
      fthedit sw00091596001ugu_rw.img @patch.hdr
      run grismpipe  sw00091596001ugu  ../../auxil/sw00091596001pat.fits
      run uvotpy.uvotgrism.getSpec()
      copy+rename output pha files to pha dir
      rename xselect_driver.xco
      rename timeslice.evt
      rename image.raw, image.det, image.sky
         
      append new names  pha file to some phafiles.list 
   
   when done,    
   run uvotpy ; read phafiles.list; make sum image; make sum spectra (interactive) 
   
   '''
   import os
   import pyfits
   import numpy as np
   
   fileroot = eventfile.split('uguw1')[0]
   hdu = pyfits.open("../event/"+eventfile)
   gti = hdu['STDGTI']
   hdu.close()
   Ngti = len(gti)
   
   uvot_event._edit_horizons_file(horizonsfile)
   fRA, fDEC = get_target_position(horizonsfile+'.data')
   
   for kgti in range(Ngti):
      gti_tstart,gti_tstop = gti.data[kgti]
      times = range(gti_tstart,gti_tstop,delta_t)
      N = len(times)-1
      for i in range(N):
         ra_start = fRA(times[i]  )
         ra_stop  = fRA(times[i+1])
         dec_start = fDEC(times[i]  )
         dec_stop  = fDEC(times[i+1])
   
         d_ra = np.abs(ra_start-ra_stop) / np.abs(np.sin(dec_start)) 
         d_dec = np.abs(dec_start-dec_stop)
         dist = sqrt(d_ra**2+d_dec**2)
         if dist > 0.0003: print ("WARNING: timestep implies a step in RA,DEC > 1 arcsec:   ", dist,"  arcsec")
         
         command = "uvotrawevtimg eventfile=timeslice.evt attfile=../../auxil/"+fileroot+\
                   "pat.fits outfile=../work/"+fileroot+\
                   "ugu_rw.img teldeffile='CALDB'"
                   #"/sciencesoft/CALDB/data/swift/uvota/bcf/teldef/swugu0160_20041120v105.teldef"
         
         status = os.system(command)
         print (status)
         
         raw = fileroot+"ugu_rw.img"
         _make_hdr_patch(times[i])
         
def split_eventlist_simple(eventfile,times=[],dt=10.0, specfile='usnob1.spec',
   attfile=None, 
   catspecdir='/Volumes/users/Users/kuin/specfiles/', 
   cleanup=True,clobber=True,chatter=0):
   """
   Split image event file into snapshots, using times, find the aspect correction, 
   update the attitude file,  
   
   Parameters
   ----------
   eventfile : str
      path of the event file
   times : list
      list of [start,end] swift times (s) as float values
      if times not supplied, the times are taken from the 
      STDGTI extension of the event file, or from tstart and tstop
      note that in the last case there may be a VERY large number 
      of extensions.
   dt : float
      time step in seconds   
   specfile : str
      path of the spec file for the catalog server (e.g., usnob1.spec or GSC.spec)
   attfile : str
      path of the attitude file
   cleanup :  bool
      cleanup parameter value used in uvot/headas/ftools calls  
   chatter : int
      verbosity used in uvot/headas/ftools calls 
      
   Notes
   -----
   A need for this arises when the source is too bright for aperture photometry and 
   the readout streak needs to be used. 
   
   Prior to processing (after running coordinator) the event file should be 
   screened with :
      uvotscreen aoexpr="ANG_DIST < 100. && ELV > 10. && SAA==0 && SAFEHOLD==0" \
              evexpr="QUALITY.eq.0.or.QUALITY.eq.256"
   (use old version 1.18 uvotscreen) 
   This screening does not screen for bad attitude, which would remove too much data.
   
   The events in sky coordinates (x,y) have each been corrected for attitude during the 
   (standard) processing. 
   
   """
   
   import os
   import numpy as np
   from astropy.io import fits
   from astropy import wcs
   
   mode = "simple"
   
   if cleanup: 
      clean = 'yes' 
   else: 
      clean = 'no'
      
   evt = fits.open(eventfile)
   evt_hdr = evt['EVENTS'].header
   evt_data = evt['EVENTS'].data
   primary = evt['PRIMARY']
   tstart_all = float(primary.header['tstart'])
   tstop_all = float(primary.header['tstop'])
   filtername = evt_hdr['FILTER'].strip()
   detnam = evt_hdr['DETNAM'].strip()
   # need to check what the values actually are for 'detnam'
   if detnam == '0160' : 
      filtername='UGRISM0160'; filtervv = 'gu' 
   if detnam == '0200' : 
      filtername='UGRISM0200'; filtervv = 'gu'
   if detnam == '0955' : 
      filtername='VGRISM0955'; filtervv = 'gv' 
   if detnam == '1000' : 
      filtername='VGRISM1000'; filtervv = 'gv'
   if filtername.lower() == 'b': filtervv = 'bb'
   if filtername.lower() == 'v': filtervv = 'vv'
   if filtername.lower() == 'u': filtervv = 'uu'
   if filtername.lower() == 'w1': filtervv = 'w1'    
   if filtername.lower() == 'm2': filtervv = 'm2'    
   if filtername.lower() == 'w2': filtervv = 'w2'    
   if filtername.lower() == 'uvw1': filtervv = 'w1'    
   if filtername.lower() == 'uvm2': filtervv = 'm2'    
   if filtername.lower() == 'uvw2': filtervv = 'w2'    
   if filtername.lower() == 'white': filtervv = 'wh'   
       
   # assuming filename is close to convention do an autopsy 
   eventfilename = eventfile.split('/')[-1]
   eventdir = eventfile.split(eventfilename)[0]
   file1 = eventfilename.split('.')
   if (file1[1] == 'evt') & (file1[0][0:2] == 'sw'):
      obsid = file1[0][2:13]
      filtername = file1[0][14:16] 
   else:  # substitute some typical values
      obsid = '0099909001'
      filtername='zz'
   if attfile == None: attfile="../../auxil/sw"+obsid+"pat.fits"
   
   ntimes = len(times)
   if ntimes < 1: 
      print ("no time periods found: assuming whole list in one")
      if mode == 'simple' : 
         times = _make_times(evt,dt)
         ntimes = len(times)
         if ntimes < 1:
            print ("trying some times")
            dt = int((tstart-tstop)/dt)
            t1 = np.arange(tstart_all,tstop_all,dt)
            t2 = t1+dt
            ntimes = len(t1)
            for k in range(ntimes):
                times.append([t1[k],t2[k]])      
   else:
      for i in range(ntimes): 
         if not ((times[i][0] >= tstart_all) & (times[i][1] <= tstop_all) ):
             print ("for times[',i,'] the range is outside that of the event list")
                         
   nblocks = int(ntimes/20)+1
   for kb in range(nblocks):
         nstart = 20*kb
         deltan = np.min([20,ntimes-nstart])
         for i in range(nstart,nstart+deltan):         
            [tstart,tstop] = times[i] 
            output = file1[0]+'_'+str(tstart)+'-'+str(tstop)+'.img'
            make_img_slice(eventdir,eventfilename,tstart,tstop,outfile=output,
                type='image',chatter=chatter)

    

