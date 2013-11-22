# 
# Copyright N.P.M. Kuin 2013, All rights reserved
# 
# This software is licenced for use under a BSD style licence. 
# 
#
# Developed by N.P.M. Kuin (MSSL/UCL)
#
#

def readout_streak(obsid,
       target='target',
       radec=None,
       interactive=True,
       magfile='magnitudes.txt',
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
   ***magfile*** : path
      the path of the file to append a summary of the selected sources to.
      When choosing an absolute path, multiple observations can be 
      done this way.  
   ***radec*** : list , optional
      give the sky position ra,dec (J2000/ICRS) in units of 
      decimal degrees, e.g., [305.87791,+20.767536]
         
   output
   ======
   - command line output of all readout streaks
   - writes a file with the python dictionary
   - appends magnitude and error with time and obsid+extension number to 'magfile'   
      
   Notes
   =====
   Required: - download the swift uvot auxil, uvot/hk, and uvot/image/ data. 
             - unzip the *_rw.img files if they are compressed, prior to running the code.
	     - the c-program readout_streak which can be downloaded from 
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
   
   Bugs/desired upgrades:
    - Problem with small frames: some are still not positioned correctly.
    - want to use sky file header to convert ra,dec -> raw image position
      currently, converts to det position only
    - include the readout_streak code in the distribution (requires fitsio and
      wcstools.)  
   	
   '''
   import os
   import numpy as np
   try:
     from astropy.io import fits
   except:
     import pyfits as fits  
   # write header to magfile (will append existing file)
   if magfile != None: 
      magfh = open(magfile,'a')
      magfh.write("columns (Vega mag) errors swift-time, date\n")
      magfh.write("-uvw2- -uvm2- -uvw1- ---u-- ---b-- ---v-- e_w2  e_m2  e-w1"
      "  -e_u- -e_b- -e_v- --tstart-- --date-obs------ ---obsid+ext\n")   
   # run through all bands and locate the available raw image files 
   bands =['uwh','uw2','um2','uw1','uuu','ubb','uvv']
   rawfiles = []
   result = []
   for b in bands:
     filename = 'sw'+obsid+b+'_rw.img'
     print "looking for "+filename
     if os.access(filename, os.F_OK):
        rawfiles.append(filename)
     elif os.access(filename+'.gz', os.F_OK):
        if os.system('gunzip '+filename+'.gz'):
	   rawfiles.append(filename)
   if len(rawfiles) == 0:
      print " no valid raw files found in the current directory; unzipping any present; try to rerun."
      return
   print "raw files found : ",rawfiles
   # check the small mode image size is consistent with naxis; update header
   for rf in rawfiles:
     obses=[]
     print "examining "+rf
     hdu = fits.open(rf,'update')
     n_ext= len(hdu)
     for k in range(1,n_ext):
        print "examining HDU number :",k
        ft = hdu[k].header['framtime'] 
	ax1 = hdu[k].header['naxis1']
	binx = hdu[k].header['binx']
	tstart=hdu[k].header['tstart']
	print "TSTART = ",tstart
	#
	expected = int(8*256/binx/0.0110322*ft + 0.1)	   	
        if (np.abs(expected - ax1) > 5) & (ax1 == 2048/binx) : 
            # make sub image for the extension
	    print "converting image size - checking presence HK file"
	    if os.access('../hk/sw'+obsid+'uct.hk.gz', os.F_OK): 
	       os.system('gunzip '+'../hk/sw'+obsid+'uct.hk.gz')
	    if os.access('../hk/sw'+obsid+'uct.hk', os.F_OK):
	       hk = fits.open('../hk/sw'+obsid+'uct.hk')
# The window location is encoded in the hk/sw00*uct.hk 
# parameters DW_X0, DW_Y0, DW_XSIZ,DW_YSIZ parameters, but
# the image coordinates are DW_X0*16:(DW_X0+DW_XSIZ)*16 , 
# DW_Y0*16,(DW_Y0+DW_YSIZ)*16 though the first left columns
# are looking blank. 
# Is only needed if the NAXIS1,NAXIS2=2048,2048 in the raw image header.
               # 
	       xx = np.array(tstart - hk[1].data['expid'],dtype=int)
	       nnn= np.where( np.abs (xx) == np.min(np.abs(xx)))[0][0]
	       DW_X0 = hk[1].data['DW_X0'][nnn]
	       DW_Y0 = hk[1].data['DW_Y0'][nnn]
	       DW_XSIZ = hk[1].data['dw_xsiz'][nnn]
	       DW_YSIZ = hk[1].data['dw_ysiz'][nnn]
	       x0=DW_X0*16
	       x1=(DW_X0+DW_XSIZ)*16
	       y0=DW_Y0*16
	       y1=(DW_Y0+DW_YSIZ)*16 
	       print "updating image size to x:",x0,x1," y:",y0,y1
	       hdu[k].data = hdu[k].data[x0:x1,y0:y1]
	       hdu[k].header['DW_X0'] = DW_X0
	       hdu[k].header['DW_Y0'] = DW_Y0
	       hdu[k].header['DW_XSIZ'] = DW_XSIZ
	       hdu[k].header['DW_YSIZ'] = DW_YSIZ
	       hdu[k].header['WINDOWX0'] = x0
	       hdu[k].header['WINDOWY0'] = y0
	       hdu[k].header['WINDOWDX'] = DW_XSIZ*16
	       hdu[k].header['WINDOWDY'] = DW_YSIZ*16
               hk.close()	    
	    else:
	       print 'no HK data available to update the raw image header'
	       print 'readout streak data questionable'
     hdu.writeto(rf+".new",output_verify='fix',clobber=True)
     #hdu.close()
     # apply mod8 correction to  raw image
     md = rf.split('rw')
     b = md[0][-3:-1]
     bp = md
     sk = md
     md[1]='md.img'
     md_ = ""
     for snip in md: md_ += snip
     md = md_
     bp[1]='bp.img'
     bp_ = ""
     for snip in bp: bp_ += snip
     bp = bp_
     sk[1]='sk.img'
     sk_ = ""
     for snip in sk: sk_ += snip
     sk = sk_
     print rf
     print bp
     print md
     print sk
     if not os.access(md, os.F_OK):
        command = "uvotbadpix infile="+rf+".new"+" badpixlist=CALDB outfile="+\
        bp+" compress=YES clobber=yes history=yes chatter="+str(chatter)
        print command	
        os.system(command)	         
        command = "uvotmodmap infile="+rf+".new"+" badpixfile="+bp+" outfile="+md+\
        " mod8prod=NO mod8file=CALDB nsig=3 ncell=16 subimage=NO "+\
        " xmin=0 xmax=2047 ymin=0 ymax=2047 clobber=yes history=yes chatter="\
        +str(chatter)
        print command 
        os.system(command)	    
     # run Mat's readout_streak c program on the mod8 corrected file
     resfile = "results."+b+".1234.txt"
     command = 'readout_streak infile='+md+' > '+resfile
     print command
     os.system(command)
     # process the output from readout_streak 
     hdu = fits.open(md)
     datesobs = []
     tstart_ = []
     for m in range(1,len(hdu)): 
         datesobs.append(hdu[m].header['date-obs'])
	 tstart_.append( hdu[m].header['tstart'] )   
     # convert the output to an obs dict
     obses, details = _read_readout_streak_output(
         obses,
	 inp=resfile,
	 band=md.split('_')[0][-3:],
	 dateobs=datesobs,
	 tstart=tstart_,
	 infile=md,
	 )
     # run readout_streak_mag 
     for obs in obses:
        print "============== LSS and magnitudes for : "+obs["infile"]+"+",obs["extension"]
	if radec != None:
	    if type(radec) != list: 
	       rawxy = None
	       print "radec parameter is not a list!"
	    else:   
	       ra,dec=radec
	       ext = obs['extension']
	       status,rawx,rawy = _sky_to_raw(ra,dec,
	                                      sk,rf,ext,
					      chatter=chatter)
	       if status == 0:
	          rawxy = [rawx,rawy]
	       else:
	          rawxy = None   	  	    
	lss,img_coord, det_coord  = _lss_corr(obs, interactive=interactive,rawxy=rawxy) 
        data = _readout_streak_mag(obs, target=target,lss=lss,
	       subimg_coord=img_coord,det_coord=det_coord )
	obs.update( {'magnitudes': data})
	obs.update( {'img_coord':img_coord})       
	obs.update( {'det_coord':det_coord})       
	result.append(obs)

   # mag output : cycle through filters
   formatstr =["%5.3f -1    -1    -1    -1    -1        %4.3f -1    -1    -1    -1    -1    %10i %16s %10s+%i\n",
               "-1    %5.3f -1    -1    -1    -1        -1    %4.3f -1    -1    -1    -1    %10i %16s %10s+%i\n",
               "-1    -1    %5.3f -1    -1    -1        -1    -1    %4.3f -1    -1    -1    %10i %16s %10s+%i\n",
               "-1    -1    -1    %5.3f -1    -1        -1    -1    -1    %4.3f -1    -1    %10i %16s %10s+%i\n",
               "-1    -1    -1    -1    %5.3f -1        -1    -1    -1    -1    %4.3f -1    %10i %16s %10s+%i\n",
               "-1    -1    -1    -1    -1    %5.3f     -1    -1    -1    -1    -1    %4.3f %10i %16s %10s+%i\n",
	       ]
   filts = ['uvw2','uvm2','uvw1','u','b','v']
   for (fmt,fi) in zip(formatstr,filts):
       for obj in result:
           print "obj band=",obj['band']," searching band =",fi
           if obj['band'] == fi:
	       try:
	          x = obj['img_coord'][0]
		  streaks=[]
		  for s in  obj['streak_col_SN_CR_ERR']: streaks.append(s[0])
		  # first try to pick the brightest streak within 16 subpixels
		  # if that did not work, just use the closest one in distance
		  try:  
		     kandi = np.abs(streaks-x) < 16
		     kanmag = []
		     kanerr = []
		     for k in obj['magnitudes']:
		         kanmag.append( k[2] )
			 kanerr.append( k[3] )
                     kanmag = np.array(kanmag)[kandi]
		     kanerr = np.array(kanerr)[kandi]
		     if len(kanmag) == 0:  		 	 
		         k = np.where(np.abs((streaks-x)) == np.min(np.abs(streaks - x)))
		         k = k[0][0]
	                 mag = obj['magnitudes'][k][2]
	                 err = obj['magnitudes'][k][3]
                     else:
                         k = (kanmag == np.min(kanmag))	
			 mag = kanmag[k]
			 err = kanerr[k]	         			 
		  except:
		     k = np.where(np.abs((streaks-x)) == np.min(np.abs(streaks - x)))
		     k = k[0][0]
	             mag = obj['magnitudes'][k][2]
	             err = obj['magnitudes'][k][3]
		     pass
		  # overlimit = obj['streak_col_SN_CR_ERR'][k][0] when, set errors to .9999
	          datobs = obj['dateobs']
		  ext = obj['extension']
		  tsta = obj['tstart']
		  print mag,err,tsta, datobs,ext
		  print fmt
                  magfh.write(fmt%(mag,err,tsta,datobs[0:16],obsid,ext))
               except:
	          print "there seems to be a problem with:", obj
	          pass    		  
       
   magfh.close()
   #try:
   f = open('readout_streak.results.txt','w')
   f.write("%s"%(result))
   f.close()	   
   return result	
	

def _lss_corr(obs,interactive=True,maxcr=False,figno=20,
         rawxy=None,
	 target='target',
	 chatter=0):
   '''determine the LSS correction for the readout streak source '''
   import os
   import numpy as np
   try:
      from astropy.io import fits
   except:
      import pyfits as fits   
   from pylab import figure,imshow,ginput,axvspan,\
        axhspan,plot,autumn,title,clf
   file = obs['infile']
   ext = obs['extension']
   cols = obs['streak_col_SN_CR_ERR']
   kol = []
   countrates=[]
   for k in cols:
      kol.append(k[1]) # S/N
      countrates.append(k[2])
   kol = np.array(kol)   
   if len(kol) == 0:   
      print "zero array kol in _lss_corr ???!!!!"
      print "LSS correction cannot be determined."	 
      return 1.0, (0,0), (1100.5,1100.5)
   k_sn_max = np.where(np.max(kol) == kol) # maximum s/n column
   print "k S/N max=",k_sn_max[0][0]
   kol = []
   for k in cols:
      kol.append(k[0]) # column number relative to bottom rh corner subimage
   if len(kol) == 0:   
      print "LSS correction cannot be determined."	 
      return 1.0, (0,0), (1100.5,1100.5)
   im=fits.getdata(file, ext=ext)   
   hdr=fits.getheader(file, ext=ext)
   binx = hdr['binx']
   mn = im.mean()
   sig = im.std()
   #   plot
   figure(figno)
   clf()
   autumn()
   imshow(im,vmin=mn-0.2*sig,vmax=mn+2*sig)
   if rawxy != None:
	 rawx,rawy = rawxy
	 plot(rawx/binx,rawy/binx,'o',markersize=25,color='w',alpha=0.3)
   title(u"PUT CURSOR on your OBJECT",fontsize=16)
   if not maxcr:
       count = 0
       for k in kol:
           axvspan(k-6,k+6,0.01,0.99, facecolor='k',alpha=0.3-0.02*count)
           count += 1
   else:
       k = k_sn_max[0][0]    
       axvspan(k-10,k+10,0.01,0.99, facecolor='k',alpha=0.2)
   happy = False
   count = 0
   while not happy :  
      print "put the cursor on the location of your source"
      count += 1
      coord = ginput(timeout=0)
      print "selected:",coord
      if len(coord[0]) == 2: 
         print "window corner:", hdr['windowx0'],hdr['windowy0']
         xloc = coord[0][0]+hdr['windowx0']
         yloc = coord[0][1]+hdr['windowy0']
         print "on detector (full raw image)  should be :",xloc,yloc
         #plot(coord[0][0],coord[0][1],'+',markersize=14,color='k',lw=2) #can't see it
	 axhspan(coord[0][1]-6,coord[0][1]+6,0,1,facecolor='k',alpha=0.3)
	 if rawxy != None:
	    rawx,rawy = rawxy
	    plot(rawx/binx,rawy/binx,'o',markersize=25,color='w',alpha=0.3)
         ans = raw_input("happy (yes,no): ")
         if ans.upper()[0] == 'Y': happy = True
      else:
         print "no position found"	 
      if count > 10:
         print "Too many tries: aborting" 
         happy = True
   im = ''
   try:
      lss = 1.0
      band = obs['band']
      caldb = os.getenv('CALDB')
      command = "quzcif swift uvota - "+band.upper()+\
         " SKYFLAT "+\
         obs['dateobs'].split('T')[0]+"  "+\
	 obs['dateobs'].split('T')[1]+" - > lssfile.1234.tmp"
      print command	 
      os.system(command)
      f = open('lssfile.1234.tmp')
      lssfile = f.readline()
      f.close()
      f = fits.getdata(lssfile.split()[0],ext=int(lssfile.split()[1]))
      lss = f[ yloc,xloc]
      print "lss correction = ",lss,"  coords=",coord[0], (yloc+104,xloc+78)
      return lss, coord[0], (yloc+104,xloc+78) 
   except:
      print "LSS correction cannot be determined."	 
      return 1.0, (0,0), (1100.5,1100.5)
    
def _read_readout_streak_output(obses,inp='results.1234.txt',
     band=None,dateobs=None,tstart=None,infile=None):
   '''convert output from results.1234.txt to list of obs dict 
   and append them to obses list
   
   lower limits are not given ?
   ''' 
   import numpy as np
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
   ext_meta=[]
   ext_data=[]
   ext = 0
   for r in recs:
      if r[1:10] == "Extension": 
         n_ext += 1
	 ext=r.split(',')[0].split()[-1]
	 ext_meta.append(dict(
	   ext=r.split(',')[0].split()[-1],
	   exposure=r.split(',')[1].split()[1],
	   frametime=r.split(',')[2].split()[1],
	   ))
      if r[0:3] == 'Ext':
         if int(r[3:6]) != int(ext):
	    print r
	    print "check extension failed : ",int(r[3:6])," not equal to ",int(ext) 
         ext_data.append(dict(
	    ext = ext,
	    column = r.split(',')[0].split()[4],
	    SN = r.split(',')[1].split()[2],
	    cr = r.split(',')[2].split()[2],
	    err = r.split(',')[2].split()[4],
	    ))
   for m in range(n_ext): 
      streak_col_SN_CR_ERR=[]
      streak_id=[]
      goodstreak=[]	    
      extension=ext_meta[m]['ext']
      for mm in range(len(ext_data)):
	 try:
            if ext_data[mm]['ext'] == extension:
               print "debug ",m, mm, extension, ext_data[mm]
	       streak_col_SN_CR_ERR.append([
	          float(ext_data[mm]['column']),
	          float(ext_data[mm]['SN']),
	          float(ext_data[mm]['cr']),
	          float(ext_data[mm]['err']),    ])
	       streak_id.append(None)
	       goodstreak.append(False)
	 except: pass      
      obs=dict(
         band=bandname[band == bandits][0],
         dateobs=dateobs[m],
	 tstart=tstart[m],
         infile=infile,
         extension=int(extension),
         exposure =float(ext_meta[m]['exposure']),
         frametime=float(ext_meta[m]['frametime']),
         streak_col_SN_CR_ERR=streak_col_SN_CR_ERR,
         streak_id=streak_id,
         goodstreak=goodstreak,
         )
      obses.append(obs)	 
   return obses,(n_ext,ext_meta,ext_data)

def _readout_streak_mag(obs, target='target',lss=1.0,
     subimg_coord=None, det_coord=None):
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
   
   # table 1
   frametimes = [11.0329e-3, 5.417e-3, 3.600e-3] # available uvot CCD frametimes
   S          = [     9049.,    4369.,    2855.] # 
   max_cr     = [      0.30,     0.62,     0.95] # maximum coi-corrected CR 
   # table 2 
   zp = { # provides list of Vega-system zeropoints for  
   # full frame, large window, small window 
   # frame times 0.011, 0.078, 0.036
   'v':[8.00,8.79,9.25],
   'b':[9.22,10.01,10.47],
   'u':[8.45,9.24,9.70],
   'uvw1':[7.55,8.34,8.80],
   'uvm2':[6.96,7.75,8.21],
   'uvw2':[7.49,8.28,8.74],
   'white':[10.40,11.19,11.65]
   }
   t_MCP=2.36e-4
   overlimit = False
   band = obs['band'].lower()
   print "\n%s Readout Streak for %s in the %s filter. "%(target,obs['dateobs'], band)
   
   # correction for sensitivity loss
   date=obs['dateobs']
   if len(date) < 11:
       xseconds = datetime.datetime(int(date[:4]),int(date[5:7]),
         int(date[8:10]))- datetime.datetime(2005,1,1,0,0,0)
   else:
       xseconds = datetime.datetime(int(date[:4]),int(date[5:7]),
         int(date[8:10]),int(date[11:13]), int(date[14:16]),
         int(date[17:20]),0) - datetime.datetime(2005,1,1,0,0,0)
   xyear = xseconds.days/365.26
   print "sensitivity correction = %7.3f"%((1+0.01*xyear))

   # index for the proper frametime   
   k = np.where(abs(1-np.array(frametimes)/obs['frametime']) < 1e-2)[0][0]
   
   # if not target in 'streak_id' field, set first
   xx = np.array(obs['streak_id'],dtype=bool) 
   if xx.any():
     streak = obs['streak_col_SN_CR_ERR'][np.where(
               np.array(obs['streak_id']) == target)[0][0]]
     column,SN,rate,err = streak
     return [_readout_streak_mag_sub(k,S,rate,t_MCP,err,obs,xyear,
              lss,zp,band,max_cr,overlimit)]
   else:
     result = []
     for streak in obs['streak_col_SN_CR_ERR']:
         column,SN,rate,err = streak 
	 print "column,S/N,rate,err= ",streak
	 if subimg_coord != None:
	    print "distance column to target: ", column-subimg_coord[0]
	   
	 result.append(
	   _readout_streak_mag_sub(k,S,rate,t_MCP,err,obs,xyear,lss,zp,
	      band,max_cr,overlimit)
	   )
     return result	    

def _readout_streak_mag_sub(k,S,rate,t_MCP,err,obs,xyear,
      lss,zp,band,max_cr,overlimit): 
   import numpy as np
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
       print "observed CR =%10.5f,  MCP-loss corrected CR =%10.5f"%(rate,r_coi)
   # correct for LSS
   
   r_coi = r_coi * (1+0.01*xyear)/lss
   err = err * (1+0.01*xyear)/lss	
   r_coi_u = r_coi_u * (1+0.01*xyear)/lss
   r_coi_d = r_coi_d * (1+0.01*xyear)/lss
   
   if r_coi > max_cr[k]:  overlimit=True
   mag = zp[band][k] - 2.5*np.log10(r_coi)
   mag_u = zp[band][k] - 2.5*np.log10(r_coi_u)
   mag_d = zp[band][k] - 2.5*np.log10(r_coi_d)
   if overlimit: 
      print "count rate is over the recommended limit"
      print "%s magnitude <%7.3f (+%7.3f -%7.3f)\n"%(band,mag,mag_u-mag,mag-mag_d)
      return overlimit,band,mag,.9999,.99999
   else:   
      print "%s magnitude = %7.3f +%7.3f -%7.3f\n"%(band,mag,mag_u-mag,mag-mag_d)
   return overlimit,band,mag,mag_u-mag,mag-mag_d

def _sky_to_raw(ra,dec,skyfile,rawfile,ext,chatter=1):
   '''
   use the header of the skyfile to determine the position on the raw image
   
   input parameters
   ================
   ra,dec : float
     position on the sky in decimal degrees
   skyfile,rawfile : path  
     filename of the sky file and raw file 
   ext : int
     extension number of the raw file being processed  
   
   returns
   =======
   status : int
     0 successful conversion
     non-0: did not work
   x,y : int
      position on the raw image of the source
   
   Notes
   =====
   The conversion from sky position to detector coordinate is 
   straightforward, and can be done by the Astropy WCS routines.
   
   To convert from detector coordinate to raw coordinate, the 
   uvot distortion map must be inverted. For this we need to 
   use the Swift Ftools 
   
   '''
   import os
   import numpy as np
   from astropy import wcs
   from astropy import coordinates
   try:
      from astropy.io import fits
   except ImportError:
      import pyfits   
   status = 0
   # check the files are present
   if not os.access(skyfile,os.F_OK):
      if chatter > 0: print "sky position to raw img position: skyfile not found" 
      status = 1
      return status, -1,-1
   # check for raw file was done in calling program
   # read fits sky header
   sky = fits.open(skyfile)
   raw = fits.open(rawfile)
   # match exposure id
   expid = (sky[ext].header['extname'] == raw[ext].header['extname'] )
   if not expid:  
      if chatter > 0: print "sky file extension does not match raw file"
      status = 1
      return status, -1,-1
         
   # convert ra,dec -> xdet,ydet
   W = wcs.WCS(header=sky[ext].header,key=' ',relax=True)
   Wd = wcs.WCS(header=sky[ext].header,key='D',relax=True)
   radec = np.array([[ra,dec],])
   phys = W.wcs_world2pix(radec,0)
   det = Wd.wcs_pix2world(phys,0) # now this is position on det in mm
   scl = 0.009075
   detx,dety = det[0][0],det[0][1] # mm
   x = detx/scl+1100.5 -104  # img pixels undistorted
   y = dety/scl+1100.5 -78   # img pixels undistorted
   
   # convert xdet,ydet -> x,y (raw)
   #os.system("cat "+str(detx)+","+str(dety)+" > detxy.txt")
   #command = HEADAS+'/bin/uvotapplywcs infile=detxy.txt'\
   #  ' outfile=raw.txt wcsfile=\"'+rawfile+'['+ext+']\" operation=WORLD_TO_PIX'
   return status,x,y
   

####################### end readout streak
