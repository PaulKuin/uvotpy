# -*- coding: iso-8859-15 -*-

__version__ = "1.5.5"

# version 1.0 9 Nov 2009
# version 1.1 21 Jan 2010 : adjust range for V grism
# version 1.2 1 Nov 2011 : update the LSF in the RMF file (slow)
# version 1.3 dec 2011: rewrite of writeOutput(), added rate2flux(), write_response_arf(), 
#             SpecResp(), write_rmf_file(), and helper writeOutput_()
# version 1.4 Sep 2012: added support for coi loss correction, anchor dependence of response curve XYSpecResp
# version 1.5 Mar 5, 2013 
# version 1.5.1 April 11, 2013 fixed argument call XYspecresp
# version 1.5.2 October 29, 2013 added 'f' or 'g' to output filenames indicates 
#               lenticular filters or uvotgraspcorr used for anchor
#               pyfits -> fits
# version 1.5.3 January 1, 2014 added aperture correction to background 
#               update of write spectrum: add code for a dictionary to be passed
#               add code for xspec output with coi-correction done (background-corrected, 
#               coi-corrected rates, and errors)
#               changed error computation, aperture corrected, and assume background error negligible
# version 1.5.4 February 27, 2014 updated effective area files, updated write_rmf_file 
# version 1.5.5 May 1, 2014, update of coi-computation 

try:
  from uvotpy import uvotplot,uvotmisc,uvotwcs,rationalfit,mpfit,uvotio
except:
  pass  
import uvotgetspec

interactive=uvotgetspec.interactive

def get_uvot_observation(coordinate=None,name=None,obsid=None,chatter=0):
   '''the purpose is to grab the uvot data from the archive  '''
   # Under development
   return

def rate2flux(wave, rate, wheelpos, bkgrate=None, pixno=None, sig1coef=[3.2], sigma1_limits=[2.6,4.0],\
    arf1=None, arf2=None, effarea1=None, effarea2=None,
    spectralorder=1, trackwidth = 1.0, anker=None, test=None, respfunc=False,\
    swifttime=None, option=1, fudgespec=1.322, frametime=0.0110329, debug=False, chatter=1):
   ''' 
   Convert net count rate to flux 
   
   WARNING: dependent on the parameters passed, the old CALDB (<=2012) 
       OR the new flux calibration will be used.  Since 10SEP2012 the 
       coi-factor is included in the calculation of the flux and the 
       effective area. A coi-correction is still made when using the old 
       CALDB which will be inconsistent to that calculated in the 
       writeSpectrum() which makes the output file.
   
   many of the parameters are needed to calculate the coi-factor
       
   
   Parameters
   ----------
      
   wave	: float ndarray
   	wavelength in A
	
   rate	: float ndarray
   	net count rate/bin in spectrum, aperture corrected
	
   wheelpos : int
   	filter wheel position
      
   bkgrate : ndarray
   	background rate for spectrum (corrected to 2.5 sigma half-width)
	
   pixno : ndarray
   	pixel coordinate (zero = anchor; + increasing wavelengths)
	
   sig1coef : list
   	polynomial coefficient for spectrum halfwidth
	
   sigma1_limits : list
   	sigma will be truncated at the limits
	
   arf1, arf2 : path or "CALDB", optional 
			
   effarea1, effarea2 : FITS HDU[, interpolating function]
        result from a previous call to 	readFluxCalFile() for first or second order
		
   spectralorder : int
   	the spectral order of the spectrum, usually =1
	
   trackwidth : float
   	width of the spectral extraction used in units of sigma 
	
   anker : list
   	anchor detector coordinate positions (pix) as a 2-element numpy array
	
   frametime : float
   	the frame time for the image is required for the coi-correction
	
   swifttime : int
        swift time of observation in seconds for calculating the sensitivity loss 	
	
   option : int
   	when =2 sets the method for the coi-correction [development]
	
   fudgespec : float
   	possible correction for the coi-correction  [development]
	
   debug : bool
   	 for development
	
   chatter : int
   	verbosity (0..5)
	
   respfunc : bool
        return the response function (used by writeSpectrum())
              
   Returns
   -------
   flux, coi-corrected 
   
   Notes
   -----
   2013-05-05 NPMKuin - adding support for new flux calibration files; new kwarg 
   2014-02-28 fixed. applying fnorm now to get specrespfunc, pass earlier effective area
   2014-04-30 NPMK changed coi_func parameters (option=1,fudgespec=1.322,frametime,coi_length=29)	      
   '''
   
   import numpy as np
   from uvotgetspec import coi_func
   from scipy import interpolate
   
   __version__ = '130505'
   
   h_planck = 6.626e-27  # erg/s
   lightspeed = 2.9979e10  # cm/sec
   h_c_ang = h_planck * lightspeed * 1e8 # ergs-Angstrom
   hnu = h_c_ang/(wave)      
   
   # assume uv grism
   if pixno == None:      
      dis = np.arange(len(wave)) - 370
      if spectralorder == 2: dis -= 260
   else:
      dis = pixno   
   coef = np.polyfit(dis,wave,4)	 
   binwidth = np.polyval(coef,dis+0.5) - np.polyval(coef,dis-0.5)      # width of each bin in A (= scale A/pix)
      
   if ((spectralorder == 1) & (effarea1 == None)) | ((spectralorder == 2) & (effarea2 == None)) :
      if (test != None) & (anker != None):
	 # attempt to use the new spectral response (if returns None then not available)
	 z =  readFluxCalFile(wheelpos,anchor=anker,spectralorder=spectralorder,chatter=chatter)
	 if (z == None):  
      	    specrespfunc = XYSpecResp(wheelpos=wheelpos, \
	       spectralorder=spectralorder, anker=anker, test=test,chatter=chatter)
	 else:
	    # ancher given,  default
	    hdu,fnorm = z
	    w = list(0.5*(hdu.data['WAVE_MIN']+hdu.data['WAVE_MAX']))     
	    r = list(( hdu.data['SPECRESP'] )*fnorm(w))
	    w.reverse() ; r.reverse()
	    specrespfunc = interpolate.interp1d( w, r,  bounds_error=False, fill_value=np.NaN)
	    
      elif ((anker == None) | (arf1 != None) | (arf2 != None)) : 
         hdu =  readFluxCalFile(wheelpos,spectralorder=spectralorder,chatter=chatter)
	 if (hdu == None): 
             specrespfunc = SpecResp(wheelpos, spectralorder, arf1=arf1,arf2=arf2,)
	 else:
	    w = list(0.5*(hdu.data['WAVE_MIN']+hdu.data['WAVE_MAX']))
	    r = list(hdu.data['SPECRESP'] )
	    w.reverse() ; r.reverse()
	    specrespfunc = interpolate.interp1d( w, r, bounds_error=False, fill_value=np.NaN)
	    
      else:
	 # attempt to use the new spectral response (if returns None then not available)
	 if (effarea1 != None) & (spectralorder == 1):
	     z = effarea1
	 elif (effarea2 != None) & (spectralorder == 2):   
	     z = effarea2
	 else:
             z =  readFluxCalFile(wheelpos,anchor=anker,spectralorder=spectralorder,chatter=chatter)
	 if (z == None): 
	    print "uvotio.rate2flux warning: fall back to XYSpecResp call " 
      	    specrespfunc = XYSpecResp(wheelpos=wheelpos, spectralorder=spectralorder, anker=anker,chatter=chatter)
	 else:
	    # ancher given,  default
	    hdu,fnorm = z
	    w = list(0.5*(hdu.data['WAVE_MIN']+hdu.data['WAVE_MAX']) )    
	    r = list(hdu.data['SPECRESP'])
	    w.reverse() 
	    r.reverse()
	    r = np.array(r) * fnorm(w)
	    specrespfunc = interpolate.interp1d( w, r, bounds_error=False, fill_value=np.NaN )
	    
   elif ((spectralorder == 1) & (effarea1 != None)): 
       if len(effarea1) == 2:	    
	    hdu,fnorm = effarea1
 	    w = 0.5*(hdu.data['WAVE_MIN']+hdu.data['WAVE_MAX'])    
	    r = hdu.data['SPECRESP']
	    ii = range(len(w)-1,-1,-1) 
	    r = r * fnorm(w)
	    specrespfunc = interpolate.interp1d( w[ii], r[ii], bounds_error=False, fill_value=np.NaN )
       else:  
            hdu  = effarea1
 	    w = 0.5*(hdu.data['WAVE_MIN']+hdu.data['WAVE_MAX'])    
	    r = hdu.data['SPECRESP']
	    ii = range(len(w)-1,-1,-1) 
	    specrespfunc = interpolate.interp1d( w[ii], r[ii], bounds_error=False, fill_value=np.NaN )

   elif ((spectralorder == 2) & (effarea2 != None)): 	#this is under development only
       print "second order Effective area is under development - not for science use"    
       if len(effarea2) == 2:	    
	    hdu,fnorm = effarea2
 	    w = 0.5*(hdu.data['WAVE_MIN']+hdu.data['WAVE_MAX'])    
	    r = hdu.data['SPECRESP']
	    ii = range(len(w)-1,-1,-1) 
	    r = r * fnorm(w)
	    specrespfunc = interpolate.interp1d( w[ii], r[ii], bounds_error=False, fill_value=np.NaN )
       else:  
            hdu  = effarea2
 	    w = 0.5*(hdu.data['WAVE_MIN']+hdu.data['WAVE_MAX'])    
	    r = hdu.data['SPECRESP']
	    ii = range(len(w)-1,-1,-1) 
	    specrespfunc = interpolate.interp1d( w[ii], r[ii], bounds_error=False, fill_value=np.NaN )
      	   
   else: return None
   
   if respfunc: return specrespfunc  # feed for writeSpectrum() 	 

   if swifttime != None: 
      senscorr = sensitivityCorrection(swifttime)
      print "Sensitivity correction factor for degradation over time = ", senscorr
   else: 
      senscorr = 1.0 
      print "NO Sensitivity correction applied"  
   
   if ((bkgrate != None) & (pixno != None)):
      if chatter > 0: print "performing the COI correction "
          # do the coi-correction
	  
      fcoibg = coi_func(pixno,wave,rate,bkgrate,sig1coef=sig1coef,option=option,
        fudgespec=fudgespec,coi_length=29,frametime=frametime, background=True,
        sigma1_limits=sigma1_limits, trackwidth = trackwidth,
        debug=debug,chatter=chatter)
      fcoi = coi_func(pixno,wave,rate,bkgrate,sig1coef=sig1coef,option=option,
        fudgespec=fudgespec,coi_length=29,frametime=frametime, background=False,
        sigma1_limits=sigma1_limits, trackwidth = trackwidth,
        debug=debug,chatter=chatter)
      netrate = rate*fcoi(wave)
      flux = hnu*netrate*senscorr/specrespfunc(wave)/binwidth   # [erg/s/cm2/angstrom]
   else:   
      if chatter > 0: print "WARNING rate2flux: Flux calculated without a COI-correction"
      # no coi correction
      flux = hnu*rate*senscorr/specrespfunc(wave)/binwidth   # [erg/s/cm2/angstrom]
        
   return flux
   
   
def sensitivityCorrection(swifttime,sens_rate=0.01):
   '''
   give the sensitivity correction factor to divide the rate/flux by 
   
   Parameters
   ----------
   swifttime : float
      time of observation since 2005-01-01 00:00:00  in seconds, usually TSTART 
      
   sens_rate : float
      the yearly percentage loss in sensitivity
         
   Notes
   -----
   A 1%/year decay rate since 2005-01-01 has been assumed and 
   the length of the mean Gregorian year was used
   '''
   sens_corr = 1.0/(1.0 - sens_rate*(swifttime-126230400.000)/31556952.0 )  
   return sens_corr
   

def angstrom2kev(lamb,unit='angstrom'):
   """
   conversion of units 

   Parameter
   ---------
   lamb : array
     Input photon wavelength in angstrom 

   Returns
   -------  
   The photon energy in keV.
   """
   return  12.3984191/lamb


def kev2angstrom(E,unit='keV'):
   """
   conversion of units
   
   Parameter
   ---------
   E : array
      Input photon energy in keV.

   Returns
   -------
   The photon wavelength in angstroms 
   """
   return  12.3984191/E
   

def fileinfo(filestub,ext,lfilt1=None, directory='./',chatter=0, wheelpos=None, twait=40.0):
   '''finds files for spectrum, matching attitude and lenticular images 
      uncompresses gzipped files if found
      
      Parameters
      ----------
      filestub : str
        the base of the file name, with the Swift project convention,
	consisting of "sw" + the OBSID, i.e., "sw00032301001" 
      ext : int
        the number of the extension of the grism file to match
	
      kwargs : dict
      
       - **lfilt1** : str, optional
       
        name of first lenticular filter used. Must be one of 'uvw2',
	'uvm2','uvw1','u','b','v','wh'	
	
       - **directory** : path, str
       
        path for directory. This is the directory with the grism file.
	
       - **chatter** : int
       
        verbosity
				
       - **twait** : float
       
        The maximum time allowed between begin and end of matched 
	exposures of grism-lenticular filter, for each match.
	
       - **wheelpos** : imt
        
	If given, use to discriminate between UV and Visual grisms.	
	
      Returns
      -------
      specfile, attfile: str
        filename of spectrum, 
        the attitude file name.
      
      lfilt1, lfilt2 : str
       lenticular filter file name before the grism exposure or None, 
       and the file name of the lenticular filter following the grism 
       
      lfilt1_ext,lfilt2_ext : int
       extension number for each lenticular filter matching exposure 

   '''
   import os
   try:
      from astropy.io import fits 
   except:
      import pyfits as fits
   from numpy import array
   
   ext_names =array(['uw2','um2','uw1','uuu','ubb','uvv','uwh'])
   lfiltnames=array(['uvw2','uvm2','uvw1','u','b','v','wh'])
   
   vvgrism = True
   uvgrism = True
   if wheelpos != None:
      if wheelpos < 500: vvgrism = False
      else: uvgrism = False
   
   if (not directory.endswith('/')) : 
      directory += '/' 
   auxildir = directory+'../../auxil/'
      
   # test if u or v grism file and set variable 
   specfile = ' *filename not yet initialised (directory wrong?)* '
   if uvgrism & os.access(directory+filestub+'ugu_dt.img',os.F_OK):
        specfile =  directory+filestub+'ugu_dt.img'
	if chatter > 1: print 'reading ',specfile
   elif uvgrism & os.access(directory+filestub+'ugu_dt.img.gz',os.F_OK):
        specfile =  directory+filestub+'ugu_dt.img'
	os.system( 'gunzip '+specfile+'.gz' )
	if chatter > 1: print 'reading ',specfile
   elif vvgrism & os.access(directory+filestub+'ugv_dt.img',os.F_OK):     
        specfile =  directory+filestub+'ugv_dt.img'
	if chatter > 1: print 'reading ',specfile
   elif vvgrism & os.access(directory+filestub+'ugv_dt.img.gz',os.F_OK):     
        specfile =  directory+filestub+'ugv_dt.img'
	os.system( 'gunzip '+specfile+'.gz' )
	if chatter > 1: print 'reading ',specfile
   else:
        raise IOError("FILEINFO: cannot find %s: DET file not found - pls check directory/file provided  is correct" % specfile )
		
   #    attitude file:
   if os.access(directory+filestub+'pat.fits',os.F_OK):
        attfile =  directory+filestub+'pat.fits'
	if chatter > 1: print 'found att file ',attfile
   elif os.access(directory+filestub+'pat.fits.gz',os.F_OK):
        attfile =  directory+filestub+'pat.fits'
	os.system( 'gunzip '+attfile+'.gz' )
	if chatter > 1: print 'found att file ',attfile
   elif os.access(directory+filestub+'uat.fits',os.F_OK):     
        attfile =  directory+filestub+'uat.fits'
	if chatter > 1: print 'found att file ',attfile
   elif os.access(directory+filestub+'uat.fits.gz',os.F_OK):     
        attfile =  directory+filestub+'uat.fits'
	os.system( 'gunzip '+attfile+'.gz' )
	if chatter > 1: print 'found att file ',attfile
   elif os.access(directory+filestub+'sat.fits',os.F_OK):     
        attfile =  directory+filestub+'sat.fits'
	if chatter > 1: print 'found att file ',attfile
   elif os.access(directory+filestub+'sat.fits.gz',os.F_OK):     
        attfile =  directory+filestub+'sat.fits'
	os.system( 'gunzip '+attfile+'.gz' )
	if chatter > 1: print 'found att file ',attfile
   elif os.access(auxildir+filestub+'pat.fits',os.F_OK):
        attfile =  auxildir+filestub+'pat.fits'
	if chatter > 1: print 'found att file ',attfile
   elif os.access(auxildir+filestub+'pat.fits.gz',os.F_OK):
        attfile =  auxildir+filestub+'pat.fits'
	os.system( 'gunzip '+attfile+'.gz' )
	if chatter > 1: print 'found att file ',attfile
   elif os.access(auxildir+filestub+'uat.fits',os.F_OK):     
        attfile =  auxildir+filestub+'uat.fits'
	if chatter > 1: print 'found att file ',attfile
   elif os.access(auxildir+filestub+'uat.fits.gz',os.F_OK):     
        attfile =  auxildir+filestub+'uat.fits'
	os.system( 'gunzip '+attfile+'.gz' )
	if chatter > 1: print 'found att file ',attfile
   elif os.access(auxildir+filestub+'sat.fits',os.F_OK):     
        attfile =  auxildir+filestub+'sat.fits'
	if chatter > 1: print 'found att file ',attfile
   elif os.access(auxildir+filestub+'sat.fits.gz',os.F_OK):     
        attfile =  auxildir+filestub+'sat.fits'
	os.system( 'gunzip '+attfile+'.gz' )
	if chatter > 1: print 'found att file ',attfile
   #    filter file(s)
   lfilt1,lfilt2 = None,None 
   lfilt1_ext = None; lfilt2_ext=None
   hdu = fits.open(specfile)
   if len(hdu)-1 < ext: 
      raise IOError("Input error: extension not found in Grism file.")
   hdr = hdu[int(ext)].header   
   hdu.close()
   #hdr = fits.getheader(specfile,int(ext))
   tstart = hdr['TSTART']
   tstop  = hdr['TSTOP'] 
   if chatter > 1: 
      print 'grism times : %s - %s '%(tstart,tstop)
   lfile=None
   #  
   for k in range(len(ext_names)):
      ftyp = ext_names[k]
      lfiltyp = lfiltnames[k]
      if chatter > 1: print "testting for "+directory+filestub+ftyp+'_sk.img'
      if os.access(directory+filestub+ftyp+'_sk.img',os.F_OK):
        lfile =  directory+filestub+ftyp+'_sk.img'
	if chatter > 1: 
	   print 'found lenticular sky file ',lfile 
      elif os.access(directory+filestub+ftyp+'_sk.img.gz',os.F_OK):
        lfile =  directory+filestub+ftyp+'_sk.img' 
	os.system( 'gunzip '+lfile+'.gz' )
	if chatter > 1: print 'found lenticular sky file ',lfile
      if lfile != None:	
         # check if it has an extension before or after specfile[ext] 
         xxx = fits.open(lfile)
         for i in range(1,len(xxx)):
            t1 = xxx[i].header['TSTART']
	    t2 = xxx[i].header['TSTOP']
	    if abs(t2-tstart) < twait:
	       lfilt1 = lfiltyp
	       lfilt1_ext = i
	       if chatter > 0: print "lenticular file observation preceeding grism observation"
	    if abs(t1-tstop) < twait:
	       lfilt2 = lfiltyp
	       lfilt2_ext = i   
	       if chatter > 1: print "lenticular file observation after grism observation"
	 lfile = None 
	 xxx.close()
	      
   # wrapup in case there is only one, but in lfilt2.
   if ((lfilt1 == None) & (lfilt2 != None)): 
      if chatter > 2: print "putting only filter info in filter 1"
      lfilt1 = lfilt2
      lfilt2 = None
      lfilt1_ext = lfilt2_ext
      lfilt2_ext = None
   #   
   return specfile, lfilt1, lfilt1_ext, lfilt2, lfilt2_ext, attfile	


def writeEffAreaFile (wheelpos,spectralorder,wave,specresp,specresp_err=None,
       anker=None,dxy_anker=None,fileversion='999',todir="./",clobber=False):
   ''' create an ARF file 
   
   Parameters
   ---------- 
   wheelpos : int, {160,200,955,1000}

   spectralorder: int, {1,2}
   
   wave: ndarray 
       wavelengths in Angstrom
       
   specresp: ndarray
       effective area (EA) in cm^2 for each wave
       
   specresp_err: ndarray 
       1-sigma EA error (random + systematic) 
       
   anker: list, ndarray[2]
       2-element array with position in det coordinates of EA
       
   dxy_anker: list,ndarray[2]
       EA determined for box [anker[0]+/-dxy_anker[0], anker[1]+/-dxy_anker[1]]
       
   fileversion: str
       version for this EA (spectral response) file.
       
   todir: path
       directory to place the file into    

   Returns
   -------
   the new effective area file with file name something like:
      'swugu0160_ax1100ay1100_dx150dy150_o1_20041120v001.arf'

   Notes
   ----- 
   - Modified 15-SEP-2012 by Paul Kuin.	   
   With only wheelpos, spectralorder, wave, specresp input, the output file conforms to 
   the HEASARC approved response file. The additional keywords and error column have not 
   been approved as of 15 September 2012. 
    
   - Modified 13 Feb 2013 by Paul Kuin
   Added futher keyword COIAWARE to discriminate between the old and new effective areas and 
   changed comments after keywords to be more descriptive. 
   
   - Modified 5 March 2013 by Paul Kuin
   header edited 

   - Renamed 28 Dec 2013
   first extension assumed 1-spaced wavelengths. Relaxed to allow variable wavelengths.
   
   '''
   try:
     from astropy.io import fits
   except:  
     import pyfits as fits
   import datetime
   import numpy as np
   from scipy import interpolate
   import os
   
   version = '20131228'
   a = now = datetime.date.today()
   datestring = a.isoformat()[0:4]+a.isoformat()[5:7]+a.isoformat()[8:10]
   rnu = now.day*1.2+now.month*0.99+now.year*0.3
   
   #  file name elements:  
   of1 = '_20041120v'+fileversion+'.arf'
   if spectralorder == 1: of0 = '_o1'
   if spectralorder == 2: of0 = '_o2'
   
   of2 = ''
   if (anker != None):
      of2 = '_ax'+str(anker[0])+'ay'+str(anker[1])
      
   if (dxy_anker != None):
      of2 += '_dx'+str(dxy_anker[0])+'dy'+str(dxy_anker[1])  
       
   if wheelpos == 160:
      if spectralorder == 1:  
         EXTNAME='SPECRESPUGRISM160'
         outfile = todir+'swugu0160'+of2+of0+of1
      elif spectralorder == 2:
         EXTNAME = 'SPECRESP0160GRISM2NDORDER'
         outfile = todir+'swugu0160'+of2+of0+of1
      filtername = 'UGRISM'  
      
   elif wheelpos == 200:
      if spectralorder == 1:  
         EXTNAME='SPECRESPUGRISM200'
         outfile = todir+'swugu0200'+of2+of0+of1
      elif spectralorder == 2:
         EXTNAME = 'SPECRESP0200GRISM2NDORDER'  
         outfile = todir+'swugu0200'+of2+of0+of1
      filtername = 'UGRISM'  
      
   elif wheelpos == 955: 
      if spectralorder == 1:  
         EXTNAME='SPECRESPVGRISM955'
         outfile = todir+'swugv0955'+of2+of0+of1
      elif spectralorder == 2:
         EXTNAME = 'SPECRESP0955GRISM2NDORDER'  
         outfile = todir+'swugv0955'+of2+of0+of1
      filtername = 'VGRISM'  
      
   elif wheelpos == 1000:
      if spectralorder == 1:  
         EXTNAME='SPECRESPVGRISM1000'
         outfile = todir+'swugv1000'+of2+of0+of1
      elif spectralorder == 2:
         EXTNAME = 'SPECRESP1000GRISM2NDORDER'  
         outfile = todir+'swugv1000'+of2+of0+of1
      filtername = 'VGRISM'        
      
   specrespfunc = interpolate.interp1d(wave, specresp, kind='linear', bounds_error=False )    
   specresp_errfunc = interpolate.interp1d(wave, specresp_err, kind='linear', bounds_error=False )    
   
   hdu = fits.PrimaryHDU()
   hdulist=fits.HDUList([hdu])
   hdulist[0].header.update('TELESCOP','SWIFT   ','Telescope (mission) name')                       
   hdulist[0].header.update('INSTRUME','UVOTA   ','Instrument Name')                                
   hdulist[0].header.update('COMMENT','Grism Effective area')                                
   
   #  first extension SPECRESP
   
   binwidth = 1.0 # scalar
   ax = np.arange(int(min(wave)),int(max(wave)),binwidth)
   NW = len(ax)
   wavmin = (ax-0.5*binwidth)
   wavmax = (ax+0.5*binwidth)
   binwidth = wavmax-wavmin # array
   midwave = 0.5*(wavmax+wavmin)
   energy_lo = angstrom2kev(wavmax)
   energy_hi = angstrom2kev(wavmin)
   elow  = energy_lo.min()
   ehigh = energy_hi.max()
   specresp = specrespfunc( midwave )
   specresp_err = specresp_errfunc( midwave )
   ix = range(len(ax))
   ix.reverse()
   col11 = fits.Column(name='ENERG_LO',format='E',array=energy_lo[ix],unit='KeV')
   col12 = fits.Column(name='ENERG_HI',format='E',array=energy_hi[ix],unit='KeV')
   col13 = fits.Column(name='WAVE_MIN',format='E',array=wavmin[ix],unit='angstrom')
   col14 = fits.Column(name='WAVE_MAX',format='E',array=wavmax[ix],unit='angstrom')
   col15 = fits.Column(name='SPECRESP',format='E',array=specresp[ix],unit='cm**2' )
   if specresp_err == None:
      cols1 = fits.ColDefs([col11,col12,col13,col14,col15])
   else:
      col16 = fits.Column(name='SPRS_ERR',format='E',array=specresp_err[ix],unit='cm**2' )
      cols1 = fits.ColDefs([col11,col12,col13,col14,col15,col16])
      
   tbhdu1 = fits.new_table(cols1)
   tbhdu1.header.update('EXTNAME',EXTNAME,'Name of this binary table extension')
   tbhdu1.header.update('TELESCOP','Swift','Telescope (mission) name')
   tbhdu1.header.update('INSTRUME','UVOTA','Instrument name')
   tbhdu1.header.update('FILTER',filtername)
   tbhdu1.header.update('ORIGIN','MSSL/UCL','source of FITS file')
   tbhdu1.header.update('CREATOR','uvotio.py','uvotpy python library')
   tbhdu1.header.update('COMMENT','uvotpy sources at www.github.com/PaulKuin/uvotpy')
   tbhdu1.header.update('VERSION',fileversion)
   tbhdu1.header.update('FILENAME',outfile,'file NAME')
   tbhdu1.header.update('HDUCLASS','OGIP','format conforms to OGIP standard')
   tbhdu1.header.update('HDUCLAS1','RESPONSE','RESPONSE DATA')
   tbhdu1.header.update('HDUCLAS2','SPECRESP','type of calibration data')   
   tbhdu1.header.update('CCLS0001','CPF','dataset is a calibration product file')
   tbhdu1.header.update('CCNM0001','SPECRESP','Type of calibration data')
   tbhdu1.header.update('CDES0001',filtername+' SPECTRAL RESPONSE AT ANCHOR POSITION','Description')
   tbhdu1.header.update('CDTP0001','DATA','Calibration file contains data')
   tbhdu1.header.update('CVSD0001','2004-11-20','UTC date when calibration should first be used')
   tbhdu1.header.update('CVST0001','00:00:00','UTC time when calibration should first be used')
   tbhdu1.header.update('CBD10001','FILTER('+filtername+')','Parameter boundary')
   tbhdu1.header.update('CBD20001','ENERG('+str(elow)+'-'+str(ehigh)+')keV','spectral range')
   tbhdu1.header.update('CBD30001','RADIUS(0-10)pixel','Parameter boundary')
   tbhdu1.header.update('CBD40001','THETA(0-17)arcmin','Parameter boundary')
   tbhdu1.header.update('CBD50001','WHEELPOS('+str(wheelpos)+')','Filter/Mode Selection')
   tbhdu1.header.update('CBD60001','ORDER('+str(spectralorder)+')','spectral order')  
   if (anker != None) & (dxy_anker != None):
      tbhdu1.header.update('CBD70001','ANCHOR('+str(anker[0])+','+str(anker[1])+')','anchor in pix (1100.5,1100.5)pix=(0,0)mm')  
      tbhdu1.header.update('CBD80001','ANCHOR_RANGE('+str(dxy_anker[0])+','+str(dxy_anker[1])+')','calibrared range dx,dy around anchor')  
      tbhdu1.header.update('CBD90001','COIAWARE('+'T'+')','pile-up effect taken out')  
      tbhdu1.header.update('COIVERS','1','ad-hoc solution')  
   
   tbhdu1.header.update('TTYPE1','ENERG_LO','[keV] Lower boundary of energy bin')
   tbhdu1.header.update('TTYPE2','ENERG_HI','[keV] Upper boundary of energy bin')
   tbhdu1.header.update('TTYPE5','SPECRESP','[cm**2] Effective Area')
   tbhdu1.header.update('COMMENT','created '+datestring)
   tbhdu1.header.update('COMMENT','uvotpy.writeEffAreaFile() version='+version)
   if specresp_err != None:
      tbhdu1.header.update('TTYPE6','SPRS_ERR','[cm**2] 1-sigma error effective area')
   hdulist.append(tbhdu1)
   hdulist.writeto(outfile,clobber=clobber)


def XYSpecResp(wheelpos=None,spectralorder=1,anker=[1129,1022], test=None, chatter=0):
   ''' the spectral response based on the position of the anchor of the spectrum.
   Depends on the grism mode via 'wheelpos' and the spectral order. 
   
   Parameters
   ----------
   wheelpos : int
   
   kwargs : dict
    - spectralorder : int
       order of the spectrum
       
    - anker : list
       position in detector coordinates (pixels)
   
    - test : any
       if not None then get the response at the boresight
   
   Returns
   -------
   An interpolating function for the spectral response
   based on the position (Xank,Yank) of the anchor of the spectrum.
   Depends on the grism mode via 'wheelpos' and the spectral order.
   
   Notes
   -----
   Will be superseded by `readFluxCalFile`
   '''
   import os
   
   if spectralorder == 1:
      print 'DEPRECATION NOTICE 2013-04-25: This method will be superseded by readFluxCalFile'
      print '        - available calibration files are those for the default position ony '

   Xank=anker[0]
   Yank=anker[1]
   
   # first get a list of the available calibration files, then 
   #    select the best one based on the nearest position.  
   
   if test != None:
      from scipy import interpolate
      # get boresight anchor 
      if wheelpos == 160:
	       bsx, bsy = uvotgetspec.boresight("uc160")
      elif wheelpos == 200:   
	       bsx, bsy = uvotgetspec.boresight("ug200")
      elif wheepos == 955:   
	       bsx, bsy = uvotgetspec.boresight("vc955")
      elif wheelpos == 1000:   
	       bsx, bsy = uvotgetspec.boresight("vg1000")
      # grab the spectral response for the center   
      sr_bs_func = XYSpecResp(wheelpos=wheelpos, spectralorder=spectralorder, anker = [bsx,bsy],  chatter=chatter)
      # get the normalised flux at the anchor position
      Z = _specresp (wheelpos, spectralorder, arf1 = None, arf2 = None, chatter=0)
      wmean, xresp = Z[:2]
      fnorm = None
      
   if spectralorder == 2:
      print "WARNING XYSpecResp: anchor dependence second order response has not yet been implemented"
      return SpecResp(wheelpos, spectralorder,)
      
   calfiles = os.getenv('UVOTPY')
   if calfiles == '': 
      print "please define environment variable UVOTPY before proceeding"
      raise
   else: 
      calfiles += "/calfiles/"
   status = os.system("ls -1 "+calfiles+" > tmp.1")
   if status != 0:
      print "FAIL: ls -1 "+calfiles+" > tmp.1"
   f = open("tmp.1")
   clist = f.readlines()
   f.close()      
   status = os.system("rm tmp.1") 
   
   if len(clist) == 0: 
      print "WARNING XYSpecResp: calfiles directory seems empty"
   
   if wheelpos == 160: 
      arf1 = 'swugu0160_1_20041120v999.arf'#'swugu0160_ax1130ay1030_dx70dy70_o1_20041120v001.arf'
      arf2 = None
   if wheelpos == 200: 
      arf1 = 'swugu0200_1_20041120v999.arf'#'swugu0200_ax1000ay1080_dx70dy70_o1_20041120v001.arf'
      arf2 = None
   if wheelpos == 955: 
      arf1 = 'swugv0955_1_20041120v999.arf'
      arf2 = None
   if wheelpos == 1000:
      arf1 = 'swugv1000_1_20041120v999.arf'
      arf2 = None
      
   if arf1 != None: 
      cl = arf1.split("_") 
      cl1 = arf1       
      _axy = [int(cl[1].split('ay')[0].split('ax')[1]),int(cl[1].split('ay')[1]) ]
      _dxy = cl[2]
      _ver = cl[4].split('.')[0].split('v') # [date,version]
      _dist = (Xank-_axy[0])**2 + (Yank-_axy[1])**2
   else:
      _axy = [1100,1100]
      _dxy = 'dx150dy150'
      _ver = ['20041120', '001']
      _dist = (Xank-_axy[0])**2 + (Yank-_axy[1])**2
   if chatter > 2: print "initial: _axy = %s\n_dxy = %s\n_ver = %s\n_dist = %7.2f\n" %(_axy, _dxy, _ver, _dist)
      
   for cl in clist:
       cl1 = cl
       cl = cl.split("\n")[0] 
       if chatter > 2:  print 'processing: ',cl
       #print 'wheelpos  : ',cl[5:9]
       try:	  
          if int(cl[5:9]) == wheelpos:
             cl = cl.split("_")
	     if (cl[3] == 'o'+str(spectralorder)):
                #print "spectral order OK: ", cl[3]
	        axy = [int(cl[1].split('ay')[0].split('ax')[1]),int(cl[1].split('ay')[1]) ]
	        dxy = cl[2]
	        ver = cl[4].split('.')[0].split('v') # [date,version]
                dist = (Xank-axy[0])**2 + (Yank-axy[1])**2
		if chatter > 2: print "order=%i\naxy = %s\ndxy = %s\nver = %s\ndist = %7.2f\n_dist = %7.2f\n" %(spectralorder,axy, dxy, ver, dist,_dist)
	        if ((spectralorder == 1) & (dist < _dist)) : 
		   arf1 = cl1
		   cl1 = arf1
		   print "1_____using "+arf1
	        if ((spectralorder == 2) & (dist < _dist)) : 
		   arf2 = cl1
		   cl1 = arf2
		   print "2_____using "+arf2
	        if ((spectralorder == 1) & (dist == _dist) & (ver[1] > _ver[1])) : 
		   arf1 = cl1
		   cl1 = arf1
		   print "3_____using "+arf1
	        if ((spectralorder == 2) & (dist == _dist) & (ver[1] > _ver[1])) : 
		   arf2 = cl1
		   cl1 = arf2
		   print "4_____using "+arf2
       except:
	  pass 	
	  	
   return SpecResp(wheelpos,spectralorder,arf1=arf1,arf2=arf2)

def _specresp (wheelpos, spectralorder, arf1 = None, arf2 = None, chatter=1):
   ''' Read the spectral response file [or a placeholder] and 
    
    Parameters
    ----------
    wheelpos : int, {160,200,955,1000}
    
    spectralorder : int, {1,2}
	   
    kwargs : dict
    -------------
     optional input
      
     - **arf1** : str, optional
     
       directs read of filename from $UVOTPY/calfiles/
       if None, program reads from $UVOTPY/calfiles
       if 'CALDB' read from $CALDB
     
     - **arf2** : str, optional
     
       regardless of input will read from $UVOTPY/calfiles/
       
    Returns
    -------
    An interpolating function for the spectral response
    as a function of wavelength (A)
    
    Notes
    -----
    Will be superseded by `readFluxCalFile` 
        '''
   import os
   try:
      from astropy.io import fits
   except:
      import pyfits as fits
   import numpy as np
   from scipy import interpolate
      
   caldb = os.getenv("CALDB")
   uvotpy = os.getenv("UVOTPY")
   
   if spectralorder == 1: 
      if wheelpos == 200:
         if (arf1 == None): 
            arfdbfile = 'swugu200_1_20041120v999.arf'
            EXTNAME='SPECRESPUGRISM200'
	 elif arf1 == 'CALDB':   
            arfdbfile = 'swugu0200_20041120v101.arf'
            EXTNAME='SPECRESPUGRISM200'
	 else: 
	    arfdbfile = arf1   
            EXTNAME='SPECRESPUGRISM200'
      elif wheelpos == 160:
         if (arf1 == None): 
            arfdbfile = 'swugu0160_1_20041120v999.arf'
            EXTNAME='SPECRESPUGRISM160'
	 elif (arf1 == 'CALDB'):   
            arfdbfile = 'swugu0160_20041120v101.arf'
            EXTNAME='SPECRESPUGRISM160'
	 else: 
	    arfdbfile = arf1   
            EXTNAME='SPECRESPUGRISM160'
      elif wheelpos == 1000:
         if (arf1 == None): 
            arfdbfile = 'swugv1000_1_20041120v999.arf'
            EXTNAME='SPECRESPVGRISM1000'
	 elif arf1 == 'CALDB':   
            arfdbfile = 'swugv1000_20041120v101.arf'
            EXTNAME='SPECRESPVGRISM1000'
	 else: 
	    arfdbfile = arf1   
            EXTNAME='SPECRESPVGRISM1000'
      elif wheelpos == 955: 
         if (arf1 == None): 
            arfdbfile = 'swugv0955_1_20041120v999.arf'
            EXTNAME='SPECRESPVGRISM955'
	 elif arf1 == 'CALDB':   
            arfdbfile = 'swugv0955_20041120v101.arf'
            EXTNAME='SPECRESPVGRISM955'
	 else: 
	    arfdbfile = arf1   
            EXTNAME='SPECRESPVGRISM955'
      else:   
         print "FATAL: exposure header does not have filterwheel position encoded"
         return 
      
   if (spectralorder == 2):
      if wheelpos == 160:
         EXTNAME = 'SPECRESP0160GRISM2NDORDER'  
         arfdbfile = 'swugu0160_2_20041120v999.arf'
      elif wheelpos == 200:
         EXTNAME = 'SPECRESP0200GRISM2NDORDER'  
         arfdbfile = 'swugu0200_2_20041120v999.arf'
      elif wheelpos == 955: 
         EXTNAME = 'SPECRESP0955GRISM2NDORDER'  
         arfdbfile = 'swugv0955_2_20041120v999.arf'
      elif wheelpos == 1000: 
         EXTNAME = 'SPECRESP1000GRISM2NDORDER'  
         arfdbfile = 'swugv1000_2_20041120v999.arf'
      else:   
         print "FATAL: exposure header does not have filterwheel position encoded", wheelpos
         return 
	 
      if arf2 != None: arfdbfile = arf2 
       	 
   #
   #  get spectral response [cm**2] from the ARF file 
   #   
   if (spectralorder == 2) | ((spectralorder == 1) & (arf1 != 'CALDB')):
      if chatter > 0:   
         print "opening spectral response ARF file: "+uvotpy+'/calfiles/'+arfdbfile
      f = fits.open(uvotpy+'/calfiles/'+arfdbfile)
   else:
      #print "specResp: arf1 | arf2 parameter input = :"+arf1+'  | '+arf2
      dirstub = '/data/swift/uvota/cpf/arf/'
      if chatter > 0:
         print "opening spectral response ARF file: "+caldb+dirstub+arfdbfile
      f = fits.open(caldb+dirstub+arfdbfile)
      if chatter > 0:
         print '********************************************************************'
         print '*** WARNING: EA ~10% low when no coi-correction is included ********'
         print '*** WARNING: This means your flux might be too high by ~10%  ********'
         print '********************************************************************'
   
   if chatter > 0:
      print "Read in "+str(spectralorder)+" order ARF file"    
      print f.info()   
   
   pext = f[0]
   fext = f[1]
   tab = fext.data
   elo = tab.field('energ_lo')
   ehi = tab.field('energ_hi')
   wmin= tab.field('wave_min')
   wmax= tab.field('wave_max')
   xresp = tab.field('specresp')  # response in cm2 per pixel 
   wmean = 0.5*(wmin+wmax)
   q = np.isfinite(wmean) & np.isfinite(xresp)
   wmean = wmean[q]
   xresp = xresp[q]
   if wmean[0] > wmean[-1]: 
      wmean = list(wmean)
      wmean.reverse()
      wmean=np.array(wmean)
      xresp = list(xresp)
      xresp.reverse()
      xresp = np.array(xresp)
   f.close()   
   return wmean, xresp   

def SpecResp (wheelpos, spectralorder, arf1 = None, arf2 = None):
   """
   Returns spectral response function 
   
   Parameters
   ----------
   wheelpos : int, {160,200,955,1000}
   spectralorder : int, {1,2}
	   
   kwargs : dict
   -------------
    optional input
     
    - **arf1** : str, optional
    
      directs read of filename from `$UVOTPY/calfiles/`
      if None, program reads from `$UVOTPY/calfiles`
      if `'CALDB'` read from `$CALDB`
      
    - **arf2** : str, optional
    
      regardless of input will read from `$UVOTPY/calfiles/`
       
   Returns
   -------
   An interpolating function for the spectral response
   as a function of wavelength (A)
    
   Notes
   -----
   Use `readFluxCalFile()` in case the new calibration file is present
   """
   from scipy import interpolate
   Z = _specresp (wheelpos, spectralorder, arf1 = None, arf2 = None)
   wmean, xresp = Z[:2]
   specrespfunc = interpolate.interp1d(wmean, xresp, kind='linear', bounds_error=False ) 
   return specrespfunc

def readFluxCalFile(wheelpos,anchor=None,option="default",spectralorder=1,arf=None,chatter=0):
   """Read the new flux calibration file, or return None.
   
   Parameters
   ----------
   wheelpos : int, required
      the position of the filterwheel 
   
   kwargs: dict
    - **anchor** : list, optional
      coordinate of the anchor
      
    - **option** : str
      option for output selection: 
        option=="default" + anchor==None: old flux calibration
        option=="default" + anchor : nearest flux calibration + model extrapolation
	option=="nearest" : return nearest flux calibration
	option=="model" : model 
	
    - **spectralorder** : int
        spectral order (1, or 2)
	
    - **arf**: path	
        fully qualified path to a selected response file
	

   Returns
   -------
   None if not (yet) supported
   option == 'model' returns the (astropy/pyfits) fits HDU (header+data) of the model 
   option == 'nearest'
      returns the fits HDU of the nearest calfile
   option == 'default' and anchor == None:
      returns the fits HDU of the nearest calfile 
   option == 'default' and anchor position given (in detector coordinates) 
      returns the fits HDU and an 
      interpolating function fnorm(wave in A) for the flux correction
   
   Notes
   -----	
                 
   2013-05-05 NPMKuin
   """
   try:  
     from astropy import fits
   except:
     import pyfits as fits
   import os 
   import numpy as np
   from scipy import interpolate

   grismname = "UGRISM"
   if wheelpos > 500: grismname  = "VGRISM"

   check_extension = False
   # here the "latest" version of the calibration files has been hardcoded    
   # latest update:   
   if spectralorder == 1: 
          if wheelpos == 200:          
             calfile = 'swugu0200_20041120v103.arf'
	     extname = "SPECRESPUGRISM200"
	     model   = "ZEMAXMODEL_200"
          elif wheelpos == 160:
             calfile = 'swugu0160_20041120v102.arf'
	     extname = "SPECRESPUGRISM160"
	     model   = "ZEMAXMODEL_160"
          elif wheelpos == 955: 
             calfile = 'swugv0955_20041120v102.arf'
	     extname = "SPECRESPVGRISM0955"
	     model   = "ZEMAXMODEL_955"
          elif wheelpos == 1000: 
             calfile = 'swugv1000_20041120v102.arf'
	     extname = "SPECRESPVGRISM1000"
	     model   = "ZEMAXMODEL_1000"
          else:   
             print "FATAL: invalid filterwheel position encoded"
             return None
   else: 	 
	     print "spectral order not 1 - no effective area available"
             return None
	     
   if chatter > 1:
      print "uvotio.readFluxCalFile attempt to read effective area file: "
      
   if arf != None:
      if arf.upper() == "CALDB":
   # try to get the file from the CALDB
         os.getenv("CALDB")
         command="quzcif swift uvota - "+grismname+\
          " SPECRESP now now  wheelpos.eq."+str(wheelpos)+" > quzcif.out"
         f = open("quzcif.out")
         records = f.readlines()
         f.close()
         os.system("rm -f quzcif.out")
         arf, extens = records[0].split()  
         arf = CALDB + "/data/swift/uvota/cpf/arf/"+arf     
         hdu = fits.open(arf)
       
      else:
      # path to arf is supplied
      # the format must give the full path (starting with "/" plus FITS extension
      # if no extension was supplied and there is only one, assume that's it.
      # check version==2, using presence of CBD70001 keyword and see if spectral order is right
         try:  # get extension from path 
            if len(arf.split("+") ) == 2: 
	       file, extens = arf.split("+")
	    elif len(arf.split("[") ) == 2:
	       file = arf.split("[")[0]
	       extens = arf.split("[")[1].split("]")[0] 
	    else:
	       check_extension = True
	    arf = file
         except: 
            raise IOError("The supplied effective area file name "+arf+" cannot be understood.")	           
       
         hdu = fits.open(arf)
         if check_extension:  # old version file 
            if hdu[1].header['CBD60001'].split("(")[1].split(")")[0] != spectralorder: 
               raise IOError("The supplied effective area file is not correct spectral order.")
            if ("CBD70001" not in hdu[extens].header) :  # old version
	       print "Using the oldest version of the effective area. \n"+\
	            "Flux, COI correction will be wrong."
               return hdu[extname]
   
   else:    # argument arf = None      
       uvotpy = os.getenv("UVOTPY")  
       arf = uvotpy+"/calfiles/"+calfile
       
       try:    
          hdu = fits.open(arf)
       except:
          print "UVOTPY environment variable not set or calfiles directory entries missing" 
          pass      
          return None  
      
   print "using flux calibration file: ",arf
   hdu.info()
   
   if (option == "default") | (option == "nearest"):
      if anchor == None:  # assume centre of detector
         anchor = [1000,1000]
      else:
         if (option == "default"): modelhdu = hdu[model]
	 if wheelpos < 500:
            n2 = 16
	 else: n2 = 12   
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
	 print "Nearest effective area is "+names[k]+"  - selected"
         if (option == "nearest"): 
	    return cal
	 try:  
 	    if chatter > 4: 
	       print "ReadFluxCalFile: calanchor ", calanchors[k] 
	       print "ReadFluxCalFile:    anchor ", anchor
	       print "ReadFluxCalFile:   extname ", modelhdu.header['extname'] 
	    modelcalflux = getZmxFlux (calanchors[k][0],calanchors[k][1],modelhdu)
	    modelobsflux = getZmxFlux (anchor[0],anchor[1],modelhdu)
	    q = np.isfinite(modelcalflux) & np.isfinite(modelobsflux) 
	    w = 10.0*modelhdu.data['WAVE']
	    if chatter > 4: 
	      print "ReadFluxCalFile:    check:  "
	      print "ReadFluxCalFile:    w.shape ",w.shape
	      print "ReadFluxCalFile:       =784*",n2," ?"
	    w = w.reshape(n2,784)[q,0]
	    fn = modelobsflux[q]/modelcalflux[q]
	    w1 = 1650.0
	    f1 = 1.0 # (fn[1]-fn[0])/(w[1]-w[0])*(w1-w[0]) + fn[0]
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
            fnorm = interpolate.interp1d(x,y,)	    
	    return cal, fnorm
	 except RuntimeError:
	     pass
	     print "WARNING: Failure to use the model for inter/extrapolation of the calibrated locations."
	     print "         Using Nearest Eaafective Area File for the Flux calibration."
	     fnorm = interpolate.interp1d([1600,7000],[1.,1.],)
	     return cal, fnorm 
   elif option == "model":
       return hdu[model]
   else:
       raise RuntimeError( "invalid option passed to readFluxCalFile") 

        
def getZmxFlux(x,y,model,ip=1):
   '''Interpolate model to get normalized flux. 
   
   parameters
   ----------
   x, y : float
     anchor coordinate x,y to find an interpolated solution to the model
     
   model : fits structure 
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
   if not ((type(model) != 'astropy.io.fits.hdu.table.BinTableHDU') | \
          (type(model) != 'pyfits.hdu.table.BinTableHDU') ):
	  raise IOError("getZmxFlux model parameter is not a proper FITS HDU bintable type")
	  
   n3     = 28*28
   n2     = model.header['NAXIS2']/n3
   if not ((n2 == 12) | (n2 == 16)):
      raise IOError("getZmxFlux: size of array in MODEL not correct; perhaps file corrupt?") 
   
   zmxwav = model.data['WAVE']
   xp     = model.data['XPIX']
   yp     = model.data['YPIX']
   zmxflux = model.data['FLUX']
   
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
	 print "getZmxFlux warning: at wavelength=",zmxwav[j2]," not sufficient valid points found"
            
      fx = xp[j2,q]
      fy = yp[j2,q]     
      ff = zmxflux[j2,q]
	    
      try:
         tck = interpolate.bisplrep(fx,fy,ff,xb=dminx,xe=dmax,yb=dminy,ye=dmax,kx=ip,ky=ip,)
	 flux[j2]  = interpolate.bisplev(x, y, tck)
      except:
         raise RuntimeError ("getZmxFlux ERROR in interpolation") 
	    
   return flux          	       
	       

def uvotify (spectrum, fileout=None, disp=None, wheelpos=160, lsffile=None, clean=True, chatter=1, clobber=False):
   '''
   Fold a high resolution input spectrum through the uvot spectral 
   response to "uvotify" the spectrum.
  
   Parameters
   ----------
   spectrum : path, str
      file name ASCII file, two columns wave(A), flux
	    
   kwargs : dict
   -------------	    
   
    - **fileout** : path, str, optional
    
      if given, the new spectrum will be written to an  output file (ascii)
	 
    - **wheelpos** : int
    
      the filterwheel position for selecting typical dispersion

    - **disp** : ndarray, optional
    
      dispersion coefficients (will be approximated if not given)
      
    - **lsffile** : path, str, optional
    
      the file with LSF data. If not given the file in $UVOTPY/calfiles
      will be used.   	  
	  		
    - **clean** : bool
    
      if True remove invalid data       

    - **chatter** : int
    
      verbosity
      
    - **clobber** : bool
    
      if True overwrite exisiting file   

   Returns
   -------
   (status, wave, flux) : 
      - status = 0 success  
      - wavelength and flux arrays convolved with the 
        uvot spectral response.

   - 2012-02-09 NPMK start development
   ''' 
   try:
      from astropy.io import fits
   except:   
      import pyfits as fits
   import numpy as np
   from uvotmisc import rdTab
   import os
   from scipy.ndimage import convolve
   import uvotio
   import uvotgetspec
   from scipy import interpolate
   import datetime
   
   version = '120209'
   status = 0
   now = datetime.date.today()
   datestring = now.isoformat()[0:4]+now.isoformat()[5:7]+now.isoformat()[8:10]
      
   if disp == None:   
      # typical dispersion forst order (use for whole detector):
      if wheelpos == 160:
         disp = np.array([4.1973e-10,-1.3010e-6,1.4366e-3,3.2537,2607.6])
      elif wheelpos == 200:
         disp = np.array([4.1973e-10,-1.3010e-6,1.4366e-3,3.2537,2607.6]) # placeholder
      elif wheelpos == 955:
         disp = np.array([4.1973e-10,-1.3010e-6,1.4366e-3,3.2537,2607.6]) # placeholder
      elif wheelpos == 1000:
         disp = np.array([4.1973e-10,-1.3010e-6,1.4366e-3,3.2537,2607.6]) # placeholder
      else:
         print "error in wheelpos argument" 
         status = 1	 
   
   tempfile = 'ab9804573234isfkjldsf.tmp'
   status = os.system('file '+spectrum+' > '+tempfile )
   if status == 0:
      f = open(tempfile)
      line = f.readline()
      f.close()
      os.system('rm '+tempfile)
      filetype = (line.split(':')[1]).split()[0]
      
      if chatter > 0: print "spectrum file type = "+filetype
      
      if filetype == 'ASCII':
         try:
           spect = rdTab(spectrum)
	   wave = spect[:,0]
	   flux = spect[:,1]
	   q = np.isfinite(flux) & np.isfinite(wave)
	   wav1 = wave[q]
	   flx1 = flux[q]
	   if len(wav1) < 1:
	      status = 1
	      print "number of valid wavelengths is zero"
	 except:
	   status = 1
	   print "failure reading the spectrum with routine rdTab. Make sure that the format is right."
      elif filetype == 'FITS':
        status = 1 
	print "filetype not supported"
        # future support for fits single column spectral format
	# and wave/flux columns in table SPECTRUM
      else:
         status = 1
	 print "filetype not supported"
   else:
      print "error reading file type "	 	 

   instrument_fwhm = 2.7/0.54 # in pix units
   gg = uvotgetspec.singlegaussian(np.arange(-12,12),1.0,0.,instrument_fwhm)
   gg = gg/gg.sum().flatten()  # normalised gaussian 
   
   if status == 0:         	     
      
      NN = len(wav1)  # number good points in the spectrum
      if NN < 3:
         print "uvotify: not enough valid data points. "
         return
      
      if lsffile == None:
         UVOTPY = os.getenv('UVOTPY')
         if UVOTPY == '': 
            print 'The UVOTPY environment variable has not been set'
         lsffile =   UVOTPY+'/calfiles/zemaxlsf.fit'	 

      lsffile = fits.open( lsffile )  
      tlsf = lsffile[1].data
      lsfchan = tlsf.field('channel')[0:15]   # energy value 
      lsfwav = uvotio.kev2angstrom(lsfchan)   # wavelength 
      epix    = tlsf.field('epix')[0,:]       # offset in half pixels
      lsfdata = tlsf.field('lsf')[:15,:]      # every half pixel a value
      lsffile.close()
         
         # define the LSF(wave) 
         # convolve flux(wave) * LSF(wave) function  

      flux1 = np.zeros(NN)	 
      for k in range(NN):   
         #  find index e in lsfchan and interpolate lsf
         w = wave[k]
         j = lsfwav.searchsorted(w)
         if j == 0: 
            lsf = lsfdata[0,:].flatten()
         elif ((j > 0) & (j < 15) ):
            e1 = lsfwav[j-1]
 	    e2 = lsfwav[j]
 	    frac = (w-e1)/(e2-e1)
            lsf1 = lsfdata[j-1,:]
	    lsf2 = lsfdata[j,:]
	    lsf = ((1-frac) * lsf1 + frac * lsf2).flatten()	 	 
         else:
            # j = 15
	    lsf = lsfdata[14,:].flatten()

         # convolution lsf with instrument_fwhm
         lsf_con = convolve(lsf,gg.copy(),)
      
         # assign wave to lsf_func relative to w at index k   
         # rescale lsfcon from half-pixels to channels 

         d   = np.arange(-79,79)*0.5 + (uvotgetspec.pix_from_wave(disp, w))[0]
         wave1 = np.polyval(disp,d)  
	 # wave1 is an increasing function - if not, the interpolating function fails.
         lsf_func = interpolate.interp1d(wave1, lsf_con,bounds_error=False,fill_value=0.0)
	 
	 loli = k-39
	 upli = k+39
         if loli < 0: loli = 0
	 if upli > NN: upli = NN
	 norm = np.asarray( lsf_func(wav1[loli:upli])  ).sum()
         flux1[k] = (flx1[loli:upli] * lsf_func(wav1[loli:upli]) / norm ).sum() 
	 
      if clean: 
         flux = flux1
	 wave = wav1
      else:	 	 
         flux[where(q)] = flux1
         flux[where(not q)] = np.nan	 
	    
      if fileout == None:
         return status, wave, flux
      else: 
         f = open(fileout,'w')
	 q = np.isfinite(wave) & np.isfinite(flux)
	 wave = wave[q]
	 flux = flux[q]
	 for i in range(len(wave)): f.write("%8.2f  %13.5e\n" % (wave[i],flux[i]) )
	 f.close()
   else:	 
      return status	 	 	    
	    

def writeSpectrum(ra,dec,filestub,ext, Y, fileoutstub=None, 
    arf1=None, arf2=None, fit_second=True, write_rmffile=True,
    used_lenticular=True, fileversion=1,
    history=None, chatter=1, clobber=False ) :
    
   '''Write a standard UVOT output file - Curved extraction only, not optimal extraction.
  
  Parameters
  ----------
  ra,dec : float, float
    position in decimal degrees
  filestub : str
    "sw" + obsid
  ext : int
    extension number 
  Y : tuple
    compound variable with spectral data from uvotgetspec            
  
  kwargs : dict
  -------------
  
   - **fileoutstub** : str, optional
   
     stub for the output file name
  
   - **arf1** : str, optional
     
     if 'CALDB', use the caldb effective area file 
   
   - **arf2** : str, optional
     
     if 'CALDB' use the caldb effective area file 
   
   - **fit_second** : bool
     
     if `True` tried extracting the second order          
   
   - **write_rmffile** : bool
   
     write RMF output if True (slow)
   
   - **history** : list
     
     list of processing history messages to write to header
   
   - **chatter** : int
   
   - **clobber** : bool
     overwrite files   

  Returns
  -------
  Writes the output file only. 
 
  Notes
  -----
  
  **Output file composition**
  
  For details, see the output file format description.
  
       Main header: 
          wheelpos, filter, orders, author
   
       For fileversion=1:
         The first extension is named  SPECTRUM (future: 'FIRST_ORDER_PHA_SPECTRUM') and 
         contains the standard input for XSPEC, 
	  - channel number, 
	  - measured uncorrected counts per channel, total counts not background corrected
         Includes processing HISTORY records from `getSpec()`
	 Errors are assumed to be poisonian
       Modifications for fileversion=2:	 
         The output now corrects for coincidence loss, and is in the form of 
	  - photon rate per second per channel with all known corrections (aperture, 
	    sensitivity, coincidence-loss) 
          - errors, including aperture correction, coincidence-loss 
	    correction (non-poissonian).
	 
       The second extension is named CALSPEC  (future: 'FIRST_ORDER_NET_SPECTRUM') 
       contains the standard input for IDL/PYTHON with 
       
       For fileversion=1: 
         First Order: 
         - pixelno(relative to anchor),
         - wave(\AA), 
         - net_count_rate(s-1 channel-1), corrected for PSF, integrated normal  
         - left_background_rate(s-1 channel-1), 
         - right_background_rate (s-1 channel-1),  
         - flux(erg cm-2 s-1 A-1), corrected for aperture, coincidence, sensitivity
         - flux_err(erg cm-2 s-1 A-1), 
         - quality flag, 
         - aper1corr
         Second order: 
         - wave2, 
	 - net2rate,(s-1 channel-1) 
	 - bg2rate, (s-1 channel-1)
	 - flux2, not calibrated, approximate (erg cm-2 s-1 A-1)
	 - flux2err, 
	 - qual2, 
	 - aper2corr
         - coincidence-loss factor as applied to the flux listed. 
       Modifications for fileversion=2: 
         The first extension now containts the corrected 
         count rates, so the uncorrected count rates are now put in the second extension.
         The flux and flux error still have all corrections applied. 
         - net count rate per second per channel in extraction track, uncorrected 
	 - bg1rate is now the background rate per second per channel in extraction track, 
	   uncorrected, new column 
         - bg1rate and bg2rate (count/s/pix) are the mean rates measured 
	   in the two backgrounds and need to be multiplied by the spectra track 
	   width in pixels for comparison to the background rate.
       
       The third extension named 'SPECTRUM_IMAGE' contains the image of the total spectrum
       
       A fourth extension may exist for spectra from summed images 
       and contains the **exposure map** 
  
  **history**
  
    - 2011-12-10 NPM Kuin
    - 2012-03-05 fixed error in calculation bg1rate(a.k.a. bg_r) and bg2rate (or bg_l) 
    - 2012-09-14 added coi correction 
    - 2013-03-06 edited header
  '''	    
   try:
      from astropy.io import fits
   except:   
      import pyfits as fits
   import datetime
   import numpy as np
   from scipy import interpolate
   import os
   from pylab import polyval
   import uvotgetspec

   now = datetime.date.today()
   rnu = now.day*1.2+now.month*0.99+now.year*0.3
   version = '120914'
   h_planck = 6.626e-27  # erg/s
   lightspeed = 2.9979e10  # cm/sec
   h_c_ang = h_planck * lightspeed * 1e8 # ergs-Angstrom
   ch = chatter
   
   trackwidth = uvotgetspec.trackwidth # half-width of the extraction in terms of sigma
   wave2 = None
   sp2netcnt = None
   bg2netcnt = None
   wave3 = None
   quality = None
   qflags = uvotgetspec.quality_flags()
   
      
   ################# prepare data for writing to output file
   ############## main curved slit extraction #####################
   #      
   #     unpack data
   
   if type(Y) == dict:
      Yfit = Y["Yfit"]
      hdr       = Y['hdr']
      wheelpos  = Y['wheelpos']
      offset    = Y['offset']
      sp_first  = Yfit['sp_first']
      bg_first  = Yfit['bg_first']
      sp_second = Yfit['sp_second']
      bg_second = Yfit['bg_second']
      apercorr  = Yfit['apercorr']
      qquality  = Yfit['quality']
      expospec  = Yfit['expospec']
      C_1       = Y['C_1']
      C_2       = Y['C_2']
      x         = Yfit['x']
      q1        = Yfit['q1']
      q2        = Yfit['q2']
      sig1coef  = Yfit['sig1coef']
      sig2coef  = Yfit['sig2coef']
      present2  = Yfit["present2"]
      dist12    = Y['dist12']
      anker     = Y['anker']
      coi_level = Y['coi_level']
      background_strip1 = Y["background_1"]
      background_strip2 = Y["background_2"]
      phx = Y['Xphi']
      phy = Y['Yphi']
      
      anker = Y['anker']
      offset = Y['offset']
      ank_c = Y['ank_c']
      exposure = hdr['exposure']
      extimg = Y['extimg']
      expmap = Y['expmap']
      zeroxy = Y['zeroxy_imgpos']
      # if present background_template extimg is in Y['template']
      effarea1 = Y['effarea1']
      effarea2 = Y['effarea2']

   else:   
      # this will be removed soon in favor of the dictionary passing method
      (Y0,Y1,Y2,Y4) = Y
      
      (filestuff), (method),(phistuff), (dist12, ankerimg, ZOpos), expmap, bgimg, bg_limits_used, bgextra = Y0   
      if filestuff != None:	
         (specfile, lfilt1_, lfilt1_ext_, lfilt2_, lfilt2_ext_, attfile) = filestuff      
         (Xphi, Yphi, date1) = phistuff
      else:
         Xphi,Yphi = 0,0	 
            
      ( (dis,spnet,angle,anker,anker2,anker_field,ank_c), (bg,bg1,bg2,extimg,spimg,spnetimg,offset), 
        (C_1,C_2,img),  hdr,m1,m2,aa,wav1 ) = Y1
      wheelpos = hdr['WHEELPOS']  
      background_strip1 = bg1
      background_strip2 = bg2
	   
      ((present0,present1,present2,present3),(q0,q1,q2,q3), \
       (y0,dlim0L,dlim0U,sig0coef,sp_zeroth),(y1,dlim1L,dlim1U,sig1coef,sp_first),\
       (y2,dlim2L,dlim2U,sig2coef,sp_second),(y3,dlim3L,dlim3U,sig3coef,sp_third),\
       (x,xstart,xend,sp_all,qquality) ), (coef0,coef1,coef2,coef3), \
       (bg_zeroth,bg_first,bg_second,bg_third), (borderup,borderdown),apercorr, expospec  = Y2

      if Y4 != None: 
         wav2p, dis2p, rate2p, qual2p, dist12p = Y4[0]
      phx, phy = anker_field
      zeroxy = [1000,1000]
      effarea1 = None
      effarea2 = None
      
   # ensure that offset is a scalar  
    
   try:
      offset = offset[0]
   except:
      pass

   # first order 
   
   # full arrays   
   bkg = bg_first[q1[0]]   # background at spectrum integrated across width used in sp_first
   counts = sp_first[q1[0]]  # includes background counts
   aper1corr = apercorr[1, q1[0] ]  # aperture factor to multiply counts with  
   quality = qquality[q1[0]] 
   exposure=hdr['exposure']
   expospec1 = expospec[1,q1[0]].flatten()
   wave =  polyval(C_1, x[q1[0]])
   senscorr = sensitivityCorrection(hdr['tstart'])
   background_strip1 = background_strip1[q1[0]]
   background_strip2 = background_strip2[q1[0]]
     
   if wheelpos < 500:     
      qwave = np.where( (wave > 1660.0) & (wave < 6800) )
   else:   
      qwave = np.where( (wave > 2600.0) & (wave < 6400) )
   
   # filtered arrays       
   dis = x[q1[0]][qwave]
   bkg = bkg[qwave]
   counts = counts[qwave]
   aper1corr = aper1corr[qwave]
   expospec1 = expospec1[qwave]
   wave = wave[qwave]
   quality = quality[qwave]
   background_strip1 = background_strip1[qwave]
   background_strip2 = background_strip2[qwave]
   
   # set up channel numbers and reverse for increasing energy   
   NW = len(wave)
   aa = np.arange(NW)
   rc = range(NW)
   rc.reverse()
   channel = aa+1
   ax = range(NW-1) 
   ax1= range(1,NW) 
   dis1 = uvotgetspec.pix_from_wave(C_1,wave, spectralorder=1)
   binwidth = np.polyval(C_1,dis1+0.5) - np.polyval(C_1,dis1-0.5)# width of each bin in A (= scale A/pix)
   
   # derive corrected quantities (sensitivity, aperture, functions for coincidence)   
   sprate = (counts-bkg)*aper1corr/expospec1     # net count rate, PSF aperture corrected 
   sp1rate_err = np.sqrt( 
         np.abs( (bkg*2.5/trackwidth+            # corrected bakcground to 2.5 sigma trackwidth
	       (counts-bkg)*aper1corr)*senscorr  # using PSF aperture correction on net rate
	 *(1+apcorr_errf(trackwidth,wheelpos)))  # error in aperture correction
	       )/expospec1                       # neglects coi-error in rate error 
   bg1rate = bkg/expospec1*2.5/trackwidth        # background counts for 2.5 sigma width at spectrum
   
   
   fcoi = uvotgetspec.coi_func(dis,wave,sprate,bg1rate,sig1coef=sig1coef,option=1,\
       fudgespec=1.322,coi_length=29,frametime=hdr['framtime'], background=False, \
       sigma1_limits=[2.6,4.0], trackwidth = 2.5,debug=False,chatter=1)
   bgcoi = uvotgetspec.coi_func(dis,wave,sprate,bg1rate,sig1coef=sig1coef,option=1,\
       fudgespec=1.322,coi_length=29,frametime=hdr['framtime'], background=True, \
       sigma1_limits=[2.6,4.0], trackwidth = 2.5, debug=False,chatter=1)
       
   sprate = sprate*senscorr
   bg1rate = bg1rate*senscorr    
   
   # get effective area function       
   if arf1 != None: 
      specresp1func = SpecResp(hdr['wheelpos'], 1, arf1=arf1, arf2=arf2 )
   else:
      # position dependent response function [ should get more from rate2flux - too 
      #  much duplication right now ! 
      specresp1func = rate2flux(wave, sprate, wheelpos, bkgrate=bg1rate, 
          pixno=None, sig1coef=[3.2], sigma1_limits=[2.6,4.0], respfunc=True, 
          arf1=None, arf2=None, effarea1=effarea1, effarea2=effarea2, 
	  spectralorder=1, trackwidth = trackwidth, anker=anker, test=None, 
          option=1, fudgespec=1.32, frametime=hdr['framtime'], debug=False, chatter=1)
      #specresp1func = XYSpecResp(wheelpos=hdr['wheelpos'],spectralorder=1, Xank=anker[0], Yank=anker[1]) 
      
   hnu = h_c_ang/(wave)
   # coi-corrected flux
   flux = hnu*sprate*fcoi(wave)/specresp1func(wave)/binwidth   # [erg/s/cm2/angstrom]   #  coi correction
   flux_err = hnu*sp1rate_err*fcoi(wave)/specresp1func(wave)/binwidth 
       
   back1rate = (background_strip1*2*trackwidth*np.polyval(sig1coef,x[q1[0]])[qwave]/expospec1) # estimate bg1 * width 
   back2rate = (background_strip2*2*trackwidth*np.polyval(sig1coef,x[q1[0]])[qwave]/expospec1) # estimate   # prepare for output   
   # extname_order
   if fileversion == 1:
       spectrum_first = (channel, 
          counts[rc], 
          (np.sqrt(counts*aper1corr))[rc], 
	  quality[rc], 
	  aper1corr[rc], 
	  expospec1[rc],  )
       back_first = (channel, bkg[rc], 0.1*np.sqrt(bkg)[rc],
          quality[rc],aper1corr[rc], expospec1[rc])
       calspec_first = (dis,wave,
          sprate*fcoi(wave), # fully corrected counts in the spectrum
          bg1rate*bgcoi(wave), # fully corrected counts in the spectrum
          back2rate*bgcoi(wave), # fully corrected counts in the spectrum         
          flux, flux_err,  # fully corrected flux
	  quality,   
	  aper1corr,
	  expospec1,
	  fcoi(wave),
	  bgcoi(wave))
	  
   elif fileversion == 2:	  
       spectrum_first = (channel, 
                         sprate[rc]*fcoi(wave[rc]), 
			 sp1rate_err[rc]*fcoi(wave[rc]),
			 quality[rc] )
       back_first = (channel, bg1rate[rc], 0.1*np.sqrt(bg1rate)[rc],
          quality[rc],aper1corr[rc], expospec1[rc])
       calspec_first = (dis,wave,
          (counts-bkg)/expospec1, bkg/expospec1,   # uncorrected counts in the spectrum
          [back1rate,back2rate],
          flux, flux_err,  # fully corrected flux
	  quality,   
	  aper1corr,
	  expospec1,
	  fcoi(wave),
	  bgcoi(wave))
           
   ############### second order
   # predicted second order
   # already rate in array         
            
   # measured second order
      
   if present2 & fit_second: 
   
         wave2 = polyval(C_2, x[q2[0]]-dist12) # 2nd order wavelengths
	 channel2 = np.arange(len(wave2)) +1
	 rc2 = range(len(wave2))
	 rc2.reverse()     
         sp2counts = sp_second[q2[0]]                                   # total counts in second order
	 aper2corr = apercorr[2, q2[0]].flatten()                       # aperture correction second order
         expospec2 = expospec[2, q2[0]].flatten()
         bg_2cnt    = bg_second[q2[0]]                                  # background counts second order
	 
         sp2netcnt = sp2counts - bg_2cnt                                # net counts, uncorrected 
	 sp2rate = sp2netcnt * aper2corr / expospec2         # net count rate, aperture corrected 
	 sp2rate_err = np.sqrt(
	    np.abs(
	    (bg_2cnt*2.5/trackwidth+sp2netcnt*aper2corr)*
	    senscorr*(1+
	    apcorr_errf(trackwidth,wheelpos) )))/expospec2   # fully corrected error in net count rate
	 bg_2rate = bg_2cnt /expospec2 * 2.5/trackwidth      # aperture corrected background rate 

	 qual2 = qquality[q2[0]]                                        # quality at second order (shared with first order)
	 dis2 = uvotgetspec.pix_from_wave( C_2, wave2,spectralorder=2 )
         binwidth2 = np.polyval(C_2,dis2+0.5) - np.polyval(C_2,dis2-0.5)
	 pix2      = x[q2[0]]                                           # pixel number to align with first order
	 
         fcoi_2 = uvotgetspec.coi_func(pix2,wave2,sp2rate,bg_2rate,sig1coef=sig2coef,option=1,\
            fudgespec=1.32,coi_length=29,frametime=hdr['framtime'], background=False, \
            sigma1_limits=[2.6,6.0], trackwidth = 2.5,debug=False,chatter=1)
         bgcoi2 = uvotgetspec.coi_func(pix2,wave2,sp2rate,bg_2rate,sig1coef=sig2coef,option=1,\
            fudgespec=1.32,coi_length=29,frametime=hdr['framtime'], background=True, \
            sigma1_limits=[2.6,6.0], trackwidth = 2.5, debug=False,chatter=1)

         sp2rate = sp2rate * senscorr    # perform sensitivity loss correction
         bg_2rate = bg_2rate * senscorr

         #specresp1func = SpecResp(hdr['wheelpos'], 1, arf1=arf1, arf2=arf2 )
         if arf2 != None: 
            specresp2func = SpecResp(hdr['wheelpos'], 2, arf1=arf1, arf2=arf2 )
         else:
            # position dependent response function
            specresp2func = XYSpecResp(wheelpos=hdr['wheelpos'],spectralorder=2, anker=anker ) 
         
         hnu2 = h_c_ang/(wave2)
	 
         flux2 = hnu2*sp2rate*fcoi_2(wave2)/specresp2func(wave2)/binwidth2 # corrected [erg/s/cm2/angstrom]
         flux2_err = hnu2*sp2rate_err*fcoi_2(wave2)/specresp2func(wave2)/binwidth2 
      
         # collect data for output
	 if fileversion == 1:
            spectrum_second = (channel2, sp2counts[rc2], 
	      (np.sqrt(sp2counts))[rc2], qual2[rc2], 
	      aper2corr[rc2], expospec2[rc2] )
	    back_second = (channel2, bg_2cnt[rc2], 
	      0.1*(np.sqrt(bg_2cnt))[rc2], 
	      qual2[rc2], aper2corr[rc2],expospec2[rc2] )
            calspec_second = (pix2,wave2,
	       sp2rate*fcoi_2(wave2),
	       bg_2rate*fcoi_2(wave2),
	       flux2,flux2_err,
	       qual2,aper2corr, 
	       expospec2,fcoi_2(wave2),bgcoi2(wave2))
	 else:     
            spectrum_second = (channel2, 
	       sp2rate[rc2]*fcoi_2(wave2[rc2]), 
	       sp2rate_err[rc2]*fcoi_2(wave2[rc2]), 
	       qual2[rc2])
	    back_second = (channel2, bg_2rate[rc2]*bgcoi2(wave2[rc2]), 
	       0.1*(np.sqrt(bg_2cnt))[rc2], 
	       qual2[rc2] )
            calspec_second = (pix2,wave2,
	       (sp2counts-bg_2cnt)/expospec2,
	       bg_2cnt/expospec2,
	       flux2,flux2_err,
	       qual2,aper2corr, 
	       expospec2,fcoi_2(wave2),bgcoi2(wave2))
   else:
         spectrum_second = None	
	 back_second = None
	 calspec_second = None 
   
   if hdr['wheelpos'] > 500: present2 = False
                
   ############### FILES   Define the input/ output file names  
         
   obsid = filestub  
   if fileoutstub != None: 
      obsid = fileoutstub 
      if chatter > 2: print "output file name base is now:",obsid
   if used_lenticular: 
      lent='_f'
   else: 
      lent='_g'      
   if hdr['wheelpos'] == 200:
      outfile1 = obsid+'ugu_1ord_'+str(ext)+lent+'.pha'
      backfile1 = obsid+'ugu_1ord_'+str(ext)+lent+'_back.pha'
      outfile2 = obsid+'ugu_2ord_'+str(ext)+lent+'.pha'
      backfile2 = obsid+'ugu_2ord_'+str(ext)+lent+'_back.pha'
      rmffile1 = obsid+'ugu_1ord_'+str(ext)+lent+'.rmf'
      rmffile2 = obsid+'ugu_2ord_'+str(ext)+lent+'.rmf'
   elif hdr['wheelpos'] == 160:
      outfile1 = obsid+'ugu_1ord_'+str(ext)+lent+'.pha'
      backfile1 = obsid+'ugu_1ord_'+str(ext)+lent+'_back.pha'
      outfile2 = obsid+'ugu_2ord_'+str(ext)+lent+'.pha'
      backfile2 = obsid+'ugu_2ord_'+str(ext)+lent+'_back.pha'
      rmffile1 = obsid+'ugu_1ord_'+str(ext)+lent+'.rmf'
      rmffile2 = obsid+'ugu_2ord_'+str(ext)+lent+'.rmf'
   elif hdr['wheelpos'] == 1000:
      outfile1 = obsid+'ugv_1ord_'+str(ext)+lent+'.pha'
      backfile1 = obsid+'ugv_1ord_'+str(ext)+lent+'_back.pha'
      outfile2 = obsid+'ugv_2ord_'+str(ext)+lent+'.pha'
      backfile2 = obsid+'ugv_2ord_'+str(ext)+lent+'_back.pha'
      rmffile1 = obsid+'ugv_1ord_'+str(ext)+lent+'.rmf'
      rmffile2 = obsid+'ugv_2ord_'+str(ext)+lent+'.rmf'
   elif hdr['wheelpos'] == 955: 
      outfile1 = obsid+'ugv_1ord_'+str(ext)+lent+'.pha'
      backfile1 = obsid+'ugv_1ord_'+str(ext)+lent+'_back.pha'
      outfile2 = obsid+'ugv_2ord_'+str(ext)+lent+'.pha'
      backfile2 = obsid+'ugv_2ord_'+str(ext)+lent+'_back.pha'
      rmffile1 = obsid+'ugv_1ord_'+str(ext)+lent+'.rmf'
      rmffile2 = obsid+'ugv_2ord_'+str(ext)+lent+'.rmf'
   else:   
      print "FATAL: exposure header does not have filterwheel position encoded"
      return 
   obsid = filestub   
      
   # test for presence of outfile and clobber not set
   
   if clobber == False:
      if (os.access(outfile1,os.F_OK) ^ os.access(backfile1,os.F_OK) 
        ^ os.access(outfile2,os.F_OK) ^ os.access(backfile2,os.F_OK)):
         print 'Error: output file already present. '
         if write_rmffile & (not os.access(rmffile1,os.F_OK)):
             write_rmf_file (rmffile1, wave, hdr['wheelpos'], C_1, 
	        anchor=anker, clobber=clobber,chatter=chatter  )
         if present2 & fit_second & write_rmffile & (not os.access(rmffile2,os.F_OK)):
           # write_rmf_file (rmffile2, wave2, hdr['wheelpos'],2, C_2, 
	   #     arf1=None, arf2=arf2, clobber=clobber,chatter=chatter   )
	   print "no RMF file for second order available"
	 if interactive:
   	    answer =  raw_input('       DO YOU WANT TO REWRITE THE OUTPUT FILES (answer yes/NO)? ')
	    if len(answer) < 1:  answer = 'NO'
	    answer = answer.upper()
	    if (answer == 'Y') ^ (answer == 'YES'):  clobber = True	        
         if clobber == False: return 
   filetag = obsid+'_'+str(hdr['wheelpos'])+str(rnu)
      
   writeSpectrum_ (ra,dec,obsid,ext,hdr,anker,phx,phy,offset, 
      ank_c, exposure, history, spectrum_first, back_first, 
      calspec_first, extimg, outfile1, backfile1, rmffile1, 
      outfile2=outfile2, backfile2=backfile2, rmffile2=rmffile2, 
      expmap=expmap, spectrum_second = spectrum_second, 
      back_second = back_second, calspec_second=calspec_second, 
      present2=(present2 & fit_second), fileversion=fileversion,
      zeroxy=zeroxy,
      clobber=clobber, chatter=chatter)
      
   if write_rmffile: 
      write_rmf_file (rmffile1, wave, hdr['wheelpos'], C_1, effarea1=effarea1,  
            anchor=anker, clobber=clobber,chatter=chatter  )
         
   if present2 & fit_second & write_rmffile:
      #write_rmf_file (rmffile2, wave2, hdr['wheelpos'],2, C_2, 
      #      arf1=None, arf2=arf2, clobber=clobber,chatter=chatter   )
      print "no RMF file for second order available"
      
def apcorr_errf(trackwidth,wheelpos):
   "The additional RMS percentage rate error when making an aperture correction"
   x = (trackwidth-2.5) 
   if wheelpos < 500:
      if x < 1.1: return 5.0
      if x < 2.0: return 3.0
      return 3.0*x                 
   else:   
      if x < 1.1: return 8.0
      if x < 2.0: return 5.0
      return 5.0*x                 

def writeSpectrum_ (ra,dec,obsid,ext,hdr,anker,phx,phy,offset, ank_c, exposure,  
   history, spectrum_first, back_first, calspec_first, extimg, outfile1, 
   backfile1, rmffile1, outfile2=None, backfile2=None, rmffile2=None, expmap=None,   
   spectrum_second = None, back_second = None, calspec_second=None, present2=False, 
   fileversion = 1, zeroxy=[1000,1000],
   clobber=False, chatter=0 ):
   '''performs the actual creation of the output file (mostly FITS stuff) 
   
   See writeSpectrum() for the parameter meanings
   '''
   try: 
     from astropy.io import fits 
   except:
     import pyfits as fits
   import datetime
   import numpy as N
   from scipy import interpolate
   import os
   from pylab import polyval

   version = '140101'
   
   now = datetime.date.today()
   datestring = now.isoformat()[0:4]+now.isoformat()[5:7]+now.isoformat()[8:10]
   rnu = int(now.day*1.2+now.month*0.99+now.year*0.3)   # some number

   orders = '1'
   if present2: orders='12'
   
   outfile2nd = outfile2
   backfile2nd = backfile2
   filetag = obsid+'_'+str(ext)+'_'+str(rnu)
   #   
   #                 ============= main spectrum pha file =======
   #
   # create primary header 
   #
   hdu0 = fits.PrimaryHDU()
   hdu0.header.update('CREATED','written by uvotio.py '+version)
   hdu0.header.update('DATE',str(now))
   hdu0.header.update('AUTHOR','UVOTPY author is NPM Kuin (UCL/MSSL)')
   hdu0.header.update('WHEELPOS',hdr['wheelpos'])
   hdu0.header.update('FILTER',hdr['filter'],comment='UVOT filter used')
   hdu0.header.update('ORDERS',orders,comment='list of orders included')
   hdu0.header.update('TELESCOP','SWIFT   ',comment='Telescope (mission) name')
   hdu0.header.update('INSTRUME','UVOTA   ',comment='Instrument Name')
   hdu0.header.update('FILEVERS',fileversion,comment='UVOTPY file version')
   hdulist=fits.HDUList([hdu0])
   #
   #  Main fits part
   #
   hdr0 = hdr.copy()
   hdr0.update('ORI_FILE',obsid+'+'+str(ext),'fileid and extension of extracted spectrum')
   hdr0.update('RA_X',ra,'RA of source extracted spectrum')
   hdr0.update('DEC_X',dec,'DEC of source extracted spectrum')
   hdr0.update('DETX_X',anker[0],'XDET position source anker in DET coord') 
   hdr0.update('DETY_X',anker[1],'YDET position source anker in DET coord')
   hdr0.update('POSX_AS',phx,'angle boresight in deg in DETX direction')
   hdr0.update('POSY_AS',phy,'angle boresight in deg in DETY direction')
   hdr0.update('SPEC_OFF',offset,'distance to spectrum from anker position (DETX_X,DETY_X)')
   #
   #  first extension: first order spectrum ; add extname everywhere
   #
   if fileversion == 1:
      col11 = fits.Column(name='CHANNEL ',format='I',array=spectrum_first[0])
      col12 = fits.Column(name='COUNTS  ',format='I',array=spectrum_first[1],unit='COUNTS')
      col13 = fits.Column(name='STAT_ERR',format='E',array=spectrum_first[2],unit='COUNTS')
      col14 = fits.Column(name='QUALITY ',format='I',array=spectrum_first[3] )
      col15 = fits.Column(name='APERCORR',format='E',array=spectrum_first[4] )
      col16 = fits.Column(name='EXPOSURE',format='E',array=spectrum_first[5],unit='s' )
      cols1 = fits.ColDefs([col11,col12,col13,col14,col15,col16])
   elif fileversion == 2:
      col11 = fits.Column(name='CHANNEL ',format='I',array=spectrum_first[0])
      col12 = fits.Column(name='RATE    ',format='E',array=spectrum_first[1],unit='counts/s')
      col13 = fits.Column(name='STAT_ERR',format='E',array=spectrum_first[2],unit='counts/s')
      #col14 = fits.Column(name='SYS_ERR',format='E',array=spectrum_first[2],unit='counts/s')
      col15 = fits.Column(name='QUALITY ',format='I',array=spectrum_first[3] )
      cols1 = fits.ColDefs([col11,col12,col13,col15])
   tbhdu1 = fits.new_table(cols1)
   if fileversion == 1:
      tbhdu1.header.update('comment','COUNTS are observed, uncorrected counts')
   elif fileversion == 2:
      tbhdu1.header.update('comment','RATE are the fully corrected count rates')
      
   tbhdu1.header.update('EXPID',hdr['expid'],'Exposure ID')
   tbhdu1.header.update('EXTNAME','SPECTRUM','Name of this binary table extension')
   tbhdu1.header.update('TELESCOP','SWIFT   ',comment='Telescope (mission) name')
   tbhdu1.header.update('INSTRUME','UVOTA   ',comment='Instrument Name')
   tbhdu1.header.update('TIMESYS ',hdr['timesys'],'time system')
   tbhdu1.header.update('FILETAG',filetag,'unique set id')
   tbhdu1.header.update('MJDREFI ',hdr['mjdrefi'],'Reference MJD time integer part')
   tbhdu1.header.update('MJDREFF ',hdr['mjdreff'],'Reference MJD fractional part')
   tbhdu1.header.update('TIMEREF ',hdr['timeref'],'time reference barycentric/local')
   tbhdu1.header.update('TASSIGN ',hdr['tassign'],'time assigned by clock')
   tbhdu1.header.update('TIMEUNIT',hdr['timeunit'])
   tbhdu1.header.update('TIERRELA',hdr['TIERRELA'],'time relative error [s/s]')
   tbhdu1.header.update('TIERABSO',hdr['TIERABSO'],'timing precision in seconds')
   tbhdu1.header.update('TSTART',hdr['TSTART'])
   tbhdu1.header.update('TSTOP',hdr['TSTOP'])
   tbhdu1.header.update('DATE-OBS',hdr['DATE-OBS'])
   tbhdu1.header.update('DATE-END',hdr['DATE-END'])
   tbhdu1.header.update('CLOCKAPP',hdr['CLOCKAPP'],'if clock correction was applied')
   tbhdu1.header.update('TELAPSE',hdr['TELAPSE'],'Tstop - Tstart')
   tbhdu1.header.update('EXPOSURE',hdr['EXPOSURE'],'Average Total exposure, with all known corrections')
   tbhdu1.header.update('DEADC',hdr['DEADC'],'dead time correction')
   tbhdu1.header.update('FRAMTIME',hdr['FRAMTIME'],'frame exposure time')
   tbhdu1.header.update('DETNAM',hdr['DETNAM'])
   tbhdu1.header.update('FILTER',hdr['FILTER'])
   tbhdu1.header.update('OBS_ID',hdr['OBS_ID'],'observation id')
   tbhdu1.header.update('TARG_ID',hdr['TARG_ID'],'Target ID')
   #tbhdu1.header.update('SEQ_NUM',hdr['SEQ_NUM'])
   tbhdu1.header.update('EQUINOX',hdr['EQUINOX'])
   tbhdu1.header.update('RADECSYS',hdr['RADECSYS'])
   tbhdu1.header.update('WHEELPOS',hdr['WHEELPOS'],'filterweel position')
   tbhdu1.header.update('SPECTORD',1,'spectral order')
   try:
      tbhdu1.header.update('BLOCLOSS',hdr['BLOCLOSS'],'[s] Exposure time under BLOCKED filter')
      tbhdu1.header.update('STALLOSS',hdr['STALLOSS'],'[s] Est DPU stalling time loss')
      tbhdu1.header.update('TOSSLOSS',hdr['TOSSLOSS'],'[s] Est Shift&Toss time loss')
      tbhdu1.header.update('MOD8CORR',hdr['MOD8CORR'],'Was MOD8 correction applied')
      tbhdu1.header.update('FLATCORR',hdr['FLATCORR'],'was flat field correction applied')
   except:
      pass   
   tbhdu1.header.update('ASPCORR',hdr['ASPCORR'],'Aspect correction method')
   tbhdu1.header.update('HDUCLASS','OGIP','format attemts to follow OGIP standard')
   tbhdu1.header.update('HDUCLAS1','SPECTRUM','PHA dataset (OGIP memo OGIP-92-007')
   if fileversion == 1:
      tbhdu1.header.update('HDUCLAS2','TOTAL','Gross PHA Spectrum (source + background)')
      tbhdu1.header.update('HDUCLAS3','COUNT','PHA data stored as counts (not count/s)')
      tbhdu1.header.update('POISSERR','F','Poissonian errors not applicable')
      tbhdu1.header.update('BACKFILE',backfile1,'Background FITS file')
   elif fileversion == 2:   
      tbhdu1.header.update('HDUCLAS2','NET','Gross PHA Spectrum (source only)')
      tbhdu1.header.update('HDUCLAS3','RATE','PHA data stored as rate (not count)')
      tbhdu1.header.update('POISSERR','F','Poissonian errors not applicable')
      #tbhdu1.header.update('BACKFILE',None,'No Background FITS file')
   tbhdu1.header.update('HDUVERS1','1.1.0','Version of format (OGIP memo OGIP-92-007a)')
   tbhdu1.header.update('CHANTYPE','PHA','Type of channel PHA/PI')
   tbhdu1.header.update('TLMIN1  ',1,'Lowest legal channel number')
   tbhdu1.header.update('TLMAX1',len(spectrum_first[0]),'Highest legal channel number')
   tbhdu1.header.update('GROUPING',0,'No grouping of the data has been defined')
   tbhdu1.header.update('DETCHANS',len(spectrum_first[0]),'Total number of detector channels available')
   tbhdu1.header.update('AREASCAL',1,'Area scaling factor')
   tbhdu1.header.update('BACKSCAL',1,'Background scaling factor')
   tbhdu1.header.update('CORRSCAL',1,'Correlation scaling factor')
   tbhdu1.header.update('CORRFILE','NONE  ','Correlation FITS file')
   tbhdu1.header.update('RESPFILE','NONE','Redistribution matrix')
#   tbhdu1.header.update('RESPFILE',respfile,'Redistribution matrix')
   tbhdu1.header.update('ANCRFILE','NONE  ','Ancillary response')
   tbhdu1.header.update('XFLT0001','NONE  ','XSPEC selection filter description')
   tbhdu1.header.update('CRPIX1  ','(1,'+str(len(spectrum_first[0]))+')','Channel binning of the CHANNEL column')
   tbhdu1.header.update('PHAVERSN','1992a ','OGIP memo number for file format') 
   # convert ra,dec -> zerodetx,zerodety using uvotapplywcs?  
   # look in code uvotimgrism
   tbhdu1.header.update('ZERODETX',zeroxy[0],'zeroth order position on image')
   tbhdu1.header.update('ZERODETY',zeroxy[1],'zeroth order position on image')   
   hdulist.append(tbhdu1)
   #
   #  second extension first order
   # 
   if fileversion == 2:
       (dis,wave,sprate,bg1rate,bck_strips,flux,flux_err,
        quality,aper1corr,expospec1,coi_sp1,bgcoi_sp1)  = calspec_first
   
       col23  = fits.Column(name='BKGRATE1',format='E',array=bg1rate,unit='c/s')
       col24A = fits.Column(name='BG_L   ',format='E',array=bck_strips[0],unit='c/s')
       col24B = fits.Column(name='BG_R   ',format='E',array=bck_strips[1],unit='c/s')
   elif fileversion == 1:
       (dis,wave,
          sprate, # fully corrected counts in the spectrum
          bg1rate, # fully corrected background counts under the spectrum
          bg2rate,
          flux, flux_err,  # fully corrected flux
	  quality,   
	  aper1corr,
	  expospec1,
	  coi_sp1,
	  bgcoi_sp1 ) = calspec_first
       col24A = fits.Column(name='BG_L   ',format='E',array=bg1rate,unit='c/s')
       col24B = fits.Column(name='BG_R   ',format='E',array=bg2rate,unit='c/s')
   col20 = fits.Column(name='PIXNO  ',format='I',array=dis    ,unit='pix')
   col21 = fits.Column(name='LAMBDA ',format='E',array=wave   ,unit='A')
   col22 = fits.Column(name='NETRATE',format='E',array=sprate ,unit='c/s')   
   col25 = fits.Column(name='FLUX   ',format='E',array=flux   ,unit='erg cm-2 s-1 A-1')
   col26 = fits.Column(name='FLUXERR',format='E',array=flux_err,unit='erg cm-2 s-1 A-1')
   col27 = fits.Column(name='QUALITY',format='I',array=quality,unit='NONE')
   col28 = fits.Column(name='APERCOR1',format='E',array=aper1corr,unit='NONE')
   col29 = fits.Column(name='EXPOSURE',format='E',array=expospec1,unit='s')
   col29A = fits.Column(name='SP1_COIF',format='E',array=coi_sp1,unit='NONE')
   col29B = fits.Column(name='BG1_COIF',format='E',array=bgcoi_sp1,unit='NONE')
   if present2:  # second order
      (pix2,wave2,sp2rate,bg_2rate,flux2,flux2_err,qual2,aper2corr,expospec2,
            coi_sp2, bgcoi_sp2) = calspec_second
      col30 = fits.Column(name='PIXNO2',format='I',array=pix2,unit='pix')
      col31 = fits.Column(name='LAMBDA2',format='E',array=wave2,unit='A')
      col32 = fits.Column(name='NETRATE2',format='E',array=sp2rate,unit='c/s')
      col33 = fits.Column(name='BGRATE2',format='E',array=bg_2rate,unit='c/s')
      col34 = fits.Column(name='FLUX2',format='E',array=flux2,unit='erg cm-2 s-1 A-1')
      col35 = fits.Column(name='FLUXERR2',format='E',array=flux2_err,unit='erg cm-2 s-1 A-1')
      col36 = fits.Column(name='QUALITY2',format='I',array=qual2,unit='NONE')
      col37 = fits.Column(name='APERCOR2',format='E',array=aper2corr,unit='NONE')
      #col38 = fits.Column(name='EXPOSURE',format='E',array=expospec2,unit='s')
      col38A = fits.Column(name='SP2_COI',format='E',array=coi_sp2,unit='NONE')
      col38B = fits.Column(name='BG2_COI',format='E',array=bgcoi_sp2,unit='NONE')
      if fileversion == 1:
          cols2 = fits.ColDefs([col20,col21,col22,col24A,col25,col26,col27,col28,
	           col29,col29A,col29B,col30,col31,col32,col33,col34,col35,col36,
		   col37,col38,col38A,col24B])      
      elif fileversion == 2:
          cols2 = fits.ColDefs([col20,col21,col22,col23,col25,col26,col27,col28,
	           col29,col29A,col29B,col30,col31,col32,col33,col34,col35,col36,
		   col37,col38A,col38B,col24A,col24B])
   else: # not present2
      if fileversion == 1:  
          cols2 = fits.ColDefs([col20,col21,col22,col24A,col25,col26,col27,
	          col28,col29,col29A,col29B,col24B]) 
      elif fileversion == 2:
          cols2 = fits.ColDefs([col20,col21,col22,col23,col25,col26,col27,
	          col28,col29,col29A,col29B,col24A,col24B])
      
   tbhdu2 = fits.new_table(cols2)
   if fileversion == 1:
       tbhdu2.header.update('HISTORY','coi-loss, aperture - corrected flux and rates')
   elif fileversion == 2: 
       tbhdu2.header.update('HISTORY','no coi-loss, no aperture - uncorrected rates')
       tbhdu2.header.update('HISTORY','coi-loss, aperture - corrected flux and rates')
   if history != None:
      msg1 = history.split('\n')
      for msg in msg1: tbhdu1.header.add_history(msg)
   tbhdu2.header.update('EXPID',hdr['expid'])
#   tbhdu2.header.update('EXTNAME','FIRST_ORDER_NET_SPECTRUM')
   tbhdu2.header.update('EXTNAME','CALSPEC')
   tbhdu2.header.update('FILETAG',filetag,'unique set id')
   tbhdu2.header.update('TELESCOP','SWIFT   ',comment='Telescope (mission) name')
   tbhdu2.header.update('INSTRUME','UVOTA   ',comment='Instrument Name')
   tbhdu2.header.update('TIMESYS ',hdr['timesys'])
   tbhdu2.header.update('MJDREFI ',hdr['mjdrefi'])
   tbhdu2.header.update('MJDREFF ',hdr['mjdreff'])
   tbhdu2.header.update('TIMEREF ',hdr['timeref'])
   tbhdu2.header.update('TASSIGN ',hdr['tassign'])
   tbhdu2.header.update('TIMEUNIT',hdr['timeunit'])
   tbhdu2.header.update('TIERRELA',hdr['TIERRELA'])
   tbhdu2.header.update('TIERABSO',hdr['TIERABSO'])
   tbhdu2.header.update('TSTART',hdr['TSTART'])
   tbhdu2.header.update('TSTOP',hdr['TSTOP'])
   tbhdu2.header.update('DATE-OBS',hdr['DATE-OBS'])
   tbhdu2.header.update('DATE-END',hdr['DATE-END'])
   tbhdu2.header.update('CLOCKAPP',hdr['CLOCKAPP'])
   tbhdu2.header.update('TELAPSE',hdr['TELAPSE'])
   tbhdu2.header.update('EXPOSURE',hdr['EXPOSURE'])
   tbhdu2.header.update('DEADC',hdr['DEADC'])
   tbhdu2.header.update('FRAMTIME',hdr['FRAMTIME'])
   tbhdu2.header.update('DETNAM',hdr['DETNAM'])
   tbhdu2.header.update('FILTER',hdr['FILTER'])
   tbhdu2.header.update('OBS_ID',hdr['OBS_ID'])
   tbhdu2.header.update('TARG_ID',hdr['TARG_ID'])
   tbhdu2.header.update('EQUINOX',hdr['EQUINOX'])
   tbhdu2.header.update('RADECSYS',hdr['RADECSYS'])
   tbhdu2.header.update('WHEELPOS',hdr['WHEELPOS'])
   try:
      tbhdu2.header.update('BLOCLOSS',hdr['BLOCLOSS'])
      tbhdu2.header.update('MOD8CORR',hdr['MOD8CORR'])
      tbhdu2.header.update('FLATCORR',hdr['FLATCORR'])
      tbhdu2.header.update('STALLOSS',hdr['STALLOSS'])
      tbhdu2.header.update('TOSSLOSS',hdr['TOSSLOSS'])
   except:
      pass   
   tbhdu2.header.update('HDUCLASS','OGIP')
   tbhdu2.header.update('HDUCLAS1','SPECTRUM')
   if fileversion == 1:
      tbhdu2.header.update('HDUCLAS2','TOTAL')   
   elif fileversion == 2:
      tbhdu2.header.update('HDUCLAS2','NET')
   tbhdu2.header.update('ZERODETX',zeroxy[0],'zeroth order position on image')
   tbhdu2.header.update('ZERODETY',zeroxy[1],'zeroth order position on image')   
   hdulist.append(tbhdu2)
   #
   #  THIRD extension: extracted image
   #
   hdu3 = fits.ImageHDU(extimg)
   hdu3.header.update('EXTNAME','SPECTRUM_IMAGE')
   hdu3.header.update('EXPID',hdr['expid'])
   hdu3.header.update('ANKXIMG',ank_c[1],'Position anchor in image')
   hdu3.header.update('ANKYIMG',ank_c[0],'Position anchor in image')
   hdu3.header.update('FILETAG',filetag,'unique set id')
   hdulist.append(hdu3)
   #
   #  FOURTH extension: extracted image
   #
   if len(expmap) > 1: 
      hdu4 = fits.ImageHDU(expmap)
      hdu4.header.update('EXTNAME','EXPOSURE_MAP')
      hdu4.header.update('EXPID',hdr['expid'])
      hdu4.header.update('ANKXIMG',ank_c[1],'Position anchor in image')
      hdu4.header.update('ANKYIMG',ank_c[0],'Position anchor in image')
      hdu4.header.update('FILETAG',filetag,'unique set id')
      hdulist.append(hdu4)
   try:   
     hdulist.writeto(outfile1,clobber=clobber)
   except:
      print "WARNING : NO OUTPUT FILE CREATED.  "+outfile1+" EXISTS and CLOBBER not set" 
      pass  
   #
   #   
   ################ ============= second order spectrum pha file ======= ###############
   if present2:
     #
     # create primary header 
     #
     hdu0 = fits.PrimaryHDU()
     hdu0.header.update('CREATED','written by uvotio.py '+version)
     hdu0.header.update('DATE',str(now))
     hdu0.header.update('AUTHOR','UVOTPY author is NPM Kuin (UCL/MSSL)')
     hdu0.header.update('WHEELPOS',hdr['wheelpos'])
     hdu0.header.update('FILTER',hdr['filter'],comment='UVOT filter used')
     hdu0.header.update('TELESCOP','SWIFT   ',comment='Telescope (mission) name')
     hdu0.header.update('INSTRUME','UVOTA   ',comment='Instrument Name')
     hdulist=fits.HDUList([hdu0])
     #
     #  Main fits part
     #
     hdr0 = hdr.copy()
     hdr0.update('ORI_FILE',obsid+'+'+str(ext),'fileid and extension of extracted spectrum')
     hdr0.update('RA_X',ra,'RA of source extracted spectrum')
     hdr0.update('DEC_X',dec,'DEC of source extracted spectrum')
     hdr0.update('DETX_X',anker[0],'XDET position source anker in DET coord') 
     hdr0.update('DETY_X',anker[1],'YDET position source anker in DET coord')
     hdr0.update('POSX_AS',phx,'angle boresight in deg in DETX direction')
     hdr0.update('POSY_AS',phy,'angle boresight in deg in DETY direction')
     hdr0.update('SPEC_OFF',offset,'distance to spectrum from anker position (DETX_X,DETY_X)')
     #
     #  first extension: first order spectrum ; add extname everywhere
     #
     if fileversion == 1:
        (channel2, sp2counts, stat_err, qual2, aper2corr,expospec2 ) = spectrum_second
        col21 = fits.Column(name='CHANNEL ',format='I',array=channel2 )
        col22 = fits.Column(name='COUNTS  ',format='I',array=sp2counts,unit='COUNTS')
        col23 = fits.Column(name='STAT_ERR',format='E',array=stat_err,unit='COUNTS')
        col24 = fits.Column(name='QUALITY ',format='I',array=qual2 )
        col25 = fits.Column(name='APERCORR',format='E',array=aper2corr )
        col26 = fits.Column(name='EXPOSURE',format='E',array=expospec2,unit='s' )
        cols1 = fits.ColDefs([col21,col22,col23,col24,col25,col26])
     elif fileversion == 2:	
        (channel2, sp2rate, rate_err, qual2) = spectrum_second
        col21 = fits.Column(name='CHANNEL ',format='I',array=channel2 )
        col22 = fits.Column(name='RATE  ',format='E',array=sp2rate,unit='COUNTS/S')
        col23 = fits.Column(name='STAT_ERR',format='E',array=rate_err,unit='COUNTS/S')
        col24 = fits.Column(name='QUALITY ',format='I',array=qual2 )
        cols1 = fits.ColDefs([col21,col22,col23,col24])
     tbhdu1 = fits.new_table(cols1)
     tbhdu1.header.update('EXPID',hdr['expid'],'Exposure ID')
     tbhdu1.header.update('EXTNAME','SPECTRUM','Name of this binary table extension')
     tbhdu1.header.update('TELESCOP','SWIFT   ',comment='Telescope (mission) name')
     tbhdu1.header.update('INSTRUME','UVOTA   ',comment='Instrument Name')
     tbhdu1.header.update('TIMESYS ',hdr['timesys'],'time system')
     tbhdu1.header.update('FILETAG',filetag,'unique set id')
     tbhdu1.header.update('MJDREFI ',hdr['mjdrefi'],'Reference MJD time integer part')
     tbhdu1.header.update('MJDREFF ',hdr['mjdreff'],'Reference MJD fractional part')
     tbhdu1.header.update('TIMEREF ',hdr['timeref'],'time reference barycentric/local')
     tbhdu1.header.update('TASSIGN ',hdr['tassign'],'time assigned by clock')
     tbhdu1.header.update('TIMEUNIT',hdr['timeunit'])
     tbhdu1.header.update('TIERRELA',hdr['TIERRELA'],'time relative error [s/s]')
     tbhdu1.header.update('TIERABSO',hdr['TIERABSO'],'timing precision in seconds')
     tbhdu1.header.update('TSTART',hdr['TSTART'])
     tbhdu1.header.update('TSTOP',hdr['TSTOP'])
     tbhdu1.header.update('DATE-OBS',hdr['DATE-OBS'])
     tbhdu1.header.update('DATE-END',hdr['DATE-END'])
     tbhdu1.header.update('CLOCKAPP',hdr['CLOCKAPP'],'if clock correction was applied')
     tbhdu1.header.update('TELAPSE',hdr['TELAPSE'],'Tstop - Tstart')
     tbhdu1.header.update('EXPOSURE',hdr['EXPOSURE'],'Average Total exposure, with all known corrections')
     tbhdu1.header.update('DEADC',hdr['DEADC'],'dead time correction')
     tbhdu1.header.update('FRAMTIME',hdr['FRAMTIME'],'frame exposure time')
     tbhdu1.header.update('DETNAM',hdr['DETNAM'])
     tbhdu1.header.update('FILTER',hdr['FILTER'])
     tbhdu1.header.update('OBS_ID',hdr['OBS_ID'],'observation id')
     tbhdu1.header.update('TARG_ID',hdr['TARG_ID'],'Target ID')
     tbhdu1.header.update('EQUINOX',hdr['EQUINOX'])
     tbhdu1.header.update('RADECSYS',hdr['RADECSYS'])
     tbhdu1.header.update('WHEELPOS',hdr['WHEELPOS'],'filterweel position')
     tbhdu1.header.update('SPECTORD',2,'spectral order')
     try:
       tbhdu1.header.update('BLOCLOSS',hdr['BLOCLOSS'],'[s] Exposure time under BLOCKED filter')
       tbhdu1.header.update('STALLOSS',hdr['STALLOSS'],'[s] Est DPU stalling time loss')
       tbhdu1.header.update('TOSSLOSS',hdr['TOSSLOSS'],'[s] Est Shift&Toss time loss')
       tbhdu1.header.update('MOD8CORR',hdr['MOD8CORR'],'Was MOD8 correction applied')
       tbhdu1.header.update('FLATCORR',hdr['FLATCORR'],'was flat field correction applied')
     except:
        pass   
     tbhdu1.header.update('ASPCORR',hdr['ASPCORR'],'Aspect correction method')
     tbhdu1.header.update('HDUCLASS','OGIP','format attemts to follow OGIP standard')
     tbhdu1.header.update('HDUCLAS1','SPECTRUM','PHA dataset (OGIP memo OGIP-92-007')
     if fileversion == 1:
       tbhdu1.header.update('HDUCLAS2','TOTAL','Gross PHA Spectrum (source + background)')
       tbhdu1.header.update('HDUCLAS3','COUNT','PHA data stored as counts (not count/s)')
       tbhdu1.header.update('BACKFILE',backfile2,'Background FITS file')
     elif fileversion == 2:
       tbhdu1.header.update('HDUCLAS2','NET','NET PHA Spectrum (background subtracted)')
       tbhdu1.header.update('HDUCLAS3','RATE','PHA data stored as count/s')
     tbhdu1.header.update('HDUVERS1','1.1.0','Version of format (OGIP memo OGIP-92-007a)')
     tbhdu1.header.update('CHANTYPE','PHA','Type of channel PHA/PI')
     tbhdu1.header.update('TLMIN1  ',1,'Lowest legal channel number')
     tbhdu1.header.update('TLMAX1',len(channel2),'Highest legal channel number')
     tbhdu1.header.update('POISSERR','F','Poissonian errors not applicable')
     tbhdu1.header.update('GROUPING',0,'No grouping of the data has been defined')
     tbhdu1.header.update('DETCHANS',len(channel2),'Total number of detector channels available')
     tbhdu1.header.update('AREASCAL',1,'Area scaling factor')
     tbhdu1.header.update('BACKSCAL',1,'Background scaling factor')
     tbhdu1.header.update('CORRSCAL',1,'Correlation scaling factor')
     tbhdu1.header.update('CORRFILE','NONE  ','Correlation FITS file')
     tbhdu1.header.update('RESPFILE','NONE','Redistribution matrix')
#   tbhdu1.header.update('RESPFILE',respfile,'Redistribution matrix')
     tbhdu1.header.update('ANCRFILE','NONE  ','Ancillary response')
#    tbhdu1.header.update('ANCRFILE','NONE  ','Ancillary response')
     tbhdu1.header.update('XFLT0001','NONE  ','XSPEC selection filter description')
     tbhdu1.header.update('CRPIX1  ','(1,'+str(len(channel2))+')','Channel binning of the CHANNEL column')
     tbhdu1.header.update('PHAVERSN','1992a ','OGIP memo number for file format')   
     tbhdu1.header.update('ZERODETX',zeroxy[0],'zeroth order position on image')
     tbhdu1.header.update('ZERODETY',zeroxy[1],'zeroth order position on image')   
     hdulist.append(tbhdu1)
     try:
        hdulist.writeto(outfile2nd,clobber=clobber)
     except:
        print "WARNING : NO OUTPUT FILE CREATED.  "+outfile2nd+" EXISTS and CLOBBER not set" 
        pass  
	
   if fileversion == 1: 
   #
   # ================= background PHA files ============================
   #
   #   first order background PHA file
   #
   # create primary header 
   #
       hdu0 = fits.PrimaryHDU()
       hdu0.header.update('CREATED','written by uvotio.py '+version)
       hdu0.header.update('DATE',str(now))
       hdu0.header.update('AUTHOR','NPM Kuin (UCL/MSSL)')
       hdu0.header.update('WHEELPOS',hdr['wheelpos'])
       hdu0.header.update('FILTER',hdr['filter'],comment='UVOT filter used')
       hdu0.header.update('TELESCOP','SWIFT   ',comment='Telescope (mission) name')
       hdu0.header.update('INSTRUME','UVOTA   ',comment='Instrument Name')
       hdulist=fits.HDUList([hdu0])
   #
   #  Main fits part
   #
       hdr0 = hdr.copy()
       hdr0.update('ORI_FILE',obsid+'+'+str(ext),'fileid and extension of extracted spectrum')
       hdr0.update('RA_X',ra,'RA of source extracted spectrum')
       hdr0.update('DEC_X',dec,'DEC of source extracted spectrum')
       hdr0.update('DETX_X',anker[0],'XDET position source anker in DET coord') 
       hdr0.update('DETY_X',anker[1],'YDET position source anker in DET coord')
       hdr0.update('POSX_AS',phx,'angle boresight in deg in DETX direction')
       hdr0.update('POSY_AS',phy,'angle boresight in deg in DETY direction')
       hdr0.update('SPEC_OFF',offset,'distance to spectrum from anker position (DETX_X,DETY_X)')
   #
   #  first extension: first order spectrum ; add extname everywhere
   #
       channel = back_first[0]
       bgcounts  = back_first[1]
       bgstat_err = back_first[2]
       bgquality = back_first[3]
       channel, bgcounts, bgstat_err, bgquality, aper1corr, expospec1 = back_first
       col11 = fits.Column(name='CHANNEL ',format='I',array=channel )
       col12 = fits.Column(name='COUNTS  ',format='I',array=bgcounts  ,unit='COUNTS')
       col13 = fits.Column(name='STAT_ERR',format='E',array=bgstat_err,unit='COUNTS')
       col14 = fits.Column(name='QUALITY ',format='I',array=bgquality )
       col15 = fits.Column(name='EXPOSURE',format='E',array=expospec1 ,unit='s' )
       cols1 = fits.ColDefs([col11,col12,col13,col14,col15])
       tbhdu1 = fits.new_table(cols1)
       tbhdu1.header.update('EXPID',hdr['expid'],'Exposure ID')
#   tbhdu1.header.update('EXTNAME','FIRST_ORDER_PHA_BACKGROUND','Name of this binary table extension')
       tbhdu1.header.update('EXTNAME','SPECTRUM','Name of this binary table extension')
       tbhdu1.header.update('FILETAG',filetag,'unique set id')
       tbhdu1.header.update('TELESCOP','SWIFT   ',comment='Telescope (mission) name')
       tbhdu1.header.update('INSTRUME','UVOTA   ',comment='Instrument Name')
   #tbhdu1.header.update('CMPOTHRS',hdr['CMPOTHRS'],'overflow of lossy compression algorith')
   #tbhdu1.header.update('CMPUTHRS',hdr['CMPUTHRS'],'underflow of lossy compression algorith')
   #tbhdu1.header.update('CMPCNTMN',hdr['CMPCNTMN'],'compression losses have occurred in the image')
       tbhdu1.header.update('TIMESYS ',hdr['timesys'],'time system')
       tbhdu1.header.update('MJDREFI ',hdr['mjdrefi'],'Reference MJD time integer part')
       tbhdu1.header.update('MJDREFF ',hdr['mjdreff'],'Reference MJD fractional part')
       tbhdu1.header.update('TIMEREF ',hdr['timeref'],'time reference barycentric/local')
       tbhdu1.header.update('TASSIGN ',hdr['tassign'],'time assigned by clock')
       tbhdu1.header.update('TIMEUNIT',hdr['timeunit'])
       tbhdu1.header.update('TIERRELA',hdr['TIERRELA'],'time relative error [s/s]')
       tbhdu1.header.update('TIERABSO',hdr['TIERABSO'],'timing precision in seconds')
       tbhdu1.header.update('TSTART',hdr['TSTART'])
       tbhdu1.header.update('TSTOP',hdr['TSTOP'])
       tbhdu1.header.update('DATE-OBS',hdr['DATE-OBS'])
       tbhdu1.header.update('DATE-END',hdr['DATE-END'])
       tbhdu1.header.update('CLOCKAPP',hdr['CLOCKAPP'],'if clock correction was applied')
       tbhdu1.header.update('TELAPSE',hdr['TELAPSE'],'Tstop - Tstart')
       tbhdu1.header.update('EXPOSURE',hdr['EXPOSURE'],'Total exposure, with all known corrections')
       tbhdu1.header.update('DEADC',hdr['DEADC'],'dead time correction')
       tbhdu1.header.update('FRAMTIME',hdr['FRAMTIME'],'frame exposure time')
       tbhdu1.header.update('DETNAM',hdr['DETNAM'])
       tbhdu1.header.update('FILTER',hdr['FILTER'])
       tbhdu1.header.update('OBS_ID',hdr['OBS_ID'],'observation id')
       tbhdu1.header.update('TARG_ID',hdr['TARG_ID'],'Target ID')
   #tbhdu1.header.update('SEQ_NUM',hdr['SEQ_NUM'])
       tbhdu1.header.update('EQUINOX',hdr['EQUINOX'])
       tbhdu1.header.update('RADECSYS',hdr['RADECSYS'])
       tbhdu1.header.update('WHEELPOS',hdr['WHEELPOS'],'filterweel position')
       tbhdu1.header.update('SPECTORD',1,'spectral order')
       try:
          tbhdu1.header.update('BLOCLOSS',hdr['BLOCLOSS'],'[s] Exposure time under BLOCKED filter')
          tbhdu1.header.update('STALLOSS',hdr['STALLOSS'],'[s] Est DPU stalling time loss')
          tbhdu1.header.update('TOSSLOSS',hdr['TOSSLOSS'],'[s] Est Shift&Toss time loss')
          tbhdu1.header.update('MOD8CORR',hdr['MOD8CORR'],'Was MOD8 correction applied')
          tbhdu1.header.update('FLATCORR',hdr['FLATCORR'],'was flat field correction applied')
       except:
          pass   
       
       tbhdu1.header.update('ASPCORR','GAUSSIAN','Aspect correction method')
       tbhdu1.header.update('HDUCLASS','OGIP','format attemts to follow OGIP standard')
       tbhdu1.header.update('HDUCLAS1','SPECTRUM','PHA dataset (OGIP memo OGIP-92-007')
       tbhdu1.header.update('HDUCLAS2','TOTAL','Gross PHA Spectrum (source + background)')
       tbhdu1.header.update('HDUCLAS3','COUNT','PHA data stored as counts (not count/s)')
       tbhdu1.header.update('HDUVERS1','1.1.0','Version of format (OGIP memo OGIP-92-007a)')
       tbhdu1.header.update('CHANTYPE','PI','Type of channel PHA/PI')
       tbhdu1.header.update('TLMIN1  ',1,'Lowest legal channel number')
       tbhdu1.header.update('TLMAX1',len(channel),'Highest legal channel number')
       tbhdu1.header.update('POISSERR',False,'Poissonian errors not applicable')
       tbhdu1.header.update('GROUPING',0,'No grouping of the data has been defined')
       tbhdu1.header.update('DETCHANS',len(channel),'Total number of detector channels available')
       tbhdu1.header.update('AREASCAL',1,'Area scaling factor')
       tbhdu1.header.update('BACKSCAL',1,'Background scaling factor')
       tbhdu1.header.update('CORRSCAL',1,'Correlation scaling factor')
       tbhdu1.header.update('BACKFILE','NONE','Background FITS file')
       tbhdu1.header.update('CORRFILE','NONE  ','Correlation FITS file')
       tbhdu1.header.update('RESPFILE','NONE','Redistribution matrix')
       tbhdu1.header.update('ANCRFILE','NONE  ','Ancillary response')
       tbhdu1.header.update('XFLT0001','NONE  ','XSPEC selection filter description')
       tbhdu1.header.update('CRPIX1  ','(1,'+str(len(channel))+')','Channel binning of the CHANNEL column')
       tbhdu1.header.update('PHAVERSN','1992a ','OGIP memo number for file format')   
       tbhdu1.header.update('ZERODETX',zeroxy[0],'zeroth order position on image')
       tbhdu1.header.update('ZERODETY',zeroxy[1],'zeroth order position on image')   
       hdulist.append(tbhdu1)
       try:
          hdulist.writeto(backfile1,clobber=clobber)
       except:
            print "WARNING : NO OUTPUT FILE CREATED.  "+backfile1+" EXISTS and CLOBBER not set" 
            pass  
   #      
   #   second order background PHA file
   #
       if back_second != None:
      # create primary header 
      #
          hdu0 = fits.PrimaryHDU()
          hdu0.header.update('CREATED','written by uvotio.py '+version)
          hdu0.header.update('DATE',str(now))
          hdu0.header.update('AUTHOR','NPM Kuin (UCL/MSSL)')
          hdu0.header.update('WHEELPOS',hdr['wheelpos'])
          hdu0.header.update('FILTER',hdr['filter'],comment='UVOT filter used')
          hdu0.header.update('TELESCOP','SWIFT   ',comment='Telescope (mission) name')
          hdu0.header.update('INSTRUME','UVOTA   ',comment='Instrument Name')
          hdulist=fits.HDUList([hdu0])
      #
      #  Main fits part
      #
          hdr0 = hdr.copy()
          hdr0.update('ORI_FILE',obsid+'+'+str(ext),'fileid and extension of extracted spectrum')
          hdr0.update('RA_X',ra,'RA of source extracted spectrum')
          hdr0.update('DEC_X',dec,'DEC of source extracted spectrum')
          hdr0.update('DETX_X',anker[0],'XDET position source anker in DET coord') 
          hdr0.update('DETY_X',anker[1],'YDET position source anker in DET coord')
          hdr0.update('POSX_AS',phx,'angle boresight in deg in DETX direction')
          hdr0.update('POSY_AS',phy,'angle boresight in deg in DETY direction')
          hdr0.update('SPEC_OFF',offset,'distance to spectrum from anker position (DETX_X,DETY_X)')
      #
      #  first extension: first order spectrum ; add extname everywhere
      #
          channel = back_second[0]
          bgcounts  = back_second[1]
          bgstat_err = back_second[2]
          bgquality = back_second[3]
          channel, bgcounts, bgstat_err, bgquality, aper2corr, expospec2 = back_second
          col11 = fits.Column(name='CHANNEL ',format='I',array=channel )
          col12 = fits.Column(name='COUNTS  ',format='I',array=bgcounts  ,unit='COUNTS')
          col13 = fits.Column(name='STAT_ERR',format='E',array=bgstat_err,unit='COUNTS')
          col14 = fits.Column(name='QUALITY ',format='I',array=bgquality   )
          col15 = fits.Column(name='EXPOSURE',format='E',array=expospec2 ,unit='s' )
          cols1 = fits.ColDefs([col11,col12,col13,col14,col15])
          tbhdu1 = fits.new_table(cols1)
          tbhdu1.header.update('EXPID',hdr['expid'],'Exposure ID')
#   tbhdu1.header.update('EXTNAME','FIRST_ORDER_PHA_BACKGROUND','Name of this binary table extension')
          tbhdu1.header.update('EXTNAME','SPECTRUM','Name of this binary table extension')
          tbhdu1.header.update('FILETAG',filetag,'unique set id')
          tbhdu1.header.update('TELESCOP','SWIFT   ',comment='Telescope (mission) name')
          tbhdu1.header.update('INSTRUME','UVOTA   ',comment='Instrument Name')
          tbhdu1.header.update('TIMESYS ',hdr['timesys'],'time system')
          tbhdu1.header.update('MJDREFI ',hdr['mjdrefi'],'Reference MJD time integer part')
          tbhdu1.header.update('MJDREFF ',hdr['mjdreff'],'Reference MJD fractional part')
          tbhdu1.header.update('TIMEREF ',hdr['timeref'],'time reference barycentric/local')
          tbhdu1.header.update('TASSIGN ',hdr['tassign'],'time assigned by clock')
          tbhdu1.header.update('TIMEUNIT',hdr['timeunit'])
          tbhdu1.header.update('TIERRELA',hdr['TIERRELA'],'time relative error [s/s]')
          tbhdu1.header.update('TIERABSO',hdr['TIERABSO'],'timing precision in seconds')
          tbhdu1.header.update('TSTART',hdr['TSTART'])
          tbhdu1.header.update('TSTOP',hdr['TSTOP'])
          tbhdu1.header.update('DATE-OBS',hdr['DATE-OBS'])
          tbhdu1.header.update('DATE-END',hdr['DATE-END'])
          tbhdu1.header.update('CLOCKAPP',hdr['CLOCKAPP'],'if clock correction was applied')
          tbhdu1.header.update('TELAPSE',hdr['TELAPSE'],'Tstop - Tstart')
          tbhdu1.header.update('EXPOSURE',hdr['EXPOSURE'],'Total exposure, with all known corrections')
          tbhdu1.header.update('DEADC',hdr['DEADC'],'dead time correction')
          tbhdu1.header.update('FRAMTIME',hdr['FRAMTIME'],'frame exposure time')
          tbhdu1.header.update('DETNAM',hdr['DETNAM'])
          tbhdu1.header.update('FILTER',hdr['FILTER'])
          tbhdu1.header.update('OBS_ID',hdr['OBS_ID'],'observation id')
          tbhdu1.header.update('TARG_ID',hdr['TARG_ID'],'Target ID')
   #tbhdu1.header.update('SEQ_NUM',hdr['SEQ_NUM'])
          tbhdu1.header.update('EQUINOX',hdr['EQUINOX'])
          tbhdu1.header.update('RADECSYS',hdr['RADECSYS'])
          tbhdu1.header.update('WHEELPOS',hdr['WHEELPOS'],'filterweel position')
          tbhdu1.header.update('SPECTORD',2,'spectral order')
          try:
             tbhdu1.header.update('BLOCLOSS',hdr['BLOCLOSS'],'[s] Exposure time under BLOCKED filter')
             tbhdu1.header.update('STALLOSS',hdr['STALLOSS'],'[s] Est DPU stalling time loss')
             tbhdu1.header.update('TOSSLOSS',hdr['TOSSLOSS'],'[s] Est Shift&Toss time loss')
             tbhdu1.header.update('MOD8CORR',hdr['MOD8CORR'],'Was MOD8 correction applied')
             tbhdu1.header.update('FLATCORR',hdr['FLATCORR'],'was flat field correction applied')
          except:
             pass   
          tbhdu1.header.update('ASPCORR',hdr['ASPCORR'],'Aspect correction method')
          tbhdu1.header.update('HDUCLASS','OGIP','format attemts to follow OGIP standard')
          tbhdu1.header.update('HDUCLAS1','SPECTRUM','PHA dataset (OGIP memo OGIP-92-007')
          tbhdu1.header.update('HDUCLAS2','TOTAL','Gross PHA Spectrum (source + background)')
          tbhdu1.header.update('HDUCLAS3','COUNT','PHA data stored as counts (not count/s)')
          tbhdu1.header.update('HDUVERS1','1.1.0','Version of format (OGIP memo OGIP-92-007a)')
          tbhdu1.header.update('CHANTYPE','PI','Type of channel PHA/PI')
          tbhdu1.header.update('TLMIN1  ',1,'Lowest legal channel number')
          tbhdu1.header.update('TLMAX1',len(channel),'Highest legal channel number')
          tbhdu1.header.update('POISSERR',False,'Poissonian errors not applicable')
          tbhdu1.header.update('GROUPING',0,'No grouping of the data has been defined')
          tbhdu1.header.update('DETCHANS',len(channel),'Total number of detector channels available')
          tbhdu1.header.update('AREASCAL',1,'Area scaling factor')
          tbhdu1.header.update('BACKSCAL',1,'Background scaling factor')
          tbhdu1.header.update('CORRSCAL',1,'Correlation scaling factor')
          tbhdu1.header.update('BACKFILE','NONE','Background FITS file')
          tbhdu1.header.update('CORRFILE','NONE  ','Correlation FITS file')
          tbhdu1.header.update('RESPFILE','NONE','Redistribution matrix')
          tbhdu1.header.update('ANCRFILE','NONE  ','Ancillary response')
          tbhdu1.header.update('XFLT0001','NONE  ','XSPEC selection filter description')
          tbhdu1.header.update('CRPIX1  ','(1,'+str(len(channel))+')','Channel binning of the CHANNEL column')
          tbhdu1.header.update('PHAVERSN','1992a ','OGIP memo number for file format')   
          tbhdu1.header.update('ZERODETX',zeroxy[0],'zeroth order position on image')
          tbhdu1.header.update('ZERODETY',zeroxy[1],'zeroth order position on image')   
          hdulist.append(tbhdu1)
          try:
             hdulist.writeto(backfile2,clobber=clobber)
          except:
             print "WARNING : NO OUTPUT FILE CREATED.  "+backfile2+" EXISTS and CLOBBER not set" 
             pass  



def wr_spec(ra,dec,obsid,ext,hdr,anker,phx,phy,dis,wave,sprate,bgrate,bg1rate,bg2rate,offset,ank_c,extimg, C_1,
   quality=None,history=None,chatter=1,clobber=False,interactive=False, fileout=None):
   '''helper call to OldwriteSpectrum 
      fileout to force replacement of OBSID in filenames'''
   Y =( hdr,anker,phx,phy,dis,wave,sprate,bgrate,bg1rate,bg2rate,offset,ank_c,extimg, C_1)
   return OldwriteSpectrum( ra,dec,obsid,ext, Y, mode=1, quality=quality, history=history, chatter=chatter, \
      clobber=clobber,interactive=interactive,fileout=fileout)  


def OldwriteSpectrum(ra,dec,filestub,ext, Y, mode=1, quality=None,  
    updateRMF=False, interactive=False, fileout=None, 
    arfdbfile=None, arf2file=None, wr_outfile=True, 
    history=None,chatter=1,clobber=False):
   ''' write a standard UVOT output file

       main header: edited copy of grism det image, history?, RA DEC  extracted source; 
                 anchor point, input angles, ank_c, angle, 
		 offset (distance anchor to spectrum) in arcsec
   
       the first extension is named  SPECTRUM (future: 'FIRST_ORDER_PHA_SPECTRUM') and 
       contains the standard input for XSPEC
       
       the second extension is named CALSPEC  (future: 'FIRST_ORDER_NET_SPECTRUM') 
       contains the standard input for IDL/PYTHON with 
       pixelno(relative to anchor), wave(nm), net_count_rate(s-1 pix-1),
       left_background_rate(s-1 pix-1), right_background_rate (s-1 pix-1),  
       flux(erg cm-2 s-1 A-1), flux_err(erg cm-2 s-1 A-1), quality flag, aper1corr
       second order: wave2, net2rate, bg2rate, flux2, flux2err, qual2, aper2corr 
       
       The third extension named 'SPECTRUM_IMAGE' contains the image of the total spectrum
       
       The fourth extension may exist for summed images and contains the exposure map 
         

  revised 2011-12-10 NPM Kuin, superseded 
         
  '''	    
   try:
      from astropy.io import fits
   except:   
      import pyfits as fits
   import datetime
   import numpy as np
   from scipy import interpolate
   import os
   from pylab import polyval
   import uvotgetspec

   now = datetime.date.today()
   rnu = now.day*1.2+now.month*0.99+now.year*0.3
   version = '111020'
   h_planck = 6.626e-27  # erg/s
   lightspeed = 2.9979e10  # cm/sec
   h_c_ang = h_planck * lightspeed * 1e8 # ergs-Angstrom
   ch = chatter
   
   wave2 = None
   sp2netcnt = None
   bg2netcnt = None
   wave3 = None
   quality = None
   qflags = uvotgetspec.quality_flags()
   
      
   ################# prepare data for writing to output file
   
   if mode == 0:	    # uvotcal output  === probably obsolete
      ( (dis, spnet, angle, anker, coef_zmx, pix_zmx, wav_zmx), 
            (anker_uvw1,anker_as, anker_field,ank_c),
            (bg, bg1, bg2, extimg, spimg, spnetimg, offset) ,
            (C_zero,C_1,C_2,C_3,C_min1), 
            (xpix,ypix, zmxdis,zmxwav, wave,theta), 
            (img, xplines, yplines, dislines, lines), hdr) = Y 
      a1 = ank_c[1]-370
      a2 = ank_c[1]+1200
      # temporary hack - need to pass actual total BG counts used for spectrum
      spwidth = 13*1.5
      if hdr['wheelpos'] > 500: spwidth = 7*1.5
      if hdr['wheelpos'] > 500: a2 = ank_c[1]+500
      if a2 >= len(dis): a2=len(dis) -1
      aa = range(a1,a2)
      dis = dis[aa]
      counts = (spnet+bg)[aa] 
      bkg = bg[aa]
      exposure=hdr['exposure']
      wave =  polyval(C_1, dis)  
      NW = len(wave)
      ax = range(NW-1) ; ax1=range(1,NW) 
      binwidth = 1.0*(wave[ax1]-wave[ax])
      sprate = spnet[aa]/exposure
      bgrate = bg[aa]*spwidth/exposure
      bg1rate = bg1[aa]*spwidth/exposure
      bg2rate = bg2[aa]*spwidth/exposure
      phx, phy = anker_field
      
   elif mode == 2:        # optimal extraction mode === needs to be updated Y? defs, with apercorr, second order
      (Y0,Y1,Y2,Y3,Y4) = Y

      (specfile, lfilt1_, lfilt1_ext_, lfilt2_, lfilt2_ext_, attfile), (method), \
        (Xphi, Yphi, date1), (dist12, ankerimg, ZOpos) = Y0
      
      ( (dis,spnet,angle,anker,anker2,anker_field,ank_c), (bg,bg1,bg2,extimg,spimg,spnetimg,offset), 
           (C_1,C_2,img),  hdr,m1,m2,aa,wav1 )	= Y1
	   
      ((present0,present1,present2,present3),(q0,q1,q2,q3), \
              (y0,dlim0L,dlim0U,sig0coef,sp_zeroth),(y1,dlim1L,dlim1U,sig1coef,sp_first),\
              (y2,dlim2L,dlim2U,sig2coef,sp_second),(y3,dlim3L,dlim3U,sig3coef,sp_third),\
	      (x,xstart,xend,sp_all,qquality) ), (coef0,coef1,coef2,coef3), \
	      (bg_zeroth,bg_first,bg_second,bg_third), (borderup,borderdown), apercorr,expospec  = Y2
	      
      opcounts, variance, borderup, borderdown, (fractions,cnts,vars,newsigmas) = Y3 

      wav2p, dis2p, flux2p, qual2p, dist12p = Y4[0]
      
      bkg = bg_first[q1[0]] 
      counts = (opcounts[1,q1[0]]+bkg)
      exposure=hdr['exposure']
      wave =  polyval(C_1, x[q1[0]])  
      # limit range according to wavelenghts
      qwave = np.where( (wave > 1650.0) & (wave < 6800) )
      dis = x[q1[0]][qwave]
      bkg = bkg[qwave]
      counts = counts[qwave]
      wave = wave[qwave]
      NW = len(wave)
      aa = np.arange(NW)
      ax = range(NW-1) ; ax1=range(1,NW) 
      binwidth = 1.0*(wave[ax1]-wave[ax])
      sprate = (opcounts[1,q1[0]]/exposure)[qwave]
      bgrate = bkg/exposure
      bg1rate = (bg1[q1[0]]*2*np.polyval(sig1coef,q1[0])/exposure)[qwave]
      bg2rate = (bg2[q1[0]]*2*np.polyval(sig1coef,q1[0])/exposure)[qwave]
      phx, phy = anker_field
      #wave2 = polyval(C_2, x[q2[0]]+dist12)
      #sp2netcnt = counts[2,q2[0]]
      #bg2cnt    = bg_second[q2[0]]
      #sp2counts = sp2netcnt + bg2cnt
               	    	     
   else:                 # straight slit only
      ( hdr,anker,phx,phy,dis,wave,sprate,bgrate,bg1rate,bg2rate,offset,ank_c,extimg, C_1) = Y
      a1 = int(ank_c[1])-370
      a2 = int(ank_c[1])+1200
      # temporary hack - need to pass actual total BG counts used for spectrum
      spwidth = 13*1.5
      if hdr['wheelpos'] > 500: spwidth = 7*1.5
      if hdr['wheelpos'] > 500: a2 = ank_c[1]+500
      if a2 >= len(dis): a2=len(dis) -1
      aa  = range(a1,a2)
      dis = dis[aa]
      sprate = sprate[aa]
      bgrate = bgrate[aa]*spwidth
      bg1rate = bg1rate[aa]*spwidth
      bg2rate = bg2rate[aa]*spwidth
      exposure=hdr['exposure']
      counts = ((sprate+bgrate)*exposure)
      bkg = bgrate*exposure
      wave = wave[aa]
      NW = len(wave)
      ax = range(NW-1) ; ax1=range(1,NW) 
      binwidth = 1.0*(wave[ax1]-wave[ax])

            
   # ensure that offset is a scalar  
    
   try:
      offset = offset[0]
   except:
      pass
             
   ############### FILES   Define the input/ output file names  
   
   if arfdbfile == None: 
      arf_file_passed = False
   else:   
      arf_file_passed = True
      
   obsid = filestub  
   if fileout != None: 
      obsid = fileout 
      if chatter > 2: print "output file name base is now:",obsid
   if hdr['wheelpos'] == 200:
      outfile = obsid+'ugu_'+str(ext)+'.pha'
      backfile = obsid+'ugu_'+str(ext)+'_back.pha'
      respfile = obsid+'ugu_'+str(ext)+'.arf'
      if arfdbfile == None: arfdbfile = 'swugu0200_20041120v101.arf'
      rmffile = obsid+'ugu_'+str(ext)+'.rmf'
      EXTNAME='SPECRESPUGRISM200'
   elif hdr['wheelpos'] == 160:
      outfile = obsid+'ugu_'+str(ext)+'.pha'
      backfile = obsid+'ugu_'+str(ext)+'_back.pha'
      respfile = obsid+'ugu_'+str(ext)+'.arf'
      if arfdbfile == None: arfdbfile = 'swugu0160_20041120v101.arf'
      rmffile = obsid+'ugu_'+str(ext)+'.rmf'
      EXTNAME='SPECRESPUGRISM160'
   elif hdr['wheelpos'] == 1000:
      outfile = obsid+'ugv_'+str(ext)+'.pha'
      backfile = obsid+'ugv_'+str(ext)+'_back.pha'
      respfile = obsid+'ugv_'+str(ext)+'.arf'
      if arfdbfile == None: arfdbfile = 'swugv1000_20041120v101.arf'
      rmffile = obsid+'ugv_'+str(ext)+'.rmf'
      EXTNAME='SPECRESPVGRISM1000'
   elif hdr['wheelpos'] == 955: 
      outfile = obsid+'ugv_'+str(ext)+'.pha'
      backfile = obsid+'ugv_'+str(ext)+'_back.pha'
      respfile = obsid+'ugv_'+str(ext)+'.arf'
      if arfdbfile == None: arfdbfile = 'swugv0955_20041120v101.arf'
      rmffile = obsid+'ugv_'+str(ext)+'.rmf'
      EXTNAME='SPECRESPVGRISM955'
   else:   
      print "FATAL: exposure header does not have filterwheel position encoded"
      return 
   obsid = filestub   
      
   # test for presence of outfile and clobber not set
   
   if clobber == False:
      if os.access(outfile,os.F_OK) ^ os.access(backfile,os.F_OK) ^ os.access(outfile,os.F_OK) ^ os.access(backfile,os.F_OK):
         print 'Error: output file already present. '
	 if interactive:
   	    answer =  raw_input('       DO YOU WANT TO REWRITE THE OUTPUT FILES (answer yes/NO)? ')
	    if len(answer1) < 1:  answer = 'NO'
	    answer = answer.upper()
	    if (answer == 'Y') ^ (answer == 'YES'):  clobber = True	        
         if clobber == False: return 
   filetag = obsid+'_'+str(hdr['wheelpos'])+str(rnu)

   #
   #  get spectral response [cm**2] from the ARF file 
   #
   specresp1func = SpecResp(hdr['wheelpos'], 1, )
   if present2:   
      specresp2func = SpecResp(hdr['wheelpos'], 2, )          
   
   # sort out the inputs and prepare the arrays to write.
       
   if mode == 0:	    
      channel = range(len(aa))
      rc      = range(len(aa)) ; rc.reverse()
      channel = np.array(channel)
      stat_err = (np.sqrt(counts))[rc]
      counts  = counts[rc]
      if quality == None:
         quality = np.zeros(len(aa))  # see definition from OGIP
         q = np.where(counts < 0)
	 if q[0].shape != (0,) :
           if q[0][0] != -1:
              stat_err[q] = 0.
              quality[q] = 5	      	      
      else:
         quality = (quality[aa])[rc]	    
      
      cpp2fpa = 1     # count2flux(dis,wave,wheelpos,anker) count rate per pixel to flux per angstroem
      #flux     = cpp2fpa*sprate
      sprate = spnet[aa]/exposure
      sprate_err = np.sqrt(spnet[aa]+2*bkg)/exposure  # poisson error net rate
      dlam_per_pix = wave*0. ; dlam_per_pix[ax]=binwidth ; dlam_per_pix[-1]=binwidth[-1] 
      hnu = h_c_ang/(wave)
      flux = hnu*sprate/specrespfunc(wave)/dlam_per_pix   # [erg/s/cm2/angstrom]
      flux_err = hnu*sprate_err/specrespfunc(wave)/dlam_per_pix 
          	    	     
   else:
      channel = range(len(wave)) 
      rc      = range(len(wave))
      rc.reverse()     
      if present2:
         channel2 = range(len(wave2))
	 rc2      = range(len(wave2))
	 rc2.reverse()
       
      channel = np.array(channel)
      stat_err = (np.sqrt(counts))[rc]  # poisson error on counts (no background)
      counts  = counts[rc]
      if present2:
         channel2 = np.array(channel2)
	 stat_err = (np.sqrt(sp2counts))[rc2]
	 sp2counts = sp2counts[rc2]
	 
      if quality == None: 
         quality = np.zeros(len(aa))  # see definition from OGIP
	 quality2 = quality
      else:
         quality = quality[rc]
	 quality[np.where(quality != 0)] = 2 # see definition from OGIP	 
	 quality2 = quality
	 
      q = np.where(counts < 0)
      if q[0].shape != (0,) :
         if q[0][0] != -1:
            stat_err[q] = 0.
            quality[q] = qflags['bad']
         
      cpp2fpa = 1     # count2flux(dis,wave,wheelpos,anker) count rate per pixel to flux per angstroem
      #flux     = cpp2fpa*sprate
      if mode != 1: 
         sprate_err = (np.sqrt(sprate+2*bgrate))/exposure  # poisson error on net counts
         dlam_per_pix = wave*0. ; dlam_per_pix[ax]=binwidth ; dlam_per_pix[-1]=binwidth[-1] 
      else:
         sprate_err = sp1rate_err
	 dlam_per_pix = binwidth
	 	 
      hnu = h_c_ang/wave
      flux = hnu*sprate/specrespfunc(wave)/dlam_per_pix   # [erg/s/cm2/angstrom]
      flux_err = hnu*sprate_err/specrespfunc(wave)/dlam_per_pix 

      if present2:
         if arf2file != None:
            flux2 = hnu*sp2rate/specrespfunc2(wave)/dlam_per_pix2   # [erg/s/cm2/angstrom]
            flux2_err = hnu*sp2rate_err/specrespfunc2(wave)/dlam_per_pix2 
	 else:
	    flux2 = sp2rate*0
	    flux2_err = sp2rate_err*0 
	      
      if Y4 != None:
         if arf2file != None:
            flux2p = hnu*rate2p/specrespfunc2(wave2p)/dlam_per_pix2
	 else:
	    flux2p = rate2p*0
	       
   spectrum_first = (channel, counts, star_err, quality, np.ones(len(channel)) ) 
   back_first = (channel, bkg, sqrt(bkg), quality, np.ones(len(channel)) )
   calspec_first = (dis,wave,sprate,bg1rate,bg2rate,flux,flux_err,quality, np.ones(len(channel)) )
        
   if wr_outfile:writeSpectrum_ (ra,dec,obsid,ext,hdr,anker,phx,phy,offset, ank_c, exposure,  
   history, spectrum_first, back_first, calspec_first, extimg, outfile, 
   backfile, rmffile, outfile2=None, backfile2=None, rmffile2=None, expmap=None,   
   spectrum_second = None, back_second = None, calspec_second=None, present2=False, 
   clobber=False, chatter=chatter )
   # dothefile(ra,dec,obsid,ext,hdr,anker,phx,phy,offset,extimg,ank_c,\
   #   channel,counts,stat_err,quality,filetag,backfile,outfile,rmffile,\
   #   dis,wave,sprate,bg1rate,bg2rate,flux,flux_err, aper1corr, dis2, flux2,flux2_err, qual2\
   #   ,x,q,present2,wave2,sp2counts,sp2netcnt,exposure,sp2rate,sp2rate_err,bg_2rate,wave3,\
   #   bkg,rc,aa,C_1, pext,ax,ax1, specrespfunc,updateRMF,EXTNAME,history, clobber)      
   return flux, flux_err


def updateResponseMatrix(rmffile, C_1, clobber=True, lsffile='zemaxlsf', chatter=0):
   """ modify the response matrix lineprofiles 
       using the zemax model prediction from zemaxlsf.fit
       In addition the zemax profile is broadened by the instrumental 
       broadening of 2.7 pixels. 
       
       Parameters
       ----------
       rmffile : path, str
         The rmffile is updated by default
       
       C_1: ndarray
         The dispersion C_1 is used to convert pixels to  angstroms.

       kwargs : dict
       
        - *lsffile* : path
	
          The lsffile is in the $UVOTPY/calfiles directory
        
	- *clobber* : bool
        
	  overwrite output.
        
	- *chatter* : int
        
	  verbosity
       
       Returns
       -------
       writes RMF file
       
       Notes
       -----
       2011-09-02 Initial version NPMK (UCL/MSSL) 
       2011-12-16 Uncorrected error: LSF is inverted in energy scale.
       
       Notes
       -----
       The same algorithm was implemented in the write_rmf_file() routine which does not
       need the input rmf file produced by the "Ftool" *rmfgen*.
   """
   try:
      from astropy.io import fits
   except:   
      import pyfits as fits
   import numpy as np
   import os
   from scipy.ndimage import convolve
   #from convolve import boxcar
   import uvotio
   import uvotgetspec
   from scipy import interpolate
   
   #  instrument_fwhm = 2.7/0.54 # pix
   ww = uvotgetspec.singlegaussian(np.arange(-12,12),1.0,0.,2.7/0.54)
   ww = ww/ww.sum()
   
   UVOTPY = os.getenv('UVOTPY')
   if UVOTPY == '': 
      print 'The UVOTPY environment variable has not been set'

   lsffile = fits.open(  UVOTPY+'/calfiles/zemaxlsf.fit' )  
   tlsf = lsffile[1].data
   lsfchan = tlsf.field('channel')[0:15]   # energy value 
   lsfwav = uvotio.kev2angstrom(lsfchan)
   epix    = tlsf.field('epix')[0,:]
   lsfdata = tlsf.field('lsf')[:15,:]
   lsffile.close()
      
   hdulist = fits.open( rmffile, mode='update' )
   hdu = hdulist['MATRIX']
   matrix = hdu.data.field('MATRIX')
   n5 = len(matrix)
   k5 = len(matrix[0])
   resp = np.zeros(n5)
   for i in range(n5): resp[i] = matrix[i].sum()
   energ_lo = hdu.data.field('energ_lo')
   energ_hi = hdu.data.field('energ_hi')
   n_grp    = hdu.data.field('n_grp')
   f_chan   = hdu.data.field('f_chan')
   n_chan   = hdu.data.field('n_chan')
   
   e_mid = 0.5*(energ_lo+energ_hi) 
   wav = kev2angstrom( e_mid )
   
   wav_lo = kev2angstrom( energ_hi )
   wav_hi = kev2angstrom( energ_lo )

   dis = np.arange(-370,1150)
   C_1inv = uvotgetspec.polyinverse(C_1, dis)
      
   d_lo = np.polyval(C_1inv,wav_lo)
   d_hi = np.polyval(C_1inv,wav_hi)
   
   for k in range(len(e_mid)):
      if ((e_mid[k] != 0) | (np.isfinite(resp[k]))):
         #  find index e in lsfchan and interpolate lsf
         w = wav[k]
         j = lsfwav.searchsorted(w)
         if j == 0: 
            lsf = lsfdata[0,:]
         elif ((j > 0) & (j < 15) ):
            e1 = lsfchan[j-1]
 	    e2 = lsfchan[j]
 	    frac = (e_mid[k]-e1)/(e2-e1)
            lsf1 = lsfdata[j-1,:]
	    lsf2 = lsfdata[j,:]
	    lsf = (1-frac) * lsf1 + frac * lsf2	 	 
         else:
            # j = 15
	    lsf = lsfdata[14,:]

         # convolution lsf with instrument_fwhm and multiply with response 
         lsf_con = convolve(lsf,ww.copy(),)
      
         # assign wave to lsf array relative to w at index k in matrix (since on diagonal)   
         # rescale lsfcon from pixels to channels 

         d   = np.arange(-79,79) + int(np.polyval(C_1inv, w))
         wave = np.polyval(C_1,d)  
         ener = uvotio.angstrom2kev(wave)
         # now each pixel has a wave, energy(keV) and lsf_con value 

         # new array to fill 
         lsfnew   = np.zeros(k5)
	 ener_ = list(ener)
	 lsf_con_ = list(lsf_con)
	 ener_.reverse()
	 lsf_con_.reverse()
	 # now we have ener as an increasing function - if not, the function fails.
         inter = interpolate.interp1d(ener_, lsf_con_,bounds_error=False,fill_value=0.0)
         for i in range(k5):
	    lsfnew[i] = inter( e_mid[i]) 
	 lsfnew_norm = lsfnew.sum() 
	 if (np.isnan(lsfnew_norm)) | (lsfnew_norm <= 0.0): lsfnew_norm = 5.0e9  
         lsfnew = ( lsfnew / lsfnew_norm) * resp[k]	 
         matrix[k] =  lsfnew  
	 if (k/48*48 == k): print 'processed up to row --> ',k
          
   #hdulist.update()  
   if chatter > 1: print "updating the LSF in the response file now"     
   hdu.header.update('HISTORY','Updated LSF by uvotgetspec.uvotio.updateResponseMatrix()')
   hdulist.update_tbhdu()
   print 'updated'
   hdulist.verify()
   hdulist.flush()
   hdulist.close()

def write_rmf_file (rmffilename, wave, wheelpos, disp, anchor=[1000,1000], 
    effarea1=None, effarea2=None, chatter=1, clobber=False  ):
   '''
   Write the RMF file for the first order spectrum
   
   Parameters
   ----------
      rmffile : path, str
         file name output file
	 
      wave : ndarray
         wavelengths of the bins
	 
      wheelpos : int
         filter wheel position 	 
	 
      disp : ndarray
         dispersion coefficients 
      
      kwargs : dict
      ------
       - **chatter** : int
         verbosity
	 
       - **clobber** : bool
         if true overwrite output file if it already exists
	 
   Returns
   -------
   Writes the RMF file 
   
   Notes
   -----
   The line spread function from the uv grism at default position is
   currently used for all computations. Since the RMF file encodes also
   the effective area, this version presumes given anchor position. 
   
   2014-02-27 code cleaned up. Speed depends on number of points
   
   ''' 
#
#   *** needs to be sampled better to properly to it speed up ***	 		
#
   try:
      from astropy.io import fits
   except:   
      import pyfits as fits
   import numpy as np
   import os
   from scipy.ndimage import convolve
   import uvotio
   import uvotgetspec
   from scipy import interpolate
   import datetime
   
   version = '140227'
   now = datetime.date.today()
   datestring = now.isoformat()[0:4]+now.isoformat()[5:7]+now.isoformat()[8:10]
   if chatter > 0: print "computing RMF file. This takes ~40 sec or more on older machines"
   
   spectralorder = 1  # not possible for second order yet
   
   # get the effective area for the grism mode, anchor position and order at each wavelength
   if effarea1 != None:
      if len(effarea1 == 2):
          hdu, fnorm = effarea1          
          w = 0.5*(hdu.data['WAVE_MIN']+hdu.data['WAVE_MAX']) 
	  fnorm = fnorm(w)    
      else: 
          hdu = effarea1
          w = 0.5*(hdu.data['WAVE_MIN']+hdu.data['WAVE_MAX']) 
	  fnorm = 1    
   else:
       hdu, fnorm = uvotio.readFluxCalFile(wheelpos,anchor=anchor,spectralorder=spectralorder,chatter=chatter)
       w = 0.5*(hdu.data['WAVE_MIN']+hdu.data['WAVE_MAX'])   
       fnorm = fnorm(w)  
   r = hdu.data['SPECRESP']
   ii = range(len(w)-1,-1,-1)
   w = w[ii]
   r = r[ii]
   r = r * fnorm
   specrespfunc = interpolate.interp1d( w, r, bounds_error=False, fill_value=0.0 )
   resp = specrespfunc(wave) 
   hdu = ""           # cleanup
   fnorm = ""
   specrespfunc = ""
   
   NN = len(wave)  # number of channels
   if NN < 20:
      print "write_rmf_file: not enough valid data points. No rmf file written for wheelpos=",wheelpos,", order=",spectralorder
      return
   
   #iNL = range(0,5) 
   #for k in range(8,NN-9,NN/40): iNL.append(k) # index of sample of channels (every hundred Angstrom)
   #for k in range(NN-6,NN,1): iNL.append(k)
   #iNL = np.array(iNL,dtype=int)
   iNL = np.arange(NN,dtype=int)
   NL = len(iNL)   # number of sample channels
   
   aa = uvotgetspec.pix_from_wave(disp, wave, spectralorder=spectralorder)  # slow !
   tck_Cinv = interpolate.splrep(wave,aa,)  # B-spline coefficients to look up pixel position (wave)
   
   channel = range(NN)
   aarev   = range(NN-1,-1,-1)
   channel = np.array(channel) + 1
   # wavelengths bounding a pixel (no relation to wave spacing!, there can be gaps or overlap)
   wave_lo = np.polyval(disp,aa-0.5)  # increasing
   wave_hi = np.polyval(disp,aa+0.5)  # increasing
   energy_lo = uvotio.angstrom2kev(wave_hi[aarev]) # increasing energy channels (index reverse)
   energy_hi = uvotio.angstrom2kev(wave_lo[aarev])
   energy_mid = uvotio.angstrom2kev(wave[aarev])
   
   # output arrays
   n_grp = np.ones(NN)
   f_chan = np.zeros(NN)
   n_chan = np.ones(NN) * NN
   matrix = np.zeros( NN*NN, dtype=float).reshape(NN,NN)
   
   # low resolution arrays
   _matrix = np.zeros( NL*NN, dtype=float).reshape(NL,NN)
   
   # assuming first order
   # second order needs attention: instrument + LSF

   # telescope and image intensifier broadening   
   if wheelpos < 500:
      instrument_fwhm = 2.7/0.54 # pix
   else:    
      instrument_fwhm = 5.8/0.54 # pix
   ww = uvotgetspec.singlegaussian(np.arange(-12,12),1.0,0.,instrument_fwhm)
   ww = ww/ww.sum().flatten()  # normalised gaussian 
            
   UVOTPY = os.getenv('UVOTPY')
   if UVOTPY == '': 
      raise IOError( 'The UVOTPY environment variable has not been set; aborting RMF generation [write_rmf_file-]'+version)

   lsffile = fits.open(  UVOTPY+'/calfiles/zemaxlsf.fit' )  
   if wheelpos < 500: 
      lsfextension = 1
   else:
      print "using the LSF model of the UV grism for the Visible grism until such time as it can be incorporated"   
   lsfchan = lsffile[1].data['channel'][0:15]   # energy value 
   epix    = lsffile[1].data['epix'][0,:]       # 158 values - offset in half pixels (to be converted to wave(wave))
   lsfdata = lsffile[1].data['lsf'][:15,:]      # every half pixel a value
   lsfwav = uvotio.kev2angstrom(lsfchan)
   lsffile.close()
      
   e_mid = 0.5*(energy_lo+energy_hi)  # increasing energy 

   d_lo = np.array(interpolate.splev(wave_lo, tck_Cinv,) + 0.5,dtype=int)
   d_hi = np.array(interpolate.splev(wave_hi, tck_Cinv,) + 0.5,dtype=int)
   
   for k in range(NL):
         if (chatter > 0) & (k == (NL/10)*10):  
    	      print "RMF : ",(NL/10),"% ..."
         ii = iNL[k]
         #  find index e in lsfchan and interpolate lsf
         w = wave[ii]
         j = lsfwav.searchsorted(w)
         if j == 0: 
            lsf = lsfdata[0,:].flatten()
         elif ((j > 0) & (j < 15) ):
            e1 = lsfchan[j-1]
 	    e2 = lsfchan[j]
 	    frac = (e_mid[k]-e1)/(e2-e1)
            lsf1 = lsfdata[j-1,:]
	    lsf2 = lsfdata[j,:]
	    lsf = ((1-frac) * lsf1 + frac * lsf2).flatten()	 	 
         else:
            # j = 15
	    lsf = lsfdata[14,:].flatten()

         # convolution lsf with instrument_fwhm and multiply with response 
         lsf_con = convolve(lsf,ww.copy(),)
	 qpos = (lsf_con != 0.)
      
         # assign wave to lsf array relative to w at index k in matrix (since on diagonal)   
         # rescale lsfcon from half-pixels to channels 

         d  = (np.where(qpos)[0]-79)*0.5 + interpolate.splev(w, tck_Cinv,)  
         #d   = np.arange(-79,79)*0.5 + interpolate.splev(w, tck_Cinv,)
         wave1 = np.polyval(disp,d)  
         ener = uvotio.angstrom2kev(wave1)
         # now each pixel has a wave (wave1), energy(keV) (ener) and lsf_con value 

         # new array to fill 
         lsfnew   = np.zeros(NN)
	 ener_ = list(ener)
	 lsf_con_ = list(lsf_con[qpos])
	 ener_.reverse()
	 lsf_con_.reverse()
	 # now we have ener as an increasing function - if not, the interpolating function fails.
         inter = interpolate.interp1d(ener_, lsf_con_,bounds_error=False,fill_value=0.0)
	 
         for i in range(NN):
	    lsfnew[i] = np.abs(inter( e_mid[i] )) 
	 
	 q = np.isfinite(lsfnew)
	 qx = np.isnan(lsfnew)
	 if np.array(qpos,dtype=int).sum() > 0:   
	    lsfnew_norm = lsfnew[q].sum() 
	    if (np.isnan(lsfnew_norm)) | (lsfnew_norm <= 0.0): lsfnew_norm = 5.0e9     
            lsfnew[q] = ( lsfnew[q] / lsfnew_norm) * resp[ii]
	    if np.array(qx,dtype=int).sum() >0: lsfnew[qx[0]] = 0.	 
            matrix[NN-k-1] =  lsfnew 
	 else:
	    matrix[NN-k-1] =  np.zeros(NN)


   # for output
   if wheelpos < 500: 
      filtername = "UGRISM"
   else:
      filtername = "VGRISM"

   hdu = fits.PrimaryHDU()
   hdulist=fits.HDUList([hdu])
   hdulist[0].header.update('TELESCOP','SWIFT   ','Telescope (mission) name')                       
   hdulist[0].header.update('INSTRUME','UVOTA   ','Instrument Name')   
    
   col11 = fits.Column(name='ENERG_LO',format='E',array=energy_lo,unit='KeV')
   col12 = fits.Column(name='ENERG_HI',format='E',array=energy_hi,unit='KeV') 
   col13 = fits.Column(name='N_GRP',format='1I',array=n_grp,unit='None')
   col14 = fits.Column(name='F_CHAN',format='1I',array=f_chan,unit='None')
   col15 = fits.Column(name='N_CHAN',format='1I',array=n_chan,unit='None' )
   col16 = fits.Column(name='MATRIX',format='PE(NN)',array=matrix,unit='cm**2' )
   cols1 = fits.ColDefs([col11,col12,col13,col14,col15,col16])
   tbhdu1 = fits.new_table(cols1)    
   tbhdu1.header.update('EXTNAME','MATRIX','Name of this binary table extension')
   tbhdu1.header.update('TELESCOP','Swift','Telescope (mission) name')
   tbhdu1.header.update('INSTRUME','UVOTA','Instrument name')
   tbhdu1.header.update('FILTER',filtername)
   tbhdu1.header.update('CHANTYPE','PI', 'Type of channels (PHA, PI etc)')
   tbhdu1.header.update('HDUCLASS','OGIP','format conforms to OGIP standard')
   tbhdu1.header.update('HDUCLAS1','RESPONSE','RESPONSE DATA')
   tbhdu1.header.update('HDUCLAS2','RSP_MATRIX','contains response matrix')   
   tbhdu1.header.update('HDUCLAS3','FULL','type of stored matrix')   
   tbhdu1.header.update('HDUVERS','1.3.0','version of the file format')      
   tbhdu1.header.update('ORIGIN','MSSL/UCL','source of FITS file')
   tbhdu1.header.update('TLMIN4', 1, 'First legal channel number')                           
   tbhdu1.header.update('TLMAX4',NN, 'Last legal channel number')                           
   tbhdu1.header.update('NUMGRP',NN, 'Sum of the N_GRP column')                           
   tbhdu1.header.update('NUMELT',NN, 'Sum of the N_CHAN column')                           
   tbhdu1.header.update('DETCHANS',NN, 'Number of raw detector channels')                           
   tbhdu1.header.update('LO_THRES',1.0E-10, 'Minimum value in MATRIX column to apply')                           
   tbhdu1.header.update('DATE',datestring, 'File creation date')                           
   hdulist.append(tbhdu1)
   
   col21 = fits.Column(name='CHANNEL',format='I',array=channel,unit='channel')
   col22 = fits.Column(name='E_MIN',format='E',array=energy_lo,unit='keV')
   col23 = fits.Column(name='E_MAX',format='E',array=energy_hi,unit='keV')
   cols2 = fits.ColDefs([col21,col22,col23])
   tbhdu2 = fits.new_table(cols2)    
   tbhdu2.header.update('EXTNAME','EBOUNDS','Name of this binary table extension')
   tbhdu2.header.update('TELESCOP','Swift','Telescope (mission) name')
   tbhdu2.header.update('INSTRUME','UVOTA','Instrument name')
   tbhdu2.header.update('FILTER',filtername)
   tbhdu2.header.update('CHANTYPE','PI', 'Type of channels (PHA, PI etc)')
   tbhdu2.header.update('HDUCLASS','OGIP','format conforms to OGIP standard')
   tbhdu2.header.update('HDUCLAS1','RESPONSE','RESPONSE DATA')
   tbhdu2.header.update('HDUCLAS2','EBOUNDS','type of stored matrix')   
   tbhdu2.header.update('HDUVERS','1.2.0','version of the file format')      
   tbhdu2.header.update('DETCHANS',NN, 'Number of raw detector channels')                           
   tbhdu2.header.update('TLMIN1', 1, 'First legal channel number')                           
   tbhdu2.header.update('TLMAX1',NN, 'Last legal channel number')                              
   tbhdu2.header.update('DATE',datestring, 'File creation date')                           
   hdulist.append(tbhdu2)     
   hdulist.writeto(rmffilename,clobber=clobber)
