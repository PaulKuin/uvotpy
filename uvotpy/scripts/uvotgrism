#!/usr/local/eureka/Ureka/variants/common/bin/python
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

__version__ = '-1.1 20141020' 

'''
.. _uvotpy:

UVOTPY 
======

This is the main module in **uvotpy** package to extract spectra from 
the `Swift` "UVOT" GRISM observations.


General
-------

**This code is designed for processing Swift UVOT Grism images**
  
  The goal is to process the spectra in the image, to do 
  quality control and source identification. Because the UVOT is 
  a photon counting detector, the error handling must keep track 
  of the errors which require in principle the total exposure, 
  the background exposure, and the exposure time. The data quality
  for each pixel should be flagged also (e.g., data dropouts, 
  scattered light rings, halo around bright features), but flagging 
  is only done for nearby zeroth orders. While the count rate errors 
  follow from the observed binomial nature of the detector, e.g., 
  Kuin & Rosen (2008,MNRAS, 383,383) - though that prescription was 
  for point sources. A heuristic method was used to develop a 
  correction for coincidence loss used in this program. 
  
  Details of the accuracy and reliability of the calibration and 
  software are soon to be submitted to the Monthly Notices of the 
  RAS. 
  
  This program extracts the spectra, applies the wavelength 
  calibration file to find anchor position and wavelength 
  dispersion. The flux calibration is valid over the whole 
  detector and depends on the effective area and coincidence
  loss correction.  In *both grisms* the effective area was 
  determined at several offset positions  The accuracy of the 
  flux in the *uv grism* is of order 5% in the centre and 
  about 10% at other locations on the detector. In the 
  *visible grism* the accuracy of the flux is 20%, which error 
  is dominated by that in the coincidence loss correction.  
     
  In the nominal grism mode (wheelpos = 200, 1000) the 
  response varies by about 5% from centre to about  200 pixels 
  from the edge. In the clocked grism modes (wheelpos = 
  160, and 955) the response has a strong drop when the spectrum 
  falls in the upper left corner. For the rest of the image 
  the response varies less than ~20% in the clocked modes. 
  
  Before using this code, it is recommended to reprocess the raw 
  image using the *mod-8 correction*. Use a CALDB later than May 2010 
  for an improved distortion correction. The grism detector image 
  should be attitude corrected using uvotgraspcorr. Check the header
  keyword ASPCORR='GRASPCORR'. 
  
  Find out the RA and DEC position in decimal degrees from the 
  USNO-B1 catalog for your object  since the UVOT aspect corrections 
  also use the USNO-B1 positions, so you will avoid a systematic error 
  in positions which will translate to a shift in the wavelengths 
  derived. 
  
  You need to set the environment variable UVOTPY to point to the 
  directory with the UVOTPY code and spectral calibration files. 
  
  The input data files may have to be decompressed before running 
  the program, although it will try to do that itself, but does 
  not recompress the files again. 
  
  The program was developed running in iPython, and it is suggested to 
  run interactively in iPython rather then run as it this script. 
  
  Second order extraction and calibration are only treated very roughly at 
  this time. Zeroth orders are only minimally identified. Third order 
  location is approximate. 
   
Main functions
--------------
  getSpec : main call for spectral data extraction  
  
Other uvotpy functions
-------------------------
  curved_extraction: set curvature of orders, set quality flags, get spectral data 
  extractSpecImg : get sub image 
  findBackground : get background
  get_components : extract first, second, third order components
  getCal : get the (wave) calibration files
  predict_second_order : use first order to predict second order (very rough)
  coi_func : return an interpolating function(wave) for the coincidence loss of a spectrum [experimental]

Specialized functions are in modules
------------------------------------
  uvotgetspec: repository of main functions  
  uvotio:    writes file output
  uvotio.rate2flux : convert count rate to flux 
  uvotplot : plot routines
  uvotmisc : miscellaneous routines
  
Files  
------  
  The program assumes all data files are available in either the working 
  directory, or the directory structure complies with the Swift project 
  standard and is run from the <obsid>/uvot/images directory, while 
  the attitude file is available in the <obsid>/aux directory.  
  There is rudimentary support for running from a remote directory on the 
  same device, but the program will write some files to both the current and
  the data directories.

  The flux-calibrated 1st order spectrum is available in the second extension of the 
  output file.

  From version 1.0 onwards, the file name includes a flag "_f" for when lenticular
  filter image(s), or "_g" when "uvotgraspcorr" aspect corrections were used 
  to derive the anchor position. Both methods give similar uncertainties, but
  for the same field uvotgraspcorr will give more consistent results, while 
  the lenticular filter method works when uvotgraspcorr cannot find an aspect 
  correction (in that case the uncorrected pointing position from the star trackers 
  will be used).  
            
History
-------
  2014-Oct-20 Paul Kuin 
  Added keyword for severe background clipping 

  2013-Oct-31 Paul Kuin
  Rewritten uvotwcs to fix a bug in the calculation of the pointing. Small fixes
  all throughout. Output file names will have a flag for the anchor point method used.

  2013-May-23 Paul Kuin
  revised the uv grism clocked mode wavecal for first and second order. Missing 
  still is second order effective area. 
  
  2013-Mar-07 Paul Kuin
  Experimental new calibration files for the uv grism effective area are included 
  with the software and used if found. This file is the prototype for inclusion to 
  the Swift CALDB.
  Documentation of the software is currently being made compliant with usage with 
  Sphinx so we can auto-generate documentation.
  A document describing the wavelength calibration is included. Documents describing
  the flux calibration and coincidence-loss correction are in preparation. 
  For more current information, see my website http://www.mssl.ucl.ac.uk/~npmk/. 

  2012-SEP-15 Paul Kuin
  Recent changes have not been implemented in the routine for optimal extraction
  since the method does not yield a better result and has some serious problems 
  left. 
  The initial correction for the COI loss using a formulation based on the  
  compensating for extra losses from high backgrounds has been implemented
  in the worng way, but it works.  The effective areas (response) have been 
  determined for the case of no coincidence loss using low flux, low 
  background calibration sources and making the best available coi-corrections.
   
  The spatial dependence of the flux calibration is for now implemented by 
  searching for the closest-by effective area curve available in $UVOTPY/calfiles.
  
  When using the current calibration for a smaller aperture of 1.0 sigma (by setting 
  global uvotgrism.trackwidth = 1.0 within iPython, or changing the trackwidth line 
  below) please note that the aperture correction and coi-correction have not been 
  fine-tuned to better than ~20%. This can be remedied by also extracting with the 
  full 2.5 sigma trackwidth, and correcting for the difference. There is a small gain 
  in S/N and avoidance of contaminating zeroth orders by doing that.     
	    
'''

import sys
import re
import warnings
import optparse
import numpy as np
import pylab as plt
try: 
   from astropy import fits as pyfits
except:   
   import pyfits
try:
  import imagestats, convolve
  from convolve import boxcar
except: 
  import stsci.imagestats as imagestats
  import stsci.convolve as convolve  
  from stsci.convolve import boxcar
import scipy
from scipy import interpolate
from scipy.optimize import leastsq
from numpy import polyfit, polyval
import datetime
import os

try:
   from uvotpy import uvotplot,uvotmisc,uvotwcs,rationalfit,mpfit,uvotio
   from uvotpy.uvotmisc import interpgrid,uvotrotvec
   from uvotpy.uvotgetspec import *
except:    # old versions 
   from uvotmisc import interpgrid, uvotrotvec
   import uvotplot
   from uvotgetspec import *

# Global parameters

status = 0
do_coi_correction = True  # if not set, disable coi_correction
tempnames = list()
tempntags = list()
cval = -1.0123456789
interactive = True
update_curve = True
contour_on_img = False
give_result = False # with this set, a call to getSpec returns all data 
use_rectext = False
background_method = 'boxcar'  # alternatives 'splinefit' 'boxcar'
background_smoothing = [50,7]   # 'boxcar' default smoothing in dispersion and across dispersion in pix
trackwidth = 2.5  # width of extraction region in sigma  (alternative default = 1.0) 2.5 was used for flux calibration.
bluetrackwidth = 1.3 # multiplier width of non-order-overlapped extraction region [not yet active]
write_RMF = False
background_source_mag = 18.0
zeroth_blim_offset = 1.0
_PROFILE_BACKGROUND_ = False  # this preps the background image by strong sigma clipping 
      
      
if __name__ == '__main__':
   #in case of called from the OS

   if status == 0:
      usage = "usage: %prog [options] <images>"

      epilog = '''Required input: 
The object_name or its coordinates in decimal degrees and the obsid.  
For details on the uvot grism see http://www.mssl.ucl.ac.uk/www_astro/uvot/ and 
http://swift.gsfc.nasa.gov/ ''' 

      anchor_preset = list([None,None])
      bg_pix_limits = list([-100,-70,70,100])
      bg_lower_ = list([None,None])  # (offset, width) in pix, e.g., [20,30], default [50,50]
      bg_upper_ = list([None,None])  # (offset, width) in pix, e.g., [20,30], default [50,50]

      parser = optparse.OptionParser(usage=usage,epilog=epilog)
      
      # main options

      parser.add_option("", "--object_name", dest = "object_name",
                  help = "object name to use for position lookup (web access needed) [default: %default]",
                  default = None)

      parser.add_option("", "--ra", dest = "ra",
                  help = "RA (deg) [default: %default]",
                  default = -1.0)

      parser.add_option("", "--dec", dest = "dec",
                  help = "DEC (deg) [default: %default]",
                  default = 0.0)

      parser.add_option("", "--obsid", dest = "obsid",
                  help = "OBSID [default: %default]",
                  default = "00000000000")

      parser.add_option("", "--extension", dest = "ext",
                  help = "extension number [default: %default]",
                  default = 1)

      parser.add_option("", "--dir", dest = "indir",
                  help = "source directory [default: %default]",
                  default = './')

      # supply calibration files 

      parser.add_option("", "--wavecalfile", dest = "wavecalfile",
                  help = "wavelength calibration file [default: %default]",
                  default = None)
		  
      parser.add_option("", "--fluxcalfile", dest = "fluxcalfile",
                  help = "flux calibration file (placeholder) [default: %default]",
                  default = None)
		  
      parser.add_option("", "--RMF", dest = "RMF",
                  help = "produce an RMF file [default: %default]",
                  default = False)

      # control spectral extraction parameters		  
		  
      parser.add_option("", "--background_lower", dest = "background_lower_",
                  help = "Specify lower background location (rotated image) in sigma from 1st order [default: %default]",
                  default = bg_lower_)

      parser.add_option("", "--background_upper", dest = "background_upper_",
                  help = "Specify upper background location (rotated image) in sigma from 1st order [default: %default]",
                  default = bg_upper_)

      parser.add_option("", "--anchor_offset", dest = "anchor_offset",
                  help = "force anchor offset in pix [default: %default]",
                  default = None)

      parser.add_option("", "--anchor_position", dest = "anchor_position",
                  help = "force anchor position in image coordinates (pix) [default: %default]",
                  default = anchor_preset)

      parser.add_option("", "--angle", dest = "angle",
                  help = "force angle of subimage extraction (deg) [default: %default]",
                  default = None)

      parser.add_option("", "--fit_second_order", 
                  dest = "fit_second",
                  help = "attempt to fit second order [default: %default]",
                  default = False)

      parser.add_option("", "--predict_second_order", 
                  dest = "predict_second",
                  help = "attempt to predict second order from first order [default: %default]",
                  default = True)
		  
      parser.add_option("", "--input_order_curvature_polynomials", action="store_true",
                  dest = "get_curve_poly",
                  help = "supply your own fit to the curvature of the orders interactively [default: %default]",
                  default = False)		  
		  
      parser.add_option("", "--fit_sigmas", 
                  dest = "fit_sigmas",
                  help = "attempt to fit order width (within preset limits) [default: %default]",
                  default = True)
		  
      parser.add_option("", "--input_sigma_polynomials", action="store_true",
                  dest = "get_sigma_poly",
                  help = "supply your own fit to the gaussian width of the orders interactively [default: %default]",
                  default = False)	
		  
      parser.add_option("", "--straight_slit_width",
                  dest = "slit_width",
                  help = "if given the number of pixels for the width of a straight slit extraction [default: %default]",
                  default = 13)	
		  
      parser.add_option("", "--curved_extraction", 
                  dest = "curved_extraction",
                  help = "Follow the calibrated curvature of the orders for the extraction (keep,update)[default: %default]",
                  default = "update")
		  
      parser.add_option("", "--optimal_extraction", action="store_true",
                  dest = "optimal_extraction",
                  help = "do an optimal extraction [default: %default]",
                  default = False)	
		  
      parser.add_option("", "--skip_field_sources", action="store_true",
                  dest = "skip_field_sources",
                  help = "Do not retrieve field source positions [default: %default]",
                  default = False)	
      
		  
      # input/output/plot options 		  	  

      parser.add_option("", "--plot_image", 
                  dest = "plotimage",
                  help = "plot detector image [default: %default]",
                  default = True)

      parser.add_option("", "--plot_raw", 
                  dest = "plotraw",
                  help = "plot raw image, count spectra orders [default: %default]",
                  default = True)

      parser.add_option("", "--plot_spectrum", 
                  dest = "plotspec",
                  help = "plot calibrated spectrum [default: %default]",
                  default = True)

      parser.add_option("", "--zoom", 
                  dest = "zoom",
                  help = "zoom to source if plotting [default: %default]",
                  default = False)

      parser.add_option("", "--write_outfile", action="store_false",
                  dest = "wr_outfile",
                  help = "write output file [default: %default]",
                  default = True)
		  
      parser.add_option("", "--outfile", 
                  dest = "outfile",
                  help = "output file name base [default: derived from obsid and ext]",
                  default = None)	
		  
      parser.add_option("", "--clobber", action="store_true",
                  dest = "clobber",
                  help = "overwrite output file [default: %default]",
                  default = False)
		  
      parser.add_option("", "--chatter", dest = "chatter",
                  help = "verbosity [default: %default]",
                  default = 0)
		  
   (options, args) = parser.parse_args()

   if (len(sys.argv) == 1):
      # no arguments given 
      program_name = sys.argv[0]
      print program_name," no arguments found"
      status = 1


   if (len(args) > 0):
       #        parser.print_help()
       parser.error("Incorrect argument(s) found on command line: "+str(args))

   object_name = options.object_name
   if object_name == None:
      ra_  = float(options.ra)
      dec_ = float(options.dec)
   else:
      #from uvotgetspec import get_radec
      ra_,dec_ = get_radec(objectid=object_name)
      if ra == None:
         print "ERROR: unable to get a position from the object_name. \n       Please supply RA, DEC\n"
	 raise         

   give_result = False 
   
   if options.anchor_offset != None: 
      offsetlimit = 0.1
   else:
      offsetlimit = None   
      
   print "debug: fit_second_order:",options.fit_second
   print "debug: optimal_extraction:", options.optimal_extraction  
   
   #from uvotgetspec import getSpec
   
   getSpec(ra_,dec_,
     options.obsid,int(options.ext),
     wr_outfile=bool(options.wr_outfile),
     indir=options.indir,
     outfile=options.outfile,
     calfile=options.wavecalfile,
     fluxcalfile=options.fluxcalfile,
     offsetlimit=offsetlimit, 
     write_RMF = options.RMF,
     anchor_offset=options.anchor_offset,
     anchor_position=options.anchor_position, 
     background_lower=options.background_lower_,
     background_upper=options.background_upper_, 
     fixed_angle=options.angle,
     spextwidth=options.slit_width,
     curved=options.curved_extraction,
     fit_second=bool(options.fit_second),
     predict2nd=bool(options.predict_second),
     skip_field_src=bool(options.skip_field_sources),
     optimal_extraction=bool(options.optimal_extraction), 
     plot_img=bool(options.plotimage), 
     get_curve=options.get_curve_poly, 
     fit_sigmas=options.fit_sigmas,
     get_sigma_poly=options.get_sigma_poly,
     plot_raw=bool(options.plotraw), 
     plot_spec=bool(options.plotspec), 
     zoom=bool(options.zoom), 
     highlight=contour_on_img, 
     clobber=bool(options.clobber), chatter=int(options.chatter))
     
   yn = raw_input("Done ?")
   print "for further processing use the output files" 
   print today_
   print datestring
   print "(c) NPMK 2013-2016 - Mullard Space Science Lab, University College London"
   
else:
    
   print "uvotpy"+__version__+'  n.kuin@ucl.ac.uk'

today_ = datetime.date.today()   
datestring = today_.isoformat()[0:4]+today_.isoformat()[5:7]+today_.isoformat()[8:10]

