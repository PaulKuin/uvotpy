from distutils.core import setup
setup(name='uvotpy',
      version='1.0',
      description='Swift UVOT grism spectral processing software',
      long_description="""
      This software was written to extract astronomical spectra from grism images
      taken with the Ultraviolet and Optical Telescope on the Swift spacecraft.
      See http://www.swift.ac.uk or http://swift.gsfc.nasa.gov for details.
      
      The Swift spacecraft observes faint objects in the sky that usually cannot 
      be seen by the naked eye, concentrating mostly on variables, like 
      supernovae, gamma ray bursts, x-ray binaries and such in gamma rays, X-rays
      and the uv and optical spectral regions with three instruments. 
      
      The UVOT uv grism observes the spectrum from 165 - 500 nm, while the optical 
      grism does that from 270-650 nm. The grisms make images of 2048x2048 pixels
      which are distortion corrected by the ground processing. The Swift software 
      is part of the HEADAS software distributed by HEASARC at Goddard Space Flight 
      Center (NASA). That software is required to be installed and operational 
      for the spectral extraction to work. WCStools, written by Doug Mink at SAO
      are also required, as well as an active internet connection in order to 
      query the USNO-B1 catalogue. 
      
      The spectral extraction will locate the spectrum in the image based on 
      the provided sky coordinates which must be provided in RA, Dec (J2000). 
      The output is in the form of plots and FITS formatted files compatible
      with XSPEC.         
                       """,
      author='Paul Kuin',
      author_email='npkuin@gmail.com',
      url='http://www.mssl.ucl.ac.uk/www_astro/uvot/',
      platforms=['MacOS X 10.6.8','linux Debian'],
      #py_modules=['uvotgrism','uvotio','uvotmisc','uvotplot','rationalfit',],
      license='Releases under a 3-clause BSD style license',
      packages = ['uvotpy'],
      package_dir  = {'' : ''}, 
      package_data = {'uvotpy' : [ 'licence.txt',
                             'calfiles/usnob1.spec',
                             'calfiles/uvw1_dummy.img',
			     'calfiles/zemaxlsf.fit',
			     'calfiles/README',
			     'calfiles/*.arf',
			     'calfiles/*.fits',
                             'calfiles/superseded/*',
                             'README',
			     'README.sum_spectra',
			     'doc/*']}, 			      
      requires = ['numpy','matplotlib','scipy','astropy','stsci.convolve','stsci.imagestats'],
      # requires HEADAS, WCStools, internet connection
      # environment setup  requires UVOTPY to point to the installed uvotpy library and calfiles.
      scripts=['uvotpy/scripts/uvotgrism','uvotpy/scripts/fileinfo'],		   
      )
