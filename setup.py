from distutils.core import setup
setup(name='uvotpy',
      version='2.2.0',
      description='Swift UVOT grism spectral processing software',
      long_description="""
      This software was written to extract astronomical spectra from grism images
      taken with the Ultraviolet and Optical Telescope on the Swift spacecraft.
      See http://www.swift.ac.uk/ or http://swift.gsfc.nasa.gov for details.
      
      The Swift spacecraft observes faint objects in the sky that usually cannot 
      be seen by the naked eye, concentrating mostly on variables, like 
      supernovae, gamma ray bursts, x-ray binaries and such in gamma rays, X-rays
      and the uv and optical spectral regions with three instruments. 
      
      The UVOT uv grism observes the spectrum from 168 - 520 nm, while the optical 
      grism does that from 270-650 nm. The grisms make images of 2048x2048 pixels
      which are distortion corrected by the ground processing. The Swift software 
      is part of the HEADAS software distributed by HEASARC at Goddard Space Flight 
      Center (NASA). That software is required to be installed and operational 
      for the spectral extraction to work. WCStools, written by Doug Mink at SAO
      are also required, as well as an active internet connection. 
      
      The spectral extraction will locate the spectrum in the image based on 
      the provided sky coordinates which must be provided in RA, Dec (J2000) in 
      degrees. The output is in the form of plots and FITS formatted files 
      compatible with XSPEC.         

      This code has been registered with the Astrophysics Source Code Library as
      ascl:1410.004, and has been assigned a DOI 10.5281/zenodo.12323

      The calibration of the Swift UVOT grisms has been described in the following 
      publication:
      BIBCODE 2015MNRAS.449.2514K ; Kuin et al. 2015, Monthly Notices of the 
      Royal Astronomical Society, Vol. 449, p2514. 
                       """,
      author='Paul Kuin',
      author_email='npkuin@gmail.com',
      url='http://www.mssl.ucl.ac.uk/www_astro/uvot/',
      platforms=['MacOS X 10.6.8+','linux Debian'],
      license='Releases under a 3-clause BSD style license',
      packages = ['uvotpy'],
      package_dir  = {'' : ''}, 
      package_data = {'uvotpy' : [ 'licence/licence.txt',
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
      requires = ['numpy','matplotlib','scipy','astropy','stsci.convolve','stsci.imagestats',],
      # futurize? 
      classifiers=['Programming Language :: Python ::2.7',
                   'Programming Language :: Python ::3.5'],
      # requires HEADAS, WCStools, internet connection
      # environment setup  requires UVOTPY to point to the installed uvotpy library and calfiles.
      scripts=['uvotpy/scripts/uvotgrism','uvotpy/scripts/fileinfo','uvotpy/scripts/uvotmakermf','uvotpy/scripts/convert_sky2det2raw'],		   
      )
