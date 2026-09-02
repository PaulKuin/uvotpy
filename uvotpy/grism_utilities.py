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
import numpy as np

def query_DSS(target, ImSize=10, server="STSCI",
     version = "3", output=None, returns=False, chatter=0):
    """
    Parameters
    ==========
    target: str or list or astropy.coord    
      targetname or coordinates - Either a string giving a target name, 
      (with J2000 coordinates determined by SIMBAD (default) or NED), 
      or a 2-element list giving the J2000 right ascension in 
      *degrees* and the target declination in degrees, or an
      astropy coordinate object.

    ImSize: float 
      The size of the image to be retrieved in 
      arcminutes.

    server: float ["ESO/"STSCI"]
      ESO - Use the ESO server for image retrieval.    
      STSCI - Use the STScI server 

    output: path
      string specifying name of output FITS file. If set, then the 
      retrieved image and header are not returned to the calling 
      environment unless returns is set

    version : str, ['1','2','3','4','DSS1','DSS2-blue','DSS2-red','DSS2-infrared'] 
       Which survey to retrieve.  
          Possible version values are (STSCI)
          '1' - First Generation survey (garden variety)
          '2' - Second generation survey (incomplete)
          '3' - Check the 2nd generation; if no image is available,
                then go to the 1st generation.
          '4' - The Quick V survey (whence came the Guide Stars Catalog;
                used mostly for Phase II proposal submission)
          possible values are (ESO)
          'DSS1'          - First generation (red), this is the default
          'DSS2-blue'     - Second generation blue
          'DSS2-red'      - Second generation red
          'DSS2-infrared' - Second generation near-infrared
       Note that the DSS2 surveys do not cover the complete Sky.   


     Returns
     =======
     If output is None: (hdr, image)
     Otherwise, the result is a FITS file with the path specified in 'output'.
     
     Notes
     ===== 
     Note that the first two letters of the 'REGION' keyword in the FITS header 
     gives the bandpass 'XP' - Red IIIaF, 'XJ' - Blue IIIaJ, 'XF' - Near-IR IVN
    
     See http://archive.eso.org/dss/dss and/or 
     http://archive.stsci.edu/dss/index.html for details.
     
     """
     #actual queries going out are like:
     #http://archive.stsci.edu/cgi-bin/
     #  dss_search?v=poss1_blue&r=&d=&e=J2000&h=15.0&
     #  w=15.0&f=fits&c=none&s=on&fov=NONE&v3=34 
     # 
     #http:////archive.eso.org/dss/dss/image?ra=279.23567&dec=+-28.92777&equinox=J2000&
     #name=&x=10&y=10&Sky-Survey=DSS1&mime-type=download-fits&statsmode=WEBFORM
     
     # 2026-09-01 fixed problem in target type identification
     # 2015-07-26 initial version
     # copyright 2015 NPM Kuin

    import numpy as np, astropy
    import astropy.utils.data
    from astropy import units
    from astropy.io import fits
    from astropy import coordinates     
    import requests
    import xml.etree.ElementTree as ET
    import os
    
    if type(output) == type(None): 
       outputfile = False
       output='tmp.fit'
    else:
       outputfile = True   
    blocksize = astropy.utils.data.conf.download_block_size
    Equinox='j2000' 
    if type(target) == astropy.coordinates.sky_coordinate.SkyCoord:
        ra =  target.ra.deg
        dec = target.dec.deg
    elif type(target) = list:
        ra =  target[0].ra.deg
        dec = target[1].dec.deg  
    elif hasattr(target, 'name'):
        ra =  target.ra.deg
        dec = target.dec.deg
    elif type(target) == list:
        ra,dec = target 
    elif type(target) == str:
        pos = coordinates.SkyCoord.from_name(target,frame=coordinates.ICRS)
           #,unit=(units.deg,units.deg))
        ra =  pos.ra.deg
        dec = pos.dec.deg
    else:
        raise IOError("Parameter target is not a legal option.")    

    if server == 'STSCI':  
        if not version in ['1','2','3','4']:
           raise IOError("Version parameter %s is not correct for STSCI DSS query.")  
        baseURL="http://archive.stsci.edu/cgi-bin/dss_search"
        query = "?ra="+str(ra)+"&dec="+str(dec)+"&equinox="+str(Equinox)+\
           "&height="+str(ImSize)+"&version="+version+\
           "&width="+str(ImSize)+"&format=FITS"+"&output="+output
    elif server == "ESO":
        if not version in ['DSS1','DSS2-blue','DSS2-red','DSS2-infrared']:
            raise IOError("Version parameter %s is not correct for ESO DSS query.")  
          
        baseURL="http://archive.eso.org/dss/dss/image"
        query= "?ra="+str(ra)+"&dec=+"+str(dec)+\
                       "&x="+str(ImSize)+\
                       "&y="+str(ImSize)+\
                       "&Sky-Survey="+dss+"&mime-type=download-fits"
    else:
        raise IOError("unknown server {0}")         
                       
    r = requests.get(baseURL+query)
    if r.status_code == 200:
        f = open('tmp.fit','wb')
        for block in r.iter_content(blocksize):
            f.write(block)
        f.close()
        r.close()
        #if os.access('tmp.fit',os.F_OK):
        
        f = fits.open('tmp.fit')
        hdr,image = f[0].header, f[0].data
        f.close()
        if outputfile : 
            os.system("mv tmp.fit "+output) 
            if returns: return hdr, image
        else:
            os.system("rm -f tmp.fit")               
            return hdr, image
    else:
        raise RuntimeError("failure to connect to DSS server at %s."%server)


def decsex(ra,dec):
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
   
   sdec = dec/np.abs(dec)
   ded=int(dec)
   dem=int((dec-ded)*60.)
   des=np.abs((dec-ded-dem/60.)*3600)
   ded=sdec*np.abs(ded)
   dem=np.abs(dem)
   newdec="%0i:%02i:%04.1f"%(ded,dem,des) 
   
   return newra,newdec


def compute_offset_pointing(target, roll_angle, offset,chatter=0):
    """
    
    Parameters
    ==========
    target: astropy.coordinates.sky_coordinate.SkyCoord
        target coordinates as an astropy coordinate object
        at an offset 

    roll_angle: astropy.units.quantity.Quantity 
        the roll angle in units of deg

    offset : astropy.units.quantity.Quantity  of length 2
        Xoffset, Yoffset are the positions of the target 
        on the detector relative to the centre 
        
    returns
    =======
    pointing: astropy.coordinates.sky_coordinate.SkyCoord
        the sky position of the center of the image
    """
    import numpy as np
    from astropy import coordinates     
    from astropy import units

    if hasattr(target, 'name') | hasattr(target,'ra'):
        ra =  target.ra.deg
        dec = target.dec.deg
    else:
        raise IOError("Parameter target is not correct")    

    if hasattr(roll_angle,'to'):
       roll_angle = roll_angle.to(units.deg)
    else:   
        raise IOError("Parameter roll_angle is not correct.")    

    if hasattr(offset,'to'):
       offset = offset.to(units.arcsec)
    else:   
        raise IOError("Parameter offset is not correct.")    
        
    pa = 240.64-roll_angle
    
    raoff = ra+\
            yoff.to(units.deg)*np.sin(pa.to(units.rad))/np.cos(dec.to(units.rad))+\
            xoff.to(units.deg)*np.cos(pa.to(units.rad))/np.cos(dec.to(units.rad))
    decoff= dec-\
            yoff.to(units.deg)*np.cos(pa.to(units.rad))+\
            xoff.to(units.deg)*np.sin(pa.to(units.rad))

    pointing = coordinates.SkyCoord(raoff,decoff,frame=coordinates.ICRS,)

    if chatter > 0:
        print ('Decimal RA/DEC of offset:    (%12.7f  %12.7f)' % (raoff,decoff) )
        ranew,decnew = _decsex(raoff,decoff)
        print ('Sexigesimal RA/DEC of offset: (%s, %s)\n' % (ranew,decnew) )
        #print 'Sexigesimal RA/DEC of offset: %0i;%2i;%2i, %0i:%2i:%2i\n' % (pointing.ra.hms,pointing.dec.dms)

    return pointing


def detposition2sky(pointing,detposition, roll_angle, ):
    """
    given the detector position, compute the sky position

    Parameters
    ==========
    pointing: astropy.coordinates.sky_coordinate.SkyCoord
        target coordinates as an astropy coordinate object
        at an offset 

    roll_angle: float, or astropy.units.quantity.Quantity 
        the roll angle in deg

    offset : astropy.units.quantity.Quantity  of length 2
        Xoffset, Yoffset are the positions of the target 
        on the detector relative to the centre 
        
    returns
    =======
    sky: astropy.coordinates.sky_coordinate.SkyCoord
        the sky position of the det offset
    """
    if hasattr(detposition, 'to'):
        detx, dety =  detposition.to(units.pix)
    else:
        detx,dety = detposition   
    offset = [detx*0.54, dety*0.54]*units.arcsec # grism pixel scale used 
    sky = _compute_offset_pointing(pointing, roll_angle, offset,chatter=0)
    return sky
