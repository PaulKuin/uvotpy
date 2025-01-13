#!/usr/bin/env python
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
# part of uvotpy (c) 2009-2024, 
# this code original September 2023, update august 2024: N.P.M. Kuin
#
# update August 2024: for the clocked grism the background varies along the 
# dispersion direction, so any shift between template and spectrum will 
# present a quite different background where the background varies a lot.
# For this reason simple subtraction of a background scaled by exposure time
# fails to extract a correct spectrum. The revision extracts the background
# sources (assuming they did not change) and corrects the count rate in the 
# spectrum thus. In addition, curvature of the spectrum changes over the 
# detector and this method avoids this problem also. 

import numpy as np
from uvotpy import uvotspec, uvotgetspec, uvotio
from astropy.io import fits, ascii as ioascii

__version__ = 2 # 2024-12-19

# get uvotSpec processed parameters dictionary:
uvotgetspec.give_new_result=True
uvotgetspec.trackwidth=1.5

class withTemplateBackground(object):
    """
    Use a later observation ("template") as replacement for the background
    extraction in the final resulting spectrum
    
    No accomodation made yet for summed spectra input (needs setting parameter in 
    extract_*
    
    The template and spectrum should be taken at the same roll angle and 
    close to the same detector location if clocked mode was used. 
    
    The aim is to determine the emission from the region under the original spectrum 
    by using the template taken later. Especially the zeroth orders of other field 
    sources, but also any first order spectra from aligned sources, aligned in the 
    dispersion direction. 
    
    Both spectra need to be mod-8 corrected.
    
    Development notes 2024-12-22 npmk:
    
    (1) rotating template to the exact roll angle of the original grism image will 
        align the zeroth orders in template and original, but at the cost of rotating
        the first orders from the template away from that of the original. 
        Hence, aplying the rotation will allow correction for the underlying zeroth 
        orders but not the first order overlaps. Useful for flagging zeroth orders 
        that overlap in case the roll angle difference is larger than 0.7 deg. 
        
    (2) the current approach is to apply the rotation from the difference in roll. 
    
    (3) shifts in the anchor point between original and template will cause a difference 
        in the 1st order dispersion of original and template if the shift is larger than 
        about 120 pixels (1 arcmin) in either x or y. The exact size of the problem needs 
        to be examined.
        
    (4) the background rate in template and original may be different, in which case the
        spectra need to be extracted for each and before subtraction. If the background 
        rates are the same, the template rate image can be subtracted from the original 
        rate image. The reason for that is the coincidence loss difference that scales 
        with the rate, the background difference mainly affecting the peak rate 
        correction.
    
    """
    def __init__(self, spectra=[], templates=[], pos=None, extsp=1, obsidsp="",
        obsidtempl="", exttempl=1, redshift=0.0, outputfile_stub=None, 
        rotate_and_shift=True,chatter=1):
        # input parameters, note all spectra, templates to be PHA files
        self.spectra = spectra
        self.templates = templates
        self.spectrum_number=0
        self.pos = pos # [astropy coordinates] of source
        self.obsidsp=obsidsp
        self.obsidtempl=obsidtempl
        self.extsp = extsp
        self.exttempl = exttempl
        self.indir = "./"
        self.redshift = redshift
        self.chatter=chatter
        self.outputfile_stub=outputfile_stub
        # process variables, parameters
        self.spResult=None
        self.tmplResult=None
        self.spBackground = None
        self.tmplBackground = None
        #self.summed_sp = False     # fit header is different for summed
        #self.summed_templ = False  # ditto, for spectrum, template
        self.specimg=None
        self.templimg=None
        self.tximg2 = None # rolled image 
        self.tximg=None  # reduced specimg to overlap wrt anker
        self.sximg=None  # reduced templimg to overlap wrt anker
        self.spec_exp=50.
        self.templ_exp=0.
        self.spec_bkg=0.    # background found in first order
        self.templ_bkg=0.
        self.dimsp = (-400,1150)
        self.dimspx = (-400,1150)
        self.dimtempl = (-400,1150)
        self.dimtemplx = (-400,1150)
        self.spAnker = [980,1020]  # first guess
        self.bkg_scale=1.   # ratio backgrounds [-400,800] zero at anchor (interp1d)
        #self.ZO_scale=1.    # ratio exposure times
        #self.spectrum=None
        self.template=None
        self.anchor_templimg=[]
        self.anchor_specimg=[]
        self.anchor_teimg=[]
        self.anchor_spimg=[]
        self.ank_c = None
        self.ank_c_sximg = None
        self.ank_c_tximg = None
        self.movexy=0,0 # return from manual alignment
        self.yloc_sp = 100  # the Y location of the spectrum in spimg
        self.widthsp = 15  # pix width for optimal extraction
        #self.specimg_aligned=None
        #self.templimg_aligned=None
        # spectral extraction parameters 
        self.offsetlimit=[97,6]    # this may not work and at some point needs to match both
        self.background_lower=[None,None]
        self.background_upper=[None,None]
        # Prep so that there is just one spectrum and one teplate PHA file
        self.c = None # contour
        self.cval = -1.0123456789
        self.delx = 0 # interactive offset between template contour and spectrum image
        self.dely = 0
        self.stdx = 0
        self.stdy = 0
        self.rotate_and_shift = rotate_and_shift
        #self.test_templ_shift=None
        print ("for instruction see my youtube video (TBD)")
        
    def auto_template(self,):    
        """
        A script to use a template observation to improve the extraction of a UVOT grism 
        spectrum (filter wheel position 200 (nominal), and 160(clocked)).
        
        Call:
            from uvotpy import templating
            S = templating.withTemplateBackground(
                spectra=[],          # fits file name(s)
                templates=[],        # fits file name(s) 
                pos=None,            # position (astropy.coordinates)
                extsp=1,             # extension of fits file
                obsidsp="",          # OBSID of spectrum/original
                obsidtempl="",       # OBSID of template
                exttempl=1,          # fits extension of template file
                redshift=None,       # redshift 
                chatter=1            # verbosity 
                )
                
            template,Y = S.auto_template()
        
        Only for a single spectrum. The original spectrum image is referred to 
        as 'spectrum (or original)', the template as 'template (spectrum)'
        
           run all steps in sequence
        1. check both are mod-8 corrected   
        #1. sum spectra if needed, before using this; this is discouraged until tested
        2. to start use default extraction. This gives a baseline.
           run extract * to get headers, extracted image, ank_c, slice param.
           get exposure *_exp for scaling; set *_bkg found near anchor; scale_factor
           create specimg, templimg from extracted image extension
        3. note anchor position in spectrum: specimg, template: tempimg
        4. determine alignment parameters from template to spectrum
        5. extract the template at the location corresponding to the spectrum 
           and scale template spectrum to the exposure time of the spectrum image
        6. [dropped ] if roll angles differ, flag parts of template spectrum which are 
           contaminated by first orders that rotated into the track we are 
           examining. Exclude first order parts unless it also contains a 
           zeroth order.   
        7. subtract the matched template and spectrum, which contains the zeroth orders 
           in the the background under the spectrum, from the spectrum
                
        
        
        """
        if self.chatter > 0: print (f"Preparing data\n")
        self.set_parameter('offsetlimit',[100,0.1]) # easier matching 
        
        # check mod8
        spmod8 = fits.getval(self.spectra[self.spectrum_number],"MOD8CORR",ext=self.extsp)
        tmplmod8 = fits.getval(self.templates[self.spectrum_number],"MOD8CORR",ext=self.exttempl)
        if not spmod8: print(f"{self.spectra[self.spectrum_number]} needs a MOD8 correction first")
        if not tmplmod8: print(f"{self.templates[self.spectrum_number]} needs a MOD8 correction first")
        if not (spmod8 and tmplmod8): raise IOError("apply MOD8 correction(s) first")
        
        # find anchor for the spectrum (in self.Ysp['anker'])
        self.extract_spectrum()
        # find anchor for the template (in self.Ytmpl['anker'])
        self.extract_template()
        # now we got the img slice and anchor
        # find difference in roll angle and rotate template to spectrum 
        self.rotate_tmpl()
        # the templimg slice has been rotated around its anker and written to the file
        # reextract the corrected template 
        self.extract_template()
        # find initial transform template => spectrum
        self.match_slice()
        # find the overlap in the spectrum and templ images centred on ankers   
        self.SelectShift(spimg=self.sximg, tempimg1=self.tximg)
        # now that we got the offset between spectrum and template, we 
        # first extract the spectrum properly, and then extract the spectrum 
        # for the template at a location to match that of the original spectrum.
        # the results are written to a _spectrum. and _template. pha file each.
        self.process_and_write()
        # the output file are 
        
        
    def process_and_write(self,):
                
        offsp = self.ank_c_sximg[0]
        bglo=[50-self.dely,90-self.dely]
        bghi=[50,90]        
        trcmem = uvotgetspec.trackcentroiding

        uvotgetspec.trackcentroiding = True        
        self.Ysp = uvotgetspec.curved_extraction(        # quick draft
           self.sximg, # from match_slice
           self.ank_c_sximg, # from match_slice
           self.spAnker, # from match_slice
           self.spResult['wheelpos'], 
           expmap=self.spResult['expmap'], offset=0., 
           anker0=None, anker2=None, anker3=None, angle=None, 
           offsetlimit=[offsp,0.5],  # this will centroid - find the y-offset of the spectrum
           background_lower=bglo, # normal extraction - perhaps match to templ one? 
           background_upper=bghi,
           background_template=None, #self.template,
           trackonly=False, 
           trackfull=False,  
           caldefault=True, 
           curved="noupdate", \
           poly_1=None,poly_2=None,poly_3=None, 
           set_offset=False, 
           composite_fit=True, 
           test=None, chatter=0, 
           skip_field_sources=True,\
           predict_second_order=False, 
           ZOpos=None,
           outfull=True,   # check what is needed by I/O module
           msg='',
           fit_second=False,
           fit_third=False,
           C_1=self.spResult['C_1'] ,C_2=None,dist12=None,
           dropout_mask=None)
        uvotgetspec.trackcentroiding = trcmem   

        # now get the count rate spectra
        fitorder_S, cp2_S, (coef0_S,coef1_S,coef2_S,coef3_S), \
        (bg_zeroth, bg_first, bg_second, bg_third), \
        (borderup_S,borderdown_S), apercorr, expospec, msg_S, curved_S = self.Ysp 
        
        (present0,present1,present2,present3),(q0,q1,q2,q3), (
              y0,dlim0L,dlim0U,sig0coef,sp_zeroth,co_zeroth),(
              y1,dlim1L,dlim1U,sig1coef,sp_first,co_first),(
              y2,dlim2L,dlim2U,sig2coef,sp_second,co_second),(
              y3,dlim3L,dlim3U,sig3coef,sp_third,co_third),(
              x,xstart,xend,sp_all,quality,co_back)  = fitorder_S
        
        bg, bg1, bg2, bgsig, bgimg, bg_limits, (bg1_good, bg1_dis, bg1_dis_good, 
        bg2_good, bg2_dis, bg2_dis_good,  bgimg_lin) = self.spBackground
        
        # define
        spYfit={"coef0":coef0_S,"coef1":coef1_S,"coef2":coef2_S,"coef3":coef3_S} 
       
        self.spResult.update({"background_1":bg1,"background_2":bg2})

        spYfit.update({
      "bg_zeroth":bg_zeroth,"bg_first":bg_first,"bg_second":bg_second,"bg_third":bg_third,
      "borderup":borderup_S,"borderdown":borderdown_S,
      "sig0coef":sig0coef,"sig1coef":sig1coef,"sig2coef":sig2coef,"sig3coef":sig3coef,
      "present0":present0,"present1":present1,"present2":present2,"present3":present3,
      "q0":q0,"q1":q1,"q2":q2,"q3":q3,
      "y0":y0,"dlim0L":dlim0L,"dlim0U":dlim0U,"sp_zeroth":sp_zeroth,"bg_zeroth":bg_zeroth,"co_zeroth":co_zeroth,
      "y1":y1,"dlim1L":dlim1L,"dlim1U":dlim1U,"sp_first": sp_first, "bg_first": bg_first, "co_first": co_first,
      "y2":y2,"dlim2L":dlim2L,"dlim2U":dlim2U,"sp_second":sp_second,"bg_second":bg_second,"co_second":co_second,
      "y3":y3,"dlim3L":dlim3L,"dlim3U":dlim3U,"sp_third": sp_third, "bg_third": bg_third, "co_third":co_third,
      "x":x,"xstart":xstart,"xend":xend,"sp_all":sp_all,"quality":quality,"co_back":co_back,
      "apercorr":apercorr,"expospec":expospec,"templating":True})
        
        self.spResult.update({"Yfit":spYfit})          
        
        #===

        self.yloc_sp=fitorder_S[2][0][0]
        if self.chatter > 2: 
            print(f"from fitorder_S:yloc={self.yloc_sp}  \n from anchor ={self.ank_c_sximg}\n same ???") 
        offsp = self.ank_c_tximg[0]  # update   
        print (f"offsp = {offsp} from {self.ank_c_tximg}") 
        # extract the template spectrum using the matched image tximg
        trcmem = uvotgetspec.trackcentroiding
        uvotgetspec.trackcentroiding = False
        new_ank_c=[self.ank_c_tximg[0],self.ank_c_tximg[1],self.tmplResult['ank_c'][2],self.tmplResult['ank_c'][3]]
        self.tmplResult['ank_c']=new_ank_c
        #  We could now use the Ysp parameters to directly extract the counts for tximg
        #  but here we rerun curved_extraction which should have the same parameters 
        #  like borderup/down
        
        self.Ytmpl = uvotgetspec.curved_extraction(        # quick draft
           self.tximg2, 
           self.ank_c_tximg, 
           self.spAnker, # anker position on det to retrieve coefficients for track
           self.tmplResult['wheelpos'], 
           expmap=self.tmplResult['expmap'], offset=0., 
           anker0=None, anker2=None, anker3=None, angle=None, 
           offsetlimit=[offsp,0.5],   # match to sximg extraction offset
           background_lower=bglo, 
           background_upper=bghi,
           background_template=None, #self.template,
           trackonly=False, 
           trackfull=False,  
           caldefault=True, 
           curved="noupdate", \
           poly_1=None,poly_2=None,poly_3=None, 
           set_offset=False, 
           composite_fit=True, 
           test=None, chatter=0, 
           skip_field_sources=True,\
           predict_second_order=False, 
           ZOpos=None,
           outfull=True,   # check what is needed by I/O module
           msg='',
           fit_second=False,
           fit_third=False,
           C_1=self.tmplResult['C_1'] ,C_2=None,dist12=None,
           dropout_mask=None)
        uvotgetspec.trackcentroiding = trcmem   
                   
        fitorder_t, cp2_t, (coef0_t,coef1_t,coef2_t,coef3_t), (bg_zeroth,bg_first, bg_second,bg_third), \
        (borderup_t,borderdown_t), apercorr, expospec, msg_t, curved_t = self.Ytmpl   
                
        (present0,present1,present2,present3),(q0,q1,q2,q3), (
              y0,dlim0L,dlim0U,sig0coef,sp_zeroth,co_zeroth),(
              y1,dlim1L,dlim1U,sig1coef,sp_first,co_first),(
              y2,dlim2L,dlim2U,sig2coef,sp_second,co_second),(
              y3,dlim3L,dlim3U,sig3coef,sp_third,co_third),(
              x,xstart,xend,sp_all,quality,co_back)  = fitorder_t
        
        bg, bg1, bg2, bgsig, bgimg, bg_limits, (bg1_good, bg1_dis, bg1_dis_good, 
        bg2_good, bg2_dis, bg2_dis_good,  bgimg_lin) = self.tmplBackground
        
        tmplYfit={"coef0":coef0_t,"coef1":coef1_t,"coef2":coef2_t,"coef3":coef3_t} 
       
        self.tmplResult.update({"background_1":bg1,"background_2":bg2})

        tmplYfit.update({
      "bg_zeroth":bg_zeroth,"bg_first":bg_first,"bg_second":bg_second,"bg_third":bg_third,
      "borderup":borderup_t,"borderdown":borderdown_t,
      "sig0coef":sig0coef,"sig1coef":sig1coef,"sig2coef":sig2coef,"sig3coef":sig3coef,
      "present0":present0,"present1":present1,"present2":present2,"present3":present3,
      "q0":q0,"q1":q1,"q2":q2,"q3":q3,
      "y0":y0,"dlim0L":dlim0L,"dlim0U":dlim0U,"sp_zeroth":sp_zeroth,"bg_zeroth":bg_zeroth,"co_zeroth":co_zeroth,
      "y1":y1,"dlim1L":dlim1L,"dlim1U":dlim1U,"sp_first": sp_first, "bg_first": bg_first, "co_first": co_first,
      "y2":y2,"dlim2L":dlim2L,"dlim2U":dlim2U,"sp_second":sp_second,"bg_second":bg_second,"co_second":co_second,
      "y3":y3,"dlim3L":dlim3L,"dlim3U":dlim3U,"sp_third": sp_third, "bg_third": bg_third, "co_third":co_third,
      "x":x,"xstart":xstart,"xend":xend,"sp_all":sp_all,"quality":quality,"co_back":co_back,
      "apercorr":apercorr,"expospec":expospec,"templating":True})
        
        self.tmplResult.update({"Yfit":tmplYfit})          
    
        
        # write output
        
        #h2=self.tmplResult['hdr']
        #h2.add_history(f
        proccessing_note=\
        "The templating code has written out two files which match the template  "+\
        "to the original, first by adjusting the roll anlge (PA_PNT) difference, "+\
        "and next by shifting the template to match the sky area of the original "+\
        "observation. Only the part of the image where these match will be "+\
        "retained, be it with some wrapping at the bordes of the template image "+\
        "due to the shifting needed. The spectrum extraction parameters of the "+\
        "original are used on the adjusted template image to extract the spectrum"+\
        "from the same region on the sky in both original and template. If the "+\
        "roll angles differ much the first orders if causing overlap will not be"+\
        "converted correctly. In that case no good solution can be found."
        #self.tmplResult['hdr'] = h2
        # first update fitourder in "Yout, etc..." in spResult ,spResult['eff_area1'] should be populated.
        
        if self.outputfile_stub != None:
            filestub = outfile1 = self.outputfile_stub+"_spectrum."
        else:   
            filestub = outfile1 = "uvottemplating_spect.output"
        use_lenticular_image=True
        F = uvotio.writeSpectrum(self.pos.ra.deg,self.pos.dec.deg,filestub,
              self.extsp, self.spResult,  
              fileoutstub=outfile1, 
              arf1=None, arf2=None, 
              fit_second=False, 
              write_rmffile=False, fileversion=2,
              used_lenticular=use_lenticular_image,
              history=msg_S, 
              calibration_mode=uvotgetspec.calmode, 
              chatter=self.chatter, 
              clobber=True ) 
              
        if self.outputfile_stub != None:
            filestub = outfile2 = self.outputfile_stub+"_template."
        else:   
            filestub = outfile2 = "uvottemplating_templ.output"
        F = uvotio.writeSpectrum(self.pos.ra.deg,self.pos.dec.deg,outfile2,
              self.exttempl, self.tmplResult,  
              fileoutstub=outfile2, 
              arf1=None, arf2=None, 
              fit_second=False, 
              write_rmffile=False, fileversion=2,
              used_lenticular=use_lenticular_image,
              history=msg_t, 
              calibration_mode=uvotgetspec.calmode, 
              chatter=self.chatter, 
              clobber=True ) 

        #xx = self.extract_spectrum(background_template=self.template,wr_outfile=True,
        #    interactive=True, plotit=True) does not work, requires whole image
         
    def rotate_tmpl(self,theta=None,extimg=None):
        import scipy.ndimage as ndimage
        import os
        
        if theta != None:    # use ext-image
            revising = True
            anker = self.anchor_templimg
            cval = self.cval
            img = extimg
        else: 
            revising = False  # use pha file
            anker = self.tmplResult['anker']
            theta = +1*(self.tmpl_roll-self.spec_roll)
           
            if self.chatter > 1: print (f"Rotating template image by {theta} deg over anchor.")   
        
        # backup the template input before changes are made
        # should check if this was already done
            os.system(f"cp {self.templates[self.spectrum_number]} {self.templates[self.spectrum_number]}_ori")
            if self.chatter > 2:
               print (f"copied original {self.templates[self.spectrum_number]} to {self.templates[self.spectrum_number]}_ori")
               print (f"opening {self.templates[self.spectrum_number]}")
           
            with fits.open(f"{self.templates[self.spectrum_number]}",update=True) as ft:
            # check if updated roll angle difference already done earlier (pa_update)
               hdr = ft[self.exttempl].header
               cval = self.cval
               try: 
                  pa_update = hdr['PA_UPDAT'] 
                  if self.chatter > 1: print (f"pa_update read from header")
               except:
                  pa_update = 0.
                  if self.chatter > 1: print (f"problem reading pa_update from header: set to zero")
               img = ft[self.exttempl].data   
              
        # rotate img
        if self.chatter > 1: print (f"rotating template - pivot over anker")
        
        s1 = 0.5*img.shape[0]
        s2 = 0.5*img.shape[1]

        d1 = -(s1 - anker[1])   # distance of anker to centre img 
        d2 = -(s2 - anker[0])
        n1 = 2.*abs(d1) + img.shape[0] + 400  # extend img with 2.x the distance of anchor 
        n2 = 2.*abs(d2) + img.shape[1] + 400

        if 2*int(n1/2) == int(n1): n1 = n1 + 1
        if 2*int(n2/2) == int(n2): n2 = n2 + 1
        c1 = n1 / 2 - anker[1] 
        c2 = n2 / 2 - anker[0]
        n1 = int(n1)
        n2 = int(n2)
        c1 = int(c1)
        c2 = int(c2)
        if self.chatter > 3: print('array info : ',img.shape,d1,d2,n1,n2,c1,c2)
   
        #  the ankor is now centered in array a; initialize a with out_of_img_val
        a  = np.zeros( (n1,n2), dtype=float) + cval
        # load array in middle
        a[c1:c1+img.shape[0],c2:c2+img.shape[1]] = img
      
        # patch outer regions with something like mean to get rid of artifacts
        mask = abs(a - cval) < 1.e-8
        # Kludge:
        # test image for bad data and make a fix by putting the image average in its place
        dropouts = False
        aanan = np.isnan(a)          # process further for flagging
        aagood = np.isfinite(a)
        aaave = a[np.where(aagood)].mean()
        a[np.where(aanan)] = aaave
        if len( np.where(aanan)[0]) > 0 :
            dropouts = True
        print("extractSpecImg WARNING: BAD IMAGE DATA fixed by setting to mean of good data whole image ") 
        
        img2 = ndimage.rotate(a,theta,reshape = False,order = 1,mode = 'constant',cval = cval)
        if not revising:
           # now revert to the original size
           ft[self.exttempl].data = img2[c1:c1+img.shape[0],c2:c2+img.shape[1]] 
           ft[self.exttempl].header["COMMENT"]=f"ROTATED by {theta}"
           hdr["comment"] = f"rotated the image over an angle {theta} deg"
           hdr["PA_UPDAT"] = pa_update+theta
           ft[self.exttempl].header = hdr
           ft.close()
        else:   # now we need to get the revised parameters 
            self.templimg = img
            self.tximg = img
        if self.chatter > 2: print ("rotation of template completed")
           
           
           

    
    def yloc_spectrum2(self):
        """
        quick draft
        This is input to curved_extraction of spimg, using template after matching, scaling, etc.
        """
        net = self.spimg - self.template
        # define range where spectrum is 
        if self.redshift == None:
            x1 = self.spResult['ank_c'][1] - self.dimsp[0]
            x2 = np.min([ self.dimsp[1], 600+x1])
            x1 = np.max([x1-200, 0 ])
        else:
            # find where spectrum starts
            wbreak = 912.*(1+self.redshift)
            disp = self.spResult['C_1']
            x1 = uvotgetspec.pix_from_wave( disp, wbreak)
            # and ends
            x2 = self.dimsp[1]
        fsum = net[:,x1:x2].sum(1)
        # now find the  y-peak in fsum 
        from scipy.signal import find_peaks
        cont = fsum.std()
        peaks = find_peaks(fsum,cont)
        # this needs testing... 
        self.yloc_sp = 100 # placeholder (see uvotspec.peakfinder
        self.widthsp = 15  # pix width for optimal extraction
    

    def extract_spectrum(self,background_template=None,wr_outfile=False,
          interactive=False,plotit=False):   # needs getspec params.
        
        if self.chatter>0: print (f"getting paprameters of the spectrum ")  
        # run uvotgetspec.getSpec() -> self.spectrum, *_exp, *_bkg, ...
        self.spResult=uvotgetspec.getSpec(self.pos.ra.deg,self.pos.dec.deg,self.obsidsp, self.extsp, 
          indir=self.indir+self.obsidsp+"/uvot/image/", wr_outfile=wr_outfile, 
          outfile=None, calfile=None, fluxcalfile=None, use_lenticular_image=True,
          offsetlimit=None, anchor_offset=None, anchor_position=[None,None],
          background_lower=self.background_lower, background_upper=self.background_upper, 
          background_template=background_template, fixed_angle=None, spextwidth=13, curved="update",
          fit_second=False, predict2nd=True, skip_field_src=False,      
          optimal_extraction=False, catspec=None,write_RMF=uvotgetspec.write_RMF,
          get_curve=None,fit_sigmas=True,get_sigma_poly=False, 
          lfilt1=None, lfilt1_ext=None, lfilt2=None, lfilt2_ext=None,  
          wheelpos=None, interactive=interactive, sumimage=None, set_maglimit=None,
          plot_img=plotit, plot_raw=plotit, plot_spec=plotit, zoom=True, highlight=False, 
          uvotgraspcorr_on=True, update_pnt=True, clobber=False, chatter=self.chatter ) 
           
        self.spimg=self.spResult['extimg']
        self.spAnker = self.spResult['anker']
        self.spfilename=self.spResult['grismfile'] 
        hdr=self.spResult["hdr"]
        self.spec_exp= hdr['exposure']
        self.spec_roll= hdr['PA_PNT']
        anky,ankx,xstart,xend = ank_c= self.spResult['ank_c']
        self.anchor_specimg = [ankx,anky]
        # now find the uv first order start and end locations -> dimsp
        self.dimsp = dimL,dimu = self.set_dims(xstart,xend)
        
        bg, bg1, bg2, bgsig, bgimg, bg_limits, \
          (bg1_good, bg1_dis, bg1_dis_good, bg2_good, bg2_dis, bg2_dis_good,  bgimg_lin) =\
        self.spBackground = uvotgetspec.findBackground(self.spimg,background_lower=[None,None], 
           background_upper=[None,None],yloc_spectrum=anky, chatter=0)
        self.spec_bkg  = bgimg
      
    def extract_template(self,):
        # run uvotgetspec.getSpec() -> self.template
        """
         extimg = extracted image
         ank_c = array( [ X pos anchor, Y pos anchor, start position spectrum, end spectrum]) in extimg
         anchor1 = anchor position in original image in det coordinates
        """
        if self.chatter>0: print (f"getting paprameters of the template spectrum ")  

        self.tmplResult=uvotgetspec.getSpec(self.pos.ra.deg,self.pos.dec.deg,self.obsidtempl, self.exttempl, 
          indir=self.indir+self.obsidtempl+"/uvot/image/", wr_outfile=False, 
          outfile=None, calfile=None, fluxcalfile=None, use_lenticular_image=True,
          offsetlimit=self.offsetlimit, anchor_offset=None, anchor_position=[None,None],
          background_lower=self.background_lower, background_upper=self.background_upper, 
          background_template=None, fixed_angle=None, spextwidth=13, curved="update",
          fit_second=False, predict2nd=True, skip_field_src=False,      
          optimal_extraction=False, catspec=None,write_RMF=uvotgetspec.write_RMF,
          get_curve=None,fit_sigmas=True,get_sigma_poly=False, 
          lfilt1=None, lfilt1_ext=None, lfilt2=None, lfilt2_ext=None,  
          wheelpos=None, interactive=False, sumimage=None, set_maglimit=None,
          plot_img=False, plot_raw=False, plot_spec=False, zoom=True, highlight=False, 
          uvotgraspcorr_on=True, update_pnt=True, clobber=False, chatter=self.chatter )
        self.templimg = extimg = self.tmplResult['extimg']
        
        hdr=self.tmplResult["hdr"]
        self.tmplfilename=self.tmplResult['grismfile'] 
        self.tmpl_roll= hdr['PA_PNT']
        anker = self.tmplResult['anker']
        offset = self.tmplResult['offset']
        ank_c = self.tmplResult['ank_c']
        self.templ_exp = hdr['exposure']
        anky,ankx,xstart,xend = ank_c= self.tmplResult['ank_c']
        self.anchor_templimg = [ankx, anky]
        # now find the uv first order start and end locations -> dimtempl
        self.dimtempl = dimL,dimu = self.set_dims(xstart,xend)
        
        bg, bg1, bg2, bgsig, bgimg, bg_limits, \
          (bg1_good, bg1_dis, bg1_dis_good, bg2_good, bg2_dis, bg2_dis_good,  bgimg_lin) \
        = self.tmplBackground = uvotgetspec.findBackground(extimg,background_lower=[None,None], 
           background_upper=[None,None],yloc_spectrum=anky, chatter=0)
        self.templ_bkg  = bgimg
            
    def set_dims(self,xstart,xend):
        # length of first order with respect to ank_c[1 ? 
        dlim1L=-400
        dlim1U=3000 #1150
        if (xstart > dlim1L): dlim1L = xstart
        if (xend < dlim1U): dlim1U = xend
        if self.chatter>2: print (f"this spectrum is located at pixels {xstart} to {xend}")
        return dlim1L,dlim1U
        
    def scale_template(self,):   # obsolated aug 2024
        # first run extract_*, match, dragit
        x = self.template.copy()
        qbg = (x - 2.*self.templ_bkg[:,self.dimtempl[0]:self.dimtempl[1]]) < 0. 
        x[qbg == False] = self.templ_bkg[:,self.dimtempl[0]:self.dimtempl[1]][qbg == False] \
           * self.spec_exp/self.templ_exp # scale peaks
        x[qbg == True] = (self.spec_bkg[:,self.dimsp[0]:self.dimsp[1]]\
           / self.templ_bkg[:,self.dimtempl[0]:self.dimtempl[1]])[qbg == True]   # scale the background
        self.template = x
        
    def match_slice(self):  
        """
        operates on the extracted spectral slices 
        
        now determine where the spec and templ overlap (in x) with reference to their 
        anchors
        
        first run extract_spectrum and extract_template
        
        """
        # [x ,y] anchors 
        asp = self.anchor_specimg   # anker x, y
        atp = self.anchor_templimg
        blue = np.min([asp[0],atp[0]])  
        redsp = self.spimg.shape[1]-asp[0]
        redtm = self.templimg.shape[1]-atp[0]
        red = np.min([redsp,redtm])
        self.dimsp    = int(asp[0]-blue), int(asp[0]+red)
        self.dimtempl = int(atp[0]-blue), int(atp[0]+red)
        self.tximg = self.templimg[:,self.dimtempl[0]:self.dimtempl[1]]
        self.sximg = self.spimg [:,self.dimsp[0]   :self.dimsp[1]   ]
        self.ank_c_tximg=[asp[1], atp[0]-self.dimtempl[0], 0, self.dimtempl[1]-self.dimtempl[0] ]
        self.ank_c_sximg=[asp[1], asp[0]-self.dimsp[0],    0, self.dimsp[1]   -self.dimsp[0] ]        
        if self.chatter>0: print (f"the extracted spectrum and template match sp:{self.dimsp} and templ:{self.dimtempl}")


    def SelectShift(self,figno=42,spimg=None, tempimg1=None):
        """
        An *interactive* method to improve alignment of zeroth orders
        
        first run extract_*, match_slice
        
        The approach is to plot the spectrum with contours of the template zeroth orders.
        The user needs to point and click a zeroth order in the template contour and then 
        the corresponding one in the image. That gives a shift. Do this for three sets and
        then replot with the template shifted to judge overlap. 
    
        delxy, tempimg = SelectShift(figno=42,spimg=<path>,tempimg=<path>)
    
        The output gives the shift in pixels between the initial sp_img and 
        the template_img, and returns the aligned tempimg 
        
        if rotate_and_shift is True, then apply rotation if the selected points indicate 
        the need for that
    
        """
        import matplotlib.pyplot as plt
        import sys, time
        
        
        if self.chatter>-1: 
            print(f"Now manually try to match the spectrum and template to overlap. \n"+\
        "This will give the offset to use.\nMatch zeroth orders of some sources as good as you can.\n")
            

        # work arrays limited to the X-offset 
        if isinstance(self.spimg, np.ndarray):
            spimg=self.spimg[:,self.dimsp[0]:self.dimsp[1]]
        if isinstance(self.templimg, np.ndarray): 
            tempimg=self.templimg[:,self.dimtempl[0]:self.dimtempl[1]]
        else:
            tempimg = tempimg1[:,self.dimtempl[0]:self.dimtempl[1]].copy()
            
        fig = plt.figure(figno,figsize=[8,1.3])
        fig.clf()
        fig.set_facecolor('lightgreen')
        ax = fig.add_axes([0.03,0.1,0.94,0.87],)
        canvas = ax.figure.canvas 
        ax.set_title("start")   
        sp =  ax.imshow ( np.log(spimg-np.median(spimg)+0.01),alpha=1.0,cmap='gist_rainbow'  ) # ax.imshow(spimg)
        self.c = cont =  ax.contour( np.log(tempimg-np.median(tempimg)*2+0.06),colors='k',lw=0.5)# ax.contour(tempimg) 
        fig.show()
        delxy = 0,0
        done = False
        itry = 1
        delxs=[]
        delys=[]
        nover=9
        pairs = []
        print (f"= = = = = = = = = = = = = = = = = \nThe plot consists of the image of the spectrum and contours (black for the template)")
        print ("First select the zoom button (not the shift/scale one)\n zoom to the region around the first order spectrum ")
        print ("find a small zeroth order pair and click on the black contour center. The coordinates should appear in the terminal app.")
        print ("You may have to do this slowly, and wait for the first cross to appear. Then click the blue couterpart zeroth order.")
        print ("Now a second coordinate should be printer in the terminal, and a question if good (yes) not good (no, do again), or Q (quit)")
        print ("answer, then do the next set of zeroth orders. The first (black) takes 3x clicking, the second only one time ")
        print ("Once three points were selected, the offset will be applied to the template. If not happy, start over.")
        print ("= = = = = = = = = = = = = = = = = \n")

        while (itry < 4) & (nover>1) : # collect three differences    
            nover -=1    
 #           itry += 1
            print ("\nzoom to select region with 3 zeroth order sets then press button to continue ")
            fig.waitforbuttonpress() #timeout=20)
            while True:
                pts = []
                while (len(pts) < 2):
                    print (f'put cursor in image; then select 2 points with mouse, points number={itry} out of 3')
                    pt1 = fig.ginput(1, timeout=-1)[0]
                    ax.plot(pt1[0],pt1[1],'x',color='orangered',ms=15,mew=3)
                    ax.plot(pt1[0],pt1[1],'x',color='r',ms=15,mew=1)
                    fig.show()
                    print (f"template point {pt1}")
                    pt2 = fig.ginput(1, timeout=-1)[0]
                    ax.plot(pt2[0],pt2[1],'x',color='gold',ms=15,mew=3)  
                    ax.plot(pt2[0],pt2[1],'x',color='r',ms=15,mew=1)
                    fig.show()
                    print (f"spectrum point {pt2}")
                    time.sleep(1.5)
                    pts = np.asarray([pt1,pt2])
                ans = input (f"#={itry} - the selected values are \n{pts};\n correct ? (N mean break,Q=quit)")
                if (ans.strip() != ""):
                   if (ans.upper().strip()[0] == "N"): 
                       redo=True  
                   elif (ans.upper().strip() == 'Q'):
                       break
                   else:  
                       itry += 1
                       #print ("passed selection of a points")  
                       dx = pts[0][0]-pts[1][0]
                       dy = pts[0][1]-pts[1][1]
                       pairs.append([(pts[0][0],pts[0][1],0),(pts[1][0],pts[1][1],0)])
                       delxs.append(dx)     
                       delys.append(dy)
                       print (f"dx={dx}, dy={dy}\n") 
                       break               
                break
                
        print (f"three points collected delxs,delys: \n{delxs}\n{delys}")
        delxs=np.asarray(delxs)
        delys=np.asarray(delys)
        
        if self.rotate_and_shift: 
            pivot = list(self.tmplResult['anker'])
            pivot.append(0)
            print (f"pivot={pivot}\npairs=\n{pairs}\n")
            
            # split the delxs, delyx pais in rotation and shift
            rot,angle = self.rotate_points_around_pivot(pairs, pivot)
            # apply rotation template
            print (f"TBD apply rotation with angle {angle}") 
            ans = input("apply the angle and rotate the template ?")
            if (ans.strip() != ""):
                if (ans.upper().strip()[0] == "Y"): 
                    print ("here we apply the rotation over angle[2]")
                    self.rotate_tmpl(theta=angle[0],extimg=self.tempimg)
                elif (ans.upper().strip() == 'N'):
                    print ("no rotation over angle")
                else:
                    print ("your response has been ignored")  
            
                   
        if 1==1:    
                 delxs=np.asarray(delxs)
                 delys=np.asarray(delys)
                 delx = delxs.mean()
                 dely = delys.mean()
                 stdx = delxs.std()
                 stdy = delys.std()
                 if self.chatter > 2 : print ("measured offsets",delxs,delys)
                 print (f"the offset is {delx}+-{stdx},{dely}+-{stdy}")
                 print ("redrawing figure")
                 fig2=plt.figure(figsize=[8,1.3])
                 ax1 = fig2.add_axes([0.03,0.1,0.94,0.87],)
                 sp1 = ax1.imshow( np.log(spimg-np.median(spimg)+0.01),alpha=1.0,cmap='gist_rainbow'  ) # ax.imshow(spimg)
                 tempimg = np.roll(tempimg,-int(delx),axis=1)
                 tempimg = np.roll(tempimg,-int(dely),axis=0)
                 self.tximg2 = tempimg
                 print (f"shifted template; now plottting contours")
                 self.c = ax1.contour( np.log(tempimg-np.median(tempimg)*2+0.01),alpha=0.7,colors='k')
                 ax1.set_title("--- with measured shift ---")
                 fig2.show()
                 # the change in templ anker position due to the shift/roll: 
                 self.delx = delx
                 self.dely = dely
                 self.stdx = stdx
                 self.stdy = stdy

        #except:
        #    sys.stderr.write(f"some kind of error in process: #{itry} delxy={delxy} all={delxys} ")
        #    roll the array elements of tempimg to make them line up with the spimg (wrap-arounds)
        #   self.template = tempimg
        # document the shift
        #self.movexy = newsp.delx, newsp.dely
        #self.test_templ_shift=tempimg
        self.templimg=tempimg
        #if self.chatter>0: print (f"Great! the offset found is {self.movexy}\nIf this seems wrong, retry.")
        
    def set_parameter(self,parametername,value): 
        # eval() or exec() ?   
        # ... include 'self.' in the parameter name
        exec(f"self.{parametername} = {value}")
        
    def rotate_points_around_pivot(self, pairs, pivot):
        """
        Rotate points around a pivot point based on three pairs of coordinates.

        :param pairs: List of three pairs of coordinates. Each pair represents 
                  [before_rotation, after_rotation] where before_rotation and 
                  after_rotation are (x, y, z) tuples.
        :param pivot: Tuple of (x, y, z) representing the pivot point around which 
                  rotation occurs.
        :return: Rotation object from scipy.spatial.transform which can be used to 
             apply this rotation to other points.
             and
             angle in deg.
        """
        from scipy.spatial.transform import Rotation as R
        vectors = []
        for before, after in pairs:
            # Convert to vectors from pivot
            v1 = np.array(before) - pivot
            v2 = np.array(after) - pivot
        
            # Normalize vectors
            v1 = v1 / np.linalg.norm(v1)
            v2 = v2 / np.linalg.norm(v2)
        
            # Use vector rotation to find rotation axis and angle
            axis = np.cross(v1, v2)
            if np.linalg.norm(axis) == 0:  # If vectors are parallel
                if np.all(v1 == v2):  # If vectors are the same direction
                    continue  # No rotation needed
                else:  # Vectors are opposite
                    axis = np.array([1, 0, 0])  # Arbitrary perpendicular axis
                    angle = np.pi
            else:
                angle = np.arccos(np.dot(v1, v2))

            # Create rotation object
            rot = R.from_rotvec(angle * axis / np.linalg.norm(axis))
            vectors.append(rot)
    
        # Combine all rotations (assuming they are around the same pivot)
        if vectors:
            combined_rotation = R.concatenate(vectors)
            angle = combined_rotation.as_euler('xyz', degrees=True)[2] 
            return combined_rotation, angle
        else:
            return R.identity(), 0.  # Return identity rotation if no valid rotations found
            