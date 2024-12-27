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
from uvotpy import uvotspec, uvotgetspec
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
        obsidtempl="", exttempl=1, redshift=0.0, chatter=1):
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
        # process variables, parameters
        self.spResult=None
        self.tmplResult=None
        #self.summed_sp = False     # fit header is different for summed
        #self.summed_templ = False  # ditto, for spectrum, template
        self.specimg=None
        self.templimg=None
        self.spec_exp=50.
        self.templ_exp=0.
        self.spec_bkg=0.    # background found in first order
        self.templ_bkg=0.
        self.dimsp = (-400,1150)
        self.dimtempl = (-400,1150)
        self.bkg_scale=1.   # ratio backgrounds [-400,800] zero at anchor (interp1d)
        #self.ZO_scale=1.    # ratio exposure times
        #self.spectrum=None
        self.template=None
        self.anchor_templimg=[]
        self.anchor_specimg=[]
        self.movexy=0,0 # return from manual alignment
        self.yloc_sp = 100
        self.widthsp = 15  # pix width for optimal extraction
        #self.specimg_aligned=None
        #self.templimg_aligned=None
        # spectral extraction parameters 
        self.offsetlimit=[97,2]    # this may not work and at some point needs to match both
        self.background_lower=[None,None]
        self.background_upper=[None,None]
        # Prep so that there is just one spectrum and one teplate PHA file
        self.c = None # contour
        self.cval = -1.0123456789
        
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
        6. [future] if roll angles differ, flag parts of template spectrum which are 
           contaminated by first orders that rotated into the track we are 
           examining. Exclude first order parts unless it also contains a 
           zeroth order.   
        7. subtract the scaled template spectrum, which contains the zeroth orders 
           in the the background under the spectrum, from the spectrum
        
        obsolete methods: 
        #6. embed to get correctly sized template
        #7. extract spectrum using template (writes output)
        #8. return template array and full output Y
        
        
        
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
        # find difference in roll angle and rotate template to spectrum 
        self.rotate_tmpl()
        # find initial transform template => spectrum
        self.match_slice()
        self.dragit(spimg=self.spimg[self.dimsp[0]:self.dimsp[1]],
              tempimg=self.templimg[self.dimtempl[0]:self.dimtempl[1]])
        
        # now that we got the offset between spectrum and template, we 
        # first extract the spectrum properly, and then extract the spectrum 
        # for the template at a location to match that of the original spectrum.
        
        self.Ysp = uvotgetspec.curved_extraction(        # quick draft
           self.spimg[:,self.dimsp[0]:self.dimsp[1]], 
           self.spResult['ank_c'], 
           self.spResult['ank_c']-self.dimsp[0], # anker?? 
           self.spResult['wheelpos'], 
           expmap=self.spResult['exposure'], offset=0., 
           anker0=None, anker2=None, anker3=None, angle=None, 
           #offsetlimit=[self.yloc_sp,0.2],  <= perhaps we need this extra step for high z
           offsetlimit=[100,5],  
           background_lower=[None,None], 
           background_upper=[None,None],
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
           
        # extract the template spectrum for the correct area
        self.Ytmpl = uvotgetspec.curved_extraction(        # quick draft
           self.tmplimg[:,self.dimtmpl[0]:self.dimtmpl[1]], 
           self.tmplResult['ank_c'], 
           self.tmplResult['ank_c']-self.dimtmpl[0], # anker??
           self.tmplResult['wheelpos'], 
           expmap=self.tmplResult['exposure'], offset=0., 
           anker0=None, anker2=None, anker3=None, angle=None, 
           #offsetlimit=[self.yloc_sp,0.2],  <= perhaps we need this extra step for high z
           offsetlimit=[100,5],  
           background_lower=[None,None], 
           background_upper=[None,None],
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
        
        
           
        # now get the count rate spectrum   
        fitorder, cp2, (coef0,coef1,coef2,coef3), (bg_zeroth,bg_first, bg_second,bg_third), \
        (borderup,borderdown), apercorr, expospec, msg, curved = self.Y   
        # write output
        # first update fitourder in "Yout, etc..." in spResult ,spResult['eff_area1'] should be populated.
        outfile = "uvottemplating.output.pha"
        F = uvotio.writeSpectrum(RA,DEC,filestub,
              self.extsp, self.Y,  
              fileoutstub=outfile, 
              arf1=None, arf2=None, 
              fit_second=False, 
              write_rmffile=False, fileversion=2,
              used_lenticular=use_lenticular_image,
              history=self.spResult['msg'], 
              calibration_mode=uvotgetspec.calmode, 
              chatter=self.chatter, 
              clobber=self.clobber ) 

        #xx = self.extract_spectrum(background_template=self.template,wr_outfile=True,
        #    interactive=True, plotit=True) does not work, requires whole image
         
    def rotate_tmpl(self,):
        import scipy.ndimage as ndimage
        import os
        
        theta = self.tmpl_roll-self.spec_roll
        anker = self.anchor_templimg
        
        # backup the template input before changes are made
        # should check if this was already done
        os.system(f"cp {self.templates[self.spectrum_number]} {self.templates[self.spectrum_number]}_ori")
        if self.chatter > 2:
           print (f"copied original {self.templates[self.spectrum_number]} to {self.templates[self.spectrum_number]}_ori")
           print (f"opening {self.templates[self.spectrum_number]}")
        with fits.open(f"{self.templates[self.spectrum_number]}",update=True) as ft:
        # check if updated roll angle difference already
           hdr = ft[self.exttempl].header
           cval = self.cval
           try: 
              pa_update = hdr['pa_updated'] 
              if self.chatter > 1: print (f"pa_update read from header")
           except:
              pa_update = 0.
              if self.chatter > 1: print (f"problem reading pa_update from header: set to zero")
           # rotate img
           if self.chatter > 1: print (f"rotating template ")
           img = ft[self.exttempl].data   
        
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
           # now revert to the original size
           ft[self.exttempl].data = img2[c1:c1+img.shape[0],c2:c2+img.shape[1]] 
           ft[self.exttempl].header["COMMENT"]=f"ROTATED by {theta}"
           ft.close()
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
          offsetlimit=self.offsetlimit, anchor_offset=None, anchor_position=[None,None],
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
        self.spfilename=self.spResult['grismfile'] 
        hdr=self.spResult["hdr"]
        self.spec_exp= hdr['exposure']
        self.spec_roll= hdr['PA_PNT']
        anky,ankx,xstart,xend = ank_c= self.spResult['ank_c']
        self.anchor_specimg = [ankx,anky]
        self.dimsp = dimL,dimu = self.set_dims(xstart,xend)
        
        bg, bg1, bg2, bgsig, bgimg, bg_limits, \
          (bg1_good, bg1_dis, bg1_dis_good, bg2_good, bg2_dis, bg2_dis_good,  bgimg_lin) \
           = uvotgetspec.findBackground(self.spimg,background_lower=[None,None], 
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
        self.dimtempl = dimL,dimu = self.set_dims(xstart,xend)
        bg, bg1, bg2, bgsig, bgimg, bg_limits, \
          (bg1_good, bg1_dis, bg1_dis_good, bg2_good, bg2_dis, bg2_dis_good,  bgimg_lin) \
           = uvotgetspec.findBackground(extimg,background_lower=[None,None], 
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
        operates on the extracted spectral slice
        
        now determine where the spec and templ overlap (in x)
        
        first run extract_spectrum and extract_template
        
        """
        # [x ,y] anchors 
        asp = self.anchor_specimg
        atp = self.anchor_templimg
        
        # shift templimg along x-axis so that anchors match < removed shift
        #    self.templimg = np.roll(self.templimg,int(asp[0]-atp[0]),axis=1)
        
        #dimensions in x
        sp1,sp2 = self.dimsp
        tm1,tm2 = self.dimtempl
        # ignore the wrap from the "roll" operation
        # 
        start = np.max([sp1,tm1])
        end = np.min([sp2,tm2])
        self.dimsp = start,end
        self.dimtempl = start,end
        if self.chatter>0: print (f"the extracted spectrum and template match in range {start}:{end}")
        
    def dragit(self,figno=42,spimg=None, tempimg=None):
        """
        An interactive method to improve alignment
        
        first run extract_*, match_slice
    
        delxy, tempimg = dragit(figno=42,spimg=<path>,tempimg=<path>)
    
        The output gives the shift in pixels between the initial sp_img and 
        the template_img, and returns the aligned tempimg 
    
        """
        import matplotlib.pyplot as plt
        import sys
        
        if self.chatter>-1: 
            print(f"Now manually try to match the spectrum and template to overlap. \n"+\
        "This will give the offset to use.\nMatch zeroth orders of some sources as good as you can.\n")
        
        # work arrays limited to the X-offset 
        if isinstance(self.spimg, np.ndarray):
            spimg=self.spimg[:,self.dimsp[0]:self.dimsp[1]]
        if isinstance(self.templimg, np.ndarray): 
            tempimg=self.templimg[:,self.dimtempl[0]:self.dimtempl[1]]
            
        fig = plt.figure(figno,figsize=[10,3])
        fig.clf()
        fig.set_facecolor('lightgreen')
        ax = fig.add_axes([0.03,0.1,0.94,0.87],)
        canvas = ax.figure.canvas 
        ax.set_title("start")   
        sp =  ax.imshow ( np.log(spimg-np.median(spimg)+0.01),alpha=1.0,cmap='gist_rainbow'  ) # ax.imshow(spimg)
        self.c = cont =  ax.contour( np.log(tempimg-np.median(tempimg)*2+0.06),colors='k',lw=0.5)# ax.contour(tempimg) 
        fig.show()
        newsp = DraggableContour(ax,cont)
        fig.show()
        delxy = 0,0
        try:
            ans1 = input("Do you want to adjust ? (Y/N) ").upper()
            print("answer read = ", ans1," length = ", len(ans1))
            if len(ans1) > 0:
              if ans1.upper().strip()[0] == 'Y':
                 done = False
                 while not done:
                    print('drag the contour spectrum until match and happy')
                    ax.set_title(f"... when done press key ...")   
                    newsp.connect()
                    print ("connected")
                    print ("draw now black contour to corresponding blue feature, then answer ")
                    #delxy += newsp.out_delxy()
                    ans = input("update contour?\n\n")
                    if ans.upper().strip()[0] == 'Y':
                        # update templimg
                        newsp.disconnect()
                        ax.cla()
                        delxy += newsp.out_delxy()
                        print(f"The selected shift is {newsp.delx},{newsp.dely} and will be applied when done. ") 
                        tempimg = np.roll(tempimg,int(newsp.delx),axis=1)
                        tempimg = np.roll(tempimg,int(newsp.dely),axis=0)
                        print (f"changed templ img with shifts {tempimg.shape}; now plottting")
                        sp =  ax.imshow ( np.log(spimg-np.median(spimg)+0.01),alpha=1.0,cmap='gist_rainbow'  ) # ax.imshow(spimg)
                        self.c = ax.contour( np.log(tempimg-np.median(tempimg)*2+0.01),alpha=0.9,colors='k')
                        ax.set_title("done")
                        ax.show()
                        done = True
                 newsp.disconnect()
              elif ans1.upper().strip()[0] == 'N': 
                 done = True
              else: print(" answer Y or N ")
        except:
            sys.stderr.write(f"drag error: {delxy} ")
            newsp.disconnect()
        # roll the array elements of tempimg to make them line up with the spimg (wrap-arounds)
        # print update 
        #   self.template = tempimg
        # document the shift
        self.movexy = newsp.delx, newsp.dely
        if self.chatter>0: print (f"Great! the offset found is {self.movexy}\nIf this seems wrong, retry.")
        
    def set_parameter(self,parametername,value): 
        # eval() or exec() ?   
        # ... include 'self.' in the parameter name
        exec(f"self.{parametername} = {value}")
        
        
    def embed_template(self,):   # obsoleted Aug 2024
        sbgimg = self.spec_bkg
        sanky,sankx,sxstart,sxend = self.spResult['ank_c']
        tanky,tankx,txstart,txend = ank_c= self.spResult['ank_c']
        sdim = self.dimsp #should be same as:
        tdim = self.dimtempl
        # match anchors - this should have been done alraidy 
        # da = sankx - tankx # how the anchers are shifted in spimg/bgimg and tmplimg
        # find limits x1,x2 for drop-in
        # so typically, x1 = sankx-sdim[0] for start embedding ,x2=tdim[1]-tdim[0]+x1 for length 
        x1 = int(sdim[0]) # bed
        a1 = int(0)       # templ
        a2 = int( np.min([sdim[1]-sdim[0],sxend]) )  # crop temo, if extends too far
        x2 = a2 - a1 +x1 # must match length and offset
        print (f"x1={x1}, x2={x2} \n")
        sbgimg[:,x1:x2] = self.template[:,a1:a2]  
        # update template 
        self.template=sbgimg  
                
        

class DraggableContour(object):
    """
    Drag contour img1 on image img2 until correctly lined up 
    return shifts in x, y
    
    """
    import matplotlib as mpl
    
    def __init__(self, ax, contour):
        self.img1 = contour  # move contour over image
        self.press = None
        self.delx = 0.0
        self.dely = 0.0
        self.incx = 0.0
        self.incy = 0.0
        self.ax = ax
        self.cidpress = None
        self.cidrelease = None
        self.cidmotion = None
        self.cidkey = None
        self.startpos = [0,0]
        self.endpos = [0,0]

    def connect(self):
        'connect to all the events we need'
        self.cidpress = self.img1.axes.figure.canvas.mpl_connect(
            'button_press_event', self.on_press)
        self.cidrelease = self.img1.axes.figure.canvas.mpl_connect(
            'button_release_event', self.on_release)
        self.cidmotion = self.img1.axes.figure.canvas.mpl_connect(
            'motion_notify_event', self.on_motion)
        self.cidkey = self.img1.axes.figure.canvas.mpl_connect(
            'key_press_event', self.on_key)
        print("active")    

    def on_press(self, event):
        'on button press we will  store some data'
        if event.inaxes != self.img1.axes: return
        self.press = event.x, event.y, event.xdata, event.ydata #, self.img1.get_xdata(), self.img1.get_ydata()
        print("on_press start position (%f,%e)"%(event.xdata,event.ydata))
        self.startpos = [event.xdata,event.ydata]

    def on_motion(self, event):
        'on motion we will move the spectrum if the mouse is over us'
        if self.press is None: return
        if event.inaxes != self.img1.axes: return
        #x0, y0, xpress, ypress, xdata = self.press
        x0, y0, xpress, ypress = self.press
        dx = event.xdata - xpress
        dy = event.ydata - ypress
        self.incx = dx
        self.incy = dy
        #self.img1.set_xdata(xdata+dx) 
        '''
        # the following tried to modify the data arrays in the contour thing, 
        # but seems to fail to update...
        #
        xx = self.img1.collections
        nx = len(xx)
        for k in np.arange(nx):  # loop over collections
            xy = xx.pop() #xx[k]
            xz = xy.properties()['segments']
            ns = len(xz)
            for gs in np.arange(ns): # loop over segments
                y = xz[gs]
                y[:,0] += dx
                y[:,1] += dy
            a = xy.properties
            a().update({'segments':xz})
            # update xy 
            xy.properties = a
            xx.append(xy)
        # now we have replaced the data array with the +dx,+dy values.    
        self.img1.collections = xx
        self.img1.changed()   # this should do the update
        '''
        self.ax.figure.canvas.draw()

    def on_release(self, event):
        'on release we reset the press data'
        self.delx += self.incx
        self.dely += self.incy
        self.press = None
        self.ax.figure.canvas.draw()
        if event.inaxes == self.img1.axes:
            print("on_release end position (%f,%e)"%(event.xdata,event.ydata))
            self.endpos = [event.xdata,event.ydata]
            
    def on_key(self,event):
        'on press outside canvas disconnect '       
        print("you pushed the |%s| key"%event.key)
        print("ignoring ...")
        # retrieve out_delxy and then execute *.disconnect()

    def disconnect(self):
        print (f"position start = {self.startpos}, end = {self.endpos}")
        print (f"movement dx={self.startpos[0]-self.endpos[0]}, dy={self.startpos[1]-self.endpos[1]}")
        'disconnect all the stored connection ids'
        self.img1.axes.figure.canvas.mpl_disconnect(self.cidpress)
        self.img1.axes.figure.canvas.mpl_disconnect(self.cidrelease)
        self.img1.axes.figure.canvas.mpl_disconnect(self.cidmotion)
        self.img1.axes.figure.canvas.mpl_disconnect(self.cidkey)
        print("disconnected")
        
    def out_delxy(self):
        return self.delx,self.dely
      
