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
# part of uvotpy (c) 2009-2023, 
# this code September 2023 N.P.M. Kuin

import numpy as np
from uvotpy import uvotspec, uvotgetspec

# get uvotSpec processed parameters dictionary:
uvotgetspec.give_new_result=True

class withTemplateBackground(object):
    """
    Use a later observation ("template") as replacement for the background
    extraction in the final resulting spectrum
    
    No accomodation made yet for summed spectra input (needs setting parameter in 
    extract_*
    
    The template and spectrum should be taken at the same roll angle and 
    close to the same detector location if clocked mode was used.
    
    """
    def __init__(self, spectra=[], templates=[], pos=None, extsp=1, obsidsp="",
        obsidtempl="", exttempl=1, chatter=1):
        # input parameters, note all spectra, templates to be PHA files
        self.spectra = spectra
        self.templates = templates
        self.pos = pos # [astropy coordinates] of source
        self.obsidsp=obsidsp
        self.obsidtempl=obsidtempl
        self.extsp = extsp
        self.exttempl = exttempl
        self.indir = "./"
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
        #self.specimg_aligned=None
        #self.templimg_aligned=None
        # spectral extraction parameters 
        self.offsetlimit=[100,0.2]
        self.background_lower=[None,None]
        self.background_upper=[None,None]
        # Prep so that there is just one spectrum and one teplate PHA file
        self.c = None # contour
        
    def auto_template(self,):    
        """
        template,Y = auto_template()
        
           run all steps in sequence
        1. sum spectra if needed, before using this
        2. run extract * to get headers, extracted image, ank_c, slice param.
           get *_exp for scaling; set *_bkg found near anchor; scale_factor
           create specimg, templimg from extracted image extension
        3. note anchor position in specimg, tempimg
        4. align 
        5. scale templ
        6. embed to get correctly sized template
        7. extract spectrum using template (writes output)
        8. return template array and full output Y
        
        """
        self.set_parameter('offsetlimit',[100,0.1]) # easier matching 
        self.extract_spectrum()
        self.extract_template()
        self.match_slice()
        self.dragit(spimg=self.spimg[self.dimsp[0]:self.dimsp[1]],
              tempimg=self.templimg[self.dimtempl[0]:self.dimtempl[1]])
        self.scale_template()
        self.embed_template() # match with spimg size
        # now extract the spectrum with the template as background:
        self.Y = uvotgetspec.curved_extraction(
           self.spimg, self.tmplResult['ank_c'], 
           anchor1, 
           wheelpos, expmap=None, offset=0., 
           anker0=None, anker2=None, anker3=None, angle=None, 
           offsetlimit=None,  
           background_lower=[None,None], 
           background_upper=[None,None],
           background_template=None,
           trackonly=False, 
           trackfull=False, 
           caldefault=True, 
           curved="noupdate", \
           poly_1=None,poly_2=None,poly_3=None, 
           set_offset=False, 
           composite_fit=True, 
           test=None, chatter=0, 
           skip_field_sources=False,\
           predict_second_order=True, 
           ZOpos=None,
           outfull=False, 
           msg='',
           fit_second=True,
           fit_third=True,
           C_1=None,C_2=None,dist12=None,
           dropout_mask=None)
        #xx = self.extract_spectrum(background_template=self.template,wr_outfile=True,
        #    interactive=True, plotit=True) does not work, requires whole image
        return self.template, Y 
        
    def embed_template(self,):
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
        
    def extract_spectrum(self,background_template=None,wr_outfile=False,
          interactive=False,plotit=False):   # needs getspec params.
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
        hdr=self.spResult["hdr"]
        self.spec_exp= hdr['exposure']
        anky,ankx,xstart,xend = ank_c= self.spResult['ank_c']
        self.anchor_specimg = ankx
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
        anker = self.tmplResult['anker']
        offset = self.tmplResult['offset']
        ank_c = self.tmplResult['ank_c']
        self.templ_exp = hdr['exposure']
        anky,ankx,xstart,xend = ank_c= self.tmplResult['ank_c']
        self.anchor_templimg = ankx
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
        return dlim1L,dlim1U
        
    def scale_template(self,):
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
        now determine where the spec and templ overlap (in x)
        
        first run extrac_*
        
        """
        #x anchors
        asp = self.anchor_specimg
        atp = self.anchor_templimg
        # roll templimg so that anchors match
        self.templimg = np.roll(self.templimg,int(asp-atp),axis=1)
        #dimensions in x
        sp1,sp2 = self.dimsp
        tm1,tm2 = self.dimtempl
        # ignore the wrap from the roll operation
        # 
        start = np.max([sp1,tm1])
        end = np.min([sp2,tm2])
        self.dimsp = start,end
        self.dimtempl = start,end
        
    def dragit(self,figno=42,spimg=None, tempimg=None):
        """
        first run extract_*, match_slice
    
        delxy, tempimg = dragit(figno=42,spimg=<path>,tempimg=<path>)
    
        The output gives the shift in pixels between the initial spimg and 
        the tempimg, and returns the aligned tempimg 
    
        """
        import matplotlib.pyplot as plt
        import sys
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
                    print ("draw from black to corresponding blue feature")
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
        self.template = tempimg
        self.movexy = newsp.delx, newsp.dely

    def set_parameter(self,parametername,value): 
        # eval() or exec() ?   
        # ... include 'self.' in the parameter name
        exec(f"self.{parametername} = {value}")

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
      
