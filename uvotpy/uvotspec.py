#!/usr/bin/env python
# -*- coding: iso-8859-15 -*-
'''
   Work with UVOT spectra:
      - adjust the wavelengths 
      - flag bad quality data 
'''
# Developed by N.P.M. Kuin (MSSL/UCL)
__version__ = '20140424-0.0.2'

import sys
import numpy as np
import matplotlib.pyplot as plt
from stsci.convolve import boxcar
from astropy.io import fits
from matplotlib.lines import Line2D

# data
spdata = {
'HI':[
{'name':'Ly-alpha' ,'transition':'1s-2' ,'wavevac':1215.67,   'label':r'Ly$\alpha$'},
{'name':'Ly-beta'  ,'transition':'1s-3' ,'wavevac':1025.722,  'label':r'Ly$\beta$'},
{'name':'Ly-gamma' ,'transition':'1s-4' ,'wavevac':972.537,   'label':r'Ly$\gamma$'},
{'name':'Ly-limit' ,'transition':'1s-40','wavevac':912.3,     'label':r'Ly-limit'},
{'name':'H-alpha'  ,'transition':'2-3'  ,'wavevac':6564.63,   'label':r'H$\alpha$'},
{'name':'H-beta'   ,'transition':'2-4'  ,'wavevac':4862.69,   'label':r'H$\beta$'},
{'name':'H-gamma'  ,'transition':'2-5'  ,'wavevac':4341.69,   'label':r'H$\gamma$'},
{'name':'H-delta'  ,'transition':'2-6'  ,'wavevac':4102.899,  'label':r'H$\delta$'},
{'name':'H-epsilon','transition':'2-7'  ,'wavevac':3971.202,  'label':r'H$\epsilon$'},
{'name':'H-6'      ,'transition':'2-8'  ,'wavevac':3890.16,   'label':r'H6'},
{'name':'H-limit'  ,'transition':'2s-40','wavevac':3656,      'label':r'Ba-limit'},
{'name':'Pa-alpha' ,'transition':'3-4'  ,'wavevac':18756.096, 'label':r'Pa$\alpha$'},
{'name':'Pa-beta'  ,'transition':'3-5'  ,'wavevac':12821.576, 'label':r'Pa$\beta$'},
{'name':'Pa-gamma' ,'transition':'3-6'  ,'wavevac':10941.082, 'label':r'Pa$\gamma$'},
{'name':'Pa-delta' ,'transition':'3-7'  ,'wavevac':10052.123, 'label':r'Pa$\delta$'},
{'name':'Pa-5'     ,'transition':'3-8'  ,'wavevac':9548.587,  'label':r'Pa5'},
{'name':'Pa-limit' ,'transition':'3s-40','wavevac':8252.2,    'label':r'Pa-limit'},
     ],
'HeI':[
{'transition':'1s2p 3Po-1s3s 3S ','wavevac':7067.14   ,'label':u'HeI'},
{'transition':'1s2p 1Po-1s3d 1D ','wavevac':6679.9956 ,'label':u'HeI'},
{'transition':'1s2p 3Po-1s3d 3D ','wavevac':5877.249  ,'label':u'HeI'},
{'transition':'1s2s 1S -1s3p 1Po','wavevac':5017.0772 ,'label':u'HeI'},
{'transition':'1s2s 3Po-1s4d 4D ','wavevac':4472.735  ,'label':u'HeI'},
{'transition':'1s2s 3S -1s3p 3Po','wavevac':3889.75   ,'label':u'HeI'},
{'transition':'1s2s 3S -1s4p 3Po','wavevac':3188.667  ,'label':u'HeI'},
{'transition':'2p2  3P -2p3d 3Do','wavevac':3014.59   ,'label':u'HeI'},
{'transition':'1s2s 3S -1s5p 3Po','wavevac':2945.967  ,'label':u'HeI'},
{'transition':'1s2  1S -1s1p 1Po','wavevac':584.334   ,'label':u'HeI'},
     ],
'HeII':[
{'transition':'4 - 6','wavevac':6562.0, 'label':u'HeII'},
{'transition':'4 - 7','wavevac':5411.5, 'label':u'HeII'},# see J.D.Garcia and J.E. Mack,J.Opt.Soc.Am.55,654(1965)
{'transition':'3 - 4','wavevac':4687.1, 'label':u'HeII'},
{'transition':'3 - 5','wavevac':3203.95,'label':u'HeII'},
{'transition':'3 - 6','wavevac':2734.13,'label':u'HeII'},
{'transition':'3 - 7','wavevac':2511.2, 'label':u'HeII'},
{'transition':'3 - 8','wavevac':2385.4, 'label':u'HeII'},
{'transition':'2 - 3','wavevac':1640.47,'label':u'HeII'},
{'transition':'2 - 4','wavevac':1215.17,'label':u'HeII'},
{'transition':'2 - 6','wavevac':1025.30,'label':u'HeII'},
    ],
'nova':[ # add also H, HeI, HeII 
# 
{'transition':'','wavevac':1750  , 'label':u'NIII]'},
{'transition':'','wavevac':1908.7, 'label':u'CIII]'},
{'transition':'','wavevac':2143  , 'label':u'NII]'},
#{'transition':'','wavevac':2151.0, 'label':u'NIV]'},
{'transition':'','wavevac':2297  , 'label':u'CIII'},
{'transition':'','wavevac':2325.4, 'label':u'CII'},
{'transition':'','wavevac':2326.1, 'label':u'CII'},
{'transition':'','wavevac':2471.0, 'label':u'OII]'},
{'transition':'5D-3D','wavevac':2473, 'label':u'Ni IV]'},
{'transition':'5D-3D','wavevac':2522.5, 'label':u'Ni IV]'},
{'transition':'','wavevac':2796.4, 'label':u'MgII'},
{'transition':'','wavevac':2803.5, 'label':u'MgII'},
{'transition':'','wavevac':2937.4, 'label':u'MgII*'},
{'transition':'','wavevac':3130.0, 'label':u'OII*,OIII*,OIV*'},
{'transition':'','wavevac':3345.8, 'label':u'[NeV]'},
{'transition':'','wavevac':3425.9, 'label':u'[NeV]'},
{'transition':'','wavevac':3727  , 'label':u'[OIII]'},
{'transition':'','wavevac':3869, 'label':u'[NeIII]'},
{'transition':'','wavevac':3968, 'label':u'[NeIII]'},
{'transition':'','wavevac':4363  , 'label':u'[OIII]'},
{'transition':'','wavevac':4636  , 'label':u'NIII*'},
{'transition':'','wavevac':4643  , 'label':u'NIII*'},
{'transition':'','wavevac':4648.7, 'label':u'CIII*'},
{'transition':'','wavevac':4651.2, 'label':u'OIII*'},
{'transition':'','wavevac':4959  , 'label':u'[OIII]'},
{'transition':'','wavevac':5007  , 'label':u'[OIII]'},
{'transition':'','wavevac':5755  , 'label':u'[NII]'},
#{'transition':'','wavevac':.0, 'label':u''}
#{'transition':'','wavevac':.0, 'label':u''}
  ],

}
############################


class DraggableSpectrum:
    """
    Drag spectrum until the wavelengths are correctly lined up 
    """
    def __init__(self, ax, spectrum,):
        self.spectrum = spectrum
        self.press = None
	self.delwav = 0.0
	self.incwav = 0.0
	self.ax = ax

    def connect(self):
        'connect to all the events we need'
        self.cidpress = self.spectrum.figure.canvas.mpl_connect(
            'button_press_event', self.on_press)
        self.cidrelease = self.spectrum.figure.canvas.mpl_connect(
            'button_release_event', self.on_release)
        self.cidmotion = self.spectrum.figure.canvas.mpl_connect(
            'motion_notify_event', self.on_motion)
        self.cidkey = self.spectrum.figure.canvas.mpl_connect(
            'key_press_event', self.on_key)
	print "active"    

    def on_press(self, event):
        'on button press we will  store some data'
        if event.inaxes != self.spectrum.axes: return
        self.press = event.x, event.y, event.xdata, event.ydata, self.spectrum.get_xdata()
	print "start position (%f,%e)"%(event.xdata,event.ydata)

    def on_motion(self, event):
        'on motion we will move the spectrum if the mouse is over us'
        if self.press is None: return
        if event.inaxes != self.spectrum.axes: return
        x0, y0, xpress, ypress, xdata = self.press
        dx = event.xdata - xpress
	self.incwav = dx
        self.spectrum.set_xdata(xdata+dx) 
        self.ax.figure.canvas.draw()

    def on_release(self, event):
        'on release we reset the press data'
	self.delwav += self.incwav
        self.press = None
        self.ax.figure.canvas.draw()
	if event.inaxes == self.spectrum.axes:
	    print "end position (%f,%e)"%(event.xdata,event.ydata)
	    
    def on_key(self,event):
        'on press outside canvas disconnect '	    
        print "you pushed the |%s| key"%event.key
	print "disconnecting ..."

    def disconnect(self):
        'disconnect all the stored connection ids'
        self.spectrum.figure.canvas.mpl_disconnect(self.cidpress)
        self.spectrum.figure.canvas.mpl_disconnect(self.cidrelease)
        self.spectrum.figure.canvas.mpl_disconnect(self.cidmotion)
        self.spectrum.figure.canvas.mpl_disconnect(self.cidkey)
	print "disconnected"
	
    def out_delwav(self):
        return self.delwav
		

def adjust_wavelength_manually(file=None,openfile=None,openplot=None,
    ylim=[None,None],ions=['HI','HeII'],reference_spectrum=None,
    recalculate=False):
    """manually adjust the wavelength scale
    
    Parameters
    ----------
    file : path
       extracted spectral data (i.e., after running uvotgetspec.getSpec()
    fileopen : filehandle
       opened spectral data file
    openplot : axis       
       axis instance to use
    ylim : list(2)   
       list of length 2 with limits of the Y-axis or None
    ions : list
       list of ions to use for annotation 
       valid ions are spdata.keys()
    reference_spectrum : astropy.table.table.Table
       column 1: wavelength, column 2: flux  
    recalculate : bool
       when set, use wavelength shift to determine shift of pixno array 
       and use dispersion to recalculate the wavelengths    
       
    Notes
    -----
    The header will be updated with the value of the wavelength shift
    The wavelengths in the second extension lambda column will be shifted. 
    The response file will need to be recreated separately. 
    
    Returns the figure instance    
    
    """
    import uvotmisc
    # data
    if openfile != None:
       f = openfile 
       if f.fileinfo(1)['filemode'] != 'update' :
          print "reopening the fits file with mode set to update"
          filename = f.filename()
	  try: 
	     f.close()
	     f = fits.open(filename,mode='update')
	  except:
	     raise "reopen fits file with mode set to update, and rerun "
	          
    elif file != None:
       f = fits.open(file,mode='update')
       filename=file
    else:
       raise IOError("what ? nothing to adjust?")
    # axis instance to use   
    if openplot != None:
       fig = openplot
       fig.clf()
    else:
       fig = plt.figure()
    fig.set_facecolor('lightgreen')
    ax = fig.add_axes([0.05,0.13,0.9,0.77]) 
    canvas = ax.figure.canvas 
    ax.set_title("")   
    # initial plot to get started
    w = f[2].data['lambda']
    flx = f[2].data['flux']
    spectrum, = ax.plot(w, flx,color='b',label='spectrum to fix' )
    if reference_spectrum != None:
       refsp, = ax.plot(reference_spectrum['col1'],reference_spectrum['col2'],
            color='k',label='reference spectrum')
    # add annotation
    if ylim[0] == None: 
       ylim = ax.get_ylim()
    else:
       ax.set_ylim(ylim)  
    for io in ions:    
        plot_line_ids(ax,ylower=0.8*(ylim[1]-ylim[0])+ylim[0], ion=io) 
    ax.set_xlabel(u'wavelength($\AA$)')
    ax.set_ylabel(u'flux')
    ax.legend(loc=0)  
    ax.set_title(filename) 
    fig.show()
    print "Now is a good time to select a part of the figure to use for shifting the wavelengths."
    #  drag figure
    #background = canvas.copy_from_bbox(ax.bbox)
    newspec = DraggableSpectrum(ax,spectrum)
    done = False
    if 'WAVSHFT' in f[2].header:
        delwav0 = f[2].header['WAVSHFT']
	delwav = 0
    else:
        delwav0 = 0
	delwav = 0
    try:
        while not done:
            if raw_input("Do you want to adjust wavelengths ? (Y) ").upper()[0] == 'Y':
                print 'drag the spectrum until happy'
                ax.set_title("when done press key")   
                newspec.connect()
		print "The selected wavelength shift is ",newspec.delwav," and will be applied when done. " 
         	# register the shift from the last run        	
	        ans = raw_input("When done hit a key")
                delwav += newspec.out_delwav()
		ax.set_title("")
	        done = True
        newspec.disconnect()
	if recalculate:
	    print "recalculating wavelength scale after finding shift"
	    if 'PIXSHFT' in f[2].header: 
	        pixshift0 = f[2].header['PIXSHFT']
            else: pixshift0 = 0		
            C_1 = uvotmisc.get_dispersion_from_header(f[1].header)
            C_2 = uvotmisc.get_dispersion_from_header(f[1].header,order=2)
	    delpix = int(round(delwav / C_1[-2]))  # round to nearest int
	    pixno = f[2].data['pixno'] + delpix 
	    yes2ndorder = False
	    if 'pixno2'.upper() in f[2].data.names:
	       yes2ndorder = True
	       pixno2= f[2].data['pixno2']+ delpix 
  	    print "wavelength shift found = %f; which results in a pixno shift of %i"%(delwav,delpix)
	    f[2].data['pixno'] = pixno
	    if yes2ndorder: f[2].data['pixno2'] = pixno2 
	    f[2].data['lambda'] = np.polyval(C_1,pixno)
            f[2].header['PIXSHFT'] = (delpix+pixshift0, "pixno shift + recalc lambda from disp")
            f[1].header['PIXSHFT'] = (delpix+pixshift0, "pixno shift + recalc lambda from disp")
	    h = f[1].header['history']
	    if yes2ndorder:
	       dist12 = float(uvotmisc.get_keyword_from_history(h,'DIST12'))
	       f[2].data['lambda2'] = np.polyval(C_2,pixno2-dist12)
	    # now we should update the plot...
	else:
  	   sys.stderr.write( "wavelength shift found = %s\n"%(delwav) )
           f[2].header['WAVSHFT'] = (delwav+delwav0, "manual wavelength shift applied")
           f[2].data['LAMBDA'] = f[2].data['LAMBDA'] + delwav    
           f[1].header['WAVSHFT'] = (delwav+delwav0, "manual wavelength shift applied")
           f[1].header['COMMENT'] = "Manual wavelength shift not applied to response file."
        f.verify()
	f.flush()
	# replot 
	spectrum.set_color('c')
        f = fits.open(file)
        w = f[2].data['lambda']
        flx = f[2].data['flux']
        spectrum, = ax.plot(w, flx,color='darkblue',label='fixed spectrum' )
	ax.legend()
	ax.figure.canvas.draw()
    except:
        #sys.stderr.write("error: wavshift %f,pixshift0 %i,C_1 %s,C_2 %s\n"%(delwav,pixshift0,C_1,C_2) )
        raise RuntimeError("Some error occurred during the selection of the wavelength shift. No shift was applied.")
	newspec.disconnect()
    # apply the shift 
    return fig, ax, spectrum

def apply_shift(file,delwav,recalculate=False):
    """apply a given wavelength shift in A"""
    import uvotmisc
    f = fits.open(file,mode='update')
    delwav0 = 0
    if 'WAVSHFT' in f[2].header:
        delwav0 = f[2].header['WAVSHFT']+delwav
    if recalculate:
	if 'PIXSHFT' in f[2].header: 
	    pixshift0 = f[2].header['PIXSHFT']
	else: pixshift0 = 0    
        C_1 = uvotmisc.get_dispersion_from_header(f[1].header)
        C_2 = uvotmisc.get_dispersion_from_header(f[1].header,order=2)
	delpix = int(round(delwav / C_1[-2]))
	pixno  = f[2].data['pixno']  +delpix
	pixno2 = f[2].data['pixno2'] +delpix
	f[2].data['pixno'] = pixno  
	f[2].data['pixno2'] = pixno2  
	f[2].data['lambda'] = np.polyval(C_1,pixno)
        f[2].header['PIXSHFT'] = (delpix+pixshift0, "pixno shift + recalc lambda from disp")
        f[1].header['PIXSHFT'] = (delpix+pixshift0, "pixno shift + recalc lambda from disp")
	h = f[1].header['history']
	dist12 = float(uvotmisc.get_keyword_from_history(h,'DIST12'))
	f[2].data['lambda2'] = np.polyval(C_2,pixno2-dist12)
    else:	
        f[2].header['WAVSHFT'] = (delwav+delwav0, "manual wavelength shift applied")
        f[2].data['LAMBDA'] = f[2].data['LAMBDA'] + delwav    
        f[1].header['WAVSHFT'] = (delwav+delwav0, "manual wavelength shift applied")
    f.verify()
    f.flush()



class SelectBadRegions:
    """Select the bad regions on a spectrum interactively"""
    def __init__(self, ax, spectrum,badregions=[],eps=None,marker='^'):
        self.spectrum = spectrum
        self.yval = 0
	self.region = [0,0] # xstart and xend of bad region in data coordinates
	self.badregions = badregions # list of the Line2D for all bad regions (no check but type is matplotlib.lines.Line2D for each list element)
	self.line = Line2D([0,0],[0,0],marker=marker,color='k',lw=2,alpha=0.4,markerfacecolor='gold')
	self.ax = ax
	if eps == None:
            self.epsilon = (ax.get_xlim()[1]-ax.get_xlim()[0])/40 # angstrom
	else: self.epsilon = eps  
	self.marker=marker  
    
    def connect(self):
        'connect to all the events we need'
        self.cidpress = self.spectrum.figure.canvas.mpl_connect(
            'button_press_event', self.on_press)
        self.cidrelease = self.spectrum.figure.canvas.mpl_connect(
            'button_release_event', self.on_release)
        self.cidmotion = self.spectrum.figure.canvas.mpl_connect(
            'motion_notify_event', self.on_motion)
        self.cidkey = self.spectrum.figure.canvas.mpl_connect(
            'key_press_event', self.on_key)
	print "active"    

    def on_press(self, event):
        'on button press we check if near endpoint of region'
        if event.inaxes != self.spectrum.axes: return
	self.region[1] = None
	# check if near existing Line2D (adjust) 
	if len(self.badregions) > 0:
	    print "going through badregions"
	    for self.line in self.badregions:
	        xdat = self.line.get_xdata()
		#print "*** ",np.abs(xdat[0] - event.xdata)
		#print "*** ",np.abs(xdat[1] - event.xdata)
		#print "*** ",xdat
	        if (np.abs(xdat[0] - event.xdata) < self.epsilon) :
	            print "at point ",xdat[0]," keeping ",xdat[1]," swapping" 
		    k = self.badregions.index(self.line)
		    xx = self.badregions.pop(k)
		    self.line.set_xdata(np.array([xdat[1],event.xdata]))	      
	        elif (np.abs(xdat[1] - event.xdata) < self.epsilon):
	            print "at point ",xdat[1]," keeping ",xdat[0]
		    k = self.badregions.index(self.line)
		    xx = self.badregions.pop(k)
		    self.line.set_xdata(np.array([xdat[0],event.xdata]))	      
	        else:          
	            print  "new line"
	            self.yval = event.ydata
	            x0, y0, x1, y1 = event.xdata, self.yval, event.xdata, self.yval
	            self.line = Line2D([x0,x1],[y0,y1],marker=self.marker,color='k',lw=2,alpha=0.4,markerfacecolor='gold') 
	            self.ax.add_line(self.line)
	else:
	    # new line
	    self.yval = event.ydata
	    x0, y0, x1, y1 = event.xdata, self.yval, event.xdata, self.yval
	    self.line = Line2D([x0,x1],[y0,y1],marker=self.marker,color='k',lw=2,alpha=0.4,markerfacecolor='gold') 
	    self.ax.add_line(self.line)
	print "position [%f,*]"%(event.xdata,)

    def on_motion(self, event):
        'on motion we will move the spectrum if the mouse is over us'
        if self.region[1] is not None: return
        if event.inaxes != self.spectrum.axes: return
	xdat = self.line.get_xdata()
        xdat[-1] = event.xdata
        self.line.set_xdata(xdat) 
        self.ax.figure.canvas.draw()

    def on_release(self, event):
        'on release we reset the press data'
	if event.inaxes != self.spectrum.axes:
	    self.yval = None
	    self.region = [0,0]
	    return
        x1,y1 = event.xdata, event.ydata
	self.region[1] = event.xdata
	self.badregions.append(self.line)
        self.ax.figure.canvas.draw()
	if event.inaxes == self.spectrum.axes:
	    print "-> position (%f,%e)"%(event.xdata,event.ydata)
	    
    def on_key(self,event):
        'on press outside canvas disconnect '	    
        print "you pushed the |%s| key"%event.key

    def disconnect(self):
        'disconnect all the stored connection ids'
        self.spectrum.figure.canvas.mpl_disconnect(self.cidpress)
        self.spectrum.figure.canvas.mpl_disconnect(self.cidrelease)
        self.spectrum.figure.canvas.mpl_disconnect(self.cidmotion)
        self.spectrum.figure.canvas.mpl_disconnect(self.cidkey)
	print "disconnected"
	
    def get_badlines(self):
        lines = []
        for r in self.badregions: 
	    lines.append(r.get_xdata())
        return self.badregions, lines	
	
    def set_badregions(self,badregions):
        self.badregions = badregions	
        

def flag_bad_manually(file=None,openfile=None,openplot=None,
        ylim=[None,None], ):
    """manually flag bad parts of the spectrum
    
    Parameters
    ----------
    file : path
       extracted spectral data (i.e., after running uvotgetspec.getSpec()
    openfile : filehandle
    openplot : matplotlib.figure 
       figure handle
    ylim : list
       limits y-axis   
       
    Notes
    -----
    returns 
    ax:axes instance, fig:figure instance, [f:fits file handle if passed with openfile]
    
    The data quality flag of flagged pixels will be set to "bad"
    The header will be updated with the value of the wavelength shift 
    
    """
    from uvotgetspec import quality_flags
    if openfile != None:
       f = openfile
       if f.fileinfo(1)['filemode'] != 'update' :
          print "reopening the fits file with mode set to update"
          filename = f.filename()
	  try: 
	     f.close()
	     f = fits.open(filename,mode='update')
	  except:
	     raise "reopen fits file with mode set to update, and rerun "
    elif file != None:
       f = fits.open(file,mode='update')
    else:
       raise IOError("what ? nothing to adjust?")
    # axis instance to use   
    if openplot != None:
       fig = openplot
       fig.clf()
    else:
       fig = plt.figure()
    fig.set_facecolor('white')
    ax = fig.add_axes([0.08,0.13,0.87,0.7]) 
    canvas = ax.figure.canvas 
    ax.set_title("")   
    # initial plot to get started
    w = f[2].data['lambda']
    flx = f[2].data['flux']
    spectrum, = ax.plot(w, flx, )
    # highlight bad quality 
    q = f[2].data['quality']
    plotquality(ax,w,q,flag=['bad','zeroth','weakzeroth','overlap']) 
    # add annotation
    if ylim[0] == None: 
       ylim = ax.get_ylim()
    else:
       ax.set_ylim(ylim)  
    #for io in ions:    
    #    plot_line_ids(ax,ylower=0.8*(ylim[1]-ylim[0])+ylim[0], ion=io) 
    ax.set_xlabel(u'wavelength($\AA$)')
    ax.set_ylabel(u'flux')
    fig.show()
    print "Select bad regions: Zoom in before starting. Rerun for more regions."
    #  when animating / blitting figure
    #background = canvas.copy_from_bbox(ax.bbox)
    s = SelectBadRegions(ax,spectrum)
    s.set_badregions([])
    flag = quality_flags()
    done = False
    try:
        while not done:
            if raw_input("Do you want to mark bad regions ? (Y) ").upper()[0] == 'Y':
                print 'Select bad wavelengths in the spectrum until happy'
                ax.set_title("when done press key")   
                s.connect()
         	# register the shift from the last run        	
	        ans = raw_input("When done hit the d key, then return, or just return to abort")
		badregions, lines = s.get_badlines()
		print "got so far: "
                for br in lines: print "bad region : [%6.1f,%6.1f]"%(br[0],br[1])
		print badregions
		ax.set_title("")
                s.disconnect()
            else:		
	        done = True
    except:
        raise RuntimeError("Some error occurred during the selection of the bad regions. No changes were applied.")
	s.disconnect()
	lines = []
        #
    if len(lines) > 0:
        print "The selected bad regions are "
        for br in lines: print "bad region : [%6.1f,%6.1f]"%(br[0],br[1])
	print " and will be applied to the FITS file.\n " 
        f[2].header['comment'] = "added bad regions manually (qual=bad)"
	for br in lines:
	  #try:
	    # find points that are not flagged, but should be flagged
	    if br[1] < br[0]:
	       br3 = br[0]; br[0]=br[1]; br[1]=br3
            q1 = (check_flag(f[2].data['quality'],'bad')  == False)
	    q = ((f[2].data['lambda'] > br[0]) & 
		     (f[2].data['lambda'] < br[1]) & 
		     q1 & 
		     np.isfinite(f[2].data['quality']) )
            f[1].data['QUALITY'][q] = f[1].data['QUALITY'][q] + flag['bad'] 
            f[2].data['QUALITY'][q] = f[2].data['QUALITY'][q] + flag['bad']    
          #except:
            #    raise RuntimeError("Some error occurred during writing to file of the bad regions. No changes were applied.")
	    #    s.disconnect()
        f.verify()
	f.flush()
	print "file was updated"
    print type(f)	
    if file == None:	
       return fig, ax, f
    else:
       f.close() 
       return fig,ax   


def plot_line_ids(ax,ylower=None,ion='HI',color='k',dash=[0.07,0.10]):
   """add the line ids to the plot
   
   parameters
   ----------
   ax : plot handle
   ylower : float
     y-level where the bottom of the line should be
   ion : ['HI','HeI','HeII',]     
     key to the ion to be plotted
   """
   xlist = spdata[ion]
   xlim = ax.get_xlim()
   ylim = ax.get_ylim()
   dy = dash[0]*(ylim[1]-ylim[0])
   dy1 = dash[1]*(ylim[1]-ylim[0])
   
   wave = []
   for line in xlist:
      if (line['wavevac'] > xlim[0]) & (line['wavevac'] < xlim[1]):
          ax.text(line['wavevac'],ylower+dy1,line['label'],fontsize=8,color=color,
	      horizontalalignment='left',verticalalignment='center',
	      rotation='vertical' )
	  wave.append(line['wavevac'])
   ax.vlines(wave,ylower,ylower+dy,color='k') 	  
      
def plotquality(ax,w,quality,flag=['bad'],colors=['c','g','y','m','b','r','k'],alpha=0.2,):
       """add greyscale regions in plot for each quality flag 
       
       parameters
       ----------
       ax : matplotlib.axes.Axes instance
       w : array 
         x-axis values
       quality : array 
         quality flags matching x-axis points
       flag : list of strings
         each list value must be one of the valid keys from quality_flags()
       colors : array 
         color values	 	  
       alpha : float
         alpha value for transparency
       
       """
       from uvotgetspec import quality_flags
       flagdefs = quality_flags()
       k=0
       for fla in flag:
           fval = flagdefs[fla]
	   q = quality >= fval  # limit the search
	   indx = np.where(q)[0] # indx of wave where flag 
	   fval2 = fval*2
	   loc = quality[q]/fval2*fval2 != quality[q]
	   v = indx[loc] # indices of the points with this flag
	   if len(v) > 1:  # require at least 2
 	       vrange = []
	       v1 = v[0]
	       vlast = v1
	       for v2 in v[1:]: 
	           if v2-vlast > 1:  # require neighboring pixels
	               # 
		       vrange.append([v1,vlast])
		       v1 = v2
		       vlast = v2
	           else:
	               vlast=v2
	       if vlast > v1: vrange.append([v1,vlast])	 # last range       
	       print "for quality="+fla+" we get ranges ",vrange        
               for v1 in vrange:
	           ax.axvspan(w[v1[0]],w[v1[1]],facecolor=colors[k],alpha=alpha)
       # the algorithm skips two adjacent points which will be ignored. 		   
		   		      
def check_flag(quality,flag,chatter=0):
   """ return a logical array where the elements are 
       True if flag is set """
   from uvotgetspec import quality_flags
   loc = np.zeros(len(quality),dtype=bool) 
   if flag == 'good': return loc
   qf = quality_flags()[flag]
   mf=qf.bit_length() # bytes in binary string -2
   binflag = bin(qf) # string with binary flag
   qf = []
   for k in binflag: qf.append(k)
   qf.reverse()
   # find flag
   for i in range(0,mf): 
       if qf[i]=="1": 
           kpos=i
	   break
   if chatter> 4: print "flag = ",k," length=",mf	   	   
   # skip the good ones	   
   qz = np.where(quality > 0)[0]
   if chatter > 4: print "qual > 0 at ",qz
   if len(qz) == 0: 
      return loc
   for i in qz : 
       bq = int(quality[i]) # data quality point i
       mv = bq.bit_length() # binary length 
       binq = bin(bq)
       qf=[]
       for k in binq: 
          qf.append(k)
       qf.reverse()
       if mv < mf:
          break
       else:       
          if qf[kpos] == '1':
	      loc[i] = True
   return loc

def get_continuum_values(date,wave,flux,quality,cont_regions=[],qlimit=1):
    """give a list of good continuum bands in spectrum and 
    determine averages 
    
    parameters
    ----------
    wave : array
      wavelength array
    flux : array
      flux array  
    quality : array
      quality values
    qlimit : int
      include only quality less than this limit    
    cont_regions : list
      a list of [start, stop] wavelengths
    
    returns
    -------
    cont_list: list
       a list comprised of average wavelength, and continuum flux 
       value in each band 
       
    """
    import numpy as np
    if len(cont_regions) == 0: return
    result = [date]
    for r in cont_regions:
       q = (wave > r[0]) & (wave <= r[1]) & (quality < qlimit) & np.isfinite(flux)
       if len(np.where(q)[0]):
          result.append([wave[q].mean(),flux[q].mean(), flux[q].std()])
       else: 
          result.append([np.NaN,np.NaN,np.NaN])	  
    return result
    
def get_continuum(phafiles,regions=[],qlimit=1,tstart=0,daily=True,full=True):
    '''
    given a list of PHA spectra files and a list of (continuum)
    regions, extract the mean flux and error in the 
    (continuum) regions.
    
    parameters
    ----------
    cont_regions : list
      a list of [start, stop] wavelengths
    
    returns
    -------
    wavelength, mean flux+error in each region as an astropy table
    '''   
    import numpy as np
    from astropy.io import ascii,fits
    
    # build wavelengths array
    w = []
    for r in regions:
       w.append(np.asarray(r,dtype=float).mean())
    w = np.array(w,dtype=float)
    n = len(w)
    records=[]  
    if daily: daynorm=86400.0
    else: daynorm = 1.0
    for file in phafiles:
       sp = fits.open(file[0])
       result =  get_continuum_values(
          (sp[2].header['tstart']-tstart)/daynorm,
          sp[2].data['lambda'],
	  sp[2].data['flux'],
	  sp[2].data['quality'],
	  cont_regions=regions,
	  qlimit=qlimit)
       sp.close()	  
       records.append(result) 	  
    t=[]
    for rec in records:
        t.append(rec[0])
    allbands=[]
    k=1
    for r in regions:
        band=[]
        for rec in records:
	    band.append(rec[k])
        allbands.append(band)
        k+=1		
    if full:
       return w,t, allbands, records
    
def plot_spectrum(ax,phafile,errbars=False, errhaze=False, 
        hazecolor='grey', hazealpha=0.2, flag='all'):
    f = fits.open(phafile)
    q = f[2].data['quality'] 
    r = quality_flags_to_ranges(q)
    r = r[flag] 
    label = f[1].header['date-obs']
    w = f[2].data['lambda']
    flx = f[2].data['flux']
    err = f[2].data['fluxerr']
    if not errbars:
        for rr in r:
            ax.plot(w[rr[0]:rr[1]],flx[rr[0]:rr[1]],label=label)
	    if errhaze:
	        ax.fill_between(w[rr[0]:rr[1]],
	        flx[rr[0]:rr[1]]-err[rr[0]:rr[1]], 
	        flx[rr[0]:rr[1]]+err[rr[0]:rr[1]],
	        color=hazecolor,alpha=hazealpha)
    else:
        ax.errorbar( w[rr[0]:rr[1]],flx[rr[0]:rr[1]],
	   yerr=err[rr[0]:rr[1]],label=label)   

def quality_flags_to_ranges(quality):
       """given wavelength and quality flag arrays, reduce
       the quality to ranges of a certain quality (except 
       for "good" = 0.)
       
       parameters
       ----------
       wave : array 
         x-axis values
       quality : array 
         quality flags matching x-axis points
	 
       returns
       -------
       quality_ranges : dict
          a dictionary of ranges for each flag except 'good'  
       
       """
       from uvotgetspec import quality_flags
       flagdefs = quality_flags()
       flags=flagdefs.keys()
       quality_ranges = {}
       val = []
       for fla in flags:
           if fla == 'good': break
           fval = flagdefs[fla]
	   q = quality >= fval  # limit the search
	   indx = np.where(q)[0] # indx of wave where flag 
	   fval2 = fval*2
	   loc = quality[q]/fval2*fval2 != quality[q]
	   v = indx[loc] # indices of the points with this flag
	   if len(v) > 1:  # require at least 2
 	       vrange = []
	       v1 = v[0]
	       vlast = v1
	       for v2 in v[1:]: 
	           if v2-vlast > 1:  # require neighboring pixels
	               # 
		       vrange.append([v1,vlast])
		       val.append([v1,vlast])
		       v1 = v2
		       vlast = v2
	           else:
	               vlast=v2
	       if vlast > v1: vrange.append([v1,vlast])	 # last range       
	       quality_ranges.update({fla:vrange}) 
       quality_ranges.update({"all":val}) 	           
       return quality_ranges

def complement_of_ranges(ranges,rangestart=0,rangeend=None):
    """given a list of exclusion ranges, compute the complement"""
    print "needs to be completed"


class gaussian():
   """define a gaussian function 
   """
   def __init__(self,param=None):
       self.parameters = {"model":"gauss",
            "parinfo":[
	        {"limited":[0,0],"limits":[0,0],"value":0.0,"parname":'amp'},
	        {"limited":[0,0],"limits":[0,0],"value":0.0,"parname":'pos'}, 
        	{"limited":[0,0],"limits":[0,0],"value":0.0,"parname":'sig'},]
            }

   def parameters(self):
       return self.parameters
	    
   def value(self, x):
       par = self.parameters["parinfo"]
       for p in par:
           if (p["parname"]=='amp'):  
	       amp = p["value"]
       for p in par:
           if (p["parname"]=='pos'):  
	       pos  = p["value"]
       for p in par:
           if (p["parname"]=='sig'):  
	       sig = p["value"]
       return amp * np.exp( - ((x-pos)/sig)**2 )   

   def update(self,amp=None,pos=None,sig=None):
       par = self.parameters["parinfo"]  # this is a list
       for p in par:
           if (p["parname"]=='amp') & (amp != None):  
	       p["value"]=amp
       for p in par:
           if (p["parname"]=='pos') & (pos != None):  
	       p["value"]=pos
       for p in par:
           if (p["parname"]=='sig') & (sig != None):  
	       p["value"]=sig
       self.parameters["parinfo"]=par

class poly_background():
    import numpy as np
    def __init__(self,coef=[0]):
        self.poly_coef = coef
    
    def value(self,x):
        return np.polyval(self.poly_coef,x)
	
    def update(self,coef):
        self.poly_coef=coef
			

# following routines are under development - no testing was yet done    
class fit_spectrum():
    """
    Fit the spectrum in a way similar to Xspec. 
    
    Parameters
    ==========
    ax : matplotlib.axes.AxesSubplot instance
    spectrum : spectrum 
       spectrum [currently second extension of spectrum file uvot grism]
    fitparameters :  
       for each model to fit, a list of fit parameters    
         
       
    Notes
    =====
    The method of fitting is similar to that used in XSpec because
    non-linear fitting is better done while freezing some parameters
    at a time, depending on need, and iterating. This is superior to 
    automated fitting methods. 
    
    Initial set up:
       Though the plan is to introduce models to fit to at some point, 
       the initial fitting will be for a continuum with gaussians. 
    
    """
    import sys
    import numpy as np
    from mpfit import mpfit

    def __init__(self, ax, spectrum, fit_parameters=[], ):
        self.spectrum = spectrum
        self.fit_parameters = fit_parameters
	self.ax = ax
	self.funargs = []
	self.models = []  # active models
	self.valid_model_names=['gaussian','poly_background']  # implemented models

    
    def add_model(self,modelname):
        """
	Add a model to the fit_parameters
	
	Notes
	=====
	open a new item and add the parameters and argument names of the 
	specific model
        
	Build the parinfo list, e.g., we need a list with for each parameter: 
        parinfo = [
	{'limited': [1,0], 'limits': [0.1,0.0],     'value':    bg, 'parname': 'bg0'},
	{'limited': [0,0], 'limits': [0.0,0.0],     'value':   0.0, 'parname': 'bg1' },
	{'limited': [1,0], 'limits': [0.0,0.0],     'value':  amp1, 'parname': 'amp1'},
	{'limited': [1,1], 'limits': [pos1a,pos1b], 'value':  pos1, 'parname': 'pos1'},
	...
	]  
        where 
	limited indicated if the limits are applied (1=True,0=False)
	limits gives the parameter range
	value gives the starting(or fixed) value
	parname gives the parameter name
	if the value needs to be fixed, the limits must be set to a very small range 
	  for running mpfit (1e-5 * value?) 
	
        
	"""
	# verify valid model name ?
	#  ...
	if not (modelname in self.valid_model_names):
	   print "illegal model name"
	   print "valid names are :",self.valid_model_names
	   return
        self.models.append(modelname)
	fitparms, funargs = eval("self.model_"+modelname+"(0,init_parameters=True)")
        self.fit_parameters.append( fitparms )
        self.funargs.append( funargs )

    
    def show_parameters(self,):
        """
	print out the parameters and values
	"""
	#use write( sys.stdout, ) ??? 
	print "comp  model       param   value       lower          upper \n"
	for k in range(len(self.models)):
	    modelno = k
	    name = self.fit_parameters[k]["model"]
	    params = self.fit_parameters[k]["parinfo"]
	    for p in params:
	        print "%2i %16s %8s %12.5e %12.5e %12.5e"%(k,name,p["parname"][:-2],
		      p["value"],p["limits"][0],p["limits"][1])
	
    
    def update_parameters(self,component=None,name=None,value=None,lower=None,upper=None):
        """ 
	Change the limits, start value, etc. for a model parameter
	"""
	fitparms = self.fit_parameters
	print " not yet implemented -- need a widget "    
	if component == None:   # do all components
	   for k in range(len(self.models)): 
	       self.update_parameters(k,name=name,value=value,lower=lower,upper=upper) 
	else:
	   if component in range(len(self.models)): 
	       parms = fitparms[component]
	       parnames = []
	       for z in parms['parinfo']:
	          parnames.append(z['parname'][:-2])
	   else:
	       print "illegal component number"           
	   if name == None:  # do all variables in parms
	       for m in range(len(parms['parinfo'])):
	           if value != None: parms['parinfo'][m]['value'] = value
		   if lower != None: 
		       parms['parinfo'][m]['limits'][0] = lower
		       parms['parinfo'][m]['limited'][0] = 1
		   if upper != None: 
		       parms['parinfo'][m]['limits'][1] = upper
		       parms['parinfo'][m]['limited'][1] = 1
           else:
	       if name in parnames:
	           for i in range(len(parnames)):
	               if name == parnames[i]: 
		           m=i
			   continue
	           if value != None: parms['parinfo'][m]['value'] = value
		   if lower != None: 
		       parms['parinfo'][m]['limits'][0] = lower
		       parms['parinfo'][m]['limited'][0] = 1
		   if upper != None: 
		       parms['parinfo'][m]['limits'][1] = upper
		       parms['parinfo'][m]['limited'][1] = 1
	       else:
	           print "illegal parameter name, valid names are :",parnames	    		       

	
    def model_poly_background(self,p, fjac=None, x=None, y=None, 
            err=None,init_parameters=False):
        # up to sixth order polynomial 
        if init_parameters:
	    component_number=len(self.models)
	    cn = "%02s"%(component_number)
            return {"model":"poly_background", "parinfo":[
	      {"limited":[1,1],"limits":[1,1],"value":1,  "parname":'order'+cn}, # fixed! limits=value
	      {"limited":[0,0],"limits":[0,0],"value":0.0,"parname":'coef0'+cn},
	      {"limited":[0,0],"limits":[0,0],"value":0.0,"parname":'coef1'+cn},
	      {"limited":[0,0],"limits":[0,0],"value":0.0,"parname":'coef2'+cn},
	      {"limited":[0,0],"limits":[0,0],"value":0.0,"parname":'coef3'+cn},
	      {"limited":[0,0],"limits":[0,0],"value":0.0,"parname":'coef4'+cn},
	      {"limited":[0,0],"limits":[0,0],"value":0.0,"parname":'coef5'+cn},
	      {"limited":[0,0],"limits":[0,0],"value":0.0,"parname":'coef6'+cn},
	      ]},['x','y','err']
	order = p[0]
	if type(order) != int:
	   print "problem model_poly_background order not an int:", order      
	model = np.polyval(p[1:order+2],x)      
        status = 0
        return [status, (y-model)/err]
    
    
    def model_gaussian(self,p, fjac=None, x=None, y=None, err=None,init_parameters=False):    
        if init_parameters:
	    component_number=len(self.models)
	    cn = "%02s"%(component_number)
	    return {"model":"gauss","parinfo":[
	    {"limited":[0,0],"limits":[0,0],"value":1.0,"parname":'amp'+cn},
	    {"limited":[0,0],"limits":[0,0],"value":9.0,"parname":'pos'+cn},
	    {"limited":[0,0],"limits":[0,0],"value":0.5,"parname":'sig'+cn},
	    ]}, ['x', 'y', 'err']
	amp1,pos1,sig1 = p          
        model = amp1 * np.exp( - ((x-pos1)/sig1)**2 ) 
        status = 0
        return [status, (y-model)/err]

    
    def fit(self,):
        """ 
	call the fitter
	
	"""
	x = self.spectrum.wavelength
	y = self.spectrum.flux
	err = self.spectrum.fluxerr
	# build initial parameter value list 
        p0 = []
	pinfo = []
        for x in self.fit_parameters:
	   par = x["parinfo"]
	   for pv in par:
               p0.append(pv['value'])
	       pinfo.append(pv)
	   
        # build argument list
        fkw={'x':x,'y':y,'err':err}
	   
	# call L-M fitter   
        Z = mpfit.mpfit(fit_function,p0,functkw=fkw,parinfo=pinfo,quiet=True)
        if (Z.status <= 0): print 'fit_spectrum.mpfit error message = ', Z.errmsg
	
	# update the fit (parinfo...?), return results
	self.result = Z  
        return Z	    

    
    def fit_function(self, p, fjac=None, x=None, y=None, err=None):
        """
	Define the 1D spectrum function to fit
	 
	parameters
	----------
	x : array 
	   the wavelengths
	y : array
	   the estimated flux   
	err : array
	   the error 
	   
	p : list 
	   a list of parameters to fit        
	"""
        
	F = 0.
	i = 0
	nmod = len(self.models)
        for k in range(nmod):	    
	    npar = len(self.fitparms[k]["parinfo"])
	    arg = "("
	    p0 = []
	    for w in range(npar):
	        p[0].append( p[i]) 
		i += 1
            fjac_mod = fjac[i:i+npar,i:i+npar]	
	    # fjac is a dummy parameter in gauss and poly_background 
	    # any other use needs testing that the correct slice has 
	    # been made ... 	
            arg += ", fjac=fjac_mod, x=x, y=y, err=err)"		
	    F = F + eval("model_"+self.models[k]+arg)	    
        status = 0
	if err != None:
           return [status, (y-F)/err]
	else:
	   return [status, (y-F)]   
    
    
def dofit2gpbg(x,f,err,bg,amp1,pos1,sig1,amp2,pos2,sig2,
    amp2lim=None,fixsig=False,
    fixsiglim=0.2, fixpos=False,chatter=0):
   '''
   Fit two gaussians plus a linear varying background to f(x)
   
   Parameters
   ==========
   x : array
   f : array
   err : ?
   bg : ?
   gaussian_parameters : array
      for each gaussian the array provides the following parameters
      - amplitude f(xo)
      - position xo in x
      - sigma (width of the gaussian)  fit=A.exp( - ((x-xo)/sig)**2 )
      - lower limit on amplitude Amin or None
      - upper limit on amplitude Amax ro None
      - sig_lo lower limit on sigma
      - sig_hi upper limit on sigma  
      - fixamp boolean (True or False) for fixed amplitude
      - fixsig boolean (True or False) for fixed sigma
   
   '''
   import numpy as np
   from mpfit import mpfit
   
   gp = np.array(gaussian_parameters,)
   
   n_gaussians = len(gaussian_parameters)
   
   if np.isfinite(bg):
      bg0 = bg
   else: bg0 = 0.0   
   bg1 = 0.0 
   if np.isfinite(sig1):
      sig1 = np.abs(sig1)
   else: sig1 = 3.1   
   if np.isfinite(sig2):
      sig2 = np.abs(sig2)
   else: sig2 = 4.2        

   p0 = (bg0,bg1,amp1,pos1,sig1,amp2,pos2,sig2)
   
   # define the variables for the function 'myfunct'
   fa = {'x':x,'y':f,'err':err}

   if fixpos:
     pos1a = pos1-0.05
     pos1b = pos1+0.05
     pos2a = pos2-0.05
     pos2b = pos2+0.05
   else:  
   # adjust the limits to not cross half the predicted distance of orders
     pos1a = pos1-sig1
     pos1b = pos1+sig1
     pos2a = pos2-sig1
     pos2b = pos2+sig1
     # case :  pos1 < pos2 
     if (pos1 < pos2):
        pos1b = pos2a = 0.5*(pos1+pos2)
     else:  
        pos1a = pos2b = 0.5*(pos1+pos2)

   if fixsig:
      sig1_lo = sig1-fixsiglim
      sig1_hi = sig1+fixsiglim
      sig2_lo = sig2-fixsiglim
      sig2_hi = sig2+fixsiglim
   else:   
   # make sure lower limit sigma is OK 
      sig1_lo = max([sig1-10.,35.])
      sig2_lo = max([sig2-10.,35.])
      sig1_hi = min([sig1+10.,5.])
      sig2_hi = min([sig2+10.,5.])
     
   if amp2lim != None:
      amp2min, amp2max = amp2lim
      parinfo = [{  \
   'limited': [1,0],   'limits' : [np.min([0.0,bg0]),0.0],'value':    bg,   'parname': 'bg0'    },{  \
   'limited': [0,0],   'limits' : [0.0,0.0],           'value'  :   0.0,   'parname': 'bg1'    },{  \
   'limited': [1,0],   'limits' : [0.0,0.0],           'value'  :  amp1,   'parname': 'amp1'   },{  \
   'limited': [1,1],   'limits' : [pos1a,pos1b],       'value'  :  pos1,   'parname': 'pos1'   },{  \
   'limited': [1,1],   'limits' : [sig1_lo,sig1_hi],   'value'  :  sig1,   'parname': 'sig1'   },{  \
   'limited': [1,1],   'limits' : [amp2min,amp2max],   'value'  :  amp2,   'parname': 'amp2'   },{  \
   'limited': [1,1],   'limits' : [pos2a,pos2b],       'value'  :  pos2,   'parname': 'pos2'   },{  \
   'limited': [1,1],   'limits' : [sig2_lo,sig2_hi],   'value'  :  sig2,   'parname': 'sig2'   }]  
      
   else:  
      parinfo = [{  \
   'limited': [1,0],   'limits' : [np.min([0.0,bg0]),0.0],'value':    bg,   'parname': 'bg0'    },{  \
   'limited': [0,0],   'limits' : [0.0,0.0],           'value'  :   0.0,   'parname': 'bg1'    },{  \
   'limited': [1,0],   'limits' : [0.0,0.0],           'value'  :  amp1,   'parname': 'amp1'   },{  \
   'limited': [1,1],   'limits' : [pos1a,pos1b],       'value'  :  pos1,   'parname': 'pos1'   },{  \
   'limited': [1,1],   'limits' : [sig1_lo,sig1_hi],   'value'  :  sig1,   'parname': 'sig1'   },{  \
   'limited': [1,0],   'limits' : [0.0,0.0],           'value'  :  amp2,   'parname': 'amp2'   },{  \
   'limited': [1,1],   'limits' : [pos2a,pos2b],       'value'  :  pos2,   'parname': 'pos2'   },{  \
   'limited': [1,1],   'limits' : [sig2_lo,sig2_hi],   'value'  :  sig2,   'parname': 'sig2'   }]  

   if chatter > 4: 
      print "parinfo has been set to: " 
      for par in parinfo: print par

   Z = mpfit(_fit2,p0,functkw=fa,parinfo=parinfo,quiet=True)
   
   if (Z.status <= 0): 
      print 'uvotgetspec.runfit2.mpfit error message = ', Z.errmsg
      print "parinfo has been set to: " 
      for par in parinfo: print par
   elif (chatter > 3):   
      print "\nparameters and errors : "
      for i in range(8): print "%10.3e +/- %10.3e\n"%(Z.params[i],Z.perror[i])
   
   return Z	
       
       
def _fit2(p, fjac=None, x=None, y=None, err=None):
   import numpy as np

   (bg0,bg1,amp1,pos1,sig1,amp2,pos2,sig2) = p 
	     
   model = bg0 + bg1*x + \
           amp1 * np.exp( - ((x-pos1)/sig1)**2 ) + \
           amp2 * np.exp( - ((x-pos2)/sig2)**2 ) 
	    
   status = 0
   return [status, (y-model)/err]
    
def dofit2poly(x,f,err,coef1=[0,1],coef2=[0,1],
    chatter=0):
   '''
   Fit the sum of two first order polynomials f(x)
   
   Parameters
   ==========
   x : array
   f : array
   err : ?
   coef? : list
      coefficients of polynomial
   
   PROBLEM: gives some numbers that don't make sense...
   '''
   import numpy as np
   from mpfit import mpfit
      
   p0 = (coef1[1],coef1[0],coef2[1],coef2[0])
   
   # define the variables for the function 'myfunct'
   fa = {'x':x,'y':f,'err':err}
     
   parinfo = [{  
   'limited': [0,0],   'limits' : [0.0,0.0],'value':  coef1[1],   'parname': 'coef11'  },{  
   'limited': [0,0],   'limits' : [0.0,0.0],'value':  coef1[0],   'parname': 'coef10'  },{  
   'limited': [0,0],   'limits' : [0.0,0.0],'value':  coef2[1],   'parname': 'coef21'  },{  
   'limited': [0,0],   'limits' : [0.0,0.0],'value':  coef2[0],   'parname': 'coef20'  },]  
      
   if chatter > 4: 
      print "parinfo has been set to: " 
      for par in parinfo: print par

   Z = mpfit(_fit3,p0,functkw=fa,parinfo=parinfo,quiet=True)
   
   if (Z.status <= 0): 
      print 'uvotspec.dofit2poly.mpfit error message = ', Z.errmsg
      print "parinfo has been set to: " 
      for par in parinfo: print par
   elif (chatter > 3):   
      print "\nparameters and errors : "
      for i in range(4): print "%10.3e +/- %10.3e\n"%(Z.params[i],Z.perror[i])
   
   return Z	
              
def _fit3(p, fjac=None, x=None, y=None, err=None):
   import numpy as np
   (coef11,coef10,coef21,coef20) = p 	     
   model = np.polyval([coef10,coef11],x)+np.polyval([coef20,coef21],x)
   status = 0
   return [status, (y-model)/err]
    
