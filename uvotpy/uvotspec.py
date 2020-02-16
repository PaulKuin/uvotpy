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
# Developed by N.P.M. Kuin (MSSL/UCL) 
# uvotpy 
# (c) 2009-2014, see Licence  

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from builtins import input
from builtins import range
from builtins import object
import sys
import numpy as np
import matplotlib.pyplot as plt
from stsci.convolve import boxcar
from astropy.io import fits
from matplotlib.lines import Line2D

'''
   Stuff to work with UVOT spectra:
      - adjust the wavelengths 
      - flag bad quality data
      - plot spectrum with IDs of spectral lines 
      - sum spectra
      
   Goal (needs lots of further work):   
      - fitting of a spectrum consisting of additive and multiplative 
        elements, e.g, a background, gaussian lines, ISM extinction,.. 
      - derive physical properties from fitted spectral elements, 
        e.g., NH, line fluxes, abundances, ionisation state
      - given N dimensional parameter solution space of a model, 
        find the best fit solution       
'''

__version__ = '20190304-1.0.0'

v = sys.version
if v[0] == '2': 
    from __builtin__ import raw_input as input

# spectroscopic summary data
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
{'name':'H-7' ,'wavevac':3836.485,  'label':r'H7'},
#{'name':'H-8' ,'wavevac':3798.987,  'label':r'H8'},
#{'name':'H-9' ,'wavevac':3771.704,  'label':r'H9'},
{'name':'H-limit'  ,'transition':'2s-40','wavevac':3656,      'label':r'Ba-limit'},
{'name':'Pa-alpha' ,'transition':'3-4'  ,'wavevac':18756.096, 'label':r'Pa$\alpha$'},
{'name':'Pa-beta'  ,'transition':'3-5'  ,'wavevac':12821.576, 'label':r'Pa$\beta$'},
{'name':'Pa-gamma' ,'transition':'3-6'  ,'wavevac':10941.082, 'label':r'Pa$\gamma$'},
{'name':'Pa-delta' ,'transition':'3-7'  ,'wavevac':10052.123, 'label':r'Pa$\delta$'},
{'name':'Pa-5'     ,'transition':'3-8'  ,'wavevac':9548.587,  'label':r'Pa5'},
{'name':'Pa-limit' ,'transition':'3s-40','wavevac':8252.2,    'label':r'Pa-limit'},
     ],
'HeI':[
# singlets
{'transition':'1s2p 1Po - 1s3s 1S','wavevac':7283.4 ,'label':u'HeIs'},
{'transition':'1s2p 1Po-1s3d 1D ','wavevac':6680.0 ,'label':u'HeIs'},
{'transition':'1s2s 1S -1s3p 1Po','wavevac':5017.08 ,'label':u'HeIs'},
{'transition':'1s2p 1Po -1s4d 1D','wavevac':4923.3 ,'label':u'HeIs'},
{'transition':'1s2s 1S -1s4p 1Po','wavevac':3965.85 ,'label':u'HeIs'},
{'transition':'1s2p 1Po -1s4s 1S','wavevac':5049.1 ,'label':u'HeIs'},
# triplets
{'transition':'1s2p 3Po-1s2s 3S ','wavevac':10830.3  ,'label':u'HeI'},
{'transition':'1s2p 3Po-1s3s 3S ','wavevac':7067.14   ,'label':u'HeI'},
{'transition':'1s2p 3Po-1s4s 3S ','wavevac':4714.46   ,'label':u'HeI'},
{'transition':'1s2p 3Po-1s5d 3D ','wavevac':4027.32   ,'label':u'HeI'},
{'transition':'1s2p 3Po-1s3d 3D ','wavevac':5877.249  ,'label':u'HeI'},
{'transition':'1s2s 3S -1s3p 3Po','wavevac':3889.75   ,'label':u'HeI'},
{'transition':'1s2s 3S -1s4p 3Po','wavevac':3188.667  ,'label':u'HeI'},
{'transition':'2p2  3P -2p3d 3Do','wavevac':3014.59   ,'label':u'HeI'},
{'transition':'1s2s 3S -1s5p 3Po','wavevac':2945.967  ,'label':u'HeI'},
{'transition':'1s2s 3Po-1s4d 3D ','wavevac':4472.735  ,'label':u'HeI'},
#{'wavevac':2829.9,'label':u'HeI'},
#{'wavevac':2819.0,'label':u'HeI'},
#{'wavevac':2764.6,'label':u'HeI'},
#{'wavevac':2723.99,'label':u'HeI'},
#{'wavevac':2696.9,'label':u'HeI'},
#{'wavevac':2677.9,'label':u'HeI'},
{'wavevac':2578.4,'label':u'HeI'},
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
'FeIIuv3':[
{'transition':'2 - 2','wavevac':2381., 'label':u''},
{'transition':'1 - 2','wavevac':2365.,'label':u''},
{'transition':'1 - 1','wavevac':2344.,'label':u'FeII - uv3'},
    ],
'FeIIuv2':[
{'transition':'2 - 2','wavevac':2396., 'label':u''},
{'transition':'2 - 1','wavevac':2374.,'label':u'FeII - uv2'},
{'transition':'1 - 1','wavevac':2382.,'label':u''},
    ],
'FeIIuv1':[
{'transition':'2 - 3','wavevac':2632., 'label':u''},
{'transition':'2 - 2','wavevac':2612., 'label':u''},
{'transition':'2 - 1','wavevac':2586.,'label':u'FeII - uv1'},
{'transition':'1 - 2','wavevac':2626.,'label':u'`'},
{'transition':'1 - 1','wavevac':2600.,'label':u''},
    ],
'nova':[ # add also H, HeI, HeII 
# 
{'transition':'','wavevac':1750 , 'label':u'N III]'},
{'transition':"",'wavevac':1810 , 'label':'Si II'},
{'transition':'','wavevac':1862.3  , 'label':u'Al III]'},
{'transition':'','wavevac':1908.7, 'label':u'C III]'},
{'transition':'','wavevac':2143  , 'label':u'N II]'},
#{'transition':'','wavevac':2151.0, 'label':u'N IV]'},
{'transition':'','wavevac':2297  , 'label':u'C III'},
{'transition':'','wavevac':2325.4, 'label':u'C II'},
{'transition':'','wavevac':2335.4, 'label':u'Si II'},
#{'transition':'','wavevac':2326.1, 'label':u'C II'},
{'transition':'','wavevac':2332.1, 'label':u'[O III]'},
{'transition':'','wavevac':2471.0, 'label':u'O II]'},
#{'transition':'5D-3D','wavevac':2473, 'label':u'Ni IV]'},
#{'transition':'5D-3D','wavevac':2522.5, 'label':u'Ni IV]'},
{'transition':'','wavevac':2796.4, 'label':u'Mg II'},
{'transition':'','wavevac':2803.5, 'label':u'Mg II'},
{'transition':'','wavevac':2937.4, 'label':u'Mg II*'},
{'transition':'','wavevac':3130.0, 'label':u'O III*'},
{'transition':'','wavevac':3345.8, 'label':u'[Ne V]'},
{'transition':'','wavevac':3425.9, 'label':u'[Ne V]'},
{'transition':'','wavevac':3727  , 'label':u'[O III]'},
{'transition':'','wavevac':3869, 'label':u'[Ne III]'},
{'transition':'','wavevac':3968, 'label':u'[Ne III]'},
{'transition':'','wavevac':4363  , 'label':u'[O III]'},
{'transition':'','wavevac':4636  , 'label':u'N III*'},
{'transition':'','wavevac':4643  , 'label':u'N III*'},
{'transition':'','wavevac':4648.7, 'label':u'C III*'},
{'transition':'','wavevac':4651.2, 'label':u'O III*'},
{'transition':'','wavevac':4959  , 'label':u'[O III]'},
{'transition':'','wavevac':5007  , 'label':u'[O III]'},
{'transition':'','wavevac':5755  , 'label':u'[N II]'},
#{'transition':'','wavevac':.0, 'label':u''}
#{'transition':'','wavevac':.0, 'label':u''}
  ],
}
  
############################ 

spdata2 = {

'V339_Del':[ # add also H, HeI, HeII 
# 
{'transition':'2-4'  ,'wavevac':4862.69,   'label':r'H$\beta$'},
{'transition':'2-5'  ,'wavevac':4341.69,   'label':r'H$\gamma$'},
{'transition':'2-6'  ,'wavevac':4102.899,  'label':r'H$\delta$'},
{'transition':'2-7'  ,'wavevac':3971.202,  'label':r'H$\epsilon$'},
#{'transition':'4 - 7','wavevac':5411.5, 'label':u'HeII'},
{'transition':'3 - 4','wavevac':4687.1, 'label':u'He II'},
{'transition':'3 - 5','wavevac':3203.95,'label':u'He II'},
{'transition':'3 - 6','wavevac':2734.13,'label':u'He II+O II*'},
# unclutter {'transition':'3 - 6','wavevac':2734.13,'label':u'HeII'},
{'transition':'2s2.2p2(3P)4s-2s2.2p2(3P)3p','wavevac':2747.4, 'label':u'OII*'},
{'transition':'3 - 7','wavevac':2511.2, 'label':u'He II'},
{'transition':'3 - 8','wavevac':2385.4, 'label':u'He II'},
{'transition':'','wavevac':1750  , 'label':u'N III]'},
{'transition':'','wavevac':1908.7, 'label':u'C III]'},
#{'transition':'','wavevac':1987.7, 'label':u'S IX]*'},
# declutter {'transition':'','wavevac':2143  , 'label':u'N II]'},
{'transition':'','wavevac':2147  , 'label':u'N II]+IV]'},
# declutter {'transition':'','wavevac':2151.0, 'label':u'N IV]'},
#{'transition':'','wavevac':2321.66, 'label':u'O III]'},
{'transition':'','wavevac':2325.6, 'label':u'C II+[O II]'},
{'transition':'','wavevac':2332.1, 'label':''},
# unclutter {'transition':'','wavevac':2325.6, 'label':u'C II'},
# unclutter {'transition':'','wavevac':2332.1, 'label':u'[O III]'},
#{'transition':'5D-3D','wavevac':2437.2, 'label':u'Ni V]'},
{'transition':'','wavevac':2471.0, 'label':u'O II]'},
#{'transition':'','wavevac':2522.5, 'label':u'Ni V]'},
#{'transition':'','wavevac':2784, 'label':u'Mg V]'},
{'transition':'','wavevac':2800, 'label':u'Mg II'},
{'transition':'','wavevac':2839, 'label':  u'N III] 1750(2)'},
{'transition':'','wavevac':2844.9, 'label':u'              C III]'},
{'transition':'','wavevac':2937.4, 'label':u'Mg II*'},
#{'transition':'','wavevac':2949, 'label':u'Mg V]'},
# unsure of this one {'transition':'','wavevac':2990, 'label':u'Ni VII]'},
{'transition':u'2s2 2p(2P°)3p 3S 1-2s2 2p(2P°)3d 3P° 2','wavevac':3133.77, 'label':u'O III*'},
{'transition':'','wavevac':3287.5, 'label':u'C III 1909(2)'},
#{'transition':'','wavevac':.0, 'label':u''}
#{'transition':'','wavevac':3132, 'label':u'?Be II ?Fe II '},
#{'transition':'2s2 2p2 3P 0-2s2 2p2 1D 2','wavevac':3301.4, 'label':u'[NeV]'},
{'transition':'2s2 2p2 3P 1-2s2 2p2 1D 2','wavevac':3346.8, 'label':u'[NeV]'},
{'transition':'2s2 2p2 3P 1-2s2 2p2 1D 2','wavevac':3426.9, 'label':u'[NeV]'},
#{'transition':'3d4 5D-3d4 3G','wavevac':3446.61, 'label':u'[FeV]'},
{'transition':'','wavevac':3448, 'label':u'N IV*'},
#declutter {'transition':'','wavevac':3444.6, 'label':u'N IV*'},
#declutter {'transition':'','wavevac':3461.4, 'label':u'N IV*'}, # opt. thick, especially at later times 
{'transition':'2s2 2p4 3P 2-2s2 2p4 3P 0','wavevac':3461.7, 'label':u'[Ca XIII]'},
#{'transition':'3d4 5D-3d4 3G','wavevac':3464.5, 'label':u'[FeV]'},
{'transition':'','wavevac':3727  , 'label':u'[O III]'},
{'transition':'','wavevac':4363  , 'label':u'[O III]'},
# declutter {'transition':'','wavevac':4640  , 'label':u'N III*'},
{'transition':'','wavevac':4645, 'label':u'C III*+N III*'},
# declutter {'transition':'','wavevac':4649, 'label':u'C III*'},
{'transition':'','wavevac':4959  , 'label':u'[O III]'},
{'transition':'','wavevac':5007  , 'label':u'[O III]'},
{'transition':'','wavevac':5755  , 'label':u'[N II]'},
#{'transition':'','wavevac':.0, 'label':u''}
#{'transition':'','wavevac':.0, 'label':u''}
  ],

'V5668Sgr':[ # add also H, HeI, HeII 
# 
{'transition':'2-4'  ,'wavevac':4862.69,   'label':r'H$\beta$'},
{'transition':'2-5'  ,'wavevac':4341.69,   'label':r'H$\gamma$'},
{'transition':'2-6'  ,'wavevac':4102.892,  'label':r'H$\delta$'},
{'transition':'2-7'  ,'wavevac':3971.2,  'label':r'H$\epsilon$'},
{'name':'H-6'      ,'transition':'2-8'  ,'wavevac':3890.16,   'label':r'H6'},
{'name':'H-7' ,'wavevac':3836.485,  'label':r'H7'},
{'transition':'','wavevac':1750  , 'label':u'N III]'},
{'transition':'','wavevac':1908.7, 'label':u'C III]'},
{'transition':'','wavevac':2143  , 'label':u'N II]'},
{'transition':'','wavevac':2325.6, 'label':u'C II'},
{'transition':'','wavevac':2471.0, 'label':u'O II]'},
{'transition':'','wavevac':2800, 'label':u'Mg II'},
{'transition':'','wavevac':4645, 'label':u'C III+N III'},
#{'transition':'','wavevac':4649, 'label':u'C III*'},
{'transition':'','wavevac':4959  , 'label':u'[O III]'},
{'transition':'','wavevac':5007  , 'label':u'[O III]'},
{'transition':'','wavevac':5755  , 'label':u'[N II]'},
#{'transition':'','wavevac':.0, 'label':u''}
  ],

'V1369_Cen':  [
{'transition':'2-4'  ,'wavevac':4862.69,   'label':r'H$\beta$'},
{'transition':'2-5'  ,'wavevac':4341.69,   'label':r'H$\gamma$'},
{'transition':'2-6'  ,'wavevac':4102.899,  'label':r'H$\delta$'},
{'transition':'2-7'  ,'wavevac':3971.202,  'label':r'H$\epsilon$'},
{'transition':'4 - 7','wavevac':5411.5, 'label':u'HeII'},
{'transition':'3 - 4','wavevac':4687.1, 'label':u'He II'},
{'transition':'3 - 5','wavevac':3203.95,'label':u'He II'},
{'transition':'3 - 6','wavevac':2734.13,'label':u'He II+O II*'},
# unclutter {'transition':'2s2.2p2(3P)4s-2s2.2p2(3P)3p','wavevac':2747.4, 'label':u'OII*'},
#{'transition':'3 - 7','wavevac':2511.2, 'label':u'He II'},
#{'transition':'3 - 8','wavevac':2385.4, 'label':u'He II'},
{'transition':'','wavevac':1750  , 'label':u'N III]'},
{'transition':'','wavevac':1814  , 'label':u'Si II'},
{'transition':'','wavevac':1860  , 'label':u'Al III'},
{'transition':'','wavevac':1908.7, 'label':u'C III]'},
#{'transition':'','wavevac':1987.7, 'label':u'S IX]*'},
# declutter {'transition':'','wavevac':2143  , 'label':u'N II]'},
{'transition':'','wavevac':2147  , 'label':u'N II]+IV]'},
# declutter {'transition':'','wavevac':2151.0, 'label':u'N IV]'},
#{'transition':'','wavevac':2321.66, 'label':u'O III]'},
{'transition':'','wavevac':2325.6, 'label':u'C II+[O II]'},
{'transition':'','wavevac':2332.1, 'label':''},
# unclutter {'transition':'','wavevac':2325.6, 'label':u'C II'},
# unclutter {'transition':'','wavevac':2332.1, 'label':u'[O III]'},
#{'transition':'5D-3D','wavevac':2437.2, 'label':u'Ni V]'},
{'transition':'','wavevac':2471.0, 'label':u'O II]'},
#{'transition':'','wavevac':2522.5, 'label':u'Ni V]'},
#{'transition':'','wavevac':2784, 'label':u'Mg V]'},
{'transition':'','wavevac':2800, 'label':u'Mg II'},
{'transition':'','wavevac':2844.9, 'label':u'2nd order line'},
#{'transition':'','wavevac':2937.4, 'label':u'Mg II*'},
#{'transition':'','wavevac':2949, 'label':u'Mg V]'},
# unsure of this one {'transition':'','wavevac':2990, 'label':u'Ni VII]'},
{'transition':'2s2.2p2(3P)4s-2s2.2p2(3P)3p','wavevac':3134.2, 'label':u'O II*'},
# unclutter {'transition':'2s2.2p2(3P)4s-2s2.2p2(3P)3p','wavevac':3287.5, 'label':u'OII*'},
{'transition':'','wavevac':3291, 'label':u'2nd order line'},
#{'transition':'','wavevac':.0, 'label':u''}
#{'transition':'','wavevac':3132, 'label':u'?Be II ?Fe II '},
#{'transition':'2s2 2p2 3P 0-2s2 2p2 1D 2','wavevac':3301.4, 'label':u'[NeV]'},
{'transition':'2s2 2p2 3P 1-2s2 2p2 1D 2','wavevac':3346.8, 'label':u'[NeV]'},
{'transition':'2s2 2p2 3P 1-2s2 2p2 1D 2','wavevac':3426.9, 'label':u'[NeV]'},
#{'transition':'3d4 5D-3d4 3G','wavevac':3446.61, 'label':u'[FeV]'},
#{'transition':'','wavevac':3448, 'label':u'N IV*'},
#declutter {'transition':'','wavevac':3444.6, 'label':u'N IV*'},
#declutter {'transition':'','wavevac':3461.4, 'label':u'N IV*'}, # opt. thick, especially at later times 
#{'transition':'2s2 2p4 3P 2-2s2 2p4 3P 0','wavevac':3461.7, 'label':u'[Ca XIII]'},
#{'transition':'3d4 5D-3d4 3G','wavevac':3464.5, 'label':u'[FeV]'},
{'transition':'','wavevac':3869, 'label':u'[NeIII]'},
{'transition':'','wavevac':3968, 'label':u'[NeIII]'},
{'transition':'','wavevac':3729.875, 'label':u'[OII]'},
{'transition':'','wavevac':4363  , 'label':u'[O III]+2nd order line'},
# declutter {'transition':'','wavevac':4640  , 'label':u'N III*'},
{'transition':'','wavevac':4649, 'label':u'C III*'},
# declutter {'transition':'','wavevac':4649, 'label':u'C III*'},
{'transition':'','wavevac':4959  , 'label':u'[O III]'},
{'transition':'','wavevac':5007  , 'label':u'[O III]'},
{'transition':'','wavevac':5755  , 'label':u'[N II]'},
#{'transition':'','wavevac':.0, 'label':u''}
#{'transition':'','wavevac':.0, 'label':u''}
  ],
'WN':  [   # strong HeII spectrum 
# 1720 NIV?, 1750 NIII mult 1,4 
{'name':'H-beta'   ,'transition':'2-4'  ,'wavevac':4862.69,   'label':r'H$\beta$'},
{'name':'H-gamma'  ,'transition':'2-5'  ,'wavevac':4341.69,   'label':r'H$\gamma$'},
{'name':'H-delta'  ,'transition':'2-6'  ,'wavevac':4102.899,  'label':r'H$\delta$'},
{'transition':'4 - 6','wavevac':6562.0, 'label':u'HeII'},
{'transition':'4 - 7','wavevac':5411.5, 'label':u'HeII'},# see J.D.Garcia and J.E. Mack,J.Opt.Soc.Am.55,654(1965)
{'transition':'3 - 4','wavevac':4687.1, 'label':u'HeII'},
{'transition':'3 - 5','wavevac':3203.95,'label':u'HeII'},
{'transition':'3 - 6','wavevac':2734.13,'label':u'HeII'},
{'transition':'3 - 7','wavevac':2511.2, 'label':u'HeII'},
{'transition':'3 - 8','wavevac':2385.4, 'label':u'HeII'},
#{'transition':'3 - 9','wavevac':2306.0, 'label':u'He II'},
{'transition':'2 - 3','wavevac':1640.47,'label':u'HeII'},
{'transition':'2 - 4','wavevac':1215.17,'label':u'HeII'},
{'transition':'2 - 6','wavevac':1025.30,'label':u'HeII'},
{'transition':'','wavevac':1874.0, 'label':u'NIII 1876.6'},
{'transition':'','wavevac':1984.0, 'label':u'N VI 1989.3'},
{'transition':'','wavevac':2660.0, 'label':u'NIV 2647'},
{'transition':'','wavevac':2903.0, 'label':u'N IV 2897.2'},
{'transition':'','wavevac':2990.0, 'label':u'2980.8 N III'},
{'transition':'','wavevac':3485.0, 'label':u'N IV 3481.8'},
{'transition':'','wavevac':3743.0, 'label':u'3748.6 N IV'},
{'transition':'','wavevac':3909.0, 'label':u'O II 3915.5'},
{'transition':'','wavevac':4058.0, 'label':u'N IV 4058.9'},
{'transition':'','wavevac':4207.0, 'label':u'N III 4201.3'},
{'transition':'','wavevac':4550.0, 'label':u'N III 4544.4'},
{'transition':'','wavevac':4630.0, 'label':u'N III 4639.8'}, # blend HeII 4686
{'transition':'','wavevac':5520.0, 'label':u'[O V] 5523.9'},
  ],
}  
  
def continuum_nova_del_lc(regions = [[2010,2040],[2600,2700],
     [3530,3580],[3580,3630],[3930,3960],[4030,4050],
     [4170,4200],[4430,4460],[4750,4800]],
     phafiles=[]):
    import numpy as np
    #phafiles = rdTab('list_phafiles_g.txt')

    z = get_continuum(phafiles,regions=regions, tstart=398093612.4,)
    wave=z[0]
    time = np.asarray(z[1])
    sp = np.array(z[2]) 
    # byindex: (spectral bands, observations, [0=mean wave,1=flux,2=err])
    # find a normalisation of the lc using t~110 days
    q = (time > 60.) & (time < 120.) & np.isfinite(sp[0,:,1])
    norm=[]
    M = len(regions)
    for k in range(M):
        norm.append(sp[k,q,1].mean()*1e13)
    norm = np.array(norm)
    norm = norm/norm[1]           
    # find late time background fit
    q = (time > 107.)
    q1 = np.isfinite(sp[0,q,1]) & np.isfinite(sp[1,q,1]) & np.isfinite(sp[2,q,1])\
       & np.isfinite(sp[3,q,1]) & np.isfinite(sp[4,q,1]) & np.isfinite(sp[5,q,1])\
       & np.isfinite(sp[6,q,1]) & np.isfinite(sp[7,q,1]) & np.isfinite(sp[8,q,1])
    sptot = sp[0,q,1][q1]
    for k in range(1,M):
        sptot += sp[k,q,1][q1]
    spmean = sptot/M # (this the unnormalised bg lc-s)
    #   linear fits don't work : use log time
    coef2 = np.polyfit(np.log10(time[q][q1]),spmean,2)
    coef1 = np.polyfit(np.log10(time[q][q1]),spmean,1)
    normmean = norm.mean()  #use for scaling norm to mean flux spmean
    return (wave,time,sp, norm), coef, coef2, spmean,normmean    
    
def continuum_nova_del_byobs(obsday,wave,time,sp,norm):
    import numpy as np
    from scipy.interpolate import interp1d
    norm = np.asarray(norm)
    k = np.where( np.abs(obsday - time) < 0.04)[0]
    print('processing day = ',time[k])
    bgcoef = np.polyfit(wave,sp[:,k,1].flatten()/norm,1)
    w = np.arange(1650,7000,25)
    ww = [1650]
    for w1 in wave: ww.append(w1)
    ww.append(7000)
    nn = [norm[0]]
    for n1 in norm: nn.append(n1)
    nn.append(norm[-1])
    fnorm = interp1d(ww,nn,bounds_error=False,)
    return interp1d(w,np.polyval(bgcoef,w) * fnorm(w),bounds_error=False ) 
  
def actual_line_flux(wavelength,flux, center=None,pass_it=True):
    """Measure actual line flux:
    
    parameters
    ----------
    wavelength: float array
    flux: float array
    center: float 
       wavelength to center plot on
    
    output parameters
    -----------------
    flux in line integrated over region
    flux in background over same region
    
    Notes
    -----
    In novae the line profile is composed of a superposition of emission from 
    different regions, sometimes optically thick, sometimes thin, but not gaussian in 
    shape. 
    
    Here we plot the profile 
    provide endpoints (w1,f1), (w2,f2) at the flux level of the background
    """
    import numpy as np
    from pylab import plot,xlim, ylim, title, xlabel, ylabel, ginput, figure, subplot
    from scipy.interpolate import interp1d
    # find plot center and range 
    if type(center) == type(None): center=wavelength.mean()
    x1 = center - 7./300*center
    x2 = center + 7./300*center
    q = (wavelength > x1) & (wavelength < x2)
    w = wavelength[q]
    flx = flux[q]
    y2 = flx.max()
    f = figure()
    ax = subplot(111)
    getit = True
    while getit:
       ax.plot(w,flx,ls='steps',color='darkblue')
       print ("please click the desired limits of the profile at the background level")
       print ("no timeout")
       aa = ginput(n=2,timeout=0)
       x1,y1 = aa[0]
       x2,y2 = aa[1]
       x1 = float(x1)
       x2 = float(x2)
       y1 = float(y1)
       y2 = float(y2)
       q = (w >= x1) & (w <= x2)
       bg = interp1d([x1,x2],[y1,y2],)
       ax.fill_between(w[q], bg(w[q]), flx[q], color='c')
       ans = input("Do you want to continue ?")
       if (ans.upper()[0] != 'Y') & (pass_it == False) :
           print ("Answer is not yes\n TRY AGAIN ")
           ax.cla()
       else:
           getit = False           
    # not compute the fluxes
    w = w[q]
    flx = flx[q]
    tot_flx = []
    tot_bkg = (y2+y1)*(x2-x1)*0.5 
    for k in range(1,len(w)):
        tot_flx.append(0.25*(flx[k-1]+flx[k])*(w[k-1]+w[k]))
    line_flx = np.asarray(tot_flx).sum() - tot_bkg
    print (type(line_flx), line_flx)
    print (type(tot_bkg), tot_bkg)
    print (type( 0.5*(x2+x1) ), 0.5*(x2+x1) )
    print ( (" wavelength = %10.2f\n line flux = %10.3e\n"+
        " background flux = %10.2e\n FWZI = %10.2f\n")
        %( (x2+x1)*0.5, line_flx, tot_bkg, (x2-x1) ) )   
    return {'wavelength':(x2+x1)*0.5,
            'line_flux':line_flx,
            'integrated_background_flux':tot_bkg,
            "FWZI":(x2-x1)}
      
############################

def plot_line_ids(ax,ylower=None,ion='HI',color='k',dash=[0.07,0.10],
       fontsize=8,spdata=spdata):
       
   """add the line ids to the plot
   
   parameters
   ----------
   ax : plot handle
   ylower : float
     y-level where the bottom of the line should be
   ion : ['HI','HeI','HeII',]     
     key to the ion to be plotted 
     one at a time
   spdata : dict  
     dictionary of lines
   dash : [scale_factor, offset_text ]
     dash line from ylower to ylower+scale_factor
     text from ylower+offset
     The default is fine for linear axes, but for log axis, it fails
       try for a log scale.a scale_factor that is sized by the log scale 
       from ylower to ylower+scale_factor (so if ylower is 2x** and 
       scale_factor=0.1, the line will go from 2-3x**.  
    fontsize : int
       font size   
       
   """
   
   xlist = spdata[ion]
   xlim = ax.get_xlim()
   ylim = ax.get_ylim()
   dy = dash[0]*(ylim[1]-ylim[0])
   dy1 = dash[1]*(ylim[1]-ylim[0]) 
   
   wave = []
   for line in xlist:
      if (line['wavevac'] > xlim[0]) & (line['wavevac'] < xlim[1]):
          ax.text(line['wavevac'],ylower+dy1,line['label'],fontsize=fontsize,color=color,
              horizontalalignment='center',verticalalignment='bottom',
              rotation='vertical' )
          wave.append(line['wavevac'])
   ax.vlines(wave,ylower,ylower+dy,color='k')     


############################


class DraggableSpectrum(object):
    """
    Drag spectrum until the wavelengths are correctly lined up 
    """
    def __init__(self, ax, spectrum,):
        self.spectrum = spectrum
        self.press = None
        self.delwav = 0.0
        self.incwav = 0.0
        self.ax = ax
        self.cidpress = None
        self.cidrelease = None
        self.cidmotion = None
        self.cidkey = None


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
        print("active")    

    def on_press(self, event):
        'on button press we will  store some data'
        if event.inaxes != self.spectrum.axes: return
        self.press = event.x, event.y, event.xdata, event.ydata, self.spectrum.get_xdata()
        print("start position (%f,%e)"%(event.xdata,event.ydata))

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
            print("end position (%f,%e)"%(event.xdata,event.ydata))
            
    def on_key(self,event):
        'on press outside canvas disconnect '       
        print("you pushed the |%s| key"%event.key)
        print("disconnecting ...")

    def disconnect(self):
        'disconnect all the stored connection ids'
        self.spectrum.figure.canvas.mpl_disconnect(self.cidpress)
        self.spectrum.figure.canvas.mpl_disconnect(self.cidrelease)
        self.spectrum.figure.canvas.mpl_disconnect(self.cidmotion)
        self.spectrum.figure.canvas.mpl_disconnect(self.cidkey)
        print("disconnected")
        
    def out_delwav(self):
        return self.delwav
                

def adjust_wavelength_manually(file=None,openfile=None,openplot=None,
    ylim=[None,None],ions=['HI','HeII'],reference_spectrum=None,
    recalculate=True, figno=None):
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
    import sys
    from uvotpy import uvotmisc
    from matplotlib.pyplot import fignum_exists
    
    # data
    if openfile != None:
       f = openfile 
       filename = openfile.filename()
       if f.fileinfo(1)['filemode'] != 'update' :
          print("reopening the fits file %s with mode set to update"%(filename))
          filename = f.filename()
          try: 
             f.close()
             f = fits.open(filename,mode='update')
          except:
             raise "reopen fits file %s with mode set to update, and rerun "%(filename)
                  
    elif file != None:
       f = fits.open(file,mode='update')
       filename=file
    else:
       raise IOError("what ? nothing to adjust?")
          
    # axis instance to use   
    if openplot != None:
       fig = openplot
       filename = openplot.filename()
       fig.clf()
    else:
       fig = plt.figure(figno)
       fig.clf()
    fignum = fig.number   
    fig.set_facecolor('lightgreen')
    ax = fig.add_axes([0.05,0.13,0.9,0.77]) 
    canvas = ax.figure.canvas 
    ax.set_title("")   
    # initial plot to get started
    if len(f) <= 2:
       # up to one extension : probably a summed spectrum
       try: 
          w   = f['SUMMED_SPECTRUM'].data['wave']
          flx = f['SUMMED_SPECTRUM'].data['flux']
          extname = 'SUMMED_SPECTRUM'
          recalculate = False 
          # in general the summed spectra can be from different detector 
          # locations with different dispersion coefficients, so at this 
          # stage no adjustments can be made. Adjust the wavelengths 
          # before summing to within ~20A is recommendation. 
       except:
          raise IOError("unknown kind of spectral file")          
    else:
       try:
          w = f['CALSPEC'].data['lambda']
          flx = f['CALSPEC'].data['flux']
          extname = 'CALSPEC'
       except:
          raise IOError("unknown kind of spectral file")
                  
    spectrum, = ax.plot(w, flx,color='b',label='spectrum to fix' )
    if reference_spectrum != None:
       colname = list(reference_spectrum.columns.keys())
       refsp, = ax.plot(reference_spectrum[colname[0]],reference_spectrum[colname[1]],
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
    print("Before continuing select a part of the figure to use for shifting the wavelengths.")
    #  drag figure
    #background = canvas.copy_from_bbox(ax.bbox)
    newspec = DraggableSpectrum(ax,spectrum)
    done = False
    if 'WAVSHFT' in f[extname].header:
        delwav0 = f[extname].header['WAVSHFT']
        delwav = 0
    else:
        delwav0 = 0
        delwav = 0
    try:
        #ans1 = input("Do you want to adjust wavelengths ? (Y/N) ").upper()
        ans1 = input("Do you want to adjust wavelengths ? (Y/N) ").upper()
        print("answer read = ", ans1," length = ", len(ans1))
        if len(ans1) > 0:
          if ans1[0] == 'Y':
             while not done:
                print('drag the spectrum until happy')
                ax.set_title(filename+" when done press key")   
                newspec.connect()
                print("The selected wavelength shift is ",newspec.delwav," and will be applied when done. ") 
                # register the shift from the last run          
                ans = input("When done hit a key\n")
                delwav += newspec.out_delwav()
                ax.set_title(filename)
                done = True
             newspec.disconnect()
          elif ans1[0] == 'N': 
             done = True
          else: print(" answer Y or N ")
    except:
        sys.stderr.write("error: wavshift %f\n"%(delwav) )
        newspec.disconnect()
        raise RuntimeError(
        "An error occurred during the selection of the wavelength shift."+\
        "\nNo shift was applied.")
    if recalculate:
            print("recalculating wavelength scale after finding shift")
            if 'PIXSHFT' in f[extname].header: 
                pixshift0 = f[extname].header['PIXSHFT']
            else: pixshift0 = 0         
            C_1 = uvotmisc.get_dispersion_from_header(f['SPECTRUM'].header)
            C_2 = uvotmisc.get_dispersion_from_header(f['SPECTRUM'].header,order=2)
            delpix = int(round(delwav / C_1[-2]))  # round to nearest int
            pixno = f[extname].data['pixno'] + delpix 
            yes2ndorder = False
            if 'pixno2'.upper() in f[extname].data.names:
               yes2ndorder = True
               pixno2= f[extname].data['pixno2']+ delpix 
            print("wavelength shift found = %f; which results in a pixno shift of %i"%(delwav,delpix))
            f[extname].data['pixno'] = pixno
            if yes2ndorder: f[extname].data['pixno2'] = pixno2 
            f[extname].data['lambda'] = np.polyval(C_1,pixno)
            f[extname].header['PIXSHFT'] = (delpix+pixshift0, "pixno shift + recalc lambda from disp")
            f[extname].header['PIXSHFT'] = (delpix+pixshift0, "pixno shift + recalc lambda from disp")
            h = f['SPECTRUM'].header['history']
            if yes2ndorder:
               dist12 = float(uvotmisc.get_keyword_from_history(h,'DIST12'))
               f[extname].data['lambda2'] = np.polyval(C_2,pixno2-dist12)
            # now we should update the plot...
    else:
           sys.stderr.write( "wavelength shift found = %s\n"%(delwav) )
           f[extname].header['WAVSHFT'] = (delwav+delwav0, "manual wavelength shift applied")
           if extname == 'CALSPEC':
              f[extname].data['LAMBDA'] = f[extname].data['LAMBDA'] + delwav  
              f['SPECTRUM'].header['WAVSHFT'] = (delwav+delwav0, "manual wavelength shift applied")
              f['SPECTRUM'].header['COMMENT'] = "Manual wavelength shift not applied to response file."
           elif extname == 'SUMMED_SPECTRUM':
              f[extname].data['WAVE'] = f[extname].data['WAVE'] + delwav      
    f.verify()
    f.flush() 
    spectrum.set_color('c')
    #if openfile != None:
    #   f = openfile 
    #elif file != None:
    #   f = fits.open(file,)
    if extname == 'SUMMED_SPECTRUM':
            w = f[extname].data['wave']
            flx = f[extname].data['flux']
    else:
            w = f[extname].data['lambda']
            flx = f[extname].data['flux']
    spectrum, = ax.plot(w, flx,color='darkblue',label='fixed spectrum' )
    ax.set_title(filename) 
    ax.legend()
    if fignum_exists(fignum):
        ax.figure.canvas.draw()
    return fig, ax, spectrum, f

def apply_shift(file,delwav,recalculate=False):
    """apply a given wavelength shift in A"""
    from uvotpy import uvotmisc
    f = fits.open(file,mode='update')
    # check type of file
    getspecoutput = False
    for ff in f: 
        if 'CALSPEC' in ff.header: getspecoutput = True
    if getspecoutput:    
        delwav0 = 0
        if 'WAVSHFT' in f['CALSPEC'].header:
            delwav0 = f['CALSPEC'].header['WAVSHFT']+delwav
        if recalculate:
            if 'PIXSHFT' in f['CALSPEC'].header: 
                pixshift0 = f['CALSPEC'].header['PIXSHFT']
            else: 
                pixshift0 = 0    
            C_1 = uvotmisc.get_dispersion_from_header(f[1].header)
            C_2 = uvotmisc.get_dispersion_from_header(f[1].header,order=2)
            delpix = int(round(delwav / C_1[-2]))
            pixno  = f['CALSPEC'].data['pixno']  +delpix
            f['CALSPEC'].data['pixno'] = pixno  
            f['CALSPEC'].data['lambda'] = np.polyval(C_1,pixno)
            f['CALSPEC'].header['PIXSHFT'] = (delpix+pixshift0, "pixno shift + recalc lambda from disp")
            h = f['SPECTRUM'].header['history']
            dist12 = float(uvotmisc.get_keyword_from_history(h,'DIST12'))
            if 'PIXNO2' in  f['CALSPEC'].header:
               pixno2 = f['CALSPEC'].data['pixno2'] +delpix   
               f['CALSPEC'].data['pixno2'] = pixno2  
               f['CALSPEC'].header['PIXSHFT2'] = (delpix+pixshift0, "pixno shift + recalc lambda from disp")
               f['CALSPEC'].data['lambda2'] = np.polyval(C_2,pixno2-dist12)
        else:       
            f['CALSPEC'].header['WAVSHFT'] = (delwav+delwav0, "manual wavelength shift applied")
            f['CALSPEC'].data['LAMBDA'] = f['CALSPEC'].data['LAMBDA'] + delwav    
            f['SPECTRUM'].header['WAVSHFT'] = (delwav+delwav0, "manual wavelength shift applied")
        f.verify()
        f.flush()
    else:
        extname = 'SUMMED_SPECTRUM'
        print ("currently this program cannot recalculate the dispersion")
        f.verify()
        f.flush()
        return
        delwav0 = 0
        if 'WAVSHFT' in f[extname].header:
            delwav0 = f[extname].header['WAVSHFT']+delwav
        if recalculate:
            if 'PIXSHFT' in f[extname].header: 
                pixshift0 = f[extname].header['PIXSHFT']
            else: 
                pixshift0 = 0    
            C_1 = uvotmisc.get_dispersion_from_header(f[1].header)
            C_2 = uvotmisc.get_dispersion_from_header(f[1].header,order=2)
            delpix = int(round(delwav / C_1[-2]))
            pixno  = f[extname].data['pixno']  +delpix
            f[extname].data['pixno'] = pixno  
            f[extname].data['lambda'] = np.polyval(C_1,pixno)
            f[extname].header['PIXSHFT'] = (delpix+pixshift0, "pixno shift + recalc lambda from disp")
            if 'PIXNO2' in  f[extname].header:
               pixno2 = f[extname].data['pixno2'] +delpix   
               f[extname].data['pixno2'] = pixno2  
               f[extname].header['PIXSHFT2'] = (delpix+pixshift0, "pixno shift + recalc lambda from disp")
               f[extname].data['lambda2'] = np.polyval(C_2,pixno2-dist12)
        else:       
            f[extname].header['WAVSHFT'] = (delwav+delwav0, "manual wavelength shift applied")
            f[extname].data['LAMBDA'] = f[extname].data['LAMBDA'] + delwav    
        f.verify()
        f.flush()
                



class SelectBadRegions(object):
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
        print("active")    

    def on_press(self, event):
        'on button press we check if near endpoint of region'
        if event.inaxes != self.spectrum.axes: return
        self.region[1] = None
        # check if near existing Line2D (adjust) 
        if len(self.badregions) > 0:
            print("going through badregions")
            for self.line in self.badregions:
                xdat = self.line.get_xdata()
                #print "*** ",np.abs(xdat[0] - event.xdata)
                #print "*** ",np.abs(xdat[1] - event.xdata)
                #print "*** ",xdat
                if (np.abs(xdat[0] - event.xdata) < self.epsilon) :
                    print("at point ",xdat[0]," keeping ",xdat[1]," swapping") 
                    k = self.badregions.index(self.line)
                    xx = self.badregions.pop(k)
                    self.line.set_xdata(np.array([xdat[1],event.xdata]))              
                elif (np.abs(xdat[1] - event.xdata) < self.epsilon):
                    print("at point ",xdat[1]," keeping ",xdat[0])
                    k = self.badregions.index(self.line)
                    xx = self.badregions.pop(k)
                    self.line.set_xdata(np.array([xdat[0],event.xdata]))              
                else:          
                    print("new line")
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
        print("position [%f,*]"%(event.xdata,))

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
            print("-> position (%f,%e)"%(event.xdata,event.ydata))
            
    def on_key(self,event):
        'on press outside canvas disconnect '       
        print("you pushed the |%s| key"%event.key)

    def disconnect(self):
        'disconnect all the stored connection ids'
        self.spectrum.figure.canvas.mpl_disconnect(self.cidpress)
        self.spectrum.figure.canvas.mpl_disconnect(self.cidrelease)
        self.spectrum.figure.canvas.mpl_disconnect(self.cidmotion)
        self.spectrum.figure.canvas.mpl_disconnect(self.cidkey)
        print("disconnected")
        
    def get_badlines(self):
        lines = []
        for r in self.badregions: 
            lines.append(r.get_xdata())
        return self.badregions, lines   
        
    def set_badregions(self,badregions):
        self.badregions = badregions    
        

def flag_bad_manually(file=None,openfile=None,openplot=None,
        ylim=[None,None],apply=[] ):
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
    apply : list
       list of [a,b] lists where the wavelengths a-b must be flagged 'bad'   
       disables the interactive way
       
    Notes
    -----
    returns 
    ax:axes instance, fig:figure instance, [f:fits file handle if passed with openfile]
    
    The data quality flag of flagged pixels will be set to "bad"
    The header will be updated with the value of the wavelength shift 
    
    """
    from uvotpy.uvotgetspec import quality_flags
    if openfile != None:
       f = openfile
       if f.fileinfo(1)['filemode'] != 'update' :
          print("reopening the fits file with mode set to update")
          filename = f.filename()
          try: 
             f.close()
             f = fits.open(filename,mode='update')
          except:
             raise IOError("reopen fits file with mode set to update, and rerun ")
    elif file != None:
       f = fits.open(file,mode='update')
    else:
       raise IOError("what ? nothing to adjust?")
       
    interactive = True   
    if apply != []: interactive=False  
    
    lines = []   
    if interactive:
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
       
    # determine if summed spectrum
    ext = 2
    wave = 'lambda'
    if f[1].header['extname'] == 'SUMMED_SPECTRUM':
       ext = 1
       wave = 'wave'
       
    if interactive:   
       # initial plot to get started
       w = f[ext].data[wave]
       flx = f[ext].data['flux']
       spectrum, = ax.plot(w, flx, )
       
    # highlight bad quality 
    q = f[ext].data['quality']
    flag = quality_flags()
    if interactive:
       plotquality(ax,w,q,flux=flx,flag=['bad','zeroth','weakzeroth','overlap','too_bright']) 
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
       print("Select bad regions: Zoom in before starting. Rerun for more regions.")
       #  when animating / blitting figure
       #background = canvas.copy_from_bbox(ax.bbox)
       s = SelectBadRegions(ax,spectrum)
       s.set_badregions([])
    #flag = quality_flags()
       done = False
       try:
           while not done:
               ans = input("Do you want to mark bad regions ? (Y) ").upper()
               if len(ans) > 0:
                   if ans[0] == 'Y':
                       print('Select bad wavelengths in the spectrum until happy')
                       ax.set_title("when done press key")   
                       s.connect()
                       # register the shift from the last run              
                       ans = input("When done hit the d key, then return, or just return to abort")
                       badregions, lines = s.get_badlines()
                       print("got so far: ")
                       for br in lines: print("bad region : [%6.1f,%6.1f]"%(br[0],br[1]))
                       print(badregions)
                       ax.set_title("")
                       s.disconnect()
                   else:
                       print("Done for now...")            
                       done = True
       except:
           raise RuntimeError("Some error occurred during the selection of the bad regions. No changes were applied.")
           s.disconnect()
           lines = []
           #
    if not interactive: lines=apply       
    if len(lines) > 0:
        print("The selected bad regions are ")
        for br in lines: print("bad region : [%6.1f,%6.1f]"%(br[0],br[1]))
        print(" and will be applied to the FITS file.\n ") 
        f[ext].header['comment'] = "added bad regions manually (qual=bad)"
        for br in lines:
          #try:
            # find points that are not flagged, but should be flagged
            if br[1] < br[0]:
               br3 = br[0]; br[0]=br[1]; br[1]=br3
            q1 = (check_flag(f[ext].data['quality'],'bad')  == False)
            q = ((f[ext].data['lambda'] > br[0]) & 
                     (f[ext].data['lambda'] < br[1]) & 
                     q1 & 
                     np.isfinite(f[ext].data['quality']) )
            if ext == 2: 
               f[1].data['QUALITY'][q] = f[1].data['QUALITY'][q] + flag['bad'] 
            f[ext].data['QUALITY'][q] = f[ext].data['QUALITY'][q] + flag['bad']    
          #except:
            #    raise RuntimeError("Some error occurred during writing to file of the bad regions. No changes were applied.")
            #    s.disconnect()
        f.verify()
        f.flush()
        print("file was updated")
    print(type(f))      
    if file == None:    
       return fig, ax, f
    else:
       f.close() 
       if interactive:
          return fig,ax   


def plotquality(ax,
       w, quality,
       flux = None,
       flag=['bad','weakzeroth','zeroth','too_bright'],
       colors=['c','y','m','b','r','g','k'],alpha=0.2,
       quallegend={'bad':True,'weakzeroth':True,'zeroth':True,'overlap':True,'too_bright':True,'first':True},
       marker='x',
       speccolor='b', 
       label=None,
       chatter=0):
       """ mark up plot with data quality
       either add vertical greyscale regions in plot for each 
          quality flag (when marker=None) 
       or plot a symbol 'marker' on each data point
       
       parameters
       ----------
       ax : matplotlib.axes.Axes instance
       w : array 
         x-axis values
       quality : array 
       flux : array 
         flux values 
         quality flags matching x-axis points
       flag : list of strings
         each list value must be one of the valid keys from quality_flags()
       colors : array 
         color values             
       alpha : float
         alpha value for transparency
       marker : nonetype or plot character string  
         if None then plot grey scale regions, otherwise, plot marker on bad data 
         requires flux array 
          
       Note
       -----
       Should add an option to plot quality in a different way. 
       """
       from .uvotgetspec import quality_flags
       
       typeNone = type(None)
       
       if (type(flux) == typeNone) & (type(marker) != typeNone): 
          raise IOError("plotquality with marker requires flux array")
       
       flagdefs = quality_flags()
       k=0
       for fla in flag:
           if chatter > 2: 'processing flag = ',fla
           dolabel = True
           fval = flagdefs[fla]
           if fval == 1:
               loc = quality == fval
               v = np.where(loc)[0]
           else:
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
               if vlast > v1: vrange.append([v1,vlast])  # last range       
               if chatter > 2: print("for quality="+fla+" we get ranges ",vrange)        
               for v1 in vrange: 
                   flab = None
                   if dolabel & quallegend[fla]: 
                      flab = fla
                      dolabel = False
                   if type(marker) == typeNone:   
                       ax.axvspan(w[v1[0]],w[v1[1]],facecolor=colors[k],alpha=alpha,label=flab)
                   else:
                       qp = (w >= w[v1[0]]) & (w <= w[v1[1]])
                       # mfc marker face colors
                       ax.plot(w[qp],flux[qp], marker=marker, mfc=colors[k],ms=3,
                          label=flab,color=speccolor)  
       # the algorithm skips two adjacent points which will be ignored.                    
                                      
def check_flag(quality,flag,chatter=0):
   """ return a logical array where the elements are 
       True if flag is set """
   from uvotpy.uvotgetspec import quality_flags
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
   if chatter> 4: print("flag = ",k," length=",mf)                 
   # skip the good ones    
   qz = np.where(quality > 0)[0]
   if chatter > 4: print("qual > 0 at ",qz)
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

def _write_xspec_xcm_script(scriptfilename, 
         E_BmV=0.2, 
         vel=1000.0,
         ions=['V339_Del'],
         clobber=False,chatter=0):
    """
    Convenience function to help create an xspec script file for 
    the continuum and lines
    
    Parameters
    -----------
    scriptfilename : path
       path+name for the script file
    E_BmV : float
       E(B-V) colour excess 
    vel : float
       velocity responsible for line broadening in km/s   
    
    """
    import os 
    
    #conversion factor
    kev2A = 12.3984191
    
    # check if scriptfilename exists
    present = os.access(scriptfilename,os.F_OK)
    if present and not clobber:
       raise IOError('script file is present and clobber is not set')

    # initialize model line
    model = "model redden(bbody + powerlaw + redge "
    
    # initialize body 
    body = []
    # add redden parameter
    body.append("%15.3f%11.3f%11.3e%11.3e,%11.3e%11.3e\n"%(E_BmV,-1e-3,1e-5,1e-3,1,10))  
    # add BB parameters
    body.append("%15.3f%11.3f%11.3e%11.3e%11.3e%11.3e\n"%(0.025,0.01,0,0,100,200))
    body.append("%15.3f%11.3f%11.3e%11.3e%11.3e%11.3e\n"%(0.5,0.01,0,0,1e+20,1e+24))
    # add Powerlaw parameters
    body.append("%15.3f%11.3f%11.3e%11.3e%11.3e%11.3e\n"%(1.5,0.01,-3,-2,9,10))
    body.append("%15.3f%11.3f%11.3e%11.3e%11.3e%11.3e\n"%(4.5,0.01,0,0,1e+20,1e+24))
    # add redge parameters
    body.append("%15.3f%11.3f%11.3e%11.3e%11.3e%11.3e\n"%(3.167e-3,3e-4,1e-3,1.8e-3,7.4e-3,9e-3))
    body.append("%15.3f%11.3f%11.3e%11.3e%11.3e%11.3e\n"%(1e-3,1e-3,1e-3,1e-3,100,100))
    body.append("%15.3f%11.3f%11.3e%11.3e%11.3e%11.3e\n"%(10,0.01,0,0,1e+20,1e+24))
     
    for ion in ions:
       if not ion in list(spdata.keys()):
          print("unrecognised ion",ion)
       else:   
          data = spdata[ion]
          for rec in data:
             en = kev2A/rec['wavevac'] 
             if chatter > 0: print(rec['wavevac'], en)    
             de = vel/3e5 * en
             body.append("%15.5e%11.3f%11.3e%11.3e%11.3e%11.3e\n"%(en,-0.05,1e-3,1.8e-3,7.4e-3,9e-3))
             body.append("%15.5e%11.3f%11.3e%11.3e%11.3e%11.3e\n"%(de,0.05,1e-6,5e-6,3e-5,8e-5))
             body.append("%15.3e%11.3f%11.3e%11.3e%11.3e%11.3e\n"%(10,0.01,0,0,1e+20,1e+24))
             model += " + gaussian"
    # close model line
    model += ")\n"
    out = open(scriptfilename,"w")
    out.write(model)
    for rec in body:
        out.write(rec)
    out.close()
        


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
       a list comprised of average wavelength, mean continuum flux, and 
         standard deviation from the mean flux each band 
       
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
    
def get_continuum(phafiles,regions=[],qlimit=1,tstart=0,daily=True,
    full=True,chatter=0):
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
    for filex in phafiles:
       if chatter > 1: print("reading ",filex)
       try:
          sp = fits.open(filex)
       except:   
          sp = fits.open(filex[0])
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
    
    
    
def plot_spectrum(ax,spectrumfile,
        errbars=False, errhaze=False, hazecolor='grey', hazealpha=0.2, 
        flag='all',
        ylim=[6e-15,9e-13],
        speccolor = 'g',
        plot_quality=True, quality_marker='x',
        qual_flags=['bad','weakzeroth','zeroth','overlap','too_bright','first'],
        qualcolors=['c', 'g', 'y', 'm', 'b', 'r', 'k'], 
        qualalpha=0.2,
        quallegend={'bad':True,'weakzeroth':True,'zeroth':True,'overlap':True,'too_bright':True,'first':True},
        smooth=1, 
        linewidth=1,
        label=None,
        offset=0,offsetfactor=1,
        ebmv=0.00, Rv=3.1,
        redshift=0.,
        wrange=[1680,6800],
        chatter=0):
    """
    make a quick plot of a PHA/summed spectrum 
    
    Parameters
    ==========
    ax : matplotlib.axes.AxesSubplot instance
       The spectrum will be drawn in 'ax'
    spectrumfile: path
       the full filename of the spectrum
    wrange: list
       limits of the wavelength range to plot [wmin,wmax]  
    ylim : list 
       y-axis limits
    speccolor : color   
    errbars: bool
       if False, draw spectrum with optional error as a haze
       (see errhaze, hazecolor, hazealpha)
       if True, draw data with error bars 
    errhaze, hazecolor, hazealpha : bool, string, float
       parameters to control the display of the error region 
       around the spectrum
    flag: one of quality_flags()
       for PHA spectrum: plot data excluding this flag
       e.g., 'all' include only 'good' data (no flag set)
    plot_quality : bool 
       (only for PHA files)
    quality_marker : plot string character or None 
       if None, plot with color shading, otherwise, plot symbol  
    qualflag,qualcolors,qualalpha : list, list, float
       (only for PHA files) parameters passed to plotquality 
       - qualflag is a list of flags (from uvotgetspec.quality_flags().keys())
       - qualcolors is a list of colors for the shading of the regions by quality
       - qualalpha is the alpha factor for the shading   
       - quallegend (for each flag, add to legend)
    smooth: int
       if not empty, apply a boxcar smooth over N points   
    label: str
       force this label    
    ebmv, Rv: float
       the value of E(B-V), and Rv to correct for reddening 
       (requires the photometry2 module)
       default: no reddening
    redshift: float
       z value   
    offset: float
       additive offset
    offsetfactor: float 
       multiplicative offset  
    chatter: int (0...5)
       verbosity  
       
    If a summed spectrum is presented, it needs to be FITS
    generated using the  uvotspec.sum_PHAspectra() program.               
    
    """
    import numpy as np
    from stsci.convolve import boxcar
    try:
       from photometry2 import Cardelli
    except: pass
     
    if type(ax) != 'matplotlib.axes.AxesSubplot' :
       if chatter > 2: print("ax type ?",type( ax ))
    if smooth < 1: smooth = 1   
    first = True   
    f = fits.open(spectrumfile)
    if f[1].header['extname'].upper() == 'SPECTRUM': 
        w = f[2].data['lambda']/(1.+redshift)
        flx = (f[2].data['flux']+offset)*offsetfactor
        err = f[2].data['fluxerr']
        q = f[2].data['quality']
        filterin = np.isfinite(flx) 
        w = w[filterin]
        flx = flx[filterin]
        err = err[filterin]
        q = q[filterin]
        rx = quality_flags_to_ranges(q)
        rx = rx[flag] 
        r = complement_of_ranges(rx,rangestart=0,rangeend=len(q)-1)
        if chatter > 1: 
           print("ranges",r)
        if type(label) == type(None):   # default label
           label = f[1].header['date-obs']
        if ebmv != 0.:
           X = Cardelli(wave=w*1e-4,Rv=Rv)
        else: 
           X = 0. 
        flx = flx*10**(0.4*X*ebmv)
        qw_ = (w > wrange[0]) & (w < wrange[1])
        if chatter > 4: 
           print ("wavelengths: ",w)
           print ("flux: ",flx)
           print ("err: ",err)
           print ("q: ",q)
           print ("r: ",r)
        if len(r) == 0:
           r = [[0,len(w)]]
        if not errbars:
            if chatter > 2: print("errbars False")
            for rr in r:
                if chatter > 4:
                    print ("segment indices rr: ",rr)
                    print ("len w[]:",len(w))
                if first:
                    qw = (w[rr[0]:rr[1]] > wrange[0]) & (w[rr[0]:rr[1]] < wrange[1])
                    ax.plot(w[rr[0]:rr[1]][qw],
                        boxcar( flx[rr[0]:rr[1]],(smooth,))[qw],
                        color=speccolor,
                        label=label)
                    first = False
                else:
                    qw = (w[rr[0]:rr[1]] > wrange[0]) & (w[rr[0]:rr[1]] < wrange[1])
                    ax.plot(w[rr[0]:rr[1]][qw],
                         boxcar(flx[rr[0]:rr[1]],(smooth,))[qw],
                         color=speccolor)
                if errhaze:
                    smi = np.int( 5*smooth )
                    qw = (w[rr[0]:rr[1]] > wrange[0]) & (w[rr[0]:rr[1]] < wrange[1])
                    flxlower=flx[rr[0]:rr[1]]-boxcar(err[rr[0]:rr[1]],(smi,))[qw]/np.sqrt(smooth)
                    flxupper=flx[rr[0]:rr[1]]+boxcar(err[rr[0]:rr[1]],(smi,))[qw]/np.sqrt(smooth)
                    flxlower[flxlower <= 0] = 1e-27
                    flxupper[flxupper < 0] = 1e-27
                    ax.fill_between(w[rr[0]:rr[1]][qw],
                    flxlower, 
                    flxupper,
                    where=flxlower > 0,
                    color=hazecolor,alpha=hazealpha)
        else:
            if chatter > 2: print("errbars True")
            for rr in r:
                if first:
                   qw = (w[rr[0]:rr[1]] > wrange[0]) & (w[rr[0]:rr[1]] < wrange[1])
                   ax.errorbar( w[rr[0]:rr[1]][qw],
                      boxcar(flx[rr[0]:rr[1]],(smooth,))[qw],
                      yerr=err[rr[0]:rr[1]]/np.sqrt(smooth)[qw],
                      label=label,color=speccolor,fmt='x')  
                   first = False
                else:
                   qw = (w[rr[0]:rr[1]] > wrange[0]) & (w[rr[0]:rr[1]] < wrange[1])
                   ax.errorbar( w[rr[0]:rr[1]][qw],
                      boxcar(flx[rr[0]:rr[1]],(smooth,))[qw],
                      yerr=err[rr[0]:rr[1]]/np.sqrt(smooth)[qw],
                      color=speccolor,fmt='x')  
        if plot_quality:
            plotquality(ax,w[qw_],q[qw_],flux=flx[qw_],flag=qual_flags,colors=qualcolors,
                alpha=qualalpha,marker=quality_marker,speccolor=speccolor)                                           
                   
    elif f[1].header['extname'].upper() == 'SUMMED_SPECTRUM':
        w = f['SUMMED_SPECTRUM'].data['wave']/(1+redshift)
        qw = (w > wrange[0]) & (w < wrange[1])
        if ebmv != 0.:
           X = Cardelli(wave=w*1e-4,Rv=Rv)
        else: 
           X = 0. 
        flx = (boxcar(f['SUMMED_SPECTRUM'].data['flux'],(smooth,))+offset)*offsetfactor*10**(0.4*X*ebmv)
        err = boxcar(f['SUMMED_SPECTRUM'].data['fluxerr'],(2*smooth,))/np.sqrt(smooth)
        if type(label) == type(None):
           label = f[1].header['date-obs']
        #n_spec = f['SUMMED_SPECTRUM'].data['n_spec']
        sector = f['SUMMED_SPECTRUM'].data['sector']
        sect = np.min(sector)
        last = np.max(sector)
        for s in range(sect,last+1):
            q = sector[qw] == s
            if not errbars:
                if first:
                    ax.plot(w[qw][q],flx[qw][q],label=label,color=speccolor)
                    first = False
                else:
                    ax.plot(w[qw][q],flx[qw][q],color=speccolor)       
                if errhaze:
                    flxlower=flx[qw][q]-err[qw][q]
                    flxlower[flxlower <= 0] = 1e-27
                    flxupper=flx[qw][q]+err[qw][q]
                    ax.fill_between(w[qw][q],
                    flxlower, 
                    flxupper,
                    where=flxlower > 0.,
                    color=hazecolor,alpha=hazealpha)
            else:
                if first:
                    ax.errorbar(w[qw][q],flx[qw][q],yerr=err[qw][q],label=label,color=speccolor)
                    first = False
                else:     
                    ax.errorbar(w[qw][q],flx[qw][q],yerr=err[qw][q],color=speccolor)
    else:
        raise IOError("Spectrum type (extname) is not recognised")
    ax.legend(loc=0)    
    ax.set_ylim(ylim[0],ylim[1])       
    ax.figure.show()


def quality_flags_to_ranges(quality,chatter=0):
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
       try:
          from uvotpy.uvotgetspec import quality_flags
       except:
          from .uvotgetspec import quality_flags   
       
       flagdefs = quality_flags()
       flags=list(flagdefs.keys())
       quality_ranges = {}
       val = []
       for fla in flags:
           if fla == 'good': continue
           fval = flagdefs[fla]
           #q = quality >= fval  # limit the search
           #indx = np.where(q)[0] # indx of wave where flag 
           #fval2 = fval*2
           #loc = quality[q]/fval2*fval2 != quality[q]
           #v = indx[loc] # indices of the points with this flag
           q = quality == fval 
           v = np.where(q)[0]
           if chatter > 4:
              print("flag=",fla,"  indices:",v) 
           if len(v) > 1:  # require at least 2
               vrange = []
               v1 = v[0]
               vlast = v1
               for v2 in v[1:]: 
                   if v2-vlast > 1:  # require neighboring pixels
                       # 
                       vrange.append([v1,vlast])
                       val.append([v1,vlast])
                       if chatter > 3: 
                          print("flag=",fla," +range=",[v1,vlast])
                       v1 = v2
                       vlast = v2
                   else:
                       vlast=v2
               if vlast > v1: 
                   vrange.append([v1,vlast])     # last range  
                   val.append([v1,vlast])     
                   if chatter > 3: 
                      print("flag=",fla," +range=",[v1,vlast])
               quality_ranges.update({fla:vrange}) 
       quality_ranges.update({"all":val})                  
       return quality_ranges


def complement_of_ranges(ranges,rangestart=0,rangeend=None):
    """given a list of exclusion ranges, compute the complement
    
    Parameters
    ==========
    range : list 
       the range list consists of elements that each specify a
       bad range in the spectrum. The complement needs also 
       the start and end of the whole range in order to add 
       the leading and trailing complement ranges.
    rangestart, rangeend : int
       start and end index of the wavelenght array. Usually
       that is 0, len(wave)  
    
    """
    if rangeend == None:
       raise IOError("complement_of_ranges requires a value for the last index of the range+1")
    out = []
    xr0 = 0
    for r in ranges:
       xr1 = r[0]
       if xr1-xr0 > 1: 
          out.append([xr0,xr1])
       xr0 = r[1]+1
    out.append([xr0,rangeend])   
    return out


class gaussian(object):
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

    
def fit2g_bg(x,f,err,bg,amp1,pos1,sig1,amp2,pos2,sig2,
    amp2lim=None,fixsig=False,
    fixsiglim=0.2, fixpos=False,
    fixamp=False, chatter=0):
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
   import mpfit
   
   #gp = np.array(gaussian_parameters,)
   
   #n_gaussians = len(gaussian_parameters)
   
   if np.isfinite(bg):
      bg0 = bg
   else: bg0 = 0.0   
   bg1 = 1.e-7
   if np.isfinite(sig1):
      sig1 = np.abs(sig1)
   else: sig1 = 15.0 
   if np.isfinite(sig2):
      sig2 = np.abs(sig2)
   else: sig2 = 15.0        

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
      sig1_lo = max([sig1-10.,3.])
      sig2_lo = max([sig2-10.,3.])
      sig1_hi = min([sig1+10.,70.])
      sig2_hi = min([sig2+10.,70.])
     
   if amp2lim != None:
      amp2min, amp2max = amp2lim
      parinfo = [{  \
   'limited': [1,0],   'limits' : [np.min([0.0,bg0]),0]   ,'value'  :    bg,   'parname': 'bg0' ,'fixed':False   },{  \
   'limited': [0,0],   'limits' : [None,None],             'value'  :   0.0,   'parname': 'bg1' ,'fixed':False   },{  \
   'limited': [1,0],   'limits' : [0.0,None],              'value'  :  amp1,   'parname': 'amp1','fixed':fixamp  },{  \
   'limited': [1,1],   'limits' : [pos1a,pos1b],           'value'  :  pos1,   'parname': 'pos1','fixed':fixpos   },{  \
   'limited': [1,1],   'limits' : [sig1_lo,sig1_hi],       'value'  :  sig1,   'parname': 'sig1','fixed':fixsig   },{  \
   'limited': [1,1],   'limits' : [amp2min,amp2max],       'value'  :  amp2,   'parname': 'amp2','fixed':fixamp  },{  \
   'limited': [1,1],   'limits' : [pos2a,pos2b],           'value'  :  pos2,   'parname': 'pos2','fixed':fixpos  },{  \
   'limited': [1,1],   'limits' : [sig2_lo,sig2_hi],       'value'  :  sig2,   'parname': 'sig2','fixed':fixsig  }]  
      
   else:  
      parinfo = [{  \
   'limited': [1,0],   'limits' : [np.min([0.0,bg0]),None],'value'  :    bg,   'parname': 'bg0'    },{  \
   'limited': [0,0],   'limits' : [None,None],             'value'  :   0.0,   'parname': 'bg1'    },{  \
   'limited': [1,0],   'limits' : [0.0,None],              'value'  :  amp1,   'parname': 'amp1'   },{  \
   'limited': [1,1],   'limits' : [pos1a,pos1b],           'value'  :  pos1,   'parname': 'pos1'   },{  \
   'limited': [1,1],   'limits' : [sig1_lo,sig1_hi],       'value'  :  sig1,   'parname': 'sig1'   },{  \
   'limited': [1,0],   'limits' : [0.0,None],              'value'  :  amp2,   'parname': 'amp2'   },{  \
   'limited': [1,1],   'limits' : [pos2a,pos2b],           'value'  :  pos2,   'parname': 'pos2'   },{  \
   'limited': [1,1],   'limits' : [sig2_lo,sig2_hi],       'value'  :  sig2,   'parname': 'sig2'   }]  

   if chatter > 4: 
      print("parinfo has been set to: ") 
      for par in parinfo: print(par)

   Z = mpfit(_fit2g,p0,functkw=fa,parinfo=parinfo,quiet=False)
   
   if (Z.status <= 0): 
      print('uvotgetspec.runfit2.mpfit error message = ', Z.errmsg)
      print("parinfo has been set to: ") 
      for par in parinfo: print(par)
   elif (chatter > 3):   
      print("\nparameters and errors : ")
      for i in range(8): print("%10.3e +/- %10.3e\n"%(Z.params[i],Z.perror[i]))
   
   return Z     
       
       
def _fit2g(p, fjac=None, x=None, y=None, err=None):
   import numpy as np
   
   from pylab import plot,figure

   (bg0,bg1,amp1,pos1,sig1,amp2,pos2,sig2) = p 
             
   model = bg0*(1 + bg1*(x-pos1)) + \
           amp1 * np.exp( - ((x-pos1)/sig1)**2 ) + \
           amp2 * np.exp( - ((x-pos2)/sig2)**2 ) 
            
   status = 0
   figure(12)
   plot(x,(model-y)/err)
   return [status, (model-y)/err]
    
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
   import mpfit
      
   p0 = (coef1[1],coef1[0],coef2[1],coef2[0])
   
   # define the variables for the function 'myfunct'
   fa = {'x':x,'y':f,'err':err}
     
   parinfo = [{  
   'limited': [0,0],   'limits' : [0.0,0.0],'value':  coef1[1],   'parname': 'coef11'  },{  
   'limited': [0,0],   'limits' : [0.0,0.0],'value':  coef1[0],   'parname': 'coef10'  },{  
   'limited': [0,0],   'limits' : [0.0,0.0],'value':  coef2[1],   'parname': 'coef21'  },{  
   'limited': [0,0],   'limits' : [0.0,0.0],'value':  coef2[0],   'parname': 'coef20'  },]  
      
   if chatter > 4: 
      print("parinfo has been set to: ") 
      for par in parinfo: print(par)

   Z = mpfit(_fit3,p0,functkw=fa,parinfo=parinfo,quiet=True)
   
   if (Z.status <= 0): 
      print('uvotspec.dofit2poly.mpfit error message = ', Z.errmsg)
      print("parinfo has been set to: ") 
      for par in parinfo: print(par)
   elif (chatter > 3):   
      print("\nparameters and errors : ")
      for i in range(4): print("%10.3e +/- %10.3e\n"%(Z.params[i],Z.perror[i]))
   
   return Z     
              
def _fit3(p, fjac=None, x=None, y=None, err=None):
   import numpy as np
   (coef11,coef10,coef21,coef20) = p         
   model = np.polyval([coef10,coef11],x)+np.polyval([coef20,coef21],x)
   status = 0
   return [status, (y-model)/err]

########################## summung spectra #########################################
# a routine to sum spectra. Orginal was in module uvotgetspec. This version outputs 
# a fits file when the file name ends with ".fit". 
#
####################################################################################


def sum_PHAspectra(phafiles, outfile=None, 
      ignore_flags=False, use_flags=['bad'], 
      interactive=False,figno=14,ylim=[-0.2e-14,5e-13],
      flag_bad_areas=False, exclude_wave=[], 
      adjust_wavelengths=False, wave_shifts=[], 
      objectname='unknown', object_position=None,
      wave_adjust_method=1, exclude_method=1,
      scale=None,
      chatter=1, clobber=True,
      ):
   '''
   Read a list of phafiles. Sum the spectra after applying optional wave_shifts.
   The sum is weighted by the errors.  
   
   Parameters
   ----------
   phafiles : list
      list of filenames
   outfile : str 
      name for output file. If "None" then write to 'sumpha.txt', 
      if ending in '.fit' or '.fits' a fits file will be written.
   flag_bad_areas : bool [optional]
      interactively select areas of each spectrum not to include in each spectrum
   adjust_wavelengths : bool [optional]
      interactively select a wavelength shift for each spectrum to apply before 
      summing the spectra      
   ylim : list [optional]
      force limits of Y-axis figure (interactive)      
   figno : int, or list [optional]
      numbers for figures or (if only one) the start number of figures (interactive)    
      
   wave_shifts : list [optional]
      list of shifts to add to the wavelength scale; same length as phafiles
   exclude_wave : list [optional]
      list of lists of exclude regions; same length as pha files; one list per file
      for an indivisual file the the list element is like [[1600,1900],[2700,2750],] 
   ignore_flags : bool [optional]
      if True, do not automatically convert flagged sections of the 
      spectrum to exclude_wave regions, if False, use the quality flags from "use_flags" 
   use_flags : list [optional]
      list of flags (except - 'good') to exclude. 
      Valid keyword values for the flags are defined in quality_flags(),
   objectname : str [optional]
      name of the object. This will be entered as a keyword in the summed_spectrum fits file.
   object_position: astropy.coordinates [optional]    
   scale: list, default np.ones()
      if given, the flux and fluxerr read from file n will be multiplied by scale[n]
      this is useful for summing spectra of a time-varying source      
   
   Returns
   -------
   debug information when `outfile=None`.
   
   example
   -------
   Non-variable source. For variable sources add scale factor. 
   
   phafiles = ['sw00031935002ugu_1ord_1_f.pha',
               'sw00031935002ugu_1ord_2_f.pha',
               'sw00031935002ugu_1ord_3_f.pha',
               'sw00031935002ugu_1ord_4_f.pha']
   
   sum_PHAspectra(phafiles,ignore_flags=True,flag_bad_areas=True,adjust_wavelengths=True)
 
   This will interactively (1) select bad regions and (2) 
   ask for shifts to the wavelengths of one spectra compared 
   to one chosen as reference. 
   
   or 
   
   uvotspec.sum_PHAspectra(phafiles,outfile='sum_2010-01-23T19:07_to_-24T06:30UT.fits',
     ...: wave_shifts=[0,0,0,0,0],exclude_wave=[[[1600,1730],[3350,3450]],[[1600,1730],
     ...: [3370,3420]],[[1600,1730],[3370,3415]],[[1600,1712],[3380,3430]],[[1600,1730],
     ...: [3370,3430]]],objectname='Nova KT Eri 2009',chatter=5,ignore_flags=True)
     
   Here the wavelength shifts are set to zero, and the exclusion regions have been given, 
   while no exclusion regions are taken from the quality in the file.  
      
   
   Notes
   -----
   Two figures are shown, one with flux for all spectra after shifts, one with 
   broad sum of counts in a region which includes the spectrum, unscaled, not even
   by exposure. 
   
   ** not yet implemented:  selection on flags using use-flags 
   ** smooth each spectrum, correlate for shift [all quality='good' data points]
   
   BUGS: the quality is not read from the file into exclusion sets at the moment 
   
   '''
   import os, sys
   import datetime
   try:
      from astropy.io import fits
   except:   
      import pyfits as fits
   import numpy as np
   import copy
   
# check parameters
   if outfile == None: 
      outfile = 'sumpha.txt'
      returnout = True
   else: returnout = False   
   
   nfiles = len(phafiles)
   now = datetime.date.today().isoformat()
      
   # check  phafiles are all valid paths
   for fx in phafiles:
      if not os.access(fx.strip(),os.F_OK):
         raise IOError("input file : %s not found \n"%(fx))

   if interactive & ((not flag_bad_areas) & (not adjust_wavelengths)):
      sys.stderr.write(
      "\n\nWARNING: If interactive, set at least one of flag_bad_areas, adjust_wavelengths.\n\n")
   
   if flag_bad_areas | adjust_wavelengths :
      if type(figno) != list: figno = [figno]
      if not interactive:
          sys.stderr.write(
   "WARNING: forcing interactive since flag_bad_areas or adjust_wavelengths has been set.") 
      interactive = True
   else:
      if interactive:
          sys.stderr.write("    interactive off\n\n")
      interactive = False
      
   if chatter > 1:
      sys.stdout.write("Parameters set : ")
      if interactive: sys.stdout.write("  interactive set,") 
      else: sys.stdout.write("  interactive not set,")    
      if flag_bad_areas: sys.stdout.write("  flag_bad_areas set,") 
      else: sys.stdout.write("  flag_bad_areas not set,")    
      if adjust_wavelengths: sys.stdout.write("  adjust_wavelengths set.\n") 
      else: sys.stdout.write("  adjust_wavelengths not set.\n")    
         
   # check wave_shifts and exclude_wave are lists
   if (type(wave_shifts) != list) | (type(exclude_wave) != list):
      raise IOError("parameters wave_list and exclude_wave must be a list")      
      
   exclude_wave_copy = copy.deepcopy(exclude_wave) 

   f = []    #  list of open fits file handles
   for fx in phafiles:
       # check fx is a uvot format file
       # tbd... how
       f.append( fits.open(fx) )       
   
   if len(wave_shifts)  != nfiles: 
      wave_shifts = []
      for i in range(nfiles):
         wave_shifts.append(0.0)
   elif chatter > 2: sys.stdout.write("  sum_PHAspectra: input wave_shifts = %s\n"%(wave_shifts))      
          
   if len(exclude_wave) != nfiles:  
      if chatter > 2: sys.stdout.write("exclude_wave length does not match number of files")
      exclude_wave = []
      for i in range(nfiles):
         exclude_wave.append([]) 
   elif chatter > 2: sys.stdout.write("  sum_PHAspectra: input excludes are %s\n"%(exclude_wave))      
         
   # update exclude_wave with values extracted from spectrum file using 'use_flags' values 

   if not interactive: 
       # valid ranges in fits file used to determine exclude sections 
       exclude_wave = _sum_exclude_sub1(f, nfiles, wave_shifts, 
                       exclude_wave_copy, exclude_wave, ignore_flags, 
                       use_flags, chatter=chatter) 
       if chatter > 2: 
          sys.stdout.write( "revised exclude_wave _sum_exclude_sub1: %s\n"%(exclude_wave))        

       # if auto_shift_wave:
       #   (autocorrelate using middle part spectrum[not excluded] 
                                              
   if interactive:
                                                
         # interactively adjust wavelength shifts and clipping ranges
      
         # first flag the bad ranges for each spectrum
         if chatter > 1: sys.stderr.write("Determine valid ranges for each spectrum")
         
         if flag_bad_areas:      
             if exclude_method == 1: 
                 exclude_wave = _sum_exclude_sub2(phafiles,nfiles, exclude_wave, 
                   ignore_flags, use_flags, flag_bad_areas, figno, ylim, chatter)
             elif exclude_method == 2:    
                 for i in range(nfiles):  
                      Z = flag_bad_manually(file=None,openfile=f[i],openplot=None, ylim=ylim, )
                      f[i] = Z[2]   # updated file handle
                 exclude_wave = _sum_exclude_sub1(f, nfiles, wave_shifts, 
                            exclude_wave_copy,exclude_wave, ignore_flags, 
                            ['bad'], chatter=0) 
             else: 
                 raise IOError("sum_PHAspectra: parameter exclude_method must be 1, or 2")                                              
             if chatter > 2: 
                     sys.stdout.write( "revised exclude_wave _sum_exclude_sub2: %s\n"%(exclude_wave))        
         
         if adjust_wavelengths: 
             if wave_adjust_method == 1:                          
                 wave_shifts = _sum_waveshifts_sub3(phafiles,nfiles, adjust_wavelengths, exclude_wave, 
                                            figno, ylim, chatter)
             elif wave_adjust_method == 2:
                 for i in range(nfiles):
                     Z = adjust_wavelength_manually(file=None,openfile=f[i],openplot=None,
                         ylim=ylim,ions=['HI','HeI','HeII'],reference_spectrum=None,
                         recalculate=True)
                     f[i] = Z[3]  # updated file handle                  
                     close(Z[0])  # close the figure 
                 wave_shifts = list(np.zeros(nfiles))   
             else: 
                 raise IOError("sum_PHAspectra: parameter wave_adjust_method must be either 1 or 2.")         
                  
             if chatter > 2: 
                 sys.stdout.write( "revised wave_shifts _sum_waveshifts_sub3: %s\n"%(wave_shifts))        
             
             Z = ''
                        
   if chatter > 0:
      sys.stderr.write("\n INPUT =============================================================================\n")
      sys.stderr.write("sum_PHAspectra(\nphafiles;%s,\nwave_shifts=%s,\nexclude_wave=%s,\nignore_flags=%s\n" %(
           phafiles,wave_shifts,exclude_wave,ignore_flags))
      sys.stderr.write("interactive=%s, outfile=%s, \nfigno=%s, chatter=%i, clobber=%s)\n" % (
           interactive,outfile,figno,chatter,clobber) )
      sys.stderr.write("====================================================================================\n")
                                                   
   (wave, wf, wvar, mf, svar, serr, nsummed, q, sector), D, result = _sum_weightedsum(
                                     f,copy.deepcopy(exclude_wave), wave_shifts, scalefactor=scale, chatter=chatter)

      # write output
   _sum_output_sub4(phafiles,nfiles, outfile,wave_shifts, exclude_wave,
                       wave,wf,serr,nsummed,sector,q,f,objectname,
                       object_position, wvar,mf,svar,clobber, chatter)        

   if returnout:
       return D

# DONE   
           
def _sum_exclude_sub1(f, nfiles, wave_shifts, exclude_wave_copy, exclude_wave, 
    ignore_flags, use_flags, chatter=0):
    """ Helper routine for sum_PHAspectra()
         
    Create wavelength exclusion ranges using the valid ranges for each spectrum 
    found in the fits file using quality flags
    """
    from astropy.io import fits
    import sys
         
    if chatter > 1: 
        sys.stderr.write("using the valid ranges for each spectrum found"+
        " in the fits file; and determine the shifts\n")
    exwave = []  # final
    for i in range(nfiles):
        if len(wave_shifts)  != nfiles: 
            wave_shifts.append(0)
        excl = exclude_wave[i]  # per file 
        if not ignore_flags:
            fx = f[i]
            extnames = [""]
            for k in range(1,len(fx)): extnames.append(fx[k].header['extname'])
            extnames.append("") # so it has one more element
            if extnames[2] == "CALSPEC":
               W  = fx[2].data['lambda']
               FL = fx[2].data['quality']
            elif extnames[1] == "SUMMED_SPECTRUM":   
               # fix the missing sections (assume that the wavelengths are all whole numbers)
               Wtmp  = fx[1].data['wave']
               W = np.arange(Wtmp[0],Wtmp[-1])
               FL = np.ones(len(W)) # fx[1].data['quality'] is bad
               for x in Wtmp: 
                   FL[x == W] = 0  # quality is good
            else: 
               raise IOError("input file %s has no valid extension name"%(fx))   
            ex = []
            if len(use_flags) == 0: # exclude ALL flags 
                if chatter > 1: 
                    sys.stderr.write("creating/updating exclude_wave\n") 
                if FL[0] != 0: 
                    ex=[0]
                for i in range(1,len(W)):
                    same = ((W[i] == 0) & (W[i-1] == 0)) | ( 
                            (W[i] != 0) & (W[i-1] !=0) )
                    good = (FL[i] == 0)
                    if not same:
                        if good: ex.append[i]
                        else:    ex = [i]
                    if len(ex) == 2: 
                        excl.append(ex)
                        ex = []
                    if (i == (len(W)-1)) & (len(ex) == 1): 
                        ex.append(len(W))
                    if len(ex) > 0: excl.append(ex) 
            else:
                quality_range = quality_flags_to_ranges(FL)
                for flg in use_flags:
                    if flg in quality_range:
                        pixranges = quality_range[flg]  
                        for pixes in pixranges:
                            waverange=W[pixes]
                            ex.append(list(waverange))
                    excl.append(ex) 
        if len(excl) == 0: excl.append([])                                                     
        exwave.append(excl)                                     
    if chatter > 1: 
        sys.stderr.write("_sum_exclude_sub1 returns: %s\n"%exwave)
    return exwave # updated exclude_wave
         
def _sum_exclude_sub2(phafiles,nfiles, exclude_wave, 
    ignore_flags, use_flags,flag_bad_areas, figno,ylim, chatter):
    """ Helper routine for sum_PHAspectra() 
    
        Interactively determine the exclusion ranges of each spectrum
    """

    import sys
    from astropy.io import fits
    import pylab as plt
         
    for i in range(nfiles):
        if chatter > 1: 
            sys.stderr.write(50*"v"+"\n")
            sys.stderr.write(
              " valid ranges for file number %i - file name = %s\n"
              % (i,phafiles[i]))
                         
        f = fits.open(phafiles[i])
        extnames = [""]
        for k in range(1,len(f)): extnames.append(f[k].header['extname'])
        extnames.append("")
        if extnames[2] == "CALSPEC":
            W = f[2].data['lambda']
            F = f[2].data['flux']
            E = f[2].data['fluxerr']
            FL = quality = f[2].data['quality']
        elif extnames[1] == "SUMMED_SPECTRUM":
            # fix the missing sections (assume that the wavelengths are all whole numbers)
            Wtmp  = f[1].data['wave']
            Ftmp = f[1].data['flux']
            Etmp = f[1].data['fluxerr']
            W = np.arange(Wtmp[0],Wtmp[-1])
            FL = np.ones(len(W)) # fx[1].data['quality'] is bad
            F = np.zeros(len(W))
            E = np.zeros(len(W))
            for x,y,z in zip(Wtmp,Ftmp,Etmp): 
                F[x == W] = y
                E[x == W] = z
                FL[x == W] = 0  # quality is good
        else: 
            raise IOError("input file %s has no valid extension name"%(fx))   
        try:
            COI = f[2].data['sp1_coif']
            do_COI = True
        except: 
            COI = np.ones(len(W)) 
            do_COI = False  
        q = np.isfinite(F)

        if figno != None:
            if type(figno) != list: 
                figno = [figno]
            fig=plt.figure(figno[0])
        else: 
            fig=plt.figure()
        fig.clf()
        OK = True
            
        excl_ = exclude_wave[i]
        if len(excl_) != 0: 
            sys.stdout.write( 
               "exclusions passed by argument for file %s are: %s\n"%
               (phafiles[i],excl_) )
        if (not ignore_flags) & (len(use_flags) > 1) : 
            quality_range = quality_flags_to_ranges(quality)
            for flg in use_flags:
                if flg in quality_range:
                    pixranges=quality_range[flg]
                    for pixes in pixranges:
                        waverange=W[pixes]
                        excl_.append(list(waverange))
            sys.stderr.write(
               "exclusions - including those from selected quality flags - "+
               "for file %s are: %s\n"% (phafiles[i],excl_))      

        if len(excl_) > 0:
            sys.stdout.write( 
               "wavelength exclusions for this file are: %s\n"%(excl_))
            ans = input(" change this ? (y/N) : ")
            if ans.upper()[0] == 'Y' :  OK = True
            else:                       OK = False
        else: 
            OK = True   
         
        if flag_bad_areas:
            if chatter > 1: 
                 sys.stderr.write("update wavelength exclusions\n")            
            nix1 = 0
            while OK:     # update the wavelength exclusions
              try:
                nix1 += 1
                OK = nix1 < 10
                excl = []  # note different from excl_
                    #   consider adding an image panel (resample image on wavelength scale)
                    #
                fig.clf()
                ax1 = fig.add_subplot(2,1,1)
                ax1.fill_between(W[q],F[q]-E[q],F[q]+E[q],color='y',alpha=0.4,)
                ax1.plot(W[q],F[q],label='current spectrum + error' ) 
                ax1.set_title(phafiles[i]+' FLAGGING BAD PARTS ')
                ax1.legend(loc=0)
                ax1.set_ylim(ylim)
                ax1.set_xlabel('wavelength in $\AA$')
                  
                ax2 = fig.add_subplot(2,1,2)
                ax2.plot(W[q],FL[q],ls='steps',label='QUALITY FLAG')
                if do_COI: 
                    ax2.plot(W[q],COI[q],ls='steps',label='COI-FACTOR')
                ax2.legend(loc=0)
                ax2.set_xlabel('wavelength in $\AA$')
                                  
                EXCL = True
                nix0 = 0
                while EXCL:
                    nix0 +=1 
                    if nix0 > 15: break
                    sys.stdout.write( "exclusion wavelengths are : %s\n"%excl_)
                    ans = input('Exclude a wavelength region ?')
                    if len(ans) > 0:
                        EXCL = not (ans.upper()[0] == 'N')
                        if ans.upper()[0] == 'N': break              
                        ans = eval(input(
    'Give the exclusion wavelength range as two numbers separated by a comma: '))
                        lans = list(ans)
                        if len(lans) != 2: 
                            sys.stderr.write( 
    "input either the range like: 20,30  or: [20,30] ;was %s \n"%(lans))
                            continue
                        excl_.append(lans)      
                OK = False
              except:
                sys.stderr.write(
    "Problem encountered with the selection of exclusion regions\nTry again\n")
            exclude_wave[i] = excl_
    if chatter > 1: 
        sys.stderr.write("_sum_exclude_sub2 returns: %s\n"%exclude_wave)
    return exclude_wave     

def _sum_waveshifts_sub3(phafiles, nfiles, adjust_wavelengths, exclude_wave, 
    figno, ylim, chatter):
    """Helper routine for sum_PHAspectra() 
    
    Interactively determine wave shifts
    """
    import sys
    from astropy.io import fits
    import pylab as plt

    wave_shifts = []

    # get wavelength shifts for each spectrum
    # if already passed as argument:  ?
            
    sys.stdout.write(" number  filename \n")
    for i in range(nfiles):
        sys.stdout.write(" %2i --- %s\n" % (i,phafiles[i]))
    try:   
        fselect = input(
          " give the number of the file to use as reference, or 0 : ")
        if (fselect < 0) | (fselect >= nfiles):
            sys.stderr.write("Error in file number, assuming 0\n")
            fselect=0     
        ref = fits.open(phafiles[fselect])   
    except: 
        fselect = 0
        ref = fits.open(phafiles[0])
    extnames = [""]
    for k in range(1,len(ref)): extnames.append(ref[k].header['extname'])
    extnames.append("")
    if extnames[2] == 'CALSPEC':
        refW = ref['CALSPEC'].data['lambda']
        refF = ref['CALSPEC'].data['flux']
        refE = ref['CALSPEC'].data['fluxerr']
        refexcl = exclude_wave[fselect]  
    elif extnames[1] == "SUMMED_SPECTRUM":
            # fix the missing sections (assume that the wavelengths are all whole numbers)
            Wtmp  = ref[1].data['wave']
            Ftmp  = ref[1].data['flux']
            Etmp  = ref[1].data['fluxerr']
            refexcl = exclude_wave[fselect]  
            refW = np.arange(Wtmp[0],Wtmp[-1])
            refF = np.zeros(len(refW))
            refE = np.zeros(len(refW))
            for x,y,z in zip(Wtmp,Ftmp,Etmp): 
                refF[x == refW] = y
                refE[x == refW] = z
    else:    
        raise IOError("input file %s has no valid extension name"%(fx))   

    if 'wheelpos' in ref[1].header:
        wheelpos = ref['SPECTRUM'].header['wheelpos']
    else: wheelpos = 160    
    if wheelpos < 500:
            q = np.isfinite(refF) & (refW > 1700.) & (refW < 5800)
    else:   
            q = np.isfinite(refF) & (refW > 2850.) & (refW < 6600)
    if len(refexcl) > 0:   
            if chatter > 0: print("refexcl:",refexcl)
            for ex in refexcl:
               q[ (refW > ex[0]) & (refW < ex[1]) ] = False

    if adjust_wavelengths:
            if figno != None: 
               if len(figno) > 1: fig1=plt.figure(figno[1]) 
               else: fig1 = plt.figure(figno[0])
            else: fig1 = plt.plot.figure()      
            for i in range(nfiles):
               if i == fselect:
                   wave_shifts.append( 0 )
               else:
                  f = fits.open(phafiles[i])
                  extnames = [""]
                  for k in range(1,len(f)): extnames.append(f[k].header['extname'])
                  extnames.append("")
                  if extnames[2] == 'CALSPEC':
                      W = f[2].data['lambda']
                      F = f[2].data['flux']
                      E = f[2].data['fluxerr']
                  elif extnames[1] == "SUMMED_SPECTRUM":
                      # fix the missing sections (assume that the wavelengths are all whole numbers)
                      Wtmp = f[1].data['wave']
                      Ftmp = f[1].data['flux']
                      Etmp = f[1].data['fluxerr']
                      refW = np.arange(Wtmp[0],Wtmp[-1])
                      refF = np.zeros(len(refW))
                      refE = np.zeros(len(refW))
                      for x,y,z in zip(Wtmp,Ftmp,Etmp): 
                          refF[x == refW] = y
                          refE[x == refW] = z
                  else:    
                    raise IOError("input file %s has no valid extension name"%(fx))   
                      
                  excl = exclude_wave[i]
                  print("lengths W,F:",len(W),len(F))
                  if wheelpos < 500:
                      p = np.isfinite(F) & (W > 1700.) & (W < 5400)
                  else:   
                      p = np.isfinite(F) & (W > 2850.) & (W < 6600)
                  if len(excl) > 0:  
                     if chatter > 1: print("excl:",excl)
                     for ex in excl:
                        if len(ex) == 2:
                           p[ (W > ex[0]) & (W < ex[1]) ] = False
                  if chatter > 0:
                      print("length p ",len(p))
                      sys.stderr.write("logical array p has  %s  good values\n"%( p.sum() ))
                  OK = True
                  sh = 0
                  while OK: 
                     fig1.clf()
                     ax = fig1.add_subplot(111)
                     ax.plot(refW[q],refF[q],'k',lw=1.5,ls='steps',label='wavelength reference')          
                     ax.fill_between(refW[q],(refF-refE)[q],(refF+refE)[q],color='k',alpha=0.1) 
                  
                     ax.plot(W[p]+sh,F[p],'b',ls='steps',label='spectrum to shift')       
                     ax.fill_between(W[p]+sh,(F-E)[p],(F+E)[p],color='b',alpha=0.1)
                  
                     ax.plot(W[p],F[p],'r--',alpha=0.6,lw=1.5,label='original unshifted spectrum')                        
                  
                     ax.set_title('file %i applied shift of %e' % (i,sh))
                     ax.set_xlabel('wavelength $\AA$')
                     if len(ylim) == 2: ax.set_ylim(ylim)
                     ax.legend(loc=0)
                     try:
                        sh1 = eval(input("give number of Angstrom shift to apply (e.g., 2.5, 0=done) : "))  
                        if np.abs(sh1) < 1e-3:
                           wave_shifts.append(sh)
                           OK = False
                     except: 
                        print("input problem. No shift applied")
                        sh1 = 0   
                  
                     if chatter > 0: sys.stderr.write("current wave_shifts = %s \n"%(wave_shifts))
                     if not OK: print('should have gone to next file')      
                     sh += sh1
                     if chatter > 1: print("total shift = ",sh," A") 
            #
            #  TBD use mean of shifts instead of reference spectrum ?
            #      drag spectrum ? 
            #      do autoshift on final two pixels?  
            if chatter > 1: 
                sys.stderr.write( "selected shifts = %s\n"%(wave_shifts))
                #sys.stderr.write("computing weighted average of spectrum\n")
    else:  #adjust_wavelengths = False
             for i in range(nfiles):
                 wave_shifts.append(0)  
    return wave_shifts             

def _sum_output_sub4(phafiles,nfiles, outfile,wave_shifts, exclude_wave,
   wave,wf,serr,nsummed,sector,q,f,objectname, objposition,wvar,mf,svar, clobber, chatter):
                    
   import os
   import sys
   from astropy.io import fits
   from uvotpy.uvotmisc import swtime2JD, get_keyword_from_history
   import datetime
   
   now = datetime.date.today().isoformat()

   print("outsub exclude_wave: ",exclude_wave)


   if os.access(outfile,os.F_OK) & (not clobber):
          sys.stderr.write("output file %s already present\nGive new filename (same will overwrite)"%(outfile)) 
          outfile = input("new filename = ")
          if type(outfile) != string: 
             outfile = "invalid_filename.txt"
             sys.stderr.write("invalid filename, writing emergency file %s"%(outfile))
      
   if outfile.rsplit('.')[-1][:3].lower() == 'fit':
         qq = np.isfinite(wf[q])
         print("writing fits file")
         hdu = fits.PrimaryHDU()
         hdulist=fits.HDUList([hdu])
         hdulist[0].header['TELESCOP']=('SWIFT   ','Telescope (mission) name' )                     
         hdulist[0].header['INSTRUME']=('UVOTA   ','Instrument Name' )                       
         col1 = fits.Column(name='wave',format='E',array=wave[q][qq],unit='0.1nm')            
         col2 = fits.Column(name='weighted_flux',format='E',array=wf[q][qq],unit='erg cm-2 s-1 Angstrom-1')            
         #col3 = fits.Column(name='fluxvar',format='E',array=wvar[q],unit='erg cm-2 s-1 Angstrom-1')            
         col4 = fits.Column(name='flux',format='E',array=mf[q][qq],unit='erg cm-2 s-1 Angstrom-1')            
         col5 = fits.Column(name='sqrtvariance',format='E',array=svar[q][qq],unit='erg cm-2 s-1 Angstrom-1')            
         col6 = fits.Column(name='fluxerr',format='E',array=serr[q][qq],unit='erg cm-2 s-1 Angstrom-1')            
         col7 = fits.Column(name='n_spec',format='I',array=nsummed[q][qq],unit='erg cm-2 s-1 Angstrom-1')            
         col8 = fits.Column(name='sector',format='I',array=sector[q][qq],unit='0.1nm')   
         cols = fits.ColDefs([col1,col2,col4,col5,col6,col7,col8])
         hdu1 = fits.BinTableHDU.from_columns(cols)   
         hdu1.header['EXTNAME']=('SUMMED_SPECTRUM','Name of this binary table extension')
         hdu1.header['TELESCOP']=('Swift','Telescope (mission) name')
         hdu1.header['INSTRUME']=('UVOTA','Instrument name')
         hdu1.header['FILTER'] =(f[0][1].header['FILTER'],'filter identification')
         hdu1.header['ORIGIN'] ='UCL/MSSL','source of FITS file'
         hdu1.header['CREATOR']=('uvotspec.py','uvotpy python library')
         hdu1.header['COMMENT']='uvotpy sources at www.github.com/PaulKuin/uvotpy'
         hdu1.header['OBJECT'] = (objectname,'object name')
         hi = f[0][1].header['history']
         tstart  = f[0][1].header['tstart']
         tstop  = f[-1][1].header['tstop']
         exposure = 0
         for fk in f:
            exposure += fk[1].header['exposure']
         RA = -999.9
         DEC = -999.9
         if chatter > 4: 
            print (50*"=")
            print(hi)
            print (50*"=")
            #return hi
         if len(hi) > 0:
            #if (hi[0].split()[0] != 'merged'):
            for hiline in hi:
                if len(hiline) > 0:
                    if (hiline.split()[0] == 'merged'):
                        break
                    elif (len(hiline.split()) > 0):
                        if hiline.split()[1] == 'RA,DEC':
                            RA = hiline.split()[3]
                            DEC = hiline.split()[4]
                            break
         hdu1.header['RA_OBJ'] = (RA ,'Right Ascension')
         hdu1.header['DEC_OBJ'] = (DEC,'Declination')
         hdu1.header['EQUINOX'] = 2000.0
         hdu1.header['RADECSYS'] = 'FK5'
         hdu1.header['HDUCLASS'] =  ('UVOT','document class defined by UVOT/MSSL')
         hdu1.header['HDUDOC'] = 'http://www.ucl.ac.uk/mssl/astro/space_missions/swift'
         hdu1.header['HDUVERS'] = ('1.0','initial format')
         hdu1.header['HDUCLAS1'] = ('SPECTRUM','spectrum is sum from spectra from exposures')
         hdu1.header['HDUCLAS2'] = ('FLUX','spectrum in units or energy per sec per area')
         hdu1.header['tstart'] = (tstart,"Swift time in seconds")
         hdu1.header['tstop']  = (tstop,"Swift time in seconds")
         xstart = swtime2JD(float(tstart))
         xstop  = swtime2JD(tstop)
         hdu1.header['date-obs']=(xstart[3],'start of summed images')
         hdu1.header['date-end']=(xstop[3],'end of summed images')
         hdu1.header['exposure']=(exposure,'total exposure time, corrected for deadtime')
         hdu1.header['comment'] = 'flux weighted by errors ' 
         hdu1.header['comment'] = 'sectors define unbroken stretches of summed spectrum'
         hdu1.header['comment'] = 'n_spec is the number of spectra used for the data point'
         hdu1.header['date'] = (now,'creation date of this file')
         #   "#columns: 
         #   wave     wave(A),
         #   wf       error weighted flux(erg cm-2 s-1 A-1),   
         # !  wvar     variance of the weighted flux, 
         # !  mf       plain mean of the flux (no weighting at all) (erg cm-2 s-1 A-1), 
         # !  svar     plain flux error (deviations from mean),  
         #   serr     weighted flux error (mean noise) ~ sqrt(wvar), 
         #   nsummed  number of data summed, 
         #   sector   sectors of unbroken spectrum (e.g., from zeroth order at same place in all inputs)"
         hdu1.header['history']= "merged fluxes from the following files"
         hdu1.header['history']= "#  file   wave_shift &  exclude_wave(s)"
         for i in range(nfiles):              
             if chatter > 4: 
                 print ("%2i %s %5.1f "%
                 (i,phafiles[i].rsplit('/')[-1],wave_shifts[i]) )
                 print ("%2i excl=%s"%(i,exclude_wave[i]) )
             hdu1.header['history'] = ("%2i %s %5.1f "%
                (i,phafiles[i].rsplit('/')[-1],wave_shifts[i]) )
             hdu1.header['history'] = ("%2i excl=%s"%(i,exclude_wave[i]) )
         hdulist.append(hdu1)           
         hdulist.writeto(outfile,overwrite=True)
   else:            
          print("writing output to ascii file: ",outfile)
          fout = open(outfile,'w')
          fout.write("#merged fluxes from the following files\n")
          for i in range(nfiles):             
              fout.write("#%2i,  %s, wave-shift:%5.1f, exclude_wave=%s\n" % 
                  (i,phafiles[i],wave_shifts[i],exclude_wave[i]))
          fout.write("#columns: wave(A),  weighted flux(erg cm-2 s-1 A-1), variance weighted flux, \n"\
            +"#          flux(erg cm-2 s-1 A-1), flux error (deviations from mean),  \n"\
            +"#          flux error (mean noise), number of data summed, sector\n")
          if chatter > 4: 
            print("len arrays : %i\nlen (q) : %i"%(len(wave),len(q[0])))   
          for i in range(len(q[0])):
            if np.isfinite(wf[q][i]): 
               fout.write( ("%8.2f %12.5e %12.5e %12.5e %12.5e %12.5e %4i %3i\n") % \
                    (wave[q][i],wf[q][i],wvar[q][i],
                     mf[q][i],svar[q][i],serr[q][i],
                     nsummed[q][i],sector[q][i]))
          fout.close()                 
      
def _sum_weightedsum(f,exclude_wave, wave_shifts, scalefactor=None, chatter=0):

    import numpy as np
    import copy
    from scipy import interpolate
    import sys

    # create the summed spectrum
    result = None
    # find wavelength range 
    wmin = 7000; wmax = 1500
    tstart = 9999999999.0
    tstop  = 0.0
    exposure=0
    nfiles=len(f)
    if type(scalefactor) == type(None):
        scalefactor=np.ones(nfiles) 
    for fx in f:    
        extnames = [""]
        for k in range(1,len(fx)): extnames.append(fx[k].header['extname'])
        extnames.append("")
        if extnames[2] == 'CALSPEC':
           if chatter > 2:
               sys.stderr.write("reading extension CALSPEC \n")
           q = np.isfinite(fx[2].data['flux'])
           wmin = np.min([wmin, np.min(fx[2].data['lambda'][q]) ])
           wmax = np.max([wmax, np.max(fx[2].data['lambda'][q]) ])
           tstart = np.min([tstart,fx[2].header['tstart']])
           tstop  = np.max([tstop ,fx[2].header['tstop' ]])
           exposure += fx[2].header['exposure']
        elif extnames[1] == 'SUMMED_SPECTRUM':
           if chatter > 2:
               sys.stderr.write("reading extension SUMMED_SPECTRUM \n")
           q = np.isfinite(fx[1].data['flux'])
           wmin = np.min([wmin, np.min(fx[1].data['wave'][q]) ])
           wmax = np.max([wmax, np.max(fx[1].data['wave'][q]) ])           
           tstart = np.min([tstart,fx[1].header['tstart']])
           tstop  = np.max([tstop ,fx[1].header['tstop' ]])
           exposure += fx[1].header['exposure']
        else:
           raise IOError("_sum_weightedsum 2537: input file error %s"%(fx.filename()))   
    if chatter > 1: 
        sys.stderr.write( '_sum_weightedsum: wav min %6.1f\n'%wmin)
        sys.stderr.write( '_sum_weightedsum: wav max %6.1f\n'%wmax)
         
    # create output arrays
    wave = np.arange(int(wmin+0.5), int(wmax-0.5),1)  # wavelength in 1A steps at integer values 
    nw = len(wave)                     # number of wavelength points
    flux = np.zeros(nw,dtype=float)    # flux
    error = np.zeros(nw,dtype=float)   # mean RMS errors in quadrature
    nsummed = np.zeros(nw,dtype=int)   # number of spectra summed for the given point - if only one, 
                                       # add the typical RMS variance found for points with multiple spectra
    # local arrays                                               
    err_in  = np.zeros(nw,dtype=float) # error in flux
    err_rms = np.zeros(nw,dtype=float) # RMS error from variance
    mf = np.zeros(nw,dtype=float)      # mean flux
    wf = np.zeros(nw,dtype=float)      # weighted flux
    var = np.zeros(nw,dtype=float)     # variance 
    err = np.zeros(nw,dtype=float)     # RMS error
    wgt = np.zeros(nw,dtype=float)     # weight
    wvar= np.zeros(nw,dtype=float)     # weighted variance
    one = np.ones(nw,dtype=int)        # unit
    sector = np.zeros(nw,dtype=int)    # sector numbers for disconnected sections of the spectrum
         
    D = []
    if type(scalefactor) == type(None): 
        scalefactor = np.ones(nfiles)
    for i in range(nfiles):
        sc = scalefactor[i]
        if chatter > 3 : 
            sys.stderr.write("working on %s:\n"%(f[i][len(f[i])-1].header['filetag']))
        fx = f[i]
        excl = exclude_wave[i]
        if chatter > 3: 
            sys.stderr.write( "_sum_weightedsum excl=%s\n"%excl)
            if type(excl) != list: 
                sys.stderr.write("WARNING: type excl is not list but %s\n"% type(excl))
        extnames = [""]
        for k in range(1,len(fx)): extnames.append(fx[k].header['extname'])
        extnames.append("")
        if extnames[2] == 'CALSPEC' :
           if chatter > 3:
               sys.stderr.write("reading extension CALSPEC \n")
           W = fx['CALSPEC'].data['lambda']+wave_shifts[i]
           F = sc * fx['CALSPEC'].data['flux']   
           E = sc * np.abs(fx['CALSPEC'].data['fluxerr'])
        elif extnames[1] == 'SUMMED_SPECTRUM' :   
           if chatter > 3:
               sys.stderr.write("reading extension SUMMED_SPECTRUM \n")
           W = fx['SUMMED_SPECTRUM'].data['wave']+wave_shifts[i]
           F = sc * fx['SUMMED_SPECTRUM'].data['flux']   
           E = sc * np.abs(fx['SUMMED_SPECTRUM'].data['fluxerr'])
        else:
           raise IOError("_sum_weightedsum 2583: input file error %s"%(fx.filename()))   
        p = np.isfinite(F) & (W > 1600.)
        fF = interpolate.interp1d( W[p], F[p], )
        fE = interpolate.interp1d( W[p], E[p]+0.01*F[p], ) 
         
        M = np.ones(len(wave),dtype=bool)     # mask for valid 'wave' set to True
        M[wave < W[p][0]] = False
        M[wave > W[p][-1]] = False
        while len(excl) > 0:
            try:
                exclpop = excl.pop()
                w1,w2 = exclpop
                if chatter > 1: 
                    sys.stderr.write(
'_sum_weightedsum: Excluding from file %i - %f-%f\t popped from excl:%s remains to do:%s\n'%
(i,w1,w2,exclpop,excl))
                M[ (wave >= w1) & (wave <= w2) ] = False
            except: 
                sys.stderr.write(
"_sum_weightedsum: ERROR Excluding a range problem with\n"+
" popped original excl=%s\n"%(excl))
                pass 
         
        flux[M]    = fF(wave[M])
        error[M]   = fE(wave[M])
        nsummed[M] += one[M] 
        mf[M]      += flux[M]                 # => mean flux 
        wf[M]      += flux[M]/error[M]**2     # sum weight * flux
        wvar[M]    += flux[M]**2/error[M]**2  # sum weight * flux**2
        var[M]     += flux[M]**2               # first part     
        err[M]     += error[M]**2
        wgt[M]     += 1.0/error[M]**2    # sum weights
        D.append(((W,F,E,p,fF,fE),(M,wave,flux,error,nsummed),(mf,wf,wvar),
                  (var,err,wgt)))
         
    # make sectors based on continuous parts spectrum
    if chatter > 3 : 
        sys.stderr.write( "_sum_weightedsum: making sectors /n")
    sect = 0
    for i in range(1,len(nsummed),1):
          if (nsummed[i] != 0) & (nsummed[i-1] != 0): sector[i] = sect
          elif (nsummed[i] != 0) & (nsummed[i-1] == 0): 
             sect += 1
             sector[i]=sect
    q = np.where(nsummed > 0)        
    mf[q] = mf[q]/nsummed[q]                      # mean flux
    var[q] = np.abs(var[q]/nsummed[q] - mf[q]**2) # variance in flux (deviations from mean of measurements)
    err[q] = err[q]/nsummed[q]                    # mean variance from errors in measurements   
    wf[q] = wf[q]/wgt[q]                          # mean weighted flux
    wvar[q] = np.abs(wvar[q]/wgt[q] - wf[q]**2)   # variance weighted from measurement errors           
    svar = np.sqrt(var)
    serr = np.sqrt(err)
    result = wave[q], wf[q], wvar[q], mf[q], svar[q], serr[q], nsummed[q], \
             wave_shifts, exclude_wave, sector[q]
    
    # debug :    
    D.append( ((W,F,E,p,fF,fE),(M,wave,flux,error,nsummed,sector),(mf,wf,wvar)
              ,(var,err,wgt)) )
    for fx in f:                                  # cleanup
         fx.close()
    
    return (wave, wf, wvar, mf, svar, serr, nsummed, q, sector), D, result 
  

