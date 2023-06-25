#
# readout_streak (Mat Page) c-code translation to python code
# NPMK 2020-08-(13-19)
# this is a translation from c to python.
# npmk 2023-06025 corrected extension number output by adding 1
# 
import sys
import numpy as np
from astropy.io import fits

class ros():


#mask_struct = im_array # value of x,y is *(im_array + (y * naxis[0]) + x) 1=good 0=bad
# "ros" derives from Read-Out-Streak

  def __init__(self,infile=None, outfile=None,
      maskedout=None, fixedout=None, columnplots=False,
      snthresh=6.0,chatter=0):
      """
      ros - extract basic readout streak data 
      
      input parameters:
         infile : path 
            mod-8 corrected image file
         maskedout : path 
         fixedout : path 
         columnplots : bool
         sntresh : float
         chatter : int 
      """ 

      if chatter > 0: 
         sys.stdout.write ("\n Running readout_streak V2.1 (python version) \n")
      self.infile = infile
      self.snthresh = snthresh
      self.chatter = chatter
      self.maskedout = maskedout
      self.fixedout = fixedout
      self.columnplots = columnplots
      self.fixedfileflag = fixedout     != None # True if requested
      self.maskedfileflag = maskedout   != None # True if requested
      self.columnfileflag = columnplots
      
      self.fpimage = fits.open(infile,)
      if (self.fixedfileflag):
         self.pfixed = fits.PrimaryHDU()         # create main HDU
      if self.maskedfileflag:
         self.pmasked = fits.PrimaryHDU()
      self.im_transposed = False
      self.hdr = None
      self.BRIGHT_THRESH = 0.4  # counts per frame bright source threshold 
      if outfile == None:
         try:
            self.foutput = open("results.{infile.split('_rw')[0][-3:]}_txt","w")
         except: 
            raise IOError(f"ros.py - cannot open output {outfile}\n")   
      else:
         self.foutput = open(outfile,"w")

# now cycle through HDUs until they are all done.

  def process(self):
    self.hdunum = 0
    result = []
    for hdu in self.fpimage:
        hdr = hdu.header
        self.hdr = hdr
        if (hdr['naxis'] == 2):
           if (hdr['xtension'] != 'IMAGE'):
               sys.stderr.write("The current HDU has not an XTENSION=IMAGE card\n")
           else:      
             naxis1 = hdr['naxis1'] # imagedata->naxis 0
             naxis2 = hdr['naxis2'] # imagedata->naxis 1
             deltax = 0
             binx = hdr['binx'] # imagedata->binx
             # OM binx = hdr['binax1'] 
             # OM deltax = hdr['windowdx']
             # OM binx = deltax/naxis1
             deltay = 0
             biny = hdr['biny']
             exposure = hdr['exposure']
             try:
                exposure_uncertainty = hdr['exp_unc']
                if hdr['exp_unc'] > 0.01:
                   sys.stdout.write(
                   f"in extension {self.hdunum}\nexposure uncertainty = {exposure_uncertainty}\n")
             except:
                exposure_uncertainty = None   
                pass
             try:     
                cntexp = hdr['cntexp']
                exp_unc = (cntexp - exposure)/exposure   
                if np.abs(exp-unc) > 0.03:
                   sys.stdout.write(
                   f"in extension {self.hdunum}: uncertain exposure. CNTEXP={cntexp}\n")
             except:
                exp_unc = None  
                pass    
             frametime = hdr['framtime']
             # test for frametime in ms rather than in s, as in XMM-OM files
             # if frametime > 1.0: frametime = 0.001 * frametime
             im_array = hdu.data
             im_dims = im_array.shape
             # make sure the dimensions are as we think they are:
             #print (f"Extension={self.hdunum}: naxis1,naxis2 = {naxis1},{naxis2} \n")
             im = np.transpose(np.copy(im_array))
             im_transposed = True
             self.foutput.write(f"\n Extension {self.hdunum+1}, exposure {exposure}, frametime {frametime}\n")   
             image_data, image_mask, masked_image = self.clean_sources(im, hdr)
             column_stuff = self.collapse_columns(image_data, np.copy(image_mask),  
                         self.columnfileflag, self.snthresh,hdr,
                         chatter=self.chatter)
             # write images 
             # transpose back ? 
             #result.append((self.hdunum+1,column_stuff, (image_data, image_mask, masked_image), im_transposed ))
             result.append(column_stuff)
             self.hdunum += 1
             #print (f"hdunum={self.hdunum}\n\n")
    self.fpimage.close()
    self.foutput.close()
    if self.fixedfileflag :
         self.pfixed.writeto(fixedout)
    if self.maskedfileflag: 
         self.pmasked.writeto(maskedout)
    if self.chatter > 0: 
        return result
      

  def clean_sources(self,im,hdr):
      boxsize = 10
      bright_exclusion_size = 48  #; remove 48 pix around bright sources 
      naxis1, naxis2 = im.shape
      box_image = np.zeros( (naxis1,naxis2) )
      masked_image = np.zeros( (naxis1,naxis2) )
      image_column = np.zeros ( (naxis2) )
      image_mask = np.ones( (naxis1,naxis2) , dtype=np.int) # all good
      
      # use a sliding box to find bright sources 
      box_x = boxsize // np.int(hdr['binx'])
      box_y = boxsize // np.int(hdr['biny'])
      
      for i in range(naxis1): # for (i = 0; i < naxis1; i++){
         for j in range(naxis2): # for (j = 0; j < naxis2; j++){
         #  /* initialise to box image to zero and mask to 1 */
         #  /* get the limits for the box, without going over the edges */
           startx = i - box_x
           starty = j - box_y
           if (startx < 0): startx = 0
           if (starty < 0): starty = 0 
           stopx = startx + box_x + box_x + 1
           stopy = starty + box_y + box_y + 1
           if (stopx > naxis1):
             stopx = naxis1
             startx = stopx - box_x - box_x - 1
             if (startx < 0): startx = 0
           
           if (stopy > naxis2):
             stopy = naxis2
             starty = stopy - box_y - box_y - 1
             if (starty < 0): startx = 0  # <-- ERROR? changed to starty = 0 
           #/* now compute the value for pixel i,j */
           box_image[i,j] = im[startx:stopx,starty:stopy].sum()
           
      # find all "bright source" pixels 
      excl_radius = bright_exclusion_size // hdr['binx']
      for i in range(0,naxis1): # ( i = 0; i < naxis1; i++){
        for j in range(0,naxis2): # (j = 0; j < naxis2; j++){
          if box_image[i,j] / hdr['exposure'] * hdr['framtime'] > self.BRIGHT_THRESH :
            #/* get the limits for a box in which to mask pixels */
            startx = i - excl_radius
            starty = j - excl_radius
            if (startx < 0): startx = 0
            if (starty < 0): starty = 0 
            stopx = startx + excl_radius + excl_radius + 1
            stopy = starty + excl_radius + excl_radius + 1
            if (stopx > naxis1):
              stopx = naxis1
              startx = stopx - excl_radius - excl_radius -1
              if (startx < 0): startx = 0
            
            if (stopy > naxis2):
              stopy = naxis2
              starty = stopy - excl_radius - excl_radius -1
              if (starty < 0): startx = 0  # <-- ERROR? changed startx to starty = 0
            
            #/* now mask pixels i,j */
            for k in range(startx,stopx): 
              for l in range(starty,stopy):
                if (((k - i) * (k - i)) + ((l - j) * (l - j)) < 
                    (excl_radius * excl_radius)):
                   image_mask[k,l] = 0
      
      # now mask bright pixels by column in original and smoothed images 
      for i in range(0,naxis1): # (i = 0; i < naxis1; i++){
        #/* original image first */
        number_pixels = 0
        for j in range(0,naxis2):
          if image_mask[i,j] == 1: 
             image_column[number_pixels] = im[i,j]
             number_pixels += 1
           
        if (number_pixels > 0): 
           column_median = np.median(image_column[:number_pixels])
        else:
           column_median = 1.0
       
        # exclude any pixels over median + 3 * sqrt (median) 
        # deal with low stats case that median is < 1 
        if (column_median < 1.0): 
          pix_thresh = column_median + 3.0;
        else:
          pix_thresh = column_median + 3.0 * np.sqrt(column_median);
        
        for j in range(0,naxis2): 
           if im[i,j] > pix_thresh :
              image_mask[i,j] = 0
        
        # then smoothed image 
        number_pixels = 0
        for j in range(0,naxis2): 
          if image_mask[i,j] == 1: 
             image_column[number_pixels] = box_image[i,j]
             number_pixels += 1
        
        if (number_pixels > 0):
           column_median = np.median(image_column[:number_pixels])
        else:
           column_median = 1.0
        
        # exclude any pixels over median + 3 * sqrt (median) */
        # deal with low stats case that median is < 1 */
        if (column_median < 1.0):
           pix_thresh = column_median + 3.0
        else:
           pix_thresh = column_median + 3.0 * np.sqrt(column_median)
        
        for j in range(0,naxis2): 
           if box_image[i,j] > pix_thresh: 
              image_mask[i,j] = 0
      
      masked_image[image_mask == 1] = im[image_mask == 1]

      return im, image_mask, masked_image
      
      
  def collapse_columns(self,image_data, image_mask, 
                      columnfileflag, SNthresh,
                      hdr, chatter=0):
      """
      *summed_columns; /* this is countrate in mod-8 cell region */
      *summed_error; /* this is poisson error in mod-8 cell region */
      *column_mask; /* 0 is bad, 1 is good */
      colthresh; /* only clean 2 sigma individual columns for default SN>6 */
      """
      columnsfilename=f"columns{self.hdunum - 1}.qdp" #  qdp output filename
      colthresh = SNthresh / 3.0  #have to reduce threshold below 2 to avoid infinite loops
      naxis1, naxis2 = image_data.shape  
      mod8_cellsizex = 16 // hdr['binx'] #; /* aperture 2 phys pixels wide */
      mod8_cellsizey = 16 // hdr['biny'] #  /* times 2 phys pixel deep */
      column_total = np.zeros((naxis1),dtype=np.float)
      column_pixels = np.zeros((naxis1),dtype=np.int)
      column_average = np.zeros((naxis1),dtype=np.float)
      summed_columns = np.zeros((naxis1),dtype=np.float)
      summed_error = np.zeros((naxis1),dtype=np.float)
      summed_sigtonoise = np.zeros((naxis1),dtype=np.float)
      column_error = np.zeros((naxis1),dtype=np.float)
      smoothed_columns = np.zeros((naxis1),dtype=np.float)
      working_columns = np.zeros((naxis1),dtype=np.float)
      column_mask = np.zeros((naxis1),dtype=int)
      column_out = []
      fixed_image = np.copy(image_data)
      
      #/* find the totals and averages */
      
      for i in range(naxis1): #(i = 0; i < naxis1; i++){  # all columns one by one
         column_pixels[i] = (image_mask[i,:] == 1).sum()  
         column_total[i] = image_data[i,image_mask[i,:] == 1].sum()
         if column_pixels[i] > 0:
            column_average[i] = column_total[i]/column_pixels[i]
            column_error[i] = np.sqrt(column_total[i]) / column_pixels[i]
            column_mask[i] = 1
         else:
            column_average[i] = 0.0
            column_error[i] = 1.0
            column_mask[i] = 0   
      
         #/* now do a median smoothing */

      smooth_box = 64 // hdr['binx']

      for i in range(0,naxis1): 
        #/* get the limits for a box in which to smooth pixels */
        startx = i - smooth_box
        if (startx < 0): startx = 0 
        stopx = startx + smooth_box + smooth_box + 1
        if (stopx > naxis1):
          stopx = naxis1
          startx = stopx - smooth_box - smooth_box -1
          if (startx < 0): startx = 0
        
        num_values = 0
        working_columns[i] = 0
        for j in range(startx,stopx): 
          if (column_mask[j] == 1):
            working_columns[num_values] = column_average[j]
            num_values += 1
        
        if (num_values > 0):
          smoothed_columns[i] = np.median(working_columns[:num_values])
        else:
          smoothed_columns[i] = 0.0

      #/* if requested write out the columns plot */
      if (columnfileflag):
        fptout = open(columnsfilename,"w")
        fptout.write("read serr 2\n")
        fptout.write("line stepped on\n")
        for i in range(0,naxis1): 
          fptout.write(" %4.4d %lf %lf %lf\n" % ( 
          i, column_average[i], 
          column_error[i], smoothed_columns[i]))
        fptout.close()

      #/* now pass a 1 physical-pixel cell over the row and detect sources */
      #/* until there are no more left */

      sources_left = True
      while sources_left : #{
        #/* first pass the cell over */
        for i in range(0, naxis1 + 1 - mod8_cellsizex):  # changed to naxis1
          summed_columns[i] = 0.0
          summed_error[i] = 0.0
          num_values = 0
          for j in range(0, mod8_cellsizex):
            if (column_mask[i+j] == 1):
              summed_error[i] += column_total[i + j] # i.e. Sum(error**2+...)
              summed_columns[i] += (column_total[i + j] - smoothed_columns[i + j] 
                  * column_pixels[i + j]) 
              num_values += column_pixels[i + j]
            
          if (num_values > 0) :
            summed_error[i] = np.sqrt(summed_error[i])
            summed_error[i] /= num_values
            summed_error[i] *= mod8_cellsizey
            summed_error[i] *= mod8_cellsizex
            summed_error[i] /= hdr['exposure']
            summed_columns[i] /= num_values
            summed_columns[i] *= mod8_cellsizey
            summed_columns[i] *= mod8_cellsizex
            summed_columns[i] /= hdr['exposure']
            summed_sigtonoise[i] = summed_columns[i] / summed_error[i]
          else:
            summed_sigtonoise[i] = 0.0
            
        #/* then find the highest S/N cell position */
        highest_SN = 0.0
        highest_col = 0
        for i in range( 0, naxis1 + 1 - mod8_cellsizex): # changed to naxis1
          if (summed_sigtonoise[i] > highest_SN):
            highest_col = i
            highest_SN = summed_sigtonoise[i]
        
        #/* if the S/N is above threshold, report */
        #/* and clean streak from image and column_total */
        if (highest_SN > SNthresh):
          Hcol = highest_col + (mod8_cellsizex / 2)
          HCR = summed_columns[highest_col]
          Herr = summed_error[highest_col] 
          self.foutput.write( 
          "Ext%3.3d  Streak at column %4.4d, S/N = %5.1lf, CR = %lf +- %lf\n"%(
          (self.hdunum), highest_col + (mod8_cellsizex / 2), highest_SN, 
          summed_columns[highest_col], summed_error[highest_col]))
          
          column_out.append({"extension":self.hdunum+1,"exposure":hdr['exposure'],
              "frametime":hdr['framtime'],
             f"streak_{self.hdunum}_{Hcol}":[Hcol,highest_SN,HCR,Herr]})
          
          for i in range(highest_col, highest_col + mod8_cellsizex):
            column_difference = column_average[i] - smoothed_columns[i]
            if ((column_difference / column_error[i]) > colthresh):
               column_average[i] -= column_difference
               column_total[i] -= column_difference * column_pixels[i]
               fixed_image[i,:] -= column_difference
        else:
          #/* stop searching as no more significant sources */
          sources_left = False

      return column_out  
      #return (column_total,column_pixels,column_average,summed_columns,summed_error,
      #  summed_sigtonoise,column_error,smoothed_columns,working_columns,column_mask,
      #  fixed_image)  # for debug












