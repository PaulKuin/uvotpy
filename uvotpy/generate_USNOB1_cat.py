import numpy as np
from astroquery.vizier import Vizier
from astropy.coordinates import SkyCoord
import astropy.units as u
from astropy import table
import astropy


usnob_cat = Vizier.find_catalogs('USNO-B1.0')

def get_usnob1_cat(ra, dec, blim,radius=15.0*u.arcmin):
    """
    Assume that the ra,dec are given as float parameters in units of deg
    
    if (type(ra) == float) & (type(dec) == float):
       ra_u = ra*u.degree
       dec_u = dec*u.degree
    elif (type(ra) == np.float64) & (type(dec) == np.float64):
       ra_u = ra*u.degree
       dec_u = dec*u.degree
    elif hasattr(ra,'to'):  
       ra_u = ra.to(u.deg)
       dec_u = dec.to(u.deg)  
    else:
       raise IOError(f"ra,dec not in correct format\n") 
    """     
    ra_u = ra * u.deg
    dec_u = dec * u.deg
    coords = SkyCoord(ra_u, dec_u, frame='icrs') #Should this be ICRS or FK5
    #Only Class 0 (stars) - unable to implement this at the current time. Need to understand
    #USNO-B1 s/g classification
    v = Vizier(columns=['USNO-B1.0', '_RAJ2000', '_DEJ2000', 
                        'B1mag', 'R1mag', 'B2mag', 'R2mag', 
                        'pmRA', 'pmDE', 'Imag', 'B1s/g', '_r'],
                row_limit=500000,
               column_filters={"B2mag":">6", 'B2mag':'<{}'.format(blim)}) #B2mag fainter than 6, brighter than blim
    new_table_list = v.query_region(coords, 
                                 radius=radius, #Search 900 arcseconds
                                 catalog = 'I/284')
    if len(new_table_list) ==0:
        return None
    else:
        new_table = new_table_list[0]
    #Get the 5000 closest
    new_table.sort('_r')
    if len(new_table) > 5000:
        new_table.remove_rows(np.arange(5001, len(new_table)))
    #Sort with brightest star first
    new_table.sort(['B2mag'])
    

    #Fill in blank values with 99.99
    new_table['B1mag'].fill_value = 99.99
    new_table['R1mag'].fill_value = 99.99
    new_table['B2mag'].fill_value = 99.99
    new_table['R2mag'].fill_value = 99.99
    new_table['Imag'].fill_value = 99.99
    new_table['pmRA'].fill_value = 99.99
    new_table['pmDE'].fill_value = 99.99
    filled_table = new_table.filled()
    filled_table.write('search.ub1', overwrite=True, format='ascii.fixed_width_no_header', delimiter=' ')
    
    searchcenter_ofile = open('searchcenter.ub1', 'w')
    searchcenter_ofile.write('{},{}'.format(ra, dec))
    searchcenter_ofile.close()
    return 'success'
    
        
gaia_dr2 = u'I/345' 
def get_gaiadr2(ra, dec, blim):
    ra_u = ra*u.degree
    dec_u = dec*u.degree
    coords = SkyCoord(ra_u, dec_u, frame='icrs') #Should this be ICRS or FK5
    v = Vizier(columns=['RAdeg','DEdeg','yr',
              'Plx','pmRA','pmDE','Gmag','BPmag','RPmag','RV','e_RV',
              'RAJ2000','DEJ2000','e_Gmag','e_BPmag','e_RPmag','_r'], 
              row_limit=200000,
              column_filters={"Gmag":">6", 'Gmag':'<{}'.format(blim)})  # gaia2.sam file  
    new_table_list = v.query_region(coords, 
               radius=900*u.arcsecond, #Search 900 arcseconds
               catalog = 'I/345/gaia2.sam')
               
    if len(new_table_list) == 0:
        return None
    else:
        new_table = new_table_list[0]
    #Get the 5000 closest
    new_table.sort('_r')
    if len(new_table) > 5000:
        new_table.remove_rows(np.arange(5001, len(new_table)))
    # unfinished subroutine 
    
    