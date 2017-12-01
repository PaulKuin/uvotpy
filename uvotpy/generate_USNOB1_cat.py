from astroquery.vizier import Vizier
from astropy.coordinates import SkyCoord
import astropy.units as u
from astropy import table

usnob_cat = Vizier.find_catalogs('USNO-B1.0')

def get_usnob1_cat(ra, dec):
    ra_u = ra*u.degree
    dec_u = dec*u.degree
    coords = SkyCoord(ra_u, dec_u, frame='icrs')
    v = Vizier(columns=['USNO-B1.0', '_RAJ2000', '_DEJ2000', 'B1mag', 'R1mag', 'B2mag', 'R2mag', 'pmRA', 'pmDE', 'Imag', 'B1s/g', '_r'])
    new_table = v.query_region(coords, 
                                 radius=0.1*u.degree,
                                 catalog = 'I/284')[0]
    #Fill in blank values with 99.99
    new_table['B1mag'].fill_value = 99.99
    new_table['R1mag'].fill_value = 99.99
    new_table['B2mag'].fill_value = 99.99
    new_table['R2mag'].fill_value = 99.99
    new_table['Imag'].fill_value = 99.99
    new_table['pmRA'].fill_value = 99.99
    new_table['pmDE'].fill_value = 99.99
    filled_table = new_table.filled()
    new_table.write('search.ub1', overwrite=True, format='ascii.fixed_width_no_header', delimiter=' ')
    
    searchcenter_ofile = open('searchcenter.ub1', 'w')
    searchcenter_ofile.write('{},{}'.format(ra, dec))
    serachcenter_ofile.close()