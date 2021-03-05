#! /usr/bin/env python

import time
import numpy
import ephem


def sunradec(utime):
    """Return the [RA,Dec] (in radians) of the Moon for a given unix time"""
    sun = ephem.Sun()
    utime = time.localtime(utime) # get the time tuple
    sun.compute((utime[0],utime[1],utime[2],utime[3],utime[4],utime[5]))
    return [sun.ra/1.0,sun.dec/1.0]

# Vector operations

def cross(v2,v1):
    """Calculate the cross product of 2 3D vectors"""
    vec = numpy.array([0,0,0],numpy.float)
    vec[0] = v2[1]*v1[2]-v2[2]*v1[1]
    vec[1] = v2[2]*v1[0]-v2[0]*v1[2]
    vec[2] = v2[0]*v1[1]-v2[1]*v1[0]
    return vec

def vecnorm(vec):
    """Normalise a vector."""
    mag = numpy.sqrt(sum(vec**2))
    return vec/mag

def radec2vec(ra,dec):
    """Convert RA/Dec angle (in radians) to a vector"""

    v1 = numpy.cos(dec)*numpy.cos(ra)
    v2 = numpy.cos(dec)*numpy.sin(ra)
    v3 = numpy.sin(dec)

    return numpy.array([v1,v2,v3])

def ICSdateconv(date):
    """Convert the date format used in the ICS to standard UNIX time"""
    x = date.replace("/"," ").replace("-"," ").replace(":"," ").split()
    base = time.mktime((int(x[0]),1,0,0,0,0,0,0,0))
    return base + ((int(x[1]))*86400 + (float(x[2])) * 3600 + float(x[3])*60 + float (x[4]))


def optimum_roll(ra,dec,utime):
    """Calculate the optimum Swift Roll angle (in radians) for a given Ra, Dec and Unix Time"""
    # Calcuate the vector for the Sun position
    sun = sunradec(utime)
    vSun = radec2vec(sun[0],sun[1])

    # define the target vector
    vT = radec2vec(ra,dec)

    # get cross product of Sun vector and target vector (vSun x vT) :  result is vector Y

    vY = cross(vT,vSun)
    vnY = vecnorm(vY) # Normalise the Y vector

    # get cross product of normalized Y and target vector: result is vector Z

    vZ = cross(vnY,vT)
    vnZ = vecnorm(vZ)

    # newroll = -atan(vnY[2]/vnZ[2]), to get proper newroll use function atan2
    # JAK Note: For some reason this doesn't work - so I got rid of the sign and it did.

    if (vnY[2] != 0 and vnZ[2] != 0):
        newroll = numpy.arctan2(vnY[2],vnZ[2])
        newroll = newroll/dtor
    else:
        # roll is not uniquely defined, arbitrarily pick 0.0 or 180.0
        newroll = 0.
        if (vSun[0]*(-cos(newroll)*sin(dec)*cos(ra) - sin(newroll)*sin(ra))
            + vSun[1]*(-cos(newroll)*sin(dec)*sin(ra) + sin(newroll)*cos(ra))
            + vSun[2]*cos(newroll)*cos(dec) < 0.0):
            # 0.0 would put the sun vector in -Z, so use 180.0 deg
            newroll = 180.0

    if (newroll < 0.0):
        newroll += 360.0
    return newroll*dtor


dtor = numpy.pi/180.0

lt = time.gmtime()
year = lt[0]
day = lt[7]

#ra = 266*dtor
#dec = -29*dtor

rollcons = 9.5
suncons = 45.

def table(ra,dec,year=year,startday=0,endday=265):
    """ Produce a table of approximate roll ranges for Swift observation of a target 
        with given position.
        
        Parameters:
        
        ra, dec : float
           position in decimal degrees (J2000)
        year : int (optional)
           year for the table (default = current year)     
    """
    print("Roll angle ranges for swift observation of target at\n RA,Dec=(%.2f,%.2f) year=%i\n"%
       (ra,dec,year))
    ra = ra*dtor
    dec = dec*dtor
    print("%3s %7s %7s %7s %7s %7s %7s"%("Day","SunRA","SunDec","SunDist","Roll","MinRoll","MaxRoll"))
    for i in range(startday,endday+1):
        now = ICSdateconv("%s-%03d-00:00:00"%(year,i))
        sra,sdec = sunradec(now)
        oroll = optimum_roll(ra,dec,now)
        sundist = ephem.separation([ra,dec],[sra,sdec])
        roll_range = rollcons/numpy.sin(sundist);
        if sundist/dtor > suncons:
            #print(f"{i:3d} {sra/dtor:7.3f} {sdec/dtor:7.3f} {sundist/dtor:7.3f} 
            #{oroll/dtor:7.3f} {oroll/dtor-roll_range:7.3f} {oroll/dtor+roll_range:7.3f}")
            print("%3d %7.3f %7.3f %7.3f %7.3f %7.3f %7.3f"%(i,sra/dtor,sdec/dtor,
              sundist/dtor,oroll/dtor,oroll/dtor-roll_range,oroll/dtor+roll_range))

def forday(ra,dec,year=year,day=0):
    ra = ra*dtor
    dec = dec*dtor
    now = ICSdateconv("%s-%03d-00:00:00"%(year,day))
    sra,sdec = sunradec(now)
    oroll = optimum_roll(ra,dec,now)
    sundist = ephem.separation([ra,dec],[sra,sdec])
    roll_range = rollcons/numpy.sin(sundist)
    return {'valid_sun_dist':sundist/dtor > suncons,'day':day,'sun_ra_deg':sra/dtor,'sun_dec_deg':sdec/dtor,\
           'sun_dist_deg':sundist/dtor,'sun_limit_deg':suncons,\
           'roll':oroll/dtor,'min_roll':oroll/dtor-roll_range,\
           'max_roll':oroll/dtor+roll_range}
