'''Rational Function program
   
   Do a Least Squares Fit for a rational function 
   = ratio of polynomials of order Nn for de nominator
   , Nd for the denominator.

   The LSF is done using curve_fit from scipy.optimize.minpack
'''
import numpy as np
from scipy.optimize.minpack import curve_fit
from scipy import odr

def ratfunct(x, A, full=False,chatter=0):
   '''Calculates the rational function in one variable:
   
   Parameters
   ----------
   x : real
     is the function argument, 
   
   A : list
     is a list of the parameters; first element is list of coefficients nominator,
     second element a list of the denominator coefficients
   
   returns
   -------
   a list of the function value R, partial derivative (dR/dx), and 
   if full = True, the complex values of the poles (roots of the Denomenator)
   '''
   Cn = A[0]
   Cd = A[1]
   Nn = len(Cn)       # number of coefficients is Nn+1
   Nd = len(Cd)       # number of coefficients is Nd+1
   if chatter > 2: 
     print 'Cn ', Cn
     print 'Cd ', Cd
   Cn = np.asarray(Cn)
   Cd = np.asarray(Cd)
   Den = np.polyval(Cd,x)  # value of numerator polynomial in points x
   Nom = np.polyval(Cn,x)  # value of denominator polynomial in points x  
   R = Den/Nom
   #Poles = np.roots(Den) oversensitive 
   p = np.poly1d(Cn)
   Cn1 = np.polyder(p)
   Den_deriv = np.polyval(Cn1,x)
   p = np.poly1d(Cd)
   cd1 = np.polyder(p)
   Nom_deriv = np.polyval(Cn1,x)
   dR = (Nom_deriv - R*Den_deriv)/Den
   if full:
      return R, dR, Poles 
   else:
      return R, dR
 
def ratfunctval(x, *args):
   ''' value of rational function in x for use with curve_fit
   This function is called by curve_fit
   
   Parameters
   ----------
   x : ndarray
   
   kwargs : dict
   - *a* list [order nominator, order denominator, nominator 
         coefficients, denominator coefficients]
   
   Returns
   -------
   function value R
   '''
   a = np.array(args)
   n1 = a[0]
   n2 = a[1]
   Cn = a[2:2+n1]
   Cd = a[2+n1:2+n1+n2]
   A = Cn, Cd
   R, dr = ratfunct(x,A)
   return R

def ratfunctderiv(x,Cn, Cd):
   ''' derivative of rational function in x '''
   A = (Cn, Cd)
   R, dr = ratfunct(x,A)
   return dr

def ratfunctinit(x,y,Nn,Nd):
   ''' first guess for rational function 
   Parameters
   ----------
   x, y :  
   Nn : order of nominator
   Nd : order of denominator
   
   Returns
   -------
   starting guess for function
   '''
   A = (np.ones(Nn),np.ones(Nd)/2**np.arange(Nd)) 
   return A

def ratfit(x, y, Nom_deg, Den_deg, full=False):
    """
    Least squares fit of a rational function (ratio of two polynomials) to 1D data.

    Fit a rational function 
    ``R(x) = (p[0] * x**Nom_deg + ... + p[Nom_deg]) /
    (q[0] * x**Den_deg + ... + q[Den_deg])`` 
    with the degree `Nom_deg` for the numerator polynomial and 
    `Den_deg` for the degree of the Denominator polynomial to points `(x, y)`. 
    Returns a list with two vectors of coefficients `(p,q)` that minimises 
    the squared error.

    Parameters
    ----------
    x : array_like, shape (M,)
        x-coordinates of the M sample points ``(x[i], y[i])``.
    y : array_like, shape (M,) or (M, K)
        y-coordinates of the sample points. Several data sets of sample
        points sharing the same x-coordinates can be fitted at once by
        passing in a 2D-array that contains one dataset per column.
    Nom_deg, Den_deg : int
        Degrees of the fitting polynomials in nominator and denominator.
    full : bool, optional
        Switch determining nature of return value. When it is
        False (the default) just the coefficients are returned, when True
        diagnostic information from the singular value decomposition is also
        returned.

    Returns
    -------
    p, q : ndarray, shape (M,) or (M, K)
        Polynomial coefficients, of nominator and denominator, 
	highest power first.
        If `y` was 2-D, the coefficients for `k`-th data set are in ``p[:,k]``.

    covariance matrix : present only if `full` = True

    See Also
    --------
    ratfunct : computes rational function values and derivatives.
    scipy.optimize.curve_fit : Computes a least-squares fit.
    
    Notes
    -----
    The solution minimizes the squared error

    .. math ::
        E = \\sum_{j=0}^k |R(x_j) - y_j|^2

    """
    from scipy.optimize.minpack import curve_fit
    import numpy as np
    
    x = np.asarray(x) + 0.0
    y = np.asarray(y) + 0.0
    
    # check arguments.
    if Nom_deg < 0 :
        raise ValueError, "expected Nominator Nom_deg >= 0"
    if Den_deg < 0 :
        raise ValueError, "expected Denomination Den_deg >= 0"
    if x.ndim != 1:
        raise TypeError, "expected 1D vector for x"
    if x.size == 0:
        raise TypeError, "expected non-empty vector for x"
    if y.ndim < 1 or y.ndim > 2 :
        raise TypeError, "expected 1D or 2D array for y"
    if x.shape[0] != y.shape[0] :
        raise TypeError, "expected x and y to have same length"
   
    Cn = np.ones(Nom_deg+1)
    Cd = np.ones(Den_deg+1)
    args = (len(Cn), len(Cd) )
    args = list (args)
    for i in Cn: args.append(i)
    for i in Cd: args.append(i)
    opt_param, covar_param  = curve_fit(ratfunctval, x, y, args)
   
    if full :
        return opt_param, covar_param
    else :
        return opt_param
