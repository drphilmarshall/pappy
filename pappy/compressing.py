# ======================================================================

''' 
Functions for computing and presemting 
compressed quantities from input arrays of 
samples, such as means, percentiles and so on.
'''

# ======================================================================
# Globally useful modules:

import string,numpy,pylab,sys,getopt,os

import pappy

# ======================================================================
# Subroutine to return mean and stdev of numpy.array x

def meansd(x,wht=None):

  N = len(x)
  if (wht == None):
    wht = numpy.ones(N)
  elif len(wht) != N:
    print "ERROR: data and wht arrays don't match in meansd:",N,len(wht)
    return numpy.inf,numpy.inf,numpy.inf,numpy.inf

# Simple sum of weights:
  Neff = pappy.effective_number_of_samples(wht,method='simple')

# How many samples to get 95% of the weight?
  N95 = pappy.number_with_combined_probability_mass_of(0.95,wht)
  if N95 == 0: N95 = int(Neff)

# Weighted mean and stdev:  
  mean = numpy.sum(x*wht)/numpy.sum(wht)
  var = numpy.sum((x-mean)*wht*(x-mean))/numpy.sum(wht)
  stdev = numpy.sqrt(var)

  return mean,stdev,Neff,N95

# ======================================================================
# Effective number of samples:

def effective_number_of_samples(wht,method='simple_sum'):

  if method == 'Neal':  
    # Radford Neal's Neff:  
    meanwht = numpy.sum(wht*wht)/numpy.sum(wht)
    varwht = numpy.sum((wht-meanwht)*wht*(wht-meanwht))/numpy.sum(wht)
    Neff = N/(1.0+varwht)

  elif method == 'Entropy':
    # Neff based on the Shannon entropy formula
    normedwht = wht/wht.sum()
    H = -numpy.sum(normedwht*numpy.log(normedwht + 1E-300))
    Neff = numpy.exp(H)

  else:
    # Simple sum:
    w = wht/wht.max()
    Neff = numpy.sum(w)
    
  return Neff  

# ======================================================================
# Compute number of samples with combined probability mass of p:

def number_with_combined_probability_mass_of(p,wht):

  swht = numpy.sort(wht)
  cswht = swht.cumsum()
  level = swht[cswht > cswht.max()*(1-p)].min()

  return len(numpy.where(swht > level)[0])

# ======================================================================
# Compute probability of x being greater/less than x0:

def probability_of(x,cf,x0,wht):

  index = numpy.argsort(x)
  xs = x[index]
  ws = wht[index]
  cs = ws.cumsum()
  # Normalise:
  cs = cs/cs[-1]
  p = cs[xs>x0].min()
  
  if cf == 'less than':
    percentage = (1.0 - p)*100.0
  elif cf == 'greater than':
    percentage = p*100.0
  else:
    raise "Unrecognised comparison"+cf
    
  return nint(percentage)

# ======================================================================
# Given normalised histogram, return median and uncertainty based on
# percentiles.

def compress_histogram(h,x,ci=68):

  c = h.cumsum()
  # Normalise:
  c = c/c[-1]
  
  return percentiles_from_cumulant(x,c,ci)

# ======================================================================
# Given normalised histogram, return median and uncertainty based on
# percentiles.

def compress_samples(x,wht,ci=68):

  index = numpy.argsort(x)
  xs = x[index]
  ws = wht[index]
  cs = ws.cumsum()
  # Normalise:
  cs = cs/cs[-1]
     
  return percentiles_from_cumulant(xs,cs,ci)

# ======================================================================

def percentiles_from_cumulant(x,cumulant,ci):

  low  = (1.0 - ci/100.0)/2.0    # 0.16 for ci=68
  high = 1.0 - low               # 0.84 for ci=68
  
  pctlow = x[cumulant>low].min()
  pct50 = x[cumulant>0.50].min()
  pcthigh = x[cumulant>high].min()
  median = pct50
  errplus = numpy.abs(pcthigh - pct50)
  errminus = numpy.abs(pct50 - pctlow)

#   # Check for failure:
#   if errplus == 0 or errminus == 0:
#     print "WARNING: zero width credible region. "
#     print "             lower = ",pctlow
#     print "            median = ",median
#     print "            higher = ",pcthigh
#     print "   Here are the cumulant histogram values:"
#     print cumulant
#     print "   and the corresponding ordinates:"
#     print x

  return median,errplus,errminus

# ======================================================================

def format_point_estimate(x,a,b):

# How many sf should we use? a and b are positive definite, so take logs:
  # print "format_point_estimate: x,a,b = ",x,a,b

  # if a <= 0 or b <= 0:
    # print "Warning: one error bar is zero-sized: x +a/-b where a,b = ",a,b
    # return 'No estimate possible'

  if a > 0:
    loga = numpy.log10(a)
    if loga > 0:
      intloga = int(loga)
    else:
      intloga = int(loga) - 1
  else: 
    intloga = 1e32    
  if b > 0:    
    logb = numpy.log10(b)
    if logb > 0:
      intlogb = int(logb)
    else:
      intlogb = int(logb) - 1
  else: 
    intlogb = 1e32    
  # print "intloga,intlogb = ", intloga,intlogb
  
  k = int(numpy.min([intloga,intlogb]))
  # Go one dp further for extra precision...
  k -= 1
  
  if k > 100:
    estimate = "undefined, PDF too narrow to measure"
  
  else:
    base = 10.0**k
    # print "k,base = ",k,base

    # Now round off numbers to this no. of SF:
    if k >=0:
      fmt = "%d"
    else:
      fmt = "%."+str(abs(k))+"f"
    rx = fmt % (base * nint(x/base))
    ra = fmt % (base * nint(a/base))
    rb = fmt % (base * nint(b/base))
    # print "base*nint(x/base) = ",base*nint(x/base)
    # print "rx,ra,rb = ",rx,ra,rb

    estimate = "$ "+rx+"^{+"+ra+"}_{-"+rb+"} $"
    # print "estimate = ",estimate

  return estimate

def nint(x):
  i = int(x)
  d = x - i
  if d < 0.5:
    ii = i
  else:
    ii = i + 1
  return ii

# ======================================================================
# Subroutine to return median and percentiles of numpy.array x
# 
# def percentiles(x,w):
# 
#   # First sort the sample x values, and find corresponding weights:
#   index = numpy.argsort(x)
#   xx = x[index]
#   ww = w[index]
# 
#   # Check weights - if they are all the same there is a shortcut!
# 
#   wmin = numpy.min(ww)
#   wmax = numpy.max(ww)
# 
#   if wmin == wmax:
# 
#     N = len(xx)
#     mark = numpy.array([int(0.16*N),int(0.50*N),int(0.84*N)],dtype=int)
# 
#     p = xx[mark]
# 
#   else:
# 
#     # Make weighted array, and work out values of integral to each percentile:
#     wx = xx*ww
#     N = numpy.sum(wx)
#     mark = numpy.array([0.16*N,0.50*N,0.84*N],dtype=int)
#     # Now accumulate the array until the marks are passed (this is very slow...):
#     p = numpy.zeros(3)
#     j = 0
#     for i in range(len(x)):
#       cumulant = numpy.sum(wx[0:i])
#       if cumulant >= mark[j]:
#         p[j] = x[i]
#         j += 1
#     # Done. This will probably take ages...
# 
#   return p
# 
# ======================================================================
# Testing:

if __name__ == '__main__':
    
  datafile = os.environ['PAPPY_DIR']+'/examples/localgroup.cpt' 
  data = numpy.loadtxt(datafile)
  wht = data[:,0].copy()
  labels,limits,dummy = pappy.read_header(datafile)

  col = 1  # M_MW

  d = data[:,col].copy()
  mean,stdev,Neff,N95 = pappy.meansd(d,wht=wht)
  median,errplus,errminus = pappy.compress_samples(d,wht=wht,ci=95)
  estimate = pappy.format_point_estimate(median,errplus,errminus)  
  print "  95% limits: ",labels[col],"=",estimate
   
  col = 3 # M_M31 / M_MW
  
  d = data[:,col].copy()
  p = probability_of(d,'greater than',0.0,wht)
  print "  Pr(M1 > M2) = ",p,"%"


# ======================================================================
