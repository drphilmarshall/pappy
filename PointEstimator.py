#!/usr/bin/env python
# ======================================================================

# Globally useful modules:

import matplotlib
# Force matplotlib to not use any Xwindows backend:
matplotlib.use('Agg')

# Fonts, latex:
matplotlib.rc('font',**{'family':'serif', 'serif':['TimesNewRoman']})
matplotlib.rc('text', usetex=True)

import string,numpy,pylab,sys,getopt

import pappy

# ======================================================================

def PointEstimator(argv):
  """
  NAME
    PointEstimator.py

  PURPOSE
    Compute a point estimate for a required parameter, given samples
    drawn from that parameter's PDF.

  COMMENTS
    Expected data format is plain text, header marked by # in column 1,  with
    header lines listing:
      1) Parameter names separated by commas. These names will
           be used as labels on the plots, and should be in latex.
      2) Parameter ranges [min,max,] for plot axes.

  USAGE
    PointEstimator.py [flags]

  FLAGS
    -h            Print this message [0]
    -v            Verbose operation [0]

  INPUTS
    file          Name of textfile containing data

  OPTIONAL INPUTS
    -n --columns  List of columns to plot [all] NB. inputs are one-indexed!
    -w wcol       Column number of sample weight
    -L Lcol       Index of column containing likelihood of sample (to be ignored)
    -t --terse    Terse output (one line per parameter)

  OUTPUTS
    stdout        Useful information

  EXAMPLES

    PointEstimator.py -w 1 thetas.cpt
    
    PointEstimator.py -t -w 1 -n 2,3,4,5 thetas.cpt

  BUGS

  HISTORY
    2012-05-03 started Marshall (Oxford)
  """

  # --------------------------------------------------------------------

  try:
      opts, args = getopt.getopt(argv, "hvtw:n:",["help","verbose","terse","columns"])
  except getopt.GetoptError, err:
      # print help information and exit:
      print str(err) # will print something like "option -a not recognized"
      print PointEstimator.__doc__
      return

  vb = False
  longwinded = True
  wcol = -1
  Lcol = -1
  columns = 'All'
  # NB. wcol is assumed to be entered indexed to 1!
  for o,a in opts:
      if o in ("-v", "--verbose"):
          vb = True
      elif o in ("-t", "--terse"):
          longwinded = False
      elif o in ("-n","--columns"):
          columns = a
      elif o in ("-w"):
          wcol = int(a) - 1
      elif o in ("-L"):
          Lcol = int(a) - 1
      elif o in ("-h", "--help"):
          print PointEstimator.__doc__
          return
      else:
          assert False, "unhandled option"
  
  # Check for datafiles in array args:

  if len(args) == 1:
    datafile = args[0]
    if vb or longwinded:
      print "Point estimate(s) of parameter(s) in file "+datafile
  else :
    print PointEstimator.__doc__
    return

# --------------------------------------------------------------------
# Read in data:
  
  data = numpy.loadtxt(datafile)

  # Start figuring out how many parameters we have - index will be a
  # list of column numbers containg the parameter values.
  # NB. ALL files must have same parameters/weights structure!
  npars = data.shape[1]
  index = numpy.arange(npars)

  # Some cols may not be data, adjust index accordingly.

  # Weights/importances:
  if wcol >= 0 and wcol < data.shape[1]:
    if vb: print "using weight in column: ",wcol
    wht = data[:,wcol].copy()
    keep = (index != wcol)
    index = index[keep]
    npars = npars - 1
  else:
    if (wcol >= data.shape[1]):
      print "WARNING: only ",data.shape[1]," columns are available"
    if vb: print "Setting weights to 1"
    wht = 0.0*data[:,0].copy() + 1.0

  # Likelihood values:
  if Lcol >= 0:
    Lhood = data[:,Lcol].copy()
    if ( k == 0 ):
      keep = (index != Lcol)
      index = index[keep]
      npars = npars - 1
    if vb: print "using lhood in column: ",Lcol
  else:
    Lhood = 0.0*data[:,0].copy() + 1.0

  # Having done all that, optionally overwrite index with specified list
  # of column numbers. Note conversion to zero-indexed python:
  if columns != 'All':
    pieces = columns.split(',')
    index = []
    for piece in pieces:
      index.append(int(piece) - 1)
    npars = len(index)
    if vb: print "Only using data in",npars,"columns (",index,"): "

  # Now parameter list is in index - which is fixed for other datasets

  labels,limits,dummy = pappy.read_header(datafile)

# --------------------------------------------------------------------
# Loop over parameters, doing calculations:
  
  for i in range(npars):

    col = index[i]

    d = data[:,col].copy()

    mean,stdev,Neff,N95 = pappy.meansd(d,wht=wht)
    median,errplus,errminus = pappy.compress_samples(d,wht=wht,ci=68)

    estimate = pappy.format_point_estimate(median,errplus,errminus)
    if longwinded: 
      print "  Par no.",col+1,",",labels[col],"=",estimate
    else:
      print estimate

    if vb: 
      print "  Par no.",col+1," mean,stdev,Neff,N95 = ",mean,stdev,Neff,N95

# --------------------------------------------------------------------

  return

# ======================================================================

if __name__ == '__main__':
  PointEstimator(sys.argv[1:])

# ======================================================================
