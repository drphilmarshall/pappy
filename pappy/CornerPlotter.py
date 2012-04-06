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

# ======================================================================

def CornerPlotter(argv):
  """
  NAME
    CornerPlotter.py

  PURPOSE
    Plot 2D projections of ND parameter space as triangular
    array of panels. Include 1D marginalised distributions
    on the diagonal.

  COMMENTS
    Expected data format is plain text, header marked by # in column 1,  with
    header lines listing:
      1) Parameter names separated by commas. These names will
           be used as labels on the plots, and should be in latex.
      2) Parameter ranges [min,max,] for plot axes.

  USAGE
    CornerPlotter.py [flags]

  FLAGS
    -u            Print this message [0]
    --eps         Postscript output

  INPUTS
    file1,color1  Name of textfile containing data, and color to use
    file2,color2   etc.

  OPTIONAL INPUTS
    -n --columns  List of columns to plot [all] NB. inputs are one-indexed!
    -w iw         Index of column containing weight of sample
    -L iL         Index of column containing likelihood of sample
    --plot-points Plot the samples themselves
    -o --output   Name of output file

  OUTPUTS
    stdout        Useful information
    pngfile       Output plot in png format


  EXAMPLES

    CornerPlotter.py thetas.cpt,blue,shaded

    CornerPlotter.py -w 1 -L 2 --plot-points J2141-disk_bulge.txt,red,shaded

    CornerPlotter.py -w 1 -L 2 -n 3,4,5 J2141-disk_bulge.txt,gray,outlines

  BUGS
    - Tick labels overlap, cannot remove first and last tick mark
    - Figure has no legend
    - no overlay of 1-D priors

  HISTORY
    2010-05-06 started Marshall/Auger (UCSB)
    2011-06-24 generalized Marshall (Courmayeur/Bologna)
  """

  # --------------------------------------------------------------------

  try:
      opts, args = getopt.getopt(argv, "hvew:L:n:o:",["help","verbose","eps","plot-points","columns","output"])
  except getopt.GetoptError, err:
      # print help information and exit:
      print str(err) # will print something like "option -a not recognized"
      print CornerPlotter.__doc__
      return

  vb = False
  wcol = -1
  Lcol = -1
  plotpoints = False
  eps = False
  columns = 'All'
  output = 'Null'
  # NB. wcol and Lcol are assumed to be entered indexed to 1!
  for o,a in opts:
      if o in ("-v", "--verbose"):
          vb = True
      elif o in ("-n","--columns"):
          columns = a
      elif o in ("--plot-points"):
          plotpoints = True
      elif o in ("-w"):
          wcol = int(a) - 1
      elif o in ("-L"):
          Lcol = int(a) - 1
      elif o in ("--eps"):
          eps = True
      elif o in ("-o","--output"):
          output = a
      elif o in ("-h", "--help"):
          print CornerPlotter.__doc__
          return
      else:
          assert False, "unhandled option"

  # Check for datafiles in array args:
  if len(args) > 0:
    datafiles = []
    colors = []
    styles = []
    for i in range(len(args)):
      pieces = args[i].split(',')
      if len(pieces) != 3:
        print "ERROR: supply input data as 'filename,color,style'"
        exit()
      datafiles = datafiles + [pieces[0]]
      colors = colors + [pieces[1]]
      styles = styles + [pieces[2]]
    if vb:
      print "Making corner plot of data in following files:",datafiles
      print "using following colors:",colors
      if eps: "Output will be postscript"
  else :
    print CornerPlotter.__doc__
    return

  # --------------------------------------------------------------------

  # Start figure, set up viewing area:
  figprops = dict(figsize=(8.0, 8.0), dpi=128)                                          # Figure properties
  fig = pylab.figure(**figprops)

  # Need small space between subplots to avoid deletion due to overlap...
  adjustprops = dict(\
    left=0.1,\
    bottom=0.1,\
    right=0.95,\
    top=0.95,\
    wspace=0.04,\
    hspace=0.08)
  fig.subplots_adjust(**adjustprops)

  # No. of bins used:
  nbins = 81
  # Small tweak to prevent too many numeric tick marks
  tiny = 0.01

  # --------------------------------------------------------------------

  # Plot files in turn, using specified color scheme.

  for k in range(len(datafiles)):

    datafile = datafiles[k]
    color = colors[k]
    #     if (color == 'black' or eps):
    #       style = 'outlines'
    #     else:
    #       style = 'shaded'
    style = styles[k]
    legend = datafile

    print "\nPlotting PDFs given in "+datafile+" as "+color+" "+style

    # Read in data:
    data = numpy.loadtxt(datafile)

    # Start figuring out how many parameters we have - index will be a
    # list of column numbers containg the parameter values.
    # NB. ALL files must have same parameters/weights structure!
    if ( k == 0 ):
      npars = data.shape[1]
      index = numpy.arange(npars)

    # Some cols may not be data, adjust index accordingly.

    # Weights/importances:
    if wcol >= 0 and wcol < data.shape[1]:
      if vb: print "using weight in column: ",wcol
      wht = data[:,wcol].copy()
      if ( k == 0 ):
        keep = (index != wcol)
        index = index[keep]
        npars = npars - 1
        if vb: print " index = ",index
    else:
      if (wcol >= data.shape[1]):
        print "WARNING: only ",data.shape[1]," columns are available"
        print "Setting weights to 1"
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
      print "Only using data in",npars,"columns (",index,"): "

    # Now parameter list is in index - which is fixed for other datasets

    # Font sizes - can only be set when no. of panels is known:
    #   Big font sizes: {npars,bfs}={1,14},{2,13},{3,12},{4,10}
    if npars < 4:
      bfs = 15 - npars
    else:
      bfs = 10
    sfs = bfs - 2

    params = { 'axes.labelsize': bfs,
                'text.fontsize': bfs,
              'legend.fontsize': sfs,
              'xtick.labelsize': sfs,
              'ytick.labelsize': sfs}
    pylab.rcParams.update(params)

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    # 1st data file defines labels and parameter ranges:

    if k == 0:

      limits = numpy.zeros([npars,2])
      hmax = numpy.zeros([npars])

      # NB: file MUST be of correct format!

      # Read comma-separated string axis labels from line 1:
      file = open(datafile)
      labelsline = file.readline().strip()
      # Read comma-separated axis limits from line 2
      limitsline = file.readline().strip()
      file.close()

      lineafterhash = labelsline.split('#')
      if len(lineafterhash) != 2:
        print 'ERROR: first line of file is not #-marked, comma-separated list of labels'
        exit()
      else:
        labels = lineafterhash[1].split(',')
        if vb: print "Plotting",npars," parameters: ",labels

      lineafterhash = limitsline.split('#')
      if len(lineafterhash) != 2:
        if vb: print 'No axis limits found, using 5-sigma ranges'
        usedylimits = 1
      else:
        limitstrings = lineafterhash[1].split(',')
        nlimits = len(limitstrings)
        if (nlimits/2 != data.shape[1]):
          print 'ERROR: found ',nlimits,'axis limits for',data.shape[1],'columns:'
          print limitstrings
          exit()
        else:
          ii = 0
          for i in index:
            for j in range(2):
              l = 2*i + j
              limits[ii,j] = float(limitstrings[l])*1.0
            ii = ii + 1
          if vb: print "Plot limits: ",limits
        usedylimits = 0

    # OK, back to any datafile.

    # Set up dynamic axis limits, and smoothing scales:
    dylimits = numpy.zeros([npars,2])
    smooth = numpy.zeros(npars)
    for i in range(npars):
      col = index[i]
      # Get data subarray, and measure its mean and stdev:
      d = data[:,col].copy()
      mean,stdev = meansd(d,wht=wht)
      if vb: print "col = ",col," mean,stdev = ",mean,stdev
      # Set smoothing scale for this parameter, in physical units:
      smooth[i] = 0.1*stdev
      # Cf Jullo et al 2007, who use a bin size given by
      #  w = 2*IQR/N^(1/3)  for N samples, interquartile range IQR
      # For a Gaussian, IQR is not too different from 2sigma. 4sigma/N^1/3?
      # Also need N to be the effective number of parameters - return
      # form meansd as sum of weights!
      # Set 5 sigma limits:
      dylimits[i,0] = mean - 5*stdev
      dylimits[i,1] = mean + 5*stdev

    # Now set up bin arrays based on dynamic limits,
    # and convert smoothing to pixels:
    bins = numpy.zeros([npars,nbins])
    for i in range(npars):
      col = index[i]
      bins[i] = numpy.linspace(dylimits[i,0],dylimits[i,1],nbins)
      smooth[i] = smooth[i]/((dylimits[i,1]-dylimits[i,0])/float(nbins))
      if vb: print "col = ",col," smooth = ",smooth[i]
      if vb: print "binning limits:",dylimits[i,0],dylimits[i,1]

    if (k == 0):
      # Finalise limits, again at 1st datafile:
      if (usedylimits == 1): limits = dylimits

      for i in range(npars):
        limits[i,0] = limits[i,0] + tiny*abs(limits[i,0])
        limits[i,1] = limits[i,1] - tiny*abs(limits[i,1])

    # Good - limits are set.

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    # Loop over plotting panels. Arrangement is bottom left hand corner,
    # most important parameter as x in first column, (row=0,col=0) is
    # top left hand corner panel.

    # panel = (col,row):

    # 1=0,0  |  2=1,0  |  3=2,0  |  4=3,0  |  5=4,0
    # ---------------------------------------------
    # 6=0,1  |  7=1,1  |  8=2,1  |  9=3,1  | 10=4,1
    # ---------------------------------------------
    #11=0,2  | 12=1,2  | 13=2,2  | 14=3,2  | 15=4,2
    # ---------------------------------------------
    #16=0,3  | 17=1,3  | 18=2,3  | 19=3,3  | 20=4,3
    # ---------------------------------------------
    #21=0,4  | 22=1,4  | 23=2,4  | 24=3,4  | 25=4,4

    for i in range(npars):
      col = index[i]

      # Get data subarray:
      d1 = data[:,col].copy()
      if vb: print "Read in ",col,"th column of data: min,max = ",min(d1),max(d1)

      for j in range(i,npars):
        row = index[j]

        # Move to next subplot:
        panel = j*npars+i+1
        pylab.subplot(npars,npars,panel)

        if j==i:

          # Percentiles etc are too slow - get PDF1D to do it?
#           # Report some statistcs:
#           if vb:
#             pct = percentiles(d1,wht)
#             print "  Percentiles (16,50,84th) =",pct
#             median = pct[1]
#             errplus = pct[2] - pct[1]
#             errminus = pct[1] - pct[0]
#             print "  -> ",labels[col]+" = $",median,"^{",errplus,"}_{",errminus,"}$"

          # Plot 1D PDF, defined in subroutine below
          dummy,estimate = pdf1d(d1,wht,bins[i],smooth[i],color)
          print "  Par no.",col+1,",",labels[col],"=",estimate
          if k == 0: hmax[i] = dummy

          # Force axes to obey limits:
          pylab.axis([limits[i,0],limits[i,1],0.0,1.2*hmax[i]])
          # Adjust axes of 1D plots:
          ax = pylab.gca()
          # Turn off the y axis tick labels for all 1D panels:
          ax.yaxis.set_major_formatter(pylab.NullFormatter())
          # Turn off the x axis tick labels for all but the last 1D panel:
          if j<(npars-1):
            ax.xaxis.set_major_formatter(pylab.NullFormatter())
          pylab.xticks(rotation=45)
          # Label x axis, only on the bottom panels:
          if j==npars-1:
            pylab.xlabel(labels[col])

        else:

          # Get 2nd data set:
          d2 = data[:,row].copy()
          if vb: print "Read in ",row,"th column of data: min,max = ",min(d2),max(d2)

          # Plot 2D PDF, defined in subroutine below
          if vb: print "Calling pdf2d for col,row = ",col,row
          fwhm = 0.5*(smooth[i]+smooth[j])
          pdf2d(d1,d2,wht,bins[i],bins[j],fwhm,color,style)

          # If we are just plotting one file, overlay samples:
          if (len(datafiles) == 1 and plotpoints):
            pylab.plot(d1,d2,'ko',ms=0.1)

          # Force axes to obey limits:
          pylab.axis([limits[i,0],limits[i,1],limits[j,0],limits[j,1]])
          # Adjust axes of 2D plots:
          ax = pylab.gca()
          if i>0:
            # Turn off the y axis tick labels
            ax.yaxis.set_major_formatter(pylab.NullFormatter())
          if j<npars-1:
            # Turn off the x axis tick labels
            ax.xaxis.set_major_formatter(pylab.NullFormatter())
          # Rotate ticks so that axis labels don't overlap
          pylab.xticks(rotation=45)
          # Label x axes, only on the bottom panels:
          if j==npars-1:
            pylab.xlabel(labels[col])
          if i==0 and j>0:
          # Label y axes in the left-hand panels
            pylab.ylabel(labels[row])

        if vb: print "Done subplot", panel,"= (", i, j,")"
        if vb: print "  - plotting",labels[col],"vs",labels[row]
        if vb: print "--------------------------------------------------"

      # endfor
    # endfor

  # endfor

  # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

  # Plot graph to file:
  if output == 'Null':
    if eps:
      output = "cornerplot.eps"
    else:
      output = "cornerplot.png"

  pylab.savefig(output,dpi=300)
  print "\nFigure saved to file:",output

  exit()

# ======================================================================
# Subroutine to plot 1D PDF as histogram

def pdf1d(d,imp,bins,smooth,color):

  pylab.xlim([bins[0],bins[-1]])

  # Bin the data in 1D, return bins and heights
  # samples,positions,ax = pylab.hist(d,bins,fc='white')
  samples,positions = numpy.histogram(d,weights=imp,bins=bins,range=[bins[0],bins[-1]])

  # Normalise:
  norm = sum(samples)
  curve = samples/norm

  # Report some statistics:
  cumulant = curve.cumsum()
  # 1-sigma, for the old school:
  pct16 = positions[cumulant>0.16].min()
  pct50 = positions[cumulant>0.50].min()
  pct84 = positions[cumulant>0.84].min()
  median = pct50
  errplus = numpy.abs(pct84 - pct50)
  errminus = numpy.abs(pct50 - pct16)

  # Check for failure:
  if errplus == 0 or errminus == 0:
    print "ERROR: zero width credible region. Here's the histogram:"
    print curve
    print "ERROR: And the corresponding positions:"
    print positions
    print "ERROR: And here's the requested bins:"
    print bins
    sys.exit()

#  result = "$"+str(median)+"^{+"+str(errplus)+"}_{-"+str(errminus)+"}$"
  result = format_point_estimate(median,errplus,errminus)

  # Plot the PDF
  pylab.plot(positions[:-1],curve,drawstyle='steps-mid',color=color)

  # # Smooth the PDF:
  # H = ndimage.gaussian_filter(H,smooth)

  # print "1D histogram: min,max = ",curve.min(),curve.max()
  hmax = curve.max()

#   print "Plotted 1D histogram with following axes limits:"
#   print "  extent =",(bins[0],bins[-1])

  return hmax,result

# ======================================================================
# Subroutine to plot 2D PDF as contours

def pdf2d(ax,ay,imp,xbins,ybins,smooth,color,style):

  from scipy import ndimage

  pylab.xlim([xbins[0],xbins[-1]])
  pylab.ylim([ybins[0],ybins[-1]])

  # npts = int((ax.size/4)**0.5)
  H,x,y = pylab.histogram2d(ax,ay,weights=imp,bins=[xbins,ybins])

  # Smooth the PDF:
  H = ndimage.gaussian_filter(H,smooth)

  sortH = numpy.sort(H.flatten())
  cumH = sortH.cumsum()
  # 1, 2, 3-sigma, for the old school:
  lvl00 = 2*sortH.max()
  lvl68 = sortH[cumH>cumH.max()*0.32].min()
  lvl95 = sortH[cumH>cumH.max()*0.05].min()
  lvl99 = sortH[cumH>cumH.max()*0.003].min()

#   print "2D histogram: min,max = ",H.min(),H.max()
#   print "Contour levels: ",[lvl00,lvl68,lvl95,lvl99]

  if style == 'shaded':

    # Plot shaded areas first:
    pylab.contourf(H.T,[lvl99,lvl95],colors=color,alpha=0.1,\
                   extent=(xbins[0],xbins[-1],ybins[0],ybins[-1]))
    pylab.contourf(H.T,[lvl95,lvl68],colors=color,alpha=0.4,\
                   extent=(xbins[0],xbins[-1],ybins[0],ybins[-1]))
    pylab.contourf(H.T,[lvl68,lvl00],colors=color,alpha=0.7,\
                   extent=(xbins[0],xbins[-1],ybins[0],ybins[-1]))
  # endif

  # Always plot outlines:
  pylab.contour(H.T,[lvl68,lvl95,lvl99],colors=color,\
                  extent=(xbins[0],xbins[-1],ybins[0],ybins[-1]))



# ======================================================================
# Subroutine to return mean and stdev of numpy.array x

def meansd(x,wht=[0]):

  N = len(x)
  if len(wht) == 1:
    wht = numpy.ones(N)
  elif len(wht) != N:
    print "ERROR: data and wht arrays don't match in meansd:",N,len(wht)
    sys.exit()

  mean = numpy.sum(x*wht)/numpy.sum(wht)
  var = numpy.sum((x-mean)*wht*(x-mean))/numpy.sum(wht)
  stdev = numpy.sqrt((var)*float(N)/float(N-1))

  return mean,stdev

# ======================================================================
# Subroutine to return median and percentiles of numpy.array x

def percentiles(x,w):

  # First sort the sample x values, and find corresponding weights:
  index = numpy.argsort(x)
  xx = x[index]
  ww = w[index]

  # Check weights - if they are all the same there is a shortcut!

  wmin = numpy.min(ww)
  wmax = numpy.max(ww)

  if wmin == wmax:

    N = len(xx)
    mark = numpy.array([int(0.16*N),int(0.50*N),int(0.84*N)],dtype=int)

    p = xx[mark]

  else:

    # Make weighted array, and work out values of integral to each percentile:
    wx = xx*ww
    N = numpy.sum(wx)
    mark = numpy.array([0.16*N,0.50*N,0.84*N],dtype=int)
    # Now accumulate the array until the marks are passed (this is very slow...):
    p = numpy.zeros(3)
    j = 0
    for i in range(len(x)):
      cumulant = numpy.sum(wx[0:i])
      if cumulant >= mark[j]:
        p[j] = x[i]
        j += 1
    # Done. This will probably take ages...

  return p

# ======================================================================

def format_point_estimate(x,a,b):

# How many sf should we use? a and b are positive definite, so take logs:
  # print "x,a,b = ",x,a,b

  if a <= 0 or b <= 0:
    print "ERROR: this should not happen: a,b = ",a,b
    sys.exit()

  loga = numpy.log10(a)
  if loga > 0:
    intloga = int(loga)
  else:
    intloga = int(loga) - 1
  logb = numpy.log10(a)
  if logb > 0:
    intlogb = int(logb)
  else:
    intlogb = int(logb) - 1
  # print "intloga,intlogb = ", intloga,intlogb
  # Go one dp further for extra precision...
  k = numpy.min([intloga,intlogb]) - 1
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
  # print "fmt = ",fmt
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

if __name__ == '__main__':
  CornerPlotter(sys.argv[1:])

# ======================================================================
