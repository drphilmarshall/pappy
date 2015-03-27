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
    -h                Print this message [0]
    -v                Verbose operation [0]
    --eps             Postscript output
    --just-2D         Don't bother plotting 1D PDFs.
    -c --conditional  Plot 2D conditional distributions only

  INPUTS
    file1,color1  Name of textfile containing data, and color to use
    file2,color2   etc.

  OPTIONAL INPUTS
    -n --columns  List of columns to plot [all] NB. inputs are one-indexed!
    -w iw         Index of column containing weight of sample
    -L iL         Index of column containing likelihood of sample
    --plot-points Plot the samples themselves
    -s --smooth x Plot contours that are smoother by factor x [def=1]
    -o --output   Name of output file
    --test        Write out a set of test commands and exit

  OUTPUTS
    stdout        Useful information
    pngfile       Output plot in png format


  EXAMPLES

     CornerPlotter.py --test
     
  BUGS
    - Only works from command line at the moment!
    - Tick labels overlap, cannot remove first and last tick mark
    - Figure has no legend
    - Overlay of 1D estimates not enabled
    - No overlay of 1-D priors
    - Lines are too thin

  HISTORY
    2010-05-06 started Marshall/Auger (UCSB)
    2011-06-24 generalized Marshall (Courmayeur/Bologna)
    2012-05-03 split into pappy module Marshall (Oxford)
  """

  # --------------------------------------------------------------------

  try:
      opts, args = getopt.getopt(argv, "hvcew:L:n:o:s:",["help","verbose","test","conditional","eps","smooth","plot-points","just-2D","columns","output"])
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
  plotlegend = True
  
  plot2D = True
  plot1D = True
  conditional = False
  
  columns = 'All'
  outfile = None
  
  test = False
  
  smoothscale = 1.0
  
  # NB. wcol and Lcol are assumed to be entered indexed to 1!
  for o,a in opts:      
      if o in ("-v", "--verbose"):
          vb = True
      elif o in ("--test"):
          test = True
      elif o in ("-n", "--columns"):
          columns = a
      elif o in ("--plot-points"):
          plotpoints = True
      elif o in ("-s","--smooth"):
          smoothscale = float(a)
      elif o in ("--just-2D"):
          plot1D = False
      elif o in ("-c", "--conditional"):
          conditional = True
          plot1D = False
      elif o in ("-w"):
          wcol = int(a) - 1
      elif o in ("-L"):
          Lcol = int(a) - 1
      elif o in ("--eps"):
          eps = True
      elif o in ("-o", "--output"):
          outfile = a
      elif o in ("-h", "--help"):
          print CornerPlotter.__doc__
          return
      else:
          print "Couldn't understand option,argument: ",o,a
          assert False

  # Test cases:
  if test:
    print 'Try these:'
    print '  CornerPlotter.py -o test1.png examples/thetas.cpt,blue,shaded'
    print '  CornerPlotter.py -o test2.png -w 1 -n 2,3,4 examples/localgroup.cpt,red,shaded'
    print '  CornerPlotter.py -o test3.png --just-2D -w 1 -n 3,4 examples/localgroup.cpt,purple,shaded'
    print '  CornerPlotter.py -o test4.png --conditional --plot-points -n 2,3,4 examples/localgroup.cpt,green,shaded'
    return
  
  # Sort out output filename:
  if outfile == None:
    if eps:
      outfile = "cornerplot.eps"
    else:
      outfile = "cornerplot.png"
  
  # Check for datafiles in array args:

  # BUG: calling CornerPlotter from python, this line is needed for args 
  # to be treated as a list of strings, not a list of characters 
  # args = [args]
  # But then the command line version does not work...
  # Partial solution - split useful functions into pappy module. PJM

  # args = numpy.array([args])  might help?

  if len(args) > 0:
    datafiles = []
    colors = []
    styles = []
    legends = []
    for i in range(len(args)):
      pieces = args[i].split(',')
      if len(pieces) == 1: pieces = pieces + ['black']
      if len(pieces) == 2: pieces = pieces + ['outlined']
      if len(pieces) == 3: 
                           pieces = pieces + [' ']
                           plotlegend = False
      datafiles = datafiles + [pieces[0]]
      colors = colors + [pieces[1]]
      styles = styles + [pieces[2]]
      legends = legends + [pieces[3]]
    if vb:
      print "Making corner plot of data in following files:",datafiles
      print "using following colors:",colors
      if not plot1D: print "Leaving out 1D plots"
      print "Output will be stored in "+outfile
      if eps: print "and will be in postscript format"
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
    hspace=0.04)
  fig.subplots_adjust(**adjustprops)

  # Need to set global linewidth thicker here!

  # No. of bins used:
  nbins = 161
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
    legend = legends[k]

    print "\nPlotting PDFs given in "+datafile+" as "+color+" "+style+" contours, and labelled as '"+legend+"'"

    # Read in data:
    data = numpy.loadtxt(datafile)

    # Start figuring out how many parameters we have - index will be a
    # list of column numbers containg the parameter values.
    # NB. ALL files must have same parameters/weights structure!
    if ( k == 0 ):
      npars = data.shape[1]
      index = numpy.arange(npars)
      hmax = numpy.zeros([npars])

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
      print "Only using data in",npars,"columns ",index,": "

    # Now parameter list is in index - which is fixed for other datasets

    # Font sizes - can only be set when no. of panels in grid is known:
    #   Big font sizes: {npars,bfs}={1,14},{2,13},{3,12},{4,10}
    
    if plot1D:
      ngrid = npars
    else:
      ngrid = npars - 1
    
    if ngrid < 4:
      bfs = 20 - 2*ngrid
    else:
      bfs = 12
    sfs = bfs - 2

    params = { 'axes.labelsize': bfs,
                'text.fontsize': bfs,
              'legend.fontsize': bfs,
              'xtick.labelsize': sfs,
              'ytick.labelsize': sfs}
    pylab.rcParams.update(params)

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    # 1st data file defines labels and parameter ranges:
    if k == 0:
 
      alllabels,alllimits,usedylimits = pappy.read_header(datafiles[k])
      if (usedylimits and vb): print 'No axis limits found, using 5-sigma ranges'
      
      # Now pull out just the labels and limits we need:
      limits = numpy.zeros([npars,2])
      ii = 0
      for i in index:
        limits[ii,:] = alllimits[i,:]
        ii = ii + 1
      labels = alllabels[:]

    # Set up dynamic axis limits, and smoothing scales:
    dylimits = numpy.zeros([npars,2])
    smooth = numpy.zeros(npars)
    for i in range(npars):
      col = index[i]
      # Get data subarray, and measure its mean and stdev:
      d = data[:,col].copy()
      mean,stdev,Neff,N95 = pappy.meansd(d,wht=wht)
      if vb: print "col = ",col," mean,stdev,nd = ",mean,stdev,Neff,N95
      
      # Set smoothing scale for this parameter, in physical units.
      smooth[i] = smoothscale*0.5*4.0*stdev/(N95**0.33)
      
      # Cf Jullo et al 2007, who use a bin size given by
      #  w = 2*IQR/N^(1/3)  for N samples, interquartile range IQR
      # For a Gaussian, IQR is not too different from 2sigma. 4sigma/N^1/3?
      # Also need N to be the effective number of parameters - return
      # form meansd as sum of weights!
      # Set 10 sigma limits:
      
      dylimits[i,0] = mean - 10*stdev
      dylimits[i,1] = mean + 10*stdev

    # Now set up bin arrays based on dynamic limits,
    # and convert smoothing to pixels:
    bins = numpy.zeros([npars,nbins])
    for i in range(npars):
      col = index[i]
      bins[i] = numpy.linspace(dylimits[i,0],dylimits[i,1],nbins)
      smooth[i] = smooth[i]/((dylimits[i,1]-dylimits[i,0])/float(nbins))
      if vb: print "col = ",col," smooth = ",smooth[i]
      if vb: print "binning limits:",dylimits[i,0],dylimits[i,1]

    # Optional: use dylimits as limits, again at 1st datafile:
    if (k == 0 and usedylimits == 1): limits = dylimits

    for i in range(npars):
        limits[i,0] = limits[i,0] + tiny*abs(limits[i,0])
        limits[i,1] = limits[i,1] - tiny*abs(limits[i,1])

    # Good - limits are set.
    # Report stats:
    
    print "Effective number of samples: Neff,N95,N =",Neff,N95,len(d)
    
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

        if plot1D and j==i:

          # Move to next subplot:
          panel = j*ngrid+i+1
          pylab.subplot(ngrid,ngrid,panel)

#           # Percentiles etc are too slow - get PDF1D to do it?
#           # Report some statistcs:
#           if vb:
#             pct = percentiles(d1,wht)
#             print "  Percentiles (16,50,84th) =",pct
#             median = pct[1]
#             errplus = pct[2] - pct[1]
#             errminus = pct[1] - pct[0]
#             print "  -> ",labels[col]+" = $",median,"^{",errplus,"}_{",errminus,"}$"

          # Plot 1D PDF, defined in subroutine below
          dummy,estimate = pappy.pdf1d(d1,wht,bins[i],smooth[i],color)
          print "  Par no.",col+1,",",labels[col],"=",estimate
          if k == 0: hmax[i] = dummy

          # Write the estimate on the plot?
          
          

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
          plottedsomething = True

        elif plot2D and j!=i:

          # Move to next subplot:
          if plot1D:
            panel = j*ngrid+i+1
          else:
            #  i j panel
            #  0 1   1 = (j-1)*ngrid+i+1
            #  0 2   3
            #  1 2   4
            panel = (j-1)*ngrid+i+1
            # print "XXXXX i,j,panel = ",i,j,panel
          pylab.subplot(ngrid,ngrid,panel)

         # Get 2nd data set:
          d2 = data[:,row].copy()
          if vb: print "Read in ",row,"th column of data: min,max = ",min(d2),max(d2)

          # Plot 2D PDF, defined in subroutine below
          if vb: print "Calling pdf2d for col,row = ",col,row
          fwhm = 0.5*(smooth[i]+smooth[j])
          pappy.pdf2d(d1,d2,wht,bins[i],bins[j],fwhm,color,style,conditional=conditional)          

          # If we are just plotting one file, overlay samples:
          if (len(datafiles) == 1 and plotpoints):
            ptsize = 4.0/numpy.log10(len(d))  # N=100 -> 2, N=10,000 -> 1
            pylab.plot(d1,d2,'ko',ms=ptsize)

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
          plottedsomething = True
          
        else:  
          plottedsomething = False

        if plottedsomething:
          if vb: print "Done subplot", panel,"= (", i, j,")"
          if vb: print "  - plotted",labels[col],"vs",labels[row]
          if vb: print "--------------------------------------------------"

      # endfor
    # endfor

  # endfor

  # Plot legend? In opposite corner if ngrid > 1, otherwise?
  if plotlegend:
    pass
  
  # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

  # Plot graph to file:
  
  pylab.savefig(outfile,dpi=300)
  print "\nFigure saved to file:",outfile

  return

# ======================================================================

if __name__ == '__main__':
  CornerPlotter(sys.argv[1:])

# ======================================================================
