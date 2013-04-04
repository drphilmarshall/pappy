# ======================================================================

''' 
Functions for plotting PDFs from samples, as smoothed histograms.
'''

# ======================================================================
# Globally useful modules:

import string,numpy,pylab,sys,getopt

import pappy

# ======================================================================
# Subroutine to plot 1D PDF as histogram

def pdf1d(d,imp,bins,smooth,color):

  from scipy import ndimage

  pylab.xlim([bins[0],bins[-1]])

  # Bin the data in 1D, return histogram h(x):
  h,x = numpy.histogram(d,weights=imp,bins=bins,range=[bins[0],bins[-1]])
  norm = numpy.sum(h)
  h = h/norm

  # Report some statistics:
  median,errplus,errminus = pappy.compress_histogram(h,x,ci=68)
  result = pappy.format_point_estimate(median,errplus,errminus)

  # Smooth and normalise the histogram into a PDF:
  p = ndimage.gaussian_filter1d(h,smooth)
  norm = numpy.sum(p)
  p = p/norm

  # Plot:
  pylab.plot(x[:-1],p,drawstyle='line',color=color)  

  return p.max(),result

# ======================================================================
# Subroutine to plot 2D PDF as contours

def pdf2d(ax,ay,imp,xbins,ybins,smooth,color,style,conditional=False):

  from scipy import ndimage

  pylab.xlim([xbins[0],xbins[-1]])
  pylab.ylim([ybins[0],ybins[-1]])

  # npts = int((ax.size/4)**0.5)
  H,x,y = pylab.histogram2d(ax,ay,weights=imp,bins=[xbins,ybins])
  
  totalmass = sum(H.flatten())

  # Smooth the histogram into a PDF:
  if conditional:
  
    ncolumns = len(H[0,:])
    totalstdev = numpy.sqrt(numpy.var(ay))
    for i in range(ncolumns):
      p = H[i,:]
      # Need to choose smoothing scale carefully here! Columns with fewer points need
      # bigger smoothing scales:
      norm = numpy.sum(p)
      if (norm > 0.0):
        yy = ybins[1:]
        mean,stdev,Neff,N95 = pappy.meansd(yy,wht=p)
        blur = (smooth + smooth*(stdev/totalstdev)**2)
        # print "i,norm,totalmass, smooth,blur = ",i,norm,totalmass,smooth,blur
        H[i,:] = ndimage.gaussian_filter1d(p.flatten(),blur)
        
  else:
    H = ndimage.gaussian_filter(H,smooth)

  # For a conditional PDF Pr(y|x), normalise PDF in columns (constant x):
  if conditional:
    for i in range(len(H[0,:])):
      p = H[i,:]
      norm = numpy.sum(p.flatten())
      # Can only estimate conditional where there are enough points! Rough
      # estimate - focus on 99.9% of the mass:
      norm = norm * (norm > 0.001*totalmass)
      H[i,:] = p * (norm > 0.0) / (norm + (norm == 0.0))
  else:
    norm = numpy.sum(H.flatten())
    H = H * (norm > 0.0) / (norm + (norm == 0.0))
  
  sortH = numpy.sort(H.flatten())
  cumH = sortH.cumsum()
  # 1, 2, 3-sigma, for the old school:
  lvl00 = 2*sortH.max()
  lvl68 = sortH[cumH>cumH.max()*0.32].min()
  lvl95 = sortH[cumH>cumH.max()*0.05].min()
  lvl997 = sortH[cumH>cumH.max()*0.003].min()

#   print "2D histogram: min,max = ",H.min(),H.max()
#   print "Contour levels: ",[lvl00,lvl68,lvl95,lvl997]

  if style == 'shaded':

    # Plot shaded areas first:
    pylab.contourf(H.T,[lvl997,lvl95],colors=color,alpha=0.1,\
                   extent=(xbins[0],xbins[-1],ybins[0],ybins[-1]))
    pylab.contourf(H.T,[lvl95,lvl68],colors=color,alpha=0.4,\
                   extent=(xbins[0],xbins[-1],ybins[0],ybins[-1]))
    pylab.contourf(H.T,[lvl68,lvl00],colors=color,alpha=0.7,\
                   extent=(xbins[0],xbins[-1],ybins[0],ybins[-1]))
  # endif

  # Always plot outlines:
  pylab.contour(H.T,[lvl68,lvl95,lvl997],colors=color,\
                  extent=(xbins[0],xbins[-1],ybins[0],ybins[-1]))
#   pylab.contour(H.T,[lvl68,lvl95],colors=color,\
#                   extent=(xbins[0],xbins[-1],ybins[0],ybins[-1]))

  return

# ======================================================================


