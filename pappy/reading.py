# ======================================================================

''' 
Functions for reading samples from pappy format text files
'''

# ======================================================================
# Globally useful modules:

import string,numpy,pylab,sys,getopt

import pappy

vb = 0

# ======================================================================
# NB: file MUST be in pappy format!

def read_header(datafile):

   file = open(datafile)
   # Read comma-separated string axis labels from line 1:
   labelsline = file.readline().strip()
   # Read comma-separated axis limits from line 2
   limitsline = file.readline().strip()
   # Read one line of data just to count columns:
   dataline = file.readline().strip()
   row = numpy.array(dataline.split( ),dtype=numpy.float)
   file.close()

   # Count columns:
   ncols = len(row)
   
   # Parse labels:
   lineafterhash = labelsline.split('#')
   if len(lineafterhash) != 2:
     raise 'ERROR: first line of file is not #-marked, comma-separated list of labels'
   else:
     labels = lineafterhash[1].split(',')
     if vb: print "read_header: labels: ",labels

   # Parse limits:
   limits = numpy.zeros([ncols,2])
   lineafterhash = limitsline.split('#')
   if len(lineafterhash) != 2:
     usedylimits = 1
   else:
     usedylimits = 0
     limitstrings = lineafterhash[1].split(',')
     nlimits = len(limitstrings)
     if (nlimits/2 != ncols):
       print 'ERROR: found ',nlimits,'axis limits for',ncols,'columns:'
       print limitstrings
       exit()
     else:
       for i in range(ncols):
         for j in range(2):
           k = 2*i + j
           limits[i,j] = float(limitstrings[k])*1.0
       if vb: print "read_header:  limits: ",limits

   return labels, limits, usedylimits
 

# ======================================================================


