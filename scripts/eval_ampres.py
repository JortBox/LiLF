#!/usr/bin/env python3

import os, sys, pickle, glob
import numpy as np
from losoto import h5parm

# get input
h5parmFiles = sorted(glob.glob('*/cal-bp.h5'))
soltabName = 'amplitudeRes'
badobs_picklefile = '/homes/fdg/storage/scripts/survey/LoLSS-bkp/badobsids.pickle'

allvalstd = []; allvaldynrange = []; alltimes = [];  allids = []; badids = []
for h5parmFile in h5parmFiles:
    #print(f"Working on: {h5parmFile}")
    h5 = h5parm.h5parm(h5parmFile, readonly=True)
    obsid = int(h5parmFile[2:].split('_-_')[0])
    solset = h5.getSolset('sol000')
    soltab = solset.getSoltab(soltabName)
    
    vals = soltab.getValues(retAxesVals=False)
    weights = soltab.getValues(weight=True, retAxesVals=False)
    times = soltab.getAxisValues('time')
    if np.all(weights == 0):
        print(f'{h5parmFile} - All flagged - BAD!')
        badids.append(obsid)
        valstd = 999
        valdynrange = 999
    else:
        valstd = np.std(vals[weights!=0])
        valmin = np.min(vals[weights!=0])
        valmax = np.max(vals[weights!=0])
        valdynrange = valmax-valmin

        if (valstd > 0.1 and valdynrange > 1):
            print(f'{h5parmFile} - Std all residuals: {valstd} - Dyn range: {valdynrange} - BAD!')
            badids.append(obsid)
        else:
            print(f'{h5parmFile} - Std all residuals: {valstd} - Dyn range: {valdynrange}')
    
    allvalstd.append(valstd)
    allvaldynrange.append(valdynrange)
    alltimes.append(times[0])
    allids.append(obsid)
    h5.close()

pickle.dump(badids, open(badobs_picklefile, 'wb'))
pickle.dump([alltimes,allvalstd,allvaldynrange,allids], open('stats.pickle', 'wb'))
