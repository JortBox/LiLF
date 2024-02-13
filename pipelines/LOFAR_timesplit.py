#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Data preparation for selfcal, apply cal solutions
# and split SB in time and concatenate in frequency.

import sys, os, glob, re
import numpy as np
from astropy.time import Time
import casacore.tables as pt

########################################################
LOCATION = "/net/voorrijn/data2/boxelaar/scripts/LiLF_dev"
sys.path.append(LOCATION)

from LiLF_lib import lib_ms, lib_util, lib_log
logger_obj = lib_log.Logger('pipeline-timesplit')
logger = lib_log.logger
s = lib_util.Scheduler(log_dir = logger_obj.log_dir, dry = False)
w = lib_util.Walker('pipeline-timesplit.walker')

# parse parset
parset = lib_util.getParset()
logger.info('Parset: '+str(dict(parset['LOFAR_timesplit'])))
parset_dir = parset.get('LOFAR_timesplit','parset_dir')
data_dir = parset.get('LOFAR_timesplit','data_dir')
copy_dir = parset.get('LOFAR_timesplit','copy_dir')
cal_dir = parset.get('LOFAR_timesplit','cal_dir')
ngroups = parset.getint('LOFAR_timesplit','ngroups')
initc = parset.getint('LOFAR_timesplit','initc') # initial tc num (useful for multiple observation of same target)
bl2flag = parset.get('flag','stations')

#################################################

# Clean
with w.if_todo('clean'):
    logger.info('Cleaning...')
    lib_util.check_rm('mss*')
### DONE

with w.if_todo('copy'):
    MSs = lib_ms.AllMSs( glob.glob(data_dir+'/*MS'), s )

    logger.info('Copy data...')
    for MS in MSs.getListObj():
        # if min(MS.getFreqs()) > 30.e6:
        # overwrite=True to prevent updating the weights twice
        MS.move(copy_dir+MS.nameMS+'.MS', keepOrig=True, overwrite=True)
### DONE

MSs = lib_ms.AllMSs( glob.glob(copy_dir+'/*MS'), s )

##################################################
# Find solutions to apply
if cal_dir == '':
    obsid = MSs.getListObj()[0].getObsID()
    # try standard location
    cal_dir = glob.glob('../id%i_-_*3[c|C]196' % obsid)+glob.glob('../id%i_-_*3[c|C]295' % obsid)+glob.glob('../id%i_-_*3[c|C]380' % obsid)
    if len(cal_dir) > 0:
        cal_dir = cal_dir[0]
    else:
        logger.error('Cannot find solutions.')
        sys.exit()
else:
    cal_dir = './'+cal_dir

logger.info('Calibrator directory: %s' % cal_dir)
h5_pa = cal_dir+'/cal-pa.h5'
h5_amp = cal_dir+'/cal-amp.h5'
h5_iono = cal_dir+'/cal-iono.h5'
if not os.path.exists(h5_pa) or not os.path.exists(h5_amp) or not os.path.exists(h5_iono):
    logger.error("Missing solutions in %s" % cal_dir)
    sys.exit()

####################################################
# Correct fist for BP(diag)+TEC+Clock and then for beam
with w.if_todo('apply'):

    # Apply cal sol - SB.MS:DATA -> SB.MS:CORRECTED_DATA (polalign corrected)
    logger.info('Apply solutions (pa)...')
    MSs.run('DP3 '+parset_dir+'/DP3-cor.parset msin=$pathMS \
            cor.parmdb='+h5_pa+' cor.correction=polalign', log='$nameMS_cor1.log', commandType='DP3')

    # Apply cal sol - SB.MS:CORRECTED_DATA -> SB.MS:CORRECTED_DATA (polalign corrected, calibrator corrected+reweight, beam corrected+reweight)
    logger.info('Apply solutions (amp/ph)...')
    if MSs.isLBA:
        MSs.run('DP3 '+parset_dir+'/DP3-cor.parset msin=$pathMS msin.datacolumn=CORRECTED_DATA cor.steps=[amp,ph] \
                cor.amp.parmdb='+h5_amp+' cor.amp.correction=amplitudeSmooth cor.amp.updateweights=True\
                cor.ph.parmdb='+h5_iono+' cor.ph.correction=phaseOrig000', log='$nameMS_cor2.log', commandType='DP3')
    elif MSs.isHBA:
        MSs.run('DP3 '+parset_dir+'/DP3-cor.parset msin=$pathMS msin.datacolumn=CORRECTED_DATA cor.steps=[amp,clock] \
                cor.amp.parmdb='+h5_amp+' cor.amp.correction=amplitudeSmooth cor.amp.updateweights=True\
                cor.clock.parmdb='+h5_iono+' cor.clock.correction=clockMed000', log='$nameMS_cor2.log', commandType='DP3')

    # Beam correction CORRECTED_DATA -> CORRECTED_DATA (polalign corrected, beam corrected+reweight)
    logger.info('Beam correction...')
    MSs.run('DP3 '+parset_dir+'/DP3-beam.parset msin=$pathMS corrbeam.updateweights=True', log='$nameMS_beam.log', commandType='DP3')
### DONE

###################################################################################################
# Create groups
groupnames = []
logger.info('Concatenating in frequency...')

times = set()
for measurement in glob.glob(copy_dir+'/*MS'):
    if not measurement.endswith(".MS"):
        continue
    times.add(measurement.split("_")[-2])
ngroups = len(times)

for i, msg in enumerate(np.array_split(sorted(glob.glob(copy_dir+'*MS')), ngroups)):
    if ngroups == 1:
        groupname = 'mss'
    else:
        groupname = 'mss-%02i' % i
    groupnames.append(groupname)

    # skip if already done
    if not os.path.exists('mss'): 
        os.makedirs('mss')
       
    # add missing SB with a fake name not to leave frequency holes
    min_nu = pt.table(MSs.getListStr()[0]).OBSERVATION[0]['LOFAR_OBSERVATION_FREQUENCY_MIN']
    max_nu = pt.table(MSs.getListStr()[0]).OBSERVATION[0]['LOFAR_OBSERVATION_FREQUENCY_MAX']
    num_init = lib_util.lofar_nu2num(min_nu)+1  # +1 because FREQ_MIN/MAX somewhat have the lowest edge of the SB freq
    num_fin = lib_util.lofar_nu2num(max_nu)+1
    prefix = re.sub('SB[0-9]*.MS','',msg[0])
    msg = []
    for j in range(num_init, num_fin+1):
        msg.append(prefix+'SB%03i.MS' % j)

    # check that nchan is divisible by 48 - necessary in dd pipeline; discard high freq unused channels
    nchan_init = MSs.getListObj()[0].getNchan()*len(msg)
    nchan = nchan_init - nchan_init % 48
    logger.info('Reducing total channels: %ich -> %ich)' % (nchan_init, nchan))

    # prepare concatenated mss - SB.MS:CORRECTED_DATA -> group#.MS:DATA (cal corr data, beam corrected)
    s.add('DP3 '+parset_dir+'/DP3-concat.parset msin="['+','.join(msg)+']" msin.nchan='+str(nchan)+'  msout=mss/'+groupname+'.MS', \
                log=groupname+'_DP3_concat.log', commandType='DP3')
    s.run(check=True)

MSs = lib_ms.AllMSs( glob.glob('mss*/*MS'), s )

#############################################################
# Flagging on concatenated dataset - also flag low-elevation
with w.if_todo('flag'):
    logger.info('Flagging...')
    flag_strat = '/HBAdefaultwideband.lua' if MSs.isHBA else '/LBAdefaultwideband.lua'
    MSs.run('DP3 '+parset_dir+'/DP3-flag.parset msin=$pathMS ant.baseline=\"' + bl2flag + '\" \
            aoflagger.strategy='+parset_dir+flag_strat,
            log='$nameMS_DP3_flag.log', commandType='DP3')

    logger.info('Remove bad timestamps...')
    MSs.run( 'flagonmindata.py -f 0.5 $pathMS', log='$nameMS_flagonmindata.log', commandType='python')

    logger.info('Plot weights...')
    MSs.run('reweight.py $pathMS -v -p -a %s' % (MSs.getListObj()[0].getAntennas()[0]),
            log='$nameMS_weights.log', commandType='python')
    lib_util.check_rm('plots-weights')
    os.system('mkdir plots-weights; mv *png plots-weights')
### DONE

#####################################
# Create time-chunks
with w.if_todo('timesplit'):

    logger.info('Splitting in time...')
    tc = initc
    for groupname in groupnames:
        ms = 'mss/'+groupname+'.MS'
        if not os.path.exists(ms): continue
        t = pt.table(ms, ack=False)
        starttime = t[0]['TIME']
        endtime   = t[t.nrows()-1]['TIME']
        hours = (endtime-starttime)/3600.
        logger.debug(ms+' has length of '+str(hours)+' h.')

        for timerange in np.array_split(sorted(set(t.getcol('TIME'))), round(hours)):
            logger.info('%02i - Splitting timerange %f %f' % (tc, timerange[0], timerange[-1]))
            t1 = t.query('TIME >= ' + str(timerange[0]) + ' && TIME <= ' + str(timerange[-1]), sortlist='TIME,ANTENNA1,ANTENNA2')
            splitms = 'mss/TC%02i.MS' % tc
            lib_util.check_rm(splitms)
            t1.copy(splitms, True)
            t1.close()
            tc += 1
        t.close()

        lib_util.check_rm(ms) # remove not-timesplitted file
### DONE

logger.info('Cleaning up...')
os.system('rm -r '+copy_dir)

logger.info("Done.")
