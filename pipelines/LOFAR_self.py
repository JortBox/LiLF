#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# perform self-calibration on a group of SBs concatenated in TCs.
# they need to be in "./mss/"

import sys, os, glob
import numpy as np
from regions import Regions
import lsmtool
from astropy.coordinates import Angle

########################################################
from LiLF import lib_ms, lib_img, lib_util, lib_log, lib_dd
logger_obj = lib_log.Logger('pipeline-self')
logger = lib_log.logger
s = lib_util.Scheduler(log_dir = logger_obj.log_dir, dry = False)
w = lib_util.Walker('pipeline-self.walker')

parset = lib_util.getParset()
logger.info('Parset: '+str(dict(parset['LOFAR_self'])))
parset_dir = parset.get('LOFAR_self','parset_dir')
subfield_min_flux = parset.getfloat('LOFAR_self','subfield_min_flux') # default 40 Jy
subfield = parset.get('LOFAR_self','subfield') # possible to provide a ds9 box region customized sub-field. DEfault='' -> Automated detection using subfield_min_flux.
maxIter = parset.getint('LOFAR_self','maxIter') # default = 2 (try also 3)
phaseSolMode = parset.get('LOFAR_self', 'ph_sol_mode') # tecandphase, tec, phase
sourcedb = parset.get('model','sourcedb')
apparent = parset.getboolean('model','apparent')
userReg = parset.get('model','userReg')

#############################################################################

def clean_self(c, MSs, MSsClean, imagename, imgsizepix, cc_fit_order, userReg, subfield_kwargs, do_predict=True):
    """ Do normal (hres) imaging of the self-calibrated data. """
   
    maskname = imagename + '-mask.fits'
    imagenameM = imagename.replace('wide','wideM')
   
    lib_util.run_wsclean(s, 'wsclean-c' + str(c) + '.log', MSsClean.getStrWsclean(), concat_mss=True,
                         name=imagename, no_update_model_required='', size=imgsizepix, scale='4arcsec',
                         weight='briggs -0.3', niter=1000000, minuv_l=30,
                         parallel_gridding=6, parallel_deconvolution=1024, baseline_averaging='', mgain=0.8,
                         local_rms='', auto_mask=5, auto_threshold=4.,
                         join_channels='', fit_spectral_pol=cc_fit_order, channels_out=MSsClean.getChout(4.e6),
                         multiscale='', multiscale_scale_bias=0.6, deconvolution_channels=cc_fit_order, 
                         **subfield_kwargs)

    # masking
    if userReg != '':
            s.add('breizorro.py -t 6.5 -r %s -b 50 -o %s --merge %s' % (imagename+'-MFS-image.fits', maskname, userReg), 
                    log='makemask-'+str(c)+'.log', commandType='python' )
            s.run()        
    else:
            s.add('breizorro.py -t 6.5 -r %s -b 50 -o %s' % (imagename+'-MFS-image.fits', maskname), 
                    log='makemask-'+str(c)+'.log', commandType='python' )
            s.run()    
    # TEST: try to remove the second imaging and just reduce the models to masked regions

    lib_util.run_wsclean(s, 'wscleanM-c' + str(c) + '.log', MSsClean.getStrWsclean(), concat_mss=True,
                         name=imagenameM, no_update_model_required='', save_source_list='', size=imgsizepix, scale='4arcsec',
                         weight='briggs -0.3', niter=1000000, minuv_l=30,
                         parallel_gridding=6, parallel_deconvolution=1024, baseline_averaging='', mgain=0.8,
                         local_rms='', auto_mask=5, auto_threshold=3., fits_mask=maskname,
                         join_channels='', fit_spectral_pol=cc_fit_order, channels_out=MSsClean.getChout(4.e6),
                         multiscale='', multiscale_scale_bias=0.6, deconvolution_channels=cc_fit_order, 
                         **subfield_kwargs)

    os.system('cat ' + logger_obj.log_dir + '/wscleanM-c' + str(c) + '.log | grep "background noise"')

    # when wasclean allow station selection, then we can remove MSsClean and this predict can go in the previous call with do_predict=True
    if do_predict:
        logger.info('Predict model...')
        pred_str = f'wsclean -predict -padding 1.8 -name img/wideM-{c} -j {s.max_processors} -channels-out {MSs.getChout(4e6)}'
        if 'shift' in subfield_kwargs:
            pred_str += f' -shift {shift}'
        s.add(f'{pred_str} {MSs.getStrWsclean()}', log='wscleanPRE-c' + str(c) + '.log', commandType='wsclean', processors='max')
        s.run(check=True)

# Clear
with w.if_todo('cleaning'):
    logger.info('Cleaning...')
    lib_util.check_rm('img')
    os.makedirs('img')

    # here images, models, solutions for each group will be saved
    lib_util.check_rm('self')
    if not os.path.exists('self/images'): os.makedirs('self/images')
    if not os.path.exists('self/solutions'): os.makedirs('self/solutions')
    if not os.path.exists('self/plots'): os.makedirs('self/plots')
    if not os.path.exists('self/skymodel'): os.makedirs('self/skymodel')

### DONE

MSs = lib_ms.AllMSs( glob.glob('mss/TC*[0-9].MS'), s )

try:
    MSs.print_HAcov()
except:
    logger.error('Problem with HAcov, continue anyway.')

# make beam to the first mid null
phasecentre = MSs.getListObj()[0].getPhaseCentre()
MSs.getListObj()[0].makeBeamReg('self/beam.reg', freq='mid', to_null=True)
beamReg = 'self/beam.reg'

# set image size
imgsizepix_wide = int(2.1*MSs.getListObj()[0].getFWHM(freq='mid')*3600/4.)
imgsizepix_lr = int(5*MSs.getListObj()[0].getFWHM(freq='mid')*3600/30.)
imgsizepix_p = int(2.1*MSs.getListObj()[0].getFWHM(freq='mid')*3600/10.)

# set clean componet fit order (use 5 for large BW)
if MSs.getChout(4.e6) >= 7:  # Bandwidth of 28 MHz or more
    cc_fit_order = 5
else: cc_fit_order = 3

fullband = MSs.getBandwidth()
nchan = MSs.mssListObj[0].getNchan()
tint = MSs.mssListObj[0].getTimeInt()
if int(np.rint(fullband / nchan < 195.3e3/4)):
    base_nchan = int(np.rint((195.3e3/4)/(fullband/nchan))) # this is 1 for ducth observations, and larger (2,4) for IS observations
else: base_nchan = 1
if MSs.hasIS:
    base_solint = 1
elif tint < 4:
    base_solint = int(np.rint(4/tint)) # this is 2 for dutch SPARSE observations
else: base_solint = 1

#################################################################
# Get online model
if sourcedb == '':
    if not os.path.exists('tgts.skymodel'):
        fwhm = MSs.getListObj()[0].getFWHM(freq='min')
        radeg = phasecentre[0]
        decdeg = phasecentre[1]
        # get model the size of the image (radius=fwhm/2)
        os.system('wget -O tgts.skymodel "https://lcs165.lofar.eu/cgi-bin/gsmv1.cgi?coord=%f,%f&radius=%f&unit=deg"' % (radeg, decdeg, fwhm/2.)) # ASTRON
        lsm = lsmtool.load('tgts.skymodel', beamMS=MSs.getListStr()[0])#, beamMS=MSs.getListObj()[0])
        lsm.remove('I<1')
        lsm.write('tgts-beam.skymodel', applyBeam=True, clobber=True)
        lsm.write('tgts.skymodel', applyBeam=False, clobber=True)
        apparent = False
    sourcedb = 'tgts.skymodel'

#################################################################################################
# Add model to MODEL_DATA
# copy sourcedb into each MS to prevent concurrent access from multiprocessing to the sourcedb
sourcedb_basename = sourcedb.split('/')[-1]
for MS in MSs.getListStr():
    lib_util.check_rm(MS + '/' + sourcedb_basename)
    logger.debug('Copy: ' + sourcedb + ' -> ' + MS)
    os.system('cp -r ' + sourcedb + ' ' + MS)

# Here the model is added only to CS+RS, IS used only for FR and model is not needed
with w.if_todo('init_model'):
    logger.info('Add model to MODEL_DATA...')
    if apparent:
        MSs.run('DP3 ' + parset_dir + '/DP3-predict.parset msin=$pathMS pre.usebeammodel=false pre.sourcedb=$pathMS/' + sourcedb_basename,
            log='$nameMS_pre.log', commandType='DP3')
    else:
        MSs.run('DP3 ' + parset_dir + '/DP3-predict.parset msin=$pathMS pre.usebeammodel=true pre.usechannelfreq=True \
                 pre.beammode=array_factor pre.onebeamperpatch=True pre.sourcedb=$pathMS/' + sourcedb_basename,
                log='$nameMS_pre.log', commandType='DP3')
### DONE

with w.if_todo('solve_cor_fr'):
    logger.info('Add column CIRC_PHASEDIFF_DATA...')
    MSs.addcol('CIRC_PHASEDIFF_DATA', 'DATA', usedysco=False)  # No dysco here as off diag elements are 0 and dysco does not like 0s
    # Probably we do not need smoothing since we have long time intervals and smoothnessconstraint?

    logger.info('Converting to circular...')
    MSs.run('mslin2circ.py -s -i $pathMS:CIRC_PHASEDIFF_DATA -o $pathMS:CIRC_PHASEDIFF_DATA',
            log='$nameMS_lincirc.log', commandType='python', maxThreads=2)

    # Get circular phase diff CIRC_PHASEDIFF_DATA -> CIRC_PHASEDIFF_DATA
    logger.info('Get circular phase difference...')
    MSs.run('taql "UPDATE $pathMS SET\
         CIRC_PHASEDIFF_DATA[,0]=0.5*EXP(1.0i*(PHASE(CIRC_PHASEDIFF_DATA[,0])-PHASE(CIRC_PHASEDIFF_DATA[,3]))), \
         CIRC_PHASEDIFF_DATA[,3]=CIRC_PHASEDIFF_DATA[,0], \
         CIRC_PHASEDIFF_DATA[,1]=0+0i, \
         CIRC_PHASEDIFF_DATA[,2]=0+0i"', log='$nameMS_taql_phdiff.log', commandType='general')

    logger.info('Creating FR_MODEL_DATA...')  # take from MODEL_DATA but overwrite
    MSs.addcol('FR_MODEL_DATA', 'MODEL_DATA', usedysco=False)
    MSs.run('taql "UPDATE $pathMS SET FR_MODEL_DATA[,0]=0.5+0i, FR_MODEL_DATA[,1]=0.0+0i, FR_MODEL_DATA[,2]=0.0+0i, \
         FR_MODEL_DATA[,3]=0.5+0i"', log='$nameMS_taql_frmodel.log', commandType='general')

    # Solve cal_SB.MS:CIRC_PHASEDIFF_DATA against FR_MODEL_DATA (only solve - solint=2m nchan=0 as it has the smoothnessconstrain)
    logger.info('Solving circ phase difference ...')
    MSs.run('DP3 ' + parset_dir + '/DP3-solFR.parset msin=$pathMS sol.h5parm=$pathMS/fr.h5 sol.solint=' + str(30 * base_solint),
            log='$nameMS_solFR.log', commandType="DP3")
    lib_util.run_losoto(s, f'fr', [ms + '/fr.h5' for ms in MSs.getListStr()], [parset_dir + '/losoto-fr.parset'])
    os.system('mv cal-fr.h5 self/solutions/')
    os.system('mv plots-fr self/plots/')
    # Delete cols again to not waste space
    MSs.run('taql "ALTER TABLE $pathMS DELETE COLUMN CIRC_PHASEDIFF_DATA, FR_MODEL_DATA"',
            log='$nameMS_taql_delcol.log', commandType='general')

    # Correct FR with results of solve - group*_TC.MS:DATA -> group*_TC.MS:FR_CORRECTED_DATA
    logger.info('Correcting FR...')
    MSs.run('DP3 ' + parset_dir + '/DP3-cor.parset msin=$pathMS msin.datacolumn=DATA msout.datacolumn=FR_CORRECTED_DATA \
                cor.parmdb=self/solutions/cal-fr.h5 cor.correction=rotationmeasure000',
            log='$nameMS_corFR.log', commandType='DP3')
### DONE

#####################################################################################################
# Self-cal cycle
for c in range(maxIter):
    logger.info('Start selfcal cycle: '+str(c))
    if c == 0:
        with w.if_todo('set_corrected_data_c%02i' % c):
            logger.info('Creating CORRECTED_DATA from FR_CORRECTED_DATA...')
            MSs.addcol('CORRECTED_DATA', 'FR_CORRECTED_DATA')
    else:
        with w.if_todo('set_corrected_data_c%02i' % c):
            logger.info('Set CORRECTED_DATA = SUBFIELD_DATA...')
            MSs.run('taql "update $pathMS set CORRECTED_DATA = SUBFIELD_DATA"', log='$nameMS_taql-c' + str(c) + '.log',
                    commandType='general')
    ### DONE

    with w.if_todo('smooth_model_c%02i' % c):
        # Smooth MODEL_DATA -> MODEL_DATA
        MSs.run_Blsmooth('MODEL_DATA', 'MODEL_DATA', logstr=f'smooth-c{c}')
    ### DONE

    with w.if_todo('solve_tec1_c%02i' % c):
        # Smooth CORRECTED_DATA -> SMOOTHED_DATA
        MSs.run_Blsmooth('CORRECTED_DATA', logstr=f'smooth-c{c}')

        # solve ionosphere phase - ms:SMOOTHED_DATA (1m 2SB)
        logger.info('Solving TEC1...')
        if phaseSolMode == 'phase': #phase
            # TODO optimize smoothnessconstraint 2e6 -> 1e6
            solver_params = f'sol.mode=scalarcomplexgain sol.smoothnessconstraint=1e6 sol.smoothnessreffrequency=54e6'
            losoto_parsets = [parset_dir+'/losoto-plot-scalar.parset']
        else: # TEC or TecAndPhase
            solver_params = f'sol.mode={phaseSolMode} sol.approximatetec=True sol.maxapproxiter=250 sol.approxtolerance=1e-3'
            losoto_parsets = [parset_dir+'/losoto-plot-tec.parset']

        MSs.run(f"DP3 {parset_dir}/DP3-solTEC.parset msin=$pathMS sol.h5parm=$pathMS/tec1.h5 sol.solint={base_solint} {solver_params}",
                log='$nameMS_solTEC-c'+str(c)+'.log', commandType='DP3')
        # MSs.run('DP3 '+parset_dir+'/DP3-solTEC.parset msin=$pathMS sol.h5parm=$pathMS/tec1.h5 \
        #         msin.baseline="[CR]*&&;!RS208LBA;!RS210LBA;!RS307LBA;!RS310LBA;!RS406LBA;!RS407LBA;!RS409LBA;!RS508LBA;!RS509LBA;!PL*;!IE*;!UK*;!DE*;!FR*;!SE*" \
        #         sol.solint='+str(15*base_solint), \
        #         #+' sol.nchan='+str(8*base_nchan), sol.antennaconstraint=[[CS002LBA,CS003LBA,CS004LBA,CS005LBA,CS006LBA,CS007LBA]] \
        #         log='$nameMS_solTEC-c'+str(c)+'.log', commandType='DP3')

        lib_util.run_losoto(s, 'tec1-c'+str(c), [ms+'/tec1.h5' for ms in MSs.getListStr()], losoto_parsets)
        os.system('mv cal-tec1-c'+str(c)+'.h5 self/solutions/')
        os.system('mv plots-tec1-c'+str(c)+' self/plots/')
    ### DONE

    with w.if_todo('cor_tec1_c%02i' % c):
        # correct TEC - group*_TC.MS:CORRECTED_DATA -> group*_TC.MS:CORRECTED_DATA
        if phaseSolMode in ['tec', 'tecandphase']:
            logger.info('Correcting TEC1...')
            MSs.run('DP3 '+parset_dir+'/DP3-cor.parset msin=$pathMS msin.datacolumn=CORRECTED_DATA\
                    cor.parmdb=self/solutions/cal-tec1-c'+str(c)+'.h5 cor.correction=tec000',
                    log='$nameMS_corTEC-c'+str(c)+'.log', commandType='DP3')
        if phaseSolMode in ['phase', 'tecandphase']:
            logger.info('Correcting ph1...')
            MSs.run('DP3 '+parset_dir+'/DP3-cor.parset msin=$pathMS msin.datacolumn=CORRECTED_DATA\
                    cor.parmdb=self/solutions/cal-tec1-c'+str(c)+'.h5 cor.correction=phase000',
                    log='$nameMS_corTEC-c'+str(c)+'.log', commandType='DP3')
    ### DONE

    # with w.if_todo('solve_tec2_c%02i' % c):
    #     # Smooth CORRECTED_DATA -> SMOOTHED_DATA
    #     MSs.run_Blsmooth('CORRECTED_DATA', logstr=f'smooth-c{c}')
    #
    #     # solve TEC - ms:SMOOTHED_DATA (4s, 1SB)
    #     logger.info('Solving TEC2...')
    #     MSs.run('DP3 '+parset_dir+'/DP3-solTEC.parset msin=$pathMS sol.h5parm=$pathMS/tec2.h5 \
    #             sol.antennaconstraint=[[CS001LBA,CS002LBA,CS003LBA,CS004LBA,CS005LBA,CS006LBA,CS007LBA,CS011LBA,CS013LBA,CS017LBA,CS021LBA,CS024LBA,CS026LBA,CS028LBA,CS030LBA,CS031LBA,CS032LBA,CS101LBA,CS103LBA,CS201LBA,CS301LBA,CS302LBA,CS401LBA,CS501LBA,RS106LBA,RS205LBA,RS305LBA,RS306LBA,RS503LBA]] \
    #             sol.solint='+str(base_solint), \
    #             #+' sol.nchan='+str(4*base_nchan), \
    #             log='$nameMS_solTEC-c'+str(c)+'.log', commandType='DP3')
    #
    #     lib_util.run_losoto(s, 'tec2-c'+str(c), [ms+'/tec2.h5' for ms in MSs.getListStr()], [parset_dir+'/losoto-plot-tec.parset'])
    #     os.system('mv cal-tec2-c'+str(c)+'.h5 self/solutions/')
    #     os.system('mv plots-tec2-c'+str(c)+' self/plots/')
    # ### DONE
    #
    # with w.if_todo('cor_tec2_c%02i' % c):
    #     # correct TEC - group*_TC.MS:CORRECTED_DATA -> group*_TC.MS:CORRECTED_DATA
    #     logger.info('Correcting TEC2...')
    #     MSs.run('DP3 '+parset_dir+'/DP3-cor.parset msin=$pathMS msin.datacolumn=CORRECTED_DATA\
    #             cor.parmdb=self/solutions/cal-tec2-c'+str(c)+'.h5 cor.correction=tec000',
    #             log='$nameMS_corTEC-c'+str(c)+'.log', commandType='DP3')
    #     MSs.run('DP3 '+parset_dir+'/DP3-cor.parset msin=$pathMS msin.datacolumn=CORRECTED_DATA\
    #             cor.parmdb=self/solutions/cal-tec2-c'+str(c)+'.h5 cor.correction=phase000',
    #             log='$nameMS_corTEC-c'+str(c)+'.log', commandType='DP3')
    # ### DONE

    # AMP DIE correction in last iteration
    #if c == maxIter-1:
    #    with w.if_todo('solve_g_c%02i' % c):
    #        # DIE Calibration - ms:CORRECTED_DATA (8m, 4SB)
    #        logger.info('Solving slow G (full jones)...')
    #        MSs.run('DP3 '+parset_dir+'/DP3-solGfj.parset msin=$pathMS sol.h5parm=$pathMS/g.h5 sol.solint='+str(120*base_solint)+' sol.nchan='+str(16*base_nchan),
    #                log='$nameMS_solG-c'+str(c)+'.log', commandType='DP3')
    #        lib_util.run_losoto(s, 'g-c'+str(c), [MS+'/g.h5' for MS in MSs.getListStr()],
    #                [parset_dir+'/losoto-plot-fullj.parset', parset_dir+'/losoto-bp.parset'])
    #        os.system('mv plots-g-c'+str(c)+' self/plots/')
    #        os.system('mv cal-g-c'+str(c)+'.h5 self/solutions/')
    #    ### DONE
    #
    #    with w.if_todo('cor_g_c%02i' % c):
    #        # correct G - group*_TC.MS:CORRECTED_DATA -> group*_TC.MS:CORRECTED_DATA
    #        logger.info('Correcting G...')
    #        MSs.run('DP3 '+parset_dir+'/DP3-cor.parset msin=$pathMS msin.datacolumn=CORRECTED_DATA \
    #                cor.parmdb=self/solutions/cal-g-c'+str(c)+'.h5 cor.correction=fulljones cor.soltab=[amplitudeSmooth,phase000]',
    #                log='$nameMS_corG-c'+str(c)+'.log', commandType='DP3')
    #    ### DONE

    ###################################################################################################################
    # clean on concat.MS:CORRECTED_DATA

    # if IS are present, copy the MS and split a dataset with just CS+RS
    if MSs.hasIS:
        logger.info('Splitting out international stations...')
        lib_util.check_rm('mss-noIS')
        os.system('mkdir mss-noIS')
        MSs.run('DP3 msin=$pathMS msin.datacolumn=CORRECTED_DATA msin.baseline="[CR]S*&" msout=mss-noIS/$nameMS.MS steps=[]',
                 log='$nameMS_splitDutch.log', commandType="DP3")
        MSsClean = lib_ms.AllMSs( glob.glob('mss-noIS/TC*[0-9].MS'), s )
    else:
        MSsClean = MSs

    imagename = 'img/wide-'+str(c)
    if (c == 0) or (c == maxIter - 1): # make wide image
        imgsizepix = imgsizepix_wide
        subfield_kwargs = {}
    else: # make small image (field subtracted)
        imgsizepix = int(1.3 * field_size * 3600 / 4.)
        img_ra, img_dec = Angle(f'{field_center[0]}deg').hms, Angle(f'{field_center[1]}deg').dms
        shift = f'{int(img_ra.h)}h{int(img_ra.m)}m{img_ra.s:.4f}s {int(img_dec.d)}d{int(img_dec.m)}m{img_dec.s:.4f}s'
        subfield_kwargs = {'shift': shift}

    if c < maxIter - 1:
        with w.if_todo('imaging_c%02i' % c):
            clean_self(c, MSs, MSsClean, imagename, imgsizepix, cc_fit_order, userReg, subfield_kwargs)
        ### DONE

    if c == 0:
        with w.if_todo('lowres_setdata_c%02i' % c):
            # Subtract model from all TCs - ms:CORRECTED_DATA - MODEL_DATA -> ms:CORRECTED_DATA (selfcal corrected, beam corrected, high-res model subtracted)
            logger.info('Subtracting high-res model (CORRECTED_DATA = CORRECTED_DATA - MODEL_DATA)...')
            MSs.run('taql "update $pathMS set CORRECTED_DATA = CORRECTED_DATA - MODEL_DATA"', log='$nameMS_taql-c'+str(c)+'.log', commandType='general')
        ### DONE
    
        with w.if_todo('lowres_img_c%02i' % c):
            # Making beam mask
            logger.info('Preparing mask for low-res clean...')
            lib_util.run_wsclean(s, 'wscleanLRmask.log', MSs.getStrWsclean(), name='img/tmp', size=imgsizepix_lr, scale='30arcsec')
            os.system('mv img/tmp-image.fits img/wide-lr-mask.fits')
            lib_img.blank_image_reg('img/wide-lr-mask.fits', beamReg, blankval = 0.)
            lib_img.blank_image_reg('img/wide-lr-mask.fits', beamReg, blankval = 1., inverse=True)
    
            # reclean low-resolution
            logger.info('Cleaning low-res...')
            imagename_lr = 'img/wide-lr'
            lib_util.run_wsclean(s, 'wscleanLR.log', MSs.getStrWsclean(), name=imagename_lr, do_predict=True,
                    parallel_gridding=4, temp_dir='./', size=imgsizepix_lr, scale='30arcsec',
                    weight='briggs -0.3', niter=50000, no_update_model_required='', minuv_l=30, maxuvw_m=6000,
                    taper_gaussian='200arcsec', mgain=0.85, parallel_deconvolution=512, baseline_averaging='',
                    local_rms='', auto_mask=3, auto_threshold=1.5, fits_mask='img/wide-lr-mask.fits',
                    join_channels='', channels_out=MSs.getChout(2.e6))
            # Test of we cam just do a do_predict
            # s.add('wsclean -predict -padding 1.8 -name '+imagename_lr+' -j '+str(s.max_processors)+' -channels-out '+str(MSs.getChout(2e6))+' '+MSs.getStrWsclean(), \
            #       log='wscleanLR-PRE-c'+str(c)+'.log', commandType='wsclean', processors='max')
            # s.run(check=True)
        ### DONE

        with w.if_todo('lowres_sub_c%02i' % c):
            # Subtract low-res model - CORRECTED_DATA = CORRECTED_DATA - MODEL_DATA
            logger.info('Subtracting low-res model (CORRECTED_DATA = CORRECTED_DATA - MODEL_DATA)...')
            MSs.run('taql "update $pathMS set CORRECTED_DATA = CORRECTED_DATA - MODEL_DATA"', log='$nameMS_taql-c'+str(c)+'.log', commandType='general')
        ### DONE

        with w.if_todo('lowres_lsimg_c%02i' % c):
            logger.info('Cleaning large-scale...')
            imagename_ls = 'img/wide-largescale'
            #                     intervals_out=len(MSs.mssListObj)*4,
            #use_idg = '', aterm_kernel_size = 16, aterm_config = parset_dir + '/aconfig.txt',
            lib_util.run_wsclean(s, 'wscleanLS.log', MSs.getStrWsclean(), name=imagename_ls, do_predict=False,
                                 temp_dir='./', size=2000, scale='20arcsec',
                                 no_fit_beam='', circular_beam='', beam_size='200arcsec',
                                 multiscale='', multiscale_scales='0,4,8,16,32,64',
                                 weight='briggs -0.3', niter=10000, no_update_model_required='', minuv_l=20,
                                 maxuvw_m=5000, taper_gaussian='200arcsec', mgain=0.85,
                                 parallel_deconvolution=512, baseline_averaging='', local_rms='', auto_mask=1.5,
                                 auto_threshold=0.5, join_channels='', channels_out=MSs.getChout(4.e6))
        ### DONE

        with w.if_todo('lowres_corrupt_c%02i' % c):    
            # corrupt model with TEC+FR+Beam2ord solutions - ms:MODEL_DATA -> ms:MODEL_DATA
            if phaseSolMode in ['tec', 'tecandphase']:
                logger.info('Corrupt low-res model: TEC1...')
                MSs.run('DP3 '+parset_dir+'/DP3-cor.parset msin=$pathMS msin.datacolumn=MODEL_DATA msout.datacolumn=MODEL_DATA  \
                        cor.parmdb=self/solutions/cal-tec1-c'+str(c)+'.h5 cor.correction=tec000 cor.invert=False',
                        log='$nameMS_corrupt.log', commandType='DP3')
            if phaseSolMode in ['phase', 'tecandphase']:
                logger.info('Corrupt low-res model: ph1...')
                MSs.run('DP3 '+parset_dir+'/DP3-cor.parset msin=$pathMS msin.datacolumn=MODEL_DATA msout.datacolumn=MODEL_DATA  \
                    cor.parmdb=self/solutions/cal-tec1-c'+str(c)+'.h5 cor.correction=phase000 cor.invert=False',
                    log='$nameMS_corrupt.log', commandType='DP3')
            # logger.info('Corrupt low-res model: TEC+Ph 2...')
            # MSs.run('DP3 '+parset_dir+'/DP3-cor.parset msin=$pathMS msin.datacolumn=MODEL_DATA msout.datacolumn=MODEL_DATA  \
            #         cor.parmdb=self/solutions/cal-tec2-c'+str(c)+'.h5 cor.correction=tec000 cor.invert=False',
            #         log='$nameMS_corrupt.log', commandType='DP3')
            # MSs.run('DP3 '+parset_dir+'/DP3-cor.parset msin=$pathMS msin.datacolumn=MODEL_DATA msout.datacolumn=MODEL_DATA  \
            #         cor.parmdb=self/solutions/cal-tec2-c'+str(c)+'.h5 cor.correction=phase000 cor.invert=False',
            #         log='$nameMS_corrupt.log', commandType='DP3')

        with w.if_todo('lowres_subtract_c%02i' % c):
            # Permanently subtract low-res sidelobe model - FR_CORRECTED_DATA = FR_CORRECTED_DATA - MODEL_DATA.
            logger.info('Subtracting low-res sidelobe model (FR_CORRECTED_DATA = FR_CORRECTED_DATA - MODEL_DATA)...')
            MSs.run('taql "update $pathMS set FR_CORRECTED_DATA = FR_CORRECTED_DATA - MODEL_DATA"',
                    log='$nameMS_taql-c'+str(c)+'.log', commandType='general')
        ### DONE

        # Prepare region and models for subfield
        if subfield:
            subfield_path = subfield
            if len(Regions.read(subfield_path)) > 1:
                raise ValueError(f'Manual subfield region {subfield} contains more than one region')
        else:
            subfield_path = 'self/skymodel/subfield.reg'

        with w.if_todo('extreg_preapre_c%02i' % c):
            if not subfield: # automatically find subfield
                sm = lsmtool.load(f'img/wideM-{c}-sources.txt')
                sm.remove('img/wide-lr-mask.fits=1')  # remove sidelobe sources that were subtracted
                sm.remove('MajorAxis > 80')  # remove largest scales
                field_center1, field_size1 = lib_dd.make_subfield_region(subfield_path, MSs.getListObj()[0], sm,
                                                                         subfield_min_flux, debug_dir='img/')
            # prepare model of central/external regions
            logger.info('Blanking central region of model files and reverse...')
            for im in glob.glob('img/wideM-0*model.fits'):
                wideMint = im.replace('wideM','wideMint')
                os.system('cp %s %s' % (im, wideMint))
                lib_img.blank_image_reg(wideMint, subfield_path, blankval = 0., inverse=True)
                wideMext = im.replace('wideM','wideMext')
                os.system('cp %s %s' % (im, wideMext))
                lib_img.blank_image_reg(wideMext, subfield_path, blankval = 0.)
        # DONE
        subfield_reg = Regions.read(subfield_path)[0]
        field_center = subfield_reg.center.ra.deg, subfield_reg.center.dec.deg
        field_size = np.max([subfield_reg.width.to_value('deg'), subfield_reg.height.to_value('deg')])

        with w.if_todo('extreg_predict_corrupt_subtract_c%02i' % c):
            # Recreate MODEL_DATA of external region for subtraction
            logger.info('Predict model of external region...')
            s.add('wsclean -predict -padding 1.8 -name img/wideMext-'+str(c)+' -j '+str(s.max_processors)+' -channels-out '+str(MSs.getChout(4e6))+' '+MSs.getStrWsclean(), \
                  log='wscleanPRE-c'+str(c)+'.log', commandType='wsclean', processors='max')
            s.run(check=True)

            # corrupt model with TEC+FR+Beam2ord solutions - ms:MODEL_DATA -> ms:MODEL_DATA
            if phaseSolMode in ['tec', 'tecandphase']:
                logger.info('Corrupt low-res model: TEC1...')
                MSs.run('DP3 '+parset_dir+'/DP3-cor.parset msin=$pathMS msin.datacolumn=MODEL_DATA msout.datacolumn=MODEL_DATA  \
                        cor.parmdb=self/solutions/cal-tec1-c'+str(c)+'.h5 cor.correction=tec000 cor.invert=False',
                        log='$nameMS_corrupt.log', commandType='DP3')
            if phaseSolMode in ['phase', 'tecandphase']:
                logger.info('Corrupt low-res model: ph1...')
                MSs.run('DP3 '+parset_dir+'/DP3-cor.parset msin=$pathMS msin.datacolumn=MODEL_DATA msout.datacolumn=MODEL_DATA  \
                        cor.parmdb=self/solutions/cal-tec1-c'+str(c)+'.h5 cor.correction=phase000 cor.invert=False',
                        log='$nameMS_corrupt.log', commandType='DP3')
            # logger.info('Corrupt low-res model: TEC+Ph 2...')
            # MSs.run('DP3 '+parset_dir+'/DP3-cor.parset msin=$pathMS msin.datacolumn=MODEL_DATA msout.datacolumn=MODEL_DATA  \
            #         cor.parmdb=self/solutions/cal-tec2-c'+str(c)+'.h5 cor.correction=tec000 cor.invert=False',
            #         log='$nameMS_corrupt.log', commandType='DP3')
            # MSs.run('DP3 '+parset_dir+'/DP3-cor.parset msin=$pathMS msin.datacolumn=MODEL_DATA msout.datacolumn=MODEL_DATA  \
            #         cor.parmdb=self/solutions/cal-tec2-c'+str(c)+'.h5 cor.correction=phase000 cor.invert=False',
            #         log='$nameMS_corrupt.log', commandType='DP3')
            # logger.info('Corrupt low-res model: G...')
            # MSs.run('DP3 '+parset_dir+'/DP3-cor.parset msin=$pathMS msin.datacolumn=MODEL_DATA msout.datacolumn=MODEL_DATA \
            #         cor.parmdb=self/solutions/cal-g-c'+str(c)+'.h5 cor.correction=amplitudeSmooth cor.invert=False',
            #         log='$nameMS_corrupt.log', commandType='DP3')

            # subtract external region from FR_CORRECTED_DATA (sidelobe subtracted) to create SUBFIELD_DATA
            MSs.addcol('SUBFIELD_DATA','FR_CORRECTED_DATA')
            logger.info('Subtracting external region model (SUBFIELD_DATA = FR_CORRECTED_DATA - MODEL_DATA)...')
            MSs.run('taql "update $pathMS set SUBFIELD_DATA = FR_CORRECTED_DATA - MODEL_DATA"', log='$nameMS_taql-c'+str(c)+'.log', commandType='general')
        ### DONE
        
        with w.if_todo('flag_c%02i' % c):
            # Flag on residuals (SUBFIELD_DATA)
            logger.info('Flagging residuals...')
            MSs.run('DP3 '+parset_dir+'/DP3-flag.parset msin=$pathMS msin.datacolumn=SUBFIELD_DATA aoflagger.strategy='+parset_dir+'/LBAdefaultwideband.lua',
                    log='$nameMS_flag-c'+str(c)+'.log', commandType='DP3')
        ### DONE

        with w.if_todo('centralreg_predict_c%02i' % c):
            # Recreate MODEL_DATA of internal region for next calibration cycle
            logger.info('Predict model of internal region...')
            s.add('wsclean -predict -padding 1.8 -name img/wideMint-'+str(c)+' -j '+str(s.max_processors)+' -channels-out '+str(MSs.getChout(4e6))+' '+MSs.getStrWsclean(), \
                   log='wscleanPRE-c'+str(c)+'.log', commandType='wsclean', processors='max')
            s.run(check=True)
        ### DONE

with w.if_todo('final_correct'):
    # correct model with TEC+Beam2ord solutions - ms:FR_CORRECTED_DATA -> ms:CORRECTED_DATA
    #logger.info('Correcting G...')
    #MSs.run(f'DP3 {parset_dir}/DP3-cor.parset msin=$pathMS msin.datacolumn=FR_CORRECTED_DATA msout.datacolumn=CORRECTED_DATA \
    #        cor.parmdb=self/solutions/cal-g-c{c}.h5 cor.correction=fulljones cor.soltab=[amplitudeSmooth,phase000]',
    #        log='$nameMS_finalcor.log', commandType='DP3')
    MSs.run('taql "update $pathMS set CORRECTED_DATA = FR_CORRECTED_DATA"', log='$nameMS_taql-c'+str(c)+'.log', commandType='general')
    if phaseSolMode in ['tec', 'tecandphase']:
        logger.info('Correcting TEC...')
        MSs.run(f'DP3 {parset_dir}/DP3-cor.parset msin=$pathMS msin.datacolumn=CORRECTED_DATA msout.datacolumn=CORRECTED_DATA  \
            cor.parmdb=self/solutions/cal-tec1-c{c}.h5 cor.correction=tec000',
            log='$nameMS_finalcor.log', commandType='DP3')
    if phaseSolMode in ['phase', 'tecandphase']:
        logger.info('Correcting ph...')
        MSs.run(f'DP3 {parset_dir}/DP3-cor.parset msin=$pathMS msin.datacolumn=CORRECTED_DATA msout.datacolumn=CORRECTED_DATA  \
            cor.parmdb=self/solutions/cal-tec1-c{c}.h5 cor.correction=phase000',
            log='$nameMS_finalcor.log', commandType='DP3')
    # logger.info('Correct low-res model: TEC+Ph 2...')
    # MSs.run(f'DP3 {parset_dir}/DP3-cor.parset msin=$pathMS msin.datacolumn=CORRECTED_DATA msout.datacolumn=CORRECTED_DATA  \
    #         cor.parmdb=self/solutions/cal-tec2-c{c}.h5 cor.correction=tec000',
    #         log='$nameMS_finalcor.log', commandType='DP3')
    # MSs.run(f'DP3 {parset_dir}/DP3-cor.parset msin=$pathMS msin.datacolumn=CORRECTED_DATA msout.datacolumn=CORRECTED_DATA  \
    #         cor.parmdb=self/solutions/cal-tec2-c{c}.h5 cor.correction=phase000',
    #         log='$nameMS_finalcor.log', commandType='DP3')
### DONE

with w.if_todo('imaging-final'):
    clean_self(c, MSs, MSsClean, imagename, imgsizepix, cc_fit_order, userReg, subfield_kwargs, do_predict=False)

# polarisation imaging
with w.if_todo('imaging-pol'):
    logger.info('Cleaning (Pol)...')
    imagenameP = 'img/wideP'
    lib_util.run_wsclean(s, 'wscleanP.log', MSs.getStrWsclean(), name=imagenameP, pol='QUV',
        size=imgsizepix_p, scale='10arcsec', weight='briggs -0.3', niter=0, no_update_model_required='',
        parallel_gridding=2, baseline_averaging='', minuv_l=30, maxuv_l=4500,
        join_channels='', channels_out=MSs.getChout(4.e6))

MSs.run('taql "ALTER TABLE $pathMS DELETE COLUMN SUBFIELD_DATA, FR_CORRECTED_DATA"',
        log='$nameMS_taql_delcol.log', commandType='general')

# Copy images
[ os.system('mv img/wideM-'+str(c)+'-MFS-image*.fits self/images') for c in range(maxIter) ]
[ os.system('mv img/wideM-'+str(c)+'-MFS-residual*.fits self/images') for c in range(maxIter) ]
[ os.system('mv img/wideM-'+str(c)+'-sources*.txt self/images') for c in range(maxIter) ]
os.system('mv img/wideP-MFS-*-image.fits self/images')
os.system('mv img/wide-lr-MFS-image.fits self/images')
os.system('mv img/wide-largescale-MFS-image.fits self/images')

# Copy model
os.system(f'mv img/wideM-{maxIter-1}-*-model.fits self/skymodel')

logger.info("Done.")
