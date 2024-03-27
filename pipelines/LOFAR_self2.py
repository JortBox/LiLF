#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# perform self-calibration on a group of SBs concatenated in TCs.
# they need to be in "./mss/"

import sys, os, glob, re
import numpy as np
import casacore.tables as pt
import lsmtool

########################################################
from LiLF import lib_ms, lib_img, lib_util, lib_log, lib_dd
logger_obj = lib_log.Logger('pipeline-self')
logger = lib_log.logger
s = lib_util.Scheduler(log_dir = logger_obj.log_dir, dry = False)
w = lib_util.Walker('pipeline-self.walker')

parset = lib_util.getParset()
logger.info('Parset: '+str(dict(parset['LOFAR_self'])))
parset_dir = parset.get('LOFAR_self','parset_dir')
subfield_min_flux = parset.get('LOFAR_self','subfield_min_flux')
maxIter = parset.getint('LOFAR_self','maxIter')
sourcedb = parset.get('model','sourcedb')
apparent = parset.getboolean('model','apparent')
userReg = parset.get('model','userReg')
subfield_min_flux = 50.

#############################################################################
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

# make beam to the first mid null and central 1deg region
phasecentre = MSs.getListObj()[0].getPhaseCentre()
MSs.getListObj()[0].makeBeamReg('self/beam.reg', freq='mid', to_null=True)
beamReg = 'self/beam.reg'

# set image size
imgsizepix_wide = int(2.1*MSs.getListObj()[0].getFWHM(freq='mid')*3600/10.)
if imgsizepix_wide%2 != 0: imgsizepix_wide += 1 # prevent odd img sizes

imgsizepix = imgsizepix_wide # iteration 0 - start with wide.
imgcenter = phasecentre

# set clean componet fit order (use 5 for large BW)
bandwidth = MSs.getBandwidth()
if bandwidth > 25e6: cc_fit_order = 5
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
    # NOTE: do not add MODEL_DATA or the beam is transported from DATA, while we want it without beam applied

    logger.info('Add model to MODEL_DATA...')
    if apparent:
        MSs.run(
            'DP3 ' + parset_dir + '/DP3-predict.parset msin=$pathMS pre.usebeammodel=false pre.sourcedb=$pathMS/' + sourcedb_basename,
            log='$nameMS_pre.log', commandType='DP3')
    else:
        MSs.run('DP3 ' + parset_dir + '/DP3-predict.parset msin=$pathMS pre.usebeammodel=true pre.usechannelfreq=True \
                 pre.beammode=array_factor pre.onebeamperpatch=True pre.sourcedb=$pathMS/' + sourcedb_basename,
                 log='$nameMS_pre.log', commandType='DP3')
### DONE
#####################################################################################################
# Self-cal cycle
for c in range(maxIter):
    logger.info('Start selfcal cycle: '+str(c))
    if c == 0:
        with w.if_todo('set_corrected_data_c%02i' % c):
            logger.info('Creating CORRECTED_DATA...')
            MSs.addcol('CORRECTED_DATA', 'DATA')
            # logger.info('Set CORRECTED_DATA = DATA...')
            # MSs.run('taql "update $pathMS set CORRECTED_DATA = DATA"', log='$nameMS_taql-c'+str(c)+'.log', commandType='general')
    else:
        with w.if_todo('set_corrected_data_c%02i' % c):
            logger.info('Set CORRECTED_DATA = SUBFIELD_DATA...')
            MSs.run('taql "update $pathMS set CORRECTED_DATA = SUBFIELD_DATA"', log='$nameMS_taql-c' + str(c) + '.log',
                    commandType='general')
    ### DONE

    with w.if_todo('solve_tec1_c%02i' % c):
        # Smooth CORRECTED_DATA -> SMOOTHED_DATA
        logger.info('BL-based smoothing...')
        MSs.run('BLsmooth.py -c 8 -n 8 -r -i CORRECTED_DATA -o SMOOTHED_DATA $pathMS', log='$nameMS_smooth-c'+str(c)+'.log', commandType='python')
        MSs.run('BLsmooth.py -c 8 -n 8 -r -i MODEL_DATA -o MODEL_DATA $pathMS', log='$nameMS_smooth-c'+str(c)+'.log', commandType='python')

        # solve TEC - ms:SMOOTHED_DATA (1m 2SB)
        logger.info('Solving TEC1...')
        MSs.run('DP3 '+parset_dir+'/DP3-solTEC.parset msin=$pathMS sol.h5parm=$pathMS/tec1.h5 \
                msin.baseline="[CR]*&&;!RS208LBA;!RS210LBA;!RS307LBA;!RS310LBA;!RS406LBA;!RS407LBA;!RS409LBA;!RS508LBA;!RS509LBA;!PL*;!IE*;!UK*;!DE*;!FR*;!SE*" \
                sol.solint='+str(15*base_solint), \
                #+' sol.nchan='+str(8*base_nchan), sol.antennaconstraint=[[CS002LBA,CS003LBA,CS004LBA,CS005LBA,CS006LBA,CS007LBA]] \
                log='$nameMS_solTEC-c'+str(c)+'.log', commandType='DP3')

        lib_util.run_losoto(s, 'tec1-c'+str(c), [ms+'/tec1.h5' for ms in MSs.getListStr()], [parset_dir+'/losoto-plot-tec.parset'])
        os.system('mv cal-tec1-c'+str(c)+'.h5 self/solutions/')
        os.system('mv plots-tec1-c'+str(c)+' self/plots/')
    ### DONE

    with w.if_todo('cor_tec1_c%02i' % c):
        # correct TEC - group*_TC.MS:CORRECTED_DATA -> group*_TC.MS:CORRECTED_DATA
        logger.info('Correcting TEC1...')
        MSs.run('DP3 '+parset_dir+'/DP3-cor.parset msin=$pathMS msin.datacolumn=CORRECTED_DATA\
                cor.parmdb=self/solutions/cal-tec1-c'+str(c)+'.h5 cor.correction=tec000',
                log='$nameMS_corTEC-c'+str(c)+'.log', commandType='DP3')
        MSs.run('DP3 '+parset_dir+'/DP3-cor.parset msin=$pathMS msin.datacolumn=CORRECTED_DATA\
                cor.parmdb=self/solutions/cal-tec1-c'+str(c)+'.h5 cor.correction=phase000',
                log='$nameMS_corTEC-c'+str(c)+'.log', commandType='DP3')
    ### DONE

    with w.if_todo('solve_tec2_c%02i' % c):
        # Smooth CORRECTED_DATA -> SMOOTHED_DATA
        logger.info('BL-based smoothing...')
        MSs.run('BLsmooth.py -c 8 -n 8 -r -i CORRECTED_DATA -o SMOOTHED_DATA $pathMS', log='$nameMS_smooth-c'+str(c)+'.log', commandType='python')

        # solve TEC - ms:SMOOTHED_DATA (4s, 1SB)
        logger.info('Solving TEC2...')
        MSs.run('DP3 '+parset_dir+'/DP3-solTEC.parset msin=$pathMS sol.h5parm=$pathMS/tec2.h5 \
                sol.antennaconstraint=[[CS001LBA,CS002LBA,CS003LBA,CS004LBA,CS005LBA,CS006LBA,CS007LBA,CS011LBA,CS013LBA,CS017LBA,CS021LBA,CS024LBA,CS026LBA,CS028LBA,CS030LBA,CS031LBA,CS032LBA,CS101LBA,CS103LBA,CS201LBA,CS301LBA,CS302LBA,CS401LBA,CS501LBA,RS106LBA,RS205LBA,RS305LBA,RS306LBA,RS503LBA]] \
                sol.solint='+str(base_solint), \
                #+' sol.nchan='+str(4*base_nchan), \
                log='$nameMS_solTEC-c'+str(c)+'.log', commandType='DP3')

        lib_util.run_losoto(s, 'tec2-c'+str(c), [ms+'/tec2.h5' for ms in MSs.getListStr()], [parset_dir+'/losoto-plot-tec.parset'])
        os.system('mv cal-tec2-c'+str(c)+'.h5 self/solutions/')
        os.system('mv plots-tec2-c'+str(c)+' self/plots/')
    ### DONE

    with w.if_todo('cor_tec2_c%02i' % c):
        # correct TEC - group*_TC.MS:CORRECTED_DATA -> group*_TC.MS:CORRECTED_DATA
        logger.info('Correcting TEC2...')
        MSs.run('DP3 '+parset_dir+'/DP3-cor.parset msin=$pathMS msin.datacolumn=CORRECTED_DATA\
                cor.parmdb=self/solutions/cal-tec2-c'+str(c)+'.h5 cor.correction=tec000',
                log='$nameMS_corTEC-c'+str(c)+'.log', commandType='DP3')
        MSs.run('DP3 '+parset_dir+'/DP3-cor.parset msin=$pathMS msin.datacolumn=CORRECTED_DATA\
                cor.parmdb=self/solutions/cal-tec2-c'+str(c)+'.h5 cor.correction=phase000',
                log='$nameMS_corTEC-c'+str(c)+'.log', commandType='DP3')
    ### DONE

    # AMP DIE correction in last iteration
    if c == maxIter-1:
        with w.if_todo('solve_g_c%02i' % c):
            # DIE Calibration - ms:CORRECTED_DATA (8m, 4SB)
            logger.info('Solving slow G...')
            MSs.run('DP3 '+parset_dir+'/DP3-solG.parset msin=$pathMS sol.h5parm=$pathMS/g.h5 sol.solint='+str(120*base_solint)+' sol.nchan='+str(16*base_nchan),
                    log='$nameMS_solG-c'+str(c)+'.log', commandType='DP3')
            lib_util.run_losoto(s, 'g-c'+str(c), [MS+'/g.h5' for MS in MSs.getListStr()],
                    [parset_dir+'/losoto-plot-amp.parset', parset_dir+'/losoto-plot-ph.parset', parset_dir+'/losoto-amp.parset'])
            os.system('mv plots-g-c'+str(c)+' self/plots/')
            os.system('mv cal-g-c'+str(c)+'.h5 self/solutions/')
        ### DONE

        with w.if_todo('cor_g_c%02i' % c):
            # correct G - group*_TC.MS:CORRECTED_DATA -> group*_TC.MS:CORRECTED_DATA
            logger.info('Correcting G...')
            MSs.run('DP3 '+parset_dir+'/DP3-cor.parset msin=$pathMS msin.datacolumn=CORRECTED_DATA \
                    cor.parmdb=self/solutions/cal-g-c'+str(c)+'.h5 cor.correction=amplitudeSmooth',
                    log='$nameMS_corG-c'+str(c)+'.log', commandType='DP3')
        ### DONE

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

    imagename = 'img/wide-0'
    maskname = imagename + '-mask.fits'
    imagenameM = 'img/wideM-'+str(c)
    with w.if_todo('imaging_c%02i' % c):
        logger.info('Cleaning (cycle: '+str(c)+')...')
        if c == 0:
            # make temp mask for cycle 0, in cycle 1 use the maske made from cycle 0 image
            lib_util.run_wsclean(s, 'wsclean-c' + str(c) + '.log', MSsClean.getStrWsclean(), name=imagename,
                                 size=imgsizepix, scale='10arcsec',
                                 weight='briggs -0.3', niter=1000000, no_update_model_required='', minuv_l=30,
                                 parallel_gridding=2, baseline_averaging='', maxuv_l=4500, mgain=0.85,
                                 parallel_deconvolution=512, local_rms='', auto_threshold=4,
                                 join_channels='', fit_spectral_pol=cc_fit_order, channels_out=MSsClean.getChout(4.e6),
                                 deconvolution_channels=cc_fit_order)
            im = lib_img.Image(imagename + '-MFS-image.fits', userReg=userReg)
            im.makeMask(threshpix=5, atrous_do=True)

            kwargs = {'do_predict':False, 'reuse_dirty':imagename, 'reuse_psf':imagename}
        else: 
            kwargs = {}

        lib_util.run_wsclean(s, 'wscleanM-c'+str(c)+'.log', MSsClean.getStrWsclean(), name=imagenameM, save_source_list='',
                size=imgsizepix, scale='10arcsec',
                weight='briggs -0.3', niter=1000000, no_update_model_required='', minuv_l=30,
                parallel_gridding=2, baseline_averaging='', maxuv_l=4500, mgain=0.85,
                parallel_deconvolution=512, auto_threshold=3., fits_mask=maskname,
                join_channels='', fit_spectral_pol=cc_fit_order, channels_out=MSsClean.getChout(4.e6),
                multiscale='', multiscale_scale_bias=0.6,
                deconvolution_channels=cc_fit_order, **kwargs)

        os.system('cat '+logger_obj.log_dir+'/wscleanM-c'+str(c)+'.log | grep "background noise"')

        # when wasclean allow station selection, then we can remove MSsClean and this predict can go in the previous call with do_predict=True
        if c == 0:
            logger.info('Predict model...')
            s.add('wsclean -predict -padding 1.8 -name img/wideM-'+str(c)+' -j '+str(s.max_processors)+' -channels-out '+str(MSs.getChout(4e6))+' '+MSs.getStrWsclean(), \
                   log='wscleanPRE-c'+str(c)+'.log', commandType='wsclean', processors='max')
            s.run(check=True)
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
            lib_util.run_wsclean(s, 'wscleanLRmask.log', MSs.getStrWsclean(), name='img/tmp', size=imgsizepix, scale='30arcsec')
            os.system('mv img/tmp-image.fits img/wide-lr-mask.fits')
            lib_img.blank_image_reg('img/wide-lr-mask.fits', beamReg, blankval = 0.)
            lib_img.blank_image_reg('img/wide-lr-mask.fits', beamReg, blankval = 1., inverse=True)
    
            # reclean low-resolution
            logger.info('Cleaning low-res...')
            imagename_lr = 'img/wide-lr'
            lib_util.run_wsclean(s, 'wscleanLR.log', MSs.getStrWsclean(), name=imagename_lr, do_predict=False,
                    parallel_gridding=4, temp_dir='./', size=imgsizepix, scale='30arcsec',
                    weight='briggs -0.3', niter=50000, no_update_model_required='', minuv_l=30, maxuvw_m=6000,
                    taper_gaussian='200arcsec', mgain=0.85, parallel_deconvolution=512, baseline_averaging='',
                    local_rms='', auto_mask=3, auto_threshold=1.5, fits_mask='img/wide-lr-mask.fits',
                    join_channels='', channels_out=MSs.getChout(2.e6))

            s.add('wsclean -predict -padding 1.8 -name '+imagename_lr+' -j '+str(s.max_processors)+' -channels-out '+str(MSs.getChout(2e6))+' '+MSs.getStrWsclean(), \
                  log='wscleanLR-PRE-c'+str(c)+'.log', commandType='wsclean', processors='max')
            s.run(check=True)
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
            logger.info('Corrupt low-res model: TEC+Ph 1...')
            MSs.run('DP3 '+parset_dir+'/DP3-cor.parset msin=$pathMS msin.datacolumn=MODEL_DATA msout.datacolumn=MODEL_DATA  \
                    cor.parmdb=self/solutions/cal-tec1-c'+str(c)+'.h5 cor.correction=tec000 cor.invert=False',
                    log='$nameMS_corrupt.log', commandType='DP3')
            MSs.run('DP3 '+parset_dir+'/DP3-cor.parset msin=$pathMS msin.datacolumn=MODEL_DATA msout.datacolumn=MODEL_DATA  \
                    cor.parmdb=self/solutions/cal-tec1-c'+str(c)+'.h5 cor.correction=phase000 cor.invert=False',
                    log='$nameMS_corrupt.log', commandType='DP3')
            logger.info('Corrupt low-res model: TEC+Ph 2...')
            MSs.run('DP3 '+parset_dir+'/DP3-cor.parset msin=$pathMS msin.datacolumn=MODEL_DATA msout.datacolumn=MODEL_DATA  \
                    cor.parmdb=self/solutions/cal-tec2-c'+str(c)+'.h5 cor.correction=tec000 cor.invert=False',
                    log='$nameMS_corrupt.log', commandType='DP3')
            MSs.run('DP3 '+parset_dir+'/DP3-cor.parset msin=$pathMS msin.datacolumn=MODEL_DATA msout.datacolumn=MODEL_DATA  \
                    cor.parmdb=self/solutions/cal-tec2-c'+str(c)+'.h5 cor.correction=phase000 cor.invert=False',
                    log='$nameMS_corrupt.log', commandType='DP3')
        #     logger.info('Corrupt low-res model: G...')
        #     MSs.run('DP3 '+parset_dir+'/DP3-cor.parset msin=$pathMS msin.datacolumn=MODEL_DATA msout.datacolumn=MODEL_DATA \
        #             cor.parmdb=self/solutions/cal-g-c'+str(c)+'.h5 cor.correction=amplitudeSmooth cor.invert=False',
        #             log='$nameMS_corrupt.log', commandType='DP3')
        # ### DONE

        with w.if_todo('lowres_subtract_c%02i' % c):
            # Permanently subtract low-res sidelobe model - SUBTRACTED_DATA = DATA - MODEL_DATA.
            # This could be done from DATA, but the we can't restart the pipeline as easily.
            MSs.addcol('SUBTRACTED_DATA','DATA')
            logger.info('Subtracting low-res sidelobe model (SUBTRACTED_DATA = DATA - MODEL_DATA)...')
            MSs.run('taql "update $pathMS set SUBTRACTED_DATA = DATA - MODEL_DATA"', log='$nameMS_taql-c'+str(c)+'.log', commandType='general')
        ### DONE

        with w.if_todo('extreg_preapre_c%02i' % c):
            sm = lsmtool.load(f'img/wideM-{c}-sources.txt')
            sm.remove('./wide-lr-mask.fits=1') # remove sidelobe sources that were subtracted
            subfield_reg = 'self/skymodel/subfield.reg'
            field_center, field_size = lib_dd.make_subfield_region(MSs.getListObj()[0], sm, subfield_min_flux, name=subfield_reg)
            # prepare model of central/external regions
            logger.info('Blanking central region of model files and reverse...')
            for im in glob.glob('img/wideM-0*model.fits'):
                wideMint = im.replace('wideM','wideMint')
                os.system('cp %s %s' % (im, wideMint))
                lib_img.blank_image_reg(wideMint, subfield_reg, blankval = 0., inverse=True)
                wideMext = im.replace('wideM','wideMext')
                os.system('cp %s %s' % (im, wideMext))
                lib_img.blank_image_reg(wideMext, subfield_reg, blankval = 0.)
        # DONE

        with w.if_todo('extreg_predict_corrupt_subtract_c%02i' % c):
            # Recreate MODEL_DATA of external region for subtraction
            logger.info('Predict model of external region...')
            s.add('wsclean -predict -padding 1.8 -name img/wideMext-'+str(c)+' -j '+str(s.max_processors)+' -channels-out '+str(MSs.getChout(4e6))+' '+MSs.getStrWsclean(), \
                  log='wscleanPRE-c'+str(c)+'.log', commandType='wsclean', processors='max')
            s.run(check=True)

            # corrupt model with TEC+FR+Beam2ord solutions - ms:MODEL_DATA -> ms:MODEL_DATA
            logger.info('Corrupt low-res model: TEC+Ph 1...')
            MSs.run('DP3 '+parset_dir+'/DP3-cor.parset msin=$pathMS msin.datacolumn=MODEL_DATA msout.datacolumn=MODEL_DATA  \
                    cor.parmdb=self/solutions/cal-tec1-c'+str(c)+'.h5 cor.correction=tec000 cor.invert=False',
                    log='$nameMS_corrupt.log', commandType='DP3')
            MSs.run('DP3 '+parset_dir+'/DP3-cor.parset msin=$pathMS msin.datacolumn=MODEL_DATA msout.datacolumn=MODEL_DATA  \
                    cor.parmdb=self/solutions/cal-tec1-c'+str(c)+'.h5 cor.correction=phase000 cor.invert=False',
                    log='$nameMS_corrupt.log', commandType='DP3')
            logger.info('Corrupt low-res model: TEC+Ph 2...')
            MSs.run('DP3 '+parset_dir+'/DP3-cor.parset msin=$pathMS msin.datacolumn=MODEL_DATA msout.datacolumn=MODEL_DATA  \
                    cor.parmdb=self/solutions/cal-tec2-c'+str(c)+'.h5 cor.correction=tec000 cor.invert=False',
                    log='$nameMS_corrupt.log', commandType='DP3')
            MSs.run('DP3 '+parset_dir+'/DP3-cor.parset msin=$pathMS msin.datacolumn=MODEL_DATA msout.datacolumn=MODEL_DATA  \
                    cor.parmdb=self/solutions/cal-tec2-c'+str(c)+'.h5 cor.correction=phase000 cor.invert=False',
                    log='$nameMS_corrupt.log', commandType='DP3')
            # logger.info('Corrupt low-res model: G...')
            # MSs.run('DP3 '+parset_dir+'/DP3-cor.parset msin=$pathMS msin.datacolumn=MODEL_DATA msout.datacolumn=MODEL_DATA \
            #         cor.parmdb=self/solutions/cal-g-c'+str(c)+'.h5 cor.correction=amplitudeSmooth cor.invert=False',
            #         log='$nameMS_corrupt.log', commandType='DP3')

            # subtract external region from SUBTRACTED_DATA (sidelobe subtracted) to create SUBFIELD_DATA
            MSs.addcol('SUBFIELD_DATA','SUBTRACTED_DATA')
            logger.info('Subtracting external region model (SUBFIELD_DATA = SUBTRACTED_DATA - MODEL_DATA)...')
            MSs.run('taql "update $pathMS set SUBFIELD_DATA = SUBTRACTED_DATA - MODEL_DATA"', log='$nameMS_taql-c'+str(c)+'.log', commandType='general')
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
    # corrupt model with TEC+FR+Beam2ord solutions - ms:MODEL_DATA -> ms:MODEL_DATA
    logger.info('Correct low-res model: G...')
    MSs.run(f'DP3 {parset_dir}/DP3-cor.parset msin=$pathMS msin.datacolumn=DATA msout.datacolumn=CORRECTED_DATA \
            cor.parmdb=self/solutions/cal-g-c{c}.h5 cor.correction=amplitudeSmooth',
            log='$nameMS_finalcor.log', commandType='DP3')
    logger.info('Correct low-res model: TEC+Ph 1...')
    MSs.run(f'DP3 {parset_dir}/DP3-cor.parset msin=$pathMS msin.datacolumn=CORRECTED_DATA msout.datacolumn=CORRECTED_DATA  \
            cor.parmdb=self/solutions/cal-tec1-c{c}.h5 cor.correction=tec000',
            log='$nameMS_finalcor.log', commandType='DP3')
    MSs.run(f'DP3 {parset_dir}/DP3-cor.parset msin=$pathMS msin.datacolumn=CORRECTED_DATA msout.datacolumn=CORRECTED_DATA  \
            cor.parmdb=self/solutions/cal-tec1-c{c}.h5 cor.correction=phase000',
            log='$nameMS_finalcor.log', commandType='DP3')
    logger.info('Correct low-res model: TEC+Ph 2...')
    MSs.run(f'DP3 {parset_dir}/DP3-cor.parset msin=$pathMS msin.datacolumn=CORRECTED_DATA msout.datacolumn=CORRECTED_DATA  \
            cor.parmdb=self/solutions/cal-tec2-c{c}.h5 cor.correction=tec000',
            log='$nameMS_finalcor.log', commandType='DP3')
    MSs.run(f'DP3 {parset_dir}/DP3-cor.parset msin=$pathMS msin.datacolumn=CORRECTED_DATA msout.datacolumn=CORRECTED_DATA  \
            cor.parmdb=self/solutions/cal-tec2-c{c}.h5 cor.correction=phase000',
            log='$nameMS_finalcor.log', commandType='DP3')
### DONE

#TODO: Final imaging

# polarisation imaging
with w.if_todo('imaging-pol'):
    logger.info('Cleaning (Pol)...')
    imagenameP = 'img/wideP'
    lib_util.run_wsclean(s, 'wscleanP.log', MSs.getStrWsclean(), name=imagenameP, pol='QUV',
        size=imgsizepix, scale='10arcsec', weight='briggs -0.3', niter=0, no_update_model_required='',
        parallel_gridding=2, baseline_averaging='', minuv_l=30, maxuv_l=4500,
        join_channels='', channels_out=MSs.getChout(4.e6))

# Copy images
[ os.system('mv img/wideM-'+str(c)+'-MFS-image*.fits self/images') for c in range(maxIter) ]
[ os.system('mv img/wideM-'+str(c)+'-MFS-residual*.fits self/images') for c in range(maxIter) ]
[ os.system('mv img/wideM-'+str(c)+'-sources*.txt self/images') for c in range(maxIter) ]
os.system('mv img/wideP-MFS-*-image.fits self/images')
os.system('mv img/wideM-0-*-model.fits self/images')
os.system('mv img/wide-lr-MFS-image.fits self/images')
os.system('mv img/wide-largescale-MFS-image.fits self/images')

logger.info("Done.")
