#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Pipeline for direction dependent calibration

# TODO: remove regions and move to masks

import sys, os, glob, re, pickle, collections, fileinput
import numpy as np
from astropy.table import Table as astrotab
from astropy.coordinates import SkyCoord
from astropy import units as u
from astropy.io import fits
from astropy.wcs import WCS
import regions
import pyrap.tables as pt
import lsmtool

#######################################################
from LiLF import lib_ms, lib_img, lib_util, lib_log, lib_dd, lib_h5
logger_obj = lib_log.Logger('pipeline-dd')
logger = lib_log.logger
s = lib_util.Scheduler(log_dir = logger_obj.log_dir, dry = False)
w = lib_util.Walker('pipeline-dd.walker')

# parse parset
parset = lib_util.getParset()
logger.info('Parset: '+str(dict(parset['LOFAR_dd'])))
parset_dir = parset.get('LOFAR_dd','parset_dir')
userReg = parset.get('model','userReg')
maxIter = parset.getint('LOFAR_dd','maxIter')
min_cal_flux60 = parset.getfloat('LOFAR_dd','minCalFlux60')
solve_amp = parset.getboolean('LOFAR_dd','solve_amp')
manual_dd_cal = parset.get('LOFAR_dd','manual_dd_cal') # ds9 circle region file containing a manual dd-calibrator

def clean(p, MSs, res='normal', size=[1,1], empty=False, imagereg=None):
    """
    p = patch name
    mss = list of mss to clean
    size = in deg of the image
    """
    # set pixscale and imsize
    pixscale = MSs.resolution

    if res == 'normal':
        pixscale = float('%.1f'%(pixscale/2.5))
    elif res == 'high':
        pixscale = float('%.1f'%(pixscale/3.5))
    elif res == 'low':
        pass # no change

    imsize = [int(size[0]*1.5/(pixscale/3600.)), int(size[1]*1.5/(pixscale/3600.))] # add 50%
    imsize[0] += imsize[0] % 2
    imsize[1] += imsize[1] % 2
    if imsize[0] < 256: imsize[0] = 256
    if imsize[1] < 256: imsize[1] = 256

    logger.debug('Image size: '+str(imsize)+' - Pixel scale: '+str(pixscale))

    if res == 'normal':
        weight = 'briggs -0.3'
        maxuv_l = None
    elif res == 'high':
        weight = 'briggs -0.6'
        maxuv_l = None
    elif res == 'low':
        weight = 'briggs 0'
        maxuv_l = 3500
    else:
        logger.error('Wrong "res": %s.' % str(res))
        sys.exit()

    if empty:

        logger.info('Cleaning empty ('+str(p)+')...')
        imagename = 'img/empty-'+str(p)
        lib_util.run_wsclean(s, 'wscleanE-'+str(p)+'.log', MSs.getStrWsclean(), name=imagename, data_column='SUBTRACTED_DATA',
                size=imsize, scale=str(pixscale)+'arcsec',
                weight=weight, niter=0, no_update_model_required='', minuv_l=30, mgain=0,
                baseline_averaging='')
    else:

        # clean 1
        logger.info('Cleaning ('+str(p)+')...')
        imagename = 'img/ddcal-'+str(p)
        lib_util.run_wsclean(s, 'wscleanA-'+str(p)+'.log', MSs.getStrWsclean(), name=imagename,
                size=imsize, scale=str(pixscale)+'arcsec',
                weight=weight, niter=10000, no_update_model_required='', minuv_l=30, maxuv_l=maxuv_l, mgain=0.85,
                baseline_averaging='', parallel_deconvolution=512, auto_threshold=5,
                join_channels='', fit_spectral_pol=3, channels_out=ch_out, deconvolution_channels=3)
    
        # make mask
        if imagereg is not None:
            s.add('breizorro.py -t 6.5 -r %s -b 50 -o %s --merge %s' % (imagename+'-MFS-image.fits', imagename+'-mask.fits', imagereg), 
                    log='makemask-'+str(p)+'.log', commandType='python' )
            s.run()        
        else:
            s.add('breizorro.py -t 6.5 -r %s -b 50 -o %s' % (imagename+'-MFS-image.fits', imagename+'-mask.fits'), 
                    log='makemask-'+str(p)+'.log', commandType='python' )
            s.run()        

        # clean 2
        # TODO: add deconvolution_channels when bug fixed
        logger.info('Cleaning w/ mask ('+str(p)+')...')
        imagenameM = 'img/ddcalM-'+str(p)
        lib_util.run_wsclean(s, 'wscleanB-'+str(p)+'.log', MSs.getStrWsclean(), name=imagenameM, do_predict=True,
                size=imsize, save_source_list='', scale=str(pixscale)+'arcsec', reuse_psf=imagename, reuse_dirty=imagename,
                weight=weight, niter=100000, no_update_model_required='', minuv_l=30, maxuv_l=maxuv_l, mgain=0.85,
                multiscale='', multiscale_scale_bias=0.7, multiscale_scales='0,10,20,40,80', 
                baseline_averaging='',  auto_threshold=0.75, auto_mask=1.5, fits_mask=imagename+'-mask.fits',
                join_channels='', fit_spectral_pol=3, channels_out=ch_out)  #, deconvolution_channels=3) #local_rms

        os.system('cat '+logger_obj.log_dir+'/wscleanB-'+str(p)+'.log | grep "background noise"')

#############################################################
with w.if_todo('cleaning'):
    logger.info('Cleaning...')
    lib_util.check_rm('ddcal')
    os.makedirs('ddcal/init')
    os.system('cp self/skymodel/wideM-*model.fits ddcal/init/')
    os.system('cp '+sorted(glob.glob("self/images/wideM-*image.fits"))[-1]+' ddcal/init/')
    lib_util.check_rm('img')
    os.makedirs('img')
    lib_util.check_rm('mss-avg')
### DONE

# goes down to 8 seconds and multiple of 48 chans (this should be already the case as it's done in timesplit)
if not os.path.exists('mss-avg'):
    MSs = lib_ms.AllMSs( glob.glob('mss/TC*[0-9].MS'), s )
    timeint = MSs.getListObj()[0].getTimeInt()
    avgtimeint = int(round(8/timeint))  # to 8 seconds
    nchan_init = MSs.getListObj()[0].getNchan()
    # chan: avg (x8) sol (x6) - we need a multiple of 8x6=48, the largest that is <nchan
    # survey after avg (x8): 60, final number of sol 10
    # pointed after avg (x8): 120, final number of sol 20
    nchan = nchan_init - nchan_init % 48
    os.makedirs('mss-avg')
    logger.info('Averaging in time (%is -> %is), channels: %ich -> %ich)' % (timeint,timeint*avgtimeint,nchan_init,nchan))
    MSs.run('DP3 '+parset_dir+'/DP3-avg.parset msin=$pathMS msout=mss-avg/$nameMS.MS msin.datacolumn=CORRECTED_DATA msin.nchan='+str(nchan)+' \
            avg.timestep='+str(avgtimeint)+' avg.freqstep=1',
            log='$nameMS_initavg.log', commandType='DP3')

MSs = lib_ms.AllMSs(glob.glob('mss-avg/TC*[0-9].MS'), s, check_flags=True)

fwhm = MSs.getListObj()[0].getFWHM(freq='mid')
detectability_dist = MSs.getListObj()[0].getFWHM(freq='max')*1.7/2.  # 1.8 to go to close to the null
freq_min = np.min(MSs.getFreqs())
freq_mid = np.mean(MSs.getFreqs())
phase_center = MSs.getListObj()[0].getPhaseCentre()
timeint = MSs.getListObj()[0].getTimeInt()
ch_out = MSs.getChout(4e6)  # for full band (48e6 MHz) is 12
ch_out_idg = 12  # better 24, but slow
imgsizepix = int(1.7 * MSs.getListObj()[0].getFWHM(freq='mid') * 3600 / 4.)
if imgsizepix > 10000: imgsizepix = 10000 # keep SPARSE doable
if imgsizepix % 2 != 0: imgsizepix += 1  # prevent odd img sizes

with w.if_todo('add_columns'):
    logger.info('Add columns...')
    # TODO using mix of ms.addcol and addcol2ms because ms.addcol does not work with non-data columns
    MSs.addcol('CORRECTED_DATA', 'DATA', log='$nameMS_addcol.log')
    MSs.addcol('SUBTRACTED_DATA', 'DATA', log='$nameMS_addcol.log')
    MSs.run('addcol2ms.py -m $pathMS -c FLAG_BKP -i FLAG', log='$nameMS_addcol.log', commandType='python')
    MSs.run('addcol2ms.py -m $pathMS -c FLAG_PREDD -i FLAG', log='$nameMS_addcol.log', commandType='python')

##############################################################
full_image = lib_img.Image('ddcal/init/wideM-1-MFS-image.fits', userReg=userReg)

for cmaj in range(maxIter):
    logger.info('Starting major cycle: %i' % cmaj)
    
    # cycle specific variables
    picklefile = 'ddcal/directions-c%02i.pickle' % cmaj
    interp_h5parm = 'ddcal/c%02i/solutions/interp.h5' % cmaj
    interp_h5parm_old = 'ddcal/c%02i/solutions/interp.h5' % (cmaj-1)
    #aterm_config_file = 'ddcal/c%02i/aterm/aterm.config' % cmaj
    #mask_ddcal = full_image.imagename.replace('.fits', '_mask-ddcal.fits')  # this is used to find calibrators
    mask_ddcal = 'ddcal/c%02i/skymodels/mask-ddcal-c%02i.fits' % (cmaj, cmaj)  # this is used to find calibrators

    if not os.path.exists('ddcal/c%02i' % cmaj): os.makedirs('ddcal/c%02i' % cmaj)
    for subdir in ['plots','images','solutions','skymodels']:
        if not os.path.exists('ddcal/c%02i/%s' % (cmaj, subdir)): os.makedirs('ddcal/c%02i/%s' % (cmaj, subdir))

    if not os.path.exists(picklefile):
        directions = []

        # making skymodel from image
        full_image.makeMask(threshpix=4, atrous_do=False, maskname=mask_ddcal, write_srl=True, write_ds9=True)
        
        # locating DD-calibrators
        cal = astrotab.read(mask_ddcal.replace('fits','cat.fits'), format='fits')
        cal.remove_rows((cal['Total_flux'] < 0.01) | (cal['Isl_Total_flux'] < 0.1)) # remove very faint to speedup
        cal['Cluster_id'] = 'None           '
        cal['Flux_ratio'] = cal['Total_flux']/cal['Peak_flux']
        # grouping nearby sources
        grouper = lib_dd.Grouper(list(zip(cal['RA'],cal['DEC'])), cal['Total_flux'],
                                 look_distance=0.1, kernel_size=0.07, grouping_distance=0.03)
        grouper.run()
        grouper.grouping()
        grouper.plot()
        # assert sources in same island are in same group
        populated_isl = [isl for isl, n in collections.Counter(cal['Isl_id']).items() if n > 1]  # isls with more than 1 source
        ids_to_merge = [np.flatnonzero(cal['Isl_id'] == this_isl) for this_isl in populated_isl]  # list of lists of source ids that belong to populated cluster
        [grouper.merge_ids(ids) for ids in ids_to_merge]  # merge ids
        clusters = grouper.clusters
        os.system('mv grouping*png ddcal/c%02i/plots/' % cmaj)
        img_beam = full_image.getBeam()

        # reorder clusters based on flux
        fluxes = []
        for cluster_num, cluster_idxs in enumerate(clusters):
            fluxes.append(np.sum(cal['Total_flux'][cluster_idxs]))
        clusters = [x for _,x in sorted(zip(fluxes,clusters))][::-1]

        logger.info('Finding direction calibrators...')
        for cluster_num, cluster_idxs in enumerate(clusters):
            name = 'ddcal%04i' % cluster_num
            cal['Cluster_id'][cluster_idxs] = name  # just for debug
            fluxes = np.sum(cal['Total_flux'][cluster_idxs])
            spidx_coeffs = -0.8
            localrms = np.max(cal['Isl_rms'][cluster_idxs])

            # remove clusters with diffuse calibrators
            class ContinueI(Exception):
                pass
            continue_i = ContinueI()
            try:
                good_flux = 0
                for subcal in cal[cluster_idxs]:
                    if (subcal['Flux_ratio'] < 5):
                        good_flux += subcal['Total_flux'] 
                
                if good_flux < 0.7*fluxes:
                    logger.debug("%s: found extended source compact flux: %.0f%% (skip)" % (name,100*good_flux/fluxes))
                    cal['Cluster_id'][cluster_idxs] = '_'+name  # identify unused sources for debug
                    raise continue_i
            except ContinueI:
                continue

            d = lib_dd.Direction(name, soltypes=['ph1','amp1','amp2'])
            d.fluxes = fluxes
            d.spidx_coeffs = spidx_coeffs
            d.ref_freq = freq_mid
            d.localrms = localrms

            # skip faint directions
            if d.get_flux(60e6) < min_cal_flux60:
                logger.debug("%s: flux density @ 60 MHz: %.1f mJy (skip)" % (name, 1e3 * d.get_flux(60e6)))
                cal['Cluster_id'][cluster_idxs] = '_'+name  # identify unused sources for debug
            else:
                logger.debug("%s: flux density @ 60 MHz: %.1f mJy (good)" % (name, 1e3 * d.get_flux(60e6)))
                #print('DEBUG:',name,fluxes,spidx_coeffs,gauss_area,freq_mid,size,img_beam,lsm.getColValues('MajorAxis')[idx])
                ra = np.mean(cal['RA'][cluster_idxs])
                dec = np.mean(cal['DEC'][cluster_idxs])
                d.set_position([ra, dec], distance_peeloff=detectability_dist, phase_center=phase_center)
                d.set_size(cal['RA'][cluster_idxs], cal['DEC'][cluster_idxs], cal['Maj'][cluster_idxs], img_beam[0]/3600)
                d.set_region(loc='ddcal/c%02i/skymodels' % cmaj)
                model_root = 'ddcal/c%02i/skymodels/%s-init' % (cmaj, name)
                for model_file in glob.glob(full_image.root+'*[0-9]-model.fits') + glob.glob(full_image.root+'*[0-9]-model-pb.fits'):
                    os.system('cp %s %s' % (model_file, model_file.replace(full_image.root, model_root)))
                d.set_model(model_root, typ='init', apply_region=True)
                directions.append(d)

        # take care of manually provided region to include in cal
        # TODO: if this clutters the pipeline, we can move it to lib_dd!
        if manual_dd_cal != '':
            man_cal = regions.read_ds9(manual_dd_cal)
            logger.warning(
                f'Using DS9 region {manual_dd_cal} for manual dd-calibrator (experimental).')
            assert len(man_cal) == 1
            # get flux in manual reg
            reg_flux = full_image.calc_flux(manual_dd_cal)
            logger.info(f'Flux in manual region: {reg_flux:.1f} Jy')
            full_residual_image = fits.open(full_image.imagename.replace('MFS-image','MFS-residual'))  # use residual so we can get data for rms
            hdr, data_res = lib_img.flatten(full_residual_image)
            wcs = WCS(hdr)
            for i, d in enumerate(directions):  # remove dd-cals that are already in region
                ra, dec = d.position
                sc = SkyCoord(ra * u.deg, dec * u.deg, frame='fk5')
                if man_cal[0].contains(sc, wcs):
                    logger.info(f'{d.name} containted in manual region, remove auto-found direction.')
                    directions.pop(i)
            name = f'ddcal-' + manual_dd_cal.split('.')[0]
            d = lib_dd.Direction(name)
            d.fluxes = reg_flux
            d.spidx_coeffs = -0.8
            d.ref_freq = freq_mid
            # get rms in manual reg
            d.localrms = np.std(man_cal[0].to_pixel(wcs).to_mask().cutout(data_res))
            ra, dec = man_cal[0].center.ra.to_value('deg'), man_cal[0].center.dec.to_value('deg')
            d.set_position([ra, dec], distance_peeloff=detectability_dist, phase_center=phase_center)
            d.set_size([ra], [dec], [man_cal[0].radius.to_value('deg')], img_beam[0] / 3600)
            d.set_region(loc='ddcal/c%02i/skymodels' % cmaj)
            model_root = 'ddcal/c%02i/skymodels/%s-init' % (cmaj, name)
            for model_file in glob.glob(full_image.root + '*[0-9]-model.fits') + glob.glob(full_image.root + '*[0-9]-model-pb.fits'):
                os.system('cp %s %s' % (model_file, model_file.replace(full_image.root, model_root)))
            d.set_model(model_root, typ='init', apply_region=True)
            directions.insert(0, d)
        # create a concat region for debugging
        os.system('cat ddcal/c%02i/skymodels/ddcal*reg > ddcal/c%02i/skymodels/all-c%02i.reg' % (cmaj,cmaj,cmaj))
        # save catalogue for debugging
        cal.write('ddcal/c%02i/skymodels/cat-c%02i.fits' % (cmaj,cmaj), format='fits', overwrite=True)

        # order directions from the fluxiest one
        directions = [x for _, x in sorted(zip([d.get_flux(freq_mid) for d in directions],directions))][::-1]


        logger.info('Found {} cals brighter than {} Jy (expected at 60 MHz):'.format(len(directions), min_cal_flux60))
        for d in directions:
            if not d.peel_off:
                logger.info('%s: flux: %.2f Jy (rms:%.2f mJy)' % (d.name, d.get_flux(freq_mid), d.localrms*1e3))
            else:
                logger.info('%s: flux: %.2f Jy (rms: %.2f mJy - peel off)' % (d.name, d.get_flux(freq_mid), d.localrms*1e3))

        pickle.dump( directions, open( picklefile, "wb" ) )
    else:
        directions = pickle.load( open( picklefile, "rb" ) )

    # do a full predict of the sky (if c>0, corrupted with solutions of prev cycle)
    with w.if_todo('c%02i-fullpredict' % cmaj):
        # wsclean predict
        logger.info('Predict full model...')
        if cmaj == 0:
            s.add('wsclean -predict -padding 1.8 -name '+full_image.root+' -j '+str(s.max_processors)+' -channels-out '+str(ch_out)+' \
                    -reorder -parallel-reordering 4 '+MSs.getStrWsclean(),
                    log='wscleanPRE-c'+str(cmaj)+'.log', commandType='wsclean', processors='max')
            s.run(check=True)
        else:
            full_image.nantozeroModel()
            s.add('wsclean -predict -padding 1.8 -name '+full_image.root+' -j '+str(s.max_processors)+' -channels-out '+str(ch_out)+' \
                    -apply-facet-beam -use-differential-lofar-beam -facet-beam-update 120 \
                    -facet-regions '+facetregname+' -diagonal-solutions \
                    -apply-facet-solutions '+interp_h5parm_old+' '+correct_for+' \
                    -reorder -parallel-reordering 4 '+MSs.getStrWsclean(),
                    log='wscleanPRE-c'+str(cmaj)+'.log', commandType='wsclean', processors='max')
            s.run(check=True)
    ### DONE

    if cmaj > 0:
        with w.if_todo('c%02i-fulljsol' % cmaj):
            logger.info('Solving slow G (full jones)...')
            MSs.run('DP3 '+parset_dir+'/DP3-solGfj.parset msin=$pathMS sol.h5parm=$pathMS/g.h5 sol.solint=10 sol.nchan=16',
                    log='$nameMS_solG-c%02i.log' % cmaj, commandType='DP3')
            lib_util.run_losoto(s, 'g-c%02i' % cmaj, [MS+'/g.h5' for MS in MSs.getListStr()],
                    [parset_dir+'/losoto-plot-fullj.parset', parset_dir+'/losoto-bp.parset'])
            os.system('mv plots-g-c%02i ddcal/c%02i/plots/' % (cmaj, cmaj))
            os.system('mv cal-g-c%02i.h5 ddcal/c%02i/solutions/' % (cmaj, cmaj))
        ### DONE

            with w.if_todo('c%02i-fulljcor' % cmaj):
                # correct G - group*_TC.MS:DATA -> group*_TC.MS:CORRECTED_DATA
                logger.info('Correcting G...')
                MSs.run('DP3 '+parset_dir+'/DP3-correct.parset msin=$pathMS msin.datacolumn=DATA \
                    cor.parmdb=ddcal/c%02i/solutions/cal-g-c%02i.h5 cor.correction=fulljones cor.soltab=[amplitudeSmooth,phase000]' % (cmaj, cmaj),
                    log='$nameMS_corG-c%02i.log' % cmaj, commandType='DP3')
            ### DONE


    with w.if_todo('c%02i-fullsub' % cmaj):
        if cmaj == 0:
            # first cycle no corrections - ms:CORRECTED_DATA = DATA
            logger.info('Set CORRECTED_DATA = DATA...')
            MSs.run('taql "update $pathMS set CORRECTED_DATA = DATA"',
                    log='$nameMS_taql.log', commandType='general')
        # subtract - ms:SUBTRACTED_DATA = CORRECTED_DATA - MODEL_DATA
        logger.info('Set SUBTRACTED_DATA = CORRECTED_DATA - MODEL_DATA...')
        MSs.run('taql "update $pathMS set SUBTRACTED_DATA = CORRECTED_DATA - MODEL_DATA"',
                    log='$nameMS_taql.log', commandType='general')

        ### TESTTESTTEST: empty image
        #if not os.path.exists('img/empty-init-c'+str(cmaj)+'-image.fits'):
        #    clean('init-c'+str(cmaj), MSs, size=(fwhm*1.5, fwhm*1.5), res='normal', empty=True)
        ###
    ### DONE

    for dnum, d in enumerate(directions):

        logger.info('Working on direction %s/%s: %s (%f Jy - %f deg)' % (dnum+1, len(directions), d.name, d.get_flux(freq_mid), d.size))
        if d.size > 0.5: logger.warning('Patch size large: %f' % d.size)
        logstring = 'c%02i-%s' % (cmaj, d.name)

        # either faint sources that were not detected before or residuals of peeled sources - skip?
        if cmaj == 1 and d.peel_off:
            logger.info('This sources is far in the outkirts - skip.')
            continue

        with w.if_todo('%s-predict' % logstring):

            logger.info('Predict model...')
            if cmaj == 0:
                # Predict - ms:MODEL_DATA
                s.add('wsclean -predict -padding 1.8 -name '+d.get_model('init')+' -j '+str(s.max_processors)+' -channels-out '+str(ch_out)+' \
                        -reorder -parallel-reordering 4 '+MSs.getStrWsclean(),
                        log='wscleanPRE-'+logstring+'.log', commandType='wsclean', processors='max')
                s.run(check=True)
            else:
                # Predict - ms:MODEL_DATA
                s.add('wsclean -predict -padding 1.8 -name '+d.get_model('init')+' -j '+str(s.max_processors)+' -channels-out '+str(ch_out)+' \
                    -apply-facet-beam -use-differential-lofar-beam -facet-beam-update 120 \
                    -facet-regions '+full_image.root+'_facets.reg -diagonal-solutions \
                    -apply-facet-solutions '+interp_h5parm_old+' amplitude000,phase000 \
                    -reorder -parallel-reordering 4 '+MSs.getStrWsclean(),
                    log='wscleanPRE-'+logstring+'.log', commandType='wsclean', processors='max')
                s.run(check=True)
    
            # Add back the model previously subtracted for this dd-cal
            logger.info('Set SUBTRACTED_DATA = SUBTRACTED_DATA + MODEL_DATA...')
            MSs.run('taql "update $pathMS set SUBTRACTED_DATA = SUBTRACTED_DATA + MODEL_DATA"',
                    log='$nameMS_taql.log', commandType='general')
    
            ### TTESTTESTTEST: empty image but with the DD cal
            # if not os.path.exists('img/empty-butcal-%02i-%s-image.fits' % (dnum, logstring)):
            #     clean('butcal-%02i-%s' % (dnum, logstring), MSs, size=(fwhm*1.5,fwhm*1.5), res='normal', empty=True)
    
        ### DONE

        with w.if_todo('%s-shift' % logstring):
            logger.info('Phase shift and avg...')

            lib_util.check_rm('mss-dir')
            os.makedirs('mss-dir')

            # Shift - ms:SUBTRACTED_DATA -> ms:DATA (->8/16/32 s and 1 chan every 2 SBs: tot of 60 or 120 chan)
            if d.get_flux(freq_mid) > 10: avgtimeint = int(round(8/timeint))
            elif d.get_flux(freq_mid) > 4: avgtimeint = int(round(16/timeint))
            else: avgtimeint = int(round(32/timeint))
            avgfreqint = int(round(MSs.getListObj()[0].getNchan() / MSs.getChout(size=2*0.192e6))) # avg to 1 ch every 2 SBs
            if not (avgfreqint == 8 or avgfreqint == 16):
                logger.warning('Strange averaging of channels (%i): %i -> %i' % (avgfreqint,MSs.getListObj()[0].getNchan(),int(MSs.getListObj()[0].getNchan()/avgfreqint)))
            MSs.run('DP3 '+parset_dir+'/DP3-shiftavg.parset msin=$pathMS msout=mss-dir/$nameMS.MS msin.datacolumn=SUBTRACTED_DATA msout.datacolumn=DATA \
                    avg.timestep='+str(avgtimeint)+' avg.freqstep='+str(avgfreqint)+' shift.phasecenter=\['+str(d.position[0])+'deg,'+str(d.position[1])+'deg\]', \
                    log='$nameMS_shift-'+logstring+'.log', commandType='DP3')

            # save some info for debug
            d.avg_t = avgtimeint
            d.avg_f = avgfreqint
        ### DONE

        MSs_dir = lib_ms.AllMSs( glob.glob('mss-dir/*MS'), s, check_flags=False )

        with w.if_todo('%s-flag' % logstring):
            logger.info('Flag on mindata...')
            MSs_dir.run('flagonmindata.py -f 0.5 $pathMS', log='$nameMS_flagonmindata.log', commandType='python')
        ### DONE

        # Correct for beam in that direction
        if not d.peel_off:
            with w.if_todo('%s-beamcorr' % logstring):
                logger.info('Correcting beam...')
                # Convince DP3 that DATA is corrected for the beam in the phase centre and correct for the new direction
                MSs_dir.run('DP3 '+parset_dir+'/DP3-beam.parset msin=$pathMS msin.datacolumn=DATA msout.datacolumn=DATA \
                        setbeam.direction=\['+str(phase_center[0])+'deg,'+str(phase_center[1])+'deg\] \
                        corrbeam.direction=\['+str(d.position[0])+'deg,'+str(d.position[1])+'deg\] corrbeam.invert=True',
                        log='$nameMS_beam-'+logstring+'.log', commandType='DP3')
            ### DONE

        with w.if_todo('%s-preimage' % logstring):
            logger.info('Pre-imaging...')
            clean('%s-pre' % logstring, MSs_dir, res='normal', size=[d.size,d.size])#, imagereg=d.get_region())
        ### DONE
        
        # get initial noise and set iterators for timeint solutions
        image = lib_img.Image('img/ddcalM-%s-pre-MFS-image.fits' % logstring, userReg=userReg)
        rms_noise_pre = image.getNoise(); rms_noise_init = rms_noise_pre
        mm_ratio_pre = image.getMaxMinRatio(); mm_ratio_init = mm_ratio_pre
        doamp = False
        # usually there are 3600/32=112 or 3600/16=225 or 3600/8=450 timesteps and \
        # 60 (halfband)/120 (fullband) chans, try to use multiple numbers
        iter_ph_solint = lib_util.Sol_iterator([8, 4, 1])  # 32 or 16 or 8 * [8,4,1] s
        iter_amp_solint = lib_util.Sol_iterator([30, 20, 10])  # 32 or 16 or 8 * [30,20,10] s
        iter_amp2_solint = lib_util.Sol_iterator([120, 60])
        iter_ph_soltype = 'scalarphase' if (d.get_flux(freq_mid) < 5 and cmaj != 0) else 'diagonalphase'
        logger.info('RMS noise (init): %f' % (rms_noise_pre))
        logger.info('MM ratio (init): %f' % (mm_ratio_pre))

        for cdd in range(20):

            logger.info('c%02i - %s: Starting dd cycle: %02i' % (cmaj, d.name, cdd))
            logstringcal = logstring+'-cdd%02i' % cdd

            ################################################################
            # Calibrate
            solint_ph = next(iter_ph_solint)
            d.add_h5parm('ph1', 'ddcal/c%02i/solutions/cal-ph1-%s.h5' % (cmaj,logstringcal) )
            if doamp:
                solint_amp1 = next(iter_amp_solint)
                solch_amp1 = int(round(MSs_dir.getListObj()[0].getNchan() / ch_out / 4)) # TEST /4
                d.add_h5parm('amp1', 'ddcal/c%02i/solutions/cal-amp1-%s.h5' % (cmaj,logstringcal) )
                solint_amp2 = next(iter_amp2_solint)
                d.add_h5parm('amp2', 'ddcal/c%02i/solutions/cal-amp2-%s.h5' % (cmaj,logstringcal) )
            else:
                # not necessary but cleaner
                d.add_h5parm('amp1', None )
                d.add_h5parm('amp2', None )
   
            with w.if_todo('%s-calibrate' % logstringcal):
                # Smoothing - ms:DATA -> ms:SMOOTHED_DATA
                MSs_dir.run_Blsmooth(logstr=f'smooth-{logstringcal}')

                # Calibration - ms:SMOOTHED_DATA
                logger.info('Gain phase1 calibration (solint: %i)...' % solint_ph)
                MSs_dir.run('DP3 '+parset_dir+'/DP3-solG.parset msin=$pathMS msin.datacolumn=SMOOTHED_DATA sol.h5parm=$pathMS/cal-ph1.h5 \
                            sol.mode='+iter_ph_soltype+' sol.solint='+str(solint_ph)+' sol.smoothnessconstraint=3e6 sol.smoothnessreffrequency=54e6',
                            log='$nameMS_solGph1-'+logstringcal+'.log', commandType='DP3')
                lib_util.run_losoto(s, 'ph1', [ms+'/cal-ph1.h5' for ms in MSs_dir.getListStr()],
                    [parset_dir+'/losoto-plot-ph1.parset'], plots_dir='ddcal/c%02i/plots/plots-%s' % (cmaj,logstringcal))
                os.system('mv cal-ph1.h5 %s' % d.get_h5parm('ph1'))

                # correct ph - ms:DATA -> ms:CORRECTED_DATA
                logger.info('Correct ph1...')
                MSs_dir.run('DP3 '+parset_dir+'/DP3-correct.parset msin=$pathMS msin.datacolumn=DATA msout.datacolumn=CORRECTED_DATA \
                             cor.parmdb='+d.get_h5parm('ph1')+' cor.correction=phase000',
                             log='$nameMS_correct-'+logstringcal+'.log', commandType='DP3')
                
                if doamp:
                   # Smoothing - ms:CORRECTED_DATA -> ms:SMOOTHED_DATA
                    MSs_dir.run_Blsmooth('CORRECTED_DATA', logstr=f'smooth-{logstringcal}')

                    logger.info('Gain amp calibration 1 (solint: %i, solch: %i)...' % (solint_amp1, solch_amp1))
                    # Calibration - ms:CORRECTED_DATA
                    # possible to put nchan=6 if less channels are needed in the h5parm (e.g. for IDG)
                    MSs_dir.run('DP3 '+parset_dir+'/DP3-solG.parset msin=$pathMS msin.datacolumn=SMOOTHED_DATA sol.h5parm=$pathMS/cal-amp1.h5 \
                        sol.mode=diagonal sol.solint='+str(solint_amp1)+' sol.nchan='+str(solch_amp1)+' sol.minvisratio=0.5 \
                        sol.antennaconstraint=[[CS001LBA,CS002LBA,CS003LBA,CS004LBA,CS005LBA,CS006LBA,CS007LBA,CS011LBA,CS013LBA,CS017LBA,CS021LBA,CS024LBA,CS026LBA,CS028LBA,CS030LBA,CS031LBA,CS032LBA,CS101LBA,CS103LBA,CS201LBA,CS301LBA,CS302LBA,CS401LBA,CS501LBA,RS106LBA,RS205LBA,RS208LBA,RS210LBA,RS305LBA,RS306LBA,RS307LBA,RS310LBA,RS406LBA,RS407LBA,RS409LBA,RS503LBA,RS508LBA,RS509LBA]]', \
                        log='$nameMS_solGamp1-'+logstringcal+'.log', commandType='DP3')

                    #if d.peel_off:
                        #losoto_parsets = [parset_dir+'/losoto-clip.parset', parset_dir+'/losoto-norm.parset', parset_dir+'/losoto-plot-amp1.parset']
                    #else:
                    losoto_parsets = [parset_dir+'/losoto-norm.parset', parset_dir+'/losoto-plot-amp1.parset']
                    lib_util.run_losoto(s, 'amp1', [ms+'/cal-amp1.h5' for ms in MSs_dir.getListStr()], losoto_parsets,
                        plots_dir='ddcal/c%02i/plots/plots-%s' % (cmaj,logstringcal))
                    os.system('mv cal-amp1.h5 %s' % d.get_h5parm('amp1'))

                    logger.info('Correct amp 1...')
                    # correct amp - ms:CORRECTED_DATA -> ms:CORRECTED_DATA
                    MSs_dir.run('DP3 '+parset_dir+'/DP3-correct.parset msin=$pathMS msin.datacolumn=CORRECTED_DATA msout.datacolumn=CORRECTED_DATA \
                        cor.parmdb='+d.get_h5parm('amp1')+' cor.correction=amplitude000',
                        log='$nameMS_correct-'+logstringcal+'.log', commandType='DP3')

                   # Smoothing - ms:CORRECTED_DATA -> ms:SMOOTHED_DATA
                    MSs_dir.run_Blsmooth('CORRECTED_DATA', logstr=f'smooth-{logstringcal}')

                    logger.info('Gain amp calibration 2 (solint: %i)...' % solint_amp2)
                    # Calibration - ms:CORRECTED_DATA
                    MSs_dir.run('DP3 '+parset_dir+'/DP3-solG.parset msin=$pathMS msin.datacolumn=CORRECTED_DATA sol.h5parm=$pathMS/cal-amp2.h5 \
                        sol.mode=diagonal sol.solint='+str(solint_amp2)+' sol.smoothnessconstraint=10e6 sol.minvisratio=0.5',
                        log='$nameMS_solGamp2-'+logstringcal+'.log', commandType='DP3')

                    #if d.peel_off:
                        #losoto_parsets = [parset_dir+'/losoto-clip2.parset', parset_dir+'/losoto-norm.parset', parset_dir+'/losoto-plot-amp2.parset']
                    #else:
                    losoto_parsets = [parset_dir+'/losoto-norm.parset', parset_dir+'/losoto-plot-amp2.parset']
                    lib_util.run_losoto(s, 'amp2', [ms+'/cal-amp2.h5' for ms in MSs_dir.getListStr()], losoto_parsets,
                        plots_dir='ddcal/c%02i/plots/plots-%s' % (cmaj,logstringcal))
                    os.system('mv cal-amp2.h5 %s' % d.get_h5parm('amp2'))

                    logger.info('Correct amp 2...')
                    # correct amp2 - ms:CORRECTED_DATA -> ms:CORRECTED_DATA
                    MSs_dir.run('DP3 '+parset_dir+'/DP3-correct.parset msin=$pathMS msin.datacolumn=CORRECTED_DATA msout.datacolumn=CORRECTED_DATA \
                        cor.parmdb='+d.get_h5parm('amp2')+' cor.correction=amplitude000',
                        log='$nameMS_correct-'+logstringcal+'.log', commandType='DP3')
            ### DONE

            ###########################################################################
            # Imaging
            with w.if_todo('%s-image' % logstringcal):

                logger.info('%s (cdd: %02i): imaging...' % (d.name, cdd))
                clean('%s' % logstringcal, MSs_dir, res='normal', size=[d.size,d.size])#, imagereg=d.get_region())
            ### DONE

            image = lib_img.Image('img/ddcalM-%s-MFS-image.fits' % logstringcal, userReg=userReg)
            # something went wrong during last imaging, break
            if not os.path.exists(image.imagename):
                logger.warning('Breaking because the imaging diverged...')
                break

            # get noise, if larger than prev cycle: break
            rms_noise = image.getNoise()
            mm_ratio = image.getMaxMinRatio()
            d.add_rms_mm(rms_noise, mm_ratio) # track values for debug
            logger.info('RMS noise (cdd:%02i): %f' % (cdd,rms_noise))
            logger.info('MM ratio (cdd:%02i): %f' % (cdd,mm_ratio))
            # if noise incresed and mm ratio decreased - or noise increased a lot!
            if (rms_noise > 0.99*rms_noise_pre and mm_ratio < 1.01*mm_ratio_pre) or rms_noise > 1.2*rms_noise_pre:
                   if (mm_ratio < 10 and cdd >= 2) or \
                   (mm_ratio < 20 and cdd >= 3) or \
                   (cdd >= 4): 
                       logger.debug('BREAK ddcal self cycle with noise: %f, noise_pre: %f, mmratio: %f, mmratio_pre: %f' % (rms_noise,rms_noise_pre,mm_ratio,mm_ratio_pre))
                       break

            if cmaj > 0 and cdd >= 3 and ((d.get_flux(freq_mid) > 1 and mm_ratio >= 30) or d.get_flux(freq_mid) > 5) and solve_amp:
                logger.debug('START AMP WITH MODE 1 - flux: %f - mmratio: %f - dist: %f' % (d.get_flux(freq_mid), mm_ratio, d.dist_from_centre))
                doamp = True
            # correct more amp in the outskirts
            elif cmaj > 0 and d.dist_from_centre >= fwhm/4. and cdd >= 3 and ((d.get_flux(freq_mid) > 1 and mm_ratio >= 25) or d.get_flux(freq_mid) > 3) and solve_amp:
                logger.debug('START AMP WITH MODE 2 - flux: %f - mmratio: %f - dist: %f' % (d.get_flux(freq_mid), mm_ratio, d.dist_from_centre))
                doamp = True
            elif cmaj > 0 and d.dist_from_centre >= fwhm/2. and cdd >= 3 and ((d.get_flux(freq_mid) > 1 and mm_ratio >= 20) or d.get_flux(freq_mid) > 2) and solve_amp:
                logger.debug('START AMP WITH MODE 3 - flux: %f - mmratio: %f - dist: %f' % (d.get_flux(freq_mid), mm_ratio, d.dist_from_centre))
                doamp = True

            d.set_model(image.root, typ='best', apply_region=False)  # current best model
            rms_noise_pre = rms_noise
            mm_ratio_pre = mm_ratio

        # End calibration cycle
        ##################################

        # if died the first cycle or diverged
        if cdd == 0 or (rms_noise_pre > rms_noise_init) or \
            ((rms_noise_pre*2 > rms_noise_init) and (mm_ratio_pre/2 < mm_ratio_init)):
            
            d.converged = False
            logger.warning('%s: something went wrong during the first self-cal cycle or noise did not decrease.' % (d.name))
            d.clean()
            with w.if_todo('%s-subtract' % logstring):
                # Remove the ddcal to clean up the SUBTRACTED_DATA
                logger.info('Set SUBTRACTED_DATA = SUBTRACTED_DATA - MODEL_DATA')
                MSs.run('taql "update $pathMS set SUBTRACTED_DATA = SUBTRACTED_DATA - MODEL_DATA"',
                        log='$nameMS_taql.log', commandType='general')
            ### DONE

        # converged
        else:
            d.converged = True
            logger.info('%s: converged.' % d.name)
            # copy in the ddcal dir the best model
            model_skymodel = 'ddcal/c%02i/skymodels/%s-best-source.txt' % (cmaj, d.name)
            model_skydb = 'ddcal/c%02i/skymodels/%s-best-source.skydb' % (cmaj, d.name)
            os.system('cp %s %s' % (d.get_model('best')+'-sources.txt', model_skymodel))

            # restrict to initial mask
            logger.info('Restrict model to initial region')
            os.system('cp '+d.model['best']+'-mask.fits'+' '+d.model['best']+'-mask-restricted.fits')
            lib_img.blank_image_reg(d.model['best']+'-mask-restricted.fits', d.get_region(), inverse=True, blankval=0.)
            lsm = lsmtool.load(model_skymodel)
            lsm.select('%s == True' % (d.model['best']+'-mask-restricted.fits'))
            lsm.write(model_skymodel, format='makesourcedb', clobber=True)

            lib_util.check_rm(model_skydb)
            s.add('makesourcedb outtype="blob" format="<" in="%s" out="%s"' % (model_skymodel, model_skydb), log='makesourcedb_cl.log', commandType='general' )
            s.run()

            # remove the DD-cal from original dataset using new solutions
            with w.if_todo('%s-subtract' % logstring):
    
                # Predict - ms:MODEL_DATA
                logger.info('Add best model to MODEL_DATA...')
                MSs.run('DP3 '+parset_dir+'/DP3-predict.parset msin=$pathMS pre.sourcedb='+model_skydb,
                    log='$nameMS_pre-'+logstring+'.log', commandType='DP3')
    
                # Store FLAGS - just for sources to peel as they might be visible only for a fraction of the band
                if d.peel_off:
                    MSs.run('taql "update $pathMS set FLAG_BKP = FLAG"',
                             log='$nameMS_taql.log', commandType='general')
        
                # Corrput now model - ms:MODEL_DATA -> MODEL_DATA
                # note that these are all scalar or diagonal, so they commute
                logger.info('Corrupt ph...')
                MSs.run('DP3 '+parset_dir+'/DP3-correct.parset msin=$pathMS msin.datacolumn=MODEL_DATA msout.datacolumn=MODEL_DATA \
                            cor.invert=False cor.parmdb='+d.get_h5parm('ph1',-2)+' cor.correction=phase000',
                            log='$nameMS_corruping'+logstring+'.log', commandType='DP3')
        
                if not d.get_h5parm('amp1',-2) is None:
                    logger.info('Corrupt amp...')
                    MSs.run('DP3 '+parset_dir+'/DP3-correct.parset msin=$pathMS msin.datacolumn=MODEL_DATA msout.datacolumn=MODEL_DATA \
                           cor.invert=False cor.parmdb='+d.get_h5parm('amp1',-2)+' cor.correction=amplitude000',
                           log='$nameMS_corrupt-'+logstring+'.log', commandType='DP3')
                if not d.get_h5parm('amp2',-2) is None:
                    MSs.run('DP3 '+parset_dir+'/DP3-correct.parset msin=$pathMS msin.datacolumn=MODEL_DATA msout.datacolumn=MODEL_DATA \
                           cor.invert=False cor.parmdb='+d.get_h5parm('amp2',-2)+' cor.correction=amplitude000',
                           log='$nameMS_corrupt-'+logstring+'.log', commandType='DP3')
        
                if not d.peel_off:
                    # Corrupt for the beam
                    logger.info('Corrupting beam...')
                    # Convince DP3 that MODELDATA is corrected for the beam in the dd-cal direction, so I can corrupt
                    MSs.run('DP3 '+parset_dir+'/DP3-beam.parset msin=$pathMS msin.datacolumn=MODEL_DATA msout.datacolumn=MODEL_DATA \
                           setbeam.direction=\['+str(d.position[0])+'deg,'+str(d.position[1])+'deg\] \
                           corrbeam.direction=\['+str(d.position[0])+'deg,'+str(d.position[1])+'deg\] corrbeam.invert=False', \
                           log='$nameMS_beam-'+logstring+'.log', commandType='DP3')
                    MSs.run('DP3 '+parset_dir+'/DP3-beam2.parset msin=$pathMS msin.datacolumn=MODEL_DATA msout.datacolumn=MODEL_DATA \
                           corrbeam.direction=\['+str(phase_center[0])+'deg,'+str(phase_center[1])+'deg\] corrbeam.beammode=element corrbeam.invert=True', \
                           log='$nameMS_beam-'+logstring+'.log', commandType='DP3')
        
                if d.peel_off:
                    # Set MODEL_DATA = 0 where data are flagged, then unflag everything
                    MSs.run('taql "update $pathMS set MODEL_DATA[FLAG] = 0"',
                            log='$nameMS_taql.log', commandType='general')
                    # Restore of FLAGS
                    MSs.run('taql "update $pathMS set FLAG = FLAG_BKP"',
                            log='$nameMS_taql.log', commandType='general')
    
                    # if it's a source to peel, remove it from the data column used for imaging
                    logger.info('Source to peel: set CORRECTED_DATA = CORRECTED_DATA - MODEL_DATA')
                    MSs.run('taql "update $pathMS set CORRECTED_DATA = CORRECTED_DATA - MODEL_DATA"', \
                        log='$nameMS_taql.log', commandType='general')
    
                # Remove the ddcal again
                logger.info('Set SUBTRACTED_DATA = SUBTRACTED_DATA - MODEL_DATA')
                MSs.run('taql "update $pathMS set SUBTRACTED_DATA = SUBTRACTED_DATA - MODEL_DATA"',
                        log='$nameMS_taql.log', commandType='general')

            ### DONE

        ### TTESTTESTTEST: empty image
        # if not os.path.exists('img/empty-%02i-%s-image.fits' % (dnum, logstring)):
        #     clean('%02i-%s' % (dnum, logstring), MSs, size=(fwhm*1.5,fwhm*1.5), res='normal', empty=True)
        ###

    #####################################################
    # print a debug table
    logger.info("################################################")
    for d in directions:
        if d.peel_off and cmaj == 0:
            logger.info("### Direction (PEEL!): %s -- %.2f Jy" % (d.name, np.sum(d.fluxes)))
        elif d.peel_off and cmaj == 1:
            continue
        else:
            logger.info("### Direction: %s -- %.2f Jy" % (d.name, np.sum(d.fluxes)))
        logger.info("- Averaging: %i s - %i ch" % (d.avg_t, d.avg_f))
        logger.info("- Converged: %s" % str(d.converged))
        for ic, (rms_noise, mm_ratio) in enumerate(zip(d.rms_noise,d.mm_ratio)):

            tables_to_print = '['
            for sol_type in ['ph1','amp1','amp2']: # fr
                if d.get_h5parm(sol_type, pos=ic) is not None:
                    tables_to_print += sol_type+','
            tables_to_print = tables_to_print[:-1] + ']'

            if ic == len(d.rms_noise)-2 and d.converged:
                logger.info('%02i: Rms: %f, MMratio: %f - %s ***' % (ic,rms_noise,mm_ratio,tables_to_print))
            else:
                logger.info('%02i: Rms: %f, MMratio: %f - %s' % (ic,rms_noise,mm_ratio,tables_to_print))

        # replace color in the region file to distinguish region that converged from those that didn't
        if d.converged:
            for line in fileinput.input('ddcal/c%02i/skymodels/all-c%02i.reg' % (cmaj,cmaj), inplace=True):
                if d.name in line:
                    if d.get_h5parm('amp1',-2) is not None:
                        print(line.replace('color=red','color=blue'), end='')
                    else:
                        print(line.replace('color=red','color=green'), end='')
                else:
                    print(line, end='')

    logger.info("################################################")

    ######################################################
    # full imaging

    # be sure not to use flagged MS as ddf doesn't like them
    MSs = lib_ms.AllMSs(glob.glob('mss-avg/TC*[0-9].MS'), s, check_flags=True)
    
    imagename = 'img/wideDD-c%02i' % (cmaj)
    maskname = imagename+'_mask.fits'
    facetregname = 'ddcal/c%02i/images/wideDD-c%02i_facets.reg' % (cmaj, cmaj)

    # combine the h5parms
    h5parms = {'ph':[], 'amp1':[], 'amp2':[]}
    for d in directions:
        # only those who converged
        if d.peel_off:
            continue
        if not d.converged:
            continue

        h5parms['ph'].append(d.get_h5parm('ph1',-2))
        if d.get_h5parm('amp1',-2) is not None:
            h5parms['amp1'].append(d.get_h5parm('amp1',-2))
        if d.get_h5parm('amp2',-2) is not None:
            h5parms['amp2'].append(d.get_h5parm('amp2',-2))

        log = '%s: Phase (%s)' % (d.name, d.get_h5parm('ph1',-2))
        log += ' Amp1 (%s)' % (d.get_h5parm('amp1',-2))
        log += ' Amp2 (%s)' % (d.get_h5parm('amp2',-2))
        logger.info(log)

    # it might happens that no directions ended up with amp solutions, restrict to phase only correction
    correct_for = 'phase000'
    if len(h5parms['amp1']) != 0: correct_for += ',amplitude000'

    with w.if_todo('c%02i-interpsol' % cmaj):
        logger.info("Imaging - preparing solutions:")

        for typ, h5parm_list in h5parms.items():
            # rename direction
            for h5parmFile in h5parm_list:
                dirname = h5parmFile.split('-')[3]
                lib_h5.repoint(h5parmFile, dirname)
                if typ == 'ph':
                    lib_h5.addpol(h5parmFile, 'phase000')
                    #lib_h5.addpol(h5parmFile, 'amplitude000')
                    s.add('losoto -v '+h5parmFile+' '+parset_dir+'/losoto-refph.parset ', log='h5parm_collector.log', commandType='python' )
                    s.run()
                    # reset high-res amplitudes in ph-solve
                    #s.add('losoto -v '+h5parmFile+' '+parset_dir+'/losoto-resetamp.parset ', log='h5parm_collector.log', commandType='python' )
                    #s.run()
                if typ == 'amp1' or typ == 'amp2':
                    s.add('losoto -v '+h5parmFile+' '+parset_dir+'/losoto-resetph.parset ', log='h5parm_collector.log', commandType='python' )
                    s.run()
    
        lib_util.check_rm(interp_h5parm)
        logger.info('Interpolating solutions...')
        s.add('H5parm_interpolator.py -o '+interp_h5parm+' '+' '.join(h5parms['ph']+h5parms['amp1']+h5parms['amp2']), log='h5parm_interpolator.log', commandType='python' )
        s.run()

    with w.if_todo('c%02i-imaging' % cmaj):

        logger.info('Preparing region file...')
        s.add('ds9_facet_generator.py --ms '+MSs.getListStr()[0]+' --h5 '+interp_h5parm+' --imsize '+str(imgsizepix)+' \
                --pixelscale 4 --writevoronoipoints --output '+facetregname,
                log='facet_generator.log', commandType='python')
        s.run()

        # might want to add non-circ beam at low dec eventually
        # if phase_center[1] < 24:
        #     logger.info(f'Low-declination observation ({phase_center[1]}deg). Use non-circular PSF')

        # update_model=True to make continue after masking
        logger.info('Cleaning 1...')
        lib_util.run_wsclean(s, 'wsclean-c'+str(cmaj)+'.log', MSs.getStrWsclean(), concat_mss=True, name=imagename, data_column='CORRECTED_DATA', size=imgsizepix, scale='4arcsec',
                weight='briggs -0.3', niter=1000000, gridder='wgridder', parallel_gridding=8, update_model_required='', minuv_l=30, mgain=0.8, parallel_deconvolution=1024,
                auto_threshold=3.0, auto_mask=5.0, join_channels='', fit_spectral_pol=3, channels_out=str(ch_out), deconvolution_channels=3,
                multiscale='', multiscale_scale_bias=0.6, pol='i', nmiter=1, dd_psf_grid='25 25', beam_size=15,
                apply_facet_beam='', facet_beam_update=120, use_differential_lofar_beam='', facet_regions=facetregname, diagonal_solutions='', apply_facet_solutions=f'{interp_h5parm} {correct_for}' )

        # masking
        s.add('breizorro.py -t 6 -r %s -b 50 -o %s' % (imagename+'-MFS-image.fits', maskname), 
                log='makemask-'+str(cmaj)+'.log', commandType='python' )
        s.run()        

        # if defined, add userReg to the mask
        if userReg != '': lib_img.blank_image_reg(maskname, userReg, blankval = 1.)

        # TODO: Add force-reuse psf (and beam) once wsclean bug is fixed.
        # HE: What is optimal choice of subimage size and parallel gridding? Is cleaning to 3sigma enough?
        logger.info('Cleaning 2...')
        lib_util.run_wsclean(s, 'wsclean-c'+str(cmaj)+'.log', MSs.getStrWsclean(), concat_mss=True, name=imagename, data_column='CORRECTED_DATA', size=imgsizepix, scale='4arcsec',
                weight='briggs -0.3', niter=1000000, gridder='wgridder', parallel_gridding=8, no_update_model_required='', minuv_l=30, mgain=0.8, parallel_deconvolution=1024,
                auto_threshold=3.0, auto_mask=5.0, fits_mask=maskname, join_channels='', fit_spectral_pol=3, channels_out=str(ch_out), deconvolution_channels=3,
                multiscale='', multiscale_scale_bias=0.6, pol='i', dd_psf_grid='25 25', beam_size=15, cont=True,
                apply_facet_beam='', facet_beam_update=120, use_differential_lofar_beam='', facet_regions=facetregname, diagonal_solutions='', apply_facet_solutions=f'{interp_h5parm} {correct_for}')
 
        os.system('mv %s*MFS*fits %s-0*fits %s_mask.fits ddcal/c%02i/images' % (imagename, imagename, imagename, cmaj))

    ### DONE

    full_image = lib_img.Image('ddcal/c%02i/images/%s-MFS-image.fits' % (cmaj, imagename.split('/')[-1]), userReg=userReg)
    min_cal_flux60 *= 0.6  # go deeper

##############################################################################################################
### Calibration finished - additional images with scientific value

with w.if_todo('output-vstokes'):
    imagenameV = 'img/wideDD-v-c%02i' % (cmaj)
    logger.info('Cleaning (V-stokes)...')
    lib_util.run_wsclean(s, 'wscleanV-c'+str(cmaj)+'.log', MSs.getStrWsclean(), concat_mss=True, name=imagenameV, data_column='CORRECTED_DATA', size=imgsizepix, scale='4arcsec',
                weight='briggs -0.3', niter=1000000, gridder='wgridder', parallel_gridding=2, no_update_model_required='', minuv_l=30, mgain=0.85, parallel_deconvolution=512,
                auto_threshold=3.0, join_channels='', fit_spectral_pol=3, channels_out=6, deconvolution_channels=3,
                pol='v')

    os.system('mv %s*MFS*.fits ddcal/c%02i/images' % (imagenameV, cmaj))
### DONE

# TODO: the model to subtract should be done from a high-res image to remove only point sources
with w.if_todo('output-lres'):

    # now make a low res and source subtracted map for masking extended sources
    logger.info('Predicting DD-corrupted...')
    s.add('wsclean -predict -padding 1.8 -name '+full_image.root+' -j '+str(s.max_processors)+' -channels-out '+str(ch_out)+' \
                    -apply-facet-beam -use-differential-lofar-beam -facet-beam-update 120 \
                    -facet-regions '+facetregname+' -diagonal-solutions \
                    -apply-facet-solutions '+interp_h5parm+' amplitude000,phase000 \
                    -reorder -parallel-reordering 4 '+MSs.getStrWsclean(),
                    log='wscleanPRE4LR-c'+str(cmaj)+'.log', commandType='wsclean', processors='max')
    s.run(check=True)

    logger.info('Set SUBTRACTED_DATA = CORRECTED_DATA - MODEL_DATA...')
    MSs.run('taql "update $pathMS set SUBTRACTED_DATA = CORRECTED_DATA - MODEL_DATA"',
        log='$nameMS_taql.log', commandType='general')

    imagenameL = 'img/wideDD-lres-c%02i' % (cmaj)
    logger.info('Cleaning (low res)...')
    lib_util.run_wsclean(s, 'wscleanLR-c'+str(cmaj)+'.log', MSs.getStrWsclean(), concat_mss=True, name=imagenameL, data_column='SUBTRACTED_DATA', size=int(imgsizepix/4), scale='16arcsec',
                weight='briggs 0', niter=1000000, gridder='wgridder', parallel_gridding=2, no_update_model_required='', minuv_l=30, mgain=0.8, parallel_deconvolution=512,
                auto_threshold=3.0, join_channels='', fit_spectral_pol=3, channels_out=6, deconvolution_channels=3,
                multiscale='', multiscale_scale_bias=0.6, pol='i', dd_psf_grid='5 5', taper_gaussian='60arcsec',
                apply_facet_beam='', use_differential_lofar_beam='', facet_beam_update=120, facet_regions=facetregname, diagonal_solutions='', apply_facet_solutions=f'{interp_h5parm} {correct_for}')
 
    os.system('mv %s*MFS*.fits ddcal/c%02i/images' % (imagenameL, cmaj))
### DONE

with w.if_todo('output_PB'):
    logger.info('Make primary beam...')
    s.add('makepb.py -o ddcal/primarybeam.fits -s 10 -p 120 %s' % MSs.getStrWsclean(),
          log='makepb.log', commandType='python', processors='max')
    s.run(check=True)
### DONE

# remove unwanted columns
with w.if_todo('remove_col'):
    logger.info('Removing unwanted columns...')
    MSs.run('taql "ALTER TABLE $pathMS DELETE COLUMN FLAG_BKP,SUBTRACTED_DATA,MODEL_DATA"', log='$nameMS_delcol.log', commandType='python')
### DONE

logger.info("Done.")
