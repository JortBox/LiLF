#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys, os, glob
import numpy as np
import pyregion

from pipelines.pipeline3C.calibration import SelfCalibration
from pyregion.parser_helper import Shape
from astropy.table import Table as astrotab

sys.path.append("/net/voorrijn/data2/boxelaar/scripts/LiLF")

from LiLF_lib import lib_img, lib_util, lib_log
from LiLF_lib.lib_ms import AllMSs as MeasurementSets

logger = lib_log.logger
WALKER = lib_util.Walker("pipeline-3c.walker")

# parse parset
parset = lib_util.getParset()
parset_dir = parset.get("LOFAR_3c_core", "parset_dir")

TARGET = os.getcwd().split("/")[-1]


def subtract_field(MSs: MeasurementSets, s: lib_util.Scheduler, region):
    imagename = "img/img-wide"
    imagenameM = "img/img-wideM"
    
    # Low res image
    logger.info("Cleaning wide 1...")
    lib_util.run_wsclean(
        s,
        "wsclean-wide.log",
        MSs.getStrWsclean(),
        name=imagename,
        parallel_gridding=4,
        baseline_averaging="",
        size=2500,
        scale="10arcsec",
        weight="briggs -0.7",
        taper_gaussian="30arcsec",
        niter=1000000,
        no_update_model_required="",
        minuv_l=30,
        mgain=0.75,
        nmiter=0,
        auto_threshold=5,
        local_rms="",
        local_rms_method="rms-with-min",
        join_channels="",
        fit_spectral_pol=2,
        channels_out=2,
    )
    os.system('cat logs/wsclean-wide.log | grep "background noise"')
    
    # makemask
    im = lib_img.Image(imagename + "-MFS-image.fits", userReg=region)
    im.makeMask(mode="default", threshpix=5, rmsbox=(50, 5))
    maskfits = imagename + "-mask.fits"

    logger.info("Cleaning wide 2...")
    lib_util.run_wsclean(
        s,
        "wsclean-wide.log",
        MSs.getStrWsclean(),
        name=imagenameM,
        do_predict=True,
        parallel_gridding=4,
        baseline_averaging="",
        size=2500,
        reuse_psf=imagename,
        reuse_dirty=imagename,
        scale="10arcsec",
        weight="briggs -0.7",
        taper_gaussian="30arcsec",
        fits_mask=maskfits,
        niter=1000000,
        no_update_model_required="",
        minuv_l=30,
        mgain=0.75,
        nmiter=0,
        auto_threshold=1,
        auto_mask=3,
        local_rms="",
        local_rms_method="rms-with-min",
        join_channels="",
        fit_spectral_pol=2,
        channels_out=2,
    )

    # subtract everything
    logger.info("Subtract model: CORRECTED_DATA = CORRECTED_DATA - MODEL_DATA...")
    MSs.run(
        'taql "update $pathMS set CORRECTED_DATA = CORRECTED_DATA - MODEL_DATA"',
        log="$nameMS_taql.log",
        commandType="general",
    )

def peel(MSs: MeasurementSets, s: lib_util.Scheduler, mask, peel_max: int = 2, original: bool = True):
    _, beam07reg, region = mask
    #cal = SelfCalibration(MSs, mask=mask, schedule=s)
    
    
    
    # 2: Sub field + peel
    with WALKER.if_todo("sub-field"):
        subtract_field(MSs, s, region)
        

    # load skymodel
    imagename = "img/img-wideM"
    full_image = lib_img.Image(imagename + "-MFS-image.fits", userReg=region)
    mask_ddcal = full_image.imagename.replace(".fits", "_mask-ddcal.fits")  # this is used to find calibrators
    
    full_image.makeMask(
        mode="default",
        threshpix=5,
        atrous_do=False,
        maskname=mask_ddcal,
        write_srl=True,
        write_ds9=True,
    )
    
    cal = astrotab.read(mask_ddcal.replace("fits", "cat.fits"), format="fits")
    cal = cal[np.where(cal["Total_flux"] > 10)]
    cal.sort(["Total_flux"], reverse=True)
    
    print(cal["Source_id"])
    #print(cal.colnames)
    print(cal["Total_flux"])
    #sys.exit()

    # cycle on sources to peel
    phasecentre = MSs.getListObj()[0].getPhaseCentre()
    for peelsou in cal:
        name = str(peelsou["Source_id"])
        # Skip if source is close to phase centre
        dist = lib_util.distanceOnSphere(
            phasecentre[0], phasecentre[1], peelsou["RA"], peelsou["DEC"]
        )
        
        if dist < 0.1: #deg
            continue

        logger.info(f"Peeling {name} ({peelsou['Total_flux']:.1f} Jy)")
        with WALKER.if_todo("peel-%s" % name):
            lib_util.check_rm("peel-%s" % name)
            os.system("mkdir peel-%s" % name)

            # make a region
            peel_region_file = f"peel-{name}/{name}.reg"
            sh = Shape("circle", None)
            sh.coord_format = "fk5"  # type: ignore
            sh.coord_list = [peelsou["RA"], peelsou["DEC"], 0.075] # ra, dec, diam # type: ignore
            sh.coord_format = "fk5"  # type: ignore
            sh.attr = ( # type: ignore
                [], 
                {
                    "width": "2", 
                    "point": "cross", 
                    "font": '"helvetica 16 normal roman"'
                }
            )  
            sh.comment = 'color=red text="%s"' % (name + ".reg")  # type: ignore
            regions = pyregion.ShapeList([sh])
            lib_util.check_rm(peel_region_file)
            regions.write(peel_region_file)

            # copy and blank models
            logger.info("Peel - Cleanup model images...")
            os.system("cp " + imagename + "*model.fits peel-" + name)
            imagename_peel = "peel-" + name + "/" + imagename.split("/")[-1]
            for model_file in glob.glob(imagename_peel + "*model.fits"):
                lib_img.blank_image_reg(
                    model_file, peel_region_file, blankval=0.0, inverse=True
                )

            # predict the source to peel
            logger.info("Peel - Predict init...")
            s.add(
                f"wsclean -predict -name {imagename_peel} \
                    -j {s.max_processors} -channels-out 2 -reorder \
                    -parallel-reordering 4 {MSs.getStrWsclean()}",
                log="wsclean-pre.log",
                commandType="wsclean",
                processors="max",
            )
            s.run(check=True)
            
            
            # add the source to peel back
            logger.info(
                "Peel - add model: CORRECTED_DATA = CORRECTED_DATA + MODEL_DATA..."
            )
            MSs.run(
                'taql "update $pathMS set CORRECTED_DATA = CORRECTED_DATA + MODEL_DATA"',
                log="$nameMS_taql.log",
                commandType="general",
            )
            

            # phaseshift + avg
            logger.info("Peel - Phaseshift+avg...")
            lib_util.check_rm("mss-dir")
            os.makedirs("mss-dir")
            MSs.run(
                f"DP3 {parset_dir}/DP3-shiftavg.parset msin=$pathMS \
                    msout=mss-dir/$nameMS.MS msin.datacolumn=CORRECTED_DATA \
                    msout.datacolumn=DATA avg.timestep=8 avg.freqstep=16 \
                    shift.phasecenter=[{peelsou['RA']}deg,{peelsou['DEC']}deg]",
                log="$nameMS_shift.log",
                commandType="DP3",
            )
            
            MSs_shift = MeasurementSets(
                glob.glob("mss-dir/*.MS"), 
                s, 
                check_flags=False, 
                check_sun=True
            )

            # image
            logger.info("Peel - Image...")
            imagename_peel = "peel-%s/img_%s" % (name, name)
            lib_util.run_wsclean(
                s,
                "wsclean-peel.log",
                MSs_shift.getStrWsclean(),
                do_predict=True,
                name=imagename_peel,
                size=512,
                parallel_gridding=4,
                baseline_averaging="",
                scale="2.5arcsec",
                niter=100000,
                no_update_model_required="",
                minuv_l=30,
                mgain=0.4,
                nmiter=0,
                auto_threshold=5,
                local_rms="",
                local_rms_method="rms-with-min",
                join_channels="",
                fit_spectral_pol=2,
                channels_out=2,
            )

            
            # calibrate
            logger.info("Peel - Calibrate...")
            MSs_shift.run(
                "DP3 "
                + parset_dir
                + "/DP3-solG.parset msin=$pathMS msin.datacolumn=DATA \
                    sol.h5parm=$pathMS/calGp.h5 sol.mode=scalar \
                    sol.solint=10 sol.smoothnessconstraint=1e6",
                log="$nameMS_solGp-peel.log",
                commandType="DP3",
            )
            
            if original:
                lib_util.run_losoto(
                    s,
                    "Gp-peel_%s" % name,
                    [ms + "/calGp.h5" for ms in MSs_shift.getListStr()],
                    [
                        parset_dir + "/losoto-plot2d.parset",
                        parset_dir + "/losoto-plot.parset",
                    ],
                    plots_dir="peel-%s" % name,
                )
            
            if not original:
                lib_util.run_losoto(
                    s, 
                    "Gp-peel_%s" % name,
                    [ms + "/calGp.h5" for ms in MSs_shift.getListStr()],
                    [
                        parset_dir+'/losoto-ampnorm-scalar.parset',
                        parset_dir+'/losoto-plot2d.parset', 
                        parset_dir+'/losoto-plot.parset'
                        
                    ]
                )

            # predict in MSs
            logger.info("Peel - Predict final...")
            for model_file in glob.glob(imagename_peel + "*model.fits"):
                lib_img.blank_image_reg(
                    model_file, peel_region_file, blankval=0.0, inverse=True
                )
                
            s.add(
                f"wsclean -predict -name {imagename_peel} \
                    -j {s.max_processors} -channels-out 2 -reorder\
                    -parallel-reordering 4 {MSs_shift.getStrWsclean()}",  # was MSs
                log="wsclean-pre.log",
                commandType="wsclean",
                processors="max",
            )
            s.run(check=True)

            # corrupt
            MSs_shift.run( # was MSs
                f"DP3 {parset_dir}/DP3-cor.parset msin=$pathMS \
                    msin.datacolumn=MODEL_DATA msout.datacolumn=MODEL_DATA \
                    cor.invert=False cor.parmdb=cal-Gp-peel_{name}.h5 cor.correction=phase000",
                log="$nameMS_corrupt.log",
                commandType="DP3",
            )
            
            # TEMPORARY CALL
            MSs_shift.addcol("CORRECTED_DATA", "DATA")
            MSs_shift.run( # was MSs
                'taql "update $pathMS set CORRECTED_DATA = DATA"',
                log="$nameMS_taql.log",
                commandType="general",
            )

            # subtract
            logger.info(
                "Subtract model: CORRECTED_DATA = CORRECTED_DATA - MODEL_DATA..."
            )
            MSs_shift.run( # was MSs
                'taql "update $pathMS set CORRECTED_DATA = CORRECTED_DATA - MODEL_DATA"',
                log="$nameMS_taql.log",
                commandType="general",
            )
            
            
            
            # image
            logger.info("Peel - Image...")
            lib_util.run_wsclean(
                s,
                "wsclean-test-2.log",
                MSs_shift.getStrWsclean(),  # was MSs
                name="img/test_2_%s" % (name),
                size=512,
                parallel_gridding=4,
                baseline_averaging="",
                scale="2.5arcsec",
                niter=100000,
                no_update_model_required="",
                minuv_l=30,
                mgain=0.4,
                nmiter=0,
                auto_threshold=5,
                local_rms="",
                local_rms_method="rms-with-min",
                join_channels="",
                fit_spectral_pol=2,
                channels_out=2,
            )
        # DONE
        
    sys.exit()
    
    with WALKER.if_todo("reprepare dataset"):
        # blank models
        logger.info("Cleanup model images...")
        for model_file in glob.glob(imagename + "*model.fits"):
            lib_img.blank_image_reg(model_file, beam07reg, blankval=0.0, inverse=True)

        # ft models
        s.add(
            f"wsclean -predict -name {imagename} -j {s.max_processors} -channels-out 2 \
                -reorder -parallel-reordering 4 {MSs.getStrWsclean()}",
            log="wsclean-pre.log",
            commandType="wsclean",
            processors="max",
        )
        s.run(check=True)

        # prepare new data
        logger.info("Subtract model: DATA = CORRECTED_DATA + MODEL_DATA...")
        MSs.run(
            'taql "update $pathMS set DATA = CORRECTED_DATA + MODEL_DATA"',
            log="$nameMS_taql.log",
            commandType="general",
        )
    # DONE