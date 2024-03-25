#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys, os, glob
import numpy as np
import pyregion

from shutil import move
from .calibration import SelfCalibration
import pipelines.pipeline3C.pipeline_utils as pipeline
from pyregion.parser_helper import Shape
from astropy.table import Table as AstroTab

sys.path.append("/net/voorrijn/data2/boxelaar/scripts/LiLF")

from LiLF_lib import lib_img, lib_util, lib_log
from LiLF_lib.lib_ms import AllMSs as MeasurementSets

logger = lib_log.logger
WALKER = lib_util.Walker("pipeline-3c.walker")

# parse parset
parset = lib_util.getParset()
parset_dir = parset.get("LOFAR_3c_core", "parset_dir")

TARGET = os.getcwd().split("/")[-1]

def image_quick(measurements: MeasurementSets, imagename: str, data_column: str="CORRECTED_DATA", predict: bool=True, ):
    logger.info(f'imaging {imagename}... ')
    lib_util.run_wsclean(
        SCHEDULE,
        "wsclean-peel.log",
        measurements.getStrWsclean(),
        do_predict=True,
        name=imagename,
        data_column=data_column,
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


def clean_peeling(MSs):
    lib_util.check_rm("*.MS*peel")
    logger.info("copying data to -> *-peel...")
    for measurement in MSs.getListStr():
        os.system('cp -r %s %s' % (measurement, measurement + "-peel") )
        
    lib_util.check_rm("peel-*")
    
    
def get_field_model(MSs: MeasurementSets, region):
    imagename_pre = "img/img-wide"
    imagenameM = "img/img-wideM"
    
    # Low res image
    logger.info("Cleaning wide 1...")
    lib_util.run_wsclean(
        SCHEDULE,
        "wsclean-wide.log",
        MSs.getStrWsclean(),
        name=imagename_pre,
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
    im = lib_img.Image(imagename_pre + "-MFS-image.fits", userReg=region)
    im.makeMask(mode="default", threshpix=5, rmsbox=(50, 5))
    maskfits = imagename_pre + "-mask.fits"

    logger.info("Cleaning wide 2...")
    lib_util.run_wsclean(
        SCHEDULE,
        "wsclean-wide.log",
        MSs.getStrWsclean(),
        name=imagenameM,
        do_predict=True,
        parallel_gridding=4,
        baseline_averaging="",
        size=2500,
        reuse_psf=imagename_pre,
        reuse_dirty=imagename_pre,
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

    
    
def peel_single_source(MSs_shift, peel_source, do_test: bool = False):
    cal = SelfCalibration(MSs_shift, (None, None), SCHEDULE)
    name = str(peel_source["Source_id"])
    imagename_peel = "peel-%s/img_%s" % (name, name)
    
    image_quick(cal.mss, imagename_peel, data_column="DATA")
    
    # Align MODEL_DATA with source to peel
    set_model_to_peel_source(cal.mss, peel_source, imagename_peel)
    
    cal.solve_gain("scalar", solint=1)
    cal.solve_gain("fulljones", solint=16)
    
    move(f'cal-Gp-c{cal.cycle:02d}-{cal.stats}-ampnorm.h5', f'peel-{name}/')
    move(f'plots-Gp-c{cal.cycle:02d}-{cal.stats}-ampnorm', f'peel-{name}/')
    
    move(f'cal-Ga-c{cal.cycle:02d}-{cal.stats}-ampnorm.h5', f'peel-{name}/')
    move(f'plots-Ga-c{cal.cycle:02d}-{cal.stats}-ampnorm', f'peel-{name}/')
    
    image_quick(cal.mss, f'peel-{name}/test-pre-{name}')
    
    sol_suffix = f'c{cal.cycle:02d}-{cal.stats}-ampnorm'
    
    logger.info(f'Scalarphase corruption... ({name})')
    cal.mss.run(
        f'DP3 {parset_dir}/DP3-cor.parset msin=$pathMS msin.datacolumn=MODEL_DATA \
            msout.datacolumn=CORRUPT_MODEL_DATA cor.updateweights=False \
            cor.parmdb=peel-{name}/cal-Gp-{sol_suffix}.h5 \
            cor.correction=phase000 cor.invert=False',
        log=f'$nameMS_corrupt_Gp-{sol_suffix}.log', 
        commandType='DP3'
    )

    logger.info(f'Full-Jones corruption... ({name})')
    cal.mss.run(
        f'DP3 {parset_dir}/DP3-cor.parset msin=$pathMS msin.datacolumn=CORRUPT_MODEL_DATA \
            msout.datacolumn=CORRUPT_MODEL_DATA cor.correction=fulljones \
            cor.parmdb=peel-{name}/cal-Ga-{sol_suffix}.h5 \
            cor.soltab=[amplitude000,phase000] cor.invert=False', 
        log=f'$nameMS_corrupt_Ga-{sol_suffix}.log', 
        commandType='DP3'
    )
    
    logger.info(f'SET SUBTRACTED_DATA = {cal.data_column} - CORRUPT_MODEL_DATA ({name})')
    cal.mss.addcol('SUBTRACTED_DATA', cal.data_column)
    cal.mss.run(
        f'taql "UPDATE $pathMS SET SUBTRACTED_DATA = {cal.data_column} - CORRUPT_MODEL_DATA"',
        log='$nameMS_taql_subtract.log',
        commandType='general'
    )
    '''
    logger.info(f'Scalarphase correction... ({name})')
    cal.mss.run(
        f'DP3 {parset_dir}/DP3-cor.parset msin=$pathMS \
            msin.datacolumn=SUBTRACTED_DATA msout.datacolumn=SUBTRACTED_DATA \
            cor.parmdb=peel-{name}/cal-Gp-{sol_suffix}.h5 \
            cor.updateweights=False cor.correction=phase000',
        log=f'$nameMS_cor_Gp_subtracted-{sol_suffix}.log', 
        commandType='DP3'
    )
    '''
    if do_test:        
        logger.info('Test empty... (SUBTRACTED_DATA)')
        lib_util.run_wsclean(
            SCHEDULE, 
            f'wsclean-peel.log', 
            cal.mss.getStrWsclean(), 
            weight='briggs -0.5',
            data_column='SUBTRACTED_DATA', 
            channels_out=3,
            name=f'peel-{name}/test-corrupted-model-{name}', 
            scale='2.0arcsec', 
            size=2000, 
            niter=10000, 
            nmiter=0,
            no_update_model_required='', 
            minuv_l=30 
        )
    
    
def set_model_to_peel_source(MSs, peel_source, imagename, make_region: bool =True):
    name = str(peel_source["Source_id"])
    imagename_peel = "peel-" + name + "/" + imagename.split("/")[-1]
    peel_region_file = f"peel-{name}/{name}.reg"
    
    if make_region:
        # make a region
        
        sh = Shape("circle", None)
        sh.coord_format = "fk5"  # type: ignore
        sh.coord_list = [peel_source["RA"], peel_source["DEC"], 0.075] # ra, dec, diam # type: ignore
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
    
    for model_file in glob.glob(imagename_peel + "*model.fits"):
        lib_img.blank_image_reg(
            model_file, peel_region_file, blankval=0.0, inverse=True
        ) 
        
    # predict the source to peel
    logger.info("Peel - Predict init...")
    SCHEDULE.add(
        f"wsclean -predict -name {imagename_peel} \
            -j {SCHEDULE.max_processors} -channels-out 2 -reorder \
            -parallel-reordering 4 {MSs.getStrWsclean()}",
        log="wsclean-pre.log",
        commandType="wsclean",
        processors="max",
    )
    SCHEDULE.run(check=True)
    
    
def load_sky(MSs: MeasurementSets, region: str|None):
    # load skymodel
    full_image = lib_img.Image(IMAGENAME + "-MFS-image.fits", userReg=region)
    mask_ddcal = full_image.imagename.replace(".fits", "_mask-ddcal.fits")  # this is used to find calibrators
    
    full_image.makeMask(
        mode="default",
        threshpix=5,
        atrous_do=False,
        maskname=mask_ddcal,
        write_srl=True,
        write_ds9=True,
    )
    
    table = AstroTab.read(mask_ddcal.replace("fits", "cat.fits"), format="fits")
    table = table[np.where(table["Total_flux"] > 10)]
    table.sort(["Total_flux"], reverse=True)
    
    phasecentre = MSs.getListObj()[0].getPhaseCentre()
    distances = list()
    for peel_source in table:
        distances.append(
            lib_util.distanceOnSphere(
                phasecentre[0], 
                phasecentre[1], 
                peel_source["RA"], 
                peel_source["DEC"]
            )
        )
    
    table["dist"] = distances
    return table


def peel(original_mss: MeasurementSets, s: lib_util.Scheduler, peel_max: int = 2, original: bool = True):
    global SCHEDULE
    global IMAGENAME
    SCHEDULE = s
    
    IMAGENAME = "img/img-wideM"
    mask = pipeline.make_beam_region(original_mss, TARGET)
    _, beam07reg, region = mask
    #cal = SelfCalibration(MSs, mask=mask, schedule=s)   
    
    with WALKER.if_todo("clean-peel"):
        clean_peeling(original_mss)
        
    MSs = MeasurementSets(
        glob.glob(f'*.MS*peel'), 
        s, 
        check_flags=False, 
        check_sun=True
    )  
  
    # get model for entire sub-field
    with WALKER.if_todo("sub-field"):
        get_field_model(MSs, region)   
        
        
    table = load_sky(MSs, region)
    central_sources = table[table["dist"]<0.1]
    satellite_sources = table[table["dist"] >= 0.1]
    phase_center = MSs.getListObj()[0].getPhaseCentre()
    
    n_to_peel = peel_max
    if len(satellite_sources) < peel_max:
        n_to_peel = len(satellite_sources)
    
    #First, subtract central source
    for peel_source in central_sources:
        name = str(peel_source["Source_id"])
        os.system("mkdir peel-%s" % name)
        
        logger.info(f"Peeling central source {name} ({peel_source['Total_flux']:.1f} Jy)")
        with WALKER.if_todo("peel-%s" % name):
            set_model_to_peel_source(MSs, peel_source, IMAGENAME)
            
            image_quick(MSs, f'peel-{name}/test-central-{name}')
            
            fulljones_solution = sorted(glob.glob("cal-Ga*core-ampnorm.h5"))
            solution = sorted(glob.glob("cal-Gp*core.h5"))
            #logger.info(f"Correction Gain of {fulljones_solution[-1]}")
            # correcting CORRECTED_DATA -> CORRECTED_DATA
            
            logger.info(f'Scalarphase corruption... ({name})')
            MSs.run(
                f'DP3 {parset_dir}/DP3-cor.parset msin=$pathMS msin.datacolumn=MODEL_DATA \
                    msout.datacolumn=CORRUPT_MODEL_DATA cor.updateweights=False \
                    cor.parmdb={solution[-1]} \
                    cor.correction=phase000 cor.invert=False',
                log=f'$nameMS_corrupt_Gp.log', 
                commandType='DP3'
            )

            logger.info(f'Full-Jones corruption... ({name})')
            MSs.run(
                f'DP3 {parset_dir}/DP3-cor.parset msin=$pathMS msin.datacolumn=CORRUPT_MODEL_DATA \
                    msout.datacolumn=CORRUPT_MODEL_DATA cor.correction=fulljones \
                    cor.parmdb={fulljones_solution[-1]} \
                    cor.soltab=[amplitude000,phase000] cor.invert=False', 
                log=f'$nameMS_corrupt_Ga.log', 
                commandType='DP3'
            )
            
            logger.info(
                "Subtract model s0 (central source): \
                    CORRECTED_DATA = CORRECTED_DATA - CORRUPT_MODEL_DATA..."
                )
            MSs.run(
                'taql "update $pathMS set CORRECTED_DATA = CORRECTED_DATA - CORRUPT_MODEL_DATA"',
                log="$nameMS_taql.log",
                commandType="general",
            )
            
            image_quick(MSs, f'peel-{name}/test-subtract-central-{name}')
            
    sys.exit()

      
    # cycle on sources to peel
    for i, peel_source in enumerate(satellite_sources):
        name = str(peel_source["Source_id"])

        logger.info(f"Peeling {name} ({peel_source['Total_flux']:.1f} Jy)")
        with WALKER.if_todo("peel-%s" % name):
            os.system("mkdir peel-%s" % name)
            
            # predict and blank model to the source to peel
            set_model_to_peel_source(MSs, peel_source, IMAGENAME)
            

            lib_util.check_rm("mss-dir")
            os.makedirs("mss-dir")
            
            # phaseshift
            logger.info("Peel - Phaseshift...")
            MSs.run(
                f"DP3 {parset_dir}/DP3-shift.parset msin=$pathMS \
                    msin.datacolumn=CORRECTED_DATA \
                    msout=mss-dir/$nameMS.MS msout.datacolumn=DATA \
                    shift.phasecenter=[{peel_source['RA']}deg,{peel_source['DEC']}deg]",
                log="$nameMS_shift.log",
                commandType="DP3",
            )
            
            MSs_shift = MeasurementSets(
                glob.glob("mss-dir/*.MS"), 
                s, 
                check_flags=False, 
                check_sun=True
            )
            
            peel_single_source(MSs_shift, peel_source, do_test=True)
            
            # phaseshift back to central source
            logger.info("Peel - inverse-Phaseshift...")
            MSs_shift.run(
                f"DP3 {parset_dir}/DP3-shift.parset msin=$pathMS \
                    msin.datacolumn=CORRECTED_DATA \
                    msout=mss-dir/$nameMS-inverse.MS msout.datacolumn=DATA \
                    shift.phasecenter=[{phase_center[0]}deg,{phase_center[1]}deg]",
                log="$nameMS_shift.log",
                commandType="DP3",
            )
            
            
            if i + 1 == n_to_peel:
                # all sources peeled
                break
            else:
                # subtract peeled source 
                pass
            
    # add central Sources back
    for peel_source in central_sources:
        name = str(peel_source["Source_id"])
        logger.info(f"adding central source {name} back ({peel_source['Total_flux']:.1f} Jy)")
        with WALKER.if_todo("add-%s" % name):
            set_model_to_peel_source(MSs, peel_source, IMAGENAME)
            
            logger.info(
                "adding model s0 (central source) back: \
                    CORRECTED_DATA = CORRECTED_DATA + MODEL_DATA..."
                )
            MSs.run(
                'taql "update $pathMS set CORRECTED_DATA = CORRECTED_DATA + MODEL_DATA"',
                log="$nameMS_taql.log",
                commandType="general",
            )


    sys.exit()
    with WALKER.if_todo("reprepare dataset"):
        # blank models
        logger.info("Cleanup model images...")
        for model_file in glob.glob(IMAGENAME + "*model.fits"):
            lib_img.blank_image_reg(model_file, beam07reg, blankval=0.0, inverse=True)

        # ft models
        s.add(
            f"wsclean -predict -name {IMAGENAME} -j {s.max_processors} -channels-out 2 \
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
    
    #remove obsolete mss-dir
    lib_util.check_rm("mss-dir")