#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys, os, glob
import numpy as np
import pyregion

sys.path.append("/data/scripts/LiLF")

import pipelines.pipeline3C.pipeline_utils as pipeline
from pyregion.parser_helper import Shape
from astropy.table import Table as AstroTab



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
        do_predict=predict,
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

def peel_single_source_original(MSs_shift, s, name, peel_region_file):
    # image
    logger.info("Peel - Image...")
    imagename_peel = "peel-%s/img_%s" % (name, name)
    
    #image and predict source to peel
    image_quick(MSs_shift, imagename_peel, data_column="DATA")
    
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
    
    image_quick(MSs_shift, f'peel-{name}/test-corrupted-model-{name}', predict=False)
    
    '''
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
    '''    
    
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
    
    name_test = imagename.split("/")[-1]    
    # predict the source to peel
    logger.info("Peel - Predict init...")
    SCHEDULE.add(
        f"wsclean -predict -name {name_test} \
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

def subtract_model(MSs: MeasurementSets, col_in: str = "CORRECTED_DATA", col_out: str = "CORRECTED_DATA"):
    logger.info(f"Subtract model: {col_out} = {col_in} - MODEL_DATA...")
    MSs.run(
        f'taql "update $pathMS set {col_out} = {col_in} - MODEL_DATA"',
        log="$nameMS_taql.log",
        commandType="general",
    )
    
def add_model(MSs: MeasurementSets, col_in: str = "CORRECTED_DATA", col_out: str = "CORRECTED_DATA"):
    logger.info(f"Add model back: {col_out} = {col_in} + MODEL_DATA...")
    MSs.run(
        f'taql "update $pathMS set {col_out} = {col_in} + MODEL_DATA"',
        log="$nameMS_taql.log",
        commandType="general",
    )

def peel(original_mss: MeasurementSets, s: lib_util.Scheduler, peel_max: int = 2, original: bool = True, do_test: bool = False):
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
        
        # subtract model of entire field
        subtract_model(MSs)
        
        
    table = load_sky(MSs, region)
    central_sources = table[table["dist"]<0.1]
    satellite_sources = table[table["dist"] >= 0.1]
    phase_center = MSs.getListObj()[0].getPhaseCentre()
    
    n_to_peel = peel_max
    if len(satellite_sources) < peel_max:
        n_to_peel = len(satellite_sources)
    
    for i, peel_source in enumerate(satellite_sources):
        if i + 1 > n_to_peel:
            break
        
        name = str(peel_source["Source_id"])
        imagename_peel = "peel-" + name + "/" + IMAGENAME.split("/")[-1]
        
        #dist = lib_util.distanceOnSphere(phase_center[0], phase_center[1], peel_source["RA"], peel_source["DEC"])

        logger.info(f"Peeling {name} ({peel_source['Total_flux']:.1f} Jy)")
        with WALKER.if_todo("peel-%s" % name):
            os.system("mkdir peel-%s" % name)

            # predict and blank model to the source to peel
            set_model_to_peel_source(MSs, peel_source, imagename_peel)
            
            # add the source to peel back
            add_model(MSs)
            
            if do_test:
                image_quick(MSs, f'peel-{name}/test-add-{name}')
            
            lib_util.check_rm("mss-dir")
            os.makedirs("mss-dir")
            
            # phaseshift + avg
            logger.info("Peel - Phaseshift+avg...")
            MSs.run(
                f"DP3 {parset_dir}/DP3-shiftavg.parset msin=$pathMS \
                    msout=mss-dir/$nameMS.MS msin.datacolumn=CORRECTED_DATA \
                    msout.datacolumn=DATA avg.timestep=8 avg.freqstep=16 \
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
            
            peel_region_file = f"peel-{name}/{name}.reg"
            if original:
                peel_single_source_original(MSs_shift, s, name, peel_region_file)
            else:
                pass
                #peel_single_source(MSs_shift, s, name, peel_region_file, do_test=True)
    
    
    
    sys.exit()
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
    
if __name__ == "__main__":
    Logger_obj = lib_log.Logger('pipeline-3c-peel.logger')
    logger = lib_log.logger
    SCHEDULE = lib_util.Scheduler(log_dir=Logger_obj.log_dir, dry=False)
    WALKER = lib_util.Walker('pipeline-3c-peel.walker')
    
    # parse parset
    parset = lib_util.getParset()
    parset_dir = parset.get("LOFAR_3c_core", "parset_dir")

    TARGET = os.getcwd().split("/")[-1]
    
    original_mss = MeasurementSets(
        glob.glob(f'*.MS-phaseup'), 
        SCHEDULE, 
        check_flags=False, 
        check_sun=True
    ) 
    
    peel(original_mss, SCHEDULE, peel_max=2, original=True)