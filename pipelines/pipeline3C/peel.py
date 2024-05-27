#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys, os, glob
import numpy as np
import pyregion

from pyregion.parser_helper import Shape
from astropy.table import Table as AstroTab

sys.path.append("/data/scripts/LiLF")

from LiLF_lib import lib_img, lib_util, lib_log
from LiLF_lib.lib_ms import AllMSs as MeasurementSets
import pipelines.pipeline3C.pipeline_utils as pipeline


def image_quick(
    measurements: MeasurementSets, 
    imagename: str, 
    data_column: str="CORRECTED_DATA",
    predict: bool=True,
    empty: bool = False,
    wide=False
):
    if empty:
        niter = 0
    else:
        niter = 100000
        
    if wide:
        size = 2500
        scale = "10arcsec"
        taper_gaussian = "30arcsec"
    else:
        size = 512
        scale = "2.5arcsec"
        taper_gaussian = "2.5arcsec"
        
    logger.info(f'imaging {imagename}... ')
    lib_util.run_wsclean(
        SCHEDULE,
        "wsclean-peel.log",
        measurements.getStrWsclean(),
        do_predict=predict,
        name=imagename,
        data_column=data_column,
        size=size,
        parallel_gridding=4,
        baseline_averaging="",
        scale=scale,
        niter=niter,
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

def clean_peeling_dir(msets: MeasurementSets):
    lib_util.check_rm("*.MS*peel")
    lib_util.check_rm("peel-*")
    
    logger.info("copying data to -> *-peel...")
    for ms in msets.getListStr():
        os.system('cp -r %s %s' % (ms, ms + "-peel") )
        
        
def predict_field(msets: MeasurementSets, region_file: str|None, imagenameM: str, predict: bool = True):
    imagename_pre = imagenameM + "pre"
    
    # Low res image
    logger.info("Cleaning wide 1...")
    lib_util.run_wsclean(
        msets.scheduler,
        "wsclean-wide.log",
        msets.getStrWsclean(),
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
    
    # makemask
    im = lib_img.Image(imagename_pre + "-MFS-image.fits", userReg=region_file)
    im.makeMask(mode="default", threshpix=5, rmsbox=(50, 5))
    maskfits = imagename_pre + "-mask.fits"

    logger.info("Cleaning wide 2...")
    lib_util.run_wsclean(
        msets.scheduler,
        "wsclean-wide.log",
        msets.getStrWsclean(),
        name=imagenameM,
        do_predict=predict,
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


def load_sky(imagename: str, phasecentre: tuple, region: str|None):
    # load skymodel
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
    
    table = AstroTab.read(mask_ddcal.replace("fits", "cat.fits"), format="fits")
    table = table[np.where(table["Total_flux"] > 8)] # all 3c sources ar at least 10 Jy 
    table.sort(["Total_flux"], reverse=True)
    
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


def isolate_source_model(msets: MeasurementSets, source, imagename: str, predict:bool=True):
    name = str(source["Source_id"])
    peel_region_file = f"peel-{name}/{name}.reg"
    imagename_peel = f"peel-{name}/{imagename.split('/')[-1]}"
    
    lib_util.check_rm(peel_region_file)
    
    # make a region to model
    peel_region_file = f"peel-{name}/{name}.reg"
    sh = Shape("circle", None)
    sh.coord_format = "fk5"  # type: ignore
    sh.coord_list = [source["RA"], source["DEC"], 0.075] # ra, dec, diam # type: ignore
    sh.attr = ( # type: ignore
        [], 
        {
            "width": "2", 
            "point": "cross", 
            "font": '"helvetica 16 normal roman"'
        }
    )  
    sh.comment = 'color=red text="%s"' % (name + ".reg")  # type: ignore
    pyregion.ShapeList([sh]).write(peel_region_file)
    
    # copy and blank models
    logger.info(f"copy model fits to ./peel-{name}")
    os.system(f"cp {imagename}*model.fits peel-{name}")
    
    for model_file in glob.glob(imagename_peel + "*model.fits"):
        lib_img.blank_image_reg(
            model_file, peel_region_file, blankval=0.0, inverse=True
        )
    
    if predict: 
        logger.info("Peel - Predict init...")
        msets.scheduler.add(
            f"wsclean -predict -name {imagename_peel} \
                -j {msets.scheduler.max_processors} -channels-out 2 -reorder \
                -parallel-reordering 4 {msets.getStrWsclean()}",
            log="wsclean-pre.log",
            commandType="wsclean",
            processors="max",
        )
        msets.scheduler.run(check=True)


def phaseshift_to_source(msets: MeasurementSets, source, column_in:str="DATA") -> MeasurementSets:
    logger.info("Peel - Phaseshift+avg...")
    msets.run(
        f"DP3 {parset_dir}/DP3-shiftavg.parset msin=$pathMS \
            msout=mss-dir/$nameMS.MS msin.datacolumn={column_in} \
            msout.datacolumn=DATA avg.timestep=8 avg.freqstep=8 \
            shift.phasecenter=[{source['RA']}deg,{source['DEC']}deg]",
        log="$nameMS_shift.log",
        commandType="DP3",
    )
    return MeasurementSets(glob.glob("mss-dir/*.MS"), SCHEDULE)


def calibrate(msets: MeasurementSets, mode: str, name: str):
    if mode == "scalar":
        logger.info("Peel - Calibrate scalarphase...")
        msets.run(
            "DP3 "
            + parset_dir
            + "/DP3-solG.parset msin=$pathMS msin.datacolumn=DATA \
                sol.h5parm=$pathMS/calGp-peel.h5 sol.mode=scalarphase \
                sol.solint=1 sol.smoothnessconstraint=1e6",
            log="$nameMS_solGp-peel.log",
            commandType="DP3",
        )
        
        lib_util.run_losoto(
            msets.scheduler,
            f"Gp-peel_{name}",
            [ms + "/calGp-peel.h5" for ms in msets.getListStr()],
            [
                parset_dir + "/losoto-plot2d.parset",
                parset_dir + "/losoto-plot.parset",
            ],
            plots_dir=f"peel-{name}",
        )
        
        logger.info('Correction PH...')
        command = f'DP3 {parset_dir}/DP3-cor.parset msin=$pathMS msin.datacolumn=DATA \
            cor.parmdb=cal-Gp-peel_{name}.h5 cor.correction=phase000' 
        msets.run(
            command, 
            log=f'$nameMS_corGph-peel-{name}.log', 
            commandType='DP3'
        )
        return f"cal-Gp-peel_{name}.h5"
    
    
    
    elif mode == "fulljones":
        logger.info("Peel - Calibrate fulljones...")
        msets.run(
            "DP3 "
            + parset_dir
            + "/DP3-solG.parset msin=$pathMS msin.datacolumn=CORRECTED_DATA \
                sol.h5parm=$pathMS/calGa.h5 sol.mode=fulljones \
                sol.solint=50 ",
            log="$nameMS_solGa-peel.log",
            commandType="DP3",
        )
        
        lib_util.run_losoto(
            msets.scheduler,
            f"Ga-peel_{name}",
            [ms + "/calGa.h5" for ms in msets.getListStr()],
            [
                parset_dir + "/losoto-ampnorm-full-diagonal.parset",
                parset_dir + "/losoto-plot2d.parset",
                parset_dir+'/losoto-plot2d-pol.parset', 
                parset_dir+'/losoto-plot-pol.parset'
            ],
            plots_dir=f"peel-{name}",
        )
        
        logger.info('Correction slow AMP+PH...')
        command = f'DP3 {parset_dir}/DP3-cor.parset msin=$pathMS msin.datacolumn=CORRECTED_DATA \
            cor.parmdb=cal-Ga-peel_{name}.h5 cor.correction=fulljones \
            cor.soltab=[amplitude000,phase000]'  
        msets.run(
            command,
            log=f'$nameMS_corGa-peel-{name}.log', 
            commandType='DP3'
        )
        return f"cal-Ga-peel_{name}.h5"    


def corrupt(msets: MeasurementSets, solution: str):
    # corrupt
    correction=""
    if "Gp" in solution:
        correction = "phase000"
        soltab = ""
    elif "Ga" in solution:
        correction = "fulljones"
        soltab = "cor.soltab=[amplitude000,phase000]"    
        
    msets.run( # was MSs
        f"DP3 {parset_dir}/DP3-cor.parset msin=$pathMS \
            msin.datacolumn=MODEL_DATA msout.datacolumn=MODEL_DATA \
            cor.invert=False cor.parmdb={solution} \
            cor.correction={correction} {soltab}",
        log="$nameMS_corrupt.log",
        commandType="DP3",
    )



def subtract_model(
    msets: MeasurementSets, column_in: str = "DATA", column_out: str = "DATA"
):
    msets.run(
        f'taql "update $pathMS set {column_in} = {column_out} - MODEL_DATA"',
        log="$nameMS_taql.log",
        commandType="general",
    )


def peel(
    msets: MeasurementSets, 
    s: lib_util.Scheduler, 
    peel_max: int = 2, 
    do_test: bool = False
):
    imagename = "img/img-wideM"
    _, beam07reg, region = pipeline.make_beam_region(msets, TARGET)
    
    with WALKER.if_todo("clean-peel"):
        clean_peeling_dir(msets)
        
    mss_peel = MeasurementSets(glob.glob(f'*.MS*peel'), s)
    
    with WALKER.if_todo("predict-field"):    
        predict_field(mss_peel, region, imagename)
    
    phasecentre = mss_peel.getListObj()[0].getPhaseCentre()
    bright_sources = load_sky(imagename, phasecentre, region)
    #central_sources = bright_sources[bright_sources["dist"]<0.1]
    satellite_sources = bright_sources[bright_sources["dist"] >= 0.1]
    
    n_to_peel = peel_max
    if len(bright_sources) < peel_max:
        n_to_peel = len(bright_sources)
    
    for i, source in enumerate(bright_sources):
        if i + 1 > n_to_peel:
            break
        
        name = str(source["Source_id"])
        #imagename_peel = f"peel-{name}/{imagename.split('/')[-1]}"
        
        logger.info(f"Peeling {name} ({source['Total_flux']:.1f} Jy)")
        with WALKER.if_todo(f"peel-{name}"):
            lib_util.check_rm(f"peel-{name}")
            lib_util.check_rm("mss-dir")
            os.makedirs(f"peel-{name}")
            os.makedirs("mss-dir")
            
            solutions = list()
            mss_shift = phaseshift_to_source(mss_peel, source, column_in="DATA")
            
            isolate_source_model(mss_shift, source, imagename)
            
            
            mss_shift.addcol("CORRECTED_DATA", "DATA")
            if do_test:
                image_quick(mss_shift, f"peel-{name}/after-shift-{name}", predict=True, data_column="CORRECTED_DATA")
            
            
            solutions.append(calibrate(mss_shift, "scalar", name))
            solutions.append(calibrate(mss_shift, "fulljones", name))
            
            # we somehow have to subtract the source from mss_peel, with info from mss_shift
            # assume just corrupting works works for that
            if do_test:
                image_quick(mss_shift, f"peel-{name}/data-after-correct", data_column="CORRECTED_DATA", predict=False, empty=True, wide=True)
                image_quick(mss_shift, f"peel-{name}/model-before-corrupt", data_column="MODEL_DATA", predict=False, empty=True)
                image_quick(mss_peel, f"peel-{name}/data-wide-before", data_column="DATA", predict=False, empty=True, wide=True)
                
            
            isolate_source_model(mss_peel, source, imagename)
            
            for sol in solutions:
                corrupt(mss_peel, sol)
            
            if do_test:
                image_quick(mss_peel, f"peel-{name}/model-after-corrupt", data_column="MODEL_DATA", predict=False, empty=True, wide=True)
            
            subtract_model(mss_peel)

            if do_test:
                #image_quick(mss_peel, f"peel-{name}/after-peel", data_column="DATA", predict=False)
                image_quick(mss_peel, f"peel-{name}/data-wide-after", data_column="DATA", predict=False, empty=True, wide=True)
                #predict_field(mss_peel, region, f"peel-{name}/wide-after")
    
    with WALKER.if_todo("image-final"):
        predict_field(mss_peel, region, f"peel-{name}/peel-all")
    
    

    
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
        glob.glob(f'*.MS-phaseup-final'), 
        SCHEDULE, 
        check_flags=False, 
        check_sun=True
    ) 
    
    peel(original_mss, SCHEDULE, peel_max=1, do_test=True)
