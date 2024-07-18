#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys, os, glob
import lsmtool #type: ignore 
import numpy as np
import argparse

from astroquery.ipac.ned import Ned
from astropy.coordinates import SkyCoord
import astropy.units as u

#from argparse import ArgumentParser
#sys.path.append("/data/scripts/LiLF")

from LiLF_lib import lib_util as lilf, lib_log
from LiLF_lib.lib_ms import AllMSs as MeasurementSets

import pipeline3C as pipeline

def get_argparser() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='3C-pipeline [options]')
    parser.add_argument('-t', '--target', dest="target", type=str, default=os.getcwd().split('/')[-1], help="Target name")
    parser.add_argument('-d', '--data_dir', dest="data_dir", type=str, default="/home/iranet/groups/lofar/j.boxelaar/data/3CsurveyIS/tgts/", help="Data directory excluding target name")
    parser.add_argument('-s', '--stations', dest="stations", nargs='+', type=str, default=["core", "dutch", "int"], help="Stations to process")
    parser.add_argument('-cc', '--cycles_core', dest='total_cycles_core', type=int, default=None)
    parser.add_argument('-ca', '--cycles_all', dest='total_cycles_all', type=int, default=None)
    parser.add_argument('-m', '--manual_mask', dest='manual_mask', action='store_true', default=False)
    parser.add_argument('--do_core_scalar_solve', dest='do_core_scalar_solve', action='store_true', default=False)
    parser.add_argument('--do_test', dest='do_test', action='store_true', default=False)
    parser.add_argument('--no_phaseup', dest='no_phaseup', action='store_true', default=False)
    parser.add_argument('--bl_smooth_fj', dest='bl_smooth_fj', action='store_true', default=False)
    parser.add_argument('--smooth_all_pols', dest='smooth_all_pols', action='store_true', default=False)
    parser.add_argument('--scalar_only', dest='scalar_only', action='store_true', default=False)
    parser.add_argument('--no_fulljones', dest='no_fulljones', action='store_true', default=False)
    parser.add_argument("--apply_beam", dest="apply_beam", action="store_true", default=False)
    parser.add_argument("--use_own_model", dest="use_own_model", action="store_true", default=False)
    return parser.parse_args()

    
def run_test(measurements: MeasurementSets) -> None:
    # Do diagonal solve to check core stations are corrected perfectly before phasing up 
    for measurement in measurements.getListStr():
        os.system('cp -r %s %s' % (measurement, measurement + "-test") )
        
    test_mss = MeasurementSets(
        glob.glob(f'*.MS-test'), 
        SCHEDULE, 
        check_flags=False, 
        check_sun=True
    )    
    
    predict(test_mss, doBLsmooth=True)
    
    Logger.info(f'Solving diagonal test...')
    test_mss.run(
        f'DP3 {parset_dir}/DP3-solG.parset msin=$pathMS \
            msin.datacolumn=SMOOTHED_DATA sol.mode=diagonal \
            sol.h5parm=$pathMS/cal-diag.h5 \
            sol.solint=15 sol.smoothnessconstraint=1e6',
        log=f'$nameMS_sol-diag_test.log', 
        commandType="DP3"
    )
    
    lilf.run_losoto(
        SCHEDULE, 
        f'diag-test', 
        [f'{ms}/cal-diag.h5' for ms in test_mss.getListStr()],
        [
            parset_dir+'/losoto-clip.parset', 
            parset_dir+'/losoto-plot2d.parset', 
            parset_dir+'/losoto-plot2d-pol.parset', 
            parset_dir+'/losoto-plot-pol.parset',
        ]  
    )
    
    for ms in test_mss.getListStr():
        lilf.check_rm(ms)

def get_cal_dir(timestamp: str, logger = None, int_data = False) -> list:
    """
    Get the proper cal directory from a timestamp
    """
    dirs = list()
    if int_data:
        cal_directories = sorted(glob.glob('/home/iranet/groups/lofar/j.boxelaar/data/icals/3c*'))
    else:
        cal_directories = sorted(glob.glob('/home/local/work/j.boxelaar/data/3Csurvey/cals/3c*'))
        
    for cal_dir in cal_directories:
        calibrator = cal_dir.split("/")[-1]
        cal_timestamps = set()
        for ms in glob.glob(cal_dir+'/20*/data-bkp/*MS'):
            cal_timestamps.add("_".join(ms.split("/")[-1].split("_")[:2]))
            
        if f"{calibrator}_t{timestamp}" in cal_timestamps:
            if logger is not None:
                logger.info('Calibrator found: %s (t=%s)' % (cal_dir, timestamp))
            if int_data:
                dirs.append(f"{cal_dir}/{timestamp}")
            else:
                dirs.append(f"{cal_dir}/{timestamp[:8]}/solutions")
        else:
            pass
        
    if dirs == []:
        if logger is not None:
            logger.error('Missing calibrator.')
        sys.exit()
    return dirs  

def correct_from_callibrator(MSs: MeasurementSets, timestamp: str) -> None:
    cal_dir = get_cal_dir(timestamp, logger=Logger)
    cal_dir_is = get_cal_dir(timestamp, logger=Logger, int_data=INT_DATA)
    
    using_dir = cal_dir[-1]
    using_dir_is = cal_dir_is[-1]
    for dir in cal_dir:
        if TARGET in dir:
            using_dir = dir
    
    Logger.info(f"Using cal: {using_dir_is}")
        
    h5_pa = using_dir_is + '/cal-pa.h5'
    h5_amp = using_dir_is + '/cal-bp.h5'
    h5_iono = using_dir_is + '/cal-iono.h5'
    h5_iono_cs = using_dir_is + '/cal-iono-cs.h5'
    h5_fr = using_dir_is + '/cal-fr.h5'
    assert os.path.exists(h5_pa)
    assert os.path.exists(h5_amp)
    assert os.path.exists(h5_iono)
    assert os.path.exists(h5_fr)
    
    
    # Apply cal sol - SB.MS:DATA -> SB.MS:CORRECTED_DATA (polalign corrected)
    Logger.info('Apply solutions (pa)...')
    MSs.run(
        f'DP3 {parset_dir}/DP3-cor.parset msin=$pathMS msin.datacolumn=DATA msout.datacolumn=DATA\
            cor.parmdb={h5_pa} cor.correction=polalign', 
        log='$nameMS_cor1_pa.log', 
        commandType='DP3'
    )
    
    # Beam correction CORRECTED_DATA -> CORRECTED_DATA (polalign corrected, beam corrected+reweight)
    Logger.info('Beam correction (beam)...')
    MSs.run(
        'DP3 '+parset_dir+'/DP3-beam.parset msin=$pathMS msin.datacolumn=DATA \
            msout.datacolumn=DATA corrbeam.updateweights=True', 
        log='$nameMS_cor1_beam.log', 
        commandType='DP3'
    )
    
    # Correct amp BP CORRECTED_DATA -> CORRECTED_DATA
    Logger.info('BP correction...')
    MSs.run(
        f'DP3 {parset_dir}/DP3-cor.parset msin=$pathMS cor.parmdb={h5_amp} \
            msin.datacolumn=DATA msout.datacolumn=DATA \
            cor.correction=amplitudeSmooth cor.updateweights=False',
        log='$nameMS_corBP.log', 
        commandType="DP3"
    )
    
    # Correct FR CORRECTED_DATA -> CORRECTED_DATA
    Logger.info('Faraday rotation correction...')
    MSs.run(
        f'DP3 {parset_dir}/DP3-cor.parset msin=$pathMS cor.parmdb={h5_fr} \
            msin.datacolumn=DATA msout.datacolumn=DATA cor.correction=rotationmeasure000', 
        log='$nameMS_corFR2.log', 
        commandType="DP3"
    )
    
    # Correct iono concat_all:CORRECTED_DATA -> CORRECTED_DATA
    Logger.info('Iono correction...')
    MSs.run(
        f"DP3 {parset_dir}/DP3-cor.parset msin=$pathMS cor.parmdb={h5_iono_cs} \
            msin.datacolumn=DATA msout.datacolumn=DATA cor.correction=phase000", 
        log='$nameMS_corIONO_CS.log', 
        commandType="DP3"
    )
    MSs.run(
        f"DP3 {parset_dir}/DP3-cor.parset msin=$pathMS cor.parmdb={h5_iono} \
            msin.datacolumn=DATA msout.datacolumn=DATA \
            cor.correction=phase000",
        log='$nameMS_corIONO.log', 
        commandType="DP3"
    )




        
def get_cal_dir_IS(timestamp: str, logger = None, int_data = False) -> list:
    """
    Get the proper cal directory from a timestamp
    """
    dirs = list()
    if int_data:
        cal_directories = sorted(glob.glob('/home/iranet/groups/lofar/j.boxelaar/data/icals/3c*'))
    else:
        cal_directories = sorted(glob.glob('/data/data/3Csurvey/cals/3c*'))
        
    for cal_dir in cal_directories:
        calibrator = cal_dir.split("/")[-1]
        cal_timestamps = set()
        for ms in glob.glob(cal_dir+'/20*/data-bkp/*MS'):
            cal_timestamps.add("_".join(ms.split("/")[-1].split("_")[:2]))
            
        if f"{calibrator}_t{timestamp}" in cal_timestamps:
            if logger is not None:
                logger.info('Calibrator found: %s (t=%s)' % (cal_dir, timestamp))
            dirs.append(f"{cal_dir}/{timestamp}")
            #dirs.append(f"{cal_dir}/{timestamp}/solutions")
        else:
            pass
        
    if dirs == []:
        if logger is not None:
            logger.error('Missing calibrator.')
        sys.exit()
    return dirs  
'''
def correct_from_callibrator(MSs: MeasurementSets, timestamp: str) -> None:
    cal_dir = get_cal_dir(timestamp, logger=Logger, int_data=INT_DATA)
    
    using_dir = cal_dir[0]
    for dir in cal_dir:
        if TARGET in dir:
            using_dir = dir
            Logger.info(f"Using cal: {using_dir}")
        elif "3c196" in dir:
            using_dir = dir
            Logger.info(f"Using cal: {using_dir}")
        elif "3c380" in dir:
            using_dir = dir
            Logger.info(f"Using cal: {using_dir}")
        
    h5_pa = using_dir + '/cal-pa.h5'
    h5_bp = using_dir + '/cal-bp.h5'
    h5_iono_cs = using_dir + '/cal-iono-cs.h5'
    h5_iono = using_dir + '/cal-iono.h5'
    h5_fr = using_dir + '/cal-fr.h5'
    assert os.path.exists(h5_pa), f"Missing {h5_pa}"
    assert os.path.exists(h5_bp), f"Missing {h5_bp}"
    assert os.path.exists(h5_iono_cs), f"Missing {h5_iono_cs}"
    assert os.path.exists(h5_iono), f"Missing {h5_iono}"
    assert os.path.exists(h5_fr), f"Missing {h5_fr}"
    
    
    ## Pol align correction concat_all-phaseup.MS:DATA -> CORRECTED_DATA
    Logger.info('Polalign correction...')
    MSs.run(
        f'DP3 {parset_dir}/DP3-cor.parset msin=$pathMS msin.datacolumn=DATA \
            cor.parmdb={h5_pa} cor.correction=polalign', 
        log='$nameMS_corPA.log', 
        commandType="DP3"
    )
    
    # Correct beam concat_all-phaseup.MS:CORRECTED_DATA -> CORRECTED_DATA
    Logger.info('Beam correction...')
    MSs.run(
        f"DP3 {parset_dir}/DP3-beam.parset msin=$pathMS corrbeam.updateweights=True", 
        log='$nameMS_beam.log', 
        commandType="DP3"
    )
    
    # Correct bp concat_all:CORRECTED_DATA -> DATA
    Logger.info('Bandpass correction...')
    MSs.run(
        f'DP3 {parset_dir}/DP3-cor.parset msin=$pathMS \
            msin.datacolumn=CORRECTED_DATA cor.parmdb={h5_bp} \
            cor.correction=amplitudeSmooth cor.updateweights=True', 
        log='$nameMS_cor1_bp.log', 
        commandType='DP3'
    )

    # Correct iono concat_all:CORRECTED_DATA -> CORRECTED_DATA
    Logger.info('Iono correction...')
    MSs.run(
        f"DP3 {parset_dir}/DP3-cor.parset msin=$pathMS cor.parmdb={h5_iono_cs} \
            msin.datacolumn=CORRECTED_DATA cor.correction=phase000", 
        log='$nameMS_corIONO_CS.log', 
        commandType="DP3"
    )
    MSs.run(
        f"DP3 {parset_dir}/DP3-cor.parset msin=$pathMS cor.parmdb={h5_iono} \
            msin.datacolumn=CORRECTED_DATA msout.datacolumn=DATA \
            cor.correction=phase000",
        log='$nameMS_corIONO.log', 
        commandType="DP3"
    )
    
    empty_clean(MSs, f"img/{timestamp}-img-empty-int")

'''
def align_phasecenter(MSs: MeasurementSets, timestamp) -> str:
    phasecenter = MSs.getListObj()[0].getPhaseCentre()
    phasecenter = SkyCoord(
        ra=phasecenter[0], 
        dec=phasecenter[1], 
        unit=(u.deg, u.deg), 
        frame='fk4'
    )
    
    table = Ned.query_object(TARGET)
    target_coord = SkyCoord(
        ra = float(table["RA"]),  # type: ignore
        dec = float(table["DEC"]),  # type: ignore
        unit = (u.deg, u.deg), 
        frame = 'fk4'
    )
    del table
    
    seperation = phasecenter.separation(target_coord).arcmin

    if seperation > 5: #type: ignore
        Logger.info(f"Source is {seperation:.1f} arcmin away from phasecenter. aligning phases to source")
        MSs.run(
            f"DP3 {parset_dir}/DP3-shift.parset \
                msin=$pathMS msout=$pathMS-shift \
                msin.datacolumn=DATA msout.datacolumn=DATA \
                shift.phasecenter=[{target_coord.ra.deg}deg,{target_coord.dec.deg}deg]", #type: ignore
            log="$nameMS_shift.log",
            commandType="DP3",
        )
        
        return f'{TARGET}_t{timestamp}_concat_int.MS-shift'
    else:
        Logger.info(f"Source is {seperation:.1f} arcmin away from phasecenter. No phase shift needed")
        return f'{TARGET}_t{timestamp}_concat_int.MS'


def setup() -> None:
    global INT_DATA
    MSs_list = MeasurementSets( 
        sorted(glob.glob(DATA_DIR+'/data/*MS')), 
        SCHEDULE, 
        check_flags=False
    ).getListStr()
    
    for timestamp in list(set([ os.path.basename(ms).split('_')[1][1:] for ms in MSs_list ])):
        mss_toconcat = sorted(glob.glob(f'{DATA_DIR}/data/{TARGET}_t{timestamp}_SB*.MS'))
        MS_concat_bkp = f'{TARGET}_t{timestamp}_concat.MS-bkp'
        
        MS_concat_core = f'{TARGET}_t{timestamp}_concat_core.MS'
        MS_concat_dutch = f'{TARGET}_t{timestamp}_concat_dutch.MS'
        MS_concat_int = f'{TARGET}_t{timestamp}_concat_int.MS'
        
        #if mss_toconcat.hasIS():
        INT_DATA = True
    
        if os.path.exists(MS_concat_bkp): 
            Logger.info('Restoring bkp data: %s...' % MS_concat_bkp)
            lilf.check_rm(MS_concat_core)
            lilf.check_rm(MS_concat_dutch)
            lilf.check_rm(MS_concat_int)
            os.system('cp -r %s %s' % (MS_concat_bkp, MS_concat_int) )
    
        else:
            Logger.info('Making %s...' % MS_concat_int)
            SCHEDULE.add(
                f'DP3 {parset_dir}/DP3-avg.parset msin=\"{str(mss_toconcat)}\" \
                    msin.baseline="*&" msout={MS_concat_int} avg.freqstep=2 \
                    avg.timestep=2',
                log=MS_concat_int+'_avg.log', 
                commandType='DP3'
            )
            SCHEDULE.run(check=True, maxThreads=1)                
    
            MSs = MeasurementSets([MS_concat_int], SCHEDULE)
            
            # flag bad stations, and low-elev
            flagging(MSs)
            
            #demix A-team sources if needed
            demix(MSs)
            
            # Correct data from calibrator step (pa, amp, beam, iono)
            correct_from_callibrator(MSs, timestamp)
            
            #align phases to source if mismatch >5 arcmin
            #try: MS_concat_int = align_phasecenter(MSs, timestamp)
            #except:
            #    Logger.warning("No Internet connection, skipping phasecenter alignment")
            
            # bkp
            Logger.info('Making backup...')
            os.system('cp -r %s %s' % (MS_concat_int, MS_concat_bkp) ) # do not use MS.move here as it resets the MS path to the moved one
        
    
        Mss = MeasurementSets([MS_concat_int], SCHEDULE)

        Logger.info('Splitting data in Core and Remote...')
        Mss.run(
            f'DP3 {parset_dir}/DP3-concat.parset msin=$pathMS msout={MS_concat_core} filter.baseline="CS*&&" ',
            log="$nameMS_split.log", 
            commandType="DP3"
        )
        Mss.run(
            f'DP3 {parset_dir}/DP3-concat.parset msin=$pathMS msout={MS_concat_dutch} filter.baseline="[CR]S*&&"',
            log="$nameMS_split.log", 
            commandType="DP3"
        )
    
    

def phaseup(MSs: MeasurementSets, stats: str, do_test: bool = True, sol_dir:str="") -> MeasurementSets:
    data_in = "DATA"
    
    Logger.info('Correcting CS...')
    fulljones_solution = sorted(glob.glob(sol_dir+"cal-Ga*core-ampnorm.h5"))
    solution = sorted(glob.glob(sol_dir+"cal-Gp*core-ampnorm.h5"))
    final_cycle_sol = 0
    final_cycle_fj = 0
    
    if len(solution) != 0 or len(fulljones_solution) != 0:
        rms_history = np.loadtxt(f'{sol_dir}rms_noise_history_core.csv', delimiter=",")
        ratio_history = np.loadtxt(f'{sol_dir}mm_ratio_noise_history_core.csv', delimiter=",")
        
        assert len(rms_history) == len(ratio_history)
        if np.argmax(ratio_history) == 0:
            correct_cycle = 1
        elif np.argmin(rms_history) == len(rms_history) - 1 and np.argmax(ratio_history) == len(ratio_history) - 1:
            correct_cycle = - 1
        else:
            correct_cycle = np.argmax(ratio_history) - len(ratio_history)
    
    if len(solution) != 0:
        final_cycle_sol = int(solution[-1].split("-")[2][1:])
    if len(fulljones_solution) != 0:
        final_cycle_fj = int(fulljones_solution[-1].split("-")[2][1:])
        

    if final_cycle_sol >= final_cycle_fj  and  final_cycle_sol > 0:
        Logger.info(f"correction Gain-scalar of {solution[correct_cycle]}")
        # correcting CORRECTED_DATA -> CORRECTED_DATA
        MSs.run(
            f'DP3 {parset_dir}/DP3-cor.parset msin=$pathMS msin.datacolumn={data_in} \
                cor.parmdb={solution[correct_cycle]} cor.correction=phase000',
            log='$nameMS_corPH-core.log', 
            commandType='DP3'
        )
        data_in = "CORRECTED_DATA"
    else:
        Logger.warning(f"No phase Core corrections found. Phase-up not recommended")
    
    
    if final_cycle_fj >= final_cycle_sol and final_cycle_fj > 0:
        Logger.info(f"Correction Gain of {fulljones_solution[correct_cycle]}")
        # correcting CORRECTED_DATA -> CORRECTED_DATA
        MSs.run(
            f'DP3 {parset_dir}/DP3-cor.parset msin=$pathMS msin.datacolumn={data_in} \
                cor.parmdb={fulljones_solution[correct_cycle]} cor.correction=fulljones \
                cor.soltab=[amplitude000,phase000]',
            log='$nameMS_corAMPPHslow-core.log', 
            commandType='DP3'
        )
        data_in = "CORRECTED_DATA"
    else:
        Logger.warning(f"No fulljones Core corrections found. Phase-up not recommended")
    
    if do_test:
        run_test(MSs)
        
        
    if stats == "dutch":
        if not args.no_phaseup:
            source_angular_diameter = 0.
            baseline = "CS*"
            stations = "{SuperStLBA:'%s'}" % baseline
            
            # Phaseup CORRECTED_DATA -> DATA
            Logger.info('Phasing up all Core Stations...')
            Logger.debug('Phasing up: '+ baseline)
            lilf.check_rm(f'*{stats}.MS-phaseup')
        
            MSs.run(
                f"DP3 {parset_dir}/DP3-phaseup.parset msin=$pathMS \
                    msin.datacolumn={data_in} msout=$pathMS-phaseup \
                    msout.datacolumn=DATA stationadd.stations={stations} filter.baseline=!{baseline}",       
                log=f'$nameMS_phaseup.log', 
                commandType="DP3"
            )
        else:
            try:
                source_angular_diameter = pipeline.source_angular_size(TARGET)
                baseline = pipeline.stations_to_phaseup(source_angular_diameter, central_freq=57.9) 
                stations = "{SuperStLBA:["+baseline+"]}"
                Logger.info(f"Using adaptive phase-up. Source diameter (arcmin): {source_angular_diameter}")
            except:
                source_angular_diameter = 0.
                baseline = "CS00[2-7]*"
                stations = "{SuperStLBA:'%s'}" % baseline
                Logger.info(f"Not using adaptive phase-up. Phasing up all core stations")

            # Phaseup CORRECTED_DATA -> DATA
            Logger.info('Phasing up Core Stations...')
            Logger.debug('Phasing up: '+ baseline)
            lilf.check_rm(f'*{stats}.MS-phaseup')
            
            if baseline == "" or args.no_phaseup:
                for MS in MSs.getListStr():
                    os.system(f'cp -r {MS} {MS}-phaseup')
                    
                MSs_phaseup = MeasurementSets(
                    glob.glob(f'*concat_{stats}.MS-phaseup'), SCHEDULE
                )
                
                MSs_phaseup.run(
                    'taql "update $pathMS set DATA = CORRECTED_DATA"',
                    log='$nameMS_taql.log', 
                    commandType='general'
                )
            
            else:
                MSs.run(
                    f"DP3 {parset_dir}/DP3-phaseup.parset msin=$pathMS \
                        msin.datacolumn={data_in} msout=$pathMS-phaseup \
                        msout.datacolumn=DATA stationadd.stations={stations} filter.baseline=!{baseline}",       
                    log=f'$nameMS_phaseup.log', 
                    commandType="DP3"
                )
        
        os.system(f'rm -r *concat_dutch.MS')
        os.system(f'rm -r *concat_core.MS')
    
    
    elif stats == "int":
        dutch_sol_ph = sorted(glob.glob(sol_dir+"cal-Gp*dutch-ampnorm.h5"))[-2]
        dutch_sol_fj = sorted(glob.glob(sol_dir+"cal-Ga*dutch-ampnorm.h5"))[-2]
        
        if os.path.exists(dutch_sol_ph):
            Logger.info(f"correction Gain-scalar of {dutch_sol_ph}")
            MSs.run(  
                f'DP3 {parset_dir}/DP3-cor.parset msin=$pathMS msin.datacolumn={data_in} \
                    cor.parmdb={dutch_sol_ph} cor.correction=phase000',
                log='$nameMS_corPH-core.log', 
                commandType='DP3'
            )
            data_in = "CORRECTED_DATA"
        else:
            Logger.warning(f"No phase Core corrections found.")
        
        
        if os.path.exists(dutch_sol_fj):
            Logger.info(f"Correction Gain of {dutch_sol_fj}")
            MSs.run(
                f'DP3 {parset_dir}/DP3-cor.parset msin=$pathMS msin.datacolumn={data_in} \
                    cor.parmdb={dutch_sol_fj} cor.correction=fulljones \
                    cor.soltab=[amplitude000,phase000]',
                log='$nameMS_corAMPPHslow-core.log', 
                commandType='DP3'
            )
            data_in = "CORRECTED_DATA"
        else:
            Logger.warning(f"No fulljones Core corrections found.")
        
        
        source_angular_diameter = 0.
        baseline = "CS*"
        stations = "{SuperStLBA:'%s'}" % baseline
        
        # Phaseup CORRECTED_DATA -> DATA
        Logger.info('Phasing up all Core Stations...')
        Logger.debug('Phasing up: '+ baseline)
        lilf.check_rm(f'*{stats}.MS-phaseup')
    
        MSs.run(
            f"DP3 {parset_dir}/DP3-phaseup.parset msin=$pathMS \
                msin.datacolumn={data_in} msout=$pathMS-phaseup \
                msout.datacolumn=DATA stationadd.stations={stations} filter.baseline=!{baseline}",       
            log=f'$nameMS_phaseup.log', 
            commandType="DP3"
        )
        os.system(f'rm -r *concat_{stats}.MS')
            
    MSs = MeasurementSets(
        glob.glob(f'*concat_{stats}.MS-phaseup'), 
        SCHEDULE, 
        check_flags=False, 
        check_sun=True
    )
    return MSs          
    
def flagging(MSs: MeasurementSets) -> None:
    # flag bad stations, and low-elev
    Logger.info('Flagging...')
    MSs.run('DP3 '+parset_dir+'/DP3-flag.parset msin=$pathMS msout=. \
                aoflagger.strategy='+parset_dir+'/LBAdefaultwideband.lua ant.baseline=\"'+bl2flag+'\"',
                log='$nameMS_flag.log', commandType='DP3')

def demix(MSs: MeasurementSets):
    for ateam in ['VirA', 'TauA', 'CygA', 'CasA']:
        sep = MSs.getListObj()[0].distBrightSource(ateam)
        Logger.info(f'{ateam} - sep: {sep:.0f} deg')
        
        if sep > 2 and sep < 25 and (ateam != 'CasA' and ateam != 'CygA'): # type: ignore
            # CasA and CygA are already demixed in preprocessing
            Logger.warning(f'Demix of {ateam} (sep: {sep:.1f} deg)')
            
            for MS in MSs.getListStr():
                lilf.check_rm(MS + '/' + os.path.basename(SKYDB_DEMIX))
                os.system(f'cp -r {SKYDB_DEMIX} {MS}/{os.path.basename(SKYDB_DEMIX)}')

            Logger.info('Demixing...')
            MSs.run(
                f'DP3 {parset_dir}/DP3-demix.parset msin=$pathMS msout=$pathMS \
                    demixer.skymodel=$pathMS/{os.path.basename(SKYDB_DEMIX)} \
                    demixer.instrumentmodel=$pathMS/instrument_demix \
                    demixer.subtractsources=[{ateam }]',
                log='$nameMS_demix.log', 
                commandType='DP3'
            )


def predict(MSs: MeasurementSets, doBLsmooth:bool = True) -> None:
    Logger.info('Preparing model...')
    sourcedb = 'tgts.skydb'
    #if not os.path.exists(sourcedb):
    phasecentre = MSs.getListObj()[0].getPhaseCentre()
    fwhm = MSs.getListObj()[0].getFWHM(freq='min')
    radeg = phasecentre[0]
    decdeg = phasecentre[1]
    
    if TARGET in ["3c196", "3c380", "3c295"]:
        os.system(f"cp /data/scripts/LiLF/models/calib-simple.skydb {sourcedb}")
        os.system(f"cp /data/scripts/LiLF/models/calib-simple.skymodel tgts.skymodel")
        calname = MSs.getListObj()[0].getNameField()
        
        # Predict MODEL_DATA
        Logger.info('Predict (DP3)...')
        MSs.run(
            f'DP3 {parset_dir}/DP3-predict.parset msin=$pathMS pre.usebeammodel=true pre.sourcedb={sourcedb} pre.sources={calname}', 
            log='$nameMS_pre.log', 
            commandType='DP3'
        )
        
    elif TARGET == "3c274":
        os.system(f"cp /data/data/3Csurvey/tgts/3c274/VirLow.skymodel tgts.skymodel")
        os.system('makesourcedb outtype="blob" format="<" in=tgts.skymodel out=tgts.skydb')
        
        # Predict MODEL_DATA
        Logger.info('Predict (DP3)...')
        MSs.run(
            f'DP3 {parset_dir}/DP3-predict.parset msin=$pathMS pre.usebeammodel=true pre.sourcedb={sourcedb} pre.sources=VirA', 
            log='$nameMS_pre.log', 
            commandType='DP3'
        )
        
    else: 
        if not os.path.exists('tgts.skydb'):   
            if args.use_own_model:
                try:
                    os.system(f"cp /home/local/work/j.boxelaar/data/3Csurvey/tgts/{TARGET}/{TARGET}.skymodel tgts.skymodel")
                    lsm = lsmtool.load('tgts.skymodel')#, beamMS=MSs.getListStr()[0])
                except:
                    Logger.error(f"Missing manual model for {TARGET}")
                    os.system('wget -O tgts.skymodel "https://lcs165.lofar.eu/cgi-bin/gsmv1.cgi?coord=%f,%f&radius=%f&unit=deg"' % (radeg, decdeg, fwhm)) # ASTRON
                    lsm = lsmtool.load('tgts.skymodel')#, beamMS=MSs.getListStr()[0])
            else:
                # get model the size of the image (radius=fwhm/2)
                os.system('wget -O tgts.skymodel "https://lcs165.lofar.eu/cgi-bin/gsmv1.cgi?coord=%f,%f&radius=%f&unit=deg"' % (radeg, decdeg, fwhm)) # ASTRON
                lsm = lsmtool.load('tgts.skymodel')#, beamMS=MSs.getListStr()[0])
                
            
            lsm.remove('I<0.5')
            #lsm.write('tgts-beam.skymodel', applyBeam=True, clobber=True) #TODO Beam is still not used?
            lsm.write('tgts.skymodel', applyBeam=False, clobber=True)
            os.system('makesourcedb outtype="blob" format="<" in=tgts.skymodel out=tgts.skydb')
        
        # Predict MODEL_DATA
        Logger.info('Predict (DP3)...')
        MSs.run(
            f'DP3 {parset_dir}/DP3-predict.parset msin=$pathMS pre.usebeammodel=true pre.sourcedb={sourcedb}', 
            log='$nameMS_pre.log', 
            commandType='DP3'
        )
    
    if doBLsmooth:
        # Smooth DATA -> DATA
        Logger.info('BL-based smoothing...')
        MSs.run(
            'BLsmooth_pol.py -d -r -s 0.8 -i DATA -o SMOOTHED_DATA $pathMS', 
            log='$nameMS_smooth1.log', 
            commandType='python'
        )

def clean_specific(mode: str) -> None :
    Logger.info('Cleaning ' + mode + ' dirs...')
    lilf.check_rm(f'cal*{mode}-ampnorm.h5')
    lilf.check_rm(f'plots*{mode}-ampnorm')
    lilf.check_rm('peel*')
    lilf.check_rm(f'img/img-{mode}*')
    lilf.check_rm(f'*.MS-phaseup-final')
    
    if not os.path.exists("img/"):
        os.makedirs('img')
        

    
def main(args: argparse.Namespace) -> None:
    stopping=False
    setup()
    for stations in args.stations:
        
        
        try:
            MSs = MeasurementSets(
                sorted(glob.glob(f'*concat_{stations}.MS')), 
                SCHEDULE, 
                check_flags=False
            )   
        except:
            pass
        '''
        calibration = pipeline.SelfCalibration(
            MSs, 
            schedule=SCHEDULE, 
            total_cycles=5, 
            mask = pipeline.make_beam_region(MSs, TARGET), 
            stats=stations,
            target=TARGET
        )
        
        calibration.empty_clean(f"img/{TARGET}-img-empty-int")
        sys.exit()
        '''
        if stations == "dutch" or stations == "int": 
            with WALKER.if_todo('phaseupCS ' + stations):
                MSs = phaseup(MSs, stations, do_test=args.do_test)#, sol_dir='/home/local/work/j.boxelaar/data/3Csurvey/tgts/3c401/')
            
            MSs = MeasurementSets(
                glob.glob(f'*concat_{stations}.MS-phaseup'), 
                SCHEDULE, 
                check_flags=False, 
                check_sun=True
            )
            
        with WALKER.if_todo(f"clean_{stations}"):
            clean_specific(stations)  
            
        # make beam region files
        masking = pipeline.make_beam_region(MSs, TARGET, parset_dir)
        
        rms_noise_pre = np.inf
        mm_ratio_pre = 0
        
        if stations == "core":
            if args.total_cycles_core is None:
                total_cycles = 5
            else:
                total_cycles = args.total_cycles_core
            
        elif stations == "dutch":
            if args.total_cycles_all is None:
                total_cycles = 20
            else:
                total_cycles = args.total_cycles_all
        else:
            total_cycles = 10

        calibration = pipeline.SelfCalibration(
            MSs, 
            schedule=SCHEDULE, 
            total_cycles=total_cycles, 
            mask=masking, 
            stats=stations,
            target=TARGET
        )
        
        if args.no_phaseup:
            calibration.phased_up = False
        else:
            calibration.phased_up = True 
            
        # Predict model    
        with WALKER.if_todo('predict_' + stations):  
            if stations == "int":
                #predict initial model from imaging step
                calibration.clean(f"img/{TARGET}-img-predict")
                #calibration.predict_from_img("img/3c401-image.fits")
            else:
                predict(MSs, doBLsmooth=False)
            
        for cycle in calibration:
            with WALKER.if_todo(f"cal_{stations}_c{cycle}"):
                
                if stations == "core":
                    if cycle == 1 or args.do_core_scalar_solve:
                        calibration.solve_gain('phase') 
                    
                    if not args.scalar_only and cycle > 1:
                        calibration.solve_gain(
                            "fulljones", 
                            bl_smooth_fj=args.bl_smooth_fj, 
                            smooth_all_pols=args.smooth_all_pols
                        )    
                    
                elif stations == "dutch":
                    if cycle > 15 and not calibration.doamp:
                        calibration.doamp = True
                        Logger.info("Amplitude solve activated")
                           
                    if calibration.doph:
                        calibration.solve_gain('scalar')
                    
                    if calibration.doamp and not args.scalar_only and cycle > 1:
                        if args.no_fulljones:
                            Logger.info("No fulljones, scalar amplitude solve")
                            calibration.solve_gain("amplitude")
                        else:
                            calibration.solve_gain(
                                'fulljones', 
                                bl_smooth_fj=args.bl_smooth_fj, 
                                smooth_all_pols=args.smooth_all_pols
                            )
                            
                elif stations == "int":
                    calibration.solve_gain('phase')
                    
                    #if cycle > 1 and calibration.doamp:
                    #    calibration.solve_gain('amplitude')
                            

            with WALKER.if_todo(f"image-{stations}-c{cycle}" ):
                #calibration.empty_clean(f"img/img-empty-c{cycle}")
                
                imagename = f'img/img-{stations}-{cycle:02d}'
                try:
                    calibration.clean(imagename, apply_beam=args.apply_beam, uvlambdamin=100)
                    rms_noise_pre, mm_ratio_pre, stopping = calibration.prepare_next_iter(imagename, rms_noise_pre, mm_ratio_pre)
                except RuntimeError:
                    Logger.error(f"Failed to clean {imagename}")
                    rms_noise_pre, mm_ratio_pre, stopping = np.inf, 0, True
                    calibration.rms_history.append(rms_noise_pre)
                    calibration.ratio_history.append(mm_ratio_pre)
                    break
                
            #if stopping or cycle == calibration.stop:
            #    Logger.info("Start Peeling")                
            #    #pipeline.peel(peel_mss, calibration.s)
            #    break
            
        with WALKER.if_todo(f"save_{stations}_history"):
            np.savetxt(
                f'rms_noise_history_{stations}.csv', 
                np.asarray(calibration.rms_history), 
                delimiter=",", 
                header="rms noise after every calibration cycle (Jy/beam)"
            )
            np.savetxt(
                f'mm_ratio_noise_history_{stations}.csv', 
                np.asarray(calibration.ratio_history), 
                delimiter=",", 
                header="mm ratio  after every calibration cycle"
            )
            '''
            if stations == "dutch":
                pipeline.rename_final_images(sorted(glob.glob('img/img-all-*')), target = TARGET) 
            
                Logger.info(f"Saving model to {TARGET}.skymodel")
                os.system(f"python /data/scripts/revoltek-scripts/fits2sky.py \
                    img/{TARGET}-img-final-MFS {TARGET}.skymodel --ref_freq 57.7e6 \
                    --fits_mask img/img-all-01-mask.fits --min_peak_flux_jy 0.005"
                )
            '''
        #if stations == "dutch":
        #    calibration.clean(f"img/{TARGET}-img-deep", deep=True)
        #    calibration.low_resolution_clean("img/img-low")   
        
        
    
    
    # copy the calibrated measurementsets into final file 
    try:
        MSs.run(
            f"DP3 {parset_dir}/DP3-avg.parset msin=$pathMS \
                msin.datacolumn=CORRECTED_DATA msout=$pathMS-final \
                msout.datacolumn=DATA",       
            log=f'$nameMS_final.log', 
            commandType="DP3"
        )
    except:
        pass
                          
    Logger.info("Done.")
    
    
def empty_clean(msets: MeasurementSets, imagename: str, uvlambdamin: int = 30):
    kwargs1 = {'weight': 'briggs -0.8', "size": 2500, "scale": "2.0arcsec"} # type: ignore
    
    Logger.info('Cleaning empty')
    lilf.run_wsclean(
        msets.scheduler, 
        'wsclean-empty.log', 
        msets.getStrWsclean(),
        name=imagename,
        parallel_gridding=4, 
        baseline_averaging='',
        niter=0, 
        no_update_model_required='',
        minuv_l=uvlambdamin, 
        mgain=0.4, 
        nmiter=1,
        auto_threshold=5, 
        local_rms='', 
        local_rms_method='rms-with-min',
        join_channels='', 
        fit_spectral_pol=2, 
        channels_out=2, 
        **kwargs1
    )


if __name__ == "__main__":
    args = get_argparser()
    TARGET = args.target
    DATA_DIR = args.data_dir + args.target
    
    print(args.data_dir, args.target, DATA_DIR)
    
    os.chdir(DATA_DIR)
    
    Logger_obj = lib_log.Logger('pipeline-3c.logger')
    Logger = lib_log.logger
    SCHEDULE = lilf.Scheduler(log_dir=Logger_obj.log_dir, dry=False)
    WALKER = lilf.Walker('pipeline-3c.walker')

    # parse parset
    parset = lilf.getParset()
    parset_dir = parset.get('LOFAR_3c_int', 'parset_dir')
    SKYDB_DEMIX = parset.get('LOFAR_demix','demix_model')
    bl2flag = parset.get('flag', 'stations')
    
    Logger.info("Executing python call:" + ' '.join(sys.argv))

    
    if not os.path.exists(DATA_DIR+"/data"):
        os.makedirs(DATA_DIR+"/data")
        os.system(f"mv {DATA_DIR}/*.MS {DATA_DIR}/data/")
    
    main(args) 
    
