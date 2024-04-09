#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys, os, glob
import lsmtool #type: ignore 
import numpy as np
import argparse

#from argparse import ArgumentParser
sys.path.append("/net/voorrijn/data2/boxelaar/scripts/LiLF")

from LiLF_lib import lib_util as lilf, lib_log
from LiLF_lib.lib_ms import AllMSs as MeasurementSets

import pipeline3C as pipeline


#TARGET = os.getcwd().split('/')[-1]
#DATA_DIR = f'/net/voorrijn/data2/boxelaar/data/3Csurvey/tgts/{TARGET}'

def get_argparser() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='3C-pipeline [options]')
    parser.add_argument('-t', '--target', dest="target", type=str, default=os.getcwd().split('/')[-1], 
                        help="Target name")
    parser.add_argument('-d', '--data_dir', dest="data_dir", type=str, default="/net/voorrijn/data2/boxelaar/data/3Csurvey/tgts/", 
                        help="Data directory excluding target name")
    parser.add_argument('-s', '--stations', dest="stations", nargs='+', type=str, default=["core", "all"])
    parser.add_argument('-c', '--cycles', dest='total_cycles', type=int, default=None)
    parser.add_argument('--do_test', dest='do_test', action='store_true', default=False)
    parser.add_argument('--no_phaseup', dest='no_phaseup', action='store_true', default=False)
    parser.add_argument('--bl_smooth_fj', dest='bl_smooth_fj', action='store_true', default=False)
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

def correct_from_callibrator(MSs: MeasurementSets, timestamp: str) -> None:
    cal_dir = pipeline.get_cal_dir(timestamp, logger = Logger)[0]
    h5_pa = cal_dir + '/cal-pa.h5'
    h5_amp = cal_dir + '/cal-amp.h5'
    h5_iono = cal_dir + '/cal-iono.h5'
    h5_fr = cal_dir + '/cal-fr.h5'
    assert os.path.exists(h5_pa)
    assert os.path.exists(h5_amp)
    assert os.path.exists(h5_iono)
    assert os.path.exists(h5_fr)
    
    
    # Apply cal sol - SB.MS:DATA -> SB.MS:CORRECTED_DATA (polalign corrected)
    Logger.info('Apply solutions (pa)...')
    MSs.run(
        f'DP3 {parset_dir}/DP3-cor.parset msin=$pathMS msin.datacolumn=DATA\
            cor.parmdb={h5_pa} cor.correction=polalign', 
        log='$nameMS_cor1_pa.log', 
        commandType='DP3'
    )
    
    # Apply cal sol - SB.MS:CORRECTED_DATA -> SB.MS:CORRECTED_DATA (polalign corrected, calibrator corrected+reweight, beam corrected+reweight)
    Logger.info('Apply solutions (amp)...')
    MSs.run(
        f'DP3 {parset_dir}/DP3-cor.parset msin=$pathMS \
            msin.datacolumn=CORRECTED_DATA cor.parmdb={h5_amp} \
            cor.correction=amplitudeSmooth cor.updateweights=True', 
        log='$nameMS_cor1_amp.log', 
        commandType='DP3'
    )
    
    # Beam correction CORRECTED_DATA -> CORRECTED_DATA (polalign corrected, beam corrected+reweight)
    Logger.info('Beam correction (beam)...')
    MSs.run(
        'DP3 '+parset_dir+'/DP3-beam.parset msin=$pathMS msin.datacolumn=CORRECTED_DATA \
            msout.datacolumn=CORRECTED_DATA corrbeam.updateweights=True', 
        log='$nameMS_cor1_beam.log', 
        commandType='DP3'
    )
    
    # Apply cal sol - SB.MS:CORRECTED_DATA -> SB.MS:CORRECTED_DATA
    Logger.info('Apply solutions (iono)...')
    MSs.run(
        f'DP3 {parset_dir}/DP3-cor.parset msin=$pathMS msin.datacolumn=CORRECTED_DATA\
            cor.parmdb={h5_iono} cor.correction=phase000', 
        log='$nameMS_cor1_iono.log', 
        commandType='DP3'
    )

    # Move CORRECTED_DATA -> DATA
    Logger.info('Move CORRECTED_DATA -> DATA...')
    MSs.run(
        'taql "update $pathMS set DATA = CORRECTED_DATA"',
        log='$nameMS_taql.log', 
        commandType='general'
    )
    

def setup() -> None:
    MSs_list = MeasurementSets( 
        glob.glob(DATA_DIR+'/data/*MS'), 
        SCHEDULE, 
        check_flags=False
    ).getListStr()
    
    
    for timestamp in set([ os.path.basename(ms).split('_')[1][1:] for ms in MSs_list ]):
        mss_toconcat = sorted(glob.glob(f'{DATA_DIR}/data/{TARGET}_t{timestamp}_SB*.MS'))
        MS_concat_core = f'{TARGET}_t{timestamp}_concat_core.MS'
        MS_concat_all = f'{TARGET}_t{timestamp}_concat_all.MS'
        MS_concat_bkp = f'{TARGET}_t{timestamp}_concat.MS-bkp'
    
        if os.path.exists(MS_concat_bkp): 
            Logger.info('Restoring bkp data: %s...' % MS_concat_bkp)
            lilf.check_rm(MS_concat_all)
            lilf.check_rm(MS_concat_core)
            os.system('cp -r %s %s' % (MS_concat_bkp, MS_concat_all) )
    
        else:
            Logger.info('Making %s...' % MS_concat_all)
            SCHEDULE.add(
                f'DP3 {parset_dir}/DP3-avg.parset msin=\"{str(mss_toconcat)}\" msin.baseline="*&" msout={MS_concat_all}',
                log=MS_concat_all+'_avg.log', 
                commandType='DP3'
            )
            SCHEDULE.run(check=True, maxThreads=1)                
    
            MSs = MeasurementSets([MS_concat_all], SCHEDULE)
            
            demix(MSs)
            
            # Correct data from calibrator step (pa, amp, beam, iono)
            correct_from_callibrator(MSs, timestamp)

            # bkp
            Logger.info('Making backup...')
            os.system('cp -r %s %s' % (MS_concat_all, MS_concat_bkp) ) # do not use MS.move here as it resets the MS path to the moved one
        
        split_stations(
            MeasurementSets([MS_concat_all], SCHEDULE), msout=MS_concat_core
        )

            
def split_stations(measurements: MeasurementSets, msout: str = "", source_angular_diameter: float = 0.):
    """
    if msout == "":
        msout = measurements.getListStr()[0]
        
    if source_angular_diameter == 0.:
        baseline = "CS*&&"
    else:
        baseline = stations_to_phaseup(source_angular_diameter, central_freq=57.9)
    """
    
    Logger.info('Splitting data in Core and Remote...')
    measurements.run(
        f'DP3 {parset_dir}/DP3-filter.parset msin=$pathMS msout={msout}',
        log="$nameMS_split.log", 
        commandType="DP3"
    )
    
    

def phaseup(MSs: MeasurementSets, stats: str, do_test: bool = True) -> MeasurementSets:
    if stats == "all":
        Logger.info('Correcting CS...')
        fulljones_solution = sorted(glob.glob("cal-Ga*core-ampnorm.h5"))
        solution = sorted(glob.glob("cal-Gp*core-ampnorm.h5"))
        final_cycle_sol = 0
        final_cycle_fj = 0
        data_in = "DATA"
        
        if len(solution) != 0:
            final_cycle_sol = int(solution[-1].split("-")[2][1:])
        if len(fulljones_solution) != 0:
            final_cycle_fj = int(fulljones_solution[-1].split("-")[2][1:])

        print(final_cycle_sol, final_cycle_fj)
        print(0 > final_cycle_sol >= final_cycle_fj)

        if final_cycle_sol >= final_cycle_fj > 0:
            Logger.info(f"correction Gain-scalar of {solution[-1]}")
            # correcting CORRECTED_DATA -> CORRECTED_DATA
            MSs.run(
                f'DP3 {parset_dir}/DP3-cor.parset msin=$pathMS msin.datacolumn={data_in} \
                    cor.parmdb={solution[-1]} cor.correction=phase000',
                log='$nameMS_corPH-core.log', 
                commandType='DP3'
            )
            data_in = "CORRECTED_DATA"
        
        
        if final_cycle_fj >= final_cycle_sol > 0:
            Logger.info(f"Correction Gain of {fulljones_solution[-1]}")
            # correcting CORRECTED_DATA -> CORRECTED_DATA
            MSs.run(
                f'DP3 {parset_dir}/DP3-cor.parset msin=$pathMS msin.datacolumn={data_in} \
                    cor.parmdb={fulljones_solution[-1]} cor.correction=fulljones \
                    cor.soltab=[amplitude000,phase000]',
                log='$nameMS_corAMPPHslow-core.log', 
                commandType='DP3'
            )
            data_in = "CORRECTED_DATA"
        
  
        else:
            Logger.warning(f"No Core corrections found. Phase-up not recommended")
        
        if do_test:
            run_test(MSs)
    
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
            MSs.run(
                f"DP3 {parset_dir}/DP3-avg.parset msin=$pathMS \
                    msin.datacolumn={data_in} msout=$pathMS-phaseup \
                    msout.datacolumn=DATA",       
                log=f'$nameMS_phaseup.log', 
                commandType="DP3"
            )
        
        else:
            MSs.run(
                f"DP3 {parset_dir}/DP3-phaseup.parset msin=$pathMS \
                    msin.datacolumn={data_in} msout=$pathMS-phaseup \
                    msout.datacolumn=DATA stationadd.stations={stations} filter.baseline=!{baseline}",       
                log=f'$nameMS_phaseup.log', 
                commandType="DP3"
            )
        
        os.system(f'rm -r *concat_all.MS')
        os.system(f'rm -r *concat_core.MS')
    
    
    elif stats == "def":
        with WALKER.if_todo('phaseupCS_' + stats):
            # Phasing up the cose stations
            # Phaseup CORRECTED_DATA -> DATA
            Logger.info('Phasing up superterp Stations...')
            lilf.check_rm(f'*{stats}.MS-phaseup')
            MSs.run(
                f"DP3 {parset_dir}/DP3-phaseup-def.parset msin=$pathMS \
                    msin.datacolumn=DATA msout=$pathMS-phaseup ", 
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
        os.system(f"cp /net/voorrijn/data2/boxelaar/scripts/LiLF/models/calib-simple.skydb {sourcedb}")
        os.system(f"cp /net/voorrijn/data2/boxelaar/scripts/LiLF/models/calib-simple.skymodel tgts.skymodel")
        calname = MSs.getListObj()[0].getNameField()
        
        # Predict MODEL_DATA
        Logger.info('Predict (DP3)...')
        MSs.run(
            f'DP3 {parset_dir}/DP3-predict.parset msin=$pathMS pre.usebeammodel=true pre.sourcedb={sourcedb} pre.sources={calname}', 
            log='$nameMS_pre.log', 
            commandType='DP3'
        )
        
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
            '/net/voorrijn/data2/boxelaar/scripts/LiLF/scripts/BLsmooth.py\
                -r -s 0.8 -i DATA -o SMOOTHED_DATA $pathMS', 
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
    #with WALKER.if_todo('setup'):
        #set up corected data
    setup() 
    
    for stations in args.stations:
        try:
            MSs = MeasurementSets(
                glob.glob(f'*concat_{stations}.MS'), 
                SCHEDULE, 
                check_flags=False
            )   
        except:
            pass

        if stations != "core": 
            with WALKER.if_todo('phaseupCS ' + stations):
                MSs = phaseup(MSs, stations, do_test=args.do_test)
            
            MSs = MeasurementSets(
                glob.glob(f'*concat_{stations}.MS-phaseup'), 
                SCHEDULE, 
                check_flags=False, 
                check_sun=True
            )
            
        with WALKER.if_todo(f"clean_{stations}"):
            clean_specific(stations)  
            
        # make beam region files
        masking = pipeline.make_beam_region(MSs, TARGET)
        
        # Predict model    
        with WALKER.if_todo('predict_' + stations):  
            predict(MSs, doBLsmooth=False)
        
        rms_noise_pre = np.inf
        mm_ratio_pre = 0
        doslow = True
        
        if stations == "core":
            total_cycles = 3
        elif stations == "all":
            if args.total_cycles is None:
                total_cycles = 14
            else:
                total_cycles = args.total_cycles
        else:
            total_cycles = 10

        calibration = pipeline.SelfCalibration(MSs, schedule=SCHEDULE, total_cycles=total_cycles, mask=masking, stats=stations)
        
        for cycle in calibration:
            #calibration.empty_clean(f"img/img-empty-c{cycle}")
            
            with WALKER.if_todo(f"cal_{stations}_c{cycle}"):
                
                if stations == "core":
                    #if cycle <= 3:
                    calibration.solve_gain('scalar')
                        
                    calibration.solve_gain("fulljones", bl_smooth_fj=args.bl_smooth_fj)
                    
                else:   
                    if calibration.doph:
                        calibration.solve_gain('scalar')
                        
                    if calibration.doamp and doslow: # or (total_cycles - cycle <= 1):
                        calibration.solve_gain('fulljones', bl_smooth_fj=args.bl_smooth_fj)

            with WALKER.if_todo(f"image-{stations}-c{cycle}" ):
                #calibration.empty_clean(f"img/img-empty-c{cycle}")
                
                imagename = f'img/img-{stations}-{cycle:02d}'
                calibration.clean(imagename)
                rms_noise_pre, mm_ratio_pre, stopping = calibration.prepare_next_iter(imagename, rms_noise_pre, mm_ratio_pre)
                
            #if stopping or cycle == calibration.stop:
            #    Logger.info("Start Peeling")                
            #    #pipeline.peel(peel_mss, calibration.s)
            #    break
            
        if stations == "all":
            pipeline.rename_final_images(sorted(glob.glob('img/img-all-*')), target = TARGET)    
            
            calibration.clean(f"img/{TARGET}-img-deep", deep=True)
            calibration.low_resolution_clean("img/img-low")   
        
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
    
def do_peel():
    MSs = MeasurementSets(
        glob.glob(f'*concat_all.MS-phaseup'), 
        SCHEDULE, 
        check_flags=False, 
        check_sun=True
        )
    
    pipeline.peel(MSs, SCHEDULE)
    
    peel_mss = MeasurementSets(
        glob.glob(f'*.MS*peel'), 
        SCHEDULE, 
        check_flags=False, 
        check_sun=True
    )    
    
    with WALKER.if_todo('clean-after-peel'):
        peel_mss.run(
            'taql "update $pathMS set CORRECTED_DATA = DATA"',
            log="$nameMS_taql.log",
            commandType="general",
        )
        
        mask = pipeline.make_beam_region(MSs, TARGET)
        cal = pipeline.SelfCalibration(peel_mss, schedule=SCHEDULE, total_cycles=2, mask=mask)
        cal.clean(f'img/img-after-peeling')
        
def test_clean():
    MSs = MeasurementSets(glob.glob(f'*concat_all.MS-phaseup'), SCHEDULE)
    
    masking = pipeline.make_beam_region(MSs, TARGET)
    calibration = pipeline.SelfCalibration(
        MSs, schedule=SCHEDULE, mask=masking, stats="all"
    )
    
    Logger.info('Correcting CS...')
    fulljones_solution = sorted(glob.glob("cal-Ga*all-ampnorm.h5"))
    solution = sorted(glob.glob("cal-Gp*all-ampnorm.h5"))
    
    if len(solution) != 0:
        Logger.info(f"correction Gain-scalar of {solution[-1]}")
        # correcting DATA -> CORRECTED_DATA
        calibration.mss.run(
            f'DP3 {parset_dir}/DP3-cor.parset msin=$pathMS msin.datacolumn=DATA \
                cor.parmdb={solution[-1]} cor.correction=phase000',
            log='$nameMS_corPH-all.log', 
            commandType='DP3'
        )
    
    if len(fulljones_solution) != 0:
        Logger.info(f"Correction Gain of {fulljones_solution[-1]}")
        # correcting CORRECTED_DATA -> CORRECTED_DATA
        calibration.mss.run(
            f'DP3 {parset_dir}/DP3-cor.parset msin=$pathMS msin.datacolumn=CORRECTED_DATA \
                cor.parmdb={fulljones_solution[-1]} cor.correction=fulljones \
                cor.soltab=[amplitude000,phase000]',
            log='$nameMS_corAMPPHslow-all.log', 
            commandType='DP3'
        )
    
    imagename = f'img/img-clean-test'
    calibration.clean(imagename)
    calibration.prepare_next_iter(imagename, 0, 0)


if __name__ == "__main__":
    args = get_argparser()
    TARGET = args.target
    DATA_DIR = args.data_dir + args.target
    
    os.chdir(DATA_DIR)
    
    Logger_obj = lib_log.Logger('pipeline-3c.logger')
    Logger = lib_log.logger
    SCHEDULE = lilf.Scheduler(log_dir=Logger_obj.log_dir, dry=False)
    WALKER = lilf.Walker('pipeline-3c.walker')

    # parse parset
    parset = lilf.getParset()
    parset_dir = parset.get('LOFAR_3c_core', 'parset_dir')
    SKYDB_DEMIX = parset.get('LOFAR_demix','demix_model')
    bl2flag = parset.get('flag', 'stations')

    
    if not os.path.exists(DATA_DIR+"/data"):
        os.makedirs(DATA_DIR+"/data")
        os.system(f"mv {DATA_DIR}/*.MS {DATA_DIR}/data/")
    
    main(args) 
    #test_clean() 
    #do_peel()      
    
