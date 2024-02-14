#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#from functools import total_ordering
import sys, os, glob
import numpy as np
import lsmtool
import casacore.tables as pt

sys.path.append("/net/voorrijn/data2/boxelaar/scripts/LiLF")

from LiLF_lib import lib_img, lib_util, lib_log
from LiLF_lib.lib_ms import AllMSs as MeasurementSets
from calibration import SelfCalibration

Logger_obj = lib_log.Logger('pipeline-3c.logger')
Logger = lib_log.logger
SCHEDULE = lib_util.Scheduler(log_dir=Logger_obj.log_dir, dry=False)
WALKER = lib_util.Walker('pipeline-3c.walker')

# parse parset
parset = lib_util.getParset()
parset_dir = parset.get('LOFAR_3c_core', 'parset_dir')
SKYDB_DEMIX = parset.get('LOFAR_demix','demix_model')
bl2flag = parset.get('flag', 'stations')

TARGET = os.getcwd().split('/')[-1]
DATA_DIR = f'/net/voorrijn/data2/boxelaar/data/3Csurvey/tgts/{TARGET}'

RSISlist = [
    'RS106LBA','RS205LBA','RS208LBA','RS210LBA','RS305LBA','RS306LBA',
    'RS307LBA','RS310LBA','RS406LBA','RS407LBA','RS409LBA','RS503LBA',
    'RS508LBA','RS509LBA','DE601LBA','DE602LBA','DE603LBA','DE604LBA',
    'DE605LBA','DE609LBA','FR606LBA','SE607LBA','UK608LBA','PL610LBA',
    'PL611LBA','PL612LBA','IE613LBA','LV614LBA'
]

CS_list = [
    'CS001LBA','CS002LBA','CS003LBA','CS004LBA','CS005LBA','CS006LBA',
    'CS007LBA','CS011LBA','CS013LBA','CS017LBA','CS021LBA','CS024LBA',
    'CS026LBA','CS028LBA','CS030LBA','CS031LBA','CS032LBA','CS101LBA',
    'CS103LBA','CS201LBA','CS301LBA','CS302LBA','CS401LBA','CS501LBA',
]
    
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
    
    predict(test_mss)
    
    Logger.info(f'Solving diagonal test...')
    test_mss.run(
        f'DP3 {parset_dir}/DP3-solG.parset msin=$pathMS \
            msin.datacolumn=CORRECTED_DATA sol.mode=diagonal \
            sol.h5parm=$pathMS/cal-diag.h5 \
            sol.solint=15 sol.smoothnessconstraint=1e6',
        log=f'$nameMS_sol-diag_test.log', 
        commandType="DP3"
    )
    
    lib_util.run_losoto(
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
    
    [lib_util.check_rm(ms) for ms in test_mss.getListStr()]
    

def get_cal_dir(timestamp: str) -> list:
    """
    Get the proper cal directory from a timestamp
    """
    dirs = list()
    for cal_dir in sorted(glob.glob('../../cals/3c*')):
        calibrator = cal_dir.split("/")[-1]
        cal_timestamps = set()
        for ms in glob.glob(cal_dir+'/20*/data-bkp/*MS'):
            cal_timestamps.add("_".join(ms.split("/")[-1].split("_")[:2]))
            
        if f"{calibrator}_t{timestamp}" in cal_timestamps:
            Logger.info('Calibrator found: %s (t=%s)' % (cal_dir, timestamp))
            dirs.append(f"{cal_dir}/{timestamp[:8]}/solutions")
        else:
            pass
        
    if dirs == []:
        Logger.error('Missing calibrator.')
        sys.exit()
    
    return dirs

def make_beam_region(MSs: MeasurementSets) -> tuple[str, str|None]:
    MSs.print_HAcov('plotHAelev.png')
    MSs.getListObj()[0].makeBeamReg('beam02.reg', freq='mid', pb_cut=0.2)
    beam02Reg = 'beam02.reg'
    MSs.getListObj()[0].makeBeamReg('beam07.reg', freq='mid', pb_cut=0.7)
    beam07reg = 'beam07.reg'

    region = f'{parset_dir}/regions/{TARGET}.reg'
    if not os.path.exists(region): 
        region = None
         
    return beam02Reg, region

def correct_from_callibrator(MSs: MeasurementSets, timestamp: str) -> None:
    cal_dir = get_cal_dir(timestamp)[0]
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
            lib_util.check_rm(MS_concat_all)
            lib_util.check_rm(MS_concat_core)
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
            
            '''
            # flag bad stations, and low-elev
            # Flag DATA -> DATA
            Logger.info('Flagging...')
            MSs.run(
                f'DP3 {parset_dir}/DP3-flag.parset msin=$pathMS msout=.\
                    aoflagger.strategy={parset_dir}/LBAdefaultwideband.lua\
                    ant.baseline=\"{bl2flag}\"', #Using default wideband here! Data is narrowband!
                log="$nameMS_flag.log", 
                commandType="DP3"
            )
            '''
            
            # Correct data from calibrator step (pa, amp, beam, iono)
            correct_from_callibrator(MSs, timestamp)

            # bkp
            Logger.info('Making backup...')
            os.system('cp -r %s %s' % (MS_concat_all, MS_concat_bkp) ) # do not use MS.move here as it resets the MS path to the moved one
            
        #BEAM CORRECTION CORE STATIONS 
        # Beam correction CORRECTED_DATA -> CORRECTED_DATA (polalign corrected, beam corrected+reweight)
        #Logger.info('Beam correction (beam)...')
        #MSs.run(
        #    'DP3 '+parset_dir+'/DP3-beam.parset msin=$pathMS msin.datacolumn=DATA \
        #        msout.datacolumn=DATA corrbeam.updateweights=True \
        #        corrbeam.noapplystations="['+','.join(RSISlist)+']"', 
        #    log='$nameMS_cor1_beam.log', 
        #    commandType='DP3'
        #)   
        
        split_stations(
            MeasurementSets([MS_concat_all], SCHEDULE), msout=MS_concat_core
        )

            
def split_stations(measurement: MeasurementSets, msout: str = "", target_angular_size: float = 0.):
    baseline = ""
    if target_angular_size == 0.:
        baseline = "CS*&&"
        
    if msout == "":
        msout = measurement.getListStr()[0]
    
    #offsets = get_antenna_offset(measurement.getListStr()[0])
    

    Logger.info('Splitting data in Core and Remote...')
    measurement.run(
        f'DP3 {parset_dir}/DP3-filter.parset msin=$pathMS msout={msout}',
        log="$nameMS_split.log", 
        commandType="DP3"
    )
    
    

def phaseup(MSs: MeasurementSets, stats: str) -> MeasurementSets:
    if stats == "all":
        Logger.info('Correcting CS...')
        fulljones_solution = sorted(glob.glob("cal-Ga*core.h5"))
        
        """
        solution = sorted(glob.glob("cal-Gp*core.h5"))
        if len(solution) != 0:
                Logger.info(f"correction Gain-scalar of {solution[-1]}")
                # correcting CORRECTED_DATA -> CORRECTED_DATA
                MSs.run(
                    f'DP3 {parset_dir}/DP3-cor.parset msin=$pathMS msin.datacolumn=DATA \
                        cor.parmdb={solution[-1]} cor.correction=phase000',
                    log='$nameMS_corPH-core.log', 
                    commandType='DP3'
                )
        """
        
        if len(fulljones_solution) != 0:
            Logger.info(f"Correction Gain of {fulljones_solution[-1]}")
            # correcting CORRECTED_DATA -> CORRECTED_DATA
            MSs.run(
                f'DP3 {parset_dir}/DP3-cor.parset msin=$pathMS msin.datacolumn=DATA \
                    cor.parmdb={fulljones_solution[-1]} cor.correction=fulljones \
                    cor.soltab=\[amplitude000,phase000\]',
                log='$nameMS_corAMPPHslow-core.log', 
                commandType='DP3'
            )
  
        else:
            Logger.warning(f"No Core corrections found. Phase-up not recommended")
        
        run_test(MSs)

        # Phaseup CORRECTED_DATA -> DATA
        Logger.info('Phasing up Core Stations...')
        lib_util.check_rm(f'*{stats}.MS-phaseup')
        MSs.run(
            f"DP3 {parset_dir}/DP3-phaseup.parset msin=$pathMS \
                msin.datacolumn=CORRECTED_DATA msout=$pathMS-phaseup \
                msout.datacolumn=DATA",                
            log=f'$nameMS_phaseup.log', 
            commandType="DP3"
        )
        
        os.system(f'rm -r *concat_all.MS')
        os.system(f'rm -r *concat_core.MS')
    
    
    if stats == "def":
        with WALKER.if_todo('phaseupCS_' + stats):
            # Phasing up the cose stations
            # Phaseup CORRECTED_DATA -> DATA
            Logger.info('Phasing up superterp Stations...')
            lib_util.check_rm(f'*{stats}.MS-phaseup')
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
        
        if sep > 2 and sep < 25 and (ateam != 'CasA' and ateam != 'CygA'):
            # CasA and CygA are already demixed in preprocessing
            Logger.warning(f'Demix of {ateam} (sep: {sep:.1f} deg)')
            
            for MS in MSs.getListStr():
                lib_util.check_rm(MS + '/' + os.path.basename(SKYDB_DEMIX))
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
    lib_util.check_rm(f'cal*{mode}.h5')
    lib_util.check_rm(f'plots*{mode}')
    lib_util.check_rm('peel*')
    lib_util.check_rm(f'img/img-{mode}*')
    if not os.path.exists("img/"):
        os.makedirs('img')
        
    
def main() -> None:
    #calibration_modes = ["def"]
    calibration_modes = ["core","all"]   
    
    with WALKER.if_todo('setup'):
            #set up corected data
            setup() 
    
    for stations in calibration_modes:
        MSs = MeasurementSets(
            glob.glob(f'*concat_{stations}.MS'), 
            SCHEDULE, 
            check_flags=False
        )
        
        with WALKER.if_todo(f"clean_{stations}"):
            clean_specific(stations)       

        if stations != "core": 
            with WALKER.if_todo('phaseupCS ' + stations):
                MSs = phaseup(MSs, stations)
            
            MSs = MeasurementSets(
                glob.glob(f'*concat_{stations}.MS-phaseup'), 
                SCHEDULE, 
                check_flags=False, 
                check_sun=True
            )
            
            #BEAM CORRECTION ALL BUT SUPERLBA
            # Beam correction DATA -> DATA (polalign corrected, beam corrected+reweight)
            #Logger.info('Beam correction (beam)...')
            #MSs.run(
            #    'DP3 '+parset_dir+'/DP3-beam.parset msin=$pathMS msin.datacolumn=DATA \
            #        msout.datacolumn=DATA corrbeam.updateweights=True \
            #        corrbeam.noapplystations="[SuperStLBA]"', 
            #    log='$nameMS_cor1_beam.log', 
            #    commandType='DP3'
            #)   

        # make beam region files
        masking = make_beam_region(MSs)
        
        # Predict model    
        with WALKER.if_todo('predict_' + stations):  
            predict(MSs, doBLsmooth=False)
        
        rms_noise_pre = np.inf
        mm_ratio_pre = 0
        doslow = True
        
        if stations == "core":
            total_cycles = 4
        elif stations == "all":
            total_cycles = 12
        else:
            total_cycles = 10

        calibration = SelfCalibration(MSs, schedule=SCHEDULE, total_cycles=total_cycles, mask=masking, stats=stations)
        
        for cycle in calibration:
            #with WALKER.if_todo(f"cal_tec_{stations}"):
            #    if cycle == 1:
            #        calibration.solve_tec()
            
            # Smooth DATA -> DATA
            Logger.info('BL-based smoothing...')
            calibration.mss.run(
                '/net/voorrijn/data2/boxelaar/scripts/LiLF/scripts/BLsmooth.py\
                    -r -s 0.8 -i DATA -o SMOOTHED_DATA $pathMS', 
                log='$nameMS_smooth1.log', 
                commandType='python'
            )
                
            with WALKER.if_todo(f"cal_{stations}_c{cycle}"):
                if stations == "core":
                    if cycle == 1:
                        calibration.solve_gain('scalar')
                        
                    calibration.solve_gain("fulljones")
                    
                else:   
                    if calibration.doph:
                        calibration.solve_gain('scalar')
                        
                    if calibration.doamp and doslow: # or (total_cycles - cycle <= 1):
                        calibration.solve_gain('fulljones')
                    

            #if stations == "all":
            with WALKER.if_todo(f"image-{stations}-c{cycle}" ):
                imagename = f'img/img-{stations}-{cycle:02d}'
                calibration.clean(imagename)
                
                rms_noise_pre, mm_ratio_pre, stopping = calibration.prepare_next_iter(imagename, rms_noise_pre, mm_ratio_pre)
                if stopping: 
                    break         
                                  
    Logger.info("Done.")


if __name__ == "__main__":
    if not os.path.exists(DATA_DIR+"/data"):
        os.makedirs(DATA_DIR+"/data")
        os.system(f"mv {DATA_DIR}/*.MS {DATA_DIR}/data/")
    
    main()        
    