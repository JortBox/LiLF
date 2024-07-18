#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys, os, glob
import numpy as np

#sys.path.append("/localdata/scripts/LiLF")
from LiLF_lib import lib_img, lib_util, lib_log
from LiLF_lib.lib_ms import AllMSs as MeasurementSets

logger = lib_log.logger
WALKER = lib_util.Walker('pipeline-3c.walker')

# parse parset
parset = lib_util.getParset()
parset_dir = parset.get('LOFAR_3c_core', 'parset_dir')

TARGET = os.getcwd().split('/')[-1]

extended_targets = [
    '3c223','3c284','3c274',
    '3c285','3c293','3c31',#'3c296',
    '3c310','3c326',
    '3c33','3c35','3c382','3c386','3c442a','3c449',
    '3c454.3','3c465','3c84', '4c73.08', 'ngc6109'#, 'ngc6251'
]

very_extended_targets = ['da240','3c236', 'ngc6251']

difficult_targets = ["4c12.03", "3c212", "3c191"]

class SelfCalibration(object):
    def __init__(
            self, 
            MSs: MeasurementSets, 
            mask: tuple, 
            schedule: lib_util.Scheduler,
            total_cycles: int = 10, 
            doslow: bool = False, 
            stats: str = "",
            target: str =  ""
        ):
        global TARGET
        if target != "":
            TARGET = target
        
        self.mss = MSs
        self.stop = total_cycles
        self.cycle = 0
        self.ampcycle = 0
        self.mask = mask
        self.stats = stats
        self.s = schedule
        
        if stats == "core":
            self.solint_fj = lib_util.Sol_iterator([200,100,50,10])
            self.solint_amp = lib_util.Sol_iterator([200,100,50,10])
            self.solint_ph = lib_util.Sol_iterator([1])
        elif stats == "int":
            self.solint_fj = lib_util.Sol_iterator([800, 400, 200,100,50])
            self.solint_amp = lib_util.Sol_iterator([800, 400, 200,100,50])
            self.solint_ph = lib_util.Sol_iterator([16,16,16,16,10,5,1])
        else:
            self.solint_fj = lib_util.Sol_iterator([400, 200,100,50])
            self.solint_amp = lib_util.Sol_iterator([400, 200,100,50])
            self.solint_ph = lib_util.Sol_iterator([10,5,1])
        
        self.doslow = doslow
        self.doamp = False
        self.doph = True
        self.phased_up = False
        self.data_column = "DATA"
        self.rms_history = list()
        self.ratio_history = list()
        
        
    def __iter__(self):
        return self
        
    def __next__(self) -> int:
        if self.cycle + 1 > self.stop:
            raise StopIteration
        
        else:
            self.data_column = 'DATA'
            if self.doamp:
                self.ampcycle += 1
            self.cycle += 1
            logger.info('== Start cycle: %s ==' % self.cycle)  
            return self.cycle
        
    def subtract_infield_source(self, sub_source: str, path_to_model: str = ""):
        logger.info(f'Subtracting {sub_source}...')
        
        logger.info(f'load {sub_source} model into SUB_MODEL_DATA (DP3)...')
        '''
        # Predict MODEL_DATA
        os.system(f'makesourcedb outtype="blob" format="<" in={path_to_model}/{sub_source}.skymodel out={sub_source}.skydb')
        self.mss.run(
            f'DP3 {parset_dir}/DP3-predict.parset msin=$pathMS \
                msout.datacolumn=SUB_MODEL_DATA pre.usebeammodel=true pre.sourcedb={sub_source}.skydb', 
            log='$nameMS_pre.log', 
            commandType='DP3'
        )
        '''
        
        model_img = sorted(glob.glob(f'{path_to_model}/img/img-all-*-MFS-model.fits'))[-2]
        self.mss.scheduler.add(
            f"wsclean -predict -name {model_img} \
                -j {self.mss.scheduler.max_processors} -channels-out 2 -reorder \
                -parallel-reordering 4 {self.mss.getStrWsclean()}",
            log="wsclean-pre.log",
            commandType="wsclean",
            processors="max",
        )
        self.mss.scheduler.run(check=True)
        
        # corrupt
        solutions = sorted(glob.glob(f"cal-G*-c{self.cycle}-{self.stats}-ampnorm.h5"))[::-1]
        for solution in solutions:
            correction=""
            if "Gp" in solution:
                correction = "phase000"
                soltab = ""
            elif "Ga" in solution:
                correction = "fulljones"
                soltab = "cor.soltab=[amplitude000,phase000]"
            
            logger.info(f'corrupt model with {solution}...')
            self.mss.run( # was MSs
                f"DP3 {parset_dir}/DP3-cor.parset msin=$pathMS \
                    msin.datacolumn=SUB_MODEL_DATA msout.datacolumn=SUB_MODEL_DATA \
                    cor.invert=False cor.parmdb={solution} \
                    cor.correction={correction} {soltab}",
                log="$nameMS_corrupt.log",
                commandType="DP3",
            )
            
        #subtract
        logger.info('Subtracting model from CORRECTED_DATA...')
        self.mss.run(
            f'taql "update $pathMS set CORRECTED_DATA = CORRECTED_DATA - SUB_MODEL_DATA"',
            log="$nameMS_taql.log",
            commandType="general",
        )
        
        
    def predict_from_img(self, path:str, sourcedb: str = "tgts.skydb"):
        import lsmtool #type: ignore 
    
        if not os.path.exists(path.split("-")[0]+"-mask.skymodel"):
            im = lib_img.Image(path)
            im.makeMask(threshpix=2, rmsbox=(36,12), atrous_do=True, write_gaul=True)
        
        os.system(f"cp {path.split('-')[0]}-mask.skymodel tgts.skymodel")
        
        #lsm = lsmtool.load('tgts.skymodel')
        #lsm.remove('I<0.5')
        #lsm.write('tgts.skymodel', applyBeam=False, clobber=True)
        os.system(f'makesourcedb outtype="blob" format="<" in=tgts.skymodel out={sourcedb}')
        
        # Predict MODEL_DATA
        logger.info('Predict (DP3)...')
        self.mss.run(
            f'DP3 {parset_dir}/DP3-predict.parset msin=$pathMS pre.usebeammodel=true pre.sourcedb={sourcedb}', 
            log='$nameMS_pre.log', 
            commandType='DP3'
        )
        
    def smooth_baselines(self, mode, smooth_fj, smooth_all_pols) -> tuple[str, str]:
        if mode == "fulljones" and not smooth_fj:
            return self.data_column, "MODEL_DATA"
        
        if smooth_all_pols or mode in ["phase", "scalar", "amplitude"]:
            command = f'-r -d -s 0.8 -i {self.data_column} -o SMOOTHED_DATA $pathMS'
        else:
            command = f'-r -d -s 0.8 -i {self.data_column} -o SMOOTHED_DATA $pathMS'
            
        logger.info(f'Smoothing {self.data_column} -> SMOOTHED_DATA...')
        self.mss.run(
            f'BLsmooth_pol.py {command}', 
            log='$nameMS_smooth1.log', 
            commandType='python'
        )
          
        if mode in ["phase", "scalar"]:
            logger.info('Smoothing MODEL_DATA -> SMOOTHED_MODEL_DATA...')
            self.mss.run(
                f'BLsmooth_pol.py -r -d -s 0.8 -i MODEL_DATA -o SMOOTHED_MODEL_DATA $pathMS', 
                log='$nameMS_smooth2.log', 
                commandType='python'
            ) 
            model_in = "SMOOTHED_MODEL_DATA"
            

        else:
            model_in = "MODEL_DATA" 

        return "SMOOTHED_DATA", model_in
    
    
 
        
        
    def solve_gain(self, mode:str, solint = None, bl_smooth_fj = False, smooth_all_pols = False, smoothnessconstraint:str="") -> None:
        assert mode in ["phase", "amplitude", "scalar", "fulljones"]
        if solint is None:
            if mode in ["phase", "scalar"]:
                solint = next(self.solint_ph)
            elif mode == "amplitude":
                solint = next(self.solint_amp)
            else: 
                solint = next(self.solint_fj)
        else:
            solint = int(solint)
            
        data_in, model_in = self.smooth_baselines(mode, bl_smooth_fj, smooth_all_pols)
        
        
        logger.info(f'Solving {mode} (Datacolumn: {self.data_column})...')
        if mode == 'phase':
            if self.stats == "int":
                smoothcons = "0.2e6"
                constraint = "sol.coreconstraint=70000"
            else:
                smoothcons = "1e6"
                constraint = ""
            self.mss.run(
                f'DP3 {parset_dir}/DP3-solG.parset msin=$pathMS \
                    msin.datacolumn=SMOOTHED_DATA sol.mode=scalarphase \
                    sol.h5parm=$pathMS/calGph-{self.stats}.h5 \
                    sol.modeldatacolumns=[{model_in}] {constraint}\
                    sol.solint={solint} sol.smoothnessconstraint={smoothcons}',
                log=f'$nameMS_solGph-c{self.cycle:02d}.log', 
                commandType="DP3"
            )
            
            losoto_ops = [
                f'{parset_dir}/losoto-plot2d.parset', 
                f'{parset_dir}/losoto-plot.parset'
            ]
            #if self.stats == "all": 
            #    losoto_ops.insert(0, f'{parset_dir}/losoto-ref-ph.parset')
                
            lib_util.run_losoto(
                self.s, 
                f'Gph-c{self.cycle:02d}-{self.stats}-ampnorm', 
                [f'{ms}/calGph-{self.stats}.h5' for ms in self.mss.getListStr()],
                losoto_ops
            )
        
            # Correct DATA -> CORRECTED_DATA
            logger.info('Correction PH...')
            command = f'DP3 {parset_dir}/DP3-cor.parset msin=$pathMS msin.datacolumn={self.data_column} \
                cor.parmdb=cal-Gph-c{self.cycle:02d}-{self.stats}-ampnorm.h5 cor.correction=phase000' 
            self.mss.run(
                command, 
                log=f'$nameMS_corGph-c{self.cycle:02d}.log', 
                commandType='DP3'
            )
            self.data_column = "CORRECTED_DATA"
            
        elif mode == 'amplitude':
            self.mss.run(
                f'DP3 {parset_dir}/DP3-solG.parset msin=$pathMS \
                    msin.datacolumn=SMOOTHED_DATA sol.mode=scalaramplitude \
                    sol.h5parm=$pathMS/calGsa-{self.stats}.h5 \
                    sol.modeldatacolumns=[{model_in}] \
                    sol.solint={solint} sol.smoothnessconstraint=1e6',
                log=f'$nameMS_solGsa-c{self.cycle:02d}.log', 
                commandType="DP3"
            )
            
            losoto_ops = [
                parset_dir + '/losoto-ampnorm-scalar.parset',
                parset_dir+'/losoto-clip.parset', 
                parset_dir+'/losoto-plot2d.parset',
                parset_dir+'/losoto-plot.parset'
            ]
            if self.stats != "core" and self.phased_up: 
                try: losoto_ops.insert(0, f'{parset_dir}/losoto-ref-ph.parset')
                except: pass
                    
            lib_util.run_losoto(
                self.s, 
                f'Gsa-c{self.cycle:02d}-{self.stats}-ampnorm', 
                [f'{ms}/calGsa-{self.stats}.h5' for ms in self.mss.getListStr()],
                losoto_ops
            )

            # Correct DATA -> CORRECTED_DATA
            logger.info('Correction PH...')
            command = f'DP3 {parset_dir}/DP3-cor.parset msin=$pathMS msin.datacolumn={self.data_column} \
                cor.parmdb=cal-Gsa-c{self.cycle:02d}-{self.stats}-ampnorm.h5 cor.correction=amplitude000' 
            self.mss.run(
                command, 
                log=f'$nameMS_corGsa-c{self.cycle:02d}.log', 
                commandType='DP3'
            )
            self.data_column = "CORRECTED_DATA"
            
        elif mode == 'scalar':
            self.mss.run(
                f'DP3 {parset_dir}/DP3-solG.parset msin=$pathMS \
                    msin.datacolumn=SMOOTHED_DATA sol.mode=scalar \
                    sol.h5parm=$pathMS/calGp-{self.stats}.h5 \
                    sol.modeldatacolumns=[{model_in}] \
                    sol.solint={solint} sol.smoothnessconstraint=1e6',
                log=f'$nameMS_solGp-c{self.cycle:02d}.log', 
                commandType="DP3"
            )
            
            losoto_ops = [
                f'{parset_dir}/losoto-clip-large.parset', 
                f'{parset_dir}/losoto-plot2d.parset', 
                f'{parset_dir}/losoto-plot.parset'
            ]
            if self.stats != "core" and self.phased_up: 
                try: losoto_ops.insert(0, f'{parset_dir}/losoto-ref-ph.parset')
                except: pass
                
            lib_util.run_losoto(
                self.s, 
                f'Gp-c{self.cycle:02d}-{self.stats}-ampnorm', 
                [f'{ms}/calGp-{self.stats}.h5' for ms in self.mss.getListStr()],
                losoto_ops
            )
        
            # Correct DATA -> CORRECTED_DATA
            logger.info('Correction PH...')
            command = f'DP3 {parset_dir}/DP3-cor.parset msin=$pathMS msin.datacolumn={self.data_column} \
                cor.parmdb=cal-Gp-c{self.cycle:02d}-{self.stats}-ampnorm.h5 cor.correction=phase000' 
            self.mss.run(
                command, 
                log=f'$nameMS_corGp-c{self.cycle:02d}.log', 
                commandType='DP3'
            )
            self.data_column = "CORRECTED_DATA"

        elif mode == 'fulljones':
            #if self.stats == "core": smoothcons = '0'
            #else: smoothcons = '0.1e6'
            if TARGET in extended_targets or TARGET in difficult_targets:
                smoothcons = '1.e6'
            else:
                smoothcons = '0.1 e6'
                
            self.mss.run(
                f'DP3 {parset_dir}/DP3-solG.parset msin=$pathMS \
                    msin.datacolumn={data_in} sol.mode=fulljones \
                    sol.h5parm=$pathMS/calGa-{self.stats}.h5 \
                    sol.modeldatacolumns=[{model_in}] \
                    sol.solint={solint} sol.smoothnessconstraint={smoothcons}',
                log=f'$nameMS_solGa-c{self.cycle:02d}.log', 
                commandType="DP3"
            )
            
            losoto_ops = [
                parset_dir+'/losoto-clip.parset', 
                parset_dir+'/losoto-plot2d.parset', 
                parset_dir+'/losoto-plot2d-pol.parset', 
                parset_dir+'/losoto-plot-pol.parset'
            ]
            #if self.stats == "core":
            losoto_ops.insert(0, f'{parset_dir}/losoto-ampnorm-diagonal.parset')  
            if self.stats != "core" and self.phased_up:
                try: losoto_ops.insert(0, f'{parset_dir}/losoto-ref-ph.parset')
                except: pass
                
            lib_util.run_losoto(
                self.s, 
                f'Ga-c{self.cycle:02d}-{self.stats}-ampnorm', 
                [ms+'/calGa-'+self.stats+'.h5' for ms in self.mss.getListStr()],
                losoto_ops
            )
                        
            # Correct CORRECTED_DATA -> CORRECTED_DATA
            logger.info('Correction slow AMP+PH...')
            command = f'DP3 {parset_dir}/DP3-cor.parset msin=$pathMS msin.datacolumn={self.data_column} \
                cor.parmdb=cal-Ga-c{self.cycle:02d}-{self.stats}-ampnorm.h5 cor.correction=fulljones \
                cor.soltab=[amplitude000,phase000]'  
            self.mss.run(
                command,
                log=f'$nameMS_corGa-c{self.cycle:02d}.log', 
                commandType='DP3'
            )
            self.data_column = "CORRECTED_DATA"
            
    def correct(self, solution: str, column_in: str = "DATA", column_out: str = "CORRECTED_DATA", mode = None):
        if "cal-Gp" in solution:
            correction = "phase000"
            soltab = ""
        elif "cal-Ga" in solution:
            correction = "fulljones"
            soltab = "cor.soltab=[amplitude000,phase000]"

        logger.info(f'Correction of {solution}...')
        self.mss.run(
            f'DP3 {parset_dir}/DP3-cor.parset msin=$pathMS msin.datacolumn={column_in} \
                msout.datacolumn={column_out} cor.parmdb={solution} \
                cor.correction={correction} {soltab}' , 
            log=f'$nameMS_cor_{solution}.log', 
            commandType='DP3'
        )
         
    def apply_mask(self, imagename: str, maskfits: str) -> None:
        beam02Reg, _, region = self.mask
        # check if hand-made mask is available
        # Use masking scheme from LOFAR_dd_wsclean
        im = lib_img.Image(imagename+'-MFS-image.fits')
        #im.makeMask(self.s, self.cycle, mode="breizorro", threshpix=5, rmsbox=(50,5), atrous_do=True)#, maskname=maskfits) #Pybdsf step here
        if TARGET in very_extended_targets or TARGET in extended_targets:
            threshold = 3
        else:
            threshold = 5
            
        if self.stats == "core":
            im.makeMask(threshpix=5, rmsbox=(50,5), atrous_do=True)
        else:
            im.makeMask(threshpix=threshold, rmsbox=(50,5), atrous_do=True)
        
        if self.stats != "core" and TARGET in extended_targets:
            logger.info("Manual masks used")
            lib_img.blank_image_reg(maskfits, beam02Reg, blankval = 0.)
            lib_img.blank_image_reg(maskfits, region, inverse=True, blankval = 0.)
            lib_img.blank_image_reg(maskfits, region, blankval = 1.)
        elif (region is not None) and (self.stats != "core") and (not self.doamp):
            logger.info("Manual masks used")
            lib_img.blank_image_reg(maskfits, beam02Reg, blankval = 0.)
            lib_img.blank_image_reg(maskfits, region, blankval = 1.)
        else:
            logger.info("NO Manual mask used")
            
            
    def clean(self, imagename: str, uvlambdamin: int = 30, deep: bool = False, size: int = 2500, predict: bool = True, apply_beam: bool = False, image_per_hour: bool = False) -> None:
        # special for extended sources:
        print("TARGET:", TARGET)
        if TARGET in very_extended_targets:
            kwargs1 = {
                'weight': 'briggs 0.5', 
                'taper_gaussian': '50arcsec', 
                'multiscale': '', 
                'multiscale_scale_bias':0.5, 
                'multiscale_scales':'0,30,60,120,340'
            }
            kwargs2 = {
                'weight': 'briggs 0.5', 
                'taper_gaussian': '50arcsec', 
                'multiscale_scale_bias':0.5,
                'multiscale_scales': '0,30,60,120,340'
            }
        elif TARGET in extended_targets:
            kwargs1 = {
                'weight': 'briggs -0.7', 
                'multiscale_scale_bias':0.5, 
                'taper_gaussian': '15arcsec'
            }
            kwargs2 = {
                'weight': 'briggs -0.7',
                'multiscale_scale_bias':0.5,  
                'taper_gaussian': '15arcsec', 
                'multiscale_scales': '0,15,30,60,120,240'
            }
        elif self.stats == "int":
            kwargs1 = {'weight': 'briggs -1.5'}
            kwargs2 = {
                'weight': 'briggs -1.5', 
                'multiscale_scales': '0,3,6,10,20,40,80,160',
                'multiscale_scale_bias':0.8, 
            }
        else:
            kwargs1 = {'weight': 'briggs -0.8'}
            kwargs2 = {
                'weight': 'briggs -0.8', 
                'multiscale_scales': '0,10,20,40,80,160',
                'multiscale_scale_bias':0.8, 
            }
        
        if TARGET == "3c274":
            kwargs1.update({'weight': 'briggs -1.0'}) # type: ignore
            kwargs2.update({'weight': 'briggs -1.0'}) # type: ignore
            kwargs2.update({'multiscale_scale_bias':0.5}) # type: ignore
        
        kwargs1.update({"size": size}) # type: ignore
        kwargs2.update({"size": size}) # type: ignore
        kwargs1.update({"scale": "2arcsec"}) # type: ignore
        kwargs2.update({"scale": "2arcsec"}) # type: ignore
            
        if self.stats == "core":
            kwargs1["size"] = 500; kwargs1["scale"] = "50.0arcsec" # type: ignore
            kwargs2["size"] = 500; kwargs2["scale"] = "50.0arcsec" # type: ignore
            kwargs2["weight"] = "briggs -0.8" # type: ignore
        elif self.stats == "int":
            kwargs1["size"] = size; kwargs1["scale"] = "0.15arcsec" # type: ignore
            kwargs2["size"] = size; kwargs2["scale"] = "0.15arcsec" # type: ignore
            #kwargs2["weight"] = "briggs -1.5" # type: ignore
        
        if apply_beam:
            kwargs1.update({"apply_primary_beam": ""})
            kwargs2.update({"apply_primary_beam": ""})
            
        if deep:
            kwargs1.update({"circular_beam": ""})
            kwargs2.update({"circular_beam": ""})
        
        print("kwargs1, kwargs2")   
        print(kwargs1)
        print(kwargs2)
        
        #all_mms = [ms.pathMS for ms in self.mss.getListObj()][0]

        # if next is a "cont" then I need the do_predict
        
        for ms in self.mss.getListStr():
            if image_per_hour:
                input_filename = ms
                imagename_to_use = imagename + "-" + ms.split("/")[-1].split("_")[1]
                logger.info(f'Cleaning by hour ({ms})...')
            else:
                input_filename = self.mss.getStrWsclean()
                imagename_to_use = imagename
                logger.info('Cleaning all ms...')

            logger.info('Cleaning shallow (cycle: '+str(self.cycle)+')...')
            lib_util.run_wsclean(
                self.s, 
                'wsclean1-c%02i.log' % self.cycle, 
                #all_mms, #
                input_filename, 
                do_predict=predict, 
                name=imagename_to_use,
                parallel_gridding=4, 
                baseline_averaging='',
                niter=1000, 
                no_update_model_required='',
                #circular_beam='',
                save_source_list='',
                minuv_l=uvlambdamin, 
                mgain=0.4, 
                nmiter=0,
                auto_threshold=5, 
                local_rms='', 
                local_rms_method='rms-with-min',
                join_channels='', 
                fit_spectral_pol=2, 
                channels_out=2, 
                **kwargs1
            )
            
            maskfits = imagename_to_use+'-mask.fits'
            self.apply_mask(imagename_to_use, maskfits)

            logger.info('Cleaning full (cycle: '+str(self.cycle)+')...')
            lib_util.run_wsclean(
                self.s, 
                'wsclean2-c%02i.log' % self.cycle, 
                #all_mms, #
                input_filename, 
                name=imagename_to_use,
                do_predict=predict, 
                cont=True, 
                parallel_gridding=4,
                niter=1000000, 
                no_update_model_required='',
                #circular_beam='',
                minuv_l=uvlambdamin,
                mgain=0.4, 
                nmiter=0,
                auto_threshold=0.5, 
                auto_mask=2., 
                local_rms='', 
                local_rms_method='rms-with-min', 
                fits_mask=maskfits,
                multiscale='',
                join_channels='', 
                fit_spectral_pol=2, 
                channels_out=2, 
                **kwargs2
            )
            os.system('cat logs/wsclean-c%02i.log | grep "background noise"' % self.cycle)    
            
            if not image_per_hour:
                break
        
        
    def low_resolution_clean(self, imagename: str, uvlambdamin: int = 30, taper: float=25):
        # Low res image
        gaussian_taper = str(taper)+"arcsec"
        logger.info(f'Cleaning low resoluton ({taper} arcsec)...')
        lib_util.run_wsclean(
            self.s, 
            'wsclean-lr.log', 
            self.mss.getStrWsclean(), 
            name=imagename, 
            save_source_list='',
            parallel_gridding=4, 
            size=500, 
            scale='5arcsec', 
            weight='briggs -0.7', 
            taper_gaussian=gaussian_taper,
            niter=1000000, 
            no_update_model_required='', 
            minuv_l=uvlambdamin, 
            mgain=0.75, 
            nmiter=0,
            auto_threshold=0.5, 
            auto_mask=1, 
            local_rms='',
            multiscale='', 
            multiscale_scale_bias=0.8,
            multiscale_scales='0,10,20,40,80,160',
            join_channels='', 
            fit_spectral_pol=2, 
            channels_out=2
        )
        os.system('cat logs/wsclean-lr.log | grep "background noise"') 
        
    def empty_clean(self, imagename: str, uvlambdamin: int = 30):
        kwargs1 = {'weight': 'briggs -0.8', "size": 2500, "scale": "2.0arcsec"} # type: ignore
        
        logger.info('Cleaning empty (cycle: '+str(self.cycle)+')...')
        lib_util.run_wsclean(
            self.s, 
            'wsclean-empty-c%02i.log' % self.cycle, 
            self.mss.getStrWsclean(),
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
    
    def prepare_next_iter(self, imagename: str, rms_noise_pre: float, mm_ratio_pre: float) -> tuple[float, float, bool]:
        stopping = False
        
        
        im = lib_img.Image(imagename+'-MFS-image.fits')
        im.makeMask(threshpix=5, rmsbox=(500,30), atrous_do=False )
        rms_noise = float(im.getNoise()) 
        mm_ratio = float(im.getMaxMinRatio())
        logger.info('RMS noise: %f - MM ratio: %f' % (rms_noise, mm_ratio))
    
        if self.doamp and rms_noise > 0.99*rms_noise_pre and mm_ratio < 1.01*mm_ratio_pre and self.cycle > 6:
            stopping = True  # if already doing amp and not getting better, quit
        
        if TARGET in extended_targets or TARGET in very_extended_targets:
            if rms_noise > rms_noise_pre and mm_ratio < mm_ratio_pre:
                self.doamp = True
        else:
            if rms_noise > 0.95*rms_noise_pre and mm_ratio < 1.05*mm_ratio_pre:
                self.doamp = True    
        
        #if rms_noise > rms_noise_pre and mm_ratio < mm_ratio_pre:
        #    self.doamp = True
        
        self.rms_history.append(rms_noise)
        self.ratio_history.append(mm_ratio)
        return rms_noise, mm_ratio, stopping
