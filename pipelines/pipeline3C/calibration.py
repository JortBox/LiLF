#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys, os
import numpy as np

sys.path.append("/net/voorrijn/data2/boxelaar/scripts/LiLF")

from LiLF_lib import lib_img, lib_util, lib_log
from LiLF_lib.lib_ms import AllMSs as MeasurementSets

logger = lib_log.logger
WALKER = lib_util.Walker('pipeline-3c.walker')

# parse parset
parset = lib_util.getParset()
parset_dir = parset.get('LOFAR_3c_core', 'parset_dir')

TARGET = os.getcwd().split('/')[-1]

extended_targets = [
    '3c223','3c231','3c236','3c264','3c274','3c284',
    '3c285','3c293','3c296','3c31','3c310','3c326',
    '3c33','3c35','3c382','3c386','3c442a','3c449',
    '3c454.3','3c465','3c84'
]
very_extended_targets = ['3c138','da240']

class SelfCalibration(object):
    def __init__(
            self, 
            MSs: MeasurementSets, 
            mask: tuple, 
            schedule: lib_util.Scheduler,
            total_cycles: int = 10, 
            doslow: bool = False, 
            stats: str = ""
        ):
        self.mss = MSs
        self.stop = total_cycles
        self.cycle = 0
        self.ampcycle = 0
        self.mask = mask
        self.stats = stats
        self.s = schedule
        
        if stats == "core":
            self.solint_amp = lib_util.Sol_iterator([200,100,50,10,5])
            self.solint_ph = lib_util.Sol_iterator([10,3,1])
        else:
            self.solint_amp = lib_util.Sol_iterator([200,100,50])
            self.solint_ph = lib_util.Sol_iterator([10,3,1])
        
        self.doslow = doslow
        self.doamp = False
        self.doph = True
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
        
        
    def solve_gain(self, mode:str, solint: int|None = None) -> None:
        assert mode in ["scalar", "fulljones"]
        if solint is None:
            if mode == "scalar":
                solint = next(self.solint_ph)
            else: 
                solint = next(self.solint_amp)
        else:
            solint = int(solint)
        
        '''
        # Smooth CORRECTED_DATA -> SMOOTHED_DATA
        logger.info('BL-based smoothing...')
        self.mss.run(
            f'/net/voorrijn/data2/boxelaar/scripts/LiLF/scripts/BLsmooth.py\
                -r -s 0.8 -i {self.data_column} -o SMOOTHED_DATA $pathMS', 
            log='$nameMS_smooth1.log', 
            commandType='python'
        )     
        '''
        
        logger.info(f'Solving {mode} (Datacolumn: {self.data_column})...')
        if mode == 'scalar':
            # solve G - group*_TC.MS:CORRECTED_DATA
            #solint = next(self.solint_ph)
            self.mss.run(
                f'DP3 {parset_dir}/DP3-solG.parset msin=$pathMS \
                    msin.datacolumn={self.data_column} sol.mode=scalar \
                    sol.h5parm=$pathMS/calGp-{self.stats}.h5 \
                    sol.solint={solint} sol.smoothnessconstraint=1e6',
                log=f'$nameMS_solGp-c{self.cycle:02d}.log', 
                commandType="DP3"
            )
            

            lib_util.run_losoto(
                self.s, 
                f'Gp-c{self.cycle:02d}-{self.stats}-ampnorm', 
                [f'{ms}/calGp-{self.stats}.h5' for ms in self.mss.getListStr()],
                [
                    #parset_dir+'/losoto-ampnorm-scalar.parset',
                    parset_dir+'/losoto-clip-large.parset', 
                    parset_dir+'/losoto-plot2d.parset', 
                    parset_dir+'/losoto-plot.parset'
                ]
            )
        
            # Correct DATA -> CORRECTED_DATA
            logger.info('Correction PH...')
            command = f'DP3 {parset_dir}/DP3-cor.parset msin=$pathMS msin.datacolumn={self.data_column} \
                cor.parmdb=cal-Gp-c{self.cycle:02d}-{self.stats}-ampnorm.h5 cor.correction=phase000'
                
            self.mss.run(
                command, log=f'$nameMS_corGp-c{self.cycle:02d}.log', commandType='DP3'
            )
            self.data_column = "CORRECTED_DATA"

        elif mode == 'fulljones':
            # solve G - group*_TC.MS:CORRECTED_DATA
            #sol.antennaconstraint=[[RS509LBA,...]] \
            #solint = next(self.solint_amp)
            self.mss.run(
                f'DP3 {parset_dir}/DP3-solG.parset msin=$pathMS \
                    msin.datacolumn={self.data_column} sol.mode=fulljones \
                    sol.h5parm=$pathMS/calGa-{self.stats}.h5  \
                    sol.solint={solint}',
                log=f'$nameMS_solGa-c{self.cycle:02d}.log', 
                commandType="DP3"
            )

            lib_util.run_losoto(
                self.s, 
                f'Ga-c{self.cycle:02d}-{self.stats}-ampnorm', 
                [ms+'/calGa-'+self.stats+'.h5' for ms in self.mss.getListStr()],
                [
                    #parset_dir+'/losoto-ampnorm-full-diagonal.parset',
                    parset_dir+'/losoto-clip.parset', 
                    parset_dir+'/losoto-plot2d.parset', 
                    parset_dir+'/losoto-plot2d-pol.parset', 
                    parset_dir+'/losoto-plot-pol.parset'
                ]  
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

            
    def solve_tec(self) -> None:
        logger.info("BL-based smoothing...")
        self.mss.run(
            '/net/voorrijn/data2/boxelaar/scripts/LiLF/scripts/BLsmooth.py \
                -c 8 -n 8 -r -i '+self.data_column+' -o SMOOTHED_DATA $pathMS', 
            log='$nameMS_smooth-c'+str(self.cycle)+'.log', 
            commandType='python'
        )
        
        if self.stats == "core":
            # solve TEC - ms:SMOOTHED_DATA (1m 2SB)
            logger.info('Solving TEC1...')
            self.mss.run(
                'DP3 '+parset_dir+'/DP3-solTEC.parset msin=$pathMS sol.h5parm=$pathMS/tec1.h5 \
                    sol.antennaconstraint=[[CS002LBA,CS003LBA,CS004LBA,CS005LBA,CS006LBA,CS007LBA]] \
                    sol.solint='+str(15), # HARDCODED
                    #+' sol.nchan='+str(8*base_nchan), \
                log=f'$nameMS_solTEC1-c{self.cycle}.log', 
                commandType='DP3'
            )

            lib_util.run_losoto(
                self.s, 'tec1-c'+str(self.cycle), 
                [ms+'/tec1.h5' for ms in self.mss.getListStr()], 
                [parset_dir+'/losoto-plot-tec.parset']
            )
            #os.system('mv cal-tec1-c'+str(self.cycle)+'.h5 self/solutions/')
            #os.system('mv plots-tec1-c'+str(self.cycle)+' self/plots/')
            
            # correct TEC - group*_TC.MS:CORRECTED_DATA -> group*_TC.MS:CORRECTED_DATA
            logger.info('Correcting TEC1...')
            self.mss.run(
                f'DP3 {parset_dir}/DP3-cor.parset msin=$pathMS msin.datacolumn={self.data_column}\
                    cor.parmdb=cal-tec1-c{self.cycle}.h5 cor.correction=tec000',
                log='$nameMS_corTEC1-c'+str(self.cycle)+'.log', 
                commandType='DP3'
            )
        
          
        else:
            # solve TEC - ms:SMOOTHED_DATA (4s, 1SB)
            logger.info('Solving TEC2...')
            self.mss.run(
                'DP3 '+parset_dir+'/DP3-solTEC.parset msin=$pathMS sol.h5parm=$pathMS/tec2.h5 \
                    sol.solint='+str(15), # HARDCODED
                    #+' sol.nchan='+str(8*base_nchan), \
                log=f'$nameMS_solTEC2-c{self.cycle}.log', 
                commandType='DP3'
            )

            lib_util.run_losoto(
                self.s, 'tec2-c'+str(self.cycle), 
                [ms+'/tec2.h5' for ms in self.mss.getListStr()], 
                [parset_dir+'/losoto-plot-tec.parset']
            )
            #os.system('mv cal-tec2-c'+str(self.cycle)+'.h5 self/solutions/')
            #os.system('mv plots-tec2-c'+str(self.cycle)+' self/plots/')

            # correct TEC - group*_TC.MS:CORRECTED_DATA -> group*_TC.MS:CORRECTED_DATA
            logger.info('Correcting TEC2...')
            self.mss.run(
                f'DP3 {parset_dir}/DP3-cor.parset msin=$pathMS msin.datacolumn={self.data_column}\
                    cor.parmdb=cal-tec2-c{self.cycle}.h5 cor.correction=tec000',
                log='$nameMS_corTEC2-c'+str(self.cycle)+'.log', 
                commandType='DP3'
            )
        
            
        self.data_column = "CORRECTED_DATA"
         
         
    def apply_mask(self, imagename: str, maskfits: str) -> None:
        beam02Reg, _, region = self.mask
        # check if hand-made mask is available
        # Use masking scheme from LOFAR_dd_wsclean
        im = lib_img.Image(imagename+'-MFS-image.fits')
        #im.makeMask(self.s, self.cycle, mode="breizorro", threshpix=5, rmsbox=(50,5), atrous_do=True)#, maskname=maskfits) #Pybdsf step here
        if self.stats == "core":
            im.makeMask(mode="default", threshpix=5, rmsbox=(50,5), atrous_do=True)
        else:
            im.makeMask(mode="default", threshpix=5, rmsbox=(100,27), atrous_do=True)
            
        if (region is not None) and (self.stats == "all") and (not self.doamp):
            logger.info("Manual masks used")
            lib_img.blank_image_reg(maskfits, beam02Reg, blankval = 0.)
            lib_img.blank_image_reg(maskfits, region, blankval = 1.)
        else:
            logger.info("NO Manual mask used")
            
            
    def clean(self, imagename: str, uvlambdamin: int = 30, deep: bool = False) -> None:
        # special for extended sources:
        
        if TARGET in very_extended_targets:
            kwargs1 = {
                'weight': 'briggs -0.5', 
                'taper_gaussian': '75arcsec', 
                'multiscale': '', 
                'multiscale_scale_bias':0.5, 
                'multiscale_scales':'0,30,60,120,340'
            }
            kwargs2 = {
                'weight': 'briggs -0.5', 
                'taper_gaussian': '75arcsec', 
                'multiscale_scales': '0,30,60,120,340'
            }
        elif TARGET in extended_targets:
            kwargs1 = {
                'weight': 'briggs -0.7', 
                'taper_gaussian': '25arcsec'
            }
            kwargs2 = {
                'weight': 'briggs -0.7', 
                'taper_gaussian': '25arcsec', 
                'multiscale_scales': '0,15,30,60,120,240'
            }
        else:
            kwargs1 = {'weight': 'briggs -0.8'}
            kwargs2 = {
                'weight': 'briggs -0.6', 
                'multiscale_scales': '0,10,20,40,80,160'
            }
        
        kwargs1.update({"size": 2500}) # type: ignore
        kwargs2.update({"size": 2500}) # type: ignore
        kwargs1.update({"scale": "2.0arcsec"}) # type: ignore
        kwargs2.update({"scale": "2.0arcsec"}) # type: ignore
            
        if self.stats == "core":
            kwargs1["size"] = 500; kwargs1["scale"] = "50.0arcsec" # type: ignore
            kwargs2["size"] = 500; kwargs2["scale"] = "50.0arcsec" # type: ignore

        # if next is a "cont" then I need the do_predict
        logger.info('Cleaning shallow (cycle: '+str(self.cycle)+')...')
        lib_util.run_wsclean(
            self.s, 
            'wsclean1-c%02i.log' % self.cycle, 
            self.mss.getStrWsclean(), 
            do_predict=True, 
            name=imagename,
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
        
        maskfits = imagename+'-mask.fits'
        self.apply_mask(imagename, maskfits)
        
        if not deep:
            logger.info('Cleaning full (cycle: '+str(self.cycle)+')...')
            lib_util.run_wsclean(
                self.s, 
                'wsclean2-c%02i.log' % self.cycle, 
                self.mss.getStrWsclean(), 
                name=imagename,
                do_predict=True, 
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
                multiscale_scale_bias=0.8,
                join_channels='', 
                fit_spectral_pol=2, 
                channels_out=2, 
                **kwargs2
            )
            os.system('cat logs/wsclean-c%02i.log | grep "background noise"' % self.cycle)
        
        else:
            logger.info('Cleaning full (cycle: '+str(self.cycle)+')...')
            lib_util.run_wsclean(
                self.s, 
                'wsclean2-c%02i.log' % self.cycle, 
                self.mss.getStrWsclean(), 
                name=imagename,
                do_predict=True, 
                cont=True, 
                parallel_gridding=4,
                niter=1000000, 
                no_update_model_required='',
                minuv_l=uvlambdamin, 
                mgain=0.4, 
                nmiter=0,
                auto_threshold=0.5, 
                auto_mask=2., 
                local_rms='', 
                local_rms_method='rms-with-min', 
                fits_mask=maskfits,
                multiscale='', 
                multiscale_scale_bias=0.6,
                join_channels='', 
                fit_spectral_pol=2, 
                channels_out=2, 
                **kwargs2
            )
            os.system('cat logs/wsclean-c%02i.log | grep "background noise"' % self.cycle)            
        
        
    def low_resolution_clean(self, imagename: str, uvlambdamin: int = 30, taper: float=60):
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
            scale='10arcsec', 
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
        self.rms_history.append(rms_noise_pre)
        self.ratio_history.append(mm_ratio_pre)
        
        im = lib_img.Image(imagename+'-MFS-image.fits')
        im.makeMask(self.s, self.cycle, threshpix=5, rmsbox=(500,30), atrous_do=False )
        rms_noise = float(im.getNoise()) 
        mm_ratio = float(im.getMaxMinRatio())
        logger.info('RMS noise: %f - MM ratio: %f' % (rms_noise, mm_ratio))
    
        if self.doamp and rms_noise > 0.99*rms_noise_pre and mm_ratio < 1.01*mm_ratio_pre and self.cycle > 6:
            stopping = True  # if already doing amp and not getting better, quit
        if rms_noise > 0.95*rms_noise_pre and mm_ratio < 1.05*mm_ratio_pre:
            self.doamp = True
            
        return rms_noise, mm_ratio, stopping
