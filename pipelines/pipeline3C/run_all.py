#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys, os
import json
from datetime import datetime

sys.path.append("/data/scripts/LiLF")

from LiLF_lib import lib_util as lilf, lib_log # type: ignore 


Logger_obj = lib_log.Logger('pipeline-3c-all.logger')
Logger = lib_log.logger
WALKER = lilf.Walker('pipeline-3c-all.walker')

MY_DIR = os.getcwd()
DATA_DIR = "/data/data/3Csurvey/tgts/"
PARSET_DIR = "/data/scripts/LiLF/parsets/LOFAR_3c_core/"
SCRIPT_PATH = "/data/scripts/LiLF/pipelines/LOFAR_3c.py"

def backup_target(target: str):
    target_dir = DATA_DIR + target
    
    timestamp = str(datetime.now())
    date, time = timestamp.split(" ")
    time = str(time).split(".")[0]
    backup_dir = target_dir + "/backup_" + date + "_" + time
    
    Logger.info("Backing up to dir: " + backup_dir)
    
    os.mkdir(backup_dir)
    os.system(f"mv {target_dir}/img {backup_dir}")
    os.system(f"mv {target_dir}/plots* {backup_dir}")
    os.system(f"mv {target_dir}/cal* {backup_dir}")
    os.system(f"mv {target_dir}/pipeline* {backup_dir}")
    os.system(f"mv {target_dir}/logs* {backup_dir}")
    os.system(f"mv {target_dir}/rms_noise* {backup_dir}")
    os.system(f"mv {target_dir}/mm_ratio* {backup_dir}")
    os.system(f"cp -r {target_dir}/img {backup_dir}")
    os.system(f"mv {target_dir}/{target}_t*phaseup-final {backup_dir}")

    lilf.check_rm(f"{target_dir}/{target}_t*phaseup")

with open(PARSET_DIR + "source_angular_diameter.json") as file:
    json_data = json.load(file)["data"]
    targets_with_size = [item["name"] for item in json_data]
    

if __name__ == "__main__":
    all_targets = sorted(os.listdir(DATA_DIR))
    Logger.info(", ".join(all_targets))
    
    for target in all_targets[:20]:
        short_completed = False
        target_dir = DATA_DIR + target
        print("my dir:", MY_DIR)
        
        '''
        if target not in targets_with_size:
            with WALKER.if_todo(target + "_short"):
                if os.path.exists(target_dir + "/pipeline-3c.walker"):
                    Logger.warning(f"Walker file exists. Backing up: {target}")
                    backup_target(target)
                
                Logger.info(f"Running core and short all pipeline for: {target}")
                try: 
                    os.system(f"python {SCRIPT_PATH} -t {target} -ca 2 --do_test --do_core_scalar_solve --bl_smooth_fj")
                    os.chdir(MY_DIR)
                    short_completed = True
                except:
                    Logger.warning(f"Failed completeing short pipeline for: {target}")
        else:
            short_completed = True
        '''
        
        short_completed = True
        
        if short_completed:       
            with WALKER.if_todo(target):
                if os.path.exists(target_dir + "/pipeline-3c.walker"):
                    Logger.warning(f"Walker file exists. Backing up: {target}")
                    backup_target(target)
                    
                Logger.info(f"Running full pipeline for target: {target}")
                
                lilf.check_rm(f"{target_dir}/{target}_t*")
                
                try:
                    os.system(f"python {SCRIPT_PATH} -t {target} --do_core_scalar_solve --bl_smooth_fj")
                    os.chdir(MY_DIR)
                except:
                    Logger.warning(f"Failed completeing full pipeline for: {target}")
            
            