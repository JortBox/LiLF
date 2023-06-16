import os, sys, re, time, pickle, random, shutil, glob
import socket

from casacore import tables
import numpy as np
import multiprocessing, subprocess
from threading import Thread
from queue import Queue
import pyregion
import gc
import warnings
from astropy import wcs
from LiLF.lib_img import flatten

if (sys.version_info > (3, 0)):
    from configparser import ConfigParser
else:
    from ConfigParser import ConfigParser

# load here to be sure to have "Agg" at the beginning
import matplotlib as mpl
mpl.use("Agg")

from LiLF.lib_log import logger

def getParset(parsetFile=''):
    """
    Get parset file and return dict of values
    """
    def add_default(section, option, val):
        if not config.has_option(section, option): config.set(section, option, val)
    
    if parsetFile == '' and os.path.exists('lilf.config'): parsetFile='lilf.config'
    if parsetFile == '' and os.path.exists('../lilf.config'): parsetFile='../lilf.config'

    config = ConfigParser(defaults=None)
    config.read(parsetFile)
    
    # add pipeline sections and defaul parset dir:
    for pipeline in glob.glob(os.path.dirname(__file__)+'/../parsets/*'):
        pipeline = os.path.basename(pipeline)
        if not config.has_section(pipeline): config.add_section(pipeline)
        if not config.has_option(pipeline, 'parset_dir'):
                config.set(pipeline, 'parset_dir', os.path.dirname(__file__)+'/../parsets/'+pipeline)
    # add other sections
    if not config.has_section('flag'): config.add_section('flag')
    if not config.has_section('model'): config.add_section('model')
    if not config.has_section('PiLL'): config.add_section('PiLL')

    ### LOFAR ###

    # PiLL
    add_default('PiLL', 'working_dir', os.getcwd())
    add_default('PiLL', 'redo_cal', 'False') # re-do the calibrator although it is in the archive
    add_default('PiLL', 'download_file', '') # html.txt file to use instead of staging
    add_default('PiLL', 'project', '')
    add_default('PiLL', 'target', '')
    add_default('PiLL', 'obsid', '') # unique ID
    # preprocess
    add_default('LOFAR_preprocess', 'fix_table', 'True') # fix bug in some old observations
    add_default('LOFAR_preprocess', 'renameavg', 'True')
    add_default('LOFAR_preprocess', 'flag_elev', 'True')
    add_default('LOFAR_preprocess', 'keep_IS', 'False')
    add_default('LOFAR_preprocess', 'backup_full_res', 'False')
    # demix
    add_default('LOFAR_demix', 'data_dir', './data-bkp/')
    add_default('LOFAR_demix', 'include_target', 'False')
    add_default('LOFAR_demix', 'demix_model', os.path.dirname(__file__)+'/../models/demix_all.skymodel')
    # cal
    add_default('LOFAR_cal', 'data_dir', './data-bkp/')
    add_default('LOFAR_cal', 'skymodel', '') # by default use calib-simple.skydb for LBA and calib-hba.skydb for HBA
    add_default('LOFAR_cal', 'imaging', 'False')
    # timesplit
    add_default('LOFAR_timesplit', 'data_dir', './data-bkp/')
    add_default('LOFAR_timesplit', 'cal_dir', '') # by default the repository is tested, otherwise ../obsid_3[c|C]*
    add_default('LOFAR_timesplit', 'ngroups', '1')
    add_default('LOFAR_timesplit', 'initc', '0')
    # quick-self
    add_default('LOFAR_quick-self', 'data_dir', './data-bkp/')
    # dd-parallel - deprecated
    #add_default('LOFAR_dd-parallel', 'maxniter', '10')
    #add_default('LOFAR_dd-parallel', 'calFlux', '1.5')
    # dd
    add_default('LOFAR_dd', 'maxIter', '2')
    add_default('LOFAR_dd', 'minCalFlux60', '1')
    add_default('LOFAR_dd', 'removeExtendedCutoff', '0.0005')
    add_default('LOFAR_dd', 'target_dir', '') # ra,dec
    # extract
<<<<<<< HEAD
    add_default('LOFAR_extract', 'max_niter', '10')
    add_default('LOFAR_extract', 'extract_region', 'target.reg')
    add_default('LOFAR_extract', 'subtract_region', '') # Sources inside extract-reg that should still be subtracted! Use this e.g. for individual problematic sources in a large extractReg
    add_default('LOFAR_extract', 'ph_sol_mode', 'phase') # tecandphase, phase
    add_default('LOFAR_extract', 'amp_sol_mode', 'diagonal') # diagonal, fulljones
    add_default('LOFAR_extract', 'beam_cut', '0.3') # up to which distance a pointing will be considered
    add_default('LOFAR_extract', 'fits_mask', '') # use a fits mask for cleaning - needs to have same dimensions as output image, cannot be combined with userReg
    add_default('LOFAR_extract', 'no_selfcal', 'False') # just extract the data, do not perform selfcal - use this if u want to use e.g. Reinout van Weeren's facet_seflcal script
    # quality
    add_default('LOFAR_quality', 'self_dir', 'self')
    add_default('LOFAR_quality', 'ddcal_dir', 'ddcal')
=======
    add_default('LOFAR_extract', 'maxniter', '10')
    add_default('LOFAR_extract', 'phSolMode', 'phase') # tecandphase, phase
    add_default('LOFAR_extract', 'beam_cut', '0.3') # up to which distance a pointing will be considered
    add_default('LOFAR_extract', 'ampcal', 'auto')
    add_default('LOFAR_extract', 'extractRegion', 'target.reg')
>>>>>>> master
    # virgo
    add_default('LOFAR_virgo', 'cal_dir', '')
    add_default('LOFAR_virgo', 'data_dir', './')
    # m87
    add_default('LOFAR_m87', 'data_dir', './')
    add_default('LOFAR_m87', 'updateweights', 'False')
    add_default('LOFAR_m87', 'skipmodel', 'False')
    add_default('LOFAR_m87', 'model_dir', '')
    # peel
    #add_default('LOFAR_peel', 'peelReg', 'peel.reg')
    #add_default('LOFAR_peel', 'predictReg', '')
    #add_default('LOFAR_peel', 'cal_dir', '')
    #add_default('LOFAR_peel', 'data_dir', './')

    ### uGMRT ###
    # init - deprecated
    #add_default('uGMRT_init', 'data_dir', './datadir')
    # cal - deprecated
    #add_default('uGMRT_cal', 'skymodel', os.path.dirname(__file__)+'/../models/calib-simple.skydb')

    ### General ###

    # flag
    add_default('flag', 'stations', '') # LOFAR
    add_default('flag', 'antennas', '') # uGMRT
    # model
    add_default('model', 'sourcedb', '')
    add_default('model', 'fits_model', '')
    add_default('model', 'apparent', 'False')
    add_default('model', 'userReg', '')


    return config

def create_extregion(ra, dec, extent):
    """
    Parameters
    ----------
    ra
    dec
    extent

    Returns
    -------
    DS9 region centered on ra, dec with radius = extent
    """

    regtext = ['# Region file format: DS9 version 4.1']
    regtext.append(
        'global color=yellow dashlist=8 3 width=1 font="helvetica 10 normal roman" select=1 highlite=1 dash=0 fixed=0 edit=1 move=1 delete=1 include=1 source=1')
    regtext.append('fk5')
    regtext.append('circle(' + str(ra) + ',' + str(dec) + f',{extent})')
    nline = '\n'
    target = f"{nline}{nline.join(regtext)}"

    return target


def columnAddSimilar(pathMS, columnNameNew, columnNameSimilar, dataManagerInfoNameNew, overwrite = False, fillWithOnes = True, comment = "", verbose = False):
    # more to lib_ms
    """
    Add a column to a MS that is similar to a pre-existing column (in shape, but not in values).
    pathMS:                 path of the MS
    columnNameNew:          name of the column to be added
    columnNameSimilar:      name of the column from which properties are copied (e.g. "DATA")
    dataManagerInfoNameNew: string value for the data manager info (DMI) keyword "NAME" (should be unique in the MS)
    overwrite:              whether or not to overwrite column 'columnNameNew' if it already exists
    fillWithOnes:           whether or not to fill the newly-made column with ones
    verbose:                whether or not to produce abundant output
    """
    t = tables.table(pathMS, readonly = False)

    if (columnExists(t, columnNameNew) and not overwrite):
        logger.warning("Attempt to add column '" + columnNameNew + "' aborted, as it already exists and 'overwrite = False' in columnAddSimilar(...).")
    else: # Either the column does not exist yet, or it does but overwriting is allowed.

        # Remove column if necessary.
        if (columnExists(t, columnNameNew)):
            logger.info("Removing column '" + columnNameNew + "'...")
            t.removecols(columnNameNew)

        # Add column.
        columnDescription       = t.getcoldesc(columnNameSimilar)
        dataManagerInfo         = t.getdminfo(columnNameSimilar)

        if (verbose):
            logger.debug("columnDescription:")
            logger.debug(columnDescription)
            logger.debug("dataManagerInfo:")
            logger.debug(dataManagerInfo)

        columnDescription["comment"] = ""
        # What about adding something here like:
        #columnDescription["dataManagerGroup"] = ...?
        dataManagerInfo["NAME"]      = dataManagerInfoNameNew

        if (verbose):
            logger.debug("columnDescription (updated):")
            logger.debug(columnDescription)
            logger.debug("dataManagerInfo (updated):")
            logger.debug(dataManagerInfo)

        logger.info("Adding column '" + columnNameNew + "'...")
        t.addcols(tables.makecoldesc(columnNameNew, columnDescription), dataManagerInfo)

        # Fill with ones if desired.
        if (fillWithOnes):
            logger.info("Filling column '" + columnNameNew + "' with ones...")
            columnDataSimilar = t.getcol(columnNameSimilar)
            t.putcol(columnNameNew, np.ones_like(columnDataSimilar))

    # Close the table to avoid that it is locked for further use.
    t.close()


def getCalibratorProperties():
    """
    Return properties of known calibrators.
    The lists below (sorted in RA) are incomplete,
    and should be expanded to include all calibrators that could possibly be used.
    """

    calibratorRAs           = np.array([24.4220808, 85.6505746, 123.4001379, 202.784479167, 202.8569, 212.835495, 277.3824204, 299.8681525]) # in degrees
    calibratorDecs          = np.array([33.1597594, 49.8520094, 48.2173778,  30.509088,     25.1429,  52.202770,  48.7461556,  40.7339156])  # in degrees
    calibratorNames         = np.array(["3C48",     "3C147",    "3C196",     "3C286",       "3C287",  "3C295",    "3C380",     "CygA"])

    return calibratorRAs, calibratorDecs, calibratorNames


def distanceOnSphere(RAs1, Decs1, RAs2, Decs2, rad=False):
    """
    Return the distances on the sphere from the set of points '(RAs1, Decs1)' to the
    set of points '(RAs2, Decs2)' using the spherical law of cosines.

    Using 'numpy.clip(..., -1, 1)' is necessary to counteract the effect of numerical errors, that can sometimes
    incorrectly cause '...' to be slightly larger than 1 or slightly smaller than -1. This leads to NaNs in the arccosine.
    """
    if rad: # rad in rad out
        return np.radians(np.arccos(np.clip(
            np.sin(Decs1) * np.sin(Decs2) +
            np.cos(Decs1) * np.cos(Decs2) *
            np.cos(RAs1 - RAs2), -1, 1)))
    else: # deg in deg out
        return np.degrees(np.arccos(np.clip(
               np.sin(np.radians(Decs1)) * np.sin(np.radians(Decs2)) +
               np.cos(np.radians(Decs1)) * np.cos(np.radians(Decs2)) *
               np.cos(np.radians(RAs1 - RAs2)), -1, 1)))


def check_rm(regexp):
    """
    Check if file exists and remove it
    Handle reg exp of glob and spaces
    """
    filenames = regexp.split(' ')
    for filename in filenames:
        # glob is used to check if file exists
        for f in glob.glob(filename):
            os.system("rm -r " + f)


class Sol_iterator(object):
    """
    Iterator on a list that keeps on returing
    the last element when the list is over
    """

    def __init__(self, vals=[]):
        self.vals = vals
        self.pos = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.pos < len(self.vals):
            val = self.vals[self.pos]
            self.pos += 1
            return val
        else:
            return self.vals[-1]


def lofar_nu2num(nu):
    """
    Get LOFAR SB number from the freq
    """
    nu_clk = 200. # 160 or 200 MHz, clock freq
    # nyquist zone (1 for LBA, 2 for HBA low, 3 for HBA mid-high)
    if nu < 90:
        n = 1
    elif nu < 190:
        n = 2
    else:
        n = 3

    if nu_clk == 200:
        SBband = 195312.5/1e6
    elif nu_clk == 160:
        SBband = 156250.0/1e6

    return np.int(np.floor((1024./nu_clk) * (nu - (n-1) * nu_clk/2.)))

def run_losoto(s, c, h5s, parsets, plots_dir=None) -> object:
    """
    s : scheduler
    c : cycle name, e.g. "final"
    h5s : lists of H5parm files or string of 1 h5parm
    parsets : lists of parsets to execute
    """

    logger.info("Running LoSoTo...")

    h5out = 'cal-'+c+'.h5'

    if type(h5s) is str: h5s = [h5s]

    # convert from killMS
    for i, h5 in enumerate(h5s):
        if h5[-3:] == 'npz':
            newh5 = h5.replace('.npz','.h5')
            s.add('killMS2H5parm.py -V --nofulljones %s %s ' % (newh5, h5), log='losoto-'+c+'.log', commandType="python", processors='max')
            s.run(check = True)
            h5s[i] = newh5

    # concat/move
    if len(h5s) > 1:
        check_rm(h5out)
        s.add('H5parm_collector.py -V -s sol000 -o '+h5out+' '+' '.join(h5s), log='losoto-'+c+'.log', commandType="python", processors='max')
        s.run(check = True)
    else:
        os.system('cp -r %s %s' % (h5s[0], h5out) )

    check_rm('plots')
    os.makedirs('plots')

    for parset in parsets:
        logger.debug('-- executing '+parset+'...')
        s.add('losoto -V '+h5out+' '+parset, log='losoto-'+c+'.log', logAppend=True, commandType="python", processors='max')
        s.run(check = True)

    if plots_dir is None:
        check_rm('plots-' + c)
        os.system('mv plots plots-' + c)
    else:
        if not os.path.exists(plots_dir): os.system('mkdir '+plots_dir)
        os.system('mv plots/* '+plots_dir)
        check_rm('plots')


def run_wsclean(s, logfile, MSs_files, do_predict=False, **kwargs):
    """
    s : scheduler
    args : parameters for wsclean, "_" are replaced with "-", any parms=None is ignored.
           To pass a parameter with no values use e.g. " no_update_model_required='' "
    """
    
    wsc_parms = []
    reordering_processors = np.min([len(MSs_files),s.max_processors])

    # basic parms
    wsc_parms.append( '-j '+str(s.max_processors)+' -reorder -parallel-reordering 4 ' )
    if 'use_idg' in kwargs.keys():
        if s.get_cluster() == 'Hamburg_fat' and socket.gethostname() in ['node31', 'node32', 'node33', 'node34', 'node35']:
            wsc_parms.append( '-idg-mode hybrid' )
            wsc_parms.append( '-mem 10' )
        else:
            wsc_parms.append( '-idg-mode cpu' )

    # other stanrdard parms
    wsc_parms.append( '-clean-border 1' )
    # temp dir
    #if s.get_cluster() == 'Hamburg_fat' and not 'temp_dir' in list(kwargs.keys()):
    #    wsc_parms.append( '-temp-dir /localwork.ssd' )
    # user defined parms
    for parm, value in list(kwargs.items()):
        if value is None: continue
        if parm == 'baseline_averaging' and value == '':
            scale = float(kwargs['scale'].replace('arcsec','')) # arcsec
            value = 1.87e3*60000.*2.*np.pi/(24.*60.*60*np.max(kwargs['size'])) # the np.max() is OK with both float and arrays
            if value > 10: value=10
            if value < 1: continue
        if parm == 'cont': 
            parm = 'continue'
            value = ''
        if parm == 'size' and type(value) is int: value = '%i %i' % (value, value)
        if parm == 'size' and type(value) is list: value = '%i %i' % (value[0], value[1])
        wsc_parms.append( '-%s %s' % (parm.replace('_','-'), str(value)) )

    # files
    wsc_parms.append( MSs_files )

    # create command string
    command_string = 'wsclean '+' '.join(wsc_parms)
    s.add(command_string, log=logfile, commandType='wsclean', processors='max')
    s.run(check=True)

    # Predict in case update_model_required cannot be used
    if do_predict == True:
        wsc_parms = []
        # keep imagename and channel number
        for parm, value in list(kwargs.items()):
            if value is None: continue
            #if 'min' in parm or 'max' in parm or parm == 'name' or parm == 'channels_out':
            if parm == 'name' or parm == 'channels_out' or parm == 'use_wgridder' or parm == 'wgridder_accuracy':
                wsc_parms.append( '-%s %s' % (parm.replace('_','-'), str(value)) )

        # files
        wsc_parms.append( MSs_files )
        # Test without reorder as it apperas to be faster
        # wsc_parms.insert(0, ' -reorder -parallel-reordering 4 ')
        command_string = 'wsclean -predict ' \
                         '-j '+str(s.max_processors)+' '+' '.join(wsc_parms)
        s.add(command_string, log=logfile, commandType='wsclean', processors='max')
        s.run(check=True)

def run_DDF(s, logfile, **kwargs):
    """
    s : scheduler
    args : parameters for ddfacet, "_" are replaced with "-", any parms=None is ignored.
           To pass a parameter with no values use e.g. " no_update_model_required='' "
    """
    
    ddf_parms = []

    # basic parms
    ddf_parms.append( '--Log-Boring 1 --Debug-Pdb never --Parallel-NCPU %i --Misc-IgnoreDeprecationMarking=1 ' % (s.max_processors) )

    # cache dir
    if not 'Cache_Dir' in list(kwargs.keys()):
        ddf_parms.append( '--Cache-Dir .' )

    # user defined parms
    for parm, value in list(kwargs.items()):
        if value is None: continue
        if isinstance(value, str):
            if '$' in value: # escape dollar signs (e.g. of BeamFits)
                value = "'" + value + "'"
        ddf_parms.append( '--%s=%s' % (parm.replace('_','-'), str(value)) )

    # files
    #wsc_parms.append( MSs_files )

    # create command string
    command_string = 'DDF.py '+' '.join(ddf_parms)
    s.add(command_string, log=logfile, commandType='DDFacet', processors='max')
    s.run(check=True)


class Region_helper():
    """
    Simple class to get the extent of a ds9 region file containing one or more circles or polygons.
    All properties are returned in degrees.

    Parameters
    ----------
    filename: str
        Path to ds9 region file.
    """
    def __init__(self, filename):
        self.filename = filename
        self.reg_list = pyregion.open(filename)
        min_ra, max_ra, min_dec, max_dec = [], [], [], []
        for r in self.reg_list:
            # TODO: if necessary, box, ellipse and polygon can be added.
            if r.name == 'circle':
                c = r.coord_list # c_ra, c_dec, radius
                # how much RA does the radius correspond to
                radius_ra = np.rad2deg(2*np.arcsin(np.sin(np.deg2rad(c[2])/2)/np.cos(np.deg2rad(c[1]))))
                min_ra.append(c[0] - radius_ra)
                max_ra.append(c[0] + radius_ra)
                min_dec.append(c[1] - c[2])
                max_dec.append(c[1] + c[2])
            elif r.name == 'polygon':
                c = np.array(r.coord_list) # ra_i, dec_i, ra_i+1, dec_i+1
                ra_mask = np.zeros(len(c), dtype=bool)
                ra_mask[::2] = True
                p_ra  = c[ra_mask]
                p_dec = c[~ra_mask]
                min_ra.append(np.min(p_ra))
                max_ra.append(np.max(p_ra))
                min_dec.append(np.min(p_dec))
                max_dec.append(np.max(p_dec))
            else:
                logger.error('Region type {} not supported.'.format(r.name))
                sys.exit(1)
        self.min_ra = np.min(min_ra)
        self.max_ra = np.max(max_ra)
        self.min_dec = np.min(min_dec)
        self.max_dec = np.max(max_dec)

    def get_center(self):
        """ Return center point [ra, dec] """
        return 0.5 * np.array([self.min_ra + self.max_ra, self.min_dec + self.max_dec])

    def get_width(self):
        """ Return RA width in degree (at center declination)"""
        delta_ra = self.max_ra - self.min_ra
        width = 2*np.arcsin(np.cos(np.deg2rad(self.get_center()[1]))*np.sin(np.deg2rad(delta_ra/2)))
        width = np.rad2deg(width)
        return width

    def get_height(self):
        """ Return height in degree"""
        return self.max_dec - self.min_dec

    def __len__(self):
        return len(self.reg_list)


class Skip(Exception):
    pass


class Walker():
    """
    An object of this class may be used to re-run a pipeline without repeating steps that were completed previously.
    Use like:
    w = Walker("filename.walker")
    with w.if_todo("stepname"):
        Do whatever...

    Adopted from https://stackoverflow.com/questions/12594148/skipping-execution-of-with-block
    """
    def __init__(self, filename):
        open(filename, 'a').close() # create the file if doesn't exists
        self.filename = os.path.abspath(filename)
        self.__skip__ = False
        self.__step__ = None

    def if_todo(self, stepname):
        """
        This is basically a way to get a context manager to accept an argument. Will return "self" as context manager
        if called as context manager.
        """
        self.__skip__ = False
        self.__step__ = stepname
        with open(self.filename, "r") as f:
            for stepname_done in f:
                if stepname == stepname_done.rstrip():
                    self.__skip__ = True
        return self

    def __enter__(self):
        """
        Skips body of with-statement if __skip__.
        This uses some kind of dirty hack that might only work in CPython.
        """
        if self.__skip__:
            sys.settrace(lambda *args, **keys: None)
            frame = sys._getframe(1)
            frame.f_trace = self.trace
        else:
            logger.log(20, '>> start >> {}'.format(self.__step__))


    def trace(self, frame, event, arg):
        raise Skip()

    def __exit__(self, type, value, traceback):
        """
        Catch "Skip" errors, if not skipped, write to file after exited without exceptions.
        """
        if type is None:
            with open(self.filename, "a") as f:
                f.write(self.__step__ + '\n')
            logger.info('<< done << {}'.format(self.__step__))
            return  # No exception
        if issubclass(type, Skip):
            logger.warning('>> skip << {}'.format(self.__step__))
            return True  # Suppress special SkipWithBlock exception

class Scheduler():
    def __init__(self, qsub = None, maxThreads = None, max_processors = None, log_dir = 'logs', dry = False):
        """
        qsub:           if true call a shell script which call qsub and then wait
                        for the process to finish before returning
        maxThreads:    max number of parallel processes
        dry:            don't schedule job
        max_processors: max number of processors in a node (ignored if qsub=False)
        """
        self.hostname = socket.gethostname()
        self.cluster = self.get_cluster()
        self.log_dir = log_dir
        self.qsub    = qsub
        # if qsub/max_thread/max_processors not set, guess from the cluster
        # if they are set, double check number are reasonable
        if (self.qsub == None):
            if (self.cluster == "Hamburg"):
                self.qsub = True
            else:
                self.qsub = False
        else:
            if ((self.qsub is False and self.cluster == "Hamburg") or
               (self.qsub is True and (self.cluster == "Leiden" or self.cluster == "CEP3" or
                                       self.cluster == "Hamburg_fat" or self.cluster == "Pleiadi" or self.cluster == "Herts"))):
                logger.critical('Qsub set to %s and cluster is %s.' % (str(qsub), self.cluster))
                sys.exit(1)

        if (maxThreads is None):
            if (self.cluster == "Hamburg"):
                self.maxThreads = 32
            else:
                self.maxThreads = multiprocessing.cpu_count()
        else:
            self.maxThreads = maxThreads

        if (max_processors == None):
            if   (self.cluster == "Hamburg"):
                self.max_processors = 6
            else:
                self.max_processors = multiprocessing.cpu_count()
        else:
            self.max_processors = max_processors

        self.dry = dry
<<<<<<< HEAD
        logger.info("Scheduler initialised for cluster " + self.cluster + ": " + self.hostname + " (maxThreads: " + str(self.maxThreads) + ", qsub (multinode): " +
                     str(self.qsub) + ", max_processors: " + str(self.max_processors) + ").")
=======
        #logger.info("Scheduler initialised for cluster " + self.cluster + " (maxThreads: " + str(self.maxThreads) + ", qsub (multinode): " +
                    # str(self.qsub) + ", max_processors: " + str(self.max_processors) + ").")
>>>>>>> master

        self.action_list = []
        self.log_list    = []  # list of 2-tuples of the type: (log filename, type of action)


    def get_cluster(self):
        """
        Find in which computing cluster the pipeline is running
        """
        hostname = self.hostname
        if (hostname == 'lgc1' or hostname == 'lgc2'):
            return "Hamburg"
        elif ('r' == hostname[0] and 'c' == hostname[3] and 's' == hostname[6]):
            return "Pleiadi"
        elif ('node3' in hostname):
            return "Hamburg_fat"
        elif ('node' in hostname):
            return "Herts"
        elif ('leidenuniv' in hostname):
            return "Leiden"
        elif (hostname[0 : 3] == 'lof'):
            return "CEP3"
        else:
            logger.warning('Hostname %s unknown.' % hostname)
            return "Unknown"


    def add(self, cmd = '', log = '', logAppend = True, commandType = '', processors = None):
        """
        Add a command to the scheduler list
        cmd:         the command to run
        log:         log file name that can be checked at the end
        logAppend:  if True append, otherwise replace
        commandType: can be a list of known command types as "wsclean", "DP3", ...
        processors:  number of processors to use, can be "max" to automatically use max number of processors per node
        """

        if (log != ''):
            log = self.log_dir + '/' + log

            if (logAppend):
                cmd += " >> "
            else:
                cmd += " > "
            cmd += log + " 2>&1"

        # if running wsclean add the string
        if commandType == 'wsclean':
            logger.debug('Running wsclean: %s' % cmd)
        elif commandType == 'DP3':
            logger.debug('Running DP3: %s' % cmd)
        #elif commandType == 'singularity':
        #    cmd = 'SINGULARITY_TMPDIR=/dev/shm singularity exec -B /tmp,/dev/shm,/localwork,/localwork.ssd,/home /home/fdg/node31/opt/src/lofar_sksp_ddf.simg ' + cmd
        #    logger.debug('Running singularity: %s' % cmd)
        elif (commandType.lower() == "ddfacet" or commandType.lower() == 'ddf'):
            logger.debug('Running DDFacet: %s' % cmd)
        elif commandType == 'python':
            logger.debug('Running python: %s' % cmd)

        if (processors != None and processors == 'max'):
            processors = self.max_processors

        if self.qsub:
            # if number of processors not specified, try to find automatically
            if (processors == None):
                processors = 1 # default use single CPU
                if ("DP3" == cmd[ : 4]):
                    processors = 1
                if ("wsclean" == cmd[ : 7]):
                    processors = self.max_processors
            if (processors > self.max_processors):
                processors = self.max_processors

            self.action_list.append([str(processors), '\'' + cmd + '\''])
        else:
            self.action_list.append(cmd)

        if (log != ""):
            self.log_list.append((log, commandType))


    def run(self, check = False, maxThreads = None):
        """
        If 'check' is True, a check is done on every log in 'self.log_list'.
        If max_thread != None, then it overrides the global values, useful for special commands that need a lower number of threads.
        """

        def worker(queue):
            for cmd in iter(queue.get, None):
                if self.qsub and self.cluster == "Hamburg":
                    cmd = 'salloc --job-name LBApipe --time=24:00:00 --nodes=1 --tasks-per-node='+cmd[0]+\
                            ' /usr/bin/srun --ntasks=1 --nodes=1 --preserve-env \''+cmd[1]+'\''
                gc.collect()
                subprocess.call(cmd, shell = True)

        # limit threads only when qsub doesn't do it
        if (maxThreads == None):
            maxThreads_run = self.maxThreads
        else:
            maxThreads_run = min(maxThreads, self.maxThreads)

        q       = Queue()
        threads = [Thread(target = worker, args=(q,)) for _ in range(maxThreads_run)]

        for i, t in enumerate(threads): # start workers
            t.daemon = True
            t.start()

        for action in self.action_list:
            if (self.dry):
                continue # don't schedule if dry run
            q.put_nowait(action)
        for _ in threads:
            q.put(None) # signal no more commands
        for t in threads:
            t.join()

        # check outcomes on logs
        if (check):
            for log, commandType in self.log_list:
                self.check_run(log, commandType)

        # reset list of commands
        self.action_list = []
        self.log_list    = []


    def check_run(self, log = "", commandType = ""):
        """
        Produce a warning if a command didn't close the log properly i.e. it crashed
        NOTE: grep, -L inverse match, -l return only filename
        """

        if (not os.path.exists(log)):
            logger.warning("No log file found to check results: " + log)
            return 1

        if (commandType == "DP3"):
            out = subprocess.check_output('grep -L "Finishing processing" '+log+' ; exit 0', shell = True, stderr = subprocess.STDOUT)
            out += subprocess.check_output('grep -l "Segmentation fault\|Killed" '+log+' ; exit 0', shell = True, stderr = subprocess.STDOUT)
            # TODO: This needs to be uncommented once the malloc_consolidate stuff is fixed
            # out += subprocess.check_output('grep -l "Aborted (core dumped)" '+log+' ; exit 0', shell = True, stderr = subprocess.STDOUT)
            out += subprocess.check_output('grep -i -l "Exception" '+log+' ; exit 0', shell = True, stderr = subprocess.STDOUT)
            out += subprocess.check_output('grep -l "**** uncaught exception ****" '+log+' ; exit 0', shell = True, stderr = subprocess.STDOUT)
            # this interferes with the missingantennabehaviour=error option...
            # out += subprocess.check_output('grep -l "error" '+log+' ; exit 0', shell = True, stderr = subprocess.STDOUT)
            out += subprocess.check_output('grep -l "misspelled" '+log+' ; exit 0', shell = True, stderr = subprocess.STDOUT)

        elif (commandType == "CASA"):
            out = subprocess.check_output('grep -l "[a-z]Error" '+log+' ; exit 0', shell = True, stderr = subprocess.STDOUT)
            out += subprocess.check_output('grep -l "An error occurred running" '+log+' ; exit 0', shell = True, stderr = subprocess.STDOUT)
            out += subprocess.check_output('grep -l "\*\*\* Error \*\*\*" '+log+' ; exit 0', shell = True, stderr = subprocess.STDOUT)

        elif (commandType == "wsclean"):
            out = subprocess.check_output('grep -l "exception occur" '+log+' ; exit 0', shell = True, stderr = subprocess.STDOUT)
            out += subprocess.check_output('grep -l "Segmentation fault\|Killed" '+log+' ; exit 0', shell = True, stderr = subprocess.STDOUT)
            out += subprocess.check_output('grep -l "Aborted" '+log+' ; exit 0', shell = True, stderr = subprocess.STDOUT)
            # out += subprocess.check_output('grep -L "Cleaning up temporary files..." '+log+' ; exit 0', shell = True, stderr = subprocess.STDOUT)

        elif (commandType.lower() == "ddfacet" or commandType.lower() == 'ddf'):
            out = subprocess.check_output('grep -l "Traceback (most recent call last):" '+log+' ; exit 0', shell = True, stderr = subprocess.STDOUT)
            out += subprocess.check_output('grep -l "exception occur" ' + log + ' ; exit 0', shell=True, stderr=subprocess.STDOUT)
            out += subprocess.check_output('grep -l "raise Exception" '+log+' ; exit 0', shell = True, stderr = subprocess.STDOUT)
            out += subprocess.check_output('grep -l "Segmentation fault\|Killed" ' + log + ' ; exit 0', shell=True,
                                           stderr=subprocess.STDOUT)
            out += subprocess.check_output('grep -l "killed by signal" ' + log + ' ; exit 0', shell=True,
                                           stderr=subprocess.STDOUT)
            out += subprocess.check_output('grep -l "Aborted" ' + log + ' ; exit 0', shell=True, stderr=subprocess.STDOUT)

        elif (commandType == "python"):
            out = subprocess.check_output('grep -l "Traceback (most recent call last):" '+log+' ; exit 0', shell = True, stderr = subprocess.STDOUT)
            out += subprocess.check_output('grep -l "Segmentation fault\|Killed" '+log+' ; exit 0', shell = True, stderr = subprocess.STDOUT)
            out += subprocess.check_output('grep -i -l \'(?=^((?!error000).)*$).*Error.*\' '+log+' ; exit 0', shell = True, stderr = subprocess.STDOUT)
            out += subprocess.check_output('grep -i -l "Critical" '+log+' ; exit 0', shell = True, stderr = subprocess.STDOUT)
            out += subprocess.check_output('grep -l "ERROR" '+log+' ; exit 0', shell = True, stderr = subprocess.STDOUT)
            out += subprocess.check_output('grep -l "raise Exception" '+log+' ; exit 0', shell = True, stderr = subprocess.STDOUT)

#        elif (commandType == "singularity"):
#            out = subprocess.check_output('grep -l "Traceback (most recent call last):" '+log+' ; exit 0', shell = True, stderr = subprocess.STDOUT)
#            out += subprocess.check_output('grep -i -l \'(?=^((?!error000).)*$).*Error.*\' '+log+' ; exit 0', shell = True, stderr = subprocess.STDOUT)
#            out += subprocess.check_output('grep -i -l "Critical" '+log+' ; exit 0', shell = True, stderr = subprocess.STDOUT)

        elif (commandType == "general"):
            out = subprocess.check_output('grep -l -i "error" '+log+' ; exit 0', shell = True, stderr = subprocess.STDOUT)

        else:
            logger.warning("Unknown command type for log checking: '" + commandType + "'")
            return 1

        if out != b'':
            out = out.split(b'\n')[0].decode()
            logger.error(commandType+' run problem on:\n'+out)
            raise RuntimeError(commandType+' run problem on:\n'+out)

        return 0


class radiomap:
    """As from Martin Hardcastle's radioflux script"""

    def __init__(self, fitsfile, verbose=False):
        # Catch warnings to avoid datfix errors
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            gfactor = 2.0 * np.sqrt(2.0 * np.log(2.0))
            self.f = fitsfile[0]
            self.prhd = fitsfile[0].header

            # Get units and resolution
            self.units = self.prhd.get('BUNIT')
            if self.units is None:
                self.units = self.prhd.get('UNIT')
            if self.units != 'JY/BEAM' and self.units != 'Jy/beam':
                print('Warning: units are', self.units, 'but code expects JY/BEAM')
            self.bmaj = self.prhd.get('BMAJ')
            self.bmin = self.prhd.get('BMIN')
            if self.bmaj is None:
                # Try RESOL1 and RESOL2
                self.bmaj = self.prhd.get('RESOL1')
                self.bmin = self.prhd.get('RESOL2')
            if self.bmaj is None:
                if verbose:
                    print('Can\'t find BMAJ in headers, checking history')
                try:
                    history = self.prhd['HISTORY']
                except KeyError:
                    history = None
                if history is not None:
                    for line in history:
                        if 'HISTORY' in line:
                            continue  # stops it finding nested history
                        if 'CLEAN BMAJ' in line:
                            bits = line.split()
                            self.bmaj = float(bits[3])
                            self.bmin = float(bits[5])

            if self.bmaj is None:
                raise RadioError('No beam information found')

            w = wcs.WCS(self.prhd)
            cd1 = -w.wcs.cdelt[0]
            cd2 = w.wcs.cdelt[1]
            if ((cd1 - cd2) / cd1) > 1.0001 and ((self.bmaj - self.bmin) / self.bmin) > 1.0001:
                raise RadioError('Pixels are not square (%g, %g) and beam is elliptical' % (cd1, cd2))

            self.bmaj /= cd1
            self.bmin /= cd2
            if verbose:
                print('beam is', self.bmaj, 'by', self.bmin, 'pixels')

            self.area = 2.0 * np.pi * (self.bmaj * self.bmin) / (gfactor * gfactor)
            if verbose:
                print('beam area is', self.area, 'pixels')

            # Remove any PC... keywords we may have, they confuse the pyregion WCS
            for i in range(1, 5):
                for j in range(1, 5):
                    self.quiet_remove('PC0%i_0%i' % (i, j))

            # Now check what sort of a map we have
            naxis = len(fitsfile[0].data.shape)
            if verbose: print('We have', naxis, 'axes')
            self.cube = False
            if naxis < 2 or naxis > 4:
                raise RadioError('Too many or too few axes to proceed (%i)' % naxis)
            if naxis > 2:
                # a cube, what sort?
                frequency = 0
                self.cube = True
                freqaxis = -1
                stokesaxis = -1
                for i in range(3, naxis + 1):
                    ctype = self.prhd.get('CTYPE%i' % i)
                    if 'FREQ' in ctype:
                        freqaxis = i
                    elif 'STOKES' in ctype:
                        stokesaxis = i
                    elif 'VOPT' in ctype:
                        pass
                    else:
                        print('Warning: unknown CTYPE %i = %s' % (i, ctype))
                if verbose:
                    print('This is a cube with freq axis %i and Stokes axis %i' % (freqaxis, stokesaxis))
                if stokesaxis > 0:
                    nstokes = self.prhd.get('NAXIS%i' % stokesaxis)
                    if nstokes > 1:
                        raise RadioError('Multiple Stokes parameters present, not handled')
                if freqaxis > 0:
                    nchans = self.prhd.get('NAXIS%i' % freqaxis)
                    if verbose:
                        print('There are %i channel(s)' % nchans)
                    self.nchans = nchans
            else:
                self.nchans = 1

            # Various possibilities for the frequency. It's possible
            # that a bad (zero) value will be present, so keep
            # checking if one is found.

            if not (self.cube) or freqaxis < 0:
                # frequency, if present, must be in another keyword
                frequency = self.prhd.get('RESTFRQ')
                if frequency is None or frequency == 0:
                    frequency = self.prhd.get('RESTFREQ')
                if frequency is None or frequency == 0:
                    frequency = self.prhd.get('FREQ')
                if frequency is None or frequency == 0:
                    # It seems some maps present with a FREQ ctype
                    # even if they don't have the appropriate axes!
                    # The mind boggles.
                    for i in range(5):
                        type_s = self.prhd.get('CTYPE%i' % i)
                        if type_s is not None and type_s[0:4] == 'FREQ':
                            frequency = self.prhd.get('CRVAL%i' % i)
                self.frq = [frequency]
                if self.cube:
                    # a cube with no freq axis, e.g. VOPT. need to flatten
                    header, data = flatten(fitsfile, freqaxis=freqaxis)
                    self.headers = [header]
                    self.d = [data]
                    self.nchans = 1
                else:
                    # now if there _are_ extra headers, get rid of them so pyregion WCS can work
                    for i in range(3, 5):
                        for k in ['CTYPE', 'CRVAL', 'CDELT', 'CRPIX', 'CROTA', 'CUNIT']:
                            self.quiet_remove(k + '%i' % i)
                    self.headers = [self.prhd]
                    self.d = [fitsfile[0].data]
            else:
                # if this is a cube, frequency/ies should be in freq header
                basefreq = self.prhd.get('CRVAL%i' % freqaxis)
                deltafreq = self.prhd.get('CDELT%i' % freqaxis)
                self.frq = [basefreq + deltafreq * i for i in range(nchans)]
                self.d = []
                self.headers = []
                for i in range(nchans):
                    header, data = flatten(fitsfile, freqaxis=freqaxis, channel=i)
                    self.d.append(data)
                    self.headers.append(header)
            for i, f in enumerate(self.frq):
                if f is None:
                    print('Warning, can\'t get frequency %i -- set to zero' % i)
                    self.frq[i] = 0
            if verbose:
                print('Frequencies are', self.frq, 'Hz')

    def quiet_remove(self, keyname):
        if self.prhd.get(keyname, None) is not None:
            self.prhd.remove(keyname)

#            self.fhead,self.d=flatten(fitsfile)