############################################################################
# AUTHORS: Pedro Margolles & David Soto
# EMAIL: pmargolles@bcbl.eu, dsoto@bcbl.eu
# COPYRIGHT: Copyright (C) 2021-2022, pyDecNef
# URL: https://pedromargolles.github.io/pyDecNef/
# INSTITUTION: Basque Center on Cognition, Brain and Language (BCBL), Spain
# LICENCE: GNU General Public License v3.0
############################################################################

from pathlib import Path
import time
from colorama import init, Fore
import sys
import os
import configparser
from joblib import load,dump
init(autoreset=True)

#############################################################################################
# EXP CLASS
#############################################################################################

# Main configuration file 
# Here you can set experimental, MRI scanner parameters and directory routes

config_file =  os.path.join(Path(__file__).absolute().parent.parent.parent.parent,"config.ini")
config = configparser.ConfigParser()
config.read(config_file)


def string_to_bool(string):
   if string == "True":
      return True
   if string == "False":
      return False
   raise Exception("Invalid input")

class Exp:

    #############################################################################################
    # EXPERIMENTAL PARAMETERS
    #############################################################################################

    # Volumes processing information
    n_heatup_vols = int(config["experiment"]["n_heatup_vols"]) # Number of volumes needed for heating up MRI scanner (heatup duration = TR * n_heatup_vols)
    #(ie, n_heatup_vols = 0 --> 1 heatup volume)

    n_baseline_vols = int(config["experiment"]["n_baseline_vols"]) # Number of baseline volumes after heatup to sample before beggining with the experimental task (baseline duration = TR * n_baseline_vols)

    HRF_peak_onset = int(config["experiment"]["HRF_peak_onset"]) # HRF peak onset threshold (seconds from trial onset). Default HRF_peak_onset = 5 for decoding procedure = 'average_hrf_peak_vols' or 'average_probs'.
                       # Set to 0 if do you want to decode volumes from trial onset in dynamic neurofeedback experiments.

    HRF_peak_offset = int(config["experiment"]["HRF_peak_offset"]) #float("inf")# HRF peak offset threshold (seconds from trial onset). Default HRF_peak_offset = 11 for decoding procedure = 'average_hrf_peak_vols' or 'average_probs'.
                         # Set to float("inf") if using an undetermined number of HRF peak volumes within each trial of dynamic neurofeedback experiments.

    TR = int(config["experiment"]["TR"]) # Repetition time. Volumes sampling rate (in seconds)


    # Volumes tracking
    first_vol_idx = int(config["experiment"]["first_vol_idx"]) # First volume index to track in EXP.raw_volumes_folder
    index_format = config["experiment"]["index_format"] # How volumes indexes are left-zero padded in fMRI cls.raw_volumes_folder folder? (ex., index_format = '04d' for IM-0001.dcm)
    #index format size only matters if the number of 0 at the end of the .dcm file is smaller than the specified by index_format (ie, 4)
    # Z-scoring procedure
    zscoring_procedure = config["experiment"]["zscoring_procedure"] # 'to_baseline' (each task volume will be z-scored relative to data from that run baseline in specific R.O.I. For example, volumes 51 will be z-scored to n_baseline_vols data)
                                            # 'to_timeseries' (each task volume will be z-scored relative to that run previous volumes in specific R.O.I.. For example, volume 51 will be z-scored using data from volume 0 to that volume)
                                            # 'to_model_session' (default) (each task volume will be z-scored relative to model construction session data in specific R.O.I. using its mean and standard deviation).

    # Decoding settings
    decoding_procedure = config["experiment"]["decoding_procedure"] # 'average_hrf_peak_vols' (average volumes within a trial HRF peak before decoding a single averaged volume to increase signal-to-noise ratio)
                                         # 'average_probs' (default) (average decoding probabilities of volumes within a trial HRF peak to increase feedbacks variability)
                                         # 'dynamic' (all volumes within a trial HRF peak, will be decoded independently and sent individually to experimental software as feedback)
    coadaptation_active = string_to_bool(config["experiment"]["coadaptation_active"])
    coadaptation_background_warmup = string_to_bool(config["experiment"]["coadaptation_background_warmup"])
    coadaptation_vol_acceptance_criteria = float(config["experiment"]["coadaptation_vol_acceptance_criteria"])
    classifier_type = config["experiment"]["classifier_type"]

    @classmethod # This method ensures participants data and directory routes can be inherited by all other classes
    def _new_participant(cls):

        """ Request new participant data (participant, session, run) each time main.py runs to set directories routes """

        def check_file(file):

            """ Check if a essential file exists or not. If not then cancel script execution. """

            if not os.path.exists(file):
                sys.exit(Fore.RED + f'[ERROR] File/Directory "{file}" does not exist. Check that you are pointing to a correct path. Breaking main.py execution.')

        #############################################################################################
        # DIRECTORIES & DATA
        #############################################################################################

        # First, ask for participant info to find/generate its corresponding folders
        print(Fore.YELLOW + 'Specify participants data before initialization:')
        cls.subject = input(Fore.YELLOW + '\nParticipant number: ')
        cls.session = input(Fore.YELLOW + 'Session number: ')
        cls.run = input(Fore.YELLOW + 'Run number: ')
        print('\n')

	#PATH TO COPY IN THE TERMINAL: cd /firmm/rt-fmri/META-BRAIN/pyDecNef-static/2.neurofeedback_training/1.realtime_fMRI_scripts'''

        # Package directory
        cls.moduledir = Path(__file__).absolute().parent.parent.parent

        # fMRI raw volumes output folder
        if string_to_bool(config["experiment"]["simulated_experiment"]):
            cls.raw_volumes_folder_path = os.path.join(cls.moduledir.parent,'2.MRI_simulator','outputs') # To use with fMRI simulator  
        else:
            cls.raw_volumes_folder_path =  config["files_and_dir"]["raw_volumes_folder_path"] # To use in a real experiment setting
        cls.raw_volumes_folder = Path(cls.raw_volumes_folder_path) 
        check_file(cls.raw_volumes_folder)
 
        # Required resources directory 
        # Contains pretrained model, region of interest mask, reference functional volume & z-scoring information
        cls.resources_dir = os.path.join(cls.moduledir.parent,'required_resources' ,f'sub-{cls.subject}')
        check_file(cls.resources_dir)

        # Pretrained model path
        cls.model_name = config["files_and_dir"]["model_name"] 
        cls.model_file =  os.path.join(cls.resources_dir , 'models',cls.model_name)
        check_file(cls.model_file)
        if cls.coadaptation_active:
            cls.coadaptation_model_name = cls.model_name + "_coadapted"
            cls.coadaptation_model_file = os.path.join(cls.resources_dir,'models',cls.coadaptation_model_name)
            if not os.path.exists(cls.coadaptation_model_file):
                model = load(cls.model_file)
                dump(model,cls.coadaptation_model_file)
            cls.model_file = cls.coadaptation_model_file


        # Region of interest mask path (as .nii to maximize load speed)
        cls.mask_name = config["files_and_dir"]["mask_name"]
        cls.mask_file = os.path.join(cls.resources_dir, 'masks',cls.mask_name)
        check_file(cls.mask_file)

        # Reference functional volume path (from model construction session. As .nii to maximize load speed)
        cls.ref_vol_name = config["files_and_dir"]["ref_vol_name"] 
        cls.ref_vol_file = os.path.join(cls.resources_dir, 'training_session_ref_image',cls.ref_vol_name)
        check_file(cls.ref_vol_file)
        
        # ROI reference data for z-scoring (if zscoring_procedure is 'to_model_session')
        if cls.zscoring_procedure == 'to_model_session':
            cls.zscoring_mean_file_name = config["files_and_dir"]["zscoring_mean_file_name"]
            cls.zscoring_std_file_name = config["files_and_dir"]["zscoring_std_file_name"]
            cls.zscoring_mean =  os.path.join(cls.resources_dir , 'training_zscoring_data',cls.zscoring_mean_file_name) # Numpy array containing mean of model construction session data
            cls.zscoring_std = os.path.join(cls.resources_dir, 'training_zscoring_data',cls.zscoring_std_file_name)  # Numpy array containing standard deviation of model construction session data
        
        if (cls.coadaptation_active or cls.coadaptation_background_warmup):
            cls.coadaptation_base_training_data_dir_name = config["files_and_dir"]["coadaptation_base_training_data_dir_name"] 
            cls.coadaptation_training_data_file_name = config["files_and_dir"]["coadaptation_training_data_file_name"] 
            cls.coadaptation_training_data_labels_file_name = config["files_and_dir"]["coadaptation_training_data_labels_file_name"] 
            cls.coadaptation_training_data_dir = os.path.join(cls.resources_dir,cls.coadaptation_base_training_data_dir_name)
            cls.coadaptation_training_data_file = os.path.join(cls.coadaptation_training_data_dir,cls.coadaptation_training_data_file_name)
            cls.coadaptation_training_data_labels_file = os.path.join(cls.coadaptation_training_data_dir,cls.coadaptation_training_data_labels_file_name)
            check_file(cls.coadaptation_training_data_dir)
            check_file(cls.coadaptation_training_data_file)
            check_file(cls.coadaptation_training_data_labels_file)


        # Create an outputs directory to store participant session log files and preprocessed volumes
        cls.outputs_dir = os.path.join(cls.moduledir,'outputs',f'sub-{cls.subject}_session-{cls.session}')
        Path(cls.outputs_dir).mkdir(parents=True, exist_ok=True)
        
        # main.py script run time
        script_run_time = time.strftime('%Y-%m-%d_%H-%M-%S') # Get main.py script run time, to create an unique run folder 
                                                             # and avoid folder replacement problems when wrongly typing runs number

        # Make a run directory inside outputs dir to store all participant log files and preprocessed volumes
        cls.run_dir = os.path.join(cls.outputs_dir, f'run-{cls.run}_{script_run_time}')
        Path(cls.run_dir).mkdir(parents=True, exist_ok=True)

        # Make a trials directory inside run directory to store all masked volumes and information classified by trial
        cls.trials_dir = os.path.join(cls.run_dir , 'trials')
        Path(cls.trials_dir).mkdir(parents=True, exist_ok=True)

        # Make a logs directory inside run directory to store run logs data
        cls.logs_dir = os.path.join(cls.run_dir , 'logs_dir')
        Path(cls.logs_dir).mkdir(parents=True, exist_ok=True)

        # Make a preprocessed volumes directory inside run directory to store all outputs corresponding to preprocessed volumes in that run
        cls.preprocessed_dir = os.path.join(cls.run_dir, 'preprocessed')
        Path(cls.preprocessed_dir).mkdir(parents=True, exist_ok=True)