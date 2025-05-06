############################################################################
# AUTHORS: Pedro Margolles & David Soto
# EMAIL: pmargolles@bcbl.eu, dsoto@bcbl.eu
# COPYRIGHT: Copyright (C) 2021-2022, pyDecNef
# URL: https://pedromargolles.github.io/pyDecNef/
# INSTITUTION: Basque Center on Cognition, Brain and Language (BCBL), Spain
# LICENCE: GNU General Public License v3.0
############################################################################
"""

This module implements a listener class capable of handling client requests in an 
asynchronous manner, ensuring that multiple operations can be processed without blocking
each other. The listener class is designed to support flexible request-action pairings,
allowing for customizable experimental setups.

"""
import os
import time
import threading
from colorama import Fore
from modules.classes.classes import Trial
from modules.config import shared_instances

#############################################################################################
# LISTENER CLASS
#############################################################################################

# Listener class match specific client requests (i.e., experimental software requests) with specific server actions
# This class can be customized to create new requests-actions pairings

class Listener:
    def __init__(self):
        pass

    def listen(self):
        listener_thread = threading.Thread(name = 'listener', 
                                           target = self._start_listen)
        listener_thread.start() # Keep listening to client requests in a new thread

    def _start_listen(self):
        while True:
            self.client_request = shared_instances.server.listen() # Start listening to potential client requests as a dictionary

            process_requests_thread = threading.Thread(name = 'process_requests',
                                                       target = self._process_client_requests)
            process_requests_thread.start() # Once received process client requests in an independent thread to avoid requests collision 
    
    #############################################################################################
    # CLIENT REQUESTS - SERVER ACTIONS PAIRINGS
    #############################################################################################


    def _process_client_requests(self):
        # If client request signals a trial onset
        if self.client_request['request_type'] == 'trial_onset':
            shared_instances.new_trial = Trial()
            shared_instances.new_trial.trial_idx = self.client_request['trial_idx']
            shared_instances.new_trial.ground_truth = self.client_request['ground_truth']
            shared_instances.new_trial.stimuli = self.client_request['stimuli']
            shared_instances.new_trial.trial_onset = time.time() # Set an onset time when we receive trial onset signal
                                                                 # There might be some delay (miliseconds) with respect to when the onset actually occurred in the experimental software computer 
                                                                 # However, by doing this we avoid clock synchronization problems between experimental computer and server computer

            shared_instances.server.send('ok') # Send an OK to the client when request is processed

        # If client request from experimental software signals to start with decoding of HRF peak volumes in this trial
        elif self.client_request['request_type'] == 'feedback_start':
            feedback_thread = threading.Thread(
                                               name = 'decoding_trial',
                                               target = shared_instances.new_trial._decode
                                              ) # Call new_trial.decode function and pass server object as an argument, to send back 
                                                # resulting information to experimental software when decoding is finished
                                                # Decode trial volumes in a new thread for not interrupting volumes' filewatcher
            
            feedback_thread.start() # Start a new thread
            feedback_thread.join() # Wait until decoding is finished, then continue

            if shared_instances.new_trial.decoding_done == True:
                shared_instances.server.send('ok') # End of feedback procedure. Continue with the next experimental phase

        # For dynamic neurofeedback: If client request from experimental software signals to end with decoding of HRF peak volumes in this trial 
        # before continuing with the next trial (i.e., for example, with an endless trial dynamic neurofeedback approach)
        elif self.client_request['request_type'] == 'feedback_end':
            shared_instances.new_trial.HRF_peak_end = True

        # If client request is a request to finish this experimental run
        elif self.client_request['request_type'] == 'end_run':
            print(Fore.GREEN + '[FINISHING] Experimental run is over.')
            shared_instances.server.send('ok') # End experimental software script and exit.
            os._exit(1) # End server execution

        else:
            print(Fore.RED + f'[ERROR] Request {self.client_request} not recognized.')