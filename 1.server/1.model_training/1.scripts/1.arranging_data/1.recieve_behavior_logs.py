############################################################################
# AUTHORS: Najemeddine Abdennour
# EMAIL: nabdennour@bcbl.eu
# COPYRIGHT: Copyright (C) 2024-2025, pyDecNef
# URL: https://github.com/najemabdennour/pyDecNef
# INSTITUTION: Basque Center on Cognition, Brain and Language (BCBL), Spain
# LICENCE: GNU General Public License v3.0
############################################################################
from server_client_connexion import Connection
import os
import signal
import time
import pandas as pd
def timeout(signum, frame):
    """Raises a TimeoutError to break out of the loop when the server operation times out."""
    raise TimeoutError("The server operation timed out.")

decoder_training_folder_path = os.path.abspath(os.path.join(
    os.path.abspath(__file__), os.pardir, os.pardir, os.pardir
))

i = 1

while True:
    try:
        # Initialize and start the server connection
        server = Connection()
        server.start_server()

        # Set up a timeout for the server operation
        signal.signal(signal.SIGALRM, timeout)
        signal.alarm(5)  # Wait for 5 seconds before raising TimeoutError

        # Listen to the server for incoming data
        raw_columns = server.listen()

        time.sleep(2)

        # Retrieve and process the raw data from the server
        raw_data = server.listen()

        # Create DataFrame from raw data using columns from previous listen call
        data = pd.DataFrame(raw_data, columns=raw_columns)

        # Ensure the output directory exists
        output_folder = os.path.join(decoder_training_folder_path, "2.data", "raw", "behav")
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        # Save processed data to CSV file
        file_name = f"behav_run_{i}.csv"
        file_path = os.path.join(output_folder, file_name)
        data.to_csv(file_path, index=False)

        print(f"Run {i} file saved in: {file_path}")
        print("Data received successfully.")
        i += 1
    except TimeoutError:
        break
        print("Finished receiving files.")

