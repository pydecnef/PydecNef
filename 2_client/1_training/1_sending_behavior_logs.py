"""
This script is designed to send the behavior log files from the client side to the server side.

The script will need to connect to the server side through the '1_server/1_model_training/1_scripts/1_arranging_data/1_receive_behavior_logs.py'
 script before sending the files and placing a copy at 'files_sent' directory that will be created at the same path of the script.

Note: The process of moving the log files in this package is still in the experimental phase and can encounter some issues, alternatively these files can be moved manually by the user.
"""
import pandas as pd
import time
import os
import shutil
from server_client_connexion import Connection

csv_files_list = [ i for i in  os.listdir(".") if i.endswith(".csv") ]
print("The existing csv files (created by opensesame) : ",csv_files_list)
os.makedirs("files_sent",exist_ok=True)
for csv_file in csv_files_list:
    if os.path.exists(os.path.join("files_sent",csv_file)):
        os.remove(os.path.join("files_sent",csv_file))
    shutil.copy(csv_file,os.path.join("files_sent",csv_file))
file_list =  os.listdir("files_sent")  
print("files to be sent:", file_list)
for file in file_list:
    client = Connection()
    client.start_client()
    path = os.path.join("files_sent",file)
    df = pd.read_csv(path)
    raw_data = df.to_numpy()
    raw_columns = df.columns.to_list()
    client.send(raw_columns)
    time.sleep(2) ### sleep time needs to be more that the intersegments timeout time in big_data_connectionW
    client.send(raw_data)
    print(f"file {file} sent")
    time.sleep(3)

