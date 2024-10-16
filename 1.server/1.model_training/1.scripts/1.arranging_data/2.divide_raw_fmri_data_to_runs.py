import os
from datetime import datetime
import shutil
recorded_data_main_folder = "/firmm/" # the path where the MRI scanner dump the scans
test = True
if test:
    recorded_data_main_folder = os.path.join(os.path.abspath(os.path.join(os.getcwd(), os.pardir,os.pardir)),"2.data/recorded_data")
print("The main folder for recorded data path is:",recorded_data_main_folder)
print("the main folder :",os.path.abspath(os.path.join(os.path.abspath(__file__),os.pardir,os.pardir,os.pardir, "2.data/recorded_data")) ) 
any_folders =  [os.path.isdir(i) for i in os.listdir(recorded_data_main_folder)] 
if any(any_folders):
    print("there is a folder")
    recorded_data_directories_list = [i for i in os.listdir(recorded_data_main_folder) if not i.startswith(".")] # filter to remove the hidden files from the list
    if test:
        recorded_directories_list_today = recorded_data_directories_list
    recorded_directories_list_today = [i for i in recorded_data_directories_list if i.startswith(datetime.today().strftime("%Y%m%d"))] # filter the directories by the ones created today  
    recorded_data_directories_list_paths = [os.path.join(recorded_data_main_folder,i) for i in recorded_directories_list_today]
    recorded_data_directories_list_paths.sort(key=lambda x: os.path.getmtime(x)) # sort by the most recently created files/folders
    try:
        raw_training_data_path = recorded_data_directories_list_paths[-1]
    except IndexError:
        print("No new folders created today, please make sure that you have new folder for the raw data")
        print(f"using the main folder for data with path:{recorded_data_main_folder}")
        raw_training_data_path =recorded_data_main_folder
else:
    raw_training_data_path = recorded_data_main_folder
print(raw_training_data_path)
unique_runs = list(set([i.split("_")[1] for i in os.listdir(raw_training_data_path)]))
unique_runs.sort()

print("The chosen folder to extract the runs from:",raw_training_data_path)
print("The number of runs found:",len(unique_runs),"| The prefix of the found runs:",unique_runs )
decoder_training_folder_path = os.path.abspath(os.path.join(os.path.abspath(__file__),os.pardir,os.pardir,os.pardir,"2.data"))
print("decoder training data folder path:",decoder_training_folder_path)
for i,pattern in zip(range(1,len(unique_runs)+1),unique_runs) :
    if not os.path.exists(os.path.join(decoder_training_folder_path,"raw/func/",f"run_{i}")):
        os.makedirs(os.path.join(decoder_training_folder_path,"raw/func/",f"run_{i}"))
        folders_exits = False
        print(f"Moving the runs files to the decoder training directory for: run_{i}")
        for file in os.listdir(raw_training_data_path):
            if pattern in file.split("_")[1]:
                shutil.copy(src=os.path.join(raw_training_data_path,file),dst= os.path.join(decoder_training_folder_path,"raw/func/",f"run_{i}",file))
                #shutil.move()
    else:
        print(f"The run_{i} folder already exists, no action taken")
        folders_exits= True
if not folders_exits:
    print("The runs files moved successfully to the raw/func folder path in the decoder training data folder.")
 
