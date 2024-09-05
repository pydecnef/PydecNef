import os
from datetime import datetime
import shutil
test = True
firmm_main_folder = "/firmm/" 
if test:
    firmm_main_folder = os.path.join(os.path.abspath(os.path.join(os.getcwd(), os.pardir)),"firmm")
    print(firmm_main_folder)
    if mac:
firmm_directories_list = [i for i in os.listdir(firmm_main_folder) if not i.startswith(".")] # filter to remove the hidden files from the list
if test:
    firmm_directories_list_today = firmm_directories_list
else:
    firmm_directories_list_today = [i for i in firmm_directories_list if i.startswith(datetime.today().strftime("%Y%m%d"))] # filter the directories by the ones created today  
firmm_directories_list_paths = [os.path.join(firmm_main_folder,i) for i in firmm_directories_list_today]
firmm_directories_list_paths.sort(key=lambda x: os.path.getmtime(x)) # sort by the most recently created files/folders
try:
    raw_training_data_path = firmm_directories_list_paths[-1]
except IndexError:
    print("No new folders created today, please make sure that you have new folder for the raw data")


print(os.listdir(raw_training_data_path))
unique_runs = list(set([i.split("_")[1] for i in os.listdir(raw_training_data_path)]))
unique_runs.sort()

print("\nThe list of files/folders in the main firmm folder:\n",firmm_directories_list)
print("The chosen folder to extract the runs from:",firmm_directories_list_paths[-1])
print("The number of runs found:",len(unique_runs),"| The prefix of the found runs:",unique_runs )
decoder_training_folder_path = os.path.abspath(os.path.join(os.path.abspath(__file__),os.pardir,os.pardir))
print("decoder_training_folder_path:",decoder_training_folder_path)
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
    print("The runs files moved successfully.")
 
