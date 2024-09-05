from server_client_connexion import Connection
import time
import pandas as pd
import os
import signal
def timeout(signum,frame):
    raise TimeoutError

decoder_training_folder_path = os.path.abspath(os.path.join(os.path.abspath(__file__),os.pardir,os.pardir,os.pardir))
i = 1
while True:
    try:
        server = Connection()
        server.start_server()
        signal.signal(signal.SIGALRM,timeout)
        signal.alarm(5)
        raw_columns = server.listen()
        time.sleep(2)
        raw_data = server.listen()
        data = pd.DataFrame(raw_data,columns= raw_columns)
        if not os.path.exists(os.path.join(decoder_training_folder_path,"2.data/raw/behav")):
            os.makedirs(os.path.join(decoder_training_folder_path,"2.data/raw/behav"))
        data.to_csv(os.path.join(decoder_training_folder_path,"2.data/raw/behav",f"behav_run_{i}.csv"),index=False)
        print(f"run_{i} file saved in :",os.path.join(decoder_training_folder_path,"2.data/raw/behav",f"behav_run_{i}.csv"))
        print("data received successfully.")
        i+=1
    except TimeoutError:
        break
        print("finished receiving files")

