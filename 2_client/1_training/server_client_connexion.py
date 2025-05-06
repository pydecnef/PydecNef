############################################################################
# AUTHORS: Pedro Margolles & David Soto & Najemeddine Abdennour
# EMAIL: pmargolles@bcbl.eu, dsoto@bcbl.eu, nabdennour@bcbl.eu
# COPYRIGHT: Copyright (C) 2024-2025, pyDecNef
# URL: https://github.com/najemabdennour/pyDecNef
# INSTITUTION: Basque Center on Cognition, Brain and Language (BCBL), Spain
# LICENCE: GNU General Public License v3.0
############################################################################
"""
This script is used to establish a server client connexion between two terminals (two computers) 
"""
import pickle
import signal
import socket
import sys
import code
from colorama import init, Fore

init(autoreset=True)

def timeout(signum, frame):
    raise TimeoutError

#IP = socket.gethostname() # To use with localhost address
IP = '192.168.242.119' # Local server computer IP
PORT = 12345 # Server PORT
FORMAT = 'utf-8' # Data format for bytes encoding and decoding
N_BYTES = 2000 # Maximun number of bytes to expect from connection
TIMEOUT = 500 # Maximum number of seconds of innactivity

class Connection:

    """ A class to establish connection between two terminals through sockets"""
    
    def __init__(self):

        self.ip = IP #Set the IP address for connection.
        self.port = PORT #Set the port number for connection.
        self.format = FORMAT #Set the data format (e.g., 'utf-8').
        self.n_bytes = N_BYTES #Set the maximum number of bytes to receive.
        socket.setdefaulttimeout(TIMEOUT) # Set a timeout (when client/server do not send/receive any 
                                      # data for this timeframe, then close connection)

    def start_server(self):

        """ A class method to initialize server connection side """

        self.server = socket.socket(socket.AF_INET, # Start TCP/IP socket
                                    socket.SOCK_STREAM)
        
        self.server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1) # To be able to create again a new server on this
                                                                          # IP and PORT after closing actual server terminal

        self.type = 'server' # Connection type

        try: # Try to bind server to IP and PORT
            print(Fore.YELLOW + '[STARTING] Server is starting...')
            self.server.bind((self.IP, # Set IP address to be the server address
                              self.PORT)) # Set a port to connect with clients (usually above 1000))
        except socket.error:
            print(Fore.RED + f'[ERROR] Server connection to {self.IP} on port {self.PORT} has failed.')
            print(Fore.YELLOW + '[CLOSING] Closing server...')
            sys.exit() # To close server script in case connection to specified IP and PORT fails

        self.server.listen() # Listen for new connections
        print(Fore.YELLOW + f'[WAITING] Waiting for connection with the experimental software on {self.IP} and port {self.PORT}')
        while True: # Loop awaiting for a connection
            self.client, self.client_address = self.server.accept() # When connection to a client occurs assigns client data 
                                                                    # to client_address (IP:PORT) and stablish a client_socket
            if self.client:
                break
        print(Fore.GREEN + f'[CONNECTED] Connection with {self.client_address} has been established.')


    def start_client(self):

        """A class method to initialize client connection side """

        self.client = socket.socket(socket.AF_INET, # Start TCP/IP socket
                                   socket.SOCK_STREAM)

        self.type = 'client' # Connection type

        try:
            print(Fore.YELLOW + '[STARTING] Client is starting...')
            self.client.connect((self.IP, # To connect to the server side
                                 self.PORT)) # Set a port to connect to (usually above 1000))
            print(Fore.GREEN + f'[CONNECTED] You are now connected to {self.IP} on port {self.PORT}.')
        except socket.error:
            print(Fore.RED + f'[ERROR] Connection to server {self.IP} on port {self.PORT} has failed.')
            print(Fore.YELLOW + '[CLOSING] Closing client...')
            sys.exit() # To close the client script in case connection to specified IP and PORT fails

    def listen(self):

        """A class method to listen for incoming packets"""

        print(Fore.YELLOW + '[WAITING] Waiting for messages...')
        data = []
        signal.signal(signal.SIGALRM, timeout)
        signal.alarm(1) ### 1 seconds time out for the data segment (intersegments timeout time)
        while True:
            try:
                packet = self.client.recv(self.N_BYTES) # Receive the message
            except TimeoutError:
                print("finished listening..")
                break
            if not packet:
                print("end of packet segments")
            data.append(packet)
        print("finished receiving all data segments.")
        message = pickle.loads(b"".join(data)) # Unpickle the message
        print(Fore.GREEN + f'[RECEIVED] Message: "{message}" received.')
        return message

    def send(self, message):

        """A class method to send packets"""

        print(Fore.YELLOW + f'[SENDING] Sending "{message}"...')
        message = pickle.dumps(message) # Pickle the message
        self.client.send(message) # Send message

if __name__ == '__main__':
    # Initialize connection settings
    conn = Connection()
    conn.start_server()# Start a server connection
    code.interact(local=locals()) # Continue in interactive mode
