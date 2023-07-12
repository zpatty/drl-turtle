# import socket

# class Client():
#    def __init__(self,Adress=('',5000)):
#       self.s = socket.socket()
#       self.s.connect(Adress)

# c = Client()

import socket

def Main():

    host='192.168.8.150' #client ip
    port = 4005
    
    server = ('192.168.8.244', 4000)
    
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.bind((host,port))
    
    message = input("-> ")
    while 1:
        s.sendto(message.encode('utf-8'), server)
        data, addr = s.recvfrom(1024)
        data = data.decode('utf-8')
        print("Received from server: " + data)
        message = input("-> ")
    s.close()

if __name__=='__main__':
    Main()