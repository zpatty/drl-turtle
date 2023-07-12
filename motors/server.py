import socket
import time

def Main():
   
    host = '' #Server ip
    port = 4000

    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.setblocking(False)
    s.bind((host, port))

    print("Server Started")
    while True:
        try:
            data, addr = s.recvfrom(1024)
            data = data.decode('utf-8')
            print("Message from: " + str(addr))
            print("From connected user: " + data)
            data = data.upper()
            print("Sending: " + data)
            s.sendto(data.encode('utf-8'), addr)
        except:
            time.sleep(1)
            print("No message")
    c.close()

if __name__=='__main__':
    Main()

# import socket, time,os, random

# class Server():
#   def __init__(self,Adress=('',5000),MaxClient=1):
#       self.s = socket.socket()
#       self.s.bind(Adress)
#       self.s.listen(MaxClient)
#   def WaitForConnection(self):
#       self.Client, self.Adr=(self.s.accept())
#       print('Got a connection from: '+str(self.Client)+'.')


# s = Server()
# s.WaitForConnection()