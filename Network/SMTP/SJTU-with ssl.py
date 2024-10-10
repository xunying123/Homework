import socket
import ssl
import base64
from private import username, password, from_mail, to_mail
msg = "\r\n I love computer networks!" 
endmsg = "\r\n.\r\n"

mailserver = 'mail.sjtu.edu.cn'
mailport = 465
 
context = ssl.create_default_context()
clientSocket = context.wrap_socket(socket.socket(socket.AF_INET, socket.SOCK_STREAM), server_hostname=mailserver)
clientSocket.connect((mailserver, mailport))

recv = clientSocket.recv(1024).decode()
print(recv) 
if recv[:3] != '220': 
    print('220 reply not received from server.') 
 
heloCommand = "HELO xunying\r\n"
clientSocket.send(heloCommand.encode()) 
recv1 = clientSocket.recv(1024).decode()
print(recv1) 
if recv1[:3] != '250': 
    print('250 reply not received from server.') 

auth_command = 'AUTH LOGIN\r\n'
clientSocket.send(auth_command.encode())
recv = clientSocket.recv(1024).decode()
print("AUTH Response:", recv)

clientSocket.send(base64.b64encode(username.encode()) + b'\r\n')
recv = clientSocket.recv(1024).decode()
print("Username Response:", recv)

clientSocket.send(base64.b64encode(password.encode()) + b'\r\n')
recv = clientSocket.recv(1024).decode()
print("Password Response:", recv)

mail_from_command = f"MAIL FROM:<{from_mail}>\r\n"
clientSocket.send(mail_from_command.encode())
recv = clientSocket.recv(1024).decode()
print("MAIL FROM Response:", recv)

rcpt_to_command = f"RCPT TO:<{to_mail}>\r\n"
clientSocket.send(rcpt_to_command.encode())
recv = clientSocket.recv(1024).decode()
print("RCPT TO Response:", recv)

data_command = "DATA\r\n"
clientSocket.send(data_command.encode())
recv = clientSocket.recv(1024).decode()
print("DATA Response:", recv)

message = "From: XZY_is_my_son\nTo: xun_ying_is_my_dad\nSubject: only test\n\n this is a test"
clientSocket.send((message + "\r\n.\r\n").encode())
recv = clientSocket.recv(1024).decode()
print("Message Sending Response:", recv)

quit_command = "QUIT\r\n"
clientSocket.send(quit_command.encode())
recv = clientSocket.recv(1024).decode()
print("QUIT Response:", recv) 


