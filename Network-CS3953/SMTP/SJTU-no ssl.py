import socket
from private import from_mail, to_mail
#For security, I have to hide the from_mail and to_mail in private.py

#sjtu mail for school student, but with no security
mailserver = 'mail.sjtu.edu.cn'
mailport = 25
 
clientSocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
clientSocket.connect((mailserver, mailport))

#step 0: Connection
recv = clientSocket.recv(1024).decode()
print(recv) 
 
#step 1: HELO
heloCommand = "HELO xunying\r\n"
clientSocket.send(heloCommand.encode()) 
recv1 = clientSocket.recv(1024).decode()
print("HELO Response:", recv1) 

#step 2: MAIL FROM
mail_from_command = f"MAIL FROM:<{from_mail}>\r\n"
clientSocket.send(mail_from_command.encode())
recv = clientSocket.recv(1024).decode()
print("MAIL FROM Response:", recv)

#step 3: RCPT TO
rcpt_to_command = f"RCPT TO:<{to_mail}>\r\n"
clientSocket.send(rcpt_to_command.encode())
recv = clientSocket.recv(1024).decode()
print("RCPT TO Response:", recv)

#step 4: DATA
data_command = "DATA\r\n"
clientSocket.send(data_command.encode())
recv = clientSocket.recv(1024).decode()
print("DATA Response:", recv)

#step 5: Send Message
message = f"From: yyu@apex.sjtu.cn\nTo: {to_mail}\nSubject: 放假通知\n\n 后天放假，望周知"
clientSocket.send((message + "\r\n.\r\n").encode())
recv = clientSocket.recv(1024).decode()
print("Message Sending Response:", recv)

#step 6: QUIT
quit_command = "QUIT\r\n"
clientSocket.send(quit_command.encode())
recv = clientSocket.recv(1024).decode()
print("QUIT Response:", recv) 


