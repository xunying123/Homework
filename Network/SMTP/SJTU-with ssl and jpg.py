import socket
import ssl
import base64
import os
from private import username, password, from_mail, to_mail
# For security, I have to hide the from_mail and to_mail in private.py

# sjtu mail for school student, and with security
mailserver = 'mail.sjtu.edu.cn'
mailport = 465

context = ssl.create_default_context()
clientSocket = context.wrap_socket(socket.socket(socket.AF_INET, socket.SOCK_STREAM), server_hostname=mailserver)
clientSocket.connect((mailserver, mailport))

#step 0: Connection
recv = clientSocket.recv(1024).decode()
print(recv)  

#step 1: HELO
heloCommand = "HELO xunying\r\n"
clientSocket.send(heloCommand.encode()) 
recv1 = clientSocket.recv(1024).decode()
print("HELO Response:", recv1) 

#step 2: AUTH LOGIN
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

#step 3: MAIL FROM
mail_from_command = f"MAIL FROM:<{from_mail}>\r\n"
clientSocket.send(mail_from_command.encode())
recv = clientSocket.recv(1024).decode()
print("MAIL FROM Response:", recv)

#step 4: RCPT TO
rcpt_to_command = f"RCPT TO:<{to_mail}>\r\n"
clientSocket.send(rcpt_to_command.encode())
recv = clientSocket.recv(1024).decode()
print("RCPT TO Response:", recv)

#step 5: DATA
data_command = "DATA\r\n"
clientSocket.send(data_command.encode())
recv = clientSocket.recv(1024).decode()
print("DATA Response:", recv)

#step 6: Send Message
boundary = "this_is_a_boundary"

#To create a message with attachment, we need to use MIME
message = f"""\
From: test
To: {to_mail}
Subject: Just for test
MIME-Version: 1.0
Content-Type: multipart/mixed; boundary="{boundary}"

--{boundary}
Content-Type: text/plain; charset="utf-8"
Content-Transfer-Encoding: 7bit

This is a test email with an attachment sent using socket programming.

--{boundary}
"""

#This is the jpg file path
file_path="D:\\111.jpg"
filename = os.path.basename(file_path)

with open(file_path, 'rb') as attachment:
    file_data = attachment.read()
    encoded_file = base64.b64encode(file_data).decode('utf-8')

message += f"""\
Content-Type: image/jpeg; name="{filename}"
Content-Transfer-Encoding: base64
Content-Disposition: attachment; filename="{filename}"

{encoded_file}
--{boundary}--
"""


clientSocket.send((message + "\r\n.\r\n").encode())
recv = clientSocket.recv(1024).decode()
print("Message Sending Response:", recv)

#step 7: QUIT
quit_command = "QUIT\r\n"
clientSocket.send(quit_command.encode())
recv = clientSocket.recv(1024).decode()
print("QUIT Response:", recv) 


