import os
from urllib.parse import urlparse
from socket import *

# Create a server socket, bind it to a port and start listening
tcpSerSock = socket(AF_INET, SOCK_STREAM)
tcpSerSock.bind(('', 8080))  # 绑定到本地端口 8888
tcpSerSock.listen(10)

while True:
    # Start receiving data from the client
    print('Ready to serve...')
    tcpCliSock, addr = tcpSerSock.accept()
    print('Received a connection from:', addr)

    # Get the client request message
    message = tcpCliSock.recv(1024).decode()  # 接收客户端消息
    if not message:
        tcpCliSock.close()
        continue

    print(message)
    # Parse the client's request URL
    try:
        url = message.split()[1]  # 提取URL
        parsed_url = urlparse(url)  # 使用urlparse解析URL
        hostn = parsed_url.netloc  # 提取主机名
        path = parsed_url.path  # 提取路径

        if path == '/':
            path = '/index.html'  # 默认文件路径

        print('Host:', hostn)
        print('Path:', path)

    except IndexError:
        tcpCliSock.close()
        continue
    fileExist = "FAlse"
    if fileExist == "false":
        # Create a socket on the proxy server
        c = socket(AF_INET, SOCK_STREAM)
        try:
            # Connect to the target server
            c.connect((hostn, 80))
                
                # 创建向目标服务器发送的请求
            request = f"GET {path} HTTP/1.0\r\nHost: {hostn}\r\n\r\n"
            c.sendall(request.encode())

            while True:
                buffer = c.recv(4096)
                if len(buffer) > 0:
                    tcpCliSock.sendall(buffer)  # 发送数据到客户端
                else:
                    break  # 当读取不到更多数据时，跳出循环

                
            tcpCliSock.sendall(buffer)


        except Exception as e:
            print("Error:", e)
            tcpCliSock.send(b"HTTP/1.0 404 Not Found\r\n")
            tcpCliSock.send(b"Content-Type:text/html\r\n\r\n")
            tcpCliSock.send(b"<html><body><h1>404 Not Found</h1></body></html>\r\n")

        c.close()

    tcpCliSock.close()
