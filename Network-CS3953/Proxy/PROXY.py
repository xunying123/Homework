import os
from urllib.parse import urlparse
import socket
import threading
import select

# 对于CONNECT与GET以外的请求的处理
def Forward(tcpCliSock, message):
    # 提取目标主机和路径
    target_host = message.split()[1]
    headers = ""
    body = b""

    # 接收完整的请求头部
    while True:
        line = tcpCliSock.recv(1024).decode()
        headers += line
        if "\r\n\r\n" in headers:  # 请求头与请求体之间的分界
            break

    # 构造发送到目标服务器的完整请求
    request = message + headers + "\r\n" + body.decode()

    # 与目标服务器建立连接
    target_host_name = target_host.split(":")[0]
    target_port = 80 if len(target_host.split(":")) == 1 else int(target_host.split(":")[1])

    try:
        server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_sock.connect((target_host_name, target_port))

        # 将请求转发给目标服务器
        server_sock.sendall(request.encode())

        # 接收目标服务器的响应并转发给客户端
        response = b""
        while True:
            buffer = server_sock.recv(4096)
            if buffer:
                response += buffer
            else:
                break

        # 一次性发送目标服务器的完整响应给客户端
        tcpCliSock.sendall(response)
        server_sock.close()

    except Exception as e:
        print(f"Error while forwarding request: {e}")
        tcpCliSock.send(b"HTTP/1.0 502 Bad Gateway\r\n\r\n")

    finally:
        tcpCliSock.close()

# 对于GET请求的处理
def GET(tcpCliSock, message):
    try:
        # 提取目标主机和路径
        url = message.split()[1]  
        parsed_url = urlparse(url)  
        hostn = parsed_url.netloc  
        path = parsed_url.path  

        if path == '/':
            path = '/index.html'  # 默认文件路径

        print('Host:', hostn)
        print('Path:', path)

        # 缓存路径修改为 ./net/data/host/path 形式
        cache_dir = os.path.join("./net/data", hostn)
        cache_file = os.path.join(cache_dir, path.lstrip("/"))

        # 如果路径中的目录不存在，自动创建
        if not os.path.exists(os.path.dirname(cache_file)):
            os.makedirs(os.path.dirname(cache_file))

    except IndexError:
        tcpCliSock.close()
        return

    fileExist = "false"
    print('Cache file path:', cache_file)

    try:
        f = open(cache_file, "rb")  # 从缓存中读取文件
        outputdata = f.readlines()
        fileExist = "true"
        
        tcpCliSock.send(b"HTTP/1.0 200 OK\r\n")
        tcpCliSock.send(b"Content-Type:text/html\r\n\r\n")

        for data in outputdata:
            tcpCliSock.send(data)  # 从缓存中读取文件并发送

        print('Read from cache')

    except IOError:
        if fileExist == "false":
            c = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            try:
                c.connect((hostn, 80))
                
                # 创建向目标服务器发送的请求
                request = f"GET {path} HTTP/1.0\r\nHost: {hostn}\r\n\r\n"
                c.sendall(request.encode())

                response_header = b""

                while True:
                    chunck = c.recv(4096)
                    response_header += chunck
                    if b"\r\n\r\n" in response_header:
                        break
                    # 读取来自目标服务器的数据并发送给客户端

                header, body = response_header.split(b"\r\n\r\n", 1)
                tcpCliSock.sendall(header + b"\r\n\r\n")  # 将响应头发送给客户端
                
                tcpCliSock.sendall(body)  # 将响应体发送给客户端
                with open(cache_file, "wb") as tmpFile:  # 将响应体存储到缓存文件中
                    tmpFile.write(body)

                while True:
                    buffer = c.recv(4096)
                    if len(buffer) > 0:
                        tcpCliSock.sendall(buffer)  # 发送数据到客户端
                        with open(cache_file, "ab") as tmpFile:  # 存储到缓存文件中
                            tmpFile.write(buffer)
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

# 对于CONNECT请求的处理
def CONNECT(tcpCliSock, message):
    # 解析 CONNECT 请求
    target_host_port = message.split()[1]
    target_host, target_port = target_host_port.split(":")
    target_port = int(target_port)

    try:
        # 建立与目标服务器的连接
        target_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        target_sock.connect((target_host, target_port))

        # 向客户端发送200连接成功的响应
        tcpCliSock.send(b"HTTP/1.1 200 Connection Established\r\n\r\n")

        # 使用select进行双向数据转发
        sockets = [tcpCliSock, target_sock]
        while True:
            readable, _, _ = select.select(sockets, [], [])
            for sock in readable:
                data = sock.recv(8192)
                if not data:  # 如果连接关闭，退出循环
                    break
                if sock is tcpCliSock:
                    print("length of data from client:", len(data))
                    target_sock.send(data)
                else:
                    print("length of data from server:", len(data))
                    tcpCliSock.send(data)
    except Exception as e:
        print(f"Error handling CONNECT request: {e}")
        tcpCliSock.send(b"HTTP/1.1 502 Bad Gateway\r\n\r\n")
    finally:
        tcpCliSock.close()

def handle_client(tcpCliSock):
    try:
        message = tcpCliSock.recv(1024).decode()  # 接收客户端消息
        if not message:
            tcpCliSock.close()
            return

        print(message)

        if message.startswith("GET"):
            GET(tcpCliSock, message)
        elif message.startswith("CONNECT"):
            CONNECT(tcpCliSock, message)
        else:
            Forward(tcpCliSock, message)
    except Exception as e:
        print(f"Error handling client request: {e}")
        tcpCliSock.close()

# Create a server socket, bind it to a port and start listening
tcpSerSock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
tcpSerSock.bind(('', 8080))  # 绑定到本地端口 8080
tcpSerSock.listen(10)

while True:
    # Start receiving data from the client
    print('Ready to serve...')
    tcpCliSock, addr = tcpSerSock.accept()
    print('Received a connection from:', addr)

    # 为每个连接启动一个新的线程
    client_thread = threading.Thread(target=handle_client, args=(tcpCliSock,))
    client_thread.start()
    