import os
from urllib.parse import urlparse
import socket
import time

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

def GET(tcpCliSock, message):
    try:
        url = message.split()[1]  # 提取URL
        parsed_url = urlparse(url)  # 使用urlparse解析URL
        hostn = parsed_url.netloc  # 提取主机名
        path = parsed_url.path  # 提取路径

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
        # Check whether the file exists in the cache
        f = open(cache_file, "rb")  # 从缓存中读取文件
        outputdata = f.readlines()
        fileExist = "true"
        
        # ProxyServer finds a cache hit and generates a response message
        tcpCliSock.send(b"HTTP/1.0 200 OK\r\n")
        tcpCliSock.send(b"Content-Type:text/html\r\n\r\n")

        for data in outputdata:
            tcpCliSock.send(data)  # 从缓存中读取文件并发送

        print('Read from cache')

    except IOError:
        if fileExist == "false":
            # Create a socket on the proxy server
            c = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            try:
                # Connect to the target server
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

def CONNECT(tcpCliSock, message):
    # 解析主机和端口
    target_host_port = message.split()[1]
    target_host, target_port = target_host_port.split(':')
    print(f"Connecting to {target_host}:{target_port}")

    try:
        # 与目标主机建立连接
        target_port = int(target_port)
        server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_sock.connect((target_host, target_port))

        # 向客户端返回连接成功的消息
        tcpCliSock.send(b"HTTP/1.1 200 Connection Established\r\n\r\n")

        # 将客户端和目标服务器的套接字设置为非阻塞模式
        tcpCliSock.setblocking(False)
        server_sock.setblocking(False)

        # 设定超时时间（单位：秒）
        timeout = 3
        last_activity_time = time.time()

        # 在客户端与目标服务器之间转发数据
        while True:
            current_time = time.time()
            if current_time - last_activity_time > timeout:
                print("Connection timed out due to inactivity.")
                break  # 超时，退出循环

            try:
                # 从客户端接收数据并发送到服务器
                client_data = tcpCliSock.recv(4096)
                if client_data:
                    print(f"Received {len(client_data)} bytes from client")
                    server_sock.sendall(client_data)
                    last_activity_time = current_time  # 更新计时器
                elif client_data == b'':  # 客户端关闭连接
                    print("Client closed connection.")
                    break
            except BlockingIOError:
                pass
            except ConnectionAbortedError:
                print("Client connection aborted.")
                break
            except ConnectionResetError:
                print("Client connection reset.")
                break

            try:
                # 从服务器接收数据并发送到客户端
                server_data = server_sock.recv(4096)
                if server_data:
                    print(f"Received {len(server_data)} bytes from server")
                    tcpCliSock.sendall(server_data)
                    last_activity_time = current_time  # 更新计时器
                elif server_data == b'':  # 服务器关闭连接
                    print("Server closed connection.")
                    break
            except BlockingIOError:
                pass
            except ConnectionAbortedError:
                print("Server connection aborted.")
                break
            except ConnectionResetError:
                print("Server connection reset.")
                break

    except Exception as e:
        print(f"Error in CONNECT request handling: {e}")
        tcpCliSock.send(b"HTTP/1.1 502 Bad Gateway\r\n\r\n")

    finally:
        # 关闭连接
        print("Closing connection")
        tcpCliSock.close()
        server_sock.close()

# Create a server socket, bind it to a port and start listening
tcpSerSock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
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

    if message.startswith("GET"):
        GET(tcpCliSock, message)
        continue
    elif message.startswith("CONNECT"):
        CONNECT(tcpCliSock, message)
        continue
    else:
        Forward(tcpCliSock, message)
        continue

    # Parse the client's request URL
    