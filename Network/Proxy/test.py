import socket 
PROXY_HOST = '127.0.0.1'
PROXY_PORT = 8080

proxy_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
proxy_socket.bind((PROXY_HOST, PROXY_PORT))
proxy_socket.listen(10)
print(f'Proxy server is listening on {PROXY_HOST}:{PROXY_PORT}')

while 1: 

    client_socket, client_address = proxy_socket.accept()
    print(f'Accepted connection from {client_address}')
    request_data = client_socket.recv(8192)
    message = request_data.decode('utf-8')
    print(message) 
    fileExist = "false" 
    filename = message.split()[1].partition("/")[2]

    filetouse = "/" + filename
    print("Final path to use:", filetouse)

    try: 
        # Check wether the file exist in the cache 
        f = open(filetouse[1:], "rb") 
        outputdata = f.readlines() 
        fileExist = "true" 
        # ProxyServer finds a cache hit and generates a response message 
        proxy_socket.send(b"HTTP/1.0 200 OK\r\n") 
        proxy_socket.send(b"Content-Type:text/html\r\n") 

        for data in outputdata:
            proxy_socket.send(data)  # 从缓存中读取文件并发送给客户端
# 关闭 socket 连接
        proxy_socket.close()
        print('Read from cache') 
# Error handling for file not found in cache 
    except IOError: 
        if fileExist == "false": 
# Create a socket on the proxyserver 
            c = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            hostn = filename.replace("www.","",1) 
            print(hostn) 
            try: 
# Connect to the socket to port 80 
                c.connect((hostn, 80))
# Connect to the socket to port 80 
# Fill in start. 
# Fill in end. 
# Create a temporary file on this socket and ask port 80 
# for the file requested by the client 
                fileobj = c.makefile('r', 0) 
                fileobj.write("GET "+"http://" + filename + " HTTP/1.0\n\n") 
# Read the response into buffer 
# Fill in start. 
                response = b""
                while True:
                    data = client_socket.recv(4096)
                    if not data:
                        break
                    response += data

                c.close()
# Fill in end. 
# Create a new file in the cache for the requested file. 
# Also send the response in the buffer to client socket 
# and the corresponding file in the cache 
                response_str = response.decode("utf-8", errors="ignore")
                headers, _, body = response_str.partition("\r\n\r\n")  # 分离响应头和正文
                tmpFile = open("./" + filename,"wb") 
                tmpFile.write(body.encode())
                tmpFile.close()
                client_socket.send(response)
                client_socket.close()
# Fill in start. 
# Fill in end. 
            except: 
                print("Illegal request") 
        else: 
# HTTP response message for file not found 
# Fill in start. 
# Fill in end. 
# Close the client and the server sockets 
            client_socket.close()
# Fill in start. 
# Fill in end. 
