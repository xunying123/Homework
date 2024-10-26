import socket
import sys
import time

def parse_port(port_string):
    try:
        port = int(port_string)
        if port < 1024 or port > 65535:
            raise ValueError()
        return port
    except Exception:
        print("Error: port number must be in the range 1024 to 65535")
        sys.exit(1)

def print_invalid_arguments():
    print("Error: invalid arguments")
    sys.exit(1)

def tcp_server(port):
    # 创建socket，listen并接收connect
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind(('', port))
    server_socket.listen(1)
    # print(f"Server listening on port {port}")

    # 等待客户端连接
    conn, addr = server_socket.accept()
    # print(f"Connected by {addr}")

    total_data = 0
    start_time = time.time()

    # 按照每1000 bytes进行读取，直到读完
    while True:
        data = conn.recv(1000)  # 每次读取1000字节
        if not data:
            break
        total_data += len(data)

    end_time = time.time()
    duration = end_time - start_time
    rate = (total_data / 1000000) / duration * 8  # KB/s 转为 Mbps

    print(f"Received {total_data // 1000} KB, Rate: {rate:.3f} Mbps")
    
    # 关闭
    conn.close()
    server_socket.close()

def tcp_client(host, port, duration):
    # 创建socket
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect((host, port))
    # print(f"Connected to {host}:{port}")

    total_data = 0
    start_time = time.time()

    # 按照1000 bytes进行输出，直到duration，记录发送数据大小以及时间
    while time.time() - start_time < duration:
        client_socket.sendall(b'x' * 1000)  # 发送1000字节的数据
        total_data += 1000

    # 关闭
    client_socket.close()
    rate = (total_data / 1000000) / duration * 8  # KB/s to Mbps
    print(f"Sent {total_data // 1000} KB, Rate: {rate:.3f} Mbps")

if __name__ == '__main__':
    if len(sys.argv) == 8 and sys.argv[1] == "-c" and sys.argv[2] == "-h" and sys.argv[4] == "-p" and sys.argv[6] == "-t":
        port = parse_port(sys.argv[5])
        try:
            duration = int(sys.argv[7])
        except Exception:
            print_invalid_arguments()
        tcp_client(sys.argv[3], port, duration)
    elif len(sys.argv) == 4 and sys.argv[1] == "-s" and sys.argv[2] == "-p":
        port = parse_port(sys.argv[3])
        tcp_server(port)
    else:
        print_invalid_arguments()
