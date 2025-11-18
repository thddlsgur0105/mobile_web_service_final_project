import socket
import datetime
import os

# 저장 경로 설정
SAVE_DIR = "./request"
os.makedirs(SAVE_DIR, exist_ok=True)

# 서버 설정
HOST = '0.0.0.0'
PORT = 9000

server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.bind((HOST, PORT))
server_socket.listen(1)
print(f"📡 Socket Server started on {HOST}:{PORT}")

try:
    while True:
        client_socket, addr = server_socket.accept()
        print(f"[CONNECTED] {addr}")

        # 데이터 수신
        data = client_socket.recv(4096)
        if data:
            filename = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S") + ".bin"
            filepath = os.path.join(SAVE_DIR, filename)
            
            # 파일로 저장
            with open(filepath, "wb") as f:
                f.write(data)
            
            print(f"✅ Request saved: {filepath}")
            
            # 응답 전송 (선택)
            client_socket.sendall(b"Request received successfully.")

        client_socket.close()
except KeyboardInterrupt:
    print("\nServer shutting down...")
    server_socket.close()
