import socket

HOST = '127.0.0.1'  # socket_server.py 실행 중인 서버
PORT = 9000         # 동일한 포트

request = """GET /api_root/Post/ HTTP/1.1
Host: 10.0.2.2:8000
Authorization: Token bf46b8f9337d1d27b4ef2511514c798be1a954b8
User-Agent: Python-Client
Connection: close

"""

client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client_socket.connect((HOST, PORT))
client_socket.sendall(request.encode('utf-8'))
client_socket.close()

print("✅ 요청 전송 완료")
