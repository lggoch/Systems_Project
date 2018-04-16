screen -d -m -S server python3 sgd_server.py 50051
sleep 5
python3 client.py
screen -X -S server quit


