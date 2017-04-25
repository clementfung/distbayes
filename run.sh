#!/bin/sh
killall xterm
go build -o client_b ./client/client_go.go
go build -o server_b ./server/server_go.go
xterm -e ./server_b 127.0.0.1:6666 s1&
sleep 2
xterm -e ./client_b hospital1 127.0.0.1:6664 127.0.0.1:6666 testdata/x1.txt testdata/y1.txt testdata/xv.txt testdata/yv.txt h1 &
xterm -e ./client_b hospital2 127.0.0.1:6662 127.0.0.1:6666 testdata/x2.txt testdata/y2.txt testdata/xv.txt testdata/yv.txt h2 &
xterm -e ./client_b hospital3 127.0.0.1:6663 127.0.0.1:6666 testdata/x3.txt testdata/y3.txt testdata/xv.txt testdata/yv.txt h3 &