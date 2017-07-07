#!/bin/sh
killall xterm
export GOROOT=/usr/local/go
export GOPATH=$HOME/distbayes
export PATH=$GOPATH/bin:$GOROOT/bin:$PATH
go build -o client_b ./client/client_go.go
go build -o server_b ./server/server_go.go
./server_b 127.0.0.1:6672 s1&
sleep 2
xterm -e ./client_b hospital1 127.0.0.1:6664 127.0.0.1:6672 h1 &
xterm -e ./client_b hospital2 127.0.0.1:6662 127.0.0.1:6672 h2 &
xterm -e ./client_b hospital3 127.0.0.1:6663 127.0.0.1:6672 h3 &