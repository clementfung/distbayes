package main

import (
	"fmt"
	"net/rpc"
	"os"
	"golang.org/x/net/proxy"
)

/*
	An attempt at Tor via RPC. 
*/
	
var ONION_HOST string = "4255ru2izmph3efw.onion:6666"

type RegisterArgs struct {
	Node        NodeInfo
	NumFeatures int
}

type NodeInfo struct {
	NodeName string
	NodeIp   string
}

func main() {

	// Create proxy dialer using Tor SOCKS proxy
	fmt.Println("Trying 9150")
	torDialer, err := proxy.SOCKS5("tcp", "127.0.0.1:9150", nil, proxy.Direct)
	checkError(err)

	fmt.Println("SOCKS5 Dial success!")

	rpcCaller, err := TorDial(torDialer, "tcp", ONION_HOST)
	checkError(err)

	fmt.Println("TOR Dial Success!")

	var reply int = 0
	
	nodeInfo := NodeInfo{"aName", "127.0.0.1:7777"}
	args := RegisterArgs{nodeInfo, 100}

	for reply != 1 {
		fmt.Println(args)
		fmt.Println("Calling the RPC.")
		err = rpcCaller.Call("Server.RequestJoin", args, &reply)
		checkError(err)
	}

	fmt.Printf("Node has registered with server\n")

	select {}

}

// DialHTTPPath connects to an HTTP RPC server
// at the specified network address and path.
func TorDial(dialer proxy.Dialer, network, address string) (*rpc.Client, error) {
	
	conn, err := dialer.Dial(network, address)
  	checkError(err)

  	return rpc.NewClient(conn), err
}

// Error checking function
func checkError(err error) {
	if err != nil {
		fmt.Fprintf(os.Stderr, "Fatal error: %s", err.Error())
		os.Exit(1)
	}
}
