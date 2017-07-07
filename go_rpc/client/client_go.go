package main

import (
	"flag"
	"fmt"
	"github.com/arcaneiceman/GoVector/govec"
	"math/rand"
	"net"
	"net/http"
	"net/rpc"
	"os"
)

const BUFFSIZE = 4096

// TODO: put this PMPML.lib file (shared header file)
// RPC types
type Gradient struct {
	Array [20]float64
}

type NodeInfo struct {
	NodeName string
	NodeIp   string
}

type Node string

// Global variables
var (
	name      string
	inputargs []string
	myaddr    *net.TCPAddr
	svaddr    *net.TCPAddr
	logger    *govec.GoLog
)

func main() {
	// Parse arguments
	parseArgs()
	fmt.Printf("Node initialized as %v.\n", name)

	// Registering the local node's remote procedure
	node := new(Node)
	rpc.Register(node)

	// Register by doing RPC on server
	args := NodeInfo{name, myaddr.String()}
	var reply int = 0
	rpcCaller, err := rpc.DialHTTP("tcp", svaddr.String())

	for reply != 1 {
		err = rpcCaller.Call("Server.RequestJoin", args, &reply)
		checkError(err)
	}

	fmt.Printf("Node has registered with server\n")

	// Setting up RPC listener for local node
	rpc.HandleHTTP()
	listener, err := net.Listen("tcp", myaddr.String())
	checkError(err)
	go http.Serve(listener, nil)

	// Keeps local node running
	for {

	}
}

// Remote procedure for perfoming gradient descent updates
func (t *Node) RequestUpdate(args Gradient, reply *Gradient) error {
	// TODO:
	// perform gradient descent here
	// get updated weights and return deltas
	for i := 0; i < len(args.Array); i++ {
		(*reply).Array[i] = rand.Float64()
	}
	//fmt.Printf("%f", args)
	//reply = rand.Float64()
	return nil
}

// Argument parsing function
func parseArgs() {
	flag.Parse()
	inputargs = flag.Args()
	var err error
	if len(inputargs) < 4 {
		fmt.Printf("Not enough inputs.\n")
		return
	}
	name = inputargs[0]
	myaddr, err = net.ResolveTCPAddr("tcp", inputargs[1])
	checkError(err)
	svaddr, err = net.ResolveTCPAddr("tcp", inputargs[2])
	logger = govec.Initialize(inputargs[0], inputargs[3])
}

// Error checking function
func checkError(err error) {
	if err != nil {
		fmt.Fprintf(os.Stderr, "Fatal error: %s", err.Error())
	}
}
