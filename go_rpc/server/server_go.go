package main

import (
	"bufio"
	"flag"
	"fmt"
	"github.com/arcaneiceman/GoVector/govec"
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

// TODO: This should be storing all the shared RPC state
type Server string

var (
	maxnode     int = 0
	myaddr      *net.TCPAddr
	client      map[string]int
	claddr      map[int]*net.TCPAddr
	tcplistener *net.TCPListener
	logger      *govec.GoLog
	globalW     Gradient
	deltas      Gradient
)

func main() {
	// Initialize data/parse arguments
	client = make(map[string]int)
	claddr = make(map[int]*net.TCPAddr)
	parseArgs()

	// Registering the server's remote procedure
	server := new(Server)
	rpc.Register(server)

	// Setting up RPC listener for server
	rpc.HandleHTTP()
	listener, err := net.Listen("tcp", myaddr.String())
	checkError(err)
	go http.Serve(listener, nil)

	fmt.Printf("Server initialized.\n")
	fmt.Print("Enter command: ")

	// User input parsing goroutine
	go parseUserInput()

	// Keeps server running
	for {

	}
}

// Remote procedure for handling client join/rejoin requests
func (t *Server) RequestJoin(args NodeInfo, reply *int) error {
	// process depending on if it is a new node or a returning one
	if _, ok := client[args.NodeName]; !ok {
		// adding a node that has never been added before
		id := maxnode
		maxnode++
		client[args.NodeName] = id
		claddr[id], _ = net.ResolveTCPAddr("tcp", args.NodeIp)
		fmt.Printf("\n--- Added %v as node%v.\n", args.NodeName, id)
	} else {
		// node is rejoining, update address and resend the unfinished test requests
		id := client[args.NodeName]
		claddr[id], _ = net.ResolveTCPAddr("tcp", args.NodeIp)
		fmt.Printf("\n--- %v at node%v is back online.\n", args.NodeName, id)
	}
	fmt.Print("Enter command: ")

	*reply = 1 // indicates successful join
	return nil
}

// User input parsing function
func parseUserInput() {
	for {
		var ident string
		reader := bufio.NewReader(os.Stdin)
		text, _ := reader.ReadString('\n')
		if len(text) <= 1 {
			fmt.Print("Enter command: ")
			continue
		}

		//Windows adds its own strange carriage return, the following lines fix it
		if text[len(text)-2] == '\r' {
			ident = text[0 : len(text)-2]
		} else {
			ident = text[0 : len(text)-1]
		}
		switch ident {
		case "getNums":
			fmt.Printf("getNums just received by server!!\n")
			// TODO: This is synch for now. needs to become asynch
			for name, id := range client {
				rpcCaller, err := rpc.DialHTTP("tcp", claddr[id].String()) // This fails only when a local node is down (possible), so we do not panic if err is not nil
				if err != nil {
					// TODO: Deregistration of dead clients (so that you dont contact it again)
					fmt.Printf("\nUnable to contact %s(%s)\n\n", name, claddr[id].String())

				} else {
					err = rpcCaller.Call("Node.RequestUpdate", globalW, &deltas)
					checkError(err) // Something failed during the RPC call, so we should panic
					fmt.Printf("\nFor client address: %s, globalW = %v, deltas = %v\n\n", claddr[id].String(), globalW, deltas)
				}
			}
			fmt.Print("Enter command: ")
		default:
			fmt.Printf(" Command not recognized: %v.\n\n", ident)
			fmt.Printf("  Choose from the following commands\n")
			fmt.Printf("  getNums  -- Pushes weight and prediction to server\n")
			fmt.Print("Enter command: ")
		}
	}
}

// Argument parsing function
func parseArgs() {
	flag.Parse()
	inputargs := flag.Args()
	var err error
	if len(inputargs) < 2 {
		fmt.Printf("Not enough inputs.\n")
		return
	}
	myaddr, err = net.ResolveTCPAddr("tcp", inputargs[0])
	checkError(err)
	logger = govec.Initialize(inputargs[1], inputargs[1])
}

// Error checking function
func checkError(err error) {
	if err != nil {
		fmt.Fprintf(os.Stderr, "Fatal error: %s", err.Error())
		os.Exit(1)
	}
}
