package main

import (
	"bufio"
	"flag"
	"fmt"
	"github.com/arcaneiceman/GoVector/govec"
	"github.com/sbinet/go-python"
	"net"
	"net/http"
	"net/rpc"
	"os"
	"time"
)

const BUFFSIZE = 4096

// TODO: put this PMPML.lib file (shared header file)
// RPC types

type Weights struct {
	Array []float64
}

type RegisterArgs struct {
	Node        NodeInfo
	NumFeatures int
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
	globalW     Weights
	deltas      Weights
	modelType   string
	testModule  *python.PyObject
	testFunc    *python.PyObject
)

func init() {
	err := python.Initialize()
	if err != nil {
		panic(err.Error())
	}
}

func main() {
	sysPath := python.PySys_GetObject("path")
	python.PyList_Insert(sysPath, 0, python.PyString_FromString("./"))
	python.PyList_Insert(sysPath, 0, python.PyString_FromString("../ML/code"))

	// Initialize data/parse arguments
	client = make(map[string]int)
	claddr = make(map[int]*net.TCPAddr)
	parseArgs()

	switch modelType {

	case "log":
		testModule = python.PyImport_ImportModule("logistic_model_test")

	case "lin":
		testModule = python.PyImport_ImportModule("linear_model_test")

	case "linL2":
		testModule = python.PyImport_ImportModule("linear_model_test")

	}

	testFunc = testModule.GetAttrString("test")

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
	select {}
}

// Remote procedure for handling client join/rejoin requests
func (t *Server) RequestJoin(args RegisterArgs, reply *int) error {
	// process depending on if it is a new node or a returning one
	if _, ok := client[args.Node.NodeName]; !ok {
		// adding a node that has never been added before
		id := maxnode
		maxnode++
		client[args.Node.NodeName] = id
		claddr[id], _ = net.ResolveTCPAddr("tcp", args.Node.NodeIp)
		fmt.Printf("\n--- Added %v as node%v.\n", args.Node.NodeName, id)
	} else {
		// node is rejoining, update address and resend the unfinished test requests
		id := client[args.Node.NodeName]
		claddr[id], _ = net.ResolveTCPAddr("tcp", args.Node.NodeIp)
		fmt.Printf("\n--- %v at node%v is back online.\n", args.Node.NodeName, id)
	}

	if globalW.Array == nil {
		globalW.Array = make([]float64, args.NumFeatures)
	}

	if deltas.Array == nil {
		deltas.Array = make([]float64, args.NumFeatures)
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
		case "train_logistic":
			start := time.Now()
			// TODO: This is synch for now. needs to become asynch
			for i := 1; i <= 1000; i++ {
				fmt.Printf("Iteration %d started.\n", i)
				for name, id := range client {
					rpcCaller, err := rpc.DialHTTP("tcp", claddr[id].String())
					//fmt.Println("Post dial")
					if err != nil {
						// TODO: Deregistration of dead clients (so that you don't contact it again)
						fmt.Printf("\nUnable to contact %s(%s).\n\n", name, claddr[id].String())
						os.Exit(1)
					} else {
						deltas.Array = []float64{}
						err = rpcCaller.Call("Node.RequestUpdateLog", globalW, &deltas)
						rpcCaller.Close()
						if err != nil {
							//fmt.Println(err)
							fmt.Printf("\nRemote procedure call to %s(%s) failed.\n\n", name, claddr[id].String())
							os.Exit(1)
						} else {
							for j := 0; j < len(deltas.Array); j++ {
								globalW.Array[j] += deltas.Array[j]
							}
						}
					}
				}
			}

			fmt.Printf("Global weights (after completion): %v\n", globalW.Array)
			fmt.Printf("Total latency: %s\n", time.Since(start))
			fmt.Print("Enter command: ")

		case "train_linear":
			start := time.Now()
			// TODO: This is synch for now. needs to become asynch
			for i := 1; i <= 15000; i++ {
				fmt.Printf("Iteration %d started.\n", i)
				for name, id := range client {
					rpcCaller, err := rpc.DialHTTP("tcp", claddr[id].String())
					if err != nil {
						// TODO: Deregistration of dead clients (so that you don't contact it again)
						fmt.Printf("\nUnable to contact %s(%s).\n\n", name, claddr[id].String())
						os.Exit(1)
					} else {
						deltas.Array = []float64{}
						err = rpcCaller.Call("Node.RequestUpdateLin", globalW, &deltas)
						rpcCaller.Close()
						if err != nil {
							fmt.Printf("\nRemote procedure call to %s(%s) failed.\n\n", name, claddr[id].String())
							os.Exit(1)
						} else {
							for j := 0; j < len(deltas.Array); j++ {
								globalW.Array[j] += deltas.Array[j]
							}
						}
					}
				}
			}

			fmt.Printf("Global weights (after completion): %v\n", globalW.Array)
			fmt.Printf("Total latency: %s\n", time.Since(start))
			fmt.Print("Enter command: ")

		case "test":

			argArray := python.PyList_New(len(globalW.Array))

			for i := 0; i < len(globalW.Array); i++ {
				python.PyList_SetItem(argArray, i, python.PyFloat_FromDouble(globalW.Array[i]))
			}

			result := testFunc.CallFunction(argArray)
			err := python.PyFloat_AsDouble(result)
			fmt.Printf("The loss is %f\n", err)
			fmt.Print("Enter command: ")

		default:
			fmt.Printf(" Command not recognized: %v.\n\n", ident)
			fmt.Printf("  Choose from the following commands\n")
			fmt.Printf("  train_logistic  -- Trains a global logistic model through gradient descent on registered local nodes\n")
			fmt.Printf("  train_linear  -- Trains a global linear model through gradient descent on registered local nodes\n")
			fmt.Printf("  test  -- Tests the global model on a test set and reports the test error\n")
			fmt.Print("Enter command: ")
		}
	}
}

// Argument parsing function
func parseArgs() {
	flag.Parse()
	inputargs := flag.Args()
	var err error
	if len(inputargs) < 3 {
		fmt.Printf("Not enough inputs.\n")
		fmt.Printf("Argument 1: Local IP address. Dotted IPv4 notation.\n")
		fmt.Printf("Argument 2: Name of file for GoVector logging.\n")
		fmt.Printf("Argument 3: Type of model to be trained. Either \"log\" for logistic regression or \"lin\" for linear regression.\n")
		return
	}
	myaddr, err = net.ResolveTCPAddr("tcp", inputargs[0])
	checkError(err)
	logger = govec.InitGoVector(inputargs[1], inputargs[1])
	modelType = inputargs[2]
}

// Error checking function
func checkError(err error) {
	if err != nil {
		fmt.Fprintf(os.Stderr, "Fatal error: %s", err.Error())
		os.Exit(1)
	}
}
