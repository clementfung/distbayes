package main

import (
	"fmt"
	"math/rand"
	"net"
	"net/http"
	"os"
	"github.com/DistributedClocks/GoVector/govec"
	"github.com/sbinet/go-python"
)

type MessageData struct {
	Type 	string
	Node 	NodeInfo
	Deltas 	[]float64
}

type NodeInfo struct {
	NodeId	 	string
	NumFeatures int
}

var (
	maxnode     		int = 0
	numFeatures			int = 0
	myWeights 			[]float64
	registeredNodes		map[string]string

	// Test Module for python
	testModule  *python.PyObject
	testFunc    *python.PyObject
	trainFunc    *python.PyObject

)

func handler(w http.ResponseWriter, r *http.Request) {

	req := r.URL.Path[1:]
    
    fmt.Fprintf(w, "Welcome %s!\n\n", r.URL.Path[1:])
    fmt.Fprintf(w, "Num nodes: %d\n", maxnode)
    fmt.Fprintf(w, "Weights: %v\n", myWeights)

    if req == "test" {
    	train_error, test_error := testModel()
    	fmt.Fprintf(w, "Train Loss: %f\n", train_error)	
    	fmt.Fprintf(w, "Test Loss: %f\n", test_error)	
    }

}

func httpHandler() {
	http.HandleFunc("/", handler)
	http.ListenAndServe(":8080", nil)

	fmt.Printf("HTTP initialized.\n")
}

func init() {
	err := python.Initialize()
	if err != nil {
		panic(err.Error())
	}
}

func pyInit() {

	sysPath := python.PySys_GetObject("path")
	python.PyList_Insert(sysPath, 0, python.PyString_FromString("./"))
	python.PyList_Insert(sysPath, 0, python.PyString_FromString("../ML/code"))

	testModule = python.PyImport_ImportModule("logistic_model_test")
	trainFunc = testModule.GetAttrString("train_error")
	testFunc = testModule.GetAttrString("test_error")

}

func main() {

	fmt.Println("Launching server...")

	pyInit()
	
	go httpHandler()
	go tcpSetup("127.0.0.1:5005")

    // Keeps server running
	select {}
}

// Sets up the TCP connection, and attaches GoVector.
func tcpSetup(address string) {

	Logger := govec.InitGoVector("torserver", "torserverfile")

	// listen on all interfaces
	myaddr, err := net.ResolveTCPAddr("tcp", address)
	checkError(err)

	ln, err := net.ListenTCP("tcp", myaddr)
	checkError(err)

	buf := make([]byte, 2048)
	outBuf := make([]byte, 2048)
	registeredNodes = make(map[string]string)

	for {

		fmt.Printf("Listening for TCP....\n")

		conn, err := ln.Accept()
		checkError(err)

		fmt.Println("Got message")

		// Get the message from client
		conn.Read(buf)

		var incomingData MessageData
		Logger.UnpackReceive("Received Message From Client", buf[0:], &incomingData)
	
		var ok bool

		switch incomingData.Type {
			
			// Client requests copy of model, return myWeights
			case "req":
				outBuf = Logger.PrepareSend("Replying", myWeights)

			// Client is sending a gradient update. Apply it and return myWeights
			case "grad":
				ok = gradUpdate(incomingData.Node.NodeId, incomingData.Deltas)
				outBuf = Logger.PrepareSend("Replying", myWeights)
			
			// Add new nodes
			case "join":
				
				ok = processJoin(incomingData.Node)

				if ok {
			  		outBuf = Logger.PrepareSend("Replying", 1)
				} else {
					outBuf = Logger.PrepareSend("Replying", 0)
				}

			case "beat":

				fmt.Printf("Heartbeat from %s\n", incomingData.Node)
				outBuf = Logger.PrepareSend("Replying to heartbeat", 1)

			default:
				ok = false
				outBuf = nil
				
		}

	  	conn.Write(outBuf)
		fmt.Printf("Done processing data from %s\n", incomingData.Node.NodeId)

	}

}

func testModel() (float64, float64) {

	argArray := python.PyList_New(len(myWeights))

	for i := 0; i < len(myWeights); i++ {
		python.PyList_SetItem(argArray, i, python.PyFloat_FromDouble(myWeights[i]))
	}

	test_result := testFunc.CallFunction(argArray)
	test_err := python.PyFloat_AsDouble(test_result)

	train_result := trainFunc.CallFunction(argArray)
	train_err := python.PyFloat_AsDouble(train_result)
	
	return train_err, test_err

}

func processJoin(node NodeInfo) bool {

	_, exists := registeredNodes[node.NodeId]
	
	if exists {
		fmt.Printf("Rejected a join from: %s\n", node.NodeId)
		return false
	}

	// After the first registration: set the number of features. Fail otherwise
	if numFeatures != 0 && numFeatures != node.NumFeatures {
		return false
	} else if numFeatures == 0 {
		
		numFeatures = node.NumFeatures
		myWeights = make([]float64, node.NumFeatures)

		for i := 0; i < numFeatures; i++ {
			myWeights[i] = rand.Float64()
 		}

	}

	// Add node
	registeredNodes[node.NodeId] = "in" 
	fmt.Println("Joined " + node.NodeId)
	maxnode++

	return true

}

func gradUpdate(nodeId string, deltas []float64) bool {

	_, exists := registeredNodes[nodeId]
	
	//fmt.Printf("Deltas: %v", deltas)

	if exists { 
		// Add in the deltas
		for j := 0; j < len(deltas); j++ {
			myWeights[j] += deltas[j]
		}
	}

	fmt.Printf("Grad update from: %s %t \n", nodeId, exists)
	return exists

}

// Error checking function
func checkError(err error) {
	if err != nil {
		fmt.Fprintf(os.Stderr, "Fatal error: %s", err.Error())
		os.Exit(1)
	}
}