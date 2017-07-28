package main

import (
	"fmt"
	"math/rand"
	"net"
	"net/http"
	"os"
	"github.com/DistributedClocks/GoVector/govec"
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
)

func handler(w http.ResponseWriter, r *http.Request) {
    
    fmt.Fprintf(w, "Welcome %s!\n\n", r.URL.Path[1:])
    fmt.Fprintf(w, "Num nodes: %d\n", maxnode)
    fmt.Fprintf(w, "Weights: %v", myWeights)
}

func httpHandler() {
	http.HandleFunc("/", handler)
	http.ListenAndServe(":8080", nil)

	fmt.Printf("HTTP initialized.\n")
}

func main() {

	fmt.Println("Launching server...")
	go httpHandler()
	go tcpSetup("127.0.0.1:6677")

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

	fmt.Printf("Listening for TCP....\n")
	buf := make([]byte, 512)
	outBuf := make([]byte, 512)
	registeredNodes = make(map[string]string)

	for {

		conn, err := ln.Accept()
		checkError(err)

		fmt.Println("Got message")

		// Get the message from client
		conn.Read(buf)

		var incomingData MessageData
		Logger.UnpackReceive("Received Message From Client", buf[0:], &incomingData)
		fmt.Printf("Received %v\n", incomingData)

		var ok bool

		switch incomingData.Type {
			
			// A gradient update
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

			default:
				ok = false
				outBuf = nil
		}

	  	conn.Write(outBuf)
		fmt.Printf("State %v\n", myWeights)

	}

}

func trainGlobal() {

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