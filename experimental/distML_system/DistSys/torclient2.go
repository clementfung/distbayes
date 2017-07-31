package main

import (
	"encoding/binary"
	"flag"
	"fmt"
	"math"
	"os"
	"golang.org/x/net/proxy"
	"github.com/DistributedClocks/GoVector/govec"
	"github.com/sbinet/go-python"
)

/*
	An attempt at Tor via TCP. 
*/
var ONION_HOST string = "4255ru2izmph3efw.onion:6677"
var TOR_PROXY string = "127.0.0.1:9150"

var (
	name string
	datasetName string
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
	node 			NodeInfo
	logModule      	*python.PyObject
	logInitFunc     *python.PyObject
	logPrivFunc     *python.PyObject
	numFeatures     *python.PyObject
	pulledGradient  []float64
)

func init() {
	err := python.Initialize()
	if err != nil {
		panic(err.Error())
	}
}

func pyInit(datasetName string) {

	sysPath := python.PySys_GetObject("path")
	python.PyList_Insert(sysPath, 0, python.PyString_FromString("./"))
	python.PyList_Insert(sysPath, 0, python.PyString_FromString("../ML/code"))

	logModule = python.PyImport_ImportModule("logistic_model")
	logInitFunc = logModule.GetAttrString("init")
	logPrivFunc = logModule.GetAttrString("privateFun")
	numFeatures = logInitFunc.CallFunction(python.PyString_FromString(datasetName))

  	node.NumFeatures = python.PyInt_AsLong(numFeatures)
  	pulledGradient = make([]float64, node.NumFeatures)

}

func main() {

	parseArgs()
	logger := govec.InitGoVector(name, name)
	torDialer := getTorDialer()
	node.NodeId = name

	// Initialize the python side
	pyInit(datasetName)

  	joined := sendJoinMessage(logger, torDialer)

  	if joined == 0 {
  		fmt.Println("Could not join.")
  		return
  	}

  	for i := 0; i < 200; i++ { 
    	sendGradMessage(logger, torDialer, pulledGradient)
  	}

  	fmt.Println("The end")

}

func parseArgs() {
	flag.Parse()
	inputargs := flag.Args()
	if len(inputargs) < 2 {
		fmt.Println("USAGE: go run torclient.go pName datasetName")
		os.Exit(1)
	}
	name = inputargs[0]
	datasetName = inputargs[1]
	fmt.Println("Done parsing args.")
}

func getTorDialer() proxy.Dialer {

	// Create proxy dialer using Tor SOCKS proxy
	fmt.Println("Trying 9150")
	torDialer, err := proxy.SOCKS5("tcp", TOR_PROXY, nil, proxy.Direct)
	checkError(err)

	fmt.Println("SOCKS5 Dial success!")
	return torDialer

}

func sendGradMessage(logger *govec.GoLog, torDialer proxy.Dialer, globalW []float64) int {
	
	// Connect to the server via Tor
	conn, err := torDialer.Dial("tcp", ONION_HOST)
	checkError(err)

	var msg MessageData
	msg.Type = "grad"
	msg.Node = node
	msg.Deltas, err = oneGradientStep(globalW)
	checkError(err)

	outBuf := logger.PrepareSend("Sending packet to torserver", msg)
	
	_, err = conn.Write(outBuf)
	checkError(err)
	
	inBuf := make([]byte, 512)
	n, errRead := conn.Read(inBuf)
	checkError(errRead)

	var incomingMsg []float64
	logger.UnpackReceive("Received Message from server", inBuf[0:n], &incomingMsg)

	conn.Close()

	pulledGradient = incomingMsg
	return 1

}

func sendJoinMessage(logger *govec.GoLog, torDialer proxy.Dialer) int {

	conn, err := torDialer.Dial("tcp", ONION_HOST)
  	checkError(err)

  	fmt.Println("TOR Dial Success!")
	fmt.Println("Sending Join")

	var msg MessageData
    msg.Type = "join"
    msg.Node = node
    msg.Deltas = make([]float64, node.NumFeatures)

    outBuf := logger.PrepareSend("Sending packet to torserver", msg)
    	
	_, errWrite := conn.Write(outBuf)
	checkError(errWrite)
	
	inBuf := make([]byte, 512)
	n, errRead := conn.Read(inBuf)
	checkError(errRead)

	var incomingMsg int
	logger.UnpackReceive("Received Message from server", inBuf[0:n], &incomingMsg)

	conn.Close()

	return incomingMsg

}

func oneGradientStep(globalW []float64) ([]float64, error) {
	
	argArray := python.PyList_New(len(globalW))

	for i := 0; i < len(globalW); i++ {
		python.PyList_SetItem(argArray, i, python.PyFloat_FromDouble(globalW[i]))
	}

	result := logPrivFunc.CallFunction(python.PyInt_FromLong(1), argArray)
	
	// Convert the resulting array to a go byte array
	pyByteArray := python.PyByteArray_FromObject(result)
	goByteArray := python.PyByteArray_AsBytes(pyByteArray)

	var goFloatArray []float64
	size := len(goByteArray) / 8

	for i := 0; i < size; i++ {
		currIndex := i * 8
		bits := binary.LittleEndian.Uint64(goByteArray[currIndex : currIndex+8])
		aFloat := math.Float64frombits(bits)
		goFloatArray = append(goFloatArray, aFloat)
	}
	
	return goFloatArray, nil

}

// Error checking function
func checkError(err error) {
	if err != nil {
		fmt.Fprintf(os.Stderr, "Fatal error: %s", err.Error())
		os.Exit(1)
	}
}
