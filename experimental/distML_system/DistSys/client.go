package main

import (
	"flag"
	"fmt"
	"github.com/arcaneiceman/GoVector/govec"
	"github.com/sbinet/go-python"
	//"math/rand"
	"encoding/binary"
	"math"
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

type Node string

// Global variables
var (
	name               string
	inputargs          []string
	myaddr             *net.TCPAddr
	svaddr             *net.TCPAddr
	logger             *govec.GoLog
	logModule          *python.PyObject
	logInitFunc        *python.PyObject
	logPrivFunc        *python.PyObject
	linModule          *python.PyObject
	linInitFunc        *python.PyObject
	linPrivFunc        *python.PyObject
	dataset            string
	goFloatsToPyObject time.Duration = 0
	gopyLatency        time.Duration = 0
	pyObjectToGoFloats time.Duration = 0
	curIteration       int           = 0
	modelType          string
)

func init() {
	err := python.Initialize()
	if err != nil {
		panic(err.Error())
	}
}

func main() {
	// Parse arguments
	sysPath := python.PySys_GetObject("path")
	python.PyList_Insert(sysPath, 0, python.PyString_FromString("./"))
	python.PyList_Insert(sysPath, 0, python.PyString_FromString("../ML/code"))

	parseArgs()
	fmt.Printf("Node initialized as %v.\n", name)

	numFeatures := python.PyInt_FromLong(0)

	switch modelType {
	case "log":
		logModule = python.PyImport_ImportModule("logistic_model")
		logInitFunc = logModule.GetAttrString("init")
		numFeatures = logInitFunc.CallFunction(python.PyString_FromString(dataset))
		logPrivFunc = logModule.GetAttrString("privateFun")

	case "lin":
		linModule = python.PyImport_ImportModule("linear_model")
		linInitFunc = linModule.GetAttrString("init")
		numFeatures = linInitFunc.CallFunction(python.PyString_FromString(dataset))
		linPrivFunc = linModule.GetAttrString("privateFun")

	case "linL2":
		linModule = python.PyImport_ImportModule("linear_model")
		linInitFunc = linModule.GetAttrString("init")
		numFeatures = linInitFunc.CallFunction(python.PyString_FromString(dataset))
		linPrivFunc = linModule.GetAttrString("privateFunL2")

	}

	// Registering the local node's remote procedure
	node := new(Node)
	rpc.Register(node)

	// Register by doing RPC on server
	nodeInfo := NodeInfo{name, myaddr.String()}
	args := RegisterArgs{nodeInfo, python.PyInt_AsLong(numFeatures)}

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
	select {}
}

// Remote procedure for perfoming gradient descent updates for logistic regression
func (t *Node) RequestUpdateLog(args Weights, reply *Weights) error {
	curIteration++
	start := time.Now()

	argArray := python.PyList_New(len(args.Array))

	for i := 0; i < len(args.Array); i++ {
		python.PyList_SetItem(argArray, i, python.PyFloat_FromDouble(args.Array[i]))
	}

	goFloatsToPyObject += time.Since(start)

	start1 := time.Now()
	result := logPrivFunc.CallFunction(python.PyInt_FromLong(1), argArray)

	gopyLatency += time.Since(start1)

	start2 := time.Now()
	pyByteArray := python.PyByteArray_FromObject(result)

	goByteArray := python.PyByteArray_AsBytes(pyByteArray)

	var goFloatArray []float64
	size := len(goByteArray) / 8

	for i := 0; i < size; i++ {
		currIndex := i * 8
		bits := binary.LittleEndian.Uint64(goByteArray[currIndex : currIndex+8])
		float := math.Float64frombits(bits)
		goFloatArray = append(goFloatArray, float)
	}
	//fmt.Println("Post GoFloatArray initalization")
	pyObjectToGoFloats += time.Since(start2)
	(*reply).Array = goFloatArray

	//fmt.Println("End of method")

	if curIteration == 1000 {
		fmt.Printf("Go float array -> Python List: %s\n", goFloatsToPyObject)
		fmt.Printf("go-python function call: %s\n", gopyLatency)
		fmt.Printf("Python list -> Go float array: %s\n", pyObjectToGoFloats)
	}

	return nil
}

// Remote procedure for perfoming gradient descent updates for linear regression
func (t *Node) RequestUpdateLin(args Weights, reply *Weights) error {
	curIteration++
	start := time.Now()

	argArray := python.PyList_New(len(args.Array))

	for i := 0; i < len(args.Array); i++ {
		python.PyList_SetItem(argArray, i, python.PyFloat_FromDouble(args.Array[i]))
	}

	goFloatsToPyObject += time.Since(start)

	start1 := time.Now()
	result := linPrivFunc.CallFunction(python.PyInt_FromLong(1), argArray, python.PyInt_FromLong(5))

	gopyLatency += time.Since(start1)

	start2 := time.Now()
	pyByteArray := python.PyByteArray_FromObject(result)

	goByteArray := python.PyByteArray_AsBytes(pyByteArray)

	var goFloatArray []float64
	size := len(goByteArray) / 8

	for i := 0; i < size; i++ {
		currIndex := i * 8
		bits := binary.LittleEndian.Uint64(goByteArray[currIndex : currIndex+8])
		float := math.Float64frombits(bits)
		goFloatArray = append(goFloatArray, float)
	}

	pyObjectToGoFloats += time.Since(start2)
	(*reply).Array = goFloatArray

	if curIteration == 15000 {
		fmt.Printf("Go float array -> Python List: %s\n", goFloatsToPyObject)
		fmt.Printf("go-python function call: %s\n", gopyLatency)
		fmt.Printf("Python list -> Go float array: %s\n", pyObjectToGoFloats)
	}

	return nil
}

// Argument parsing function
func parseArgs() {
	flag.Parse()
	inputargs = flag.Args()
	var err error
	if len(inputargs) < 6 {
		fmt.Printf("Not enough inputs.\n")
		fmt.Printf("Argument 1: Hospital name.\n")
		fmt.Printf("Argument 2: Local IP address. Dotted IPv4 notation.\n")
		fmt.Printf("Argument 3: Server IP address. Dotted IPv4 notation.\n")
		fmt.Printf("Argument 4: Name of file for GoVector logging.\n")
		fmt.Printf("Argument 5: Name of dataset file to train on.\n")
		fmt.Printf("Argument 6: Type of model to be trained. Either \"log\" for logistic regression or \"lin\" for linear regression.\n")
		return
	}
	name = inputargs[0]
	myaddr, err = net.ResolveTCPAddr("tcp", inputargs[1])
	checkError(err)
	svaddr, err = net.ResolveTCPAddr("tcp", inputargs[2])
	logger = govec.Initialize(inputargs[0], inputargs[3])
	dataset = inputargs[4]
	modelType = inputargs[5]
}

// Error checking function
func checkError(err error) {
	if err != nil {
		fmt.Fprintf(os.Stderr, "Fatal error: %s", err.Error())
	}
}
