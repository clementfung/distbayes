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
	name      string
	inputargs []string
	myaddr    *net.TCPAddr
	svaddr    *net.TCPAddr
	logger    *govec.GoLog
	logModule *python.PyObject
	initFunc  *python.PyObject
	privFunc  *python.PyObject
	dataset   string
	//linBuilder *python.PyObject
	//linModel   *python.PyObject
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

	logModule = python.PyImport_ImportModule("logistic_model")

	parseArgs()
	fmt.Printf("Node initialized as %v.\n", name)

	initFunc = logModule.GetAttrString("init")
	numFeatures := initFunc.CallFunction(python.PyString_FromString(dataset))

	privFunc = logModule.GetAttrString("privateFun")

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

// Remote procedure for perfoming gradient descent updates
func (t *Node) RequestUpdate(args Weights, reply *Weights) error {
	// TODO:
	// perform gradient descent here
	// get updated weights and return deltas
	/*for i := 0; i < len(args.Array); i++ {
		(*reply).Array[i] = rand.Float64()
	}*/
	//fmt.Printf("%f", args)
	//reply = rand.Float64()
	//deltas := linModel.GetAttrString("privateFun").CallFunction(python.PyInt_FromLong(2), linModel.GetAttrString("w"))
	//fmt.Printf("%s\n", python.PyString_AsString(deltas.Str()))
	//fmt.Println("Pre PyList allocation")
	argArray := python.PyList_New(len(args.Array))
	//fmt.Println("Post PyList allocation")

	//fmt.Println("Pre PyList initalization")
	for i := 0; i < len(args.Array); i++ {
		python.PyList_SetItem(argArray, i, python.PyFloat_FromDouble(args.Array[i]))
	}
	//fmt.Println("Pre PyList initialization")

	//fmt.Println("Pre privateFun call")
	result := privFunc.CallFunction(python.PyInt_FromLong(1), argArray)
	//fmt.Println("Post privateFun call")

	//fmt.Println("Pre PyByteArray conversion")
	pyByteArray := python.PyByteArray_FromObject(result)
	//fmt.Println("Post PyByteArray conversion")

	//fmt.Println("Pre GoByteArray conversion")
	goByteArray := python.PyByteArray_AsBytes(pyByteArray)
	//fmt.Println("Post GoByteArray conversion")

	//fmt.Println("Pre GoFloatArray allocation")
	var goFloatArray []float64
	size := len(goByteArray) / 8
	//fmt.Println("Post GoFloatArray allocation")

	//fmt.Println("Pre GoFloatArray initialization")
	for i := 0; i < size; i++ {
		currIndex := i * 8
		bits := binary.LittleEndian.Uint64(goByteArray[currIndex : currIndex+8])
		float := math.Float64frombits(bits)
		goFloatArray = append(goFloatArray, float)
	}
	//fmt.Println("Post GoFloatArray initalization")

	(*reply).Array = goFloatArray

	//fmt.Println("End of method")

	return nil
}

// Argument parsing function
func parseArgs() {
	flag.Parse()
	inputargs = flag.Args()
	var err error
	if len(inputargs) < 5 {
		fmt.Printf("Not enough inputs.\n")
		return
	}
	name = inputargs[0]
	myaddr, err = net.ResolveTCPAddr("tcp", inputargs[1])
	checkError(err)
	svaddr, err = net.ResolveTCPAddr("tcp", inputargs[2])
	logger = govec.Initialize(inputargs[0], inputargs[3])
	dataset = inputargs[4]
}

// Error checking function
func checkError(err error) {
	if err != nil {
		fmt.Fprintf(os.Stderr, "Fatal error: %s", err.Error())
	}
}
