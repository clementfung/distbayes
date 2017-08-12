package main

import (
	"encoding/binary"
	"encoding/gob"
	"flag"
	"fmt"
	"github.com/arcaneiceman/GoVector/govec"
	"github.com/sbinet/go-python"
	"math"
	"net"
	"os"
	"time"
)

var (
	commitNum  int                          = 0 // Current commit number (to be assigned to a model when a new commit is made)
	nodeNum    int                          = 0 // Current node ID (to be assigned to a new node when it first joins)
	myaddr     *net.TCPAddr                     // Server IP address
	cnumhist   map[int]int                      // Commit number map that is used to map commit numbers to node IDs (key: commit number, value: node ID)
	client     map[string]int                   // Client map that is used to map node names to node IDs (key: node name, value: node ID)
	claddr     map[int]*net.TCPAddr             // Client map that is used to map node IDs to node IP addresses (key: node ID, value: node IP address)
	tempmodels map[int]ServerModelWithState     // Model map that is used to map node IDs to server representations of committed local models
	testqueue  map[int]map[int]bool             // For a particular node's model, this returns an array (map[int]bool) that shows which nodes have a test request pending for that model
	models     map[int]ServerModelWithState     // Model map that is used to map node IDs to server representations of committed local models that have been sufficiently validated
	totalSize  float64                          // Server's running sum of total data across the nodes
	channel    chan message                     // Used to deliver test results for models (from different nodes) asynchronously to updateGlobal() to prevent loss of test results
	logger     *govec.GoLog                     // GoVector logger for debugging purposes (currently useless, useful logging still needs to be implemented in this system)
	l          *net.TCPListener                 // Listens for messages from clients (blocks until it receives a message)
	gempty     ILGlobalModel                    // Empty global model (sent to nodes in certain contexts where sending the global model is not necessary)
	gmodel     ILGlobalModel                    // Server variable holding the current global model
	server     *python.PyObject                 // Server variable that holds reference to server.py module
	genGlobal  *python.PyObject                 // Server variable that holds reference to GenGlobal function in server.py module
)

// Local model struct
type ILModel struct {
	Model      string  // Pickled string representing an Python sklearn model
	Size       float64 // The number of rows the model was trained on
	LocalError float64 // The error of the model on the node's training data
}

// Global model struct
// 'models' is an array pickled strings and weights is a normalized array of weights (weight at an index is the weight for the model at the same index in models slice))
type ILGlobalModel struct {
	Models  []string
	Weights []float64
}

// Server representation of local models
type ServerModelWithState struct {
	commitNum int             // Commit number (used to map this model to the original node that created it)
	model     string          // Pickled string representing an Python sklearn model
	errors    map[int]float64 // Errors of this model on the different nodes (map/array structure)
	totalSize float64         // Running sum of how many rows the model has been validated on
	trainSize float64         // How many rows the model was originally trained on
}

// Message struct used for server-client communication
type message struct {
	Id       int           // Message ID (this is mainly used to determine the commit number for the model and map that back to the node ID of the original node that trained the model)
	NodeIp   string        // String representation of sender's IP address
	NodeName string        // Name of the node sending the mssage
	Type     string        // The type of the message (request or response for example)
	Model    ILModel       // A local model (usage in the message depends on context)
	GModel   ILGlobalModel // Global model (usage in the message depends on context)
}

// Used to respond to client requests
type response struct {
	Resp  string
	Error string
}

// Runs before the main function; starts Python interpreter and handles module importing
func init() {

	// Initializes Python interpreter
	err := python.Initialize()
	if err != nil {
		panic(err.Error())
	}

	// Required so that server.py can be imported directly as a module using relative paths
	sysPath := python.PySys_GetObject("path")
	python.PyList_Insert(sysPath, 0, python.PyString_FromString("./"))
	python.PyList_Insert(sysPath, 0, python.PyString_FromString("../python/code"))

	// Imports server.py and gets the GenGlobal function as *python.PyObject
	server = python.PyImport_ImportModule("server")
	genGlobal = server.GetAttrString("GenGlobal")
}

func main() {
	//Initialize bookkeeping and other server variables
	client = make(map[string]int)
	claddr = make(map[int]*net.TCPAddr)
	models = make(map[int]ServerModelWithState)
	totalSize = 0.0
	tempmodels = make(map[int]ServerModelWithState)
	testqueue = make(map[int]map[int]bool)
	cnumhist = make(map[int]int)
	channel = make(chan message)

	//Start asynchronous global model update goroutine
	go updateGlobal(channel)

	// Parsing input arguments
	parseArgs()

	// Initialize TCP Connection and listener
	l, _ = net.ListenTCP("tcp", myaddr)
	fmt.Printf("Server initialized.\n")

	// Listen for clients and launch handler when one joins
	for {
		conn, err := l.AcceptTCP()
		checkError(err)
		go connHandler(conn)
	}

}

// Function for handling client requests
func connHandler(conn *net.TCPConn) {

	// TODO GoVector.... use PrepareSend and UnpackReceive

	var msg message
	dec := gob.NewDecoder(conn)
	enc := gob.NewEncoder(conn)
	err := dec.Decode(&msg)
	checkError(err)
	switch msg.Type {
	case "commit_request":
		// Node is sending a model, must forward to others for testing
		flag := checkQueue(client[msg.NodeName]) // Checks to see if the committing node has pending requests
		fmt.Printf("<-- Received commit request from %v.\n", msg.NodeName)
		if flag {
			// Accept commit from node and process outgoing test requests
			processTestRequest(msg, conn)
		} else {
			// Deny commit request
			enc.Encode(response{"NO", "Pending tests are not complete"})
			fmt.Printf("--> Denied commit request from %v.\n", msg.NodeName)
			conn.Close()
		}
	case "global_request":
		// Node is requesting the global model, server generates and sends the global model
		enc.Encode(response{"OK", ""})
		fmt.Printf("<-- Received global model request from %v.\n", msg.NodeName)
		genGlobalModel()
		sendGlobal(msg)
		conn.Close()
	case "test_complete":
		// Node is submitting test results
		fmt.Printf("<-- Received completed test results from %v.\n", msg.NodeName)

		// Check to see if the incoming test results are for an outdated model
		notOutdated := true
		for i := msg.Id + 1; i < len(cnumhist); i++ {
			if cnumhist[i] == cnumhist[msg.Id] {
				notOutdated = false
				break
			}
		}

		if notOutdated && testqueue[client[msg.NodeName]][cnumhist[msg.Id]] {
			// If not outdated, update testqueue to resolve test and pass the test results on to the channel
			testqueue[client[msg.NodeName]][cnumhist[msg.Id]] = false
			channel <- msg
			enc.Encode(response{"OK", "Test Processed"})
		} else {
			// Duplicated or outdated results
			enc.Encode(response{"NO", "Duplicate Test"})
			fmt.Printf("--> Ignored test results from %v.\n", msg.NodeName)
		}
		conn.Close()
	case "join_request":
		// Services a joining node
		processJoin(msg)
		enc.Encode(response{"OK", "Joined"})
		conn.Close()
	default:
		// Unknown request
		fmt.Printf("something weird happened!\n")
		enc.Encode(response{"NO", "Unknown Request"})
		conn.Close()
	}

}

// Global model update function (this runs asynchronously after the server is started and runs forever using an infinite loop)
func updateGlobal(ch chan message) {
	// Function that aggregates the global model and commits when ready
	for {
		// This puts test results into m
		m := <-ch

		// This retrieves the ID of the node that generated the model by using the commit number
		id := cnumhist[m.Id]

		// This updates the server representation of local models appropriately using test results from other nodes
		tempmodel := tempmodels[id]
		tempmodel.totalSize += m.Model.Size
		tempmodel.errors[client[m.NodeName]] = m.Model.LocalError
		tempmodels[id] = tempmodel

		// This checks whether the model has been validated on more rows of data than the current server maximum and updates accordingly
		if totalSize < tempmodel.totalSize {
			totalSize = tempmodel.totalSize
		}

		// If the model being updated has been validated on enough data or not (if so, it will be added to the global model the next time the global model is generated)
		if float64(tempmodel.totalSize) > float64(totalSize)*0.6 {
			models[id] = tempmodel
			t := time.Now()
			logger.LogLocalEvent(fmt.Sprintf("%s - Committed model%v by %v at partial commit %v.", t.Format("15:04:05.0000"), id, client[m.NodeName], tempmodel.totalSize/totalSize*100.0))
			fmt.Printf("--- Committed model%v for commit number: %v.\n", id, tempmodel.commitNum)
		}
	}
}

// Generate global model from partial commits
func genGlobalModel() {

	// Python lists that are passed as arguments to the global model generator function
	// errors is basically the 2D array which holds the errors for models across different nodes
	// sizes is basically the array that holds the number of rows that models at the different nodes were originally trained on (basically n at each of the nodes)
	errors := python.PyList_New(len(client) * len(client))
	sizes := python.PyList_New(len(client))

	// This temporary slice holds the pickled Python models that are part of the overall global model
	var modelsSlice []string

	// Ierate over all nodes that joined at some point
	for i := 0; i < nodeNum; i++ {
		// Checks to see if there is a committed model (which has been validated sufficiently) for the current node
		if currModel, ok := models[i]; ok {

			// If there is such a model, the pickled string is appended to the temp slice and the size is appended to one of our Python argument lists
			modelsSlice = append(modelsSlice, currModel.model)
			python.PyList_SetItem(sizes, i, python.PyFloat_FromDouble(currModel.trainSize))

			// Iterate over the errors of each model across different nodes and adds those errors to one of our Python argument lists, adds negative inifinity if no error is found
			for j := 0; j < nodeNum; j++ {
				if currError, valid := currModel.errors[j]; valid {
					python.PyList_SetItem(errors, i*len(client)+j, python.PyFloat_FromDouble(currError))
				} else {
					python.PyList_SetItem(errors, i*len(client)+j, python.PyFloat_FromDouble(math.Inf(-1)))
				}
			}

		} else {
			// If there is no such model, the empty string is appened and the training size is set to 0 (which is used to skip over models when the global model is generated)
			modelsSlice = append(modelsSlice, "")
			python.PyList_SetItem(sizes, i, python.PyFloat_FromDouble(0.0))
		}
	}

	// Global model pickled strings slice is set
	gmodel.Models = modelsSlice

	// This calls Python to generate the weights; the global model weights are generated by the Python function and set by the helper function
	pyWeights := genGlobal.CallFunction(errors, sizes)
	gmodel.Weights = pyFloatArrayToGoFloatArray(pyWeights)
}

// Function that generates test requests following a commit request
func processTestRequest(m message, conn *net.TCPConn) {
	// Getting a commit number that associates test results for the committed model with the committed model itself
	tempcnum := commitNum
	commitNum++
	cnumhist[tempcnum] = client[m.NodeName]

	enc := gob.NewEncoder(conn)

	// Creates a map for storing the model errors across nodes and initializes with error from committing node
	temperrors := make(map[int]float64)
	temperrors[client[m.NodeName]] = m.Model.LocalError
	tempmodels[client[m.NodeName]] = ServerModelWithState{tempcnum, m.Model.Model, temperrors, m.Model.Size, m.Model.Size}

	// Loops through all the other nodes and creates a test request for the committed model (the test queue is created if it didn't exist already)
	for _, id := range client {
		if id != client[m.NodeName] {
			if queue, ok := testqueue[id]; !ok {
				queue := make(map[int]bool)
				queue[cnumhist[tempcnum]] = true
				testqueue[id] = queue
			} else {
				queue[cnumhist[tempcnum]] = true
			}
		}
	}

	fmt.Printf("--- Processed commit %v for node %v.\n", tempcnum, m.NodeName)
	// Sanitize the model for testing (hides model information from other nodes to which test requests are sent)
	m.Model.LocalError = 0.0
	m.Model.Size = 0.0
	enc.Encode(response{"OK", "Committed"})
	conn.Close()

	// The test requests for this committed model are sent to the other nodes (those created above)
	for name, id := range client {
		if id != client[m.NodeName] {
			sendTestRequest(name, id, tempcnum, m.Model)
		}
	}
}

// Function that sends test requests via TCP
func sendTestRequest(name string, id, tcnum int, tmodel ILModel) {
	// Create test request
	msg := message{tcnum, myaddr.String(), "server", "test_request", tmodel, gempty}
	// Send the request
	fmt.Printf("--> Sending test request from %v to %v.", cnumhist[tcnum], name)
	err := tcpSend(claddr[id], msg)
	if err != nil {
		fmt.Printf(" [NO!]\n*** Could not send test request to %v.\n", name)
	}
}

// Helper function that sends the global model to a requesting node
func sendGlobal(m message) {
	fmt.Printf("--> Sending global model to %v.", m.NodeName)
	msg := message{m.Id, myaddr.String(), "server", "global_grant", m.Model, gmodel}
	tcpSend(claddr[client[m.NodeName]], msg)
}

// Helper function for sending messages to nodes via TCP
func tcpSend(addr *net.TCPAddr, msg message) error {
	conn, err := net.DialTCP("tcp", nil, addr)
	if err == nil {
		enc := gob.NewEncoder(conn)
		dec := gob.NewDecoder(conn)
		err := enc.Encode(msg)
		checkError(err)
		var r response
		err = dec.Decode(&r)
		checkError(err)
		if r.Resp == "OK" {
			fmt.Printf(" [OK]\n")
		} else {
			fmt.Printf(" [%s]\n<-- Request was denied by node: %v.\nEnter command: ", r.Resp, r.Error)
		}
	}
	return err
}

// Helper function that checks the testqueue for outstanding tests
func checkQueue(id int) bool {
	for _, v := range testqueue[id] {
		if v {
			return false
		}
	}
	return true
}

// Helper function that processes join requests
func processJoin(m message) {
	// Process depending on if it is a new node or a returning one
	if _, ok := client[m.NodeName]; !ok {
		// Adding a node that has never been added before
		id := nodeNum
		nodeNum++
		client[m.NodeName] = id
		claddr[id], _ = net.ResolveTCPAddr("tcp", m.NodeIp)
		fmt.Printf("--- Added %v as node%v.\n", m.NodeName, id)

		// Creates test requests to be sent to joining node for committed models
		queue := make(map[int]bool)
		for k, _ := range tempmodels {
			queue[k] = true
		}
		testqueue[id] = queue

		// Sends test requests that were created above
		for _, v := range tempmodels {
			sendTestRequest(m.NodeName, id, v.commitNum, ILModel{v.model, 0.0, 0.0})
		}

	} else {
		// Node is rejoining, update address and resend the unfinished test requests
		id := client[m.NodeName]
		claddr[id], _ = net.ResolveTCPAddr("tcp", m.NodeIp)
		fmt.Printf("--- %v at node%v is back online.\n", m.NodeName, id)

		for k, v := range testqueue[id] {
			if v {
				sendTestRequest(m.NodeName, id, tempmodels[k].commitNum, ILModel{tempmodels[k].model, 0.0, 0.0})
			}
		}
	}
}

// Helper function that converts a python array of floats to a golang array of floats
func pyFloatArrayToGoFloatArray(pyFloats *python.PyObject) []float64 {
	pyFloatsByteArray := python.PyByteArray_FromObject(pyFloats)
	goFloatsByteArray := python.PyByteArray_AsBytes(pyFloatsByteArray)

	var goFloats []float64
	size := len(goFloatsByteArray) / 8

	for i := 0; i < size; i++ {
		currIndex := i * 8
		bits := binary.LittleEndian.Uint64(goFloatsByteArray[currIndex : currIndex+8])
		float := math.Float64frombits(bits)
		goFloats = append(goFloats, float)
	}

	return goFloats
}

// Helper function for input parsing
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
	logger = govec.InitGoVector(inputargs[1], inputargs[1])
}

// Helper function for error checking purposes
func checkError(err error) {
	if err != nil {
		fmt.Fprintf(os.Stderr, "Fatal error: %s", err.Error())
		//os.Exit(1)
	}
}
