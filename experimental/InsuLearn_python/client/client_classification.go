package main

import (
	"bufio"
	"encoding/gob"
	"flag"
	"fmt"
	"github.com/arcaneiceman/GoVector/govec"
	"github.com/sbinet/go-python"
	"net"
	"os"
)

var (
	name       string                  // Name of the local node (passed as argument)
	myaddr     *net.TCPAddr            // Client's IP address (passed as argument)
	svaddr     *net.TCPAddr            // Server's IP address (passed as argument)
	model      ILModel                 // Client variable holding the current local model
	logger     *govec.GoLog            // GoVector logger for debugging purposes (currently useless, useful logging still needs to be implemented in this system)
	trainset   string                  // Training set name (passed as argument)
	testset    string                  // Testing set name (passed as argument)
	l          *net.TCPListener        // Listens for messages from server (blocks until it receives a message)
	gmodel     ILGlobalModel           // Client variable holding the current global model
	gempty     ILGlobalModel           // Empty global model (sent to server because it doesn't make sense for the client to send to the server)
	isjoining  bool             = true // Used to determine whether the node has joined successfully yet or not (it will set this to false once it joins)
	modeltype  string                  // Determines the type of model that will be trained on the Python side
	client     *python.PyObject        // Client variable that holds reference to client_classification.py module
	read       *python.PyObject        // Client variable that holds reference to Read function in client_classification.py module
	trainlocal *python.PyObject        // Client variable that holds reference to Train function in client_classification.py module
	train      *python.PyObject        // Client variable that holds reference to TrainErrorLocal function in client_classification.py module
	valid      *python.PyObject        // Client variable that holds reference to TrainErrorGlobal function in client_classification.py module
	test       *python.PyObject        // Client variable that holds reference to TestErrorLocal function in client_classification.py module
	testg      *python.PyObject        // Client variable that holds reference to TestErrorGlobal function in client_classification.py module
)

// Local model struct
type ILModel struct {
	Model      string
	Size       float64
	LocalError float64
}

// Global model struct
// 'models' is an array pickled strings and weights is a normalized array of weights (weight at an index is the weight for the model at the same index in models slice))
type ILGlobalModel struct {
	Models  []string
	Weights []float64
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

	// Required so that client_classification.py can be imported directly as a module using relative paths
	sysPath := python.PySys_GetObject("path")
	python.PyList_Insert(sysPath, 0, python.PyString_FromString("./"))
	python.PyList_Insert(sysPath, 0, python.PyString_FromString("../python/code"))

	// Imports client_classification.py and gets the GenGlobal function as *python.PyObject
	client = python.PyImport_ImportModule("client_classification")
	read = client.GetAttrString("Read")
	trainlocal = client.GetAttrString("Train")
	train = client.GetAttrString("TrainErrorLocal")
	valid = client.GetAttrString("TrainErrorGlobal")
	test = client.GetAttrString("TestErrorLocal")
	testg = client.GetAttrString("TestErrorGlobal")
}

func main() {
	// Parsing inputargs
	parseArgs()

	// Initialize TCP Connection and listener
	l, _ = net.ListenTCP("tcp", myaddr)
	fmt.Printf("Node initialized as %v.\n", name)
	go listener()

	// Repeat until join is complete
	for isjoining {
		requestJoin()
	}

	// Main function of this server
	for {
		parseUserInput()
	}

}

// Listens for messages from the server and responds according to the message
func listener() {
	for {
		conn, err := l.AcceptTCP()
		checkError(err)
		go connHandler(conn)
	}
}

// Responds according to message sent by the server
func connHandler(conn *net.TCPConn) {
	var msg message
	enc := gob.NewEncoder(conn)
	dec := gob.NewDecoder(conn)
	err := dec.Decode(&msg)
	checkError(err)
	switch msg.Type {
	case "test_request":
		// Server is asking me to test
		enc.Encode(response{"OK", ""})
		go testModel(msg.Id, msg.Model)
	case "global_grant":
		// Server is sending global model
		enc.Encode(response{"OK", ""})
		gmodel = msg.GModel
		fmt.Printf("\n <-- Pulled global model from server.\nEnter command: ")
	default:
		// Respond to ping
		enc.Encode(response{"NO", "Unknown Command"})
	}
	conn.Close()
}

// Responds according to user input
func parseUserInput() {
	var ident string
	reader := bufio.NewReader(os.Stdin)
	fmt.Print("Enter command: ")
	text, _ := reader.ReadString('\n')
	// Windows adds its own strange carriage return, the following lines fix it
	if text[len(text)-2] == '\r' {
		ident = text[0 : len(text)-2]
	} else {
		ident = text[0 : len(text)-1]
	}
	switch ident {

	// Reads the training and test sets again on the Python side
	case "read":
		read.CallFunction(python.PyString_FromString(trainset), python.PyString_FromString(testset))

	// Trains the model on the Python side; the model, training error, and training set size are returned
	case "train":
		trainDict := trainlocal.CallFunction(python.PyString_FromString(modeltype))

		model.Model = python.PyString_AsString(python.PyDict_GetItem(trainDict, python.PyString_FromString("model")))
		model.LocalError = python.PyFloat_AsDouble(python.PyDict_GetItem(trainDict, python.PyString_FromString("error")))
		model.Size = python.PyFloat_AsDouble(python.PyDict_GetItem(trainDict, python.PyString_FromString("size")))

		fmt.Printf(" --- Local model error on local data is: %v.\n", model.LocalError)

	// Commits/pushes the local model to the server
	case "push":
		requestCommit()

	// Pulls the global model from the server
	case "pull":
		requestGlobal()

	// Validates the global model using the local training set on the Python side; the error is returned
	case "valid":
		models := python.PyList_New(len(gmodel.Models))
		for i := 0; i < len(gmodel.Models); i++ {
			python.PyList_SetItem(models, i, python.PyString_FromString(gmodel.Models[i]))
		}

		weights := python.PyList_New(len(gmodel.Weights))
		for i := 0; i < len(gmodel.Weights); i++ {
			python.PyList_SetItem(weights, i, python.PyFloat_FromDouble(gmodel.Weights[i]))
		}

		fmt.Printf(" --- Global model error on local data is: %v.\n", python.PyFloat_AsDouble(valid.CallFunction(models, weights)))

	// Tests the local model on the test set on the Python side; the error is returned
	case "test":
		testDict := test.CallFunction(python.PyString_FromString(model.Model))
		err := python.PyFloat_AsDouble(python.PyDict_GetItem(testDict, python.PyString_FromString("error")))

		fmt.Printf(" --- Local model error on test data is: %v.\n", err)

	// Tests the global model on the test set on the Python side; the error is returned
	case "testg":
		models := python.PyList_New(len(gmodel.Models))
		for i := 0; i < len(gmodel.Models); i++ {
			python.PyList_SetItem(models, i, python.PyString_FromString(gmodel.Models[i]))
		}

		weights := python.PyList_New(len(gmodel.Weights))
		for i := 0; i < len(gmodel.Weights); i++ {
			python.PyList_SetItem(weights, i, python.PyFloat_FromDouble(gmodel.Weights[i]))
		}

		fmt.Printf(" --- Global model error on test data is: %v.\n", python.PyFloat_AsDouble(testg.CallFunction(models, weights)))

	// Prints the client name
	case "who":
		fmt.Printf("%v\n", name)

	// User inputs an unsupported command
	default:
		fmt.Printf(" Command not recognized: %v.\n\n", ident)
		fmt.Printf("  Choose from the following commands\n")
		fmt.Printf("  read  -- Read data from disk\n")
		fmt.Printf("  push  -- Push trained model to server\n")
		fmt.Printf("  pull  -- Obtain global model from server\n")
		fmt.Printf("  train -- Train model from data (reports error)\n")
		fmt.Printf("  valid -- Validate global model with local data\n")
		fmt.Printf("  test  -- Test local model with test data\n")
		fmt.Printf("  testg -- Test global model with test data\n")
		fmt.Printf("  who   -- Print node name\n\n")
	}
}

// Helper function which is used to send join requests to the server
func requestJoin() {
	msg := message{0, myaddr.String(), name, "join_request", model, gempty}
	fmt.Printf(" --> Asking server to join.")
	tcpSend(msg)
}

// Helper function which is used to commit the local model to the server
func requestCommit() {
	msg := message{0, myaddr.String(), name, "commit_request", model, gempty}
	fmt.Printf(" --> Pushing local model to server.")
	tcpSend(msg)
}

// Helper function which is used to request the global model from the server
func requestGlobal() {
	msg := message{0, myaddr.String(), name, "global_request", model, gempty}
	fmt.Printf(" --> Requesting global model from server.")
	tcpSend(msg)
}

// Helper function which is used to test a model sent from the server and respond to the server with the results
func testModel(id int, testmodel ILModel) {
	fmt.Printf("\n <-- Received test requset.\nEnter command: ")
	testDict := train.CallFunction(python.PyString_FromString(testmodel.Model))
	testmodel.LocalError = python.PyFloat_AsDouble(python.PyDict_GetItem(testDict, python.PyString_FromString("error")))
	testmodel.Size = python.PyFloat_AsDouble(python.PyDict_GetItem(testDict, python.PyString_FromString("size")))
	msg := message{id, myaddr.String(), name, "test_complete", testmodel, gempty}
	fmt.Printf("\n --> Sending completed test requset.")
	tcpSend(msg)
	fmt.Printf("Enter command: ")
}

// Helper function for sending messages to nodes via TCP
func tcpSend(msg message) {
	conn, err := net.DialTCP("tcp", nil, svaddr)
	checkError(err)
	enc := gob.NewEncoder(conn)
	dec := gob.NewDecoder(conn)
	err = enc.Encode(&msg)
	checkError(err)
	var r response
	err = dec.Decode(&r)
	checkError(err)
	if r.Resp == "OK" {
		fmt.Printf(" [OK]\n")
		if r.Error == "Joined" {
			isjoining = false
		}
	} else if r.Resp == "NO" {
		fmt.Printf(" [%s]\n *** Request was denied by server: %v.\nEnter command: ", r.Resp, r.Error)
	} else {
		fmt.Printf(" [%s]\n *** Something strange Happened: %v.\nEnter command: ", r.Resp, r.Error)
	}
}

// Helper function for input parsing
func parseArgs() {
	flag.Parse()
	inputargs := flag.Args()
	var err error
	if len(inputargs) < 7 {
		fmt.Printf("Not enough inputs.\n")
		return
	}
	name = inputargs[0]
	myaddr, err = net.ResolveTCPAddr("tcp", inputargs[1])
	checkError(err)
	svaddr, err = net.ResolveTCPAddr("tcp", inputargs[2])
	checkError(err)
	trainset = inputargs[3]
	testset = inputargs[4]
	modeltype = inputargs[5]
	logger = govec.InitGoVector(inputargs[0], inputargs[6])
}

// Helper function for error checking purposes
func checkError(err error) {
	if err != nil {
		fmt.Fprintf(os.Stderr, "Fatal error: %s", err.Error())
		//os.Exit(1)
	}
}
