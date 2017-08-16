package main

import (
	"bytes"
	"encoding/gob"
	"flag"
	"fmt"
	"github.com/arcaneiceman/GoVector/govec"
	"github.com/coreos/etcd/raft"
	"github.com/coreos/etcd/raft/raftpb"
	"golang.org/x/net/context"
	"io/ioutil"
	"math"
	"math/rand"
	"net"
	"os"
	"strconv"
	"strings"
	"time"
)

const hb = 5

var (
	naddr     map[int]string
	logger    *govec.GoLog
	nID       int
	myaddr    *net.TCPAddr
	channel   chan message
	models    map[int]ServerModelWithState
	totalSize float64
	gempty    ILGlobalModel
	gmodel    ILGlobalModel
	mynode    *node
	server    *python.PyObject
	genGlobal *python.PyObject
)

type node struct {
	id         uint64
	ctx        context.Context
	store      *raft.MemoryStorage
	cfg        *raft.Config
	raft       raft.Node
	propID     map[int]bool
	nodeNum    int
	commitNum  int
	cnumhist   map[int]int
	client     map[string]int
	tempmodels map[int]ServerModelWithState
	testqueue  map[int]map[int]bool
	claddr     map[int]*net.TCPAddr
	ticker     <-chan time.Time
	done       <-chan struct{}
}

type state struct {
	PropID int
	Msg    message
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

// Local model struct
type ILModel struct {
	Model      string  // Pickled string representing an Python sklearn model
	Size       float64 // The number of rows the model was trained on
	LocalError float64 // The error of the model on the node's training data
}

// Server representation of local models
type ServerModelWithState struct {
	CommitNum int             // Commit number (used to map this model to the original node that created it)
	Model     string          // Pickled string representing an Python sklearn model
	Errors    map[int]float64 // Errors of this model on the different nodes (map/array structure)
	TotalSize float64         // Running sum of how many rows the model has been validated on
	TrainSize float64         // How many rows the model was originally trained on
}

// Global model struct
// 'models' is an array pickled strings and weights is a normalized array of weights (weight at an index is the weight for the model at the same index in models slice))
type ILGlobalModel struct {
	Models  []string
	Weights []float64
}

// Function to initialize a new Raft node
func newNode(id uint64, peers []raft.Peer) *node {
	store := raft.NewMemoryStorage()
	n := &node{
		id:    id,
		ctx:   context.TODO(),
		store: store,
		cfg: &raft.Config{
			ID:              id,
			ElectionTick:    10 * hb,
			HeartbeatTick:   1 * hb,
			Storage:         store,
			MaxSizePerMsg:   math.MaxUint16,
			MaxInflightMsgs: 1024,
		},
		propID:     make(map[int]bool),
		nodeNum:    0,
		commitNum:  0,
		client:     make(map[string]int),
		claddr:     make(map[int]*net.TCPAddr),
		tempmodels: make(map[int]ServerModelWithState),
		testqueue:  make(map[int]map[int]bool),
		cnumhist:   make(map[int]int),
		ticker:     time.Tick(time.Second / 10),
		done:       make(chan struct{}),
	}
	n.raft = raft.StartNode(n.cfg, peers)
	return n
}

// Raft state machine
func (n *node) run() {
	for {
		select {
		case <-n.ticker:
			n.raft.Tick()
		case rd := <-n.raft.Ready():
			n.saveToStorage(rd.HardState, rd.Entries, rd.Snapshot)
			n.send(rd.Messages)
			if !raft.IsEmptySnap(rd.Snapshot) {
				n.processSnapshot(rd.Snapshot)
			}
			for _, entry := range rd.CommittedEntries {
				n.process(entry)
				if entry.Type == raftpb.EntryConfChange {
					var cc raftpb.ConfChange
					cc.Unmarshal(entry.Data)
					n.raft.ApplyConfChange(cc)
				}
			}
			n.raft.Advance()
		case <-n.done:
			return
		}
	}
}

// Raft operations for saving snapshots
func (n *node) saveToStorage(hardState raftpb.HardState, entries []raftpb.Entry, snapshot raftpb.Snapshot) {
	n.store.Append(entries)
	if !raft.IsEmptyHardState(hardState) {
		n.store.SetHardState(hardState)
	}
	if !raft.IsEmptySnap(snapshot) {
		n.store.ApplySnapshot(snapshot)
	}
}

// Raft send function using gob encoder [between Raft nodes]
func (n *node) send(messages []raftpb.Message) {
	for _, m := range messages {
		conn, err := net.Dial("tcp", naddr[int(m.To)])
		if err == nil {
			enc := gob.NewEncoder(conn)
			enc.Encode(m)
		} else {
			fmt.Printf("*** Could not send message to Raft node: %v.\n", int(m.To))
		}
	}
}

// Raft receive function using gob encoder [between Raft nodes]
func (n *node) receive(conn *net.TCPConn) {
	// Echo all incoming data.
	var imsg raftpb.Message
	dec := gob.NewDecoder(conn)
	err := dec.Decode(&imsg)
	checkError(err)
	conn.Close()
	n.raft.Step(n.ctx, imsg)
}

// Raft function for loading snapshots [not implemented]
func (n *node) processSnapshot(snapshot raftpb.Snapshot) {
	panic(fmt.Sprintf("Applying snapshot on node %v is not implemented", n.id))
}

// Raft process functions, idempotent modifications to the key-value stores and join/commit history updates
func (n *node) process(entry raftpb.Entry) {
	if entry.Type == raftpb.EntryNormal && entry.Data != nil {
		var repstate state
		var buf bytes.Buffer
		dec := gob.NewDecoder(&buf)
		buf.Write(entry.Data)
		dec.Decode(&repstate)
		msg := repstate.Msg
		switch msg.Type {

		case "join_request":
			id := n.nodeNum
			n.nodeNum++
			n.client[msg.NodeName] = id
			n.claddr[id], _ = net.ResolveTCPAddr("tcp", msg.NodeIp)
			queue := make(map[int]bool)
			for k, _ := range n.tempmodels {
				queue[k] = true
			}
			n.testqueue[id] = queue
			fmt.Printf("--- Added %v as node%v.\n", msg.NodeName, id)

		case "rejoin_request":
			id := n.client[msg.NodeName]
			n.claddr[id], _ = net.ResolveTCPAddr("tcp", msg.NodeIp)

		case "commit_request":
			tempcnum := n.commitNum
			n.commitNum++
			n.cnumhist[tempcnum] = n.client[msg.NodeName]

			tempweight := make(map[int]float64)
			tempweight[n.client[msg.NodeName]] = msg.Model.LocalError
			n.tempmodels[n.client[msg.NodeName]] = ServerModelWithState{tempcnum, msg.Model.Model, tempweight, msg.Model.Size, msg.Model.Size}

			for _, id := range n.client {
				if id != n.client[msg.NodeName] {
					if queue, ok := n.testqueue[id]; !ok {
						queue := make(map[int]bool)
						queue[n.cnumhist[tempcnum]] = true
						n.testqueue[id] = queue
					} else {
						queue[n.cnumhist[tempcnum]] = true
					}
				}
			}
			fmt.Printf("--- Processed commit %v for node %v.\n", tempcnum, msg.NodeName)

		case "test_complete":

			n.testqueue[n.client[msg.NodeName]][n.cnumhist[msg.Id]] = false
			channel <- msg

		default:
			// Do nothing
		}
		n.propID[repstate.PropID] = true
	}
}

// Replicate function that coordinates Raft nodes and blocks until all nodes have been synced
//   The map used to check commit grows as more replication proccesses are done and will start to
//   cause problems when the number of commits increase past a certain amount. It would be wise to
//   replace this with a better alternative.
// TODO: Replace this
func replicate(m state) bool {
	flag := false

	r := rand.Intn(999999999999)
	m.PropID = r
	var buf bytes.Buffer
	enc := gob.NewEncoder(&buf)
	err := enc.Encode(m)
	err = mynode.raft.Propose(mynode.ctx, buf.Bytes())

	if err == nil {
		//block and check the status of the proposal
		for !flag {
			time.Sleep(time.Duration(1 * time.Second))
			if mynode.propID[r] {
				flag = true
			}
		}
	}

	return flag
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
	//Parsing inputargs
	parseArgs()
	raftaddr, _ := net.ResolveTCPAddr("tcp", naddr[nID])
	sl, err := net.ListenTCP("tcp", raftaddr)
	checkError(err)
	cl, err := net.ListenTCP("tcp", myaddr)
	checkError(err)

	// Wait for proposed entry to be commited in cluster.
	// Apperently when should add an uniq id to the message and wait until it is
	// commited in the node.
	models = make(map[int]ServerModelWithState)
	totalSize = 0.0
	channel = make(chan message)
	// start a cluster R = 7
	//mynode = newNode(uint64(nID), []raft.Peer{{ID: 1}, {ID: 2}, {ID: 3}, {ID: 4}, {ID: 5}, {ID: 6}, {ID: 7}})
	// start a cluster R = 5
	mynode = newNode(uint64(nID), []raft.Peer{{ID: 1}, {ID: 2}, {ID: 3}, {ID: 4}, {ID: 5}})

	go mynode.run()

	if nID == 1 {
		time.Sleep(time.Duration(5 * time.Second))
		mynode.raft.Campaign(mynode.ctx)
	}

	//Initialize TCP Connection and listenerfmt.Printf("Server initialized.\n")

	go updateGlobal(channel)

	go clientListener(cl)

	go printLeader()

	for {
		conn, err := sl.AcceptTCP()
		if err == nil {
			go mynode.receive(conn)
		} else {
			fmt.Printf("*** Could not accept connection from Raft node: %s.", err)
		}
	}

}

// Function to periodically print Raft Leader
func printLeader() {
	for {
		time.Sleep(time.Duration(5 * time.Second))
		sts := mynode.raft.Status()
		fmt.Printf("--- Current leader is %v.\n", sts.Lead)
	}
}

// Client listener function
func clientListener(listen *net.TCPListener) {
	for {
		connC, err := listen.AcceptTCP()
		if err == nil {
			go connHandler(connC)
		} else {
			fmt.Printf("*** Could not accept connection from Client node: %s.", err)
		}
	}
}

// Function for handling client requests and replicating on Raft if necessary
func connHandler(conn *net.TCPConn) {
	var msg message
	dec := gob.NewDecoder(conn)
	enc := gob.NewEncoder(conn)
	err := dec.Decode(&msg)
	checkError(err)
	switch msg.Type {

	case "commit_request":
		// node is sending a model, checking to see if testing is complete
		flag := checkQueue(mynode.client[msg.NodeName])
		fmt.Printf("<-- Received commit request from %v.\n", msg.NodeName)
		if flag {
			// accept commit from node and process outgoing test requests
			processTestRequest(msg, conn)
		} else {
			// deny commit request
			enc.Encode(response{"NO", "Pending tests are not complete"})
			fmt.Printf("--> Denied commit request from %v.\n", msg.NodeName)
			conn.Close()
		}

	case "global_request":
		// node is requesting the global model -> generate and forward
		enc.Encode(response{"OK", ""})
		fmt.Printf("<-- Received global model request from %v.\n", msg.NodeName)
		genGlobalModel()
		sendGlobal(msg)
		conn.Close()

	case "test_complete":
		// node is submitting test results, update testqueue on all replicas
		fmt.Printf("<-- Received completed test results from %v.\n", msg.NodeName)

		// Check to see if the incoming test results are for an outdated model
		notOutdated := true
		for i := msg.Id + 1; i < len(mynode.cnumhist); i++ {
			if mynode.cnumhist[i] == mynode.cnumhist[msg.Id] {
				notOutdated = false
				break
			}
		}

		if notOutdated & mynode.testqueue[mynode.client[msg.NodeName]][mynode.cnumhist[msg.Id]] {
			repstate := state{0, msg}
			flag := replicate(repstate)
			if flag {
				enc.Encode(response{"OK", "Test Processed"})
			} else {
				// if testqueue could not be replicated
				enc.Encode(response{"NO", "Try Again"})
			}
		} else {
			// if testqueue is already empty
			enc.Encode(response{"NO", "Duplicate Test"})
			fmt.Printf("--> Ignored test results from %v.\n", msg.NodeName)
		}
		conn.Close()

	case "join_request":
		// node is requesting to join or rejoin
		fmt.Printf("<-- Received join request from %v.\n", msg.NodeName)
		flag := processJoin(msg)
		if flag {
			enc.Encode(response{"OK", "Joined"})
		} else {
			fmt.Printf("*** Could not process join for node %v.\n", msg.NodeName)
			enc.Encode(response{"NO", "Failed Join"})
		}
		conn.Close()

	default:
		fmt.Printf("something weird happened!\n")
		enc.Encode(response{"NO", "Unknown Request"})
		conn.Close()
	}
}

// Global model update function
func updateGlobal(ch chan message) {
	// Function that aggregates the global model and commits when ready
	for {
		m := <-ch
		id := mynode.cnumhist[m.Id]

		tempmodel := mynode.tempmodels[id]
		tempmodel.TotalSize += m.Model.Size

		// TODO: Discuss why we're using += instead of = with Al
		tempmodel.Errors[mynode.client[m.NodeName]] += m.Model.LocalError
		//tempAggregate.R[mynode.client[m.NodeName]] += m.Model.Weight

		mynode.tempmodels[id] = tempmodel

		// This checks whether the model has been validated on more rows of data than the current server maximum and updates accordingly
		if totalSize < tempmodel.TotalSize {
			totalSize = tempmodel.TotalSize
		}

		// If the model being updated has been validated on enough data or not (if so, it will be added to the global model the next time the global model is generated)
		if float64(tempmodel.TotalSize) > float64(totalSize)*0.6 {
			models[id] = tempmodel
			t := time.Now()
			logger.LogLocalEvent(fmt.Sprintf("%s - Committed model%v by %v at partial commit %v.", t.Format("15:04:05.0000"), id, mynode.client[m.NodeName], tempAggregate.D/modelD*100.0))
			//logger.LogLocalEvent(fmt.Sprintf("%v %v %v", t, id, tempAggregate.d))
			fmt.Printf("--- Committed model%v for commit number: %v.\n", id, tempAggregate.Cnum)
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

// Function that generates test request following a commit request
func processTestRequest(m message, conn *net.TCPConn) {
	repstate := state{0, m}
	flag := replicate(repstate)
	enc := gob.NewEncoder(conn)
	if flag {
		//sanitize the model for testing
		m.Model.LocalError = 0.0
		m.Model.Size = 0.0
		tempcnum := 0
		//get the latest cnum, necessary as cnum is updated in raft
		for k, v := range mynode.cnumhist {
			if v == mynode.client[m.NodeName] {
				tempcnum = k
			}
		}
		enc.Encode(response{"OK", "Committed"})
		conn.Close()
		for name, id := range mynode.client {
			if id != mynode.client[m.NodeName] {
				sendTestRequest(name, id, tempcnum, m.Model)
			}
		}
	} else {
		enc.Encode(response{"NO", "Try Again!"})
		conn.Close()
		fmt.Printf("--> Failed to commit request from %v.\n", m.NodeName)
	}
}

// Function that sends test requests via TCP
func sendTestRequest(name string, id, tcnum int, tmodel ILModel) {
	//create test request
	msg := message{tcnum, "server", "server", "test_request", tmodel, gempty}
	//send the request
	fmt.Printf("--> Sending test request from %v to %v.", mynode.cnumhist[tcnum], name)
	err := tcpSend(mynode.claddr[id], msg)
	if err != nil {
		fmt.Printf(" [NO!]\n*** Could not send test request to %v.\n", name)
	}
}

// Function to forward global model
func sendGlobal(m message) {
	fmt.Printf("--> Sending global model to %v.", m.NodeName)
	msg := message{m.Id, myaddr.String(), "server", "global_grant", m.Model, gmodel}
	tcpSend(mynode.claddr[mynode.client[m.NodeName]], msg)
}

// Function for sending messages to nodes via TCP
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
	for _, v := range mynode.testqueue[id] {
		if v {
			return false
		}
	}
	return true
}

// Function that processes join requests and forwards response to Raft nodes
func processJoin(m message) bool {
	//process depending on if it is a new node or a returning one
	flag := false

	if id, ok := mynode.client[m.NodeName]; !ok {
		//adding a node that has never been added before
		repstate := state{0, m}
		flag = replicate(repstate)
		if flag {
			for _, v := range mynode.tempmodels {
				sendTestRequest(m.NodeName, mynode.client[m.NodeName], v.CommitNum, v.Model)
			}
		}
	} else {
		//node is rejoining, update address and resend the unfinished test requests
		m.Type = "rejoin_request"
		repstate := state{0, m}
		flag = replicate(repstate)
		if flag {
			//time.Sleep(time.Duration(2 * time.Second))
			for k, v := range mynode.testqueue[id] {
				if v {
					aggregate := mynode.tempmodels[k]
					sendTestRequest(m.NodeName, id, aggregate.CommitNum, aggregate.Model)
				}
			}
		}
	}
	return flag
}

// Input parser
func parseArgs() {
	naddr = make(map[int]string)
	flag.Parse()
	inputargs := flag.Args()
	var err error
	if len(inputargs) < 2 {
		fmt.Printf("Not enough inputs.\n")
		return
	}
	myaddr, err = net.ResolveTCPAddr("tcp", inputargs[0])
	checkError(err)
	logger = govec.Initialize(inputargs[3], inputargs[3])
	getNodeAddr(inputargs[1])
	temp, _ := strconv.ParseInt(inputargs[2], 10, 64)
	nID = int(temp)
}

// Helper function for reading server addresses from file
func getNodeAddr(slavefile string) {
	dat, err := ioutil.ReadFile(slavefile)
	checkError(err)
	nodestr := strings.Split(string(dat), " ")
	for i := 0; i < len(nodestr)-1; i++ {
		naddr[i+1] = nodestr[i]
	}
}

// Helper function for error checking purposes
func checkError(err error) {
	if err != nil {
		fmt.Fprintf(os.Stderr, "Fatal error: %s", err.Error())
		//os.Exit(1)
	}
}
