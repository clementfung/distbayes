package main

import (
	"bufio"
	"fmt"
	"net"
	"net/rpc"
	"net/http"
	"os"
	"strings"
)

var ONION_HOST string = "4255ru2izmph3efw.onion"

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
)

func handler(w http.ResponseWriter, r *http.Request) {
    fmt.Fprintf(w, "Welcome %s!", r.URL.Path[1:])
    fmt.Fprintf(w, "Num nodes: %s", maxnode)
}

func httpHandler() {
	http.HandleFunc("/", handler)
	http.ListenAndServe(":8080", nil)

	fmt.Printf("HTTP initialized.\n")
}

func main() {

	///netSetup()
	fmt.Println("Launching server...")
	go httpHandler()

	fmt.Printf("Listening for TCP....\n")

	// listen on all interfaces
	ln, _ := net.Listen("tcp", ":6666")

	for {

	  	// accept connection on port
		conn, _ := ln.Accept()
		    
	    // will listen for message to process ending in newline (\n)
		message, _ := bufio.NewReader(conn).ReadString('\n')
	    // output message received
		fmt.Print("Message Received:", string(message))
	    // sample process for string received
		newmessage := strings.ToUpper(message)
	    // send new string back to client
		conn.Write([]byte(newmessage + "\n"))

		fmt.Println("It's done.")
		maxnode++

	}

    // Keeps server running
	select {}
}

func netSetup() {

	// Registering the server's remote procedure
	server := new(Server)
	rpc.Register(server)

	// Setting up RPC listener for server
	rpc.HandleHTTP()
	listener, err := net.Listen("tcp", "127.0.0.1:6666")
	checkError(err)
	go http.Serve(listener, nil)

}

// Remote procedure for handling client join/rejoin requests
func (t *Server) RequestJoin(args RegisterArgs, reply *int) error {
	
	fmt.Println("I heard a join")
	maxnode++
	*reply = 1 // indicates successful join

	return nil
}

// Error checking function
func checkError(err error) {
	if err != nil {
		fmt.Fprintf(os.Stderr, "Fatal error: %s", err.Error())
		os.Exit(1)
	}
}