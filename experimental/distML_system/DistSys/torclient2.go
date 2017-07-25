package main

import (
	"bufio"
	"fmt"
	"os"
	"golang.org/x/net/proxy"
)

/*
	An attempt at Tor via TCP. 
*/
var ONION_HOST string = "4255ru2izmph3efw.onion:6666"

func main() {

	// Create proxy dialer using Tor SOCKS proxy
	fmt.Println("Trying 9150")
	torDialer, err := proxy.SOCKS5("tcp", "127.0.0.1:9150", nil, proxy.Direct)
	checkError(err)

	fmt.Println("SOCKS5 Dial success!")

  	// connect to this socket
  	conn, err := torDialer.Dial("tcp", ONION_HOST)
  	checkError(err)

  	fmt.Println("TOR Dial Success!")

  	gotOne := false

  	for !gotOne { 
    	
    	// read in input from stdin
    	fmt.Print("Text to send: ")
    	text := "a message that I send"
    	
    	// send to socket
    	fmt.Fprintf(conn, text + "\n")
    	
    	// listen for reply
    	message, _ := bufio.NewReader(conn).ReadString('\n')
    	fmt.Print("Message from server: "+message)

    	gotOne = true

  	}

  	fmt.Println("The end")

}

// Error checking function
func checkError(err error) {
	if err != nil {
		fmt.Fprintf(os.Stderr, "Fatal error: %s", err.Error())
		os.Exit(1)
	}
}
