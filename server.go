package main

import (
	"net/http"

	"github.com/gin-gonic/gin"
	"github.com/gorilla/websocket"
)

// var tokenChan chan string

var upgrader = websocket.Upgrader{
	CheckOrigin: func(r *http.Request) bool {
		return true // localhost only for now
	},
}

func main() {

	// Initialize your token channel.
	// Replace 4096 with your llama context size later.
	tokenChan = make(chan string, 4096)

	router := gin.Default()

	router.GET("/ws", websocketHandler)

	router.Run(":8080")
}

func websocketHandler(c *gin.Context) {

	conn, err := upgrader.Upgrade(c.Writer, c.Request, nil)
	if err != nil {
		return
	}
	defer conn.Close()

	// Read messages from browser.
	go func() {

		for {
			_, msg, err := conn.ReadMessage()
			if err != nil {
				return
			}

			println("Received:", string(msg))

			// TODO:
			// Parse message and start/stop generation.
		}

	}()

	// Send generated tokens to browser.
	for token := range tokenChan {

		err := conn.WriteMessage(
			websocket.TextMessage,
			[]byte(token),
		)

		if err != nil {
			return
		}
	}
}