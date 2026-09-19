package main

import (
	"encoding/json"
	"fmt"
	"net/http"

	"github.com/gin-gonic/gin"
	"github.com/gorilla/websocket"
)

var upgrader = websocket.Upgrader{
	CheckOrigin: func(r *http.Request) bool {
		return true // localhost only for now
	},
}

type wSMessage struct {
	Type      string `json:"type"`
	Text      string `json:"text"`
	Watermark bool   `json:"watermark"`
}

func main() {

	err := InitModel()
	if err != nil {
		panic(err)
	}

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

	TokenChan = make(chan string, 32)

	// Read messages from browser.
	go func() {

		for {
			_, msg, err := conn.ReadMessage()
			if err != nil {
				return
			}

			var message wSMessage

			err = json.Unmarshal(msg, &message)
			if err != nil {
				println("Invalid JSON")
				continue
			}

			fmt.Println("Received message:", message.Text, "Watermark:", message.Watermark, "Type:", message.Type)

			StartGenerationwithParams(message.Text, message.Watermark, TokenChan)

		}

	}()

	// Send generated tokens to browser.
	for token := range TokenChan {

		// fmt.Println("Token Before Write: ",token)
		err := conn.WriteMessage(
			websocket.TextMessage,
			[]byte(token),
		)

		if err != nil {
			return
		}
	}
}

// {
//     "type": "something",
//     "text": "Hello Model, how are you doing today?"
//     "watermark": true
// }
