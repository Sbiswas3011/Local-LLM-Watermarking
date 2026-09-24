package main

/*
#cgo CFLAGS: -IC:/Users/JAYANTA/Desktop/llamaClone/llama.cpp/include -IC:/Users/JAYANTA/Desktop/llamaClone/llama.cpp/ggml/include
#cgo LDFLAGS: -L"C:/Users/JAYANTA/Desktop/llamaClone/llama.cpp/build/src/Release" -lllama
#include "llama.h"
#include <stdlib.h>
static void silent_log_callback(
    enum ggml_log_level level,
    const char * text,
    void * user_data) {
    (void) level;
    (void) text;
    (void) user_data;
}
static void disable_llama_logs(void) {
    llama_log_set(silent_log_callback, NULL);
}
*/
import "C"

import (
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strconv"

	"github.com/gin-gonic/gin"
	"github.com/gorilla/websocket"
)

var upgrader = websocket.Upgrader{
	CheckOrigin: func(r *http.Request) bool {
		return true // localhost only for now
	},
}

type wSMessage struct {
	Type        string  `json:"type"`
	Text        string  `json:"text"`
	Watermark   bool    `json:"watermark"`
	HistorySize int     `json:"history_size"`
	Seed        string  `json:"seed"`
	Gamma       float64 `json:"gamma"`
	LogitBias   float64 `json:"logit_bias"`
}

type ProcessRequest struct {
	Text        string  `json:"text"`
	Seed        string  `json:"seed"`
	Gamma       float64 `json:"gamma"`
	HistorySize int     `json:"history_size"`
}

type Server struct {
	Data     PromptData
	Sessions map[string]*Session
}

type Session struct {
	ID         string
	Watermark  *bool
	Ctx        *C.struct_llama_context
	Smpl       *C.struct_llama_sampler
	// ResultChan chan TokenResult
}

func (s *Server) getOrCreateSession(id string) (*Session, error) {

	// s.SessionsMu.RLock()
	session, exists := s.Sessions[id]
	// s.SessionsMu.RUnlock()

	if exists {
		return session, nil
	}

	ctx := C.llama_init_from_model(Model, Ctx_params)
	if ctx == nil {
		return nil, fmt.Errorf("Faild to create context")
	}

	// smpl := C.llama_sampler_chain_init(C.llama_sampler_chain_default_params())
	// if smpl == nil {
	// 	return nil, fmt.Errorf("Faild to create sampler")
	// }

	// Create context + sampler here
	session = &Session{
		ID:  id,
		Ctx: ctx,
		// Smpl:       smpl,
		// ResultChan: make(chan TokenResult, 32),
	}

	// s.SessionsMu.Lock()

	// Check again because another request could
	// have created it while we were creating ours.
	// existing, exists := s.Sessions[id]

	// if exists {
	//     s.SessionsMu.Unlock()

	//     // Don't leak the context/sampler we just created.
	//     C.llama_free(session.Ctx)
	//     C.llama_sampler_free(session.Smpl)

	//     return existing
	// }

	s.Sessions[id] = session
	// s.SessionsMu.Unlock()

	return session, nil
}

func main() {

	Data, err := InitModel()
	if err != nil {
		panic(err)
	}

	server := &Server{
		Data:     Data,
		Sessions: make(map[string]*Session),
	}

	router := gin.Default()

	router.GET("/", func(c *gin.Context) { c.File("./index.html") })
	router.GET("/ws", server.websocketHandler)
	router.POST("/process", server.processTextHandler)

	router.Run(":8080")
}

func (s *Server) websocketHandler(c *gin.Context) {


	conn, err := upgrader.Upgrade(c.Writer, c.Request, nil)
	if err != nil {
		return
	}
	defer conn.Close()

	sessionID := c.Query("session_id")
	session, err := s.getOrCreateSession(sessionID)
	if err != nil || session == nil {
		println("Failed to Create Session")
		return
	}

	// TokenChan = make(chan string, 32)
	// TokenIDChan = make(chan C.llama_token, 32)
	ResultChan := make(chan TokenResult, 32)

	// Read messages from browser.
	go func() {

		for {
			_, msg, err := conn.ReadMessage()
			if err != nil {
				println(err)
				return
			}

			var message wSMessage

			err = json.Unmarshal(msg, &message)
			if err != nil {
				println("Invalid JSON")
				continue
			}

			fmt.Println("Received message:", message.Text, "Watermark:", message.Watermark, "Type:", message.Type)

			if session.Watermark != nil && session.Watermark == &message.Watermark {
				s.Data.changeWaterMarkStatus = false
			} else {
				smpl := C.llama_sampler_chain_init(C.llama_sampler_chain_default_params())
				if smpl == nil {
					print("Failed to create sampler")
				}
				session.Smpl = smpl
				s.Data.changeWaterMarkStatus = true
			}

			s.Data.enableWatermark = message.Watermark
			s.Data.prompt = message.Text
			s.Data.ResultChan = ResultChan
			s.Data.seed = message.Seed
			s.Data.gamma = message.Gamma
			s.Data.logitbias = message.LogitBias
			s.Data.historySize = message.HistorySize
			s.Data.ctx = session.Ctx
			s.Data.smpl = session.Smpl

			generationOver, err := StartGenerationwithParams(s.Data)

			if err != nil {
				println("Error Occured During Generation", err)
				break
			} else if generationOver {
				break
			}

		}

	}()

	// Send generated tokens to browser.
	for token := range ResultChan {

		data, err := json.Marshal(token)
		if err != nil {
			return
		}

		// fmt.Println("Token Before Write: ",token)
		err = conn.WriteMessage(
			websocket.TextMessage,
			data,
		)
		if err != nil {
			return
		}
	}

}

func (s *Server) processTextHandler(c *gin.Context) {

	var request ProcessRequest
	var text string

	switch c.ContentType() {

	case "application/json":

		if err := c.ShouldBindJSON(&request); err != nil {
			c.JSON(http.StatusBadRequest, gin.H{
				"error": "invalid JSON",
			})
			return
		}

		text = request.Text

	case "multipart/form-data":

		request.Seed = c.PostForm("seed")

		gamma, err := strconv.ParseFloat(c.PostForm("gamma"), 64)
		if err != nil {
			c.JSON(http.StatusBadRequest, gin.H{
				"error": "invalid gamma",
			})
			return
		}
		request.Gamma = gamma

		historySize, err := strconv.Atoi(c.PostForm("history_size"))
		if err != nil {
			c.JSON(http.StatusBadRequest, gin.H{
				"error": "invalid history_size",
			})
			return
		}
		request.HistorySize = historySize

		file, err := c.FormFile("file")
		if err != nil {
			c.JSON(http.StatusBadRequest, gin.H{
				"error": "file is required",
			})
			return
		}

		src, err := file.Open()
		if err != nil {
			c.JSON(http.StatusInternalServerError, gin.H{
				"error": "failed to open file",
			})
			return
		}
		defer src.Close()

		data, err := io.ReadAll(src)
		if err != nil {
			c.JSON(http.StatusInternalServerError, gin.H{
				"error": "failed to read file",
			})
			return
		}

		request.Text = string(data)

	default:

		c.JSON(http.StatusUnsupportedMediaType, gin.H{
			"error": "use application/json or multipart/form-data",
		})
		return
	}

	// Common validation
	if text == "" {
		c.JSON(http.StatusBadRequest, gin.H{
			"error": "text or file is required",
		})
		return
	}

	if request.Gamma <= 0 || request.Gamma >= 1 {
		c.JSON(http.StatusBadRequest, gin.H{
			"error": "gamma must be between 0 and 1",
		})
		return
	}

	// Same processing regardless of input format
	totalCnt, totalGreenCnt, zscore, err := ProcessText(request, s.Data)

	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{
			"error": err.Error(),
		})
		return
	}

	c.JSON(http.StatusOK, gin.H{
		"total_count": totalCnt,
		"green_count": totalGreenCnt,
		"z_score":     zscore,
		"tokens_spent": "invalid",
		"total_avaialble_tokens": "invalid",
	})
}

// {
//     "type": "something",
//     "text": "Tell me a 50 line story about harry potter, it is very important that you think before generating.",
//     "watermark": true,
// 	"history_size": 4,
//     "seed": "i_am_a_llm",
//     "gamma": 0.6,
//     "logit_bias": 4.0,
// }

// handle watermark sample swithcing
//handle context cancelling and resetting
// for non watermaring ignore variables sent
//handle context exceeded
//handle tokenhistory sampling