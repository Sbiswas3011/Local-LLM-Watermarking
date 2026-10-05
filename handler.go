package main

/*
#cgo linux CFLAGS: -I/app/llama.cpp/include -I/app/llama.cpp/ggml/include
#cgo linux LDFLAGS: -L/app/llama.cpp/build/bin -l:libllama.so

#cgo windows,amd64 CFLAGS: -IC:/Users/JAYANTA/Desktop/LLM_work/LlamaFork/llama.cpp/include -IC:/Users/JAYANTA/Desktop/LLM_work/LlamaFork/llama.cpp/ggml/include
#cgo windows,amd64 LDFLAGS: -LC:/Users/JAYANTA/Desktop/LLM_work/LlamaFork/llama.cpp/build/src/Release -l:llama.lib

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

	"github.com/gin-contrib/cors"
	"github.com/gin-gonic/gin"
	"github.com/gorilla/websocket"
)

var upgrader = websocket.Upgrader{
	CheckOrigin: func(r *http.Request) bool {
		return true // localhost only for now
	},
}

type wSMessage struct {
	Type          string  `json:"type"`
	Text          string  `json:"text"`
	Watermark     bool    `json:"watermark"`
	HistorySize   int     `json:"history_size"`
	Seed          string  `json:"seed"`
	Gamma         float64 `json:"gamma"`
	LogitBias     float64 `json:"logit_bias"`
	ResetSampler  bool    `json:"reset_sampler"`
	WatermarkType string  `json:"watermark_type"`
	Keys          []int64 `json:"keys"`
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
	ID               string
	Watermark        *bool
	Ctx              *C.struct_llama_context
	Smpl             *C.struct_llama_sampler
	ResultChan       chan TokenResult
	Data             PromptData
	CloseResultChan  chan bool
	TokenSpent       int
	TokenTotal       int
	TokensPerSec     float64
	WeightedMean     float64
	Zscore           float64
	InternalMessages []InternalMessage
}

type InternalMessage struct {
	Typ               string
	Role              string
	Data              []string
	Greensplit        []bool
	Thinkdata         []string
	Thinkgreensplit   []bool
	ThinkWeightedMean []float64
	DataWeightedMean  []float64
	Watermarked       bool
	Logitbias         float64
	Gamma             float64
	Seed              string
	HistorySize       int
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
	router.Use(cors.Default())

	router.GET("/", func(c *gin.Context) { c.File("./index.html") })
	router.GET("/ping",server.ping)
	router.GET("/ws", server.websocketHandler)
	router.GET("/getsession", server.getSession)
	router.POST("/process", server.processTextHandler)
	router.GET("/resetctx", server.resetContext)
	router.GET("/resetmsgs", server.resetMessages)
	router.GET("/closechan", server.closeResultChan)
	router.Run(":8080")
}

func (s *Server) ping(c *gin.Context){
	c.JSON(http.StatusOK, gin.H{
        "status": "ok",
    })
}

func (s *Server) getOrCreateSession(id string) (*Session, bool, error) {

	session, exists := s.Sessions[id]
	if exists {
		return session, true, nil
	}

	ctx := C.llama_init_from_model(Model, Ctx_params)
	if ctx == nil {
		return nil, false, fmt.Errorf("Faild to create context")
	}

	smpl := C.llama_sampler_chain_init(C.llama_sampler_chain_default_params())
	if smpl == nil {
		return nil, false, fmt.Errorf("Faild to create sampler")
	}

	// Create context + sampler here
	session = &Session{
		ID:   id,
		Ctx:  ctx,
		Smpl: smpl,
		Data: PromptData{
			model:   s.Data.model,
			vocab:   s.Data.vocab,
			n_vocab: s.Data.n_vocab,
			keys: []int64{1,2,3,4},
			seed: "i_am_a_llm",
			historySize: 4,
			logitbias: 2,
			gamma: 0.6,
			watermarkType: "RedGreen",
			enableWatermark: false,

		},
		// ResultChan: make(chan TokenResult, 32),
		// CloseResultChan: make(chan bool, 1),
	}

	s.Sessions[id] = session

	return session, false, nil
}

func (s *Server) resetContext(c *gin.Context) {

	sessionID := c.Query("session_id")
	session, lookupExists, err := s.getOrCreateSession(sessionID)
	if err != nil {
		c.JSON(http.StatusBadRequest, gin.H{
			"error": "Failed to Create Session",
		})
		return
	}

	if lookupExists {

		oldctx := session.Ctx
		if oldctx != nil {
			C.llama_free(oldctx)
		}
		newctx := C.llama_init_from_model(Model, Ctx_params)
		if newctx == nil {
			c.JSON(http.StatusBadRequest, gin.H{
				"error": "Failed to Create Context",
			})
			return
		}
		session.Ctx = newctx

		c.JSON(http.StatusOK, gin.H{
			"success": "Context Was Reset",
		})

		return
	}

	c.JSON(http.StatusOK, gin.H{
		"success": "New Context Was Created",
	})
}

func (s *Server) resetMessages(c *gin.Context) {

	sessionID := c.Query("session_id")
	session, lookupExists, err := s.getOrCreateSession(sessionID)
	if err != nil {
		c.JSON(http.StatusBadRequest, gin.H{
			"error": "Failed to Get/Create Session",
		})
		return
	}

	if lookupExists {
		newctx := C.llama_init_from_model(Model, Ctx_params)
		if newctx == nil {
			c.JSON(http.StatusBadRequest, gin.H{
				"error": "Failed to Create Context",
			})
			return
		}
		session.InternalMessages = []InternalMessage{}

		c.JSON(http.StatusOK, gin.H{
			"success": "Messages Were Reset",
		})

		return
	}

	c.JSON(http.StatusOK, gin.H{
		"success": "New Context Was Created",
	})
}

func (s *Server) closeResultChan(c *gin.Context) {
	sessionID := c.Query("session_id")

	session, _, err := s.getOrCreateSession(sessionID)
	if err != nil || session == nil {
		c.JSON(http.StatusNotFound, gin.H{
			"error": "Session not found",
		})
		return
	}

	if session.CloseResultChan == nil {
		c.JSON(http.StatusNotFound, gin.H{
			"error": "Close channel not found",
		})
		return
	}

	select {
	case session.CloseResultChan <- false:
		c.JSON(http.StatusOK, gin.H{
			"message": "Close signal sent",
		})
	default:
		c.JSON(http.StatusConflict, gin.H{
			"message": "Close signal already pending",
		})
	}
}

func (s *Server) websocketHandler(c *gin.Context) {

	conn, err := upgrader.Upgrade(c.Writer, c.Request, nil)
	if err != nil {
		return
	}
	defer conn.Close()

	sessionID := c.Query("session_id")
	fmt.Println("sessionID", sessionID)
	session, _, err := s.getOrCreateSession(sessionID)
	if err != nil || session == nil {
		println("Failed to Create Session")
		return
	}

	// TokenChan = make(chan string, 32)
	// TokenIDChan = make(chan C.llama_token, 32)
	ResultChan := make(chan TokenResult, 10)
	CloseResultChan := make(chan bool, 1)
	session.ResultChan = ResultChan
	session.CloseResultChan = CloseResultChan

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

			if message.ResetSampler {
				oldSmpl := session.Smpl
				if oldSmpl != nil {
					C.llama_sampler_free(oldSmpl)
					session.Smpl = nil
				}
				smpl := C.llama_sampler_chain_init(C.llama_sampler_chain_default_params())
				if smpl == nil {
					print("Failed to create sampler")
				}
				session.Smpl = smpl
				session.Data.resetSampler = true
				session.Data.enableWatermark = message.Watermark
				session.Data.watermarkType = message.WatermarkType
				session.Data.seed = message.Seed
				session.Data.gamma = message.Gamma
				session.Data.logitbias = message.LogitBias
				session.Data.historySize = message.HistorySize
				session.Data.keys = message.Keys
				session.Data.weightedMean = 0.0
			} else {
				session.Data.resetSampler = false
			}

			// session.Smpl = smpl
			// session.Data.enableWatermark = message.Watermark
			session.Data.prompt = message.Text
			session.Data.ResultChan = ResultChan
			session.Data.CloseResultChan = CloseResultChan
			// session.Data.seed = message.Seed
			// session.Data.gamma = message.Gamma
			// session.Data.logitbias = message.LogitBias
			// session.Data.historySize = message.HistorySize
			session.Data.ctx = session.Ctx
			session.Data.smpl = session.Smpl

			generationOver, err := StartGenerationwithParams(session, session.Data)

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

		// fmt.Println("writing to socket", data)

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

func (s *Server) getSession(c *gin.Context) {

	sessionID := c.Query("session_id")
	session, _, err := s.getOrCreateSession(sessionID)
	if err != nil || session == nil {
		c.JSON(http.StatusNotFound, gin.H{
			"error": "Session not found",
		})
		return
	}

	c.JSON(http.StatusOK, gin.H{
		"z_score":                session.Zscore,
		"tokens_spent":           session.TokenSpent,
		"total_available_tokens": session.TokenTotal,
		"tokens_per_sec":         session.TokensPerSec,
		"weighted_mean":          session.WeightedMean,
		"keys":                   session.Data.keys,
		"watermark":              session.Data.enableWatermark,
		"logit_bias":             session.Data.logitbias,
		"history_size":           session.Data.historySize,
		"gamma":                  session.Data.gamma,
		"seed":                   session.Data.seed,
		"watermark_type": 		  session.Data.watermarkType,
		"messages":               session.InternalMessages,
	})
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
		"total_count":            totalCnt,
		"green_count":            totalGreenCnt,
		"z_score":                zscore,
		"tokens_spent":           "invalid",
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
//check if reset contxt breaks sampler,
//handle canclel and multi request
//add padding handling in manual checker
