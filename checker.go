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
	"fmt"
	"math"
	"time"
	"unsafe"
)

type TokenResult struct {
	Token           string
	GreenCount      int
	TotalCount      int
	GreenPercentage float64
	ZScore          float64
	ContextUsed     int
	TotalContext    int
	IsGreen         bool
	TokensPerSecond float64
	Watermarked     bool
	WeightedMean    float64
}

func BasicGreenStreamPercentage(prompt PromptData, startTime time.Time) (int, int, float64, TokenResult, bool) {

	result := TokenResult{}
	prompt.totalTokenCnt++
	// Keys := []int64{1, 2, 3, 4, 5}
	// weightedMean := 0.0

	if !prompt.enableWatermark {
		result.Token = prompt.piece
		result.ContextUsed = int(prompt.nctxUsed)
		result.TotalContext = int(prompt.nctx)
		result.Watermarked = prompt.enableWatermark
	} else {
		seedC := C.CString(prompt.seed)
		defer C.free(unsafe.Pointer(seedC))

		switch prompt.watermarkType {
		case "RedGreen":
			isGreen := false
			isGreen = bool(C.llama_sampler_check_basic_watermarkv2(prompt.newTokenID, C.float(prompt.gamma), seedC, prompt.tokenhistoryPtr, C.size_t(prompt.historySize), C.int32_t(prompt.n_vocab)))
			if isGreen {
				prompt.totalGreenTokenCnt++
				result.IsGreen = true
			} else {
				result.IsGreen = false
			}

			expected := float64(prompt.totalTokenCnt) * prompt.gamma
			variance := float64(prompt.totalTokenCnt) * prompt.gamma * (1.0 - prompt.gamma)
			if variance > 0 {
				prompt.CurrentZscore = (float64(prompt.totalGreenTokenCnt) - expected) / math.Sqrt(variance)
			} else {
				prompt.CurrentZscore = 0
			}

		case "SynthID":
			prompt.CurrentZscore = 0
			// fmt.Println("Weighted Mean Before: ",prompt.weightedMean)
			prompt.weightedMean = float64(C.llama_sampler_check_synthId_watermark(prompt.newTokenID, (*C.int64_t)(unsafe.Pointer(&prompt.keys[0])), C.size_t(len(prompt.keys)), seedC, prompt.tokenhistoryPtr, C.size_t(prompt.historySize), C.double(prompt.weightedMean), C.uint32_t(prompt.totalTokenCnt-1)))
			result.WeightedMean = prompt.weightedMean
			// fmt.Println("Weighted Mean After: ",prompt.weightedMean)
			if prompt.weightedMean > 0.5{
				prompt.totalGreenTokenCnt++
				result.IsGreen = true
			}else {
				result.IsGreen = false
			}
		}

		result.Token = prompt.piece
		// result.GreenPercentage = float64(prompt.totalGreenTokenCnt * 100 / prompt.totalTokenCnt)
		result.ZScore = prompt.CurrentZscore
		result.ContextUsed = int(prompt.nctxUsed)
		result.TotalContext = int(prompt.nctx)
		result.Watermarked = prompt.enableWatermark
	}

	elapsedTime := time.Since(startTime).Seconds()
	tokensPerSecond := 1 / elapsedTime
	// fmt.Println("elapsed, count and tokens/s: ", elapsedTime, 1, tokensPerSecond)
	result.TokensPerSecond = tokensPerSecond

	select {
	case <-prompt.CloseResultChan:
		return -1, -1, 0.0, result, true

	case prompt.ResultChan <- result:
	}

	return prompt.totalTokenCnt, prompt.totalGreenTokenCnt, prompt.CurrentZscore, result, false
}

func ProcessText(request ProcessRequest, data PromptData) (int, int, float64, error) {

	data.historySize = request.HistorySize
	data.gamma = request.Gamma
	data.totalTokenCnt = 0
	data.totalGreenTokenCnt = 0
	seedC := C.CString(request.Seed)
	defer C.free(unsafe.Pointer(seedC))

	data.prompt = request.Text
	cPrompt := C.CString(data.prompt)
	defer C.free(unsafe.Pointer(cPrompt))

	nPromptTokens := -C.llama_tokenize(data.vocab, cPrompt, C.int32_t(len(data.prompt)), nil, 0, true, true)

	promptTokens := make([]C.llama_token, nPromptTokens)

	var tokenPtr *C.llama_token
	if nPromptTokens > 0 {
		tokenPtr = (*C.llama_token)(unsafe.Pointer(&promptTokens[0]))
	}

	if len(promptTokens) <= data.historySize {
		return 0, 0, 0, fmt.Errorf("text has too few tokens for history size %d", data.historySize)
	}

	data.totalTokenCnt = data.historySize

	ret := C.llama_tokenize(data.vocab, cPrompt, C.int32_t(len(data.prompt)), tokenPtr, C.int32_t(len(promptTokens)), true, true)
	if ret < 0 {
		return 0, 0, 0.0, fmt.Errorf("failed to tokenize prompt")
	}

	// startToken := 0
	// currentToken := data.historySize
	// startToken := currentToken - data.historySize
	// data.newTokenID = promptTokens[currentToken]

	// gotokenhistory := promptTokens[startToken:currentToken]

	// var tokenhistoryPtr *C.llama_token
	// if len(gotokenhistory) > 0 {
	// 	tokenhistoryPtr = (*C.llama_token)(unsafe.Pointer(&gotokenhistory[0]))
	// }
	// data.tokenhistoryPtr = tokenhistoryPtr

	currentToken := data.historySize

	for {
		// fmt.Println("Current Token & mPromptTokens: ", currentToken, int(nPromptTokens)-1)
		startToken := currentToken - data.historySize
		data.newTokenID = promptTokens[currentToken]

		gotokenhistory := promptTokens[startToken:currentToken]

		var tokenhistoryPtr *C.llama_token
		if len(gotokenhistory) > 0 {
			tokenhistoryPtr = (*C.llama_token)(unsafe.Pointer(&gotokenhistory[0]))
		}
		data.tokenhistoryPtr = tokenhistoryPtr

		isGreen := false

		isGreen = bool(C.llama_sampler_check_basic_watermarkv2(data.newTokenID, C.float(data.gamma), seedC, data.tokenhistoryPtr, C.size_t(data.historySize), C.int32_t(data.n_vocab)))

		if isGreen {
			data.totalGreenTokenCnt++
		}

		data.totalTokenCnt++
		currentToken++

		if currentToken >= int(nPromptTokens)-1 {
			break
		}

	}

	// nCtx := C.llama_n_ctx(Data.ctx)
	// Data.nctx = nCtx

	expected := float64(data.totalTokenCnt) * request.Gamma
	variance := float64(data.totalTokenCnt) * request.Gamma * (1.0 - request.Gamma)
	data.CurrentZscore = (float64(data.totalGreenTokenCnt) - expected) / math.Sqrt(variance)

	return data.totalTokenCnt, data.totalGreenTokenCnt, data.CurrentZscore, nil
}
