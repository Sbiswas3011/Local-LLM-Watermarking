
import { useRef, useState } from "react";
import "./TextBox.css";

function makeID() {
  if (window.crypto?.randomUUID) {
    return window.crypto.randomUUID();
  }

  return Date.now().toString(36) + Math.random().toString(36).slice(2);
}

function getSessionID() {
  let id;

  try {
    id = localStorage.getItem("session_id");
  } catch (e) {
    console.log("localStorage Unavailable to Get")
  }

  if (!id) {
    id = makeID();

    try {
      localStorage.setItem("session_id", id);
    } catch (e) {
      console.log("localStorage Unavailable to Set")
    }
  }

  return id;
}

function TextBox() {

  const [text, setText] = useState("");
  const [messages, setMessages] = useState([]);
  const [isGenerating, setIsGenerating] = useState(false);
  const [isThinking, setIsThinking] = useState(false);
  const thinkingRef = useRef(false);

  const handleChange = (event) => setText(event.target.value);

  function handleSubmit(event) {
    event.preventDefault();

    const prompt = text.trim();

    if (!prompt || isGenerating) {
      return;
    }

    setText("");
    setIsGenerating(true);

    // Add the user's message and an empty assistant message.
    setMessages((previous) => [
      ...previous,
      { role: "user", content: prompt, thinking: "", showThinking: false},
      { role: "assistant", content: "", thinking: "", showThinking: false},
    ]);

    const protocol =
      window.location.protocol === "https:" ? "wss:" : "ws:";

    const sessionID = getSessionID();

    const wsUrl = `${protocol}//localhost:8080/ws?session_id=${sessionID}`

    console.log("Connecting to:", wsUrl);

    const ws = new WebSocket(wsUrl);

    ws.onopen = () => {
      ws.send(
        JSON.stringify({
          type: "placeholder",
          text: prompt,
          watermark: false,
          logit_bias: 4.0,
          gamma: 0.6,
          history_size: 4,
          seed: "i_am_a_llm",
        })
      );
    };

    ws.onmessage = (event) => {
      const result = JSON.parse(event.data);
      const token = result.Token;

      if (token.includes("<think>")) {
        console.log("Setting True")
        thinkingRef.current = true;
        setIsThinking(true);
        return;
      }

      if (token.includes("</think>")) {
        console.log("Setting False")
        thinkingRef.current = false
        setIsThinking(false);
        return;
      }

      console.log("iThinking, token:", isThinking, token)

      setMessages((previous) => {
        const updated = [...previous];
        const lastIndex = updated.length - 1;

        if (thinkingRef.current){
          updated[lastIndex] = {
            ...updated[lastIndex],
            thinking: updated[lastIndex].thinking + result.Token,
          };
        }else{
          updated[lastIndex] = {
            ...updated[lastIndex],
            content: updated[lastIndex].content + result.Token,
          };
        }

        return updated;
      });
    };

    ws.onerror = (error) => {
      console.error("WebSocket error:", error);
      setMessages((previous) => {
        const updated = [...previous];
        const lastIndex = updated.length - 1;

        updated[lastIndex] = {
          ...updated[lastIndex],
          content:
            updated[lastIndex].content ||
            "WebSocket error. Check your Go server.",
        };

        return updated;
      });
    };

    ws.onclose = (event) => {
      console.log("WebSocket closed");
      console.log("Code:", event.code);
      console.log("Reason:", event.reason);
      console.log("Clean:", event.wasClean);
      setIsGenerating(false);
    };
  }

  return (
    <div className="app">
      <main className="chat">

        {/* <header className="chat-header">
          <h1>Local LLM (Swift-Qwen3.8-27B-Q4_K_M)</h1>
        </header> */}

        <div className={`thinking-indicator ${isThinking ? "thinking" : ""}`}>
          <span className="thinking-dot"></span>
            Generating Think Tokens
        </div>

        <div className="messages">
          {messages.length === 0 && (
            <>
              <h1>Local LLM (Swift-Qwen3.8-27B-Q4_K_M)</h1>
              <p>Ask your model anything.</p>
            </>
          )}

          {messages.map((message, index) => (
            <div
              key={index}
              className={`message ${message.role}`}
            >

               {message.role === "user" && (
                  message.content
                )}

                {message.role === "assistant" && (
                <>
                  <div className="assistant-content">
                    {message.content}
                  </div>
                
                <button
                      className="thinking-toggle"
                      onClick={() => {
                        setMessages((previous) => {
                          const updated = [...previous];

                          updated[index] = {
                            ...updated[index],
                            showThinking: !updated[index].showThinking,
                          };

                          return updated;
                        });
                      }}
                    >
                      Show Thinking {message.showThinking ? "▲" : "▼"}
                </button>

                {message.thinking && (
                  <>
                    {message.showThinking && (
                      <div className="thinking-box">
                        {message.thinking}
                      </div>
                    )}
                  </>
                )}
                </ >
                )}
            </div>
          ))}
        </div>

        <form className="composer" onSubmit={handleSubmit}>
          <textarea
            value={text}
            onChange={handleChange}
            placeholder="Message your model..."
            rows={1}
            disabled={isGenerating}
            onKeyDown={(event) => {
              if (
                event.key === "Enter" &&
                !event.shiftKey &&
                !event.nativeEvent.isComposing
              ) {
                event.preventDefault();
                event.currentTarget.form.requestSubmit();
              }
            }}
          />

          <button
            type="submit"
            disabled={isGenerating || !text.trim()}
            aria-label="Send message"
          >
            <span className="send-arrow">🡅</span>
          </button>
        </form>
      </main>
    </div>
  );
}

export default TextBox;