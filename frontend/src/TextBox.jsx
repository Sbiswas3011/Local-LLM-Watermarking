import { useEffect, useRef, useState } from "react";
import "./TextBox.css";
import Settings from "./Settings.jsx";

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
    console.log("localStorage Unavailable to Get");
  }

  if (!id) {
    id = makeID();

    try {
      localStorage.setItem("session_id", id);
    } catch (e) {
      console.log("localStorage Unavailable to Set");
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
  const newlineRef = useRef(true);
  const [page, setPage] = useState("chat");
  const [stats, setStats] = useState({
    zScore: 0,
    contextUsed: 0,
    totalContext: 0,
  });

  const [settings, setSettings] = useState({
    watermark: false,
    logit_bias: 4.0,
    gamma: 0.6,
    history_size: 4,
    seed: "i_am_a_llm",
  });

  const API_HOST = `${window.location.hostname}:8080`;

  useEffect(() => {
    const sessionID = getSessionID();

    fetch(`http://${API_HOST}/getsession?session_id=${sessionID}`)
      .then((response) => {
        if (!response.ok) {
          throw new Error("Failed to get session");
        }
        return response.json();
      })
      .then((data) => {
        setStats({
          zScore: data.z_score,
          contextUsed: data.tokens_spent,
          totalContext: data.total_available_tokens,
        });

        setSettings({
          watermark: data.watermark,
          logit_bias: data.logit_bias,
          gamma: data.gamma,
          history_size: data.history_size,
          seed: data.seed,
        });

        const renderedMessages = data.messages.map((message) => {
          if (message.Role === "user") {
            return {
              role: "user",
              content: message.Data[0] || "",
            };
          }

          return {
            role: "assistant",

            content: message.Data.map((text, index) => ({
              text: text,
              isGreen: message.Greensplit[index],
            })),

            thinking: message.Thinkdata.map((text, index) => ({
              text: text,
              isGreen: message.Thinkgreensplit[index],
            })),
            watermarked: message.Watermarked,
            showThinking: false,
          };
        });

        renderedMessages.forEach((message) => {
          if (message.role !== "assistant") return;

          if (message.content.length > 0) {
            message.content[0].text = message.content[0].text.replace(
              /^\n+/,
              "",
            );
          }

          if (message.thinking.length > 0) {
            message.thinking[0].text = message.content[0].text.replace(
              /^\n+/,
              "",
            );
          }
        });

        setMessages(renderedMessages);
      })
      .catch((error) => {
        console.error("Failed to load session:", error);
      });
  }, []);

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
      { role: "user", content: prompt },
      { role: "assistant", content: [], thinking: [], showThinking: false },
    ]);

    const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";

    const sessionID = getSessionID();

    const wsUrl = `${protocol}//${API_HOST}/ws?session_id=${sessionID}`;

    console.log("Connecting to:", wsUrl);

    const ws = new WebSocket(wsUrl);

    ws.onopen = () => {
      ws.send(
        JSON.stringify({
          type: "chat",
          text: prompt,
          watermark: settings.watermark,
          logit_bias: settings.logit_bias,
          gamma: settings.gamma,
          history_size: settings.history_size,
          seed: settings.seed,
          reset_sampler: true,
        }),
      );
    };

    ws.onmessage = (event) => {
      const result = JSON.parse(event.data);
      let token = result.Token;
      const isGreen = result.IsGreen;

      if (token.includes("<think>")) {
        // console.log("Setting True")
        thinkingRef.current = true;
        newlineRef.current = true;
        setIsThinking(true);
        return;
      }

      if (token.includes("</think>")) {
        // console.log("Setting False")
        newlineRef.current = true;
        thinkingRef.current = false;
        setIsThinking(false);
        return;
      }

      if (newlineRef.current) {
        token = token.replace(/^\n+/, "");
        newlineRef.current = false;
      }

      console.log("iThinking, token:", isThinking, token);

      setStats({
        zScore: result.ZScore,
        contextUsed: result.ContextUsed,
        totalContext: result.TotalContext,
      });

      setMessages((previous) => {
        const updated = [...previous];
        const lastIndex = updated.length - 1;

        if (thinkingRef.current) {
          updated[lastIndex] = {
            ...updated[lastIndex],
            thinking: [
              ...updated[lastIndex].thinking,
              {
                text: token,
                isGreen: isGreen,
                watermarked: result.Watermarked,
              },
            ],
          };
        } else {
          updated[lastIndex] = {
            ...updated[lastIndex],
            content: [
              ...updated[lastIndex].content,
              {
                text: token,
                isGreen: isGreen,
                watermarked: result.Watermarked,
              },
            ],
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

  const handleStop = async (event) => {
    event.preventDefault();
    const sessionID = getSessionID();

    try {
      await fetch(`http://${API_HOST}/closechan?session_id=${sessionID}`);
    } catch (error) {
      console.error("Failed to stop generation:", error);
    }
  };

  return (
    <div className="app">
      <main className="chat">
        {messages.length > 0 && (
          <div className="chat-header">
            <h1>Local LLM (Swift-Qwen3.8-27B-Q4_K_M)</h1>
          </div>
        )}

        <div className={`thinking-indicator ${isThinking ? "thinking" : ""}`}>
          <span className="thinking-dot"></span>
          Generating Think Tokens
        </div>

        <div className="generation-stats">
          <div>
            Context: {stats.contextUsed} / {stats.totalContext}
          </div>
          <div>Z-Score: {Number(stats.zScore).toFixed(2)}</div>
        </div>

        <div className="messages">
          {messages.length === 0 && (
            <>
              <h1>Local LLM (Swift-Qwen3.8-27B-Q4_K_M)</h1>
              <p>Ask your model anything.</p>
            </>
          )}

          {messages.map((message, index) => (
            <div key={index} className={`message ${message.role}`}>
              {message.role === "user" && message.content}

              {message.role === "assistant" && (
                <>
                  <div className="assistant-content">
                    {message.content.map((item, index) => (
                      <span
                        key={index}
                        // style={{ color: item.isGreen ? "green" : "red" }}
                        style={
                          (message.watermarked ?? item.watermarked)
                            ? { color: item.isGreen ? "green" : "red" }
                            : {}
                        }
                      >
                        {item.text}
                      </span>
                    ))}
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
                          {message.thinking.map((item, index) => (
                            <span
                              key={index}
                              // style={{
                              //   color: item.isGreen ? "green" : "red",
                              // }}
                              style={
                                (message.watermarked ?? item.watermarked)
                                  ? { color: item.isGreen ? "green" : "red" }
                                  : {}
                              }
                            >
                              {item.text}
                            </span>
                          ))}
                        </div>
                      )}
                    </>
                  )}
                </>
              )}
            </div>
          ))}
        </div>

        <form
          className="composer"
          onSubmit={isGenerating ? handleStop : handleSubmit}
        >
          <button
            className="settings-button"
            type="button"
            onClick={() => setPage(page === "settings" ? "chat" : "settings")}
          >
            <span className="setting-gear">⚙️</span>
          </button>

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
            className="send-button"
            type="submit"
            // disabled={isGenerating || !text.trim()}
            aria-label={isGenerating ? "Stop generation" : "Send message"}
          >
            <span className="send-arrow">{isGenerating ? "■" : "🡅"}</span>
          </button>
        </form>
      </main>

      {page === "settings" && (
        <Settings
          settings={settings}
          setSettings={setSettings}
          setPage={setPage}
          sessionID={getSessionID()}
        />
      )}
    </div>
  );
}

export default TextBox;
