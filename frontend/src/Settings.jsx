import "./Settings.css";

function Settings({
  settings,
  setSettings,
  setPage,
  sessionID,
  closeSettings,
  serverStatus,
  isGenerating,
}) {
  // const API_HOST = `http://${window.location.hostname}:8080`;
  // const API_HOST = "http://localhost:8080";
  const API_HOST = "";
  const resetContext = async () => {
    try {
      await fetch(`${API_HOST}/api/resetctx?session_id=${sessionID}`);
      // await fetch(`${API_HOST}/resetctx?session_id=${sessionID}`);
      setPage("chat");
    } catch (error) {
      console.error("Failed to reset context:", error);
    }
  };

  const resetMessages = async () => {
    try {
      await fetch(`${API_HOST}/api/resetmsgs?session_id=${sessionID}`);
      // await fetch(`${API_HOST}/resetmsgs?session_id=${sessionID}`);
      window.location.reload();
    } catch (error) {
      console.error("Failed to reset messages:", error);
    }
  };

  const watermarkEnabled = settings.watermark;
  const isRedGreen = settings.watermark_type === "RedGreen";
  const isSynthID = settings.watermark_type === "SynthID";

  console.log("serverStatus:", serverStatus);
  console.log("type:", typeof serverStatus);

  return (
    <div className="settings-overlay">
      <div className="settings">
        <button
          className="close-settings"
          onClick={closeSettings}
          type="button"
        >
          ×
        </button>
        <label>
          Watermark
          <input
            type="checkbox"
            checked={settings.watermark}
            onChange={(e) =>
              setSettings({
                ...settings,
                watermark: e.target.checked,
              })
            }
          />
        </label>

        <label>
          Watermark Type
          <select
            value={settings.watermark_type}
            disabled={!watermarkEnabled}
            onChange={(e) =>
              setSettings({
                ...settings,
                watermark_type: e.target.value,
              })
            }
          >
            <option value="RedGreen">Red-Green</option>
            <option value="SynthID">SynthID</option>
          </select>
        </label>

        <label>
          Logit Bias
          <input
            type="number"
            step="0.1"
            value={settings.logit_bias}
            disabled={!watermarkEnabled || !isRedGreen}
            placeholder="2"
            onFocus={(e) => e.target.select()}
            onChange={(e) =>
              setSettings({
                ...settings,
                logit_bias: e.target.value,
              })
            }
          />
        </label>

        <label>
          Gamma
          <input
            type="number"
            step="0.1"
            min="0"
            max="1"
            value={settings.gamma}
            disabled={!watermarkEnabled || !isRedGreen}
            placeholder="0.6"
            onFocus={(e) => e.target.select()}
            onChange={(e) =>
              setSettings({
                ...settings,
                gamma: e.target.value,
              })
            }
          />
        </label>

        <label>
          History Size
          <input
            type="number"
            min="0"
            value={settings.history_size}
            disabled={!watermarkEnabled}
            placeholder="4"
            onFocus={(e) => e.target.select()}
            onChange={(e) =>
              setSettings({
                ...settings,
                history_size: e.target.value,
              })
            }
          />
        </label>

        <label>
          Seed
          <input
            type="text"
            value={settings.seed}
            disabled={!watermarkEnabled}
            placeholder="i_am_a_llm"
            onChange={(e) =>
              setSettings({
                ...settings,
                seed: e.target.value,
              })
            }
          />
        </label>

        <label>
          Keys
          <input
            type="text"
            value={settings.keys}
            disabled={!watermarkEnabled || !isSynthID}
            placeholder="1,2,3,4"
            onChange={(e) =>
              setSettings({
                ...settings,
                keys: e.target.value,
              })
            }
          />
        </label>
        <button type="button" disabled={!serverStatus || !isGenerating} onClick={resetContext}>
          Reset Context
        </button>

        <button type="button" disabled={!serverStatus || !isGenerating} onClick={resetMessages}>
          Clear Screen
        </button>
      </div>
    </div>
  );
}

export default Settings;
