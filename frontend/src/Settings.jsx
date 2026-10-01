import "./Settings.css";

function Settings({ settings, setSettings, setPage, sessionID }) {
  const resetContext = async () => {
    try {
      await fetch(`http://localhost:8080/resetctx?session_id=${sessionID}`);
      setPage("chat")
    } catch (error) {
      console.error("Failed to reset context:", error);
    }
  };

  const resetMessages = async () => {
    try {
      await fetch(`http://localhost:8080/resetmsgs?session_id=${sessionID}`);

      window.location.reload();
    } catch (error) {
      console.error("Failed to reset messages:", error);
    }
  };
  return (
    <div className="settings-overlay">
      <div className="settings">
        <button
          className="close-settings"
          onClick={() => setPage("chat")}
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
          Logit Bias
          <input
            type="number"
            step="0.1"
            value={settings.logit_bias}
            onFocus={(e) => e.target.select()}
            onChange={(e) =>
              setSettings({
                ...settings,
                logit_bias: Number(e.target.value),
              })
            }
          />
        </label>

        <label>
          Gamma
          <input
            type="number"
            step="0.05"
            value={settings.gamma}
            onFocus={(e) => e.target.select()}
            onChange={(e) =>
              setSettings({
                ...settings,
                gamma: Number(e.target.value),
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
            onFocus={(e) => e.target.select()}
            onChange={(e) =>
              setSettings({
                ...settings,
                history_size: Number(e.target.value),
              })
            }
          />
        </label>

        <label>
          Seed
          <input
            type="text"
            value={settings.seed}
            onChange={(e) =>
              setSettings({
                ...settings,
                seed: e.target.value,
              })
            }
          />
        </label>
        <button type="button" onClick={resetContext}>
          Reset Context
        </button>

        <button type="button" onClick={resetMessages}>
          Clear Screen
        </button>
      </div>
    </div>
  );
}

export default Settings;
