import "./Settings.css";

function Settings({ settings, setSettings, setPage }) {
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
      </div>
    </div>
  );
}

export default Settings;
