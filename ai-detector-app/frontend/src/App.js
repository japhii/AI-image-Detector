import React, { useState, useEffect, useRef, useCallback } from "react";
import "./App.css";

function App() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [previewUrl,   setPreviewUrl]   = useState(null);
  const [loading,      setLoading]      = useState(false);
  const [prediction,   setPrediction]   = useState(null);
  const [error,        setError]        = useState(null);
  const [history,      setHistory]      = useState([]);

  const fileInputRef = useRef(null);

  const fetchHistory = useCallback(async () => {
    try {
      const res = await fetch("/api/history");
      if (!res.ok) throw new Error("Failed to load history.");
      const data = await res.json();
      setHistory(data);
    } catch (err) {
      console.error("Error fetching history:", err);
    }
  }, []);

  useEffect(() => {
    fetchHistory();
  }, [fetchHistory]);

  const handleFileChange = (e) => {
    const file = e.target.files[0];
    if (!file) return;

    if (!file.type.startsWith("image/")) {
      setError("Please select a valid image file.");
      return;
    }

    setSelectedFile(file);
    setPreviewUrl(URL.createObjectURL(file));
    setPrediction(null);
    setError(null);
  };

  const handleDragOver = (e) => {
    e.preventDefault();
  };

  const handleDrop = (e) => {
    e.preventDefault();

    const file = e.dataTransfer.files?.[0];
    if (!file) return;

    if (!file.type.startsWith("image/")) {
      setError("Please drop a valid image file.");
      return;
    }

    setSelectedFile(file);
    setPreviewUrl(URL.createObjectURL(file));
    setPrediction(null);
    setError(null);
  };

  const handleAnalyze = async () => {
    if (!selectedFile) {
      setError("Please select an image first.");
      return;
    }

    setLoading(true);
    setError(null);

    const formData = new FormData();
    formData.append("image", selectedFile);

    try {
      const res  = await fetch("/api/predict", { method: "POST", body: formData });
      const data = await res.json();

      if (!res.ok || data.status !== "success") {
        throw new Error(data.message || "Prediction failed.");
      }

      setPrediction(data);
      fetchHistory();
    } catch (err) {
      setError(err.message || "An error occurred while connecting to the backend.");
    } finally {
      setLoading(false);
    }
  };

  const handleReset = () => {
    setSelectedFile(null);
    setPreviewUrl(null);
    setPrediction(null);
    setError(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = "";
    }
  };

  const getConfidencePercent = () => {
    if (!prediction) return 0;
    const score = prediction.is_ai
      ? prediction.confidence_score
      : 1.0 - prediction.confidence_score;
    return (score * 100).toFixed(1);
  };

  return (
    <div className="app-container">

      <header className="app-header">
        <div className="header-badge">Deep Learning CNN</div>
        <h1>AI vs Real Image Detector</h1>
        <p>Upload any photo to analyze whether it&apos;s authentic or AI-generated</p>
      </header>

      <main className="main-content">

        <section className="upload-card">
          <div
            className={`drop-zone ${selectedFile ? "has-file" : ""}`}
            onDragOver={handleDragOver}
            onDrop={handleDrop}
            onClick={() => fileInputRef.current?.click()}
          >
            <input
              type="file"
              ref={fileInputRef}
              onChange={handleFileChange}
              accept="image/*"
              style={{ display: "none" }}
            />

            {previewUrl ? (
              <div className="preview-container">
                <img src={previewUrl} alt="Preview" className="image-preview" />
                <p className="file-name">{selectedFile?.name}</p>
              </div>
            ) : (
              <div className="upload-placeholder">
                <div className="upload-icon">📷</div>
                <h3>Drag &amp; Drop your image here</h3>
                <p>or click to browse from your device</p>
                <span className="file-types">Supports JPG, PNG, WEBP, BMP</span>
              </div>
            )}
          </div>

          {error && <div className="error-message">⚠️ {error}</div>}

          <div className="button-group">
            <button
              id="analyze-btn"
              className="btn btn-primary"
              onClick={handleAnalyze}
              disabled={!selectedFile || loading}
            >
              {loading ? (
                <span className="spinner-container">
                  <span className="spinner"></span> Analyzing...
                </span>
              ) : (
                "Analyze Image"
              )}
            </button>

            {selectedFile && (
              <button
                id="reset-btn"
                className="btn btn-secondary"
                onClick={handleReset}
                disabled={loading}
              >
                Reset
              </button>
            )}
          </div>
        </section>

        {prediction && (
          <section className="result-card">
            <h2>Analysis Result</h2>
            <div className="result-body">

              <div className={`badge ${prediction.is_ai ? "badge-ai" : "badge-real"}`}>
                {prediction.is_ai ? "🤖 AI-Generated" : "📸 Real Image"}
              </div>

              <div className="confidence-section">
                <div className="confidence-header">
                  <span>Detection Confidence</span>
                  <span className="confidence-value">{getConfidencePercent()}%</span>
                </div>
                <div className="progress-bar-bg">
                  <div
                    className={`progress-bar-fill ${prediction.is_ai ? "fill-ai" : "fill-real"}`}
                    style={{ width: `${getConfidencePercent()}%` }}
                  />
                </div>
              </div>

              <div className="result-details">
                <p><strong>File Name:</strong> {prediction.filename}</p>
                <p>
                  <strong>Raw AI Probability:</strong>{" "}
                  {(prediction.confidence_score * 100).toFixed(2)}%
                </p>
              </div>
            </div>
          </section>
        )}

        <section className="history-card">
          <h2>Recent Classifications (SQLite Log)</h2>
          {history.length === 0 ? (
            <p className="no-history">
              No predictions yet. Upload an image above to get started!
            </p>
          ) : (
            <div className="table-responsive">
              <table className="history-table">
                <thead>
                  <tr>
                    <th>ID</th>
                    <th>Filename</th>
                    <th>Result</th>
                    <th>Confidence</th>
                    <th>Timestamp</th>
                  </tr>
                </thead>
                <tbody>
                  {history.map((row) => {
                    const isAi = row.is_ai;
                    const conf = isAi
                      ? (row.confidence_score * 100).toFixed(1)
                      : ((1 - row.confidence_score) * 100).toFixed(1);

                    return (
                      <tr key={row.id}>
                        <td>#{row.id}</td>
                        <td className="table-filename">{row.filename}</td>
                        <td>
                          <span className={`table-badge ${isAi ? "tbl-ai" : "tbl-real"}`}>
                            {isAi ? "AI-Generated" : "Real Image"}
                          </span>
                        </td>
                        <td>{conf}%</td>
                        <td className="table-timestamp">
                          {new Date(row.timestamp).toLocaleString()}
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          )}
        </section>

      </main>

      <footer className="app-footer">
        <p>Built with Flask · TensorFlow · SQLite · React.js</p>
      </footer>

    </div>
  );
}

export default App;
