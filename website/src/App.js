import { useState, useEffect, useRef, useMemo, useCallback } from "react";
import { Line } from "react-chartjs-2";
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
} from "chart.js";
import "./App.css";

ChartJS.register(
    CategoryScale,
    LinearScale,
    PointElement,
    LineElement,
    Title,
    Tooltip,
    Legend
);

const chartOptions = {
  responsive: true,
  animation: false,
  plugins: {
    legend: {
      position: "top",
      labels: { color: "white" },
      color: "#CBD5E0",
      font: { size: 12 }
    },
  },
  scales: {
    y: {
      ticks: { color: "white" },
      min: -20,
      max: 20,
      grid: { color: "rgba(160, 174, 192, 0.1)" }
    },
    x: {
      grid: { color: "rgba(160, 174, 192, 0.05)" },
      ticks: { display: false },
    },
  },
};

const SensorChart = ({ data, sensorType }) => {
  const axisMap = useMemo(() => ({
    Acc: { X: 'ax', Y: 'ay', Z: 'az' },
    Gyro: { X: 'gx', Y: 'gy', Z: 'gz' }
  }), []);

  const colors = useMemo(() => ({
    Acc: ["#FF6B6B", "#4ECDC4", "#FFE66D"],
    Gyro: ["#6BFFB8", "#FF9F6B", "#9B6BFF"]
  }), []);

  const chartData = useMemo(() => ({
    labels: Array(data.length).fill(""),
    datasets: ['X', 'Y', 'Z'].map((axis, i) => ({
      label: `${sensorType}${axis}`,
      data: data.map(p => p[axisMap[sensorType][axis]]),
      borderColor: colors[sensorType][i],
      borderWidth: 1.2,
      tension: 0.2,
      pointRadius: 0,
      pointHoverRadius: 3
    }))
  }), [data, sensorType, axisMap, colors]);

  return (
      <div className="chart-card">
        <h3 className="chart-title">{sensorType} Data</h3>
        <div className="chart-container">
          <Line options={chartOptions} data={chartData} />
        </div>
      </div>
  );
};

const ActivityBadge = ({ activity, confidence }) => {
  const activityColors = {
    walking: "#48BB78",
    running: "#4299E1",
    sitting: "#F6AD55",
    standing: "#9F7AEA",
    unknown: "#A0AEC0"
  };

  return (
      <div className="activity-badge">
        <div
            className="activity-dot"
            style={{ backgroundColor: activityColors[activity] || "#A0AEC0" }}
        />
        <div className="activity-info">
          <h2 className="activity-text">{activity?.toUpperCase() || "UNKNOWN"}</h2>
          <p className="confidence-text">
            Confidence: {(confidence * 100 || 0).toFixed(1)}%
          </p>
        </div>
      </div>
  );
};

const SensorValue = ({ label, value }) => (
    <div className="sensor-value">
      <span className="sensor-label">{label}:</span>
      <span className="sensor-number">{value.toFixed(2)}</span>
    </div>
);

function App() {
  const [currentMessage, setCurrentMessage] = useState(null);
  const [error, setError] = useState(null);
  const dataRef = useRef(Array(50).fill({ ax: 0, ay: 0, az: 0, gx: 0, gy: 0, gz: 0 }));
  const socketRef = useRef(null);
  const [, forceUpdate] = useState(0);

  const handleMessage = useCallback((data) => {
    dataRef.current = [...dataRef.current.slice(1), data.data];
    forceUpdate(prev => prev + 1);
    setCurrentMessage(data);
  }, []);

  useEffect(() => {
    socketRef.current = new WebSocket("ws://played-presence-throws-rugby.trycloudflare.com:8080/display");

    socketRef.current.onopen = () => setError(null);
    socketRef.current.onerror = () => setError("Connection failed");
    socketRef.current.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        if (data?.data) handleMessage(data);
      } catch (err) {
        console.error("Error parsing data:", err);
      }
    };

    return () => {
      if (socketRef.current?.readyState === WebSocket.OPEN) {
        socketRef.current.close();
      }
    };
  }, [handleMessage]);

  return (
      <div className="App">
        <header className="App-header">
          {error && <div className="error-banner">{error}</div>}

          <div className="dashboard-grid">
            <div className="status-panel">
              <ActivityBadge
                  activity={currentMessage?.activity}
                  confidence={currentMessage?.confidence}
              />

              <div className="sensor-grid">
                {currentMessage?.data && [
                  ["AccX", "ax"], ["AccY", "ay"], ["AccZ", "az"],
                  ["GyroX", "gx"], ["GyroY", "gy"], ["GyroZ", "gz"]
                ].map(([label, key]) => (
                    <SensorValue
                        key={label}
                        label={label}
                        value={currentMessage.data[key]}
                    />
                ))}
              </div>
            </div>

            <div className="charts-panel">
              <SensorChart data={dataRef.current} sensorType="Acc" />
              <SensorChart data={dataRef.current} sensorType="Gyro" />
            </div>
          </div>
        </header>
      </div>
  );
}

export default App;