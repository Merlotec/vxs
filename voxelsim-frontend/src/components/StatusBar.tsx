import React from "react";
import { StatusInfo } from "../types/simulation";
import "./StatusBar.css";

interface StatusBarProps {
  status: StatusInfo;
}

const getStatusLabel = (status: string): string => {
  switch (status) {
    case "checking":
      return "Checking...";
    case "connected":
      return "Running";
    case "disconnected":
      return "Not running";
    case "error":
      return "Error";
    default:
      return "Waiting...";
  }
};

export const StatusBar: React.FC<StatusBarProps> = ({ status }) => {
  return (
    <div className="status-bar">
      <div className="status-item">
        Proxy Server:{" "}
        <span className={`status-${status.proxy}`}>
          {getStatusLabel(status.proxy)}
        </span>
      </div>
      <div className="status-item">
        WebSocket:{" "}
        <span className={`status-${status.websocket}`}>
          {getStatusLabel(status.websocket)}
        </span>
      </div>
      <div className="status-item">
        Simulation:{" "}
        <span className={`status-${status.simulation}`}>
          {getStatusLabel(status.simulation)}
        </span>
      </div>
    </div>
  );
};
