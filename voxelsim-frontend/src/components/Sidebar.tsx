import React from 'react';
import './Sidebar.css';

export const Sidebar: React.FC = () => {
  return (
    <div className="sidebar">
      <h2>Controls</h2>
      <div className="sidebar-section">
        <h3>Simulation</h3>
        <p>Speed: Normal</p>
        <p>Status: Running</p>
      </div>
      <div className="sidebar-section">
        <h3>Camera</h3>
        <p>Position: (0, 0, 0)</p>
      </div>
      <div className="sidebar-section">
        <h3>Stats</h3>
        <p>FPS: 60</p>
        <p>Objects: 0</p>
      </div>
    </div>
  );
};
