import React from 'react';
import './LoadingScreen.css';

interface LoadingScreenProps {
  error?: string;
}

export const LoadingScreen: React.FC<LoadingScreenProps> = ({ error }) => {
  if (error) {
    return (
      <div className="loading-screen">
        <h1>Error Loading Renderer</h1>
        <p>{error}</p>
        <p>Check console for details</p>
      </div>
    );
  }

  return (
    <div className="loading-screen">
      <h1>Loading VoxelSim Renderer...</h1>
      <p>Initializing WebAssembly and Bevy...</p>
    </div>
  );
};
