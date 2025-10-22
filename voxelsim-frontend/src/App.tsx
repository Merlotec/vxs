import React, { useState, useEffect } from 'react';
import { LoadingScreen } from './components/LoadingScreen';
import { StatusBar } from './components/StatusBar';
import { Canvas } from './components/Canvas';
import { StatusInfo } from './types/simulation';
import { WebSocketService, getWebSocketUrl } from './services/websocket';
import { initRenderer } from './services/renderer';
import './styles/global.css';

function App() {
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [status, setStatus] = useState<StatusInfo>({
    proxy: 'checking',
    websocket: 'checking',
    simulation: 'checking'
  });

  useEffect(() => {
    const wsUrl = getWebSocketUrl();
    const wsService = new WebSocketService(wsUrl);

    async function initialize() {
      try {
        // Check proxy/backend connection
        console.log(`Checking backend at ${wsUrl}...`);
        const proxyStatus = await wsService.checkConnection();

        setStatus(prev => ({ ...prev, proxy: proxyStatus }));

        if (proxyStatus !== 'connected') {
          console.warn(
            `Backend server not detected at ${wsUrl}. Make sure backend is running.`
          );
        }

        // Initialize WASM renderer
        console.log('Initializing WASM renderer...');
        await initRenderer();

        setLoading(false);
        setStatus(prev => ({
          ...prev,
          websocket: 'connected',
          simulation: 'connected'
        }));

        console.log('VoxelSim Renderer initialized!');
        console.log(`Waiting for simulation data on ${wsUrl}...`);
      } catch (err) {
        console.error('Failed to initialize:', err);
        setError(err instanceof Error ? err.message : String(err));
        setLoading(false);
        setStatus(prev => ({
          ...prev,
          websocket: 'error',
          simulation: 'error'
        }));
      }
    }

    initialize();

    // Cleanup
    return () => {
      wsService.disconnect();
    };
  }, []);

  return (
    <>
      {loading && <LoadingScreen error={error ?? undefined} />}
      <StatusBar status={status} />
      <Canvas />
    </>
  );
}

export default App;
