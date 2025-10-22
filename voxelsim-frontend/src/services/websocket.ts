import { ConnectionStatus } from '../types/simulation';

export class WebSocketService {
  private ws: WebSocket | null = null;
  private url: string;

  constructor(url: string) {
    this.url = url;
  }

  async checkConnection(): Promise<ConnectionStatus> {
    return new Promise((resolve) => {
      try {
        const testWs = new WebSocket(this.url);

        testWs.onopen = () => {
          testWs.close();
          resolve('connected');
        };

        testWs.onerror = () => {
          resolve('disconnected');
        };

        // Timeout after 3 seconds
        setTimeout(() => {
          if (testWs.readyState === WebSocket.CONNECTING) {
            testWs.close();
            resolve('disconnected');
          }
        }, 3000);
      } catch (error) {
        console.error('WebSocket check error:', error);
        resolve('error');
      }
    });
  }

  connect(onMessage?: (data: MessageEvent) => void): Promise<void> {
    return new Promise((resolve, reject) => {
      try {
        this.ws = new WebSocket(this.url);

        this.ws.onopen = () => {
          console.log(`WebSocket connected to ${this.url}`);
          resolve();
        };

        this.ws.onerror = (error) => {
          console.error('WebSocket error:', error);
          reject(error);
        };

        this.ws.onmessage = (event) => {
          if (onMessage) {
            onMessage(event);
          }
        };

        this.ws.onclose = () => {
          console.log('WebSocket connection closed');
        };
      } catch (error) {
        reject(error);
      }
    });
  }

  disconnect(): void {
    if (this.ws) {
      this.ws.close();
      this.ws = null;
    }
  }

  send(data: unknown): void {
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify(data));
    }
  }
}

// Auto-detect WebSocket URL (localhost or production)
export const getWebSocketUrl = (): string => {
  if (window.location.hostname === 'localhost') {
    return 'ws://localhost:9080';
  }
  return `wss://${window.location.hostname.replace('voxelsim', 'voxelsim-backend')}/ws`;
};
