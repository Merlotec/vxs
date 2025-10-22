export type ConnectionStatus = 'checking' | 'connected' | 'disconnected' | 'error';

export interface StatusInfo {
  proxy: ConnectionStatus;
  websocket: ConnectionStatus;
  simulation: ConnectionStatus;
}

export interface WebSocketMessage {
  type: string;
  data?: unknown;
}
