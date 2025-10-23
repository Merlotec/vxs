const API_BASE_URL = import.meta.env.VITE_API_URL || "http://localhost:8099";

export interface RunConfig {
  user_goal: string;
  iterations: number;
  episodes: number;
  render: boolean;
  duration_seconds?: number;
  max_steps?: number;
  delta?: number;
  provider?: "anthropic" | "openai";
  model?: string;
  use_px4?: boolean;
  success_threshold?: number;
}

export interface RunResponse {
  run_id: string;
}

export interface CodeResponse {
  run_id: string;
  code: string;
}

export class BgenAPI {
  private baseUrl: string;

  constructor(baseUrl: string = API_BASE_URL) {
    this.baseUrl = baseUrl;
  }

  async startRun(config: RunConfig): Promise<RunResponse> {
    const response = await fetch(`${this.baseUrl}/run`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(config),
    });

    if (!response.ok) {
      throw new Error(`Failed to start run: ${response.statusText}`);
    }

    return response.json();
  }

  async getCode(runId: string): Promise<CodeResponse> {
    const response = await fetch(`${this.baseUrl}/code/${runId}`);

    if (!response.ok) {
      throw new Error(`Failed to get code: ${response.statusText}`);
    }

    return response.json();
  }

  async checkHealth(): Promise<{ status: string }> {
    const response = await fetch(`${this.baseUrl}/health`);

    if (!response.ok) {
      throw new Error(`Health check failed: ${response.statusText}`);
    }

    return response.json();
  }
}

export const apiClient = new BgenAPI();
