import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Separator } from "@/components/ui/separator";
import { Textarea } from "@/components/ui/textarea";
import React, { useEffect, useRef, useState } from "react";
import { toast } from "sonner";
import { apiClient, RunConfig } from "../services/api";
import { WebSocketService, getRenderWebSocketUrl } from "../services/websocket";

interface ProgressMessage {
  type:
    | "step"
    | "episode_summary"
    | "critique"
    | "done"
    | "error"
    | "connected";
  step?: any;
  summary?: any;
  critique?: string;
  aggregate?: any;
  message?: string;
  timing?: any;
}

export const Sidebar: React.FC = () => {
  const [userGoal, setUserGoal] = useState(
    "Reach target while minimizing collisions"
  );
  const [iterations, setIterations] = useState(1);
  const [episodes, setEpisodes] = useState(3);
  const [duration, setDuration] = useState(30);
  const [provider, setProvider] = useState<"anthropic" | "openai">("openai");
  const [model, setModel] = useState("");
  const [isRunning, setIsRunning] = useState(false);
  const [runId, setRunId] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [progress, setProgress] = useState<string>("");
  const [currentIteration, setCurrentIteration] = useState(0);
  const [currentEpisode, setCurrentEpisode] = useState(0);
  const [generatedCode, setGeneratedCode] = useState<string>("");
  const [currentStep, setCurrentStep] = useState<any>(null);
  const [lastEpisodeSummary, setLastEpisodeSummary] = useState<any>(null);
  const [lastCritique, setLastCritique] = useState<string>("");
  const [aggregateStats, setAggregateStats] = useState<any>(null);
  const wsRef = useRef<WebSocketService | null>(null);
  const renderWsRefs = useRef<WebSocketService[]>([]);
  const lastToastRef = useRef<{ [key: string]: number }>({});

  useEffect(() => {
    if (runId && isRunning) {
      const progressWsUrl = `ws://localhost:8089/ws/progress/${runId}`;
      const progressWs = new WebSocketService(progressWsUrl);

      // Connect to render WebSocket channels
      const renderChannels = ['world', 'agents', 'pov_world_0', 'pov_agents_0'];
      renderChannels.forEach(channel => {
        const renderWsUrl = getRenderWebSocketUrl(runId, channel);
        const renderWs = new WebSocketService(renderWsUrl);

        renderWs.connect((event) => {
          try {
            const data = JSON.parse(event.data);
            if (data.type === 'frame') {
              console.log(`Received ${channel} frame:`, data.len, 'bytes');
              // Binary data is in data.payload_b64 (base64 encoded)
              // This will be processed by the WASM renderer
            }
          } catch (err) {
            console.error(`Failed to parse ${channel} frame:`, err);
          }
        }).catch((err) => {
          console.error(`Failed to connect to ${channel} WebSocket:`, err);
        });

        renderWsRefs.current.push(renderWs);
      });

      progressWs
        .connect((event) => {
          try {
            const data: ProgressMessage = JSON.parse(event.data);
            console.log("Progress update:", data);

            switch (data.type) {
              case "connected":
                setProgress("Connected to training pipeline...");
                // Only show connected toast once
                const now = Date.now();
                if (!lastToastRef.current.connected || now - lastToastRef.current.connected > 5000) {
                  toast.success("Connected to training pipeline");
                  lastToastRef.current.connected = now;
                }
                break;
              case "step":
                setCurrentStep(data.step);
                setProgress(`Step ${data.step?.t?.toFixed(2)}s`);
                // Only show toast every 5 seconds to avoid spam
                if (data.step?.t) {
                  const stepKey = `step_${Math.floor(data.step.t / 5) * 5}`;
                  const now = Date.now();
                  if (!lastToastRef.current[stepKey] || now - lastToastRef.current[stepKey] > 4000) {
                    if (Math.floor(data.step.t) % 5 === 0 && data.step.t > 0) {
                      toast.info(`Simulation running: ${data.step.t.toFixed(1)}s`, {
                        description: data.step.distance_to_target
                          ? `Distance: ${data.step.distance_to_target.toFixed(2)}m`
                          : undefined,
                        id: stepKey
                      });
                      lastToastRef.current[stepKey] = now;
                    }
                  }
                }
                break;
              case "episode_summary":
                setCurrentEpisode((prev) => prev + 1);
                setLastEpisodeSummary(data.summary);
                setProgress(`Episode ${currentEpisode + 1} completed - ${data.summary?.success ? 'Success' : 'Failed'}`);
                if (data.summary?.success) {
                  toast.success(`Episode ${currentEpisode + 1} completed!`, {
                    description: `Steps: ${data.summary.steps}, Collisions: ${data.summary.collisions_total}`
                  });
                } else {
                  toast.error(`Episode ${currentEpisode + 1} failed`, {
                    description: `Steps: ${data.summary.steps}, Collisions: ${data.summary.collisions_total}`
                  });
                }
                break;
              case "critique":
                setCurrentIteration((prev) => prev + 1);
                setLastCritique(data.critique || "");
                setAggregateStats(data.aggregate);
                setProgress(
                  `Iteration ${currentIteration + 1} complete. Success rate: ${
                    (data.aggregate?.success_rate * 100 || 0).toFixed(1)
                  }%`
                );
                toast.info(`Iteration ${currentIteration + 1} complete`, {
                  description: `Success rate: ${(data.aggregate?.success_rate * 100 || 0).toFixed(1)}%`
                });
                break;
              case "done":
                setProgress(
                  `Training complete! ${
                    data.timing
                      ? `Speedup: ${data.timing.speedup_x?.toFixed(2)}x`
                      : ""
                  }`
                );
                setIsRunning(false);
                toast.success("Training complete!", {
                  description: data.timing
                    ? `Speedup: ${data.timing.speedup_x?.toFixed(2)}x`
                    : undefined
                });
                // Fetch the generated code when training is done
                if (runId) {
                  fetchGeneratedCode(runId);
                }
                break;
              case "error":
                setError(data.message || "Unknown error");
                setIsRunning(false);
                toast.error("Training failed", {
                  description: data.message || "Unknown error"
                });
                break;
            }
          } catch (err) {
            console.error("Failed to parse progress message:", err);
          }
        })
        .catch((err) => {
          console.error("Failed to connect to progress WebSocket:", err);
        });

      wsRef.current = progressWs;

      return () => {
        progressWs.disconnect();
        renderWsRefs.current.forEach(ws => ws.disconnect());
        renderWsRefs.current = [];
      };
    }
  }, [runId, isRunning]);

  const handleStartRun = async () => {
    setError(null);
    setIsRunning(true);
    setProgress("Starting training...");
    setCurrentIteration(0);
    setCurrentEpisode(0);

    try {
      const config: RunConfig = {
        user_goal: userGoal,
        iterations,
        episodes,
        render: true,
        duration_seconds: duration,
        use_px4: true,
        provider: provider,
        model: model || undefined,
      };

      const response = await apiClient.startRun(config);
      setRunId(response.run_id);
      console.log("Run started with ID:", response.run_id);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to start run");
      setIsRunning(false);
      setProgress("");
    }
  };

  const handleStopRun = () => {
    if (wsRef.current) {
      wsRef.current.send({ type: "terminate" });
    }
    setIsRunning(false);
    setProgress("Stopped");
  };

  const fetchGeneratedCode = async (id: string) => {
    try {
      const response = await apiClient.getCode(id);
      setGeneratedCode(response.code);
    } catch (err) {
      console.error("Failed to fetch generated code:", err);
    }
  };

  return (
    <div className="fixed right-0 top-0 w-80 h-full bg-background border-l border-border overflow-y-auto z-50 p-3 pointer-events-auto">
      <div className="flex flex-col gap-4">
        <div className="flex flex-col gap-4">
          <div>
            <h3 className="text-md font-medium px-1">Training Setup</h3>
          </div>

          <div className="flex flex-col gap-2">
            <Label htmlFor="goal" className="px-1">
              Mission Goal
            </Label>
            <Textarea
              id="goal"
              value={userGoal}
              onChange={(e) => setUserGoal(e.target.value)}
              disabled={isRunning}
              rows={3}
              placeholder="Describe the agent's objective..."
            />
          </div>

          <div className="grid grid-cols-2 gap-4">
            <div className="flex flex-col gap-2">
              <Label htmlFor="iterations" className="px-1">
                Iterations
              </Label>
              <Input
                id="iterations"
                type="number"
                value={iterations}
                onChange={(e) => setIterations(parseInt(e.target.value) || 1)}
                disabled={isRunning}
                min={1}
                max={10}
              />
            </div>

            <div className="flex flex-col gap-2">
              <Label htmlFor="episodes" className="px-1">
                Episodes
              </Label>
              <Input
                id="episodes"
                type="number"
                value={episodes}
                onChange={(e) => setEpisodes(parseInt(e.target.value) || 1)}
                disabled={isRunning}
                min={1}
                max={20}
              />
            </div>
          </div>

          <div className="flex flex-col gap-2">
            <Label htmlFor="duration" className="px-1">
              Duration (seconds)
            </Label>
            <Input
              id="duration"
              type="number"
              value={duration}
              onChange={(e) => setDuration(parseInt(e.target.value) || 30)}
              disabled={isRunning}
              min={10}
              max={300}
            />
          </div>

          <Separator />

          <div className="flex flex-col gap-2">
            <Label htmlFor="provider" className="px-1">
              LLM Provider
            </Label>
            <select
              id="provider"
              value={provider}
              onChange={(e) => setProvider(e.target.value as "anthropic" | "openai")}
              disabled={isRunning}
              className="flex h-10 w-full rounded-md border border-input bg-background px-3 py-2 text-sm ring-offset-background focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:cursor-not-allowed disabled:opacity-50"
            >
              <option value="openai">OpenAI (GPT-4o-mini)</option>
              <option value="anthropic">Anthropic (Claude)</option>
            </select>
          </div>

          <div className="flex flex-col gap-2">
            <Label htmlFor="model" className="px-1">
              Model (optional)
            </Label>
            <Input
              id="model"
              type="text"
              value={model}
              onChange={(e) => setModel(e.target.value)}
              disabled={isRunning}
              placeholder={provider === "openai" ? "gpt-4o-mini" : "claude-3-5-sonnet-20241022"}
            />
          </div>

          {!isRunning ? (
            <Button onClick={handleStartRun} className="w-full" size="lg">
              Start Training
            </Button>
          ) : (
            <Button
              onClick={handleStopRun}
              className="w-full"
              variant="destructive"
              size="lg"
            >
              Stop
            </Button>
          )}

          {error && (
            <div className="p-3 text-sm text-destructive bg-destructive/10 border border-destructive rounded-xl">
              {error}
            </div>
          )}
        </div>

        <Separator />

        <div className="flex flex-col gap-2">
          <h3 className="text-md font-regular">Status</h3>
          {runId ? (
            <>
              <div className="flex items-center gap-2">
                <div
                  className={`h-2 w-2 rounded-full ${
                    isRunning ? "bg-green-500 animate-pulse" : "bg-gray-400"
                  }`}
                />
                <span className="text-sm font-medium">
                  {isRunning ? "Running" : "Stopped"}
                </span>
              </div>
              {(currentIteration > 0 || currentEpisode > 0) && (
                <p className="text-sm text-muted-foreground">
                  Progress: Iter {currentIteration}/{iterations}, Ep{" "}
                  {currentEpisode}/{episodes}
                </p>
              )}
              {progress && (
                <p className="text-sm text-primary font-medium italic">
                  {progress}
                </p>
              )}
            </>
          ) : (
            <p className="text-sm text-muted-foreground">Ready to train</p>
          )}
        </div>

        {currentStep && (
          <>
            <Separator />
            <div className="flex flex-col gap-2">
              <h3 className="text-md font-regular">Real-time Metrics</h3>
              <div className="text-xs bg-muted p-3 rounded-md space-y-1">
                <div><span className="font-medium">Time:</span> {currentStep.t?.toFixed(2)}s</div>
                <div><span className="font-medium">Position:</span> ({currentStep.agent_pos?.[0]?.toFixed(1)}, {currentStep.agent_pos?.[1]?.toFixed(1)}, {currentStep.agent_pos?.[2]?.toFixed(1)})</div>
                <div><span className="font-medium">Coord:</span> ({currentStep.agent_coord?.[0]}, {currentStep.agent_coord?.[1]}, {currentStep.agent_coord?.[2]})</div>
                <div><span className="font-medium">Collisions:</span> {currentStep.collisions_count}</div>
                {currentStep.distance_to_target != null && (
                  <div><span className="font-medium">Distance to Target:</span> {currentStep.distance_to_target?.toFixed(2)}</div>
                )}
                <div><span className="font-medium">Frame Time:</span> {currentStep.frame_time_ms?.toFixed(1)}ms</div>
              </div>
            </div>
          </>
        )}

        {lastEpisodeSummary && (
          <>
            <Separator />
            <div className="flex flex-col gap-2">
              <h3 className="text-md font-regular">Episode Summary</h3>
              <div className="text-xs bg-muted p-3 rounded-md space-y-1">
                <div><span className="font-medium">Status:</span> {lastEpisodeSummary.success ? '✓ Success' : '✗ Failed'}</div>
                <div><span className="font-medium">Steps:</span> {lastEpisodeSummary.steps}</div>
                <div><span className="font-medium">Total Collisions:</span> {lastEpisodeSummary.collisions_total}</div>
                {lastEpisodeSummary.path_len_est != null && (
                  <div><span className="font-medium">Path Length:</span> {lastEpisodeSummary.path_len_est?.toFixed(2)}</div>
                )}
                {lastEpisodeSummary.avg_command_len != null && (
                  <div><span className="font-medium">Avg Command Len:</span> {lastEpisodeSummary.avg_command_len?.toFixed(2)}</div>
                )}
              </div>
            </div>
          </>
        )}

        {aggregateStats && (
          <>
            <Separator />
            <div className="flex flex-col gap-2">
              <h3 className="text-md font-regular">Aggregate Statistics</h3>
              <div className="text-xs bg-muted p-3 rounded-md space-y-1">
                <div><span className="font-medium">Success Rate:</span> {(aggregateStats.success_rate * 100).toFixed(1)}%</div>
                {aggregateStats.avg_steps != null && (
                  <div><span className="font-medium">Avg Steps:</span> {aggregateStats.avg_steps?.toFixed(1)}</div>
                )}
                {aggregateStats.avg_collisions != null && (
                  <div><span className="font-medium">Avg Collisions:</span> {aggregateStats.avg_collisions?.toFixed(1)}</div>
                )}
                {aggregateStats.avg_path_len != null && (
                  <div><span className="font-medium">Avg Path Length:</span> {aggregateStats.avg_path_len?.toFixed(2)}</div>
                )}
              </div>
            </div>
          </>
        )}

        {lastCritique && (
          <>
            <Separator />
            <div className="flex flex-col gap-2">
              <h3 className="text-md font-regular">LLM Critique</h3>
              <div className="text-xs bg-muted p-3 rounded-md max-h-48 overflow-y-auto whitespace-pre-wrap">
                {lastCritique}
              </div>
            </div>
          </>
        )}

        {generatedCode && (
          <>
            <Separator />
            <div className="flex flex-col gap-2">
              <h3 className="text-md font-regular">Generated Policy</h3>
              <pre className="text-xs bg-muted p-3 rounded-md overflow-x-auto max-h-96 overflow-y-auto">
                <code>{generatedCode}</code>
              </pre>
            </div>
          </>
        )}
      </div>
    </div>
  );
};
