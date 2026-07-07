"use client";

import { useMemo, useState } from "react";
import ReactFlow, {
  Background, BackgroundVariant, Edge, Handle, Node, Position, ReactFlowProvider,
} from "reactflow";
import "reactflow/dist/style.css";
import { TraceEvent } from "@/lib/useJobStream";

const AGENT_NODES = [
  { key: "eda", label: "EDA", icon: "🔍", x: 0 },
  { key: "planner", label: "Planner", icon: "🧠", x: 1 },
  { key: "features", label: "Features", icon: "🔧", x: 2 },
  { key: "experiments", label: "Train", icon: "🏋️", x: 3 },
  { key: "critic", label: "Critic", icon: "🔬", x: 4 },
  { key: "evaluation", label: "Evaluation", icon: "📊", x: 5 },
  { key: "report", label: "Report", icon: "📄", x: 6 },
];

interface Props {
  currentAgent: string;
  status: "idle" | "running" | "completed" | "failed";
  trace: TraceEvent[];
  activeRetry: { from: string; to: string; reason: string } | null;
  onSelectAgent?: (agentKey: string | null) => void;
}

function AgentNodeBox({ data }: { data: { icon: string; label: string; status: string } }) {
  const colors: Record<string, string> = {
    running: "border-blue-400 bg-blue-50 text-blue-800 shadow-blue-200",
    done: "border-green-400 bg-green-50 text-green-800",
    error: "border-red-400 bg-red-50 text-red-800",
    pending: "border-gray-200 bg-gray-50 text-gray-400",
  };
  return (
    <div className={`px-4 py-2 rounded-xl border-2 shadow-sm text-center min-w-[110px] transition-all ${colors[data.status]}`}>
      <Handle type="target" position={Position.Left} className="opacity-0" />
      <div className="text-lg leading-none mb-1">
        {data.status === "running" ? (
          <span className="inline-block animate-pulse">{data.icon}</span>
        ) : (
          data.icon
        )}
      </div>
      <div className="text-xs font-semibold">{data.label}</div>
      <Handle type="source" position={Position.Right} className="opacity-0" />
    </div>
  );
}

const nodeTypes = { agentNode: AgentNodeBox };

export function AgentGraph({ currentAgent, status, trace, activeRetry, onSelectAgent }: Props) {
  const [selected, setSelected] = useState<string | null>(null);
  const currentIndex = AGENT_NODES.findIndex((n) => n.key === currentAgent);

  function nodeStatus(index: number): string {
    if (status === "failed" && index === currentIndex) return "error";
    if (status === "completed") return "done";
    if (index < currentIndex) return "done";
    if (index === currentIndex) return "running";
    return "pending";
  }

  const nodes: Node[] = useMemo(
    () =>
      AGENT_NODES.map((n, i) => ({
        id: n.key,
        type: "agentNode",
        position: { x: n.x * 150, y: n.key === "critic" ? 90 : 0 },
        data: { icon: n.icon, label: n.label, status: nodeStatus(i) },
        draggable: false,
      })),
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [currentAgent, status]
  );

  const retryActive = !!activeRetry;
  const edges: Edge[] = useMemo(() => {
    const base: Edge[] = [];
    for (let i = 0; i < AGENT_NODES.length - 1; i++) {
      base.push({
        id: `${AGENT_NODES[i].key}-${AGENT_NODES[i + 1].key}`,
        source: AGENT_NODES[i].key,
        target: AGENT_NODES[i + 1].key,
        animated: i === currentIndex - 1 && status === "running",
        style: { stroke: "#94a3b8", strokeWidth: 2 },
      });
    }
    // Loop-back edges — only visually emphasized while a retry is actually in flight
    base.push({
      id: "critic-features-retry",
      source: "critic",
      target: "features",
      type: "smoothstep",
      animated: retryActive && activeRetry?.to === "features",
      style: {
        stroke: retryActive && activeRetry?.to === "features" ? "#ef4444" : "#e5e7eb",
        strokeWidth: retryActive && activeRetry?.to === "features" ? 3 : 1.5,
        strokeDasharray: "6 4",
      },
      label: retryActive && activeRetry?.to === "features" ? "retry" : undefined,
    });
    base.push({
      id: "critic-train-retry",
      source: "critic",
      target: "experiments",
      type: "smoothstep",
      animated: retryActive && activeRetry?.to === "train",
      style: {
        stroke: retryActive && activeRetry?.to === "train" ? "#ef4444" : "#e5e7eb",
        strokeWidth: retryActive && activeRetry?.to === "train" ? 3 : 1.5,
        strokeDasharray: "6 4",
      },
      label: retryActive && activeRetry?.to === "train" ? "retry" : undefined,
    });
    return base;
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [currentIndex, status, retryActive, activeRetry]);

  function handleNodeClick(_: any, node: Node) {
    const next = selected === node.id ? null : node.id;
    setSelected(next);
    onSelectAgent?.(next);
  }

  return (
    <div className="h-64 bg-white rounded-xl border border-gray-200 overflow-hidden">
      <ReactFlowProvider>
        <ReactFlow
          nodes={nodes}
          edges={edges}
          nodeTypes={nodeTypes}
          onNodeClick={handleNodeClick}
          fitView
          fitViewOptions={{ padding: 0.3 }}
          proOptions={{ hideAttribution: true }}
          nodesConnectable={false}
          elementsSelectable={true}
          panOnDrag={true}
          zoomOnScroll={false}
        >
          <Background variant={BackgroundVariant.Dots} gap={16} size={1} color="#e5e7eb" />
        </ReactFlow>
      </ReactFlowProvider>
    </div>
  );
}
