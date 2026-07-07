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

function AgentNodeBox({ data }: { data: { icon: string; label: string; status: string; selected: boolean } }) {
  const ring: Record<string, string> = {
    running: "ring-2 ring-brand-400 ring-offset-2",
    done: "",
    error: "ring-2 ring-rose-400 ring-offset-2",
    pending: "",
  };
  const badge: Record<string, string> = {
    running: "bg-gradient-to-br from-brand-500 to-violet-600 text-white shadow-lg shadow-brand-200",
    done: "bg-emerald-500 text-white",
    error: "bg-rose-500 text-white",
    pending: "bg-slate-100 text-slate-400",
  };
  const label: Record<string, string> = {
    running: "text-brand-700",
    done: "text-slate-700",
    error: "text-rose-700",
    pending: "text-slate-400",
  };

  return (
    <div
      className={`flex flex-col items-center gap-2 px-2 py-1 rounded-2xl transition-all cursor-pointer ${
        data.selected ? "bg-brand-50/80" : ""
      }`}
    >
      <Handle type="target" position={Position.Left} className="opacity-0" />
      <div
        className={`relative w-14 h-14 rounded-2xl flex items-center justify-center text-xl transition-all ${badge[data.status]} ${ring[data.status]}`}
      >
        {data.status === "running" && (
          <span className="absolute inset-0 rounded-2xl bg-brand-400 animate-ping opacity-30" />
        )}
        <span className="relative">{data.icon}</span>
        {data.status === "done" && (
          <span className="absolute -bottom-1 -right-1 w-5 h-5 rounded-full bg-white flex items-center justify-center shadow-sm">
            <span className="w-3.5 h-3.5 rounded-full bg-emerald-500 flex items-center justify-center text-white text-[9px]">✓</span>
          </span>
        )}
      </div>
      <div className={`text-[11px] font-semibold tracking-wide ${label[data.status]}`}>{data.label}</div>
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
        position: { x: n.x * 132, y: n.key === "critic" ? 110 : 0 },
        data: { icon: n.icon, label: n.label, status: nodeStatus(i), selected: selected === n.key },
        draggable: false,
      })),
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [currentAgent, status, selected]
  );

  const retryActive = !!activeRetry;
  const edges: Edge[] = useMemo(() => {
    const base: Edge[] = [];
    for (let i = 0; i < AGENT_NODES.length - 1; i++) {
      const isActive = i === currentIndex - 1 && status === "running";
      base.push({
        id: `${AGENT_NODES[i].key}-${AGENT_NODES[i + 1].key}`,
        source: AGENT_NODES[i].key,
        target: AGENT_NODES[i + 1].key,
        animated: isActive,
        style: {
          stroke: isActive ? "#4f46e5" : "#cbd5e1",
          strokeWidth: isActive ? 2.5 : 2,
        },
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
        stroke: retryActive && activeRetry?.to === "features" ? "#e11d48" : "#e2e8f0",
        strokeWidth: retryActive && activeRetry?.to === "features" ? 3 : 1.5,
        strokeDasharray: "6 4",
      },
      label: retryActive && activeRetry?.to === "features" ? "↺ retry" : undefined,
      labelStyle: { fill: "#e11d48", fontWeight: 700, fontSize: 11 },
      labelBgStyle: { fill: "#fff1f2" },
    });
    base.push({
      id: "critic-train-retry",
      source: "critic",
      target: "experiments",
      type: "smoothstep",
      animated: retryActive && activeRetry?.to === "train",
      style: {
        stroke: retryActive && activeRetry?.to === "train" ? "#e11d48" : "#e2e8f0",
        strokeWidth: retryActive && activeRetry?.to === "train" ? 3 : 1.5,
        strokeDasharray: "6 4",
      },
      label: retryActive && activeRetry?.to === "train" ? "↺ retry" : undefined,
      labelStyle: { fill: "#e11d48", fontWeight: 700, fontSize: 11 },
      labelBgStyle: { fill: "#fff1f2" },
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
    <div className="h-72 bg-white rounded-2xl border border-slate-200 shadow-soft overflow-hidden">
      <ReactFlowProvider>
        <ReactFlow
          nodes={nodes}
          edges={edges}
          nodeTypes={nodeTypes}
          onNodeClick={handleNodeClick}
          fitView
          fitViewOptions={{ padding: 0.35 }}
          proOptions={{ hideAttribution: true }}
          nodesConnectable={false}
          elementsSelectable={true}
          panOnDrag={true}
          zoomOnScroll={false}
        >
          <Background variant={BackgroundVariant.Dots} gap={18} size={1} color="#e2e8f0" />
        </ReactFlow>
      </ReactFlowProvider>
    </div>
  );
}
