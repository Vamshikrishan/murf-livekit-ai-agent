'use client';

import { useEffect, useState } from 'react';
import { useAgent, useSessionContext } from '@livekit/components-react';

interface StatusIndicatorProps {
  className?: string;
}

export function StatusIndicator({ className }: StatusIndicatorProps) {
  const session = useSessionContext();
  const isConnected = session.isConnected;
  const isConnecting = session.connectionState === 'connecting';
  const { state: agentState } = useAgent();
  const [lastConnected, setLastConnected] = useState(false);

  useEffect(() => {
    if (isConnected) setLastConnected(true);
  }, [isConnected]);

  const renderState = () => {
    if (!isConnected && !isConnecting && !lastConnected) return { label: 'Ready', color: 'bg-green-100 text-green-800' };
    if (isConnecting) return { label: 'Connecting', color: 'bg-yellow-100 text-yellow-800' };
    if (isConnected && (agentState === 'listening' || agentState === 'idle'))
      return { label: 'Listening', color: 'bg-blue-100 text-blue-800' };
    if (isConnected && agentState === 'thinking') return { label: 'Thinking', color: 'bg-purple-100 text-purple-800' };
    if (isConnected && agentState === 'speaking') return { label: 'Speaking', color: 'bg-indigo-100 text-indigo-800' };
    if (!isConnected && lastConnected) return { label: 'Call ended', color: 'bg-red-100 text-red-800' };

    return { label: 'Idle', color: 'bg-muted-foreground text-foreground' };
  };

  const { label, color } = renderState();

  return (
    <div className={`rounded-lg border p-2 pr-3 shadow-md ${className ?? ''}`}>
      <div className="flex items-center gap-3">
        <div className={`rounded-full size-3 ${color}`} aria-hidden />
        <div className="text-sm font-semibold">{label}</div>
      </div>
      <div className="mt-1 text-xs text-muted-foreground">{agentState ?? '—'}</div>
    </div>
  );
}
