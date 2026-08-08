'use client';

import { useState } from 'react';
import { Button } from '@/components/ui/button';

interface MicErrorBannerProps {
  message?: string;
  onClear?: () => void;
}

export function MicErrorBanner({ message, onClear }: MicErrorBannerProps) {
  const [busy, setBusy] = useState(false);

  if (!message) return null;

  const handleRetry = async () => {
    setBusy(true);
    try {
      await navigator.mediaDevices.getUserMedia({ audio: true });
      // got permission — reload to allow app to initialize devices
      window.location.reload();
    } catch (e) {
      // still blocked
      console.error(e);
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="fixed left-1/2 top-20 z-60 w-[min(680px,calc(100%-32px))] -translate-x-1/2 rounded-lg bg-destructive/5 p-3 shadow-md">
      <div className="flex items-start justify-between gap-4">
        <div>
          <div className="font-semibold text-destructive">Microphone access denied</div>
          <div className="text-sm text-muted-foreground mt-1">{message}</div>
          <div className="text-sm mt-2">Try: allow microphone in your browser, check OS privacy settings, or retry below.</div>
        </div>

        <div className="flex items-start gap-2">
          <Button onClick={handleRetry} disabled={busy} size="sm">
            {busy ? 'Retrying…' : 'Retry microphone'}
          </Button>
          <Button variant="ghost" onClick={onClear} size="sm">
            Dismiss
          </Button>
        </div>
      </div>
    </div>
  );
}
