'use client';

import React, { useCallback, useEffect, useRef, useState } from 'react';
import { AnimatePresence, type MotionProps, motion } from 'motion/react';
import { Track } from 'livekit-client';
import {
  useAgent,
  useLocalParticipant,
  useRoomContext,
  useSessionContext,
  useSessionMessages,
} from '@livekit/components-react';
import { AgentChatTranscript } from '@/components/agents-ui/agent-chat-transcript';
import {
  AgentControlBar,
  type AgentControlBarControls,
} from '@/components/agents-ui/agent-control-bar';
import { StatusIndicator } from '@/components/app/status-indicator';
import { MicErrorBanner } from '@/components/app/mic-error-banner';
import { Shimmer } from '@/components/ai-elements/shimmer';
import { cn } from '@/lib/shadcn/utils';
import { TileLayout } from './tile-view';

const MotionMessage = motion.create(Shimmer);

const BOTTOM_VIEW_MOTION_PROPS: MotionProps = {
  variants: {
    visible: {
      opacity: 1,
      translateY: '0%',
    },
    hidden: {
      opacity: 0,
      translateY: '100%',
    },
  },
  initial: 'hidden',
  animate: 'visible',
  exit: 'hidden',
  transition: {
    duration: 0.3,
    delay: 0.5,
    ease: 'easeOut',
  },
};

const CHAT_MOTION_PROPS: MotionProps = {
  variants: {
    hidden: {
      opacity: 0,
      transition: {
        ease: 'easeOut',
        duration: 0.3,
      },
    },
    visible: {
      opacity: 1,
      transition: {
        delay: 0.2,
        ease: 'easeOut',
        duration: 0.3,
      },
    },
  },
  initial: 'hidden',
  animate: 'visible',
  exit: 'hidden',
};

const SHIMMER_MOTION_PROPS: MotionProps = {
  variants: {
    visible: {
      opacity: 1,
      transition: {
        ease: 'easeIn',
        duration: 0.5,
        delay: 0.8,
      },
    },
    hidden: {
      opacity: 0,
      transition: {
        ease: 'easeIn',
        duration: 0.5,
        delay: 0,
      },
    },
  },
  initial: 'hidden',
  animate: 'visible',
  exit: 'hidden',
};

interface FadeProps {
  top?: boolean;
  bottom?: boolean;
  className?: string;
}

export function Fade({ top = false, bottom = false, className }: FadeProps) {
  return (
    <div
      className={cn(
        'from-background pointer-events-none h-4 bg-linear-to-b to-transparent',
        top && 'bg-linear-to-b',
        bottom && 'bg-linear-to-t',
        className
      )}
    />
  );
}

// ─── Mic status badge ────────────────────────────────────────────────────────
interface MicStatusBadgeProps {
  enabled: boolean;
  pending: boolean;
}

function MicStatusBadge({ enabled, pending }: MicStatusBadgeProps) {
  if (pending) {
    return (
      <span className="inline-flex items-center gap-1 rounded-full bg-yellow-500/15 px-2 py-0.5 text-xs font-semibold text-yellow-600 dark:text-yellow-400">
        <span className="h-1.5 w-1.5 animate-pulse rounded-full bg-yellow-500" />
        MIC STARTING…
      </span>
    );
  }
  if (enabled) {
    return (
      <span className="inline-flex items-center gap-1 rounded-full bg-green-500/15 px-2 py-0.5 text-xs font-semibold text-green-700 dark:text-green-400">
        <span className="h-1.5 w-1.5 rounded-full bg-green-500" />
        MIC ON
      </span>
    );
  }
  return (
    <span className="inline-flex items-center gap-1 rounded-full bg-destructive/10 px-2 py-0.5 text-xs font-semibold text-destructive">
      <span className="h-1.5 w-1.5 rounded-full bg-destructive" />
      MIC OFF
    </span>
  );
}

export interface AgentSessionView_01Props {
  /**
   * Message shown above the controls before the first chat message is sent.
   *
   * @default 'Agent is listening, ask it a question'
   */
  preConnectMessage?: string;
  /**
   * Enables or disables the chat toggle and transcript input controls.
   *
   * @default true
   */
  supportsChatInput?: boolean;
  /**
   * Enables or disables camera controls in the bottom control bar.
   *
   * @default true
   */
  supportsVideoInput?: boolean;
  /**
   * Enables or disables screen sharing controls in the bottom control bar.
   *
   * @default true
   */
  supportsScreenShare?: boolean;
  /**
   * Shows a pre-connect buffer state with a shimmer message before messages appear.
   *
   * @default true
   */
  isPreConnectBufferEnabled?: boolean;

  /** Selects the visualizer style rendered in the main tile area. */
  audioVisualizerType?: 'bar' | 'wave' | 'grid' | 'radial' | 'aura';
  /** Primary hex color used by supported audio visualizer variants. */
  audioVisualizerColor?: `#${string}`;
  /** Hue shift intensity used by certain visualizers. */
  audioVisualizerColorShift?: number;
  /** Number of bars to render when `audioVisualizerType` is `bar`. */
  audioVisualizerBarCount?: number;
  /** Number of rows in the visualizer when `audioVisualizerType` is `grid`. */
  audioVisualizerGridRowCount?: number;
  /** Number of columns in the visualizer when `audioVisualizerType` is `grid`. */
  audioVisualizerGridColumnCount?: number;
  /** Number of radial bars when `audioVisualizerType` is `radial`. */
  audioVisualizerRadialBarCount?: number;
  /** Base radius of the radial visualizer when `audioVisualizerType` is `radial`. */
  audioVisualizerRadialRadius?: number;
  /** Stroke width of the wave path when `audioVisualizerType` is `wave`. */
  audioVisualizerWaveLineWidth?: number;
  /** Optional class name merged onto the outer `<section>` container. */
  className?: string;
}

export function AgentSessionView_01({
  preConnectMessage = 'Agent is listening, ask it a question',
  supportsChatInput = true,
  supportsVideoInput = true,
  supportsScreenShare = true,
  isPreConnectBufferEnabled = true,

  audioVisualizerType,
  audioVisualizerColor,
  audioVisualizerColorShift,
  audioVisualizerBarCount,
  audioVisualizerGridRowCount,
  audioVisualizerGridColumnCount,
  audioVisualizerRadialBarCount,
  audioVisualizerRadialRadius,
  audioVisualizerWaveLineWidth,
  ref,
  className,
  ...props
}: React.ComponentProps<'section'> & AgentSessionView_01Props) {
  const session = useSessionContext();
  const { messages } = useSessionMessages(session);
  const [chatOpen, setChatOpen] = useState(false);
  const scrollAreaRef = useRef<HTMLDivElement>(null);
  const { state: agentState } = useAgent();
  const [micError, setMicError] = useState<string | undefined>(undefined);

  // ── Mic auto-enable state ──────────────────────────────────────────────────
  const { localParticipant } = useLocalParticipant();
  const room = useRoomContext();
  const [micEnabled, setMicEnabled] = useState(false);
  const [micPending, setMicPending] = useState(false);
  // Prevent re-triggering the auto-enable once it has run successfully
  const micAutoEnabledRef = useRef(false);

  // ── Diagnostic logger (safe — no credentials logged) ──────────────────────
  const logMicDiagnostics = useCallback(() => {
    if (typeof window === 'undefined') return;
    const participant = localParticipant;
    const micPub = participant?.getTrackPublication(Track.Source.Microphone);
    const audioTracks = participant?.audioTrackPublications;

    console.group('[AarogyaMitra] Microphone Diagnostics');
    console.log('Room name       :', room?.name ?? 'N/A');
    console.log('Room state      :', room?.state ?? 'N/A');
    console.log('Participant ID  :', participant?.identity ?? 'N/A');
    console.log('Mic enabled     :', participant?.isMicrophoneEnabled ?? false);
    console.log('Mic publication :', micPub ? `SID=${micPub.trackSid ?? 'pending'}` : 'none');
    console.log('Mic muted       :', micPub?.isMuted ?? 'N/A');
    console.log('Audio tracks    :', audioTracks?.size ?? 0);
    // Check browser permission without logging any secrets
    if (typeof navigator !== 'undefined' && navigator.permissions) {
      navigator.permissions.query({ name: 'microphone' as PermissionName }).then((result) => {
        console.log('Browser mic perm:', result.state); // 'granted' | 'denied' | 'prompt'
        console.groupEnd();
      }).catch(() => {
        console.log('Browser mic perm: (unavailable)');
        console.groupEnd();
      });
    } else {
      console.groupEnd();
    }
  }, [localParticipant, room]);

  // ── Auto-enable microphone once the room is connected ─────────────────────
  // This is the core fix: LiveKit does not auto-publish the mic track unless
  // the user has a saved preference (usePersistentUserChoices). On a fresh
  // browser session with no localStorage the mic stays off. We explicitly
  // call setMicrophoneEnabled(true) the first time the room connects so that
  // real audio reaches Deepgram STT on the backend.
  useEffect(() => {
    if (!session.isConnected) return;
    if (micAutoEnabledRef.current) return;
    if (!localParticipant) return;

    // If the mic is already on (saved preference restored), just reflect that
    if (localParticipant.isMicrophoneEnabled) {
      micAutoEnabledRef.current = true;
      setMicEnabled(true);
      console.log('[AarogyaMitra] Microphone already enabled — skipping auto-enable');
      logMicDiagnostics();
      return;
    }

    // Enable the microphone
    console.log('[AarogyaMitra] Session connected — enabling microphone…');
    setMicPending(true);

    localParticipant
      .setMicrophoneEnabled(true)
      .then(() => {
        micAutoEnabledRef.current = true;
        setMicEnabled(true);
        setMicPending(false);
        console.log('[AarogyaMitra] Microphone enabled and track published successfully');
        logMicDiagnostics();
      })
      .catch((err: Error) => {
        setMicPending(false);
        setMicEnabled(false);
        const name = (err as any)?.name ?? '';
        const msg = err?.message ?? String(err);
        console.error('[AarogyaMitra] Failed to enable microphone:', name, msg);
        if (name === 'NotAllowedError' || /permission|denied/i.test(msg)) {
          setMicError(
            'Microphone permission was denied. ' +
            'Please allow microphone access in your browser and reload the page to speak with AarogyaMitra.'
          );
        } else if (name === 'NotFoundError' || /device|found/i.test(msg)) {
          setMicError(
            'No microphone was found on this device. ' +
            'Please connect a microphone and reload the page.'
          );
        } else {
          setMicError(`Microphone could not be started: ${msg}`);
        }
        logMicDiagnostics();
      });
  }, [session.isConnected, localParticipant, logMicDiagnostics]);

  // ── Keep micEnabled in sync with the actual track state ──────────────────
  // This handles the user manually toggling the mic button in the control bar.
  useEffect(() => {
    if (!localParticipant) return;
    setMicEnabled(localParticipant.isMicrophoneEnabled);
  }, [localParticipant, localParticipant?.isMicrophoneEnabled]);

  const controls: AgentControlBarControls = {
    leave: true,
    microphone: true,
    chat: supportsChatInput,
    camera: supportsVideoInput,
    screenShare: supportsScreenShare,
  };

  useEffect(() => {
    const lastMessage = messages.at(-1);
    const lastMessageIsLocal = lastMessage?.from?.isLocal === true;

    if (scrollAreaRef.current && lastMessageIsLocal) {
      scrollAreaRef.current.scrollTop = scrollAreaRef.current.scrollHeight;
    }
  }, [messages]);

  return (
    <section
      ref={ref}
      className={cn('bg-background relative z-10 h-full w-full overflow-hidden', className)}
      {...props}
    >
      <StatusIndicator className="absolute top-6 right-6 z-60" />
      <MicErrorBanner message={micError} onClear={() => setMicError(undefined)} />
      <Fade top className="absolute inset-x-4 top-0 z-10 h-40" />
      {/* transcript */}

      <div className="absolute top-0 bottom-[135px] flex w-full flex-col md:bottom-[170px]">
        <AnimatePresence>
          {chatOpen && (
            <motion.div
              {...CHAT_MOTION_PROPS}
              className="flex h-full w-full flex-col gap-4 space-y-3 transition-opacity duration-300 ease-out"
            >
              <AgentChatTranscript
                agentState={agentState}
                messages={messages}
                className="mx-auto w-full max-w-2xl [&_.is-user>div]:rounded-[22px] [&>div>div]:px-4 [&>div>div]:pt-40 md:[&>div>div]:px-6"
              />
            </motion.div>
          )}
        </AnimatePresence>
      </div>
      {/* Tile layout */}
      <TileLayout
        chatOpen={chatOpen}
        audioVisualizerType={audioVisualizerType}
        audioVisualizerColor={audioVisualizerColor}
        audioVisualizerColorShift={audioVisualizerColorShift}
        audioVisualizerBarCount={audioVisualizerBarCount}
        audioVisualizerRadialBarCount={audioVisualizerRadialBarCount}
        audioVisualizerRadialRadius={audioVisualizerRadialRadius}
        audioVisualizerGridRowCount={audioVisualizerGridRowCount}
        audioVisualizerGridColumnCount={audioVisualizerGridColumnCount}
        audioVisualizerWaveLineWidth={audioVisualizerWaveLineWidth}
      />
      {/* Bottom */}
      <motion.div
        {...BOTTOM_VIEW_MOTION_PROPS}
        className="absolute inset-x-3 bottom-0 z-50 md:inset-x-12"
      >
        {/* Pre-connect message */}
        {isPreConnectBufferEnabled && (
          <AnimatePresence>
            {messages.length === 0 && (
              <MotionMessage
                key="pre-connect-message"
                duration={2}
                aria-hidden={messages.length > 0}
                {...SHIMMER_MOTION_PROPS}
                className="pointer-events-none mx-auto block w-full max-w-2xl pb-4 text-center text-sm font-semibold"
              >
                {preConnectMessage}
              </MotionMessage>
            )}
          </AnimatePresence>
        )}
        <div className="bg-background relative mx-auto max-w-2xl pb-3 md:pb-12">
          <Fade bottom className="absolute inset-x-0 top-0 h-4 -translate-y-full" />

          {/* Mic status badge — shown above the control bar */}
          <div className="mb-2 flex justify-center">
            <MicStatusBadge enabled={micEnabled} pending={micPending} />
          </div>

          <AgentControlBar
            variant="livekit"
            controls={controls}
            isChatOpen={chatOpen}
            isConnected={session.isConnected}
            onDisconnect={session.end}
            onDeviceError={(err) => {
              if (err.source === Track.Source.Microphone) {
                setMicEnabled(false);
                setMicPending(false);
                const name = (err.error as any)?.name ?? '';
                const msg = err.error?.message ?? String(err.error ?? '');
                if (name === 'NotAllowedError' || /permission/i.test(msg)) {
                  setMicError(
                    msg ||
                    'Microphone permission denied. Please allow microphone access in your browser.'
                  );
                } else if (name === 'NotFoundError' || /device|found/i.test(msg)) {
                  setMicError('No microphone device found. Please connect a microphone.');
                } else {
                  setMicError(`Microphone error: ${msg}`);
                }
              }
            }}
            onIsChatOpenChange={setChatOpen}
          />
        </div>
      </motion.div>
    </section>
  );
}
