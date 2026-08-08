import { Button } from '@/components/ui/button';

interface HeroProps {
  onStart?: () => void;
}

export function Hero({ onStart }: HeroProps) {
  return (
    <section className="flex flex-col items-center gap-4 text-center">
      <img src="/murf-logo.svg" alt="Murf" className="h-12" />

      <h1 className="text-2xl font-bold">Murf Falcon Voice Agent</h1>

      <p className="max-w-prose text-foreground/90">
        A friendly voice assistant demo powered by Murf Falcon — the fastest TTS API. Try
        recommended voices: Anisha, Samar, Pooja.
      </p>

      <Button
        size="lg"
        onClick={onStart}
        className="mt-2 w-64 rounded-full font-mono text-xs font-bold tracking-wider uppercase"
      >
        Start talking
      </Button>
    </section>
  );
}
