'use client';

import { useEffect, useMemo, useState } from 'react';

type AnalyticsSummary = {
  total_calls: number;
  successful_calls: number;
  failed_calls: number;
  success_rate: number;
};

type CallRecord = {
  call_id: string;
  started_at: string;
  ended_at: string | null;
  duration_seconds: number | null;
  channel: string;
  outcome: 'success' | 'failed';
  failure_reason: string | null;
};

const cardStyles =
  'rounded-2xl border border-slate-700/80 bg-slate-900/80 p-5 shadow-lg shadow-slate-950/20';

function formatDate(value: string | null) {
  if (!value) return '—';
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return value;
  return new Intl.DateTimeFormat('en-IN', {
    dateStyle: 'medium',
    timeStyle: 'short',
  }).format(date);
}

function formatDuration(value: number | null) {
  if (value === null || value === undefined) return '—';
  const minutes = Math.floor(value / 60);
  const seconds = value % 60;
  if (minutes > 0) return `${minutes}m ${seconds}s`;
  return `${seconds}s`;
}

export default function AnalyticsPage() {
  const [summary, setSummary] = useState<AnalyticsSummary | null>(null);
  const [calls, setCalls] = useState<CallRecord[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    async function loadData() {
      try {
        setLoading(true);
        const [summaryResponse, callsResponse] = await Promise.all([
          fetch('/api/analytics', { cache: 'no-store' }),
          fetch('/api/analytics/calls?limit=10', { cache: 'no-store' }),
        ]);

        if (!summaryResponse.ok || !callsResponse.ok) {
          throw new Error('Unable to load call analytics.');
        }

        const summaryData = await summaryResponse.json();
        const callsData = await callsResponse.json();

        setSummary({
          total_calls: Number(summaryData.total_calls ?? 0),
          successful_calls: Number(summaryData.successful_calls ?? 0),
          failed_calls: Number(summaryData.failed_calls ?? 0),
          success_rate: Number(summaryData.success_rate ?? 0),
        });
        setCalls(Array.isArray(callsData.calls) ? callsData.calls : []);
      } catch (loadError) {
        const message = loadError instanceof Error ? loadError.message : 'An unknown error occurred.';
        setError(message);
      } finally {
        setLoading(false);
      }
    }

    void loadData();
  }, []);

  const stats = useMemo(
    () => [
      {
        label: 'Total Calls',
        value: summary?.total_calls ?? 0,
        accent: 'from-indigo-500 to-violet-500',
      },
      {
        label: 'Successful Calls',
        value: summary?.successful_calls ?? 0,
        accent: 'from-emerald-500 to-teal-500',
      },
      {
        label: 'Failed Calls',
        value: summary?.failed_calls ?? 0,
        accent: 'from-rose-500 to-red-500',
      },
      {
        label: 'Success Rate',
        value: `${summary?.success_rate ?? 0}%`,
        accent: 'from-cyan-500 to-sky-500',
      },
    ],
    [summary]
  );

  return (
    <main className="min-h-screen bg-slate-950 px-4 py-10 text-slate-50">
      <div className="mx-auto max-w-6xl">
        <div className="mb-8 flex items-center justify-between gap-4">
          <div>
            <p className="text-sm uppercase tracking-[0.22em] text-indigo-300">Operations</p>
            <h1 className="mt-2 text-3xl font-semibold md:text-4xl">Call Analytics Dashboard</h1>
          </div>
          <a
            href="/"
            className="rounded-full border border-slate-700 bg-slate-900 px-4 py-2 text-sm font-medium text-slate-200 transition hover:border-indigo-400 hover:text-white"
          >
            Back to agent
          </a>
        </div>

        {error ? (
          <div className="rounded-2xl border border-red-500/50 bg-red-500/10 p-6 text-red-100">
            {error}
          </div>
        ) : null}

        <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-4">
          {stats.map((stat) => (
            <div key={stat.label} className={`${cardStyles} overflow-hidden`}>
              <div className={`mb-5 h-1.5 w-full rounded-full bg-gradient-to-r ${stat.accent}`} />
              <p className="text-sm text-slate-300">{stat.label}</p>
              <p className="mt-4 text-3xl font-semibold tracking-tight">
                {loading ? '…' : stat.value}
              </p>
            </div>
          ))}
        </div>

        <section className={`${cardStyles} mt-8`}>
          <div className="mb-4 flex items-center justify-between gap-4">
            <h2 className="text-xl font-semibold">Recent Calls</h2>
            <span className="text-sm text-slate-400">Latest completed calls</span>
          </div>

          {calls.length === 0 ? (
            <div className="rounded-xl border border-dashed border-slate-700 bg-slate-950/60 px-4 py-8 text-center text-slate-300">
              No completed calls have been recorded yet.
            </div>
          ) : (
            <div className="overflow-x-auto">
              <table className="min-w-full border-separate border-spacing-y-2 text-left">
                <thead>
                  <tr className="text-xs uppercase tracking-[0.18em] text-slate-400">
                    <th className="px-3 py-2 font-medium">Call</th>
                    <th className="px-3 py-2 font-medium">Channel</th>
                    <th className="px-3 py-2 font-medium">Started</th>
                    <th className="px-3 py-2 font-medium">Ended</th>
                    <th className="px-3 py-2 font-medium">Duration</th>
                    <th className="px-3 py-2 font-medium">Outcome</th>
                  </tr>
                </thead>
                <tbody>
                  {calls.map((call) => (
                    <tr key={call.call_id} className="rounded-xl bg-slate-950/60 text-sm text-slate-200">
                      <td className="rounded-l-xl px-3 py-3 font-mono text-xs text-indigo-200">
                        {call.call_id}
                      </td>
                      <td className="px-3 py-3 uppercase">{call.channel}</td>
                      <td className="px-3 py-3">{formatDate(call.started_at)}</td>
                      <td className="px-3 py-3">{formatDate(call.ended_at)}</td>
                      <td className="px-3 py-3">{formatDuration(call.duration_seconds)}</td>
                      <td className="rounded-r-xl px-3 py-3">
                        <span
                          className={`inline-flex rounded-full px-2.5 py-1 text-xs font-medium ${
                            call.outcome === 'success'
                              ? 'bg-emerald-500/15 text-emerald-300'
                              : 'bg-rose-500/15 text-rose-300'
                          }`}
                        >
                          {call.outcome}
                        </span>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </section>
      </div>
    </main>
  );
}
