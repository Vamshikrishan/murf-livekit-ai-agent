import { NextResponse } from 'next/server';
import { spawnSync } from 'node:child_process';
import path from 'node:path';

const backendRoot = path.resolve(process.cwd(), '..', 'backend');
const analyticsScript = path.join(backendRoot, 'src', 'analytics_api.py');

function getPythonCommand() {
  const localPython = process.platform === 'win32'
    ? path.join(backendRoot, '.venv', 'Scripts', 'python.exe')
    : path.join(backendRoot, '.venv', 'bin', 'python');

  if (process.platform === 'win32' && require('node:fs').existsSync(localPython)) {
    return localPython;
  }

  if (process.platform !== 'win32' && require('node:fs').existsSync(localPython)) {
    return localPython;
  }

  return process.platform === 'win32' ? 'python' : 'python3';
}

function readAnalyticsPayload(mode: 'summary' | 'calls', limit = 20) {
  const pythonCommand = getPythonCommand();
  const args = [analyticsScript, mode];
  if (mode === 'calls') {
    args.push(String(Math.max(1, Number(limit) || 20)));
  }

  const result = spawnSync(pythonCommand, args, {
    cwd: backendRoot,
    encoding: 'utf-8',
    env: process.env,
  });

  if (result.error) {
    throw result.error;
  }

  const stdout = (result.stdout || '').trim();
  if (!stdout) {
    return mode === 'calls' ? { calls: [] } : { total_calls: 0, successful_calls: 0, failed_calls: 0, success_rate: 0 };
  }

  const parsed = JSON.parse(stdout);
  if (mode === 'calls') {
    return { calls: Array.isArray(parsed) ? parsed : parsed.calls ?? [] };
  }

  return parsed;
}

export const revalidate = 0;

export async function GET(request: Request) {
  try {
    const { searchParams } = new URL(request.url);
    const mode = searchParams.get('mode') === 'calls' ? 'calls' : 'summary';
    const limit = Number(searchParams.get('limit') ?? '20');

    const payload = readAnalyticsPayload(mode, limit);

    return NextResponse.json(payload, {
      headers: {
        'Cache-Control': 'no-store',
      },
    });
  } catch (error) {
    console.error('analytics route failed', error);
    return NextResponse.json(
      {
        total_calls: 0,
        successful_calls: 0,
        failed_calls: 0,
        success_rate: 0,
        calls: [],
        error: 'Analytics data is unavailable right now.',
      },
      { status: 500 }
    );
  }
}
