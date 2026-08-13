import { GET as getAnalytics } from '../route';

export const revalidate = 0;

export async function GET(request: Request) {
  const url = new URL(request.url);
  url.searchParams.set('mode', 'calls');
  return getAnalytics(new Request(url.toString(), { method: 'GET' }));
}
