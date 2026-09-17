export const DEFAULT_POSE_URL = 'ws://127.0.0.1:8765/pose';

export function poseEndpoints(address) {
  let socket;
  try { socket = new URL(address); }
  catch { throw new Error('Enter a WebSocket address, such as ws://127.0.0.1:8765/pose.'); }
  if (!['ws:', 'wss:'].includes(socket.protocol) || socket.username || socket.password || socket.hash) {
    throw new Error('Use a ws:// or wss:// address without credentials or a fragment.');
  }
  const health = new URL('/health', socket);
  health.protocol = socket.protocol === 'wss:' ? 'https:' : 'http:';
  return { socket: socket.href, health: health.href };
}

export function parsePose(message) {
  const pose = JSON.parse(message);
  const vector = (value, length) => Array.isArray(value) && value.length === length && value.every(Number.isFinite);
  if (!pose || !Number.isFinite(pose.timestamp) || !vector(pose.position, 3) || !vector(pose.quaternion, 4)) {
    throw new Error('Expected a timestamp, position [x,y,z], and quaternion [x,y,z,w].');
  }
  const norm = Math.hypot(...pose.quaternion);
  if (!Number.isFinite(norm) || norm === 0) throw new Error('Invalid pose quaternion.');
  return { timestamp: pose.timestamp, position: pose.position, quaternion: pose.quaternion.map(value => value / norm) };
}

export class PoseConnection {
  constructor({ onChange = () => {}, fetch = globalThis.fetch.bind(globalThis),
    createSocket = url => new WebSocket(url), now = () => performance.now(),
    timeoutMs = 30000, staleMs = 1000 } = {}) {
    Object.assign(this, { onChange, fetch, createSocket, now, timeoutMs, staleMs });
    this.state = 'disconnected';
    this.message = '';
    this.pose = null;
    this.receivedAt = null;
    this.generation = 0;
  }

  get active() { return ['connecting', 'waiting', 'live'].includes(this.state); }

  get snapshot() {
    const age = this.receivedAt === null ? null : Math.max(0, this.now() - this.receivedAt);
    return { state: this.state === 'live' && age >= this.staleMs ? 'stale' : this.state,
      message: this.message, pose: this.pose, age };
  }

  reset() {
    ++this.generation;
    clearTimeout(this.timer);
    this.controller?.abort();
    this.controller = null;
    this.socket?.close();
    this.socket = null;
    this.pose = null;
    this.receivedAt = null;
  }

  disconnect() {
    this.reset();
    this.state = 'disconnected';
    this.message = '';
    this.onChange();
  }

  fail(message) {
    this.reset();
    this.state = 'error';
    this.message = message;
    this.onChange();
  }

  async connect(address) {
    this.reset();
    let urls;
    try { urls = poseEndpoints(address); }
    catch (error) { this.fail(error.message); return; }
    const generation = this.generation;
    const current = () => this.generation === generation;
    this.state = 'connecting';
    this.message = '';
    this.controller = new AbortController();
    this.timer = setTimeout(() => {
      if (current()) this.fail('Connection timed out. Check the server and allow local network access, then retry.');
    }, this.timeoutMs);
    this.onChange();
    try {
      // A simple HTTP request lets Chrome prompt for local access before the socket.
      const response = await this.fetch(urls.health, { signal: this.controller.signal, cache: 'no-store', credentials: 'omit' });
      if (!current()) return;
      if (!response.ok) throw new Error(response.status === 403
        ? 'This website is not allowed by the pose server. Check its allowed origins.'
        : `Pose server health check failed (HTTP ${response.status}).`);
      const socket = this.createSocket(urls.socket);
      this.socket = socket;
      socket.onopen = () => {
        if (!current()) return;
        clearTimeout(this.timer);
        this.controller = null;
        this.state = 'waiting';
        this.onChange();
      };
      socket.onmessage = event => {
        if (!current()) return;
        try {
          if (typeof event.data !== 'string') throw new Error('Expected a JSON text message.');
          this.pose = parsePose(event.data);
        } catch {
          this.fail('The server sent an invalid pose. Expected a timestamp, position, and quaternion.');
          return;
        }
        this.receivedAt = this.now();
        this.state = 'live';
        this.onChange();
      };
      socket.onerror = () => {
        if (current()) this.fail('Could not open the pose stream. Check the address and local network permission.');
      };
      socket.onclose = () => {
        if (current()) this.fail('Pose connection closed. Check the server, then reconnect.');
      };
    } catch (error) {
      if (current()) this.fail(error instanceof TypeError
        ? 'Cannot reach the pose server. Start it, check the address, and allow local network access.'
        : error.message);
    }
  }
}
