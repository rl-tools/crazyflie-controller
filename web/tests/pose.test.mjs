import test from 'node:test';
import assert from 'node:assert/strict';
import { DEFAULT_POSE_URL, PoseConnection, parsePose, poseEndpoints } from '../dist/pose.mjs';

const sample = JSON.stringify({ timestamp: 1700000000.125, position: [1, -2, 0.3], quaternion: [0, 0, 0, 2] });
const deferred = () => {
  let resolve;
  const promise = new Promise(done => { resolve = done; });
  return { promise, resolve };
};

function setup(t, options = {}) {
  const sockets = [];
  const requests = [];
  let time = 0;
  const client = new PoseConnection({
    now: () => time,
    fetch: async (url, init) => { requests.push({ url, ...init }); return { ok: true }; },
    createSocket: url => {
      const socket = { url, closed: false,
        close() { this.closed = true; this.onclose?.(); },
        open() { this.onopen(); },
        send(data = sample) { this.onmessage({ data }); },
      };
      sockets.push(socket);
      return socket;
    },
    ...options,
  });
  t.after(() => client.disconnect());
  return { client, sockets, requests, advance: ms => { time += ms; } };
}

test('derive HTTP health endpoints from ws and wss addresses', () => {
  assert.deepEqual(poseEndpoints(DEFAULT_POSE_URL), {
    socket: DEFAULT_POSE_URL, health: 'http://127.0.0.1:8765/health',
  });
  assert.deepEqual(poseEndpoints('wss://localhost:9000/pose?subject=cf'), {
    socket: 'wss://localhost:9000/pose?subject=cf', health: 'https://localhost:9000/health',
  });
  for (const url of ['localhost:8765', 'http://localhost/pose', 'ws://user:secret@localhost/pose', 'ws://localhost/pose#fragment']) {
    assert.throws(() => poseEndpoints(url));
  }
});

test('pose validation preserves units and XYZW order and normalizes orientation', () => {
  assert.deepEqual(parsePose(sample), { timestamp: 1700000000.125, position: [1, -2, 0.3], quaternion: [0, 0, 0, 1] });
  const valid = JSON.parse(sample);
  for (const invalid of [null, {}, { ...valid, timestamp: '1' }, { ...valid, position: [1, 2] },
    { ...valid, position: [1, null, 3] }, { ...valid, quaternion: [0, 0, 0, 0] },
    { ...valid, quaternion: [0, 0, 1] }]) {
    assert.throws(() => parsePose(JSON.stringify(invalid)));
  }
  assert.throws(() => parsePose(sample.replace('1700000000.125', '1e400')));
  assert.throws(() => parsePose('not JSON'));
});

test('health check precedes the socket; receipt age marks stale poses and fresh samples recover', async t => {
  const health = deferred();
  const { client, sockets, advance } = setup(t, { fetch: () => health.promise });
  const connecting = client.connect(DEFAULT_POSE_URL);
  assert.equal(client.snapshot.state, 'connecting');
  assert.equal(sockets.length, 0);
  health.resolve({ ok: true });
  await connecting;
  const [socket] = sockets;
  assert.equal(socket.url, DEFAULT_POSE_URL);
  socket.open();
  assert.equal(client.snapshot.state, 'waiting');
  socket.send();
  assert.equal(client.snapshot.state, 'live');
  advance(999);
  assert.equal(client.snapshot.state, 'live');
  advance(1);
  assert.equal(client.snapshot.state, 'stale');
  assert.equal(client.snapshot.age, 1000);
  assert.equal(client.active, true);
  socket.send();
  assert.equal(client.snapshot.state, 'live');
  assert.equal(client.snapshot.age, 0);
  client.disconnect();
  assert.equal(socket.closed, true);
  assert.equal(client.snapshot.pose, null);
  assert.equal(client.snapshot.state, 'disconnected');
});

test('cancelling an HTTP request prevents a late response from connecting', async t => {
  const health = deferred();
  let signal;
  const { client, sockets } = setup(t, { fetch: (_, init) => { signal = init.signal; return health.promise; } });
  const connecting = client.connect(DEFAULT_POSE_URL);
  client.disconnect();
  assert.equal(signal.aborted, true);
  health.resolve({ ok: true });
  await connecting;
  assert.equal(sockets.length, 0);
  assert.equal(client.snapshot.state, 'disconnected');
});

test('reconnect ignores callbacks from the previous socket', async t => {
  const { client, sockets } = setup(t);
  await client.connect(DEFAULT_POSE_URL);
  const oldSocket = sockets[0];
  await client.connect(DEFAULT_POSE_URL);
  assert.equal(oldSocket.closed, true);
  sockets[1].open();
  sockets[1].send();
  oldSocket.open();
  oldSocket.send('invalid');
  oldSocket.onerror();
  oldSocket.onclose();
  assert.equal(client.snapshot.state, 'live');
  assert.deepEqual(client.snapshot.pose, parsePose(sample));
});

test('invalid frames close the stream and clear the previous pose', async t => {
  const { client, sockets } = setup(t);
  for (const data of ['{}', new Uint8Array([1, 2])]) {
    await client.connect(DEFAULT_POSE_URL);
    const socket = sockets.at(-1);
    socket.open();
    socket.send();
    socket.send(data);
    assert.equal(client.snapshot.state, 'error');
    assert.match(client.snapshot.message, /invalid pose/);
    assert.equal(client.snapshot.pose, null);
    assert.equal(socket.closed, true);
  }
});

test('health failures provide useful errors and never open a socket', async t => {
  for (const [fetch, message] of [
    [async () => ({ ok: false, status: 403 }), /allowed origins/],
    [async () => ({ ok: false, status: 503 }), /HTTP 503/],
    [async () => { throw new TypeError('Failed to fetch'); }, /Cannot reach/],
  ]) {
    const { client, sockets } = setup(t, { fetch });
    await client.connect(DEFAULT_POSE_URL);
    assert.equal(client.snapshot.state, 'error');
    assert.match(client.snapshot.message, message);
    assert.equal(sockets.length, 0);
  }
});

test('remote close and socket errors allow another connection', async t => {
  const { client, sockets, requests } = setup(t);
  for (const event of ['onclose', 'onerror']) {
    await client.connect(DEFAULT_POSE_URL);
    const socket = sockets.at(-1);
    socket.open();
    socket.send();
    socket[event]();
    assert.equal(client.snapshot.state, 'error');
    assert.equal(client.snapshot.pose, null);
    assert.equal(client.active, false);
  }
  await client.connect(DEFAULT_POSE_URL);
  sockets.at(-1).open();
  sockets.at(-1).send();
  assert.equal(client.snapshot.state, 'live');
  assert.equal(requests[0].url, 'http://127.0.0.1:8765/health');
  assert.equal(requests[0].credentials, 'omit');
});

test('timeout cancels health checks and sockets that never open', async t => {
  t.mock.timers.enable({ apis: ['setTimeout'] });
  const health = deferred();
  const pending = setup(t, { fetch: () => health.promise });
  const connecting = pending.client.connect(DEFAULT_POSE_URL);
  t.mock.timers.tick(30000);
  assert.equal(pending.client.snapshot.state, 'error');
  assert.match(pending.client.snapshot.message, /timed out/);
  health.resolve({ ok: true });
  await connecting;
  assert.equal(pending.sockets.length, 0);

  const { client, sockets } = setup(t);
  await client.connect(DEFAULT_POSE_URL);
  t.mock.timers.tick(30000);
  assert.equal(client.snapshot.state, 'error');
  assert.equal(sockets[0].closed, true);
  sockets[0].open();
  assert.equal(client.snapshot.state, 'error');
});
