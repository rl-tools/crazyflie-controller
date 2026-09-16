import test from 'node:test';
import assert from 'node:assert/strict';
import { CrazyflieBluetooth, DownlinkDecoder, PolicyStream, LEARNED_PACKET, CRTP, CRTP_UP, CRTP_DOWN, SERVICE, parameterCommand, parseConfiguration, encodeValue, uplinkFrames } from '../dist/crazyflie.mjs';

const bytes = value => Array.from(value);
const settle = () => new Promise(resolve => setImmediate(resolve));

function fragments(packet, pid = 0) {
  const first = Uint8Array.of(0x80 | (pid << 5) | (packet.length - 1), ...packet.slice(0, 19));
  return packet.length <= 19 ? [first] : [first, Uint8Array.of(pid << 5, ...packet.slice(19))];
}

function fakeBluetooth({ legacy = false, status = 0, acknowledge = true, polling = false, startupPackets = 2 } = {}) {
  const writes = [];
  let busy = false;
  let prefix;
  let expectedLength;
  const down = new EventTarget();
  down.startNotifications = async () => down;
  const notify = packet => {
    for (const data of legacy ? [Uint8Array.of(0x80 | packet.length, ...packet)] : fragments(packet, 2)) {
      down.value = new DataView(data.buffer);
      down.dispatchEvent(new Event('characteristicvaluechanged'));
    }
  };
  // STM32 releases one queued downlink per uplink, including null packets.
  // Early nRF downlinks can contain zeros instead of their original payload.
  const pending = Array.from({ length: polling ? startupPackets : 0 }, () => new Uint8Array(9));
  const respond = packet => polling ? pending.push(packet) : notify(packet);
  const reply = packet => {
    if (polling && pending.length) notify(pending.shift());
    if ((packet[0] & 0xf3) === 0x20 && packet[1] === 3) respond(Uint8Array.of(0x20, 3, 200, 1, 0, 0, 0, 0));
    else if (acknowledge && (packet[0] & 0xf3) === 0x23 && packet[1] === 0) {
      const first = packet.indexOf(0, 2);
      const second = packet.indexOf(0, first + 1);
      respond(Uint8Array.of(0x23, ...packet.slice(1, second + 1), status));
    }
  };
  const characteristic = uuid => ({
    async writeValueWithResponse(value) {
      assert.equal(busy, false, 'overlapping GATT operations');
      busy = true;
      writes.push({ uuid, value: value.slice() });
      await settle();
      if (uuid === CRTP) reply(value);
      else if (value[0] & 0x80) {
        expectedLength = value[0] & 31;
        prefix = value.slice(1);
      } else {
        const packet = Uint8Array.of(...prefix, ...value.slice(1));
        assert.equal(packet.length, expectedLength, 'nRF uses actual uplink length');
        reply(packet);
      }
      busy = false;
    },
  });
  const raw = characteristic(CRTP);
  const up = characteristic(CRTP_UP);
  const device = new EventTarget();
  device.name = 'Crazyflie-123456';
  device.gatt = {
    connected: false,
    async connect() { this.connected = true; return this; },
    async getPrimaryService(uuid) {
      assert.equal(uuid, SERVICE);
      return { getCharacteristic: async id => ({ [CRTP]: raw, [CRTP_UP]: up, [CRTP_DOWN]: down })[id] };
    },
    disconnect() { this.connected = false; device.dispatchEvent(new Event('gattserverdisconnected')); },
  };
  const bluetooth = {
    async requestDevice(options) {
      assert.deepEqual(options, { filters: [{ namePrefix: 'Crazyflie' }], optionalServices: [SERVICE] });
      return device;
    },
  };
  return { bluetooth, device, writes, notify };
}

test('learned trigger has port 7, meta channel 1, and command 1', () => {
  assert.deepEqual(bytes(LEARNED_PACKET), [0x7d, 1]);
  assert.equal(LEARNED_PACKET[0] >> 4, 7);
  assert.equal(LEARNED_PACKET[0] & 3, 1);
});

test('parameter packets match the firmware SET_BY_NAME layout', () => {
  assert.deepEqual(bytes(parameterCommand('rlt.wn', 'uint8', '4').packet),
    [0x2f, 0, 114, 108, 116, 0, 119, 110, 0, 8, 4]);
  const height = parameterCommand('rlt.target_z', 'float', '0.5').packet;
  assert.equal(height.length, 20);
  assert.deepEqual(bytes(height.slice(-5)), [6, 0, 0, 0, 0x3f]);
  assert.deepEqual(bytes(encodeValue('uint64', '18446744073709551615').bytes), Array(8).fill(255));
  assert.deepEqual(bytes(encodeValue('int64', '-9223372036854775808').bytes), [0, 0, 0, 0, 0, 0, 0, 128]);
  assert.deepEqual(bytes(encodeValue('int16', '-2').bytes), [254, 255]);
});

test('invalid names, types and numeric values fail before transmission', () => {
  for (const [type, value] of [['uint8', '-1'], ['uint8', '256'], ['int8', '128'], ['uint16', '65536'], ['uint32', '4294967296'], ['uint64', '18446744073709551616'], ['int64', '-9223372036854775809'], ['uint8', '1.5'], ['float', 'NaN'], ['float', 'Infinity'], ['float', '1e100'], ['double', ''], ['constructor', '1']]) {
    assert.throws(() => encodeValue(type, value), undefined, `${type} ${value}`);
  }
  assert.throws(() => parameterCommand('rlt.wn.extra', 'uint8', 0), /name/);
  assert.throws(() => parameterCommand('longgroup.verylongparametername', 'uint8', 0), /limit/);
  assert.throws(() => parseConfiguration('rlt.wn uint8 1\nrlt.wn uint8 2'), /Line 2: Duplicate/);
  assert.throws(() => parseConfiguration('# comment'), /at least one/);
  assert.equal(parseConfiguration('# comment\nrlt.wn uint8 1 # inline\n').length, 1);
});

test('uplink uses raw writes up to 20 bytes and firmware-compatible fragmentation above', () => {
  const short = parameterCommand('rlt.target_z', 'float', '0.3').packet;
  assert.equal(uplinkFrames(short).length, 1);
  assert.equal(uplinkFrames(short)[0].uuid, CRTP);
  const long = parameterCommand('rlt.motor_warmup', 'uint8', '1').packet;
  const frames = uplinkFrames(long);
  assert.equal(long.length, 21);
  assert.equal(frames[0].bytes[0], 0x80 | 21);
  assert.equal(frames[0].bytes.length, 20);
  assert.equal(frames[1].bytes[0], 0);
  assert.deepEqual(bytes(Uint8Array.of(...frames[0].bytes.slice(1), ...frames[1].bytes.slice(1))), bytes(long));
  assert.throws(() => uplinkFrames(new Uint8Array(32)), /1–31/);
});

test('standard downlink preserves fragments and rejects wrong IDs and incomplete responses', () => {
  const decoder = new DownlinkDecoder();
  const info = Uint8Array.of(0x20, 3, 1, 0, 0, 0, 0, 0);
  assert.deepEqual(decoder.push(fragments(info)[0]), info);
  assert.equal(decoder.mode, 'standard');
  for (const length of [19, 20, 31]) {
    const packet = Uint8Array.from({ length }, (_, i) => i);
    const parts = fragments(packet, 3);
    let result;
    for (const part of parts) result = decoder.push(part);
    assert.deepEqual(result, packet);
  }
  const parts = fragments(new Uint8Array(25), 1);
  assert.equal(decoder.push(parts[0]), null);
  assert.equal(decoder.push(Uint8Array.of(0, ...parts[1].slice(1))), null);
  decoder.push(parts[0]);
  assert.equal(decoder.push(parts[1].slice(0, -1)), null);
});

test('legacy short downlink works; broken long notifications never produce a packet', () => {
  const decoder = new DownlinkDecoder();
  const packet = Uint8Array.of(0x20, 3, 1, 0, 0, 0, 0, 0);
  assert.deepEqual(decoder.push(Uint8Array.of(0x88, ...packet)), packet);
  assert.equal(decoder.mode, 'legacy');
  assert.equal(decoder.push(Uint8Array.of(0x94, ...new Uint8Array(19))), null);
  assert.equal(decoder.push(Uint8Array.of(0x94, 0)), null);
});

test('connect only probes, then short and fragmented parameter writes wait for acknowledgements', async () => {
  const fake = fakeBluetooth();
  const client = new CrazyflieBluetooth(fake);
  assert.deepEqual(await client.connect(), { name: 'Crazyflie-123456', framing: 'standard' });
  assert.equal(fake.writes.length, 1);
  assert.deepEqual(bytes(fake.writes[0].value), [0x2c, 3]);
  await client.writeParameter(parameterCommand('rlt.target_z', 'float', '0.3'));
  await client.writeParameter(parameterCommand('rlt.motor_warmup', 'uint8', '1'));
  assert.equal(fake.writes.length, 4);
  client.disconnect();
  assert.equal(client.ready, false);
});

test('legacy mode rejects unconfirmable long names before writing', async () => {
  const fake = fakeBluetooth({ legacy: true });
  const client = new CrazyflieBluetooth(fake);
  await client.connect();
  await client.writeParameter(parameterCommand('rlt.target_z', 'float', '0.3'));
  const before = fake.writes.length;
  await assert.rejects(client.writeParameter(parameterCommand('rlt.motor_warmup', 'uint8', '1')), /corrupts long BLE replies/);
  assert.equal(fake.writes.length, before);
  client.disconnect();
});

test('requests poll queued firmware replies without repeating writes or polling while idle', async () => {
  const fake = fakeBluetooth({ legacy: true, polling: true });
  const client = new CrazyflieBluetooth(fake);
  await client.connect();
  assert.equal(client.ready, true);
  assert.deepEqual(fake.writes.map(write => bytes(write.value)), [[0x2c, 3], [0xff], [0xff]]);
  const command = parameterCommand('rlt.wn', 'uint8', '1');
  await client.writeParameter(command);
  assert.deepEqual(fake.writes.slice(3).map(write => bytes(write.value)), [bytes(command.packet), [0xff]]);
  const count = fake.writes.length;
  await new Promise(resolve => setTimeout(resolve, 100));
  assert.equal(fake.writes.length, count, 'polling stops once the acknowledgement arrives');
  client.disconnect();
});

test('connection drains a boot backlog that exceeds the normal request timeout', async () => {
  const fake = fakeBluetooth({ legacy: true, polling: true, startupPackets: 65 });
  const client = new CrazyflieBluetooth(fake);
  await client.connect();
  assert.equal(client.ready, true);
  assert.equal(fake.writes.length, 66);
  assert.deepEqual(bytes(fake.writes[0].value), [0x2c, 3]);
  assert.ok(fake.writes.slice(1).every(write => write.value.length === 1 && write.value[0] === 0xff));
  client.disconnect();
});

test('firmware errors propagate without claiming success', async () => {
  for (const [status, message] of [[2, /not found/], [13, /read-only/], [22, /type does not match/]]) {
    const client = new CrazyflieBluetooth(fakeBluetooth({ status }));
    await client.connect();
    await assert.rejects(client.writeParameter(parameterCommand('rlt.wn', 'uint8', '1')), message);
    client.disconnect();
  }
});

test('GATT writes serialize and cancelled queued triggers never send', async () => {
  const fake = fakeBluetooth();
  const client = new CrazyflieBluetooth(fake);
  await client.connect();
  let allowed = true;
  const first = client.send(Uint8Array.of(0xff));
  const cancelled = client.send(LEARNED_PACKET, () => allowed);
  const last = client.send(Uint8Array.of(0xff));
  allowed = false;
  assert.deepEqual(await Promise.all([first, cancelled, last]), [true, false, true]);
  assert.equal(fake.writes.some(write => write.value[0] === 0x7d), false);
  client.disconnect();
});

test('timeout closes the link and late acknowledgements cannot confirm another write', async () => {
  const fake = fakeBluetooth({ acknowledge: false });
  const client = new CrazyflieBluetooth(fake);
  await client.connect();
  const command = parameterCommand('rlt.wn', 'uint8', '1');
  await assert.rejects(client.request(command.packet, () => false, 75), /No firmware acknowledgement/);
  assert.equal(client.ready, false);
  assert.equal(fake.device.gatt.connected, false);
  const count = fake.writes.length;
  await new Promise(resolve => setTimeout(resolve, 100));
  assert.equal(fake.writes.length, count, 'polling stops on timeout');
  fake.notify(Uint8Array.of(0x23, ...command.prefix, 0));
  await assert.rejects(client.writeParameter(command), /Connect/);
});

test('hardware disconnect rejects an outstanding request and invalidates queued work', async () => {
  const fake = fakeBluetooth({ acknowledge: false });
  const client = new CrazyflieBluetooth(fake);
  await client.connect();
  const request = client.writeParameter(parameterCommand('rlt.wn', 'uint8', '1'));
  const rejection = assert.rejects(request, /disconnected/);
  fake.device.gatt.disconnect();
  await rejection;
  assert.equal(client.ready, false);
});

test('default stream timers preserve the browser receiver', async t => {
  const scheduled = [];
  const cancelled = [];
  t.mock.method(globalThis, 'setTimeout', function (callback, delay) {
    assert.equal(this, globalThis, 'setTimeout requires the browser global receiver');
    scheduled.push({ callback, delay });
    return 42;
  });
  t.mock.method(globalThis, 'clearTimeout', function (timer) {
    assert.equal(this, globalThis, 'clearTimeout requires the browser global receiver');
    cancelled.push(timer);
  });
  const errors = [];
  const stream = new PolicyStream(async () => true, { clock: () => 0, onError: error => errors.push(error) });
  stream.stop();
  stream.start();
  await settle();
  assert.deepEqual(errors, []);
  assert.equal(scheduled.length, 1);
  assert.equal(scheduled[0].delay, 50);
  stream.stop();
  assert.deepEqual(cancelled, [undefined, 42]);
  assert.equal(stream.active, false);
});

test('hold stream cancels pending writes and does not restart after release', async () => {
  let permit;
  let finish;
  const scheduled = [];
  const stream = new PolicyStream((packet, allowed) => {
    assert.deepEqual(packet, LEARNED_PACKET);
    permit = allowed;
    return new Promise(resolve => { finish = resolve; });
  }, { schedule: fn => scheduled.push(fn), cancel: () => {}, clock: () => 0 });
  stream.start();
  assert.equal(permit(), true);
  stream.stop();
  assert.equal(permit(), false);
  finish(true);
  await settle();
  assert.equal(stream.active, false);
  assert.equal(scheduled.length, 0);
});

test('hold stream targets 50 ms intervals and stops after a scheduler stall', async () => {
  let now = 0;
  let next;
  let writes = 0;
  const errors = [];
  const stream = new PolicyStream(async () => { ++writes; now += 10; return true; }, {
    clock: () => now,
    schedule: (fn, delay) => { assert.equal(delay, 40); next = fn; return 1; },
    cancel: () => {}, onError: error => errors.push(error),
  });
  stream.start();
  await settle();
  assert.equal(writes, 1);
  now = 50;
  await next();
  assert.equal(writes, 2);
  now = 250;
  await next();
  assert.equal(writes, 2);
  assert.equal(stream.active, false);
  assert.match(errors[0].message, /paused too long/);
});
