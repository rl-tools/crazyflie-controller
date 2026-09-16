// Wire formats follow param_logic.c, crtp_commander.c and ble_crazyflies.c.
export const SERVICE = '00000201-1c7f-4f9e-947b-43b7c00a9a08';
export const CRTP = '00000202-1c7f-4f9e-947b-43b7c00a9a08';
export const CRTP_UP = '00000203-1c7f-4f9e-947b-43b7c00a9a08';
export const CRTP_DOWN = '00000204-1c7f-4f9e-947b-43b7c00a9a08';
export const LEARNED_PACKET = Uint8Array.of(0x7d, 0x01); // Port 7, channel 1; reserved bits set like cflib.
const types = {
  int8: [0, 1, 'setInt8'], int16: [1, 2, 'setInt16'], int32: [2, 4, 'setInt32'],
  int64: [3, 8, 'setBigInt64'], float: [6, 4, 'setFloat32'], double: [7, 8, 'setFloat64'],
  uint8: [8, 1, 'setUint8'], uint16: [9, 2, 'setUint16'], uint32: [10, 4, 'setUint32'],
  uint64: [11, 8, 'setBigUint64'],
};

export function encodeValue(type, text) {
  if (!Object.hasOwn(types, type)) throw new Error(`Unsupported type: ${type}`);
  const [code, size, setter] = types[type];
  if (!String(text).trim()) throw new Error('A value is required.');
  const bytes = new Uint8Array(size);
  const view = new DataView(bytes.buffer);
  if (type.includes('int')) {
    if (!/^[+-]?\d+$/.test(String(text))) throw new Error(`${type} requires a decimal integer.`);
    const value = BigInt(text);
    const bits = BigInt(size * 8);
    const unsigned = type.startsWith('u');
    const min = unsigned ? 0n : -(1n << (bits - 1n));
    const max = (1n << (unsigned ? bits : bits - 1n)) - 1n;
    if (value < min || value > max) throw new Error(`Value is out of range for ${type}.`);
    view[setter](0, size === 8 ? value : Number(value), true);
  } else {
    const value = Number(text);
    if (!Number.isFinite(value) || (type === 'float' && !Number.isFinite(Math.fround(value)))) {
      throw new Error(`${type} requires a finite, representable number.`);
    }
    view[setter](0, value, true);
  }
  return { code, bytes };
}

export function parameterCommand(name, type, value) {
  if (!/^[A-Za-z0-9_]+\.[A-Za-z0-9_]+$/.test(name)) throw new Error(`Invalid parameter name: ${name}`);
  const { code, bytes } = encodeValue(type, value);
  const names = new TextEncoder().encode(name.replace('.', '\0') + '\0');
  const packet = Uint8Array.of(0x2f, 0, ...names, code, ...bytes); // PARAM / MISC / SET_BY_NAME.
  if (packet.length > 31) throw new Error(`${name}: name and value exceed the CRTP packet limit.`);
  return { name, type, value: String(value), packet, prefix: Uint8Array.of(0, ...names) };
}

export function parseConfiguration(text) {
  const commands = [];
  const seen = new Set();
  for (const [index, line] of text.split(/\r?\n/).entries()) {
    const content = line.split('#')[0].trim();
    if (!content) continue;
    try {
      const fields = content.split(/\s+/);
      if (fields.length !== 3) throw new Error('Use: group.name type value');
      if (seen.has(fields[0])) throw new Error(`Duplicate parameter: ${fields[0]}`);
      commands.push(parameterCommand(...fields));
      seen.add(fields[0]);
    } catch (error) { throw new Error(`Line ${index + 1}: ${error.message}`); }
  }
  if (!commands.length) throw new Error('Enter at least one parameter.');
  return commands;
}

// Short packets use the raw characteristic. The actual nRF CRTPUP receiver
// (including current upstream) expects total length, unlike the protocol prose.
export function uplinkFrames(packet) {
  if (!packet.length || packet.length > 31) throw new Error('CRTP packets must contain 1–31 bytes.');
  return packet.length <= 20
    ? [{ uuid: CRTP, bytes: packet }]
    : [{ uuid: CRTP_UP, bytes: Uint8Array.of(0x80 | packet.length, ...packet.slice(0, 19)) },
      { uuid: CRTP_UP, bytes: Uint8Array.of(0, ...packet.slice(19)) }];
}

export class DownlinkDecoder {
  mode = null;
  pending = null;
  push(bytes) {
    if (bytes.length < 2 || bytes.length > 20) return null;
    const start = Boolean(bytes[0] & 0x80);
    const pid = (bytes[0] >> 5) & 3;
    const length = bytes[0] & 31;
    const data = bytes.slice(1);
    if (start) {
      this.pending = null;
      // A GET_INFO_V2 response (8 raw bytes) identifies either nRF variant.
      if (data.length < 19) {
        if (length === data.length) this.mode ??= 'legacy';
        else if (length + 1 === data.length) this.mode ??= 'standard';
        else return null;
      }
      const total = length + (this.mode === 'standard' ? 1 : 0);
      if (total === data.length) return data;
      if (this.mode === 'standard' && total > data.length && total <= 31) {
        this.pending = { pid, total, data };
      }
      // Legacy long notifications lose a payload byte in the nRF transmitter.
      // Never manufacture missing bytes or accept their corrupt acknowledgements.
      return null;
    }
    const pending = this.pending;
    this.pending = null;
    if (!pending || pid !== pending.pid) return null;
    const combined = Uint8Array.of(...pending.data, ...data);
    return combined.length === pending.total ? combined : null;
  }
}

export class CrazyflieBluetooth {
  constructor({ bluetooth = globalThis.navigator?.bluetooth, onState = () => {}, onProgress = () => {}, onError = () => {} } = {}) {
    this.bluetooth = bluetooth;
    this.onState = onState;
    this.onProgress = onProgress;
    this.onError = onError;
    this.generation = 0;
    this.queue = Promise.resolve();
    this.waiter = null;
    this.ready = false;
    this.connecting = false;
  }

  async connect() {
    if (!this.bluetooth) throw new Error('Web Bluetooth is unavailable. Use a supported Chrome browser on HTTPS or localhost.');
    if (this.ready || this.connecting) throw new Error('A connection is already active.');
    this.connecting = true;
    const generation = ++this.generation;
    this.decoder = new DownlinkDecoder();
    try {
      // Filter by name: this firmware advertises Device Information, not SERVICE.
      const device = await this.bluetooth.requestDevice({ filters: [{ namePrefix: 'Crazyflie' }], optionalServices: [SERVICE] });
      if (generation !== this.generation) throw new Error('Connection cancelled.');
      this.device = device;
      this.onProgress(`Connecting to ${device.name}…`);
      this.disconnected = () => this.disconnect();
      device.addEventListener('gattserverdisconnected', this.disconnected);
      const server = await device.gatt.connect();
      const service = await server.getPrimaryService(SERVICE);
      this.raw = await service.getCharacteristic(CRTP);
      this.up = await service.getCharacteristic(CRTP_UP);
      this.down = await service.getCharacteristic(CRTP_DOWN);
      this.notification = event => {
        const value = event.target.value;
        const packet = this.decoder.push(new Uint8Array(value.buffer, value.byteOffset, value.byteLength));
        if (packet && this.waiter?.matches(packet)) this.waiter.resolve(packet);
      };
      this.down.addEventListener('characteristicvaluechanged', this.notification);
      await this.down.startNotifications();
      this.onProgress('Waiting for firmware…');
      // Probe the parameter service without changing any device state.
      // Boot console messages share this reply queue and can take several seconds
      // to drain before parameter information arrives.
      await this.request(Uint8Array.of(0x2c, 3), p => (p[0] & 0xf3) === 0x20 && p[1] === 3 && p.length === 8,
        10000, 'Bluetooth connected, but the firmware did not answer the parameter-info request. No parameters changed.');
      if (generation !== this.generation) throw new Error('Connection cancelled.');
      this.ready = true;
      this.onState(true);
      return { name: device.name, framing: this.decoder.mode };
    } catch (error) {
      this.disconnect();
      throw error;
    } finally { this.connecting = false; }
  }

  disconnect() {
    ++this.generation;
    this.ready = false;
    this.waiter?.reject(new Error('Bluetooth disconnected.'));
    if (this.down) this.down.removeEventListener('characteristicvaluechanged', this.notification);
    if (this.device) {
      this.device.removeEventListener('gattserverdisconnected', this.disconnected);
      if (this.device.gatt.connected) this.device.gatt.disconnect();
    }
    this.raw = this.up = this.down = this.device = null;
    this.queue = Promise.resolve();
    this.onState(false);
  }

  send(packet, allowed = () => true) {
    const generation = this.generation;
    const job = this.queue.then(async () => {
      if (!allowed()) return false;
      if (generation !== this.generation || !this.device?.gatt.connected) throw new Error('Not connected.');
      for (const frame of uplinkFrames(packet)) {
        if (!allowed()) return false;
        if (generation !== this.generation) throw new Error('Bluetooth disconnected.');
        const characteristic = frame.uuid === CRTP ? this.raw : this.up;
        await characteristic.writeValueWithResponse(frame.bytes);
      }
      return true;
    });
    this.queue = job.catch(() => {});
    return job;
  }

  async request(packet, matches, timeout = 3000, timeoutMessage = 'No firmware acknowledgement. The write may have been applied; reconnect before retrying.') {
    if (this.waiter) throw new Error('Another parameter request is in progress.');
    let timer;
    let pollTimer;
    let pending = true;
    let waiter;
    const reply = new Promise((resolve, reject) => {
      waiter = {
        matches,
        resolve: value => { pending = false; resolve(value); },
        reject: error => { pending = false; reject(error); },
      };
      this.waiter = waiter;
      timer = setTimeout(() => waiter.reject(new Error(timeoutMessage)), timeout);
    });
    // STM32 radiolink releases one queued reply per incoming packet, even over
    // BLE. Null CRTP packets drain that queue without repeating a parameter write.
    const poll = async () => {
      try {
        await this.send(Uint8Array.of(0xff), () => pending);
        if (pending) pollTimer = setTimeout(poll, 50);
      } catch (error) { waiter.reject(error); }
    };
    // Attach rejection handling immediately; a disconnect may precede GATT completion.
    const write = this.send(packet).then(sent => {
      if (pending) pollTimer = setTimeout(poll, 50);
      return sent;
    }).catch(error => { waiter.reject(error); throw error; });
    try {
      const [, response] = await Promise.all([write, reply]);
      return response;
    } catch (error) {
      // Closing prevents a late acknowledgement from confirming a later request.
      this.disconnect();
      throw error;
    } finally {
      pending = false;
      clearTimeout(timer);
      clearTimeout(pollTimer);
      if (this.waiter === waiter) this.waiter = null;
    }
  }

  validateCommand(command) {
    if (!this.ready) throw new Error('Connect to a Crazyflie first.');
    if (this.decoder.mode === 'legacy' && command.prefix.length + 2 > 19) {
      throw new Error(`${command.name}: this nRF firmware corrupts long BLE replies. Update its BLE downlink implementation before writing this parameter. Nothing sent.`);
    }
  }

  async writeParameter(command) {
    this.validateCommand(command);
    const response = await this.request(command.packet, packet =>
      (packet[0] & 0xf3) === 0x23 && packet.length === command.prefix.length + 2 &&
      command.prefix.every((byte, index) => packet[index + 1] === byte));
    const status = response.at(-1);
    if (status !== 0) {
      const message = { 2: 'parameter not found', 13: 'parameter is read-only', 22: 'parameter type does not match' }[status];
      throw new Error(`${command.name}: ${message || `firmware error ${status}`}`);
    }
  }
}

// One write at a time, no queued trigger backlog, and no resumption after stop.
export class PolicyStream {
  constructor(send, { onChange = () => {}, onError = () => {}, clock = () => performance.now(), schedule = (fn, delay) => globalThis.setTimeout(fn, delay), cancel = timer => globalThis.clearTimeout(timer) } = {}) {
    Object.assign(this, { send, onChange, onError, clock, schedule, cancel });
    this.token = 0;
    this.active = false;
    this.count = 0;
  }
  start() {
    if (this.active) return;
    const token = ++this.token;
    this.active = true;
    this.count = 0;
    let last = this.clock();
    const tick = async () => {
      if (!this.active || token !== this.token) return;
      const now = this.clock();
      if (now - last >= 150) {
        this.stop();
        this.onError(new Error('Transmission paused too long. Release and hold again to resume.'));
        return;
      }
      last = now;
      try {
        const sent = await this.send(LEARNED_PACKET, () => this.active && token === this.token && this.clock() - now < 150);
        if (!this.active || token !== this.token) return;
        if (!sent) throw new Error('Trigger expired before transmission.');
        this.onChange(true, ++this.count);
        this.timer = this.schedule(tick, Math.max(0, 50 - (this.clock() - now)));
      } catch (error) {
        if (token !== this.token) return;
        this.stop();
        this.onError(error);
      }
    };
    this.onChange(true, 0);
    void tick();
  }
  stop() {
    ++this.token;
    this.cancel(this.timer);
    this.active = false;
    this.onChange(false, this.count);
  }
}
