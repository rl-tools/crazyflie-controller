import test from 'node:test';
import assert from 'node:assert/strict';
import { ConsoleReceiver, ConsoleBuffer } from '../dist/firmware-console.mjs';
import { DownlinkDecoder } from '../dist/crazyflie.mjs';

const encode = text => new TextEncoder().encode(text);
const packet = text => Uint8Array.of(0, ...encode(text));
function frames(data, legacy = false) {
  const header = 0x80 | (legacy ? data.length : data.length - 1);
  if (data.length <= 19) return [Uint8Array.of(header, ...data)];
  return [Uint8Array.of(header, ...data.slice(0, 19)), legacy
    ? Uint8Array.of(header, ...data.slice(20), 0x58) // Firmware's out-of-bounds byte is not console text.
    : Uint8Array.of(0, ...data.slice(19))];
}

test('standard console reassembles long frames and UTF-8 across CRTP packets', () => {
  let text = '';
  const receiver = new ConsoleReceiver(chunk => { text += chunk; });
  for (const frame of frames(packet('SYS: Starting stabilizer loop'))) receiver.push(frame, 'standard');
  receiver.push(frames(Uint8Array.of(0, 0xe2, 0x82))[0], 'standard');
  receiver.push(frames(Uint8Array.of(0, 0xac, 10))[0], 'standard');
  assert.equal(text, 'SYS: Starting stabilizer loop€\n');
});

test('legacy console marks the missing byte and omits the extra byte without relaxing acknowledgements', () => {
  let text = '';
  const receiver = new ConsoleReceiver(chunk => { text += chunk; });
  const strict = new DownlinkDecoder();
  strict.mode = 'legacy';
  const data = packet('1234567890123456789abcdef\n');
  for (const frame of frames(data, true)) {
    assert.equal(strict.push(frame), null);
    receiver.push(frame, 'legacy');
  }
  assert.equal(text, '123456789012345678⟦missing byte⟧abcdef\n');
  text = '';
  for (const frame of frames(Uint8Array.of(0x23, ...new Uint8Array(23).fill(65)), true)) receiver.push(frame, 'legacy');
  assert.equal(text, '', 'parameter data is never shown as console text');
});

test('captures boot console before framing identification and marks zeroed and interrupted packets', () => {
  let text = '';
  const receiver = new ConsoleReceiver(chunk => { text += chunk; });
  for (const frame of frames(packet('1234567890123456789abcdef\n'), true)) receiver.push(frame, null);
  assert.equal(text, '');
  receiver.push(frames(packet('Ready\n'), true)[0], 'legacy');
  assert.equal(text, '123456789012345678⟦missing byte⟧abcdef\nReady\n');
  receiver.push(frames(new Uint8Array(8), true)[0], 'legacy');
  assert.match(text, /⟦BLE data lost⟧\n$/);
  receiver.push(frames(packet('1234567890123456789abcdef\n'), true)[0], 'legacy');
  receiver.close();
  assert.match(text, /123456789012345678⟦incomplete BLE packet⟧\n$/);
});

test('console view wraps and pages without changing a paused snapshot', () => {
  const buffer = new ConsoleBuffer(3);
  buffer.append('one\ntwo\nthree\npart');
  assert.deepEqual(buffer.page(5, 2), { text: 'three\npart', start: 2, end: 4, total: 4, older: true, newer: false });
  buffer.older(5, 2);
  assert.equal(buffer.page(5, 2).text, 'one\ntwo');
  buffer.append('ial\nfive\nsix\n');
  assert.equal(buffer.page(5, 2).text, 'one\ntwo');
  assert.equal(buffer.lines.length, 3);
  buffer.newer(5, 2);
  assert.equal(buffer.page(5, 2).text, 'three\npart');
  buffer.resume();
  assert.equal(buffer.page(5, 2).text, 'five\nsix');
  assert.equal(buffer.live, true);
});

test('a partial final line stays visible, and long unbroken output is bounded', () => {
  const buffer = new ConsoleBuffer(2);
  buffer.append('SYS: par');
  buffer.append('tial');
  assert.equal(buffer.page(40, 5).text, 'SYS: partial');
  buffer.pause();
  buffer.append('\nnew line\n');
  assert.equal(buffer.page(40, 5).text, 'SYS: partial');
  buffer.resume();
  assert.equal(buffer.page(40, 5).text, 'SYS: partial\nnew line');
  buffer.append('x'.repeat(20000));
  assert.ok(buffer.partial.length < 4096);
  assert.equal(buffer.lines.length, 2);
});
