import { CrazyflieBluetooth, PolicyStream, LEARNED_PACKET, parseConfiguration } from './crazyflie.mjs';
import { Journal } from './journal.mjs';
import { ConsoleBuffer } from './firmware-console.mjs';
import { mountPosePanel } from './pose-panel.mjs';

const $ = id => document.getElementById(id);
const ui = Object.fromEntries(['connect', 'connection', 'link-state', 'link-detail', 'configuration', 'apply', 'validate', 'write-status', 'draft-state', 'line-count', 'hold', 'once', 'stop', 'policy-status', 'packet-count', 'command-state', 'activity', 'event-time', 'event-level', 'event-position', 'history-mode', 'event-older', 'event-newer', 'event-latest', 'operator-hint', 'reference', 'reference-open', 'reference-close'].map(id => [id, $(id)]));
const supported = window.isSecureContext && Boolean(navigator.bluetooth);
for (const id of ['parameters-view', 'console-view', 'parameters-page', 'console-page', 'console-status', 'console-position', 'console-older', 'console-newer', 'console-pause', 'firmware-output', 'console-note']) ui[id] = $(id);
const journal = new Journal();
const firmwareConsole = new ConsoleBuffer();
const consoleMeasure = document.createElement('canvas').getContext('2d');
let consoleRenderTimer;
let consoleColumns = 40;
let consoleRows = 10;
let consoleDamaged = false;
const draftKey = 'crazyflie.configuration';
let busy = false;
let writing = false;
let connecting = false;
let actionEpoch = 0;
let activePointer = null;
let activeKey = null;
let draftState = 'UNSENT';
let connectionFailed = false;
let connectedBefore = false;
let intentionalDisconnect = false;
let lastIssue = false;

function text(element, value) {
  if (element.textContent !== String(value)) element.textContent = value;
}
function renderJournal() {
  const event = journal.current;
  if (!event) return;
  text(ui.activity, event.message);
  text(ui['event-time'], event.time.toLocaleTimeString([], { hour12: false }));
  ui['event-time'].dateTime = event.time.toISOString();
  text(ui['event-level'], event.level);
  ui.activity.closest('.journal').dataset.level = event.level;
  text(ui['event-position'], `${journal.index + 1} / ${journal.entries.length}`);
  text(ui['history-mode'], journal.live ? 'LATEST' : 'HISTORY');
  ui['event-older'].disabled = journal.index <= 0;
  ui['event-newer'].disabled = journal.live || journal.index >= journal.entries.length - 1;
  ui['event-latest'].disabled = journal.live;
}
function log(message, level = 'INFO') {
  journal.append(message, level);
  renderJournal();
}
function report(error) {
  lastIssue = true;
  log(error.message, 'ERROR');
  update();
}
function inspectDraft() {
  try {
    const commands = parseConfiguration(ui.configuration.value);
    text(ui['line-count'], `${commands.length} parameter${commands.length === 1 ? '' : 's'}`);
  } catch {
    text(ui['line-count'], 'Check syntax');
  }
  text(ui['draft-state'], draftState);
}
function edited() {
  draftState = 'EDITED';
  text(ui['write-status'], 'Edited locally · not sent');
  try { sessionStorage.setItem(draftKey, ui.configuration.value); } catch { /* Editor remains usable without storage. */ }
  inspectDraft();
}

function renderConsole() {
  clearTimeout(consoleRenderTimer);
  consoleRenderTimer = null;
  if (ui['console-page'].hidden) return;
  const output = ui['firmware-output'];
  const style = getComputedStyle(output);
  consoleMeasure.font = style.font;
  const width = output.clientWidth - parseFloat(style.paddingLeft) - parseFloat(style.paddingRight);
  const height = output.clientHeight - parseFloat(style.paddingTop) - parseFloat(style.paddingBottom);
  consoleColumns = Math.max(1, Math.floor(width / consoleMeasure.measureText('M').width));
  consoleRows = Math.max(1, Math.floor(height / parseFloat(style.lineHeight)));
  const page = firmwareConsole.page(consoleColumns, consoleRows);
  text(output, page.total ? page.text : client.ready || connecting ? 'Waiting for firmware output…' : 'Connect to receive firmware output.');
  text(ui['console-status'], firmwareConsole.live ? client.ready || connecting ? 'LIVE' : 'OFFLINE' : 'PAUSED');
  text(ui['console-position'], page.total ? `${page.start + 1}–${page.end} / ${page.total}` : '0 lines');
  text(ui['console-pause'], firmwareConsole.live ? 'Pause' : 'Live');
  ui['console-pause'].disabled = !page.total;
  ui['console-older'].disabled = !page.older;
  ui['console-newer'].disabled = !page.newer;
  text(ui['console-note'], consoleDamaged ? '⟦…⟧ = data lost by BLE firmware. Text is incomplete.' : 'Read-only · capture continues while the view is paused.');
}
function scheduleConsole() {
  if (!consoleRenderTimer) consoleRenderTimer = setTimeout(renderConsole, 100);
}
function selectWorkspace(showConsole) {
  ui['parameters-page'].hidden = showConsole;
  ui['console-page'].hidden = !showConsole;
  ui['draft-state'].hidden = showConsole;
  ui['parameters-view'].setAttribute('aria-pressed', String(!showConsole));
  ui['console-view'].setAttribute('aria-pressed', String(showConsole));
  renderConsole();
}

const client = new CrazyflieBluetooth({
  onConsole(chunk) {
    if (chunk.includes('⟦')) consoleDamaged = true;
    firmwareConsole.append(chunk);
    scheduleConsole();
  },
  onError: report,
  onProgress(message) { text(ui.connection, message); },
  onState(connected) {
    if (!connected) {
      stop();
      text(ui.connection, 'Disconnected');
      if (connectedBefore) {
        draftState = 'UNVERIFIED';
        text(ui['write-status'], 'Device disconnected · values unverified');
        if (!intentionalDisconnect) log('Bluetooth connection closed. Transmission stopped; device values are unverified.', 'WARN');
      }
    }
    connectedBefore = connected;
    inspectDraft();
    scheduleConsole();
    update();
  },
});

const stream = new PolicyStream((packet, allowed) => client.send(packet, allowed), {
  onChange(active, count) {
    ui.hold.setAttribute('aria-pressed', String(active));
    text(ui['policy-status'], active ? 'Transmitting' : 'Idle');
    text(ui['packet-count'], count);
    update();
  },
  onError: report,
});

function update() {
  ui.connect.disabled = connecting || !supported || (busy && !client.ready);
  text(ui.connect, connecting ? 'Connecting…' : client.ready ? 'Disconnect' : 'Connect');
  ui['link-state'].dataset.state = connecting ? 'connecting' : client.ready ? 'connected' : connectionFailed ? 'error' : 'disconnected';
  text(ui['link-detail'], connecting ? 'READ-ONLY HANDSHAKE' : client.ready ? 'LINK ESTABLISHED' : 'No active link');
  ui.apply.disabled = !client.ready || busy || stream.active;
  ui.validate.disabled = busy || stream.active;
  ui.configuration.disabled = busy || stream.active;
  ui.hold.disabled = !client.ready || busy;
  ui.once.disabled = !client.ready || busy || stream.active;
  ui.stop.disabled = !stream.active && (!busy || writing);
  text(ui['command-state'], stream.active ? 'SENDING' : busy ? 'BUSY' : client.ready ? 'READY' : 'OFFLINE');
  text(ui['operator-hint'], lastIssue ? 'Error recorded · use Latest in the event journal.' : stream.active ? 'Release or press Esc to stop sending.' : busy ? 'Operation in progress.' : client.ready ? 'Connected · commands use the device’s current parameters.' : 'Connect to enable device commands.');
}

function stop() {
  ++actionEpoch;
  activePointer = null;
  activeKey = null;
  stream.stop();
}

ui.connect.addEventListener('click', async () => {
  if (client.ready) {
    stop();
    intentionalDisconnect = true;
    client.disconnect();
    intentionalDisconnect = false;
    log('Disconnected.');
    return;
  }
  connecting = true;
  connectionFailed = false;
  lastIssue = false;
  if (firmwareConsole.received || firmwareConsole.partial) firmwareConsole.append('\n── New connection ──\n');
  scheduleConsole();
  text(ui.connection, 'Select a device in Chrome…');
  update();
  try {
    const result = await client.connect();
    text(ui.connection, `Connected · ${result.name}`);
    draftState = 'UNSENT';
    text(ui['write-status'], 'Not sent on this connection');
    inspectDraft();
    log(`Connected to ${result.name}. No parameters changed.${result.framing === 'legacy' ? ' Legacy nRF: long parameter acknowledgements are unsupported.' : ''}`, 'OK');
  } catch (error) {
    connectionFailed = true;
    report(error);
  } finally {
    connecting = false;
    scheduleConsole();
    update();
  }
});

ui.validate.addEventListener('click', () => {
  try {
    const commands = parseConfiguration(ui.configuration.value);
    log(`Syntax valid: ${commands.length} parameters. Names and types will be checked by the firmware when written. Nothing sent.`, 'OK');
    lastIssue = false;
    update();
  } catch (error) { report(error); }
});

ui.apply.addEventListener('click', async () => {
  if (!client.ready || busy || stream.active) return;
  let confirmed = 0;
  try {
    const commands = parseConfiguration(ui.configuration.value);
    commands.forEach(command => client.validateCommand(command));
    const generation = client.generation;
    busy = true;
    writing = true;
    lastIssue = false;
    draftState = 'WRITING';
    inspectDraft();
    update();
    for (const command of commands) {
      if (client.generation !== generation) throw new Error('Connection changed; remaining writes cancelled.');
      text(ui['write-status'], `${confirmed + 1}/${commands.length} · ${command.name}`);
      await client.writeParameter(command);
      ++confirmed;
      log(`Confirmed ${command.name} = ${command.value} (${command.type}).`, 'OK');
    }
    draftState = 'CONFIRMED';
    text(ui['write-status'], `${confirmed} / ${commands.length} acknowledged`);
  } catch (error) {
    draftState = confirmed ? 'PARTIAL' : 'UNCONFIRMED';
    text(ui['write-status'], `Stopped · ${confirmed} acknowledged`);
    report(error);
  } finally {
    busy = false;
    writing = false;
    inspectDraft();
    update();
  }
});
ui.configuration.addEventListener('input', edited);

ui.once.addEventListener('click', async () => {
  if (!client.ready || busy || stream.active || document.hidden) return;
  const epoch = actionEpoch;
  const started = performance.now();
  busy = true;
  lastIssue = false;
  update();
  try {
    const sent = await client.send(LEARNED_PACKET, () => epoch === actionEpoch && !document.hidden && performance.now() - started < 150);
    log(sent ? 'One learned-controller packet sent. This command has no firmware acknowledgement.' : 'Single packet cancelled before transmission.', sent ? 'OK' : 'INFO');
  } catch (error) { report(error); }
  finally { busy = false; update(); }
});

ui.hold.addEventListener('pointerdown', event => {
  if (event.button !== 0 || ui.hold.disabled || stream.active) return;
  event.preventDefault();
  ui.hold.focus();
  activePointer = event.pointerId;
  ui.hold.setPointerCapture(event.pointerId);
  lastIssue = false;
  stream.start();
});
for (const eventName of ['pointerup', 'pointercancel', 'lostpointercapture']) {
  window.addEventListener(eventName, event => { if (event.pointerId === activePointer) stop(); });
}
ui.hold.addEventListener('contextmenu', event => event.preventDefault());
ui.hold.addEventListener('keydown', event => {
  if (![' ', 'Enter'].includes(event.key)) return;
  event.preventDefault();
  if (event.repeat || ui.hold.disabled || stream.active) return;
  activeKey = event.key;
  lastIssue = false;
  stream.start();
});
window.addEventListener('keyup', event => { if (event.key === activeKey) stop(); });
window.addEventListener('keydown', event => { if (event.key === 'Escape') stop(); });
ui.hold.addEventListener('blur', stop);
ui.stop.addEventListener('click', stop);
window.addEventListener('blur', stop);
document.addEventListener('visibilitychange', () => { if (document.hidden) stop(); });
window.addEventListener('pagehide', () => { stop(); client.disconnect(); });

ui['event-older'].addEventListener('click', () => { journal.older(); renderJournal(); });
ui['event-newer'].addEventListener('click', () => { journal.newer(); renderJournal(); });
ui['event-latest'].addEventListener('click', () => { journal.latest(); renderJournal(); });
ui['parameters-view'].addEventListener('click', () => selectWorkspace(false));
ui['console-view'].addEventListener('click', () => selectWorkspace(true));
ui['console-older'].addEventListener('click', () => { firmwareConsole.older(consoleColumns, consoleRows); renderConsole(); });
ui['console-newer'].addEventListener('click', () => { firmwareConsole.newer(consoleColumns, consoleRows); renderConsole(); });
ui['console-pause'].addEventListener('click', () => { if (firmwareConsole.live) firmwareConsole.pause(); else firmwareConsole.resume(); renderConsole(); });
new ResizeObserver(scheduleConsole).observe(ui['firmware-output']);
ui['reference-open'].addEventListener('click', () => { stop(); ui.reference.showModal(); });
ui['reference-close'].addEventListener('click', () => ui.reference.close());
for (const button of document.querySelectorAll('[data-topic]')) {
  button.addEventListener('click', () => {
    for (const topic of document.querySelectorAll('[data-topic]')) topic.setAttribute('aria-pressed', String(topic === button));
    for (const page of document.querySelectorAll('[data-reference]')) page.hidden = page.dataset.reference !== button.dataset.topic;
  });
}

try {
  const saved = sessionStorage.getItem(draftKey);
  if (saved !== null) ui.configuration.value = saved;
} catch { /* Storage is optional. */ }
inspectDraft();
log(supported ? 'Ready. Connect to a powered-on Crazyflie. The editor contains a local draft; no parameters have been sent.' : !window.isSecureContext ? 'Bluetooth requires HTTPS or localhost. A plain HTTP address on your local network will not work.' : 'Web Bluetooth is unavailable. Use Chrome on a supported platform. Chrome on iOS is unsupported; Linux requires experimental Web Platform features.', supported ? 'INFO' : 'ERROR');
update();

mountPosePanel($('pose-panel'), log);

// Agent integration can only edit a local draft or read status. Device operations
// require the same explicit controls as ordinary use.
const context = document.modelContext;
if (context?.registerTool) {
  const lifecycle = new AbortController();
  const register = tool => {
    try { Promise.resolve(context.registerTool(tool, { signal: lifecycle.signal })).catch(report); }
    catch (error) { report(error); }
  };
  register({
    name: 'get_crazyflie_status',
    description: 'Read this page’s Bluetooth connection and learned-controller transmission status.',
    inputSchema: { type: 'object', properties: {}, additionalProperties: false },
    annotations: { readOnlyHint: true },
    execute(input = {}) {
      if (!input || typeof input !== 'object' || Array.isArray(input) || Object.keys(input).length) throw new Error('This tool takes no arguments.');
      return { connected: client.ready, device: client.device?.name ?? null, transmitting: stream.active, configuration: ui.configuration.value };
    },
  });
  register({
    name: 'stage_crazyflie_configuration',
    description: 'Validate and replace the configuration editor. Does not write to the drone.',
    inputSchema: { type: 'object', properties: { configuration: { type: 'string' } }, required: ['configuration'], additionalProperties: false },
    annotations: { readOnlyHint: false },
    execute(input) {
      if (!input || typeof input.configuration !== 'string' || Object.keys(input).some(key => key !== 'configuration')) throw new Error('Provide only a configuration string.');
      if (busy || stream.active) throw new Error('Stop transmission and wait for writes to finish before editing.');
      const commands = parseConfiguration(input.configuration);
      ui.configuration.value = input.configuration;
      edited();
      return { staged: commands.length, sent: false };
    },
  });
  window.addEventListener('pagehide', () => lifecycle.abort(), { once: true });
}
