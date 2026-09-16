import { CrazyflieBluetooth, PolicyStream, LEARNED_PACKET, parseConfiguration } from './crazyflie.mjs';

const $ = id => document.getElementById(id);
const ui = Object.fromEntries(['connect', 'connection', 'compatibility', 'configuration', 'apply', 'write-status', 'hold', 'once', 'stop', 'policy-status', 'activity'].map(id => [id, $(id)]));
const supported = window.isSecureContext && Boolean(navigator.bluetooth);
let busy = false;
let connecting = false;
let actionEpoch = 0;
let activePointer = null;
let activeKey = null;
const logLines = [];

function log(message) {
  logLines.push(`${new Date().toLocaleTimeString()}  ${message}`);
  if (logLines.length > 40) logLines.shift();
  ui.activity.textContent = logLines.join('\n');
  ui.activity.scrollTop = ui.activity.scrollHeight;
}

function report(error) { log(`Error: ${error.message}`); }

const client = new CrazyflieBluetooth({
  onState(connected) {
    if (!connected) {
      stop();
      ui.connection.textContent = 'Disconnected';
    }
    update();
  },
});

const stream = new PolicyStream((packet, allowed) => client.send(packet, allowed), {
  onChange(active, count) {
    ui.hold.setAttribute('aria-pressed', String(active));
    ui['policy-status'].textContent = active ? `Transmitting · ${count} packets sent` : 'Not transmitting';
    update();
  },
  onError: report,
});

function update() {
  ui.connect.disabled = connecting || !supported;
  ui.connect.textContent = connecting ? 'Connecting…' : client.ready ? 'Disconnect' : 'Connect Crazyflie';
  ui.apply.disabled = !client.ready || busy || stream.active;
  ui.configuration.disabled = busy || stream.active;
  ui.hold.disabled = !client.ready || busy;
  ui.once.disabled = !client.ready || busy || stream.active;
  ui.stop.disabled = !stream.active && !busy;
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
    client.disconnect();
    log('Disconnected.');
    return;
  }
  connecting = true;
  ui.connection.textContent = 'Choose a Crazyflie in the Bluetooth dialog…';
  update();
  try {
    const result = await client.connect();
    ui.connection.textContent = `Connected to ${result.name}`;
    log(`Connected to ${result.name}. No parameters changed.`);
    if (result.framing === 'legacy') log('Legacy nRF downlink detected. Long parameter acknowledgements are unsupported; see web/README.md.');
  } catch (error) {
    report(error);
  } finally {
    connecting = false;
    update();
  }
});

ui.apply.addEventListener('click', async () => {
  if (!client.ready || busy || stream.active) return;
  let confirmed = 0;
  try {
    const commands = parseConfiguration(ui.configuration.value);
    // Validate the entire batch before any write; firmware application is sequential.
    commands.forEach(command => client.validateCommand(command));
    const generation = client.generation;
    busy = true;
    update();
    for (const command of commands) {
      if (client.generation !== generation) throw new Error('Connection changed; remaining writes cancelled.');
      ui['write-status'].textContent = `Writing ${command.name}…`;
      await client.writeParameter(command);
      ++confirmed;
      log(`Confirmed ${command.name} = ${command.value} (${command.type}).`);
    }
    ui['write-status'].textContent = `${confirmed} parameters confirmed`;
  } catch (error) {
    ui['write-status'].textContent = `Stopped · ${confirmed} writes confirmed`;
    report(error);
  } finally {
    busy = false;
    update();
  }
});
ui.configuration.addEventListener('input', () => { ui['write-status'].textContent = 'Edited · not sent'; });

ui.once.addEventListener('click', async () => {
  if (!client.ready || busy || stream.active || document.hidden) return;
  const epoch = actionEpoch;
  const started = performance.now();
  busy = true;
  update();
  try {
    const sent = await client.send(LEARNED_PACKET, () => epoch === actionEpoch && !document.hidden && performance.now() - started < 150);
    log(sent ? 'One learned-controller packet sent (no firmware acknowledgement for this command).' : 'Single packet cancelled before transmission.');
  } catch (error) { report(error); }
  finally { busy = false; update(); }
});

ui.hold.addEventListener('pointerdown', event => {
  if (event.button !== 0 || ui.hold.disabled || stream.active) return;
  event.preventDefault();
  ui.hold.focus();
  activePointer = event.pointerId;
  ui.hold.setPointerCapture(event.pointerId);
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
  stream.start();
});
window.addEventListener('keyup', event => { if (event.key === activeKey) stop(); });
window.addEventListener('keydown', event => { if (event.key === 'Escape') stop(); });
ui.hold.addEventListener('blur', stop);
ui.stop.addEventListener('click', stop);
window.addEventListener('blur', stop);
document.addEventListener('visibilitychange', () => { if (document.hidden) stop(); });
window.addEventListener('pagehide', () => { stop(); client.disconnect(); });

if (!window.isSecureContext || !navigator.bluetooth) {
  ui.compatibility.hidden = false;
  ui.compatibility.textContent = !window.isSecureContext
    ? 'Bluetooth requires HTTPS or localhost. A plain HTTP address on your local network will not work.'
    : 'Web Bluetooth is unavailable. Use Chrome on a supported platform; on Linux enable experimental Web Platform features. Chrome on iOS is unsupported.';
  ui.connect.disabled = true;
}

// Optional agent integration only stages text or reads status. Connecting and
// sending motor-triggering packets remain explicit actions in the visible UI.
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
      ui['write-status'].textContent = 'Edited · not sent';
      return { staged: commands.length, sent: false };
    },
  });
  window.addEventListener('pagehide', () => lifecycle.abort(), { once: true });
}
