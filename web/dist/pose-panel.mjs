import { DEFAULT_POSE_URL, PoseConnection } from './pose.mjs';

export function mountPosePanel(root, log) {
  const get = id => root.querySelector(`#${id}`);
  const address = get('pose-address');
  const button = get('pose-connect');
  const stateLabel = get('pose-state');
  const status = get('pose-status');
  const ageLabel = get('pose-age');
  const timestamp = get('pose-timestamp');
  const fields = ['px', 'py', 'pz', 'qx', 'qy', 'qz', 'qw'].map(key => get(`pose-${key}`));
  const text = (element, value) => { if (element.textContent !== value) element.textContent = value; };
  let frame;
  let previousState = 'disconnected';
  const client = new PoseConnection({ onChange() {
    if (client.state !== previousState) {
      if (client.state === 'waiting') log('Pose server connected. Waiting for a sample.', 'OK');
      if (client.state === 'error') log(client.message, 'ERROR');
      if (client.state === 'disconnected') log('Pose server disconnected.');
      previousState = client.state;
    }
    if (!frame) frame = requestAnimationFrame(render);
  } });

  function render() {
    frame = null;
    const { state, message, pose, age } = client.snapshot;
    root.dataset.state = state;
    text(stateLabel, state.toUpperCase());
    address.disabled = client.active;
    text(button, state === 'connecting' ? 'Cancel' : client.active ? 'Disconnect' : 'Connect');
    text(status, ({ disconnected: 'Start the local pose server, then connect.',
      connecting: 'Connecting… Allow local network access if prompted.',
      waiting: 'Connected. Waiting for the first pose…',
      live: 'Receiving poses.', stale: 'Pose stream is stale. Waiting for a new sample.', error: message })[state]);
    text(ageLabel, age === null ? 'No samples' : `${(age / 1000).toFixed(1)} s since last pose`);
    text(timestamp, pose ? `${pose.timestamp.toFixed(3)} s` : '—');
    const values = pose ? [...pose.position, ...pose.quaternion] : [];
    fields.forEach((element, index) => text(element, pose ? values[index].toFixed(index < 3 ? 3 : 4) : '—'));
  }

  address.value = DEFAULT_POSE_URL;
  try { address.value = sessionStorage.getItem('crazyflie.poseAddress') || DEFAULT_POSE_URL; } catch { /* Optional. */ }
  get('pose-form').addEventListener('submit', event => {
    event.preventDefault();
    if (client.active) client.disconnect();
    else {
      try { sessionStorage.setItem('crazyflie.poseAddress', address.value.trim()); } catch { /* Optional. */ }
      void client.connect(address.value.trim());
    }
  });
  const refresh = () => { if (!frame) frame = requestAnimationFrame(render); };
  let timer = setInterval(refresh, 100);
  window.addEventListener('pagehide', () => {
    client.disconnect();
    clearInterval(timer);
    cancelAnimationFrame(frame);
    frame = null;
  });
  window.addEventListener('pageshow', event => {
    if (event.persisted) timer = setInterval(refresh, 100);
  });
  render();
}
