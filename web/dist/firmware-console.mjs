// Console-only BLE recovery. Damaged data must never become a parameter reply.
export class ConsoleReceiver {
  constructor(onText) {
    this.onText = onText;
    this.decoder = new TextDecoder();
    this.mode = null;
    this.buffered = [];
    this.pending = null;
  }
  text(bytes) {
    this.onText(this.decoder.decode(bytes, { stream: true }).replace(/[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]/g, '�'));
  }
  gap(reason, newline = false) {
    this.onText(this.decoder.decode() + `⟦${reason}⟧${newline ? '\n' : ''}`);
    this.decoder = new TextDecoder();
  }
  packet(data) {
    if (data.length < 2) return;
    if (data.every(byte => byte === 0)) {
      this.gap('BLE data lost', true);
    } else if ((data[0] & 0xf3) === 0) {
      this.text(data.slice(1));
    }
  }
  push(frame, mode) {
    if (frame.length < 2 || frame.length > 20) return;
    if (!this.mode) {
      if (!mode) {
        if (this.buffered.length === 256) {
          this.buffered.shift();
          if (!this.dropped) { this.gap('unidentified BLE data omitted', true); this.dropped = true; }
        }
        this.buffered.push(frame.slice());
        return;
      }
      this.mode = mode;
      for (const saved of this.buffered) this.decode(saved);
      this.buffered = [];
    }
    this.decode(frame);
  }
  decode(frame) {
    const start = Boolean(frame[0] & 0x80);
    const pid = (frame[0] >> 5) & 3;
    const data = frame.slice(1);
    const pending = this.pending;
    this.pending = null;
    if (pending) {
      if (this.mode === 'legacy' && frame[0] === pending.header && data.length === pending.total - 19) {
        // The legacy transmitter repeats its start header, skips byte 19 and
        // reads one byte past the packet. Keep known bytes; mark, never guess,
        // the missing byte, and discard the out-of-bounds trailing byte.
        if (pending.data.every(byte => byte === 0) && data.slice(0, -1).every(byte => byte === 0)) {
          this.gap('BLE data lost', true);
        } else if ((pending.data[0] & 0xf3) === 0) {
          this.text(pending.data.slice(1));
          this.gap('missing byte');
          this.text(data.slice(0, -1));
        }
        return;
      }
      if (this.mode === 'standard' && !start && pid === pending.pid && pending.data.length + data.length === pending.total) {
        this.packet(Uint8Array.of(...pending.data, ...data));
        return;
      }
      if ((pending.data[0] & 0xf3) === 0) {
        this.text(pending.data.slice(1));
        this.gap('incomplete BLE packet', true);
      }
    }
    if (!start) return;
    const total = (frame[0] & 31) + (this.mode === 'standard' ? 1 : 0);
    if (total < 1 || total > 31) return;
    if (total === data.length) this.packet(data);
    else if (data.length === 19 && total > 19) this.pending = { header: frame[0], pid, total, data };
  }
  close() {
    if (this.pending && (this.pending.data[0] & 0xf3) === 0) {
      this.text(this.pending.data.slice(1));
      this.gap('incomplete BLE packet', true);
    }
    const rest = this.decoder.decode();
    if (rest) this.onText(rest);
    this.pending = null;
    this.buffered = [];
  }
}

// Bounded capture and a frozen review snapshot. Pausing affects the view only.
export class ConsoleBuffer {
  constructor(limit = 1000) {
    this.limit = limit;
    this.lines = [];
    this.partial = '';
    this.received = 0;
    this.snapshot = null;
    this.end = Infinity;
  }
  append(text) {
    for (const char of text.replace(/\r/g, '')) {
      if (char === '\n') this.commit();
      else {
        this.partial += char;
        if (this.partial.length >= 4096) {
          this.partial += ' ⟦line continued⟧';
          this.commit();
        }
      }
    }
  }
  commit() {
    this.lines.push(this.partial);
    this.partial = '';
    ++this.received;
    if (this.lines.length > this.limit) this.lines.shift();
  }
  get live() { return this.snapshot === null; }
  contents() { return [...this.lines, ...(this.partial ? [this.partial] : [])]; }
  pause() {
    if (this.live) { this.snapshot = this.contents(); this.end = Infinity; }
  }
  resume() { this.snapshot = null; this.end = Infinity; }
  page(columns, count) {
    columns = Math.max(1, columns);
    count = Math.max(1, count);
    const rows = [];
    for (const line of this.snapshot ?? this.contents()) {
      const chars = Array.from(line.replace(/\t/g, '    '));
      if (!chars.length) rows.push('');
      for (let i = 0; i < chars.length; i += columns) rows.push(chars.slice(i, i + columns).join(''));
    }
    const end = Math.min(this.end, rows.length);
    const start = Math.max(0, end - count);
    return { text: rows.slice(start, end).join('\n'), start, end, total: rows.length, older: start > 0, newer: end < rows.length };
  }
  older(columns, count) {
    this.pause();
    this.end = Math.max(1, this.page(columns, count).start);
  }
  newer(columns, count) { this.end = Math.min(this.page(columns, count).total, this.page(columns, count).end + count); }
}
