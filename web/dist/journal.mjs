// A bounded journal with a stable history selection. Incoming events never move
// the operator away from an event they are reviewing.
export class Journal {
  constructor(limit = 200) {
    this.limit = limit;
    this.entries = [];
    this.selected = null;
    this.sequence = 0;
  }
  append(message, level = 'INFO', time = new Date()) {
    this.entries.push({ id: ++this.sequence, message, level, time });
    if (this.entries.length > this.limit) this.entries.shift();
    if (this.selected !== null && this.selected < this.entries[0].id) this.selected = this.entries[0].id;
  }
  get index() {
    return this.selected === null ? this.entries.length - 1 : this.entries.findIndex(entry => entry.id === this.selected);
  }
  get current() { return this.entries[this.index]; }
  get live() { return this.selected === null; }
  older() { this.selected = this.entries[Math.max(0, this.index - 1)]?.id ?? null; }
  newer() {
    const next = this.index + 1;
    this.selected = next >= this.entries.length - 1 ? null : this.entries[next].id;
  }
  latest() { this.selected = null; }
}
