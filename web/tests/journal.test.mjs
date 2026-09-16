import test from 'node:test';
import assert from 'node:assert/strict';
import { Journal } from '../dist/journal.mjs';

test('history stays on the selected event as new events arrive, then rejoins the live view', () => {
  const journal = new Journal();
  journal.append('First');
  journal.append('Second');
  journal.older();
  journal.append('Third');
  assert.equal(journal.current.message, 'First');
  assert.equal(journal.live, false);
  journal.newer();
  assert.equal(journal.current.message, 'Second');
  journal.newer();
  assert.equal(journal.current.message, 'Third');
  assert.equal(journal.live, true);
  journal.append('Fourth');
  assert.equal(journal.current.message, 'Fourth');
});

test('bounded history retains a valid selection when the oldest selected event expires', () => {
  const journal = new Journal(3);
  for (const value of ['One', 'Two', 'Three']) journal.append(value);
  journal.older();
  journal.older();
  journal.append('Four', 'ERROR');
  assert.equal(journal.entries.length, 3);
  assert.equal(journal.current.message, 'Two');
  journal.older();
  assert.equal(journal.current.message, 'Two');
  journal.latest();
  assert.equal(journal.current.message, 'Four');
  assert.equal(journal.current.level, 'ERROR');
  assert.equal(journal.live, true);
});
