// sync-results.mjs: extend results.csv from runs/*.json without touching hand-scored cells.
//
// Run from the repo root.
//   Dry run: node scripts/sync-results.mjs --old 2fa3b1d --new <hash> --new-from <first-new-log-file>
//   Apply:   same command plus --write
//
// - Existing rows keep every hand-scored cell. Blank new columns are filled from their log when possible.
// - Logs not yet listed in results.csv get a new row. Failures (unparseable output, API errors) are
//   marked detected=failure, never "no": a failed run is not a miss.
// - harness = log.harness if the log recorded one, else --old for log files before --new-from, else --new.

import fs from 'node:fs';
import path from 'node:path';

const argv = process.argv.slice(2);
const opt = (name) => { const i = argv.indexOf(`--${name}`); return i >= 0 ? argv[i + 1] : undefined; };
const WRITE = argv.includes('--write');
const OLD = opt('old'), NEW = opt('new'), NEW_FROM = opt('new-from');
if (!OLD || !NEW || !NEW_FROM) {
  console.error('Need --old <hash> --new <hash> --new-from <log file name>');
  process.exit(1);
}

const COLUMNS = ['case', 'run', 'harness', 'top_n', 'model', 'bug_file_rank', 'bug_in_prompt', 'detected',
  'localized', 'concern_correct', 'other_concerns', 'invented_flag', 'finish_reason', 'parse_ok',
  'tokens_in', 'tokens_out', 'thoughts_tokens', 'time_ms', 'log_file', 'notes'];

const readText = (p) => fs.readFileSync(p, 'utf8').replace(/^\uFEFF/, '');

function parseCsv(text) {
  const rows = []; let row = [], field = '', q = false;
  const flush = () => { row.push(field); field = ''; if (row.some((x) => x !== '')) rows.push(row); row = []; };
  for (let i = 0; i < text.length; i++) {
    const c = text[i];
    if (q) {
      if (c === '"' && text[i + 1] === '"') { field += '"'; i++; }
      else if (c === '"') q = false;
      else field += c;
    } else if (c === '"') q = true;
    else if (c === ',') { row.push(field); field = ''; }
    else if (c === '\n' || c === '\r') { if (c === '\r' && text[i + 1] === '\n') i++; flush(); }
    else field += c;
  }
  if (field !== '' || row.length) flush();
  return rows;
}
const esc = (v) => { const s = String(v ?? ''); return /[",\n\r]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s; };

// ground truth
const cases = fs.existsSync('cases.json') ? (JSON.parse(readText('cases.json')).cases ?? []) : [];
const byHead = new Map(cases.map((c) => [c.head, c]));

// existing results
let existing = [];
if (fs.existsSync('results.csv')) {
  const [header, ...body] = parseCsv(readText('results.csv'));
  existing = body.map((r) => Object.fromEntries(header.map((h, i) => [h, r[i] ?? ''])));
}
const listed = new Set(existing.map((r) => r.log_file));

function loadLog(file) {
  return JSON.parse(readText(path.join('runs', file)));
}

function analyse(file) {
  const log = loadLog(file);
  const llm = log.llm ?? {};
  const parsed = log.parsed && typeof log.parsed === 'object' ? log.parsed : null;
  const c = byHead.get(log.head);
  const bugFile = c?.bugFile ?? null;
  const prompt = log.userPrompt ?? '';
  const risks = Array.isArray(parsed?.risks) ? parsed.risks : [];
  const sent = (prompt.match(/### File \d+:/g) ?? []).length;
  const rankIdx = bugFile ? (log.ranked ?? []).findIndex((r) => r.file === bugFile) : -1;
  const inPrompt = bugFile ? prompt.includes(`: ${bugFile}\n`) : null;
  const failed = !parsed || Boolean(llm.error || log.error);

  const mentioned = [...risks.map((r) => r.file), ...(Array.isArray(parsed?.check_first) ? parsed.check_first : [])]
    .filter((f) => typeof f === 'string');
  const invented = [...new Set(mentioned.filter((f) => !prompt.includes(f)))];

  const notes = [];
  if (failed) notes.push(`FAILURE (not a miss): ${llm.error || log.error || 'output could not be parsed as JSON'}`);
  else if (bugFile && !inPrompt) notes.push(`bug file not sent (rank ${rankIdx + 1}, top_n ${sent})`);
  if (llm.attempts > 1) notes.push(`attempts=${llm.attempts}`);

  let detected;
  if (failed) detected = 'failure';
  else if (!bugFile) detected = 'n/a';
  else if (!inPrompt) detected = 'no';
  else detected = risks.some((r) => r.file === bugFile) ? 'yes' : 'no';

  return {
    case: c?.id ?? `control-${log.head}`,
    harness: log.harness ?? (file >= NEW_FROM ? NEW : OLD),
    top_n: sent,
    model: log.settings?.MODEL ?? llm.model ?? '',
    bug_file_rank: rankIdx >= 0 ? rankIdx + 1 : '',
    bug_in_prompt: bugFile ? (inPrompt ? 'yes' : 'no') : 'n/a',
    detected,
    localized: failed ? 'n/a' : '',            // hand-scored
    concern_correct: failed ? 'n/a' : '',      // hand-scored
    other_concerns: failed ? '' : risks.filter((r) => r.file !== bugFile).length,
    invented_flag: failed ? '' : (invented.length ? invented.join(' ') : 'none'),
    finish_reason: llm.finishReason ?? '',
    parse_ok: parsed ? 'true' : 'false',
    tokens_in: llm.inputTokens ?? '',
    tokens_out: llm.outputTokens ?? '',
    thoughts_tokens: llm.thoughtsTokens ?? '',
    time_ms: llm.ms ?? '',
    log_file: file,
    notes: notes.join('; '),
  };
}

// 1. fill blank new columns on existing rows (never overwrite a filled cell)
for (const r of existing) {
  if (!r.harness) r.harness = 'unrecorded';
  const f = r.log_file;
  if (f && fs.existsSync(path.join('runs', f))) {
    const a = analyse(f);
    for (const k of ['finish_reason', 'parse_ok', 'thoughts_tokens']) if (!r[k]) r[k] = a[k];
  } else if (f) {
    console.warn(`warning: results.csv lists ${f} but runs/${f} does not exist`);
  }
}

// 2. new rows for unlisted logs
const files = fs.existsSync('runs') ? fs.readdirSync('runs').filter((f) => f.endsWith('.json')).sort() : [];
const fresh = files.filter((f) => !listed.has(f)).map(analyse);
const counts = new Map();
const keyOf = (r) => `${r.case}|${r.top_n}|${r.harness}`;
for (const r of existing) counts.set(keyOf(r), (counts.get(keyOf(r)) ?? 0) + 1);
for (const r of fresh) { const k = keyOf(r); counts.set(k, (counts.get(k) ?? 0) + 1); r.run = counts.get(k); }

// report
const tally = {};
for (const r of fresh) { const k = `${r.harness} / ${r.detected}`; tally[k] = (tally[k] ?? 0) + 1; }
console.log(`existing rows: ${existing.length}   new rows from unlisted logs: ${fresh.length}`);
console.log('new rows by harness / detected:', tally);
for (const r of fresh.filter((x) => x.detected === 'failure')) console.log('  failure:', r.log_file, '-', r.notes);

if (!WRITE) { console.log('\n(dry run: nothing written; add --write to apply)'); process.exit(0); }
const all = [...existing, ...fresh];
fs.writeFileSync('results.csv', [COLUMNS.join(','), ...all.map((r) => COLUMNS.map((c) => esc(r[c])).join(','))].join('\n') + '\n');
console.log(`wrote results.csv (${all.length} rows)`);
