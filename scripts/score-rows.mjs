// score-rows.mjs: set localized / concern_correct (and optionally notes) for named log files in
// results.csv, without touching any other cell. Avoids hand-editing a 20-column CSV.
//
// Usage:  node scripts\score-rows.mjs scores.json [results.csv] [--force]
//
// scores.json maps a log file name to the cells to set. Example:
//   { "2026-11-15T02-23-39-669Z.json": { "localized": "yes", "concern_correct": "partial",
//                                        "notes": "NaN only when first hex digit is A-F" } }
//
// Safety rules:
//  - Only empty localized / concern_correct cells are filled. A non-empty cell that differs is an
//    error unless you pass --force, so earlier hand scoring is never overwritten silently.
//  - "notes", if given, REPLACES the notes cell (omit it to leave the notes alone).
//  - Every log file named in scores.json must exist in the CSV, or nothing is written.
//  - Avoid commas and double quotes in notes (use semicolons).
import fs from 'node:fs';

const args = process.argv.slice(2).filter((a) => !a.startsWith('--'));
const force = process.argv.includes('--force');
const [scoresPath, csvPath = 'results.csv'] = args;
if (!scoresPath) { console.error('Usage: node scripts\\score-rows.mjs scores.json [results.csv] [--force]'); process.exit(1); }

function parseCsv(text) {
  const rows = []; let row = []; let field = ''; let inQ = false;
  for (let i = 0; i < text.length; i++) {
    const c = text[i];
    if (inQ) {
      if (c === '"') { if (text[i + 1] === '"') { field += '"'; i++; } else inQ = false; }
      else field += c;
    } else if (c === '"') inQ = true;
    else if (c === ',') { row.push(field); field = ''; }
    else if (c === '\n' || c === '\r') {
      if (c === '\r' && text[i + 1] === '\n') i++;
      row.push(field); field = ''; rows.push(row); row = [];
    } else field += c;
  }
  if (field !== '' || row.length) { row.push(field); rows.push(row); }
  return rows;
}
const esc = (v) => (/[",\n\r]/.test(v) ? `"${v.replace(/"/g, '""')}"` : v);

const raw = fs.readFileSync(csvPath, 'utf8');
const bom = raw.charCodeAt(0) === 0xfeff ? '\ufeff' : '';
const text = bom ? raw.slice(1) : raw;
const eol = text.includes('\r\n') ? '\r\n' : '\n';
const endsWithEol = /\r?\n$/.test(text);
const rows = parseCsv(text);
const header = rows[0];
const col = Object.fromEntries(header.map((h, i) => [h, i]));
for (const need of ['log_file', 'localized', 'concern_correct', 'notes']) {
  if (!(need in col)) throw new Error(`Column "${need}" not found in ${csvPath}`);
}
const scores = JSON.parse(fs.readFileSync(scoresPath, 'utf8'));

const problems = []; const changes = [];
for (const [logFile, cells] of Object.entries(scores)) {
  const matches = rows.slice(1).filter((r) => r[col.log_file] === logFile);
  if (matches.length !== 1) { problems.push(`${logFile}: found in ${matches.length} rows (expected 1)`); continue; }
  const r = matches[0];
  for (const key of ['localized', 'concern_correct', 'notes']) {
    if (!(key in cells)) continue;
    const next = String(cells[key]);
    if (/[",]/.test(next)) problems.push(`${logFile}: "${key}" contains a comma or quote; use semicolons`);
    const old = r[col[key]];
    if (old === next) continue;
    if (key !== 'notes' && old !== '' && !force) { problems.push(`${logFile}: ${key} already "${old}" (use --force to overwrite)`); continue; }
    changes.push({ logFile, key, old, next, r });
  }
}
if (problems.length) { console.error('Nothing written:\n  ' + problems.join('\n  ')); process.exit(1); }
if (!changes.length) { console.log('No changes needed.'); process.exit(0); }

for (const c of changes) { c.r[col[c.key]] = c.next; console.log(`${c.logFile.slice(11, 19)}  ${c.key}: "${c.old}" -> "${c.next}"`); }
const bad = rows.filter((r) => r.length !== header.length);
if (bad.length) { console.error('Refusing to write: a row has the wrong number of fields.'); process.exit(1); }
fs.writeFileSync(csvPath, bom + rows.map((r) => r.map(esc).join(',')).join(eol) + (endsWithEol ? eol : ''));
console.log(`\nWrote ${csvPath} (${rows.length - 1} data rows, ${changes.length} cells changed).`);
