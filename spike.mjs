// spike.mjs: THROWAWAY spike. git diff -> dependency-based risk ranking -> one LLM call.
//
// Usage:  node spike.mjs <repoPath> <baseRef> <headRef> [srcDir]
// Setup:  npm init -y && npm i simple-git madge @google/genai   (pin the versions)
//         set GEMINI_API_KEY (PowerShell: $env:GEMINI_API_KEY = "..."); never in a committed file; .env goes in .gitignore
//
// RULES: public repos only. Never send private code or participant data to the LLM.
// Steps marked TODO(you) are yours to write. Check each library's docs for the exact API.

import { simpleGit } from 'simple-git';
import madge from 'madge';
import { GoogleGenAI } from '@google/genai';
import fs from 'node:fs';
import path from 'node:path';

const MAX_FILES_TO_LLM = Number(process.env.TOP_N ?? 3);        // only the top-N ranked files go to the model
const MAX_HUNK_CHARS = Number(process.env.MAX_HUNK ?? 6000);    // truncate each file's diff to cap tokens/cost
const MODEL = process.env.MODEL ?? 'gemini-3-flash-preview'; // check AI Studio for the current model name

// ---------------------------------------------------------------------------
// STEP 1: read the diff
// Return: [{ file, added, removed, hunk }]   (file paths relative to the repo root)
// Hints: git.diffSummary([base, head]) gives per-file insertions/deletions;
//        git.diff([base, head, '--', file]) gives the raw hunk text.
//        Skip binary files and lockfiles (package-lock.json etc.), which will dominate.
// ---------------------------------------------------------------------------
const LOCKFILES = /(^|\/)(package-lock\.json|npm-shrinkwrap\.json|yarn\.lock|pnpm-lock\.yaml)$/;
const CODE_EXT = /\.(js|mjs|cjs|jsx|ts|tsx)$/;

// Normalise to forward slashes with no leading "./" so git and madge paths can be compared.
const toPosix = (p) => p.replace(/\\/g, '/').replace(/^\.\//, '');

async function getChangedFiles(repoPath, base, head, srcDir = '.') {
  const git = simpleGit(repoPath);
  const src = toPosix(srcDir);
  const inSrc = (f) => src === '.' || src === '' || f.startsWith(src.replace(/\/?$/, '/'));

  // --no-renames: report a rename as delete + add, so file names stay plain paths
  const summary = await git.diffSummary([base, head, '--no-renames']);
  const out = [];
  for (const f of summary.files) {
    const file = toPosix(f.file);
    if (f.binary) continue;                 // binary files have no text diff
    if (LOCKFILES.test(file)) continue;     // lockfiles dominate the ranking
    if (!CODE_EXT.test(file)) continue;     // madge only knows code files
    if (!inSrc(file)) continue;             // keep only files madge will scan
    const hunk = await git.diff([base, head, '--no-renames', '--', file]);
    out.push({ file, added: f.insertions, removed: f.deletions, hunk });
  }
  return out;
}

// ---------------------------------------------------------------------------
// STEP 2: who depends on each changed file?
// Return: Map<file, string[]>  (the files that import it)
// Hints: madge(srcDirAbs, { fileExtensions: ['js','ts','jsx','tsx'], tsConfig? }).then(m => m.obj())
//        gives { file: [its imports] }; invert it to get dependents.
// GOTCHA TO NOTE IN YOUR JOURNAL: madge paths are relative to srcDir, git paths to the repo
// root. You will need to map between them (prefix/strip srcDir). Also check path aliases.
// ---------------------------------------------------------------------------
async function buildDependents(repoPath, srcDir) {
  const absSrc = path.resolve(repoPath, srcDir);
  const m = await madge(absSrc, { fileExtensions: ['js', 'ts', 'jsx', 'tsx'] });
  const graph = m.obj(); // { 'a.js': ['b.js', 'c.js'] }  means a imports b and c

  // madge paths are relative to srcDir; git paths are relative to the repo root.
  const src = toPosix(srcDir);
  const toRepoPath = (p) => {
    const q = toPosix(p);
    return src === '.' || src === '' ? q : path.posix.join(src, q);
  };

  // Invert: importer -> [imports]  becomes  imported -> [importers]
  const dependents = new Map();
  for (const [importer, imports] of Object.entries(graph)) {
    for (const imported of imports) {
      const key = toRepoPath(imported);
      if (!dependents.has(key)) dependents.set(key, []);
      dependents.get(key).push(toRepoPath(importer));
    }
  }
  return dependents;
}

// ---------------------------------------------------------------------------
// STEP 3: crude risk score
// First guess:  score = (added + removed) * (1 + dependents.length)
// Return the files sorted by score, highest first, each with { score, dependents }.
// Deliberately crude. Write down in your journal anything that ranks wrongly.
// ---------------------------------------------------------------------------
function scoreFiles(changed, dependentsMap) {
  return changed
    .map((f) => {
      const dependents = [...new Set(dependentsMap.get(f.file) ?? [])];
      const score = (f.added + f.removed) * (1 + dependents.length);
      return { ...f, dependents, score };
    })
    .sort((a, b) => b.score - a.score);
}

// ---------------------------------------------------------------------------
// STEP 4: the LLM prompt (written for you: tweak it, but note every change)
// The "evidence" field forces the model to point at the diff, which makes grading easier.
// ---------------------------------------------------------------------------
const SYSTEM_PROMPT = `You are a careful code-change analyst helping a developer who has just pulled someone else's changes.
Rules:
- Use ONLY the diff text and dependency information provided. Do not assume code you cannot see.
- Refer to files by their exact path as given. Never mention a file that was not provided.
- If something cannot be determined from what you were given, say so in "unknowns".
- Do not invent bugs. If a change looks routine, say it looks low-risk.
- Reply with ONE JSON object and nothing else (no prose, no code fences).

Schema:
{
  "summary": "plain-English description of what changed (<= 300 chars)",
  "risks": [
    {
      "file": "exact path from the input",
      "concern": "what could go wrong (<= 200 chars)",
      "evidence": "short pointer to the diff that supports this (<= 150 chars)",
      "confidence": "high" | "medium" | "low"
    }
  ],
  "check_first": ["exact file paths, most important first"],
  "unknowns": ["things you could not determine from the provided context"]
}`;

function buildUserPrompt({ repoName, base, head, topFiles }) {
  const blocks = topFiles.map((f, i) => {
    const truncated = f.hunk.length > MAX_HUNK_CHARS;
    return [
      `### File ${i + 1}: ${f.file}`,
      `Risk score: ${f.score} (+${f.added} / -${f.removed} lines)`,
      `Imported by (${f.dependents.length}): ${f.dependents.slice(0, 15).join(', ') || 'none found'}`,
      `Diff${truncated ? ' (TRUNCATED)' : ''}:`,
      f.hunk.slice(0, MAX_HUNK_CHARS),
    ].join('\n');
  });
  return `Repository: ${repoName}\nComparing ${base} -> ${head}\n\n${blocks.join('\n\n')}\n\nReturn ONLY the JSON object.`;
}

// ---------------------------------------------------------------------------
// STEP 5: one LLM call, logging tokens and time
// ---------------------------------------------------------------------------
async function callLLM(userPrompt) {
  const ai = new GoogleGenAI({}); // reads GEMINI_API_KEY from the environment
  const t0 = Date.now();
  const res = await ai.models.generateContent({
    model: MODEL,
    contents: userPrompt,
    config: {
      systemInstruction: SYSTEM_PROMPT,
      maxOutputTokens: 4000, // thinking models can spend part of this budget before the JSON
      responseMimeType: 'application/json',
    },
  });
  return {
    text: res.text ?? '',
    model: MODEL,
    ms: Date.now() - t0,
    inputTokens: res.usageMetadata?.promptTokenCount,
    outputTokens: res.usageMetadata?.candidatesTokenCount,
  };
}

// Tolerant parse: strip fences, grab the first {...}. Returns null if it still fails.
function parseJson(text) {
  const cleaned = text.trim().replace(/^```(?:json)?\s*/i, '').replace(/\s*```$/, '');
  const m = cleaned.match(/\{[\s\S]*\}/);
  try { return JSON.parse(m ? m[0] : cleaned); } catch { return null; }
}

// Cheap automatic check: did the model mention any file it was NOT given?
function findInventedFiles(parsed, givenFiles) {
  const given = new Set(givenFiles);
  const mentioned = [
    ...(parsed?.risks ?? []).map((r) => r.file),
    ...(parsed?.check_first ?? []),
  ];
  return mentioned.filter((f) => !given.has(f));
}

// ---------------------------------------------------------------------------
async function main() {
  const [repoPath, base, head, srcDir = '.'] = process.argv.slice(2);
  if (!repoPath || !base || !head) {
    console.error('Usage: node spike.mjs <repoPath> <baseRef> <headRef> [srcDir]');
    process.exit(1);
  }

  const changed = await getChangedFiles(repoPath, base, head, srcDir);
  const dependents = await buildDependents(repoPath, srcDir);
  const ranked = scoreFiles(changed, dependents);

  console.table(ranked.map(({ file, added, removed, score, dependents }) =>
    ({ file, added, removed, dependents: dependents.length, score })));

  const topFiles = ranked.slice(0, MAX_FILES_TO_LLM);
  const userPrompt = buildUserPrompt({ repoName: path.basename(repoPath), base, head, topFiles });
  const llm = await callLLM(userPrompt);
  const parsed = parseJson(llm.text);

  console.log('\n--- LLM result ---');
  console.log(parsed ?? `(could not parse JSON)\n${llm.text}`);
  console.log(`\ntokens in/out: ${llm.inputTokens}/${llm.outputTokens}   time: ${llm.ms} ms`);
  console.log('invented files:', findInventedFiles(parsed, topFiles.flatMap((f) => [f.file, ...f.dependents])));

  // Log everything for your dev journal and, later, the dissertation's evaluation.
  fs.mkdirSync('runs', { recursive: true });
  const stamp = new Date().toISOString().replace(/[:.]/g, '-');
    fs.writeFileSync(`runs/${stamp}.json`, JSON.stringify(
    { repoPath, base, head, settings: { MAX_FILES_TO_LLM, MAX_HUNK_CHARS, MODEL }, ranked, userPrompt, llm, parsed }, null, 2));
}

main().catch((e) => { console.error(e); process.exit(1); });
