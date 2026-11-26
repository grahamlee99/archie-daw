// Builds the Archie DAW manual as ONE self-contained HTML page (it prints to PDF from any browser: Ctrl+P, Save as PDF).
// Run: node scripts/build-desktop-manual.mjs [outFile]      The installer script runs it and puts the page in the install folder (manual\Archie-DAW-Manual.html); the program's Help menu opens it.
// The key tables and the command list are GENERATED from desktop-host/tools/studio-keymap.json and studio-tools.json, so the manual cannot drift from the real keys and commands.
// It is written from our own program in plain words. Nothing in it is taken from any other program's manual.
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const tools = JSON.parse(fs.readFileSync(path.join(root, "desktop-host", "tools", "studio-tools.json"), "utf8"));
const keymap = JSON.parse(fs.readFileSync(path.join(root, "desktop-host", "tools", "studio-keymap.json"), "utf8"));
const esc = (s) => String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
export function readVersion() { const m = fs.readFileSync(path.join(root, "desktop-host", "CMakeLists.txt"), "utf8").match(/project\(ArchieStudioHost VERSION (\d+\.\d+\.\d+)/); return m ? m[1] : "?"; }

const toolByName = new Map(tools.tools.map((t) => [t.name, t]));
const label = (cmd) => (keymap.uiCommands && keymap.uiCommands[cmd]) || (toolByName.get(cmd) && toolByName.get(cmd).summary) || cmd;
const ctxOrder = ["global", ...Object.keys(keymap.contexts || {}).filter((c) => c !== "global")];
const ready = (keymap.bindings || []).filter((b) => b.status === "ready");

export function buildManual() {
  const v = readVersion();
  let h = `<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Archie DAW manual</title><style>
:root{color-scheme:light dark;--bg:#fff;--fg:#1d2428;--mut:#5a666d;--ac:#1f7a4d;--line:#d6dde0;--code:#eef3f1}
@media (prefers-color-scheme:dark){:root{--bg:#161b1e;--fg:#e4eaec;--mut:#9aa8ae;--ac:#4fc58a;--line:#2b353a;--code:#1f282c}}
body{margin:0;background:var(--bg);color:var(--fg);font:16px/1.55 system-ui,Segoe UI,sans-serif}main{max-width:820px;margin:0 auto;padding:24px 20px 80px}
h1{font-size:2rem;margin:.2em 0}h2{margin-top:2em;border-bottom:2px solid var(--ac);padding-bottom:.2em}h3{margin-top:1.4em}
table{border-collapse:collapse;width:100%;margin:.6em 0}th,td{text-align:left;padding:6px 8px;border-bottom:1px solid var(--line);vertical-align:top}th{color:var(--mut);font-weight:600}
kbd{background:var(--code);border:1px solid var(--line);border-radius:4px;padding:1px 6px;font:13px ui-monospace,Consolas,monospace;white-space:nowrap}
.mut{color:var(--mut)}.box{background:var(--code);border-left:4px solid var(--ac);padding:10px 14px;margin:1em 0}code{background:var(--code);padding:1px 5px;border-radius:3px}
@media print{main{max-width:none;padding:0}h2{break-after:avoid}tr{break-inside:avoid}}
</style></head><body><main>
<h1>Archie DAW</h1><p class="mut">Manual for version ${esc(v)}. The newest download on the Desktop Studio page is always the latest version; each new update has its own number and this manual comes with it. The program is in testing: tell us what you find.</p>

<h2>1. What it is</h2>
<p>Archie DAW is a free program for your own computer where you make music. You can record and arrange sound, play your own VST3 plug-ins, mix, and write a finished file. The pages of Ask Archie open inside it, and everything shares one play button.</p>
<p>Making music is five jobs, over and over: <b>capture</b> an idea, <b>shape</b> it, <b>hear</b> it as a whole, <b>finish</b> it, and <b>share or keep</b> it. The program is built around those. Where a job has a traditional name that is confusing, we use a plain one and give the usual name once.</p>

<h2>2. Getting started</h2>
<ol><li>Run the installer. It installs for you only (no administrator), puts <b>Archie DAW</b> in the Start menu and, if you ticked it, on your Desktop. It needs the Microsoft Edge WebView2 runtime, which Windows already has on most computers.</li>
<li>Open it. Choose your sound card in <b>Options</b> if you do not hear anything. Your VST3 plug-ins are found automatically (your standard VST3 folder, and the Local Builder's output folder).</li>
<li>Press <kbd>Space</kbd> to play. Right-click almost anything: tracks, clips and plug-ins all have a menu.</li>
<li><b>File &gt; Save</b> writes your song to a file on your own computer. The next Save goes to the same file. <b>File &gt; Open Recent</b> brings back your last ten songs.</li></ol>

<h2>3. Plain words</h2>
<table><tr><th>We say</th><th>You may know it as</th></tr>
<tr><td><b>Sub routing</b> (group tracks and send them together)</td><td>folders, buses, sub-mixes</td></tr>
<tr><td><b>Sends and returns</b> (FX channels)</td><td>aux sends, FX buses</td></tr>
<tr><td><b>Clip</b>: a piece of music or sound on the timeline</td><td>region, block, item</td></tr>
<tr><td><b>Warp</b>: make a sound follow the song's tempo</td><td>time-stretch, flex</td></tr>
<tr><td><b>Bounce</b>: turn a clip into a sound</td><td>freeze, render in place</td></tr>
<tr><td><b>Template</b>: a song you start new songs from</td><td>project template</td></tr></table>

<h2>4. Things the program does to keep you safe</h2>
<ul><li><b>Your song is kept every minute.</b> If the program is ever closed by a crash, the next start offers your song back.</li>
<li><b>A plug-in that closes the program is remembered.</b> The next time, the program tells you which one it was and asks before loading it again. It never blocks a plug-in for good. When you open a song that contains one, that plug-in is left out and named, so the song still opens.</li>
<li><b>Exports work on a private copy</b>, so you can keep playing while a file is made, and you can cancel.</li>
<li><b>Deleting recordings goes to the Recycle Bin</b>, only after you have seen the list and said yes.</li></ul>

<h2>5. Key commands</h2>
<p class="mut">These are the keys the program has now. You can change any of them in the keyboard shortcuts panel. Keys marked "not built yet" are on the list and are not shown here.</p>`;
  for (const ctx of ctxOrder) {
    const rows = ready.filter((b) => b.context === ctx); if (!rows.length) continue;
    const desc = keymap.contexts && keymap.contexts[ctx] ? (typeof keymap.contexts[ctx] === "string" ? keymap.contexts[ctx] : keymap.contexts[ctx].description || "") : "";
    h += `<h3>${esc(ctx === "global" ? "Everywhere" : ctx)}</h3>${desc ? `<p class="mut">${esc(desc)}</p>` : ""}<table><tr><th>Key</th><th>What it does</th></tr>`;
    for (const b of rows) h += `<tr><td>${b.keys.map((k) => `<kbd>${esc(k)}</kbd>`).join(" or ")}</td><td>${esc(label(b.command))}</td></tr>`;
    h += "</table>";
  }
  h += `<h2>6. Every command, for your own AI</h2>
<p>Everything the program can do is a named command with a plain sentence. Your own AI (Claude, ChatGPT, or an agent here) can read this list, call the commands, and teach you the program. A command marked <b>ask</b> is one the program wants your OK for first; <b>yes</b> means an assistant may use it freely; <b>no</b> means it is not offered to assistants.</p>
<table><tr><th>Command</th><th>Kind</th><th>Assistant</th><th>What it does</th></tr>`;
  for (const t of [...tools.tools].sort((a, b) => a.name.localeCompare(b.name))) h += `<tr><td><code>${esc(t.name)}</code></td><td>${esc(t.class)}</td><td>${esc(t.agent)}</td><td>${esc(t.summary)}</td></tr>`;
  h += `</table>
<h2>7. What is still rough</h2>
<ul><li>It is almost complete and in testing. Please report anything that closes it or surprises you.</li>
<li>Cut works on the last track only; pasted and duplicated channels go to the end. Join Clips joins a clip with the next one. Bounce plays the clip's own track only. Bounce Audio to MIDI finds when, not which pitch.</li>
<li>The installer is not code-signed yet, so Windows may show a blue "protected your PC" screen the first time: choose More info, then Run anyway, after checking the SHA-256 on the download page matches.</li>
<li>Windows 10 and 11, 64-bit. A Mac version has not been tested.</li></ul>
<p class="mut">Archie DAW is made by Ask Archie (askarchie.net). Your songs, recordings and exports are ordinary files on your own computer.</p>
</main></body></html>`;
  return h;
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  const out = process.argv[2] ? path.resolve(process.argv[2]) : path.join(root, "desktop-host", "installer", "manual-preview.html");
  fs.mkdirSync(path.dirname(out), { recursive: true }); fs.writeFileSync(out, buildManual());
  console.log("wrote", out, (fs.statSync(out).size / 1024).toFixed(0) + " KB");
}
