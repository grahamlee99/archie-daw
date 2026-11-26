// Builds the Windows installer for Archie Studio (desktop).
// Run: node scripts/build-desktop-studio-installer.mjs      (needs Inno Setup 6 and a Release build of desktop-host, see desktop-host/CMakeLists.txt)
// What it does: stages ArchieStudio.exe (the program carries its own console page inside it), compiles desktop-host/installer/archie-studio.iss with Inno Setup into
// installer/out/ArchieStudio-Setup-<version>.exe, and writes archie-studio.manifest.json (size, SHA-256, unsigned) beside it.
// It uploads NOTHING and writes nothing under public/. Publishing is the separate, explicit step scripts/publish-desktop-studio.mjs.
//
// FILTERING/MONITORING MECHANISM: checkBuiltExe().
// What it screens: the exe must exist, be a 64-bit Windows program, and must not depend on the Visual C++ runtime DLLs (the build links the runtime statically, so a person's computer needs nothing extra).
// What it deliberately does NOT judge: whether the program works (the host's own tests do) or whether the setup file is safe on a given machine (it is unsigned, and says so).
// How it can be adjusted: change FORBIDDEN_DLLS below; tests/desktopStudioSite.test.ts pins the rule.
import fs from "node:fs";
import path from "node:path";
import crypto from "node:crypto";
import { spawnSync } from "node:child_process";
import { buildManual } from "./build-desktop-manual.mjs";
import { fileURLToPath } from "node:url";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
export const PUBLISHER = "Ask Archie (askarchie.net)";
export const FORBIDDEN_DLLS = ["vcruntime140.dll", "vcruntime140_1.dll", "msvcp140.dll", "msvcp140_1.dll", "concrt140.dll", "ucrtbase.dll"];
const sha256 = (buf) => crypto.createHash("sha256").update(buf).digest("hex");

export function readVersion() {
  const cmake = fs.readFileSync(path.join(root, "desktop-host", "CMakeLists.txt"), "utf8");
  const m = cmake.match(/project\(ArchieStudioHost VERSION (\d+\.\d+\.\d+)/);
  if (!m) throw new Error("desktop-host/CMakeLists.txt has no project VERSION");
  return m[1];
}

// the DLL names a PE file imports, read straight from its bytes (no tools needed)
export function importedDlls(buf) {
  const peOff = buf.readUInt32LE(0x3c);
  if (buf.toString("latin1", peOff, peOff + 4) !== "PE\0\0") throw new Error("not a PE file");
  const numSections = buf.readUInt16LE(peOff + 6), optSize = buf.readUInt16LE(peOff + 20), opt = peOff + 24;
  const magic = buf.readUInt16LE(opt);
  if (magic !== 0x20b) throw new Error("not a 64-bit Windows program");
  const importRva = buf.readUInt32LE(opt + 112 + 8);
  const sections = [];
  for (let i = 0; i < numSections; i++) { const o = opt + optSize + i * 40; sections.push({ va: buf.readUInt32LE(o + 12), size: buf.readUInt32LE(o + 8), raw: buf.readUInt32LE(o + 20) }); }
  const toFile = (rva) => { for (const s of sections) if (rva >= s.va && rva < s.va + Math.max(s.size, 1)) return rva - s.va + s.raw; return -1; };
  const names = [];
  let d = toFile(importRva);
  if (importRva === 0 || d < 0) return names;
  for (;; d += 20) {
    const nameRva = buf.readUInt32LE(d + 12);
    if (nameRva === 0) break;
    const f = toFile(nameRva); if (f < 0) break;
    let e = f; while (buf[e] !== 0) e++;
    names.push(buf.toString("latin1", f, e).toLowerCase());
  }
  return names;
}

export function checkBuiltExe(buf) {
  const problems = [];
  let dlls = [];
  try { dlls = importedDlls(buf); } catch (e) { problems.push(String(e.message)); }
  for (const bad of FORBIDDEN_DLLS) if (dlls.includes(bad)) problems.push(`the program imports ${bad}: the build must link the Visual C++ runtime statically`);
  return { problems, dlls };
}

export function buildManifest({ version, file, bytes, sha, builtAt }) {
  return { publisher: PUBLISHER, product: "Archie DAW (desktop)", version, file, sizeBytes: bytes, sha256: sha, signed: false, platform: "windows-x64", install: "per-user, no administrator rights, no start at login", builtAt };
}

function findIscc() {
  const candidates = [process.env.ISCC_PATH, path.join(process.env.LOCALAPPDATA || "", "Programs", "Inno Setup 6", "ISCC.exe"), "C:\\Program Files (x86)\\Inno Setup 6\\ISCC.exe", "C:\\Program Files\\Inno Setup 6\\ISCC.exe"].filter(Boolean);
  const found = candidates.find((p) => fs.existsSync(p));
  if (!found) throw new Error("Inno Setup 6 (ISCC.exe) was not found. Install it, or set ISCC_PATH.");
  return found;
}

export function buildInstaller(opts = {}) {
  const version = readVersion();
  const exe = opts.exe || process.env.ARCHIE_STUDIO_EXE || "D:\\AskArchieHostBuild\\ArchieStudio_artefacts\\Release\\Archie Studio.exe";
  if (!fs.existsSync(exe)) throw new Error("The built program was not found at " + exe + ". Build desktop-host in Release first.");
  const exeBuf = fs.readFileSync(exe);
  const { problems } = checkBuiltExe(exeBuf);
  if (problems.length) throw new Error("The built program fails its checks:\n- " + problems.join("\n- "));
  const stage = opts.stage || path.join(root, "desktop-host", "installer", "stage");
  const out = opts.out || path.join(root, "installer", "out");
  fs.rmSync(stage, { recursive: true, force: true }); fs.mkdirSync(stage, { recursive: true }); fs.mkdirSync(out, { recursive: true });
  fs.copyFileSync(exe, path.join(stage, "ArchieStudio.exe"));
  // the devices that come with the program: Archie Drums (a VST3) with its small 808 sound pack beside it, installed in a Devices folder the program scans by itself. The build must exist; the installer is never made without it.
  const drumsBundle = process.env.ARCHIE_DRUMS_VST3 || "D:\\AskArchieDrumsBuild\\ArchieDrums_artefacts\\Release\\VST3\\Archie Drums.vst3";
  if (!fs.existsSync(path.join(drumsBundle, "Contents", "x86_64-win", "Archie Drums.vst3"))) throw new Error("Archie Drums was not found at " + drumsBundle + ". Build drum-machine in Release first.");
  const devBundle = path.join(stage, "Devices", "Archie Drums.vst3");
  fs.cpSync(drumsBundle, devBundle, { recursive: true });
  fs.cpSync(path.join(root, "drum-machine", "packs", "fischer-808"), path.join(devBundle, "Contents", "x86_64-win", "packs", "fischer-808"), { recursive: true });
  // the instruments and effects that come with the program, in a Plug-ins folder the program scans by itself: every finished Windows build kept in D:\AskArchieBundled\Plug-ins (copy a new build there when one is made, or the old one ships). The ones listed here must be there: the installer is never made without them.
  const bundled = process.env.ARCHIE_BUNDLED_PLUGINS || "D:\\AskArchieBundled\\Plug-ins";
  for (const need of ["Archie Acid 303.vst3", "Archie Sampler V2.vst3", "Archies Alien Delay.vst3", "Warm Echo.vst3"]) if (!fs.existsSync(path.join(bundled, need))) throw new Error(need + " was not found in " + bundled + ". Install the finished build from the Vault and copy it there.");
  fs.mkdirSync(path.join(stage, "Plug-ins"), { recursive: true }); for (const e of fs.readdirSync(bundled, { withFileTypes: true })) if (e.isDirectory() && e.name.toLowerCase().endsWith(".vst3")) fs.cpSync(path.join(bundled, e.name), path.join(stage, "Plug-ins", e.name), { recursive: true });
  // starter songs: they appear under File > New from Template
  const starters = path.join(root, "desktop-host", "starter-songs"); if (fs.existsSync(starters)) { fs.mkdirSync(path.join(stage, "Starter songs"), { recursive: true }); for (const f of fs.readdirSync(starters)) if (f.endsWith(".archiestudio")) fs.copyFileSync(path.join(starters, f), path.join(stage, "Starter songs", f)); }
  // optional (a bundle test, or a bundled release): a folder of ready .vst3 bundles that travels with the program into a Plug-ins folder the program scans by itself
  if (opts.bundle) { const pl = path.join(stage, "Plug-ins"); fs.mkdirSync(pl, { recursive: true }); for (const e of fs.readdirSync(opts.bundle, { withFileTypes: true })) if (e.isDirectory() && e.name.toLowerCase().endsWith(".vst3")) fs.cpSync(path.join(opts.bundle, e.name), path.join(pl, e.name), { recursive: true }); }
  // the manual: one self-contained page, generated from the keymap and the command list, opened from the program's Help menu
  fs.mkdirSync(path.join(stage, "manual"), { recursive: true }); fs.writeFileSync(path.join(stage, "manual", "Archie-DAW-Manual.html"), buildManual());
  // the licence notices of the third-party code the program contains (MIT: the notice must travel with it)
  const licDir = path.join(stage, "licenses"); fs.mkdirSync(licDir, { recursive: true });
  for (const [from, to] of [["LICENSE", "Archie-DAW-LICENSE-GPLv3.txt"], ["NOTICE.md", "Archie-DAW-NOTICE.md"], ["third_party/asio/LICENSE.txt", "Steinberg-ASIO-SDK-LICENSE.txt"]]) fs.copyFileSync(path.join(root, "desktop-host", ...from.split("/")), path.join(licDir, to));
  for (const [from, to] of [["signalsmith-stretch", "Signalsmith-Stretch-LICENSE.txt"], ["signalsmith-linear", "Signalsmith-Linear-LICENSE.txt"]]) fs.copyFileSync(path.join(root, "desktop-host", "third_party", from, "LICENSE.txt"), path.join(licDir, to));
  const iscc = spawnSync(findIscc(), ["/Q", `/DAppVersion=${version}`, `/DSourceDir=${stage}`, `/DOutDir=${out}`, `/DConsentFile=${path.join(root, "desktop-host", "installer", "consent.txt")}`, path.join(root, "desktop-host", "installer", "archie-studio.iss")], { encoding: "utf8" });
  if (iscc.status !== 0) throw new Error("Inno Setup failed:\n" + (iscc.stdout || "") + (iscc.stderr || ""));
  const file = `ArchieStudio-Setup-${version}.exe`;
  const buf = fs.readFileSync(path.join(out, file));
  const manifest = buildManifest({ version, file, bytes: buf.length, sha: sha256(buf), builtAt: new Date().toISOString() });
  fs.writeFileSync(path.join(out, "archie-studio.manifest.json"), JSON.stringify(manifest, null, 2) + "\n");
  fs.writeFileSync(path.join(out, file + ".sha256"), `${manifest.sha256}  ${file}\n`);
  return { manifest, out, file };
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  try {
    const { manifest, out } = buildInstaller();
    console.log(`Built ${manifest.file} (${(manifest.sizeBytes / 1048576).toFixed(1)} MB) in ${out}`);
    console.log(`SHA-256 ${manifest.sha256}`);
    console.log("Nothing was uploaded or published. Publishing is scripts/publish-desktop-studio.mjs, a separate explicit step.");
  } catch (e) { console.error("BUILD FAILED: " + e.message); process.exit(1); }
}
