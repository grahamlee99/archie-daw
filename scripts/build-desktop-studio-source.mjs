// Makes the SOURCE PACKAGE of Archie DAW (GPLv3): one zip holding exactly what is listed here, nothing else from the repository. Nothing is uploaded.
//   node scripts/build-desktop-studio-source.mjs      -> installer/out/ArchieStudio-Source-<version>.zip
import fs from "node:fs";
import path from "node:path";
import crypto from "node:crypto";
import { spawnSync } from "node:child_process";
import { fileURLToPath } from "node:url";
import { readVersion } from "./build-desktop-manual.mjs";

const root = path.join(path.dirname(fileURLToPath(import.meta.url)), "..");
// an ALLOW list: a new file is not published until it is named (or sits in one of these folders)
export const SOURCE_DIRS = ["desktop-host/Source", "desktop-host/ui", "desktop-host/tools", "desktop-host/assets", "desktop-host/tests", "desktop-host/third_party", "desktop-host/starter-songs"];
export const SOURCE_FILES = ["desktop-host/CMakeLists.txt", "desktop-host/LICENSE", "desktop-host/NOTICE.md", "desktop-host/BUILDING.md", "desktop-host/release-notes.json", "desktop-host/installer/archie-studio.iss", "desktop-host/installer/consent.txt",
  "scripts/build-desktop-studio-installer.mjs", "scripts/build-desktop-manual.mjs", "scripts/build-desktop-studio-source.mjs"];      // what builds the program. The scripts that upload it to our own site are not part of the program and stay out
export const SKIP = /\.(txt\.bak|bak|old|log|tmp)$|\.working-|(^|\/)stage\//i;

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  const version = readVersion(), name = "ArchieStudio-Source-" + version, out = path.join(root, "installer", "out"), stage = path.join(out, name);
  fs.rmSync(stage, { recursive: true, force: true });
  const add = (rel) => { const to = path.join(stage, rel); fs.mkdirSync(path.dirname(to), { recursive: true }); fs.copyFileSync(path.join(root, rel), to); };
  const walk = (rel) => { for (const e of fs.readdirSync(path.join(root, rel), { withFileTypes: true })) { const r = rel + "/" + e.name; if (SKIP.test(r + (e.isDirectory() ? "/" : ""))) continue; if (e.isDirectory()) walk(r); else add(r); } };
  SOURCE_DIRS.forEach(walk); SOURCE_FILES.forEach(add);
  // the front page and licence of the source repository sit at its top
  fs.copyFileSync(path.join(root, "desktop-host", "REPO_README.md"), path.join(stage, "README.md")); fs.copyFileSync(path.join(root, "desktop-host", "LICENSE"), path.join(stage, "LICENSE"));
  const zip = path.join(out, name + ".zip"); fs.rmSync(zip, { force: true });
  // Windows ships a tar that writes real zip files; the one that comes with Git does not (it would write a tar file called .zip)
  const tar = process.platform === "win32" ? path.join(process.env.SystemRoot || "C:\\Windows", "System32", "tar.exe") : "tar";
  const z = spawnSync(tar, ["-a", "-c", "-f", name + ".zip", name], { cwd: out, encoding: "utf8" });
  if (z.status !== 0) throw new Error("could not make the zip: " + (z.stderr || z.error));
  const buf = fs.readFileSync(zip);
  if (buf.subarray(0, 2).toString() !== "PK") throw new Error("the file made is not a real zip");
  console.log("Built " + path.relative(root, zip) + " (" + (buf.length / 1048576).toFixed(2) + " MB)\nSHA-256 " + crypto.createHash("sha256").update(buf).digest("hex") + "\nNothing was uploaded or published.");
}
