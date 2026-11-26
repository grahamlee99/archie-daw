# Building Archie DAW from source (Windows)

You need Visual Studio 2022 Build Tools (C++), CMake 3.22 or later and Node.js 20 or later.

1. Make a dependencies folder holding:
   - `JUCE` : https://github.com/juce-framework/JUCE at tag `8.0.4`
   - `WebView2` : the NuGet package `Microsoft.Web.WebView2`, unpacked so that this folder contains `Microsoft.Web.WebView2.<version>`
2. Configure and build (from the folder that holds `desktop-host` and `scripts`):

       cmake -S desktop-host -B build -DARCHIE_LB=<your dependencies folder>
       cmake --build build --config Release --target ArchieStudio

3. Tests: build the targets `render_test`, `transport_test` and `analyser_test` and run them (set `ASKARCHIE_HOST_TEST=1` first, so a test never touches your own settings).
4. The Windows setup file is made by `node scripts/build-desktop-studio-installer.mjs` (it needs Inno Setup, and expects the optional bundled instruments, which are separate works and not part of this source; edit that list to build without them).

Licence: GNU GPL version 3, see `desktop-host/LICENSE` and `desktop-host/NOTICE.md`.
