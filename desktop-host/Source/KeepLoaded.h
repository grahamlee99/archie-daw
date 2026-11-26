#pragma once
#include <juce_core/juce_core.h>
#if JUCE_WINDOWS
 #ifndef NOMINMAX
  #define NOMINMAX
 #endif
 #include <windows.h>
 #include <psapi.h>
#endif

namespace archie
{
// ONE WINDOW TYPE SHARED BY EVERY ARCHIE PLUG-IN. Builds from the Vault each register a window type called "ArchieOwnedWebView2ChildWindow" for the web page in their window, and register it for the whole PROGRAM. Only the
// first registration counts, so every later plug-in's window was handled by the FIRST plug-in's code, which read and wrote the later one's data as if it were its own: a fault when that window was closed (Sampler V2 first,
// then the 303 or Warm Echo), or a fault in unloaded code once the first plug-in was removed. The program claims the name before any plug-in loads, with plain window handling that touches nobody's data.
// (What this costs: a plug-in window that is destroyed lets go of its web page without closing it first, so a little memory is held until the program ends. The lasting fix is in the plug-in builder: register the type for
// the plug-in's own file. Once builds do that, they no longer meet this registration at all.)
inline bool claimSharedWebViewWindowClass()
{
   #if JUCE_WINDOWS
    WNDCLASSEXW wc {}; wc.cbSize = sizeof (WNDCLASSEXW); wc.lpfnWndProc = DefWindowProcW; wc.hInstance = GetModuleHandleW (nullptr); wc.lpszClassName = L"ArchieOwnedWebView2ChildWindow";
    return RegisterClassExW (&wc) != 0 || GetLastError() == ERROR_CLASS_ALREADY_EXISTS;
   #else
    return true;
   #endif
}

// A PLUG-IN'S CODE STAYS LOADED UNTIL THE PROGRAM ENDS. Removing the last copy of a plug-in (or closing the program) used to unload its file while something it had left behind (a hidden window, a timer, a thread) could
// still run: Windows then closed the program with a fault in "<name>.vst3_unloaded" (seen with Archie Sampler V2, Tuned Drum, Layer Drum and Acid 303). Pinning the file means that late call still lands in real code.
// Call it after a plug-in has been loaded. It only pins files ending in .vst3 that are already in this process; calling it again is harmless. It returns how many such files are loaded.
// (macOS: bundles are handled differently and this fault has not been seen there, so nothing is done.)
inline int keepPluginModulesLoaded()
{
   #if JUCE_WINDOWS
    HMODULE mods[2048]; DWORD bytes = 0; int found = 0;
    if (! K32EnumProcessModules (GetCurrentProcess(), mods, (DWORD) sizeof (mods), &bytes)) return 0;
    const int n = (int) juce::jmin ((size_t) (bytes / sizeof (HMODULE)), (size_t) 2048);
    for (int i = 0; i < n; ++i)
    {
        wchar_t path[2048]; const DWORD len = GetModuleFileNameW (mods[i], path, 2048);
        if (len < 5 || len >= 2048 || _wcsicmp (path + len - 5, L".vst3") != 0) continue;
        HMODULE pinned = nullptr;
        if (GetModuleHandleExW (GET_MODULE_HANDLE_EX_FLAG_PIN | GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS, reinterpret_cast<LPCWSTR> (mods[i]), &pinned)) ++found;
    }
    return found;
   #else
    return 0;
   #endif
}
}
