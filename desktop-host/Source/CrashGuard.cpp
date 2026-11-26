// Archie Studio host -- the crash note. Kept apart from the JUCE files because it needs the Windows headers.
// FILTERING/MONITORING MECHANISM: when the program is about to be closed by a fault nobody handled, this notes WHERE: the file the fault was in, then the files of the calls that led there (newest first),
// one path per line, in crash-module.bin. The next start reads it (HostEngine::readCrashModule) and flags the nearest plug-in file (.vst3) in that list, so a fault that lands in Windows or in a shared
// runtime while a plug-in is on the stack is still pinned on the plug-in. It judges only "this file was on the stack of the fault", writes only paths, and lets Windows carry on (the normal crash report is unchanged).
// It can be adjusted in HostEngine::readCrashModule, which decides what is flagged (only .vst3 files, never the program or Windows).
#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <wchar.h>

static wchar_t gCrashFile[1100] = { 0 };
static DWORD gMainThread = 0;

static void appendModule (wchar_t* out, size_t cap, DWORD64 addr, wchar_t* last)
{
    HMODULE mod = nullptr; wchar_t path[1024] = { 0 };
    if (GetModuleHandleExW (GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT, (LPCWSTR) addr, &mod) && mod != nullptr) GetModuleFileNameW (mod, path, 1023);
    if (path[0] == 0 || wcscmp (path, last) == 0) return;                 // nothing found, or the same file as the line before
    wcsncpy_s (last, 1024, path, _TRUNCATE);
    if (wcslen (out) + wcslen (path) + 2 < cap) { wcscat_s (out, cap, path); wcscat_s (out, cap, L"\n"); }
}

static LONG WINAPI archieCrashFilter (EXCEPTION_POINTERS* ep)
{
    if (gCrashFile[0] != 0 && ep != nullptr && ep->ExceptionRecord != nullptr)
    {
        static wchar_t text[8192]; text[0] = 0; wchar_t last[1024] = { 0 };
        wcscat_s (text, 8192, GetCurrentThreadId() == gMainThread ? L"# on the main (window) thread\n" : L"# on another thread (audio or a helper)\n");
        appendModule (text, 8192, (DWORD64) ep->ExceptionRecord->ExceptionAddress, last);
#if defined(_M_X64)
        if (ep->ContextRecord != nullptr)
        {
            CONTEXT ctx = *ep->ContextRecord;
            for (int i = 0; i < 48 && ctx.Rip != 0; ++i)                    // walk up the stack, using the program's own unwind tables
            {
                appendModule (text, 8192, ctx.Rip, last);
                DWORD64 base = 0; PRUNTIME_FUNCTION fn = RtlLookupFunctionEntry (ctx.Rip, &base, nullptr);
                if (fn == nullptr) { ctx.Rip = *(DWORD64*) ctx.Rsp; ctx.Rsp += 8; }          // a leaf with no table: the return address is on top of the stack
                else { PVOID handlerData = nullptr; DWORD64 frame = 0; RtlVirtualUnwind (UNW_FLAG_NHANDLER, base, ctx.Rip, fn, &ctx, &handlerData, &frame, nullptr); }
            }
        }
#endif
        HANDLE h = CreateFileW (gCrashFile, GENERIC_WRITE, 0, nullptr, CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr);
        if (h != INVALID_HANDLE_VALUE) { DWORD n = 0; WriteFile (h, text, (DWORD) (wcslen (text) * sizeof (wchar_t)), &n, nullptr); CloseHandle (h); }
    }
    return EXCEPTION_CONTINUE_SEARCH;
}

void installArchieCrashFilter (const wchar_t* file)
{
    wcsncpy_s (gCrashFile, 1100, file, _TRUNCATE);
    gMainThread = GetCurrentThreadId();                                    // installed from the main thread
    SetUnhandledExceptionFilter (archieCrashFilter);
}
#endif
