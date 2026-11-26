// Archie Studio host (desktop) -- the bridge between the web pages in the window and the native engine.
// One native function, archieHost(command, params), and one event stream, "archie", are given to every page the window shows:
//   * the host's own console page (compiled into the program)
//   * pages from askarchie.net (the online Archie's Studio and the rest of the site)
// No other site is ever shown in this window: any other address opens in the person's own browser instead.
#pragma once
#include <juce_gui_extra/juce_gui_extra.h>
#include "Engine.h"
#include <thread>
#include "Export.h"
#include "BinaryData.h"

namespace archie
{

inline const char* kAppName = "Archie DAW (desktop)";
#ifndef ARCHIE_APP_VERSION
#define ARCHIE_APP_VERSION "0.0.0"
#endif
inline const char* kAppVersion = ARCHIE_APP_VERSION;      // the version in CMakeLists.txt, so the program, the installer and the site always say the same

class HostBrowser : public juce::WebBrowserComponent
{
public:
    HostBrowser (const Options& o, juce::StringArray allowedHosts) : juce::WebBrowserComponent (o), hosts (std::move (allowedHosts)) {}

    bool pageAboutToLoad (const juce::String& url) override
    {
        if (url.isEmpty()) return true;
        if (url.startsWith (getResourceProviderRoot()) || url.startsWith ("about:") || url.startsWith ("data:")) return true;
        const juce::URL u (url);
        const auto host = u.getDomain().toLowerCase();
        for (auto& h : hosts) if (host == h) return true;
        u.launchInDefaultBrowser();        // anything that is not the Ask Archie site opens in the person's own browser, never in this window
        return false;
    }
    void pageFinishedLoading (const juce::String& url) override { if (url != "about:blank") { loaded = true; loadedUrl = url; } }
    bool pageLoadHadNetworkError (const juce::String&) override { loaded = false; return true; }

    // Navigation requested before the window has a peer is dropped by the web view, so the page is asked for again until it reports it has loaded.
    void wantUrl (const juce::String& url) { wanted = url; loaded = false; tries = 0; goToURL (url); }
    bool isLoaded() const { return loaded; }
    void retryIfNeeded()
    {
        if (wanted.isEmpty() || loaded || ! isShowing() || tries >= 40) return;
        ++tries; goToURL (wanted);
    }
private:
    juce::StringArray hosts;
    juce::String wanted, loadedUrl; bool loaded = false; int tries = 0;
};

class Bridge : private juce::Timer
{
public:
    explicit Bridge (HostEngine& e) : engine (e)
    {
        const auto f = engine.settingsDir.getChildFile ("skin.json");
        if (f.existsAsFile()) currentSkin = cleanSkin (juce::JSON::parse (f.loadFileAsString()));
        // ---- RECOVERY: a song is kept every minute; if the program is closed by a crash, the next start offers it back ----
        // FILTERING/MONITORING MECHANISM: "the last session ended without a clean exit" is judged ONLY by a flag file (running.flag) that a normal quit removes. It does not judge why it ended, nor the song's quality.
        // The kept song (recovery.archiestudio) is held, not overwritten, until the person has answered the offer, and the answer is always theirs (Recover or not); the old one is kept as recovery-previous.archiestudio.
        {
            const auto sfx = juce::SystemStats::getEnvironmentVariable ("ASKARCHIE_HOST_TEST", {}) == "1" ? "-test" : juce::String();
            runningFlag = engine.settingsDir.getChildFile ("running" + sfx + ".flag"); recoveryFile = engine.settingsDir.getChildFile ("recovery" + sfx + ".archiestudio");
            recoveryPending = runningFlag.existsAsFile() && recoveryFile.existsAsFile(); runningFlag.replaceWithText (juce::String (juce::Time::currentTimeMillis()));
        }
        startTimerHz (30);
    }
    juce::File runningFlag, recoveryFile; bool recoveryPending = false; juce::int64 lastKeep = juce::Time::currentTimeMillis();
    void keepSong()
    {
        if (recoveryPending || exportJob != nullptr) return;                   // not while an answer is waiting, nor during an export
        const auto now = juce::Time::currentTimeMillis(); if (now - lastKeep < 60000) return; lastKeep = now;
        recoveryFile.replaceWithText (juce::JSON::toString (engine.saveProject()));
    }

    juce::var getSkin() const { return currentSkin; }
    // for the window's own (native) controls: the same commands the pages send, and the engine they read
    juce::var invoke (const juce::String& cmd, const juce::var& p = {}) { return handle (cmd, p); }
    HostEngine& getEngine() { return engine; }
    std::function<void (const juce::var&)> onSkin;     // the window re-colours its own tab bar to match
    ~Bridge() override { stopTimer(); runningFlag.deleteFile(); }     // a normal quit: the flag goes, so the next start knows it was clean

    void addBrowser (HostBrowser* b) { browsers.addIfNotAlreadyThere (b); }
    void removeBrowser (HostBrowser* b) { browsers.removeFirstMatchingValue (b); }

    juce::WebBrowserComponent::Options makeOptions()
    {
        auto data = juce::File::getSpecialLocation (juce::File::userApplicationDataDirectory).getChildFile ("AskArchie").getChildFile ("Studio").getChildFile (juce::SystemStats::getEnvironmentVariable ("ASKARCHIE_HOST_TEST", {}) == "1" ? "webview-test" : "webview");   // a test copy beside a real window must not share its browser process
        data.createDirectory();
        using O = juce::WebBrowserComponent::Options;
        return O {}
            .withBackend (O::Backend::webview2)
            .withWinWebView2Options (O::WinWebView2 {}.withUserDataFolder (data).withStatusBarDisabled())
            .withNativeIntegrationEnabled()
            .withKeepPageLoadedWhenBrowserIsHidden()
            .withNativeFunction ("archieHost", [this] (const juce::Array<juce::var>& args, juce::WebBrowserComponent::NativeFunctionCompletion done)
            {
                done (handle (args.size() > 0 ? args[0].toString() : juce::String(), args.size() > 1 ? args[1] : juce::var()));
            })
            .withResourceProvider ([] (const juce::String& fullPath) -> std::optional<juce::WebBrowserComponent::Resource>
            {
                const auto path = fullPath.upToFirstOccurrenceOf ("?", false, false);       // the floating panel asks for the page with a query (?panel=1&kind=...)
                if (path == "/palette.json")
                {
                    juce::WebBrowserComponent::Resource r; r.mimeType = "application/json"; r.data.resize ((size_t) BinaryData::studiopalette_jsonSize);
                    std::memcpy (r.data.data(), BinaryData::studiopalette_json, (size_t) BinaryData::studiopalette_jsonSize);
                    return r;
                }
                if (path == "/keymap.json")                  // the default shortcuts: one data file, the same one the tests check
                {
                    juce::WebBrowserComponent::Resource r; r.mimeType = "application/json"; r.data.resize ((size_t) BinaryData::studiokeymap_jsonSize);
                    std::memcpy (r.data.data(), BinaryData::studiokeymap_json, (size_t) BinaryData::studiokeymap_jsonSize);
                    return r;
                }
                if (path == "/" || path == "/index.html" || path == "/host.html")
                {
                    juce::WebBrowserComponent::Resource r;
                    r.mimeType = "text/html";
                    r.data.resize ((size_t) BinaryData::host_htmlSize);
                    std::memcpy (r.data.data(), BinaryData::host_html, (size_t) BinaryData::host_htmlSize);
                    return r;
                }
                return std::nullopt;
            });
    }

    std::function<void (const juce::String&)> onOpenPanel;    // the floating Track Panel window: the page's query string says what it shows
    std::function<void()> onClosePanel;
    std::function<void (int)> onShowTab;   // 0 console, 1 studio
    std::function<void (const juce::String&)> onNavigateStudio;

private:
    // A skin is { name, vars: { "--css-variable": "colour or size" } } and nothing else. Anything that is not plainly that is refused, so a page or a file can only ever change how the program looks.
    static juce::var cleanSkin (const juce::var& in)
    {
        auto* src = in.getDynamicObject();
        if (src == nullptr) return {};
        auto vars = src->getProperty ("vars");
        auto* vo = vars.getDynamicObject();
        if (vo == nullptr || vo->getProperties().size() == 0 || vo->getProperties().size() > 80) return {};
        auto* outVars = new juce::DynamicObject();
        for (const auto& p : vo->getProperties())
        {
            const auto key = p.name.toString(); const auto val = p.value.toString();
            if (! key.startsWith ("--") || key.length() > 30 || ! key.containsOnly ("-abcdefghijklmnopqrstuvwxyz0123456789")) return {};
            if (val.length() > 140 || val.containsAnyOf (";{}\"'\\<>@") || val.containsIgnoreCase ("url(") || val.containsIgnoreCase ("expression") || val.containsIgnoreCase ("javascript")) return {};
            outVars->setProperty (p.name, val);
        }
        auto* o = new juce::DynamicObject();
        o->setProperty ("name", src->getProperty ("name").toString().substring (0, 40).trim().isEmpty() ? juce::String ("My skin") : src->getProperty ("name").toString().substring (0, 40));
        o->setProperty ("vars", juce::var (outVars));
        return juce::var (o);
    }

    void applySkin (const juce::var& skin, const juce::String& src)
    {
        currentSkin = skin;
        engine.settingsDir.getChildFile ("skin.json").replaceWithText (juce::JSON::toString (skin));
        if (onSkin) onSkin (skin);
        auto* o = new juce::DynamicObject();
        o->setProperty ("name", skin.getProperty ("name", {})); o->setProperty ("vars", skin.getProperty ("vars", {})); o->setProperty ("src", src);
        emit ("skin", juce::var (o));
    }

    static juce::var ok (juce::var extra = {}) { auto* o = new juce::DynamicObject(); o->setProperty ("ok", true); if (auto* e = extra.getDynamicObject()) for (auto& p : e->getProperties()) o->setProperty (p.name, p.value); return juce::var (o); }
    static juce::var fail (const juce::String& why) { auto* o = new juce::DynamicObject(); o->setProperty ("ok", false); o->setProperty ("error", why); return juce::var (o); }
    static double num (const juce::var& p, const char* k, double d = 0.0) { return p.hasProperty (k) ? (double) p.getProperty (k, d) : d; }
    static int inum (const juce::var& p, const char* k, int d = 0) { return p.hasProperty (k) ? (int) p.getProperty (k, d) : d; }

    juce::var handle (const juce::String& cmd, const juce::var& p)
    {
        auto& tr = engine.transport;
        if (cmd == "hello")
        {
            engine.scanBundledDevicesOnce();
            auto* o = new juce::DynamicObject();
            o->setProperty ("app", kAppName); o->setProperty ("version", kAppVersion); o->setProperty ("tracks", engine.trackCount.load()); o->setProperty ("maxTracks", kTracks);
            o->setProperty ("capabilities", juce::StringArray { "transport", "mixer", "plugins", "midi-clock", "mtc", "mmc", "clips", "notes", "project" }.joinIntoString (","));
            return ok (juce::var (o));
        }
        if (cmd == "getState") { auto* o = new juce::DynamicObject(); o->setProperty ("state", engine.snapshot()); o->setProperty ("transport", engine.transportVar()); return ok (juce::var (o)); }

        // ---- transport ----
        if (cmd == "play")      { tr.play(); return ok(); }
        if (cmd == "stop")      { tr.stop(); return ok(); }
        if (cmd == "pause")     { tr.pause(); return ok(); }
        if (cmd == "toggle")    { tr.togglePlay(); return ok(); }
        if (cmd == "record")    { tr.setRecording (p.hasProperty ("on") ? (bool) p.getProperty ("on", false) : ! tr.recording.load()); return ok(); }
        if (cmd == "locate")    { tr.locate (num (p, "beats")); return ok(); }
        if (cmd == "bpm")       { engine.editTempo ("Tempo", "bpm", [&] { tr.setBpm (num (p, "bpm", 120.0)); }); return ok(); }
        if (cmd == "tap") { const double b = engine.tapTempo(); auto* o = new juce::DynamicObject(); o->setProperty ("bpm", b); o->setProperty ("taps", engine.tapper.count()); return ok (juce::var (o)); }
        if (cmd == "tempoAdd")    { engine.editTempo ("Add tempo change", "", [&] { tr.addTempoChange (num (p, "beat"), num (p, "bpm", 120.0)); }); return ok(); }
        if (cmd == "tempoRemove") { engine.editTempo ("Remove tempo change", "", [&] { tr.removeTempoChange (inum (p, "index", 0)); }); return ok(); }
        if (cmd == "timesig")   { engine.editTempo ("Time signature", "", [&] { tr.setTimeSignature (inum (p, "num", 4), inum (p, "den", 4)); }); return ok(); }
        if (cmd == "cue")
        {
            const auto op = p.getProperty ("op", {}).toString(); const int i = inum (p, "index", -1); if (i < 0 || i >= archie::Transport::kCues) return fail ("index must be 0, 1 or 2 (cues 1 to 3)");
            if (op == "trigger") { if (! tr.triggerCue (i)) return fail ("that cue is empty"); return ok(); }
            if (op == "here") { tr.cueHere (i); return ok(); }
            if (op == "set") { if (! tr.setCue (i, num (p, "start", tr.loopStart.load()), num (p, "end", tr.loopEnd.load()))) return fail ("the end must be after the start"); return ok(); }
            if (op == "remove") { if (! tr.removeCue (i)) return fail ("cue 1 cannot be removed"); return ok(); }
            if (op == "learn") { engine.cueLearn.store (engine.cueLearn.load() == i ? -1 : i); return ok(); }
            if (op == "unlearn") { engine.clearCuePad (i); engine.cueLearn.store (-1); return ok(); }
            return fail ("op must be trigger, here, set, remove, learn or unlearn");
        }
        if (cmd == "loopJump")  { const int m = inum (p, "mode", -1); if (m < 0 || m > 4) return fail ("mode must be 0 (at once), 1 (next 16th), 2 (next beat), 3 (end of bar) or 4 (end of loop)"); tr.loopJump.store (m); if (m == 0 && tr.loopPending.load()) { tr.loopPending.store (false); tr.setLoop (true, tr.pendStart.load(), tr.pendEnd.load()); } return ok(); }
        if (cmd == "loop")      { tr.requestLoop ((bool) p.getProperty ("on", false), num (p, "start", tr.loopStart.load()), num (p, "end", tr.loopEnd.load())); return ok(); }
        if (cmd == "ppqn")      { const int v = inum (p, "ppqn", 480); static const int allowed[] = { 24, 48, 96, 192, 384, 480, 960 }; for (int a : allowed) if (a == v) { tr.ppqn.store (v); return ok(); } return fail ("PPQN must be 24, 48, 96, 192, 384, 480 or 960"); }
        if (cmd == "sync")      { tr.syncMode.store (juce::jlimit (0, 2, inum (p, "mode", 0))); return ok(); }
        if (cmd == "fps")       { tr.fps.store (juce::jlimit (0, 3, inum (p, "fps", 3))); return ok(); }
        if (cmd == "tcOffset")  { tr.tcOffsetSeconds.store (juce::jmax (0.0, num (p, "seconds"))); return ok(); }
        if (cmd == "clockOut")  { tr.clockOut.store ((bool) p.getProperty ("on", false)); return ok(); }
        if (cmd == "mtcOut")    { tr.mtcOut.store ((bool) p.getProperty ("on", false)); return ok(); }
        if (cmd == "mmcOut")    { tr.mmcOut.store ((bool) p.getProperty ("on", false)); return ok(); }
        if (cmd == "mmcIn")     { tr.mmcIn.store ((bool) p.getProperty ("on", false)); return ok(); }
        if (cmd == "mmcDevice") { tr.mmcDeviceId.store (juce::jlimit (0, 127, inum (p, "id", 127))); return ok(); }
        if (cmd == "metronome") { engine.metronome.store ((bool) p.getProperty ("on", false)); return ok(); }

        // ---- tracks ----
        if (cmd == "track")
        {
            const int ti = inum (p, "track", -1);
            if (engine.trackAt (ti) == nullptr) return fail ("no such track");
            if (! engine.editChainParam (ti, p.getProperty ("key", {}).toString(), p.getProperty ("value", {}))) return fail ("unknown track setting");
            return ok();
        }
        if (cmd == "addTrack")    { if (! engine.editAddTrack()) return fail ("that is the most tracks there can be (" + juce::String (kTracks) + ")"); return ok(); }
        if (cmd == "removeTrack") { if (! engine.editRemoveTrack (inum (p, "track", -1))) return fail ("that track cannot be removed"); return ok(); }
        if (cmd == "swing") { tr.swing.store (juce::jlimit (0.0, 1.0, num (p, "amount"))); if (p.hasProperty ("grid")) tr.swingGrid.store (num (p, "grid") <= 0.3 ? 0.25 : 0.5); engine.stateDirty.store (true); return ok(); }
        if (cmd == "countIn")     { tr.countInBars.store (juce::jlimit (0, 4, inum (p, "bars", 0))); return ok(); }
        if (cmd == "send")
        {
            auto* t = engine.trackAt (inum (p, "track", -1)); const int s = inum (p, "send", -1);
            if (t == nullptr || s < 0 || s >= kSends) return fail ("no such send");
            engine.editSend (inum (p, "track", -1), s, (float) num (p, "db", -100.0), p.hasProperty ("pre") ? (bool) p.getProperty ("pre", false) : t->sendPre[s].load());
            return ok();
        }
        if (cmd == "sendOn" || cmd == "sendLevel" || cmd == "sendPre")
        {
            auto* t = engine.trackAt (inum (p, "track", -1)); const int s = inum (p, "send", -1);
            if (t == nullptr || s < 0 || s >= kSends) return fail ("no such send");
            if (cmd == "sendOn") engine.editSendOn (inum (p, "track", -1), s, (bool) p.getProperty ("on", true));
            else if (cmd == "sendLevel") engine.editSendLevel (inum (p, "track", -1), s, (float) num (p, "db", 0.0));
            else engine.editSendPre (inum (p, "track", -1), s, (bool) p.getProperty ("pre", false));
            return ok();
        }
        if (cmd == "ret")
        {
            auto* r = engine.chainAt (kReturnBase + inum (p, "ret", -1));
            if (r == nullptr) return fail ("no such return");
            if (! engine.editChainParam (kReturnBase + inum (p, "ret", -1), p.getProperty ("key", {}).toString(), p.getProperty ("value", {}))) return fail ("unknown return setting");
            return ok();
        }
        if (cmd == "eq") { if (! engine.editEq (inum (p, "chain", -1), p.getProperty ("key", {}).toString(), (double) p.getProperty ("value", 0.0))) return fail ("chain must be a track number or 200 for the master, key one of b0on b0hz b0gain b0q ... b3q, hpOn hpHz lpOn lpHz"); return ok(); }
        if (cmd == "getEq") { auto v = engine.eqVar (inum (p, "chain", -1)); if (v.isVoid()) return fail ("no EQ on that channel (tracks and the master have one)"); auto* o = new juce::DynamicObject(); o->setProperty ("eq", v); return ok (juce::var (o)); }
        if (cmd == "eqSpectrum") { auto v = engine.eqSpectrum (inum (p, "chain", -1)); if (v.isVoid()) return fail ("no EQ on that channel"); auto* o = new juce::DynamicObject(); o->setProperty ("spectrum", v); return ok (juce::var (o)); }
        if (cmd == "openEqWindow") { const int ch = inum (p, "chain", 200); if (engine.eqPanelOf (ch) == nullptr) return fail ("no EQ on that channel"); if (onOpenPanel) onOpenPanel ("?eqwin=1&chain=" + juce::String (ch)); return ok(); }
        if (cmd == "openPanelWindow")
        {
            const auto kind = p.getProperty ("kind", "track").toString();
            if (kind != "track" && kind != "return" && kind != "master") return fail ("kind must be track, return or master");
            if (onOpenPanel) onOpenPanel ("?panel=1&kind=" + kind + "&id=" + juce::String (inum (p, "id", 0)));
            return ok();
        }
        if (cmd == "audioOption") { if (! engine.setAudioOptions (p)) return fail ("give latencyAdjustMs (milliseconds) or maxThreads (0 to 7, or -1 for automatic)"); return ok(); }
        if (cmd == "popOutView")
        {
            const auto view = p.getProperty ("view", {}).toString();
            if (view != "strips" && view != "rack" && view != "mixer" && view != "settings") return fail ("view must be strips, rack, mixer or settings");
            auto page = p.getProperty ("page", {}).toString().retainCharacters ("abcdefghijklmnopqrstuvwxyz");      // which Settings page to open on (letters only)
            if (onOpenPanel) onOpenPanel ("?view=" + view + (view == "settings" && page.isNotEmpty() ? "&page=" + page : juce::String()));
            return ok();
        }
        if (cmd == "closePanelWindow") { if (onClosePanel) onClosePanel(); return ok(); }
        if (cmd == "selectTrack") { engine.selectTrack (inum (p, "track", 0)); return ok(); }
        if (cmd == "groove") { if (! engine.editGrooveChan (inum (p, "ch", 0), p.getProperty ("key", {}).toString(), p.getProperty ("value", {}))) return fail ("ch must be 1 to 31, key one of on, patch (an existing patch name or empty), strength, timing, vel, len, rand, slide, shuffle, anchor"); return ok(); }
        if (cmd == "extractGroove") { juce::String err; const int n = engine.extractGroove (inum (p, "contentId", 0), p.getProperty ("name", {}).toString(), num (p, "grid"), err); if (n == 0) return fail (err); auto* o = new juce::DynamicObject(); o->setProperty ("patches", n); o->setProperty ("name", engine.grooveCopy().patches.back().name); return ok (juce::var (o)); }
        if (cmd == "deleteGroove") { if (! engine.deleteGroove (p.getProperty ("name", {}).toString())) return fail ("no such groove of yours (the built-in ones cannot be deleted)"); return ok(); }
        if (cmd == "controlRoom") { if (! engine.editControlRoomParam (p.getProperty ("key", {}).toString(), p.getProperty ("value", {}))) return fail ("key must be crDb, crDim or crPair"); return ok(); }
        if (cmd == "masterSound") { if (! engine.editMasterSound (p.getProperty ("key", {}).toString(), p.getProperty ("value", {}))) return fail ("key must be amOn (true or false), amDrive (0 to 1) or amCeil (-3 to 0 dB)"); return ok(); }
        if (cmd == "masterComp") { if (! engine.editMasterCompParam (p.getProperty ("key", {}).toString(), p.getProperty ("value", {}))) return fail ("key must be mcOn, mcPeak, mcThr, mcRatio, mcAtk, mcRel or mcMakeup"); return ok(); }
        if (cmd == "masterDb") { engine.editMasterDb (num (p, "db")); return ok(); }
        if (cmd == "noteOn")  { engine.noteOn (inum (p, "track"), inum (p, "note", 60), inum (p, "vel", 100), inum (p, "ch", 1)); return ok(); }
        if (cmd == "noteOff") { engine.noteOff (inum (p, "track"), inum (p, "note", 60), inum (p, "ch", 1)); return ok(); }
        if (cmd == "panic") { engine.panicAll(); return ok(); }
        if (cmd == "noteTrace") { auto f = juce::File::getSpecialLocation (juce::File::userDocumentsDirectory).getChildFile ("Archie DAW note trace.txt"); if (! f.replaceWithText (engine.noteTraceText())) return fail ("could not write the note trace"); auto* o = new juce::DynamicObject(); o->setProperty ("ok", true); o->setProperty ("file", f.getFullPathName()); return juce::var (o); }
        if (cmd == "allNotesOff") { engine.allNotesOff (inum (p, "track")); return ok(); }
        if (cmd == "setClip") { engine.editSetClip (inum (p, "track"), p.getProperty ("clip", {})); return ok(); }
        if (cmd == "getClip") { if (auto* t = engine.trackAt (inum (p, "track"))) { auto* o = new juce::DynamicObject(); o->setProperty ("clip", engine.clipToVar (*t)); return ok (juce::var (o)); } return fail ("no such track"); }
        if (cmd == "clearClip") { if (engine.editClearClip (inum (p, "track"))) return ok(); return fail ("no such track"); }
        // ---- clips on the timeline (instances of shared content) ----
        if (cmd == "importAudio")
        {
            juce::String err; const int id = engine.editImportAudio (inum (p, "track"), juce::File (p.getProperty ("path", {}).toString()), p.hasProperty ("start") ? num (p, "start") : engine.transport.posBeats.load(), err);
            if (id == 0) return fail (err.isNotEmpty() ? err : "could not import that");
            auto* o = new juce::DynamicObject(); o->setProperty ("instance", id); return ok (juce::var (o));
        }
        if (cmd == "chopInstance")
        {
            const int n = engine.editChopInstance (inum (p, "instance"), p.getProperty ("mode", "grid").toString(), p.getProperty ("mode", "grid").toString() == "hits" ? (p.hasProperty ("sensitivity") ? num (p, "sensitivity") : 0.5) : num (p, "grid", 1.0));
            if (n < 2) return fail ("nothing to cut there (the clip must be a set length, and the cuts must leave pieces of at least a few milliseconds)");
            auto* o = new juce::DynamicObject(); o->setProperty ("pieces", n); return ok (juce::var (o));
        }
        if (cmd == "quantiseClips")
        {
            std::vector<int> ids; if (auto* arr = p.getProperty ("instances", {}).getArray()) for (const auto& v : *arr) ids.push_back ((int) v);
            const int n = engine.editQuantiseInstances (inum (p, "track"), ids, num (p, "grid", 1.0), p.hasProperty ("strength") ? num (p, "strength") : 1.0, (bool) p.getProperty ("fill", false));
            auto* o = new juce::DynamicObject(); o->setProperty ("moved", n); return ok (juce::var (o));
        }
        if (cmd == "detectTempo") { const int ci = inum (p, "contentId"); auto v = engine.detectTempoVar (ci); auto* o = new juce::DynamicObject(); o->setProperty ("bpm", v.getProperty ("bpm", 0.0)); o->setProperty ("confidence", v.getProperty ("confidence", 0.0)); return ok (juce::var (o)); }
        // ---- the person's own keyboard shortcuts: { "<command>": ["Ctrl+Shift+X", ...] }, kept beside the skin; the defaults ship inside the program ----
        if (cmd == "getUserKeymap")
        {
            const auto f = engine.settingsDir.getChildFile ("keymap-user.json"); auto v = f.existsAsFile() ? juce::JSON::parse (f.loadFileAsString()) : juce::var();
            auto* o = new juce::DynamicObject(); o->setProperty ("keys", v.getDynamicObject() != nullptr ? v : juce::var (new juce::DynamicObject())); return ok (juce::var (o));
        }
        if (cmd == "setUserKeymap" || cmd == "resetKeymap")
        {
            const auto f = engine.settingsDir.getChildFile ("keymap-user.json");
            if (cmd == "resetKeymap") { f.deleteFile(); return ok(); }
            const auto command = p.getProperty ("command", {}).toString();
            if (command.isEmpty() || command.length() > 48) return fail ("no command given");
            auto cur = f.existsAsFile() ? juce::JSON::parse (f.loadFileAsString()) : juce::var(); auto* obj = cur.getDynamicObject(); if (obj == nullptr) { obj = new juce::DynamicObject(); cur = juce::var (obj); }
            if (! p.hasProperty ("keys")) obj->removeProperty (command);                    // no keys given: back to the default
            else
            {
                juce::Array<juce::var> keys;
                if (auto* arr = p.getProperty ("keys", {}).getArray()) for (const auto& k : *arr) { const auto s = k.toString(); if (s.length() > 0 && s.length() <= 28 && s.containsOnly ("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+ ")) keys.add (s); else return fail ("that is not a key combination"); }
                obj->setProperty (command, keys);
            }
            if (! f.replaceWithText (juce::JSON::toString (cur, true))) return fail ("could not save the shortcuts");
            return ok();
        }
        // ---- automation lanes (a track, folder or FX return; chain numbers as everywhere) ----
        if (cmd == "getAutomation") { auto* o = new juce::DynamicObject(); o->setProperty ("lanes", engine.autoVar (inum (p, "chain"))); return ok (juce::var (o)); }
        if (cmd == "getParams") { auto* o = new juce::DynamicObject(); o->setProperty ("params", engine.paramsVar (inum (p, "chain"), inum (p, "slot"))); return ok (juce::var (o)); }
        if (cmd == "automationLane")
        {
            const auto op = p.getProperty ("op", "add").toString(); const int chain = inum (p, "chain");
            if (op == "add") { const int id = engine.editAutoAddLane (chain, inum (p, "kind"), inum (p, "slot", -1), inum (p, "param", -1)); if (id == 0) return fail ("cannot automate that (no such channel, plug-in or parameter, and the master has no lanes yet)"); auto* o = new juce::DynamicObject(); o->setProperty ("lane", id); return ok (juce::var (o)); }
            if (op == "remove" || op == "enable") { if (! engine.editAutoLane (chain, inum (p, "lane"), op, p.hasProperty ("on") ? (bool) p.getProperty ("on", true) : true)) return fail ("no such lane"); return ok(); }
            return fail ("op must be add, remove or enable");
        }
        if (cmd == "autoRecord") { engine.setAutoRecord ((bool) p.getProperty ("on", true)); return ok(); }
        if (cmd == "autoLearn") { if ((bool) p.getProperty ("on", true)) { engine.startAutoLearn (inum (p, "chain")); if (engine.learn.chain < 0) return fail ("that channel has no plug-ins to learn from (the master has no lanes)"); } else engine.stopAutoLearn(); return ok(); }
        if (cmd == "setAutomationPoints") { if (! engine.editAutoSetPoints (inum (p, "chain"), inum (p, "lane"), p.getProperty ("points", {}))) return fail ("no such lane"); return ok(); }
        // ---- export the song (or the loop) as an audio file ----
        if (cmd == "exportAudio")
        {
            if (exportJob != nullptr) return fail ("an export is already running");
            static const bool testMode2 = juce::SystemStats::getEnvironmentVariable ("ASKARCHIE_HOST_TEST", {}) == "1";
            if (testMode2 && p.hasProperty ("path"))           // a test copy can export straight to a path (no save box to click)
            { exportJob = std::make_unique<MixdownJob> (engine, juce::File (p.getProperty ("path", {}).toString()), inum (p, "bits", 24), p.hasProperty ("from") ? num (p, "from") : -1.0, p.hasProperty ("to") ? num (p, "to") : -1.0, p.hasProperty ("tail") ? num (p, "tail") : 2.0, [this] (const juce::var& v) { emit ("export", v); }); return ok(); }
            const int bits = inum (p, "bits", 24); const double from = p.hasProperty ("from") ? num (p, "from") : -1.0, to = p.hasProperty ("to") ? num (p, "to") : -1.0, tail = p.hasProperty ("tail") ? num (p, "tail") : 2.0;
            const auto name = juce::String (p.getProperty ("name", "Archie DAW mixdown").toString());
            chooser = std::make_shared<juce::FileChooser> ("Export audio", juce::File::getSpecialLocation (juce::File::userMusicDirectory).getChildFile (name + ".wav"), "*.wav");
            chooser->launchAsync (juce::FileBrowserComponent::saveMode | juce::FileBrowserComponent::canSelectFiles | juce::FileBrowserComponent::warnAboutOverwriting, [this, bits, from, to, tail] (const juce::FileChooser& fc)
            {
                auto f = fc.getResult(); if (f == juce::File() || exportJob != nullptr) return;
                if (f.getFileExtension().isEmpty()) f = f.withFileExtension (".wav");
                exportJob = std::make_unique<MixdownJob> (engine, f, bits, from, to, tail, [this] (const juce::var& v) { emit ("export", v); });
            });
            return ok();
        }
        if (cmd == "exportMidi" || cmd == "importMidi")
        {
            const bool isImport = cmd == "importMidi"; static const bool testMode3 = juce::SystemStats::getEnvironmentVariable ("ASKARCHIE_HOST_TEST", {}) == "1";
            auto doIt = [this, isImport] (const juce::File& f)
            {
                juce::String err; const int n = isImport ? engine.importMidiFile (f, err) : engine.exportMidiFile (f, err);
                auto* o = new juce::DynamicObject(); o->setProperty ("state", n > 0 ? "done" : "error"); o->setProperty ("text", err); o->setProperty ("kind", isImport ? "midi-import" : "midi-export"); o->setProperty ("file", f.getFullPathName()); o->setProperty ("tracks", n); emit ("midi", juce::var (o));
            };
            if (testMode3 && p.hasProperty ("path")) { doIt (juce::File (p.getProperty ("path", {}).toString())); return ok(); }
            if (isImport) { chooser = std::make_shared<juce::FileChooser> ("Import a MIDI file", juce::File::getSpecialLocation (juce::File::userMusicDirectory), "*.mid;*.midi"); chooser->launchAsync (juce::FileBrowserComponent::openMode | juce::FileBrowserComponent::canSelectFiles, [doIt] (const juce::FileChooser& fc) { auto f = fc.getResult(); if (f != juce::File()) doIt (f); }); }
            else { chooser = std::make_shared<juce::FileChooser> ("Export a MIDI file", juce::File::getSpecialLocation (juce::File::userMusicDirectory).getChildFile ("Archie DAW song.mid"), "*.mid"); chooser->launchAsync (juce::FileBrowserComponent::saveMode | juce::FileBrowserComponent::canSelectFiles | juce::FileBrowserComponent::warnAboutOverwriting, [doIt] (const juce::FileChooser& fc) { auto f = fc.getResult(); if (f == juce::File()) return; if (f.getFileExtension().isEmpty()) f = f.withFileExtension (".mid"); doIt (f); }); }
            return ok();
        }
        // ---- bounce: one clip, played through its own track (instrument, effects, sends, folder, master), written as a wav ----
        if (cmd == "bounce")
        {
            if (exportJob != nullptr) return fail ("an export is already running");
            const auto mode = p.getProperty ("mode", "place").toString(); const int id = inum (p, "instance", 0); int idx = -1; const int track = engine.trackOfInstance (id, &idx);
            if (track < 0) return fail ("no such clip");
            const ClipInst in = (*engine.trackAt (track)->arrSnap.load()).insts[(size_t) idx];
            if (in.length <= 0.0) return fail ("give the clip a length first: a clip that goes on for ever has no end to bounce to");
            if (mode != "place" && mode != "recording" && mode != "sample" && mode != "disk") return fail ("unknown bounce kind");
            const double tail = p.hasProperty ("tail") ? num (p, "tail") : 2.0; const auto trackName = engine.trackAt (track)->name;
            auto dir = engine.recordingsDir != juce::File() ? engine.recordingsDir : engine.settingsDir.getChildFile ("recordings"); if (mode == "sample") dir = dir.getParentDirectory().getChildFile ("bounced-samples"); dir.createDirectory();
            const auto base = trackName.retainCharacters ("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 _-") + " bounce " + juce::String (juce::Time::currentTimeMillis() % 100000000);
            auto launch = [this, mode, id, in, track, tail, trackName] (juce::File f)
            {
                if (exportJob != nullptr) return;
                if (f.getFileExtension().isEmpty()) f = f.withFileExtension (".wav");
                exportJob = std::make_unique<MixdownJob> (engine, f, 24, in.start, in.start + in.length, tail, [this, mode, id, in, trackName, f] (const juce::var& v)
                {
                    emit ("export", v);
                    if (v.getProperty ("state", {}).toString() != "done") return;
                    auto* o = new juce::DynamicObject(); o->setProperty ("state", "done"); o->setProperty ("file", f.getFullPathName());
                    if (mode == "place" || mode == "recording")
                    {
                        juce::String err; if (! engine.editAddTrack()) { o->setProperty ("state", "error"); o->setProperty ("text", "the bounce is saved but there is no room for another track: " + f.getFullPathName()); emit ("bounce", juce::var (o)); return; }
                        const int nt = engine.trackCount.load() - 1; engine.trackAt (nt)->name = (trackName + " bounce").substring (0, 40);
                        if (engine.editImportAudio (nt, f, in.start, err) == 0) { o->setProperty ("state", "error"); o->setProperty ("text", err); }
                        else if (mode == "place") { auto* pr = new juce::DynamicObject(); pr->setProperty ("muted", true); engine.editSetInstance (id, juce::var (pr)); o->setProperty ("text", "bounced to a new track; the original clip is muted"); }
                        else o->setProperty ("text", "bounced to a new track as a new recording; the original clip still plays");
                    }
                    else o->setProperty ("text", mode == "sample" ? "saved as a new sample in the bounced-samples folder" : "saved");
                    emit ("bounce", juce::var (o));
                }, track);
            };
            static const bool testMode4 = juce::SystemStats::getEnvironmentVariable ("ASKARCHIE_HOST_TEST", {}) == "1";
            if (mode == "disk" && ! (testMode4 && p.hasProperty ("path")))
            {
                chooser = std::make_shared<juce::FileChooser> ("Bounce clip to disk", juce::File::getSpecialLocation (juce::File::userMusicDirectory).getChildFile (base + ".wav"), "*.wav");
                chooser->launchAsync (juce::FileBrowserComponent::saveMode | juce::FileBrowserComponent::canSelectFiles | juce::FileBrowserComponent::warnAboutOverwriting, [launch] (const juce::FileChooser& fc) { auto f = fc.getResult(); if (f != juce::File()) launch (f); });
            }
            else launch (testMode4 && p.hasProperty ("path") ? juce::File (p.getProperty ("path", {}).toString()) : dir.getChildFile (base + ".wav"));
            return ok();
        }
        if (cmd == "copyChannel") { if (! engine.copyChannel (inum (p, "track", -1))) return fail ("no such track"); return ok(); }
        if (cmd == "cutChannel")
        {
            const int ti = inum (p, "track", -1); if (engine.trackAt (ti) == nullptr) return fail ("no such track");
            if (ti != engine.trackCount.load() - 1) return fail ("only the last track can be cut (tracks cannot be taken out of the middle yet); use Copy instead");
            if (engine.trackCount.load() <= 1) return fail ("the last remaining track cannot be cut");
            engine.copyChannel (ti); engine.editRemoveTrack(); return ok();
        }
        if (cmd == "pasteChannel" || cmd == "duplicateChannel")
        {
            const bool dup = cmd == "duplicateChannel";
            auto report = [this] (juce::String err) { auto* o = new juce::DynamicObject(); o->setProperty ("state", err.isEmpty() ? "done" : "error"); o->setProperty ("text", err.isEmpty() ? "the channel and its plug-ins are in" : "the channel is in, but: " + err); emit ("channel", juce::var (o)); };
            int ni;
            if (dup) { if (engine.trackAt (inum (p, "track", -1)) == nullptr) return fail ("no such track"); ni = engine.editDuplicateChannel (inum (p, "track", -1), report); }
            else { if (! engine.hasChannelClipboard()) return fail ("nothing has been copied yet"); ni = engine.editPasteChannel (engine.channelClipboard, {}, report); }
            if (ni < 0) return fail ("that is the most tracks there can be (" + juce::String (kTracks) + ")");
            auto* o = new juce::DynamicObject(); o->setProperty ("track", ni); return ok (juce::var (o));
        }
        if (cmd == "joinInstance")
        {
            juce::String err; const int id = engine.editJoinInstance (inum (p, "instance"), err); if (id == 0) return fail (err);
            auto* o = new juce::DynamicObject(); o->setProperty ("instance", id); return ok (juce::var (o));
        }
        if (cmd == "audioToMidi")
        {
            juce::String err; const int nt = engine.editAudioToMidi (inum (p, "instance"), p.hasProperty ("sensitivity") ? num (p, "sensitivity") : 0.5, err); if (nt < 0) return fail (err);
            auto* o = new juce::DynamicObject(); o->setProperty ("track", nt); return ok (juce::var (o));
        }
        if (cmd == "unusedRecordings")
        {
            auto files = engine.unusedRecordings(); juce::int64 bytes = 0; juce::Array<juce::var> names; for (auto& f : files) { bytes += f.getSize(); names.add (f.getFileName()); }
            auto* o = new juce::DynamicObject(); o->setProperty ("count", (int) files.size()); o->setProperty ("megabytes", (double) bytes / 1048576.0); o->setProperty ("files", names); return ok (juce::var (o));
        }
        if (cmd == "deleteUnusedRecordings")
        {
            if (! (bool) p.getProperty ("confirm", false)) return fail ("this deletes files for good: ask for unusedRecordings first, then send confirm true");
            int n = 0; for (auto& f : engine.unusedRecordings()) if (f.moveToTrash()) ++n;      // the Recycle Bin, so a mistake can be put back
            auto* o = new juce::DynamicObject(); o->setProperty ("deleted", n); return ok (juce::var (o));
        }
        if (cmd == "cancelExport") { if (exportJob != nullptr) exportJob->cancel(); return ok(); }
        if (cmd == "warpMarker")
        {
            const auto why = engine.editWarpMarker (inum (p, "instance"), p.getProperty ("op", "set").toString(), num (p, "sec"), num (p, "beat"), inum (p, "index", -1));
            if (why.isNotEmpty()) return fail (why);
            return ok();
        }
        if (cmd == "getTransients" && p.hasProperty ("instance")) { auto* o = new juce::DynamicObject(); o->setProperty ("hits", engine.hitBeatsVar (inum (p, "instance"), p.hasProperty ("sensitivity") ? num (p, "sensitivity") : 0.5)); return ok (juce::var (o)); }
        if (cmd == "getTransients") { auto* o = new juce::DynamicObject(); o->setProperty ("times", engine.transientsVar (inum (p, "contentId"), p.hasProperty ("sensitivity") ? num (p, "sensitivity") : 0.5)); return ok (juce::var (o)); }
        if (cmd == "getWaveform") { auto* o = new juce::DynamicObject(); o->setProperty ("peaks", engine.waveformVar (inum (p, "contentId"), inum (p, "bins", 400))); return ok (juce::var (o)); }
        if (cmd == "getArrangement") { if (auto* t = engine.trackAt (inum (p, "track"))) { auto* o = new juce::DynamicObject(); o->setProperty ("arrangement", engine.arrangementVar (*t)); return ok (juce::var (o)); } return fail ("no such track"); }
        if (cmd == "addInstance")
        {
            const int id = engine.editAddInstance (inum (p, "track"), num (p, "start"), num (p, "length"), num (p, "offset"), p.getProperty ("clip", {}), inum (p, "contentId", 0));
            if (id == 0) return fail ("could not place that clip (no such track, or no clip with that contentId)");
            auto* o = new juce::DynamicObject(); o->setProperty ("instance", id); return ok (juce::var (o));
        }
        if (cmd == "setInstance")       { if (! engine.editSetInstance (inum (p, "instance"), p)) return fail ("no such clip instance"); return ok(); }
        if (cmd == "removeInstance")    { if (! engine.editRemoveInstance (inum (p, "instance"))) return fail ("no such clip instance"); return ok(); }
        if (cmd == "duplicateInstance")
        {
            const int id = engine.editDuplicateInstance (inum (p, "instance"), num (p, "start"), p.hasProperty ("shared") ? (bool) p.getProperty ("shared", true) : true, inum (p, "toTrack", -1));
            if (id == 0) return fail ("no such clip instance");
            auto* o = new juce::DynamicObject(); o->setProperty ("instance", id); return ok (juce::var (o));
        }
        if (cmd == "makeUnique")        { if (! engine.editMakeUnique (inum (p, "instance"))) return fail ("no such clip instance"); return ok(); }
        if (cmd == "splitInstance")
        {
            const int id = engine.editSplitInstance (inum (p, "instance"), num (p, "at"));
            if (id == 0) return fail ("could not split there (no such clip, or the point is outside it)");
            auto* o = new juce::DynamicObject(); o->setProperty ("instance", id); return ok (juce::var (o));
        }
        if (cmd == "quantiseNotes")
        {
            std::vector<int> ix; if (auto* arr = p.getProperty ("indices", {}).getArray()) for (const auto& v : *arr) ix.push_back ((int) v);
            if (! engine.editQuantiseNotes (inum (p, "contentId"), num (p, "grid", 0.25), p.hasProperty ("strength") ? num (p, "strength") : 1.0, (bool) p.getProperty ("lengths", false), ix)) return fail ("nothing to quantise (no such note clip, or every note is already on the grid)");
            return ok();
        }
        if (cmd == "transposeNotes")
        {
            std::vector<int> ix; if (auto* arr = p.getProperty ("indices", {}).getArray()) for (const auto& v : *arr) ix.push_back ((int) v);
            if (! engine.editTransposeNotes (inum (p, "contentId"), inum (p, "semitones"), ix)) return fail ("nothing to transpose (no such note clip, or no change)");
            return ok();
        }
        if (cmd == "editContent")       { if (! engine.editContent (inum (p, "contentId"), p.getProperty ("clip", {}))) return fail ("no clip with that contentId"); return ok(); }
        if (cmd == "loadFile")
        {
            const int track = inum (p, "track");
            chooser = std::make_shared<juce::FileChooser> ("Choose an audio file", juce::File::getSpecialLocation (juce::File::userMusicDirectory), "*.wav;*.aif;*.aiff;*.flac;*.mp3;*.ogg");
            chooser->launchAsync (juce::FileBrowserComponent::openMode | juce::FileBrowserComponent::canSelectFiles, [this, track] (const juce::FileChooser& fc)
            {
                const auto f = fc.getResult();
                if (f == juce::File()) return;
                juce::String err; if (engine.editImportAudio (track, f, engine.transport.posBeats.load(), err) == 0) emit ("error", err);
                engine.stateDirty.store (true);
            });
            return ok();
        }

        // ---- plugins ----
        if (cmd == "scan")      { engine.startScan (p.hasProperty ("system") ? (bool) p.getProperty ("system", true) : true); return ok(); }
        if (cmd == "addPlugin")
        {
            const auto id = p.getProperty ("id", {}).toString();
            auto desc = engine.knownPlugins.getTypeForIdentifierString (id);
            if (desc == nullptr) return fail ("that plugin is not in the list: scan for plugins first");
            { juce::String nm; int times = 0;
              if (! (bool) p.getProperty ("force", false) && engine.crashedBeforeDesc (*desc, &nm, &times))
              {
                  auto* w = new juce::DynamicObject(); w->setProperty ("id", id); w->setProperty ("name", nm); w->setProperty ("times", times); w->setProperty ("track", inum (p, "track")); emit ("crashwarn", juce::var (w));
                  return fail (nm + " stopped Archie DAW while it was loading " + juce::String (times) + (times == 1 ? " time" : " times") + " before, so it was not loaded this time. You can choose to try it anyway.");
              } }
            const int track = inum (p, "track");
            engine.editAddPlugin (track, *desc, [this] (juce::String err) { if (err.isNotEmpty()) emit ("error", err); });
            return ok();
        }
        if (juce::SystemStats::getEnvironmentVariable ("ASKARCHIE_HOST_TEST", {}) == "1" && cmd == "testCrash") { std::thread ([] { volatile int* p = nullptr; *p = 1; }).detach(); return ok(); }      // test copy only: a fault in a thread of the program, to prove the crash note is written
        if (cmd == "openManual")
        {
            const auto man = juce::File::getSpecialLocation (juce::File::currentExecutableFile).getParentDirectory().getChildFile ("manual").getChildFile ("Archie-DAW-Manual.html");
            if (! man.existsAsFile()) return fail ("the manual is installed with the program; this copy was not run from an installed copy, so it is not here");
            if (! man.startAsProcess()) return fail ("could not open the manual in your browser");
            return ok();
        }
        if (cmd == "forgetCrash") { engine.forgetCrash (p.getProperty ("id", {}).toString()); return ok(); }
        if (cmd == "getCrashed") { auto* o = new juce::DynamicObject(); o->setProperty ("plugins", engine.crashedVar()); return ok (juce::var (o)); }
        if (cmd == "setDeviceMeta")
        {
            if (! engine.setDeviceMeta (p.getProperty ("id", {}).toString(), p)) return fail ("no device id given");
            return ok();
        }
        if (cmd == "removePlugin") { engine.editRemovePlugin (inum (p, "track"), inum (p, "slot")); return ok(); }
        if (cmd == "moveSlotTo")
        {
            const int at = engine.editMoveSlotTo (inum (p, "track"), inum (p, "slot"), p.hasProperty ("toTrack") ? inum (p, "toTrack") : inum (p, "track"), inum (p, "to"));
            if (at < 0) return fail ("no such device or rack");
            auto* o = new juce::DynamicObject(); o->setProperty ("slot", at); return ok (juce::var (o));
        }
        if (cmd == "moveSlot")     { engine.editMoveSlot (inum (p, "track"), inum (p, "slot"), inum (p, "dir")); return ok(); }
        if (cmd == "deviceLinkAll") { return engine.editDeviceLinkAll ((bool) p.getProperty ("on", true)); }
        if (cmd == "deviceShow")   { juce::String e; if (! engine.editDeviceShow (inum (p, "track"), inum (p, "slot"), e)) return fail (e); return ok(); }
        if (cmd == "deviceLink")   { juce::String e; if (! engine.editDeviceLink (inum (p, "track"), inum (p, "slot"), (bool) p.getProperty ("on", true), e)) return fail (e); return ok(); }
        if (cmd == "deviceExtras") { auto* t = engine.chainAt (inum (p, "track")); const int si = inum (p, "slot"); if (t == nullptr || si < 0 || si >= (int) t->slots.size()) return fail ("no such device"); auto* o = new juce::DynamicObject(); o->setProperty ("ok", true); o->setProperty ("extras", HostEngine::archieDrumsExtras (*t->slots[(size_t) si]->proc, true)); return juce::var (o); }
        if (cmd == "bypass")       { engine.editBypass (inum (p, "track"), inum (p, "slot"), (bool) p.getProperty ("on", false)); return ok(); }
        if (cmd == "undo") { const auto label = juce::String (engine.undo.undoLabel()); if (! engine.undo.undo()) return fail ("nothing to undo"); engine.stateDirty.store (true); auto* o = new juce::DynamicObject(); o->setProperty ("label", label); return ok (juce::var (o)); }
        if (cmd == "redo") { const auto label = juce::String (engine.undo.redoLabel()); if (! engine.undo.redo()) return fail ("nothing to redo"); engine.stateDirty.store (true); auto* o = new juce::DynamicObject(); o->setProperty ("label", label); return ok (juce::var (o)); }
        if (cmd == "openEditor")   { engine.openEditor (inum (p, "track"), inum (p, "slot")); return ok(); }

        // ---- MIDI devices and audio settings ----
        if (cmd == "midiIn")  { engine.setMidiInputEnabled (p.getProperty ("id", {}).toString(), (bool) p.getProperty ("on", true)); engine.stateDirty.store (true); return ok(); }
        if (cmd == "midiOut") { if (! engine.openMidiOutput (p.getProperty ("id", {}).toString())) return fail ("could not open that MIDI output"); engine.stateDirty.store (true); return ok(); }
        if (cmd == "audioSettings")
        {
            auto* sel = new juce::AudioDeviceSelectorComponent (engine.deviceManager, 0, HostEngine::kMaxInputs, 0, 2, false, false, true, false);
            sel->setSize (520, 420);
            juce::DialogWindow::LaunchOptions lo;
            lo.content.setOwned (sel); lo.dialogTitle = "Audio settings"; lo.dialogBackgroundColour = juce::Colour (0xff202225);
            lo.escapeKeyTriggersCloseButton = true; lo.useNativeTitleBar = true; lo.resizable = false;
            lo.launchAsync();
            return ok();
        }

        // ---- projects ----
        if (cmd == "newProject")
        {
            engine.loadProject (engine.blankProject(), [this] (juce::String) { currentProject = juce::File(); emit ("opened", "a new song"); });
            return ok();
        }
        if (cmd == "getFiles")
        {
            auto* o = new juce::DynamicObject(); juce::Array<juce::var> rc, tp; for (auto& s : readRecent()) rc.add (s);
            for (const auto& f : templatesDir().findChildFiles (juce::File::findFiles, false, "*.archiestudio")) tp.add (f.getFileNameWithoutExtension());
            for (const auto& f : starterDir().findChildFiles (juce::File::findFiles, false, "*.archiestudio")) if (! tp.contains (f.getFileNameWithoutExtension())) tp.add (f.getFileNameWithoutExtension());      // the starter songs that come with the program
            o->setProperty ("recovery", recoveryPending); if (recoveryPending) o->setProperty ("recoveryTime", recoveryFile.getLastModificationTime().formatted ("%d %b %H:%M"));
            o->setProperty ("recent", rc); o->setProperty ("templates", tp); o->setProperty ("current", currentProject.getFullPathName()); return ok (juce::var (o));
        }
        if (cmd == "openRecent")
        {
            const juce::File f (p.getProperty ("path", {}).toString()); if (! f.existsAsFile() || f.getFileExtension() != ".archiestudio") return fail ("that file is not there any more");
            openProjectFile (f, false); return ok();
        }
        if (cmd == "recoverSession")
        {
            if (! recoveryPending) return fail ("there is nothing to recover");
            recoveryPending = false; auto prev = recoveryFile.getSiblingFile (recoveryFile.getFileNameWithoutExtension() + "-previous.archiestudio"); prev.deleteFile(); recoveryFile.copyFileTo (prev);
            openProjectFile (prev, true); return ok();
        }
        if (cmd == "dismissRecovery")
        {
            if (recoveryPending) { recoveryPending = false; auto prev = recoveryFile.getSiblingFile (recoveryFile.getFileNameWithoutExtension() + "-previous.archiestudio"); prev.deleteFile(); recoveryFile.copyFileTo (prev); lastKeep = juce::Time::currentTimeMillis(); }
            return ok();
        }
        if (cmd == "clearRecent") { recentList().deleteFile(); return ok(); }
        if (cmd == "saveTemplate")
        {
            const auto nm = p.getProperty ("name", {}).toString().retainCharacters ("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 _-").trim(); if (nm.isEmpty()) return fail ("give the template a name (letters, numbers, spaces)");
            const auto f = templatesDir().getChildFile (nm + ".archiestudio"); if (! f.replaceWithText (juce::JSON::toString (engine.saveProject()))) return fail ("could not write the template");
            emit ("saved", "template " + nm); return ok();
        }
        if (cmd == "newFromTemplate")
        {
            const auto tn = p.getProperty ("name", {}).toString().retainCharacters ("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 _-") + ".archiestudio";
            auto f = templatesDir().getChildFile (tn); if (! f.existsAsFile()) f = starterDir().getChildFile (tn); if (! f.existsAsFile()) return fail ("no such template");
            openProjectFile (f, true); return ok();
        }
        if (cmd == "saveProject")
        {
            if (currentProject != juce::File() && ! (bool) p.getProperty ("as", false)) { emit (currentProject.replaceWithText (juce::JSON::toString (engine.saveProject())) ? "saved" : "error", currentProject.getFileName()); return ok(); }
            chooser = std::make_shared<juce::FileChooser> ("Save project", juce::File::getSpecialLocation (juce::File::userDocumentsDirectory).getChildFile ("Archie DAW project.archiestudio"), "*.archiestudio");
            chooser->launchAsync (juce::FileBrowserComponent::saveMode | juce::FileBrowserComponent::canSelectFiles | juce::FileBrowserComponent::warnAboutOverwriting, [this] (const juce::FileChooser& fc)
            {
                auto f = fc.getResult();
                if (f == juce::File()) return;
                if (f.getFileExtension().isEmpty()) f = f.withFileExtension (".archiestudio");
                const bool wrote = f.replaceWithText (juce::JSON::toString (engine.saveProject())); if (wrote) { currentProject = f; addRecent (f); }
                emit (wrote ? "saved" : "error", f.getFileName());
            });
            return ok();
        }
        if (cmd == "loadProject")
        {
            chooser = std::make_shared<juce::FileChooser> ("Open project", juce::File::getSpecialLocation (juce::File::userDocumentsDirectory), "*.archiestudio");
            chooser->launchAsync (juce::FileBrowserComponent::openMode | juce::FileBrowserComponent::canSelectFiles, [this] (const juce::FileChooser& fc)
            {
                const auto f = fc.getResult();
                if (f == juce::File()) return;
                openProjectFile (f, false);
            });
            return ok();
        }

        // ---- the Analyser: measures a plug-in in a separate process, so a plug-in that crashes cannot take the program down ----
        if (cmd == "analyse")
        {
            juce::PluginDescription desc; bool found = false;
            if (p.hasProperty ("id")) { if (auto d = engine.knownPlugins.getTypeForIdentifierString (p.getProperty ("id", {}).toString())) { desc = *d; found = true; } }
            else if (auto* t = engine.chainAt (inum (p, "track", -1)))
            {
                const int slot = inum (p, "slot", -1);
                if (slot >= 0 && slot < (int) t->slots.size()) { desc = t->slots[(size_t) slot]->desc; found = true; }
            }
            if (! found) return fail ("choose a plug-in from the list first");
            const juce::File file (desc.fileOrIdentifier);
            if (! file.exists()) return fail ("that plug-in's file could not be found");
            if (! engine.startAnalysis (file, desc.createIdentifierString(), desc.name)) return fail ("an analysis is already running");
            return ok();
        }
        if (cmd == "cancelAnalysis") { engine.cancelAnalysis(); return ok(); }
        if (cmd == "getAnalysis") { auto* o = new juce::DynamicObject(); o->setProperty ("result", engine.getLastAnalysis()); o->setProperty ("error", engine.getAnalysisState().error); return ok (juce::var (o)); }

        // ---- skins ----
        if (cmd == "getSkin") { auto* o = new juce::DynamicObject(); if (currentSkin.isObject()) o->setProperty ("skin", currentSkin); return ok (juce::var (o)); }
        if (cmd == "setSkin")
        {
            auto skin = cleanSkin (p);
            if (! skin.isObject()) return fail ("that is not a valid skin");
            applySkin (skin, p.getProperty ("src", {}).toString());
            return ok();
        }
        if (cmd == "loadSkinFile")
        {
            chooser = std::make_shared<juce::FileChooser> ("Open a skin", juce::File::getSpecialLocation (juce::File::userDocumentsDirectory), "*.json;*.archieskin");
            chooser->launchAsync (juce::FileBrowserComponent::openMode | juce::FileBrowserComponent::canSelectFiles, [this] (const juce::FileChooser& fc)
            {
                const auto f = fc.getResult();
                if (f == juce::File()) return;
                if (f.getSize() > 64 * 1024) { emit ("error", "that skin file is too big"); return; }
                auto skin = cleanSkin (juce::JSON::parse (f.loadFileAsString()));
                if (! skin.isObject()) { emit ("error", "that file is not a valid skin: it needs a name and a list of CSS variables"); return; }
                applySkin (skin, {});
            });
            return ok();
        }
        if (cmd == "saveSkinFile")
        {
            auto skin = cleanSkin (p);
            if (! skin.isObject()) return fail ("nothing to save");
            auto* o = skin.getDynamicObject(); o->setProperty ("archieSkin", 1);
            auto text = juce::JSON::toString (skin);
            chooser = std::make_shared<juce::FileChooser> ("Save this skin", juce::File::getSpecialLocation (juce::File::userDocumentsDirectory).getChildFile (skin.getProperty ("name", "My skin").toString() + ".archieskin"), "*.archieskin");
            chooser->launchAsync (juce::FileBrowserComponent::saveMode | juce::FileBrowserComponent::canSelectFiles | juce::FileBrowserComponent::warnAboutOverwriting, [this, text] (const juce::FileChooser& fc)
            {
                auto f = fc.getResult();
                if (f == juce::File()) return;
                if (f.getFileExtension().isEmpty()) f = f.withFileExtension (".archieskin");
                emit (f.replaceWithText (text) ? "saved" : "error", f.getFileName());
            });
            return ok();
        }

        // ---- test hooks: only when the program was started with ASKARCHIE_HOST_TEST=1. A web page can never load a project itself (a project names plug-in files), so these are not offered otherwise. ----
        static const bool testMode = juce::SystemStats::getEnvironmentVariable ("ASKARCHIE_HOST_TEST", {}) == "1";
        if (testMode && cmd == "testMidiIn") { juce::uint8 d[3] = { (juce::uint8) inum (p, "status", 0x90), (juce::uint8) inum (p, "d1", 60), (juce::uint8) inum (p, "d2", 100) }; engine.handleIncomingMidiMessage (nullptr, juce::MidiMessage (d, 3)); return ok(); }
        if (testMode && cmd == "testGetProject") { auto* o = new juce::DynamicObject(); o->setProperty ("project", engine.saveProject()); return ok (juce::var (o)); }
        if (testMode && cmd == "testSetProject") { engine.loadProject (p.getProperty ("project", {}), [this] (juce::String err) { emit (err.isEmpty() ? "opened" : "error", err.isEmpty() ? juce::String ("test project") : err); }); return ok(); }

        // ---- window ----
        if (cmd == "showTab") { if (onShowTab) onShowTab (inum (p, "tab")); return ok(); }
        if (cmd == "openStudioPage") { if (onNavigateStudio) onNavigateStudio (p.getProperty ("path", "/daw").toString()); if (onShowTab) onShowTab (1); return ok(); }
        return fail ("unknown command: " + cmd);
    }

    void emit (const juce::String& type, const juce::var& data)
    {
        auto* o = new juce::DynamicObject();
        o->setProperty ("type", type); o->setProperty ("data", data);
        broadcast (juce::var (o));
    }

    void broadcast (const juce::var& v) { for (auto* b : browsers) b->emitEventIfBrowserIsVisible ("archie", v); }

    void timerCallback() override
    {
        engine.service();
        keepSong();
        if (exportJob != nullptr && exportJob->isFinished()) exportJob.reset();
        auto* tick = new juce::DynamicObject();
        tick->setProperty ("type", "tick");
        tick->setProperty ("transport", engine.transportVar());
        tick->setProperty ("meters", engine.meters());
        broadcast (juce::var (tick));
        if (engine.getScanProgress().running) engine.stateDirty.store (true);
        if (engine.getAnalysisState().running) engine.stateDirty.store (true);
        if (const int serial = engine.analysisSerial.load(); serial != lastAnalysisSerial)
        {
            lastAnalysisSerial = serial;
            auto* o = new juce::DynamicObject();
            o->setProperty ("result", engine.getLastAnalysis()); o->setProperty ("error", engine.getAnalysisState().error); o->setProperty ("name", engine.getAnalysisState().name);
            emit ("analysis", juce::var (o));
        }
        if (engine.stateDirty.exchange (false)) emit ("state", engine.snapshot());
    }

    HostEngine& engine;
    int lastAnalysisSerial = 0;
    juce::var currentSkin;
    juce::Array<HostBrowser*> browsers;
    std::shared_ptr<juce::FileChooser> chooser;
    juce::File currentProject;                         // the file Save writes to (none = a new song: Save asks where)
    juce::File recentList() const { return engine.settingsDir.getChildFile (juce::SystemStats::getEnvironmentVariable ("ASKARCHIE_HOST_TEST", {}) == "1" ? "recent-files-test.json" : "recent-files.json"); }
    juce::File starterDir() const { return juce::File::getSpecialLocation (juce::File::currentExecutableFile).getSiblingFile ("Starter songs"); }      // songs that come with the program (they open as a new, untitled song, like a template)
    juce::File templatesDir() const { auto d = engine.settingsDir.getChildFile (juce::SystemStats::getEnvironmentVariable ("ASKARCHIE_HOST_TEST", {}) == "1" ? "templates-test" : "templates"); d.createDirectory(); return d; }
    juce::StringArray readRecent() const
    {
        juce::StringArray out; const auto parsedRecent = juce::JSON::parse (recentList().loadFileAsString()); if (auto* a = parsedRecent.getArray()) for (const auto& v : *a) { const auto s = v.toString(); if (s.isNotEmpty() && juce::File (s).existsAsFile()) out.add (s); }
        return out;
    }
    void addRecent (const juce::File& f)
    {
        auto list = readRecent(); list.removeString (f.getFullPathName()); list.insert (0, f.getFullPathName()); while (list.size() > 10) list.remove (list.size() - 1);
        juce::Array<juce::var> a; for (auto& s : list) a.add (s); recentList().replaceWithText (juce::JSON::toString (juce::var (a)));
    }
    void openProjectFile (const juce::File& f, bool asTemplate)
    {
        auto parsed = juce::JSON::parse (f.loadFileAsString());
        engine.loadProject (parsed, [this, f, asTemplate] (juce::String err)
        {
            if (err.isEmpty() || err.isNotEmpty()) { if (asTemplate) currentProject = juce::File(); else { currentProject = f; addRecent (f); } }
            emit (err.isEmpty() ? "opened" : "error", err.isEmpty() ? f.getFileName() : err);
        });
    }
    std::unique_ptr<MixdownJob> exportJob;          // an audio export in progress (it works on a private copy of the engine)
};

} // namespace archie
