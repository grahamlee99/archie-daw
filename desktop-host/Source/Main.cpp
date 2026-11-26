// Archie Studio host (desktop) -- the program and its window.
#include <juce_gui_extra/juce_gui_extra.h>
#include <map>
#include "Engine.h"
#include "Bridge.h"
#include "Analyser.h"

using namespace archie;

#if JUCE_WINDOWS
extern "C" __declspec (dllimport) unsigned int __stdcall SetErrorMode (unsigned int);   // declared here so windows.h need not be pulled in next to JUCE
#endif

// The floating Track Panel: the same page as the console, showing only the panel, in its own window that stays on top. It talks to the same engine through the same bridge.
#if JUCE_WINDOWS
void installArchieCrashFilter (const wchar_t* file);        // CrashGuard.cpp
#endif

class PanelWindow : public juce::DocumentWindow, private juce::Timer
{
public:
    PanelWindow (Bridge& b, const juce::StringArray& hosts, const juce::String& windowKey, const juce::String& title) : juce::DocumentWindow (title, juce::Colour (0xff15171a), juce::DocumentWindow::closeButton | juce::DocumentWindow::minimiseButton | juce::DocumentWindow::maximiseButton), bridge (b), key (windowKey)
    {
        setUsingNativeTitleBar (true);
        auto* br = new HostBrowser (b.makeOptions(), hosts);
        browser = br;
        setContentOwned (br, false);
        bridge.addBrowser (br);
        setResizable (true, true);
        setResizeLimits (640, 300, 3000, 1400);
        setAlwaysOnTop (true);
        centreWithSize (key == "panel" ? 1000 : 1200, key == "panel" ? 480 : 700);
        { const auto f = stateFile(); if (f.existsAsFile()) { const auto st = f.loadFileAsString().trim(); if (st.isNotEmpty()) restoreWindowStateFromString (st); } }     // each window comes back where and as big as it was left (own file per window)
        startTimer (500);
    }
    ~PanelWindow() override { saveState(); stopTimer(); bridge.removeBrowser (browser); }
    void show (const juce::String& query)
    {
        browser->wantUrl (juce::WebBrowserComponent::getResourceProviderRoot() + query);
        setVisible (true); toFront (true);
    }
    void closeButtonPressed() override { saveState(); setVisible (false); }
private:
    juce::File stateFile() const { return juce::File::getSpecialLocation (juce::File::userApplicationDataDirectory).getChildFile ("AskArchie").getChildFile ("Studio").getChildFile ("window-" + key + (juce::SystemStats::getEnvironmentVariable ("ASKARCHIE_HOST_TEST", {}) == "1" ? "-test" : "") + ".txt"); }      // a test copy keeps its windows apart from the real program's
    void saveState() { const auto f = stateFile(); f.getParentDirectory().createDirectory(); f.replaceWithText (getWindowStateAsString()); }
    void timerCallback() override { browser->retryIfNeeded(); }
    Bridge& bridge; juce::String key; HostBrowser* browser = nullptr;
};

class MainComponent : public juce::Component, private juce::Timer
{
public:
    MainComponent (Bridge& b) : bridge (b)
    {
        studioBase = juce::SystemStats::getEnvironmentVariable ("ASKARCHIE_STUDIO_URL", "https://askarchie.net");
        juce::StringArray hosts { "askarchie.net", "www.askarchie.net" };
        const auto overrideHost = juce::URL (studioBase).getDomain().toLowerCase();
        if (overrideHost.isNotEmpty()) hosts.addIfNotAlreadyThere (overrideHost);

        console = std::make_unique<HostBrowser> (bridge.makeOptions(), hosts);
        allowedHosts = hosts;
        addAndMakeVisible (*console);
        bridge.addBrowser (console.get());

        for (auto* tb : { &tabConsole, &tabStudio, &tabSite })
        {
            tb->setClickingTogglesState (false);
            tb->setColour (juce::TextButton::buttonColourId, juce::Colour (0xff2a2d33));
            tb->setColour (juce::TextButton::buttonOnColourId, juce::Colour (0xff6d4aff));
            addAndMakeVisible (*tb);
        }
        // THE TRANSPORT DOCK (2026-11-21, the transport module, unhooked from the page and used at the bottom, like Reason and Logic): the SAME transport module the Console shows (same code, same look),
        // taken off the Console page and shown in its own small web view locked to the bottom of the window, so it is there on every tab and on every page of the site.
        // (created once the Console page has loaded: the web view builds its browsers one at a time, and a second visible browser asked for at the same moment starves the first)
        tabConsole.setButtonText ("Console");
        tabStudio.setButtonText ("Archie's Studio (online)");
        tabSite.setButtonText ("askarchie.net");
        tabConsole.onClick = [this] { showTab (0); };
        tabStudio.onClick  = [this] { showTab (1, "/daw"); };
        tabSite.onClick    = [this] { showTab (1, "/"); };

        bridge.onShowTab = [this] (int t) { showTab (t); };
        // the Track Panel, the Rack, the Mixer and the Strips each open in a window of their own (a second screen), each remembered separately
        bridge.onOpenPanel = [this] (const juce::String& q)
        {
            const juce::String key = q.contains ("eqwin=1") ? "eq" : q.contains ("view=rack") ? "rack" : q.contains ("view=mixer") ? "mixer" : q.contains ("view=strips") ? "strips" : q.contains ("view=settings") ? "settings" : "panel";
            const juce::String title = key == "eq" ? "Archie DAW - EQ" : key == "rack" ? "Archie DAW - Rack" : key == "mixer" ? "Archie DAW - Mixer" : key == "strips" ? "Archie DAW - Strips" : key == "settings" ? "Archie DAW - Settings" : "Track panel";
            auto& w = panels[key]; if (w == nullptr) w = std::make_unique<PanelWindow> (bridge, allowedHosts, key, title);
            w->show (q);
        };
        bridge.onClosePanel = [this] { auto it = panels.find ("panel"); if (it != panels.end() && it->second != nullptr) it->second->setVisible (false); };
        bridge.onSkin = [this] (const juce::var& skin) { applySkin (skin); };
        applySkin (bridge.getSkin());
        bridge.onNavigateStudio = [this] (const juce::String& path) { ensureWeb(); webPath = path; web->wantUrl (studioBase + path); webLoaded = true; };

        console->wantUrl (juce::WebBrowserComponent::getResourceProviderRoot());
        startTimer (500);
        setSize (1280, 820);
        showTab (0);
    }

    ~MainComponent() override { if (transportDock != nullptr) bridge.removeBrowser (transportDock.get()); bridge.onOpenPanel = nullptr; bridge.onClosePanel = nullptr; panels.clear(); bridge.removeBrowser (console.get()); bridge.removeBrowser (web.get()); }

    // The online page is created the first time it is asked for. The web view builds its browsers one at a time, so a hidden second browser must not queue ahead of the visible one.
    void ensureWeb()
    {
        if (web != nullptr) return;
        web = std::make_unique<HostBrowser> (bridge.makeOptions(), allowedHosts);
        addChildComponent (*web);
        bridge.addBrowser (web.get());
    }

    void showTab (int t, const juce::String& path = {})
    {
        if (t == 1) ensureWeb();
        if (t == 1 && (path.isNotEmpty() && path != webPath || ! webLoaded)) { webPath = path.isNotEmpty() ? path : "/daw"; web->wantUrl (studioBase + webPath); webLoaded = true; }
        console->setVisible (t == 0);
        if (web != nullptr) web->setVisible (t == 1);
        tabConsole.setToggleState (t == 0, juce::dontSendNotification);
        tabStudio.setToggleState (t == 1 && webPath == "/daw", juce::dontSendNotification);
        tabSite.setToggleState (t == 1 && webPath != "/daw", juce::dontSendNotification);
        resized();
    }

    void ensureDock()
    {
        if (transportDock != nullptr || ! console->isLoaded()) return;
        transportDock = std::make_unique<HostBrowser> (bridge.makeOptions(), allowedHosts);
        addAndMakeVisible (*transportDock); bridge.addBrowser (transportDock.get());
        transportDock->wantUrl (juce::WebBrowserComponent::getResourceProviderRoot() + "?dock=transport");
        resized();
    }
    void timerCallback() override { ensureDock(); console->retryIfNeeded(); if (transportDock != nullptr) transportDock->retryIfNeeded(); if (web != nullptr) web->retryIfNeeded(); }

    // The native tab bar and window background follow the skin (#rgb or #rrggbb values only; anything else keeps the default colour).
    static juce::Colour skinColour (const juce::var& skin, const char* key, juce::Colour fallback)
    {
        auto s = skin.getProperty ("vars", {}).getProperty (key, {}).toString().trim();
        if (s.startsWith ("#") && (s.length() == 4 || s.length() == 7) && s.substring (1).containsOnly ("0123456789abcdefABCDEF"))
        {
            if (s.length() == 4) s = "#" + juce::String::charToString (s[1]) + s[1] + s[2] + s[2] + s[3] + s[3];
            return juce::Colour::fromString ("ff" + s.substring (1));
        }
        return fallback;
    }
    void applySkin (const juce::var& skin)
    {
        bg = skinColour (skin, "--bg", juce::Colour (0xff121412));
        const auto tab = skinColour (skin, "--ctl-bg", juce::Colour (0xff2f342f)), on = skinColour (skin, "--accent", juce::Colour (0xffd9a856));
        const auto text = skinColour (skin, "--ctl-text", juce::Colour (0xffd7e3d7)), ink = skinColour (skin, "--accent-ink", juce::Colour (0xff1c1f1c));
        for (auto* tb : { &tabConsole, &tabStudio, &tabSite })
        {
            tb->setColour (juce::TextButton::buttonColourId, tab); tb->setColour (juce::TextButton::buttonOnColourId, on);
            tb->setColour (juce::TextButton::textColourOffId, text); tb->setColour (juce::TextButton::textColourOnId, ink);
        }
        repaint();
    }

    void paint (juce::Graphics& g) override { g.fillAll (bg); }
    void resized() override
    {
        auto r = getLocalBounds();
        auto bar = r.removeFromTop (30);
        bar.reduce (6, 3);
        tabConsole.setBounds (bar.removeFromLeft (110)); bar.removeFromLeft (4);
        tabStudio.setBounds (bar.removeFromLeft (190)); bar.removeFromLeft (4);
        tabSite.setBounds (bar.removeFromLeft (130));
        if (transportDock != nullptr) transportDock->setBounds (r.removeFromBottom (kDockHeight));
        console->setBounds (r); if (web != nullptr) web->setBounds (r);
    }

private:
    Bridge& bridge;
    juce::String studioBase, webPath;
    juce::StringArray allowedHosts;
    bool webLoaded = false;
    std::unique_ptr<HostBrowser> console, web;
    std::map<juce::String, std::unique_ptr<PanelWindow>> panels;
    std::unique_ptr<HostBrowser> transportDock;
    static constexpr int kDockHeight = 58;                // one slim line of the transport module
    juce::TextButton tabConsole, tabStudio, tabSite;
    juce::Colour bg { 0xff121412 };
};

class ArchieStudioApp : public juce::JUCEApplication
{
public:
    const juce::String getApplicationName() override { return "Archie DAW"; }
    const juce::String getApplicationVersion() override { return kAppVersion; }
    // The program starts itself headless as a child process for plug-in scanning ("--scan") and for the Analyser ("--analyse"); those must be allowed alongside the open window.
    bool moreThanOneInstanceAllowed() override
    {
        const auto cl = getCommandLineParameters();
        return cl.contains ("--scan") || cl.contains ("--analyse") || juce::SystemStats::getEnvironmentVariable ("ASKARCHIE_HOST_TEST", {}) == "1";   // the last one lets a test copy run beside a person's own open window
    }

    void initialise (const juce::String& commandLine) override
    {
        const auto args = juce::StringArray::fromTokens (commandLine, true);
        if (args.size() >= 3 && (args[0] == "--scan" || args[0] == "--analyse"))
        {
            headless = true;
#if JUCE_WINDOWS
            // A broken plug-in file makes Windows show a "Bad Image" or crash dialog that waits for a click. In these background processes nobody can click it, so switch every such dialog off: the load just fails and the parent carries on.
            SetErrorMode (0x0001 /* SEM_FAILCRITICALERRORS */ | 0x0002 /* SEM_NOGPFAULTERRORBOX */ | 0x8000 /* SEM_NOOPENFILEERRORBOX */);
#endif
            int code = 0;
            const juce::File in (args[1].unquoted()), out (args[2].unquoted());
            if (args[0] == "--scan")
            {
                // list the plug-ins in one file and write them as XML; a crash here is exactly what the parent process is protecting the program from
                juce::AudioPluginFormatManager fm; fm.addDefaultFormats();
                juce::KnownPluginList list; juce::OwnedArray<juce::PluginDescription> descs;
                for (auto* f : fm.getFormats()) if (f->getName() == "VST3") f->findAllTypesForFile (descs, in.getFullPathName());
                for (auto* d : descs) list.addType (*d);
                if (auto xml = list.createXml()) { xml->writeTo (out); } else code = 1;
                if (descs.isEmpty()) code = 1;
            }
            else
            {
                std::string error;
                const auto id = args.size() > 3 ? args[3].unquoted() : juce::String();
                const auto json = Analyser::run (in, id, error);
                if (json.empty()) { out.replaceWithText ("{\"error\":\"" + juce::String (error).replace ("\"", "'") + "\"}"); code = 1; }
                else out.replaceWithText (juce::String (json));
            }
            setApplicationReturnValue (code);
            quit();
            return;
        }
        engine = std::make_unique<HostEngine>();
#if JUCE_WINDOWS
        { const auto cf = engine->crashModuleFile().getFullPathName(); if (cf.length() < 1090) installArchieCrashFilter (cf.toWideCharPointer()); }
#endif
        bridge = std::make_unique<Bridge> (*engine);
        engine->startEmpty();                                           // the Transport and the Master only: tracks are dragged in from the browser or added with + Track
        window = std::make_unique<Window> (getApplicationName(), *bridge);
        engine->start();
    }

    void shutdown() override
    {
        if (headless) return;
        window.reset();
        bridge.reset();
        engine.reset();
    }

    void systemRequestedQuit() override { quit(); }
    void anotherInstanceStarted (const juce::String&) override { if (window != nullptr) window->toFront (true); }

private:
    struct Window : public juce::DocumentWindow
    {
        Window (const juce::String& name, Bridge& b)
            : juce::DocumentWindow (name, juce::Colour (0xff15171a), juce::DocumentWindow::allButtons)
        {
            setUsingNativeTitleBar (true);
            setContentOwned (new MainComponent (b), true);
            setResizable (true, true);
            setResizeLimits (900, 560, 4000, 3000);
            centreWithSize (getWidth(), getHeight());
            setVisible (true);
        }
        void closeButtonPressed() override { juce::JUCEApplication::getInstance()->systemRequestedQuit(); }
    };

    bool headless = false;
    std::unique_ptr<HostEngine> engine;
    std::unique_ptr<Bridge> bridge;
    std::unique_ptr<Window> window;
};

START_JUCE_APPLICATION (ArchieStudioApp)
