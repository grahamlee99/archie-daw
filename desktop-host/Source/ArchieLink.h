// ArchieLink.h -- connects a hand-built Archie device hosted in the DAW (Archie Drums first) to the person's Ask Archie account.
//
// The DAW does the connecting for the device, once, in one place; the device itself needs no connection code.
//   1. start(): if a saved connection token exists, use it. Otherwise make a secret, open the browser at /connect-device carrying only the
//      secret's SHA-256 (the signed-in page claims with it, after the person confirms), and poll /api/connect/status until it is claimed,
//      then trade the secret for the connection token at /api/connect/token (the same routes the Builder plug-ins use).
//   2. Connected: about twice a second, push the device's parameters to /api/connect/state and drain /api/connect/command/next, applying
//      every parameter the website or Archie asked for. Nothing here touches the audio thread: parameter values are read and set through the
//      plug-in's own parameter objects, which is what a host's automation does.
//   3. A token the server no longer accepts (401) ends the link with a plain message; connecting again starts over.
// The network is injected (Http), so the checks run the whole conversation against a fake server.
#pragma once
#include <juce_core/juce_core.h>
#include <juce_audio_processors/juce_audio_processors.h>
#include <juce_cryptography/juce_cryptography.h>
#include <functional>
#include <deque>

namespace archielink
{
struct Http
{
    // POSTs `body` as JSON to `path` on the site. Returns false when the site could not be reached at all; otherwise fills status and reply.
    std::function<bool (const juce::String& path, const juce::var& body, int& status, juce::var& reply)> post;
};

inline Http realHttp (juce::String base = "https://askarchie.net")
{
    Http h;
    h.post = [base] (const juce::String& path, const juce::var& body, int& status, juce::var& reply)
    {
        status = 0;
        auto url = juce::URL (base + path).withPOSTData (juce::JSON::toString (body, true));
        auto opts = juce::URL::InputStreamOptions (juce::URL::ParameterHandling::inPostData).withExtraHeaders ("Content-Type: application/json\r\n").withConnectionTimeoutMs (8000).withStatusCode (&status).withHttpRequestCmd ("POST");
        auto stream = url.createInputStream (opts);
        if (stream == nullptr) return false;
        reply = juce::JSON::parse (stream->readEntireStreamAsString());
        return true;
    };
    return h;
}

inline juce::String sha256Hex (const juce::String& s) { return juce::SHA256 (s.toUTF8()).toHexString().toLowerCase(); }

class DeviceLink : public juce::Thread
{
public:
    enum State { Off, Waiting, Connected, Failed };

    DeviceLink (juce::AudioProcessor& p, juce::String kind, juce::String runtimeId, juce::File tokenFile, Http h, std::function<void (const juce::String&)> openBrowser, int pollMilliseconds = 500, int pairPollMilliseconds = 1500)
        : juce::Thread ("Archie device link"), proc (p), deviceKind (std::move (kind)), runtimeInstanceId (std::move (runtimeId)), store (std::move (tokenFile)), http (std::move (h)), open (std::move (openBrowser)), pollMs (pollMilliseconds), pairPollMs (pairPollMilliseconds) {}
    ~DeviceLink() override { alive->store (false); stopThread (4000); }

    // Extra live state for the website (for Archie Drums: the pattern being shown). The provider is run on the MESSAGE thread (plug-in state is read there), about once a second; the heartbeat sends the last answer.
    void setExtras (std::function<juce::var (juce::AudioProcessor&)> provider) { extrasProvider = std::move (provider); }
    // The transport, read on every heartbeat (cheap atomics): the website draws the playhead from it and shows tempo and position.
    void setHostInfo (std::function<void (juce::DynamicObject&)> fill) { hostInfo = std::move (fill); }
    void refreshExtrasNow() { if (! extrasProvider) return; auto v = extrasProvider (proc); const juce::ScopedLock l (lock); extrasCache = v; }

    // Which saved connection this link uses. By default the device kind; each copy of a device in the rack has its own key, so several copies each keep their own connection.
    void setTokenKey (juce::String k) { if (k.isNotEmpty()) tokenKey = std::move (k); }
    void start() { if (! isThreadRunning()) { setState (Waiting, "Starting..."); startThread(); } }
    void stop() { stopThread (4000); setState (Off, "Not connected"); }
    State state() const { return st.load(); }
    juce::String message() const { const juce::ScopedLock l (lock); return msg; }
    int pushes() const { return pushCount.load(); }
    int applied() const { return appliedCount.load(); }
    bool hasSavedToken() const { return readToken().isNotEmpty(); }
    // "Show in Ask Archie": tells the website this device is the one to show (it follows if the page is open) and opens the page. Runs in the background; false if not connected.
    bool showInAskArchie (const juce::String& pagePath = "/vst-builder")
    {
        juce::String token; { const juce::ScopedLock l (lock); token = currentToken; }
        if (st.load() != Connected || token.isEmpty()) return false;
        auto h = http; auto openIt = open; const auto gen = (double) juce::Time::currentTimeMillis();
        juce::Thread::launch ([h, token, gen] { juce::DynamicObject::Ptr b = new juce::DynamicObject(); b->setProperty ("connectionToken", token); b->setProperty ("activationGeneration", gen); int s = 0; juce::var r; h.post ("/api/connect/activate", juce::var (b.get()), s, r); });
        if (openIt) openIt (pagePath);
        return true;
    }

    // names of the states, for the page
    static const char* stateName (State s) { return s == Waiting ? "waiting" : s == Connected ? "connected" : s == Failed ? "failed" : "off"; }

private:
    void run() override
    {
        juce::String token = readToken();
        if (token.isNotEmpty())
        {
            setState (Waiting, "Reconnecting...");
            const int r = push (token);
            if (r == 401) { forgetToken (); token = {}; }
            else if (r == 0) { setState (Failed, "Could not reach Ask Archie. Press ARCHIE to try again."); return; }
        }
        if (token.isEmpty())
        {
            token = pair();
            if (token.isEmpty()) return;
        }
        { const juce::ScopedLock l (lock); currentToken = token; }
        setState (Connected, "Connected to your Ask Archie account");
        juce::uint32 lastBeat = 0;
        while (! threadShouldExit())
        {
            const auto now = juce::Time::getMillisecondCounter();
            if (lastBeat == 0 || now - lastBeat >= (juce::uint32) pollMs)        // a heartbeat on its own rhythm, whatever else is going on
            {
                lastBeat = now;
                const int r = push (token);
                if (r == 401) { forgetToken(); setState (Failed, "Ask Archie ended this connection. Press ARCHIE to connect again."); return; }
                if (r == 0) setState (Waiting, "Cannot reach Ask Archie, trying again...");
                else { if (st.load() != Connected) setState (Connected, "Connected to your Ask Archie account"); drainCommands (token); }
            }
            applyEdits (120);                                                    // step edits, a slice at a time, so a long list never stalls the heartbeat
            const auto since = juce::Time::getMillisecondCounter() - lastBeat;
            wait (editQueue.empty() ? (int) juce::jmax<juce::int64> (1, (juce::int64) pollMs - (juce::int64) since) : 1);
        }
    }

    // the browser step: returns the connection token, or an empty string when it failed or was stopped
    juce::String pair()
    {
        const juce::String secret = juce::Uuid().toDashedString().removeCharacters ("-") + juce::Uuid().toDashedString().removeCharacters ("-");
        open ("/connect-device?kind=" + juce::URL::addEscapeChars (deviceKind, true) + "&h=" + sha256Hex (secret) + "&i=" + juce::URL::addEscapeChars (runtimeInstanceId, true));
        setState (Waiting, "Confirm the connection in your browser (sign in to Ask Archie there first if asked)");
        const auto until = juce::Time::getMillisecondCounter() + 10u * 60u * 1000u;
        while (! threadShouldExit() && juce::Time::getMillisecondCounter() < until)
        {
            wait (pairPollMs); if (threadShouldExit()) break;
            int status = 0; juce::var reply;
            juce::DynamicObject::Ptr b = new juce::DynamicObject(); b->setProperty ("pairingCode", secret);
            if (! http.post ("/api/connect/status", juce::var (b.get()), status, reply)) continue;
            const auto s = reply.getProperty ("status", {}).toString();
            if (s == "claimed" || s == "connected")
            {
                int ts = 0; juce::var tr;
                if (! http.post ("/api/connect/token", juce::var (b.get()), ts, tr) || ts != 200) { setState (Failed, "Ask Archie would not give a connection. Press ARCHIE to try again."); return {}; }
                const auto token = tr.getProperty ("connectionToken", {}).toString();
                if (token.isEmpty()) { setState (Failed, "Ask Archie sent no connection. Press ARCHIE to try again."); return {}; }
                saveToken (token); return token;
            }
            if (s == "expired") { setState (Failed, "The confirmation link ran out of time. Press ARCHIE to try again."); return {}; }
        }
        if (! threadShouldExit()) setState (Failed, "No confirmation came from the browser. Press ARCHIE to try again.");
        return {};
    }

    // one heartbeat: every parameter, as the website and Archie read them. Returns the HTTP status, or 0 when unreachable.
    int push (const juce::String& token)
    {
        juce::Array<juce::var> params;
        for (auto* p : proc.getParameters())
        {
            juce::String id; if (auto* w = dynamic_cast<juce::AudioProcessorParameterWithID*> (p)) id = w->paramID; else if (auto* hp = dynamic_cast<juce::HostedAudioProcessorParameter*> (p)) id = hp->getParameterID();
            if (id.isEmpty()) continue;
            if (p->getName (64).startsWithIgnoreCase ("MIDI ") || p->getName (64).startsWith ("Step edit")) continue;          // the plug-in wrapper's own MIDI controller parameters: hundreds of them, not part of the device
            juce::DynamicObject::Ptr o = new juce::DynamicObject(); o->setProperty ("id", id); o->setProperty ("name", p->getName (64)); o->setProperty ("value", (double) p->getValue()); params.add (juce::var (o.get()));
        }
        juce::DynamicObject::Ptr host = new juce::DynamicObject(); host->setProperty ("device", deviceKind); host->setProperty ("hostedBy", "Archie DAW");
        if (hostInfo) hostInfo (*host);
        if (extrasProvider) { requestExtras(); const juce::ScopedLock l (lock); if (auto* eo = extrasCache.getDynamicObject()) for (auto& kv : eo->getProperties()) host->setProperty (kv.name, kv.value); }
        juce::DynamicObject::Ptr b = new juce::DynamicObject(); b->setProperty ("connectionToken", token); b->setProperty ("hostContext", juce::var (host.get())); b->setProperty ("parameters", params);
        int status = 0; juce::var reply; if (! http.post ("/api/connect/state", juce::var (b.get()), status, reply)) return 0;
        if (status == 200) ++pushCount;
        return status;
    }

// Archie Drums keeps eight mailboxes ("Step edit 1" to "Step edit 8") for edits from the website. A batch sets up to eight, waits for the plug-in to read them (it reads once per audio block), then sets them all back to idle.
    static constexpr int kMailboxIdle = 12288;
    void findMailboxes()
    {
        if (! mailboxes.empty()) return;
        for (int k = 1; k <= 8; ++k) for (auto* p : proc.getParameters()) if (p->getName (64) == "Step edit " + juce::String (k)) { mailboxes.push_back (p); break; }
        if (mailboxes.size() != 8) mailboxes.clear();
    }
    int currentPatternIndex()
    {
        int bank = 0, slot = 0; for (auto* p : proc.getParameters()) { const auto n = p->getName (64); if (n == "Bank") bank = (int) std::lround (p->getValue() * 3.0f); else if (n == "Pattern") slot = (int) std::lround (p->getValue() * 3.0f); }
        return juce::jlimit (0, 15, bank * 4 + slot);
    }
    void applyEdits (int budgetMs)
    {
        if (editQueue.empty()) return; findMailboxes(); if (mailboxes.empty()) { editQueue.clear(); return; }
        const auto until = juce::Time::getMillisecondCounter() + (juce::uint32) budgetMs;
        while (! editQueue.empty() && ! threadShouldExit() && juce::Time::getMillisecondCounter() < until)
        {
            const int n = (int) juce::jmin<size_t> (8, editQueue.size());
            for (int k = 0; k < n; ++k) { mailboxes[(size_t) k]->setValueNotifyingHost ((float) editQueue[(size_t) k] / (float) kMailboxIdle); }
            wait (25);
            for (int k = 0; k < n; ++k) mailboxes[(size_t) k]->setValueNotifyingHost (1.0f);
            wait (25);
            for (int k = 0; k < n; ++k) { editQueue.pop_front(); ++appliedCount; }
        }
    }

    void drainCommands (const juce::String& token)
    {
        juce::DynamicObject::Ptr b = new juce::DynamicObject(); b->setProperty ("connectionToken", token);
        int status = 0; juce::var reply; if (! http.post ("/api/connect/command/next", juce::var (b.get()), status, reply) || status != 200) return;
        auto apply = [this] (const juce::var& e)
        {
            const auto id = e.getProperty ("parameterId", {}).toString(); if (id.isEmpty() || ! e.hasProperty ("normalizedValue")) return;
            const float v = juce::jlimit (0.0f, 1.0f, (float) (double) e.getProperty ("normalizedValue", 0.0));
            for (auto* p : proc.getParameters())
            {
                juce::String pid; if (auto* w = dynamic_cast<juce::AudioProcessorParameterWithID*> (p)) pid = w->paramID; else if (auto* hp = dynamic_cast<juce::HostedAudioProcessorParameter*> (p)) pid = hp->getParameterID();
                if (pid == id) { p->beginChangeGesture(); p->setValueNotifyingHost (v); p->endChangeGesture(); ++appliedCount; return; }
            }
        };
        // step edits for Archie Drums are queued here and applied by applyEdits (the plug-in has eight mailboxes, so eight edits go in at once)
        auto stepEdits = [this] (const juce::var& ops)
        {
            auto* arr = ops.getArray(); if (arr == nullptr) return;
            for (auto& o : *arr)
            {
                const int d = (int) o.getProperty ("drum", -1), st2 = (int) o.getProperty ("step", -1), v = (int) o.getProperty ("value", -1);
                const int pat = o.hasProperty ("pattern") ? (int) o.getProperty ("pattern", -1) : currentPatternIndex();
                if (d < 0 || d > 15 || st2 < 0 || st2 > 15 || v < 0 || v > 2 || pat < 0 || pat > 15) continue;
                editQueue.push_back (((pat * 16 + d) * 16 + st2) * 3 + v);
            }
        };
        if (reply.getProperty ("command", {}).isObject()) apply (reply.getProperty ("command", {}));
        if (auto* batch = reply.getProperty ("parameterBatch", {}).getArray()) for (auto& e : *batch) { if (e.hasProperty ("deviceOps")) stepEdits (e.getProperty ("deviceOps", {})); else apply (e); }
    }

    juce::String readToken() const
    {
        if (! store.existsAsFile()) return {};
        auto v = juce::JSON::parse (store); return v.getProperty (tokenKey, {}).toString();
    }
    void saveToken (const juce::String& t)
    {
        juce::var v = store.existsAsFile() ? juce::JSON::parse (store) : juce::var();
        if (! v.isObject()) v = juce::var (new juce::DynamicObject());
        v.getDynamicObject()->setProperty (tokenKey, t);
        store.getParentDirectory().createDirectory(); store.replaceWithText (juce::JSON::toString (v, false));
    }
    void forgetToken() { saveToken ({}); }
    void requestExtras()
    {
        const auto now = juce::Time::getMillisecondCounter(); if (now - lastExtras < 900u) return; lastExtras = now;
        auto flag = alive; auto* self = this;
        juce::MessageManager::callAsync ([flag, self] { if (flag->load()) self->refreshExtrasNow(); });
    }
    void setState (State s, const juce::String& m) { st.store (s); const juce::ScopedLock l (lock); msg = m; }

    juce::AudioProcessor& proc; juce::String deviceKind, runtimeInstanceId, tokenKey = deviceKind; juce::File store; Http http; std::function<void (const juce::String&)> open;
    int pollMs, pairPollMs; std::atomic<State> st { Off }; std::atomic<int> pushCount { 0 }, appliedCount { 0 };
    mutable juce::CriticalSection lock; juce::String msg { "Not connected" }, currentToken;
    std::deque<int> editQueue; std::vector<juce::AudioProcessorParameter*> mailboxes;
    std::function<void (juce::DynamicObject&)> hostInfo; std::function<juce::var (juce::AudioProcessor&)> extrasProvider; juce::var extrasCache; juce::uint32 lastExtras = 0; std::shared_ptr<std::atomic<bool>> alive = std::make_shared<std::atomic<bool>> (true);
};
} // namespace archielink
