// Offline checks of the MIXER RENDERER (no sound card, no window): plug-in delay compensation, the channel rules, bypass, sends/returns/master, and that the result is BIT-IDENTICAL
// whether one thread or several did the work. Every expectation is worked out here independently of the renderer.
#include "../Source/Engine.h"
#include "../Source/Warp.h"
#include "../Source/Pitch.h"
#include "../Source/Export.h"
#include "../Source/ArchieLink.h"
#include "../Source/KeepLoaded.h"
#include <cstdio>
#include <set>
#include <array>
#include <thread>
#include <atomic>
#include <chrono>
#include <cstring>

#if defined(_WIN32)
extern "C" __declspec (dllimport) unsigned int __stdcall SetErrorMode (unsigned int);
#endif
using namespace archie;
static int failures = 0, checks = 0;
#define CHECK(cond, msg) do { ++checks; if (!(cond)) { ++failures; std::printf("FAIL: %s  (line %d)\n", msg, __LINE__); } } while (0)

// a test processor with chosen channel counts and a chosen reported latency; it delays by exactly that many samples and can scale or replace the signal
class TestProc : public juce::AudioProcessor
{
public:
    TestProc (int ins, int outs, int latency, float gain = 1.0f, float constant = 0.0f, bool replace = false)
        : juce::AudioProcessor (makeBuses (ins, outs)), lat (latency), g (gain), k (constant), rep (replace), nIn (ins), nOut (outs) { ring.setSize (2, 4096); ring.clear(); setLatencySamples (latency); }
    static BusesProperties makeBuses (int ins, int outs)
    {
        BusesProperties b;
        if (ins > 0) b = b.withInput ("in", ins == 1 ? juce::AudioChannelSet::mono() : juce::AudioChannelSet::stereo(), true);
        if (outs > 0) b = b.withOutput ("out", outs == 1 ? juce::AudioChannelSet::mono() : juce::AudioChannelSet::stereo(), true);
        return b;
    }
    const juce::String getName() const override { return "TestProc"; }
    void prepareToPlay (double, int) override { ring.clear(); w = 0; }
    void releaseResources() override {}
    void processBlock (juce::AudioBuffer<float>& b, juce::MidiBuffer&) override
    {
        const int n = b.getNumSamples();
        for (int c = 0; c < b.getNumChannels() && c < 2; ++c)
        {
            float* x = b.getWritePointer (c); float* r = ring.getWritePointer (c); int wp = w;
            for (int i = 0; i < n; ++i) { r[wp & 4095] = x[i]; const float d = lat > 0 ? r[(wp - lat) & 4095] : x[i]; x[i] = rep ? k : d * g + k; ++wp; }
        }
        w = (w + n) & 4095;
    }
    double getTailLengthSeconds() const override { return 0; }
    bool acceptsMidi() const override { return false; }
    bool producesMidi() const override { return false; }
    juce::AudioProcessorEditor* createEditor() override { return nullptr; }
    bool hasEditor() const override { return false; }
    int getNumPrograms() override { return 1; }
    int getCurrentProgram() override { return 0; }
    void setCurrentProgram (int) override {}
    const juce::String getProgramName (int) override { return {}; }
    void changeProgramName (int, const juce::String&) override {}
    void getStateInformation (juce::MemoryBlock&) override {}
    void setStateInformation (const void*, int) override {}
private:
    int lat; float g, k; bool rep; int nIn, nOut; juce::AudioBuffer<float> ring; int w = 0;
};
// the renderer asks the processor for its latency through the base class: report it there
struct LatProc : TestProc { using TestProc::TestProc; };

// an effect with a side-chain input (a second input bus, off until switched on): it passes its main sound and keeps what arrived on the side-chain
class ScProc : public TestProc
{
public:
    ScProc() : TestProc (2, 2, 0) { addBus (true); if (auto* b = getBus (true, 1)) b->enable (false); }
    bool canAddBus (bool isInput) const override { return isInput; }
    bool isBusesLayoutSupported (const BusesLayout& l) const override { return l.getMainInputChannelSet() == juce::AudioChannelSet::stereo() && l.getMainOutputChannelSet() == juce::AudioChannelSet::stereo(); }
    void processBlock (juce::AudioBuffer<float>& b, juce::MidiBuffer&) override
    {
        auto* bus = getBus (true, 1); const bool on = bus != nullptr && bus->isEnabled();
        for (int i = 0; i < b.getNumSamples(); ++i) heard.push_back (on ? b.getSample (getChannelIndexInProcessBlockBuffer (true, 1, 0), i) : 0.0f);
    }
    float peakFrom (size_t a) const { float pk = 0; for (size_t i = a; i < heard.size(); ++i) pk = std::max (pk, std::abs (heard[i])); return pk; }
    std::vector<float> heard;
};

// stands in for a heavy plug-in: a fixed amount of arithmetic per block
class HeavyProc : public TestProc
{
public:
    HeavyProc() : TestProc (2, 2, 0) {}
    void processBlock (juce::AudioBuffer<float>& b, juce::MidiBuffer& m) override
    {
        float acc = 0.0f; for (int i = 0; i < 60000; ++i) acc = std::sin (acc + (float) i * 1.0e-4f) * 0.999f;
        sink += acc * 1.0e-30f;
        TestProc::processBlock (b, m);
    }
    float sink = 0;
};

static std::vector<float> run (HostEngine& e, int blocks, int bs, int channel = 0)
{
    std::vector<float> out; std::vector<float> l ((size_t) bs), r ((size_t) bs); float* o[2] = { l.data(), r.data() };
    for (int b = 0; b < blocks; ++b) { e.processOffline (nullptr, 0, o, 2, bs); const auto& src = channel == 0 ? l : r; out.insert (out.end(), src.begin(), src.end()); }
    return out;
}

int main()
{
    juce::ScopedJuceInitialiser_GUI init; std::setvbuf (stdout, nullptr, _IONBF, 0);
   #if JUCE_WINDOWS
    SetErrorMode (0x0001 | 0x0002);      // a crash ends the process at once instead of waiting on the error-report machinery
   #endif
    const double sr = 48000.0; const int bs = 256;
    if (auto* plugs = std::getenv ("ARCHIE_PROBE_UNLOAD"))      // UNLOAD PROBE (not a check): load real plug-ins ("a.vst3;b.vst3") with their windows, remove them in the same order, then close down, and see that the program lives. ARCHIE_PROBE_KEEP=1 keeps their code loaded.
    {
        if (std::getenv ("ARCHIE_PROBE_GUARD") != nullptr) std::printf ("probe: shared window type guarded %d\n", claimSharedWebViewWindowClass() ? 1 : 0);
        struct Probe : public juce::Timer
        {
            juce::AudioPluginFormatManager fm; juce::StringArray paths; std::vector<std::unique_ptr<juce::AudioPluginInstance>> insts; std::vector<std::unique_ptr<PluginWindow>> eds; bool keep = false; int next = 0, gone = 0, ticks = 0, idle = 0;
            void timerCallback() override
            {
                if (++ticks < 250) return; ticks = 0;      // five seconds a step: long enough for the web page in a plug-in window to be fully up, and for anything late to arrive
                if (next < paths.size())
                {
                    juce::OwnedArray<juce::PluginDescription> d; for (auto* f : fm.getFormats()) if (f->getName() == "VST3") f->findAllTypesForFile (d, paths[next]);
                    juce::String err; std::unique_ptr<juce::AudioPluginInstance> in; if (d.size() > 0) in = fm.createPluginInstance (*d[0], 48000.0, 256, err); if (in == nullptr) { std::printf ("probe: could not load %s %s\n", paths[next].toRawUTF8(), err.toRawUTF8()); ++next; return; }
                    if (keep) keepPluginModulesLoaded();
                    in->prepareToPlay (48000.0, 256);
                    std::unique_ptr<PluginWindow> ed; if (std::getenv ("ARCHIE_PROBE_NOWINDOW") == nullptr) ed = std::make_unique<PluginWindow> (*in);
                    std::printf ("probe: loaded %s\n", in->getName().toRawUTF8()); insts.push_back (std::move (in)); eds.push_back (std::move (ed)); ++next; return;
                }
                if (gone < (int) insts.size()) { eds[(size_t) gone].reset(); insts[(size_t) gone].reset(); std::printf ("probe: removed number %d\n", gone + 1); ++gone; return; }
                if (++idle < 2) return;      // one more step with everything gone
                stopTimer(); juce::MessageManager::getInstance()->stopDispatchLoop();
            }
        };
        {
            Probe pr; pr.fm.addDefaultFormats(); pr.paths.addTokens (juce::String::fromUTF8 (plugs), ";", ""); pr.keep = std::getenv ("ARCHIE_PROBE_KEEP") != nullptr; pr.startTimer (20);
            juce::MessageManager::getInstance()->runDispatchLoop(); pr.stopTimer();
        }
        std::printf ("probe: all removed, closing down\n");
        return 0;
    }
    if (auto* probe = std::getenv ("ARCHIE_PROBE_SONG"))      // LEVEL PROBE (not a check): load a saved song with its real plug-ins, play it offline and print each track's peak, the master's peak and the master's loudness
    {
        struct Probe : public juce::Timer
        {
            HostEngine e; bool done = false; int stage = 0, waited = 0; juce::var song; double sr = 48000.0; int bs = 256;
            void timerCallback() override
            {
                e.service();
                if (stage == 0) { e.loadProject (song, [this] (juce::String err) { done = true; if (err.isNotEmpty()) std::printf ("load: %s\n", err.toRawUTF8()); }); stage = 1; return; }
                if (stage == 1) { if (! done && ++waited < 1500) return; stage = 2; waited = 0; return; }
                if (stage == 2) { if (++waited < 100) return; stage = 3; }      // let the plug-ins settle
                stopTimer();
                auto db = [] (double v) { return v <= 1.0e-9 ? -999.0 : 20.0 * std::log10 (v); };
                std::printf ("probe: loaded %d, %d tracks\n", (int) done, e.trackCount.load());
                e.transport.locate (0.0); e.transport.play();
                const int secs = 24, blocksPerSec = (int) (sr / bs); std::vector<double> tpk ((size_t) e.trackCount.load(), 0.0); double mpk = 0, sumSq = 0; long n = 0, over = 0;
                for (int s2 = 0; s2 < secs * blocksPerSec; ++s2)
                {
                    std::vector<float> l ((size_t) bs), r ((size_t) bs); float* o[2] = { l.data(), r.data() }; e.processOffline (nullptr, 0, o, 2, bs);
                    for (int i = 0; i < bs; ++i) { const double a = std::max (std::abs (l[(size_t) i]), std::abs (r[(size_t) i])); mpk = std::max (mpk, a); sumSq += (double) l[(size_t) i] * l[(size_t) i]; ++n; if (a > 1.0) ++over; }
                    if (s2 % 8 == 0) { e.service(); auto m = e.meters(); if (auto* ta = m.getProperty ("tracks", {}).getArray()) for (int t = 0; t < ta->size() && t < (int) tpk.size(); ++t) { auto* p = ta->getReference (t).getArray(); if (p) tpk[(size_t) t] = std::max (tpk[(size_t) t], std::max ((double) (*p)[0], (double) (*p)[1])); } }
                }
                for (int t = 0; t < e.trackCount.load(); ++t) std::printf ("track %d \"%s\": fader %+.1f dB, peak after fader %.1f dBFS\n", t + 1, e.trackAt (t)->name.toRawUTF8(), (double) e.trackAt (t)->gainDb.load(), db (tpk[(size_t) t]));
                std::printf ("master fader %+.1f dB: peak %.1f dBFS, loudness (RMS) %.1f dBFS, samples over full scale: %ld of %ld\n", (double) e.snapshot().getProperty ("masterDb", 0.0), db (mpk), db (std::sqrt (sumSq / (double) std::max (1L, n))), over, n);
                juce::MessageManager::getInstance()->stopDispatchLoop();
            }
        };
        Probe pr; pr.e.prepareOffline (sr, bs, 0); pr.song = juce::JSON::parse (juce::File (juce::String::fromUTF8 (probe))); pr.sr = sr; pr.bs = bs; pr.startTimer (20);
        juce::MessageManager::getInstance()->runDispatchLoop();
        pr.stopTimer();
        return 0;
    }

    // ---- a tone on one track comes out at its level, through the strip, the master section and out ----
    {
        HostEngine e; e.prepareOffline (sr, bs, 0);
        e.trackAt (0)->source.store (2);
        auto o = run (e, 20, bs);
        float pk = 0; for (size_t i = 2048; i < o.size(); ++i) pk = std::max (pk, std::abs (o[i]));
        std::printf ("  tone peak %.6f over %zu samples\n", pk, o.size());
        CHECK (std::abs (pk - 0.25f) < 1e-3f, "a 0.25 sine on track 1 leaves the mixer at 0.25");
    }

    // ---- PLUG-IN DELAY COMPENSATION: two tracks, one of them has a plug-in that delays by 100 samples: their impulses still arrive together ----
    {
        HostEngine e; e.prepareOffline (sr, bs, 0);
        e.trackAt (0)->source.store (4); e.trackAt (1)->source.store (4);                       // impulse at sample 0, 0.8 on both sides
        e.addProcessor (0, std::make_unique<LatProc> (2, 2, 100), "delay 100");
        auto o = run (e, 8, bs);
        int hits = 0, at = -1; float v = 0; for (size_t i = 0; i < o.size(); ++i) if (std::abs (o[i]) > 1e-4f) { ++hits; at = (int) i; v = o[i]; }
        CHECK (hits == 1 && at == 100, "with delay compensation the two impulses are ONE impulse, 100 samples after the input");
        CHECK (std::abs (v - 1.6f) < 1e-3f, "and they add: 0.8 + 0.8");
        CHECK (e.pdcTotal.load() == 100, "the total reported latency is 100 samples");
        // without compensation there would have been two: prove the test could see it by turning the track-1 delay off
    }

    // ---- delay compensation across SENDS and RETURNS ----
    {
        HostEngine e; e.prepareOffline (sr, bs, 0);
        e.trackAt (0)->source.store (4); e.trackAt (1)->source.store (4);
        e.addProcessor (0, std::make_unique<LatProc> (2, 2, 100), "track delay");          // track 1: 100 samples
        e.addProcessor (kReturnBase, std::make_unique<LatProc> (2, 2, 50), "return delay");   // FX return 1: 50 samples
        e.setSend (*e.trackAt (0), 0, 0.0f, false); e.setSend (*e.trackAt (1), 0, 0.0f, true);   // both tracks send to return 1, one POST one PRE
        auto o = run (e, 12, bs);
        int hits = 0, at = -1; float v = 0; for (size_t i = 0; i < o.size(); ++i) if (std::abs (o[i]) > 1e-4f) { ++hits; at = (int) i; v = o[i]; }
        CHECK (hits == 1 && at == 150, "dry tracks (100) and sends through a 50-sample return all arrive together at sample 150");
        CHECK (std::abs (v - 3.2f) < 1e-3f, "dry 0.8 + 0.8 and return 0.8 + 0.8 add to 3.2");
        CHECK (e.pdcTotal.load() == 150, "total latency 100 + 50 = 150");
    }
    // ---- master inserts add their latency to everything ----
    {
        HostEngine e; e.prepareOffline (sr, bs, 0);
        e.trackAt (0)->source.store (4);
        e.addProcessor (kMasterChain, std::make_unique<LatProc> (2, 2, 30), "master delay");
        auto o = run (e, 6, bs);
        int at = -1; for (size_t i = 0; i < o.size(); ++i) if (std::abs (o[i]) > 1e-4f) { at = (int) i; break; }
        CHECK (at == 30 && e.pdcTotal.load() == 30, "a 30-sample master insert delays the output by 30");
    }

    // ---- channel rules ----
    {
        HostEngine e; e.prepareOffline (sr, bs, 0);
        e.trackAt (0)->source.store (4);
        e.addProcessor (0, std::make_unique<TestProc> (1, 1, 0, 1.0f), "mono");              // mono in = L + R (0.8 + 0.8), mono out copied to both sides
        auto o = run (e, 4, bs); auto r = run (e, 0, bs);
        CHECK (std::abs (o[0] - 1.6f) < 1e-3f, "a mono-in plug-in is fed L + R and its mono output goes to both sides");
    }
    {
        HostEngine e; e.prepareOffline (sr, bs, 0);
        e.trackAt (0)->source.store (4);
        e.addProcessor (0, std::make_unique<TestProc> (0, 2, 0, 1.0f, 0.1f, true), "instrument");   // no input: its output REPLACES the signal
        auto o = run (e, 4, bs);
        CHECK (std::abs (o[0] - 0.1f) < 1e-4f && std::abs (o[100] - 0.1f) < 1e-4f, "a plug-in with no audio input replaces the signal with its own output");
    }
    {
        HostEngine e; e.prepareOffline (sr, bs, 0);
        e.trackAt (0)->source.store (4);
        e.addProcessor (0, std::make_unique<TestProc> (2, 0, 0), "analyser");                 // no output: the signal passes untouched
        auto o = run (e, 4, bs);
        CHECK (std::abs (o[0] - 0.8f) < 1e-3f, "a plug-in with no audio output leaves the signal as it was");
    }
    // ---- bypass ----
    {
        HostEngine e; e.prepareOffline (sr, bs, 0);
        e.trackAt (0)->source.store (2);                      // a steady 0.25 sine
        e.addProcessor (0, std::make_unique<TestProc> (2, 2, 0, 0.5f), "half");
        auto peakOf = [] (const std::vector<float>& v) { float pk = 0; for (size_t i = v.size() / 2; i < v.size(); ++i) pk = std::max (pk, std::abs (v[i])); return pk; };
        auto a = run (e, 12, bs);
        e.setBypass (0, 0, true);
        auto b = run (e, 12, bs);
        CHECK (std::abs (peakOf (a) - 0.125f) < 2e-3f && std::abs (peakOf (b) - 0.25f) < 2e-3f, "bypass passes the signal around the plug-in at once, with no republish");
    }

    // ---- a plug-in with more channels than the program has room for (a drum machine or sampler with an output for every pad) is never handed a short buffer ----
    {
        struct WideProc : public juce::AudioProcessor
        {
            explicit WideProc (int outs) : juce::AudioProcessor (BusesProperties().withInput ("in", juce::AudioChannelSet::stereo(), true).withOutput ("out", juce::AudioChannelSet::discreteChannels (outs), true)) {}
            const juce::String getName() const override { return "Wide"; }
            void prepareToPlay (double, int) override {}
            void releaseResources() override {}
            void processBlock (juce::AudioBuffer<float>& b, juce::MidiBuffer&) override { ++calls; for (int c = 0; c < b.getNumChannels(); ++c) juce::FloatVectorOperations::fill (b.getWritePointer (c), 0.5f, b.getNumSamples()); }
            double getTailLengthSeconds() const override { return 0; }
            bool acceptsMidi() const override { return false; }
            bool producesMidi() const override { return false; }
            juce::AudioProcessorEditor* createEditor() override { return nullptr; }
            bool hasEditor() const override { return false; }
            int getNumPrograms() override { return 1; }
            int getCurrentProgram() override { return 0; }
            void setCurrentProgram (int) override {}
            const juce::String getProgramName (int) override { return {}; }
            void changeProgramName (int, const juce::String&) override {}
            void getStateInformation (juce::MemoryBlock&) override {}
            void setStateInformation (const void*, int) override {}
            std::atomic<int> calls { 0 };
        };
        auto peakOf = [] (const std::vector<float>& v) { float pk = 0; for (size_t i = v.size() / 2; i < v.size(); ++i) pk = std::max (pk, std::abs (v[i])); return pk; };
        { HostEngine e; e.prepareOffline (sr, bs, 0); e.trackAt (0)->source.store (2);
          auto w = std::make_unique<WideProc> (40); auto* wp = w.get(); e.addProcessor (0, std::move (w), "forty outs");
          auto a1 = run (e, 12, bs);
          CHECK (wp->calls.load() == 0 && std::abs (peakOf (a1) - 0.25f) < 2e-3f, "a plug-in that wants 40 channels is not run at all (the sound passes untouched) rather than being given 16 and writing past the end"); }
        { HostEngine g; g.prepareOffline (sr, bs, 0); g.trackAt (0)->source.store (2);
          auto w = std::make_unique<WideProc> (16); auto* wp = w.get(); g.addProcessor (0, std::move (w), "sixteen outs");
          auto b1 = run (g, 12, bs);
          CHECK (wp->calls.load() > 0 && peakOf (b1) > 0.4f, "while one that fits in 16 channels is run as before"); }
    }

    // ---- BROKEN AUDIO: a plug-in that sends non-numbers is muted; it never silences the rest of the mix ----
    {
        struct BadProc : TestProc
        {
            BadProc() : TestProc (2, 2, 0) {}
            void reset() override { ++resets; }
            void processBlock (juce::AudioBuffer<float>& b, juce::MidiBuffer& m) override { TestProc::processBlock (b, m); if (bad) b.setSample (1, 7, value); }
            std::atomic<bool> bad { false }; float value = std::numeric_limits<float>::quiet_NaN(); int resets = 0;
        };
        for (float value : { std::numeric_limits<float>::quiet_NaN(), std::numeric_limits<float>::infinity(), -1.0e20f })
        {
            HostEngine e; e.prepareOffline (sr, bs, 0); e.trackAt (0)->source.store (2); e.trackAt (1)->source.store (2);
            auto pr = std::make_unique<BadProc>(); auto* p = pr.get(); p->value = value; e.addProcessor (0, std::move (pr), "bad one");
            e.addProcessor (kMasterChain, std::make_unique<TestProc> (2, 2, 0), "after");
            auto good = run (e, 12, bs);
            const bool quiet = e.meters().getProperty ("broken", {}).size() == 0;
            p->bad = true; auto during = run (e, 12, bs);
            const auto listed = e.meters().getProperty ("broken", {});
            auto peakOf = [] (const std::vector<float>& v) { float pk = 0; for (size_t i = v.size() / 2; i < v.size(); ++i) pk = std::max (pk, std::abs (v[i])); return pk; };
            bool finite = true; for (float x : during) if (! std::isfinite (x)) finite = false;
            CHECK (quiet && std::abs (peakOf (good) - 0.5f) < 2e-3f, "two tones play and no device is reported");
            CHECK (finite && std::abs (peakOf (during) - 0.25f) < 2e-3f, "a device sending broken audio is muted and the other track still plays");
            CHECK (listed.size() == 1 && listed[0].getProperty ("name", {}).toString() == "bad one" && (int) listed[0].getProperty ("chain", -1) == 0, "and it is named, with its track");
            CHECK (p->resets == 1, "and reset once, when the fault begins");
            p->bad = false; auto after = run (e, 420, bs);
            CHECK (std::abs (peakOf (after) - 0.5f) < 2e-3f && e.meters().getProperty ("broken", {}).size() == 0, "a device that recovers is heard again and the report clears");
        }
    }

    // ---- NO HANGING NOTES: a release always arrives (after a track switch, while bypassed, and when the queue is full) ----
    {
        struct KeyProbe : public juce::AudioProcessor
        {
            KeyProbe() : juce::AudioProcessor (BusesProperties().withOutput ("out", juce::AudioChannelSet::stereo(), true)) {}
            const juce::String getName() const override { return "KeyProbe"; }
            void prepareToPlay (double, int) override {} void releaseResources() override {}
            void processBlock (juce::AudioBuffer<float>& b, juce::MidiBuffer& m) override
            {
                for (const auto meta : m) { const auto x = meta.getMessage(); if (x.isNoteOn()) down.insert (x.getNoteNumber()); else if (x.isNoteOff()) down.erase (x.getNoteNumber()); else if (x.isController() && x.getControllerNumber() == 64) pedal = x.getControllerValue(); else if (x.isPitchWheel()) bend = x.getPitchWheelValue(); }
                for (int c = 0; c < b.getNumChannels(); ++c) juce::FloatVectorOperations::fill (b.getWritePointer (c), 0.5f, b.getNumSamples());
            }
            double getTailLengthSeconds() const override { return 0; } bool acceptsMidi() const override { return true; } bool producesMidi() const override { return false; }
            juce::AudioProcessorEditor* createEditor() override { return nullptr; } bool hasEditor() const override { return false; }
            int getNumPrograms() override { return 1; } int getCurrentProgram() override { return 0; } void setCurrentProgram (int) override {}
            const juce::String getProgramName (int) override { return {}; } void changeProgramName (int, const juce::String&) override {}
            void getStateInformation (juce::MemoryBlock&) override {} void setStateInformation (const void*, int) override {}
            std::set<int> down; int pedal = 0, bend = 8192;
        };
        using M = juce::MidiMessage;
        auto build = [&] (HostEngine& e, KeyProbe*& a, KeyProbe*& b) { e.prepareOffline (sr, bs, 0); auto pa = std::make_unique<KeyProbe>(); a = pa.get(); e.addProcessor (0, std::move (pa), "a"); auto pb = std::make_unique<KeyProbe>(); b = pb.get(); e.addProcessor (1, std::move (pb), "b"); e.selectTrack (0); run (e, 2, bs); };
        { HostEngine e; KeyProbe *a, *b; build (e, a, b);
          e.handleIncomingMidiMessage (nullptr, M::noteOn (1, 60, (juce::uint8) 100)); e.handleIncomingMidiMessage (nullptr, M::controllerEvent (1, 64, 127)); run (e, 2, bs);
          const bool pressed = a->down.count (60) == 1 && a->pedal == 127 && b->down.empty();
          e.selectTrack (1); e.handleIncomingMidiMessage (nullptr, M::noteOff (1, 60)); e.handleIncomingMidiMessage (nullptr, M::controllerEvent (1, 64, 0)); run (e, 2, bs);
          CHECK (pressed && a->down.empty() && a->pedal == 0, "a key and a pedal let go after another track is selected still reach the track that heard them pressed");
          e.handleIncomingMidiMessage (nullptr, M::noteOn (1, 62, (juce::uint8) 100)); e.handleIncomingMidiMessage (nullptr, M::noteOff (1, 62)); run (e, 2, bs);
          CHECK (a->down.empty() && b->down.empty() && ! a->down.count (62), "and the next key goes only to the track now selected"); }
        { HostEngine e; KeyProbe *a, *b; build (e, a, b);
          e.trackAt (1)->mute.store (true); e.updateSolo();      // (the other probe sounds all the time: it is muted so that only the bypassed one is listened for)
          e.handleIncomingMidiMessage (nullptr, M::noteOn (1, 61, (juce::uint8) 100)); run (e, 2, bs); e.setBypass (0, 0, true); run (e, 8, bs);
          e.handleIncomingMidiMessage (nullptr, M::noteOff (1, 61)); auto out = run (e, 2, bs);
          float loud = 0; for (float v : out) loud = std::max (loud, std::abs (v));
          CHECK (a->down.empty() && loud < 1.0e-6f, "a bypassed instrument hears the key let go, and is still not heard"); }
        { HostEngine e; KeyProbe *a, *b; build (e, a, b);
          e.handleIncomingMidiMessage (nullptr, M::noteOn (1, 63, (juce::uint8) 100)); e.handleIncomingMidiMessage (nullptr, M::controllerEvent (1, 64, 127)); e.handleIncomingMidiMessage (nullptr, M::pitchWheel (1, 12000)); run (e, 2, bs);
          for (int i = 0; i < 3000; ++i) e.handleIncomingMidiMessage (nullptr, M::controllerEvent (1, 1, i % 128));
          e.handleIncomingMidiMessage (nullptr, M::noteOff (1, 63)); e.handleIncomingMidiMessage (nullptr, M::controllerEvent (1, 64, 0)); e.handleIncomingMidiMessage (nullptr, M::pitchWheel (1, 8192));
          const int lost = (int) e.meters().getProperty ("midiLost", 0); run (e, 2, bs);
          CHECK (a->down.empty() && a->pedal == 0 && a->bend == 8192, "with the queue flooded the key and pedal are still released and the pitch wheel still returns to centre");
          CHECK (lost > 1000 && (int) e.meters().getProperty ("midiLost", 0) == 0, "and the skipped messages are counted once, to be said"); }
    }

    // ---- EVERY NOTE ONCE, whatever the buffer: a note on the seam between two pieces of the song is sent by exactly one of them ----
    {
        struct Count : public juce::AudioProcessor
        {
            Count() : juce::AudioProcessor (BusesProperties().withOutput ("out", juce::AudioChannelSet::stereo(), true)) { ons.reserve (4096); }
            const juce::String getName() const override { return "Count"; }
            void prepareToPlay (double, int) override {} void releaseResources() override {}
            void processBlock (juce::AudioBuffer<float>&, juce::MidiBuffer& m) override { for (const auto meta : m) { const auto x = meta.getMessage(); if (x.isNoteOn()) ons.push_back (x.getNoteNumber()); else if (x.isNoteOff()) ++offs; } }
            double getTailLengthSeconds() const override { return 0; } bool acceptsMidi() const override { return true; } bool producesMidi() const override { return false; }
            juce::AudioProcessorEditor* createEditor() override { return nullptr; } bool hasEditor() const override { return false; }
            int getNumPrograms() override { return 1; } int getCurrentProgram() override { return 0; } void setCurrentProgram (int) override {}
            const juce::String getProgramName (int) override { return {}; } void changeProgramName (int, const juce::String&) override {}
            void getStateInformation (juce::MemoryBlock&) override {} void setStateInformation (const void*, int) override {}
            std::vector<int> ons; int offs = 0;
        };
        bool exact = true;
        for (double bpm : { 120.0, 140.0, 128.0, 173.7 }) for (int block : { 16, 32, 48, 64, 375 })
        {
            HostEngine e; e.prepareOffline (48000.0, block, 0); e.transport.setBpm (bpm); run (e, 1, block); auto p = std::make_unique<Count>(); auto* pr = p.get(); e.addProcessor (0, std::move (p), "count");
            auto* c = new juce::DynamicObject(); c->setProperty ("lengthBeats", 16.0); juce::Array<juce::var> notes; for (int i = 0; i < 64; ++i) { auto* n = new juce::DynamicObject(); n->setProperty ("beat", i * 0.25); n->setProperty ("len", 0.25); n->setProperty ("note", 40 + i); n->setProperty ("vel", 100); notes.add (juce::var (n)); } c->setProperty ("notes", notes);
            e.editAddInstance (0, 0.0, 16.0, 0.0, juce::var (c), 0); e.transport.locate (0.0); e.transport.play(); run (e, (int) (15.9 * 60.0 / bpm * 48000.0) / block + 1, block);
            bool once = pr->ons.size() == 64; for (size_t i = 0; once && i < 64; ++i) once = pr->ons[i] == 40 + (int) i;
            if (! once || pr->offs < 63) { exact = false; std::printf ("  %.1f bpm, block %d: %d note-ons, %d note-offs\n", bpm, block, (int) pr->ons.size(), pr->offs); }
        }
        CHECK (exact, "at buffers from 16 samples up, and four tempos, each of 64 back-to-back notes is sent once and in order");
    }

    // ---- NO CLICKS AT SWITCHES: bypass crossfades (a delaying plug-in stays in line), and adding or removing a plug-in is smoothed ----
    {
        auto biggest = [] (const std::vector<float>& v) { double m = 0; for (size_t i = 1; i < v.size(); ++i) m = std::max (m, (double) std::abs (v[i] - v[i - 1])); return m; };
        auto join = [] (std::vector<float>& to, const std::vector<float>& more) { to.insert (to.end(), more.begin(), more.end()); };
        for (int late : { 0, 240, 1500 })
        {
            HostEngine e; e.prepareOffline (sr, bs, 0); e.trackAt (0)->source.store (2); e.addProcessor (0, std::make_unique<LatProc> (2, 2, late, 1.0f), "late");
            run (e, 40, bs); auto before = run (e, 20, bs); const double own = biggest (before);
            std::vector<float> all = before; e.setBypass (0, 0, true); join (all, run (e, 30, bs)); e.setBypass (0, 0, false); join (all, run (e, 40, bs));
            CHECK (biggest (all) < 1.5 * own, "bypassing and un-bypassing a plug-in (with or without delay) makes no jump");
            float worst = 0; for (size_t i = 0; i < before.size() && i + 90 * (size_t) bs < all.size(); ++i) {}
            double r1 = 0, r2 = 0; for (size_t i = all.size() - before.size(); i < all.size(); ++i) r2 += (double) all[i] * all[i]; for (float v : before) r1 += (double) v * v;
            CHECK (std::abs (r2 - r1) < 0.05 * r1, "and the sound comes back at the same level");
            (void) worst;
        }
        {
            HostEngine e; e.prepareOffline (sr, bs, 0); e.trackAt (0)->source.store (2); run (e, 20, bs); auto before = run (e, 20, bs); const double own = biggest (before);
            std::vector<float> all = before; e.addProcessor (0, std::make_unique<TestProc> (2, 2, 0, 0.25f), "quiet"); join (all, run (e, 20, bs)); e.removePlugin (0, 0); join (all, run (e, 20, bs));
            CHECK (biggest (all) < 1.5 * own, "putting a plug-in on a playing track and taking it off makes no jump");
        }
    }

    // ---- PANIC: a hung note can always be shut down ----
    {
        struct PanicProbe : public juce::AudioProcessor
        {
            PanicProbe() : juce::AudioProcessor (BusesProperties().withOutput ("out", juce::AudioChannelSet::stereo(), true)) {}
            const juce::String getName() const override { return "PanicProbe"; }
            void prepareToPlay (double, int) override {}
            void releaseResources() override {}
            void reset() override { ++resets; }
            void processBlock (juce::AudioBuffer<float>& b, juce::MidiBuffer& m) override
            {
                b.clear();
                for (const auto meta : m) { const auto msg = meta.getMessage(); if (msg.isController() && msg.getControllerNumber() == 120) ++soundOff; if (msg.isController() && msg.getControllerNumber() == 123) ++notesOff; }
            }
            double getTailLengthSeconds() const override { return 0; }
            bool acceptsMidi() const override { return true; }
            bool producesMidi() const override { return false; }
            juce::AudioProcessorEditor* createEditor() override { return nullptr; }
            bool hasEditor() const override { return false; }
            int getNumPrograms() override { return 1; }
            int getCurrentProgram() override { return 0; }
            void setCurrentProgram (int) override {}
            const juce::String getProgramName (int) override { return {}; }
            void changeProgramName (int, const juce::String&) override {}
            void getStateInformation (juce::MemoryBlock&) override {}
            void setStateInformation (const void*, int) override {}
            std::atomic<int> resets { 0 }, soundOff { 0 }, notesOff { 0 };
        };
        { HostEngine e; e.prepareOffline (sr, bs, 0);
          auto pr = std::make_unique<PanicProbe>(); auto* p = pr.get(); e.addProcessor (0, std::move (pr), "probe");
          run (e, 6, bs);
          CHECK (p->resets.load() == 0 && p->soundOff.load() == 0, "nothing is sent until Panic is pressed");
          e.panicAll(); run (e, 6, bs);
          CHECK (p->soundOff.load() >= 16 && p->notesOff.load() >= 16, "Panic sends all sound off and all notes off on all 16 channels");
          CHECK (p->resets.load() == 1, "and resets the plug-in once (that ends hung voices and clears tails)"); }
        { HostEngine e; e.prepareOffline (sr, bs, 0);
          auto pr = std::make_unique<PanicProbe>(); auto* p = pr.get(); e.addProcessor (0, std::move (pr), "probe");
          e.transport.play(); run (e, 6, bs); const int before = p->notesOff.load();
          e.transport.stop(); run (e, 6, bs);
          CHECK (p->notesOff.load() >= before + 16, "stopping the transport also tells the plug-in all notes off, on every channel"); }
    }

    // ---- BIT-IDENTICAL whatever the thread count ----
    {
        auto build = [&] (HostEngine& e)
        {
            for (int i = 4; i < 12; ++i) e.addTrack();
            for (int i = 0; i < 12; ++i) { auto* t = e.trackAt (i); t->source.store (i % 2 == 0 ? 2 : 5); t->gainDb.store (-3.0f * (float) (i % 4)); t->pan.store ((float) (i % 5 - 2) * 0.4f); }
            e.addProcessor (3, std::make_unique<LatProc> (2, 2, 64, 0.9f), "d3"); e.addProcessor (7, std::make_unique<LatProc> (2, 2, 200, 0.8f), "d7");
            e.addProcessor (kReturnBase + 1, std::make_unique<LatProc> (2, 2, 33, 0.7f), "r1");
            for (int i = 0; i < 12; ++i) { e.setSend (*e.trackAt (i), i % 3, -6.0f - (float) i, i % 2 == 0); }
            e.addProcessor (kMasterChain, std::make_unique<LatProc> (2, 2, 17, 0.95f), "m");
        };
        HostEngine a, b, c;
        a.prepareOffline (sr, bs, 0); b.prepareOffline (sr, bs, 3); c.prepareOffline (sr, bs, 7);
        a.setParallelThresholdMicros (0.0); b.setParallelThresholdMicros (0.0); c.setParallelThresholdMicros (0.0);   // force the worker path
        build (a); build (b); build (c);
        auto oa = run (a, 120, bs), ob = run (b, 120, bs), oc = run (c, 120, bs);
        CHECK (oa.size() == ob.size() && std::memcmp (oa.data(), ob.data(), oa.size() * sizeof (float)) == 0, "12 tracks, sends, returns and a master insert: 0 and 3 worker threads give the SAME BITS");
        CHECK (std::memcmp (oa.data(), oc.data(), oa.size() * sizeof (float)) == 0, "and 7 worker threads too");
        float pk = 0; for (float x : oa) pk = std::max (pk, std::abs (x));
        CHECK (pk > 0.1f, "(and the mix is not silent)");
        CHECK (a.pdcTotal.load() == b.pdcTotal.load() && a.pdcTotal.load() > 200, "the compensation is the same too (200 + 33-ish + 17)");
    }

    // ================= R3: lock-free hand-offs =================
    // ---- the live MIDI queue: four threads push 50,000 messages each while one pops: every message arrives exactly once, in each thread's own order ----
    {
        MidiFifo q; const int perThread = 50000, producers = 4;
        std::atomic<bool> go { false };
        std::vector<std::thread> ts;
        for (int p = 0; p < producers; ++p)
            ts.emplace_back ([&q, &go, p, perThread] { while (! go.load()) std::this_thread::yield(); for (int i = 0; i < perThread; ++i) { LiveMsg m; m.b[0] = (juce::uint8) p; m.size = 1; m.timeMs = (double) i; while (! q.push (m)) std::this_thread::yield(); } });
        go.store (true);
        std::vector<double> last ((size_t) producers, -1.0); std::vector<int> count ((size_t) producers, 0); bool ordered = true; int got = 0;
        const auto t0 = juce::Time::getMillisecondCounterHiRes();
        while (got < perThread * producers && juce::Time::getMillisecondCounterHiRes() - t0 < 20000)
        {
            LiveMsg m;
            if (q.pop (m)) { ++got; if (m.timeMs <= last[m.b[0]]) ordered = false; last[m.b[0]] = m.timeMs; ++count[m.b[0]]; } else std::this_thread::yield();
        }
        for (auto& t : ts) t.join();
        CHECK (got == perThread * producers && count[0] == perThread && count[1] == perThread && count[2] == perThread && count[3] == perThread, "200,000 messages from 4 threads all arrive, none twice, none lost");
        CHECK (ordered, "and each thread's messages stay in the order it sent them");
        LiveMsg none; CHECK (! q.pop (none), "the queue is empty afterwards");
        MidiFifo tiny; int accepted = 0; LiveMsg x; x.size = 1; for (int i = 0; i < 5000; ++i) if (tiny.push (x)) ++accepted;
        CHECK (accepted == 1024, "a full queue refuses (drops) a message instead of waiting");
    }
    // ---- live notes reach the plug-in at the earliest sample, in order ----
    {
        struct MidiProbe : public juce::AudioProcessor
        {
            MidiProbe() : juce::AudioProcessor (BusesProperties().withOutput ("out", juce::AudioChannelSet::stereo(), true)) {}
            const juce::String getName() const override { return "probe"; }
            void prepareToPlay (double, int) override {} void releaseResources() override {}
            void processBlock (juce::AudioBuffer<float>&, juce::MidiBuffer& m) override { for (const auto meta : m) if (meta.getMessage().isNoteOn()) seen.push_back ({ meta.samplePosition, meta.getMessage().getNoteNumber() }); }
            double getTailLengthSeconds() const override { return 0; } bool acceptsMidi() const override { return true; } bool producesMidi() const override { return false; }
            juce::AudioProcessorEditor* createEditor() override { return nullptr; } bool hasEditor() const override { return false; }
            int getNumPrograms() override { return 1; } int getCurrentProgram() override { return 0; } void setCurrentProgram (int) override {}
            const juce::String getProgramName (int) override { return {}; } void changeProgramName (int, const juce::String&) override {}
            void getStateInformation (juce::MemoryBlock&) override {} void setStateInformation (const void*, int) override {}
            std::vector<std::pair<int, int>> seen;
        };
        HostEngine e; e.prepareOffline (sr, bs, 0);
        auto probe = std::make_unique<MidiProbe>(); auto* pr = probe.get();
        e.addProcessor (0, std::move (probe), "probe");
        e.noteOn (0, 60, 100); e.noteOn (0, 64, 100); e.noteOn (0, 67, 100);          // a chord that arrived before the next block
        run (e, 1, bs);
        CHECK (pr->seen.size() == 3 && pr->seen[0] == std::make_pair (0, 60) && pr->seen[1] == std::make_pair (0, 64) && pr->seen[2] == std::make_pair (0, 67), "a chord pressed before the block arrives at sample 0, in the order it was played");
        pr->seen.clear();
        LiveMsg late; late.b[0] = 0x90; late.b[1] = 72; late.b[2] = 100; late.size = 3; late.timeMs = juce::Time::getMillisecondCounterHiRes() + 100000.0;   // claims to be far in the future
        e.trackAt (0)->liveMidi.push (late);
        run (e, 1, bs);
        CHECK (pr->seen.size() == 1 && pr->seen[0].first == bs - 1 && pr->seen[0].second == 72, "a message stamped in the future is kept to the end of the block, never lost or held to a later block");
    }
    // ---- replacing a pattern while the audio runs: another thread publishes new patterns as fast as it can, nothing breaks ----
    {
        HostEngine e; e.prepareOffline (sr, bs, 0);
        e.transport.setLoop (true, 0.0, 4.0); e.transport.play();
        std::atomic<bool> stop { false }; std::atomic<int> published { 0 };
        std::thread writer ([&] { int i = 0; while (! stop.load()) { juce::var clip (new juce::DynamicObject()); auto* o = clip.getDynamicObject(); o->setProperty ("lengthBeats", 4.0); juce::Array<juce::var> notes; for (int k = 0; k < 1 + (i % 6); ++k) { auto* n = new juce::DynamicObject(); n->setProperty ("beat", k * 0.5); n->setProperty ("len", 0.25); n->setProperty ("note", 50 + k); n->setProperty ("vel", 100); notes.add (juce::var (n)); } o->setProperty ("notes", notes); e.setClip (0, clip); ++i; published.fetch_add (1); std::this_thread::sleep_for (std::chrono::microseconds (200)); } });
        bool finite = true;
        int blocksRun = 0;
        for (int b = 0; (b < 600 || published.load() < 300) && b < 400000; ++b) { auto o = run (e, 1, bs); ++blocksRun; for (float v : o) if (! std::isfinite (v)) finite = false; }
        std::printf ("  %d blocks rendered while %d patterns were published\n", blocksRun, published.load());
        stop.store (true); writer.join();
        CHECK (finite && published.load() > 100, "600 blocks rendered while a second thread published a new pattern over and over: no crash, no bad numbers");
    }

    // ---- UNDO / REDO ----
    {
        // the stack on its own: a drag is one step, a redo is cleared by a new edit, a limit holds
        UndoStack st; double now = 1000.0; st.clock = [&] { return now; };
        int v = 0;
        auto set = [&] (int to) { const int before = v; v = to; st.record ("set", "v", [&v, before] { v = before; }, [&v, to] { v = to; }); };
        set (1); now += 100; set (2); now += 100; set (3);
        CHECK (st.depth() == 1, "edits with the same key close together are one step");
        CHECK (st.undo() && v == 0, "one undo puts the value back to where the drag started");
        CHECK (st.redo() && v == 3, "and redo goes to where the drag ended");
        now += 5000; set (4); CHECK (st.depth() == 2, "a later edit with the same key is a new step");
        CHECK (st.undo() && v == 3 && st.canRedo(), "undo one step");
        set (9); CHECK (! st.canRedo(), "a new edit clears what could be redone");
        st.clear(); CHECK (! st.canUndo() && ! st.undo() && ! st.redo(), "a cleared history has nothing to undo");
        for (int i = 0; i < 400; ++i) { now += 5000; set (i); }
        CHECK (st.depth() <= 300, "the history is bounded");
    }
    {
        // the engine: mixer, clip, tempo, tracks, all through the same edit path as the bridge
        HostEngine e; e.prepareOffline (sr, bs, 0);
        e.undo.clock = [&] { static double t = 0; t += 5000; return t; };       // every edit is its own step unless a test says otherwise
        auto* t0 = e.trackAt (0);
        e.editChainParam (0, "gainDb", -6.0); e.editChainParam (0, "pan", 0.5); e.editChainParam (0, "mute", true);
        CHECK (e.undo.depth() == 3, "three track edits are three steps");
        e.undo.undo(); CHECK (! t0->mute.load() && std::abs (t0->pan.load() - 0.5f) < 1e-6f, "undo un-mutes");
        e.undo.undo(); CHECK (std::abs (t0->pan.load()) < 1e-6f, "undo puts the pan back");
        e.undo.redo(); CHECK (std::abs (t0->pan.load() - 0.5f) < 1e-6f, "redo moves it again");
        e.undo.undo(); e.undo.undo(); CHECK (std::abs (t0->gainDb.load()) < 1e-6f, "undo puts the level back to 0 dB");
        CHECK (! e.undo.canUndo(), "and the history is empty");
        e.editChainParam (0, "gainDb", 0.0); CHECK (! e.undo.canUndo(), "setting a value to what it already is records nothing");
        e.editChainParam (0, "armed", true); CHECK (! e.undo.canUndo(), "arming is a live state, not an edit");
        e.editChainParam (0, "armed", false);

        // a fader drag: many edits in one gesture are one step
        { HostEngine d; d.prepareOffline (sr, bs, 0); double now = 0; d.undo.clock = [&] { return now; };
          for (int i = 1; i <= 30; ++i) { now += 20; d.editChainParam (1, "gainDb", -(double) i); }
          CHECK (d.undo.depth() == 1, "a fader drag is one undo step");
          d.undo.undo(); CHECK (std::abs (d.trackAt (1)->gainDb.load()) < 1e-6f, "one undo returns the fader to where the drag began"); }

        e.editMasterDb (-12.0); e.undo.undo(); CHECK (std::abs (e.master->gainDb.load()) < 1e-6f, "master level undo");
        e.editSend (0, 2, -10.0f, true);
        CHECK (e.trackAt (0)->sendProc[2] != nullptr && e.trackAt (0)->sendPre[2].load(), "a send is on, tapped pre-fader");
        e.undo.undo(); CHECK (e.trackAt (0)->sendProc[2] == nullptr && e.trackAt (0)->sendDb[2].load() < -90.0f, "undo turns the send off again");
        e.undo.redo(); CHECK (e.trackAt (0)->sendProc[2] != nullptr && std::abs (e.trackAt (0)->sendDb[2].load() + 10.0f) < 1e-4f, "redo turns it back on at its level");
        e.undo.undo();

        // clip edits
        juce::DynamicObject* co = new juce::DynamicObject(); co->setProperty ("lengthBeats", 4.0);
        juce::Array<juce::var> notes; { auto* n = new juce::DynamicObject(); n->setProperty ("beat", 0.0); n->setProperty ("len", 1.0); n->setProperty ("note", 60); n->setProperty ("vel", 100); notes.add (juce::var (n)); }
        co->setProperty ("notes", notes); const juce::var coV (co);       // one owner: a var made from the raw pointer twice would free it twice
        e.editSetClip (0, coV); CHECK (e.trackAt (0)->primaryClip().notes.size() == 1, "a clip is set");
        e.undo.undo(); CHECK (e.trackAt (0)->primaryClip().lengthBeats == 0.0, "undo removes it again");
        e.undo.redo(); CHECK (e.trackAt (0)->primaryClip().notes.size() == 1, "redo puts it back");
        e.editClearClip (0); CHECK (e.trackAt (0)->primaryClip().lengthBeats == 0.0, "a clip is cleared");
        e.undo.undo(); CHECK (e.trackAt (0)->primaryClip().notes.size() == 1, "undo of a clear brings the notes back");
        e.undo.undo();

        // tempo
        e.editTempo ("Tempo", "bpm", [&] { e.transport.setBpm (90.0); });
        e.editTempo ("Add tempo change", "", [&] { e.transport.addTempoChange (8.0, 140.0); });
        CHECK (e.transport.tempoEntries().size() == 2, "a tempo change is added");
        e.undo.undo(); CHECK (e.transport.tempoEntries().size() == 1 && std::abs (e.transport.tempoEntries()[0].bpm - 90.0) < 1e-9, "undo takes the tempo change away");
        e.undo.undo(); CHECK (std::abs (e.transport.tempoEntries()[0].bpm - 120.0) < 1e-9, "and the first undo puts the tempo back to 120");
        e.editTempo ("Tempo", "bpm", [&] { e.transport.setBpm (120.0); });
        CHECK (! e.undo.canUndo(), "setting the tempo to what it is records nothing");

        // tracks: add, remove (settings and the pattern come back)
        const int n0 = e.trackCount.load();
        e.editAddTrack(); CHECK (e.trackCount.load() == n0 + 1, "a track is added");
        e.undo.undo(); CHECK (e.trackCount.load() == n0, "undo removes it");
        e.undo.redo(); CHECK (e.trackCount.load() == n0 + 1, "redo adds it");
        auto* last = e.trackAt (n0);
        e.editChainParam (n0, "name", "Bass"); e.editChainParam (n0, "gainDb", -3.0); e.editSend (n0, 1, -8.0f, false); e.editSetClip (n0, coV);
        e.editRemoveTrack(); CHECK (e.trackCount.load() == n0, "the last track is removed");
        e.undo.undo(); last = e.trackAt (n0);
        CHECK (e.trackCount.load() == n0 + 1 && last->name == "Bass" && std::abs (last->gainDb.load() + 3.0f) < 1e-4f, "undo of a remove brings the track back with its name and level");
        CHECK (std::abs (last->sendDb[1].load() + 8.0f) < 1e-4f && last->sendProc[1] != nullptr && last->primaryClip().notes.size() == 1, "and its send and its pattern");
        e.undo.redo(); CHECK (e.trackCount.load() == n0, "redo removes it again");

        // any track can be removed, not only the last: the ones after it move up, and undo puts it back in its place
        e.editAddTrack(); e.editAddTrack(); e.editAddTrack();
        e.editChainParam (n0, "name", "A"); e.editChainParam (n0 + 1, "name", "B"); e.editChainParam (n0 + 2, "name", "C");
        e.editChainParam (n0, "gainDb", -1.0); e.editChainParam (n0 + 1, "gainDb", -2.0); e.editChainParam (n0 + 2, "gainDb", -3.0); e.editSend (n0 + 2, 0, -9.0f, false);
        e.editRemoveTrack (n0 + 1);
        CHECK (e.trackCount.load() == n0 + 2, "a track from the middle is removed");
        CHECK (e.trackAt (n0)->name == "A" && e.trackAt (n0 + 1)->name == "C", "the one after it moves up into its place");
        CHECK (e.trackAt (n0 + 1)->index == n0 + 1 && std::abs (e.trackAt (n0 + 1)->gainDb.load() + 3.0f) < 1e-4f && std::abs (e.trackAt (n0 + 1)->sendDb[0].load() + 9.0f) < 1e-4f, "with its own number, level and send");
        e.undo.undo();
        CHECK (e.trackCount.load() == n0 + 3 && e.trackAt (n0 + 1)->name == "B" && e.trackAt (n0 + 2)->name == "C" && e.trackAt (n0 + 1)->index == n0 + 1, "undo puts it back between A and C");
        CHECK (std::abs (e.trackAt (n0 + 1)->gainDb.load() + 2.0f) < 1e-4f && std::abs (e.trackAt (n0 + 2)->gainDb.load() + 3.0f) < 1e-4f, "with the right levels");
        e.undo.redo(); CHECK (e.trackCount.load() == n0 + 2 && e.trackAt (n0 + 1)->name == "C", "redo removes it again");
        e.editRemoveTrack (n0); CHECK (e.trackCount.load() == n0 + 1 && e.trackAt (n0)->name == "C" && e.trackAt (n0)->index == n0, "the first of them can go too");
        e.editRemoveTrack (n0); CHECK (e.trackCount.load() == n0, "and the last one");
        e.editRemoveTrack (999); CHECK (e.trackCount.load() == n0, "a track number that does not exist removes nothing");

        // a built-in device cannot be rebuilt from a note: the history is dropped rather than made wrong
        e.editChainParam (0, "gainDb", -2.0);
        e.addProcessor (0, std::make_unique<TestProc> (2, 2, 0), "internal");
        e.editRemovePlugin (0, 0);
        CHECK (! e.undo.canUndo(), "removing a built-in device clears the history instead of recording something that cannot be undone");

        // a new project forgets the history
        e.editChainParam (0, "gainDb", -4.0); CHECK (e.undo.canUndo(), "an edit is recorded");
        e.undo.clear(); CHECK (! e.undo.canUndo(), "clearing forgets it");
    }

    // ---- CLIPS AS INSTANCES OF SHARED CONTENT ----
    {
        // what a plug-in on the track would receive: every note-on / note-off with the sample it lands on, turned into beats (120 bpm, 48 kHz = 24000 samples a beat)
        struct Ev { double beat; bool on; int note; };
        auto probeRun = [&] (HostEngine& e, std::vector<Ev>& log, double fromBeat, double toBeat)
        {
            struct Probe : TestProc
            {
                Probe (std::vector<Ev>* l, double b) : TestProc (2, 2, 0), out (l), base (b) {}
                bool acceptsMidi() const override { return true; }
                void processBlock (juce::AudioBuffer<float>&, juce::MidiBuffer& m) override
                { for (const auto meta : m) { auto msg = meta.getMessage(); if (msg.isNoteOn() || msg.isNoteOff()) out->push_back ({ base + (double) (done + meta.samplePosition) / 24000.0, msg.isNoteOn(), msg.getNoteNumber() }); } done += 256; }
                std::vector<Ev>* out; double base; long long done = 0;
            };
            e.addProcessor (0, std::make_unique<Probe> (&log, fromBeat), "probe");
            e.transport.locate (fromBeat); e.transport.play();
            run (e, (int) std::ceil ((toBeat - fromBeat) * 24000.0 / bs) + 2, bs);
        };
        auto makeClip = [] (double len, std::vector<std::array<double, 3>> notes /* beat, len, note */)
        {
            auto* c = new juce::DynamicObject(); c->setProperty ("lengthBeats", len); juce::Array<juce::var> arr;
            for (auto& n : notes) { auto* x = new juce::DynamicObject(); x->setProperty ("beat", n[0]); x->setProperty ("len", n[1]); x->setProperty ("note", (int) n[2]); x->setProperty ("vel", 100); arr.add (juce::var (x)); }
            c->setProperty ("notes", arr); return juce::var (c);
        };
        auto ons = [] (const std::vector<Ev>& l, int note) { std::vector<double> r; for (auto& e : l) if (e.on && e.note == note) r.push_back (e.beat); return r; };
        auto offs = [] (const std::vector<Ev>& l, int note) { std::vector<double> r; for (auto& e : l) if (! e.on && e.note == note) r.push_back (e.beat); return r; };
        auto nearTo = [] (const std::vector<double>& v, std::initializer_list<double> want) { if (v.size() != want.size()) return false; size_t i = 0; for (double w : want) if (std::abs (v[i++] - w) > 0.02) return false; return true; };

        { // an instance starts where it is placed, loops its content to the end of the instance, and stops
            HostEngine e; e.prepareOffline (sr, bs, 0); std::vector<Ev> log;
            e.editAddInstance (0, 4.0, 4.0, 0.0, makeClip (2.0, { { 0, 1, 60 } }), 0);
            probeRun (e, log, 0.0, 12.0);
            CHECK (nearTo (ons (log, 60), { 4.0, 6.0 }), "a 2-beat clip placed at beat 4 for 4 beats plays at 4 and 6, and nothing before or after");
            CHECK (nearTo (offs (log, 60), { 5.0, 7.0 }), "and each note ends after its own length");
        }
        { // an instance that ends in the middle of a note ends the note there
            HostEngine e; e.prepareOffline (sr, bs, 0); std::vector<Ev> log;
            e.editAddInstance (0, 0.0, 2.0, 0.0, makeClip (4.0, { { 1, 3, 62 } }), 0);
            probeRun (e, log, 0.0, 6.0);
            CHECK (nearTo (ons (log, 62), { 1.0 }) && nearTo (offs (log, 62), { 2.0 }), "a note longer than the instance is cut off at the instance end");
        }
        { // the offset starts the content part-way in
            HostEngine e; e.prepareOffline (sr, bs, 0); std::vector<Ev> log;
            e.editAddInstance (0, 0.0, 4.0, 2.0, makeClip (4.0, { { 0, 0.5, 60 }, { 2, 0.5, 64 } }), 0);
            probeRun (e, log, 0.0, 6.0);
            CHECK (nearTo (ons (log, 64), { 0.0 }) && nearTo (ons (log, 60), { 2.0 }), "with an offset of 2 the clip begins at its second half, then wraps");
        }
        { // shared content: editing the content changes every instance; make unique breaks the link
            HostEngine e; e.prepareOffline (sr, bs, 0);
            const int a = e.editAddInstance (0, 0.0, 4.0, 0.0, makeClip (4.0, { { 0, 1, 60 } }), 0);
            const int cid = e.trackAt (0)->arrSnap.load()->insts[0].content->id;
            const int b = e.editAddInstance (0, 8.0, 4.0, 0.0, {}, cid);
            CHECK (a != 0 && b != 0 && b != a, "a second instance of the same clip is placed");
            CHECK (e.trackAt (0)->arrSnap.load()->insts[0].content == e.trackAt (0)->arrSnap.load()->insts[1].content, "and it points at the same content");
            e.editContent (cid, makeClip (4.0, { { 0, 1, 72 } }));
            CHECK (e.trackAt (0)->arrSnap.load()->insts[0].content->notes[0].note == 72 && e.trackAt (0)->arrSnap.load()->insts[1].content->notes[0].note == 72, "editing the clip's content changes both instances");
            e.undo.undo(); CHECK (e.trackAt (0)->arrSnap.load()->insts[0].content->notes[0].note == 60 && e.trackAt (0)->arrSnap.load()->insts[1].content->notes[0].note == 60, "and undo puts both back");
            e.undo.redo();
            e.editMakeUnique (b);
            const int cid2 = e.trackAt (0)->arrSnap.load()->insts[1].content->id;
            CHECK (cid2 != cid, "make unique gives the instance content of its own");
            e.editContent (cid2, makeClip (4.0, { { 0, 1, 50 } }));
            CHECK (e.trackAt (0)->arrSnap.load()->insts[0].content->notes[0].note == 72 && e.trackAt (0)->arrSnap.load()->insts[1].content->notes[0].note == 50, "after that, editing one no longer changes the other");
            const int d = e.editDuplicateInstance (a, 16.0, true, 1);
            CHECK (d != 0 && e.trackAt (1)->arrSnap.load()->insts.size() == 1 && e.trackAt (1)->arrSnap.load()->insts[0].content == e.trackAt (0)->arrSnap.load()->insts[0].content, "a shared copy on another track still points at the same content");
            CHECK (e.editRemoveInstance (d) && e.trackAt (1)->arrSnap.load()->insts.empty(), "an instance can be deleted");
            e.undo.undo(); CHECK (e.trackAt (1)->arrSnap.load()->insts.size() == 1, "and the deletion undone");
        }
        { // split keeps playing the same notes, in two halves
            HostEngine e; e.prepareOffline (sr, bs, 0); std::vector<Ev> log;
            const int a = e.editAddInstance (0, 0.0, 8.0, 0.0, makeClip (4.0, { { 0, 1, 60 } }), 0);
            CHECK (e.editSplitInstance (a, 4.0) != 0, "a clip is split at beat 4");
            CHECK (e.trackAt (0)->arrSnap.load()->insts.size() == 2, "into two instances");
            probeRun (e, log, 0.0, 10.0);
            CHECK (nearTo (ons (log, 60), { 0.0, 4.0 }), "which together play exactly what the one clip played");
        }
        { // starting in the middle of a note starts that note (chase)
            HostEngine e; e.prepareOffline (sr, bs, 0); std::vector<Ev> log;
            e.editAddInstance (0, 0.0, 8.0, 0.0, makeClip (4.0, { { 1, 2, 65 } }), 0);
            probeRun (e, log, 2.0, 6.0);
            CHECK (! ons (log, 65).empty() && std::abs (ons (log, 65)[0] - 2.0) < 0.02 && nearTo (offs (log, 65), { 3.0 }), "playing from beat 2, inside a note that began at 1, sounds it at once and ends it at 3");
        }
        { // recording goes into the clip instance under the first note, so every copy gets it
            HostEngine e; e.prepareOffline (sr, bs, 0);
            e.editAddInstance (0, 0.0, 4.0, 0.0, makeClip (4.0, { { 0, 1, 60 } }), 0);
            const int cid = e.trackAt (0)->arrSnap.load()->insts[0].content->id;
            e.editAddInstance (0, 8.0, 4.0, 0.0, {}, cid);
            auto& t = *e.trackAt (0);
            RecEvent on {}, off {}; on.beat = 9.0; on.b[0] = 0x90; on.b[1] = 64; on.b[2] = 100; off.beat = 9.5; off.b[0] = 0x80; off.b[1] = 64; off.b[2] = 0;
            t.pending.push_back (on); t.pending.push_back (off); e.commitRecording (t);
            CHECK (t.arrSnap.load()->insts.size() == 2 && t.arrSnap.load()->insts[0].content == t.arrSnap.load()->insts[1].content, "a take over the second copy is merged into the shared content");
            CHECK (t.arrSnap.load()->insts[0].content->notes.size() == 2 && std::abs (t.arrSnap.load()->insts[0].content->notes.back().beat - 1.0) < 1e-6, "at its place inside the clip (beat 9 of the song = beat 1 of the clip), so the first copy has it too");
            RecEvent on2 = on, off2 = off; on2.beat = 20.0; off2.beat = 20.5;
            t.pending.push_back (on2); t.pending.push_back (off2); e.commitRecording (t);
            CHECK (t.arrSnap.load()->insts.size() == 3 && std::abs (t.arrSnap.load()->insts.back().start - 20.0) < 1e-6, "a take where there is no clip makes a new clip placed there");
            e.undo.undo(); CHECK (t.arrSnap.load()->insts.size() == 2, "and undo takes it away again");
        }
        { // save and load keep the instances AND the sharing
            HostEngine e; e.prepareOffline (sr, bs, 0);
            e.editAddInstance (0, 0.0, 4.0, 0.0, makeClip (4.0, { { 0, 1, 60 } }), 0);
            const int cid = e.trackAt (0)->arrSnap.load()->insts[0].content->id;
            e.editAddInstance (0, 8.0, 4.0, 1.0, {}, cid); e.editAddInstance (1, 2.0, 4.0, 0.0, {}, cid);
            auto proj = e.saveProject();
            HostEngine f; f.prepareOffline (sr, bs, 0); bool called = false; juce::String err;
            f.loadProject (proj, [&] (juce::String s) { called = true; err = s; });
            CHECK (called && err.isEmpty(), "the project loads");
            const auto& a0 = f.trackAt (0)->arrSnap.load()->insts; const auto& a1 = f.trackAt (1)->arrSnap.load()->insts;
            CHECK (a0.size() == 2 && a1.size() == 1 && std::abs (a0[1].start - 8.0) < 1e-9 && std::abs (a0[1].offset - 1.0) < 1e-9, "all three instances come back with their places and offsets");
            CHECK (a0[0].content == a0[1].content && a0[0].content == a1[0].content, "and the three still share one content, across tracks");
        }
        { // the old single-pattern calls still mean one pattern that loops for ever
            HostEngine e; e.prepareOffline (sr, bs, 0); std::vector<Ev> log;
            e.setClip (0, makeClip (2.0, { { 0, 0.5, 67 } }));
            probeRun (e, log, 0.0, 7.0);
            CHECK (nearTo (ons (log, 67), { 0.0, 2.0, 4.0, 6.0 }), "setClip is one pattern looping from the start");
            CHECK (e.trackAt (0)->primaryClip().notes.size() == 1 && e.trackAt (0)->primaryClip().lengthBeats == 2.0, "and getClip reads it back");
        }
    }

    // ---- TRACK MIXING: sum, solo, delay compensation, sends and thread independence (folders are no longer part of the DAW: every track goes straight to the master) ----
    {
        auto peak = [&] (HostEngine& e, int blocks = 24) { auto o = run (e, blocks, bs); float pk = 0; for (size_t i = 4096; i < o.size(); ++i) pk = std::max (pk, std::abs (o[i])); return pk; };
        auto impulses = [&] (HostEngine& e, int blocks, float& v) { auto o = run (e, blocks, bs); int hits = 0, at = -1; v = 0; for (size_t i = 0; i < o.size(); ++i) if (std::abs (o[i]) > 1e-4f) { ++hits; if (at < 0) at = (int) i; v = o[i]; } return std::make_pair (hits, at); };

        { // every track goes straight to the master and has a fader of its own
            HostEngine e; e.prepareOffline (sr, bs, 0);
            e.trackAt (0)->source.store (2); e.trackAt (1)->source.store (2); e.trackAt (2)->source.store (2);
            CHECK (std::abs (peak (e) - 0.75f) < 3e-3f, "three 0.25 tones sum to 0.75 at the master");
            e.editChainParam (0, "gainDb", -6.0); e.editChainParam (1, "gainDb", -6.0);
            CHECK (std::abs (peak (e) - (0.5f * std::pow (10.0f, -6.0f / 20.0f) + 0.25f)) < 3e-3f, "a track fader (-6 dB) scales that track only");
            e.editChainParam (1, "mute", true); e.editChainParam (0, "mute", true);
            CHECK (std::abs (peak (e) - 0.25f) < 3e-3f, "muting a track silences it and leaves the others");
        }
        { // solo: only the soloed tracks are heard
            HostEngine e; e.prepareOffline (sr, bs, 0);
            e.trackAt (0)->source.store (2); e.trackAt (1)->source.store (2); e.trackAt (2)->source.store (2);
            e.editChainParam (0, "solo", true);
            CHECK (std::abs (peak (e) - 0.25f) < 2e-3f, "soloing one track hears only that track");
            e.editChainParam (1, "solo", true);
            CHECK (std::abs (peak (e) - 0.5f) < 2e-3f, "soloing two hears those two and nothing else");
        }
        { // delay compensation: a plug-in on a track delays that track, and every other path is lined up with it
            HostEngine e; e.prepareOffline (sr, bs, 0);
            e.trackAt (0)->source.store (4); e.trackAt (1)->source.store (4); e.trackAt (2)->source.store (4);
            e.addProcessor (0, std::make_unique<LatProc> (2, 2, 30), "track delay 30");
            float v = 0; auto [hits, at] = impulses (e, 8, v);
            CHECK (hits == 1 && at == 30 && std::abs (v - 2.4f) < 1e-3f, "30 samples on one track are matched on the other tracks: one impulse at 30, 3 x 0.8");
            CHECK (e.pdcTotal.load() == 30, "and the total latency is 30");
        }
        { // sends from different tracks still reach the return together
            HostEngine e; e.prepareOffline (sr, bs, 0);
            e.trackAt (0)->source.store (4); e.trackAt (1)->source.store (4);
            e.addProcessor (0, std::make_unique<LatProc> (2, 2, 40), "track delay 40");
            e.setSend (*e.trackAt (0), 0, 0.0f, false); e.setSend (*e.trackAt (1), 0, 0.0f, false);
            e.addProcessor (kReturnBase, std::make_unique<LatProc> (2, 2, 25), "return delay 25");
            float v = 0; auto [hits, at] = impulses (e, 10, v);
            CHECK (hits == 1 && at == 65, "dry (track 40) and the sends from both tracks (through a 25-sample return) all arrive together at 65");
            CHECK (e.pdcTotal.load() == 65, "total latency 65");
        }
        { // bit-identical whatever the number of threads
            auto build = [&] (HostEngine& e)
            {
                for (int i = 4; i < 12; ++i) e.addTrack();
                for (int i = 0; i < 12; ++i) e.trackAt (i)->source.store (i % 2 ? 5 : 2);     // (noise would differ run to run: its generator is seeded from the clock)
                e.addProcessor (1, std::make_unique<LatProc> (2, 2, 17, 0.7f), "t1");
                e.addProcessor (6, std::make_unique<LatProc> (2, 2, 5, 0.9f), "t6");
                e.setSend (*e.trackAt (1), 0, -6.0f, false); e.setSend (*e.trackAt (9), 0, -6.0f, true);
                e.addProcessor (kReturnBase, std::make_unique<LatProc> (2, 2, 11, 0.8f), "ret");
            };
            HostEngine e0; e0.prepareOffline (sr, bs, 0); build (e0); auto o0 = run (e0, 60, bs, 0);
            HostEngine e3; e3.prepareOffline (sr, bs, 3); e3.setParallelThresholdMicros (0.0); build (e3); auto o3 = run (e3, 60, bs, 0);
            CHECK (o0.size() == o3.size() && std::memcmp (o0.data(), o3.data(), o0.size() * sizeof (float)) == 0, "tracks render exactly the same on one thread and on several");
        }
        { // an older project that had folders: it loads, says so, and its tracks go straight to the master (the folder's fader, mute and inserts are gone)
            HostEngine e; e.prepareOffline (sr, bs, 0);
            e.trackAt (0)->source.store (2); e.trackAt (1)->source.store (2); e.trackAt (2)->source.store (2);
            auto proj = e.saveProject();
            if (auto* po = proj.getDynamicObject())
            {
                juce::Array<juce::var> fa; for (const char* nm : { "Drums", "Bus" }) { auto* f = new juce::DynamicObject(); f->setProperty ("index", fa.size()); f->setProperty ("name", nm); f->setProperty ("parent", -1); f->setProperty ("gainDb", -6.0); f->setProperty ("mute", true); f->setProperty ("slots", juce::Array<juce::var>()); fa.add (juce::var (f)); }
                po->setProperty ("folders", fa);
                if (auto* tv = po->getProperty ("tracks").getArray()) for (int i = 0; i < tv->size(); ++i) if (auto* to = (*tv)[i].getDynamicObject()) to->setProperty ("parent", i < 2 ? 0 : -1);
            }
            HostEngine g; g.prepareOffline (sr, bs, 0); bool called = false; juce::String err;
            g.loadProject (proj, [&] (juce::String s) { called = true; err = s; });
            CHECK (called && err.contains ("2 folder"), "an older project that had folders loads and says how many it had");
            CHECK (std::abs (peak (g) - 0.75f) < 3e-3f, "and its tracks go straight to the master: all three are heard, whatever the folder's fader and mute said");
            auto saved = g.saveProject(); CHECK (! saved.getDynamicObject()->hasProperty ("folders"), "saving it again leaves the folders out");
        }
    }

    // ---- AUDIO OPTIONS (Settings, Audio page): a manual latency adjustment and a thread limit, saved with the program ----
    {
        const auto dir = juce::File::getSpecialLocation (juce::File::tempDirectory).getChildFile ("archie-audio-options-test"); dir.deleteRecursively(); dir.createDirectory();      // a temp folder: never the person's real settings
        HostEngine e; e.prepareOffline (sr, bs, 0); e.settingsDir = dir; e.audioOptionsFile().deleteFile(); e.latencyAdjustMs.store (0.0); e.maxThreadsSetting.store (-1);
        e.recordLatencyOverrideSec = 0.010;
        CHECK (std::abs (e.roundTripLatencySec() - 0.010) < 1e-9, "with no adjustment the latency is exactly the figure the sound card gives");
        auto mk = [] (const char* json) { return juce::JSON::parse (json); };
        CHECK (e.setAudioOptions (mk ("{\"latencyAdjustMs\": 5}")) && std::abs (e.roundTripLatencySec() - 0.015) < 1e-9, "an adjustment of 5 ms is added to it");
        CHECK (e.setAudioOptions (mk ("{\"latencyAdjustMs\": -4}")) && std::abs (e.roundTripLatencySec() - 0.006) < 1e-9, "a negative adjustment is taken from it");
        CHECK (e.setAudioOptions (mk ("{\"latencyAdjustMs\": -500}")) && std::abs (e.latencyAdjustMs.load() + 200.0) < 1e-9 && e.roundTripLatencySec() == 0.0, "an adjustment is held to 200 ms and the latency never goes below nothing");
        CHECK (e.setAudioOptions (mk ("{\"maxThreads\": 99}")) && e.maxThreadsSetting.load() == 7, "the thread limit is held to 7");
        CHECK (e.setAudioOptions (mk ("{\"latencyAdjustMs\": 7.5, \"maxThreads\": 2}")), "both can be set together");
        CHECK (! e.setAudioOptions (mk ("{\"nothing\": 1}")), "a request with neither option is refused");
        HostEngine g; g.settingsDir = dir; g.loadAudioOptions();
        CHECK (std::abs (g.latencyAdjustMs.load() - 7.5) < 1e-9 && g.maxThreadsSetting.load() == 2, "the options are saved and come back when the program next starts");
        HostEngine h; h.settingsDir = dir; h.audioOptionsFile().deleteFile(); h.loadAudioOptions();
        CHECK (h.latencyAdjustMs.load() == 0.0 && h.maxThreadsSetting.load() == -1, "with no saved options it starts with no adjustment and an automatic thread count");
        auto st = e.snapshot(); CHECK (std::abs ((double) st.getProperty ("latencyAdjustMs", 0.0) - 7.5) < 1e-9 && (int) st.getProperty ("maxThreads", -9) == 2, "the Audio page can read both back from the program's state");
        dir.deleteRecursively();
    }

    // ---- CHANNEL INPUT STAGE (Mixer, Input section): gain -24 to +24 dB and polarity ----
    {
        auto samples = [&] (HostEngine& e, int blocks = 24) { return run (e, blocks, bs); };
        auto peakOf = [&] (const std::vector<float>& o) { float pk = 0; for (size_t i = 4096; i < o.size(); ++i) pk = std::max (pk, std::abs (o[i])); return pk; };
        auto tone = [&] (HostEngine& e) { e.trackAt (0)->source.store (2); };
        std::vector<float> base;
        { HostEngine e; e.prepareOffline (sr, bs, 0); tone (e); base = samples (e); }
        CHECK (std::abs (peakOf (base) - 0.25f) < 3e-3f, "a channel playing a tone is the level it always was");
        { HostEngine e; e.prepareOffline (sr, bs, 0); tone (e); e.editChainParam (0, "inGainDb", 0.0); e.editChainParam (0, "inInvert", false);
          auto o = samples (e); CHECK (o.size() == base.size() && std::memcmp (o.data(), base.data(), o.size() * sizeof (float)) == 0, "neutral (0 dB, not inverted) changes nothing at all: the sound is bit for bit the same"); }
        { HostEngine e; e.prepareOffline (sr, bs, 0); tone (e); CHECK (e.editChainParam (0, "inGainDb", 6.0206), "the input gain can be set");
          CHECK (std::abs (peakOf (samples (e)) - 0.5f) < 4e-3f, "+6 dB doubles the level"); }
        { HostEngine e; e.prepareOffline (sr, bs, 0); tone (e); e.editChainParam (0, "inGainDb", -6.0206);
          CHECK (std::abs (peakOf (samples (e)) - 0.125f) < 3e-3f, "-6 dB halves it"); }
        { HostEngine e; e.prepareOffline (sr, bs, 0); tone (e); e.editChainParam (0, "inInvert", true);
          auto o = samples (e); float worst = 0; for (size_t i = 4096; i < o.size() && i < base.size(); ++i) worst = std::max (worst, std::abs (o[i] + base[i]));
          CHECK (worst < 1e-4f, "invert flips the sound exactly (adding it to the original gives silence)"); }
        { HostEngine e; e.prepareOffline (sr, bs, 0); tone (e); e.editChainParam (0, "inGainDb", 99.0); CHECK (e.trackAt (0)->inGainDb.load() == 24.0f, "the gain is held to +24 dB");
          e.editChainParam (0, "inGainDb", -99.0); CHECK (e.trackAt (0)->inGainDb.load() == -24.0f, "and to -24 dB"); }
        { HostEngine e; e.prepareOffline (sr, bs, 0); tone (e); e.undo.clock = [] { static double c = 0; c += 5000; return c; };
          e.editChainParam (0, "inGainDb", -12.0); e.editChainParam (0, "inInvert", true);
          e.undo.undo(); CHECK (! e.trackAt (0)->inInvert.load() && e.trackAt (0)->inGainDb.load() == -12.0f, "undo puts the polarity back");
          e.undo.undo(); CHECK (e.trackAt (0)->inGainDb.load() == 0.0f, "and then the gain");
          e.undo.redo(); CHECK (e.trackAt (0)->inGainDb.load() == -12.0f, "redo does it again"); }
        { HostEngine e; e.prepareOffline (sr, bs, 0); e.editChainParam (0, "inGainDb", 7.5); e.editChainParam (0, "inInvert", true); e.editChainParam (1, "inGainDb", -3.0);
          auto proj = e.saveProject(); HostEngine g; g.prepareOffline (sr, bs, 0); juce::String er; g.loadProject (proj, [&] (juce::String x) { er = x; });
          CHECK (g.trackAt (0)->inGainDb.load() == 7.5f && g.trackAt (0)->inInvert.load() && g.trackAt (1)->inGainDb.load() == -3.0f && ! g.trackAt (1)->inInvert.load(), "the input settings are saved with the project and come back with it");
          juce::var st = g.snapshot(); auto* tr = st.getProperty ("tracks", {}).getArray(); CHECK (tr != nullptr && (double) (*tr)[0].getProperty ("inGainDb", 0.0) == 7.5 && (bool) (*tr)[0].getProperty ("inInvert", false), "and the Mixer can read them from the program's state");
          HostEngine h; h.prepareOffline (sr, bs, 0); h.loadProject (proj, [&] (juce::String) {}); h.loadProject (HostEngine().saveProject(), [&] (juce::String) {});
          CHECK (h.trackAt (0)->inGainDb.load() == 0.0f && ! h.trackAt (0)->inInvert.load(), "loading a song that never used it puts the input back to neutral"); }
        { HostEngine e; e.prepareOffline (sr, bs, 0); CHECK (! e.editChainParam (kReturnBase, "inGainDb", 3.0) && ! e.editChainParam (kReturnBase, "inInvert", true), "an FX return has no input stage");
          const int lane = e.editAutoAddLane (0, 3, -1, -1); CHECK (lane > 0, "a track's input gain can be automated");
          CHECK (e.editAutoAddLane (kReturnBase, 3, -1, -1) == 0, "but not an FX return's"); }
        {   // the automation really drives it: a lane from 0 dB at the start to +12 dB at beat 4
            auto pts3 = [] (std::initializer_list<std::pair<double, double>> v) { juce::Array<juce::var> arr; for (auto& q : v) { auto* o = new juce::DynamicObject(); o->setProperty ("beat", q.first); o->setProperty ("value", q.second); arr.add (juce::var (o)); } return juce::var (arr); };
            HostEngine e; e.prepareOffline (sr, bs, 0); tone (e); const int lane = e.editAutoAddLane (0, 3, -1, -1);
            CHECK (e.editAutoSetPoints (0, lane, pts3 ({ { 0.0, 0.0 }, { 4.0, 12.0 } })), "its points are set");
            e.transport.play(); auto o = run (e, 1000, bs);
            auto pk = [&] (double from, double to) { float m = 0; for (size_t i = (size_t) (from * 48000.0); i < (size_t) (to * 48000.0) && i < o.size(); ++i) m = std::max (m, std::abs (o[i])); return m; };
            CHECK (std::abs (pk (0.1, 0.19) - 0.25f * std::pow (10.0f, 0.9f / 20.0f)) < 0.012f, "just after the start it is only a little above 0 dB");
            CHECK (std::abs (pk (1.0, 1.09) - 0.25f * std::pow (10.0f, 6.0f / 20.0f)) < 0.04f, "half way along (beat 2) it is +6 dB");
            CHECK (std::abs (pk (3.0, 4.0) - 0.25f * std::pow (10.0f, 12.0f / 20.0f)) < 0.2f && std::abs (e.trackAt (0)->inGainDb.load() - 12.0f) < 0.1f, "after the last point it holds +12 dB, and the Mixer's knob shows where the automation has put it");
        }
    }

    // ---- CHANNEL EQ STAGE (Mixer, EQ section): a high-pass and a low-pass filter, 12 dB per octave ----
    {
        auto samples = [&] (HostEngine& e, int blocks = 60) { return run (e, blocks, bs); };
        auto peakOf = [&] (const std::vector<float>& o) { float pk = 0; for (size_t i = 8192; i < o.size(); ++i) pk = std::max (pk, std::abs (o[i])); return pk; };
        auto tone = [&] (HostEngine& e) { e.trackAt (0)->source.store (2); };          // a 440 Hz tone at 0.25
        std::vector<float> base;
        { HostEngine e; e.prepareOffline (sr, bs, 0); tone (e); base = samples (e); }
        { HostEngine e; e.prepareOffline (sr, bs, 0); tone (e); e.editChainParam (0, "eqHpHz", 500.0); e.editChainParam (0, "eqLpHz", 5000.0);
          auto o = samples (e); CHECK (o.size() == base.size() && std::memcmp (o.data(), base.data(), o.size() * sizeof (float)) == 0, "with both filters off, whatever their frequencies, the sound is bit for bit the same"); }
        { HostEngine e; e.prepareOffline (sr, bs, 0); tone (e); e.editChainParam (0, "eqHpOn", true); e.editChainParam (0, "eqHpHz", 440.0);
          CHECK (std::abs (peakOf (samples (e)) - 0.25f * 0.70711f) < 6e-3f, "a high-pass at the tone's own frequency takes 3 dB off it"); }
        { HostEngine e; e.prepareOffline (sr, bs, 0); tone (e); e.editChainParam (0, "eqHpOn", true); e.editChainParam (0, "eqHpHz", 2000.0);
          const float want = 0.25f / std::sqrt (1.0f + std::pow (2000.0f / 440.0f, 4.0f)); CHECK (std::abs (peakOf (samples (e)) - want) < 3e-3f, "a high-pass well above the tone cuts it hard (12 dB per octave)"); }
        { HostEngine e; e.prepareOffline (sr, bs, 0); tone (e); e.editChainParam (0, "eqHpOn", true); e.editChainParam (0, "eqHpHz", 20.0);
          CHECK (std::abs (peakOf (samples (e)) - 0.25f) < 4e-3f, "a high-pass far below the tone leaves it alone"); }
        { HostEngine e; e.prepareOffline (sr, bs, 0); tone (e); e.editChainParam (0, "eqLpOn", true); e.editChainParam (0, "eqLpHz", 1000.0);
          const float want = 0.25f / std::sqrt (1.0f + std::pow (440.0f / 1000.0f, 4.0f)); CHECK (std::abs (peakOf (samples (e)) - want) < 4e-3f, "a low-pass above the tone lets nearly all of it through"); }
        { HostEngine e; e.prepareOffline (sr, bs, 0); tone (e); e.editChainParam (0, "eqLpOn", true); e.editChainParam (0, "eqLpHz", 20000.0);
          CHECK (std::abs (peakOf (samples (e)) - 0.25f) < 4e-3f, "a low-pass at 20 kHz leaves it alone"); }
        { HostEngine e; e.prepareOffline (sr, bs, 0); tone (e); e.editChainParam (0, "eqHpOn", true); e.editChainParam (0, "eqHpHz", 440.0); e.editChainParam (0, "eqLpOn", true); e.editChainParam (0, "eqLpHz", 1000.0);
          const float want = 0.25f * 0.70711f / std::sqrt (1.0f + std::pow (440.0f / 1000.0f, 4.0f)); CHECK (std::abs (peakOf (samples (e)) - want) < 6e-3f, "both together multiply"); }
        { HostEngine e; e.prepareOffline (sr, bs, 0); tone (e); e.editChainParam (0, "eqHpHz", 1.0e6); e.editChainParam (0, "eqLpHz", 5.0);
          CHECK (e.trackAt (0)->eqHpHz.load() == 2000.0f && e.trackAt (0)->eqLpHz.load() == 1000.0f, "the frequencies are held to their ranges (high-pass 20 Hz to 2 kHz, low-pass 1 kHz to 20 kHz)"); }
        {   // switching on in the middle of the sound crossfades: no jump
            HostEngine e; e.prepareOffline (sr, bs, 0); tone (e); e.editChainParam (0, "eqHpHz", 2000.0);
            std::vector<float> a = samples (e, 12); e.editChainParam (0, "eqHpOn", true); std::vector<float> b = samples (e, 12);
            float worst = 0; for (size_t i = 1; i < b.size(); ++i) worst = std::max (worst, std::abs (b[i] - b[i - 1]));
            CHECK (worst < 0.05f, "switching a filter on while the sound is playing does not click");
            e.editChainParam (0, "eqHpOn", false); std::vector<float> c = samples (e, 12); float w2 = 0; for (size_t i = 1; i < c.size(); ++i) w2 = std::max (w2, std::abs (c[i] - c[i - 1]));
            CHECK (w2 < 0.05f, "and neither does switching it off"); }
        { HostEngine e; e.prepareOffline (sr, bs, 0); e.undo.clock = [] { static double c = 0; c += 5000; return c; };
          e.editChainParam (0, "eqHpOn", true); e.editChainParam (0, "eqHpHz", 300.0); e.editChainParam (0, "eqLpOn", true);
          e.undo.undo (); CHECK (! e.trackAt (0)->eqLpOn.load() && e.trackAt (0)->eqHpHz.load() == 300.0f, "undo puts the low-pass switch back");
          e.undo.undo (); CHECK (e.trackAt (0)->eqHpHz.load() == 80.0f && e.trackAt (0)->eqHpOn.load(), "then the high-pass frequency"); e.undo.redo (); CHECK (e.trackAt (0)->eqHpHz.load() == 300.0f, "redo does it again"); }
        { HostEngine e; e.prepareOffline (sr, bs, 0); e.editChainParam (0, "eqHpOn", true); e.editChainParam (0, "eqHpHz", 123.0); e.editChainParam (0, "eqLpOn", true); e.editChainParam (0, "eqLpHz", 7777.0);
          auto proj = e.saveProject(); HostEngine g; g.prepareOffline (sr, bs, 0); g.loadProject (proj, [&] (juce::String) {});
          CHECK (g.trackAt (0)->eqHpOn.load() && g.trackAt (0)->eqLpOn.load() && g.trackAt (0)->eqHpHz.load() == 123.0f && g.trackAt (0)->eqLpHz.load() == 7777.0f && ! g.trackAt (1)->eqHpOn.load(), "the filters are saved with the project and come back with it");
          juce::var st = g.snapshot(); auto* tr = st.getProperty ("tracks", {}).getArray(); CHECK (tr != nullptr && (bool) (*tr)[0].getProperty ("eqHpOn", false) && (double) (*tr)[0].getProperty ("eqLpHz", 0.0) == 7777.0, "and the Mixer can read them from the state");
          HostEngine h; h.prepareOffline (sr, bs, 0); h.loadProject (proj, [&] (juce::String) {}); h.loadProject (HostEngine().saveProject(), [&] (juce::String) {});
          CHECK (! h.trackAt (0)->eqHpOn.load() && h.trackAt (0)->eqHpHz.load() == 80.0f && h.trackAt (0)->eqLpHz.load() == 12000.0f, "a song that never used them loads with both off at their starting frequencies");
          CHECK (! e.editChainParam (kReturnBase, "eqHpOn", true), "an FX return has no EQ stage"); }
    }

    // ---- CHANNEL COMPRESSOR (Mixer, Dynamics section) ----
    {
        auto samples = [&] (HostEngine& e, int blocks = 80) { return run (e, blocks, bs); };
        auto peakFrom = [&] (const std::vector<float>& o, size_t from, size_t to) { float pk = 0; for (size_t i = from; i < to && i < o.size(); ++i) pk = std::max (pk, std::abs (o[i])); return pk; };
        auto steady = [&] (const std::vector<float>& o) { return peakFrom (o, o.size() - 4096, o.size()); };
        auto tone = [&] (HostEngine& e) { e.trackAt (0)->source.store (2); };                  // 440 Hz at 0.25, which is -12.04 dB
        auto set = [&] (HostEngine& e, double thr, double ratio, double rel, bool peak) { e.editChainParam (0, "compThr", thr); e.editChainParam (0, "compRatio", ratio); e.editChainParam (0, "compRel", rel); e.editChainParam (0, "compPeak", peak); e.editChainParam (0, "compOn", true); };
        std::vector<float> base;
        { HostEngine e; e.prepareOffline (sr, bs, 0); tone (e); base = samples (e); }
        { HostEngine e; e.prepareOffline (sr, bs, 0); tone (e); e.editChainParam (0, "compThr", -30.0); e.editChainParam (0, "compRatio", 20.0); e.editChainParam (0, "compMakeup", 12.0);
          auto o = samples (e); CHECK (o.size() == base.size() && std::memcmp (o.data(), base.data(), o.size() * sizeof (float)) == 0, "with the compressor off, whatever its settings, the sound is bit for bit the same"); }
        { HostEngine e; e.prepareOffline (sr, bs, 0); tone (e); set (e, -24.0, 4.0, 0.4, true);
          const float want = 0.25f * std::pow (10.0f, -9.0f / 20.0f); auto o = samples (e); CHECK (std::abs (steady (o) - want) < want * 0.10f, "a tone 12 dB over the threshold at 4:1 is brought down 9 dB (12 dB over becomes 3 dB over)"); }
        { HostEngine e; e.prepareOffline (sr, bs, 0); tone (e); set (e, -6.0, 8.0, 0.4, true);
          CHECK (std::abs (steady (samples (e)) - 0.25f) < 0.005f, "a tone below the threshold is left alone"); }
        { HostEngine e; e.prepareOffline (sr, bs, 0); tone (e); set (e, -24.0, 20.0, 0.4, true);
          const float want = 0.25f * std::pow (10.0f, -(12.0f * (1.0f - 1.0f / 20.0f)) / 20.0f); CHECK (std::abs (steady (samples (e)) - want) < want * 0.15f, "at 20:1 it comes close to a limiter"); }
        { HostEngine e; e.prepareOffline (sr, bs, 0); tone (e); set (e, -24.0, 2.0, 0.4, true);
          const float want = 0.25f * std::pow (10.0f, -6.0f / 20.0f); CHECK (std::abs (steady (samples (e)) - want) < want * 0.10f, "at 2:1 a tone 12 dB over is brought down 6 dB"); }
        { HostEngine e; e.prepareOffline (sr, bs, 0); tone (e); set (e, -24.0, 4.0, 0.4, true); e.editChainParam (0, "compMakeup", 6.0206);
          const float want = 0.25f * std::pow (10.0f, -9.0f / 20.0f) * 2.0f; CHECK (std::abs (steady (samples (e)) - want) < want * 0.10f, "make-up gain is added after the reduction"); }
        { HostEngine e; e.prepareOffline (sr, bs, 0); tone (e); set (e, -24.0, 4.0, 0.4, false);
          const float want = 0.25f * std::pow (10.0f, -(((12.0f - 3.0103f) * 0.75f)) / 20.0f); CHECK (std::abs (steady (samples (e)) - want) < want * 0.12f, "the RMS detector reads a steady tone 3 dB lower than the peak detector does, so it reduces less"); }
        {   // attack: the start of a loud sound gets through before the compressor closes, and FAST closes sooner
            HostEngine slow; slow.prepareOffline (sr, bs, 0); tone (slow); set (slow, -24.0, 20.0, 0.4, true); auto a = samples (slow, 40);
            HostEngine fast; fast.prepareOffline (sr, bs, 0); tone (fast); set (fast, -24.0, 20.0, 0.4, true); fast.editChainParam (0, "compFast", true); auto b = samples (fast, 40);
            const float finalLevel = steady (a);
            CHECK (peakFrom (a, 0, 120) > finalLevel * 1.5f, "the first few milliseconds of a loud sound get through before the compressor closes (5 ms attack)");
            CHECK (peakFrom (b, 0, 120) < peakFrom (a, 0, 120), "and with FAST (0.5 ms) it closes sooner so less gets through"); }
        {   // release: after the loud part ends the level comes back, slowly with a long release and quickly with a short one
            auto recover = [&] (double rel) { HostEngine e; e.prepareOffline (sr, bs, 0); tone (e); set (e, -24.0, 20.0, rel, true); samples (e, 60); e.trackAt (0)->source.store (0); auto o = samples (e, 6); return o; };
            (void) recover;
            HostEngine e1; e1.prepareOffline (sr, bs, 0); tone (e1); set (e1, -24.0, 20.0, 0.05, true); auto a1 = samples (e1, 60); const float g1 = e1.trackAt (0)->compStage.grDb;
            HostEngine e2; e2.prepareOffline (sr, bs, 0); tone (e2); set (e2, -24.0, 20.0, 1.2, true); auto a2 = samples (e2, 60); const float g2 = e2.trackAt (0)->compStage.grDb;
            CHECK (g1 > 8.0f && g2 > 8.0f && std::abs (g1 - g2) < 3.0f, "the gain reduction settles to about the same amount whatever the release"); (void) a1; (void) a2; }
        { HostEngine e; e.prepareOffline (sr, bs, 0); tone (e); set (e, -24.0, 4.0, 0.4, true); samples (e, 60); CHECK (e.trackAt (0)->compGr.load() > 7.0f, "the Mixer's gain reduction meter is fed: it reads the reduction"); }
        { HostEngine e; e.prepareOffline (sr, bs, 0); tone (e); e.editChainParam (0, "compThr", -24.0); e.editChainParam (0, "compRatio", 4.0); e.editChainParam (0, "compRel", 0.4);
          samples (e, 20); e.editChainParam (0, "compOn", true); auto on = samples (e, 20); float w = 0; for (size_t i = 1; i < on.size(); ++i) w = std::max (w, std::abs (on[i] - on[i - 1]));
          e.editChainParam (0, "compOn", false); auto off = samples (e, 40); float w2 = 0; for (size_t i = 1; i < off.size(); ++i) w2 = std::max (w2, std::abs (off[i] - off[i - 1]));
          CHECK (w < 0.05f && w2 < 0.05f, "switching the compressor on or off while the sound is playing does not click"); }
        { HostEngine e; e.prepareOffline (sr, bs, 0); e.editChainParam (0, "compThr", -99.0); e.editChainParam (0, "compRatio", 500.0); e.editChainParam (0, "compRel", 99.0); e.editChainParam (0, "compMakeup", 99.0);
          const auto* t = e.trackAt (0); CHECK (t->compThr.load() == -40.0f && t->compRatio.load() == 20.0f && t->compRel.load() == 1.2f && t->compMakeup.load() == 24.0f, "the settings are held to their ranges");
          e.editChainParam (0, "compThr", 5.0); e.editChainParam (0, "compRatio", 0.1); e.editChainParam (0, "compRel", 0.0); CHECK (t->compThr.load() == 0.0f && t->compRatio.load() == 1.1f && t->compRel.load() == 0.05f, "at both ends"); }
        { HostEngine e; e.prepareOffline (sr, bs, 0); e.undo.clock = [] { static double c = 0; c += 5000; return c; };
          e.editChainParam (0, "compOn", true); e.editChainParam (0, "compRatio", 8.0); e.editChainParam (0, "compFast", true);
          e.undo.undo (); CHECK (! e.trackAt (0)->compFast.load() && e.trackAt (0)->compRatio.load() == 8.0f, "undo puts the attack switch back");
          e.undo.undo (); CHECK (e.trackAt (0)->compRatio.load() == 4.0f && e.trackAt (0)->compOn.load(), "then the ratio"); e.undo.redo (); CHECK (e.trackAt (0)->compRatio.load() == 8.0f, "redo does it again"); }
        { HostEngine e; e.prepareOffline (sr, bs, 0); e.editChainParam (0, "compOn", true); e.editChainParam (0, "compThr", -22.0); e.editChainParam (0, "compRatio", 6.0); e.editChainParam (0, "compRel", 0.7); e.editChainParam (0, "compFast", true); e.editChainParam (0, "compPeak", false); e.editChainParam (0, "compMakeup", 3.0);
          auto proj = e.saveProject(); HostEngine g; g.prepareOffline (sr, bs, 0); g.loadProject (proj, [&] (juce::String) {}); const auto* t = g.trackAt (0);
          CHECK (t->compOn.load() && t->compFast.load() && ! t->compPeak.load() && t->compThr.load() == -22.0f && t->compRatio.load() == 6.0f && t->compRel.load() == 0.7f && t->compMakeup.load() == 3.0f && ! g.trackAt (1)->compOn.load(), "the compressor is saved with the project and comes back with it");
          juce::var st = g.snapshot(); auto* tr = st.getProperty ("tracks", {}).getArray(); CHECK (tr != nullptr && (bool) (*tr)[0].getProperty ("compOn", false) && (double) (*tr)[0].getProperty ("compRatio", 0.0) == 6.0, "and the Mixer can read it from the state");
          HostEngine h; h.prepareOffline (sr, bs, 0); h.loadProject (proj, [&] (juce::String) {}); h.loadProject (HostEngine().saveProject(), [&] (juce::String) {});
          CHECK (! h.trackAt (0)->compOn.load() && h.trackAt (0)->compThr.load() == -18.0f && h.trackAt (0)->compRatio.load() == 4.0f && h.trackAt (0)->compPeak.load(), "a song that never used it loads with it off at its starting settings");
          CHECK (! e.editChainParam (kReturnBase, "compOn", true), "an FX return has no compressor"); }
    }

    // ---- CHANNEL GATE (Mixer, Dynamics section) ----
    {
        auto samples = [&] (HostEngine& e, int blocks = 80) { return run (e, blocks, bs); };
        auto steady = [&] (const std::vector<float>& o) { float pk = 0; for (size_t i = o.size() - 4096; i < o.size(); ++i) pk = std::max (pk, std::abs (o[i])); return pk; };
        auto tone = [&] (HostEngine& e) { e.trackAt (0)->source.store (2); };                  // 440 Hz at 0.25, which is -12.04 dB
        auto gate = [&] (HostEngine& e, double thr, double range, double rel, double hold, bool exp) { e.editChainParam (0, "gateThr", thr); e.editChainParam (0, "gateRange", range); e.editChainParam (0, "gateRel", rel); e.editChainParam (0, "gateHold", hold); e.editChainParam (0, "gateExp", exp); e.editChainParam (0, "gateOn", true); };
        std::vector<float> base;
        { HostEngine e; e.prepareOffline (sr, bs, 0); tone (e); base = samples (e); }
        { HostEngine e; e.prepareOffline (sr, bs, 0); tone (e); e.editChainParam (0, "gateThr", -6.0); e.editChainParam (0, "gateRange", 80.0); e.editChainParam (0, "gateRel", 0.05);
          auto o = samples (e); CHECK (o.size() == base.size() && std::memcmp (o.data(), base.data(), o.size() * sizeof (float)) == 0, "with the gate off, whatever its settings, the sound is bit for bit the same"); }
        { HostEngine e; e.prepareOffline (sr, bs, 0); tone (e); gate (e, -30.0, 40.0, 0.05, 0.0, false);
          CHECK (std::abs (steady (samples (e)) - 0.25f) < 0.004f, "a sound above the threshold goes straight through"); }
        { HostEngine e; e.prepareOffline (sr, bs, 0); tone (e); gate (e, -6.0, 40.0, 0.05, 0.0, false);
          const float want = 0.25f * 0.01f; auto o = samples (e); CHECK (std::abs (steady (o) - want) < want * 0.25f, "a sound below the threshold is turned down by the range (40 dB)"); }
        { HostEngine e; e.prepareOffline (sr, bs, 0); tone (e); gate (e, -6.0, 20.0, 0.05, 0.0, false);
          const float want = 0.25f * 0.1f; CHECK (std::abs (steady (samples (e)) - want) < want * 0.15f, "and by 20 dB when the range is 20 dB"); }
        { HostEngine e; e.prepareOffline (sr, bs, 0); tone (e); gate (e, -6.0, 0.0, 0.05, 0.0, false);
          CHECK (std::abs (steady (samples (e)) - 0.25f) < 0.004f, "a range of 0 dB does nothing"); }
        { HostEngine e; e.prepareOffline (sr, bs, 0); tone (e); gate (e, -6.0, 40.0, 0.05, 0.0, true);
          const float want = 0.25f * std::pow (10.0f, -(6.0f - 0.0f) / 20.0f); CHECK (std::abs (steady (samples (e)) - want) < want * 0.20f, "in EXP mode a sound 6 dB below the threshold is turned down 6 dB, not shut"); }
        {   // hold and release: the sound is open, then falls below the threshold
            auto closedAfter = [&] (double hold, double rel, int blocks) { HostEngine e; e.prepareOffline (sr, bs, 0); tone (e); gate (e, -30.0, 40.0, rel, hold, false); samples (e, 40); const float open = e.trackAt (0)->gateStage.grDb; e.editChainParam (0, "gateThr", -6.0); samples (e, blocks); return std::make_pair (open, e.trackAt (0)->gateStage.grDb); };
            auto noHold = closedAfter (0.0, 0.05, 20); auto longHold = closedAfter (0.3, 0.05, 20); auto longHoldLater = closedAfter (0.3, 0.05, 120);
            CHECK (noHold.first < 0.5f && noHold.second > 30.0f, "with no hold it closes soon after the sound drops below the threshold");
            CHECK (longHold.second < 1.0f, "with a 300 ms hold it is still open 100 ms later");
            CHECK (longHoldLater.second > 30.0f, "and has closed after the hold and the release"); }
        {   // the release time: a long release closes slowly
            HostEngine e; e.prepareOffline (sr, bs, 0); tone (e); gate (e, -30.0, 40.0, 2.0, 0.0, false); samples (e, 40); e.editChainParam (0, "gateThr", -6.0); samples (e, 40);
            CHECK (e.trackAt (0)->gateStage.grDb > 1.0f && e.trackAt (0)->gateStage.grDb < 12.0f, "a long release closes slowly (a fifth of a second in, it is only part way)"); }
        { HostEngine e; e.prepareOffline (sr, bs, 0); tone (e); gate (e, -6.0, 40.0, 0.05, 0.0, false); samples (e, 60); CHECK (e.trackAt (0)->gateGr.load() > 30.0f, "the Mixer's gate meter is fed: it reads the reduction"); }
        { HostEngine e; e.prepareOffline (sr, bs, 0); tone (e); e.editChainParam (0, "gateThr", -6.0); e.editChainParam (0, "gateRange", 40.0); e.editChainParam (0, "gateRel", 0.05); samples (e, 20); e.editChainParam (0, "gateOn", true); auto on = samples (e, 20);
          float w = 0; for (size_t i = 1; i < on.size(); ++i) w = std::max (w, std::abs (on[i] - on[i - 1])); e.editChainParam (0, "gateOn", false); auto off = samples (e, 40); float w2 = 0; for (size_t i = 1; i < off.size(); ++i) w2 = std::max (w2, std::abs (off[i] - off[i - 1]));
          CHECK (w < 0.05f && w2 < 0.05f, "switching the gate on or off while the sound is playing does not click"); }
        { HostEngine e; e.prepareOffline (sr, bs, 0); e.editChainParam (0, "gateThr", -200.0); e.editChainParam (0, "gateRange", 500.0); e.editChainParam (0, "gateRel", 99.0); e.editChainParam (0, "gateHold", 99.0);
          const auto* t = e.trackAt (0); CHECK (t->gateThr.load() == -80.0f && t->gateRange.load() == 80.0f && t->gateRel.load() == 4.0f && t->gateHold.load() == 2.0f, "the settings are held to their ranges");
          e.editChainParam (0, "gateThr", 5.0); e.editChainParam (0, "gateRange", -3.0); e.editChainParam (0, "gateRel", 0.0); e.editChainParam (0, "gateHold", -1.0); CHECK (t->gateThr.load() == 0.0f && t->gateRange.load() == 0.0f && t->gateRel.load() == 0.05f && t->gateHold.load() == 0.0f, "at both ends"); }
        { HostEngine e; e.prepareOffline (sr, bs, 0); e.undo.clock = [] { static double c = 0; c += 5000; return c; };
          e.editChainParam (0, "gateOn", true); e.editChainParam (0, "gateRange", 60.0); e.editChainParam (0, "gateExp", true);
          e.undo.undo (); CHECK (! e.trackAt (0)->gateExp.load() && e.trackAt (0)->gateRange.load() == 60.0f, "undo puts the mode back");
          e.undo.undo (); CHECK (e.trackAt (0)->gateRange.load() == 40.0f && e.trackAt (0)->gateOn.load(), "then the range"); e.undo.redo (); CHECK (e.trackAt (0)->gateRange.load() == 60.0f, "redo does it again"); }
        { HostEngine e; e.prepareOffline (sr, bs, 0); e.editChainParam (0, "gateOn", true); e.editChainParam (0, "gateExp", true); e.editChainParam (0, "gateFast", true); e.editChainParam (0, "gateThr", -33.0); e.editChainParam (0, "gateRange", 55.0); e.editChainParam (0, "gateRel", 0.9); e.editChainParam (0, "gateHold", 0.4);
          auto proj = e.saveProject(); HostEngine g; g.prepareOffline (sr, bs, 0); g.loadProject (proj, [&] (juce::String) {}); const auto* t = g.trackAt (0);
          CHECK (t->gateOn.load() && t->gateExp.load() && t->gateFast.load() && t->gateThr.load() == -33.0f && t->gateRange.load() == 55.0f && t->gateRel.load() == 0.9f && t->gateHold.load() == 0.4f && ! g.trackAt (1)->gateOn.load(), "the gate is saved with the project and comes back with it");
          juce::var st = g.snapshot(); auto* tr = st.getProperty ("tracks", {}).getArray(); CHECK (tr != nullptr && (bool) (*tr)[0].getProperty ("gateOn", false) && (double) (*tr)[0].getProperty ("gateRange", 0.0) == 55.0, "and the Mixer can read it from the state");
          HostEngine h; h.prepareOffline (sr, bs, 0); h.loadProject (proj, [&] (juce::String) {}); h.loadProject (HostEngine().saveProject(), [&] (juce::String) {});
          CHECK (! h.trackAt (0)->gateOn.load() && h.trackAt (0)->gateThr.load() == -45.0f && h.trackAt (0)->gateRange.load() == 40.0f && ! h.trackAt (0)->gateExp.load(), "a song that never used it loads with it off at its starting settings");
          CHECK (! e.editChainParam (kReturnBase, "gateOn", true), "an FX return has no gate"); }
    }

    // ---- CHANNEL WIDTH (Mixer, Width): mid/side, 0 mono, 1 unchanged, 2 twice as wide ----
    {
        auto buf2 = [&] (float l, float r, int n) { juce::AudioBuffer<float> b (2, n); for (int i = 0; i < n; ++i) { b.setSample (0, i, l); b.setSample (1, i, r); } return b; };
        {   // the stage on its own
            WidthStage ws; auto b = buf2 (0.4f, 0.0f, 4800); ws.process (b, 4800, 1.0f, 48000.0);
            CHECK (b.getSample (0, 4799) == 0.4f && b.getSample (1, 4799) == 0.0f && b.getSample (0, 0) == 0.4f, "a width of 1 leaves the sound exactly as it is");
            WidthStage w0; { auto d = buf2 (0.4f, 0.0f, 4800); w0.process (d, 4800, 0.0f, 48000.0); } auto b0 = buf2 (0.4f, 0.0f, 4800); w0.process (b0, 4800, 0.0f, 48000.0);
            CHECK (std::abs (b0.getSample (0, 4799) - 0.2f) < 1e-5f && std::abs (b0.getSample (1, 4799) - 0.2f) < 1e-5f, "a width of 0 makes it mono: left and right both become the middle");
            WidthStage w2; { auto d = buf2 (0.4f, 0.0f, 4800); w2.process (d, 4800, 2.0f, 48000.0); } auto b2 = buf2 (0.4f, 0.0f, 4800); w2.process (b2, 4800, 2.0f, 48000.0);
            CHECK (std::abs (b2.getSample (0, 4799) - 0.6f) < 1e-5f && std::abs (b2.getSample (1, 4799) + 0.2f) < 1e-5f, "a width of 2 doubles the side: left 0.4 and right 0 become 0.6 and -0.2");
            WidthStage w3; auto b3 = buf2 (0.3f, 0.3f, 9600); { auto d = buf2 (0.3f, 0.3f, 4800); w3.process (d, 4800, 0.0f, 48000.0); } w3.process (b3, 9600, 2.0f, 48000.0); bool same = true; for (int i = 0; i < 9600; ++i) same = same && b3.getSample (0, i) == 0.3f && b3.getSample (1, i) == 0.3f;
            CHECK (same, "a mono sound is never changed, whatever the width");
            WidthStage w4; auto b4 = buf2 (0.4f, 0.0f, 4800); w4.process (b4, 4800, 0.0f, 48000.0); float step = 0, prev = b4.getSample (0, 0); for (int i = 1; i < 4800; ++i) { step = std::max (step, std::abs (b4.getSample (0, i) - prev)); prev = b4.getSample (0, i); }
            CHECK (step < 0.001f && std::abs (b4.getSample (0, 0) - 0.4f) < 0.001f && std::abs (b4.getSample (0, 4799) - 0.2f) < 0.005f, "moving the width is smoothed (about 20 ms), with no jump");
            WidthStage w5; { auto d = buf2 (0.4f, 0.1f, 4800); w5.process (d, 4800, 7.0f, 48000.0); } auto b5 = buf2 (0.4f, 0.1f, 4800); w5.process (b5, 4800, 7.0f, 48000.0); CHECK (std::abs (b5.getSample (0, 4799) - (0.25f + 0.15f * 2.0f)) < 1e-4f, "a width above 2 is held to 2"); }
        {   // through a channel: a stereo sound hard left (left 0.4, right 0) on track 0
            auto f = juce::File::getSpecialLocation (juce::File::tempDirectory).getChildFile ("archie-hardleft.wav"); f.deleteFile();
            { juce::WavAudioFormat wav; auto out = f.createOutputStream(); std::unique_ptr<juce::AudioFormatWriter> w (wav.createWriterFor (out.get(), 48000.0, 2, 32, {}, 0)); if (w) out.release(); juce::AudioBuffer<float> b (2, 48000 * 4); for (int i = 0; i < b.getNumSamples(); ++i) { b.setSample (0, i, 0.4f); b.setSample (1, i, 0.0f); } w->writeFromAudioSampleBuffer (b, 0, b.getNumSamples()); w.reset(); }
            auto both = [&] (double width) { HostEngine e; e.prepareOffline (sr, bs, 0); e.transport.play(); juce::String err; e.editImportAudio (0, f, 0.0, err); e.editChainParam (0, "width", width);
                std::vector<float> l, r; std::vector<float> lb ((size_t) bs), rb ((size_t) bs); float* o[2] = { lb.data(), rb.data() }; for (int b = 0; b < 400; ++b) { e.processOffline (nullptr, 0, o, 2, bs); l = lb; r = rb; } return std::make_pair (l[100], r[100]); };
            auto w1 = both (1.0), w0 = both (0.0), w2 = both (2.0);
            CHECK (std::abs (w1.first - 0.4f) < 1e-4f && std::abs (w1.second) < 1e-4f, "at the default width the hard-left sound comes out hard left");
            CHECK (std::abs (w0.first - 0.2f) < 1e-3f && std::abs (w0.second - 0.2f) < 1e-3f, "at width 0 it comes out in the middle at half the level on each side");
            CHECK (std::abs (w2.first - 0.6f) < 1e-3f && std::abs (w2.second + 0.2f) < 1e-3f, "at width 2 the sides are doubled"); f.deleteFile(); }
        { HostEngine e; e.prepareOffline (sr, bs, 0); e.trackAt (0)->source.store (2); std::vector<float> a = run (e, 40, bs); HostEngine g; g.prepareOffline (sr, bs, 0); g.trackAt (0)->source.store (2); g.editChainParam (0, "width", 0.3); std::vector<float> b = run (g, 40, bs);
          CHECK (a.size() == b.size() && std::memcmp (a.data(), b.data(), a.size() * sizeof (float)) == 0, "a tone that is the same in both channels is untouched by any width, bit for bit"); }
        { HostEngine e; e.prepareOffline (sr, bs, 0); e.undo.clock = [] { static double c = 0; c += 5000; return c; }; CHECK (e.trackAt (0)->width.load() == 1.0f, "a channel starts at width 1");
          e.editChainParam (0, "width", 99.0); CHECK (e.trackAt (0)->width.load() == 2.0f, "the width is held to 2"); e.editChainParam (0, "width", -5.0); CHECK (e.trackAt (0)->width.load() == 0.0f, "and to 0"); e.editChainParam (0, "width", 0.5);
          e.undo.undo (); CHECK (e.trackAt (0)->width.load() == 0.0f, "undo puts it back"); e.undo.redo (); CHECK (e.trackAt (0)->width.load() == 0.5f, "redo does it again"); }
        { HostEngine e; e.prepareOffline (sr, bs, 0); e.editChainParam (0, "width", 0.25); e.editChainParam (1, "width", 1.75);
          auto proj = e.saveProject(); HostEngine g; g.prepareOffline (sr, bs, 0); g.loadProject (proj, [&] (juce::String) {});
          CHECK (g.trackAt (0)->width.load() == 0.25f && g.trackAt (1)->width.load() == 1.75f && g.trackAt (2)->width.load() == 1.0f, "the widths are saved with the project and come back with it");
          juce::var st = g.snapshot(); auto* tr = st.getProperty ("tracks", {}).getArray(); CHECK (tr != nullptr && (double) (*tr)[0].getProperty ("width", 0.0) == 0.25, "and the Mixer can read them from the state");
          HostEngine h; h.prepareOffline (sr, bs, 0); h.loadProject (proj, [&] (juce::String) {}); h.loadProject (HostEngine().saveProject(), [&] (juce::String) {});
          CHECK (h.trackAt (0)->width.load() == 1.0f, "a song that never used it loads at width 1");
          CHECK (! e.editChainParam (kReturnBase, "width", 0.5), "an FX return has no width control"); }
    }

    // ---- MASTER COMPRESSOR (Mixer, Master Section): across the whole mix, after the master inserts, before the master fader ----
    {
        auto samples = [&] (HostEngine& e, int blocks = 80) { return run (e, blocks, bs); };
        auto peakFrom = [&] (const std::vector<float>& o, size_t from, size_t to) { float pk = 0; for (size_t i = from; i < to && i < o.size(); ++i) pk = std::max (pk, std::abs (o[i])); return pk; };
        auto steady = [&] (const std::vector<float>& o) { return peakFrom (o, o.size() - 4096, o.size()); };
        auto tone = [&] (HostEngine& e) { e.trackAt (0)->source.store (2); };                  // 440 Hz at 0.25 at the master
        auto set = [&] (HostEngine& e, double thr, double ratio, double rel, bool peak) { e.editMasterCompParam ("mcThr", thr); e.editMasterCompParam ("mcRatio", ratio); e.editMasterCompParam ("mcRel", rel); e.editMasterCompParam ("mcPeak", peak); e.editMasterCompParam ("mcOn", true); };
        std::vector<float> base;
        { HostEngine e; e.prepareOffline (sr, bs, 0); tone (e); base = samples (e); }
        { HostEngine e; e.prepareOffline (sr, bs, 0); tone (e); e.editMasterCompParam ("mcThr", -30.0); e.editMasterCompParam ("mcRatio", 20.0); e.editMasterCompParam ("mcMakeup", 12.0);
          auto o = samples (e); CHECK (o.size() == base.size() && std::memcmp (o.data(), base.data(), o.size() * sizeof (float)) == 0, "with the master compressor off, whatever its settings, the sound is bit for bit the same"); }
        { HostEngine e; e.prepareOffline (sr, bs, 0); tone (e); set (e, -24.0, 4.0, 0.4, true);
          const float want = 0.25f * std::pow (10.0f, -9.0f / 20.0f); CHECK (std::abs (steady (samples (e)) - want) < want * 0.10f, "the mix 12 dB over the threshold at 4:1 is brought down 9 dB"); }
        { HostEngine e; e.prepareOffline (sr, bs, 0); tone (e); set (e, -6.0, 8.0, 0.4, true); CHECK (std::abs (steady (samples (e)) - 0.25f) < 0.005f, "a mix below the threshold is left alone"); }
        { HostEngine e; e.prepareOffline (sr, bs, 0); tone (e); set (e, -24.0, 4.0, 0.4, false);
          const float want = 0.25f * std::pow (10.0f, -(((12.0f - 3.0103f) * 0.75f)) / 20.0f); CHECK (std::abs (steady (samples (e)) - want) < want * 0.12f, "the RMS detector (the master's default) reduces less than the peak detector on a steady tone"); }
        { HostEngine e; e.prepareOffline (sr, bs, 0); tone (e); set (e, -24.0, 4.0, 0.4, true); e.editMasterCompParam ("mcMakeup", 6.0206);
          const float want = 0.25f * std::pow (10.0f, -9.0f / 20.0f) * 2.0f; CHECK (std::abs (steady (samples (e)) - want) < want * 0.10f, "make-up gain is added after the reduction"); }
        {   // the attack: ten milliseconds in, a 0.1 ms attack has closed far more than a 30 ms one
            HostEngine slow; slow.prepareOffline (sr, bs, 0); tone (slow); set (slow, -24.0, 20.0, 0.4, true); slow.editMasterCompParam ("mcAtk", 0.03); samples (slow, 2); const float gs = slow.master->comp.grDb;
            HostEngine fast; fast.prepareOffline (sr, bs, 0); tone (fast); set (fast, -24.0, 20.0, 0.4, true); fast.editMasterCompParam ("mcAtk", 0.0001); samples (fast, 2); const float gf = fast.master->comp.grDb;
            CHECK (gf > 10.0f && gs > 1.0f && gs < 6.0f && gf > gs * 2.0f, "ten milliseconds in, a 0.1 ms attack has closed far more than a 30 ms attack"); }
        { HostEngine e; e.prepareOffline (sr, bs, 0); tone (e); set (e, -24.0, 4.0, 0.4, true); samples (e, 60); CHECK (e.master->cGr.load() > 7.0f, "the Master Section's reduction meter is fed"); }
        {   // it sits after the master inserts's place and before the fader: the master level is applied after the compression
            HostEngine e; e.prepareOffline (sr, bs, 0); tone (e); set (e, -24.0, 4.0, 0.4, true); e.editMasterDb (-6.0206);
            const float want = 0.25f * std::pow (10.0f, -9.0f / 20.0f) * 0.5f; CHECK (std::abs (steady (samples (e)) - want) < want * 0.10f, "the master fader comes after it (the compression does not depend on the master level)"); }
        { HostEngine e; e.prepareOffline (sr, bs, 0); tone (e); e.editMasterCompParam ("mcThr", -24.0); e.editMasterCompParam ("mcRatio", 4.0); e.editMasterCompParam ("mcRel", 0.4); samples (e, 20); e.editMasterCompParam ("mcOn", true); auto on = samples (e, 20);
          float w = 0; for (size_t i = 1; i < on.size(); ++i) w = std::max (w, std::abs (on[i] - on[i - 1])); e.editMasterCompParam ("mcOn", false); auto off = samples (e, 40); float w2 = 0; for (size_t i = 1; i < off.size(); ++i) w2 = std::max (w2, std::abs (off[i] - off[i - 1]));
          CHECK (w < 0.05f && w2 < 0.05f, "switching it on or off while the mix is playing does not click"); }
        { HostEngine e; e.prepareOffline (sr, bs, 0); e.editMasterCompParam ("mcThr", -99.0); e.editMasterCompParam ("mcRatio", 500.0); e.editMasterCompParam ("mcAtk", 5.0); e.editMasterCompParam ("mcRel", 99.0); e.editMasterCompParam ("mcMakeup", 99.0);
          CHECK (e.master->cThr.load() == -40.0f && e.master->cRatio.load() == 20.0f && e.master->cAtk.load() == 0.03f && e.master->cRel.load() == 1.2f && e.master->cMake.load() == 24.0f, "the settings are held to their ranges");
          e.editMasterCompParam ("mcAtk", 0.0); CHECK (e.master->cAtk.load() == 0.0001f, "and the attack to 0.1 ms at the other end"); CHECK (! e.editMasterCompParam ("nonsense", 1.0), "an unknown setting is refused"); }
        { HostEngine e; e.prepareOffline (sr, bs, 0); e.undo.clock = [] { static double c = 0; c += 5000; return c; };
          CHECK (! e.master->cOn.load() && e.master->cThr.load() == -12.0f && e.master->cRatio.load() == 2.0f && ! e.master->cPeak.load(), "it starts off, at -12 dB, 2:1, with the RMS detector");
          e.editMasterCompParam ("mcOn", true); e.editMasterCompParam ("mcRatio", 6.0); e.editMasterCompParam ("mcPeak", true);
          e.undo.undo (); CHECK (! e.master->cPeak.load() && e.master->cRatio.load() == 6.0f, "undo puts the detector back");
          e.undo.undo (); CHECK (e.master->cRatio.load() == 2.0f && e.master->cOn.load(), "then the ratio"); e.undo.redo (); CHECK (e.master->cRatio.load() == 6.0f, "redo does it again"); }
        { HostEngine e; e.prepareOffline (sr, bs, 0); e.editMasterCompParam ("mcOn", true); e.editMasterCompParam ("mcThr", -20.0); e.editMasterCompParam ("mcRatio", 3.0); e.editMasterCompParam ("mcAtk", 0.02); e.editMasterCompParam ("mcRel", 0.6); e.editMasterCompParam ("mcPeak", true); e.editMasterCompParam ("mcMakeup", 2.0);
          auto proj = e.saveProject(); HostEngine g; g.prepareOffline (sr, bs, 0); g.loadProject (proj, [&] (juce::String) {}); const auto* m = g.master;
          CHECK (m->cOn.load() && m->cPeak.load() && m->cThr.load() == -20.0f && m->cRatio.load() == 3.0f && m->cAtk.load() == 0.02f && m->cRel.load() == 0.6f && m->cMake.load() == 2.0f, "the master compressor is saved with the project and comes back with it");
          juce::var st = g.snapshot(); const auto mc = st.getProperty ("masterComp", {}); CHECK ((bool) mc.getProperty ("mcOn", false) && (double) mc.getProperty ("mcRatio", 0.0) == 3.0, "and the Mixer can read it from the state");
          HostEngine h; h.prepareOffline (sr, bs, 0); h.loadProject (proj, [&] (juce::String) {}); { HostEngine d; d.prepareOffline (sr, bs, 0); h.loadProject (d.saveProject(), [&] (juce::String) {}); }
          CHECK (! h.master->cOn.load() && h.master->cThr.load() == -12.0f && h.master->cRatio.load() == 2.0f, "a song that never used it loads with it off at its starting settings");
          HostEngine k; k.prepareOffline (sr, bs, 0); k.editMasterCompParam ("mcOn", true); k.editMasterCompParam ("mcRatio", 9.0); k.loadProject (HostEngine().saveProject(), [&] (juce::String) {});
          CHECK (! k.master->cOn.load() && k.master->cThr.load() == -12.0f && k.master->cRatio.load() == 2.0f, "and so does a song saved by something with no master section at all"); }
    }

    // ---- CONTROL ROOM (Mixer, Master Section): level and Dim on what the sound card plays, never on a bounce or an export ----
    {
        auto tone = [&] (HostEngine& e) { e.trackAt (0)->source.store (2); };                 // 440 Hz at 0.25 at the master
        auto live = [&] (HostEngine& e, int nOut, int blocks) { std::vector<std::vector<float>> ch ((size_t) nOut, std::vector<float> ((size_t) bs)); std::vector<float*> o; for (auto& v : ch) o.push_back (v.data()); std::vector<std::vector<float>> all ((size_t) nOut);
            for (int b = 0; b < blocks; ++b) { e.processLive (nullptr, 0, o.data(), nOut, bs); for (int c = 0; c < nOut; ++c) all[(size_t) c].insert (all[(size_t) c].end(), ch[(size_t) c].begin(), ch[(size_t) c].end()); } return all; };
        auto steady = [&] (const std::vector<float>& v) { float pk = 0; for (size_t i = v.size() - 2048; i < v.size(); ++i) pk = std::max (pk, std::abs (v[i])); return pk; };
        std::vector<float> offlineBase; { HostEngine e; e.prepareOffline (sr, bs, 0); tone (e); offlineBase = run (e, 40, bs); }
        { HostEngine e; e.prepareOffline (sr, bs, 0); tone (e); auto a = live (e, 2, 40); CHECK (a[0].size() == offlineBase.size() && std::memcmp (a[0].data(), offlineBase.data(), a[0].size() * sizeof (float)) == 0, "at the default (0 dB, no Dim, main outputs) what the sound card gets is bit for bit what it always was"); }
        { HostEngine e; e.prepareOffline (sr, bs, 0); tone (e); e.editControlRoomParam ("crDb", -6.0206); auto a = live (e, 2, 40); CHECK (std::abs (steady (a[0]) - 0.125f) < 0.003f && std::abs (steady (a[1]) - 0.125f) < 0.003f, "the control room level turns what the sound card plays down (-6 dB halves it)"); }
        { HostEngine e; e.prepareOffline (sr, bs, 0); tone (e); e.editControlRoomParam ("crDim", true); auto a = live (e, 2, 40); CHECK (std::abs (steady (a[0]) - 0.025f) < 0.001f, "Dim turns it down a further 20 dB"); }
        { HostEngine e; e.prepareOffline (sr, bs, 0); tone (e); e.editControlRoomParam ("crDb", -6.0206); e.editControlRoomParam ("crDim", true); auto a = live (e, 2, 40); CHECK (std::abs (steady (a[0]) - 0.0125f) < 0.0007f, "the level and Dim add up (-26 dB)"); }
        { HostEngine e; e.prepareOffline (sr, bs, 0); tone (e); e.editControlRoomParam ("crDb", -90.0); e.editControlRoomParam ("crDim", true); auto a = live (e, 2, 40); CHECK (steady (a[0]) < 1e-4f, "at the bottom of the level it is silent"); }
        { HostEngine e; e.prepareOffline (sr, bs, 0); tone (e); e.editControlRoomParam ("crDb", 6.0206); auto a = live (e, 2, 40); CHECK (std::abs (steady (a[0]) - 0.5f) < 0.01f, "and +6 dB doubles it"); }
        { HostEngine e; e.prepareOffline (sr, bs, 0); tone (e); e.editControlRoomParam ("crDb", -20.0); e.editControlRoomParam ("crDim", true); e.editControlRoomParam ("crPair", 3); auto o = run (e, 40, bs);
          CHECK (o.size() == offlineBase.size() && std::memcmp (o.data(), offlineBase.data(), o.size() * sizeof (float)) == 0, "a bounce or an export never hears the control room: offline the sound is bit for bit the same whatever it is set to"); }
        {   // its own output pair: outputs 3 and 4 carry the monitoring feed, outputs 1 and 2 carry the master unchanged
            HostEngine e; e.prepareOffline (sr, bs, 0); tone (e); e.editControlRoomParam ("crDb", -6.0206); e.editControlRoomParam ("crPair", 1); auto a = live (e, 4, 40);
            CHECK (std::abs (steady (a[0]) - 0.25f) < 0.004f && std::abs (steady (a[1]) - 0.25f) < 0.004f, "with its own output pair, the main outputs 1 and 2 carry the master unchanged");
            CHECK (std::abs (steady (a[2]) - 0.125f) < 0.003f && std::abs (steady (a[3]) - 0.125f) < 0.003f, "and outputs 3 and 4 carry the monitoring feed at the control room level");
            HostEngine e2; e2.prepareOffline (sr, bs, 0); tone (e2); e2.editControlRoomParam ("crPair", 1); e2.editControlRoomParam ("crDim", true); auto b = live (e2, 6, 40);
            CHECK (std::abs (steady (b[0]) - 0.25f) < 0.004f && std::abs (steady (b[2]) - 0.025f) < 0.001f && steady (b[4]) < 1e-6f, "Dim applies to the monitoring pair only, and the other outputs are silent");
            HostEngine e3; e3.prepareOffline (sr, bs, 0); tone (e3); e3.editControlRoomParam ("crDb", -6.0206); e3.editControlRoomParam ("crPair", 2); auto c = live (e3, 2, 40);
            CHECK (std::abs (steady (c[0]) - 0.125f) < 0.003f, "a pair the sound card does not have falls back to the main outputs"); }
        {   // Dim while the sound plays: no click
            HostEngine e; e.prepareOffline (sr, bs, 0); tone (e); live (e, 2, 20); e.editControlRoomParam ("crDim", true); auto a = live (e, 2, 20); float w = 0; for (size_t i = 1; i < a[0].size(); ++i) w = std::max (w, std::abs (a[0][i] - a[0][i - 1]));
            e.editControlRoomParam ("crDim", false); auto b = live (e, 2, 40); float w2 = 0; for (size_t i = 1; i < b[0].size(); ++i) w2 = std::max (w2, std::abs (b[0][i] - b[0][i - 1]));
            CHECK (w < 0.05f && w2 < 0.05f, "switching Dim on or off while the sound plays does not click");
            CHECK (std::abs (steady (b[0]) - 0.25f) < 0.004f, "and switching it off brings the level back"); }
        { HostEngine e; e.prepareOffline (sr, bs, 0); e.editControlRoomParam ("crDb", -500.0); e.editControlRoomParam ("crPair", 99);
          CHECK (e.crDb.load() == -90.0f && e.crPair.load() == 8, "the settings are held to their ranges"); e.editControlRoomParam ("crDb", 50.0); e.editControlRoomParam ("crPair", -3); CHECK (e.crDb.load() == 6.0f && e.crPair.load() == 0, "at both ends"); CHECK (! e.editControlRoomParam ("nonsense", 1.0), "an unknown setting is refused"); }
        { HostEngine e; e.prepareOffline (sr, bs, 0); e.undo.clock = [] { static double c = 0; c += 5000; return c; };
          e.editControlRoomParam ("crDb", -10.0); e.editControlRoomParam ("crDim", true); e.undo.undo (); CHECK (! e.crDim.load() && e.crDb.load() == -10.0f, "undo puts Dim back"); e.undo.undo (); CHECK (e.crDb.load() == 0.0f, "then the level"); e.undo.redo (); CHECK (e.crDb.load() == -10.0f, "redo does it again"); }
        { HostEngine e; e.prepareOffline (sr, bs, 0); e.editControlRoomParam ("crDb", -12.5); e.editControlRoomParam ("crDim", true); e.editControlRoomParam ("crPair", 2);
          auto proj = e.saveProject(); HostEngine g; g.prepareOffline (sr, bs, 0); g.loadProject (proj, [&] (juce::String) {});
          CHECK (g.crDb.load() == -12.5f && g.crDim.load() && g.crPair.load() == 2, "the control room settings are saved with the project and come back with it");
          juce::var st = g.snapshot(); const auto c = st.getProperty ("controlRoom", {}); CHECK ((double) c.getProperty ("crDb", 0.0) == -12.5 && (bool) c.getProperty ("crDim", false), "and the Mixer can read them from the state");
          HostEngine h; h.prepareOffline (sr, bs, 0); h.loadProject (proj, [&] (juce::String) {}); { HostEngine d; d.prepareOffline (sr, bs, 0); h.loadProject (d.saveProject(), [&] (juce::String) {}); }
          CHECK (h.crDb.load() == 0.0f && ! h.crDim.load() && h.crPair.load() == 0, "a song that never used it loads at 0 dB, no Dim, on the main outputs"); }
    }

    // ---- GROOVE CHANNELS (Groove dock): patches, strength, impacts, slide, shuffle, random, anchor; extract; assignment ----
    {
        struct NoteOut { double t; int vel; };
        auto clip4 = [] (std::initializer_list<std::array<double, 4>> nts /* beat, len, note, vel */) { auto c = std::make_shared<Clip>(); c->lengthBeats = 4.0; c->id = 1; for (auto& q : nts) { NoteEv n; n.beat = q[0]; n.len = q[1]; n.note = (int) q[2]; n.vel = (int) q[3]; c->notes.push_back (n); } return c; };
        auto inst = [] (std::shared_ptr<Clip> c) { ClipInst in; in.id = 7; in.start = 0.0; in.length = 4.0; in.content = c; return in; };
        auto play = [&] (const ClipInst& in, const GrooveEval* g, double sw = 0.0) { std::vector<NoteOut> v; walkInstance (in, 0.0, 4.0, false, [&] (const NoteEv& n, double t) { v.push_back ({ t, n.vel }); }, [&] (const NoteEv&, double) {}, [&] (const NoteEv&) {}, sw, 0.5, g); std::sort (v.begin(), v.end(), [] (const NoteOut& x, const NoteOut& y) { return x.t < y.t; }); return v; };
        auto offsOf = [&] (const ClipInst& in, const GrooveEval* g) { std::vector<double> v; walkInstance (in, 0.0, 4.0, false, [&] (const NoteEv&, double) {}, [&] (const NoteEv&, double t) { v.push_back (t); }, [&] (const NoteEv&) {}, 0.0, 0.5, g); std::sort (v.begin(), v.end()); return v; };
        GrooveSet gs = defaultGrooveSet(); auto cfg = [&] (const char* patch, float strength) { gs = defaultGrooveSet(); gs.ch[1].on = true; gs.ch[1].patch = patch; gs.ch[1].strength = strength; return GrooveEval::make (gs, 1, 7); };
        auto sixteenths = clip4 ({ { 0.0, 0.25, 60, 100 }, { 0.25, 0.25, 60, 100 }, { 0.5, 0.25, 60, 100 }, { 0.75, 0.25, 60, 100 } });
        const auto straight = play (inst (sixteenths), nullptr);
        CHECK (straight.size() == 16 || straight.size() == 4, "(setup) the clip plays"); const size_t per = 4;
        { GrooveEval off = cfg ("16th swing 66%", 0.0f); auto r = play (inst (sixteenths), &off); bool same = r.size() == straight.size(); for (size_t i = 0; same && i < r.size(); ++i) same = r[i].t == straight[i].t && r[i].vel == straight[i].vel;
          CHECK (! off.active() && same, "a groove at strength 0 does nothing at all: every note plays exactly as written"); }
        { gs = defaultGrooveSet(); gs.ch[1].on = false; gs.ch[1].patch = "16th swing 66%"; auto g = GrooveEval::make (gs, 1, 7); auto r = play (inst (sixteenths), g.active() ? &g : nullptr); bool same = r.size() == straight.size(); for (size_t i = 0; same && i < r.size(); ++i) same = r[i].t == straight[i].t;
          CHECK (same, "and so does a channel that is switched off"); }
        { GrooveEval g = cfg ("16th swing 66%", 1.0f); auto r = play (inst (sixteenths), &g);
          CHECK (std::abs (r[0].t - 0.0) < 1e-12 && std::abs (r[1].t - (0.25 + 0.32 * 0.25)) < 1e-6 && std::abs (r[2].t - 0.5) < 1e-12 && std::abs (r[3].t - (0.75 + 0.32 * 0.25)) < 1e-6, "a 66% sixteenth swing delays every second sixteenth by 0.32 of a step, and leaves the others alone"); }
        { GrooveEval g = cfg ("16th swing 66%", 0.5f); auto r = play (inst (sixteenths), &g); CHECK (std::abs (r[1].t - (0.25 + 0.16 * 0.25)) < 1e-6, "at half strength it delays them half as much"); }
        { GrooveEval g = cfg ("16th swing 66%", 1.0f); gs.ch[1].timing = 50.0f; auto g2 = GrooveEval::make (gs, 1, 7); auto r = play (inst (sixteenths), &g2); CHECK (std::abs (r[1].t - (0.25 + 0.16 * 0.25)) < 1e-6, "a timing impact of 50% does the same"); (void) g; }
        { auto c = clip4 ({ { 0.27, 0.25, 60, 100 }, { 0.5, 0.25, 60, 100 } }); GrooveEval g = cfg ("16th swing 66%", 1.0f); auto r = play (inst (c), &g); CHECK (std::abs (r[0].t - (0.27 + 0.08)) < 1e-6, "a note keeps its own small deviation: the groove adds to where it is, it does not snap it"); }
        { GrooveEval g = cfg ("Accent the beat", 1.0f); auto r = play (inst (sixteenths), &g);
          CHECK (r[0].vel == 120 && r[1].vel == 85 && r[2].vel == 106 && r[3].vel == 85 && std::abs (r[1].t - 0.25) < 1e-12, "a velocity groove makes the beat 20% louder, the offbeats 15% softer and the eighths 6% louder, without moving anything"); }
        { GrooveEval g = cfg ("Accent the beat", 1.0f); gs.ch[1].vel = 0.0f; auto g2 = GrooveEval::make (gs, 1, 7); auto r = play (inst (sixteenths), &g2); CHECK (r[0].vel == 100 && r[1].vel == 100, "with the velocity impact at 0 the velocities are untouched"); (void) g; }
        { GrooveEval g = cfg ("Staccato", 1.0f); auto r = offsOf (inst (sixteenths), &g); CHECK (r.size() == per && std::abs (r[0] - 0.1375) < 1e-6, "a length groove (staccato, -45%) ends a quarter-beat note at 0.1375"); }
        { GrooveEval g = cfg ("Legato", 1.0f); auto r = offsOf (inst (sixteenths), &g); CHECK (std::abs (r[0] - 0.35) < 1e-6, "and legato (+40%) at 0.35"); }
        { GrooveEval g = cfg ("Straight", 1.0f); gs.ch[1].slide = 20.0f; gs.ch[1].shuffle = 0.0f; auto g2 = GrooveEval::make (gs, 1, 7); auto r = play (inst (sixteenths), &g2); CHECK (std::abs (r[0].t - 0.05) < 1e-6 && std::abs (r[2].t - 0.55) < 1e-6, "slide moves every note later by a percent of a step (20% of a sixteenth is 0.05 beat)"); (void) g; }
        { GrooveEval g = cfg ("Straight", 1.0f); gs.ch[1].slide = -50.0f; auto g2 = GrooveEval::make (gs, 1, 7); auto r = play (inst (sixteenths), &g2); CHECK (std::abs (r[0].t - 0.0) < 1e-12 && std::abs (r[1].t - 0.125) < 1e-6, "slide earlier never moves a note before the start of its clip"); (void) g; }
        { GrooveEval g = cfg ("Straight", 1.0f); gs.ch[1].shuffle = 100.0f; auto g2 = GrooveEval::make (gs, 1, 7); auto r = play (inst (sixteenths), &g2); CHECK (std::abs (r[0].t) < 1e-12 && std::abs (r[1].t - (0.25 + 0.25 / 3.0)) < 1e-6, "shuffle delays every second step by a third of a step at 100%"); (void) g; }
        { GrooveEval g = cfg ("Straight", 1.0f); gs.ch[1].rand = 24.0f; auto g2 = GrooveEval::make (gs, 1, 7); auto r1 = play (inst (sixteenths), &g2), r2 = play (inst (sixteenths), &g2); bool repeat = true, within = true, differ = false;
          for (size_t i = 0; i < r1.size(); ++i) { repeat = repeat && r1[i].t == r2[i].t; within = within && std::abs (r1[i].t - straight[i].t) <= 24.0 / 480.0 + 1e-12; differ = differ || std::abs (r1[i].t - straight[i].t) > 1e-9; }
          CHECK (repeat && within && differ, "random timing nudges each note by up to its ticks (24 of 480 is 0.05 beat), the same way every time it plays"); (void) g; }
        { GrooveEval g = cfg ("16th swing 66%", 1.0f); gs.ch[1].anchor = 1; auto g2 = GrooveEval::make (gs, 1, 7); auto r = play (inst (sixteenths), &g2); CHECK (std::abs (r[0].t - 0.08) < 1e-6 && std::abs (r[1].t - 0.25) < 1e-6, "the anchor shifts which step of the pattern meets the start of the clip"); (void) g; }
        { gs = defaultGrooveSet(); gs.ch[1].on = true; gs.ch[1].patch = "16th swing 66%"; auto g = GrooveEval::make (gs, 1, 7); auto withSwing = play (inst (sixteenths), &g, 1.0 / 6.0); auto swingOnly = play (inst (sixteenths), nullptr, 1.0 / 6.0);
          CHECK (withSwing.size() == swingOnly.size() && std::abs (withSwing[1].t - swingOnly[1].t) > 0.01, "the global swing and a groove channel work together"); }
        { auto c = std::make_shared<Clip>(); c->lengthBeats = 2.0; c->id = 1; NoteEv n; n.beat = 1.75; n.len = 0.25; n.note = 60; c->notes.push_back (n); ClipInst in; in.id = 7; in.start = 0.0; in.length = 4.0; in.content = c;
          gs = defaultGrooveSet(); gs.ch[1].on = true; gs.ch[1].patch = "Laid back"; auto g = GrooveEval::make (gs, 1, 7); std::vector<double> t; walkInstance (in, 0.0, 4.0, false, [&] (const NoteEv&, double tt) { t.push_back (tt); }, [&] (const NoteEv&, double) {}, [&] (const NoteEv&) {}, 0.0, 0.5, &g);
          CHECK (t.size() == 2 && std::abs (t[0] - (1.75 + 0.03)) < 1e-6 && std::abs (t[1] - (3.75 + 0.03)) < 1e-6, "a groove repeats with a looping clip, and a late note at the end of the clip stays inside it"); }
        {   // through the engine: a track plays its notes through a channel
            struct Ev2 { double beat; bool on; };
            auto probe = [&] (HostEngine& e, std::vector<Ev2>& log, double toBeat)
            {
                struct Probe : TestProc { Probe (std::vector<Ev2>* l) : TestProc (2, 2, 0), out (l) {} bool acceptsMidi() const override { return true; }
                    void processBlock (juce::AudioBuffer<float>&, juce::MidiBuffer& m) override { for (const auto meta : m) { auto msg = meta.getMessage(); if (msg.isNoteOn()) out->push_back ({ (double) (done + meta.samplePosition) / 24000.0, true }); } done += 256; }
                    std::vector<Ev2>* out; long long done = 0; };
                e.addProcessor (0, std::make_unique<Probe> (&log), "probe"); e.transport.locate (0.0); e.transport.play(); run (e, (int) std::ceil (toBeat * 24000.0 / bs) + 2, bs);
            };
            auto mk = [] { auto* c = new juce::DynamicObject(); c->setProperty ("lengthBeats", 4.0); juce::Array<juce::var> arr; for (double b : { 0.0, 0.25, 0.5, 0.75 }) { auto* x = new juce::DynamicObject(); x->setProperty ("beat", b); x->setProperty ("len", 0.2); x->setProperty ("note", 60); x->setProperty ("vel", 100); arr.add (juce::var (x)); } c->setProperty ("notes", arr); return juce::var (c); };
            auto times = [&] (std::vector<Ev2>& l) { std::vector<double> v; for (auto& q : l) v.push_back (q.beat); return v; };
            { HostEngine e; e.prepareOffline (sr, bs, 0); e.editAddInstance (0, 0.0, 4.0, 0.0, mk(), 0); std::vector<Ev2> log; probe (e, log, 1.0); auto t = times (log); CHECK (t.size() == 4 && std::abs (t[1] - 0.25) < 0.001, "(setup) through the engine the clip plays straight"); }
            { HostEngine e; e.prepareOffline (sr, bs, 0); e.editAddInstance (0, 0.0, 4.0, 0.0, mk(), 0); CHECK (e.editGrooveChan (1, "patch", "16th swing 66%") && e.editGrooveChan (1, "on", true) && e.editChainParam (0, "groove", 1), "a groove is set up and the track is given channel 1");
              std::vector<Ev2> log; probe (e, log, 1.0); auto t = times (log); CHECK (t.size() == 4 && std::abs (t[0]) < 0.001 && std::abs (t[1] - 0.33) < 0.001 && std::abs (t[3] - 0.83) < 0.001, "a track given a groove channel plays through it: the second and fourth sixteenths land at 0.33 and 0.83 (to the sample)"); }
            { HostEngine e; e.prepareOffline (sr, bs, 0); e.editAddInstance (0, 0.0, 4.0, 0.0, mk(), 0); e.editGrooveChan (1, "patch", "16th swing 66%"); e.editGrooveChan (1, "on", true); e.editChainParam (0, "groove", 1);
              const auto arr = e.trackAt (0)->arrSnap.load(); const int id = arr->insts[0].id; juce::DynamicObject* po = new juce::DynamicObject(); po->setProperty ("groove", -1); CHECK (e.editSetInstance (id, juce::var (po)), "a clip can choose not to play through its track's groove");
              std::vector<Ev2> log; probe (e, log, 1.0); auto t = times (log); CHECK (t.size() == 4 && std::abs (t[1] - 0.25) < 0.001, "and then it plays straight"); }
            { HostEngine e; e.prepareOffline (sr, bs, 0); e.editAddInstance (0, 0.0, 4.0, 0.0, mk(), 0); e.editGrooveChan (2, "patch", "Laid back"); e.editGrooveChan (2, "on", true); const int id = e.trackAt (0)->arrSnap.load()->insts[0].id; juce::DynamicObject* po = new juce::DynamicObject(); po->setProperty ("groove", 2); e.editSetInstance (id, juce::var (po));
              std::vector<Ev2> log; probe (e, log, 1.0); auto t = times (log); CHECK (t.size() == 4 && std::abs (t[0] - 0.03) < 0.001, "a clip can have a groove channel of its own, whatever its track has"); }
        }
        {   // extract a groove from a clip
            HostEngine e; e.prepareOffline (sr, bs, 0); auto* c = new juce::DynamicObject(); c->setProperty ("lengthBeats", 4.0); juce::Array<juce::var> arr;
            for (int i = 0; i < 16; ++i) { auto* x = new juce::DynamicObject(); x->setProperty ("beat", i * 0.25 + ((i & 1) ? 0.05 : 0.0)); x->setProperty ("len", 0.2); x->setProperty ("note", 60); x->setProperty ("vel", (i % 4 == 0) ? 120 : 80); arr.add (juce::var (x)); }
            c->setProperty ("notes", arr); e.editAddInstance (0, 0.0, 4.0, 0.0, juce::var (c), 0); const int cid = e.trackAt (0)->arrSnap.load()->insts[0].content->id;
            juce::String err; const int n0 = (int) e.grooveCopy().patches.size(); const int n1 = e.extractGroove (cid, "My feel", 0.25, err);
            CHECK (n1 == n0 + 1 && err.isEmpty(), "a groove is extracted from a note clip"); const auto gcopy = e.grooveCopy(); const auto* p = gcopy.find ("My feel");
            CHECK (p != nullptr && ! p->builtin && p->steps.size() == 16 && std::abs (p->steps[1].timing - 0.2f) < 1e-4f && std::abs (p->steps[0].timing) < 1e-4f && std::abs (p->steps[2].timing) < 1e-4f, "it has the clip's feel: the odd sixteenths a fifth of a step late, the others on the grid");
            CHECK (p != nullptr && p->steps[0].vel > 25.0f && p->steps[1].vel < -8.0f, "and its accents: the beat louder than the clip's average, the offbeats softer");
            juce::String err2; CHECK (e.extractGroove (cid, "My feel", 0.25, err2) == n1 + 1 && e.grooveCopy().find ("My feel 2") != nullptr, "a second extraction with the same name gets its own name");
            CHECK (e.extractGroove (99999, "x", 0.25, err2) == 0 && err2.isNotEmpty(), "extracting from a clip that is not there is refused");
            CHECK (e.editGrooveChan (3, "patch", "My feel") && e.editGrooveChan (3, "patch", "") && ! e.editGrooveChan (3, "patch", "no such groove"), "a channel can choose an extracted groove or none, and a groove that does not exist is refused");
            e.editGrooveChan (3, "patch", "My feel"); CHECK (e.deleteGroove ("My feel") && e.grooveCopy().find ("My feel") == nullptr && ! e.deleteGroove ("Straight") && ! e.deleteGroove ("nothing"), "an extracted groove can be deleted, the built-in ones and ones that do not exist cannot");
            e.undo.undo (); CHECK (e.grooveCopy().find ("My feel") != nullptr, "undo brings a deleted groove back"); }
        {   // settings: ranges, undo, save and load
            HostEngine e; e.prepareOffline (sr, bs, 0); e.undo.clock = [] { static double cc = 0; cc += 5000; return cc; };
            e.editGrooveChan (1, "strength", 9.0); e.editGrooveChan (1, "timing", 999.0); e.editGrooveChan (1, "vel", -5.0); e.editGrooveChan (1, "rand", 500.0); e.editGrooveChan (1, "slide", -99.0); e.editGrooveChan (1, "shuffle", 300.0); e.editGrooveChan (1, "anchor", 200.0);
            { const auto g = e.grooveCopy(); CHECK (g.ch[1].strength == 1.0f && g.ch[1].timing == 200.0f && g.ch[1].vel == 0.0f && g.ch[1].rand == 120.0f && g.ch[1].slide == -50.0f && g.ch[1].shuffle == 100.0f && g.ch[1].anchor == 63, "the settings are held to their ranges"); }
            CHECK (! e.editGrooveChan (0, "on", true) && ! e.editGrooveChan (32, "on", true) && ! e.editGrooveChan (1, "nonsense", 1.0), "channel 0 (the global swing), channels that do not exist and unknown settings are refused");
            e.editGrooveChan (5, "on", true); e.editGrooveChan (5, "strength", 0.4); e.undo.undo (); CHECK (e.grooveCopy().ch[5].strength == 1.0f && e.grooveCopy().ch[5].on, "undo puts a setting back"); e.undo.undo (); CHECK (! e.grooveCopy().ch[5].on, "and then the switch"); e.undo.redo (); CHECK (e.grooveCopy().ch[5].on, "redo does it again"); }
        {
            HostEngine e; e.prepareOffline (sr, bs, 0); e.editGrooveChan (4, "on", true); e.editGrooveChan (4, "patch", "Funk 16ths"); e.editGrooveChan (4, "strength", 0.7); e.editGrooveChan (4, "rand", 12.0); e.editGrooveChan (4, "slide", 10.0); e.editChainParam (1, "groove", 4);
            auto* c = new juce::DynamicObject(); c->setProperty ("lengthBeats", 4.0); juce::Array<juce::var> arr; for (int i = 0; i < 8; ++i) { auto* x = new juce::DynamicObject(); x->setProperty ("beat", i * 0.5 + ((i & 1) ? 0.1 : 0.0)); x->setProperty ("len", 0.25); x->setProperty ("note", 60); x->setProperty ("vel", (i % 2) ? 70 : 110); arr.add (juce::var (x)); } c->setProperty ("notes", arr);
            e.editAddInstance (2, 0.0, 4.0, 0.0, juce::var (c), 0); { const int id = e.trackAt (2)->arrSnap.load()->insts[0].id; auto* po = new juce::DynamicObject(); po->setProperty ("groove", 6); e.editSetInstance (id, juce::var (po)); }
            juce::String err; e.extractGroove (e.trackAt (2)->arrSnap.load()->insts[0].content->id, "Saved feel", 0.5, err);
            auto proj = e.saveProject(); HostEngine g; g.prepareOffline (sr, bs, 0); g.loadProject (proj, [&] (juce::String) {}); const auto gs2 = g.grooveCopy();
            CHECK (gs2.ch[4].on && gs2.ch[4].patch == "Funk 16ths" && std::abs (gs2.ch[4].strength - 0.7f) < 1e-6f && gs2.ch[4].rand == 12.0f && gs2.ch[4].slide == 10.0f, "the groove channels are saved with the project and come back with it");
            CHECK (gs2.find ("Saved feel") != nullptr && ! gs2.find ("Saved feel")->builtin && gs2.find ("Saved feel")->steps.size() == 8 && gs2.find ("Funk 16ths") != nullptr && gs2.find ("Funk 16ths")->builtin, "an extracted groove is saved with the song (the built-in ones are always there)");
            CHECK (g.trackAt (1)->grooveCh.load() == 4 && g.trackAt (0)->grooveCh.load() == -1 && g.trackAt (2)->arrSnap.load()->insts[0].groove == 6, "which channel a track and a clip play through is saved too");
            juce::var st = g.snapshot(); const auto gv = st.getProperty ("groove", {}); auto* chs = gv.getProperty ("channels", {}).getArray(); auto* pats = gv.getProperty ("patches", {}).getArray();
            CHECK (chs != nullptr && chs->size() == 32 && (bool) (*chs)[4].getProperty ("on", false) && pats != nullptr && pats->size() >= 12, "and the Groove dock can read the channels and the patch names from the state");
            HostEngine h; h.prepareOffline (sr, bs, 0); h.loadProject (proj, [&] (juce::String) {}); { HostEngine d; d.prepareOffline (sr, bs, 0); h.loadProject (d.saveProject(), [&] (juce::String) {}); }
            CHECK (! h.grooveCopy().ch[4].on && h.grooveCopy().find ("Saved feel") == nullptr && h.trackAt (1)->grooveCh.load() == -1, "a song that never used grooves loads with every channel off and no extracted grooves"); }
    }

    // ---- PITCH and COMP spikes (S12): Vocal Edit and Comp Edit sound-making, proven before any screen ----
    {
        const double sr2 = 48000.0; const double PI2 = 6.283185307179586;
        auto sine = [&] (double hz, double sec, double amp = 0.5) { AudioData a; a.rate = sr2; const int n = (int) (sec * sr2); a.buf.setSize (1, n); double ph = 0; for (int i = 0; i < n; ++i) { a.buf.setSample (0, i, (float) (amp * std::sin (ph))); ph += PI2 * hz / sr2; } return a; };
        auto voiceLike = [&] (double hz, double sec) { AudioData a; a.rate = sr2; const int n = (int) (sec * sr2); a.buf.setSize (1, n); double ph = 0; for (int i = 0; i < n; ++i) { float v = 0; for (int h = 1; h <= 6; ++h) v += (float) (std::sin (ph * h) / h); a.buf.setSample (0, i, 0.25f * v); ph += PI2 * hz / sr2; } return a; };
        auto medianHz = [] (const std::vector<PitchFrame>& f, double t0, double t1) { std::vector<double> v; for (auto& x : f) if (x.voiced && x.sec >= t0 && x.sec <= t1) v.push_back (x.hz); if (v.empty()) return 0.0; std::sort (v.begin(), v.end()); return v[v.size() / 2]; };
        auto centsOff = [] (double hz, double ref) { return 1200.0 * std::log2 (hz / ref); };
        // the tracker
        for (double hz : { 110.0, 220.0, 440.0, 880.0 }) { auto fr = trackPitch (sine (hz, 1.0)); const double m = medianHz (fr, 0.1, 0.9); CHECK (m > 0 && std::abs (centsOff (m, hz)) < 3.0, "the pitch tracker reads a steady tone to within 3 cents"); }
        { auto fr = trackPitch (voiceLike (196.0, 1.0)); CHECK (std::abs (centsOff (medianHz (fr, 0.1, 0.9), 196.0)) < 5.0, "it also reads a tone with harmonics (like a voice) on the right octave, within 5 cents"); }
        { AudioData sil; sil.rate = sr2; sil.buf.setSize (1, 48000); sil.buf.clear(); auto fr = trackPitch (sil); int v = 0; for (auto& f : fr) v += f.voiced; CHECK (! fr.empty() && v == 0, "silence is never voiced"); }
        { AudioData nz; nz.rate = sr2; nz.buf.setSize (1, 48000); juce::Random rnd (7); for (int i = 0; i < 48000; ++i) nz.buf.setSample (0, i, (rnd.nextFloat() - 0.5f) * 0.4f); auto fr = trackPitch (nz); int v = 0; for (auto& f : fr) v += f.voiced; CHECK (v <= (int) fr.size() / 10, "noise is almost never called voiced (at most one frame in ten)"); }
        { AudioData tn = sine (330.0, 0.6); AudioData both; both.rate = sr2; both.buf.setSize (1, 48000); both.buf.clear(); for (int i = 0; i < tn.numSamples(); ++i) both.buf.setSample (0, 12000 + i, tn.buf.getSample (0, i)); auto fr = trackPitch (both); double tv = 0; for (auto& f : fr) if (f.voiced && (f.sec < 0.2 || f.sec > 0.9)) tv++; CHECK (tv == 0, "no pitch is claimed in the silence before and after a note"); }
        // the shifter
        { auto a = sine (220.0, 1.5); auto r = retunePitch (a, [] (double) { return 12.0; }); CHECK (r && r->numSamples() == a.numSamples(), "shifting keeps the length exactly");
          auto fr = trackPitch (*r); CHECK (std::abs (centsOff (medianHz (fr, 0.3, 1.2), 440.0)) < 10.0, "+12 semitones lands an octave up within 10 cents"); }
        { auto a = sine (440.0, 1.5); auto r = retunePitch (a, [] (double) { return -5.0; }); auto fr = trackPitch (*r); CHECK (std::abs (centsOff (medianHz (fr, 0.3, 1.2), 440.0 * std::pow (2.0, -5.0 / 12.0))) < 10.0, "-5 semitones lands a fourth down within 10 cents"); }
        { auto a = voiceLike (200.0, 1.5); auto r = retunePitch (a, [] (double) { return 0.0; }); double num = 0, den = 0; for (int i = 20000; i < 50000; ++i) { const double x = a.buf.getSample (0, i), y = r->buf.getSample (0, i); num += (x - y) * (x - y); den += x * x; } CHECK (std::sqrt (num / den) < 0.15, "0 semitones through the shifter stays close to the original (error under 15%; the library is not bit-exact)"); }
        { auto a = sine (220.0, 2.0); auto r = retunePitch (a, [] (double s) { return s < 1.0 ? 0.0 : 7.0; }); auto fr = trackPitch (*r); CHECK (std::abs (centsOff (medianHz (fr, 0.2, 0.8), 220.0)) < 10.0 && std::abs (centsOff (medianHz (fr, 1.3, 1.9), 220.0 * std::pow (2.0, 7.0 / 12.0))) < 10.0, "the shift can change part-way through: the first second is untouched, the second a fifth up"); }
        { auto a = sine (220.0, 1.0); std::atomic<bool> stop { true }; CHECK (retunePitch (a, [] (double) { return 1.0; }, true, &stop) == nullptr, "a shift can be cancelled"); }
        { AudioData st2; st2.rate = sr2; st2.buf.setSize (2, 48000); for (int i = 0; i < 48000; ++i) { st2.buf.setSample (0, i, 0.4f * (float) std::sin (PI2 * 220.0 * i / sr2)); st2.buf.setSample (1, i, 0.2f * (float) std::sin (PI2 * 220.0 * i / sr2)); } auto r = retunePitch (st2, [] (double) { return 12.0; }, false);
          AudioData lc; lc.rate = sr2; lc.buf.setSize (1, 48000); lc.buf.copyFrom (0, 0, r->buf, 0, 0, 48000); AudioData rc; rc.rate = sr2; rc.buf.setSize (1, 48000); rc.buf.copyFrom (0, 0, r->buf, 1, 0, 48000);
          CHECK (r->buf.getNumChannels() == 2 &&std::abs (centsOff (medianHz (trackPitch (lc), 0.2, 0.8), 440.0)) < 15.0 && std::abs (centsOff (medianHz (trackPitch (rc), 0.2, 0.8), 440.0)) < 15.0 && r->buf.getMagnitude (1, 12000, 24000) < 0.3f && r->buf.getMagnitude (0, 12000, 24000) > 0.3f, "both channels of a stereo recording (the same voice, louder on the left) are shifted, and keep their balance"); }
        { auto v = voiceLike (180.0, 1.5); auto up = retunePitch (v, [] (double) { return 3.0; }, true); auto pl = retunePitch (v, [] (double) { return 3.0; }, false);
          const double lv = 20.0 * std::log10 (v.buf.getRMSLevel (0, 20000, 30000)), lu = 20.0 * std::log10 (up->buf.getRMSLevel (0, 20000, 30000)), lp = 20.0 * std::log10 (pl->buf.getRMSLevel (0, 20000, 30000));
          CHECK (std::abs (lp - lv) < 1.5, "a voice-like tone moved up three semitones keeps its level within 1.5 dB (plain mode)"); CHECK (std::abs (lu - lv) < 3.5, "and with the voice colour kept (formant compensation on) within 3.5 dB"); }
        // correct to scale
        { ScaleSpec cmaj; CHECK (nearestScaleMidi (cmaj, 61.3) == 60.0 || nearestScaleMidi (cmaj, 61.3) == 62.0, "the nearest C major note to a note between C# and D is C or D");
          CHECK (nearestScaleMidi (cmaj, 60.4) == 60.0 && nearestScaleMidi (cmaj, 66.9) == 67.0 && nearestScaleMidi (cmaj, 70.7) == 71.0, "and it picks the closest scale note each time");
          ScaleSpec dmaj; dmaj.key = 2; for (int i = 0; i < 12; ++i) dmaj.mask[i] = false; for (int i : { 0, 2, 4, 5, 7, 9, 11 }) dmaj.mask[i] = true;
          CHECK (nearestScaleMidi (dmaj, 63.8) == 64.0 && nearestScaleMidi (dmaj, 65.6) == 66.0, "a scale in another key follows the key note (D major has E and F sharp)"); }
        { const double flat = 440.0 * std::pow (2.0, -0.35 / 12.0); auto a = sine (flat, 1.5); ScaleSpec cmaj; auto r = correctToScale (a, cmaj, 1.0); auto fr = trackPitch (*r);
          CHECK (r && r->numSamples() == a.numSamples() && std::abs (centsOff (medianHz (fr, 0.3, 1.2), 440.0)) < 10.0, "a note 35 cents flat of A is pulled onto A (within 10 cents) with the length unchanged");
          auto h = correctToScale (a, cmaj, 0.5); const double mh = medianHz (trackPitch (*h), 0.3, 1.2); CHECK (std::abs (centsOff (mh, flat) - 17.5) < 10.0, "half strength moves it about half way (17 cents of 35)");
          auto z = correctToScale (a, cmaj, 0.0); CHECK (std::abs (centsOff (medianHz (trackPitch (*z), 0.3, 1.2), flat)) < 8.0, "strength 0 leaves the pitch where it was"); }
        { const double sharp = 440.0 * std::pow (2.0, 1.4 / 12.0); AudioData a = sine (sharp, 1.5); ScaleSpec cmaj; auto fr = trackPitch (*correctToScale (a, cmaj, 1.0)); const double m = medianHz (fr, 0.3, 1.2); CHECK (std::abs (centsOff (m, 440.0 * std::pow (2.0, 2.0 / 12.0))) < 12.0, "a note 140 cents sharp of A goes to B (the nearer scale note), not back to A"); }
        { auto a = sine (440.0, 0.5); AudioData both; both.rate = sr2; both.buf.setSize (1, 96000); both.buf.clear(); for (int i = 0; i < a.numSamples(); ++i) both.buf.setSample (0, i, a.buf.getSample (0, i)); ScaleSpec cmaj; auto r = correctToScale (both, cmaj, 1.0); double tail = 0; for (int i = 60000; i < 96000; ++i) tail = juce::jmax (tail, (double) std::abs (r->buf.getSample (0, i))); CHECK (tail < 0.01, "silence after a note stays silent after correction"); }
        // the comp
        {
            auto A = sine (200.0, 2.0, 0.8), B = sine (200.0, 2.0, 0.4); AudioData C; C.rate = sr2; C.buf.setSize (1, (int) (2.0 * sr2)); for (int i = 0; i < C.numSamples(); ++i) C.buf.setSample (0, i, 0.1f); std::vector<const AudioData*> T { &A, &B, &C };
            auto c = compTakes (T, { { 0.0, 1.0, 0 }, { 1.0, 2.0, 1 } }, 2.0, sr2, 0.02); CHECK (c && c->numSamples() == 96000, "a comp is exactly as long as asked");
            bool same1 = true, same2 = true; for (int i = 0; i < 47000; ++i) if (c->buf.getSample (0, i) != A.buf.getSample (0, i)) same1 = false; for (int i = 49000; i < 96000; ++i) if (c->buf.getSample (0, i) != B.buf.getSample (0, i)) same2 = false;
            CHECK (same1 && same2, "away from the join, each section is exactly its take, sample for sample");
            double maxStep = 0; for (int i = 47000; i < 49000; ++i) maxStep = juce::jmax (maxStep, (double) std::abs (c->buf.getSample (0, i) - c->buf.getSample (0, i - 1))); CHECK (maxStep < 0.1, "the join is a smooth crossfade: no jump larger than the signal's own steepest slope");
            auto N1 = sine (200.0, 2.0, 0.5), N2 = sine (317.0, 2.0, 0.5); std::vector<const AudioData*> T2 { &N1, &N2 }; auto e = compTakes (T2, { { 0.0, 1.0, 0 }, { 1.0, 2.0, 1 } }, 2.0, sr2, 0.04);
            auto rms = [&] (const AudioData& d, int a, int b) { double s = 0; for (int i = a; i < b; ++i) s += (double) d.buf.getSample (0, i) * d.buf.getSample (0, i); return std::sqrt (s / (b - a)); };
            const double before = rms (*e, 30000, 40000), mid = rms (*e, 47000, 49000); CHECK (std::abs (mid / before - 1.0) < 0.15, "unrelated takes keep about the same power through the join (equal-power crossfade)");
            auto g = compTakes (T, { { 0.2, 0.6, 0 }, { 1.2, 1.5, 2 } }, 2.0, sr2, 0.02); double gap = 0; for (int i = 30000; i < 55000; ++i) gap = juce::jmax (gap, (double) std::abs (g->buf.getSample (0, i))); CHECK (gap == 0.0, "a gap between sections is silence, and a section at its edges starts and ends hard");
            auto o = compTakes (T, { { 1.5, 5.0, 1 }, { -1.0, 0.0, 0 }, { 0.5, 0.5, 0 }, { 0.0, 1.0, 9 } }, 2.0, sr2, 0.02); CHECK (o && o->numSamples() == 96000 && std::abs (o->buf.getSample (0, 95000) - B.buf.getSample (0, 95000)) < 1e-6, "sections past the end are clipped, and empty, backwards or unknown-take sections are ignored");
            auto sh = compTakes (T, { { 0.0, 3.0, 0 } }, 3.0, sr2, 0.02); CHECK (sh && sh->numSamples() == 144000 && sh->buf.getSample (0, 140000) == 0.0f, "a take shorter than the comp plays silence for the rest");
            CHECK (compTakes ({}, {}, 1.0, sr2) == nullptr && compTakes (T, {}, 0.0, sr2) == nullptr, "no takes or no length makes nothing");
            auto hc = compTakes (T, { { 0.0, 1.0, 0 }, { 1.0, 2.0, 1 } }, 2.0, sr2, 0.0); CHECK (hc && hc->buf.getSample (0, 47999) == A.buf.getSample (0, 47999) && hc->buf.getSample (0, 48000) == B.buf.getSample (0, 48000), "a fade of 0 is a hard cut exactly on the join");
        }
    }

    // ---- PANIC and a stubborn synth: one that ignores all-sound-off, all-notes-off and reset() (the Archie Mini Synth did) still lets go of a held note ----
    {
        auto peakOf = [] (const std::vector<float>& v) { float m = 0; for (float x : v) m = std::max (m, std::abs (x)); return m; };
        struct StubbornSynth : public juce::AudioProcessor
        {
            StubbornSynth() : juce::AudioProcessor (BusesProperties().withOutput ("out", juce::AudioChannelSet::stereo(), true)) {}
            const juce::String getName() const override { return "StubbornSynth"; }
            void prepareToPlay (double, int) override {}
            void releaseResources() override {}
            void reset() override {}                                                       // (ignored, like the real one)
            void processBlock (juce::AudioBuffer<float>& b, juce::MidiBuffer& m) override
            {
                for (const auto meta : m) { const auto msg = meta.getMessage(); if (msg.isNoteOn()) held[(size_t) (msg.getNoteNumber())] = true; else if (msg.isNoteOff()) held[(size_t) (msg.getNoteNumber())] = false; }   // (controllers are ignored)
                int n = 0; for (bool h : held) n += h ? 1 : 0;
                for (int c = 0; c < b.getNumChannels(); ++c) for (int i = 0; i < b.getNumSamples(); ++i) b.setSample (c, i, n > 0 ? 0.3f : 0.0f);
            }
            double getTailLengthSeconds() const override { return 0; }
            bool acceptsMidi() const override { return true; }
            bool producesMidi() const override { return false; }
            juce::AudioProcessorEditor* createEditor() override { return nullptr; }
            bool hasEditor() const override { return false; }
            int getNumPrograms() override { return 1; }
            int getCurrentProgram() override { return 0; }
            void setCurrentProgram (int) override {}
            const juce::String getProgramName (int) override { return {}; }
            void changeProgramName (int, const juce::String&) override {}
            void getStateInformation (juce::MemoryBlock&) override {}
            void setStateInformation (const void*, int) override {}
            std::array<bool, 128> held {};
        };
        { HostEngine e; e.prepareOffline (sr, bs, 0); e.addProcessor (0, std::make_unique<StubbornSynth>(), "stubborn");
          e.noteOn (0, 60, 100); e.noteOn (0, 67, 100, 3);
          auto a = run (e, 12, bs); CHECK (peakOf (a) > 0.2f, "a held note sounds");
          auto h = run (e, 12, bs); CHECK (peakOf (h) > 0.2f, "and keeps sounding: nothing releases it");
          e.panicAll();
          auto z = run (e, 12, bs); float tail = 0; for (size_t i = (size_t) bs * 2; i < z.size(); ++i) tail = std::max (tail, std::abs (z[i]));
          CHECK (tail < 1e-6f, "Panic silences it even though it ignores all-sound-off, all-notes-off and reset: notes on two different channels both let go"); }
        { HostEngine e; e.prepareOffline (sr, bs, 0); e.addProcessor (0, std::make_unique<StubbornSynth>(), "stubborn");
          e.noteOn (0, 62, 100); run (e, 6, bs); e.panicAll(); run (e, 6, bs); e.noteOn (0, 64, 100); auto n = run (e, 6, bs);
          CHECK (peakOf (n) > 0.2f, "after Panic the next note you play sounds as normal"); }
    }

    // ---- LOOP WRAP and a plug-in that follows the host position (a self-sequencing instrument such as Archie Acid 303) ----
    {
        struct PosProbe : public juce::AudioProcessor
        {
            PosProbe() : juce::AudioProcessor (BusesProperties().withOutput ("out", juce::AudioChannelSet::stereo(), true)) {}
            const juce::String getName() const override { return "PosProbe"; }
            void prepareToPlay (double, int) override {}
            void releaseResources() override {}
            void processBlock (juce::AudioBuffer<float>& b, juce::MidiBuffer&) override
            {
                b.clear(); double ppq = -1.0; if (auto* ph = getPlayHead()) if (auto pos = ph->getPosition()) if (auto q = pos->getPpqPosition()) ppq = *q;
                blocks.push_back ({ ppq, b.getNumSamples() });
            }
            double getTailLengthSeconds() const override { return 0; }
            bool acceptsMidi() const override { return true; }
            bool producesMidi() const override { return false; }
            juce::AudioProcessorEditor* createEditor() override { return nullptr; }
            bool hasEditor() const override { return false; }
            int getNumPrograms() override { return 1; }
            int getCurrentProgram() override { return 0; }
            void setCurrentProgram (int) override {}
            const juce::String getProgramName (int) override { return {}; }
            void changeProgramName (int, const juce::String&) override {}
            void getStateInformation (juce::MemoryBlock&) override {}
            void setStateInformation (const void*, int) override {}
            std::vector<std::pair<double, int>> blocks;      // (the position the plug-in was told, how many samples it was asked for)
        };
        // 121 bpm: a 4-beat loop is 1.9835 s = 95207.4 samples, so the wrap falls part-way through a block (it would fall exactly on a block edge only at a tempo that divides evenly)
        HostEngine e; e.prepareOffline (sr, bs, 0); e.transport.setBpm (121.0);
        auto pr = std::make_unique<PosProbe>(); auto* p = pr.get(); p->setPlayHead (e.hostPlayHead()); e.addProcessor (0, std::move (pr), "pos");
        e.transport.setLoop (true, 0.0, 4.0); e.transport.play(); run (e, 900, bs);
        // every block a plug-in is given must end at or before the loop end, and the block after a wrap must start exactly at the loop start: no block runs on past the loop end
        const double bps = 121.0 / 60.0 / sr; const auto& v = p->blocks; double maxEnd = 0; int wraps = 0; bool wrapStartsAtLoopStart = true, wrapEndsAtLoopEnd = true; long long total = 0; for (const auto& bl : v) total += bl.second;
        for (size_t i = 0; i < v.size(); ++i)
        {
            const double endPpq = v[i].first + (double) v[i].second * bps; maxEnd = std::max (maxEnd, endPpq);
            if (i > 0 && v[i].first < 1.0e-3 && v[i - 1].first > 3.0) { ++wraps; wrapStartsAtLoopStart = wrapStartsAtLoopStart && std::abs (v[i].first) < 1.0e-6; wrapEndsAtLoopEnd = wrapEndsAtLoopEnd && std::abs ((v[i - 1].first + (double) v[i - 1].second * bps) - 4.0) < 2.0 * bps; }
        }
        CHECK (wraps == 2 && maxEnd < 4.0 + 2.0 * bps, "a plug-in is told the loop wrap as it happens: no block it is given runs on past the loop end");
        CHECK (wrapStartsAtLoopStart && wrapEndsAtLoopEnd, "the piece before the wrap ends exactly at the loop end (within a sample) and the piece after starts exactly at the loop start, so the first note after the wrap is not cut off");
        CHECK (total == 900LL * bs, "and the pieces add up to exactly the blocks the sound card asked for (no sample lost or repeated)");
        CHECK (maxEnd <= 4.0, "and not even by a fraction of a sample: a self-sequencing plug-in would play the step ON the loop end as well as the one at the loop start (the first note after a wrap sounded twice)");
        for (const double le2 : { 0.25, 7.75, 9.25, 32.25, 33.0625 })      // loop ends on sixteenth-note positions (the snap setting of the ruler): the playhead must come back every time
        {
            HostEngine e2; e2.prepareOffline (sr, bs, 0); e2.transport.setBpm (140.0); auto pr2 = std::make_unique<PosProbe>(); auto* p2 = pr2.get(); p2->setPlayHead (e2.hostPlayHead()); e2.addProcessor (0, std::move (pr2), "pos");
            e2.transport.setLoop (true, 0.0, le2 < 1.0 ? 1.0 : le2); e2.transport.play(); run (e2, 4000, bs);
            const double end2 = le2 < 1.0 ? 1.0 : le2; double mx = 0; for (const auto& bl : p2->blocks) mx = std::max (mx, bl.first + (double) bl.second * 140.0 / 60.0 / sr);
            CHECK (mx <= end2 + 1.0e-9 && e2.transport.posBeats.load() <= end2 + 1.0e-9, "a loop that ends on a sixteenth position keeps the playhead inside the loop");
        }
    }

    // ---- AUTOMATION LEARN: move a control in a plug-in and its parameter gets a lane ----
    {
        struct LearnProbe : public juce::AudioProcessor
        {
            LearnProbe() : juce::AudioProcessor (BusesProperties().withOutput ("out", juce::AudioChannelSet::stereo(), true))
            {
                for (int i = 0; i < 6; ++i) { auto* q = new juce::AudioParameterFloat (juce::ParameterID { "p" + juce::String (i), 1 }, "Param " + juce::String (i), 0.0f, 1.0f, 0.5f); addParameter (q); pars.push_back (q); }
                auto* na = new juce::AudioParameterFloat (juce::ParameterID { "fixed", 1 }, "Fixed", juce::NormalisableRange<float> (0.0f, 1.0f), 0.5f, juce::AudioParameterFloatAttributes().withAutomatable (false)); addParameter (na); pars.push_back (na);
            }
            const juce::String getName() const override { return "LearnProbe"; }
            void prepareToPlay (double, int) override {}
            void releaseResources() override {}
            void processBlock (juce::AudioBuffer<float>& b, juce::MidiBuffer&) override { b.clear(); }
            double getTailLengthSeconds() const override { return 0; }
            bool acceptsMidi() const override { return false; }
            bool producesMidi() const override { return false; }
            juce::AudioProcessorEditor* createEditor() override { return nullptr; }
            bool hasEditor() const override { return false; }
            int getNumPrograms() override { return 1; }
            int getCurrentProgram() override { return 0; }
            void setCurrentProgram (int) override {}
            const juce::String getProgramName (int) override { return {}; }
            void changeProgramName (int, const juce::String&) override {}
            void getStateInformation (juce::MemoryBlock&) override {}
            void setStateInformation (const void*, int) override {}
            std::vector<juce::AudioParameterFloat*> pars;
        };
        auto laneCount = [] (HostEngine& e) { return (int) e.trackAt (0)->autoSnap.load()->lanes.size(); };
        { HostEngine e; e.prepareOffline (sr, bs, 0); auto pr = std::make_unique<LearnProbe>(); auto* p = pr.get(); e.addProcessor (0, std::move (pr), "probe");
          for (int i = 0; i < 8; ++i) e.service();
          CHECK (! e.learn.found && laneCount (e) == 0, "nothing is learned until Learn is started, even if something changes");
          e.startAutoLearn (0); CHECK (e.learn.chain == 0, "Learn starts on a channel");
          auto wobble = [&] (int n) { for (int i = 0; i < n; ++i) { p->pars[1]->setValueNotifyingHost (i % 2 ? 0.2f : 0.9f); e.service(); } };       // parameter 1 moves by itself, the whole time
          wobble (8);
          CHECK (! e.learn.found, "a parameter that moves by itself is ignored");
          p->pars[6]->setValueNotifyingHost (0.9f); wobble (2);
          CHECK (! e.learn.found && laneCount (e) == 0, "a parameter the host may not automate is ignored");
          p->pars[3]->setValueNotifyingHost (0.8f); wobble (2);
          CHECK (e.learn.found && e.learn.foundSlot == 0 && e.learn.foundParam == 3 && laneCount (e) == 1, "moving a control makes the lane for exactly that parameter");
          const auto lane = e.trackAt (0)->autoSnap.load()->lanes[0]; CHECK (lane.kind == 2 && lane.slot == 0 && lane.param == 3 && lane.label.contains ("Param 3"), "it is a plug-in parameter lane with the parameter's name");
          CHECK (e.learn.chain < 0, "and Learn switches itself off");
          const auto st = e.snapshot(); const auto al = st.getProperty ("autoLearn", {}); CHECK ((bool) al.getProperty ("found", false) && (int) al.getProperty ("param", -1) == 3 && ! (bool) al.getProperty ("on", true), "the screen is told which parameter was found");
          e.startAutoLearn (0); for (int i = 0; i < 6; ++i) { p->pars[1]->setValueNotifyingHost (i % 2 ? 0.2f : 0.9f); e.service(); } p->pars[3]->setValueNotifyingHost (0.1f); e.service(); e.service();
          CHECK (laneCount (e) == 1, "learning a parameter that already has a lane does not make a second one"); }
        { HostEngine e; e.prepareOffline (sr, bs, 0); auto pr = std::make_unique<LearnProbe>(); e.addProcessor (0, std::move (pr), "probe"); e.startAutoLearn (0);
          for (int i = 0; i < 700; ++i) e.service();
          CHECK (e.learn.timedOut && e.learn.chain < 0 && laneCount (e) == 0, "it gives up after about 20 seconds with nothing moved");
          e.startAutoLearn (kMasterChain); CHECK (e.learn.chain < 0, "the master has no lanes, so Learn does not start there"); }
    }

    // ---- RECORD AUTOMATION: moves made while the song plays are written into lanes ----
    {
        struct RecProbe : public juce::AudioProcessor
        {
            RecProbe() : juce::AudioProcessor (BusesProperties().withOutput ("out", juce::AudioChannelSet::stereo(), true))
            { for (int i = 0; i < 4; ++i) { auto* q = new juce::AudioParameterFloat (juce::ParameterID { "p" + juce::String (i), 1 }, "Param " + juce::String (i), 0.0f, 1.0f, 0.5f); addParameter (q); pars.push_back (q); } }
            const juce::String getName() const override { return "RecProbe"; }
            void prepareToPlay (double, int) override {}
            void releaseResources() override {}
            void processBlock (juce::AudioBuffer<float>& b, juce::MidiBuffer&) override { b.clear(); }
            double getTailLengthSeconds() const override { return 0; }
            bool acceptsMidi() const override { return false; }
            bool producesMidi() const override { return false; }
            juce::AudioProcessorEditor* createEditor() override { return nullptr; }
            bool hasEditor() const override { return false; }
            int getNumPrograms() override { return 1; }
            int getCurrentProgram() override { return 0; }
            void setCurrentProgram (int) override {}
            const juce::String getProgramName (int) override { return {}; }
            void changeProgramName (int, const juce::String&) override {}
            void getStateInformation (juce::MemoryBlock&) override {}
            void setStateInformation (const void*, int) override {}
            std::vector<juce::AudioParameterFloat*> pars;
        };
        auto lanesOf = [] (HostEngine& e, int chain) { return e.chainAt (chain)->autoSnap.load()->lanes; };
        { HostEngine e; e.prepareOffline (sr, bs, 0); auto pr = std::make_unique<RecProbe>(); auto* p = pr.get(); e.addProcessor (0, std::move (pr), "probe");
          e.transport.play(); run (e, 4, bs);
          p->pars[2]->beginChangeGesture(); p->pars[2]->setValueNotifyingHost (0.3f); e.service(); p->pars[2]->endChangeGesture(); e.service();
          CHECK (lanesOf (e, 0).empty(), "with Record automation off, moving a control writes nothing");
          e.editChainParam (0, "armed", true); e.transport.setRecording (true); e.service();
          e.transport.stop(); e.service(); p->pars[1]->setValueNotifyingHost (0.9f); e.service(); e.service();
          CHECK (lanesOf (e, 0).empty(), "a control moved while the song is stopped is not recorded"); e.transport.play(); run (e, 4, bs); e.service(); e.service();
          p->pars[2]->beginChangeGesture(); e.service();
          for (int i = 0; i < 5; ++i) { run (e, 120, bs); p->pars[2]->setValueNotifyingHost (0.2f + 0.12f * (float) i); e.service(); }
          p->pars[2]->endChangeGesture(); e.service();
          juce::Thread::sleep (50); for (int i = 0; i < 20; ++i) e.service();       // the hold runs out ~0.3 s after the grab is let go, then the take is closed
          const auto ln = lanesOf (e, 0); CHECK (ln.size() == 1 && ln[0].kind == 2 && ln[0].slot == 0 && ln[0].param == 2, "moving a held control while the song plays makes the lane for that parameter");
          if (! ln.empty()) { const auto& L = ln[0]; bool sorted = true; for (size_t i = 1; i < L.pts.size(); ++i) sorted = sorted && L.pts[i].beat > L.pts[i - 1].beat;
            CHECK (L.pts.size() >= 4 && sorted, "with a point for each stretch of the move, in time order"); CHECK (L.pts.front().value >= 0.19 && L.pts.front().value <= 0.31 && std::abs (L.pts.back().value - 0.68) < 0.02, "holding the values that were played");
            CHECK (L.pts.back().beat > L.pts.front().beat + 0.5, "spread over the part of the song that played while the control was moved"); }
          const int before = (int) lanesOf (e, 0)[0].pts.size(); e.transport.stop(); e.service(); e.undo.undo (); CHECK ((int) lanesOf (e, 0).size() == 1 && lanesOf (e, 0)[0].pts.empty() && before > 0, "one undo takes the whole take back (the empty lane stays)");
          e.transport.stop(); e.service(); }
        { HostEngine e; e.prepareOffline (sr, bs, 0); auto pr = std::make_unique<RecProbe>(); auto* p = pr.get(); e.addProcessor (0, std::move (pr), "probe");
          e.editChainParam (0, "armed", true); e.transport.setRecording (true); e.transport.play(); run (e, 4, bs); e.service();
          for (int i = 0; i < 6; ++i) { run (e, 100, bs); { std::thread th ([p, i] { p->pars[1]->setValueNotifyingHost (0.1f + 0.15f * (float) i); }); th.join(); } e.service(); }       // a plug-in screen that never reports a grab, moved from its own thread
          juce::Thread::sleep (50); for (int i = 0; i < 20; ++i) e.service();
          auto ln = lanesOf (e, 0); CHECK (ln.size() == 1 && ln[0].kind == 2 && ln[0].param == 1 && ln[0].pts.size() >= 4, "a control moved by hand on an armed track is drawn into its lane even when the plug-in reports no grab"); e.transport.stop(); e.service(); }
        { HostEngine e; e.prepareOffline (sr, bs, 0); auto pr = std::make_unique<RecProbe>(); auto* p = pr.get(); e.addProcessor (0, std::move (pr), "probe");
          e.editChainParam (0, "armed", true); e.transport.setRecording (true); e.transport.play(); run (e, 4, bs); e.service();
          for (int i = 0; i < 5; ++i) { run (e, 60, bs); p->pars[1]->setValueNotifyingHost (0.1f + 0.1f * (float) i); e.service(); }       // a ramp up to 0.5, then the hand stops
          for (int i = 0; i < 12; ++i) { run (e, 120, bs); e.service(); }                                                                    // a long pause: no timer lets the old line pull it back
          auto ln = lanesOf (e, 0); CHECK (std::abs (p->pars[1]->get() - 0.5f) < 0.01f, "a control you let go of keeps its last position while the song goes on");
          CHECK (ln.size() == 1 && ! ln[0].pts.empty() && ln[0].pts.back().beat > ln[0].pts.front().beat + 3.0 && std::abs (ln[0].pts.back().value - 0.5) < 0.01, "and the lane holds that last position on to where the song has got to (not back to where it began)");
          const double lastBeat = e.transport.posBeats.load(); e.transport.locate (0.0); run (e, 4, bs); e.service(); e.service();
          CHECK (lastBeat > 3.0 && std::abs (p->pars[1]->get() - 0.1f) < 0.15f, "looping back to before the edit lets the lane play again"); e.transport.stop(); e.service(); }
        // EMPTY START: the program begins with the Transport and the Master only; every new track has a colour of its own
        { HostEngine e; e.prepareOffline (sr, bs, 0); e.startEmpty();
          CHECK (e.trackCount.load() == 0, "the program starts with no tracks (the Transport and the Master only)");
          e.transport.play(); run (e, 8, bs); e.transport.stop(); CHECK (true, "and plays (silence) with no tracks without trouble");
          CHECK (e.addTrack() && e.addTrack() && e.trackCount.load() == 2, "tracks are added when asked for");
          CHECK (e.trackAt (0)->colour.load() >= 0 && e.trackAt (1)->colour.load() >= 0 && e.trackAt (0)->colour.load() != e.trackAt (1)->colour.load(), "each new track has its own colour from the start");
          CHECK (e.editRemoveTrack (0) && e.editRemoveTrack (0) && e.trackCount.load() == 0, "and the last track can be removed too");
          bool done = false; e.loadProject (e.blankProject(), [&] (juce::String) { done = true; }); for (int i = 0; i < 50 && ! done; ++i) { e.service(); juce::Thread::sleep (10); }
          CHECK (done && e.trackCount.load() == 0, "New gives a song with no tracks"); }
        // SENDS AS ON A CONSOLE: nothing reaches an FX return until the send is turned up; POST follows the fader, PRE does not; the level scales what the effect gets
        {
            // a steady 0.25 tone on the track; the FX return holds an effect that passes the sound on unchanged, so what the effect receives shows up as a plain addition to the dry sound
            auto level = [&] (float sendDb, bool pre, float faderDb, bool turnSendOn) {
                HostEngine e; e.prepareOffline (sr, bs, 0); e.trackAt (0)->source.store (2); e.addProcessor (kReturnBase, std::make_unique<LatProc> (2, 2, 0), "reverb on return 1");
                e.editChainParam (0, "gainDb", (double) faderDb); if (turnSendOn) e.setSend (*e.trackAt (0), 0, sendDb, pre);
                auto o = run (e, 200, bs); float m = 0; for (size_t i = o.size() - 4096; i < o.size(); ++i) m = std::max (m, std::abs (o[i])); return m;       // after the fader has settled
            };
            CHECK (std::abs (level (0.0f, false, 0.0f, false) - 0.25f) < 2e-3f, "with the send off, nothing reaches the effect on the FX return: only the dry sound (0.25)");
            CHECK (std::abs (level (0.0f, false, 0.0f, true) - 0.5f) < 3e-3f, "a send turned up to 0 dB adds what the effect returns to the dry sound (0.25 + 0.25)");
            CHECK (std::abs (level (-6.0f, false, 0.0f, true) - 0.25f * (1.0f + 0.5012f)) < 3e-3f, "a send at -6 dB adds half as much");
            const float post = level (0.0f, false, -60.0f, true), pre = level (0.0f, true, -60.0f, true);
            std::printf ("  fader at -60 dB: POST send total %.5f, PRE send total %.5f\n", post, pre);
            CHECK (post < 0.002f, "POST: the send follows the fader, so a track pulled right down sends (almost) nothing to the effect");
            CHECK (std::abs (pre - 0.25f) < 3e-3f, "PRE: the send is taken before the fader, so the effect still gets the full sound while the dry track is pulled down");
        }
        // SEND ON SWITCH: every send has an ON switch of its own, apart from its level. Nothing is heard unless it is ON; the level is kept while it is off; one track's send never touches another's
        {
            auto peakOf = [&] (HostEngine& e) { auto o = run (e, 200, bs); float m = 0; for (size_t i = o.size() - 4096; i < o.size(); ++i) m = std::max (m, std::abs (o[i])); return m; };      // after the faders have settled
            auto make = [&] (HostEngine& e, int tracks) { e.prepareOffline (sr, bs, 0); for (int i = 0; i < tracks; ++i) e.trackAt (i)->source.store (2); e.addProcessor (kReturnBase, std::make_unique<LatProc> (2, 2, 0), "effect on return 1"); };
            { HostEngine e; make (e, 1); CHECK (! e.trackAt (0)->sendOn[0].load() && e.trackAt (0)->sendProc[0] == nullptr, "a new track's sends are all OFF"); CHECK (std::abs (peakOf (e) - 0.25f) < 2e-3f, "and nothing reaches the effect: only the dry sound (0.25)"); }
            { HostEngine e; make (e, 1); e.setSendLevel (*e.trackAt (0), 0, 0.0f); CHECK (std::abs (peakOf (e) - 0.25f) < 2e-3f && ! e.trackAt (0)->sendOn[0].load(), "turning the send's LEVEL up does not switch it on: still nothing reaches the effect"); }
            { HostEngine e; make (e, 1); e.setSendLevel (*e.trackAt (0), 0, 0.0f); e.setSendOn (*e.trackAt (0), 0, true); CHECK (std::abs (peakOf (e) - 0.5f) < 3e-3f, "with the send ON at 0 dB the effect adds its sound (0.25 + 0.25)"); }
            { HostEngine e; make (e, 1); e.setSendLevel (*e.trackAt (0), 0, -6.0f); e.setSendOn (*e.trackAt (0), 0, true); e.setSendOn (*e.trackAt (0), 0, false); const float off = peakOf (e);
              e.setSendOn (*e.trackAt (0), 0, true); const float on = peakOf (e);
              CHECK (std::abs (off - 0.25f) < 2e-3f && std::abs (on - 0.25f * (1.0f + 0.5012f)) < 3e-3f && std::abs (e.trackAt (0)->sendLvl[0].load() + 6.0f) < 1e-4f, "switching a send off and on again brings it back at the level it had (-6 dB)"); }
            {   // one track's send never touches another's
                HostEngine e; make (e, 2); e.setSendLevel (*e.trackAt (0), 0, 0.0f); e.setSendOn (*e.trackAt (0), 0, true); const float one = peakOf (e);                     // track 1 on, track 2 off: 0.25 + 0.25 + 0.25
                e.setSendLevel (*e.trackAt (1), 0, 6.0f); const float other = peakOf (e);                                                                                           // track 2's level moved while its send is off
                CHECK (std::abs (one - 0.75f) < 4e-3f && std::abs (other - one) < 2e-3f, "moving one track's send level changes nothing for any other track (its send is off, so nothing reaches the effect)");
                e.setSendOn (*e.trackAt (1), 0, true); const float both = peakOf (e); CHECK (std::abs (both - (0.5f + 0.25f + 0.25f * 1.9953f)) < 6e-3f, "and each track's send adds only its own amount");
            }
            {   // PRE and POST still work on top of ON
                HostEngine e; make (e, 1); e.editChainParam (0, "gainDb", -60.0); e.setSendOn (*e.trackAt (0), 0, true); e.setSendLevel (*e.trackAt (0), 0, 0.0f); e.setSendPre (*e.trackAt (0), 0, false); const float post = peakOf (e);
                e.setSendPre (*e.trackAt (0), 0, true); const float pre = peakOf (e); CHECK (post < 0.002f && std::abs (pre - 0.25f) < 3e-3f, "POST follows the fader, PRE does not (switching PRE and POST while the send is ON)");
            }
            {   // undo, redo, and the song file
                HostEngine e; make (e, 1); e.editSendLevel (0, 2, -9.0f); e.editSendOn (0, 2, true); e.editSendPre (0, 2, true);
                CHECK (e.trackAt (0)->sendOn[2].load() && e.trackAt (0)->sendPre[2].load() && std::abs (e.trackAt (0)->sendLvl[2].load() + 9.0f) < 1e-4f, "level, ON and PRE are set");
                e.undo.undo (); CHECK (! e.trackAt (0)->sendPre[2].load() && e.trackAt (0)->sendOn[2].load(), "undo takes PRE back");
                e.undo.undo (); CHECK (! e.trackAt (0)->sendOn[2].load() && e.trackAt (0)->sendProc[2] == nullptr, "undo switches the send off again");
                e.undo.redo (); e.undo.redo (); CHECK (e.trackAt (0)->sendOn[2].load() && e.trackAt (0)->sendPre[2].load(), "redo brings them back");
                auto song = e.saveProject (); HostEngine f; f.prepareOffline (sr, bs, 0); bool done = false; f.loadProject (song, [&] (juce::String) { done = true; }); for (int i = 0; i < 400 && ! done; ++i) { f.service(); juce::Thread::sleep (10); }
                CHECK (done && f.trackAt (0)->sendOn[2].load() && f.trackAt (0)->sendPre[2].load() && std::abs (f.trackAt (0)->sendLvl[2].load() + 9.0f) < 1e-4f, "a saved song keeps each send's ON, level and PRE");
                e.editSendOn (0, 2, false); auto song2 = e.saveProject (); HostEngine g; g.prepareOffline (sr, bs, 0); done = false; g.loadProject (song2, [&] (juce::String) { done = true; }); for (int i = 0; i < 400 && ! done; ++i) { g.service(); juce::Thread::sleep (10); }
                CHECK (done && ! g.trackAt (0)->sendOn[2].load() && std::abs (g.trackAt (0)->sendLvl[2].load() + 9.0f) < 1e-4f, "and a send that is off keeps its level through a save");
                // an older song had only a level: above off means on, at that level
                auto old = e.saveProject (); if (auto* tr = old.getProperty ("tracks", {}).getArray()) if (tr->size() > 0) if (auto* sd = (*tr)[0].getProperty ("sends", {}).getArray()) if (sd->size() > 3) { if (auto* o = (*sd)[3].getDynamicObject()) { o->removeProperty ("on"); o->removeProperty ("lvl"); o->setProperty ("db", -12.0); o->setProperty ("pre", false); } }
                HostEngine h; h.prepareOffline (sr, bs, 0); done = false; h.loadProject (old, [&] (juce::String) { done = true; }); for (int i = 0; i < 400 && ! done; ++i) { h.service(); juce::Thread::sleep (10); }
                CHECK (done && h.trackAt (0)->sendOn[3].load() && std::abs (h.trackAt (0)->sendLvl[3].load() + 12.0f) < 1e-4f, "an older song (a level only) opens with that send ON at its level");
            }
        }
        // SEND AUTOMATION: a send's level has a lane of its own (one lane per track per send); it moves only that send; knob moves are recorded into it
        {
            auto lanesOf2 = [] (HostEngine& e, int chain) { return e.chainAt (chain)->autoSnap.load()->lanes; };
            HostEngine e; e.prepareOffline (sr, bs, 0);
            const int lane = e.editAutoAddLane (0, 4, 1, -1);
            auto ls = lanesOf2 (e, 0);
            CHECK (lane != 0 && ls.size() == 1 && ls[0].kind == 4 && ls[0].slot == 1 && ls[0].label == "Send 2 level", "a send's level can have an automation lane (Send 2 level)");
            CHECK (e.editAutoAddLane (0, 4, 1, -1) == lane && lanesOf2 (e, 0).size() == 1, "asking again for the same send gives the same lane");
            CHECK (e.editAutoAddLane (0, 4, 8, -1) == 0 && e.editAutoAddLane (kReturnBase, 4, 0, -1) == 0, "there is no lane for a send that does not exist, nor for an FX return (sends belong to tracks)");
            juce::Array<juce::var> pts; auto pt = [&] (double b, double v) { auto* o = new juce::DynamicObject(); o->setProperty ("beat", b); o->setProperty ("value", v); pts.add (juce::var (o)); };
            pt (0.0, -60.0); pt (4.0, 0.0);
            CHECK (e.editAutoSetPoints (0, lane, juce::var (pts)), "its line is drawn: from off (-60 dB) at the start to 0 dB at beat 4");
            e.setSendOn (*e.trackAt (0), 1, true); e.setSendLevel (*e.trackAt (0), 0, -20.0f); e.setSendLevel (*e.trackAt (0), 2, -7.0f);
            e.transport.play(); run (e, 190, bs); e.service();                      // about 1 s = 2 beats at 120 bpm
            const float mid = e.trackAt (0)->sendLvl[1].load(); std::printf ("  send 2 level after 2 beats: %.2f dB\n", mid);
            CHECK (mid > -34.0f && mid < -26.0f, "while the song plays, the send's level follows its lane (about -30 dB two beats in)");
            CHECK (std::abs (e.trackAt (0)->sendLvl[0].load() + 20.0f) < 1e-4f && std::abs (e.trackAt (0)->sendLvl[2].load() + 7.0f) < 1e-4f, "and no other send is touched: sends 1 and 3 stay where they were");
            e.transport.stop(); e.service();
            // recording a knob move into a send's lane
            HostEngine r; r.prepareOffline (sr, bs, 0); r.editChainParam (0, "armed", true); r.setSendOn (*r.trackAt (0), 3, true); r.transport.setRecording (true); r.transport.play(); run (r, 6, bs); r.service();
            r.editSendLevel (0, 3, -12.0f); run (r, 40, bs); r.editSendLevel (0, 3, -6.0f); r.service();
            auto rl = lanesOf2 (r, 0); bool ok = rl.size() == 1 && rl[0].kind == 4 && rl[0].slot == 3 && rl[0].pts.size() >= 2;
            CHECK (ok && std::abs (rl[0].pts.front().value + 12.0) < 1e-3 && std::abs (rl[0].pts.back().value + 6.0) < 1e-3, "recording: moving a send knob while the song plays writes that send's lane (and only that one)");
            r.transport.stop(); r.service();
        }
        // A CLIP THAT LOOPS FOR EVER inside a song loop of the same length: every pass plays the same notes, no more (a note that was not there must not sound when the loop comes round)
        {
            struct Ev3 { long long smp; bool on; int note; };
            struct Probe3 : TestProc { Probe3 (std::vector<Ev3>* l) : TestProc (2, 2, 0), out (l) {} bool acceptsMidi() const override { return true; }
                void processBlock (juce::AudioBuffer<float>& b, juce::MidiBuffer& m) override { for (const auto meta : m) { auto msg = meta.getMessage(); if (msg.isNoteOn() || msg.isNoteOff()) out->push_back ({ done + meta.samplePosition, msg.isNoteOn(), msg.getNoteNumber() }); } done += b.getNumSamples(); }
                std::vector<Ev3>* out; long long done = 0; };
            for (const double loopEnd : { 16.0, 32.5 }) for (const int bsz : { 64, 256, 480, 1024 })
            {
            HostEngine e; e.prepareOffline (sr, bsz, 0); e.transport.setBpm (140.0);
            auto* c = new juce::DynamicObject(); c->setProperty ("lengthBeats", 16.0); juce::Array<juce::var> arr;
            const double nb[9] = { 0.979958, 3.019958, 4.999958, 6.999958, 8.999958, 11.019958, 12.959958, 13.499958, 14.939958 }, nl[9] = { 0.5, 0.44, 0.5, 0.5, 0.52, 0.54, 0.36, 0.44, 0.44 };
            for (int i = 0; i < 9; ++i) { auto* x = new juce::DynamicObject(); x->setProperty ("beat", nb[i]); x->setProperty ("len", nl[i]); x->setProperty ("note", 67); x->setProperty ("vel", 80); arr.add (juce::var (x)); }
            c->setProperty ("notes", arr); { const int iid = e.editAddInstance (0, 0.0, 16.0, 0.0, juce::var (c), 0); auto* po = new juce::DynamicObject(); po->setProperty ("length", 0.0); e.editSetInstance (iid, juce::var (po)); }      // length 0 = it goes on looping for ever (the infinity sign on the clip)
            std::vector<Ev3> log; e.addProcessor (0, std::make_unique<Probe3> (&log), "probe"); e.transport.setLoop (true, 0.0, loopEnd); e.transport.locate (0.0); e.transport.play();
            const double secPerPass = loopEnd * 60.0 / 140.0; run (e, (int) std::ceil (3.2 * secPerPass * sr / bsz), bsz);
            int ons[3] = { 0, 0, 0 }, offs[3] = { 0, 0, 0 };
            for (auto& q : log) { const int pass = (int) std::floor ((double) q.smp / (secPerPass * sr) + 1.0e-4); if (pass >= 0 && pass < 3) (q.on ? ons : offs)[pass]++; }
            std::printf ("  clip loop: note-ons per pass %d %d %d, note-offs %d %d %d\n", ons[0], ons[1], ons[2], offs[0], offs[1], offs[2]);
            const int want = loopEnd < 20.0 ? 9 : 18; CHECK (ons[0] == want && ons[1] == want && ons[2] == want, "a clip that loops for ever inside a same-length song loop sounds exactly its notes on every pass (9 for a 16-beat loop, 18 for 32.5)");
            CHECK (offs[0] + offs[1] + offs[2] >= 3 * want - 2, "and every note ends");
            std::printf ("  (block size %d)\n", bsz);
            }
        }
        // PATTERN BLOCKS: a block on a track tells a self-sequencing plug-in which pattern to play; the plug-in's own pattern control is set at the block edge, link to the bass off
        {
            struct PatProbe : public juce::AudioProcessor
            {
                PatProbe() : juce::AudioProcessor (BusesProperties().withOutput ("out", juce::AudioChannelSet::stereo(), true))
                { addParameter (link = new juce::AudioParameterBool ({ "drumLink", 1 }, "Drum Link", true)); addParameter (pat = new juce::AudioParameterInt ({ "drumBank", 1 }, "Drum Pattern", 0, 7, 0)); }
                const juce::String getName() const override { return "PatProbe"; }
                void prepareToPlay (double, int) override {} void releaseResources() override {}
                void processBlock (juce::AudioBuffer<float>& b, juce::MidiBuffer&) override { b.clear(); double ppq = -1.0; int pl = 1; if (auto* ph = getPlayHead()) if (auto pos = ph->getPosition()) { if (auto q = pos->getPpqPosition()) ppq = *q; pl = pos->getIsPlaying() ? 1 : 0; } seen.push_back ({ ppq, b.getNumSamples(), pat->get(), link->get() ? 1 : 0, pl }); }
                double getTailLengthSeconds() const override { return 0; } bool acceptsMidi() const override { return true; } bool producesMidi() const override { return false; }
                juce::AudioProcessorEditor* createEditor() override { return nullptr; } bool hasEditor() const override { return false; }
                int getNumPrograms() override { return 1; } int getCurrentProgram() override { return 0; } void setCurrentProgram (int) override {} const juce::String getProgramName (int) override { return {}; } void changeProgramName (int, const juce::String&) override {}
                void getStateInformation (juce::MemoryBlock&) override {} void setStateInformation (const void*, int) override {}
                struct Seen { double ppq; int n, pattern, link, playing; }; std::vector<Seen> seen; juce::AudioParameterBool* link = nullptr; juce::AudioParameterInt* pat = nullptr;
            };
            HostEngine e; e.prepareOffline (sr, bs, 0); e.transport.setBpm (140.0);
            auto pr = std::make_unique<PatProbe>(); auto* pp = pr.get(); pp->setPlayHead (e.hostPlayHead()); e.addProcessor (0, std::move (pr), "pattern probe");
            auto block = [&] (double start, double len, int bank) { auto* c = new juce::DynamicObject(); c->setProperty ("lengthBeats", len); c->setProperty ("patternBank", bank); return e.editAddInstance (0, start, len, 0.0, juce::var (c), 0); };
            CHECK (block (0.0, 4.0, 1) != 0 && block (4.0, 4.0, 3) != 0, "two pattern blocks can be placed on a track (no notes: they only name a pattern)");
            CHECK (e.arrangementVar (*e.trackAt (0)).getProperty ("instances", {}).getArray()->getReference (0).getProperty ("patternBank", -1).operator int() == 1, "the arrangement reports each block's pattern");
            for (int i = 0; i < 4; ++i) e.service();
            e.transport.locate (0.0); e.transport.play(); for (int i = 0; i < 60; ++i) { run (e, 40, bs); e.service(); }
            const double bps = 140.0 / 60.0 / sr; bool firstOk = true, secondOk = true, linkOff = true, gapSilent = true, blocksPlay = true, startsAtZero = true; int seenFirst = 0, seenSecond = 0, seenGap = 0; double at = 0.0, firstOfSecond = -1.0;      // at: where the song is, in beats
            for (const auto& b : pp->seen)
            {
                const double end = at + b.n * bps;
                if (end <= 4.0 - 1.0e-6) { ++seenFirst; firstOk = firstOk && b.pattern == 1; linkOff = linkOff && b.link == 0; blocksPlay = blocksPlay && b.playing == 1; }
                else if (at >= 4.0 - 1.0e-9 && end <= 8.0) { ++seenSecond; secondOk = secondOk && b.pattern == 3; blocksPlay = blocksPlay && b.playing == 1; if (firstOfSecond < 0.0) firstOfSecond = b.ppq; startsAtZero = startsAtZero && b.ppq < 4.0 + 1.0e-6; }
                else if (at >= 8.0 + 1.0e-6) { ++seenGap; gapSilent = gapSilent && b.playing == 0; }
                at = end;
            }
            std::printf ("  pattern blocks: %d blocks in block 1, %d in block 2\n", seenFirst, seenSecond);
            CHECK (seenFirst > 20 && firstOk && linkOff, "while the first block plays the plug-in's pattern is 1 and its link to the bass is off");
            CHECK (seenSecond > 20 && secondOk, "from the second block's edge on it is 3: no block that holds a step of block 2 still has pattern 1");
            CHECK (firstOfSecond >= 0.0 && firstOfSecond < 0.02 && startsAtZero, "the plug-in counts the song from the start of the block it is playing, so each block's pattern begins at its own first step");
            CHECK (blocksPlay && seenGap > 20 && gapSilent, "after the last block (a gap with no block) the plug-in is told the song is not running: no new hit, silence");
            {   // Archie Drums holds its pattern as TWO controls: "Pattern" (A-D) and "Bank" (1-4). A block must set both: A1 = letter A, number 1 ... D4 = letter D, number 4
                struct TwoProbe : public juce::AudioProcessor
                {
                    TwoProbe() : juce::AudioProcessor (BusesProperties().withOutput ("out", juce::AudioChannelSet::stereo(), true))
                    { addParameter (letter = new juce::AudioParameterChoice ({ "pattern", 1 }, "Pattern", juce::StringArray { "A", "B", "C", "D" }, 0)); addParameter (number = new juce::AudioParameterChoice ({ "bank", 1 }, "Bank", juce::StringArray { "1", "2", "3", "4" }, 0)); }
                    const juce::String getName() const override { return "TwoProbe"; }
                    void prepareToPlay (double, int) override {} void releaseResources() override {}
                    void processBlock (juce::AudioBuffer<float>& b, juce::MidiBuffer&) override { b.clear(); seen.push_back ({ b.getNumSamples(), letter->getIndex() * 4 + number->getIndex() }); }
                    double getTailLengthSeconds() const override { return 0; } bool acceptsMidi() const override { return true; } bool producesMidi() const override { return false; }
                    juce::AudioProcessorEditor* createEditor() override { return nullptr; } bool hasEditor() const override { return false; }
                    int getNumPrograms() override { return 1; } int getCurrentProgram() override { return 0; } void setCurrentProgram (int) override {} const juce::String getProgramName (int) override { return {}; } void changeProgramName (int, const juce::String&) override {}
                    void getStateInformation (juce::MemoryBlock&) override {} void setStateInformation (const void*, int) override {}
                    struct Seen { int n, pattern; }; std::vector<Seen> seen; juce::AudioParameterChoice *letter = nullptr, *number = nullptr;
                };
                HostEngine g; g.prepareOffline (sr, bs, 0); g.transport.setBpm (140.0);
                auto pr2 = std::make_unique<TwoProbe>(); auto* tp = pr2.get(); tp->setPlayHead (g.hostPlayHead()); g.addProcessor (0, std::move (pr2), "two-control probe");
                g.autoPatternBlock (0); CHECK (g.patternCountOf (*g.trackAt (0)) == 16, "a plug-in with a Pattern (A-D) and a Bank (1-4) control has sixteen patterns, A1 to D4");
                auto blk = [&] (double start, double len, int bank) { auto* c = new juce::DynamicObject(); c->setProperty ("lengthBeats", len); c->setProperty ("patternBank", bank); return g.editAddInstance (0, start, len, 0.0, juce::var (c), 0); };
                CHECK (g.trackAt (0)->arrSnap.load()->insts.size() == 1 && g.trackAt (0)->arrSnap.load()->insts[0].content->patternBank == 0, "the new track starts with a block on A1");
                { auto* c = new juce::DynamicObject(); c->setProperty ("lengthBeats", 4.0); c->setProperty ("patternBank", 6); g.editContent (g.trackAt (0)->arrSnap.load()->insts[0].content->id, juce::var (c)); }   // 6 = B3
                CHECK (blk (16.0, 4.0, 13) != 0, "a second block (13 = D2) is placed");
                for (int i = 0; i < 4; ++i) g.service();
                g.transport.locate (0.0); g.transport.play(); for (int i = 0; i < 400; ++i) { run (g, 40, bs); g.service(); }      // well past beat 16 (7 s)
                bool sawB3 = false, sawD2 = false, firstIsB3 = ! tp->seen.empty() && tp->seen.front().pattern == 6; for (const auto& s : tp->seen) { sawB3 = sawB3 || s.pattern == 6; sawD2 = sawD2 || s.pattern == 13; }
                CHECK (sawB3 && firstIsB3, "block 6 sets Pattern to B and Bank to 3 (B3) from the first step");
                CHECK (sawD2, "and a block 13 sets Pattern D and Bank 2 (D2) when it is reached");
                g.transport.stop(); g.service();
            }
            {   // Archie Acid 303 holds its sixteen patterns in one control, "Active Bank" (0..15 = A1..D4)
                struct NoSteps : public juce::AudioParameterInt { using juce::AudioParameterInt::AudioParameterInt; int getNumSteps() const override { return juce::AudioProcessor::getDefaultNumParameterSteps(); } };      // a built plug-in states no step count
                struct BankProbe : public juce::AudioProcessor
                {
                    BankProbe() : juce::AudioProcessor (BusesProperties().withOutput ("out", juce::AudioChannelSet::stereo(), true)) { addParameter (bank = new NoSteps ({ "activeBank", 1 }, "Active Bank", 0, 15, 0)); }
                    const juce::String getName() const override { return "BankProbe"; }
                    void prepareToPlay (double, int) override {} void releaseResources() override {}
                    void processBlock (juce::AudioBuffer<float>& b, juce::MidiBuffer&) override { b.clear(); seen.push_back (bank->get()); }
                    double getTailLengthSeconds() const override { return 0; } bool acceptsMidi() const override { return true; } bool producesMidi() const override { return false; }
                    juce::AudioProcessorEditor* createEditor() override { return nullptr; } bool hasEditor() const override { return false; }
                    int getNumPrograms() override { return 1; } int getCurrentProgram() override { return 0; } void setCurrentProgram (int) override {} const juce::String getProgramName (int) override { return {}; } void changeProgramName (int, const juce::String&) override {}
                    void getStateInformation (juce::MemoryBlock&) override {} void setStateInformation (const void*, int) override {}
                    std::vector<int> seen; NoSteps* bank = nullptr;
                };
                HostEngine g; g.prepareOffline (sr, bs, 0); g.transport.setBpm (140.0);
                auto pr3 = std::make_unique<BankProbe>(); auto* bp = pr3.get(); bp->setPlayHead (g.hostPlayHead()); g.addProcessor (0, std::move (pr3), "bank probe");
                g.autoPatternBlock (0); CHECK (g.patternCountOf (*g.trackAt (0)) == 16, "a plug-in whose patterns are one Active Bank control (the Acid 303) has sixteen, A1 to D4, and gets a block on A1");
                { auto* c = new juce::DynamicObject(); c->setProperty ("lengthBeats", 16.0); c->setProperty ("patternBank", 11); g.editContent (g.trackAt (0)->arrSnap.load()->insts[0].content->id, juce::var (c)); }   // 11 = C4
                for (int i = 0; i < 4; ++i) g.service();
                g.transport.locate (0.0); g.transport.play(); for (int i = 0; i < 100; ++i) { run (g, 40, bs); g.service(); }
                bool allC4 = ! bp->seen.empty(); for (size_t i = 1; i < bp->seen.size(); ++i) allC4 = allC4 && bp->seen[i] == 11;
                CHECK (allC4, "a block on C4 sets its Active Bank to 11 and keeps it there");
                g.transport.stop(); g.service();
            }
            {   // a song saved on another computer: its plug-in files are not where they were saved, so they are found again by name and maker
                auto here = juce::File::getSpecialLocation (juce::File::tempDirectory).getChildFile ("Archie Starter Probe.vst3"); here.replaceWithText ("x");
                HostEngine g; juce::PluginDescription known; known.name = "Starter Probe"; known.pluginFormatName = "VST3"; known.manufacturerName = "Ask Archie"; known.fileOrIdentifier = here.getFullPathName(); known.uniqueId = 4242; g.knownPlugins.addType (known);
                juce::PluginDescription saved = known; saved.fileOrIdentifier = "C:/Users/Someone Else/AppData/Local/AskArchie/Plug-ins/Archie Starter Probe.vst3";
                auto* o = new juce::DynamicObject(); o->setProperty ("desc", saved.createXml()->toString()); const juce::var sv (o);
                CHECK (g.savedDesc (sv).fileOrIdentifier == here.getFullPathName(), "a plug-in saved with a file path that is not on this computer is found again by its name and maker");
                saved.fileOrIdentifier = here.getFullPathName(); auto* o2 = new juce::DynamicObject(); o2->setProperty ("desc", saved.createXml()->toString());
                CHECK (g.savedDesc (juce::var (o2)).fileOrIdentifier == here.getFullPathName(), "and one whose file is there is used as saved");
                here.deleteFile();
            }
            {   // a track that is given a pattern plug-in starts with one block (pattern A1, four bars); a track that already has something on it is left alone
                HostEngine g; g.prepareOffline (sr, bs, 0); g.addProcessor (0, std::make_unique<PatProbe>(), "pattern probe"); g.autoPatternBlock (0);
                const auto gi = g.trackAt (0)->arrSnap.load()->insts;
                CHECK (gi.size() == 1 && gi[0].content && gi[0].content->patternBank == 0 && std::abs (gi[0].length - 16.0) < 1e-9 && std::abs (gi[0].start) < 1e-9, "a new track with a pattern plug-in is given a block: pattern A1, from the start, 16 beats");
                g.autoPatternBlock (0); CHECK (g.trackAt (0)->arrSnap.load()->insts.size() == 1, "and it is not given a second one");
                CHECK (g.patternCountOf (*g.trackAt (0)) == 8, "the engine reports how many patterns the plug-in has (8 for this one)");
                HostEngine h2; h2.prepareOffline (sr, bs, 0); h2.addProcessor (0, std::make_unique<LatProc> (2, 2, 0), "plain"); h2.autoPatternBlock (0); CHECK (h2.trackAt (0)->arrSnap.load()->insts.empty(), "a plug-in without pattern controls gets nothing");
            }
            {   // a saved song keeps the blocks
                auto song = e.saveProject (); HostEngine f; f.prepareOffline (sr, bs, 0); bool done = false; f.loadProject (song, [&] (juce::String) { done = true; }); for (int i = 0; i < 100 && ! done; ++i) { f.service(); juce::Thread::sleep (10); }
                const auto ins = f.trackAt (0)->arrSnap.load()->insts; std::printf ("  loaded: done %d, %d instances, banks %d %d\n", (int) done, (int) ins.size(), ins.size() > 0 && ins[0].content ? ins[0].content->patternBank : -9, ins.size() > 1 && ins[1].content ? ins[1].content->patternBank : -9); CHECK (done && ins.size() == 2 && ins[0].content && ins[0].content->patternBank == 1 && ins[1].content && ins[1].content->patternBank == 3 && ins[1].content->notes.empty(), "a saved song keeps its pattern blocks (their patterns and places)");
            }
            e.transport.stop(); e.service();
        }
        // NOTES PLAYED OVER PATTERN BLOCKS stay: through another pattern, through note edits, and in the block and bar they were played in
        {
            HostEngine e; e.prepareOffline (sr, bs, 0);
            auto block = [&] (double start, double len, double contentLen, int bank) { auto* c = new juce::DynamicObject(); c->setProperty ("lengthBeats", contentLen); c->setProperty ("patternBank", bank); return e.editAddInstance (0, start, len, 0.0, juce::var (c), 0); };
            block (0.0, 16.0, 4.0, 0); block (16.0, 4.0, 4.0, 2);
            auto& t = *e.trackAt (0);
            auto play = [&] (double beat, int note) { RecEvent on; on.beat = beat; on.b[0] = 0x90; on.b[1] = (juce::uint8) note; on.b[2] = 100; RecEvent off = on; off.beat = beat + 0.25; off.b[0] = 0x80; off.b[2] = 0; t.pending.push_back (on); t.pending.push_back (off); };
            auto content = [&] (int i) { return t.arrSnap.load()->insts[(size_t) i].content; };
            auto set = [&] (int i, std::function<void (juce::DynamicObject&)> fill) { auto* c = new juce::DynamicObject(); c->setProperty ("lengthBeats", content (i)->lengthBeats); fill (*c); e.editContent (content (i)->id, juce::var (c)); };
            play (9.0, 36); play (17.0, 38); play (25.0, 40); e.commitRecording (t);
            CHECK (t.arrSnap.load()->insts.size() == 3, "a take over two blocks and the empty bar after them leaves both blocks and makes one note clip");
            CHECK (content (0)->patternBank == 0 && content (0)->notes.size() == 1 && std::abs (content (0)->notes[0].beat - 9.0) < 1e-9 && std::abs (content (0)->lengthBeats - 16.0) < 1e-9, "a hit in bar 3 of a block stays in bar 3 and is not repeated on every pass");
            CHECK (content (1)->patternBank == 2 && content (1)->notes.size() == 1 && content (1)->notes[0].note == 38 && std::abs (content (1)->notes[0].beat - 1.0) < 1e-9, "a hit over the second block stays in the second block");
            CHECK (content (2)->patternBank < 0 && content (2)->notes.size() == 1 && std::abs (t.arrSnap.load()->insts[2].start - 24.0) < 1e-9 && std::abs (content (2)->notes[0].beat - 1.0) < 1e-9, "a hit where there is no block gets a note clip of its own");
            set (0, [] (juce::DynamicObject& c) { c.setProperty ("patternBank", 5); });
            CHECK (content (0)->patternBank == 5 && content (0)->notes.size() == 1 && content (0)->notes[0].note == 36, "choosing another pattern keeps the notes played over the block");
            set (0, [] (juce::DynamicObject& c) { c.setProperty ("notes", juce::var (juce::Array<juce::var>())); });
            CHECK (content (0)->patternBank == 5 && content (0)->notes.empty(), "editing the notes (here: deleting them all) leaves it a pattern block");
            e.undo.undo(); CHECK (content (0)->notes.size() == 1, "and undo brings them back");
            juce::String err; CHECK (e.editJoinInstance (t.arrSnap.load()->insts[0].id, err) == 0 && err.contains ("same pattern") && t.arrSnap.load()->insts.size() == 3, "blocks with different patterns are not joined");
            set (0, [] (juce::DynamicObject& c) { c.setProperty ("patternBank", 5); });      // (the undo above also took back the pattern change made moments before: close edits are one step)
            set (1, [] (juce::DynamicObject& c) { c.setProperty ("patternBank", 5); });
            const int joined = e.editJoinInstance (t.arrSnap.load()->insts[0].id, err);
            CHECK (joined != 0 && content (0)->patternBank == 5 && content (0)->notes.size() == 2 && std::abs (content (0)->notes[1].beat - 17.0) < 1e-9, "blocks with the same pattern join into one block holding both blocks' notes");
        }
        {   // a block longer than what it holds: what was already repeating still repeats, and the new hit sounds once
            HostEngine e; e.prepareOffline (sr, bs, 0);
            auto* c = new juce::DynamicObject(); c->setProperty ("lengthBeats", 4.0); c->setProperty ("patternBank", 1); juce::Array<juce::var> ns; auto* n = new juce::DynamicObject(); n->setProperty ("beat", 0.5); n->setProperty ("len", 0.25); n->setProperty ("note", 42); ns.add (juce::var (n)); c->setProperty ("notes", ns);
            e.editAddInstance (0, 4.0, 12.0, 0.0, juce::var (c), 0);
            auto& t = *e.trackAt (0); RecEvent on; on.beat = 14.0; on.b[0] = 0x90; on.b[1] = 36; on.b[2] = 90; RecEvent off = on; off.beat = 14.25; off.b[0] = 0x80; off.b[2] = 0; t.pending.push_back (on); t.pending.push_back (off); e.commitRecording (t);
            const auto k = t.arrSnap.load()->insts[0].content; int old = 0, fresh = 0; double at = -1.0; for (const auto& x : k->notes) { if (x.note == 42) ++old; else { ++fresh; at = x.beat; } }
            CHECK (std::abs (k->lengthBeats - 12.0) < 1e-9 && old == 3 && fresh == 1 && std::abs (at - 10.0) < 1e-9, "the old note still comes round three times and the new one is once, where it was played");
        }
        // THE EQ PANEL: four parametric bands on every channel and on the master, and a spectrum of the sound after them
        {
            auto rmsOf = [] (const std::vector<float>& v, size_t from) { double s = 0; size_t n = 0; for (size_t i = from; i < v.size(); ++i) { s += (double) v[i] * v[i]; ++n; } return std::sqrt (s / (double) std::max<size_t> (1, n)); };
            auto level = [&] (std::function<void (HostEngine&)> setup) { HostEngine e; e.prepareOffline (sr, bs, 0); e.trackAt (0)->source.store (2); setup (e); e.transport.play(); const auto o = run (e, 120, bs); e.transport.stop(); return rmsOf (o, o.size() / 2); };      // the 440 Hz tone, after the EQ
            const double base = level ([] (HostEngine&) {}); auto dbOf = [&] (double v) { return 20.0 * std::log10 (v / base); };
            CHECK (std::abs (level ([] (HostEngine& e) { e.editEq (0, "b1on", 1.0); e.editEq (0, "b1hz", 440.0); e.editEq (0, "b1gain", 0.0); }) - base) < base * 1.0e-3, "a band at 0 dB changes nothing");
            CHECK (std::abs (dbOf (level ([] (HostEngine& e) { e.editEq (0, "b1on", 1.0); e.editEq (0, "b1hz", 440.0); e.editEq (0, "b1gain", 12.0); e.editEq (0, "b1q", 1.0); })) - 12.0) < 0.3, "a bell at 440 Hz with +12 dB lifts a 440 Hz tone by 12 dB");
            CHECK (std::abs (dbOf (level ([] (HostEngine& e) { e.editEq (0, "b2on", 1.0); e.editEq (0, "b2hz", 440.0); e.editEq (0, "b2gain", -12.0); })) + 12.0) < 0.3, "and a cut of 12 dB takes 12 dB off (band 3, also a bell)");
            CHECK (std::abs (dbOf (level ([] (HostEngine& e) { e.editEq (0, "b2on", 1.0); e.editEq (0, "b2hz", 5000.0); e.editEq (0, "b2gain", 12.0); })) ) < 0.5, "a bell far away from the tone leaves it alone");
            CHECK (std::abs (dbOf (level ([] (HostEngine& e) { e.editEq (0, "b0on", 1.0); e.editEq (0, "b0hz", 2000.0); e.editEq (0, "b0gain", 12.0); })) - 12.0) < 1.0, "a low shelf at 2 kHz lifts everything below it (a 440 Hz tone by about 12 dB)");
            CHECK (std::abs (dbOf (level ([] (HostEngine& e) { e.editEq (0, "b3on", 1.0); e.editEq (0, "b3hz", 100.0); e.editEq (0, "b3gain", -12.0); })) + 12.0) < 1.0, "and a high shelf at 100 Hz lowers everything above it");
            CHECK (std::abs (dbOf (level ([] (HostEngine& e) { e.editEq (0, "b1on", 0.0); e.editEq (0, "b1gain", 12.0); }))) < 0.05, "a band that is switched off does nothing, whatever its gain");
            CHECK (std::abs (dbOf (level ([] (HostEngine& e) { e.editEq (200, "b1on", 1.0); e.editEq (200, "b1hz", 440.0); e.editEq (200, "b1gain", -12.0); })) + 12.0) < 0.3, "the master has the same EQ: a 12 dB cut at 440 Hz takes 12 dB off the whole mix");
            { HostEngine e; e.prepareOffline (sr, bs, 0); CHECK (e.editEq (0, "b0gain", 99.0) && std::abs (e.eqGet (0, "b0gain") - 18.0) < 1e-6 && e.editEq (0, "b1q", 0.0) && std::abs (e.eqGet (0, "b1q") - 0.3) < 1e-6 && e.editEq (0, "b2hz", 5.0) && std::abs (e.eqGet (0, "b2hz") - 20.0) < 1e-6, "the settings are held to their ranges: gain 18 dB, Q 0.3, frequency 20 Hz");
              CHECK (! e.editEq (0, "b4on", 1.0) && ! e.editEq (0, "nonsense", 1.0) && ! e.editEq (kReturnBase, "b0on", 1.0), "an unknown setting, a band that does not exist and an FX return (which has no EQ panel) are refused");
              e.editEq (0, "b1gain", 7.0); e.undo.undo (); CHECK (std::abs (e.eqGet (0, "b1gain")) < 1e-6, "an EQ change can be undone"); e.undo.redo (); CHECK (std::abs (e.eqGet (0, "b1gain") - 7.0) < 1e-6, "and done again"); }
            {   // saved with the song: the channel's bands and the master's bands and filters
                HostEngine e; e.prepareOffline (sr, bs, 0); e.editEq (0, "b2on", 1.0); e.editEq (0, "b2hz", 3100.0); e.editEq (0, "b2gain", -4.5); e.editEq (200, "b3on", 1.0); e.editEq (200, "b3gain", 3.0); e.editEq (200, "hpOn", 1.0); e.editEq (200, "hpHz", 35.0); auto song = e.saveProject();
                HostEngine l; l.prepareOffline (sr, bs, 0); bool done = false; l.loadProject (song, [&] (juce::String) { done = true; }); for (int i = 0; i < 100 && ! done; ++i) { l.service(); juce::Thread::sleep (10); }
                CHECK (done && l.eqGet (0, "b2on") == 1.0 && std::abs (l.eqGet (0, "b2hz") - 3100.0) < 1e-3 && std::abs (l.eqGet (0, "b2gain") + 4.5) < 1e-6 && l.eqGet (200, "b3on") == 1.0 && std::abs (l.eqGet (200, "b3gain") - 3.0) < 1e-6 && l.eqGet (200, "hpOn") == 1.0 && std::abs (l.eqGet (200, "hpHz") - 35.0) < 1e-3, "a saved song keeps every channel's bands and the master's bands and filters");
            }
            {   // the spectrum: a 440 Hz tone shows as a peak near 440 Hz
                HostEngine e; e.prepareOffline (sr, bs, 0); e.trackAt (0)->source.store (2); e.transport.play(); e.eqSpectrum (0); run (e, 60, bs); const auto sp = e.eqSpectrum (0); run (e, 20, bs); const auto sp2 = e.eqSpectrum (0);
                int best = -1; double bv = -999; if (auto* a = sp2.getArray()) for (int i = 0; i < a->size(); ++i) if ((double) (*a)[i] > bv) { bv = (double) (*a)[i]; best = i; }
                CHECK (sp.getArray() != nullptr && sp.getArray()->size() == 256 && best >= 111 && best <= 118 && bv > -20.0 && bv < -8.0, "the spectrum has 256 points and the 0.25 tone (-12 dB) peaks at the point for 440 Hz");
                { int flat = 0; if (auto* a = sp2.getArray()) for (int i = 1; i < 60; ++i) if (std::abs ((double) (*a)[i] - (double) (*a)[i - 1]) < 1.0e-9 && (double) (*a)[i] > -99.0) ++flat; CHECK (flat == 0, "and the bass end has no flat shelves: every point is blended between the analysis steps"); }
                e.transport.stop();
            }
        }
        // MUTE AND SOLO LANES: a mute lane silences a track along the song, a solo lane solos it
        {
            auto lanesOf2 = [] (HostEngine& e, int chain) { return e.chainAt (chain)->autoSnap.load()->lanes; };
            auto pts = [] (std::initializer_list<std::pair<double, double>> v) { juce::Array<juce::var> arr; for (auto& q : v) { auto* o = new juce::DynamicObject(); o->setProperty ("beat", q.first); o->setProperty ("value", q.second); arr.add (juce::var (o)); } return juce::var (arr); };
            auto rms = [] (const std::vector<float>& v, size_t a0, size_t a1) { double s = 0; for (size_t i = a0; i < a1 && i < v.size(); ++i) s += (double) v[i] * v[i]; return std::sqrt (s / (double) std::max<size_t> (1, a1 - a0)); };
            HostEngine e; e.prepareOffline (sr, bs, 0); e.transport.setBpm (120.0); e.trackAt (0)->source.store (2);      // a tone, 2 beats per second
            const int ln = e.editAutoAddLane (0, 7, -1, -1); CHECK (ln != 0 && lanesOf2 (e, 0).size() == 1 && lanesOf2 (e, 0)[0].label == "Mute", "a track can have a Mute lane");
            CHECK (e.editAutoAddLane (kReturnBase, 7, -1, -1) == 0, "(tracks only)");
            e.editAutoSetPoints (0, ln, pts ({ { 0.0, 0.0 }, { 2.0, 0.0 }, { 2.05, 1.0 } })); e.transport.locate (0.0); e.transport.play();
            const auto out = run (e, (int) (3.0 * 2.0 * sr / bs / 2.0 * 2.0), bs);      // 6 beats = 3 s
            const size_t beat = (size_t) (sr / 2.0);
            CHECK (rms (out, beat / 2, beat * 3 / 2) > 0.05, "before the lane's first mute the track is heard");
            CHECK (rms (out, beat * 5 / 2, beat * 5) < 1.0e-4, "from beat 2.5 on the Mute lane has silenced it");
            e.transport.stop(); e.service();
            HostEngine s; s.prepareOffline (sr, bs, 0); s.editAddTrack(); const int sl = s.editAutoAddLane (0, 8, -1, -1); CHECK (sl != 0 && lanesOf2 (s, 0)[0].label == "Solo", "a track can have a Solo lane");
            s.editAutoSetPoints (0, sl, pts ({ { 0.0, 1.0 }, { 4.0, 1.0 } })); s.transport.locate (0.0); s.transport.play(); run (s, 10, bs); for (int i = 0; i < 4; ++i) s.service();
            CHECK (s.trackAt (0)->solo.load() && ! s.trackAt (0)->effMute.load() && s.trackAt (1)->effMute.load(), "a Solo lane solos its track: the other track is silenced");
            s.transport.stop(); s.service();
        }
        // THE ARCHIE SOUND (master): drive and a limiter that keeps the mix at or under its ceiling
        {
            auto goertzel = [] (const std::vector<float>& x, size_t from, double hz, double fs) { double w = 2.0 * 3.14159265358979 * hz / fs, c = 2.0 * std::cos (w), s1 = 0, s2 = 0; size_t n = 0; for (size_t i = from; i < x.size(); ++i, ++n) { const double s0 = x[i] + c * s1 - s2; s2 = s1; s1 = s0; } return std::sqrt (std::max (0.0, s1 * s1 + s2 * s2 - c * s1 * s2)) * 2.0 / (double) std::max<size_t> (1, n); };
            auto mk = [&] (std::function<void (HostEngine&)> setup) { auto e = std::make_unique<HostEngine> (); e->prepareOffline (sr, bs, 0); e->trackAt (0)->source.store (2); setup (*e); e->transport.play(); return e; };      // the 440 Hz tone at 0.25
            { auto e = mk ([] (HostEngine&) {}); const auto base = run (*e, 120, bs); auto f = mk ([] (HostEngine& x) { x.editMasterSound ("amOn", true); x.editMasterSound ("amDrive", 0.0); }); const auto o = run (*f, 120, bs);
              double rb = 0, ro = 0; for (size_t i = base.size() / 2; i < base.size(); ++i) { rb += (double) base[i] * base[i]; ro += (double) o[i] * o[i]; }
              CHECK (std::abs (std::sqrt (ro / rb) - 1.0) < 0.002, "with no drive and a signal under the ceiling the Archie sound leaves the level alone"); }
            { auto e = mk ([] (HostEngine& x) { x.editChainParam (0, "gainDb", 12.0); x.editMasterSound ("amOn", true); x.editMasterSound ("amDrive", 0.0); x.editMasterSound ("amCeil", -1.0); x.applyMasterSoundParam ("amOn", true); }); const auto o = run (*e, 200, bs);
              float pk = 0; for (size_t i = 2000; i < o.size(); ++i) pk = std::max (pk, std::abs (o[i])); CHECK (pk <= 0.8914f && pk > 0.85f, "a tone pushed 12 dB over the ceiling is held at the ceiling (-1 dB = 0.891), not over"); }
            { auto e = mk ([] (HostEngine& x) { x.editChainParam (0, "gainDb", 12.0); x.editMasterSound ("amOn", true); x.editMasterSound ("amCeil", -3.0); x.applyMasterSoundParam ("amDrive", 0.0); }); auto o = run (*e, 200, bs); float pk = 0; for (size_t i = 2000; i < o.size(); ++i) pk = std::max (pk, std::abs (o[i])); CHECK (pk <= 0.7080f && pk > 0.65f, "and the ceiling can be moved (-3 dB = 0.708)"); }
            {   // the master level is the input to it: pushed 20 dB up (the master fader), the output still cannot pass the ceiling
                auto e = mk ([] (HostEngine& x) { x.editMasterSound ("amOn", true); x.editMasterSound ("amDrive", 0.5); x.editMasterSound ("amCeil", -1.0); x.master->gainDb.store (12.0f); x.editChainParam (0, "gainDb", 12.0); }); const auto o = run (*e, 200, bs);
                float pk = 0; for (size_t i = 2000; i < o.size(); ++i) pk = std::max (pk, std::abs (o[i])); CHECK (pk <= 0.8914f, "with the master fader and a track both at +12 dB the output is still under the ceiling"); }
            {   // drive adds harmonics: a pure tone gains a third harmonic, and with no drive it has none
                auto d0 = mk ([] (HostEngine& x) { x.editMasterSound ("amOn", true); x.editMasterSound ("amDrive", 0.0); x.editChainParam (0, "gainDb", 6.0); }); const auto a = run (*d0, 200, bs);
                auto d1 = mk ([] (HostEngine& x) { x.editMasterSound ("amOn", true); x.editMasterSound ("amDrive", 1.0); x.editChainParam (0, "gainDb", 6.0); }); const auto bb = run (*d1, 200, bs);
                const double f0a = goertzel (a, 20000, 440.0, sr), h3a = goertzel (a, 20000, 1320.0, sr), f0b = goertzel (bb, 20000, 440.0, sr), h3b = goertzel (bb, 20000, 1320.0, sr), h2b = goertzel (bb, 20000, 880.0, sr);
                CHECK (h3a / f0a < 0.001, "with no drive a pure tone has no third harmonic"); CHECK (h3b / f0b > 0.02 && h2b / f0b > 0.005, "with full drive it gains both a third and a second harmonic (the warmth)"); }
            {   // meters: the limiter's reduction and the average loudness
                auto e = mk ([] (HostEngine& x) { x.editChainParam (0, "gainDb", 12.0); x.editMasterSound ("amOn", true); x.editMasterSound ("amDrive", 0.0); }); run (*e, 100, bs); const auto m = e->meters(); CHECK ((double) m.getProperty ("amGr", 0.0) > 0.5 && (double) m.getProperty ("amGr", 0.0) < 1.6, "the meter reports the limiter turning the mix down (about 1 dB: a tone 12 dB hot peaks at 1.0, the ceiling is 0.891)");
                auto q = mk ([] (HostEngine&) {}); run (*q, 4500, bs); const double avg = (double) q->meters().getProperty ("avg", -100.0); CHECK (std::abs (avg + 15.05) < 1.0, "the average loudness meter reads about -15 dBFS for the 0.25 tone"); }
            {   // saved with the song; an older song (none saved) has it off; a new song has it on
                HostEngine e; e.prepareOffline (sr, bs, 0); e.editMasterSound ("amOn", true); e.editMasterSound ("amDrive", 0.6); e.editMasterSound ("amCeil", -2.0); auto song = e.saveProject();
                HostEngine l; l.prepareOffline (sr, bs, 0); bool done = false; l.loadProject (song, [&] (juce::String) { done = true; }); for (int i = 0; i < 100 && ! done; ++i) { l.service(); juce::Thread::sleep (10); }
                CHECK (done && (bool) l.masterSoundParam ("amOn") && std::abs ((double) l.masterSoundParam ("amDrive") - 0.6) < 1e-6 && std::abs ((double) l.masterSoundParam ("amCeil") + 2.0) < 1e-6, "a saved song keeps the Archie sound's settings");
                auto old = e.saveProject(); if (auto* o = old.getDynamicObject()) o->removeProperty ("masterSound"); HostEngine l2; l2.prepareOffline (sr, bs, 0); l2.editMasterSound ("amOn", true); bool d2 = false; l2.loadProject (old, [&] (juce::String) { d2 = true; }); for (int i = 0; i < 100 && ! d2; ++i) { l2.service(); juce::Thread::sleep (10); }
                CHECK (d2 && ! (bool) l2.masterSoundParam ("amOn"), "a song made before it existed loads with it off, so it sounds as it did");
                HostEngine l3; l3.prepareOffline (sr, bs, 0); bool d3 = false; l3.loadProject (l3.blankProject(), [&] (juce::String) { d3 = true; }); for (int i = 0; i < 100 && ! d3; ++i) { l3.service(); juce::Thread::sleep (10); }
                CHECK (d3 && (bool) l3.masterSoundParam ("amOn") && std::abs ((double) l3.masterSoundParam ("amDrive") - 0.25) < 1e-6, "a new song starts with it on, a little drive and the ceiling at -1 dB");
                CHECK (! e.editMasterSound ("nonsense", 1.0) && e.editMasterSound ("amCeil", 5.0) && std::abs ((double) e.masterSoundParam ("amCeil")) < 1e-6 && e.editMasterSound ("amDrive", -3.0) && (double) e.masterSoundParam ("amDrive") == 0.0, "an unknown key is refused and the settings are held to their ranges");
                e.editMasterSound ("amOn", false); e.undo.undo (); CHECK ((bool) e.masterSoundParam ("amOn"), "a change can be undone"); }
        }
        // THE ARCHIE SOUND, part two: loudness in LUFS, push, bass and tone, and a loudness target
        {
            auto sine = [] (juce::AudioBuffer<float>& b, double hz, double amp, double fs, long long& ph) { for (int i = 0; i < b.getNumSamples(); ++i) { const float v = (float) (amp * std::sin (6.283185307179586 * hz * (double) (ph + i) / fs)); b.setSample (0, i, v); b.setSample (1, i, v); } ph += b.getNumSamples(); };
            auto gainOf = [&] (std::function<void (ArchieSound&, float&)> cfg, double hz) {      // the gain of the sound stage alone at one frequency, dB
                ArchieSound s; s.prepare (sr, bs); juce::AudioBuffer<float> b (2, bs); long long ph = 0; double inSq = 0, outSq = 0;
                for (int blk = 0; blk < 400; ++blk) { sine (b, hz, 0.1, sr, ph); float dummy = 0; cfg (s, dummy); if (blk >= 200) for (int i = 0; i < bs; ++i) inSq += b.getSample (0, i) * (double) b.getSample (0, i); s.process (b, bs, dummy, dummy, dummy, 0.0f, 0.0f); if (blk >= 200) for (int i = 0; i < bs; ++i) outSq += b.getSample (0, i) * (double) b.getSample (0, i); }
                return 10.0 * std::log10 (outSq / inSq); };
            { ArchieSound s; s.prepare (sr, bs); juce::AudioBuffer<float> b (2, bs); long long ph = 0; double inSq = 0, outSq = 0;
              for (int blk = 0; blk < 400; ++blk) { sine (b, 60.0, 0.1, sr, ph); if (blk >= 200) for (int i = 0; i < bs; ++i) inSq += b.getSample (0, i) * (double) b.getSample (0, i); s.process (b, bs, 6.0f, 0.0f, 0.0f, 0.0f, 0.0f); if (blk >= 200) for (int i = 0; i < bs; ++i) outSq += b.getSample (0, i) * (double) b.getSample (0, i); }
              CHECK (std::abs (10.0 * std::log10 (outSq / inSq) - 6.0) < 0.1, "PUSH of 6 dB is 6 dB more level into the limiter"); }
            { ArchieSound s; s.prepare (sr, bs); juce::AudioBuffer<float> b (2, bs); long long ph = 0; double inSq = 0, outSq = 0;
              for (int blk = 0; blk < 400; ++blk) { sine (b, 60.0, 0.1, sr, ph); if (blk >= 200) for (int i = 0; i < bs; ++i) inSq += b.getSample (0, i) * (double) b.getSample (0, i); s.process (b, bs, 0.0f, 6.0f, 0.0f, 0.0f, 0.0f); if (blk >= 200) for (int i = 0; i < bs; ++i) outSq += b.getSample (0, i) * (double) b.getSample (0, i); }
              const double g60 = 10.0 * std::log10 (outSq / inSq); ArchieSound s2; s2.prepare (sr, bs); ph = 0; inSq = outSq = 0;
              for (int blk = 0; blk < 400; ++blk) { sine (b, 1000.0, 0.1, sr, ph); if (blk >= 200) for (int i = 0; i < bs; ++i) inSq += b.getSample (0, i) * (double) b.getSample (0, i); s2.process (b, bs, 0.0f, 6.0f, 0.0f, 0.0f, 0.0f); if (blk >= 200) for (int i = 0; i < bs; ++i) outSq += b.getSample (0, i) * (double) b.getSample (0, i); }
              CHECK (g60 > 4.5 && g60 < 6.5 && std::abs (10.0 * std::log10 (outSq / inSq)) < 0.3, "BASS of 6 dB lifts 60 Hz by about 6 dB and leaves 1 kHz alone"); }
            { ArchieSound s; s.prepare (sr, bs); juce::AudioBuffer<float> b (2, bs); long long ph = 0; double inSq = 0, outSq = 0;
              for (int blk = 0; blk < 400; ++blk) { sine (b, 12000.0, 0.1, sr, ph); if (blk >= 200) for (int i = 0; i < bs; ++i) inSq += b.getSample (0, i) * (double) b.getSample (0, i); s.process (b, bs, 0.0f, 0.0f, 3.0f, 0.0f, 0.0f); if (blk >= 200) for (int i = 0; i < bs; ++i) outSq += b.getSample (0, i) * (double) b.getSample (0, i); }
              const double g12k = 10.0 * std::log10 (outSq / inSq); ArchieSound s2; s2.prepare (sr, bs); ph = 0; inSq = outSq = 0;
              for (int blk = 0; blk < 400; ++blk) { sine (b, 200.0, 0.1, sr, ph); if (blk >= 200) for (int i = 0; i < bs; ++i) inSq += b.getSample (0, i) * (double) b.getSample (0, i); s2.process (b, bs, 0.0f, 0.0f, -3.0f, 0.0f, 0.0f); if (blk >= 200) for (int i = 0; i < bs; ++i) outSq += b.getSample (0, i) * (double) b.getSample (0, i); }
              CHECK (g12k > 2.0 && g12k < 3.3 && std::abs (10.0 * std::log10 (outSq / inSq)) < 0.3, "TONE of +3 dB lifts the top (12 kHz) by about 3 dB and leaves 200 Hz alone"); }
            (void) gainOf;
            {   // loudness: a 0.25 tone in both channels is about -12.5 LUFS (K-weighting at 440 Hz is close to flat); integrated agrees for a steady sound and starts again when asked
                HostEngine e; e.prepareOffline (sr, bs, 0); e.trackAt (0)->source.store (2); e.transport.play(); run (e, 800, bs); const auto m = e.meters();
                const double M = (double) m.getProperty ("lufsM", -100.0), S = (double) m.getProperty ("lufsS", -100.0), I = (double) m.getProperty ("lufsI", -100.0);
                CHECK (std::abs (S + 12.5) < 0.6 && std::abs (M - S) < 0.3 && std::abs (I - S) < 0.3, "the loudness meter reads about -12.5 LUFS for the 0.25 tone: momentary, short-term and integrated agree");
                e.editMasterSound ("amResetLufs", 1.0); run (e, 4, bs); const auto m2 = e.meters(); CHECK ((double) m2.getProperty ("lufsI", 0.0) < -90.0, "and the integrated loudness can be started again");
                HostEngine q; q.prepareOffline (sr, bs, 0); q.transport.play(); run (q, 300, bs); CHECK ((double) q.meters().getProperty ("lufsS", 0.0) < -90.0, "silence reads as nothing, not as a number"); }
            {   // a loudness target: PUSH follows it until the finished mix is on target
                HostEngine e; e.prepareOffline (sr, bs, 0); e.trackAt (0)->source.store (2); e.editMasterSound ("amOn", true); e.editMasterSound ("amDrive", 0.0); e.editMasterSound ("amCeil", -0.5); e.editMasterSound ("amTarget", -8.0); e.transport.play();
                for (int i = 0; i < 4400; ++i) run (e, 1, bs); const auto m = e.meters(); const double S = (double) m.getProperty ("lufsS", -100.0), push = (double) m.getProperty ("amPush", 0.0);
                CHECK (std::abs (S + 8.0) < 1.0 && push > 3.5 && push < 6.0, "with a target of -8 LUFS the push climbs by itself (about 4.5 dB here) until the mix reads -8");
                e.editMasterSound ("amTarget", 0.0); CHECK (std::abs ((double) e.masterSoundParam ("amTarget")) < 1e-9 && ! e.editMasterSound ("amTarget", -8.0) == false, "a target of 0 means no target");
                e.editMasterSound ("amTarget", -3.0); CHECK (std::abs ((double) e.masterSoundParam ("amTarget") + 6.0) < 1e-6, "a target is held between -23 and -6 LUFS"); }
            {   // saved with the song
                HostEngine e; e.prepareOffline (sr, bs, 0); e.editMasterSound ("amPush", 7.5); e.editMasterSound ("amBass", 3.0); e.editMasterSound ("amTone", -1.5); e.editMasterSound ("amTarget", -14.0); auto song = e.saveProject();
                HostEngine l; l.prepareOffline (sr, bs, 0); bool done = false; l.loadProject (song, [&] (juce::String) { done = true; }); for (int i = 0; i < 100 && ! done; ++i) { l.service(); juce::Thread::sleep (10); }
                CHECK (done && std::abs ((double) l.masterSoundParam ("amPush") - 7.5) < 1e-6 && std::abs ((double) l.masterSoundParam ("amBass") - 3.0) < 1e-6 && std::abs ((double) l.masterSoundParam ("amTone") + 1.5) < 1e-6 && std::abs ((double) l.masterSoundParam ("amTarget") + 14.0) < 1e-6, "a saved song keeps push, bass, tone and the loudness target"); }
        }
        // THE ARCHIE SOUND, part three: the 20 Hz cutoff, and the loudness target that caps, settles and holds
        {
            auto gainAt = [&] (double hz) { ArchieSound s; s.prepare (sr, bs); juce::AudioBuffer<float> b (2, bs); long long ph = 0; double inSq = 0, outSq = 0;
                for (int blk = 0; blk < 800; ++blk) { for (int i = 0; i < bs; ++i) { const float v = (float) (0.1 * std::sin (6.283185307179586 * hz * (double) (ph + i) / sr)); b.setSample (0, i, v); b.setSample (1, i, v); } ph += bs; if (blk >= 500) for (int i = 0; i < bs; ++i) inSq += b.getSample (0, i) * (double) b.getSample (0, i); s.process (b, bs, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f); if (blk >= 500) for (int i = 0; i < bs; ++i) outSq += b.getSample (0, i) * (double) b.getSample (0, i); }
                return 10.0 * std::log10 (outSq / inSq); };
            CHECK (gainAt (10.0) < -20.0 && std::abs (gainAt (20.0) + 3.0) < 0.6 && std::abs (gainAt (100.0)) < 0.1, "the 20 Hz cutoff: 10 Hz is more than 20 dB down, 20 Hz is 3 dB down and 100 Hz is untouched");
            {   // settles and holds: with a target it climbs, then stops moving once on target, and says so
                HostEngine e; e.prepareOffline (sr, bs, 0); e.trackAt (0)->source.store (2); e.editMasterSound ("amOn", true); e.editMasterSound ("amDrive", 0.0); e.editMasterSound ("amCeil", -0.5); e.editMasterSound ("amTarget", -8.0); e.transport.play();
                for (int i = 0; i < 6000; ++i) run (e, 1, bs); const double p1 = (double) e.meters().getProperty ("amPush", 0.0);
                for (int i = 0; i < 3000; ++i) run (e, 1, bs); const auto m = e.meters(); const double p2 = (double) m.getProperty ("amPush", 0.0);
                CHECK ((int) m.getProperty ("amState", 0) == 2 && std::abs (p2 - p1) < 0.05, "once the mix has been on target for a few seconds the push holds still and the state says so"); }
            {   // cannot reach: a very quiet mix and a loud target stops at +12 dB and says so
                HostEngine e; e.prepareOffline (sr, bs, 0); e.trackAt (0)->source.store (2); e.editChainParam (0, "gainDb", -20.0); e.editMasterSound ("amOn", true); e.editMasterSound ("amDrive", 0.0); e.editMasterSound ("amTarget", -9.0); e.transport.play();
                for (int i = 0; i < 9000; ++i) run (e, 1, bs); const auto m = e.meters(); const double p = (double) m.getProperty ("amPush", 0.0);
                CHECK (p > 11.9 && p <= 12.01 && (int) m.getProperty ("amState", 0) == 3, "a target that cannot be reached stops the push at +12 dB and says so"); }
            {   // an old song that had the push at 18 comes down to the cap once a target is on
                HostEngine e; e.prepareOffline (sr, bs, 0); e.trackAt (0)->source.store (2); e.editMasterSound ("amOn", true); e.editMasterSound ("amPush", 18.0); e.editMasterSound ("amTarget", -9.0); e.transport.play(); run (e, 800, bs);
                CHECK ((double) e.meters().getProperty ("amPush", 18.0) <= 12.01, "a push of +18 under a target is brought down to the +12 cap"); }
            {   // no target: the push is the knob's and nothing moves it
                HostEngine e; e.prepareOffline (sr, bs, 0); e.trackAt (0)->source.store (2); e.editMasterSound ("amOn", true); e.editMasterSound ("amPush", 5.0); e.transport.play(); run (e, 1500, bs);
                const auto m = e.meters(); CHECK (std::abs ((double) m.getProperty ("amPush", 0.0) - 5.0) < 1e-6 && (int) m.getProperty ("amState", -1) == 0, "with no target the push stays where it is set"); }
        }
        // THE ARCHIE SOUND, part four: the 20 Hz cut is a button, and the master has automation lanes for the Archie sound
        {
            auto pts = [] (std::initializer_list<std::pair<double, double>> v) { juce::Array<juce::var> arr; for (auto& q : v) { auto* o = new juce::DynamicObject(); o->setProperty ("beat", q.first); o->setProperty ("value", q.second); arr.add (juce::var (o)); } return juce::var (arr); };
            { ArchieSound s; s.prepare (sr, bs); juce::AudioBuffer<float> b (2, bs); long long ph = 0; double inSq = 0, outSq = 0;
              for (int blk = 0; blk < 800; ++blk) { for (int i = 0; i < bs; ++i) { const float v = (float) (0.1 * std::sin (6.283185307179586 * 10.0 * (double) (ph + i) / sr)); b.setSample (0, i, v); b.setSample (1, i, v); } ph += bs; if (blk >= 500) for (int i = 0; i < bs; ++i) inSq += b.getSample (0, i) * (double) b.getSample (0, i); s.process (b, bs, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, false); if (blk >= 500) for (int i = 0; i < bs; ++i) outSq += b.getSample (0, i) * (double) b.getSample (0, i); }
              CHECK (10.0 * std::log10 (outSq / inSq) > -1.0, "with the 20 Hz cut switched off, 10 Hz passes"); }
            { HostEngine e; e.prepareOffline (sr, bs, 0);
              const int lp = e.editAutoAddLane (kMasterChain, 9, -1, -1), lc = e.editAutoAddLane (kMasterChain, 15, -1, -1);
              CHECK (lp != 0 && lc != 0 && e.editAutoAddLane (kMasterChain, 9, -1, -1) == lp, "the master can have lanes for the Archie sound (push, and the 20 Hz cut here); asking again gives the same lane");
              CHECK (e.editAutoAddLane (0, 9, -1, -1) == 0 && e.editAutoAddLane (kMasterChain, 7, -1, -1) == 0 && e.editAutoAddLane (kMasterChain, 0, -1, -1) == 0 && e.editAutoAddLane (kReturnBase, 10, -1, -1) == 0, "a track cannot have a master lane, and the master cannot have a track lane");
              e.editAutoSetPoints (kMasterChain, lp, pts ({ { 0.0, 0.0 }, { 4.0, 12.0 } })); e.editAutoSetPoints (kMasterChain, lc, pts ({ { 0.0, 1.0 }, { 2.0, 1.0 }, { 2.05, 0.0 } }));
              e.editMasterSound ("amOn", true); e.transport.setBpm (120.0); e.transport.locate (0.0); e.transport.play(); for (int i = 0; i < 420; ++i) { run (e, 1, bs); if (i % 4 == 0) e.service(); }
              CHECK ((double) e.masterSoundParam ("amPush") > 9.5 && ! (bool) e.masterSoundParam ("amCut"), "playing, the push lane climbs along the song and the cut lane switches the 20 Hz cut off");
              e.transport.stop(); e.service();
              auto song = e.saveProject(); HostEngine l; l.prepareOffline (sr, bs, 0); bool done = false; l.loadProject (song, [&] (juce::String) { done = true; }); for (int i = 0; i < 100 && ! done; ++i) { l.service(); juce::Thread::sleep (10); }
              CHECK (done && l.chainAt (kMasterChain)->autoSnap.load()->lanes.size() == 2, "a saved song keeps the master's lanes"); }
            { HostEngine e; e.prepareOffline (sr, bs, 0); e.editMasterSound ("amTarget", -14.0); const int lp = e.editAutoAddLane (kMasterChain, 9, -1, -1); e.editAutoSetPoints (kMasterChain, lp, pts ({ { 0.0, 15.0 }, { 4.0, 15.0 } })); e.transport.play(); for (int i = 0; i < 40; ++i) { run (e, 1, bs); e.service(); }
              CHECK ((double) e.masterSoundParam ("amPush") < 1e-6, "while a loudness target is on, a push lane waits (the target moves the push)"); e.transport.stop(); }
        }
        // EQ AUTOMATION: sweeps on tracks and the master; and every kind of lane survives saving and loading
        {
            auto pts = [] (std::initializer_list<std::pair<double, double>> v) { juce::Array<juce::var> arr; for (auto& q : v) { auto* o = new juce::DynamicObject(); o->setProperty ("beat", q.first); o->setProperty ("value", q.second); arr.add (juce::var (o)); } return juce::var (arr); };
            auto rms = [] (const std::vector<float>& v, size_t a0, size_t a1) { double s = 0; for (size_t i = a0; i < a1 && i < v.size(); ++i) s += (double) v[i] * v[i]; return std::sqrt (s / (double) std::max<size_t> (1, a1 - a0)); };
            { HostEngine e; e.prepareOffline (sr, bs, 0); e.transport.setBpm (120.0);
              const int f = e.editAutoAddLane (0, 16, -1, 5), g = e.editAutoAddLane (0, 16, -1, 6), on = e.editAutoAddLane (0, 16, -1, 4);
              CHECK (f != 0 && g != 0 && on != 0 && f != g && e.editAutoAddLane (0, 16, -1, 5) == f, "each EQ setting can have its own lane (band 2 on, frequency, gain); asking again gives the same one");
              CHECK (e.editAutoAddLane (0, 16, -1, 20) == 0 && e.editAutoAddLane (0, 16, -1, -1) == 0 && e.editAutoAddLane (kReturnBase, 16, -1, 5) == 0 && e.editAutoAddLane (kMasterChain, 16, -1, 17) != 0, "a setting that does not exist and an FX return are refused; the master takes EQ lanes too");
              e.editAutoSetPoints (0, on, pts ({ { 0.0, 1.0 } })); e.editAutoSetPoints (0, f, pts ({ { 0.0, 0.0 }, { 4.0, 1.0 } })); e.editAutoSetPoints (0, g, pts ({ { 0.0, -12.0 }, { 4.0, 12.0 } }));
              e.transport.locate (0.0); e.transport.play(); run (e, 188, bs);      // about 1 s = 2 beats: half way
              CHECK (e.eqGet (0, "b1on") == 1.0 && std::abs (e.eqGet (0, "b1hz") - 632.0) < 40.0 && std::abs (e.eqGet (0, "b1gain")) < 0.8, "half way along, a frequency sweep is at the musical middle (about 632 Hz, not 10 kHz) and the gain is half way too");
              run (e, 220, bs); CHECK (e.eqGet (0, "b1hz") > 19000.0 && std::abs (e.eqGet (0, "b1gain") - 12.0) < 0.1, "and at the end it is at the top: 20 kHz and +12 dB");
              e.transport.stop(); }
            { HostEngine e; e.prepareOffline (sr, bs, 0); e.transport.setBpm (120.0); e.trackAt (0)->source.store (2);      // the 440 Hz tone, swept past by a low-pass going up from 1 kHz and a high-pass rising from 20 Hz to 2 kHz
              const int hon = e.editAutoAddLane (0, 16, -1, 16), hf = e.editAutoAddLane (0, 16, -1, 17); e.editAutoSetPoints (0, hon, pts ({ { 0.0, 1.0 } })); e.editAutoSetPoints (0, hf, pts ({ { 0.0, 0.0 }, { 4.0, 1.0 } }));
              e.transport.locate (0.0); e.transport.play(); const auto o = run (e, 375, bs);
              CHECK (rms (o, 4800, 14400) > 0.15 && rms (o, o.size() - 9600, o.size()) < 0.02, "an automated high-pass sweep is heard: the tone is there at the start and gone once the sweep has passed it");
              e.transport.stop(); }
            { HostEngine e; e.prepareOffline (sr, bs, 0); e.transport.setBpm (120.0); e.trackAt (0)->source.store (2);
              const int on = e.editAutoAddLane (kMasterChain, 16, -1, 4), g = e.editAutoAddLane (kMasterChain, 16, -1, 6), f = e.editAutoAddLane (kMasterChain, 16, -1, 5);
              e.editAutoSetPoints (kMasterChain, on, pts ({ { 0.0, 1.0 } })); e.editAutoSetPoints (kMasterChain, f, pts ({ { 0.0, std::log (440.0 / 20.0) / std::log (1000.0) } })); e.editAutoSetPoints (kMasterChain, g, pts ({ { 0.0, 0.0 }, { 2.0, -12.0 } }));
              e.transport.locate (0.0); e.transport.play(); const auto o = run (e, 375, bs);
              CHECK (rms (o, 1000, 4000) > 2.5 * rms (o, o.size() - 9600, o.size()), "a master EQ lane works too: an automated cut takes the tone down");
              e.transport.stop(); }
            {   // every kind of lane comes back as itself
                HostEngine e; e.prepareOffline (sr, bs, 0); e.editAutoAddLane (0, 4, 1, -1); e.editAutoAddLane (0, 5, -1, -1); e.editAutoAddLane (0, 6, -1, -1); e.editAutoAddLane (0, 7, -1, -1); e.editAutoAddLane (0, 8, -1, -1); e.editAutoAddLane (0, 16, -1, 9);
                e.editAutoAddLane (kMasterChain, 9, -1, -1); e.editAutoAddLane (kMasterChain, 15, -1, -1); e.editAutoAddLane (kMasterChain, 16, -1, 19); auto song = e.saveProject();
                HostEngine l; l.prepareOffline (sr, bs, 0); bool done = false; l.loadProject (song, [&] (juce::String) { done = true; }); for (int i = 0; i < 100 && ! done; ++i) { l.service(); juce::Thread::sleep (10); }
                auto kinds = [] (HostEngine& h, int chain) { std::multiset<std::pair<int, int>> s; for (const auto& ln : h.chainAt (chain)->autoSnap.load()->lanes) s.insert ({ ln.kind, ln.kind == 16 ? ln.param : -1 }); return s; };
                CHECK (done && kinds (l, 0) == kinds (e, 0) && kinds (l, kMasterChain) == kinds (e, kMasterChain), "a saved song brings every lane back as the kind it was (send, pattern, mute, solo, EQ, and the master's)");
                CHECK (HostEngine::kindFromLabel ("Mute") == 7 && HostEngine::kindFromLabel ("Send 3 level") == 4 && HostEngine::kindFromLabel ("20 Hz cut") == 15 && HostEngine::kindFromLabel ("Input gain") == 3, "and a lane saved as the wrong kind is put right from its name");
            }
        }
        // RECORDING EQ AND ARCHIE SOUND MOVES: while the song records, what is moved is written into its lane
        {
            HostEngine e; e.prepareOffline (sr, bs, 0); e.transport.setBpm (120.0);
            e.editEq (0, "b1hz", 500.0); CHECK (e.chainAt (0)->autoSnap.load()->lanes.empty(), "stopped, moving the EQ writes no lane");
            e.transport.play(); run (e, 20, bs); e.editEq (0, "b1hz", 600.0); CHECK (e.chainAt (0)->autoSnap.load()->lanes.empty(), "nor when playing without record");
            e.transport.locate (0.0); e.transport.setRecording (true); e.transport.play();
            const double freqs[] = { 100.0, 200.0, 400.0, 800.0, 1600.0, 3200.0 };
            for (int i = 0; i < 6; ++i) { run (e, 40, bs); e.editEq (0, "b1on", 1.0); e.editEq (0, "b1hz", freqs[i]); e.editEq (0, "b1gain", (double) i); run (e, 1, bs); CHECK (std::abs (e.eqGet (0, "b1hz") - freqs[i]) < 0.5, "while recording, the lane does not pull the setting back from under your hand"); e.service(); }
            const auto lanes = e.chainAt (0)->autoSnap.load()->lanes;
            auto find = [&] (const std::vector<AutoLane>& ls, int kind, int param) -> const AutoLane* { for (const auto& l : ls) if (l.kind == kind && (kind != 16 || l.param == param)) return &l; return nullptr; };
            const AutoLane* hzL = find (lanes, 16, 5); const AutoLane* gL = find (lanes, 16, 6); const AutoLane* onL = find (lanes, 16, 4);
            CHECK (lanes.size() == 3 && hzL != nullptr && gL != nullptr && onL != nullptr, "dragging a band makes its frequency and gain lanes (and the on lane) by itself, with no arming");
            CHECK (hzL != nullptr && hzL->pts.size() >= 5 && hzL->pts.front().value < hzL->pts.back().value && std::abs (eqLaneSetting (5, hzL->pts.back().value) - 3200.0) < 5.0 && gL != nullptr && std::abs (gL->pts.back().value - 5.0) < 0.01, "the lanes hold what was moved, at the beats it was moved (frequency on its logarithmic scale, gain in dB)");
            e.editEq (kMasterChain, "b2gain", -6.0); e.editMasterSound ("amOn", true); e.editMasterSound ("amPush", 4.0); run (e, 20, bs); e.editMasterSound ("amPush", 6.0);
            const auto ml = e.chainAt (kMasterChain)->autoSnap.load()->lanes; const AutoLane* pushL = find (ml, 9, -1);
            CHECK (find (ml, 16, 10) != nullptr && pushL != nullptr && pushL->pts.size() >= 2 && std::abs (pushL->pts.back().value - 6.0) < 0.01, "the master EQ and the Archie sound controls record into the master's lanes");
            e.transport.stop(); e.transport.setRecording (false); e.service(); e.service();
            juce::Thread::sleep (450); e.eqApply (0, "b1hz", 50.0); e.transport.locate (0.0); e.transport.play(); run (e, 600, bs);
            CHECK (std::abs (e.eqGet (0, "b1hz") - 3200.0) < 20.0, "played back, the setting follows the recorded lane");
            e.transport.stop(); e.service();
        }
        // SENDS START OFF: nothing is heard from an effect until a send is turned up
        {
            HostEngine e; e.prepareOffline (sr, bs, 0); bool allOff = true; for (int s = 0; s < 8; ++s) allOff = allOff && e.trackAt (0)->sendLvl[s].load() <= -59.5f && ! e.trackAt (0)->sendOn[s].load();
            CHECK (allOff, "a new track's sends are all off (level off, switch off)");
            e.trackAt (0)->source.store (2); e.setSendLevel (*e.trackAt (0), 0, 0.0f); e.setSendOn (*e.trackAt (0), 0, true); e.setSendLevel (*e.trackAt (0), 2, -6.0f);
            CHECK (e.isEmptyReturn (kReturnBase) && ! e.isEmptyReturn (0) && ! e.isEmptyReturn (kMasterChain), "an FX return with nothing on it counts as empty; a track and the master do not");
            e.quietSends (kReturnBase);
            CHECK (e.trackAt (0)->sendLvl[0].load() <= -59.5f && e.trackAt (0)->sendOn[0].load() && std::abs (e.trackAt (0)->sendLvl[2].load() + 6.0f) < 1e-4f, "putting an effect on an empty return sets that send's level to off on every track, leaving the switch and the other sends alone");
            auto song = e.saveProject (); if (auto* tr = song.getProperty ("tracks", {}).getArray()) if (auto* sd = (*tr)[0].getProperty ("sends", {}).getArray()) if (auto* o = (*sd)[1].getDynamicObject()) { o->setProperty ("lvl", 0.0); o->setProperty ("on", false); }
            HostEngine l; l.prepareOffline (sr, bs, 0); bool done = false; l.loadProject (song, [&] (juce::String) { done = true; }); for (int i = 0; i < 100 && ! done; ++i) { l.service(); juce::Thread::sleep (10); }
            CHECK (done && l.trackAt (0)->sendLvl[1].load() <= -59.5f, "a song opened with nothing on a return shows a send left at the old starting level as off (a level someone set is kept)");
        }
        // HOT CUE: the notes sounding when the song jumps are ended, so nothing hangs
        {
            struct Log : public juce::AudioProcessor
            {
                Log() : juce::AudioProcessor (BusesProperties().withOutput ("out", juce::AudioChannelSet::stereo(), true)) {}
                const juce::String getName() const override { return "Log"; } void prepareToPlay (double, int) override {} void releaseResources() override {}
                void processBlock (juce::AudioBuffer<float>& b, juce::MidiBuffer& m) override { b.clear(); for (const auto meta : m) { const auto x = meta.getMessage(); const double at = (double) (done + meta.samplePosition) / 48000.0 * 2.0; if (x.isNoteOn()) ons.push_back (at); else if (x.isNoteOff() && x.getNoteNumber() == 60) offs.push_back (at); } done += b.getNumSamples(); }
                double getTailLengthSeconds() const override { return 0; } bool acceptsMidi() const override { return true; } bool producesMidi() const override { return false; } juce::AudioProcessorEditor* createEditor() override { return nullptr; } bool hasEditor() const override { return false; }
                int getNumPrograms() override { return 1; } int getCurrentProgram() override { return 0; } void setCurrentProgram (int) override {} const juce::String getProgramName (int) override { return {}; } void changeProgramName (int, const juce::String&) override {} void getStateInformation (juce::MemoryBlock&) override {} void setStateInformation (const void*, int) override {}
                long long done = 0; std::vector<double> ons, offs;
            };
            HostEngine e; e.prepareOffline (sr, bs, 0); e.transport.setBpm (120.0); auto pr = std::make_unique<Log>(); auto* lg = pr.get(); e.addProcessor (0, std::move (pr), "log");
            auto* c = new juce::DynamicObject(); c->setProperty ("lengthBeats", 64.0); juce::Array<juce::var> ns; auto* n = new juce::DynamicObject(); n->setProperty ("beat", 0.0); n->setProperty ("len", 20.0); n->setProperty ("note", 60); n->setProperty ("vel", 100); n->setProperty ("ch", 1); ns.add (juce::var (n)); c->setProperty ("notes", ns); e.editSetClip (0, juce::var (c));      // one long note from the start
            e.transport.setLoop (true, 0.0, 16.0); e.transport.loopJump.store (2); e.transport.locate (0.0); e.transport.play(); run (e, 250, bs);      // about 2.7 beats
            lg->offs.clear(); e.transport.requestLoop (true, 32.0, 48.0); run (e, 150, bs);
            CHECK (! lg->offs.empty() && lg->offs.front() > 2.9 && lg->offs.front() < 3.1, "at the cue point (beat 3) the note still sounding is ended, so it does not hang after the jump");
            CHECK (e.transport.posBeats.load() > 32.0 && e.transport.posBeats.load() < 34.0, "and the song is playing from the new loop");
            e.transport.stop();
        }
        // DEVICE METERS: the level into and out of each device in the rack, by chain
        {
            HostEngine e; e.prepareOffline (sr, bs, 0); e.trackAt (0)->source.store (2);      // a 440 Hz tone at 0.25 into one effect
            e.addProcessor (0, std::make_unique<LatProc> (2, 2, 0), "plain effect"); e.transport.play(); run (e, 40, bs);
            auto m = e.meters(); auto sp = m.getProperty ("slotPk", {}); auto row = sp.getProperty ("0", {});
            const bool shaped = row.getArray() != nullptr && row.getArray()->size() == 1; double in = 0, out = 0; if (shaped) { in = (double) (*row.getArray())[0][0]; out = (double) (*row.getArray())[0][1]; }
            CHECK (shaped && in > 0.2 && in < 0.3 && out > 0.2 && out < 0.3, "each device reports the level into it and out of it (a 0.25 tone passes a plain effect: about 0.25 in and out)");
            auto m2 = e.meters(); auto r2 = m2.getProperty ("slotPk", {}).getProperty ("0", {}); CHECK (r2.getArray() != nullptr && (double) (*r2.getArray())[0][0] == 0.0, "and reading the meter clears it until the next block");
            e.setBypass (0, 0, true); run (e, 10, bs); e.meters(); run (e, 10, bs); auto r3 = e.meters().getProperty ("slotPk", {}).getProperty ("0", {}); CHECK (r3.getArray() != nullptr && (double) (*r3.getArray())[0][1] == 0.0, "a bypassed device shows no level");
            e.transport.stop(); e.service();
        }
        // STEP PATTERN: how a track's note clips play (off, loop, one shot) and how far they are moved (transpose, in semitones); both can be automated
        {
            struct NoteLog : public juce::AudioProcessor
            {
                NoteLog() : juce::AudioProcessor (BusesProperties().withOutput ("out", juce::AudioChannelSet::stereo(), true)) {}
                const juce::String getName() const override { return "NoteLog"; }
                void prepareToPlay (double, int) override {} void releaseResources() override {}
                void processBlock (juce::AudioBuffer<float>& b, juce::MidiBuffer& midi) override
                { b.clear(); for (const auto meta : midi) { const auto m = meta.getMessage(); const double beat = (double) (done + meta.samplePosition) / 48000.0 * 2.0; if (m.isNoteOn()) ons.push_back ({ beat, m.getNoteNumber() }); else if (m.isNoteOff() && m.getNoteNumber() >= 0 && ! m.isAllNotesOff()) offs.push_back ({ beat, m.getNoteNumber() }); } done += b.getNumSamples(); }
                double getTailLengthSeconds() const override { return 0; } bool acceptsMidi() const override { return true; } bool producesMidi() const override { return false; }
                juce::AudioProcessorEditor* createEditor() override { return nullptr; } bool hasEditor() const override { return false; }
                int getNumPrograms() override { return 1; } int getCurrentProgram() override { return 0; } void setCurrentProgram (int) override {} const juce::String getProgramName (int) override { return {}; } void changeProgramName (int, const juce::String&) override {}
                void getStateInformation (juce::MemoryBlock&) override {} void setStateInformation (const void*, int) override {}
                long long done = 0; std::vector<std::pair<double, int>> ons, offs;
            };
            auto clipVar = [] (double len, double n2len) { auto* c = new juce::DynamicObject(); c->setProperty ("lengthBeats", 4.0); juce::Array<juce::var> ns; auto note = [&] (double beat, double l, int nn) { auto* o = new juce::DynamicObject(); o->setProperty ("beat", beat); o->setProperty ("len", l); o->setProperty ("note", nn); o->setProperty ("vel", 100); o->setProperty ("ch", 1); ns.add (juce::var (o)); }; note (0.0, len, 60); note (2.0, n2len, 64); c->setProperty ("notes", ns); return juce::var (c); };
            auto go = [&] (double len, double n2len, std::function<void (HostEngine&)> setup, std::function<void (HostEngine&, int)> perBlock = nullptr)
            {
                HostEngine e; e.prepareOffline (sr, bs, 0); e.transport.setBpm (120.0); auto pr = std::make_unique<NoteLog>(); auto* lg = pr.get(); e.addProcessor (0, std::move (pr), "note log"); e.editSetClip (0, clipVar (len, n2len)); setup (e);
                e.transport.locate (0.0); e.transport.play(); const int blocks = (int) (4.0 * sr / bs) + 2; for (int i = 0; i < blocks; ++i) { if (perBlock) perBlock (e, i); run (e, 1, bs); if (i % 8 == 0) e.service(); }       // 4 s = 8 beats = two passes of the one-bar pattern
                e.transport.stop(); e.service();
                auto cut = [] (std::vector<std::pair<double, int>> v) { v.erase (std::remove_if (v.begin(), v.end(), [] (const std::pair<double, int>& x) { return x.first >= 7.99; }), v.end()); return v; };      // (only the eight beats asked for: a note exactly on beat 8 belongs to the next pass)
                return std::make_pair (cut (lg->ons), cut (lg->offs));
            };
            { auto r = go (0.5, 0.5, [] (HostEngine&) {}); CHECK (r.first.size() == 4, "by default the step pattern loops: two notes in each of two passes"); }
            { auto r = go (0.5, 0.5, [] (HostEngine& e) { e.editChainParam (0, "patPlay", 2.0); }); CHECK (r.first.size() == 2 && r.first[0].second == 60 && r.first[1].second == 64, "ONE SHOT plays the pattern once, from the start of the clip, and then it is silent"); }
            { auto r = go (0.5, 0.5, [] (HostEngine& e) { e.editChainParam (0, "patPlay", 0.0); }); CHECK (r.first.empty() && r.second.empty(), "OFF plays none of it"); }
            { auto r = go (0.5, 0.5, [] (HostEngine& e) { e.editChainParam (0, "patTrans", 7.0); }); CHECK (r.first.size() == 4 && r.first[0].second == 67 && r.first[1].second == 71, "transpose +7 moves every note up a fifth"); }
            { auto r = go (0.5, 0.5, [] (HostEngine& e) { e.editChainParam (0, "patTrans", -100.0); }); CHECK (r.first.size() == 4 && r.first[0].second == 36, "transpose is held to 24 semitones either way"); }
            {   // a note that is still sounding when the transpose moves ends on the pitch it started on
                auto r = go (3.5, 0.5, [] (HostEngine&) {}, [] (HostEngine& e, int i) { if (i == 40) e.editChainParam (0, "patTrans", 5.0); if (i == 120) e.editChainParam (0, "patTrans", -3.0); });
                std::multiset<int> a, b; for (auto& x : r.first) a.insert (x.second); for (auto& x : r.second) b.insert (x.second);
                CHECK (a == b && ! a.empty(), "every note ends on the pitch it started on, even when the transpose moves while it sounds (no hanging notes)");
            }
            {   // automation: transpose climbing a step at a time (a riser), and the mode switched off by its lane
                juce::Array<juce::var> up; auto pt = [&] (juce::Array<juce::var>& a, double bt, double v) { auto* o = new juce::DynamicObject(); o->setProperty ("beat", bt); o->setProperty ("value", v); a.add (juce::var (o)); }; pt (up, 0.0, 0.0); pt (up, 8.0, 8.0);
                auto r = go (0.5, 0.5, [&] (HostEngine& e) { const int ln = e.editAutoAddLane (0, 5, -1, -1); CHECK (ln != 0, "a lane for the step pattern transpose can be made"); e.editAutoSetPoints (0, ln, juce::var (up)); });
                bool rising = r.first.size() == 4; for (size_t i = 1; i < r.first.size(); ++i) rising = rising && (r.first[i].second - (i % 2 == 0 ? 60 : 64)) >= (r.first[i - 1].second - ((i - 1) % 2 == 0 ? 60 : 64)); rising = rising && r.first.size() == 4 && (r.first[3].second - 64) > (r.first[0].second - 60);
                CHECK (rising, "an automated transpose moves the pattern up as the song goes on (a riser)");
                juce::Array<juce::var> off; pt (off, 0.0, 0.0); pt (off, 8.0, 0.0);
                auto r2 = go (0.5, 0.5, [&] (HostEngine& e) { const int ln = e.editAutoAddLane (0, 6, -1, -1); e.editAutoSetPoints (0, ln, juce::var (off)); });
                CHECK (r2.first.empty(), "a pattern-play lane at 0 switches the pattern off");
                HostEngine s; s.prepareOffline (sr, bs, 0); s.editChainParam (0, "patPlay", 2.0); s.editChainParam (0, "patTrans", -5.0); auto song = s.saveProject();
                HostEngine l2; l2.prepareOffline (sr, bs, 0); bool done = false; l2.loadProject (song, [&] (juce::String) { done = true; }); for (int i = 0; i < 100 && ! done; ++i) { l2.service(); juce::Thread::sleep (10); }
                CHECK (done && l2.trackAt (0)->patPlay.load() == 2 && l2.trackAt (0)->patTrans.load() == -5, "a saved song keeps the step pattern's mode and transpose");
            }
        }
        // LOOP RECORDING: what is played in a pass sounds on the next pass while recording goes on (it used to wait until recording was switched off), and is shown as ghost notes while it goes in
        {
            HostEngine e; e.prepareOffline (sr, bs, 0); e.editChainParam (0, "armed", true); e.transport.setLoop (true, 0.0, 8.0); e.transport.setRecording (true); e.transport.play();
            auto ticks = [&] (int n) { for (int k = 0; k < n; ++k) { run (e, 10, bs); e.service(); } };       // the 30 Hz tick runs between blocks, as it does in the program
            const size_t undoStart = e.undo.depth();
            e.noteOn (0, 60, 100); ticks (1); e.noteOff (0, 60); ticks (2); e.noteOn (0, 64, 100); ticks (2);                   // a short note, then one that is still held when the loop wraps
            auto g = e.recGhostsVar (*e.trackAt (0)); CHECK (g.getArray() != nullptr && g.getArray()->size() == 2, "the notes played so far are offered as ghost notes (one finished, one held)");
            CHECK (e.trackAt (0)->arrSnap.load()->insts.empty(), "and are not yet part of the song");
            ticks (100);                                                                                                         // on past the end of the loop (8 beats = 4 s)
            const auto in1 = e.trackAt (0)->arrSnap.load()->insts;
            CHECK (in1.size() == 1 && in1[0].content != nullptr && in1[0].content->notes.size() == 2, "when the loop returns to its start, the notes of that pass are in the song at once (while still recording)");
            if (in1.size() == 1) CHECK (std::abs (in1[0].start) < 1e-9 && std::abs (in1[0].length - 8.0) < 1e-9 && std::abs (in1[0].content->lengthBeats - 8.0) < 1e-9, "as one clip that covers the loop");
            if (in1.size() == 1) { bool held = false; for (const auto& n : in1[0].content->notes) if (n.note == 64 && n.beat + n.len > 7.9 && n.beat + n.len < 8.1) held = true; CHECK (held, "a note still held at the wrap ends at the end of the loop"); }
            CHECK (e.recGhostsVar (*e.trackAt (0)).getArray()->isEmpty(), "and the ghosts are gone (they are real notes now)");
            e.noteOn (0, 67, 100); ticks (1); e.noteOff (0, 67); ticks (2);                                                      // a second pass: more notes, laid over the first
            e.transport.stop(); e.service(); e.service();
            const auto in2 = e.trackAt (0)->arrSnap.load()->insts; bool has60 = false, has64 = false, has67 = false;
            if (in2.size() == 1 && in2[0].content) for (const auto& n : in2[0].content->notes) { has60 = has60 || n.note == 60; has64 = has64 || n.note == 64; has67 = has67 || n.note == 67; }
            CHECK (in2.size() == 1 && has60 && has64 && has67, "the second pass is laid over the first in the same clip, and stopping keeps everything");
            CHECK (e.undo.depth() == undoStart + 1 && e.undo.undoLabel() == "Record notes", "a whole loop take, however many passes, is ONE undo step");
            e.undo.undo(); CHECK (e.trackAt (0)->arrSnap.load()->insts.empty(), "and one undo takes the whole take back");
            e.undo.redo(); { const auto in3 = e.trackAt (0)->arrSnap.load()->insts; bool all = in3.size() == 1 && in3[0].content != nullptr; int found = 0; if (all) for (const auto& n : in3[0].content->notes) found += (n.note == 60 || n.note == 64 || n.note == 67) ? 1 : 0; CHECK (all && found == 3, "and one redo brings every pass back"); }
        }
        // REMOTE: a controller knob assigned to a plug-in control
        { HostEngine e; e.prepareOffline (sr, bs, 0); auto pr = std::make_unique<RecProbe>(); auto* p = pr.get(); e.addProcessor (0, std::move (pr), "probe");
          e.startRemote (0); for (int i = 0; i < 6; ++i) e.service();
          p->pars[2]->setValueNotifyingHost (0.7f); for (int i = 0; i < 3; ++i) e.service();
          CHECK (e.remoteStage.load() == 2 && e.remoteParam == 2, "Remote: moving a control in the plug-in picks it and then waits for a controller knob");
          CHECK (e.remoteConsume (juce::MidiMessage::controllerEvent (1, 74, 10)), "the first controller message is taken as the assignment"); e.service();
          CHECK (e.remoteStage.load() == 0 && e.remoteCount (0) == 1 && lanesOf (e, 0).empty(), "which assigns it (and makes no automation lane)");
          CHECK (e.remoteConsume (juce::MidiMessage::controllerEvent (1, 74, 127)) && ! e.remoteConsume (juce::MidiMessage::controllerEvent (1, 20, 5)), "that knob is used up by the assignment; other controllers are still played");
          e.service(); CHECK (std::abs (p->pars[2]->get() - 1.0f) < 0.01f, "and moves the control in the plug-in: all the way up");
          e.remoteConsume (juce::MidiMessage::controllerEvent (1, 74, 0)); e.service(); CHECK (p->pars[2]->get() < 0.01f, "and all the way down");
          auto saved = e.remotesVar (0); HostEngine e2; e2.prepareOffline (sr, bs, 0); e2.loadRemotesVar (0, saved); CHECK (e2.remoteCount (0) == 1, "assignments are kept with the song");
          e.clearRemotes (0); CHECK (e.remoteCount (0) == 0 && ! e.remoteConsume (juce::MidiMessage::controllerEvent (1, 74, 5)), "and can be cleared"); }
        { HostEngine e; e.prepareOffline (sr, bs, 0); e.editChainParam (0, "armed", true); e.transport.setRecording (true); e.transport.play(); run (e, 6, bs);
          e.editChainParam (0, "gainDb", -6.0); run (e, 40, bs); e.editChainParam (0, "gainDb", -12.0); e.service();
          auto ln = lanesOf (e, 0); CHECK (ln.size() == 1 && ln[0].kind == 0 && ln[0].pts.size() == 2, "moving a track's level while the song plays records it into a level lane (made for it)");
          if (ln.size() == 1 && ln[0].pts.size() == 2) CHECK (std::abs (ln[0].pts[0].value + 6.0) < 1e-6 && std::abs (ln[0].pts[1].value + 12.0) < 1e-6 && ln[0].pts[1].beat > ln[0].pts[0].beat, "with the values moved to, at the beats they were moved");
          CHECK (e.trackAt (0)->autoTouchUntilMs.load() > 0.0, "and the lane stops pulling the fader back while it is held");
          e.editChainParam (0, "pan", 0.5); CHECK (lanesOf (e, 0).size() == 2, "pan gets its own lane");
          e.transport.stop(); e.service(); }
        { HostEngine e; e.prepareOffline (sr, bs, 0); e.transport.play(); run (e, 6, bs); e.editChainParam (0, "gainDb", -6.0); CHECK (lanesOf (e, 0).empty(), "with Record automation off, a level move leaves the lanes alone");
          e.editChainParam (0, "armed", true); e.transport.setRecording (true); e.transport.stop(); e.service(); e.editChainParam (0, "gainDb", -3.0); CHECK (lanesOf (e, 0).empty(), "and with it on but the song stopped, nothing is recorded either"); }
    }

    // ---- AUDIO TRANSPOSE (Audio Edit): a clip's pitch moved with its length and timing unchanged, rendered in the background ----
    {
        auto makeTone = [&] (const char* name, double hz, double dur)
        {
            auto f = juce::File::getSpecialLocation (juce::File::tempDirectory).getChildFile (name); f.deleteFile();
            juce::WavAudioFormat wav; auto out = f.createOutputStream();
            std::unique_ptr<juce::AudioFormatWriter> w (wav.createWriterFor (out.get(), 48000.0, 2, 32, {}, 0)); if (w) out.release();
            const int n = (int) (dur * 48000.0); juce::AudioBuffer<float> b (2, n); for (int i = 0; i < n; ++i) { const float v = 0.4f * (float) std::sin (6.283185307179586 * hz * i / 48000.0); b.setSample (0, i, v); b.setSample (1, i, v); }
            w->writeFromAudioSampleBuffer (b, 0, n); w.reset(); return f;
        };
        auto props = [] (std::initializer_list<std::pair<const char*, juce::var>> kv) { auto* p = new juce::DynamicObject(); for (auto& x : kv) p->setProperty (x.first, x.second); return juce::var (p); };
        auto pitchOf = [&] (const std::vector<float>& o, double t0, double t1) { AudioData a; a.rate = sr; a.buf.setSize (1, (int) o.size()); for (size_t i = 0; i < o.size(); ++i) a.buf.setSample (0, (int) i, o[i]); auto fr = trackPitch (a); std::vector<double> v; for (auto& f : fr) if (f.voiced && f.sec >= t0 && f.sec <= t1) v.push_back (f.hz); if (v.empty()) return 0.0; std::sort (v.begin(), v.end()); return v[v.size() / 2]; };
        auto f220 = makeTone ("archie-tone220.wav", 220.0, 4.0);
        { HostEngine e; e.prepareOffline (sr, bs, 0); e.transport.play(); juce::String err; const int id = e.editImportAudio (0, f220, 0.0, err);
          CHECK (e.trackAt (0)->arrSnap.load()->insts[0].transpose == 0.0 && ! e.warpBusy(), "a new clip is not transposed and nothing is rendered");
          e.editSetInstance (id, props ({ { "transpose", 12.0 }, { "keepFormant", false }, { "length", 8.0 } }));
          CHECK (e.warpBusy() || e.trackAt (0)->arrSnap.load()->insts[0].pitched != nullptr, "transposing asks for a background render");
          { auto st = e.arrangementVar (*e.trackAt (0)).getProperty ("instances", {})[0].getProperty ("pitchState", {}).toString(); CHECK (st == "rendering" || st == "ready", "and the clip says so"); }
          e.waitForWarp(); const auto& in = e.trackAt (0)->arrSnap.load()->insts[0];
          CHECK (in.pitched != nullptr && in.pitched->numSamples() == in.content->audio->numSamples() && ! e.warpBusy(), "the render arrives with exactly the source's length");
          auto o = run (e, 400, bs); const double hz = pitchOf (o, 0.5, 1.5); CHECK (hz > 0 && std::abs (1200.0 * std::log2 (hz / 440.0)) < 15.0, "the clip now sounds an octave up (440 Hz from 220)");
          e.editSetInstance (id, props ({ { "transpose", 0.0 } })); e.waitForWarp(); CHECK (e.trackAt (0)->arrSnap.load()->insts[0].pitched == nullptr, "transpose 0 drops the render: the clip plays its source again");
          {   // MASTER TUNE: audio clips only, measured
              HostEngine m; m.audioOptionsFile().deleteFile(); m.loadAudioOptions(); m.prepareOffline (sr, bs, 0); m.transport.play(); juce::String er3; const int mi = m.editImportAudio (0, f220, 0.0, er3); m.editSetInstance (mi, props ({ { "length", 8.0 }, { "keepFormant", false } }));
              CHECK (m.setAudioOptions (props ({ { "masterTuneCents", 100.0 } })), "master tune can be set"); m.waitForWarp(); auto o1 = run (m, 400, bs); const double h1 = pitchOf (o1, 0.5, 1.5);
              CHECK (h1 > 0 && std::abs (1200.0 * std::log2 (h1 / 233.0819)) < 15.0, "100 cent up: a 220 Hz clip sounds at 233 Hz");
              CHECK (m.trackAt (0)->arrSnap.load()->insts[0].transpose == 0.0, "the clip's own transpose is untouched");
              HostEngine h2; h2.loadAudioOptions(); CHECK (h2.masterTuneCents.load() == 100, "and it is remembered");
              m.setAudioOptions (props ({ { "masterTuneCents", 999.0 } })); CHECK (m.masterTuneCents.load() == 100, "limited to one semitone either way");
              m.setAudioOptions (props ({ { "masterTuneCents", 0.0 } })); m.waitForWarp(); CHECK (m.trackAt (0)->arrSnap.load()->insts[0].pitched == nullptr, "back at 0 the clip plays its source again");
              m.audioOptionsFile().deleteFile();
          }
          { HostEngine z; z.prepareOffline (sr, bs, 0); z.transport.play(); juce::String er2; z.editImportAudio (0, f220, 0.0, er2); HostEngine q; q.prepareOffline (sr, bs, 0); q.transport.play(); const int qi = q.editImportAudio (0, f220, 0.0, er2); q.editSetInstance (qi, props ({ { "transpose", 5.0 } })); q.waitForWarp(); q.editSetInstance (qi, props ({ { "transpose", 0.0 } })); q.waitForWarp();
            auto a = run (z, 100, bs), b = run (q, 100, bs); bool same = a.size() == b.size(); for (size_t i = 0; same && i < a.size(); ++i) if (a[i] != b[i]) same = false; CHECK (same, "a clip never transposed and one set back to 0 play bit for bit the same"); } }
        { HostEngine e; e.prepareOffline (sr, bs, 0); e.transport.play(); juce::String err; e.undo.clock = [] { static double c = 0; c += 5000; return c; }; const int id = e.editImportAudio (0, f220, 0.0, err);
          e.editSetInstance (id, props ({ { "transpose", -12.0 }, { "keepFormant", false }, { "length", 8.0 } })); e.waitForWarp();
          const double hz = pitchOf (run (e, 400, bs), 0.5, 1.5); CHECK (hz > 0 && std::abs (1200.0 * std::log2 (hz / 110.0)) < 15.0, "an octave down gives 110 Hz");
          e.editSetInstance (id, props ({ { "transpose", 99.0 } })); CHECK (std::abs (e.trackAt (0)->arrSnap.load()->insts[0].transpose - 24.0) < 1e-9, "transpose is limited to 24 semitones");
          e.editSetInstance (id, props ({ { "transpose", 3.456 } })); CHECK (std::abs (e.trackAt (0)->arrSnap.load()->insts[0].transpose - 3.46) < 1e-9, "and kept to hundredths of a semitone (cents)");
          e.editSetInstance (id, props ({ { "transpose", 7.0 } })); e.waitForWarp(); e.undo.undo (); e.waitForWarp(); CHECK (std::abs (e.trackAt (0)->arrSnap.load()->insts[0].transpose - 3.46) < 1e-9 && e.trackAt (0)->arrSnap.load()->insts[0].pitched != nullptr, "undo brings back the earlier transpose and its render");
          const auto saved = e.saveProject(); HostEngine h; h.prepareOffline (sr, bs, 0); h.loadProject (saved, [&] (juce::String) {}); h.waitForWarp(); const auto& li = h.trackAt (0)->arrSnap.load()->insts[0];
          CHECK (std::abs (li.transpose - 3.46) < 1e-9 && ! li.keepFormant && li.pitched != nullptr, "transpose and the voice colour switch are saved with the song and the render is made again on load"); }
        { HostEngine e; e.prepareOffline (sr, bs, 0); e.transport.play(); juce::String err; const int id = e.editImportAudio (0, f220, 0.0, err);
          e.editSetInstance (id, props ({ { "warp", true }, { "srcBpm", 120.0 }, { "length", 4.0 }, { "transpose", 12.0 }, { "keepFormant", false } })); e.waitForWarp(); const auto& in = e.trackAt (0)->arrSnap.load()->insts[0];
          CHECK (in.warped != nullptr && in.pitched == nullptr, "a warped clip makes one render (the stretched sound, moved), not two");
          const double hz = pitchOf (run (e, 300, bs), 0.4, 1.4); CHECK (hz > 0 && std::abs (1200.0 * std::log2 (hz / 440.0)) < 20.0, "and it sounds an octave up");
          e.editSetInstance (id, props ({ { "warp", false } })); e.waitForWarp(); CHECK (e.trackAt (0)->arrSnap.load()->insts[0].pitched != nullptr, "warp switched off: the transposed source takes over"); }
    }

    // ---- DEVICE BROWSER: what a device is, and the notes kept about it ----
    {
        auto mk = [] (const char* name, const char* cat, bool inst) { juce::PluginDescription d; d.name = name; d.category = cat; d.isInstrument = inst; d.pluginFormatName = "VST3"; d.manufacturerName = "Maker"; d.fileOrIdentifier = juce::String ("/x/") + name + ".vst3"; d.uniqueId = 7; return d; };
        auto c1 = HostEngine::classifyDevice (mk ("Syn", "Instrument|Synth", true));
        CHECK (c1.kind == "instrument" && c1.cats.size() == 1 && c1.cats[0] == "Synth", "an instrument keeps its sub-category and loses the word Instrument");
        auto c2 = HostEngine::classifyDevice (mk ("Comp", "Fx|Dynamics", false));
        CHECK (c2.kind == "effect" && c2.cats.size() == 1 && c2.cats[0] == "Dynamics", "an effect is an effect, with its sub-category");
        auto c3 = HostEngine::classifyDevice (mk ("Meter", "Fx|Analyzer", false));
        CHECK (c3.kind == "utility" && c3.cats[0] == "Analyzer", "an analyser is a utility");
        auto c4 = HostEngine::classifyDevice (mk ("Plain", "", false));
        CHECK (c4.kind == "effect" && c4.cats.isEmpty(), "a plug-in that says nothing about itself is an effect with no category (nothing is guessed from its name)");
        auto c5 = HostEngine::classifyDevice (mk ("Multi", "Fx|Delay|Reverb", false));
        CHECK (c5.cats.size() == 2, "several sub-categories are all kept");

        auto tmp = juce::File::getSpecialLocation (juce::File::tempDirectory).getChildFile ("archie-device-meta-test.json"); tmp.deleteFile();
        { HostEngine e; e.deviceMetaFile = tmp; e.deviceMeta = juce::var (new juce::DynamicObject());
          juce::DynamicObject* patch = new juce::DynamicObject(); patch->setProperty ("favourite", true);
          juce::Array<juce::var> tags; tags.add ("  Drums "); tags.add ("DRUMS"); tags.add ("my fav!"); tags.add (""); patch->setProperty ("tags", tags);
          CHECK (e.setDeviceMeta ("id-1", juce::var (patch)), "notes are set on a device");
          auto* m = e.metaFor ("id-1", false);
          CHECK (m != nullptr && (bool) m->getProperty ("fav"), "it is a favourite");
          auto* ts = m->getProperty ("tags").getArray();
          CHECK (ts != nullptr && ts->size() == 2 && (*ts)[0].toString() == "drums" && (*ts)[1].toString() == "my fav", "tags are tidied: trimmed, lower case, no duplicates, no blanks, no odd characters");
          e.bumpDeviceUse ("id-1"); e.bumpDeviceUse ("id-1");
          CHECK ((int) e.metaFor ("id-1", false)->getProperty ("uses") == 2, "each use is counted");
          CHECK (! e.setDeviceMeta ("", juce::var()), "no id, no notes"); }
        { HostEngine e; e.deviceMetaFile = tmp; e.loadDeviceMeta();
          auto* m = e.metaFor ("id-1", false);
          CHECK (m != nullptr && (bool) m->getProperty ("fav") && (int) m->getProperty ("uses") == 2, "the notes are still there after the program is started again");
          // a device the program has not met yet has no notes and costs nothing
          CHECK (e.metaFor ("never-seen", false) == nullptr, "an unknown device has no notes"); }
        tmp.deleteFile();
    }

    // ---- RACK: devices move within a rack and between racks (same plug-in instance), and it all undoes ----
    {
        auto meanOf = [&] (HostEngine& e) { auto o = run (e, 40, bs); double m = 0; int n = 0; for (size_t i = 8192; i < o.size(); ++i) { m += o[i]; ++n; } return m / n; };
        HostEngine e; e.prepareOffline (sr, bs, 0);
        e.trackAt (0)->source.store (2);
        e.addProcessor (0, std::make_unique<TestProc> (2, 2, 0, 2.0f, 0.0f), "double");        // x * 2
        e.addProcessor (0, std::make_unique<TestProc> (2, 2, 0, 1.0f, 0.1f), "plus 0.1");       // x + 0.1
        CHECK (std::abs (meanOf (e) - 0.1) < 1.5e-2, "double then add 0.1 leaves a 0.1 offset (the order matters, so a move can be heard)");
        auto* a0 = e.trackAt (0)->slots[0].get(); auto* a1 = e.trackAt (0)->slots[1].get();
        CHECK (e.editMoveSlotTo (0, 0, 0, 1) == 1, "the first device is moved below the second");
        CHECK (e.trackAt (0)->slots[0].get() == a1 && e.trackAt (0)->slots[1].get() == a0, "they swap, and they are the same instances");
        CHECK (std::abs (meanOf (e) - 0.2) < 1.5e-2, "add 0.1 then double leaves 0.2");
        e.undo.undo(); CHECK (e.trackAt (0)->slots[0].get() == a0 && std::abs (meanOf (e) - 0.1) < 1.5e-2, "undo puts the order back");
        e.undo.redo(); e.undo.undo();
        CHECK (e.editMoveSlotTo (0, 0, kMasterChain, 0) == 0, "the first device goes to the master rack");
        CHECK (e.trackAt (0)->slots.size() == 1 && e.chainAt (kMasterChain)->slots.size() == 1 && e.chainAt (kMasterChain)->slots[0].get() == a0, "it left the track and arrived in the master rack as the same instance");
        CHECK (std::abs (meanOf (e) - 0.2) < 1.5e-2, "plus 0.1 on the track then double at the master leaves 0.2");
        e.undo.undo(); CHECK (e.trackAt (0)->slots.size() == 2 && e.chainAt (kMasterChain)->slots.empty() && e.trackAt (0)->slots[0].get() == a0, "undo puts it back in its place on the track");
        CHECK (std::abs (meanOf (e) - 0.1) < 1.5e-2, "and the sound");
        CHECK (e.editMoveSlotTo (0, 5, 0, 0) == -1 && e.editMoveSlotTo (0, 0, 999, 0) == -1, "a missing device or rack is refused");
        CHECK (e.editMoveSlotTo (0, 0, 0, 0) == 0 && e.trackAt (0)->slots.size() == 2, "moving a device to where it already is does nothing");
    }

    // ---- AUDIO CLIPS: a sound on the timeline, cropped, split, repeated, shared ----
    {
        // a test sound: a rising ramp, value = 0.8 * t / duration, as a 32-bit float wav so the numbers can be read back exactly
        auto makeRamp = [&] (const char* name, double dur, double rate)
        {
            auto f = juce::File::getSpecialLocation (juce::File::tempDirectory).getChildFile (name); f.deleteFile();
            juce::WavAudioFormat wav; auto out = f.createOutputStream();
            std::unique_ptr<juce::AudioFormatWriter> w (wav.createWriterFor (out.get(), rate, 2, 32, {}, 0)); if (w) out.release();
            const int n = (int) (dur * rate); juce::AudioBuffer<float> b (2, n);
            for (int i = 0; i < n; ++i) { const float v = (float) ((double) i / (double) n) * 0.8f; b.setSample (0, i, v); b.setSample (1, i, v); }
            w->writeFromAudioSampleBuffer (b, 0, n); w.reset(); return f;
        };
        auto at = [&] (const std::vector<float>& o, double sec) { return o[(size_t) std::lround (sec * sr)]; };     // 48 kHz, 120 bpm: 0.5 s a beat
        auto f4 = makeRamp ("archie-ramp4.wav", 4.0, 48000.0);                                                       // 0 -> 0.8 over 4 seconds
        auto prop = [] (const char* k, double v) { auto* p = new juce::DynamicObject(); p->setProperty (k, v); return juce::var (p); };

        { HostEngine e; e.prepareOffline (sr, bs, 0); e.transport.play();
          juce::String err; const int id = e.editImportAudio (0, f4, 2.0, err);                                      // at beat 2 = 1 second
          CHECK (id != 0 && err.isEmpty(), "a sound file is put on a track at beat 2");
          const auto ins = e.trackAt (0)->arrSnap.load()->insts[0];
          CHECK (ins.content->audio != nullptr && std::abs (ins.start - 2.0) < 1e-9 && std::abs (ins.length - 8.0) < 0.01, "the clip is as long as the sound: 4 seconds = 8 beats at 120");
          auto o = run (e, 1250, bs);
          CHECK (std::abs (at (o, 0.5)) < 1e-4f, "nothing plays before the clip begins");
          CHECK (std::abs (at (o, 3.0) - 0.4f) < 0.01f, "two seconds into the sound (3 s on the timeline) it is at 0.4, halfway up its ramp");
          CHECK (std::abs (at (o, 4.9) - 0.78f) < 0.02f, "nearTo its end it is nearTo 0.8");
          CHECK (std::abs (at (o, 6.0)) < 1e-4f, "and it is silent after");
          const auto wf = e.waveformVar (ins.content->id, 8); CHECK (wf.getArray() != nullptr && wf.getArray()->size() == 8, "the waveform has the number of bins asked for");
          CHECK ((double) (*wf.getArray())[7] > 0.7 && (double) (*wf.getArray())[0] < 0.2, "it rises (the ramp)");
          CHECK (std::abs (at (o, 1.0)) < 0.01f, "the start fades in rather than clicking in"); }
        { // crop the front by 2 beats (1 s): the clip starts later and plays from further into the sound
          HostEngine e; e.prepareOffline (sr, bs, 0); e.transport.play();
          juce::String err; const int id = e.editImportAudio (0, f4, 0.0, err);
          CHECK (e.editSetInstance (id, prop ("trimStart", 2.0)), "the front is cropped to beat 2");
          const auto ins = e.trackAt (0)->arrSnap.load()->insts[0];
          CHECK (std::abs (ins.start - 2.0) < 1e-9 && std::abs (ins.offset - 1.0) < 1e-6 && std::abs (ins.length - 6.0) < 0.01, "it now starts at beat 2, 1 second into the sound, and is 6 beats long");
          auto o = run (e, 340, bs);
          CHECK (std::abs (at (o, 0.5)) < 1e-4f && std::abs (at (o, 1.5) - 0.3f) < 0.01f, "at 1.5 s on the timeline (0.5 s after the new start) it plays 1.5 s into the sound: 0.3");
          e.undo.undo(); CHECK (std::abs (e.trackAt (0)->arrSnap.load()->insts[0].start) < 1e-9, "undo puts the front back"); }
        { // repeat: an instance longer than the sound loops it
          HostEngine e; e.prepareOffline (sr, bs, 0); e.transport.play();
          auto f1 = makeRamp ("archie-ramp1.wav", 1.0, 48000.0);
          juce::String err; const int id = e.editImportAudio (0, f1, 0.0, err);
          e.editSetInstance (id, prop ("length", 6.0));                                                              // 6 beats = 3 s of a 1 s sound
          auto o = run (e, 700, bs);
          CHECK (std::abs (at (o, 0.5) - 0.4f) < 0.01f && std::abs (at (o, 1.5) - 0.4f) < 0.01f && std::abs (at (o, 2.5) - 0.4f) < 0.01f, "a 1-second sound in a 3-second clip repeats: the same value every second");
          CHECK (std::abs (at (o, 3.2)) < 1e-4f, "and the clip ends where it was made to end");
          f1.deleteFile(); }
        { // split: the two halves together play what the whole did
          HostEngine e; e.prepareOffline (sr, bs, 0); e.transport.play();
          juce::String err; const int id = e.editImportAudio (0, f4, 0.0, err);
          CHECK (e.editSplitInstance (id, 2.0) != 0, "split at beat 2");
          const auto& v = e.trackAt (0)->arrSnap.load()->insts;
          CHECK (v.size() == 2 && std::abs (v[1].offset - 1.0) < 1e-6 && v[0].content == v[1].content, "the second half starts 1 s into the same sound");
          auto o = run (e, 700, bs);
          CHECK (std::abs (at (o, 0.5) - 0.1f) < 0.01f && std::abs (at (o, 1.5) - 0.3f) < 0.01f && std::abs (at (o, 3.0) - 0.6f) < 0.01f, "the sound carries on across the cut: 0.1, 0.3, 0.6"); }
        { // sharing, saving, loading
          HostEngine e; e.prepareOffline (sr, bs, 0);
          juce::String err; e.editImportAudio (0, f4, 0.0, err);
          const int cid = e.trackAt (0)->arrSnap.load()->insts[0].content->id;
          e.editAddInstance (1, 4.0, 0.0, 0.0, {}, cid);
          CHECK (e.trackAt (1)->arrSnap.load()->insts[0].content->audio == e.trackAt (0)->arrSnap.load()->insts[0].content->audio, "a second instance on another track shares the one sound in memory");
          auto proj = e.saveProject();
          HostEngine g; g.prepareOffline (sr, bs, 0); juce::String er2 = "unset"; g.loadProject (proj, [&] (juce::String s) { er2 = s; });
          CHECK (er2.isEmpty() && g.trackAt (0)->arrSnap.load()->insts.size() == 1 && g.trackAt (1)->arrSnap.load()->insts.size() == 1, "a project with audio clips loads them back");
          CHECK (g.trackAt (0)->arrSnap.load()->insts[0].content->audio != nullptr && g.trackAt (0)->arrSnap.load()->insts[0].content->audio == g.trackAt (1)->arrSnap.load()->insts[0].content->audio, "still sharing one sound");
          auto moved = juce::JSON::parse (juce::JSON::toString (proj));
          moved.getProperty ("tracks", {})[0].getProperty ("arrangement", {}).getProperty ("contents", {})[0].getDynamicObject()->setProperty ("path", "Z:/not/here.wav");
          HostEngine h; h.prepareOffline (sr, bs, 0); juce::String er3; h.loadProject (moved, [&] (juce::String s) { er3 = s; });
          CHECK (er3.contains ("not there"), "a sound file that is no longer where it was is reported by name when a project loads");
          CHECK (e.editImportAudio (0, juce::File ("Z:/not/here.wav"), 0.0, err) == 0 && err.contains ("not there"), "importing a missing file is refused with the reason"); }
        f4.deleteFile();
    }

    // ---- AUDIO CLIPS, STAGE B: gain, fades, reverse, and finding where hits start ----
    {
        auto makeRamp = [&] (const char* name, double dur, double rate)
        {
            auto f = juce::File::getSpecialLocation (juce::File::tempDirectory).getChildFile (name); f.deleteFile();
            juce::WavAudioFormat wav; auto out = f.createOutputStream();
            std::unique_ptr<juce::AudioFormatWriter> w (wav.createWriterFor (out.get(), rate, 2, 32, {}, 0)); if (w) out.release();
            const int n = (int) (dur * rate); juce::AudioBuffer<float> b (2, n);
            for (int i = 0; i < n; ++i) { const float v = (float) ((double) i / (double) n) * 0.8f; b.setSample (0, i, v); b.setSample (1, i, v); }
            w->writeFromAudioSampleBuffer (b, 0, n); w.reset(); return f;
        };
        auto at = [&] (const std::vector<float>& o, double sec) { return o[(size_t) std::lround (sec * sr)]; };
        auto f4 = makeRamp ("archie-ramp4b.wav", 4.0, 48000.0);                                                      // 0 -> 0.8 over 4 s
        auto props = [] (std::initializer_list<std::pair<const char*, juce::var>> kv) { auto* p = new juce::DynamicObject(); for (auto& x : kv) p->setProperty (x.first, x.second); return juce::var (p); };

        { // gain
          HostEngine e; e.prepareOffline (sr, bs, 0); e.transport.play(); juce::String err;
          const int id = e.editImportAudio (0, f4, 0.0, err);
          e.editSetInstance (id, props ({ { "gainDb", -6.0206 } }));
          auto o = run (e, 440, bs);
          CHECK (std::abs (at (o, 2.0) - 0.2f) < 0.005f, "-6 dB halves the clip: 0.4 becomes 0.2");
          e.undo.undo (); CHECK (std::abs (e.trackAt (0)->arrSnap.load()->insts[0].gainDb) < 1e-6f, "undo puts the gain back");
          e.editSetInstance (id, props ({ { "gainDb", 100.0 } })); CHECK (std::abs (e.trackAt (0)->arrSnap.load()->insts[0].gainDb - 24.0f) < 1e-4f, "a silly gain is held to +24 dB");
          e.editSetInstance (id, props ({ { "fadeIn", 0.0 } })); CHECK (e.trackAt (0)->arrSnap.load()->insts[0].fadeIn >= 0.0005, "a fade can be made very short but never zero (a hard edge clicks)"); }
        { // mute and the clip's own name
          HostEngine e; e.prepareOffline (sr, bs, 0); e.transport.play(); juce::String err;
          const int id = e.editImportAudio (0, f4, 0.0, err);
          e.editSetInstance (id, props ({ { "muted", true } }));
          auto o = run (e, 440, bs); float pk = 0; for (float v : o) pk = juce::jmax (pk, std::abs (v));
          CHECK (e.trackAt (0)->arrSnap.load()->insts[0].muted && pk < 1.0e-6f, "a muted audio clip is silent");
          e.undo.undo (); CHECK (! e.trackAt (0)->arrSnap.load()->insts[0].muted, "undo un-mutes it");
          e.undo.redo ();
          e.editSetInstance (id, props ({ { "label", "Verse vocal" } }));
          auto proj = e.saveProject(); HostEngine g; g.prepareOffline (sr, bs, 0); g.loadProject (proj, [] (juce::String) {});
          const auto& gi = g.trackAt (0)->arrSnap.load()->insts;
          CHECK (gi.size() == 1 && gi[0].muted && gi[0].label == "Verse vocal", "mute and the name survive save and load");
          const auto av = e.arrangementVar (*e.trackAt (0)); CHECK (juce::JSON::toString (av).contains ("Verse vocal"), "the name reaches the console");
        }
        { // a muted note clip plays no notes
          HostEngine e; e.prepareOffline (sr, bs, 0);
          auto* c = new juce::DynamicObject(); c->setProperty ("lengthBeats", 4.0); juce::Array<juce::var> a; auto* n = new juce::DynamicObject(); n->setProperty ("beat", 0.0); n->setProperty ("len", 1.0); n->setProperty ("note", 60); n->setProperty ("vel", 100); a.add (juce::var (n)); c->setProperty ("notes", a);
          e.editAddInstance (0, 0.0, 4.0, 0.0, juce::var (c), 0); const int id = e.trackAt (0)->arrSnap.load()->insts[0].id;
          int ons = 0; const auto& in0 = e.trackAt (0)->arrSnap.load()->insts[0];
          walkInstance (in0, 0.0, 4.0, false, [&] (const NoteEv&, double) { ++ons; }, [&] (const NoteEv&, double) {}, [&] (const NoteEv&) {}); CHECK (ons == 1, "the clip plays its note");
          e.editSetInstance (id, props ({ { "muted", true } })); ons = 0;
          walkInstance (e.trackAt (0)->arrSnap.load()->insts[0], 0.0, 4.0, false, [&] (const NoteEv&, double) { ++ons; }, [&] (const NoteEv&, double) {}, [&] (const NoteEv&) {}); CHECK (ons == 0, "muted, it plays nothing");
        }
        { // fades
          HostEngine e; e.prepareOffline (sr, bs, 0); e.transport.play(); juce::String err;
          const int id = e.editImportAudio (0, f4, 0.0, err);
          e.editSetInstance (id, props ({ { "fadeIn", 1.0 }, { "fadeOut", 1.0 } }));
          auto o = run (e, 800, bs);
          CHECK (std::abs (at (o, 0.5) - 0.1f * 0.5f) < 0.005f, "half way through a 1 s fade-in the sound is at half level: ramp 0.1 x 0.5");
          CHECK (std::abs (at (o, 1.5) - 0.3f) < 0.005f, "after the fade-in it is at full level");
          CHECK (std::abs (at (o, 3.5) - 0.7f * 0.5f) < 0.01f, "half way through the 1 s fade-out it is at half level: ramp 0.7 x 0.5"); }
        { // reverse
          HostEngine e; e.prepareOffline (sr, bs, 0); e.transport.play(); juce::String err;
          const int id = e.editImportAudio (0, f4, 0.0, err);
          e.editSetInstance (id, props ({ { "reverse", true } }));
          auto o = run (e, 700, bs);
          CHECK (std::abs (at (o, 1.0) - 0.6f) < 0.01f && std::abs (at (o, 3.0) - 0.2f) < 0.01f, "reversed, the ramp falls: 0.6 at 1 s and 0.2 at 3 s"); }
        { // each copy has its own volume, fades and direction; a split puts a short fade at the cut
          HostEngine e; e.prepareOffline (sr, bs, 0); juce::String err;
          const int a = e.editImportAudio (0, f4, 0.0, err);
          const int b = e.editDuplicateInstance (a, 10.0, true, -1);
          e.editSetInstance (b, props ({ { "gainDb", -12.0 }, { "reverse", true } }));
          const auto& v = e.trackAt (0)->arrSnap.load()->insts;
          CHECK (v.size() == 2 && v[0].content == v[1].content && std::abs (v[0].gainDb) < 1e-6f && ! v[0].reverse && v[1].reverse && std::abs (v[1].gainDb + 12.0f) < 1e-4f, "two copies of one sound, different volume and direction");
          e.editSetInstance (a, props ({ { "fadeOut", 1.5 } }));
          const int c = e.editSplitInstance (a, 4.0); (void) c;
          const auto& w = e.trackAt (0)->arrSnap.load()->insts;
          CHECK (w.size() == 3 && std::abs (w[0].fadeOut) < 1e-9 && std::abs (w[1].fadeIn) < 1e-9 && std::abs (w[1].fadeOut - 1.5) < 1e-9, "a split leaves the joint without a fade (the sound carries on across it) and keeps the long fade-out at the real end");
          auto proj = e.saveProject(); HostEngine g; g.prepareOffline (sr, bs, 0); juce::String er; g.loadProject (proj, [&] (juce::String s) { er = s; });
          const auto& x = g.trackAt (0)->arrSnap.load()->insts;
          CHECK (er.isEmpty() && x.size() == 3 && std::abs (x[2].gainDb + 12.0f) < 1e-4f && x[2].reverse && std::abs (x[1].fadeOut - 1.5) < 1e-9, "gain, fades and direction survive saving and loading"); }
        f4.deleteFile();

        // finding where hits start: clicks, a steady tone, and a slow swell
        {
            AudioData d; d.rate = 44100.0; const int N = (int) (5.2 * d.rate); d.buf.setSize (1, N); d.buf.clear(); float* x = d.buf.getWritePointer (0);
            juce::Random rnd (7);
            for (double t0 : { 0.5, 1.2, 2.0 }) for (int i = 0; i < (int) (0.2 * d.rate); ++i) { const int p = (int) (t0 * d.rate) + i; if (p < N) x[p] += 0.6f * std::exp (-(float) i / (0.03f * (float) d.rate)) * (rnd.nextFloat() * 2.0f - 1.0f); }
            for (int i = 0; i < (int) (1.0 * d.rate); ++i) x[(int) (2.5 * d.rate) + i] += 0.3f * std::sin (6.2831853f * 220.0f * (float) i / (float) d.rate);
            for (int i = 0; i < (int) (1.2 * d.rate); ++i) x[(int) (3.7 * d.rate) + i] += 0.5f * ((float) i / (float) (1.2 * d.rate)) * std::sin (6.2831853f * 330.0f * (float) i / (float) d.rate);
            const auto hits = detectTransients (d, 0.5);
            auto nearTo = [&] (double want) { for (double h : hits) if (std::abs (h - want) < 0.012) return true; return false; };
            CHECK (nearTo (0.5) && nearTo (1.2) && nearTo (2.0), "the three noise clicks are found to within 12 ms");
            CHECK (nearTo (2.5) && nearTo (3.7), "so are the start of the steady tone and the start of the swell (a sound beginning out of silence is a start)");
            CHECK (hits.size() == 5, "and nothing else: not the middle of the tone, not the body of the slow swell, not the quiet gaps");
            CHECK (detectTransients (d, 0.9).size() >= detectTransients (d, 0.1).size(), "a higher sensitivity never finds fewer");
            AudioData quiet; quiet.rate = 44100.0; quiet.buf.setSize (2, 44100); quiet.buf.clear();
            CHECK (detectTransients (quiet, 0.9).empty(), "silence has no hits");
            AudioData tiny; tiny.rate = 44100.0; tiny.buf.setSize (1, 100); CHECK (detectTransients (tiny, 0.5).empty(), "a sound too short to listen to has none");
        }
    }

    // ---- AUDIO CLIPS, STAGE C: chops and quantise ----
    {
        // a test sound with four noise clicks at 0.5, 1.5, 2.25 and 3.0 s (4 s long, 48 kHz)
        auto makeClicks = [&] (const char* name)
        {
            auto f = juce::File::getSpecialLocation (juce::File::tempDirectory).getChildFile (name); f.deleteFile();
            juce::WavAudioFormat wav; auto out = f.createOutputStream();
            std::unique_ptr<juce::AudioFormatWriter> w (wav.createWriterFor (out.get(), 48000.0, 2, 32, {}, 0)); if (w) out.release();
            const int n = 4 * 48000; juce::AudioBuffer<float> b (2, n); b.clear(); juce::Random rnd (99);
            for (double t0 : { 0.5, 1.5, 2.25, 3.0 }) for (int i = 0; i < 12000; ++i) { const int p = (int) (t0 * 48000.0) + i; const float v = 0.7f * std::exp (-(float) i / 1900.0f) * (rnd.nextFloat() * 2.0f - 1.0f); b.addSample (0, p, v); b.addSample (1, p, v); }
            w->writeFromAudioSampleBuffer (b, 0, n); w.reset(); return f;
        };
        auto fc = makeClicks ("archie-clicks4.wav");
        auto props = [] (std::initializer_list<std::pair<const char*, juce::var>> kv) { auto* p = new juce::DynamicObject(); for (auto& x : kv) p->setProperty (x.first, x.second); return juce::var (p); };
        auto maxDiff = [] (const std::vector<float>& a, const std::vector<float>& b, size_t n) { float m = 0; for (size_t i = 0; i < n && i < a.size() && i < b.size(); ++i) m = std::max (m, std::abs (a[i] - b[i])); return m; };
        const int blocks = 780;                                                                                       // about 4.2 s

        std::vector<float> whole;
        { HostEngine e; e.prepareOffline (sr, bs, 0); e.transport.play(); juce::String err; e.editImportAudio (0, fc, 0.0, err); whole = run (e, blocks, bs); }

        { // chop at hits: five pieces, same sound
          HostEngine e; e.prepareOffline (sr, bs, 0); e.transport.play(); juce::String err; const int id = e.editImportAudio (0, fc, 0.0, err);
          const int depth = (int) e.undo.depth();
          const int n = e.editChopInstance (id, "hits", 0.5);
          CHECK (n == 5, "four hits make five pieces");
          const auto& v = e.trackAt (0)->arrSnap.load()->insts;
          CHECK (v.size() == 5 && std::abs (v[1].start - 1.0) < 0.02 && std::abs (v[2].start - 3.0) < 0.02 && std::abs (v[3].start - 4.5) < 0.02 && std::abs (v[4].start - 6.0) < 0.02, "the cuts are at the hits: beats 1, 3, 4.5 and 6 (0.5, 1.5, 2.25, 3.0 s at 120 bpm)");
          CHECK (v[0].content == v[4].content && v[2].content->audio == v[0].content->audio, "all five share the one sound");
          CHECK (std::abs (v[1].fadeIn) < 1e-9 && std::abs (v[0].fadeOut) < 1e-9 && v[0].fadeIn > 0.0 && v[4].fadeOut > 0.0, "the joints have no fade; the real start and end keep theirs");
          CHECK ((int) e.undo.depth() == depth + 1, "the chop is one undo step");
          auto o = run (e, blocks, bs);
          CHECK (maxDiff (whole, o, whole.size()) < 2e-4f, "the five pieces together play exactly what the whole clip did");
          e.undo.undo (); CHECK (e.trackAt (0)->arrSnap.load()->insts.size() == 1, "undo makes it one clip again"); }
        { // join: the reverse of chop
          HostEngine e; e.prepareOffline (sr, bs, 0); e.transport.play(); juce::String err; const int id = e.editImportAudio (0, fc, 0.0, err);
          e.editChopInstance (id, "hits", 0.5); juce::String jerr;
          for (int k = 0; k < 4; ++k) { const auto& v = e.trackAt (0)->arrSnap.load()->insts; int first = 0; for (size_t q = 0; q < v.size(); ++q) if (v[q].start < v[(size_t) first].start) first = (int) q; CHECK (e.editJoinInstance (v[(size_t) first].id, jerr) != 0, "a piece joins the next piece of its own sound"); }
          const auto& v = e.trackAt (0)->arrSnap.load()->insts;
          CHECK (v.size() == 1 && std::abs (v[0].start) < 1e-9 && std::abs (v[0].length - 8.0) < 0.02, "four joins make the one clip again");
          auto o = run (e, blocks, bs); CHECK (maxDiff (whole, o, whole.size()) < 2e-4f, "and it plays exactly what the first clip did");
          CHECK (e.editJoinInstance (v[0].id, jerr) == 0 && jerr.contains ("no clip after"), "the last clip has nothing to join to");
          e.undo.undo (); CHECK (e.trackAt (0)->arrSnap.load()->insts.size() == 2, "undo splits the last join again"); }
        { // joining two different sounds is refused
          HostEngine e; e.prepareOffline (sr, bs, 0); juce::String err, jerr; const int a = e.editImportAudio (0, fc, 0.0, err); e.editImportAudio (0, fc, 8.0, err);
          CHECK (e.editJoinInstance (a, jerr) == 0 && jerr.contains ("not two pieces of the same sound"), "two separate imports of a sound are not pieces of one sound"); }
        { // audio to MIDI: one note at each hit
          HostEngine e; e.prepareOffline (sr, bs, 0); juce::String err; const int id = e.editImportAudio (0, fc, 0.0, err); const int nb = e.trackCount.load();
          const int nt = e.editAudioToMidi (id, 0.5, err);
          CHECK (nt == nb && e.trackCount.load() == nb + 1, "a new track is made");
          const auto& in = e.trackAt (nt)->arrSnap.load()->insts[0]; CHECK (in.content->notes.size() == 4, "four hits make four notes");
          CHECK (std::abs (in.content->notes[0].beat - 1.0) < 0.03 && std::abs (in.content->notes[1].beat - 3.0) < 0.03, "at the beats where the hits are (0.5 s = beat 1, 1.5 s = beat 3)");
          e.undo.undo (); CHECK (e.trackCount.load() == nb, "undo takes the track away"); }
        { // join two note clips into one
          HostEngine e; e.prepareOffline (sr, bs, 0); juce::String jerr;
          auto mk = [] (double beat) { auto* c = new juce::DynamicObject(); c->setProperty ("lengthBeats", 4.0); juce::Array<juce::var> a; auto* n = new juce::DynamicObject(); n->setProperty ("beat", beat); n->setProperty ("len", 0.5); n->setProperty ("note", 60); n->setProperty ("vel", 100); a.add (juce::var (n)); c->setProperty ("notes", a); return juce::var (c); };
          e.editAddInstance (0, 0.0, 4.0, 0.0, mk (1.0), 0); e.editAddInstance (0, 4.0, 4.0, 0.0, mk (2.0), 0);
          const int id = e.trackAt (0)->arrSnap.load()->insts[0].id; CHECK (e.editJoinInstance (id, jerr) != 0, "two note clips join");
          const auto& v = e.trackAt (0)->arrSnap.load()->insts; std::vector<double> bs2; for (auto& n : v[0].content->notes) bs2.push_back (n.beat); std::sort (bs2.begin(), bs2.end());
          CHECK (v.size() == 1 && bs2.size() == 2 && std::abs (bs2[0] - 1.0) < 1e-6 && std::abs (bs2[1] - 6.0) < 1e-6 && std::abs (v[0].length - 8.0) < 1e-6, "one clip, 8 beats, the second note now at beat 6"); }
        { // unused recordings
          HostEngine e; e.prepareOffline (sr, bs, 0); auto dir = juce::File::getSpecialLocation (juce::File::tempDirectory).getChildFile ("archie-rec-test"); dir.deleteRecursively(); dir.createDirectory(); e.recordingsDir = dir;
          auto used = dir.getChildFile ("used.wav"); fc.copyFileTo (used); auto spare = dir.getChildFile ("spare.wav"); fc.copyFileTo (spare); juce::String err; e.editImportAudio (0, used, 0.0, err);
          const auto u = e.unusedRecordings(); CHECK (u.size() == 1 && u[0].getFileName() == "spare.wav", "only the recording no clip uses is listed"); dir.deleteRecursively(); }
        { // chop at the grid, with the clip reversed and made longer than the sound (looped): still the same sound
          HostEngine e; e.prepareOffline (sr, bs, 0); e.transport.play(); juce::String err; const int id = e.editImportAudio (0, fc, 0.0, err);
          e.editSetInstance (id, props ({ { "reverse", true }, { "length", 12.0 } }));
          HostEngine r; r.prepareOffline (sr, bs, 0); r.transport.play(); const int rid = r.editImportAudio (0, fc, 0.0, err); r.editSetInstance (rid, props ({ { "reverse", true }, { "length", 12.0 } }));
          const auto ref = run (r, 1200, bs);
          CHECK (e.editChopInstance (id, "grid", 2.0) == 6, "a 12-beat clip chopped every 2 beats is six pieces");
          auto o = run (e, 1200, bs);
          CHECK (maxDiff (ref, o, ref.size()) < 2e-4f, "reversed and looping, the pieces still play exactly what the one clip did"); }
        { // chop a cropped clip (offset): the same again
          HostEngine e; e.prepareOffline (sr, bs, 0); e.transport.play(); juce::String err; const int id = e.editImportAudio (0, fc, 0.0, err);
          e.editSetInstance (id, props ({ { "trimStart", 1.0 } }));
          HostEngine r; r.prepareOffline (sr, bs, 0); r.transport.play(); const int rid = r.editImportAudio (0, fc, 0.0, err); r.editSetInstance (rid, props ({ { "trimStart", 1.0 } }));
          const auto ref = run (r, blocks, bs);
          CHECK (e.editChopInstance (id, "hits", 0.5) >= 3, "a cropped clip is cut at the hits that are still inside it");
          CHECK (maxDiff (ref, run (e, blocks, bs), ref.size()) < 2e-4f, "and plays as before");
          CHECK (e.editChopInstance (999, "grid", 1.0) == 0, "a clip that is not there is not cut"); }
        { // quantise
          HostEngine e; e.prepareOffline (sr, bs, 0); juce::String err; const int id = e.editImportAudio (0, fc, 0.0, err);
          e.editChopInstance (id, "hits", 0.5);
          { auto ins = std::make_shared<HostEngine::Insts> (e.trackAt (0)->arrSnap.load()->insts); (*ins)[1].start += 0.2; (*ins)[2].start -= 0.15; (*ins)[3].start += 0.3; e.putArr (0, ins); }     // played a little off the beat
          const int moved = e.editQuantiseInstances (0, {}, 1.0, 1.0, false);
          const auto& v = e.trackAt (0)->arrSnap.load()->insts;
          CHECK (moved >= 3 && std::abs (v[1].start - 1.0) < 1e-9 && std::abs (v[2].start - 3.0) < 1e-9 && std::abs (v[3].start - 5.0) < 1e-9, "strength 1 puts the starts exactly on the whole beats");
          CHECK (v[1].fadeIn >= 0.002 && v[2].fadeOut >= 0.002, "a clip that moved gets small fades on its new edges");
          e.undo.undo ();
          e.editQuantiseInstances (0, {}, 1.0, 0.5, false);
          const auto& w = e.trackAt (0)->arrSnap.load()->insts;
          CHECK (std::abs (w[1].start - 1.1) < 1e-9, "strength 0.5 goes half way: 1.2 becomes 1.1");
          e.undo.undo ();
          { auto ins = std::make_shared<HostEngine::Insts> (e.trackAt (0)->arrSnap.load()->insts); (*ins)[0].length = 3.0; e.putArr (0, ins); }                                               // the first piece now runs over the second
          e.editQuantiseInstances (0, {}, 0.25, 1.0, true);
          const auto& x = e.trackAt (0)->arrSnap.load()->insts;
          CHECK (x[0].start + x[0].length <= x[1].start + 1e-9, "fill shortens a clip that runs into the next one");
          CHECK (e.editQuantiseInstances (0, { 99999 }, 1.0, 1.0, false) == 0 && e.editQuantiseInstances (9, {}, 1.0, 1.0, false) == 0, "nothing to move is not an error, and a missing track is refused"); }
        { // a grid chop of a note clip too
          HostEngine e; e.prepareOffline (sr, bs, 0);
          const int a = e.editAddInstance (0, 0.0, 8.0, 0.0, [] { auto* c = new juce::DynamicObject(); c->setProperty ("lengthBeats", 4.0); c->setProperty ("notes", juce::Array<juce::var>()); return juce::var (c); }(), 0);
          CHECK (e.editChopInstance (a, "grid", 4.0) == 2 && e.trackAt (0)->arrSnap.load()->insts[1].offset == 0.0, "a note clip can be cut on the grid too");
          CHECK (e.editChopInstance (e.trackAt (0)->arrSnap.load()->insts[0].id, "hits", 0.5) == 0, "but it has no hits to cut at"); }
        fc.deleteFile();
    }

    // ---- WARP (the stretching itself): a sound made at one tempo lands on the beats of another ----
    {
        auto clickTrain = [] (double bpm, double firstSec, double dur, double rate)
        {
            AudioData d; d.rate = rate; const int N = (int) (dur * rate); d.buf.setSize (2, N); d.buf.clear(); juce::Random rnd (5); const double period = 60.0 / bpm;
            for (double t = firstSec; t < dur - 0.2; t += period) for (int i = 0; i < (int) (0.12 * rate); ++i) { const int p = (int) (t * rate) + i; const float v = 0.7f * std::exp (-(float) i / (0.02f * (float) rate)) * (rnd.nextFloat() * 2.0f - 1.0f); d.buf.addSample (0, p, v); d.buf.addSample (1, p, v); }
            return d;
        };
        auto tone = [] (double hz, double dur, double rate) { AudioData d; d.rate = rate; const int N = (int) (dur * rate); d.buf.setSize (2, N); for (int i = 0; i < N; ++i) { const float v = 0.5f * std::sin (6.2831853f * (float) hz * (float) i / (float) rate); d.buf.setSample (0, i, v); d.buf.setSample (1, i, v); } return d; };
        auto crossings = [] (const AudioData& d, double fromSec, double toSec) { int n = 0; const float* x = d.buf.getReadPointer (0); for (int i = (int) (fromSec * d.rate); i + 1 < (int) (toSec * d.rate) && i + 1 < d.numSamples(); ++i) if (x[i] <= 0.0f && x[i + 1] > 0.0f) ++n; return n / (toSec - fromSec); };

        // where the loudest sample is within 30 ms of `want`, and how loud the stretcher's smear is in the 90..40 ms before it, relative to that peak
        auto peakNear = [] (const AudioData& d, double want) { const float* x = d.buf.getReadPointer (0); const int a = juce::jmax (0, (int) ((want - 0.03) * d.rate)), z = juce::jmin (d.numSamples() - 1, (int) ((want + 0.03) * d.rate));
            float pk = 0; int at = a; for (int i = a; i <= z; ++i) if (std::abs (x[i]) > pk) { pk = std::abs (x[i]); at = i; }
            float pre = 0; for (int i = juce::jmax (0, (int) ((want - 0.09) * d.rate)); i < juce::jmax (0, (int) ((want - 0.04) * d.rate)); ++i) pre = std::max (pre, std::abs (x[i]));
            return std::make_pair ((double) at / d.rate, pk > 0 ? (double) pre / pk : 0.0); };
        TempoMap tm120 (std::vector<TempoEntry> { TempoEntry { 0.0, 120.0, 4, 4 } });
        auto train = clickTrain (100.0, 0.1, 6.0, 44100.0);                                                         // clicks every 0.6 s: 100 beats a minute
        { WarpParams p; p.startBeat = 0.0; p.lengthBeats = 10.0; p.srcBpm = 100.0;                                 // 10 source beats = 6 s; at 120 they take 5 s
          auto w = renderWarp (train, p, tm120, 48000.0);
          CHECK (w != nullptr && std::abs (w->durationSec() - 5.0) < 0.01, "6 seconds at 100 bpm laid on a 120 bpm timeline for 10 beats take 5 seconds");
          int on = 0; double worstPre = 0;
          for (int k = 0; k < 10; ++k) { const double want = 0.5 * ((double) k + 0.1 * 100.0 / 60.0); const auto r = peakNear (*w, want); if (std::abs (r.first - want) < 0.012) ++on; worstPre = std::max (worstPre, r.second); }
          CHECK (on >= 10, "every click's loudest point is on its 120 bpm beat (every half second, the first a sixth of a beat in), within 12 ms");
          CHECK (worstPre < 0.6, "and what the stretcher smears ahead of a click stays below 60% of the click (the price of stretching noise bursts)"); }
        { // the tempo map changes part-way: the clicks follow it
          TempoMap tmc (std::vector<TempoEntry> { TempoEntry { 0.0, 120.0, 4, 4 }, TempoEntry { 4.0, 90.0, 4, 4 } });
          WarpParams p; p.startBeat = 0.0; p.lengthBeats = 9.0; p.srcBpm = 100.0;
          auto w = renderWarp (train, p, tmc, 48000.0); CHECK (w != nullptr, "a clip across a tempo change is rendered");
          int on = 0; for (int k = 0; k < 9; ++k) { const double want = tmc.secondsAt ((double) k + 0.1 * 100.0 / 60.0); if (std::abs (peakNear (*w, want).first - want) < 0.015) ++on; }
          CHECK (on >= 9, "after the tempo drops to 90, the clicks slow down with it and every one still sits on its beat"); }
        { // keep pitch: the sound keeps its pitch while its length changes; repitch: the pitch moves with the speed
          auto t440 = tone (440.0, 4.0, 44100.0);
          TempoMap tm60 (std::vector<TempoEntry> { TempoEntry { 0.0, 60.0, 4, 4 } });
          WarpParams p; p.startBeat = 0.0; p.lengthBeats = 8.0; p.srcBpm = 120.0;                                   // half speed: 4 s of sound over 8 beats at 60 = 8 s
          auto kp = renderWarp (t440, p, tm60, 48000.0);
          CHECK (kp != nullptr && std::abs (kp->durationSec() - 8.0) < 0.01, "stretched to twice the length");
          CHECK (std::abs (crossings (*kp, 2.0, 6.0) - 440.0) < 12.0, "keep pitch: still 440 Hz");
          p.repitch = true; auto rp = renderWarp (t440, p, tm60, 48000.0);
          CHECK (rp != nullptr && std::abs (crossings (*rp, 2.0, 6.0) - 220.0) < 8.0, "repitch: half speed is half the pitch, 220 Hz");
          p.repitch = false; p.srcBpm = 60.0; TempoMap tm120b (std::vector<TempoEntry> { TempoEntry { 0.0, 120.0, 4, 4 } }); p.lengthBeats = 4.0;      // squeezed to half the length
          auto sq = renderWarp (t440, p, tm120b, 48000.0);
          CHECK (sq != nullptr && std::abs (sq->durationSec() - 2.0) < 0.01 && std::abs (crossings (*sq, 0.5, 1.5) - 440.0) < 12.0, "squeezed to half the length, still 440 Hz"); }
        { // a plain 1:1 warp leaves the sound as it was, and the odd cases are refused
          auto t = tone (330.0, 2.0, 48000.0); TempoMap tm (std::vector<TempoEntry> { TempoEntry { 0.0, 120.0, 4, 4 } });
          WarpParams p; p.startBeat = 0.0; p.lengthBeats = 4.0; p.srcBpm = 120.0;
          auto one = renderWarp (t, p, tm, 48000.0); float worst = 0; for (int i = 6000; i < 90000; ++i) worst = std::max (worst, std::abs (one->buf.getSample (0, i) - t.buf.getSample (0, i)));
          { int bestLag = 0; float bestErr = 1e9f; for (int lag = -3000; lag <= 3000; lag += 1) { float e = 0; for (int i = 10000; i < 30000; i += 7) e += std::abs (one->buf.getSample (0, i + lag) - t.buf.getSample (0, i)); if (e < bestErr) { bestErr = e; bestLag = lag; } } std::printf ("  identity warp: best lag %d samples, worst err %.4f\n", bestLag, worst); }
          CHECK (one != nullptr && worst < 0.03f, "warping to the tempo it already has changes almost nothing (within 3% of full level)");
          p.srcBpm = 0.0; CHECK (renderWarp (t, p, tm, 48000.0) == nullptr, "no source tempo: nothing to make");
          p.srcBpm = 120.0; p.lengthBeats = 0.0; CHECK (renderWarp (t, p, tm, 48000.0) == nullptr, "no length: nothing to make");
          std::atomic<bool> stop { true }; p.lengthBeats = 4.0; CHECK (renderWarp (t, p, tm, 48000.0, &stop) == nullptr, "a cancelled render gives nothing");
          AudioData tiny; tiny.rate = 48000.0; tiny.buf.setSize (2, 1); CHECK (renderWarp (tiny, p, tm, 48000.0) == nullptr, "a one-sample sound gives nothing"); }
        { // guessing the tempo
          for (double bpm : { 90.0, 100.0, 128.0, 140.0 })
          { auto d = clickTrain (bpm, 0.2, 12.0, 44100.0); const auto g = detectTempo (d);
            char msg[96]; std::snprintf (msg, sizeof (msg), "a click train at %.0f bpm is guessed within 1.5 bpm (got %.2f)", bpm, g.bpm);
            CHECK (std::abs (g.bpm - bpm) < 1.5 && g.confidence > 0.6, msg); }
          AudioData one; one.rate = 44100.0; one.buf.setSize (1, 44100); one.buf.clear(); CHECK (detectTempo (one).bpm == 0.0, "a silent sound has no tempo");
          auto tn = tone (220.0, 5.0, 44100.0); CHECK (detectTempo (tn).bpm == 0.0 || detectTempo (tn).confidence < 0.5, "a steady tone has no beat to find"); }
    }

    // ---- WARP in the engine: a clip that follows the tempo, rendered in the background ----
    {
        auto makeTrain = [&] (const char* name, double bpm, double firstSec, double dur)
        {
            auto f = juce::File::getSpecialLocation (juce::File::tempDirectory).getChildFile (name); f.deleteFile();
            juce::WavAudioFormat wav; auto out = f.createOutputStream();
            std::unique_ptr<juce::AudioFormatWriter> w (wav.createWriterFor (out.get(), 44100.0, 2, 32, {}, 0)); if (w) out.release();
            const int n = (int) (dur * 44100.0); juce::AudioBuffer<float> b (2, n); b.clear(); juce::Random rnd (5);
            for (double t = firstSec; t < dur - 0.2; t += 60.0 / bpm) for (int i = 0; i < 5292; ++i) { const int p = (int) (t * 44100.0) + i; const float v = 0.7f * std::exp (-(float) i / 882.0f) * (rnd.nextFloat() * 2.0f - 1.0f); b.addSample (0, p, v); b.addSample (1, p, v); }
            w->writeFromAudioSampleBuffer (b, 0, n); w.reset(); return f;
        };
        auto peakNear = [] (const std::vector<float>& x, double want) { const int a = juce::jmax (0, (int) ((want - 0.03) * 48000.0)), z = juce::jmin ((int) x.size() - 1, (int) ((want + 0.03) * 48000.0)); float pk = 0; int at = a; for (int i = a; i <= z; ++i) if (std::abs (x[(size_t) i]) > pk) { pk = std::abs (x[(size_t) i]); at = i; } return (double) at / 48000.0; };
        auto props = [] (std::initializer_list<std::pair<const char*, juce::var>> kv) { auto* p = new juce::DynamicObject(); for (auto& x : kv) p->setProperty (x.first, x.second); return juce::var (p); };
        auto f100 = makeTrain ("archie-train100.wav", 100.0, 0.1, 6.0);                                                // clicks every 0.6 s

        { HostEngine e; e.prepareOffline (sr, bs, 0); e.transport.play(); juce::String err; e.undo.clock = [] { static double c = 0; c += 5000; return c; };          // (every edit its own undo step)
          const int id = e.editImportAudio (0, f100, 0.0, err);
          e.editSetInstance (id, props ({ { "warp", true }, { "srcBpm", 100.0 }, { "length", 10.0 } }));
          CHECK (e.warpBusy() || e.trackAt (0)->arrSnap.load()->insts[0].warped != nullptr, "switching warp on asks for a background render");
          { auto v = e.arrangementVar (*e.trackAt (0)); auto st = v.getProperty ("instances", {})[0].getProperty ("warpState", {}).toString(); CHECK (st == "rendering" || st == "ready", "and the clip says so (rendering, then ready)"); }
          e.waitForWarp();
          const auto& in = e.trackAt (0)->arrSnap.load()->insts[0];
          CHECK (in.warped != nullptr && ! e.warpBusy(), "the render arrives and is attached to the clip");
          auto o = run (e, 1100, bs);                                                                                    // 5.9 s
          int on = 0; for (int k = 0; k < 10; ++k) { const double want = 0.5 * ((double) k + 0.1 * 100.0 / 60.0); if (std::abs (peakNear (o, want) - want) < 0.012) ++on; }
          CHECK (on >= 10, "the clicks sound on the 120 bpm beats");
          // the tempo drops to 90: the clip is made again for the new tempo
          e.editTempo ("Tempo", "bpm", [&] { e.transport.setBpm (90.0); });
          CHECK (e.trackAt (0)->arrSnap.load()->insts[0].warped == nullptr && e.warpBusy(), "a tempo change drops the old render and asks for a new one (until then it plays unwarped)");
          e.waitForWarp(); CHECK (e.trackAt (0)->arrSnap.load()->insts[0].warped != nullptr, "the new render arrives");
          e.transport.locate (0.0); auto o2 = run (e, 1300, bs);
          int on2 = 0; for (int k = 0; k < 8; ++k) { const double want = (60.0 / 90.0) * ((double) k + 0.1 * 100.0 / 60.0); if (std::abs (peakNear (o2, want) - want) < 0.015) ++on2; }
          CHECK (on2 >= 8, "at 90 bpm they sound on the 90 bpm beats");
          e.undo.undo (); e.waitForWarp(); CHECK (std::abs (e.transport.tempoEntries()[0].bpm - 120.0) < 1e-9 && e.trackAt (0)->arrSnap.load()->insts[0].warped != nullptr, "undoing the tempo change brings the 120 bpm render back");
          // turning warp off plays the sound as it was
          e.editSetInstance (id, props ({ { "warp", false } })); CHECK (e.trackAt (0)->arrSnap.load()->insts[0].warped == nullptr && ! e.trackAt (0)->arrSnap.load()->insts[0].warp, "warp off drops the render");
          e.undo.undo (); e.waitForWarp (); CHECK (e.trackAt (0)->arrSnap.load()->insts[0].warp && e.trackAt (0)->arrSnap.load()->insts[0].warped != nullptr, "and undo turns it back on, rendered again"); }
        { // settings survive saving; the render is remade on loading
          HostEngine e; e.prepareOffline (sr, bs, 0); juce::String err;
          const int id = e.editImportAudio (0, f100, 0.0, err); e.editSetInstance (id, props ({ { "warp", true }, { "srcBpm", 100.0 }, { "repitch", true }, { "length", 10.0 } })); e.waitForWarp();
          auto proj = e.saveProject(); HostEngine g; g.prepareOffline (sr, bs, 0); juce::String er; g.loadProject (proj, [&] (juce::String s) { er = s; });
          const auto& x = g.trackAt (0)->arrSnap.load()->insts[0];
          CHECK (er.isEmpty() && x.warp && x.repitch && std::abs (x.srcBpm - 100.0) < 1e-9, "warp, source tempo and repitch are saved and loaded");
          g.waitForWarp (); CHECK (g.trackAt (0)->arrSnap.load()->insts[0].warped != nullptr, "and the render is made again after loading"); }
        { // sourceBeats: "this whole sound is 10 beats" sets the source tempo; chopping a warped clip at its hits cuts where the hits sit on the timeline
          HostEngine e; e.prepareOffline (sr, bs, 0); juce::String err;
          const int id = e.editImportAudio (0, f100, 0.0, err);
          e.editSetInstance (id, props ({ { "warp", true }, { "sourceBeats", 10.0 } }));
          CHECK (std::abs (e.trackAt (0)->arrSnap.load()->insts[0].srcBpm - 100.0) < 0.01, "a 6-second sound said to be 10 beats long was made at 100 bpm");
          e.editSetInstance (id, props ({ { "length", 10.0 } })); e.waitForWarp ();
          CHECK (e.editChopInstance (id, "hits", 0.5) >= 9, "a warped clip is chopped at its hits");
          const auto& v = e.trackAt (0)->arrSnap.load()->insts; bool ok = true;
          for (size_t i = 1; i < v.size() && i < 10; ++i) { const double want = (double) (i - 1) + 0.1 * 100.0 / 60.0; if (std::abs (v[i].start - want) > 0.03) ok = false; }
          CHECK (ok, "and the cuts are at the beats the hits fall on after warping (1/6 beat, 1 1/6, 2 1/6, ...)");
          e.waitForWarp (); }
        { // a clip with no source tempo, and a non-audio clip, are left alone
          HostEngine e; e.prepareOffline (sr, bs, 0); juce::String err;
          const int id = e.editImportAudio (0, f100, 0.0, err); e.editSetInstance (id, props ({ { "warp", true } }));
          CHECK (e.trackAt (0)->arrSnap.load()->insts[0].srcBpm > 0.0, "warp switched on without a source tempo assumes the project tempo, so nothing changes");
          const int m = e.editAddInstance (1, 0.0, 4.0, 0.0, [] { auto* c = new juce::DynamicObject(); c->setProperty ("lengthBeats", 4.0); return juce::var (c); }(), 0);
          e.editSetInstance (m, props ({ { "warp", true }, { "srcBpm", 100.0 } }));
          CHECK (! e.trackAt (1)->arrSnap.load()->insts[0].warp || e.trackAt (1)->arrSnap.load()->insts[0].warped == nullptr, "a note clip is never warped");
          e.waitForWarp (); }
        f100.deleteFile();
    }

    // ---- WARP MARKERS: pulling a drifting sound onto the grid ----
    {
        { // the mapping itself
          WarpMap m; m.slope = 2.0; m.pts = { { 1.0, 0.0 }, { 2.0, 3.0 } };                                       // between 1 s and 2 s: 3 beats (3 beats a second), outside: 2 beats a second
          CHECK (std::abs (m.f (1.5) - 1.5) < 1e-12 && std::abs (m.f (2.0) - 3.0) < 1e-12 && std::abs (m.f (0.5) - (-1.0)) < 1e-12 && std::abs (m.f (3.0) - 5.0) < 1e-12, "between markers the sound runs at its own rate; outside them at the source tempo");
          for (double s : { 0.2, 1.0, 1.3, 2.0, 2.7 }) CHECK (std::abs (m.inv (m.f (s)) - s) < 1e-12, "f and inv undo each other");
          WarpMap loop; loop.slope = 2.0; const double L = 4.0;                                                    // a plain 4 s sound at 2 beats a second loops every 8 beats
          CHECK (std::abs (warpSecAt (loop, 9.0, L) - 4.5) < 1e-12 && std::abs (warpBeatAt (loop, 4.5, L) - 9.0) < 1e-12, "a looping sound's second pass continues the mapping"); }
        { // a hand-played click track that drifts: clicks at 0.1, 0.65, 1.25, 1.8 s
          AudioData d; d.rate = 44100.0; d.buf.setSize (2, (int) (3.0 * d.rate)); d.buf.clear(); juce::Random rnd (3);
          for (double t : { 0.1, 0.65, 1.25, 1.8 }) for (int i = 0; i < 4000; ++i) { const float v = 0.7f * std::exp (-(float) i / 700.0f) * (rnd.nextFloat() * 2.0f - 1.0f); d.buf.addSample (0, (int) (t * d.rate) + i, v); d.buf.addSample (1, (int) (t * d.rate) + i, v); }
          TempoMap tm120 (std::vector<TempoEntry> { TempoEntry { 0.0, 120.0, 4, 4 } });
          WarpParams p; p.startBeat = 0.0; p.lengthBeats = 3.5; p.srcBpm = 100.0; p.offsetSec = 0.1;                // the clip starts at the first click
          p.markers = { { 0.1, 0.1 * 100.0 / 60.0 }, { 0.65, 0.1 * 100.0 / 60.0 + 1.0 }, { 1.25, 0.1 * 100.0 / 60.0 + 2.0 }, { 1.8, 0.1 * 100.0 / 60.0 + 3.0 } };
          auto loose = p; loose.markers.clear();
          auto unpinned = renderWarp (d, loose, tm120, 48000.0), pinned = renderWarp (d, p, tm120, 48000.0);
          auto peakAt = [] (const AudioData& w, double want) { const float* x = w.buf.getReadPointer (0); const int a = juce::jmax (0, (int) ((want - 0.04) * w.rate)), z = juce::jmin (w.numSamples() - 1, (int) ((want + 0.04) * w.rate)); float pk = 0; int at = a; for (int i = a; i <= z; ++i) if (std::abs (x[i]) > pk) { pk = std::abs (x[i]); at = i; } return (double) at / w.rate; };
          double worstUn = 0, worstPin = 0; for (int k = 0; k < 4; ++k) { worstUn = std::max (worstUn, std::abs (peakAt (*unpinned, 0.5 * k + 0.0) - 0.5 * k)); worstPin = std::max (worstPin, std::abs (peakAt (*pinned, 0.5 * k) - 0.5 * k)); }
          CHECK (pinned != nullptr && worstPin < 0.012, "with a marker on each click, every click sounds exactly on a beat of the 120 bpm grid (within 12 ms)");
          CHECK (worstUn > 0.03, "without markers the same clicks are off the beats (they drifted: the markers did the work)"); }
        { // the engine: add, move, remove, undo, refusals, and the clip's front stays put
          auto mk = [&] (const char* name) { auto f = juce::File::getSpecialLocation (juce::File::tempDirectory).getChildFile (name); f.deleteFile(); juce::WavAudioFormat wav; auto out = f.createOutputStream();
              std::unique_ptr<juce::AudioFormatWriter> w (wav.createWriterFor (out.get(), 44100.0, 2, 32, {}, 0)); if (w) out.release(); const int n = 6 * 44100; juce::AudioBuffer<float> b (2, n); b.clear(); juce::Random rnd (8);
              for (double t : { 0.1, 0.7, 1.3, 1.9, 2.5 }) for (int i = 0; i < 4000; ++i) { const float v = 0.7f * std::exp (-(float) i / 700.0f) * (rnd.nextFloat() * 2.0f - 1.0f); b.addSample (0, (int) (t * 44100.0) + i, v); b.addSample (1, (int) (t * 44100.0) + i, v); }
              w->writeFromAudioSampleBuffer (b, 0, n); w.reset(); return f; };
          auto f = mk ("archie-drift.wav");
          auto props = [] (std::initializer_list<std::pair<const char*, juce::var>> kv) { auto* p = new juce::DynamicObject(); for (auto& x : kv) p->setProperty (x.first, x.second); return juce::var (p); };
          HostEngine e; e.prepareOffline (sr, bs, 0); e.undo.clock = [] { static double c = 0; c += 5000; return c; }; juce::String err;
          const int id = e.editImportAudio (0, f, 0.0, err);
          CHECK (e.editWarpMarker (id, "set", 0.7, 1.0, -1).isNotEmpty(), "markers need warp and a source tempo first (refused otherwise)");
          e.editSetInstance (id, props ({ { "warp", true }, { "srcBpm", 100.0 }, { "trimStart", 0.0 } })); e.waitForWarp();
          const double front0 = HostEngine::warpMapOf (e.trackAt (0)->arrSnap.load()->insts[0]).f (0.0);
          CHECK (e.editWarpMarker (id, "set", 0.7, 1.5, -1).isEmpty(), "a marker pins the click at 0.7 s of the sound to timeline beat 1.5");
          { const auto& a = e.trackAt (0)->arrSnap.load()->insts[0]; CHECK (a.markers.size() == 2 && std::abs (HostEngine::warpMapOf (a).f (0.0) - front0) < 1e-9, "the first marker also pins the front of the clip (so the stretch happens between them) and the front stays where it was"); }
          { const auto hits = e.hitBeats (e.trackAt (0)->arrSnap.load()->insts[0], 0.5); bool pinned = false; for (auto& h : hits) if (std::abs (h.first - 0.7) < 0.03 && std::abs (h.second - 1.5) < 0.05) pinned = true; CHECK (pinned, "the hit at 0.7 s is now heard at beat 1.5 on the timeline"); }
          { const auto v = e.arrangementVar (*e.trackAt (0)); const auto mk0 = v.getProperty ("instances", {})[0].getProperty ("markers", {}); CHECK (mk0.getArray() != nullptr && mk0.getArray()->size() == 2 && std::abs ((double) (*mk0.getArray())[1].getProperty ("beat", 0.0) - 1.5) < 1e-9, "the clip reports its markers, the pinned one at timeline beat 1.5"); }
          CHECK (e.editWarpMarker (id, "set", 0.703, 1.8, -1).isEmpty() && e.trackAt (0)->arrSnap.load()->insts[0].markers.size() == 2, "setting nearTo an existing marker moves it instead of adding another");
          CHECK (e.editWarpMarker (id, "set", 1.3, 2.9, -1).isEmpty() && e.trackAt (0)->arrSnap.load()->insts[0].markers.size() == 3, "a second pinned moment adds a marker");
          CHECK (e.editWarpMarker (id, "set", 1.9, 2.0, -1).isNotEmpty(), "a marker that would run the sound backwards is refused");
          CHECK (e.editWarpMarker (id, "set", 1.9, 30.0, -1).isNotEmpty(), "and so is one that would stretch a part more than 4 times");
          CHECK (e.editWarpMarker (id, "set", 99.0, 4.0, -1).isNotEmpty(), "and one outside the sound");
          const size_t depth = e.undo.depth();
          e.editWarpMarker (id, "remove", 0, 0, 2); CHECK (e.trackAt (0)->arrSnap.load()->insts[0].markers.size() == 2 && e.undo.depth() == depth + 1, "a marker is removed, as one undo step");
          e.undo.undo (); CHECK (e.trackAt (0)->arrSnap.load()->insts[0].markers.size() == 3, "undo brings it back");
          CHECK (e.trackAt (0)->arrSnap.load()->insts[0].warped == nullptr && e.warpBusy(), "every marker edit asks for a new render");
          e.waitForWarp (); CHECK (e.trackAt (0)->arrSnap.load()->insts[0].warped != nullptr, "which arrives");
          // markers survive saving and loading
          auto proj = e.saveProject(); HostEngine g; g.prepareOffline (sr, bs, 0); juce::String er; g.loadProject (proj, [&] (juce::String s) { er = s; });
          CHECK (er.isEmpty() && g.trackAt (0)->arrSnap.load()->insts[0].markers.size() == 3, "markers are saved with the project");
          g.waitForWarp ();
          // splitting keeps the sound where it was: the second half's start is the sound position heard at the cut
          const int sp = e.editSplitInstance (id, 3.0); CHECK (sp != 0, "a clip with markers is split");
          { const auto& v = e.trackAt (0)->arrSnap.load()->insts; const double want = e.audioPosAt (v[0], 3.0); CHECK (std::abs (v[1].offset - want) < 1e-6, "the second half begins at the point of the sound that was heard at the cut"); }
          e.editWarpMarker (e.trackAt (0)->arrSnap.load()->insts[0].id, "clear", 0, 0, -1); CHECK (e.trackAt (0)->arrSnap.load()->insts[0].markers.empty(), "clear removes them all");
          e.waitForWarp ();
          f.deleteFile(); }
    }

    // ---- NOTES: quantise and transpose a note clip ----
    {
        HostEngine e; e.prepareOffline (sr, bs, 0); e.undo.clock = [] { static double c = 0; c += 5000; return c; };
        auto* c = new juce::DynamicObject(); c->setProperty ("lengthBeats", 4.0); juce::Array<juce::var> arr;
        auto note = [&] (double beat, double len, int nn) { auto* n = new juce::DynamicObject(); n->setProperty ("beat", beat); n->setProperty ("len", len); n->setProperty ("note", nn); n->setProperty ("vel", 100); arr.add (juce::var (n)); };
        note (0.07, 0.4, 60); note (1.2, 0.5, 64); note (1.9, 0.3, 67); note (3.98, 0.02, 72);                      // played a little loose; the last one is almost at the clip's end
        c->setProperty ("notes", arr); const juce::var clip (c);
        const int a = e.editAddInstance (0, 0.0, 4.0, 0.0, clip, 0); (void) a;
        const int cid = e.trackAt (0)->arrSnap.load()->insts[0].content->id;
        const int b = e.editAddInstance (1, 8.0, 4.0, 0.0, {}, cid); (void) b;
        auto notes = [&] { return e.trackAt (0)->arrSnap.load()->insts[0].content->notes; };

        CHECK (e.editQuantiseNotes (cid, 0.25, 1.0, false, {}), "the notes are quantised to sixteenths");
        { const auto n = notes(); CHECK (std::abs (n[0].beat - 0.0) < 1e-9 && std::abs (n[1].beat - 1.25) < 1e-9 && std::abs (n[2].beat - 2.0) < 1e-9, "starts land on the grid (0.07 -> 0, 1.2 -> 1.25, 1.9 -> 2.0)");
          CHECK (std::abs (n[0].len - 0.47) < 1e-9, "a note's end stays where it was when only the start is quantised (0.07 + 0.4 = 0.47)");
          CHECK (n[3].beat < 4.0 && n[3].beat + n[3].len <= 4.0 + 1e-9, "a note that would land on the end of the clip stays inside it"); }
        CHECK (e.trackAt (1)->arrSnap.load()->insts[0].content->notes[1].beat == notes()[1].beat, "every copy of the clip, on any track, changed with it");
        e.undo.undo (); CHECK (std::abs (notes()[1].beat - 1.2) < 1e-9, "undo puts the loose playing back");
        CHECK (e.editQuantiseNotes (cid, 0.5, 0.5, false, {}), "half-strength quantise");
        CHECK (std::abs (notes()[1].beat - 1.225) > 0 && std::abs (notes()[1].beat - 1.1) < 1e-9, "1.2 goes half way to the half-beat grid line at 1.0: 1.1");
        e.undo.undo ();
        CHECK (e.editQuantiseNotes (cid, 0.25, 1.0, true, { 1 }), "quantising starts and ends of one note");
        { const auto n = notes(); CHECK (std::abs (n[1].beat - 1.25) < 1e-9 && std::abs (n[1].beat + n[1].len - 1.75) < 1e-9 && std::abs (n[0].beat - 0.07) < 1e-9, "only that note moved: start 1.25, end 1.75 (1.7 rounds to 1.75); the others are as played"); }
        e.undo.undo ();
        CHECK (! e.editQuantiseNotes (cid, 0.0, 1.0, false, {}) && ! e.editQuantiseNotes (999, 0.25, 1.0, false, {}), "no grid, or no such clip: refused");
        e.editQuantiseNotes (cid, 0.25, 1.0, false, {}); CHECK (! e.editQuantiseNotes (cid, 0.25, 1.0, false, {}), "already on the grid: nothing to do (and no undo step)");
        e.undo.undo ();

        CHECK (e.editTransposeNotes (cid, 12, {}) && notes()[0].note == 72 && notes()[3].note == 84, "up an octave");
        CHECK (e.editTransposeNotes (cid, 100, {}) && notes()[0].note == 127 && notes()[3].note == 127, "held within 0 to 127");
        e.undo.undo (); e.undo.undo ();
        CHECK (e.editTransposeNotes (cid, -2, { 0, 2 }) && notes()[0].note == 58 && notes()[1].note == 64 && notes()[2].note == 65, "only the chosen notes move");
        CHECK (! e.editTransposeNotes (cid, 0, {}), "no change is not an edit");
        // drags merge into one undo step (the same kind of edit within 700 ms), a different kind does not
        HostEngine f; f.prepareOffline (sr, bs, 0); const int ia = f.editAddInstance (0, 0.0, 4.0, 0.0, clip, 0); (void) ia;
        const int fid = f.trackAt (0)->arrSnap.load()->insts[0].content->id; double now = 0; f.undo.clock = [&] { return now; };
        for (int k = 1; k <= 10; ++k) { now += 30; auto* cc = new juce::DynamicObject (); cc->setProperty ("lengthBeats", 4.0); juce::Array<juce::var> an; auto* n0 = new juce::DynamicObject (); n0->setProperty ("beat", 0.1 * k); n0->setProperty ("len", 0.5); n0->setProperty ("note", 60); an.add (juce::var (n0)); cc->setProperty ("notes", an); f.editContent (fid, juce::var (cc)); }
        CHECK (f.undo.depth() == 2, "ten edits of one gesture (a note dragged) are one undo step (plus the one that added the clip)");
        f.undo.undo (); CHECK (f.trackAt (0)->arrSnap.load()->insts[0].content->notes.size() == 4, "and one undo puts back the notes as they were before the drag");
    }

    // ---- RECORDING AUDIO from the sound-card input ----
    {
        auto runIn = [&] (HostEngine& e, int blocks, int phase0, float amp) {
            std::vector<float> l ((size_t) bs), r ((size_t) bs), ol ((size_t) bs), orr ((size_t) bs); const float* in[2] = { l.data(), r.data() }; float* o[2] = { ol.data(), orr.data() };
            for (int b = 0; b < blocks; ++b) { for (int i = 0; i < bs; ++i) { const float v = amp * std::sin (6.2831853f * 440.0f * (float) (phase0 + b * bs + i) / (float) sr); l[(size_t) i] = v; r[(size_t) i] = -v; } e.processOffline (in, 2, o, 2, bs); }
        };
        auto tmp = juce::File::getSpecialLocation (juce::File::tempDirectory).getChildFile ("archie-rec-test"); tmp.deleteRecursively(); tmp.createDirectory();
        { HostEngine e; e.prepareOffline (sr, bs, 0); e.recordingsDir = tmp; e.recordLatencyOverrideSec = 0.01;
          e.trackAt (0)->source.store (1); e.editChainParam (0, "armed", true);
          CHECK (e.trackAt (0)->audioRec.load() != nullptr, "arming a track makes its recording buffer (off the audio thread)");
          e.transport.setRecording (true); e.transport.play(); runIn (e, 60, 0, 0.5f); e.drainAudioRecordings (true);
          CHECK (e.trackAt (0)->arrSnap.load()->insts.empty(), "while recording goes on nothing is a clip yet");
          e.transport.stop(); runIn (e, 3, 0, 0.0f); e.drainAudioRecordings (true);
          const auto& v = e.trackAt (0)->arrSnap.load()->insts;
          CHECK (v.size() == 1 && v[0].content->audio != nullptr, "when recording stops, the take becomes a clip on the armed track");
          if (v.size() == 1)
          {
              const auto& a = *v[0].content->audio;
              CHECK (std::abs (v[0].start) < 0.01 && std::abs (a.durationSec() - 60.0 * bs / sr) < 0.02, "it begins where recording began (beat 0) and is as long as what was heard (0.32 s)");
              CHECK (std::abs (v[0].offset - 0.01) < 1e-6 && v[0].length > 0.0, "and starts 10 ms into the take: the sound card's round-trip delay is taken off the front");
              float pk = 0; for (int i = 100; i < a.numSamples() - 100; ++i) pk = std::max (pk, std::abs (a.buf.getSample (0, i))); CHECK (std::abs (pk - 0.5f) < 0.01f, "what was recorded is what was heard (a 0.5 sine)");
              CHECK (std::abs (a.buf.getSample (1, 300) + a.buf.getSample (0, 300)) < 1e-5f, "both channels are kept (the right was the left upside down)");
              CHECK (juce::File (a.path).existsAsFile() && a.path.contains ("take-"), "and it is saved as a sound file in the recordings folder");
          }
          CHECK (e.takeSerial.load() == 1 && e.lastTakeName.startsWith ("take-"), "the take is announced (for the console)");
          e.undo.undo (); CHECK (e.trackAt (0)->arrSnap.load()->insts.empty(), "undo removes the take from the timeline (the file stays)"); }
        { // not armed, or the source is not the input: nothing is kept
          HostEngine e; e.prepareOffline (sr, bs, 0); e.recordingsDir = tmp; e.recordLatencyOverrideSec = 0.0;
          e.trackAt (0)->source.store (1); e.trackAt (1)->source.store (2); e.editChainParam (1, "armed", true); e.transport.setRecording (true); e.transport.play(); runIn (e, 40, 0, 0.5f); e.transport.stop(); runIn (e, 2, 0, 0.0f); e.drainAudioRecordings (true);
          CHECK (e.trackAt (0)->arrSnap.load()->insts.empty() && e.trackAt (1)->arrSnap.load()->insts.empty(), "a track that is not armed, or is not listening to the input, records nothing"); }
        { // recording only counts while the song is recording AND playing
          HostEngine e; e.prepareOffline (sr, bs, 0); e.recordingsDir = tmp; e.recordLatencyOverrideSec = 0.0;
          e.trackAt (0)->source.store (1); e.editChainParam (0, "armed", true); e.transport.play(); runIn (e, 40, 0, 0.5f); e.transport.stop(); e.drainAudioRecordings (true);
          CHECK (e.trackAt (0)->arrSnap.load()->insts.empty(), "playing without record keeps nothing");
          e.transport.setRecording (true); runIn (e, 20, 0, 0.5f); e.drainAudioRecordings (true); CHECK (e.trackAt (0)->arrSnap.load()->insts.empty(), "record armed but stopped keeps nothing"); }
        { // the count-in is not recorded: the take begins at the start of the song
          HostEngine e; e.prepareOffline (sr, bs, 0); e.recordingsDir = tmp; e.recordLatencyOverrideSec = 0.0;
          e.trackAt (0)->source.store (1); e.editChainParam (0, "armed", true); e.transport.countInBars.store (1); e.transport.setRecording (true); e.transport.play();
          runIn (e, 520, 0, 0.5f); e.transport.stop(); runIn (e, 2, 0, 0.0f); e.drainAudioRecordings (true);
          const auto& v = e.trackAt (0)->arrSnap.load()->insts;
          CHECK (v.size() == 1 && std::abs (v[0].start) < 0.05, "after a one-bar count-in the take still begins at beat 0, not before it");
          if (v.size() == 1) CHECK (v[0].content->audio->durationSec() < 2.5, "and does not include the count-in's two seconds"); }
        tmp.deleteRecursively();
    }

    // ---- AUTOMATION: a lane moves a channel's level, pan or a plug-in parameter with the song ----
    {
        struct ParamProc : TestProc
        {
            ParamProc() : TestProc (2, 2, 0) { addParameter (amt = new juce::AudioParameterFloat (juce::ParameterID { "amt", 1 }, "Amount", 0.0f, 1.0f, 0.0f)); addParameter (other = new juce::AudioParameterFloat (juce::ParameterID { "oth", 1 }, "Other", 0.0f, 1.0f, 0.5f)); }
            void processBlock (juce::AudioBuffer<float>& b, juce::MidiBuffer&) override { b.applyGain (amt->get()); }
            juce::AudioParameterFloat* amt = nullptr; juce::AudioParameterFloat* other = nullptr;
        };
        auto pts = [] (std::initializer_list<std::pair<double, double>> v) { juce::Array<juce::var> a; for (auto& p : v) { auto* o = new juce::DynamicObject(); o->setProperty ("beat", p.first); o->setProperty ("value", p.second); a.add (juce::var (o)); } return juce::var (a); };
        auto peakIn = [] (const std::vector<float>& o, double fromSec, double toSec) { float pk = 0; for (size_t i = (size_t) (fromSec * 48000.0); i < (size_t) (toSec * 48000.0) && i < o.size(); ++i) pk = std::max (pk, std::abs (o[i])); return pk; };

        { HostEngine e; e.prepareOffline (sr, bs, 0); e.undo.clock = [] { static double c = 0; c += 5000; return c; };
          e.trackAt (0)->source.store (2);
          const int lane = e.editAutoAddLane (0, 0, -1, -1);
          CHECK (lane > 0 && e.editAutoAddLane (0, 0, -1, -1) == lane, "a level lane is added, and asking again for the same target gives the same lane");
          CHECK (e.editAutoSetPoints (0, lane, pts ({ { 4.0, -12.0 }, { 0.0, 0.0 } })), "its points are set (given out of order: sorted for you)");
          CHECK (e.trackAt (0)->autoSnap.load()->lanes[0].pts[0].beat == 0.0, "sorted");
          e.transport.play();
          auto o = run (e, 1000, bs);                                                                                  // 5.3 s
          CHECK (std::abs (peakIn (o, 0.1, 0.19) - 0.25f * std::pow (10.0f, -0.6f / 20.0f)) < 0.012f, "at the start the level is about 0 dB");
          CHECK (std::abs (peakIn (o, 1.0, 1.09) - 0.25f * std::pow (10.0f, -6.0f / 20.0f)) < 0.012f, "half way along the line (beat 2) it is -6 dB");
          CHECK (std::abs (peakIn (o, 2.2, 2.3) - 0.25f * std::pow (10.0f, -12.0f / 20.0f)) < 0.006f && std::abs (peakIn (o, 4.5, 5.0) - 0.25f * std::pow (10.0f, -12.0f / 20.0f)) < 0.006f, "after the last point it holds the last value (-12 dB)");
          CHECK (std::abs (e.trackAt (0)->gainDb.load() + 12.0f) < 0.1f, "and the fader itself shows where the automation has put it");
          e.editAutoLane (0, lane, "enable", false); e.trackAt (0)->gainDb.store (0.0f); e.transport.locate (0.0); auto o2 = run (e, 300, bs);
          CHECK (std::abs (peakIn (o2, 0.5, 1.0) - 0.25f) < 0.01f, "with the lane switched off the fader is the person's again (0 dB)");
          e.undo.undo (); CHECK (e.trackAt (0)->autoSnap.load()->lanes[0].on == true, "undoing the off switch turns the lane back on"); }
        { HostEngine e; e.prepareOffline (sr, bs, 0); e.trackAt (0)->source.store (2);
          const int lane = e.editAutoAddLane (0, 1, -1, -1); e.editAutoSetPoints (0, lane, pts ({ { 0.0, -1.0 }, { 4.0, 1.0 } })); e.transport.play();
          auto l = run (e, 1000, bs, 0), r = run (e, 1, bs, 1); (void) r;
          HostEngine f; f.prepareOffline (sr, bs, 0); f.trackAt (0)->source.store (2); const int ln = f.editAutoAddLane (0, 1, -1, -1); f.editAutoSetPoints (0, ln, pts ({ { 0.0, -1.0 }, { 4.0, 1.0 } })); f.transport.play(); auto rr = run (f, 1000, bs, 1);
          CHECK (peakIn (l, 0.03, 0.06) > 0.2f && peakIn (rr, 0.03, 0.06) < 0.08f, "pan automated from hard left: at the start the sound is almost all in the left ear");
          CHECK (peakIn (l, 3.8, 3.99) < 0.06f && peakIn (rr, 3.8, 3.99) > 0.2f, "and by the end of the lane it is in the right ear"); }
        { HostEngine e; e.prepareOffline (sr, bs, 0); auto pp = std::make_unique<ParamProc>(); auto* raw = pp.get(); e.addProcessor (0, std::move (pp), "param test");
          const int lane = e.editAutoAddLane (0, 2, 0, 0);
          CHECK (lane > 0 && e.trackAt (0)->autoSnap.load()->lanes[0].label.contains ("Amount"), "a plug-in parameter lane is added and named after the parameter");
          CHECK (e.editAutoAddLane (0, 2, 0, 9) == 0 && e.editAutoAddLane (0, 2, 5, 0) == 0 && e.editAutoAddLane (kMasterChain, 0, -1, -1) == 0 && e.editAutoAddLane (0, 9, -1, -1) == 0, "a parameter that does not exist, a rack slot that does not exist, the master (no lanes yet) and an unknown kind are refused");
          e.editAutoSetPoints (0, lane, pts ({ { 0.0, 0.0 }, { 4.0, 1.0 }, { 8.0, 0.2 } }));
          auto at = [&] (double b) { e.transport.locate (b); run (e, 1, bs); e.applyParamAutomation(); };
          at (2.0); CHECK (std::abs (raw->amt->get() - 0.5f) < 1e-3f, "moving the playhead to beat 2 puts the parameter half way up");
          at (6.0); CHECK (std::abs (raw->amt->get() - 0.6f) < 1e-3f, "beat 6 is between 1.0 and 0.2: 0.6");
          at (20.0); CHECK (std::abs (raw->amt->get() - 0.2f) < 1e-3f, "past the end it holds 0.2");
          CHECK (std::abs (raw->other->get() - 0.5f) < 1e-6f, "other parameters are not touched");
          const auto pv = e.paramsVar (0, 0); CHECK (pv.getArray() != nullptr && pv.getArray()->size() == 2 && (*pv.getArray())[0].getProperty ("name", {}).toString() == "Amount", "the parameters of a rack slot can be listed to choose from");
          CHECK (e.editAutoLane (0, lane, "remove", false) && e.trackAt (0)->autoSnap.load()->lanes.empty(), "a lane is removed"); e.undo.undo (); CHECK (e.trackAt (0)->autoSnap.load()->lanes.size() == 1, "and undo brings it back with its points"); }
        { // a drag of a point is ONE undo step; lanes are saved and loaded
          HostEngine e; e.prepareOffline (sr, bs, 0); const int lane = e.editAutoAddLane (1, 0, -1, -1); double now = 0; e.undo.clock = [&] { return now; };
          for (int k = 1; k <= 8; ++k) { now += 30; e.editAutoSetPoints (1, lane, pts ({ { 0.0, 0.0 }, { 4.0, -3.0 * k } })); }
          CHECK (e.undo.depth() == 2, "eight edits of one drag are one step (plus adding the lane)");
          e.undo.undo (); CHECK (e.trackAt (1)->autoSnap.load()->lanes[0].pts.empty(), "one undo puts the lane back as it was before the drag");
          e.undo.redo ();
          auto proj = e.saveProject(); HostEngine g; g.prepareOffline (sr, bs, 0); juce::String er; g.loadProject (proj, [&] (juce::String s) { er = s; });
          const auto& L = g.trackAt (1)->autoSnap.load()->lanes; CHECK (er.isEmpty() && L.size() == 1 && L[0].kind == 0 && L[0].pts.size() == 2 && std::abs (L[0].pts[1].value + 24.0) < 1e-9, "lanes are saved with the project and come back with their points");
          CHECK (e.editAutoSetPoints (1, 999, pts ({ { 0.0, 0.0 } })) == false, "editing a lane that is not there is refused"); }
    }

    // ---- TAP TEMPO ----
    {
        TapTempo t;
        CHECK (t.tap (0.0) == 0.0 && t.tap (500.0) == 0.0, "the first two taps say nothing yet");
        const double b3 = t.tap (1000.0); CHECK (std::abs (b3 - 120.0) < 0.01, "three taps half a second apart are 120 bpm");
        TapTempo j; double bj = 0; const double jit[] = { 0, 510, 995, 1512, 2003, 2498, 3010 }; for (double x : jit) bj = j.tap (x);
        CHECK (std::abs (bj - 120.0) < 2.0, "slightly loose tapping still gives about 120");
        TapTempo o; for (double x : { 0.0, 500.0, 1000.0, 1500.0 }) o.tap (x); const double bo = o.tap (1620.0); (void) bo; const double after = o.tap (2000.0);
        CHECK (std::abs (after - 120.0) < 6.0, "a stray double tap is thrown out: still about 120");
        TapTempo g; g.tap (0.0); g.tap (500.0); g.tap (1000.0); CHECK (g.tap (5000.0) == 0.0 && g.count() == 1, "a gap of more than two seconds starts again");
        TapTempo f; double bf = 0; for (int i = 0; i < 10; ++i) bf = f.tap ((double) i * 300.0); CHECK (std::abs (bf - 200.0) < 0.01 && f.count() == 8, "only the last eight taps are used; 300 ms apart is 200 bpm");
        TapTempo fast; double bz = 0; for (int i = 0; i < 6; ++i) bz = fast.tap ((double) i * 100.0); CHECK (bz == 400.0, "a tempo above 400 is held to 400");
        TapTempo slow; double bs2 = 0; for (int i = 0; i < 3; ++i) bs2 = slow.tap ((double) i * 1900.0); CHECK (std::abs (bs2 - 60000.0 / 1900.0) < 0.01 && bs2 >= 20.0, "a slow tap just inside the gap limit is about 31.6 bpm");
        HostEngine e; e.prepareOffline (sr, bs, 0); e.undo.clock = [] { static double c = 0; c += 5000; return c; };
        e.tapper.reset(); e.tapper.tap (juce::Time::getMillisecondCounterHiRes()); juce::Thread::sleep (300); e.tapTempo (); juce::Thread::sleep (300); const double got = e.tapTempo ();
        CHECK (got > 150.0 && got < 230.0 && std::abs (e.transport.tempoEntries()[0].bpm - got) < 1e-9, "tapping the engine for real (300 ms apart, about 200 bpm) sets the song's tempo");
        e.undo.undo (); CHECK (std::abs (e.transport.tempoEntries()[0].bpm - 120.0) < 1e-9, "and undo puts the tempo back");
    }

    // ---- COLOURS ----
    {
        HostEngine e; e.prepareOffline (sr, bs, 0); e.undo.clock = [] { static double c = 0; c += 5000; return c; };
        CHECK (e.trackAt (0)->colour.load() == -1, "a track starts with no colour");
        e.editChainParam (0, "colour", 4); CHECK (e.trackAt (0)->colour.load() == 4, "a track is given a colour (its number in the palette)");
        e.editChainParam (0, "colour", 99); CHECK (e.trackAt (0)->colour.load() == 30, "a number past the palette is held to the last colour");
        e.editChainParam (0, "colour", -5); CHECK (e.trackAt (0)->colour.load() == -1, "and below none is none");
        e.editChainParam (0, "colour", 7); e.undo.undo (); CHECK (e.trackAt (0)->colour.load() == -1, "undo puts the colour back");
        CHECK (e.editChainParam (kReturnBase, "colour", 3) == false, "an FX return does not take a colour (yet)");
        const int a = e.editAddInstance (0, 0.0, 4.0, 0.0, [] { auto* c = new juce::DynamicObject(); c->setProperty ("lengthBeats", 4.0); return juce::var (c); }(), 0);
        auto prop = [] (int c) { auto* p = new juce::DynamicObject(); p->setProperty ("colour", c); return juce::var (p); };
        CHECK (e.trackAt (0)->arrSnap.load()->insts[0].colour == -1, "a clip starts by taking its track's colour (-1)");
        e.editSetInstance (a, prop (9)); CHECK (e.trackAt (0)->arrSnap.load()->insts[0].colour == 9, "a clip is given its own colour");
        const int d = e.editDuplicateInstance (a, 8.0, true, -1); CHECK (e.trackAt (0)->arrSnap.load()->insts[1].colour == 9 && d != 0, "a copy keeps it");
        e.editSetInstance (a, prop (-1)); CHECK (e.trackAt (0)->arrSnap.load()->insts[0].colour == -1 && e.trackAt (0)->arrSnap.load()->insts[1].colour == 9, "and the colour belongs to the clip, not the shared notes");
        e.editSetInstance (a, prop (200)); CHECK (e.trackAt (0)->arrSnap.load()->insts[0].colour == 30, "a clip colour past the palette is held to the last");
        e.editChainParam (0, "colour", 18); e.editSetInstance (a, prop (5));
        auto proj = e.saveProject(); HostEngine g; g.prepareOffline (sr, bs, 0); juce::String er; g.loadProject (proj, [&] (juce::String s) { er = s; });
        CHECK (g.trackAt (0)->colour.load() == 18 && g.trackAt (0)->arrSnap.load()->insts[0].colour == 5, "colours are saved with the project: track and clip (a folder from an older project is left out)");
        HostEngine h; h.prepareOffline (sr, bs, 0); h.undo.clock = [] { static double c = 0; c += 5000; return c; }; h.editAddTrack (); const int last = h.trackCount.load() - 1; h.editChainParam (last, "colour", 21);
        h.editRemoveTrack (); h.undo.undo (); CHECK (h.trackAt (last)->colour.load() == 21, "a removed track comes back with its colour");
    }

    // ---- GROOVE (swing) ----
    {
        CHECK (std::abs (grooveMap (0.0, 0.5, 1.0 / 6.0)) < 1e-12 && std::abs (grooveMap (0.5, 0.5, 1.0 / 6.0) - 2.0 / 3.0) < 1e-12 && std::abs (grooveMap (1.0, 0.5, 1.0 / 6.0) - 1.0) < 1e-12 && std::abs (grooveMap (1.5, 0.5, 1.0 / 6.0) - (1.0 + 2.0 / 3.0)) < 1e-12, "swing 1 on eighths: the off-beat eighths land a third of an eighth late, the on-beats do not move");
        bool mono = true; double prev = -1; for (double b = 0.0; b < 8.0; b += 0.01) { const double m = grooveMap (b, 0.5, 1.0 / 6.0); if (m < prev - 1e-12) mono = false; prev = m; }
        CHECK (mono, "the bend is monotonic: notes never swap places");
        CHECK (grooveMap (0.7, 0.5, 0.0) == 0.7 && grooveMap (-0.2, 0.5, 0.1) < 0.0 + 1e-9, "no swing leaves every position as it is");
        auto c = std::make_shared<Clip>(); c->lengthBeats = 2.0; c->id = 1; for (double b : { 0.0, 0.5, 1.0, 1.5 }) { NoteEv n; n.beat = b; n.len = 0.25; n.note = 60; c->notes.push_back (n); }
        ClipInst in; in.id = 1; in.start = 0.0; in.length = 4.0; in.content = c;
        auto ons = [&] (double d) { std::vector<double> v; walkInstance (in, 0.0, 2.0, false, [&] (const NoteEv&, double t) { v.push_back (t); }, [&] (const NoteEv&, double) {}, [&] (const NoteEv&) {}, d, 0.5); std::sort (v.begin(), v.end()); return v; };
        const auto straight = ons (0.0), swung = ons (1.0 / 6.0);
        CHECK (straight.size() == 4 && std::abs (straight[1] - 0.5) < 1e-12, "without swing the note clip plays as written");
        CHECK (swung.size() == 4 && std::abs (swung[0]) < 1e-12 && std::abs (swung[1] - 2.0 / 3.0) < 1e-12 && std::abs (swung[2] - 1.0) < 1e-12 && std::abs (swung[3] - 5.0 / 3.0) < 1e-12, "with full swing the playing notes land at 0, 2/3, 1 and 1 2/3");
        std::vector<double> offs; walkInstance (in, 0.0, 2.0, false, [&] (const NoteEv&, double) {}, [&] (const NoteEv&, double t) { offs.push_back (t); }, [&] (const NoteEv&) {}, 1.0 / 6.0, 0.5); std::sort (offs.begin(), offs.end());
        CHECK (offs.size() == 4 && offs[0] > swung[0] && offs[1] > swung[1] && offs[1] < swung[2], "each note still ends after it starts and before the next one starts");
        HostEngine e; e.prepareOffline (sr, bs, 0); e.transport.swing.store (0.6); e.transport.swingGrid.store (0.25);
        auto proj = e.saveProject(); HostEngine g; g.prepareOffline (sr, bs, 0); juce::String er; g.loadProject (proj, [&] (juce::String s) { er = s; });
        CHECK (er.isEmpty() && std::abs (g.transport.swing.load() - 0.6) < 1e-12 && g.transport.swingGrid.load() == 0.25, "swing and its grid are saved with the project");
    }

    // ---- EXPORT: the song mixed down ----
    {
        auto mkEngine = [&] (HostEngine& e) { e.prepareOffline (sr, bs, 0); e.trackAt (0)->source.store (2); e.trackAt (1)->source.store (0);
            auto* c = new juce::DynamicObject(); c->setProperty ("lengthBeats", 4.0); e.editAddInstance (0, 0.0, 8.0, 0.0, juce::var (c), 0); };
        { HostEngine e; e.prepareOffline (sr, bs, 0); CHECK (songEndBeat (e) == 0.0, "an empty song ends at beat 0"); }
        HostEngine e; mkEngine (e);
        CHECK (std::abs (songEndBeat (e) - 8.0) < 1e-9, "the song ends where its last clip ends");
        std::vector<float> got; const long long n = renderMixdown (e, 0.0, 2.0, 0.0, 256, [&] (const float* l, const float*, int c) { got.insert (got.end(), l, l + c); });
        CHECK (n == 48000 && got.size() == 48000, "two beats at 120 bpm are one second: 48000 samples");
        float pk = 0; for (size_t i = 2048; i < got.size(); ++i) pk = std::max (pk, std::abs (got[i]));
        CHECK (std::abs (pk - 0.25f) < 0.005f, "and the tone track is in it at its own level");
        std::vector<float> tailed; renderMixdown (e, 0.0, 1.0, 0.5, 256, [&] (const float* l, const float*, int c) { tailed.insert (tailed.end(), l, l + c); });
        CHECK (tailed.size() == 24000 + 24000, "a tail adds its seconds after the end");
        std::atomic<bool> stop { false }; std::vector<float> cut; renderMixdown (e, 0.0, 8.0, 0.0, 256, [&] (const float* l, const float*, int c) { cut.insert (cut.end(), l, l + c); if (cut.size() > 5000) stop.store (true); }, &stop);
        CHECK (cut.size() < 8000, "a cancelled render stops at once");
        std::vector<float> a2, b2; HostEngine f1; mkEngine (f1); renderMixdown (f1, 2.0, 4.0, 0.0, 256, [&] (const float* l, const float*, int c) { a2.insert (a2.end(), l, l + c); });
        CHECK (a2.size() == 48000, "a part of the song can be exported (beats 2 to 4)");
        // a private copy made from the saved project renders the same as the engine it was saved from
        HostEngine src; mkEngine (src); auto proj = src.saveProject(); HostEngine copy; copy.prepareOffline (sr, bs, 3); juce::String er; copy.loadProject (proj, [&] (juce::String s) { er = s; });
        std::vector<float> x, y; renderMixdown (src, 0.0, 4.0, 0.0, 512, [&] (const float* l, const float*, int c) { x.insert (x.end(), l, l + c); });
        renderMixdown (copy, 0.0, 4.0, 0.0, 512, [&] (const float* l, const float*, int c) { y.insert (y.end(), l, l + c); });
        CHECK (er.isEmpty() && x.size() == y.size() && std::memcmp (x.data(), y.data(), x.size() * sizeof (float)) == 0, "the private copy built from the saved project renders exactly what the original does, even on several threads");
        // the file
        auto tmp = juce::File::getSpecialLocation (juce::File::tempDirectory).getChildFile ("archie-export-test.wav"); tmp.deleteFile();
        { juce::WavAudioFormat wav; auto o = tmp.createOutputStream(); std::unique_ptr<juce::AudioFormatWriter> w (wav.createWriterFor (o.get(), sr, 2, 24, {}, 0)); o.release();
          renderMixdown (e, 0.0, 2.0, 0.0, 512, [&] (const float* l, const float* r, int c) { const float* ch[2] = { l, r }; w->writeFromFloatArrays (ch, 2, c); }); }
        { juce::AudioFormatManager fm; fm.registerBasicFormats(); std::unique_ptr<juce::AudioFormatReader> rd (fm.createReaderFor (tmp));
          CHECK (rd != nullptr && rd->lengthInSamples == 48000 && rd->bitsPerSample == 24 && rd->numChannels == 2, "a 24-bit stereo wav comes out with the right length");
          if (rd) { juce::AudioBuffer<float> b (2, 48000); rd->read (&b, 0, 48000, 0, true, true); float p2 = 0; for (int i = 4096; i < 48000; ++i) p2 = std::max (p2, std::abs (b.getSample (0, i))); CHECK (std::abs (p2 - 0.25f) < 0.005f, "and reads back as the tone it was"); } }
        tmp.deleteFile();
    }

    // ---- COPY, PASTE AND DUPLICATE CHANNELS ----
    {
        HostEngine e; e.prepareOffline (sr, bs, 0); e.undo.clock = [] { static double c = 0; c += 5000; return c; };
        auto* c = new juce::DynamicObject(); c->setProperty ("lengthBeats", 4.0); juce::Array<juce::var> a; auto* n = new juce::DynamicObject(); n->setProperty ("beat", 1.0); n->setProperty ("len", 0.5); n->setProperty ("note", 62); n->setProperty ("vel", 90); a.add (juce::var (n)); c->setProperty ("notes", a);
        e.editAddInstance (0, 0.0, 8.0, 0.0, juce::var (c), 0); e.trackAt (0)->name = "Lead"; e.editChainParam (0, "gainDb", -3.0); e.editChainParam (0, "pan", 0.25); e.editChainParam (0, "colour", 5); e.editSend (0, 1, -12.0f, true);
        const int nb = e.trackCount.load();
        const int ni = e.editDuplicateChannel (0, nullptr);
        CHECK (ni == nb && e.trackCount.load() == nb + 1, "a duplicate is added at the end");
        auto& d = *e.trackAt (ni);
        CHECK (d.name == "Lead copy" && std::abs (d.gainDb.load() + 3.0f) < 1e-4f && std::abs (d.pan.load() - 0.25f) < 1e-4f && d.colour.load() == 5, "with the name, level, pan and colour");
        CHECK (std::abs (d.sendDb[1].load() + 12.0f) < 1e-4f && d.sendPre[1].load(), "and its sends");
        const auto& di = d.arrSnap.load()->insts; const auto& oi = e.trackAt (0)->arrSnap.load()->insts;
        CHECK (di.size() == 1 && std::abs (di[0].length - 8.0) < 1e-9 && di[0].content->notes.size() == 1 && di[0].content != oi[0].content, "its clips are new copies, not shared with the original");
        e.undo.undo (); CHECK (e.trackCount.load() == nb, "undo takes the duplicate away");
        e.undo.redo (); CHECK (e.trackCount.load() == nb + 1 && e.trackAt (nb)->name == "Lead copy", "and redo brings it back");
        CHECK (e.copyChannel (0) && e.hasChannelClipboard(), "a channel can be copied");
        const int pi = e.editPasteChannel (e.channelClipboard, {}, nullptr); CHECK (pi == nb + 1 && e.trackAt (pi)->name == "Lead", "and pasted as a new track at the end");
    }

    // ---- INPUT ROUTING: an audio track listens to the input channel it is routed to ----
    {
        std::vector<float> c0 ((size_t) bs, 0.1f), c1 ((size_t) bs, 0.2f), c2 ((size_t) bs, 0.3f), c3 ((size_t) bs, 0.4f); const float* ins[4] = { c0.data(), c1.data(), c2.data(), c3.data() };
        auto levels = [&] (HostEngine& e, float& l, float& r)
        {
            std::vector<float> L ((size_t) bs), R ((size_t) bs); float* outs[2] = { L.data(), R.data() }; l = r = 0.0f;
            for (int b = 0; b < 12; ++b) { e.processOffline (ins, 4, outs, 2, bs); if (b >= 8) { l = L[(size_t) bs - 1]; r = R[(size_t) bs - 1]; } }
        };
        float l = 0, r = 0;
        { HostEngine e; e.prepareOffline (sr, bs, 0); e.trackAt (0)->source.store (1); e.editChainParam (0, "inChan", 0); e.editChainParam (0, "inStereo", false); levels (e, l, r);
          const float base = l; CHECK (base > 0.01f && std::abs (l - r) < 1e-4f, "a mono input is heard on both sides");
          e.editChainParam (0, "inChan", 2); levels (e, l, r); CHECK (std::abs (l / base - 3.0f) < 0.02f && std::abs (r / base - 3.0f) < 0.02f, "routed to input 3 it hears that channel (three times louder than input 1)");
          CHECK (e.trackAt (0)->inPeak.load() > 0.29f && e.trackAt (0)->inPeak.load() < 0.31f, "and its input level reads that channel");
          e.editChainParam (0, "inChan", 1); e.editChainParam (0, "inStereo", true); levels (e, l, r); CHECK (std::abs (l / base - 2.0f) < 0.02f && std::abs (r / base - 3.0f) < 0.02f, "stereo on 2+3: the left side is input 2, the right side is input 3");
          e.editChainParam (0, "inChan", 31); levels (e, l, r); CHECK (std::abs (l) < 1e-6f && std::abs (r) < 1e-6f, "an input that is not there is silence, not a crash");
          e.editChainParam (0, "inChan", 2); e.editChainParam (0, "inStereo", false); e.editChainParam (0, "monitor", false); levels (e, l, r); CHECK (std::abs (l) < 1e-6f && std::abs (r) < 1e-6f, "with Monitor off it is not heard");
          CHECK (e.trackAt (0)->inPeak.load() > 0.29f, "but its level is still read (so it can still be recorded)");
          auto proj = e.saveProject(); HostEngine g; g.prepareOffline (sr, bs, 0); g.loadProject (proj, [] (juce::String) {});
          CHECK (g.trackAt (0)->inChan.load() == 2 && ! g.trackAt (0)->inStereo.load() && ! g.trackAt (0)->monitor.load(), "the routing is saved with the song and comes back"); }
    }

    // ---- TRACK TYPES: an instrument track or an audio track, kept by save, load, copy and undo ----
    {
        HostEngine e; e.prepareOffline (sr, bs, 0);
        CHECK (e.trackAt (0)->kind.load() == 0, "a track is an instrument track at first");
        e.editChainParam (1, "kind", 1); e.editChainParam (1, "source", 1);
        CHECK (e.trackAt (1)->kind.load() == 1, "a track can be made an audio track");
        e.editChainParam (1, "kind", 7); CHECK (e.trackAt (1)->kind.load() == 1, "an unknown type is held to the two that exist");
        auto proj = e.saveProject(); HostEngine g; g.prepareOffline (sr, bs, 0); g.loadProject (proj, [] (juce::String) {});
        CHECK (g.trackAt (1)->kind.load() == 1 && g.trackAt (0)->kind.load() == 0, "the types survive save and load");
        const int ni = g.editDuplicateChannel (1, nullptr); CHECK (g.trackAt (ni)->kind.load() == 1, "and a copy of an audio track is an audio track");
        g.editChainParam (0, "kind", 1); g.undo.undo (); CHECK (g.trackAt (0)->kind.load() == 0, "undo puts the type back");
        HostEngine k; k.prepareOffline (sr, bs, 0); k.editChainParam (2, "kind", 1); k.loadProject (k.blankProject (kStartTracks), [] (juce::String) {});
        CHECK (k.trackAt (2)->kind.load() == 0, "a new song is all instrument tracks again");
    }

    // ---- NEW SONG ----
    {
        HostEngine e; e.prepareOffline (sr, bs, 0);
        auto* c = new juce::DynamicObject(); c->setProperty ("lengthBeats", 4.0); juce::Array<juce::var> a; auto* n = new juce::DynamicObject(); n->setProperty ("beat", 1.0); n->setProperty ("len", 0.5); n->setProperty ("note", 62); n->setProperty ("vel", 90); a.add (juce::var (n)); c->setProperty ("notes", a);
        e.editAddInstance (0, 0.0, 8.0, 0.0, juce::var (c), 0); e.trackAt (0)->name = "Lead"; e.editChainParam (0, "gainDb", -6.0); e.editChainParam (0, "colour", 4); e.addTrack(); e.addTrack(); e.editSend (1, 0, -6.0f, false);
        e.transport.setBpm (90.0); e.chainAt (kReturnBase)->name = "Space";
        bool called = false; juce::String err; e.loadProject (e.blankProject (kStartTracks), [&] (juce::String s) { called = true; err = s; });
        CHECK (called && err.isEmpty(), "a blank song loads");
        CHECK (e.trackCount.load() == kStartTracks, "back to the starting number of tracks");
        bool clean = true; for (int ti = 0; ti < e.trackCount.load(); ++ti) { auto& tr = *e.trackAt (ti); clean = clean && tr.primaryClip().notes.empty() && tr.arrSnap.load()->insts.size() <= 1 && std::abs (tr.gainDb.load()) < 1e-6f && tr.colour.load() == -1 && tr.name == "Track " + juce::String (ti + 1) && tr.slots.empty() && tr.sendDb[0].load() < -90.0f && tr.sendDb[1].load() < -90.0f; }
        CHECK (clean, "every track is empty, at 0 dB, uncoloured, named Track n, with no sends");
        CHECK (std::abs (e.transport.bpm.load() - 120.0) < 1e-9, "the tempo is 120 again");
        CHECK (e.chainAt (kReturnBase)->name == "FX 1", "the FX channels are named as at the start");
        CHECK (e.undo.depth() == 0, "and there is nothing to undo");
    }

    // ---- NEW SONG leaves nothing behind ----
    {
        auto blank = [&] (HostEngine& e, const juce::var& p) { bool done = false; e.loadProject (p, [&] (juce::String) { done = true; }); for (int i = 0; i < 100 && ! done; ++i) { e.service(); juce::Thread::sleep (10); } };
        auto text = [] (HostEngine& e) { juce::var v = e.saveProject(); if (auto* tr = v.getProperty ("tracks", {}).getArray()) for (auto& t : *tr) if (auto* o = t.getDynamicObject()) { o->removeProperty ("uid"); } return juce::JSON::toString (v); };
        HostEngine used; used.prepareOffline (sr, bs, 0); blank (used, used.blankProject()); used.addTrack(); used.addTrack();
        auto* c = new juce::DynamicObject(); c->setProperty ("lengthBeats", 4.0); juce::Array<juce::var> a; auto* n = new juce::DynamicObject(); n->setProperty ("beat", 1.0); n->setProperty ("len", 0.5); n->setProperty ("note", 62); n->setProperty ("vel", 90); a.add (juce::var (n)); c->setProperty ("notes", a);
        used.editAddInstance (0, 0.0, 8.0, 0.0, juce::var (c), 0);
        for (int ti = 0; ti < 2; ++ti)
        {
            auto& t = *used.trackAt (ti); t.eqHpOn.store (true); t.compOn.store (true); t.compThr.store (-30.0f); t.gateOn.store (true); t.width.store (0.2f); t.inGainDb.store (6.0f); t.patTrans.store (5);
            used.eqApply (ti, "b1on", 1.0); used.eqApply (ti, "b1gain", 9.0);
            auto set = std::make_shared<AutoSet>(); AutoLane l; l.id = 900 + ti; l.kind = 0; l.pts.push_back ({ 0.0, -12.0 }); set->lanes.push_back (l); used.putAuto (ti, set);
        }
        used.eqApply (kMasterChain, "b2on", 1.0); used.eqApply (kMasterChain, "b2gain", -7.0); used.eqApply (kMasterChain, "hpOn", 1.0);
        { juce::Array<juce::var> rm; auto* o = new juce::DynamicObject(); o->setProperty ("slot", 0); o->setProperty ("param", 1); o->setProperty ("cc", 20); rm.add (juce::var (o)); used.loadRemotesVar (0, juce::var (rm)); }
        used.transport.setTempoEntries ({ TempoEntry(), [] { TempoEntry e2; e2.beat = 16.0; e2.bpm = 150.0; return e2; }() });
        HostEngine fresh; fresh.prepareOffline (sr, bs, 0); blank (fresh, fresh.blankProject());
        blank (used, used.blankProject());
        CHECK (used.trackCount.load() == 0, "a new song has no tracks");
        CHECK (used.transport.tempoNow()->entries().size() == 1, "and no tempo changes from the last song");
        used.addTrack(); fresh.addTrack(); used.addTrack(); fresh.addTrack();
        CHECK (used.trackAt (0)->arrSnap.load()->insts.size() == fresh.trackAt (0)->arrSnap.load()->insts.size() && used.autoVar (0).size() == 0 && used.remoteCount (0) == 0, "a track added to it has no clips, automation or controller assignments from before");
        const auto u = text (used), w = text (fresh);
        CHECK (u == w, "and the whole song reads the same as one made in a program that has just started");
        if (u != w) { juce::File::getSpecialLocation (juce::File::tempDirectory).getChildFile ("newsong-used.json").replaceWithText (u); juce::File::getSpecialLocation (juce::File::tempDirectory).getChildFile ("newsong-fresh.json").replaceWithText (w); }
    }

    // ---- a track's place is used again: the new one is clean, and undo gives the old one back ----
    {
        HostEngine e; e.prepareOffline (sr, bs, 0); e.startEmpty(); e.addTrack(); e.addTrack(); e.addTrack(); e.undo.clear();
        auto map = [&] (int ti, int cc) { juce::Array<juce::var> rm; auto* o = new juce::DynamicObject(); o->setProperty ("slot", 0); o->setProperty ("param", 1); o->setProperty ("cc", cc); rm.add (juce::var (o)); e.loadRemotesVar (ti, juce::var (rm)); };
        auto ccOf = [&] (int ti) { const auto v = e.remotesVar (ti); return v.size() == 1 ? (int) v[0].getProperty ("cc", 0) : -v.size(); };
        auto set = [&] (int ti, float thr, int lane, int cc) { auto& t = *e.trackAt (ti); t.compOn.store (true); t.compThr.store (thr); t.width.store (0.5f); t.eqHpOn.store (true); e.eqApply (ti, "b1on", 1.0); e.eqApply (ti, "b1gain", thr / -6.0); auto s = std::make_shared<AutoSet>(); AutoLane l; l.id = lane; l.kind = 0; l.pts.push_back ({ 0.0, (double) thr }); s->lanes.push_back (l); e.putAuto (ti, s); map (ti, cc); };
        set (1, -30.0f, 701, 21); set (2, -24.0f, 702, 22);
        auto has = [&] (int ti, float thr, int lane, int cc) { auto& t = *e.trackAt (ti); const auto a = e.autoVar (ti); return t.compOn.load() && std::abs (t.compThr.load() - thr) < 1e-4f && std::abs (t.width.load() - 0.5f) < 1e-4f && t.eqHpOn.load() && std::abs (e.eqGet (ti, "b1gain") - thr / -6.0) < 1e-4 && a.size() == 1 && (int) a[0].getProperty ("id", 0) == lane && ccOf (ti) == cc; };
        auto clean = [&] (int ti) { auto& t = *e.trackAt (ti); return ! t.compOn.load() && std::abs (t.compThr.load() + 18.0f) < 1e-4f && std::abs (t.width.load() - 1.0f) < 1e-4f && ! t.eqHpOn.load() && e.eqGet (ti, "b1on") == 0.0 && std::abs (e.eqGet (ti, "b1gain")) < 1e-6 && e.autoVar (ti).size() == 0 && e.remoteCount (ti) == 0; };
        e.editRemoveTrack (1);
        CHECK (e.trackCount.load() == 2 && has (1, -24.0f, 702, 22), "removing a track leaves the one after it with its own settings, automation and controller assignment");
        e.editAddTrack();
        CHECK (e.trackCount.load() == 3 && clean (2), "a track added afterwards has nothing from the removed one");
        e.undo.undo(); e.undo.undo();
        CHECK (e.trackCount.load() == 3 && has (1, -30.0f, 701, 21) && has (2, -24.0f, 702, 22), "undo brings the removed track back with all it had, and the other keeps its own");
        e.undo.redo(); CHECK (e.trackCount.load() == 2 && has (1, -24.0f, 702, 22), "redo removes it again");
        e.undo.undo(); CHECK (e.trackCount.load() == 3 && has (1, -30.0f, 701, 21) && has (2, -24.0f, 702, 22), "and undo brings it back again");
        e.editRemoveTrack (2); e.editAddTrack(); CHECK (clean (2), "the last track too: its place is clean when used again");
    }

    // ---- automation and controller assignments follow their plug-in ----
    {
        HostEngine e; e.prepareOffline (sr, bs, 0); e.startEmpty(); e.addTrack(); e.addTrack();
        for (const char* n : { "A", "B", "C" }) e.addProcessor (0, std::make_unique<LatProc> (2, 2, 0), n);
        auto set = std::make_shared<AutoSet>(); for (int s = 0; s < 3; ++s) { AutoLane l; l.id = 800 + s; l.kind = 2; l.slot = s; l.param = 0; set->lanes.push_back (l); } { AutoLane v; v.id = 899; v.kind = 0; set->lanes.push_back (v); } e.putAuto (0, set);
        { juce::Array<juce::var> rm; for (int s = 0; s < 3; ++s) { auto* o = new juce::DynamicObject(); o->setProperty ("slot", s); o->setProperty ("param", 0); o->setProperty ("cc", 30 + s); rm.add (juce::var (o)); } e.loadRemotesVar (0, juce::var (rm)); }
        // each plug-in's lane (800 + n) and controller (30 + n) must be wherever that plug-in is
        auto follows = [&] (int chain) { auto* t = e.chainAt (chain); const auto a = e.autoVar (chain); const auto m = e.remotesVar (chain); for (int s = 0; s < (int) t->slots.size(); ++s) { const int who = t->slots[(size_t) s]->desc.name[0] - 'A'; bool lane = false, map = false; for (int i = 0; i < a.size(); ++i) if ((int) a[i].getProperty ("kind", 0) == 2 && (int) a[i].getProperty ("slot", -1) == s) lane = (int) a[i].getProperty ("id", 0) == 800 + who; for (int i = 0; i < m.size(); ++i) if ((int) m[i].getProperty ("slot", -1) == s) map = (int) m[i].getProperty ("cc", 0) == 30 + who; if (! lane || ! map) return false; } int plug = 0; for (int i = 0; i < a.size(); ++i) if ((int) a[i].getProperty ("kind", 0) == 2) ++plug; return plug == (int) t->slots.size() && m.size() == (int) t->slots.size(); };
        CHECK (follows (0), "each plug-in starts with its own lane and controller");
        e.moveSlot (0, 0, 1); CHECK (follows (0), "swapping two plug-ins swaps their lanes and controllers");
        e.moveSlotRaw (0, 2, 0, 0); CHECK (follows (0), "moving one to the top takes its lane and controller");
        e.moveSlotRaw (0, 1, 1, 0); CHECK (follows (0) && follows (1), "moving one to another track takes them to that track");
        e.removePlugin (0, 0); CHECK (follows (0), "removing one takes only its own away");
        bool volume = false; { const auto a = e.autoVar (0); for (int i = 0; i < a.size(); ++i) volume = volume || (int) a[i].getProperty ("id", 0) == 899; } CHECK (volume, "the track's level lane is untouched");
    }

    // ---- CRASH GUARD ----
    {
        auto dir = juce::File::getSpecialLocation (juce::File::tempDirectory).getChildFile ("archie-guard-test"); dir.deleteRecursively(); dir.createDirectory();
        HostEngine e; e.prepareOffline (sr, bs, 0); e.settingsDir = dir;                     // (a temp folder: never the person's real settings)
        CHECK (! e.crashedBefore ("VST3-Roland-909"), "nothing is flagged at first");
        e.loadingNow["VST3-Roland-909"] = "TR-909"; e.writeLoadingMarker();
        CHECK (e.loadingMarker().existsAsFile(), "a load in progress leaves a marker file");
        HostEngine f2; f2.prepareOffline (sr, bs, 0); f2.settingsDir = dir; f2.readLoadingMarker();          // the next start, after the program died
        juce::String nm; int times = 0;
        CHECK (f2.crashedBefore ("VST3-Roland-909", &nm, &times) && nm == "TR-909" && times == 1, "the plug-in is flagged, with its name, once");
        CHECK (! f2.loadingMarker().existsAsFile(), "and the marker is cleared so it only counts once");
        f2.loadingNow["VST3-Roland-909"] = "TR-909"; f2.writeLoadingMarker(); HostEngine f3; f3.prepareOffline (sr, bs, 0); f3.settingsDir = dir; f3.readLoadingMarker();
        f3.crashedBefore ("VST3-Roland-909", &nm, &times); CHECK (times == 2, "a second crash makes it two");
        f3.loadingNow.clear(); f3.writeLoadingMarker(); CHECK (! f3.loadingMarker().existsAsFile(), "a load that finishes removes the marker");
        f3.forgetCrash ("VST3-Roland-909"); HostEngine f4; f4.prepareOffline (sr, bs, 0); f4.settingsDir = dir; f4.readLoadingMarker(); CHECK (! f4.crashedBefore ("VST3-Roland-909"), "forgetting it stays forgotten");
        { // a crash in the plug-in's own thread, found by the file it was in
          HostEngine g; g.prepareOffline (sr, bs, 0); g.settingsDir = dir; g.crashedPlugins.clear();
          const juce::String modPath = "C:/Program Files/Roland Cloud/TR-909/TR-909(VST3 64bit).vst3"; const juce::String w16 = modPath;
          juce::MemoryBlock mb (w16.toUTF16().getAddress(), (size_t) w16.length() * 2); g.crashModuleFile().replaceWithData (mb.getData(), mb.getSize());
          g.readCrashModule();
          juce::PluginDescription d; d.name = "TR-909"; d.fileOrIdentifier = modPath; d.pluginFormatName = "VST3"; d.uniqueId = 7;
          juce::String nm; int times = 0; CHECK (g.crashedBeforeDesc (d, &nm, &times) && times == 1, "a fault found in the plug-in's file flags that plug-in");
          juce::PluginDescription other; other.name = "Other"; other.fileOrIdentifier = "C:/Program Files/Common Files/VST3/Other.vst3"; other.pluginFormatName = "VST3"; other.uniqueId = 8;
          CHECK (! g.crashedBeforeDesc (other), "and not a different plug-in");
          CHECK (! g.crashModuleFile().existsAsFile(), "the note is used up");
          const juce::String exe = "C:/Users/x/AppData/Local/AskArchie/StudioApp/ArchieStudio.exe"; juce::MemoryBlock mb2 (exe.toUTF16().getAddress(), (size_t) exe.length() * 2); g.crashModuleFile().replaceWithData (mb2.getData(), mb2.getSize());
          HostEngine h2; h2.prepareOffline (sr, bs, 0); h2.settingsDir = juce::File::getSpecialLocation (juce::File::tempDirectory).getChildFile ("archie-guard-test2"); h2.settingsDir.createDirectory(); h2.crashedPlugins.clear();
          h2.crashModuleFile().replaceWithData (mb2.getData(), mb2.getSize()); h2.readCrashModule(); CHECK (h2.crashedPlugins.empty(), "a fault in the program itself flags nothing"); h2.settingsDir.deleteRecursively(); }
        dir.deleteRecursively();
    }

    // ---- MIDI FILES ----
    {
        auto tmp = juce::File::getSpecialLocation (juce::File::tempDirectory).getChildFile ("archie-midi-test.mid"); tmp.deleteFile();
        auto notesClip = [&] (std::initializer_list<std::array<double, 4>> ns, double len) { auto* c = new juce::DynamicObject(); c->setProperty ("lengthBeats", len); juce::Array<juce::var> a; for (auto& n : ns) { auto* o = new juce::DynamicObject(); o->setProperty ("beat", n[0]); o->setProperty ("len", n[1]); o->setProperty ("note", (int) n[2]); o->setProperty ("vel", (int) n[3]); a.add (juce::var (o)); } c->setProperty ("notes", a); return juce::var (c); };
        HostEngine e; e.prepareOffline (sr, bs, 0); juce::String err;
        CHECK (e.exportMidiFile (tmp, err) == 0 && err.contains ("no note clips"), "a song with no notes cannot be exported, and says why");
        e.trackAt (0)->name = "Lead"; e.editAddInstance (0, 0.0, 8.0, 0.0, notesClip ({ { 0.0, 1.0, 60, 100 }, { 1.5, 0.5, 64, 80 }, { 3.0, 0.75, 67, 120 } }, 4.0), 0);
        e.trackAt (1)->name = "Bass"; e.editAddInstance (1, 4.0, 4.0, 0.0, notesClip ({ { 0.0, 2.0, 36, 90 } }, 4.0), 0);
        e.transport.setTempoEntries ({ TempoEntry { 0.0, 96.0, 3, 4 }, TempoEntry { 8.0, 140.0, 3, 4 } });
        CHECK (e.exportMidiFile (tmp, err) == 2 && tmp.existsAsFile(), "two tracks with notes are written");
        { juce::MidiFile mf; auto in = tmp.createInputStream(); mf.readFrom (*in); CHECK (mf.getNumTracks() == 3 && mf.getTimeFormat() == 480, "as a type-1 file: a tempo track and two note tracks, 480 ticks a quarter note"); }
        HostEngine g; g.prepareOffline (sr, bs, 0); g.undo.clock = [] { static double c = 0; c += 5000; return c; };
        const int made = g.importMidiFile (tmp, err);
        CHECK (made == 2 && g.trackCount.load() == 6, "importing makes a new track for each track with notes");
        { const int t0 = 4; const auto& n = g.trackAt (t0)->arrSnap.load()->insts[0].content->notes; CHECK (g.trackAt (t0)->name == "Lead" && n.size() == 3 + 3, "the first is called Lead and has its notes (the clip played twice over its 8 beats: 3 + 3)"); }
        { const auto& c = *g.trackAt (4)->arrSnap.load()->insts[0].content; std::vector<NoteEv> first; for (auto& x : c.notes) if (x.beat < 4.0) first.push_back (x); std::sort (first.begin(), first.end(), [] (const NoteEv& a, const NoteEv& b) { return a.beat < b.beat; });
          CHECK (first.size() == 3 && std::abs (first[1].beat - 1.5) < 0.003 && std::abs (first[1].len - 0.5) < 0.003 && first[1].note == 64 && first[1].vel == 80 && first[2].vel == 120, "notes keep their place, length, pitch and loudness (to within a 480th of a beat)"); }
        { const auto& i2 = g.trackAt (5)->arrSnap.load()->insts[0]; CHECK (g.trackAt (5)->name == "Bass" && std::abs (i2.content->notes[0].beat - 4.0) < 0.003, "the second track's notes start where its clip started (beat 4)"); }
        const auto te = g.transport.tempoEntries(); CHECK (te.size() == 2 && std::abs (te[0].bpm - 96.0) < 0.01 && te[0].tsNum == 3 && std::abs (te[1].beat - 8.0) < 1e-6 && std::abs (te[1].bpm - 140.0) < 0.01, "the tempo map and the 3/4 time signature came with it");
        g.undo.undo (); CHECK (g.trackCount.load() == 4 && std::abs (g.transport.tempoEntries()[0].bpm - 120.0) < 1e-9, "undo takes the new tracks and the tempo back");
        g.undo.redo (); CHECK (g.trackCount.load() == 6 && g.trackAt (4)->arrSnap.load()->insts.size() == 1, "and redo brings them again");
        // errors
        auto bad = juce::File::getSpecialLocation (juce::File::tempDirectory).getChildFile ("archie-not-midi.mid"); bad.replaceWithText ("this is not a MIDI file"); HostEngine h; h.prepareOffline (sr, bs, 0); juce::String er2;
        CHECK (h.importMidiFile (bad, er2) == 0 && er2.contains ("not a MIDI file"), "a file that is not MIDI is refused with the reason");
        CHECK (h.importMidiFile (juce::File ("Z:/missing.mid"), er2) == 0 && er2.isNotEmpty(), "a missing file is refused");
        { juce::MidiMessageSequence s; s.addEvent (juce::MidiMessage::textMetaEvent (3, "empty"), 0.0); juce::MidiFile mf; mf.setTicksPerQuarterNote (480); mf.addTrack (s); bad.deleteFile(); { auto o = bad.createOutputStream(); if (o != nullptr) mf.writeTo (*o, 1); }
          CHECK (h.importMidiFile (bad, er2) == 0 && er2.contains ("no notes"), "a MIDI file with no notes is refused"); }
        { juce::MidiMessageSequence s; for (int i = 0; i < 40; ++i) { s.addEvent (juce::MidiMessage::noteOn (1, 60, (juce::uint8) 90), 0.0); s.addEvent (juce::MidiMessage::noteOff (1, 60), 480.0); } juce::MidiFile mf; mf.setTicksPerQuarterNote (480); for (int i = 0; i < 40; ++i) mf.addTrack (s);
          bad.deleteFile(); { auto o = bad.createOutputStream(); if (o != nullptr) mf.writeTo (*o, 1); } CHECK (h.importMidiFile (bad, er2) == 0 && er2.contains ("room"), "a file with more tracks than there is room for is refused, saying how many fit"); }
        tmp.deleteFile(); bad.deleteFile();
    }

    // ---- PARALLEL RENDERING really is faster: 16 tracks, each with a heavy plug-in ----
    {
        auto timeIt = [&] (int workers)
        {
            HostEngine e; e.prepareOffline (sr, bs, workers);
            for (int i = 4; i < 16; ++i) e.addTrack();
            for (int i = 0; i < 16; ++i) { e.trackAt (i)->source.store (2); e.addProcessor (i, std::make_unique<HeavyProc>(), "heavy"); }
            run (e, 5, bs);                                              // warm up
            const auto t0 = juce::Time::getMillisecondCounterHiRes();
            run (e, 120, bs);
            return juce::Time::getMillisecondCounterHiRes() - t0;
        };
        const double one = timeIt (0), four = timeIt (3);
        const int cores = juce::SystemStats::getNumPhysicalCpus();
        std::printf ("  16 heavy tracks, 120 blocks: %.0f ms on one thread, %.0f ms with three workers (%.2fx), %d physical cores\n", one, four, one / four, cores);
        if (cores >= 4) CHECK (one / four > 1.5, "with several cores, three worker threads render heavy tracks at least 1.5x faster");
    }

    {   // the Archie device link: the whole conversation against a fake server
        struct P : juce::AudioProcessor, juce::AudioProcessorParameter::Listener
        {
            P() : juce::AudioProcessor (BusesProperties().withOutput ("o", juce::AudioChannelSet::stereo(), true)) { addParameter (a = new juce::AudioParameterFloat ({ "lvl0", 1 }, "Bass drum level", 0.0f, 1.0f, 0.8f)); addParameter (m = new juce::AudioParameterBool ({ "mute0", 1 }, "Bass drum mute", false)); for (int k = 0; k < 8; ++k) { auto* m8 = new juce::AudioParameterInt ({ "stepEdit" + juce::String (k), 1 }, "Step edit " + juce::String (k + 1), 0, 12288, 12288); box[(size_t) k] = m8; addParameter (m8); m8->addListener (this); } addParameter (bankP = new juce::AudioParameterInt ({ "bank", 1 }, "Bank", 0, 3, 0)); addParameter (patP = new juce::AudioParameterInt ({ "pattern", 1 }, "Pattern", 0, 3, 2)); }
            void parameterValueChanged (int idx, float v) override { if (idx < 2 || idx > 9) return; const juce::ScopedLock l (cs); seen.push_back ({ idx - 2, (int) std::lround (v * 12288.0f) }); } void parameterGestureChanged (int, bool) override {}
            juce::CriticalSection cs; std::vector<std::pair<int, int>> seen; std::array<juce::AudioParameterInt*, 8> box {}; juce::AudioParameterInt* bankP = nullptr; juce::AudioParameterInt* patP = nullptr;
            const juce::String getName() const override { return "P"; } void prepareToPlay (double, int) override {} void releaseResources() override {} void processBlock (juce::AudioBuffer<float>&, juce::MidiBuffer&) override {}
            juce::AudioProcessorEditor* createEditor() override { return nullptr; } bool hasEditor() const override { return false; } bool acceptsMidi() const override { return false; } bool producesMidi() const override { return false; } double getTailLengthSeconds() const override { return 0; }
            int getNumPrograms() override { return 1; } int getCurrentProgram() override { return 0; } void setCurrentProgram (int) override {} const juce::String getProgramName (int) override { return {}; } void changeProgramName (int, const juce::String&) override {}
            void getStateInformation (juce::MemoryBlock&) override {} void setStateInformation (const void*, int) override {}
            juce::AudioParameterFloat* a = nullptr; juce::AudioParameterBool* m = nullptr;
        };
        struct Fake { std::atomic<int> statusCalls { 0 }, states { 0 }, tokens { 0 }; std::atomic<bool> commandSent { false }, rejectToken { false }, opsSent { false }; juce::CriticalSection cs; juce::String lastParams, openedUrl, lastHost; };
        auto fake = std::make_shared<Fake>();
        archielink::Http http; http.post = [fake] (const juce::String& path, const juce::var& body, int& status, juce::var& reply)
        {
            status = 200; auto* o = new juce::DynamicObject(); reply = juce::var (o);
            if (path == "/api/connect/status") { if (++fake->statusCalls < 3) o->setProperty ("status", "not_found"); else o->setProperty ("status", "claimed"); return true; }
            if (path == "/api/connect/token") { ++fake->tokens; o->setProperty ("connectionToken", "tok-123"); return true; }
            if (path == "/api/connect/state") { if (fake->rejectToken.load()) { status = 401; return true; } ++fake->states; const juce::ScopedLock l (fake->cs); fake->lastParams = juce::JSON::toString (body.getProperty ("parameters", {}), true); fake->lastHost = juce::JSON::toString (body.getProperty ("hostContext", {}), true); o->setProperty ("ok", true); return true; }
            if (path == "/api/connect/command/next")
            {
                if (fake->commandSent.load() && ! fake->opsSent.exchange (true)) { juce::Array<juce::var> b; auto* ops = new juce::Array<juce::var>(); auto addOp = [&] (int pat, int drum, int step, int value) { auto* o2 = new juce::DynamicObject(); if (pat >= 0) o2->setProperty ("pattern", pat); o2->setProperty ("drum", drum); o2->setProperty ("step", step); o2->setProperty ("value", value); ops->add (juce::var (o2)); }; addOp (5, 2, 2, 1); addOp (-1, 3, 5, 2); addOp (0, 99, 0, 1); auto* eo = new juce::DynamicObject(); eo->setProperty ("deviceOps", juce::var (*ops)); delete ops; b.add (juce::var (eo)); o->setProperty ("parameterBatch", b); return true; }
                if (! fake->commandSent.exchange (true)) { juce::Array<juce::var> b; auto* e = new juce::DynamicObject(); e->setProperty ("parameterId", "lvl0"); e->setProperty ("normalizedValue", 0.25); b.add (juce::var (e)); auto* e2 = new juce::DynamicObject(); e2->setProperty ("parameterId", "mute0"); e2->setProperty ("normalizedValue", 1.0); b.add (juce::var (e2)); auto* e3 = new juce::DynamicObject(); e3->setProperty ("parameterId", "nope"); e3->setProperty ("normalizedValue", 1.0); b.add (juce::var (e3)); o->setProperty ("parameterBatch", b); }
                return true;
            }
            return false;
        };
        auto tokenFile = juce::File::getSpecialLocation (juce::File::tempDirectory).getChildFile ("archie-link-test-" + juce::String (juce::Time::currentTimeMillis()) + ".json");
        auto waitFor = [] (std::function<bool()> f, int ms) { const auto t = juce::Time::getMillisecondCounter(); while (! f() && juce::Time::getMillisecondCounter() - t < (juce::uint32) ms) juce::Thread::sleep (5); return f(); };
        {
            P proc; archielink::DeviceLink link (proc, "archie-drums", "daw-1", tokenFile, http, [fake] (const juce::String& u) { const juce::ScopedLock l (fake->cs); fake->openedUrl = u; }, 20, 20);
            link.setHostInfo ([] (juce::DynamicObject& h) { h.setProperty ("playing", true); h.setProperty ("bpm", 133.0); h.setProperty ("ppqPosition", 7.5); });
            link.setExtras ([] (juce::AudioProcessor&) -> juce::var { auto* d = new juce::DynamicObject(); d->setProperty ("pattern", 3); d->setProperty ("steps", "0120"); auto* o = new juce::DynamicObject(); o->setProperty ("drums", juce::var (d)); return juce::var (o); }); link.refreshExtrasNow();
            CHECK (! link.hasSavedToken() && link.state() == archielink::DeviceLink::Off, "a new link starts off with no saved connection");
            link.start();
            CHECK (waitFor ([&] { return link.state() == archielink::DeviceLink::Connected; }, 5000), "it pairs and reaches Connected once the browser claims it");
            { const juce::ScopedLock l (fake->cs); CHECK (fake->openedUrl.startsWith ("/connect-device?kind=archie-drums&h=") && fake->openedUrl.contains ("&i=daw-1"), "the browser is sent to the connect page with the device kind and the hash"); CHECK (fake->openedUrl.length() < 160, "and the page address is short (a hash, not the secret)"); }
            CHECK (fake->tokens.load() == 1 && link.hasSavedToken(), "the connection token is traded once and saved");
            CHECK (waitFor ([&] { return link.pushes() >= 3; }, 3000), "it keeps pushing its state");
            { const juce::ScopedLock l (fake->cs); CHECK (fake->lastParams.contains ("lvl0") && fake->lastParams.contains ("Bass drum mute"), "the pushed state lists the parameters by id and name"); }
            { const juce::ScopedLock l (fake->cs); CHECK (fake->lastHost.contains ("\"drums\"") && fake->lastHost.contains ("0120") && fake->lastHost.contains ("archie-drums"), "and the extra live state (the pattern on view) travels with it"); CHECK (fake->lastHost.contains ("133") && fake->lastHost.contains ("ppqPosition") && fake->lastHost.contains ("playing"), "as does the transport, for the playhead"); }
            CHECK (waitFor ([&] { return link.applied() >= 2; }, 3000), "the two known parameter commands are applied (the unknown one is ignored)");
            CHECK (std::abs (proc.a->get() - 0.25f) < 1e-3f && proc.m->get(), "and the parameters really changed");
            CHECK (waitFor ([&] { const juce::ScopedLock l (proc.cs); return proc.seen.size() >= 4; }, 3000), "step edits from the website go to the plug-in's mailboxes");
            { const juce::ScopedLock l (proc.cs); const int e0 = ((5 * 16 + 2) * 16 + 2) * 3 + 1, e1 = ((2 * 16 + 3) * 16 + 5) * 3 + 2;   // the second has no pattern: it means the pattern on view (bank 0, pattern 3 is slot 2 here)
              const std::vector<std::pair<int, int>> want { { 0, e0 }, { 1, e1 }, { 0, 12288 }, { 1, 12288 } };
              const bool sameEdits = proc.seen.size() == 4 && std::equal (want.begin(), want.end(), proc.seen.begin()); CHECK (sameEdits, "two edits go in together (one with its pattern, one for the pattern on view), then every mailbox is set back to idle, and an invalid edit is dropped"); }
            link.stop(); CHECK (link.state() == archielink::DeviceLink::Off, "stopping turns it off");
        }
        {   // a saved token is used straight away: no browser, no pairing
            fake->statusCalls = 0; fake->tokens = 0; { const juce::ScopedLock l (fake->cs); fake->openedUrl = {}; }
            P proc; archielink::DeviceLink link (proc, "archie-drums", "daw-1", tokenFile, http, [fake] (const juce::String& u) { const juce::ScopedLock l (fake->cs); fake->openedUrl = u; }, 20, 20);
            CHECK (link.hasSavedToken(), "the saved connection is still there"); link.start();
            CHECK (waitFor ([&] { return link.state() == archielink::DeviceLink::Connected && link.pushes() >= 1; }, 3000), "it reconnects with no browser step");
            { const juce::ScopedLock l (fake->cs); CHECK (fake->openedUrl.isEmpty() && fake->statusCalls.load() == 0 && fake->tokens.load() == 0, "no pairing was needed"); }
            fake->rejectToken = true;
            CHECK (waitFor ([&] { return link.state() == archielink::DeviceLink::Failed; }, 3000), "when the site stops accepting the connection it says so");
            CHECK (link.message().contains ("connect again") && ! link.hasSavedToken(), "with a plain message, and the dead connection is forgotten");
        }
        {   // the site cannot be reached
            archielink::Http dead; dead.post = [] (const juce::String&, const juce::var&, int& s, juce::var&) { s = 0; return false; };
            tokenFile.deleteFile(); P proc; std::atomic<bool> opened { false }; archielink::DeviceLink link (proc, "archie-drums", "daw-1", tokenFile, dead, [&] (const juce::String&) { opened = true; }, 20, 20);
            link.start(); juce::Thread::sleep (150); CHECK (link.state() == archielink::DeviceLink::Waiting, "with no connection to the site it keeps waiting rather than failing at once");
            link.stop(); CHECK (opened.load(), "(the browser had been opened)");
        }
        tokenFile.deleteFile();
    }

    // HOT CUES: saved with the song, and a controller pad learned for a cue fires it instead of playing a note
    {
        HostEngine e; e.prepareOffline (48000.0, 512, 0); e.cuePadFile().deleteFile(); for (int i = 0; i < archie::Transport::kCues; ++i) e.clearCuePad (i);
        e.transport.setLoop (true, 0.0, 16.0); e.transport.setCue (1, 32.0, 40.0); e.transport.setCue (2, 64.0, 80.0); e.transport.triggerCue (2);
        auto proj = e.saveProject(); HostEngine g; g.prepareOffline (48000.0, 512, 0); g.loadProject (proj, [&] (juce::String) {});
        CHECK (g.transport.activeCue.load() == 2 && g.transport.loopStart.load() == 64.0 && g.transport.cues[0].start.load() == 0.0 && g.transport.cues[0].end.load() == 16.0 && g.transport.cues[1].set.load() && g.transport.cues[1].start.load() == 32.0 && g.transport.cues[1].end.load() == 40.0, "the three cues and the playing one are saved with the song");
        g.loadProject (HostEngine().saveProject(), [&] (juce::String) {});
        CHECK (g.transport.activeCue.load() == 0 && ! g.transport.cues[1].set.load() && ! g.transport.cues[2].set.load(), "a new song starts with cue 1 only");
        e.cueLearn.store (0); CHECK (e.cueConsume (juce::MidiMessage::noteOn (10, 36, (juce::uint8) 100)) && e.cueLearn.load() == -1, "the next controller pad is learned for the cue");
        CHECK (e.transport.activeCue.load() == 2, "learning it does not fire it");
        CHECK (e.cueConsume (juce::MidiMessage::noteOn (10, 36, (juce::uint8) 90)) && e.transport.activeCue.load() == 0 && e.transport.loopStart.load() == 0.0, "pressing that pad jumps to the cue");
        CHECK (e.cueConsume (juce::MidiMessage::noteOff (10, 36)) && ! e.cueConsume (juce::MidiMessage::noteOn (10, 37, (juce::uint8) 90)) && ! e.cueConsume (juce::MidiMessage::noteOn (1, 36, (juce::uint8) 90)), "its release is taken too, while other notes and channels play as before");
        e.cueLearn.store (1); e.cueConsume (juce::MidiMessage::controllerEvent (1, 20, 127)); e.cueConsume (juce::MidiMessage::controllerEvent (1, 20, 0)); CHECK (e.transport.activeCue.load() == 0, "a controller switch can be learned too");
        e.cueConsume (juce::MidiMessage::controllerEvent (1, 20, 127)); CHECK (e.transport.activeCue.load() == 1, "and fires on its press");
        e.service(); { HostEngine h; CHECK (h.cuePadText (0) == "Note 36, channel 10" && h.cuePadText (1) == "CC 20, channel 1", "learned pads are remembered on this computer"); }
        e.cueLearn.store (1); e.cueConsume (juce::MidiMessage::noteOn (10, 36, (juce::uint8) 100)); CHECK (e.cuePadText (0).isEmpty() && e.cuePadText (1) == "Note 36, channel 10", "one control serves one cue");
        for (int i = 0; i < archie::Transport::kCues; ++i) e.clearCuePad (i); e.cuePadFile().deleteFile();
    }

    // PLAY IN BACKGROUND: on by default; off, the card is let go after a second away (never while recording) and taken back on return
    {
        using B = HostEngine::Background;
        CHECK (HostEngine::backgroundAction (true, false, false, false, 60.0) == B::nothing, "left on, the sound card is kept however long another program is in front");
        CHECK (HostEngine::backgroundAction (false, false, false, false, 0.4) == B::nothing && HostEngine::backgroundAction (false, false, false, false, 1.0) == B::release, "off: let go after a second away, not for a passing pop-up");
        CHECK (HostEngine::backgroundAction (false, false, false, true, 60.0) == B::nothing, "never while recording");
        CHECK (HostEngine::backgroundAction (false, false, true, false, 60.0) == B::nothing && HostEngine::backgroundAction (false, true, true, false, 0.0) == B::restore, "it stays let go while away and comes back with the program");
        CHECK (HostEngine::backgroundAction (true, false, true, false, 60.0) == B::restore, "ticking it again takes the card back");
        HostEngine e; e.audioOptionsFile().deleteFile(); e.loadAudioOptions(); CHECK (e.playInBackground.load(), "on unless the person unticks it");
        auto* o = new juce::DynamicObject(); o->setProperty ("playInBackground", false); CHECK (e.setAudioOptions (juce::var (o)) && ! e.playInBackground.load(), "it can be switched off");
        HostEngine g; g.loadAudioOptions(); CHECK (! g.playInBackground.load(), "and is remembered"); e.audioOptionsFile().deleteFile();
    }

    // RENDER WITH THE SOUND CARD'S BUFFER: off = pieces of 512; on = the card's own piece size; remembered
    {
        HostEngine e; e.audioOptionsFile().deleteFile(); e.loadAudioOptions(); e.prepareOffline (48000.0, 128, 0);
        CHECK (! e.renderWithCardBuffer.load() && e.renderPieceSize() == 512, "exports use pieces of 512 unless asked otherwise");
        auto* o = new juce::DynamicObject(); o->setProperty ("renderWithCardBuffer", true); CHECK (e.setAudioOptions (juce::var (o)) && e.renderPieceSize() == 128, "ticked, they use the sound card's buffer size");
        HostEngine g; g.loadAudioOptions(); CHECK (g.renderWithCardBuffer.load(), "and the choice is remembered"); e.audioOptionsFile().deleteFile();
    }

    // MIDI CONTROLLER LANES: pitch bend, mod wheel, sustain and expression in note clips
    {
        struct CtlProbe : public juce::AudioProcessor
        {
            CtlProbe() : juce::AudioProcessor (BusesProperties().withOutput ("out", juce::AudioChannelSet::stereo(), true)) {}
            const juce::String getName() const override { return "CtlProbe"; }
            void prepareToPlay (double, int) override {}
            void releaseResources() override {}
            void processBlock (juce::AudioBuffer<float>& b, juce::MidiBuffer& m) override
            {
                b.clear(); double ppq = 0.0; if (auto* ph = getPlayHead()) if (auto pos = ph->getPosition()) if (auto q = pos->getPpqPosition()) ppq = *q;
                for (const auto meta : m) { CtlEv e; if (ctlOfMidi (meta.getMessage(), e)) { e.beat = ppq + (double) meta.samplePosition / 48000.0 * 2.0; got.push_back (e); } }
            }
            double getTailLengthSeconds() const override { return 0; }
            bool acceptsMidi() const override { return true; }
            bool producesMidi() const override { return false; }
            juce::AudioProcessorEditor* createEditor() override { return nullptr; }
            bool hasEditor() const override { return false; }
            int getNumPrograms() override { return 1; }
            int getCurrentProgram() override { return 0; }
            void setCurrentProgram (int) override {}
            const juce::String getProgramName (int) override { return {}; }
            void changeProgramName (int, const juce::String&) override {}
            void getStateInformation (juce::MemoryBlock&) override {}
            void setStateInformation (const void*, int) override {}
            std::vector<CtlEv> got;
            std::vector<CtlEv> of (int type) const { std::vector<CtlEv> v; for (const auto& e : got) if (e.type == type) v.push_back (e); return v; }
        };
        auto ctl = [] (std::initializer_list<std::array<double, 3>> list) { juce::Array<juce::var> a; for (auto& x : list) { juce::Array<juce::var> e; e.add (x[0]); e.add ((int) x[1]); e.add ((int) x[2]); e.add (1); a.add (juce::var (e)); } return juce::var (a); };
        auto clipOf = [&] (juce::var ctls) { auto* c = new juce::DynamicObject(); c->setProperty ("lengthBeats", 4.0); c->setProperty ("notes", juce::var (juce::Array<juce::var>())); if (! ctls.isVoid()) c->setProperty ("ctls", ctls); return juce::var (c); };
        const double beatsPerBlock = bs / 48000.0 * 2.0;
        auto playTo = [&] (HostEngine& e, double beat) { for (int i = 0; i < 20000 && e.transport.posBeats.load() < beat; ++i) run (e, 1, bs); };
        auto nearBeat = [&] (double a, double b) { return std::abs (a - b) <= beatsPerBlock + 0.01; };
        {
            HostEngine e; e.prepareOffline (sr, bs, 0); e.transport.setBpm (120.0);
            auto pr = std::make_unique<CtlProbe>(); auto* p = pr.get(); p->setPlayHead (e.hostPlayHead()); e.addProcessor (0, std::move (pr), "ctl probe");
            const int id = e.editAddInstance (0, 0.0, 8.0, 0.0, clipOf (ctl ({ { 1.0, 128, 12000 }, { 2.0, 128, 8192 }, { 0.0, 1, 40 }, { 3.0, 64, 127 }, { 0.5, 11, 90 }, { 1.0, 7, 5 } })), 0);
            CHECK (id != 0 && e.trackAt (0)->arrSnap.load()->insts[0].content->ctls.size() == 5, "a clip keeps pitch bend, mod wheel, sustain and expression, and nothing else");
            e.transport.play(); playTo (e, 3.5);
            const auto bend = p->of (128);
            CHECK (bend.size() == 2 && bend[0].value == 12000 && nearBeat (bend[0].beat, 1.0) && bend[1].value == 8192 && nearBeat (bend[1].beat, 2.0), "pitch bend plays at its places");
            CHECK (p->of (1).size() == 1 && p->of (1)[0].value == 40 && p->of (11).size() == 1 && p->of (64).size() == 1 && p->of (64)[0].value == 127, "mod wheel, expression and sustain play too");
            playTo (e, 4.4);
            { const auto s = p->of (64), x = p->of (11); CHECK (s.size() == 2 && s[1].value == 0 && nearBeat (s[1].beat, 4.0), "each time round the clip starts from rest, so a pedal held to its end is let go"); CHECK (x.size() >= 2 && x[1].value == 127 && nearBeat (x[1].beat, 4.0), "expression goes back to full"); }
            playTo (e, 7.5); CHECK (p->of (64).size() == 3 && p->of (64).back().value == 127, "and the second time round plays them again");
            playTo (e, 8.4); CHECK (p->of (64).back().value == 0 && p->of (1).back().value == 0, "where the clip ends everything it moved is back at rest");
        }
        {
            HostEngine e; e.prepareOffline (sr, bs, 0); e.transport.setBpm (120.0);
            auto pr = std::make_unique<CtlProbe>(); auto* p = pr.get(); p->setPlayHead (e.hostPlayHead()); e.addProcessor (0, std::move (pr), "ctl probe");
            e.editAddInstance (0, 0.0, 16.0, 0.0, clipOf (ctl ({ { 0.25, 128, 3000 }, { 0.5, 64, 127 } })), 0);
            e.transport.locate (2.0); e.transport.play(); run (e, 4, bs);
            CHECK (p->of (128).size() == 1 && p->of (128)[0].value == 3000 && p->of (64).size() == 1, "starting part-way through brings the instrument to the clip's values");
            e.transport.stop(); run (e, 4, bs);
            CHECK (p->of (128).back().value == 8192 && p->of (64).back().value == 0, "stopping lets the pedal go and centres the bend");
            const size_t n = p->got.size(); run (e, 20, bs); CHECK (p->got.size() == n, "and while stopped the clips send nothing, so the player's own wheels are not overridden");
        }
        {   // saving, editing the notes, splitting, joining and MIDI files
            HostEngine e; e.prepareOffline (sr, bs, 0);
            const int id = e.editAddInstance (0, 0.0, 4.0, 0.0, clipOf (ctl ({ { 1.0, 1, 64 }, { 3.0, 128, 100 } })), 0);
            const int cid = e.trackAt (0)->arrSnap.load()->insts[0].content->id;
            { auto* c = new juce::DynamicObject(); c->setProperty ("lengthBeats", 4.0); juce::Array<juce::var> ns; auto* n = new juce::DynamicObject(); n->setProperty ("beat", 0.0); n->setProperty ("len", 1.0); n->setProperty ("note", 60); ns.add (juce::var (n)); c->setProperty ("notes", ns); e.editContent (cid, juce::var (c)); }
            CHECK (e.trackAt (0)->arrSnap.load()->insts[0].content->ctls.size() == 2 && e.trackAt (0)->arrSnap.load()->insts[0].content->notes.size() == 1, "editing the notes keeps the controller data");
            auto song = e.saveProject(); HostEngine g; g.prepareOffline (sr, bs, 0); g.loadProject (song, [] (juce::String) {});
            CHECK (g.trackAt (0)->arrSnap.load()->insts[0].content->ctls.size() == 2 && g.trackAt (0)->arrSnap.load()->insts[0].content->ctls[1].value == 100, "it is saved with the song");
            const int second = e.editSplitInstance (id, 2.0); juce::String err; const int joined = e.editJoinInstance (id, err);
            CHECK (second != 0 && joined != 0 && e.trackAt (0)->arrSnap.load()->insts[0].content->ctls.size() >= 2, "split and join keep it");
            auto mid = juce::File::getSpecialLocation (juce::File::tempDirectory).getChildFile ("archie-ctl.mid"); CHECK (e.exportMidiFile (mid, err) == 1, "it goes into a MIDI file");
            HostEngine m; m.prepareOffline (sr, bs, 0); m.importMidiFile (mid, err);
            bool mod = false, bd = false; if (m.trackCount.load() > 0) for (const auto& c : m.trackAt (m.trackCount.load() - 1)->arrSnap.load()->insts[0].content->ctls) { mod = mod || (c.type == 1 && c.value == 64); bd = bd || (c.type == 128 && c.value == 100); }
            CHECK (mod && bd, "and comes back from one"); mid.deleteFile();
        }
        {   // overdub: a new movement replaces the old one over the stretch it was played
            Clip c; c.lengthBeats = 8.0; c.ctls = { { 0.5, 1, 10, 1 }, { 2.0, 1, 20, 1 }, { 3.0, 1, 30, 1 }, { 6.0, 1, 60, 1 }, { 2.5, 11, 50, 1 } };
            HostEngine::mergeCtls (c, { { 1.5, 1, 100, 1 }, { 3.5, 1, 110, 1 } });
            std::vector<int> mods; for (const auto& e : c.ctls) if (e.type == 1) mods.push_back (e.value);
            CHECK ((mods == std::vector<int> { 10, 100, 110, 60 }) && c.ctls.size() == 5, "old values inside the stretch are replaced; the rest, and other controllers, stay");
            HostEngine::mergeCtls (c, { { 7.0, 1, 1, 1 }, { 1.0, 1, 2, 1 } }); mods.clear(); for (const auto& e : c.ctls) if (e.type == 1) mods.push_back (e.value);
            CHECK ((mods == std::vector<int> { 2, 100, 110, 60, 1 }), "also when the playing went round the end of the clip");
        }
    }

    // SIDE-CHAIN: the compressor, gate and master compressor can listen to another track
    {
        auto peakAfter = [&] (HostEngine& e) { auto o = run (e, 120, bs); float pk = 0; for (size_t i = o.size() / 2; i < o.size(); ++i) pk = std::max (pk, std::abs (o[i])); return pk; };
        auto set = [] (HostEngine& e, int track, const char* key, juce::var v) { return e.editChainParam (track, key, v); };
        HostEngine e; e.prepareOffline (sr, bs, 0);
        e.trackAt (0)->source.store (2);                                  // the listener: a steady tone
        const int keyUid = e.trackAt (2)->uid.load();                     // the key is a LATER track, so it has to be rendered first
        CHECK (keyUid > 0 && keyUid != e.trackAt (0)->uid.load(), "every track has a number of its own");
        set (e, 0, "compOn", true); set (e, 0, "compThr", -30.0); set (e, 0, "compRatio", 20.0);
        const float own = peakAfter (e); CHECK (own < 0.12f, "on its own sound the compressor turns the tone down");
        CHECK (set (e, 0, "compKey", keyUid), "a track can be chosen as the key");
        run (e, 800, bs); const float quiet = peakAfter (e); CHECK (std::abs (quiet - 0.25f) < 0.005f, "with a silent key nothing is turned down");
        e.trackAt (2)->source.store (2); set (e, 2, "mute", true);
        const float ducked = peakAfter (e); CHECK (ducked < 0.12f, "a loud key ducks the track, even while the key track is muted");
        CHECK (e.planNow()->levels.size() == 2, "the key is rendered before the track that listens to it");
        e.moveTrackTo (2, 3); CHECK (peakAfter (e) < 0.12f, "the key follows its track when tracks move");
        auto song = e.saveProject(); HostEngine g; g.prepareOffline (sr, bs, 0); g.loadProject (song, [] (juce::String) {});
        CHECK (g.trackAt (0)->compKey.load() == keyUid && g.trackAt (3)->uid.load() == keyUid, "and is saved with the song");
        g.editAddTrack(); CHECK (g.trackAt (g.trackCount.load() - 1)->uid.load() > keyUid, "a track added afterwards gets a new number");
        set (e, 3, "compKey", e.trackAt (0)->uid.load());                 // each keys the other
        const auto* plan = e.planNow(); CHECK (! (plan->tracks[0].compKey >= 0 && plan->tracks[3].compKey >= 0), "two tracks keying each other: one of the two is left out, never both");
        peakAfter (e); set (e, 3, "compKey", 0);
        // the gate keyed from another track
        set (e, 0, "compKey", 0); set (e, 0, "compOn", false); set (e, 0, "gateOn", true); set (e, 0, "gateThr", -30.0); set (e, 0, "gateRange", 80.0); set (e, 0, "gateKey", keyUid);
        e.trackAt (3)->source.store (0); const float shut = peakAfter (e); e.trackAt (3)->source.store (2); for (int i = 0; i < 3; ++i) run (e, 60, bs); const float open = peakAfter (e);
        CHECK (shut < 0.01f && std::abs (open - 0.25f) < 0.01f, "the gate stays shut until its key is loud");
        set (e, 0, "gateOn", false); set (e, 0, "gateKey", 0);
        // the master compressor keyed from a muted track
        e.editMasterCompParam ("mcOn", true); e.editMasterCompParam ("mcThr", -30.0); e.editMasterCompParam ("mcRatio", 20.0); e.editMasterCompParam ("mcKey", keyUid);
        e.trackAt (3)->source.store (0); for (int i = 0; i < 3; ++i) run (e, 120, bs); const float mQuiet = peakAfter (e); e.trackAt (3)->source.store (2); const float mDuck = peakAfter (e);
        CHECK (std::abs (mQuiet - 0.25f) < 0.01f && mDuck < 0.12f, "the master compressor listens to its key track instead of the mix");
        e.editRemoveTrack (3); CHECK (e.planNow()->masterKey < 0 && peakAfter (e) >= 0.0f, "when the key track is removed the compressor listens to its own sound again");
    }
    // SIDE-CHAIN INPUTS (2026-11-28): tempo pulses, the trigger filter, FX returns, Duck below, and plug-ins
    {
        auto set = [] (HostEngine& e, int track, const char* key, juce::var v) { return e.editChainParam (track, key, v); };
        auto peakIn = [] (const std::vector<float>& o, int a, int b) { float pk = 0; for (int i = a; i < b && i < (int) o.size(); ++i) pk = std::max (pk, std::abs (o[(size_t) i])); return pk; };
        auto listener = [&] (HostEngine& e) { e.prepareOffline (sr, bs, 0); e.transport.setBpm (120.0); e.trackAt (0)->source.store (2); set (e, 0, "compOn", true); set (e, 0, "compThr", -30.0); set (e, 0, "compRatio", 20.0); set (e, 0, "compRel", 0.05); };
        const int beat = 24000;
        {
            HostEngine e; listener (e); CHECK (set (e, 0, "compKey", -1) && e.trackAt (0)->compKey.load() == -1, "a 4/4 pulse can be chosen as the trigger");
            auto stopped = run (e, 400, bs); CHECK (std::abs (peakIn (stopped, 60000, 100000) - 0.25f) < 0.005f, "with the song stopped a pulse does nothing");
            e.transport.locate (0.0); e.transport.play(); auto o = run (e, 400, bs);
            bool onEvery = true, backBetween = true; for (int k = 1; k < 4; ++k) { if (peakIn (o, k * beat + 480, k * beat + 1900) > 0.12f) onEvery = false; if (peakIn (o, k * beat + 20000, k * beat + 23800) < 0.2f) backBetween = false; }
            CHECK (onEvery, "4/4 ducks on every beat"); CHECK (backBetween, "and the sound comes back before the next");
            HostEngine g; listener (g); set (g, 0, "compKey", -2); g.transport.locate (0.0); g.transport.play(); auto two = run (g, 500, bs);
            CHECK (peakIn (two, 2 * beat + 480, 2 * beat + 1900) < 0.12f && peakIn (two, beat + 480, beat + 1900) > 0.2f && peakIn (two, 4 * beat + 480, 4 * beat + 1900) < 0.12f, "2/4 ducks on beats 1 and 3 only");
            double since = 0; CHECK (archieSC::pulseAt (-3, 0.0, 120.0, since) >= 0.0f && std::abs (archieSC::pulseAt (-3, 1.5 + 0.02, 120.0, since) - 1.0f) < 1.0e-4f && std::abs (archieSC::pulseAt (-3, 0.52, 120.0, since) - 0.55f) < 1.0e-4f, "6/8 pulses every eighth, strong on the first and fourth");
            CHECK (archieSC::pulseAt (-5, 0.27, 120.0, since) > 0.9f && archieSC::pulseAt (-4, 0.27, 120.0, since) < 0.01f, "1/16 pulses twice as often as 1/8");
        }
        {   // the trigger filter: the key is a 440 Hz tone
            HostEngine e; listener (e); e.trackAt (2)->source.store (2); set (e, 2, "mute", true); set (e, 0, "compKey", e.trackAt (2)->uid.load());
            auto level = [&] { run (e, 200, bs); auto o = run (e, 100, bs); return peakIn (o, 0, (int) o.size()); };
            const float open = level(); CHECK (open < 0.12f, "wide open, the key ducks");
            set (e, 0, "compKeyHi", 60.0); const float noLows = level(); set (e, 0, "compKeyHi", 20000.0); set (e, 0, "compKeyLo", 4000.0); const float noHighs = level();
            CHECK (noLows > 0.2f && noHighs > 0.2f, "a High cut below the key, or a Low cut above it, takes it out of the trigger"); set (e, 0, "compKeyLo", 20.0); CHECK (level() < 0.12f, "and opening the filter brings it back");
            set (e, 0, "compKeyListen", true); set (e, 0, "compOn", false); run (e, 40, bs); CHECK (std::abs (level() - 0.25f) < 0.01f, "Listen plays the trigger");
            set (e, 0, "compKeyLo", 4000.0); CHECK (level() < 0.03f, "through its filter");
            set (e, 0, "compKeyListen", false); set (e, 0, "compKeyLo", 20.0);
            // Duck below
            set (e, 0, "compOn", true); set (e, 0, "compBelow", 47); const float split = level(); CHECK (split > 0.2f, "Duck below 47 Hz leaves a 440 Hz tone alone");
            set (e, 0, "compBelow", 0); CHECK (level() < 0.12f, "All ducks it again"); CHECK (! (set (e, 0, "compBelow", 99), e.trackAt (0)->compBelow.load() != 0), "only 22, 35 and 47 are Duck below points");
            // saved with the song, and an older song is untouched
            set (e, 0, "compKeyLo", 35.0); set (e, 0, "compKeyHi", 120.0); set (e, 0, "compBelow", 35); set (e, 0, "gateKey", -4);
            auto song = e.saveProject(); HostEngine g; g.prepareOffline (sr, bs, 0); g.loadProject (song, [] (juce::String) {});
            auto& t = *g.trackAt (0); CHECK (std::abs (t.compSide.lo.load() - 35.0f) < 0.01f && std::abs (t.compSide.hi.load() - 120.0f) < 0.01f && t.compBelow.load() == 35 && t.gateKey.load() == -4, "the trigger settings are saved with the song");
            HostEngine old; old.prepareOffline (sr, bs, 0); CHECK (old.trackAt (0)->compSide.lo.load() == 20.0f && old.trackAt (0)->compSide.hi.load() == 20000.0f && old.trackAt (0)->compBelow.load() == 0, "a new or older song starts fully open");
        }
        {   // an FX return as the trigger
            HostEngine e; listener (e); e.trackAt (2)->source.store (2); set (e, 2, "gainDb", -90.0); e.setSend (*e.trackAt (2), 0, 0.0f, true); CHECK (set (e, kReturnBase, "gainDb", -90.0), "the return is turned down so only the listener is measured"); set (e, 0, "compKey", -101);
            run (e, 300, bs); auto o = run (e, 100, bs); std::printf ("  return key: isKey %d, level %.3f, from %d\n", (int) e.planNow()->returns[0].isKey, peakIn (o, 0, (int) o.size()), (int) e.planNow()->returns[0].fromTracks.size()); CHECK (e.planNow()->returns[0].isKey && peakIn (o, 0, (int) o.size()) < 0.16f, "an FX return can trigger the compressor");
        }
        {   // a plug-in's side-chain input
            HostEngine e; e.prepareOffline (sr, bs, 0); e.transport.setBpm (120.0); e.trackAt (0)->source.store (2);
            auto pr = std::make_unique<ScProc>(); auto* p = pr.get(); e.addProcessor (0, std::move (pr), "side probe");
            auto& slot = *e.trackAt (0)->slots[0]; CHECK (slot.scBus == 1 && ! p->getBus (true, 1)->isEnabled(), "an effect with a side-chain has it off until something is chosen");
            e.trackAt (2)->source.store (2); set (e, 2, "mute", true); const int uid = e.trackAt (2)->uid.load();
            CHECK (e.editSlotSide (0, 0, "key", uid) && p->getBus (true, 1)->isEnabled() && slot.scCount.load() == 2, "choosing a track switches the side-chain on");
            p->heard.clear(); auto o = run (e, 100, bs); CHECK (std::abs (p->peakFrom (p->heard.size() / 2) - 0.25f) < 0.01f, "the plug-in hears that track, even muted");
            CHECK (std::abs (peakIn (o, 12000, (int) o.size()) - 0.25f) < 0.01f, "and its own sound is what comes out");
            CHECK (e.planNow()->levels.size() == 2, "the track it listens to is rendered first");
            e.editSlotSide (0, 0, "hi", 60.0); run (e, 100, bs); p->heard.clear(); run (e, 60, bs); CHECK (p->peakFrom (0) < 0.02f, "the High cut shapes what the plug-in hears"); e.editSlotSide (0, 0, "hi", 20000.0);
            CHECK (e.editSlotSide (0, 0, "key", -1), "a pulse can feed it"); run (e, 20, bs); p->heard.clear(); run (e, 100, bs); CHECK (p->peakFrom (0) < 1.0e-6f, "silent while stopped");
            e.transport.locate (0.0); p->heard.clear(); e.transport.play(); run (e, 200, bs);
            size_t first = 0; while (first < p->heard.size() && std::abs (p->heard[first]) < 0.01f) ++first; CHECK (first < 200 && p->peakFrom (0) > 0.5f, "and thuds on the beat from the moment the song plays");
            e.transport.stop(); e.editSlotSide (0, 0, "key", uid);
            auto song = e.saveProject(); CHECK (juce::JSON::toString (song).contains ("\"sc\""), "the choice is saved with the song");
            CHECK (! e.editSlotSide (0, 0, "key", e.trackAt (0)->uid.load()) || slot.scKey.load() == 0, "a track never feeds its own plug-in");
            e.editSlotSide (0, 0, "key", uid); CHECK (e.editSlotSide (0, 0, "key", 0) && ! p->getBus (true, 1)->isEnabled() && slot.scCount.load() == 0, "Off switches the side-chain off again");
            e.undo.undo(); CHECK (slot.scKey.load() == uid, "and Undo brings the choice back");
        }
        {   // a plug-in without a side-chain, an instrument that takes its trigger on its own input, and a plug-in after one that adds delay
            HostEngine e; e.prepareOffline (sr, bs, 0); e.transport.setBpm (120.0);
            e.addProcessor (0, std::make_unique<TestProc> (2, 2, 0), "plain"); CHECK (! e.editSlotSide (0, 0, "key", -1), "a plug-in with no side-chain input refuses one");
            auto& slot = *e.trackAt (0)->slots[0]; slot.desc.isInstrument = true; slot.scBus = 0; e.trackAt (2)->source.store (2); set (e, 2, "mute", true);
            CHECK (e.editSlotSide (0, 0, "key", e.trackAt (2)->uid.load()), "an instrument with an audio input takes the trigger there"); run (e, 100, bs); auto o = run (e, 60, bs); CHECK (std::abs (peakIn (o, 0, (int) o.size()) - 0.25f) < 0.01f, "it arrives at the instrument's input");
            HostEngine g; g.prepareOffline (sr, bs, 0); g.transport.setBpm (120.0); g.addProcessor (0, std::make_unique<LatProc> (2, 2, 1000), "late"); auto pr = std::make_unique<ScProc>(); auto* p = pr.get(); g.addProcessor (0, std::move (pr), "side probe");
            g.editSlotSide (0, 1, "key", -1); run (g, 8, bs); p->heard.clear(); g.transport.locate (0.0); g.transport.play(); run (g, 100, bs);
            size_t first = 0; while (first < p->heard.size() && std::abs (p->heard[first]) < 0.01f) ++first; CHECK (first >= 990 && first < 1200, "a pulse waits for the sound it works on when a plug-in before it adds delay");
        }
    }

    // GROOVE: pre-align and one global shuffle
    {
        GrooveSet g = defaultGrooveSet(); g.ch[1].on = true; g.ch[1].patch = "Straight"; g.ch[1].shuffle = 60.0f; g.globalShuffle = 30.0f;
        auto place = [&] (double at) { NoteEv n; n.beat = at; n.len = 0.2; double b = 0, l = 0; int v = 100; GrooveEval::make (g, 1, 1).apply (n, b, l, v); return b; };
        CHECK (std::abs (place (0.27) - (0.27 + 0.6 * 0.25 / 3.0)) < 1e-6, "without pre-align a late note keeps its lateness and takes the channel's shuffle");
        g.ch[1].preAlign = true; CHECK (std::abs (place (0.27) - (0.25 + 0.6 * 0.25 / 3.0)) < 1e-6, "pre-align puts it on the grid first");
        g.ch[1].useGlobal = true; CHECK (std::abs (place (0.27) - (0.25 + 0.3 * 0.25 / 3.0)) < 1e-6, "global shuffle uses the shared amount instead of the channel's knob");
        HostEngine e; CHECK (e.editGrooveChan (1, "pre", true) && e.editGrooveChan (1, "glob", true) && e.editGrooveChan (1, "globalShuffle", 45.0), "all three can be set");
        auto song = e.saveProject(); HostEngine f; f.prepareOffline (sr, bs, 0); f.loadProject (song, [] (juce::String) {}); const auto gs = f.grooveCopy();
        CHECK (gs.ch[1].preAlign && gs.ch[1].useGlobal && std::abs (gs.globalShuffle - 45.0f) < 1e-4f, "and are saved with the song");
    }

    // ROUTING: mix channels (buses), parallel channels, solo and latency along the routes
    {
        auto peak = [&] (HostEngine& e) { run (e, 200, bs); auto o = run (e, 60, bs); float pk = 0; for (float v : o) pk = std::max (pk, std::abs (v)); return pk; };
        auto set = [] (HostEngine& e, int track, const char* key, juce::var v) { return e.editChainParam (track, key, v); };
        HostEngine e; e.prepareOffline (sr, bs, 0); e.undo.clock = [] { static double c = 0; c += 5000; return c; };
        for (int i = 0; i < 3; ++i) e.trackAt (i)->source.store (2);             // three 0.25 tones, started together so they add in step
        e.editChainParam (1, "mute", true); e.editChainParam (2, "mute", true);
        const int bus = e.editAddBus(); const int busUid = e.trackAt (bus)->uid.load();
        CHECK (bus == 4 && e.trackAt (bus)->kind.load() == 2 && e.trackAt (bus)->name == "Bus 1", "Add Mix Channel makes a bus");
        CHECK (std::abs (peak (e) - 0.25f) < 0.002f, "a track goes to the master until it is routed");
        CHECK (set (e, 0, "outTo", busUid), "and can be routed to the bus");
        CHECK (std::abs (peak (e) - 0.25f) < 0.002f, "through the bus it sounds the same");
        set (e, bus, "gainDb", -6.0206); CHECK (std::abs (peak (e) - 0.125f) < 0.002f, "the bus fader sets the level of what is routed into it");
        set (e, bus, "mute", true); CHECK (peak (e) < 0.001f, "muting the bus mutes it"); set (e, bus, "mute", false); set (e, bus, "gainDb", 0.0);
        set (e, 1, "mute", false); set (e, 1, "outTo", busUid); CHECK (std::abs (peak (e) - 0.5f) < 0.004f, "two tracks add together in the bus");
        set (e, 2, "mute", false);                                                // a third tone straight to the master
        set (e, 0, "solo", true); CHECK (std::abs (peak (e) - 0.25f) < 0.004f, "soloing a track keeps its bus open and nothing else"); set (e, 0, "solo", false);
        set (e, bus, "solo", true); CHECK (std::abs (peak (e) - 0.5f) < 0.004f, "soloing a bus plays everything routed into it"); set (e, bus, "solo", false);
        set (e, 2, "mute", true); set (e, 1, "mute", true);
        // buses into buses, and no circles
        const int bus2 = e.editAddBus(); const int u2 = e.trackAt (bus2)->uid.load();
        CHECK (set (e, bus, "outTo", u2) && std::abs (peak (e) - 0.25f) < 0.002f, "a bus can go into another bus");
        CHECK (! set (e, bus2, "outTo", busUid) && ! set (e, bus, "outTo", busUid) && ! set (e, 0, "outTo", e.trackAt (1)->uid.load()), "a circle, itself, and a track that is not a bus are refused");
        set (e, bus2, "gainDb", -6.0206); CHECK (std::abs (peak (e) - 0.125f) < 0.002f, "and the second bus's fader applies to the whole chain");
        auto song = e.saveProject(); HostEngine g; g.prepareOffline (sr, bs, 0); g.loadProject (song, [] (juce::String) {});
        CHECK (g.trackAt (0)->outTo.load() == busUid && g.trackAt (bus)->kind.load() == 2 && std::abs (peak (g) - 0.125f) < 0.002f, "the routing is saved with the song");
        e.editRemoveTrack (bus2); CHECK (std::abs (peak (e) - 0.25f) < 0.002f, "removing a bus sends what fed it to the master");
        e.undo.undo(); for (int i = 0; i < 50; ++i) e.service(); CHECK (e.trackIndexOfUid (u2) >= 0 && std::abs (peak (e) - 0.125f) < 0.002f, "undo brings the bus and its routing back");
    }
    {
        auto peak = [&] (HostEngine& e) { run (e, 200, bs); auto o = run (e, 60, bs); float pk = 0; for (float v : o) pk = std::max (pk, std::abs (v)); return pk; };
        HostEngine e; e.prepareOffline (sr, bs, 0); e.undo.clock = [] { static double c = 0; c += 5000; return c; }; e.trackAt (0)->source.store (2); e.trackAt (0)->name = "Drums";
        const int nb = e.editRouteToNewBus (0); CHECK (nb == 4 && e.trackAt (0)->outTo.load() == e.trackAt (nb)->uid.load() && std::abs (peak (e) - 0.25f) < 0.002f, "Route to New Output Bus makes the bus and routes the track, in one step");
        e.undo.undo(); CHECK (e.trackCount.load() == 4 && e.trackAt (0)->outTo.load() == 0, "one undo takes both back"); e.undo.redo(); CHECK (e.trackCount.load() == 5 && e.trackAt (0)->outTo.load() == e.trackAt (4)->uid.load(), "and redo restores them");
        const int p = e.editParallelChannel (0); CHECK (p == 5 && e.trackAt (p)->name == "Drums parallel" && e.trackAt (p)->outTo.load() == e.trackAt (0)->outTo.load(), "a parallel channel goes where its track goes");
        CHECK (std::abs (peak (e) - 0.5f) < 0.004f, "it hears the same sound and adds beside it");
        e.editChainParam (0, "gainDb", -90.0); CHECK (std::abs (peak (e) - 0.25f) < 0.004f, "its level does not depend on the first channel's fader");
        e.editChainParam (p, "gainDb", -6.0206); CHECK (std::abs (peak (e) - 0.125f) < 0.004f, "and it has a fader of its own"); e.editChainParam (0, "gainDb", 0.0); e.editChainParam (p, "gainDb", 0.0);
        e.editChainParam (0, "solo", true); CHECK (std::abs (peak (e) - 0.5f) < 0.004f, "soloing a track keeps its parallel channel with it");
    }
    {   // latency: a plug-in's delay on one routed track is made up on the others and on the track beside the bus
        HostEngine e; e.prepareOffline (sr, bs, 0);
        e.addProcessor (0, std::make_unique<LatProc> (2, 2, 100), "late"); const int bus = e.editAddBus(); const int u = e.trackAt (bus)->uid.load();
        e.editChainParam (0, "outTo", u); e.editChainParam (1, "outTo", u); e.addProcessor (bus, std::make_unique<LatProc> (2, 2, 50), "later");
        const auto* plan = e.planNow();
        CHECK (plan->tracks[0].delay == 0 && plan->tracks[1].delay == 100, "inside a bus every input lines up with the slowest");
        CHECK (plan->tracks[(size_t) bus].delay == 0 && plan->tracks[2].delay == 150 && plan->trackLatency == 150, "and a track outside lines up with the whole bus path");
    }

    std::printf("%d checks, %d failed\n", checks, failures);
    return failures == 0 ? 0 : 1;
}
