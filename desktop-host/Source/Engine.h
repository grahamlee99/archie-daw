// Archie Studio host (desktop) -- the audio engine.
// One audio callback: the master transport advances, tracks make MIDI (live keys, recorded clip, notes sent by a web page) and audio
// (test signals, a loaded audio file, the sound-card input), each track runs through its chain of hosted VST3 plugins, then a mixer
// strip (gain, pan, mute/solo, meter) into the master and out.
#pragma once
#include <juce_audio_utils/juce_audio_utils.h>
#include <juce_audio_processors/juce_audio_processors.h>
#include <juce_audio_devices/juce_audio_devices.h>
#include <juce_audio_formats/juce_audio_formats.h>
#include <juce_dsp/juce_dsp.h>
#include <atomic>
#include <array>
#include <map>
#include <set>
#include <set>
#include <algorithm>
#include <memory>
#include "KeepLoaded.h"
#include "Transport.h"
#include "Mixer.h"
#include "Renderer.h"
#include "Undo.h"
#include "Warp.h"
#include "Pitch.h"
#include "TapTempo.h"
#include <mutex>

namespace archie
{

class HostEngine : public juce::AudioIODeviceCallback, public juce::MidiInputCallback
{
public:
    HostEngine()
    {
        formatManager.addDefaultFormats();
        for (int i = 0; i < kTracks; ++i)
        {
            auto t = std::make_unique<TrackState>();
            t->index = i; t->name = "Track " + juce::String (i + 1);
            tracks[(size_t) i] = std::move(t);
        }
        for (int r = 0; r < kReturns; ++r) { auto t = std::make_unique<TrackState>(); t->index = kReturnBase + r; t->name = "FX " + juce::String (r + 1); returns[(size_t) r] = std::move (t); }
        masterChain = std::make_unique<TrackState>(); masterChain->index = kMasterChain; masterChain->name = "Master";
        for (auto& t : tracks) t->blk = &blk; for (auto& t : returns) t->blk = &blk;
        settingsDir = juce::File::getSpecialLocation (juce::File::userApplicationDataDirectory).getChildFile ("AskArchie").getChildFile ("Studio");
        settingsDir.createDirectory();
        loadAudioOptions();
        loadKnownPlugins();
        readLoadingMarker();
        deviceMetaFile = settingsDir.getChildFile (juce::SystemStats::getEnvironmentVariable ("ASKARCHIE_HOST_TEST", {}) == "1" ? "device-meta-test.json" : "device-meta.json");     // a test copy never touches the real notes
        loadDeviceMeta();
        loadCuePads();
        claimSharedWebViewWindowClass();      // before any plug-in loads: see KeepLoaded.h
        if (juce::SystemStats::getEnvironmentVariable ("ASKARCHIE_HOST_TEST", {}) == "1") recordingsDir = settingsDir.getChildFile ("recordings-test");          // a test copy keeps its takes apart from the real ones
    }

    ~HostEngine() override { shutdown(); }

    // An audio interface can have many inputs (18 on a 1824c). They are all opened (up to kMaxInputs), so each audio track can be routed to any one of them.
    static constexpr int kMaxInputs = 32;
    void enableAllInputs()
    {
        if (auto* d = deviceManager.getCurrentAudioDevice())
        {
            const int n = juce::jmin (kMaxInputs, d->getInputChannelNames().size());
            if (n > 2) { auto s = deviceManager.getAudioDeviceSetup(); s.useDefaultInputChannels = false; s.inputChannels.clear(); s.inputChannels.setRange (0, n, true); deviceManager.setAudioDeviceSetup (s, true); }
        }
    }
    juce::StringArray inputChannelNames() const
    {
        juce::StringArray out; if (auto* d = deviceManager.getCurrentAudioDevice()) { auto all = d->getInputChannelNames(); auto act = d->getActiveInputChannels(); int k = 0; for (int i = 0; i < all.size() && k < kMaxInputs; ++i) if (act[i]) { out.add (all[i]); ++k; } }
        return out;
    }
    void start()
    {
        auto err = deviceManager.initialiseWithDefaultDevices (2, 2);
        lastDeviceError = err;
        enableAllInputs();
        buildMixer();
        deviceManager.addAudioCallback (this);
        refreshMidiInputs(); updateTargets();
    }

    void shutdown()
    {
        if (shutDown) return;
        shutDown = true;
        warpCancel.store (true); warpPool.removeAllJobs (true, 3000);
        scanThread.reset(); analysisThread.reset();
        deviceManager.removeAudioCallback (this);
        audioActive.store (false);
        renderer.shutdown();
        for (auto& t : tracks) t->slots.clear();
        for (auto& id : openMidiInputs) deviceManager.removeMidiInputDeviceCallback (id, this);
        openMidiInputs.clear();
        midiOut.reset();
        deviceManager.closeAudioDevice();
    }

    //==========================================================================
    // audio thread
    void audioDeviceAboutToStart (juce::AudioIODevice* d) override
    {
        prepareAudio (d->getCurrentSampleRate(), d->getCurrentBufferSizeSamples(), -1);
        for (auto& t : tracks) if (t->filePath.isNotEmpty()) loadAudioFile (t->index, juce::File (t->filePath));
        audioActive.store (true);
    }
    void audioDeviceStopped() override { audioActive.store (false); }

    // Everything the mixer will touch is allocated here, on the message thread, before audio runs. workers < 0 = a sensible number for this computer (ASKARCHIE_WORKERS overrides it).
    void prepareAudio (double sr, int bs, int workers)
    {
        audioActive.store (false);
        sampleRate = sr; blockSize = bs;
        renderMax = juce::jmax (bs * 2, 1024);
        scratch.setSize (kMaxInputs, renderMax); outBuf.setSize (2, renderMax);
        int w = workers;
        if (w < 0) { const auto env = juce::SystemStats::getEnvironmentVariable ("ASKARCHIE_WORKERS", {}); w = env.isNotEmpty() ? env.getIntValue() : maxThreadsSetting.load() >= 0 ? maxThreadsSetting.load() : juce::jlimit (0, 3, juce::SystemStats::getNumPhysicalCpus() - 1); }
        renderer.prepare (sr, renderMax, juce::jlimit (0, 7, w));
        if (masterOwner == nullptr) buildMixer();          // headless use (tests) never goes through start()
        prepareAllProcessors();
        publishPlan();
    }
    // headless use (tests, bounces): the same renderer, no sound card
    void prepareOffline (double sr, int bs, int workers) { prepareAudio (sr, bs, workers); }
    void setParallelThresholdMicros (double us) { renderer.setParallelThresholdMicros (us); }      // tests force the parallel path with 0
    void processOffline (const float* const* in, int nIn, float* const* out, int nOut, int n) { processBlockCore (in, nIn, out, nOut, n, false); }       // a bounce, an export or a test: the control room does not touch it
    void processLive (const float* const* in, int nIn, float* const* out, int nOut, int n) { processBlockCore (in, nIn, out, nOut, n, true); }              // what the sound card gets (the tests use it too)

    void audioDeviceIOCallbackWithContext (const float* const* in, int nIn, float* const* out, int nOut, int n, const juce::AudioIODeviceCallbackContext&) override { processBlockCore (in, nIn, out, nOut, n, true); }

    // GROOVES (Groove dock): thirty-one groove channels (A2 to D8) and the patch library, read by the audio thread through a snapshot. Every edit copies the set, changes the copy and publishes it, and is one undo step.
    Snapshot<GrooveSet, 16> grooveSnap { std::make_unique<GrooveSet> (defaultGrooveSet()) };
    GrooveSet grooveCopy() const { return *grooveSnap.load(); }
    void putGroove (const GrooveSet& g) { grooveSnap.publish (std::make_unique<GrooveSet> (g)); stateDirty.store (true); }
    static juce::var grooveChanParam (const GrooveSet& g, int ch, const juce::String& key)
    {
        if (ch < 1 || ch >= kGrooveChans) return {}; const auto& c = g.ch[ch];
        if (key == "on") return c.on; if (key == "patch") return c.patch; if (key == "strength") return (double) c.strength; if (key == "timing") return (double) c.timing; if (key == "vel") return (double) c.vel;
        if (key == "len") return (double) c.len; if (key == "rand") return (double) c.rand; if (key == "slide") return (double) c.slide; if (key == "shuffle") return (double) c.shuffle; if (key == "anchor") return c.anchor;
        return {};
    }
    static bool applyGrooveChanParam (GrooveSet& g, int ch, const juce::String& key, const juce::var& v)
    {
        if (ch < 1 || ch >= kGrooveChans) return false; auto& c = g.ch[ch];
        if (key == "on") c.on = (bool) v;
        else if (key == "patch") { const auto n = v.toString(); if (n.isNotEmpty() && g.find (n) == nullptr) return false; c.patch = n; }
        else if (key == "strength") c.strength = (float) juce::jlimit (0.0, 1.0, (double) v);
        else if (key == "timing") c.timing = (float) juce::jlimit (0.0, 200.0, (double) v); else if (key == "vel") c.vel = (float) juce::jlimit (0.0, 200.0, (double) v); else if (key == "len") c.len = (float) juce::jlimit (0.0, 200.0, (double) v);
        else if (key == "rand") c.rand = (float) juce::jlimit (0.0, 120.0, (double) v); else if (key == "slide") c.slide = (float) juce::jlimit (-50.0, 50.0, (double) v); else if (key == "shuffle") c.shuffle = (float) juce::jlimit (0.0, 100.0, (double) v);
        else if (key == "anchor") c.anchor = juce::jlimit (0, 63, (int) v); else return false;
        return true;
    }
    bool editGrooveChan (int ch, const juce::String& key, const juce::var& v)
    {
        auto before = grooveCopy(); if (grooveChanParam (before, ch, key).isVoid() && key != "patch") return false;
        auto after = before; if (! applyGrooveChanParam (after, ch, key, v)) return false; if (grooveChanParam (before, ch, key) == grooveChanParam (after, ch, key)) return true;
        putGroove (after); const bool dragged = key != "on" && key != "patch" && key != "anchor";
        undo.record (("Groove " + juce::String (ch) + " " + key).toStdString(), dragged ? ("g:" + juce::String (ch) + ":" + key).toStdString() : std::string(), [this, before] { putGroove (before); }, [this, after] { putGroove (after); });
        return true;
    }
    static juce::String uniqueGrooveName (const GrooveSet& g, juce::String base) { if (base.trim().isEmpty()) base = "My groove"; base = base.trim().substring (0, 40); auto n = base; for (int i = 2; g.find (n) != nullptr; ++i) n = base + " " + juce::String (i); return n; }
    // a groove from the playing of a note clip: per step of a one-bar cycle, how early or late its notes are (as a fraction of a step), how loud (relative to the clip's average) and how long (likewise)
    int extractGroove (int contentId, const juce::String& name, double grid, juce::String& err)
    {
        std::shared_ptr<const Clip> clip;
        for (int ti = 0; ti < trackCount.load() && clip == nullptr; ++ti) for (const auto& in : tracks[(size_t) ti]->arrSnap.load()->insts) if (in.content != nullptr && in.content->id == contentId) { clip = in.content; break; }
        if (clip == nullptr || clip->notes.empty()) { err = clip == nullptr ? "no such clip" : "that clip has no notes"; return 0; }
        grid = grid >= 0.37 ? 0.5 : 0.25; GroovePatch p; p.grid = grid; p.cycle = 4.0; p.steps.resize ((size_t) p.stepCount()); const int n = p.stepCount();
        double velMean = 0, lenMean = 0; for (const auto& nt : clip->notes) { velMean += nt.vel; lenMean += nt.len; } velMean /= (double) clip->notes.size(); lenMean /= (double) clip->notes.size();
        std::vector<double> tSum ((size_t) n, 0.0), vSum ((size_t) n, 0.0), lSum ((size_t) n, 0.0); std::vector<int> cnt ((size_t) n, 0);
        for (const auto& nt : clip->notes) { const long long idx = std::llround (nt.beat / grid); const int s = (int) (((idx % n) + n) % n); tSum[(size_t) s] += (nt.beat - (double) idx * grid) / grid; vSum[(size_t) s] += nt.vel / velMean - 1.0; lSum[(size_t) s] += nt.len / lenMean - 1.0; ++cnt[(size_t) s]; }
        for (int s = 0; s < n; ++s) if (cnt[(size_t) s] > 0) { p.steps[(size_t) s].timing = (float) (tSum[(size_t) s] / cnt[(size_t) s]); p.steps[(size_t) s].vel = (float) (100.0 * vSum[(size_t) s] / cnt[(size_t) s]); p.steps[(size_t) s].len = (float) (100.0 * lSum[(size_t) s] / cnt[(size_t) s]); }
        auto before = grooveCopy(); auto after = before; p.name = uniqueGrooveName (after, name); after.patches.push_back (p); putGroove (after);
        undo.record ("Extract groove", "", [this, before] { putGroove (before); }, [this, after] { putGroove (after); });
        return (int) after.patches.size();
    }
    bool deleteGroove (const juce::String& name)
    {
        auto before = grooveCopy(); auto after = before; bool done = false;
        for (size_t i = 0; i < after.patches.size(); ++i) if (after.patches[i].name == name && ! after.patches[i].builtin) { after.patches.erase (after.patches.begin() + (long) i); done = true; break; }
        if (! done) return false; putGroove (after); undo.record ("Delete groove", "", [this, before] { putGroove (before); }, [this, after] { putGroove (after); }); return true;
    }
    juce::var grooveVar (bool forProject)
    {
        const auto& g = *grooveSnap.load(); auto* o = new juce::DynamicObject(); juce::Array<juce::var> chs, pats;
        for (int c = 0; c < kGrooveChans; ++c) { auto* x = new juce::DynamicObject(); for (const char* k : { "on", "patch", "strength", "timing", "vel", "len", "rand", "slide", "shuffle", "anchor" }) x->setProperty (k, c == 0 ? juce::var() : grooveChanParam (g, c, k)); chs.add (juce::var (x)); }
        for (const auto& p : g.patches)
        {
            if (forProject && p.builtin) continue;
            auto* x = new juce::DynamicObject(); x->setProperty ("name", p.name); x->setProperty ("builtin", p.builtin); x->setProperty ("grid", p.grid); x->setProperty ("cycle", p.cycle);
            if (forProject) { juce::Array<juce::var> st; for (const auto& m : p.steps) { juce::Array<juce::var> a; a.add ((double) m.timing); a.add ((double) m.vel); a.add ((double) m.len); st.add (juce::var (a)); } x->setProperty ("steps", st); }
            pats.add (juce::var (x));
        }
        o->setProperty ("channels", chs); o->setProperty ("patches", pats); return juce::var (o);
    }
    void loadGroove (const juce::var& p)
    {
        GrooveSet g = defaultGrooveSet(); const auto gv = p.getProperty ("groove", {});
        if (auto* pats = gv.getProperty ("patches", {}).getArray())
            for (const auto& pv : *pats)
            {
                GroovePatch q; q.name = pv.getProperty ("name", {}).toString(); q.grid = (double) pv.getProperty ("grid", 0.25) >= 0.37 ? 0.5 : 0.25; q.cycle = 4.0; q.steps.resize ((size_t) q.stepCount());
                if (auto* st = pv.getProperty ("steps", {}).getArray()) for (size_t i = 0; i < q.steps.size() && i < (size_t) st->size(); ++i) if (auto* a = (*st)[(int) i].getArray()) if (a->size() >= 3) q.steps[i] = { (float) (double) (*a)[0], (float) (double) (*a)[1], (float) (double) (*a)[2] };
                if (q.name.isNotEmpty() && g.find (q.name) == nullptr) g.patches.push_back (q);
            }
        if (auto* chs = gv.getProperty ("channels", {}).getArray())
            for (int c = 1; c < kGrooveChans && c < chs->size(); ++c) { const auto& x = (*chs)[c]; for (const char* k : { "on", "patch", "strength", "timing", "vel", "len", "rand", "slide", "shuffle", "anchor" }) { const auto v = x.getProperty (k, {}); if (! v.isVoid()) applyGrooveChanParam (g, c, k, v); } }
        putGroove (g);
    }
    // THE CONTROL ROOM (Mixer, Master Section): the monitoring feed. Its level and DIM (-20 dB) apply to what the sound card plays, never to a bounce or an export. It can be the main outputs themselves
    // (crPair 0: everything that carries the master is turned down) or a pair of its own (crPair 1 is outputs 3/4, 2 is 5/6 and so on: the main outputs 1/2 then carry the master unchanged, the others are silent).
    // At 0 dB, no DIM and the main outputs it touches nothing (bit for bit); moving the level or DIM is smoothed over 20 ms.
    std::atomic<float> crDb { 0.0f }; std::atomic<bool> crDim { false }; std::atomic<int> crPair { 0 };
    juce::SmoothedValue<float> crGain { 1.0f }; double crRate = 0.0;
    void writeOutputs (float* const* out, int nOut, const juce::AudioBuffer<float>& view, int n, bool live)
    {
        const float target = live ? juce::Decibels::decibelsToGain (crDb.load() - (crDim.load() ? 20.0f : 0.0f), -90.0f) : 1.0f;
        const int pair = live ? crPair.load() : 0;
        if (crRate != sampleRate) { crRate = sampleRate; crGain.reset (sampleRate, 0.02); crGain.setCurrentAndTargetValue (1.0f); }
        const bool separate = pair > 0 && nOut >= 2 * pair + 2;
        if (target == 1.0f && ! separate && crGain.getTargetValue() == 1.0f && ! crGain.isSmoothing())
        { for (int c = 0; c < nOut; ++c) juce::FloatVectorOperations::copy (out[c], view.getReadPointer (juce::jmin (c, 1)), n); return; }
        crGain.setTargetValue (target);
        if (! separate) { for (int i = 0; i < n; ++i) { const float g = crGain.getNextValue(); for (int c = 0; c < nOut; ++c) out[c][i] = view.getSample (juce::jmin (c, 1), i) * g; } }
        else
        {
            for (int c = 0; c < 2; ++c) juce::FloatVectorOperations::copy (out[c], view.getReadPointer (c), n);
            for (int i = 0; i < n; ++i) { const float g = crGain.getNextValue(); out[2 * pair][i] = view.getSample (0, i) * g; out[2 * pair + 1][i] = view.getSample (1, i) * g; }
        }
    }
    void processBlockCore (const float* const* in, int nIn, float* const* out, int nOut, int n, bool live = false)
    {
        juce::ScopedNoDenormals noDenormals;
        for (int c = 0; c < nOut; ++c) juce::FloatVectorOperations::clear (out[c], n);
        if (n > renderMax || shutDown) return;

        outMidi.clear();
        Segment segs[2]; int ns = 0; bool started = false, stopped = false, jumped = false;
        transport.advance (n, sampleRate, outMidi, segs, ns, started, stopped, jumped);
        blk.numSegs = ns; blk.segs[0] = segs[0]; blk.segs[1] = segs[1];
        blk.playing = transport.playing.load(); blk.recording = transport.recording.load();
        blk.blockStartMs = juce::Time::getMillisecondCounterHiRes(); blk.panic = stopped || jumped; blk.chase = started || jumped; blk.bpm = transport.bpm.load(); blk.sr = sampleRate;
        blk.groove = grooveSnap.load();
        { const double g = transport.swingGrid.load(); blk.swingGrid = g; blk.swingDelay = juce::jlimit (0.0, 1.0, transport.swing.load()) * g / 3.0; }
        scratch.clear();
        if (nIn > 0 && in != nullptr)
        {
            const int nc = juce::jmin (nIn, kMaxInputs);
            for (int c = 0; c < nc; ++c) scratch.copyFrom (c, 0, in[c], n);
            if (nc == 1) scratch.copyFrom (1, 0, in[0], n);
        }
        juce::AudioBuffer<float> inView (scratch.getArrayOfWritePointers(), juce::jlimit (2, kMaxInputs, nIn), n), view (outBuf.getArrayOfWritePointers(), 2, n);
        // A block that crosses the loop end is rendered as TWO pieces, each told its own true position. A plug-in follows the host's position (a self-sequencing instrument such as Archie Acid 303 reads the
        // beat at the start of every block and works out the steps from it): told only the first piece's position for the whole block, it carried straight on past the loop end, then saw the next block jump back
        // by the whole loop and shut its sounding note: the first note after the wrap, which had just started, was cut off. Every other host does this split. One piece (every block but a wrap) is exactly as before.
        if (ns <= 1) { updatePlayHead (segs[0]); renderer.render (inView, view, n); }
        else
        {
            const Segment keep[2] = { segs[0], segs[1] }; const bool keepPanic = blk.panic, keepChase = blk.chase; const double keepStartMs = blk.blockStartMs;
            for (int k = 0; k < ns; ++k)
            {
                const int off = keep[k].startSample, len = keep[k].numSamples; if (len <= 0) continue;
                Segment one = keep[k]; one.startSample = 0; blk.numSegs = 1; blk.segs[0] = one;
                blk.panic = k == 0 ? keepPanic : false; blk.chase = k == 0 ? keepChase : false; blk.blockStartMs = keepStartMs + (double) off * 1000.0 / sampleRate;
                updatePlayHead (one);
                float* ip[kMaxInputs]; const int inCh = juce::jlimit (2, kMaxInputs, nIn); for (int c2 = 0; c2 < inCh; ++c2) ip[c2] = scratch.getWritePointer (c2) + off;
                float* op[2] = { outBuf.getWritePointer (0) + off, outBuf.getWritePointer (1) + off };
                juce::AudioBuffer<float> inPart (ip, inCh, len), outPart (op, 2, len);
                renderer.render (inPart, outPart, len);
            }
            blk.numSegs = ns; blk.segs[0] = keep[0]; blk.segs[1] = keep[1]; blk.panic = keepPanic; blk.chase = keepChase; blk.blockStartMs = keepStartMs;
        }
        addMetronome (view, segs, ns);

        writeOutputs (out, nOut, view, n, live);

        if (! outMidi.isEmpty())
        {
            const juce::SpinLock::ScopedTryLockType lock (midiOutLock);
            if (lock.isLocked() && midiOut != nullptr)
                midiOut->sendBlockOfMessages (outMidi, juce::Time::getMillisecondCounterHiRes() + 1.0, sampleRate);
        }
    }

    // MIDI thread: hardware keyboards, controllers, clocks
    void handleIncomingMidiMessage (juce::MidiInput* src, const juce::MidiMessage& m) override
    {
        const double now = juce::Time::getMillisecondCounterHiRes();
        transport.handleIncomingSync (m, now);
        if (cueConsume (m)) return;                                            // a controller pad assigned (or just being assigned) to a hot cue
        if (remoteConsume (m)) return;                                         // an assigned (or just-assigned) controller knob
        const bool isPlayed = m.isNoteOnOrOff() || m.isController() || m.isPitchWheel() || m.isChannelPressure() || m.isAftertouch() || m.isProgramChange();
        if (isPlayed)
        {
            auto copy = m; copy.setTimeStamp (now * 0.001);
            LiveMsg lm; if (liveOf (copy, now, lm)) for (auto& t : tracks) if (t->recTarget.load()) t->liveMidi.push (lm);
        }
        // what the console shows: any message except the constant background ones (active sensing; clock only when nothing else has arrived lately)
        if (m.getRawDataSize() > 0 && m.getRawData()[0] != 0xFE)
        {
            const bool clock = m.isMidiClock();
            if (! clock || now - lastPlayedMs.load() > 1500.0)
            {
                const juce::SpinLock::ScopedLockType lock (midiTextLock);
                lastMidiText = (src != nullptr ? src->getName() : juce::String ("MIDI input")) + ": " + (clock ? juce::String ("MIDI clock running") : m.getDescription());
            }
            lastMidiMs.store (now);
            if (! clock) lastPlayedMs.store (now);
        }
    }

    // Hardware MIDI goes to every armed track; when none is armed it goes to the track selected in the console, so a keyboard always plays something.
    void selectTrack (int i) { if (i >= 0 && i < trackCount.load()) { monitorTrack.store (i); updateTargets(); } }
    void updateTargets()
    {
        bool any = false;
        for (auto& t : tracks) any = any || t->armed.load();
        for (auto& t : tracks) t->recTarget.store (t->armed.load() || (! any && t->index == monitorTrack.load()));
    }

    //==========================================================================
    // message thread API (the bridge calls these)
    void pushLive (TrackState& t, const juce::MidiMessage& m) { LiveMsg lm; if (liveOf (m, juce::Time::getMillisecondCounterHiRes(), lm)) t.liveMidi.push (lm); }
    void noteOn (int track, int note, int vel, int ch = 1) { if (auto* t = trackAt (track)) pushLive (*t, juce::MidiMessage::noteOn (ch, note, (juce::uint8) vel)); }
    void noteOff (int track, int note, int ch = 1)         { if (auto* t = trackAt (track)) pushLive (*t, juce::MidiMessage::noteOff (ch, note)); }
    // PANIC: silence everything. Every track gets 'all sound off' and 'all notes off' on all 16 channels (for a plug-in that ignores one of them), and every plug-in on every track, FX channel, folder and the
    // master is reset on the audio thread before its next block (that ends hung voices and clears delay, reverb and feedback tails). It does not stop the transport and does not change any setting or the song.
    juce::AudioPlayHead* hostPlayHead() { return &playHead; }          // the position every hosted plug-in is told (for the offline checks)
    // The note trace as text: for each track the last notes it sent to its plug-in (beat, on or off, note, velocity), oldest first, with the loop and tempo.
    juce::String noteTraceText (int maxPerTrack = 300)
    {
        juce::String s; s << "Archie DAW note trace. Tempo " << juce::String (transport.bpm.load(), 2) << ", loop " << (transport.looping.load() ? "on " : "off ") << juce::String (transport.loopStart.load(), 3) << " to " << juce::String (transport.loopEnd.load(), 3) << "\n";
        for (int ti = 0; ti < trackCount.load(); ++ti)
        {
            auto& t = *tracks[(size_t) ti]; const int n = t.traceN.load(), from = juce::jmax (0, n - juce::jmin (512, maxPerTrack));
            s << "\nTrack " << (ti + 1) << " " << t.name << "\n"; for (int i = from; i < n; ++i) { const auto& e = t.trace[i & 511]; s << juce::String (e.beat, 4) << (e.on ? "  ON   " : "  off  ") << (int) e.note << "  " << (int) e.vel << "\n"; }
        }
        return s;
    }
    void panicAll()
    {
        for (int ti = 0; ti < trackCount.load(); ++ti)
        {
            auto& t = *tracks[(size_t) ti];
            for (int ch = 1; ch <= 16; ++ch) { pushLive (t, juce::MidiMessage::controllerEvent (ch, 120, 0)); pushLive (t, juce::MidiMessage::allNotesOff (ch)); }
            t.killNotes.store (true);                 // and an explicit note-off for every note (see TrackSourceProcessor::processBlock): the Archie Mini Synth ignored the two controllers and reset() and kept sounding
            for (auto& s : t.slots) s->resetNext.store (true);
        }
        for (auto& r : returns) for (auto& s : r->slots) s->resetNext.store (true);
        for (auto& s : masterChain->slots) s->resetNext.store (true);
    }
    void allNotesOff (int track) { if (auto* t = trackAt (track)) for (int ch = 1; ch <= 16; ++ch) pushLive (*t, juce::MidiMessage::allNotesOff (ch)); }

    TrackState* trackAt (int i) { return (i >= 0 && i < trackCount.load()) ? tracks[(size_t) i].get() : nullptr; }

    // The program starts with kStartTracks tracks; a track (with its own place in the audio graph) is added when asked for, up to kTracks, and any track can be removed (it moves to the end, then goes).
    bool addTrack()
    {
        const int n = trackCount.load();
        if (n >= kTracks) return false;
        auto& t = *tracks[(size_t) n];
        t.name = "Track " + juce::String (n + 1); t.kind.store (0); t.inChan.store (0); t.inStereo.store (true); t.monitor.store (true); t.source.store (0); t.gainDb.store (0.0f); t.pan.store (0.0f);
        clearSends (t);
        t.mute.store (false); t.solo.store (false); t.armed.store (false); t.arrSnap.publish (emptyArrangement()); t.fileSnap.publish (std::make_unique<FileAudio>()); t.filePath = {};
        t.colour.store (kAutoColours[n % (int) (sizeof (kAutoColours) / sizeof (kAutoColours[0]))]);                  // every track has a colour of its own from the start
        ensureTrackNodes (n);
        trackCount.store (n + 1);
        publishPlan(); updateSolo(); updateTargets(); stateDirty.store (true);
        return true;
    }
    bool removeLastTrack()
    {
        const int n = trackCount.load();
        if (n <= 0) return false;
        auto& t = *tracks[(size_t) (n - 1)];
        for (auto& s : t.slots) s->window.reset();
        trackCount.store (n - 1);
        if (monitorTrack.load() >= n - 1) monitorTrack.store (0);
        for (int s = 0; s < kSends; ++s) { t.sendDb[s].store (-100.0f); t.sendLvl[s].store (-60.0f); t.sendOn[s].store (false); t.sendPre[s].store (false); }
        publishPlan(); quiesce();                      // the audio thread no longer reaches this track: only now is it safe to destroy it
        t.slots.clear();
        for (int s = 0; s < kSends; ++s) t.sendProc[s].reset();
        t.sourceProc.reset(); t.stripProc.reset();
        t.armed.store (false); t.mute.store (false); t.solo.store (false);
        updateSolo(); updateTargets(); stateDirty.store (true);
        return true;
    }
    void ensureTrackNodes (int i)
    {
        auto& t = *tracks[(size_t) i];
        if (t.sourceProc == nullptr) { t.sourceProc = std::make_unique<TrackSourceProcessor> (t, blk); prepareProc (*t.sourceProc); }
        if (t.stripProc == nullptr)  { t.stripProc  = std::make_unique<TrackStripProcessor> (t); prepareProc (*t.stripProc); }
    }
    void prepareProc (juce::AudioProcessor& p) { p.setRateAndBufferSizeDetails (sampleRate, blockSize); p.prepareToPlay (sampleRate, blockSize); }
    void prepareAllProcessors()
    {
        for (auto& t : tracks) { if (t->sourceProc) prepareProc (*t->sourceProc); if (t->stripProc) prepareProc (*t->stripProc); for (auto& sp : t->sendProc) if (sp) prepareProc (*sp); for (auto& s : t->slots) prepareProc (*s->proc); }
        for (auto& r : returns) { if (r->stripProc) prepareProc (*r->stripProc); for (auto& s : r->slots) prepareProc (*s->proc); }
        for (auto& s : masterChain->slots) prepareProc (*s->proc);
        if (masterOwner) prepareProc (*masterOwner);
    }
    std::atomic<int> trackCount { kStartTracks };
    std::atomic<int> pdcTotal { 0 };

    // a plug-in chain by number: a track (0..), an FX return (100..107) or the master inserts (200)
    TrackState* chainAt (int id)
    {
        if (id >= 0 && id < trackCount.load()) return tracks[(size_t) id].get();
        if (id >= kReturnBase && id < kReturnBase + kReturns) return returns[(size_t) (id - kReturnBase)].get();
        if (id == kMasterChain) return masterChain.get();
        return nullptr;
    }

    // Sends, as on a console. Each send of a track has its own ON switch, its own level and its PRE/POST choice. Nothing reaches the FX return unless the send is ON. A send exists (as a node in the audio graph)
    // only while it is ON; its level is kept while it is off, so switching it back on returns it to where it was. The level and the switch belong to ONE track's send: no other track is touched.
    float sendEff (const TrackState& t, int s) const { return t.sendOn[s].load() && t.sendLvl[s].load() > -59.5f ? t.sendLvl[s].load() : -100.0f; }
    void applySend (TrackState& t, int s, bool preChanged)
    {
        const bool want = t.sendOn[s].load(), has = t.sendProc[s] != nullptr;
        t.sendDb[s].store (sendEff (t, s));
        if (want && ! has) { t.sendProc[s] = std::make_unique<SendProcessor> (t, s); prepareProc (*t.sendProc[s]); renderer.ensureSendBuffer (t.index, s); publishPlan(); }
        else if (! want && has) { publishPlan(); quiesce(); t.sendProc[s].reset(); }       // the plan without it is live before it is destroyed
        else if (want && has && preChanged) publishPlan();
        stateDirty.store (true);
    }
    void setSendOn (TrackState& t, int s, bool on) { if (s < 0 || s >= kSends) return; t.sendOn[s].store (on); applySend (t, s, false); }
    void setSendLevel (TrackState& t, int s, float db) { if (s < 0 || s >= kSends) return; db = juce::jlimit (-60.0f, 6.0f, db); t.sendLvl[s].store (db <= -59.5f ? -60.0f : db); applySend (t, s, false); }
    void setSendPre (TrackState& t, int s, bool pre) { if (s < 0 || s >= kSends) return; const bool changed = t.sendPre[s].load() != pre; t.sendPre[s].store (pre); applySend (t, s, changed); }
    // the older way of saying it (a level, off at the bottom): a level above off switches the send ON at that level; off switches it off and keeps the level it had
    void setSend (TrackState& t, int s, float db, bool pre)
    {
        if (s < 0 || s >= kSends) return;
        db = juce::jlimit (-100.0f, 6.0f, db);
        const bool nowOn = db > -59.5f, preChanged = t.sendPre[s].load() != pre;
        t.sendPre[s].store (pre); if (nowOn) t.sendLvl[s].store (db); t.sendOn[s].store (nowOn);
        applySend (t, s, preChanged);
    }
    // a song's send (older songs have only a level: above off means on)
    void loadSendVar (TrackState& t, int k, const juce::var& v)
    {
        if (k < 0 || k >= kSends) return;
        const double db = (double) v.getProperty ("db", -100.0); const bool pre = (bool) v.getProperty ("pre", false);
        const bool on = v.hasProperty ("on") ? (bool) v.getProperty ("on", false) : db > -59.5;
        const float lvl = v.hasProperty ("lvl") ? (float) (double) v.getProperty ("lvl", -60.0) : (db > -59.5 ? (float) db : -60.0f);
        t.sendLvl[k].store (juce::jlimit (-60.0f, 6.0f, lvl)); t.sendPre[k].store (pre); t.sendOn[k].store (on); applySend (t, k, true);
    }
    // AN EFFECT PUT ON AN EMPTY FX RETURN MAKES NO SOUND until someone turns a send up: every track's level for that send goes to off (the ON switches stay as they are). Songs that load with an empty return show it off too.
    bool isEmptyReturn (int chain) { if (chain < kReturnBase || chain >= kReturnBase + kReturns) return false; auto* t = chainAt (chain); return t != nullptr && t->slots.empty(); }
    void quietSends (int chain)
    {
        if (chain < kReturnBase || chain >= kReturnBase + kReturns) return; const int s = chain - kReturnBase;
        for (int i = 0; i < trackCount.load(); ++i) { auto& t = *tracks[(size_t) i]; t.sendLvl[s].store (-60.0f); applySend (t, s, false); }
    }
    void clearSends (TrackState& t)
    {
        bool any = false;
        for (int s = 0; s < kSends; ++s) { t.sendDb[s].store (-100.0f); t.sendLvl[s].store (-60.0f); t.sendOn[s].store (false); t.sendPre[s].store (false); any = any || t.sendProc[s] != nullptr; }
        if (any) { publishPlan(); quiesce(); for (int s = 0; s < kSends; ++s) t.sendProc[s].reset(); }
    }

    // Solo: with anything soloed, only the soloed tracks are heard (FX returns are not silenced by it).
    void updateSolo()
    {
        bool any = false;
        for (auto& t : tracks) any = any || t->solo.load();
        for (auto& t : tracks) { t->effMute.store (t->mute.load() || (any && ! t->solo.load())); t->soloGate.store (any && ! t->solo.load()); }
        for (auto& r : returns) r->effMute.store (r->mute.load());
    }

    static constexpr int kPaletteSize = 31;                      // the colours in tools/studio-palette.json
    // ---- standard MIDI files ----
    // Export: the song's note clips as a type-1 file (one track per track that has notes, 480 ticks a quarter note), with the tempo map and time signatures. Each clip instance is written out over its whole
    // length (a clip that repeats is written as many times as it plays), notes cut at the clip's end. Audio clips are not in a MIDI file. Returns the number of tracks written (0 = nothing to write).
    int exportMidiFile (const juce::File& file, juce::String& err)
    {
        juce::MidiFile mf; mf.setTicksPerQuarterNote (480); int written = 0;
        { juce::MidiMessageSequence conductor; const auto& te = transport.tempoNow()->entries();
          for (const auto& t : te) { const double tick = t.beat * 480.0; conductor.addEvent (juce::MidiMessage::tempoMetaEvent ((int) std::llround (60000000.0 / t.bpm)), tick); conductor.addEvent (juce::MidiMessage::timeSignatureMetaEvent (t.tsNum, t.tsDen), tick); }
          conductor.addEvent (juce::MidiMessage::textMetaEvent (3, "Archie DAW"), 0.0); conductor.updateMatchedPairs(); mf.addTrack (conductor); }
        for (int ti = 0; ti < trackCount.load(); ++ti)
        {
            auto& t = *tracks[(size_t) ti]; juce::MidiMessageSequence seq; int notes = 0;
            for (const auto& in : t.arrSnap.load()->insts)
            {
                if (! in.content || in.content->audio || in.content->lengthBeats <= 0.0 || in.content->notes.empty()) continue;
                const double L = in.content->lengthBeats, end = in.length > 0.0 ? in.start + in.length : in.start + L;
                walkInstance (in, in.start, end, false,
                    [&] (const NoteEv& nt, double tb) { seq.addEvent (juce::MidiMessage::noteOn (nt.ch, nt.note, (juce::uint8) nt.vel), tb * 480.0); ++notes; },
                    [&] (const NoteEv& nt, double tb) { seq.addEvent (juce::MidiMessage::noteOff (nt.ch, nt.note), tb * 480.0); },
                    [&] (const NoteEv&) {});
            }
            if (notes == 0) continue;
            seq.addEvent (juce::MidiMessage::textMetaEvent (3, t.name), 0.0); seq.sort(); seq.updateMatchedPairs(); mf.addTrack (seq); ++written;
        }
        if (written == 0) { err = "there are no note clips to write (audio is not part of a MIDI file)"; return 0; }
        file.deleteFile(); auto out = file.createOutputStream(); if (out == nullptr || ! mf.writeTo (*out, 1)) { err = "could not write " + file.getFullPathName(); return 0; }
        return written;
    }
    // Import: each track of the file that has notes becomes a new track here, holding one note clip from the start (its length rounded up to whole bars); the file's tempo and time signature
    // changes become the song's tempo map. Controllers, pitch bend and other messages are not kept (this is notes and tempo). One undo step. Returns the number of tracks made, or 0 with the reason.
    int importMidiFile (const juce::File& file, juce::String& err)
    {
        juce::MidiFile mf; { auto in = file.createInputStream(); if (in == nullptr || ! mf.readFrom (*in)) { err = file.getFileName() + " is not a MIDI file"; return 0; } }
        const int tf = mf.getTimeFormat(); if (tf <= 0) { err = "this MIDI file counts time in frames (SMPTE), which is not supported"; return 0; }
        const double ppq = (double) tf;
        struct Imported { juce::String name; std::vector<NoteEv> notes; double end = 0.0; };
        std::vector<Imported> found; std::vector<TempoEntry> tempo; tempo.push_back (TempoEntry {});
        bool gotTempo = false;
        for (int ti = 0; ti < mf.getNumTracks(); ++ti)
        {
            const auto* seq = mf.getTrack (ti); Imported im; std::map<int, std::pair<double, int>> open;
            for (int i = 0; i < seq->getNumEvents(); ++i)
            {
                const auto& m = seq->getEventPointer (i)->message; const double beat = m.getTimeStamp() / ppq;
                if (m.isTempoMetaEvent()) { const double bpm = 60.0 / juce::jmax (1.0e-6, m.getTempoSecondsPerQuarterNote()); if (beat <= 1.0e-9) { tempo[0].bpm = bpm; gotTempo = true; } else { TempoEntry e2 = tempo.back(); e2.beat = beat; e2.bpm = bpm; tempo.push_back (e2); } }
                else if (m.isTimeSignatureMetaEvent()) { int n = 4, d = 4; m.getTimeSignatureInfo (n, d); if (beat <= 1.0e-9) { tempo[0].tsNum = n; tempo[0].tsDen = d; } else { TempoEntry e2 = tempo.back(); e2.beat = beat; e2.tsNum = n; e2.tsDen = d; tempo.push_back (e2); } }
                else if (m.isTrackNameEvent()) im.name = m.getTextFromTextMetaEvent();
                else if (m.isNoteOn()) open[m.getNoteNumber() + m.getChannel() * 128] = { beat, m.getVelocity() };
                else if (m.isNoteOff() || (m.isNoteOn() && false))
                {
                    auto it = open.find (m.getNoteNumber() + m.getChannel() * 128);
                    if (it != open.end()) { NoteEv n; n.beat = it->second.first; n.len = juce::jmax (0.01, beat - n.beat); n.note = m.getNoteNumber(); n.vel = juce::jmax (1, it->second.second); n.ch = m.getChannel(); im.notes.push_back (n); im.end = juce::jmax (im.end, n.beat + n.len); open.erase (it); }
                }
            }
            for (auto& kv : open) { NoteEv n; n.beat = kv.second.first; n.len = 0.25; n.note = kv.first % 128; n.vel = kv.second.second; n.ch = kv.first / 128; im.notes.push_back (n); im.end = juce::jmax (im.end, n.beat + n.len); }
            if (! im.notes.empty()) found.push_back (std::move (im));
        }
        if (found.empty()) { err = "that MIDI file has no notes"; return 0; }
        const int room = kTracks - trackCount.load();
        if ((int) found.size() > room) { err = "that file has " + juce::String ((int) found.size()) + " tracks of notes and there is room for " + juce::String (room) + " more (the most is " + juce::String (kTracks) + ")"; return 0; }
        // before
        const int nBefore = trackCount.load(); auto tempoBefore = transport.tempoEntries();
        std::vector<std::shared_ptr<Insts>> instsBefore; std::vector<juce::String> namesBefore; for (int i = 0; i < nBefore; ++i) { instsBefore.push_back (instsOf (*tracks[(size_t) i])); namesBefore.push_back (tracks[(size_t) i]->name); }
        // build
        const double bar = (double) tempo[0].tsNum * 4.0 / (double) tempo[0].tsDen; int made = 0;
        if (gotTempo || tempo.size() > 1 || tempo[0].tsNum != 4) transport.setTempoEntries (tempo);
        for (auto& im : found)
        {
            if (! addTrack()) break; const int idx = trackCount.load() - 1; ++made;
            auto& t = *tracks[(size_t) idx]; if (im.name.isNotEmpty()) t.name = im.name.substring (0, 40);
            auto c = std::make_shared<Clip>(); c->id = nextContentId.fetch_add (1); c->lengthBeats = juce::jmax (bar, std::ceil (im.end / bar) * bar); c->notes = im.notes;
            ClipInst in; in.id = nextInstanceId.fetch_add (1); in.start = 0.0; in.length = c->lengthBeats; in.content = c;
            auto ins = std::make_shared<Insts>(); ins->push_back (in); putArr (idx, ins);
        }
        const int nAfter = trackCount.load(); auto tempoAfter = transport.tempoEntries();
        std::vector<std::shared_ptr<Insts>> instsAfter; std::vector<juce::String> namesAfter; for (int i = 0; i < nAfter; ++i) { instsAfter.push_back (instsOf (*tracks[(size_t) i])); namesAfter.push_back (tracks[(size_t) i]->name); }
        undo.record ("Import MIDI file", "",
            [this, nBefore, tempoBefore, instsBefore, namesBefore] { transport.setTempoEntries (tempoBefore); while (trackCount.load() > nBefore) removeLastTrack(); for (int i = 0; i < nBefore; ++i) { putArr (i, instsBefore[(size_t) i]); tracks[(size_t) i]->name = namesBefore[(size_t) i]; } rewarpAll(); stateDirty.store (true); },
            [this, nAfter, tempoAfter, instsAfter, namesAfter] { transport.setTempoEntries (tempoAfter); while (trackCount.load() < nAfter) addTrack(); for (int i = 0; i < nAfter; ++i) { putArr (i, instsAfter[(size_t) i]); tracks[(size_t) i]->name = namesAfter[(size_t) i]; } rewarpAll(); stateDirty.store (true); });
        stateDirty.store (true);
        return made;
    }

    // ---- tap tempo ----
    TapTempo tapper;
    // a tap: once there are three, the song's tempo follows what is being tapped (one undo step per burst of taps). Returns the tempo it now has, or 0 if it is still listening.
    double tapTempo() { const double bpm = tapper.tap (juce::Time::getMillisecondCounterHiRes()); if (bpm > 0.0) editTempo ("Tap tempo", "bpm", [&] { transport.setBpm (bpm); }); return bpm; }

    // ---- automation ----
    std::atomic<int> nextLaneId { 1 }; double lastAutoBeat = -1.0e9; bool lastAutoPlaying = false;
    void putAuto (int chain, const std::shared_ptr<AutoSet>& v) { if (auto* t = chainAt (chain)) { auto a = std::make_unique<AutoSet> (*v); t->autoSnap.publish (std::move (a)); stateDirty.store (true); } }
    juce::var autoVar (int chain)
    {
        juce::Array<juce::var> lanes; auto* t = chainAt (chain);
        if (t) for (const auto& l : t->autoSnap.load()->lanes)
        {
            auto* o = new juce::DynamicObject(); o->setProperty ("id", l.id); o->setProperty ("kind", l.kind); o->setProperty ("slot", l.slot); o->setProperty ("param", l.param); o->setProperty ("on", l.on); o->setProperty ("label", l.label);
            juce::Array<juce::var> pts; for (const auto& p : l.pts) { auto* po = new juce::DynamicObject(); po->setProperty ("beat", p.beat); po->setProperty ("value", p.value); pts.add (juce::var (po)); }
            o->setProperty ("points", pts); lanes.add (juce::var (o));
        }
        return lanes;
    }
    static int kindFromLabel (const juce::String& s)
    {
        static const char* names[] = { "Pattern transpose", "Pattern play", "Mute", "Solo", "Push", "Drive", "Ceiling", "Bass", "Tone", "Archie sound on", "20 Hz cut" };
        for (int i = 0; i < 11; ++i) if (s == names[i]) return 5 + i;
        if (s.startsWith ("Send ") && s.endsWith (" level")) return 4;
        return 3;
    }
    void loadAutoVar (int chain, const juce::var& v)
    {
        auto set = std::make_shared<AutoSet>();
        if (auto* arr = v.getArray())
            for (const auto& lv : *arr)
            {
                AutoLane l; l.id = nextLaneId.fetch_add (1); l.kind = juce::jlimit (0, 16, (int) lv.getProperty ("kind", 0)); l.slot = (int) lv.getProperty ("slot", -1); l.param = (int) lv.getProperty ("param", -1); l.on = (bool) lv.getProperty ("on", true); l.label = lv.getProperty ("label", {}).toString();
                if (l.kind == 3) l.kind = kindFromLabel (l.label);      // (songs saved while lanes were squeezed into 0..3: the name still says what the lane is)
                if (auto* pa = lv.getProperty ("points", {}).getArray()) for (const auto& pv : *pa) l.pts.push_back ({ (double) pv.getProperty ("beat", 0.0), (double) pv.getProperty ("value", 0.0) });
                std::stable_sort (l.pts.begin(), l.pts.end(), [] (const AutoPt& a, const AutoPt& b) { return a.beat < b.beat; });
                set->lanes.push_back (l);
            }
        putAuto (chain, set);
    }
    // the parameters of a hosted plug-in, so a lane can be pointed at one: [{ index, name, value }] (capped at 600)
    juce::var paramsVar (int chain, int slot)
    {
        juce::Array<juce::var> out; auto* t = chainAt (chain);
        if (t == nullptr || slot < 0 || slot >= (int) t->slots.size()) return out;
        const auto& ps = t->slots[(size_t) slot]->proc->getParameters();
        for (int i = 0; i < ps.size() && i < 600; ++i) { auto* o = new juce::DynamicObject(); o->setProperty ("index", i); o->setProperty ("name", ps[i]->getName (48)); o->setProperty ("value", (double) ps[i]->getValue()); out.add (juce::var (o)); }
        return out;
    }
    // add a lane for a target (a second request for the same target gives back the lane it has); returns its id, or 0
    // ---- RECORD AUTOMATION (touch): with it on and the song playing, a move of a Level, Pan or Input gain control, or of a control in a plug-in's own window, is written into that control's lane at the song's position,
    // making the lane if there is none. Plug-in controls are only recorded inside a gesture (the plug-in tells the host when a control is grabbed and let go), so a plug-in moving its own values, or a lane being played back, is
    // never written. While a control is held its lane does not pull it back; ~0.3 s after it is let go the lane takes over again. Points are thinned, and one take is one undo step. Points the take passes over are replaced. ----
    std::atomic<bool> autoRecord { false };
    // TOUCH = LATCH: a control you move while recording is yours from that moment: its lane does not pull it anywhere, and its last position is simply kept (written as the song goes on), until the song stops or loops back to before
    // where you first touched it. There is no timer.
    std::map<std::pair<Slot*, int>, double> latchBeat; std::map<int, double> chainLatch, sendLatch;       // sendLatch: chain * 8 + send
    void latchKey (const std::pair<Slot*, int>& k) { touchHold[k] = 1.0e18; if (latchBeat.find (k) == latchBeat.end()) latchBeat[k] = transport.posBeats.load(); }
    void releaseLatches() { touchHold.clear(); latchBeat.clear(); for (auto& c : chainLatch) if (auto* t = chainAt (c.first)) t->autoTouchUntilMs.store (0.0); chainLatch.clear(); for (auto& c : sendLatch) if (auto* t = chainAt (c.first / 8)) t->autoTouchSendMs[c.first % 8].store (0.0); sendLatch.clear(); }
    int recWatching = 0, recMoves = 0, recWrites = 0; juce::String recTrackName;          // what the recorder sees, shown in the log while recording (so it can be told where a take stops)
    std::map<std::pair<Slot*, int>, float> recSeen;                      // each plug-in control's value as last seen: a change nobody (no lane) made is a hand on it, whichever thread the plug-in reported it from
    bool applyingLanes = false;                                          // true while the lanes themselves are setting plug-in values (message thread): those are never a hand on a control
    struct RecEvt { Slot* s = nullptr; int param = 0, kind = 0; float v = 0.0f; double beat = 0.0; bool human = false; };
    std::array<RecEvt, 1024> recEvts; juce::AbstractFifo recFifo { 1024 }; juce::SpinLock recLock;
    std::map<std::pair<Slot*, int>, double> touchHold;                 // a plug-in control being held (until a time; huge while the gesture lasts)
    struct RecSession { int chain = 0, kind = 0, slot = -1, param = -1, lane = 0, idle = 0; std::shared_ptr<AutoSet> before; double bLast = -1.0e9, lastBeat = -1.0e9, lastValue = 0.0; bool wrote = false; };
    std::vector<RecSession> recSessions;
    void pushRecEvt (Slot* s, int param, float v, int kind)
    {
        if (! transport.recording.load()) return; const juce::SpinLock::ScopedLockType l (recLock); int s1, z1, s2, z2; recFifo.prepareToWrite (1, s1, z1, s2, z2); if (z1 <= 0) return;
        recEvts[(size_t) s1] = { s, param, kind, v, transport.posBeats.load(), ! applyingLanes && juce::MessageManager::existsAndIsCurrentThread() }; recFifo.finishedWrite (1);
    }
    void setAutoRecord (bool on)
    {
        autoRecord.store (on); stateDirty.store (true);
        if (! on) { finishRecSessions (true); touchHold.clear(); for (auto& t : tracks) for (auto& sl : t->slots) sl->touch.reset(); for (auto& r : returns) for (auto& sl : r->slots) sl->touch.reset(); }
    }
    void ensureTouchListeners()
    {
        auto one = [this] (TrackState& t) { for (auto& sl : t.slots) if (sl->touch == nullptr) { Slot* sp = sl.get(); sl->touch = std::make_unique<ParamTouch> (*sl->proc, [this, sp] (int i, float v, int kind) { pushRecEvt (sp, i, v, kind); }); } };
        for (int ti = 0; ti < trackCount.load(); ++ti) one (*tracks[(size_t) ti]); for (auto& r : returns) one (*r);
    }
    static double recTolerance (int kind) { return kind == 0 ? 0.1 : kind == 1 ? 0.01 : kind == 3 || kind == 4 ? 0.1 : 0.004; }
    // one point of a take
    void autoRecWrite (int chain, int kind, int slot, int param, double beat, double value)
    {
        auto* t = chainAt (chain); if (t == nullptr) return;
        const bool byHand = kind == kEqLaneKind || (kind >= 9 && kind <= 15);                // an EQ or Archie sound move is recorded on whichever channel it is made on (the master too): moving it is the choice
        if (! byHand && chain == kMasterChain) return;
        if (! byHand && chain < kReturnBase && ! t->recTarget.load()) return;                // other automation is recorded on a record-armed track, with the song playing and recording
        int laneId = 0; for (const auto& l : t->autoSnap.load()->lanes) if (l.kind == kind && (kind == kEqLaneKind ? l.param == param : ((kind != 2 && kind != 4) || (l.slot == slot && (kind == 4 || l.param == param))))) laneId = l.id;
        if (laneId == 0) laneId = editAutoAddLane (chain, kind, slot, param); if (laneId == 0) return;
        RecSession* ss = nullptr; for (auto& r : recSessions) if (r.chain == chain && r.lane == laneId) ss = &r;
        if (ss == nullptr) { RecSession n; n.chain = chain; n.kind = kind; n.slot = slot; n.param = param; n.lane = laneId; n.before = std::make_shared<AutoSet> (*t->autoSnap.load()); recSessions.push_back (n); ss = &recSessions.back(); }
        ss->idle = 0;
        if (beat < ss->bLast - 1.0e-6) { ss->bLast = beat - 1.0e-6; ss->lastBeat = -1.0e9; }                       // the song looped or was moved back: a new stretch of the same take
        if (ss->lastBeat > -1.0e8 && (std::abs (beat - ss->lastBeat) < 0.03 || (std::abs (beat - ss->lastBeat) < 0.25 && std::abs (value - ss->lastValue) < recTolerance (kind)))) return;      // thinned
        auto set = std::make_shared<AutoSet> (*t->autoSnap.load()); AutoLane* lane = nullptr; for (auto& l : set->lanes) if (l.id == laneId) lane = &l; if (lane == nullptr) return;
        const double lo = ss->bLast < -1.0e8 ? beat - 1.0e-6 : ss->bLast; lane->pts.erase (std::remove_if (lane->pts.begin(), lane->pts.end(), [lo, beat] (const AutoPt& p) { return p.beat > lo && p.beat <= beat; }), lane->pts.end());       // the old line this take passes over
        double v = value; v = lane->kind == 0 ? juce::jlimit (-90.0, 12.0, v) : lane->kind == 1 ? juce::jlimit (-1.0, 1.0, v) : lane->kind == 3 ? juce::jlimit ((double) InputStage::kMinDb, (double) InputStage::kMaxDb, v) : lane->kind == 4 ? juce::jlimit (-60.0, 6.0, v) : lane->kind == 5 ? juce::jlimit (-24.0, 24.0, v) : lane->kind == 6 ? juce::jlimit (0.0, 2.0, v) : lane->kind == 9 ? juce::jlimit (0.0, 18.0, v) : lane->kind == 11 ? juce::jlimit (-3.0, 0.0, v) : lane->kind == 12 ? juce::jlimit (0.0, 6.0, v) : lane->kind == 13 ? juce::jlimit (-3.0, 3.0, v) : (lane->kind == kEqLaneKind && lane->param < 16 && lane->param % 4 == 2) ? juce::jlimit (-18.0, 18.0, v) : juce::jlimit (0.0, 1.0, v);
        lane->pts.push_back ({ juce::jmax (0.0, beat), v }); std::stable_sort (lane->pts.begin(), lane->pts.end(), [] (const AutoPt& a, const AutoPt& b) { return a.beat < b.beat; });
        if (lane->pts.size() > 2000) lane->pts.resize (2000);
        putAuto (chain, set); ++recWrites; stateDirty.store (true); ss->bLast = beat; ss->lastBeat = beat; ss->lastValue = v; ss->wrote = true;
    }
    void finishRecSessions (bool all)
    {
        for (size_t i = 0; i < recSessions.size();)
        {
            auto& ss = recSessions[i];
            if (! all && ++ss.idle <= 12) { ++i; continue; }
            if (ss.wrote) { auto* t = chainAt (ss.chain); if (t != nullptr) { auto before = ss.before; auto after = std::make_shared<AutoSet> (*t->autoSnap.load()); const int ch = ss.chain; undo.record ("Record automation", "", [this, ch, before] { putAuto (ch, before); }, [this, ch, after] { putAuto (ch, after); }); } }
            recSessions.erase (recSessions.begin() + (long) i);
        }
    }
    bool recordingMoves() { return transport.recording.load() && transport.playing.load(); }
    double soundHandUntilMs[16] {};      // an Archie sound control that was just moved while recording: its lane leaves it alone until this time
    // a Level, Pan or Input gain control was moved (by the page): written to its lane, and the lane does not pull it back while it is held
    void autoRecFromControl (int chain, const juce::String& key, double value, int slot = -1)
    {
        const int kind = key == "gainDb" ? 0 : key == "pan" ? 1 : key == "inGainDb" ? 3 : key == "sendLvl" ? 4 : -1; if (kind < 0 || ! transport.recording.load() || ! transport.playing.load()) return;
        auto* t = chainAt (chain); if (t == nullptr || ! t->recTarget.load()) return;
        if (kind == 4) { if (slot < 0 || slot >= kSends) return; t->autoTouchSendMs[slot].store (1.0e18); if (sendLatch.find (chain * 8 + slot) == sendLatch.end()) sendLatch[chain * 8 + slot] = transport.posBeats.load(); autoRecWrite (chain, 4, slot, -1, transport.posBeats.load(), value); return; }
        t->autoTouchUntilMs.store (1.0e18); if (chainLatch.find (chain) == chainLatch.end()) chainLatch[chain] = transport.posBeats.load();
        autoRecWrite (chain, kind, -1, -1, transport.posBeats.load(), value);
    }
    void pollAutoRecord()
    {
        if (! transport.recording.load()) { if (! recSessions.empty()) finishRecSessions (true); releaseLatches(); return; }
        { const double bt = transport.posBeats.load(); for (auto it = latchBeat.begin(); it != latchBeat.end();) { if (bt < it->second - 0.02) { touchHold.erase (it->first); it = latchBeat.erase (it); } else ++it; }
          for (auto it = chainLatch.begin(); it != chainLatch.end();) { if (bt < it->second - 0.02) { if (auto* t = chainAt (it->first)) t->autoTouchUntilMs.store (0.0); it = chainLatch.erase (it); } else ++it; } }
        ensureTouchListeners();
        const double now = juce::Time::getMillisecondCounterHiRes();
        int s1, z1, s2, z2; recFifo.prepareToRead (recFifo.getNumReady(), s1, z1, s2, z2); std::vector<RecEvt> evs; for (int i = 0; i < z1; ++i) evs.push_back (recEvts[(size_t) (s1 + i)]); for (int i = 0; i < z2; ++i) evs.push_back (recEvts[(size_t) (s2 + i)]); recFifo.finishedRead (z1 + z2);
        for (const auto& e : evs)
        {
            const auto key = std::make_pair (e.s, e.param);
            if (e.kind == 1) { latchKey (key); continue; }
            if (e.kind == 2) continue;                                                      // let go: the control keeps its last position (latched)
            if (e.human && transport.playing.load()) latchKey (key);       // a plug-in screen that never reports grab/release: a move from the user's side still counts as a hand on it
            auto it = touchHold.find (key); if (it == touchHold.end() || now > it->second || ! transport.playing.load()) continue;         // not held: the plug-in's own movement, or a lane being played back
            int chain = -1, slot = -1;
            for (int ti = 0; ti < trackCount.load() && chain < 0; ++ti) for (size_t si = 0; si < tracks[(size_t) ti]->slots.size(); ++si) if (tracks[(size_t) ti]->slots[si].get() == e.s) { chain = ti; slot = (int) si; break; }
            for (int r = 0; r < kReturns && chain < 0; ++r) for (size_t si = 0; si < returns[(size_t) r]->slots.size(); ++si) if (returns[(size_t) r]->slots[si].get() == e.s) { chain = kReturnBase + r; slot = (int) si; break; }
            if (chain >= 0) autoRecWrite (chain, 2, slot, e.param, e.beat, (double) e.v);
        }
        // hand moves found by looking: every control of the armed tracks' plug-ins is compared with its last value; a change no lane made is a hand on it (a plug-in screen need not report grabs or moves to the host)
        if (transport.playing.load())
            for (int ti = 0; ti < trackCount.load(); ++ti)
            {
                auto& tr = *tracks[(size_t) ti]; if (! tr.recTarget.load()) continue; recTrackName = tr.name; int watched = 0;
                for (auto& sl : tr.slots)
                {
                    const auto& ps = sl->proc->getParameters(); Slot* sp = sl.get();
                    for (int pi = 0; pi < ps.size(); ++pi, ++watched)
                    {
                        const float v = ps[pi]->getValue(); const auto key = std::make_pair (sp, pi); auto it = recSeen.find (key);
                        if (it == recSeen.end()) { recSeen[key] = v; continue; }
                        if (std::abs (it->second - v) > 1.0e-4f) { it->second = v; if (ps[pi]->isMetaParameter() || ! ps[pi]->isAutomatable()) continue; latchKey (key); ++recMoves; stateDirty.store (true); }
                    }
                }
                if (recWatching != watched) { recWatching = watched; stateDirty.store (true); }
            }
        else { recSeen.clear(); releaseLatches(); }
        // a control that is being held is also SAMPLED every tick, so the lane is drawn continuously even if the plug-in only reports some of its moves
        if (transport.playing.load())
            for (const auto& th : touchHold)
            {
                if (now > th.second) continue; Slot* sp = th.first.first; const int pi = th.first.second; int chain = -1, slot = -1;
                for (int ti = 0; ti < trackCount.load() && chain < 0; ++ti) for (size_t si = 0; si < tracks[(size_t) ti]->slots.size(); ++si) if (tracks[(size_t) ti]->slots[si].get() == sp) { chain = ti; slot = (int) si; break; }
                for (int r = 0; r < kReturns && chain < 0; ++r) for (size_t si = 0; si < returns[(size_t) r]->slots.size(); ++si) if (returns[(size_t) r]->slots[si].get() == sp) { chain = kReturnBase + r; slot = (int) si; break; }
                const auto& ps = sp->proc->getParameters(); if (chain >= 0 && pi >= 0 && pi < ps.size()) autoRecWrite (chain, 2, slot, pi, transport.posBeats.load(), (double) ps[pi]->getValue());
            }
        for (auto it = touchHold.begin(); it != touchHold.end();) { if (now > it->second) it = touchHold.erase (it); else ++it; }
        if (! transport.playing.load()) finishRecSessions (true); else finishRecSessions (false);
    }
    // ---- REMOTE: assign a hardware controller (a knob or slider sending a MIDI controller number) to a control of a plug-in, the way the "Remote" button at the top of a plug-in window does in other programs.
    // Press Remote, move a control in the plug-in (found the same way Learn finds one), then move the knob on your controller: from then on that knob moves that control. The controller message is used up (it is not also played).
    // Assignments are kept with the song, per track. A controller number belongs to one control: assigning it again moves it. ----
    struct RemoteMap { int chain = 0, slot = 0, param = 0, cc = 0; };
    std::vector<RemoteMap> remotes; juce::SpinLock remoteLock;
    struct RemotePend { int chain, slot, param; float v; }; std::vector<RemotePend> remotePend;
    std::atomic<int> remoteStage { 0 }, remoteGotCC { -1 };           // 0 off, 1 waiting for a control to move, 2 waiting for a controller knob
    int remoteChain = -1, remoteSlot = -1, remoteParam = -1; juce::String remoteLabel, remoteDone;
    void startRemote (int chain) { startAutoLearn (chain); learn.remote = true; remoteStage.store (1); remoteDone = {}; stateDirty.store (true); }
    void cancelRemote() { if (learn.remote) learn.chain = -1; remoteStage.store (0); remoteGotCC.store (-1); stateDirty.store (true); }
    void clearRemotes (int chain) { const juce::SpinLock::ScopedLockType l (remoteLock); remotes.erase (std::remove_if (remotes.begin(), remotes.end(), [chain] (const RemoteMap& r) { return r.chain == chain; }), remotes.end()); stateDirty.store (true); }
    // MIDI thread: true when the message is a controller that is assigned (or is being learnt) and so is not played
    // HOT CUE PADS FROM A CONTROLLER: a note or a switch (controller) can be learned for each cue. It is kept on this computer (a controller belongs to the desk, not the song).
    struct CuePad { std::atomic<int> kind { 0 }, channel { 0 }, number { -1 }; };      // kind 0 none, 1 note, 2 controller
    std::array<CuePad, Transport::kCues> cuePads; std::atomic<int> cueLearn { -1 }; std::atomic<bool> cuePadsDirty { false };
    bool cueConsume (const juce::MidiMessage& m)
    {
        const bool note = m.isNoteOn(), off = m.isNoteOff(), cc = m.isController(); if (! (note || off || cc)) return false;
        const int kind = cc ? 2 : 1, ch = m.getChannel(), num = cc ? m.getControllerNumber() : m.getNoteNumber(); const bool press = note || (cc && m.getControllerValue() >= 64);
        const int learn = cueLearn.load();
        if (learn >= 0 && learn < Transport::kCues && press)
        {
            for (auto& p : cuePads) if (p.kind.load() == kind && p.channel.load() == ch && p.number.load() == num) p.kind.store (0);      // one control, one cue
            cuePads[(size_t) learn].kind.store (kind); cuePads[(size_t) learn].channel.store (ch); cuePads[(size_t) learn].number.store (num); cueLearn.store (-1); cuePadsDirty.store (true); return true;
        }
        for (int i = 0; i < Transport::kCues; ++i)
        {
            auto& p = cuePads[(size_t) i]; if (p.kind.load() != kind || p.channel.load() != ch || p.number.load() != num) continue;
            if (press) { if (! transport.cues[(size_t) i].set.load() && i > 0) transport.cueHere (i); else transport.triggerCue (i); }
            return true;      // its release belongs to the pad too: it is never played as a note
        }
        return false;
    }
    juce::String cuePadText (int i) const { const auto& p = cuePads[(size_t) i]; const int k = p.kind.load(); return k == 0 ? juce::String() : (k == 1 ? "Note " : "CC ") + juce::String (p.number.load()) + ", channel " + juce::String (p.channel.load()); }
    void clearCuePad (int i) { if (i >= 0 && i < Transport::kCues) { cuePads[(size_t) i].kind.store (0); cuePadsDirty.store (true); } }
    juce::File cuePadFile() const { return settingsDir.getChildFile (juce::SystemStats::getEnvironmentVariable ("ASKARCHIE_HOST_TEST", {}) == "1" ? "cue-pads-test.json" : "cue-pads.json"); }
    void saveCuePads() { juce::Array<juce::var> a; for (auto& p : cuePads) { auto* o = new juce::DynamicObject(); o->setProperty ("kind", p.kind.load()); o->setProperty ("channel", p.channel.load()); o->setProperty ("number", p.number.load()); a.add (juce::var (o)); } cuePadFile().replaceWithText (juce::JSON::toString (juce::var (a))); }
    void loadCuePads() { const auto v = juce::JSON::parse (cuePadFile().loadFileAsString()); if (auto* a = v.getArray()) for (int i = 0; i < Transport::kCues && i < a->size(); ++i) { cuePads[(size_t) i].kind.store ((int) (*a)[i].getProperty ("kind", 0)); cuePads[(size_t) i].channel.store ((int) (*a)[i].getProperty ("channel", 0)); cuePads[(size_t) i].number.store ((int) (*a)[i].getProperty ("number", -1)); } }
    juce::var cuesVar()
    {
        juce::Array<juce::var> a; const int act = transport.activeCue.load(); const bool pend = transport.loopPending.load();
        for (int i = 0; i < Transport::kCues; ++i)
        {
            auto& c = transport.cues[(size_t) i]; auto* o = new juce::DynamicObject(); const bool set = c.set.load() || i == 0;
            o->setProperty ("set", set); o->setProperty ("start", i == act && pend ? transport.pendStart.load() : c.set.load() ? c.start.load() : transport.loopStart.load()); o->setProperty ("end", i == act && pend ? transport.pendEnd.load() : c.set.load() ? c.end.load() : transport.loopEnd.load()); o->setProperty ("pad", cuePadText (i)); a.add (juce::var (o));
        }
        return juce::var (a);
    }
    bool remoteConsume (const juce::MidiMessage& m)
    {
        if (! m.isController()) return false; const int cc = m.getControllerNumber();
        if (remoteStage.load() == 2 && remoteGotCC.load() < 0) { remoteGotCC.store (cc); return true; }
        const juce::SpinLock::ScopedLockType l (remoteLock); bool used = false;
        for (const auto& r : remotes) if (r.cc == cc) { remotePend.push_back ({ r.chain, r.slot, r.param, (float) m.getControllerValue() / 127.0f }); used = true; }
        if (remotePend.size() > 512) remotePend.erase (remotePend.begin(), remotePend.end() - 256);
        return used;
    }
    void pollRemote()
    {
        if (remoteStage.load() == 2 && remoteGotCC.load() >= 0)
        {
            const int cc = remoteGotCC.load(); { const juce::SpinLock::ScopedLockType l (remoteLock);
              remotes.erase (std::remove_if (remotes.begin(), remotes.end(), [&] (const RemoteMap& r) { return r.cc == cc || (r.chain == remoteChain && r.slot == remoteSlot && r.param == remoteParam); }), remotes.end());
              remotes.push_back ({ remoteChain, remoteSlot, remoteParam, cc }); }
            remoteDone = "Controller " + juce::String (cc) + " now moves " + remoteLabel; remoteStage.store (0); remoteGotCC.store (-1); stateDirty.store (true);
        }
        std::vector<RemotePend> pend; { const juce::SpinLock::ScopedLockType l (remoteLock); pend.swap (remotePend); }
        for (const auto& p : pend)
        {
            auto* t = chainAt (p.chain); if (t == nullptr || p.slot < 0 || p.slot >= (int) t->slots.size()) continue;
            const auto& ps = t->slots[(size_t) p.slot]->proc->getParameters(); if (p.param < 0 || p.param >= ps.size()) continue;
            ps[p.param]->setValueNotifyingHost (p.v);
        }
    }
    int remoteCount (int chain) { const juce::SpinLock::ScopedLockType l (remoteLock); int n = 0; for (const auto& r : remotes) if (r.chain == chain) ++n; return n; }
    juce::var remotesVar (int chain)
    {
        juce::Array<juce::var> a; const juce::SpinLock::ScopedLockType l (remoteLock);
        for (const auto& r : remotes) if (r.chain == chain) { auto* o = new juce::DynamicObject(); o->setProperty ("slot", r.slot); o->setProperty ("param", r.param); o->setProperty ("cc", r.cc); a.add (juce::var (o)); }
        return juce::var (a);
    }
    void loadRemotesVar (int chain, const juce::var& v)
    {
        if (auto* a = v.getArray()) { const juce::SpinLock::ScopedLockType l (remoteLock); for (const auto& x : *a) remotes.push_back ({ chain, (int) x.getProperty ("slot", 0), (int) x.getProperty ("param", 0), (int) x.getProperty ("cc", 0) }); }
    }
    // ---- LEARN a plug-in parameter for automation (finding the parameter in a list of hundreds is hard). Start it on a channel, then move any control in one of its plug-ins' own windows: the first parameter
    // that moves gets its lane made. Read from the 30 Hz tick. Parameters that move by themselves (a meter, a modulated value) are noticed in the first tenth of a second and ignored, only parameters the host may
    // automate count, and it gives up after 20 seconds. Nothing is changed in any plug-in; it only reads their values. ----
    struct AutoLearn { bool remote = false; int chain = -1, tick = 0, foundLane = 0, foundSlot = -1, foundParam = -1; bool found = false, timedOut = false; juce::String foundLabel; std::vector<std::vector<float>> base, base2; std::vector<std::vector<char>> noisy; };
    AutoLearn learn;
    static std::vector<std::vector<float>> readAllParams (TrackState& t)
    {
        std::vector<std::vector<float>> v;
        for (auto& s : t.slots) { std::vector<float> row; const auto& ps = s->proc->getParameters(); for (auto* p : ps) row.push_back (p->getValue()); v.push_back (std::move (row)); }
        return v;
    }
    void startAutoLearn (int chain) { auto* t = chainAt (chain); learn = AutoLearn(); if (t == nullptr || chain == kMasterChain) return; learn.chain = chain; stateDirty.store (true); }
    void stopAutoLearn() { if (learn.chain >= 0) { learn.chain = -1; stateDirty.store (true); } }
    void pollAutoLearn()
    {
        if (learn.chain < 0) return; auto* t = chainAt (learn.chain); if (t == nullptr) { learn.chain = -1; return; }
        if (learn.tick == 0) learn.base = readAllParams (*t);
        else if (learn.tick == 3)
        {
            learn.base2 = readAllParams (*t); learn.noisy.clear();
            for (size_t s = 0; s < learn.base2.size(); ++s) { std::vector<char> n; for (size_t i = 0; i < learn.base2[s].size(); ++i) n.push_back (s < learn.base.size() && i < learn.base[s].size() && std::abs (learn.base2[s][i] - learn.base[s][i]) > 1.0e-4f ? 1 : 0); learn.noisy.push_back (std::move (n)); }
        }
        else if (learn.tick > 3)
        {
            const auto cur = readAllParams (*t);
            for (size_t s = 0; s < cur.size() && s < learn.base2.size() && learn.found == false; ++s)
            {
                const auto& ps = t->slots[s]->proc->getParameters();
                for (size_t i = 0; i < cur[s].size() && i < learn.base2[s].size(); ++i)
                    if (std::abs (cur[s][i] - learn.base2[s][i]) > 0.004f && ! learn.noisy[s][i] && ps[(int) i]->isAutomatable() && ! ps[(int) i]->isMetaParameter())
                    {
                        const int lane = learn.remote ? 1 : editAutoAddLane (learn.chain, 2, (int) s, (int) i);
                        if (learn.remote) { remoteSlot = (int) s; remoteParam = (int) i; remoteChain = learn.chain; remoteLabel = t->slots[s]->desc.name + ": " + ps[(int) i]->getName (40); remoteGotCC.store (-1); remoteStage.store (2); learn.found = true; learn.chain = -1; stateDirty.store (true); break; }
                        if (lane != 0) { learn.found = true; learn.foundLane = lane; learn.foundSlot = (int) s; learn.foundParam = (int) i; learn.foundLabel = t->slots[s]->desc.name + ": " + ps[(int) i]->getName (40); learn.chain = -1; stateDirty.store (true); return; }
                    }
            }
        }
        if (++learn.tick > 650) { learn.timedOut = true; learn.chain = -1; stateDirty.store (true); }
    }
    juce::var autoLearnVar() const
    {
        auto* o = new juce::DynamicObject(); o->setProperty ("remoteStage", remoteStage.load()); o->setProperty ("remoteDone", remoteDone); o->setProperty ("record", transport.recording.load()); o->setProperty ("recWatching", recWatching); o->setProperty ("recMoves", recMoves); o->setProperty ("recWrites", recWrites); o->setProperty ("recTrack", recTrackName); o->setProperty ("on", learn.chain >= 0); o->setProperty ("chain", learn.chain); o->setProperty ("found", learn.found); o->setProperty ("timedOut", learn.timedOut);
        if (learn.found) { o->setProperty ("lane", learn.foundLane); o->setProperty ("slot", learn.foundSlot); o->setProperty ("param", learn.foundParam); o->setProperty ("label", learn.foundLabel); }
        return juce::var (o);
    }
    int editAutoAddLane (int chain, int kind, int slot, int param)
    {
        auto* t = chainAt (chain); if (t == nullptr) return 0; const bool isMaster = chain == kMasterChain;
        const bool isEq = kind == kEqLaneKind; if (isEq && (param < 0 || param >= kEqLaneParams || (! isMaster && (chain < 0 || chain >= kReturnBase)))) return 0;
        if (! isEq && (isMaster ? (kind < 9 || kind > 15) : (kind < 0 || kind > 8 || (kind >= 3 && (chain < 0 || chain >= kReturnBase)) || (kind == 4 && (slot < 0 || slot >= kSends))))) return 0;      // kinds 3 (input gain) and 4 (a send's level) belong to tracks only; 9 to 15 (the Archie sound) belong to the master only
        juce::String label = kind == 0 ? "Level" : kind == 1 ? "Pan" : kind == 3 ? "Input gain" : kind == 4 ? "Send " + juce::String (slot + 1) + " level" : kind == 5 ? "Pattern transpose" : kind == 6 ? "Pattern play" : kind == 7 ? "Mute" : kind == 8 ? "Solo" : kind == 9 ? "Push" : kind == 10 ? "Drive" : kind == 11 ? "Ceiling" : kind == 12 ? "Bass" : kind == 13 ? "Tone" : kind == 14 ? "Archie sound on" : kind == 15 ? "20 Hz cut" : "";
        if (isEq) { static const char* what[] = { " on", " frequency", " gain", " Q" }; label = param < 16 ? "EQ band " + juce::String (param / 4 + 1) + what[param % 4] : param == 16 ? "EQ high-pass on" : param == 17 ? "EQ high-pass frequency" : param == 18 ? "EQ low-pass on" : "EQ low-pass frequency"; }
        if (kind == 2)
        {
            if (slot < 0 || slot >= (int) t->slots.size()) return 0; const auto& ps = t->slots[(size_t) slot]->proc->getParameters(); if (param < 0 || param >= ps.size()) return 0;
            label = t->slots[(size_t) slot]->desc.name + ": " + ps[param]->getName (40);
        }
        auto cur = std::make_shared<AutoSet> (*t->autoSnap.load());
        for (const auto& l : cur->lanes) if (l.kind == kind && (kind == kEqLaneKind ? l.param == param : ((kind != 2 && kind != 4) || (l.slot == slot && (kind == 4 || l.param == param))))) return l.id;
        auto set = std::make_shared<AutoSet> (*cur); AutoLane l; l.id = nextLaneId.fetch_add (1); l.kind = kind; l.slot = (kind == 2 || kind == 4) ? slot : -1; l.param = (kind == 2 || kind == kEqLaneKind) ? param : -1; l.label = label; set->lanes.push_back (l);
        putAuto (chain, set);
        undo.record ("Add automation lane", "", [this, chain, cur] { putAuto (chain, cur); }, [this, chain, set] { putAuto (chain, set); });
        return l.id;
    }
    // replace a lane's points (sorted, values held to the lane's range); a drag is one undo step
    bool editAutoSetPoints (int chain, int laneId, const juce::var& pts)
    {
        auto* t = chainAt (chain); if (t == nullptr) return false;
        auto cur = std::make_shared<AutoSet> (*t->autoSnap.load()); auto set = std::make_shared<AutoSet> (*cur); AutoLane* lane = nullptr;
        for (auto& l : set->lanes) if (l.id == laneId) lane = &l;
        if (lane == nullptr) return false;
        lane->pts.clear();
        if (auto* arr = pts.getArray())
            for (const auto& pv : *arr)
            {
                double v = (double) pv.getProperty ("value", 0.0); v = lane->kind == 0 ? juce::jlimit (-90.0, 12.0, v) : lane->kind == 1 ? juce::jlimit (-1.0, 1.0, v) : lane->kind == 3 ? juce::jlimit ((double) InputStage::kMinDb, (double) InputStage::kMaxDb, v) : lane->kind == 4 ? juce::jlimit (-60.0, 6.0, v) : lane->kind == 5 ? juce::jlimit (-24.0, 24.0, v) : lane->kind == 6 ? juce::jlimit (0.0, 2.0, v) : lane->kind == 9 ? juce::jlimit (0.0, 18.0, v) : lane->kind == 11 ? juce::jlimit (-3.0, 0.0, v) : lane->kind == 12 ? juce::jlimit (0.0, 6.0, v) : lane->kind == 13 ? juce::jlimit (-3.0, 3.0, v) : (lane->kind == kEqLaneKind && lane->param < 16 && lane->param % 4 == 2) ? juce::jlimit (-18.0, 18.0, v) : juce::jlimit (0.0, 1.0, v);
                lane->pts.push_back ({ juce::jmax (0.0, (double) pv.getProperty ("beat", 0.0)), v });
            }
        std::stable_sort (lane->pts.begin(), lane->pts.end(), [] (const AutoPt& a, const AutoPt& b) { return a.beat < b.beat; });
        if (lane->pts.size() > 2000) lane->pts.resize (2000);
        putAuto (chain, set);
        undo.record ("Edit automation", "auto:" + std::to_string (chain) + ":" + std::to_string (laneId), [this, chain, cur] { putAuto (chain, cur); }, [this, chain, set] { putAuto (chain, set); });
        return true;
    }
    bool editAutoLane (int chain, int laneId, const juce::String& op, bool on)
    {
        auto* t = chainAt (chain); if (t == nullptr) return false;
        auto cur = std::make_shared<AutoSet> (*t->autoSnap.load()); auto set = std::make_shared<AutoSet> (*cur); bool found = false;
        for (size_t i = 0; i < set->lanes.size(); ++i) if (set->lanes[i].id == laneId) { found = true; if (op == "remove") set->lanes.erase (set->lanes.begin() + (long) i); else set->lanes[i].on = on; break; }
        if (! found) return false;
        putAuto (chain, set); undo.record (op == "remove" ? "Remove automation lane" : "Switch automation", "", [this, chain, cur] { putAuto (chain, cur); }, [this, chain, set] { putAuto (chain, set); });
        return true;
    }
    // plug-in parameters follow their lanes from the 30 Hz tick: while the song plays, and whenever the playhead is moved
    void applyParamAutomation()
    {
        const double beat = transport.posBeats.load(); const bool playing = transport.playing.load();
        if (! playing && ! lastAutoPlaying && std::abs (beat - lastAutoBeat) < 1.0e-9) return;
        lastAutoBeat = beat; lastAutoPlaying = playing;
        struct Flag { bool& b; Flag (bool& x) : b (x) { b = true; } ~Flag() { b = false; } } applying (applyingLanes);
        auto doChain = [&] (TrackState& t)
        {
            for (const auto& lane : t.autoSnap.load()->lanes)
            {
                if (lane.kind >= 9 && lane.kind <= 15 && lane.on && ! lane.pts.empty())      // the Archie sound's lanes on the master: the control follows the lane
                {
                    if (juce::Time::getMillisecondCounterHiRes() < soundHandUntilMs[lane.kind]) continue;      // a hand is on it
                    const double v = autoValueAt (lane, beat);
                    switch (lane.kind) { case 9: if (master != nullptr && master->amTarget.load() < -1.0f) break; applyMasterSoundParam ("amPush", v); break;      // (a loudness target moves the push itself, so a push lane waits)
                        case 10: applyMasterSoundParam ("amDrive", v); break; case 11: applyMasterSoundParam ("amCeil", v); break; case 12: applyMasterSoundParam ("amBass", v); break; case 13: applyMasterSoundParam ("amTone", v); break;
                        case 14: applyMasterSoundParam ("amOn", v >= 0.5); break; default: applyMasterSoundParam ("amCut", v >= 0.5); break; }
                    continue;
                }
                if (lane.kind == 8 && lane.on && ! lane.pts.empty()) { const bool want = autoValueAt (lane, beat) >= 0.5; if (t.solo.load() != want) { t.solo.store (want); updateSolo(); } continue; }      // a SOLO lane
                if (lane.kind != 2 || ! lane.on || lane.pts.empty() || lane.slot < 0 || lane.slot >= (int) t.slots.size()) continue;
                const auto& ps = t.slots[(size_t) lane.slot]->proc->getParameters(); if (lane.param < 0 || lane.param >= ps.size()) continue;
                { const auto th = touchHold.find (std::make_pair (t.slots[(size_t) lane.slot].get(), lane.param)); if (th != touchHold.end() && juce::Time::getMillisecondCounterHiRes() < th->second) continue; }       // being held, and recorded
                const float v = (float) juce::jlimit (0.0, 1.0, autoValueAt (lane, beat)); if (std::abs (ps[lane.param]->getValue() - v) > 1.0e-4f) ps[lane.param]->setValue (v); recSeen[std::make_pair (t.slots[(size_t) lane.slot].get(), lane.param)] = ps[lane.param]->getValue();
            }
        };
        for (int ti = 0; ti < trackCount.load(); ++ti) doChain (*tracks[(size_t) ti]);
        for (auto& r : returns) doChain (*r);
        if (masterChain != nullptr) doChain (*masterChain);
    }

    // ---- warp: making audio clips follow the tempo ----
    // renderWarp is slow (a second or more for a long sound), so it runs on a background thread and the finished sound is attached to the clip when it is ready (applyWarpResults, from the 30 Hz tick).
    // Whatever edit changes what a warped clip must sound like (its place, length, offset, source tempo, the tempo map, the sample rate) changes its warpKey, which drops the old render and asks for a new one.
    struct WarpResult { int instId; uint64_t key; std::shared_ptr<AudioData> data; bool pitchOnly = false; };
    std::mutex warpMutex; std::vector<WarpResult> warpDone; std::set<uint64_t> warpInFlight; std::atomic<bool> warpCancel { false };
    juce::ThreadPool warpPool { 1 };                                // (after the mutex and the list it reports into, so it is joined first)
    uint64_t warpKeyFor (const ClipInst& in, const std::vector<TempoEntry>& tempo) const
    {
        juce::String s = juce::String::formatted ("%d|%.9f|%.9f|%.9f|%.6f|%d|%d|%.1f|t%.4f:%d|", in.content ? in.content->id : 0, in.start, in.length, in.offset, in.srcBpm, in.reverse ? 1 : 0, in.repitch ? 1 : 0, sampleRate, clipSemis (in), in.keepFormant ? 1 : 0);
        for (const auto& e : tempo) s << juce::String::formatted ("%.6f:%.6f;", e.beat, e.bpm);
        if (! in.reverse) for (const auto& m : in.markers) s << juce::String::formatted ("m%.6f:%.6f;", m.sec, m.beat);
        return (uint64_t) s.hashCode64();
    }
    uint64_t pitchKeyFor (const ClipInst& in) const { return (uint64_t) juce::String::formatted ("p%d|%.4f|%d|%.1f", in.content ? in.content->id : 0, clipSemis (in), in.keepFormant ? 1 : 0, sampleRate).hashCode64(); }
    void schedulePitch (const ClipInst& in, uint64_t key)
    {
        if (! warpInFlight.insert (key).second) return;
        auto src = in.content->audio; const double semis = clipSemis (in); const bool keep = in.keepFormant; const int id = in.id;
        warpPool.addJob ([this, src, semis, keep, key, id]
        {
            auto out = retunePitch (*src, [semis] (double) { return semis; }, keep, &warpCancel);
            const std::lock_guard<std::mutex> l (warpMutex); warpDone.push_back ({ id, key, out, true });
        });
    }
    void scheduleWarp (const ClipInst& in, uint64_t key, const std::vector<TempoEntry>& tempo)
    {
        if (! warpInFlight.insert (key).second) return;
        WarpParams wp; wp.startBeat = in.start; wp.lengthBeats = in.length; wp.offsetSec = in.offset; wp.srcBpm = in.srcBpm; wp.reverse = in.reverse; wp.repitch = in.repitch; wp.markers = in.markers;
        auto src = in.content->audio; const double rate = sampleRate > 0.0 ? sampleRate : 48000.0; const int id = in.id; const double semis = clipSemis (in); const bool keep = in.keepFormant;
        warpPool.addJob ([this, src, wp, tempo, rate, key, id, semis, keep]
        {
            TempoMap tm (tempo);
            auto out = renderWarp (*src, wp, tm, rate, &warpCancel);
            if (out != nullptr && semis != 0.0) out = retunePitch (*out, [semis] (double) { return semis; }, keep, &warpCancel);       // the stretched sound, then moved in pitch
            const std::lock_guard<std::mutex> l (warpMutex); warpDone.push_back ({ id, key, out });
        });
    }
    // called whenever an arrangement is published: every warped clip whose render is missing or out of date drops it and has a new one made
    void refreshWarps (std::vector<ClipInst>& v)
    {
        const auto tempo = transport.tempoEntries();
        for (auto& in : v)
        {
            // a clip that is not warped but is transposed plays its whole source moved in pitch (the render does not depend on the tempo)
            if (in.content && in.content->audio && clipSemis (in) != 0.0 && ! (in.warp && in.srcBpm > 0.0 && in.length > 0.0))
            {
                const auto pk = pitchKeyFor (in); if (in.pitched == nullptr || in.pitchKey != pk) { in.pitched = nullptr; in.pitchKey = 0; schedulePitch (in, pk); }
            }
            else { in.pitched = nullptr; in.pitchKey = 0; }
            if (! in.warp || ! in.content || ! in.content->audio || in.srcBpm <= 0.0 || in.length <= 0.0) { in.warped = nullptr; in.warpKey = 0; continue; }
            const auto key = warpKeyFor (in, tempo);
            if (in.warped != nullptr && in.warpKey == key) continue;
            in.warped = nullptr; in.warpKey = 0; scheduleWarp (in, key, tempo);
        }
    }
    void applyWarpResults()
    {
        std::vector<WarpResult> done; { const std::lock_guard<std::mutex> l (warpMutex); done.swap (warpDone); }
        if (done.empty()) return;
        const auto tempo = transport.tempoEntries();
        for (auto& r : done)
        {
            warpInFlight.erase (r.key);
            if (r.data == nullptr) continue;
            int idx = -1; const int track = trackOfInstance (r.instId, &idx); if (track < 0) continue;
            auto ins = instsOf (*tracks[(size_t) track]); auto& in = (*ins)[(size_t) idx];
            if (r.pitchOnly) { if (in.content && in.content->audio && clipSemis (in) != 0.0 && pitchKeyFor (in) == r.key) { in.pitched = r.data; in.pitchKey = r.key; putArr (track, ins); stateDirty.store (true); } continue; }
            if (! in.warp || ! in.content || ! in.content->audio || warpKeyFor (in, tempo) != r.key) continue;         // changed again while it was being made
            in.warped = r.data; in.warpKey = r.key; putArr (track, ins); stateDirty.store (true);
        }
    }
    bool warpBusy() const { return ! warpInFlight.empty(); }
    void rewarpAll() { for (int ti = 0; ti < trackCount.load(); ++ti) { bool any = false; for (const auto& in : tracks[(size_t) ti]->arrSnap.load()->insts) any = any || in.warp; if (any) putArr (ti, instsOf (*tracks[(size_t) ti])); } }
    // for tests and for a project load that wants to hear warped clips at once: wait for every render in flight (message thread)
    void waitForWarp (int timeoutMs = 20000) { const auto end = juce::Time::getMillisecondCounter() + (juce::uint32) timeoutMs; while (juce::Time::getMillisecondCounter() < end) { applyWarpResults(); if (! warpBusy()) break; juce::Thread::sleep (5); } applyWarpResults(); }

    // ---- audio clips ----
    juce::StringArray loadWarnings;                                    // things a project load could not bring back (a sound file that has moved)
    // Reads a sound file into memory (any format JUCE reads: wav, aiff, flac, mp3, ogg), keeps it at its own sample rate, and works out the waveform's peaks. Message thread.
    std::shared_ptr<AudioData> loadAudioData (const juce::File& f, juce::String& err)
    {
        if (! f.existsAsFile()) { err = "the file " + f.getFileName() + " is not there"; return nullptr; }
        juce::AudioFormatManager fm; fm.registerBasicFormats();
        std::unique_ptr<juce::AudioFormatReader> r (fm.createReaderFor (f));
        if (r == nullptr || r->lengthInSamples <= 0) { err = f.getFileName() + " could not be read as sound"; return nullptr; }
        if (r->lengthInSamples > 60LL * 60 * 48000) { err = f.getFileName() + " is longer than an hour"; return nullptr; }
        auto a = std::make_shared<AudioData>();
        a->rate = r->sampleRate > 0.0 ? r->sampleRate : 48000.0; a->path = f.getFullPathName(); a->name = f.getFileNameWithoutExtension();
        a->buf.setSize ((int) juce::jmin<unsigned int> (2, r->numChannels), (int) r->lengthInSamples);
        r->read (&a->buf, 0, (int) r->lengthInSamples, 0, true, true);
        const int n = a->buf.getNumSamples(), B = AudioData::kPeakBlock;
        a->peaks.assign ((size_t) ((n + B - 1) / B), 0.0f);
        for (int c = 0; c < a->buf.getNumChannels(); ++c)
        {
            const float* x = a->buf.getReadPointer (c);
            for (int i = 0; i < n; ++i) { float& p = a->peaks[(size_t) (i / B)]; p = juce::jmax (p, std::abs (x[i])); }
        }
        return a;
    }
    // the waveform as `bins` numbers between 0 and 1 (the loudest point in each stretch of the sound)
    // ---- where things are in a warped (or plain) audio clip ----
    static WarpMap warpMapOf (const ClipInst& a) { return makeWarpMap (a.markers, a.srcBpm > 0.0 ? a.srcBpm : 120.0, a.reverse); }
    // the position in the sound (seconds, 0..length) that is heard at timeline beat `atBeat` in this audio clip
    double audioPosAt (const ClipInst& a, double atBeat) const
    {
        const double L = a.content->audio->durationSec(); if (L <= 0.0) return 0.0;
        double s;
        if (a.warp && a.srcBpm > 0.0) { const auto m = warpMapOf (a); s = warpSecAt (m, (atBeat - a.start) + m.f (a.offset), L); }
        else { const auto* tm = transport.tempoNow(); s = a.offset + tm->secondsAt (atBeat) - tm->secondsAt (a.start); }
        s = std::fmod (s, L); return s < 0.0 ? s + L : s;
    }
    // where the hits of this clip's sound are heard on the timeline (timeline beats, with the second in the sound each one is), within the clip, in order
    std::vector<std::pair<double, double>> hitBeats (const ClipInst& a, double sensitivity) const
    {
        std::vector<std::pair<double, double>> out;
        if (! a.content || ! a.content->audio || a.length <= 0.0) return out;
        const double L = a.content->audio->durationSec(), end = a.start + a.length; if (L <= 0.0) return out;
        const auto* tm = transport.tempoNow(); const double ws = tm->secondsAt (a.start), we = tm->secondsAt (end);
        const bool warped = a.warp && a.srcBpm > 0.0; const auto m = warpMapOf (a); const double F0 = m.f (a.offset);
        for (double tau : detectTransients (*a.content->audio, juce::jlimit (0.0, 1.0, sensitivity)))
            for (int k = -1; k < 100000; ++k)
            {
                double bt;
                if (warped && ! a.reverse) bt = a.start + warpBeatAt (m, tau + (double) k * L, L) - F0;
                else
                {
                    const double p = a.reverse ? L - tau : tau, t = p - a.offset + (double) k * L;                 // heard t seconds after the clip starts (a warped, reversed clip runs at its source tempo)
                    bt = warped ? a.start + t * a.srcBpm / 60.0 : tm->beatAt (ws + t);
                    if (t < 0.0) continue;
                    if (! warped && ws + t >= we) break;
                }
                if (bt < a.start - 1.0e-9) continue;
                if (bt >= end - 1.0e-9) break;
                out.push_back ({ tau, bt });
            }
        std::sort (out.begin(), out.end(), [] (const auto& x, const auto& y) { return x.second < y.second; });
        return out;
    }
    // edit a warped clip's markers: "set" pins the moment `sec` of the sound to timeline beat `atBeat` (moving the marker already near it, else adding one), "remove" drops marker `index`, "clear" drops all.
    // The clip's front (what is at its start) stays where it is. A marker that would make the sound run backwards, or stretch a stretch of it more than 4x or less than 1/4, is refused. Returns "" or why not.
    juce::String editWarpMarker (int instId, const juce::String& op, double sec, double atBeat, int index)
    {
        int idx = -1; const int track = trackOfInstance (instId, &idx); if (track < 0) return "no such clip";
        auto before = instsOf (*tracks[(size_t) track]); auto ins = std::make_shared<Insts> (*before); auto& a = (*ins)[(size_t) idx];
        if (! a.content || ! a.content->audio) return "markers are for sound clips";
        if (a.srcBpm <= 0.0) return "set the tempo the sound was made at first";
        const double L = a.content->audio->durationSec(); const auto oldMap = warpMapOf (a); const double keepF = oldMap.f (a.offset);
        std::vector<WarpMarker> pts;
        if (op == "clear") { if (a.markers.empty()) return {}; }
        else if (op == "remove") { if (index < 0 || index >= (int) a.markers.size()) return "no such marker"; pts = a.markers; pts.erase (pts.begin() + index); }
        else if (op == "set")
        {
            if (a.reverse) return "markers do not apply to a reversed clip";
            if (sec < 0.0 || sec > L) return "that is outside the sound";
            pts = a.markers; const double F = (atBeat - a.start) + keepF;
            if (pts.empty() && std::abs (sec - a.offset) > 0.006) pts.push_back ({ a.offset, keepF });          // the first marker also pins the front of the clip, so the stretch happens between them and not before
            int hit = -1; for (size_t i = 0; i < pts.size(); ++i) if (std::abs (pts[i].sec - sec) < 0.006) hit = (int) i;
            if (hit >= 0) pts[(size_t) hit].beat = F; else pts.push_back ({ sec, F });
            if (pts.size() > 128) return "that is the most markers there can be (128)";
        }
        else return "unknown marker operation";
        std::sort (pts.begin(), pts.end(), [] (const WarpMarker& x, const WarpMarker& y) { return x.sec < y.sec; });
        const double slope = a.srcBpm / 60.0;
        for (size_t i = 0; i < pts.size(); ++i)
        {
            if (i == 0) continue;
            const double ds = pts[i].sec - pts[i - 1].sec, dF = pts[i].beat - pts[i - 1].beat;
            if (ds < 0.001 || dF < 0.01) return "markers must stay in order: that would make the sound run backwards";
            const double ratio = (dF / ds) / slope; if (ratio < 0.25 || ratio > 4.0) return "that would stretch part of the sound more than 4 times (or less than a quarter)";
        }
        a.markers = pts;
        if (! pts.empty()) { const auto nm = warpMapOf (a); const double d = keepF - nm.f (a.offset); for (auto& m : a.markers) m.beat += d; }       // the clip's front stays where it was
        putArr (track, ins); recordArrChange (track, op == "clear" ? "Clear warp markers" : op == "remove" ? "Remove warp marker" : "Move warp marker", before, "wm:" + std::to_string (instId));
        return {};
    }
    // a guess at the tempo the sound was made at, from its hits (nothing is changed); { bpm, confidence }
    juce::var detectTempoVar (int contentId)
    {
        auto c = findContent (contentId); auto* o = new juce::DynamicObject();
        if (c && c->audio) { const auto g = detectTempo (*c->audio); o->setProperty ("bpm", g.bpm); o->setProperty ("confidence", g.confidence); } else { o->setProperty ("bpm", 0.0); o->setProperty ("confidence", 0.0); }
        return juce::var (o);
    }
    // the starts the transient detector hears in a sound, in seconds from its start
    // the hits of a clip as heard on ITS timeline (with warp and markers applied): [{ sec, beat }]
    juce::var hitBeatsVar (int instId, double sensitivity)
    {
        juce::Array<juce::var> out; int idx = -1; const int track = trackOfInstance (instId, &idx); if (track < 0) return out;
        for (const auto& h : hitBeats (tracks[(size_t) track]->arrSnap.load()->insts[(size_t) idx], sensitivity)) { auto* o = new juce::DynamicObject(); o->setProperty ("sec", h.first); o->setProperty ("beat", h.second); out.add (juce::var (o)); }
        return out;
    }
    juce::var transientsVar (int contentId, double sensitivity)
    {
        auto c = findContent (contentId); juce::Array<juce::var> out;
        if (c == nullptr || ! c->audio) return out;
        for (double t : detectTransients (*c->audio, sensitivity)) out.add (t);
        return out;
    }
    juce::var waveformVar (int contentId, int bins)
    {
        auto c = findContent (contentId); juce::Array<juce::var> out;
        if (c == nullptr || ! c->audio) return out;
        const auto& pk = c->audio->peaks; bins = juce::jlimit (1, 4000, bins);
        for (int b = 0; b < bins; ++b)
        {
            const size_t i0 = (size_t) ((double) b * (double) pk.size() / bins), i1 = juce::jmax (i0 + 1, (size_t) ((double) (b + 1) * (double) pk.size() / bins));
            float m = 0.0f; for (size_t i = i0; i < i1 && i < pk.size(); ++i) m = juce::jmax (m, pk[i]);
            out.add (juce::jmin (1.0f, m));
        }
        return out;
    }

    // ---- clips as instances of shared content ----
    // A track's clips are an arrangement: instances (a place on the timeline) that point at shared content (the notes). The first versions had ONE pattern that looped for ever;
    // that is now simply one instance with start 0 and length 0 (= for ever), so setClip / getClip / clearClip keep working for the pages that still use them.
    using Insts = std::vector<ClipInst>;
    std::atomic<int> nextContentId { 1 }, nextInstanceId { 1 };

    std::shared_ptr<Clip> contentFromVar (const juce::var& v)
    {
        auto c = std::make_shared<Clip>();
        c->id = nextContentId.fetch_add (1);
        c->lengthBeats = juce::jlimit (0.25, 512.0, (double) v.getProperty ("lengthBeats", 4.0));
        { const int pb = (int) v.getProperty ("patternBank", -1); c->patternBank = pb >= 0 ? juce::jmin (pb, 63) : -1; }          // a pattern block
        if (auto* arr = v.getProperty ("notes", {}).getArray())
            for (const auto& n : *arr)
            {
                NoteEv e; e.beat = (double) n.getProperty ("beat", 0.0); e.len = juce::jmax (0.01, (double) n.getProperty ("len", 0.25));
                e.note = juce::jlimit (0, 127, (int) n.getProperty ("note", 60)); e.vel = juce::jlimit (1, 127, (int) n.getProperty ("vel", 100)); e.ch = juce::jlimit (1, 16, (int) n.getProperty ("ch", 1));
                if (e.beat >= 0 && e.beat < c->lengthBeats) c->notes.push_back (e);
            }
        return c;
    }
    std::shared_ptr<Insts> instsOf (TrackState& t) { return std::make_shared<Insts> (t.arrSnap.load()->insts); }
    void putArr (int track, const std::shared_ptr<Insts>& v)
    {
        auto* t = trackAt (track); if (t == nullptr) return;
        auto a = std::make_unique<Arrangement>(); a->insts = *v;
        std::stable_sort (a->insts.begin(), a->insts.end(), [] (const ClipInst& x, const ClipInst& y) { return x.start < y.start; });
        refreshWarps (a->insts);
        t->arrSnap.publish (std::move (a)); stateDirty.store (true);
    }
    void setClip (int track, const juce::var& v)
    {
        auto* t = trackAt (track);
        if (t == nullptr) return;
        auto ins = std::make_shared<Insts>();
        ClipInst in; in.id = nextInstanceId.fetch_add (1); in.content = contentFromVar (v);
        ins->push_back (in);
        putArr (track, ins);
    }

    static juce::var clipVar (const Clip& c)
    {
        auto* o = new juce::DynamicObject();
        juce::Array<juce::var> notes;
        o->setProperty ("lengthBeats", c.lengthBeats); o->setProperty ("id", c.id); if (c.patternBank >= 0) o->setProperty ("patternBank", c.patternBank);
        if (c.audio) { o->setProperty ("audio", true); o->setProperty ("path", c.audio->path); o->setProperty ("name", c.audio->name); o->setProperty ("durationSec", c.audio->durationSec()); o->setProperty ("rate", c.audio->rate); }
        for (const auto& n : c.notes) { auto* e = new juce::DynamicObject(); e->setProperty ("beat", n.beat); e->setProperty ("len", n.len); e->setProperty ("note", n.note); e->setProperty ("vel", n.vel); e->setProperty ("ch", n.ch); notes.add (juce::var (e)); }
        o->setProperty ("notes", notes);
        return juce::var (o);
    }
    juce::var clipToVar (const TrackState& t) const { return clipVar (t.primaryClip()); }       // the first instance's content: "the pattern" of the single-pattern pages
    // every instance of a track and the content they point at (content shared by several instances appears once)
    juce::var arrangementVar (const TrackState& t) const
    {
        auto* o = new juce::DynamicObject(); juce::Array<juce::var> insts, contents; std::set<int> seen;
        o->setProperty ("track", t.index);
        for (const auto& in : t.arrSnap.load()->insts)
        {
            if (in.content == nullptr) continue;
            auto* i = new juce::DynamicObject();
            i->setProperty ("id", in.id); i->setProperty ("start", in.start); i->setProperty ("length", in.length); i->setProperty ("offset", in.offset);
            i->setProperty ("colour", in.colour); i->setProperty ("muted", in.muted); i->setProperty ("label", in.label); i->setProperty ("groove", in.groove); i->setProperty ("contentId", in.content->id); i->setProperty ("patternBank", in.content->patternBank); i->setProperty ("contentLength", in.content->lengthBeats); i->setProperty ("noteCount", (int) in.content->notes.size());
            if (! in.content->audio)       // the notes of the clip as [beat, length, note] (up to 500), so the track view can draw them inside the clip as Reason does
            {
                juce::Array<juce::var> ns; int cnt = 0;
                for (const auto& n : in.content->notes) { if (++cnt > 500) break; juce::Array<juce::var> a; a.add (n.beat); a.add (n.len); a.add (n.note); ns.add (juce::var (a)); }
                i->setProperty ("notes", ns);
            }
            if (in.content->audio)
            {
                const auto* tm = transport.tempoNow();
                { juce::Array<juce::var> ma; if (in.warp && ! in.reverse) { const auto mm = warpMapOf (in); const double F0 = mm.f (in.offset); for (const auto& k : in.markers) { auto* mo = new juce::DynamicObject(); mo->setProperty ("sec", k.sec); mo->setProperty ("beat", in.start + k.beat - F0); ma.add (juce::var (mo)); } } i->setProperty ("markers", ma); }
                { juce::Array<juce::var> mr; for (const auto& k : in.markers) { auto* mo = new juce::DynamicObject(); mo->setProperty ("sec", k.sec); mo->setProperty ("beat", k.beat); mr.add (juce::var (mo)); } i->setProperty ("markersRaw", mr); }
                i->setProperty ("warp", in.warp); i->setProperty ("srcBpm", in.srcBpm); i->setProperty ("repitch", in.repitch); i->setProperty ("transpose", in.transpose); i->setProperty ("keepFormant", in.keepFormant); i->setProperty ("pitchState", in.transpose == 0.0 ? "off" : (in.warp && in.srcBpm > 0.0 ? in.warped != nullptr : in.pitched != nullptr) ? "ready" : "rendering"); i->setProperty ("warpState", ! in.warp ? "off" : in.warped ? "ready" : in.srcBpm > 0.0 ? "rendering" : "needs a tempo");
                i->setProperty ("gainDb", (double) in.gainDb); i->setProperty ("fadeIn", in.fadeIn); i->setProperty ("fadeOut", in.fadeOut); i->setProperty ("reverse", in.reverse);
                i->setProperty ("audio", true); i->setProperty ("name", in.content->audio->name); i->setProperty ("durationSec", in.content->audio->durationSec());
                i->setProperty ("startSec", tm->secondsAt (in.start)); i->setProperty ("lengthSec", in.length > 0.0 ? tm->secondsAt (in.start + in.length) - tm->secondsAt (in.start) : 0.0);
            }
            insts.add (juce::var (i));
            if (seen.insert (in.content->id).second) contents.add (clipVar (*in.content));
        }
        o->setProperty ("instances", insts); o->setProperty ("contents", contents);
        return juce::var (o);
    }
    // loads a saved arrangement; `known` keeps content that two tracks share as ONE content
    void loadArrangementVar (int track, const juce::var& a, std::map<int, std::shared_ptr<Clip>>& known)
    {
        std::map<int, std::shared_ptr<Clip>> mine;
        if (auto* cs = a.getProperty ("contents", {}).getArray())
            for (const auto& cv : *cs)
            {
                const int fid = (int) cv.getProperty ("id", 0);
                if (auto it = known.find (fid); fid != 0 && it != known.end()) { mine[fid] = it->second; continue; }
                std::shared_ptr<Clip> c;
                if ((bool) cv.getProperty ("audio", false))
                {
                    juce::String err; auto data = loadAudioData (juce::File (cv.getProperty ("path", {}).toString()), err);
                    if (data == nullptr) { loadWarnings.add (cv.getProperty ("name", "an audio clip").toString() + ": " + err); continue; }
                    c = std::make_shared<Clip>(); c->id = nextContentId.fetch_add (1); c->lengthBeats = 0.0; c->audio = data;
                }
                else c = contentFromVar (cv);
                known[fid] = c; mine[fid] = c;
            }
        auto ins = std::make_shared<Insts>();
        if (auto* is = a.getProperty ("instances", {}).getArray())
            for (const auto& iv : *is)
            {
                auto it = mine.find ((int) iv.getProperty ("contentId", 0)); if (it == mine.end()) continue;
                ClipInst in; in.id = nextInstanceId.fetch_add (1); in.start = juce::jmax (0.0, (double) iv.getProperty ("start", 0.0)); in.length = juce::jmax (0.0, (double) iv.getProperty ("length", 0.0));
                in.offset = juce::jmax (0.0, (double) iv.getProperty ("offset", 0.0)); in.content = it->second;
                in.gainDb = (float) juce::jlimit (-60.0, 24.0, (double) iv.getProperty ("gainDb", 0.0)); in.fadeIn = juce::jmax (0.0005, (double) iv.getProperty ("fadeIn", 0.002)); in.fadeOut = juce::jmax (0.0005, (double) iv.getProperty ("fadeOut", 0.002)); in.reverse = (bool) iv.getProperty ("reverse", false);
                if (auto* ma = iv.getProperty ("markersRaw", {}).getArray()) for (const auto& mv : *ma) in.markers.push_back ({ (double) mv.getProperty ("sec", 0.0), (double) mv.getProperty ("beat", 0.0) });
                in.colour = juce::jlimit (-1, kPaletteSize - 1, (int) iv.getProperty ("colour", -1)); in.muted = (bool) iv.getProperty ("muted", false); in.groove = juce::jlimit (-2, kGrooveChans - 1, (int) iv.getProperty ("groove", -2)); in.label = iv.getProperty ("label", {}).toString().substring (0, 60);
                in.warp = (bool) iv.getProperty ("warp", false); in.srcBpm = juce::jlimit (0.0, 400.0, (double) iv.getProperty ("srcBpm", 0.0)); in.repitch = (bool) iv.getProperty ("repitch", false); in.transpose = juce::jlimit (-24.0, 24.0, (double) iv.getProperty ("transpose", 0.0)); in.keepFormant = (bool) iv.getProperty ("keepFormant", true);
                ins->push_back (in);
            }
        putArr (track, ins);
    }

    // ---- recording the sound-card input as audio ----
    // Arm a track, choose "Sound-card input" as its source, press record: while the song is playing and recording, what the input hears is kept; when recording stops it becomes a sound file in the
    // recordings folder and a clip on the track, placed where the take began. The clip is moved earlier by the sound card's round-trip delay (what you played was heard that long after it sounded).
    juce::File recordingsDir;                                   // default: next to the program's settings
    double recordLatencyOverrideSec = -1.0;                     // tests set this; normally the sound card's own figure is used
    // AUDIO OPTIONS (Settings, Audio page), saved in the settings folder: a manual latency adjustment (ms, added to the figure the sound card reports, for placing a recording) and the most audio threads to use (-1 = a sensible number for this computer; it takes effect the next time the program starts)
    std::atomic<double> latencyAdjustMs { 0.0 }; std::atomic<int> maxThreadsSetting { -1 };
    // PLAY IN BACKGROUND (on by default). Off: when another program comes to the front the song is paused and the sound card let go (an ASIO driver serves one program at a time), and it is taken back on return.
    // MASTER TUNE, FOR AUDIO CLIPS ONLY (-100 to 100 cents, 0 = off). Recorded and imported sounds are moved in pitch, on top of each clip's own transpose, with their length unchanged. Instruments and plug-ins are NOT
    // retuned: there is no tuning command every plug-in obeys. Like a transpose, the new sound is made in the background and the clip plays as it was until it is ready.
    std::atomic<int> masterTuneCents { 0 };
    double clipSemis (const ClipInst& in) const { return in.transpose + (double) masterTuneCents.load() / 100.0; }
    void retuneClips()
    {
        for (int ti = 0; ti < trackCount.load(); ++ti) { bool any = false; for (const auto& in : tracks[(size_t) ti]->arrSnap.load()->insts) any = any || (in.content && in.content->audio); if (any) putArr (ti, instsOf (*tracks[(size_t) ti])); }
    }
    // RENDER WITH THE SOUND CARD'S BUFFER (off by default). On: an export or bounce works in pieces the size of the card's buffer, so a plug-in that behaves differently with the piece size sounds as it did while playing.
    std::atomic<bool> renderWithCardBuffer { false };
    int renderPieceSize() const { return renderWithCardBuffer.load() && blockSize > 0 ? juce::jlimit (16, 8192, blockSize) : 512; }
    std::atomic<bool> playInBackground { true }; bool audioReleased = false; double awaySince = 0.0, nextRestore = 0.0;
    enum class Background { nothing, release, restore };
    static Background backgroundAction (bool keepPlaying, bool inFront, bool released, bool recording, double secondsAway)
    {
        if (released) return keepPlaying || inFront ? Background::restore : Background::nothing;
        return ! keepPlaying && ! inFront && ! recording && secondsAway >= 1.0 ? Background::release : Background::nothing;      // a second's grace, so a passing pop-up never drops the sound; never while recording
    }
    static bool programIsInFront()
    {
       #if JUCE_WINDOWS
        HWND fg = GetForegroundWindow(); if (fg == nullptr) return true;
        DWORD a = 0, b = 0; GetWindowThreadProcessId (fg, &a); if (auto root = GetAncestor (fg, GA_ROOTOWNER)) GetWindowThreadProcessId (root, &b);      // a pop-up of our own web view belongs to us
        return a == GetCurrentProcessId() || b == GetCurrentProcessId();
       #else
        return juce::Process::isForegroundProcess();
       #endif
    }
    void serviceBackground()
    {
        const bool keep = playInBackground.load(); if (keep && ! audioReleased) { awaySince = 0.0; return; }
        const double now = juce::Time::getMillisecondCounterHiRes() / 1000.0; const bool front = keep || programIsInFront();
        if (front) awaySince = 0.0; else if (awaySince <= 0.0) awaySince = now;
        const auto act = backgroundAction (keep, front, audioReleased, transport.recording.load(), front ? 0.0 : now - awaySince);
        if (act == Background::release && deviceManager.getCurrentAudioDevice() != nullptr)
        {
            if (transport.playing.load()) transport.stop();
            deviceManager.closeAudioDevice(); audioReleased = true; stateDirty.store (true);
        }
        else if (act == Background::restore && now >= nextRestore)
        {
            deviceManager.restartLastAudioDevice(); nextRestore = now + 2.0;      // another program may still hold it: try again shortly
            if (deviceManager.getCurrentAudioDevice() != nullptr) { audioReleased = false; stateDirty.store (true); }
        }
    }
    juce::File audioOptionsFile() const { return settingsDir.getChildFile (juce::SystemStats::getEnvironmentVariable ("ASKARCHIE_HOST_TEST", {}) == "1" ? "audio-options-test.json" : "audio-options.json"); }
    void loadAudioOptions()
    {
        latencyAdjustMs.store (0.0); maxThreadsSetting.store (-1); playInBackground.store (true); renderWithCardBuffer.store (false); masterTuneCents.store (0);
        const auto f = audioOptionsFile(); if (! f.existsAsFile()) return;
        const auto v = juce::JSON::parse (f); if (! v.isObject()) return;
        latencyAdjustMs.store (juce::jlimit (-200.0, 200.0, (double) v.getProperty ("latencyAdjustMs", 0.0)));
        maxThreadsSetting.store (juce::jlimit (-1, 7, (int) v.getProperty ("maxThreads", -1)));
        playInBackground.store ((bool) v.getProperty ("playInBackground", true));
        renderWithCardBuffer.store ((bool) v.getProperty ("renderWithCardBuffer", false));
        masterTuneCents.store (juce::jlimit (-100, 100, (int) v.getProperty ("masterTuneCents", 0)));
    }
    bool setAudioOptions (const juce::var& p)
    {
        bool any = false;
        if (p.hasProperty ("latencyAdjustMs")) { latencyAdjustMs.store (juce::jlimit (-200.0, 200.0, (double) p.getProperty ("latencyAdjustMs", 0.0))); any = true; }
        if (p.hasProperty ("maxThreads")) { maxThreadsSetting.store (juce::jlimit (-1, 7, (int) p.getProperty ("maxThreads", -1))); any = true; }
        if (p.hasProperty ("playInBackground")) { playInBackground.store ((bool) p.getProperty ("playInBackground", true)); any = true; }
        if (p.hasProperty ("renderWithCardBuffer")) { renderWithCardBuffer.store ((bool) p.getProperty ("renderWithCardBuffer", false)); any = true; }
        if (p.hasProperty ("masterTuneCents")) { const int c = juce::jlimit (-100, 100, juce::roundToInt ((double) p.getProperty ("masterTuneCents", 0))); if (masterTuneCents.exchange (c) != c) retuneClips(); any = true; }
        if (! any) return false;
        auto* o = new juce::DynamicObject(); o->setProperty ("latencyAdjustMs", latencyAdjustMs.load()); o->setProperty ("maxThreads", maxThreadsSetting.load()); o->setProperty ("playInBackground", playInBackground.load()); o->setProperty ("renderWithCardBuffer", renderWithCardBuffer.load()); o->setProperty ("masterTuneCents", masterTuneCents.load());
        audioOptionsFile().replaceWithText (juce::JSON::toString (juce::var (o))); stateDirty.store (true);
        return true;
    }
    juce::String lastTakeName; std::atomic<int> takeSerial { 0 }; int lastTakeTrack = -1;
    void ensureAudioRec (TrackState& t) { if (t.audioRecOwner == nullptr) { t.audioRecOwner = std::make_unique<AudioRec>(); t.audioRec.store (t.audioRecOwner.get()); } }
    double roundTripLatencySec()
    {
        const double adj = latencyAdjustMs.load() / 1000.0;
        if (recordLatencyOverrideSec >= 0.0) return juce::jmax (0.0, recordLatencyOverrideSec + adj);
        if (auto* d = deviceManager.getCurrentAudioDevice()) return juce::jmax (0.0, (double) (d->getInputLatencyInSamples() + d->getOutputLatencyInSamples()) / juce::jmax (1.0, d->getCurrentSampleRate()) + adj);
        return juce::jmax (0.0, adj);
    }
    void drainAudioRecordings (bool finishIfStopped)
    {
        const bool recNow = transport.recording.load() && transport.playing.load();
        for (auto& tp : tracks)
        {
            auto* rec = tp->audioRec.load(); if (rec == nullptr) continue;
            const int n = rec->fifo.getNumReady();
            if (n > 0)
            {
                int s1, z1, s2, z2; rec->fifo.prepareToRead (n, s1, z1, s2, z2);
                if (rec->takeStartSec < 0.0) rec->takeStartSec = rec->startSec.load();
                if (rec->takeL.size() < (size_t) (sampleRate * 1800.0))                  // (half an hour is the most one take keeps)
                    for (int c = 0; c < 2; ++c) { auto& v = c == 0 ? rec->takeL : rec->takeR; v.insert (v.end(), rec->ring[c].begin() + s1, rec->ring[c].begin() + s1 + z1); if (z2 > 0) v.insert (v.end(), rec->ring[c].begin() + s2, rec->ring[c].begin() + s2 + z2); }
                rec->fifo.finishedRead (z1 + z2);
            }
            if (finishIfStopped && ! recNow && ! rec->takeL.empty()) commitAudioTake (*tp);
        }
    }
    void commitAudioTake (TrackState& t)
    {
        auto& rec = *t.audioRec.load(); const int n = (int) rec.takeL.size();
        const double start = rec.takeStartSec; const bool lost = rec.overflow.exchange (false);
        juce::AudioBuffer<float> b (2, n); std::memcpy (b.getWritePointer (0), rec.takeL.data(), (size_t) n * sizeof (float)); std::memcpy (b.getWritePointer (1), rec.takeR.data(), (size_t) n * sizeof (float));
        rec.takeL.clear(); rec.takeR.clear(); rec.takeStartSec = -1.0; rec.startSec.store (-1.0);
        if (n < 512 || start < 0.0) return;                                                           // a touch of the record button, not a take
        auto dir = recordingsDir != juce::File() ? recordingsDir : settingsDir.getChildFile ("recordings"); dir.createDirectory();
        auto f = dir.getChildFile ("take-" + juce::Time::getCurrentTime().formatted ("%Y%m%d-%H%M%S") + "-track" + juce::String (t.index + 1) + ".wav"); int k = 2;
        while (f.existsAsFile()) { f = dir.getChildFile (f.getFileNameWithoutExtension().upToFirstOccurrenceOf ("~", false, false) + "~" + juce::String (k++) + ".wav"); }
        { juce::WavAudioFormat wav; auto out = f.createOutputStream(); if (out == nullptr) return;
          std::unique_ptr<juce::AudioFormatWriter> w (wav.createWriterFor (out.get(), sampleRate, 2, 32, {}, 0)); if (w == nullptr) return; out.release(); w->writeFromAudioSampleBuffer (b, 0, n); }
        juce::String err; const double lat = juce::jmin (roundTripLatencySec(), (double) n / sampleRate * 0.5);
        if (editImportAudio (t.index, f, transport.tempoNow()->beatAt (start), err, lat) == 0) { lastTakeName = "could not use the take: " + err; }
        else lastTakeName = f.getFileName() + (lost ? " (the computer could not keep up: part of it is missing)" : "");
        lastTakeTrack = t.index; takeSerial.fetch_add (1); stateDirty.store (true);
    }

    // recording: called from the 30 Hz service tick
    // PATTERN BLOCKS. Once per plug-in, find the controls that choose its pattern ("Drum Pattern") and tie it to the bass ("Drum Link", if it has one), and give it a position of its own to listen to (see PatternPlayHead),
    // so a block can set them from the audio thread. How many patterns a track's plug-in has (0 = it has no pattern control, so no pattern blocks).
    int patternCountOf (const TrackState& t) const
    {
        for (const auto& sl : t.slots) if (sl != nullptr && sl->patBank.load() != nullptr) return sl->patTotal();
        return 0;
    }
    std::map<Slot*, int> patMismatch;
    void resolvePatternSlots (TrackState& t)
    {
        for (auto& sl : t.slots)
        {
            if (sl == nullptr || sl->proc == nullptr || sl->patChecked.load()) continue;
            juce::AudioProcessorParameter *letter = nullptr, *number = nullptr, *activeBank = nullptr;
            for (auto* p : sl->proc->getParameters()) { const auto nm = p->getName (64); if (nm.equalsIgnoreCase ("Drum Pattern")) sl->patBank.store (p); else if (nm.equalsIgnoreCase ("Drum Link")) sl->patLink.store (p); else if (nm.equalsIgnoreCase ("Pattern")) letter = p; else if (nm.equalsIgnoreCase ("Bank")) number = p; else if (nm.equalsIgnoreCase ("Active Bank")) activeBank = p; }
            if (sl->patBank.load() == nullptr && activeBank != nullptr && activeBank->getNumSteps() > 1) sl->patBank.store (activeBank);      // Archie Acid 303: its sixteen patterns A1..D4 are its Active Bank (the drums, where a plug-in has them, are chosen by Drum Pattern instead: drums are drums, bass is bass)
            // Archie Drums holds its pattern as two stepped controls: "Pattern" (A-D) and "Bank" (1-4). Together they are A1..D4.
            if (sl->patBank.load() == nullptr && letter != nullptr && number != nullptr && letter->getNumSteps() > 1 && number->getNumSteps() > 1) { sl->patBank.store (letter); sl->patSub.store (number); }
            if (sl->patBank.load() != nullptr) { sl->patHead = std::make_unique<PatternPlayHead>(); sl->patHead->base = &playHead; sl->proc->setPlayHead (sl->patHead.get()); }
            sl->patChecked.store (true);
        }
    }
    // A track that has just been given a pattern plug-in (Archie Drums, Archie Acid 303) and has nothing on it starts with one pattern block, so it plays at once: pattern A1, four bars.
    void autoPatternBlock (int chain)
    {
        auto* t = chainAt (chain); if (t == nullptr || chain >= kReturnBase || ! t->arrSnap.load()->insts.empty()) return;
        resolvePatternSlots (*t); if (patternCountOf (*t) == 0) return;
        auto* c = new juce::DynamicObject(); c->setProperty ("lengthBeats", 16.0); c->setProperty ("patternBank", 0);
        editAddInstance (chain, 0.0, 16.0, 0.0, juce::var (c), 0);
    }
    void servicePatterns()
    {
        for (int ti = 0; ti < trackCount.load(); ++ti)
        {
            auto& t = *tracks[(size_t) ti]; resolvePatternSlots (t);
            // FOLLOW THE UNIT: a pattern chosen on the plug-in's own panel while a block plays is written into that block (it must differ for a moment, so our own switch at a block edge is never taken for it)
            const int want = t.patWant.load();
            for (auto& sl : t.slots)
            {
                auto* pb = sl != nullptr ? sl->patBank.load() : nullptr; if (pb == nullptr) continue;
                if (want < 0 || ! transport.playing.load()) { patMismatch.erase (sl.get()); continue; }
                const int now = sl->patGet();
                if (now == want) { patMismatch.erase (sl.get()); continue; }
                if (++patMismatch[sl.get()] < 4) continue;
                patMismatch.erase (sl.get());
                const double beat = transport.posBeats.load();
                for (const auto& in : t.arrSnap.load()->insts)
                    if (in.content != nullptr && in.content->patternBank >= 0 && in.start <= beat && (in.length <= 0.0 || beat < in.start + in.length) && ! in.muted)
                    { auto* c = new juce::DynamicObject(); c->setProperty ("lengthBeats", in.content->lengthBeats); c->setProperty ("patternBank", now); editContent (in.content->id, juce::var (c)); break; }
            }
        }
    }
    void service()
    {
        serviceBackground();
        if (cuePadsDirty.exchange (false)) { saveCuePads(); stateDirty.store (true); }
        drainAudioRecordings (true);
        servicePatterns();
        applyParamAutomation();
        pollAutoLearn(); pollAutoRecord(); pollRemote();
        if (++midiPollTick >= 60) { midiPollTick = 0; pollMidiDevices(); }
        if (++latTick >= 30) { latTick = 0; refreshLatency(); }
        applyWarpResults();
        // LOOP RECORDING: while recording inside a loop, the notes played in a pass are put into the song the moment the loop returns to its start, so the next pass PLAYS them (and new notes are laid over them) instead of
        // staying silent until the song is stopped. The song position moving back while recording inside a loop is how a wrap is seen.
        { const bool rp = transport.recording.load() && transport.playing.load(); if (! rp) recSessionLive = false; else if (! recSessionLive) { recSessionLive = true; ++recSessionId; } }      // a take = from recording starting to it stopping
        const double recBeatNow = transport.posBeats.load();
        const bool loopRecording = transport.recording.load() && transport.playing.load() && transport.looping.load();
        const bool loopWrapped = loopRecording && lastRecServiceBeat >= 0.0 && recBeatNow < lastRecServiceBeat - 0.05;
        lastRecServiceBeat = loopRecording ? recBeatNow : -1.0;
        for (auto& t : tracks)
        {
            int s1, z1, s2, z2;
            t->recFifo.prepareToRead (t->recFifo.getNumReady(), s1, z1, s2, z2);
            for (int i = 0; i < z1; ++i) t->pending.push_back (t->recBuf[s1 + i]);
            for (int i = 0; i < z2; ++i) t->pending.push_back (t->recBuf[s2 + i]);
            t->recFifo.finishedRead (z1 + z2);
            if (! t->pending.empty() && transport.recording.load() && ++ghostTick >= 3) { ghostTick = 0; stateDirty.store (true); }          // the page shows the take as it goes in
            if (loopWrapped && ! t->pending.empty()) commitRecording (*t, true);
            else if (! t->pending.empty() && ! (transport.recording.load() && transport.playing.load())) commitRecording (*t);
        }
    }

    double lastRecServiceBeat = -1.0; int ghostTick = 0; int recSessionId = 0; bool recSessionLive = false;
    // GHOST NOTES: the notes played so far in the take in progress, as [beat, length, note], so the page can show them going in. A note still held lasts up to where the song is now.
    juce::var recGhostsVar (TrackState& t)
    {
        juce::Array<juce::var> out; if (t.pending.empty()) return juce::var (out);
        std::map<int, std::pair<double, int>> open; const double now = transport.posBeats.load();
        auto add = [&out] (double beat, double len, int note) { juce::Array<juce::var> n; n.add (beat); n.add (len); n.add (note); out.add (juce::var (n)); };
        for (const auto& e : t.pending)
        {
            const int status = e.b[0] & 0xF0, key = e.b[1] + (e.b[0] & 0x0F) * 128; const bool on = status == 0x90 && e.b[2] > 0;
            if (on) open[key] = { e.beat, e.b[1] };
            else if (status == 0x80 || status == 0x90) { auto it = open.find (key); if (it != open.end()) { add (it->second.first, juce::jmax (0.05, e.beat - it->second.first), it->second.second); open.erase (it); } }
            if (out.size() > 600) break;
        }
        for (const auto& kv : open) add (kv.second.first, juce::jmax (0.05, now >= kv.second.first ? now - kv.second.first : transport.loopEnd.load() - kv.second.first), kv.second.second);
        return juce::var (out);
    }
    // atLoopWrap: the loop has just returned to its start. Only the events of the pass that ended are taken (the first event whose beat is earlier than the one before it begins the new pass and stays pending),
    // a note still held at the wrap ends at the end of the loop, and a first take becomes a clip that covers the loop, so later passes are laid over it.
    void commitRecording (TrackState& t, bool atLoopWrap = false)
    {
        std::vector<RecEvent> ev; ev.swap (t.pending);
        const double loopS = transport.loopStart.load(), loopE = transport.loopEnd.load(); const bool loopTake = atLoopWrap && loopE > loopS + 0.01;
        if (atLoopWrap)
        {
            size_t split = ev.size(); for (size_t i = 1; i < ev.size(); ++i) if (ev[i].beat < ev[i - 1].beat - 1.0) { split = i; break; }
            t.pending.assign (ev.begin() + (long) split, ev.end()); ev.resize (split);
        }
        std::map<int, std::pair<double, int>> open;
        std::vector<NoteEv> got; double maxBeat = 0.0;
        for (const auto& e : ev)
        {
            const int status = e.b[0] & 0xF0, ch = (e.b[0] & 0x0F) + 1, note = e.b[1];
            const bool on = status == 0x90 && e.b[2] > 0;
            if (on) open[note + ch * 128] = { e.beat, e.b[2] };
            else if (auto it = open.find (note + ch * 128); it != open.end())
            {
                NoteEv n; n.beat = it->second.first; n.len = juce::jmax (0.05, e.beat - n.beat); n.note = note; n.vel = it->second.second; n.ch = ch;
                got.push_back (n); maxBeat = juce::jmax (maxBeat, n.beat + n.len); open.erase (it);
            }
        }
        for (auto& kv : open) { NoteEv n; n.beat = kv.second.first; n.len = loopTake ? juce::jmax (0.05, loopE - n.beat) : 0.25; n.note = kv.first % 128; n.vel = kv.second.second; n.ch = kv.first / 128; got.push_back (n); maxBeat = juce::jmax (maxBeat, n.beat + n.len); }
        if (got.empty()) return;
        auto before = instsOf (t);
        auto ins = std::make_shared<Insts> (*before);
        const double bar = (double) transport.tsNum.load();
        double firstBeat = got[0].beat; for (const auto& n : got) firstBeat = juce::jmin (firstBeat, n.beat);
        // recording goes into the clip instance that is playing where the first note was played (every instance of that content changes with it); otherwise a new clip is made
        int hit = -1;
        for (int i = 0; i < (int) ins->size(); ++i) { const auto& in = (*ins)[(size_t) i]; if (in.content && in.content->lengthBeats > 0.0 && firstBeat >= in.start && (in.length <= 0.0 || firstBeat < in.start + in.length)) { hit = i; break; } }
        if (hit >= 0)
        {
            const ClipInst in = (*ins)[(size_t) hit];
            auto c = std::make_shared<Clip> (*in.content);
            for (auto n : got) { n.beat = std::fmod (juce::jmax (0.0, n.beat - in.start + in.offset), c->lengthBeats); c->notes.push_back (n); }
            for (auto& x : *ins) if (x.content == in.content) x.content = c;
        }
        else
        {
            auto c = std::make_shared<Clip>(); c->id = nextContentId.fetch_add (1);
            ClipInst in; in.id = nextInstanceId.fetch_add (1);
            if (ins->empty() && ! loopTake)
            {
                c->lengthBeats = juce::jmax (bar, std::ceil (maxBeat / bar) * bar);                        // the first take: one pattern that loops
                for (auto n : got) { n.beat = std::fmod (n.beat, c->lengthBeats); c->notes.push_back (n); }
            }
            else
            {
                in.start = std::floor ((loopTake ? loopS : firstBeat) / bar) * bar;                      // a later take elsewhere on the timeline (or a loop take: the clip covers the loop): its own clip, played once
                c->lengthBeats = juce::jmax (bar, std::ceil (((loopTake ? loopE : maxBeat) - in.start) / bar) * bar); in.length = c->lengthBeats;
                for (auto n : got) { n.beat -= in.start; c->notes.push_back (n); }
            }
            in.content = c; ins->push_back (in);
        }
        putArr (t.index, ins);
        recordArrChange (t.index, "Record notes", before, "take:" + std::to_string (recSessionId) + ":" + std::to_string (t.index), 1.0e12);       // every pass of one take is ONE undo step
    }

    //==========================================================================
    // audio files
    bool loadAudioFile (int track, const juce::File& f)
    {
        auto* t = trackAt (track);
        if (t == nullptr || ! f.existsAsFile()) return false;
        juce::AudioFormatManager fm; fm.registerBasicFormats();
        std::unique_ptr<juce::AudioFormatReader> r (fm.createReaderFor (f));
        if (r == nullptr || r->lengthInSamples <= 0 || r->lengthInSamples > 60LL * 60 * 48000) return false;
        juce::AudioBuffer<float> src ((int) juce::jmin<unsigned int> (2, r->numChannels), (int) r->lengthInSamples);
        r->read (&src, 0, (int) r->lengthInSamples, 0, true, true);
        const double ratio = r->sampleRate / sampleRate;
        const int outLen = (int) std::ceil ((double) src.getNumSamples() / ratio);
        auto fa = std::make_unique<FileAudio>(); fa->buf.setSize (src.getNumChannels(), outLen); fa->valid = true;
        auto* dst = &fa->buf;
        for (int c = 0; c < src.getNumChannels(); ++c)
        {
            juce::LagrangeInterpolator interp;
            interp.process (ratio, src.getReadPointer (c), dst->getWritePointer (c), outLen);
        }
        t->fileSnap.publish (std::move (fa));
        t->filePath = f.getFullPathName();
        return true;
    }

    //==========================================================================
    // Undo and redo. The functions elsewhere in this class are the RAW edits: they never record, so loading a project or undoing cannot record itself.
    // The edit* functions below are what a person's actions go through (the bridge, the recording tick): each makes the change, then records how to put it back and how to do it again.
    UndoStack undo;

    juce::var trackParam (TrackState& t, const juce::String& key)
    {
        if (key == "source") return t.source.load();
        if (key == "kind") return t.kind.load();
        if (key == "inChan") return t.inChan.load();
        if (key == "inStereo") return t.inStereo.load();
        if (key == "monitor") return t.monitor.load();
        if (key == "gainDb") return (double) t.gainDb.load();
        if (key == "pan")    return (double) t.pan.load();
        if (key == "inGainDb") return (double) t.inGainDb.load();
        if (key == "inInvert") return t.inInvert.load();
        if (key == "patPlay") return t.patPlay.load();
        if (key == "patTrans") return t.patTrans.load();
        if (key == "groove") return t.grooveCh.load();
        if (key == "width") return (double) t.width.load();
        if (key == "gateOn") return t.gateOn.load();
        if (key == "gateExp") return t.gateExp.load();
        if (key == "gateFast") return t.gateFast.load();
        if (key == "gateThr") return (double) t.gateThr.load();
        if (key == "gateRange") return (double) t.gateRange.load();
        if (key == "gateRel") return (double) t.gateRel.load();
        if (key == "gateHold") return (double) t.gateHold.load();
        if (key == "compOn") return t.compOn.load();
        if (key == "compFast") return t.compFast.load();
        if (key == "compPeak") return t.compPeak.load();
        if (key == "compThr") return (double) t.compThr.load();
        if (key == "compRatio") return (double) t.compRatio.load();
        if (key == "compRel") return (double) t.compRel.load();
        if (key == "compMakeup") return (double) t.compMakeup.load();
        if (key == "eqHpOn") return t.eqHpOn.load();
        if (key == "eqHpHz") return (double) t.eqHpHz.load();
        if (key == "eqLpOn") return t.eqLpOn.load();
        if (key == "eqLpHz") return (double) t.eqLpHz.load();
        if (key == "mute")   return t.mute.load();
        if (key == "solo")   return t.solo.load();
        if (key == "armed")  return t.armed.load();
        if (key == "name")   return t.name;
        if (key == "colour") return t.colour.load();
        return {};
    }
    bool applyTrackParam (TrackState& t, const juce::String& key, const juce::var& v)
    {
        if (key == "source") t.source.store (juce::jlimit (0, 6, (int) v));
        else if (key == "kind") t.kind.store (juce::jlimit (0, 1, (int) v));
        else if (key == "inChan") t.inChan.store (juce::jlimit (0, kMaxInputs - 1, (int) v));
        else if (key == "inStereo") t.inStereo.store ((bool) v);
        else if (key == "monitor") t.monitor.store ((bool) v);
        else if (key == "gainDb") t.gainDb.store ((float) juce::jlimit (-90.0, 12.0, (double) v));
        else if (key == "pan") t.pan.store ((float) juce::jlimit (-1.0, 1.0, (double) v));
        else if (key == "inGainDb") t.inGainDb.store ((float) juce::jlimit ((double) InputStage::kMinDb, (double) InputStage::kMaxDb, (double) v));
        else if (key == "inInvert") t.inInvert.store ((bool) v);
        else if (key == "patPlay") t.patPlay.store (juce::jlimit (0, 2, (int) v));
        else if (key == "patTrans") t.patTrans.store (juce::jlimit (-24, 24, (int) v));
        else if (key == "groove") { const int v2 = (int) v; t.grooveCh.store (v2 >= 1 && v2 < kGrooveChans ? v2 : -1); }
        else if (key == "width") t.width.store ((float) juce::jlimit (0.0, (double) WidthStage::kMax, (double) v));
        else if (key == "gateOn") t.gateOn.store ((bool) v);
        else if (key == "gateExp") t.gateExp.store ((bool) v);
        else if (key == "gateFast") t.gateFast.store ((bool) v);
        else if (key == "gateThr") t.gateThr.store ((float) juce::jlimit ((double) GateStage::kThrMin, (double) GateStage::kThrMax, (double) v));
        else if (key == "gateRange") t.gateRange.store ((float) juce::jlimit (0.0, (double) GateStage::kRangeMax, (double) v));
        else if (key == "gateRel") t.gateRel.store ((float) juce::jlimit ((double) GateStage::kRelMin, (double) GateStage::kRelMax, (double) v));
        else if (key == "gateHold") t.gateHold.store ((float) juce::jlimit (0.0, (double) GateStage::kHoldMax, (double) v));
        else if (key == "compOn") t.compOn.store ((bool) v);
        else if (key == "compFast") t.compFast.store ((bool) v);
        else if (key == "compPeak") t.compPeak.store ((bool) v);
        else if (key == "compThr") t.compThr.store ((float) juce::jlimit ((double) CompStage::kThrMin, (double) CompStage::kThrMax, (double) v));
        else if (key == "compRatio") t.compRatio.store ((float) juce::jlimit ((double) CompStage::kRatioMin, (double) CompStage::kRatioMax, (double) v));
        else if (key == "compRel") t.compRel.store ((float) juce::jlimit ((double) CompStage::kRelMin, (double) CompStage::kRelMax, (double) v));
        else if (key == "compMakeup") t.compMakeup.store ((float) juce::jlimit (0.0, (double) CompStage::kMakeMax, (double) v));
        else if (key == "eqHpOn") t.eqHpOn.store ((bool) v);
        else if (key == "eqHpHz") t.eqHpHz.store ((float) juce::jlimit ((double) EqStage::kHpMin, (double) EqStage::kHpMax, (double) v));
        else if (key == "eqLpOn") t.eqLpOn.store ((bool) v);
        else if (key == "eqLpHz") t.eqLpHz.store ((float) juce::jlimit ((double) EqStage::kLpMin, (double) EqStage::kLpMax, (double) v));
        else if (key == "mute") { t.mute.store ((bool) v); updateSolo(); }
        else if (key == "solo") { t.solo.store ((bool) v); updateSolo(); }
        else if (key == "armed") { t.armed.store ((bool) v); if ((bool) v) ensureAudioRec (t); updateTargets(); }
        else if (key == "name") t.name = v.toString().substring (0, 40);
        else if (key == "colour") t.colour.store (juce::jlimit (-1, kPaletteSize - 1, (int) v));
        else return false;
        stateDirty.store (true);
        return true;
    }
    // a track (0..) or an FX return (100..): one setting. Levels and pan are dragged, so they merge into one undo step; the rest are separate steps. Arming is a live state, not an edit.
    bool editChainParam (int chain, const juce::String& key, const juce::var& v)
    {
        auto* t = chainAt (chain);
        if (t == nullptr || chain == kMasterChain) return false;                       // (the master level has its own edit)
        if (chain >= kReturnBase && ! (key == "gainDb" || key == "pan" || key == "mute" || key == "name")) return false;
        const juce::var before = trackParam (*t, key);
        if (! applyTrackParam (*t, key, v)) return false;
        const juce::var after = trackParam (*t, key);
        if (transport.recording.load() && (key == "gainDb" || key == "pan" || key == "inGainDb")) autoRecFromControl (chain, key, (double) after);
        if (key == "armed" || before == after) return true;
        const bool dragged = key == "gainDb" || key == "pan" || key == "inGainDb" || key == "eqHpHz" || key == "eqLpHz" || key == "compThr" || key == "compRatio" || key == "compRel" || key == "compMakeup" || key == "gateThr" || key == "gateRange" || key == "gateRel" || key == "gateHold" || key == "width";
        const juce::String what = key == "gainDb" ? "level" : key == "source" ? "sound source" : key == "inGainDb" ? "input gain" : key == "inInvert" ? "input polarity" : key == "groove" ? "groove channel" : key == "width" ? "stereo width" : key == "gateOn" ? "gate switch" : key == "gateExp" ? "gate mode" : key == "gateFast" ? "gate attack" : key == "gateThr" ? "gate threshold" : key == "gateRange" ? "gate range" : key == "gateRel" ? "gate release" : key == "gateHold" ? "gate hold" : key == "compOn" ? "compressor switch" : key == "compFast" ? "compressor attack" : key == "compPeak" ? "compressor detector" : key == "compThr" ? "compressor threshold" : key == "compRatio" ? "compressor ratio" : key == "compRel" ? "compressor release" : key == "compMakeup" ? "compressor make-up" : key == "eqHpOn" ? "high-pass switch" : key == "eqHpHz" ? "high-pass frequency" : key == "eqLpOn" ? "low-pass switch" : key == "eqLpHz" ? "low-pass frequency" : key;
        undo.record ((t->name + " " + what).toStdString(), dragged ? ("p:" + juce::String (chain) + ":" + key).toStdString() : std::string(),
                     [this, chain, key, before] { if (auto* x = chainAt (chain)) applyTrackParam (*x, key, before); },
                     [this, chain, key, after]  { if (auto* x = chainAt (chain)) applyTrackParam (*x, key, after); });
        return true;
    }
    // ---- THE EQ PANEL (channels and the master): four parametric bands behind the high-pass and low-pass. Keys: b0on, b0hz, b0gain, b0q ... b3q for the bands (0 low shelf, 1 and 2 bells, 3 high shelf), and hpOn, hpHz, lpOn, lpHz ----
    EqPanel* eqPanelOf (int chain) { if (chain == kMasterChain) return master != nullptr ? &master->eq : nullptr; auto* t = chainAt (chain); return (t != nullptr && chain >= 0 && chain < kReturnBase) ? &t->eq : nullptr; }
    static bool eqKey (const juce::String& key, int& band, juce::String& what) { if (key.length() >= 3 && key[0] == 'b' && key[1] >= '0' && key[1] < '0' + kEqBands) { band = key[1] - '0'; what = key.substring (2); return what == "on" || what == "hz" || what == "gain" || what == "q"; } band = -1; what = key; return what == "hpOn" || what == "hpHz" || what == "lpOn" || what == "lpHz"; }
    double eqGet (int chain, const juce::String& key)
    {
        auto* p = eqPanelOf (chain); if (p == nullptr) return 0.0; int b; juce::String w; if (! eqKey (key, b, w)) return 0.0;
        if (b >= 0) return w == "on" ? (p->band[b].on.load() ? 1.0 : 0.0) : w == "hz" ? (double) p->band[b].hz.load() : w == "gain" ? (double) p->band[b].gain.load() : (double) p->band[b].q.load();
        if (chain == kMasterChain && master != nullptr) return w == "hpOn" ? (master->eqHpOn.load() ? 1.0 : 0.0) : w == "hpHz" ? (double) master->eqHpHz.load() : w == "lpOn" ? (master->eqLpOn.load() ? 1.0 : 0.0) : (double) master->eqLpHz.load();
        auto* t = chainAt (chain); if (t == nullptr) return 0.0;
        return w == "hpOn" ? (t->eqHpOn.load() ? 1.0 : 0.0) : w == "hpHz" ? (double) t->eqHpHz.load() : w == "lpOn" ? (t->eqLpOn.load() ? 1.0 : 0.0) : (double) t->eqLpHz.load();
    }
    bool eqApply (int chain, const juce::String& key, double v)
    {
        auto* p = eqPanelOf (chain); if (p == nullptr) return false; int b; juce::String w; if (! eqKey (key, b, w)) return false;
        if (b >= 0)
        {
            if (w == "on") p->band[b].on.store (v != 0.0); else if (w == "hz") p->band[b].hz.store ((float) juce::jlimit ((double) ParaEq::kHzMin, (double) ParaEq::kHzMax, v));
            else if (w == "gain") p->band[b].gain.store ((float) juce::jlimit (-(double) ParaEq::kGainMax, (double) ParaEq::kGainMax, v)); else p->band[b].q.store ((float) juce::jlimit ((double) ParaEq::kQMin, (double) ParaEq::kQMax, v));
        }
        else if (chain == kMasterChain && master != nullptr)
        { if (w == "hpOn") master->eqHpOn.store (v != 0.0); else if (w == "hpHz") master->eqHpHz.store ((float) juce::jlimit ((double) EqStage::kHpMin, (double) EqStage::kHpMax, v)); else if (w == "lpOn") master->eqLpOn.store (v != 0.0); else master->eqLpHz.store ((float) juce::jlimit ((double) EqStage::kLpMin, (double) EqStage::kLpMax, v)); }
        else { auto* t = chainAt (chain); if (t == nullptr) return false; if (w == "hpOn") t->eqHpOn.store (v != 0.0); else if (w == "hpHz") t->eqHpHz.store ((float) juce::jlimit ((double) EqStage::kHpMin, (double) EqStage::kHpMax, v)); else if (w == "lpOn") t->eqLpOn.store (v != 0.0); else t->eqLpHz.store ((float) juce::jlimit ((double) EqStage::kLpMin, (double) EqStage::kLpMax, v)); }
        stateDirty.store (true); return true;
    }
    bool editEq (int chain, const juce::String& key, double value)
    {
        int b; juce::String w; if (eqPanelOf (chain) == nullptr || ! eqKey (key, b, w)) return false;
        const double before = eqGet (chain, key); if (! eqApply (chain, key, value)) return false; const double after = eqGet (chain, key);
        if (recordingMoves())      // recording: the move goes into the setting's own lane (the take is the undo step)
        {
            const int param = b >= 0 ? b * 4 + (w == "on" ? 0 : w == "hz" ? 1 : w == "gain" ? 2 : 3) : w == "hpOn" ? 16 : w == "hpHz" ? 17 : w == "lpOn" ? 18 : 19;
            if (auto* p = eqPanelOf (chain)) p->handUntilMs[(size_t) param].store (juce::Time::getMillisecondCounterHiRes() + 400.0);
            autoRecWrite (chain, kEqLaneKind, -1, param, transport.posBeats.load(), eqLaneValue (param, after)); return true;
        }
        if (before == after) return true;
        const juce::String who = chain == kMasterChain ? juce::String ("Master") : (chainAt (chain) != nullptr ? chainAt (chain)->name : juce::String ("Track"));
        undo.record ((who + " EQ").toStdString(), (w == "on" || w == "hpOn" || w == "lpOn") ? std::string() : ("eq:" + juce::String (chain) + ":" + key).toStdString(),
                     [this, chain, key, before] { eqApply (chain, key, before); }, [this, chain, key, after] { eqApply (chain, key, after); });
        return true;
    }
    juce::var eqVar (int chain)
    {
        auto* p = eqPanelOf (chain); if (p == nullptr) return {};
        auto* o = new juce::DynamicObject(); juce::Array<juce::var> bands;
        for (int i = 0; i < kEqBands; ++i) { auto* bo = new juce::DynamicObject(); bo->setProperty ("on", p->band[i].on.load()); bo->setProperty ("hz", (double) p->band[i].hz.load()); bo->setProperty ("gain", (double) p->band[i].gain.load()); bo->setProperty ("q", (double) p->band[i].q.load()); bands.add (juce::var (bo)); }
        o->setProperty ("bands", bands); for (const char* k : { "hpOn", "hpHz", "lpOn", "lpHz" }) o->setProperty (k, eqGet (chain, k));
        return juce::var (o);
    }
    void loadEqVar (int chain, const juce::var& v)
    {
        if (eqPanelOf (chain) == nullptr || ! v.isObject()) return;
        if (auto* bands = v.getProperty ("bands", {}).getArray())
            for (int i = 0; i < kEqBands && i < bands->size(); ++i)
            {
                const auto& bo = (*bands)[i]; const juce::String k = "b" + juce::String (i);
                eqApply (chain, k + "on", (bool) bo.getProperty ("on", false) ? 1.0 : 0.0); eqApply (chain, k + "hz", (double) bo.getProperty ("hz", 1000.0)); eqApply (chain, k + "gain", (double) bo.getProperty ("gain", 0.0)); eqApply (chain, k + "q", (double) bo.getProperty ("q", 1.0));
            }
        if (chain == kMasterChain) for (const char* k : { "hpOn", "hpHz", "lpOn", "lpHz" }) if (! v.getProperty (k, {}).isVoid()) eqApply (chain, k, (double) v.getProperty (k, 0.0));
    }
    // the spectrum of the sound after the EQ for the panel: 256 points from 20 Hz to 20 kHz, in dB below full scale (-100 = nothing). Asking keeps the audio thread feeding the panel for the next 1.5 seconds.
    // The window is 8192 samples (steps of about 6 Hz at 48 kHz, so the bass has detail). Where several steps fall inside one point the loudest is taken; where a point is narrower than a step (the bass) the value is
    // blended smoothly between the neighbouring steps, so the line has no flat shelves.
    juce::var eqSpectrum (int chain)
    {
        auto* p = eqPanelOf (chain); if (p == nullptr) return {};
        p->scopeUntilMs.store (juce::Time::getMillisecondCounterHiRes() + 1500.0);
        constexpr int N = 8192, kPoints = 256; static thread_local juce::dsp::FFT fft (13); std::vector<float> data (2 * N, 0.0f);
        const int pos = p->ringPos.load (std::memory_order_acquire);
        for (int i = 0; i < N; ++i) { const float w = 0.5f - 0.5f * std::cos (juce::MathConstants<float>::twoPi * (float) i / (float) (N - 1)); data[(size_t) i] = p->ring[(size_t) ((pos - N + i) & (EqPanel::kRing - 1))].load (std::memory_order_relaxed) * w; }
        fft.performFrequencyOnlyForwardTransform (data.data());
        std::vector<double> db ((size_t) (N / 2)); for (int b = 0; b < N / 2; ++b) db[(size_t) b] = juce::jmax (-100.0, 20.0 * std::log10 (juce::jmax (1.0e-9, (double) data[(size_t) b] * 2.0 / ((double) N * 0.5))));
        juce::Array<juce::var> out; const double binHz = sampleRate / (double) N;
        for (int k = 0; k < kPoints; ++k)
        {
            const double lo = 20.0 * std::pow (1000.0, (double) k / kPoints), hi = 20.0 * std::pow (1000.0, (double) (k + 1) / kPoints), mid = 20.0 * std::pow (1000.0, ((double) k + 0.5) / kPoints);
            const int b0 = juce::jlimit (1, N / 2 - 1, (int) std::ceil (lo / binHz)), b1 = juce::jlimit (1, N / 2 - 1, (int) std::floor (hi / binHz)); double v;
            if (b1 > b0) { v = -100.0; for (int b = b0; b <= b1; ++b) v = juce::jmax (v, db[(size_t) b]); }
            else
            {
                const double x = juce::jlimit (1.0, (double) (N / 2 - 3), mid / binHz); const int i = (int) x; const double f = x - (double) i;
                const double a = db[(size_t) (i - 1)], b = db[(size_t) i], c = db[(size_t) (i + 1)], d = db[(size_t) (i + 2)];
                v = b + 0.5 * f * (c - a + f * (2.0 * a - 5.0 * b + 4.0 * c - d + f * (3.0 * (b - c) + d - a)));      // (a Catmull-Rom curve through the four neighbouring steps)
            }
            out.add (juce::jlimit (-100.0, 6.0, v));
        }
        return juce::var (out);
    }
    // ---- THE ARCHIE SOUND (the end of the master section): amOn, amDrive (0 to 1) and amCeil (-3 to 0 dB). Off in an engine that has not been told otherwise; a new song and the starter song switch it on. ----
    juce::var masterSoundParam (const juce::String& key)
    {
        if (master == nullptr) return {};
        if (key == "amOn") return master->amOn.load(); if (key == "amDrive") return (double) master->amDrive.load(); if (key == "amCeil") return (double) master->amCeil.load();
        if (key == "amCut") return master->amCut.load(); if (key == "amPush") return (double) master->amPush.load(); if (key == "amBass") return (double) master->amBass.load(); if (key == "amTone") return (double) master->amTone.load(); if (key == "amTarget") return (double) master->amTarget.load();
        return {};
    }
    bool applyMasterSoundParam (const juce::String& key, const juce::var& v)
    {
        if (master == nullptr) return false;
        if (key == "amOn") master->amOn.store ((bool) v); else if (key == "amCut") master->amCut.store ((bool) v); else if (key == "amDrive") master->amDrive.store ((float) juce::jlimit (0.0, 1.0, (double) v));
        else if (key == "amCeil") master->amCeil.store ((float) juce::jlimit ((double) ArchieSound::kCeilMin, (double) ArchieSound::kCeilMax, (double) v));
        else if (key == "amPush") master->amPush.store ((float) juce::jlimit (0.0, 18.0, (double) v)); else if (key == "amBass") master->amBass.store ((float) juce::jlimit (0.0, 6.0, (double) v)); else if (key == "amTone") master->amTone.store ((float) juce::jlimit (-3.0, 3.0, (double) v));
        else if (key == "amTarget") { const double t = (double) v; master->amTarget.store (t > -1.0 ? 0.0f : (float) juce::jlimit (-23.0, -6.0, t)); } else return false;
        stateDirty.store (true); return true;
    }
    bool editMasterSound (const juce::String& key, const juce::var& v)
    {
        if (key == "amResetLufs" && master != nullptr) { master->lm.resetIntegrated(); return true; }      // start the integrated loudness again (not saved, not undone)
        if (master == nullptr || masterSoundParam (key).isVoid()) return false;
        const juce::var before = masterSoundParam (key); if (! applyMasterSoundParam (key, v)) return false;
        const juce::var after = masterSoundParam (key);
        if (recordingMoves())
        {
            static const char* keys[] = { "amPush", "amDrive", "amCeil", "amBass", "amTone", "amOn", "amCut" };
            for (int i = 0; i < 7; ++i) if (key == keys[i]) { soundHandUntilMs[9 + i] = juce::Time::getMillisecondCounterHiRes() + 400.0; autoRecWrite (kMasterChain, 9 + i, -1, -1, transport.posBeats.load(), (double) after); return true; }
        }
        if (before == after) return true;
        undo.record (juce::String ("Archie sound " + juce::String (key == "amOn" ? "switch" : key == "amCut" ? "20 Hz cut" : key == "amDrive" ? "drive" : key == "amCeil" ? "ceiling" : key == "amPush" ? "push" : key == "amBass" ? "bass" : key == "amTone" ? "tone" : "loudness target")).toStdString(), key == "amOn" ? std::string() : ("s:" + key).toStdString(),
                     [this, key, before] { applyMasterSoundParam (key, before); }, [this, key, after] { applyMasterSoundParam (key, after); });
        return true;
    }
    juce::var masterSoundVar()
    {
        auto* o = new juce::DynamicObject(); for (const char* k : { "amOn", "amCut", "amDrive", "amCeil", "amPush", "amBass", "amTone", "amTarget" }) o->setProperty (k, masterSoundParam (k)); return juce::var (o);
    }
    void loadMasterSound (const juce::var& p)      // a song that has none (one made before the Archie sound) keeps its master as it was: off
    {
        if (master == nullptr) return; const auto s = p.getProperty ("masterSound", {});
        applyMasterSoundParam ("amOn", s.isObject() ? (bool) s.getProperty ("amOn", false) : false); applyMasterSoundParam ("amCut", s.isObject() ? (bool) s.getProperty ("amCut", true) : true); applyMasterSoundParam ("amDrive", s.isObject() ? (double) s.getProperty ("amDrive", 0.25) : 0.25); applyMasterSoundParam ("amCeil", s.isObject() ? (double) s.getProperty ("amCeil", -1.0) : -1.0);
        for (const char* k : { "amPush", "amBass", "amTone", "amTarget" }) applyMasterSoundParam (k, s.isObject() ? (double) s.getProperty (k, 0.0) : 0.0);
    }
    // the master compressor: one setting at a time, each one undoable (the knobs merge while dragged)
    juce::var masterCompParam (const juce::String& key)
    {
        if (master == nullptr) return {};
        if (key == "mcOn") return master->cOn.load(); if (key == "mcPeak") return master->cPeak.load();
        if (key == "mcThr") return (double) master->cThr.load(); if (key == "mcRatio") return (double) master->cRatio.load(); if (key == "mcAtk") return (double) master->cAtk.load();
        if (key == "mcRel") return (double) master->cRel.load(); if (key == "mcMakeup") return (double) master->cMake.load();
        return {};
    }
    bool applyMasterCompParam (const juce::String& key, const juce::var& v)
    {
        if (master == nullptr) return false;
        if (key == "mcOn") master->cOn.store ((bool) v); else if (key == "mcPeak") master->cPeak.store ((bool) v);
        else if (key == "mcThr") master->cThr.store ((float) juce::jlimit ((double) CompStage::kThrMin, (double) CompStage::kThrMax, (double) v));
        else if (key == "mcRatio") master->cRatio.store ((float) juce::jlimit ((double) CompStage::kRatioMin, (double) CompStage::kRatioMax, (double) v));
        else if (key == "mcAtk") master->cAtk.store ((float) juce::jlimit (0.0001, 0.03, (double) v));
        else if (key == "mcRel") master->cRel.store ((float) juce::jlimit ((double) CompStage::kRelMin, (double) CompStage::kRelMax, (double) v));
        else if (key == "mcMakeup") master->cMake.store ((float) juce::jlimit (0.0, (double) CompStage::kMakeMax, (double) v));
        else return false;
        stateDirty.store (true); return true;
    }
    bool editMasterCompParam (const juce::String& key, const juce::var& v)
    {
        if (master == nullptr || masterCompParam (key).isVoid()) return false;
        const juce::var before = masterCompParam (key); if (! applyMasterCompParam (key, v)) return false;
        const juce::var after = masterCompParam (key); if (before == after) return true;
        const bool dragged = key != "mcOn" && key != "mcPeak";
        const juce::String what = key == "mcOn" ? "switch" : key == "mcPeak" ? "detector" : key == "mcThr" ? "threshold" : key == "mcRatio" ? "ratio" : key == "mcAtk" ? "attack" : key == "mcRel" ? "release" : "make-up";
        undo.record (("Master compressor " + what).toStdString(), dragged ? ("m:" + key).toStdString() : std::string(),
                     [this, key, before] { applyMasterCompParam (key, before); }, [this, key, after] { applyMasterCompParam (key, after); });
        return true;
    }
    juce::var masterCompVar()
    {
        auto* o = new juce::DynamicObject();
        for (const char* k : { "mcOn", "mcPeak", "mcThr", "mcRatio", "mcAtk", "mcRel", "mcMakeup" }) o->setProperty (k, masterCompParam (k));
        return juce::var (o);
    }
    juce::var controlRoomParam (const juce::String& key)
    {
        if (key == "crDb") return (double) crDb.load(); if (key == "crDim") return crDim.load(); if (key == "crPair") return crPair.load();
        return {};
    }
    bool applyControlRoomParam (const juce::String& key, const juce::var& v)
    {
        if (key == "crDb") crDb.store ((float) juce::jlimit (-90.0, 6.0, (double) v)); else if (key == "crDim") crDim.store ((bool) v); else if (key == "crPair") crPair.store (juce::jlimit (0, 8, (int) v)); else return false;
        stateDirty.store (true); return true;
    }
    bool editControlRoomParam (const juce::String& key, const juce::var& v)
    {
        if (controlRoomParam (key).isVoid()) return false;
        const juce::var before = controlRoomParam (key); if (! applyControlRoomParam (key, v)) return false;
        const juce::var after = controlRoomParam (key); if (before == after) return true;
        undo.record (("Control room " + juce::String (key == "crDb" ? "level" : key == "crDim" ? "Dim" : "output")).toStdString(), key == "crDb" ? std::string ("cr:level") : std::string(),
                     [this, key, before] { applyControlRoomParam (key, before); }, [this, key, after] { applyControlRoomParam (key, after); });
        return true;
    }
    juce::var controlRoomVar() { auto* o = new juce::DynamicObject(); for (const char* k : { "crDb", "crDim", "crPair" }) o->setProperty (k, controlRoomParam (k)); return juce::var (o); }
    void loadControlRoom (const juce::var& p)
    {
        const auto c = p.getProperty ("controlRoom", {});
        auto num = [&] (const char* k, double d) { const auto v = c.getProperty (k, {}); return v.isVoid() ? d : (double) v; };
        crDb.store ((float) juce::jlimit (-90.0, 6.0, num ("crDb", 0.0))); crDim.store (num ("crDim", 0.0) != 0.0); crPair.store (juce::jlimit (0, 8, (int) num ("crPair", 0.0)));
    }
    void loadMasterComp (const juce::var& p)
    {
        if (master == nullptr) return;
        const auto mc = p.getProperty ("masterComp", {});
        auto num = [&] (const char* k, double d) { const auto v = mc.getProperty (k, {}); return v.isVoid() ? d : (double) v; };          // a song that has none (or an empty one) starts from the defaults
        master->cOn.store (num ("mcOn", 0.0) != 0.0); master->cPeak.store (num ("mcPeak", 0.0) != 0.0);
        master->cThr.store ((float) juce::jlimit ((double) CompStage::kThrMin, (double) CompStage::kThrMax, num ("mcThr", -12.0)));
        master->cRatio.store ((float) juce::jlimit ((double) CompStage::kRatioMin, (double) CompStage::kRatioMax, num ("mcRatio", 2.0)));
        master->cAtk.store ((float) juce::jlimit (0.0001, 0.03, num ("mcAtk", 0.01)));
        master->cRel.store ((float) juce::jlimit ((double) CompStage::kRelMin, (double) CompStage::kRelMax, num ("mcRel", 0.3)));
        master->cMake.store ((float) juce::jlimit (0.0, (double) CompStage::kMakeMax, num ("mcMakeup", 0.0)));
    }
    void editMasterDb (double db)
    {
        if (master == nullptr) return;
        const float before = master->gainDb.load(), after = (float) juce::jlimit (-90.0, 12.0, db);
        master->gainDb.store (after); stateDirty.store (true);
        if (before == after) return;
        undo.record ("Master level", "master", [this, before] { if (master) master->gainDb.store (before); stateDirty.store (true); },
                                               [this, after]  { if (master) master->gainDb.store (after);  stateDirty.store (true); });
    }
    // a send's own switch, level and PRE/POST, each with undo (one track's send only)
    void editSendOn (int track, int s, bool on)
    {
        auto* t = trackAt (track); if (t == nullptr || s < 0 || s >= kSends) return;
        const bool b = t->sendOn[s].load(); if (b == on) return;
        setSendOn (*t, s, on);
        undo.record ((t->name + " send " + juce::String (s + 1) + (on ? " on" : " off")).toStdString(), {},
                     [this, track, s, b] { if (auto* x = trackAt (track)) setSendOn (*x, s, b); }, [this, track, s, on] { if (auto* x = trackAt (track)) setSendOn (*x, s, on); });
    }
    void editSendLevel (int track, int s, float db)
    {
        auto* t = trackAt (track); if (t == nullptr || s < 0 || s >= kSends) return;
        const float b = t->sendLvl[s].load(); setSendLevel (*t, s, db); const float a = t->sendLvl[s].load(); if (b == a) return;
        if (transport.recording.load() && transport.playing.load()) autoRecFromControl (track, "sendLvl", (double) a, s);          // while recording, a move of the send knob is written into that send's lane
        undo.record ((t->name + " send " + juce::String (s + 1)).toStdString(), ("sl:" + juce::String (track) + ":" + juce::String (s)).toStdString(),
                     [this, track, s, b] { if (auto* x = trackAt (track)) setSendLevel (*x, s, b); }, [this, track, s, a] { if (auto* x = trackAt (track)) setSendLevel (*x, s, a); });
    }
    void editSendPre (int track, int s, bool pre)
    {
        auto* t = trackAt (track); if (t == nullptr || s < 0 || s >= kSends) return;
        const bool b = t->sendPre[s].load(); if (b == pre) return;
        setSendPre (*t, s, pre);
        undo.record ((t->name + " send " + juce::String (s + 1) + (pre ? " pre" : " post")).toStdString(), {},
                     [this, track, s, b] { if (auto* x = trackAt (track)) setSendPre (*x, s, b); }, [this, track, s, pre] { if (auto* x = trackAt (track)) setSendPre (*x, s, pre); });
    }
    void editSend (int track, int s, float db, bool pre)
    {
        auto* t = trackAt (track);
        if (t == nullptr || s < 0 || s >= kSends) return;
        const float bdb = t->sendDb[s].load(); const bool bpre = t->sendPre[s].load();
        setSend (*t, s, db, pre);
        const float adb = t->sendDb[s].load(); const bool apre = t->sendPre[s].load();
        if (bdb == adb && bpre == apre) return;
        undo.record ((t->name + " send " + juce::String (s + 1)).toStdString(), ("s:" + juce::String (track) + ":" + juce::String (s)).toStdString(),
                     [this, track, s, bdb, bpre] { if (auto* x = trackAt (track)) setSend (*x, s, bdb, bpre); },
                     [this, track, s, adb, apre] { if (auto* x = trackAt (track)) setSend (*x, s, adb, apre); });
    }

    // clips: the arrangement before and after (instances are small; the content is shared, not copied)
    void recordArrChange (int track, const std::string& label, const std::shared_ptr<Insts>& before, const std::string& key = {}, double windowMs = -1.0)
    {
        auto* t = trackAt (track);
        if (t == nullptr) return;
        auto lean = [] (const std::shared_ptr<Insts>& v) { auto c = std::make_shared<Insts> (*v); for (auto& i : *c) { i.warped = nullptr; i.warpKey = 0; i.pitched = nullptr; i.pitchKey = 0; } return c; };       // (the renders are made again on undo: they are big)
        auto after = lean (instsOf (*t)); auto b2 = lean (before);
        undo.record (label, key, [this, track, b2] { putArr (track, b2); }, [this, track, after] { putArr (track, after); }, windowMs);
    }
    void editSetClip (int track, const juce::var& v)
    {
        auto* t = trackAt (track); if (t == nullptr) return;
        auto before = instsOf (*t);
        setClip (track, v);
        recordArrChange (track, "Edit clip", before);
    }
    bool editClearClip (int track)
    {
        auto* t = trackAt (track); if (t == nullptr) return false;
        auto before = instsOf (*t);
        putArr (track, std::make_shared<Insts>());
        recordArrChange (track, "Clear clip", before);
        return true;
    }
    // which track holds this instance (-1 = none)
    int trackOfInstance (int instId, int* index = nullptr)
    {
        for (int ti = 0; ti < trackCount.load(); ++ti) { const auto& v = tracks[(size_t) ti]->arrSnap.load()->insts; for (size_t i = 0; i < v.size(); ++i) if (v[i].id == instId) { if (index) *index = (int) i; return ti; } }
        return -1;
    }
    std::shared_ptr<const Clip> findContent (int contentId)
    {
        for (int ti = 0; ti < trackCount.load(); ++ti) for (const auto& in : tracks[(size_t) ti]->arrSnap.load()->insts) if (in.content && in.content->id == contentId) return in.content;
        return nullptr;
    }
    // put a clip on a track's timeline: new content from `clip`, or an instance of content that already exists (`contentId`), so the two stay one clip. Returns the new instance's id, or 0.
    int editAddInstance (int track, double start, double length, double offset, const juce::var& clip, int contentId)
    {
        auto* t = trackAt (track); if (t == nullptr) return 0;
        ClipInst in; in.id = nextInstanceId.fetch_add (1);
        if (contentId > 0) { in.content = findContent (contentId); if (in.content == nullptr) return 0; }
        else in.content = contentFromVar (clip);
        in.start = juce::jmax (0.0, start); in.length = juce::jmax (0.0, length); in.offset = juce::jmax (0.0, offset);
        if (in.length <= 0.0) in.length = in.content->lengthBeats;                                   // left out = one pass of the clip
        auto before = instsOf (*t); auto ins = std::make_shared<Insts> (*before); ins->push_back (in);
        putArr (track, ins); recordArrChange (track, "Add clip", before);
        return in.id;
    }
    // Put a sound file on a track as a clip at a beat. Its length is the sound's own length at the tempo there. Returns the new instance's id, or 0 (and says why in `err`).
    int editImportAudio (int track, const juce::File& f, double startBeat, juce::String& err, double offsetSec = 0.0)
    {
        auto* t = trackAt (track); if (t == nullptr) { err = "no such track"; return 0; }
        auto data = loadAudioData (f, err); if (data == nullptr) return 0;
        auto c = std::make_shared<Clip>(); c->id = nextContentId.fetch_add (1); c->lengthBeats = 0.0; c->audio = data;
        ClipInst in; in.id = nextInstanceId.fetch_add (1); in.content = c; in.start = juce::jmax (0.0, startBeat);
        const auto* tm = transport.tempoNow();
        in.offset = juce::jlimit (0.0, juce::jmax (0.0, data->durationSec() - 0.01), offsetSec);
        in.length = juce::jmax (0.01, tm->beatAt (tm->secondsAt (in.start) + data->durationSec() - in.offset) - in.start);
        auto before = instsOf (*t); auto ins = std::make_shared<Insts> (*before); ins->push_back (in);
        putArr (track, ins); recordArrChange (track, "Import " + data->name.toStdString(), before);
        return in.id;
    }
    // move, resize or re-offset an instance. Dragging it is one undo step.
    bool editSetInstance (int instId, const juce::var& p)
    {
        int idx = -1; const int track = trackOfInstance (instId, &idx); if (track < 0) return false;
        auto before = instsOf (*tracks[(size_t) track]); auto ins = std::make_shared<Insts> (*before);
        auto& in = (*ins)[(size_t) idx];
        if (p.hasProperty ("trimStart"))       // crop the front: the clip now begins at this beat, showing what was at that beat before (the end stays where it is)
        {
            const double ns = juce::jmax (0.0, (double) p.getProperty ("trimStart", in.start)), end = in.length > 0.0 ? in.start + in.length : 0.0;
            if (end <= 0.0 || ns < end - 0.01)
            {
                const auto* tm = transport.tempoNow(); (void) tm;
                double delta = in.content && in.content->audio ? 0.0 : ns - in.start;      // seconds into the sound (a warped clip: its own beats), or beats into the notes
                if (in.content && in.content->audio) in.offset = audioPosAt (in, ns);
                else in.offset = juce::jmax (0.0, in.offset + delta);
                in.start = ns; if (end > 0.0) in.length = end - ns;
            }
        }
        if (p.hasProperty ("start"))  in.start  = juce::jmax (0.0, (double) p.getProperty ("start", in.start));
        if (p.hasProperty ("length")) in.length = juce::jmax (0.0, (double) p.getProperty ("length", in.length));
        if (p.hasProperty ("offset")) in.offset = juce::jmax (0.0, (double) p.getProperty ("offset", in.offset));
        if (p.hasProperty ("start") || p.hasProperty ("trimStart")) in.fadeIn = juce::jmax (in.fadeIn, 0.002);          // an edge that has just moved is a new edge: it gets a small fade so it cannot click
        if (p.hasProperty ("start") || p.hasProperty ("length")) in.fadeOut = juce::jmax (in.fadeOut, 0.002);
        if (in.content && in.content->audio)
        {
            if (p.hasProperty ("srcBpm")) in.srcBpm = juce::jlimit (20.0, 400.0, (double) p.getProperty ("srcBpm", 120.0));
            if (p.hasProperty ("sourceBeats") && (double) p.getProperty ("sourceBeats", 0.0) > 0.0 && in.content->audio->durationSec() > 0.0) in.srcBpm = juce::jlimit (20.0, 400.0, (double) p.getProperty ("sourceBeats", 0.0) * 60.0 / in.content->audio->durationSec());   // "this whole sound is N beats long"
            if (p.hasProperty ("repitch")) in.repitch = (bool) p.getProperty ("repitch", false);
            if (p.hasProperty ("transpose")) in.transpose = std::round (juce::jlimit (-24.0, 24.0, (double) p.getProperty ("transpose", 0.0)) * 100.0) / 100.0;
            if (p.hasProperty ("keepFormant")) in.keepFormant = (bool) p.getProperty ("keepFormant", true);
            if (p.hasProperty ("warp")) { in.warp = (bool) p.getProperty ("warp", false); if (in.warp && in.srcBpm <= 0.0) in.srcBpm = transport.tempoNow()->bpmAt (in.start); }       // switched on with no tempo given: assume the sound already fits (nothing changes until the tempo is set)
        }
        if (p.hasProperty ("colour")) in.colour = juce::jlimit (-1, kPaletteSize - 1, (int) p.getProperty ("colour", -1));
        if (p.hasProperty ("muted")) in.muted = (bool) p.getProperty ("muted", false);
        if (p.hasProperty ("groove")) in.groove = juce::jlimit (-2, kGrooveChans - 1, (int) p.getProperty ("groove", -2));
        if (p.hasProperty ("label")) in.label = p.getProperty ("label", {}).toString().substring (0, 60);
        const bool colouring = (p.hasProperty ("colour") || p.hasProperty ("muted") || p.hasProperty ("label")) && ! p.hasProperty ("start") && ! p.hasProperty ("length");
        const bool warping = p.hasProperty ("warp") || p.hasProperty ("srcBpm") || p.hasProperty ("sourceBeats") || p.hasProperty ("repitch");
        const bool pitching = p.hasProperty ("transpose") || p.hasProperty ("keepFormant");
        const bool shaping = warping || pitching || p.hasProperty ("gainDb") || p.hasProperty ("fadeIn") || p.hasProperty ("fadeOut") || p.hasProperty ("reverse");
        if (p.hasProperty ("gainDb"))  in.gainDb  = (float) juce::jlimit (-60.0, 24.0, (double) p.getProperty ("gainDb", 0.0));
        if (p.hasProperty ("fadeIn"))  in.fadeIn  = juce::jlimit (0.0005, 60.0, (double) p.getProperty ("fadeIn", 0.002));
        if (p.hasProperty ("fadeOut")) in.fadeOut = juce::jlimit (0.0005, 60.0, (double) p.getProperty ("fadeOut", 0.002));
        if (p.hasProperty ("reverse")) in.reverse = (bool) p.getProperty ("reverse", false);
        putArr (track, ins); recordArrChange (track, colouring ? p.hasProperty ("muted") ? "Mute clip" : p.hasProperty ("label") ? "Rename clip" : "Clip colour" : warping ? "Change warp" : pitching ? "Transpose clip" : shaping ? "Change clip sound" : "Move clip", before, std::string (shaping ? (warping ? "shape-warp:" : pitching ? "shape-pitch:" : p.hasProperty ("gainDb") ? "shape-gain:" : p.hasProperty ("fadeIn") ? "shape-fi:" : p.hasProperty ("fadeOut") ? "shape-fo:" : "shape-rev:") : "inst:") + std::to_string (instId));
        return true;
    }
    bool editRemoveInstance (int instId)
    {
        int idx = -1; const int track = trackOfInstance (instId, &idx); if (track < 0) return false;
        auto before = instsOf (*tracks[(size_t) track]); auto ins = std::make_shared<Insts> (*before);
        ins->erase (ins->begin() + idx);
        putArr (track, ins); recordArrChange (track, "Delete clip", before);
        return true;
    }
    // a copy of an instance placed at `start`: it shares the content (change one, all change) unless `shared` is false, which gives it content of its own
    int editDuplicateInstance (int instId, double start, bool shared, int toTrack)
    {
        int idx = -1; const int track = trackOfInstance (instId, &idx); if (track < 0) return 0;
        const int dest = toTrack >= 0 ? toTrack : track; auto* dt = trackAt (dest); if (dt == nullptr) return 0;
        const ClipInst src = tracks[(size_t) track]->arrSnap.load()->insts[(size_t) idx];
        ClipInst in = src; in.id = nextInstanceId.fetch_add (1); in.start = juce::jmax (0.0, start);
        if (src.length <= 0.0 && src.content) in.length = src.content->lengthBeats;               // a copy of a for-ever pattern is one pass of it
        if (! shared) { auto c = std::make_shared<Clip> (*src.content); c->id = nextContentId.fetch_add (1); in.content = c; }
        auto before = instsOf (*dt); auto ins = std::make_shared<Insts> (*before); ins->push_back (in);
        putArr (dest, ins); recordArrChange (dest, shared ? "Duplicate clip (shared)" : "Duplicate clip", before);
        return in.id;
    }
    // this instance gets content of its own, so editing it no longer changes the others
    bool editMakeUnique (int instId)
    {
        int idx = -1; const int track = trackOfInstance (instId, &idx); if (track < 0) return false;
        auto before = instsOf (*tracks[(size_t) track]); auto ins = std::make_shared<Insts> (*before);
        auto c = std::make_shared<Clip> (*(*ins)[(size_t) idx].content); c->id = nextContentId.fetch_add (1); (*ins)[(size_t) idx].content = c;
        putArr (track, ins); recordArrChange (track, "Make clip unique", before);
        return true;
    }
    // cut an instance in two at a beat; both halves keep pointing at the same content
    int editSplitInstance (int instId, double at)
    {
        int idx = -1; const int track = trackOfInstance (instId, &idx); if (track < 0) return 0;
        auto before = instsOf (*tracks[(size_t) track]); auto ins = std::make_shared<Insts> (*before);
        ClipInst a = (*ins)[(size_t) idx];
        const double end = a.length > 0.0 ? a.start + a.length : kForever;
        if (at <= a.start + 1.0e-6 || at >= end - 1.0e-6) return 0;
        ClipInst b = a; b.id = nextInstanceId.fetch_add (1); b.start = at;
        if (a.content && a.content->audio) b.offset = audioPosAt (a, at);
        else b.offset = a.offset + (at - a.start);
        b.length = a.length > 0.0 ? (a.start + a.length) - at : 0.0;
        (*ins)[(size_t) idx].length = at - a.start; (*ins)[(size_t) idx].fadeOut = 0.0; b.fadeIn = 0.0;        // the sound is continuous across the cut, so the joint has no fade; the original's fades stay at the real ends
        b.fadeOut = a.fadeOut;
        ins->push_back (b);
        putArr (track, ins); recordArrChange (track, "Split clip", before);
        return b.id;
    }
    // Cut a clip into pieces that share its sound, at the places its hits are (mode "hits", param = sensitivity 0..1) or every `param` beats of the timeline (mode "grid").
    // The pieces together play exactly what the clip did (offsets are worked out from the tempo map; looping and reverse are followed), and each joint has no fade because the sound carries on across it.
    // One undo step. Returns how many pieces there are (2 or more), or 0 if nothing was cut.
    int editChopInstance (int instId, const juce::String& mode, double param)
    {
        int idx = -1; const int track = trackOfInstance (instId, &idx); if (track < 0) return 0;
        auto before = instsOf (*tracks[(size_t) track]); const ClipInst a = (*before)[(size_t) idx];
        if (a.content == nullptr || a.length <= 0.0) return 0;
        const double end = a.start + a.length; const auto* tm = transport.tempoNow(); std::vector<double> cuts;
        if (mode == "grid")
        {
            const double step = juce::jmax (0.0625, param);
            for (double b = std::ceil ((a.start + 1.0e-6) / step) * step; b < end - 1.0e-6; b += step) cuts.push_back (b);
        }
        else if (mode == "hits" && a.content->audio) { for (const auto& h : hitBeats (a, juce::jlimit (0.0, 1.0, param))) cuts.push_back (h.second); }
        else return 0;
        std::sort (cuts.begin(), cuts.end());
        std::vector<double> keep; double prev = a.start;
        for (double c : cuts) if (c - prev >= 0.02 && end - c >= 0.02) { keep.push_back (c); prev = c; }
        if (keep.empty()) return 0;
        auto ins = std::make_shared<Insts> (*before); ins->erase (ins->begin() + idx);
        std::vector<double> edges { a.start }; for (double c : keep) edges.push_back (c); edges.push_back (end);
        for (size_t i = 0; i + 1 < edges.size(); ++i)
        {
            ClipInst b = a; b.id = (i == 0) ? a.id : nextInstanceId.fetch_add (1); b.start = edges[i]; b.length = edges[i + 1] - edges[i];
            if (i > 0) { if (a.content->audio) b.offset = audioPosAt (a, edges[i]); else { const double L = a.content->lengthBeats; b.offset = L > 0.0 ? std::fmod (a.offset + (edges[i] - a.start), L) : a.offset + (edges[i] - a.start); } }
            if (i > 0) b.fadeIn = 0.0; if (i + 2 < edges.size()) b.fadeOut = 0.0;
            ins->push_back (b);
        }
        putArr (track, ins); recordArrChange (track, "Chop clip", before);
        return (int) edges.size() - 1;
    }
    // Join a clip with the NEXT clip on its track. Two pieces of the same sound that carry on from each other (what chopping makes) become one clip again; two note clips become one note clip holding
    // both sets of notes (the second one's notes moved to where they sit after the first); two different sounds cannot be joined. One undo step. Returns the surviving clip's id, or 0 with the reason.
    int editJoinInstance (int instId, juce::String& err)
    {
        int idx = -1; const int track = trackOfInstance (instId, &idx); if (track < 0) { err = "no such clip"; return 0; }
        auto before = instsOf (*tracks[(size_t) track]); std::vector<ClipInst> sorted (before->begin(), before->end());
        const ClipInst a = (*before)[(size_t) idx]; int nextIdx = -1; double bestStart = 1.0e30;
        for (size_t i = 0; i < before->size(); ++i) if ((int) i != idx && (*before)[i].start >= a.start + 1.0e-9 && (*before)[i].start < bestStart) { bestStart = (*before)[i].start; nextIdx = (int) i; }
        if (nextIdx < 0) { err = "there is no clip after this one on the track"; return 0; }
        const ClipInst b = (*before)[(size_t) nextIdx];
        if (a.length <= 0.0 || b.length <= 0.0 || ! a.content || ! b.content) { err = "a clip that goes on for ever cannot be joined"; return 0; }
        if ((a.content->audio != nullptr) != (b.content->audio != nullptr)) { err = "an audio clip and a note clip cannot be joined"; return 0; }
        auto ins = std::make_shared<Insts> (*before); ClipInst r = a;
        if (a.content->audio)
        {
            const bool same = a.content == b.content && a.reverse == b.reverse && a.warp == b.warp && std::abs (a.start + a.length - b.start) < 0.02 && std::abs (audioPosAt (a, b.start) - b.offset) < 0.01;
            if (! same) { err = "these are not two pieces of the same sound, one after the other"; return 0; }
            r.length = b.start + b.length - a.start; r.fadeOut = b.fadeOut;
        }
        else
        {
            const double la = a.content->lengthBeats, lb = b.content->lengthBeats;
            if (a.content == b.content && std::abs (a.start + a.length - b.start) < 0.02 && la > 0.0 && std::abs (std::fmod (a.offset + a.length - b.offset + 2.0 * la, la)) < 0.02) r.length = b.start + b.length - a.start;
            else
            {
                auto c = std::make_shared<Clip>(); c->id = nextContentId.fetch_add (1); c->lengthBeats = juce::jmax (0.25, b.start + b.length - a.start);
                for (const auto& d : { std::make_pair (a, 0.0), std::make_pair (b, b.start - a.start) })
                {
                    ClipInst piece = d.first; const double off = d.second; const double L = piece.content->lengthBeats; if (L <= 0.0) continue;
                    walkInstance (piece, piece.start, piece.start + piece.length, false,
                        [&] (const NoteEv& nt, double tb) { NoteEv n = nt; n.beat = tb - a.start; (void) off; const double endB = juce::jmin (c->lengthBeats, n.beat + n.len); n.len = juce::jmax (0.01, endB - n.beat); if (n.beat < c->lengthBeats) c->notes.push_back (n); },
                        [&] (const NoteEv&, double) {}, [&] (const NoteEv&) {});
                }
                r.content = c; r.offset = 0.0; r.length = c->lengthBeats;
            }
        }
        std::vector<size_t> drop { (size_t) idx, (size_t) nextIdx }; std::sort (drop.rbegin(), drop.rend()); for (auto d : drop) ins->erase (ins->begin() + (long) d);
        ins->push_back (r); putArr (track, ins); recordArrChange (track, "Join clips", before);
        return r.id;
    }
    // Turn the hits of an audio clip into notes: a new track with a note clip, one short note at each hit (louder hits are louder notes). The pitch is a fixed middle C -- this finds WHEN, not WHAT PITCH.
    // One undo step (the new track and its clip together). Returns the new track's number, or -1 with the reason.
    int editAudioToMidi (int instId, double sensitivity, juce::String& err)
    {
        int idx = -1; const int track = trackOfInstance (instId, &idx); if (track < 0) { err = "no such clip"; return -1; }
        const ClipInst a = (*instsOf (*tracks[(size_t) track]))[(size_t) idx];
        if (! a.content || ! a.content->audio || a.length <= 0.0) { err = "this is not an audio clip with a length"; return -1; }
        const auto hits = hitBeats (a, juce::jlimit (0.0, 1.0, sensitivity)); if (hits.empty()) { err = "no hits were found in this sound"; return -1; }
        if (trackCount.load() >= kTracks) { err = "that is the most tracks there can be"; return -1; }
        const int nBefore = trackCount.load(); if (! addTrack()) { err = "could not add a track"; return -1; }
        const int nt = trackCount.load() - 1; tracks[(size_t) nt]->name = (tracks[(size_t) track]->name + " MIDI").substring (0, 40);
        auto c = std::make_shared<Clip>(); c->id = nextContentId.fetch_add (1); const double bar = 4.0; c->lengthBeats = juce::jmax (bar, std::ceil ((a.length) / bar) * bar);
        for (const auto& h : hits) { NoteEv n; n.beat = h.second - a.start; n.len = 0.125; n.note = 60; n.vel = 100; n.ch = 1; if (n.beat >= 0.0 && n.beat < c->lengthBeats) c->notes.push_back (n); }
        ClipInst in; in.id = nextInstanceId.fetch_add (1); in.start = a.start; in.length = c->lengthBeats; in.content = c;
        auto ins = std::make_shared<Insts>(); ins->push_back (in); putArr (nt, ins); stateDirty.store (true);
        const juce::String nm = tracks[(size_t) nt]->name;
        undo.record ("Audio clip to MIDI", "", [this, nBefore] { while (trackCount.load() > nBefore) removeLastTrack(); },
                     [this, nm, ins] { if (addTrack()) { const int k = trackCount.load() - 1; tracks[(size_t) k]->name = nm; putArr (k, ins); } });
        return nt;
    }
    // Sounds in the recordings folder that no clip in this song uses. (Another saved song may still use one: the screen says so before anything is deleted.)
    std::vector<juce::File> unusedRecordings()
    {
        std::vector<juce::File> out; auto dir = recordingsDir != juce::File() ? recordingsDir : settingsDir.getChildFile ("recordings"); if (! dir.isDirectory()) return out;
        std::set<juce::String> used;
        for (int ti = 0; ti < trackCount.load(); ++ti) for (const auto& in : tracks[(size_t) ti]->arrSnap.load()->insts) if (in.content && in.content->audio) used.insert (juce::File (in.content->audio->path).getFullPathName().toLowerCase());
        for (const auto& f : dir.findChildFiles (juce::File::findFiles, false, "*.wav")) if (used.count (f.getFullPathName().toLowerCase()) == 0) out.push_back (f);
        return out;
    }

    // Move clip starts toward the nearest grid line: strength 1 puts them exactly on it, 0.5 half way. With `fill`, a clip that now runs into the next one is shortened to end where the next begins.
    // `ids` empty = every audio clip on the track. A clip that moved gets a small fade at both ends, since its edges are new edges. One undo step. Returns how many moved.
    int editQuantiseInstances (int track, std::vector<int> ids, double grid, double strength, bool fill)
    {
        auto* t = trackAt (track); if (t == nullptr || grid <= 0.0) return 0;
        auto before = instsOf (*t); auto ins = std::make_shared<Insts> (*before);
        strength = juce::jlimit (0.0, 1.0, strength); int moved = 0; std::vector<size_t> chosen;
        for (size_t i = 0; i < ins->size(); ++i)
        {
            const auto& in = (*ins)[i];
            const bool want = ids.empty() ? (in.content && in.content->audio) : std::find (ids.begin(), ids.end(), in.id) != ids.end();
            if (! want) continue;
            chosen.push_back (i);
            const double target = std::floor (in.start / grid + 0.5) * grid, ns = in.start + (target - in.start) * strength;
            if (std::abs (ns - in.start) > 1.0e-9) { (*ins)[i].start = juce::jmax (0.0, ns); (*ins)[i].fadeIn = juce::jmax ((*ins)[i].fadeIn, 0.002); (*ins)[i].fadeOut = juce::jmax ((*ins)[i].fadeOut, 0.002); ++moved; }
        }
        if (fill)
        {
            std::sort (chosen.begin(), chosen.end(), [&] (size_t x, size_t y) { return (*ins)[x].start < (*ins)[y].start; });
            for (size_t k = 0; k + 1 < chosen.size(); ++k)
            {
                auto& a = (*ins)[chosen[k]]; const auto& b = (*ins)[chosen[k + 1]];
                if (a.length > 0.0 && a.start + a.length > b.start + 1.0e-9) { a.length = juce::jmax (0.01, b.start - a.start); a.fadeOut = juce::jmax (a.fadeOut, 0.002); ++moved; }
            }
        }
        if (moved == 0) return 0;
        putArr (track, ins); recordArrChange (track, "Quantise clips", before);
        return moved;
    }

    // replace the notes of a clip's CONTENT: every instance of it, on any track, changes together
    bool editContent (int contentId, const juce::var& clip) { auto old = findContent (contentId); if (old == nullptr || old->audio) return false; auto nc = contentFromVar (clip); nc->id = contentId; return replaceContent (old, nc, "Edit notes"); }
    // Change a note clip's notes with a function (quantise, transpose...). `label` names the undo step; the same kind of edit within 700 ms (a drag, a held key) is ONE step.
    bool editNotes (int contentId, const std::string& label, const std::function<bool (Clip&)>& change)
    {
        auto old = findContent (contentId); if (old == nullptr || old->audio) return false;
        auto nc = std::make_shared<Clip> (*old); if (! change (*nc)) return false;
        return replaceContent (old, nc, label);
    }
    // Quantise the starts (and, with `lengths`, the ends) of notes toward the nearest grid line; strength 1 = exactly on it. `indices` empty = every note. A note that would land at or past the end of the clip stays inside it.
    bool editQuantiseNotes (int contentId, double grid, double strength, bool lengths, const std::vector<int>& indices)
    {
        if (grid <= 0.0) return false; strength = juce::jlimit (0.0, 1.0, strength);
        return editNotes (contentId, "Quantise notes", [&] (Clip& c)
        {
            bool any = false;
            for (size_t i = 0; i < c.notes.size(); ++i)
            {
                if (! indices.empty() && std::find (indices.begin(), indices.end(), (int) i) == indices.end()) continue;
                auto& n = c.notes[i]; const double end = n.beat + n.len, tgt = std::floor (n.beat / grid + 0.5) * grid, ns = n.beat + (tgt - n.beat) * strength;
                double ne = end; if (lengths) { const double et = std::floor (end / grid + 0.5) * grid; ne = end + (et - end) * strength; }
                const double nb = juce::jlimit (0.0, juce::jmax (0.0, c.lengthBeats - 0.0625), ns), nl = juce::jmax (0.0625, (lengths ? ne : end) - nb);
                if (std::abs (nb - n.beat) > 1.0e-9 || std::abs (nl - n.len) > 1.0e-9) { n.beat = nb; n.len = juce::jmin (nl, c.lengthBeats - nb); any = true; }
            }
            return any;
        });
    }
    bool editTransposeNotes (int contentId, int semitones, const std::vector<int>& indices)
    {
        if (semitones == 0) return false;
        return editNotes (contentId, "Transpose notes", [&] (Clip& c)
        {
            bool any = false;
            for (size_t i = 0; i < c.notes.size(); ++i) { if (! indices.empty() && std::find (indices.begin(), indices.end(), (int) i) == indices.end()) continue; const int nn = juce::jlimit (0, 127, c.notes[i].note + semitones); if (nn != c.notes[i].note) { c.notes[i].note = nn; any = true; } }
            return any;
        });
    }
    bool replaceContent (const std::shared_ptr<const Clip>& old, const std::shared_ptr<Clip>& nc, const std::string& label)
    {
        const int contentId = old->id; nc->id = contentId;
        std::vector<std::pair<int, std::shared_ptr<Insts>>> was, now;
        for (int ti = 0; ti < trackCount.load(); ++ti)
        {
            auto b = instsOf (*tracks[(size_t) ti]); bool touched = false; auto a = std::make_shared<Insts> (*b);
            for (auto& in : *a) if (in.content == old) { in.content = nc; touched = true; }
            if (touched) { was.push_back ({ ti, b }); now.push_back ({ ti, a }); }
        }
        for (auto& p : now) putArr (p.first, p.second);
        undo.record (label, label == "Edit notes" ? "notes:" + std::to_string (contentId) : std::string(), [this, was] { for (auto& p : was) putArr (p.first, p.second); }, [this, now] { for (auto& p : now) putArr (p.first, p.second); });
        return true;
    }

    // tempo and time signature: the whole entry list before and after
    static bool sameTempo (const std::vector<TempoEntry>& a, const std::vector<TempoEntry>& b)
    {
        if (a.size() != b.size()) return false;
        for (size_t i = 0; i < a.size(); ++i) if (a[i].beat != b[i].beat || a[i].bpm != b[i].bpm || a[i].tsNum != b[i].tsNum || a[i].tsDen != b[i].tsDen) return false;
        return true;
    }
    void editTempo (const std::string& label, const std::string& key, const std::function<void()>& change)
    {
        auto before = transport.tempoEntries(); change(); auto after = transport.tempoEntries();
        stateDirty.store (true);
        if (! sameTempo (before, after)) rewarpAll();
        if (sameTempo (before, after)) return;
        undo.record (label, key, [this, before] { transport.setTempoEntries (before); stateDirty.store (true); rewarpAll(); },
                                 [this, after]  { transport.setTempoEntries (after);  stateDirty.store (true); rewarpAll(); });
    }

    // tracks
    struct PlugSnap { juce::PluginDescription desc; juce::String state; bool bypass = false; bool internal = false; };
    struct TrackSnap
    {
        juce::String name, filePath; int colour = -1; int source = 0; float gainDb = 0, pan = 0; bool mute = false, solo = false;
        std::shared_ptr<Insts> insts; float sendDb[kSends] {}, sendLvl[kSends] {}; bool sendPre[kSends] {}, sendOn[kSends] {}; std::vector<std::shared_ptr<PlugSnap>> plugs;
    };
    bool snapSlot (int chain, int slot, PlugSnap& out)
    {
        auto* t = chainAt (chain);
        if (t == nullptr || slot < 0 || slot >= (int) t->slots.size()) return false;
        auto& s = *t->slots[(size_t) slot];
        out.desc = s.desc; out.bypass = s.bypass.load(); out.internal = s.desc.pluginFormatName == "Internal";
        juce::MemoryBlock mb; s.proc->getStateInformation (mb); out.state = mb.toBase64Encoding();
        return true;
    }
    // loads a plug-in onto a chain and puts it back at its old place with its old setting; `then` runs when it is in (or failed)
    void restorePlugin (int chain, int slot, std::shared_ptr<PlugSnap> sn, std::function<void()> then = {})
    {
        addPluginAsync (chain, sn->desc, sn->state, [this, chain, slot, sn, then] (juce::String err)
        {
            if (err.isEmpty())
                if (auto* t = chainAt (chain))
                {
                    const int last = (int) t->slots.size() - 1, to = juce::jlimit (0, last, slot);
                    if (to < last) std::rotate (t->slots.begin() + to, t->slots.end() - 1, t->slots.end());
                    t->slots[(size_t) to]->bypass.store (sn->bypass);
                    publishPlan(); stateDirty.store (true);
                }
            if (then) then();
        });
    }
    void restoreTrack (const std::shared_ptr<TrackSnap>& sn, std::function<void()> done = {})
    {
        if (! addTrack()) return;
        const int idx = trackCount.load() - 1;
        auto& t = *tracks[(size_t) idx];
        t.colour.store (sn->colour); t.name = sn->name; t.source.store (sn->source); t.gainDb.store (sn->gainDb); t.pan.store (sn->pan); t.mute.store (sn->mute); t.solo.store (sn->solo);
        updateSolo();
        if (sn->insts) putArr (idx, sn->insts);
        for (int s = 0; s < kSends; ++s) { t.sendLvl[s].store (sn->sendLvl[s]); t.sendPre[s].store (sn->sendPre[s]); t.sendOn[s].store (sn->sendOn[s]); applySend (t, s, true); }
        if (sn->filePath.isNotEmpty()) { const int src = sn->source; if (loadAudioFile (idx, juce::File (sn->filePath))) t.source.store (src); }
        auto next = std::make_shared<std::function<void (size_t)>>();
        *next = [this, idx, sn, next, done] (size_t i) { if (i < sn->plugs.size()) restorePlugin (idx, (int) i, sn->plugs[i], [next, i] { (*next) (i + 1); }); else if (done) done(); };
        (*next) (0);
        stateDirty.store (true);
    }
    bool editAddTrack()
    {
        if (! addTrack()) return false;
        undo.record ("Add track", "", [this] { removeLastTrack(); }, [this] { addTrack(); });
        return true;
    }
    // Moves the track at `from` to position `to` (the ones between shift by one). The track objects stay whole (their plug-ins, clips, automation and graph nodes
    // go with them); only their places and numbers change, then a new plan is published so the audio thread picks up the new order.
    void moveTrackTo (int from, int to)
    {
        const int n = trackCount.load(); if (from == to || from < 0 || to < 0 || from >= n || to >= n) return;
        const int mt = monitorTrack.load();
        if (from < to) std::rotate (tracks.begin() + from, tracks.begin() + from + 1, tracks.begin() + to + 1);
        else           std::rotate (tracks.begin() + to, tracks.begin() + from, tracks.begin() + from + 1);
        const int lo = juce::jmin (from, to), hi = juce::jmax (from, to);
        for (int i = lo; i <= hi; ++i)
        {
            auto& t = *tracks[(size_t) i]; t.index = i;
            for (int s = 0; s < kSends; ++s) if (t.sendProc[s] != nullptr) renderer.ensureSendBuffer (i, s);
        }
        monitorTrack.store (mt == from ? to : (from < to ? (mt > from && mt <= to ? mt - 1 : mt) : (mt >= to && mt < from ? mt + 1 : mt)));
        publishPlan(); quiesce(); updateTargets(); stateDirty.store (true);
    }
    // Removes any track (not only the last): it is moved to the end, then removed the way the last track always was. Undo puts it back where it was.
    bool editRemoveTrack (int idx = -1)
    {
        const int n = trackCount.load();
        if (n <= 0) return false;
        if (idx < 0) idx = n - 1;
        if (idx >= n) return false;
        auto& t = *tracks[(size_t) idx];
        auto sn = std::make_shared<TrackSnap>();
        sn->colour = t.colour.load(); sn->name = t.name; sn->filePath = t.filePath; sn->source = t.source.load(); sn->gainDb = t.gainDb.load(); sn->pan = t.pan.load(); sn->mute = t.mute.load(); sn->solo = t.solo.load();
        sn->insts = instsOf (t);
        for (int s = 0; s < kSends; ++s) { sn->sendDb[s] = t.sendDb[s].load(); sn->sendLvl[s] = t.sendLvl[s].load(); sn->sendOn[s] = t.sendOn[s].load(); sn->sendPre[s] = t.sendPre[s].load(); }
        bool restorable = true;
        for (int i = 0; i < (int) t.slots.size(); ++i) { auto p = std::make_shared<PlugSnap>(); snapSlot (idx, i, *p); restorable = restorable && ! p->internal; sn->plugs.push_back (p); }
        if (idx != n - 1) moveTrackTo (idx, n - 1);
        if (! removeLastTrack()) return false;
        if (! restorable) { undo.clear(); return true; }       // a built-in device cannot be rebuilt from a note: the history is no longer trustworthy
        undo.record ("Remove track", "", [this, sn, idx, n] { restoreTrack (sn, [this, idx, n] { if (idx != n - 1) moveTrackTo (n - 1, idx); }); }, [this, idx, n] { if (idx != n - 1) moveTrackTo (idx, n - 1); removeLastTrack(); });
        return true;
    }

    // plug-ins: the plug-in's own setting is kept as its saved state, and re-read when the step is redone
    void recordPluginPair (const juce::String& label, int chain, int slot, const std::shared_ptr<PlugSnap>& sn, bool added)
    {
        std::function<void()> rem = [this, chain, slot, sn] { snapSlot (chain, slot, *sn); removePlugin (chain, slot); };
        std::function<void()> add = [this, chain, slot, sn] { restorePlugin (chain, slot, sn); };
        if (added) undo.record (label.toStdString(), "", rem, add); else undo.record (label.toStdString(), "", add, rem);
    }
    void editAddPlugin (int chain, const juce::PluginDescription& d, std::function<void (juce::String)> done)
    {
        const bool intoEmptyReturn = isEmptyReturn (chain);
        addPluginAsync (chain, d, {}, [this, chain, d, done, intoEmptyReturn] (juce::String err)
        {
            if (err.isEmpty() && intoEmptyReturn) quietSends (chain);
            if (err.isEmpty()) bumpDeviceUse (d.createIdentifierString());
            if (err.isEmpty())
                if (auto* t = chainAt (chain)) { auto sn = std::make_shared<PlugSnap>(); sn->desc = d; recordPluginPair ("Add " + d.name, chain, (int) t->slots.size() - 1, sn, true); }
            if (err.isEmpty()) autoPatternBlock (chain);
            done (err);
        });
    }
    void editRemovePlugin (int chain, int slot)
    {
        auto sn = std::make_shared<PlugSnap>();
        if (! snapSlot (chain, slot, *sn)) return;
        removePlugin (chain, slot);
        if (sn->internal) { undo.clear(); return; }
        recordPluginPair ("Remove " + sn->desc.name, chain, slot, sn, false);
    }
    void editMoveSlot (int chain, int slot, int dir)
    {
        auto* t = chainAt (chain);
        const int j = slot + dir;
        if (t == nullptr || slot < 0 || j < 0 || slot >= (int) t->slots.size() || j >= (int) t->slots.size() || dir == 0) return;
        const auto nm = t->slots[(size_t) slot]->desc.name;
        moveSlot (chain, slot, dir);
        undo.record ("Move " + nm.toStdString(), "", [this, chain, j, dir] { moveSlot (chain, j, -dir); }, [this, chain, slot, dir] { moveSlot (chain, slot, dir); });
    }
    // Move a device to a place in the same rack, or into another rack (a track's, a return's, the master's). It is the SAME plug-in instance: its settings, its window and its state go with it.
    // Returns the index it ended up at, or -1. Raw: nothing is recorded.
    int moveSlotRaw (int fromChain, int from, int toChain, int to)
    {
        auto* a = chainAt (fromChain); auto* b = chainAt (toChain);
        if (a == nullptr || b == nullptr || from < 0 || from >= (int) a->slots.size()) return -1;
        if (a == b)
        {
            const int last = (int) a->slots.size() - 1; to = juce::jlimit (0, last, to);
            if (to == from) return from;
            if (from < to) std::rotate (a->slots.begin() + from, a->slots.begin() + from + 1, a->slots.begin() + to + 1);
            else           std::rotate (a->slots.begin() + to, a->slots.begin() + from, a->slots.begin() + from + 1);
        }
        else
        {
            to = juce::jlimit (0, (int) b->slots.size(), to);
            auto s = std::move (a->slots[(size_t) from]);
            a->slots.erase (a->slots.begin() + from);
            b->slots.insert (b->slots.begin() + to, std::move (s));
        }
        publishPlan(); stateDirty.store (true);
        return to;
    }
    int editMoveSlotTo (int fromChain, int from, int toChain, int to)
    {
        auto* a = chainAt (fromChain); if (a == nullptr || from < 0 || from >= (int) a->slots.size()) return -1;
        const auto nm = a->slots[(size_t) from]->desc.name.toStdString();
        const bool intoEmptyReturn = fromChain != toChain && isEmptyReturn (toChain);
        const int at = moveSlotRaw (fromChain, from, toChain, to);
        if (at >= 0 && intoEmptyReturn) quietSends (toChain);
        if (at < 0 || (fromChain == toChain && at == from)) return at;
        undo.record ("Move " + nm, "", [this, toChain, at, fromChain, from] { moveSlotRaw (toChain, at, fromChain, from); }, [this, fromChain, from, toChain, at] { moveSlotRaw (fromChain, from, toChain, at); });
        return at;
    }
    void editBypass (int chain, int slot, bool b)
    {
        auto* t = chainAt (chain);
        if (t == nullptr || slot < 0 || slot >= (int) t->slots.size()) return;
        const bool before = t->slots[(size_t) slot]->bypass.load();
        setBypass (chain, slot, b);
        if (before == b) return;
        undo.record ((b ? "Bypass " : "Un-bypass ") + t->slots[(size_t) slot]->desc.name.toStdString(), "", [this, chain, slot, before] { setBypass (chain, slot, before); }, [this, chain, slot, b] { setBypass (chain, slot, b); });
    }
    juce::var undoVar() const
    {
        auto* o = new juce::DynamicObject();
        o->setProperty ("canUndo", undo.canUndo()); o->setProperty ("canRedo", undo.canRedo());
        o->setProperty ("undoLabel", juce::String (undo.undoLabel())); o->setProperty ("redoLabel", juce::String (undo.redoLabel())); o->setProperty ("depth", (int) undo.depth());
        return juce::var (o);
    }

    //==========================================================================
    // plugins
    struct ScanProgress { juce::String current; bool running = false; int found = 0; };

    // The devices that come with the program (Archie Drums) live in a Devices folder next to it. The first time the console asks, any that are not known yet are scanned in, so they are in the list with no
    // trip to "Scan for plugins". Devices already known (same file, same date) are skipped by the scan itself.
    juce::File devicesFolder() const { return juce::File::getSpecialLocation (juce::File::currentExecutableFile).getParentDirectory().getChildFile ("Devices"); }
    void scanBundledDevicesOnce()
    {
        if (bundledChecked) return; bundledChecked = true;
        juce::Array<juce::File> found;
        for (const auto& d : { devicesFolder(), juce::File::getSpecialLocation (juce::File::currentExecutableFile).getSiblingFile ("Plug-ins") }) if (d.isDirectory()) ScanThread::collect (d, found);   // Devices (Archie Drums) and Plug-ins (the bundled instruments and effects)
        if (found.isEmpty()) return; bool fresh = false;
        for (auto& f : found) { bool have = false; for (const auto& t : knownPlugins.getTypes()) if (t.fileOrIdentifier == f.getFullPathName()) have = true; fresh = fresh || ! have; }
        if (fresh) startScan (false);
    }
    bool bundledChecked = false;
    void startScan (bool includeSystemFolder)
    {
        if (scanThread != nullptr && scanThread->isThreadRunning()) return;
        juce::StringArray dirs;
        dirs.add (juce::File::getSpecialLocation (juce::File::userApplicationDataDirectory).getParentDirectory().getChildFile ("Local").getChildFile ("AskArchie").getChildFile ("LocalBuilder").getChildFile ("output").getFullPathName());
        dirs.add (devicesFolder().getFullPathName());        // the devices that come with the program (Archie Drums)
        dirs.add (juce::File::getSpecialLocation (juce::File::currentExecutableFile).getSiblingFile ("Plug-ins").getFullPathName());      // the instruments and effects that come with the program, if the installer carried any
        for (auto& d : extraPluginFolders) dirs.add (d);
        if (includeSystemFolder)
        {
            const auto sp = juce::VST3PluginFormat().getDefaultLocationsToSearch();
            for (int i = 0; i < sp.getNumPaths(); ++i) dirs.add (sp[i].getFullPathName());
        }
        scanThread = std::make_unique<ScanThread> (*this, dirs);
        scanThread->startThread();
    }

    class ScanThread : public juce::Thread
    {
    public:
        ScanThread (HostEngine& e, juce::StringArray d) : juce::Thread ("Plugin scan"), eng (e), dirs (std::move (d)) { eng.scanState.running = true; eng.scanState.found = 0; }
        ~ScanThread() override { stopThread (4000); }
        // Every candidate file is scanned by a short-lived CHILD PROCESS of this program ("--scan <file> <out.xml>") with a time limit. A plug-in file that crashes or hangs while it is being
        // scanned therefore cannot take the program down: it is remembered in scan-bad.txt and skipped from then on.
        static void collect (const juce::File& dir, juce::Array<juce::File>& out)
        {
            for (const auto& f : juce::RangedDirectoryIterator (dir, false, "*", juce::File::findFilesAndDirectories))
            {
                const auto file = f.getFile();
                if (file.getFileExtension().equalsIgnoreCase (".vst3")) out.add (file);        // a bundle folder or a single file: either is one candidate, never descended into
                else if (f.isDirectory()) collect (file, out);
            }
        }
        void run() override
        {
            juce::Array<juce::File> candidates;
            for (auto& d : dirs) { juce::File dir (d); if (dir.isDirectory()) collect (dir, candidates); }
            const auto badFile = eng.settingsDir.getChildFile ("scan-bad.txt");
            auto bad = juce::StringArray::fromLines (badFile.loadFileAsString());
            const auto exe = juce::File::getSpecialLocation (juce::File::currentExecutableFile);
            const auto tmp = eng.settingsDir.getChildFile ("scan-tmp.xml");
            int index = 0;
            for (const auto& file : candidates)
            {
                if (threadShouldExit()) return;
                { const juce::ScopedLock sl (eng.scanLock); eng.scanState.current = file.getFileName() + "  (" + juce::String (++index) + " of " + juce::String (candidates.size()) + ")"; eng.scanState.found = eng.knownPlugins.getNumTypes(); }
                if (bad.contains (file.getFullPathName())) continue;
                {   // a file that is already in the list and has not changed since is not scanned again
                    bool known = false;
                    for (const auto& d : eng.knownPlugins.getTypes()) if (d.fileOrIdentifier == file.getFullPathName() && d.lastFileModTime == file.getLastModificationTime()) { known = true; break; }
                    if (known) continue;
                }
                tmp.deleteFile();
                juce::ChildProcess child;
                bool ok = child.start (juce::StringArray { exe.getFullPathName(), "--scan", file.getFullPathName(), tmp.getFullPathName() });
                if (ok) { ok = child.waitForProcessToFinish (45000); if (! ok) child.kill(); else ok = child.getExitCode() == 0; }
                if (! ok || ! tmp.existsAsFile()) { bad.addIfNotAlreadyThere (file.getFullPathName()); badFile.replaceWithText (bad.joinIntoString ("\n")); continue; }
                if (auto xml = juce::XmlDocument::parse (tmp))
                    for (auto* e : xml->getChildIterator()) { juce::PluginDescription d; if (d.loadFromXml (*e)) eng.knownPlugins.addType (d); }
            }
            tmp.deleteFile();
            // a plug-in whose file has gone (a moved or removed device) leaves the list, so the list never offers something that cannot load
            for (const auto& d : eng.knownPlugins.getTypes()) if (juce::File::isAbsolutePath (d.fileOrIdentifier) && ! juce::File (d.fileOrIdentifier).exists()) eng.knownPlugins.removeType (d);
            eng.saveKnownPlugins();
            { const juce::ScopedLock sl (eng.scanLock); eng.scanState.running = false; eng.scanState.current = {}; eng.scanState.found = eng.knownPlugins.getNumTypes(); }
            eng.stateDirty.store (true);
        }
    private:
        HostEngine& eng; juce::StringArray dirs;
    };

    //==========================================================================
    // The Analyser runs in a CHILD PROCESS too ("--analyse <plug-in> <out.json>"): a plug-in that crashes while being measured cannot take the program down.
    struct AnalysisState { bool running = false; juce::String name; juce::String error; };
    class AnalysisThread : public juce::Thread
    {
    public:
        AnalysisThread (HostEngine& e, juce::File f, juce::String id, juce::String label) : juce::Thread ("Analyser"), eng (e), file (std::move (f)), identifier (std::move (id)), name (std::move (label)) {}
        ~AnalysisThread() override { stopThread (3000); }
        void run() override
        {
            const auto exe = juce::File::getSpecialLocation (juce::File::currentExecutableFile);
            const auto dir = eng.settingsDir.getChildFile ("analysis"); dir.createDirectory();
            const auto out = dir.getChildFile ("running.json"); out.deleteFile();
            juce::StringArray args { exe.getFullPathName(), "--analyse", file.getFullPathName(), out.getFullPathName() };
            if (identifier.isNotEmpty()) args.add (identifier);
            juce::ChildProcess child; juce::String error; juce::var result;
            if (! child.start (args)) error = "could not start the analyser";
            else
            {
                const auto t0 = juce::Time::getMillisecondCounter();
                while (child.isRunning() && ! threadShouldExit() && juce::Time::getMillisecondCounter() - t0 < 240000) wait (200);
                if (child.isRunning()) { child.kill(); error = threadShouldExit() ? "cancelled" : "the analyser took longer than 4 minutes and was stopped"; }
                else if (! out.existsAsFile()) error = "the plug-in crashed or closed while it was being measured (the program itself is fine)";
            }
            if (error.isEmpty())
            {
                result = juce::JSON::parse (out.loadFileAsString());
                if (! result.isObject()) error = "the analyser produced no readable result";
                else if (result.hasProperty ("error")) { error = result.getProperty ("error", {}).toString(); result = juce::var(); }
                else
                {
                    auto stamp = juce::Time::getCurrentTime().formatted ("%Y%m%d-%H%M%S");
                    out.copyFileTo (dir.getChildFile (juce::File::createLegalFileName (name) + "-" + stamp + ".json"));
                }
            }
            out.deleteFile();
            { const juce::ScopedLock sl (eng.scanLock); eng.analysisState.running = false; eng.analysisState.error = error; eng.lastAnalysis = result; }
            eng.analysisSerial.fetch_add (1);
            eng.stateDirty.store (true);
        }
    private:
        HostEngine& eng; juce::File file; juce::String identifier, name;
    };

    bool startAnalysis (const juce::File& file, const juce::String& identifier, const juce::String& label)
    {
        { const juce::ScopedLock sl (scanLock); if (analysisState.running) return false; analysisState.running = true; analysisState.name = label; analysisState.error = juce::String(); lastAnalysis = juce::var(); }
        analysisThread.reset();
        analysisThread = std::make_unique<AnalysisThread> (*this, file, identifier, label);
        analysisThread->startThread();
        stateDirty.store (true);
        return true;
    }
    void cancelAnalysis() { if (analysisThread != nullptr) analysisThread->signalThreadShouldExit(); }
    AnalysisState getAnalysisState() { const juce::ScopedLock sl (scanLock); return analysisState; }
    juce::var getLastAnalysis() { const juce::ScopedLock sl (scanLock); return lastAnalysis; }
    std::atomic<int> analysisSerial { 0 };

    ScanProgress getScanProgress() { const juce::ScopedLock sl (scanLock); return scanState; }

    //==========================================================================
    // The device browser's own notes about each plug-in: which are favourites, the person's tags, and how often and how lately each was used. Kept beside the plug-in list, never inside a project.
    // A device is told apart from the others by its identifier string, the same one the plug-in list uses.
    juce::File deviceMetaFile;
    juce::var deviceMeta { new juce::DynamicObject() };           // id -> { fav, tags: [..], uses, last }
    void loadDeviceMeta()
    {
        if (deviceMetaFile.existsAsFile()) { auto v = juce::JSON::parse (deviceMetaFile.loadFileAsString()); if (v.getDynamicObject() != nullptr) deviceMeta = v; }
        if (deviceMeta.getDynamicObject() == nullptr) deviceMeta = juce::var (new juce::DynamicObject());
    }
    void saveDeviceMeta() { if (deviceMetaFile != juce::File()) deviceMetaFile.replaceWithText (juce::JSON::toString (deviceMeta, true)); }
    static juce::String cleanTag (juce::String t) { t = t.trim().toLowerCase().retainCharacters ("abcdefghijklmnopqrstuvwxyz0123456789 -_&+"); return t.trim().substring (0, 24); }
    juce::DynamicObject* metaFor (const juce::String& id, bool create)
    {
        auto* all = deviceMeta.getDynamicObject(); if (all == nullptr || id.isEmpty()) return nullptr;
        if (! all->hasProperty (id)) { if (! create) return nullptr; all->setProperty (id, juce::var (new juce::DynamicObject())); }
        return all->getProperty (id).getDynamicObject();
    }
    // patch: { favourite: bool, tags: [string] } (either or both)
    bool setDeviceMeta (const juce::String& id, const juce::var& patch)
    {
        if (id.isEmpty()) return false;
        auto* m = metaFor (id, true); if (m == nullptr) return false;
        if (patch.hasProperty ("favourite")) m->setProperty ("fav", (bool) patch.getProperty ("favourite", false));
        if (auto* arr = patch.getProperty ("tags", {}).getArray())
        {
            juce::StringArray tags;
            for (const auto& t : *arr) { const auto c = cleanTag (t.toString()); if (c.isNotEmpty() && tags.size() < 12) tags.addIfNotAlreadyThere (c); }
            juce::Array<juce::var> out; for (const auto& t : tags) out.add (t);
            m->setProperty ("tags", out);
        }
        saveDeviceMeta(); stateDirty.store (true);
        return true;
    }
    void bumpDeviceUse (const juce::String& id)
    {
        if (auto* m = metaFor (id, true)) { m->setProperty ("uses", (int) m->getProperty ("uses") + 1); m->setProperty ("last", juce::Time::currentTimeMillis()); saveDeviceMeta(); }
    }
    // Instrument, effect or utility, and the sub-categories the plug-in says it belongs to (VST3 writes them as "Fx|Dynamics" or "Instrument|Synth"). Nothing is guessed from the name.
    struct DeviceClass { juce::String kind; juce::StringArray cats; };
    static DeviceClass classifyDevice (const juce::PluginDescription& d)
    {
        DeviceClass c; c.kind = d.isInstrument ? "instrument" : "effect";
        for (const auto& tok : juce::StringArray::fromTokens (d.category, "|", ""))
        {
            const auto t = tok.trim(); if (t.isEmpty()) continue;
            if (t.equalsIgnoreCase ("Fx") || t.equalsIgnoreCase ("Instrument") || t.equalsIgnoreCase ("Effect")) continue;
            c.cats.addIfNotAlreadyThere (t);
            if (! d.isInstrument && (t.equalsIgnoreCase ("Analyzer") || t.equalsIgnoreCase ("Analyser") || t.equalsIgnoreCase ("Tools") || t.equalsIgnoreCase ("Utility") || t.equalsIgnoreCase ("Generator"))) c.kind = "utility";
        }
        return c;
    }
    juce::var deviceListVar()
    {
        juce::Array<juce::var> pl;
        for (const auto& d : knownPlugins.getTypes())
        {
            auto* o = new juce::DynamicObject(); const auto id = d.createIdentifierString(); const auto cls = classifyDevice (d);
            o->setProperty ("id", id); o->setProperty ("name", d.name); o->setProperty ("maker", d.manufacturerName);
            o->setProperty ("instrument", d.isInstrument); o->setProperty ("category", d.category); o->setProperty ("format", d.pluginFormatName);
            o->setProperty ("kind", cls.kind); o->setProperty ("cats", cls.cats); o->setProperty ("version", d.version); o->setProperty ("file", d.fileOrIdentifier);
            if (auto* m = metaFor (id, false)) { o->setProperty ("fav", (bool) m->getProperty ("fav")); o->setProperty ("tags", m->getProperty ("tags")); o->setProperty ("uses", (int) m->getProperty ("uses")); o->setProperty ("last", m->getProperty ("last")); }
            pl.add (juce::var (o));
        }
        return pl;
    }

    void loadKnownPlugins()
    {
        auto f = settingsDir.getChildFile ("known-plugins.xml");
        if (auto xml = juce::XmlDocument::parse (f)) knownPlugins.recreateFromXml (*xml);
    }
    void saveKnownPlugins()
    {
        if (auto xml = knownPlugins.createXml()) xml->writeTo (settingsDir.getChildFile ("known-plugins.xml"));
    }

    // ---- CRASH GUARD for loading plug-ins ----
    // FILTERING/MONITORING MECHANISM: a plug-in that crashes while it is being loaded takes the whole program with it (plug-ins run inside this program). Before each load the plug-in's id is written to
    // loading-plugin.txt, and removed when the load finishes. If the file is still there at the next start, the program died while loading that plug-in: it is added to crashed-plugins.json with a count.
    // It judges ONLY "the program died while this was loading" -- not the plug-in's quality, not crashes later (editor, playing). It never blocks a plug-in for good: the person is told, and can load it anyway
    // (addPlugin with force). The list can be cleared plug-in by plug-in (forgetCrash). It is itself watched: the list is shown in the state so it can be checked.
    std::map<juce::String, juce::String> loadingNow;                       // id -> name, for loads that have not finished
    juce::File loadingMarker() const { return settingsDir.getChildFile (juce::SystemStats::getEnvironmentVariable ("ASKARCHIE_HOST_TEST", {}) == "1" ? "loading-plugin-test.txt" : "loading-plugin.txt"); }
    juce::File crashedFile() const { return settingsDir.getChildFile (juce::SystemStats::getEnvironmentVariable ("ASKARCHIE_HOST_TEST", {}) == "1" ? "crashed-plugins-test.json" : "crashed-plugins.json"); }
    std::map<juce::String, std::pair<juce::String, int>> crashedPlugins;   // id -> (name, times)
    void saveCrashed()
    {
        juce::Array<juce::var> a; for (auto& kv : crashedPlugins) { auto* o = new juce::DynamicObject(); o->setProperty ("id", kv.first); o->setProperty ("name", kv.second.first); o->setProperty ("times", kv.second.second); a.add (juce::var (o)); }
        crashedFile().replaceWithText (juce::JSON::toString (juce::var (a)));
    }
    void readLoadingMarker()
    {
        crashedPlugins.clear();
        const auto parsedCrashed = juce::JSON::parse (crashedFile().loadFileAsString()); if (auto* a = parsedCrashed.getArray()) for (const auto& v : *a) crashedPlugins[v.getProperty ("id", {}).toString()] = { v.getProperty ("name", {}).toString(), (int) v.getProperty ("times", 1) };
        auto m = loadingMarker();
        if (m.existsAsFile())
        {
            juce::StringArray lines; lines.addLines (m.loadFileAsString());
            for (int i = 0; i + 1 < lines.size(); i += 2) { auto& e = crashedPlugins[lines[i]]; e.first = lines[i + 1]; ++e.second; }
            m.deleteFile(); saveCrashed();
        }
        readCrashModule();
        for (auto& tf : settingsDir.findChildFiles (juce::File::findFiles, false, ".recovery_temp*")) tf.deleteFile();      // half-written keep-copies from a crash
    }
    void writeLoadingMarker() { juce::String t; for (auto& kv : loadingNow) t << kv.first << "\n" << kv.second << "\n"; if (t.isEmpty()) loadingMarker().deleteFile(); else loadingMarker().replaceWithText (t); }
    // A crash in a plug-in's OWN thread (after it has loaded) is caught by the crash filter in Main.cpp, which writes the name of the file the fault was in to crash-module.bin. The next start reads it here.
    juce::String lastCrashTrace;
    juce::File crashModuleFile() const { return settingsDir.getChildFile (juce::SystemStats::getEnvironmentVariable ("ASKARCHIE_HOST_TEST", {}) == "1" ? "crash-module-test.bin" : "crash-module.bin"); }
    static juce::String normPath (juce::String s) { return s.replaceCharacter ('\\', '/').toLowerCase(); }
    void readCrashModule()
    {
        auto f = crashModuleFile(); if (! f.existsAsFile()) return;
        juce::MemoryBlock mb; f.loadFileAsData (mb); f.deleteFile();
        if (mb.getSize() < 4) return;
        const juce::String all = juce::String (juce::CharPointer_UTF16 ((const juce::CharPointer_UTF16::CharType*) mb.getData()), mb.getSize() / 2);
        lastCrashTrace = all;                                                              // kept (crash-trace.txt) so the stack of the last crash can be read
        crashModuleFile().getSiblingFile ("crash-trace.txt").replaceWithText (all);
        juce::StringArray lines; lines.addLines (all);
        for (auto& line : lines)                                                           // newest first: the nearest plug-in file on the stack is the one flagged
        {
            const auto path = line.trim();
            if (! normPath (path).contains (".vst3")) continue;                           // only plug-in files are flagged: the program and Windows are not
            auto& e = crashedPlugins["module:" + normPath (path)]; e.first = juce::File (path).getFileNameWithoutExtension(); ++e.second; saveCrashed();
            break;
        }
    }
    // is this plug-in flagged, by its id or by the file a crash was found in?
    bool crashedBeforeDesc (const juce::PluginDescription& d, juce::String* name = nullptr, int* times = nullptr) const
    {
        if (crashedBefore (d.createIdentifierString(), name, times)) return true;
        const auto me = normPath (d.fileOrIdentifier); if (me.isEmpty()) return false;
        for (auto& kv : crashedPlugins) if (kv.first.startsWith ("module:") && normPath (kv.first.substring (7)).startsWith (me)) { if (name) *name = d.name.isNotEmpty() ? d.name : kv.second.first; if (times) *times = kv.second.second; return true; }
        return false;
    }
    bool crashedBefore (const juce::String& id, juce::String* name = nullptr, int* times = nullptr) const { auto it = crashedPlugins.find (id); if (it == crashedPlugins.end()) return false; if (name) *name = it->second.first; if (times) *times = it->second.second; return true; }
    void forgetCrash (const juce::String& id) { crashedPlugins.erase (id); saveCrashed(); stateDirty.store (true); }
    juce::var crashedVar() const { juce::Array<juce::var> a; for (auto& kv : crashedPlugins) { auto* o = new juce::DynamicObject(); o->setProperty ("id", kv.first); o->setProperty ("name", kv.second.first); o->setProperty ("times", kv.second.second); a.add (juce::var (o)); } return juce::var (a); }

    void addPluginAsync (int track, const juce::PluginDescription& d, const juce::String& stateB64, std::function<void (juce::String)> done, juce::String linkIdHint = {})
    {
        auto* t = chainAt (track);
        if (t == nullptr) { done ("no such track"); return; }
        const auto guardId = d.createIdentifierString(); loadingNow[guardId] = d.name; writeLoadingMarker();
        formatManager.createPluginInstanceAsync (d, sampleRate > 0 ? sampleRate : 48000.0, blockSize > 0 ? blockSize : 512,
            [this, track, d, stateB64, done, guardId, linkIdHint] (std::unique_ptr<juce::AudioPluginInstance> inst, const juce::String& err)
            {
                struct Release { HostEngine& e; juce::String id; ~Release() { e.loadingNow.erase (id); e.writeLoadingMarker(); } } release { *this, guardId };   // the load is over, one way or another
                keepPluginModulesLoaded();      // its code stays loaded until the program ends, so removing it can never leave a late call pointing at nothing (KeepLoaded.h)
                auto* tr = chainAt (track);
                if (inst == nullptr || tr == nullptr) { done (err.isNotEmpty() ? err : "could not load the plugin"); return; }
                // THE BUSES: the program mixes through ONE stereo pair per plug-in (see Renderer::runInsert). A plug-in with many outputs (a drum machine or sampler with an output for every pad) or a side-chain input
                // would otherwise be handed more channels than the program gives it, and it writes past the end of the buffer and closes the program. So only the MAIN input and output are on, as a stereo pair
                // where the plug-in allows it, and every extra bus is switched off (those sounds come out of the main pair).
                if (auto* b = inst->getBus (true, 0)) b->enable (true);
                if (auto* b = inst->getBus (false, 0)) b->enable (true);
                auto layout = inst->getBusesLayout();
                if (inst->getBusCount (true) > 0 && layout.inputBuses.size() > 0) layout.inputBuses.getReference (0) = juce::AudioChannelSet::stereo();
                if (inst->getBusCount (false) > 0 && layout.outputBuses.size() > 0) layout.outputBuses.getReference (0) = juce::AudioChannelSet::stereo();
                for (int bi = 1; bi < layout.inputBuses.size(); ++bi) layout.inputBuses.getReference (bi) = juce::AudioChannelSet::disabled();
                for (int bi = 1; bi < layout.outputBuses.size(); ++bi) layout.outputBuses.getReference (bi) = juce::AudioChannelSet::disabled();
                if (inst->checkBusesLayoutSupported (layout)) inst->setBusesLayout (layout);
                else
                {   // the plug-in will not take that exactly: keep what it offers, with the extra buses off where it lets us
                    auto alt = inst->getBusesLayout();
                    for (int bi = 1; bi < alt.inputBuses.size(); ++bi) alt.inputBuses.getReference (bi) = juce::AudioChannelSet::disabled();
                    for (int bi = 1; bi < alt.outputBuses.size(); ++bi) alt.outputBuses.getReference (bi) = juce::AudioChannelSet::disabled();
                    if (inst->checkBusesLayoutSupported (alt)) inst->setBusesLayout (alt);
                }
                inst->setPlayHead (&playHead);
                prepareProc (*inst);
                if (stateB64.isNotEmpty())
                {
                    juce::MemoryBlock mb; mb.fromBase64Encoding (stateB64);
                    if (mb.getSize() > 0) inst->setStateInformation (mb.getData(), (int) mb.getSize());
                }
                auto slot = std::make_unique<Slot>();
                slot->desc = d; slot->bypass.store (false);
                if (linkIdHint.isNotEmpty() && ! linkIdInUse (linkIdHint)) slot->linkId = linkIdHint;      // the song remembers which connection this copy had (a pasted copy does not take its original's)
                slot->proc = std::move (inst);
                tr->slots.push_back (std::move (slot));
                autoLinkIfSaved (track, (int) tr->slots.size() - 1);
                publishPlan();
                stateDirty.store (true);
                done ({});
            });
    }

    // Puts one of OUR OWN processors (a built-in device, or a test processor) on a chain, the same way a hosted plug-in goes on. Returns the slot number, or -1.
    int addProcessor (int chain, std::unique_ptr<juce::AudioProcessor> proc, const juce::String& name)
    {
        auto* t = chainAt (chain);
        if (t == nullptr || proc == nullptr) return -1;
        prepareProc (*proc);
        auto slot = std::make_unique<Slot>();
        slot->desc.name = name; slot->desc.manufacturerName = "Ask Archie"; slot->desc.pluginFormatName = "Internal";
        slot->proc = std::move (proc);
        t->slots.push_back (std::move (slot));
        publishPlan(); stateDirty.store (true);
        return (int) t->slots.size() - 1;
    }

    void removePlugin (int track, int slot)
    {
        auto* t = chainAt (track);
        if (t == nullptr || slot < 0 || slot >= (int) t->slots.size()) return;
        auto gone = std::move (t->slots[(size_t) slot]);
        gone->window.reset();
        t->slots.erase (t->slots.begin() + slot);
        publishPlan(); quiesce();          // the plan without it is live, and the audio thread has moved on, before it is destroyed
        gone.reset();
        stateDirty.store (true);
    }

    void moveSlot (int track, int slot, int dir)
    {
        auto* t = chainAt (track);
        const int j = slot + dir;
        if (t == nullptr || slot < 0 || j < 0 || slot >= (int) t->slots.size() || j >= (int) t->slots.size()) return;
        std::swap (t->slots[(size_t) slot], t->slots[(size_t) j]);
        publishPlan(); stateDirty.store (true);
    }

    void setBypass (int track, int slot, bool b)
    {
        auto* t = chainAt (track);
        if (t == nullptr || slot < 0 || slot >= (int) t->slots.size()) return;
        t->slots[(size_t) slot]->bypass.store (b);
        stateDirty.store (true);
    }

    void openEditor (int track, int slot)
    {
        auto* t = chainAt (track);
        if (t == nullptr || slot < 0 || slot >= (int) t->slots.size()) return;
        auto& s = *t->slots[(size_t) slot];
        if (s.window == nullptr)
        {
            const auto gid = s.desc.createIdentifierString(); loadingNow[gid] = s.desc.name; writeLoadingMarker();      // opening a plug-in's window is the other thing that brings plug-ins down
            s.window = std::make_unique<PluginWindow> (*s.proc, [this, track] { if (learn.chain == track) stopAutoLearn(); else startAutoLearn (track); }, [this, track] { return learn.chain == track && ! learn.remote; },
                [this, track] (int action) { if (action == 1) startRemote (track); else if (action == 2) { clearRemotes (track); remoteDone = "Controller assignments cleared for this track"; stateDirty.store (true); } else cancelRemote(); },
                [this] { return remoteStage.load(); });
            loadingNow.erase (gid); writeLoadingMarker();
        }
        else { s.window->setVisible (true); s.window->toFront (true); }
    }

    //==========================================================================
    // MIDI devices
    // Opens every input the person has not switched off. On Windows a MIDI input can be held by only one program at a time, so an input another program (a DAW, a plug-in host) already has open
    // cannot be opened here: that is recorded per input (midiOpenOk) and shown in the console, never hidden.
    void refreshMidiInputs()
    {
        for (auto& id : openMidiInputs) deviceManager.removeMidiInputDeviceCallback (id, this);
        openMidiInputs.clear();
        const auto devices = juce::MidiInput::getAvailableDevices();
        juce::StringArray key; for (const auto& d : devices) key.add (d.identifier);
        lastMidiDeviceKey = key.joinIntoString ("|");
        const juce::ScopedLock sl (midiOpenLock);
        midiOpenOk.clear();
        for (const auto& d : devices)
        {
            if (disabledMidiIns.indexOf (d.identifier) >= 0) continue;
            deviceManager.setMidiInputDeviceEnabled (d.identifier, true);
            const bool ok = deviceManager.isMidiInputDeviceEnabled (d.identifier);
            midiOpenOk[d.identifier] = ok;
            if (ok) { deviceManager.addMidiInputDeviceCallback (d.identifier, this); openMidiInputs.add (d.identifier); }
        }
        stateDirty.store (true);
    }
    // a keyboard plugged in (or taken out) while the program is running is picked up within a couple of seconds
    void pollMidiDevices()
    {
        juce::StringArray key; for (const auto& d : juce::MidiInput::getAvailableDevices()) key.add (d.identifier);
        if (key.joinIntoString ("|") != lastMidiDeviceKey) refreshMidiInputs();
    }
    bool isMidiInputOpen (const juce::String& id) { const juce::ScopedLock sl (midiOpenLock); auto it = midiOpenOk.find (id); return it != midiOpenOk.end() && it->second; }
    void setMidiInputEnabled (const juce::String& id, bool on)
    {
        if (on) disabledMidiIns.removeString (id); else disabledMidiIns.addIfNotAlreadyThere (id);
        refreshMidiInputs();
    }
    bool openMidiOutput (const juce::String& id)
    {
        std::unique_ptr<juce::MidiOutput> o;
        if (id.isNotEmpty()) { o = juce::MidiOutput::openDevice (id); if (o == nullptr) return false; o->startBackgroundThread(); }
        const juce::SpinLock::ScopedLockType lock (midiOutLock);
        midiOut = std::move (o);
        midiOutId = id;
        return true;
    }
    juce::String getMidiOutputId() const { return midiOutId; }

    //==========================================================================
    // the whole state for the pages
    juce::var snapshot()
    {
        auto* root = new juce::DynamicObject();
        root->setProperty ("sampleRate", sampleRate);
        root->setProperty ("blockSize", blockSize);
        if (auto* dev = deviceManager.getCurrentAudioDevice())
        {
            root->setProperty ("audioDevice", dev->getName());
            { juce::Array<juce::var> ins; for (auto& nm : inputChannelNames()) ins.add (nm); root->setProperty ("inputs", ins); }
            root->setProperty ("audioType", dev->getTypeName());
            root->setProperty ("inputLatencyMs", 1000.0 * dev->getInputLatencyInSamples() / sampleRate);
            root->setProperty ("outputLatencyMs", 1000.0 * dev->getOutputLatencyInSamples() / sampleRate);
        }
        root->setProperty ("audioError", lastDeviceError);
        root->setProperty ("outputChannels", deviceManager.getCurrentAudioDevice() != nullptr ? deviceManager.getCurrentAudioDevice()->getActiveOutputChannels().countNumberOfSetBits() : 0);
        root->setProperty ("latencyAdjustMs", latencyAdjustMs.load()); root->setProperty ("maxThreads", maxThreadsSetting.load()); root->setProperty ("playInBackground", playInBackground.load()); root->setProperty ("audioReleased", audioReleased); root->setProperty ("renderWithCardBuffer", renderWithCardBuffer.load()); root->setProperty ("masterTuneCents", masterTuneCents.load()); root->setProperty ("cpuCores", juce::SystemStats::getNumPhysicalCpus());
        juce::Array<juce::var> trs;
        for (int ti = 0; ti < trackCount.load(); ++ti)
        {
            auto& t = *tracks[(size_t) ti];
            auto* o = new juce::DynamicObject();
            o->setProperty ("index", t.index); o->setProperty ("name", t.name); o->setProperty ("colour", t.colour.load()); o->setProperty ("latency", chainLatency (t));
            o->setProperty ("kind", t.kind.load()); o->setProperty ("inChan", t.inChan.load()); o->setProperty ("inStereo", t.inStereo.load()); o->setProperty ("inGainDb", (double) t.inGainDb.load()); o->setProperty ("inInvert", t.inInvert.load()); o->setProperty ("patPlay", t.patPlay.load()); o->setProperty ("patTrans", t.patTrans.load()); o->setProperty ("eq", eqVar (t.index)); o->setProperty ("width", (double) t.width.load()); o->setProperty ("groove", t.grooveCh.load()); o->setProperty ("gateOn", t.gateOn.load()); o->setProperty ("gateExp", t.gateExp.load()); o->setProperty ("gateFast", t.gateFast.load()); o->setProperty ("gateThr", (double) t.gateThr.load()); o->setProperty ("gateRange", (double) t.gateRange.load()); o->setProperty ("gateRel", (double) t.gateRel.load()); o->setProperty ("gateHold", (double) t.gateHold.load()); o->setProperty ("compOn", t.compOn.load()); o->setProperty ("compFast", t.compFast.load()); o->setProperty ("compPeak", t.compPeak.load()); o->setProperty ("compThr", (double) t.compThr.load()); o->setProperty ("compRatio", (double) t.compRatio.load()); o->setProperty ("compRel", (double) t.compRel.load()); o->setProperty ("compMakeup", (double) t.compMakeup.load()); o->setProperty ("eqHpOn", t.eqHpOn.load()); o->setProperty ("eqHpHz", (double) t.eqHpHz.load()); o->setProperty ("eqLpOn", t.eqLpOn.load()); o->setProperty ("eqLpHz", (double) t.eqLpHz.load()); o->setProperty ("monitor", t.monitor.load()); o->setProperty ("source", t.source.load()); o->setProperty ("gainDb", t.gainDb.load()); o->setProperty ("pan", t.pan.load());
            o->setProperty ("mute", t.mute.load()); o->setProperty ("solo", t.solo.load()); o->setProperty ("armed", t.armed.load());
            o->setProperty ("file", juce::File (t.filePath).getFileName());
            const Clip pc = t.primaryClip();
            o->setProperty ("clipLength", pc.lengthBeats); o->setProperty ("clipNotes", (int) pc.notes.size()); o->setProperty ("recNotes", recGhostsVar (t)); o->setProperty ("instances", (int) t.arrSnap.load()->insts.size());
            { double ah = 0; for (const auto& l : t.autoSnap.load()->lanes) { ah = ah * 1.000003 + l.id * 7.0 + (l.on ? 3.0 : 0.0) + (double) l.pts.size() * 11.0; for (const auto& p : l.pts) ah = ah * 1.000001 + p.beat * 13.0 + p.value * 17.0; } o->setProperty ("autoRev", ah); }
            { int nl = 0, no = 0; for (const auto& l : t.autoSnap.load()->lanes) { ++nl; if (l.on) ++no; } o->setProperty ("autoLanes", nl); o->setProperty ("autoOn", no); }          // for the mixer's AUTO button
            { double h = 0; for (const auto& in : t.arrSnap.load()->insts) { for (const auto& mk : in.markers) h = h * 1.000007 + mk.sec * 83.0 + mk.beat * 89.0; h = h * 1.000003 + in.id * 7.0 + in.start * 13.0 + in.length * 17.0 + in.offset * 19.0 + (in.content ? in.content->id * 23.0 + (double) in.content->notes.size() * 29.0 : 0.0) + (in.warp ? 37.0 : 0.0) + (in.warped ? 41.0 : 0.0) + in.srcBpm * 43.0 + (in.repitch ? 47.0 : 0.0) + in.gainDb * 53.0 + in.fadeIn * 59.0 + in.fadeOut * 61.0 + (in.reverse ? 67.0 : 0.0) + in.colour * 97.0 + (in.muted ? 101.0 : 0.0) + (double) in.label.hashCode() * 0.001; } o->setProperty ("arrRev", h); }      // changes whenever a clip is changed, by anyone
            juce::Array<juce::var> sl;
            for (auto& s : t.slots) { auto* so = new juce::DynamicObject(); so->setProperty ("name", s->desc.name); so->setProperty ("maker", s->desc.manufacturerName); so->setProperty ("bypass", s->bypass.load()); so->setProperty ("latency", juce::jmax (0, s->proc->getLatencySamples())); so->setProperty ("instrument", s->desc.isInstrument); so->setProperty ("linkable", linkKindOf (*s).isNotEmpty()); so->setProperty ("linkId", s->linkId); if (s->link != nullptr) { so->setProperty ("link", archielink::DeviceLink::stateName (s->link->state())); so->setProperty ("linkMessage", s->link->message()); } else so->setProperty ("link", "off"); sl.add (juce::var (so)); }
            o->setProperty ("slots", sl);
            juce::Array<juce::var> sends; for (int s = 0; s < kSends; ++s) { auto* so = new juce::DynamicObject(); so->setProperty ("db", sendEff (t, s)); so->setProperty ("lvl", t.sendLvl[s].load()); so->setProperty ("on", t.sendOn[s].load()); so->setProperty ("pre", t.sendPre[s].load()); sends.add (juce::var (so)); }
            o->setProperty ("sends", sends); o->setProperty ("patterns", patternCountOf (t));
            trs.add (juce::var (o));
        }
        root->setProperty ("tracks", trs);
        root->setProperty ("returns", chainsVar());
        { auto* mo = new juce::DynamicObject(); mo->setProperty ("latency", chainLatency (*masterChain)); mo->setProperty ("slots", slotsVar (*masterChain));
          { double ah = 0; int nl = 0; for (const auto& l : masterChain->autoSnap.load()->lanes) { ++nl; ah = ah * 1.000003 + l.id * 7.0 + (l.on ? 3.0 : 0.0) + (double) l.pts.size() * 11.0; for (const auto& p : l.pts) ah = ah * 1.000001 + p.beat * 13.0 + p.value * 17.0; } mo->setProperty ("autoRev", ah); mo->setProperty ("autoLanes", nl); }
          root->setProperty ("masterSection", juce::var (mo)); }
        root->setProperty ("masterDb", master != nullptr ? master->gainDb.load() : 0.0f); root->setProperty ("masterComp", masterCompVar()); root->setProperty ("masterSound", masterSoundVar()); root->setProperty ("autoLearn", autoLearnVar()); root->setProperty ("controlRoom", controlRoomVar()); root->setProperty ("groove", grooveVar (false));
        root->setProperty ("plugins", deviceListVar());
        auto sc = getScanProgress();
        { auto an = getAnalysisState(); root->setProperty ("analysing", an.running); root->setProperty ("analysisName", an.name); root->setProperty ("analysisError", an.error); root->setProperty ("analysisSerial", analysisSerial.load()); }
        root->setProperty ("scanning", sc.running); root->setProperty ("scanCurrent", sc.current); root->setProperty ("pluginCount", knownPlugins.getNumTypes());
        juce::Array<juce::var> mi, mo;
        for (const auto& d : juce::MidiInput::getAvailableDevices()) { auto* o = new juce::DynamicObject(); o->setProperty ("id", d.identifier); o->setProperty ("name", d.name); o->setProperty ("enabled", disabledMidiIns.indexOf (d.identifier) < 0); o->setProperty ("open", isMidiInputOpen (d.identifier)); mi.add (juce::var (o)); }
        for (const auto& d : juce::MidiOutput::getAvailableDevices()) { auto* o = new juce::DynamicObject(); o->setProperty ("id", d.identifier); o->setProperty ("name", d.name); mo.add (juce::var (o)); }
        root->setProperty ("midiInputs", mi); root->setProperty ("midiOutputs", mo); root->setProperty ("midiOutputId", midiOutId);
        root->setProperty ("metronome", metronome.load());
        root->setProperty ("undo", undoVar());
        { auto* lt = new juce::DynamicObject(); lt->setProperty ("serial", takeSerial.load()); lt->setProperty ("name", lastTakeName); lt->setProperty ("track", lastTakeTrack); root->setProperty ("lastTake", juce::var (lt)); }
        { bool any = false; for (auto& t : tracks) { auto* r = t->audioRec.load(); if (r && ! r->takeL.empty()) any = true; } root->setProperty ("capturing", any); }
        return juce::var (root);
    }

    // Archie devices that can be connected to the person's Ask Archie account from the rack (the DAW does the connecting, the device needs no code of its own)
    static juce::String linkKindOf (const Slot& s) { return (s.desc.name == "Archie Drums" && s.desc.manufacturerName == "Ask Archie") ? juce::String ("archie-drums") : juce::String(); }
    juce::String runtimeIdForLinks()
    {
        auto f = settingsDir.getChildFile ("device-link-id.txt");
        if (f.existsAsFile()) { auto t = f.loadFileAsString().trim(); if (t.isNotEmpty()) return t; }
        auto id = "daw-" + juce::Uuid().toDashedString().removeCharacters ("-").substring (0, 16); settingsDir.createDirectory(); f.replaceWithText (id); return id;
    }
    // Every chain that can hold a device, by number (tracks, FX returns, the master inserts)
    std::vector<int> allChainIds()
    {
        std::vector<int> ids; for (int i = 0; i < trackCount.load(); ++i) ids.push_back (i);
        for (int r = 0; r < kReturns; ++r) ids.push_back (kReturnBase + r);
        ids.push_back (kMasterChain); return ids;
    }
    // "Connect all to Archie": every device that can connect, in one go. Each copy has its own connection (its own pairing the first time).
    juce::var editDeviceLinkAll (bool on)
    {
        int connected = 0, skipped = 0, devices = 0; juce::StringArray notes;
        for (int chain : allChainIds())
            if (auto* t = chainAt (chain))
                for (int si = 0; si < (int) t->slots.size(); ++si)
                {
                    if (linkKindOf (*t->slots[(size_t) si]).isEmpty()) continue; ++devices;
                    juce::String e; if (editDeviceLink (chain, si, on, e)) { if (on) ++connected; } else { ++skipped; notes.addIfNotAlreadyThere (e); }
                }
        auto* o = new juce::DynamicObject(); o->setProperty ("ok", true); o->setProperty ("devices", devices); o->setProperty ("connected", connected); o->setProperty ("skipped", skipped); o->setProperty ("notes", notes.joinIntoString (" "));
        return juce::var (o);
    }
    bool editDeviceShow (int chain, int slot, juce::String& err)
    {
        auto* t = chainAt (chain); if (t == nullptr || slot < 0 || slot >= (int) t->slots.size()) { err = "no such device"; return false; }
        auto& s = *t->slots[(size_t) slot];
        if (s.link == nullptr || ! s.link->showInAskArchie()) { err = "Connect this device to Ask Archie first (the ARCHIE button)."; return false; }
        return true;
    }
    // What the website's drum panel needs that is not a parameter: the pattern on view. The plug-in keeps all sixteen in its saved state, which is read here (on the message thread) and cut down to the one shown.
    // With `diagnose` it also reports what the saved state looked like, for the "deviceExtras" check.
    struct DrumsExtrasState { juce::String last; int calls = 0; };       // what was last sent of all sixteen patterns
    static juce::var archieDrumsExtras (juce::AudioProcessor& p, bool diagnose, std::shared_ptr<DrumsExtrasState> sent = nullptr)
    {
        juce::MemoryBlock mb; p.getStateInformation (mb);
        auto xml = juce::AudioProcessor::getXmlFromBinary (mb.getData(), (int) mb.getSize());
        // A hosted VST3 hands its state back wrapped: <VST3PluginState><IComponent>base64</IComponent></VST3PluginState>. The plug-in's own saved state (JUCE's XML-in-binary, which starts "VC2!") is inside it.
        if (xml != nullptr && xml->hasTagName ("VST3PluginState"))
        {
            std::unique_ptr<juce::XmlElement> inner;
            if (auto* ic = xml->getChildByName ("IComponent"))
            {
                juce::MemoryBlock raw; raw.fromBase64Encoding (ic->getAllSubText().trim());
                const auto* b = static_cast<const unsigned char*> (raw.getData());
                for (size_t i = 0; i + 8 < raw.getSize(); ++i)
                    if (b[i] == 'V' && b[i + 1] == 'C' && b[i + 2] == '2' && b[i + 3] == '!') { inner = juce::AudioProcessor::getXmlFromBinary (b + i, (int) (raw.getSize() - i)); if (inner != nullptr) break; }
            }
            xml = std::move (inner);
        }
        auto* o = new juce::DynamicObject(); juce::var out (o);
        if (xml == nullptr) { if (diagnose) o->setProperty ("error", "state is not XML, " + juce::String ((int) mb.getSize()) + " bytes"); return diagnose ? out : juce::var(); }
        int bank = 0, slot = 0; juce::StringArray names;
        for (auto* c : xml->getChildIterator()) { const auto id = c->getStringAttribute ("id"); names.add (c->getTagName()); if (id == "bank") bank = (int) c->getDoubleAttribute ("value"); else if (id == "pattern") slot = (int) c->getDoubleAttribute ("value"); }
        const int idx = juce::jlimit (0, 15, bank * 4 + slot); juce::String steps;
        if (auto* pats = xml->getChildByName ("Patterns")) steps = pats->getStringAttribute ("p" + juce::String (idx));
        auto* d = new juce::DynamicObject(); d->setProperty ("pattern", idx); d->setProperty ("steps", steps);
        // All sixteen patterns (4096 digits), for saving a sound with its patterns: sent when they change and every few seconds, not on every heartbeat (the page keeps the last it saw).
        if (auto* pats = xml->getChildByName ("Patterns"))
        {
            juce::String all; for (int k = 0; k < 16; ++k) all << pats->getStringAttribute ("p" + juce::String (k));
            if (all.length() == 4096 && (sent == nullptr || all != sent->last || (++sent->calls % 5) == 0)) { d->setProperty ("all", all); if (sent != nullptr) sent->last = all; }
        }
        o->setProperty ("drums", juce::var (d));
        if (diagnose) { o->setProperty ("rootTag", xml->getTagName()); names.removeDuplicates (false); o->setProperty ("childTags", names.joinIntoString (",")); o->setProperty ("stepsLength", steps.length()); }
        return out;
    }
    // Where the saved connections are kept: one entry per copy of a device (its link id), plus, from before copies had their own, one entry per device kind.
    // The site the devices connect to (askarchie.net; a test copy may point elsewhere)
    static juce::String siteBase() { auto b = juce::SystemStats::getEnvironmentVariable ("ASKARCHIE_SITE", {}); return b.isNotEmpty() ? b : juce::String ("https://askarchie.net"); }
    juce::File linkTokenFile() const { return settingsDir.getChildFile (juce::SystemStats::getEnvironmentVariable ("ASKARCHIE_HOST_TEST", {}) == "1" ? "device-links-test.json" : "device-links.json"); }
    bool linkIdInUse (const juce::String& id)
    {
        for (int chain : allChainIds()) if (auto* t = chainAt (chain)) for (auto& s : t->slots) if (s->linkId == id) return true;
        return false;
    }
    // A device that was connected before reconnects by itself when it is loaded (a new session, a reopened song), so the website keeps showing it without pressing ARCHIE again.
    void autoLinkIfSaved (int chain, int slot)
    {
        auto* t = chainAt (chain); if (t == nullptr || slot < 0 || slot >= (int) t->slots.size()) return;
        auto& s = *t->slots[(size_t) slot]; const auto kind = linkKindOf (s); if (kind.isEmpty()) return;
        const auto file = linkTokenFile(); if (! file.existsAsFile()) return;
        auto v = juce::JSON::parse (file); if (! v.isObject()) return;
        if (s.linkId.isNotEmpty()) { if (v.getProperty (s.linkId, {}).toString().isNotEmpty()) { juce::String e; editDeviceLink (chain, slot, true, e); } return; }
        if (v.getProperty (kind, {}).toString().isNotEmpty())        // an older connection kept under the device kind: the first copy to load takes it over
        {
            s.linkId = "L" + juce::Uuid().toDashedString().removeCharacters ("-").substring (0, 10);
            v.getDynamicObject()->setProperty (s.linkId, v.getProperty (kind, {})); v.getDynamicObject()->setProperty (kind, juce::String());
            file.replaceWithText (juce::JSON::toString (v, false));
            juce::String e; editDeviceLink (chain, slot, true, e);
        }
    }
    bool editDeviceLink (int chain, int slot, bool on, juce::String& err)
    {
        auto* t = chainAt (chain); if (t == nullptr || slot < 0 || slot >= (int) t->slots.size()) { err = "no such device"; return false; }
        auto& s = *t->slots[(size_t) slot]; const auto kind = linkKindOf (s);
        if (kind.isEmpty()) { err = "this device does not connect to Ask Archie"; return false; }
        if (! on) { if (s.link != nullptr) s.link->stop(); return true; }
        if (s.link == nullptr)
        {
            if (s.linkId.isEmpty()) s.linkId = "L" + juce::Uuid().toDashedString().removeCharacters ("-").substring (0, 10);
            const auto tokenFile = linkTokenFile();
            s.link = std::make_unique<archielink::DeviceLink> (*s.proc, kind, runtimeIdForLinks() + "-" + s.linkId, tokenFile, archielink::realHttp (siteBase()), [] (const juce::String& path) { if (juce::SystemStats::getEnvironmentVariable ("ASKARCHIE_LINK_NO_BROWSER", {}) != "1") juce::URL (siteBase() + path).launchInDefaultBrowser(); });
            s.link->setTokenKey (s.linkId);
            if (kind == "archie-drums")
                { auto sent = std::make_shared<DrumsExtrasState>(); s.link->setExtras ([sent] (juce::AudioProcessor& p) -> juce::var { return archieDrumsExtras (p, false, sent); }); }
            s.link->setHostInfo ([this] (juce::DynamicObject& h) { h.setProperty ("playing", transport.playing.load()); h.setProperty ("recording", transport.recording.load()); h.setProperty ("bpm", transport.bpm.load()); h.setProperty ("ppqPosition", transport.posBeats.load()); h.setProperty ("sentAtMs", (double) juce::Time::currentTimeMillis()); h.setProperty ("timeSigNumerator", transport.tsNum.load()); h.setProperty ("timeSigDenominator", transport.tsDen.load()); });
        }
        if (s.link->state() == archielink::DeviceLink::Connected || s.link->state() == archielink::DeviceLink::Waiting) return true;
        s.link->stop(); s.link->start(); return true;
    }
    juce::var slotsVar (TrackState& t)
    {
        juce::Array<juce::var> sl;
        for (auto& s : t.slots) { auto* so = new juce::DynamicObject(); so->setProperty ("name", s->desc.name); so->setProperty ("maker", s->desc.manufacturerName); so->setProperty ("bypass", s->bypass.load()); so->setProperty ("latency", juce::jmax (0, s->proc->getLatencySamples())); so->setProperty ("instrument", s->desc.isInstrument); so->setProperty ("linkable", linkKindOf (*s).isNotEmpty()); so->setProperty ("linkId", s->linkId); if (s->link != nullptr) { so->setProperty ("link", archielink::DeviceLink::stateName (s->link->state())); so->setProperty ("linkMessage", s->link->message()); } else so->setProperty ("link", "off"); sl.add (juce::var (so)); }
        return sl;
    }
    juce::var chainsVar()
    {
        juce::Array<juce::var> a;
        for (auto& r : returns) { auto* o = new juce::DynamicObject(); o->setProperty ("index", r->index); o->setProperty ("name", r->name); o->setProperty ("gainDb", r->gainDb.load()); o->setProperty ("pan", r->pan.load()); o->setProperty ("mute", r->mute.load()); { int nl = 0, no = 0; for (const auto& l : r->autoSnap.load()->lanes) { ++nl; if (l.on) ++no; } o->setProperty ("autoLanes", nl); o->setProperty ("autoOn", no); } o->setProperty ("latency", chainLatency (*r)); o->setProperty ("slots", slotsVar (*r)); a.add (juce::var (o)); }
        return a;
    }


    juce::var meters()
    {
        auto* o = new juce::DynamicObject();
        juce::Array<juce::var> a;
        for (int ti = 0; ti < trackCount.load(); ++ti) { auto& t = tracks[(size_t) ti]; juce::Array<juce::var> p; p.add (t->peakL.exchange (0.0f)); p.add (t->peakR.exchange (0.0f)); a.add (juce::var (p)); }
        o->setProperty ("tracks", a);
        { juce::Array<juce::var> ga; for (int ti = 0; ti < trackCount.load(); ++ti) ga.add ((double) tracks[(size_t) ti]->compGr.exchange (0.0f)); o->setProperty ("gr", ga); }
        { juce::Array<juce::var> gg; for (int ti = 0; ti < trackCount.load(); ++ti) gg.add ((double) tracks[(size_t) ti]->gateGr.exchange (0.0f)); o->setProperty ("gg", gg); }          // the gates' reduction, dB          // the compressors' gain reduction, dB
        { juce::Array<juce::var> ia; for (int ti = 0; ti < trackCount.load(); ++ti) ia.add (tracks[(size_t) ti]->inPeak.exchange (0.0f)); o->setProperty ("ins", ia); }
        { juce::Array<juce::var> ra; for (auto& r : returns) { juce::Array<juce::var> p; p.add (r->peakL.exchange (0.0f)); p.add (r->peakR.exchange (0.0f)); ra.add (juce::var (p)); } o->setProperty ("returns", ra); }
        juce::Array<juce::var> m; if (master != nullptr) { m.add (master->peakL.exchange (0.0f)); m.add (master->peakR.exchange (0.0f)); }
        o->setProperty ("master", m);
        {   // the level into and out of every device in the racks, by chain (a track is its own number, an FX return is 100 + its number, the master section is 200): [[in, out], ...] for each device
            auto* sp = new juce::DynamicObject();
            auto chainPk = [] (TrackState& t) { juce::Array<juce::var> a; for (auto& s : t.slots) { juce::Array<juce::var> p; p.add (s != nullptr ? (double) s->inPk.exchange (0.0f) : 0.0); p.add (s != nullptr ? (double) s->outPk.exchange (0.0f) : 0.0); a.add (juce::var (p)); } return juce::var (a); };
            for (int ti = 0; ti < trackCount.load(); ++ti) sp->setProperty (juce::String (ti), chainPk (*tracks[(size_t) ti]));
            for (size_t ri = 0; ri < returns.size(); ++ri) sp->setProperty (juce::String (100 + (int) ri), chainPk (*returns[ri]));
            if (masterChain != nullptr) sp->setProperty ("200", chainPk (*masterChain));
            o->setProperty ("slotPk", juce::var (sp));
        }
        o->setProperty ("mgr", master != nullptr ? (double) master->cGr.exchange (0.0f) : 0.0);
        o->setProperty ("amGr", master != nullptr ? (double) master->amGr.exchange (0.0f) : 0.0);          // the Archie sound limiter: dB turned down
        o->setProperty ("lufsM", master != nullptr ? (double) master->lm.momentary.load() : -100.0); o->setProperty ("lufsS", master != nullptr ? (double) master->lm.shortTerm.load() : -100.0); o->setProperty ("lufsI", master != nullptr ? (double) master->lm.integrated.load() : -100.0);      // loudness in LUFS: momentary (400 ms), short-term (3 s) and integrated (the song so far, gated)
        o->setProperty ("amPush", master != nullptr ? (double) master->amPush.load() : 0.0); o->setProperty ("amState", master != nullptr ? master->amState.load() : 0);      // 0 no target, 1 following it, 2 holding on target, 3 cannot reach it      // (push moves by itself when a loudness target is set)
        o->setProperty ("avg", master != nullptr ? 10.0 * std::log10 (juce::jmax (1.0e-10, (double) master->avgMsOut.load())) : -100.0);      // the average loudness of the mix, dBFS (about 3 seconds)          // the master compressor's gain reduction, dB
        return juce::var (o);
    }

    juce::var transportVar()
    {
        auto* o = new juce::DynamicObject();
        o->setProperty ("playing", transport.playing.load()); o->setProperty ("recording", transport.recording.load()); o->setProperty ("looping", transport.looping.load());
        o->setProperty ("beats", transport.posBeats.load()); o->setProperty ("bpm", transport.bpm.load()); o->setProperty ("bpmNow", transport.tempoNow()->bpmAt (transport.posBeats.load()));
        { juce::Array<juce::var> tm; for (const auto& te : transport.tempoNow()->entries()) { auto* x = new juce::DynamicObject(); x->setProperty ("beat", te.beat); x->setProperty ("bpm", te.bpm); x->setProperty ("tsNum", te.tsNum); x->setProperty ("tsDen", te.tsDen); tm.add (juce::var (x)); } o->setProperty ("tempoMap", tm); }
        o->setProperty ("tsNum", transport.tsNum.load()); o->setProperty ("tsDen", transport.tsDen.load()); o->setProperty ("ppqn", transport.ppqn.load());
        o->setProperty ("swing", transport.swing.load()); o->setProperty ("swingGrid", transport.swingGrid.load());
        const bool pendingLoop = transport.loopPending.load(); o->setProperty ("loopStart", pendingLoop ? transport.pendStart.load() : transport.loopStart.load()); o->setProperty ("loopEnd", pendingLoop ? transport.pendEnd.load() : transport.loopEnd.load());      // a loop waiting for its cue point shows where it is going
        o->setProperty ("loopPending", pendingLoop); o->setProperty ("loopJump", transport.loopJump.load()); o->setProperty ("cues", cuesVar()); o->setProperty ("activeCue", transport.activeCue.load()); o->setProperty ("playingCue", transport.playingCue.load()); o->setProperty ("cueLearn", cueLearn.load());
        o->setProperty ("syncMode", transport.syncMode.load()); o->setProperty ("fps", transport.fps.load());
        o->setProperty ("clockOut", transport.clockOut.load()); o->setProperty ("mtcOut", transport.mtcOut.load()); o->setProperty ("mmcOut", transport.mmcOut.load()); o->setProperty ("mmcIn", transport.mmcIn.load());
        o->setProperty ("mmcDevice", transport.mmcDeviceId.load()); o->setProperty ("tcOffset", transport.tcOffsetSeconds.load());
        o->setProperty ("timecode", transport.timecodeNow());
        o->setProperty ("seconds", transport.tempoNow()->secondsAt (transport.posBeats.load()));
        o->setProperty ("metronome", metronome.load());
        { const double ago = juce::Time::getMillisecondCounterHiRes() - lastMidiMs.load(); o->setProperty ("midiAgoMs", lastMidiMs.load() > 0 ? ago : 1.0e9); const juce::SpinLock::ScopedLockType lock (midiTextLock); o->setProperty ("midiLast", lastMidiText); }
        o->setProperty ("monitorTrack", monitorTrack.load()); o->setProperty ("pdcSamples", pdcTotal.load()); o->setProperty ("workers", renderer.workerCount());
        o->setProperty ("countIn", transport.countInBars.load()); o->setProperty ("countingIn", transport.countingIn.load()); o->setProperty ("countInEnd", transport.countInEndBeat.load());
        return juce::var (o);
    }

    //==========================================================================
    // projects
    juce::var saveSlots (TrackState& t)
    {
        juce::Array<juce::var> sl;
        for (auto& s : t.slots)
        {
            auto* so = new juce::DynamicObject();
            so->setProperty ("desc", s->desc.createXml()->toString()); so->setProperty ("bypass", s->bypass.load());
            if (s->linkId.isNotEmpty()) so->setProperty ("linkId", s->linkId);
            juce::MemoryBlock mb; s->proc->getStateInformation (mb);
            so->setProperty ("state", mb.toBase64Encoding());
            sl.add (juce::var (so));
        }
        return sl;
    }

    // One track as a value (the same thing a saved project holds for it): its sound source, level, pan, colour, automation, clips, plug-ins with their settings, and sends.
    juce::var trackVar (int ti)
    {
            auto& t = *tracks[(size_t) ti]; auto* o = new juce::DynamicObject();
            o->setProperty ("name", t.name); o->setProperty ("kind", t.kind.load()); o->setProperty ("inChan", t.inChan.load()); o->setProperty ("inStereo", t.inStereo.load()); o->setProperty ("inGainDb", (double) t.inGainDb.load()); o->setProperty ("inInvert", t.inInvert.load()); o->setProperty ("patPlay", t.patPlay.load()); o->setProperty ("patTrans", t.patTrans.load()); o->setProperty ("eq", eqVar (t.index)); o->setProperty ("width", (double) t.width.load()); o->setProperty ("groove", t.grooveCh.load()); o->setProperty ("gateOn", t.gateOn.load()); o->setProperty ("gateExp", t.gateExp.load()); o->setProperty ("gateFast", t.gateFast.load()); o->setProperty ("gateThr", (double) t.gateThr.load()); o->setProperty ("gateRange", (double) t.gateRange.load()); o->setProperty ("gateRel", (double) t.gateRel.load()); o->setProperty ("gateHold", (double) t.gateHold.load()); o->setProperty ("compOn", t.compOn.load()); o->setProperty ("compFast", t.compFast.load()); o->setProperty ("compPeak", t.compPeak.load()); o->setProperty ("compThr", (double) t.compThr.load()); o->setProperty ("compRatio", (double) t.compRatio.load()); o->setProperty ("compRel", (double) t.compRel.load()); o->setProperty ("compMakeup", (double) t.compMakeup.load()); o->setProperty ("eqHpOn", t.eqHpOn.load()); o->setProperty ("eqHpHz", (double) t.eqHpHz.load()); o->setProperty ("eqLpOn", t.eqLpOn.load()); o->setProperty ("eqLpHz", (double) t.eqLpHz.load()); o->setProperty ("monitor", t.monitor.load()); o->setProperty ("source", t.source.load()); o->setProperty ("gainDb", t.gainDb.load()); o->setProperty ("pan", t.pan.load());
            o->setProperty ("mute", t.mute.load()); o->setProperty ("solo", t.solo.load()); o->setProperty ("armed", t.armed.load());
            o->setProperty ("colour", t.colour.load()); o->setProperty ("automation", autoVar (t.index)); o->setProperty ("remotes", remotesVar (t.index));
            o->setProperty ("filePath", t.filePath); o->setProperty ("clip", clipToVar (t)); o->setProperty ("arrangement", arrangementVar (t));
            juce::Array<juce::var> sl;
            for (auto& s : t.slots)
            {
                auto* so = new juce::DynamicObject();
                so->setProperty ("desc", s->desc.createXml()->toString()); so->setProperty ("bypass", s->bypass.load());
            if (s->linkId.isNotEmpty()) so->setProperty ("linkId", s->linkId);
                juce::MemoryBlock mb; s->proc->getStateInformation (mb);
                so->setProperty ("state", mb.toBase64Encoding());
                sl.add (juce::var (so));
            }
            o->setProperty ("slots", sl);
            juce::Array<juce::var> sends; for (int s = 0; s < kSends; ++s) { auto* so = new juce::DynamicObject(); so->setProperty ("db", sendEff (t, s)); so->setProperty ("lvl", t.sendLvl[s].load()); so->setProperty ("on", t.sendOn[s].load()); so->setProperty ("pre", t.sendPre[s].load()); sends.add (juce::var (so)); }
            o->setProperty ("sends", sends); o->setProperty ("patterns", patternCountOf (t));
        return juce::var (o);
    }
    // A track made from such a value, added at the end. Clips are NEW copies (not shared with the track it came from). Plug-ins load in the background; `done` is told when they are all in (empty text = fine).
    // Returns the new track's number, or -1 if there is no room.
    int addTrackFromVar (const juce::var& tv, const juce::String& newName, std::function<void (juce::String)> done)
    {
        if (! addTrack()) return -1;
        const int i = trackCount.load() - 1; auto& t = *tracks[(size_t) i];
        t.colour.store (juce::jlimit (-1, kPaletteSize - 1, (int) tv.getProperty ("colour", -1)));
        t.name = newName.isNotEmpty() ? newName : tv.getProperty ("name", t.name).toString(); t.kind.store (juce::jlimit (0, 1, (int) tv.getProperty ("kind", 0))); t.inGainDb.store ((float) juce::jlimit ((double) InputStage::kMinDb, (double) InputStage::kMaxDb, (double) tv.getProperty ("inGainDb", 0.0))); t.inInvert.store ((bool) tv.getProperty ("inInvert", false)); t.patPlay.store (juce::jlimit (0, 2, (int) tv.getProperty ("patPlay", 1))); t.patTrans.store (juce::jlimit (-24, 24, (int) tv.getProperty ("patTrans", 0))); loadEqVar (t.index, tv.getProperty ("eq", {})); t.width.store ((float) juce::jlimit (0.0, (double) WidthStage::kMax, (double) tv.getProperty ("width", 1.0))); t.grooveCh.store (juce::jlimit (-1, kGrooveChans - 1, (int) tv.getProperty ("groove", -1)) >= 1 ? (int) tv.getProperty ("groove", -1) : -1); t.gateOn.store ((bool) tv.getProperty ("gateOn", false)); t.gateExp.store ((bool) tv.getProperty ("gateExp", false)); t.gateFast.store ((bool) tv.getProperty ("gateFast", false)); t.gateThr.store ((float) juce::jlimit ((double) GateStage::kThrMin, (double) GateStage::kThrMax, (double) tv.getProperty ("gateThr", -45.0))); t.gateRange.store ((float) juce::jlimit (0.0, (double) GateStage::kRangeMax, (double) tv.getProperty ("gateRange", 40.0))); t.gateRel.store ((float) juce::jlimit ((double) GateStage::kRelMin, (double) GateStage::kRelMax, (double) tv.getProperty ("gateRel", 0.3))); t.gateHold.store ((float) juce::jlimit (0.0, (double) GateStage::kHoldMax, (double) tv.getProperty ("gateHold", 0.05))); t.compOn.store ((bool) tv.getProperty ("compOn", false)); t.compFast.store ((bool) tv.getProperty ("compFast", false)); t.compPeak.store ((bool) tv.getProperty ("compPeak", true)); t.compThr.store ((float) juce::jlimit ((double) CompStage::kThrMin, (double) CompStage::kThrMax, (double) tv.getProperty ("compThr", -18.0))); t.compRatio.store ((float) juce::jlimit ((double) CompStage::kRatioMin, (double) CompStage::kRatioMax, (double) tv.getProperty ("compRatio", 4.0))); t.compRel.store ((float) juce::jlimit ((double) CompStage::kRelMin, (double) CompStage::kRelMax, (double) tv.getProperty ("compRel", 0.3))); t.compMakeup.store ((float) juce::jlimit (0.0, (double) CompStage::kMakeMax, (double) tv.getProperty ("compMakeup", 0.0))); t.eqHpOn.store ((bool) tv.getProperty ("eqHpOn", false)); t.eqLpOn.store ((bool) tv.getProperty ("eqLpOn", false)); t.eqHpHz.store ((float) juce::jlimit ((double) EqStage::kHpMin, (double) EqStage::kHpMax, (double) tv.getProperty ("eqHpHz", 80.0))); t.eqLpHz.store ((float) juce::jlimit ((double) EqStage::kLpMin, (double) EqStage::kLpMax, (double) tv.getProperty ("eqLpHz", 12000.0))); t.inChan.store (juce::jlimit (0, kMaxInputs - 1, (int) tv.getProperty ("inChan", 0))); t.inStereo.store ((bool) tv.getProperty ("inStereo", true)); t.monitor.store ((bool) tv.getProperty ("monitor", true)); t.source.store ((int) tv.getProperty ("source", 0));
        t.gainDb.store ((float) (double) tv.getProperty ("gainDb", 0.0)); t.pan.store ((float) (double) tv.getProperty ("pan", 0.0));
        t.mute.store ((bool) tv.getProperty ("mute", false)); t.solo.store (false); t.armed.store (false);
        std::map<int, std::shared_ptr<Clip>> knownContent;
        if (tv.getProperty ("automation", {}).isArray()) loadAutoVar (i, tv.getProperty ("automation", {})); loadRemotesVar (i, tv.getProperty ("remotes", {}));
        if (tv.getProperty ("arrangement", {}).isObject()) loadArrangementVar (i, tv.getProperty ("arrangement", {}), knownContent);
        else setClip (i, tv.getProperty ("clip", {}));
        const auto fp = tv.getProperty ("filePath", {}).toString(); t.filePath = fp; if (fp.isNotEmpty()) loadAudioFile (i, juce::File (fp));
        if (auto* sends = tv.getProperty ("sends", {}).getArray()) for (int k = 0; k < sends->size() && k < kSends; ++k) loadSendVar (t, k, (*sends)[k]);
        auto pending = std::make_shared<int> (1); auto errors = std::make_shared<juce::StringArray>();
        auto finish = [pending, errors, done] { if (--(*pending) == 0 && done) done (errors->joinIntoString ("; ")); };
        if (auto* sl = tv.getProperty ("slots", {}).getArray())
            for (const auto& sv : *sl)
            {
                auto d = savedDesc (sv);
                if (crashedBeforeDesc (d)) { errors->add (d.name + " was left out: it stopped the program while loading before (load it by hand with Try it anyway)"); continue; }
                const bool bypass = (bool) sv.getProperty ("bypass", false); ++(*pending);
                addPluginAsync (i, d, sv.getProperty ("state", {}).toString(), [this, i, bypass, errors, finish] (juce::String err)
                { if (err.isNotEmpty()) errors->add (err); else if (bypass) setBypass (i, (int) tracks[(size_t) i]->slots.size() - 1, true); finish(); }, sv.getProperty ("linkId", {}).toString());
            }
        finish(); stateDirty.store (true);
        return i;
    }
    // the channel clipboard (copy / cut / paste of whole channels) and duplicating
    juce::var channelClipboard;
    bool copyChannel (int ti) { if (trackAt (ti) == nullptr) return false; channelClipboard = trackVar (ti); return true; }
    bool hasChannelClipboard() const { return channelClipboard.isObject(); }
    int editPasteChannel (const juce::var& tv, const juce::String& name, std::function<void (juce::String)> done)
    {
        const int i = addTrackFromVar (tv, name, std::move (done)); if (i < 0) return -1;
        const juce::String nm = tracks[(size_t) i]->name;
        undo.record ("Paste channel", "", [this] { removeLastTrack(); }, [this, tv, nm] { addTrackFromVar (tv, nm, nullptr); });
        return i;
    }
    int editDuplicateChannel (int ti, std::function<void (juce::String)> done)
    {
        if (trackAt (ti) == nullptr) return -1; return editPasteChannel (trackVar (ti), (tracks[(size_t) ti]->name + " copy").substring (0, 40), std::move (done));
    }

    // An empty song, as a project value: the starting tracks with nothing on them, no plug-ins anywhere, 120 bpm in 4/4. Loading it is "New".
    // the program's own start (and "New"): no tracks, only the Transport and the Master
    void startEmpty() { while (trackCount.load() > 0) removeLastTrack(); monitorTrack.store (0); stateDirty.store (true); }
    juce::var blankProject (int nTracks = kNewSongTracks)
    {
        auto* root = new juce::DynamicObject(); root->setProperty ("format", "archie-studio-project"); root->setProperty ("version", 1); root->setProperty ("masterDb", 0.0);
        { auto* ms = new juce::DynamicObject(); ms->setProperty ("amOn", true); ms->setProperty ("amCut", true); ms->setProperty ("amDrive", 0.25); ms->setProperty ("amCeil", -1.0); root->setProperty ("masterSound", juce::var (ms)); }      // a new song has the Archie sound on (a little drive, a ceiling at -1 dB)
        juce::Array<juce::var> trs;
        for (int i = 0; i < nTracks; ++i)
        {
            auto* o = new juce::DynamicObject(); o->setProperty ("name", "Track " + juce::String (i + 1)); o->setProperty ("source", 0); o->setProperty ("gainDb", 0.0); o->setProperty ("pan", 0.0);
            o->setProperty ("mute", false); o->setProperty ("solo", false); o->setProperty ("armed", false); o->setProperty ("colour", -1);
            o->setProperty ("automation", juce::Array<juce::var>()); o->setProperty ("clip", juce::var (new juce::DynamicObject())); o->setProperty ("filePath", ""); o->setProperty ("slots", juce::Array<juce::var>());
            juce::Array<juce::var> sends; for (int s = 0; s < kSends; ++s) { auto* so = new juce::DynamicObject(); so->setProperty ("db", -100.0); so->setProperty ("pre", false); sends.add (juce::var (so)); }
            o->setProperty ("sends", sends); trs.add (juce::var (o));
        }
        root->setProperty ("tracks", trs);
        juce::Array<juce::var> rv; for (int r = 0; r < kReturns; ++r) { auto* o = new juce::DynamicObject(); o->setProperty ("name", "FX " + juce::String (r + 1)); o->setProperty ("gainDb", 0.0); o->setProperty ("pan", 0.0); o->setProperty ("mute", false); o->setProperty ("slots", juce::Array<juce::var>()); o->setProperty ("automation", juce::Array<juce::var>()); rv.add (juce::var (o)); }
        root->setProperty ("returns", rv);
        auto* mo = new juce::DynamicObject(); mo->setProperty ("slots", juce::Array<juce::var>()); root->setProperty ("masterSection", juce::var (mo));
        return juce::var (root);
    }

    juce::var saveProject()
    {
        auto* root = new juce::DynamicObject();
        root->setProperty ("format", "archie-studio-project"); root->setProperty ("version", 1);
        auto* tp = new juce::DynamicObject();
        tp->setProperty ("bpm", transport.bpm.load()); tp->setProperty ("tsNum", transport.tsNum.load()); tp->setProperty ("tsDen", transport.tsDen.load());
        { juce::Array<juce::var> tm; for (const auto& te : transport.tempoNow()->entries()) { auto* x = new juce::DynamicObject(); x->setProperty ("beat", te.beat); x->setProperty ("bpm", te.bpm); x->setProperty ("tsNum", te.tsNum); x->setProperty ("tsDen", te.tsDen); tm.add (juce::var (x)); } tp->setProperty ("tempoMap", tm); }
        tp->setProperty ("cues", cuesVar()); tp->setProperty ("activeCue", transport.activeCue.load()); tp->setProperty ("loopJump", transport.loopJump.load()); tp->setProperty ("ppqn", transport.ppqn.load()); tp->setProperty ("loop", transport.looping.load()); tp->setProperty ("loopStart", transport.loopStart.load()); tp->setProperty ("loopEnd", transport.loopEnd.load());
        tp->setProperty ("swing", transport.swing.load()); tp->setProperty ("swingGrid", transport.swingGrid.load()); tp->setProperty ("fps", transport.fps.load()); tp->setProperty ("tcOffset", transport.tcOffsetSeconds.load()); tp->setProperty ("countIn", transport.countInBars.load());
        root->setProperty ("transport", juce::var (tp));
        root->setProperty ("masterDb", master != nullptr ? master->gainDb.load() : 0.0f); root->setProperty ("masterComp", masterCompVar()); root->setProperty ("masterEq", eqVar (kMasterChain)); root->setProperty ("masterSound", masterSoundVar()); root->setProperty ("controlRoom", controlRoomVar()); root->setProperty ("groove", grooveVar (true));
        juce::Array<juce::var> trs;
        for (int ti = 0; ti < trackCount.load(); ++ti)
        {
            trs.add (trackVar (ti));
        }
        root->setProperty ("tracks", trs);
        { juce::Array<juce::var> rv; for (auto& r : returns) { auto* o = new juce::DynamicObject(); o->setProperty ("name", r->name); o->setProperty ("gainDb", r->gainDb.load()); o->setProperty ("pan", r->pan.load()); o->setProperty ("mute", r->mute.load()); o->setProperty ("slots", saveSlots (*r)); o->setProperty ("automation", autoVar (r->index)); rv.add (juce::var (o)); } root->setProperty ("returns", rv); }
        { auto* mo = new juce::DynamicObject(); mo->setProperty ("slots", saveSlots (*masterChain)); mo->setProperty ("automation", autoVar (kMasterChain)); root->setProperty ("masterSection", juce::var (mo)); }
        return juce::var (root);
    }

    // A plug-in saved in a song is found again by its file. When that file is not there (another computer, another user name, a folder that moved) it is found by its name and maker among the plug-ins this computer knows.
    juce::PluginDescription savedDesc (const juce::var& sv) const
    {
        juce::PluginDescription d; if (auto xml = juce::XmlDocument::parse (sv.getProperty ("desc", {}).toString())) d.loadFromXml (*xml);
        if (d.name.isNotEmpty() && juce::File::isAbsolutePath (d.fileOrIdentifier) && ! juce::File (d.fileOrIdentifier).exists())
            for (const auto& k : knownPlugins.getTypes()) if (k.name == d.name && k.pluginFormatName == d.pluginFormatName && k.manufacturerName == d.manufacturerName && juce::File (k.fileOrIdentifier).exists()) return k;
        return d;
    }
    void loadProject (const juce::var& p, std::function<void (juce::String)> done)
    {
        if (p.getProperty ("format", {}).toString() != "archie-studio-project") { done ("this is not an Archie Studio project file"); return; }
        loadWarnings.clear();
        undo.clear();                                  // a different project: the old history would put things back into the wrong one
        transport.stop(); transport.locate (0.0);
        {
            std::vector<std::unique_ptr<Slot>> retired;
            auto take = [&] (TrackState& t) { for (auto& s : t.slots) { s->window.reset(); retired.push_back (std::move (s)); } t.slots.clear(); };
            for (auto& t : tracks) { take (*t); for (int s = 0; s < kSends; ++s) { t->sendDb[s].store (-100.0f); t->sendLvl[s].store (-60.0f); t->sendOn[s].store (false); t->sendPre[s].store (false); } }
            for (auto& r : returns) take (*r);
            take (*masterChain);
            publishPlan(); quiesce();                  // nothing the audio thread can reach is destroyed until it has moved on
            retired.clear();
            for (auto& t : tracks) for (int s = 0; s < kSends; ++s) t->sendProc[s].reset();
        }
        auto tp = p.getProperty ("transport", {});
        transport.setBpm ((double) tp.getProperty ("bpm", 120.0)); transport.setTimeSignature ((int) tp.getProperty ("tsNum", 4), (int) tp.getProperty ("tsDen", 4));
        if (auto* tm = tp.getProperty ("tempoMap", {}).getArray()) { std::vector<TempoEntry> es; for (const auto& x : *tm) { TempoEntry te; te.beat = (double) x.getProperty ("beat", 0.0); te.bpm = (double) x.getProperty ("bpm", 120.0); te.tsNum = (int) x.getProperty ("tsNum", 4); te.tsDen = (int) x.getProperty ("tsDen", 4); es.push_back (te); } if (! es.empty()) transport.setTempoEntries (es); }
        transport.countInBars.store (juce::jlimit (0, 4, (int) tp.getProperty ("countIn", 0))); transport.swing.store (juce::jlimit (0.0, 1.0, (double) tp.getProperty ("swing", 0.0))); transport.swingGrid.store ((double) tp.getProperty ("swingGrid", 0.5) == 0.25 ? 0.25 : 0.5); transport.ppqn.store ((int) tp.getProperty ("ppqn", 480)); transport.fps.store ((int) tp.getProperty ("fps", 3)); transport.tcOffsetSeconds.store ((double) tp.getProperty ("tcOffset", 0.0));
        transport.loopPending.store (false); transport.loopJump.store (juce::jlimit (0, 4, (int) tp.getProperty ("loopJump", 0))); {      // the cues first (a song without them has only cue 1: its loop)
            for (auto& c : transport.cues) c.set.store (false); const auto cl = tp.getProperty ("cues", {});
            if (auto* ca = cl.getArray()) for (int i = 0; i < Transport::kCues && i < ca->size(); ++i) { const auto& cv = (*ca)[i]; const double s = (double) cv.getProperty ("start", 0.0), e = (double) cv.getProperty ("end", 16.0); if ((bool) cv.getProperty ("set", false) && e > s + 0.01) { transport.cues[(size_t) i].start.store (s); transport.cues[(size_t) i].end.store (e); transport.cues[(size_t) i].set.store (true); } }
            const int act = juce::jlimit (0, Transport::kCues - 1, (int) tp.getProperty ("activeCue", 0)); transport.activeCue.store (act == 0 || transport.cues[(size_t) act].set.load() ? act : 0);
        }
        transport.setLoop ((bool) tp.getProperty ("loop", false), (double) tp.getProperty ("loopStart", 0.0), (double) tp.getProperty ("loopEnd", 16.0));
        if (master != nullptr) master->gainDb.store ((float) (double) p.getProperty ("masterDb", 0.0));
        loadMasterComp (p); loadMasterSound (p); loadEqVar (kMasterChain, p.getProperty ("masterEq", {})); loadControlRoom (p); loadGroove (p);
        std::map<int, std::shared_ptr<Clip>> knownContent;
        auto pending = std::make_shared<int> (0);
        auto errors = std::make_shared<juce::StringArray>();
        auto finish = [this, pending, errors, done] { if (--(*pending) == 0) { for (auto& w : loadWarnings) errors->add (w); undo.clear(); stateDirty.store (true); done (errors->joinIntoString ("; ")); } };
        ++(*pending);   // guard so we do not finish before everything has been queued
        {   // the project's own number of tracks
            const int want = juce::jlimit (0, kTracks, p.getProperty ("tracks", {}).getArray() != nullptr ? p.getProperty ("tracks", {}).getArray()->size() : kStartTracks);
            while (trackCount.load() < want) addTrack();
            while (trackCount.load() > want) removeLastTrack();
        }
        // Folders are no longer part of Archie DAW. An older project that has some opens with its tracks going straight to the master (no folder is created, so no track can be parented to one).
        if (auto* fv = p.getProperty ("folders", {}).getArray()) if (fv->size() > 0) loadWarnings.add ("this project had " + juce::String (fv->size()) + " folder(s); folders are no longer in Archie DAW, so its tracks now go straight to the master");
        if (auto* arr = p.getProperty ("tracks", {}).getArray())
            for (int i = 0; i < arr->size() && i < trackCount.load(); ++i)
            {
                const auto& tv = (*arr)[i]; auto& t = *tracks[(size_t) i];
                { t.colour.store (juce::jlimit (-1, kPaletteSize - 1, (int) tv.getProperty ("colour", -1))); }
                t.name = tv.getProperty ("name", t.name).toString(); t.kind.store (juce::jlimit (0, 1, (int) tv.getProperty ("kind", 0))); t.inGainDb.store ((float) juce::jlimit ((double) InputStage::kMinDb, (double) InputStage::kMaxDb, (double) tv.getProperty ("inGainDb", 0.0))); t.inInvert.store ((bool) tv.getProperty ("inInvert", false)); t.patPlay.store (juce::jlimit (0, 2, (int) tv.getProperty ("patPlay", 1))); t.patTrans.store (juce::jlimit (-24, 24, (int) tv.getProperty ("patTrans", 0))); loadEqVar (t.index, tv.getProperty ("eq", {})); t.width.store ((float) juce::jlimit (0.0, (double) WidthStage::kMax, (double) tv.getProperty ("width", 1.0))); t.grooveCh.store (juce::jlimit (-1, kGrooveChans - 1, (int) tv.getProperty ("groove", -1)) >= 1 ? (int) tv.getProperty ("groove", -1) : -1); t.gateOn.store ((bool) tv.getProperty ("gateOn", false)); t.gateExp.store ((bool) tv.getProperty ("gateExp", false)); t.gateFast.store ((bool) tv.getProperty ("gateFast", false)); t.gateThr.store ((float) juce::jlimit ((double) GateStage::kThrMin, (double) GateStage::kThrMax, (double) tv.getProperty ("gateThr", -45.0))); t.gateRange.store ((float) juce::jlimit (0.0, (double) GateStage::kRangeMax, (double) tv.getProperty ("gateRange", 40.0))); t.gateRel.store ((float) juce::jlimit ((double) GateStage::kRelMin, (double) GateStage::kRelMax, (double) tv.getProperty ("gateRel", 0.3))); t.gateHold.store ((float) juce::jlimit (0.0, (double) GateStage::kHoldMax, (double) tv.getProperty ("gateHold", 0.05))); t.compOn.store ((bool) tv.getProperty ("compOn", false)); t.compFast.store ((bool) tv.getProperty ("compFast", false)); t.compPeak.store ((bool) tv.getProperty ("compPeak", true)); t.compThr.store ((float) juce::jlimit ((double) CompStage::kThrMin, (double) CompStage::kThrMax, (double) tv.getProperty ("compThr", -18.0))); t.compRatio.store ((float) juce::jlimit ((double) CompStage::kRatioMin, (double) CompStage::kRatioMax, (double) tv.getProperty ("compRatio", 4.0))); t.compRel.store ((float) juce::jlimit ((double) CompStage::kRelMin, (double) CompStage::kRelMax, (double) tv.getProperty ("compRel", 0.3))); t.compMakeup.store ((float) juce::jlimit (0.0, (double) CompStage::kMakeMax, (double) tv.getProperty ("compMakeup", 0.0))); t.eqHpOn.store ((bool) tv.getProperty ("eqHpOn", false)); t.eqLpOn.store ((bool) tv.getProperty ("eqLpOn", false)); t.eqHpHz.store ((float) juce::jlimit ((double) EqStage::kHpMin, (double) EqStage::kHpMax, (double) tv.getProperty ("eqHpHz", 80.0))); t.eqLpHz.store ((float) juce::jlimit ((double) EqStage::kLpMin, (double) EqStage::kLpMax, (double) tv.getProperty ("eqLpHz", 12000.0))); t.inChan.store (juce::jlimit (0, kMaxInputs - 1, (int) tv.getProperty ("inChan", 0))); t.inStereo.store ((bool) tv.getProperty ("inStereo", true)); t.monitor.store ((bool) tv.getProperty ("monitor", true)); t.source.store ((int) tv.getProperty ("source", 0));
                t.gainDb.store ((float) (double) tv.getProperty ("gainDb", 0.0)); t.pan.store ((float) (double) tv.getProperty ("pan", 0.0));
                t.mute.store ((bool) tv.getProperty ("mute", false)); t.solo.store ((bool) tv.getProperty ("solo", false)); t.armed.store ((bool) tv.getProperty ("armed", false));
                if (tv.getProperty ("automation", {}).isArray()) loadAutoVar (i, tv.getProperty ("automation", {})); loadRemotesVar (i, tv.getProperty ("remotes", {}));
                if (tv.getProperty ("arrangement", {}).isObject()) loadArrangementVar (i, tv.getProperty ("arrangement", {}), knownContent);
                else setClip (i, tv.getProperty ("clip", {}));         // a project saved before clips were instances
                const auto fp = tv.getProperty ("filePath", {}).toString();
                t.filePath = fp; if (fp.isNotEmpty()) loadAudioFile (i, juce::File (fp));
                if (auto* sl = tv.getProperty ("slots", {}).getArray())
                    for (const auto& sv : *sl)
                    {
                        auto d = savedDesc (sv);
                if (crashedBeforeDesc (d)) { errors->add (d.name + " was left out: it stopped the program while loading before (load it by hand with Try it anyway)"); continue; }
                        const bool bypass = (bool) sv.getProperty ("bypass", false);
                        ++(*pending);
                        addPluginAsync (i, d, sv.getProperty ("state", {}).toString(), [this, i, bypass, errors, finish] (juce::String err)
                        {
                            if (err.isNotEmpty()) errors->add (err);
                            else if (bypass) setBypass (i, (int) tracks[(size_t) i]->slots.size() - 1, true);
                            finish();
                        }, sv.getProperty ("linkId", {}).toString());
                    }
            }
        // sends, FX returns and the master section
        if (auto* arr = p.getProperty ("tracks", {}).getArray())
            for (int i = 0; i < arr->size() && i < trackCount.load(); ++i)
                if (auto* sends = (*arr)[i].getProperty ("sends", {}).getArray())
                    for (int s = 0; s < sends->size() && s < kSends; ++s) loadSendVar (*tracks[(size_t) i], s, (*sends)[s]);
        { const auto rets = p.getProperty ("returns", {}); for (int r = 0; r < kReturns; ++r) { const auto rv = rets.getArray() != nullptr && r < rets.getArray()->size() ? (*rets.getArray())[r] : juce::var(); const auto sl = rv.getProperty ("slots", {}); if (sl.getArray() == nullptr || sl.getArray()->isEmpty()) for (int i = 0; i < trackCount.load(); ++i) { auto& t = *tracks[(size_t) i]; if (! t.sendOn[r].load() && t.sendLvl[r].load() == 0.0f) { t.sendLvl[r].store (-60.0f); applySend (t, r, false); } } } }      // nothing on a return and a send left at the old starting level: it shows off (a level someone set, or a send that is on, is kept: an effect that failed to load must not cost the mix)
        auto restoreChain = [this, pending, errors, finish] (int chainId, const juce::var& slotsVar)
        {
            if (auto* sl = slotsVar.getArray())
                for (const auto& sv : *sl)
                {
                    auto d = savedDesc (sv);
                if (crashedBeforeDesc (d)) { errors->add (d.name + " was left out: it stopped the program while loading before (load it by hand with Try it anyway)"); continue; }
                    const bool bypass = (bool) sv.getProperty ("bypass", false);
                    ++(*pending);
                    addPluginAsync (chainId, d, sv.getProperty ("state", {}).toString(), [this, chainId, bypass, errors, finish] (juce::String err)
                    {
                        if (err.isNotEmpty()) errors->add (err);
                        else if (bypass) { if (auto* c = chainAt (chainId)) setBypass (chainId, (int) c->slots.size() - 1, true); }
                        finish();
                    }, sv.getProperty ("linkId", {}).toString());
                }
        };
        if (auto* rv = p.getProperty ("returns", {}).getArray())
            for (int r = 0; r < rv->size() && r < kReturns; ++r)
            {
                const auto& v = (*rv)[r]; auto& rt = *returns[(size_t) r];
                rt.name = v.getProperty ("name", rt.name).toString(); rt.gainDb.store ((float) (double) v.getProperty ("gainDb", 0.0)); rt.pan.store ((float) (double) v.getProperty ("pan", 0.0)); rt.mute.store ((bool) v.getProperty ("mute", false));
                restoreChain (kReturnBase + r, v.getProperty ("slots", {})); if (v.getProperty ("automation", {}).isArray()) loadAutoVar (kReturnBase + r, v.getProperty ("automation", {}));
            }
        restoreChain (kMasterChain, p.getProperty ("masterSection", {}).getProperty ("slots", {}));
        loadAutoVar (kMasterChain, p.getProperty ("masterSection", {}).getProperty ("automation", {}));      // the master's automation lanes (the Archie sound)
        updateSolo(); updateTargets();
        finish();
    }

    //==========================================================================
    Transport transport;
    juce::AudioDeviceManager deviceManager;
    juce::AudioPluginFormatManager formatManager;
    juce::KnownPluginList knownPlugins;
    std::atomic<bool> metronome { false }, stateDirty { true };
    std::atomic<double> lastMidiMs { 0.0 }, lastPlayedMs { 0.0 };
    std::atomic<int> monitorTrack { 0 };
    juce::SpinLock midiTextLock; juce::String lastMidiText;
    std::unique_ptr<MasterProcessor> masterOwner;
    MasterProcessor* master = nullptr;
    juce::StringArray extraPluginFolders;
    juce::File settingsDir;
    double sampleRate = 48000.0; int blockSize = 512;
    juce::String lastDeviceError;

private:
    void buildMixer()
    {
        masterOwner = std::make_unique<MasterProcessor>(); master = masterOwner.get(); if (masterChain != nullptr) { masterChain->blk = &blk; master->autoFrom = masterChain.get(); } prepareProc (*masterOwner);
        for (auto& r : returns) { r->stripProc = std::make_unique<TrackStripProcessor> (*r); prepareProc (*r->stripProc); }
        for (int i = 0; i < trackCount.load(); ++i) ensureTrackNodes (i);
        publishPlan();
    }

    // The audio thread only ever follows a RenderPlan: an immutable description of who feeds whom and how much each path is delayed. Every edit that changes the shape builds a new plan and publishes it.
    // PLUG-IN DELAY COMPENSATION: a plug-in that reports a latency of N samples delays its chain by N. So that every path reaches the output lined up:
    //   - all tracks are delayed to match the slowest track (their sends too, because a send leaves the track after that point);
    //   - the dry sum is delayed by the slowest FX return, and each return by the difference to the slowest, so returns line up with the dry signal and with each other;
    //   - the master inserts then delay everything equally (their latency is the last term of the total).
    static int chainLatency (const TrackState& t) { int l = 0; for (auto& s : t.slots) l += juce::jmax (0, s->proc->getLatencySamples()); return l; }
    static std::vector<PlugRef> chainRefs (TrackState& t) { std::vector<PlugRef> v; for (auto& s : t.slots) v.push_back ({ s->proc.get(), s.get() }); return v; }
    void publishPlan()
    {
        auto plan = std::make_unique<RenderPlan>();
        std::vector<int> lat;                                           // per plan track: the delay its own inserts cause
        for (int ti = 0; ti < trackCount.load(); ++ti)
        {
            auto& t = *tracks[(size_t) ti];
            if (t.sourceProc == nullptr || t.stripProc == nullptr) continue;
            TrackPlan tp; tp.ts = &t; tp.index = ti; tp.inserts = chainRefs (t);
            for (size_t si = 0; si < t.slots.size(); ++si) if (t.slots[si]->desc.isInstrument) tp.inputAt = si + 1;        // the input stage sits just after the last instrument
            for (int s = 0; s < kSends; ++s)
                if (t.sendProc[s] != nullptr && t.sendOn[s].load()) { tp.sends[s] = { t.sendProc[s].get(), t.sendPre[s].load() }; tp.anyPre = tp.anyPre || t.sendPre[s].load(); renderer.ensureSendBuffer (ti, s); }
            lat.push_back (chainLatency (t));
            plan->tracks.push_back (std::move (tp));
        }
        // every track goes straight to the master mix: all are delayed to line up with the slowest one
        int rootMax = 0;
        for (size_t i = 0; i < plan->tracks.size(); ++i) { plan->rootTracks.push_back ((int) i); rootMax = juce::jmax (rootMax, lat[i]); }
        for (int ti : plan->rootTracks) plan->tracks[(size_t) ti].delay = rootMax - lat[(size_t) ti];
        int tapMax = 0;                                                  // sends leave a track after its inserts: their taps are lined up with each other the same way
        for (auto& tp : plan->tracks) { bool any = false; for (int s = 0; s < kSends; ++s) any = any || tp.sends[s].proc != nullptr; if (any) tapMax = juce::jmax (tapMax, rootMax); }
        for (size_t i = 0; i < plan->tracks.size(); ++i)
        {
            auto& tp = plan->tracks[i];
            for (int s = 0; s < kSends; ++s)
                if (tp.sends[s].proc != nullptr) { tp.sendExtra[s] = tapMax - rootMax; if (tp.sendExtra[s] > 0) renderer.ensureSendDelay (tp.index, s); }
        }
        int retMax = 0; std::array<int, kReturns> retLat {};
        for (int r = 0; r < kReturns; ++r)
        {
            auto& rt = *returns[(size_t) r]; auto& rp = plan->returns[(size_t) r];
            if (rt.stripProc == nullptr) continue;
            rp.ts = &rt; rp.chain = chainRefs (rt); retLat[(size_t) r] = chainLatency (rt);
            for (size_t i = 0; i < plan->tracks.size(); ++i) if (plan->tracks[i].sends[r].proc != nullptr) rp.fromTracks.push_back ((int) i);
            rp.active = ! rp.chain.empty() || ! rp.fromTracks.empty();
            if (! rp.fromTracks.empty()) retMax = juce::jmax (retMax, retLat[(size_t) r]);
        }
        bool anyRet = false; for (int r = 0; r < kReturns; ++r) anyRet = anyRet || ! plan->returns[(size_t) r].fromTracks.empty();
        const int pDry = rootMax, pRet = tapMax + retMax, Z = anyRet ? juce::jmax (pDry, pRet) : pDry;      // the dry mix and the returns arrive together at Z
        for (int r = 0; r < kReturns; ++r) if (! plan->returns[(size_t) r].fromTracks.empty()) plan->returns[(size_t) r].delay = (retMax - retLat[(size_t) r]) + (Z - pRet);
        plan->dryMixDelay = Z - pDry;
        plan->masterChain = chainRefs (*masterChain); plan->master = master;
        plan->trackLatency = rootMax; plan->returnLatency = retMax; plan->masterLatency = chainLatency (*masterChain);
        plan->totalLatency = Z + plan->masterLatency;
        pdcTotal.store (plan->totalLatency);
        latencySignature = signatureOf (*plan);
        renderer.publish (std::move (plan));
    }
    static int signatureOf (const RenderPlan& p) { int h = p.trackLatency * 31 + p.returnLatency * 17 + p.masterLatency; for (auto& t : p.tracks) h = h * 7 + t.delay; for (auto& r : p.returns) h = h * 5 + r.delay; return h; }
    // a plug-in can change the latency it reports (a lookahead limiter, a mode switch): noticed here, and the plan is rebuilt
    void refreshLatency()
    {
        std::vector<int> lat; int h = 0;
        for (int ti = 0; ti < trackCount.load(); ++ti) h = h * 13 + chainLatency (*tracks[(size_t) ti]);
        for (auto& r : returns) h = h * 13 + chainLatency (*r);
        h = h * 13 + chainLatency (*masterChain);
        if (h != lastLatencyHash) { lastLatencyHash = h; publishPlan(); stateDirty.store (true); }
    }
    // Wait (on the message thread) until the audio thread has finished any block that started before the last publish: after that it cannot be reading anything the previous plan pointed at.
    void quiesce()
    {
        if (! audioActive.load()) return;
        const auto e = renderer.epoch.load();
        const auto until = juce::Time::getMillisecondCounter() + 400;
        while (renderer.epoch.load() < e + 2 && juce::Time::getMillisecondCounter() < until) juce::Thread::sleep (1);
    }

    void updatePlayHead (const Segment& seg)
    {
        auto& i = playHead.info;
        const double bpm = seg.map != nullptr ? seg.map->bpmAt (seg.startBeat) : transport.bpm.load();
        i.setBpm (bpm);
        i.setTimeSignature (juce::AudioPlayHead::TimeSignature { seg.map != nullptr ? seg.map->tsNumAt (seg.startBeat) : transport.tsNum.load(), seg.map != nullptr ? seg.map->tsDenAt (seg.startBeat) : transport.tsDen.load() });
        // A piece that ends at the loop end must not LOOK as if it runs a fraction of a sample past it: the sound card asks for whole samples, so the piece is rounded up, and a self-sequencing plug-in (Archie Drums,
        // Acid 303) works out the steps in the piece from its start position and its length. Told a piece that reaches 0.00001 beats beyond the loop end, it plays the step that sits ON the loop end, and then the step
        // at the loop start in the next piece: the first note after the wrap sounded twice. So the position it is told is moved back by that sliver, and the piece ends at the loop end exactly.
        double ppqStart = seg.startBeat;
        if (transport.looping.load()) { const double le = transport.loopEnd.load(), span = (double) seg.numSamples * bpm / 60.0 / sampleRate; if (seg.startBeat < le && seg.endBeat >= le - 1.0e-6) ppqStart = juce::jmin (seg.startBeat, le - span - 1.0e-7); }
        i.setPpqPosition (ppqStart);
        i.setTimeInSeconds (seg.startSec);
        i.setTimeInSamples ((juce::int64) std::llround (seg.startSec * sampleRate));
        i.setIsPlaying (transport.playing.load()); i.setIsRecording (transport.recording.load()); i.setIsLooping (transport.looping.load());
        i.setLoopPoints (juce::AudioPlayHead::LoopPoints { transport.loopStart.load(), transport.loopEnd.load() });
        const double bar = (double) transport.tsNum.load() * 4.0 / (double) transport.tsDen.load();
        i.setPpqPositionOfLastBarStart (std::floor (seg.startBeat / bar) * bar);
        auto fr = juce::AudioPlayHead::FrameRate();
        switch ((Fps) transport.fps.load())
        {
            case Fps::F24: fr = fr.withBaseRate (24); break;
            case Fps::F25: fr = fr.withBaseRate (25); break;
            case Fps::F2997Drop: fr = fr.withBaseRate (30).withDrop().withPullDown(); break;
            case Fps::F30: fr = fr.withBaseRate (30); break;
        }
        i.setFrameRate (fr);
    }

    void addMetronome (juce::AudioBuffer<float>& buf, const Segment (&segs)[2], int ns)
    {
        if ((! metronome.load() && ! transport.countingIn.load()) || ! transport.playing.load()) return;
        const int num = transport.tsNum.load();
        auto* l = buf.getWritePointer (0); auto* r = buf.getWritePointer (1);
        for (int s = 0; s < ns; ++s)
        {
            const auto& sg = segs[s];
            const double b0 = sg.startBeat, b1 = sg.endBeat;
            long long k = (long long) std::ceil (b0 - 1.0e-9);
            std::array<int, 8> hits {}; std::array<bool, 8> acc {}; int nh = 0;
            for (; (double) k < b1 - 1.0e-12 && nh < 8; ++k) { hits[(size_t) nh] = (int) std::lround (sg.sampleOfBeat ((double) k)); acc[(size_t) nh] = (((k % num) + num) % num) == 0; ++nh; }
            int hi = 0;
            for (int i = 0; i < sg.numSamples; ++i)
            {
                const int at = sg.startSample + i;
                while (hi < nh && hits[(size_t) hi] <= at) { clickRemain = (int) (sampleRate * 0.03); clickFreq = acc[(size_t) hi] ? 1500.0 : 1000.0; clickPhase = 0; ++hi; }
                if (clickRemain > 0)
                {
                    const float env = (float) clickRemain / (float) (sampleRate * 0.03);
                    const float v = 0.25f * env * std::sin ((float) clickPhase);
                    l[at] += v; r[at] += v;
                    clickPhase += 2.0 * juce::MathConstants<double>::pi * clickFreq / sampleRate; --clickRemain;
                }
            }
        }
    }

    std::array<std::unique_ptr<TrackState>, kTracks> tracks;
    std::array<std::unique_ptr<TrackState>, kReturns> returns;
    std::unique_ptr<TrackState> masterChain;
    MixRenderer renderer;
    juce::AudioBuffer<float> outBuf;
    std::atomic<bool> audioActive { false };
    int renderMax = 1024, latencySignature = 0, lastLatencyHash = 0, latTick = 0;
    BlockInfo blk;
    HostPlayHead playHead;
    juce::AudioBuffer<float> scratch;
    juce::MidiBuffer outMidi;
    std::unique_ptr<juce::MidiOutput> midiOut; juce::SpinLock midiOutLock; juce::String midiOutId;
    juce::StringArray openMidiInputs, disabledMidiIns; juce::String lastMidiDeviceKey; juce::CriticalSection midiOpenLock; std::map<juce::String, bool> midiOpenOk; int midiPollTick = 0;
    std::unique_ptr<ScanThread> scanThread; juce::CriticalSection scanLock; ScanProgress scanState; AnalysisState analysisState; juce::var lastAnalysis; std::unique_ptr<AnalysisThread> analysisThread;
    bool shutDown = false;
    int clickRemain = 0; double clickFreq = 1000.0, clickPhase = 0.0;
};

} // namespace archie
