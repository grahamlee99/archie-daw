// Archie Studio host (desktop) -- the MIXER RENDERER.
// Replaces the general-purpose audio graph with the fixed shape a mixer really has, so it can be faster, exact about delay, and predictable:
//
//   stage 1 (PARALLEL, one task per track):   source -> inserts -> [PRE tap] -> fader/pan -> [POST tap] -> sends
//   stage 2 (PARALLEL, one task per return):  sum of the sends that feed it -> its own inserts -> its strip
//   stage 3 (one thread, fixed order):        sum of every track and return -> master inserts -> master level -> out
//
// * The sums in stage 3 are always made in the same order, so the result is bit-identical whether one thread or four did the work.
// * PLUG-IN DELAY COMPENSATION: every path is delayed so that it reaches the output lined up with every other path (see Engine::publishPlan, which works out the delays from each chain's reported latency).
// * Nothing here waits for the message thread: edits publish a new immutable RenderPlan (Snapshot) which the audio thread picks up at the start of its next block.
// * The same render() is used for real time and for offline tests/bounces.
#pragma once
#include "Mixer.h"
#include <cstdint>
#include <condition_variable>
#include <mutex>
#if JUCE_INTEL && JUCE_MSVC
 #include <intrin.h>
#endif

namespace archie
{

//==============================================================================
// A few high-priority worker threads plus the audio thread itself share the tasks of one stage. parallelFor() returns when every task is done.
class WorkerPool
{
public:
    ~WorkerPool() { stop(); }

    void start (int numWorkers)
    {
        stop();
        running.store (true);
        for (int i = 0; i < numWorkers; ++i) { auto w = std::make_unique<Worker> (*this); w->startThread (juce::Thread::Priority::highest); workers.push_back (std::move (w)); }
    }
    void stop()
    {
        running.store (false);
        cv.notify_all();
        for (auto& w : workers) w->stopThread (2000);
        workers.clear();
    }
    int size() const { return (int) workers.size(); }

    template <typename F>
    void parallelFor (int count, F&& fn)
    {
        if (count <= 0) return;
        if (workers.empty() || count == 1) { for (int i = 0; i < count; ++i) fn (i); return; }
        using Fn = std::remove_reference_t<F>;
        job.call = [] (void* c, int i) { (*static_cast<Fn*> (c)) (i); };
        job.ctx = (void*) &fn; job.count = count;
        done.store (0, std::memory_order_relaxed);
        const uint64_t g = ++generation;
        next.store (g << 32, std::memory_order_release);
        gen.store (g, std::memory_order_release);
        if (sleepers.load (std::memory_order_relaxed) > 0) cv.notify_all();
        work (g);                                            // the calling (audio) thread takes tasks too
        while (done.load (std::memory_order_acquire) < count) relax();
    }

private:
    struct Job { void (*call) (void*, int) = nullptr; void* ctx = nullptr; int count = 0; };

    static void relax()
    {
#if JUCE_INTEL && JUCE_MSVC
        _mm_pause();
#else
        std::this_thread::yield();
#endif
    }
    // claim tasks of generation g until none are left; a worker that arrives late for an old generation claims nothing
    void work (uint64_t g)
    {
        for (;;)
        {
            uint64_t v = next.load (std::memory_order_acquire);
            if ((v >> 32) != g) return;
            const int idx = (int) (v & 0xffffffffu);
            if (idx >= job.count) return;
            if (! next.compare_exchange_weak (v, v + 1, std::memory_order_acq_rel)) continue;
            job.call (job.ctx, idx);
            done.fetch_add (1, std::memory_order_release);
        }
    }

    struct Worker : public juce::Thread
    {
        explicit Worker (WorkerPool& p) : juce::Thread ("Mixer worker"), pool (p) {}
        void run() override
        {
#if JUCE_WINDOWS
            // Windows' multimedia class scheduler: a thread that registers as "Pro Audio" is scheduled like the audio thread itself, which keeps it from being starved by other programs.
            {
                juce::DynamicLibrary avrt ("avrt.dll");
                using Fn = void* (__stdcall*) (const wchar_t*, unsigned long*);
                if (auto fn = (Fn) avrt.getFunction ("AvSetMmThreadCharacteristicsW")) { unsigned long idx = 0; fn (L"Pro Audio", &idx); }
                avrt.close();
            }
#endif
            uint64_t seen = 0; int idle = 0;
            while (pool.running.load() && ! threadShouldExit())
            {
                const uint64_t g = pool.gen.load (std::memory_order_acquire);
                if (g != seen) { seen = g; idle = 0; pool.work (g); continue; }
                if (++idle < 1500) relax();
                else
                {
                    std::unique_lock<std::mutex> lk (pool.m);
                    pool.sleepers.fetch_add (1);
                    pool.cv.wait_for (lk, std::chrono::milliseconds (2), [this, seen] { return pool.gen.load() != seen || ! pool.running.load(); });
                    pool.sleepers.fetch_sub (1);
                }
            }
        }
        WorkerPool& pool;
    };

    std::vector<std::unique_ptr<Worker>> workers;
    std::atomic<bool> running { false };
    std::atomic<uint64_t> gen { 0 }, next { 0 };
    std::atomic<int> done { 0 }, sleepers { 0 };
    uint64_t generation = 0;
    Job job;
    std::mutex m; std::condition_variable cv;
};

//==============================================================================
// A plain delay of a whole number of samples (stereo). It always keeps writing, so a delay that is raised later reads real history, never garbage.
class StereoDelay
{
public:
    static constexpr int kCapacity = 1 << 17;      // 131072 samples: about 2.7 s at 48 kHz
    void prepare() { ring.setSize (2, kCapacity); ring.clear(); w = 0; delay = 0; }
    void setDelay (int d) { delay = juce::jlimit (0, kCapacity - 1, d); }
    int getDelay() const { return delay; }
    void process (juce::AudioBuffer<float>& b, int n)
    {
        const int mask = kCapacity - 1;
        for (int ch = 0; ch < 2; ++ch)
        {
            float* x = b.getWritePointer (ch); float* r = ring.getWritePointer (ch); int wp = w;
            if (delay == 0) { for (int i = 0; i < n; ++i) { r[wp & mask] = x[i]; ++wp; } }
            else            { for (int i = 0; i < n; ++i) { r[wp & mask] = x[i]; x[i] = r[(wp - delay) & mask]; ++wp; } }
        }
        w = (w + n) & mask;
    }
private:
    juce::AudioBuffer<float> ring; int w = 0, delay = 0;
};

//==============================================================================
struct PlugRef { juce::AudioProcessor* proc = nullptr; Slot* slot = nullptr; };
struct SendPlan { SendProcessor* proc = nullptr; bool pre = false; };
struct TrackPlan
{
    TrackState* ts = nullptr;
    std::vector<PlugRef> inserts;
    size_t inputAt = 0;                       // the channel input stage runs before this insert: just after the last instrument, or first when there is none
    SendPlan sends[kSends];
    bool anyPre = false;
    int index = 0, delay = 0;                 // delay: samples added so this track lines up with the slowest track
    std::vector<int> inputs; int parallelFrom = -1; bool isSource = false;      // routing: the plan tracks mixed into this bus, the track a parallel channel hears, and whether this track is heard by one
    int compKey = -1, gateKey = -1; bool isKey = false;      // side-chain: the plan tracks whose sound the compressor / gate listen to, and whether this track is anyone's key
    int sendExtra[kSends] {};                 // extra delay on a send tap so every track's tap still arrives together at the return
};
struct ReturnPlan
{
    TrackState* ts = nullptr;
    std::vector<PlugRef> chain;
    bool active = false;                      // has inserts, or receives at least one send
    std::vector<int> fromTracks;              // plan-track numbers that send into it
    int delay = 0;
};
struct RenderPlan
{
    std::vector<TrackPlan> tracks;
    std::array<ReturnPlan, kReturns> returns;
    std::vector<int> rootTracks;              // the tracks that go into the master mix, in a fixed order
    std::vector<PlugRef> masterChain;
    MasterProcessor* master = nullptr;
    std::vector<std::vector<int>> levels;     // the tracks in the order they can be rendered: a side-chain key before whoever listens to it
    int masterKey = -1, masterKeyDelay = 0;
    int dryMixDelay = 0;                      // the dry (track) sum is delayed this much to line up with the returns
    int trackLatency = 0, returnLatency = 0, masterLatency = 0, totalLatency = 0;
};

//==============================================================================
class MixRenderer
{
public:
    // message thread: allocate everything the audio thread will touch
    void prepare (double sampleRate, int maxBlockSize, int numWorkers)
    {
        sr = sampleRate; maxBlock = maxBlockSize;
        for (auto& t : tw) { t = std::make_unique<TrackWork>(); t->buf.setSize (2, maxBlock); t->pre.setSize (2, maxBlock); t->scratch.setSize (16, maxBlock); t->midi.ensureSize (32768); t->scratchMidi.ensureSize (32768); t->post.prepare(); t->preDelay.prepare(); t->keyOut.setSize (2, maxBlock); t->keyOut.clear(); t->srcOut.setSize (2, maxBlock); t->srcOut.clear(); for (auto& s : t->sendOut) s.reset(); }
        for (auto& r : rw) { r = std::make_unique<ReturnWork>(); r->buf.setSize (2, maxBlock); r->scratch.setSize (16, maxBlock); r->out.prepare(); }
        mix.setSize (2, maxBlock); mscratch.setSize (16, maxBlock); dryMix.prepare(); masterKeyBuf.setSize (2, maxBlock); masterKeyLine.prepare();
        pool.start (numWorkers);
    }
    void shutdown() { pool.stop(); }
    int workerCount() const { return pool.size(); }

    // message thread: a send needs its own output buffer before any plan that uses it is published
    void ensureSendBuffer (int track, int send)
    {
        if (track < 0 || track >= kTracks || send < 0 || send >= kSends || tw[(size_t) track] == nullptr) return;
        auto& p = tw[(size_t) track]->sendOut[send];
        if (p == nullptr) { auto b = std::make_unique<juce::AudioBuffer<float>> (2, maxBlock); b->clear(); p = std::move (b); }
    }
    // a send tap that has to be delayed to line up with taps from other tracks needs a delay line of its own (made only when needed)
    void ensureSendDelay (int track, int send)
    {
        if (track < 0 || track >= kTracks || send < 0 || send >= kSends || tw[(size_t) track] == nullptr) return;
        auto& p = tw[(size_t) track]->sendDelay[send];
        if (p == nullptr) { auto d = std::make_unique<StereoDelay>(); d->prepare(); p = std::move (d); }
    }
    void publish (std::unique_ptr<RenderPlan> plan) { plans.publish (std::move (plan)); }
    const RenderPlan* currentPlan() const { return plans.load(); }
    std::atomic<uint64_t> epoch { 0 };         // +1 every block: the message thread waits for it to move before it destroys anything a plan pointed at

    // audio thread
    void render (const juce::AudioBuffer<float>& in, juce::AudioBuffer<float>& out, int n)
    {
        epoch.fetch_add (1, std::memory_order_acq_rel);
        const RenderPlan* P = plans.load();
        if (n > maxBlock) return;
        if (P->levels.size() <= 1) runStage ((int) P->tracks.size(), [&] (int i) { renderTrack (*P, i, in, n); }, trackStage);
        else for (const auto& level : P->levels) runStage ((int) level.size(), [&] (int i) { renderTrack (*P, level[(size_t) i], in, n); }, trackStage);
        runStage (kReturns, [&] (int r) { renderReturn (*P, r, n); }, returnStage);
        renderMaster (*P, out, n);
    }

    // Waking the workers costs a few tens of microseconds, so a stage whose total work is small is simply done on the audio thread. Every task is timed (in either mode), the running average of a
    // stage's total work decides the mode, and a heavy project switches to the workers within a few blocks. A quiet program therefore leaves the workers asleep.
    void setParallelThresholdMicros (double us) { parallelThresholdUs = us; }
    double stageWorkMicros() const { return trackStage.emaUs; }

private:
    struct StageStat { double emaUs = 0.0; std::atomic<int64_t> ticks { 0 }; };
    template <typename F>
    void runStage (int count, F&& fn, StageStat& st)
    {
        st.ticks.store (0, std::memory_order_relaxed);
        auto timed = [&] (int i) { const auto t0 = juce::Time::getHighResolutionTicks(); fn (i); st.ticks.fetch_add (juce::Time::getHighResolutionTicks() - t0, std::memory_order_relaxed); };
        if (pool.size() > 0 && count > 1 && st.emaUs > parallelThresholdUs) pool.parallelFor (count, timed);
        else for (int i = 0; i < count; ++i) timed (i);
        const double us = juce::Time::highResolutionTicksToSeconds (st.ticks.load (std::memory_order_relaxed)) * 1.0e6;
        st.emaUs = st.emaUs * 0.85 + us * 0.15;
    }
    StageStat trackStage, returnStage;
    double parallelThresholdUs = 150.0;

    struct TrackWork
    {
        juce::AudioBuffer<float> buf, pre, scratch, keyOut, srcOut; juce::MidiBuffer midi, scratchMidi;
        StereoDelay post, preDelay;
        std::unique_ptr<juce::AudioBuffer<float>> sendOut[kSends];
        std::unique_ptr<StereoDelay> sendDelay[kSends];
    };
    struct ReturnWork { juce::AudioBuffer<float> buf, scratch; juce::MidiBuffer emptyMidi, scratchMidi; StereoDelay out; bool ran = false; };

    static juce::AudioBuffer<float> view (juce::AudioBuffer<float>& b, int channels, int n) { return juce::AudioBuffer<float> (b.getArrayOfWritePointers(), channels, n); }

    // one hosted processor in a chain. Channel rules (the same the graph used): a stereo-in plug-in gets L/R; a mono-in one gets L+R; one with no input gets silence and its output replaces the signal;
    // a mono-out one is copied to both sides; one with no output leaves the signal untouched (an analyser, say).
    static void runInsert (const PlugRef& ref, juce::AudioBuffer<float>& buf, const juce::MidiBuffer& midiIn, juce::AudioBuffer<float>& scratch, juce::MidiBuffer& scratchMidi, int n)
    {
        if (ref.proc == nullptr || (ref.slot != nullptr && ref.slot->bypass.load (std::memory_order_relaxed))) return;
        auto* p = ref.proc;
        if (ref.slot != nullptr && ref.slot->resetNext.exchange (false)) p->reset();             // Panic
        const int nIn = p->getTotalNumInputChannels(), nOut = p->getTotalNumOutputChannels(), ch = juce::jmin (16, juce::jmax (nIn, nOut));
        if (ch == 0) return;
        if (juce::jmax (nIn, nOut) > scratch.getNumChannels()) return;    // more channels than this program has room for: leave the sound as it is rather than let the plug-in write past the buffer (the buses are limited when it is loaded; this is the second line of defence)
        for (int c = 0; c < ch; ++c) juce::FloatVectorOperations::clear (scratch.getWritePointer (c), n);
        if (nIn >= 2) { juce::FloatVectorOperations::copy (scratch.getWritePointer (0), buf.getReadPointer (0), n); juce::FloatVectorOperations::copy (scratch.getWritePointer (1), buf.getReadPointer (1), n); }
        else if (nIn == 1) { juce::FloatVectorOperations::add (scratch.getWritePointer (0), buf.getReadPointer (0), n); juce::FloatVectorOperations::add (scratch.getWritePointer (0), buf.getReadPointer (1), n); }
        scratchMidi.clear();
        if (p->acceptsMidi()) scratchMidi.addEvents (midiIn, 0, n, 0);
        auto peakOf = [n] (const float* a, const float* b) { float pk = 0.0f; for (const float* q : { a, b }) { if (q == nullptr) continue; const auto mm = juce::FloatVectorOperations::findMinAndMax (q, n); pk = juce::jmax (pk, std::abs (mm.getStart()), std::abs (mm.getEnd())); } return pk; };
        if (ref.slot != nullptr && nIn > 0) { const float pk = peakOf (scratch.getReadPointer (0), nIn >= 2 ? scratch.getReadPointer (1) : nullptr); if (pk > ref.slot->inPk.load (std::memory_order_relaxed)) ref.slot->inPk.store (pk, std::memory_order_relaxed); }      // the level INTO the device (an instrument has no input)
        juce::AudioBuffer<float> v (scratch.getArrayOfWritePointers(), ch, n);
        p->processBlock (v, scratchMidi);
        if (ref.slot != nullptr && nOut > 0) { const float pk = peakOf (scratch.getReadPointer (0), nOut >= 2 ? scratch.getReadPointer (1) : nullptr); if (pk > ref.slot->outPk.load (std::memory_order_relaxed)) ref.slot->outPk.store (pk, std::memory_order_relaxed); }      // and OUT of it
        if (nOut >= 2) { juce::FloatVectorOperations::copy (buf.getWritePointer (0), scratch.getReadPointer (0), n); juce::FloatVectorOperations::copy (buf.getWritePointer (1), scratch.getReadPointer (1), n); }
        else if (nOut == 1) { juce::FloatVectorOperations::copy (buf.getWritePointer (0), scratch.getReadPointer (0), n); juce::FloatVectorOperations::copy (buf.getWritePointer (1), scratch.getReadPointer (0), n); }
    }

    // PATTERN BLOCKS: before a plug-in that has pattern controls runs, put its pattern (and its link to the bass off) where the block says, once per change
    static void applyPattern (const PlugRef& ref, int want, double origin)
    {
        if (ref.slot == nullptr) return;
        if (auto* h = ref.slot->patHead.get()) { h->silent.store (want == -2, std::memory_order_relaxed); h->origin.store (want >= 0 ? origin : 0.0, std::memory_order_relaxed); }
        if (want < 0) { ref.slot->patApplied.store (-1); return; }
        auto* pb = ref.slot->patBank.load (std::memory_order_relaxed); if (pb == nullptr || ref.slot->patApplied.load() == want) return;
        if (auto* pl = ref.slot->patLink.load (std::memory_order_relaxed)) pl->setValue (0.0f);
        ref.slot->patSet (want); ref.slot->patApplied.store (want);
    }

    void renderTrack (const RenderPlan& P, int i, const juce::AudioBuffer<float>& in, int n)
    {
        const auto& tp = P.tracks[(size_t) i]; auto& w = *tw[(size_t) tp.index];
        auto buf = view (w.buf, 2, n);
        // what an audio track hears: the input channel it is routed to (mono: that one on both sides; stereo: that channel and the next); any other track gets the first two inputs as before
        if (tp.ts->source.load() == 1)
        {
            const int avail = in.getNumChannels(), c0 = tp.ts->inChan.load(), c1 = tp.ts->inStereo.load() ? c0 + 1 : c0;
            if (c0 < 0 || c0 >= avail) { buf.clear(); }
            else { juce::FloatVectorOperations::copy (buf.getWritePointer (0), in.getReadPointer (c0), n); juce::FloatVectorOperations::copy (buf.getWritePointer (1), in.getReadPointer (c1 < avail ? c1 : c0), n); }
        }
        else
        {
            juce::FloatVectorOperations::copy (buf.getWritePointer (0), in.getReadPointer (0), n);
            juce::FloatVectorOperations::copy (buf.getWritePointer (1), in.getReadPointer (juce::jmin (1, in.getNumChannels() - 1)), n);
        }
        w.midi.clear();
        tp.ts->sourceProc->processBlock (buf, w.midi);
        // ROUTING: a bus takes in the tracks sent to it (already rendered, after their faders, lined up with each other); a parallel channel takes in its source track's sound
        for (int c : tp.inputs) { const auto& from = tw[(size_t) P.tracks[(size_t) c].index]->buf; for (int ch = 0; ch < 2; ++ch) juce::FloatVectorOperations::add (buf.getWritePointer (ch), from.getReadPointer (ch), n); }
        if (tp.parallelFrom >= 0) { const auto& from = tw[(size_t) P.tracks[(size_t) tp.parallelFrom].index]->srcOut; for (int ch = 0; ch < 2; ++ch) juce::FloatVectorOperations::add (buf.getWritePointer (ch), from.getReadPointer (ch), n); }
        {   // the channel input stage: automation first (the input gain can be automated), then gain and polarity where the sound enters the effects
            auto& ts = *tp.ts;
            if (ts.blk != nullptr && ts.blk->playing && ts.blk->numSegs > 0)
            {
                const auto* as = ts.autoSnap.load(); const double beat = ts.blk->segs[0].startBeat;
                const bool touching = ts.blk->blockStartMs < ts.autoTouchUntilMs.load();
                for (const auto& lane : as->lanes) if (lane.kind == 3 && lane.on && ! lane.pts.empty() && ! touching) ts.inGainDb.store ((float) juce::jlimit ((double) InputStage::kMinDb, (double) InputStage::kMaxDb, autoValueAt (lane, beat)));
            }
            const float inDb = ts.inGainDb.load(); const bool inInv = ts.inInvert.load();
            for (size_t k = 0; k <= tp.inserts.size(); ++k)
            {
                if (k == tp.inputAt)
                {
                    if (tp.isSource) for (int c = 0; c < 2; ++c) juce::FloatVectorOperations::copy (w.srcOut.getWritePointer (c), buf.getReadPointer (c), n);
                    ts.inputStage.process (buf, n, inDb, inInv, sr);
                    const juce::AudioBuffer<float>* cKey = tp.compKey >= 0 ? &tw[(size_t) P.tracks[(size_t) tp.compKey].index]->keyOut : nullptr;
                    const juce::AudioBuffer<float>* gKey = tp.gateKey >= 0 ? &tw[(size_t) P.tracks[(size_t) tp.gateKey].index]->keyOut : nullptr;
                    { const float gr = ts.compStage.process (buf, n, ts.compOn.load(), ts.compThr.load(), ts.compRatio.load(), ts.compRel.load(), ts.compFast.load(), ts.compPeak.load(), ts.compMakeup.load(), sr, -1.0f, cKey != nullptr ? cKey->getReadPointer (0) : nullptr, cKey != nullptr ? cKey->getReadPointer (1) : nullptr); if (gr > ts.compGr.load (std::memory_order_relaxed)) ts.compGr.store (gr, std::memory_order_relaxed); }
                    { const float gg = ts.gateStage.process (buf, n, ts.gateOn.load(), ts.gateExp.load(), ts.gateFast.load(), ts.gateThr.load(), ts.gateRange.load(), ts.gateRel.load(), ts.gateHold.load(), sr, gKey != nullptr ? gKey->getReadPointer (0) : nullptr, gKey != nullptr ? gKey->getReadPointer (1) : nullptr); if (gg > ts.gateGr.load (std::memory_order_relaxed)) ts.gateGr.store (gg, std::memory_order_relaxed); }
                    if (ts.blk != nullptr && ts.blk->playing && ts.blk->numSegs > 0) { const auto* eas = ts.autoSnap.load(); if (eas != nullptr) for (const auto& lane : eas->lanes) if (lane.kind == kEqLaneKind && lane.on && ! lane.pts.empty()) applyEqLane (ts.eq, ts.eqHpOn, ts.eqHpHz, ts.eqLpOn, ts.eqLpHz, lane.param, autoValueAt (lane, ts.blk->segs[0].startBeat)); }      // EQ sweeps: the EQ lanes, block by block
                    ts.eqStage.process (buf, n, ts.eqHpOn.load(), ts.eqHpHz.load(), ts.eqLpOn.load(), ts.eqLpHz.load(), sr);
                    ts.paraEq.process (buf, n, ts.eq, sr); ts.eq.feed (buf, n);
                }
                if (k < tp.inserts.size()) { applyPattern (tp.inserts[k], ts.patWant.load (std::memory_order_relaxed), ts.patOrigin.load (std::memory_order_relaxed)); runInsert (tp.inserts[k], buf, w.midi, w.scratch, w.scratchMidi, n); }
            }
        }
        tp.ts->widthStage.process (buf, n, tp.ts->width.load(), sr);                                    // the channel's width: after its effects, before the sends tap and the pan
        if (tp.isKey) for (int c = 0; c < 2; ++c) juce::FloatVectorOperations::copy (w.keyOut.getWritePointer (c), buf.getReadPointer (c), n);      // the key: after the effects, before the fader, so a muted track still keys
        if (tp.anyPre) { auto pre = view (w.pre, 2, n); juce::FloatVectorOperations::copy (pre.getWritePointer (0), buf.getReadPointer (0), n); juce::FloatVectorOperations::copy (pre.getWritePointer (1), buf.getReadPointer (1), n); }
        juce::MidiBuffer none;
        tp.ts->stripProc->processBlock (buf, none);
        w.post.setDelay (tp.delay); w.post.process (buf, n);
        if (tp.anyPre) { auto pre = view (w.pre, 2, n); w.preDelay.setDelay (tp.delay); w.preDelay.process (pre, n); }
        auto preV = view (w.pre, 2, n);
        for (int s = 0; s < kSends; ++s)
            if (tp.sends[s].proc != nullptr && w.sendOut[s] != nullptr)
            {
                auto out = view (*w.sendOut[s], 2, n);
                const auto& src = tp.sends[s].pre ? preV : buf;
                juce::FloatVectorOperations::copy (out.getWritePointer (0), src.getReadPointer (0), n);
                juce::FloatVectorOperations::copy (out.getWritePointer (1), src.getReadPointer (1), n);
                if (w.sendDelay[s] != nullptr) { w.sendDelay[s]->setDelay (tp.sendExtra[s]); w.sendDelay[s]->process (out, n); }
                tp.sends[s].proc->processBlock (out, none);
            }
    }

    void renderReturn (const RenderPlan& P, int r, int n)
    {
        const auto& rp = P.returns[(size_t) r]; auto& w = *rw[(size_t) r];
        w.ran = false;
        if (! rp.active) return;
        auto buf = view (w.buf, 2, n);
        buf.clear();
        for (int ti : rp.fromTracks)
        {
            const auto& tp = P.tracks[(size_t) ti]; auto& src = tw[(size_t) tp.index]->sendOut[r];
            if (src == nullptr) continue;
            juce::FloatVectorOperations::add (buf.getWritePointer (0), src->getReadPointer (0), n);
            juce::FloatVectorOperations::add (buf.getWritePointer (1), src->getReadPointer (1), n);
        }
        for (const auto& ins : rp.chain) runInsert (ins, buf, w.emptyMidi, w.scratch, w.scratchMidi, n);
        juce::MidiBuffer none;
        rp.ts->stripProc->processBlock (buf, none);
        w.out.setDelay (rp.delay); w.out.process (buf, n);
        w.ran = true;
    }

    void renderMaster (const RenderPlan& P, juce::AudioBuffer<float>& out, int n)
    {
        auto m = view (mix, 2, n);
        m.clear();
        for (int ti : P.rootTracks)              // fixed order: always the same sum (the tracks in order)
        {
            auto& w = *tw[(size_t) P.tracks[(size_t) ti].index];
            juce::FloatVectorOperations::add (m.getWritePointer (0), w.buf.getReadPointer (0), n);
            juce::FloatVectorOperations::add (m.getWritePointer (1), w.buf.getReadPointer (1), n);
        }
        dryMix.setDelay (P.dryMixDelay); dryMix.process (m, n);
        for (int r = 0; r < kReturns; ++r)
            if (rw[(size_t) r]->ran)
            {
                juce::FloatVectorOperations::add (m.getWritePointer (0), rw[(size_t) r]->buf.getReadPointer (0), n);
                juce::FloatVectorOperations::add (m.getWritePointer (1), rw[(size_t) r]->buf.getReadPointer (1), n);
            }
        juce::MidiBuffer none;
        for (const auto& ins : P.masterChain) runInsert (ins, m, none, mscratch, mscratchMidi, n);
        if (P.master != nullptr)
        {
            P.master->keyL = P.master->keyR = nullptr;
            if (P.masterKey >= 0)      // the master compressor's key, lined up with the mix
            {
                auto k = view (masterKeyBuf, 2, n); const auto& src = tw[(size_t) P.tracks[(size_t) P.masterKey].index]->keyOut;
                for (int c = 0; c < 2; ++c) juce::FloatVectorOperations::copy (k.getWritePointer (c), src.getReadPointer (c), n);
                masterKeyLine.setDelay (P.masterKeyDelay); masterKeyLine.process (k, n); P.master->keyL = k.getReadPointer (0); P.master->keyR = k.getReadPointer (1);
            }
            P.master->processBlock (m, none);
        }
        for (int ch = 0; ch < out.getNumChannels(); ++ch) juce::FloatVectorOperations::copy (out.getWritePointer (ch), m.getReadPointer (juce::jmin (ch, 1)), n);
    }

    double sr = 48000.0; int maxBlock = 512;
    Snapshot<RenderPlan> plans { std::make_unique<RenderPlan>() };
    std::array<std::unique_ptr<TrackWork>, kTracks> tw;
    std::array<std::unique_ptr<ReturnWork>, kReturns> rw;
    juce::AudioBuffer<float> mix, mscratch, masterKeyBuf; juce::MidiBuffer mscratchMidi; StereoDelay dryMix, masterKeyLine;
    WorkerPool pool;
};

} // namespace archie
