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
// What one listener's trigger is made from: a plan track, an FX return (heard one block late) or a tempo pulse. delay: samples the trigger waits so it lines up with the sound it works on (for a
// pulse: how late that sound is, so the pulse is placed on the same moment of the song).
struct KeyPlan { int track = -1, ret = -1, pulse = 0, delay = 0; bool any() const { return track >= 0 || ret >= 0 || pulse != 0; } };
struct PlugRef { juce::AudioProcessor* proc = nullptr; Slot* slot = nullptr; KeyPlan key; };
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
    KeyPlan compFrom, gateFrom;                               // the same, with returns, pulses and the line-up delay
    int sendExtra[kSends] {};                 // extra delay on a send tap so every track's tap still arrives together at the return
};
struct ReturnPlan
{
    TrackState* ts = nullptr;
    std::vector<PlugRef> chain;
    bool active = false;                      // has inserts, or receives at least one send
    std::vector<int> fromTracks;              // plan-track numbers that send into it
    int delay = 0; bool isKey = false;        // isKey: something listens to this return
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
        sr = sampleRate; maxBlock = maxBlockSize; masterSeam = SeamFade();
        for (auto& t : tw) { t = std::make_unique<TrackWork>(); t->buf.setSize (2, maxBlock); t->pre.setSize (2, maxBlock); t->scratch.setSize (16, maxBlock); t->midi.ensureSize (32768); t->scratchMidi.ensureSize (32768); t->post.prepare(); t->preDelay.prepare(); t->keyOut.setSize (2, maxBlock); t->keyOut.clear(); t->srcOut.setSize (2, maxBlock); t->srcOut.clear(); t->keyWork.setSize (2, maxBlock); for (auto& s : t->sendOut) s.reset(); }
        for (auto& r : rw) { r = std::make_unique<ReturnWork>(); r->buf.setSize (2, maxBlock); r->scratch.setSize (16, maxBlock); r->out.prepare(); r->keyOut.setSize (2, maxBlock); r->keyOut.clear(); r->keyWork.setSize (2, maxBlock); }
        mix.setSize (2, maxBlock); mscratch.setSize (16, maxBlock); dryMix.prepare(); masterKeyBuf.setSize (2, maxBlock); masterKeyLine.prepare(); masterWork.setSize (2, maxBlock);
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

    // WHEN THE PLUG-INS ON A CHAIN CHANGE (added, removed, moved) the sound would jump. The jump is measured on the first sample after the change and faded away over a few milliseconds.
    struct SeamFade
    {
        std::uint64_t sig = 0; bool known = false; float last[2] { 0.0f, 0.0f }, gap[2] { 0.0f, 0.0f }; int left = 0, length = 1;
        static std::uint64_t signature (const std::vector<PlugRef>& chain) { std::uint64_t h = 1469598103934665603ull; for (const auto& r : chain) h = (h ^ (std::uint64_t) (std::uintptr_t) r.proc) * 1099511628211ull; return h; }
        void apply (juce::AudioBuffer<float>& b, int n, std::uint64_t now, int fadeSamples)
        {
            if (n <= 0) return;
            if (known && now != sig) { for (int c = 0; c < 2; ++c) gap[c] = last[c] - b.getReadPointer (c)[0]; length = left = juce::jmax (1, fadeSamples); }
            sig = now; known = true;
            for (int i = 0; i < n && left > 0; ++i, --left) { const float g = 0.5f - 0.5f * std::cos (juce::MathConstants<float>::pi * (float) left / (float) length); for (int c = 0; c < 2; ++c) b.getWritePointer (c)[i] += gap[c] * g; }
            for (int c = 0; c < 2; ++c) last[c] = b.getReadPointer (c)[n - 1];
        }
    };

    struct TrackWork
    {
        juce::AudioBuffer<float> buf, pre, scratch, keyOut, srcOut, keyWork; juce::MidiBuffer midi, scratchMidi;
        SeamFade seam;
        StereoDelay post, preDelay;
        std::unique_ptr<juce::AudioBuffer<float>> sendOut[kSends];
        std::unique_ptr<StereoDelay> sendDelay[kSends];
    };
    struct ReturnWork { juce::AudioBuffer<float> buf, scratch, keyOut, keyWork; juce::MidiBuffer emptyMidi, scratchMidi; StereoDelay out; SeamFade seam; bool ran = false; };

    static juce::AudioBuffer<float> view (juce::AudioBuffer<float>& b, int channels, int n) { return juce::AudioBuffer<float> (b.getArrayOfWritePointers(), channels, n); }

    // true when any sample is NaN, infinite or louder than 1000 (tested on the bits, so no compiler setting can optimise the NaN test away)
    static bool audioIsBroken (const float* x, int n)
    {
        juce::uint32 worst = 0; for (int i = 0; i < n; ++i) { juce::uint32 bits; std::memcpy (&bits, x + i, sizeof (bits)); worst = juce::jmax (worst, bits & 0x7fffffffu); }
        return worst > 0x447a0000u;      // 1000.0f
    }

    // A bypassed plug-in is not heard, but it must still hear keys being LET GO (or it holds them for ever and sounds them again when the bypass comes off). So in a block that carries a release, it is run on
    // silence with only the releases (note-off, pedal up, all notes or all sound off), and what it makes is thrown away.
    static void releaseWhileBypassed (juce::AudioProcessor& p, const juce::MidiBuffer& midiIn, juce::AudioBuffer<float>& scratch, juce::MidiBuffer& scratchMidi, int n)
    {
        if (midiIn.isEmpty() || ! p.acceptsMidi()) return;
        scratchMidi.clear();
        for (const auto meta : midiIn)
        {
            const auto m = meta.getMessage();
            if (m.isNoteOff() || (m.isController() && ((m.getControllerNumber() == 64 && m.getControllerValue() < 64) || m.getControllerNumber() == 120 || m.getControllerNumber() == 123))) scratchMidi.addEvent (m, meta.samplePosition);
        }
        if (scratchMidi.isEmpty()) return;
        const int need = juce::jmax (p.getTotalNumInputChannels(), p.getTotalNumOutputChannels()), ch = juce::jmin (16, need);
        if (ch == 0 || need > scratch.getNumChannels()) return;
        for (int c = 0; c < ch; ++c) juce::FloatVectorOperations::clear (scratch.getWritePointer (c), n);
        juce::AudioBuffer<float> v (scratch.getArrayOfWritePointers(), ch, n);
        p.processBlock (v, scratchMidi);
    }

    // one hosted processor in a chain. Channel rules (the same the graph used): a stereo-in plug-in gets L/R; a mono-in one gets L+R; one with no input gets silence and its output replaces the signal;
    // a mono-out one is copied to both sides; one with no output leaves the signal untouched (an analyser, say).
    static void runInsert (const PlugRef& ref, juce::AudioBuffer<float>& buf, const juce::MidiBuffer& midiIn, juce::AudioBuffer<float>& scratch, juce::MidiBuffer& scratchMidi, int n, int fadeSamples, const float* keyL = nullptr, const float* keyR = nullptr)
    {
        if (ref.proc == nullptr) return;
        if (ref.slot != nullptr && ref.slot->hold.load()) return;      // its inputs are being changed: the sound goes round it for that moment
        auto* p = ref.proc;
        auto* slot = ref.slot;
        const bool off = slot != nullptr && slot->bypass.load (std::memory_order_relaxed);
        if (slot != nullptr && slot->wet < 0.0f) slot->wet = off ? 0.0f : 1.0f;                   // a plug-in's first block starts where its switch is
        // the unprocessed path, delayed to match the plug-in (nothing is stored for a plug-in without delay, or with more than the store holds)
        int late = slot != nullptr ? juce::jmax (0, p->getLatencySamples()) : 0; if (late + n > Slot::kDry) late = 0;
        if (late > 0) for (int c = 0; c < 2; ++c) { const float* from = buf.getReadPointer (c); float* ring = slot->dry.data() + c * Slot::kDry; for (int i = 0; i < n; ++i) ring[(slot->dryPos + i) & (Slot::kDry - 1)] = from[i]; }
        auto dryAt = [&] (int c, int i) { return late > 0 ? slot->dry[(size_t) (c * Slot::kDry + ((slot->dryPos + i - late) & (Slot::kDry - 1)))] : buf.getReadPointer (c)[i]; };
        if (off && slot->wet <= 0.0f)      // fully bypassed: the plug-in is not heard (and only hears keys being let go)
        {
            releaseWhileBypassed (*p, midiIn, scratch, scratchMidi, n); slot->warm = late;
            if (late > 0) { for (int c = 0; c < 2; ++c) { float* to = buf.getWritePointer (c); for (int i = 0; i < n; ++i) to[i] = dryAt (c, i); } slot->dryPos = (slot->dryPos + n) & (Slot::kDry - 1); }
            return;
        }
        if (ref.slot != nullptr && ref.slot->resetNext.exchange (false)) p->reset();             // Panic
        const int nIn = p->getTotalNumInputChannels(), nOut = p->getTotalNumOutputChannels(), ch = juce::jmin (16, juce::jmax (nIn, nOut));
        if (ch == 0) return;
        if (juce::jmax (nIn, nOut) > scratch.getNumChannels()) return;    // more channels than this program has room for: leave the sound as it is rather than let the plug-in write past the buffer (the buses are limited when it is loaded; this is the second line of defence)
        for (int c = 0; c < ch; ++c) juce::FloatVectorOperations::clear (scratch.getWritePointer (c), n);
        if (nIn >= 2) { juce::FloatVectorOperations::copy (scratch.getWritePointer (0), buf.getReadPointer (0), n); juce::FloatVectorOperations::copy (scratch.getWritePointer (1), buf.getReadPointer (1), n); }
        else if (nIn == 1) { juce::FloatVectorOperations::add (scratch.getWritePointer (0), buf.getReadPointer (0), n); juce::FloatVectorOperations::add (scratch.getWritePointer (0), buf.getReadPointer (1), n); }
        if (slot != nullptr && keyL != nullptr)      // SIDE-CHAIN INPUT: the trigger goes on the plug-in's side-chain channels (for an instrument, its own input)
        {
            const int first = slot->scFirst.load(), count = juce::jmin (2, slot->scCount.load());
            if (count > 0 && first >= 0 && first + count <= ch)
            {
                if (count == 2) { juce::FloatVectorOperations::copy (scratch.getWritePointer (first), keyL, n); juce::FloatVectorOperations::copy (scratch.getWritePointer (first + 1), keyR, n); }
                else { float* to = scratch.getWritePointer (first); for (int i = 0; i < n; ++i) to[i] = 0.5f * (keyL[i] + keyR[i]); }
            }
        }
        scratchMidi.clear();
        if (p->acceptsMidi()) scratchMidi.addEvents (midiIn, 0, n, 0);
        auto peakOf = [n] (const float* a, const float* b) { float pk = 0.0f; for (const float* q : { a, b }) { if (q == nullptr) continue; const auto mm = juce::FloatVectorOperations::findMinAndMax (q, n); pk = juce::jmax (pk, std::abs (mm.getStart()), std::abs (mm.getEnd())); } return pk; };
        if (ref.slot != nullptr && nIn > 0) { const float pk = peakOf (scratch.getReadPointer (0), nIn >= 2 ? scratch.getReadPointer (1) : nullptr); if (pk > ref.slot->inPk.load (std::memory_order_relaxed)) ref.slot->inPk.store (pk, std::memory_order_relaxed); }      // the level INTO the device (an instrument has no input)
        juce::AudioBuffer<float> v (scratch.getArrayOfWritePointers(), ch, n);
        p->processBlock (v, scratchMidi);
        // FILTERING/MONITORING MECHANISM: the broken-audio guard. It looks at every sample a hosted device sends out and mutes that device's block when any sample is not a number, infinite, or beyond 1000 times
        // full scale (+60 dB): one such sample would otherwise stay in the mixer's filters and silence the whole mix. It does NOT judge loudness, clipping or how anything sounds below that limit, and it never
        // changes a sample of a healthy block. The limit is audioIsBroken()'s; the report to the person is Slot::broken (read by Engine::meters), and the device is reset once when the fault begins.
        if (nOut > 0)
        {
            bool bad = false; for (int c = 0; c < juce::jmin (2, nOut) && ! bad; ++c) bad = audioIsBroken (scratch.getReadPointer (c), n);
            if (bad) for (int c = 0; c < ch; ++c) juce::FloatVectorOperations::clear (scratch.getWritePointer (c), n);
            if (ref.slot != nullptr)
            {
                if (bad) { ref.slot->cleanBlocks = 0; if (! ref.slot->broken.exchange (true, std::memory_order_relaxed)) p->reset(); }
                else if (ref.slot->broken.load (std::memory_order_relaxed) && ++ref.slot->cleanBlocks > 400) ref.slot->broken.store (false, std::memory_order_relaxed);
            }
        }
        if (ref.slot != nullptr && nOut > 0) { const float pk = peakOf (scratch.getReadPointer (0), nOut >= 2 ? scratch.getReadPointer (1) : nullptr); if (pk > ref.slot->outPk.load (std::memory_order_relaxed)) ref.slot->outPk.store (pk, std::memory_order_relaxed); }      // and OUT of it
        if (slot == nullptr || (! off && slot->wet >= 1.0f))
        {
            if (nOut >= 2) { juce::FloatVectorOperations::copy (buf.getWritePointer (0), scratch.getReadPointer (0), n); juce::FloatVectorOperations::copy (buf.getWritePointer (1), scratch.getReadPointer (1), n); }
            else if (nOut == 1) { juce::FloatVectorOperations::copy (buf.getWritePointer (0), scratch.getReadPointer (0), n); juce::FloatVectorOperations::copy (buf.getWritePointer (1), scratch.getReadPointer (0), n); }
        }
        else      // the switch has moved: a straight crossfade over a few milliseconds between the unprocessed sound and the plug-in's
        {
            float w = slot->wet; const float step = (off ? -1.0f : 1.0f) / (float) juce::jmax (1, fadeSamples);
            float *l = buf.getWritePointer (0), *r = buf.getWritePointer (1);
            for (int i = 0; i < n; ++i)
            {
                if (! off && slot->warm > 0) --slot->warm; else w = juce::jlimit (0.0f, 1.0f, w + step);      // coming back: wait until the plug-in's delay has filled with the present sound
                const float dl = dryAt (0, i), dr = dryAt (1, i);
                const float wl = nOut >= 1 ? scratch.getReadPointer (0)[i] : dl, wr = nOut >= 2 ? scratch.getReadPointer (1)[i] : nOut == 1 ? scratch.getReadPointer (0)[i] : dr;
                l[i] = dl + (wl - dl) * w; r[i] = dr + (wr - dr) * w;
            }
            slot->wet = w;
        }
        if (late > 0) slot->dryPos = (slot->dryPos + n) & (Slot::kDry - 1);
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

    // Makes one listener's trigger into `work` and says whether it has one. sound = true for a plug-in's input (a pulse is then a short 60 Hz thud a detector can hear), false for the built-in
    // compressor and gate (a clean level). A channel trigger is delayed to line up and filtered by the listener's Low cut and High cut; a pulse is silent unless the song plays.
    bool makeKey (const RenderPlan& P, const KeyPlan& key, archieSC::SideTap& tap, juce::AudioBuffer<float>& work, int n, bool sound, const BlockInfo* blk)
    {
        if (! key.any()) return false;
        float* l = work.getWritePointer (0); float* r = work.getWritePointer (1);
        if (key.pulse != 0)
        {
            for (int i = 0; i < n; ++i)
            {
                float v = 0.0f;
                if (blk != nullptr && blk->playing && blk->numSegs > 0)
                {
                    const int s = blk->numSegs > 1 && i >= blk->segs[1].startSample ? 1 : 0;
                    double since = 0.0; const float level = archieSC::pulseAt (key.pulse, blk->segs[s].beatOfSample (i - key.delay), blk->bpm, since);
                    v = sound ? level * 0.9f * (float) std::sin (6.283185307179586 * 60.0 * since) : level;
                }
                l[i] = r[i] = v;
            }
            return true;
        }
        const juce::AudioBuffer<float>* from = key.track >= 0 ? &tw[(size_t) P.tracks[(size_t) key.track].index]->keyOut : &rw[(size_t) key.ret]->keyOut;
        juce::FloatVectorOperations::copy (l, from->getReadPointer (0), n); juce::FloatVectorOperations::copy (r, from->getReadPointer (1), n);
        tap.delay (l, r, n, key.delay);
        tap.filter.process (l, r, n, tap.lo.load (std::memory_order_relaxed), tap.hi.load (std::memory_order_relaxed), sr);
        return true;
    }
    // a plug-in with whatever feeds its side-chain, and LISTEN
    void runKeyed (const RenderPlan& P, const PlugRef& ins, juce::AudioBuffer<float>& buf, const juce::MidiBuffer& midi, juce::AudioBuffer<float>& scratch, juce::MidiBuffer& scratchMidi, juce::AudioBuffer<float>& work, int n, const BlockInfo* blk)
    {
        const bool keyed = ins.slot != nullptr && makeKey (P, ins.key, ins.slot->side, work, n, true, blk);
        runInsert (ins, buf, midi, scratch, scratchMidi, n, fadeLength(), keyed ? work.getReadPointer (0) : nullptr, keyed ? work.getReadPointer (1) : nullptr);
        if (ins.slot != nullptr) ins.slot->side.hear (buf, keyed ? work.getReadPointer (0) : nullptr, keyed ? work.getReadPointer (1) : nullptr, n, sr);
    }
    static const BlockInfo* blockOf (const RenderPlan& P) { return P.tracks.empty() || P.tracks[0].ts == nullptr ? nullptr : P.tracks[0].ts->blk; }

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
                    {
                        const bool keyed = makeKey (P, tp.compFrom, ts.compSide, w.keyWork, n, false, ts.blk);
                        const float* kl = keyed ? w.keyWork.getReadPointer (0) : nullptr; const float* kr = keyed ? w.keyWork.getReadPointer (1) : nullptr;
                        const float gr = ts.compStage.process (buf, n, ts.compOn.load(), ts.compThr.load(), ts.compRatio.load(), ts.compRel.load(), ts.compFast.load(), ts.compPeak.load(), ts.compMakeup.load(), sr, -1.0f, kl, kr, (float) ts.compBelow.load());
                        if (gr > ts.compGr.load (std::memory_order_relaxed)) ts.compGr.store (gr, std::memory_order_relaxed);
                        ts.compSide.hear (buf, kl, kr, n, sr);
                    }
                    {
                        const bool keyed = makeKey (P, tp.gateFrom, ts.gateSide, w.keyWork, n, false, ts.blk);
                        const float* kl = keyed ? w.keyWork.getReadPointer (0) : nullptr; const float* kr = keyed ? w.keyWork.getReadPointer (1) : nullptr;
                        const float gg = ts.gateStage.process (buf, n, ts.gateOn.load(), ts.gateExp.load(), ts.gateFast.load(), ts.gateThr.load(), ts.gateRange.load(), ts.gateRel.load(), ts.gateHold.load(), sr, kl, kr);
                        if (gg > ts.gateGr.load (std::memory_order_relaxed)) ts.gateGr.store (gg, std::memory_order_relaxed);
                        ts.gateSide.hear (buf, kl, kr, n, sr);
                    }
                    if (ts.blk != nullptr && ts.blk->playing && ts.blk->numSegs > 0) { const auto* eas = ts.autoSnap.load(); if (eas != nullptr) for (const auto& lane : eas->lanes) if (lane.kind == kEqLaneKind && lane.on && ! lane.pts.empty()) applyEqLane (ts.eq, ts.eqHpOn, ts.eqHpHz, ts.eqLpOn, ts.eqLpHz, lane.param, autoValueAt (lane, ts.blk->segs[0].startBeat)); }      // EQ sweeps: the EQ lanes, block by block
                    ts.eqStage.process (buf, n, ts.eqHpOn.load(), ts.eqHpHz.load(), ts.eqLpOn.load(), ts.eqLpHz.load(), sr);
                    ts.paraEq.process (buf, n, ts.eq, sr); ts.eq.feed (buf, n);
                }
                if (k < tp.inserts.size()) { applyPattern (tp.inserts[k], ts.patWant.load (std::memory_order_relaxed), ts.patOrigin.load (std::memory_order_relaxed)); runKeyed (P, tp.inserts[k], buf, w.midi, w.scratch, w.scratchMidi, w.keyWork, n, ts.blk); }
            }
        }
        w.seam.apply (buf, n, SeamFade::signature (tp.inserts), fadeLength());
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
        if (! rp.active) { if (rp.isKey) w.keyOut.clear(); return; }
        auto buf = view (w.buf, 2, n);
        buf.clear();
        for (int ti : rp.fromTracks)
        {
            const auto& tp = P.tracks[(size_t) ti]; auto& src = tw[(size_t) tp.index]->sendOut[r];
            if (src == nullptr) continue;
            juce::FloatVectorOperations::add (buf.getWritePointer (0), src->getReadPointer (0), n);
            juce::FloatVectorOperations::add (buf.getWritePointer (1), src->getReadPointer (1), n);
        }
        // (a return keyed from a return reads the block before, which that return may be writing now: the sound is at most one block of it, never anything outside the buffers)
        for (const auto& ins : rp.chain) runKeyed (P, ins, buf, w.emptyMidi, w.scratch, w.scratchMidi, w.keyWork, n, blockOf (P));
        w.seam.apply (buf, n, SeamFade::signature (rp.chain), fadeLength());
        if (rp.isKey) for (int c = 0; c < 2; ++c) juce::FloatVectorOperations::copy (w.keyOut.getWritePointer (c), buf.getReadPointer (c), n);      // the return as a key: after its effects, before its fader
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
        for (const auto& ins : P.masterChain) runKeyed (P, ins, m, none, mscratch, mscratchMidi, masterWork, n, blockOf (P));
        masterSeam.apply (m, n, SeamFade::signature (P.masterChain), fadeLength());
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
    SeamFade masterSeam;
    int fadeLength() const noexcept { return juce::jmax (16, (int) (0.008 * sr)); }      // every switch fades over 8 ms
    Snapshot<RenderPlan> plans { std::make_unique<RenderPlan>() };
    std::array<std::unique_ptr<TrackWork>, kTracks> tw;
    std::array<std::unique_ptr<ReturnWork>, kReturns> rw;
    juce::AudioBuffer<float> mix, mscratch, masterKeyBuf, masterWork; juce::MidiBuffer mscratchMidi; StereoDelay dryMix, masterKeyLine;
    WorkerPool pool;
};

} // namespace archie
