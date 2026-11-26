// Archie Studio host -- EXPORT: the song (or the loop) mixed down to an audio file, faster than real time.
// The mixdown runs on a SECOND, private copy of the engine: the project is saved from the live engine into it, so the person can keep playing, and the file is the song exactly as saved (every plug-in is
// loaded again with its own settings). The copy renders block by block through the same mixer the sound card hears (parallel tracks, delay compensation, folders, sends, automation), writes the wav as it goes,
// and is thrown away. Nothing here touches the live engine except reading its saved project.
#pragma once
#include "Engine.h"
#include <atomic>

namespace archie
{

// The end of the music in beats: the last clip's end (a for-ever clip counts as one pass of its notes); 0 for an empty song.
inline double songEndBeat (HostEngine& e)
{
    double end = 0.0;
    for (int ti = 0; ti < e.trackCount.load(); ++ti)
        for (const auto& in : e.trackAt (ti)->arrSnap.load()->insts)
            end = juce::jmax (end, in.start + (in.length > 0.0 ? in.length : (in.content ? in.content->lengthBeats : 0.0)));
    return end;
}

// Render [fromBeat, toBeat) plus `tailSec` seconds of ring-out through the engine (which must be prepared offline), handing each block to `sink(left, right, n)`.
// Returns the number of samples handed over. `cancel` stops it early; `progress` goes from 0 to 1.
template <typename Sink>
inline long long renderMixdown (HostEngine& e, double fromBeat, double toBeat, double tailSec, int blockSize, Sink&& sink, const std::atomic<bool>* cancel = nullptr, std::atomic<double>* progress = nullptr)
{
    const double sr = e.sampleRate; const auto* tm = e.transport.tempoNow();
    const double sec = juce::jmax (0.0, tm->secondsAt (toBeat) - tm->secondsAt (fromBeat)) + juce::jmax (0.0, tailSec);
    const long long total = (long long) std::ceil (sec * sr);
    e.transport.setLoop (false, e.transport.loopStart.load(), e.transport.loopEnd.load()); e.transport.countInBars.store (0); e.transport.setRecording (false);
    e.transport.locate (fromBeat);
    { std::vector<float> a ((size_t) blockSize), b ((size_t) blockSize); float* o[2] = { a.data(), b.data() }; e.processOffline (nullptr, 0, o, 2, blockSize); }   // lets the locate land (nothing is kept)
    e.transport.locate (fromBeat); e.transport.play();
    std::vector<float> l ((size_t) blockSize), r ((size_t) blockSize); float* out[2] = { l.data(), r.data() };
    long long done = 0; int tick = 0;
    while (done < total)
    {
        if (cancel != nullptr && cancel->load()) break;
        const int n = (int) juce::jmin ((long long) blockSize, total - done);
        if (++tick >= 4) { tick = 0; e.applyParamAutomation(); }             // plug-in parameter lanes (the live program does this from its 30 Hz tick)
        e.processOffline (nullptr, 0, out, 2, blockSize);
        sink (l.data(), r.data(), n); done += n;
        if (progress != nullptr) progress->store (total > 0 ? (double) done / (double) total : 1.0);
    }
    e.transport.stop();
    return done;
}

// The whole job for the program: save the live project into a private engine, wait for its warped clips, render, write the file, report. Created and started on the message thread.
class MixdownJob : private juce::Timer, private juce::Thread
{
public:
    // fromBeat/toBeat < 0 = the whole song; `bits` 16, 24 or 32 (32 = floating point)
    MixdownJob (HostEngine& live, juce::File outFile, int bits, double fromBeat, double toBeat, double tailSec, std::function<void (const juce::var&)> report, int onlyTrackIndex = -1, bool withMasterTune = true)
        : juce::Thread ("Archie export"), file (std::move (outFile)), bitDepth (bits == 16 || bits == 32 ? bits : 24), from (fromBeat), to (toBeat), tail (tailSec), emit (std::move (report)), rate (live.sampleRate), only (onlyTrackIndex)
    {
        project = live.saveProject();
        copy = std::make_unique<HostEngine>();
        piece = live.renderPieceSize();
        copy->prepareOffline (rate, piece, 3);
        copy->masterTuneCents.store (withMasterTune ? live.masterTuneCents.load() : 0);      // a bounce that returns to the song as a clip is tuned there, so not in the file as well
        copy->loadProject (project, [this] (juce::String err) { loadError = err; loaded = true; });
        startTimer (100);
    }
    ~MixdownJob() override { stopTimer(); cancelFlag.store (true); stopThread (5000); }
    void cancel() { cancelFlag.store (true); }
    bool isFinished() const { return finished.load(); }

private:
    void timerCallback() override
    {
        if (! loaded) return;
        if (! muteApplied) { muteApplied = true; if (only >= 0) for (int i = 0; i < copy->trackCount.load(); ++i) if (auto* tt = copy->trackAt (i)) { tt->mute.store (i != only && ! copy->isBusOf (i, only) ? true : false); tt->solo.store (false); } }   // a bounce: only this track is heard (with its sends, folder and the master)
        copy->applyWarpResults();
        if (copy->warpBusy()) { emit (status ("preparing the stretched sounds...", 0.02)); return; }
        stopTimer();
        const double songEnd = songEndBeat (*copy);
        a = from >= 0.0 ? from : 0.0; b = to >= 0.0 ? to : songEnd;
        if (b <= a + 1.0e-6) { finish ("There is nothing to export: the song is empty.", {}); return; }
        startThread();
    }
    void run() override
    {
        auto out = file.createOutputStream();
        if (out == nullptr) { finish ("Could not write " + file.getFullPathName(), {}); return; }
        juce::WavAudioFormat wav; std::unique_ptr<juce::AudioFormatWriter> w (wav.createWriterFor (out.get(), rate, 2, bitDepth, {}, 0));
        if (w == nullptr) { finish ("Could not start the audio file.", {}); return; }
        out.release();
        float peak = 0.0f; std::atomic<double> prog { 0.0 }; double lastReport = -1.0;
        const long long n = renderMixdown (*copy, a, b, tail, piece, [&] (const float* l, const float* r, int count)
        {
            const float* ch[2] = { l, r }; w->writeFromFloatArrays (ch, 2, count);
            for (int i = 0; i < count; ++i) peak = juce::jmax (peak, std::abs (l[i]), std::abs (r[i]));
            if (prog.load() - lastReport > 0.02) { lastReport = prog.load(); auto st = status ("rendering...", prog.load()); juce::MessageManager::callAsync ([this, st] { if (! finished.load()) emit (st); }); }
        }, &cancelFlag, &prog);
        w.reset();
        if (cancelFlag.load()) { file.deleteFile(); finish ("Export cancelled.", {}); return; }
        auto* o = new juce::DynamicObject(); o->setProperty ("file", file.getFullPathName()); o->setProperty ("seconds", (double) n / rate); o->setProperty ("peakDb", peak > 0.0f ? 20.0 * std::log10 (peak) : -200.0); o->setProperty ("clipped", peak > 1.0f);
        finish ({}, juce::var (o));
    }
    juce::var status (const juce::String& text, double progress) const { auto* o = new juce::DynamicObject(); o->setProperty ("state", "working"); o->setProperty ("text", text); o->setProperty ("progress", progress); return juce::var (o); }
    void finish (const juce::String& error, const juce::var& result)
    {
        juce::MessageManager::callAsync ([this, error, result]
        {
            auto* o = new juce::DynamicObject(); o->setProperty ("state", error.isEmpty() ? "done" : "error"); o->setProperty ("progress", 1.0); o->setProperty ("text", error);
            if (result.isObject()) { o->setProperty ("file", result.getProperty ("file", {})); o->setProperty ("seconds", result.getProperty ("seconds", 0.0)); o->setProperty ("peakDb", result.getProperty ("peakDb", -200.0)); o->setProperty ("clipped", result.getProperty ("clipped", false)); }
            finished.store (true); emit (juce::var (o));
        });
    }

    juce::File file; int bitDepth; double from, to, tail, a = 0.0, b = 0.0; std::function<void (const juce::var&)> emit; double rate; int only; bool muteApplied = false; int piece = 512;
    juce::var project; std::unique_ptr<HostEngine> copy; std::atomic<bool> cancelFlag { false }, finished { false }; bool loaded = false; juce::String loadError;
};

} // namespace archie
