// Archie Studio host (desktop) -- the one master transport.
// Everything that keeps the program, the plugins, the web pages and the outside world playing together hangs off this clock:
//   * play / stop / pause / record / loop / locate, tempo, time signature, PPQN resolution
//   * outputs: MIDI clock (24 per quarter note) with start/continue/stop/song position, MIDI Time Code (quarter frames and full frame),
//     MIDI Machine Control commands
//   * inputs: MIDI clock follow (tempo, start/continue/stop, song position), MTC chase, MMC commands
// The audio thread calls advance() once per block. Every other thread only sets atomics (commands) or reads atomics (display).
#pragma once
#include <juce_audio_basics/juce_audio_basics.h>
#include <atomic>
#include <cmath>
#include "TempoMap.h"

namespace archie
{

enum class SyncMode { Internal = 0, MidiClockIn = 1, MtcIn = 2 };
enum class Fps { F24 = 0, F25 = 1, F2997Drop = 2, F30 = 3 };

// A stretch of one audio block with ONE continuous run of the timeline (two when the loop wraps inside the block). It knows its own map, so any consumer can ask
// "which sample is beat b" and "which beat is sample s" and get the right answer even if the tempo changes inside the block.
struct Segment
{
    int startSample = 0; int numSamples = 0; double startBeat = 0.0;
    double endBeat = 0.0, startSec = 0.0, sr = 48000.0; const TempoMap* map = nullptr;
    bool cut = false;      // this stretch starts after a HOT CUE jump: notes still sounding from before must be ended here
    double sampleOfBeat (double beat) const { return map != nullptr ? (double) startSample + (map->secondsAt (beat) - startSec) * sr : (double) startSample; }
    double beatOfSample (int sample) const { return map != nullptr ? map->beatAt (startSec + (double) (sample - startSample) / sr) : startBeat; }
};

struct Timecode { int h = 0, m = 0, s = 0, f = 0; };

inline double fpsNominal (Fps f) { switch (f) { case Fps::F24: return 24.0; case Fps::F25: return 25.0; case Fps::F2997Drop: return 30000.0 / 1001.0; case Fps::F30: return 30.0; } return 30.0; }
inline int fpsFramesPerSecondInt (Fps f) { switch (f) { case Fps::F24: return 24; case Fps::F25: return 25; default: return 30; } }
inline int fpsRateCode (Fps f) { switch (f) { case Fps::F24: return 0; case Fps::F25: return 1; case Fps::F2997Drop: return 2; case Fps::F30: return 3; } return 3; }

// Seconds -> hh:mm:ss:ff, including SMPTE 29.97 drop-frame.
inline Timecode secondsToTimecode (double seconds, Fps fps)
{
    if (seconds < 0) seconds = 0;
    Timecode t;
    if (fps == Fps::F2997Drop)
    {
        long long frames = (long long) std::floor (seconds * 30000.0 / 1001.0 + 1.0e-9);
        const long long d = frames / 17982, m = frames % 17982;
        frames += 18 * d + (m > 1 ? 2 * ((m - 2) / 1798) : 0);
        t.f = (int) (frames % 30); t.s = (int) ((frames / 30) % 60); t.m = (int) ((frames / 1800) % 60); t.h = (int) ((frames / 108000) % 24);
        return t;
    }
    const int n = fpsFramesPerSecondInt (fps);
    const long long frames = (long long) std::floor (seconds * n + 1.0e-9);
    t.f = (int) (frames % n); t.s = (int) ((frames / n) % 60); t.m = (int) ((frames / ((long long) n * 60)) % 60); t.h = (int) ((frames / ((long long) n * 3600)) % 24);
    return t;
}

inline double timecodeToSeconds (const Timecode& t, Fps fps)
{
    if (fps == Fps::F2997Drop)
    {
        const long long totalMinutes = t.h * 60 + t.m;
        const long long frames = ((long long) t.h * 3600 + (long long) t.m * 60 + t.s) * 30 + t.f - 2 * (totalMinutes - totalMinutes / 10);
        return (double) frames * 1001.0 / 30000.0;
    }
    const int n = fpsFramesPerSecondInt (fps);
    return (double) t.h * 3600 + t.m * 60 + t.s + (double) t.f / n;
}

inline juce::String timecodeString (const Timecode& t, Fps fps)
{
    return juce::String::formatted ("%02d:%02d:%02d%c%02d", t.h, t.m, t.s, fps == Fps::F2997Drop ? ';' : ':', t.f);
}

class Transport
{
public:
    // ---- commands (any thread) ----
    // Every start and every stop is COUNTED, so the audio thread notices a restart even when stop and play both happen inside one audio block (the playing flag alone would look unchanged).
    void setPlaying (bool on) { if (playing.exchange (on) != on) (on ? startEdges : stopEdges).fetch_add (1); }
    void play()               { recordPaused.store (false); pausedFlag.store (false); setPlaying (true); }
    void stop()               { if (! playing.load()) { locate (0.0); } else { pausedFlag.store (false); setPlaying (false); } }
    void pause()              { if (playing.load()) { pausedFlag.store (true); setPlaying (false); } }
    void togglePlay()         { if (playing.load()) stop(); else play(); }
    void setRecording (bool r){ recording.store (r); }
    void locate (double beats){ locateReq.store (juce::jmax (0.0, beats)); }
    // The tempo and time signature at the START of the song (the first map entry); further tempo changes are map entries of their own.
    void setBpm (double b)    { auto e = tempoEntries(); e[0].bpm = juce::jlimit (20.0, 400.0, b); setTempoEntries (e); }
    void setTimeSignature (int n, int d) { auto e = tempoEntries(); e[0].tsNum = n; e[0].tsDen = d; setTempoEntries (e); }
    void addTempoChange (double beat, double bpmValue) { auto e = tempoEntries(); TempoEntry x = e[0]; x.beat = juce::jmax (0.0, beat); x.bpm = bpmValue; e.push_back (x); setTempoEntries (e); }
    void removeTempoChange (int index) { auto e = tempoEntries(); if (index > 0 && index < (int) e.size()) { e.erase (e.begin() + index); setTempoEntries (e); } }
    std::vector<TempoEntry> tempoEntries() const { return tempoSnap.load()->entries(); }
    const TempoMap* tempoNow() const { return tempoSnap.load(); }
    void setTempoEntries (std::vector<TempoEntry> entries)
    {
        const juce::SpinLock::ScopedLockType lock (tempoLock);
        auto map = std::make_unique<TempoMap> (std::move (entries));
        const auto& first = map->entries()[0];
        bpm.store (first.bpm); tsNum.store (first.tsNum); tsDen.store (first.tsDen);
        tempoSnap.publish (std::move (map));
    }
    void setLoop (bool on, double s, double e)
    {
        if (e > s + 0.01) { loopStart.store (s); loopEnd.store (e); if (! loopPending.load()) { auto& c = cues[(size_t) activeCue.load()]; c.start.store (s); c.end.store (e); c.set.store (true); playingCue.store (activeCue.load()); } }
        looping.store (on && e > s + 0.01);
    }
    // HOT CUES: three stored loops. Cue 1 is always there (it is the loop on the ruler until another cue is pressed); cues 2 and 3 are switched on when wanted. The loop that plays is the ACTIVE cue's.
    // Pressing a cue makes it the active one and jumps to its start: at once, or on the ticked point (see requestLoop). It keeps repeating until another is pressed. Safe to call from any thread, including the MIDI thread.
    static constexpr int kCues = 3;
    struct Cue { std::atomic<bool> set { false }; std::atomic<double> start { 0.0 }, end { 16.0 }; };
    std::array<Cue, kCues> cues; std::atomic<int> activeCue { 0 }, playingCue { 0 };      // chosen, and heard (they differ while a jump waits)
    bool triggerCue (int i)
    {
        if (i < 0 || i >= kCues) return false; if (! cues[(size_t) i].set.load()) { if (i != 0) return false; cues[0].start.store (loopStart.load()); cues[0].end.store (loopEnd.load()); cues[0].set.store (true); }
        const double s = cues[(size_t) i].start.load(), e = cues[(size_t) i].end.load();
        if (playing.load() && ! countingIn.load()) { pendStart.store (s); pendEnd.store (e); pendAt.store (loopJump.load() == 0 ? -1.0e9 : cueTrigger (posBeats.load())); loopPending.store (true); activeCue.store (i); }
        else { loopPending.store (false); activeCue.store (i); setLoop (true, s, e); if (! countingIn.load()) locate (s); }
        return true;
    }
    // give a cue its loop (from the ruler or a pad); the active one moves the playing loop the way a drag does
    bool setCue (int i, double s, double e)
    {
        if (i < 0 || i >= kCues || e <= s + 0.01 || s < 0.0) return false;
        cues[(size_t) i].start.store (s); cues[(size_t) i].end.store (e); cues[(size_t) i].set.store (true);
        if (i == activeCue.load()) { if (loopPending.load()) { pendStart.store (s); pendEnd.store (e); } else requestLoop (looping.load(), s, e); }
        return true;
    }
    // an empty pad pressed: a loop as long as the playing one, starting at the bar the song is in
    bool cueHere (int i)
    {
        if (i < 0 || i >= kCues) return false; const double pos = juce::jmax (0.0, posBeats.load()), bar = juce::jmax (0.25, tempoSnap.load()->barLengthBeatsAt (pos)), len = loopEnd.load() - loopStart.load() > 0.01 ? loopEnd.load() - loopStart.load() : bar * 4.0, s = std::floor (pos / bar + 1.0e-6) * bar;
        return setCue (i, s, s + len);
    }
    bool removeCue (int i)
    {
        if (i <= 0 || i >= kCues) return false; cues[(size_t) i].set.store (false);
        if (activeCue.load() == i) { loopPending.store (false); activeCue.store (0); if (cues[0].set.load()) setLoop (looping.load(), cues[0].start.load(), cues[0].end.load()); }
        return true;
    }
    // HOT CUE (loop jump): with a mode other than 0, moving the loop while the song plays does not take effect at once. The song plays on to the next chosen point and then jumps to the start of the new loop:
    // 1 the next 16th note, 2 the next beat, 3 the end of the bar, 4 the end of the loop that is playing. Never later than the end of the playing loop. Switching the loop on or off, or moving it while stopped, is immediate.
    void requestLoop (bool on, double s, double e)
    {
        if (! on && loopPending.load()) { loopPending.store (false); setLoop (false, pendStart.load(), pendEnd.load()); return; }      // switched off while a jump waits: off, at where it was going
        const bool pend = loopPending.load(); const double curS = pend ? pendStart.load() : loopStart.load(), curE = pend ? pendEnd.load() : loopEnd.load();
        const bool moved = std::abs (s - curS) > 1.0e-6 || std::abs (e - curE) > 1.0e-6;
        if (loopJump.load() == 0 || ! playing.load() || countingIn.load() || ! on || e <= s + 0.01 || ! moved) { if (moved || ! on) loopPending.store (false); if (pend && ! moved && on) return; setLoop (on, s, e); return; }
        pendStart.store (s); pendEnd.store (e); pendAt.store (cueTrigger (posBeats.load())); loopPending.store (true);
        { auto& c = cues[(size_t) activeCue.load()]; c.start.store (s); c.end.store (e); c.set.store (true); }
    }
    double cueTrigger (double pos) const
    {
        const double barLen = juce::jmax (0.25, tempoSnap.load()->barLengthBeatsAt (pos)), bar = (std::floor (pos / barLen + 1.0e-6) + 1.0) * barLen, le = loopEnd.load();
        const bool inLoop = looping.load() && pos < le - 1.0e-6; const int m = loopJump.load();
        double t = m == 1 ? (std::floor (pos * 4.0 + 1.0e-6) + 1.0) / 4.0 : m == 2 ? std::floor (pos + 1.0e-6) + 1.0 : m == 3 ? bar : (inLoop ? le : bar);
        return inLoop ? juce::jmin (t, le) : t;
    }
    std::atomic<int> loopJump { 0 }; std::atomic<bool> loopPending { false }; std::atomic<double> pendStart { 0.0 }, pendEnd { 0.0 }, pendAt { 0.0 };

    // ---- settings (any thread) ----
    std::atomic<bool> playing { false }, recording { false }, looping { false }, pausedFlag { false }, recordPaused { false };
    std::atomic<double> bpm { 120.0 }, loopStart { 0.0 }, loopEnd { 16.0 }, tcOffsetSeconds { 0.0 };
    std::atomic<int> tsNum { 4 }, tsDen { 4 }, ppqn { 480 }, syncMode { 0 }, fps { 3 }, mmcDeviceId { 0x7F };
    std::atomic<bool> clockOut { false }, mtcOut { false }, mmcOut { false }, mmcIn { false };
    // count-in: when recording starts from a stop, the transport first runs this many bars BEFORE the start position (the position is negative while it does), with the clicks on
    std::atomic<int> countInBars { 0 };
    std::atomic<double> swing { 0.0 }, swingGrid { 0.5 };      // groove: swing 0..1 (1 = a full triplet feel), grid 0.5 = eighths, 0.25 = sixteenths
    std::atomic<bool> countingIn { false };
    std::atomic<double> countInEndBeat { 0.0 };

    // ---- read-only display state (written by the audio thread) ----
    std::atomic<double> posBeats { 0.0 };
    std::atomic<long long> samplesRun { 0 };

    juce::String timecodeNow() const
    {
        const auto f = (Fps) fps.load();
        return timecodeString (secondsToTimecode (tcOffsetSeconds.load() + tempoSnap.load()->secondsAt (posBeats.load()), f), f);
    }

    // ---- inputs from MIDI (MIDI thread) ----
    void handleIncomingSync (const juce::MidiMessage& m, double timeMs)
    {
        const int mode = syncMode.load();
        const auto* d = m.getRawData();
        const int size = m.getRawDataSize();
        if (m.isMidiClock())
        {
            if (mode != (int) SyncMode::MidiClockIn) return;
            if (lastPulseMs > 0.0)
            {
                const double dt = timeMs - lastPulseMs;
                if (dt > 1.0 && dt < 200.0) { pulseAvgMs = pulseAvgMs <= 0.0 ? dt : pulseAvgMs * 0.9 + dt * 0.1; setBpm (60000.0 / (pulseAvgMs * 24.0)); }
            }
            lastPulseMs = timeMs;
        }
        else if (m.isMidiStart())    { if (mode == (int) SyncMode::MidiClockIn) { locateReq.store (0.0); setPlaying (true); } }
        else if (m.isMidiContinue()) { if (mode == (int) SyncMode::MidiClockIn) setPlaying (true); }
        else if (m.isMidiStop())     { if (mode == (int) SyncMode::MidiClockIn) { setPlaying (false); lastPulseMs = 0.0; pulseAvgMs = 0.0; } }
        else if (m.isSongPositionPointer()) { if (mode == (int) SyncMode::MidiClockIn) locateReq.store (m.getSongPositionPointerMidiBeat() / 4.0); }
        else if (m.isQuarterFrame())
        {
            if (mode != (int) SyncMode::MtcIn) return;
            const int piece = m.getQuarterFrameSequenceNumber(), v = m.getQuarterFrameValue();
            mtcPieces[piece & 7] = v;
            lastQfMs.store (timeMs);
            if ((piece & 7) == 7)
            {
                Timecode t;
                t.f = mtcPieces[0] | (mtcPieces[1] << 4);
                t.s = mtcPieces[2] | (mtcPieces[3] << 4);
                t.m = mtcPieces[4] | (mtcPieces[5] << 4);
                const int hi = mtcPieces[6] | ((mtcPieces[7] & 1) << 4);
                t.h = hi;
                const Fps f = (Fps) (((mtcPieces[7] >> 1) & 3) == 0 ? 0 : ((mtcPieces[7] >> 1) & 3) == 1 ? 1 : ((mtcPieces[7] >> 1) & 3) == 2 ? 2 : 3);
                // the eight pieces describe the frame two frames ago
                double sec = timecodeToSeconds (t, f) + 2.0 / fpsNominal (f);
                chaseTo (sec);
            }
        }
        else if (m.isSysEx() && size >= 6)
        {
            // MMC: F0 7F <device> 06 <cmd> ... F7
            if (d[0] == 0xF0 && d[1] == 0x7F && d[3] == 0x06 && mmcIn.load())
            {
                const int dev = d[2];
                if (dev == 0x7F || dev == mmcDeviceId.load())
                {
                    switch (d[4])
                    {
                        case 0x01: stop(); break;
                        case 0x02: case 0x03: play(); break;
                        case 0x04: locate (posBeats.load() + tsNum.load()); break;
                        case 0x05: locate (juce::jmax (0.0, posBeats.load() - tsNum.load())); break;
                        case 0x06: setRecording (true); play(); break;
                        case 0x07: setRecording (false); break;
                        case 0x09: pause(); break;
                        case 0x44: if (size >= 12 && d[5] >= 6 && d[6] == 0x01) { Timecode t { d[7] & 0x1F, d[8], d[9], d[10] }; locate (secondsToBeats (timecodeToSeconds (t, (Fps) fps.load()) - tcOffsetSeconds.load())); } break;
                        default: break;
                    }
                }
            }
            // MTC full frame: F0 7F <device> 01 01 hr mn sc fr F7
            else if (d[0] == 0xF0 && d[1] == 0x7F && d[3] == 0x01 && d[4] == 0x01 && size >= 10 && mode == (int) SyncMode::MtcIn)
            {
                Timecode t { d[5] & 0x1F, d[6], d[7], d[8] };
                chaseTo (timecodeToSeconds (t, (Fps) fps.load()));
            }
        }
    }

    double secondsToBeats (double sec) const { return tempoSnap.load()->beatAt (juce::jmax (0.0, sec)); }

    // ---- audio thread ----
    // Fills up to two segments for this block (two when the loop wraps inside it) and appends the MIDI that goes to the outside world.
    void advance (int numSamples, double sampleRate, juce::MidiBuffer& outMidi, Segment (&segs)[2], int& numSegs, bool& startedNow, bool& stoppedNow, bool& jumped)
    {
        numSegs = 0; startedNow = stoppedNow = jumped = false;
        double pos = posBeats.load();
        const double loc = locateReq.exchange (-1.0);
        const bool located = loc >= 0.0;
        if (located) { pos = loc; jumped = true; }
        const bool play = playing.load();

        // external MTC chase: if the quarter frames stop arriving, stop
        if (syncMode.load() == (int) SyncMode::MtcIn && play && juce::Time::getMillisecondCounterHiRes() - lastQfMs.load() > 250.0) { setPlaying (false); }

        const bool wasPlaying = wasPlayingAudio;
        const int se = startEdges.load(), te = stopEdges.load();
        const bool sawStart = se != lastStartEdges, sawStop = te != lastStopEdges;
        lastStartEdges = se; lastStopEdges = te;
        if (wasPlaying && (! play || sawStop)) stoppedNow = true;           // a stop (even if a start already followed it inside this block)
        if (play && (! wasPlaying || sawStart)) startedNow = true;
        wasPlayingAudio = play;
        if (located || stoppedNow) { if (countingIn.load() && stoppedNow && ! located) pos = countInEndBeat.load(); countingIn.store (false); if (loopPending.load()) takePendingLoop(); cutNext = false; }
        if (startedNow && recording.load() && countInBars.load() > 0 && syncMode.load() == (int) SyncMode::Internal)
        {
            const double len = (double) countInBars.load() * tempoSnap.load()->barLengthBeatsAt (pos);
            countInEndBeat.store (pos); pos -= len; countingIn.store (true);
        }

        const Fps f = (Fps) fps.load();
        const TempoMap* map = tempoSnap.load();
        const double sr = sampleRate;

        // ---- transitions out to the world ----
        if (stoppedNow)
        {
            if (clockOut.load()) outMidi.addEvent (juce::MidiMessage::midiStop(), 0);
            if (mmcOut.load()) addMmc (outMidi, 0, pausedFlag.load() ? 0x09 : 0x01);
        }
        if (startedNow)
        {
            if (clockOut.load())
            {
                if (pos > 0.0001) { addSongPosition (outMidi, 0, pos); outMidi.addEvent (juce::MidiMessage::midiContinue(), 0); }
                else outMidi.addEvent (juce::MidiMessage::midiStart(), 0);
            }
            if (mmcOut.load()) addMmc (outMidi, 0, recording.load() ? 0x06 : 0x02);
            if (mtcOut.load()) addFullFrame (outMidi, 0, pos, f);
            qfIndex = 0; qfCountdown = 0.0;
            lastClockPulse = (long long) std::ceil (pos * 24.0 - 1.0e-9) - 1;
        }
        if (jumped)
        {
            if (! play)
            {
                if (clockOut.load()) addSongPosition (outMidi, 0, pos);
                if (mtcOut.load()) addFullFrame (outMidi, 0, pos, f);
            }
            else if (mtcOut.load()) { addFullFrame (outMidi, 0, pos, f); qfIndex = 0; qfCountdown = 0.0; }
            if (mmcOut.load() && located) addMmcLocate (outMidi, 0, pos, f);
            lastClockPulse = (long long) std::ceil (pos * 24.0 - 1.0e-9) - 1;
        }
        const bool recNow = recording.load();
        if (recNow != lastRecording) { if (mmcOut.load()) addMmc (outMidi, 0, recNow ? 0x06 : 0x07); lastRecording = recNow; }

        if (! play)
        {
            posBeats.store (pos);
            Segment s0; s0.startSample = 0; s0.numSamples = numSamples; s0.startBeat = pos; s0.endBeat = pos; s0.map = map; s0.sr = sr; s0.startSec = map->secondsAt (pos);
            segs[0] = s0;      // not advancing: a zero-length playhead
            numSegs = 1;
            return;
        }

        // ---- advance through the tempo map, splitting at the loop end ----
        bool loop = looping.load();
        double ls = loopStart.load(), le = loopEnd.load();
        int done = 0;
        double p = pos;
        while (done < numSamples && numSegs < 2)
        {
            if (loopPending.load() && p >= pendAt.load() - 1.0e-9) { takePendingLoop(); loop = looping.load(); ls = loopStart.load(); le = loopEnd.load(); p = ls; cutNext = true; lastClockPulse = (long long) std::ceil (p * 24.0 - 1.0e-9) - 1; }      // the cue point has been reached
            int len = numSamples - done;
            bool wraps = false, cue = false;
            const double pSec = map->secondsAt (p);
            if (loopPending.load())
            {
                const int toCue = (int) std::ceil ((map->secondsAt (pendAt.load()) - pSec) * sr - 1.0e-9);
                if (toCue <= len) { len = juce::jmax (1, toCue); cue = true; }
            }
            if (loop && p < le)
            {
                const int toEnd = (int) std::ceil ((map->secondsAt (le) - pSec) * sr - 1.0e-9);
                if (toEnd <= len) { len = juce::jmax (1, toEnd); wraps = true; }
            }
            Segment sg; sg.startSample = done; sg.numSamples = len; sg.startBeat = p; sg.map = map; sg.sr = sr; sg.startSec = pSec;
            sg.endBeat = map->beatAt (pSec + (double) len / sr); sg.cut = cutNext; cutNext = false;
            segs[numSegs++] = sg;
            emitRealTimeEvents (outMidi, sg, f);
            p = sg.endBeat;
            done += len;
            if (cue && p >= pendAt.load() - 1.0e-9) { takePendingLoop(); loop = looping.load(); ls = loopStart.load(); le = loopEnd.load(); p = ls; cutNext = true; wraps = false; if (mtcOut.load()) { addFullFrame (outMidi, done, p, f); qfIndex = 0; qfCountdown = 0.0; } lastClockPulse = (long long) std::ceil (p * 24.0 - 1.0e-9) - 1; }
            if (wraps && p >= le - 1.0e-9) { p = ls; if (mtcOut.load()) { addFullFrame (outMidi, done, p, f); qfIndex = 0; qfCountdown = 0.0; } lastClockPulse = (long long) std::ceil (p * 24.0 - 1.0e-9) - 1; }
        }
        if (done < numSamples)
        {
            const int rest = numSamples - done;
            segs[numSegs - 1].numSamples += rest;
            p = map->beatAt (map->secondsAt (p) + (double) rest / sr);
            segs[numSegs - 1].endBeat = p;
        }
        if (countingIn.load() && p >= countInEndBeat.load() - 1.0e-9) countingIn.store (false);
        posBeats.store (p);
        samplesRun.fetch_add (numSamples);
    }

private:
    void takePendingLoop() { loopPending.store (false); setLoop (true, pendStart.load(), pendEnd.load()); }
    bool cutNext = false;
    void chaseTo (double seconds)
    {
        const double beats = secondsToBeats (seconds - tcOffsetSeconds.load());
        const double now = posBeats.load();
        if (! playing.load() || std::abs (beats - now) > 0.25 * tempoSnap.load()->bpmAt (now) / 60.0) locateReq.store (beats);
        setPlaying (true);
    }

    static void addSongPosition (juce::MidiBuffer& mb, int at, double beats)
    {
        const int sixteenths = juce::jmax (0, (int) std::floor (beats * 4.0 + 1.0e-6));
        const juce::uint8 d[3] = { 0xF2, (juce::uint8) (sixteenths & 0x7F), (juce::uint8) ((sixteenths >> 7) & 0x7F) };
        mb.addEvent (juce::MidiMessage (d, 3), at);
    }

    void addMmc (juce::MidiBuffer& mb, int at, int cmd) const
    {
        const juce::uint8 d[6] = { 0xF0, 0x7F, (juce::uint8) mmcDeviceId.load(), 0x06, (juce::uint8) cmd, 0xF7 };
        mb.addEvent (juce::MidiMessage (d, 6), at);
    }

    void addMmcLocate (juce::MidiBuffer& mb, int at, double beats, Fps f) const
    {
        const auto t = secondsToTimecode (tcOffsetSeconds.load() + tempoSnap.load()->secondsAt (beats), f);
        const juce::uint8 d[13] = { 0xF0, 0x7F, (juce::uint8) mmcDeviceId.load(), 0x06, 0x44, 0x06, 0x01, (juce::uint8) (t.h | (fpsRateCode (f) << 5)), (juce::uint8) t.m, (juce::uint8) t.s, (juce::uint8) t.f, 0x00, 0xF7 };
        mb.addEvent (juce::MidiMessage (d, 13), at);
    }

    void addFullFrame (juce::MidiBuffer& mb, int at, double beats, Fps f) const
    {
        const auto t = secondsToTimecode (tcOffsetSeconds.load() + tempoSnap.load()->secondsAt (beats), f);
        const juce::uint8 d[10] = { 0xF0, 0x7F, 0x7F, 0x01, 0x01, (juce::uint8) (t.h | (fpsRateCode (f) << 5)), (juce::uint8) t.m, (juce::uint8) t.s, (juce::uint8) t.f, 0xF7 };
        mb.addEvent (juce::MidiMessage (d, 10), at);
    }

    void emitRealTimeEvents (juce::MidiBuffer& out, const Segment& sg, Fps f)
    {
        const int segStart = sg.startSample, len = sg.numSamples;
        if (clockOut.load())
        {
            long long k = juce::jmax (lastClockPulse + 1, (long long) std::ceil (sg.startBeat * 24.0 - 1.0e-9));
            for (; (double) k / 24.0 < sg.endBeat - 1.0e-12; ++k)
            {
                const int at = juce::jlimit (segStart, segStart + len - 1, (int) std::lround (sg.sampleOfBeat ((double) k / 24.0)));
                out.addEvent (juce::MidiMessage::midiClock(), at);
                lastClockPulse = k;
            }
        }
        if (mtcOut.load())
        {
            const double qfPeriod = sg.sr / (4.0 * fpsNominal (f));
            double t = qfCountdown;                // samples until the next quarter frame, relative to the start of this segment
            while (t < (double) len)
            {
                const int sendAt = juce::jlimit (0, len - 1, (int) std::ceil (juce::jmax (0.0, t) - 1.0e-9));   // the sample the event really goes out on
                if (qfIndex == 0) qfFrame = secondsToTimecode (tcOffsetSeconds.load() + sg.startSec + (double) sendAt / sg.sr, f);
                int nib = 0;
                switch (qfIndex)
                {
                    case 0: nib = qfFrame.f & 0xF; break;        case 1: nib = (qfFrame.f >> 4) & 0x1; break;
                    case 2: nib = qfFrame.s & 0xF; break;        case 3: nib = (qfFrame.s >> 4) & 0x3; break;
                    case 4: nib = qfFrame.m & 0xF; break;        case 5: nib = (qfFrame.m >> 4) & 0x3; break;
                    case 6: nib = qfFrame.h & 0xF; break;        default: nib = ((qfFrame.h >> 4) & 0x1) | (fpsRateCode (f) << 1); break;
                }
                const juce::uint8 d[2] = { 0xF1, (juce::uint8) ((qfIndex << 4) | (nib & 0xF)) };
                out.addEvent (juce::MidiMessage (d, 2), segStart + sendAt);
                qfIndex = (qfIndex + 1) & 7;
                t += qfPeriod;
            }
            qfCountdown = t - (double) len;
        }
    }

    Snapshot<TempoMap> tempoSnap { std::make_unique<TempoMap>() };
    juce::SpinLock tempoLock;
    std::atomic<double> locateReq { -1.0 };
    std::atomic<int> startEdges { 0 }, stopEdges { 0 };
    int lastStartEdges = 0, lastStopEdges = 0;
    bool wasPlayingAudio = false, lastRecording = false;
    long long lastClockPulse = -1;
    int qfIndex = 0; double qfCountdown = 0.0; Timecode qfFrame;
    // MIDI-thread-only follow state
    double lastPulseMs = 0.0, pulseAvgMs = 0.0; int mtcPieces[8] {};
    std::atomic<double> lastQfMs { 0.0 };
};

} // namespace archie
