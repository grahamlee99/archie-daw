// Offline checks of the host's master transport (no sound card, no window). Run: build target transport_test.
// Every check compares the transport's output with a number worked out independently here.
#include <juce_audio_basics/juce_audio_basics.h>
#include "../Source/Transport.h"
#include <cstdio>
#include <vector>

using namespace archie;
static int failures = 0, checks = 0;
#define CHECK(cond, msg) do { ++checks; if (!(cond)) { ++failures; std::printf("FAIL: %s  (%s:%d)\n", msg, __FILE__, __LINE__); } } while (0)

struct Run { std::vector<std::pair<int, juce::MidiMessage>> events; int blocks = 0; };

// run n blocks and collect the outgoing MIDI with absolute sample positions
static Run run (Transport& t, int blocks, int blockSize, double sr, int firstBlock = 0)
{
    Run r;
    for (int b = 0; b < blocks; ++b)
    {
        juce::MidiBuffer mb; Segment segs[2]; int ns = 0; bool st = false, sp = false, jm = false;
        t.advance (blockSize, sr, mb, segs, ns, st, sp, jm);
        for (const auto meta : mb) r.events.push_back ({ (firstBlock + b) * blockSize + meta.samplePosition, meta.getMessage() });
    }
    r.blocks = blocks;
    return r;
}
static int countIf (const Run& r, std::function<bool (const juce::MidiMessage&)> f) { int n = 0; for (auto& e : r.events) if (f (e.second)) ++n; return n; }

int main()
{
    const double sr = 48000.0; const int bs = 480;

    // ---- timecode maths ----
    {
        auto t = secondsToTimecode (3661.5, Fps::F25); CHECK (t.h == 1 && t.m == 1 && t.s == 1 && t.f == 12, "25 fps 01:01:01:12");
        t = secondsToTimecode (59.0, Fps::F24); CHECK (t.s == 59 && t.f == 0, "24 fps 59 s");
        // 29.97 drop-frame: frame 1800 (one minute of frames at 30) is 00:01:00;02 because frames 0 and 1 are skipped at each minute not divisible by 10
        const double sec1800 = 1800.0 * 1001.0 / 30000.0;
        t = secondsToTimecode (sec1800 + 1e-6, Fps::F2997Drop); CHECK (t.m == 1 && t.s == 0 && t.f == 2, "drop-frame minute boundary lands on ;02");
        // ten minutes of drop-frame time: 17982 real frames = exactly 00:10:00;00
        t = secondsToTimecode (17982.0 * 1001.0 / 30000.0 + 1e-6, Fps::F2997Drop); CHECK (t.m == 10 && t.s == 0 && t.f == 0, "drop-frame ten-minute boundary is exact");
        // round trip
        for (Fps f : { Fps::F24, Fps::F25, Fps::F2997Drop, Fps::F30 })
            for (double s : { 0.0, 1.0, 59.99, 60.0, 600.0, 3599.9, 3600.5 })
            {
                const auto tc = secondsToTimecode (s, f); const double back = timecodeToSeconds (tc, f);
                CHECK (std::abs (back - s) <= 1.0 / fpsNominal (f) + 1e-6 && back <= s + 1e-6, "timecode round trip within one frame");
            }
        CHECK (timecodeString (secondsToTimecode (1.0, Fps::F2997Drop), Fps::F2997Drop).contains (";"), "drop-frame uses a semicolon");
    }

    // ---- position advance, tempo, locate ----
    {
        Transport t; t.setBpm (120.0); t.play();
        run (t, 100, bs, sr);                                   // 100 x 480 = 48000 samples = 1 s = 2 beats at 120
        CHECK (std::abs (t.posBeats.load() - 2.0) < 1e-9, "1 s at 120 bpm is 2 beats");
        t.setBpm (90.0); run (t, 100, bs, sr);
        CHECK (std::abs (t.posBeats.load() - 3.5) < 1e-9, "tempo change keeps the position continuous (2 + 1.5 beats)");
        t.stop(); run (t, 1, bs, sr); const double held = t.posBeats.load(); run (t, 10, bs, sr);
        CHECK (t.posBeats.load() == held, "stopped transport does not move");
        t.stop(); run (t, 1, bs, sr);
        CHECK (t.posBeats.load() == 0.0, "stop again returns to the start");
        t.locate (8.0); run (t, 1, bs, sr); CHECK (std::abs (t.posBeats.load() - 8.0) < 1e-9, "locate moves the position");
    }

    // ---- loop wraps inside a block, no gap and no overlap ----
    {
        Transport t; t.setBpm (120.0); t.setLoop (true, 0.0, 4.0); t.play();
        double expectBeat = 0.0; bool allOk = true; int wraps = 0;
        for (int b = 0; b < 800; ++b)
        {
            juce::MidiBuffer mb; Segment segs[2]; int ns = 0; bool st, sp, jm;
            t.advance (bs, sr, mb, segs, ns, st, sp, jm);
            int total = 0;
            for (int s = 0; s < ns; ++s)
            {
                if (std::abs (segs[s].startBeat - expectBeat) > 1e-6 || segs[s].startSample != total) allOk = false;
                total += segs[s].numSamples; expectBeat += segs[s].numSamples / (sr * 60.0 / 120.0);
                if (expectBeat >= 4.0 - 1e-6) { expectBeat = 0.0; ++wraps; }
            }
            if (total != bs) allOk = false;
        }
        CHECK (allOk, "loop segments are contiguous and add up to the block");
        CHECK (wraps == 4, "8 seconds at 120 bpm in a 2 s (4 beat) loop wraps 4 times");
    }

    // ---- MIDI clock out: 24 per beat, evenly spaced, with start and stop ----
    {
        Transport t; t.setBpm (120.0); t.clockOut.store (true); t.locate (0.0); t.play();
        auto r = run (t, 200, bs, sr);                          // 2 s = 4 beats
        const int clocks = countIf (r, [] (const juce::MidiMessage& m) { return m.isMidiClock(); });
        CHECK (clocks == 96, "4 beats send 96 clock pulses");
        CHECK (countIf (r, [] (const juce::MidiMessage& m) { return m.isMidiStart(); }) == 1, "one start from the beginning");
        std::vector<int> pos; for (auto& e : r.events) if (e.second.isMidiClock()) pos.push_back (e.first);
        bool even = true; const double period = sr * 60.0 / 120.0 / 24.0;
        for (size_t i = 1; i < pos.size(); ++i) if (std::abs ((pos[i] - pos[i - 1]) - period) > 1.01) even = false;
        CHECK (even, "clock pulses are spaced 1000 samples apart (within a sample)");
        t.stop(); auto r2 = run (t, 1, bs, sr);
        CHECK (countIf (r2, [] (const juce::MidiMessage& m) { return m.isMidiStop(); }) == 1, "stop sends MIDI stop");
        // continue from the middle: song position pointer then continue
        t.locate (4.0); run (t, 1, bs, sr);
        t.play(); auto r3 = run (t, 2, bs, sr);
        CHECK (countIf (r3, [] (const juce::MidiMessage& m) { return m.isMidiContinue(); }) == 1, "play from the middle sends continue");
        bool spp = false; for (auto& e : r3.events) if (e.second.isSongPositionPointer() && e.second.getSongPositionPointerMidiBeat() == 16) spp = true;
        CHECK (spp, "continue is preceded by song position 16 sixteenths (beat 4)");
    }

    // ---- MTC out: 4 quarter frames per frame, the eight pieces carry the timecode ----
    {
        for (Fps f : { Fps::F24, Fps::F25, Fps::F2997Drop, Fps::F30 })
        {
            Transport t; t.setBpm (120.0); t.fps.store ((int) f); t.mtcOut.store (true); t.locate (0.0); t.play();
            auto r = run (t, 400, bs, sr);                      // 4 s
            const int qf = countIf (r, [] (const juce::MidiMessage& m) { return m.isQuarterFrame(); });
            const double expect = 4.0 * 4.0 * fpsNominal (f);
            CHECK (std::abs (qf - expect) <= 2.0, "quarter frame count is 4 per frame");
            // decode every complete 8-piece sequence and compare with the time it started
            int decoded = 0, bad = 0; int pieces[8] {}; int have = 0, startSample = 0;
            for (auto& e : r.events)
            {
                if (! e.second.isQuarterFrame()) continue;
                const int pn = e.second.getQuarterFrameSequenceNumber();
                if (pn == 0) { have = 1; startSample = e.first; pieces[0] = e.second.getQuarterFrameValue(); }
                else if (have == pn) { pieces[pn] = e.second.getQuarterFrameValue(); ++have; if (pn == 7) { have = 0; ++decoded;
                    Timecode tc; tc.f = pieces[0] | (pieces[1] << 4); tc.s = pieces[2] | (pieces[3] << 4); tc.m = pieces[4] | (pieces[5] << 4); tc.h = pieces[6] | ((pieces[7] & 1) << 4);
                    const int rate = (pieces[7] >> 1) & 3; if (rate != fpsRateCode (f)) ++bad;
                    const double got = timecodeToSeconds (tc, f), at = startSample / sr;
                    if (got > at + 1e-6 || at - got > 1.5 / fpsNominal (f)) { if (bad == 0) std::printf("  mtc fps %.3f: at %.5f got %.5f tc %02d:%02d:%02d:%02d\n", fpsNominal (f), at, got, tc.h, tc.m, tc.s, tc.f); ++bad; } } }
                else have = 0;
            }
            CHECK (decoded >= (int) (2 * fpsNominal (f)) - 3 && bad == 0, "every 8-piece MTC sequence (two frames long) decodes to the right time and rate");
        }
    }

    // ---- MMC out: exact bytes ----
    {
        Transport t; t.mmcOut.store (true); t.mmcDeviceId.store (0x7F); t.locate (0.0); run (t, 1, bs, sr);
        t.play(); auto r = run (t, 1, bs, sr);
        bool play = false; for (auto& e : r.events) if (e.second.isSysEx() && e.second.getRawDataSize() == 6 && e.second.getRawData()[3] == 0x06 && e.second.getRawData()[4] == 0x02) play = true;
        CHECK (play, "play sends F0 7F 7F 06 02 F7");
        t.setRecording (true); auto rr = run (t, 1, bs, sr);
        bool rec = false; for (auto& e : rr.events) if (e.second.isSysEx() && e.second.getRawData()[4] == 0x06) rec = true;
        CHECK (rec, "record sends record strobe (06)");
        t.setRecording (false); t.pause(); auto rp = run (t, 1, bs, sr);
        bool pause = false; for (auto& e : rp.events) if (e.second.isSysEx() && e.second.getRawData()[4] == 0x09) pause = true;
        CHECK (pause, "pause sends MMC pause (09)");
        t.locate (8.0); t.fps.store ((int) Fps::F25); auto rl = run (t, 1, bs, sr);       // 8 beats at 120 = 4 s = 00:00:04:00
        bool loc = false; for (auto& e : rl.events) { auto* d = e.second.getRawData(); if (e.second.isSysEx() && e.second.getRawDataSize() == 13 && d[4] == 0x44 && d[7] == (0 | (1 << 5)) && d[8] == 0 && d[9] == 4 && d[10] == 0) loc = true; }
        CHECK (loc, "locate sends GOTO with 00:00:04:00 at 25 fps and the rate bits");
    }

    // ---- MMC in ----
    {
        Transport t; t.mmcIn.store (true); t.mmcDeviceId.store (0x7F);
        auto sx = [] (std::initializer_list<juce::uint8> b) { std::vector<juce::uint8> v (b); return juce::MidiMessage::createSysExMessage (v.data() + 1, (int) v.size() - 2); };
        t.handleIncomingSync (sx ({ 0xF0, 0x7F, 0x7F, 0x06, 0x02, 0xF7 }), 0.0);
        CHECK (t.playing.load(), "MMC play starts the transport");
        t.handleIncomingSync (sx ({ 0xF0, 0x7F, 0x7F, 0x06, 0x01, 0xF7 }), 0.0);
        CHECK (! t.playing.load(), "MMC stop stops it");
        t.handleIncomingSync (sx ({ 0xF0, 0x7F, 0x7F, 0x06, 0x06, 0xF7 }), 0.0);
        CHECK (t.recording.load() && t.playing.load(), "MMC record strobe records and plays");
        t.handleIncomingSync (sx ({ 0xF0, 0x7F, 0x7F, 0x06, 0x07, 0xF7 }), 0.0);
        CHECK (! t.recording.load(), "MMC record exit stops recording");
        t.mmcIn.store (false); t.stop(); t.handleIncomingSync (sx ({ 0xF0, 0x7F, 0x7F, 0x06, 0x02, 0xF7 }), 0.0);
        CHECK (! t.playing.load(), "MMC is ignored when Obey MMC is off");
        t.mmcIn.store (true); t.mmcDeviceId.store (5); t.handleIncomingSync (sx ({ 0xF0, 0x7F, 0x09, 0x06, 0x02, 0xF7 }), 0.0);
        CHECK (! t.playing.load(), "MMC for another device id is ignored");
        t.handleIncomingSync (sx ({ 0xF0, 0x7F, 0x05, 0x06, 0x02, 0xF7 }), 0.0);
        CHECK (t.playing.load(), "MMC for this device id is obeyed");
    }

    // ---- MIDI clock in: tempo follow, start / continue / stop / song position ----
    {
        Transport t; t.syncMode.store ((int) SyncMode::MidiClockIn);
        double now = 1000.0; const double pulseMs = 60000.0 / (133.0 * 24.0);
        for (int i = 0; i < 200; ++i) { t.handleIncomingSync (juce::MidiMessage::midiClock(), now); now += pulseMs; }
        CHECK (std::abs (t.bpm.load() - 133.0) < 0.5, "tempo follows an outside clock (133 bpm)");
        t.handleIncomingSync (juce::MidiMessage::midiStart(), now);
        CHECK (t.playing.load(), "outside start starts the transport");
        t.handleIncomingSync (juce::MidiMessage::midiStop(), now); CHECK (! t.playing.load(), "outside stop stops it");
        t.handleIncomingSync (juce::MidiMessage::songPositionPointer (32), now); t.handleIncomingSync (juce::MidiMessage::midiContinue(), now);
        run (t, 1, bs, sr);
        CHECK (t.playing.load() && std::abs (t.posBeats.load() - 8.0) < 0.2, "song position 32 sixteenths then continue plays from beat 8");
        Transport u; u.handleIncomingSync (juce::MidiMessage::midiStart(), now);
        CHECK (! u.playing.load(), "outside start is ignored in internal-clock mode");
    }

    // ---- MTC in: chase ----
    {
        Transport t; t.setBpm (120.0); t.syncMode.store ((int) SyncMode::MtcIn); t.fps.store ((int) Fps::F25);
        // send eight pieces for 00:00:10:00 (+ the 2-frame latency the receiver adds)
        auto send = [&] (int h, int m, int s, int f, int rate, double ms)
        {
            const int nib[8] = { f & 15, (f >> 4) & 1, s & 15, (s >> 4) & 3, m & 15, (m >> 4) & 3, h & 15, ((h >> 4) & 1) | (rate << 1) };
            for (int i = 0; i < 8; ++i) t.handleIncomingSync (juce::MidiMessage::quarterFrame (i, nib[i]), ms + i);
        };
        send (0, 0, 10, 0, 1, juce::Time::getMillisecondCounterHiRes());
        Segment segs[2]; juce::MidiBuffer mb; int ns; bool a, b, c; t.advance (bs, sr, mb, segs, ns, a, b, c);
        CHECK (t.playing.load(), "incoming timecode starts the transport");
        CHECK (std::abs (t.posBeats.load() - (10.08 * 2.0)) < 0.1, "it locks to the incoming time (10 s + 2 frames at 120 bpm is 20.16 beats)");
    }

    // ---- count-in: recording from a stop runs the bars first, with a negative position, then lands exactly on the start ----
    {
        Transport t; t.setBpm (120.0); t.countInBars.store (2); t.setRecording (true); t.locate (0.0); run (t, 1, bs, sr); t.play();
        run (t, 1, bs, sr);
        CHECK (t.countingIn.load(), "a count-in is running");
        CHECK (t.posBeats.load() < -7.0, "two 4/4 bars before the start: the position begins at -8 beats");
        run (t, 399, bs, sr);       // 4 s = 8 beats
        CHECK (! t.countingIn.load(), "the count-in has ended after two bars");
        CHECK (std::abs (t.posBeats.load()) < 0.02, "and the position lands exactly on beat 0");
        run (t, 100, bs, sr);
        CHECK (t.posBeats.load() > 1.9, "then it runs on normally");
        Transport u; u.setBpm (120.0); u.countInBars.store (1); u.setRecording (true); u.locate (4.0); run (u, 1, bs, sr); u.play(); run (u, 10, bs, sr); u.stop(); run (u, 1, bs, sr);
        CHECK (! u.countingIn.load() && std::abs (u.posBeats.load() - 4.0) < 1e-9, "stopping during a count-in returns to where it started");
        Transport v; v.setBpm (120.0); v.countInBars.store (2); v.play(); run (v, 10, bs, sr);
        CHECK (! v.countingIn.load() && v.posBeats.load() > 0.0, "no count-in when not recording");
    }

    // ---- stop and start inside ONE audio block is still a restart (count-in, chase and notes-off all happen) ----
    {
        Transport t; t.setBpm (120.0); t.play(); run (t, 20, bs, sr);
        t.stop(); t.countInBars.store (1); t.setRecording (true); t.play();                  // all before the next block
        juce::MidiBuffer mb; Segment s[2]; int ns; bool st = false, sp = false, jm = false; t.advance (bs, sr, mb, s, ns, st, sp, jm);
        CHECK (st && sp, "a stop and a start inside one block are reported as both a stop and a start");
        CHECK (t.countingIn.load(), "so the count-in runs");
        Transport u; u.clockOut.store (true); u.play(); run (u, 5, bs, sr); u.stop(); u.play();
        juce::MidiBuffer m2; Segment s2[2]; int n2; bool a = false, b = false, c = false; u.advance (bs, sr, m2, s2, n2, a, b, c);
        int stops = 0, starts = 0, conts = 0; for (const auto meta : m2) { auto m = meta.getMessage(); if (m.isMidiStop()) ++stops; if (m.isMidiStart()) ++starts; if (m.isMidiContinue()) ++conts; }
        CHECK (stops == 1 && starts + conts == 1, "and the outside world gets a MIDI stop then a start or continue");
    }

    // ---- a loop wrap is not a jump (no panic), a locate is ----
    {
        Transport t; t.setBpm (120.0); t.setLoop (true, 0.0, 4.0); t.play();
        bool wrapJump = false; for (int b = 0; b < 300; ++b) { juce::MidiBuffer mb; Segment s[2]; int ns; bool st, sp, jm; t.advance (bs, sr, mb, s, ns, st, sp, jm); if (jm) wrapJump = true; }
        CHECK (! wrapJump, "a loop wrap does not report a jump (so notes are not cut)");
        t.locate (1.0); juce::MidiBuffer mb; Segment s[2]; int ns; bool st, sp, jm; t.advance (bs, sr, mb, s, ns, st, sp, jm);
        CHECK (jm, "a locate does report a jump");
    }

    // ---- PPQN is only a resolution: position in beats does not depend on it ----
    {
        for (int p : { 24, 96, 480, 960 }) { Transport t; t.ppqn.store (p); t.play(); run (t, 100, bs, sr); CHECK (std::abs (t.posBeats.load() - 2.0) < 1e-9, "position is the same at every PPQN"); }
    }

    // ================= the TEMPO MAP (time base) =================
    {
        TempoMap one;
        CHECK (std::abs (one.secondsAt (4.0) - 2.0) < 1e-12 && std::abs (one.beatAt (2.0) - 4.0) < 1e-12, "one tempo: 4 beats are 2 s at 120 bpm");
        std::vector<TempoEntry> es { { 0.0, 120.0, 4, 4 }, { 4.0, 60.0, 4, 4 }, { 8.0, 240.0, 3, 4 } };
        TempoMap m (es);
        CHECK (std::abs (m.secondsAt (4.0) - 2.0) < 1e-12, "120 bpm for the first 4 beats is 2 s");
        CHECK (std::abs (m.secondsAt (8.0) - 6.0) < 1e-12, "then 4 beats at 60 bpm is 4 s more");
        CHECK (std::abs (m.secondsAt (10.0) - 6.5) < 1e-12, "then 2 beats at 240 bpm is 0.5 s more");
        for (double b : { 0.0, 0.3, 3.999, 4.0, 5.5, 7.99, 8.0, 12.25, 100.0 }) CHECK (std::abs (m.beatAt (m.secondsAt (b)) - b) < 1e-9, "beats -> seconds -> beats comes back to the same beat");
        CHECK (m.bpmAt (3.9) == 120.0 && m.bpmAt (4.0) == 60.0 && m.bpmAt (9.0) == 240.0, "the tempo at a beat is the entry in force");
        CHECK (m.tsNumAt (9.0) == 3 && m.tsNumAt (1.0) == 4 && std::abs (m.barLengthBeatsAt (9.0) - 3.0) < 1e-12, "time signature changes are part of the map (3/4 from beat 8: a bar is 3 beats)");
        CHECK (std::abs (m.secondsAt (-2.0) - (-1.0)) < 1e-12, "before beat 0 (a count-in) runs at the first tempo");
        TempoMap dup (std::vector<TempoEntry> { { 0.0, 100.0, 4, 4 }, { 8.0, 90.0, 4, 4 }, { 8.0, 80.0, 4, 4 } });
        CHECK (dup.entries().size() == 2 && dup.bpmAt (9.0) == 80.0, "two entries at the same beat: the later one wins");
        TempoMap odd (std::vector<TempoEntry> { { 5.0, 9999.0, 99, 7 } });
        CHECK (odd.entries()[0].beat == 0.0 && odd.entries()[0].bpm == 400.0 && odd.entries()[0].tsNum == 32 && odd.entries()[0].tsDen == 4, "a map always starts at beat 0 and is clamped to valid values");
    }

    // ---- a tempo change in the middle of the song moves the playhead through the right number of beats ----
    {
        Transport t; t.setBpm (120.0); t.addTempoChange (4.0, 60.0); t.play();
        run (t, 300, bs, sr);                                 // 3.0 s: 2 s for the first 4 beats at 120, then 1 s at 60 = 1 beat
        CHECK (std::abs (t.posBeats.load() - 5.0) < 1e-6, "3 s with a change to 60 bpm at beat 4 lands on beat 5");
        CHECK (t.tempoEntries().size() == 2, "the change is in the map");
        t.removeTempoChange (1); t.locate (0.0); run (t, 1, bs, sr); run (t, 100, bs, sr);
        CHECK (std::abs (t.posBeats.load() - 2.02) < 1e-6, "with the change removed 1.01 s is 2.02 beats again");
    }
    // ---- the change falls INSIDE a block: still exact ----
    {
        Transport t; t.setBpm (120.0); t.addTempoChange (0.51, 60.0); t.play();       // 0.51 beats = 12240 samples = block 25, sample 240
        double expectSec = 0.0; bool segOk = true;
        TempoMap ref (std::vector<TempoEntry> { { 0.0, 120.0, 4, 4 }, { 0.51, 60.0, 4, 4 } });
        for (int b = 0; b < 100; ++b)
        {
            juce::MidiBuffer mb; Segment s[2]; int ns; bool st, sp, jm; t.advance (bs, sr, mb, s, ns, st, sp, jm);
            expectSec += (double) bs / sr;
            if (std::abs (t.posBeats.load() - ref.beatAt (expectSec)) > 1e-9) segOk = false;
            if (ns == 1 && std::abs (s[0].sampleOfBeat (s[0].endBeat) - (double) (s[0].startSample + s[0].numSamples)) > 1e-6) segOk = false;
        }
        CHECK (segOk, "block by block the position equals the analytic integral, and a segment's end beat maps to its last sample");
    }
    // ---- MIDI clock pulses follow the tempo: spacing doubles when the tempo halves ----
    {
        Transport t; t.setBpm (120.0); t.addTempoChange (4.0, 60.0); t.clockOut.store (true); t.play();
        auto r = run (t, 600, bs, sr);                         // 6 s
        std::vector<int> pos; for (auto& e : r.events) if (e.second.isMidiClock()) pos.push_back (e.first);
        bool early = true, late = true;                        // 120 bpm: 1000 samples per pulse; 60 bpm: 2000
        for (size_t i = 1; i < pos.size(); ++i)
        {
            const int gap = pos[i] - pos[i - 1];
            if (pos[i] < 90000 - 3000) { if (std::abs (gap - 1000) > 1) early = false; }
            else if (pos[i - 1] > 96000 + 3000) { if (std::abs (gap - 2000) > 1) late = false; }
        }
        CHECK (early, "before the change the clock runs at 1000 samples per pulse");
        CHECK (late, "after the change it runs at 2000 samples per pulse");
        CHECK ((int) pos.size() == 192, "6 s through the change sends exactly 192 pulses (8 beats)");
    }
    // ---- timecode follows real time, not beats ----
    {
        Transport t; t.setBpm (120.0); t.addTempoChange (4.0, 60.0); t.fps.store ((int) Fps::F30); t.play(); run (t, 300, bs, sr);
        CHECK (t.timecodeNow() == "00:00:03:00", "3 s of real time is 00:00:03:00 whatever the tempo did");
    }
    // ---- a loop across a tempo change takes the right time ----
    {
        Transport t; t.setBpm (120.0); t.addTempoChange (2.0, 60.0); t.setLoop (true, 0.0, 4.0); t.play();   // 1 s + 2 s = a 3 s loop
        run (t, 300, bs, sr);                                  // exactly one loop
        CHECK (t.posBeats.load() < 0.05 || t.posBeats.load() > 3.95, "a 3 s loop (4 beats across a tempo change) wraps after 3 s");
        run (t, 150, bs, sr);                                  // 1.5 s more: 1.5 s at 120 = 3 beats
        CHECK (std::abs (t.posBeats.load() - 2.5) < 0.05, "and 1.5 s into the next lap is beat 2.5 (1 s for the first two beats, then 0.5 s at 60 bpm)");
    }
    // ---- count-in length follows the time signature ----
    {
        Transport t; t.setBpm (120.0); t.setTimeSignature (3, 4); t.countInBars.store (1); t.setRecording (true); t.locate (0.0); run (t, 1, bs, sr); t.play(); run (t, 1, bs, sr);
        CHECK (t.countingIn.load() && t.posBeats.load() < -2.9 && t.posBeats.load() > -3.1, "a one-bar count-in in 3/4 is three beats");
    }

    // HOT CUE: a loop moved while the song plays takes effect at the chosen point, and the song jumps to its start
    {
        const double sr = 48000.0; const int bs = 256; const double beatsPerBlock = bs / sr * 2.0;      // 120 bpm
        auto upTo = [&] (Transport& t, double beat) { for (int i = 0; i < 20000 && t.posBeats.load() < beat; ++i) run (t, 1, bs, sr); };
        auto setUp = [&] (Transport& t, int mode, double ls, double le, double at) { t.setBpm (120.0); t.setLoop (true, ls, le); t.loopJump.store (mode); t.locate (ls); run (t, 1, bs, sr); t.play(); upTo (t, at); };
        { Transport t; setUp (t, 0, 0.0, 16.0, 9.0); t.requestLoop (true, 32.0, 48.0); CHECK (! t.loopPending.load() && t.loopStart.load() == 32.0, "with hot cue off a moved loop takes effect at once, as before"); }
        auto jumpsAt = [&] (int mode, double le, double from, double want, const char* what)
        {
            Transport t; setUp (t, mode, 0.0, le, from); t.requestLoop (true, 32.0, 48.0);
            const bool waiting = t.loopPending.load() && t.loopStart.load() == 0.0 && std::abs (t.pendAt.load() - want) < 1.0e-9;
            double before = t.posBeats.load(), highest = before; for (int i = 0; i < 20000 && t.posBeats.load() < 31.0; ++i) { run (t, 1, bs, sr); if (t.posBeats.load() < 31.0) highest = std::max (highest, t.posBeats.load()); }
            const double landed = t.posBeats.load();
            CHECK (waiting && highest <= want + 1.0e-6 && highest > want - 2.0 * beatsPerBlock && landed >= 32.0 && landed < 32.0 + 2.0 * beatsPerBlock && ! t.loopPending.load() && t.loopStart.load() == 32.0 && t.loopEnd.load() == 48.0, what);
        };
        jumpsAt (1, 16.0, 9.3, 9.5, "next 16th: from beat 9.3 it plays to 9.5, then jumps to the start of the new loop");
        jumpsAt (2, 16.0, 9.3, 10.0, "next beat: it plays to beat 10, then jumps");
        jumpsAt (3, 16.0, 9.3, 12.0, "end of bar: it plays to the end of the bar (beat 12), then jumps");
        jumpsAt (4, 16.0, 9.3, 16.0, "end of loop: from bar 3 it plays to the end of the playing loop (beat 16), then jumps");
        jumpsAt (3, 10.0, 9.3, 10.0, "never later than the end of the playing loop: a bar that ends after the loop does jumps at the loop end");
        { Transport t; setUp (t, 4, 0.0, 16.0, 5.0); t.requestLoop (true, 32.0, 48.0); t.requestLoop (true, 64.0, 80.0); CHECK (t.loopPending.load() && t.pendStart.load() == 64.0 && t.loopStart.load() == 0.0, "moving it again before the cue point replaces where it is going");
          upTo (t, 63.0); CHECK (t.posBeats.load() >= 64.0 && t.loopStart.load() == 64.0, "and it jumps to the latest one"); }
        { Transport t; setUp (t, 4, 0.0, 16.0, 5.0); t.requestLoop (false, 0.0, 16.0); CHECK (! t.loopPending.load() && ! t.looping.load(), "switching the loop off is immediate"); }
        { Transport t; setUp (t, 4, 0.0, 16.0, 5.0); t.requestLoop (true, 32.0, 48.0); t.stop(); run (t, 2, bs, sr); CHECK (! t.loopPending.load() && t.loopStart.load() == 32.0, "stopping while it waits puts the loop where it was going"); }
        { Transport t; t.setBpm (120.0); t.setLoop (true, 0.0, 16.0); t.loopJump.store (4); t.requestLoop (true, 8.0, 12.0); CHECK (! t.loopPending.load() && t.loopStart.load() == 8.0, "and while stopped a move is immediate"); }
    }

    // HOT CUES: three stored loops; pressing one jumps to it and it repeats until another is pressed
    {
        const double sr = 48000.0; const int bs = 256; const double beatsPerBlock = bs / sr * 2.0;
        auto upTo = [&] (Transport& t, double beat) { for (int i = 0; i < 20000 && t.posBeats.load() < beat; ++i) run (t, 1, bs, sr); };
        auto setUp = [&] (Transport& t, int mode) { t.setBpm (120.0); t.setLoop (true, 0.0, 16.0); t.setCue (1, 32.0, 40.0); t.loopJump.store (mode); t.locate (0.0); run (t, 1, bs, sr); t.play(); upTo (t, 5.0); };
        { Transport t; t.setBpm (120.0); t.setLoop (true, 4.0, 20.0); CHECK (t.cues[0].set.load() && t.cues[0].start.load() == 4.0 && t.cues[0].end.load() == 20.0 && ! t.cues[1].set.load() && ! t.triggerCue (1), "cue 1 is the loop on the ruler, and cues 2 and 3 start empty"); }
        { Transport t; setUp (t, 0); CHECK (t.loopStart.load() == 0.0 && t.cues[1].set.load(), "giving an unplayed cue its loop does not move the playing one");
          t.triggerCue (1); run (t, 2, bs, sr); CHECK (t.activeCue.load() == 1 && t.loopStart.load() == 32.0 && t.posBeats.load() >= 32.0 && t.posBeats.load() < 32.0 + 3.0 * beatsPerBlock, "with nothing ticked a pad jumps at once");
          for (int i = 0; i < 20000 && t.posBeats.load() < 39.9; ++i) run (t, 1, bs, sr); run (t, 60, bs, sr); CHECK (t.posBeats.load() >= 32.0 && t.posBeats.load() < 40.0, "and that loop keeps repeating");
          CHECK (t.cues[0].start.load() == 0.0 && t.cues[0].end.load() == 16.0, "cue 1 keeps its own loop"); t.triggerCue (0); run (t, 2, bs, sr); CHECK (t.activeCue.load() == 0 && t.loopStart.load() == 0.0 && t.posBeats.load() < 3.0 * beatsPerBlock, "pressing cue 1 goes back to it"); }
        { Transport t; setUp (t, 3); t.triggerCue (1); CHECK (t.loopPending.load() && t.activeCue.load() == 1 && t.loopStart.load() == 0.0, "with Bar ticked the pad waits");
          upTo (t, 31.0); CHECK (t.posBeats.load() >= 32.0 && t.loopStart.load() == 32.0 && t.cues[0].start.load() == 0.0, "and jumps at the end of the bar, leaving cue 1 as it was"); }
        { Transport t; setUp (t, 4); t.triggerCue (1); t.requestLoop (false, 0.0, 16.0); CHECK (! t.looping.load() && t.cues[1].start.load() == 32.0 && t.cues[1].end.load() == 40.0, "switching the loop off while a pad waits keeps that cue's loop"); }
        { Transport t; setUp (t, 4); t.triggerCue (1); t.setCue (1, 48.0, 56.0); upTo (t, 47.0); CHECK (t.loopStart.load() == 48.0 && t.posBeats.load() >= 48.0, "moving a waiting cue changes where it lands"); }
        { Transport t; t.setBpm (120.0); t.setLoop (true, 0.0, 16.0); t.setCue (2, 64.0, 72.0); t.triggerCue (2); run (t, 2, bs, sr); CHECK (! t.playing.load() && t.loopStart.load() == 64.0 && t.posBeats.load() == 64.0, "while stopped a pad moves the loop and the playhead to its start");
          CHECK (t.removeCue (2) && t.activeCue.load() == 0 && t.loopStart.load() == 0.0 && ! t.cues[2].set.load() && ! t.removeCue (0), "removing the playing cue falls back to cue 1, which cannot be removed"); }
        { Transport t; t.setBpm (120.0); t.setLoop (true, 0.0, 8.0); t.locate (21.5); run (t, 1, bs, sr); t.cueHere (1); CHECK (t.cues[1].start.load() == 20.0 && t.cues[1].end.load() == 28.0 && t.loopStart.load() == 0.0, "an empty pad makes a loop at the current bar, as long as the playing one"); }
    }

    std::printf("%d checks, %d failed\n", checks, failures);
    return failures == 0 ? 0 : 1;
}
