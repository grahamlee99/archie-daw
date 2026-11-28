// ARCHIE DAW -- THE POLISH PROBE (2026-11-27, Release 1 polish brief). It MEASURES the engine as it is and prints numbers with a stated limit and PASS or FAIL. It changes nothing in the program
// and is not part of render_test: a FAIL here is a finding to work on, not a broken build. Offline (no sound card, no window). Build target polish_probe, run it with ASKARCHIE_HOST_TEST=1.
#include "../Source/Engine.h"
#include "../../alien-delay/Source/DelayEngine.h"
#include <cstdio>
#include <vector>
#include <map>
#include <set>

#if defined(_WIN32)
extern "C" __declspec (dllimport) unsigned int __stdcall SetErrorMode (unsigned int);
#endif
using namespace archie;

static int passes = 0, fails = 0;
static void verdict (const char* area, bool ok, const juce::String& what) { ok ? ++passes : ++fails; std::printf ("%s  [%s] %s\n", ok ? "PASS" : "FAIL", area, what.toRawUTF8()); }
static const double SR = 48000.0;

struct Base : public juce::AudioProcessor
{
    explicit Base (bool instrument) : juce::AudioProcessor (instrument ? BusesProperties().withOutput ("out", juce::AudioChannelSet::stereo(), true)
                                                                       : BusesProperties().withInput ("in", juce::AudioChannelSet::stereo(), true).withOutput ("out", juce::AudioChannelSet::stereo(), true)) {}
    const juce::String getName() const override { return "probe"; }
    void prepareToPlay (double, int) override {} void releaseResources() override {}
    double getTailLengthSeconds() const override { return 0; } bool acceptsMidi() const override { return true; } bool producesMidi() const override { return false; }
    juce::AudioProcessorEditor* createEditor() override { return nullptr; } bool hasEditor() const override { return false; }
    int getNumPrograms() override { return 1; } int getCurrentProgram() override { return 0; } void setCurrentProgram (int) override {}
    const juce::String getProgramName (int) override { return {}; } void changeProgramName (int, const juce::String&) override {}
    void getStateInformation (juce::MemoryBlock&) override {} void setStateInformation (const void*, int) override {}
};
// records every note message with the sample it arrived on, and keeps count of the keys it believes are down
struct NoteProbe : public Base
{
    struct Ev { long long at; int note; bool on; };
    NoteProbe() : Base (true) { seen.reserve (200000); }
    void processBlock (juce::AudioBuffer<float>& b, juce::MidiBuffer& m) override
    {
        for (const auto meta : m)
        {
            const auto msg = meta.getMessage();
            if (msg.isNoteOn()) { seen.push_back ({ pos + meta.samplePosition, msg.getNoteNumber(), true }); down.insert (msg.getNoteNumber()); }
            else if (msg.isNoteOff()) { seen.push_back ({ pos + meta.samplePosition, msg.getNoteNumber(), false }); down.erase (msg.getNoteNumber()); }
            else if (msg.isAllNotesOff() || msg.isAllSoundOff()) down.clear();
        }
        pos += b.getNumSamples(); b.clear();
    }
    void reset() override {}
    std::vector<Ev> seen; std::set<int> down; long long pos = 0;
};
// a steady tone (an instrument that never stops), for listening to switches
struct Tone : public Base
{
    Tone() : Base (true) {}
    void processBlock (juce::AudioBuffer<float>& b, juce::MidiBuffer&) override
    {
        for (int i = 0; i < b.getNumSamples(); ++i) { const float v = 0.25f * (float) std::sin (ph); ph += 2.0 * juce::MathConstants<double>::pi * 220.0 / SR; for (int c = 0; c < b.getNumChannels(); ++c) b.setSample (c, i, v); }
    }
    double ph = 0.0;
};
// an effect: passes the sound at a chosen level, a chosen number of samples late
struct Fx : public Base
{
    Fx (float g, int late) : Base (false), gain (g), delay (late) { ring.setSize (2, 4096); ring.clear(); setLatencySamples (late); }
    void processBlock (juce::AudioBuffer<float>& b, juce::MidiBuffer&) override
    {
        for (int i = 0; i < b.getNumSamples(); ++i, ++w) for (int c = 0; c < 2; ++c) { float* x = b.getWritePointer (c); ring.setSample (c, w & 4095, x[i]); x[i] = gain * ring.getSample (c, (w - delay) & 4095); }
    }
    float gain; int delay, w = 0; juce::AudioBuffer<float> ring;
};
struct Pattern : public Base
{
    Pattern() : Base (true) { addParameter (bank = new juce::AudioParameterInt ({ "activeBank", 1 }, "Active Bank", 0, 15, 0)); }
    void processBlock (juce::AudioBuffer<float>& b, juce::MidiBuffer&) override
    {
        const int now = bank->get();
        if (now != last) { double ppq = -1.0; if (auto* h = getPlayHead()) if (auto p = h->getPosition()) ppq = p->getPpqPosition().orFallback (-1.0); changes.push_back ({ pos, now, ppq }); last = now; }
        pos += b.getNumSamples(); b.clear();
    }
    struct Change { long long at; int bank; double ppqSeen; };
    juce::AudioParameterInt* bank = nullptr; int last = 0; long long pos = 0; std::vector<Change> changes;
};

static int heard = 0;      // which output channel run() keeps
static std::vector<float> run (HostEngine& e, int samples, int bs)
{
    std::vector<float> out; out.reserve ((size_t) samples + (size_t) bs); std::vector<float> l ((size_t) bs), r ((size_t) bs); float* o[2] = { l.data(), r.data() };
    for (int done = 0; done < samples; done += bs) { e.processOffline (nullptr, 0, o, 2, bs); const auto& side = heard == 0 ? l : r; out.insert (out.end(), side.begin(), side.end()); }
    return out;
}
static juce::var makeClip (double beats, const std::vector<std::array<double, 3>>& notes)      // { beat, length, note }
{
    auto* c = new juce::DynamicObject(); c->setProperty ("lengthBeats", beats); juce::Array<juce::var> a;
    for (const auto& n : notes) { auto* o = new juce::DynamicObject(); o->setProperty ("beat", n[0]); o->setProperty ("len", n[1]); o->setProperty ("note", (int) n[2]); o->setProperty ("vel", 100); a.add (juce::var (o)); }
    c->setProperty ("notes", a); return juce::var (c);
}
static double biggestStep (const std::vector<float>& x, size_t from, size_t to) { double m = 0; for (size_t i = std::max<size_t> (1, from); i < std::min (to, x.size()); ++i) m = std::max (m, (double) std::abs (x[i] - x[i - 1])); return m; }

int main()
{
    juce::ScopedJuceInitialiser_GUI init; std::setvbuf (stdout, nullptr, _IONBF, 0);
   #if JUCE_WINDOWS
    SetErrorMode (0x0001 | 0x0002);
   #endif
    const int sizes[] = { 32, 64, 256, 512, 1024 };

    std::printf ("\n== START ==  (limit: a note on the first beat sounds on the first sample)\n");
    for (int bs : sizes)
    {
        HostEngine e; e.prepareOffline (SR, bs, 0); auto p = std::make_unique<NoteProbe>(); auto* pr = p.get(); e.addProcessor (0, std::move (p), "p");
        e.editAddInstance (0, 0.0, 4.0, 0.0, makeClip (4.0, { { 0, 0.5, 60 } }), 0); e.transport.locate (0.0); run (e, 4 * bs, bs); const long long before = pr->pos; pr->seen.clear(); e.transport.play(); run (e, 4 * bs, bs);
        const long long at = pr->seen.empty() ? -1 : pr->seen[0].at - before; verdict ("start", at == 0, "block " + juce::String (bs) + ": first note " + (at < 0 ? juce::String ("never arrived") : juce::String ((int) at) + " samples after play"));
    }

    std::printf ("\n== TIMING ==  (limit: every note within 1 sample of its place, none missing)\n");
    for (double bpm : { 120.0, 140.0, 173.7 }) for (int bs : sizes)
    {
        HostEngine e; e.prepareOffline (SR, bs, 0); e.transport.setBpm (bpm); run (e, bs, bs); auto p = std::make_unique<NoteProbe>(); auto* pr = p.get(); e.addProcessor (0, std::move (p), "p");
        std::vector<std::array<double, 3>> notes; for (int i = 0; i < 128; ++i) notes.push_back ({ i * 0.25, 0.125, 48.0 + (i % 12) });
        e.editAddInstance (0, 0.0, 32.0, 0.0, makeClip (32.0, notes), 0); e.transport.locate (0.0); run (e, 2 * bs, bs); const long long from = pr->pos; pr->seen.clear(); e.transport.play(); run (e, (int) (32.0 * 60.0 / bpm * SR) + 4 * bs, bs);
        // for each of the 128 places: how many note-ons arrived near it (a step is a quarter beat), and how far the first of them was from it
        const double step = 0.25 * 60.0 / bpm * SR; std::vector<int> count (128, 0); long long worst = 0; int extra = 0;
        for (const auto& v : pr->seen) if (v.on) { const long long at = v.at - from; const int k = (int) std::llround ((double) at / step); if (k < 0 || k >= 128) continue; if (count[(size_t) k]++ == 0) worst = std::max (worst, std::llabs (at - (long long) std::llround (k * step))); else ++extra; }
        int missing = 0; for (int c : count) if (c == 0) ++missing;
        verdict ("timing", missing == 0 && extra == 0 && worst <= 1, juce::String (bpm, 1) + " bpm, block " + juce::String (bs) + ": missing " + juce::String (missing) + ", played twice " + juce::String (extra) + ", worst " + juce::String ((int) worst) + " samples out");
    }

    std::printf ("\n== LOOP ==  (limit: every turn evenly spaced to 1 sample, no note left down)\n");
    for (int bs : sizes)
    {
        HostEngine e; e.prepareOffline (SR, bs, 0); e.transport.setBpm (137.0); run (e, bs, bs); auto p = std::make_unique<NoteProbe>(); auto* pr = p.get(); e.addProcessor (0, std::move (p), "p");
        e.editAddInstance (0, 0.0, 4.0, 0.0, makeClip (4.0, { { 0, 0.5, 60 }, { 3.5, 0.5, 62 } }), 0); e.transport.setLoop (true, 0.0, 4.0); e.transport.locate (0.0); run (e, 2 * bs, bs); pr->seen.clear(); e.transport.play(); run (e, (int) (80.0 * 60.0 / 137.0 * SR), bs);
        std::vector<long long> ons; for (const auto& v : pr->seen) if (v.on && v.note == 60) ons.push_back (v.at); const double want = 4.0 * 60.0 / 137.0 * SR; double worst = 0; for (size_t i = 1; i < ons.size(); ++i) worst = std::max (worst, std::abs ((double) (ons[i] - ons[0]) - want * (double) i));
        e.transport.stop(); run (e, 4 * bs, bs);
        verdict ("loop", ons.size() >= 19 && worst <= 1.5 && pr->down.empty(), "block " + juce::String (bs) + ": " + juce::String ((int) ons.size()) + " turns, worst drift " + juce::String (worst, 1) + " samples, keys left down " + juce::String ((int) pr->down.size()));
    }

    std::printf ("\n== PATTERN BLOCKS ==  (limit: the new pattern is chosen within 1 ms (48 samples) of its block's edge)\n");
    for (int bs : sizes)
    {
        HostEngine e; e.prepareOffline (SR, bs, 0); e.transport.setBpm (140.0); run (e, bs, bs); auto p = std::make_unique<Pattern>(); auto* pp = p.get(); e.addProcessor (0, std::move (p), "p"); e.service();
        for (int i = 0; i < 16; ++i) { auto* c = new juce::DynamicObject(); c->setProperty ("lengthBeats", 1.0); c->setProperty ("patternBank", 1 + (i % 2)); e.editAddInstance (0, (double) i, 1.0, 0.0, juce::var (c), 0); }
        e.service(); e.transport.locate (0.0); run (e, 2 * bs, bs); const long long from = pp->pos; pp->changes.clear(); pp->last = pp->bank->get(); e.transport.play(); run (e, (int) (15.5 * 60.0 / 140.0 * SR), bs);
        double worst = 0; int n = 0; for (const auto& ch : pp->changes) { const double beat = (double) (ch.at - from) / SR * 140.0 / 60.0; const double edge = std::floor (beat + 0.5); if (edge < 1.0) continue; worst = std::max (worst, std::abs (beat - edge) * 60.0 / 140.0 * SR); ++n; }
        verdict ("pattern", n >= 14 && worst <= 48.0, "block " + juce::String (bs) + ": " + juce::String (n) + " changes, the furthest from its edge " + juce::String ((int) std::lround (worst)) + " samples (" + juce::String (worst / SR * 1000.0, 1) + " ms)");
    }

    std::printf ("\n== MIDI OVERFLOW ==  (limit: no key is ever left down; no note-off lost)\n");
    for (int burst : { 200, 500, 600, 2000 })
    {
        HostEngine e; e.prepareOffline (SR, 256, 0); auto p = std::make_unique<NoteProbe>(); auto* pr = p.get(); e.addProcessor (0, std::move (p), "p"); run (e, 1024, 256);
        for (int i = 0; i < burst; ++i) { e.noteOn (0, i % 100, 100); e.noteOff (0, i % 100); }
        run (e, 4096, 256);
        verdict ("overflow", pr->down.empty(), juce::String (burst * 2) + " notes messages inside one block: keys left down " + juce::String ((int) pr->down.size()));
    }
    for (int flood : { 500, 900, 1100, 3000 })
    {
        HostEngine e; e.prepareOffline (SR, 256, 0); auto p = std::make_unique<NoteProbe>(); auto* pr = p.get(); e.addProcessor (0, std::move (p), "p"); e.selectTrack (0); run (e, 1024, 256);
        e.handleIncomingMidiMessage (nullptr, juce::MidiMessage::noteOn (1, 60, (juce::uint8) 100)); run (e, 256, 256);
        if (pr->down.empty()) { verdict ("overflow", false, "(the probe never got the key: the test is wrong)"); continue; }
        for (int i = 0; i < flood; ++i) e.handleIncomingMidiMessage (nullptr, juce::MidiMessage::controllerEvent (1, 1, i % 128));
        e.handleIncomingMidiMessage (nullptr, juce::MidiMessage::noteOff (1, 60)); run (e, 4096, 256);
        verdict ("overflow", pr->down.empty(), "a held key, " + juce::String (flood) + " wheel messages, then the key let go: " + (pr->down.empty() ? "released" : "STILL SOUNDING"));
    }

    std::printf ("\n== HANGING NOTES ==  (limit: no key left down after the player has let go)\n");
    {
        HostEngine e; e.prepareOffline (SR, 256, 0); auto a = std::make_unique<NoteProbe>(); auto* pa = a.get(); e.addProcessor (0, std::move (a), "a"); auto b = std::make_unique<NoteProbe>(); auto* pb = b.get(); e.addProcessor (1, std::move (b), "b");
        e.selectTrack (0); run (e, 1024, 256); e.handleIncomingMidiMessage (nullptr, juce::MidiMessage::noteOn (1, 60, (juce::uint8) 100)); run (e, 512, 256);
        e.selectTrack (1); e.handleIncomingMidiMessage (nullptr, juce::MidiMessage::noteOff (1, 60)); run (e, 1024, 256);
        verdict ("hanging", pa->down.empty() && pb->down.empty(), juce::String ("another track selected while a key is held: first track ") + (pa->down.empty() ? "released" : "STILL SOUNDING"));
    }
    {
        HostEngine e; e.prepareOffline (SR, 256, 0); auto a = std::make_unique<NoteProbe>(); auto* pa = a.get(); e.addProcessor (0, std::move (a), "a"); e.selectTrack (0); run (e, 1024, 256);
        e.handleIncomingMidiMessage (nullptr, juce::MidiMessage::noteOn (1, 61, (juce::uint8) 100)); run (e, 512, 256); e.setBypass (0, 0, true); run (e, 512, 256);
        e.handleIncomingMidiMessage (nullptr, juce::MidiMessage::noteOff (1, 61)); run (e, 512, 256); e.setBypass (0, 0, false); run (e, 1024, 256);
        verdict ("hanging", pa->down.empty(), juce::String ("key let go while the instrument is bypassed, then bypass off: ") + (pa->down.empty() ? "released" : "STILL SOUNDING"));
    }
    {
        HostEngine e; e.prepareOffline (SR, 256, 0); auto a = std::make_unique<NoteProbe>(); auto* pa = a.get(); e.addProcessor (0, std::move (a), "a"); run (e, 512, 256);
        e.editAddInstance (0, 0.0, 8.0, 0.0, makeClip (8.0, { { 0, 6.0, 65 } }), 0); e.transport.locate (0.0); e.transport.play(); run (e, 12000, 256); e.setBypass (0, 0, true); run (e, 480000, 256); e.transport.stop(); run (e, 1024, 256); e.setBypass (0, 0, false); run (e, 1024, 256);
        verdict ("hanging", pa->down.empty(), juce::String ("a clip note ends while its instrument is bypassed, then bypass off: ") + (pa->down.empty() ? "released" : "STILL SOUNDING"));
    }
    {
        HostEngine e; e.prepareOffline (SR, 256, 0); auto a = std::make_unique<NoteProbe>(); auto* pa = a.get(); e.addProcessor (0, std::move (a), "a"); e.selectTrack (0); run (e, 1024, 256);
        e.handleIncomingMidiMessage (nullptr, juce::MidiMessage::noteOn (1, 62, (juce::uint8) 100)); run (e, 512, 256); e.panicAll(); run (e, 1024, 256);
        verdict ("hanging", pa->down.empty(), "Panic releases a held key");
    }

    std::printf ("\n== CLICKS AT SWITCHES ==  (limit: no step between samples bigger than 1.5 times the tone's own biggest step)\n");
    {
        auto trial = [] (const char* name, std::function<void (HostEngine&)> setup, std::function<void (HostEngine&, bool)> flip)
        {
            double worst = 0, base = 0, change = 0;
            for (int offset = 0; offset < 8; ++offset)      // at eight places in the tone's cycle
            {
                HostEngine e; e.prepareOffline (SR, 256, 0); if (juce::String (name) == "instrument bypass") e.addProcessor (0, std::make_unique<Tone>(), "tone"); else e.trackAt (0)->source.store (2); setup (e); auto a = run (e, 9600 + offset * 256, 256); base = std::max (base, biggestStep (a, a.size() - 2400, a.size()));
                for (bool on : { true, false }) { const std::vector<float> was (a.end() - 4096, a.end()); flip (e, on); auto b = run (e, 9600, 256); base = std::max (base, biggestStep (b, b.size() - 2400, b.size()));      // (the louder of the two sounds is the yardstick)
                     if (on) { double d = 0; for (size_t i = 0; i < 4096; ++i) d = std::max (d, (double) std::abs (std::abs (b[b.size() - 4096 + i]) - std::abs (was[i]))); double r1 = 0, r2 = 0; for (size_t i = 0; i < 4096; ++i) { r1 += (double) was[i] * was[i]; r2 += (double) b[b.size() - 4096 + i] * b[b.size() - 4096 + i]; } change = std::max (change, std::abs (r2 - r1) / std::max (1.0e-12, r1) + (b[b.size() - 1] * was.back() < 0 ? 0.0 : 0.0)); } b.insert (b.begin(), a.back()); worst = std::max (worst, biggestStep (b, 1, b.size())); a = b; }
            }
            const double ratio = worst / std::max (1.0e-9, base);
            verdict ("click", ratio <= 1.5, juce::String (name) + ": biggest step " + juce::String (ratio, 1) + " times the tone's own" + (change < 0.01 ? " (level did not change)" : " (level changed)"));
        };
        trial ("track mute", [] (HostEngine&) {}, [] (HostEngine& e, bool on) { e.trackAt (0)->mute.store (on); e.updateSolo(); });
        trial ("fader jump of 24 dB", [] (HostEngine&) {}, [] (HostEngine& e, bool on) { e.trackAt (0)->gainDb.store (on ? -24.0f : 0.0f); });
        heard = 1;
        trial ("pan hard left and back", [] (HostEngine&) {}, [] (HostEngine& e, bool on) { e.trackAt (0)->pan.store (on ? -1.0f : 0.0f); });
        heard = 0;
        trial ("effect bypass (effect halves the level)", [] (HostEngine& e) { e.addProcessor (0, std::make_unique<Fx> (0.5f, 0), "half"); }, [] (HostEngine& e, bool on) { e.setBypass (0, 0, on); });
        trial ("effect bypass (effect with 5 ms delay)", [] (HostEngine& e) { e.addProcessor (0, std::make_unique<Fx> (1.0f, 240), "late"); }, [] (HostEngine& e, bool on) { e.setBypass (0, 0, on); });
        trial ("instrument bypass", [] (HostEngine&) {}, [] (HostEngine& e, bool on) { e.setBypass (0, 0, on); });
        trial ("adding and removing an effect", [] (HostEngine&) {}, [] (HostEngine& e, bool on) { if (on) e.addProcessor (0, std::make_unique<Fx> (0.5f, 0), "half"); else e.removePlugin (0, 0); });
        trial ("channel high-pass on and off", [] (HostEngine&) {}, [] (HostEngine& e, bool on) { e.trackAt (0)->eqHpHz.store (400.0f); e.trackAt (0)->eqHpOn.store (on); });
        trial ("input polarity", [] (HostEngine&) {}, [] (HostEngine& e, bool on) { e.trackAt (0)->inInvert.store (on); });
        trial ("Panic", [] (HostEngine&) {}, [] (HostEngine& e, bool on) { if (on) e.panicAll(); });
    }

    std::printf ("\n== SAMPLE AND HOLD (Alien Delay) ==  (limit: the new value is fully there within 1 ms of the beat)\n");
    {
        const double bpm = 140.0, per = bpm / 60.0 / SR, len = 0.5;
        alien::Engine d; d.prepare (SR); alien::Settings s; s.mask = alien::kStraight; s.sync = true; s.mix = 0.45f; s.sh = true; s.shBeats = (float) len; s.shDepth = 1.0f; s.highCutHz = 7000.0f;
        std::vector<double> cut; juce::Random r (7);
        for (int i = 0; i < (int) (8.0 / per); ++i) { float a = 0.25f * (r.nextFloat() * 2.0f - 1.0f), b = a; d.process (&a, &b, 1, s, bpm, i * per, true); cut.push_back (d.highCutNow()); }
        double worstStart = 0, worstDone = 0; int n = 0;
        for (int k = 1; k < 16; ++k)
        {
            const int edge = (int) std::ceil (k * len / per - 1.0e-6); const double before = cut[(size_t) edge - 1], after = cut[(size_t) (edge + (int) (0.1 / per)) - 1]; if (std::abs (after - before) < 200.0) continue;
            int start = -1, done = -1; for (int i = edge - 4; i < edge + (int) (0.1 / per); ++i) { const double moved = (cut[(size_t) i] - before) / (after - before); if (start < 0 && moved > 0.01) start = i; if (done < 0 && moved > 0.95) done = i; }
            worstStart = std::max (worstStart, (double) std::abs (start - edge)); worstDone = std::max (worstDone, (double) (done - edge)); ++n;
        }
        verdict ("s&h", worstStart <= 48, juce::String (n) + " steps: the move starts within " + juce::String ((int) worstStart) + " samples of the beat");
        verdict ("s&h", worstDone <= 48, "filter mode: 95 % of the way there " + juce::String (worstDone / SR * 1000.0, 1) + " ms after the beat");
        // how much can be heard: the loudness above 3 kHz of the whole output, step by step, at the preset's depth and mix
        auto swing = [&] (float depth, float mix)
        {
            alien::Engine x; x.prepare (SR); alien::Settings q = s; q.shDepth = depth; q.mix = mix; q.feedback = 0.65f; juce::Random rr (7); double lo = 1.0e9, hi = -1.0e9, acc = 0; int count = 0; float last = 0; int step = 0;
            for (int i = 0; i < (int) (16.0 / per); ++i) { float a = 0.25f * (rr.nextFloat() * 2.0f - 1.0f), b = a; x.process (&a, &b, 1, q, bpm, i * per, true); const float h = a - last; last = a; acc += (double) h * h; ++count;
                if ((int) std::floor ((i + 1) * per / len) != step) { const double v = 10.0 * std::log10 (acc / count + 1.0e-20); if (step > 4) { lo = std::min (lo, v); hi = std::max (hi, v); } acc = 0; count = 0; ++step; } }
            return hi - lo;
        };
        const double preset = swing (0.8f, 0.45f), strongest = swing (1.0f, 1.0f);
        verdict ("s&h", preset >= 6.0, "the preset (depth 80 %, mix 45 %): brightness differs between steps by " + juce::String (preset, 1) + " dB (limit: at least 6 dB to be plainly heard)");
        std::printf ("      (for comparison, depth 100 %% with only the echoes: %.1f dB)\n", strongest);
    }

    std::printf ("\n== WEIGHT ==  (recorded, no limit yet)\n");
    {
        const auto t0 = juce::Time::getMillisecondCounterHiRes(); auto e = std::make_unique<HostEngine>(); e->prepareOffline (SR, 256, 0); const auto t1 = juce::Time::getMillisecondCounterHiRes();
        run (*e, 480000, 256); const auto t2 = juce::Time::getMillisecondCounterHiRes();
        std::printf ("      engine ready in %.0f ms; ten seconds of %d empty tracks rendered in %.0f ms (%.1f %% of one core)\n", t1 - t0, e->trackCount.load(), t2 - t1, (t2 - t1) / 10000.0 * 100.0);
    }
    std::printf ("\n%d passed, %d failed\n", passes, fails);
    return 0;
}
