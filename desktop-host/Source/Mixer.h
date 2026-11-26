// Archie Studio host (desktop) -- the mixer's building blocks: the per-track state, and the small processors a channel is made of
// (source, fader strip, send, master). They are ordinary audio processors that the MixRenderer (Renderer.h) calls directly, in a fixed order, on the audio thread.
#pragma once
#include <juce_audio_utils/juce_audio_utils.h>
#include <juce_audio_processors/juce_audio_processors.h>
#include <juce_audio_devices/juce_audio_devices.h>
#include <juce_audio_formats/juce_audio_formats.h>
#include <juce_dsp/juce_dsp.h>
#if JUCE_WINDOWS
 #ifndef NOMINMAX
  #define NOMINMAX
 #endif
 #include <windows.h>       // (only for the plug-in window screenshot: PrintWindow)
#endif
#include <atomic>
#include <array>
#include <memory>
#include "Transport.h"
#include "ArchieLink.h"

namespace archie
{

constexpr int kTracks = 32;        // the most tracks there can be; the program starts with kStartTracks and the person adds more as they go
constexpr int kStartTracks = 4;
constexpr int kNewSongTracks = 0;    // the program (and "New") starts with the Transport and the Master only, as Reason does: instruments are dragged in or added
constexpr int kAutoColours[] = { 2, 21, 13, 27, 8, 17, 1, 23, 29, 4 };    // each new track takes the next of these palette colours (Orange, Sky Blue, Kelly Green, Plum, Pineapple, Turquoise, Red, Slate Blue, Deep Purple, Ochre)
constexpr int kSends = 8, kReturns = 8;     // each track has kSends sends; each send feeds the FX return of the same number
constexpr int kReturnBase = 100, kMasterChain = 200;   // plug-in chains are addressed by number: 0.. a track, 100.. an FX return, 200 the master inserts

struct NoteEv { double beat = 0, len = 0.25; int note = 60, vel = 100, ch = 1; };
// The sound of an audio clip: read once from a file, kept at the file's own sample rate, and never changed afterwards (the audio thread reads it without waiting).
struct AudioData
{
    juce::AudioBuffer<float> buf;                 // 1 or 2 channels
    double rate = 48000.0;
    juce::String path, name;
    std::vector<float> peaks;                     // the loudest sample (either channel) in each block of kPeakBlock samples: what the waveform is drawn from
    static constexpr int kPeakBlock = 256;
    int numSamples() const { return buf.getNumSamples(); }
    double durationSec() const { return rate > 0.0 ? (double) buf.getNumSamples() / rate : 0.0; }
};
// A clip's content is notes (MIDI) or sound (audio). An audio clip has no notes and lengthBeats 0; its length is the sound's own length in seconds.
// patternBank >= 0 makes the clip a PATTERN BLOCK: it has no notes, it says which pattern a self-sequencing plug-in (Archie Drums) plays for as long as the block lasts (the DAW sets the plug-in's own pattern control at the block edges).
struct Clip   { double lengthBeats = 4.0; std::vector<NoteEv> notes; int id = 0; std::shared_ptr<const AudioData> audio; int patternBank = -1; };

// A clip INSTANCE: a place on the timeline that shows some shared CONTENT (the notes). Several instances can point at the same content, and editing the content changes all of them.
// start/length are in beats on the song timeline (length 0 = it goes on looping for ever, which is what the single pattern of the first versions was); offset is how far into the content
// the instance begins (beats for a MIDI clip, SECONDS into the sound for an audio clip). The content loops inside the instance whenever the instance is longer than the content.
// A warp marker pins a moment of the sound (sec: seconds into the sound) to a place in the sound's own beat count (beat: beats from the sound's beat 0; the clip's place on the timeline is worked
// out from it). Between two markers the sound is stretched evenly; outside them it keeps the source tempo.
struct WarpMarker { double sec = 0.0, beat = 0.0; };
// (gainDb, fadeIn, fadeOut and reverse belong to an AUDIO instance, not to the sound: two copies of one sound can have different volumes, fades and direction.)
struct ClipInst
{
    int id = 0; double start = 0.0, length = 0.0, offset = 0.0; std::shared_ptr<const Clip> content;
    // WARP (audio clips): the sound follows the tempo. srcBpm is the tempo the sound was made at. `warped` is the sound already stretched to this clip on this tempo map (made in the background by
    // renderWarp); warpKey says which clip/tempo it was made for. While it is not ready the clip plays unwarped.
    std::vector<WarpMarker> markers;       // sorted, both columns strictly increasing; ignored for a reversed clip
    bool warp = false, repitch = false; double srcBpm = 0.0; std::shared_ptr<const AudioData> warped; uint64_t warpKey = 0;
    // TRANSPOSE (audio clips): the clip's pitch moved by this many semitones (-24 to 24, cents allowed) with its length and timing unchanged, made in the background by retunePitch. keepFormant keeps a voice's colour.
    // A warped clip's transposed sound is part of `warped` (one render); a clip that is not warped plays `pitched` (the whole source, moved) in place of its source. While a render is not ready the clip plays unmoved.
    double transpose = 0.0; bool keepFormant = true; std::shared_ptr<const AudioData> pitched; uint64_t pitchKey = 0;
    bool muted = false; juce::String label;  // a muted clip is silent (and kept); the name is the clip's own label (empty = none)
    int colour = -1;                         // the clip's own colour (palette number), or -1 = it takes its track's colour
    int groove = -2;                         // the groove channel this clip plays through (1 to 31): -2 = the track's own choice, -1 = none
    float gainDb = 0.0f; double fadeIn = 0.002, fadeOut = 0.002; bool reverse = false;     // fades in seconds; 0 = none, which is right only where this clip joins another piece of the SAME sound with nothing between
};
struct Arrangement { std::vector<ClipInst> insts; };       // sorted by start, immutable once published (the audio thread reads it without waiting)

// what the current audio block is doing (written by the audio callback before the mixer renders, read by the track processors)
struct BlockInfo
{
    Segment segs[2];
    int numSegs = 0;
    bool playing = false, recording = false, panic = false, chase = false;
    double bpm = 120.0, sr = 48000.0;
    double swingDelay = 0.0, swingGrid = 0.5;   // groove: every second grid step is played this many beats late (0 = straight)
    const struct GrooveSet* groove = nullptr;   // the groove channels (read by the track processors)
    double blockStartMs = 0.0;           // when this block started (the hi-res millisecond clock the MIDI timestamps use)
};

struct RecEvent { double beat = 0; juce::uint8 b[3] {}; };

inline std::unique_ptr<Clip> emptyClip() { auto c = std::make_unique<Clip>(); c->lengthBeats = 0.0; return c; }     // "no pattern" is a clip of length 0
// Where the sound starts a new note, hit or word, as times in seconds from the start of the sound. Nothing is guessed from a name or a tempo: it listens for the sound getting suddenly louder
// than it has been for the last 60 ms (measured on the sudden changes in the signal, so a swelling pad does not count), and at sensitivity 0 only big hits count, at 1 small ones too.
// FILTERING/MONITORING MECHANISM: this reads the loudness of the sound and decides which moments are "starts". It does NOT judge pitch, rhythm, tempo or what the sound is. It is adjusted
// by its sensitivity (0 to 1), and the test suite feeds it clicks, a steady tone and a swell to check what it finds and what it leaves alone.
inline std::vector<double> detectTransients (const AudioData& a, double sensitivity)
{
    std::vector<double> out; const int n = a.numSamples(); if (n < 1024 || a.rate <= 0.0) return out;
    const int ch = a.buf.getNumChannels(); const float* L = a.buf.getReadPointer (0); const float* R = a.buf.getReadPointer (ch > 1 ? 1 : 0);
    auto mono = [&] (int i) { return 0.5f * (L[i] + R[i]); };
    const int hop = juce::jmax (32, (int) (a.rate * 0.003)), win = hop * 2, frames = (n - win) / hop; if (frames < 8) return out;
    std::vector<double> e ((size_t) frames), os ((size_t) frames, 0.0);
    for (int f = 0; f < frames; ++f) { double s = 0.0; for (int i = f * hop; i < f * hop + win; ++i) { const double d = (double) mono (i + 1) - (double) mono (i); s += d * d; } e[(size_t) f] = s / win; }
    const int back = 20;                                           // ~60 ms
    for (int f = 1; f < frames; ++f)
    {
        double avg = 0.0; int c = 0; for (int k = juce::jmax (0, f - back); k < f; ++k) { avg += e[(size_t) k]; ++c; } avg /= c;
        os[(size_t) f] = juce::jmax (0.0, std::log10 (e[(size_t) f] + 1.0e-10) - std::log10 (avg + 1.0e-10));
    }
    const double thr = 1.3 - 1.0 * juce::jlimit (0.0, 1.0, sensitivity);        // in decades of energy: 1.3 = 20x louder, 0.3 = 2x
    const int gap = (int) (0.05 * a.rate / hop); int last = -1000000;
    for (int f = 2; f < frames - 2; ++f)
    {
        if (os[(size_t) f] < thr || e[(size_t) f] < 1.0e-9) continue;
        bool peak = true; for (int k = f - 2; k <= f + 2; ++k) if (os[(size_t) k] > os[(size_t) f]) peak = false;
        if (! peak || f - last < gap) continue;
        // back to where the rise began, then to the first sample that is clearly above the quiet before it
        int s = f; while (s > 1 && os[(size_t) (s - 1)] > 0.5 * os[(size_t) f]) --s;
        const int w0 = juce::jmax (0, s * hop - hop), w1 = juce::jmin (n - 1, s * hop + 3 * hop);
        float pk = 0.0f; for (int i = w0; i <= w1; ++i) pk = juce::jmax (pk, std::abs (mono (i)));
        int at = s * hop + hop; for (int i = w0; i <= w1; ++i) if (std::abs (mono (i)) > 0.1f * pk) { at = i; break; }
        out.push_back ((double) at / a.rate); last = f;
    }
    return out;
}
inline std::unique_ptr<Arrangement> emptyArrangement() { return std::make_unique<Arrangement>(); }
constexpr double kForever = 1.0e300;
// Every note-on/note-off of one instance that falls in the beat range [b0, b1): calls on(note, tBeat) and off(note, tBeat). A note the instance's end cuts short is ended AT the end.
// `chase` also reports the notes already sounding at b0 (started before it, still held) through chaseOn.
// GROOVE (swing): the beats inside a pair of grid steps are bent so the second step lands `d` beats late: [0, g) is stretched to [0, g + d) and [g, 2g) squeezed to [g + d, 2g). It is monotonic, so notes
// never swap order or overlap more than they did, and it repeats every 2g. Measured from the start of the clip's notes.
inline double grooveMap (double b, double g, double d)
{
    if (d <= 0.0 || g <= 0.0 || d >= g) return b;
    const double P = 2.0 * g, k = std::floor (b / P), r = b - k * P;
    return k * P + (r < g ? r * (g + d) / g : (g + d) + (r - g) * (g - d) / g);
}
// ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
// GROOVES (Groove dock): a groove PATCH is a repeating pattern over a grid (usually sixteenths, one bar long): for each step it says how much later or earlier the notes on that step play (timing, as a fraction of
// a step), how much louder or softer (velocity, percent) and how much longer or shorter (length, percent). A groove CHANNEL plays a patch through a strength and four more settings: the TIMING, VELOCITY and LENGTH
// impacts (percent of the patch's own amounts), a random timing in ticks (480 to the beat, repeatable: the same note always gets the same nudge), SLIDE (all notes later or earlier, as a percent of a step), SHUFFLE (a swing
// on every second step) and the ANCHOR (which step of the pattern lines up with the start of the clip). STRENGTH scales everything; at 0 the channel does nothing at all and the notes play exactly as written.
// A note keeps its own small deviations: the groove adds to where the note is, it does not snap it to the grid. Applied when the notes are played, never to the clip itself.
struct StepMod { float timing = 0.0f, vel = 0.0f, len = 0.0f; };
struct GroovePatch { juce::String name; double grid = 0.25, cycle = 4.0; std::vector<StepMod> steps; bool builtin = false; int stepCount() const { return juce::jmax (1, (int) std::lround (cycle / juce::jmax (1.0e-6, grid))); } };
struct GrooveChan { bool on = false; juce::String patch; float strength = 1.0f, timing = 100.0f, vel = 100.0f, len = 100.0f, rand = 0.0f, slide = 0.0f, shuffle = 0.0f; int anchor = 0; };
constexpr int kGrooveChans = 32;       // four banks of eight; channel 0 (A1) is the global Swing, so the patch channels are 1 to 31
struct GrooveSet
{
    GrooveChan ch[kGrooveChans]; std::vector<GroovePatch> patches;
    const GroovePatch* find (const juce::String& name) const { for (const auto& p : patches) if (p.name == name) return &p; return nullptr; }
};
inline std::vector<GroovePatch> builtinGroovePatches()
{
    std::vector<GroovePatch> v;
    auto make = [&] (const char* name, double grid, auto fn) { GroovePatch p; p.name = name; p.grid = grid; p.cycle = 4.0; p.builtin = true; p.steps.resize ((size_t) p.stepCount()); for (int i = 0; i < p.stepCount(); ++i) fn (i, p.steps[(size_t) i]); v.push_back (std::move (p)); };
    make ("Straight", 0.25, [] (int, StepMod&) {});
    for (int pct : { 54, 58, 62, 66 }) make ((juce::String ("16th swing ") + juce::String (pct) + "%").toRawUTF8(), 0.25, [pct] (int i, StepMod& m) { if (i & 1) m.timing = (float) (pct - 50) / 50.0f; });
    for (int pct : { 58, 66 }) make ((juce::String ("8th swing ") + juce::String (pct) + "%").toRawUTF8(), 0.5, [pct] (int i, StepMod& m) { if (i & 1) m.timing = (float) (pct - 50) / 50.0f; });
    make ("Laid back", 0.25, [] (int, StepMod& m) { m.timing = 0.12f; });
    make ("Pushed", 0.25, [] (int i, StepMod& m) { m.timing = (i % 4 == 0) ? 0.0f : -0.10f; });
    make ("Accent the beat", 0.25, [] (int i, StepMod& m) { m.vel = (i % 4 == 0) ? 20.0f : (i % 2 == 0) ? 6.0f : -15.0f; });
    make ("Funk 16ths", 0.25, [] (int i, StepMod& m) { if (i & 1) m.timing = 0.22f; m.vel = (i % 4 == 0) ? 15.0f : (i & 1) ? -20.0f : -5.0f; });
    make ("Staccato", 0.25, [] (int, StepMod& m) { m.len = -45.0f; });
    make ("Legato", 0.25, [] (int, StepMod& m) { m.len = 40.0f; });
    return v;
}
inline GrooveSet defaultGrooveSet() { GrooveSet g; g.patches = builtinGroovePatches(); return g; }
// what one groove channel does to one note (message-free and allocation-free: it runs on the audio thread)
struct GrooveEval
{
    const GrooveChan* c = nullptr; const GroovePatch* p = nullptr; int instId = 0;
    bool active() const { return c != nullptr && c->on && c->strength > 0.0f; }
    static GrooveEval make (const GrooveSet& set, int ch, int instId)
    {
        GrooveEval e; if (ch < 1 || ch >= kGrooveChans) return e; e.c = &set.ch[ch]; e.instId = instId; if (e.c->on && e.c->strength > 0.0f) e.p = set.find (e.c->patch); return e;
    }
    static double hashUnit (uint64_t a, uint64_t b, uint64_t d)      // repeatable pseudo-random number in -1..1
    { uint64_t x = a * 0x9E3779B97F4A7C15ull ^ (b + 0x632BE59BD9B4E019ull) * 0xBF58476D1CE4E5B9ull ^ (d + 0x94D049BB133111EBull) * 0xD6E8FEB86659FD93ull; x ^= x >> 31; x *= 0x7FB5D329728EA185ull; x ^= x >> 27; x *= 0x81DADEF4BC2DD44Dull; x ^= x >> 33; return (double) (x >> 11) / 9007199254740992.0 * 2.0 - 1.0; }
    void apply (const NoteEv& nt, double& beat, double& len, int& vel) const
    {
        const double grid = p != nullptr ? p->grid : 0.25, s = (double) c->strength; const int n = p != nullptr ? p->stepCount() : 16;
        const long long idx = std::llround (nt.beat / grid); const int step = (int) (((idx + c->anchor) % n + n) % n);
        StepMod m; if (p != nullptr && (size_t) step < p->steps.size()) m = p->steps[(size_t) step];
        double off = (double) m.timing * grid * ((double) c->timing / 100.0) * s + (double) c->slide / 100.0 * grid * s;
        if (idx & 1) off += (double) c->shuffle / 100.0 * grid / 3.0 * s;
        if (c->rand > 0.0f) off += hashUnit ((uint64_t) instId, (uint64_t) std::llround (nt.beat * 480.0), (uint64_t) nt.note) * (double) c->rand / 480.0 * s;
        beat = nt.beat + off;
        len = juce::jmax (0.02, nt.len * (1.0 + (double) m.len / 100.0 * ((double) c->len / 100.0) * s));
        vel = juce::jlimit (1, 127, (int) std::lround ((double) nt.vel * (1.0 + (double) m.vel / 100.0 * ((double) c->vel / 100.0) * s)));
    }
};
template <typename On, typename Off, typename Chase>
inline void walkInstance (const ClipInst& in, double b0, double b1, bool chase, On&& on, Off&& off, Chase&& chaseOn, double swingDelay = 0.0, double swingGrid = 0.5, const GrooveEval* gv = nullptr, bool oneShot = false)
{
    const Clip* c = in.content.get();
    if (c == nullptr || c->lengthBeats <= 0.0 || in.muted) return;
    const double ws = in.start, we = in.length > 0.0 ? in.start + in.length : kForever, L = c->lengthBeats, base = ws - in.offset;
    if (we <= b0 - 1.0e-9 || ws >= b1) return;
    for (const auto& nt0 : c->notes)
    {
        NoteEv nt = nt0; double gBeat = nt0.beat, gLen = nt0.len;
        if (gv != nullptr) { gv->apply (nt0, gBeat, gLen, nt.vel); gBeat = juce::jlimit (0.0, juce::jmax (0.0, L - 0.001), gBeat); }       // the groove: a little earlier or later, louder or softer, longer or shorter (never before the clip's start)
        const double nb = grooveMap (gBeat, swingGrid, swingDelay), ne = grooveMap (gBeat + gLen, swingGrid, swingDelay);
        const double len = juce::jmin (juce::jmax (0.01, ne - nb), L - 0.01);
        int k = juce::jmax (0, (int) std::floor ((b0 - base - nb - len) / L) - 1);
        for (;; ++k)
        {
            const double tOn = base + nb + (double) k * L;
            if (tOn >= b1 - 1.0e-12 || tOn >= we) break;
            if (tOn < ws - 1.0e-9) continue;                       // before the instance begins (the part of the content that the offset skips)
            if (oneShot && tOn >= base + L - 1.0e-9) break;        // ONE SHOT: the content plays once, from the clip's start, and is then silent for the rest of the clip
            const double tOff = juce::jmin (tOn + len, we);
            if (tOn >= b0 - 1.0e-9) on (nt, tOn);
            else if (chase && tOff > b0 + 1.0e-9) chaseOn (nt);
            if (tOff >= b0 - 1.0e-9 && tOff < b1 - 1.0e-12 && tOff > tOn) off (nt, tOff);
        }
    }
}
struct FileAudio { juce::AudioBuffer<float> buf; bool valid = false; };

// One live MIDI message (never a system-exclusive one): up to three bytes and the time it arrived.
struct LiveMsg { juce::uint8 b[3] {}; juce::uint8 size = 0; double timeMs = 0.0; };

// A lock-free queue that many threads (MIDI inputs, the console, a web page) can push to while the audio thread pops. Nobody ever waits and nobody takes a lock,
// which the library's message collector could not promise: it takes a lock the audio thread shares with the MIDI thread. (Bounded multi-producer queue after D. Vyukov.)
class MidiFifo
{
public:
    MidiFifo() { for (size_t i = 0; i < kSize; ++i) cells[i].seq.store (i, std::memory_order_relaxed); }
    bool push (const LiveMsg& m)
    {
        size_t pos = tail.load (std::memory_order_relaxed); Cell* c = nullptr;
        for (;;)
        {
            c = &cells[pos & kMask];
            const size_t seq = c->seq.load (std::memory_order_acquire);
            const std::intptr_t dif = (std::intptr_t) seq - (std::intptr_t) pos;
            if (dif == 0) { if (tail.compare_exchange_weak (pos, pos + 1, std::memory_order_relaxed)) break; }
            else if (dif < 0) return false;              // full: the message is dropped, never waited for
            else pos = tail.load (std::memory_order_relaxed);
        }
        c->msg = m; c->seq.store (pos + 1, std::memory_order_release);
        return true;
    }
    bool pop (LiveMsg& m)
    {
        size_t pos = head.load (std::memory_order_relaxed); Cell* c = nullptr;
        for (;;)
        {
            c = &cells[pos & kMask];
            const size_t seq = c->seq.load (std::memory_order_acquire);
            const std::intptr_t dif = (std::intptr_t) seq - (std::intptr_t) (pos + 1);
            if (dif == 0) { if (head.compare_exchange_weak (pos, pos + 1, std::memory_order_relaxed)) break; }
            else if (dif < 0) return false;              // empty
            else pos = head.load (std::memory_order_relaxed);
        }
        m = c->msg; c->seq.store (pos + kMask + 1, std::memory_order_release);
        return true;
    }
private:
    static constexpr size_t kSize = 1024, kMask = kSize - 1;
    struct Cell { std::atomic<size_t> seq; LiveMsg msg; };
    Cell cells[kSize];
    std::atomic<size_t> head { 0 }, tail { 0 };
};
inline bool liveOf (const juce::MidiMessage& m, double timeMs, LiveMsg& out)
{
    const int n = m.getRawDataSize();
    if (n < 1 || n > 3 || m.isSysEx()) return false;
    for (int i = 0; i < n; ++i) out.b[i] = m.getRawData()[i];
    out.size = (juce::uint8) n; out.timeMs = timeMs;
    return true;
}

// A hosted plug-in's own window, with a slim toolbar above the plug-in's editor (the layout other DAWs use): KEEP ON TOP (stays above the other windows while you work in the DAW) and AUTOMATE (press it,
// then move a control in the plug-in: the parameter you move gets an automation lane; press it again to stop). The toolbar only calls back into the engine; it never touches the plug-in.
class PluginWindow : public juce::DocumentWindow, private juce::Timer
{
public:
    PluginWindow (juce::AudioProcessor& p, std::function<void()> automate = {}, std::function<bool()> learning = {}, std::function<void (int)> remote = {}, std::function<int()> remoteStage = {})
        : juce::DocumentWindow (p.getName(), juce::Colour (0xff202225), juce::DocumentWindow::closeButton | juce::DocumentWindow::minimiseButton), onAutomate (std::move (automate)), onRemote (std::move (remote)), remoteStageFn (std::move (remoteStage)), isLearning (std::move (learning))
    {
        setUsingNativeTitleBar (true);
        std::unique_ptr<juce::Component> ed;
        if (auto* e = p.hasEditor() ? p.createEditorIfNeeded() : nullptr) ed.reset (e); else ed = std::make_unique<juce::GenericAudioProcessorEditor> (p);
        holder = new Holder (std::move (ed), [this] { setAlwaysOnTop (! isAlwaysOnTop()); refreshButtons(); }, [this] { if (onAutomate) onAutomate(); refreshButtons(); }, [this] { remoteMenu(); }, [this] { screenshot(); });
        setContentOwned (holder, true);
        setResizable (false, false);
        centreWithSize (getWidth(), getHeight());
        setVisible (true);
        refreshButtons(); startTimer (250);
    }
    void closeButtonPressed() override { setVisible (false); }
private:
    struct Holder : public juce::Component
    {
        static constexpr int kBar = 28;
        Holder (std::unique_ptr<juce::Component> e, std::function<void()> keepFn, std::function<void()> autoFn, std::function<void()> remoteFn, std::function<void()> shotFn) : editor (std::move (e))
        {
            addAndMakeVisible (*editor); addAndMakeVisible (keep); addAndMakeVisible (autom); addAndMakeVisible (remote); addAndMakeVisible (shot);
            remote.setTooltip ("Remote: assign a knob or slider on your MIDI controller to a control in this plug-in"); shot.setTooltip ("Save a picture of this plug-in window to your Pictures folder");
            remote.onClick = std::move (remoteFn); shot.onClick = std::move (shotFn);
            keep.setClickingTogglesState (false); autom.setClickingTogglesState (false);
            keep.setTooltip ("Keep this plug-in window above the other windows"); autom.setTooltip ("Press, then move a control in the plug-in: that parameter gets an automation lane");
            keep.onClick = std::move (keepFn); autom.onClick = std::move (autoFn);
            setSize (juce::jmax (260, editor->getWidth()), editor->getHeight() + kBar);
        }
        void resized() override
        {
            auto r = getLocalBounds(); auto bar = r.removeFromTop (kBar).reduced (4, 3);
            keep.setBounds (bar.removeFromLeft (110)); bar.removeFromLeft (6); autom.setBounds (bar.removeFromLeft (150)); bar.removeFromLeft (6); remote.setBounds (bar.removeFromLeft (190)); bar.removeFromLeft (6); shot.setBounds (bar.removeFromLeft (100));
            editor->setBounds (r.withSizeKeepingCentre (editor->getWidth(), juce::jmin (r.getHeight(), editor->getHeight())).withX (juce::jmax (0, (r.getWidth() - editor->getWidth()) / 2)).withY (r.getY()));
        }
        void paint (juce::Graphics& g) override { g.fillAll (juce::Colour (0xff202225)); }
        std::unique_ptr<juce::Component> editor; juce::TextButton keep { "Keep on top" }, autom { "Automate" }, remote { "Remote" }, shot { "Screenshot" };
    };
    void refreshButtons()
    {
        if (holder == nullptr) return;
        holder->keep.setToggleState (isAlwaysOnTop(), juce::dontSendNotification); holder->keep.setButtonText (isAlwaysOnTop() ? "Kept on top" : "Keep on top");
        const bool on = isLearning && isLearning(); holder->autom.setToggleState (on, juce::dontSendNotification); holder->autom.setButtonText (on ? "Move a control..." : "Automate");
        const int rs = remoteStageFn ? remoteStageFn() : 0; holder->remote.setToggleState (rs != 0, juce::dontSendNotification); holder->remote.setButtonText (rs == 1 ? "Move a control..." : rs == 2 ? "Now move a controller knob..." : "Remote");
    }
    void remoteMenu()
    {
        if (! onRemote) return; if ((remoteStageFn ? remoteStageFn() : 0) != 0) { onRemote (0); refreshButtons(); return; }
        juce::PopupMenu mn; mn.addItem (1, "Assign a controller: move a control here, then a knob on your controller"); mn.addItem (2, "Clear the controller assignments of this track");
        mn.showMenuAsync (juce::PopupMenu::Options().withTargetComponent (&holder->remote), [this] (int r) { if (r != 0 && onRemote) onRemote (r); refreshButtons(); });
    }
    // a picture of the plug-in's window as it looks on screen (a plug-in with a web page inside it is only captured by asking the system for the window itself)
    void screenshot()
    {
        juce::Image img;
#if JUCE_WINDOWS
        if (auto* peer = getPeer())
            if (auto hwnd = (HWND) peer->getNativeHandle())
            {
                RECT rc {}; GetClientRect (hwnd, &rc); const int w = rc.right - rc.left, h = rc.bottom - rc.top;
                if (w > 0 && h > 0)
                {
                    BITMAPINFO bi {}; bi.bmiHeader.biSize = sizeof (BITMAPINFOHEADER); bi.bmiHeader.biWidth = w; bi.bmiHeader.biHeight = -h; bi.bmiHeader.biPlanes = 1; bi.bmiHeader.biBitCount = 32; bi.bmiHeader.biCompression = BI_RGB;
                    void* bits = nullptr; HDC dc = GetDC (nullptr); HDC mem = CreateCompatibleDC (dc); HBITMAP bmp = CreateDIBSection (dc, &bi, DIB_RGB_COLORS, &bits, nullptr, 0);
                    if (bmp != nullptr && bits != nullptr)
                    {
                        auto old = SelectObject (mem, bmp); PrintWindow (hwnd, mem, 2 /* PW_RENDERFULLCONTENT */); SelectObject (mem, old);
                        img = juce::Image (juce::Image::ARGB, w, h, false); juce::Image::BitmapData bd (img, juce::Image::BitmapData::writeOnly);
                        for (int y = 0; y < h; ++y) { auto* src = (const juce::uint32*) bits + (size_t) y * (size_t) w; auto* dst = (juce::uint32*) bd.getLinePointer (y); for (int x = 0; x < w; ++x) dst[x] = src[x] | 0xff000000u; }
                    }
                    if (bmp != nullptr) DeleteObject (bmp); DeleteDC (mem); ReleaseDC (nullptr, dc);
                }
            }
#endif
        if (! img.isValid() && holder != nullptr) img = holder->createComponentSnapshot (holder->getLocalBounds());
        if (! img.isValid()) return;
        auto dir = juce::File::getSpecialLocation (juce::File::userPicturesDirectory).getChildFile ("Archie DAW"); dir.createDirectory();
        auto f = dir.getNonexistentChildFile (getName() + " " + juce::Time::getCurrentTime().formatted ("%Y-%m-%d %H-%M-%S"), ".png", false);
        if (auto os = std::unique_ptr<juce::FileOutputStream> (f.createOutputStream())) { juce::PNGImageFormat png; png.writeImageToStream (img, *os); }
        f.revealToUser();
    }
    void timerCallback() override { refreshButtons(); }
    Holder* holder = nullptr; std::function<void()> onAutomate; std::function<void (int)> onRemote; std::function<int()> remoteStageFn; std::function<bool()> isLearning;
};

// RECORD AUTOMATION: told when a person grabs, moves or lets go of a control in a plug-in's own window (the plug-in reports it to the host as a "gesture"). It only passes the news on; the engine decides what to write.
// A change that is not inside a gesture (the plug-in moving its own value, or a lane the host is playing back) is passed on too, marked as such, so it can be ignored.
class ParamTouch : public juce::AudioProcessorParameter::Listener
{
public:
    ParamTouch (juce::AudioProcessor& p, std::function<void (int, float, int)> cb) : proc (p), onEvent (std::move (cb)) { for (auto* q : proc.getParameters()) q->addListener (this); }
    ~ParamTouch() override { for (auto* q : proc.getParameters()) q->removeListener (this); }
    void parameterValueChanged (int i, float v) override { onEvent (i, v, 0); }
    void parameterGestureChanged (int i, bool starting) override { onEvent (i, 0.0f, starting ? 1 : 2); }
private:
    juce::AudioProcessor& proc; std::function<void (int, float, int)> onEvent;
};

// What a self-sequencing plug-in (Archie Drums) is told about the song when pattern blocks are used: the song position counted from the start of the block it is playing (so its pattern starts at the block's first step), and,
// in the gaps between blocks, that the song is not running (so it plays no new hit: silence, the tails of earlier hits ring out). Everything else is the host's own position.
struct PatternPlayHead : public juce::AudioPlayHead
{
    juce::AudioPlayHead* base = nullptr; std::atomic<bool> silent { false }; std::atomic<double> origin { 0.0 };
    juce::Optional<PositionInfo> getPosition() const override
    {
        if (base == nullptr) return {};
        auto p = base->getPosition(); if (! p.hasValue()) return p;
        if (silent.load (std::memory_order_relaxed)) p->setIsPlaying (false);
        const double o = origin.load (std::memory_order_relaxed);
        if (o != 0.0) { if (auto q = p->getPpqPosition()) p->setPpqPosition (*q - o); if (auto b = p->getPpqPositionOfLastBarStart()) p->setPpqPositionOfLastBarStart (*b - o); }
        return p;
    }
};
struct Slot
{
    std::unique_ptr<PatternPlayHead> patHead;          // (declared before the plug-in so it outlives it)
    std::unique_ptr<juce::AudioProcessor> proc;      // a hosted plug-in, or one of our own devices
    std::unique_ptr<ParamTouch> touch;                // (declared right after the plug-in so it is gone before the plug-in is: it removes itself from the plug-in's parameters)
    juce::PluginDescription desc;
    std::atomic<bool> bypass { false };               // read live by the audio thread: toggling it needs no republish
    std::atomic<juce::AudioProcessorParameter*> patBank { nullptr }, patLink { nullptr };      // a self-sequencing plug-in's pattern and link controls (found by name from the message thread), set by pattern blocks; they live and die with the plug-in
    std::atomic<juce::AudioProcessorParameter*> patSub { nullptr };                            // a plug-in that splits its pattern over two controls (Archie Drums: "Pattern" A-D and "Bank" 1-4): patBank is the letter, patSub the number
    std::atomic<float> inPk { 0.0f }, outPk { 0.0f };    // the loudest sample into and out of this device since the console last asked (audio thread writes, the 30 Hz meter reads and clears)
    std::atomic<bool> patChecked { false }; std::atomic<int> patApplied { -1 };                // (patApplied: set by the audio thread)
    // The pattern as ONE number 0..N-1 (A1 = 0, A2 = 1 ... D4 = 15), whichever way the plug-in holds it: a single stepped control, or letter and number controls
    int patTotal() const
    {
        auto* pb = patBank.load(); if (pb == nullptr) return 0;
        if (auto* ps = patSub.load()) return pb->getNumSteps() * ps->getNumSteps();
        const int n = pb->getNumSteps(); if (n > 1 && n < 1024) return n;
        // a plug-in that does not state its step count (a built plug-in's whole-number control reports "no steps"): read the top of the range from the control's own text ("15" at the top = sixteen patterns); sixteen if that is not a number
        const int top = pb->getText (1.0f, 16).getIntValue(); return (top >= 1 && top < 64) ? top + 1 : 16;
    }
    int patGet() const
    {
        auto* pb = patBank.load(); if (pb == nullptr) return 0;
        if (auto* ps = patSub.load())
        {
            const int a = pb->getNumSteps(), b = ps->getNumSteps();
            return juce::jlimit (0, a - 1, (int) std::lround (pb->getValue() * (float) (a - 1))) * b + juce::jlimit (0, b - 1, (int) std::lround (ps->getValue() * (float) (b - 1)));
        }
        const int top = patTotal() - 1; return juce::jlimit (0, top, (int) std::lround (pb->getValue() * (float) top));
    }
    void patSet (int want) const
    {
        auto* pb = patBank.load(); if (pb == nullptr) return;
        if (auto* ps = patSub.load())
        {
            const int a = pb->getNumSteps(), b = ps->getNumSteps(), k = juce::jlimit (0, a * b - 1, want);
            pb->setValue ((float) (k / b) / (float) juce::jmax (1, a - 1)); ps->setValue ((float) (k % b) / (float) juce::jmax (1, b - 1)); return;
        }
        const int top = patTotal() - 1; pb->setValue (juce::jlimit (0.0f, 1.0f, (float) want / (float) top));
    }
    std::atomic<bool> resetNext { false };            // Panic: the audio thread calls the plug-in's reset() before its next block (ends every voice and clears delay and reverb tails)
    std::unique_ptr<PluginWindow> window;
    juce::String linkId;                                  // names this copy of a connectable device across sessions (saved with the song), so each copy keeps its own connection to Ask Archie
    std::unique_ptr<archielink::DeviceLink> link;        // the connection to the person's Ask Archie account (Archie devices only); declared after proc so it ends first
};

// Audio being recorded on a track: the audio thread writes what the sound card hears into the ring (only while the track is armed, its source is the sound-card input, and the song is recording and
// playing); the message thread drains it into a take. It is made when a track is first armed (about 1.5 MB), never on the audio thread.
struct AudioRec
{
    static constexpr int kCap = 192000;                      // 4 s at 48 kHz: the message thread drains it 30 times a second
    juce::AbstractFifo fifo { kCap };
    std::vector<float> ring[2] { std::vector<float> ((size_t) kCap), std::vector<float> ((size_t) kCap) };
    std::atomic<double> startSec { -1.0 };                   // timeline second of the first sample captured
    std::atomic<bool> overflow { false };
    std::vector<float> takeL, takeR; double takeStartSec = -1.0;       // message thread only
};
// AUTOMATION: a lane is a list of (beat, value) points that a parameter follows while the song plays. kind 0 = the channel's level (dB), 1 = its pan (-1..1), 2 = a plug-in parameter (0..1) in rack slot `slot`.
// Level and pan are applied by the audio thread at the start of every block (the fader's own 20 ms smoothing makes the steps smooth); plug-in parameters are applied from the 30 Hz tick.
struct AutoPt { double beat = 0.0, value = 0.0; };
struct AutoLane { int id = 0, kind = 0, slot = -1, param = -1; bool on = true; juce::String label; std::vector<AutoPt> pts; };
struct AutoSet { std::vector<AutoLane> lanes; };
// the lane's value at a beat: before its first point it holds the first value, after its last it holds the last, between them a straight line
inline double autoValueAt (const AutoLane& l, double beat)
{
    if (l.pts.empty()) return 0.0;
    if (beat <= l.pts.front().beat) return l.pts.front().value;
    if (beat >= l.pts.back().beat) return l.pts.back().value;
    size_t lo = 0, hi = l.pts.size() - 1; while (hi - lo > 1) { const size_t m = (lo + hi) / 2; if (l.pts[m].beat <= beat) lo = m; else hi = m; }
    const double span = l.pts[hi].beat - l.pts[lo].beat; if (span <= 1.0e-12) return l.pts[hi].value;
    return l.pts[lo].value + (l.pts[hi].value - l.pts[lo].value) * (beat - l.pts[lo].beat) / span;
}
class TrackSourceProcessor; class TrackStripProcessor; class SendProcessor;

// THE CHANNEL INPUT STAGE (Mixer, Input section): a gain of -24 to +24 dB and a polarity invert, applied to the channel's sound where it enters its effects (after the instrument, before the first effect).
// Neutral (0 dB, not inverted) touches nothing, so a channel that has not used it sounds exactly as before. The gain is smoothed over 20 ms so moving it, or inverting, never clicks.
struct InputStage
{
    static constexpr float kMinDb = -24.0f, kMaxDb = 24.0f;
    static float targetGain (float gainDb, bool invert) { const float g = juce::Decibels::decibelsToGain (juce::jlimit (kMinDb, kMaxDb, gainDb)); return invert ? -g : g; }
    void process (juce::AudioBuffer<float>& buf, int n, float gainDb, bool invert, double sr)
    {
        const float target = targetGain (gainDb, invert);
        if (rate != sr) { rate = sr; g.reset (sr, 0.02); g.setCurrentAndTargetValue (1.0f); }                  // it starts neutral and ramps to what is set
        if (target == 1.0f && g.getTargetValue() == 1.0f && ! g.isSmoothing()) return;                         // neutral: leave the sound exactly as it is
        g.setTargetValue (target);
        auto* l = buf.getWritePointer (0); auto* r = buf.getNumChannels() > 1 ? buf.getWritePointer (1) : nullptr;
        for (int i = 0; i < n; ++i) { const float v = g.getNextValue(); l[i] *= v; if (r != nullptr) r[i] *= v; }
    }
    juce::SmoothedValue<float> g { 1.0f }; double rate = 0.0;
};

// THE CHANNEL EQ STAGE (Mixer, EQ section): a high-pass and a low-pass filter, 12 dB per octave each, with their own on switch and frequency. It runs right after the input stage.
// With both off it touches nothing (a channel that has not used it sounds exactly as before). Switching a filter on or off is crossfaded over 5 ms so it never clicks.
struct EqStage
{
    static constexpr float kHpMin = 20.0f, kHpMax = 2000.0f, kLpMin = 1000.0f, kLpMax = 20000.0f;
    struct Biquad
    {
        float b0 = 1, b1 = 0, b2 = 0, a1 = 0, a2 = 0, z1[2] { 0, 0 }, z2[2] { 0, 0 };
        void clear() { z1[0] = z1[1] = z2[0] = z2[1] = 0; }
        void set (bool highPass, double sr, float hz)
        {
            const double f = juce::jlimit (10.0, sr * 0.45, (double) hz), w0 = juce::MathConstants<double>::twoPi * f / sr, c = std::cos (w0), alpha = std::sin (w0) / (2.0 * 0.70710678118654752), a0 = 1.0 + alpha;
            const double nb0 = highPass ? (1.0 + c) * 0.5 : (1.0 - c) * 0.5, nb1 = highPass ? -(1.0 + c) : (1.0 - c);
            b0 = (float) (nb0 / a0); b1 = (float) (nb1 / a0); b2 = (float) (nb0 / a0); a1 = (float) (-2.0 * c / a0); a2 = (float) ((1.0 - alpha) / a0);
        }
        float run (int ch, float x) { const float y = b0 * x + z1[ch]; z1[ch] = b1 * x - a1 * y + z2[ch]; z2[ch] = b2 * x - a2 * y; return y; }
    };
    void process (juce::AudioBuffer<float>& buf, int n, bool hpOn, float hpHz, bool lpOn, float lpHz, double sr)
    {
        if (rate != sr) { rate = sr; step = (float) (1.0 / juce::jmax (1.0, 0.005 * sr)); hpMix = lpMix = 0.0f; hpHzNow = lpHzNow = -1.0f; hp.clear(); lp.clear(); }
        const bool hpActive = hpOn || hpMix > 0.0f, lpActive = lpOn || lpMix > 0.0f;
        if (! hpActive && ! lpActive) { hp.clear(); lp.clear(); return; }                                  // neutral: leave the sound exactly as it is
        hpHz = juce::jlimit (kHpMin, kHpMax, hpHz); lpHz = juce::jlimit (kLpMin, kLpMax, lpHz);
        if (hpHz != hpHzNow) { hpHzNow = hpHz; hp.set (true, sr, hpHz); }
        if (lpHz != lpHzNow) { lpHzNow = lpHz; lp.set (false, sr, lpHz); }
        const int chs = juce::jmin (2, buf.getNumChannels());
        for (int i = 0; i < n; ++i)
        {
            hpMix = hpOn ? juce::jmin (1.0f, hpMix + step) : juce::jmax (0.0f, hpMix - step);
            lpMix = lpOn ? juce::jmin (1.0f, lpMix + step) : juce::jmax (0.0f, lpMix - step);
            for (int c = 0; c < chs; ++c)
            {
                float x = buf.getSample (c, i);
                if (hpActive) { const float y = hp.run (c, x); x += hpMix * (y - x); }
                if (lpActive) { const float y = lp.run (c, x); x += lpMix * (y - x); }
                buf.setSample (c, i, x);
            }
        }
    }
    Biquad hp, lp; float hpMix = 0.0f, lpMix = 0.0f, hpHzNow = -1.0f, lpHzNow = -1.0f, step = 0.01f; double rate = 0.0;
};

// THE PARAMETRIC EQ (the EQ panel, on every channel and on the master): four bands after the high-pass and low-pass filters. Band 1 is a LOW SHELF, bands 2 and 3 are BELLS, band 4 is a HIGH SHELF; each has an
// on switch, a frequency, a gain (-18 to +18 dB) and a Q (the width of a bell: 0.3 wide to 10 narrow; a shelf's steepness). Filters are the standard audio-EQ-cookbook biquads. A band that is off, or sits at 0 dB, changes
// nothing; switching one on or off is crossfaded over 5 ms so it never clicks. The panel also shows a live spectrum of the sound after the EQ: the audio thread keeps the last 16384 samples of it (only while a panel asks).
constexpr int kEqBands = 4;
struct EqBand { std::atomic<bool> on { false }; std::atomic<float> hz { 1000.0f }, gain { 0.0f }, q { 1.0f }; };
struct EqPanel
{
    EqPanel()
    {
        static const float hz[kEqBands] = { 100.0f, 400.0f, 2500.0f, 8000.0f }, q[kEqBands] = { 0.71f, 1.0f, 1.0f, 0.71f };
        for (int i = 0; i < kEqBands; ++i) { band[i].hz.store (hz[i]); band[i].q.store (q[i]); }
        for (auto& r : ring) r.store (0.0f);
        for (auto& hm : handUntilMs) hm.store (0.0);
    }
    static constexpr int kRing = 16384;
    EqBand band[kEqBands];
    std::array<std::atomic<double>, 20> handUntilMs {};      // an EQ setting that was just moved while recording: its lane leaves it alone until this time
    std::atomic<double> scopeUntilMs { 0.0 };            // a panel is showing the spectrum until this time
    std::array<std::atomic<float>, kRing> ring; std::atomic<int> ringPos { 0 };
    void feed (const juce::AudioBuffer<float>& b, int n)      // (audio thread) keep the sound for the panel's spectrum, left and right added
    {
        if (juce::Time::getMillisecondCounterHiRes() > scopeUntilMs.load (std::memory_order_relaxed)) return;
        int p = ringPos.load (std::memory_order_relaxed); const float* l = b.getReadPointer (0); const float* r = b.getNumChannels() > 1 ? b.getReadPointer (1) : l;
        for (int i = 0; i < n; ++i) { ring[(size_t) (p & (kRing - 1))].store (0.5f * (l[i] + r[i]), std::memory_order_relaxed); ++p; }
        ringPos.store (p, std::memory_order_release);
    }
};
// EQ AUTOMATION (lane kind 16): one lane per setting. param 0..15 is band param / 4 with param % 4 = on, frequency, gain, Q; 16 high-pass on, 17 high-pass frequency, 18 low-pass on, 19 low-pass frequency.
// A gain lane holds dB. Frequency and Q lanes hold 0 to 1 on a LOGARITHMIC scale, so a straight line drawn in the lane is an even, musical sweep. An on lane holds 0 or 1.
constexpr int kEqLaneKind = 16, kEqLaneParams = 20;
inline double eqLaneSetting (int param, double v)      // what the lane's value means for that setting
{
    const double u = juce::jlimit (0.0, 1.0, v);
    if (param >= 0 && param < 16) { const int what = param % 4; if (what == 0) return v >= 0.5 ? 1.0 : 0.0; if (what == 1) return 20.0 * std::pow (1000.0, u); if (what == 2) return juce::jlimit (-18.0, 18.0, v); return 0.3 * std::pow (10.0 / 0.3, u); }
    if (param == 16 || param == 18) return v >= 0.5 ? 1.0 : 0.0;
    return param == 17 ? 20.0 * std::pow (100.0, u) : 1000.0 * std::pow (20.0, u);      // high-pass 20 Hz to 2 kHz, low-pass 1 to 20 kHz
}
inline double eqLaneValue (int param, double setting)      // the other way: what a setting is as a lane value
{
    if (param >= 0 && param < 16) { const int what = param % 4; if (what == 0) return setting >= 0.5 ? 1.0 : 0.0; if (what == 1) return juce::jlimit (0.0, 1.0, std::log (juce::jmax (1.0, setting) / 20.0) / std::log (1000.0)); if (what == 2) return juce::jlimit (-18.0, 18.0, setting); return juce::jlimit (0.0, 1.0, std::log (juce::jmax (0.01, setting) / 0.3) / std::log (10.0 / 0.3)); }
    if (param == 16 || param == 18) return setting >= 0.5 ? 1.0 : 0.0;
    return juce::jlimit (0.0, 1.0, param == 17 ? std::log (juce::jmax (1.0, setting) / 20.0) / std::log (100.0) : std::log (juce::jmax (1.0, setting) / 1000.0) / std::log (20.0));
}
inline void applyEqLane (EqPanel& p, std::atomic<bool>& hpOn, std::atomic<float>& hpHz, std::atomic<bool>& lpOn, std::atomic<float>& lpHz, int param, double v)
{
    if (param < 0 || param >= kEqLaneParams) return;
    if (juce::Time::getMillisecondCounterHiRes() < p.handUntilMs[(size_t) param].load (std::memory_order_relaxed)) return;      // a hand is on it
    const double s = eqLaneSetting (param, v);
    if (param < 16) { auto& b = p.band[param / 4]; const int what = param % 4; if (what == 0) b.on.store (s != 0.0); else if (what == 1) b.hz.store ((float) s); else if (what == 2) b.gain.store ((float) s); else b.q.store ((float) s); return; }
    if (param == 16) hpOn.store (s != 0.0); else if (param == 17) hpHz.store ((float) s); else if (param == 18) lpOn.store (s != 0.0); else lpHz.store ((float) s);
}

struct ParaEq
{
    static constexpr float kGainMax = 18.0f, kQMin = 0.3f, kQMax = 10.0f, kHzMin = 20.0f, kHzMax = 20000.0f;
    struct Bq
    {
        float b0 = 1, b1 = 0, b2 = 0, a1 = 0, a2 = 0, z1[2] { 0, 0 }, z2[2] { 0, 0 };
        void clear() { z1[0] = z1[1] = z2[0] = z2[1] = 0; }
        float run (int ch, float x) { const float y = b0 * x + z1[ch]; z1[ch] = b1 * x - a1 * y + z2[ch]; z2[ch] = b2 * x - a2 * y; return y; }
        void set (int type, double sr, double hz, double gainDb, double q)      // type 0 low shelf, 1 bell, 2 high shelf
        {
            const double f = juce::jlimit (10.0, sr * 0.45, hz), w0 = juce::MathConstants<double>::twoPi * f / sr, c = std::cos (w0), s = std::sin (w0), A = std::pow (10.0, gainDb / 40.0), alpha = s / (2.0 * juce::jmax (0.1, q)), sA = std::sqrt (A);
            double nb0, nb1, nb2, a0, na1, na2;
            if (type == 1) { nb0 = 1.0 + alpha * A; nb1 = -2.0 * c; nb2 = 1.0 - alpha * A; a0 = 1.0 + alpha / A; na1 = -2.0 * c; na2 = 1.0 - alpha / A; }
            else if (type == 0) { nb0 = A * ((A + 1) - (A - 1) * c + 2 * sA * alpha); nb1 = 2 * A * ((A - 1) - (A + 1) * c); nb2 = A * ((A + 1) - (A - 1) * c - 2 * sA * alpha); a0 = (A + 1) + (A - 1) * c + 2 * sA * alpha; na1 = -2 * ((A - 1) + (A + 1) * c); na2 = (A + 1) + (A - 1) * c - 2 * sA * alpha; }
            else { nb0 = A * ((A + 1) + (A - 1) * c + 2 * sA * alpha); nb1 = -2 * A * ((A - 1) + (A + 1) * c); nb2 = A * ((A + 1) + (A - 1) * c - 2 * sA * alpha); a0 = (A + 1) - (A - 1) * c + 2 * sA * alpha; na1 = 2 * ((A - 1) - (A + 1) * c); na2 = (A + 1) - (A - 1) * c - 2 * sA * alpha; }
            b0 = (float) (nb0 / a0); b1 = (float) (nb1 / a0); b2 = (float) (nb2 / a0); a1 = (float) (na1 / a0); a2 = (float) (na2 / a0);
        }
    };
    static int typeOf (int band) { return band == 0 ? 0 : band == 3 ? 2 : 1; }
    void process (juce::AudioBuffer<float>& buf, int n, EqPanel& p, double sr)
    {
        if (rate != sr) { rate = sr; step = (float) (1.0 / juce::jmax (1.0, 0.005 * sr)); for (int i = 0; i < kEqBands; ++i) { mix[i] = 0.0f; hzNow[i] = gNow[i] = qNow[i] = -999.0f; bq[i].clear(); } }
        bool any = false; bool on[kEqBands];
        for (int i = 0; i < kEqBands; ++i)
        {
            on[i] = p.band[i].on.load (std::memory_order_relaxed); active[i] = on[i] || mix[i] > 0.0f; any = any || active[i];
            if (! active[i]) { bq[i].clear(); continue; }
            const float hz = juce::jlimit (kHzMin, kHzMax, p.band[i].hz.load()), g = juce::jlimit (-kGainMax, kGainMax, p.band[i].gain.load()), q = juce::jlimit (kQMin, kQMax, p.band[i].q.load());
            if (hz != hzNow[i] || g != gNow[i] || q != qNow[i]) { hzNow[i] = hz; gNow[i] = g; qNow[i] = q; bq[i].set (typeOf (i), sr, hz, g, q); }
        }
        if (! any) return;                                                         // neutral: leave the sound exactly as it is
        const int chs = juce::jmin (2, buf.getNumChannels());
        for (int s = 0; s < n; ++s)
        {
            for (int i = 0; i < kEqBands; ++i) if (active[i]) mix[i] = on[i] ? juce::jmin (1.0f, mix[i] + step) : juce::jmax (0.0f, mix[i] - step);
            for (int c = 0; c < chs; ++c)
            {
                float x = buf.getSample (c, s);
                for (int i = 0; i < kEqBands; ++i) if (active[i]) { const float y = bq[i].run (c, x); x += mix[i] * (y - x); }
                buf.setSample (c, s, x);
            }
        }
    }
    Bq bq[kEqBands]; float mix[kEqBands] {}, hzNow[kEqBands] {}, gNow[kEqBands] {}, qNow[kEqBands] {}, step = 0.01f; bool active[kEqBands] {}; double rate = 0.0;
};

// THE CHANNEL COMPRESSOR (Mixer, Dynamics section): a feed-forward compressor with both channels linked (one gain for left and right, so the stereo image holds still).
// The detector is a PEAK follower (or RMS, about 3 dB lower on a steady tone), the curve has a hard knee, and the gain reduction is smoothed with an attack (5 ms, or 0.5 ms on FAST) and a release (50 ms to 1.2 s).
// Make-up gain is added after the reduction. Off, it touches nothing (bit for bit); switching it on or off is crossfaded over 5 ms so it never clicks. It runs after the input stage and before the EQ.
struct CompStage
{
    static constexpr float kThrMin = -40.0f, kThrMax = 0.0f, kRatioMin = 1.1f, kRatioMax = 20.0f, kRelMin = 0.05f, kRelMax = 1.2f, kMakeMax = 24.0f;
    // returns the biggest gain reduction (dB, positive) seen in the block
    float process (juce::AudioBuffer<float>& buf, int n, bool on, float thr, float ratio, float relSec, bool fast, bool peakMode, float makeupDb, double sr, float attackSec = -1.0f)
    {
        if (rate != sr) { rate = sr; mixStep = (float) (1.0 / juce::jmax (1.0, 0.005 * sr)); mix = 0.0f; env = ms = 0.0f; grDb = 0.0f; }
        if (! on && mix <= 0.0f) { env = ms = 0.0f; grDb = 0.0f; return 0.0f; }                          // off: leave the sound exactly as it is
        thr = juce::jlimit (kThrMin, kThrMax, thr); ratio = juce::jlimit (kRatioMin, kRatioMax, ratio); relSec = juce::jlimit (kRelMin, kRelMax, relSec); makeupDb = juce::jlimit (0.0f, kMakeMax, makeupDb);
        const float atkSec = attackSec > 0.0f ? juce::jlimit (0.0001f, 0.1f, attackSec) : (fast ? 0.0005f : 0.005f);
        const float atk = (float) std::exp (-1.0 / ((double) atkSec * sr)), rel = (float) std::exp (-1.0 / (relSec * sr));
        const float detRel = (float) std::exp (-1.0 / (0.010 * sr)), rmsK = (float) (1.0 - std::exp (-1.0 / (0.020 * sr)));
        const float slope = 1.0f - 1.0f / ratio;
        const int chs = juce::jmin (2, buf.getNumChannels());
        auto* l = buf.getWritePointer (0); auto* r = chs > 1 ? buf.getWritePointer (1) : nullptr;
        float worst = 0.0f;
        for (int i = 0; i < n; ++i)
        {
            mix = on ? juce::jmin (1.0f, mix + mixStep) : juce::jmax (0.0f, mix - mixStep);
            const float xl = l[i], xr = r != nullptr ? r[i] : xl;
            float lvl;
            if (peakMode) { const float a = juce::jmax (std::abs (xl), std::abs (xr)); env = a > env ? a : a + (env - a) * detRel; lvl = env; }
            else { ms += ((xl * xl + xr * xr) * 0.5f - ms) * rmsK; lvl = std::sqrt (ms); }
            const float levelDb = 20.0f * std::log10 (juce::jmax (lvl, 1.0e-6f)), over = levelDb - thr, target = over > 0.0f ? over * slope : 0.0f;
            grDb = target + (grDb - target) * (target > grDb ? atk : rel);
            worst = juce::jmax (worst, grDb);
            const float g = juce::Decibels::decibelsToGain (makeupDb - grDb), k = 1.0f + mix * (g - 1.0f);
            l[i] = xl * k; if (r != nullptr) r[i] = xr * k;
        }
        return worst;
    }
    float env = 0.0f, ms = 0.0f, grDb = 0.0f, mix = 0.0f, mixStep = 0.01f; double rate = 0.0;
};

// THE CHANNEL GATE (Mixer, Dynamics section): turns the sound down when it falls below a threshold. Both channels are linked (one gain for left and right).
// GATE mode: below the threshold (with 2 dB of hysteresis, so it does not chatter) the gate waits for HOLD, then closes over RELEASE down to RANGE dB of reduction; above it, it opens over a short attack.
// EXP mode: a gentle 2:1 downward expander instead (each dB below the threshold costs one more dB), never deeper than RANGE. Off, it touches nothing (bit for bit); on/off is crossfaded over 5 ms.
struct GateStage
{
    static constexpr float kThrMin = -80.0f, kThrMax = 0.0f, kRangeMax = 80.0f, kRelMin = 0.05f, kRelMax = 4.0f, kHoldMax = 2.0f;
    // returns the biggest gain reduction (dB, positive) seen in the block
    float process (juce::AudioBuffer<float>& buf, int n, bool on, bool expander, bool fast, float thr, float range, float relSec, float holdSec, double sr)
    {
        if (rate != sr) { rate = sr; mixStep = (float) (1.0 / juce::jmax (1.0, 0.005 * sr)); mix = 0.0f; env = 0.0f; grDb = 0.0f; hold = 0; open = false; }
        if (! on && mix <= 0.0f) { env = 0.0f; grDb = 0.0f; hold = 0; open = false; return 0.0f; }               // off: leave the sound exactly as it is
        thr = juce::jlimit (kThrMin, kThrMax, thr); range = juce::jlimit (0.0f, kRangeMax, range); relSec = juce::jlimit (kRelMin, kRelMax, relSec); holdSec = juce::jlimit (0.0f, kHoldMax, holdSec);
        const float atk = (float) std::exp (-1.0 / ((fast ? 0.00005 : 0.0005) * sr)), rel = (float) std::exp (-1.0 / (relSec * sr)), detRel = (float) std::exp (-1.0 / (0.010 * sr));
        const int holdSamples = (int) (holdSec * sr), chs = juce::jmin (2, buf.getNumChannels());
        auto* l = buf.getWritePointer (0); auto* r = chs > 1 ? buf.getWritePointer (1) : nullptr;
        float worst = 0.0f;
        for (int i = 0; i < n; ++i)
        {
            mix = on ? juce::jmin (1.0f, mix + mixStep) : juce::jmax (0.0f, mix - mixStep);
            const float xl = l[i], xr = r != nullptr ? r[i] : xl, a = juce::jmax (std::abs (xl), std::abs (xr));
            env = a > env ? a : a + (env - a) * detRel;
            const float levelDb = 20.0f * std::log10 (juce::jmax (env, 1.0e-6f));
            float target;
            if (expander) target = levelDb < thr ? juce::jmin (range, thr - levelDb) : 0.0f;
            else
            {
                if (levelDb > thr) { open = true; hold = holdSamples; }
                else if (open && levelDb < thr - 2.0f) { if (hold > 0) --hold; else open = false; }
                target = open ? 0.0f : range;
            }
            grDb = target + (grDb - target) * (target < grDb ? atk : rel);              // opening is the attack, closing is the release
            worst = juce::jmax (worst, grDb);
            const float g = juce::Decibels::decibelsToGain (-grDb), k = 1.0f + mix * (g - 1.0f);
            l[i] = xl * k; if (r != nullptr) r[i] = xr * k;
        }
        return worst;
    }
    float env = 0.0f, grDb = 0.0f, mix = 0.0f, mixStep = 0.01f; int hold = 0; bool open = false; double rate = 0.0;
};

// THE CHANNEL WIDTH (Mixer, Width): the stereo width of the channel's sound, 0 (mono: left and right the same) to 2 (twice as wide), 1 = unchanged. It works on the mid (left plus right) and the side (left minus right):
// the side is scaled, the mid is left alone, so a mono sound is never changed and the loudness of the centre holds still. At 1 it touches nothing (bit for bit); moving it is smoothed over 20 ms. It runs after the effects, before the pan.
struct WidthStage
{
    static constexpr float kMax = 2.0f;
    void process (juce::AudioBuffer<float>& buf, int n, float width, double sr)
    {
        if (rate != sr) { rate = sr; w.reset (sr, 0.02); w.setCurrentAndTargetValue (1.0f); }
        const float target = juce::jlimit (0.0f, kMax, width);
        if (target == 1.0f && w.getTargetValue() == 1.0f && ! w.isSmoothing()) return;                  // unchanged: leave the sound exactly as it is
        w.setTargetValue (target);
        if (buf.getNumChannels() < 2) return;
        auto* l = buf.getWritePointer (0); auto* r = buf.getWritePointer (1);
        for (int i = 0; i < n; ++i) { const float k = w.getNextValue(), m = (l[i] + r[i]) * 0.5f, s = (l[i] - r[i]) * 0.5f * k; l[i] = m + s; r[i] = m - s; }
    }
    juce::SmoothedValue<float> w { 1.0f }; double rate = 0.0;
};

struct TrackState
{
    int index = 0;
    juce::String name;
    // NOTE TRACE (a diagnostic, Help > Save Note Trace): the last 512 notes this track sent to its plug-in, with the song beat each one was placed at. Written by the audio thread, read only when asked.
    struct TraceEv { double beat = 0.0; juce::uint8 note = 0, vel = 0, on = 0; };
    TraceEv trace[512]; std::atomic<int> traceN { 0 };
    std::atomic<int> patWant { -1 };       // the pattern a pattern block asks for in the piece being played: -1 = the track has no pattern blocks (or the song is stopped), -2 = it has some but none is here (silence), 0 or more = that pattern
    std::atomic<double> patOrigin { 0.0 }; // where that block begins (the plug-in's pattern starts there)
    // INPUT ROUTING of an audio track (source = sound-card input): which input channel it listens to, mono (that one channel to both sides) or stereo (that channel and the next), whether you hear it while it comes in, and its input level
    std::atomic<int> inChan { 0 }; std::atomic<bool> inStereo { true }, monitor { true }; std::atomic<float> inPeak { 0.0f };
    std::atomic<int> grooveCh { -1 };    // the groove channel (1 to 31) this track's note clips play through, or -1 = none
    std::atomic<int> kind { 0 };         // the track's TYPE in the track view: 0 instrument track, 1 audio track (the sound is the same engine; the type decides the header, the default source and what 'add' offers)
    std::atomic<int> source { 0 };       // 0 off, 1 audio in, 2 sine, 3 noise, 4 impulse, 5 sweep, 6 audio file
    std::atomic<float> gainDb { 0.0f }, pan { 0.0f };
    std::atomic<int> patPlay { 1 }, patTrans { 0 };      // the step pattern (the note clips): how it plays (0 = off, 1 = loop, 2 = one shot) and how far it is moved up or down, in semitones; both can be automated
    std::atomic<float> inGainDb { 0.0f }; std::atomic<bool> inInvert { false };     // the channel input stage (gain in dB, polarity)
    InputStage inputStage;                                                         // the audio thread's smoothing state for it
    std::atomic<bool> eqHpOn { false }, eqLpOn { false }; std::atomic<float> eqHpHz { 80.0f }, eqLpHz { 12000.0f };      // the channel EQ: a high-pass and a low-pass filter
    EqPanel eq; ParaEq paraEq;                                                                                       // and the four parametric bands behind them, with the spectrum the EQ panel shows
    EqStage eqStage;
    std::atomic<bool> compOn { false }, compFast { false }, compPeak { true }; std::atomic<float> compThr { -18.0f }, compRatio { 4.0f }, compRel { 0.3f }, compMakeup { 0.0f };     // the channel compressor
    std::atomic<float> compGr { 0.0f };                                                    // the most gain reduction (dB) since the page last asked: the Mixer's meter
    CompStage compStage;
    std::atomic<float> width { 1.0f }; WidthStage widthStage;                              // the channel's stereo width (0 mono, 1 unchanged, 2 twice as wide)
    std::atomic<bool> gateOn { false }, gateExp { false }, gateFast { false }; std::atomic<float> gateThr { -45.0f }, gateRange { 40.0f }, gateRel { 0.3f }, gateHold { 0.05f };      // the channel gate
    std::atomic<float> gateGr { 0.0f };                                                    // the most gate reduction (dB) since the page last asked
    GateStage gateStage;
    std::atomic<double> autoTouchUntilMs { 0.0 };                  // while recording automation: a fader or pan the person is moving is not pulled back by its lane until this time
    Snapshot<AutoSet, 8> autoSnap { std::make_unique<AutoSet>() };      // automation lanes: published by the message thread, read by the audio thread
    const BlockInfo* blk = nullptr;                                      // what the current block is doing (set once by the engine)
    std::atomic<int> colour { -1 };      // the track's colour: a number in tools/studio-palette.json, or -1 for none
    std::atomic<int> autoMute { -1 }; std::atomic<bool> soloGate { false };      // autoMute: what a mute lane says right now (-1 = no lane, 0 = heard, 1 = muted), set by the audio thread; soloGate: another track is soloed and this one is not
    bool mutedNow() const { const int a = autoMute.load (std::memory_order_relaxed); return a >= 0 ? (a == 1 || soloGate.load (std::memory_order_relaxed)) : effMute.load (std::memory_order_relaxed); }
    std::atomic<bool> mute { false }, solo { false }, effMute { false }, armed { false }, recTarget { false };   // recTarget: armed, or (when none is armed) the selected track
    std::atomic<float> peakL { 0.0f }, peakR { 0.0f };
    // sends: level in dB (-100 = off, no node exists), and whether the send taps BEFORE the fader (pre) or after it (post)
    std::atomic<float> sendDb[8];                      // the level the send really has: its own level while it is ON, else -100 (kept for older readers)
    std::atomic<float> sendLvl[8];                     // the send's own level (what its knob shows), kept while the send is off
    std::atomic<bool> sendOn[8];                       // the send's ON switch: nothing reaches its FX return unless it is on
    std::atomic<bool> sendPre[8];
    std::atomic<double> autoTouchSendMs[8] {};          // a send knob being moved while recording: its own lane stands back (until 0 / a time), the other lanes of the track are not touched
    std::unique_ptr<SendProcessor> sendProc[8];        // exists only while that send is ON
    TrackState() { for (int i = 0; i < 8; ++i) { sendDb[i].store (-100.0f); sendLvl[i].store (-60.0f); sendOn[i].store (false); sendPre[i].store (false); } }
    MidiFifo liveMidi;                                   // live notes (keys, pages, hardware): pushed by any thread, popped by the audio thread
    std::atomic<bool> killNotes { false };               // Panic: the audio thread sends an explicit note-off for every note on every channel before its next block (a plug-in that ignores all-notes-off and reset still lets go)
    Snapshot<Arrangement, 16> arrSnap { emptyArrangement() };   // the clip instances: published by the message thread, read by the audio thread without waiting
    // the first instance's content: "the pattern" of the single-pattern pages (a copy; message thread)
    Clip primaryClip() const { const auto* a = arrSnap.load(); if (a->insts.empty() || a->insts[0].content == nullptr) { Clip c; c.lengthBeats = 0.0; return c; } return *a->insts[0].content; }
    Snapshot<FileAudio, 3> fileSnap { std::make_unique<FileAudio>() };   // the audio file (a few versions only: they are big)
    juce::String filePath;
    std::atomic<AudioRec*> audioRec { nullptr };         // set once, by the message thread, when the track is first armed
    std::unique_ptr<AudioRec> audioRecOwner;
    juce::AbstractFifo recFifo { 1024 };
    RecEvent recBuf[1024];
    std::vector<RecEvent> pending;                       // message thread only
    std::vector<std::unique_ptr<Slot>> slots;            // message thread only
    std::unique_ptr<TrackSourceProcessor> sourceProc;  // makes the track's MIDI and its test-signal / file / input audio
    std::unique_ptr<TrackStripProcessor> stripProc;     // fader, pan, mute, meter
    ~TrackState();
};

//==============================================================================
template <typename Base = juce::AudioProcessor>
class SimpleProc : public Base
{
public:
    using Base::Base;
    static typename Base::BusesProperties stereoInOut()
    {
        return typename Base::BusesProperties()
            .withInput ("In", juce::AudioChannelSet::stereo(), true)
            .withOutput ("Out", juce::AudioChannelSet::stereo(), true);
    }
    void releaseResources() override {}
    double getTailLengthSeconds() const override { return 0.0; }
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
};


//==============================================================================
class TrackSourceProcessor : public SimpleProc<>
{
public:
    TrackSourceProcessor (TrackState& t, const BlockInfo& b) : SimpleProc<> (stereoInOut()), ts (t), blk (b) {}
    // every note the CLIP has started and not yet ended, so Stop (or a jump, or a replaced pattern) can end each one with a real note-off: many plug-ins ignore the all-notes-off controller
    bool clipHeld[16][128] {}; signed char heldShift[16][128] {}; const Arrangement* lastArr = nullptr; int lastPlay = 1;      // (heldShift: how far each held note was moved up or down when it started, so its note-off finds it again even if the transpose moves meanwhile)
    const juce::String getName() const override { return "Track source"; }
    bool acceptsMidi() const override { return true; }
    bool producesMidi() const override { return true; }
    bool isBusesLayoutSupported (const BusesLayout& l) const override
    { return l.getMainInputChannelSet() == juce::AudioChannelSet::stereo() && l.getMainOutputChannelSet() == juce::AudioChannelSet::stereo(); }
    void prepareToPlay (double sr, int) override { rate = sr; phase = 0; sweepT = 0; impulseCount = 0; }

    void processBlock (juce::AudioBuffer<float>& buf, juce::MidiBuffer& midi) override
    {
        const int n = buf.getNumSamples();
        midi.clear();

        // ---- MIDI: live (keys, pages, hardware) ----
        // Live notes are delivered at the EARLIEST sample they can be: one that arrived before this block started goes at sample 0 (never held back for a timestamp), and one that arrived
        // while the block was being prepared keeps its small offset, so the order and spacing of a chord or a quick run are kept.
        LiveMsg lm;
        while (ts.liveMidi.pop (lm))
        {
            const int pos = lm.timeMs >= blk.blockStartMs ? juce::jlimit (0, n - 1, (int) ((lm.timeMs - blk.blockStartMs) * 0.001 * blk.sr)) : 0;
            const juce::MidiMessage m (lm.b, (int) lm.size);
            midi.addEvent (m, pos);
            if (blk.recording && blk.playing && ts.recTarget.load() && (m.isNoteOn() || m.isNoteOff()))
            {
                double beat = blk.segs[0].startBeat;
                for (int s = 0; s < blk.numSegs; ++s) if (pos >= blk.segs[s].startSample) beat = blk.segs[s].beatOfSample (pos);
                int s1, z1, s2, z2;
                ts.recFifo.prepareToWrite (1, s1, z1, s2, z2);
                if (z1 > 0 && beat >= 0.0) { auto& r = ts.recBuf[s1]; r.beat = beat; r.b[0] = m.getRawData()[0]; r.b[1] = m.getRawData()[1]; r.b[2] = m.getRawData()[2]; ts.recFifo.finishedWrite (1); }
            }
        }
        if (ts.killNotes.exchange (false))        // PANIC: a plug-in is free to ignore all-sound-off (CC120), all-notes-off (CC123) and reset(), and some do: a plain note-off for every note is the one message every instrument understands
            for (int ch = 1; ch <= 16; ++ch) for (int nt = 0; nt < 128; ++nt) midi.addEvent (juce::MidiMessage::noteOff (ch, nt), 0);
        const Arrangement* arr = ts.arrSnap.load();
        if (blk.playing && blk.numSegs > 0)      // the step pattern's mode and transpose can be automated (lanes 5 and 6)
        {
            const auto* as = ts.autoSnap.load(); const double abeat = blk.segs[0].startBeat;
            if (as != nullptr) for (const auto& lane : as->lanes) if (lane.on && ! lane.pts.empty()) { if (lane.kind == 5) ts.patTrans.store (juce::jlimit (-24, 24, (int) std::lround (autoValueAt (lane, abeat)))); else if (lane.kind == 6) ts.patPlay.store (juce::jlimit (0, 2, (int) std::lround (autoValueAt (lane, abeat)))); }
        }
        {   // a MUTE lane (lane 7) mutes the track along the song; a SOLO lane (8) is followed from the 30 Hz tick (it changes what the other tracks hear)
            int am = -1; const auto* as7 = ts.autoSnap.load();
            if (blk.playing && blk.numSegs > 0 && as7 != nullptr) for (const auto& lane : as7->lanes) if (lane.kind == 7 && lane.on && ! lane.pts.empty()) am = autoValueAt (lane, blk.segs[0].startBeat) >= 0.5 ? 1 : 0;
            ts.autoMute.store (am, std::memory_order_relaxed);
        }
        const int play = ts.patPlay.load(), trans = ts.patTrans.load();
        if (blk.panic || arr != lastArr || (play == 0 && lastPlay != 0))      // stopped, jumped, or the arrangement was edited: end every note the clips had started
        {
            for (int ch = 0; ch < 16; ++ch) for (int nt = 0; nt < 128; ++nt) if (clipHeld[ch][nt]) { midi.addEvent (juce::MidiMessage::noteOff (ch + 1, juce::jlimit (0, 127, nt + (int) heldShift[ch][nt])), 0); clipHeld[ch][nt] = false; }
            if (blk.panic) for (int ch = 1; ch <= 16; ++ch) midi.addEvent (juce::MidiMessage::allNotesOff (ch), 0);      // stopping (or jumping) also tells the plug-in 'all notes off', so a note that was started some other way cannot hang
            lastArr = arr;
        }

        // ---- MIDI: the clip instances ----
        lastPlay = play;
        if (blk.playing && play != 0)
            for (int s = 0; s < blk.numSegs; ++s)
            {
                const auto& sg = blk.segs[s];
                if (sg.cut) for (int ch = 0; ch < 16; ++ch) for (int nt = 0; nt < 128; ++nt) if (clipHeld[ch][nt]) { midi.addEvent (juce::MidiMessage::noteOff (ch + 1, juce::jlimit (0, 127, nt + (int) heldShift[ch][nt])), sg.startSample); clipHeld[ch][nt] = false; }      // a hot cue jump: end the notes from before it
                const double b0 = sg.startBeat, b1 = sg.endBeat;
                auto at = [&] (double t) { return sg.startSample + juce::jlimit (0, sg.numSamples - 1, (int) std::lround (sg.sampleOfBeat (t) - (double) sg.startSample)); };
                for (const auto& in : arr->insts)
                {
                    GrooveEval ge; const GrooveEval* gp = nullptr;
                    { const int gi = in.groove == -2 ? ts.grooveCh.load() : in.groove; if (blk.groove != nullptr && gi >= 1 && gi < kGrooveChans) { ge = GrooveEval::make (*blk.groove, gi, in.id); if (ge.active()) gp = &ge; } }
                    walkInstance (in, b0, b1, blk.chase && s == 0,
                        [&] (const NoteEv& nt, double t) { heldShift[juce::jlimit (1, 16, nt.ch) - 1][juce::jlimit (0, 127, nt.note)] = (signed char) trans; midi.addEvent (juce::MidiMessage::noteOn (nt.ch, juce::jlimit (0, 127, nt.note + trans), (juce::uint8) nt.vel), at (t)); clipHeld[juce::jlimit (1, 16, nt.ch) - 1][juce::jlimit (0, 127, nt.note)] = true; },
                        [&] (const NoteEv& nt, double t) { midi.addEvent (juce::MidiMessage::noteOff (nt.ch, juce::jlimit (0, 127, nt.note + (int) heldShift[juce::jlimit (1, 16, nt.ch) - 1][juce::jlimit (0, 127, nt.note)])), at (t)); clipHeld[juce::jlimit (1, 16, nt.ch) - 1][juce::jlimit (0, 127, nt.note)] = false; },
                        // NOTE CHASE: starting (or jumping) in the middle of a note starts that note at once, so a sustained note is not missing
                        [&] (const NoteEv& nt) { heldShift[juce::jlimit (1, 16, nt.ch) - 1][juce::jlimit (0, 127, nt.note)] = (signed char) trans; midi.addEvent (juce::MidiMessage::noteOn (nt.ch, juce::jlimit (0, 127, nt.note + trans), (juce::uint8) nt.vel), sg.startSample); clipHeld[juce::jlimit (1, 16, nt.ch) - 1][juce::jlimit (0, 127, nt.note)] = true; },
                        blk.swingDelay, blk.swingGrid, gp, play == 2);
                }
            }

        {   // PATTERN BLOCKS: the pattern wanted for this piece of the song (the block that covers its last instant, so a block edge inside the piece gives the new pattern to the piece that holds the step ON the edge)
            int want = -1; double origin = 0.0;
            if (blk.playing && blk.numSegs > 0)
            {
                const double bb = blk.segs[blk.numSegs - 1].endBeat - 1.0e-9; bool any = false;
                for (const auto& in : arr->insts) if (in.content != nullptr && in.content->patternBank >= 0) { any = true; if (! in.muted && in.start <= bb && (in.length <= 0.0 || bb < in.start + in.length)) { want = in.content->patternBank; origin = in.start; } }
                if (any && want < 0) want = -2;
            }
            ts.patOrigin.store (origin, std::memory_order_relaxed); ts.patWant.store (want);
        }
        for (const auto meta : midi)       // the note trace: what is about to reach the plug-in, and at which beat
        {
            const auto m = meta.getMessage(); if (! (m.isNoteOn() || m.isNoteOff()) || (blk.panic && m.isNoteOff())) continue;
            const int pos = meta.samplePosition; double beat = blk.segs[0].startBeat;
            for (int s = 0; s < blk.numSegs; ++s) if (pos >= blk.segs[s].startSample) beat = blk.segs[s].beatOfSample (pos);
            const int i = ts.traceN.load(); auto& e = ts.trace[i & 511]; e.beat = beat; e.note = (juce::uint8) m.getNoteNumber(); e.vel = (juce::uint8) m.getVelocity(); e.on = m.isNoteOn() ? 1 : 0; ts.traceN.store (i + 1);
        }
        // ---- audio ----
        const int mode = ts.source.load();   // (audio CLIPS are mixed in after this, whatever the source setting)
        auto* l = buf.getWritePointer (0);
        auto* r = buf.getWritePointer (1);
        switch (mode)
        {
            case 1: break;   // sound-card input already in the buffer
            case 2: for (int i = 0; i < n; ++i) { const float v = 0.25f * std::sin ((float) phase); l[i] = r[i] = v; phase += 2.0 * juce::MathConstants<double>::pi * 440.0 / rate; if (phase > 6.2831853) phase -= 6.2831853; } break;
            case 3: for (int i = 0; i < n; ++i) { const float v = 0.15f * (rnd.nextFloat() * 2.0f - 1.0f); l[i] = r[i] = v; } break;
            case 4: for (int i = 0; i < n; ++i) { const float v = (impulseCount == 0) ? 0.8f : 0.0f; l[i] = r[i] = v; if (++impulseCount >= (long long) rate) impulseCount = 0; } break;
            case 5: for (int i = 0; i < n; ++i)
                    {
                        const double f = 20.0 * std::pow (1000.0, sweepT / 10.0);
                        const float v = 0.25f * std::sin ((float) phase);
                        l[i] = r[i] = v;
                        phase += 2.0 * juce::MathConstants<double>::pi * f / rate; if (phase > 6.2831853) phase -= 6.2831853;
                        sweepT += 1.0 / rate; if (sweepT >= 10.0) sweepT = 0.0;
                    } break;
            case 6:
            {
                buf.clear();
                const FileAudio* fa = ts.fileSnap.load();
                if (fa->valid && blk.playing)
                    for (int s = 0; s < blk.numSegs; ++s)
                    {
                        const auto& sg = blk.segs[s];
                        const long long start = (long long) std::llround (sg.startSec * blk.sr);        // audio follows real time, not the tempo
                        for (int i = 0; i < sg.numSamples; ++i)
                        {
                            const long long p = start + i;
                            if (p >= 0 && p < fa->buf.getNumSamples())
                            {
                                l[sg.startSample + i] = fa->buf.getSample (0, (int) p);
                                r[sg.startSample + i] = fa->buf.getSample (fa->buf.getNumChannels() > 1 ? 1 : 0, (int) p);
                            }
                        }
                    }
                break;
            }
            default: buf.clear(); break;
        }
        // THE SOUND-CARD INPUT, before anything else is mixed in: its level, recording it (the dry signal, so a take never records the clips that are playing back), and monitoring (not hearing it while it comes in)
        if (mode == 1) { float pk = 0.0f; for (int i = 0; i < n; ++i) pk = juce::jmax (pk, std::abs (l[i]), std::abs (r[i])); ts.inPeak.store (juce::jmax (ts.inPeak.load(), pk)); }
        if (mode == 1 && blk.recording && blk.playing && ts.armed.load())
            if (auto* rec = ts.audioRec.load())
                for (int s = 0; s < blk.numSegs; ++s)
                {
                    const auto& sg = blk.segs[s]; if (sg.numSamples <= 0 || sg.startBeat < 0.0) continue;       // not during the count-in
                    int s1, z1, s2, z2; rec->fifo.prepareToWrite (sg.numSamples, s1, z1, s2, z2);
                    if (z1 + z2 < sg.numSamples) { rec->overflow.store (true); continue; }
                    if (rec->startSec.load() < 0.0) rec->startSec.store (sg.startSec);
                    const float* src[2] = { buf.getReadPointer (0), buf.getReadPointer (1) };
                    for (int c = 0; c < 2; ++c) { std::memcpy (rec->ring[c].data() + s1, src[c] + sg.startSample, (size_t) z1 * sizeof (float)); if (z2 > 0) std::memcpy (rec->ring[c].data() + s2, src[c] + sg.startSample + z1, (size_t) z2 * sizeof (float)); }
                    rec->fifo.finishedWrite (z1 + z2);
                }
        if (mode == 1 && ! ts.monitor.load()) buf.clear();
        mixAudioClips (buf, arr);
    }

    // Audio clips: each instance plays its sound from where the timeline is, for as long as the instance lasts, looping the sound when the instance is longer than it. The sound follows REAL time
    // (it is not stretched to the tempo); where an instance sits on the timeline follows the tempo map. A 2 ms fade at each end of an instance keeps it from clicking.
    void mixAudioClips (juce::AudioBuffer<float>& buf, const Arrangement* arr)
    {
        if (! blk.playing) return;
        auto* l = buf.getWritePointer (0); auto* r = buf.getWritePointer (1);
        for (int s = 0; s < blk.numSegs; ++s)
        {
            const auto& sg = blk.segs[s];
            if (sg.map == nullptr || sg.numSamples <= 0) continue;
            const double blockEnd = sg.startSec + (double) sg.numSamples / sg.sr;
            for (const auto& in : arr->insts)
            {
                if (in.muted) continue;
                const AudioData* A = in.content != nullptr ? in.content->audio.get() : nullptr;
                if (in.pitched != nullptr && ! (in.warp && in.srcBpm > 0.0 && in.length > 0.0)) A = in.pitched.get();          // the same sound, moved in pitch (same length, so the offset, loop and reverse below are unchanged)
                const bool isW = in.warped != nullptr && in.warp;                   // the stretched version: already laid out for this clip, so it plays from its own start, forwards
                if (isW) A = in.warped.get();
                if (A == nullptr || A->numSamples() < 2) continue;
                const double ws = sg.map->secondsAt (in.start), we = in.length > 0.0 ? sg.map->secondsAt (in.start + in.length) : 1.0e30, len = A->durationSec();
                const float gain = std::pow (10.0f, in.gainDb / 20.0f); const double fIn = juce::jmax (0.0, in.fadeIn), fOut = juce::jmax (0.0, in.fadeOut);
                if (we <= sg.startSec || ws >= blockEnd) continue;
                const int nCh = A->buf.getNumChannels(); const float* L = A->buf.getReadPointer (0); const float* R = A->buf.getReadPointer (nCh > 1 ? 1 : 0); const int N = A->numSamples();
                for (int i = 0; i < sg.numSamples; ++i)
                {
                    const double sec = sg.startSec + (double) i / sg.sr;
                    if (sec < ws || sec >= we) continue;
                    const double t = sec - ws;
                    double pos = (isW ? 0.0 : in.offset) + t; if (pos >= len) pos = std::fmod (pos, len);
                    if (in.reverse && ! isW) pos = juce::jmax (0.0, len - pos - 1.0 / A->rate);                  // the same stretch of sound, played backwards
                    const double src = pos * A->rate; int i0 = (int) src; const float fr = (float) (src - (double) i0);
                    if (i0 >= N) i0 = N - 1; const int i1 = i0 + 1 < N ? i0 + 1 : i0;
                    const float g = gain * (float) juce::jmin (fIn > 0.0 ? juce::jmin (1.0, t / fIn) : 1.0, fOut > 0.0 ? juce::jmin (1.0, (we - sec) / fOut) : 1.0);
                    l[sg.startSample + i] += g * (L[i0] + fr * (L[i1] - L[i0]));
                    r[sg.startSample + i] += g * (R[i0] + fr * (R[i1] - R[i0]));
                }
            }
        }
    }

private:
    TrackState& ts; const BlockInfo& blk;
    double rate = 48000.0, phase = 0.0, sweepT = 0.0; long long impulseCount = 0;
    juce::Random rnd;
};

class TrackStripProcessor : public SimpleProc<>
{
public:
    TrackStripProcessor (TrackState& t) : SimpleProc<> (stereoInOut()), ts (t) {}
    const juce::String getName() const override { return "Track strip"; }
    bool isBusesLayoutSupported (const BusesLayout& l) const override { return l.getMainInputChannelSet() == juce::AudioChannelSet::stereo() && l.getMainOutputChannelSet() == juce::AudioChannelSet::stereo(); }
    void prepareToPlay (double sr, int) override { gain.reset (sr, 0.02); pl.reset (sr, 0.02); pr.reset (sr, 0.02); gain.setCurrentAndTargetValue (target()); }
    void processBlock (juce::AudioBuffer<float>& buf, juce::MidiBuffer&) override
    {
        if (ts.blk != nullptr && ts.blk->playing && ts.blk->numSegs > 0)             // automation: read at the start of the block
        {
            const auto* as = ts.autoSnap.load(); const double beat = ts.blk->segs[0].startBeat;
            const bool touching = ts.blk->blockStartMs < ts.autoTouchUntilMs.load();
            for (const auto& lane : as->lanes)
            {
                if (lane.kind == 4 && lane.on && ! lane.pts.empty() && lane.slot >= 0 && lane.slot < 8 && ts.blk->blockStartMs >= ts.autoTouchSendMs[lane.slot].load()) { ts.sendLvl[lane.slot].store ((float) juce::jlimit (-60.0, 6.0, autoValueAt (lane, beat))); continue; }       // a send's level follows its lane
                if (! lane.on || lane.pts.empty() || touching) continue;
                if (lane.kind == 0) ts.gainDb.store ((float) juce::jlimit (-90.0, 12.0, autoValueAt (lane, beat)));
                else if (lane.kind == 1) ts.pan.store ((float) juce::jlimit (-1.0, 1.0, autoValueAt (lane, beat)));
            }
        }
        gain.setTargetValue (target());
        const float pan = ts.pan.load();
        pl.setTargetValue (pan <= 0.0f ? 1.0f : std::cos (pan * juce::MathConstants<float>::halfPi));
        pr.setTargetValue (pan >= 0.0f ? 1.0f : std::cos (-pan * juce::MathConstants<float>::halfPi));
        const int n = buf.getNumSamples();
        auto* l = buf.getWritePointer (0); auto* r = buf.getWritePointer (1);
        float peakL = 0, peakR = 0;
        for (int i = 0; i < n; ++i)
        {
            const float g = gain.getNextValue();
            l[i] *= g * pl.getNextValue(); r[i] *= g * pr.getNextValue();
            peakL = juce::jmax (peakL, std::abs (l[i])); peakR = juce::jmax (peakR, std::abs (r[i]));
        }
        if (peakL > ts.peakL.load()) ts.peakL.store (peakL);
        if (peakR > ts.peakR.load()) ts.peakR.store (peakR);
    }
private:
    float target() const { return ts.mutedNow() ? 0.0f : juce::Decibels::decibelsToGain (ts.gainDb.load(), -90.0f); }
    TrackState& ts;
    juce::SmoothedValue<float> gain { 1.0f }, pl { 1.0f }, pr { 1.0f };
};

// a send: the track's signal (taken before or after the fader, by how it is connected) scaled by the send level; muted with the track
class SendProcessor : public SimpleProc<>
{
public:
    SendProcessor (TrackState& t, int idx) : SimpleProc<> (stereoInOut()), ts (t), i (idx) {}
    const juce::String getName() const override { return "Send"; }
    bool isBusesLayoutSupported (const BusesLayout& l) const override { return l.getMainInputChannelSet() == juce::AudioChannelSet::stereo() && l.getMainOutputChannelSet() == juce::AudioChannelSet::stereo(); }
    void prepareToPlay (double sr, int) override { g.reset (sr, 0.02); g.setCurrentAndTargetValue (target()); }
    void processBlock (juce::AudioBuffer<float>& buf, juce::MidiBuffer&) override
    {
        g.setTargetValue (target());
        const int n = buf.getNumSamples(); auto* l = buf.getWritePointer (0); auto* r = buf.getWritePointer (1);
        for (int k = 0; k < n; ++k) { const float v = g.getNextValue(); l[k] *= v; r[k] *= v; }
    }
private:
    float target() const { if (ts.mutedNow() || ! ts.sendOn[i].load()) return 0.0f; const float db = ts.sendLvl[i].load(); return db <= -59.5f ? 0.0f : juce::Decibels::decibelsToGain (db); }
    TrackState& ts; int i; juce::SmoothedValue<float> g { 0.0f };
};

// a junction: whatever arrives (summed by the graph) leaves unchanged. Used for each FX return's input and for the master section's input.
class PassProcessor : public SimpleProc<>
{
public:
    PassProcessor() : SimpleProc<> (stereoInOut()) {}
    const juce::String getName() const override { return "Bus"; }
    bool isBusesLayoutSupported (const BusesLayout& l) const override { return l.getMainInputChannelSet() == juce::AudioChannelSet::stereo() && l.getMainOutputChannelSet() == juce::AudioChannelSet::stereo(); }
    void prepareToPlay (double, int) override {}
    void processBlock (juce::AudioBuffer<float>&, juce::MidiBuffer&) override {}
};

// THE LOUDNESS METER: loudness in LUFS as streaming services measure it (ITU-R BS.1770 / EBU R128): the sound is K-weighted (a high shelf and a high-pass that make it match how loud we hear it), its mean square is taken
// over 100 ms blocks, and the loudness is -0.691 + 10 log10 of that. MOMENTARY is the last 400 ms, SHORT-TERM the last 3 seconds, INTEGRATED the whole song so far with the two gates (quieter than -70 LUFS is ignored, then
// anything more than 10 LU under the average of the rest is ignored). The audio thread feeds it; it holds no lock and allocates nothing.
struct LoudnessMeter
{
    struct Bd { double b0 = 1, b1 = 0, b2 = 0, a1 = 0, a2 = 0, z1[2] { 0, 0 }, z2[2] { 0, 0 }; double run (int c, double x) { const double y = b0 * x + z1[c]; z1[c] = b1 * x - a1 * y + z2[c]; z2[c] = b2 * x - a2 * y; return y; } };
    static constexpr int kBins = 800, kRing = 30;      // integrated: 0.1 LU bins from -70 to +10 LUFS; the ring holds 3 seconds of 100 ms blocks
    static double loud (double ms) { return -0.691 + 10.0 * std::log10 (juce::jmax (ms, 1.0e-10)); }
    void prepare (double sr)
    {
        const double pi = juce::MathConstants<double>::pi; blockN = juce::jmax (1, (int) std::lround (sr / 10.0));
        { const double G = 3.999843853973347, f0 = 1681.974450955533, Q = 0.7071752369554196, K = std::tan (pi * f0 / sr), Vh = std::pow (10.0, G / 20.0), Vb = std::pow (Vh, 0.4996667741545416), a0 = 1.0 + K / Q + K * K;
          s1.b0 = (Vh + Vb * K / Q + K * K) / a0; s1.b1 = 2.0 * (K * K - Vh) / a0; s1.b2 = (Vh - Vb * K / Q + K * K) / a0; s1.a1 = 2.0 * (K * K - 1.0) / a0; s1.a2 = (1.0 - K / Q + K * K) / a0; }
        { const double f0 = 38.13547087602444, Q = 0.5003270373238773, K = std::tan (pi * f0 / sr), a0 = 1.0 + K / Q + K * K; s2.b0 = 1.0; s2.b1 = -2.0; s2.b2 = 1.0; s2.a1 = 2.0 * (K * K - 1.0) / a0; s2.a2 = (1.0 - K / Q + K * K) / a0; }
        clearAll();
    }
    void resetIntegrated() { resetReq.store (true); }
    // feed one block of the finished mix; returns true when a new 100 ms block has just been measured (the caller can then act on the loudness)
    bool process (const float* l, const float* r, int n)
    {
        if (resetReq.exchange (false)) clearIntegrated();
        bool fresh = false;
        for (int i = 0; i < n; ++i)
        {
            const double a = s2.run (0, s1.run (0, (double) l[i])), b = s2.run (1, s1.run (1, (double) r[i])); acc += a * a + b * b;
            if (++cnt >= blockN)
            {
                ring[(size_t) rp] = acc / (double) cnt; rp = (rp + 1) % kRing; if (have < kRing) ++have; acc = 0.0; cnt = 0; fresh = true;
                auto meanOf = [&] (int k) { k = juce::jmin (k, have); double s = 0.0; for (int j = 1; j <= k; ++j) s += ring[(size_t) ((rp - j + 2 * kRing) % kRing)]; return s / (double) juce::jmax (1, k); };
                const double m4 = meanOf (4); momentary.store ((float) loud (m4), std::memory_order_relaxed); shortTerm.store ((float) loud (meanOf (kRing)), std::memory_order_relaxed);
                if (have >= 4)
                {
                    const double lk = loud (m4);
                    if (lk > -70.0) { const int bin = juce::jlimit (0, kBins - 1, (int) std::floor ((lk + 70.0) * 10.0)); binN[(size_t) bin] += 1.0; binE[(size_t) bin] += m4; totN += 1.0; totE += m4; }
                    if (totN > 0.0)
                    {
                        const double gate = loud (totE / totN) - 10.0; double sn = 0.0, se = 0.0; const int from = juce::jlimit (0, kBins, (int) std::ceil ((gate + 70.0) * 10.0));
                        for (int k = from; k < kBins; ++k) { sn += binN[(size_t) k]; se += binE[(size_t) k]; }
                        integrated.store (sn > 0.0 ? (float) loud (se / sn) : -100.0f, std::memory_order_relaxed);
                    }
                }
            }
        }
        return fresh;
    }
    std::atomic<float> momentary { -100.0f }, shortTerm { -100.0f }, integrated { -100.0f };
private:
    void clearIntegrated() { for (auto& x : binN) x = 0.0; for (auto& x : binE) x = 0.0; totN = totE = 0.0; integrated.store (-100.0f, std::memory_order_relaxed); }
    void clearAll() { s1.z1[0] = s1.z1[1] = s1.z2[0] = s1.z2[1] = s2.z1[0] = s2.z1[1] = s2.z2[0] = s2.z2[1] = 0.0; for (auto& x : ring) x = 0.0; rp = have = cnt = 0; acc = 0.0; clearIntegrated(); momentary.store (-100.0f); shortTerm.store (-100.0f); }
    Bd s1, s2; int blockN = 4800, cnt = 0, rp = 0, have = 0; double acc = 0.0, totN = 0.0, totE = 0.0; std::array<double, kRing> ring {}; std::array<double, kBins> binN {}, binE {}; std::atomic<bool> resetReq { false };
};

// THE ARCHIE SOUND (the end of the master section): the program's own sound, and the boundary the mix has to stay inside. Two stages after the master level:
//   DRIVE: a soft saturation that rounds the peaks of drums and bass and adds a little even and odd harmonic (the push of a hot console channel), run at twice the sample rate so it does not alias. At 0 it does nothing.
//   LIMITER: a look-ahead limiter (1.5 ms, release 80 ms, both channels linked) that holds the output at or under the CEILING, so pushing the master fader or the drive makes the mix denser, never over.
// With it off, the master is exactly as before. The master level is the input to it: raise the master fader and the mix is pushed into the drive and the limiter. It adds 1.5 ms of delay to the master only.
struct ArchieSound
{
    static constexpr float kCeilMin = -3.0f, kCeilMax = 0.0f;
    void prepare (double sr, int maxBlock)
    {
        rate = sr; L = juce::jmax (8, (int) std::lround (0.0015 * sr)); maxN = juce::jmax (256, maxBlock);
        os = std::make_unique<juce::dsp::Oversampling<float>> (2, 1, juce::dsp::Oversampling<float>::filterHalfBandPolyphaseIIR, false, false); os->initProcessing ((size_t) maxN); os->reset();
        dl[0].assign ((size_t) L, 0.0f); dl[1].assign ((size_t) L, 0.0f); need.assign ((size_t) L, 1.0f); hist.assign ((size_t) L, 1.0f); pos = 0; sum = (double) L; rg = 1.0f; dcX[0] = dcX[1] = dcY[0] = dcY[1] = 0.0f; drvNow = 0.0f; pushNow = 1.0f; bassNow = toneNow = 0.0f; bassSet = -1.0f; toneSet = -99.0f; bass.clear(); tone.clear();
        relCoef = (float) (1.0 - std::exp (-1.0 / (0.080 * sr)));
        { const double w0 = juce::MathConstants<double>::twoPi * 20.0 / sr, c = std::cos (w0), s = std::sin (w0); const double qs[2] = { 0.5411961, 1.3065630 };      // the two sections of a 4th-order Butterworth
          for (int k = 0; k < 2; ++k) { const double al = s / (2.0 * qs[k]), a0 = 1.0 + al; auto& f = lc[k]; f.b0 = (1.0 + c) * 0.5 / a0; f.b1 = -(1.0 + c) / a0; f.b2 = f.b0; f.a1 = -2.0 * c / a0; f.a2 = (1.0 - al) / a0; f.z1[0] = f.z1[1] = f.z2[0] = f.z2[1] = 0.0; } }
    }
    // returns the biggest gain reduction of the limiter in this block, in dB (positive)
    float process (juce::AudioBuffer<float>& buf, int n, float pushDb, float bassDb, float toneDb, float drive, float ceilDb, bool cut = true)
    {
        if (os == nullptr || buf.getNumChannels() < 2) return 0.0f;
        if (cut) lowCut (buf, n); else lcOff = true;
        pushStage (buf, n, pushDb, bassDb, toneDb);
        drive = juce::jlimit (0.0f, 1.0f, drive); const float ceilLin = juce::Decibels::decibelsToGain (juce::jlimit (kCeilMin, kCeilMax, ceilDb));
        for (int off = 0; off < n; off += maxN)
        {
            const int m = juce::jmin (maxN, n - off);
            if (drive > 0.0f || drvNow > 0.0f) saturate (buf, off, m, drive);
        }
        float worst = 0.0f; auto* l = buf.getWritePointer (0); auto* r = buf.getWritePointer (1);
        for (int i = 0; i < n; ++i)
        {
            const float pk = juce::jmax (std::abs (l[i]), std::abs (r[i])), nd = pk > ceilLin ? ceilLin / pk : 1.0f;
            need[(size_t) pos] = nd; float gmin = 1.0f; for (int k = 0; k < L; ++k) gmin = juce::jmin (gmin, need[(size_t) k]);      // the lowest gain asked for by any sample in the last 1.5 ms
            const float gain = (float) (sum / (double) L);                                                                  // the average of the last L smoothed gains: it has reached the lowest gain asked for by the time the sample leaves the delay
            rg = gmin < rg ? gmin : rg + (gmin - rg) * relCoef; sum += (double) rg - (double) hist[(size_t) pos]; hist[(size_t) pos] = rg;
            const float xl = dl[0][(size_t) pos], xr = dl[1][(size_t) pos]; dl[0][(size_t) pos] = l[i]; dl[1][(size_t) pos] = r[i];
            l[i] = juce::jlimit (-ceilLin, ceilLin, xl * gain); r[i] = juce::jlimit (-ceilLin, ceilLin, xr * gain);
            worst = juce::jmax (worst, 1.0f - gain); if (++pos >= L) pos = 0;
        }
        return worst > 0.0f ? -20.0f * std::log10 (juce::jmax (1.0e-4f, 1.0f - worst)) : 0.0f;
    }
private:
    // THE 20 HZ CUT: a definite cutoff at the bottom of the audible range (a 4th-order Butterworth high-pass, 24 dB per octave), so nothing below 20 Hz takes headroom from the limiter or feeds the drive. It is part of the Archie sound being on.
    void lowCut (juce::AudioBuffer<float>& buf, int n)
    {
        auto* ch0 = buf.getWritePointer (0); auto* ch1 = buf.getWritePointer (1);
        for (int i = 0; i < n; ++i) for (int c = 0; c < 2; ++c) { float* p = c == 0 ? ch0 : ch1; double x = (double) p[i]; for (int s = 0; s < 2; ++s) { auto& f = lc[s]; const double y = f.b0 * x + f.z1[c]; f.z1[c] = f.b1 * x - f.a1 * y + f.z2[c]; f.z2[c] = f.b2 * x - f.a2 * y; x = y; } p[i] = (float) x; }
    }
    // PUSH is gain into the drive and the limiter (0 to 18 dB); BASS is a low shelf at 90 Hz (0 to 6 dB) and TONE a high shelf at 6 kHz (-3 to +3 dB), both before the drive so the drive works on them too
    void pushStage (juce::AudioBuffer<float>& buf, int n, float pushDb, float bassDb, float toneDb)
    {
        pushDb = juce::jlimit (0.0f, 18.0f, pushDb); bassDb = juce::jlimit (0.0f, 6.0f, bassDb); toneDb = juce::jlimit (-3.0f, 3.0f, toneDb);
        const float pg = juce::Decibels::decibelsToGain (pushDb); const bool bassOn = bassDb > 0.01f || bassNow > 0.01f, toneOn = std::abs (toneDb) > 0.01f || std::abs (toneNow) > 0.01f;
        if (! bassOn && ! toneOn && std::abs (pushNow - pg) < 1.0e-5f && std::abs (pg - 1.0f) < 1.0e-5f) { pushNow = 1.0f; return; }
        if (bassDb != bassSet) { bassSet = bassDb; bass.set (0, rate, 90.0, bassDb, 0.7); } if (toneDb != toneSet) { toneSet = toneDb; tone.set (2, rate, 6000.0, toneDb, 0.7); }
        auto* l = buf.getWritePointer (0); auto* r = buf.getWritePointer (1);
        for (int i = 0; i < n; ++i)
        {
            pushNow += (pg - pushNow) * 0.002f; bassNow += (bassDb - bassNow) * 0.002f; toneNow += (toneDb - toneNow) * 0.002f;
            float a = l[i] * pushNow, b = r[i] * pushNow;
            if (bassOn) { a = bass.run (0, a); b = bass.run (1, b); } if (toneOn) { a = tone.run (0, a); b = tone.run (1, b); }
            l[i] = a; r[i] = b;
        }
    }
    void saturate (juce::AudioBuffer<float>& buf, int off, int m, float drive)
    {
        float* ch[2] = { buf.getWritePointer (0) + off, buf.getWritePointer (1) + off }; juce::dsp::AudioBlock<float> blk (ch, 2, (size_t) m);
        auto up = os->processSamplesUp (blk); const size_t nu = up.getNumSamples();
        for (size_t i = 0; i < nu; ++i)
        {
            drvNow += (drive - drvNow) * 0.0005f;                                      // the drive moves smoothly (about 20 ms at the oversampled rate)
            const float k = 1.0f + 3.0f * drvNow, b = 0.12f * drvNow, tb = std::tanh (k * b), norm = k * (1.0f - tb * tb);
            for (size_t c = 0; c < 2; ++c) { float* d = up.getChannelPointer (c); const float s = (std::tanh (k * (d[i] + b)) - tb) / norm; d[i] += drvNow * (s - d[i]); }      // (the shaped sound is blended in by the drive amount, so a little drive is a little warmth)
        }
        os->processSamplesDown (blk);
        for (int c = 0; c < 2; ++c) for (int i = 0; i < m; ++i) { const float x = ch[c][i], y = x - dcX[c] + 0.9995f * dcY[c]; dcX[c] = x; dcY[c] = y; ch[c][i] = y; }      // the asymmetry leaves a little DC: take it out
    }
    double rate = 48000.0; int L = 72, maxN = 4096, pos = 0; double sum = 72.0; float rg = 1.0f, relCoef = 0.0003f, drvNow = 0.0f, dcX[2] { 0, 0 }, dcY[2] { 0, 0 }, pushNow = 1.0f, bassNow = 0.0f, toneNow = 0.0f, bassSet = -1.0f, toneSet = -99.0f; ParaEq::Bq bass, tone;
    bool lcOff = false;
    struct Hp2 { double b0 = 1, b1 = 0, b2 = 0, a1 = 0, a2 = 0, z1[2] { 0, 0 }, z2[2] { 0, 0 }; } lc[2];
    std::unique_ptr<juce::dsp::Oversampling<float>> os; std::vector<float> dl[2], need, hist;
};

class MasterProcessor : public SimpleProc<>
{
public:
    MasterProcessor() : SimpleProc<> (stereoInOut()) {}
    const juce::String getName() const override { return "Master"; }
    bool isBusesLayoutSupported (const BusesLayout& l) const override { return l.getMainInputChannelSet() == juce::AudioChannelSet::stereo() && l.getMainOutputChannelSet() == juce::AudioChannelSet::stereo(); }
    void prepareToPlay (double sr, int bs) override { rate = sr; gain.reset (sr, 0.02); gain.setCurrentAndTargetValue (juce::Decibels::decibelsToGain (gainDb.load(), -90.0f)); sound.prepare (sr, juce::jmax (bs, 2048)); lm.prepare (sr); avgMs = 0.0; }
    void processBlock (juce::AudioBuffer<float>& buf, juce::MidiBuffer&) override
    {
        if (autoFrom != nullptr && autoFrom->blk != nullptr && autoFrom->blk->playing && autoFrom->blk->numSegs > 0) { const auto* eas = autoFrom->autoSnap.load(); if (eas != nullptr) for (const auto& lane : eas->lanes) if (lane.kind == kEqLaneKind && lane.on && ! lane.pts.empty()) applyEqLane (eq, eqHpOn, eqHpHz, eqLpOn, eqLpHz, lane.param, autoValueAt (lane, autoFrom->blk->segs[0].startBeat)); }      // the master EQ lanes
        eqStage.process (buf, buf.getNumSamples(), eqHpOn.load(), eqHpHz.load(), eqLpOn.load(), eqLpHz.load(), rate); paraEq.process (buf, buf.getNumSamples(), eq, rate); eq.feed (buf, buf.getNumSamples());      // the master EQ comes first, then the compressor
        { const float gr = comp.process (buf, buf.getNumSamples(), cOn.load(), cThr.load(), cRatio.load(), cRel.load(), false, cPeak.load(), cMake.load(), rate, cAtk.load()); if (gr > cGr.load (std::memory_order_relaxed)) cGr.store (gr, std::memory_order_relaxed); }       // the master compressor: after the master inserts, before the master fader
        gain.setTargetValue (juce::Decibels::decibelsToGain (gainDb.load(), -90.0f));
        const int n = buf.getNumSamples();
        auto* l = buf.getWritePointer (0); auto* r = buf.getWritePointer (1);
        for (int i = 0; i < n; ++i) { const float g = gain.getNextValue(); l[i] *= g; r[i] *= g; }
        if (amOn.load (std::memory_order_relaxed)) { const float gr = sound.process (buf, n, amPush.load (std::memory_order_relaxed), amBass.load (std::memory_order_relaxed), amTone.load (std::memory_order_relaxed), amDrive.load (std::memory_order_relaxed), amCeil.load (std::memory_order_relaxed), amCut.load (std::memory_order_relaxed)); if (gr > amGr.load (std::memory_order_relaxed)) amGr.store (gr, std::memory_order_relaxed); }      // the Archie sound: drive, then the limiter
        float pl = 0, pr = 0; double sq = 0.0;
        for (int i = 0; i < n; ++i) { pl = juce::jmax (pl, std::abs (l[i])); pr = juce::jmax (pr, std::abs (r[i])); sq += 0.5 * ((double) l[i] * l[i] + (double) r[i] * r[i]); }
        if (pl > peakL.load()) peakL.store (pl);
        if (pr > peakR.load()) peakR.store (pr);
        avgMs += (sq / (double) juce::jmax (1, n) - avgMs) * (1.0 - std::exp (-(double) n / (3.0 * rate)));      // the average loudness of the mix over about 3 seconds (mean square)
        avgMsOut.store ((float) avgMs, std::memory_order_relaxed);
        if (lm.process (l, r, n))      // a new 100 ms of loudness. With a TARGET set, PUSH follows it by itself, slowly (at most 1.2 dB a second) and never past +12 dB. Once the mix has stayed within half a dB of the target for 3 seconds
        {                              // PUSH is held where it is (so the sound stops moving); it starts following again only if the mix drifts more than 2 dB away for 2 seconds.
            const float tgt = amTarget.load (std::memory_order_relaxed), st = lm.shortTerm.load (std::memory_order_relaxed); const bool want = amOn.load (std::memory_order_relaxed) && tgt < -1.0f;
            if (! want) { holding = false; settleCnt = driftCnt = 0; lastTgt = 0.0f; amState.store (0, std::memory_order_relaxed); }
            else if (st > -60.0f)
            {
                if (tgt != lastTgt) { lastTgt = tgt; holding = false; settleCnt = driftCnt = 0; }
                const float err = tgt - st; float push = amPush.load (std::memory_order_relaxed);
                if (holding) { if (std::abs (err) > 2.0f) { if (++driftCnt >= 20) { holding = false; settleCnt = driftCnt = 0; } } else driftCnt = 0; }
                else
                {
                    push = juce::jlimit (0.0f, kAutoPushMax, push + juce::jlimit (-0.12f, 0.12f, std::abs (err) > 0.2f ? err * 0.03f : 0.0f));
                    if (std::abs (err) < 0.5f) { if (++settleCnt >= 30) holding = true; } else settleCnt = 0;
                    amPush.store (push, std::memory_order_relaxed);
                }
                amState.store (holding ? 2 : (push >= kAutoPushMax - 0.01f && err > 0.5f) ? 3 : 1, std::memory_order_relaxed);      // 1 following, 2 holding (on target), 3 cannot reach the target at the cap
            }
        }
    }
    std::atomic<float> gainDb { 0.0f }, peakL { 0.0f }, peakR { 0.0f };
    // THE MASTER COMPRESSOR (Mixer, Master Section): the same compressor as a channel's, across the whole mix, with its own attack
    std::atomic<bool> cOn { false }, cPeak { false }; std::atomic<float> cThr { -12.0f }, cRatio { 2.0f }, cAtk { 0.01f }, cRel { 0.3f }, cMake { 0.0f }, cGr { 0.0f };
    CompStage comp;
    std::atomic<bool> eqHpOn { false }, eqLpOn { false }; std::atomic<float> eqHpHz { 80.0f }, eqLpHz { 12000.0f };      // THE MASTER EQ: the same high-pass and low-pass, and the same four bands, as a channel's
    EqStage eqStage; EqPanel eq; ParaEq paraEq;
    std::atomic<bool> amOn { false }; std::atomic<float> amDrive { 0.25f }, amCeil { -1.0f }, amGr { 0.0f }, avgMsOut { 0.0f };      // THE ARCHIE SOUND (see above): on, drive 0 to 1, ceiling in dB, the limiter's gain reduction since the console last asked, and the average loudness
    static constexpr float kAutoPushMax = 12.0f;
    std::atomic<bool> amCut { true };      // the 20 Hz cutoff (a button, and it can be automated)
    std::atomic<int> amState { 0 }; bool holding = false; int settleCnt = 0, driftCnt = 0; float lastTgt = 0.0f;      // what the loudness target is doing (see the servo)
    std::atomic<float> amPush { 0.0f }, amBass { 0.0f }, amTone { 0.0f }, amTarget { 0.0f };      // push into the drive (dB), bass and tone (dB), and the loudness target in LUFS (0 = none: push is as set)
    ArchieSound sound; LoudnessMeter lm; TrackState* autoFrom = nullptr;      // (the master chain: where the master's lanes live)
private:
    double avgMs = 0.0;
    double rate = 48000.0;
    juce::SmoothedValue<float> gain { 1.0f };
};

struct HostPlayHead : public juce::AudioPlayHead
{
    juce::Optional<PositionInfo> getPosition() const override { return info; }
    PositionInfo info;
};

//==============================================================================
inline TrackState::~TrackState() = default;

} // namespace archie
