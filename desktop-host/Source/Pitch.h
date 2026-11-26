// Archie Studio host -- PITCH and COMP: the audio spikes behind Vocal Edit and Comp Edit (S12).
// A SPIKE proves the sound-making part before any screen is built on it. Nothing here is wired to the engine, the bridge or the page yet; offline checks (tests/render_test.cpp) hold every claim.
//   - trackPitch:   what pitch a recording is singing, frame by frame (YIN: difference function, cumulative mean normalisation, parabolic refinement).
//   - retunePitch:  the same recording with its pitch moved by a different number of semitones at each moment, length unchanged (the Signalsmith Stretch library's transposition, which Warp.h already uses for time).
//   - correctToScale: the two joined: track, pick the nearest note of a scale for each stretch of steady pitch, move it there by an amount (strength), leave unvoiced sound alone.
//   - compTakes:    one recording built from chosen sections of several takes, with equal-power crossfades at the joins (Comp Edit: swipe a section of a take into the comp).
// FILTERING/MONITORING MECHANISM: trackPitch reads a recording and decides, per frame, whether it is voiced and at what pitch. It judges only periodicity: it cannot tell a voice from any other steady tone,
// it gives up (unvoiced) on noise, breath, whispers and very quiet frames, and it does not follow two notes sounding at once. correctToScale moves only frames this tracker called voiced; everything else
// passes untouched. The threshold, the quietest level and the pitch range are parameters, so a person (or a later screen) can adjust them for one recording.
#pragma once
#include "Mixer.h"
#include <cmath>
#include <vector>
#include <algorithm>
#include <signalsmith-stretch/signalsmith-stretch.h>

namespace archie
{

struct PitchFrame { double sec = 0.0, hz = 0.0, confidence = 0.0; bool voiced = false; };

inline double hzToMidi (double hz) { return 69.0 + 12.0 * std::log2 (hz / 440.0); }
inline double midiToHz (double m) { return 440.0 * std::pow (2.0, (m - 69.0) / 12.0); }

// Pitch of the first channel in frames of `win` samples every `hop` samples. A frame is voiced when YIN's best dip is below `threshold` and the frame is louder than `minDb` (dBFS rms).
inline std::vector<PitchFrame> trackPitch (const AudioData& a, double minHz = 70.0, double maxHz = 1000.0, double threshold = 0.15, double minDb = -50.0, int hop = 512)
{
    std::vector<PitchFrame> out; const int N = a.numSamples(); if (N < 64 || a.rate <= 0.0) return out;
    const int tauMin = juce::jmax (2, (int) std::floor (a.rate / maxHz)), tauMax = (int) std::ceil (a.rate / minHz), W = tauMax * 2;
    if (N < W + tauMax + 2) return out;
    const float* x = a.buf.getReadPointer (0); std::vector<double> d ((size_t) tauMax + 2), cm ((size_t) tauMax + 2);
    for (int f0 = 0; f0 + W + tauMax < N; f0 += hop)
    {
        PitchFrame fr; fr.sec = ((double) f0 + (double) W / 2.0) / a.rate;
        double e = 0.0; for (int i = 0; i < W; ++i) e += (double) x[f0 + i] * x[f0 + i];
        const double db = 10.0 * std::log10 (juce::jmax (1.0e-12, e / (double) W));
        if (db > minDb)
        {
            for (int t = 1; t <= tauMax + 1 && t < (int) d.size(); ++t) { double s = 0.0; for (int i = 0; i < W; ++i) { const double df = (double) x[f0 + i] - x[f0 + i + t]; s += df * df; } d[(size_t) t] = s; }
            double run = 0.0; cm[0] = 1.0; for (int t = 1; t <= tauMax + 1; ++t) { run += d[(size_t) t]; cm[(size_t) t] = run > 0.0 ? d[(size_t) t] * (double) t / run : 1.0; }
            int best = -1; for (int t = tauMin; t <= tauMax; ++t) if (cm[(size_t) t] < threshold) { best = t; while (best + 1 <= tauMax && cm[(size_t) best + 1] < cm[(size_t) best]) ++best; break; }
            if (best > 0)
            {
                double tt = (double) best; const double y0 = cm[(size_t) best - 1], y1 = cm[(size_t) best], y2 = cm[(size_t) best + 1]; const double den = y0 - 2.0 * y1 + y2;
                if (std::abs (den) > 1.0e-12) tt += juce::jlimit (-1.0, 1.0, 0.5 * (y0 - y2) / den);
                fr.hz = a.rate / tt; fr.confidence = juce::jlimit (0.0, 1.0, 1.0 - cm[(size_t) best]); fr.voiced = true;
            }
        }
        out.push_back (fr);
    }
    return out;
}

// The recording with its pitch moved by semis(sec) semitones (read at the middle of each 512-sample step), the same length, both channels (up to two). keepFormant moves the pitch but not the vocal
// colour (the library's formant compensation), which is what a voice needs; instruments can leave it off.
template <typename SemisAt>
inline std::shared_ptr<AudioData> retunePitch (const AudioData& src, SemisAt&& semisAt, bool keepFormant = true, const std::atomic<bool>* cancel = nullptr)
{
    const int N = src.numSamples(); if (N < 2 || src.rate <= 0.0) return nullptr;
    const int nCh = juce::jmin (2, src.buf.getNumChannels());
    signalsmith::stretch::SignalsmithStretch<float> st; st.presetDefault (nCh, (float) src.rate);
    const int inLat = st.inputLatency(), outLat = st.outputLatency(), chunk = 512, T = N + outLat;
    auto fetch = [&] (int c, long long i) -> float { return i < 0 || i >= N ? 0.0f : src.buf.getSample (c, (int) i); };
    std::vector<float> pre[2], in[2], res[2]; for (int c = 0; c < nCh; ++c) { res[c].assign ((size_t) chunk, 0.0f); pre[c].resize ((size_t) inLat); for (int j = 0; j < inLat; ++j) pre[c][(size_t) j] = fetch (c, j); }
    { float* pp[2] = { pre[0].data(), nCh > 1 ? pre[1].data() : nullptr }; st.seek (pp, inLat, 1.0); }
    auto out = std::make_shared<AudioData>(); out->rate = src.rate; out->name = src.name; out->buf.setSize (nCh, N); out->buf.clear();
    int produced = 0;
    while (produced < T)
    {
        if (cancel != nullptr && cancel->load()) return nullptr;
        const int n = juce::jmin (chunk, T - produced);
        const double semis = semisAt (((double) (produced - outLat) + (double) n / 2.0) / src.rate);
        st.setTransposeSemitones ((float) semis); if (keepFormant) st.setFormantFactor (1.0f, true); else st.setFormantFactor (1.0f, false);
        for (int c = 0; c < nCh; ++c) { in[c].resize ((size_t) n); for (int j = 0; j < n; ++j) in[c][(size_t) j] = fetch (c, (long long) produced + inLat + j); }
        float* ip[2] = { in[0].data(), nCh > 1 ? in[1].data() : nullptr }; float* op[2] = { res[0].data(), nCh > 1 ? res[1].data() : nullptr };
        st.process (ip, n, op, n);
        for (int j = 0; j < n; ++j) { const int k = produced + j - outLat; if (k >= 0 && k < N) for (int c = 0; c < nCh; ++c) out->buf.setSample (c, k, res[c][(size_t) j]); }
        produced += n;
    }
    return out;
}

// Move every steady, voiced stretch to the nearest note of the scale. `scale` is a mask of the twelve notes (index 0 = the key note), `key` is the pitch class of the key note (0 = C). strength 0 does nothing
// at all, 1 puts the pitch exactly on the note. The correction is held steady across a voiced stretch and smoothed across the joins, so it does not warble.
struct ScaleSpec { int key = 0; bool mask[12] = { true, false, true, false, true, true, false, true, false, true, false, true }; };      // C major by default
inline double nearestScaleMidi (const ScaleSpec& sc, double midi)
{
    double best = std::floor (midi + 0.5), bd = 1.0e9;
    for (int m = (int) std::floor (midi) - 2; m <= (int) std::ceil (midi) + 2; ++m) { if (! sc.mask[(((m - sc.key) % 12) + 12) % 12]) continue; const double d = std::abs ((double) m - midi); if (d < bd) { bd = d; best = (double) m; } }
    return best;
}
inline std::shared_ptr<AudioData> correctToScale (const AudioData& src, const ScaleSpec& sc, double strength = 1.0, bool keepFormant = true, const std::atomic<bool>* cancel = nullptr)
{
    const auto fr = trackPitch (src); if (fr.size() < 2 || strength <= 0.0) return retunePitch (src, [] (double) { return 0.0; }, keepFormant, cancel);
    std::vector<double> shift (fr.size(), 0.0);
    for (size_t i = 0; i < fr.size(); ++i)                    // the pitch of a frame, taken as the median of its neighbours' so a single wild frame does not pull the note
    {
        if (! fr[i].voiced) continue;
        std::vector<double> m; for (size_t j = (i > 2 ? i - 2 : 0); j <= juce::jmin (fr.size() - 1, i + 2); ++j) if (fr[j].voiced) m.push_back (hzToMidi (fr[j].hz));
        std::sort (m.begin(), m.end()); const double med = m[m.size() / 2]; shift[i] = (nearestScaleMidi (sc, med) - med) * strength;
    }
    auto semis = [fr, shift] (double sec)                      // linear between frames (silence and unvoiced frames hold 0, so the correction fades in and out around them)
    {
        if (sec <= fr.front().sec) return shift.front(); if (sec >= fr.back().sec) return shift.back();
        const double h = fr[1].sec - fr[0].sec; const size_t i = (size_t) ((sec - fr[0].sec) / h); const double t = (sec - fr[i].sec) / h; return shift[i] + t * (shift[juce::jmin (i + 1, shift.size() - 1)] - shift[i]);
    };
    return retunePitch (src, semis, keepFormant, cancel);
}

// COMP: one recording from sections of several takes. A section says which take plays from startSec to endSec (seconds on the comp's own timeline, all takes lined up from 0). Sections outside the comp's
// length are clipped; any gap plays silence. Where two sections meet, the join is an equal-power crossfade of `fadeSec` centred on the join (never longer than half the shorter section), so a join does
// not click. A section whose take is shorter than the section plays silence for the rest.
struct CompSection { double startSec = 0.0, endSec = 0.0; int take = 0; };
inline std::shared_ptr<AudioData> compTakes (const std::vector<const AudioData*>& takes, std::vector<CompSection> sections, double lengthSec, double rate, double fadeSec = 0.02)
{
    if (takes.empty() || lengthSec <= 0.0 || rate <= 0.0) return nullptr;
    const int N = (int) std::llround (lengthSec * rate); int nCh = 1; for (auto* t : takes) if (t != nullptr) nCh = juce::jmax (nCh, juce::jmin (2, t->buf.getNumChannels()));
    auto out = std::make_shared<AudioData>(); out->rate = rate; out->buf.setSize (nCh, N); out->buf.clear();
    std::sort (sections.begin(), sections.end(), [] (const CompSection& a, const CompSection& b) { return a.startSec < b.startSec; });
    auto takeAt = [&] (const AudioData* t, int c, long long i) -> float { if (t == nullptr || i < 0 || i >= t->numSamples() || t->rate != rate) return 0.0f; return t->buf.getSample (juce::jmin (c, t->buf.getNumChannels() - 1), (int) i); };
    for (size_t k = 0; k < sections.size(); ++k)
    {
        const auto& s = sections[k]; if (s.take < 0 || s.take >= (int) takes.size() || s.endSec <= s.startSec) continue;
        const long long a = juce::jlimit (0, N, (int) std::llround (s.startSec * rate)), b = juce::jlimit (0, N, (int) std::llround (s.endSec * rate)); if (b <= a) continue;
        // the fade at each end covers half of fadeSec on each side of the join, but only if a neighbour is there to fade with; the first and last section of the comp start and end hard
        const bool joinL = k > 0 && std::abs (sections[k - 1].endSec - s.startSec) < 1.0 / rate * 2.0, joinR = k + 1 < sections.size() && std::abs (sections[k + 1].startSec - s.endSec) < 1.0 / rate * 2.0;
        const double half = juce::jmin (fadeSec / 2.0, (double) (b - a) / rate / 2.0) * rate;          // in samples
        const long long lo = joinL ? juce::jmax (0LL, a - (long long) half) : a, hi = joinR ? juce::jmin ((long long) N, b + (long long) half) : b;
        for (long long i = lo; i < hi; ++i)
        {
            double g = 1.0;
            if (joinL && half > 0.0 && i < a + (long long) half) g = std::sin (juce::jlimit (0.0, 1.0, ((double) (i - a) + half) / (2.0 * half)) * 1.5707963267948966);
            if (joinR && half > 0.0 && i >= b - (long long) half) g *= std::cos (juce::jlimit (0.0, 1.0, ((double) (i - b) + half) / (2.0 * half)) * 1.5707963267948966);
            for (int c = 0; c < nCh; ++c) out->buf.setSample (c, (int) i, out->buf.getSample (c, (int) i) + (float) g * takeAt (takes[(size_t) s.take], c, i));
        }
    }
    return out;
}

} // namespace archie
