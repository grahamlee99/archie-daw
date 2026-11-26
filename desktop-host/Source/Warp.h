// Archie Studio host -- WARP: making a sound follow the song's tempo.
// A sound recorded at its own tempo (say 100 beats a minute) is laid on a timeline at another (say 120). Warping stretches or squeezes it in time so that its beats fall on the timeline's beats,
// and keeps following if the timeline's tempo changes part-way. Two ways:
//   - KEEP PITCH (the default): the sound is time-stretched without changing its pitch, by the Signalsmith Stretch library (MIT licence, see third_party/README.md).
//   - REPITCH: the sound is simply played faster or slower, so its pitch moves with the speed, like a record or a tape.
// The work is done AHEAD of playback, off the audio thread: renderWarp makes a new sound that already has the right speed at every moment, and playback is then the ordinary playback of that sound.
// Nothing here knows about the mixer or the sound card.
#pragma once
#include "Mixer.h"
#include "TempoMap.h"
#include <atomic>
#include <cmath>
#include <memory>
#include <vector>
#include <signalsmith-stretch/signalsmith-stretch.h>

namespace archie
{

// The mapping between a sound's own time and its own beats: straight at srcBpm, bent through the warp markers. f: seconds -> beats, inv: beats -> seconds. Outside the markers it carries on at the source tempo.
struct WarpMap
{
    std::vector<std::pair<double, double>> pts;       // (sound seconds, sound beats), both strictly increasing
    double slope = 2.0;                                // beats per second outside the markers = srcBpm / 60
    double f (double s) const
    {
        if (pts.empty()) return s * slope;
        if (s <= pts.front().first) return pts.front().second + (s - pts.front().first) * slope;
        if (s >= pts.back().first) return pts.back().second + (s - pts.back().first) * slope;
        size_t lo = 0, hi = pts.size() - 1; while (hi - lo > 1) { const size_t m = (lo + hi) / 2; if (pts[m].first <= s) lo = m; else hi = m; }
        return pts[lo].second + (s - pts[lo].first) * (pts[hi].second - pts[lo].second) / (pts[hi].first - pts[lo].first);
    }
    double inv (double F) const
    {
        if (pts.empty()) return F / slope;
        if (F <= pts.front().second) return pts.front().first + (F - pts.front().second) / slope;
        if (F >= pts.back().second) return pts.back().first + (F - pts.back().second) / slope;
        size_t lo = 0, hi = pts.size() - 1; while (hi - lo > 1) { const size_t m = (lo + hi) / 2; if (pts[m].second <= F) lo = m; else hi = m; }
        return pts[lo].first + (F - pts[lo].second) * (pts[hi].first - pts[lo].first) / (pts[hi].second - pts[lo].second);
    }
};
inline WarpMap makeWarpMap (const std::vector<WarpMarker>& markers, double srcBpm, bool ignoreMarkers = false)
{
    WarpMap m; m.slope = srcBpm > 0.0 ? srcBpm / 60.0 : 2.0;
    if (! ignoreMarkers) for (const auto& k : markers) m.pts.push_back ({ k.sec, k.beat });
    return m;
}
// a sound that LOOPS (a clip longer than its sound): the mapping repeats, each pass adding the beats one pass of the sound takes. `s` and the result run on past the end of the sound.
inline double warpSecAt (const WarpMap& m, double F, double L)
{
    const double T = m.f (L) - m.f (0.0); if (T <= 1.0e-9 || L <= 0.0) return m.inv (F);
    const double k = std::floor ((F - m.f (0.0)) / T); return k * L + m.inv (F - k * T);
}
inline double warpBeatAt (const WarpMap& m, double s, double L)
{
    const double T = m.f (L) - m.f (0.0); if (T <= 1.0e-9 || L <= 0.0) return m.f (s);
    const double k = std::floor (s / L); return m.f (s - k * L) + k * T;
}

struct WarpParams
{
    std::vector<WarpMarker> markers;                 // warp markers (ignored when reverse)
    double startBeat = 0.0, lengthBeats = 0.0;      // where the clip sits on the timeline
    double offsetSec = 0.0;                          // how far into the sound the clip begins (seconds of the sound)
    double srcBpm = 120.0;                           // the tempo the sound was made at: this many of ITS beats a minute
    bool reverse = false, repitch = false;
};

// The sound as it must be heard over the clip's whole length on this tempo map, at `outRate`. Returns nullptr if there is nothing to make (no length, no tempo, an empty sound, a clip longer than
// 15 minutes) or if `cancel` was raised while it worked. The source's loop (a clip longer than its sound repeats the sound) is followed.
inline std::shared_ptr<AudioData> renderWarp (const AudioData& src, const WarpParams& p, const TempoMap& tm, double outRate, const std::atomic<bool>* cancel = nullptr)
{
    if (src.numSamples() < 2 || src.rate <= 0.0 || outRate <= 0.0 || p.srcBpm <= 0.0 || p.lengthBeats <= 0.0) return nullptr;
    const double ws = tm.secondsAt (p.startBeat), span = tm.secondsAt (p.startBeat + p.lengthBeats) - ws;
    if (span <= 0.0 || span > 900.0) return nullptr;
    // the source at the output rate, both channels, reversed if asked
    const int nCh = juce::jmin (2, src.buf.getNumChannels()); const int M = juce::jmax (2, (int) std::ceil ((double) src.numSamples() * outRate / src.rate));
    std::vector<float> S[2]; for (auto& v : S) v.assign ((size_t) M, 0.0f);
    for (int c = 0; c < 2; ++c)
    {
        const float* in = src.buf.getReadPointer (juce::jmin (c, nCh - 1));
        if (src.rate == outRate) std::copy (in, in + juce::jmin (M, src.numSamples()), S[c].begin());
        else { juce::LagrangeInterpolator li; li.process (src.rate / outRate, in, S[c].data(), M); }
        if (p.reverse) std::reverse (S[c].begin(), S[c].end());
    }
    auto fetch = [&] (int c, long long i) -> float { return i < 0 ? 0.0f : S[c][(size_t) (i % M)]; };
    const double L = (double) src.numSamples() / src.rate; const WarpMap wm = makeWarpMap (p.markers, p.srcBpm, p.reverse); const double F0 = wm.f (p.offsetSec);
    auto tau = [&] (double ts) { return warpSecAt (wm, (tm.beatAt (ws + ts) - p.startBeat) + F0, L); };     // where in the sound (seconds) the clip is at ts seconds after its start: its own beats, bent through the markers

    const int N = (int) std::ceil (span * outRate);
    auto out = std::make_shared<AudioData>(); out->rate = outRate; out->buf.setSize (2, N);
    if (p.repitch)
    {
        for (int k = 0; k < N; ++k)
        {
            if ((k & 8191) == 0 && cancel != nullptr && cancel->load()) return nullptr;
            const double pos = tau ((double) k / outRate) * outRate; const long long i0 = (long long) std::floor (pos); const float fr = (float) (pos - (double) i0);
            for (int c = 0; c < 2; ++c) { const float a = fetch (c, i0), b = fetch (c, i0 + 1); out->buf.setSample (c, k, a + fr * (b - a)); }
        }
        return out;
    }
    signalsmith::stretch::SignalsmithStretch<float> st; st.presetDefault (2, (float) outRate);
    const int inLat = st.inputLatency(), outLat = st.outputLatency(), chunk = 512, T = N + outLat;
    const long long i0 = (long long) std::llround (tau (0.0) * outRate);
    std::vector<float> pre[2], in[2], res[2]; for (auto& v : res) v.assign ((size_t) chunk, 0.0f);
    for (int c = 0; c < 2; ++c) { pre[c].resize ((size_t) inLat); for (int j = 0; j < inLat; ++j) pre[c][(size_t) j] = fetch (c, i0 + j); }          // (the library wants the FIRST inputLatency samples of the sound here: its processing time then matches the start of the sound)
    { float* pp[2] = { pre[0].data(), pre[1].data() }; st.seek (pp, inLat, juce::jmax (0.05, (tau (0.01) - tau (0.0)) / 0.01)); }
    int produced = 0;
    while (produced < T)
    {
        if (cancel != nullptr && cancel->load()) return nullptr;
        const int n = juce::jmin (chunk, T - produced);
        const long long a = (long long) std::llround (tau ((double) produced / outRate) * outRate) + inLat, b = (long long) std::llround (tau ((double) (produced + n) / outRate) * outRate) + inLat;          // the first inLat samples went in by seek()
        const int inLen = (int) juce::jmax ((long long) 0, b - a);
        for (int c = 0; c < 2; ++c) { in[c].resize ((size_t) juce::jmax (1, inLen)); for (int j = 0; j < inLen; ++j) in[c][(size_t) j] = fetch (c, a + j); }
        float* ip[2] = { in[0].data(), in[1].data() }; float* op[2] = { res[0].data(), res[1].data() };
        st.process (ip, inLen, op, n);
        for (int j = 0; j < n; ++j) { const int k = produced + j - outLat; if (k >= 0 && k < N) { out->buf.setSample (0, k, res[0][(size_t) j]); out->buf.setSample (1, k, res[1][(size_t) j]); } }
        produced += n;
    }
    return out;
}

// A guess at the tempo a sound was made at, from where its hits fall: the tempo (70 to 190 a minute) at which the hits line up best on a regular beat. It says how sure it is (0 to 1), and says
// nothing (bpm 0) when the sound has fewer than four hits.
// FILTERING/MONITORING MECHANISM: this reads the TIMES of the hits the transient detector found and decides one number from them. It does not judge pitch, groove, swing or what the sound is, it cannot tell
// a tempo from its half or double (it prefers the one nearest 120), and a sound with no regular beat gets a low confidence. It is adjusted by the detector's sensitivity (fixed at 0.7 here) and the
// person always has the last word: the source tempo is a number they can type.
struct TempoGuess { double bpm = 0.0, confidence = 0.0; };
inline TempoGuess detectTempo (const AudioData& a)
{
    const auto hits = detectTransients (a, 0.7); const size_t N = hits.size();
    if (N < 4) return {};
    std::vector<double> score; double best = 0.0;
    for (double bpm = 70.0; bpm <= 190.0; bpm += 0.05)
    {
        const double w = 6.283185307179586 * bpm / 60.0; double sr = 0.0, si = 0.0;
        for (double h : hits) { sr += std::cos (w * h); si += std::sin (w * h); }
        const double s = std::sqrt (sr * sr + si * si) / (double) N; score.push_back (s); best = juce::jmax (best, s);
    }
    double pick = 0.0, pickDist = 1.0e9;
    for (size_t i = 1; i + 1 < score.size(); ++i)
        if (score[i] >= 0.95 * best && score[i] >= score[i - 1] && score[i] >= score[i + 1])
        {
            const double bpm = 70.0 + 0.05 * (double) i, dist = std::abs (std::log2 (bpm / 120.0));
            if (dist < pickDist) { pickDist = dist; pick = bpm; }
        }
    const double floorScore = 1.0 / std::sqrt ((double) N);           // what hits with no pattern at all would score by chance
    return { pick, juce::jlimit (0.0, 1.0, (best - floorScore) / (1.0 - floorScore)) };
}

} // namespace archie
