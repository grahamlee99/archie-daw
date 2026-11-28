#pragma once
// SIDE-CHAIN: what a compressor, a gate or a plug-in's side-chain input listens to instead of its own sound. A source is a number:
//   0 = none      > 0 = a track (its uid)      -1..-5 = a tempo pulse (4/4, 2/4, 6/8, 1/8, 1/16)      -101..-108 = FX return 1..8
// A channel source can be narrowed by a Low cut and a High cut (they shape the TRIGGER only, never what is heard) and listened to. Pulses follow the song and run only while it plays.
#include <juce_audio_basics/juce_audio_basics.h>
#include <atomic>
#include <cmath>
#include <vector>

namespace archieSC {

constexpr int kPulse44 = -1, kPulse24 = -2, kPulse68 = -3, kPulse8 = -4, kPulse16 = -5, kReturnBase = -101, kReturnCount = 8;
constexpr float kLoOpen = 20.0f, kHiOpen = 20000.0f;
inline bool isPulse (int k) { return k <= kPulse44 && k >= kPulse16; }
inline int returnOf (int k) { return k <= kReturnBase && k > kReturnBase - kReturnCount ? kReturnBase - k : -1; }
inline int sanitize (int k) { return (k > 0 || isPulse (k) || returnOf (k) >= 0) ? k : 0; }

// A pulse's level (0..1) at a song position in quarter notes, and the time since it began. 4/4 = every beat; 2/4 = every second beat (1 and 3 of a 4/4 bar); 6/8 = six eighths to a bar with the
// first and fourth strong; 1/8 and 1/16 = every eighth or sixteenth. Each rises in 1 ms, holds 40 ms and is gone by about 110 ms (shorter at very fast rates, so one always ends before the next).
inline float pulseAt (int type, double beat, double bpm, double& since)
{
    const double q = type == kPulse44 ? 1.0 : type == kPulse24 ? 2.0 : type == kPulse16 ? 0.25 : 0.5;
    const double tempo = bpm > 1.0 ? bpm : 120.0, n = std::floor (beat / q + 1.0e-7), into = juce::jmax (0.0, beat - n * q);
    since = into * 60.0 / tempo;
    float strength = 1.0f;
    if (type == kPulse68) { const long long i = (long long) n; const int in = (int) (((i % 6) + 6) % 6); strength = (in == 0 || in == 3) ? 1.0f : 0.55f; }
    const double scale = juce::jmin (1.0, q * 60.0 / tempo / 0.085), hold = 0.040 * scale, tail = 0.012 * scale;
    double e = since < 0.001 ? since / 0.001 : since < hold ? 1.0 : std::exp (-(since - hold) / tail);
    if (e < 0.001) e = 0.0;
    return (float) e * strength;
}

// The trigger filter: 24 dB per octave low cut and high cut on a stereo pair, gliding to where the controls are. Fully open, it touches nothing.
struct KeyFilter
{
    float lo = kLoOpen, hi = kHiOpen; double s[2][4][2] {}; double rate = 0.0; float k = 0.002f;
    void reset() { for (auto& c : s) for (auto& f : c) f[0] = f[1] = 0.0; }
    void process (float* l, float* r, int n, float wantLo, float wantHi, double sr)
    {
        if (rate != sr) { rate = sr; k = (float) (1.0 - std::exp (-1.0 / (0.012 * sr))); lo = kLoOpen; hi = kHiOpen; reset(); }
        wantLo = juce::jlimit (kLoOpen, 4000.0f, wantLo); wantHi = juce::jlimit (22.0f, kHiOpen, wantHi);
        const bool useLo = wantLo > kLoOpen || lo > kLoOpen * 1.01f, useHi = wantHi < kHiOpen || hi < kHiOpen * 0.99f;
        if (! useLo && ! useHi) { lo = kLoOpen; hi = kHiOpen; reset(); return; }
        static constexpr double q[2] = { 0.5411961, 1.3065630 };
        for (int i = 0; i < n; ++i)
        {
            lo *= std::pow (wantLo / lo, k); hi *= std::pow (wantHi / hi, k);
            const double gl = std::tan (3.14159265358979 * juce::jmin ((double) lo, 0.45 * sr) / sr), gh = std::tan (3.14159265358979 * juce::jmin ((double) hi, 0.45 * sr) / sr);
            for (int c = 0; c < 2; ++c)
            {
                float* x = c == 0 ? l : r;
                double v = x[i];
                for (int st = 0; st < 4; ++st)
                {
                    const bool high = st < 2;            // stages 0, 1: the low cut (a high-pass); 2, 3: the high cut (a low-pass)
                    if (high ? ! useLo : ! useHi) continue;
                    const double g = high ? gl : gh, d = 1.0 / q[st & 1], a1 = 1.0 / (1.0 + g * (g + d)), a2 = g * a1, a3 = g * a2;
                    auto& m = s[c][st]; const double v3 = v - m[1], v1 = a1 * m[0] + a2 * v3, v2 = m[1] + a2 * m[0] + a3 * v3; m[0] = 2.0 * v1 - m[0]; m[1] = 2.0 * v2 - m[1];
                    v = high ? v - d * v1 - v2 : v2;
                }
                x[i] = (float) v;
            }
        }
    }
};

// One listener's own settings and state: the filter, the delay that lines the trigger up with the sound it works on, and LISTEN.
struct SideTap
{
    static constexpr int kCapacity = 1 << 15;      // 32768 samples: about 0.68 s at 48 kHz
    std::atomic<float> lo { kLoOpen }, hi { kHiOpen };
    std::atomic<bool> listen { false };
    KeyFilter filter; std::vector<float> ring; int w = 0; float listenMix = 0.0f;
    // message thread, before a plan that delays this listener's trigger is published
    void ensureRing() { if (ring.empty()) ring.assign ((size_t) kCapacity * 2, 0.0f); }
    void delay (float* l, float* r, int n, int samples)
    {
        if (ring.empty() || samples <= 0) return;
        samples = juce::jmin (samples, kCapacity - 1);
        for (int c = 0; c < 2; ++c) { float* x = c == 0 ? l : r; float* m = ring.data() + c * kCapacity; int p = w; for (int i = 0; i < n; ++i, ++p) { m[p & (kCapacity - 1)] = x[i]; x[i] = m[(p - samples) & (kCapacity - 1)]; } }
        w = (w + n) & (kCapacity - 1);
    }
    // LISTEN: the sound is replaced by the trigger, faded over 8 ms either way
    void hear (juce::AudioBuffer<float>& buf, const float* keyL, const float* keyR, int n, double sr)
    {
        const float target = listen.load (std::memory_order_relaxed) && keyL != nullptr ? 1.0f : 0.0f;
        if (target == 0.0f && listenMix <= 0.0f) return;
        const float step = (float) (1.0 / juce::jmax (16.0, 0.008 * sr)); float m = listenMix;
        for (int c = 0; c < juce::jmin (2, buf.getNumChannels()); ++c)
        {
            float* x = buf.getWritePointer (c); const float* k = keyL != nullptr ? (c == 0 ? keyL : keyR) : nullptr; m = listenMix;
            for (int i = 0; i < n; ++i) { m = target > m ? juce::jmin (target, m + step) : juce::jmax (target, m - step); x[i] += m * ((k != nullptr ? k[i] : 0.0f) - x[i]); }
        }
        listenMix = m;
    }
};

} // namespace archieSC
