// Archie Studio host (desktop) -- the Analyser: measures a VST3 plug-in the way a host would, offline (audio goes through processBlock; no sound card), with the JUCE-free core in ab_core.h.
// It runs in a CHILD PROCESS of the program (`ArchieStudio.exe --analyse <plug-in> <out.json>`), so a plug-in that crashes while being measured cannot take the program down.
//
// Honesty rules (admin-docs/archie-audio-lab-brief-2026-11-20.md): every number here is MEASURED; every result carries its conditions; a rule is a measured value next to a stated limit;
// nothing is invented, and what was NOT measured is listed in every result. There is no "analog reference" here, so nothing is compared with one.
//
// FILTERING/MONITORING MECHANISM: the rules in Analyser::run (effectRules / instrumentRules).
// What they screen: a plug-in's output for non-finite numbers, DC offset, noise on silence, aliasing at a moderate level, pitch error, non-harmonic energy, and a reported latency that disagrees with the measured delay.
// What they deliberately do NOT judge: taste. A distortion, detune, chorus or noise that is the point of a plug-in will show as "above the limit"; the number is the fact, the limit is only a stated yardstick.
// How they can be adjusted: every limit is a named constant in this file, printed with the result; the tests (desktop-host/tests) pin the rule list.
#pragma once
#include <juce_audio_processors/juce_audio_processors.h>
#include "ab_core.h"
#include <algorithm>
#include <sstream>

namespace archie
{

class Analyser
{
public:
    static constexpr size_t N = 65536;
    static constexpr double SR = 48000.0;
    static constexpr int BLOCK = 512;
    // the yardsticks (printed with every result)
    static constexpr double kDcLimitDb = -70.0, kNoiseLimitDb = -80.0, kAliasLimitDb = -60.0, kLatencyLimitSamples = 2.0, kPitchLimitCents = 10.0, kNonHarmonicLimitDb = -40.0, kSoundLimitDb = -60.0;

    // Loads the plug-in and measures it. Returns the JSON text; `error` is set (and the text empty) when it could not be loaded.
    static std::string run (const juce::File& vst3, const juce::String& identifier, std::string& error)
    {
        juce::AudioPluginFormatManager fm; fm.addDefaultFormats();
        juce::OwnedArray<juce::PluginDescription> descs;
        for (auto* f : fm.getFormats()) if (f->getName() == "VST3") f->findAllTypesForFile (descs, vst3.getFullPathName());
        if (descs.isEmpty()) { error = "no VST3 plug-in found in that file"; return {}; }
        juce::PluginDescription* desc = descs[0];
        if (identifier.isNotEmpty()) for (auto* d : descs) if (d->createIdentifierString() == identifier) desc = d;
        auto trace = [] (const char* what) { if (std::getenv ("ARCHIE_TRACE") != nullptr) { std::fprintf (stderr, "[analyser] %s\n", what); std::fflush (stderr); } };   // set ARCHIE_TRACE=1 to see how far it got
        trace ("loading the plug-in");
        juce::String err;
        auto inst = fm.createPluginInstance (*desc, SR, BLOCK, err);
        trace ("loaded");
        if (inst == nullptr) { error = err.isNotEmpty() ? err.toStdString() : "could not load the plug-in"; return {}; }
        trace ("setting the buses");
        inst->enableAllBuses();
        auto layout = inst->getBusesLayout();
        if (inst->getBusCount (true) > 0 && layout.inputBuses.size() > 0) layout.inputBuses.getReference (0) = juce::AudioChannelSet::stereo();
        if (inst->getBusCount (false) > 0 && layout.outputBuses.size() > 0) layout.outputBuses.getReference (0) = juce::AudioChannelSet::stereo();
        if (inst->checkBusesLayoutSupported (layout)) inst->setBusesLayout (layout);
        inst->setPlayConfigDetails (juce::jmax (0, inst->getTotalNumInputChannels()), juce::jmax (1, inst->getTotalNumOutputChannels()), SR, BLOCK);
        // a host gives a plug-in the song position and tempo; some plug-ins read them on the first block (the analysis plays at 120 bpm, 4/4, from the start, not rolling)
        struct FixedPlayHead : public juce::AudioPlayHead { juce::Optional<PositionInfo> getPosition() const override { PositionInfo i; i.setBpm (120.0); i.setTimeSignature (juce::AudioPlayHead::TimeSignature { 4, 4 }); i.setPpqPosition (0.0); i.setTimeInSeconds (0.0); i.setTimeInSamples (0); i.setIsPlaying (true); i.setIsRecording (false); i.setPpqPositionOfLastBarStart (0.0); return i; } } fixedHead;
        inst->setPlayHead (&fixedHead);
        trace ("preparing");
        inst->prepareToPlay (SR, BLOCK);
        trace ("prepared; running the suite");

        const bool instrument = desc->isInstrument || (inst->acceptsMidi() && inst->getTotalNumInputChannels() == 0);
        Runner r { *inst, 0 };
        std::vector<ab::Rule> rules;
        std::ostringstream body;
        if (instrument) instrumentSuite (r, body, rules); else effectSuite (r, body, rules);

        std::ostringstream j;
        j << "{\"tool\":\"Archie Studio Analyser 1 (offline, through a host, in a separate process)\",\"plugin\":{\"name\":" << ab::jstr (desc->name.toStdString()) << ",\"maker\":" << ab::jstr (desc->manufacturerName.toStdString())
          << ",\"kind\":" << ab::jstr (instrument ? "instrument" : "effect") << ",\"latencyReportedSamples\":" << inst->getLatencySamples() << ",\"parameters\":" << inst->getParameters().size() << ",\"file\":" << ab::jstr (vst3.getFileName().toStdString()) << "}"
          << ",\"conditions\":{\"sampleRateHz\":" << ab::jnum (SR) << ",\"hostBlockSize\":" << BLOCK << ",\"recordLength\":" << N << ",\"parameters\":\"as the plug-in starts (its defaults)\",\"channels\":\"stereo, the same signal on both sides\"}"
          << ",\"provenance\":{\"everything\":\"measured\",\"analogReference\":\"not available: nothing here is compared with an analog unit\"}"
          << "," << body.str()
          << ",\"verdict\":{\"allWithinStatedLimits\":" << (ab::allPass (rules) ? "true" : "false") << ",\"note\":\"A limit is a yardstick, not a judgement: an effect or instrument that distorts, detunes or adds noise on purpose will exceed some of them.\",\"rules\":" << ab::jrules (rules) << "}"
          << ",\"notMeasured\":[\"every parameter other than at its default (this first suite does not sweep parameters)\",\"CPU cost\",\"behaviour when automated\",\"behaviour inside a host other than this one\",\"real music instead of test signals\",\"anything against an analog reference\"]}";
        return j.str();
    }

private:
    struct Out { std::vector<double> l, r; };
    struct Runner
    {
        juce::AudioPluginInstance& inst; int nonFinite;
        // pushes the stereo input (same signal both sides, empty = silence) and the MIDI events (absolute sample positions) through the plug-in
        Out run (const std::vector<double>& in, size_t total, const std::vector<std::pair<size_t, juce::MidiMessage>>& midiEvents = {})
        {
            Out o; o.l.assign (total, 0.0); o.r.assign (total, 0.0);
            const int ch = juce::jmax (2, juce::jmax (inst.getTotalNumInputChannels(), inst.getTotalNumOutputChannels()));
            juce::AudioBuffer<float> buf (ch, BLOCK); juce::MidiBuffer midi;
            for (size_t off = 0; off < total; off += BLOCK)
            {
                const int n = (int) std::min<size_t> (BLOCK, total - off);
                buf.clear(); midi.clear();
                for (int i = 0; i < n; ++i) { const float v = (off + (size_t) i < in.size()) ? (float) in[off + (size_t) i] : 0.0f; buf.setSample (0, i, v); buf.setSample (1, i, v); }
                for (const auto& e : midiEvents) if (e.first >= off && e.first < off + (size_t) n) midi.addEvent (e.second, (int) (e.first - off));
                inst.processBlock (buf, midi);
                for (int i = 0; i < n; ++i)
                {
                    const double a = (double) buf.getSample (0, i), b = (double) buf.getSample (juce::jmin (1, buf.getNumChannels() - 1), i);
                    if (! std::isfinite (a) || ! std::isfinite (b)) ++nonFinite;
                    o.l[off + (size_t) i] = std::isfinite (a) ? a : 0.0; o.r[off + (size_t) i] = std::isfinite (b) ? b : 0.0;
                }
            }
            return o;
        }
        void reset() { inst.reset(); }
    };

    static std::vector<double> toneIn (size_t k, double amp, size_t total)
    {
        std::vector<double> x (total);
        for (size_t n = 0; n < total; ++n) x[n] = amp * std::sin (2 * ab::kPi * (double) (k * n % N) / (double) N);
        return x;
    }
    static double rmsDb (const std::vector<double>& x, size_t from, size_t to)
    {
        double s = 0; for (size_t i = from; i < to && i < x.size(); ++i) s += x[i] * x[i];
        return 10.0 * std::log10 (s / (double) std::max<size_t> (1, to - from) + 1e-30);
    }
    static std::string b (bool v) { return v ? "true" : "false"; }

    //==========================================================================
    static void effectSuite (Runner& r, std::ostringstream& j, std::vector<ab::Rule>& rules)
    {
        const size_t warm = (size_t) SR;
        j << "\"effect\":{";
        // ---- A. impulse response at the defaults ----
        r.reset();
        std::vector<double> in (warm + N, 0.0); in[warm] = 0.1;
        auto o = r.run (in, warm + N);
        std::vector<double> h (N); for (size_t n = 0; n < N; ++n) h[n] = o.l[warm + n] / 0.1;
        ab::Validity v; ab::Response resp = ab::responseFromImpulse (h, SR, &v);
        size_t pk = 0; double best = 0; for (size_t n = 0; n < N; ++n) if (std::abs (h[n]) > best) { best = std::abs (h[n]); pk = n; }
        double gain1k = 0; { size_t bi = 0; double bd = 1e300; for (size_t i = 0; i < resp.freqHz.size(); ++i) { const double d = std::abs (resp.freqHz[i] - 1000.0); if (d < bd) { bd = d; bi = i; } } gain1k = resp.magDb[bi]; }
        const ab::Peak peak = ab::findPeak (resp, 20.0, 20000.0);
        j << "\"impulse\":{\"inputPeak\":0.1,\"validity\":{\"impulseDecayed\":" << b (v.impulseDecayed) << ",\"tailDb\":" << ab::jnum (v.tailDb) << ",\"finite\":" << b (v.finite) << ",\"note\":\"" << (v.impulseDecayed ? "the response died away inside the record" : "the response had NOT died away in 1.36 s (a long tail, delay or reverb): the curve is for the first 1.36 s only") << "\"}"
          << ",\"peakDelaySamples\":" << pk << ",\"gainAt1kHzDb\":" << ab::jnum (gain1k) << ",\"peakDb\":" << ab::jnum (peak.magDb) << ",\"peakHz\":" << ab::jnum (peak.freqHz)
          << ",\"curve\":" << ab::jcurve (resp, 120, 20.0, 20000.0) << "}";
        if (v.impulseDecayed && best > 0.05) rules.push_back (ab::makeRule ("reported latency vs measured peak delay (samples)", std::abs ((double) pk - (double) r.inst.getLatencySamples()), kLatencyLimitSamples));

        // ---- B. tones at several levels: gain, harmonics, aliasing ----
        struct Tone { const char* name; size_t k; };
        const Tone tones[] = { { "1 kHz", 1365 }, { "5 kHz", 6827 }, { "10 kHz", 13653 } };
        const double levels[] = { 0.01, 0.1, 0.25, 0.5, 1.0 };
        j << ",\"tones\":{\"note\":\"each tone has an exact whole number of cycles in the record, so every component sits on one FFT bin: true harmonics are told apart from folded (aliased) ones\",\"cases\":[";
        bool first = true; double alias5kMid = -400; bool haveAlias = false, allCoherent = true;
        for (const auto& t : tones) for (double a : levels)
        {
            r.reset();
            auto out = r.run (toneIn (t.k, a, warm + N), warm + N);
            std::vector<double> cap (out.l.begin() + (long) warm, out.l.end());
            const ab::Harmonics hh = ab::analyseTone (cap, SR, t.k);
            const double gain = hh.fundamentalDb - 20.0 * std::log10 (a);
            if (! hh.coherent) allCoherent = false;
            if (std::string (t.name) == "5 kHz" && a == 0.25 && hh.coherent) { alias5kMid = hh.aliasDb; haveAlias = true; }
            if (! first) j << ","; first = false;
            j << "{\"tone\":" << ab::jstr (t.name) << ",\"inputPeak\":" << ab::jnum (a) << ",\"inputDbfs\":" << ab::jnum (20.0 * std::log10 (a)) << ",\"outputDbfs\":" << ab::jnum (hh.fundamentalDb) << ",\"gainDb\":" << ab::jnum (gain)
              << ",\"thdDb\":" << ab::jnum (hh.thdDb) << ",\"evenDb\":" << ab::jnum (hh.evenDb) << ",\"oddDb\":" << ab::jnum (hh.oddDb) << ",\"aliasDb\":" << ab::jnum (hh.aliasDb) << ",\"otherDb\":" << ab::jnum (hh.otherDb) << ",\"coherent\":" << b (hh.coherent) << "}";
        }
        j << "],\"allCoherent\":" << b (allCoherent) << ",\"coherenceNote\":" << ab::jstr (allCoherent ? "every tone was periodic in the record, so the harmonic and aliasing figures are valid" : "at least one tone was NOT periodic in the record: this plug-in changes its output over time at its defaults (a modulation effect, a delay or a reverb), so its harmonic and aliasing split is NOT valid for those tones (marked coherent:false) and the aliasing rule was not applied") << "}";
        if (haveAlias) rules.push_back (ab::makeRule ("aliased energy, 5 kHz tone at -12 dBFS (dB re the tone)", alias5kMid, kAliasLimitDb));

        // ---- C. silence in: noise floor and DC ----
        r.reset();
        auto sil = r.run ({}, warm + N);
        double mean = 0; for (size_t n = warm; n < warm + N; ++n) mean += sil.l[n]; mean /= (double) N;
        const double noiseDb = rmsDb (sil.l, warm, warm + N), dcDb = 20.0 * std::log10 (std::abs (mean) + 1e-12);
        j << ",\"silence\":{\"noiseRmsDbfs\":" << ab::jnum (noiseDb) << ",\"dcOffsetDbfs\":" << ab::jnum (dcDb) << "}";
        rules.push_back (ab::makeRule ("noise on silent input (dBFS RMS)", noiseDb, kNoiseLimitDb));
        rules.push_back (ab::makeRule ("DC offset on silent input (dBFS)", dcDb, kDcLimitDb));

        // ---- D. stability: a loud burst (+6 dBFS noise), then silence: finite, and does it recover ----
        r.reset();
        std::vector<double> loud (warm, 0.0); { uint32_t s = 777; for (auto& x : loud) { s = s * 1664525u + 1013904223u; x = 2.0 * (((double) (s >> 8) / 8388608.0) - 1.0); } }
        loud.resize (warm * 2, 0.0);
        auto lo = r.run (loud, warm * 2 + (size_t) SR);
        double peakOut = 0; for (size_t n = 0; n < warm; ++n) peakOut = std::max (peakOut, std::abs (lo.l[n]));
        const double afterDb = rmsDb (lo.l, warm * 2, warm * 2 + (size_t) SR);
        j << ",\"stability\":{\"burst\":\"1 s of noise at +6 dBFS peak, then 2 s of silence\",\"peakOutput\":" << ab::jnum (peakOut) << ",\"rmsAfterBurstDbfs\":" << ab::jnum (afterDb) << "}";

        // ---- E. null test against the input (informational) ----
        r.reset();
        std::vector<double> noise (N + warm); { uint32_t s = 12345; for (auto& x : noise) { s = s * 1664525u + 1013904223u; x = 0.25 * (((double) (s >> 8) / 8388608.0) - 1.0); } }
        auto no = r.run (noise, N + warm);
        int bestLag = 0; double bestErr = 1e300, bestGain = 1.0;
        for (int lag = 0; lag <= 512; ++lag)
        {
            double num = 0, den = 1e-300;
            for (size_t n = warm; n < warm + N; ++n) { num += no.l[n] * noise[n - (size_t) lag]; den += noise[n - (size_t) lag] * noise[n - (size_t) lag]; }
            const double g = num / den; double e = 0;
            for (size_t n = warm; n < warm + N; ++n) { const double d = no.l[n] - g * noise[n - (size_t) lag]; e += d * d; }
            const double rr = e / den; if (rr < bestErr) { bestErr = rr; bestLag = lag; bestGain = g; }
        }
        j << ",\"nullTest\":{\"input\":\"white noise at 0.25 peak\",\"bestAlignmentLagSamples\":" << bestLag << ",\"gainAtBestAlignmentDb\":" << ab::jnum (20.0 * std::log10 (std::abs (bestGain) + 1e-30)) << ",\"residualAfterGainAndLagDb\":" << ab::jnum (10.0 * std::log10 (bestErr + 1e-40)) << ",\"meaning\":\"how far the output is from a delayed, scaled copy of the input: very low = the plug-in is (nearly) transparent at its defaults\"}";
        rules.insert (rules.begin(), ab::makeRule ("non-finite output samples across all tests", (double) r.nonFinite, 0.0));
        j << ",\"nonFiniteSamples\":" << r.nonFinite << "}";
    }

    //==========================================================================
    // spectrum of a note with a 4-term Blackman-Harris window: amplitudes in dBFS from band powers
    struct Spectrum
    {
        std::vector<double> p;   // |X|^2, bins 0..N/2
        double binHz = SR / (double) N, sumW2 = 0;
        static constexpr long kLobe = 5;      // half-width, in bins, of a sine's main lobe with this window (the lobe itself is +-4)
        static Spectrum of (const std::vector<double>& x)
        {
            Spectrum s; std::vector<ab::cd> X (N);
            for (size_t n = 0; n < N; ++n)
            {
                const double ph = 2.0 * ab::kPi * (double) n / (double) N;
                const double w = 0.35875 - 0.48829 * std::cos (ph) + 0.14128 * std::cos (2.0 * ph) - 0.01168 * std::cos (3.0 * ph);
                s.sumW2 += w * w; X[n] = x[n] * w;
            }
            ab::fft (X);
            s.p.resize (N / 2 + 1); for (size_t k = 0; k <= N / 2; ++k) s.p[k] = std::norm (X[k]);
            return s;
        }
        double power (long lo, long hi) const { double s = 0; for (long k = std::max<long> (1, lo); k <= std::min<long> ((long) N / 2 - 1, hi); ++k) s += p[(size_t) k]; return s; }
        double ampDb (double pw) const { return 10.0 * std::log10 (4.0 * pw / ((double) N * sumW2) + 1e-40); }   // peak amplitude of a sine whose windowed power in the main lobe is pw (Parseval)
    };

    // What one steady note looks like in a Hann-windowed spectrum (N samples): where its fundamental really is, how loud, its harmonics, and how much energy sits OFF the whole multiples of the note
    // (aliasing, detune, modulation, noise). Public so the tests can check it against synthetic notes whose answers are known.
public:
    struct NoteMeasure { double fMeas = 0, cents = 0, levelDb = -400, thdDb = -400, nonDb = -400, bigDb = -400, bigHz = 0, bigCents = 0; int bigHarmonic = 1; bool steady = true; std::string harmonics; };
    static constexpr double kDetuneCents = 40.0;   // a strongest off-multiple line within this many cents of a harmonic reads as detune or modulation; further away it reads as aliasing or noise
    static NoteMeasure measureNote (const std::vector<double>& cap, double f0)
    {
        NoteMeasure m;
        const Spectrum S = Spectrum::of (cap);
        const double binHz = S.binHz, expBin = f0 / binHz;
        long lo = (long) std::floor (expBin * 0.97), hi = (long) std::ceil (expBin * 1.03); lo = std::max<long> (2, lo - 2);
        long kmax = lo; for (long k = lo; k <= hi; ++k) if (S.p[(size_t) k] > S.p[(size_t) kmax]) kmax = k;
        const double pf = S.power (kmax - Spectrum::kLobe, kmax + Spectrum::kLobe);
        { const double la = std::log (S.p[(size_t) kmax - 1] + 1e-300), lb = std::log (S.p[(size_t) kmax] + 1e-300), lc = std::log (S.p[(size_t) kmax + 1] + 1e-300); const double den = la - 2.0 * lb + lc; const double off = std::abs (den) > 1e-12 ? 0.5 * (la - lc) / den : 0.0; m.fMeas = ((double) kmax + juce::jlimit (-0.5, 0.5, off)) * binHz; }
        m.cents = m.fMeas > 0 ? 1200.0 * std::log2 (m.fMeas / f0) : 0.0;
        m.levelDb = S.ampDb (pf);
        std::vector<char> mask (N / 2 + 1, 0);
        for (long k = 0; k < 20 && k <= (long) N / 2; ++k) mask[(size_t) k] = 1;     // DC and the lowest bins are not counted
        for (long k = std::max<long> (0, kmax - Spectrum::kLobe); k <= std::min<long> ((long) N / 2, kmax + Spectrum::kLobe); ++k) mask[(size_t) k] = 1;
        std::ostringstream hs; double trueP = 0; int shown = 0;
        for (int mh = 2; (double) mh * f0 < SR * 0.5 - 8.0 * binHz; ++mh)
        {
            const double eb = (double) mh * m.fMeas / binHz; const long tol = 3 + (long) std::lround (eb * 0.003);
            const long bk = (long) std::lround (eb); long bb = bk; for (long k = bk - tol; k <= bk + tol; ++k) if (k >= 1 && k <= (long) N / 2 && S.p[(size_t) k] > S.p[(size_t) bb]) bb = k;
            const double ph = S.power (bb - Spectrum::kLobe, bb + Spectrum::kLobe); trueP += ph;
            for (long k = bk - tol - Spectrum::kLobe; k <= bk + tol + Spectrum::kLobe; ++k) if (k >= 0 && k <= (long) N / 2) mask[(size_t) k] = 1;
            if (shown < 8) { if (shown) hs << ","; hs << ab::jnum (10.0 * std::log10 (ph / pf + 1e-40)); ++shown; }
        }
        double nonP = 0, bigP = 0; long bigK = 0;
        for (long k = 1; k < (long) N / 2; ++k) if (! mask[(size_t) k]) { nonP += S.p[(size_t) k]; if (S.p[(size_t) k] > bigP) { bigP = S.p[(size_t) k]; bigK = k; } }
        m.nonDb = 10.0 * std::log10 (nonP / pf + 1e-40); m.bigDb = 10.0 * std::log10 (bigP / pf + 1e-40); m.thdDb = 10.0 * std::log10 (trueP / pf + 1e-40);
        m.bigHz = (double) bigK * binHz; m.harmonics = hs.str();
        m.bigHarmonic = std::max (1, (int) std::lround (m.bigHz / f0)); m.bigCents = m.bigHz > 0 ? 1200.0 * std::log2 (m.bigHz / ((double) m.bigHarmonic * f0)) : 0.0;
        m.steady = std::abs (rmsDb (cap, 0, N / 4) - rmsDb (cap, N - N / 4, N)) < 3.0;   // a decaying note smears its spectrum
        return m;
    }
private:

    static void instrumentSuite (Runner& r, std::ostringstream& j, std::vector<ab::Rule>& rules)
    {
        const size_t lead = (size_t) (0.4 * SR), hold = lead + N, total = hold + (size_t) (4.0 * SR);
        const int notes[] = { 36, 48, 60, 72, 84, 96, 108 };
        j << "\"instrument\":{\"method\":\"each note is held 1.77 s at velocity 100; a 1.36 s Blackman-Harris-windowed spectrum is taken after the first 0.4 s; then the note is released and the decay is timed\",\"notes\":[";
        bool first = true; double worstPitch = 0, worstNonHarm = -400; bool anyNote = false, anySound = false;
        for (int nn : notes)
        {
            r.reset();
            const double f0 = 440.0 * std::pow (2.0, ((double) nn - 69.0) / 12.0);
            std::vector<std::pair<size_t, juce::MidiMessage>> ev { { 0, juce::MidiMessage::noteOn (1, nn, (juce::uint8) 100) }, { hold, juce::MidiMessage::noteOff (1, nn) } };
            auto out = r.run ({}, total, ev);
            std::vector<double> cap (out.l.begin() + (long) lead, out.l.begin() + (long) (lead + N));
            const NoteMeasure nm = measureNote (cap, f0);
            const double fMeas = nm.fMeas, cents = nm.cents, levelDb = nm.levelDb, nonDb = nm.nonDb, bigDb = nm.bigDb, thdDb = nm.thdDb; const bool steady = nm.steady;
            // onset and release
            size_t onset = 0; for (size_t n = 0; n < total; ++n) if (std::abs (out.l[n]) > 1.0e-3) { onset = n; break; }
            const size_t w10 = (size_t) (0.01 * SR); double refDb = -400; for (size_t n = hold - 10 * w10; n + w10 <= hold; n += w10) refDb = std::max (refDb, rmsDb (out.l, n, n + w10));
            double releaseMs = -1; for (size_t n = hold; n + w10 <= total; n += w10) if (rmsDb (out.l, n, n + w10) < refDb - 60.0) { releaseMs = 1000.0 * (double) (n - hold) / SR; break; }
            if (! first) j << ","; first = false;
            j << "{\"note\":" << nn << ",\"expectedHz\":" << ab::jnum (f0) << ",\"measuredHz\":" << ab::jnum (fMeas) << ",\"pitchErrorCents\":" << ab::jnum (cents) << ",\"levelDbfs\":" << ab::jnum (levelDb)
              << ",\"harmonicsDbReFundamental\":[" << nm.harmonics << "],\"trueHarmonicsDb\":" << ab::jnum (thdDb) << ",\"nonHarmonicDb\":" << ab::jnum (nonDb) << ",\"largestNonHarmonicDb\":" << ab::jnum (bigDb) << ",\"largestNonHarmonicHz\":" << ab::jnum (nm.bigHz) << ",\"largestNonHarmonicNearHarmonic\":" << nm.bigHarmonic << ",\"largestNonHarmonicCentsFromIt\":" << ab::jnum (nm.bigCents) << ",\"reads\":" << ab::jstr (nm.nonDb < -60.0 ? "clean: nothing measurable off the whole multiples" : std::abs (nm.bigCents) <= kDetuneCents ? "detune or modulation: the strongest off-multiple line sits within 40 cents of a harmonic (a second oscillator, unison or chorus does this)" : "aliasing or noise: the strongest off-multiple line is more than 40 cents from any harmonic")
              << ",\"steady\":" << b (steady) << ",\"onsetMs\":" << ab::jnum (1000.0 * (double) onset / SR) << ",\"releaseTo60dBMs\":" << (releaseMs < 0 ? std::string ("null") : ab::jnum (releaseMs)) << "}";
            if (levelDb >= kSoundLimitDb) anySound = true;
            if (steady && levelDb > kSoundLimitDb && nn <= 96) { anyNote = true; worstPitch = std::max (worstPitch, std::abs (cents)); worstNonHarm = std::max (worstNonHarm, nonDb); }
        }
        j << "]";
        // velocity response on middle C
        j << ",\"velocity\":[";
        for (int vi = 0; vi < 3; ++vi)
        {
            const int vel[] = { 20, 64, 127 };
            r.reset();
            std::vector<std::pair<size_t, juce::MidiMessage>> ev { { 0, juce::MidiMessage::noteOn (1, 60, (juce::uint8) vel[vi]) }, { (size_t) (0.6 * SR), juce::MidiMessage::noteOff (1, 60) } };
            auto out = r.run ({}, (size_t) (0.8 * SR), ev);
            if (vi) j << ","; j << "{\"velocity\":" << vel[vi] << ",\"rmsDbfs\":" << ab::jnum (rmsDb (out.l, (size_t) (0.1 * SR), (size_t) (0.5 * SR))) << "}";
        }
        j << "]";
        // silence: no notes
        r.reset();
        auto sil = r.run ({}, (size_t) (2.0 * SR));
        const double noiseDb = rmsDb (sil.l, (size_t) SR, (size_t) (2.0 * SR));
        j << ",\"silence\":{\"noRmsDbfsWithNoNotes\":" << ab::jnum (noiseDb) << "}}";
        rules.insert (rules.begin(), ab::makeRule ("non-finite output samples across all tests", (double) r.nonFinite, 0.0));
        rules.push_back (ab::makeRule ("the plug-in makes sound (above -60 dBFS) for at least one note tested (0 = yes)", anySound ? 0.0 : 1.0, 0.0));
        rules.push_back (ab::makeRule ("noise with no notes (dBFS RMS)", noiseDb, kNoiseLimitDb));
        if (anyNote)
        {
            rules.push_back (ab::makeRule ("worst pitch error on steady notes 36-96 (cents)", worstPitch, kPitchLimitCents));
            rules.push_back (ab::makeRule ("worst energy off the note's whole multiples, steady notes 36-96 (dB re the note)", worstNonHarm, kNonHarmonicLimitDb));
        }
    }
};

} // namespace archie
