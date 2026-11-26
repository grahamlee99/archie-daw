// COPY of the lab analyser/core/ab_core.h (the lab, in D:/AskArchieDspLab, keeps the master and its selftest). Do not edit here: change the lab copy and re-copy.
// Archie Analyser: the measurement core (offline spike, 2026-11-20). Pure C++17, NO JUCE dependency, double precision analysis.
// Brief: D:\GitHub\askarchie\admin-docs\archie-audio-lab-brief-2026-11-20.md (honesty rules: every number is measured or computed from a stated formula; every result carries its conditions, provenance and validity flags).
// What it does: turns an impulse response (what a block did to a unit impulse) into magnitude, phase and group delay, compares that with an IDEAL response given as a function, and finds the peak and the -3 dB point. It never invents a score: a verdict is computed from measured errors against tolerances that are written next to it.
#pragma once
#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstdio>
#include <functional>
#include <sstream>
#include <string>
#include <vector>

namespace ab {

using cd = std::complex<double>;
constexpr double kPi = 3.14159265358979323846;

inline bool isPow2(size_t n) { return n && !(n & (n - 1)); }

// In-place iterative radix-2 FFT. inverse = true does the unscaled inverse.
inline void fft(std::vector<cd>& a, bool inverse = false) {
  const size_t n = a.size();
  for (size_t i = 1, j = 0; i < n; ++i) {
    size_t bit = n >> 1;
    for (; j & bit; bit >>= 1) j ^= bit;
    j ^= bit;
    if (i < j) std::swap(a[i], a[j]);
  }
  for (size_t len = 2; len <= n; len <<= 1) {
    const double ang = 2.0 * kPi / (double) len * (inverse ? 1.0 : -1.0);
    const cd wl(std::cos(ang), std::sin(ang));
    for (size_t i = 0; i < n; i += len) {
      cd w(1.0, 0.0);
      for (size_t k = 0; k < len / 2; ++k) {
        const cd u = a[i + k], v = a[i + k + len / 2] * w;
        a[i + k] = u + v;
        a[i + k + len / 2] = u - v;
        w *= wl;
      }
    }
  }
}

// What the block did, as a response over frequency. Bin k is at k * fs / N, k = 1 .. N/2 (DC is left out: a log-frequency view has no use for it).
struct Response {
  double fs = 0;
  std::vector<double> freqHz, magDb, phaseRad, groupDelaySamples;
  std::vector<cd> h;   // complex response at each bin
};

// Frequency response from a measured impulse response (length a power of two). Validity: tailDb says how much of the energy is in the last eighth of the record (a response that has not died away is cut off and its spectrum is smeared).
struct Validity {
  bool impulseDecayed = false;
  double tailDb = 0;
  bool finite = true;
  size_t length = 0;
};

inline Response responseFromImpulse(const std::vector<double>& imp, double fs, Validity* v = nullptr) {
  const size_t N = imp.size();
  Response r;
  r.fs = fs;
  if (!isPow2(N)) { if (v) v->finite = false; return r; }
  std::vector<cd> X(N);
  bool finite = true;
  for (size_t i = 0; i < N; ++i) { X[i] = imp[i]; if (!std::isfinite(imp[i])) finite = false; }
  if (v) {
    double total = 0, tail = 0;
    for (size_t i = 0; i < N; ++i) { total += imp[i] * imp[i]; if (i >= N - N / 8) tail += imp[i] * imp[i]; }
    v->tailDb = total > 0 ? 10.0 * std::log10(tail / total + 1e-300) : 0.0;
    v->impulseDecayed = v->tailDb < -80.0;
    v->finite = finite;
    v->length = N;
  }
  if (!finite) return r;
  fft(X);
  const size_t K = N / 2;
  r.freqHz.resize(K); r.magDb.resize(K); r.phaseRad.resize(K); r.groupDelaySamples.assign(K, 0.0); r.h.resize(K);
  double prev = 0.0, offset = 0.0;
  for (size_t k = 1; k <= K; ++k) {
    const cd Xk = X[k % N];
    r.h[k - 1] = Xk;
    r.freqHz[k - 1] = (double) k * fs / (double) N;
    r.magDb[k - 1] = 20.0 * std::log10(std::abs(Xk) + 1e-300);
    double ph = std::arg(Xk);
    if (k > 1) { double d = ph + offset - prev; while (d > kPi) { offset -= 2 * kPi; d -= 2 * kPi; } while (d < -kPi) { offset += 2 * kPi; d += 2 * kPi; } }
    r.phaseRad[k - 1] = ph + offset;
    prev = r.phaseRad[k - 1];
  }
  const double dw = 2.0 * kPi / (double) N;   // radians per sample per bin
  for (size_t i = 1; i + 1 < K; ++i) r.groupDelaySamples[i] = -(r.phaseRad[i + 1] - r.phaseRad[i - 1]) / (2.0 * dw);
  return r;
}

// An ideal response is a function of the digital frequency in radians per sample (omega = 2 pi f / fs).
using IdealFn = std::function<cd(double /*omega*/)>;

struct Compare {
  double maxMagErrDb = 0, maxMagErrAtHz = 0, rmsMagErrDb = 0;
  double maxPhaseErrDeg = 0, maxPhaseErrAtHz = 0;
  size_t bins = 0;
};
// Compared over fLoHz .. fHiHz only (the band is part of the result, never hidden). Phase error is the angle of measured / ideal, wrapped, so a whole-cycle offset is not an error.
// floorDb (optional): bins where the IDEAL response is below this level are not compared. Deep in a stop band both filters are far below anything audible and a dB error there says nothing; the floor is part of the stated conditions, never hidden.
inline Compare compare(const Response& m, const IdealFn& ideal, double fLoHz, double fHiHz, double floorDb = -1e9) {
  Compare c;
  double sum2 = 0;
  for (size_t i = 0; i < m.freqHz.size(); ++i) {
    const double f = m.freqHz[i];
    if (f < fLoHz || f > fHiHz) continue;
    const cd id = ideal(2.0 * kPi * f / m.fs);
    if (20.0 * std::log10(std::abs(id) + 1e-300) < floorDb) continue;
    const double magErr = std::abs(m.magDb[i] - 20.0 * std::log10(std::abs(id) + 1e-300));
    const double phErr = std::abs(std::arg(m.h[i] / id)) * 180.0 / kPi;
    if (magErr > c.maxMagErrDb) { c.maxMagErrDb = magErr; c.maxMagErrAtHz = f; }
    if (phErr > c.maxPhaseErrDeg) { c.maxPhaseErrDeg = phErr; c.maxPhaseErrAtHz = f; }
    sum2 += magErr * magErr;
    ++c.bins;
  }
  c.rmsMagErrDb = c.bins ? std::sqrt(sum2 / (double) c.bins) : 0.0;
  return c;
}

struct Peak { double freqHz = 0, magDb = 0; };
// Highest point of the magnitude curve between fLoHz and fHiHz, refined with a parabola through its neighbours.
inline Peak findPeak(const Response& m, double fLoHz, double fHiHz) {
  Peak p; p.magDb = -1e300;
  size_t best = 0; bool any = false;
  for (size_t i = 1; i + 1 < m.freqHz.size(); ++i) {
    if (m.freqHz[i] < fLoHz || m.freqHz[i] > fHiHz) continue;
    if (m.magDb[i] > p.magDb) { p.magDb = m.magDb[i]; best = i; any = true; }
  }
  if (!any) return Peak{};
  const double a = m.magDb[best - 1], b = m.magDb[best], c = m.magDb[best + 1];
  const double den = a - 2 * b + c;
  const double off = std::abs(den) > 1e-12 ? 0.5 * (a - c) / den : 0.0;
  p.freqHz = m.freqHz[best] + off * (m.freqHz[best + 1] - m.freqHz[best]);
  p.magDb = b - 0.25 * (a - c) * off;
  return p;
}

// First frequency (searching upward from fFromHz) where the magnitude falls below `levelDb`; linear interpolation. 0 when it never does.
inline double firstCrossingBelow(const Response& m, double levelDb, double fFromHz) {
  for (size_t i = 1; i < m.freqHz.size(); ++i) {
    if (m.freqHz[i] < fFromHz) continue;
    if (m.magDb[i - 1] >= levelDb && m.magDb[i] < levelDb) {
      const double t = (m.magDb[i - 1] - levelDb) / (m.magDb[i - 1] - m.magDb[i]);
      return m.freqHz[i - 1] + t * (m.freqHz[i] - m.freqHz[i - 1]);
    }
  }
  return 0.0;
}

// A verdict is a computed fact: each rule is a measured number, a limit and whether it was met. The rules are printed with the result, so nothing is a free-hand label.
struct Rule { std::string name; double value; double limit; bool pass; };
inline Rule makeRule(const std::string& name, double value, double limit) { return Rule{name, value, limit, std::isfinite(value) && value <= limit}; }
inline bool allPass(const std::vector<Rule>& rules) { for (const auto& r : rules) if (!r.pass) return false; return !rules.empty(); }

// --- a very small JSON writer (numbers that are not finite are written as null) ---
inline std::string jnum(double v) {
  if (!std::isfinite(v)) return "null";
  char buf[48]; std::snprintf(buf, sizeof buf, "%.6g", v); return buf;
}
inline std::string jstr(const std::string& s) {
  std::string o = "\"";
  for (char ch : s) { if (ch == '"' || ch == '\\') { o += '\\'; o += ch; } else if (ch == '\n') o += "\\n"; else o += ch; }
  return o + "\"";
}
inline std::string jrules(const std::vector<Rule>& rules) {
  std::ostringstream o; o << "[";
  for (size_t i = 0; i < rules.size(); ++i) {
    if (i) o << ",";
    o << "{\"rule\":" << jstr(rules[i].name) << ",\"measured\":" << jnum(rules[i].value) << ",\"limit\":" << jnum(rules[i].limit) << ",\"met\":" << (rules[i].pass ? "true" : "false") << "}";
  }
  o << "]"; return o.str();
}
// the curve, sampled at `points` log-spaced frequencies (nearest bin), for the views in the concept images
inline std::string jcurve(const Response& m, int points, double fLoHz, double fHiHz) {
  std::ostringstream o; o << "{\"freqHz\":[";
  std::vector<size_t> idx;
  for (int p = 0; p < points; ++p) {
    const double f = fLoHz * std::pow(fHiHz / fLoHz, (double) p / (double) (points - 1));
    size_t best = 0; double bd = 1e300;
    for (size_t i = 0; i < m.freqHz.size(); ++i) { const double d = std::abs(std::log(m.freqHz[i] / f)); if (d < bd) { bd = d; best = i; } }
    idx.push_back(best);
  }
  auto arr = [&](const std::vector<double>& v, double mul) { for (size_t i = 0; i < idx.size(); ++i) { if (i) o << ","; o << jnum(v[idx[i]] * mul); } };
  arr(m.freqHz, 1.0); o << "],\"magDb\":["; arr(m.magDb, 1.0);
  o << "],\"phaseDeg\":["; arr(m.phaseRad, 180.0 / kPi);
  o << "],\"groupDelaySamples\":["; arr(m.groupDelaySamples, 1.0); o << "]}";
  return o.str();
}

// ---------------- harmonics and aliasing (step 2) ----------------
// A coherent tone: the input sine has EXACTLY k cycles in the N captured samples (k odd, so no harmonic of it, true or folded, lands on another component's bin for any order below N). Then every component of a periodic output sits on one exact FFT bin and no window is needed.
// Harmonic m of a tone at bin k is at bin m*k. While m*k < N/2 it is a TRUE harmonic; beyond Nyquist it folds back to bin |m*k mod N| (mirrored into 0..N/2): that is an ALIASED component, and it is counted separately. Everything else (bins that are neither the tone, a harmonic nor a folded harmonic) is "other" (noise, modulation, leakage).
struct Harmonics {
  double fundamentalDb = 0;     // level of the tone (dB re full scale 1.0, peak)
  double thdDb = -400;          // true harmonics 2..maxHarm, dB re the tone
  double aliasDb = -400;        // folded harmonics, dB re the tone
  double otherDb = -400;        // everything else except DC, dB re the tone
  double evenDb = -400, oddDb = -400;   // true harmonics split by parity, dB re the tone (the character of a non-linearity)
  std::vector<double> harmonicDb;       // index m-2 = harmonic m (true or folded), dB re the tone
  std::vector<int> folded;              // 1 where harmonic m was folded (aliased)
  bool coherent = true;                 // false when much energy sits outside the tone and its harmonics (the tone was not periodic in the record)
};
inline Harmonics analyseTone(const std::vector<double>& x, double /*fs*/, size_t k, int maxHarm = 60) {
  Harmonics H;
  const size_t N = x.size();
  if (!isPow2(N) || k < 1 || k >= N / 2) { H.coherent = false; return H; }
  std::vector<cd> X(N);
  for (size_t i = 0; i < N; ++i) X[i] = x[i];
  fft(X);
  const size_t Kmax = N / 2;
  auto pw = [&](size_t b) { return std::norm(X[b]); };
  const double pf = pw(k);
  if (pf <= 0) { H.coherent = false; return H; }
  std::vector<char> used(Kmax + 1, 0);
  used[0] = 1; used[k] = 1;
  double trueP = 0, aliasP = 0, evenP = 0, oddP = 0;
  for (int m = 2; m <= maxHarm; ++m) {
    size_t b = (size_t) m * k % N;
    const bool fold = (size_t) m * k >= Kmax;
    if (b > Kmax) b = N - b;
    if (b == 0 || b > Kmax) { H.harmonicDb.push_back(-400); H.folded.push_back(fold ? 1 : 0); continue; }
    const double p = pw(b);
    used[b] = 1;
    H.harmonicDb.push_back(10.0 * std::log10(p / pf + 1e-40));
    H.folded.push_back(fold ? 1 : 0);
    if (fold) aliasP += p; else { trueP += p; if (m % 2 == 0) evenP += p; else oddP += p; }
  }
  double otherP = 0;
  for (size_t b = 1; b <= Kmax; ++b) if (!used[b]) otherP += pw(b);
  H.fundamentalDb = 20.0 * std::log10(2.0 * std::sqrt(pf) / (double) N + 1e-300);
  H.thdDb = 10.0 * std::log10(trueP / pf + 1e-40);
  H.aliasDb = 10.0 * std::log10(aliasP / pf + 1e-40);
  H.otherDb = 10.0 * std::log10(otherP / pf + 1e-40);
  H.evenDb = 10.0 * std::log10(evenP / pf + 1e-40);
  H.oddDb = 10.0 * std::log10(oddP / pf + 1e-40);
  H.coherent = H.otherDb < -20.0;   // a tone that is not periodic in the record smears its energy into the other bins
  return H;
}
inline std::string jharm(const Harmonics& h) {
  std::ostringstream o;
  o << "{\"fundamentalDb\":" << jnum(h.fundamentalDb) << ",\"thdDb\":" << jnum(h.thdDb) << ",\"aliasDb\":" << jnum(h.aliasDb) << ",\"otherDb\":" << jnum(h.otherDb) << ",\"evenHarmonicsDb\":" << jnum(h.evenDb) << ",\"oddHarmonicsDb\":" << jnum(h.oddDb) << ",\"coherent\":" << (h.coherent ? "true" : "false") << ",\"harmonicDb\":[";
  for (size_t i = 0; i < h.harmonicDb.size() && i < 12; ++i) { if (i) o << ","; o << jnum(h.harmonicDb[i]); }
  o << "]}";
  return o.str();
}

}  // namespace ab
