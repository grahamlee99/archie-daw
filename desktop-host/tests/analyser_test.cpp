// Checks the Analyser's note measurement against SYNTHETIC notes whose answers are known (no plug-in involved): if the instrument figures can be wrong, these catch it.
// Build target analyser_test, then run it.
#include "../Source/Analyser.h"
#include <cstdio>

using archie::Analyser;
static int failures = 0, checks = 0;
#define CHECK(cond, msg) do { ++checks; if (!(cond)) { ++failures; std::printf("FAIL: %s  (line %d)\n", msg, __LINE__); } } while (0)

static std::vector<double> make (double f0, int kind, double amp = 0.5)
{
    const size_t N = Analyser::N; const double sr = Analyser::SR;
    std::vector<double> x (N, 0.0);
    for (size_t n = 0; n < N; ++n)
    {
        const double t = (double) n / sr; double v = 0;
        if (kind == 0) v = std::sin (2 * ab::kPi * f0 * t);                                                    // a pure sine
        else if (kind == 1) { const double ph = std::fmod (f0 * t, 1.0); v = 2.0 * ph - 1.0; }                  // a NAIVE sawtooth (aliases)
        else if (kind == 2) { for (int h = 1; (double) h * f0 < sr * 0.5; ++h) v += std::sin (2 * ab::kPi * f0 * h * t) / h; v *= 2.0 / ab::kPi * 0.9; }   // a BAND-LIMITED sawtooth
        else if (kind == 3) v = 0.5 * std::sin (2 * ab::kPi * f0 * t) + 0.5 * std::sin (2 * ab::kPi * f0 * 1.0116 * t);   // two oscillators, the second about 20 cents sharp
        x[n] = amp * v;
    }
    return x;
}

int main()
{
    const double notes[] = { 65.406, 130.813, 261.626, 523.251, 1046.502, 2093.005, 4186.009 };
    for (double f0 : notes)
    {
        const auto sine = Analyser::measureNote (make (f0, 0), f0);
        CHECK (std::abs (sine.cents) < 0.5, "a pure sine's pitch is measured to within half a cent");
        CHECK (std::abs (sine.levelDb - 20.0 * std::log10 (0.5)) < 0.7, "a pure sine's level is measured to within 0.7 dB (the scalloping of a Blackman-Harris window)");
        CHECK (sine.nonDb < -85.0, "a pure sine has no energy off its multiples");
        const auto band = Analyser::measureNote (make (f0, 2), f0);
        CHECK (std::abs (band.cents) < 0.5, "a band-limited saw's pitch is right");
        CHECK (band.nonDb < -60.0, "a band-limited saw has almost nothing off its multiples");
        const auto naive = Analyser::measureNote (make (f0, 1), f0);
        CHECK (std::abs (naive.cents) < 0.5, "a naive saw's pitch is right");
        if (f0 > 1000.0) CHECK (naive.nonDb > band.nonDb + 15.0, "a naive saw shows far more off-multiple (aliased) energy than a band-limited one at high pitch");
        if (f0 > 2000.0) CHECK (naive.nonDb > -25.0, "a naive saw at 2 kHz and above alias strongly");
        const auto det = Analyser::measureNote (make (f0, 3), f0);
        if (f0 > 500.0) CHECK (det.nonDb > -12.0, "two oscillators 20 cents apart put energy off the multiples (once they are further apart than one main lobe)");
        std::printf("  f0 %.1f: sine %+.2fc %.1fdB nonH %.0f | band-limited saw nonH %.1f | naive saw nonH %.1f | detuned pair nonH %.1f\n", f0, sine.cents, sine.levelDb, sine.nonDb, band.nonDb, naive.nonDb, det.nonDb);
    }
    std::printf("%d checks, %d failed\n", checks, failures);
    return failures == 0 ? 0 : 1;
}
