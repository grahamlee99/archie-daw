// Archie Studio host -- TAP TEMPO: the tempo a person taps out with a button or a key.
// The times come from the program's own clock (not from the page, so a slow page cannot spoil them). The last eight taps are used; a gap of more than two seconds starts again; an odd tap
// (a double tap, a late one) is thrown out when it is more than a seventh away from the typical gap; the tempo is the average of the gaps that are left. The first two taps say nothing yet.
// FILTERING/MONITORING MECHANISM: this reads the TIMES of button presses and decides one number. It does not judge rhythm, swing or whether the person meant a half or double tempo
// (it says what was tapped); it is adjusted by nothing but the person tapping again, and any tempo box can still be typed into.
#pragma once
#include <algorithm>
#include <vector>

namespace archie
{

class TapTempo
{
public:
    // Register a tap at `nowMs`. Returns the tempo in beats a minute, or 0 if there are not enough taps yet.
    double tap (double nowMs)
    {
        if (! times.empty() && nowMs - times.back() > kResetMs) times.clear();
        times.push_back (nowMs);
        if (times.size() > (size_t) kKeep) times.erase (times.begin());
        if (times.size() < 3) return 0.0;
        std::vector<double> gaps; for (size_t i = 1; i < times.size(); ++i) gaps.push_back (times[i] - times[i - 1]);
        std::vector<double> sorted = gaps; std::sort (sorted.begin(), sorted.end()); const double med = sorted[sorted.size() / 2];
        double sum = 0.0; int n = 0; for (double g : gaps) if (std::abs (g - med) <= 0.15 * med) { sum += g; ++n; }
        if (n == 0 || sum <= 0.0) return 0.0;
        const double bpm = 60000.0 / (sum / (double) n);
        return std::min (400.0, std::max (20.0, bpm));
    }
    void reset() { times.clear(); }
    int count() const { return (int) times.size(); }
private:
    static constexpr double kResetMs = 2000.0; static constexpr int kKeep = 8;
    std::vector<double> times;
};

} // namespace archie
