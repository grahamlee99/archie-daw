// Archie Studio host -- the TIME BASE.
// Time is measured in two ways that must always agree: BEATS (quarter notes, what music is written in) and SECONDS (what the sound card plays, and what audio files and timecode follow).
// The TEMPO MAP is the one thing that converts between them: a list of tempo and time-signature entries by beat position (a tempo change is an entry). Everything that needs "where in the block does this beat fall" or
// "which beat are we on after this many samples" asks the map, so a tempo change in the middle of a block, a clip, a clock pulse or a metronome click all land on exactly the right sample.
// This replaces the first version, which assumed one tempo for the whole song.
#pragma once
#include <algorithm>
#include <atomic>
#include <cmath>
#include <memory>
#include <vector>

namespace archie
{

struct TempoEntry { double beat = 0.0; double bpm = 120.0; int tsNum = 4; int tsDen = 4; };

class TempoMap
{
public:
    TempoMap() { e.push_back (TempoEntry{}); rebuild(); }
    explicit TempoMap (std::vector<TempoEntry> entries) : e (std::move (entries)) { normalise(); rebuild(); }

    // tempo changes are instant (a step at the entry's beat); a position before beat 0 (a count-in) runs at the first tempo
    double secondsAt (double beat) const
    {
        if (beat <= 0.0) return beat * 60.0 / e[0].bpm;
        const size_t i = indexForBeat (beat);
        return cumSec[i] + (beat - e[i].beat) * 60.0 / e[i].bpm;
    }
    double beatAt (double seconds) const
    {
        if (seconds <= 0.0) return seconds * e[0].bpm / 60.0;
        const size_t i = indexForSeconds (seconds);
        return e[i].beat + (seconds - cumSec[i]) * e[i].bpm / 60.0;
    }
    double bpmAt (double beat) const { return e[indexForBeat (beat)].bpm; }
    int tsNumAt (double beat) const { return e[indexForBeat (beat)].tsNum; }
    int tsDenAt (double beat) const { return e[indexForBeat (beat)].tsDen; }
    double barLengthBeatsAt (double beat) const { return (double) tsNumAt (beat) * 4.0 / (double) tsDenAt (beat); }
    const std::vector<TempoEntry>& entries() const { return e; }

private:
    size_t indexForBeat (double beat) const
    {
        size_t lo = 0, hi = e.size();                                   // last entry whose beat <= beat
        while (hi - lo > 1) { const size_t mid = (lo + hi) / 2; if (e[mid].beat <= beat) lo = mid; else hi = mid; }
        return lo;
    }
    size_t indexForSeconds (double sec) const
    {
        size_t lo = 0, hi = e.size();
        while (hi - lo > 1) { const size_t mid = (lo + hi) / 2; if (cumSec[mid] <= sec) lo = mid; else hi = mid; }
        return lo;
    }
    void normalise()
    {
        if (e.empty()) e.push_back (TempoEntry{});
        for (auto& x : e)
        {
            x.beat = std::max (0.0, x.beat); x.bpm = std::min (400.0, std::max (20.0, x.bpm));
            x.tsNum = std::min (32, std::max (1, x.tsNum)); if (x.tsDen != 2 && x.tsDen != 4 && x.tsDen != 8 && x.tsDen != 16) x.tsDen = 4;
        }
        std::stable_sort (e.begin(), e.end(), [] (const TempoEntry& a, const TempoEntry& b) { return a.beat < b.beat; });
        e[0].beat = 0.0;
        std::vector<TempoEntry> out;                                    // two entries at the same beat: the later one wins
        for (const auto& x : e) { if (! out.empty() && std::abs (out.back().beat - x.beat) < 1.0e-9) out.back() = x, out.back().beat = out.size() == 1 ? 0.0 : x.beat; else out.push_back (x); }
        e.swap (out);
    }
    void rebuild()
    {
        cumSec.assign (e.size(), 0.0);
        for (size_t i = 1; i < e.size(); ++i) cumSec[i] = cumSec[i - 1] + (e[i].beat - e[i - 1].beat) * 60.0 / e[i - 1].bpm;
    }

    std::vector<TempoEntry> e;
    std::vector<double> cumSec;
};

// A value the audio thread reads without ever waiting. The message (or MIDI) thread publishes a NEW immutable value and the audio thread picks it up on its next block;
// the old values are kept alive for the last KeepN publishes (the template argument: small for big things like audio files), which is far longer than any audio block, so a block that is still reading one is never left with a dangling pointer.
// publish() is not thread-safe against another publish(): callers hold their own lock.
template <typename T, size_t KeepN = 64>
class Snapshot
{
public:
    explicit Snapshot (std::unique_ptr<T> first) { cur.store (first.get(), std::memory_order_release); keep.push_back (std::move (first)); }
    const T* load() const { return cur.load (std::memory_order_acquire); }
    void publish (std::unique_ptr<T> next)
    {
        const T* p = next.get();
        keep.push_back (std::move (next));
        cur.store (p, std::memory_order_release);
        while (keep.size() > kKeep) keep.erase (keep.begin());
    }
private:
    static constexpr size_t kKeep = KeepN;
    std::atomic<const T*> cur { nullptr };
    std::vector<std::unique_ptr<T>> keep;
};

} // namespace archie
