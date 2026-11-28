// SIDE-CHAIN PROBE: the DAW engine with a REAL plug-in. Loads a built Bagri Bass Banger (or any plug-in with an audio input or a side-chain bus), holds a note, and feeds its side-chain from another
// track and from a tempo pulse. Not a build gate (it needs a built plug-in). Usage: sidechain_probe <path to the .vst3>
#include "../Source/Engine.h"
#include <cstdio>
#include <cmath>
#include <vector>
using namespace archie;
static int fails = 0, checks = 0;
#define CHECK(c, msg) do { ++checks; if (! (c)) { ++fails; std::printf ("FAIL: %s\n", msg); } else std::printf ("  ok: %s\n", msg); } while (0)
static const double SR = 48000.0; static const int BS = 256;
static std::vector<float> run (HostEngine& e, int blocks) { std::vector<float> out, l ((size_t) BS), r ((size_t) BS); float* o[2] = { l.data(), r.data() }; for (int b = 0; b < blocks; ++b) { e.processOffline (nullptr, 0, o, 2, BS); out.insert (out.end(), l.begin(), l.end()); } return out; }
static double rms (const std::vector<float>& v, int a, int b) { double s = 0; int n = 0; for (int i = a; i < b && i < (int) v.size(); ++i, ++n) s += (double) v[(size_t) i] * v[(size_t) i]; return std::sqrt (s / std::max (1, n)); }

int main (int argc, char** argv)
{
    juce::ScopedJuceInitialiser_GUI gui; std::setvbuf (stdout, nullptr, _IONBF, 0);
    if (argc < 2) { std::printf ("usage: sidechain_probe <plugin.vst3>\n"); return 2; }
    juce::VST3PluginFormat format; juce::OwnedArray<juce::PluginDescription> types; format.findAllTypesForFile (types, juce::String (argv[1]));
    if (types.isEmpty()) { std::printf ("could not read the plug-in\n"); return 2; }
    HostEngine e; e.prepareOffline (SR, BS, 0); e.transport.setBpm (120.0);
    bool done = false; juce::String error;
    e.addPluginAsync (0, *types[0], {}, [&] (juce::String err) { error = err; done = true; });
    for (int i = 0; i < 600 && ! done; ++i) { juce::MessageManager::getInstance()->runDispatchLoopUntil (20); }
    if (! done || error.isNotEmpty() || e.trackAt (0)->slots.empty()) { std::printf ("could not load: %s\n", error.toRawUTF8()); return 2; }
    auto& slot = *e.trackAt (0)->slots[0]; auto& plug = *slot.proc;
    std::printf ("loaded %s (instrument %d, side-chain way %d)\n", plug.getName().toRawUTF8(), (int) slot.desc.isInstrument, slot.scBus);
    auto param = [&] (const juce::String& name, float norm) { for (auto* q : plug.getParameters()) if (q->getName (200) == name) { q->setValueNotifyingHost (norm); return true; } return false; };
    CHECK (slot.scBus >= 0, "the DAW sees a side-chain input on it");
    CHECK (param ("Side Chain Duck", 1.0f) && param ("Side Chain Sensitivity", 1.0f) && param ("Duck Depth", 1.0f) && param ("Sequencer", 0.0f), "its side-chain control is switched on");
    e.noteOn (0, 45, 100);
    auto level = [&] { run (e, 150); auto o = run (e, 150); return rms (o, 0, (int) o.size()); };
    const double alone = level();
    std::printf ("  held note, nothing feeding it: %.3f\n", alone); CHECK (alone > 0.05, "the bass sounds");
    e.trackAt (2)->source.store (2); e.editChainParam (2, "mute", true);
    CHECK (e.editSlotSide (0, 0, "key", e.trackAt (2)->uid.load()), "a track can be chosen to feed it");
    e.noteOn (0, 45, 100); const double fed = level();
    std::printf ("  fed by a muted track: %.3f\n", fed); CHECK (fed < alone * 0.35, "the bass ducks to that track, even muted");
    e.editSlotSide (0, 0, "hi", 60.0); const double filtered = level(); std::printf ("  the same with a 60 Hz High cut on a 440 Hz trigger: %.3f\n", filtered); CHECK (filtered > alone * 0.7, "the High cut takes the trigger away and the bass returns");
    e.editSlotSide (0, 0, "hi", 20000.0);
    CHECK (e.editSlotSide (0, 0, "key", -1), "a 4/4 pulse can feed it");
    run (e, 100); const double waiting = level(); CHECK (waiting > alone * 0.7, "stopped: the pulse leaves it alone");
    e.transport.locate (0.0); e.transport.play(); e.noteOn (0, 45, 100); auto o = run (e, 760);
    double on = 0, between = 0; for (int k = 2; k < 7; ++k) { on += rms (o, k * 24000 + 700, k * 24000 + 2600); between += rms (o, k * 24000 + 16000, k * 24000 + 23500); }
    std::printf ("  playing with the pulse: on the beat %.3f, between beats %.3f\n", on / 5, between / 5);
    CHECK (on < between * 0.5, "the bass pumps on every beat");
    e.transport.stop(); CHECK (e.editSlotSide (0, 0, "key", 0), "and Off returns it to normal"); e.noteOn (0, 45, 100); CHECK (level() > alone * 0.7, "the bass is back");
    e.noteOff (0, 45);
    std::printf ("%d checks, %d failed\n", checks, fails);
    return fails != 0;
}
