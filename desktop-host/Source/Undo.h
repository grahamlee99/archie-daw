// Archie Studio host -- undo and redo.
// A plain history of edits: each entry knows how to put things back (undo) and how to do them again (redo). Nothing here knows about audio.
// Message thread only (the bridge, the service tick): the audio thread never touches it.
//
// Coalescing: dragging a fader makes dozens of tiny edits. Edits with the same key that arrive within a short time of each other become ONE entry:
// it keeps the first edit's undo (the value before the drag) and the last edit's redo (the value after it), so one Ctrl+Z puts the fader back where it started.
#pragma once
#include <functional>
#include <string>
#include <vector>
#include <chrono>

namespace archie
{

class UndoStack
{
public:
    struct Entry
    {
        std::string label, key;
        double timeMs = 0.0;
        std::function<void()> undo, redo;
    };

    // How long (ms) two edits with the same key are still "the same gesture". The clock can be replaced (tests use a fake one).
    double coalesceMs = 700.0;
    std::function<double()> clock = [] { using namespace std::chrono; return duration<double, std::milli> (steady_clock::now().time_since_epoch()).count(); };

    // windowMs: how long this key stays "the same gesture" (default coalesceMs). A whole recording take passes a huge window, so every pass of a loop take is one entry.
    void record (const std::string& label, const std::string& key, std::function<void()> undoFn, std::function<void()> redoFn, double windowMs = -1.0)
    {
        if (applying) return;                         // the undo and redo functions must not record themselves
        const double now = clock();
        redoStack.clear();
        const double win = windowMs >= 0.0 ? windowMs : coalesceMs;
        if (! key.empty() && ! undoStack.empty() && undoStack.back().key == key && now - undoStack.back().timeMs <= win)
        {
            undoStack.back().redo = std::move (redoFn);   // keep the first edit's undo: the state before the whole gesture
            undoStack.back().timeMs = now;
            return;
        }
        undoStack.push_back ({ label, key, now, std::move (undoFn), std::move (redoFn) });
        if (undoStack.size() > kMaxDepth) undoStack.erase (undoStack.begin());
    }

    bool undo()
    {
        if (undoStack.empty()) return false;
        Entry e = std::move (undoStack.back()); undoStack.pop_back();
        applying = true; if (e.undo) e.undo(); applying = false;
        e.key.clear();                                // an undone entry is never merged into
        redoStack.push_back (std::move (e));
        return true;
    }
    bool redo()
    {
        if (redoStack.empty()) return false;
        Entry e = std::move (redoStack.back()); redoStack.pop_back();
        applying = true; if (e.redo) e.redo(); applying = false;
        e.key.clear();
        undoStack.push_back (std::move (e));
        return true;
    }
    // The history can no longer be trusted (a new project was opened, or something happened that cannot be put back): forget it rather than risk a wrong undo.
    void clear() { undoStack.clear(); redoStack.clear(); }

    bool canUndo() const { return ! undoStack.empty(); }
    bool canRedo() const { return ! redoStack.empty(); }
    std::string undoLabel() const { return undoStack.empty() ? std::string() : undoStack.back().label; }
    std::string redoLabel() const { return redoStack.empty() ? std::string() : redoStack.back().label; }
    size_t depth() const { return undoStack.size(); }
    bool isApplying() const { return applying; }

private:
    static constexpr size_t kMaxDepth = 300;
    std::vector<Entry> undoStack, redoStack;
    bool applying = false;
};

} // namespace archie
