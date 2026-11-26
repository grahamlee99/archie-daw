# Third-party code in this folder

Both are MIT licensed (their licence files sit beside them) and are used only for audio time-stretching in Archie Studio (desktop), "warp".

- **signalsmith-stretch** (Signalsmith Audio, https://github.com/Signalsmith-Audio/signalsmith-stretch) - commit 88061fd4eda56a2353c5d65c74aa3f4ccf76f9da, version 1.3.2. Header-only pitch / time stretcher.
- **signalsmith-linear** (Signalsmith Audio, https://github.com/Signalsmith-Audio/linear) - commit 7bb1c8f36794d6980080ce817c91f8da8fe56a66. The FFT it needs (the `include/` forwarding folder of that repository is not copied: the stretch header includes `signalsmith-linear/stft.h`, which is this folder).

They were copied unchanged. Nothing here is built unless Warp.h is included. Attribution must stay with the licence files if Archie Studio is distributed.

- **asio** (Steinberg Media Technologies GmbH, ASIO SDK headers) - `asio.h`, `asiosys.h`, `iasiodrv.h` and their `LICENSE.txt`, copied unchanged from the JUCE 9.0.2 repository (modules/juce_audio_devices/native/asio). Used here under the GNU GPL version 3 option; see ../NOTICE.md.
