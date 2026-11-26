cmake_minimum_required(VERSION 3.24)
project(ArchieStudioHost VERSION 0.1.153 LANGUAGES CXX)
# Archie Studio host (desktop): loads the person's VST3 plugins, plays them, and hosts the Ask Archie web pages in a window with one shared transport.
# Uses the SAME JUCE tree and WebView2 SDK the Local Builder already provisions, so nothing new is downloaded to build it.
#   cmake -S desktop-host -B D:/AskArchieHostBuild -G "Visual Studio 17 2022" -A x64
#   cmake --build D:/AskArchieHostBuild --config Release
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Static Visual C++ runtime: the finished program needs no redistributable on the person's computer (scripts/build-desktop-studio-installer.mjs checks this).
set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded")
set(ARCHIE_LB "$ENV{LOCALAPPDATA}/AskArchie/LocalBuilder/deps" CACHE PATH "Local Builder dependency folder")
set(JUCE_WEBVIEW2_PACKAGE_LOCATION "${ARCHIE_LB}/WebView2" CACHE PATH "folder that contains Microsoft.Web.WebView2.*")
add_subdirectory("${ARCHIE_LB}/JUCE" juce)

juce_add_gui_app(ArchieStudio
  PRODUCT_NAME "Archie Studio"
  ICON_BIG "${CMAKE_CURRENT_SOURCE_DIR}/assets/archie-daw-512.png"
  COMPANY_NAME "Ask Archie"
  BUNDLE_ID "net.askarchie.studio"
  VERSION 0.1.153
  NEEDS_WEBVIEW2 TRUE
  MICROPHONE_PERMISSION_ENABLED FALSE)

juce_add_binary_data(ArchieStudioUI SOURCES ui/host.html tools/studio-keymap.json tools/studio-palette.json)

target_sources(ArchieStudio PRIVATE Source/Main.cpp Source/CrashGuard.cpp)
target_include_directories(ArchieStudio PRIVATE Source third_party third_party/asio)
target_compile_definitions(ArchieStudio PRIVATE
  JUCE_ASIO=1          # Steinberg ASIO, used under its GPLv3 option (third_party/asio): the program is GPLv3, see NOTICE.md
  JUCE_WEB_BROWSER=1
  ARCHIE_APP_VERSION="${PROJECT_VERSION}"
  JUCE_USE_WIN_WEBVIEW2=1
  JUCE_USE_WIN_WEBVIEW2_WITH_STATIC_LINKING=1
  JUCE_USE_CURL=0
  JUCE_PLUGINHOST_VST3=1
  JUCE_PLUGINHOST_AU=0
  JUCE_VST3_CAN_REPLACE_VST2=0
  JUCE_MODAL_LOOPS_PERMITTED=0)
target_link_libraries(ArchieStudio PRIVATE
  ArchieStudioUI
  juce::juce_audio_utils
  juce::juce_audio_processors
  juce::juce_audio_devices
  juce::juce_audio_formats
  juce::juce_dsp
  juce::juce_gui_extra
  juce::juce_cryptography
  juce::juce_recommended_config_flags
  juce::juce_recommended_warning_flags)
if (MSVC)
  target_compile_options(ArchieStudio PRIVATE /EHsc /MP)
endif()

# Offline checks of the master transport (no sound card, no window): build target transport_test, then run it.
juce_add_console_app(transport_test PRODUCT_NAME "transport_test")
target_sources(transport_test PRIVATE tests/transport_test.cpp)
target_include_directories(transport_test PRIVATE Source)
target_compile_definitions(transport_test PRIVATE JUCE_USE_CURL=0 JUCE_WEB_BROWSER=0 JUCE_STANDALONE_APPLICATION=1)
target_link_libraries(transport_test PRIVATE juce::juce_audio_basics juce::juce_recommended_config_flags)

# Checks of the Analyser's note measurement against synthetic notes with known answers: build target analyser_test, then run it.
juce_add_console_app(analyser_test PRODUCT_NAME "analyser_test")
target_sources(analyser_test PRIVATE tests/analyser_test.cpp)
target_include_directories(analyser_test PRIVATE Source)
target_compile_definitions(analyser_test PRIVATE JUCE_USE_CURL=0 JUCE_WEB_BROWSER=0 JUCE_STANDALONE_APPLICATION=1 JUCE_PLUGINHOST_VST3=1)
target_link_libraries(analyser_test PRIVATE juce::juce_audio_processors juce::juce_audio_basics juce::juce_recommended_config_flags)

# Offline checks of the mixer renderer (delay compensation, channel rules, thread-count independence): build target render_test, then run it.
juce_add_console_app(render_test PRODUCT_NAME "render_test")
target_sources(render_test PRIVATE tests/render_test.cpp)
set_target_properties(render_test PROPERTIES OUTPUT_NAME render_suite)
target_include_directories(render_test PRIVATE Source third_party)
if (MSVC)
  target_link_options(render_test PRIVATE /STACK:33554432)     # the offline checks build many engines as locals in one function: give them a 32 MB stack
endif()
target_compile_definitions(render_test PRIVATE JUCE_USE_CURL=0 JUCE_WEB_BROWSER=0 JUCE_STANDALONE_APPLICATION=1 JUCE_PLUGINHOST_VST3=1 JUCE_MODAL_LOOPS_PERMITTED=0)
target_link_libraries(render_test PRIVATE juce::juce_cryptography juce::juce_audio_utils juce::juce_audio_processors juce::juce_audio_devices juce::juce_audio_formats juce::juce_dsp juce::juce_gui_extra juce::juce_recommended_config_flags)
