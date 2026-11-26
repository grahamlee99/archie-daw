; Inno Setup script for "Archie Studio" (desktop). Compiled by scripts/build-desktop-studio-installer.mjs, which passes AppVersion, SourceDir (the staged folder), OutDir and ConsentFile on the command line.
; Rules this script keeps: per-user install, no administrator, no start at login, nothing downloaded by the installer, saved settings kept on uninstall unless the person ticks the box.

#ifndef AppVersion
  #error AppVersion must be passed in (ISCC /DAppVersion=...)
#endif
#ifndef SourceDir
  #error SourceDir must be passed in
#endif
#ifndef OutDir
  #define OutDir "out"
#endif
#ifndef ConsentFile
  #define ConsentFile "consent.txt"
#endif

[Setup]
AppId={{B7D14E52-3C9A-4F68-A1E0-5D2C8B91F4A6}
AppName=Archie DAW
AppVersion={#AppVersion}
AppVerName=Archie DAW {#AppVersion}
AppPublisher=Ask Archie (askarchie.net)
AppPublisherURL=https://askarchie.net
AppSupportURL=https://askarchie.net
AppUpdatesURL=https://askarchie.net
VersionInfoVersion={#AppVersion}
VersionInfoCompany=Ask Archie (askarchie.net)
VersionInfoDescription=Archie DAW setup
DefaultDirName={localappdata}\AskArchie\StudioApp
DefaultGroupName=Ask Archie
DisableProgramGroupPage=yes
DisableDirPage=auto
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
OutputDir={#OutDir}
OutputBaseFilename=ArchieStudio-Setup-{#AppVersion}
Compression=lzma2/max
SolidCompression=yes
WizardStyle=modern
InfoBeforeFile={#ConsentFile}
UninstallDisplayName=Archie DAW
UninstallDisplayIcon={app}\ArchieStudio.exe
SetupIconFile={#SourceDir}\..\..\assets\archie-daw.ico
CloseApplications=yes
RestartApplications=no
MinVersion=10.0

[Tasks]
Name: "desktopicon"; Description: "Put a shortcut to Archie DAW on my Desktop"; GroupDescription: "Shortcuts:"

[Files]
Source: "{#SourceDir}\*"; DestDir: "{app}"; Flags: recursesubdirs createallsubdirs ignoreversion comparetimestamp

[Icons]
Name: "{group}\Archie DAW"; Filename: "{app}\ArchieStudio.exe"; WorkingDir: "{app}"; Comment: "Play and test your VST plug-ins, with Ask Archie inside"
Name: "{group}\Archie DAW manual"; Filename: "{app}\manual\Archie-DAW-Manual.html"; Comment: "How to use Archie DAW"
Name: "{group}\Uninstall Archie DAW"; Filename: "{uninstallexe}"
Name: "{autodesktop}\Archie DAW"; Filename: "{app}\ArchieStudio.exe"; WorkingDir: "{app}"; Tasks: desktopicon

[Run]
Filename: "{app}\ArchieStudio.exe"; Description: "Open Archie DAW now"; WorkingDir: "{app}"; Flags: postinstall nowait skipifsilent

[InstallDelete]
; the program used to be called Archie Studio: take away its old shortcuts so an update does not leave two
Type: files; Name: "{group}\Archie Studio.lnk"
Type: files; Name: "{group}\Uninstall Archie Studio.lnk"
Type: files; Name: "{autodesktop}\Archie Studio.lnk"
Type: files; Name: "{group}\Archie Studio manual.lnk"

[UninstallDelete]
Type: filesandordirs; Name: "{app}"

[Code]
// Archie DAW shows web pages through Microsoft's WebView2 runtime. Windows 11 and most Windows 10 computers already have it (it comes with Microsoft Edge).
// If it is missing this says so plainly and offers Microsoft's own download page; nothing is downloaded by this installer.
function WebView2Present(): Boolean;
var
  V: String;
begin
  Result := RegQueryStringValue(HKLM, 'SOFTWARE\WOW6432Node\Microsoft\EdgeUpdate\Clients\{F3017226-FE2A-4295-8BDF-00C3A9A7E4C5}', 'pv', V)
         or RegQueryStringValue(HKCU, 'SOFTWARE\Microsoft\EdgeUpdate\Clients\{F3017226-FE2A-4295-8BDF-00C3A9A7E4C5}', 'pv', V);
  if Result then Result := (V <> '') and (V <> '0.0.0.0');
end;

function InitializeSetup(): Boolean;
var
  Code: Integer;
begin
  Result := True;
  if (not WebView2Present()) and (not WizardSilent) then
    if MsgBox('Archie DAW needs the Microsoft Edge WebView2 runtime, and it was not found on this computer.' + #13#10 + #13#10 + 'Choose Yes to open Microsoft''s download page for it (install it, then run this setup again), or No to install Archie Studio anyway.', mbConfirmation, MB_YESNO) = IDYES then
    begin
      ShellExec('open', 'https://go.microsoft.com/fwlink/p/?LinkId=2124703', '', '', SW_SHOWNORMAL, ewNoWait, Code);
      Result := False;
    end;
end;

// Archie DAW's saved plug-in list and its sign-in to askarchie.net live in the person's own user folder, outside the install folder, so removing the program keeps them. Asking is the only way they are deleted, and a silent uninstall never deletes them. Project files are in the person's own Documents and are never touched.
procedure CurUninstallStepChanged(CurUninstallStep: TUninstallStep);
var
  Saved: String;
begin
  if CurUninstallStep = usUninstall then
  begin
    Saved := ExpandConstant('{userappdata}\AskArchie\Studio');
    if DirExists(Saved) and (not UninstallSilent) then
      if MsgBox('Also delete Archie DAW''s saved settings (its plug-in list and its sign-in to askarchie.net)?' + #13#10 + #13#10 + 'They are kept in ' + Saved + ' on this computer. Your project files are not touched either way. Choose No to keep them.', mbConfirmation, MB_YESNO or MB_DEFBUTTON2) = IDYES then
        DelTree(Saved, True, True, True);
  end;
end;
