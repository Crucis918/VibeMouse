[Setup]
AppId={{5D3AFC84-B0A3-4FC7-87A9-4A7E11C6C4F0}
AppName=VibeMouse
AppVersion=0.1.0
AppPublisher=VibeMouse
DefaultDirName={autopf}\VibeMouse
DefaultGroupName=VibeMouse
OutputDir=..\..\dist
OutputBaseFilename=VibeMouse-Setup
Compression=lzma
SolidCompression=yes
ArchitecturesAllowed=x64
ArchitecturesInstallIn64BitMode=x64
WizardStyle=modern
PrivilegesRequired=lowest
UninstallDisplayIcon={app}\VibeMouse.exe

[Languages]
Name: "chinesesimplified"; MessagesFile: "compiler:Default.isl,ChineseSimplified.isl"

[Files]
Source: "..\..\dist\VibeMouse\*"; DestDir: "{app}"; Flags: recursesubdirs ignoreversion

[Icons]
Name: "{autoprograms}\VibeMouse"; Filename: "{app}\VibeMouse.exe"
Name: "{autodesktop}\VibeMouse"; Filename: "{app}\VibeMouse.exe"

[Registry]
Root: HKCU; Subkey: "Software\Microsoft\Windows\CurrentVersion\Run"; ValueType: string; ValueName: "VibeMouse"; ValueData: """{app}\VibeMouse.exe"""; Flags: uninsdeletevalue

[Dirs]
Name: "{localappdata}\VibeMouse"

[Run]
Filename: "{app}\VibeMouse.exe"; Description: "启动 VibeMouse"; Flags: nowait postinstall skipifsilent

[UninstallDelete]
Type: filesandordirs; Name: "{localappdata}\VibeMouse"
