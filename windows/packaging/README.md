# WiX packaging for Windows MSI installer

This directory contains the WiX Toolset v4 source for creating the Windows MSI
installer.

## Build the MSI

After PyInstaller produces `dist\AudioMuse-AI\`:

```powershell
# Install WiX Toolset v5 (if not already)
dotnet tool install --global wix

# Build the MSI
wix build `
  -arch x64 `
  -d BundleDir=dist\AudioMuse-AI `
  -d Version=1.0.0 `
  -o dist\AudioMuse-AI-amd64-windows.msi `
  windows\packaging\AudioMuse-AI.wxs
```

## What the MSI does

- Installs to `C:\Program Files\AudioMuse-AI\`
- Creates a Start Menu shortcut
- Adds an uninstall entry in "Add or Remove Programs"
- Registers the `.audiomuse` file extension (future)

## Harvesting

For production, use WiX Heat to auto-harvest the entire `dist\AudioMuse-AI\`
directory into component groups. The current `.wxs` is a minimal template;
expand it to include the harvested components.
