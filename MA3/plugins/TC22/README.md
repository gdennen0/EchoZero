# Autosave Showfile Plugin

Standalone grandMA3 Lua plugin that periodically saves the current showfile with a timestamped name.

## Load

```lua
Lua "dofile('/path/to/MA3/plugins/TC22/autosave_showfile.lua')"
```

## Use

```lua
EZ_AutosaveShow.SetBaseName("TourBackup")  -- stores base name in MA3 GlobalVars
EZ_AutosaveShow.Start(10)                  -- autosave every 10 minutes
EZ_AutosaveShow.Start(5, "TourBackup")     -- custom interval + base name
EZ_AutosaveShow.SaveNow()                  -- immediate timestamped save
EZ_AutosaveShow.Status()                   -- print status
EZ_AutosaveShow.Stop()                     -- stop future autosaves
```

Generated names append the timestamp to the base name:

```text
TourBackup_2026-05-15_10-01-00
```

## Global base-name variable

By default the plugin reads/writes this MA3 global variable:

```lua
EZ_AUTOSAVE_BASENAME
```

You can set it through the helper:

```lua
EZ_AutosaveShow.SetBaseName("TourBackup")
```

Or directly in MA3 Lua if needed:

```lua
SetVar(GlobalVars(), "EZ_AUTOSAVE_BASENAME", "TourBackup")
```

`SaveNow()` and scheduled saves will then produce names like:

```text
TourBackup_2026-05-15_10-01-00
```

If the global variable is missing, the plugin falls back to `EZ_AutosaveShow.config.prefix`.

## Command syntax

The plugin centralizes the MA3 save command here:

```lua
EZ_AutosaveShow.config.commandTemplate = 'SaveShow "%s" /nc'
```

If a specific grandMA3 version/site expects a different save-as command syntax, adjust only that template. The `%s` receives the sanitized timestamped filename.

## Behavior notes

- Uses MA3 `Timer()` to schedule one save per interval.
- `Stop()` disables the active generation. An already-scheduled timer may wake once, but it will exit without saving.
- Filenames are sanitized for common macOS/Windows-invalid characters.
- The plugin does not delete old autosaves yet; retention cleanup can be added as a second slice after the exact MA3 save command is field-confirmed.
