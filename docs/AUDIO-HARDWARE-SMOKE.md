# Audio Hardware Smoke

Status: active
Last verified: 2026-05-10


Use this checklist when validating output-device changes on physical hardware.
Automated tests use fake devices and streams; this is the human hardware path.

## Command

```bash
python3 run_echozero.py
```

## Checklist

- Import one real audio file into a new project.
- Press play, pause, seek, and stop; confirm playback stays clean.
- Open app preferences and switch the output device.
- Confirm playback state restores sanely: stopped stays stopped, playing resumes when supported.
- Confirm diagnostics show the selected/default device, resolved sample rate, channel count, and reinit count.
- Toggle mute on the active song layer and confirm audio mutes/unmutes without static.
- On stereo hardware, route a secondary layer to outputs 3/4 and confirm it does not leak to outputs 1/2.
- On multichannel hardware, route one layer to outputs 1/2 and another to outputs 3/4 and confirm both routes are audible on the expected outputs.
- Switch back to the system default device and repeat play, seek, pause, mute, and route checks.

## Report

Record the device names tested, whether each step passed, and the playback diagnostics before and after each device switch.
