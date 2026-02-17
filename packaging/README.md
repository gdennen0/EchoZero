# Packaging assets

## Network entitlement (macOS, required for auth)

`EchoZero.entitlements` is included in the repo and grants the packaged app **outgoing network (client)** access. Without it, the .app cannot resolve hostnames or open HTTPS connections, so auth verification fails with "Could not reach the verification server" / "nodename nor servname provided, or not known". Do not remove this file when building for macOS.

## App icon (macOS)

The .app bundle uses `EchoZero.icns` in this folder. **When you run a build on macOS** (e.g. `pyinstaller echozero.spec` or `python scripts/build_app.py`), the spec automatically regenerates `EchoZero.icns` from `ezicon.png`. So after updating `ezicon.png`, repackaging is enough; the new icon is used.

To regenerate `EchoZero.icns` manually without building (e.g. to commit it or preview), run from this folder:

```bash
rm -rf EchoZero.iconset && mkdir EchoZero.iconset
for size in 16 32 64 128 256 512; do
  sips -z $size $size ezicon.png --out EchoZero.iconset/icon_${size}x${size}.png
done
# @2x Retina sizes
sips -z 32 32 ezicon.png --out EchoZero.iconset/icon_16x16@2x.png
sips -z 64 64 ezicon.png --out EchoZero.iconset/icon_32x32@2x.png
sips -z 256 256 ezicon.png --out EchoZero.iconset/icon_128x128@2x.png
sips -z 512 512 ezicon.png --out EchoZero.iconset/icon_256x256@2x.png
sips -z 1024 1024 ezicon.png --out EchoZero.iconset/icon_512x512@2x.png
iconutil -c icns EchoZero.iconset -o EchoZero.icns
rm -rf EchoZero.iconset
```

If `ezicon.png` is missing at build time, the .app uses any existing `EchoZero.icns` or the system default icon.

## Prohibitory symbol on the app icon (macOS)

If the .app shows a white circle with a line through it (prohibitory symbol), macOS considers the app incompatible or damaged. Common causes:

1. **Architecture mismatch** – The app was built for a different CPU (e.g. built on Intel, run on Apple Silicon, or the reverse). Build on the same architecture you run on, or use a matching Python/venv (e.g. `arch -x86_64` for Intel build on an M1 Mac). The spec uses the current architecture only (`target_arch=None`).
2. **UPX** – UPX compression is disabled for macOS in the spec because it can trigger Gatekeeper/SIP and cause the "damaged" behaviour. If you changed the spec to enable UPX on darwin, set it off for macOS again.
3. **Code signing** – For distribution, sign (and optionally notarize) the .app; unsigned apps can be quarantined or blocked. The spec does not set `codesign_identity`; sign after build if needed.
