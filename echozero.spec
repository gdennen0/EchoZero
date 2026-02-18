# -*- mode: python ; coding: utf-8 -*-
# EchoZero PyInstaller spec - production packaging
# Build: pyinstaller echozero.spec   or: python scripts/build_app.py
# Output: dist/EchoZero/ (one-folder); on macOS also dist/EchoZero.app

import json
import os
import sys
from PyInstaller.utils.hooks import collect_submodules

# Project root (directory containing this spec file)
SPEC_DIR = os.path.dirname(os.path.abspath(SPEC))

# Load packaging config (version, bundle id, etc.)
def _load_packaging_config():
    path = os.path.join(SPEC_DIR, 'packaging_config.json')
    if os.path.isfile(path):
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {
        'app_name': 'EchoZero',
        'version': '0.1.0',
        'bundle_identifier': 'dev.speedoflight.echozero',
        'company': 'Speed of Light',
        'copyright': 'Copyright (c) Speed of Light. All rights reserved.',
        'pyinstaller': {'console': False, 'strip': False, 'upx': True},
    }

_config = _load_packaging_config()
APP_NAME = _config.get('app_name', 'EchoZero')
APP_VERSION = _config.get('version', '0.1.0')
BUNDLE_ID = _config.get('bundle_identifier', 'dev.speedoflight.echozero')
COPYRIGHT = _config.get('copyright', '')
pyi_opts = _config.get('pyinstaller', {})
# macOS: UPX can cause "damaged" / prohibitory symbol; disable for darwin
_use_upx = pyi_opts.get('upx', True) and (sys.platform != 'darwin')

# Bundled data
datas = []
if os.path.isdir(os.path.join(SPEC_DIR, 'data')):
    datas.append((os.path.join(SPEC_DIR, 'data'), 'data'))
env_example = os.path.join(SPEC_DIR, '.env.example')
if os.path.isfile(env_example):
    datas.append((env_example, '.'))
# Build-time bundled config (created by scripts/build_app.py when MEMBERSTACK_APP_SECRET set)
# Enables shipping a complete package with zero user configuration
bundled_config = os.path.join(SPEC_DIR, 'build', 'bundled_config.env')
if os.path.isfile(bundled_config):
    datas.append((bundled_config, '.'))

# Ensure these packages are bundled as data so they are always found at runtime
for _pkg_name, _bundle_name in [('dotenv', 'dotenv'), ('httpx', 'httpx')]:
    try:
        _pkg = __import__(_pkg_name)
        _pkg_dir = os.path.dirname(getattr(_pkg, '__file__', '') or '')
        if _pkg_dir and os.path.isdir(_pkg_dir):
            datas.append((_pkg_dir, _bundle_name))
    except Exception:
        pass

# Hidden imports
hiddenimports = (
    collect_submodules('src')
    + collect_submodules('ui')
    + [
        'dotenv',
        'httpx',
        'PyQt6.QtCore',
        'PyQt6.QtGui',
        'PyQt6.QtWidgets',
    ]
)

a = Analysis(
    [os.path.join(SPEC_DIR, 'main_qt.py')],
    pathex=[SPEC_DIR],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        'tkinter',
        'numpy.distutils.tests',
        'tensorboard',  # Optional; avoids protobuf/tensorboard compat issues in frozen build
    ],
    noarchive=False,
    optimize=0,
)

pyz = PYZ(a.pure)

# Entitlements (macOS): allow outgoing network so auth verification works in the .app
_entitlements_path = os.path.join(SPEC_DIR, 'packaging', 'EchoZero.entitlements')
_entitlements_file = _entitlements_path if os.path.isfile(_entitlements_path) else None

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name=APP_NAME,
    debug=False,
    bootloader_ignore_signals=False,
    strip=pyi_opts.get('strip', False),
    upx=_use_upx,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=pyi_opts.get('console', False),
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=_entitlements_file if sys.platform == 'darwin' else None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=pyi_opts.get('strip', False),
    upx=_use_upx,
    upx_exclude=[],
    name=APP_NAME,
)

# macOS: regenerate .icns from packaging/ezicon.png so repackaging always uses current icon
if sys.platform == 'darwin':
    _ezicon_png = os.path.join(SPEC_DIR, 'packaging', 'ezicon.png')
    _icns_out = os.path.join(SPEC_DIR, 'packaging', 'EchoZero.icns')
    _iconset_dir = os.path.join(SPEC_DIR, 'packaging', 'EchoZero.iconset')
    if os.path.isfile(_ezicon_png):
        import shutil
        import subprocess
        if os.path.isdir(_iconset_dir):
            shutil.rmtree(_iconset_dir)
        os.makedirs(_iconset_dir, exist_ok=True)
        for size in (16, 32, 64, 128, 256, 512):
            out = os.path.join(_iconset_dir, 'icon_{0}x{0}.png'.format(size))
            subprocess.run(['sips', '-z', str(size), str(size), _ezicon_png, '--out', out], check=True, capture_output=True)
        subprocess.run(['sips', '-z', '32', '32', _ezicon_png, '--out', os.path.join(_iconset_dir, 'icon_16x16@2x.png')], check=True, capture_output=True)
        subprocess.run(['sips', '-z', '64', '64', _ezicon_png, '--out', os.path.join(_iconset_dir, 'icon_32x32@2x.png')], check=True, capture_output=True)
        subprocess.run(['sips', '-z', '256', '256', _ezicon_png, '--out', os.path.join(_iconset_dir, 'icon_128x128@2x.png')], check=True, capture_output=True)
        subprocess.run(['sips', '-z', '512', '512', _ezicon_png, '--out', os.path.join(_iconset_dir, 'icon_256x256@2x.png')], check=True, capture_output=True)
        subprocess.run(['sips', '-z', '1024', '1024', _ezicon_png, '--out', os.path.join(_iconset_dir, 'icon_512x512@2x.png')], check=True, capture_output=True)
        subprocess.run(['iconutil', '-c', 'icns', _iconset_dir, '-o', _icns_out], check=True, capture_output=True)
        shutil.rmtree(_iconset_dir)

# macOS: use packaging/EchoZero.icns for the .app bundle (generated above from ezicon.png)
_icon_path = os.path.join(SPEC_DIR, 'packaging', 'EchoZero.icns')
_icon = _icon_path if os.path.isfile(_icon_path) else None
if sys.platform == 'darwin':
    app = BUNDLE(
        coll,
        name=APP_NAME + '.app',
        icon=_icon,
        bundle_identifier=BUNDLE_ID,
        version=APP_VERSION,
        info_plist={
            'CFBundleName': APP_NAME,
            'CFBundleDisplayName': APP_NAME,
            'CFBundleIdentifier': BUNDLE_ID,
            'CFBundleVersion': APP_VERSION,
            'CFBundleShortVersionString': APP_VERSION,
            'CFBundlePackageType': 'APPL',
            'CFBundleSignature': '????',
            'CFBundleExecutable': APP_NAME,
            'NSPrincipalClass': 'NSApplication',
            'NSHighResolutionCapable': True,
            'NSHumanReadableCopyright': COPYRIGHT,
            'LSMinimumSystemVersion': '10.13',
        },
    )
