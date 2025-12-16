# -*- mode: python ; coding: utf-8 -*-

import os
import sys
from pathlib import Path
try:
    from PyInstaller.utils.hooks import collect_submodules
except Exception:  # pragma: no cover - when building without PyInstaller around
    def collect_submodules(*_args, **_kwargs):
        return []

project_root = (
    Path(__file__).resolve().parent if '__file__' in globals() else Path.cwd()
)


def collect_data_dir(relative_path: str) -> list[tuple[str, str]]:
    base_path = project_root / relative_path
    if not base_path.exists():
        return []
    files = []
    for file_path in base_path.rglob('*'):
        if file_path.is_file():
            dest = Path(relative_path) / file_path.relative_to(base_path)
            files.append((str(file_path), str(dest)))
    return files


datas = [
    (str(project_root / 'config.json'), 'config.json'),
    (str(project_root / 'particle_types.json'), 'particle_types.json'),
]
datas += collect_data_dir('languages')
datas += collect_data_dir('_internal')


a = Analysis(
    ['app.py'],
    pathex=[],
    binaries=[],
    datas=datas,
    hiddenimports=collect_submodules('screeninfo'),
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='SpaceParticles',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='SpaceParticles',
)
if sys.platform == 'darwin':
    mac_codesign_identity = os.environ.get('CODESIGN_IDENTITY') or '-'
    mac_bundle_identifier = os.environ.get('MACOS_BUNDLE_IDENTIFIER') or 'ru.msu.spaceparticles'
    entitlements_candidate = (
        os.environ.get('MACOS_ENTITLEMENTS_PATH') or (project_root / 'scripts' / 'entitlements.plist')
    )
    entitlements_file = str(entitlements_candidate) if Path(entitlements_candidate).exists() else None
    app = BUNDLE(
        coll,
        name='SpaceParticles.app',
        icon=None,
        bundle_identifier=mac_bundle_identifier,
        codesign_identity=mac_codesign_identity,
        entitlements_file=entitlements_file,
    )
