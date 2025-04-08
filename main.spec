# -*- mode: python ; coding: utf-8 -*-


a = Analysis(
    ['main.py'],
    pathex=[],
    binaries=[],
    datas=[],
    hiddenimports=[
        'cv2', 'numpy', 'scipy', 'sklearn', 'polars', 'xlsxwriter',
        'PyQt6.QtCore', 'PyQt6.QtGui', 'PyQt6.QtWidgets', 'pyscipopt'
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        'tkinter', 'matplotlib', 'pandas', 'IPython', 'jupyter', 'torch', 'tensorflow',
        'PIL', 'PyQt6.Qt3D*', 'PyQt6.QtBluetooth', 'PyQt6.QtDesigner', 'PyQt6.QtHelp',
        'PyQt6.QtMultimedia', 'PyQt6.QtNetwork', 'PyQt6.QtOpenGL', 'PyQt6.QtSql',
        'PyQt6.QtTest', 'PyQt6.QtWeb*', 'PyQt6.QtXml', 'PyQt6.uic'
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='Colony Counter',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=['Qt6Core.dll', 'Qt6Gui.dll', 'Qt6Widgets.dll'],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon='assets\icon.ico'
)
