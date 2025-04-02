block_cipher = None

a = Analysis(['main.py'],
             pathex=['.'],
             binaries=[],  # Add binaries here if needed, e.g., [('C:/path/to/scip.dll', '.')]
             datas=[],     # Add data files here if needed, e.g., [('assets/icon.ico', 'assets')]
             hiddenimports=['cv2', 'numpy', 'scipy', 'pyscipopt', 'sklearn', 'polars', 'xlsxwriter', 'pyqt6'],
             hookspath=[],
             runtime_hooks=[],
             excludes=[],
             win_no_prefer_redirects=False,
             win_private_assemblies=False,
             cipher=block_cipher,
             noarchive=False)
pyz = PYZ(a.pure, a.zipped_data,
          cipher=block_cipher)
exe = EXE(pyz,
          a.scripts,
          a.binaries,
          a.zipfiles,
          a.datas,
          [],
          name='colony_counter',
          debug=False,
          bootloader_ignore_signals=False,
          strip=False,
          upx=True,
          upx_exclude=[],
          runtime_tmpdir=None,
          console=False,  # Hides console for GUI apps
          icon='assets/icon.ico')  # Optional: custom icon