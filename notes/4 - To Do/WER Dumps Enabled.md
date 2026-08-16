WER local crash dumps enabled for ARC.exe — 2026-05-14

Why: capture a .dmp the next time ARC.exe crashes (currently dying with
0xC0000409 / STATUS_STACK_BUFFER_OVERRUN during benchmark runs, after
the puzzles.clear() leak fix didn't address the actual fault).

Where dumps land:
  C:\Users\nitro\Desktop\ARC\dumps\ARC.exe.<pid>.dmp

Settings (per-app, scoped to ARC only — no global impact):
  Registry key:
    HKLM\SOFTWARE\Microsoft\Windows\Windows Error Reporting\LocalDumps\ARC.exe
  Values:
    DumpFolder = C:\Users\nitro\Desktop\ARC\dumps   (REG_EXPAND_SZ)
    DumpCount  = 10                   (REG_DWORD)  -- rotates after 10
    DumpType   = 2                    (REG_DWORD)  -- full mini (heap + stacks)

Performance impact: none during normal run. Only triggers on crash;
writing a dump takes <1s.

To disable later (admin PowerShell):
  Remove-Item 'HKLM:\SOFTWARE\Microsoft\Windows\Windows Error Reporting\LocalDumps\ARC.exe' -Recurse

To open a dump: WinDbg (already installed at C:\Tools\WinDbg\cdb.exe),
or drop the .dmp on Visual Studio.
