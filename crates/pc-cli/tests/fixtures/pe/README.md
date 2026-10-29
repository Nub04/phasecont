# PE fixture

`fixture.dll` is a small 64-bit Windows DLL for the `pc-cli dll-imports`
tests. `fixture.c` is our own source. It imports `GetTickCount` from
`kernel32.dll` and `MessageBeep` from `user32.dll`, which is outside the
allow-list on purpose, so the DLL also covers the failure path.

Built once on the development machine (the import table does not change, so
the tests do not rebuild it):

```
x86_64-w64-mingw32-gcc -shared -O2 -s -o fixture.dll fixture.c -luser32
```

Compiler: `x86_64-w64-mingw32-gcc (GCC) 16.2.1 20260819 (Fedora MinGW 16.2.1-1.fc45)`.
The DLL pulls in no MinGW runtime DLL, so `-static-libgcc` was not needed.

`x86_64-w64-mingw32-objdump -p fixture.dll | grep 'DLL Name'` prints:

```
	DLL Name: KERNEL32.dll
	DLL Name: msvcrt.dll
	DLL Name: USER32.dll
```
