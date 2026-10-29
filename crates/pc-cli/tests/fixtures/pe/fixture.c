/* Test fixture for `pc-cli dll-imports`. Imports one function from
   kernel32.dll (allowed) and one from user32.dll (outside the allow-list). */
#include <windows.h>

__declspec(dllexport) unsigned long pc_fixture_tick(void)
{
    MessageBeep(0);
    return GetTickCount();
}
