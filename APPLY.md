# Apply Object Event TTC v4.2

Expected Git HEAD: `e16e6e76c5367dc5a8ad916fc255cd9a1e9ff0a0` with the v4.1 patch present in the working tree.

```powershell
git apply --check .\e_jepa_ttc_object_event_v4_2.patch
git apply .\e_jepa_ttc_object_event_v4_2.patch

uv run --no-sync python -m py_compile `
  src\e_jepa_ttc\models\object_event_v4_2.py `
  src\e_jepa_ttc\training\object_event_v4_2.py `
  scripts\train_e_jepa_object_event_v4_2.py `
  scripts\preflight_object_event_v4_2.py

uv run --no-sync pytest -q `
  tests\unit\test_object_event_v4_2.py `
  tests\integration\test_object_event_v4_2_step.py

& .\scripts\run_object_event_v4_2_screen.ps1 -Device cuda -Force
```
