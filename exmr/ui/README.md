# ExMR web UI

A small local app around ExMR: pick a clip, reconstruct its 3D motion, watch it
on a VRM avatar, and export the result as a `.vrma`.

```
exmr/ui
├─ server.py       FastAPI server: job runner, motion endpoint, .vrma export
├─ vrma.py         glTF/VRMC_vrm_animation writer (from the rtmlib app)
├─ assets/         avatar(s) + idle animation
├─ videos/         clips offered in the UI (created on first run)
└─ web/            Vite + three.js front end (built into exmr/ui/dist)
```

## Run

```bash
cd exmr/ui/web && npm install && npm run build   # once, and after src changes
cd ../../.. && .venv/Scripts/python -m exmr.ui.server --port 8010
```

Then open <http://127.0.0.1:8010>.

The job uses `tools/demo.py` from the same interpreter, so the models in
`checkpoints/` must be present (the status pill says `ready` when they are).

## How it works

1. **Run ExMR** starts `tools/demo.py --no_render` for the selected clip and
   streams its phase lines back to the browser.
2. `results.npz` → `joints_global` (SMPL-X body joints, world space) is served to
   the page, one frame at a time.
3. The page retargets those joints onto the VRM humanoid (`web/src/retarget.js`,
   adapted from the rtmlib mocap retargeter) and drives the avatar in sync with
   the `<video>` element.
4. **Export .vrma** bakes the avatar's bone quaternions while replaying every
   frame and posts them to `/api/jobs/{id}/export`, which writes
   `outputs/ui_exports/<clip>_<job>/<clip>.vrma`.

The viewer is orbit-controlled: drag to rotate, scroll to zoom.

Audio is not part of the export — the `.vrma` carries the motion only.
