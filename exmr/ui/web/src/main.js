import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { BVHLoader } from 'three/examples/jsm/loaders/BVHLoader.js';
import { VRMLoaderPlugin, VRMUtils } from '@pixiv/three-vrm';
import {
  createVRMAnimationClip,
  VRMAnimationLoaderPlugin,
} from '@pixiv/three-vrm-animation';
import { MocapRetargeter } from './retarget.js';
import { BVHRetargeter } from './bvhretarget.js';
import './style.css';

const $ = (id) => document.getElementById(id);
const preview = $('preview');
const statusText = $('status-text');
const statusPill = $('status-pill');
const clips = $('clips');
const phase = $('phase');
const bar = $('bar');
const exportInfo = $('export-info');
const downloads = $('downloads');

const ANIM_EXPRESSIONS = [
  'happy', 'angry', 'sad', 'relaxed', 'surprised',
  'aa', 'ih', 'ou', 'ee', 'oh', 'blink', 'blinkLeft', 'blinkRight',
];

const state = {
  video: '',
  job: '',
  motion: null,
  playing: false,
  lastIndex: -1,
  retargeter: null,
  vrm: null,
  mixer: null,
  idleMixer: null,
  idleAction: null,
  bvhRetarget: null,
  bvhData: null,
  busy: false,
  runId: 0,
  // BVH preview state
  bvhMode: false,
  bvhGroup: null,
  bvhMixer: null,
  bvhAction: null,
  bvhClip: null,
  bvhSkeleton: null,
  bvhLastVideoTime: -1,
};

/* ---------------------------------------------------------------- scene */
const canvas = $('scene');
const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true });
renderer.setPixelRatio(Math.min(2, window.devicePixelRatio || 1));
renderer.outputColorSpace = THREE.SRGBColorSpace;

const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(32, 1, 0.05, 60);
camera.position.set(0, 1.5, 3.1);
camera.lookAt(0, 1.05, 0);

const controls = new OrbitControls(camera, canvas);
controls.target.set(0, 1.05, 0);
controls.enableDamping = true;
controls.dampingFactor = 0.08;
controls.minDistance = 0.8;
controls.maxDistance = 12;
controls.maxPolarAngle = Math.PI * 0.52;
controls.update();

scene.add(new THREE.HemisphereLight(0xbfe8ff, 0x10141c, 2.2));
const key = new THREE.DirectionalLight(0xffffff, 2.4);
key.position.set(1.6, 3.2, 2.4);
scene.add(key);

const ground = new THREE.Mesh(
  new THREE.CircleGeometry(2.4, 64),
  new THREE.MeshStandardMaterial({ color: 0x11161f, roughness: 0.95, metalness: 0 }),
);
ground.rotation.x = -Math.PI / 2;
ground.position.y = -0.002;
scene.add(ground);

const avatarRoot = new THREE.Group();
scene.add(avatarRoot);

// Separate group for the BVH skeleton preview — toggled on/off with the BVH button
const bvhRoot = new THREE.Group();
bvhRoot.visible = false;
scene.add(bvhRoot);

function resize() {
  const box = canvas.parentElement.getBoundingClientRect();
  const width = Math.max(320, box.width);
  const height = Math.max(320, box.height - 120);
  renderer.setSize(width, height, false);
  camera.aspect = width / height;
  camera.updateProjectionMatrix();
}
window.addEventListener('resize', resize);
if (typeof ResizeObserver !== 'undefined') {
  new ResizeObserver(() => resize()).observe(canvas.parentElement);
}

/* ----------------------------------------------------------------- ui */
function toast(message, error = false) {
  phase.textContent = message;
  phase.classList.toggle('error', error);
}

function setStatus(text, ok) {
  statusText.textContent = text;
  statusPill.classList.toggle('ok', ok === true);
  statusPill.classList.toggle('bad', ok === false);
}

async function api(url, options) {
  const response = await fetch(url, options);
  const data = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(data.error || `HTTP ${response.status}`);
  return data;
}

/* --------------------------------------------------------------- avatar */
function loadGLTF(url, plugin) {
  return new Promise((resolve, reject) => {
    const loader = new GLTFLoader();
    if (plugin) loader.register(plugin);
    loader.load(url, resolve, undefined, reject);
  });
}

let vrmToken = 0;

async function loadVRM(url) {
  const token = ++vrmToken;
  const gltf = await loadGLTF(url, (parser) => new VRMLoaderPlugin(parser));
  // a newer load started while this one was downloading: drop the stale result
  // instead of adding a second avatar next to the one already on screen
  if (token !== vrmToken) return;
  const vrm = gltf.userData.vrm;
  if (!vrm) throw new Error('not a VRM file');
  VRMUtils.removeUnnecessaryVertices(gltf.scene);
  VRMUtils.rotateVRM0(vrm);
  vrm.scene.traverse((object) => {
    if (object.isMesh) {
      object.castShadow = true;
      object.frustumCulled = false;
    }
  });
  // drop the previous avatar (and its GPU buffers) so a reload never leaves a
  // second character standing in the scene
  stopIdle();
  for (const child of [...avatarRoot.children]) {
    avatarRoot.remove(child);
    child.traverse((object) => {
      if (!object.isMesh) return;
      object.geometry?.dispose();
      const materials = Array.isArray(object.material) ? object.material : [object.material];
      for (const material of materials) {
        material?.map?.dispose();
        material?.dispose?.();
      }
    });
  }
  avatarRoot.add(vrm.scene);
  vrm.scene.updateMatrixWorld(true);
  state.vrm = vrm;
  state.retargeter = new MocapRetargeter(vrm, avatarRoot);
  bindBVH();
  if (vrm.lookAt) vrm.lookAt.target = camera;
  state.mixer = null;
  state.idleMixer = null;
  state.idleAction = null;
  // frame the avatar so a reload or an orbit never leaves it off-screen
  const box = new THREE.Box3().setFromObject(vrm.scene);
  if (!box.isEmpty()) {
    const size = box.getSize(new THREE.Vector3());
    const center = box.getCenter(new THREE.Vector3());
    controls.target.copy(center);
    camera.position.set(center.x, center.y, center.z + Math.max(1.6, size.y * 1.9));
    camera.lookAt(center);
    controls.update();
  }
  try {
    const idle = await loadGLTF('/api/idle', (parser) => new VRMAnimationLoaderPlugin(parser));
    const animation = idle.userData.vrmAnimations?.[0];
    if (animation) {
      const mixer = new THREE.AnimationMixer(vrm.scene);
      const action = mixer.clipAction(createVRMAnimationClip(animation, vrm));
      state.idleMixer = mixer;
      state.idleAction = action;
      if (state.motion) stopIdle();
      else playIdle();
    }
  } catch {
    /* idle is optional */
  }
  toast(`avatar loaded: ${url.split('/').pop()}`);
}

/* The idle loop is only for a resting avatar. While mocap drives the humanoid
   the mixer must stay off: both write the same normalized bones every frame,
   and the two poses alternating reads as heavy jitter. */
function playIdle() {
  if (!state.idleAction || state.motion) return;
  state.idleAction.reset().play();
  state.mixer = state.idleMixer;
}

function stopIdle() {
  state.idleAction?.stop();
  state.mixer = null;
}

/* ----------------------------------------------------------- BVH preview */
function clearBVH() {
  state.bvhMixer?.stopAllAction();
  state.bvhMixer = null;
  state.bvhAction = null;
  state.bvhClip = null;
  state.bvhSkeleton = null;
  state.bvhLastVideoTime = -1;
  for (const child of [...bvhRoot.children]) {
    bvhRoot.remove(child);
    child.traverse((obj) => {
      if (obj.geometry) obj.geometry.dispose();
      if (obj.material) {
        const mats = Array.isArray(obj.material) ? obj.material : [obj.material];
        for (const m of mats) m.dispose();
      }
    });
  }
  state.bvhGroup = null;
}

/* ExMR writes the motion twice: `results.npz` (joints_global) and a BVH. The
   BVH is the shipped skeleton, and its bone rotations are transferred one to
   one onto the VRM humanoid (see bvhretarget.js). */
async function fetchBVH(jobId) {
  const loader = new BVHLoader();
  const result = await loader.loadAsync(`/api/jobs/${jobId}/bvh`);

  // BVH from ExMR is in centimeters. Convert every offset and every
  // position track to meters so the skeleton lives in the same unit
  // space as the VRM avatar (~1.5 m tall) and visual decorations
  // (spheres, lines) render at a sensible size.
  for (const bone of result.skeleton.bones) bone.position.multiplyScalar(0.01);
  for (const track of result.clip.tracks) {
    if (!track.name.endsWith('.position')) continue;
    const values = track.values;
    for (let i = 0; i < values.length; i += 1) values[i] *= 0.01;
  }
  return result;
}

async function loadBVH(jobId, preloaded = null) {
  clearBVH();
  const url = `/api/jobs/${jobId}/bvh`;
  try {
    const result = preloaded || await fetchBVH(jobId);
    const bones = result.skeleton.bones;
    const rootBone = bones[0];
    // Use a plain group (no scale) so child geometry uses scene units
    const group = new THREE.Group();
    group.add(rootBone);

    // Bone lines: SkeletonHelper renders 1-pixel lines in WebGL, so we
    // also build a cylinder-based representation for thickness.
    const helper = new THREE.SkeletonHelper(rootBone);
    helper.material.color = new THREE.Color(0x4bf0b6);
    group.add(helper);

    // Joint spheres at each named bone — 1.5 cm radius in scene units
    const sphereGeo = new THREE.SphereGeometry(0.015, 10, 8);
    const sphereMat = new THREE.MeshBasicMaterial({ color: 0x4bf0b6 });
    const jointSpheres = [];
    for (const bone of bones) {
      if (bone.name === 'ENDSITE') continue;
      const dot = new THREE.Mesh(sphereGeo, sphereMat);
      bone.add(dot);
      jointSpheres.push({ bone, dot });
    }

    bvhRoot.add(group);
    state.bvhGroup = group;
    state.bvhSkeleton = result.skeleton;
    state.bvhClip = result.clip;

    // Animation mixer — driven by mixer.update(dt) in draw(), synced
    // to the video element's currentTime.
    const mixer = new THREE.AnimationMixer(rootBone);
    const action = mixer.clipAction(result.clip);
    action.play();
    state.bvhMixer = mixer;
    state.bvhAction = action;
    state.bvhLastVideoTime = -1;

    // Apply frame 0 so the skeleton has a rest pose before play
    mixer.update(0);
    rootBone.updateMatrixWorld(true);

    // Frame the skeleton
    const box = new THREE.Box3().setFromObject(group);
    if (!box.isEmpty()) {
      const size = box.getSize(new THREE.Vector3());
      const center = box.getCenter(new THREE.Vector3());
      controls.target.copy(center);
      camera.position.set(center.x, center.y, center.z + Math.max(1.6, size.y * 1.9));
      camera.lookAt(center);
      controls.update();
    }

    toast(`BVH loaded: ${result.clip.duration.toFixed(1)}s`);
    return true;
  } catch (err) {
    toast(`BVH load failed: ${err.message}`, true);
    return false;
  }
}

function setBvhMode(enabled) {
  state.bvhMode = enabled;
  $('btn-bvh').classList.toggle('active', enabled);

  // Toggle visibility of the two preview layers
  bvhRoot.visible = enabled;
  if (state.vrm) {
    state.vrm.scene.visible = !enabled;
  }

  // When entering BVH mode, sync the mixer to the video's current time
  if (enabled && state.bvhMixer) {
    const t = preview.currentTime;
    const dur = state.bvhClip?.duration || 0;
    state.bvhMixer.setTime(Math.min(t, dur));
    state.bvhLastVideoTime = t;
    state.bvhSkeleton?.bones[0]?.updateMatrixWorld(true);
  }

  // Frame whichever preview is active
  const target = enabled ? state.bvhGroup : state.vrm?.scene;
  if (target) {
    const box = new THREE.Box3().setFromObject(target);
    if (!box.isEmpty()) {
      const size = box.getSize(new THREE.Vector3());
      const center = box.getCenter(new THREE.Vector3());
      controls.target.copy(center);
      camera.position.set(center.x, center.y, center.z + Math.max(1.6, size.y * 1.9));
      camera.lookAt(center);
      controls.update();
    }
  }
}

/* --------------------------------------------------------------- clips */
function renderClips(videos) {
  clips.innerHTML = '';
  for (const video of videos) {
    const item = document.createElement('li');
    const button = document.createElement('button');
    button.type = 'button';
    button.className = 'clip';
    button.textContent = video.name;
    button.classList.toggle('active', video.name === state.video);
    button.addEventListener('click', () => pickClip(video));
    item.append(button);
    clips.append(item);
  }
}

function pickClip(video) {
  state.video = video.name;
  preview.src = video.url;
  preview.load();
  state.motion = null;
  state.job = '';
  $('btn-play').disabled = true;
  $('btn-export').disabled = true;
  $('btn-bvh').disabled = true;
  $('btn-bvh').classList.remove('active');
  setBvhMode(false);
  clearBVH();
  playIdle();
  downloads.innerHTML = '';
  exportInfo.textContent = '';
  for (const button of clips.querySelectorAll('.clip')) {
    button.classList.toggle('active', button.textContent === video.name);
  }
  toast('ready to run');
}

async function refreshClips() {
  const data = await api('/api/videos');
  renderClips(data.videos);
  if (!state.video && data.videos.length) pickClip(data.videos[0]);
}

/* ---------------------------------------------------------------- jobs */
function setBusy(busy, text = '') {
  state.busy = busy;
  $('btn-run').textContent = busy ? 'Stop' : 'Run ExMR';
  $('btn-run').classList.toggle('primary', !busy);
  if (text) toast(text);
}

async function run() {
  if (state.busy) {
    if (state.job) await api(`/api/jobs/${state.job}/cancel`, { method: 'POST' });
    return;
  }
  if (!state.video) return toast('pick a clip first', true);
  state.runId += 1;
  const runId = state.runId;
  try {
    const job = await api('/api/jobs', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        video: state.video,
        static_cam: $('opt-static').checked,
        snap_to_ground: $('opt-snap').checked,
      }),
    });
    state.job = job.id;
    state.motion = null;
    $('btn-play').disabled = true;
    $('btn-export').disabled = true;
    $('btn-bvh').disabled = true;
    $('btn-bvh').classList.remove('active');
    setBvhMode(false);
    clearBVH();
    downloads.innerHTML = '';
    setBusy(true, 'starting ExMR');
    poll(runId);
  } catch (err) {
    setBusy(false);
    toast(`ExMR failed: ${err.message}`, true);
  }
}

/**
 * Load a finished job: motion frames, the BVH that drives them, and the BVH
 * preview. Also used when resuming a job after a server restart.
 */
async function openJob(jobId) {
  state.job = jobId;
  state.motion = await api(`/api/jobs/${jobId}/motion`);
  state.lastIndex = -1;
  state.retargeter?.reset();
  stopIdle();
  $('btn-play').disabled = false;
  $('btn-export').disabled = false;

  // Drive the avatar from the BVH ExMR already exported: its bone rotations
  // are copied onto the VRM humanoid, so the pose matches the skeleton exactly.
  let bvh = null;
  try {
    bvh = await fetchBVH(jobId);
  } catch {
    bvh = null;
  }
  state.bvhData = bvh;
  bindBVH();
  const ok = await loadBVH(jobId, bvh);
  $('btn-bvh').disabled = !ok;
}

/** (Re)build the BVH -> VRM rig; needs both a loaded avatar and a BVH. */
function bindBVH() {
  state.bvhRetarget = null;
  if (!state.vrm || !state.bvhData) return;
  try {
    state.bvhRetarget = new BVHRetargeter(state.vrm, avatarRoot,
      state.bvhData.skeleton, state.bvhData.clip);
    if (state.motion) state.bvhRetarget.update(preview.currentTime || 0);
  } catch (err) {
    state.bvhRetarget = null;
    toast(`BVH rig failed: ${err.message}`, true);
  }
}

async function poll(runId) {
  if (runId !== state.runId) return;
  let status;
  try {
    status = await api(`/api/jobs/${state.job}`);
  } catch (err) {
    setBusy(false);
    return toast(`ExMR failed: ${err.message}`, true);
  }
  bar.style.width = `${Math.min(96, 6 + (status.elapsed || 0) * 1.4)}%`;
  if (status.message) toast(status.message);
  if (status.state === 'done') {
    bar.style.width = '100%';
    setBusy(false, 'motion ready - press Play preview');
    await openJob(state.job);
    return;
  }
  if (status.state === 'error' || status.state === 'cancelled') {
    setBusy(false, status.error || 'cancelled');
    bar.style.width = '0%';
    toast(status.error || 'cancelled', status.state === 'error');
    return;
  }
  setTimeout(() => poll(runId), 700);
}

/* -------------------------------------------------------------- preview */
function draw() {
  requestAnimationFrame(draw);
  const dt = Math.min(0.05, clock.getDelta());
  // the idle loop only drives a resting avatar; the mocap pose owns the
  // humanoid otherwise, and running both is what made the preview jitter
  if (!state.motion) state.mixer?.update(dt);
  controls?.update();

  if (state.bvhMode && state.bvhMixer) {
    // Sync BVH animation to the video element's currentTime.
    // Use mixer.update(dt) for smooth playback, and setTime() only for
    // seeks (when the video jumps backwards or forward by more than 0.3s).
    const t = preview.currentTime;
    const dur = state.bvhClip?.duration || 0;
    const clampedT = Math.min(t, dur);
    if (state.playing) {
      const delta = t - state.bvhLastVideoTime;
      if (delta < -0.1 || delta > 0.3) {
        // Seek detected — jump to the target time
        state.bvhMixer.setTime(clampedT);
      } else {
        // Normal forward playback — advance the mixer by the real elapsed time
        state.bvhMixer.update(Math.max(0, delta));
      }
    }
    state.bvhLastVideoTime = t;
    // Always update world matrices so the skeleton helper + spheres render
    state.bvhSkeleton?.bones[0]?.updateMatrixWorld(true);
  } else {
    const frames = state.motion?.frames;
    const rt = state.retargeter;
    const bvh = state.bvhRetarget;
    if (bvh && state.motion) {
      // The BVH path is a pure function of the video clock: seek, pause and
      // scrub all land on the exact pose, with no frame stepping.
      bvh.update(preview.currentTime, dt);
    } else if (frames && rt && state.playing) {
      const fps = state.motion.fps;
      const index = Math.min(frames.length - 1, Math.floor(preview.currentTime * fps));
      // A seek, a fresh play or a long stall can leave the avatar far from the
      // video. Rather than replaying every skipped frame (which looks like the
      // animation racing ahead), settle on the target frame: warm the filters
      // with the few frames before it, then continue one frame at a time.
      if (index < state.lastIndex || index > state.lastIndex + 6) {
        rt.reset();
        const from = Math.max(0, index - 8);
        for (let i = from; i <= index; i += 1) rt.update(frames[i], 1 / fps);
        state.lastIndex = index;
      } else {
        // step through every pending frame: the video is 25 fps while rAF is
        // 60 Hz, so this is normally a single update, and dropping frames here
        // made the pose freeze and then snap (heavy jitter)
        while (state.lastIndex < index) {
          state.lastIndex += 1;
          rt.update(frames[state.lastIndex], 1 / fps);
        }
      }
    }
    // the retargeter writes on the normalized humanoid, so the avatar's raw
    // skeleton (and its spring bones) only follows once three-vrm copies it
    if (state.vrm) {
      state.vrm.update(dt);
      rt?.vrm.scene.updateMatrixWorld(true);
    }
  }
  renderer.render(scene, camera);
}
const clock = new THREE.Clock();

function syncPlay() {
  const playing = !preview.paused && !preview.ended;
  state.playing = playing;
  $('btn-play').textContent = playing ? 'Pause preview' : 'Play preview';
  if (!playing) return;
  // keep the current frame position; draw() resyncs if the video was seeked
  preview.play().catch(() => {});
}

/* --------------------------------------------------------------- export */
function captureFrame(t) {
  const rt = state.retargeter;
  const em = state.vrm?.expressionManager;
  if (!rt || !em) return null;
  const bones = {};
  for (const [name, bone] of Object.entries(rt.b)) {
    if (!bone) continue;
    const q = bone.quaternion;
    bones[name] = [
      Math.round(q.x * 1e4) / 1e4,
      Math.round(q.y * 1e4) / 1e4,
      Math.round(q.z * 1e4) / 1e4,
      Math.round(q.w * 1e4) / 1e4,
    ];
  }
  const expr = {};
  for (const name of ANIM_EXPRESSIONS) {
    const value = em.getValue(name);
    if (value > 0.001) expr[name] = Math.round(value * 1000) / 1000;
  }
  return { t: Math.round(t * 1000) / 1000, bones, expr };
}

async function exportVRMA() {
  if (!state.motion || !state.retargeter) return;
  const rt = state.retargeter;
  const bvh = state.bvhRetarget;
  exportInfo.textContent = 'baking the animation…';
  downloads.innerHTML = '';
  await new Promise((resolve) => setTimeout(resolve, 30));
  const frames = state.motion.frames;
  const out = [];
  if (bvh) {
    bvh.reset();
    for (const frame of frames) {
      bvh.update(frame.t, 1 / state.motion.fps);
      const captured = captureFrame(frame.t);
      if (captured) out.push(captured);
    }
    bvh.reset();
  } else {
    rt.reset();
    for (const frame of frames) {
      rt.update(frame, 1 / state.motion.fps);
      const captured = captureFrame(frame.t);
      if (captured) out.push(captured);
    }
    rt.reset();
  }
  if (out.length < 2) {
    exportInfo.textContent = 'nothing was tracked';
    return;
  }
  try {
    const data = await api(`/api/jobs/${state.job}/export`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ fps: state.motion.fps, frames: out }),
    });
    exportInfo.textContent = `${data.frames} frames · ${data.bones} bones`;
    downloads.innerHTML = '';
    for (const [label, url, className] of [
      ['Download .vrma', data.animation, 'primary'],
    ]) {
      if (!url) continue;
      const link = document.createElement('a');
      link.href = url;
      link.textContent = label;
      link.className = className;
      link.setAttribute('download', '');
      downloads.append(link);
    }
  } catch (err) {
    exportInfo.textContent = `export failed: ${err.message}`;
  }
}

/* --------------------------------------------------------------- wiring */
$('btn-add').addEventListener('click', () => $('file').click());
$('file').addEventListener('change', async () => {
  const file = $('file').files?.[0];
  $('file').value = '';
  if (!file) return;
  try {
    const data = await api(`/api/upload?name=${encodeURIComponent(file.name)}`, {
      method: 'POST',
      body: file,
    });
    await refreshClips();
    pickClip(data);
  } catch (err) {
    toast(`upload failed: ${err.message}`, true);
  }
});
$('btn-remove').addEventListener('click', () => {
  state.video = '';
  state.motion = null;
  preview.removeAttribute('src');
  preview.load();
  $('btn-play').disabled = true;
  $('btn-export').disabled = true;
  $('btn-bvh').disabled = true;
  $('btn-bvh').classList.remove('active');
  setBvhMode(false);
  clearBVH();
  downloads.innerHTML = '';
  for (const button of clips.querySelectorAll('.clip')) button.classList.remove('active');
});

$('btn-run').addEventListener('click', run);
$('btn-play').addEventListener('click', () => {
  if (preview.paused) preview.play().catch(() => {});
  else preview.pause();
});
$('btn-bvh').addEventListener('click', () => {
  if (!state.bvhClip) return;
  setBvhMode(!state.bvhMode);
});
preview.addEventListener('play', syncPlay);
preview.addEventListener('pause', syncPlay);
preview.addEventListener('ended', syncPlay);

$('btn-export').addEventListener('click', exportVRMA);

$('btn-vrm').addEventListener('click', () => $('vrm-file').click());
$('vrm-file').addEventListener('change', async () => {
  const file = $('vrm-file').files?.[0];
  $('vrm-file').value = '';
  if (!file) return;
  try {
    const data = await api(`/api/vrms?name=${encodeURIComponent(file.name)}`, {
      method: 'POST',
      body: file,
    });
    await refreshVRMs(data.name);
    await loadVRM(`${data.url}?v=${Date.now()}`);
    toast(`avatar loaded: ${data.name}`);
  } catch (err) {
    toast(`avatar failed: ${err.message}`, true);
  }
});

async function refreshVRMs(selected) {
  const data = await api('/api/vrms');
  const select = $('vrm');
  select.innerHTML = '';
  for (const vrm of data.vrms) {
    const option = document.createElement('option');
    option.value = vrm.url;
    option.textContent = vrm.name;
    select.append(option);
  }
  if (selected) select.value = data.vrms.find((v) => v.name === selected)?.url || select.value;
}

$('vrm').addEventListener('change', async () => {
  try {
    await loadVRM(`${$('vrm').value}${$('vrm').value.includes('?') ? '&' : '?'}v=${Date.now()}`);
  } catch (err) {
    toast(`avatar failed: ${err.message}`, true);
  }
});

/* ---------------------------------------------------------------- boot */
window.__exmr = state;
// test hook: lets a finished job be reopened without running ExMR again
window.__exmrDebug = { openJob, fetchBVH, bindBVH };

async function boot() {
  resize();
  draw();
  try {
    const status = await api('/api/status');
    if (!status.available.ok) {
      setStatus(status.available.reason, false);
    } else {
      setStatus('ready', true);
    }
  } catch {
    setStatus('server unreachable', false);
  }
  try {
    const data = await api('/api/vrms');
    if (!data.vrms.length) {
      toast('no VRM avatar in exmr/ui/assets', true);
    } else {
      await refreshVRMs();
      await loadVRM(`${data.vrms[0].url}&v=${Date.now()}`);
    }
  } catch (err) {
    toast(`avatar failed: ${err.message}`, true);
  }
  try {
    await refreshClips();
  } catch (err) {
    toast(`clip list failed: ${err.message}`, true);
  }
}

boot();
