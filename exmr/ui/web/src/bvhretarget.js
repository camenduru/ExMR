import * as THREE from 'three';

/**
 * BVH joint -> VRM humanoid bone.
 *
 * A BVH joint is the *start* of the segment that ends at its first child, so
 * e.g. `left_shoulder` (shoulder -> elbow) is the avatar's `leftUpperArm`, and
 * `left_elbow` (elbow -> wrist) is `leftLowerArm`.
 */
const BONE_MAP = {
  pelvis: 'hips',
  spine1: 'spine',
  spine2: 'chest',
  spine3: 'upperChest',
  neck: 'neck',
  head: 'head',

  left_collar: 'leftShoulder',
  left_shoulder: 'leftUpperArm',
  left_elbow: 'leftLowerArm',
  left_wrist: 'leftHand',
  right_collar: 'rightShoulder',
  right_shoulder: 'rightUpperArm',
  right_elbow: 'rightLowerArm',
  right_wrist: 'rightHand',

  left_hip: 'leftUpperLeg',
  left_knee: 'leftLowerLeg',
  left_ankle: 'leftFoot',
  left_foot: 'leftToes',
  right_hip: 'rightUpperLeg',
  right_knee: 'rightLowerLeg',
  right_ankle: 'rightFoot',
  right_foot: 'rightToes',
};

// Hands: SMPL-H numbers the segments 1..3 from the wrist outwards; VRM names
// them by segment. VRM 0 avatars may only have the older thumb names, so each
// entry carries a fallback.
const FINGER_MAP = [
  ['index1', ['IndexProximal']],
  ['index2', ['IndexIntermediate']],
  ['index3', ['IndexDistal']],
  ['middle1', ['MiddleProximal']],
  ['middle2', ['MiddleIntermediate']],
  ['middle3', ['MiddleDistal']],
  ['ring1', ['RingProximal']],
  ['ring2', ['RingIntermediate']],
  ['ring3', ['RingDistal']],
  ['pinky1', ['LittleProximal', 'PinkyProximal']],
  ['pinky2', ['LittleIntermediate', 'PinkyIntermediate']],
  ['pinky3', ['LittleDistal', 'PinkyDistal']],
  ['thumb1', ['ThumbMetacarpal', 'ThumbProximal']],
  ['thumb2', ['ThumbProximal', 'ThumbIntermediate']],
  ['thumb3', ['ThumbDistal']],
];

for (const side of ['left', 'right']) {
  for (const [suffix, targets] of FINGER_MAP) {
    BONE_MAP[`${side}_${suffix}`] = targets.map((name) => `${side}${name}`);
  }
}

const TAU_ROOT = 0.09;
const clamp = (v, lo, hi) => Math.min(hi, Math.max(lo, v));
const damp = (dt, tau) => (tau <= 0 ? 1 : 1 - Math.exp(-dt / tau));

/**
 * Drive a VRM humanoid directly from a BVH skeleton's bone rotations.
 *
 * Aiming bones at sampled joint positions reproduces limb *directions* but
 * loses each bone's twist, so the avatar can never match the tracked skeleton
 * exactly. This transfers the full world rotation of every BVH bone instead:
 *
 *     W_target = A * W_bvh          A = W_vrmRest * W_bvhRest^-1
 *
 * `A` is measured once in each rig's rest pose, so the difference in bone axis
 * conventions (and in the avatar's own root orientation) is absorbed by a
 * constant, and from there the motion is carried rigidly: every bone ends up
 * with the BVH bone's orientation, twist included.
 */
export class BVHRetargeter {
  constructor(vrm, root, skeleton, clip) {
    this.vrm = vrm;
    this.root = root;
    this.clip = clip;
    this.fps = 25;

    // The visible BVH preview has its own mixer; drive a clone so scrubbing
    // the avatar never fights the debug skeleton.
    this.rootBone = skeleton.bones[0].clone(true);
    this.holder = new THREE.Group();
    this.holder.add(this.rootBone);
    this.mixer = new THREE.AnimationMixer(this.rootBone);
    this.mixer.clipAction(clip).play();
    this.mixer.setTime(0);
    this.holder.updateMatrixWorld(true);

    this.entries = [];
    this.pelvis = this.rootBone;
    this.rootRef = null;
    this.torsoScale = 1;
    this._w = new THREE.Quaternion();
    this._p = new THREE.Quaternion();
    this._v = new THREE.Vector3();
    this._v2 = new THREE.Vector3();

    this.bind();
    this.root.position.set(0, 0, 0);
  }

  /** Resolve the VRM counterpart of each BVH bone and measure the alignment. */
  bind() {
    const vrmRest = this.restQuaternions();

    // Hierarchy order (parent before child) from the clone that is actually
    // driven; its rest pose is every bone at identity.
    const bones = [];
    this.rootBone.traverse((object) => { if (object.isBone) bones.push(object); });
    for (const bone of bones) bone.quaternion.identity();
    this.holder.updateMatrixWorld(true);
    const bvhRest = new Map(bones.map((bone) => [
      bone.name, bone.getWorldQuaternion(new THREE.Quaternion()),
    ]));

    const getNode = (name) => {
      const names = Array.isArray(name) ? name : [name];
      for (const candidate of names) {
        const node = this.vrm.humanoid.getNormalizedBoneNode(candidate);
        if (node) return node;
      }
      return null;
    };

    // The BVH skeleton list is in hierarchy order (parent before child), which
    // is what the update loop needs.
    for (const bone of bones) {
      const target = BONE_MAP[bone.name];
      if (!target) continue;
      const node = getNode(target);
      if (!node || !bvhRest.has(bone.name)) continue;
      const align = vrmRest.get(node)?.clone();
      if (!align) continue;
      align.multiply(bvhRest.get(bone.name).clone().invert());
      this.entries.push({ bvh: bone, vrm: node, align });
    }

    this.torsoScale = this.measureTorsoScale(bones);
    if (this.entries.length) this.entries[0].vrm.parent.updateWorldMatrix(true, false);
  }

  /** Save the VRM normalized rest orientations (identity) for alignment. */
  restQuaternions() {
    const nodes = [];
    for (const bone of Object.values(this.vrm.humanoid.normalizedHumanBones ?? {})) {
      if (bone?.node) nodes.push(bone.node);
    }

    const saved = new Map();
    for (const node of nodes) {
      saved.set(node, node.quaternion.clone());
      node.quaternion.identity();
    }
    this.vrm.scene.updateMatrixWorld(true);
    const rest = new Map();
    for (const node of nodes) rest.set(node, node.getWorldQuaternion(new THREE.Quaternion()));
    for (const [node, q] of saved) node.quaternion.copy(q);
    this.vrm.scene.updateMatrixWorld(true);
    return rest;
  }

  /**
   * The BVH is the tracked person's skeleton in metres; the avatar has its own
   * proportions. Scaling the root travel by the torso ratio keeps the avatar
   * covering the same ground as the person.
   */
  measureTorsoScale(bones) {
    const byName = new Map(bones.map((bone) => [bone.name, bone]));
    const mid = (a, b) => a.clone().add(b).multiplyScalar(0.5);
    const person = (() => {
      const hip = mid(byName.get('left_hip').getWorldPosition(new THREE.Vector3()),
        byName.get('right_hip').getWorldPosition(new THREE.Vector3()));
      const shoulder = mid(byName.get('left_shoulder').getWorldPosition(new THREE.Vector3()),
        byName.get('right_shoulder').getWorldPosition(new THREE.Vector3()));
      return hip.distanceTo(shoulder);
    })();

    const node = (name) => this.vrm.humanoid.getNormalizedBoneNode(name);
    const avatar = (() => {
      const hip = mid(node('leftUpperLeg').getWorldPosition(new THREE.Vector3()),
        node('rightUpperLeg').getWorldPosition(new THREE.Vector3()));
      const shoulder = mid(node('leftUpperArm').getWorldPosition(new THREE.Vector3()),
        node('rightUpperArm').getWorldPosition(new THREE.Vector3()));
      return hip.distanceTo(shoulder);
    })();

    if (!(person > 0.05) || !(avatar > 0.05)) return 1;
    return clamp(avatar / person, 0.4, 2.5);
  }

  /**
   * Pose the avatar at `time` seconds into the clip. Every bone is set in
   * hierarchy order, so a child's local rotation is solved against the parent
   * that was already posed this frame.
   */
  update(time, dt = 1 / 60) {
    if (!this.entries.length) return;
    this.mixer.setTime(Math.max(0, time));
    this.holder.updateMatrixWorld(true);
    this.vrm.scene.updateMatrixWorld(true);

    for (const { bvh, vrm, align } of this.entries) {
      if (!bvh) continue;
      bvh.getWorldQuaternion(this._w);
      this._w.premultiply(align);          // A * W_bvh
      vrm.parent.getWorldQuaternion(this._p);
      this._p.invert().multiply(this._w);  // parent^-1 * A * W_bvh
      vrm.quaternion.copy(this._p);
      vrm.updateWorldMatrix(false, false);
    }

    // Root travel: the pelvis moves the whole avatar, scaled to its own size.
    if (!this.pelvis) return;
    this.pelvis.getWorldPosition(this._v);
    if (!this.rootRef) this.rootRef = this._v.clone();
    this._v.sub(this.rootRef).multiplyScalar(this.torsoScale);
    this._v.x = clamp(this._v.x, -1.6, 1.6);
    this._v.y = clamp(this._v.y, -0.6, 0.6);
    this._v.z = clamp(this._v.z, -1.6, 1.6);
    this.root.position.lerp(this._v, damp(dt, TAU_ROOT));
  }

  reset() {
    this.rootRef = null;
    this.root.position.set(0, 0, 0);
    this.mixer.setTime(0);
  }
}
