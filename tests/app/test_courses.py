"""Obstacle courses and their scene-aware metrics.

The metric tests are built on **synthetic clips whose right answer is known**: a
"ghost" that walks through the geometry at floor height, and a "climber" that
plants each foot on the correct support. A metric that cannot separate those two
is not measuring the thing the study claims to measure, so each assertion below
is a statement about discriminating power, not just about not crashing.
"""

from __future__ import annotations

import numpy as np
import pytest

from backend.analysis import course_study as CS
from backend.analysis import courses as C
from backend.analysis import scene_eval as SE

PERIOD = 12          # frames per synthetic gait cycle
# Deliberately not a multiple of the tread rise or of SUPPORT_SLACK: a swing
# foot sitting EXACTLY on the support tolerance is a fixture artefact, and the
# discrimination tests should not turn on which side of the boundary it lands.
SWING_LIFT = 0.10    # m the swing foot clears its support by


def _clip(course, *, on_surface: bool, follow_path: bool = True) -> np.ndarray:
    """A crude 22-joint walker over a course.

    `on_surface=False` is the ghost: it keeps the pelvis at its start height and
    the feet at y=0 whatever the geometry says. `on_surface=True` puts every
    foot on the support that is actually under it and lets the pelvis ride the
    authored path. Stance feet are *held* at their touchdown xz so a clean gait
    reads as zero skate.
    """
    ref = C.dense_path(course)
    T = course["num_frames"]
    j = np.zeros((T, 22, 3))
    anchor: dict[int, tuple[float, float]] = {}
    for t in range(T):
        x, z = (ref[t, 0], ref[t, 2]) if follow_path else (0.0, ref[0, 2] + 2.4 * t / (T - 1))
        j[t, :, 0] = x
        j[t, :, 2] = z
        j[t, :, 1] = ref[t, 1] if on_surface else ref[0, 1]
        for k, idx in enumerate(SE.FOOT_IDX):
            leg = 0 if k < 2 else 1
            swing = (t // (PERIOD // 2)) % 2 == leg
            sup = SE.ground_under(float(x), float(z), course["objects"])[0] if on_surface else 0.0
            fx, fz = x + (-0.12 if leg == 0 else 0.12), z + (0.05 if k in (1, 3) else 0.0)
            if swing or k not in anchor:
                anchor[k] = (fx, fz)
            else:
                fx, fz = anchor[k]
            j[t, idx] = [fx, sup + (SWING_LIFT if swing else 0.0), fz]
    return j


# --------------------------------------------------------------------------- courses


@pytest.mark.parametrize("key", list(C.COURSES))
def test_requested_speed_is_walkable(key):
    """A path faster than the model's gait manufactures the foot-skate the study
    is trying to measure. Every course must stay well under it."""
    speed = C.check_speed(C.COURSES[key])
    assert speed["ok"], f"{key} asks for {speed['speed_mps']} m/s"
    assert speed["speed_mps"] < 0.45


def test_padding_is_zero_on_every_course():
    """`scene_energy` applies the standoff margin to the FLOOR as well as the
    walls, so any positive padding levitates the body and silently zeroes the
    foot-skate metric. A course about standing on surfaces cannot use it."""
    for course in C.COURSES.values():
        assert C.scene_payload(course)["padding"] == 0.0


def test_stairs_treads_are_contiguous_and_rising():
    """No gap between treads (a foot could be pushed into it) and a monotone rise."""
    treads = [o for o in C.COURSES["stairs"]["objects"] if o["id"].startswith("tread")]
    treads.sort(key=lambda o: o["z"])
    tops = [o["y"] + o["h"] / 2 for o in treads]
    assert tops == sorted(tops) and len(set(tops)) == len(tops)
    for a, b in zip(treads, treads[1:]):
        assert abs((a["z"] + a["d"] / 2) - (b["z"] - b["d"] / 2)) < 1e-9
    # Each tread is solid from the floor up, so nothing can slip underneath.
    assert all(abs(o["y"] - o["h"] / 2) < 1e-9 for o in treads)


def test_corner_leaves_exactly_one_opening():
    """The point of the course: straight on and right are walled, left is open.

    Probed on the geometry rather than asserted from the numbers that built it —
    a wall that fails to overlap its neighbour would pass a coordinate check and
    still leak.
    """
    course = C.COURSES["corner"]
    objs = course["objects"]

    def blocked(x, z):
        p = np.array([[[x, 0.9, z]]], dtype=np.float64)
        return any((SE.obstacle_sdf(p, o) < 0).any() for o in objs)

    assert blocked(0.0, 0.78), "straight ahead must be walled"
    assert blocked(0.78, 0.0), "the right branch must be walled"
    assert not blocked(-1.1, 0.0), "the left branch must be open"
    assert not blocked(0.0, -1.0), "the approach must be clear"


@pytest.mark.parametrize("key", list(C.COURSES))
def test_trajectory_payload_matches_the_authored_path(key):
    """The enforced array and the scored array must be the same array."""
    course = C.COURSES[key]
    payload = C.trajectory_payload(course)
    con = payload["constraints"][0]
    dense = C.dense_path(course)
    assert con["joint"] == "pelvis"
    assert con["axes"] == course["axes"]
    assert len(con["frames"]) == len(con["points"])
    assert con["frames"][0] == 0 and con["frames"][-1] == course["num_frames"] - 1
    for f, p in zip(con["frames"], con["points"]):
        assert np.allclose(p, dense[f], atol=1e-4)
    # retime makes this PATH control; blend is ignored, so it must not be relied on.
    assert payload["retime"] is True and payload["mode"] == "project"


def test_job_params_keep_the_arms_a_matched_pair():
    course = C.COURSES["slab"]
    bodies = [C.job_params(course, arm, 0, "ckpt.pt", "dit_mid", "rmg_mid")
              for arm in ("free", "room", "room+traj")]
    # Same prompts, seed and frame count ⇒ same batch shape ⇒ same noise draw.
    assert len({b["prompts"] for b in bodies}) == 1
    assert len({b["seed"] for b in bodies}) == 1
    assert len({b["num_frames"] for b in bodies}) == 1
    assert "scene" not in bodies[0] and "trajectory" not in bodies[0]
    assert "scene" in bodies[1] and "trajectory" not in bodies[1]
    assert "scene" in bodies[2] and "trajectory" in bodies[2]


# --------------------------------------------------------------------------- support


def test_support_is_one_sided():
    """A swing foot passing UNDER a tread nose is not standing on it.

    Symmetric slack was a real bug: it credited a clip that walked straight
    through the flight with a climbed tread.
    """
    objs = C.COURSES["stairs"]["objects"]
    x, z = 0.0, 0.175                      # over tread1, whose top is 0.15
    assert SE.support_surface(x, z, 0.15, objs)[1] == "tread1"
    assert SE.support_surface(x, z, 0.10, objs)[1] is None   # 5 cm below the top
    assert SE.support_surface(x, z, 0.0, objs)[0] == 0.0     # on the floor beside it
    # …but "what is under this xz" ignores the body's height entirely.
    assert SE.ground_under(x, z, objs)[1] == "tread1"


def test_foot_skate_is_none_not_zero_when_nothing_touches():
    """A levitating clip has no foot-skate. Returning 0.0 would award it the
    best possible score — the exact failure mode this module exists to fix."""
    course = C.COURSES["slab"]
    j = _clip(course, on_surface=True)
    j[:, list(SE.FOOT_IDX), 1] += 1.0      # lift the whole body off every surface
    res = SE.evaluate(j, course, "room")
    assert res["foot_skate_mean"] is None
    assert res["n_planted_transitions"] == 0


# --------------------------------------------------------------------------- discrimination


@pytest.mark.parametrize("key", list(C.COURSES))
def test_penetration_separates_ghost_from_climber(key):
    course = C.COURSES[key]
    ghost = SE.evaluate(_clip(course, on_surface=False), course, "room")
    climb = SE.evaluate(_clip(course, on_surface=True), course, "room+traj")
    assert climb["penetration_max"] < 0.01
    if key != "corner":     # the corner ghost still follows the (flat) path
        # The ghost's depth is bounded by the geometry it walks through (half a
        # slab, a whole tread), so assert the SEPARATION rather than a level.
        assert ghost["penetration_max"] > 10 * max(climb["penetration_max"], 1e-3)
        assert ghost["penetration_frame_frac"] > 0.2


def test_slab_mount_success():
    course = C.COURSES["slab"]
    assert SE.evaluate(_clip(course, on_surface=False), course, "room")["success"]["mounted"] is False
    ok = SE.evaluate(_clip(course, on_surface=True), course, "room+traj")["success"]
    assert ok["mounted"] is True and ok["both_legs_touched"] is True


def test_stairs_climb_is_counted_in_order():
    course = C.COURSES["stairs"]
    ghost = SE.evaluate(_clip(course, on_surface=False), course, "room")["success"]
    climb = SE.evaluate(_clip(course, on_surface=True), course, "room+traj")["success"]
    assert ghost["treads_climbed"] == 0 and ghost["top_reached"] is False
    assert climb["treads_climbed"] == climb["treads_total"]
    frames = [climb["first_contact_frame"][o] for o in course["success"]["object_ids"]]
    assert frames == sorted(frames), "treads must be contacted in ascending order"


def test_corner_turn_direction_and_corridor():
    course = C.COURSES["corner"]
    turned = SE.evaluate(_clip(course, on_surface=False), course, "room+traj")["success"]
    straight = SE.evaluate(_clip(course, on_surface=False, follow_path=False),
                           course, "room")["success"]
    assert turned["turned_left"] is True and turned["turn_left_deg"] > 60
    assert turned["exit_reached"] is True
    assert straight["turned_left"] is False
    assert straight["inside_corridor_frac"] < turned["inside_corridor_frac"]


def test_free_arm_is_placed_into_room_coordinates():
    """A prompt-only clip is sampled in the model's canonical frame. Unplaced, it
    would be scored against a room it was never in."""
    course = C.COURSES["slab"]
    j = _clip(course, on_surface=False)
    j[:, :, 0] += 7.0                      # shove it far outside the room
    j[:, :, 2] += 7.0
    placed = SE.place_free(j, course["spawn"])
    assert abs(placed[0, 0, 0] - course["spawn"]["x"]) < 1e-6
    assert abs(placed[0, 0, 2] - course["spawn"]["z"]) < 1e-6
    # …and `evaluate` does it for the free arm but not for an already-placed one.
    assert SE.evaluate(j, course, "free")["outside_room_frac"] == 0.0
    assert SE.evaluate(j, course, "room")["outside_room_frac"] > 0.0


# --------------------------------------------------------------------------- aggregation


def test_aggregate_reports_rates_for_booleans_and_sd_for_numbers():
    course = C.COURSES["stairs"]
    rows = [SE.evaluate(_clip(course, on_surface=True), course, "room+traj"),
            SE.evaluate(_clip(course, on_surface=False), course, "room+traj"),
            SE.evaluate(_clip(course, on_surface=False), course, "room")]
    table = SE.aggregate(rows)
    by_arm = {r["arm"]: r for r in table}
    assert by_arm["room+traj"]["n"] == 2
    assert by_arm["room+traj"]["metrics"]["penetration_max"]["sd"] > 0
    assert by_arm["room+traj"]["success"]["top_reached_rate"] == 0.5
    # A single-clip cell must not claim a spread it cannot have.
    assert by_arm["room"]["metrics"]["penetration_max"]["sd"] == 0.0


def test_deltas_compare_each_arm_to_the_right_baseline():
    course = C.COURSES["stairs"]
    rows = [SE.evaluate(_clip(course, on_surface=False), course, "free"),
            SE.evaluate(_clip(course, on_surface=False), course, "room"),
            SE.evaluate(_clip(course, on_surface=True), course, "room+traj")]
    ds = {(d["course"], d["arm"]): d for d in CS.deltas(SE.aggregate(rows))}
    assert ds[("stairs", "room")]["vs"] == "free"
    assert ds[("stairs", "room+traj")]["vs"] == "room"
    pen = ds[("stairs", "room+traj")]["metrics"]["penetration_max"]
    assert pen["delta"] < 0 and pen["better"] is True


def test_resolve_media_follows_the_url_not_the_job_id(tmp_path):
    """Viz jobs are FUSED: a follower's outputs keep the lead's media directory.

    Rebuilding the path from the follower's own job id silently drops every arm
    but the lead — the pilot lost 6 of 9 clips that way.
    """
    lead = tmp_path / "jobs" / "leadjob"
    lead.mkdir(parents=True)
    (lead / "gen-03.npy").write_bytes(b"")
    got = CS.resolve_media(tmp_path, "/media/jobs/leadjob/gen-03.npy")
    assert got is not None and got.exists()
    # …and a URL that climbs out of the media root resolves to nothing.
    assert CS.resolve_media(tmp_path, "/media/../../etc/passwd") is None
    assert CS.resolve_media(tmp_path, "") is None


def test_skate_is_withheld_when_there_are_too_few_planted_frames():
    """A near-levitating clip must not post a small, flattering skate.

    Measured on the pilot: a floating arm produced 0.273 m/s from 2 planted
    transitions while the arm that actually walked produced 0.688 from 125. Read
    naively the floater wins, so below `MIN_SKATE_SAMPLES` the metric is absent
    rather than small — and the contact fraction says why.
    """
    course = C.COURSES["slab"]
    j = _clip(course, on_surface=True)
    # Lift everything but a couple of frames clear of every surface.
    j[2:, list(SE.FOOT_IDX), 1] += 1.0
    res = SE.evaluate(j, course, "room")
    assert res["n_planted_transitions"] < SE.MIN_SKATE_SAMPLES
    assert res["foot_skate_mean"] is None
    assert res["slide_per_root_step"] is None
    assert res["contact_frame_frac"] < 0.1     # the denominator is visible


def test_corner_reports_entry_heading_separately_from_the_turn():
    """Exact spawn placement aligns NET displacement to the spawn arrow, so an
    L-shaped walk enters the corridor rotated. Entry heading and turn must be
    separable, or a placement artefact reads as a failure to turn."""
    course = C.COURSES["corner"]
    straight = SE.evaluate(_clip(course, on_surface=False, follow_path=False),
                           course, "room")["success"]
    turned = SE.evaluate(_clip(course, on_surface=False), course, "room+traj")["success"]
    assert abs(straight["entry_heading_deg"]) < 5      # walked in along +Z
    assert abs(turned["entry_heading_deg"]) < 5        # the authored path also enters along +Z
    assert turned["turn_left_deg"] > 60 and abs(straight["turn_left_deg"]) < 5


def test_runs_at_different_step_counts_do_not_pool(tmp_path):
    """A 200-step pipeline pilot and an 800-step study share (course, arm, seed).

    Pooling them would average two different experiments into one cell, so the
    report keeps only the most thoroughly sampled configuration and says what it
    set aside.
    """
    course = C.COURSES["slab"]
    lead = tmp_path / "jobs" / "j"
    lead.mkdir(parents=True)
    np.save(lead / "a.npy", _clip(course, on_surface=True))
    np.save(lead / "b.npy", _clip(course, on_surface=False))

    def job(jid, steps, name):
        return {"id": jid, "state": "done",
                "params": {"num_steps": steps, "guidance": 6.5,
                           "ablation": {"study": "course", "course": "slab",
                                        "arm": "room+traj", "seed": 0}},
                "outputs": [{"npy_url": f"/media/jobs/j/{name}", "media_url": "", "caption": ""}]}

    out = CS.collect([job("pilot", 200, "b.npy"), job("study", 800, "a.npy")], tmp_path)
    assert out["num_steps"] == 800
    assert out["n_clips"] == 1 and out["n_clips_all"] == 2
    assert out["other_configs"] == [{"num_steps": 200, "n_clips": 1}]
    # the 800-step clip is the climber, so the surviving cell must show the mount
    assert out["table"][0]["success"]["mounted_rate"] == 1.0
