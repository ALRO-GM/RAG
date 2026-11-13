"""Hardware probe, plan persistence and plan application."""

import os
import subprocess
import sys
from pathlib import Path
from typing import List, Set

import pytest

from src.adapters.outbound import torch_runtime
from src.adapters.outbound.system_probe import SystemProbe, parse_cpu_list
from src.adapters.outbound.torch_runtime import (
    apply_plan,
    load_measured,
    prepare,
    save_measured,
)
from src.domain.runtime import HardwareProfile, RuntimePlan


def fake_sysfs(
    tmp_path: Path,
    core_ids: List[int],
    max_freq: List[int],
    hybrid_lists: bool = False,
) -> SystemProbe:
    """Build a sysfs tree: one entry per logical CPU."""
    cpu_dir = tmp_path / "cpu"
    for cpu, (core, freq) in enumerate(zip(core_ids, max_freq)):
        topology = cpu_dir / f"cpu{cpu}" / "topology"
        topology.mkdir(parents=True)
        (topology / "core_id").write_text(f"{core}\n")
        (topology / "physical_package_id").write_text("0\n")
        freq_dir = cpu_dir / f"cpu{cpu}" / "cpufreq"
        freq_dir.mkdir()
        (freq_dir / "cpuinfo_max_freq").write_text(f"{freq}\n")
    devices = tmp_path / "devices"
    if hybrid_lists:
        (devices / "cpu_core").mkdir(parents=True)
        (devices / "cpu_atom").mkdir(parents=True)
        (devices / "cpu_core" / "cpus").write_text("0-3\n")
        (devices / "cpu_atom" / "cpus").write_text("4-5\n")
    meminfo = tmp_path / "meminfo"
    meminfo.write_text(
        "MemTotal:       16777216 kB\nMemAvailable:    8388608 kB\n"
    )
    probe = SystemProbe(str(cpu_dir), str(devices), str(meminfo), False)
    allowed: Set[int] = set(range(len(core_ids)))
    probe.allowed_cpus = lambda: allowed  # type: ignore[method-assign]
    return probe


def test_parse_cpu_list() -> None:
    """Kernel CPU lists with ranges, singles and junk are parsed."""
    assert parse_cpu_list("0-3,8,10-11\n") == [0, 1, 2, 3, 8, 10, 11]
    assert parse_cpu_list("") == []
    assert parse_cpu_list("x,2") == [2]


def test_probe_reads_intel_hybrid_layout(tmp_path: Path) -> None:
    """cpu_core / cpu_atom lists mark the performance cores."""
    probe = fake_sysfs(
        tmp_path, [0, 0, 1, 1, 2, 3], [5000] * 4 + [3800] * 2, True
    )
    profile = probe.profile()
    assert profile.cores == {0: [0, 1], 1: [2, 3], 2: [4], 3: [5]}
    assert profile.performance_cores == [0, 1]
    assert profile.memory_gb == 16.0
    assert profile.available_memory_gb == 8.0
    assert profile.gpus == []


def test_probe_falls_back_to_frequencies(tmp_path: Path) -> None:
    """Without the hybrid lists, slower cores are found by max frequency."""
    probe = fake_sysfs(tmp_path, [0, 1, 2, 3], [5000, 4900, 3800, 3800])
    assert probe.profile().performance_cores == [0, 1]
    same = fake_sysfs(tmp_path / "b", [0, 0, 1, 1], [4000] * 4)
    profile = same.profile()
    assert profile.performance_cores == [0, 1]
    assert not profile.is_hybrid()


def test_probe_without_sysfs_treats_cpus_as_equal_cores(
    tmp_path: Path,
) -> None:
    """Unreadable topology: every CPU is its own core, none slower."""
    probe = SystemProbe(
        str(tmp_path / "no"), str(tmp_path / "no"), str(tmp_path / "no"),
        False,
    )
    probe.allowed_cpus = lambda: {0, 1, 2}  # type: ignore[method-assign]
    profile = probe.profile()
    assert len(profile.cores) == 3
    assert not profile.is_hybrid()
    assert profile.memory_gb >= 0.0


def test_probe_respects_the_allowed_cpus(tmp_path: Path) -> None:
    """CPUs outside the process affinity are not offered."""
    probe = fake_sysfs(tmp_path, [0, 0, 1, 1], [4000] * 4)
    probe.allowed_cpus = lambda: {0, 1}  # type: ignore[method-assign]
    assert probe.profile().cores == {0: [0, 1]}


def test_real_probe_describes_this_machine() -> None:
    """On the real host the profile is consistent."""
    profile = SystemProbe(detect_gpu=False).profile()
    assert set(profile.logical_cpus()) == set(os.sched_getaffinity(0))
    assert profile.performance_cores
    assert set(profile.performance_cores) <= set(profile.cores)
    assert profile.memory_gb > 0


def profile_of(cpus: int) -> HardwareProfile:
    """Return a homogeneous profile with ``cpus`` single-thread cores."""
    return HardwareProfile(
        cores={c: [c] for c in range(cpus)},
        performance_cores=list(range(cpus)),
        memory_gb=8.0, available_memory_gb=8.0,
    )


def test_measured_plan_round_trip(tmp_path: Path) -> None:
    """A saved plan is loaded back for the same hardware only."""
    path = str(tmp_path / "sub" / "plan.json")
    plan = RuntimePlan(threads=2, cpus=[0, 1], name="fast")
    save_measured(path, profile_of(4), plan)
    assert load_measured(path, profile_of(4)) == plan
    assert load_measured(path, profile_of(8)) is None
    assert load_measured(str(tmp_path / "missing.json"), profile_of(4)) is None
    Path(path).write_text("not json")
    assert load_measured(path, profile_of(4)) is None


def test_measured_plan_with_unavailable_cpus_is_ignored(
    tmp_path: Path,
) -> None:
    """A plan pinning CPUs this process cannot use is not applied."""
    path = str(tmp_path / "plan.json")
    wide = profile_of(4)
    save_measured(path, wide, RuntimePlan(threads=2, cpus=[2, 3]))
    narrow = wide.model_copy(
        update={"cores": {0: [0], 1: [1], 2: [4], 3: [5]}}
    )
    assert narrow.signature() == wide.signature()
    assert load_measured(path, narrow) is None


def test_apply_plan_pins_once_and_sets_threads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The first plan is applied; later ones do not change the process."""
    import torch

    pinned: List[Set[int]] = []
    threads: List[int] = []
    monkeypatch.setattr(
        os, "sched_setaffinity", lambda pid, cpus: pinned.append(set(cpus))
    )
    monkeypatch.setattr(torch, "set_num_threads", threads.append)
    monkeypatch.setattr(torch_runtime, "_applied", None)
    first = apply_plan(RuntimePlan(threads=2, cpus=[0, 1]))
    second = apply_plan(RuntimePlan(threads=9, cpus=[5]))
    assert first is second
    assert (pinned, threads) == ([{0, 1}], [2])
    assert prepare(None) == "cpu"
    assert threads == [2]


def test_apply_plan_survives_a_refused_affinity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A platform that refuses pinning still gets its thread count."""
    import torch

    def refuse(pid: int, cpus: Set[int]) -> None:
        raise OSError("not permitted")

    threads: List[int] = []
    monkeypatch.setattr(os, "sched_setaffinity", refuse)
    monkeypatch.setattr(torch, "set_num_threads", threads.append)
    monkeypatch.setattr(torch_runtime, "_applied", None)
    assert apply_plan(RuntimePlan(threads=3, cpus=[0])).threads == 3
    assert threads == [3]


def test_default_without_plan_keeps_every_cpu(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Adapters built without a plan behave as before: all CPUs."""
    import torch

    threads: List[int] = []
    monkeypatch.setattr(torch, "set_num_threads", threads.append)
    monkeypatch.setattr(torch_runtime, "_applied", None)
    assert prepare(None) == "cpu"
    assert threads == [os.cpu_count() or 1]


def test_system_command_reports_hardware_and_plan(tmp_path: Path) -> None:
    """``system`` prints the CPU, the plan and the chosen judge model."""
    done = subprocess.run(
        [sys.executable, "-m", "src", "system"],
        capture_output=True, text=True, timeout=120,
    )
    assert done.returncode == 0, done.stderr
    for word in ("CPU:", "Memory:", "GPU:", "Plan: device", "Skills judge"):
        assert word in done.stdout


def test_system_command_rejects_unknown_device() -> None:
    """A device the machine lacks is a one-line error."""
    done = subprocess.run(
        [sys.executable, "-m", "src", "system", "--device", "tpu"],
        capture_output=True, text=True, timeout=120,
    )
    assert done.returncode == 1
    assert done.stderr.strip() == (
        "Error: Unknown device 'tpu'; choose auto, cpu, cuda or mps."
    )


def test_probe_lists_graphics_cards_and_their_driver(tmp_path: Path) -> None:
    """Display devices on the PCI bus are reported with their driver."""
    pci = tmp_path / "pci"
    drivers = tmp_path / "drivers"
    for name, vendor, kind, driver in (
        ("0000:01:00.0", "0x10de", "0x030000", "nouveau"),
        ("0000:01:00.1", "0x10de", "0x040300", "snd_hda_intel"),
        ("0000:00:02.0", "0x8086", "0x030000", ""),
    ):
        device = pci / name
        device.mkdir(parents=True)
        (device / "vendor").write_text(vendor + "\n")
        (device / "class").write_text(kind + "\n")
        if driver:
            (drivers / driver).mkdir(parents=True)
            (device / "driver").symlink_to(drivers / driver)
    probe = SystemProbe(detect_gpu=False, sysfs_pci=str(pci))
    cards = probe.cards()
    assert [(c.vendor, c.driver) for c in cards] == [
        ("Intel", ""), ("NVIDIA", "nouveau"),
    ]
    assert SystemProbe(
        detect_gpu=False, sysfs_pci=str(tmp_path / "none")
    ).cards() == []
