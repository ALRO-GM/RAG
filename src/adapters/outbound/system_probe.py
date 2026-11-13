"""Hardware probe adapter: CPU topology, memory and GPUs of this host."""

import os
from pathlib import Path
from typing import Dict, List, Optional, Set

from src.domain.runtime import GraphicsCard, Gpu, HardwareProfile

SYSFS_CPU = "/sys/devices/system/cpu"
SYSFS_DEVICES = "/sys/devices"
SYSFS_PCI = "/sys/bus/pci/devices"
MEMINFO = "/proc/meminfo"
PCI_VENDORS = {"0x10de": "NVIDIA", "0x1002": "AMD", "0x8086": "Intel"}
FAST_RATIO = 0.9


def parse_cpu_list(text: str) -> List[int]:
    """Parse a kernel CPU list such as ``0-3,8,10-11``."""
    cpus: List[int] = []
    for part in text.strip().split(","):
        if not part:
            continue
        first, _, last = part.partition("-")
        try:
            cpus.extend(range(int(first), int(last or first) + 1))
        except ValueError:
            continue
    return cpus


def _read(path: Path) -> Optional[str]:
    """Return the text of a small system file, if readable."""
    try:
        return path.read_text(encoding="ascii", errors="ignore").strip()
    except OSError:
        return None


class SystemProbe:
    """Describe the machine from the operating system.

    On Linux the core layout comes from sysfs. Hybrid processors are
    recognised by the kernel's ``cpu_core`` / ``cpu_atom`` devices
    (Intel), else by ``cpu_capacity`` (ARM big.LITTLE), else by the
    maximum frequency of each core. Anywhere else, or when sysfs is
    unreadable, every logical CPU is treated as its own equal core,
    which makes the plan fall back to "one thread per CPU".
    """

    def __init__(
        self,
        sysfs_cpu: str = SYSFS_CPU,
        sysfs_devices: str = SYSFS_DEVICES,
        meminfo: str = MEMINFO,
        detect_gpu: bool = True,
        sysfs_pci: str = SYSFS_PCI,
    ):
        """Read from the given system paths (overridable for tests)."""
        self.sysfs_pci = Path(sysfs_pci)
        self.sysfs_cpu = Path(sysfs_cpu)
        self.sysfs_devices = Path(sysfs_devices)
        self.meminfo = Path(meminfo)
        self.detect_gpu = detect_gpu

    def allowed_cpus(self) -> Set[int]:
        """Return the logical CPUs this process may run on."""
        if hasattr(os, "sched_getaffinity"):
            return set(os.sched_getaffinity(0))
        return set(range(os.cpu_count() or 1))

    def cores(self, allowed: Set[int]) -> Dict[int, List[int]]:
        """Group the allowed logical CPUs by physical core."""
        groups: Dict[str, List[int]] = {}
        for cpu in sorted(allowed):
            topology = self.sysfs_cpu / f"cpu{cpu}" / "topology"
            core = _read(topology / "core_id")
            package = _read(topology / "physical_package_id") or "0"
            key = f"{package}:{core}" if core is not None else f"cpu{cpu}"
            groups.setdefault(key, []).append(cpu)
        return {i: cpus for i, cpus in enumerate(groups.values())}

    def _rating(self, cpu: int) -> Optional[float]:
        """Return a speed rating of a CPU: capacity, else max frequency."""
        base = self.sysfs_cpu / f"cpu{cpu}"
        for name in ("cpu_capacity", "cpufreq/cpuinfo_max_freq"):
            value = _read(base / name)
            if value and value.isdigit():
                return float(value)
        return None

    def performance_cores(self, cores: Dict[int, List[int]]) -> List[int]:
        """Return the cores of the fastest class."""
        declared = _read(self.sysfs_devices / "cpu_core" / "cpus")
        efficient = _read(self.sysfs_devices / "cpu_atom" / "cpus")
        if declared and efficient:
            fast = set(parse_cpu_list(declared))
            chosen = [c for c, cpus in cores.items() if set(cpus) & fast]
            if chosen:
                return chosen
        ratings = {c: self._rating(min(cpus)) for c, cpus in cores.items()}
        known = [r for r in ratings.values() if r is not None]
        if len(known) < len(cores) or not known:
            return list(cores)
        top = max(known)
        return [
            c for c, r in ratings.items()
            if r is not None and r >= FAST_RATIO * top
        ]

    def memory(self) -> tuple[float, float]:
        """Return (total, available) memory in GB."""
        text = _read(self.meminfo) or ""
        values: Dict[str, float] = {}
        for line in text.splitlines():
            name, _, rest = line.partition(":")
            fields = rest.split()
            if fields and fields[0].isdigit():
                values[name] = int(fields[0]) / 1024 / 1024
        if "MemTotal" in values:
            total = values["MemTotal"]
            return total, values.get("MemAvailable", total)
        try:
            pages = os.sysconf("SC_PHYS_PAGES")
            total = pages * os.sysconf("SC_PAGE_SIZE") / 1024 ** 3
            return total, total
        except (ValueError, OSError, AttributeError):
            return 0.0, 0.0

    def gpus(self) -> List[Gpu]:
        """Return the accelerators the installed PyTorch can use."""
        if not self.detect_gpu:
            return []
        try:
            import torch
        except ImportError:
            return []
        found: List[Gpu] = []
        try:
            if torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    props = torch.cuda.get_device_properties(i)
                    found.append(Gpu(
                        name=str(props.name),
                        memory_gb=props.total_memory / 1024 ** 3,
                    ))
            mps = getattr(torch.backends, "mps", None)
            if mps is not None and mps.is_available():
                total, _ = self.memory()
                found.append(Gpu(
                    name="Apple GPU", memory_gb=total * 0.6, backend="mps"
                ))
        except (RuntimeError, AssertionError):
            return found
        return found

    def cards(self) -> List[GraphicsCard]:
        """Return the graphics cards on the PCI bus and their drivers.

        This sees a card even when no driver or framework can use it,
        which is what makes a useful diagnosis possible.
        """
        found: List[GraphicsCard] = []
        try:
            devices = sorted(self.sysfs_pci.iterdir())
        except OSError:
            return found
        for device in devices:
            kind = _read(device / "class") or ""
            if not kind.startswith("0x03"):
                continue
            vendor = _read(device / "vendor") or ""
            try:
                driver = os.path.basename(os.readlink(device / "driver"))
            except OSError:
                driver = ""
            found.append(GraphicsCard(
                vendor=PCI_VENDORS.get(vendor, vendor), driver=driver
            ))
        return found

    def torch_build(self) -> tuple[str, bool]:
        """Return PyTorch's version and whether it was built with CUDA."""
        if not self.detect_gpu:
            return "", False
        try:
            import torch
        except ImportError:
            return "not installed", False
        return str(torch.__version__), torch.version.cuda is not None

    def profile(self) -> HardwareProfile:
        """Return the hardware profile of this process."""
        cores = self.cores(self.allowed_cpus())
        total, available = self.memory()
        return HardwareProfile(
            cores=cores,
            performance_cores=self.performance_cores(cores),
            memory_gb=round(total, 2),
            available_memory_gb=round(available, 2),
            gpus=self.gpus(),
            cards=self.cards(),
            torch_build=self.torch_build()[0],
            torch_cuda=self.torch_build()[1],
        )
