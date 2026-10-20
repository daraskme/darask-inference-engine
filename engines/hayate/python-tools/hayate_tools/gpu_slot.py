"""GPU slot (protocol v2, docs/GPU_SHARING.md): one heavy GPU job at a time, next to serve's resident LLM.

  python3 -m hayate_tools.gpu_slot --need 30 --who "hayate-train krea2" [--priority 5] [--label TEXT] [--timeout S] [--lock F] -- cmd...

replaces `flock gpu.lock cmd...`. Waiters take the slot in (priority, since) order: lower priority first (0 most
urgent, default 5); only the first declaration in that order tries the flock. The waiter declares its VRAM need in gpu.d/<pid>.json next to the lock (written
atomically), takes the flock (`hayate serve` hands it over, unloading idle Krea 2 / MOSS / H3 only when the free VRAM does
not cover the need), waits until need + 1 GiB is free, removes the declaration, and runs. The command inherits the
lock, so it is held until both the command and this process have ended. The exit status is the command's (128 + N
for a signal); 124 when --timeout passes first. In Python:

    with gpu_slot(30, "hayate-train krea2", on_wait=lambda: print("waiting")):
        ...

$HAYATE_GPU_LOCK overrides the lock path (default /mnt/NVME4TB/darask-inference-engine/local/hayate/spill/gpu.lock). Standard library only; the
free VRAM comes from nvidia-smi, or from torch when nvidia-smi is missing."""
import argparse
import contextlib
import fcntl
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

GIB = 1 << 30
MARGIN_GIB = 1  # serve uses the same margin (crates/server/src/gpu.rs)
POLL = 0.5
DEFAULT_PRIORITY = 5  # serve uses the same default
AGE_SECONDS = 30 * 60  # aging: one priority step per 30 min waited ...
AGE_FLOOR = 2  # ... never past 2, so user requests (0/1) stay ahead (serve uses the same)
DEFAULT_LOCK = "/mnt/NVME4TB/darask-inference-engine/local/hayate/spill/gpu.lock"
DESKTOP_GIB = 1  # never free: compositor, browsers (the display is on a separate GT 710 since 2026-10-17); a need above
# total - this would hold the lock forever


def lock_path(lock=None):
    return Path(lock or os.environ.get("HAYATE_GPU_LOCK", DEFAULT_LOCK))


def declaration_dir(lock=None):
    return lock_path(lock).with_name("gpu.d")


def free_gib():
    """Free VRAM of the first GPU in GiB."""
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, check=True, timeout=30).stdout
        return float(out.split()[0]) / 1024
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        import torch
        return torch.cuda.mem_get_info()[0] / GIB


def total_gib():
    """Total VRAM of the first GPU in GiB (None if unknown)."""
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, check=True, timeout=30).stdout
        return float(out.split()[0]) / 1024
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        return None


def boot_time():
    with open("/proc/stat") as f:
        return next(int(line.split()[1]) for line in f if line.startswith("btime "))


def alive(pid, since=None):
    """pid is running and, when `since` is given, the declaration was written after this boot (pids are reused)."""
    return pid > 0 and Path(f"/proc/{pid}").exists() and (since is None or since >= boot_time())


def effective_priority(d, now=None):
    """Priority after aging against starvation: max(2, priority - waited // 30 min); 0..2 never change."""
    if d["priority"] <= AGE_FLOOR:
        return d["priority"]
    waited = max(0, int(time.time() if now is None else now) - d["since"])
    return max(AGE_FLOOR, d["priority"] - waited // AGE_SECONDS)


def declarations(lock=None):
    """Live declarations [{pid, who, need_gib, since, priority, label}] in slot order (effective priority, since,
    pid); dead pids and broken files are skipped."""
    out = []
    for path in declaration_dir(lock).glob("[!.]*.json"):
        try:
            d = json.loads(path.read_text())
            d = dict(pid=int(d["pid"]), who=str(d["who"]), need_gib=float(d["need_gib"]), since=int(d["since"]),
                     priority=int(d.get("priority", DEFAULT_PRIORITY)), label=d.get("label"))
            if alive(d["pid"], d["since"]) and d["need_gib"] >= 0:
                out.append(d)
        except (OSError, ValueError, KeyError, TypeError):
            continue
    now = time.time()
    return sorted(out, key=lambda d: (effective_priority(d, now), d["since"], d["pid"]))


def declare(need_gib, who, lock=None, priority=DEFAULT_PRIORITY, label=None):
    """Write gpu.d/<pid>.json atomically (temporary file, then rename) and return its path."""
    directory = declaration_dir(lock)
    directory.mkdir(parents=True, exist_ok=True)
    pid = os.getpid()
    path, temporary = directory / f"{pid}.json", directory / f".{pid}.json.tmp"
    temporary.write_text(json.dumps(dict(pid=pid, who=who, need_gib=need_gib, since=int(time.time()),
                                         priority=priority, label=label), ensure_ascii=False))
    os.replace(temporary, path)
    return path


def _flock(file, deadline):
    """Block in flock (other holders see a waiter in /proc/locks); TimeoutError after `deadline`."""
    if deadline is None:
        fcntl.flock(file, fcntl.LOCK_EX)
        return
    # The lock belongs to the open file, shared by the duplicate: if we give up, the thread's flock
    # still completes later and closing the last descriptor releases it at once.
    fd = os.dup(file.fileno())
    result = {}

    def run():
        try:
            fcntl.flock(fd, fcntl.LOCK_EX)
        except OSError as error:
            result["error"] = error
        finally:
            os.close(fd)
            done.set()

    done = threading.Event()
    threading.Thread(target=run, daemon=True).start()
    if not done.wait(max(deadline - time.monotonic(), 0)):
        raise TimeoutError("timed out waiting for the GPU lock")
    if "error" in result:
        raise result["error"]


def first_in_line(lock=None, pid=None):
    """Whether `pid`'s declaration is first in slot order (or no live declaration precedes it)."""
    pid = os.getpid() if pid is None else pid
    line = declarations(lock)
    return not line or line[0]["pid"] == pid or all(d["pid"] != pid for d in line)


def acquire(need_gib, who, lock=None, timeout=None, on_wait=None, free=free_gib, priority=DEFAULT_PRIORITY,
            label=None):
    """Declare, take the lock, wait for need_gib + 1 GiB free, withdraw the declaration. Returns the open lock file
    (release() it). on_wait() is called once if the lock is busy or the VRAM is short."""
    path = lock_path(lock)
    deadline = None if timeout is None else time.monotonic() + timeout
    waited = False

    def wait():
        nonlocal waited
        if not waited:
            waited = True
            if on_wait:
                on_wait()

    if free is free_gib and (total := total_gib()) is not None and need_gib + MARGIN_GIB > total - DESKTOP_GIB:
        raise ValueError(f"need {need_gib:g} GiB can never be free ({total:.1f} GiB total, about {DESKTOP_GIB} GiB "
                         f"always in use by the desktop): declare at most {int(total - DESKTOP_GIB - MARGIN_GIB)} GiB")
    mine = declare(need_gib, who, lock, priority, label)
    file = open(path, "a")
    try:
        while True:
            while not first_in_line(lock):
                if deadline is not None and time.monotonic() >= deadline:
                    raise TimeoutError("timed out waiting for higher-priority GPU jobs")
                wait()
                time.sleep(POLL)
            try:
                fcntl.flock(file, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                wait()
                _flock(file, deadline)
            # flock wakes its waiters in no particular order: a higher-priority job that declared while this one
            # was blocked in flock still goes first, so let go and line up again behind it.
            if first_in_line(lock):
                break
            fcntl.flock(file, fcntl.LOCK_UN)
        while (available := free()) < need_gib + MARGIN_GIB:
            if deadline is not None and time.monotonic() >= deadline:
                raise TimeoutError(f"timed out waiting for {need_gib + MARGIN_GIB:.1f} GiB free VRAM "
                                   f"({available:.1f} GiB free)")
            wait()
            time.sleep(POLL)
    except BaseException:
        file.close()
        raise
    finally:
        mine.unlink(missing_ok=True)
    return file


def release(file):
    fcntl.flock(file, fcntl.LOCK_UN)
    file.close()


@contextlib.contextmanager
def gpu_slot(need_gib, who, lock=None, timeout=None, on_wait=None, priority=DEFAULT_PRIORITY, label=None):
    """Hold the GPU slot for the block; see acquire()."""
    file = acquire(need_gib, who, lock, timeout, on_wait, priority=priority, label=label)
    try:
        yield
    finally:
        release(file)


def main(argv=None):
    parser = argparse.ArgumentParser(prog="python3 -m hayate_tools.gpu_slot",
                                     description="Run a command in the GPU slot (replaces `flock gpu.lock cmd`).")
    parser.add_argument("--need", type=float, required=True, help="free VRAM the command needs, GiB")
    parser.add_argument("--who", required=True, help="shown to serve and Studio, e.g. 'hayate-train krea2'")
    parser.add_argument("--priority", type=int, default=DEFAULT_PRIORITY, help="lower goes first (0 most urgent)")
    parser.add_argument("--label", help="short text for Studio's queue display")
    parser.add_argument("--lock", help="lock file (default $HAYATE_GPU_LOCK or %s)" % DEFAULT_LOCK)
    parser.add_argument("--timeout", type=float, help="give up after this many seconds (exit 124)")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("missing command after --")

    def on_wait():
        print(f"gpu_slot: {args.who} waiting for the GPU slot ({args.need:g} GiB)", file=sys.stderr, flush=True)

    try:
        file = acquire(args.need, args.who, args.lock, args.timeout, on_wait, priority=args.priority, label=args.label)
    except TimeoutError as error:
        print(f"gpu_slot: {error}", file=sys.stderr)
        return 124
    except ValueError as error:
        print(f"gpu_slot: {error}", file=sys.stderr)
        return 2
    try:
        code = subprocess.call(command, pass_fds=(file.fileno(),))
    finally:
        release(file)
    return code if code >= 0 else 128 - code


if __name__ == "__main__":
    sys.exit(main())
