"""CPU checks for hayate_tools.gpu_slot with real flocks in a temporary directory; free VRAM is a stand-in."""
import fcntl
import json
import os
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

from hayate_tools import gpu_slot


def plenty():
    return 1000.0


class Slot(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.lock = Path(self.directory.name) / "gpu.lock"
        self.decl = Path(self.directory.name) / "gpu.d"
        patch = mock.patch.dict(os.environ, {"HAYATE_GPU_LOCK": str(self.lock)})
        patch.start()
        self.addCleanup(patch.stop)

    def held_elsewhere(self):
        with open(self.lock, "a") as other:
            try:
                fcntl.flock(other, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                return True
            return False

    def hold(self):
        holder = open(self.lock, "a")
        fcntl.flock(holder, fcntl.LOCK_EX)
        self.addCleanup(holder.close)
        return holder

    def test_paths_follow_the_environment_and_argument(self):
        self.assertEqual(gpu_slot.lock_path(), self.lock)
        self.assertEqual(gpu_slot.declaration_dir(), self.decl)
        self.assertEqual(gpu_slot.declaration_dir("/x/y.lock"), Path("/x/gpu.d"))

    def test_declaration_is_atomic_listed_while_waiting_and_removed_once_held(self):
        holder = self.hold()
        waits, seen = [], []
        thread = threading.Thread(target=lambda: seen.append(
            gpu_slot.acquire(30, "hayate-train krea2", on_wait=lambda: waits.append(1), free=plenty)))
        thread.start()
        deadline = time.time() + 5
        while not gpu_slot.declarations() and time.time() < deadline:
            time.sleep(0.01)
        (d,) = gpu_slot.declarations()
        self.assertEqual((d["pid"], d["who"], d["need_gib"]), (os.getpid(), "hayate-train krea2", 30))
        self.assertLessEqual(abs(d["since"] - time.time()), 5)
        self.assertEqual(sorted(p.name for p in self.decl.iterdir()), [f"{os.getpid()}.json"])
        self.assertEqual(waits, [1])
        thread.join(0.2)
        self.assertTrue(thread.is_alive())
        fcntl.flock(holder, fcntl.LOCK_UN)
        thread.join(5)
        self.assertEqual(gpu_slot.declarations(), [])
        self.assertEqual(list(self.decl.iterdir()), [])
        self.assertTrue(self.held_elsewhere())
        gpu_slot.release(seen[0])
        self.assertFalse(self.held_elsewhere())

    def test_dead_pids_broken_json_and_temporary_files_are_ignored(self):
        self.decl.mkdir()
        dead = subprocess.Popen(["true"])
        dead.wait()
        (self.decl / f"{dead.pid}.json").write_text(json.dumps(dict(pid=dead.pid, who="x", need_gib=1, since=1)))
        (self.decl / "2.json").write_text('{"pid": 1, "who"')
        (self.decl / ".1.json.tmp").write_text(json.dumps(dict(pid=1, who="tmp", need_gib=1, since=1)))
        now = int(time.time())
        (self.decl / "1.json").write_text(json.dumps(dict(pid=1, who="init", need_gib=2.5, since=now)))
        (self.decl / "3.json").write_text(json.dumps(dict(pid=1, who="before boot", need_gib=1, since=3)))
        self.assertEqual(gpu_slot.declarations(),
                         [dict(pid=1, who="init", need_gib=2.5, since=now, priority=5, label=None)])

    def test_higher_priority_declaration_goes_first(self):
        # pid 1 is always alive: it stands in for another waiting process.
        self.decl.mkdir()
        other = self.decl / "1.json"
        other.write_text(json.dumps(dict(pid=1, who="urgent", need_gib=1, since=int(time.time()) + 60, priority=0)))
        seen, waits = [], []
        with mock.patch.object(gpu_slot, "POLL", 0.01):
            thread = threading.Thread(target=lambda: seen.append(gpu_slot.acquire(
                3, "background", on_wait=lambda: waits.append(1), free=plenty, priority=6, label="GLM quant")))
            thread.start()
            thread.join(0.3)
            self.assertTrue(thread.is_alive())
            self.assertFalse(self.held_elsewhere())
            (mine,) = [d for d in gpu_slot.declarations() if d["pid"] == os.getpid()]
            self.assertEqual((mine["priority"], mine["label"]), (6, "GLM quant"))
            self.assertEqual([d["who"] for d in gpu_slot.declarations()], ["urgent", "background"])
            other.unlink()
            thread.join(5)
        self.assertEqual(waits, [1])
        self.assertTrue(self.held_elsewhere())
        gpu_slot.release(seen[0])

    def test_higher_priority_arriving_while_blocked_in_flock_goes_first(self):
        holder = self.hold()
        seen = []
        with mock.patch.object(gpu_slot, "POLL", 0.01):
            thread = threading.Thread(target=lambda: seen.append(
                gpu_slot.acquire(3, "background", free=plenty, priority=6)))
            thread.start()
            deadline = time.time() + 5
            while not any(d["pid"] == os.getpid() for d in gpu_slot.declarations()):
                self.assertLess(time.time(), deadline)
                time.sleep(0.01)
            thread.join(0.1)  # first in line, so now blocked in flock behind the holder
            # pid 1 is always alive: it stands in for a priority-1 job declaring while this one is blocked.
            urgent = self.decl / "1.json"
            urgent.write_text(json.dumps(dict(pid=1, who="urgent", need_gib=1, since=int(time.time()), priority=1)))
            fcntl.flock(holder, fcntl.LOCK_UN)
            thread.join(0.3)
            self.assertTrue(thread.is_alive())
            self.assertFalse(self.held_elsewhere())  # let go for the urgent job
            self.assertEqual([d["who"] for d in gpu_slot.declarations()], ["urgent", "background"])
            urgent.unlink()
            thread.join(5)
        self.assertTrue(self.held_elsewhere())
        gpu_slot.release(seen[0])

    def test_aging_lifts_old_background_jobs_but_never_past_user_requests(self):
        now = 10_000_000
        d = lambda priority, waited: dict(priority=priority, since=now - waited, pid=1)
        self.assertEqual(gpu_slot.effective_priority(d(7, 0), now), 7)
        self.assertEqual(gpu_slot.effective_priority(d(7, 29 * 60), now), 7)
        self.assertEqual(gpu_slot.effective_priority(d(7, 30 * 60), now), 6)
        self.assertEqual(gpu_slot.effective_priority(d(7, 3 * 3600), now), 2)
        self.assertEqual(gpu_slot.effective_priority(d(7, 100 * 3600), now), 2)
        self.assertEqual(gpu_slot.effective_priority(d(1, 100 * 3600), now), 1)
        self.assertEqual(gpu_slot.effective_priority(d(0, 0), now), 0)
        # An old priority-7 job goes before a fresh priority-6 one; a user job before both.
        self.decl.mkdir()
        me = os.getpid()
        for name, priority, since in (("old7", 7, int(time.time()) - 3600), ("new6", 6, int(time.time())),
                                      ("user", 1, int(time.time()))):
            (self.decl / f"{name}.json").write_text(json.dumps(dict(pid=me, who=name, need_gib=1, since=since,
                                                                    priority=priority)))
        self.assertEqual([x["who"] for x in gpu_slot.declarations()], ["user", "old7", "new6"])

    def test_lower_priority_declaration_does_not_block(self):
        self.decl.mkdir()
        (self.decl / "1.json").write_text(json.dumps(dict(pid=1, who="later", need_gib=1, since=int(time.time()), priority=9)))
        with mock.patch.object(gpu_slot, "free_gib", plenty):
            with gpu_slot.gpu_slot(3, "now", priority=1, timeout=2):
                self.assertTrue(self.held_elsewhere())

    def test_no_wait_without_contention(self):
        waits = []
        with mock.patch.object(gpu_slot, "free_gib", plenty):
            with gpu_slot.gpu_slot(3, "t", on_wait=lambda: waits.append(1)):
                self.assertTrue(self.held_elsewhere())
        self.assertFalse(self.held_elsewhere())
        self.assertEqual(waits, [])

    def test_waits_for_free_vram_under_the_lock(self):
        readings = iter([1.0, 2.0, 3.9, 4.0])
        waits = []
        with mock.patch.object(gpu_slot, "POLL", 0.01):
            file = gpu_slot.acquire(3, "t", on_wait=lambda: waits.append(1), free=lambda: next(readings))
        self.assertEqual(next(readings, None), None)  # 4.0 = 3 + the 1 GiB margin was the first enough
        self.assertEqual(waits, [1])
        self.assertTrue(self.held_elsewhere())
        gpu_slot.release(file)

    def test_vram_timeout_releases_the_lock_and_withdraws(self):
        with mock.patch.object(gpu_slot, "POLL", 0.01), self.assertRaises(TimeoutError):
            gpu_slot.acquire(3, "t", timeout=0.1, free=lambda: 0.0)
        self.assertFalse(self.held_elsewhere())
        self.assertEqual(gpu_slot.declarations(), [])

    def test_lock_timeout_withdraws_and_a_late_flock_does_not_keep_the_lock(self):
        holder = self.hold()
        start = time.time()
        with self.assertRaises(TimeoutError):
            gpu_slot.acquire(3, "t", timeout=0.2, free=plenty)
        self.assertGreater(time.time() - start, 0.15)
        self.assertEqual(gpu_slot.declarations(), [])
        fcntl.flock(holder, fcntl.LOCK_UN)
        deadline = time.time() + 5
        while self.held_elsewhere():  # the abandoned flock completes and lets go
            self.assertLess(time.time(), deadline)
            time.sleep(0.01)

    def test_cli_runs_the_command_holding_the_lock_and_passes_its_status(self):
        probe = ("import fcntl, sys\nf = open(sys.argv[1], 'a')\n"
                 "try:\n    fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)\nexcept BlockingIOError:\n    sys.exit(3)\n")
        with mock.patch.object(gpu_slot, "free_gib", plenty):
            self.assertEqual(gpu_slot.main(["--need", "1", "--who", "t", "--", sys.executable, "-c", probe,
                                            str(self.lock)]), 3)  # held while the command runs
            self.assertEqual(gpu_slot.main(["--need", "1", "--who", "t", "--", "true"]), 0)
            self.assertEqual(gpu_slot.main(["--need", "1", "--who", "t", "--", "sh", "-c", "kill -TERM $$"]), 143)
        self.assertFalse(self.held_elsewhere())
        self.hold()
        with mock.patch.object(gpu_slot, "free_gib", plenty):
            self.assertEqual(gpu_slot.main(["--need", "1", "--who", "t", "--timeout", "0.1", "--", "true"]), 124)

    def test_cli_process_exit_status_with_nvidia_smi(self):
        bin_dir = Path(self.directory.name) / "bin"
        bin_dir.mkdir()
        smi = bin_dir / "nvidia-smi"
        smi.write_text("#!/bin/sh\necho 81920\n")  # MiB free
        smi.chmod(0o755)
        env = dict(os.environ, PATH=f"{bin_dir}:{os.environ['PATH']}")
        result = subprocess.run([sys.executable, "-m", "hayate_tools.gpu_slot", "--need", "30", "--who", "t", "--",
                                 "sh", "-c", "exit 7"], env=env, cwd=Path(__file__).parent.parent,
                                capture_output=True, text=True, timeout=60)
        self.assertEqual(result.returncode, 7, result.stderr)
        self.assertEqual(list(self.decl.iterdir()), [])

if __name__ == "__main__":
    unittest.main()
