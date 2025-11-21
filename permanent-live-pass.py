#!/usr/bin/env python3
import os
import sys
import time
import signal
import threading
import subprocess
import random
import re
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict

# --------- Lines ----------
LINES = [
	"bakerloo",
	"jubilee", "victoria", "piccadilly", "elizabeth",
	"liberty", "lioness", "mildmay", "suffragette", "weaver", "windrush",
]

# --------- Paths / env ----------
LIVE_PASS_ENTRY = os.getenv("LIVE_PASS_ENTRY", "live-pass.py")
LOG_DIR = Path("live-pass-logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)

# Runtime tweaks
BASE_THROTTLE_S = float(os.getenv("THROTTLE_S", "0.2"))
BASE_POLL_SEC   = float(os.getenv("POLL_SEC", "3"))

# --------- Startup pacing ----------
# Used to not overload API
LAUNCH_WAVE_SIZE  = int(os.getenv("LAUNCH_WAVE_SIZE", "3"))     # lines per wave
LAUNCH_WAVE_GAP_S = float(os.getenv("LAUNCH_WAVE_GAP_S", "20")) # seconds between waves

# Per-child jitter within a wave
LAUNCH_JITTER_MIN_S = float(os.getenv("LAUNCH_JITTER_MIN_S", "1.0"))
LAUNCH_JITTER_MAX_S = float(os.getenv("LAUNCH_JITTER_MAX_S", "4.0"))

# Randomization ranges for child env to desynchronize their calls
THROTTLE_JITTER_ADD_MAX = float(os.getenv("THROTTLE_JITTER_ADD_MAX", "0.2"))
POLL_JITTER_ADD_MAX     = float(os.getenv("POLL_JITTER_ADD_MAX", "1")) 

# --------- Timings ----------
READINESS_CHECK_PERIOD_SEC = 5 * 60

# --------- 429 detection helpers ----------
# Ignore "429" when it appears as part of arrival prints
IGNORE_429_IN_ARRIVALS = re.compile(r"(?:\btts\s*=\s*\d+\b|\btimeToStation\b)", re.IGNORECASE)
# Only treat as rate limit if it follows rate limit format
HTTP_429_PATTERN = re.compile(
	r"""
	(?:\bhttp[^\n]*\b)?                   # optional 'http' wording
	\b(?:status(?:\s*code)?|response)\s*  # 'status' / 'status code' / 'response'
	[:=]?\s*429\b                         # followed by 429
	| \b429\b\s*[-–]?\s*too\s+many\s+requests\b
	| \btoo\s+many\s+requests\b
	""",
	re.IGNORECASE | re.VERBOSE
)

def now_hms() -> str:
	return datetime.now().strftime("%H:%M:%S")


def child_env_for(line: str) -> Dict[str, str]:
	"""Create per-child env with randomized desyncs"""
	env = os.environ.copy()
	env["LINE_KEY"] = line

	# Randomize throttle a bit so their schedules drift
	throttle = max(0.0, BASE_THROTTLE_S + random.random() * THROTTLE_JITTER_ADD_MAX)
	poll     = max(1.0, BASE_POLL_SEC   + random.random() * POLL_JITTER_ADD_MAX)

	env["THROTTLE_S"] = f"{throttle:.3f}"
	env["POLL_SEC"]   = f"{poll:.3f}"
	return env


class LineProcess:
	"""
	Wraps a single live-pass subprocess for a line.
	- Streams stdout into live-pass-logs/{line}_live-pass.log
	- Sets ready_event once it sees "[loop] polling arrivals"
	- Detects HTTP 429 / 'Too Many Requests' in child output
	"""
	def __init__(self, line: str):
		self.line = line
		self.proc: Optional[subprocess.Popen] = None
		self.stdout_thread: Optional[threading.Thread] = None
		self.ready_event = threading.Event()
		self.log_path = LOG_DIR / f"{self.line}_live-pass.log"
		self._alive = False
		self._saw_429 = False
		self._err_line: Optional[str] = None

	def _spawn(self):
		env = child_env_for(self.line)
		self.proc = subprocess.Popen(
			[sys.executable, "-u", LIVE_PASS_ENTRY],
			stdout=subprocess.PIPE,
			stderr=subprocess.STDOUT,
			bufsize=1,
			universal_newlines=True,
			env=env,
		)
		self._alive = True
		self._saw_429 = False
		self._err_line = None
		self.ready_event.clear()
		self.stdout_thread = threading.Thread(target=self._pump_stdout, daemon=True)
		self.stdout_thread.start()

	def _line_is_real_429(self, text: str) -> bool:
		"""
		Return True if the line indicates an HTTP 429 / Too Many Requests
		"""
		if IGNORE_429_IN_ARRIVALS.search(text):
			return False

		low = text.lower()

		looks_like_net = any(tag in low for tag in ("[net]", "[error]", "[err]", "http", "request", "response", "status"))
		if not looks_like_net:
			return False

		if HTTP_429_PATTERN.search(text):
			return True

		if re.search(r"\b429\b", text) and any(kw in low for kw in ("status", "too many requests", "rate limit")):
			return True

		return False

	def _pump_stdout(self):
		assert self.proc and self.proc.stdout
		with self.log_path.open("a", encoding="utf-8") as logf:
			for line in self.proc.stdout:
				if line is None:
					break
				# Only file logging (keep terminal clean)
				logf.write(line)
				logf.flush()

				if "[loop] polling arrivals" in line:
					self.ready_event.set()

				if self._line_is_real_429(line):
					self._saw_429 = True
					self._err_line = line.strip()
		self._alive = False

	def start(self):
		print(f"[{now_hms()}] [launch] starting live-pass for line='{self.line}' …")
		self._spawn()

	def terminate(self, grace: float = 8.0):
		if self.proc and self.proc.poll() is None:
			try:
				self.proc.terminate()
			except Exception:
				pass
			t0 = time.time()
			while (time.time() - t0) < grace and self.proc.poll() is None:
				time.sleep(0.1)
			if self.proc.poll() is None:
				try:
					self.proc.kill()
				except Exception:
					pass
		self._alive = False

	def join(self):
		if self.stdout_thread:
			self.stdout_thread.join()

	def is_alive(self) -> bool:
		return self.proc is not None and self.proc.poll() is None

	def saw_429(self) -> bool:
		return self._saw_429

	def last_error_line(self) -> Optional[str]:
		return self._err_line


def main():
	stop_all = threading.Event()

	def handle_sigint(signum, frame):
		if not stop_all.is_set():
			print(f"\n[{now_hms()}] [stop] interrupt received, shutting down children …")
			stop_all.set()
		else:
			os._exit(1)

	signal.signal(signal.SIGINT, handle_sigint)
	if hasattr(signal, "SIGTERM"):
		signal.signal(signal.SIGTERM, handle_sigint)

	# Launch in waves with jitter
	procs: list[LineProcess] = []
	total = len(LINES)
	idx = 0
	while idx < total:
		wave = LINES[idx: idx + LAUNCH_WAVE_SIZE]
		for i, line in enumerate(wave):
			jitter = random.uniform(LAUNCH_JITTER_MIN_S, LAUNCH_JITTER_MAX_S)
			time.sleep(jitter)
			lp = LineProcess(line)
			lp.start()
			procs.append(lp)
		idx += LAUNCH_WAVE_SIZE
		if idx < total:
			time.sleep(LAUNCH_WAVE_GAP_S)

	# Watchdog loop:
	#    - If any process reports 429, print error and terminate all.
	try:
		while not stop_all.is_set():
			for _ in range(READINESS_CHECK_PERIOD_SEC):
				if stop_all.is_set():
					break
				for lp in procs:
					if lp.saw_429():
						msg = lp.last_error_line() or "status 429"
						print(f"[{now_hms()}] [error] line='{lp.line}' reported 429 — terminating all …")
						print(f"[detail] {msg}")
						stop_all.set()
						break
				if stop_all.is_set():
					break
				time.sleep(1)
			if stop_all.is_set():
				break

	finally:
		for lp in procs:
			lp.terminate()
		for lp in procs:
			lp.join()
		print(f"[{now_hms()}] [done] all children stopped.")


if __name__ == "__main__":
	main()
