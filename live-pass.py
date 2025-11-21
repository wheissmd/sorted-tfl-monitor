#!/usr/bin/env python3
import os
import re
import time
import yaml
import json
import traceback
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Tuple, Any, Set, DefaultDict
from collections import defaultdict
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# ===================== RUNTIME PARAMS =====================
LINE_KEY        = os.getenv("LINE_KEY", "weaver").strip() 
PROJECT_DIR     = os.getenv("PROJECT_DIR", os.path.dirname(__file__)).strip()

CACHED_STOPS_DIR = os.path.join(PROJECT_DIR, "cached-stops")
POLL_SEC        = float(os.getenv("POLL_SEC", "3.0"))
HTTP_TIMEOUT_S  = int(os.getenv("HTTP_TIMEOUT_S", "15"))
THROTTLE_S      = float(os.getenv("THROTTLE_S", "0.0"))
BASE            = "https://api.tfl.gov.uk"
TFL_APP_KEY     = os.getenv("TFL_APP_KEY", "").strip()
DEBUG           = int(os.getenv("DEBUG", "1")) 

# Through-linking window
MAX_LINK_MIN        = int(os.getenv("MAX_LINK_MIN", "35"))

# Terminus behavior
SOON_THRESHOLD_SEC  = int(os.getenv("SOON_THRESHOLD_SEC", "120"))  # appear/confirm window at B or neighbor
ATTEMPT_WINDOW_MIN  = int(os.getenv("ATTEMPT_WINDOW_MIN", "5"))    # min gone before confirm depart
UNK_CONFIRM_MIN     = int(os.getenv("UNK_CONFIRM_MIN", "12"))      # up to 10 min to see the train at next station

# DUE criteria (used only by the "impossible" path)
DUE_TTS_SEC         = int(os.getenv("DUE_TTS_SEC", "90"))
DUE_ETA_SEC         = int(os.getenv("DUE_ETA_SEC", "90"))

# "B-terminus-through" path epsilon for etaB >= etaA + epsilon (seconds)
BTERM_EPSILON_SEC   = int(os.getenv("BTERM_EPSILON_SEC", "30"))

RUNS_DIR = os.path.join(PROJECT_DIR, "real-runs")

# Debug printing
def _dbg(msg: str):
	if DEBUG:
		print(msg)

# ===================== MOVE ONE TERMINI =====================
MOVE_ONE_TERMINI: Set[str] = {
	"Highbury & Islington",
	"Enfield Town"
}

# ===================== VARIABLE_RUN TERMINI =====================
VARIABLE_RUN_TERMINI: Set[str] = {
	"Liverpool Street",
	"Reading",
	"Maidenhead",
}

# ===================== SHUTTLE SERVICE =====================
SHUTTLE_SERVICE: Set[str] = {
	"waterloo",
}

# ===================== IMPOSSIBLE TERMINI =====================
IMPOSSIBLE_TERMINI: Set[str] = {
	"Heathrow Terminal 4 Underground Station",
	"Heathrow Terminal 5 Underground Station",
	"Mill Hill East Underground Station",
	"New Cross Rail Station",
	"Crystal Palace Rail Station",
	"Chesham Underground Station",
	"Amersham Underground Station",
	"Stratford International", # Workaround for DLR network complexity
}

# ===================== HELPERS =====================
SUFFIXES = (
	" (Elizabeth line)",
	" Rail Station",
	" Underground Station",
	" DLR Station",
	" Station",
)

UNKNOWN_DEST_LABELS = {
	"check front of train",
	"check front of the train",
	"unknown",
}

def strip_suffixes(name: str) -> str:
	"""
	For Elizabeth line: removes suffixes until name doesn't end with suffix
	For other modes:    removes one suffix
	Elizabeth Line special case is required due to buggy logic further, but it seems to work.
	"""
	s = (name or "").strip()
	if LINE_KEY == "elizabeth":
		changed = True
		while changed:
			changed = False
			for suf in SUFFIXES:
				if s.endswith(suf):
					s = s[: -len(suf)]
					changed = True
					break
	else:
		for suf in SUFFIXES:
			if s.endswith(suf):
				s = s[: -len(suf)]
				break
	return s.strip()

def expand_with_suffixes(base_names: list[str], station_suffix: str) -> list[str]:
	"""
	For Elizabeth line: return [name + ' (Elizabeth line)', name + ' Rail Station', name]
	For other modes: return [name + station_suffix, name]
	Removes duplicates but keeps general order.
	"""
	seen = set()
	out = []
	for nm in base_names or []:
		base = strip_suffixes(nm)
		candidates = (
			[f"{base} (Elizabeth line)", f"{base} Rail Station", base]
			if station_suffix == " (Elizabeth line)"
			else [f"{base}{station_suffix}", base]
		)
		for c in candidates:
			if c not in seen:
				seen.add(c)
				out.append(c)
	return out

def canon(s: str) -> str:
	base = strip_suffixes(s or "").strip()
	# Drops general case unusual directions i. e. "Richmond (London)"
	base = re.sub(r"\s*\([^)]*\)\s*$", "", base).strip()
	# List to correct common spelling variants
	ALIASES = {
		"Canon's Park": "Canons Park",
		"St Johns Wood": "St John's Wood",
		"Kings Cross St Pancras": "King's Cross St Pancras",
		"Regents Park": "Regent's Park",
		"Green park": "Green Park",
		"Wood green": "Wood Green",
		"West ham": "West Ham",
		"London Euston": "Euston",
		"London Liverpool Street": "Liverpool Street",
		"Heathrow Airport Terminal 4": "Heathrow Terminal 4",
		"Heathrow Airport Terminal 5": "Heathrow Terminal 5",
		"New Cross ELL": "New Cross"
	}
	for old, new in ALIASES.items():
		if base.lower() == old.lower():
			base = new
			break
	return re.sub(r"[^a-z0-9]", "", base.lower())

def parse_dt(s: str | None) -> datetime | None:
	if not s: return None
	try: return datetime.fromisoformat(s.replace("Z","+00:00"))
	except Exception: return None

def hhmmss_local(dt: datetime | None) -> str:
	if not isinstance(dt, datetime): return "-"
	return (dt.astimezone()).strftime("%H:%M:%S")

def hhmm_local(dt: datetime) -> str:
	#Round to the nearest minute
	return (dt.astimezone() + timedelta(seconds=30)).strftime("%H:%M")

def window_from_departure(dt_local: datetime) -> Tuple[datetime, datetime, str]:
	"""
	Sets the window of the decided departure, returns Tuple with start and end of the window and hour window
	"""
	start = dt_local.replace(minute=0, second=0, microsecond=0)
	end   = start + timedelta(hours=1)
	hour_key = f"{start.strftime('%H')}-{end.strftime('%H')}"
	return start, end, hour_key

def headways_from_times(times: List[str], start_hhmm: str, end_hhmm: str) -> List[int]:
	"""
	Returns list of headways based on raw departure times
	"""
	if not times: return []
	start_hm = f"{start_hhmm[:2]}:{start_hhmm[2:]}"
	end_hm   = f"{end_hhmm[:2]}:{end_hhmm[2:]}"
	prev = None; within = []
	for hm in sorted(times):
		if hm < start_hm:
			prev = hm
		elif start_hm <= hm < end_hm:
			within.append(hm)
	if not within: return []
	anchor = datetime(2000,1,1)
	def to_dt(hm: str) -> datetime:
		return anchor.replace(hour=int(hm[:2]), minute=int(hm[3:]), second=0)
	gaps = []; last_dt = to_dt(prev) if prev else None
	for hm in within:
		cur = to_dt(hm)
		if last_dt is not None:
			gaps.append(int(round((cur - last_dt).total_seconds()/60)))
		last_dt = cur
	return gaps

# ===================== HTTP =====================
def make_session():
	sess = requests.Session()
	retry = Retry(
		total=5, connect=3, read=3,
		backoff_factor=0.5,
		status_forcelist=(429, 500, 502, 503, 504),
		allowed_methods=frozenset(["GET", "HEAD"]),
		raise_on_status=False,
	)
	adapter = HTTPAdapter(max_retries=retry, pool_connections=40, pool_maxsize=40, pool_block=True)
	sess.mount("https://", adapter); sess.mount("http://", adapter)
	sess.headers.update({"Accept":"application/json"})
	return sess

# ===================== YAML LOADERS =====================
def load_yaml(path: str) -> dict:
	with open(path, "r", encoding="utf-8") as f:
		return yaml.safe_load(f) or {}

def load_line_config(line_key: str):
	cfg_path = os.path.join(PROJECT_DIR, "line_config", f"{line_key}.yaml")
	if not os.path.exists(cfg_path):
		raise SystemExit(f"Config not found: {cfg_path}")
	with open(cfg_path, "r", encoding="utf-8") as f:
		cfg = yaml.safe_load(f)

	line = cfg.get("line", {})
	sections_in = cfg.get("sections", {})

	# Normalize sections to {id: {"points": {...}}}
	sections = {}
	if isinstance(sections_in, list):
		sections = {sec["id"]: {"points": sec["points"]} for sec in sections_in}
	else:
		for sid, sec in sections_in.items():
			if isinstance(sec, dict):
				if "points" in sec:
					pts = sec["points"]
					other = {k:v for k,v in sec.items() if k!="points"}
					sections[sid] = {"points": pts, **other}
				else:
					sections[sid] = {"points": sec}
			else:
				sections[sid] = {"points": sec}
	return line, sections

def load_routes(line_key: str):
	routes_path = os.path.join(PROJECT_DIR, "line_routes", f"{line_key}.yaml")
	if not os.path.exists(routes_path):
		raise SystemExit(f"Routes not found: {routes_path}")
	routes = load_yaml(routes_path)

	# Collect weights across base and any branch dicts
	stations_weights = routes.get("stations") or {}
	for k, v in routes.items():
		if isinstance(k, str) and k.startswith("stations_branch") and isinstance(v, dict):
			stations_weights.update(v)

	weight_by_canon = {canon(k): int(v) for k, v in stations_weights.items()}
	launch_dir_default = routes.get("launch_dir_default") or {}
	min_run_map = routes.get("min_run_ab_default_min") or {}

	# Best-effort value to display (no suffix)
	value_to_display: Dict[int, List[str]] = defaultdict(list)
	for name, val in stations_weights.items():
		try:
			value_to_display[int(val)].append(strip_suffixes(str(name)))
		except Exception:
			pass

	return weight_by_canon, launch_dir_default, min_run_map, value_to_display

# ===================== TfL StopPoint Resolver =====================
class TfLSess:
	def __init__(self, app_key: str, http_timeout: int = 12, throttle_sec: float = 0.0,
				 default_modes: List[str] | None = None, preferred_suffix: str = " Underground Station"):
		self.sess = make_session()
		self.key = app_key
		self.connect_timeout = min(5, max(2, int(http_timeout // 3)))
		self.read_timeout    = max(20, int(http_timeout * 2))
		self.throttle_sec    = throttle_sec
		self._search_cache_one   = {}
		self._search_cache_all   = {}
		self.default_modes = default_modes or ["tube"]
		self.preferred_suffix = preferred_suffix

	def _sleep(self):
		if self.throttle_sec > 0:
			time.sleep(self.throttle_sec)

	def search_stop_id_one(self, name: str, modes: List[str] | None = None) -> str:
		if name in self._search_cache_one:
			return self._search_cache_one[name]

		bp = strip_suffixes(name)
		variants = [name, bp]
		modes_param = modes if modes is not None else self.default_modes

		def try_query(qname: str, modes_use):
			url = f"{BASE}/StopPoint/Search/{requests.utils.quote(qname)}"
			params = {"maxResults": 24}
			if self.key: params["app_key"] = self.key
			if modes_use: params["modes"] = ",".join(modes_use)
			r = self.sess.get(url, params=params,
							  timeout=(self.connect_timeout, self.read_timeout))
			if r.status_code == 404: return []
			r.raise_for_status()
			return (r.json() or {}).get("matches") or []

		matches = []
		for v in variants:
			matches = try_query(v, modes_param)
			if matches: break
		if not matches:
			matches = try_query(bp, None)
		if not matches:
			raise RuntimeError(f"Stop not found for name: {name}")

		def score(m):
			nm = (m.get("name") or "")
			s = 0
			if nm == name: s += 1000
			if strip_suffixes(nm).lower() == bp.lower(): s += 120
			if name.lower() in nm.lower(): s += 80
			if self.preferred_suffix and nm.endswith(self.preferred_suffix):
				s += 15
			if nm.endswith(" Underground Station"):
				s += 10
			return s

		best = max(matches, key=score)
		sid = best.get("id")
		if not sid: raise RuntimeError(f"No StopPoint id for {name}")
		self._search_cache_one[name] = sid
		self._sleep()
		_dbg(f"[resolve] {name} -> {sid} ({best.get('name')})")
		return sid

	def search_stop_ids_all(self, name: str, modes: List[str] | None = None) -> List[str]:
		if name in self._search_cache_all:
			return list(self._search_cache_all[name])

		parent_id = self.search_stop_id_one(name, modes=modes)
		ids: Set[str] = set([parent_id])

		def fetch_children(url_suffix: str):
			url = f"{BASE}/StopPoint/{requests.utils.quote(parent_id)}/{url_suffix}"
			params = {}
			if self.key: params["app_key"] = self.key
			try:
				r = self.sess.get(url, params=params, timeout=(self.connect_timeout, self.read_timeout))
				if r.status_code == 404:
					return []
				r.raise_for_status()
				return r.json() or []
			except Exception:
				return []

		for child in fetch_children("Children"):
			cid = child.get("id")
			if cid: ids.add(cid)

		for child in fetch_children("StopPoints"):
			cid = (child.get("id") or "")
			if cid: ids.add(cid)

		try:
			url = f"{BASE}/StopPoint/{requests.utils.quote(parent_id)}"
			params = {}
			if self.key: params["app_key"] = self.key
			r = self.sess.get(url, params=params, timeout=(self.connect_timeout, self.read_timeout))
			if r.status_code != 404:
				j = r.json() or {}
				for child in (j.get("children") or []):
					cid = child.get("id")
					if cid: ids.add(cid)
		except Exception:
			pass

		if not ids:
			ids = {parent_id}

		self._search_cache_all[name] = set(ids)
		self._sleep()
		_dbg(f"[resolve-all] {name} -> {len(ids)} stopIds")
		return list(ids)

# ===================== PASS/APPEAR tracker =====================
class PassTracker:
	def __init__(self, nid_to_name: Dict[str,str] | None = None):
		self.nid_to_name = nid_to_name or {}
		self.prev_keys: Set[Tuple[str,str,str]] = set()
		self.prev_rows: Dict[Tuple[str,str,str], Dict[str,Any]] = {}
		self._linger: Dict[Tuple[str,str,str], Dict[str,Any]] = {}
		self._linger_grace = timedelta(seconds=15)

	def _now_local(self) -> datetime:
		return datetime.now(timezone.utc).astimezone()

	def _last_expected(self, row: Dict[str,Any]) -> datetime | None:
		dt = parse_dt(row.get("expectedArrival") or "")
		return dt.astimezone() if dt else None

	def update(self, rows: List[Dict[str,Any]]) -> List[Dict[str,Any]]:
		now_local = self._now_local()

		# Keep best (lowest tts) per (vehicle, naptan, dir)
		best: Dict[Tuple[str,str,str], Dict[str,Any]] = {}
		curr_keys: Set[Tuple[str,str,str]] = set()
		for r in rows:
			vid = str(r.get("vehicleId") or "")
			nid = str(r.get("naptanId") or "")
			d   = str(r.get("platformName") or r.get("direction") or "")
			if not (vid and nid and d):
				continue
			k = (vid, nid, d)
			prev = best.get(k)
			if prev is None or int(r.get("timeToStation") or 1e9) < int(prev.get("timeToStation") or 1e9):
				best[k] = r
			curr_keys.add(k)

		events: List[Dict[str,Any]] = []

		# Disappearances -> PASS
		disappeared = self.prev_keys - curr_keys
		for k in disappeared:
			last = self.prev_rows.get(k) or {}
			station = last.get("stationName") or self.nid_to_name.get(k[1], "")
			if not station:
				continue
			exp = self._last_expected(last)
			if not exp:
				continue
			if exp <= now_local + self._linger_grace:
				# Remember destination at the station with PASS
				dest_raw_at_station = (last.get("destinationName") or last.get("towards") or "")
				_dbg(f"[PASS/disappear] vid={k[0]} st={station} dir={k[2]} exp={hhmmss_local(exp)} dest='{dest_raw_at_station}'")
				events.append({
					"type": "PASS", "vehicleId": k[0], "naptanId": k[1],
					"platformDir": k[2], "stationName": station, "t_local": exp,
					"dest_raw": dest_raw_at_station,
				})
				self._linger.pop(k, None)
			else:
				self._linger[k] = last

		# Appearances -> APPEAR
		appeared = curr_keys - self.prev_keys
		for k in appeared:
			self._linger.pop(k, None)
			cur = best.get(k) or {}
			station = cur.get("stationName") or self.nid_to_name.get(k[1], "")
			if station:
				exp_local = self._last_expected(cur)
				_dbg(f"[APPEAR] vid={k[0]} st={station} dir={k[2]} tts={cur.get('timeToStation')} eta={hhmmss_local(exp_local)}")
				events.append({
					"type": "APPEAR", "vehicleId": k[0], "naptanId": k[1],
					"platformDir": k[2], "stationName": station, "t_local": self._now_local(),
					"expectedArrival_local": exp_local,
				})

		# Convert lingering entries into PASS events once their expected arrival time is reached
		for k, last in list(self._linger.items()):
			exp = self._last_expected(last)
			if exp and exp <= now_local + self._linger_grace:
				station = last.get("stationName") or self.nid_to_name.get(k[1], "")
				if station:
					dest_raw_at_station = (last.get("destinationName") or last.get("towards") or "")
					_dbg(f"[PASS/linger] vid={k[0]} st={station} dir={k[2]} exp={hhmmss_local(exp)} dest='{dest_raw_at_station}'")
					events.append({
						"type": "PASS", "vehicleId": k[0], "naptanId": k[1],
						"platformDir": k[2], "stationName": station, "t_local": exp,
						"dest_raw": dest_raw_at_station,
					})
				self._linger.pop(k, None)

		self.prev_keys = curr_keys
		self.prev_rows = best
		return events

# Check for the empty destination or placeholder
def _is_unknown_label(dest_raw: str | None) -> bool:
	s = (dest_raw or "").strip().lower()
	return (not s) or (s in UNKNOWN_DEST_LABELS)

# =========== Classic Pass-Through Path + Shared Output Layer ===========
class PairLinker:
	"""
	Core writer + classic through tracker.
	
	PairLinker serves two roles:
	
	1) Classic A→B through-tracking  
	   - Detects PASS@A followed by PASS@B (with a renewable hold at B).  
	   - Uses the PASS time at A as the departure time.  
	   - Forbids A -> B through behaviour for specific section, point pairs. (Where other paths are used)
	   - Validates travel direction at A in a very permissive way to avoid misleading reverse travel sometimes reported by the API.
	   	
	2) Shared output layer for all other trackers  
	   - Terminus trackers, shuttle trackers, variable-run logic, impossible-terminus logic, and unknown-at-A/B trackers all call `record_terminus_departure_at_point()` to emit a departure.  
	   - PairLinker is the central place where all departure events are finalised and written.
	
	Output / cache behaviour:
	   - Every finalised event is appended to a runtime cache file: {PROJECT_DIR}/runningcache/{line_name}_events.log
	   - Write uses the computed DEPARTURE time (Departure at A).
	   - Automatically removes entries older than 3 hours on each write.
	   - Cache file is cleared for this line at the beginning of each run.
	"""

	def __init__(self, *, line_name: str, mode: str, sections: dict, out_root: str,
				 max_gap_min: int, forbid_through_points: Set[Tuple[str,str]],
				 # topology info (unused for classic through, but kept for other trackers)
				 weights: Dict[str,int],
				 veh_last_dest: Dict[str,str]):
		self.line_name = line_name
		self.mode = mode
		self.sections = sections or {}
		self.out_root = out_root
		self.max_gap = timedelta(minutes=max_gap_min)
		self.forbid_through = set(forbid_through_points)  # (sec_id, point)
		self.weights = weights or {}
		self.veh_last_dest = veh_last_dest  # kept for other trackers if needed
	
		# Index mapping: station -> list of all (section_id, point_name, A, B)
		# for which this station acts as A or B.
		self.index: Dict[str, List[Tuple[str,str,str,str]]] = {}
		for sec_id, sec in self.sections.items():
			for p_name, pair in (sec.get("points") or {}).items():
				if isinstance(pair, (list, tuple)) and len(pair) == 2:
					A, B = str(pair[0]).strip(), str(pair[1]).strip()
					self.index.setdefault(A, []).append((sec_id, p_name, A, B))
					self.index.setdefault(B, []).append((sec_id, p_name, A, B))

	
		# Pending classic-through links: (sec_id, point_name, vehicleId)
		#   "dep_dt_local": datetime at A,
		#   "dest_dir_at_A": "UP"|"DOWN"|"UNKNOWN",
		#   "dest_raw_at_A": raw destination string,
		#   "A": station_name_at_A,
		#   "B": station_name_at_B,
		self.pending_through: Dict[Tuple[str,str,str], Dict[str, Any]] = {}
		self.through_holds: Dict[Tuple[str,str,str], Dict[str, Any]] = {}
		self.results: DefaultDict[str, DefaultDict[Tuple[str,str], List[str]]] = defaultdict(lambda: defaultdict(list))
		self.seen_minute: DefaultDict[str, Set[Tuple[str,str,str]]] = defaultdict(set)
		
		self._suspects: Dict[str, Dict[str, Any]] = {}
		
		self._ban_until: Dict[str, Dict[str, Any]] = {}

	
		# running cache file setup
		self._cache_dir = os.path.join(PROJECT_DIR, "runningcache")
		os.makedirs(self._cache_dir, exist_ok=True)
		self._cache_path = os.path.join(self._cache_dir, f"{self.line_name}_events.log")
		# wipe old file for this line at start of run
		try:
			with open(self._cache_path, "w", encoding="utf-8"):
				pass
		except Exception as e:
			print(f"[warn] cannot reset cache file: {e}")
	
		_dbg(f"[pair-index] built for {sum(len(v.get('points',{})) for v in sections.values())} points")
		print(f"[cache] {self._cache_path}")
	
	def _append_to_cache(self, dep_dt_local: datetime, train_id: str, sec_id: str, p_name: str, via_label: str, dest: str):
		"""
		Append a new finalized departure to the line cache file.
		
		- Dedupes similar departures per section, point, minute (Treat last departure as real, overwrite previous).
		- On the Liberty line only, also replaces any departure at the same section and point in the previous 15 minutes.
	    - Applies a direction/destination check for classic pass-through to drop impossible moves.
        - Maintains simple ghost filtering and short-lived bans to avoid repeated bad writes.
		"""
		
		def _safe_parse_time_hms_to_dt(local_now: datetime, t_str: str) -> datetime:
			"""Parse 'HH:MM:SS' as a local datetime today, if that's over 1h in the future, shift back by one day."""
			t_local = datetime.strptime(t_str, "%H:%M:%S").time()
			candidate = local_now.replace(hour=t_local.hour, minute=t_local.minute, second=t_local.second, microsecond=0)
			if (candidate - local_now).total_seconds() > 3600:  # handle just-after-midnight wrap
				candidate -= timedelta(days=1)
			return candidate
		
		try:
			now = datetime.now(timezone.utc).astimezone()
			line_key = (getattr(self, "LINE_KEY", None) or getattr(self, "line_key", "") or "").lower()
			is_liberty = (line_key == "liberty")
		
			hhmmss = dep_dt_local.astimezone().strftime("%H:%M:%S")
			hhmm    = dep_dt_local.astimezone().strftime("%H:%M")   # minute key
			section_point = f"{sec_id} {p_name}"
			new_line = f"{hhmmss} - {train_id} - {section_point} - {via_label} - Destination: {strip_suffixes(dest)}\n"
		
			# Load existing cache
			lines = []
			if os.path.exists(self._cache_path):
				with open(self._cache_path, "r", encoding="utf-8") as f:
					lines = f.readlines()
		
			# Remove any prior entry at SAME section_point and time.
			# format is "HH:MM:SS - TrainID - <Section> <PointName> - ..."
			def _line_minute_and_section(ln: str):
				try:
					ln = ln.strip("\n")
					parts = ln.split(" - ")
					if len(parts) < 3:
						return None, None
					min_key = parts[0].strip()[:5]    # HH:MM
					sec_pt  = parts[2].strip()
					return min_key, sec_pt
				except Exception:
					return None, None
		
			for i in range(len(lines) - 1, -1, -1):
				min_key, sec_pt = _line_minute_and_section(lines[i])
				if (min_key == hhmm) and (sec_pt == section_point):
					_dbg(f"[cache/minute-replace] removing prior {sec_pt} at {min_key} (idx={i})")
					del lines[i]
		
			# Liberty-specific 15-minute overwrite on same section, point
			if is_liberty:
				fifteen_min_ago = dep_dt_local - timedelta(minutes=15)
		
				for i in range(len(lines) - 1, -1, -1):
					try:
						ln = lines[i].strip("\n")
						parts = ln.split(" - ")
						if len(parts) < 3:
							continue
						sec_pt = parts[2].strip()
						if sec_pt != section_point:
							continue
						old_hms = parts[0].strip()  # HH:MM:SS
						old_dt  = _safe_parse_time_hms_to_dt(now, old_hms)
						if fifteen_min_ago <= old_dt < dep_dt_local:
							_dbg(f"[liberty/15min-replace] removing prior {sec_pt} at {old_dt.strftime('%H:%M:%S')} (idx={i})")
							del lines[i]
					except Exception as e:
						_dbg(f"[warn] liberty 15-min check failed at idx={i}: {e}")
		
			# Ban map setup
			if not hasattr(self, "_bans"):
				self._bans = {}  # (train_id, dir) -> expiry datetime
		
			# Direction helper
			def _movement_dir(section_point_str: str) -> str:
				try:
					sid, pname = section_point_str.split(" ", 1)
					sec = self.sections.get(sid, {})
					points = sec.get("points", {})
					if pname not in points:
						return "UNKNOWN"
					A, B = points[pname]
					wa = self.weights.get(canon(A))
					wb = self.weights.get(canon(B))
					if wa is None or wb is None or wa == wb:
						return "UNKNOWN"
					return "UP" if wb > wa else "DOWN"
				except Exception:
					return "UNKNOWN"
		
			# Determine direction and sanity check for classic pass-through
			if via_label == "Classic Pass-Through" and strip_suffixes(dest).lower() not in UNKNOWN_DEST_LABELS:
				try:
					sec = self.sections.get(sec_id, {})
					A, B = sec["points"][p_name]
					wA, wB = self.weights.get(canon(A)), self.weights.get(canon(B))
					wDest = self.weights.get(canon(strip_suffixes(dest)))
					direction = "UNKNOWN"
					if wA is not None and wB is not None:
						direction = "UP" if wB > wA else "DOWN"
					if wDest is not None and direction != "UNKNOWN":
						if direction == "DOWN" and wDest > wB:
							_dbg(f"[ghost/impossible-direction] vid={train_id} dir={direction} dest>{B} – skipping")
							return
						if direction == "UP" and wDest < wB:
							_dbg(f"[ghost/impossible-direction] vid={train_id} dir={direction} dest<{B} – skipping")
							return
				except Exception as e:
					_dbg(f"[warn] dest-direction check failed: {e}")
		
			# Get recent lines for this train
			train_lines = [(i, l) for i, l in enumerate(lines) if f"- {train_id} -" in l]
			last_idx = train_lines[-1][0] if train_lines else None
			second_last_idx = train_lines[-2][0] if len(train_lines) >= 2 else None
			last_line = train_lines[-1][1].strip() if train_lines else None
			second_last_line = train_lines[-2][1].strip() if len(train_lines) >= 2 else None
		
			# Replace last/second-last same-section events
			for idx in [last_idx, second_last_idx]:
				if idx is None or idx >= len(lines):
					continue
				parts = lines[idx].split(" - ")
				if len(parts) >= 3 and parts[2].strip() == section_point:
					_dbg(f"[cache/replace] vid={train_id} removing prior same-section line idx={idx}")
					lines.pop(idx)
		
			# Ghost detection for classic pass-through
			is_suspect = (
				via_label == "Classic Pass-Through"
				and strip_suffixes(dest).lower() in UNKNOWN_DEST_LABELS
			)
			ban_dir = None
			if is_suspect:
				curr_dir = _movement_dir(section_point)
				prev_dir = _movement_dir(last_line.split(" - ")[2].strip()) if last_line else "UNKNOWN"
				if prev_dir == curr_dir:
					ban_dir = curr_dir
				elif second_last_line:
					sec_pt2 = second_last_line.split(" - ")[2].strip()
					dir2 = _movement_dir(sec_pt2)
					if dir2 == curr_dir:
						ban_dir = curr_dir
				if ban_dir:
					_dbg(f"[ghost/suspect] vid={train_id} same-dir={ban_dir} sec={sec_id} point={p_name}")
				else:
					_dbg(f"[ghost/no-flag] vid={train_id} dir={curr_dir} prev={prev_dir}")
		
			# Apply ban check before appending
			curr_dir = _movement_dir(section_point)
			if (train_id, curr_dir) in self._bans:
				expiry = self._bans[(train_id, curr_dir)]
				if now < expiry:
					_dbg(f"[ghost/ban-block] vid={train_id} dir={curr_dir} (ban until {hhmmss_local(expiry)}) – skipping write")
					return
				else:
					del self._bans[(train_id, curr_dir)]
		
			# Append new line
			lines.append(new_line)
		
			# Trim old lines
			def parse_time(ln: str) -> datetime:
				try:
					tok = ln.split(" - ", 1)[0].strip()
					return _safe_parse_time_hms_to_dt(now, tok)
				except Exception:
					return now
		
			fresh = [(parse_time(ln), ln) for ln in lines if (now - parse_time(ln)) <= timedelta(hours=3)]
			fresh.sort(key=lambda x: x[0])
			lines = [ln for _, ln in fresh]
		
			# Ghost correction
			if ban_dir:
				for i in range(len(lines) - 1, -1, -1):
					if f"- {train_id} -" not in lines[i]:
						continue
					sec_pt = lines[i].split(" - ")[2].strip()
					dir_here = _movement_dir(sec_pt)
					if dir_here == ban_dir:
						_dbg(f"[ghost/remove] vid={train_id} reversed from {ban_dir} – deleting suspect line {sec_pt}")
						del lines[i]
						break
				self._bans[(train_id, ban_dir)] = now + timedelta(minutes=5)
				_dbg(f"[ghost/ban-set] vid={train_id} dir={ban_dir} until {hhmmss_local(self._bans[(train_id, ban_dir)])}")
		
			# Write updated cache
			with open(self._cache_path, "w", encoding="utf-8") as f:
				f.writelines(lines)
		
		except Exception as e:
			# Print failures
			print(f"[warn] cache ghost logic failed: {e}")
			traceback.print_exc()
	


	
	def _record_completion(self, sec_id: str, p_name: str, dep_dt_local: datetime, via: str, train_id: str = "UNKNOWN", dest: str = "UNKNOWN"):
		"""
		Finalize a single departure:
		  - update in-memory stats,
		  - log to the running cache file,
		  - rebuild the XML snapshot for the relevant hour
		"""
		# Round display time to the nearest minute (by adding 30s)
		display_dt = dep_dt_local + timedelta(seconds=30)
		hhmm = display_dt.strftime("%H:%M")
		start, end, hour_key = window_from_departure(display_dt)
		
		key3 = (sec_id, p_name, hhmm)
		self.results[hour_key][(sec_id, p_name)].append(hhmm)
		
		# Label utilised path
		path_label = {
			"arrivals-pass": "Classic Pass-Through",
			"terminus-final-left": "Terminus Departure",
			"b-terminus-through": "B-Terminus-Through",
			"impossible-terminus": "Impossible-Termini Departure",
			"unknown-at-a-terminus": "Unknown-At-A Termini Departure",
			"move-one-terminus": "Move-One Termini Departure",
			"variable-run-terminus-A": "Variable-Run Terminus A",
			"shuttle-ab": "Shuttle A→B",
		}.get(via, via)
		
		print(
			f"{hhmmss_local(datetime.now(timezone.utc))} "
			f"[XML ADDED] Train {train_id} Spotted! Destination: {strip_suffixes(dest)}, "
			f"Departure time: {hhmm}, Path: {path_label}, "
			f"Section: {sec_id} {p_name}"
		)
		
		# Append to running cache using the departure at A time
		self._append_to_cache(dep_dt_local, train_id, sec_id, p_name, path_label, dest)
		
		# XML snapshot
		self._write_hour_xml(hour_key, start, end)

	
	
	# Entry point used by terminus paths to emit a departure
	def record_terminus_departure_at_point(self, *, sec_id: str, point_name: str, dep_dt_local: datetime, via: str = "terminus-final-left", train_id: str = "UNKNOWN", dest: str = "UNKNOWN"):
		self._record_completion(sec_id, point_name, dep_dt_local, via=via, train_id=train_id, dest=dest)
	
	
	# helpers for direction gate
	def _w(self, name: str) -> int | None:
		return self.weights.get(canon(name))
	
	def _pair_dir(self, A: str, B: str) -> str | None:
		wa, wb = self._w(A), self._w(B)
		if wa is None or wb is None or wa == wb:
			return None
		return "UP" if wb > wa else "DOWN"
	
	def _dest_dir_vs_A(self, dest_raw: str | None, A: str) -> str:
		if _is_unknown_label(dest_raw):
			return "UNKNOWN"
		wa = self._w(A)
		wd = self._w(dest_raw or "")
		if wa is None or wd is None or wa == wd:
			return "UNKNOWN"
		return "UP" if wd > wa else "DOWN"
	
	def _snapshot_allows(self, dest_dir_at_A: str | None, A: str, B: str) -> Tuple[bool, str]:
		pair_dir = self._pair_dir(A, B) or "UNKNOWN"
		# Rules: unknown/not-listed dest at A => allow; unknown pair_dir => allow; else must match
		if (dest_dir_at_A is None) or (dest_dir_at_A == "UNKNOWN") or (pair_dir == "UNKNOWN"):
			return True, pair_dir
		return (dest_dir_at_A == pair_dir), pair_dir


	def observe_event(self, ev: Dict[str,Any]):
		"""
		Consume a single PASS/APPEAR event and update classic A-B through state.
		"""
		etype = ev.get("type")      # PASS or APPEAR
		veh   = str(ev.get("vehicleId") or "")
		st_nm = str(ev.get("stationName") or "").strip()
		t_loc = ev.get("t_local")
		if not (etype and veh and st_nm and isinstance(t_loc, datetime)):
			return
	
		# Classic through only cares about PASS events
		if etype != "PASS":
			return
	
		now_local = datetime.now(timezone.utc).astimezone()
		
		# For this station, find section, point, A, B pairs it participates in
		for sec_id, p_name, A, B in self.index.get(st_nm, []):
			if (sec_id, p_name) in self.forbid_through:
				continue
			key = (sec_id, p_name, veh)
	
			if st_nm == A:
				# Snapshot destination at A and compute direction
				dest_at_A_raw = ev.get("dest_raw")
				dest_dir_at_A = self._dest_dir_vs_A(dest_at_A_raw, A)
				self.pending_through[key] = {
					"dep_dt_local": t_loc,
					"dest_dir_at_A": dest_dir_at_A,
					"dest_raw_at_A": dest_at_A_raw,
					"A": A, "B": B,
				}
				_dbg(f"[through] PASS@A hold A-time set sec={sec_id} point={p_name} vid={veh} A={A} t={hhmmss_local(t_loc)} dest='{dest_at_A_raw}' dirA={dest_dir_at_A}")
				continue
	
			# try to match to a previous PASS at A for the same section, point, train at B
			if st_nm == B:
				state = self.pending_through.pop(key, None)
				if not state:
					continue
				dep_dt_local = state.get("dep_dt_local")
				if not isinstance(dep_dt_local, datetime):
					continue
				if not (timedelta(0) <= (t_loc - dep_dt_local) <= self.max_gap):
					_dbg(f"[through/drop] A->B gap out of range sec={sec_id} point={p_name} vid={veh}")
					continue
	
				hold = self.through_holds.get(key)
				if not hold:
					# First time a valid A-B pair for this train is observed, start a renewable hold
					self.through_holds[key] = {
						"deadline": now_local + timedelta(minutes=ATTEMPT_WINDOW_MIN),
						"dep_dt_local": dep_dt_local,
						"last_b_time": t_loc,
						"sec_id": sec_id,
						"p_name": p_name,
						"A": A,
						"B": B,
						"dest_dir_at_A": state.get("dest_dir_at_A", "UNKNOWN"),
						"dest_raw_at_A": state.get("dest_raw_at_A", ""),
					}
					_dbg(f"[through/hold:new] sec={sec_id} point={p_name} vid={veh} dep(A)={hhmmss_local(dep_dt_local)} deadline=+{ATTEMPT_WINDOW_MIN}m (dirA={self.through_holds[key]['dest_dir_at_A']} destA='{self.through_holds[key]['dest_raw_at_A']}')")
				else:
					# If a hold for a pair is present either renew or finalise
					if now_local <= hold["deadline"]:
						hold["deadline"] = now_local + timedelta(minutes=ATTEMPT_WINDOW_MIN)
						hold["dep_dt_local"] = dep_dt_local
						hold["last_b_time"] = t_loc
						hold["A"], hold["B"] = A, B
						# Update snapshot from the latest A (same journey, fresher info)
						hold["dest_dir_at_A"] = state.get("dest_dir_at_A", hold.get("dest_dir_at_A","UNKNOWN"))
						hold["dest_raw_at_A"] = state.get("dest_raw_at_A", hold.get("dest_raw_at_A",""))
						_dbg(f"[through/hold:renew] sec={sec_id} point={p_name} vid={veh} dep(A)={hhmmss_local(dep_dt_local)} deadline->+{ATTEMPT_WINDOW_MIN}m (dirA={hold['dest_dir_at_A']} destA='{hold['dest_raw_at_A']}')")
					else:
						# Hold expired: decide now whether to write the through event
						_dbg(f"[through/hold:lapsed] sec={sec_id} point={p_name} vid={veh} writing immediately")
						ok, pair_dir = self._snapshot_allows(hold.get("dest_dir_at_A","UNKNOWN"), hold.get("A", A), hold.get("B", B))
						if ok:
							if timedelta(0) <= (hold["last_b_time"] - hold["dep_dt_local"]) <= self.max_gap:
								self._record_completion(
									hold["sec_id"],
									hold["p_name"],
									hold["dep_dt_local"],
									via="arrivals-pass",
									train_id=veh,
									dest=(strip_suffixes(hold.get("dest_raw_at_A") or "UNKNOWN"))
								)
	
						else:
							_dbg(f"[through/filter-drop] sec={hold['sec_id']} point={hold['p_name']} vid={veh} pair_dir={pair_dir} dest_dir_at_A={hold.get('dest_dir_at_A')} destA='{hold.get('dest_raw_at_A','')}'")
						self.through_holds.pop(key, None)
	
	def tick(self, now_local: datetime):
		"""
		Runs periodic maintenance for classic A, B through tracking.
		
		Any hold whose deadline has expired is either:
		  - written as a completed departure if it still meets time/direction rules or
		  - discarded if the match is no longer valid.
		"""
		for key, hold in list(self.through_holds.items()):
			if now_local >= hold["deadline"]:
				if timedelta(0) <= (hold["last_b_time"] - hold["dep_dt_local"]) <= self.max_gap:
					# Gate using snapshot-at-A only
					sec_id, p_name = hold["sec_id"], hold["p_name"]
					A, B = hold.get("A"), hold.get("B")
					ok, pair_dir = self._snapshot_allows(hold.get("dest_dir_at_A","UNKNOWN"), A, B)
					if ok:
						_dbg(f"[through/finalize] sec={sec_id} point={p_name} vid={key[2]} dep(A)={hhmmss_local(hold['dep_dt_local'])} (dirA={hold.get('dest_dir_at_A')} pair_dir={pair_dir})")
						self._record_completion(
							sec_id,
							p_name,
							hold["dep_dt_local"],
							via="arrivals-pass",
							train_id=key[2],
							dest=(strip_suffixes(hold.get("dest_raw_at_A") or "UNKNOWN"))
						)
	
					else:
						_dbg(f"[through/filter-drop] sec={sec_id} point={p_name} vid={key[2]} pair_dir={pair_dir} dest_dir_at_A={hold.get('dest_dir_at_A')} destA='{hold.get('dest_raw_at_A','')}'")
				else:
					_dbg(f"[through/finalize:drop] gap exceeded sec={hold['sec_id']} point={hold['p_name']} vid={key[2]}")
				self.through_holds.pop(key, None)
	
	def _write_hour_xml(self, hour_key: str, start_local: datetime, end_local: datetime):
		"""
		Rebuild the XML snapshot for a single hour window by:
		- Reading all cached departures,
		- Selecting those that fall within [start_local, end_local),
		- Computing TPH and headways per (section, point),
		- Writing a fresh RealRuns XML for that hour.
		"""
		from xml.dom import minidom
		import xml.etree.ElementTree as ET
		
		# Read cache
		try:
			with open(self._cache_path, "r", encoding="utf-8") as f:
				lines = f.readlines()
		except FileNotFoundError:
			return
		
		# Parse cache lines -> collect events within the hour window
		events: Dict[Tuple[str, str], List[str]] = defaultdict(list)
		
		for raw in lines:
			line = raw.strip()
			if not line:
				continue
		
			# Expected format:
			# "HH:MM:SS - TrainID - <Section> <PointName> - <Path> - Destination: ..."
			parts = line.split(" - ", 3)
			if len(parts) < 3:
				continue
		
			dep_str = parts[0].strip()           # "HH:MM:SS"
			section_point = parts[2].strip()     # "<Section> <PointName>"
		
			# Parse departure time in the same tz as start/end
			try:
				t_only = datetime.strptime(dep_str, "%H:%M:%S").time()
			except ValueError:
				continue
		
			dep_time = datetime.combine(start_local.date(), t_only).replace(tzinfo=start_local.tzinfo)
		
			if (dep_time - end_local) > timedelta(hours=1):
				dep_time -= timedelta(days=1)
			elif (start_local - dep_time) > timedelta(hours=23):
				dep_time += timedelta(days=1)
		
			# Keep only departures within the window
			if not (start_local <= dep_time < end_local):
				continue
		
			# Split "<Section> <PointName>" -> (sec_id, point_name)
			try:
				sec_id, point_name = section_point.split(" ", 1)
			except ValueError:
				continue
		
			events[(sec_id.strip(), point_name.strip())].append(dep_time.strftime("%H:%M"))
		
		# Rebuild XML from collected events
		ymd = start_local.strftime("%Y%m%d")
		out_dir = os.path.join(self.out_root, ymd)
		os.makedirs(out_dir, exist_ok=True)
		out_path = os.path.join(out_dir, f"{self.line_name.lower()}_{hour_key}.xml")
		
		root = ET.Element("RealRuns", attrib={
			"date": start_local.strftime("%Y-%m-%d"),
			"line": self.line_name,
			"mode": self.mode,
			"tz": str(start_local.tzinfo) if start_local.tzinfo else "Europe/London",
			"window": f"{start_local.strftime('%H')}:00-{end_local.strftime('%H')}:00",
		})
		
		for sec_id, sec in self.sections.items():
			sec_el = ET.SubElement(root, "Section"); sec_el.text = sec_id
			points = (sec.get("points") or {})
			for p_name in points.keys():
				times = sorted(events.get((sec_id, p_name), []))
				start_hhmm = start_local.strftime("%H%M")
				end_hhmm   = end_local.strftime("%H%M")
				start_hm = f"{start_hhmm[:2]}:{start_hhmm[2:]}"
				end_hm   = f"{end_hhmm[:2]}:{end_hhmm[2:]}"
				tph = sum(1 for hm in times if start_hm <= hm < end_hm)
				headways = headways_from_times(times, start_hhmm, end_hhmm)
		
				p_el = ET.SubElement(sec_el, "Point"); p_el.text = p_name
				t_el = ET.SubElement(p_el, "Time");   t_el.text = f"{start_local.strftime('%H')}:00-{end_local.strftime('%H')}:00"
				tph_el = ET.SubElement(p_el, "TPH");  tph_el.text = str(tph)
				h_el = ET.SubElement(p_el, "Headways"); h_el.text = " ".join(str(x) for x in headways)
		
		xml = minidom.parseString(ET.tostring(root, encoding="utf-8")).toprettyxml(
			indent="   ", encoding="utf-8"
		).decode()
		
		with open(out_path, "w", encoding="utf-8") as f:
			f.write(xml)
		
		print(f"[xml/rebuilt] {out_path}")



# =========== Standard Terminus Tracker ===========
# Classify a destination as UP/DOWN/UNKNOWN relative to terminus based on weights
def classify_dest_vs_B(dest_raw: str, B_weight: int, weight_by_canon: dict) -> str:
	c = canon(dest_raw)
	w = weight_by_canon.get(c)
	if w is None: return "UNKNOWN"
	if w > B_weight: return "UP"
	if w < B_weight: return "DOWN"
	return "DOWN"

class TerminusTracker:
	"""
	Tracks departures from station B to determine departures from terminus (A) 
	upstream based on configured values from line modelling
	
	For a given (sec_id, point_name, A, B):
	
	- Watch trains at B that are "soon" and heading in launch_dir (or have an unknown but acceptable destination).
	- When such a train disappears from B, treat its ETA at B minus min_run_min as the departure time from (sec_id, point_name). (Represents departure time at A)
	- If the destination is known, write a "terminus-final-left" departure immediately at that section/point.
	- If the destination was unknown, wait for confirmation at the next downstream station(s) before writing.
	"""
	def __init__(self, *, sec_id: str, point_name: str,
				 A_full: str, B_full: str, B_weight: int, launch_dir: str,
				 min_run_min: int, weights: dict, value_to_display: Dict[int,List[str]], xml_linker: PairLinker,
				 station_suffix: str):
		# Sections and points to write departures for
		self.sec_id = sec_id
		self.point_name = point_name
		self.A = A_full
		self.B = B_full
		self.B_weight = B_weight                   # numeric weight of B in the line ordering
		self.launch_dir = launch_dir               # "up" or "down" direction A -> B
		self.min_run_min = int(min_run_min)
		# Line topology + display helpers
		self.weights = weights
		self.value_to_display = value_to_display
		self.linker = xml_linker
		self.station_suffix = station_suffix
		# State at B and pending decisions
		self.tracked: Dict[str, Dict] = {}
		self.prev_vids: Set[str] = set()
		self.pending_departures: Dict[str, Dict] = {}
		self.pending_unknown_confirms: Dict[str, Dict] = {}

		# Compute neighbour just beyond B in the launch direction (used for unknown dest confirms)
		step = 1 if self.launch_dir == "up" else -1
		neighbor_w = self.B_weight + step
		neighbor_names = self.value_to_display.get(neighbor_w) or []
		# Elizabeth uses different suffix configuration
		if LINE_KEY == "elizabeth":
			self.neighbor_stations_full: List[str] = expand_with_suffixes(neighbor_names, self.station_suffix)
		else:
			self.neighbor_stations_full: List[str] = [f"{nm}{self.station_suffix}" for nm in neighbor_names]

	def process_rows_at_B(self, rows_at_B: List[Dict[str,Any]], now_local: datetime, rows_by_station: Dict[str, List[Dict[str,Any]]]):
		"""
		Process one polling batch of arrivals rows specifically at station B.
		
		- Keeps best (lowest tts) row per train at B.
		- Tracks trains that are "soon" and in the launch direction (or unknown).
		- On disappearance from B, starts a short hold to infer departure time at our section.
		- After the hold:
			- known dest -> write immediately;
			- unknown dest -> optionally confirm at downstream station(s) before writing.
		"""
		best_by_vid: Dict[str, Dict] = {}
		for it in rows_at_B:
			vid = str(it.get("vehicleId") or "")
			if not vid:
				continue
			row = {
				"vid": vid,
				"dest_raw": (it.get("destinationName") or it.get("destination") or it.get("towards") or "").strip(),
				"tts": int(it.get("timeToStation") or 1e9),
				"expectedArrival": it.get("expectedArrival"),
			}
			prev = best_by_vid.get(vid)
			if prev is None or row["tts"] < prev["tts"]:
				best_by_vid[vid] = row

		curr_vids = set(best_by_vid.keys())

		# Track trains that are "soon" at B and heading in the target direction (or unknown destination)
		for vid, row in best_by_vid.items():
			if row["tts"] > SOON_THRESHOLD_SEC:
				continue

			dest_raw = row["dest_raw"]
			exp = parse_dt(row.get("expectedArrival"))
			# Direction of destination relative to B, using the weight model
			cls = classify_dest_vs_B(dest_raw, self.B_weight, self.weights)
			is_unknown_label = _is_unknown_label(dest_raw)
			is_target_dir = (cls != "UNKNOWN" and cls.lower() == self.launch_dir) or is_unknown_label

			_dbg(f"[term/B<=2m] B={self.B} vid={vid} dest='{dest_raw}' canon='{canon(dest_raw)}' cls={cls} target={is_target_dir} eta={hhmmss_local(exp)}")

			# Update existing tracked ETA at B, if vid is already known
			if vid in self.tracked and isinstance(exp, datetime):
				self.tracked[vid]["exp_arrival"] = exp.astimezone()

			# Start tracking new target trains at B
			if is_target_dir:
				if vid not in self.tracked:
					self.tracked[vid] = {
						"dest": strip_suffixes(dest_raw) if not is_unknown_label else "(unknown)",
						"exp_arrival": exp.astimezone() if isinstance(exp, datetime) else now_local,
						"appeared_at": now_local,
						"unknown": bool(is_unknown_label),
					}

		# Disappearances -> begin/refresh hold
		disappeared = self.prev_vids - curr_vids
		for vid in disappeared:
			info = self.tracked.get(vid)
			if not info:
				continue
			dest = info.get("dest","")
			exp = info.get("exp_arrival")
			dep_time = (exp - timedelta(minutes=self.min_run_min)).astimezone() if isinstance(exp, datetime) else now_local
			self.pending_departures[vid] = {
				"deadline": now_local + timedelta(minutes=ATTEMPT_WINDOW_MIN),
				"last_dep_time": dep_time,
				"dest": dest,
				"unknown": bool(info.get("unknown", False)),
			}
			_dbg(f"[term/B-miss] start hold B={self.B} vid={vid} dest={dest} dep={hhmmss_local(dep_time)}")

		# Finalize after deadline
		for vid, pend in list(self.pending_departures.items()):
			if vid in curr_vids:
				continue
			if now_local >= pend["deadline"]:
				if pend.get("unknown") and self.neighbor_stations_full:
					confirm_until = now_local + timedelta(minutes=UNK_CONFIRM_MIN)
					self.pending_unknown_confirms[vid] = {
						"confirm_until": confirm_until,
						"dep_dt_local": pend["last_dep_time"],
						"confirm_stations": list(self.neighbor_stations_full),
						"dest_hint": pend.get("dest","(unknown)"),
					}
					_dbg(f"[term/B-unknown-hold] vid={vid} until=+{UNK_CONFIRM_MIN}m next={self.neighbor_stations_full}")
				else:
					self.linker.record_terminus_departure_at_point(
						sec_id=self.sec_id,
						point_name=self.point_name,
						dep_dt_local=pend["last_dep_time"],
						via="terminus-final-left",
						train_id=vid,
						dest=(pend.get("dest") or "UNKNOWN")
					)

				self.pending_departures.pop(vid, None)
				self.tracked.pop(vid, None)

		# Check unknown confirmations
		for vid, state in list(self.pending_unknown_confirms.items()):
			if now_local > state["confirm_until"]:
				_dbg(f"[term/B-unknown-drop] vid={vid} no confirm seen")
				self.pending_unknown_confirms.pop(vid, None)
				continue

			for st_name in state["confirm_stations"]:
				for it in rows_by_station.get(st_name, []):
					if str(it.get("vehicleId") or "") != vid:
						continue
					tts = int(it.get("timeToStation") or 1e9)
					if tts <= SOON_THRESHOLD_SEC:
						self.linker.record_terminus_departure_at_point(
							sec_id=self.sec_id,
							point_name=self.point_name,
							dep_dt_local=state["dep_dt_local"],
							via="terminus-final-left",
							train_id=vid,
							dest=(state.get("dest_hint") or "UNKNOWN")
						)

						self.pending_unknown_confirms.pop(vid, None)
						break

		self.prev_vids = curr_vids

# =========== Unknown at A Terminus Tracker ===========
class UnknownAtTerminusAWatcher:
	"""
	Handles the special case where trains sit at terminus A with an unknown destination label.
	
	- Watch rows at A where:
		- destination label is one of the "unknown" labels, and
		- The train is soon (according to tts).
	- When such a train disappears from A, remember its ETA(A) as the inferred departure time from (sec_id, point_name).
	- For a limited window, watch the next station(s) beyond A in launch_dir.
		- If the same train appears there "soon", emit an 'unknown-at-a-terminus' departure at (sec_id, point_name) with dest="UNKNOWN".
	    - If no confirmation appears before the deadline, drop the candidate.
	* This path is less reliable than normal terminus tracker, since terminus departure data doesn't appear as prediction. 
	  Most of the predictions are for trains arriving to the terminus, trains that actually depart appear only for a few sec.
	"""
	def __init__(self, *, sec_id: str, point_name: str,
				 A_full: str, launch_dir: str, wa: int,
				 value_to_display: Dict[int,List[str]],
				 xml_linker: PairLinker, station_suffix: str):
		# Sections and points to write departures for
		self.sec_id = sec_id
		self.point_name = point_name
		self.A = A_full
		self.launch_dir = launch_dir           # Direction from A, UP or DOWN
		self.wa = int(wa)                      # numeric weight of A
		self.linker = xml_linker
		self.station_suffix = station_suffix
		
		# Compute stations immediately beyond A in launch_dir
		step = 1 if self.launch_dir == "up" else -1
		neighbor_w = self.wa + step
		neighbor_names = value_to_display.get(neighbor_w) or []
		if LINE_KEY == "elizabeth":
			self.next_stations_full: List[str] = expand_with_suffixes(neighbor_names, self.station_suffix)
		else:
			self.next_stations_full: List[str] = [f"{nm}{self.station_suffix}" for nm in neighbor_names]

		# tracked_at_A: vid -> info while "soon" at A with unknown dest
		# prev_vids: vids seen at A in previous poll
		# pending_confirms: vid -> waiting for downstream confirmation
		self.tracked_at_A: Dict[str, Dict] = {}
		self.prev_vids: Set[str] = set()
		self.pending_confirms: Dict[str, Dict] = {}

	def process_rows_at_A(self, rows_at_A: List[Dict[str,Any]], now_local: datetime, rows_by_station: Dict[str, List[Dict[str,Any]]]):
		"""
		Process one polling batch of arrivals rows at A, focusing only on
		trains with 'unknown' destination labels that are 'soon' at A.
		"""
		best_by_vid: Dict[str, Dict] = {}
		for it in rows_at_A:
			vid = str(it.get("vehicleId") or "")
			if not vid:
				continue
			dest_raw = (it.get("destinationName") or it.get("destination") or it.get("towards") or "").strip()
			if not _is_unknown_label(dest_raw):
				continue
			tts = int(it.get("timeToStation") or 1e9)
			if tts > SOON_THRESHOLD_SEC:
				continue
			expA = parse_dt(it.get("expectedArrival"))
			row = {
				"vid": vid,
				"tts": tts,
				"expA_local": expA.astimezone() if isinstance(expA, datetime) else now_local,
			}
			prev = best_by_vid.get(vid)
			if prev is None or row["tts"] < prev["tts"]:
				best_by_vid[vid] = row

		curr_vids = set(best_by_vid.keys())

		for vid, row in best_by_vid.items():
			if vid not in self.tracked_at_A:
				self.tracked_at_A[vid] = {"expA_local": row["expA_local"], "seen_at": now_local, "last_seen_tts": row["tts"]}
				_dbg(f"[term/A<=2m] A={self.A} vid={vid} expA={hhmmss_local(row['expA_local'])}")
			else:
				self.tracked_at_A[vid]["expA_local"] = row["expA_local"]
				self.tracked_at_A[vid]["last_seen_tts"] = row["tts"]

		disappeared = self.prev_vids - curr_vids
		for vid in disappeared:
			info = self.tracked_at_A.get(vid)
			if not info:
				continue
			dep_dt_local = info["expA_local"]
			confirm_until = now_local + timedelta(minutes=ATTEMPT_WINDOW_MIN + UNK_CONFIRM_MIN)
			self.pending_confirms[vid] = {
				"confirm_until": confirm_until,
				"dep_dt_local": dep_dt_local,
			}
			_dbg(f"[term/A-miss] start confirm A={self.A} vid={vid} dep={hhmmss_local(dep_dt_local)} next={self.next_stations_full}")
			self.tracked_at_A.pop(vid, None)

		for vid, st in list(self.pending_confirms.items()):
			if now_local > st["confirm_until"]:
				self.pending_confirms.pop(vid, None)
				_dbg(f"[term/A-drop] no confirm A={self.A} vid={vid}")
				continue

			found = False
			for st_name in self.next_stations_full:
				for it in rows_by_station.get(st_name, []):
					if str(it.get("vehicleId") or "") != vid:
						continue
					tts = int(it.get("timeToStation") or 1e9)
					if tts <= SOON_THRESHOLD_SEC:
						self.linker.record_terminus_departure_at_point(
							sec_id=self.sec_id,
							point_name=self.point_name,
							dep_dt_local=st["dep_dt_local"],
							via="unknown-at-a-terminus",
							train_id=vid,
							dest="UNKNOWN"
						)


						self.pending_confirms.pop(vid, None)
						found = True
						break
				if found:
					break

		self.prev_vids = curr_vids
		
# =========== Shuttle A -> B Tracker ===========
class ShuttleABTracker:
		"""
		Shuttle logic is unused in the current version and requires further development
		"""
		def __init__(self, *, sec_id: str, point_name: str, A_full: str, B_full: str, linker: PairLinker):
			self.sec_id = sec_id
			self.point_name = point_name
			self.A = A_full
			self.B = B_full
			self.linker = linker
		
			# A-side state
			self.last_etaA: Dict[str, datetime] = {}
			self.last_ttsA: Dict[str, int] = {}
			self.visibleA: Set[str] = set()
			self.A_timers: Dict[str, Dict[str, Any]] = {}          # vid -> {"deadline": dt, "dep_dt_local": dt}
			self.A_departed_at: Dict[str, datetime] = {}           # vid -> dep_dt_local
			self.A_seeded_at: Dict[str, datetime] = {}             # outer MAX_LINK_MIN guard
		
			self._written: Set[str] = set()
		
		def _best_eta_per_vid(self, rows: List[Dict[str, Any]], now_local: datetime) -> Dict[str, Tuple[datetime, int, str]]:
			best: Dict[str, Tuple[int, datetime, str]] = {}
			for it in rows:
				vid = str(it.get("vehicleId") or "")
				if not vid:
					continue
				try:
					tts = int(it.get("timeToStation") or 10**9)
				except Exception:
					tts = 10**9
				eta = parse_dt(it.get("expectedArrival"))
				eta_local = eta.astimezone() if isinstance(eta, datetime) else now_local
				dest_raw = (it.get("destinationName") or it.get("destination") or it.get("towards") or "").strip()
				prev = best.get(vid)
				if (prev is None) or (tts < prev[0]):
					best[vid] = (tts, eta_local, dest_raw)
			return {vid: (eta, tts, dest) for vid, (tts, eta, dest) in best.items()}
		
		def process(self, rows_by_station: Dict[str, List[Dict[str, Any]]], now_local: datetime):
			rows_A = rows_by_station.get(self.A, [])
			A_best = self._best_eta_per_vid(rows_A, now_local)
			currA: Set[str] = set()
			for vid, (etaA, ttsA, destA) in A_best.items():
				if canon(destA) == canon(self.A):
					continue
				self.last_etaA[vid] = etaA
				self.last_ttsA[vid] = int(ttsA)
				if ttsA <= SOON_THRESHOLD_SEC:
					currA.add(vid)
		
			for vid in (self.visibleA - currA):
				etaA = self.last_etaA.get(vid)
				ttsA = self.last_ttsA.get(vid, 10**9)
				if etaA and ttsA <= SOON_THRESHOLD_SEC:
					self.A_timers[vid] = {
						"deadline": now_local + timedelta(minutes=ATTEMPT_WINDOW_MIN),
						"dep_dt_local": etaA,
					}
					self.A_seeded_at.setdefault(vid, now_local)
					_dbg(f"[shuttle/A-miss] A={self.A} vid={vid} dep(A)={hhmmss_local(etaA)} -> +{ATTEMPT_WINDOW_MIN}m")
		
			for vid in currA:
				if vid in self.A_timers:
					self.A_timers[vid]["deadline"] = now_local + timedelta(minutes=ATTEMPT_WINDOW_MIN)
					etaA_now = self.last_etaA.get(vid)
					if isinstance(etaA_now, datetime) and etaA_now > self.A_timers[vid]["dep_dt_local"]:
						self.A_timers[vid]["dep_dt_local"] = etaA_now
					_dbg(f"[shuttle/A-renew] A={self.A} vid={vid} dep(A)=>{hhmmss_local(self.A_timers[vid]['dep_dt_local'])}")
		
			for vid, hold in list(self.A_timers.items()):
				if now_local >= hold["deadline"]:
					self.A_departed_at[vid] = hold["dep_dt_local"]
					_dbg(f"[shuttle/A-departed] A={self.A} vid={vid} dep(A)={hhmmss_local(hold['dep_dt_local'])}")
					del self.A_timers[vid]
		
			for vid, dep_dt in list(self.A_departed_at.items()):
				if (now_local - dep_dt) > timedelta(minutes=MAX_LINK_MIN):
					_dbg(f"[shuttle/timeout] A→B window exceeded vid={vid}")
					del self.A_departed_at[vid]
					self._written.discard(vid)
					self.A_seeded_at.pop(vid, None)

			rows_B = rows_by_station.get(self.B, [])
			B_best = self._best_eta_per_vid(rows_B, now_local)  
			for vid, (_etaB, _ttsB, destB) in B_best.items():
				if vid in self._written:
					continue
				dep_dt = self.A_departed_at.get(vid)
				if not isinstance(dep_dt, datetime):
					continue
				if canon(destB) != canon(self.B):
					continue
				if now_local < dep_dt:
					continue
		
				self.linker.record_terminus_departure_at_point(
					sec_id=self.sec_id,
					point_name=self.point_name,
					dep_dt_local=dep_dt,
					via="shuttle-ab",
					train_id=vid,
					dest=strip_suffixes(self.B)
				)
				_dbg(f"[shuttle/write] A={self.A} B={self.B} vid={vid} LEFT={hhmm_local(dep_dt)} dest=B")
				self._written.add(vid)
				self.A_departed_at.pop(vid, None)
				self.A_seeded_at.pop(vid, None)

			self.visibleA = currA


# =========== Impossible-terminus tracker ===========
class ImpossibleTerminusTracker:
	"""
	Tracker for 'impossible' termini where the normal A-B terminus logic
	is not usable (e.g. A sits alone on a branch)
	
	- At A, watch outbound trains (dest != A) that are soon (according to tts)
	- When a train first becomes due, remember it and open a short confirmation window.
    - During that window, look for the same vid at the next station beyond A (in launch_dir), "soon" there.
	- If seen, use ETA(next) minus min_run_min as the inferred departure time from the section/point associated with A, and write an 'impossible-terminus' departure.
	- If not seen by the end of the window, drop the candidate.
	
	* This path is less reliable than normal terminus tracker, since terminus departure data doesn't appear as prediction. 
	  Most of the predictions are for trains arriving to the terminus, trains that actually depart appear only for a few sec.
	"""
	def __init__(self, *, sec_id: str, point_name: str,
				 A_full: str, launch_dir: str, wa: int,
				 min_run_min: int, value_to_display: Dict[int,List[str]],
				 xml_linker: PairLinker, station_suffix: str):
		# Sections and points to write departures for
		self.sec_id = sec_id
		self.point_name = point_name
		# 'Impossible' terminus A info
		self.A = A_full
		self.launch_dir = launch_dir		   # Direction from A, UP or DOWN
		self.wa = int(wa)					   # numeric weight of A
		self.min_run_min = int(min_run_min)
		self.linker = xml_linker
		self.station_suffix = station_suffix
		
		# Next station(s) beyond A in launch_dir, used for confirmation
		step = 1 if self.launch_dir == "up" else -1
		neighbor_w = self.wa + step
		neighbor_names = value_to_display.get(neighbor_w) or []
		
		# Elizabeth line suffix difference
		if LINE_KEY == "elizabeth":
			self.next_stations_full: List[str] = expand_with_suffixes(neighbor_names, self.station_suffix)
		else:
			self.next_stations_full: List[str] = [f"{nm}{self.station_suffix}" for nm in neighbor_names]

		# vid -> "t0": first time marked as "due at A", "dest": raw dest string
		self.outbound_due_vids: Dict[str, datetime] = {}
		self.terminus_key = canon(self.A)

	def _is_due(self, row: Dict[str,Any], now_local: datetime) -> bool:
		"""
		Is this row at A effectively 'due to depart'?
		Use any of:
		  - TTS <= DUE_TTS_SEC
		  - ETA within DUE_ETA_SEC of now
		  - currentLocation mentioning 'platform'
		"""
		tts = row.get("timeToStation")
		try:
			tts = int(tts)
		except Exception:
			tts = 10**9
		eta = parse_dt(row.get("expectedArrival"))
		cur_loc = (row.get("currentLocation") or "").lower()
		due_by_tts = (tts <= DUE_TTS_SEC)
		due_by_eta = (eta is not None and (eta.astimezone() - now_local).total_seconds() <= DUE_ETA_SEC)
		due_by_loc = ("platform" in cur_loc)
		return due_by_tts or due_by_eta or due_by_loc

	def _is_outbound(self, dest: str) -> bool:
		return canon(dest) != canon(self.A)

	def process(self, rows_at_A: List[Dict[str,Any]], rows_by_station: Dict[str, List[Dict[str,Any]]], now_local: datetime):
		"""
		One poll step for an impossible terminus A:
		
		  1) At A: track outbound trains that look 'due'.
		  2) For each tracked vid within UNK_CONFIRM_MIN:
			 - look for the same vid at the next station(s) beyond A,
			   'soon' there.
			 - if found, infer departure time at A's section/point and
			   write an 'impossible-terminus' event.
			 - if not found before timeout, drop it.
		"""
		# Step 1: seed outbound 'due' trains at A
		for it in rows_at_A:
			vid = str(it.get("vehicleId") or "").strip()
			if not vid:
				continue
			dest = (it.get("destinationName") or it.get("destination") or it.get("towards") or "").strip()
			if not self._is_outbound(dest):
				continue
			if not self._is_due(it, now_local):
				continue
			if vid not in self.outbound_due_vids:
				self.outbound_due_vids[vid] = {"t0": now_local, "dest": dest}
				_dbg(f"[imp/A-due] A={self.A} vid={vid} dest='{dest}' t0={hhmmss_local(now_local)}")

		# Step 2: for existing candidates, try to confirm at the next stations
		for vid, info in list(self.outbound_due_vids.items()):
			t0 = info["t0"] if isinstance(info, dict) else info
			dest_hint = (info.get("dest") if isinstance(info, dict) else None) or "UNKNOWN"
			if now_local - t0 > timedelta(minutes=UNK_CONFIRM_MIN):
				self.outbound_due_vids.pop(vid, None)
				_dbg(f"[imp/timeout] A={self.A} vid={vid}")
				continue

			confirmed_eta_local: datetime | None = None
			for st in self.next_stations_full:
				for it in rows_by_station.get(st, []):
					if str(it.get("vehicleId") or "") != vid:
						continue
					tts = int(it.get("timeToStation") or 1e9)
					if tts <= SOON_THRESHOLD_SEC:
						eta = parse_dt(it.get("expectedArrival"))
						if eta:
							confirmed_eta_local = eta.astimezone()
							break
				if confirmed_eta_local:
					break

			if confirmed_eta_local:
				dep_dt_local = confirmed_eta_local - timedelta(minutes=self.min_run_min)
				_dbg(f"[imp/write] A={self.A} vid={vid} dep={hhmmss_local(dep_dt_local)} next_eta={hhmmss_local(confirmed_eta_local)}")
				self.linker.record_terminus_departure_at_point(
					sec_id=self.sec_id,
					point_name=self.point_name,
					dep_dt_local=dep_dt_local,
					via="impossible-terminus",
					train_id=vid,
					dest=dest_hint
				)

				self.outbound_due_vids.pop(vid, None)

# =========== Terminus B Through Tracker ==========
# Check whether station name is a terminus
def _is_terminus_name(name: str, launch_dir_default: dict, min_run_map: dict) -> bool:
	n = name.strip()
	ns = strip_suffixes(n)
	return (n in launch_dir_default) or (ns in launch_dir_default) or (n in min_run_map) or (ns in min_run_map)

class TerminusBThroughTracker:
	"""
	Timer-based tracker for A→B movements where B itself is a terminus. B isn't required to be soon.
	
	- For each vid, keep the latest best (eta, tts) at A and at B.
    - Only consider trains where:
		  - A is "soon" (according to tts),
		  - A's sighting is recent (within MAX_LINK_MIN),
		  - etaB is at least BTERM_EPSILON_SEC later than etaA, but not more than MAX_LINK_MIN later.
	- When a match is seen for a vid, start or renew a short timer (ATTEMPT_WINDOW_MIN).
	- When the timer expires, write a single "b-terminus-through" departure at sec_id, point_name using etaA as the departure time, and clear the pending state for that vid.
	
	Key differences from classic through:
	  - B does NOT have to be "soon".
	  - Always matches using the latest known A and B times per vid.
	  
	* The path's purpose is to track trains arriving at terminus, because the arrival at terminus may disappear before it actually happens (i. e. train switched destination board prior to arriving).
	"""
	def __init__(self, *, sec_id: str, point_name: str, A_full: str, B_full: str, linker: PairLinker):
		self.sec_id = sec_id
		self.point_name = point_name
		self.A = A_full
		self.B = B_full
		self.linker = linker
	
		# Latest A/B per vid
		self.last_eta_at_A: Dict[str, datetime] = {}
		self.last_tts_at_A: Dict[str, int] = {}
		self.last_seen_at_A: Dict[str, datetime] = {}
	
		self.last_eta_at_B: Dict[str, datetime] = {}
		self.last_tts_at_B: Dict[str, int] = {}
		self.last_seen_at_B: Dict[str, datetime] = {}
	
		# Pending timer holds per vid
		# pending[vid] = "dep_dt_local": dt, "etaB_local": dt, "deadline": dt
		self.pending: Dict[str, Dict[str, Any]] = {}
	
	def _best_eta_per_vid(self, rows: List[Dict[str, Any]], now_local: datetime) -> Dict[str, Tuple[datetime, int]]:
		"""Return {vid: (best_eta_local, best_tts)} using smallest TTS row per vid."""
		best: Dict[str, Tuple[int, datetime]] = {}
		for it in rows:
			vid = str(it.get("vehicleId") or "")
			if not vid:
				continue
			try:
				tts = int(it.get("timeToStation") or 10**9)
			except Exception:
				tts = 10**9
			eta = parse_dt(it.get("expectedArrival"))
			eta_local = eta.astimezone() if isinstance(eta, datetime) else now_local
			prev = best.get(vid)
			if (prev is None) or (tts < prev[0]):
				best[vid] = (tts, eta_local)
		return {vid: (eta, tts) for vid, (tts, eta) in best.items()}
	
	def process(self, rows_by_station: Dict[str, List[Dict[str, Any]]], now_local: datetime):
		rows_A = rows_by_station.get(self.A, [])
		rows_B = rows_by_station.get(self.B, [])
	
		# Update A and B caches with the latest per loop
		A_best = self._best_eta_per_vid(rows_A, now_local)  # vid -> (etaA, ttsA)
		for vid, (etaA, ttsA) in A_best.items():
			self.last_eta_at_A[vid] = etaA
			self.last_tts_at_A[vid] = int(ttsA)
			self.last_seen_at_A[vid] = now_local
	
		B_best = self._best_eta_per_vid(rows_B, now_local)  # vid -> (etaB, ttsB)
		for vid, (etaB, ttsB) in B_best.items():
			self.last_eta_at_B[vid] = etaB
			self.last_tts_at_B[vid] = int(ttsB)
			self.last_seen_at_B[vid] = now_local
	
		# Matching logic: iterate vids that were at B this loop 
		epsilon_sec = max(10, BTERM_EPSILON_SEC)
		for vid, (etaB, _ttsB_unused) in B_best.items():
			etaA = self.last_eta_at_A.get(vid)
			ttsA = self.last_tts_at_A.get(vid)
			seenA = self.last_seen_at_A.get(vid)
			if not (etaA and seenA and isinstance(ttsA, int)):
				continue
	
			# A must be soon (start counting only when A is about to depart)
			if ttsA > SOON_THRESHOLD_SEC:
				continue
	
			# Avoid very old A
			if (now_local - seenA) > timedelta(minutes=MAX_LINK_MIN):
				continue
	
			# Epsilon ordering + overall link window
			delta = (etaB - etaA).total_seconds()
			if delta < epsilon_sec or delta > MAX_LINK_MIN * 60:
				continue
	
			# Start or refresh renewable timer
			dep_dt_local = etaA
			pend = self.pending.get(vid)
			if pend and now_local < pend["deadline"]:
				# Refresh deadline and keep later dep time if etaA advances
				if dep_dt_local > pend["dep_dt_local"]:
					pend["dep_dt_local"] = dep_dt_local
				pend["etaB_local"] = etaB
				pend["deadline"] = now_local + timedelta(minutes=ATTEMPT_WINDOW_MIN)
				_dbg(f"[bterm/refresh] A={self.A} B={self.B} vid={vid} dep(A)={hhmmss_local(dep_dt_local)} "
					 f"etaB={hhmmss_local(etaB)} (+{ATTEMPT_WINDOW_MIN}m)")
			else:
				self.pending[vid] = {
					"dep_dt_local": dep_dt_local,
					"etaB_local": etaB,
					"deadline": now_local + timedelta(minutes=ATTEMPT_WINDOW_MIN),
				}
				_dbg(f"[bterm/hold:new] A={self.A} B={self.B} vid={vid} dep(A)={hhmmss_local(dep_dt_local)} "
					 f"etaB={hhmmss_local(etaB)} -> wait {ATTEMPT_WINDOW_MIN}m")
	
		# Finalise expired holds, single XML write per vid
		for vid, hold in list(self.pending.items()):
			if now_local >= hold["deadline"]:
				dep_dt_local = hold["dep_dt_local"]
				self.linker.record_terminus_departure_at_point(
					sec_id=self.sec_id,
					point_name=self.point_name,
					dep_dt_local=dep_dt_local,
					via="b-terminus-through",
					train_id=vid,
					dest="UNKNOWN"
				)
				_dbg(f"[bterm/write] A={self.A} B={self.B} vid={vid} LEFT={hhmm_local(dep_dt_local)} (timer expired)")
				self.pending.pop(vid, None)

# =========== Variable Run Terminus A Tracker ==========
class VariableRunTerminusATracker:
	"""
	Tracker for A being a variable-run terminus (e.g. When time from A to B isn't constant, like stopping / fast service combination)

    - Ignore inbound rows whose destination is A itself (canon-equal).
	- Only treat a train as 'visible at A' if soon
	- When such a train disappears from the 'soon at A' set, start/renew an ATTEMPT_WINDOW_MIN timer; expiry -> A-valid for that vid.
	- The departure time we eventually write is the latest ETA(A) seen (no fixed min_run back-calc).
	- Timer based B logic also applied to B
	- A departure is emitted only when BOTH A-valid and B-valid are true for the same vid, written at sec_id, point_name with:
			via = "variable-run-terminus-A"
			dep_dt_local = latest ETA(A).
	* This path is less reliable than normal terminus tracker, since terminus departure data doesn't appear as prediction. 
	  Most of the predictions are for trains arriving to the terminus, trains that actually depart appear only for a few sec.
	"""
	
	def __init__(self, *, sec_id: str, point_name: str, A_full: str, B_full: str, linker: PairLinker):
		self.sec_id = sec_id
		self.point_name = point_name
		self.A = A_full
		self.B = B_full
		self.linker = linker
	
		# Per-vid last "best" seen at A / B while visible
		self.last_etaA: Dict[str, datetime] = {}
		self.last_ttsA: Dict[str, int] = {}
		self.last_destA: Dict[str, str] = {}
		self.visibleA: Set[str] = set()
	
		self.last_etaB: Dict[str, datetime] = {}
		self.last_ttsB: Dict[str, int] = {}
		self.visibleB: Set[str] = set()
	
		# Renewable timers and validity flags
		self.A_timers: Dict[str, Dict[str, Any]] = {}
		self.A_valid: Set[str] = set()
		self.B_timers: Dict[str, Dict[str, Any]] = {}
		self.B_valid: Set[str] = set()
	
		# Outer time budget to avoid ghosts
		self.A_first_seed_at: Dict[str, datetime] = {}
	
	def _best_eta_per_vid(self, rows: List[Dict[str, Any]], now_local: datetime) -> Dict[str, Tuple[datetime, int, str]]:
		"""Return {vid: (best_eta_local, best_tts, dest_raw)} using smallest TTS per vid."""
		best: Dict[str, Tuple[int, datetime, str]] = {}
		for it in rows:
			vid = str(it.get("vehicleId") or "")
			if not vid:
				continue
			try:
				tts = int(it.get("timeToStation") or 10**9)
			except Exception:
				tts = 10**9
			eta = parse_dt(it.get("expectedArrival"))
			eta_local = eta.astimezone() if isinstance(eta, datetime) else now_local
			dest_raw = (it.get("destinationName") or it.get("destination") or it.get("towards") or "").strip()
			prev = best.get(vid)
			if (prev is None) or (tts < prev[0]):
				best[vid] = (tts, eta_local, dest_raw)
		return {vid: (eta, tts, dest) for vid, (tts, eta, dest) in best.items()}
	
	def process(self, rows_by_station: Dict[str, List[Dict[str, Any]]], now_local: datetime):
		"""
		Process one polling step using arrivals at A (variable-run terminus) and B.
		
		At A:
		   - Update best ETA/TTS/dest per vid.
		   - Track which vids are currently "soon" at A.
		   - On disappearance from that "soon" set, start/renew an A timer.
		
		At B:
		   - Similarly track vids that are "soon" at B.
		   - On disappearance from B's "soon" set, start/renew a B timer, but only if the ETA(B) vs ETA(A) timing is within epsilon.
		
		Mark vids A-valid or B-valid when their timers expire.
		
		When a vid is both A-valid and B-valid, emit a 'variable-run-terminus-A' departure and clear its state.
		"""
		rows_A = rows_by_station.get(self.A, [])
		rows_B = rows_by_station.get(self.B, [])
	
		# Update A visibility and state
		A_best = self._best_eta_per_vid(rows_A, now_local)
		currA: Set[str] = set()
		for vid, (etaA, ttsA, destA) in A_best.items():
			# Gate inbound-to-terminus: skip rows whose destination is the terminus itself
			if canon(destA) == canon(self.A):
				continue
			self.last_etaA[vid] = etaA
			self.last_ttsA[vid] = int(ttsA)
			self.last_destA[vid] = destA
			# Visible at A only when tts below threshold
			if ttsA <= SOON_THRESHOLD_SEC:
				currA.add(vid)
	
		# A disappearances (only count those that were previously soon visible)
		disappearedA = self.visibleA - currA
		for vid in disappearedA:
			etaA = self.last_etaA.get(vid)
			ttsA = self.last_ttsA.get(vid, 10**9)
			if etaA and ttsA <= SOON_THRESHOLD_SEC:
				# Start/renew A timer, keep latest ETA_A as departure time
				self.A_timers[vid] = {
					"deadline": now_local + timedelta(minutes=ATTEMPT_WINDOW_MIN),
					"dep_dt_local": etaA,
				}
				self.A_first_seed_at.setdefault(vid, now_local)
				_dbg(f"[varA/A-miss] A={self.A} vid={vid} dep(A)={hhmmss_local(etaA)} -> +{ATTEMPT_WINDOW_MIN}m")
	
		# If a vid reappears soon at A, reset the A timer (train hasn't departed yet)
		for vid in currA:
			if vid in self.A_timers:
				self.A_timers[vid]["deadline"] = now_local + timedelta(minutes=ATTEMPT_WINDOW_MIN)
				# Optionally advance dep_dt_local if ETA improved
				etaA_now = self.last_etaA.get(vid)
				if isinstance(etaA_now, datetime) and etaA_now > self.A_timers[vid]["dep_dt_local"]:
					self.A_timers[vid]["dep_dt_local"] = etaA_now
				_dbg(f"[varA/A-renew] A={self.A} vid={vid} dep(A)=>{hhmmss_local(self.A_timers[vid]['dep_dt_local'])} (+{ATTEMPT_WINDOW_MIN}m)")
	
		# Expire A timers -> A-valid
		for vid, hold in list(self.A_timers.items()):
			if now_local >= hold["deadline"]:
				self.A_valid.add(vid)
				_dbg(f"[varA/A-valid] A={self.A} vid={vid} dep(A)={hhmmss_local(hold['dep_dt_local'])}")
				del self.A_timers[vid]
	
		# Update B visibility and state
		B_best = self._best_eta_per_vid(rows_B, now_local)
		currB: Set[str] = set()
		for vid, (etaB, ttsB, _destB) in B_best.items():
			self.last_etaB[vid] = etaB
			self.last_ttsB[vid] = int(ttsB)
			if ttsB <= SOON_THRESHOLD_SEC:
				currB.add(vid)
	
		# If a vid reappears soon at B, reset the B timer (train hasn't departed yet)
		disappearedB = self.visibleB - currB
		for vid in disappearedB:
			etaA = self.last_etaA.get(vid)
			etaB = self.last_etaB.get(vid)
			ttsB = self.last_ttsB.get(vid, 10**9)
	
			# Only use B if B was soon and departure in relation to A is within epsilon
			if etaA and etaB and (ttsB <= SOON_THRESHOLD_SEC):
				delta = (etaB - etaA).total_seconds()
				if (delta >= max(10, BTERM_EPSILON_SEC)) and (delta <= MAX_LINK_MIN * 60):
					self.B_timers[vid] = {
						"deadline": now_local + timedelta(minutes=ATTEMPT_WINDOW_MIN),
					}
					_dbg(f"[varA/B-miss] A={self.A} B={self.B} vid={vid} Δ={int(delta)}s -> +{ATTEMPT_WINDOW_MIN}m")
	
		# If a vid reappears soon at B, reset the B timer
		for vid in currB:
			if vid in self.B_timers:
				self.B_timers[vid]["deadline"] = now_local + timedelta(minutes=ATTEMPT_WINDOW_MIN)
				_dbg(f"[varA/B-renew] B={self.B} vid={vid} (+{ATTEMPT_WINDOW_MIN}m)")
	
		# Expire B timers -> B-valid
		for vid, hold in list(self.B_timers.items()):
			if now_local >= hold["deadline"]:
				self.B_valid.add(vid)
				_dbg(f"[varA/B-valid] B={self.B} vid={vid}")
				del self.B_timers[vid]
	
		# Drop old A timers that never validate at B
		for vid in list(self.A_first_seed_at.keys()):
			if now_local - self.A_first_seed_at[vid] > timedelta(minutes=MAX_LINK_MIN):
				if vid not in self.B_valid:
					_dbg(f"[varA/timeout] A→B window exceeded vid={vid}")
				del self.A_first_seed_at[vid]
				self.A_valid.discard(vid)
	
		# Finalise when both A and B are valid
		for vid in list(self.A_valid & self.B_valid):
			dep_dt_local = self.last_etaA.get(vid)
			if isinstance(dep_dt_local, datetime):
				dest_hint = strip_suffixes(self.last_destA.get(vid, "") or "UNKNOWN")
				self.linker.record_terminus_departure_at_point(
					sec_id=self.sec_id,
					point_name=self.point_name,
					dep_dt_local=dep_dt_local,
					via="variable-run-terminus-A",
					train_id=vid,
					dest=dest_hint if canon(dest_hint) != canon(self.A) else "UNKNOWN"
				)
				_dbg(f"[varA/write] A={self.A} B={self.B} vid={vid} LEFT={hhmm_local(dep_dt_local)}")
			# clear state for this vid to avoid duplicates
			self.A_valid.discard(vid); self.B_valid.discard(vid)
			self.A_first_seed_at.pop(vid, None)
	
		# Update visibility sets for next loop
		self.visibleA = currA
		self.visibleB = currB

# =========== Move One Terminus Tracker ==========
class MoveOneTerminusTracker:
	"""
	For some special cases it might be preferable to measure departure across A+1 and B+1 rather than A and B. In case of Highbury & Islington this 
	allows separating Windrush trains from Mildmay trains (line_key doesn't work correctly on the Overground), but might be useful in other cases.
	The logic with min_run_min is applied similarly to normal terminus departure.
	"""
	def __init__(self, *, sec_id: str, point_name: str,
				 A_full: str, launch_dir: str, wa: int,
				 min_run_min: int, value_to_display: Dict[int, List[str]],
				 xml_linker: PairLinker, station_suffix: str):
		self.sec_id = sec_id
		self.point_name = point_name
		self.A = A_full
		self.launch_dir = launch_dir
		self.wa = int(wa)
		self.min_run_min = int(min_run_min)
		self.linker = xml_linker
		self.station_suffix = station_suffix

		step = 1 if self.launch_dir == "up" else -1
		w1 = self.wa + step
		w2 = self.wa + 2*step

		if LINE_KEY == "elizabeth":
			self.S1_names: List[str] = expand_with_suffixes((value_to_display.get(w1) or []), self.station_suffix)
			self.S2_names: List[str] = expand_with_suffixes((value_to_display.get(w2) or []), self.station_suffix)
		else:
			self.S1_names: List[str] = [f"{nm}{self.station_suffix}" for nm in (value_to_display.get(w1) or [])]
			self.S2_names: List[str] = [f"{nm}{self.station_suffix}" for nm in (value_to_display.get(w2) or [])]

		self.prev_S1_vids: Set[str] = set()
		self.last_exp_at_S1: Dict[str, datetime] = {}
		self.timers: Dict[str, Dict[str, Any]] = {}

	def _best_exp_by_vid_multi(self, rows_lists: List[List[Dict[str, Any]]], fallback_now: datetime) -> Dict[str, datetime]:
		best: Dict[str, Tuple[int, datetime]] = {}
		for rows in rows_lists:
			for it in rows:
				vid = str(it.get("vehicleId") or "")
				if not vid:
					continue
				try:
					tts = int(it.get("timeToStation") or 10**9)
				except Exception:
					tts = 10**9
				eta = parse_dt(it.get("expectedArrival"))
				exp_local = eta.astimezone() if isinstance(eta, datetime) else fallback_now
				prev = best.get(vid)
				if (prev is None) or (tts < prev[0]):
					best[vid] = (tts, exp_local)
		return {vid: exp for vid, (_, exp) in best.items()}

	def process(self, rows_by_station: Dict[str, List[Dict[str, Any]]], now_local: datetime):
		rows_S1_lists = [rows_by_station.get(nm, []) for nm in self.S1_names]
		rows_S2_lists = [rows_by_station.get(nm, []) for nm in self.S2_names]

		curr_S2_vids: Set[str] = set()
		for rows in rows_S2_lists:
			for it in rows:
				v = str(it.get("vehicleId") or "")
				if v:
					curr_S2_vids.add(v)

		expS1_by_vid = self._best_exp_by_vid_multi(rows_S1_lists, now_local)
		curr_S1_vids: Set[str] = set(expS1_by_vid.keys())

		for vid, exp_local in expS1_by_vid.items():
			self.last_exp_at_S1[vid] = exp_local
			if vid in self.timers:
				_dbg(f"[move1/cancel@S1] A={self.A} vid={vid} (re-appeared at S1) cancel previous timer")
				self.timers.pop(vid, None)

		disappeared = self.prev_S1_vids - curr_S1_vids
		for vid in disappeared:
			exp_local = self.last_exp_at_S1.get(vid, now_local)
			dep_dt_local = exp_local - timedelta(minutes=self.min_run_min)
			seen_at_S2_now = (vid in curr_S2_vids)

			if seen_at_S2_now:
				self.timers[vid] = {
					"deadline": now_local + timedelta(minutes=ATTEMPT_WINDOW_MIN),
					"dep_dt_local": dep_dt_local,
				}
				_dbg(f"[move1/start] A={self.A} vid={vid} dep(S1-min_run)={hhmmss_local(dep_dt_local)} -> +{ATTEMPT_WINDOW_MIN}m (S2 present now)")
			else:
				_dbg(f"[move1/skip]  A={self.A} vid={vid} dep(S1-min_run)={hhmmss_local(dep_dt_local)} (S2 NOT present now)")

		for vid, hold in list(self.timers.items()):
			if now_local >= hold["deadline"]:
				if vid not in curr_S1_vids:  # still gone
					dep_dt_local = hold["dep_dt_local"]
					self.linker.record_terminus_departure_at_point(
						sec_id=self.sec_id,
						point_name=self.point_name,
						dep_dt_local=hold["dep_dt_local"],
						via="move-one-terminus",
						train_id=vid,
						dest="UNKNOWN"
					)


				else:
					_dbg(f"[move1/drop] A={self.A} vid={vid} (came back to S1 before deadline)")
				self.timers.pop(vid, None)

		self.prev_S1_vids = curr_S1_vids

# ===================== BUILD ALL TERMINUS TRACKERS =====================
def build_all_terminus_trackers(*, sections: dict, weights: dict, launch_dir_default: dict, min_run_map: dict, value_to_display: Dict[int,List[str]], linker: PairLinker, station_suffix: str, is_shuttle: bool):
	"""
	Build all terminus-related trackers for the configured sections.
	
	For each section, point pair A-B, this function decides which
	tracker to attach based on configuration and special-case lists
	
	  - ShuttleABTracker (currently not in use).
	  - ImpossibleTerminusTracker (A is an impossible terminus).
	  - MoveOneTerminusTracker (A is in MOVE_ONE_TERMINI).
	  - VariableRunTerminusATracker (A is a variable-run terminus).
	  - TerminusBThroughTracker (B is a terminus, train running towards the terminus).
	  - TerminusTracker and UnknownAtTerminusAWatcher (normal A-terminus path and Unknown-at-A if destination is unknown).
	
	It also returns
	  - forbid: set of sec_id, point where classic through-pair linking will be disabled.
	  - extra_watch: station names that need to be included in the polling loop because some trackers confirm departures by watching neighbours.
	"""

	# Weights for cannon names
	def w(name: str) -> int | None:
		return weights.get(canon(name))

	# Lists of tracker types and station sets
	shuttle_trackers: List[ShuttleABTracker] = []
	normal_by_B: Dict[str, List[TerminusTracker]] = defaultdict(list)
	unknownA_by_A: Dict[str, List[UnknownAtTerminusAWatcher]] = defaultdict(list)
	impossible_by_A: Dict[str, List[ImpossibleTerminusTracker]] = defaultdict(list)
	bterm_through_trackers: List[TerminusBThroughTracker] = []
	move_one_trackers: List[MoveOneTerminusTracker] = []
	variable_run_trackers: List[VariableRunTerminusATracker] = []
	forbid: Set[Tuple[str,str]] = set()
	extra_watch: Set[str] = set()

	for sec_id, sec in sections.items():
		for p_name, pair in (sec.get("points") or {}).items():
			if not (isinstance(pair, (list, tuple)) and len(pair) == 2):
				continue
			A, B = str(pair[0]).strip(), str(pair[1]).strip()
		
			# SHUTTLE MODE: Build for shuttle lines, currently not in use
			if is_shuttle:
				sh = ShuttleABTracker(
					sec_id=sec_id,
					point_name=p_name,
					A_full=A,
					B_full=B,
					linker=linker,
				)
				shuttle_trackers.append(sh)
				forbid.add((sec_id, p_name))  # block through on shuttles
				continue  # skip all other tracker types for shuttles

			# Resolve launch and weights
			launch = (launch_dir_default.get(A) or launch_dir_default.get(strip_suffixes(A)) or "").strip().lower()
			if launch not in ("up","down"):
				pass

			wa, wb = w(A), w(B)
			dir_ab = None
			if (wa is not None) and (wb is not None) and (wa != wb):
				dir_ab = "up" if wb > wa else "down"

			# 1) Impossible A
			if A in IMPOSSIBLE_TERMINI:
				if wa is None:
					print(f"[terminus-skip] missing weight for A={A!r}")
					continue
				launchA = (launch_dir_default.get(A) or launch_dir_default.get(strip_suffixes(A)) or "").strip().lower()
				min_run = int(min_run_map.get(A) or min_run_map.get(strip_suffixes(A)) or 2)
				itr = ImpossibleTerminusTracker(
					sec_id=sec_id, point_name=p_name,
					A_full=A, launch_dir=launchA, wa=wa,
					min_run_min=min_run, value_to_display=value_to_display,
					xml_linker=linker, station_suffix=station_suffix
				)
				impossible_by_A[A].append(itr)
				forbid.add((sec_id, p_name))
				for nm in itr.next_stations_full:
					extra_watch.add(nm)
				continue

			# 2) MOVE-ONE
			if (strip_suffixes(A) in MOVE_ONE_TERMINI) and (dir_ab is not None) and (dir_ab == launch):
				if wa is None:
					print(f"[terminus-skip] missing weight for A={A!r}")
					continue
				min_run = int(min_run_map.get(A) or min_run_map.get(strip_suffixes(A)) or 2)
				m1 = MoveOneTerminusTracker(
					sec_id=sec_id, point_name=p_name,
					A_full=A, launch_dir=launch, wa=wa,
					min_run_min=min_run, value_to_display=value_to_display,
					xml_linker=linker, station_suffix=station_suffix
				)
				move_one_trackers.append(m1)
				forbid.add((sec_id, p_name))
				for nm in m1.S1_names + m1.S2_names:
					extra_watch.add(nm)
				continue
			
			# 2b) VARIABLE-RUN TERMINUS
			is_varA = (strip_suffixes(A) in VARIABLE_RUN_TERMINI)
			A_launch = (launch_dir_default.get(A) or launch_dir_default.get(strip_suffixes(A)) or "").strip().lower()
			
			if is_varA:
				# Need weights
				if w(A) is None or w(B) is None:
					print(f"[terminus-skip] missing weight for variable-run pair A={A!r} B={B!r}")
				else:
					# dir_ab already computed above from wa, wb
					# Only activate variable-run when A's configured launch direction matches the pair direction
					if (A_launch in ("up","down")) and (dir_ab == A_launch):
						vr = VariableRunTerminusATracker(
							sec_id=sec_id, point_name=p_name,
							A_full=A, B_full=B, linker=linker
						)
						variable_run_trackers.append(vr)
						forbid.add((sec_id, p_name)) 
						continue
					else:
						# A is variable-run but not in its launch direction relative to B:
						# Act like a normal station at A
						pass



			# 3) B-terminus-through
			def w_name(name: str) -> int | None:
				return weights.get(canon(name))
			B_is_term = _is_terminus_name(B, launch_dir_default, min_run_map)
			A_is_term = _is_terminus_name(A, launch_dir_default, min_run_map)
			A_launch  = (launch_dir_default.get(A) or launch_dir_default.get(strip_suffixes(A)) or "").strip().lower()
			dir_ab2 = None
			wa2, wb2 = w_name(A), w_name(B)
			if (wa2 is not None) and (wb2 is not None) and (wa2 != wb2):
				dir_ab2 = "up" if wb2 > wa2 else "down"
			A_towards_B = A_is_term and (A_launch in ("up","down")) and (dir_ab2 == A_launch)
			if B_is_term and (not A_towards_B) and (dir_ab2 is not None):
				bt = TerminusBThroughTracker(sec_id=sec_id, point_name=p_name, A_full=A, B_full=B, linker=linker)
				bterm_through_trackers.append(bt)
				forbid.add((sec_id, p_name))

			# 4) Normal A-terminus path
			if launch not in ("up","down"):
				continue
			if wa is None:
				print(f"[terminus-skip] missing weight for A={A!r}")
				continue

			if (wb is not None) and (dir_ab == launch):
				min_run = int(min_run_map.get(A) or min_run_map.get(strip_suffixes(A)) or 2)
				tr = TerminusTracker(
					sec_id=sec_id, point_name=p_name,
					A_full=A, B_full=B, B_weight=wb,
					launch_dir=launch, min_run_min=min_run,
					weights=weights, value_to_display=value_to_display,
					xml_linker=linker, station_suffix=station_suffix
				)
				normal_by_B[B].append(tr)
				forbid.add((sec_id, p_name))
				for nm in tr.neighbor_stations_full:
					extra_watch.add(nm)

				ua = UnknownAtTerminusAWatcher(
					sec_id=sec_id, point_name=p_name, A_full=A, launch_dir=launch, wa=wa,
					value_to_display=value_to_display, xml_linker=linker, station_suffix=station_suffix
				)
				unknownA_by_A[A].append(ua)
				for nm in ua.next_stations_full:
					extra_watch.add(nm)

	_dbg(
		f"[terminus-build] normal={sum(len(v) for v in normal_by_B.values())} "
		f"unknownA={sum(len(v) for v in unknownA_by_A.values())} "
		f"impossible={sum(len(v) for v in impossible_by_A.values())} "
		f"b-terminus-through={len(bterm_through_trackers)} "
		f"move-one={len(move_one_trackers)} "
		f"variable-run-A={len(variable_run_trackers)} "
		f"shuttle={len(shuttle_trackers)}"
	)


	return (
		normal_by_B, unknownA_by_A, impossible_by_A,
		bterm_through_trackers, move_one_trackers,
		variable_run_trackers, shuttle_trackers,
		forbid, extra_watch
	)


# ===================== MAIN =====================
def main():
	# Load configs
	line, sections = load_line_config(LINE_KEY)
	line_name = (line.get("name") or LINE_KEY).strip()
	line_id   = (line.get("id")   or line.get("line_id") or line_name or LINE_KEY).strip().lower()
	is_shuttle = (line_id in {s.lower() for s in SHUTTLE_SERVICE})
	
	API_LINE_ID_MAP = {
		"waterloo": "waterloo-city",   # Do not use, works incorrectly
	}
	api_line_id = API_LINE_ID_MAP.get(line_id, line_id)
	mode      = (line.get("mode") or "tube").strip().lower()

	# Choose suffix and search modes per mode
	if mode == "elizabeth-line":
		station_suffix = " (Elizabeth line)"
		search_modes = ["elizabeth-line"]
	elif mode == "overground":
		station_suffix = " Rail Station"
		search_modes = ["overground"]
	elif mode == "dlr":
		station_suffix = " DLR Station"           # DLR doesn't work, vid issue
		search_modes = ["dlr"]
	else:
		station_suffix = " Underground Station"   # Met, H&C, District, Circle - all don't work, vid issue
		search_modes = ["tube"]

	weights, launch_dir_default, min_run_map, value_to_display = load_routes(LINE_KEY)

	print(f"[init] line={line_name} (id={line_id}) mode={mode}")

	# Build watchlist of station names (all A and B)
	station_names: Set[str] = set()
	for sec in sections.values():
		for p_name, pair in (sec.get("points") or {}).items():
			if isinstance(pair, (list, tuple)) and len(pair) == 2:
				station_names.add(str(pair[0]).strip())
				station_names.add(str(pair[1]).strip())	
		# Expand Elizabeth line station names to try multiple suffixes when resolving StopPoints
		if mode == "elizabeth-line":
			def _expand_elizabeth_candidates(names: set[str]) -> set[str]:
				out = set()
				for nm in names:
					base = strip_suffixes(nm)
					out.add(f"{base} (Elizabeth line)")
					out.add(f"{base} Rail Station")
					out.add(base)
				return out
			station_names = _expand_elizabeth_candidates(station_names)

	_dbg(f"[watchlist] {len(station_names)} station names")

	cache_dir = os.path.join(PROJECT_DIR, "cached-stops")
	os.makedirs(cache_dir, exist_ok=True)
	cache_path = os.path.join(cache_dir, f"{line_id}.yaml")
	
	# populate name_to_ids either from cache or by resolving
	name_to_ids: Dict[str, List[str]] = {}
	cache_loaded = False
	
	try:
		cached = load_yaml(cache_path)
		if isinstance(cached, dict):
			cached_mode = (cached.get("mode") or "").strip().lower()
			cached_line = (cached.get("api_line_id") or "").strip().lower()
			if cached_mode == mode and cached_line == api_line_id:
				nti = cached.get("name_to_ids") or {}
				name_to_ids = {str(k): [str(x) for x in (v or [])] for k, v in nti.items()}
				cache_loaded = True
	except Exception as e:
		print(f"[cache] load failed ({cache_path}): {e}")
	
	# TfL session still used for polling
	api = TfLSess(
		TFL_APP_KEY,
		http_timeout=HTTP_TIMEOUT_S,
		throttle_sec=THROTTLE_S,
		default_modes=search_modes,
		preferred_suffix=station_suffix,
	)
	
	def _save_cache():
		try:
			out = {
				"mode": mode,
				"api_line_id": api_line_id,
				"created": datetime.now(timezone.utc).isoformat(),
				"name_to_ids": name_to_ids,
			}
			with open(cache_path, "w", encoding="utf-8") as f:
				yaml.safe_dump(out, f, sort_keys=True, allow_unicode=True)
			print(f"[cache] wrote {cache_path} (stations={len(name_to_ids)})")
		except Exception as e:
			print(f"[cache] write failed ({cache_path}): {e}")
	
	required_names: Set[str] = set(station_names)
	# If cache is missing any of the required base names, resolve just the missing ones.
	missing_now = sorted(n for n in required_names if n not in name_to_ids)
	
	if missing_now:
		# resolve only what’s missing
		if LINE_KEY == "elizabeth" and mode == "elizabeth-line":
			# strict resolver for Elizabeth line
			def _canon_for_match(s: str) -> str:
				return canon(strip_suffixes(s or ""))
	
			def _stop_parent_detail(api, parent_id: str) -> dict:
				params = {"app_key": TFL_APP_KEY} if TFL_APP_KEY else {}
				url = f"{BASE}/StopPoint/{requests.utils.quote(parent_id)}"
				r = api.sess.get(url, params=params, timeout=(api.connect_timeout, api.read_timeout))
				if r.status_code == 404:
					return {}
				r.raise_for_status()
				return r.json() or {}
	
			def _parent_matches_station(base_name: str, parent_detail: dict) -> bool:
				common = (parent_detail.get("commonName") or parent_detail.get("name") or "")
				return _canon_for_match(common) == _canon_for_match(base_name)
	
			def _parent_or_children_serve_line(parent_detail: dict, target_line_id: str) -> bool:
				target = (target_line_id or "").lower()
				def _has(j):
					for ln in (j.get("lines") or []):
						if (ln.get("id") or "").lower() == target:
							return True
					return False
				if _has(parent_detail):
					return True
				for ch in (parent_detail.get("children") or []):
					if _has(ch):
						return True
				return False
	
			for base_nm in missing_now:
				try:
					parent_id = api.search_stop_id_one(base_nm, modes=search_modes)
					detail = _stop_parent_detail(api, parent_id)
					if not detail:
						continue
					if not _parent_matches_station(base_nm, detail):
						_dbg(f"[resolve/skip] parent '{detail.get('commonName','?')}' != base '{base_nm}'")
						continue
					if not _parent_or_children_serve_line(detail, api_line_id):
						_dbg(f"[resolve/skip] no '{api_line_id}' among parent/children lines for '{base_nm}'")
						continue
					ids = {parent_id}
					for ch in (detail.get("children") or []):
						cid = ch.get("id")
						if cid:
							ids.add(cid)
					name_to_ids.setdefault(base_nm, []).extend(sorted(ids))
				except Exception as e:
					print(f"[warn] resolve(elizabeth:{base_nm}): {e}")
					name_to_ids.setdefault(base_nm, [])
		else:
			# generic resolver for all other lines
			for nm in missing_now:
				try:
					ids = api.search_stop_ids_all(nm, modes=search_modes)
					name_to_ids.setdefault(nm, []).extend(ids)
				except Exception as e:
					print(f"[warn] resolve_stop_ids({nm}): {e}")
					name_to_ids.setdefault(nm, [])
	
		_save_cache()
	
	# shared cache of last known destinations per vehicle
	veh_last_dest: Dict[str, str] = {}

	# Linker (forbid_through set after building trackers)
	linker = PairLinker(
		line_name=line_name,
		mode=mode,
		sections=sections,
		out_root=RUNS_DIR,
		max_gap_min=MAX_LINK_MIN,
		forbid_through_points=set(),
		weights=weights,
		veh_last_dest=veh_last_dest,
	)

	# Build trackers
	(normal_by_B, unknownA_by_A, impossible_by_A,
	 bterm_through_trackers, move_one_trackers,
	 variable_run_trackers, shuttle_trackers,
	 forbid_pairs, extra_watch) = build_all_terminus_trackers(
		sections=sections,
		weights=weights,
		launch_dir_default=launch_dir_default,
		min_run_map=min_run_map,
		value_to_display=value_to_display,
		linker=linker,
		station_suffix=station_suffix,
		is_shuttle=is_shuttle

	linker.forbid_through = set(forbid_pairs)
	
	# Add neighbours required to watch if they aren't present
	newly_resolved = False
	for nm in sorted(extra_watch):
		if nm not in name_to_ids:
			try:
				ids = api.search_stop_ids_all(nm, modes=search_modes)
				name_to_ids.setdefault(nm, []).extend(ids)
				newly_resolved = True
			except Exception as e:
				print(f"[warn] resolve_stop_ids({nm}): {e}")
				name_to_ids.setdefault(nm, [])
	if newly_resolved:
		_save_cache()


	watched_ids: Set[str] = set()
	nid_to_name: Dict[str,str] = {}
	for nm, ids in name_to_ids.items():
		for nid in ids:
			watched_ids.add(nid)
			nid_to_name[nid] = nm

	print(f"[watch] stations={len(name_to_ids)} ids={len(watched_ids)}")
	os.makedirs(RUNS_DIR, exist_ok=True)

	if is_shuttle:
		print(f"[shuttle] trackers={len(shuttle_trackers)}; through forbidden on their points")
	elif normal_by_B or unknownA_by_A or impossible_by_A or bterm_through_trackers or move_one_trackers or variable_run_trackers:
		print(
			f"[terminus] trackers: "
			f"normal={sum(len(v) for v in normal_by_B.values())}  "
			f"unknown-at-A={sum(len(v) for v in unknownA_by_A.values())}  "
			f"impossible={sum(len(v) for v in impossible_by_A.values())}  "
			f"b-terminus-through={len(bterm_through_trackers)}  "
			f"move-one={len(move_one_trackers)}  "
			f"variable-run-A={len(variable_run_trackers)}; "
			f"through forbidden on their points"
		)
	else:
		print("[terminus] no matching trackers from config")

	tracker = PassTracker(nid_to_name=nid_to_name)
	sess = api.sess
	connect_t = (api.connect_timeout, api.read_timeout)

	print("[loop] polling arrivals … Ctrl+C to stop")
	while True:
		try:
			params_req = {"app_key": TFL_APP_KEY} if TFL_APP_KEY else {}
			url = f"{BASE}/Line/{requests.utils.quote(api_line_id)}/Arrivals"
			r = sess.get(url, params=params_req, timeout=connect_t)
			rows = [] if r.status_code == 404 else (r.json() or [])
			now_local = datetime.now(timezone.utc).astimezone()
			# Normalise and filter to watched station ids (child + parent)
			filt = []
			rows_by_station: Dict[str, List[Dict[str,Any]]] = defaultdict(list)
			rows_for_B: Dict[str, List[Dict[str,Any]]] = defaultdict(list)
			rows_for_A_unknown: Dict[str, List[Dict[str,Any]]] = defaultdict(list)
			rows_for_A_impossible: Dict[str, List[Dict[str,Any]]] = defaultdict(list)

			for it in rows:
				nid = str(it.get("naptanId") or "")
				if nid not in watched_ids:
					continue
				station_name = nid_to_name.get(nid, it.get("stationName",""))
				row_min = {
					"lineId":          it.get("lineId") or line_id,
					"vehicleId":       it.get("vehicleId"),
					"naptanId":        nid,
					"platformName":    it.get("platformName") or it.get("platformNameId") or it.get("direction"),
					"direction":       it.get("direction"),
					"stationName":     station_name,
					"expectedArrival": it.get("expectedArrival"),
					"timeToStation":   it.get("timeToStation"),
					"destinationName": it.get("destinationName") or it.get("destination"),
					"towards":         it.get("towards"),
					"currentLocation": it.get("currentLocation"),
				}
				filt.append(row_min)
				rows_by_station[station_name].append(row_min)

				# update last-known destination cache
				vid = str(row_min["vehicleId"] or "")
				dest_raw = (row_min["destinationName"] or row_min["towards"] or "").strip()
				if vid and dest_raw and not _is_unknown_label(dest_raw):
					veh_last_dest[vid] = dest_raw

				# DEBUG: print every train we see at tracked stations
				if DEBUG:
					eta_local = parse_dt(row_min["expectedArrival"])
					_dbg(
						f"[arr] st={station_name} dir={row_min['platformName']} "
						f"vid={row_min['vehicleId']} tts={row_min['timeToStation']} "
						f"eta={hhmmss_local(eta_local)} dest='{row_min['destinationName']}' "
						f"towards='{row_min['towards']}' loc='{(row_min['currentLocation'] or '').lower()}'"
					)

				# feed to normal trackers at measuring B
				if station_name in normal_by_B:
					rows_for_B[station_name].append({
						"stationName": station_name,
						"vehicleId": row_min["vehicleId"],
						"timeToStation": row_min["timeToStation"],
						"expectedArrival": row_min["expectedArrival"],
						"destinationName": row_min["destinationName"],
						"towards": row_min["towards"],
					})

				# feed unknown-at-A watchers
				if station_name in unknownA_by_A:
					rows_for_A_unknown[station_name].append(row_min)

				# feed impossible trackers at the terminus A
				if station_name in impossible_by_A:
					rows_for_A_impossible[station_name].append(row_min)

			# through logic, generate PASS/APPEAR events
			events = tracker.update(filt)
			for ev in events:
				linker.observe_event(ev)

			# finalise through holds whose window expired
			linker.tick(now_local)

			# normal terminus trackers
			for B, trackers in normal_by_B.items():
				rows_at_B = rows_for_B.get(B, [])
				for tr in trackers:
					tr.process_rows_at_B(rows_at_B, now_local, rows_by_station)

			# unknown-at-A
			for A, watchers in unknownA_by_A.items():
				rows_at_A = rows_for_A_unknown.get(A, [])
				for ua in watchers:
					ua.process_rows_at_A(rows_at_A, now_local, rows_by_station)

			# impossible terminus trackers
			for A, trackers in impossible_by_A.items():
				rows_at_A = rows_for_A_impossible.get(A, [])
				for tr in trackers:
					tr.process(rows_at_A, rows_by_station, now_local)

			# B-terminus through trackers
			for bt in bterm_through_trackers:
				bt.process(rows_by_station, now_local)

			# MOVE-ONE trackers
			for m1 in move_one_trackers:
				m1.process(rows_by_station, now_local)
				
			# VARIABLE-RUN A trackers
			for vr in variable_run_trackers:
				vr.process(rows_by_station, now_local)
				
			# SHUTTLE trackers
			for sh in shuttle_trackers:
				sh.process(rows_by_station, now_local)



		except KeyboardInterrupt:
			print("\n[stop] interrupted by user.")
			break
		except requests.exceptions.RequestException as e:
			print(f"[net] {e}")
		except Exception as e:
			print(f"[err] {e}")

		time.sleep(POLL_SEC)

if __name__ == "__main__":
	main()
