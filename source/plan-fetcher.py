#!/usr/bin/env python3

import os, sys, re, io, gzip, time, csv, random
from datetime import datetime, timedelta, date
import xml.etree.ElementTree as ET
from xml.dom import minidom

import requests
import yaml
from dotenv import load_dotenv, find_dotenv
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from pathlib import Path
THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[1] if THIS_FILE.parent.name == "source" else THIS_FILE.parent

# Darwin dependencies
try:
	import boto3
except Exception:
	boto3 = None

BASE = "https://api.tfl.gov.uk"

# Cross-line exclusion cache to identify subsurface lines
_CIRCLE_EXCLUDE_CACHE: dict[str, dict[str, dict[tuple[str,str], set[str]]]] = {}

def _cache_put_all(date_ymd: str, pair: tuple[str,str], times_all: list[str]):
	day = _CIRCLE_EXCLUDE_CACHE.setdefault(date_ymd, {})
	all_bucket = day.setdefault("ALL", {})
	all_bucket.setdefault(pair, set()).update(times_all)

def _cache_put_window(date_ymd: str, window_key: str, pair: tuple[str,str], window_times: list[str]):
	day = _CIRCLE_EXCLUDE_CACHE.setdefault(date_ymd, {})
	wbucket = day.setdefault(window_key, {})
	wbucket.setdefault(pair, set()).update(window_times)

def _cache_has_window(date_ymd: str, window_key: str) -> bool:
	return date_ymd in _CIRCLE_EXCLUDE_CACHE and window_key in _CIRCLE_EXCLUDE_CACHE[date_ymd]

def _cache_get_all(date_ymd: str, pair: tuple[str,str]) -> set[str]:
	return (_CIRCLE_EXCLUDE_CACHE.get(date_ymd, {}).get("ALL", {}) or {}).get(pair, set())

def _cache_clear_day(date_ymd: str):
	_CIRCLE_EXCLUDE_CACHE.pop(date_ymd, None)

# TIPLOC map for lines with DARWIN timetable
TIPLOC_BY_NAME = {
	# Elizabeth line
	"Reading Rail Station": {"RDNGSTN"}, "Maidenhead Rail Station": {"MDNHEAD"}, "West Drayton Rail Station": {"WDRYTON"}, "Ealing Broadway Rail Station": {"EALINGB"},
	"Burnham Rail Station": {"BNHAM"}, "Heathrow Terminals 2 & 3 Rail Station": {"HTRWAPT"}, "Heathrow Terminal 4 Rail Station": {"HTRWTM4"}, "Heathrow Terminal 5 Rail Station": {"HTRWTM5"},
	"Hayes & Harlington Rail Station": {"HAYESAH"}, "West Ealing Rail Station": {"WEALING"}, "Paddington Rail Station": {"PADTON","PADTLL"}, "Bond Street Rail Station": {"BONDST"},
	"Farringdon Rail Station": {"FRNDXR"}, "Liverpool Street Rail Station": {"LIVERST","LIVST","LIVSTLL"}, "Stratford Rail Station": {"STFD","STRATFD","STRTFRD"}, "Romford Rail Station": {"ROMFORD"},
	"Gidea Park Rail Station": {"GIDEAPK"}, "Harold Wood Rail Station": {"HRLDWOD"}, "Brentwood Rail Station": {"BRTWOOD"}, "Shenfield Rail Station": {"SHENFLD"},
	"Canary Wharf Rail Station": {"CANWHRF"}, "Woolwich Rail Station": {"WOLWXR"}, "Abbey Wood Rail Station": {"ABWDXR"},
	# Suffragette line
	"Gospel Oak Rail Station": {"GOSPLOK"}, "Upper Holloway Rail Station": {"UPRHLWY"}, "Leytonstone High Road Rail Station": {"LYTNSHR"}, "Wanstead Park Rail Station": {"WNSTDPK"},
	"Barking Rail Station": {"BARKING"}, "Barking Riverside Rail Station": {"BARKRIV"},
	# Lioness line
	"Watford Junction Rail Station": {"WATFDJ","WATFDJL", "WATFJDC"}, "Watford High Street Rail Station": {"WATFDHS","WATFDHL"}, "Headstone Lane Rail Station": {"HEDSTNL"}, "Harrow & Wealdstone Rail Station": {"HROW", "HROWDC"},
	"Kenton Rail Station": {"KTON"}, "Wembley Central Rail Station": {"WMBY", "WMBYDC"}, "Stonebridge Park Rail Station": {"STNBGPK"}, "Harlesden Rail Station": {"HARLSDN"},
	"Kensal Green Rail Station": {"KENSLG"}, "Queen's Park Rail Station": {"QPRK"}, "Kilburn High Road Rail Station": {"KLBRNHR"}, "South Hampstead Rail Station": {"STHAMPD","STHMPST", "SHMPSTD"},
	"Euston Rail Station": {"EUSTON","EUSTONLL","EUSTNLL"},
	# Weaver line
	"Liverpool Street Rail Station": {"LIVERST", "LIVST","LIVSTLL"}, "Hackney Downs Rail Station": {"HACKNYD","HAKNYNM"}, "Clapton Rail Station": {"CLAPTON"}, "Highams Park Rail Station": {"HGHMSPK", "HIGHMPK"},
	"Chingford Rail Station": {"CHINGFD"}, "Rectory Road Rail Station": {"RCTRYRD","RECTRYR"}, "Silver Street Rail Station": {"SILVST","SLVRST","SIVRST"}, "Edmonton Green Rail Station": {"EDMONGN","EDMNGRN"},
	"Southbury Rail Station": {"SBURY"}, "Theobalds Grove Rail Station": {"THBLDSG"}, "Cheshunt Rail Station": {"CHESHNT"}, "Enfield Town Rail Station": {"ENFLDTN"},
	# Mildmay line
	"Stratford Rail Station": {"STFD","STRATFD","STRTFRD"}, "Caledonian Road Rail Station": {"CLDNNRB"}, "Camden Road Rail Station": {"CMDNRD"}, "Hackney Wick Rail Station": {"HACKNYW"},
	"Kensal Rise Rail Station": {"KENR"}, "Willesden Junction Rail Station": {"WLSDJHL"}, "Kentish Town West Rail Station": {"KNTSHTW"}, "Imperial Wharf Rail Station": {"CSEAH"},
	"Clapham Junction Rail Station": {"CLPHMJ1","CLPHMJC","CLPHMJM","CLPHMJN","CLPHMJW"}, "Shepherd's Bush Rail Station": {"SHPDSB"}, "South Acton Rail Station": {"SACTON"},
	"Kew Gardens Rail Station": {"KEWGRDN"}, "Richmond Rail Station": {"RICHNLL"}, "Gunnersbury Rail Station": {"GNRSBRY"},
	# Windrush line
	"Highbury & Islington Rail Station": {"HIGHBYE"}, "Dalston Junction Rail Station": {"DALS"}, "Canada Water Rail Station": {"CNDAW"}, "Surrey Quays Rail Station": {"SURREYQ"},
	"Haggerston Rail Station": {"HAGGERS"}, "New Cross Rail Station": {"NWCRELL","NWCROSS"}, "New Cross Gate Rail Station": {"NEWXGEL","NEWXGTE"}, "Sydenham Rail Station": {"SYDENHM"},
	"Forest Hill Rail Station": {"FORESTH"}, "Wandsworth Road Rail Station": {"WNDSWRD"}, "Clapham Junction Rail Station": {"CLPHMJ1","CLPHMJC","CLPHMJM","CLPHMJN","CLPHMJW"},
	"Queens Road Peckham Rail Station": {"PCKHMQD"}, "Crystal Palace Rail Station": {"CRYSTLP"}, "Norwood Junction Rail Station": {"NORWDJ"}, "West Croydon Rail Station": {"WCROYDN"},
	"Penge West Rail Station": {"PENEW"},
	# Liberty line
	"Romford Rail Station": {"ROMFORD"}, "Upminster Rail Station": {"UPMNSP6","UPMNSTR"},
}

# ----- Common utils -----

def load_key():
	load_dotenv(find_dotenv(usecwd=True))
	k = os.getenv("TFL_APP_KEY", "").strip()
	if not k:
		raise SystemExit("Missing TFL_APP_KEY in environment or .env")
	return k

def parse_iso(ts: str):
	"""Parse an ISO-8601 timestamp string into a datetime"""
	if not ts: return None
	ts = ts.replace("Z","+00:00")
	try: return datetime.fromisoformat(ts)
	except Exception: return None

def hhmm_add(hhmm: str, minutes: int) -> str:
	t = datetime.strptime(hhmm, "%H%M") + timedelta(minutes=minutes)
	return t.strftime("%H%M")

def base_place(nm: str) -> str:
	s = (nm or "").strip()
	for suf in (" (Elizabeth line)", " Rail Station", " Underground Station", " DLR Station", " Station"):
		if s.endswith(suf):
			s = s[: -len(suf)]
			break
	return s.strip()

def headways_from_times(times: list[str], start_hhmm: str, end_hhmm: str) -> list[int]:
	"""Compute minute gaps inside [start,end) using preroll as the anchor"""
	if not times: return []
	start_hm = f"{start_hhmm[:2]}:{start_hhmm[2:]}"
	end_hm   = f"{end_hhmm[:2]}:{end_hhmm[2:]}"

	prev = None
	within = []
	for hm in times:
		if hm < start_hm:
			prev = hm
		elif start_hm <= hm < end_hm:
			within.append(hm)
	if not within: return []

	anchor = datetime(2000,1,1)
	def to_dt(hm): return anchor.replace(hour=int(hm[:2]), minute=int(hm[3:]), second=0)

	gaps = []
	last_dt = to_dt(prev) if prev else None
	for hm in within:
		cur = to_dt(hm)
		if last_dt is not None:
			gaps.append(int(round((cur - last_dt).total_seconds()/60)))
		last_dt = cur
	return gaps

def hm_to_min(hhmm: str) -> int:
	return int(hhmm[:2]) * 60 + int(hhmm[2:])

def window_overlaps(start_hhmm: str, end_hhmm: str, a_hhmm: str, b_hhmm: str) -> bool:
	"""True if [start,end) overlaps [a,b)."""
	s = hm_to_min(start_hhmm); e = hm_to_min(end_hhmm)
	a = hm_to_min(a_hhmm);    b = hm_to_min(b_hhmm)
	return not (e <= a or s >= b)

# ----- TfL session -----

class TfLSess:
	def __init__(self, app_key: str, jp_mode: str, http_timeout: int, throttle_sec: float):
		self.sess = requests.Session()
		self.key = app_key
		self.jp_mode = jp_mode
		self.http_timeout = http_timeout
		self.throttle_sec = throttle_sec
		self.sess.headers.update({"Accept": "application/json"})
		self._search_cache_one = {}   d
		self._resolve_cache_multi = {} 
		self._jp_cache = {}
		self._stoppoint_cache = {}

		retry = Retry(
			total=5,
			connect=3,
			read=3,
			backoff_factor=0.6,
			status_forcelist=(429, 500, 502, 503, 504),
			allowed_methods=frozenset(['GET', 'HEAD']),
			raise_on_status=False,
		)
		adapter = HTTPAdapter(
			max_retries=retry,
			pool_connections=50,
			pool_maxsize=50,
			pool_block=True,
		)
		self.sess.mount("https://", adapter)
		self.sess.mount("http://", adapter)

		self.connect_timeout = min(5, max(2, int(self.http_timeout // 3)))
		self.read_timeout = max(20, int(self.http_timeout * 2))

	def sleep(self):
		if self.throttle_sec > 0:
			time.sleep(self.throttle_sec + random.uniform(0, 0.15))

	def _search_stop_id_one(self, name: str) -> str:
		if name in self._search_cache_one:
			return self._search_cache_one[name]

		def try_query(qname: str, modes_param):
			url = f"{BASE}/StopPoint/Search/{requests.utils.quote(qname)}"
			params = {"maxResults": 24, "app_key": self.key}
			if modes_param:
				params["modes"] = ",".join(modes_param)
			r = self.sess.get(url, params=params, timeout=(self.connect_timeout, self.read_timeout))
			if r.status_code == 404:
				return []
			r.raise_for_status()
			data = r.json()
			return data.get("matches") or []

		bp = base_place(name)
		variants = [name]
		if name.endswith(" Rail Station"):
			variants.append(f"{bp} Station")
			variants.append(bp)
		elif name.endswith(" Station"):
			variants.append(bp)

		mode_fallbacks = {
			"overground":     ["overground", "national-rail"],
			"elizabeth-line": ["elizabeth-line", "national-rail"],
			"dlr":            ["dlr"],
			"tube":           ["tube"],
		}
		modes_to_try = mode_fallbacks.get(self.jp_mode, [self.jp_mode])

		matches = []
		for v in variants:
			matches = try_query(v, modes_to_try)
			if matches: break
		if not matches:
			for v in variants:
				matches = try_query(v, None)
				if matches: break
		if not matches:
			raise RuntimeError(f"Stop not found for name: {name}")

		def score(m):
			nm = (m.get("name") or "")
			s = 0
			if nm == name: s += 1000
			if base_place(nm).lower() == bp.lower(): s += 120
			if name.lower() in nm.lower(): s += 80
			if nm.endswith(" Underground Station"): s -= 100
			return s

		best = max(matches, key=score)
		best_id = best.get("id")
		if not best_id:
			raise RuntimeError(f"Stop not found for name: {name}")

		self._search_cache_one[name] = best_id
		self.sleep()
		return best_id

	def resolve_stop_ids(self, name: str) -> list[str]:
		if name in self._resolve_cache_multi:
			return self._resolve_cache_multi[name][:]
		ids = [ self._search_stop_id_one(name) ]
		self._resolve_cache_multi[name] = ids[:]
		return ids

	def stoppoint_detail(self, sp_id: str) -> dict:
		"""Cached StopPoint detail (to discover child platform IDs etc.)."""
		if sp_id in self._stoppoint_cache:
			return self._stoppoint_cache[sp_id]
		url = f"{BASE}/StopPoint/{sp_id}"
		r = self.sess.get(url, params={"app_key": self.key},
						  timeout=(self.connect_timeout, self.read_timeout))
		r.raise_for_status()
		data = r.json()
		self._stoppoint_cache[sp_id] = data
		self.sleep()
		return data

	def resolve_platform_ids_for_line(self, station_id: str, line_id: str) -> list[str]:
		"""
		Return child platform StopPoint IDs at this station that belong to `line_id`.
		Falls back to the parent station if none found.
		"""
		data = self.stoppoint_detail(station_id)
		plats = []

		def child_has_line(child: dict) -> bool:
			for li in (child.get("lines") or []):
				cid = (li.get("id") or "").lower()
				if cid == line_id.lower():
					return True
			for lg in (child.get("lineModeGroups") or []):
				for cid in (lg.get("lineIdentifier") or []):
					if (cid or "").lower() == line_id.lower():
						return True
			for lg in (child.get("lineGroup") or []):
				for cid in (lg.get("lineIdentifier") or []):
					if (cid or "").lower() == line_id.lower():
						return True
			return False

		for child in (data.get("children") or []):
			if (child.get("stopType") or "").lower() not in ("naptanmetroplatform", "naptanmetrostationplatform"):
				continue
			if child_has_line(child):
				cid = child.get("id")
				if cid:
					plats.append(cid)

		return plats or [station_id]

	def jp(self, from_id: str, to_id: str, date_ymd: str, hhmm: str) -> dict:
		key = (from_id, to_id, date_ymd, hhmm)
		if key in self._jp_cache:
			return self._jp_cache[key]
		url = f"{BASE}/Journey/JourneyResults/{from_id}/to/{to_id}"
		params = {
			"mode": self.jp_mode,
			"timeIs": "Departing",
			"date": date_ymd,
			"time": hhmm,
			"journeyPreference": "LeastTime",
			"app_key": self.key,
		}
		try:
			r = self.sess.get(url, params=params, timeout=(self.connect_timeout, self.read_timeout))
			data = {"journeys": []} if r.status_code == 404 else r.json()
		except requests.exceptions.ReadTimeout:
			print(f"[timeout] JP {from_id}->{to_id} {date_ymd}T{hhmm} (read>{self.read_timeout}s)")
			data = {"journeys": []}
		except requests.exceptions.RequestException as e:
			print(f"[warn] JP error {from_id}->{to_id} {date_ymd}T{hhmm}: {e}")
			data = {"journeys": []}
		self._jp_cache[key] = data
		self.sleep()
		return data

# ----- JP extraction -----

def extract_line_times(payload, line_token: str) -> set[str]:
	import re
	def _norm(s: str) -> str:
		return re.sub(r'[^a-z0-9]+', '', (s or '').lower())

	# allow-list tokens
	if isinstance(line_token, (list, tuple, set)):
		tokens = [str(t) for t in line_token]
	else:
		tokens = [str(line_token or '')]

	key = _norm(tokens[0]) if tokens else ""
	if key == "hammersmithcity":
		tokens = ["hammersmith-city", "hammersmith & city", "hammersmith and city", "hammersmith & city line", "hammersmithcity"]
	elif key == "circle":
		tokens = ["circle", "circle line"]
	elif key == "metropolitan":
		tokens = ["metropolitan", "metropolitan line"]
	elif key == "district":
		tokens = ["district", "district line"]
	allow_norm = {_norm(t) for t in tokens if t}

	blacklist_map = {
		"circle": {"hammersmithcity", "district", "hammersmith-city", "hammersmith & city", "hammersmith and city", "hammersmith & city line", "metropolitan"},
		"hammersmithcity": {"circle", "metropolitan"},
		"metropolitan": {"circle", "hammersmithcity", "hammersmith-city", "hammersmith & city", "hammersmith and city", "hammersmith & city line","jubilee"},
		"district": {"circle", "hammersmithcity", "hammersmith-city", "hammersmith & city", "hammersmith and city", "hammersmith & city line"},
	}
	deny_norm = blacklist_map.get(key, set())

	def broad_hay(leg: dict) -> str:
		parts = []
		parts.append(leg.get("lineName") or "")
		lid = leg.get("lineId")
		if isinstance(lid, str):
			parts.append(lid)
		mode = leg.get("mode") or {}
		if isinstance(mode, dict):
			parts.append(mode.get("name") or "")
			parts.append(mode.get("id") or "")
		for ro in (leg.get("routeOptions") or []):
			if isinstance(ro, dict):
				parts.append(ro.get("name") or "")
				for d in (ro.get("directions") or []):
					parts.append(d or "")
				li = ro.get("lineIdentifier") or {}
				if isinstance(li, dict):
					parts.append(li.get("id") or "")
					parts.append(li.get("name") or "")
		return _norm(" ".join(p for p in parts if isinstance(p, str)))

	def primary_only(leg: dict) -> str:
		parts = []
		parts.append(leg.get("lineName") or "")
		lid = leg.get("lineId")
		if isinstance(lid, str):
			parts.append(lid)
		return _norm(" ".join(p for p in parts if isinstance(p, str)))

	out = set()
	for j in (payload.get("journeys") or []):
		legs = j.get("legs") or []
		if len(legs) != 1:
			continue

		leg = legs[0]

		hay_allow = broad_hay(leg)
		if allow_norm and not any(t in hay_allow for t in allow_norm):
			continue

		hay_primary = primary_only(leg)
		if deny_norm and any(b in hay_primary for b in deny_norm):
			continue

		sched = leg.get("scheduledDepartureTime")
		if not sched:
			continue

		dt = parse_iso(sched)
		if dt:
			out.add(dt.strftime("%H:%M"))

	return out

# ----- Darwin helper functions -----

def _env(name, *alts, default=None, required=False):
	for n in (name, *alts):
		v = os.getenv(n)
		if v and str(v).strip():
			return str(v).strip()
	if required and default is None:
		raise SystemExit(f"Missing env var {name} (or {'/'.join(alts)})")
	return default

def _mask_tail(s: str, keep=4):
	if not s: return ""
	return ("*"*(len(s)-keep)) + s[-keep:]

def s3_latest_object(s3, bucket: str, prefix: str):
	token = None
	latest = None
	while True:
		kw = {"Bucket": bucket, "Prefix": prefix}
		if token: kw["ContinuationToken"] = token
		resp = s3.list_objects_v2(**kw)
		for obj in resp.get("Contents", []) or []:
			if latest is None or obj["LastModified"] > latest["LastModified"]:
				latest = obj
		if not resp.get("IsTruncated"): break
		token = resp.get("NextContinuationToken")
	if not latest:
		raise RuntimeError(f"No objects under s3://{bucket}/{prefix}")
	return latest["Key"]

def s3_download_bytes(s3, bucket: str, key: str) -> bytes:
	return s3.get_object(Bucket=bucket, Key=key)["Body"].read()

def maybe_gunzip(key: str, blob: bytes) -> bytes:
	return gzip.GzipFile(fileobj=io.BytesIO(blob)).read() if key.lower().endswith(".gz") else blob

# Darwin timetable parsing

def _norm_yyyymmdd(s: str | None) -> str | None:
	if not s: return None
	s = s.strip().replace("-", "")
	if re.fullmatch(r"\d{8}", s):
		return s
	return None

def _weekday_mask_ok(days_val: str | None, target_weekday_m0: int) -> bool:
	"""
	Accepts either a 7-char '0/1' mask (Mon..Sun) e.g. '1111100'
	or a letters form like 'MTWTFSS'. If missing/unknown -> True (don't exclude).
	"""
	if not days_val:
		return True
	dv = str(days_val).strip().upper()
	# binary mask
	if re.fullmatch(r"[01]{7}", dv):
		return dv[target_weekday_m0] == "1"
	# letters mask: allow common variants (M T W T F S S)
	letters = ["M","T","W","T","F","S","S"]
	if len(dv) == 7 and all(ch.isalpha() for ch in dv):
		# treat any non-dash as runs
		return dv[target_weekday_m0] != "-"
	# unknown format -> don't exclude
	return True

# collect calendar data from Journey element and descendants

def _collect_calendar(el) -> dict:
	"""Harvest calendar info from Journey attributes and child elements.
	Returns dict with: 'ssd','e','days','include_dates','exclude_dates','overlay'"""
	cal = {'ssd': None, 'e': None, 'days': None,
		   'include_dates': set(), 'exclude_dates': set(), 'overlay': None}

	def grab_attr(d):
		for k in ('ssd','startDate','start','StartDate'):
			if d.get(k): cal['ssd'] = d[k].replace('-', '')
		for k in ('e','endDate','end','EndDate'):
			if d.get(k): cal['e'] = d[k].replace('-', '')
		for k in ('days','dow','Days','DOW','daysofweek','DaysOfWeek'):
			if d.get(k): cal['days'] = str(d[k]).strip()
		for k in ('overlay','Overlay','plan','Plan','stpInd','STPInd'):
			if d.get(k):
				v = str(d[k]).strip().upper()
				if v in ('STP','SHORT-TERM','Q','V'): cal['overlay'] = 'STP'
				elif v in ('WTT','BASE','P','N','O'): cal['overlay'] = 'WTT'

	grab_attr(el.attrib)

	def _normdate(s):
		s = (s or '').replace('-', '')
		return s if len(s) == 8 and s.isdigit() else None

	for child in el.iter():
		d = child.attrib or {}
		if not d: continue
		tag = child.tag.split('}')[-1].lower()
		grab_attr(d)

		for k in ('runsOn','include','dates','runDates','add'):
			if k in d:
				for piece in re.split(r'[\s,]+', d[k]):
					nd = _normdate(piece)
					if nd: cal['include_dates'].add(nd)
		for k in ('doesNotRun','exclude','excludeDates','notDates','cancelDates','remove'):
			if k in d:
				for piece in re.split(r'[\s,]+', d[k]):
					nd = _normdate(piece)
					if nd: cal['exclude_dates'].add(nd)

		if 'date' in d:
			nd = _normdate(d['date'])
			if nd:
				if tag in ('runson','include','add','run'):
					cal['include_dates'].add(nd)
				elif tag in ('doesnotrun','exclude','remove','cancel'):
					cal['exclude_dates'].add(nd)

	return cal


def _runs_on_date(cal: dict, date_ymd: str) -> bool:
	return _runs_on_date_with_mode(cal, date_ymd, mode="calendar")

def _runs_on_date_with_mode(cal: dict, date_ymd: str, mode: str = "calendar") -> bool:
	ssd = _norm_yyyymmdd(cal.get('ssd'))
	eed = _norm_yyyymmdd(cal.get('e'))
	days = cal.get('days')
	include = set(cal.get('include_dates') or [])
	exclude = set(cal.get('exclude_dates') or [])

	# Start/end are always enforced
	if ssd and date_ymd < ssd: return False
	if eed and date_ymd > eed: return False

	if mode == "dow_only":
		try:
			w = datetime.strptime(date_ymd, "%Y%m%d").weekday()
		except Exception:
			return False
		return _weekday_mask_ok(days, w)

	if date_ymd in exclude:    return False
	if include:                return (date_ymd in include)

	if not any([ssd, eed, days, include, exclude]):
		return False

	if days is not None:
		try:
			w = datetime.strptime(date_ymd, "%Y%m%d").weekday()
		except Exception:
			return False
		return _weekday_mask_ok(days, w)

	return True

def parse_calls_from_xml_bytes(xml_bytes: bytes, allowed_tocs: set[str], date_ymd: str | None, *, filter_mode: str = "calendar"):
	for ev, el in ET.iterparse(io.BytesIO(xml_bytes), events=("end",)):
		if el.tag.split("}")[-1] != "Journey":
			continue
		a = el.attrib
		toc = (a.get("toc") or "").upper()
		if toc not in allowed_tocs or not is_passenger(a):
			el.clear(); continue

		cal = _collect_calendar(el)
		if date_ymd and not _runs_on_date_with_mode(cal, date_ymd, mode=filter_mode):
			el.clear(); continue

		def calls():
			out = []
			for c in el:
				tag = c.tag.split("}")[-1]
				if tag not in ("OR","IP","DT"):
					continue
				aa = c.attrib
				tpl = aa.get("tpl") or aa.get("tiploc") or ""
				if not tpl: continue
				pta, ptd = aa.get("pta",""), aa.get("ptd","")
				time_used = ""
				if ptd:
					time_used = ptd
				elif tag in ("OR","IP") and pta:
					time_used = pta
				out.append({"tag":tag,"tiploc":tpl,"pta":pta,"ptd":ptd,"time_used":time_used})
			return out

		rid = a.get("rid","")
		uid = a.get("uid","") or a.get("UID","")
		overlay = cal.get('overlay') or 'WTT'
		yield rid, uid, calls(), overlay
		el.clear()


def is_passenger(attrs: dict) -> bool:
	if (attrs.get("isPassengerSvc","").lower() == "false"): return False
	if (attrs.get("status") or "").upper() == "S": return False
	if (attrs.get("trainCat") or "").upper() == "EE": return False
	return True

def times_from_darwin_for_pairs(section_pairs, tiploc_map, allowed_tocs: set[str], *, date_ymd: str | None, filter_mode: str = "calendar"):
	load_dotenv(find_dotenv(usecwd=True))
	bucket = _env("S3_BUCKET","NR_S3_BUCKET", required=True)
	prefix = _env("S3_OBJECT_PREFIX","NR_S3_PREFIX", required=True)
	region = _env("REGION","S3_REGION", default="eu-west-1")
	ak = _env("ACCESS_KEY","AWS_ACCESS_KEY_ID", required=True)
	sk = _env("SECRET_KEY","AWS_SECRET_ACCESS_KEY", required=True)

	if boto3 is None:
		raise SystemExit("boto3 is required for Darwin S3")

	print(f"[Darwin] S3 bucket={bucket} prefix={prefix} region={region} key={_mask_tail(ak)} secret={_mask_tail(sk)}")
	s3 = boto3.client("s3", region_name=region,
					  aws_access_key_id=ak, aws_secret_access_key=sk)

	key = s3_latest_object(s3, bucket, prefix)
	print(f"[Darwin] Latest object: s3://{bucket}/{key}")
	xml_bytes = maybe_gunzip(key, s3_download_bytes(s3, bucket, key))
	print(f"[Darwin] Snapshot bytes: {len(xml_bytes):,}")

	want_pairs = list(section_pairs)
	res = {p: set() for p in want_pairs}

	# Precompute tiploc sets
	pair_tiplocs = []
	for fr, to in want_pairs:
		fr_set = set(tiploc_map.get(fr, set()))
		to_set = set(tiploc_map.get(to, set()))
		pair_tiplocs.append((fr, to, fr_set, to_set))

	# First pass: load journeys and keep best overlay per UID
	journeys_by_uid: dict[str, tuple[str, list[dict]]] = {}
	for rid, uid, calls, overlay in parse_calls_from_xml_bytes(xml_bytes, allowed_tocs, date_ymd, filter_mode=filter_mode):
		key_uid = uid or rid  # fallback to RID if UID missing
		prev = journeys_by_uid.get(key_uid)
		if prev is None or (prev[0] != 'STP' and overlay == 'STP'):
			journeys_by_uid[key_uid] = (overlay, calls)

	print(f"[Darwin] Journeys kept after STP precedence: {len(journeys_by_uid):,}")

	# Second step: for each kept journey, add times for any matching (from,to) pair
	for overlay, calls in journeys_by_uid.values():
		if not calls: continue
		seq = [c["tiploc"] for c in calls]
		# Cache index per TIPLOC occurrence
		index_list = {}
		for i, tpl in enumerate(seq):
			index_list.setdefault(tpl, []).append(i)

		for (fr_name, to_name, fr_set, to_set) in pair_tiplocs:
			if not fr_set or not to_set:
				continue
			fr_idx_list = [i for tpl in fr_set for i in index_list.get(tpl, [])]
			if not fr_idx_list: continue
			to_idx_list = [i for tpl in to_set for i in index_list.get(tpl, [])]
			if not to_idx_list: continue

			ok = False
			chosen_idx = None
			for fi in fr_idx_list:
				for ti in to_idx_list:
					if fi < ti:
						ok = True
						chosen_idx = fi
						break
				if ok: break
			if not ok: continue

			dep = calls[chosen_idx].get("time_used","")
			if dep and ":" in dep:
				res[(fr_name,to_name)].add(dep[:5]) 

	return {k: sorted(v) for k, v in res.items()}

# XML writer
def write_hour_xml(filename: str, date_ymd: str, window, line_name: str, mode: str, sections: dict, results: dict):
	root = ET.Element("PlannedReference", attrib={
		"date": f"{date_ymd[:4]}-{date_ymd[4:6]}-{date_ymd[6:]}",
		"line": line_name,
		"mode": mode,
		"tz": "Europe/London",
		"window": f"{window[0][:2]}:{window[0][2:]}-{window[1][:2]}:{window[1][2:]}"
	})

	for section_id, sec in sections.items():
		sec_el = ET.SubElement(root, "Section")
		sec_el.text = section_id
		for point_name, _pair in sec.get("points", {}).items():
			tph, headways = results[section_id].get(point_name, (0, []))
			p_el = ET.SubElement(sec_el, "Point")
			p_el.text = point_name
			t_el = ET.SubElement(p_el, "Time")
			t_el.text = f"{window[0][:2]}:{window[0][2:]}-{window[1][:2]}:{window[1][2:]}"
			tph_el = ET.SubElement(p_el, "TPH")
			tph_el.text = str(tph)
			h_el = ET.SubElement(p_el, "Headways")
			h_el.text = " ".join(str(x) for x in headways)

	xml = minidom.parseString(ET.tostring(root, encoding="utf-8")).toprettyxml(
		indent="   ", encoding="utf-8"
	).decode()
	with open(filename, "w", encoding="utf-8") as f:
		f.write(xml)

# Config loader 
def load_line_config(line_key: str, line_config_dir: str | None = None):
	"""
	Load a line config YAML and normalise sections to:
		{section_id: {"points": {...}, [step_min], [preroll_min], [n_step_min], [n_preroll_min], ...}}
	"""
	if line_config_dir is None:
		line_config_dir = str(PROJECT_ROOT / "line_config")

	cfg_path = os.path.join(line_config_dir, f"{line_key}.yaml")
	if not os.path.exists(cfg_path):
		raise SystemExit(f"Config not found: {cfg_path}")

	with open(cfg_path, "r", encoding="utf-8") as f:
		cfg = yaml.safe_load(f)

	line = cfg["line"]
	params = cfg["parameters"]
	sections_in = cfg["sections"]

	# Normalise sections
	sections = {}
	if isinstance(sections_in, list):
		for sec in sections_in:
			sid = sec["id"]
			sections[sid] = {"points": sec["points"], **{k: v for k, v in sec.items() if k not in ("id", "points")}}
	else:
		for sid, sec in sections_in.items():
			if isinstance(sec, dict):
				if "points" in sec:
					pts = sec["points"]
					sec_params = {k: v for k, v in sec.items() if k != "points"}
					sections[sid] = {"points": pts, **sec_params}
				else:
					sections[sid] = {"points": sec}
			else:
				sections[sid] = {"points": sec}

	return line, params, sections

# PUBLIC ENTRYPOINT
def collect_hour_times_multi_jp(sess, line_token, from_ids, to_ids, DATE_YMD, start, end, sec_step_min, sec_preroll_min):
	"""Walks time in steps with preroll and extracts single-leg times."""
	if not from_ids or not to_ids:
		return []

	# build sample times with preroll
	start_dt = datetime.strptime(DATE_YMD + start, "%Y%m%d%H%M")
	end_dt = datetime.strptime(DATE_YMD + end, "%Y%m%d%H%M")
	first_dt = start_dt - timedelta(minutes=sec_preroll_min)
	cur = first_dt
	seen = set()
	out = set()
	while cur < end_dt:
		cur_hhmm = cur.strftime("%H%M")
		for fid in from_ids:
			for tid in to_ids:
				payload = sess.jp(fid, tid, DATE_YMD, cur_hhmm)
				for hm in extract_line_times(payload, line_token):
					out.add(hm)
		cur += timedelta(minutes=sec_step_min)
	return sorted(out)


def generate_planned_sections(
	LINE_KEY: str,
	DATE_YMD: str,
	WINDOWS: list[tuple[str, str]],
	*,
	out_dir: str | None = None,
	line_config_dir: str | None = None,
):
	out_dir = out_dir or str(PROJECT_ROOT / "test-gen-timetable")
	line_config_dir = line_config_dir or str(PROJECT_ROOT / "line_config")
	os.makedirs(out_dir, exist_ok=True)

	# Load cfg
	line, params, sections = load_line_config(LINE_KEY, line_config_dir=line_config_dir)

	line_name   = line.get("name", LINE_KEY)
	mode        = line.get("mode", "tube")
	line_token  = (line.get("line_token") or line_name).lower()

	# Line-level defaults
	step_min       = int(params.get("step_min", 5))
	preroll_min    = int(params.get("preroll_min", 20))
	throttle_sec   = float(params.get("throttle_sec", 0.15))
	http_timeout   = int(params.get("http_timeout", 12))
	jp_mode        = params.get("jp_mode", mode)
	night_tube     = int(params.get("night_tube", 0))  # 1 enables Night Tube sampling on Sat/Sun 02:00–05:00

	print(
		f"Loaded config: line={line_name} mode={mode} jp_mode={jp_mode} "
		f"default_step={step_min}m default_preroll={preroll_min}m night_tube={night_tube}"
	)

	# Session
	tfl_key = load_key()
	sess = TfLSess(tfl_key, jp_mode=jp_mode, http_timeout=http_timeout, throttle_sec=throttle_sec)

	# Resolve StopPoint ids
	all_names = []
	for sec in sections.values():
		for pair in sec.get("points", {}).values():
			if isinstance(pair, (list, tuple)) and len(pair) == 2:
				all_names += list(pair)

	name_to_ids = {}
	for nm in sorted(set(all_names)):
		try:
			name_to_ids[nm] = sess.resolve_stop_ids(nm)
		except Exception as e:
			print(f"[warn] resolve_stop_ids({nm}): {e}")
			name_to_ids[nm] = []
	total_ids = sum(len(v) for v in name_to_ids.values())
	print(f"Resolved {len(name_to_ids)} stations → {total_ids} StopPoints.")


	# Darwin vs JP
	use_darwin = jp_mode in ("elizabeth-line", "overground")
	allowed_tocs = {"XR"} if jp_mode == "elizabeth-line" else ({"LO"} if jp_mode == "overground" else set())

	# Darwin filter mode: from YAML parameters or env override
	darwin_filter = str(os.getenv("DARWIN_FILTER", "dow_only").lower())
	if darwin_filter not in ("calendar", "dow_only"):
		print(f"[warn] Unknown darwin_filter='{darwin_filter}', defaulting to 'calendar'.")
		darwin_filter = "calendar"

	darwin_times_by_pair = {}
	if use_darwin:
		pairs_needed = []
		for sec in sections.values():
			for pair in sec.get("points", {}).values():
				if isinstance(pair, (list, tuple)) and len(pair) == 2:
					pairs_needed.append(tuple(pair))

		missing = [nm for nm in set([p for pr in pairs_needed for p in pr]) if nm not in TIPLOC_BY_NAME]
		if missing:
			print("[warn] Missing TIPLOC mapping for:", ", ".join(sorted(missing)))
			print("       Add them to TIPLOC_BY_NAME for accurate matching.")

		darwin_times_by_pair = times_from_darwin_for_pairs(
			pairs_needed, TIPLOC_BY_NAME, allowed_tocs, date_ymd=DATE_YMD, filter_mode=darwin_filter
		)

	date_dt = datetime.strptime(DATE_YMD, "%Y%m%d")
	weekday = date_dt.weekday()  # 0=Mon ... 6=Sun

	all_windows_results = {}

	for win in WINDOWS:
		start, end = win
		window_key = f"{start}-{end}"
		results = {}
		print(f"\n=== Window {start[:2]}:{start[2:]}–{end[:2]}:{end[2:]} ===")

		night_overlap = window_overlaps(start, end, "0200", "0500")
		apply_night_rules = (jp_mode == "tube") and night_overlap

		# Decide night policy
		night_policy = "normal"
		if apply_night_rules:
			if weekday <= 4:
				night_policy = "ignore_all"  # Mon–Fri between 02:00–05:00 -> zero TPH
			else:
				if night_tube != 1:
					night_policy = "ignore_all"
				else:
					night_policy = "night_params"

		for section_id, sec in sections.items():
			pts = sec.get("points", {})
			results[section_id] = {}

			# Determine step/preroll for this section & window
			if night_policy == "ignore_all":
				# Skip entirely in night windows
				for point_name in pts.keys():
					results[section_id][point_name] = (0, [])
				print(
					f"  {section_id}: Night window on "
					f"{'Mon–Fri' if weekday<=4 else 'weekend with night_tube=0'} → all points skipped (TPH=0)."
				)
				continue

			if night_policy == "night_params":
				if "n_step_min" in sec and "n_preroll_min" in sec:
					sec_step_min = int(sec["n_step_min"])
					sec_preroll_min = int(sec["n_preroll_min"])
				else:
					for point_name in pts.keys():
						results[section_id][point_name] = (0, [])
					print(f"  {section_id}: night_tube active but missing n_step_min/n_preroll_min → skipped (TPH=0).")
					continue
			else:
				sec_step_min = int(sec.get("step_min", step_min))
				sec_preroll_min = int(sec.get("preroll_min", preroll_min))

			# Compute per point
			for point_name, pair in pts.items():
				from_nm, to_nm = pair

				if use_darwin:
					times = darwin_times_by_pair.get((from_nm, to_nm), [])
				else:
					from_ids = name_to_ids.get(from_nm, [])
					to_ids   = name_to_ids.get(to_nm, [])
					times = collect_hour_times_multi_jp(
						sess, line_token, from_ids, to_ids,
						DATE_YMD, start, end,
						sec_step_min, sec_preroll_min
					)

				# Cross-line exclusion logic
				if not use_darwin:
					# Record H&C and District departures into cache (ALL + window)
					if LINE_KEY in ("hammersmith", "district"):
						# Cache ALL-times and window-times for this pair
						_cache_put_all(DATE_YMD, (from_nm, to_nm), times)
						start_hm = f"{start[:2]}:{start[2:]}"; end_hm = f"{end[:2]}:{end[2:]}"
						window_times = [hm for hm in times if start_hm <= hm < end_hm]
						_cache_put_window(DATE_YMD, window_key, (from_nm, to_nm), window_times)

					# When running Circle, enforce cache presence and exclude cached times
					if LINE_KEY == "circle":
						if not _cache_has_window(DATE_YMD, window_key):
							raise RuntimeError(
								f"[circle] Missing cross-line cache for {DATE_YMD} {window_key}. "
								"Run hammersmith and district for the same date/time window first."
							)
						ex_all = set()
						# union of ALL-times from H&C and District
						ex_all |= _cache_get_all(DATE_YMD, (from_nm, to_nm))
						if ex_all:
							# drop all matching times so headway anchoring is correct
							times = [t for t in times if t not in ex_all]

				start_hm = f"{start[:2]}:{start[2:]}"
				end_hm   = f"{end[:2]}:{end[2:]}"
				tph = sum(1 for hm in times if start_hm <= hm < end_hm)
				headways = headways_from_times(times, start, end)
				results[section_id][point_name] = (tph, headways)

				print(f"  {section_id} • {point_name}: step={sec_step_min} preroll={sec_preroll_min} → TPH={tph} Headways={headways}")

		# write XML per window
		outname = os.path.join(out_dir, f"{LINE_KEY}_{start[:2]}-{end[:2]}.xml")
		write_hour_xml(outname, DATE_YMD, win, line_name, mode, sections, results)
		print(f"Wrote {outname}")

		all_windows_results[f"{start}-{end}"] = results

	# After Circle is done for this date, clear cache
	if not use_darwin and LINE_KEY == "circle":
		_cache_clear_day(DATE_YMD)
		print(f"[circle] Cleared cross-line cache for {DATE_YMD}")

	return all_windows_results

# ----- CLI wrapper -----

if __name__ == "__main__":
	# Defaults when running as a script
	DEFAULT_DATE_YMD = "20250919"
	DEFAULT_WINDOWS = [("1600","1700"), ("1700","1800"), ("1800","1900")]
	DEFAULT_LINE_KEY = "district" if len(sys.argv) < 2 else sys.argv[1].strip()

	generate_planned_sections(
		DEFAULT_LINE_KEY,
		DEFAULT_DATE_YMD,
		DEFAULT_WINDOWS,
		out_dir=os.path.join(PROJECT_ROOT, "test-gen-timetable"),
		line_config_dir=str(PROJECT_ROOT / "line_config"),
	)
