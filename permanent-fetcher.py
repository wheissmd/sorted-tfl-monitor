#!/usr/bin/env python3

from __future__ import annotations
import sys, os, time, json, importlib.util, traceback
from datetime import datetime, timedelta, date
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional
import xml.etree.ElementTree as ET

# ---------- Paths & dynamic import ----------
PROJECT_ROOT = Path(__file__).resolve().parent
SOURCE_FILE  = PROJECT_ROOT / "source" / "plan-fetcher.py"
if not SOURCE_FILE.exists():
	sys.exit(f"Cannot find plan-fetcher.py at: {SOURCE_FILE}")

spec = importlib.util.spec_from_file_location("plan_fetcher", SOURCE_FILE)
plan_fetcher = importlib.util.module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(plan_fetcher) 

# ---------- Config ----------
EUROPE_LONDON = "Europe/London"
try:
	from zoneinfo import ZoneInfo
	TZ_LON = ZoneInfo(EUROPE_LONDON)
except Exception:
	TZ_LON = None

STATE_FILE = PROJECT_ROOT / "permanent_fetcher_state.json"

DARWIN_LINES = {"elizabeth", "lioness", "liberty", "mildmay", "suffragette", "weaver", "windrush"}

DISTRICT_HSK_EC_CANON = {
	("High Street Kensington Underground Station", "Earl's Court Underground Station"),
	("Earl's Court Underground Station", "High Street Kensington Underground Station"),
	("High Street Kensington Station", "Earl's Court Station"),
	("Earl's Court Station", "High Street Kensington Station"),
}

# ---------- Windows helpers ----------
def hourly_windows(start_hour: int, end_hour_inclusive: int, include_2359: bool) -> List[Tuple[str,str]]:
	wins: List[Tuple[str,str]] = []
	for h in range(start_hour, min(end_hour_inclusive, 23)):
		wins.append((f"{h:02d}00", f"{(h+1):02d}00"))
	if include_2359 and end_hour_inclusive == 23:
		wins.append(("2300", "2359"))
	return wins

def day_windows_by_bucket(night_tube: int, bucket: str) -> List[Tuple[str,str]]:
	bucket = bucket.lower()
	if night_tube != 1:
		# night_tube = 0 -> always include 06..22, exclude 23 and 00..05
		return hourly_windows(6, 22, include_2359=False)

	# night_tube = 1
	if bucket == "mon-thu":
		return hourly_windows(6, 22, include_2359=False)
	if bucket == "fri":
		# include 06..23 and include 23:59 tail
		return hourly_windows(6, 23, include_2359=True)
	if bucket == "sat":
		# include all hours 00..23 and include 23:59 tail
		return hourly_windows(0, 23, include_2359=True)
	if bucket == "sun":
		# include 00..22; exclude 23
		return hourly_windows(0, 22, include_2359=False)
	return hourly_windows(6, 22, include_2359=False)

PEAK_CHECK_WINDOWS = [(f"{h:02d}00", f"{h+1:02d}00") for h in range(6, 22)]  # 06:00–22:00

# ---------- Date helpers ----------
def now_london() -> datetime:
	if TZ_LON:
		return datetime.now(TZ_LON)
	return datetime.now()

def is_first_monday_06(now: datetime) -> bool:
	if now.weekday() != 0:
		return False
	first = now.replace(day=1, hour=6, minute=0, second=0, microsecond=0)
	while first.weekday() != 0:
		first += timedelta(days=1)
	return now >= first

def next_week_monday(d: date) -> date:
	days_ahead = (7 - d.weekday()) % 7
	if days_ahead == 0:
		days_ahead = 7
	return d + timedelta(days=days_ahead)

def upcoming_weekday_on_or_after(d: date, weekday: int) -> date:
	delta = (weekday - d.weekday()) % 7
	return d + timedelta(days=delta)

def date_to_ymd(d: date) -> str:
	return d.strftime("%Y%m%d")

# ---------- Output dirs ----------
def day_bucket_name(weekday: int) -> str:
	if weekday <= 3: return "mon-thu"
	if weekday == 4: return "fri"
	if weekday == 5: return "sat"
	return "sun"

def out_dir_for(bucket: str) -> Path:
	p = PROJECT_ROOT / "planned-runs" / bucket
	p.mkdir(parents=True, exist_ok=True)
	return p

# ---------- Duplicate headway proxy ----------
def load_planned_xml_times_per_pair(xml_file: Path) -> Dict[str, Dict[str, Any]]:
	out: Dict[str, Dict[str, Any]] = {}
	try:
		tree = ET.parse(xml_file)
		root = tree.getroot()
		for sec_el in root.findall("Section"):
			sec_id = (sec_el.text or "").strip()
			out.setdefault(sec_id, {})
			for p in sec_el.findall("Point"):
				point_name = (p.text or "").strip()
				tph_el = p.find("TPH")
				head_el = p.find("Headways")
				tph = int((tph_el.text or "0").strip() or "0") if tph_el is not None else 0
				heads = []
				if head_el is not None and (head_el.text or "").strip():
					try:
						heads = [int(x) for x in (head_el.text or "").split() if x.strip().isdigit()]
					except Exception:
						heads = []
				out[sec_id][point_name] = {"TPH": tph, "Headways": heads}
	except Exception:
		return {}
	return out

def headway_dup_signal(headways: List[int]) -> bool:
	if not headways: return False
	small = sum(1 for h in headways if h <= 2)
	return small >= 2

# ---------- JP anomaly helper ----------
def is_district_hsk_ec_pair(pair: Tuple[str, str]) -> bool:
	a, b = pair
	if (a, b) in DISTRICT_HSK_EC_CANON or (b, a) in DISTRICT_HSK_EC_CANON:
		return True
	a_l = a.lower(); b_l = b.lower()
	if "high street kensington" in a_l and "earl" in b_l: return True
	if "high street kensington" in b_l and "earl" in a_l: return True
	return False

# ---------- BuildContext ----------
class BuildContext:
	def __init__(self, restart: bool, loop: bool):
		self.restart = restart
		self.loop    = loop

def run_once_or_exit(ctx: BuildContext):
	now = now_london()
	if not ctx.restart and not is_first_monday_06(now):
		print(f"[info] Not first Monday ≥ 06:00 Europe/London ({now}). No action.")
		return
	attempt_date = now.date()
	attempt_loop(ctx, attempt_date)

def attempt_loop(ctx: BuildContext, start_date: date):
	current_attempt_date = start_date
	max_retry_days = 14
	for day_try in range(max_retry_days):
		print(f"\n=== Attempt starting {current_attempt_date} ===")
		ok_all = try_build_for_anchor_date(current_attempt_date)
		if ok_all:
			next_run = schedule_next_month_first_monday_0600(current_attempt_date)
			save_state({"next_monthly_run_utc": next_run.isoformat(), "last_successful_anchor": str(current_attempt_date)})
			print(f"[done] Baseline built. Next scheduled run: {next_run} (Europe/London)")
			return
		current_attempt_date = current_attempt_date + timedelta(days=1)
		if not ctx.loop:
			save_state({"retry_anchor_date": str(current_attempt_date)})
			print(f"[info] Anomalies detected. Saved state. Re-run tomorrow or use -loop.")
			return
		sleep_until = datetime.combine(current_attempt_date, datetime.min.time()).replace(hour=6, tzinfo=TZ_LON) if TZ_LON else None
		if sleep_until:
			now = now_london()
			secs = (sleep_until - now).total_seconds()
			if secs > 0:
				print(f"[sleep] Waiting until {sleep_until} London (~{int(secs)}s).")
				time.sleep(secs)
		else:
			time.sleep(24*3600)

def schedule_next_month_first_monday_0600(anchor_day: date) -> datetime:
	y = anchor_day.year; m = anchor_day.month
	y, m = (y+1, 1) if m == 12 else (y, m+1)
	first = date(y, m, 1)
	while first.weekday() != 0:
		first += timedelta(days=1)
	dt = datetime(first.year, first.month, first.day, 6, 0, 0)
	return dt.replace(tzinfo=TZ_LON) if TZ_LON else dt

def save_state(d: Dict[str, Any]):
	try:
		with STATE_FILE.open("w", encoding="utf-8") as f:
			json.dump(d, f, indent=2)
	except Exception as e:
		print(f"[warn] Failed to write state: {e}")

# ---------- Core build ----------
def try_build_for_anchor_date(anchor_day: date) -> bool:
	base_mon = next_week_monday(anchor_day)
	fri = base_mon + timedelta(days=4)
	sat = base_mon + timedelta(days=5)
	sun = base_mon + timedelta(days=6)

	targets = {
		"mon-thu": base_mon,
		"fri":     fri,
		"sat":     sat,
		"sun":     sun,
	}

	LINE_KEYS = [
		"elizabeth","liberty","lioness","mildmay","suffragette","weaver","windrush",
		"bakerloo","central","district","hammersmith","circle","jubilee","dlr",
		"metropolitan","northern","piccadilly","victoria","waterloo",
	]

	darwin_fail = False
	jp_fail     = False

	night_tube_by_line: Dict[str, int] = {}
	for line_key in LINE_KEYS:
		try:
			line_cfg, params, _sections = plan_fetcher.load_line_config(line_key, line_config_dir=str(PROJECT_ROOT / "line_config"))
			night_tube_by_line[line_key] = int(params.get("night_tube", 0))
		except Exception as e:
			print(f"[warn] Could not load config for {line_key}: {e}")
			night_tube_by_line[line_key] = 0

	for bucket, d0 in targets.items():
		out_dir = out_dir_for(bucket)
		date_ymd = date_to_ymd(d0)
		print(f"\n--- Building {bucket.upper()} for {d0} ({date_ymd}) ---")
		for line_key in LINE_KEYS:
			try:
				is_darwin = line_key in DARWIN_LINES
				nt_flag = night_tube_by_line.get(line_key, 0)
				windows = day_windows_by_bucket(nt_flag, bucket)

				results = plan_fetcher.generate_planned_sections(
					LINE_KEY=line_key,
					DATE_YMD=date_ymd,
					WINDOWS=windows,
					out_dir=str(out_dir),
				)

				if is_darwin:
					if darwin_no_service_6_22(results):
						print(f"[anom:darwin] {line_key}: missing service in 06–22 for ≥1 section.")
						darwin_fail = True
					if has_duplicate_signal_from_outputs(out_dir, line_key, date_ymd):
						print(f"[anom:darwin] {line_key}: duplicate/twin signal detected.")
						darwin_fail = True
				else:
					if jp_missing_service(results, line_key):
						print(f"[anom:jp] {line_key}: missing service (JP rule).")
						jp_fail = True

			except Exception as e:
				print(f"[FAIL] {line_key} for {bucket}/{date_ymd}: {e}")
				traceback.print_exc()
				if line_key in DARWIN_LINES:
					darwin_fail = True
				else:
					jp_fail = True

	if darwin_fail or jp_fail:
		intent = {"last_anchor": str(anchor_day)}
		if darwin_fail:
			intent["darwin_retry"] = "tomorrow_with_dynamic_weekday_targets"
		if jp_fail:
			intent["jp_retry"] = "shift_week_plus_7"
		save_state(intent)
		print(f"[result] Build rejected (darwin_fail={darwin_fail}, jp_fail={jp_fail}).")
		return False

	print("[result] Build accepted.")
	return True

# ---------- Anomaly helpers ----------
def darwin_no_service_6_22(results: Dict[str, Any]) -> bool:
	need = set(f"{h:02d}00-{h+1:02d}00" for h in range(6, 22))
	tph_sum: Dict[Tuple[str, str], int] = {}
	for win_key, per_sec in results.items():
		if win_key not in need:
			continue
		for sec_id, points in per_sec.items():
			for point_name, (tph, _heads) in points.items():
				tph_sum[(sec_id, point_name)] = tph_sum.get((sec_id, point_name), 0) + (tph or 0)
	for (_sec_id, _point_name), s in tph_sum.items():
		if s == 0:
			return True
	return False

def has_duplicate_signal_from_outputs(out_dir: Path, line_key: str, date_ymd: str) -> bool:
	tiny_gap_hits = 0
	checked = 0
	for h in range(6, 22):
		f = out_dir / f"{line_key}_{h:02d}-{(h+1):02d}.xml"
		if not f.exists():
			continue
		data = load_planned_xml_times_per_pair(f)
		checked += 1
		for sec_id, pts in data.items():
			for point_name, v in pts.items():
				if headway_dup_signal(v.get("Headways", [])):
					tiny_gap_hits += 1
					if tiny_gap_hits >= 2:
						return True
	return False

def jp_missing_service(results: Dict[str, Any], line_key: str) -> bool:
	exceptions: set[Tuple[str, str]] = set()
	if line_key == "district":
		first_key = sorted(results.keys())[0] if results else None
		if first_key:
			for sec_id, points in results[first_key].items():
				for point_name, _ in points.items():
					pn = point_name.lower()
					if "kensington" in pn and ("earl" in pn or "olympia" in pn):
						exceptions.add((sec_id, point_name))

	total_tph: Dict[Tuple[str, str], int] = {}
	for win_key, per_sec in results.items():
		for sec_id, points in per_sec.items():
			for point_name, (tph, _heads) in points.items():
				total_tph[(sec_id, point_name)] = total_tph.get((sec_id, point_name), 0) + (tph or 0)

	for key, s in total_tph.items():
		if s == 0 and key not in exceptions:
			return True
	return False

# ---------- Entry ----------
class BuildContext:
	def __init__(self, restart: bool, loop: bool):
		self.restart = restart
		self.loop    = loop

def parse_args(argv: List[str]):
	restart = ("-restart" in argv)
	loop    = ("-loop" in argv)
	return BuildContext(restart=restart, loop=loop)

if __name__ == "__main__":
	ctx = parse_args(sys.argv[1:])
	run_once_or_exit(ctx)
