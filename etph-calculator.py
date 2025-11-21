#!/usr/bin/env python3
# etph-calculator.py
# Usage:
#   python etph-calculator.py --out DIR [--planned planned.xml] [--real real.xml]

from __future__ import annotations
import argparse, sys, xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# ---------- CLI ----------

def parse_args():
	p = argparse.ArgumentParser(description="Compute ETPH and weighted waiting time from Planned/Real XMLs.")
	p.add_argument("--planned", type=str, default=None, help="Path to PlannedReference XML")
	p.add_argument("--real", type=str, default=None, help="Path to RealRuns XML")
	p.add_argument("--out", type=str, required=True, help="Output directory (will be created if missing)")
	return p.parse_args()

# ---------- small helper functions ----------

def get_text(node: Optional[ET.Element]) -> str:
if node is None:
	return ""
return (node.text or "").strip()

def find_child_text(parent: ET.Element, tag: str) -> str:
	el = parent.find(tag)
	return (el.text or "").strip() if el is not None and el.text else ""

def parse_numbers(s: str) -> List[float]:
	if not s: return []
	out: List[float] = []
	for tok in s.replace(",", " ").split():
		try: out.append(float(tok))
		except ValueError: pass
	return out

def safe_mean(xs: List[float]) -> float:
	return sum(xs) / len(xs) if xs else 0.0

def ewt_minutes(headways: List[float]) -> float:
	"""
	Compute Expected Waiting Time (minutes) from headways in minutes,
	using EWT = (Σ h^2) / (2 * Σ h). Returns 0.0 for empty/non-positive input
	"""
	if not headways: return 0.0
	s = sum(headways)
	if s <= 0: return 0.0
	s2 = sum(h*h for h in headways)
	return s2 / (2.0 * s)

def etph_from_ewt(ewt_min: float) -> float:
	return 0.0 if ewt_min <= 0 else 30.0 / ewt_min

def derive_tph_from_headways(headways: List[float]) -> float:
	m = safe_mean(headways)
	return 60.0 / m if m > 0 else 0.0

def fmt_int_pair(a: Optional[float], b: Optional[float]) -> str:
	def f(x: Optional[float]) -> str:
		if x is None: return "E"
		return str(int(round(x)))
	return f(a) + " " + f(b)

def fmt_2dp_pair(a: Optional[float], b: Optional[float]) -> str:
	def f(x: Optional[float]) -> str:
		if x is None: return "E"
		return f"{x:.2f}"
	return f(a) + " " + f(b)

# ---------- data models ----------

class PointData:
	def __init__(self, name: str, time_str: str, tph: float, headways: List[float]):
		"""Represents a single point within a section (i.e. Northbound_In)"""
		self.name = name
		self.time = time_str
		self.tph = tph
		self.headways = headways

class SectionData:
	def __init__(self, name: str, points: Dict[str, PointData]):
		"""Represents a section in the line (i.e. JUB-STA-WEM)"""
		self.name = name
		self.points = points

class RunDoc:
	def __init__(self, sections: List[SectionData]):
		self.sections = sections

# ---------- input xml parsing ----------

def parse_run_xml(path: Path) -> RunDoc:
	"""
	Parse a Planned/Real XML file into a RunDoc.
	"""
	tree = ET.parse(path)
	root = tree.getroot()
	
	sections: List[SectionData] = []
	
	# Find all Section elements
	for sec in root.findall(".//Section"):
		sec_name = get_text(sec)             # direct text content
		pts: Dict[str, PointData] = {}
	
		# Each section contains Point elements as direct children
		for pt in sec.findall("./Point"):
			pt_name = get_text(pt)           # direct text content
	
			time_str = find_child_text(pt, "Time")
			tph_str = find_child_text(pt, "TPH")
			headways_str = find_child_text(pt, "Headways")
	
			# Parse TPH and headways
			try:
				tph = float(tph_str)
			except Exception:
				tph = 0.0
	
			headways = parse_numbers(headways_str)
	
			# If TPH not supplied or is 0, derive it from headways
			if tph == 0.0 and headways:
				tph = derive_tph_from_headways(headways)
	
			pts[pt_name] = PointData(pt_name, time_str, tph, headways)
	
		sections.append(SectionData(sec_name, pts))
	
	return RunDoc(sections)


# ---------- planned against real validation ----------

def validate_match(planned: RunDoc, real: RunDoc) -> Optional[str]:
	"""
	Check that planned and real runs have the same structure and times.
	Returns an error message string, or None if everything matches.
	"""
	if len(planned.sections) != len(real.sections):
		return "Section count mismatch between planned and real."
	for i, (sp, sr) in enumerate(zip(planned.sections, real.sections)):
		if sp.name != sr.name:
			return f"Section name mismatch at index {i}: '{sp.name}' vs '{sr.name}'."
		pts_p = set(sp.points.keys()); pts_r = set(sr.points.keys())
		if pts_p != pts_r:
			return f"Point set mismatch in section '{sp.name}': {sorted(pts_p)} vs {sorted(pts_r)}."
		for p in pts_p:
			tp = sp.points[p].time; tr = sr.points[p].time
			if tp and tr and tp != tr:
				return f"Time mismatch in section '{sp.name}', point '{p}': '{tp}' vs '{tr}'."
	return None

# ---------- metrics helpers  ----------

def point_metrics(pd: PointData) -> Tuple[float, float, float]:
	tph = pd.tph
	ewt = ewt_minutes(pd.headways)
	etph = etph_from_ewt(ewt)
	return (tph, etph, ewt)

def is_four_point_section(point_names: List[str]) -> bool:
	s = set(point_names)
	for d in ["Northbound","Southbound","Eastbound","Westbound"]:
		if f"{d}_In" in s and f"{d}_Out" in s:
			return True
	return False

def is_two_point_section(point_names: List[str]) -> bool:
	s = set(point_names)
	return s in ({"Northbound","Southbound"}, {"Eastbound","Westbound"})

def aggregate_4pt(a: Optional[Tuple[float,float,float]],
				  b: Optional[Tuple[float,float,float]]) -> Tuple[float,float,float]:
	candidates = []
	for m in (a, b):
		if m is None: continue
		tph, etph, ewt = m
		if tph > 1.0:
			candidates.append((tph, etph, ewt))
	# No valid metrics -> return zeros
	if not candidates:
		return (0.0, 0.0, 0.0)
	# Only one valid side -> return it as is
	if len(candidates) == 1:
		return candidates[0]
	# Two valid sides -> simple average of (TPH, ETPH, EWT)
	return (
		(candidates[0][0] + candidates[1][0]) / 2.0,
		(candidates[0][1] + candidates[1][1]) / 2.0,
		(candidates[0][2] + candidates[1][2]) / 2.0,
	)

# ---------- output build ----------

IND = "   "

def set_block_label_text(elem: ET.Element, label: str, level: int):
	"""
	Set elem.text so that `label` appears on its own line with IND
	"""
	elem.text = "\n" + (IND * (level + 1)) + label + "\n" + (IND * (level + 1))

def set_child_tail(child: ET.Element, level: int, is_last: bool):
	child.tail = "\n" + (IND * (level + (0 if is_last else 1)))  # tail before either next sibling or parent close
	# For our structure exact spaces are managed

def build_output(planned: Optional[RunDoc], real: Optional[RunDoc]) -> ET.ElementTree:
	"""
	Build the output XML comparing planned vs real metrics per section/direction.
	"""
	root = ET.Element("ETPHResults")
	# root starts with a newline + one indent before first <Section>
	root.text = "\n" + IND

	sections = planned.sections if planned else real.sections

	for s_i, sec in enumerate(sections):
		sec_el = ET.SubElement(root, "Section")
		set_block_label_text(sec_el, sec.name, level=1) 

		point_names = list(sec.points.keys())

		if is_four_point_section(point_names):
			dirs = [d for d in ["Northbound","Southbound","Eastbound","Westbound"]
					if f"{d}_In" in sec.points or f"{d}_Out" in sec.points]
			for d_i, d in enumerate(dirs):
				def get_pd(doc: Optional[RunDoc], suffix: str) -> Optional[PointData]:
					if not doc: return None
					return doc.sections[s_i].points.get(f"{d}_{suffix}", None)
				def mk(pd: Optional[PointData]): return point_metrics(pd) if pd else None

				# Time: planned In/Out then real In/Out
				time_val = ""
				for src in (get_pd(planned,"In"), get_pd(planned,"Out"),
							get_pd(real,"In"),    get_pd(real,"Out")):
					if src and src.time:
						time_val = src.time; break

				m_p = aggregate_4pt(mk(get_pd(planned,"In")), mk(get_pd(planned,"Out"))) if planned else None
				m_r = aggregate_4pt(mk(get_pd(real,"In")),    mk(get_pd(real,"Out")))    if real    else None

				pt_el = ET.SubElement(sec_el, "Point")
				# Point name as text line
				set_block_label_text(pt_el, d, level=2) 
				# Children of point
				t = ET.SubElement(pt_el, "Time"); t.text = time_val
				t.tail = "\n" + (IND * 3)
				tph = ET.SubElement(pt_el, "TPH_Planned_Real"); tph.text = fmt_int_pair(m_p[0] if m_p else None, m_r[0] if m_r else None)
				tph.tail = "\n" + (IND * 3)
				et = ET.SubElement(pt_el, "ETPH_Planned_Real"); et.text = fmt_2dp_pair(m_p[1] if m_p else None, m_r[1] if m_r else None)
				et.tail = "\n" + (IND * 3)
				aw = ET.SubElement(pt_el, "AVG_Wait_Time_Planned_Real"); aw.text = fmt_2dp_pair(m_p[2] if m_p else None, m_r[2] if m_r else None)
				# tail after closing </Point>
				aw.tail = "\n" + (IND * 2)

		elif is_two_point_section(point_names):
			dirs = ["Northbound","Southbound"] if set(point_names)=={"Northbound","Southbound"} else ["Eastbound","Westbound"]
			for d in dirs:
				def get_pd(doc: Optional[RunDoc]) -> Optional[PointData]:
					if not doc: return None
					return doc.sections[s_i].points.get(d, None)
				pd_p = get_pd(planned); pd_r = get_pd(real)
				time_val = (pd_p.time if (pd_p and pd_p.time) else (pd_r.time if (pd_r and pd_r.time) else ""))

				m_p = point_metrics(pd_p) if pd_p else None
				m_r = point_metrics(pd_r) if pd_r else None

				pt_el = ET.SubElement(sec_el, "Point")
				set_block_label_text(pt_el, d, level=2)
				t = ET.SubElement(pt_el, "Time"); t.text = time_val
				t.tail = "\n" + (IND * 3)
				tph = ET.SubElement(pt_el, "TPH_Planned_Real"); tph.text = fmt_int_pair(m_p[0] if m_p else None, m_r[0] if m_r else None)
				tph.tail = "\n" + (IND * 3)
				et = ET.SubElement(pt_el, "ETPH_Planned_Real"); et.text = fmt_2dp_pair(m_p[1] if m_p else None, m_r[1] if m_r else None)
				et.tail = "\n" + (IND * 3)
				aw = ET.SubElement(pt_el, "AVG_Wait_Time_Planned_Real"); aw.text = fmt_2dp_pair(m_p[2] if m_p else None, m_r[2] if m_r else None)
				aw.tail = "\n" + (IND * 2)

		else:
			# fallback: emit each point independently
			for d in point_names:
				def get_pd(doc: Optional[RunDoc]) -> Optional[PointData]:
					if not doc: return None
					return doc.sections[s_i].points.get(d, None)
				pd_p = get_pd(planned); pd_r = get_pd(real)
				time_val = (pd_p.time if (pd_p and pd_p.time) else (pd_r.time if (pd_r and pd_r.time) else ""))

				m_p = point_metrics(pd_p) if pd_p else None
				m_r = point_metrics(pd_r) if pd_r else None

				pt_el = ET.SubElement(sec_el, "Point")
				set_block_label_text(pt_el, d, level=2)
				t = ET.SubElement(pt_el, "Time"); t.text = time_val
				t.tail = "\n" + (IND * 3)
				tph = ET.SubElement(pt_el, "TPH_Planned_Real"); tph.text = fmt_int_pair(m_p[0] if m_p else None, m_r[0] if m_r else None)
				tph.tail = "\n" + (IND * 3)
				et = ET.SubElement(pt_el, "ETPH_Planned_Real"); et.text = fmt_2dp_pair(m_p[1] if m_p else None, m_r[1] if m_r else None)
				et.tail = "\n" + (IND * 3)
				aw = ET.SubElement(pt_el, "AVG_Wait_Time_Planned_Real"); aw.text = fmt_2dp_pair(m_p[2] if m_p else None, m_r[2] if m_r else None)
				aw.tail = "\n" + (IND * 2)

		# After last </Point>, put newline + indent to root level for </Section>
		sec_el.tail = "\n" + IND

	# After last section, newline before closing root
	if len(sections) > 0:
		root[-1].tail = "\n"
	return ET.ElementTree(root)

# ---------- writing ----------

def write_xml(tree: ET.ElementTree, out_path: Path) -> None:
	out_path.parent.mkdir(parents=True, exist_ok=True)
	tree.write(out_path, encoding="utf-8", xml_declaration=True)

# ---------- main ----------

def main():
	args = parse_args()
	planned_path = Path(args.planned) if args.planned else None
	real_path = Path(args.real) if args.real else None
	out_dir = Path(args.out)

	if not planned_path and not real_path:
		print("Error: provide at least one of --planned or --real.", file=sys.stderr)
		return 2

	planned_doc = parse_run_xml(planned_path) if planned_path else None
	real_doc = parse_run_xml(real_path) if real_path else None

	if planned_doc and real_doc:
		err = validate_match(planned_doc, real_doc)
		if err:
			print(f"Validation error: {err}", file=sys.stderr)
			return 2

	tree = build_output(planned_doc, real_doc)

	# Output filename (planned preferred)
	src = planned_path if planned_path else real_path
	out_dir.mkdir(parents=True, exist_ok=True)
	out_path = out_dir / f"{src.name}.etph.xml"

	write_xml(tree, out_path)
	print(f"Wrote: {out_path}")
	return 0

if __name__ == "__main__":
	sys.exit(main())
