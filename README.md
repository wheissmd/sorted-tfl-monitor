# SORTED TfL Train Distribution Monitoring System

## 1. Software Purpose  
SORTED software provides a data-driven framework for analysing the performance of high-frequency metro services, with a focus on the London Underground.
Instead of relying on traditional timetable-based metrics which do not reflect how metro systems are actually operated, the software measures how efficiently trains are distributed across different sections of the network.

It does this by reconstructing both planned and observed headways, calculating weighted average passenger waiting times, and deriving Effective Trains Per Hour (ETPH) for each direction in sections of the lines (both planned and observed).
ETPH represents the quality of service as experienced by passengers as well as infrastructure utilisation efficiency.

What insights the collected data provides:

- Trains distribution levels across the network
- Service regulation quality
- Timetable maintainability with regard to infrastructure limitations
- How observed service compares to planned

This software provides a passenger-focused, data-driven performance analysis tool that can be used both as an internal benchmark to support service improvement decisions and as a way to inform the public about how the system is performing.

Two detailed presentation recordings, one covering the analysis methodology and another explaining the software architecture, will be released soon.

---

## 2. Requirements

To run this software you need:

- **Python 3.9+** installed on your system.  
- All Python dependencies listed in **requirements.txt**.  
  Install them with:

```
pip install -r requirements.txt
```

---

## 3. How to Use

### Setting Up

Create a `.env` file and ensure it follows this format

```sh
TFL_APP_KEY=[KEY_FROM_TFL_API]
S3_BUCKET=darwin.xmltimetable
S3_OBJECT_PREFIX=PPTimetable/
ACCESS_KEY=[ACCES_KEY_FROM_DARWIN]
SECRET_KEY=[SECRET_KEY_FROM_DARWIN]
REGION=eu-west-1
```


This project works in three stages:  
**(1) Build planned timetables**, **(2) Record observed departures**, and **(3) Calculate ETPH and weighted waiting-time metrics**.

---

### **1. Generate Planned Runs (Timetable Building)**

To build planned timetables, run:

```sh
python permanent-fetcher.py
```

In theory this script can be kept running continuously, renewing the timetable every month.  
However, long-term continuous use is **untested** and may cause **429 rate-limit errors** if combined with live train tracking.  
For most purposes, **building the timetable once is sufficient**.

You can limit which lines are included by editing the list `LINE_KEYS` inside the function `try_build_for_anchor_date`.

---

### **2. Generate Actual (Observed) Runs**

To collect real-time departure data, run:

```sh
python permanent-live-pass.py
```

You can edit the list `LINES` in this file to choose which lines to monitor.

At the moment, the system reliably tracks correct train departures for **approximately 3–4 hours** before accuracy degrades.

---

### **3. Calculate ETPH and Weighted Average Waiting Times**

Once both planned and real XMLs are generated, they can be found in:

- `planned-runs/`
- `real-runs/`

Before calculating metrics, ensure that your **real-runs XML** contains enough trains.  
Only use data for hours during which the tracking software was **already running before the hour began**, otherwise results will be incomplete.

To compute ETPH and weighted average waiting time:

```sh
python etph-calculator.py --out DIR [--planned planned.xml] [--real real.xml]
```

The results will be written to:

- `etph-calculated/`

---

### **Notes and Caveats**

- This tool is designed for **general performance metrics** and **section-level monitoring**, not for handling rare operational edge cases.
- The timetable builder generates data only for **06:00–22:00**, except for lines with Night Tube service.
- The system was primarily tested during daytime service; **early morning and late-night runs may be unreliable**.
- The **23:00–00:00 (11PM–12AM)** timetable window currently does **not work** and will be fixed in a future update.
- **Weekday timetables (Mon–Thu)** are identical and are therefore only collected once to save time.
- **Do not collect Darwin-based timetables on weekends**, as planned changes for the following week can create ghost runs. Ideally collect them on **Monday afternoon** or another weekday.
- The TfL Journey Planner timetable API is **extremely slow**. Gathering timetables for all lines can take **several days** due to rate limits.


---

## 4. Supported Lines (as of now)  
### Planned Runs:
<ul>
  - Bakerloo *<br>
  - Jubilee *<br>
  - Victoria *<br>
  - Piccadilly *<br>
  - Central *<br>
  - Northern *<br>
  - Waterloo & City *<br>
  - Elizabeth **<br>
  - Liberty **<br>
  - Lioness **<br>
  - Midmay **<br>
  - Suffragette **<br>
  - Weaver **<br>
  - Windrush **<br>
  - DLR *<br>
  - Hammersmith & City *<br>
  - Circle *<br>
  - District *<br>
  - Metropolitan *

</ul>

*\* - Timetable isn't fetched from Darwin or TfL timetable, because no detailed enough data is publicly available. Instead, the algorithm relies on JouneyPlanner and iterates through departures with a certain step. This is very slow, timetable generation for all lines combined take a few days. However, just generation the Monday-Thursday snippet should be fine.*

*\*\* - Timetable is fetched from Darwin API. It works way better than JourneyPlanner approach, but still problematic. I wouldn't recommend generating timetable on weekends, because a lot of temporary planned change runs coexist with normal runs, so the timetable that ends up fetching reports ghost trains. It's better to generate real runs for these trains from Monday daytime to Friday evening.*

### Real Runs:
<ul>
  - Bakerloo<br>
  - Jubilee<br>
  - Victoria<br>
  - Piccadilly<br>
  - Waterloo & City *<br>
  - Elizabeth **<br>
  - Liberty<br>
  - Lioness<br>
  - Midmay<br>
  - Suffragette<br>
  - Weaver<br>
  - Windrush<br>

</ul>

*\* - Waterloo & City Line technically has a shuttle path of the code, that is supposed to work and it did in a few tests, but in the others it was skipping trains and reporting nonsense. It does something, but it barely does anything useful.*

*\*\* - Elizabeth Line has its own quirks with station names and ID resolving, it is a bit messy in the code, but it seems to work generally, just less reliable than other lines. Ghost trains are a bit more frequent, which might significantly affect ETPH calculations on lower frequency branches. The values in the core and eastern branches seem to be reliable though.*

NOTE: Subsurface lines and DLR aren't supported due to lack of reliable vehicleID field in API's arrivals feed. DLR is mapped, but it is not going to work. Metropolitan Line is also mapped, but reports a lot of false departures of the trains with ID "000". I understand that subsurface and DLR networks are exactly the kind of systems worth monitoring with this software, so their addition is a priority, but it requires significantly different analysis strategies. 

---
## 5. Known Limitations  
Most of the limitations are described in **How to use** section. This section will be completed soon when full documentation is done.

---

## 6. Licensing / Usage Notice

No formal licence currently applies to this software.  
However, if you use this project or its outputs in any public-facing work, **please include a reference or attribution** to the original author.

---

## 7. Development Status and Notes

This project is in a **very early stage of development**. Some modules currently lack full commenting, and there is not yet a comprehensive public documentation of the system’s internal logic — this will be released soon.

The codebase is highly nuanced due to the realities of working with **TfL’s Arrivals API**, which is *not* designed for the kind of tracking, reconstruction, and analytical use this project performs. Access to reliable Tube timetable data is also significantly more difficult than it appears. All of this complexity will be explained in detail in the upcoming documentation.

At present:

- **Do not rely on the software for long-running accuracy.**  
  Many lines start reporting incorrect or unstable data after around **3–4 hours** of continuous operation.
- Several lines are experimental or partially implemented and may not function as intended.
- A full explanation video and a keynote presentation will be released soon, outlining the system’s purpose, architecture, and methodology.

**Created with assistance from ChatGPT by OpenAI, thus include partially AI generated code**, though all code has been reviewed and cleaned to remove AI artifacts.
