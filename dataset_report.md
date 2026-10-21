# Master Dataset Audit & Validation Report

**Date Generated**: `2026-10-21`  
**Random Seed**: `42`  
**Schema**: `report_id, source, text, text_origin, event_date, flight_phase, raw_codes, cause_text, text_may_contain_cause, hfacs_code, keyword_label, llm_label, gold_label, label_source`

---

## 1. Dataset Integrity & Summary Checks

| Metric | ASRS Master (`asrs_master.csv`) | NTSB Master (`ntsb_master.csv`) | Total Combined |
| :--- | :---: | :---: | :---: |
| **Total Rows** | 38,655 | 18,224 | 56,879 |
| **Unique Report IDs** | 38,655 (100% unique) | 18,224 (100% unique) | 56,879 |
| **Duplicates Detected** | 0 (PASS) | 0 (PASS) | 0 (PASS) |
| **Null / Empty Texts** | 0 (0.00%) | 41 (0.22%) | 41 |
| **Text Origin** | `pilot_first_person` (100%) | `witness_factual` (91.8%), `investigator_narrative` (7.9%) | Mixed by source |
| **Date Coverage** | 2012-01 to 2022-03 | 2008-01-01 to 2022-12-29 | Operational range |

## 2. Text Narrative Length Statistics

### Character Count Statistics
| Source | Mean | Std Dev | Min | 25th % | Median | 75th % | Max |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **ASRS** | 1814.2 | 1474.9 | 11 | 837 | 1423 | 2338 | 23,782 |
| **NTSB** | 3525.0 | 4202.7 | 0 | 755 | 1626 | 4418 | 16,368 |

### Word Count Statistics
| Source | Mean | Std Dev | Min | 25th % | Median | 75th % | Max |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **ASRS** | 320.0 | 262.0 | 2 | 146 | 251 | 413 | 4,375 |
| **NTSB** | 565.2 | 671.8 | 0 | 123 | 263 | 707 | 2,828 |

## 3. Canonical Keyword Label Distribution

| HFACS Category | ASRS Count | ASRS % | NTSB Count | NTSB % | Combined Total |
| :--- | -: | -: | -: | -: | -: |
| **Skill-Based Error** | 2,833 |  7.33% | 656 |  3.60% | 3,489 |
| **Decision Error** | 17,875 | 46.24% | 9,008 | 49.43% | 26,883 |
| **Violation** | 680 |  1.76% | 132 |  0.72% | 812 |
| **Perceptual Error** | 658 |  1.70% | 823 |  4.52% | 1,481 |
| **Other/Unclear** | 0 |  0.00% | 0 |  0.00% | 0 |
| **Unlabeled** | 16,609 | 42.97% | 7,605 | 41.73% | 24,214 |

## 4. NTSB Cause-Leakage Detection (`text_may_contain_cause`)

- **Total NTSB Reports**: 18,224
- **Reports with `text_may_contain_cause = True`**: **50 (0.27%)**
- **Reports with `text_may_contain_cause = False`**: **18,174 (99.73%)**
- **Notes**: In 91.8% of cases, narrative text originates from the separate factual investigation (`narr_accf`), completely isolating probable cause determinations. In the remaining cases, cause-determination sentences were stripped. Only 0.27% retain trigger words, providing a leak-free benchmark for transfer evaluation.

## 5. Blinded Sample Inspection (10 Reports Per Source)

> [!IMPORTANT]
> **Blinding Protocol**: To prevent cognitive confirmation bias during qualitative inspection, narrative texts are presented below **without keyword labels**. The corresponding keyword labels are stored in Section 6.

### 5.1 ASRS Sample Narratives (First-Person Voluntary Reports)

#### Report #1: `ASRS_1627702`
- **Source**: `ASRS` | **Origin**: `pilot_first_person` | **Date**: `2019-03` | **Phase**: `Initial Approach`
- **Narrative**: *"I was working sector 12R. Frequency congestion was high; traffic volume and complexity were rated between 3 and 4. Aircraft X wanted a visual approach in to TEX. I descended the aircraft to 120 (lowest possible MIA in vicinity). I asked the aircraft if he had the airport in sight (quickly coming ..."*

#### Report #2: `ASRS_1100110`
- **Source**: `ASRS` | **Origin**: `pilot_first_person` | **Date**: `2013-07` | **Phase**: `Cruise`
- **Narrative**: *"I was on an IFR clearance at 9;000 MSL direct to RMG VOR. We were in the vicinity of 12J (Brewton; AL) when I received a TA alerting traffic 400 feet below me and approximately 1 mile to my 3 o'clock.  I queried ATC about the traffic; but Controller did not respond.  Instead; ATC responded to an ..."*

#### Report #3: `ASRS_1827974`
- **Source**: `ASRS` | **Origin**: `pilot_first_person` | **Date**: `2021-07` | **Phase**: `Landing`
- **Narrative**: *"On a VFR flight from HTL to 48D in day VMC with unrestricted visibility and high ceilings; with a right seat Private Pilot passenger. Briefed that Runway 09/27 at 48D was NOTAMed closed; and elected to land Runway 22 given westerly winds and NOTAM.  I made multiple callouts on CTAF including base..."*

#### Report #4: `ASRS_1782395`
- **Source**: `ASRS` | **Origin**: `pilot_first_person` | **Date**: `2020-12` | **Phase**: `Cruise; Parked`
- **Narrative**: *"Enroute from ZZZ1-ZZZ2 [airport]; Aircraft X; [Date]. Approaching ZZZ3 [airport] ATC informed us that ZZZ2 [airport] was not accepting any aircraft. Asked ATC for amplification and was told ZZZ1 Center [had a] COVID issue. Advised ATC of intention to divert to our scheduled alternate of ZZZ4 [air..."*

#### Report #5: `ASRS_1748605`
- **Source**: `ASRS` | **Origin**: `pilot_first_person` | **Date**: `2020-06` | **Phase**: `Parked`
- **Narrative**: *"Aircraft was MELed for an AHRS connectivity issue on DATE and was legally deferred. It states there 'may be inoperative'. However; upon looking closer 1 AHRS is required for dispatch; meaning the MEL on DATE should of been a grounding issue. This was caught on DATE around XA:49 local. It was flow..."*

#### Report #6: `ASRS_1827181`
- **Source**: `ASRS` | **Origin**: `pilot_first_person` | **Date**: `2021-07` | **Phase**: `Takeoff / Launch`
- **Narrative**: *"At XA:34; after holding short at ZZZ Runway XX; Taxiway XX; I announced on CTAF frequency that I was departing with my Skylane C172S; Aircraft X; from Runway XX for left closed traffic. The AWOS-3 weather observation was for variable winds; 3 KT. One other airplane in the vicinity reported their ..."*

#### Report #7: `ASRS_1806679`
- **Source**: `ASRS` | **Origin**: `pilot_first_person` | **Date**: `2021-05` | **Phase**: `Other Maneuvering`
- **Narrative**: *"I departed from FXE in Aircraft X for a training flight with a CFII that was supposed to consist of commercial maneuvers. I noticed an aircraft about 20-30 feet below me on a collision course with us passing from our left to right. The aircraft looked like some kind of Learjet with wing tip fuel ..."*

#### Report #8: `ASRS_1252157`
- **Source**: `ASRS` | **Origin**: `pilot_first_person` | **Date**: `2015-03` | **Phase**: `Unknown`
- **Narrative**: *"A and D flight attendant out of jumpseat during landing due to doing CPR on ill pax. Passenger was coming out of the forward restroom and had a loss of breathe. Passenger came on board with a personal oxygen container (POC). After exiting the restroom he sat on the jumpseat and complained of not ..."*

#### Report #9: `ASRS_1350152`
- **Source**: `ASRS` | **Origin**: `pilot_first_person` | **Date**: `2016-04` | **Phase**: `Parked`
- **Narrative**: *"An Aircraft had a job card assigned on a through trip. In the process of performing the assignment which was a Navigation (NAV) Data Load; I found that the reference for the process was indicated as being for a Boeing 737 instead of Airbus. The Maintenance Control Website did not have the NAV Dat..."*

#### Report #10: `ASRS_1225179`
- **Source**: `ASRS` | **Origin**: `pilot_first_person` | **Date**: `2014-12` | **Phase**: `Parked`
- **Narrative**: *"We flew the first leg of a multi-leg trip that had our charter B-747 flying two round trip charters in 4 days. On this first day of the operation we encountered a significant maintenance issue that could have jeopardized the entire sequence. After the aircraft was loaded with 286 passengers all t..."*

### 5.2 NTSB Sample Narratives (Factual Investigation Reports)

#### Report #1: `NTSB_20090908X12042`
- **Source**: `NTSB` | **Origin**: `witness_factual` | **Date**: `2009-09-07` | **Phase**: `Unknown` | **Cause Leakage Flag**: `False`
- **Narrative**: *"The pilot said he was taxiing to runway 19 for departure. The reported winds were from 190 degrees at 18 knots gusting to 26 knots. The pilot taxied to the end of the runway after performing all the preflight checks and he turned to a downwind direction to check for landing traffic.  As the airpl..."*

#### Report #2: `NTSB_20161019X50142`
- **Source**: `NTSB` | **Origin**: `witness_factual` | **Date**: `2016-10-19` | **Phase**: `Unknown` | **Cause Leakage Flag**: `False`
- **Narrative**: *"***This report was modified on June 24, 2020. Please see the docket for this accident to view the original report.***  On October 19, 2016, about 1240 eastern daylight time, an Ercoupe 415C, N415WE, was substantially damaged when it impacted terrain in Lindenwold, New Jersey. The private pilot an..."*

#### Report #3: `NTSB_20171219X65947`
- **Source**: `NTSB` | **Origin**: `witness_factual` | **Date**: `2017-12-19` | **Phase**: `Unknown` | **Cause Leakage Flag**: `False`
- **Narrative**: *"On December 19, 2017, about 1500 eastern standard time, a Piper PA-23-250, a multi-engine, retractable landing gear airplane, N40285, sustained substantial damage during a forced landing following a loss of engine power on the left engine. The accident occurred as the airplane was attempting to r..."*

#### Report #4: `NTSB_20121110X12436`
- **Source**: `NTSB` | **Origin**: `witness_factual` | **Date**: `2012-11-08` | **Phase**: `Unknown` | **Cause Leakage Flag**: `False`
- **Narrative**: *"HISTORY OF FLIGHTOn November 8, 2012, about 0904 Pacific standard time, an experimental light sport North Wing Scout X-C, weight-shift-control (WSC) trike, N467XW, collided with the ground while maneuvering about 10 miles North of Waterville, Washington. The student sport pilot and sport pilot ra..."*

#### Report #5: `NTSB_20101113X11052`
- **Source**: `NTSB` | **Origin**: `witness_factual` | **Date**: `2010-11-12` | **Phase**: `Unknown` | **Cause Leakage Flag**: `False`
- **Narrative**: *"The pilot stated that the airplane's 15-gallon fuel tank was about half-full when he departed on a local flight. At the conclusion of the one-hour flight, while on approach to the airport, the airplaneâs engine stopped producing power. The pilot set up for a forced landing to a southerly-orient..."*

#### Report #6: `NTSB_20201019102162`
- **Source**: `NTSB` | **Origin**: `witness_factual` | **Date**: `2020-10-18` | **Phase**: `Unknown` | **Cause Leakage Flag**: `False`
- **Narrative**: *"On October 18, 2020, about 1059 eastern daylight time, a Robinson Helicopter Company R22 Beta, N566BC, was substantially damaged when it was involved in an accident near Barrow County Airport (WDR), Winder, Georgia. The private pilot was seriously injured. The helicopter was operated as a Title 1..."*

#### Report #7: `NTSB_20150220X72637`
- **Source**: `NTSB` | **Origin**: `witness_factual` | **Date**: `2015-02-20` | **Phase**: `Unknown` | **Cause Leakage Flag**: `False`
- **Narrative**: *"HISTORY OF FLIGHTOn February 20, 2015, about 1017 central standard time, a Piper PA-32RT-300T single-engine airplane, N22267, overran the end of the runway during an aborted takeoff at Lubbock Executive Airpark (F82), Lubbock, Texas. The private pilot and passenger were not injured, and the airpl..."*

#### Report #8: `NTSB_20081112X70016`
- **Source**: `NTSB` | **Origin**: `witness_factual` | **Date**: `2008-11-12` | **Phase**: `Unknown` | **Cause Leakage Flag**: `False`
- **Narrative**: *"HISTORY OF FLIGHT  On November 12, 2008, at 1431 Atlantic standard time, a Robinson Helicopter R44, N452JM, was substantially damaged when it impacted a runway, following a loss of control while hovering at Fernando Luis Ribas Dominicci Airport (TJIG), San Juan, Puerto Rico. The certificated comm..."*

#### Report #9: `NTSB_20160705X24026`
- **Source**: `NTSB` | **Origin**: `witness_factual` | **Date**: `2016-07-04` | **Phase**: `Unknown` | **Cause Leakage Flag**: `False`
- **Narrative**: *"On July 4, 2016, about 1400 central daylight time, a Bellanca 7ECA airplane, N87083, was substantially damaged when it hit a ditch during an aborted takeoff and runway excursion at the Barham Airport (01LA), Oak Ridge, Louisiana. The private pilot was not injured. The airplane was owned by  Flyin..."*

#### Report #10: `NTSB_20210325102803`
- **Source**: `NTSB` | **Origin**: `witness_factual` | **Date**: `2021-03-18` | **Phase**: `Unknown` | **Cause Leakage Flag**: `False`
- **Narrative**: *"On March 18, 2021, about 1427 mountain standard time, an experimental, amateur-built Lancair Evolution, N469KS, was substantially damaged when it was involved in an accident near Phoenix, Arizona. The pilot was not injured. The airplane was operated as a Title 14 Code of Federal Regulations Part ..."*

---
## 6. Blinded Sample Key: Assigned Keyword Labels

The table below lists the canonical `KeywordWeakLabeler` assignments for the 20 sampled reports above:

| Sample Index | Report ID | Source | Text Origin | Keyword Label |
| :-: | :--- | :---: | :---: | :--- |
| ASRS # 1 | `ASRS_1627702` | ASRS | `pilot_first_person` | **Decision Error** |
| ASRS # 2 | `ASRS_1100110` | ASRS | `pilot_first_person` | **Unlabeled** |
| ASRS # 3 | `ASRS_1827974` | ASRS | `pilot_first_person` | **Decision Error** |
| ASRS # 4 | `ASRS_1782395` | ASRS | `pilot_first_person` | **Decision Error** |
| ASRS # 5 | `ASRS_1748605` | ASRS | `pilot_first_person` | **Unlabeled** |
| ASRS # 6 | `ASRS_1827181` | ASRS | `pilot_first_person` | **Decision Error** |
| ASRS # 7 | `ASRS_1806679` | ASRS | `pilot_first_person` | **Unlabeled** |
| ASRS # 8 | `ASRS_1252157` | ASRS | `pilot_first_person` | **Decision Error** |
| ASRS # 9 | `ASRS_1350152` | ASRS | `pilot_first_person` | **Unlabeled** |
| ASRS #10 | `ASRS_1225179` | ASRS | `pilot_first_person` | **Decision Error** |
| NTSB # 1 | `NTSB_20090908X12042` | NTSB | `witness_factual` | **Unlabeled** |
| NTSB # 2 | `NTSB_20161019X50142` | NTSB | `witness_factual` | **Unlabeled** |
| NTSB # 3 | `NTSB_20171219X65947` | NTSB | `witness_factual` | **Decision Error** |
| NTSB # 4 | `NTSB_20121110X12436` | NTSB | `witness_factual` | **Decision Error** |
| NTSB # 5 | `NTSB_20101113X11052` | NTSB | `witness_factual` | **Unlabeled** |
| NTSB # 6 | `NTSB_20201019102162` | NTSB | `witness_factual` | **Unlabeled** |
| NTSB # 7 | `NTSB_20150220X72637` | NTSB | `witness_factual` | **Decision Error** |
| NTSB # 8 | `NTSB_20081112X70016` | NTSB | `witness_factual` | **Decision Error** |
| NTSB # 9 | `NTSB_20160705X24026` | NTSB | `witness_factual` | **Decision Error** |
| NTSB #10 | `NTSB_20210325102803` | NTSB | `witness_factual` | **Unlabeled** |
