# CDP1 - Multi-UAV Tactical Deconfliction & Traffic Management

An end-to-end tactical collision avoidance and offboard guidance prototype for autonomous Unmanned Aerial Vehicles (UAVs) in UTM / U-Space operations.

- **Traffic Simulation:** [BlueSky Open Air Traffic Simulator](https://github.com/ProfLoek/BlueSky) (headless server mode streaming dynamic state).
- **Tactical Deconfliction Engine:** [Python-RVO2](https://github.com/sybrenstuvel/Python-RVO2) (Optimal Reciprocal Collision Avoidance / ORCA).
- **Autopilot Interface:** [PyMAVLink](https://github.com/ArduPilot/pymavlink) dispatching offboard velocity setpoints (`SET_POSITION_TARGET_LOCAL_NED`) to PX4 SITL over UDP (`127.0.0.1:14540`).

---

## Team Members & Contributors

| Name | Student ID | Institutional Email |
| :--- | :--- | :--- |
| **Tsegazeab Kebede** (Lead) | H00540222 | H00540222@hct.ac.ae |
| **Firaol Birhanu** | H00540197 | H00540197@hct.ac.ae |
| **Firew Borsamo** | H00540207 | H00540207@hct.ac.ae |
| **Elias Endashaw** | H00540220 | H00540220@hct.ac.ae |
| **Ferhan Ibrahim** | H00540217 | H00540217@hct.ac.ae |

---

## Project Structure

- `uav_deconfliction/` — Core implementation of tactical deconfliction bridge, scenario files, and pipeline launcher.
- `bluesky/` — Air traffic simulation components.
- `research_papers_for_cdp1/` — Research and literature papers on UTM, strategic and tactical deconfliction.
- `requirements.txt` — Python dependencies.
- `run_sim.sh` — Simulation execution script.

For detailed architecture, mathematical foundations, and run instructions, see [`uav_deconfliction/README.md`](uav_deconfliction/README.md).
