# Multi-UAV Tactical Deconfliction Prototype

An end-to-end tactical collision avoidance and offboard guidance prototype for autonomous Unmanned Aerial Vehicles (UAVs), built exclusively with established open-source flight systems and robotics libraries:

- **Traffic Simulation:** [BlueSky Open Air Traffic Simulator](https://github.com/ProfLoek/BlueSky) (headless server mode streaming dynamic state).
- **Tactical Deconfliction Engine:** [Python-RVO2](https://github.com/sybrenstuvel/Python-RVO2) (Optimal Reciprocal Collision Avoidance / ORCA).
- **Autopilot Interface:** [PyMAVLink](https://github.com/ArduPilot/pymavlink) dispatching offboard velocity setpoints (`SET_POSITION_TARGET_LOCAL_NED`) to PX4 SITL over UDP (`127.0.0.1:14540`).

---

## 1. Environment Setup

The prototype runs in your dedicated Python virtual environment:
`/home/t/Downloads/courses/CDP1/venv`

### Minimal Setup Commands
```bash
# 1. Activate the CDP1 virtual environment
source /home/t/Downloads/courses/CDP1/venv/bin/activate

# 2. Install PyMAVLink, PyZMQ, and PyProj
pip install pymavlink pyzmq pyproj setuptools wheel cython

# 3. Install BlueSky Open ATM Simulator
pip install bluesky-simulator

# 4. Install Python-RVO2 (compiled with CMake >= 3.5 policy)
git clone https://github.com/sybrenstuvel/Python-RVO2.git /tmp/Python-RVO2
sed -i 's/VERSION 2.8/VERSION 3.5/' /tmp/Python-RVO2/CMakeLists.txt
pip install --no-build-isolation /tmp/Python-RVO2
rm -rf /tmp/Python-RVO2
```

---

## 2. Launch Commands

### BlueSky Headless Simulation Server
BlueSky runs as a headless background server communicating via ZeroMQ (`11000`/`11001`):
```bash
/home/t/Downloads/courses/CDP1/venv/bin/bluesky --headless --scenfile /home/t/uav_deconfliction/bluesky_scenario.scn
```

### Tactical Deconfliction Bridge (`traffic_bridge.py`)
```bash
/home/t/Downloads/courses/CDP1/venv/bin/python /home/t/uav_deconfliction/traffic_bridge.py \
    --stream bluesky \
    --stream-host 127.0.0.1 \
    --stream-port 11000 \
    --mavlink udpout:127.0.0.1:14540 \
    --origin-lat 52.3676 \
    --origin-lon 4.9041 \
    --ego-start 0.0,0.0 \
    --waypoint 400.0,0.0
```

### Full Automated Pipeline Execution
```bash
/home/t/uav_deconfliction/run_pipeline.sh
```

---

## 3. Mathematical & Algorithmic Architecture

### A. Geographic Coordinate Projection
Geodetic coordinates $(\text{lat}, \text{lon})$ are projected to a Local North-East-Down (NED) Cartesian frame relative to an origin point $(\text{lat}_0, \text{lon}_0)$ using an Azimuthal Equidistant projection (`pyproj.Proj(proj="aeqd", lat_0=..., lon_0=..., datum="WGS84")`):
- $x = \text{North}$ (meters)
- $y = \text{East}$ (meters)
- $v_x = v_{\text{ground}} \cdot \cos(\text{track})$
- $v_y = v_{\text{ground}} \cdot \sin(\text{track})$

### B. RVO2 / ORCA Engine Configuration
Configured with strict tactical specifications:
- `timeStep`: `0.1` s ($10\text{ Hz}$)
- `neighborDist`: `150.0` m (tactical acquisition horizon)
- `maxNeighbors`: `10`
- `timeHorizon`: `10.0` s (lookahead conflict prediction)
- `radius`: `25.0` m ($50\text{ m}$ safe separation bubble)
- `maxSpeed`: `15.0` m/s

The preferred velocity $\vec{v}_{\text{pref}}$ directs the ego drone toward the target waypoint, smoothly tapering within the arrival radius. The engine invokes `sim.doStep()` at $10\text{ Hz}$ to calculate the conflict-free velocity $\vec{v}_{\text{cmd}} = (v_{x,\text{cmd}}, v_{y,\text{cmd}})$.

### C. MAVLink Offboard Dispatcher
Offboard setpoints are dispatched to PX4 SITL using `SET_POSITION_TARGET_LOCAL_NED` (`MAVLink` Common Message #84):
- **Target Component:** `MAV_COMP_ID_AUTOPILOT1`
- **Frame:** `MAV_FRAME_LOCAL_NED`
- **Type Mask:** `0x0DC7` (Ignoring $X, Y, Z, A_X, A_Y, A_Z, \text{Yaw}, \dot{\text{Yaw}}$)
- **Velocity Payload:** $v_x = v_{x,\text{cmd}}$, $v_y = v_{y,\text{cmd}}$, $v_z = 0.0$
- **Companion Heartbeat:** Dispatched at $1\text{ Hz}$ (`MAV_TYPE_ONBOARD_CONTROLLER`)
