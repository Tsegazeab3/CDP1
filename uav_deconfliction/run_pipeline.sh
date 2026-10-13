#!/usr/bin/env bash
# ==============================================================================
# Multi-UAV Tactical Deconfliction Prototype Execution Script
# Components:
#   1. BlueSky Open Air Traffic Simulator (Headless Server)
#   2. Ingestion & RVO2 Deconfliction Node (traffic_bridge.py)
#   3. PX4 SITL MAVLink Offboard Dispatcher (UDP 127.0.0.1:14540)
# ==============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="/home/t/Downloads/courses/CDP1/venv"
PYTHON="${VENV_DIR}/bin/python"
BLUESKY_BIN="${VENV_DIR}/bin/bluesky"
SCENARIO_FILE="${SCRIPT_DIR}/bluesky_scenario.scn"

echo "======================================================================"
echo " Starting Multi-UAV Tactical Deconfliction Pipeline"
echo " Virtualenv: ${VENV_DIR}"
echo " Python:     $(${PYTHON} --version)"
echo "======================================================================"

# Ensure scenario file is available in bluesky scenarios directory if needed
mkdir -p /home/t/bluesky/scenario
cp "${SCENARIO_FILE}" /home/t/bluesky/scenario/tactical.scn

cleanup() {
    echo ""
    echo "[Pipeline] Shutting down background processes..."
    if [[ -n "${MOCK_PID}" ]] && kill -0 "${MOCK_PID}" 2>/dev/null; then
        kill "${MOCK_PID}" 2>/dev/null || true
    fi
    if [[ -n "${BLUESKY_PID}" ]] && kill -0 "${BLUESKY_PID}" 2>/dev/null; then
        kill "${BLUESKY_PID}" 2>/dev/null || true
    fi
    echo "[Pipeline] Shutdown complete."
}
trap cleanup EXIT INT TERM

# Step 1: Start Mock PX4 SITL receiver in background (or user runs real PX4 SITL)
echo "[1/3] Starting PX4 SITL MAVLink listener on UDP 14540..."
${PYTHON} "${SCRIPT_DIR}/mock_px4_sitl.py" 60 &
MOCK_PID=$!
sleep 1

# Step 2: Start BlueSky headless simulation server
echo "[2/3] Starting BlueSky Headless Simulation Server..."
${BLUESKY_BIN} --headless --scenfile "${SCENARIO_FILE}" &
BLUESKY_PID=$!
sleep 2

# Step 3: Run Ingestion, RVO2 Deconfliction & MAVLink Offboard Node
echo "[3/3] Starting Ingestion & RVO2 Tactical Deconfliction Node..."
${PYTHON} "${SCRIPT_DIR}/traffic_bridge.py" \
    --stream bluesky \
    --stream-host 127.0.0.1 \
    --stream-port 11000 \
    --mavlink udpout:127.0.0.1:14540 \
    --origin-lat 52.3676 \
    --origin-lon 4.9041 \
    --ego-start 0.0,0.0 \
    --waypoint 400.0,0.0
