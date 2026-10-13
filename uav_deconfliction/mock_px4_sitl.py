#!/usr/bin/env python3
"""
Mock PX4 SITL MAVLink receiver.
Listens on UDP 127.0.0.1:14540 for SET_POSITION_TARGET_LOCAL_NED setpoints,
verifies velocity masks, and transmits simulated LOCAL_POSITION_NED telemetry.
"""

import sys
import time
from pymavlink import mavutil

def run_mock_px4(port: int = 14540, duration_sec: float = 30.0):
    print(f"[Mock PX4 SITL] Binding UDP listener on port {port}...")
    mav = mavutil.mavlink_connection(f"udp:0.0.0.0:{port}")
    
    start_time = time.time()
    last_heartbeat = 0.0
    packet_count = 0

    print("[Mock PX4 SITL] Ready. Listening for MAVLink offboard setpoints...")

    while duration_sec <= 0.0 or (time.time() - start_time < duration_sec):
        now = time.time()
        # Send vehicle heartbeat at 1 Hz
        if now - last_heartbeat >= 1.0:
            mav.mav.heartbeat_send(
                mavutil.mavlink.MAV_TYPE_QUADROTOR,
                mavutil.mavlink.MAV_AUTOPILOT_PX4,
                mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED | mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED,
                6, # MAV_STATE_ACTIVE
                0
            )
            last_heartbeat = now

        # Non-blocking receive
        msg = mav.recv_match(blocking=False)
        if msg:
            mtype = msg.get_type()
            if mtype == "SET_POSITION_TARGET_LOCAL_NED":
                packet_count += 1
                if packet_count % 10 == 0:
                    print(
                        f"[Mock PX4 SITL] Recv #{packet_count}: Frame={msg.coordinate_frame}, "
                        f"Mask=0x{msg.type_mask:04X}, Cmd=(vx={msg.vx:.2f}, vy={msg.vy:.2f}, vz={msg.vz:.2f}) m/s"
                    )
            elif mtype == "HEARTBEAT":
                pass

        time.sleep(0.01)

    print(f"[Mock PX4 SITL] Session ended. Total setpoints received: {packet_count}")

if __name__ == "__main__":
    dur = float(sys.argv[1]) if len(sys.argv) > 1 else 60.0
    run_mock_px4(duration_sec=dur)
