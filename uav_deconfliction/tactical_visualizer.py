#!/usr/bin/env python3
"""
tactical_visualizer.py
======================
High-Fidelity Real-Time Tactical Radar & Telemetry HUD for Multi-UAV Deconfliction.
Renders:
- 2D Cartesian North-East-Down (NED) radar scope with range rings
- Ego UAV with protected separation bubble (25m), acquisition zone (150m), and flight trail
- Commanded vs Preferred velocity vectors
- Traffic UAVs with callsigns, velocities, and predicted conflict lines
- Real-time offboard MAVLink status and ORCA active constraint telemetry
- Capable of live rendering on desktop display and exporting animated GIF artifacts
"""

import math
import os
import time
from typing import Dict, List, Optional, Tuple
from PIL import Image

import pygame

# Colors (Aerospace Tactical Palette)
BG_COLOR = (12, 17, 24)
RADAR_BG = (15, 23, 34)
GRID_COLOR = (26, 38, 54)
RING_COLOR = (35, 50, 70)
TEXT_DIM = (120, 140, 165)
TEXT_BRIGHT = (220, 235, 250)
ACCENT_BLUE = (64, 150, 255)
CYAN_PREF = (0, 220, 255)
LIME_CMD = (50, 255, 120)
WARN_AMBER = (255, 180, 40)
ALERT_RED = (255, 55, 75)
PANEL_BG = (18, 25, 38)
PANEL_BORDER = (40, 55, 80)
WAYPOINT_GOLD = (255, 215, 0)
TRAFFIC_COLOR = (255, 140, 50)


class TacticalVisualizer:
    def __init__(
        self,
        width: int = 1200,
        height: int = 800,
        radar_center_offset: Tuple[int, int] = (420, 400),
        scale_pixels_per_meter: float = 0.85,
        record_gif: bool = True,
        gif_output_path: Optional[str] = None,
        max_record_frames: int = 250,
    ):
        self.width = width
        self.height = height
        self.radar_center = radar_center_offset
        self.scale = scale_pixels_per_meter
        self.record_gif = record_gif
        self.gif_output_path = gif_output_path
        self.max_record_frames = max_record_frames

        self.recorded_frames: List[Image.Image] = []
        self.frame_count = 0
        self.ego_trail: List[Tuple[float, float]] = []

        # Pygame initialization
        pygame.init()
        pygame.font.init()
        self.font_main = pygame.font.SysFont("monospace", 14, bold=True)
        self.font_header = pygame.font.SysFont("monospace", 16, bold=True)
        self.font_title = pygame.font.SysFont("monospace", 20, bold=True)
        self.font_small = pygame.font.SysFont("monospace", 12)

        # Window setup
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Tactical Multi-UAV Deconfliction Radar [BlueSky | RVO2 | PX4 MAVLink]")
        self.clock = pygame.time.Clock()

    def ned_to_screen(self, north: float, east: float, camera_center: Tuple[float, float] = (0.0, 0.0)) -> Tuple[int, int]:
        """
        Converts Local NED (North = +X up, East = +Y right) to screen coordinates.
        Screen: X is right (East), Y is down (-North).
        """
        cx, cy = self.radar_center
        # Relative to camera center
        rel_east = east - camera_center[1]
        rel_north = north - camera_center[0]

        screen_x = int(cx + rel_east * self.scale)
        screen_y = int(cy - rel_north * self.scale)
        return screen_x, screen_y

    def draw_arrow(
        self,
        surface: pygame.Surface,
        start: Tuple[int, int],
        end: Tuple[int, int],
        color: Tuple[int, int, int],
        width: int = 3,
        arrow_size: int = 8,
    ):
        """Draws a clean directional vector arrow."""
        pygame.draw.line(surface, color, start, end, width)
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        length = math.hypot(dx, dy)
        if length < 5:
            return

        angle = math.atan2(dy, dx)
        p1 = (
            int(end[0] - arrow_size * math.cos(angle - math.pi / 6)),
            int(end[1] - arrow_size * math.sin(angle - math.pi / 6)),
        )
        p2 = (
            int(end[0] - arrow_size * math.cos(angle + math.pi / 6)),
            int(end[1] - arrow_size * math.sin(angle + math.pi / 6)),
        )
        pygame.draw.polygon(surface, color, [end, p1, p2])

    def render_frame(
        self,
        ego_pos: Tuple[float, float],
        target_waypoint: Tuple[float, float],
        v_pref: Tuple[float, float],
        v_cmd: Tuple[float, float],
        traffic: list,
        active_conflicts: int,
        orca_lines: int,
        mav_status: str,
        dt_hz: float = 10.0,
        sim_time_str: str = "",
    ) -> bool:
        """
        Renders a single tactical frame. Returns False if user closed window.
        """
        # Event processing
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE or event.key == pygame.K_q:
                    return False

        self.screen.fill(BG_COLOR)

        # 1. Draw Radar Scope Area
        radar_rect = pygame.Rect(0, 0, 840, self.height)
        pygame.draw.rect(self.screen, RADAR_BG, radar_rect)

        # Draw grid & range rings (50m, 100m, 150m, 200m, 250m) centered on ego or origin
        # We center the radar display around ego's forward motion
        cam_center = (ego_pos[0] * 0.5, ego_pos[1] * 0.5)
        cx, cy = self.radar_center

        for r_m in [100, 200, 300, 400]:
            r_px = int(r_m * self.scale)
            pygame.draw.circle(self.screen, RING_COLOR, self.ned_to_screen(0.0, 0.0, cam_center), r_px, 1)
            # Label
            lbl = self.font_small.render(f"{r_m}m", True, TEXT_DIM)
            lbl_pos = self.ned_to_screen(float(r_m), 0.0, cam_center)
            self.screen.blit(lbl, (lbl_pos[0] + 4, lbl_pos[1] - 8))

        # Origin Crosshair
        orig_s = self.ned_to_screen(0.0, 0.0, cam_center)
        pygame.draw.line(self.screen, GRID_COLOR, (orig_s[0] - 20, orig_s[1]), (orig_s[0] + 20, orig_s[1]), 1)
        pygame.draw.line(self.screen, GRID_COLOR, (orig_s[0], orig_s[1] - 20), (orig_s[0], orig_s[1] + 20), 1)

        # Cardinal directions
        north_lbl = self.font_header.render("N (+X)", True, ACCENT_BLUE)
        east_lbl = self.font_header.render("E (+Y)", True, ACCENT_BLUE)
        self.screen.blit(north_lbl, (cx - 24, 20))
        self.screen.blit(east_lbl, (840 - 70, cy - 10))

        # Traffic Feed Banner on Radar Scope
        if len(traffic) == 0:
            warn_box = pygame.Rect(cx - 190, cy - 25, 380, 50)
            pygame.draw.rect(self.screen, (40, 25, 10), warn_box)
            pygame.draw.rect(self.screen, WARN_AMBER, warn_box, 2)
            w_text = self.font_main.render("WAITING FOR BLUESKY TRAFFIC...", True, WARN_AMBER)
            w_sub = self.font_small.render("Verify BlueSky server is active on port 11000", True, TEXT_BRIGHT)
            self.screen.blit(w_text, (cx - 150, cy - 18))
            self.screen.blit(w_sub, (cx - 170, cy + 4))
        else:
            tr_info = self.font_small.render(f"BLUESKY TRAFFIC ACTIVE: {len(traffic)} INTRUDERS TRACKED", True, LIME_CMD)
            self.screen.blit(tr_info, (20, 20))

        # 2. Draw Target Waypoint
        wp_s = self.ned_to_screen(target_waypoint[0], target_waypoint[1], cam_center)
        pygame.draw.circle(self.screen, WAYPOINT_GOLD, wp_s, 6)
        pygame.draw.circle(self.screen, WAYPOINT_GOLD, wp_s, 16, 2)
        wp_lbl = self.font_main.render(f"WAYPOINT ({target_waypoint[0]:.0f}m, {target_waypoint[1]:.0f}m)", True, WAYPOINT_GOLD)
        self.screen.blit(wp_lbl, (wp_s[0] + 12, wp_s[1] - 8))

        # Line from ego to waypoint
        ego_s = self.ned_to_screen(ego_pos[0], ego_pos[1], cam_center)
        pygame.draw.line(self.screen, (70, 70, 50), ego_s, wp_s, 1)

        # Update trail
        self.ego_trail.append(ego_pos)
        if len(self.ego_trail) > 120:
            self.ego_trail.pop(0)

        # Draw trail
        if len(self.ego_trail) > 1:
            points = [self.ned_to_screen(p[0], p[1], cam_center) for p in self.ego_trail]
            pygame.draw.lines(self.screen, (30, 80, 110), False, points, 2)

        # 3. Draw Traffic UAVs
        for tr in traffic:
            tr_s = self.ned_to_screen(tr.x, tr.y, cam_center)
            # Distance to ego
            dist_to_ego = math.hypot(tr.x - ego_pos[0], tr.y - ego_pos[1])
            is_conflicting = dist_to_ego < 50.0 or (dist_to_ego < 150.0 and active_conflicts > 0)
            
            zone_color = ALERT_RED if is_conflicting else WARN_AMBER

            # 25m protected separation circle around traffic
            pygame.draw.circle(self.screen, zone_color, tr_s, int(25.0 * self.scale), 1)

            # Drone symbol (filled triangle or circle)
            pygame.draw.circle(self.screen, zone_color, tr_s, 5)

            # Velocity vector
            tr_vel_end = self.ned_to_screen(tr.x + tr.vx * 3.0, tr.y + tr.vy * 3.0, cam_center)
            self.draw_arrow(self.screen, tr_s, tr_vel_end, zone_color, width=2, arrow_size=6)

            # Callsign & Info tag
            tag_color = ALERT_RED if is_conflicting else TEXT_DIM
            tag_text = self.font_small.render(f"{tr.id} ({dist_to_ego:.0f}m)", True, tag_color)
            self.screen.blit(tag_text, (tr_s[0] + 8, tr_s[1] - 8))

        # 4. Draw Ego UAV
        ego_color = ALERT_RED if active_conflicts > 0 else LIME_CMD

        # 150m acquisition neighbor horizon
        pygame.draw.circle(self.screen, (25, 45, 60), ego_s, int(150.0 * self.scale), 1)

        # 25m protected separation zone
        pygame.draw.circle(self.screen, ego_color, ego_s, int(25.0 * self.scale), 2)
        pygame.draw.circle(self.screen, (ego_color[0], ego_color[1], ego_color[2]), ego_s, 6)

        # Preferred Velocity Vector (Cyan)
        pref_end = self.ned_to_screen(ego_pos[0] + v_pref[0] * 3.0, ego_pos[1] + v_pref[1] * 3.0, cam_center)
        self.draw_arrow(self.screen, ego_s, pref_end, CYAN_PREF, width=2, arrow_size=7)

        # Commanded Deconflicted Velocity Vector (Bright Green or Red if in conflict avoidance)
        cmd_end = self.ned_to_screen(ego_pos[0] + v_cmd[0] * 3.0, ego_pos[1] + v_cmd[1] * 3.0, cam_center)
        self.draw_arrow(self.screen, ego_s, cmd_end, LIME_CMD, width=3, arrow_size=9)

        # Ego Callout
        ego_lbl = self.font_main.render("EGO (PX4)", True, LIME_CMD)
        self.screen.blit(ego_lbl, (ego_s[0] - 30, ego_s[1] + 28))

        # 5. Right-Side Telemetry HUD Panel
        panel_rect = pygame.Rect(840, 0, self.width - 840, self.height)
        pygame.draw.rect(self.screen, PANEL_BG, panel_rect)
        pygame.draw.line(self.screen, PANEL_BORDER, (840, 0), (840, self.height), 2)

        # HUD Content
        x_pad = 860
        y = 20

        title = self.font_title.render("TACTICAL DECONFLICTION", True, ACCENT_BLUE)
        self.screen.blit(title, (x_pad, y))
        y += 24
        sub = self.font_small.render("BlueSky ATM | Python-RVO2 | PX4 MAVLink", True, TEXT_DIM)
        self.screen.blit(sub, (x_pad, y))
        y += 30

        # Status Banner
        status_text = "STATUS: DECONFLICTING (ORCA ACTIVE)" if active_conflicts > 0 else "STATUS: CLEAR PATH (NOMINAL)"
        status_col = ALERT_RED if active_conflicts > 0 else LIME_CMD
        pygame.draw.rect(self.screen, (status_col[0] // 4, status_col[1] // 4, status_col[2] // 4), (x_pad, y, 310, 28))
        pygame.draw.rect(self.screen, status_col, (x_pad, y, 310, 28), 1)
        stat_render = self.font_main.render(status_text, True, status_col)
        self.screen.blit(stat_render, (x_pad + 10, y + 6))
        y += 45

        # Section: Kinematics
        sec1 = self.font_header.render("--- EGO UAV KINEMATICS ---", True, ACCENT_BLUE)
        self.screen.blit(sec1, (x_pad, y))
        y += 22

        dist_wp = math.hypot(target_waypoint[0] - ego_pos[0], target_waypoint[1] - ego_pos[1])
        kinematics_data = [
            f"Position (N, E):  ({ego_pos[0]:6.1f}m, {ego_pos[1]:6.1f}m)",
            f"Distance to WP:   {dist_wp:6.1f} m",
            f"Target Waypoint:  ({target_waypoint[0]:.0f}m, {target_waypoint[1]:.0f}m)",
            f"Loop Frequency:   {dt_hz:.1f} Hz (Period: 100ms)",
        ]
        for line in kinematics_data:
            t_ren = self.font_small.render(line, True, TEXT_BRIGHT)
            self.screen.blit(t_ren, (x_pad, y))
            y += 18

        y += 10
        # Section: RVO2 Deconfliction Engine
        sec2 = self.font_header.render("--- RVO2 / ORCA ENGINE ---", True, ACCENT_BLUE)
        self.screen.blit(sec2, (x_pad, y))
        y += 22

        spd_pref = math.hypot(v_pref[0], v_pref[1])
        spd_cmd = math.hypot(v_cmd[0], v_cmd[1])
        rvo_data = [
            f"Protected Zone:   25.0 m (Sep: 50.0m)",
            f"Time Horizon:     10.0 s",
            f"Neighbor Acq Dist:150.0 m (Max: 10)",
            f"Active Conflicts: {active_conflicts}",
            f"ORCA Halfplanes:  {orca_lines} active constraints",
            f"V_PREF:           ({v_pref[0]:5.2f}, {v_pref[1]:5.2f}) | {spd_pref:4.1f}m/s",
            f"V_CMD (ORCA):     ({v_cmd[0]:5.2f}, {v_cmd[1]:5.2f}) | {spd_cmd:4.1f}m/s",
        ]
        for line in rvo_data:
            col = ALERT_RED if "Active Conflicts: 0" not in line and "Active Conflicts:" in line and active_conflicts > 0 else TEXT_BRIGHT
            t_ren = self.font_small.render(line, True, col)
            self.screen.blit(t_ren, (x_pad, y))
            y += 18

        y += 10
        # Vector Legend
        pygame.draw.line(self.screen, CYAN_PREF, (x_pad, y + 6), (x_pad + 30, y + 6), 3)
        self.screen.blit(self.font_small.render("V_PREF (Waypoint Goal Vector)", True, CYAN_PREF), (x_pad + 38, y))
        y += 18
        pygame.draw.line(self.screen, LIME_CMD, (x_pad, y + 6), (x_pad + 30, y + 6), 3)
        self.screen.blit(self.font_small.render("V_CMD (RVO2 Deconflicted Output)", True, LIME_CMD), (x_pad + 38, y))
        y += 28

        # Section: MAVLink SITL Dispatcher
        sec3 = self.font_header.render("--- PX4 AUTOPILOT INTERFACE ---", True, ACCENT_BLUE)
        self.screen.blit(sec3, (x_pad, y))
        y += 22

        mav_data = [
            f"Interface Protocol: MAVLink v2.0 (UDP)",
            f"Target Endpoint:    127.0.0.1:14540",
            f"Command Msg ID:     #84 SET_POSITION_TARGET",
            f"Coordinate Frame:   1 (MAV_FRAME_LOCAL_NED)",
            f"Type Mask:          0x0DC7 (Velocity Only)",
            f"Offboard Heartbeat: 1 Hz (ONBOARD_CONTROLLER)",
            f"Dispatch Status:    {mav_status[:24]}",
        ]
        for line in mav_data:
            t_ren = self.font_small.render(line, True, TEXT_BRIGHT)
            self.screen.blit(t_ren, (x_pad, y))
            y += 18

        y += 15
        # Section: Traffic Tracking Table
        sec4 = self.font_header.render(f"--- TRAFFIC RADAR ({len(traffic)} TRACKED) ---", True, ACCENT_BLUE)
        self.screen.blit(sec4, (x_pad, y))
        y += 22

        table_hdr = self.font_small.render("CALLSIGN     RANGE    NORTH     EAST    V_GROUND", True, TEXT_DIM)
        self.screen.blit(table_hdr, (x_pad, y))
        y += 18

        for tr in traffic[:6]:
            d = math.hypot(tr.x - ego_pos[0], tr.y - ego_pos[1])
            spd = math.hypot(tr.vx, tr.vy)
            row_col = ALERT_RED if d < 50.0 else WARN_AMBER if d < 150.0 else TEXT_BRIGHT
            row_str = f"{tr.id:<10}  {d:4.0f}m  {tr.x:6.1f}  {tr.y:6.1f}  {spd:4.1f}m/s"
            self.screen.blit(self.font_small.render(row_str, True, row_col), (x_pad, y))
            y += 16

        pygame.display.flip()
        self.clock.tick(30)

        # Record frame for GIF artifact (downscaled for efficiency)
        if self.record_gif and self.frame_count < self.max_record_frames:
            # Capture every 2nd frame (5 Hz sampling of 10 Hz)
            if self.frame_count % 2 == 0:
                raw_str = pygame.image.tostring(self.screen, "RGB")
                img = Image.frombytes("RGB", (self.width, self.height), raw_str)
                # Resize to 800x533 to keep file size lightweight
                img_small = img.resize((800, 533), Image.Resampling.LANCZOS)
                self.recorded_frames.append(img_small)
            self.frame_count += 1

        return True

    def save_recording(self) -> Optional[str]:
        """Saves recorded frames into an animated GIF."""
        if not self.recorded_frames or not self.gif_output_path:
            return None

        print(f"[Visualizer] Exporting {len(self.recorded_frames)} frames to animated GIF: {self.gif_output_path}")
        os.makedirs(os.path.dirname(os.path.abspath(self.gif_output_path)), exist_ok=True)
        self.recorded_frames[0].save(
            self.gif_output_path,
            save_all=True,
            append_images=self.recorded_frames[1:],
            duration=150,  # ms per frame
            loop=0,
            optimize=True,
        )
        print("[Visualizer] GIF export complete!")
        return self.gif_output_path

    def close(self):
        self.save_recording()
        pygame.quit()
