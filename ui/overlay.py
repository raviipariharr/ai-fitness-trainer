"""
ui/overlay.py

Purpose:
    All the "how it looks" code for the app's on-screen display, kept
    separate from main.py's control flow and separate from the
    exercise-tracking logic in exercises/. This module knows nothing
    about squats or push-ups — it only knows how to draw a rep count,
    a state badge, a progress bar, feedback text, and a menu, given
    plain values. That separation is what makes it possible to change
    the UI's look without touching exercise logic, and vice versa.

Design:
    A translucent dark panel in the top-left holds the exercise name,
    rep count, state badge, progress bar, and any feedback message. A
    translucent strip along the bottom shows the exercise menu, with
    the active exercise highlighted. Colors are centralized as module
    constants below so the whole look can be retuned from one place.
"""

import cv2

# --- Color palette (BGR, since OpenCV) ---
PANEL_BG_COLOR = (30, 30, 30)
PANEL_ALPHA = 0.6
TEXT_PRIMARY = (255, 255, 255)
TEXT_SECONDARY = (190, 190, 190)
ACCENT_COLOR = (80, 220, 130)       # progress fill, "good"/resting state
ACTIVE_STATE_COLOR = (0, 165, 255)  # mid-movement state (DOWN/CONTRACTED/OPEN)
WARNING_COLOR = (0, 165, 255)
DANGER_COLOR = (60, 60, 255)
MENU_BG_COLOR = (20, 20, 20)
MENU_ACTIVE_COLOR = (80, 220, 130)

# Positive feedback (a completed rep that met form/depth expectations)
# should read as reassurance, not a warning — everything else defaults
# to the warning color. Matched against the exact strings each
# exercise module emits (see exercises/pushup.py, bicep_curl.py).
POSITIVE_FEEDBACK_MESSAGES = {"Good rep", "Good depth"}

REST_STATES = {"UP", "EXTENDED", "CLOSED"}

PANEL_X = 15
PANEL_Y = 15
PANEL_WIDTH = 300


def rounded_rectangle(frame, top_left, bottom_right, radius, color, thickness=-1) -> None:
    """
    Draw a rectangle with rounded corners. OpenCV has no built-in for
    this, so it's built from a plain rectangle plus four corner
    circles (filled) or arcs (outline). thickness=-1 means filled.
    """
    x1, y1 = top_left
    x2, y2 = bottom_right
    radius = min(radius, (x2 - x1) // 2, (y2 - y1) // 2)

    if thickness < 0:
        cv2.rectangle(frame, (x1 + radius, y1), (x2 - radius, y2), color, -1)
        cv2.rectangle(frame, (x1, y1 + radius), (x2, y2 - radius), color, -1)
        for corner_x, corner_y in (
            (x1 + radius, y1 + radius), (x2 - radius, y1 + radius),
            (x1 + radius, y2 - radius), (x2 - radius, y2 - radius),
        ):
            cv2.circle(frame, (corner_x, corner_y), radius, color, -1)
    else:
        cv2.line(frame, (x1 + radius, y1), (x2 - radius, y1), color, thickness)
        cv2.line(frame, (x1 + radius, y2), (x2 - radius, y2), color, thickness)
        cv2.line(frame, (x1, y1 + radius), (x1, y2 - radius), color, thickness)
        cv2.line(frame, (x2, y1 + radius), (x2, y2 - radius), color, thickness)
        cv2.ellipse(frame, (x1 + radius, y1 + radius), (radius, radius), 180, 0, 90, color, thickness)
        cv2.ellipse(frame, (x2 - radius, y1 + radius), (radius, radius), 270, 0, 90, color, thickness)
        cv2.ellipse(frame, (x1 + radius, y2 - radius), (radius, radius), 90, 0, 90, color, thickness)
        cv2.ellipse(frame, (x2 - radius, y2 - radius), (radius, radius), 0, 0, 90, color, thickness)


def draw_translucent_rounded_panel(frame, top_left, bottom_right, color, alpha, radius=14) -> None:
    """
    Draw a rounded panel that blends into the video instead of
    stamping a hard-edged block over it. Drawn on a copy of the frame
    and blended back with cv2.addWeighted so the webcam feed stays
    visible underneath.
    """
    overlay = frame.copy()
    rounded_rectangle(overlay, top_left, bottom_right, radius, color, thickness=-1)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)


def draw_progress_bar(frame, x: int, y: int, width: int, height: int, progress: float) -> None:
    """
    Draw a horizontal progress bar: empty track, filled portion, thin
    border, and a centered percentage label. progress is clamped to
    [0, 1] defensively, since a live sensor reading can briefly
    overshoot either end.
    """
    progress = max(0.0, min(1.0, progress))

    rounded_rectangle(frame, (x, y), (x + width, y + height), radius=height // 2, color=(70, 70, 70))

    filled_width = int(width * progress)
    if filled_width > 2:
        rounded_rectangle(frame, (x, y), (x + filled_width, y + height), radius=height // 2, color=ACCENT_COLOR)

    rounded_rectangle(frame, (x, y), (x + width, y + height), radius=height // 2, color=(220, 220, 220), thickness=1)

    label = f"{int(progress * 100)}%"
    (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
    text_x = x + width // 2 - text_width // 2
    text_y = y + height // 2 + text_height // 2
    cv2.putText(frame, label, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, TEXT_PRIMARY, 1, cv2.LINE_AA)


def draw_state_badge(frame, x: int, y: int, state_text: str) -> None:
    """Draw a small colored pill showing the current exercise state."""
    color = ACCENT_COLOR if state_text in REST_STATES else ACTIVE_STATE_COLOR
    (text_width, text_height), _ = cv2.getTextSize(state_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    padding_x, padding_y = 10, 6
    badge_width = text_width + padding_x * 2
    badge_height = text_height + padding_y * 2

    rounded_rectangle(frame, (x, y), (x + badge_width, y + badge_height), radius=badge_height // 2, color=color)
    text_x = x + padding_x
    text_y = y + badge_height - padding_y - 1
    cv2.putText(frame, state_text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (10, 10, 10), 1, cv2.LINE_AA)


def draw_status_panel(frame, display_name: str, status: dict) -> None:
    """
    Draw the main HUD panel: exercise name, rep count, state badge,
    progress bar, and (if present) a detail line and feedback message.
    """
    panel_height = 195
    draw_translucent_rounded_panel(
        frame, (PANEL_X, PANEL_Y), (PANEL_X + PANEL_WIDTH, PANEL_Y + panel_height),
        PANEL_BG_COLOR, PANEL_ALPHA,
    )

    inner_x = PANEL_X + 18
    cursor_y = PANEL_Y + 30

    cv2.putText(frame, display_name, (inner_x, cursor_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, TEXT_PRIMARY, 2, cv2.LINE_AA)
    cursor_y += 45

    cv2.putText(frame, str(status["rep_count"]), (inner_x, cursor_y),
                cv2.FONT_HERSHEY_SIMPLEX, 1.3, ACCENT_COLOR, 3, cv2.LINE_AA)
    cv2.putText(frame, "reps", (inner_x + 70, cursor_y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, TEXT_SECONDARY, 1, cv2.LINE_AA)
    draw_state_badge(frame, PANEL_X + PANEL_WIDTH - 110, cursor_y - 28, status["state"])
    cursor_y += 25

    draw_progress_bar(frame, inner_x, cursor_y, PANEL_WIDTH - 36, 22, status["progress"])
    cursor_y += 40

    if not status["landmarks_visible"]:
        cv2.putText(frame, "Body not clearly visible", (inner_x, cursor_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, DANGER_COLOR, 1, cv2.LINE_AA)
        cursor_y += 24
    elif status.get("detail"):
        cv2.putText(frame, status["detail"], (inner_x, cursor_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, TEXT_SECONDARY, 1, cv2.LINE_AA)
        cursor_y += 24

    if status.get("feedback"):
        feedback_color = ACCENT_COLOR if status["feedback"] in POSITIVE_FEEDBACK_MESSAGES else WARNING_COLOR
        wrapped_feedback = _wrap_text(status["feedback"], max_chars=32)
        for line in wrapped_feedback:
            cv2.putText(frame, line, (inner_x, cursor_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, feedback_color, 1, cv2.LINE_AA)
            cursor_y += 20


def draw_menu(frame, exercise_registry: dict, selection_keys: dict, active_key: str) -> None:
    """
    Draw the exercise-selection menu as a translucent strip along the
    bottom of the frame, one entry per registered exercise, with the
    active one highlighted. Built entirely from exercise_registry (key
    -> tracker class, each with a display_name) and selection_keys
    (keypress -> registry key) — this function has no idea how many
    exercises exist or what they're called; it just reads both dicts.
    """
    frame_height, frame_width = frame.shape[:2]
    row_height = 26
    menu_height = len(exercise_registry) * row_height + 16
    top_y = frame_height - menu_height

    draw_translucent_rounded_panel(
        frame, (PANEL_X, top_y), (PANEL_X + 230, frame_height - 15),
        MENU_BG_COLOR, PANEL_ALPHA, radius=10,
    )

    key_by_registry_key = {registry_key: chr(key_code) for key_code, registry_key in selection_keys.items()}
    text_y = top_y + 22

    for registry_key, tracker_class in exercise_registry.items():
        key_label = key_by_registry_key.get(registry_key, "?")
        is_active = registry_key == active_key
        color = MENU_ACTIVE_COLOR if is_active else TEXT_SECONDARY
        prefix = ">" if is_active else " "
        label = f"{prefix} [{key_label}] {tracker_class.display_name}"
        cv2.putText(frame, label, (PANEL_X + 15, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
        text_y += row_height


def _wrap_text(text: str, max_chars: int) -> list:
    """Simple word wrap so long feedback messages don't run off-frame."""
    words = text.split()
    lines, current_line = [], ""
    for word in words:
        candidate = f"{current_line} {word}".strip()
        if len(candidate) > max_chars and current_line:
            lines.append(current_line)
            current_line = word
        else:
            current_line = candidate
    if current_line:
        lines.append(current_line)
    return lines


def render_hud(frame, display_name: str, status: dict, exercise_registry: dict, selection_keys: dict, active_key: str) -> None:
    """
    Draw the full HUD onto frame, in place: status panel (top-left)
    plus exercise menu (bottom-left). This is the one function main.py
    calls each frame — everything else in this module is a private
    building block for it.
    """
    draw_status_panel(frame, display_name, status)
    draw_menu(frame, exercise_registry, selection_keys, active_key)