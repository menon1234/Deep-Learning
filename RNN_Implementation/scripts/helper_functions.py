import numpy as np


def moving_average(data: np.ndarray, window_size: int) -> np.ndarray:
    """
    Compute the rolling moving average over a 1D array.

    Parameters:
    - data (np.ndarray): Input 1D array of data.
    - window_size (int): Size of the moving window.

    Returns:
    - np.ndarray: Smoothed array of the same length as input (padded at edges).
    """
    if window_size < 1:
        raise ValueError("window_size must be at least 1")

    # Use 'valid' to avoid edge effects, then pad to match input length
    cumsum = np.cumsum(np.insert(data, 0, 0))
    smoothed = (cumsum[window_size:] - cumsum[:-window_size]) / window_size

    # Pad result to match input length (centered)
    pad_left = window_size // 2
    pad_right = window_size - pad_left - 1
    return np.pad(smoothed, (pad_left, pad_right), mode="edge")


def compute_turn_speed(compass_deg: list[float], timestamps: np.ndarray) -> np.ndarray:
    # Convert degrees to radians
    compass_rad = np.deg2rad(compass_deg)

    # Represent compass angles as 2D unit vectors
    x = np.cos(compass_rad)
    y = np.sin(compass_rad)

    # Compute derivatives of x and y
    dx = np.diff(x)
    dy = np.diff(y)
    dt = np.diff(timestamps)
    dt[dt == 0] = 1e-3

    # Compute angular speed via cross product magnitude
    # Cross product in 2D is: x1*y2 - x2*y1 => gives scalar z-component
    angle_change = x[:-1] * dy - y[:-1] * dx
    turn_speed_rad = angle_change / dt
    turn_speed_deg = np.rad2deg(turn_speed_rad)
    # Prepend a zero to match input length
    return np.insert(turn_speed_deg, 0, 0.0)
