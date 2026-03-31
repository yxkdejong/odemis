# -*- coding: utf-8 -*-
"""
Stabilization test for hardware.
Switches gratings, tracks peak position, and comprehensively tracks absolute time,
elapsed time, and hardware movement time.
"""

import logging
import time
import datetime
import matplotlib.pyplot as plt
import numpy as np
import json
import csv

from odemis import model
from odemis.acq.align.goffset import find_peak_position

logging.getLogger().setLevel(logging.DEBUG)

def acquire_stabilization_curve(detector, n_samples=100, sample_delay=0.1):
    """
    Acquire a rapid sequence of spectra and return (abs_time, t_elapsed, intensity, peak_px).
    """
    t0 = time.time()
    curve = []

    for i in range(n_samples):
        data = detector.data.get(asap=False)
        spectrum = data.mean(axis=0)
        intensity = float(spectrum.max())

        # Try to detect peak
        try:
            peak_px = float(find_peak_position(data))
        except Exception:
            peak_px = np.nan

        # Track BOTH absolute time and elapsed time
        abs_time = time.time()
        t_elapsed = abs_time - t0

        logging.debug("[sample %02d] t=%.4fs intensity=%.1f peak=%s",
                      i, t_elapsed, intensity, peak_px)

        curve.append((abs_time, t_elapsed, intensity, peak_px))
        time.sleep(sample_delay)

    return np.array(curve)


def compute_stabilization_time(curve, tolerance_px=0.6, window=10):
    """
    Compute stabilization time based on peak position.
    """
    if curve.size == 0:
        return np.nan

    t_elapsed = curve[:, 1]  # Index 1 is elapsed time
    peak_px = curve[:, 3]    # Index 3 is peak position

    if np.isnan(peak_px).all():
        return np.nan

    # Use the last `window` valid samples to estimate the final peak
    valid_mask = ~np.isnan(peak_px)
    valid_peaks = peak_px[valid_mask]
    if valid_peaks.size == 0:
        return np.nan

    if valid_peaks.size < window:
        final_peak = np.nanmean(valid_peaks)
    else:
        final_peak = np.nanmean(valid_peaks[-window:])

    deviation = np.abs(peak_px - final_peak)

    max_start = len(peak_px) - window
    if max_start < 0:
        return np.nan

    for i in range(0, max_start + 1):
        window_devs = deviation[i:i + window]
        if np.all(window_devs <= tolerance_px):
            # Return the time of the last sample in the stabilizing window
            return float(t_elapsed[i + window - 1])

    return np.nan


def measure_stabilization(spectrograph, detector, gratings, wavelength,
                          n_cycles=10, n_samples=100, sample_delay=0.1):
    results = []

    for i in range(n_cycles):
        g = gratings[i % len(gratings)]   # Alternate between gratings

        logging.info("=== Cycle %d/%d — switching to grating %s ===",
                     i + 1, n_cycles, g)

        before = spectrograph.position.value.get("grating") if hasattr(spectrograph, "position") else None
        logging.debug("Before move: grating=%s", before)

        # Track how long the motor takes to move
        move_start_time = time.time()
        spectrograph.moveAbsSync({"grating": g, "wavelength": wavelength})
        move_duration = time.time() - move_start_time

        after = spectrograph.position.value.get("grating") if hasattr(spectrograph, "position") else None
        logging.debug("After move: grating=%s", after)

        logging.info("Move completed in %.2f seconds — starting acquisition loop", move_duration)
        curve = acquire_stabilization_curve(detector, n_samples=n_samples, sample_delay=sample_delay)

        results.append({
            "curve": curve,
            "grating": g,
            "cycle": i + 1,
            "move_time_s": move_duration,
            "stabilization_time": compute_stabilization_time(curve)
        })

    return results


def save_measurement_results(results, basename="stabilization_results"):
    """
    Save a simple summary CSV and per-cycle CSVs plus an NPZ of curves.
    """
    # Summary CSV
    summary_filename = f"{basename}_summary.csv"
    with open(summary_filename, "w", newline="") as sf:
        writer = csv.writer(sf)
        writer.writerow(["cycle", "grating", "move_time_s", "stabilization_time_s"])
        for r in results:
            stab = r["stabilization_time"]
            stab_val = "" if (stab is None or (isinstance(stab, float) and np.isnan(stab))) else float(stab)
            writer.writerow([r["cycle"], r["grating"], r["move_time_s"], stab_val])

    # Per-cycle CSVs and NPZ collection
    npz_dict = {}
    for r in results:
        cycle = r["cycle"]
        curve = r["curve"]
        csv_filename = f"{basename}_cycle_{cycle}.csv"
        with open(csv_filename, "w", newline="") as cf:
            writer = csv.writer(cf)
            writer.writerow(["abs_timestamp", "t_elapsed", "intensity", "peak_px"])
            for row in curve:
                writer.writerow(row.tolist() if isinstance(row, np.ndarray) else row)
        npz_dict[f"curve_{cycle}"] = curve

    npz_filename = f"{basename}_curves.npz"
    np.savez(npz_filename, **npz_dict)

    logging.info("Saved results: %s (summary), %s (curves NPZ), %s (per-cycle CSVs)",
                 summary_filename, npz_filename, f"{basename}_cycle_*.csv")


if __name__ == "__main__":
    spgr = model.getComponent(role="spectrograph")
    det = model.getComponent(role="ccd")

    logging.info("Loaded spectrograph: %s", spgr)
    logging.info("Loaded detector: %s", det)

    gratings = list(spgr.axes["grating"].choices.keys())
    logging.info("Available gratings: %s", gratings)

    wavelength = 0
    curves = measure_stabilization(spgr, det, gratings=gratings, wavelength=wavelength,
                                   n_cycles=10, n_samples=100, sample_delay=0.1)

    # Automatically append the current date/time to the file so you never overwrite data
    timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    file_basename = f"stabilization_run_{timestamp_str}"

    # Save results for later analysis
    save_measurement_results(curves, basename=file_basename)

    # Quick plot of stabilization times
    # Automatically generate and save the bar chart
    try:
        times = [r["stabilization_time"] for r in curves]

        # Create a 0-based index for the x-axis to match your image (0, 1, 2, 3...)
        x_indices = range(len(times))

        plt.figure(figsize=(8, 4))  # Makes the aspect ratio match your image
        plt.bar(x_indices, times)

        plt.xlabel("Cycle index")
        plt.ylabel("Stabilization time [s]")
        plt.title("Stabilization Times Across Cycles")

        plt.xticks(x_indices)  # Ensure there is a tick label under every bar
        plt.grid(True)

        # Save the plot as a PNG file automatically
        plt.savefig("stabilization time per grating-detector switch.pdf")

        # Close the figure to free up memory
        plt.close()

    except Exception as e:
        logging.error("Plotting or saving failed: %s", e)
