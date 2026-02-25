import logging
import threading
import time
from collections.abc import Iterable
from concurrent.futures import TimeoutError, CancelledError
from concurrent.futures._base import CANCELLED, FINISHED, RUNNING
from typing import Any, Dict, List, Optional, Tuple, Union, Callable

from odemis import model
from odemis.model import InstantaneousFuture
from odemis.util import executeAsyncTask, almost_equal

from odemis.acq.align.goffset import SparcAutoGratingOffset


def _mapDetectorToSelector(selector: model.Actuator,
                           detectors: List[model.Detector]) -> Tuple[str, Dict[str, Any]]:
    det_2_sel = {}
    sel_axis = None

    for an, ad in selector.axes.items():
        if hasattr(ad, "choices") and isinstance(ad.choices, dict):
            sel_axis = an
            for pos, value in ad.choices.items():
                for d in detectors:
                    if d.name in value:
                        det_2_sel[d] = pos

            if det_2_sel:
                break

    if len(det_2_sel) < len(detectors):
        raise ValueError("Failed to find all detectors (%s) in positions of selector axes %s" %
                         (", ".join(d.name for d in detectors), list(selector.axes.keys())))

    return sel_axis, det_2_sel

def _checkCancelled(future: "model.ProgressiveFuture"):
    """
    Raise CancelledError if the given future has been cancelled.
    """
    # Try centering first, then align lock
    lock_name = "_centering_lock" if hasattr(future, "_centering_lock") else "_align_lock"
    state_name = "_centering_state" if hasattr(future, "_centering_state") else "_align_state"

    lock = getattr(future, lock_name)
    state = getattr(future, state_name)

    with lock:
        if state == CANCELLED:
            raise CancelledError()


def _totalAlignmentTime(n_gratings: int,
                        n_detectors: int) -> float:
    runs = n_detectors + max(0, n_gratings - 1)
    move_time = ((n_gratings - 1) * MOVE_TIME_GRATING + (n_detectors - 1) * MOVE_TIME_DETECTOR)
    return runs * EST_ALIGN_TIME + move_time


def AutoAlignGratingDetectorOffsets(spectrograph: model.Actuator,
                                    detectors: Union[model.Detector, List[model.Detector]],
                                    selector: Optional[model.Actuator] = None,
                                    streams: Optional[List['Stream']] = None) -> model.ProgressiveFuture:
    if not isinstance(detectors, Iterable):
        detectors = [detectors]
    if not detectors:
        raise ValueError("At least one detector must be provided")
    if len(detectors) > 1 and selector is None:
        raise ValueError("No selector provided, but multiple detectors")

    if streams is None:
        streams = []

    est_start = time.time() + 0.1
    n_gratings = len(spectrograph.axes["grating"].choices)
    n_detectors = len(detectors)
    a_time = _totalAlignmentTime(n_gratings, n_detectors)
    f = model.ProgressiveFuture(start=est_start, end=est_start + a_time)
    f.task_canceller = _CancelAutoAlignGratingDetectorOffsets

    f._align_state = RUNNING
    f._align_lock = threading.Lock()
    f._subfuture = InstantaneousFuture()
    executeAsyncTask(f, _DoAutoAlignGratingDetectorOffsets, args=(f, spectrograph, detectors, selector, streams))
    return f


MOVE_TIME_GRATING = 20
MOVE_TIME_DETECTOR = 5
EST_ALIGN_TIME = 30

def _DoAutoAlignGratingDetectorOffsets(future: model.ProgressiveFuture,
                                       spectrograph: model.Actuator,
                                       detectors: List[model.Detector],
                                       selector: Optional[model.Actuator],
                                       streams: List['Stream'],
                                       stabilization_time: float = 20.0) -> Optional[Dict[Any, Any]]:
    results: dict[tuple, bool] = {}
    original_pos = {k: v for k, v in spectrograph.position.value.items()
                    if k in ("wavelength", "grating")}

    gratings = sorted(spectrograph.axes["grating"].choices.keys(),
                      key=lambda g: 0 if g == original_pos["grating"] else 1)
    logging.info(f"Available gratings: {list(spectrograph.axes['grating'].choices.keys())}")

    first_detector = detectors[0]

    if selector:
        original_selector = selector.position.value
        selector_axes, detector_to_selector = _mapDetectorToSelector(selector, detectors)

    def is_current_detector(d):
        if selector is None:
            return True
        return detector_to_selector[d] == selector.position.value[selector_axes]

    try:
        g0 = gratings[0]
        logging.info(f"Starting alignment for initial grating: {g0}")
        spectrograph.moveAbsSync({"grating": g0, "wavelength": 0})
        time.sleep(stabilization_time)

        detectors_sorted = sorted(detectors, key=is_current_detector, reverse=True)

        for d in detectors_sorted:
            _checkCancelled(future)
            logging.info(f"Starting alignment | Detector: {d.name} | Grating: {g0}")
            print(f"Starting alignment | Detector: {d.name} | Grating: {g0}")

            if selector:
                selector.moveAbsSync({selector_axes: detector_to_selector[d]})
                future._subfuture = SparcAutoGratingOffset(spectrograph, d)
                success = future._subfuture.result()
                results[(g0, d.name)] = success
                logging.info(f"Finished alignment | Detector: {d.name} | Grating: {g0} | Success: {success}")
                print(f"Finished alignment | Detector: {d.name} | Grating: {g0} | Success: {success}")
        if selector:
            selector.moveAbsSync({selector_axes: detector_to_selector[first_detector]})

        for g in gratings[1:]:
            _checkCancelled(future)
            print(f"Switching to grating: {g}")
            logging.info(f"Switching to grating: {g}")
            spectrograph.moveAbsSync({"grating": g, "wavelength": 0})
            time.sleep(stabilization_time)

            print(f"Starting alignment | Detector: {first_detector.name} | Grating: {g}")
            logging.info(f"Starting alignment | Detector: {first_detector.name} | Grating: {g}")
            future._subfuture = SparcAutoGratingOffset(spectrograph, first_detector)
            success = future._subfuture.result()
            results[(g, first_detector.name)] = success
            print(f"Finished alignment | Detector: {first_detector.name} | Grating: {g} | Success: {success}")
            logging.info(f"Finished alignment | Detector: {first_detector.name} | Grating: {g} | Success: {success}")

        return results

    except CancelledError:
        logging.info("Auto-alignment cancelled")
        raise

    finally:
        spectrograph.moveAbsSync(original_pos)
        if selector:
            selector.moveAbsSync(original_selector)

        with future._align_lock:
            future._align_state = FINISHED


def _CancelAutoAlignGratingDetectorOffsets(future: model.ProgressiveFuture) -> bool:
    logging.debug("Cancelling autoalignment...")

    with future._align_lock:
        if future._align_state == FINISHED:
            return False
        future._align_state = CANCELLED
        future._subfuture.cancel()
        logging.debug("Auto-alignment cancellation requested")

    return True