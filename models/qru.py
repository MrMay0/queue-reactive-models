from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from models.common import BaseQRSimulator, CalibrationResult, calibrate_common


@dataclass
class QRUCalibration:
    common: CalibrationResult
    unit_size: int

    @property
    def intensity_df(self) -> pd.DataFrame:
        return self.common.intensity_df


def calibrate_qru(
    event_flow_path: str,
    raw_dir: str = "data/raw",
    level: int = 1,
    min_obs: int = 50,
    common: CalibrationResult | None = None,
) -> QRUCalibration:
    if common is None:
        common = calibrate_common(event_flow_path, raw_dir=raw_dir, level=level, min_obs=min_obs)
    unit_size = int(np.ceil(common.aes_level))
    return QRUCalibration(common=common, unit_size=unit_size)


class QRUSimulator(BaseQRSimulator):
    def __init__(self, calibration: QRUCalibration):
        super().__init__(calibration.common)
        self.qru_calibration = calibration

    def _rate_table(self, n: int) -> pd.Series:
        row = self.calibration.intensity_df.set_index("n").loc[n]
        return row[["lambda_L", "lambda_C", "lambda_M"]]

    def _sample_event(self, n: int, queue_size: int, rng) -> tuple[str, int]:
        rates = self._rate_table(n)
        eta = str(rng.choice(["L", "C", "M"], p=(rates / rates.sum()).to_numpy()))
        return eta, self.qru_calibration.unit_size
