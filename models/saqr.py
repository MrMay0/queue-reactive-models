from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from models.common import BaseQRSimulator, CalibrationResult, calibrate_common


@dataclass
class SAQRCalibration:
    common: CalibrationResult
    smoothing_alpha: float = 25.0

    @property
    def intensity_df(self) -> pd.DataFrame:
        return self.common.intensity_df

    @property
    def joint_size_df(self) -> pd.DataFrame:
        return self.common.joint_size_df


def calibrate_saqr(
    event_flow_path: str,
    raw_dir: str = "data/raw",
    level: int = 1,
    min_obs: int = 50,
    smoothing_alpha: float = 25.0,
    common: CalibrationResult | None = None,
) -> SAQRCalibration:
    if common is None:
        common = calibrate_common(event_flow_path, raw_dir=raw_dir, level=level, min_obs=min_obs)
    return SAQRCalibration(common=common, smoothing_alpha=smoothing_alpha)


class SAQRSimulator(BaseQRSimulator):
    def __init__(self, calibration: SAQRCalibration):
        super().__init__(calibration.common)
        self.saqr_calibration = calibration

    def _rate_table(self, n: int) -> pd.Series:
        row = self.calibration.intensity_df.set_index("n").loc[n]
        return row[["lambda_L", "lambda_C", "lambda_M"]]

    def _sample_event(self, n: int, queue_size: int, rng) -> tuple[str, int]:
        return self.calibration.joint_sample(
            n,
            rng,
            smoothing_alpha=self.saqr_calibration.smoothing_alpha,
        )
