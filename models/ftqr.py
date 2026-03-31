from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from models.common import BaseQRSimulator, CalibrationResult, calibrate_common


@dataclass
class FTQRCalibration:
    common: CalibrationResult

    @property
    def intensity_df(self) -> pd.DataFrame:
        return self.common.ft_intensity_df


def calibrate_ftqr(
    event_flow_path: str,
    raw_dir: str = "data/raw",
    level: int = 1,
    min_obs: int = 50,
    common: CalibrationResult | None = None,
) -> FTQRCalibration:
    if common is None:
        common = calibrate_common(event_flow_path, raw_dir=raw_dir, level=level, min_obs=min_obs)
    return FTQRCalibration(common=common)


class FTQRSimulator(BaseQRSimulator):
    def __init__(self, calibration: FTQRCalibration):
        super().__init__(calibration.common)

    def _rate_table(self, n: int) -> pd.Series:
        row = self.calibration.ft_intensity_df.set_index("n").loc[n]
        return row[["lambda_L", "lambda_C", "lambda_M", "lambda_C_all", "lambda_M_all"]]

    def _sample_event(self, n: int, queue_size: int, rng) -> tuple[str, int]:
        rates = self._rate_table(n)
        eta = str(
            rng.choice(
                ["L", "C", "M", "C_all", "M_all"],
                p=(rates / rates.sum()).to_numpy(),
            )
        )
        if eta in {"C_all", "M_all"}:
            return eta, int(queue_size)
        size = self.calibration.eta_size_sample("L" if eta == "L" else eta, rng)
        return eta, size
