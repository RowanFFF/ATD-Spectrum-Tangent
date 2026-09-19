import numpy as np


class StreamingMetric:
    """Accumulate the standard forecasting metrics without storing forecasts."""

    def __init__(self):
        self.count = 0
        self.absolute_error = 0.0
        self.squared_error = 0.0
        self.absolute_percentage_error = 0.0
        self.squared_percentage_error = 0.0

    def update(self, pred, true):
        pred = np.asarray(pred)
        true = np.asarray(true)
        if pred.shape != true.shape:
            raise ValueError('pred and true must have matching shapes')
        error = true - pred
        self.count += error.size
        self.absolute_error += np.sum(np.abs(error), dtype=np.float64)
        self.squared_error += np.sum(np.square(error), dtype=np.float64)
        with np.errstate(divide='ignore', invalid='ignore'):
            percentage_error = error / true
            self.absolute_percentage_error += np.sum(
                np.abs(percentage_error), dtype=np.float64)
            self.squared_percentage_error += np.sum(
                np.square(percentage_error), dtype=np.float64)

    def compute(self):
        if self.count == 0:
            raise ValueError('at least one prediction is required')
        mae = self.absolute_error / self.count
        mse = self.squared_error / self.count
        rmse = np.sqrt(mse)
        mape = self.absolute_percentage_error / self.count
        mspe = self.squared_percentage_error / self.count
        return mae, mse, rmse, mape, mspe


def RSE(pred, true):
    return np.sqrt(np.sum((true - pred) ** 2)) / np.sqrt(np.sum((true - true.mean()) ** 2))


def CORR(pred, true):
    u = ((true - true.mean(0)) * (pred - pred.mean(0))).sum(0)
    d = np.sqrt(((true - true.mean(0)) ** 2 * (pred - pred.mean(0)) ** 2).sum(0))
    return (u / d).mean(-1)


def MAE(pred, true):
    return np.mean(np.abs(true - pred))


def MSE(pred, true):
    return np.mean((true - pred) ** 2)


def RMSE(pred, true):
    return np.sqrt(MSE(pred, true))


def MAPE(pred, true):
    return np.mean(np.abs((true - pred) / true))


def MSPE(pred, true):
    return np.mean(np.square((true - pred) / true))


def metric(pred, true):
    mae = MAE(pred, true)
    mse = MSE(pred, true)
    rmse = RMSE(pred, true)
    mape = MAPE(pred, true)
    mspe = MSPE(pred, true)

    return mae, mse, rmse, mape, mspe
