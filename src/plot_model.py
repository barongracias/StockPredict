import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.paths import RESULTS_DIR
from src.logger import get_console_logger

logger = get_console_logger()


class ModelPlotter:
    def plot_residuals(self, y_test: pd.Series, predictions: np.ndarray, model: str) -> plt.Figure:
        residuals = y_test - predictions
        fig, ax = plt.subplots()
        ax.scatter(predictions, residuals)
        ax.axhline(0, color='red', linestyle='--')
        ax.set_title(f'{model} Residuals')
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Residuals')
        return fig

    def plot_actual_predicted(self, y_test: pd.Series, predictions: np.ndarray, model: str) -> plt.Figure:
        fig, ax = plt.subplots()
        ax.scatter(y_test, predictions, alpha=0.5)
        ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        ax.set_title(f'{model} Actual vs Predicted')
        ax.set_xlabel('Actual')
        ax.set_ylabel('Predicted')
        return fig

    def plot_learning_curve(
        self,
        train_sizes: np.ndarray,
        train_scores: np.ndarray,
        test_scores: np.ndarray,
        model: str,
    ) -> plt.Figure:
        fig, ax = plt.subplots()
        ax.plot(train_sizes, train_scores, 'o-', color='r', label='Training MAE')
        ax.plot(train_sizes, test_scores, 'o-', color='g', label='CV MAE')
        ax.set_title(f'{model} Learning Curve')
        ax.set_xlabel('Training Size')
        ax.set_ylabel('MAE')
        ax.legend()
        return fig

    def save_plot(self, fig: plt.Figure, filename: str) -> None:
        try:
            fig.savefig(RESULTS_DIR / filename, bbox_inches='tight')
            logger.info(f'Saved plot: {filename}')
        except Exception as e:
            logger.error(f'Error saving plot {filename}: {e}')
        finally:
            plt.close(fig)
