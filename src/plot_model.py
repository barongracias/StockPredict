# imports
import pandas as pd
import matplotlib.pyplot as plt

# project imports
from src.paths import RESULTS_DIR
from src.logger import get_console_logger

# log run
logger = get_console_logger()

class ModelPlotter:
    def __init__(self) -> None:
        pass

    # residual plot
    def plot_residuals(self, y_test: pd.Series, predictions: pd.Series, model: str) -> plt.Figure:
        residuals = y_test - predictions
        fig, ax = plt.subplots()
        ax.scatter(predictions, residuals)
        ax.axhline(0, color='red', linestyle='--')
        ax.set_title(f'{model} residual plot')
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Residuals')
        return fig
    
    # actual vs predicted plot
    def plot_actual_predicted(self, y_test: pd.Series, predictions: pd.Series, model: str) -> plt.Figure:
        fig, ax = plt.subplots()
        ax.scatter(y_test, predictions, alpha=0.5)
        ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        ax.set_title(f'{model} Actual vs Predicted')
        ax.set_xlabel('Actual Values')
        ax.set_ylabel('Predicted Values')
        return fig
    
    # learning curve
    def plot_learning_curve(self, train_sizes, train_scores, test_scores, model: str) -> plt.Figure:
        fig, ax = plt.subplots()
        
        train_scores_mean = train_scores.mean(axis=0) if train_scores.ndim > 1 else train_scores
        test_scores_mean = test_scores.mean(axis=0) if test_scores.ndim > 1 else test_scores
    
        ax.plot(train_sizes, train_scores_mean, 'o-', color='r', label='Training score')
        ax.plot(train_sizes, test_scores_mean, 'o-', color='g', label='Cross-validation score')
        ax.set_title(f'{model} Learning Curve')
        ax.set_xlabel('Training Size')
        ax.set_ylabel('Score')
        ax.legend()
        return fig
    
    # function to save plot
    def save_plot(self, fig: plt.Figure, filename: str) -> None:
        try:
            fig.savefig(RESULTS_DIR / filename, bbox_inches='tight')
            logger.info(f'Saved plot: {filename}')
        except Exception as e:
            logger.error(f'Error saving plot {filename}: {e}')
        finally:
            plt.close(fig)