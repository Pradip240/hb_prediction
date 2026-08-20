import numpy as np
from numpy.typing import NDArray


class OneEuroFilter:
    """
    One-Euro adaptive low-pass filter for temporal smoothing.

    The filter reduces frame-to-frame jitter while remaining responsive to genuine motion.
    Slow-moving signals are smoothed aggressively, whereas rapidly changing signals are smoothed
    less by increasing the cutoff frequency.

    This implementation operates on NumPy arrays, allowing all landmark
    coordinates (or mask pixels) to be filtered simultaneously.
    """

    def __init__(self, freq: float, min_cutoff: float = 0.1, beta: float = 0.005, d_cutoff: float = 1.0) -> None:
        """
        Initialize the One-Euro filter.

        Args:
            freq: Sampling frequency in Hz (typically the video frame rate).
            min_cutoff:Minimum cutoff frequency. Lower values produce stronger
                smoothing when the signal is nearly stationary.
            beta: Speed adaptation coefficient. Larger values reduce smoothing
                as the signal moves faster.
            d_cutoff: Cutoff frequency used to smooth the estimated signal velocity.
        """
        self.freq = float(freq)
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        self.x_prev: NDArray[np.float64] | None = None
        self.dx_prev: NDArray[np.float64] | None = None


    def _smoothing_factor(self, cutoff: float | NDArray[np.float64]) -> float | NDArray[np.float64]:
        """
        Compute the exponential smoothing factor.

        Args:
            cutoff: Low-pass filter cutoff frequency in Hz.

        Returns:
            Smoothing factor in the range (0, 1].
        """
        tau = 1.0 / (2.0 * np.pi * cutoff)
        sample_interval = 1.0 / self.freq
        return 1.0 / (1.0 + tau / sample_interval)


    def reset(self) -> None:
        """
        Reset the filter state.

        The next sample is treated as the start of a new sequence.
        """
        self.x_prev = None
        self.dx_prev = None


    def __call__(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Filter a single sample.

        Args:
            x: Current sample. May contain one or more values.

        Returns:
            Smoothed sample with the same shape as the input.
        """
        # Initialize the filter from the first sample
        if (self.x_prev is None) or (self.dx_prev is None):
            self.x_prev = x.copy()
            self.dx_prev = np.zeros_like(x)
            return x.copy()

        # Estimate signal velocity (units per second)
        dx = (x - self.x_prev) * self.freq

        # Smooth the velocity estimate
        velocity_factor = self._smoothing_factor(self.d_cutoff)
        dx_hat = (velocity_factor * dx + (1.0 - velocity_factor) * self.dx_prev)

        # Increase the cutoff frequency as motion speed increases
        velocity = np.linalg.norm(dx_hat, axis=-1, keepdims=True)
        cutoff = self.min_cutoff + self.beta * velocity

        # Smooth the signal using the adaptive cutoff
        signal_factor = self._smoothing_factor(cutoff)
        x_hat = (signal_factor * x + (1.0 - signal_factor) * self.x_prev)

        # Store the current filter state
        self.x_prev = x_hat.copy()
        self.dx_prev = dx_hat.copy()

        return x_hat
