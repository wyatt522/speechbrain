import torch
import torch.nn as nn
import numpy as np
import pyrubberband as pyrb

class MultiPitchShift(nn.Module):
    """
    Applies multiple random pitch shifts to the input waveform and overlays
    them on top of each other.

    Arguments
    ---------
    sample_rate : int
        The sample rate of the input waveforms.
    n_steps_range : tuple
        A (min, max) tuple defining the range (in semitones) from which the
        pitch shifts are randomly sampled.
    num_shifts : int
        Number of pitch-shifted versions to overlay onto the original signal.

    Example
    -------
    >>> waveform = torch.randn(2, 16000) # Batch=2, Time=16000
    >>> multi_pitch = MultiPitchShift(sample_rate=16000, n_steps_range=(-2, 2), num_shifts=3)
    >>> augmented = multi_pitch(waveform)
    """

    def __init__(self, sample_rate, n_steps_range=(-2, 2), num_shifts=5):
        super().__init__()
        self.sample_rate = sample_rate
        self.n_steps_range = n_steps_range
        self.num_shifts = num_shifts

    def forward(self, waveforms):
        """
        Arguments
        ---------
        waveforms : torch.Tensor
            Shape: [batch, time] or [batch, time, channels]

        Returns
        -------
        torch.Tensor
            Augmented waveform with overlaid pitch shifts.
        """
        # Ensure waveforms have a channel dimension
        # We'll assume final shape is [batch, time, channels]
        if waveforms.dim() == 2:
            waveforms = waveforms.unsqueeze(-1)

        batch_size, time, channels = waveforms.shape
        device = waveforms.device
        dtype = waveforms.dtype

        # Move to CPU for pyrubberband
        waveforms_np = waveforms.detach().cpu().numpy()

        augmented = np.zeros_like(waveforms_np)
        for b in range(batch_size):
            for c in range(channels):
                original = waveforms_np[b, :, c]
                # Generate random pitch shifts
                pitch_shifts = np.random.uniform(self.n_steps_range[0], self.n_steps_range[1], self.num_shifts)
                overlayed = np.zeros_like(original)
                for steps in pitch_shifts:
                    shifted = pyrb.pitch_shift(original, self.sample_rate, steps)
                    # If shifted length differs due to some resampling boundary conditions, match the length
                    if len(shifted) > len(original):
                        shifted = shifted[:len(original)]
                    elif len(shifted) < len(original):
                        temp = np.zeros_like(original)
                        temp[:len(shifted)] = shifted
                        shifted = temp
                    overlayed += shifted
                # Normalize overlayed waveform
                overlayed /= self.num_shifts
                augmented[b, :, c] = overlayed

        augmented = torch.tensor(augmented, device=device, dtype=dtype)
        # If we had added a channel dimension, we may remove it if original was 2D
        return augmented.squeeze(-1) if channels == 1 else augmented


class VariablePitchShiftSmooth(nn.Module):
    """
    Applies a random continuously varying pitch shift effect to the input
    waveform. The pitch shift changes smoothly over time, and short fades
    at segment boundaries are applied to avoid artifacts.

    Arguments
    ---------
    sample_rate : int
        The sample rate of the input waveforms.
    n_steps_range : tuple
        A (min, max) tuple defining the range (in semitones) from which
        the pitch shift envelope is sampled.
    fade_duration : float
        Fade-in/out duration in seconds at each segment boundary to
        avoid clicks/pops.
    segment_duration : float
        Duration of each segment (in seconds) over which a uniform pitch
        shift is applied before moving to the next segment with a slightly
        different pitch.
    n_envelope_points : int
        Number of random pitch points used to create a smooth pitch shift
        envelope over time.

    Example
    -------
    >>> waveform = torch.randn(1, 16000) # Single waveform
    >>> var_pitch = VariablePitchShiftSmooth(sample_rate=16000, n_steps_range=(-2, 2), fade_duration=0.01)
    >>> augmented = var_pitch(waveform)
    """

    def __init__(
        self,
        sample_rate,
        n_steps_range=(-2, 2),
        fade_duration=0.01,
        segment_duration=0.1,
        n_envelope_points=100
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.n_steps_range = n_steps_range
        self.fade_duration = fade_duration
        self.segment_duration = segment_duration
        self.n_envelope_points = n_envelope_points

    def forward(self, waveforms):
        """
        Arguments
        ---------
        waveforms : torch.Tensor
            Shape: [batch, time] or [batch, time, channels]

        Returns
        -------
        torch.Tensor
            Continuously pitch-shifted waveform.
        """
        # Ensure channel dimension
        if waveforms.dim() == 2:
            waveforms = waveforms.unsqueeze(-1)

        batch_size, time, channels = waveforms.shape
        device = waveforms.device
        dtype = waveforms.dtype

        waveforms_np = waveforms.detach().cpu().numpy()
        augmented = np.zeros_like(waveforms_np)

        fade_samples = int(self.fade_duration * self.sample_rate)
        segment_samples = int(self.segment_duration * self.sample_rate)
        total_samples = time
        duration = total_samples / self.sample_rate

        # Generate a smooth pitch shift envelope
        # We'll do this per-batch element for more variability.
        for b in range(batch_size):
            for c in range(channels):
                original = waveforms_np[b, :, c]

                # Random envelope of pitch shifts
                random_steps = np.random.uniform(self.n_steps_range[0], self.n_steps_range[1], size=self.n_envelope_points)
                times = np.linspace(0, duration, num=total_samples)
                smooth_random_steps = np.interp(times, np.linspace(0, duration, num=self.n_envelope_points), random_steps)

                smoothed_waveform = np.zeros_like(original)

                # Process segments
                for start in range(0, total_samples, segment_samples):
                    end = min(start + segment_samples, total_samples)
                    segment = original[start:end]

                    # Average pitch shift for this segment
                    avg_pitch_shift = np.mean(smooth_random_steps[start:end])

                    # Apply pitch shift
                    shifted_segment = pyrb.pitch_shift(segment, self.sample_rate, avg_pitch_shift)

                    # Adjust length if needed
                    seg_len = end - start
                    if len(shifted_segment) > seg_len:
                        shifted_segment = shifted_segment[:seg_len]
                    elif len(shifted_segment) < seg_len:
                        temp = np.zeros(seg_len, dtype=shifted_segment.dtype)
                        temp[:len(shifted_segment)] = shifted_segment
                        shifted_segment = temp

                    # Apply fades to avoid pops
                    if fade_samples > 0:
                        # Fade in if not the first segment
                        if start > 0:
                            fade_in = np.linspace(0, 1, min(fade_samples, len(shifted_segment)))
                            shifted_segment[:len(fade_in)] *= fade_in
                        # Fade out if not the last segment
                        if end < total_samples:
                            fade_out = np.linspace(1, 0, min(fade_samples, len(shifted_segment)))
                            shifted_segment[-len(fade_out):] *= fade_out

                    # Add to the final waveform
                    smoothed_waveform[start:end] += shifted_segment

                # Clip and assign
                smoothed_waveform = np.clip(smoothed_waveform, -1.0, 1.0)
                augmented[b, :, c] = smoothed_waveform

        augmented = torch.tensor(augmented, device=device, dtype=dtype)
        return augmented.squeeze(-1) if channels == 1 else augmented
