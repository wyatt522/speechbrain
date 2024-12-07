import torch
import pyrubberband as pyrb
import numpy as np

class VariablePitchShiftSmooth(torch.nn.Module):
    """
    Apply a continuous pitch shift effect with smoothing transitions to a waveform.
    
    This module modifies the pitch of the input waveform in small segments while 
    maintaining the overall playback speed. It applies a fade-in and fade-out effect 
    to eliminate audible popping between segments.

    Arguments
    ---------
    n_steps_range : tuple
        Range of pitch shift steps (min, max) in semitones.
    fade_duration : float
        Duration of fade-in/out in seconds for smooth transitions.

    Example
    -------
    >>> waveform = torch.randn(1, 160000)  # Example waveform
    >>> pitch_shift_module = VariablePitchShiftSmooth(n_steps_range=(-5, 5), fade_duration=0.01)
    >>> output_waveform = pitch_shift_module(waveform)
    """
    
    def __init__(self, n_steps_range=(-5, 5), fade_duration=0.01):
        super(VariablePitchShiftSmooth, self).__init__()
        self.n_steps_range = n_steps_range
        self.fade_duration = fade_duration

    def forward(self, waveform):
        """
        Apply the continuous pitch shift effect to the input waveform.

        Arguments
        ---------
        waveform : torch.Tensor
            Input waveform of shape `[1, samples]`.

        Returns
        -------
        torch.Tensor
            Pitch-shifted waveform with smooth transitions.
        """
        # Ensure the waveform is in numpy format for pyrubberband
        waveform_np = waveform.squeeze().cpu().numpy()
        total_samples = waveform_np.shape[0]
        sample_rate = 16000  # Set a fixed sample rate for processing

        # Calculate the duration of the waveform in seconds
        duration = total_samples / sample_rate

        # Generate a smooth pitch shift envelope
        times = np.linspace(0, duration, num=total_samples)
        random_steps = np.random.uniform(self.n_steps_range[0], self.n_steps_range[1], size=100)
        smooth_random_steps = np.interp(times, np.linspace(0, duration, num=100), random_steps)

        # Segment processing for pitch shifting with fade-in/out
        segment_duration = 0.1  # Duration of each segment in seconds
        segment_samples = int(segment_duration * sample_rate)
        fade_samples = int(self.fade_duration * sample_rate)

        smoothed_waveform = np.zeros_like(waveform_np)

        for start in range(0, total_samples, segment_samples):
            end = min(start + segment_samples, total_samples)
            segment = waveform_np[start:end]

            # Apply average pitch shift for the segment
            avg_pitch_shift = np.mean(smooth_random_steps[start:end])

            # Pitch shift the segment using pyrubberband
            shifted_segment = pyrb.pitch_shift(segment, sample_rate, avg_pitch_shift)

            # Apply fade-in and fade-out to avoid popping sounds
            if fade_samples > 0:
                fade = np.linspace(0, 1, min(fade_samples, len(shifted_segment)))
                if start > 0:  # Apply fade-in at the beginning
                    shifted_segment[:len(fade)] *= fade
                if end < total_samples:  # Apply fade-out at the end
                    fade_out = np.linspace(1, 0, min(fade_samples, len(shifted_segment)))
                    shifted_segment[-len(fade_out):] *= fade_out

            # Combine the shifted segment into the final waveform
            smoothed_waveform[start:end] += shifted_segment[:end - start]

        # Normalize to avoid clipping
        smoothed_waveform = np.clip(smoothed_waveform, -1.0, 1.0)

        # Convert back to PyTorch tensor and return
        return torch.from_numpy(smoothed_waveform).to(waveform.device).unsqueeze(0)


class MultiPitchShift(torch.nn.Module):
    """
    Applies multiple pitch shifts to a spectrogram and overlays them to create a combined effect.
    This can help make the model more robust to pitch variations.

    Arguments
    ---------
    n_steps_range : tuple
        Range of pitch shift steps (min, max).
    num_shifts : int
        Number of pitch shifts to apply and overlay.
    dim : int, optional
        Dimension along which to apply the pitch shifts (1 for time, 2 for frequency).
        Default is 1.
    """

    def __init__(self, n_steps_range=(-5, 5), num_shifts=5, dim=1):
        super().__init__()
        self.n_steps_range = n_steps_range
        self.num_shifts = num_shifts
        self.dim = dim

    def forward(self, spectrogram):
        """
        Apply the MultiPitchShift augmentation to the input spectrogram.

        Arguments
        ---------
        spectrogram : torch.Tensor
            Input spectrogram of shape `[batch, time, fea]`.

        Returns
        -------
        torch.Tensor
            Augmented spectrogram with multiple pitch-shifted overlays.
        """
        if spectrogram.dim() == 4:
            spectrogram = spectrogram.view(-1, spectrogram.shape[2], spectrogram.shape[3])

        batch_size, time_duration, fea_size = spectrogram.shape

        # If `dim=1`, apply pitch shift along time; if `dim=2`, along frequency.
        if self.dim == 1:
            axis = 1  # Time axis
        else:
            axis = 2  # Frequency axis

        # Initialize the overlayed spectrogram with zeros
        overlayed_spectrogram = torch.zeros_like(spectrogram)

        # Apply pitch shift for each item in the batch
        for i in range(batch_size):
            sample = spectrogram[i].cpu().numpy()
            sample_overlayed = np.zeros_like(sample)

            # Apply the specified number of pitch shifts and overlay them
            for _ in range(self.num_shifts):
                n_steps = np.random.uniform(self.n_steps_range[0], self.n_steps_range[1])
                shifted_sample = pyrb.pitch_shift(sample, 16000, n_steps)

                # Overlay the pitch-shifted sample
                sample_overlayed += shifted_sample

            # Normalize the overlayed spectrogram to prevent clipping
            sample_overlayed /= self.num_shifts
            overlayed_spectrogram[i] = torch.from_numpy(sample_overlayed).to(spectrogram.device)

        return overlayed_spectrogram