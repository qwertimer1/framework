"""functional Module

This module has been stripped out of pytorch audio code.
The pytorch audio library does not work naturally on windows and so this is the functional module inside the pytorch audio library to allow the development of spectrogram and MEL_Spectrogram transforms for audio data

Returns:
    [type] -- [description]
"""

import torch
import math



def scale(tensor, factor):
    # type: (Tensor, int) -> Tensor
    r"""Scale audio tensor from a 16-bit integer (represented as a
    :class:`torch.FloatTensor`) to a floating point number between -1.0 and 1.0.
    Note the 16-bit number is called the "bit depth" or "precision", not to be
    confused with "bit rate".
    Args:
        tensor (torch.Tensor): Tensor of audio of size (n, c) or (c, n)
        factor (int): Maximum value of input tensor
    Returns:
        torch.Tensor: Scaled by the scale factor
    """
    if not tensor.is_floating_point():
        tensor = tensor.to(torch.float32)

    return tensor / factor

def _stft(input, n_fft, hop_length, win_length, window, center, pad_mode, normalized, onesided):
    # type: (Tensor, int, Optional[int], Optional[int], Optional[Tensor], bool, str, bool, bool) -> Tensor
    return torch.stft(input, n_fft, hop_length, win_length, window, center, pad_mode, normalized, onesided)

   
def spectrogram(sig, pad, window, n_fft, hop, ws, power, normalize):
    # type: (Tensor, int, Tensor, int, int, int, int, bool) -> Tensor
    r"""Create a spectrogram from a raw audio signal.
    Args:
        sig (torch.Tensor): Tensor of audio of size (c, n)
        pad (int): Two sided padding of signal
        window (torch.Tensor): Window_tensor
        n_fft (int): Size of fft
        hop (int): Length of hop between STFT windows
        ws (int): Window size
        power (int) : Exponent for the magnitude spectrogram,
            (must be > 0) e.g., 1 for energy, 2 for power, etc.
        normalize (bool) : Whether to normalize by magnitude after stft
    Returns:
        torch.Tensor: Channels x hops x n_fft (c, l, f), where channels
        is unchanged, hops is the number of hops, and n_fft is the
        number of fourier bins, which should be the window size divided
        by 2 plus 1.
    """
    

    if pad > 0:
        # TODO add "with torch.no_grad():" back when JIT supports it
        sig = torch.nn.functional.pad(sig, (pad, pad), "constant")

    # default values are consistent with librosa.core.spectrum._spectrogram
    spec_f = _stft(sig, n_fft, hop, ws, window,
                   True, 'reflect', False, True).transpose(1, 2)

    if normalize:
        spec_f /= window.pow(2).sum().sqrt()
    spec_f = spec_f.pow(power).sum(-1)  # get power of "complex" tensor (c, l, n_fft)
    return spec_f

def spectrogram_to_DB(spec, multiplier, amin, db_multiplier, top_db=None):
    # type: (Tensor, float, float, float, Optional[float]) -> Tensor
    r"""Turns a spectrogram from the power/amplitude scale to the decibel scale.
    This output depends on the maximum value in the input spectrogram, and so
    may return different values for an audio clip split into snippets vs. a
    a full clip.
    Args:
        spec (torch.Tensor): Normal STFT
        multiplier (float): Use 10. for power and 20. for amplitude
        amin (float): Number to clamp spec
        db_multiplier (float): Log10(max(reference value and amin))
        top_db (Optional[float]): Minimum negative cut-off in decibels.  A reasonable number
            is 80.
    Returns:
        torch.Tensor: Spectrogram in DB
    """
    spec_db = multiplier * torch.log10(torch.clamp(spec, min=amin))
    spec_db -= multiplier * db_multiplier

    if top_db is not None:
        new_spec_db_max = torch.tensor(float(spec_db.max()) - top_db, dtype=spec_db.dtype, device=spec_db.device)
        spec_db = torch.max(spec_db, new_spec_db_max)

    return spec_db


def create_fb_matrix(n_stft, f_min, f_max, n_mels):
    # type: (int, float, float, int) -> Tensor
    r""" Create a frequency bin conversion matrix.
    Args:
        n_stft (int): Number of filter banks from spectrogram
        f_min (float): Minimum frequency
        f_max (float): Maximum frequency
        n_mels (int): Number of mel bins
    Returns:
        torch.Tensor: Triangular filter banks (fb matrix)
    """
    # get stft freq bins
    stft_freqs = torch.linspace(f_min, f_max, n_stft)
    # calculate mel freq bins
    # hertz to mel(f) is 2595. * math.log10(1. + (f / 700.))
    m_min = 0. if f_min == 0 else 2595. * math.log10(1. + (f_min / 700.))
    m_max = 2595. * math.log10(1. + (f_max / 700.))
    m_pts = torch.linspace(m_min, m_max, n_mels + 2)
    # mel to hertz(mel) is 700. * (10**(mel / 2595.) - 1.)
    f_pts = 700. * (10**(m_pts / 2595.) - 1.)
    # calculate the difference between each mel point and each stft freq point in hertz
    f_diff = f_pts[1:] - f_pts[:-1]  # (n_mels + 1)
    slopes = f_pts.unsqueeze(0) - stft_freqs.unsqueeze(1)  # (n_stft, n_mels + 2)
    # create overlapping triangles
    z = torch.zeros(1)
    down_slopes = (-1. * slopes[:, :-2]) / f_diff[:-1]  # (n_stft, n_mels)
    up_slopes = slopes[:, 2:] / f_diff[1:]  # (n_stft, n_mels)
    fb = torch.max(z, torch.min(down_slopes, up_slopes))
    return fb

def create_dct(n_mfcc, n_mels, norm):
    # type: (int, int, Optional[str]) -> Tensor
    r"""Creates a DCT transformation matrix with shape (num_mels, num_mfcc),
    normalized depending on norm.
    Args:
        n_mfcc (int) : Number of mfc coefficients to retain
        n_mels (int): Number of MEL bins
        norm (Optional[str]) : Norm to use (either 'ortho' or None)
    Returns:
        torch.Tensor: The transformation matrix, to be right-multiplied to row-wise data.
    """
    outdim = n_mfcc
    dim = n_mels
    # http://en.wikipedia.org/wiki/Discrete_cosine_transform#DCT-II
    n = torch.arange(dim)
    k = torch.arange(outdim)[:, None]
    dct = torch.cos(math.pi / float(dim) * (n + 0.5) * k)
    if norm is None:
        dct *= 2.0
    else:
        assert norm == 'ortho'
        dct[0] *= 1.0 / math.sqrt(2.0)
        dct *= math.sqrt(2.0 / float(dim))
    return dct.t()   