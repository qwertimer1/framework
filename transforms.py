import torch
import math
from typing import Optional
import functional as F
import matplotlib.pyplot as plt
import PIL


class Transform():
    _order = 0
    def setup(self, dsrc): return    # 1-time setup
    def __call__(self,o):  return o  # transform
    def decode(self,o):    return o  # reverse transform for display

class ToCuda(Transform):
    """For Audio data
    Arguments:
        ad: Audio Data
    
    Returns:
        sig = cuda(sig)
        sr = sampling rate
    """
    _order=30
    def __call__(self, ad):
        sig,sr = ad
        print("ToCuda")
        return (sig.cuda(), sr)
class to_tensor(Transform):
    _order = 1
    def __call__(self, ad):
        return torch.from_numpy(ad).type_as(tensor)
class to_byte_tensor(Transform):
    _order=10
    def __call__(self, ad):  

        res = torch.ByteTensor(torch.ByteStorage.from_buffer(ad.tobytes()))
        w,h = ad.size
        print('to_byte_tensor')
        print(type(res))
        return res.view(h,w,-1).permute(2,0,1)
    

class to_float_tensor(Transform):
    _order=20    
    def __call__(self, ad):
        return ad.float().div_(255.)
    


#Image Transforms

class MakeRGB(Transform):
    def __call__(self, item): return item.convert('RGB')
    
class ResizeFixed(Transform):
    _order=10
    def __init__(self,size):
        if isinstance(size,int): size=(size,size)
        self.size = size
        
    def __call__(self, item): return item.resize(self.size, PIL.Image.BILINEAR)



class Spectrogrammer(Transform):
    _order=90
    def __init__(self, to_mel=True, to_db=True, n_fft=400, ws=None, hop=None, 
                 f_min=0.0, f_max=None, pad=0, n_mels=128, top_db=None, normalize=False):
        self.to_mel, self.to_db, self.n_fft, self.ws, self.hop, self.f_min, self.f_max, \
        self.pad, self.n_mels, self.top_db, self.normalize = to_mel, to_db, n_fft, \
        ws, hop, f_min, f_max, pad, n_mels, top_db, normalize


    def __call__(self, ad):
        sig,sr = ad
        if self.to_mel:
            spec = MelSpectrogram(sr, self.n_fft, self.ws, self.hop, self.f_min, 
                                             self.f_max, self.pad, self.n_mels)(sig)
        else: 
            spec = Spectrogram(self.n_fft, self.ws, self.hop, self.pad, 
                                          normalize=self.normalize)(sig)
        if self.to_db:
            spec = SpectrogramToDB(top_db=self.top_db)(spec)
        print(f'spec = {spec.shape}')
        spec = spec.permute(0,2,1) # reshape so it looks good to humans
        print(f'spec 2 = {spec.shape}')
        return spec

def show_spectro(img, ax=None, figsize=(6,6), with_shape=True):
    #if hasattr(img,"device") & str(img.device).startswith("cuda"): img = img.cpu()
    if ax is None: _,ax = plt.subplots(1, 1, figsize=figsize)
    ax.imshow(img if (img.shape[0]==3) else img.squeeze(0))
    #if with_shape: display(f'Tensor shape={img.shape}')


class Scale():
    """Scale audio tensor from a 16-bit integer (represented as a FloatTensor)
    to a floating point number between -1.0 and 1.0.  Note the 16-bit number is
    called the "bit depth" or "precision", not to be confused with "bit rate".
    Args:
        factor (int): maximum value of input tensor. default: 16-bit depth
    """
    __constants__ = ['factor']

    def __init__(self, factor=2**31):
        super(Scale, self).__init__()
        self.factor = factor

    def forward(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor of audio of size (Samples x Channels)
        Returns:
            Tensor: Scaled by the scale factor. (default between -1.0 and 1.0)
        """
        return F.scale(tensor, self.factor)

    def __repr__(self):
        return self.__class__.__name__ + '()'



class Spectrogram():
    """Create a spectrogram from a raw audio signal
    Args:
        n_fft (int, optional): size of fft, creates n_fft // 2 + 1 bins
        ws (int): window size. default: n_fft
        hop (int, optional): length of hop between STFT windows. default: ws // 2
        pad (int): two sided padding of signal
        window (torch windowing function): default: torch.hann_window
        power (int > 0 ) : Exponent for the magnitude spectrogram,
                        e.g., 1 for energy, 2 for power, etc.
        normalize (bool) : whether to normalize by magnitude after stft
        wkwargs (dict, optional): arguments for window function
    """
    __constants__ = ['n_fft', 'ws', 'hop', 'pad', 'power', 'normalize']

    def __init__(self, n_fft=400, ws=None, hop=None,
                 pad=0, window=torch.hann_window,
                 power=2, normalize=False, wkwargs=None):
        super(Spectrogram, self).__init__()
        self.n_fft = n_fft
        # number of fft bins. the returned STFT result will have n_fft // 2 + 1
        # number of frequecies due to onesided=True in torch.stft
        self.ws = ws if ws is not None else n_fft
        self.hop = hop if hop is not None else self.ws // 2
        window = window(self.ws) if wkwargs is None else window(self.ws, **wkwargs)
        self.window = window
        self.pad = pad
        self.power = power
        self.normalize = normalize
    
    def __call__(self, sig):
        return F.spectrogram(sig, self.pad, self.window, self.n_fft, self.hop,
                self.ws, self.power, self.normalize)


    def forward(self, sig):
        """
        Args:
            sig (Tensor): Tensor of audio of size (c, n)
        Returns:
            spec_f (Tensor): channels x hops x n_fft (c, l, f), where channels
                is unchanged, hops is the number of hops, and n_fft is the
                number of fourier bins, which should be the window size divided
                by 2 plus 1.
        """
        return F.spectrogram(sig, self.pad, self.window, self.n_fft, self.hop,
                             self.ws, self.power, self.normalize)





class MelScale():
    """This turns a normal STFT into a mel frequency STFT, using a conversion
       matrix.  This uses triangular filter banks.
       User can control which device the filter bank (`fb`) is (e.g. fb.to(spec_f.device)).
    Args:
        n_mels (int): number of mel bins
        sr (int): sample rate of audio signal
        f_max (float, optional): maximum frequency. default: `sr` // 2
        f_min (float): minimum frequency. default: 0
        n_stft (int, optional): number of filter banks from stft. Calculated from first input
            if `None` is given.  See `n_fft` in `Spectrogram`.
    """
    __constants__ = ['n_mels', 'sr', 'f_min', 'f_max']

    def __init__(self, n_mels=128, sr=16000, f_max=None, f_min=0., n_stft=None):
        super(MelScale, self).__init__()
        self.n_mels = n_mels
        self.sr = sr
        self.f_max = f_max if f_max is not None else float(sr // 2)
        self.f_min = f_min
        fb = torch.empty(0) if n_stft is None else F.create_fb_matrix(
            n_stft, self.f_min, self.f_max, self.n_mels)
        self.fb = fb
    def __call__(self, spec_f):
        if self.fb.numel() == 0:
            tmp_fb = F.create_fb_matrix(spec_f.size(2), self.f_min, self.f_max, self.n_mels)
            # Attributes cannot be reassigned outside __init__ so workaround
            self.fb.resize_(tmp_fb.size())
            self.fb.copy_(tmp_fb)
        spec_m = torch.matmul(spec_f, self.fb)  # (c, l, n_fft) dot (n_fft, n_mels) -> (c, l, n_mels)
        return spec_m


    def forward(self, spec_f):
        if self.fb.numel() == 0:
            tmp_fb = F.create_fb_matrix(spec_f.size(2), self.f_min, self.f_max, self.n_mels)
            # Attributes cannot be reassigned outside __init__ so workaround
            self.fb.resize_(tmp_fb.size())
            self.fb.copy_(tmp_fb)
        spec_m = torch.matmul(spec_f, self.fb)  # (c, l, n_fft) dot (n_fft, n_mels) -> (c, l, n_mels)
        return spec_m


class SpectrogramToDB():
    """Turns a spectrogram from the power/amplitude scale to the decibel scale.
    This output depends on the maximum value in the input spectrogram, and so
    may return different values for an audio clip split into snippets vs. a
    a full clip.
    Args:
        stype (str): scale of input spectrogram ("power" or "magnitude").  The
            power being the elementwise square of the magnitude. default: "power"
        top_db (float, optional): minimum negative cut-off in decibels.  A reasonable number
            is 80.
    """
    __constants__ = ['multiplier', 'amin', 'ref_value', 'db_multiplier']

    def __init__(self, stype="power", top_db=None):
        super(SpectrogramToDB, self).__init__()
        self.stype =stype
        if top_db is not None and top_db < 0:
            raise ValueError('top_db must be positive value')
        self.top_db = top_db
        self.multiplier = 10. if stype == "power" else 20.
        self.amin = 1e-10
        self.ref_value = 1.
        self.db_multiplier = math.log10(max(self.amin, self.ref_value))
    def __call__(self, spec):
        # numerically stable implementation from librosa
        # https://librosa.github.io/librosa/_modules/librosa/core/spectrum.html
        return F.spectrogram_to_DB(spec, self.multiplier, self.amin, self.db_multiplier, self.top_db)
 
    def forward(self, spec):
        # numerically stable implementation from librosa
        # https://librosa.github.io/librosa/_modules/librosa/core/spectrum.html
        return F.spectrogram_to_DB(spec, self.multiplier, self.amin, self.db_multiplier, self.top_db)


class MFCC():
    """Create the Mel-frequency cepstrum coefficients from an audio signal
        By default, this calculates the MFCC on the DB-scaled Mel spectrogram.
        This is not the textbook implementation, but is implemented here to
        give consistency with librosa.
        This output depends on the maximum value in the input spectrogram, and so
        may return different values for an audio clip split into snippets vs. a
        a full clip.
        Args:
        sr (int) : sample rate of audio signal
        n_mfcc (int) : number of mfc coefficients to retain
        dct_type (int) : type of DCT (discrete cosine transform) to use
        norm (string, optional) : norm to use
        log_mels (bool) : whether to use log-mel spectrograms instead of db-scaled
        melkwargs (dict, optional): arguments for MelSpectrogram
    """
    __constants__ = ['sr', 'n_mfcc', 'dct_type', 'top_db', 'log_mels']

    def __init__(self, sr=16000, n_mfcc=40, dct_type=2, norm='ortho', log_mels=False,
                 melkwargs=None):
        super(MFCC, self).__init__()
        supported_dct_types = [2]
        if dct_type not in supported_dct_types:
            raise ValueError(f'DCT type not supported{dct_type}')
        self.sr = sr
        self.n_mfcc = n_mfcc
        self.dct_type = dct_type
        self.norm = norm
        self.top_db = 80.
        self.s2db = SpectrogramToDB("power", self.top_db)

        if melkwargs is not None:
            self.MelSpectrogram = MelSpectrogram(sr=self.sr, **melkwargs)
        else:
            self.MelSpectrogram = MelSpectrogram(sr=self.sr)

        if self.n_mfcc > self.MelSpectrogram.n_mels:
            raise ValueError('Cannot select more MFCC coefficients than # mel bins')
        dct_mat = F.create_dct(self.n_mfcc, self.MelSpectrogram.n_mels, self.norm)
        self.dct_mat = dct_mat, torch.Tensor
        self.log_mels = log_mels
    def __call__(self, sig):
        """
        Args:
            sig (Tensor): Tensor of audio of size (channels [c], samples [n])
        Returns:
            spec_mel_db (Tensor): channels x hops x n_mels (c, l, m), where channels
                is unchanged, hops is the number of hops, and n_mels is the
                number of mel bins.
        """
        mel_spect = self.MelSpectrogram(sig)
        if self.log_mels:
            log_offset = 1e-6
            mel_spect = torch.log(mel_spect + log_offset)
        else:
            mel_spect = self.s2db(mel_spect)
        mfcc = torch.matmul(mel_spect, self.dct_mat)
        return mfcc


    def forward(self, sig):
        """
        Args:
            sig (Tensor): Tensor of audio of size (channels [c], samples [n])
        Returns:
            spec_mel_db (Tensor): channels x hops x n_mels (c, l, m), where channels
                is unchanged, hops is the number of hops, and n_mels is the
                number of mel bins.
        """
        mel_spect = self.MelSpectrogram(sig)
        if self.log_mels:
            log_offset = 1e-6
            mel_spect = torch.log(mel_spect + log_offset)
        else:
            mel_spect = self.s2db(mel_spect)
        mfcc = torch.matmul(mel_spect, self.dct_mat)
        return mfcc


class MelSpectrogram():
    """Create MEL Spectrograms from a raw audio signal using the stft
       function in PyTorch.
    Sources:
        * https://gist.github.com/kastnerkyle/179d6e9a88202ab0a2fe
        * https://timsainb.github.io/spectrograms-mfccs-and-inversion-in-python.html
        * http://haythamfayek.com/2016/04/21/speech-processing-for-machine-learning.html
    Args:
        sr (int): sample rate of audio signal
        ws (int): window size
        hop (int, optional): length of hop between STFT windows. default: `ws` // 2
        n_fft (int, optional): number of fft bins. default: `ws` // 2 + 1
        f_max (float, optional): maximum frequency. default: `sr` // 2
        f_min (float): minimum frequency. default: 0
        pad (int): two sided padding of signal
        n_mels (int): number of MEL bins
        window (torch windowing function): default: `torch.hann_window`
        wkwargs (dict, optional): arguments for window function
    Example:
        >>> sig, sr = torchaudio.load("test.wav", normalization=True)
        >>> spec_mel = transforms.MelSpectrogram(sr)(sig)  # (c, l, m)
    """
    __constants__ = ['sr', 'n_fft', 'ws', 'hop', 'pad', 'n_mels', 'f_min']

    def __init__(self, sr=16000, n_fft=400, ws=None, hop=None, f_min=0., f_max=None,
                 pad=0, n_mels=128, window=torch.hann_window, wkwargs=None):
        super(MelSpectrogram, self).__init__()
        self.sr = sr
        self.n_fft = n_fft
        self.ws = ws if ws is not None else n_fft
        self.hop = hop if hop is not None else self.ws // 2
        self.pad = pad
        self.n_mels = n_mels  # number of mel frequency bins
        self.f_max = f_max
        self.f_min = f_min
        self.spec = Spectrogram(n_fft=self.n_fft, ws=self.ws, hop=self.hop,
                                pad=self.pad, window=window, power=2,
                                normalize=False, wkwargs=wkwargs)
        self.fm = MelScale(self.n_mels, self.sr, self.f_max, self.f_min)
    def __call__(self, sig):
        """
        Args:
            sig (Tensor): Tensor of audio of size (channels [c], samples [n])
        Returns:
            spec_mel (Tensor): channels x hops x n_mels (c, l, m), where channels
                is unchanged, hops is the number of hops, and n_mels is the
                number of mel bins.
        """
        spec = self.spec(sig)
        spec_mel = self.fm(spec)
        return spec_mel




    def forward(self, sig):
        """
        Args:
            sig (Tensor): Tensor of audio of size (channels [c], samples [n])
        Returns:
            spec_mel (Tensor): channels x hops x n_mels (c, l, m), where channels
                is unchanged, hops is the number of hops, and n_mels is the
                number of mel bins.
        """
        spec = self.spec(sig)
        spec_mel = self.fm(spec)
        return spec_mel


