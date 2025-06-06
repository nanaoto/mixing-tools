import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import argparse
import os
import sys
import warnings
import platform
from matplotlib import font_manager

def set_chinese_font():
    """根据操作系统自动设置中文字体"""
    system = platform.system()

    try:
        if system == 'Windows':
            # Windows系统字体
            font_path = 'C:/Windows/Fonts/simhei.ttf'  # 黑体
            font_manager.fontManager.addfont(font_path)
            plt.rcParams['font.family'] = 'SimHei'
        elif system == 'Darwin':
            # Mac系统字体
            font_path = '/System/Library/Fonts/PingFang.ttc'  # 苹方
            font_manager.fontManager.addfont(font_path)
            plt.rcParams['font.family'] = 'PingFang SC'
        else:
            # Linux系统字体
            font_path = '/usr/share/fonts/wenquanyi/wqy-zenhei/wqy-zenhei.ttc'  # 文泉驿正黑
            font_manager.fontManager.addfont(font_path)
            plt.rcParams['font.family'] = 'WenQuanYi Zen Hei'

        # 解决负号显示问题
        plt.rcParams['axes.unicode_minus'] = False
        return True
    except Exception as e:
        print(f"字体设置失败: {e}")
        # 回退到Matplotlib默认支持的中文字体
        try:
            plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']  # 尝试Mac默认
            plt.rcParams['axes.unicode_minus'] = False
            return True
        except:
            print("无法设置任何中文字体，将显示为方框")
            return False

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Compare audio spectrograms')
    parser.add_argument('file1', help='First audio file path')
    parser.add_argument('file2', help='Second audio file path')
    parser.add_argument('-o', '--output', default='spectral_comparison.png',
                        help='Output image filename (default: spectral_comparison.png)')
    parser.add_argument('-t', '--threshold', type=float, default=1.0,
                        help='Difference threshold (dB), values below will be black (default: 1.0)')
    parser.add_argument('-b', '--backend', choices=['librosa', 'torch', 'tf'],
                        default='librosa', help='STFT backend (librosa/torch/tf)')
    parser.add_argument('-n', '--n_fft', type=int, default=4096,
                        help='FFT window size (default: 4096)')
    parser.add_argument('-s', '--hop_length', type=int, default=512,
                        help='Hop length (default: 512)')
    return parser.parse_args()


def load_audio(file_path, sr=None):
    """Load audio file with m4a support"""
    try:
        import librosa
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            y, sr = librosa.load(file_path, sr=sr)
        return y, sr
    except Exception as e:
        raise RuntimeError(f"Audio loading failed: {str(e)}")


def compute_stft(y, sr, n_fft, hop_length, backend='librosa'):
    """
    Compute STFT with output normalized to [-100 dB, 0 dB] range
    without using clip operations
    """
    if backend == 'librosa':
        import librosa
        stft = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
        magnitude = np.abs(stft)
    elif backend == 'torch':
        import torch
        import torchaudio
        if torch.cuda.is_available():
            device = torch.device('cuda')
        elif torch.mps.is_available():
            device = torch.device('mps')
        else:
            device = torch.device('cpu')
        y_tensor = torch.from_numpy(y).float().to(device)
        stft = torch.stft(y_tensor, n_fft=n_fft, hop_length=hop_length,
                          window=torch.hann_window(n_fft).to(device),
                          return_complex=True)
        magnitude = torch.abs(stft).cpu().numpy()
    elif backend == 'tf':
        import tensorflow as tf
        y_tensor = tf.convert_to_tensor(y, dtype=tf.float32)
        stft = tf.signal.stft(y_tensor, frame_length=n_fft, frame_step=hop_length,
                              window_fn=tf.signal.hann_window)
        magnitude = tf.abs(stft).numpy()
    else:
        raise ValueError(f"Unsupported backend: {backend}")

    # Normalize to [-100 dB, 0 dB] range
    max_mag = np.max(magnitude)
    if max_mag > 0:
        min_mag = max_mag * 10 ** (-100 / 20)  # Magnitude corresponding to -100 dB
        magnitude = np.maximum(magnitude, min_mag)  # Floor at -100 dB
        stft_db = 20 * np.log10(magnitude / max_mag)  # Normalized to [100 dB, 0 dB]
    else:
        # Handle silent input
        stft_db = -100 * np.ones_like(magnitude)

    return stft_db


def plot_spectrogram(ax, stft_db, sr, hop_length, title, cmap='CMRmap'):
    """Plot single spectrogram with frequency in kHz and time in min:sec"""
    times = librosa.times_like(stft_db, sr=sr, hop_length=hop_length)
    freqs = librosa.fft_frequencies(sr=sr, n_fft=args.n_fft) / 1000  # Convert to kHz

    img = ax.imshow(stft_db, aspect='auto', origin='lower',
                    extent=[times[0], times[-1], freqs[0], freqs[-1]],
                    cmap=cmap, vmin=-100, vmax=0)

    # Format frequency axis (kHz)
    ax.set_yticks(np.arange(0, freqs[-1], 2))
    ax.set_yticklabels([f"{int(f)}k" for f in np.arange(0, freqs[-1], 2)])
    ax.set_ylabel('Frequency (kHz)')

    # Format time axis (min:sec)
    ax.xaxis.set_major_formatter(lambda x, _: f"{int(x // 60)}:{int(x % 60):02d}")
    ax.set_xlabel('Time (min:sec)')

    ax.set_title(title)
    return img


def plot_difference(ax, diff_db, sr, hop_length, threshold, title, mode='both'):
    """Plot difference spectrogram with frequency in kHz and time in min:sec

    Parameters:
        mode: 'both' (show both positive and negative), 'positive' (only positive), 'negative' (only negative)
    """
    times = librosa.times_like(diff_db, sr=sr, hop_length=hop_length)
    freqs = librosa.fft_frequencies(sr=sr, n_fft=args.n_fft) / 1000  # Convert to kHz

    cmap = plt.get_cmap('pink').copy()
    cmap.set_bad(color='black')

    # Apply threshold and mask based on mode
    if mode == 'positive':
        diff_db_masked = np.where(diff_db < threshold, np.nan, diff_db)
        vmin = threshold
        vmax = 20
    elif mode == 'negative':
        cmap = plt.get_cmap('gnuplot_r').copy()
        cmap.set_bad(color='black')
        diff_db_masked = np.where(diff_db > -threshold, np.nan, diff_db)
        vmin = -20
        vmax = -threshold
    else:  # both
        cmap = plt.get_cmap('vanimo').copy()
        cmap.set_bad(color='black')
        diff_db_masked = np.where(np.abs(diff_db) < threshold, np.nan, diff_db)
        vmin = -20
        vmax = 20

    img = ax.imshow(diff_db_masked, aspect='auto', origin='lower',
                    extent=[times[0], times[-1], freqs[0], freqs[-1]],
                    cmap=cmap,
                    vmin=vmin, vmax=vmax)

    # Format frequency axis (kHz)
    ax.set_yticks(np.arange(0, freqs[-1], 2))
    ax.set_yticklabels([f"{int(f)}k" for f in np.arange(0, freqs[-1], 2)])
    ax.set_ylabel('Frequency (kHz)')

    # Format time axis (min:sec)
    ax.xaxis.set_major_formatter(lambda x, _: f"{int(x // 60)}:{int(x % 60):02d}")
    ax.set_xlabel('Time (min:sec)')

    ax.set_title(title)
    return img


def compare_spectrograms(args):
    """Main comparison function"""
    try:
        # Load audio
        y1, sr = load_audio(args.file1)
        y2, _ = load_audio(args.file2, sr=sr)

        # Align length
        min_len = min(len(y1), len(y2))
        y1, y2 = y1[:min_len], y2[:min_len]

        # Compute STFT (now returns dB values normalized to [-100, 0])
        print(f"Using {args.backend} backend for STFT...")
        stft1_db = compute_stft(y1, sr, args.n_fft, args.hop_length, args.backend)
        stft2_db = compute_stft(y2, sr, args.n_fft, args.hop_length, args.backend)

        # Compute difference
        diff_db = stft2_db - stft1_db

        # Create figure - now with 4 rows instead of 3
        fig, axes = plt.subplots(4, 1, figsize=(12, 16))
        fig.suptitle(f"Spectral Comparison (n_fft={args.n_fft}, hop={args.hop_length})", y=1.02)

        # Plot first spectrogram
        img1 = plot_spectrogram(axes[0], stft1_db, sr, args.hop_length,
                                f"Spectrogram: {os.path.basename(args.file1)}")

        # Plot second spectrogram
        img2 = plot_spectrogram(axes[1], stft2_db, sr, args.hop_length,
                                f"Spectrogram: {os.path.basename(args.file2)}")

        # Plot positive differences (increases)
        img3 = plot_difference(axes[2], diff_db, sr, args.hop_length, args.threshold,
                               f"Increases from {os.path.basename(args.file1)} to {os.path.basename(args.file2)}\n(black: <{args.threshold}dB)",
                               mode='positive')

        # Plot negative differences (decreases)
        img4 = plot_difference(axes[3], diff_db, sr, args.hop_length, args.threshold,
                               f"Decreases from {os.path.basename(args.file1)} to {os.path.basename(args.file2)}\n(black: <{args.threshold}dB)",
                               mode='negative')

        # Add colorbars
        fig.colorbar(img1, ax=axes[0], format='%+2.0f dB', label='Magnitude (dB)')
        fig.colorbar(img2, ax=axes[1], format='%+2.0f dB', label='Magnitude (dB)')
        fig.colorbar(img3, ax=axes[2], format='%+2.0f dB', label='Increase (dB)')
        fig.colorbar(img4, ax=axes[3], format='%+2.0f dB', label='Decrease (dB)')

        plt.tight_layout()
        plt.savefig(args.output, dpi=300, bbox_inches='tight')
        print(f"Results saved to: {args.output}")

    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)


if __name__ == '__main__':
    args = parse_args()
    if set_chinese_font():
        print("中文字体设置成功")
    else:
        print("无法设置中文字体")

    # Check backend dependencies
    if args.backend == 'torch':
        try:
            import torch
            import torchaudio
        except ImportError:
            print("Error: Please install PyTorch: pip install torch torchaudio")
            sys.exit(1)
    elif args.backend == 'tf':
        try:
            import tensorflow as tf
        except ImportError:
            print("Error: Please install TensorFlow: pip install tensorflow")
            sys.exit(1)

    compare_spectrograms(args)