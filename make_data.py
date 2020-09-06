from scipy.io import wavfile
import pyworld as pw
import numpy as np
import pysptk
from glob import glob
import torch

# fs : sampling frequency, 音楽業界では44,100Hz
# data : arrayの音声データが入る
def wav2mcep(WAV_FILE,dim):
    fs, data = wavfile.read(WAV_FILE)

    # floatでないとworldは扱えない
    data = data.astype(np.float)

    _f0, _time = pw.dio(data, fs)    # 基本周波数の抽出。pw.dioは0.005秒ごとの基本周波数を測定し、numpyとして返す。
    f0 = pw.stonemask(data, _f0, _time, fs)  # 基本周波数の修正
    sp = pw.cheaptrick(data, f0, _time, fs)  # スペクトル包絡の抽出
    ap = pw.d4c(data, f0, _time, fs)         # 非周期性指標の抽出
    mcep=pysptk.sp2mc(sp,dim,0.42)
    return torch.Tensor(mcep)


def all_mcep(path):
    all_audio=[]
    files=glob(path+"/*")
    for file in files:
        mcep=wav2mcep(file,23).T
        all_audio.append(mcep)
    return all_audio
