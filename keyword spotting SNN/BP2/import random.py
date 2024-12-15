import random
import pandas as pd
from datetime import datetime

#创建train_acc.csv和var_acc.csv文件，记录loss和accuracy
df = pd.DataFrame(columns=['time','step','train Loss','training accuracy'])#列名
df.to_csv("E:\\train2_acc.csv",index=False) #路径可以根据需要更改

#初始化train数据
t_loss = 0.4
t_acc = 0.5
for i in range(20):#假设迭代20次
    time = "%s"%datetime.now()#获取当前时间
    step = "Step[%d]"%i
    t_loss = t_loss - random.uniform(0.01,0.017)
    train_loss = "%f"%t_loss
    t_acc = t_acc + random.uniform(0.05,0.035)
    train_acc = "%g"%t_acc
#将数据保存在一维列表
    list = [time,step,train_loss,train_acc]
#由于DataFrame是Pandas库中的一种数据结构，它类似excel，是一种二维表，所以需要将list以二维列表的形式转化为DataFrame
    data = pd.DataFrame([list])
    data.to_csv('E:\\train2_acc.csv',mode='a',header=False,index=False)#mode设为a,就可以向csv文件追加数据了


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#读取csv中指定列的数据
data1 = pd.read_csv('E:\\train_acc.csv')
data2 = pd.read_csv('E:\\train2_acc.csv')
data1_loss = data1[['train Loss']] #class 'pandas.core.frame.DataFrame'
data1_acc = data1[['training accuracy']]
data2_loss = data2[['train Loss']] #class 'pandas.core.frame.DataFrame'
data2_acc = data2[['training accuracy']]
x = np.arange(0,20,1)
y1 =np.array(data1_loss)#将DataFrame类型转化为numpy数组
y2 = np.array(data1_acc)
y12 =np.array(data2_loss)#将DataFrame类型转化为numpy数组
y22 = np.array(data2_acc)
#绘图
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.plot(x,y2,label="精度1")
plt.plot(x,y22,label="accuracy2")
plt.title("loss & accuracy") 
plt.xlabel('step')
plt.ylabel('probability')
plt.legend()   #显示标签
plt.show()


import librosa
import librosa.display
import numpy as np
from scipy.fftpack import dct
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示符号


def erb_space(low_freq=50, high_freq=10000, n=64):
    """ 计算中心频率(ERB scale)
    :param min_freq: 中心频率域的最小频率。
    :param max_freq: 中心频率域的最大频率。
    :param nfilts: 滤波器的个数，即等于计算中心频率的个数。
    :return: 一组中心频率
    """
    ear_q = 9.26449
    min_bw = 24.7
    cf_array = -(ear_q * min_bw) + np.exp(
        np.linspace(1, n, n) * (-np.log(high_freq + ear_q * min_bw) + np.log(low_freq + ear_q * min_bw)) / n) \
               * (high_freq + ear_q * min_bw)
    return cf_array


def gammatone_impulse_response(samplerate_hz, length_in_samples, center_freq_hz, p):
    """ gammatone滤波器的时域公式
    :param samplerate_hz: 采样率
    :param length_in_samples: 信号长度
    :param center_freq_hz: 中心频率
    :param p: 滤波器阶数
    :return: gammatone 脉冲响应
    """
    # 生成一个gammatone filter (1990 Glasberg&Moore parametrized)
    erb = 24.7 + (center_freq_hz / 9.26449)  # equivalent rectangular bandwidth.
    # 中心频率
    an = (np.pi * np.math.factorial(2 * p - 2) * np.power(2, float(-(2 * p - 2)))) / np.square(np.math.factorial(p - 1))
    b = erb / an  # 带宽

    a = 1  # 幅度(amplitude). 这在后面的归一化过程中有所不同。
    t = np.linspace(1. / samplerate_hz, length_in_samples / samplerate_hz, length_in_samples)
    gammatone_ir = a * np.power(t, p - 1) * np.exp(-2 * np.pi * b * t) * np.cos(2 * np.pi * center_freq_hz * t)
    return gammatone_ir


def generate_filterbank(fs, fmax, L, N, p=4):
    """
    L: 在样本中测量的信号的大小
    N: 滤波器数量
    p: Gammatone脉冲响应的阶数
    """
    # 中心频率
    if fs == 8000:
        fmax = 4000
    center_freqs = erb_space(50, fmax, N)  # 中心频率列表
    center_freqs = np.flip(center_freqs)  # 反转数组
    n_center_freqs = len(center_freqs)  # 中心频率的数量

    filterbank = np.zeros((N, L))

    # 为每个中心频率生成 滤波器
    for i in range(n_center_freqs):
        # aa = gammatone_impulse_response(fs, L, center_freqs[i], p)
        filterbank[i, :] = gammatone_impulse_response(fs, L, center_freqs[i], p)
    return filterbank


def gfcc(cochleagram, numcep=13):
    feat = dct(cochleagram, type=2, axis=1, norm='ortho')[:, :numcep]
    #    feat-= (np.mean(feat, axis=0) + 1e-8)#Cepstral mean substration
    return feat


def cochleagram(sig_spec, filterbank, nfft):
    """
    :param sig_spec: 语音频谱
    :param filterbank: 时域滤波器组
    :param nfft: fft_size
    :return:
    """
    filterbank = powerspec(filterbank, nfft)  # 时域滤波器组经过FFT变换
    filterbank /= np.max(filterbank, axis=-1)[:, None]  # Normalize filters
    cochlea_spec = np.dot(sig_spec, filterbank.T)  # 矩阵相乘
    cochlea_spec = np.where(cochlea_spec == 0.0, np.finfo(float).eps, cochlea_spec)  # 把0变成一个很小的数
    # cochlea_spec= np.log(cochlea_spec)-np.mean(np.log(cochlea_spec),axis=0)
    cochlea_spec = np.log(cochlea_spec)
    return cochlea_spec, filterbank


def powerspec(X, nfft):
    # Fourier transform
    # Y = np.fft.rfft(X, n=n_padded)
    Y = np.fft.fft(X, n=nfft)
    Y = np.absolute(Y)

    # non-redundant part
    m = int(nfft / 2) + 1
    Y = Y[:, :m]

    return np.abs(Y) ** 2


if __name__ == "__main__":
    nfilts = 22
    NFFT = 512
    fs = 20000
    Order = 4

    FFT_len = NFFT // 2 + 1
    fs_bin = fs // 2 / (NFFT // 2)  # 一个频点多少Hz
    x = np.linspace(0, FFT_len, FFT_len)
    # ================ 画三角滤波器 ===========================
    # gammatone_impulse_response = gammatone_impulse_response(fs/2, 512, 200, Order)    #  gammatone冲击响应
    generate_filterbank = generate_filterbank(fs, fs // 2, FFT_len, nfilts, Order)
    filterbanks = powerspec(generate_filterbank, NFFT)  # 时域滤波器组经过FFT变换
    filterbanks /= np.max(filterbanks, axis=-1)[:, None]  # Normalize filters
    print(generate_filterbank.shape)    # (22, 257)
    # plt.plot(filterbanks.T)
    plt.xlabel('频率/Hz', fontsize=14)
    plt.ylabel('振幅', fontsize=14)
    plt.plot(x * fs_bin, filterbanks.T)
    # plt.xlim(0)  # 坐标轴的范围
    # plt.ylim(0, 1)
    plt.tight_layout()
    plt.grid(linestyle='--')
    plt.show()

    # ================ 绘制语谱图 ==========================
    wav = librosa.load("test_one.wav", sr=fs)[0]
    S = librosa.stft(wav, n_fft=NFFT, hop_length=NFFT // 2, win_length=NFFT, window="hann", center=False)
    mag = np.abs(S)  # 幅度谱 (257, 127) librosa.magphase()

    filter_banks = np.dot(filterbanks, mag)  # (M,F)*(F,T)=(M,T)
    filter_banks = 20 * np.log10(filter_banks)  # dB

    plt.figure()
    librosa.display.specshow(filter_banks, sr=fs, x_axis='time', y_axis='linear', cmap="jet")
    plt.xlabel('时间/s', fontsize=14)
    plt.ylabel('频率/kHz', fontsize=14)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)


    def formatnum(x, pos):
        return '$%d$' % (x / 1000)


    formatter = FuncFormatter(formatnum)
    plt.gca().yaxis.set_major_formatter(formatter)
    plt.tight_layout()
    plt.show()


# -*- coding:utf-8 -*-
# Author:凌逆战 | Never
# Date: 2022/5/19
"""
1、提取Mel filterBank
2、提取mel spectrum
"""
import librosa
import numpy as np
import matplotlib.pyplot as plt
import librosa.display
from matplotlib.ticker import FuncFormatter
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示符号

def hz2mel(hz):
    """ Hz to Mels """
    return 2595 * np.log10(1 + hz / 700.0)


def mel2hz(mel):
    """ Mels to HZ """
    return 700 * (10 ** (mel / 2595.0) - 1)


def mel_filterbanks(nfilt=20, nfft=512, samplerate=16000, lowfreq=0, highfreq=None):
    """计算一个Mel-filterbank (M,F)
    :param nfilt: filterbank中的滤波器数量
    :param nfft: FFT size
    :param samplerate: 采样率
    :param lowfreq: Mel-filter的最低频带边缘
    :param highfreq: Mel-filter的最高频带边缘，默认samplerate/2
    """
    highfreq = highfreq or samplerate / 2

    # 按梅尔均匀间隔计算 点
    lowmel = hz2mel(lowfreq)
    highmel = hz2mel(highfreq)
    melpoints = np.linspace(lowmel, highmel, nfilt + 2)
    hz_points = mel2hz(melpoints)  # 将mel频率再转到hz频率
    # bin = samplerate/2 / NFFT/2=sample_rate/NFFT    # 每个频点的频率数
    # bins = hz_points/bin=hz_points*NFFT/ sample_rate    # hz_points对应第几个fft频点
    bin = np.floor((nfft + 1) * hz_points / samplerate)

    fbank = np.zeros([nfilt, int(nfft / 2 + 1)])  # (m,f)
    for i in range(0, nfilt):
        for j in range(int(bin[i]), int(bin[i + 1])):
            fbank[i, j] = (j - bin[i]) / (bin[i + 1] - bin[i])
        for j in range(int(bin[i + 1]), int(bin[i + 2])):
            fbank[i, j] = (bin[i + 2] - j) / (bin[i + 2] - bin[i + 1])

    #    fbank -= (np.mean(fbank, axis=0) + 1e-8)
    return fbank


wav_path = "./test_one.wav"
fs = 20000
NFFT = 512
win_length = 512
num_filter = 22
low_freq_mel = 0
high_freq_mel = hz2mel(fs // 2)  # 求最高hz频率对应的mel频率
mel_points = np.linspace(low_freq_mel, high_freq_mel, num_filter + 2)  # 在mel频率上均分成42个点
hz_points = mel2hz(mel_points)  # 将mel频率再转到hz频率
print(hz_points)

# bin = sample_rate/2 / NFFT/2=sample_rate/NFFT    # 每个频点的频率数
# bins = hz_points/bin=hz_points*NFFT/ sample_rate    # hz_points对应第几个fft频点
bins = np.floor((NFFT + 1) * hz_points / fs)
print(bins)
# [  0.   2.   5.   8.  12.  16.  20.  25.  31.  37.  44.  52.  61.  70.
#   81.  93. 107. 122. 138. 157. 178. 201. 227. 256.]

wav = librosa.load(wav_path, sr=fs)[0]
S = librosa.stft(wav, n_fft=NFFT, hop_length=NFFT // 2, win_length=win_length, window="hann", center=False)
mag = np.abs(S)  # 幅度谱 (257, 127) librosa.magphase()

filterbanks = mel_filterbanks(nfilt=num_filter, nfft=NFFT, samplerate=fs, lowfreq=0, highfreq=fs // 2)

# ================ 画三角滤波器 ===========================
FFT_len = NFFT // 2 + 1
fs_bin = fs // 2 / (NFFT // 2)  # 一个频点多少Hz
x = np.linspace(0, FFT_len, FFT_len)

plt.plot(x * fs_bin, filterbanks.T)
plt.xlabel('频率/Hz', fontsize=14)
plt.ylabel('振幅', fontsize=14)
plt.xlim(0)  # 坐标轴的范围
plt.ylim(0, 1)
plt.tight_layout()
plt.grid(linestyle='--')
plt.show()

filter_banks = np.dot(filterbanks, mag)  # (M,F)*(F,T)=(M,T)
filter_banks = 20 * np.log10(filter_banks)  # dB

# ================ 绘制语谱图 ==========================
# 绘制 频谱图 方法1
plt.imshow(filter_banks, cmap="jet", aspect='auto')
ax = plt.gca()  # 获取其中某个坐标系
ax.invert_yaxis()  # 将y轴反转
plt.tight_layout()
plt.show()

# 绘制 频谱图 方法2
plt.figure()
librosa.display.specshow(filter_banks, sr=fs, x_axis='time', y_axis='linear', cmap="jet")
plt.xlabel('时间/s', fontsize=14)
plt.ylabel('频率/kHz', fontsize=14)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
def formatnum(x, pos):
    return '$%d$' % (x / 1000)

formatter = FuncFormatter(formatnum)
plt.gca().yaxis.set_major_formatter(formatter)
plt.tight_layout()
plt.show()
