import torch
from fairseq.tasks.wapat import GpuRirDemo, Parameter

import torchaudio
import matplotlib.pyplot as plt
import librosa

def plot_waveform(waveform, sample_rate, path, title="Waveform", xlim=None, ylim=None):
    waveform = waveform.numpy()
    num_channels, num_frames = waveform.shape
    time_axis = torch.arange(0, num_frames) / sample_rate
    figure, axes = plt.subplots(num_channels, 1)
    if num_channels == 1:
        axes = [axes]
    for c in range(num_channels):
        axes[c].plot(time_axis, waveform[c], linewidth=1)
        axes[c].grid(True)
        if num_channels > 1:
            axes[c].set_ylabel(f'Channel {c+1}')
        if xlim:
            axes[c].set_xlim(xlim)
        if ylim:
            axes[c].set_ylim(ylim)
    figure.suptitle(title)
    #plt.show()
    plt.savefig(path)

def plot_spectrogram(spec, path, title=None, ylabel='freq_bin', aspect='auto', xmax=None):
    fig, axs = plt.subplots(1, 1)
    axs.set_title(title or 'Spectrogram (db)')
    axs.set_ylabel(ylabel)
    axs.set_xlabel('frame')
    im = axs.imshow(librosa.power_to_db(spec), origin='lower', aspect=aspect)
    if xmax:
        axs.set_xlim((0, xmax))
    fig.colorbar(im, ax=axs)
    #plt.show()
    plt.savefig(path)

class Aug_guide_adv():
    def __init__(self) -> None:
        pass
    def aug_RIR(self, sample_room, tables):
        t60 = tables['t60']
        room_sz = tables['room_sz']
        beta = tables['beta']
        array_pos = tables['array_pos']
        test_code = GpuRirDemo(room_sz=room_sz, t60=t60, beta=beta, array_pos=array_pos, fs=16000)
        device_tmp = sample_room['net_input']['source'].device

        y_new = sample_room['net_input']['source']
        sample_room['net_input']['source'] = []
        for idx in range(len(y_new)):
            sample_room['net_input']['source'].append(test_code.simulate(y_new[idx])[:y_new.shape[1]])
        sample_room['net_input']['source'] = torch.tensor(sample_room['net_input']['source']).to(device_tmp).half()
        return sample_room


class Aug_guide():
    def __init__(self) -> None:
        pass
    def aug_RIR(self, sample_room):
        test_code = GpuRirDemo(
        room_sz=Parameter([3, 3, 2.5], [4, 5, 3]).getvalue(), # 此时得到随机得到[3,3,2.5]~[4,5,3]之间的一个房间尺寸
        t60=Parameter(0.2, 1.0).getvalue(), # 0.2s~1.0s之间的一个随机值
                        beta=Parameter([0.5]*6, [1.0]*6).getvalue(), # 房间反射系数
                        array_pos=Parameter([0.1, 0.1, 0.1], [0.9, 0.9, 0.5]).getvalue(),# 比例系数，实际的array_pos为 array_pos * room_sz
                        fs=16000)
        device_tmp = sample_room['net_input']['source'].device

        y_new = sample_room['net_input']['source']
        sample_room['net_input']['source'] = []
        for idx in range(len(y_new)):
            sample_room['net_input']['source'].append(test_code.simulate(y_new[idx])[:y_new.shape[1]])
        sample_room['net_input']['source'] = torch.tensor(sample_room['net_input']['source']).to(device_tmp).half()
        return sample_room
    def time_mask_augment(self, sample_room):
        #时域遮掩
        inputs = sample_room['net_input']['source']
        time_len = inputs.shape[1]
        max_mask_time = torch.randint(0, 5, (1,)).item()
        mask_num = torch.randint(0, 10, (1,)).item()
        for i in range(mask_num):
            t = np.random.uniform(low=0.0, high=max_mask_time)
            t = int(t)
            t0 = random.randint(0, time_len - t)
            inputs[:, t0:t0 + t] = 0
        sample_room['net_input']['source'] = inputs
        return sample_room
    def volume_augment(self, sample_room, min_gain_dBFS=-10, max_gain_dBFS=10):
        samples = sample_room['net_input']['source'].copy()  # frombuffer()导致数据不可更改因此使用拷贝
        data_type = samples[0].dtype
        gain = np.uniform(min_gain_dBFS, max_gain_dBFS)
        gain = 10. ** (gain / 20.)
        samples = samples * torch.tensor(gain)
        # improvement:保证输出的音频还是原类型，不然耳朵会聋
        samples = samples.astype(data_type)
        sample_room['net_input']['source'] = samples
        return sample_room
