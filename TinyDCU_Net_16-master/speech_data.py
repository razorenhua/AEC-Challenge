import os
import sys
import logging
import traceback
import json
import librosa
import random
import time
import threading
import torch
import torch.nn as nn
import numpy as np
from utils.signalprocess import *
from utils.tools import *
from utils.istft import ISTFT

try:
    from Queue import Queue
except ImportError:
    from queue import Queue


class Producer(threading.Thread):
    def __init__(self, reader):
        threading.Thread.__init__(self)
        self.reader = reader
        self.exitcode = 0
        self.stop_flag = False

    def run(self):
        try:
            min_queue_size = self.reader.cfg.min_queue_size
            while not self.stop_flag:
                idx = self.reader.next_produce_idx
                if idx < len(self.reader.clean_wav_list):
                    if self.reader.batch_queue.qsize() < min_queue_size:
                        group_list = self.reader.load_one_batch()
                        for batch in group_list:
                            self.reader.batch_queue.put(batch)
                    else:
                        time.sleep(1)
                else:
                    time.sleep(1)
        except Exception as e:
            logging.warning("producer exception: %s" % e)
            self.exitcode = 1
            traceback.print_exc()

    def stop(self):
        self.stop_flag = True


def get_file_line(file_list, cfg=None):
    line_list = []
    with open(file_list, 'r') as f:
        for line in f.readlines():
            line = line.strip().split()[0]
            if cfg is not None:
                sig_data = audio_read(line, samp_rate=cfg.sample_rate)
                if len(sig_data) >= cfg.chunk_length:
                    line_list.append(line)
            else:
                line_list.append(line)
    return line_list


def compute_reverberation(config, clean_sig, impulse_sig):
    fftsize = 16384
    if len(impulse_sig) < fftsize:
        impulse_sig = np.concatenate((impulse_sig, np.zeros((fftsize - len(impulse_sig),))))
    erir_sig = np.zeros((fftsize,))
    lrir_sig = np.zeros((fftsize,))
    erir_sig[:1024] = impulse_sig[:1024]
    lrir_sig[:fftsize] = impulse_sig[:fftsize]
    prhk = rfft(lrir_sig)
    erhk = rfft(erir_sig)
    frames = samples_to_stft_frames(len(clean_sig), size=config.frame_size, shift=config.frame_shift)
    rfrm = np.zeros((fftsize,))
    efrm = np.zeros((fftsize,))
    reverb_sig = np.zeros((len(clean_sig),))
    earlyr_sig = np.zeros((len(clean_sig),))
    for i in range(frames):
        xinp = np.zeros((fftsize,))
        xinp[:config.frame_shift] = clean_sig[i * config.frame_shift:(i + 1) * config.frame_shift]
        xink = rfft(xinp)
        rink = xink * prhk
        eink = xink * erhk
        time_signal = irfft(rink)
        rfrm[:(fftsize - config.frame_shift)] = time_signal[:(fftsize - config.frame_shift)] \
                                                + rfrm[config.frame_shift:fftsize]
        rfrm[(fftsize - config.frame_shift):fftsize] = time_signal[(fftsize - config.frame_shift):fftsize]
        time_signal = irfft(eink)
        efrm[:(fftsize - config.frame_shift)] = time_signal[:(fftsize - config.frame_shift)] \
                                                + efrm[config.frame_shift:fftsize]
        efrm[(fftsize - config.frame_shift):fftsize] = time_signal[(fftsize - config.frame_shift):fftsize]
        reverb_sig[i * config.frame_shift:(i + 1) * config.frame_shift] = rfrm[:config.frame_shift]
        earlyr_sig[i * config.frame_shift:(i + 1) * config.frame_shift] = efrm[:config.frame_shift]
    return earlyr_sig, reverb_sig


def compute_features(config, noisy_sig, clean_sig):
    frames = samples_to_stft_frames(len(clean_sig), size=config.frame_size, shift=config.frame_shift)
    sent_width = frames // 16
    sent_width = sent_width * 16
    sent_height = config.frame_size / 2 + 1
    clean_stft = stft_analysis(clean_sig, size=config.frame_size, shift=config.frame_shift)
    noisy_stft = stft_analysis(noisy_sig, size=config.frame_size, shift=config.frame_shift)
    clean_stft = clean_stft[:sent_width, :sent_height]
    noisy_stft = noisy_stft[:sent_width, :sent_height]
    frames = clean_stft.shape[0]
    frebin = clean_stft.shape[1]
    clean_feat = np.vstack((np.real(clean_stft), np.imag(clean_stft)))
    noisy_feat = np.vstack((np.real(noisy_stft), np.imag(noisy_stft)))
    noisy_feat = np.reshape(noisy_feat, [1, 2, frames, frebin])
    clean_feat = np.reshape(clean_feat, [1, 2, frames, frebin])
    return noisy_feat, clean_feat


def compute_reverb_features(config, noise_sig, reverb_sig, early_sig):
    reverb_power = np.mean(np.square(reverb_sig))
    noise_power = np.mean(np.square(noise_sig))
    snr = 12 * (np.random.rand() + 0.25)
    scale = np.sqrt(reverb_power / noise_power) * 10 ** (-snr / 10)
    # get mixture sig
    if len(reverb_sig) >= len(noise_sig):
        repeat_num = np.ceil(len(reverb_sig) / len(noise_sig)).astype(np.int32)
        repeat_noise_sig = np.tile(scale * noise_sig, repeat_num)
    else:
        repeat_idx = np.random.randint(len(noise_sig) - len(reverb_sig))
        repeat_noise_sig = scale * noise_sig[repeat_idx:repeat_idx + len(reverb_sig)]
    noisy_sig = reverb_sig + repeat_noise_sig[:len(reverb_sig)]
    normAmp = np.random.rand()
    sent_height = config.frame_size / 2 + 1
    # normAmp = np.sqrt(len(clean_sig) / np.sum(clean_sig ** 2.0))
    if normAmp < 0.1:
        normAmp = 0.1
    early_sig = early_sig * normAmp
    noisy_sig = noisy_sig * normAmp
    clean_stft = stft_analysis(early_sig, size=config.frame_size, shift=config.frame_shift)
    noisy_stft = stft_analysis(noisy_sig, size=config.frame_size, shift=config.frame_shift)
    clean_stft = clean_stft[:, :sent_height]
    noisy_stft = noisy_stft[:, :sent_height]
    frames = samples_to_stft_frames(len(early_sig), size=config.frame_size, shift=config.frame_shift)
    sent_width = frames // 16
    sent_width = 16 * sent_width
    if frames > sent_width:
        start_idx = np.random.randint(frames - sent_width)
    else:
        start_idx = 0
    clean_stft = clean_stft[start_idx:start_idx + sent_width, :sent_height]
    noisy_stft = noisy_stft[start_idx:start_idx + sent_width, :sent_height]
    clean_feat = np.vstack((np.real(clean_stft), np.imag(clean_stft)))
    noisy_feat = np.vstack((np.real(noisy_stft), np.imag(noisy_stft)))
    noisy_feats = np.reshape(noisy_feat, [1, 2, sent_width, sent_height])
    clean_feats = np.reshape(clean_feat, [1, 2, sent_width, sent_height])
    return noisy_feats, clean_feats


class SpeechReader(object):
    def __init__(self, config, job_type, clean_list=None, noisy_list=None, impulse_list=None):
        self.cfg = config
        self.job_type = job_type
        if clean_list is not None and noisy_list is not None:
            self.clean_wav_list = get_file_line(clean_list, config)
            self.noisy_wav_list = get_file_line(noisy_list, config)
        else:
            if job_type is not None:
                json_path = os.path.join(config.json_dir, self.job_type, 'files.json')
                with open(json_path, 'r') as f:
                    json_list = json.load(f)
                random.shuffle(json_list)
                self.clean_wav_list = []
                self.noisy_wav_list = []
                for wav_file_name in json_list:
                    clean_wav_file_path = os.path.join(config.dataset_dir, self.job_type, 'clean', wav_file_name)
                    noisy_wav_file_path = os.path.join(config.dataset_dir, self.job_type, 'mix', wav_file_name)
                    self.clean_wav_list.append(clean_wav_file_path)
                    self.noisy_wav_list.append(noisy_wav_file_path)
        if impulse_list is not None:
            self.impulse_wav_list = get_file_line(impulse_list)
        self.next_produce_idx = 0
        self.next_consume_idx = 0
        self.running_out_flag = 0
        self.batch_count = 0
        self.narray_window = analysis_window(config.frame_size, config.frame_shift)

    def __getitem__(self, index):
        """Reads an wave file and preprocesses it and returns."""
        print(index)
        clean_file = self.clean_wav_list[index]
        # clean_sig = audio_read(clean_file, samp_rate=self.cfg.sample_rate)
        # noise_file = self.noise_wav_list[np.random.randint(len(self.noise_wav_list))]
        # noise_sig = audio_read(noise_file, samp_rate=self.cfg.sample_rate)
        # noisy_feat, clean_feat, noisy_sig = compute_features(self.cfg, noise_sig, clean_sig)
        noisy_feat, clean_feat, noisy_sig, clean_sig = 0, 0, 0, 0
        return noisy_feat, clean_feat, noisy_sig, clean_sig

    def __len__(self):
        """Returns the total number of clean files."""
        return len(self.clean_wav_list)

    def start(self):
        self.next_produce_idx = 0
        self.next_consume_idx = 0
        self.running_out_flag = 0

    def reset(self):
        self.next_produce_idx = 0
        self.next_consume_idx = 0
        self.running_out_flag = 0
        self.batch_count = 0
        json_path = os.path.join(self.cfg.json_dir, self.job_type, 'files.json')
        with open(json_path, 'r') as f:
            json_list = json.load(f)
        random.shuffle(json_list)
        self.clean_wav_list = []
        self.noisy_wav_list = []
        for wav_file_name in json_list:
            clean_wav_file_path = os.path.join(self.cfg.dataset_dir, self.job_type, 'clean', wav_file_name)
            noisy_wav_file_path = os.path.join(self.cfg.dataset_dir, self.job_type, 'mix', wav_file_name)
            self.clean_wav_list.append(clean_wav_file_path)
            self.noisy_wav_list.append(noisy_wav_file_path)

    def shuffle_data_list(self):
        random.shuffle(self.noisy_wav_list)

    def load_samples(self, noisy_name):
        noisy_sig = audio_read(noisy_name, samp_rate=self.cfg.sample_rate)
        noisy_stft = stft_analysis(noisy_sig, size=self.cfg.frame_size, shift=self.cfg.frame_shift)
        frames = samples_to_stft_frames(len(noisy_sig), size=self.cfg.frame_size, shift=self.cfg.frame_shift)
        return noisy_sig, noisy_stft, frames

    def load_one_mixture(self):
        """Reads an wave file and preprocesses it and returns."""
        clean_file = self.clean_wav_list[np.random.randint(len(self.clean_wav_list))]
        noise_file = self.noisy_wav_list[np.random.randint(len(self.noisy_wav_list))]
        clean_sig = audio_read(clean_file, samp_rate=self.cfg.sample_rate)
        noise_sig = audio_read(noise_file, samp_rate=self.cfg.sample_rate)
        clean_power = np.mean(np.square(clean_sig))
        noise_power = np.mean(np.square(noise_sig))
        snr = 20 * (np.random.rand() - 0.25)
        scale = np.sqrt(clean_power / noise_power) * 10 ** (-snr / 10)
        # get mixture sig
        if len(clean_sig) >= len(noise_sig):
            repeat_num = np.ceil(len(clean_sig) / len(noise_sig)).astype(np.int32)
            repeat_noise_sig = np.tile(scale * noise_sig, repeat_num)
        else:
            repeat_idx = np.random.randint(len(noise_sig) - len(clean_sig))
            repeat_noise_sig = scale * noise_sig[repeat_idx:repeat_idx + len(clean_sig)]
        noisy_sig = clean_sig + repeat_noise_sig[:len(clean_sig)]
        # dump wav file
        # audio_write('./wav/clean.wav', clean_sig, 16000)
        # audio_write('./wav/noisy.wav', noisy_sig, 16000)
        frames = samples_to_stft_frames(len(noisy_sig), size=self.cfg.frame_size, shift=self.cfg.frame_shift)
        clean_stft = stft_analysis(clean_sig, size=self.cfg.frame_size, shift=self.cfg.frame_shift)
        noisy_stft = stft_analysis(noisy_sig, size=self.cfg.frame_size, shift=self.cfg.frame_shift)
        clean_magn = np.abs(clean_stft)
        noisy_magn = np.abs(noisy_stft)
        return noisy_sig, noisy_stft, noisy_magn, clean_sig, clean_stft, clean_magn, frames

    def load_one_item(self):
        """Reads an wave file and preprocesses it and returns."""
        clean_file = self.clean_wav_list[self.next_consume_idx]
        clean_sig = audio_read(clean_file, samp_rate=self.cfg.sample_rate)
        noisy_file = self.noisy_wav_list[self.next_consume_idx]
        noisy_sig = audio_read(noisy_file, samp_rate=self.cfg.sample_rate)
        clean_sig = np.reshape(clean_sig, [1, len(clean_sig)])
        noisy_sig = np.reshape(noisy_sig, [1, len(noisy_sig)])
        self.next_consume_idx = min(self.next_consume_idx + 1, len(self.clean_wav_list))
        return noisy_sig, clean_sig

    def load_one_batch(self):
        """Reads an wave file and preprocesses it and returns."""
        noisy_list, clean_list, frame_list = [], [], []
        for i in range(self.cfg.batch_size):
            next_consume_idx = self.next_consume_idx % len(self.clean_wav_list)
            clean_file = self.clean_wav_list[next_consume_idx]
            clean_sig = audio_read(clean_file, samp_rate=self.cfg.sample_rate)
            noisy_file = self.noisy_wav_list[next_consume_idx]
            noisy_sig = audio_read(noisy_file, samp_rate=self.cfg.sample_rate)
            # clean_sig = np.reshape(clean_sig, [1, len(clean_sig)])
            # noisy_sig = np.reshape(noisy_sig, [1, len(noisy_sig)])
            clean_sig = torch.FloatTensor(clean_sig)
            noisy_sig = torch.FloatTensor(noisy_sig)
            if len(clean_sig) > self.cfg.chunk_length:
                wav_start = random.randint(0, len(clean_sig) - self.cfg.chunk_length)
                clean_sig = clean_sig[wav_start:wav_start + self.cfg.chunk_length]
                noisy_sig = noisy_sig[wav_start:wav_start + self.cfg.chunk_length]
            frame_num = len(clean_sig) // self.cfg.frame_shift + 1
            clean_list.append(clean_sig)
            noisy_list.append(noisy_sig)
            frame_list.append(frame_num)
            self.next_consume_idx = self.next_consume_idx + 1
        if self.next_consume_idx >= len(self.clean_wav_list):
            self.running_out_flag = 1
        clean_list = nn.utils.rnn.pad_sequence(clean_list, batch_first=True)
        noisy_list = nn.utils.rnn.pad_sequence(noisy_list, batch_first=True)
        # print('self.next_produce_idx: ' + str(self.next_produce_idx))
        return noisy_list, clean_list, frame_list

    def load_one_norm_norm_batch(self):
        """Reads an wave file and preprocesses it and returns."""
        noisy_feat_list, clean_feat_list, frame_list = [], [], []
        for i in range(self.cfg.batch_size):
            next_consume_idx = self.next_consume_idx % len(self.clean_wav_list)
            clean_file = self.clean_wav_list[next_consume_idx]
            clean_sig = audio_read(clean_file, samp_rate=self.cfg.sample_rate)
            noisy_file = self.noisy_wav_list[next_consume_idx]
            noisy_sig = audio_read(noisy_file, samp_rate=self.cfg.sample_rate)
            if len(clean_sig) > self.cfg.chunk_length:
                wav_start = random.randint(0, len(clean_sig) - self.cfg.chunk_length)
                clean_sig = clean_sig[wav_start:wav_start + self.cfg.chunk_length]
                noisy_sig = noisy_sig[wav_start:wav_start + self.cfg.chunk_length]
            frame_num = len(clean_sig) // self.cfg.frame_shift + 1
            noisy_feat = librosa.stft(noisy_sig, n_fft=self.cfg.frame_size, hop_length=self.cfg.frame_shift,
                                      window=self.narray_window, pad_mode='constant').T
            clean_feat = librosa.stft(clean_sig, n_fft=self.cfg.frame_size, hop_length=self.cfg.frame_shift,
                                      window=self.narray_window, pad_mode='constant').T
            noisy_feat, clean_feat = noisy_feat[0:frame_num, :], clean_feat[0:frame_num, :]
            noisy_real, noisy_imag = np.real(noisy_feat), np.imag(noisy_feat)
            clean_real, clean_imag = np.real(clean_feat), np.imag(clean_feat)
            noisy_feat = torch.FloatTensor(np.concatenate(
                (noisy_real[:, :, np.newaxis].astype(np.float32), noisy_imag[:, :, np.newaxis].astype(np.float32)),
                axis=-1))
            clean_feat = torch.FloatTensor(np.concatenate(
                (clean_real[:, :, np.newaxis].astype(np.float32), clean_imag[:, :, np.newaxis].astype(np.float32)),
                axis=-1))
            noisy_feat_list.append(noisy_feat)
            clean_feat_list.append(clean_feat)
            frame_list.append(frame_num)
            self.next_consume_idx = self.next_consume_idx + 1
        if self.next_consume_idx >= len(self.clean_wav_list):
            self.running_out_flag = 1
        self.batch_count = self.batch_count + 1
        noisy_feat_list = nn.utils.rnn.pad_sequence(noisy_feat_list, batch_first=True)
        clean_feat_list = nn.utils.rnn.pad_sequence(clean_feat_list, batch_first=True)
        noisy_feat_list = noisy_feat_list.permute(0, 3, 1, 2).contiguous()
        clean_feat_list = clean_feat_list.permute(0, 3, 1, 2).contiguous()
        # print('self.next_produce_idx: ' + str(self.next_produce_idx))
        return noisy_feat_list, clean_feat_list, frame_list

    def load_one_comp_norm_batch(self):
        """Reads an wave file and preprocesses it and returns."""
        noisy_feat_list, clean_feat_list, frame_list = [], [], []
        for i in range(self.cfg.batch_size):
            next_consume_idx = self.next_consume_idx % len(self.clean_wav_list)
            clean_file = self.clean_wav_list[next_consume_idx]
            clean_sig = audio_read(clean_file, samp_rate=self.cfg.sample_rate)
            noisy_file = self.noisy_wav_list[next_consume_idx]
            noisy_sig = audio_read(noisy_file, samp_rate=self.cfg.sample_rate)
            if len(clean_sig) > self.cfg.chunk_length:
                wav_start = random.randint(0, len(clean_sig) - self.cfg.chunk_length)
                clean_sig = clean_sig[wav_start:wav_start + self.cfg.chunk_length]
                noisy_sig = noisy_sig[wav_start:wav_start + self.cfg.chunk_length]
            frame_num = len(clean_sig) // self.cfg.frame_shift + 1
            noisy_feat = librosa.stft(noisy_sig, n_fft=self.cfg.frame_size, hop_length=self.cfg.frame_shift,
                                      window=self.narray_window, pad_mode='constant').T
            clean_feat = librosa.stft(clean_sig, n_fft=self.cfg.frame_size, hop_length=self.cfg.frame_shift,
                                      window=self.narray_window, pad_mode='constant').T
            noisy_feat, clean_feat = noisy_feat[0:frame_num, :], clean_feat[0:frame_num, :]
            noisy_mag, noisy_phase = np.abs(noisy_feat), np.angle(noisy_feat)
            noisy_mag_com = np.sqrt(noisy_mag)
            noisy_real, noisy_imag = noisy_mag_com * np.cos(noisy_phase), noisy_mag_com * np.sin(noisy_phase)
            clean_real, clean_imag = np.real(clean_feat), np.imag(clean_feat)
            noisy_feat = torch.FloatTensor(np.concatenate(
                (noisy_real[:, :, np.newaxis].astype(np.float32), noisy_imag[:, :, np.newaxis].astype(np.float32)),
                axis=-1))
            clean_feat = torch.FloatTensor(np.concatenate(
                (clean_real[:, :, np.newaxis].astype(np.float32), clean_imag[:, :, np.newaxis].astype(np.float32)),
                axis=-1))
            noisy_feat_list.append(noisy_feat)
            clean_feat_list.append(clean_feat)
            frame_list.append(frame_num)
            self.next_consume_idx = self.next_consume_idx + 1
        if self.next_consume_idx >= len(self.clean_wav_list):
            self.running_out_flag = 1
        self.batch_count = self.batch_count + 1
        noisy_feat_list = nn.utils.rnn.pad_sequence(noisy_feat_list, batch_first=True)
        clean_feat_list = nn.utils.rnn.pad_sequence(clean_feat_list, batch_first=True)
        noisy_feat_list = noisy_feat_list.permute(0, 3, 1, 2).contiguous()
        clean_feat_list = clean_feat_list.permute(0, 3, 1, 2).contiguous()
        # print('self.next_produce_idx: ' + str(self.next_produce_idx))
        return noisy_feat_list, clean_feat_list, frame_list

    def load_one_norm_comp_batch(self):
        """Reads an wave file and preprocesses it and returns."""
        noisy_feat_list, clean_feat_list, frame_list = [], [], []
        for i in range(self.cfg.batch_size):
            next_consume_idx = self.next_consume_idx % len(self.clean_wav_list)
            clean_file = self.clean_wav_list[next_consume_idx]
            clean_sig = audio_read(clean_file, samp_rate=self.cfg.sample_rate)
            noisy_file = self.noisy_wav_list[next_consume_idx]
            noisy_sig = audio_read(noisy_file, samp_rate=self.cfg.sample_rate)
            if len(clean_sig) > self.cfg.chunk_length:
                wav_start = random.randint(0, len(clean_sig) - self.cfg.chunk_length)
                clean_sig = clean_sig[wav_start:wav_start + self.cfg.chunk_length]
                noisy_sig = noisy_sig[wav_start:wav_start + self.cfg.chunk_length]
            frame_num = len(clean_sig) // self.cfg.frame_shift + 1
            noisy_feat = librosa.stft(noisy_sig, n_fft=self.cfg.frame_size, hop_length=self.cfg.frame_shift,
                                      window=self.narray_window, pad_mode='constant').T
            clean_feat = librosa.stft(clean_sig, n_fft=self.cfg.frame_size, hop_length=self.cfg.frame_shift,
                                      window=self.narray_window, pad_mode='constant').T
            noisy_feat, clean_feat = noisy_feat[0:frame_num, :], clean_feat[0:frame_num, :]
            clean_mag, clean_phase = np.abs(clean_feat), np.angle(clean_feat)
            clean_mag_com = np.sqrt(clean_mag)
            noisy_real, noisy_imag = np.real(noisy_feat), np.imag(noisy_feat)
            clean_real, clean_imag = clean_mag_com * np.cos(clean_phase), clean_mag_com * np.sin(clean_phase)
            noisy_feat = torch.FloatTensor(np.concatenate(
                (noisy_real[:, :, np.newaxis].astype(np.float32), noisy_imag[:, :, np.newaxis].astype(np.float32)),
                axis=-1))
            clean_feat = torch.FloatTensor(np.concatenate(
                (clean_real[:, :, np.newaxis].astype(np.float32), clean_imag[:, :, np.newaxis].astype(np.float32)),
                axis=-1))
            noisy_feat_list.append(noisy_feat)
            clean_feat_list.append(clean_feat)
            frame_list.append(frame_num)
            self.next_consume_idx = self.next_consume_idx + 1
        if self.next_consume_idx >= len(self.clean_wav_list):
            self.running_out_flag = 1
        self.batch_count = self.batch_count + 1
        noisy_feat_list = nn.utils.rnn.pad_sequence(noisy_feat_list, batch_first=True)
        clean_feat_list = nn.utils.rnn.pad_sequence(clean_feat_list, batch_first=True)
        noisy_feat_list = noisy_feat_list.permute(0, 3, 1, 2).contiguous()
        clean_feat_list = clean_feat_list.permute(0, 3, 1, 2).contiguous()
        # print('self.next_produce_idx: ' + str(self.next_produce_idx))
        return noisy_feat_list, clean_feat_list, frame_list

    def load_one_comp_comp_batch(self):
        """Reads an wave file and preprocesses it and returns."""
        noisy_feat_list, clean_feat_list, frame_list = [], [], []
        for i in range(self.cfg.batch_size):
            next_consume_idx = self.next_consume_idx % len(self.clean_wav_list)
            clean_file = self.clean_wav_list[next_consume_idx]
            clean_sig = audio_read(clean_file, samp_rate=self.cfg.sample_rate)
            noisy_file = self.noisy_wav_list[next_consume_idx]
            noisy_sig = audio_read(noisy_file, samp_rate=self.cfg.sample_rate)
            if len(clean_sig) > self.cfg.chunk_length:
                wav_start = random.randint(0, len(clean_sig) - self.cfg.chunk_length)
                clean_sig = clean_sig[wav_start:wav_start + self.cfg.chunk_length]
                noisy_sig = noisy_sig[wav_start:wav_start + self.cfg.chunk_length]
            frame_num = len(clean_sig) // self.cfg.frame_shift + 1
            noisy_feat = librosa.stft(noisy_sig, n_fft=self.cfg.frame_size, hop_length=self.cfg.frame_shift,
                                      window=self.narray_window, pad_mode='constant').T
            clean_feat = librosa.stft(clean_sig, n_fft=self.cfg.frame_size, hop_length=self.cfg.frame_shift,
                                      window=self.narray_window, pad_mode='constant').T
            noisy_feat, clean_feat = noisy_feat[0:frame_num, :], clean_feat[0:frame_num, :]
            noisy_mag, noisy_phase = np.abs(noisy_feat), np.angle(noisy_feat)
            clean_mag, clean_phase = np.abs(clean_feat), np.angle(clean_feat)
            noisy_mag_com, clean_mag_com = np.sqrt(noisy_mag), np.sqrt(clean_mag)
            noisy_real, noisy_imag = noisy_mag_com * np.cos(noisy_phase), noisy_mag_com * np.sin(noisy_phase)
            clean_real, clean_imag = clean_mag_com * np.cos(clean_phase), clean_mag_com * np.sin(clean_phase)
            noisy_feat = torch.FloatTensor(np.concatenate(
                (noisy_real[:, :, np.newaxis].astype(np.float32), noisy_imag[:, :, np.newaxis].astype(np.float32)),
                axis=-1))
            clean_feat = torch.FloatTensor(np.concatenate(
                (clean_real[:, :, np.newaxis].astype(np.float32), clean_imag[:, :, np.newaxis].astype(np.float32)),
                axis=-1))
            noisy_feat_list.append(noisy_feat)
            clean_feat_list.append(clean_feat)
            frame_list.append(frame_num)
            self.next_consume_idx = self.next_consume_idx + 1
        if self.next_consume_idx >= len(self.clean_wav_list):
            self.running_out_flag = 1
        self.batch_count = self.batch_count + 1
        noisy_feat_list = nn.utils.rnn.pad_sequence(noisy_feat_list, batch_first=True)
        clean_feat_list = nn.utils.rnn.pad_sequence(clean_feat_list, batch_first=True)
        noisy_feat_list = noisy_feat_list.permute(0, 3, 1, 2).contiguous()
        clean_feat_list = clean_feat_list.permute(0, 3, 1, 2).contiguous()
        # print('self.next_produce_idx: ' + str(self.next_produce_idx))
        return noisy_feat_list, clean_feat_list, frame_list

    def is_running_out(self):
        if self.running_out_flag == 1:
            return True
        else:
            return False

    def next_batch(self):
        while self.producer.exitcode == 0:
            try:
                if self.batch_queue.qsize() > 0:
                    batch_data = self.batch_queue.get(block=False)
                    self.next_consume_idx = min(self.next_consume_idx + 1, len(self.clean_wav_list))
                    print('self.next_consume_idx: ' + str(self.next_consume_idx))
                    return batch_data
                else:
                    time.sleep(0.5)
            except Exception as e:
                time.sleep(3)


def get_reader(config, job_type=None, clean_list=None, noisy_list=None, impulse_list=None):
    data_reader = SpeechReader(config, job_type, clean_list, noisy_list, impulse_list)
    return data_reader
