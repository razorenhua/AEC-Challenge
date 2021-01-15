import os
import sys
import logging
import traceback
import random
import time
import threading
import numpy as np
from utils.signalprocess import *
from utils.tools import *

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
                if idx < len(self.reader.ref_wav_list):
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


def get_file_line(file_list):
    line_list = []
    with open(file_list, 'r') as f:
        for line in f.readlines():
            line = line.strip().split()[0]
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


def compute_features(config, ref_sig, des_sig, aec_sig, tar_sig):
    itn = np.minimum(len(ref_sig), len(des_sig))
    itn = np.minimum(itn, len(aec_sig))
    itn = np.minimum(itn, len(tar_sig))
    ref_sig = ref_sig[:itn]
    des_sig = des_sig[:itn]
    aec_sig = aec_sig[:itn]
    tar_sig = tar_sig[:itn]
    normAmp1 = np.random.rand()
    normAmp2 = np.random.rand()
    if normAmp1 < 0.1:
        normAmp1 = 0.1
    if normAmp2 < 0.1:
        normAmp2 = 0.1
    ref_sig = ref_sig * normAmp1
    des_sig = des_sig * normAmp2
    aec_sig = aec_sig * normAmp2
    tar_sig = tar_sig * normAmp2
    ref_stft = stft_analysis(ref_sig, size=config.frame_size, shift=config.frame_shift)
    des_stft = stft_analysis(des_sig, size=config.frame_size, shift=config.frame_shift)
    aec_stft = stft_analysis(aec_sig, size=config.frame_size, shift=config.frame_shift)
    tar_stft = stft_analysis(tar_sig, size=config.frame_size, shift=config.frame_shift)
    frames = ref_stft.shape[0]
    frebin = ref_stft.shape[1]
    ref_feat = np.vstack((np.real(ref_stft), np.imag(ref_stft)))
    des_feat = np.vstack((np.real(des_stft), np.imag(des_stft)))
    aec_feat = np.vstack((np.real(aec_stft), np.imag(aec_stft)))
    ref_feat = np.reshape(ref_feat, [2, frames, frebin])
    des_feat = np.reshape(des_feat, [2, frames, frebin])
    aec_feat = np.reshape(aec_feat, [2, frames, frebin])
    input_featr = np.concatenate((ref_feat[0, :, :], des_feat[0, :, :], aec_feat[0, :, :]))
    input_featr = np.reshape(input_featr, [1, 3, frames, frebin])
    input_feati = np.concatenate((ref_feat[1, :, :], des_feat[1, :, :], aec_feat[1, :, :]))
    input_feati = np.reshape(input_feati, [1, 3, frames, frebin])

    tar_feat = np.vstack((np.real(tar_stft), np.imag(tar_stft)))
    output_feat = np.reshape(tar_feat, [1, 2, frames, frebin])
    return input_featr, input_feati, output_feat


def audio_drc(input_sig):
    PreGain = 1
    MinGain = 0.1
    TgtLevelUpp = 8192
    TgtLevelSup = 2048
    GainUp = 1.000045
    GainDown = 0.99983
    CurrGain = 1.0
    AttFactor = 1.0 - 1000.0 / (16000 * 0.5)
    DecayFactor = 1.0 - 1000.0 / (16000 * 50.0)
    TdEnv = 0
    output_sig = np.zeros(np.shape(input_sig))
    for i in range(len(input_sig)):
        tmpGain = PreGain * CurrGain
        tmpAbs = np.abs(tmpGain * input_sig[i])
        tmpFactor = DecayFactor
        if tmpAbs < TdEnv:
            TdEnv = tmpFactor * TdEnv + (1.0 - tmpFactor) * tmpAbs
        if TdEnv > TgtLevelUpp:
            CurrGain = CurrGain * GainDown
        elif TdEnv < TgtLevelSup:
            CurrGain = CurrGain * GainUp
        if CurrGain > 1.0:
            CurrGain = 1.0
        elif CurrGain < MinGain:
            CurrGain = MinGain
        output_sig[i] = tmpGain * input_sig[i]

    return output_sig


class SpeechReader(object):
    def __init__(self, config=None, ref_sig_list=None, des_sig_list=None, aec_sig_list=None, tar_sig_list=None,
                 job_type=None):
        self.cfg = config
        if ref_sig_list is not None:
            self.ref_wav_list = get_file_line(ref_sig_list)
        if des_sig_list is not None:
            self.des_wav_list = get_file_line(des_sig_list)
        if aec_sig_list is not None:
            self.aec_wav_list = get_file_line(aec_sig_list)
        if tar_sig_list is not None:
            self.tar_wav_list = get_file_line(tar_sig_list)
        self.job_type = job_type
        self.next_produce_idx = 0
        self.next_consume_idx = 0
        self.producer = None
        if self.job_type == "Train":
            self.batch_queue = Queue()
            self.start()

    def __getitem__(self, index):
        """Reads an wave file and preprocesses it and returns."""
        ref_file = self.ref_wav_list[index]
        ref_sig = audio_read(ref_file, samp_rate=self.cfg.sample_rate)
        des_file = self.des_wav_list[index]
        des_sig = audio_read(des_file, samp_rate=self.cfg.sample_rate)
        aec_file = self.aec_wav_list[index]
        aec_sig = audio_read(aec_file, samp_rate=self.cfg.sample_rate)
        tar_file = self.tar_wav_list[index]
        tar_sig = audio_read(tar_file, samp_rate=self.cfg.sample_rate)
        input_featr, input_feati, output_feat = compute_features(self.cfg, ref_sig, des_sig, aec_sig, tar_sig)
        return input_featr, input_feati, output_feat, des_sig, tar_sig

    def __len__(self):
        """Returns the total number of clean files."""
        return len(self.ref_wav_list)

    def start(self):
        self.next_produce_idx = 0
        self.producer = Producer(self)
        self.producer.start()

    def reset(self):
        if self.batch_queue is not None:
            self.batch_queue.empty()
        self.next_produce_idx = 0
        self.next_consume_idx = 0

    def load_samples(self, ref_file, des_file, aec_file):
        ref_sig = audio_read(ref_file, samp_rate=self.cfg.sample_rate)
        des_sig = audio_read(des_file, samp_rate=self.cfg.sample_rate)
        aec_sig = audio_read(aec_file, samp_rate=self.cfg.sample_rate)
        itn = np.minimum(len(ref_sig), len(des_sig))
        itn = np.minimum(itn, len(aec_sig))
        # ref_sig = audio_drc(ref_sig[:itn])
        ref_sig = ref_sig[:itn]
        des_sig = des_sig[:itn]
        aec_sig = aec_sig[:itn]
        ref_stft = stft_analysis(ref_sig, size=self.cfg.frame_size, shift=self.cfg.frame_shift)
        des_stft = stft_analysis(des_sig, size=self.cfg.frame_size, shift=self.cfg.frame_shift)
        aec_stft = stft_analysis(aec_sig, size=self.cfg.frame_size, shift=self.cfg.frame_shift)
        frames = ref_stft.shape[0]
        frebin = ref_stft.shape[1]
        ref_feat = np.vstack((np.real(ref_stft), np.imag(ref_stft)))
        des_feat = np.vstack((np.real(des_stft), np.imag(des_stft)))
        aec_feat = np.vstack((np.real(aec_stft), np.imag(aec_stft)))
        ref_feat = np.reshape(ref_feat, [2, frames, frebin])
        des_feat = np.reshape(des_feat, [2, frames, frebin])
        aec_feat = np.reshape(aec_feat, [2, frames, frebin])
        input_featr = np.concatenate((ref_feat[0, :, :], des_feat[0, :, :], aec_feat[0, :, :]))
        input_featr = np.reshape(input_featr, [1, 3, frames, frebin])
        input_feati = np.concatenate((ref_feat[1, :, :], des_feat[1, :, :], aec_feat[1, :, :]))
        input_feati = np.reshape(input_feati, [1, 3, frames, frebin])
        return input_featr, input_feati, des_sig, des_stft, frames

    def load_one_batch(self):
        """Reads an wave file and preprocesses it and returns."""
        group_list = []
        ref_file = self.ref_wav_list[self.next_produce_idx]
        ref_sig = audio_read(ref_file, samp_rate=self.cfg.sample_rate)
        des_file = self.des_wav_list[self.next_produce_idx]
        des_sig = audio_read(des_file, samp_rate=self.cfg.sample_rate)
        aec_file = self.aec_wav_list[self.next_produce_idx]
        aec_sig = audio_read(aec_file, samp_rate=self.cfg.sample_rate)
        tar_file = self.tar_wav_list[self.next_produce_idx]
        tar_sig = audio_read(tar_file, samp_rate=self.cfg.sample_rate)
        input_featr, input_feati, output_feat = compute_features(self.cfg, ref_sig, des_sig, aec_sig, tar_sig)
        group_list.append((input_featr, input_feati, output_feat, des_sig, tar_sig))
        self.next_produce_idx = min(self.next_produce_idx + 1, len(self.ref_wav_list))
        # print('self.next_produce_idx: ' + str(self.next_produce_idx))
        return group_list

    def is_running_out(self):
        if self.next_consume_idx == len(self.ref_wav_list):
            return True
        else:
            return False

    def next_batch(self):
        while self.producer.exitcode == 0:
            try:
                if self.batch_queue.qsize() > 0:
                    batch_data = self.batch_queue.get(block=False)
                    self.next_consume_idx = min(self.next_consume_idx + 1, len(self.ref_wav_list))
                    # print('self.next_consume_idx: '+str(self.next_consume_idx))
                    return batch_data
                else:
                    time.sleep(0.5)
            except Exception as e:
                time.sleep(3)


def get_reader(config=None, ref_sig_list=None, des_sig_list=None, aec_sig_list=None, tar_sig_list=None, job_type=None):
    data_reader = SpeechReader(config, ref_sig_list, des_sig_list, aec_sig_list, tar_sig_list, job_type)
    return data_reader
