import argparse
from torch.backends import cudnn
import torch
from network import TinyDCU_Net_16
from speech_data import get_reader
import torch.nn.functional as F
from utils.signalprocess import *
from utils.tools import *
import numpy as np


def main(config):
    cudnn.benchmark = False
    if config.model_type not in ['TinyDCU_Net_16']:
        print('ERROR!! model_type should be selected in TinyDCU_Net_16')
        print('Your input for model_type was %s' % config.model_type)
        return
    # config.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config.device = 'cpu'

    unet = None
    if config.model_type == 'TinyDCU_Net_16':
        unet = TinyDCU_Net_16()

    unet.to(config.device)
    unet.load_state_dict(torch.load(config.model_path))
    unet.eval()
    print('%s is Successfully Loaded from %s' % (config.model_type, config.model_path))
    test_reader = get_reader(config)
    noisy_sig, noisy_stft, frames = test_reader.load_samples(config.testing_filename)
    if config.model_type == 'TinyDCU_Net_16':
        noisy_feat = noisy_stft
        frames = noisy_stft.shape[0]
        frebin = noisy_stft.shape[1]
        noisy_feat = np.vstack((np.real(noisy_feat), np.imag(noisy_feat)))
        noisy_feat = np.reshape(noisy_feat, [1, 2, frames, frebin])
        noisy_feat = torch.tensor(noisy_feat, dtype=torch.float32).to(config.device)
        estimate_real, estimate_imag = unet(noisy_feat[:, 0, :, :].unsqueeze(1),
                                            noisy_feat[:, 1, :, :].unsqueeze(1))
        estimate_real = estimate_real.squeeze().cpu().detach().numpy()
        estimate_imag = estimate_imag.squeeze().cpu().detach().numpy()
        estimate_stft = estimate_real + 1j * estimate_imag

    estimate_sig = stft_synthesis(estimate_stft, size=test_reader.cfg.frame_size,
                                  shift=test_reader.cfg.frame_shift,
                                  fading=True, signal_length=len(noisy_sig))
    audio_write(config.output_filename, estimate_sig, 16000)

    # reconstruct enhance wav


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # data parameters
    parser.add_argument('--sample_rate', type=int, default=16000)
    parser.add_argument('--frame_size', type=int, default=320)
    parser.add_argument('--frame_shift', type=int, default=160)

    # model hyper-parameters
    parser.add_argument('--image_size', type=int, default=224)
    parser.add_argument('--t', type=int, default=1,
                        help='t for Recurrent step of R2U_Net or R2AttU_Net and Interation step of DARCCN')

    # training hyper-parameters
    parser.add_argument('--img_ch', type=int, default=1)
    parser.add_argument('--output_ch', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--testing_filename', type=str,
                        default='./wav/nearend_mic_aec_fileid_1814_DCGRU_Net_22.wav')
    parser.add_argument('--output_filename', type=str,
                        default='./wav/nearend_mic_aec_fileid_1814_DCGRU_Net_22_TinyDCU_Net_16.wav')

    # misc
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--model_type', type=str, default='TinyDCU_Net_16', help='TinyDCU_Net_16')
    parser.add_argument('--result_path', type=str, default='./result')
    parser.add_argument('--model_path', type=str, default='./models/TinyDCU_Net_16-200.pkl')

    parser.add_argument('--cuda_idx', type=int, default=1)

    config = parser.parse_args()
    main(config)
