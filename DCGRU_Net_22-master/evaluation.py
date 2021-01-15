import argparse
from torch.backends import cudnn
import torch
from network import DCGRU_Net_22
from speech_data import get_reader
import torch.nn.functional as F
from utils.signalprocess import *
from utils.tools import *
import numpy as np


def main(config):
    cudnn.benchmark = False
    if config.model_type not in ['DCGRU_Net_22']:
        print('ERROR!! model_type should be selected in DCGRU_Net_22')
        print('Your input for model_type was %s' % config.model_type)
        return
    # config.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config.device = 'cpu'

    unet = None
    if config.model_type == 'DCGRU_Net_22':
        unet = DCGRU_Net_22()
    unet.to(config.device)
    unet.load_state_dict(torch.load(config.model_path))
    unet.eval()
    print('%s is Successfully Loaded from %s' % (config.model_type, config.model_path))

    test_reader = get_reader(config)
    input_featr, input_feati, des_sig, des_stft, frames = test_reader.load_samples(config.ref_file, config.des_file,
                                                                                   config.aec_file)
    input_featr = torch.tensor(input_featr, dtype=torch.float32).to(config.device)
    input_feati = torch.tensor(input_feati, dtype=torch.float32).to(config.device)
    if config.model_type == 'DCGRU_Net_22':
        estimate_real, estimate_imag = unet(input_featr, input_feati)
        estimate_real = estimate_real.squeeze().cpu().detach().numpy()
        estimate_imag = estimate_imag.squeeze().cpu().detach().numpy()
        estimate_stft = estimate_real + 1j * estimate_imag

    out_sig = stft_synthesis(estimate_stft, size=test_reader.cfg.frame_size,
                             shift=test_reader.cfg.frame_shift,
                             fading=True, signal_length=len(des_sig))

    audio_write(config.out_file, out_sig, test_reader.cfg.sample_rate)
    return 0
    # reconstruct enhance wav


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # data parameters
    parser.add_argument('--sample_rate', type=int, default=16000)
    parser.add_argument('--frame_size', type=int, default=320)
    parser.add_argument('--frame_shift', type=int, default=160)
    parser.add_argument('--sent_height', type=int, default=161)
    parser.add_argument('--min_queue_size', type=int, default=128)

    # model hyper-parameters
    parser.add_argument('--image_size', type=int, default=224)
    parser.add_argument('--t', type=int, default=1,
                        help='t for Recurrent step of R2U_Net or R2AttU_Net and Interation step of DARCCN')

    # training hyper-parameters
    parser.add_argument('--img_ch', type=int, default=1)
    parser.add_argument('--output_ch', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=8)

    parser.add_argument('--ref_file', type=str, default='./wav/farend_speech_fileid_1814.wav')
    parser.add_argument('--des_file', type=str, default='./wav/nearend_mic_fileid_1814.wav')
    parser.add_argument('--aec_file', type=str, default='./wav/nearend_mic_aec_fileid_1814.wav')
    parser.add_argument('--out_file', type=str, default='./wav/nearend_mic_aec_fileid_1814_DCGRU_Net_22.wav')

    # misc
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--model_type', type=str, default='DCGRU_Net_22', help='DCGRU_Net_22')
    parser.add_argument('--result_path', type=str, default='./result')
    parser.add_argument('--model_path', type=str, default='./models/DCGRU_Net_22-200.pkl')

    parser.add_argument('--cuda_idx', type=int, default=1)

    config = parser.parse_args()
    main(config)
