import argparse
import os
from solver import Solver
from speech_data import get_reader
from torch.backends import cudnn
import random
import sys
import time


def main(config):
    cudnn.benchmark = False
    if config.model_type not in ['TinyDCU_Net_16']:
        print('ERROR!! model_type should be selected in:')
        print('TinyDCU_Net_16')
        print('Your input for model_type was %s' % config.model_type)
        return

    # Create directories if not exist
    if not os.path.exists(config.model_path):
        os.makedirs(config.model_path)
    if not os.path.exists(config.result_path):
        os.makedirs(config.result_path)
    config.result_path = os.path.join(config.result_path, config.model_type)
    if not os.path.exists(config.result_path):
        os.makedirs(config.result_path)
    if not os.path.exists(config.logfile_path):
        os.makedirs(config.logfile_path)
    config.logfile_path = os.path.join(config.logfile_path, config.model_type + '.log')

    lr = random.random() * 0.0005 + 0.0000005
    epoch = 200
    decay_ratio = random.random() * 0.8
    decay_epoch = int(epoch * decay_ratio)

    config.num_epochs = epoch
    config.lr = lr
    config.num_epochs_decay = decay_epoch

    # print(config)

    train_reader = get_reader(config, 'train', './list/clean_train.lst', './list/noise_train.lst')
    valid_reader = get_reader(config, 'valid', './list/clean_valid.lst', './list/noise_valid.lst')
    test_reader = get_reader(config, 'test', './list/clean_test.lst', './list/noise_test.lst')

    solver = Solver(config, train_reader, valid_reader, test_reader)

    # Train and sample the images
    # solver.print_network(solver.unet, config.model_type)
    if config.mode == 'train':
        solver.train()
    elif config.mode == 'test':
        solver.test()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # data parameters
    parser.add_argument('--sample_rate', type=int, default=16000)
    parser.add_argument('--frame_size', type=int, default=320)
    parser.add_argument('--frame_shift', type=int, default=160)
    parser.add_argument('--sent_width', type=int, default=512)
    parser.add_argument('--sent_height', type=int, default=161)
    parser.add_argument('--min_queue_size', type=int, default=64)
    parser.add_argument('--chunk_length', type=int, default=4 * 16000)

    # model hyper-parameters
    parser.add_argument('--image_size', type=int, default=224)
    parser.add_argument('--t', type=int, default=1,
                        help='t for Recurrent step of R2U_Net or R2AttU_Net and Interation step of DARCCN')

    # training hyper-parameters
    parser.add_argument('--img_ch', type=int, default=2)
    parser.add_argument('--output_ch', type=int, default=2)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--num_epochs_decay', type=int, default=70)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--num_workers', type=int, default=16)
    parser.add_argument('--lr', type=float, default=0.0002)
    parser.add_argument('--beta1', type=float, default=0.5)  # momentum1 in Adam
    parser.add_argument('--beta2', type=float, default=0.999)  # momentum2 in Adam
    parser.add_argument('--half_lr', type=int, default=1,
                        help='Whether to decay learning rate to half scale')

    parser.add_argument('--log_step', type=int, default=2)
    parser.add_argument('--val_step', type=int, default=2)

    # misc
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--model_type', type=str, default='TinyDCU_Net_16',
                        help='TinyDCU_Net_16')
    parser.add_argument('--model_path', type=str, default='./models')
    parser.add_argument('--logfile_path', type=str, default='./logs')
    parser.add_argument('--result_path', type=str, default='./result')

    parser.add_argument('--cuda_idx', type=int, default=1)

    config = parser.parse_args()
    main(config)
