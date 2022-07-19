from asyncio import shield
import os
from readline import set_history_length
import sys
sys.path.insert(1, os.path.join(sys.path[0], '../utils'))
import argparse
import time
import logging

import torch
import torch.optim as optim
import torch.utils.data
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter

import config
from utils import mkdir, create_logging, move_data_to_device, SegmentEvaluator
from dataset import ST500_Dataset, Sampler, TestSampler, collate_fn
from model import VocalNet
from loss import loss_func
from wechatmsg import SendWeiXinWork



def train(args):

    # Arugments & parameters
    batch_size = args.batch_size
    learning_rate = args.learning_rate
    reduce_iteration = args.reduce_iteration
    early_stop = args.early_stop
    device = torch.device('cuda') if args.cuda and torch.cuda.is_available() else torch.device('cpu')

    sample_rate = config.sample_rate
    segment_seconds = config.segment_seconds
    hop_seconds = config.hop_seconds
    segment_samples = int(segment_seconds * sample_rate)
    frames_per_second = config.frames_per_second
    classes_num = config.classes_num
    num_workers = 8
    n_epochs = 100

    wechat = SendWeiXinWork()
    writer = SummaryWriter("logs")

    # Output dirs
    mkdir('checkpoints')
    checkpoints_dir = os.path.join('checkpoints', time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime()))
    mkdir(checkpoints_dir)

    # logging setting
    logs_dir = os.path.join(checkpoints_dir, 'logs')
    mkdir(logs_dir)
    create_logging(logs_dir, filemode='w')
    logging.info(args)

    if 'cuda' in str(device):
        logging.info('Using GPU.')
        device = 'cuda'
    else:
        logging.info('Using CPU.')
        device = 'cpu'
    
    
    # Model
    model = VocalNet(frames_per_second=frames_per_second, classes_num=classes_num)
    if 'cuda' in str(device):
        model.to(device)
    checkpoint = torch.load('./checkpoints/2022-07-14-12-24-34/7999_iterations.pth', map_location=device)
    model.load_state_dict(checkpoint['model'], strict=True)

    # Dataset
    st500_dataset = ST500_Dataset(segment_seconds=segment_seconds, 
                                  frames_per_second=frames_per_second, 
                                  sample_rate=sample_rate,
                                  classes_num=classes_num)

    train_sampler = Sampler(data_dir='./train', 
                            segment_seconds=segment_seconds, 
                            hop_seconds=hop_seconds, 
                            sample_rate=sample_rate,
                            batch_size=batch_size,
                            type='train')

    eval_sampler = TestSampler(data_dir='./train', 
                               segment_seconds=segment_seconds, 
                               hop_seconds=hop_seconds, 
                               sample_rate=sample_rate,
                               batch_size=batch_size,
                               type='eval')

    test_sampler = TestSampler(data_dir='./test', 
                               segment_seconds=segment_seconds, 
                               hop_seconds=hop_seconds, 
                               sample_rate=sample_rate,
                               batch_size=batch_size,
                               type='test')

    train_loader = torch.utils.data.DataLoader(dataset=st500_dataset, 
                                               batch_sampler=train_sampler, 
                                               collate_fn=collate_fn, 
                                               num_workers=num_workers, 
                                               pin_memory=True)

    eval_loader = torch.utils.data.DataLoader(dataset=st500_dataset,
                                              batch_sampler=eval_sampler, 
                                              collate_fn=collate_fn, 
                                              num_workers=num_workers, 
                                              pin_memory=True)

    test_loader = torch.utils.data.DataLoader(dataset=st500_dataset,
                                              batch_sampler=test_sampler, 
                                              collate_fn=collate_fn, 
                                              num_workers=num_workers, 
                                              pin_memory=True)

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, 
        betas=(0.9, 0.999), eps=1e-08, weight_decay=0., amsgrad=True)

    
    evaluator = SegmentEvaluator(model, batch_size)


    
    for epoch in range(n_epochs):
        iteration = 0
        train_bgn_time = time.time()
        
        for batch_data_dict in train_loader:

            # Reduce learning rate
            if iteration % reduce_iteration == 0 and iteration > 0:
                for param_group in optimizer.param_groups:
                    param_group['lr'] *= 0.9
            
            # Move data to device
            for key in batch_data_dict.keys():
                batch_data_dict[key] = move_data_to_device(batch_data_dict[key], device)
            
            model.train()

            # ------- normal training -------
            # writer.add_image("frame_roll", batch_data_dict['frame_roll'][0].unsqueeze(0).repeat(3,1,1), iteration, dataformats='CWH')
            # writer.add_image("reg_onset_roll", batch_data_dict['reg_onset_roll'][0].unsqueeze(0).repeat(3,1,1), iteration, dataformats='CWH')
            # writer.add_image("reg_offset_roll", batch_data_dict['reg_offset_roll'][0].unsqueeze(0).repeat(3,1,1), iteration, dataformats='CWH')
            batch_output_dict = model(batch_data_dict['waveform'])
            loss = loss_func(batch_output_dict, batch_data_dict)


            # Backward
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            print('\riteration: %d loss: %f'%(iteration, loss.item()), end="")

            if (iteration+1) % 200 == 0:
                idx = (iteration+1) // 200
                writer.add_image("target_roll", batch_data_dict['frame_roll'][0].unsqueeze(0).repeat(3,1,1), idx, dataformats='CWH')
                writer.add_image("output_roll", batch_output_dict['frame_output'][0].unsqueeze(0).repeat(3,1,1), idx, dataformats='CWH')

            # test current status
            if (iteration+1) % 2000 == 0:
                evaluator.evaluate(eval_loader, wechat, 'eval', iteration, writer, idx)
                evaluator.evaluate(test_loader, wechat, 'test', iteration, writer, idx)
                cost = time.time() - train_bgn_time

            # Save model
            if (iteration+1) % 10000 == 0:
                checkpoint = {
                    'iteration': iteration, 
                    'model': model.state_dict(),
                    'sampler': train_sampler.state_dict()}

                checkpoint_path = os.path.join(
                    checkpoints_dir, '{}_iterations.pth'.format(iteration))
                    
                torch.save(checkpoint, checkpoint_path)
                logging.info('Model saved to {}'.format(checkpoint_path))

            iteration += 1

        cost = time.time() - train_bgn_time
        wechat.send("ronnnhui", "--- epoch %d iter %d: %dh%dm%ds ---"%(epoch, iteration, cost//3600, (cost//60)%60, cost%60))
        break


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Example of parser. ')
    subparsers = parser.add_subparsers(dest='mode')

    parser_train = subparsers.add_parser('train') 
    parser_train.add_argument('--batch_size', type=int, required=True)
    parser_train.add_argument('--learning_rate', type=float, required=True)
    parser_train.add_argument('--reduce_iteration', type=int, required=True)
    parser_train.add_argument('--early_stop', type=int, required=True)
    parser_train.add_argument('--cuda', action='store_true', default=False)

    args = parser.parse_args()
    if args.mode == 'train':
        train(args)
    else:
        raise Exception('Error argument!')

# python3 train.py train --batch_size=2 --learning_rate=5e-4 --reduce_iteration=10000 --early_stop=100000 --cuda

