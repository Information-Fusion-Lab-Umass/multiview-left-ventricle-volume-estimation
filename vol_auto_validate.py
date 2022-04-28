import argparse
import os
from collections import OrderedDict
from glob import glob

import cv2
import pandas as pd
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import yaml
from albumentations.augmentations import transforms
from albumentations.core.composition import Compose, OneOf
from sklearn.model_selection import train_test_split
from torch.optim import lr_scheduler
from tqdm import tqdm

import archs
import losses
from dataset import VolumeDataset
from metrics import rmse_score
from utils import AverageMeter, str2bool

ARCH_NAMES = archs.__all__
LOSS_NAMES = losses.__all__


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--name', default=None,
                        help='model name: (default: arch+timestamp)')
    parser.add_argument('--epochs', default=50, type=int, metavar='N',
                        help='number of total epochs to run')
    #parser.add_argument('-b', '--batch_size', default=16, type=int,
    #                    metavar='N', help='mini-batch size (default: 16)')
    # Forced batch size of one
    
    # model
    parser.add_argument('--arch', '-a', metavar='ARCH', default='VolumeEstimation',
                        choices=ARCH_NAMES,
                        help='model architecture: ' +
                        ' | '.join(ARCH_NAMES) +
                        ' (default: VolumeEstimation)')
    #parser.add_argument('--deep_supervision', default=False, type=str2bool)
    parser.add_argument('--input_channels', default=3, type=int,
                        help='input channels')
    parser.add_argument('--num_classes', default=1, type=int,
                        help='number of classes')
    parser.add_argument('--input_w', default=96, type=int,
                        help='image width')
    parser.add_argument('--input_h', default=96, type=int,
                        help='image height')
    parser.add_argument('--seg_arch', '-s_a', metavar='SEG_ARCH', default='NestedUNet',
                        choices=ARCH_NAMES,
                        help='segmentation model architecture: ' +
                        ' | '.join(ARCH_NAMES) +
                        ' (default: NestedUNet)')
    parser.add_argument('--sax_seg_model', default='short_axis',
                        help='name of the short axis segmentation model to you')
    parser.add_argument('--lax_seg_model', default='long_axis',
                        help='name of the long axis segmentation model to you')
    parser.add_argument('--num_short', default=1, type=int,
                        help='number of short axis slices to use')
    parser.add_argument('--use_long_axis', type=str2bool, nargs='?',
                        const=True, default=False,
                        help="use long axis slices")
    parser.add_argument('--pooling_method', default='average',
                        choices=['average', 'max', 'min'],
                        help='pooling method: ' +
                        ' | '.join(['average', 'max', 'min']) +
                        ' (default: average)')
    
    # loss
    parser.add_argument('--loss', default='RMSELoss',
                        choices=LOSS_NAMES,
                        help='loss: ' +
                        ' | '.join(LOSS_NAMES) +
                        ' (default: RMSELoss)')
    
    # dataset
    parser.add_argument('--dataset', default='formatted_kaggle_data_bowl',
                        help='dataset name')
    parser.add_argument('--img_ext', default='.png',
                        help='image file extension')
    #parser.add_argument('--mask_ext', default='.png',
    #                    help='mask file extension')

    # optimizer
    parser.add_argument('--optimizer', default='SGD',
                        choices=['Adam', 'SGD'],
                        help='loss: ' +
                        ' | '.join(['Adam', 'SGD']) +
                        ' (default: Adam)')
    parser.add_argument('--lr', '--learning_rate', default=1e-3, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float,
                        help='momentum')
    parser.add_argument('--weight_decay', default=1e-4, type=float,
                        help='weight decay')
    parser.add_argument('--nesterov', default=False, type=str2bool,
                        help='nesterov')

    # scheduler
    parser.add_argument('--scheduler', default='CosineAnnealingLR',
                        choices=['CosineAnnealingLR', 'ReduceLROnPlateau', 'MultiStepLR', 'ConstantLR'])
    parser.add_argument('--min_lr', default=1e-5, type=float,
                        help='minimum learning rate')
    parser.add_argument('--factor', default=0.1, type=float)
    parser.add_argument('--patience', default=2, type=int)
    parser.add_argument('--milestones', default='1,2', type=str)
    parser.add_argument('--gamma', default=2/3, type=float)
    parser.add_argument('--early_stopping', default=-1, type=int,
                        metavar='N', help='early stopping (default: -1)')
    
    parser.add_argument('--num_workers', default=4, type=int)

    config = parser.parse_args()

    return config

def train(config, train_loader, model, criterion, optimizer):
    avg_meters = {'loss': AverageMeter(),
                  'edv_rmse': AverageMeter(),
                  'esv_rmse': AverageMeter()}

    model.train()

    pbar = tqdm(total=len(train_loader))
    #print("length of dataloader from inside the function is " + str(len(train_loader)))
    #print(train_loader)
    for input, target, patients in train_loader:
        #input = input.cuda()
        #target = target.cuda()

        #edvs = torch.Tensor()
        #esvs = torch.Tensor()
        #for i in range(batch_size):
        edv_output, esv_output = model(input)
        loss = criterion(edv_output, esv_output, target)
        edv_rmse = rmse_score(edv_output, target[0])
        esv_rmse = rmse_score(esv_output, target[1])

        # compute gradient and do optimizing step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        #avg_meters['loss'].update(loss.item(), input.size(0))
        avg_meters['loss'].update(loss.item(), 1)
        #avg_meters['edv_rmse'].update(edv_rmse, input.size(0))
        avg_meters['edv_rmse'].update(edv_rmse, 1)
        #avg_meters['esv_rmse'].update(esv_rmse, input.size(0))
        avg_meters['esv_rmse'].update(esv_rmse, 1)

        postfix = OrderedDict([
            ('loss', avg_meters['loss'].avg),
            ('edv_rmse', avg_meters['edv_rmse'].avg),
            ('esv_rmse', avg_meters['esv_rmse'].avg),
        ])
        pbar.set_postfix(postfix)
        pbar.update(1)
    pbar.close()

    return OrderedDict([('loss', avg_meters['loss'].avg),
                        ('edv_rmse', avg_meters['edv_rmse'].avg),
                        ('esv_rmse', avg_meters['esv_rmse'].avg)])


def validate(config, val_loader, model, criterion):
    avg_meters = {'loss': AverageMeter(),
                  'edv_rmse': AverageMeter(),
                  'esv_rmse': AverageMeter()}

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        pbar = tqdm(total=len(val_loader))
        for input, target, patients in val_loader:
            #input = input.cuda()
            #target = target.cuda()

            # compute output
            edv_output, esv_output = model(input)
            loss = criterion(edv_output, esv_output, target)
            edv_rmse = rmse_score(edv_output, target[0])
            esv_rmse = rmse_score(esv_output, target[1])

            #avg_meters['loss'].update(loss.item(), input.size(0))
            avg_meters['loss'].update(loss.item(), 1)
            #avg_meters['edv_rmse'].update(edv_rmse, input.size(0))
            avg_meters['edv_rmse'].update(edv_rmse, 1)
            #avg_meters['esv_rmse'].update(esv_rmse, input.size(0))
            avg_meters['esv_rmse'].update(esv_rmse, 1)

            postfix = OrderedDict([
                ('loss', avg_meters['loss'].avg),
                ('edv_rmse', avg_meters['edv_rmse'].avg),
                ('esv_rmse', avg_meters['esv_rmse'].avg),
            ])
            pbar.set_postfix(postfix)
            pbar.update(1)
        pbar.close()

    return OrderedDict([('loss', avg_meters['loss'].avg),
                        ('edv_rmse', avg_meters['edv_rmse'].avg),
                        ('esv_rmse', avg_meters['esv_rmse'].avg)])

def main_func(modelName, fileName, shortSegModel, longSegModel):
    config = vars(parse_args())
    config['name'] = modelName
    fw = open('batch_results_train/'+ fileName, 'w')
    print('config of dataset is ' + str(config['dataset']))
    fw.write('config of dataset is ' + str(config['dataset']) + '\n')    
    if config['name'] is None:
        config['name'] = '%s_%s_woDS' % (config['dataset'], config['arch'])
    os.makedirs('models/%s' % config['name'], exist_ok=True)

    print('-' * 20)
    fw.write('-' * 20 + '\n')
    for key in config:
        print('%s: %s' % (key, config[key]))
        fw.write('%s: %s' % (key, config[key]) + '\n')
    print('-' * 20)
    fw.write('-' * 20 + '\n')
    #TODO print parameters manually i think, all imports to function

    with open('models/%s/config.yml' % config['name'], 'w') as f:
        yaml.dump(config, f)

    # define loss function (criterion)
    criterion = losses.__dict__[config['loss']]().cuda()

    cudnn.benchmark = True

    # create model
    print("=> creating model %s" % config['arch'])
    fw.write("=> creating model %s" % config['arch'] + '\n')   
    model = archs.__dict__[config['arch']](shortSegModel,
                                           longSegModel,
                                           config['use_long_axis'],
                                           config['num_short'],
                                           config['pooling_method'])

    model = model.cuda()

    params = filter(lambda p: p.requires_grad, model.parameters())
    if config['optimizer'] == 'Adam':
        optimizer = optim.Adam(
            params, lr=config['lr'], weight_decay=config['weight_decay'])
    elif config['optimizer'] == 'SGD':
        optimizer = optim.SGD(params, lr=config['lr'], momentum=config['momentum'],
                              nesterov=config['nesterov'], weight_decay=config['weight_decay'])
    else:
        raise NotImplementedError

    if config['scheduler'] == 'CosineAnnealingLR':
        scheduler = lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config['epochs'], eta_min=config['min_lr'])
    elif config['scheduler'] == 'ReduceLROnPlateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, factor=config['factor'], patience=config['patience'],
                                                   verbose=1, min_lr=config['min_lr'])
    elif config['scheduler'] == 'MultiStepLR':
        scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[int(e) for e in config['milestones'].split(',')], gamma=config['gamma'])
    elif config['scheduler'] == 'ConstantLR':
        scheduler = None
    else:
        raise NotImplementedError

    # Data loading code
    # img_ids = glob(os.path.join('inputs', config['dataset'], 'images', '*' + config['img_ext']))
    # img_ids = [os.path.splitext(os.path.basename(p))[0] for p in img_ids]

    # #train_img_ids, val_img_ids = train_test_split(img_ids, test_size=0.2, random_state=41
    # val_idx = [val_set]
    # #train_idx = [2, 3, 6, 7]
    # val_img_ids = []
    # train_img_ids = []

    # for image in img_ids:
    #     im_begin = image.split('.')[0]
    #     if int(im_begin[-1]) in val_idx:
    #         val_img_ids.append(image)
    #     elif int(im_begin[-1]) in train_idx:
    #         train_img_ids.append(image)
    # #print("train img ids size is " + str(len(train_img_ids)))
    train_data_file = os.path.join('inputs', config['dataset'], 'train', 'train.csv')
    train_data = pd.read_csv(train_data_file).to_numpy().tolist()

    val_data_file = os.path.join('inputs', config['dataset'], 'validate', 'validate.csv')
    val_data = pd.read_csv(val_data_file).to_numpy().tolist()

    transform = Compose([
        transforms.Resize(config['input_h'], config['input_w']),
        transforms.Normalize(),
    ])
    
    train_dataset = VolumeDataset(
        patient_data=train_data,
        patient_dir=os.path.join('inputs', config['dataset'], 'train'),
        img_ext=config['img_ext'],
        numShortSlices=config['num_short'],
        yesLongAxis=config['use_long_axis'],
        transform=transform)
    val_dataset = VolumeDataset(
        patient_data=val_data,
        patient_dir=os.path.join('inputs', config['dataset'], 'validate'),
        img_ext=config['img_ext'],
        numShortSlices=config['num_short'],
        yesLongAxis=config['use_long_axis'],
        transform=transform)

    #print("length of train dataset is " + str(len(train_dataset)))
    #print("length of val dataset is " + str(len(val_dataset)))

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        #batch_size=config['batch_size'],
        batch_size=None,
        shuffle=True,
        num_workers=config['num_workers'],
        #drop_last=True)
        drop_last=None)
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        #batch_size=config['batch_size'],
        batch_size=None,
        shuffle=False,
        num_workers=config['num_workers'],
        #drop_last=False)
        drop_last=None)

    log = OrderedDict([
        ('epoch', []),
        ('lr', []),
        ('loss', []),
        ('edv_rmse', []),
        ('esv_rmse', []),
        ('val_loss', []),
        ('val_edv_rmse', []),
        ('val_esv_rmse', []),
    ])

    #best_iou = 0
    trigger = 0
    best_loss = np.inf
    for epoch in range(config['epochs']):
        print('Epoch [%d/%d]' % (epoch, config['epochs']))
        fw.write('Epoch [%d/%d]' % (epoch, config['epochs']) + '\n')    

        # train for one epoch
        train_log = train(config, train_loader, model, criterion, optimizer)
        # evaluate on validation set
        val_log = validate(config, val_loader, model, criterion)

        if config['scheduler'] == 'CosineAnnealingLR':
            scheduler.step()
        elif config['scheduler'] == 'ReduceLROnPlateau':
            scheduler.step(val_log['loss'])

        print('loss %.4f - edv_rmse %.4f - esv_rmse %.4f - val_loss %.4f - val_edv_rmse %.4f - val_esv_rmse %.4f'
              % (train_log['loss'], train_log['edv_rmse'], train_log['esv_rmse'], val_log['loss'], val_log['edv_rmse'], val_log['esv_rmse']))
        fw.write('loss %.4f - edv_rmse %.4f - esv_rmse %.4f - val_loss %.4f - val_edv_rmse %.4f - val_esv_rmse %.4f'
              % (train_log['loss'], train_log['edv_rmse'], train_log['esv_rmse'], val_log['loss'], val_log['edv_rmse'], val_log['esv_rmse']) + '\n')

        log['epoch'].append(epoch)
        log['lr'].append(config['lr'])
        log['loss'].append(train_log['loss'])
        log['edv_rmse'].append(train_log['edv_rmse'])
        log['esv_rmse'].append(train_log['esv_rmse'])
        log['val_loss'].append(val_log['loss'])
        log['val_edv_rmse'].append(val_log['edv_rmse'])
        log['val_esv_rmse'].append(val_log['esv_rmse'])

        pd.DataFrame(log).to_csv('models/%s/log.csv' %
                                 config['name'], index=False)

        trigger += 1
        if val_log['loss'] < best_loss:
            torch.save(model.state_dict(), 'models/%s/model.pth' %
                       config['name'])
            best_loss = val_log['loss']
            print("=> saved best model")
            fw.write("=> saved best model" + '\n')
            trigger = 0

        # early stopping
        if config['early_stopping'] >= 0 and trigger >= config['early_stopping']:
            print("=> early stopping")
            fw.write("=> early stopping" + '\n')
            break

        torch.cuda.empty_cache()

def perform_validation(modelName, fileName, shortSegModel, longSegModel):
    #args = parse_args()

    fw = open('batch_results_test/' + fileName, 'w') 
    #with open('models/%s/config.yml' % args.name, 'r') as f:
    with open('models/%s/config.yml' % modelName, 'r') as f:   
        config = yaml.load(f, Loader=yaml.FullLoader)
 
    #config['dataset'] = 'ax_crop_val_' + str(testNum) + '_' + str(testNum + 1)

    print('-'*20)
    fw.write('-'*20 + '\n')
    for key in config.keys():
        print('%s: %s' % (key, str(config[key])))
        fw.write('%s: %s' % (key, str(config[key])) + '\n')
    print('-'*20)
    fw.write('-'*20 + '\n')

    cudnn.benchmark = True

    # create model
    print("=> creating model %s" % config['arch'])
    fw.write("=> creating model %s" % config['arch'] + '\n')
    model = archs.__dict__[config['arch']](shortSegModel,
                                           longSegModel,
                                           config['use_long_axis'],
                                           config['num_short'],
                                           config['pooling_method'])

    model = model.cuda()

    # Data loading code
    # img_ids = glob(os.path.join('inputs', config['dataset'], 'images', '*' + config['img_ext']))
    # img_ids = [os.path.splitext(os.path.basename(p))[0] for p in img_ids]

    # #_, val_img_ids = train_test_split(img_ids, test_size=0.99, random_state=41)
    # val_idx = [testNum, testNum + 1]
    # val_img_ids = []
    # for img in img_ids:
    #     im_begin = img.split('.')[0]
    #     if int(im_begin[-1]) in val_idx:
    #         val_img_ids.append(img)
    test_data_file = os.path.join('inputs', config['dataset'], 'test', 'test.csv')
    test_data = pd.read_csv(test_data_file).to_numpy().tolist()

    model.load_state_dict(torch.load('models/%s/model.pth' %
                                     config['name']))
    model.eval()

    transform = Compose([
        transforms.Resize(config['input_h'], config['input_w']),
        transforms.Normalize(),
    ])

    test_dataset = VolumeDataset(
        patient_data=test_data,
        patient_dir=os.path.join('inputs', config['dataset'], 'test'),
        img_ext=config['img_ext'],
        numShortSlices=config['num_short'],
        yesLongAxis=config['use_long_axis'],
        transform=transform)
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        #batch_size=config['batch_size'],
        batch_size=None,
        shuffle=False,
        num_workers=config['num_workers'],
        #drop_last=False)
        drop_last=None)


    edv_rmse_avg_meter = AverageMeter()
    esv_rmse_avg_meter = AverageMeter()

    for c in range(config['num_classes']):
        os.makedirs(os.path.join('outputs', config['name'], str(c)), exist_ok=True)
    with torch.no_grad():
        for input, target, patients in tqdm(test_loader, total=len(test_loader)):
            #input = input.cuda()
            #target = target.cuda()

            # compute output
            edv_output, esv_output = model(input)
            edv_rmse = rmse_score(edv_output, target[0])
            esv_rmse = rmse_score(esv_output, target[1])

            #edv_rmse_avg_meter.update(edv_rmse, input.size(0))
            edv_rmse_avg_meter.update(edv_rmse, 1)
            #esv_rmse_avg_meter.update(esv_rmse, input.size(0))
            esv_rmse_avg_meter.update(esv_rmse, 1)

            #output = torch.sigmoid(output).cpu().numpy()

            #for i in range(len(output)):
            #    for c in range(config['num_classes']):
            #        cv2.imwrite(os.path.join('outputs', config['name'], str(c), meta['img_id'][i] + '.jpg'),
            #                    (output[i, c] * 255).astype('uint8'))

    print('EDV RMSE: %.4f' % edv_rmse_avg_meter.avg)
    fw.write('EDV RMSE: %.4f' % edv_rmse_avg_meter.avg)
    print('ESV RMSE: %.4f' % esv_rmse_avg_meter.avg)
    fw.write('ESV RMSE: %.4f' % esv_rmse_avg_meter.avg)

    torch.cuda.empty_cache()

def main():
    '''params = {}
    params['dataset'] = 'sa_dataset'
    params['loss'] = 'BCEDiceLoss'
    params['arch'] = 'NestedUNet'
    params['num_classes'] = 2
    params['input_channels'] = 3
    params['deep_supervision'] = False
    params['optimizer'] = 'SGD'
    params['lr'] = 1e-3
    params['weight_decay'] = 1e-4
    params['momentum'] = 0.9
    params['nesterov'] = False
    params['scheduler'] = 'CosineAnnealingLR'
    params['img_ext'] = 'png'
    params['mask_ext'] = 'png'
    params['input_h'] = 96   ## can be set to a command line argument in the future
    params['input_w'] = 96   ## can be set to a command line argument in the future
    params['batch_size'] = 16
    params['num_workers'] = 4
    params['epochs'] = 100
    params['early_stopping'] = -1
    params['min_lr'] = 1e-5
    # extras
    params['factor'] = 0.1
    params['patience'] = 2
    params['milestones'] = '1,2'
    params['gamma'] = 0.66666
    '''
    params = vars(parse_args())

    shortSegModel = archs.__dict__[params['seg_arch']](params['num_classes'], params['input_channels'])
    shortSegModel.cuda()
    shortSegModel.load_state_dict(torch.load('models/%s/model.pth' % params['sax_seg_model']))
    shortSegModel.eval()

    longSegModel = archs.__dict__[params['seg_arch']](params['num_classes'], params['input_channels'])
    longSegModel.cuda()
    longSegModel.load_state_dict(torch.load('models/%s/model.pth' % params['lax_seg_model']))
    longSegModel.eval()

    modelName = params['name']
    trainFileName = params['name'] + '_trainingResult'
    testFileName = params['name'] + '_testResult'
    main_func(modelName, trainFileName, shortSegModel, longSegModel)
    perform_validation(modelName, testFileName, shortSegModel, longSegModel)

if __name__ == '__main__':
    main()
