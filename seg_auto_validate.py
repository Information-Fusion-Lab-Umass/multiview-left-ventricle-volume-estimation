import argparse
import os
from collections import OrderedDict
from glob import glob

import cv2
import pandas as pd
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
from dataset import SegmentationDataset
from metrics import iou_score
from metrics import dice_coef
from utils import AverageMeter, str2bool

ARCH_NAMES = archs.__all__
LOSS_NAMES = losses.__all__


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--name', default=None,
                        help='model name: (default: arch+timestamp)')
    parser.add_argument('--epochs', default=100, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-b', '--batch_size', default=16, type=int,
                        metavar='N', help='mini-batch size (default: 16)')
    
    # model
    parser.add_argument('--arch', '-a', metavar='ARCH', default='NestedUNet',
                        choices=ARCH_NAMES,
                        help='model architecture: ' +
                        ' | '.join(ARCH_NAMES) +
                        ' (default: NestedUNet)')
    parser.add_argument('--deep_supervision', default=False, type=str2bool)
    parser.add_argument('--input_channels', default=3, type=int,
                        help='input channels')
    parser.add_argument('--num_classes', default=1, type=int,
                        help='number of classes')
    parser.add_argument('--input_w', default=96, type=int,
                        help='image width')
    parser.add_argument('--input_h', default=96, type=int,
                        help='image height')
    
    # loss
    parser.add_argument('--loss', default='BCEDiceLoss',
                        choices=LOSS_NAMES,
                        help='loss: ' +
                        ' | '.join(LOSS_NAMES) +
                        ' (default: BCEDiceLoss)')
    
    # dataset
    parser.add_argument('--dataset', default='dsb2018_96',
                        help='dataset name')
    parser.add_argument('--img_ext', default='.png',
                        help='image file extension')
    parser.add_argument('--mask_ext', default='.png',
                        help='mask file extension')

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
                  'iou': AverageMeter(),
                  'dice': AverageMeter()}

    model.train()

    pbar = tqdm(total=len(train_loader))
    for input, target, _ in train_loader:
        input = input.cuda()
        target = target.cuda()

        # compute output
        output = model(input)
        loss = criterion(output, target)
        iou = iou_score(output, target)
        dice = dice_coef(output, target)

        # compute gradient and do optimizing step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        avg_meters['loss'].update(loss.item(), input.size(0))
        avg_meters['iou'].update(iou, input.size(0))
        avg_meters['dice'].update(dice, input.size(0))

        postfix = OrderedDict([
            ('loss', avg_meters['loss'].avg),
            ('iou', avg_meters['iou'].avg),
            ('dice', avg_meters['dice'].avg),
        ])
        pbar.set_postfix(postfix)
        pbar.update(1)
    pbar.close()

    return OrderedDict([('loss', avg_meters['loss'].avg),
                        ('iou', avg_meters['iou'].avg),
                        ('dice', avg_meters['dice'].avg)])


def validate(config, val_loader, model, criterion):
    avg_meters = {'loss': AverageMeter(),
                  'iou': AverageMeter(),
                  'dice': AverageMeter(),}

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        pbar = tqdm(total=len(val_loader))
        for input, target, _ in val_loader:
            input = input.cuda()
            target = target.cuda()

            # compute output
            output = model(input)
            loss = criterion(output, target)
            iou = iou_score(output, target)
            dice = dice_coef(output, target)

            avg_meters['loss'].update(loss.item(), input.size(0))
            avg_meters['iou'].update(iou, input.size(0))
            avg_meters['dice'].update(dice, input.size(0))

            postfix = OrderedDict([
                ('loss', avg_meters['loss'].avg),
                ('iou', avg_meters['iou'].avg),
                ('dice', avg_meters['dice'].avg)
            ])
            pbar.set_postfix(postfix)
            pbar.update(1)
        pbar.close()

    return OrderedDict([('loss', avg_meters['loss'].avg),
                        ('iou', avg_meters['iou'].avg),
                        ('dice', avg_meters['dice'].avg)])

def main_func(train_ids, val_ids, modelName, fileName):
    config = vars(parse_args())
    config['name'] = modelName
    fw = open('batch_results_train/'+ fileName, 'w')
    print('config of dataset is ' + str(config['dataset']))
    fw.write('config of dataset is ' + str(config['dataset']) + '\n')    
    if config['name'] is None:
        if config['deep_supervision']:
            config['name'] = '%s_%s_wDS' % (config['dataset'], config['arch'])
        else:
            config['name'] = '%s_%s_woDS' % (config['dataset'], config['arch'])
    os.makedirs('models/%s' % config['name'], exist_ok=True)

    print('-' * 20)
    fw.write('-' * 20 + '\n')
    for key in config:
        print('%s: %s' % (key, config[key]))
        fw.write('%s: %s' % (key, config[key]) + '\n')
    print('-' * 20)
    fw.write('-' * 20 + '\n')

    with open('models/%s/config.yml' % config['name'], 'w') as f:
        yaml.dump(config, f)

    # define loss function (criterion)
    if config['loss'] == 'BCEWithLogitsLoss':
        criterion = nn.BCEWithLogitsLoss().cuda()
    else:
        criterion = losses.__dict__[config['loss']]().cuda()

    cudnn.benchmark = True

    # create model
    print("=> creating model %s" % config['arch'])
    fw.write("=> creating model %s" % config['arch'] + '\n')   
    model = archs.__dict__[config['arch']](config['num_classes'],
                                           config['input_channels'],
                                           config['deep_supervision'])

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
    img_ids = glob(os.path.join('inputs', config['dataset'], 'images', '*' + config['img_ext']))
    img_ids = [os.path.splitext(os.path.basename(p))[0] for p in img_ids]

    train_ids = []
    for train_group in train_ids:
        with open(train_group, 'r') as file:
            train_ids.append([line.rstrip() for line in file])

    val_ids = []
    with open(val_ids, 'r') as file:
        val_ids = [line.rstrip() for line in file]
    
    val_img_ids = []
    train_img_ids = []

    for image in img_ids:
        im_begin = image.split('.')[0]
        if int(im_begin[-1]) in val_ids:
            val_img_ids.append(image)
        elif int(im_begin[-1]) in train_ids:
            train_img_ids.append(image)

    # train_transform = Compose([
    #     transforms.Resize(config['input_h'], config['input_w']),
    #     transforms.Normalize(),
    # ])
    
    # val_transform = Compose([
    #     transforms.Resize(config['input_h'], config['input_w']),
    #     transforms.Normalize(),
    # ])
    transform = Compose([
        transforms.Resize(config['input_h'], config['input_w']),
        transforms.Normalize(),
    ])
    
    train_dataset = SegmentationDataset(
        img_ids=train_img_ids,
        img_dir=os.path.join('inputs', config['dataset'], 'images'),
        mask_dir=os.path.join('inputs', config['dataset'], 'masks'),
        img_ext=config['img_ext'],
        mask_ext=config['mask_ext'],
        num_classes=config['num_classes'],
        transform=transform)
    val_dataset = SegmentationDataset(
        img_ids=val_img_ids,
        img_dir=os.path.join('inputs', config['dataset'], 'images'),
        mask_dir=os.path.join('inputs', config['dataset'], 'masks'),
        img_ext=config['img_ext'],
        mask_ext=config['mask_ext'],
        num_classes=config['num_classes'],
        transform=transform)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers'],
        drop_last=True)
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        drop_last=False)

    log = OrderedDict([
        ('epoch', []),
        ('lr', []),
        ('loss', []),
        ('iou', []),
        ('dice', []),
        ('val_loss', []),
        ('val_iou', []),
        ('val_dice', []), 
    ])

    #best_iou = 0
    trigger = 0
    best_dice = 0
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

        print('loss %.4f - iou %.4f - dice %.4f - val_loss %.4f - val_iou %.4f val_dice %.4f'
              % (train_log['loss'], train_log['iou'], train_log['dice'], val_log['loss'], val_log['iou'], val_log['dice']))
        fw.write('loss %.4f - iou %.4f - dice %.4f - val_loss %.4f - val_iou %.4f val_dice %.4f'
              % (train_log['loss'], train_log['iou'], train_log['dice'], val_log['loss'], val_log['iou'], val_log['dice']) + '\n')

        log['epoch'].append(epoch)
        log['lr'].append(config['lr'])
        log['loss'].append(train_log['loss'])
        log['iou'].append(train_log['iou'])
        log['dice'].append(train_log['dice'])
        log['val_loss'].append(val_log['loss'])
        log['val_iou'].append(val_log['iou'])
        log['val_dice'].append(val_log['dice'])

        pd.DataFrame(log).to_csv('models/%s/log.csv' %
                                 config['name'], index=False)

        trigger += 1

        if val_log['dice'] > best_dice:
            torch.save(model.state_dict(), 'models/%s/model.pth' %
                       config['name'])
            best_dice = val_log['dice']
            print("=> saved best model")
            fw.write("=> saved best model" + '\n')
            trigger = 0

        # early stopping
        if config['early_stopping'] >= 0 and trigger >= config['early_stopping']:
            print("=> early stopping")
            fw.write("=> early stopping" + '\n')
            break

        torch.cuda.empty_cache()


def perform_validation(modelName, test_ids, fileName):
    #args = parse_args()

    fw = open('batch_results_val/' + fileName, 'w') 
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
    model = archs.__dict__[config['arch']](config['num_classes'],
                                           config['input_channels'],
                                           config['deep_supervision'])

    model = model.cuda()

    # Data loading code
    img_ids = glob(os.path.join('inputs', config['dataset'], 'images', '*' + config['img_ext']))
    img_ids = [os.path.splitext(os.path.basename(p))[0] for p in img_ids]

    #_, val_img_ids = train_test_split(img_ids, test_size=0.99, random_state=41)
    test_idx = []
    with open(test_ids, 'r') as file:
        test_idx = [line.rstrip() for line in file]

    test_img_ids = []
    for img in img_ids:
        im_begin = img.split('.')[0]
        if int(im_begin[-1]) in test_idx:
            test_img_ids.append(img)

    model.load_state_dict(torch.load('models/%s/model.pth' %
                                     config['name']))
    model.eval()

    val_transform = Compose([
        transforms.Resize(config['input_h'], config['input_w']),
        transforms.Normalize(),
    ])

    val_dataset = SegmentationDataset(
        img_ids=test_img_ids,
        img_dir=os.path.join('inputs', config['dataset'], 'images'),
        mask_dir=os.path.join('inputs', config['dataset'], 'masks'),
        img_ext=config['img_ext'],
        mask_ext=config['mask_ext'],
        num_classes=config['num_classes'],
        transform=val_transform)
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        drop_last=False)


    avg_meter = AverageMeter()
    dice_avg_meter = AverageMeter()

    for c in range(config['num_classes']):
        os.makedirs(os.path.join('outputs', config['name'], str(c)), exist_ok=True)
    with torch.no_grad():
        for input, target, meta in tqdm(val_loader, total=len(val_loader)):
            input = input.cuda()
            target = target.cuda()

            # compute output
            if config['deep_supervision']:
                output = model(input)[-1]
            else:
                output = model(input)

            iou = iou_score(output, target)
            avg_meter.update(iou, input.size(0))

            dice = dice_coef(output, target)
            dice_avg_meter.update(dice, input.size(0))

            output = torch.sigmoid(output).cpu().numpy()

            for i in range(len(output)):
                for c in range(config['num_classes']):
                    cv2.imwrite(os.path.join('outputs', config['name'], str(c), meta['img_id'][i] + '.jpg'),
                                (output[i, c] * 255).astype('uint8'))

    print('IoU: %.4f' % avg_meter.avg)
    fw.write('IoU: %.4f' % avg_meter.avg)
    print('Dice: %.4f' % dice_avg_meter.avg)
    fw.write('Dice: %.4f' % dice_avg_meter.avg)

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
    #params = vars(parse_args())
    id_groups = ['ids_group1', 'ids_group2', 'ids_group3', 'ids_group4', 'ids_group5']
    for test_ids in id_groups:
        training_groups = id_groups[id_groups != test_ids]
        for val_ids in training_groups:
            train_ids = training_groups[training_groups != val_ids]

            modelName = 'shortAxis_val_' + val_ids + '_test_' + test_ids
            trainFileName = 'shortAxis_val_' + val_ids + '_test_' + test_ids + '_trainingResult'
            testFileName = 'shortAxis_val_' + val_ids + '_test_' + test_ids + '_testResult'
            main_func(train_ids, val_ids, modelName, trainFileName)
            perform_validation(modelName, test_ids, testFileName)

if __name__ == '__main__':
    main()
