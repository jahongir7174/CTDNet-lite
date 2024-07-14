import copy
import csv
import os
import warnings
from argparse import ArgumentParser

import cv2
import numpy
import torch
import tqdm
from torch.utils import data

from nets import nn
from utils import util
from utils.dataset import Dataset

warnings.filterwarnings("ignore")

data_dir = '../Dataset/SOD'


def lr(args):
    return 1E-2 * args.batch_size * args.world_size / 64


def train(args):
    # Model
    model = nn.CTDNet()
    model = util.load_weight(model, ckpt='./weights/imagenet.pt')
    model.cuda()

    # Optimizer
    optimizer = torch.optim.SGD(util.set_params(model, lr(args)), lr(args),
                                momentum=0.9, weight_decay=0, nesterov=True)

    # EMA
    ema = util.EMA(model) if args.local_rank == 0 else None

    filenames = []
    with open(f'{data_dir}/DUTS-TR.txt') as f:
        for filename in f.readlines():
            filename = filename.rstrip().split()[0]
            filenames.append(f'{data_dir}/' + filename)

    sampler = None
    dataset = Dataset(filenames, args.input_size, train=True)

    if args.distributed:
        sampler = data.distributed.DistributedSampler(dataset)

    loader = data.DataLoader(dataset, args.batch_size, sampler is None, sampler,
                             num_workers=8, pin_memory=True, collate_fn=Dataset.collate_fn)

    # Scheduler
    scheduler = util.CosineLR(args, optimizer)

    if args.distributed:
        # DDP mode
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = torch.nn.parallel.DistributedDataParallel(module=model,
                                                          device_ids=[args.local_rank],
                                                          output_device=args.local_rank)

    best = 0
    criterion = util.ComputeLoss()
    amp_scale = torch.cuda.amp.GradScaler()
    with open('weights/step.csv', 'w') as log:
        if args.local_rank == 0:
            logger = csv.DictWriter(log, fieldnames=['epoch', 'loss', 'MAE', 'F-beta'])
            logger.writeheader()
        for epoch in range(args.epochs):
            model.train()
            if args.distributed:
                sampler.set_epoch(epoch)

            p_bar = loader

            if args.local_rank == 0:
                print(('\n' + '%10s' * 3) % ('epoch', 'memory', 'loss'))
            if args.local_rank == 0:
                p_bar = tqdm.tqdm(p_bar, total=len(loader))  # progress bar

            optimizer.zero_grad()
            avg_loss = util.AverageMeter()
            for samples, targets in p_bar:

                samples = samples.cuda()
                targets = targets.cuda()

                samples = samples.float()
                targets = targets.float()

                # Forward
                with torch.cuda.amp.autocast():
                    outputs = model(samples)
                    loss = criterion(outputs, targets)

                avg_loss.update(loss.item(), samples.size(0))

                # Backward
                amp_scale.scale(loss).backward()

                # Optimize
                amp_scale.step(optimizer)  # optimizer.step
                amp_scale.update()
                optimizer.zero_grad()
                if ema:
                    ema.update(model)

                # Log
                if args.local_rank == 0:
                    memory = f'{torch.cuda.memory_reserved() / 1E9:.4g}G'  # (GB)
                    s = ('%10s' * 2 + '%10.3g') % (f'{epoch + 1}/{args.epochs}', memory, avg_loss.avg)
                    p_bar.set_description(s)

            scheduler.step(epoch, optimizer)

            if args.local_rank == 0:
                last = test(args, ema.ema)

                logger.writerow({'epoch': str(epoch + 1).zfill(3),
                                 'loss': str(f'{avg_loss.avg:.3f}'),
                                 'MAE': str(f'{last[0]:.3f}'),
                                 'F-beta': str(f'{last[1]:.3f}')})
                log.flush()

                # Update best F-beta
                if last[1] > best:
                    best = last[1]

                # Save model
                save = {'epoch': epoch + 1,
                        'model': copy.deepcopy(ema.ema)}

                # Save last, best and delete
                torch.save(save, f='./weights/last.pt')
                if best == last[1]:
                    torch.save(save, f='./weights/best.pt')
                del save

    if args.local_rank == 0:
        util.strip_optimizer('./weights/best.pt')  # strip optimizers
        util.strip_optimizer('./weights/last.pt')  # strip optimizers

    torch.cuda.empty_cache()


@torch.no_grad()
def test(args, model=None):
    filenames = []
    with open(f'{data_dir}/DUTS-TE.txt') as f:
        for filename in f.readlines():
            filename = filename.rstrip().split()[0]
            filenames.append(f'{data_dir}/' + filename)

    dataset = Dataset(filenames, args.input_size, train=False)
    loader = data.DataLoader(dataset, batch_size=args.batch_size, num_workers=4)

    if model is None:
        model = torch.load('./weights/best.pt')
        model = model['model'].float().fuse()

    model.cuda()
    model.half()
    model.eval()

    num = 50
    p_bar = tqdm.tqdm(loader, desc=('%10s' * 3) % ('', 'F-beta', 'MAE'))

    avg_mae = 0
    avg_recall = torch.zeros(num).cuda()
    avg_precision = torch.zeros(num).cuda()
    for samples, targets in p_bar:
        samples = samples.cuda()
        targets = targets.cuda()
        samples = samples.half()
        targets = targets.float()
        # Inference
        outputs = model(samples)
        outputs = outputs.sigmoid()
        outputs = outputs.squeeze(1)
        # Metrics
        precision, recall, mae = util.compute_metric(outputs, targets, num)
        avg_precision += precision
        avg_recall += recall
        avg_mae += mae
    # Compute metrics
    avg_precision = avg_precision / len(dataset)
    avg_recall = avg_recall / len(dataset)

    f_beta = (1 + 0.3) * avg_precision * avg_recall / (0.3 * avg_precision + avg_recall + numpy.finfo(float).eps)
    avg_mae = avg_mae / len(dataset)
    if isinstance(avg_mae, torch.Tensor):
        avg_mae = avg_mae.item()
    if isinstance(f_beta, torch.Tensor):
        f_beta = f_beta.max().item()
    # Print results
    print(('%10s' + '%10.3g' * 2) % ('', f_beta, avg_mae))
    model.float()  # for training
    torch.cuda.empty_cache()
    return avg_mae, f_beta


@torch.no_grad()
def demo(args):
    filenames = []
    with open(f'{data_dir}/DUTS-TE.txt') as f:
        for filename in f.readlines():
            filename = filename.rstrip().split()[0]
            filenames.append(f'{data_dir}/' + filename)

    model = torch.load('./weights/best.pt')
    model = model['model'].float().fuse()

    model.cuda()
    model.half()
    model.eval()

    mean = numpy.array([0.406, 0.456, 0.485])
    mean = mean.reshape((1, 1, 3)).astype('float32')

    std = numpy.array([0.225, 0.224, 0.229])
    std = std.reshape((1, 1, 3)).astype('float32')

    for filename in tqdm.tqdm(filenames):
        image = cv2.imread(filename)
        # resize and normalize the image
        x = cv2.resize(image, dsize=(args.input_size, args.input_size))
        x = x.astype('float32') / 255.
        x = (x - mean) / std

        cv2.cvtColor(x, cv2.COLOR_BGR2RGB, x)
        x = x.transpose((2, 0, 1))
        x = torch.from_numpy(x)
        x = x.unsqueeze(0)
        x = x.cuda()
        x = x.half()

        # Inference
        output = model(x)
        output = output.sigmoid()
        output = torch.nn.functional.interpolate(output,
                                                 image.shape[:2],
                                                 mode='bilinear', align_corners=False)
        output = (output * 255).data.cpu().numpy()[0, 0].astype('uint8')
        cv2.imwrite('./results/' + os.path.basename(filename), output)


def profile(args):
    import thop
    shape = (1, 3, args.input_size, args.input_size)
    model = nn.CTDNet()

    model.eval()
    model(torch.zeros(shape))

    x = torch.empty(shape)
    flops, num_params = thop.profile(model, inputs=[x], verbose=False)
    flops, num_params = thop.clever_format(nums=[2 * flops, num_params], format="%.3f")

    if args.local_rank == 0:
        print(f'Number of parameters: {num_params}')
        print(f'Number of FLOPs: {flops}')
    if args.benchmark:
        # Latency
        model = nn.CTDNet().fuse()
        model.eval()

        x = torch.zeros(shape)
        for i in range(10):
            model(x)
        total = 0
        import time
        for i in range(1_000):
            start = time.perf_counter()
            with torch.no_grad():
                model(x)
            total += time.perf_counter() - start

        print(f"Latency: {total / 1_000 * 1_000:.3f} ms")


def main():
    parser = ArgumentParser()
    parser.add_argument('--input-size', default=256, type=int)
    parser.add_argument('--batch-size', default=32, type=int)
    parser.add_argument('--local-rank', default=0, type=int)
    parser.add_argument('--local_rank', default=0, type=int)
    parser.add_argument('--benchmark', action='store_true')
    parser.add_argument('--epochs', default=150, type=int)
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--demo', action='store_true')

    args = parser.parse_args()

    args.local_rank = int(os.getenv('LOCAL_RANK', 0))
    args.world_size = int(os.getenv('WORLD_SIZE', 1))
    args.distributed = int(os.getenv('WORLD_SIZE', 1)) > 1

    if args.distributed:
        torch.cuda.set_device(device=args.local_rank)
        torch.distributed.init_process_group(backend='nccl', init_method='env://')

    if args.local_rank == 0:
        if not os.path.exists('weights'):
            os.makedirs('weights')

    util.setup_seed()
    util.setup_multi_processes()

    profile(args)

    if args.train:
        train(args)
    if args.test:
        test(args)
    if args.demo:
        demo(args)


if __name__ == "__main__":
    main()
