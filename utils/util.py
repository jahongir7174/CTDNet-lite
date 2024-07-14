import copy
import random

import cv2
import math
import numpy
import torch


def setup_seed():
    """
    Setup random seed.
    """
    random.seed(0)
    numpy.random.seed(0)
    torch.manual_seed(0)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def setup_multi_processes():
    """
    Setup multi-processing environment variables.
    """
    import cv2
    from os import environ
    from platform import system

    # set multiprocess start method as `fork` to speed up the training
    if system() != 'Windows':
        torch.multiprocessing.set_start_method('fork', force=True)

    # disable opencv multithreading to avoid system being overloaded
    cv2.setNumThreads(0)

    # setup OMP threads
    if 'OMP_NUM_THREADS' not in environ:
        environ['OMP_NUM_THREADS'] = '1'

    # setup MKL threads
    if 'MKL_NUM_THREADS' not in environ:
        environ['MKL_NUM_THREADS'] = '1'


def export_onnx(args):
    import onnx  # noqa

    inputs = ['images']
    outputs = ['outputs']
    dynamic = {'outputs': {0: 'batch', 1: 'anchors'}}

    m = torch.load('./weights/best.pt')['model'].float()
    x = torch.zeros((1, 3, args.input_size, args.input_size))

    torch.onnx.export(m.cpu(), x.cpu(),
                      './weights/best.onnx',
                      verbose=False,
                      opset_version=12,
                      # WARNING: DNN inference with torch>=1.12 may require do_constant_folding=False
                      do_constant_folding=True,
                      input_names=inputs,
                      output_names=outputs,
                      dynamic_axes=dynamic or None)

    # Checks
    model_onnx = onnx.load('./weights/best.onnx')  # load onnx model
    onnx.checker.check_model(model_onnx)  # check onnx model

    onnx.save(model_onnx, './weights/best.onnx')
    # Inference example
    # https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/autobackend.py


def strip_optimizer(filename):
    x = torch.load(filename, map_location=torch.device('cpu'))
    x['model'].half()  # to FP16
    for p in x['model'].parameters():
        p.requires_grad = False
    torch.save(x, filename)


def load_weight(model, ckpt, prefix='backbone.'):
    dst = model.state_dict()
    src = torch.load(ckpt)['model'].float().state_dict()
    ckpt = {}
    for k, v in src.items():
        if prefix + k in dst and v.shape == dst[prefix + k].shape:
            ckpt[prefix + k] = v
    model.load_state_dict(state_dict=ckpt, strict=False)
    return model


def set_params(model, lr):
    p1 = []
    p2 = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        if param.ndim <= 1 or name.endswith(".bias"):  # bias (no decay)
            p1.append(param)
        else:
            p2.append(param)  # weight (with decay)
    return [{'params': p1, 'weight_decay': 0.00, 'lr': lr},
            {'params': p2, 'weight_decay': 1E-3, 'lr': lr}]


def plot_lr(args, optimizer, scheduler, num_steps):
    from matplotlib import pyplot

    optimizer = copy.copy(optimizer)
    scheduler = copy.copy(scheduler)

    y = []
    for epoch in range(args.epochs):
        for i in range(num_steps):
            step = i + num_steps * epoch
            scheduler.step(step, optimizer)
            y.append(optimizer.param_groups[0]['lr'])
    pyplot.plot(y, '.-', label='LR')
    pyplot.xlabel('step')
    pyplot.ylabel('LR')
    pyplot.grid()
    pyplot.xlim(0, args.epochs * num_steps)
    pyplot.ylim(0)
    pyplot.savefig('./weights/lr.png', dpi=200)
    pyplot.close()


def compute_metric(outputs, targets, num):
    eps = numpy.finfo(float).eps

    shape = outputs.shape
    vector = torch.linspace(1. / (num + 1), 1. - 1. / (num + 1), num, device=targets.device)

    outputs = outputs.detach()
    targets = targets.detach()
    targets.requires_grad = False
    outputs.requires_grad = False
    outputs = outputs.view(shape[0], -1)
    targets = targets.view(shape[0], -1)
    length = outputs.shape[1]
    output = outputs.expand(num, shape[0], length)
    target = targets.expand(num, shape[0], length)
    vector = vector.expand(shape[0], length, num).permute(2, 0, 1)

    bi_res = (outputs > vector).float()
    intersect = (target * bi_res).sum(dim=2)
    recall = (intersect / (target.sum(dim=2) + eps)).sum(dim=1)
    precision = (intersect / (bi_res.sum(dim=2) + eps)).sum(dim=1)
    mae = (output[0] - target[0]).abs().sum() / length

    return precision, recall, mae


class CosineLR:
    def __init__(self, args, optimizer):
        self.min_lr = 1E-5
        self.epochs = args.epochs
        self.learning_rates = [x['lr'] for x in optimizer.param_groups]

    def step(self, epoch, optimizer):
        param_groups = optimizer.param_groups
        for param_group, lr in zip(param_groups, self.learning_rates):
            alpha = math.cos(math.pi * epoch / self.epochs)
            lr = 0.5 * (lr - self.min_lr) * (1 + alpha)
            param_group['lr'] = self.min_lr + lr


class EMA:
    """
    Updated Exponential Moving Average (EMA) from https://github.com/rwightman/pytorch-image-models
    Keeps a moving average of everything in the model state_dict (parameters and buffers)
    For EMA details see https://www.tensorflow.org/api_docs/python/tf/train/ExponentialMovingAverage
    """

    def __init__(self, model, decay=0.9995, tau=2000, updates=0):
        # Create EMA
        self.ema = copy.deepcopy(model).eval()  # FP32 EMA
        self.updates = updates  # number of EMA updates
        # decay exponential ramp (to help early epochs)
        self.decay = lambda x: decay * (1 - math.exp(-x / tau))
        for p in self.ema.parameters():
            p.requires_grad_(False)

    def update(self, model):
        if hasattr(model, 'module'):
            model = model.module
        # Update EMA parameters
        with torch.no_grad():
            self.updates += 1
            d = self.decay(self.updates)

            msd = model.state_dict()  # model state_dict
            for k, v in self.ema.state_dict().items():
                if v.dtype.is_floating_point:
                    v *= d
                    v += (1 - d) * msd[k].detach()


class AverageMeter:
    def __init__(self):
        self.num = 0
        self.sum = 0
        self.avg = 0

    def update(self, v, n):
        self.num = self.num + n
        self.sum = self.sum + v * n
        self.avg = self.sum / self.num


class DiceLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.eps = 1E-5
        self.sigmoid = torch.nn.Sigmoid()
        self.bce_loss = torch.nn.BCEWithLogitsLoss()

    def forward(self, outputs, targets):
        bce = self.bce_loss(outputs, targets)

        outputs = self.sigmoid(outputs)
        dice_mul = (outputs * targets).sum()
        dice_sum = outputs.sum() + targets.sum()
        return bce + 1 - (2 * dice_mul + self.eps) / (dice_sum + self.eps)


class CTLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.bce_loss = torch.nn.BCEWithLogitsLoss()
        self.avg_pool = torch.nn.AvgPool2d(kernel_size=3, stride=1, padding=1)
        self.max_pool = torch.nn.MaxPool2d(kernel_size=5, stride=1, padding=2)

    def forward(self, outputs, targets):
        targets = self.mask_to_edge(targets)
        return self.bce_loss(outputs, targets)

    def mask_to_edge(self, targets):
        erosion = 1 - self.max_pool(1 - targets)  # erosion
        dilation = self.max_pool(targets)  # dilation
        return dilation - erosion


class ComputeLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.ct_loss = CTLoss()
        self.bce_loss = torch.nn.BCEWithLogitsLoss()

    def forward(self, outputs, targets):
        targets = targets.unsqueeze(1)
        loss1 = self.dice_loss(outputs[0], targets) + self.bce_loss(outputs[0], targets)
        loss2 = self.dice_loss(outputs[1], targets) + self.bce_loss(outputs[1], targets)
        loss3 = self.dice_loss(outputs[2], targets) + self.bce_loss(outputs[2], targets)
        loss4 = self.dice_loss(outputs[3], targets) + self.bce_loss(outputs[3], targets)
        loss5 = self.dice_loss(outputs[4], targets) + self.bce_loss(outputs[4], targets)
        loss6 = self.ct_loss(outputs[5], targets)

        return loss1 + loss2 + loss3 + loss4 + loss5 + loss6

    @staticmethod
    def dice_loss(output, target):
        output = torch.sigmoid(output)

        a = torch.sum(output, dim=(1, 2, 3))
        b = torch.sum(target, dim=(1, 2, 3))
        ab = torch.sum(output * target, dim=(1, 2, 3))
        # dice loss with Laplace smoothing
        return 1 - (2 * (ab + 1) / (a + b + 1)).mean()


class Detector:
    def __init__(self):

        self.min_size = 3
        self.mask_threshold = 0.8

        self.model = torch.load('./weights/best.pt', 'cpu')['model'].float()
        self.model.eval()

        self.mean = numpy.array([0.406, 0.456, 0.485])
        self.mean = self.mean.reshape((1, 1, 3)).astype('float32')

        self.std = numpy.array([0.225, 0.224, 0.229])
        self.std = self.std.reshape((1, 1, 3)).astype('float32')

    def mask_to_polygon(self, mask):
        boxes = []
        contours = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[0]

        for contour in contours:
            rect = cv2.minAreaRect(contour)

            if min(rect[1]) < self.min_size:
                continue

            box = cv2.boxPoints(rect)
            box = sorted(list(box), key=lambda x: x[0])

            if box[1][1] > box[0][1]:
                index_1 = 0
                index_4 = 1
            else:
                index_1 = 1
                index_4 = 0
            if box[3][1] > box[2][1]:
                index_2 = 2
                index_3 = 3
            else:
                index_2 = 3
                index_3 = 2

            box = [box[index_1], box[index_2], box[index_3], box[index_4]]
            boxes.append(numpy.array(box).astype("int32"))

        return numpy.array(boxes, dtype="int32")

    def filter_polygon(self, points, shape):
        filtered_points = []
        for point in points:
            if type(point) is list:
                point = numpy.array(point)
            point = self.clockwise_order(point)
            point = self.clip(point, shape)
            w = int(numpy.linalg.norm(point[0] - point[1]))
            h = int(numpy.linalg.norm(point[0] - point[3]))
            if w <= self.min_size or h <= self.min_size:
                continue
            filtered_points.append(point)
        return numpy.array(filtered_points)

    @staticmethod
    def clockwise_order(point):
        poly = numpy.zeros(shape=(4, 2), dtype="float32")
        s = point.sum(axis=1)
        poly[0] = point[numpy.argmin(s)]
        poly[2] = point[numpy.argmax(s)]
        tmp = numpy.delete(point, (numpy.argmin(s), numpy.argmax(s)), axis=0)
        diff = numpy.diff(numpy.array(tmp), axis=1)
        poly[1] = tmp[numpy.argmin(diff)]
        poly[3] = tmp[numpy.argmax(diff)]
        return poly

    @staticmethod
    def clip(points, shape):
        for i in range(points.shape[0]):
            points[i, 0] = int(min(max(points[i, 0], 0), shape[1] - 1))
            points[i, 1] = int(min(max(points[i, 1], 0), shape[0] - 1))
        return points

    @staticmethod
    def refine(polygon, depth_frame, window_size=5):
        refined_points = []
        h, w = depth_frame.shape

        for point in polygon:
            x, y = point
            x, y = int(x), int(y)

            # Define a local window around the point
            x1, y1 = max(0, x - window_size), max(0, y - window_size)
            x2, y2 = min(w, x + window_size + 1), min(h, y + window_size + 1)

            # Get the local depth window
            local_depth = depth_frame[y1:y2, x1:x2]

            # Find the point with the minimum depth (closest to camera)
            min_depth_idx = numpy.unravel_index(local_depth.argmin(), local_depth.shape)
            refined_x = x1 + min_depth_idx[1]
            refined_y = y1 + min_depth_idx[0]

            refined_points.append([refined_x, refined_y])

        return numpy.array(refined_points)

    def forward(self, frame):
        x = cv2.resize(frame, dsize=(256, 256))
        x = x.astype('float32') / 255.0
        x = (x - self.mean) / self.std

        cv2.cvtColor(x, cv2.COLOR_BGR2RGB, x)
        x = x.transpose((2, 0, 1))
        x = torch.from_numpy(x)
        x = x.unsqueeze(0)

        with torch.no_grad():
            y = self.model(x).sigmoid()

        y = torch.nn.functional.interpolate(y,
                                            frame.shape[:2],
                                            mode='bilinear', align_corners=True)
        return y.cpu().numpy()[0, 0]

    def apply(self, frame):
        h, w = frame.shape[:2]
        mask = self.forward(frame)
        mask = mask > self.mask_threshold
        mask = (mask * 255).astype("uint8")

        polygon = self.mask_to_polygon(mask)
        polygon = self.filter_polygon(polygon, shape=(h, w))
        return polygon
