from ultralytics import YOLO
import argparse
import os

ROOT = os.path.abspath('.') + "/"


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='/ultralytics/cfg/datasets/VOC.yaml', help='dataset.yaml path')
    parser.add_argument('--config', type=str, default='/ultralytics/cfg/models/mamba-yolo/Mamba-YOLO-B.yaml', help='model path(s)')
    parser.add_argument('--batch_size', type=int, default=4, help='batch size')
    parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--task', default='train', help='train, val, test, speed or study')
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--workers', type=int, default=4, help='max dataloader workers (per RANK in DDP mode)')
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--optimizer', default='Adam', help='SGD, Adam, AdamW')
    parser.add_argument('--amp', action='store_true', help='open amp')
    parser.add_argument('--project', default=ROOT + '/output_dir/mscoco', help='save to project/name')
    parser.add_argument('--name', default='mambayolo', help='save to project/name')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    opt = parser.parse_args()
    return opt


if __name__ == '__main__':
    opt = parse_opt()
    task = opt.task
    args = {
        "data": ROOT + opt.data,
        "epochs": opt.epochs,
        "workers": opt.workers,
        "batch": opt.batch_size,
        "optimizer": opt.optimizer,
        "device": opt.device,
        "amp": opt.amp,
        "project": ROOT + opt.project,
        "name": opt.name,
    }
    model_conf = ROOT + opt.config
    task_type = {
        "train": YOLO(model_conf).train(**args),
        "val": YOLO(model_conf).val(**args),
        "test": YOLO(model_conf).test(**args),
    }
    task_type.get(task)
    
# from ultralytics import YOLO
# import argparse
# import os
# import torch
# from torch.nn.parallel import DistributedDataParallel as DDP
# import torch.distributed as dist

# ROOT = os.path.abspath('.') + "/"


# def parse_opt():
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--data', type=str, default='/ultralytics/cfg/datasets/VOC.yaml', help='dataset.yaml path')
#     parser.add_argument('--config', type=str, default='/ultralytics/cfg/models/mamba-yolo/Mamba-YOLO-B.yaml', help='model path(s)')
#     parser.add_argument('--batch_size', type=int, default=4, help='batch size')
#     parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=640, help='inference size (pixels)')
#     parser.add_argument('--task', default='train', help='train, val, test, speed or study')
#     parser.add_argument('--device', default='0,1', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
#     parser.add_argument('--workers', type=int, default=4, help='max dataloader workers (per RANK in DDP mode)')
#     parser.add_argument('--epochs', type=int, default=300)
#     parser.add_argument('--optimizer', default='Adam', help='SGD, Adam, AdamW')
#     parser.add_argument('--amp', action='store_true', help='open amp')
#     parser.add_argument('--project', default=ROOT + '/output_dir/mscoco', help='save to project/name')
#     parser.add_argument('--name', default='mambayolo', help='save to project/name')
#     parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
#     parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
#     opt = parser.parse_args()
#     return opt


# if __name__ == '__main__':
#     opt = parse_opt()
#     task = opt.task
#     args = {
#         "data": ROOT + opt.data,
#         "epochs": opt.epochs,
#         "workers": opt.workers,
#         "batch": opt.batch_size,
#         "optimizer": opt.optimizer,
#         "device": opt.device,
#         "amp": opt.amp,
#         "project": ROOT + opt.project,
#         "name": opt.name,
#     }
#     model_conf = ROOT + opt.config
#     task_type = {
#         "train": YOLO(model_conf).train(**args),
#         "val": YOLO(model_conf).val(**args),
#         "test": YOLO(model_conf).test(**args),
#     }
#     task_type.get(task)

#     # 初始化分布式环境
#     dist.init_process_group(backend='nccl', init_method='env://')
#     local_rank = int(os.environ['LOCAL_RANK'])
#     torch.cuda.set_device(local_rank)

#     # 创建模型并将其移动到当前设备
#     model = YOLO(model_conf).model
#     model.to(local_rank)

#     # 使用DP包装模型，并启用find_unused_parameters=True
#     model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)

#     # 开始训练
#     model.train()