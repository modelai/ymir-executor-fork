"""use fake DDP to infer
1. split data with `images_rank = images[RANK::WORLD_SIZE]`
2. save splited result with `torch.save(results, f'results_{RANK}.pt')`
3. merge result
"""
import os
import sys
import warnings
from functools import partial

import torch
import torch.distributed as dist
import torch.utils.data as td
from easydict import EasyDict as edict
from tqdm import tqdm
from ymir_exc import result_writer as rw
from ymir_exc.util import YmirStage, get_merged_config

from mining.util import YmirDataset, load_image_file
from utils.general import scale_coords
from utils.ymir_yolov5 import YmirYolov5

LOCAL_RANK = int(os.getenv('LOCAL_RANK', -1))  # https://pytorch.org/docs/stable/elastic/run.html
RANK = int(os.getenv('RANK', -1))
WORLD_SIZE = int(os.getenv('WORLD_SIZE', 1))


def run(ymir_cfg: edict, ymir_yolov5: YmirYolov5):
    # eg: gpu_id = 1,3,5,7  for LOCAL_RANK = 2, will use gpu 5.
    gpu = int(ymir_yolov5.gpu_id.split(',')[LOCAL_RANK])
    device = torch.device('cuda', gpu)
    ymir_yolov5.to(device)

    load_fn = partial(load_image_file, img_size=ymir_yolov5.img_size, stride=ymir_yolov5.stride)
    batch_size_per_gpu = ymir_yolov5.batch_size_per_gpu
    gpu_count = ymir_yolov5.gpu_count
    cpu_count: int = os.cpu_count() or 1
    num_workers_per_gpu = min([
        cpu_count // max(gpu_count, 1), batch_size_per_gpu if batch_size_per_gpu > 1 else 0,
        ymir_yolov5.num_workers_per_gpu
    ])

    with open(ymir_cfg.ymir.input.candidate_index_file, 'r') as f:
        images = [line.strip() for line in f.readlines()]

    # origin dataset
    images_rank = images[RANK::WORLD_SIZE]
    origin_dataset = YmirDataset(images_rank, load_fn=load_fn)
    origin_dataset_loader = td.DataLoader(origin_dataset,
                                          batch_size=batch_size_per_gpu,
                                          shuffle=False,
                                          sampler=None,
                                          num_workers=num_workers_per_gpu,
                                          pin_memory=ymir_yolov5.pin_memory,
                                          drop_last=False)

    results = []
    dataset_size = len(images_rank)
    monitor_gap = max(1, dataset_size // 1000 // batch_size_per_gpu)
    pbar = tqdm(origin_dataset_loader) if RANK == 0 else origin_dataset_loader
    for idx, batch in enumerate(pbar):
        # batch-level sync, avoid 30min time-out error
        if LOCAL_RANK != -1:
            dist.barrier()

        with torch.no_grad():
            pred = ymir_yolov5.forward(batch['image'].float().to(device), nms=True)

        if idx % monitor_gap == 0 and RANK in [0, -1]:
            ymir_yolov5.write_monitor_logger(stage=YmirStage.TASK, p=idx * batch_size_per_gpu / dataset_size)

        preprocess_image_shape = batch['image'].shape[2:]
        for idx, det in enumerate(pred):  # per image
            result_per_image = []
            image_file = batch['image_file'][idx]
            if len(det):
                origin_image_shape = (batch['origin_shape'][0][idx], batch['origin_shape'][1][idx])
                # Rescale boxes from img_size to img size
                det[:, :4] = scale_coords(preprocess_image_shape, det[:, :4], origin_image_shape).round()
                result_per_image.append(det)
            results.append(dict(image_file=image_file, result=result_per_image))

    torch.save(results, f'/out/infer_results_{RANK}.pt')


def main() -> int:
    ymir_cfg = get_merged_config()
    ymir_yolov5 = YmirYolov5(ymir_cfg, task='infer')

    if LOCAL_RANK != -1:
        assert torch.cuda.device_count() > LOCAL_RANK, 'insufficient CUDA devices for DDP command'
        torch.cuda.set_device(LOCAL_RANK)
        dist.init_process_group(backend="nccl" if dist.is_nccl_available() else "gloo")

    run(ymir_cfg, ymir_yolov5)

    # wait all process to save the infer result
    dist.barrier()

    if RANK in [0, -1]:
        results = []
        for rank in range(WORLD_SIZE):
            results.append(torch.load(f'/out/infer_results_{rank}.pt'))

        ymir_infer_result = dict()
        for result in results:
            for img_data in result:
                img_file = img_data['image_file']
                anns = []
                for each_det in img_data['result']:
                    each_det_np = each_det.data.cpu().numpy()
                    for i in range(each_det_np.shape[0]):
                        xmin, ymin, xmax, ymax, conf, cls = each_det_np[i, :6].tolist()
                        if conf < ymir_yolov5.conf_thres:
                            continue
                        if int(cls) >= len(ymir_yolov5.class_names):
                            warnings.warn(f'class index {int(cls)} out of range for {ymir_yolov5.class_names}')
                            continue
                        ann = rw.Annotation(class_name=ymir_yolov5.class_names[int(cls)],
                                            score=conf,
                                            box=rw.Box(x=int(xmin), y=int(ymin), w=int(xmax - xmin),
                                                       h=int(ymax - ymin)))
                        anns.append(ann)
                ymir_infer_result[img_file] = anns
        rw.write_infer_result(infer_result=ymir_infer_result)

    print(f'rank: {RANK}, start destroy process group')
    dist.destroy_process_group()
    return 0


if __name__ == '__main__':
    sys.exit(main())
