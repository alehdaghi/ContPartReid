import torch
import numpy as np

# from apex import amp
from ignite.engine import Engine
from ignite.engine import Events
from torch.autograd import no_grad
from utils.calc_acc import calc_acc
from torch.nn import functional as F


def create_train_engine(model, optimizer, non_blocking=False):
    device = torch.device("cuda", torch.cuda.current_device())
    scaler = torch.cuda.amp.GradScaler()

    def _process_func(engine, batch):
        model.train()

        data, labels, cam_ids, img_paths, img_ids = batch
        epoch = engine.state.epoch

        data = data.to(device, non_blocking=non_blocking)
        labels = labels.to(device, non_blocking=non_blocking)
        cam_ids = cam_ids.to(device, non_blocking=non_blocking)

        optimizer.zero_grad()
        with torch.autocast('cuda', torch.float16):
            loss, metric = model(data, labels,
                                cam_ids=cam_ids,
                                epoch=epoch,
                                iteration = engine.state.iteration)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        return metric

    return Engine(_process_func)


def create_eval_engine(model, non_blocking=False):
    device = torch.device("cuda", torch.cuda.current_device())

    def _process_func(engine, batch):
        model.eval()

        data, labels, cam_ids, img_paths = batch[:4]

        data = data.to(device, non_blocking=non_blocking)

        with no_grad():
            feat, feat2 = model(data, cam_ids=cam_ids.to(device, non_blocking=non_blocking))

        return feat.data.float().cpu(), labels, cam_ids, np.array(img_paths), feat2.data.float().cpu()

    engine = Engine(_process_func)

    @engine.on(Events.EPOCH_STARTED)
    def clear_data(engine):
        # feat list
        if not hasattr(engine.state, "feat_list"):
            setattr(engine.state, "feat_list", [])
        else:
            engine.state.feat_list.clear()

        # id_list
        if not hasattr(engine.state, "id_list"):
            setattr(engine.state, "id_list", [])
        else:
            engine.state.id_list.clear()

        # cam list
        if not hasattr(engine.state, "cam_list"):
            setattr(engine.state, "cam_list", [])
        else:
            engine.state.cam_list.clear()

        # img path list
        if not hasattr(engine.state, "img_path_list"):
            setattr(engine.state, "img_path_list", [])
        else:
            engine.state.img_path_list.clear()

        if not hasattr(engine.state, "feat_list2"):
            setattr(engine.state, "feat_list2", [])
        else:
            engine.state.feat_list2.clear()

    @engine.on(Events.ITERATION_COMPLETED)
    def store_data(engine):
        engine.state.feat_list.append(engine.state.output[0])
        engine.state.id_list.append(engine.state.output[1])
        engine.state.cam_list.append(engine.state.output[2])
        engine.state.img_path_list.append(engine.state.output[3])
        engine.state.feat_list2.append(engine.state.output[4])

    return engine
