import os

import yaml
from tqdm import tqdm

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import torch
from torch.nn import functional as F
import logging
from data import get_test_loader, get_train_simple_loader
from data import get_train_loader
from engine import get_trainer
from models.baseline import Baseline
import pprint
from utils.eval_sysu import eval_sysu, pairwise_distance


def extract_features(model, loader, device):
    f_r = torch.empty((0)).to(device)
    f_e = torch.empty((0)).to(device)
    Y, C, M = torch.empty((0)), torch.empty((0)), torch.empty((0))
    with torch.no_grad():
        img_iter = tqdm(enumerate(loader), total=len(loader), desc='Collecting feats', ncols=0)
        for i, (Xs, ys, cam, _, _) in img_iter:
            batch_size, _, H, W = Xs.shape
            feats1, feats2 = model(Xs.to(device), cam_ids=cam.to(device))
            Y = torch.cat([Y, ys.detach()], dim=0)
            C = torch.cat([C, cam.detach()], dim=0)
            sub = (cam == 3) + (cam == 6)
            M = torch.cat([M, sub.int()], dim=0)
            f_e = torch.cat([f_e, feats1.detach()], dim=0)
            f_r = torch.cat([f_r, feats2.detach()], dim=0)

    return f_e, f_r, Y, C, M
def saveFeat(model, project_loader):
    n = len(project_loader.dataset)
    device = 'cuda'
    f_r = torch.empty((0)).to(device)
    f_e = torch.empty((0)).to(device)
    Y, C, M = torch.empty((0)), torch.empty((0)), torch.empty((0))
    f_e, f_r, Y, C, M = extract_features(model, project_loader, device)
    save_dir = f'./results/feats/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    torch.save(f_r, save_dir + '/feat_r.pt')
    torch.save(f_e, save_dir + '/feat_e.pt')

    torch.save(Y, save_dir + '/y.pt')
    torch.save(C, save_dir + '/c.pt')

def test(cfg, model, proj1, proj2):
    gallery_loader, query_loader = get_test_loader(dataset=cfg.dataset,
                                             root=cfg.data_root,
                                             batch_size=16,
                                             image_size=cfg.image_size,
                                             num_workers=4)

    q_e, q_r, q_Y, q_C, q_M = extract_features(model, loader=query_loader, device=device)
    g_e, g_r, g_Y, g_C, g_M = extract_features(model, loader=query_loader, device=device)


def train(cfg):
    # set logger
    log_dir = os.path.join("logs/", cfg.dataset, cfg.prefix)
    if not os.path.isdir(log_dir):
        os.makedirs(log_dir, exist_ok=True)

    logging.basicConfig(format="%(asctime)s %(message)s",
                        filename=log_dir + "/" + cfg.log_name,
                        filemode="w")

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    logger.addHandler(stream_handler)

    logger.info(pprint.pformat(cfg))

    project_loader = get_train_simple_loader(dataset=cfg.dataset,
                                                       root=cfg.data_root,
                                                       batch_size=16,
                                                       image_size=cfg.image_size,
                                                       num_workers=4)

    model = Baseline(num_classes=cfg.num_id,
                     backbone=cfg.backbone,
                     pattern_attention=cfg.pattern_attention,
                     modality_attention=cfg.modality_attention,
                     mutual_learning=cfg.mutual_learning,
                     drop_last_stride=cfg.drop_last_stride,
                     triplet=cfg.triplet,
                     k_size=cfg.k_size,
                     center_cluster=cfg.center_cluster,
                     center=cfg.center,
                     margin=cfg.margin,
                     num_parts=cfg.num_parts,
                     weight_KL=cfg.weight_KL,
                     weight_sid=cfg.weight_sid,
                     weight_sep=cfg.weight_sep,
                     update_rate=cfg.update_rate,
                     classification=cfg.classification,
                     margin1 = cfg.margin1,
                     margin2 = cfg.margin2,
                     dp = cfg.dp,
                     dp_w = cfg.dp_w,
                     cs_w = cfg.cs_w)

    if cfg.resume:
        checkpoint = torch.load(cfg.resume)
        model.load_state_dict(checkpoint)
        print("Model is loaded from " + cfg.resume)
    model = model.cuda()
    model.eval()
    device = 'cuda'
    # saveFeat(model, project_loader)
    # exit(0)
    save_dir = f'./results/feats/'
    F_r = torch.load(save_dir + '/feat_r.pt').to(device)
    F_e = torch.load(save_dir + '/feat_e.pt').to(device)
    Y = torch.load(save_dir + '/y.pt').to(device)
    C = torch.load(save_dir + '/c.pt').to(device)
    M = ((C == 3) + (C == 6)) .to(device)


    proj1, proj2 = torch.nn.Linear(2048,2048).to(device),torch.nn.Linear(2048,2048).to(device)
    optimizer = torch.optim.SGD(list(proj1.parameters()) + list(proj2.parameters()), lr=1e-3)


    batch = 5000
    N = len(M) // batch
    Mask = {}
    Y_u = Y.unique()
    for id in Y_u:
        Mask[id.item()] = (Y==id)
    Mask2 ={True: M, False: ~M}
    for e in range(100):
        print(f'epoch: {e}')
        for i in tqdm(range(N), total=N, desc='Collecting feats', ncols=0):
            f_e, f_r = proj1(F_e), proj2(F_r)
            f_e = F.normalize(f_e, dim=1)
            f_r = F.normalize(f_r, dim=1)
            q_r, q_e = f_r[i * batch: (i + 1) * batch ], f_e[i * batch: (i + 1) * batch]
            q_y = Y[i * batch: (i + 1) * batch]
            q_c = C[i * batch: (i + 1) * batch]
            q_m = M[i * batch: (i + 1) * batch]

            dist_e = -pairwise_distance(q_e, f_e.detach())
            dist_r = -pairwise_distance(q_r, f_r.detach())
            loss = 0
            for j in range(batch):
                p_min = min(dist_e[i][ Mask[q_y[j].item()]].mean(), dist_r[i][( Mask[q_y[j].item()]) & (Mask2[q_m[j].item()])].mean())
                n_max = max(dist_e[i][~Mask[q_y[j].item()]].mean(), dist_r[i][(~Mask[q_y[j].item()]) & (Mask2[q_m[j].item()])].mean())
                loss = loss + max(0, (0.7 - p_min + n_max))
            loss = loss / batch
            print(f'epoch: {e} loss: {loss}')
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        torch.save(proj1.state_dict(), f'{save_dir}/proj1.pt')
        torch.save(proj2.state_dict(), f'{save_dir}/proj2.pt')

    torch.save(proj1.state_dict(), f'{save_dir}/proj1.pt')
    torch.save(proj2.state_dict(), f'{save_dir}/proj2.pt')
    test(model, proj1, proj2)


if __name__ == '__main__':
    import argparse
    import random
    import numpy as np
    from configs.default import strategy_cfg
    from configs.default import dataset_cfg

    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, default="configs/SYSU.yml")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log_name", type=str, default="log.txt")
    parser.add_argument("--backbone", type=str, default="resnet50")
    parser.add_argument("--update_rate", type=float, default=0.02)
    parser.add_argument("--num_parts", type=int, default=7)
    parser.add_argument("--margin1", type=float, default=0.01)
    parser.add_argument("--margin2", type=float, default=0.7)
    parser.add_argument("--dp", type=str, default="l2")
    parser.add_argument("--dp_w", type=float, default=0.5)
    parser.add_argument("--cs_w", type=float, default=1)
    parser.add_argument("--resume", "-r", type=str, default='')
    parser.add_argument("--p_size", type=int, default=10)
    parser.add_argument("--k_size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    customized_cfg = yaml.load(open(args.cfg, "r"), Loader=yaml.SafeLoader)

    cfg = strategy_cfg
    cfg.merge_from_file(args.cfg)

    dataset_cfg = dataset_cfg.get(cfg.dataset)

    for k, v in dataset_cfg.items():
        cfg[k] = v

    cfg.log_name = args.log_name
    cfg.backbone = args.backbone
    cfg.update_rate = args.update_rate
    cfg.num_parts = args.num_parts
    cfg.prefix = f"{cfg.prefix}_{cfg.log_name}"
    cfg.margin1 = args.margin1
    cfg.margin2 = args.margin2
    cfg.dp = args.dp
    cfg.dp_w = args.dp_w
    cfg.cs_w = args.cs_w
    cfg.resume = args.resume
    cfg.p_size = args.p_size
    cfg.k_size = args.k_size
    cfg.workers = args.workers

    if cfg.sample_method == 'identity_uniform':
        cfg.batch_size = cfg.p_size * cfg.k_size

    cfg.freeze()

    train(cfg)
