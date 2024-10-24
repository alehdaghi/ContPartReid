import logging
import os
import sys

import numpy as np
import torch
import scipy.io as sio

from ignite.engine import Events
from ignite.handlers import ModelCheckpoint
from ignite.handlers import Timer

from engine.engine import create_eval_engine
from engine.engine import create_train_engine
from engine.metric import AutoKVMetric
from part.mmd import partMMD
from utils.eval_sysu import eval_sysu, pairwise_distance
from utils.eval_regdb import eval_regdb
from configs.default.dataset import dataset_cfg
from configs.default.strategy import strategy_cfg
from torch.nn import functional as F

def get_trainer(dataset, model, optimizer, lr_scheduler=None, logger=None, writer=None, non_blocking=False, log_period=10,
                save_dir="checkpoints", prefix="model", gallery_loader=None, query_loader=None,
                eval_interval=None, start_eval=None):
    if logger is None:
        logger = logging.getLogger()
        logger.setLevel(logging.WARN)

    # trainer
    trainer = create_train_engine(model, optimizer, non_blocking)

    # checkpoint handler
    handler = ModelCheckpoint(save_dir, prefix, save_interval=1, n_saved=2, create_dir=True,
                              save_as_state_dict=True, require_empty=False)
    trainer.add_event_handler(Events.EPOCH_COMPLETED, handler, {"model": model})

    # metric
    timer = Timer(average=True)

    kv_metric = AutoKVMetric()

    # evaluator
    evaluator = None
    if not type(eval_interval) == int:
        raise TypeError("The parameter 'validate_interval' must be type INT.")
    if not type(start_eval) == int:
        raise TypeError("The parameter 'start_eval' must be type INT.")
    if eval_interval > 0 and gallery_loader is not None and query_loader is not None:
        evaluator = create_eval_engine(model, non_blocking)

    @trainer.on(Events.STARTED)
    def train_start(engine):
        setattr(engine.state, "best_rank1", 0.0)

    @trainer.on(Events.COMPLETED)
    def train_completed(engine):
        torch.cuda.empty_cache()

        # extract query feature
        evaluator.run(query_loader)

        q_feats = torch.cat(evaluator.state.feat_list, dim=0)
        q_ids = torch.cat(evaluator.state.id_list, dim=0).numpy()
        q_cams = torch.cat(evaluator.state.cam_list, dim=0).numpy()
        q_img_paths = np.concatenate(evaluator.state.img_path_list, axis=0)

        # extract gallery feature
        evaluator.run(gallery_loader)

        g_feats = torch.cat(evaluator.state.feat_list, dim=0)
        g_ids = torch.cat(evaluator.state.id_list, dim=0).numpy()
        g_cams = torch.cat(evaluator.state.cam_list, dim=0).numpy()
        g_img_paths = np.concatenate(evaluator.state.img_path_list, axis=0)

        print("best rank1={:.2f}%".format(engine.state.best_rank1))
        # np.save("{}/QT.npy".format(save_dir), q_feats.numpy())
        # np.save("{}/GT.npy".format(save_dir), g_feats.numpy())
        # np.save("{}/qIDT.npy".format(save_dir), q_ids)
        # np.save("{}/gIDT.npy".format(save_dir), g_ids)
        # np.save("{}/qCamT.npy".format(save_dir), q_cams)
        # np.save("{}/gCamT.npy".format(save_dir), g_cams)
        # sys.exit(0)

        if dataset == 'sysu':
            perm = sio.loadmat(os.path.join(dataset_cfg.sysu.data_root, 'exp', 'rand_perm_cam.mat'))[
                'rand_perm_cam']
            logging.info('no aim:')
            eval_sysu(q_feats, q_ids, q_cams, g_feats, g_ids, g_cams, g_img_paths, perm, mode='all', num_shots=1, aim=False)
            eval_sysu(q_feats, q_ids, q_cams, g_feats, g_ids, g_cams, g_img_paths, perm, mode='all', num_shots=10, aim=False)
            eval_sysu(q_feats, q_ids, q_cams, g_feats, g_ids, g_cams, g_img_paths, perm, mode='indoor', num_shots=1, aim=False)
            eval_sysu(q_feats, q_ids, q_cams, g_feats, g_ids, g_cams, g_img_paths, perm, mode='indoor', num_shots=10, aim=False)
            logging.info('aim:')
            eval_sysu(q_feats, q_ids, q_cams, g_feats, g_ids, g_cams, g_img_paths, perm, mode='all', num_shots=1, aim=True, k1=4, k2=1)
            eval_sysu(q_feats, q_ids, q_cams, g_feats, g_ids, g_cams, g_img_paths, perm, mode='all', num_shots=10, aim=True, k1=20, k2=6)
            eval_sysu(q_feats, q_ids, q_cams, g_feats, g_ids, g_cams, g_img_paths, perm, mode='indoor', num_shots=1, aim=True, k1=2, k2=2)
            eval_sysu(q_feats, q_ids, q_cams, g_feats, g_ids, g_cams, g_img_paths, perm, mode='indoor', num_shots=10, aim=True, k1=20, k2=6)
        elif dataset == 'regdb':
            logging.info('infrared to visible')
            eval_regdb(q_feats, q_ids, q_cams, g_feats, g_ids, g_cams, g_img_paths, aim=False)
            logging.info('visible to infrared')
            eval_regdb(g_feats, g_ids, g_cams, q_feats, q_ids, q_cams, q_img_paths, aim=False)
            logging.info('infrared to visible aim')
            eval_regdb(q_feats, q_ids, q_cams, g_feats, g_ids, g_cams, g_img_paths, aim=True, k1=8, k2=2)
            logging.info('visible to infrared aim')
            eval_regdb(g_feats, g_ids, g_cams, q_feats, q_ids, q_cams, q_img_paths, aim=True, k1=8, k2=2)
        elif dataset == 'market':
            eval_regdb(q_feats, q_ids, q_cams, g_feats, g_ids, g_cams, g_img_paths, aim=engine.aim)


        evaluator.state.feat_list.clear()
        evaluator.state.id_list.clear()
        evaluator.state.cam_list.clear()
        evaluator.state.img_path_list.clear()
        del q_feats, q_ids, q_cams, g_feats, g_ids, g_cams

        torch.cuda.empty_cache()

    @trainer.on(Events.EPOCH_STARTED)
    def epoch_started_callback(engine):
    
        epoch = engine.state.epoch
        if model.mutual_learning:
            model.update_rate = min(100 / (epoch + 1), 1.0) * model.update_rate_

        kv_metric.reset()
        timer.reset()

    @trainer.on(Events.EPOCH_COMPLETED)
    def epoch_completed_callback(engine):
        epoch = engine.state.epoch

        if lr_scheduler is not None:
            lr_scheduler.step()

        if epoch % eval_interval == 0:
            logger.info("Model saved at {}/{}_model_{}.pth".format(save_dir, prefix, epoch))

        if evaluator and epoch % eval_interval == 0 and epoch >= start_eval:
            torch.cuda.empty_cache()

            # extract query feature
            evaluator.run(query_loader)

            q_feats = torch.cat(evaluator.state.feat_list, dim=0)
            q_ids = torch.cat(evaluator.state.id_list, dim=0).numpy()
            q_cams = torch.cat(evaluator.state.cam_list, dim=0).numpy()
            q_img_paths = np.concatenate(evaluator.state.img_path_list, axis=0)
            q_feats2 = torch.cat(evaluator.state.feat_list2, dim=0)

            # extract gallery feature
            evaluator.run(gallery_loader)

            g_feats = torch.cat(evaluator.state.feat_list, dim=0)
            g_ids = torch.cat(evaluator.state.id_list, dim=0).numpy()
            g_cams = torch.cat(evaluator.state.cam_list, dim=0).numpy()
            g_img_paths = np.concatenate(evaluator.state.img_path_list, axis=0)
            g_feats2 = torch.cat(evaluator.state.feat_list2, dim=0)

            # if writer is not None:
            #     t = min(epoch // 20, model.part_num)
            #     IZ, VZ, IV, ZZ = partMMD(t, q_feats[:, :-2048], g_feats[:, :-2048], q_ids, g_ids)
            #
            #     writer.add_scalar("mmd/IZ", IZ, epoch)
            #     writer.add_scalar("mmd/VZ", VZ, epoch)
            #     writer.add_scalar("mmd/IV", IV, epoch)
            #     writer.add_scalar("mmd/ZZ", ZZ, epoch)
            #     writer.add_scalar("mmd/t", t, epoch)

            if dataset == 'sysu':
                #breakpoint()
                perm = sio.loadmat(os.path.join(dataset_cfg.sysu.data_root, 'exp', 'rand_perm_cam.mat'))[
                    'rand_perm_cam']
                # dismatG = pairwise_distance(F.normalize(q_feats[:, -2048:],dim=1), F.normalize(g_feats[:, -2048:],dim=1))
                # dismatP = pairwise_distance(F.normalize(q_feats[:, :-2048],dim=1), F.normalize(g_feats[:, :-2048],dim=1))
                dismatA = pairwise_distance(F.normalize(q_feats,dim=1), F.normalize(g_feats,dim=1))
                dismatB = pairwise_distance(F.normalize(q_feats2, dim=1), F.normalize(g_feats2, dim=1))

                # dismatGA = dismatG + dismatP
                #
                # dismatG2 = pairwise_distance(F.normalize(q_feats2[:, -2048:], dim=1),
                #                             F.normalize(g_feats2[:, -2048:], dim=1))
                # dismatP2 = pairwise_distance(F.normalize(q_feats2[:, :-2048], dim=1),
                #                             F.normalize(g_feats2[:, :-2048], dim=1))
                # dismatA2 = pairwise_distance(F.normalize(q_feats2, dim=1), F.normalize(g_feats2, dim=1))
                # dismatGA2 = dismatG + dismatP
                q_inf = np.in1d(q_cams, [3, 6])
                q_vis = ~np.in1d(q_cams, [3, 6])

                g_inf = np.in1d(g_cams, [3, 6])
                g_vis = ~np.in1d(g_cams, [3, 6])

                mask = torch.from_numpy(q_inf[:, None] * g_inf[None, :]).float()

                dismatS = dismatA + 0.7*mask*dismatB#+ dismatA + dismatA2 + dismatGA2

                dismat = (1 - mask) * dismatA + 0.7 * mask * dismatB
                # eval_sysu(q_feats, q_ids, q_cams, g_feats, g_ids, g_cams, g_img_paths, perm, mode='all', num_shots=1, aim=False, dist_matAll=dismatG)
                # eval_sysu(q_feats, q_ids, q_cams, g_feats, g_ids, g_cams, g_img_paths, perm, mode='all', num_shots=1, aim=False, dist_matAll=dismatP)
                # eval_sysu(q_feats, q_ids, q_cams, g_feats, g_ids, g_cams, g_img_paths, perm, mode='all', num_shots=1, aim=False, dist_matAll=dismatGA)

                # eval_sysu(q_feats2, q_ids, q_cams, g_feats, g_ids, g_cams, g_img_paths, perm, mode='all', num_shots=1, aim=False, dist_matAll=dismatG2)
                # eval_sysu(q_feats2, q_ids, q_cams, g_feats, g_ids, g_cams, g_img_paths, perm, mode='all', num_shots=1, aim=False, dist_matAll=dismatP2)
                # eval_sysu(q_feats2, q_ids, q_cams, g_feats, g_ids, g_cams, g_img_paths, perm, mode='all', num_shots=1, aim=False, dist_matAll=dismatGA2)
                Q = torch.zeros((q_feats.shape[0], q_feats.shape[1] * 3))
                G = torch.zeros((g_feats.shape[0], g_feats.shape[1] * 3))
                breakpoint()
                Q[:, 0:q_feats.shape[1]] = q_feats2
                Q[:, q_feats.shape[1]:2*q_feats.shape[1]] = q_feats

                G[g_inf, 0:g_feats.shape[1]] = g_feats2[g_inf]
                G[:, g_feats.shape[1]:2 * g_feats.shape[1]] = g_feats
                G[g_vis, 2 * g_feats.shape[1]: ] = g_feats2[g_vis]
                M = pairwise_distance(F.normalize(Q, dim=1), F.normalize(G, dim=1))

                mAP, r1, r5, _, _ = eval_sysu(q_feats, q_ids, q_cams, g_feats, g_ids, g_cams, g_img_paths, perm, mode='all', num_shots=1, aim=False, dist_matAll=None)
                eval_sysu(q_feats2, q_ids, q_cams, g_feats2, g_ids, g_cams, g_img_paths, perm, mode='all', num_shots=1, aim=False, dist_matAll=None)
                eval_sysu(q_feats2, q_ids, q_cams, g_feats2, g_ids, g_cams, g_img_paths, perm, mode='all', num_shots=1,
                          aim=False, dist_matAll=dismatS)
                eval_sysu(q_feats2, q_ids, q_cams, g_feats2, g_ids, g_cams, g_img_paths, perm, mode='all', num_shots=1,
                          aim=False, dist_matAll=dismat)

            elif dataset == 'regdb':
                print('infrared to visible')
                mAP, r1, r5, _, _ = eval_regdb(q_feats, q_ids, q_cams, g_feats, g_ids, g_cams, g_img_paths, aim=False)
                print('visible to infrared')
                mAP, r1_, r5, _, _ = eval_regdb(g_feats, g_ids, g_cams, q_feats, q_ids, q_cams, q_img_paths, aim=False)
                r1 = (r1 + r1_) / 2
            elif dataset == 'market':
                mAP, r1, r5, _, _ = eval_regdb(q_feats, q_ids, q_cams, g_feats, g_ids, g_cams, g_img_paths, aim=engine.aim)
            
            if r1 > engine.state.best_rank1:
                engine.state.best_rank1 = r1
                torch.save(model.state_dict(), "{}/model_best.pth".format(save_dir))
            torch.save(model.state_dict(), "{}/model_{}.pth".format(save_dir, epoch))
            if writer is not None:

                writer.add_scalar('eval/mAP', mAP, epoch)
                writer.add_scalar('eval/r1', r1, epoch)
                writer.add_scalar('eval/r5', r5, epoch)

            evaluator.state.feat_list.clear()
            evaluator.state.id_list.clear()
            evaluator.state.cam_list.clear()
            evaluator.state.img_path_list.clear()
            evaluator.state.feat_list2.clear()
            del q_feats, q_ids, q_cams, g_feats, g_ids, g_cams

            torch.cuda.empty_cache()

    @trainer.on(Events.ITERATION_COMPLETED)
    def iteration_complete_callback(engine):
        if engine.state.output is None:
            return
        timer.step()

        # print(engine.state.output)
        kv_metric.update(engine.state.output)

        epoch = engine.state.epoch
        iteration = engine.state.iteration
        iter_in_epoch = iteration - (epoch - 1) * len(engine.state.dataloader)

        if iteration % log_period == 0:
            batch_size = engine.state.batch[0].size(0)
            speed = batch_size / timer.value()

            msg = "Ep[%d] Bat [%d]\tSpd: %.2f sam/s" % (epoch, iteration, speed)

            metric_dict = kv_metric.compute()

            # log output information
            if logger is not None:
                for k in sorted(metric_dict.keys()):
                    msg += "\t%s: %.3f" % (k, metric_dict[k])
                    if writer is not None:
                        writer.add_scalar('metric/{}'.format(k), metric_dict[k], iteration)

                logger.info(msg)

            kv_metric.reset()
            timer.reset()

    trainer.train_completed = train_completed
    trainer.epoch_completed_callback = epoch_completed_callback
    return trainer
