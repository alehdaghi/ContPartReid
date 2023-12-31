from __future__ import print_function
import argparse
import sys
import time
import os

import einops

from Transformer import SimpleViT
from part.criterion import CriterionAll, contrastive_loss, generate_edge_tensor

parser = argparse.ArgumentParser(description='PyTorch Cross-Modality Training')
parser.add_argument('--dataset', default='sysu', help='dataset name: regdb or sysu]')
parser.add_argument('--lr', default=0.1 , type=float, help='learning rate, 0.00035 for adam')
parser.add_argument('--optim', default='sgd', type=str, help='optimizer')
parser.add_argument('--arch', default='resnet50', type=str,
                    help='network baseline:resnet18 or resnet50')
parser.add_argument('--resume', '-r', default='', type=str,
                    help='resume from checkpoint')
parser.add_argument('--test-only', action='store_true', help='test only')
parser.add_argument('--model_path', default='save_model/', type=str,
                    help='model save path')
parser.add_argument('--save_epoch', default=20, type=int,
                    metavar='s', help='save model every 10 epochs')
parser.add_argument('--log_path', default='log/', type=str,
                    help='log save path')
parser.add_argument('--vis_log_path', default='log/vis_log/', type=str,
                    help='log save path')
parser.add_argument('--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--img_w', default=144, type=int,
                    metavar='imgw', help='img width')
parser.add_argument('--img_h', default=288, type=int,
                    metavar='imgh', help='img height')
parser.add_argument('--batch-size', default=8, type=int,
                    metavar='B', help='training batch size')
parser.add_argument('--test-batch', default=64, type=int,
                    metavar='tb', help='testing batch size')
parser.add_argument('--method', default='agw', type=str,
                    metavar='m', help='method type: base or agw, adp')
parser.add_argument('--margin', default=0.3, type=float,
                    metavar='margin', help='triplet loss margin')
parser.add_argument('--num_pos', default=4, type=int,
                    help='num of pos per identity in each modality')
parser.add_argument('--trial', default=1, type=int,
                    metavar='t', help='trial (only for RegDB dataset)')
parser.add_argument('--seed', default=0, type=int,
                    metavar='t', help='random seed')
parser.add_argument('--gpu', default='0', type=str,
                    help='gpu device ids for CUDA_VISIBLE_DEVICES')
parser.add_argument('--mode', default='all', type=str, help='all or indoor')

parser.add_argument('--augc', default=0 , type=int,
                    metavar='aug', help='use channel aug or not')
parser.add_argument('--rande', default= 0 , type=float,
                    metavar='ra', help='use random erasing or not and the probability')
parser.add_argument('--kl', default= 0 , type=float,
                    metavar='kl', help='use kl loss and the weight')
parser.add_argument('--alpha', default=1 , type=int,
                    metavar='alpha', help='magnification for the hard mining')
parser.add_argument('--gamma', default=1 , type=int,
                    metavar='gamma', help='gamma for the hard mining')
parser.add_argument('--square', default= 1 , type=int,
                    metavar='square', help='gamma for the hard mining')
                   
args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import torch.utils.data as data
import torchvision.transforms as transforms
import torch.nn.functional as Fn
import torchvision
from data_loader import SYSUData, RegDBData, TestData
from data_manager import *
from eval_metrics import eval_sysu, eval_regdb
from model import embed_net
from utils import *
from loss import OriTripletLoss, TripletLoss_WRT, KLDivLoss, TripletLoss_ADP, CSLoss, TripletLoss
from tensorboardX import SummaryWriter
from ChannelAug import ChannelAdap, ChannelAdapGray, ChannelRandomErasing


set_seed(args.seed)

dataset = args.dataset
if dataset == 'sysu':
    data_path = '../Datasets/SYSU-MM01/'
    log_path = args.log_path + 'sysu_log/'
    test_mode = [1, 2]  # thermal to visible
elif dataset == 'regdb':
    data_path = '../Datasets/RegDB/'
    log_path = args.log_path + 'regdb_log/'
    test_mode = [2, 1]  # visible to thermal

checkpoint_path = args.model_path

if not os.path.isdir(log_path):
    os.makedirs(log_path)
if not os.path.isdir(checkpoint_path):
    os.makedirs(checkpoint_path)
if not os.path.isdir(args.vis_log_path):
    os.makedirs(args.vis_log_path)

suffix = dataset
if args.method == 'adp':
    suffix = suffix + '_{}_joint_co_nog_ch_nog_sq{}'.format(args.method, args.square)
else:
    suffix = suffix + '_{}'.format(args.method)
#suffix = suffix + '_KL_{}'.format(args.kl)
if args.augc==1:
    suffix = suffix + '_aug_G'  
if args.rande>0:
    suffix = suffix + '_erase_{}'.format( args.rande)
    
suffix = suffix + '_p{}_n{}_lr_{}_seed_{}'.format( args.num_pos, args.batch_size, args.lr, args.seed)  

if not args.optim == 'sgd':
    suffix = suffix + '_' + args.optim

if dataset == 'regdb':
    suffix = suffix + '_trial_{}'.format(args.trial)

sys.stdout = Logger(log_path + suffix + '_os.txt')

vis_log_dir = args.vis_log_path + suffix + '/'

if not os.path.isdir(vis_log_dir):
    os.makedirs(vis_log_dir)
writer = SummaryWriter(vis_log_dir)
print("==========\nArgs:{}\n==========".format(args))
device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0

print('==> Loading data..')
# Data loading code
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
transform_train_list = [
    transforms.ToPILImage(),
    transforms.Pad(10),
    transforms.RandomCrop((args.img_h, args.img_w)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normalize]
    
transform_test = transforms.Compose( [
    transforms.ToPILImage(),
    transforms.Resize((args.img_h, args.img_w)),
    transforms.ToTensor(),
    normalize])

invTrans = transforms.Compose([
    transforms.Normalize(mean=[0., 0., 0.], std=[1 / 0.229, 1 / 0.224, 1 / 0.225]),
    transforms.Normalize(mean=[-0.485, -0.456, -0.406], std=[1., 1., 1.]),
])


if args.rande>0:
    transform_train_list = transform_train_list + [ChannelRandomErasing(probability = args.rande)]

if args.augc ==1:
    # transform_train_list = transform_train_list +  [ChannelAdap(probability =0.5)]
    transform_train_list = transform_train_list + [ChannelAdapGray(probability =0.5)]
    
transform_train = transforms.Compose( transform_train_list )

end = time.time()
if dataset == 'sysu':
    # training set
    trainset = SYSUData(data_path, transform=transform_train)
    # generate the idx of each person identity
    color_pos, thermal_pos = GenIdx(trainset.train_color_label, trainset.train_thermal_label)

    # testing set
    query_img, query_label, query_cam = process_query_sysu(data_path, mode=args.mode)
    gall_img, gall_label, gall_cam = process_gallery_sysu(data_path, mode=args.mode, trial=0)

elif dataset == 'regdb':
    # training set
    trainset = RegDBData(data_path, args.trial, transform=transform_train)
    # generate the idx of each person identity
    color_pos, thermal_pos = GenIdx(trainset.train_color_label, trainset.train_thermal_label)

    # testing set
    query_img, query_label = process_test_regdb(data_path, trial=args.trial, modal='visible')
    gall_img, gall_label = process_test_regdb(data_path, trial=args.trial, modal='thermal')

gallset  = TestData(gall_img, gall_label, transform=transform_test, img_size=(args.img_w, args.img_h))
queryset = TestData(query_img, query_label, transform=transform_test, img_size=(args.img_w, args.img_h))

# testing data loader
gall_loader = data.DataLoader(gallset, batch_size=args.test_batch, shuffle=False, num_workers=args.workers)
query_loader = data.DataLoader(queryset, batch_size=args.test_batch, shuffle=False, num_workers=args.workers)

n_class = len(np.unique(trainset.train_color_label))
nquery = len(query_label)
ngall = len(gall_label)

print('Dataset {} statistics:'.format(dataset))
print('  ------------------------------')
print('  subset   | # ids | # images')
print('  ------------------------------')
print('  visible  | {:5d} | {:8d}'.format(n_class, len(trainset.train_color_label)))
print('  thermal  | {:5d} | {:8d}'.format(n_class, len(trainset.train_thermal_label)))
print('  ------------------------------')
print('  query    | {:5d} | {:8d}'.format(len(np.unique(query_label)), nquery))
print('  gallery  | {:5d} | {:8d}'.format(len(np.unique(gall_label)), ngall))
print('  ------------------------------')
print('Data Loading Time:\t {:.3f}'.format(time.time() - end))

print('==> Building model..')
if args.method =='base':
    net = embed_net(n_class, no_local= 'off', gm_pool =  'off', arch=args.arch)
else:
    net = embed_net(n_class, no_local= 'on', gm_pool = 'on', arch=args.arch)
net.to(device)

vit = SimpleViT(token_size=7, num_classes=n_class, dim=2048, depth=3)
vit.to(device)

cudnn.benchmark = True

if len(args.resume) > 0:
    model_path = checkpoint_path + args.resume
    if os.path.isfile(model_path):
        print('==> loading checkpoint {}'.format(args.resume))
        checkpoint = torch.load(model_path)
        start_epoch = checkpoint['epoch']
        net.load_state_dict(checkpoint['net'])
        print('==> loaded checkpoint {} (epoch {})'
              .format(args.resume, checkpoint['epoch']))
    else:
        print('==> no checkpoint found at {}'.format(args.resume))

# define loss function
criterionPart = CriterionAll(num_classes=7)
contrastive = contrastive_loss

criterion_id = nn.CrossEntropyLoss()
BCE_wl = nn.BCEWithLogitsLoss()
if args.method == 'agw':
    criterion_tri = TripletLoss_WRT()
    # loader_batch = args.batch_size * args.num_pos
    # criterion_tri= OriTripletLoss(batch_size=loader_batch, margin=args.margin)
elif args.method == 'adp':
    criterion_tri = TripletLoss_ADP(alpha = args.alpha, gamma = args.gamma, square = args.square)
else:
    loader_batch = args.batch_size * args.num_pos
    criterion_tri= OriTripletLoss(batch_size=loader_batch, margin=args.margin)

cs_loss_fn = CSLoss(k_size=args.num_pos)
criterion_kl = KLDivLoss()
criterion_id.to(device)
criterion_tri.to(device)
criterion_kl.to(device)

cross_triplet_creiteron = TripletLoss(0.3, 'euclidean')

if args.optim == 'sgd':
    ignored_params = list(map(id, net.bottleneck.parameters())) \
                     + list(map(id, net.classifier.parameters()))

    base_params = filter(lambda p: id(p) not in ignored_params, net.parameters())

    optimizer = optim.SGD([
        {'params': base_params, 'lr': 0.1 * args.lr},
        {'params': vit.parameters(), 'lr': 0.1 * args.lr},
        {'params': net.bottleneck.parameters(), 'lr': 0.5 * args.lr},
        {'params': net.classifier.parameters(), 'lr': 0.5 * args.lr}],
        weight_decay=5e-4, momentum=0.9, nesterov=True)

# exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    if epoch < 10:
        lr = args.lr * (epoch + 1) / 10
    elif epoch >= 10 and epoch < 20:
        lr = args.lr
    elif epoch >= 20 and epoch < 50:
        lr = args.lr * 0.1
    elif epoch >= 50:
        lr = args.lr * 0.01

    optimizer.param_groups[0]['lr'] = 0.1 * lr
    for i in range(len(optimizer.param_groups) - 1):
        optimizer.param_groups[i + 1]['lr'] = 0.5 * lr

    return lr


def train(epoch):

    current_lr = adjust_learning_rate(optimizer, epoch)
    train_loss = AverageMeter()
    id_loss = AverageMeter()
    tri_loss = AverageMeter()
    kl_loss = AverageMeter()
    attr_m = AverageMeter()
    data_time = AverageMeter()
    batch_time = AverageMeter()
    part_seg, part_re , part_un =  AverageMeter(), AverageMeter(), AverageMeter()
    correct = 0
    total = 0


    part_sum, reg_sum, unsup_sum = 0, 0, 0
    # switch to train mode
    net.train()
    end = time.time()

    for batch_idx, (input10, input11, input2, label1, label2, p_label10, p_label11, p_label2, attr1, attr2) in enumerate(trainloader):

        labels = torch.cat((label1, label2), 0)

        input2 = Variable(input2.cuda())
        
        
        # input10 = Variable(input10.cuda())
        input1 = Variable(input10.cuda())

        labels = Variable(labels.cuda())
        
        # input1 = torch.cat((input10, input11,),0)
        input2 = Variable(input2.cuda())

        imgs = torch.cat((input1, input2,), 0)

        part_labels = torch.cat((p_label10, p_label2), 0).to(device).type(torch.cuda.LongTensor)
        attr_labels = torch.cat((attr1, attr2), 0).to(device).type(torch.cuda.LongTensor)

        data_time.update(time.time() - end)


        feat, out0, part, partsFeatX3, partsFeat, part_masks, partsScore, featsP, scoreP, attr_score, loss_mean = net(input1, input2)

        bs = imgs.shape[0]
        #parts
        edges = generate_edge_tensor(part_labels).type(torch.cuda.LongTensor)
        good_part = (part_labels != 0).type(torch.int).sum(dim=[1, 2]) > 288 * 144 * 0.15
        part_loss = criterionPart([[part[0][0][good_part], part[0][1][good_part]], [part[1][0][good_part]]],
                                  [part_labels[good_part], edges[good_part]])  # + loss_reg

        F2 = einops.rearrange(partsFeat, '(m n p) ... -> n (p m) ...', p=args.num_pos, m=bs // input2.shape[0])
        cont_part2 = sum([contrastive(f) for f in F2]) / args.batch_size
        # cont_part3 = contrastive(F.transpose(0, 1))
        unsup_part = contrastive(partsFeatX3) + cont_part2 #+ cont_part3
        loss_id_parts = sum([criterion_id(ps, labels) / 6 for ps in partsScore]) #+ criterion_id(scoreP, labels)
        part_seg.update(part_loss.item(), 2 * input1.size(0))
        part_re.update(loss_id_parts.item(), 2 * input1.size(0))
        part_un.update(unsup_part.item(), 2 * input1.size(0))

        loss_dp = 0
        for i in range(part_masks.shape[1]):
            for j in range(i + 1, part_masks.shape[1]):
                loss_dp += ((((part_masks[:, i] - part_masks[:, j]) ** 2).sum(dim=1) / (18 * 9)) ** 0.5).sum()
        loss_dp = - loss_dp / (bs * part_masks.shape[1] * (part_masks.shape[1] - 1) / 2)

        unsup_sum += unsup_part.item()

        attr_loss = torch.tensor(0)#sum([criterion_id(attr_score[i], attr_labels[:,i]) for i in range(9)] + [criterion_id(attr_score[-1], attr_labels[:,-1])])

        # feat_vit, out_vit = vit(feat.reshape(bs, -1, 2048))
        #

        
        
        # loss kl
        n = out0.shape[0]//3
        # out1 = out0.narrow(0,0,n)
        # out2 = out0.narrow(0,2*n,n)
        # loss_kl = criterion_kl(out1, Variable(out2))
        # kl_loss += criterion_kl(F.log_softmax(out2, dim = 1), F.softmax(Variable(out1), dim=1))                                           
        # F = einops.rearrange(feat, '(m n p) ... -> n (p m) ...', p=args.num_pos, m=3)
        # cont_part2 = contrastive(F.transpose(0, 1))

        # loss_tri, batch_acc = criterion_tri(feat_vit, labels)
        # loss_tri, _, _ = cs_loss_fn(feat, labels)

        feat_final, out_vit = vit(feat.reshape(bs, -1, 2048))
        F = einops.rearrange(feat_final, '(m n p) ... -> n (p m) ...', p=args.num_pos, m=bs // input2.shape[0])  # b m*p d


        color_feat, thermal_feat = torch.split(feat_final, input2.shape[0])
        color_label, thermal_label = torch.split(labels, input2.shape[0])
        loss_tri_color = cross_triplet_creiteron(color_feat, thermal_feat, thermal_feat,
                                                 color_label, thermal_label, thermal_label)
        loss_tri_thermal = cross_triplet_creiteron(thermal_feat, color_feat, color_feat,
                                                   thermal_label, color_label, color_label)

        loss_tri = (loss_tri_color + loss_tri_thermal) / 2 + contrastive(F.transpose(0, 1))

        loss_id = criterion_id(out0, labels) + criterion_id(out_vit, labels)

        # loss_tri, batch_acc = criterion_tri(feat_vit, labels)
        # correct += (batch_acc / 2)
        _, predicted = out_vit.max(1)
        correct += (predicted.eq(labels).sum().item())
        
        # pdb.set_trace()
        loss = loss_id + loss_dp + part_loss + unsup_part + loss_id_parts +loss_mean+ loss_tri #+ attr_loss
        if torch.isnan(loss):
            print(input1.sum(), input2.sum())
            print(feat)
            print(loss_id , loss_dp , part_loss , unsup_part , loss_id_parts ,loss_mean, loss_tri)
            exit(1)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # update P
        train_loss.update(loss.item(), 2 * input1.size(0))
        id_loss.update(loss_id.item(), 2 * input1.size(0))
        tri_loss.update(loss_tri.item(), 2 * input1.size(0))
        kl_loss.update(loss_dp.item(), 2 * input1.size(0))
        attr_m.update(loss_mean.item(), 2 * input1.size(0))
        total += labels.size(0)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        if batch_idx % 50 == 0:
            print('E: [{}][{}/{}] '
                  'T:{batch_time.val:.2f}({batch_time.avg:.3f}) '
                  'lr:{:.3f} '
                  'L:{train_loss.val:.2f}({train_loss.avg:.3f}) '
                  'ID:{id_loss.val:.2f}({id_loss.avg:.3f}) '
                  'TR:{tri_loss.val:.2f}({tri_loss.avg:.3f}) '
                  'KL:{kl_loss.val:.2f}({kl_loss.avg:.3f}) '
                  'Pa:{P_loss.val:.2f}({P_loss.avg:.3f}) '
                  'Re:{part_re.val:.2f}({part_re.avg:.3f}) '
                  'Un:{part_un.val:.2f}({part_un.avg:.3f}) '
                  'At:{attr_m.val:.2f}({attr_m.avg:.3f}) '
                  'A:{:.2f}'.format(
                epoch, batch_idx, len(trainloader), current_lr,
                100. * correct / total, batch_time=batch_time,
                train_loss=train_loss, id_loss=id_loss, tri_loss=tri_loss,kl_loss=kl_loss, P_loss=part_seg, part_re=part_re, part_un=part_un, attr_m=attr_m ))

        if batch_idx % 200 == 0:
            B = labels.shape[0]
            index = np.random.choice(np.arange(B), min(B, 16), replace=False)
            h,w = part[0][1].shape[2], part[0][1].shape[3]
            p = Fn.interpolate(part_labels.unsqueeze(1).expand(-1, 3, -1, -1)[index]/6, size=(h, w), mode='bilinear', align_corners=True).unsqueeze(1)
            img = Fn.interpolate(imgs[index], size=(h, w), mode='bilinear', align_corners=True).unsqueeze(1)
            mask =  part[0][1][index].unsqueeze(2).expand(-1,-1,3,-1,-1)
            pModel = (torch.argmax(part[0][1][index], dim=1) / 6).unsqueeze(1).unsqueeze(1).expand(-1,-1,3,-1,-1)
            sample = torch.cat([invTrans(img), p, pModel, mask], dim=1).view(-1, 3, h, w)

            torchvision.utils.save_image(sample, f"sample/part_{str(epoch + 1).zfill(5)}_{str(batch_idx).zfill(5)}.png", normilized=True, nrow=10)
            # with open(f"sample/part_{str(epoch + 1).zfill(5)}_{str(batch_idx).zfill(5)}.txt", 'w') as file:
            #     tr = attr_labels[index]
            #     pr = torch.stack([out[index].max(1)[1] for out in attr_score]).t()
            #     ll = ['sex', 'hair', 'glass', 'T-sh', 'V-ne', 'text', 'jack', 'skirt', 'pants', 'shoes']
            #     for i in range(len(tr)):
            #         for j in range(len(ll)):
            #             file.write(f"{ll[j]}: {tr[i][j]}-{pr[i][j]} ")
            #         file.write("\n")
            #     file.flush()




            # good_part[index]

    writer.add_scalar('total_loss', train_loss.avg, epoch)
    writer.add_scalar('id_loss', id_loss.avg, epoch)
    writer.add_scalar('tri_loss', tri_loss.avg, epoch)
    writer.add_scalar('lr', current_lr, epoch)


def test(epoch):
    # switch to evaluation mode
    net.eval()
    print('Extracting Gallery Feature...')
    start = time.time()
    ptr = 0
    gall_feat = np.zeros((ngall, 2048 + 6 * 2048))
    gall_feat_att = np.zeros((ngall, 2048+ 6 * 2048))
    gall_attr = np.zeros((ngall, 10))
    with torch.no_grad():
        for batch_idx, (input, label) in enumerate(gall_loader):
            batch_num = input.size(0)
            input = Variable(input.cuda())
            feat = net(input, input, test_mode[0])
            feat, out_vit = vit(feat.reshape(input.shape[0], -1, 2048))
            feat = net.l2norm(feat)
            gall_feat[ptr:ptr + batch_num, :] = feat.detach().cpu().numpy()
            # gall_feat_att[ptr:ptr + batch_num, :] = feat_att.detach().cpu().numpy()
            # gall_attr[ptr:ptr + batch_num, :] = torch.stack([out.max(1)[1] for out in attr_score]).t().cpu().numpy()
            ptr = ptr + batch_num
    print('Extracting Time:\t {:.3f}'.format(time.time() - start))

    # switch to evaluation
    net.eval()
    print('Extracting Query Feature...')
    start = time.time()
    ptr = 0
    query_feat = np.zeros((nquery, 2048+ 6 * 2048))
    query_feat_att = np.zeros((nquery, 2048+ 6 * 2048))
    query_attr = np.zeros((nquery, 10))
    with torch.no_grad():
        for batch_idx, (input, label) in enumerate(query_loader):
            batch_num = input.size(0)
            input = Variable(input.cuda())
            feat = net(input, input, test_mode[1])
            feat, out_vit = vit(feat.reshape(input.shape[0], -1, 2048))
            # feat_vit = net.l2norm(feat_vit)
            feat = net.l2norm(feat)
            query_feat[ptr:ptr + batch_num, :] = feat.detach().cpu().numpy()
            # query_feat_att[ptr:ptr + batch_num, :] = feat_att.detach().cpu().numpy()
            # query_attr[ptr:ptr + batch_num, :] = torch.stack([out.max(1)[1] for out in attr_score]).t().cpu().numpy()
            ptr = ptr + batch_num
    print('Extracting Time:\t {:.3f}'.format(time.time() - start))

    start = time.time()
    # compute the similarity
    distmat = np.matmul(query_feat, np.transpose(gall_feat))
    # distmat_att = np.matmul(query_feat_att, np.transpose(gall_feat_att))

    # evaluation
    if dataset == 'regdb':
        cmc, mAP, mINP      = eval_regdb(-distmat, query_label, gall_label)
        # cmc_att, mAP_att, mINP_att  = eval_regdb(-distmat_att, query_label, gall_label)
    elif dataset == 'sysu':
        cmc, mAP, mINP = eval_sysu(-distmat, query_label, gall_label, query_cam, gall_cam)
        # cmc_att, mAP_att, mINP_att = eval_sysu(-distmat_att, query_label, gall_label, query_cam, gall_cam)
    print('Evaluation Time:\t {:.3f}'.format(time.time() - start))

    writer.add_scalar('rank1', cmc[0], epoch)
    writer.add_scalar('mAP', mAP, epoch)
    writer.add_scalar('mINP', mINP, epoch)
    # writer.add_scalar('rank1_att', cmc_att[0], epoch)
    # writer.add_scalar('mAP_att', mAP_att, epoch)
    # writer.add_scalar('mINP_att', mINP_att, epoch)
    return cmc, mAP, mINP


# training
print('==> Start Training...')
for epoch in range(start_epoch, 180 ):

    print('==> Preparing Data Loader...')
    # identity sampler
    sampler = IdentitySampler(trainset.train_color_label, \
                              trainset.train_thermal_label, color_pos, thermal_pos, args.num_pos, args.batch_size,
                              epoch)

    trainset.cIndex = sampler.index1  # color index
    trainset.tIndex = sampler.index2  # thermal index
    print(epoch)
    print(trainset.cIndex)
    print(trainset.tIndex)

    loader_batch = args.batch_size * args.num_pos

    trainloader = data.DataLoader(trainset, batch_size=loader_batch, \
                                  sampler=sampler, num_workers=args.workers, drop_last=True)

    # training
    train(epoch)

    if epoch >= 0 and epoch % 2 == 0:
        print('Test Epoch: {}'.format(epoch))

        # testing
        cmc, mAP, mINP = test(epoch)
        # save model
        if cmc[0] > best_acc:  # not the real best for sysu-mm01
            best_acc = cmc[0]
            best_epoch = epoch
            state = {
                'net': net.state_dict(),
                'vit' : vit.state_dict(),
                'cmc': cmc,
                'mAP': mAP,
                'mINP': mINP,
                'epoch': epoch,
            }
            torch.save(state, checkpoint_path + suffix + '_best.t')

        # # save model
        if dataset == 'sysu' and epoch % args.save_epoch == 0:
            state = {
                'net': net.state_dict(),
                'vit': vit.state_dict(),
                'cmc': cmc,
                'mAP': mAP,
                'epoch': epoch,
            }
            torch.save(state, checkpoint_path + suffix + '_epoch_{}.t'.format(epoch))

        print('POOL:   Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}| mINP: {:.2%}'.format(
            cmc[0], cmc[4], cmc[9], cmc[19], mAP, mINP))
        # print('FC:   Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}| mINP: {:.2%}'.format(
        #     cmc_att[0], cmc_att[4], cmc_att[9], cmc_att[19], mAP_att, mINP_att))
        print('Best Epoch [{}]'.format(best_epoch))
