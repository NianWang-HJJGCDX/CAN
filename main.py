import torch
import argparse
import torch.nn as nn
import torch.utils.data as Data
import seaborn as sns
from torch.autograd import Variable
import imageio
from sklearn.metrics import confusion_matrix
import scipy.io as sio
import math
from sklearn.decomposition import PCA
import numpy as np
import time
import os
from CAN import CAN
import warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser("HSI")
parser.add_argument('--dataset', choices=['Indian', 'Salinas', 'PaviaU','Houston2013','Houston2018'],
                    default='Indian', help='dataset to use')
parser.add_argument('--gpu_id', default='0', help='gpu id')
parser.add_argument("--outputs", type=str, default="./results")
parser.add_argument("--loss_record", type=str, default="./loss_record")
parser.add_argument("--num_run", type=int, default=1, help='it can be set as an integer value larger than 1 to show the '
                                                           'average  and standard deviation of multiple runs')
parser.add_argument('--seed', type=int, default=0, help='number of seed')
parser.add_argument('--batch_size', type=int, default=128, help='number of batch size')
parser.add_argument('--test_freq', type=int, default=5, help='number of evaluation')
parser.add_argument('--patches', type=int, default=7, help='number of patches')
parser.add_argument('--epochs', type=int, default=300, help='epoch number')
parser.add_argument('--learning_rate', type=float, default=5e-4, help='learning rate')
parser.add_argument('--gamma', type=float, default=0.9, help='gamma')
parser.add_argument('--weight_decay', type=float, default=0, help='weight_decay')
parser.add_argument('--band_decay', type=float, default=0.15, help='drop redundant bands by PCA')
parser.add_argument('--P_size', type=float, default=32, help='dimensionality of primary activity vector')
parser.add_argument('--C_size', type=float, default=16, help='dimensionality of class activity vector')
args = parser.parse_args()

def load_dataset(Dataset):
    if Dataset == 'Indian':
        mat_data = sio.loadmat('./datasets/Indian_pines_corrected.mat')
        mat_gt = sio.loadmat('./datasets/Indian_pines_gt.mat')
        data_hsi = mat_data['indian_pines_corrected']
        gt_hsi = mat_gt['indian_pines_gt']
        TOTAL_SIZE = 10249
        VALIDATION_SPLIT = 0.97
        TRAIN_SIZE = math.ceil(TOTAL_SIZE * VALIDATION_SPLIT)
    if Dataset == 'Salinas':
        SV = sio.loadmat('./datasets//Salinas_corrected.mat')
        gt_SV = sio.loadmat('./datasets/Salinas_gt.mat')
        data_hsi = SV['salinas_corrected']
        gt_hsi = gt_SV['salinas_gt']
        TOTAL_SIZE = 54129
        VALIDATION_SPLIT = 0.995
        TRAIN_SIZE = math.ceil(TOTAL_SIZE * VALIDATION_SPLIT)
    if Dataset == 'PaviaU':
        uPavia = sio.loadmat('./datasets//PaviaU.mat')
        gt_uPavia = sio.loadmat('./datasets//PaviaU_gt.mat')
        data_hsi = uPavia['paviaU']
        gt_hsi = gt_uPavia['paviaU_gt']
        TOTAL_SIZE = 42776
        VALIDATION_SPLIT = 0.995
        TRAIN_SIZE = math.ceil(TOTAL_SIZE * VALIDATION_SPLIT)
    if Dataset == 'Houston2013':
        uPavia = sio.loadmat('./datasets//Houston2013.mat')
        gt_uPavia = sio.loadmat('./datasets/Houston2013_gt.mat')
        data_hsi = uPavia['Houston2013']
        gt_hsi = gt_uPavia['Houston2013_gt']
        TOTAL_SIZE = 15029
        VALIDATION_SPLIT = 0.97
        TRAIN_SIZE = math.ceil(TOTAL_SIZE * VALIDATION_SPLIT)
    if Dataset == 'Houston2018':
        uPavia = sio.loadmat('./datasets//Houston2018.mat')
        gt_uPavia = sio.loadmat('./datasets/Houston2018_gt.mat')
        data_hsi = uPavia['Houston2018']
        gt_hsi = gt_uPavia['Houston2018_gt']
        TOTAL_SIZE = 504856
        VALIDATION_SPLIT = 0.97
        TRAIN_SIZE = math.ceil(TOTAL_SIZE * VALIDATION_SPLIT)

    return data_hsi, gt_hsi, TOTAL_SIZE, TRAIN_SIZE, VALIDATION_SPLIT

def train_and_test_data(pca_image, band, train_point, test_point, true_point, patch, w, h, n_gcn):
    x_train = np.zeros((train_point.shape[0], patch, patch, band), dtype=float)
    x_test = np.zeros((test_point.shape[0], patch, patch, band), dtype=float)
    x_true = np.zeros((true_point.shape[0], patch, patch, band), dtype=float)
    corner_train = np.zeros((train_point.shape[0], 2), dtype=int)
    corner_test = np.zeros((test_point.shape[0], 2), dtype=int)
    corner_true = np.zeros((true_point.shape[0], 2), dtype=int)
    indexs_train = torch.zeros((train_point.shape[0], n_gcn),dtype=int).cuda()
    indexs_test = torch.zeros((test_point.shape[0], n_gcn),dtype=int).cuda()
    indexs_ture = torch.zeros((true_point.shape[0], n_gcn),dtype=int).cuda()
    for i in range(train_point.shape[0]):
        x_train[i,:,:,:],corner_train[i,:],indexs_train[i]= gain_neighborhood_pixel(pca_image, train_point, i, patch, w, h,n_gcn)
    for j in range(test_point.shape[0]):
        x_test[j,:,:,:],corner_test[j,:],indexs_test[j] = gain_neighborhood_pixel(pca_image, test_point, j, patch, w, h,n_gcn)
    for k in range(true_point.shape[0]):
        x_true[k,:,:,:],corner_true[k,:],indexs_ture[k] = gain_neighborhood_pixel(pca_image, true_point, k, patch, w, h,n_gcn)
    print("x_train shape = {}, type = {}".format(x_train.shape,x_train.dtype))
    print("x_test  shape = {}, type = {}".format(x_test.shape,x_test.dtype))
    print("x_true  shape = {}, type = {}".format(x_true.shape,x_test.dtype))
    print("**************************************************")
    return x_train, x_test, x_true,corner_train,corner_test,corner_true,indexs_train,indexs_test,indexs_ture

def sampling(proportion, ground_truth,CLASSES_NUM):
    train = {}
    test = {}
    train_num = []
    test_num = []
    labels_loc = {}
    for i in range(CLASSES_NUM):
        indexes = np.argwhere(ground_truth == (i + 1))
        np.random.shuffle(indexes)#打乱顺序
        labels_loc[i] = indexes
        if proportion != 1:
            if indexes.shape[0]<=60:
                nb_val = 15
            else:
                nb_val = 30
        else:
            nb_val = 0
        train_num.append(nb_val)
        test_num.append(len(indexes)-nb_val)
        train[i] = indexes[:nb_val]
        test[i] = indexes[nb_val:]
    train_indexes =train[0]
    test_indexes = test[0]
    for i in range(CLASSES_NUM-1):
        train_indexes= np.concatenate((train_indexes,train[i+1]),axis=0)
        test_indexes= np.concatenate((test_indexes,test[i+1]),axis=0)
    np.random.shuffle(train_indexes)
    np.random.shuffle(test_indexes)
    return train_indexes, test_indexes,train_num,test_num

def get_label(indices,gt_hsi):
    dim_0 = indices[:, 0]
    dim_1 = indices[:, 1]
    label=gt_hsi[dim_0,dim_1]
    return label

def get_data(dataset):
    data_hsi, gt_hsi, TOTAL_SIZE, TRAIN_SIZE,VALIDATION_SPLIT = load_dataset(dataset)
    gt = gt_hsi.reshape(np.prod(gt_hsi.shape[:2]),)
    CLASSES_NUM = max(gt)
    train_indices, test_indices,train_num,test_num = sampling(VALIDATION_SPLIT, gt_hsi, CLASSES_NUM)
    _, total_indices,_,total_num = sampling(1, gt_hsi , CLASSES_NUM)
    y_train = get_label(train_indices, gt_hsi)-1
    y_test = get_label(test_indices, gt_hsi)-1
    y_true = get_label(total_indices, gt_hsi)-1
    return  data_hsi,CLASSES_NUM,train_indices,test_indices,total_indices,y_train, y_test, y_true,gt_hsi

def to_one_hot(x, length):
    """将类别标签转换为独热码，即标签向量变为标签矩阵"""
    batch_size = x.size(0)
    x_one_hot = torch.zeros(batch_size, length)
    for i in range(batch_size):
        x_one_hot[i, x[i]] = 1.0
    return x_one_hot

def applyPCA(X, numComponents):
    newX = np.reshape(X, (-1, X.shape[2]))
    pca = PCA(n_components=numComponents, whiten=True)
    newX = pca.fit_transform(newX)
    newX = np.reshape(newX, (X.shape[0], X.shape[1], numComponents))
    return newX

def normalize(input):
    input_normalize = np.zeros(input.shape)
    for i in range(input.shape[2]):
        input_max = np.max(input[:,:,i])
        input_min = np.min(input[:,:,i])
        input_normalize[:,:,i] = (input[:,:,i]-input_min)/(input_max-input_min)
    return input_normalize

def mirror_hsi(height, width, band, input_normalize, patch=5):
    """边界拓展：镜像。在获取高光谱Cube时，方便获取边缘中心点的邻域"""
    padding = patch // 2
    mirror_hsi = np.zeros((height + 2 * padding, width + 2 * padding, band), dtype=float)
    # 中心区域
    mirror_hsi[padding:(padding + height), padding:(padding + width), :] = input_normalize
    # 左边镜像
    for i in range(padding):
        mirror_hsi[padding:(height + padding), i, :] = input_normalize[:, padding - i - 1, :]
    # 右边镜像
    for i in range(padding):
        mirror_hsi[padding:(height + padding), width + padding + i, :] = input_normalize[:, width - 1 - i, :]
    # 上边镜像
    for i in range(padding):
        mirror_hsi[i, :, :] = mirror_hsi[padding * 2 - i - 1, :, :]
    # 下边镜像
    for i in range(padding):
        mirror_hsi[height + padding + i, :, :] = mirror_hsi[height + padding - 1 - i, :, :]
    print("**************************************************")
    print("patch is : {}".format(patch))
    print("mirror_image shape : [{0},{1},{2}]".format(mirror_hsi.shape[0], mirror_hsi.shape[1], mirror_hsi.shape[2]))
    print("**************************************************")
    return mirror_hsi

def choose_top(image,cornor_index,x,y,patch,b,n_top):
    sort = image.reshape(patch * patch, b)
    sort = torch.from_numpy(sort).type(torch.FloatTensor)
    pos = (x - cornor_index[0]) * patch + (y - cornor_index[1])
    Q = torch.sum(torch.pow(sort[pos] - sort, 2), dim=1)
    _, indices = Q.topk(k=n_top, dim=0, largest=False, sorted=True)
    return indices

def gain_neighborhood_pixel(pca_image, point, i, patch, W, H,n_gcn):
    x = point[i,0]
    y = point[i,1]
    m=int((patch-1)/2)
    _,_,b=pca_image.shape
    if x<=m:
        if y<=m:
            temp_image = pca_image[0:patch, 0:patch, :]
            cornor_index = [0,0]
        if y>=(H-m):
            temp_image = pca_image[0:patch, H-patch:H, :]
            cornor_index = [0, H-patch]
        if y>m and y<H-m:
            temp_image = pca_image[0:patch, y-m:y+m+1, :]
            cornor_index = [0, y-m]
    if x>=(W-m):
        if y<=m:
            temp_image = pca_image[W-patch:W, 0:patch, :]
            cornor_index = [W-patch, 0]
        if y>=(H-m):
            temp_image = pca_image[W-patch:W, H-patch:H, :]
            cornor_index = [W - patch, H-patch]
        if y>m and y<H-m:
            temp_image = pca_image[W-patch:W, y-m:y+m+1, :]
            cornor_index = [W - patch, y-m]
    if x>m and x<W-m:
        if y<=m:
            temp_image = pca_image[x-m:x+m+1, 0:patch, :]
            cornor_index = [x-m, 0]
        if y>=(H-m):
            temp_image = pca_image[x-m:x+m+1, H-patch:H, :]
            cornor_index = [x - m, H-patch]
        if y>m and y<H-m:
            temp_image = pca_image[x-m:x+m+1, y-m:y+m+1, :]
            cornor_index = [x - m, y-m]
    index = choose_top(temp_image, cornor_index, x, y, patch, b, n_gcn)
    return temp_image,cornor_index,index

class AvgrageMeter(object):
    def __init__(self):
        self.reset()
    def reset(self):
        self.avg = 0
        self.sum = 0
        self.cnt = 0
    def update(self, val, n=1):
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt

def accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()  # .t()为转置
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res, target, pred.squeeze()

def train_epoch(model, train_loader, optimizer):
    objs = AvgrageMeter()
    tar = []
    pre = []
    for batch_idx, (batch_data, batch_target) in enumerate(train_loader):
        target_indices = batch_target
        target_one_hot = to_one_hot(target_indices, length=network.Classcaps.num_units)
        batch_data, batch_target = Variable(batch_data).cuda(), Variable(target_one_hot).cuda()
        optimizer.zero_grad()
        batch_pred = model(batch_data)  # [64,16]
        loss = network.loss(batch_data,batch_pred, batch_target)
        loss.backward()
        optimizer.step()
        v_mag = torch.sqrt((batch_pred ** 2).sum(dim=2, keepdim=True))
        batch_pred = v_mag.data.max(1, keepdim=True)[1].cpu()
        batch_pred = batch_pred.squeeze()
        n = batch_data.shape[0]
        objs.update(loss.data, n)
        tar = np.append(tar, target_indices.data.cpu().numpy())
        pre = np.append(pre, batch_pred.data.cpu().numpy())
    return objs.avg,tar, pre  # top1和objs包含[avg,cnt,sum]3个变量,tar所有像素标签，pre所有像素预测结果

def test_epoch(model, test_loader, optimizer):
    tar = []
    pre = []
    for batch_idx, (batch_data, batch_target) in enumerate(test_loader):
        target_indices = batch_target
        target_one_hot = to_one_hot(target_indices, length=network.Classcaps.num_units)
        batch_data, batch_target = Variable(batch_data).cuda(), Variable(target_one_hot).cuda()
        optimizer.zero_grad()
        batch_pred = model(batch_data)  # [64,16]
        loss = network.loss(batch_data, batch_pred, batch_target)
        loss.backward()
        optimizer.step()
        v_mag = torch.sqrt((batch_pred ** 2).sum(dim=2, keepdim=True))
        batch_pred = v_mag.data.max(1, keepdim=True)[1].cpu()
        batch_pred = batch_pred.squeeze()
        tar = np.append(tar, target_indices.data.cpu().numpy())
        pre = np.append(pre, batch_pred.data.cpu().numpy())
    return tar, pre  # top1和objs包含[avg,cnt,sum]3个变量,tar所有像素标签，pre所有像素预测结果

def output_metric(tar, pre):
    matrix = confusion_matrix(tar, pre)
    OA, AA_mean, Kappa, AA = cal_results(matrix)
    return OA, AA_mean, Kappa, AA

def color_results(arr2d, palette):
    arr_3d = np.zeros((arr2d.shape[0], arr2d.shape[1], 3), dtype=np.uint8)
    for c, i in palette.items():
        m = arr2d == c
        arr_3d[m] = i
    return arr_3d

def cal_results(matrix):
    shape = np.shape(matrix)
    number = 0
    sum = 0
    AA = np.zeros([shape[0]], dtype=float)
    for i in range(shape[0]):
        number += matrix[i, i]
        AA[i] = matrix[i, i] / np.sum(matrix[i, :])
        sum += np.sum(matrix[i, :]) * np.sum(matrix[:, i])
    OA = number / np.sum(matrix)
    AA_mean = np.mean(AA)
    pe = sum / (np.sum(matrix) ** 2)
    Kappa = (OA - pe) / (1 - pe)
    return OA, AA_mean, Kappa, AA

def metrics(best_OA2, best_AA_mean2, best_Kappa2,AA2):
    results = {}
    results["OA"] = best_OA2 * 100.0
    results['AA'] = best_AA_mean2 * 100.0
    results["Kappa"] = best_Kappa2 * 100.0
    results["class acc"] = AA2 * 100.0
    return results

def show_results(results, agregated=False):
    text = ""
    if agregated:
        accuracies = [r["OA"] for r in results]
        aa = [r['AA'] for r in results]
        kappas = [r["Kappa"] for r in results]
        class_acc = [r["class acc"] for r in results]
        class_acc_mean = np.mean(class_acc, axis=0)
        class_acc_std = np.std(class_acc, axis=0)
    else:
        accuracy = results["OA"]
        aa = results['AA']
        classacc = results["class acc"]
        kappa = results["Kappa"]
    text += "---\n"
    text += "class acc :\n"
    if agregated:
        for score, std in zip(class_acc_mean,class_acc_std):
            text += "\t{:.02f} +- {:.02f}\n".format(score, std)
    else:
        for score in classacc:
            text += "\t {:.02f}\n".format(score)
    text += "---\n"
    if agregated:
        text += ("OA: {:.02f} +- {:.02f}\n".format(np.mean(accuracies),np.std(accuracies)))
        text += ("AA: {:.02f} +- {:.02f}\n".format(np.mean(aa),np.std(aa)))
        text += ("Kappa: {:.02f} +- {:.02f}\n".format(np.mean(kappas),np.std(kappas)))
    else:
        text += "OA : {:.02f}%\n".format(accuracy)
        text += "AA: {:.02f}%\n".format(aa)
        text += "Kappa: {:.02f}\n".format(kappa)
    print(text)

#-------------------------------------------------------------------------------
## load all data
input, num_classes, total_pos_train, total_pos_test, total_pos_true, y_train, y_test, y_true ,label = get_data(args.dataset)
_, _, band = input.shape
input = applyPCA(input, numComponents=int(band*args.band_decay))
input_normalize = normalize(input)
height, width, band = input_normalize.shape
print("height={0},width={1},band={2}".format(height, width, band))

# obtain train and test data
x_train_band, x_test_band, x_true_band, corner_train, corner_test, corner_true, indexs_train, indexs_test, indexs_ture = train_and_test_data(
    input_normalize, band, total_pos_train, total_pos_test, total_pos_true, patch=args.patches, w=height, h=width,
    n_gcn=5)
x_train=torch.from_numpy(x_train_band.transpose(0,3,1,2)).type(torch.FloatTensor) #[695, 200, 7, 7]
y_train=torch.from_numpy(y_train).type(torch.LongTensor) #[695]
train=Data.TensorDataset(x_train,y_train) # TensorDataset: tensor合并，即train[0]为某个数据点，train[1]为对应的标签
x_test=torch.from_numpy(x_test_band.transpose(0,3,1,2)).type(torch.FloatTensor) # [9671, 200, 7, 7]
y_test=torch.from_numpy(y_test).type(torch.LongTensor) # [9671]
test=Data.TensorDataset(x_test,y_test)
train_loader=Data.DataLoader(train,batch_size=args.batch_size,shuffle=True)
test_loader=Data.DataLoader(test,batch_size=args.batch_size,shuffle=False)

#-------------------------------------------------------------------------------
results = []
# num_run：运行次数。为了快速获得结果，默认设置为1。论文中设置为5，记录指标的平均值和方差
for run in range(args.num_run):
    best_OA2 = 0.0
    best_AA_mean2 = 0.0
    best_Kappa2 = 0.0
    network = CAN(image_width=args.patches,
                             image_height=args.patches,
                             image_channels=band,
                             conv_inputs=band,
                             conv_outputs=128,
                             num_primary_units=8,
                             primary_unit_size=args.P_size,
                             num_output_units=num_classes,
                             output_unit_size=args.C_size).cuda()
    # 只运行一次代码时打印网络结构。多次运行时，为了观测多次运行的结果，默认选择不打印网络
    if args.num_run == 1:
        print(network)
    network = network.cuda()
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.Adam(network.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.epochs // 10, gamma=args.gamma)

    print("start training")
    tic = time.time()
    loss_list = []
    OA_list = []
    for epoch in range(args.epochs):
        scheduler.step()
        network.train()
        train_obj, tar_t, pre_t = train_epoch(network, train_loader, optimizer)
        OA1, AA_mean1, Kappa1, AA1 = output_metric(tar_t, pre_t)
        print("Epoch: {:03d} train_loss: {:.4f}".format(epoch + 1, train_obj))
        if args.num_run == 1:
            loss_record = '%i: %.4f' % (epoch+ 1,train_obj)
            loss_list.append(loss_record)
            outfile_loss = os.path.join(args.loss_record, args.dataset)
            os.makedirs(outfile_loss, exist_ok=True)
            file_name = os.path.join(outfile_loss, 'loss_record.txt')
            with open(file_name, 'wt') as loss_record:
                loss_record.write('------------ Epoch and Training_Loss -------------\n')
                lenght = len(loss_list)
                for i in range (lenght):
                    loss_record.write('%s\n' % loss_list[i])
                loss_record.write('-------------- End ----------------\n')
        # 为了加速训练，前一半的回合不测试结果，后一半回合，每5回合（test_freq）测试一次并记录结果
        if epoch % args.test_freq == 1 and epoch >= args.epochs * 0.5:
        # if epoch % args.epochs == 1: # test only once during the traning, you can use it for large-scale data like Houston2018
            network.eval()
            tar_v, pre_v = test_epoch(network, test_loader,  optimizer)
            OA2, AA_mean2, Kappa2, AA2 = output_metric(tar_v, pre_v)
            # 更新测试的最佳结果
            if args.num_run == 1:
                OA_record = '%i: %.4f' % (epoch+ 1,OA2)
                OA_list.append(OA_record)
                outfile_OA = os.path.join(args.loss_record, args.dataset)
                os.makedirs(outfile_OA, exist_ok=True)
                file_name = os.path.join(outfile_OA, 'OA_record.txt')
                with open(file_name, 'wt') as OA_record:
                    OA_record.write('------------ Epoch and OA -------------\n')
                    lenght = len(OA_list)
                    for i in range (lenght):
                        OA_record.write('%s\n' % OA_list[i])
                    OA_record.write('-------------- End ----------------\n')
            if OA2 >= best_OA2 and AA_mean2 >= best_AA_mean2 and Kappa2 >= best_Kappa2:
                best_OA2 = OA2
                best_AA_mean2 = AA_mean2
                best_Kappa2 = Kappa2
                run_results = metrics(best_OA2, best_AA_mean2, best_Kappa2,AA2)
                best_pre_v=pre_v
                best_tar_v = tar_v
    # 如果只运行代码一次，则直接输出指标；如果运行代码超过一次（本文汇报5次运行的结果），进行合并，输出多次运行的平均值和方法
    results.append(run_results)
    if args.num_run == 1:
        show_results(run_results, agregated=False)
    elif args.num_run > 1:
        show_results(results,agregated=True)
    else:
        raise ValueError("num_run should be positive integer")

    toc = time.time()
    # 为了可视化效果，为不同数据集选择了不同的调色盘（palette）
    palette = {0: (255, 255, 255)}
    if args.dataset == 'Indian':
        for k, color in enumerate(sns.color_palette("hls", 17)):
            palette[k + 1] = tuple(np.asarray(255 * np.array(color), dtype='uint8'))
    elif args.dataset == 'Salinas':
        for k, color in enumerate(sns.color_palette("Paired", 12)):
            palette[k + 1] = tuple(np.asarray(255 * np.array(color), dtype='uint8'))
        for k, color in enumerate(sns.color_palette("hls", 4)):
            palette[13 + k] = tuple(np.asarray(255 * np.array(color), dtype='uint8'))
    elif args.dataset == 'PaviaU':
        for k, color in enumerate(sns.color_palette("Paired", 12)):
            palette[k + 1] = tuple(np.asarray(255 * np.array(color), dtype='uint8'))
    elif args.dataset == 'Houston2013':
        for k, color in enumerate(sns.color_palette("hls", 17)):
            palette[k + 1] = tuple(np.asarray(255 * np.array(color), dtype='uint8'))
    elif args.dataset == 'Houston2018':
        palette = {1: (78, 170, 72), 2: (136, 186, 67), 3: (62, 132, 92), 4: (55, 131, 69), 5: (144, 82, 54),
                   6: (103, 188, 199), 7: (255, 255, 255), 8: (199, 176, 256), 9: (218, 49, 44), 10: (120, 35, 36),
                   11: (53, 101, 166), 12: (224, 220, 83), 13: (217, 142, 53), 14: (84, 48, 125),
                   15: (227, 119, 98), 16: (157, 86, 150), 17: (0, 201, 167), 18: (213, 202, 189),
                   19: (173, 197, 207), 20: (185, 61, 175)}
    else:
        raise ValueError("Wrong dataset")
    prediction=np.array(label)
    prediction[total_pos_test[:,0],total_pos_test[:,1]]= best_pre_v.astype(np.uint8)+1
    colored_gt = color_results(label, palette)
    colored_pred = color_results(prediction, palette)
    outfile = os.path.join(args.outputs, args.dataset)
    os.makedirs(outfile, exist_ok=True)
    # 储存真实和预测的可视化结果于路径：./results
    imageio.imsave(os.path.join(outfile, args.dataset + '_gt.png'), colored_gt)
    imageio.imsave(os.path.join(outfile, args.dataset+'_' + 'out.png'), colored_pred)

