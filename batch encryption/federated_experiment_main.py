import copy
import time
import torch
import torchvision.datasets as datasets
from torchvision import transforms
import numpy as np
from phe import paillier
import multiprocessing
from joblib import Parallel, delayed
from models import Net
from tqdm import tqdm
import argparse
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
from torch import nn
import encryption
import binary_float_decimal
N_JOBS = multiprocessing.cpu_count()
criterion = torch.nn.CrossEntropyLoss()
publickey, privatekey = paillier.generate_paillier_keypair(n_length=4096)
class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label

class Client():
    def __init__(self, args, dataset=None, idxs=None, w=None):
        self.args = args
        self.criterion = nn.CrossEntropyLoss()
        self.ldr_train = DataLoader(DatasetSplit(dataset, idxs), batch_size=self.args.local_bs, shuffle=True)
        self.model = Net().to(self.args.device)
        self.model.load_state_dict(w)
        # Paillier initialization
        if self.args.experiment == 'paillier' or self.args.experiment == 'batch':
            self.pub_key = publickey
            self.priv_key = privatekey

    def train(self):
        weight_old = copy.deepcopy(self.model.state_dict())
        net = copy.deepcopy(self.model)
        # train and update
        net.train()
        local_epoch_loss = []
        optimizer = torch.optim.SGD(net.parameters(), lr=self.args.lr, momentum=0.5)
        for iter in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                net.zero_grad()
                log_prob = net(images)
                loss = self.criterion(log_prob, labels)
                loss.backward()
                optimizer.step()
                batch_loss.append(loss.item())
            local_epoch_loss.append(sum(batch_loss) / len(batch_loss))
        weight_new = net.state_dict()
        update_w = {}
        if self.args.experiment == 'plain' or self.args.experiment == 'batch' or self.args.experiment == 'onlyConvert':
            for k in weight_new.keys():
                update_w[k] = weight_new[k] - weight_old[k]
        elif self.args.experiment == 'paillier':
            print('encrypting...')
            enc_start = time.time()
            for k in weight_new.keys():
                update_w[k] = weight_new[k] - weight_old[k]
                # flatten weight
                list_w = update_w[k].view(-1).cpu().tolist()  
                encry_list_w = Parallel(n_jobs=N_JOBS)(delayed(self.pub_key.encrypt)(num) for num in list_w)
                update_w[k] = encry_list_w
            enc_end = time.time()
            print('Encryption time:', enc_end - enc_start)
        else:
            raise NotImplementedError
        return update_w, sum(local_epoch_loss) / len(local_epoch_loss)

    def update(self, weight_glob):
        if self.args.experiment == 'plain':
            self.model.load_state_dict(weight_glob)
        elif self.args.experiment == 'paillier':
            # for paillier, w_glob is update_w_avg here
            update_w_avg = copy.deepcopy(weight_glob)
            print('decrypting...')
            dec_start = time.time()
            for k in update_w_avg.keys():
                # decryption
                update_w_avg[k] = Parallel(n_jobs=N_JOBS)(delayed(self.priv_key.decrypt)(num) for num in update_w_avg[k])
                # reshape to original and update
                origin_shape = list(self.model.state_dict()[k].size())
                update_w_avg[k] = torch.FloatTensor(update_w_avg[k]).to(self.args.device).view(*origin_shape)
                self.model.state_dict()[k] += update_w_avg[k]
            dec_end = time.time()
            print('Decryption time:', dec_end - dec_start)
        elif self.args.experiment == 'batch':
            update_w_avg = copy.deepcopy(self.model.state_dict())
            for k in update_w_avg.keys():
                update_w_avg[k] = torch.FloatTensor(weight_glob[k]).to(self.args.device)
                self.model.state_dict()[k] += update_w_avg[k]
        elif self.args.experiment == 'onlyConvert':
            update_w_avg = copy.deepcopy(self.model.state_dict())
            for k in update_w_avg.keys():
                update_w_avg[k] = torch.FloatTensor(weight_glob[k]).to(self.args.device)
                self.model.state_dict()[k] += update_w_avg[k]
        else:
            raise NotImplementedError


class Server():
    def __init__(self, args, w):
        self.args = args
        self.clients_update_w = []
        self.clients_loss = []
        self.model = Net().to(self.args.device)
        self.model.load_state_dict(w)

    def FedAvg(self):
        if self.args.experiment == 'plain':
            update_w_avg = copy.deepcopy(self.clients_update_w[0])
            for k in update_w_avg.keys():
                for i in range(1, len(self.clients_update_w)):
                    update_w_avg[k] += self.clients_update_w[i][k]  # update server's weight
                update_w_avg[k] = torch.div(update_w_avg[k], len(self.clients_update_w))
                self.model.state_dict()[k] += update_w_avg[k]
            return copy.deepcopy(self.model.state_dict()), sum(self.clients_loss) / len(self.clients_loss)
        elif self.args.experiment == 'paillier':
            update_w_avg = copy.deepcopy(self.clients_update_w[0])
            for k in update_w_avg.keys():
                client_num = len(self.clients_update_w)
                for i in range(1, client_num):  # client-wise sum
                    for j in range(len(update_w_avg[k])):  # element-wise sum
                        update_w_avg[k][j] += self.clients_update_w[i][k][j]
                for j in range(len(update_w_avg[k])):  # element-wise avg
                    update_w_avg[k][j] /= client_num
            return update_w_avg, sum(self.clients_loss) / len(self.clients_loss)
        elif self.args.experiment == 'batch':  
            update_w_avg: dict = copy.deepcopy(self.clients_update_w[0])  
            for k in update_w_avg.keys():
                client_num = len(self.clients_update_w)
                for i in range(1, client_num):
                    for j in range(len(update_w_avg[k])):
                        update_w_avg[k][j] += self.clients_update_w[i][k][j]
            return update_w_avg, sum(self.clients_loss) / len(self.clients_loss)
        elif self.args.experiment == 'onlyConvert':
            update_w_avg: dict = copy.deepcopy(self.clients_update_w[0])
            for k in update_w_avg.keys():
                client_num = len(self.clients_update_w)
                for i in range(1, client_num):
                    for j in range(len(update_w_avg[k])):
                        update_w_avg[k][j] = str(int(update_w_avg[k][j]) + int(self.clients_update_w[i][k][j]))
            return update_w_avg, sum(self.clients_loss) / len(self.clients_loss)
        else:
            raise NotImplementedError

    def test(self, datatest):
        self.model.eval()
        # testing
        test_loss = 0
        correct = 0
        data_loader = DataLoader(datatest, batch_size=self.args.bs)
        for idx, (data, target) in enumerate(data_loader):
            if self.args.gpu != -1:
                data, target = data.cuda(), target.cuda()
            log_probs = self.model(data)

            # sum up batch loss
            test_loss += F.cross_entropy(log_probs, target, reduction='sum').item()

            # get the index of the max log-probability
            y_pred = log_probs.data.max(1, keepdim=True)[1]
            correct += y_pred.eq(target.data.view_as(y_pred)).long().cpu().sum()

        test_loss /= len(data_loader.dataset)
        accuracy = 100.00 * correct / len(data_loader.dataset)
        return accuracy, test_loss


def load_dataset():
    data_dir = '/home/hjh/hepaper/data/cifar' # data_dir
    apply_transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

    dataset_train = datasets.CIFAR10(data_dir, train=True, download=True,
                                     transform=apply_transform)

    dataset_test = datasets.CIFAR10(data_dir, train=False, download=True,
                                    transform=apply_transform)
    return dataset_train, dataset_test


def create_client_server():
    num_items = int(len(dataset_train) / args.num_clients)
    clients, all_idxs = [], [i for i in range(len(dataset_train))]
    net_glob = Net().to(args.device)

    # divide training data, i.i.d.
    # init models with same parameters
    for i in range(args.num_clients):
        new_idxs = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - new_idxs)
        new_client = Client(args=args, dataset=dataset_train, idxs=new_idxs, w=copy.deepcopy(net_glob.state_dict()))
        clients.append(new_client)

    server = Server(args=args, w=copy.deepcopy(net_glob.state_dict()))

    return clients, server

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # global setting
    parser.add_argument('--experiment', type=str, default='paillier',
                        choices=['plain', 'paillier', 'batch', 'onlyConvert']) # choose experiment mode
    parser.add_argument('--num_clients', type=int, default=10)   # choose the number of clients
    parser.add_argument('--num_epochs', type=int, default=1)     # choose the global epoch
    parser.add_argument('--batch_size', type=int, default=10)    # choose the encryption batch size
    parser.add_argument('--dataset', type=str, default='cifar10', help="name of dataset") # choose the dataset
    parser.add_argument('--gpu', type=int, default=0, help="GPU ID, -1 for CPU")  # choose gpu
    # local setting
    parser.add_argument('--lr', type=float, default=0.015, help='learning rate')
    parser.add_argument('--local_ep', type=int, default=10, help="the number of local epochs: E")
    parser.add_argument('--local_bs', type=int, default=64, help="local batch size: B")
    parser.add_argument('--bs', type=int, default=64, help="test batch size")
    args = parser.parse_args()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    print(args)

    print('----------------------------initialization-------------------------------')
    print('load dataset for {} client'.format(args.num_clients))
    dataset_train, dataset_test = load_dataset()
    print("clients and server initialization...")
    clients, server = create_client_server()
    num_epochs = args.num_epochs
    ## build clients -->num_clients
    num_clients = args.num_clients
    # statistics for plot
    all_acc_train = []
    all_acc_test = []
    all_loss_glob = []
    print('start training...')
    print('Algorithm:', args.experiment)
    if args.experiment == 'plain':
        test_epochs = 2
        for epoch in tqdm(range(num_epochs)):  ##global epoch
            # print(f'\n | Global Training Round : {epoch + 1} |\n')
            epoch_start = time.time()
            server.clients_update_w, server.clients_loss = [], []
            for idx in range(num_clients):
                update_w, loss = clients[idx].train()
                server.clients_update_w.append(update_w)
                server.clients_loss.append(loss)
            
            w_glob, loss_glob = server.FedAvg()
            # update local weights
            for idx in range(args.num_clients):
                clients[idx].update(w_glob)
            epoch_end = time.time()
            print('=====Global Epoch {:3d}====='.format(epoch + 1))
            print('Training time:', epoch_end - epoch_start)
            # testing
            acc_train, loss_train = server.test(dataset_train)
            acc_test, loss_test = server.test(dataset_test)
            print("Training accuracy: {:.2f}".format(acc_train))
            print("Testing accuracy: {:.2f}".format(acc_test))
            print('Training average loss {:.3f}'.format(loss_glob))
            all_acc_train.append(float(acc_train))
            all_acc_test.append(float(acc_test))
            all_loss_glob.append(float(loss_glob))
        print('{}epochs training accuracy:{}'.format(num_epochs, all_acc_train))
        print('{}epochs testing accuracy:{}'.format(num_epochs, all_acc_test))

    elif args.experiment == 'paillier':
        for epoch in tqdm(range(num_epochs)):  ##global epoch
            # print(f'\n | Global Training Round : {epoch + 1} |\n')
            epoch_start = time.time()
            server.clients_update_w, server.clients_loss = [], []
            for idx in range(num_clients):
                update_w, loss = clients[idx].train()
                server.clients_update_w.append(update_w)
                server.clients_loss.append(loss)
            w_glob, loss_glob = server.FedAvg()
            # update local weights
            for idx in range(args.num_clients):
                clients[idx].update(w_glob)
            epoch_end = time.time()
            print('=====Global Epoch {:3d}====='.format(epoch + 1))
            print('Training time:', epoch_end - epoch_start)
            # testing
            server.model.load_state_dict(copy.deepcopy(clients[0].model.state_dict()))
            acc_train, loss_train = server.test(dataset_train)
            acc_test, loss_test = server.test(dataset_test)
            print("Training accuracy: {:.2f}".format(acc_train))
            print("Testing accuracy: {:.2f}".format(acc_test))
            print('Training average loss {:.3f}'.format(loss_glob))
            all_acc_train.append(float(acc_train))
            all_acc_test.append(float(acc_test))
            all_loss_glob.append(float(loss_glob))
        print('{}epochs training accuracy:{}'.format(num_epochs, all_acc_train))
        print('{}epochs testing accuracy:{}'.format(num_epochs, all_acc_test))
    elif args.experiment == 'batch':
        test_epochs = 10
        for epoch in tqdm(range(num_epochs)):  ##global epoch
            epoch_start = time.time()
            theta = 2.5
            server.clients_update_w, server.clients_loss = [], []
            clients_weight_after_train = []
            for idx in range(num_clients):
                update_w, loss = clients[idx].train()
                clients_weight_after_train.append(update_w)
                server.clients_loss.append(loss)
            clients_layer_max = []
            for client_idx in range(len(clients_weight_after_train)):
                temp_max = {}
                for k in clients_weight_after_train[client_idx].keys():
                    temp_max[k] = torch.max(clients_weight_after_train[client_idx][k])
                clients_layer_max.append(temp_max)
            clipping_thresholds = {}
            for k in clients_layer_max[0].keys():
                temp1 = []
                for idx_client in range(len(clients_layer_max)):
                    temp1.append(clients_layer_max[idx_client][k])
                clipping_thresholds[k] = max(temp1)
            print('clipping_thresholds:', clipping_thresholds)
            # clipping with threshold
            for client_idx in range(len(clients_weight_after_train)):
                for k in clients_weight_after_train[client_idx].keys():
                    clients_weight_after_train[client_idx][k] = torch.clamp(clients_weight_after_train[client_idx][k],
                                                                            -1 * clipping_thresholds[k],
                                                                            clipping_thresholds[k])
            # adding threshold
            for client_idx in range(len(clients_weight_after_train)):
                for k in clients_weight_after_train[client_idx].keys():
                    clients_weight_after_train[client_idx][k] += clipping_thresholds[k]
            integerPart = {}
            floatPart = {}
            M_main = {}
            K_main = 4
            J = 9999
            N_main = {}
            for k in clipping_thresholds.keys():
                integerPart[k] = int(clipping_thresholds[k] * 2)
                M_main[k] = binary_float_decimal.total_bits(integerPart[k] * num_clients)
                N_main[k] = binary_float_decimal.total_bits(J * num_clients)
            print('three parameters :', M_main, K_main, N_main)
            enc_grads_batch_clients = []
            og_shape_batch_clients = []
            for item in clients_weight_after_train:
                enc_grads_temp, og_shape_temp = encryption.batch_encrypt_per_layer(publickey=publickey, party=item,
                                                                                   batch_size=args.batch_size, M=M_main,
                                                                                   K=K_main, N=N_main)
                enc_grads_batch_clients.append(enc_grads_temp)
                og_shape_batch_clients.append(og_shape_temp)
            server.clients_update_w = enc_grads_batch_clients
            w_glob, loss_glob = server.FedAvg()
            wg_de_batch: dict = encryption.batch_decrypt_per_layer(privatekey=privatekey, party=w_glob,
                                                                   og_shap=og_shape_batch_clients[0],
                                                                   batch_size=args.batch_size, M=M_main, K=K_main,
                                                                   N=N_main)
            for k in wg_de_batch.keys():
                wg_de_batch[k] = 1 / num_clients * (wg_de_batch[k] - clipping_thresholds[k].cpu().numpy() * num_clients)
            for idx in range(args.num_clients):
                clients[idx].update(wg_de_batch)
            epoch_end = time.time()
            print('=====Global Epoch {:3d}====='.format(epoch + 1))
            print('Training time:', epoch_end - epoch_start)
            # testing
            server.model.load_state_dict(copy.deepcopy(clients[0].model.state_dict()))
            acc_train, loss_train = server.test(dataset_train)
            acc_test, loss_test = server.test(dataset_test)
            print("Training accuracy: {:.2f}".format(acc_train))
            print("Testing accuracy: {:.2f}".format(acc_test))
            print('Training average loss {:.3f}'.format(loss_glob))
            all_acc_train.append(float(acc_train))
            all_acc_test.append(float(acc_test))
            all_loss_glob.append(float(loss_glob))
        print('{}epochs training accuracy:{}'.format(num_epochs, all_acc_train))
        print('{}epochs testing accuracy:{}'.format(num_epochs, all_acc_test))
    elif args.experiment == 'onlyConvert':
        test_epochs = 10
        for epoch in tqdm(range(num_epochs)):  ##global epoch
            # print(f'\n | Global Training Round : {epoch + 1} |\n')
            epoch_start = time.time()
            theta = 2.5
            server.clients_update_w, server.clients_loss = [], []
            clients_weight_after_train = []
            for idx in range(num_clients):
                update_w, loss = clients[idx].train()
                clients_weight_after_train.append(update_w)
                server.clients_loss.append(loss)
            clients_layer_max = []
            for client_idx in range(len(clients_weight_after_train)):
                temp_max = {}
                for k in clients_weight_after_train[client_idx].keys():
                    temp_max[k] = torch.max(clients_weight_after_train[client_idx][k])
                clients_layer_max.append(temp_max)
            clipping_thresholds_max = {}
            for k in clients_layer_max[0].keys():
                temp1 = []
                for idx_client in range(len(clients_layer_max)):
                    temp1.append(clients_layer_max[idx_client][k])
                clipping_thresholds_max[k] = max(temp1)
            # print('clipping_thresholds_max:', clipping_thresholds_max)
            clients_layer_min = []
            for client_idx in range(len(clients_weight_after_train)):
                temp_min = {}
                for k in clients_weight_after_train[client_idx].keys():
                    temp_min[k] = torch.min(clients_weight_after_train[client_idx][k])
                clients_layer_min.append(temp_min)
            clipping_thresholds_min = {}
            for k in clients_layer_min[0].keys():
                temp1 = []
                for idx_client in range(len(clients_layer_min)):
                    temp1.append(clients_layer_min[idx_client][k])
                clipping_thresholds_min[k] = min(temp1)
            # print('clipping_thresholds_min:', clipping_thresholds_min)
            clipping_thresholds = {}
            for k in clipping_thresholds_max.keys():
                clipping_thresholds[k] = max(clipping_thresholds_max[k],abs(clipping_thresholds_min[k]))
            print('clipping_thresholds:', clipping_thresholds)
            for client_idx in range(len(clients_weight_after_train)):
                for k in clients_weight_after_train[client_idx].keys():
                    clients_weight_after_train[client_idx][k] += clipping_thresholds[k]
            integerPart = {}
            floatPart = {}
            M_main = {}
            K_main = 6
            J = 999999
            N_main = {}
            for k in clipping_thresholds.keys():
                integerPart[k] = int(clipping_thresholds[k])
                floatPart[k] = clipping_thresholds[k] - integerPart[k]
                M_main[k] = binary_float_decimal.total_bits(2 * integerPart[k] * num_clients) + 3
                N_main[k] = binary_float_decimal.total_bits(2 * J * num_clients)
            print('three parameters :', M_main, K_main, N_main)
            # converting and batch_size
            grads_batch_clients = []
            shape_batch_clients = []
            for item in clients_weight_after_train:
                grad_temp, shape_temp = encryption.batch_convert_per_layer(party=item, batch_size=args.batch_size,
                                                                           M=M_main, K=K_main, N=N_main)
                grads_batch_clients.append(grad_temp)
                shape_batch_clients.append(shape_temp)
            server.clients_update_w = grads_batch_clients
            w_glob, loss_glob = server.FedAvg()
            wg_de_convert: dict = encryption.batch_de_convert_per_layer(party=w_glob, og_shape=shape_batch_clients[0],
                                                                        batch_size=args.batch_size, M=M_main, K=K_main,
                                                                        N=N_main)
            for k in wg_de_convert.keys():
                wg_de_convert[k] = 1 / num_clients * (
                        wg_de_convert[k] - clipping_thresholds[k].cpu().numpy() * num_clients)
            for idx in range(num_clients):
                clients[idx].update(wg_de_convert)
            epoch_end = time.time()
            print('=====Global Epoch {:3d}====='.format(epoch + 1))
            print('Training time:', epoch_end - epoch_start)
            # testing
            server.model.load_state_dict(copy.deepcopy(clients[0].model.state_dict()))
            acc_train, loss_train = server.test(dataset_train)
            acc_test, loss_test = server.test(dataset_test)
            print("Training accuracy: {:.2f}".format(acc_train))
            print("Testing accuracy: {:.2f}".format(acc_test))
            print('Training average loss {:.3f}'.format(loss_glob))
            all_acc_train.append(float(acc_train))
            all_acc_test.append(float(acc_test))
            all_loss_glob.append(float(loss_glob))
        print('{}epochs training accuracy:{}'.format(num_epochs, all_acc_train))
        print('{}epochs testing accuracy::{}'.format(num_epochs, all_acc_test))
    else:
        raise NotImplementedError