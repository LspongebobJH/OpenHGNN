import copy
import dgl
import numpy as np
import torch as th
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from . import BaseFlow, register_flow
from .node_classification import NodeClassification
from ..sampler.MAGNN_sampler import MAGNN_sampler, collate_fn
import time
th.multiprocessing.set_start_method('spawn')  # TODO: note this which should be modified

# TODO: Add svm test of MAGNN
@register_flow("magnntrainer")
class MAGNNTrainer(NodeClassification):
    def __init__(self, args):
        super(MAGNNTrainer, self).__init__(args)

    def preprocess(self):
        self.train_mask = self.hg.nodes[self.category].data['train_mask'].cpu().numpy()
        self.val_mask = self.hg.nodes[self.category].data['val_mask'].cpu().numpy()
        self.test_mask = self.hg.nodes[self.category].data['test_mask'].cpu().numpy()
        self.hg = self.hg.to('cpu')

    def set_sampler(self):

        self.sampler = MAGNN_sampler(g=self.hg, mask=self.train_mask, n_layers=self.args.num_layers,
                                     category=self.args.category,
                                     metapath_list=self.model.metapath_list, num_samples=self.args.num_samples,
                                     dataset_name=self.args.dataset)

        self.dataloader = DataLoader(dataset=self.sampler, batch_size=self.args.batch_size, shuffle=True,
                                     num_workers=self.args.num_workers,
                                     collate_fn=collate_fn, drop_last=False)

    def _mini_train_step(self):
        self.sampler.mask = self.train_mask
        t = time.perf_counter()
        self.model.train()
        print("...Start the mini batch training...")
        for num_iter, (sub_g, mini_mp_inst, seed_nodes) in enumerate(self.dataloader):
            print("Sampling {} seed_nodes with duration(s): {}".format(len(seed_nodes[self.args.category]),
                                                                       time.perf_counter() - t))
            self.model.mini_reset_params(mini_mp_inst)
            sub_g = sub_g.to(self.args.device)
            pred = self.model(sub_g)
            pred = pred[self.args.category][seed_nodes[self.args.category]]
            lbl = sub_g.nodes[self.args.category].data['labels'][seed_nodes[self.args.category]]
            loss = F.cross_entropy(pred, lbl)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            t = time.perf_counter()

            if num_iter == 0:  # TODO: test
                break

        return loss.item()

    def _mini_test_step(self, split=None, logits=None):
        self.model.eval()
        with th.no_grad():

            if split is None:
                mask = np.concatenate([self.train_mask, self.val_mask, self.test_mask], dim=0)
            elif split == 'train':
                mask = self.train_mask
            elif split == 'val':
                mask = self.val_mask
            elif split == 'test':
                mask = self.test_mask

            self.sampler.mask = mask
            test_loader = DataLoader(dataset=self.sampler, batch_size=self.args.batch_size, shuffle=False,
                                     num_workers=self.args.num_workers, collate_fn=collate_fn, drop_last=False)

            logp_test_all = []
            embed_test_all = []
            for num_iter, (sub_g, mini_mp_inst, seed_nodes) in enumerate(test_loader):
                sub_g = sub_g.to(self.args.device)
                self.model.mini_reset_params(mini_mp_inst)
                pred_test, embed_test = self.model(sub_g)
                pred_test = pred_test[self.args.category][seed_nodes[self.args.category]]
                embed_test = embed_test[self.args.category][seed_nodes[self.args.category]]
                logp_test = F.log_softmax(pred_test, 1)
                logp_test_all.append(logp_test)
                embed_test_all.append(embed_test.cpu().numpy())

            lbl_test = self.hg.nodes[self.args.category].data['labels'][mask]
            lbl_test = lbl_test.cuda()
            embed_test_all = np.concatenate(embed_test_all, 0)
            loss_test = F.nll_loss(th.cat(logp_test_all, 0), lbl_test)

            if split is not None:  # test specific nodes
                metric = self.task.evaluate(embed_test_all, name=self.eval_name, mask=mask)
                return metric, loss_test
            else:  # test all nodes
                train_mask, val_mask, test_mask = \
                    np.zeros_like(self.train_mask), np.zeros_like(self.val_mask), np.zeros_like(self.test_mask)
                train_mask[0:len(self.train_mask[self.train_mask == True])] = True
                train_last_idx = np.where(train_mask is True)[0][-1]
                val_mask[train_last_idx:
                         train_last_idx +
                         len(self.val_mask[self.val_mask is True])] = True
                val_last_idx = np.where(val_mask is True)[0][-1]
                test_mask[val_last_idx:
                          val_last_idx + len(self.test_mask[self.test_mask == True])] = True

                metrics = self.task.evaluate(embed_test_all, name=self.eval_name, mask=mask)
                _metrics = {'train': metrics[train_mask],
                            'val': metrics[val_mask],
                            'test': metrics[test_mask]}
                losses = {'train': loss_test[train_mask],
                          'val': loss_test[val_mask],
                          'test': loss_test[test_mask]}
                return metrics, losses
