# TODO: This file should be deprecated, it's just for testing.
from openhgnn.models.MAGNN import MAGNN, mp_instance_sampler, mini_mp_instance_sampler
from openhgnn.sampler.MAGNN_sampler import MAGNN_sampler, collate_fn
from openhgnn.sampler.test_config import CONFIG
from operator import itemgetter
import argparse
import warnings
import dgl
import torch as th
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# def mini_train(model, hg, args):
#     sampler = MAGNN_sampler(hg, args.num_layers, model.metapath_list, args.dataset)
#     mp_instances = model.metapath_idx_dict
#
#     category = list(hg.ndata['labels'].keys())[0]
#     train_mask = hg.nodes[category].data['train_mask']
#     test_mask = hg.nodes[category].data['test_mask']
#     valid_mask = hg.nodes[category].data['val_mask']
#     train_nids = {category: hg.nodes(category)[train_mask]}
#     loader = dgl.dataloading.NodeDataLoader(
#         g=hg.to('cpu'), nids=train_nids, block_sampler=sampler, batch_size=args.batch_size, shuffle=True, drop_last=False,
#         num_workers=0, device=args.device)
#
#     model.train()
#     loss_all = 0
#     for i, (input_nodes, seeds, blocks) in enumerate(loader):
#         blocks = [blk.to(args.device) for blk in blocks]
#         lbl = blocks[-1].dstnodes[category].data['labels']
#         feat = blocks[0].srcdata['feat']
#         if args.num_layers == 1:
#             seed_nodes = {category: seeds[category]}
#         else:
#             seed_nodes = blocks[1].srcdata[dgl.NID]
#         mini_mp_instances = mini_mp_instance_sampler(seed_nodes=seed_nodes, mp_instances=mp_instances,
#                                                      block=blocks[0])
#         # TODO: Preprocess
#         model.metapath_idx_dict = mini_mp_instances
#         model.metapath_list = list(mini_mp_instances.keys())
#         model.dst_ntypes = [meta[0] for meta in model.metapath_list]
#         for layer in model.layers:
#             layer.metapath_list = model.metapath_list
#             layer.dst_ntypes = model.dst_ntypes
#
#         # TODO: Preprocess IS IT RIGHT???
#         logits = model(blocks, feat)[category]
#         loss = F.cross_entropy(logits, lbl)
#         loss_all += loss.item()
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#     return loss_all

# def mini_train(model, hg, args):

def load_hg(args):
    hg_dir = 'openhgnn/dataset/'
    hg,_ = dgl.load_graphs(hg_dir+'{}/graph.bin'.format(args.dataset), [0])
    hg = hg[0]
    return hg

if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    args = argparse.Namespace(**CONFIG)
    hg = load_hg(args)
    model = MAGNN.build_model_from_args(args, hg)
    sampler = MAGNN_sampler(g=hg, n_layers=args.num_layers, category=args.category, metapath_list=model.metapath_list)
    dataloader = DataLoader(dataset=sampler, batch_size=args.batch_size, shuffle=True, num_workers=0,
                            collate_fn=collate_fn, drop_last=False)

    optimizer = th.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # TODO: Add valuate and test step
    # TODO: just test if the whole pipeline can work without testing the effectiveness
    batch_idx = 0
    for epoch in range(args.max_epoch):
        for sub_g, mini_mp_inst in dataloader:
            model.mp_instances = mini_mp_inst
            pred = model(sub_g)[args.category]
            loss = F.cross_entropy(pred, sub_g.nodes[args.category].data['labels'])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print("batch_idx:{}, the batch_size is {}, the loss of this batch is {}".format(
                batch_idx, args.batch_size, loss.item()
            ))
            batch_idx += 1

    # next(iter(dataloader))
    print(1)




