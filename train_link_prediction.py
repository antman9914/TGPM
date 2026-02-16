import logging
import time
import sys
import os
import numpy as np
import warnings
import json
import torch
import torch.nn as nn

from models.modules import MergeLayer
from models.TGPM import PretrainModel
from utils.utils import set_random_seed, get_parameter_sizes, create_optimizer
from utils.utils import get_neighbor_sampler, NegativeEdgeSampler
from evaluate_models_utils import evaluate_model_link_prediction
from utils.metrics import get_link_prediction_metrics
from utils.DataLoader import get_idx_data_loader, get_link_prediction_data
from utils.EarlyStopping import EarlyStopping
from utils.load_configs import get_link_prediction_args
from utils.RandomWalk import get_sentences

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

if __name__ == "__main__":

    warnings.filterwarnings('ignore')

    # get arguments
    args = get_link_prediction_args()

    # get data for training, validation and testing
    datasets = args.dataset_name.strip().split(',')

    nfeat_list, efeat_list, time_list, train_data_list = [], [], [], []
    pattern_list, eid_list = [], []
    val_data, test_data, new_node_val_data, new_node_test_data = None, None, None, None
    train_sampler_list, train_loader_list = [], []
    for i in range(len(datasets)):
        node_raw_features, edge_raw_features, full_data, train_data, cur_val_data, cur_test_data, cur_new_node_val_data, cur_new_node_test_data = \
            get_link_prediction_data(dataset_name=datasets[i], val_ratio=args.val_ratio, test_ratio=args.test_ratio)
        
        patterns, eids = get_sentences(datasets[i], node_raw_features.shape[0], full_data.src_node_ids, full_data.dst_node_ids, full_data.node_interact_times, args.num_walk,
                                      args.walk_length, p=1.0, q=1.0, causal_path=args.causal_path)
        pattern_list.append(patterns)
        eid_list.append(eids)

        if i == args.eval_dataset:
            val_data = cur_val_data
            test_data = cur_test_data
            new_node_val_data = cur_new_node_val_data
            new_node_test_data = cur_new_node_test_data

        node_dim, edge_dim = node_raw_features.shape[-1], edge_raw_features.shape[-1]
        nfeat_list.append(node_raw_features)
        efeat_list.append(edge_raw_features)
        train_data_list.append(train_data)
        time_list.append(full_data.node_interact_times)

        # initialize training neighbor sampler to retrieve temporal graph
        train_neighbor_sampler = get_neighbor_sampler(data=train_data, sample_neighbor_strategy=args.sample_neighbor_strategy,
                                                    time_scaling_factor=args.time_scaling_factor, seed=0)
        train_idx_data_loader = get_idx_data_loader(indices_list=list(range(len(train_data.src_node_ids))), batch_size=args.batch_size, shuffle=False)
        train_sampler_list.append(train_neighbor_sampler)
        train_loader_list.append(train_idx_data_loader)

        if i == args.eval_dataset:
            cat_num = np.max(full_data.labels) + 1

            # initialize validation and test neighbor sampler to retrieve temporal graph
            full_neighbor_sampler = get_neighbor_sampler(data=full_data, sample_neighbor_strategy=args.sample_neighbor_strategy,
                                                        time_scaling_factor=args.time_scaling_factor, seed=1)

            # initialize negative samplers, set seeds for validation and testing so negatives are the same across different runs
            # in the inductive setting, negatives are sampled only amongst other new nodes
            # train negative edge sampler does not need to specify the seed, but evaluation samplers need to do so
            train_neg_edge_sampler = NegativeEdgeSampler(src_node_ids=train_data.src_node_ids, dst_node_ids=train_data.dst_node_ids)
            val_neg_edge_sampler = NegativeEdgeSampler(src_node_ids=full_data.src_node_ids, dst_node_ids=full_data.dst_node_ids, seed=0)
            new_node_val_neg_edge_sampler = NegativeEdgeSampler(src_node_ids=new_node_val_data.src_node_ids, dst_node_ids=new_node_val_data.dst_node_ids, seed=1)
            test_neg_edge_sampler = NegativeEdgeSampler(src_node_ids=full_data.src_node_ids, dst_node_ids=full_data.dst_node_ids, seed=2)
            new_node_test_neg_edge_sampler = NegativeEdgeSampler(src_node_ids=new_node_test_data.src_node_ids, dst_node_ids=new_node_test_data.dst_node_ids, seed=3)

            # get data loaders
            train_eval_data_loader = get_idx_data_loader(indices_list=list(range(len(val_data.src_node_ids))), batch_size=args.batch_size // 2, shuffle=False)
            val_idx_data_loader = get_idx_data_loader(indices_list=list(range(len(val_data.src_node_ids))), batch_size=args.batch_size // 2, shuffle=False)
            new_node_val_idx_data_loader = get_idx_data_loader(indices_list=list(range(len(new_node_val_data.src_node_ids))), batch_size=args.batch_size // 2, shuffle=False)
            test_idx_data_loader = get_idx_data_loader(indices_list=list(range(len(test_data.src_node_ids))), batch_size=args.batch_size // 2, shuffle=False)
            new_node_test_idx_data_loader = get_idx_data_loader(indices_list=list(range(len(new_node_test_data.src_node_ids))), batch_size=args.batch_size // 2, shuffle=False)

    val_metric_all_runs, new_node_val_metric_all_runs, test_metric_all_runs, new_node_test_metric_all_runs = [], [], [], []


    # Main code
    for run in range(args.num_runs):

        set_random_seed(seed=run)

        args.seed = run
        args.save_model_name = f'{args.run_name}_seed{args.seed}'

        # set up logger
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger()
        os.makedirs(f"./logs/{args.run_name}/{args.dataset_name}/{args.save_model_name}/", exist_ok=True)

        # create file handler that logs debug and higher level messages
        fh = logging.FileHandler(f"./logs/{args.run_name}/{args.dataset_name}/{args.save_model_name}/{str(time.time())}.log")
        # create console handler with a higher log level
        ch = logging.StreamHandler()
        ch.setLevel(logging.WARNING)
        # create formatter and add it to the handlers
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        # add the handlers to logger
        logger.addHandler(fh)
        logger.addHandler(ch)

        run_start_time = time.time()
        logger.info(f"********** Run {run + 1} starts. **********")

        logger.info(f'configuration is {args}')

        # create model
        patterns, eids = None, None
        ntp = False if args.no_ntp else True
        lt = False if args.no_lt else True
        st = False if args.no_st else True
        random_mask = True if args.random_mask else False
        dynamic_backbone = PretrainModel(node_dim=node_dim, edge_dim=edge_dim, neighbor_sampler=train_neighbor_sampler,
                                        time_feat_dim=args.time_feat_dim, hidden_dim=args.hidden_dim, num_enc_layer=args.encoder_layers,
                                        num_dec_layer=args.decoder_layers, num_heads=args.num_heads, num_tokens=args.num_neighbors, dropout=args.dropout,
                                        ntp=ntp, lt=lt, st=st, random_mask=random_mask, block_size=args.block_size, visible_ratio=args.visible_ratio)
        link_predictor = MergeLayer(input_dim1=args.hidden_dim, input_dim2=args.hidden_dim,
                                    hidden_dim=node_raw_features.shape[1], output_dim=1)
        
        model = nn.Sequential(dynamic_backbone, link_predictor)
        logger.info(f'model -> {model}')
        logger.info(f'model name: {args.run_name}, #parameters: {get_parameter_sizes(model) * 4} B, '
                    f'{get_parameter_sizes(model) * 4 / 1024} KB, {get_parameter_sizes(model) * 4 / 1024 / 1024} MB.')

        if args.pre_train:
            pt_optimizer = create_optimizer(model=model[0], optimizer_name=args.optimizer, learning_rate=args.learning_rate, weight_decay=args.weight_decay)
            ft_optimizer = create_optimizer(model=model[1], optimizer_name=args.optimizer, learning_rate=args.pred_lr, weight_decay=args.weight_decay)
        elif args.ft:
            optimizer = torch.optim.Adam([
                    {"params": model[0].parameters(), "lr": args.learning_rate},
                    {"params": model[1].parameters(), "lr": args.pred_lr}
                ], weight_decay=args.weight_decay)
        else:
            optimizer = create_optimizer(model=model, optimizer_name=args.optimizer, learning_rate=args.learning_rate, weight_decay=args.weight_decay)

        device = torch.device(args.device)
        model = model.to(device)
        if not args.ft:
            save_model_folder = f"./saved_models/{args.run_name}/{args.dataset_name}/{args.save_model_name}/"
        else:
            save_model_folder = f"./saved_models/{args.run_name}/{args.dataset_name}/{args.save_model_name}_ft_{datasets[args.eval_dataset]}/"
            pt_model_folder = f"./saved_models/{args.run_name}/{args.dataset_name}/{args.run_name}_seed0/"
            pt_model_path = os.path.join(pt_model_folder, f"{args.run_name}_seed0.pkl")
        
        if not args.inference_only:
            os.makedirs(save_model_folder, exist_ok=True)

        early_stopping = EarlyStopping(patience=args.patience, save_model_folder=save_model_folder,
                                       save_model_name=args.save_model_name, logger=logger, model_name=args.run_name)
        best_result = 0.
        duration = 0
        save_model_path = os.path.join(save_model_folder, f"{args.save_model_name}.pkl")
        loss_func = nn.BCELoss()

        if args.pre_train:
            for epoch in range(args.num_epochs):
                train_losses = []
                if not args.inference_only:
                    # If multi-dataset setting is adopted, train on all given datasets.
                    for i in range(len(datasets)):
                        logger.info("Dataset: %s" % datasets[i])
                        model[0].set_dataset(nfeat_list[i], efeat_list[i], time_list[i])
                        model[0].set_neighbor_sampler(train_sampler_list[i])
                        train_idx_data_loader = train_loader_list[i]
                        train_data = train_data_list[i]
                        for batch_idx, train_data_indices in enumerate(train_idx_data_loader):
                            train_data_indices = train_data_indices.numpy()
                            batch_src_node_ids, batch_dst_node_ids, batch_node_interact_times, batch_edge_ids = \
                                train_data.src_node_ids[train_data_indices], train_data.dst_node_ids[train_data_indices], \
                                train_data.node_interact_times[train_data_indices], train_data.edge_ids[train_data_indices]
                            loss = model[0].pretrain_node(datasets[i], pattern_list[i], eid_list[i], batch_src_node_ids, batch_dst_node_ids, batch_node_interact_times)
                            # There are circumstances where there are no valid element in extracted patterns. 
                            # In such cases, pre-training loss should be 0.
                            if loss != 0.:
                                train_losses.append(loss.item())
                                pt_optimizer.zero_grad()
                                loss.backward()
                                pt_optimizer.step()
                                model[0].ema_update()
                                if batch_idx % 100 == 0:
                                    logger.info(f'Epoch: {epoch + 1}, train for the {batch_idx + 1}-th batch, train loss: {loss.item()}')
                            else:
                                if batch_idx % 100 == 0:
                                    logger.info(f'Epoch: {epoch + 1}, train for the {batch_idx + 1}-th batch, train loss: {loss}')
                else:
                    model[0].load_state_dict(torch.load(save_model_path, map_location=device))
                
                # Extract representations in current epoch for evaluation
                logger.info("Evaluation Dataset: %s" % datasets[args.eval_dataset])
                model[0].set_dataset(nfeat_list[args.eval_dataset], efeat_list[args.eval_dataset], time_list[args.eval_dataset])
                train_data = train_data_list[args.eval_dataset]
                dataset_name = datasets[args.eval_dataset]
                src_embed_list, dst_embed_list = [], []
                src_nlist, dst_nlist = [], []
                neg_src_embed_list, neg_dst_embed_list = [], []
                neg_src_nlist, neg_dst_nlist = [], []
                src_val_list, dst_val_list = [], []
                src_val_nlist, dst_val_nlist = [], []
                neg_src_val_list, neg_dst_val_list = [], []
                neg_src_val_nlist, neg_dst_val_nlist = [], []
                src_test_list, dst_test_list = [], []
                src_test_nlist, dst_test_nlist = [], []
                neg_src_test_list, neg_dst_test_list = [], []
                neg_src_test_nlist, neg_dst_test_nlist = [], []
                train_labels, val_labels, test_labels = [], [], []
                patterns, eids = pattern_list[args.eval_dataset], eid_list[args.eval_dataset]
                with torch.no_grad():
                    model[0].set_neighbor_sampler(train_sampler_list[args.eval_dataset])
                    train_neg_edge_sampler.reset_random_state()
                    for batch_idx, train_data_indices in enumerate(train_eval_data_loader):
                        train_data_indices = train_data_indices.numpy()
                        batch_src_node_ids, batch_dst_node_ids, batch_node_interact_times, batch_edge_ids = \
                            train_data.src_node_ids[train_data_indices], train_data.dst_node_ids[train_data_indices], \
                            train_data.node_interact_times[train_data_indices], train_data.edge_ids[train_data_indices]
                        # Negative Sampling
                        _, batch_neg_dst_node_ids = train_neg_edge_sampler.sample(size=len(batch_src_node_ids))
                        batch_neg_src_node_ids = batch_src_node_ids
                        batch_edge_labels = train_data.labels[train_data_indices]

                        batch_src_node_embeddings, batch_dst_node_embeddings, src_pad_nlist, dst_pad_nlist = \
                            model[0].encode_node(dataset_name, patterns, eids, batch_src_node_ids, batch_dst_node_ids, batch_node_interact_times)
                        batch_neg_src_node_embeddings, batch_neg_dst_node_embeddings, src_neg_pad_nlist, dst_neg_pad_nlist = \
                            model[0].encode_node(dataset_name, patterns, eids, batch_neg_src_node_ids, batch_neg_dst_node_ids, batch_node_interact_times)
                        src_embed_list.append(batch_src_node_embeddings.detach().cpu())
                        dst_embed_list.append(batch_dst_node_embeddings.detach().cpu())
                        src_nlist.append(src_pad_nlist)
                        dst_nlist.append(dst_pad_nlist)
                        neg_src_embed_list.append(batch_neg_src_node_embeddings.detach().cpu())
                        neg_dst_embed_list.append(batch_neg_dst_node_embeddings.detach().cpu())
                        neg_src_nlist.append(src_neg_pad_nlist)
                        neg_dst_nlist.append(dst_neg_pad_nlist)
                        train_labels.append(batch_edge_labels)
                    
                    val_neg_edge_sampler.reset_random_state()
                    model[0].set_neighbor_sampler(full_neighbor_sampler)
                    for batch_idx, train_data_indices in enumerate(val_idx_data_loader):
                        train_data_indices = train_data_indices.numpy()
                        batch_src_node_ids, batch_dst_node_ids, batch_node_interact_times, batch_edge_ids = \
                            val_data.src_node_ids[train_data_indices], val_data.dst_node_ids[train_data_indices], \
                            val_data.node_interact_times[train_data_indices], val_data.edge_ids[train_data_indices]
                        # Negative Sampling
                        _, batch_neg_dst_node_ids = val_neg_edge_sampler.sample(size=len(batch_src_node_ids))
                        batch_neg_src_node_ids = batch_src_node_ids
                        batch_edge_label = val_data.labels[train_data_indices]
                        
                        batch_src_node_embeddings, batch_dst_node_embeddings, src_pad_nlist, dst_pad_nlist = \
                            model[0].encode_node(dataset_name, patterns, eids, batch_src_node_ids, batch_dst_node_ids, batch_node_interact_times)
                        batch_neg_src_node_embeddings, batch_neg_dst_node_embeddings, src_neg_pad_nlist, dst_neg_pad_nlist = \
                            model[0].encode_node(dataset_name, patterns, eids, batch_neg_src_node_ids, batch_neg_dst_node_ids, batch_node_interact_times)
                        src_val_list.append(batch_src_node_embeddings.detach().cpu())
                        dst_val_list.append(batch_dst_node_embeddings.detach().cpu())
                        src_val_nlist.append(src_pad_nlist)
                        dst_val_nlist.append(dst_pad_nlist)
                        neg_src_val_list.append(batch_neg_src_node_embeddings.detach().cpu())
                        neg_dst_val_list.append(batch_neg_dst_node_embeddings.detach().cpu())
                        neg_src_val_nlist.append(src_neg_pad_nlist)
                        neg_dst_val_nlist.append(dst_neg_pad_nlist)
                        val_labels.append(batch_edge_label)

                    test_neg_edge_sampler.reset_random_state()
                    model[0].set_neighbor_sampler(full_neighbor_sampler)
                    for batch_idx, train_data_indices in enumerate(test_idx_data_loader):
                        train_data_indices = train_data_indices.numpy()
                        batch_src_node_ids, batch_dst_node_ids, batch_node_interact_times, batch_edge_ids = \
                            test_data.src_node_ids[train_data_indices], test_data.dst_node_ids[train_data_indices], \
                            test_data.node_interact_times[train_data_indices], test_data.edge_ids[train_data_indices]
                        # Negative Sampling
                        _, batch_neg_dst_node_ids = test_neg_edge_sampler.sample(size=len(batch_src_node_ids))
                        batch_neg_src_node_ids = batch_src_node_ids
                        batch_edge_labels = test_data.labels[train_data_indices]
                        
                        batch_src_node_embeddings, batch_dst_node_embeddings, src_pad_nlist, dst_pad_nlist = \
                            model[0].encode_node(dataset_name, patterns, eids, batch_src_node_ids, batch_dst_node_ids, batch_node_interact_times)
                        batch_neg_src_node_embeddings, batch_neg_dst_node_embeddings, src_neg_pad_nlist, dst_neg_pad_nlist = \
                            model[0].encode_node(dataset_name, patterns, eids, batch_neg_src_node_ids, batch_neg_dst_node_ids, batch_node_interact_times)
                        src_test_list.append(batch_src_node_embeddings.detach().cpu())
                        dst_test_list.append(batch_dst_node_embeddings.detach().cpu())
                        src_test_nlist.append(src_pad_nlist)
                        dst_test_nlist.append(dst_pad_nlist)
                        neg_src_test_list.append(batch_neg_src_node_embeddings.detach().cpu())
                        neg_dst_test_list.append(batch_neg_dst_node_embeddings.detach().cpu())
                        neg_src_test_nlist.append(src_neg_pad_nlist)
                        neg_dst_test_nlist.append(dst_neg_pad_nlist)
                        test_labels.append(batch_edge_labels)

                src_embed = torch.cat(src_embed_list, dim=0)
                dst_embed = torch.cat(dst_embed_list, dim=0)
                neg_src_embed = torch.cat(neg_src_embed_list, dim=0)
                neg_dst_embed = torch.cat(neg_dst_embed_list, dim=0)
                src_val_embed = torch.cat(src_val_list, dim=0)
                dst_val_embed = torch.cat(dst_val_list, dim=0)
                neg_src_val_embed = torch.cat(neg_src_val_list, dim=0)
                neg_dst_val_embed = torch.cat(neg_dst_val_list, dim=0)
                src_test_embed = torch.cat(src_test_list, dim=0)
                dst_test_embed = torch.cat(dst_test_list, dim=0)
                neg_src_test_embed = torch.cat(neg_src_test_list, dim=0)
                neg_dst_test_embed = torch.cat(neg_dst_test_list, dim=0)

                src_nlist = np.concatenate(src_nlist, axis=0)
                dst_nlist = np.concatenate(dst_nlist, axis=0)
                neg_src_nlist = np.concatenate(neg_src_nlist, axis=0)
                neg_dst_nlist = np.concatenate(neg_dst_nlist, axis=0)
                src_val_nlist = np.concatenate(src_val_nlist, axis=0)
                dst_val_nlist = np.concatenate(dst_val_nlist, axis=0)
                neg_src_val_nlist = np.concatenate(neg_src_val_nlist, axis=0)
                neg_dst_val_nlist = np.concatenate(neg_dst_val_nlist, axis=0)
                src_test_nlist = np.concatenate(src_test_nlist, axis=0)
                dst_test_nlist = np.concatenate(dst_test_nlist, axis=0)
                neg_src_test_nlist = np.concatenate(neg_src_test_nlist, axis=0)
                neg_dst_test_nlist = np.concatenate(neg_dst_test_nlist, axis=0)

                train_labels = np.concatenate(train_labels, axis=0)
                val_labels = np.concatenate(val_labels, axis=0)
                test_labels = np.concatenate(test_labels, axis=0)

                # Train prediction head on detached representations, and then evaluate.
                val_metric_all_runs, new_node_val_metric_all_runs, test_metric_all_runs, new_node_test_metric_all_runs = [], [], [], []
                logger.info("Start Evaluation...")
                logger.info("Evaluation for temporal link prediction...")
                for r in range(args.num_runs):
                    set_random_seed(r)
                    model[1].reset_parameters()
                    eval_es = EarlyStopping(patience=15, save_model_folder=save_model_folder,
                                        save_model_name=args.save_model_name, logger=logger, model_name=args.run_name)
                    eval_batch_size = 2000
                    train_idx = torch.randperm(src_embed.size(0))
                    eval_idx = torch.randperm(src_val_embed.size(0))
                    test_idx = torch.randperm(src_test_embed.size(0))
                    eval_loss = 0
                    num_train_batches = (src_embed.size(0) + eval_batch_size - 1) // eval_batch_size
                    num_eval_batches = (src_val_embed.size(0) + eval_batch_size - 1) // eval_batch_size
                    num_test_batches = (src_test_embed.size(0) + eval_batch_size - 1) // eval_batch_size
                    for epoch in range(100):
                        for i in range(num_train_batches):
                            batch_idx = train_idx[i * eval_batch_size:(i + 1) * eval_batch_size]
                            batch_src_node_embeddings, batch_dst_node_embeddings = src_embed[batch_idx].to(device), dst_embed[batch_idx].to(device)
                            batch_neg_src_node_embeddings, batch_neg_dst_node_embeddings = neg_src_embed[batch_idx].to(device), neg_dst_embed[batch_idx].to(device)
                            batch_src_nlist, batch_dst_nlist = src_nlist[batch_idx], dst_nlist[batch_idx]
                            batch_neg_src_nlist, batch_neg_dst_nlist = neg_src_nlist[batch_idx], neg_dst_nlist[batch_idx]
                            orig_positive_probabilities = model[1](input_1=batch_src_node_embeddings, input_2=batch_dst_node_embeddings, src_nlist=batch_src_nlist, dst_nlist=batch_dst_nlist).squeeze(dim=-1)
                            orig_negative_probabilities = model[1](input_1=batch_neg_src_node_embeddings, input_2=batch_neg_dst_node_embeddings, src_nlist=batch_neg_src_nlist, dst_nlist=batch_neg_dst_nlist).squeeze(dim=-1)

                            positive_probabilities = orig_positive_probabilities.sigmoid()
                            negative_probabilities = orig_negative_probabilities.sigmoid()

                            predicts = torch.cat([positive_probabilities, negative_probabilities], dim=0)
                            labels = torch.cat([torch.ones_like(positive_probabilities), torch.zeros_like(negative_probabilities)], dim=0)

                            loss = loss_func(input=predicts, target=labels)

                            ft_optimizer.zero_grad()
                            loss.backward()
                            ft_optimizer.step()

                            if i % 10 == 0:
                                logger.info(f'evaluate training for the {i + 1}-th batch, evaluate loss: {loss.item()}')
                        
                        with torch.no_grad():
                            val_losses, val_metrics = [], []
                            for i in range(num_eval_batches):
                                batch_idx = eval_idx[i * eval_batch_size:(i + 1) * eval_batch_size]
                                batch_src_node_embeddings, batch_dst_node_embeddings = src_val_embed[batch_idx].to(device), dst_val_embed[batch_idx].to(device)
                                batch_neg_src_node_embeddings, batch_neg_dst_node_embeddings = neg_src_val_embed[batch_idx].to(device), neg_dst_val_embed[batch_idx].to(device)
                                batch_src_nlist, batch_dst_nlist = src_val_nlist[batch_idx], dst_val_nlist[batch_idx]
                                batch_neg_src_nlist, batch_neg_dst_nlist = neg_src_val_nlist[batch_idx], neg_dst_val_nlist[batch_idx]
                                positive_probabilities = model[1](input_1=batch_src_node_embeddings, input_2=batch_dst_node_embeddings, src_nlist=batch_src_nlist, dst_nlist=batch_dst_nlist).squeeze(dim=-1).sigmoid()
                                negative_probabilities = model[1](input_1=batch_neg_src_node_embeddings, input_2=batch_neg_dst_node_embeddings, src_nlist=batch_neg_src_nlist, dst_nlist=batch_neg_dst_nlist).squeeze(dim=-1).sigmoid()

                                predicts = torch.cat([positive_probabilities, negative_probabilities], dim=0)
                                labels = torch.cat([torch.ones_like(positive_probabilities), torch.zeros_like(negative_probabilities)], dim=0)
                                loss = loss_func(input=predicts, target=labels)
                                val_losses.append(loss.item())
                                val_metrics.append(get_link_prediction_metrics(predicts=predicts, labels=labels))
                                if i % 50 == 0:
                                    logger.info(f'evaluate for the {i + 1}-th batch, evaluate loss: {loss.item()}')

                        val_metric_indicator = []
                        for metric_name in val_metrics[0].keys():
                            val_metric_indicator.append((metric_name, np.mean([val_metric[metric_name] for val_metric in val_metrics]), True))
                        early_stop = eval_es.step(val_metric_indicator, model, save_ckpt=False)
                        if early_stop:
                            break
                    
                    with torch.no_grad():
                        val_losses, val_metrics = [], []
                        for i in range(num_eval_batches):
                            batch_idx = eval_idx[i * eval_batch_size:(i + 1) * eval_batch_size]
                            batch_src_node_embeddings, batch_dst_node_embeddings = src_val_embed[batch_idx].to(device), dst_val_embed[batch_idx].to(device)
                            batch_neg_src_node_embeddings, batch_neg_dst_node_embeddings = neg_src_val_embed[batch_idx].to(device), neg_dst_val_embed[batch_idx].to(device)
                            batch_src_nlist, batch_dst_nlist = src_val_nlist[batch_idx], dst_val_nlist[batch_idx]
                            batch_neg_src_nlist, batch_neg_dst_nlist = neg_src_val_nlist[batch_idx], neg_dst_val_nlist[batch_idx]
                            positive_probabilities = model[1](input_1=batch_src_node_embeddings, input_2=batch_dst_node_embeddings, src_nlist=batch_src_nlist, dst_nlist=batch_dst_nlist).squeeze(dim=-1).sigmoid()
                            negative_probabilities = model[1](input_1=batch_neg_src_node_embeddings, input_2=batch_neg_dst_node_embeddings, src_nlist=batch_neg_src_nlist, dst_nlist=batch_neg_dst_nlist).squeeze(dim=-1).sigmoid()

                            predicts = torch.cat([positive_probabilities, negative_probabilities], dim=0)
                            labels = torch.cat([torch.ones_like(positive_probabilities), torch.zeros_like(negative_probabilities)], dim=0)
                            loss = loss_func(input=predicts, target=labels)
                            val_losses.append(loss.item())
                            val_metrics.append(get_link_prediction_metrics(predicts=predicts, labels=labels))
                            if i % 50 == 0:
                                logger.info(f'evaluate for the {i + 1}-th batch, evaluate loss: {loss.item()}')
                        
                        test_losses, test_metrics = [], []
                        for i in range(num_test_batches):
                            batch_idx = test_idx[i * eval_batch_size:(i + 1) * eval_batch_size]
                            batch_src_node_embeddings, batch_dst_node_embeddings = src_test_embed[batch_idx].to(device), dst_test_embed[batch_idx].to(device)
                            batch_neg_src_node_embeddings, batch_neg_dst_node_embeddings = neg_src_test_embed[batch_idx].to(device), neg_dst_test_embed[batch_idx].to(device)
                            batch_src_nlist, batch_dst_nlist = src_test_nlist[batch_idx], dst_test_nlist[batch_idx]
                            batch_neg_src_nlist, batch_neg_dst_nlist = neg_src_test_nlist[batch_idx], neg_dst_test_nlist[batch_idx]
                            positive_probabilities = model[1](input_1=batch_src_node_embeddings, input_2=batch_dst_node_embeddings, src_nlist=batch_src_nlist, dst_nlist=batch_dst_nlist).squeeze(dim=-1).sigmoid()
                            negative_probabilities = model[1](input_1=batch_neg_src_node_embeddings, input_2=batch_neg_dst_node_embeddings, src_nlist=batch_neg_src_nlist, dst_nlist=batch_neg_dst_nlist).squeeze(dim=-1).sigmoid()

                            predicts = torch.cat([positive_probabilities, negative_probabilities], dim=0)
                            labels = torch.cat([torch.ones_like(positive_probabilities), torch.zeros_like(negative_probabilities)], dim=0)
                            loss = loss_func(input=predicts, target=labels)
                            test_losses.append(loss.item())
                            test_metrics.append(get_link_prediction_metrics(predicts=predicts, labels=labels))
                            if i % 50 == 0:
                                logger.info(f'test for the {i + 1}-th batch, evaluate loss: {loss.item()}')
                    
                    val_metric_dict, new_node_val_metric_dict, test_metric_dict, new_node_test_metric_dict = {}, {}, {}, {}

                    logger.info(f'validate loss: {np.mean(val_losses):.4f}')
                    for metric_name in val_metrics[0].keys():
                        average_val_metric = np.mean([val_metric[metric_name] for val_metric in val_metrics])
                        logger.info(f'validate {metric_name}, {average_val_metric:.4f}')
                        val_metric_dict[metric_name] = average_val_metric

                    logger.info(f'test loss: {np.mean(test_losses):.4f}')
                    for metric_name in test_metrics[0].keys():
                        average_test_metric = np.mean([test_metric[metric_name] for test_metric in test_metrics])
                        logger.info(f'test {metric_name}, {average_test_metric:.4f}')
                        test_metric_dict[metric_name] = average_test_metric
                    
                    val_metric_all_runs.append(val_metric_dict)
                    test_metric_all_runs.append(test_metric_dict)

                # store the average metrics at the log of the last run
                logger.info(f'metrics over {args.num_runs} runs:')

                for metric_name in val_metric_all_runs[0].keys():
                    logger.info(f'validate {metric_name}, {[val_metric_single_run[metric_name] for val_metric_single_run in val_metric_all_runs]}')
                    avg_val_performance = np.mean([val_metric_single_run[metric_name] for val_metric_single_run in val_metric_all_runs])
                    logger.info(f'average validate {metric_name}, {np.mean([val_metric_single_run[metric_name] for val_metric_single_run in val_metric_all_runs]):.4f} '
                                f'± {np.std([val_metric_single_run[metric_name] for val_metric_single_run in val_metric_all_runs], ddof=1):.4f}')

                for metric_name in test_metric_all_runs[0].keys():
                    logger.info(f'test {metric_name}, {[test_metric_single_run[metric_name] for test_metric_single_run in test_metric_all_runs]}')
                    logger.info(f'average test {metric_name}, {np.mean([test_metric_single_run[metric_name] for test_metric_single_run in test_metric_all_runs]):.4f} '
                                f'± {np.std([test_metric_single_run[metric_name] for test_metric_single_run in test_metric_all_runs], ddof=1):.4f}')

                if args.inference_only:
                    sys.exit()
                
                if avg_val_performance > best_result:
                    duration = 0
                    best_result = avg_val_performance
                    logger.info(f"save model {save_model_path}")
                    torch.save(model[0].state_dict(), save_model_path)
                else:
                    duration += 1
                    if duration == args.patience:
                        sys.exit()
                set_random_seed(run)

        else:
            if args.ft:
                model[0].load_state_dict(torch.load(pt_model_path, map_location=device))
            if args.inference_only:
                model.load_state_dict(torch.load(save_model_path, map_location=device))
            train_neighbor_sampler = train_sampler_list[args.eval_dataset]
            train_idx_data_loader = train_loader_list[args.eval_dataset]
            train_data = train_data_list[args.eval_dataset]
            patterns, eids = pattern_list[args.eval_dataset], eid_list[args.eval_dataset]
            model[0].set_dataset(nfeat_list[args.eval_dataset], efeat_list[args.eval_dataset], time_list[args.eval_dataset])
            if not args.inference_only:
                for epoch in range(args.num_epochs):
                    model.train()
                    # Fine-tune on specified single graph only.
                    model[0].set_neighbor_sampler(train_neighbor_sampler)
                    
                    # store train losses and metrics
                    train_losses, train_metrics = [], []
                    for batch_idx, train_data_indices in enumerate(train_idx_data_loader):
                        train_data_indices = train_data_indices.numpy()
                        batch_src_node_ids, batch_dst_node_ids, batch_node_interact_times, batch_edge_ids = \
                            train_data.src_node_ids[train_data_indices], train_data.dst_node_ids[train_data_indices], \
                            train_data.node_interact_times[train_data_indices], train_data.edge_ids[train_data_indices]

                        _, batch_neg_dst_node_ids = train_neg_edge_sampler.sample(size=len(batch_src_node_ids))
                        batch_neg_src_node_ids = batch_src_node_ids

                        # Directly train on link prediction task
                        batch_src_node_embeddings, batch_dst_node_embeddings, _, _ = \
                            model[0].encode_node(args.dataset_name, patterns, eids, batch_src_node_ids, batch_dst_node_ids, batch_node_interact_times)

                        batch_neg_src_node_embeddings, batch_neg_dst_node_embeddings, _, _ = \
                            model[0].encode_node(args.dataset_name, patterns, eids, batch_neg_src_node_ids, batch_neg_dst_node_ids, batch_node_interact_times)
                        
                        # get positive and negative probabilities, shape (batch_size, )
                        positive_probabilities = model[1](input_1=batch_src_node_embeddings, input_2=batch_dst_node_embeddings).squeeze(dim=-1).sigmoid()
                        negative_probabilities = model[1](input_1=batch_neg_src_node_embeddings, input_2=batch_neg_dst_node_embeddings).squeeze(dim=-1).sigmoid()

                        predicts = torch.cat([positive_probabilities, negative_probabilities], dim=0)
                        labels = torch.cat([torch.ones_like(positive_probabilities), torch.zeros_like(negative_probabilities)], dim=0)

                        loss = loss_func(input=predicts, target=labels)

                        train_losses.append(loss.item())
                        train_metrics.append(get_link_prediction_metrics(predicts=predicts, labels=labels))

                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                        if batch_idx % 100 == 0:
                            logger.info(f'Epoch: {epoch + 1}, train for the {batch_idx + 1}-th batch, train loss: {loss.item()}')
                    
                    val_losses, val_metrics = evaluate_model_link_prediction(logger=logger,
                                                                            model=model,
                                                                            neighbor_sampler=full_neighbor_sampler,
                                                                            evaluate_idx_data_loader=val_idx_data_loader,
                                                                            evaluate_neg_edge_sampler=val_neg_edge_sampler,
                                                                            evaluate_data=val_data,
                                                                            loss_func=loss_func,
                                                                            dataset=args.dataset_name,
                                                                            patterns=patterns,
                                                                            eids=eids)

                    new_node_val_losses, new_node_val_metrics = evaluate_model_link_prediction(logger=logger, 
                                                                                            model=model,
                                                                                            neighbor_sampler=full_neighbor_sampler,
                                                                                            evaluate_idx_data_loader=new_node_val_idx_data_loader,
                                                                                            evaluate_neg_edge_sampler=new_node_val_neg_edge_sampler,
                                                                                            evaluate_data=new_node_val_data,
                                                                                            loss_func=loss_func,
                                                                                            dataset=args.dataset_name,
                                                                                            patterns=patterns,
                                                                                            eids=eids)
                    
                    logger.info(f'Epoch: {epoch + 1}, learning rate: {optimizer.param_groups[0]["lr"]}, train loss: {np.mean(train_losses):.4f}')
                    for metric_name in train_metrics[0].keys():
                        logger.info(f'train {metric_name}, {np.mean([train_metric[metric_name] for train_metric in train_metrics]):.4f}')
                    logger.info(f'validate loss: {np.mean(val_losses):.4f}')
                    for metric_name in val_metrics[0].keys():
                        logger.info(f'validate {metric_name}, {np.mean([val_metric[metric_name] for val_metric in val_metrics]):.4f}')
                    logger.info(f'new node validate loss: {np.mean(new_node_val_losses):.4f}')
                    for metric_name in new_node_val_metrics[0].keys():
                        logger.info(f'new node validate {metric_name}, {np.mean([new_node_val_metric[metric_name] for new_node_val_metric in new_node_val_metrics]):.4f}')

                    if (epoch + 1) % args.test_interval_epochs == 0:
                        test_losses, test_metrics = evaluate_model_link_prediction(logger=logger,
                                                                                model=model,
                                                                                neighbor_sampler=full_neighbor_sampler,
                                                                                evaluate_idx_data_loader=test_idx_data_loader,
                                                                                evaluate_neg_edge_sampler=test_neg_edge_sampler,
                                                                                evaluate_data=test_data,
                                                                                loss_func=loss_func,
                                                                                dataset=args.dataset_name,
                                                                                patterns=patterns,
                                                                                eids=eids)

                        new_node_test_losses, new_node_test_metrics = evaluate_model_link_prediction(logger=logger, 
                                                                                                    model=model,
                                                                                                    neighbor_sampler=full_neighbor_sampler,
                                                                                                    evaluate_idx_data_loader=new_node_test_idx_data_loader,
                                                                                                    evaluate_neg_edge_sampler=new_node_test_neg_edge_sampler,
                                                                                                    evaluate_data=new_node_test_data,
                                                                                                    loss_func=loss_func,
                                                                                                    dataset=args.dataset_name,
                                                                                                    patterns=patterns,
                                                                                                    eids=eids)

                        logger.info(f'test loss: {np.mean(test_losses):.4f}')
                        for metric_name in test_metrics[0].keys():
                            logger.info(f'test {metric_name}, {np.mean([test_metric[metric_name] for test_metric in test_metrics]):.4f}')
                        logger.info(f'new node test loss: {np.mean(new_node_test_losses):.4f}')
                        for metric_name in new_node_test_metrics[0].keys():
                            logger.info(f'new node test {metric_name}, {np.mean([new_node_test_metric[metric_name] for new_node_test_metric in new_node_test_metrics]):.4f}')

                    # select the best model based on all the validate metrics
                    val_metric_indicator = []
                    for metric_name in val_metrics[0].keys():
                        val_metric_indicator.append((metric_name, np.mean([val_metric[metric_name] for val_metric in val_metrics]), True))
                    early_stop = early_stopping.step(val_metric_indicator, model)

                    if early_stop:
                        break
        
        if not args.inference_only:
            # load the best model
            early_stopping.load_checkpoint(model)

        # evaluate the best model
        logger.info(f'get final performance on dataset {args.dataset_name}...')

        test_losses, test_metrics = evaluate_model_link_prediction(logger=logger,
                                                                model=model,
                                                                neighbor_sampler=full_neighbor_sampler,
                                                                evaluate_idx_data_loader=test_idx_data_loader,
                                                                evaluate_neg_edge_sampler=test_neg_edge_sampler,
                                                                evaluate_data=test_data,
                                                                loss_func=loss_func,
                                                                dataset=args.dataset_name,
                                                                patterns=patterns,
                                                                eids=eids)

        new_node_test_losses, new_node_test_metrics = evaluate_model_link_prediction(logger=logger, 
                                                                                    model=model,
                                                                                    neighbor_sampler=full_neighbor_sampler,
                                                                                    evaluate_idx_data_loader=new_node_test_idx_data_loader,
                                                                                    evaluate_neg_edge_sampler=new_node_test_neg_edge_sampler,
                                                                                    evaluate_data=new_node_test_data,
                                                                                    loss_func=loss_func,
                                                                                    dataset=args.dataset_name,
                                                                                    patterns=patterns,
                                                                                    eids=eids)
        # store the evaluation metrics at the current run
        val_metric_dict, new_node_val_metric_dict, test_metric_dict, new_node_test_metric_dict = {}, {}, {}, {}

        logger.info(f'test loss: {np.mean(test_losses):.4f}')
        for metric_name in test_metrics[0].keys():
            average_test_metric = np.mean([test_metric[metric_name] for test_metric in test_metrics])
            logger.info(f'test {metric_name}, {average_test_metric:.4f}')
            test_metric_dict[metric_name] = average_test_metric

        logger.info(f'new node test loss: {np.mean(new_node_test_losses):.4f}')
        for metric_name in new_node_test_metrics[0].keys():
            average_new_node_test_metric = np.mean([new_node_test_metric[metric_name] for new_node_test_metric in new_node_test_metrics])
            logger.info(f'new node test {metric_name}, {average_new_node_test_metric:.4f}')
            new_node_test_metric_dict[metric_name] = average_new_node_test_metric

        single_run_time = time.time() - run_start_time
        logger.info(f'Run {run + 1} cost {single_run_time:.2f} seconds.')

        test_metric_all_runs.append(test_metric_dict)
        new_node_test_metric_all_runs.append(new_node_test_metric_dict)

        # avoid the overlap of logs
        if run < args.num_runs - 1:
            logger.removeHandler(fh)
            logger.removeHandler(ch)

        result_json = {
            "test metrics": {metric_name: f'{test_metric_dict[metric_name]:.4f}' for metric_name in test_metric_dict},
            "new node test metrics": {metric_name: f'{new_node_test_metric_dict[metric_name]:.4f}' for metric_name in new_node_test_metric_dict}
        }
        result_json = json.dumps(result_json, indent=4)

        save_result_folder = f"./saved_results/{args.run_name}/{args.dataset_name}"
        os.makedirs(save_result_folder, exist_ok=True)
        save_result_path = os.path.join(save_result_folder, f"{args.save_model_name}.json")

        with open(save_result_path, 'w') as file:
            file.write(result_json)

    # store the average metrics at the log of the last run
    logger.info(f'metrics over {args.num_runs} runs:')

    for metric_name in test_metric_all_runs[0].keys():
        logger.info(f'test {metric_name}, {[test_metric_single_run[metric_name] for test_metric_single_run in test_metric_all_runs]}')
        logger.info(f'average test {metric_name}, {np.mean([test_metric_single_run[metric_name] for test_metric_single_run in test_metric_all_runs]):.4f} '
                    f'± {np.std([test_metric_single_run[metric_name] for test_metric_single_run in test_metric_all_runs], ddof=1):.4f}')

    for metric_name in new_node_test_metric_all_runs[0].keys():
        logger.info(f'new node test {metric_name}, {[new_node_test_metric_single_run[metric_name] for new_node_test_metric_single_run in new_node_test_metric_all_runs]}')
        logger.info(f'average new node test {metric_name}, {np.mean([new_node_test_metric_single_run[metric_name] for new_node_test_metric_single_run in new_node_test_metric_all_runs]):.4f} '
                    f'± {np.std([new_node_test_metric_single_run[metric_name] for new_node_test_metric_single_run in new_node_test_metric_all_runs], ddof=1):.4f}')
