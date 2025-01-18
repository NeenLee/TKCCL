import os
from time import time
import dgl
import pandas as pd
import torch
import torch.nn.functional as F
import numpy as np
from TSGD.LNModel.LNGNN import myGAT
from TSGD.LNModel.uitl.batch_test import data_generator, test, LNtest
from TSGD.LNModel.uitl.helper import ensureDir
from TSGD.LNModel.uitl.parser import parse_args

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True


def load_pretrained_data(args):
    # pre_model = 'cp'
    # pretrain_path = '%s../pretrain/%s/%s.npz' % (args.proj_path, args.dataset, pre_model)
    # pretrain_path = '../pretrain/Cell_Phones_and_Accessories/2014/cp.npz'
    # pretrain_path = '../pretrain/YelpZip/cp.npz'
    # pretrain_path = '../pretrain/Yelp_New York/cp.npz'
    pretrain_path = '../pretrain/YelpChi/cp.npz'

    try:
        pretrain_data = np.load(pretrain_path)
        print('load the pretrained bprmf model parameters.')
    except Exception:
        pretrain_data = None
    return pretrain_data


if __name__ == '__main__':

    torch.manual_seed(2023)
    np.random.seed(2023)
    args = parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

    """
    *********************************************************
    Load Data from data_generator function.
    """
    config = dict()
    # 原始
    config['n_users'] = data_generator.n_users
    config['n_items'] = data_generator.n_items

    config['n_relations'] = data_generator.n_relations
    config['n_entities'] = data_generator.n_entities

    t0 = time()
    """
    *********************************************************
    Use the pretrained data to initialize the embeddings.
    """
    if args.pretrain in [-1, -2]:
        pretrain_data = load_pretrained_data(args)
    else:
        pretrain_data = None
    # print(pretrain_data)
    """
    *********************************************************
    Select one of the models.
    """
    weight_size = eval(args.layer_size)
    num_layers = len(weight_size) - 2
    heads = [args.heads] * num_layers + [1]
    print(config['n_users'], config['n_entities'], args.kge_size, config['n_relations'])

    model = myGAT(args, config['n_entities'], config['n_relations'] + 1, weight_size[-2], weight_size[-1], num_layers,
                  heads, F.elu, 0.1, 0., 0.01, False, pretrain=pretrain_data).cuda()

    adjM = data_generator.lap_list

    # print(len(adjM.nonzero()[0]))
    g = dgl.from_scipy(adjM)
    g = dgl.remove_self_loop(g)
    g = dgl.add_self_loop(g)
    g = g.to('cuda')

    #
    edge2type = {}
    for i, mat in enumerate(data_generator.kg_lap_list):
        for u, v in zip(*mat.nonzero()):
            edge2type[(u, v)] = i
    for i in range(data_generator.n_entities):
        edge2type[(i, i)] = len(data_generator.kg_lap_list)

    kg_adjM = sum(data_generator.kg_lap_list)
    kg = dgl.from_scipy(kg_adjM)
    kg = dgl.remove_self_loop(kg)
    kg = dgl.add_self_loop(kg)

    e_feat = []
    for u, v in zip(*kg.edges()):
        u = u.item()
        v = v.item()
        if u == v:
            break
        e_feat.append(edge2type[(u, v)])
    for i in range(data_generator.n_entities):
        e_feat.append(edge2type[(i, i)])
    e_feat = torch.tensor(e_feat, dtype=torch.long).to('cuda')

    kg = kg.to('cuda')

    """
        *********************************************************
        Save the model parameters.
        """
    if args.save_flag == 1:
        weights_save_path = '{}weights/{}/{}/{}_{}.pt'.format(args.weights_path, args.dataset, args.model_type,
                                                              num_layers, args.heads)
        ensureDir(weights_save_path)
        torch.save(model, weights_save_path)

    cur_best_pre_0 = 0.

    """
    *********************************************************
    Train.
    """
    loss_loger, pre_loger, rec_loger, ndcg_loger, hit_loger = [], [], [], [], []
    stopping_step = 0
    should_stop = False
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    optimizer2 = torch.optim.Adam(model.parameters(), lr=args.kg_lr)
    optimizer3 = torch.optim.Adam(model.parameters(), lr=args.cl_lr)
    dropout_rate = args.drop_rate
    for epoch in range(args.epoch):
        t1 = time()
        sub_cf_adjM = data_generator._get_cf_adj_list(is_subgraph=True, dropout_rate=dropout_rate)
        sub_cf_lap = data_generator._get_lap_list(is_subgraph=True, subgraph_adj=sub_cf_adjM)

        sub_cf_g = dgl.from_scipy(sub_cf_lap)
        sub_cf_g = dgl.add_self_loop(sub_cf_g)
        sub_cf_g = sub_cf_g.to('cuda')

        sub_kg_adjM, _ = data_generator._get_kg_adj_list(is_subgraph=True, dropout_rate=dropout_rate)
        sub_kg_lap = sum(data_generator._get_kg_lap_list(is_subgraph=True, subgraph_adj=sub_kg_adjM))
        sub_kg = dgl.from_scipy(sub_kg_lap)
        sub_kg = dgl.remove_self_loop(sub_kg)
        sub_kg = dgl.add_self_loop(sub_kg)

        sub_kg = sub_kg.to('cuda')
        loss, base_loss, kge_loss, reg_loss, cl_loss = 0., 0., 0., 0., 0.
        cf_drop, kg_drop = 0., 0.
        # 向下整除 //
        n_batch = data_generator.n_train // args.batch_size + 1
        n_kg_batch = data_generator.n_triples // args.batch_size_kg + 1
        n_cl_batch = data_generator.n_items // args.batch_size_cl + 1
        """
        *********************************************************
        Alternative Training for TSGD:
        ... phase 1: to train the Embedding.
        """
        # for idx in range(n_batch):
        #     torch.cuda.empty_cache()
        #     model.train()
        #     btime = time()
        #     batch_data = data_generator.generate_train_batch()
        #     loss, cf_drop, kg_drop = model("cf", g, sub_cf_g, sub_kg, batch_data['users'],
        #                                    batch_data['pos_items'] + data_generator.n_users,
        #                                    batch_data['neg_items'] + data_generator.n_users)
        #     optimizer.zero_grad()
        #     loss.backward()
        #     optimizer.step()

        for idx in range(n_kg_batch):
            torch.cuda.empty_cache()
            model.train()
            batch_data = data_generator.generate_train_kg_batch()
            kge_loss = model("kg", sub_kg, batch_data['heads'], batch_data['relations'], batch_data['pos_tails'],
                             batch_data['neg_tails'])

            optimizer2.zero_grad()
            kge_loss.backward()
            optimizer2.step()

        for idx in range(n_cl_batch):
            model.train()
            batch_data = data_generator.generate_train_cl_batch()
            cl_loss = model("cl", sub_cf_g, sub_kg, batch_data['items'])

            optimizer3.zero_grad()
            cl_loss.backward()
            optimizer3.step()

        del sub_cf_g, sub_kg
        show_step = 10
        if (epoch + 1) % show_step != 0:
            if args.verbose > 0 and epoch % args.verbose == 0:
                # perf_str = 'Epoch %d [%.1fs]: train==[%.5f + %.5f + %.5f] drop==[%.2f + %.2f]' % (
                #     epoch, time() - t1, float(loss), float(kge_loss), float(cl_loss), float(cf_drop), float(kg_drop))
                # print(perf_str)
                perf_str = 'Epoch %d [%.1fs]: train==[%.5f + %.5f + %.5f]' % (
                    epoch, time() - t1, float(loss), float(kge_loss), float(cl_loss))
                print(perf_str)
            continue

    """
        user_anomaly_score, item_anomaly_score output
    """
    # ui_emb = model.calc_ui_emb(g)
    # print(ui_emb.shape)
    # embedding_kg = model.calc_kg_emb(kg)
    #
    # output_user_anomaly_score = model.calc_anomaly_score(ui_emb)
    # output_item_anomaly_score = model.calc_anomaly_score(embedding_kg)
    # print("user_anomaly_score = ", output_user_anomaly_score)
    # print("item_anomaly_score = ", output_item_anomaly_score)

    """
    *********************************************************
    Test.
    """

    t2 = time()
    users_to_test = list(data_generator.test_user_dict.keys())
    user_id, score_list = LNtest(g, kg, model, users_to_test)
    dataframe = pd.DataFrame({'user_item_id': list(user_id), 'score': list(score_list)})
    # dataframe.to_csv("../output/Amazon_CPA/2014/user_item_anomalyScore_0.7_0.csv", index=False, sep=',')
    dataframe.to_csv("../output/Yelp_Chi/user_item_anomalyScore_0.7_1.csv", index=False, sep=',')
    # dataframe.to_csv("../output/Amazon_CPA/2014/user_item_anomalyScore_0.9.csv", index=False, sep=',')
    # 别忘了修改main函数中的预训练，上面
    # print(ret)
