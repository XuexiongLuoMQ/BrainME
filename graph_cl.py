def graph_cl_loss(batch_data, emb_dim):
    # for graph-level adaptation
    sup_task_nodes_emb.append(
        node_emb[cur_pos_sup_e_idx].reshape(-1, self.args.emb_dim))  # ([(#sup_set, dim), ...])
    que_task_nodes_emb.append(
        node_emb[cur_pos_que_e_idx].reshape(-1, self.args.emb_dim))  # ([(#que_set, dim), ...])
    # node-level loss on query set
    node_emb = self.gnn(x, batch_data.edge_index, batch_data.edge_attr, fast_weights)
    pos_score = torch.sum(node_emb[cur_pos_que_e_idx[0]] *
                            node_emb[cur_pos_que_e_idx[1]], dim=1)  # ([n_batch*#sup_set])
    neg_score = torch.sum(node_emb[cur_neg_que_e_idx[0]] *
                            node_emb[cur_neg_que_e_idx[1]], dim=1)
    loss = self.loss(pos_score, torch.ones_like(pos_score)) + \
            self.loss(neg_score, torch.zeros_like(neg_score))