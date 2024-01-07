# coding=utf-8
# @Author: dcteng
# @Desc: { 模块描述 }
# @Date: 2023/07/25
import torch

from models.module import *
from collections import defaultdict


def get_pretrained_model(model_type, dropout_rate):
    if model_type == 'ELECTRA':
        encoder = ElectraEncoder(dropout_rate)
    elif model_type == 'XLNet':
        encoder = XLNetEncoder(dropout_rate)
    else:
        raise ValueError(f"Model type '{model_type}' is not supported now.")
    return encoder


class ModelManager(nn.Module):

    def __init__(self, args, num_word, num_slot, num_intent, edge_types, profile_types):
        super(ModelManager, self).__init__()

        self.num_word = num_word
        self.num_slot = num_slot
        self.num_intent = num_intent
        self.edge_types = edge_types
        self.profile_types = profile_types
        self.args = args

        profile_info_dim = 0
        if args.use_info:
            profile_info_dim = self.args.profile_info_dim

            self.dropout_layer = nn.Dropout(self.args.dropout_rate)
            self.null_emb = nn.Parameter(torch.zeros(1, self.args.profile_info_dim), requires_grad=True)
            nn.init.normal_(self.null_emb, mean=0.0, std=1.0)

            graph_heads = [args.num_graph_heads for _ in range(args.num_graph_layer)]
            # Initialize an encoder for Graph nodes
            self.node_combined_encoder = CombinedEncoder(
                self.num_word,
                self.args.node_word_embedding_dim,
                self.args.node_encoder_hidden_dim,
                self.args.node_attention_hidden_dim,
                self.args.node_attention_output_dim,
                self.args.dropout_rate
            )
            node_encoder_dim = self.args.node_encoder_hidden_dim + self.args.node_attention_output_dim
            if self.args.use_pretrained and not self.args.disable_node_pretrained:
                self.node_plm_encoder = get_pretrained_model(self.args.model_type, self.args.plm_dropout_rate)
                node_encoder_dim = self.node_plm_encoder.model.config.hidden_size
            # Set input dim and output dim for various types of profile-node linear layer
            profile_type2dims = {
                'utter': (
                    self.args.encoder_hidden_dim + (0 if self.args.use_pretrained else self.args.attention_output_dim),
                    self.args.profile_info_dim),
                'kg': (
                    self.args.node_word_embedding_dim if self.args.kg_node_triplet_representation == "embed_bow"
                    else node_encoder_dim,
                    self.args.profile_info_dim),
                'up': (
                    self.args.node_word_embedding_dim if self.args.special_node_triplet_representation == "embed_bow"
                    else node_encoder_dim,
                    self.args.profile_info_dim)
            }
            profile_type2dims['ca'] = profile_type2dims['up']
            assert set(profile_type2dims.keys()) == set(self.profile_types), f"Profile types are not consistent."
            # Initialize a graph encoder
            self.profile_graph = GraphPF(
                self.args.profile_info_dim,
                self.args.graph_dropout_rate,
                edge_types,
                self.args.graph_hidden_size,
                self.args.profile_info_dim,
                graph_heads,
                profile_type2dims
            )

            self.query_projector = nn.Linear(profile_type2dims["utter"][0], self.args.profile_info_dim)

            mylogger.info(f"Edge types: {self.edge_types}")
            mylogger.info(f"Linear projection for profile types: {json.dumps(profile_type2dims, indent=4)}")

        if args.use_pretrained:
            # Initialize a PLM Encoder object for user utterance.
            self.encoder = get_pretrained_model(self.args.model_type, self.args.plm_dropout_rate)
            # info for intent detection: [encoder output, profile info]
            plm_hidden_dim = self.encoder.model.config.hidden_size
            # TODO: utterance node representation for intent detection
            self.intent_decoder = nn.Linear(
                plm_hidden_dim + profile_info_dim, self.num_intent
            )
            # Initialize a Decoder object for slot filling.
            self.slot_decoder = LSTMDecoder(
                plm_hidden_dim,
                self.args.slot_decoder_hidden_dim,
                self.num_slot, self.args.dropout_rate,
                embedding_dim=self.args.slot_embedding_dim,
                # TODO: utterance node representation for intent detection
                extra_dim=self.num_intent,
                info_dim=profile_info_dim,
                args=self.args
            )
        else:
            # Initialize a Combined Encoder (LSTM Encoder + self-attention layer + sentence aggregation layer) object for user utterance.
            self.encoder = CombinedEncoder(
                self.num_word,
                self.args.word_embedding_dim,
                self.args.encoder_hidden_dim,
                self.args.attention_hidden_dim,
                self.args.attention_output_dim,
                self.args.dropout_rate
            )

            # info for intent detection: [lstm output, self-attention output, profile info]
            self.intent_decoder = nn.Linear(
                self.args.encoder_hidden_dim + self.args.attention_output_dim + profile_info_dim, self.num_intent
            )

            # Initialize a Decoder object for slot filling.
            # Initialize a Decoder object for slot.
            self.slot_decoder = LSTMDecoder(
                self.args.encoder_hidden_dim + self.args.attention_output_dim,
                self.args.slot_decoder_hidden_dim,
                self.num_slot, self.args.dropout_rate,
                embedding_dim=self.args.slot_embedding_dim,
                info_dim=profile_info_dim,
                extra_dim=self.num_intent,
                args=self.args
            )

        self.intent_embedding = nn.Embedding(self.num_intent, self.num_intent)
        self.intent_embedding.weight.data = torch.eye(self.num_intent)
        self.intent_embedding.weight.requires_grad = False

        self.optimizers = None
        self.schedulers = None

    def define_optimizers(self):
        """
        Define the optimizers used for training.
        """
        optimizers = []
        schedulers = {}

        def _get_plm_parameters(module):
            no_decay = ['bias', 'LayerNorm.weight']
            optimizer_grouped_parameters = [
                {'params': [p for n, p in module.named_parameters() if any(nd in n for nd in no_decay)],
                 'weight_decay': 0.0},
                {'params': [p for n, p in module.named_parameters() if not any(nd in n for nd in no_decay)],
                 'weight_decay': 0.01}
            ]
            return optimizer_grouped_parameters

        excluded_param_ids = []
        if self.args.use_pretrained:
            utterance_encoder = self.encoder
            excluded_param_ids.extend(list(map(id, utterance_encoder.parameters())))
            plm_optimizer = torch.optim.AdamW(_get_plm_parameters(utterance_encoder), lr=self.args.plm_learning_rate)
            optimizers.append(plm_optimizer)
            schedulers["plm"] = lr_scheduler.ReduceLROnPlateau(
                plm_optimizer, mode='max', factor=self.args.lr_scheduler_factor,
                patience=self.args.lr_scheduler_patience, min_lr=min(self.args.plm_learning_rate, 1e-5), verbose=True)

        if self.args.use_info and self.args.use_pretrained and not self.args.disable_node_pretrained:
            # node encoder is a pretrained model
            node_encoder = self.node_plm_encoder
            excluded_param_ids.extend(list(map(id, node_encoder.parameters())))
            node_plm_optimizer = torch.optim.AdamW(_get_plm_parameters(node_encoder), lr=self.args.node_plm_learning_rate)
            optimizers.append(node_plm_optimizer)
            schedulers["node_plm"] = lr_scheduler.ReduceLROnPlateau(
                node_plm_optimizer, mode='max', factor=self.args.lr_scheduler_factor,
                patience=self.args.lr_scheduler_patience, min_lr=min(self.args.node_plm_learning_rate, 1e-5), verbose=True)
        # elif self.args.use_info:
        #     # node encoder is not a pretrained model
        #     excluded_param_ids.extend(list(map(id, self.node_encoder.parameters())))
        #     node_optimizer = torch.optim.AdamW(self.node_encoder.parameters(), lr=self.args.node_learning_rate)
        #     optimizers.append(node_optimizer)

        # same optimizer for Graph NN, lstm encoder and slot decoder
        base_params = filter(lambda p: id(p) not in excluded_param_ids, self.parameters())
        optimizer = torch.optim.AdamW(base_params, lr=self.args.learning_rate, weight_decay=self.args.weight_decay)
        optimizers.append(optimizer)
        schedulers["base"] = lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=self.args.lr_scheduler_factor,
            patience=self.args.lr_scheduler_patience, min_lr=1e-4, verbose=True)

        self.optimizers = optimizers
        self.schedulers = schedulers

    def backward_loss(self, loss):
        """
            Backward loss.
        """
        loss.backward()
        # gradient clipping
        torch.nn.utils.clip_grad_norm_(self.parameters(), self.args.clip)
        for optimizer in self.optimizers:
            optimizer.step()

    def apply_scheduler(self, metric):
        def _get_lr(optimizer):
            for param_group in optimizer.param_groups:
                return param_group['lr']

        for name, scheduler in self.schedulers.items():
            old_lr = _get_lr(scheduler.optimizer)
            scheduler.step(metric)
            new_lr = _get_lr(scheduler.optimizer)
            # record learning rate changes
            if old_lr != new_lr:
                mylogger.info(f"Learning rate of '{name}' changes from {old_lr} to {new_lr}.")

    def _cuda(self, x, d_type=None):
        if USE_CUDA:
            if d_type is not None:
                return torch.as_tensor(x, dtype=d_type).cuda()
            return torch.Tensor(x).cuda()
        else:
            if d_type is not None:
                return torch.as_tensor(x, dtype=d_type)
            return torch.Tensor(x)

    def construct_graph(self, batch_graph_list: list, node_nums: list):
        """
                Construct graph for batch_graph_list
                return [list of [batched DGLGraph for an edge]]
        """
        edge_dgl_graph, batched_dgl_graph = defaultdict(list), {}
        for i, graph in enumerate(batch_graph_list):
            node_num = node_nums[i]
            for edge_type, edges in graph.items():
                u = self._cuda(np.concatenate([edges[0], edges[1]]), d_type=torch.int64)
                v = self._cuda(np.concatenate([edges[1], edges[0]]), d_type=torch.int64)
                # ignore the last empty node
                g = dgl.graph((u, v), num_nodes=node_num - 1)
                # g.add_nodes(node_num - 1)
                # g.add_edges(u, v)
                g = dgl.add_self_loop(g)
                # if USE_CUDA:
                #     g = g.to("cuda:0")
                edge_dgl_graph[edge_type].append(g)
        for edge_type, edge_list in edge_dgl_graph.items():
            batched_dgl_graph[edge_type] = dgl.batch(edge_list)
        return batched_dgl_graph

    def match_token(self, hiddens, span, seq_lens):
        max_len = max(seq_lens)
        # based on offset mappings, select hidden state for each token in the utterance
        hiddens_span = torch.zeros(hiddens.size(0), max_len, hiddens.size(2)).to(hiddens)
        for i in range(len(span)):
            for idx, span_i in enumerate(span[i]):
                hiddens_span[i][idx] = hiddens[i][span_i]
        return hiddens_span

    def get_node_representations(self, nodes):
        """
        Not Batch Version: Get node representations
        """
        node_texts_list, node_lengths_list, node_element_idx = nodes[:3]
        node_sent_rep_list = []
        if self.args.use_pretrained and not self.args.disable_node_pretrained:
            tokenized_list, tokenized_lengths_list = nodes[3:]
            for tokenized, tokenized_lengths in zip(tokenized_list, tokenized_lengths_list):
                _, sent_rep = self.node_plm_encoder(tokenized, seq_lens=tokenized_lengths)
                node_sent_rep_list.append(sent_rep)
        else:
            for node_texts, node_lengths in zip(node_texts_list, node_lengths_list):
                # sent_rep = torch.zeros(node_texts.shape[0], self.args.node_encoder_hidden_dim + self.args.node_attention_output_dim, requires_grad=True).to(node_texts.device)
                _, sent_rep = self.node_combined_encoder(node_texts, node_lengths, enforce_sorted=True)
                node_sent_rep_list.append(sent_rep)
        node_sent_rep = torch.cat(node_sent_rep_list, dim=0)
        # node_hiddens: [num_nodes, MEM_TOKEN_SIZE or 1, hidden_dim]
        node_hiddens = torch.index_select(node_sent_rep, 0, node_element_idx.view(-1)).view(*node_element_idx.shape, -1)
        if node_element_idx.size(1) == 1:
            node_hiddens.squeeze_(dim=1)
        else:
            node_hiddens.masked_fill_(node_element_idx.unsqueeze(-1) == 0, 0)
            node_hiddens = node_hiddens.sum(dim=-2) / node_element_idx.count_nonzero(dim=-1).unsqueeze(-1)
        return node_hiddens

    def init_node_tensors(self, batch_data, text_rep):
        # Initialize node tensors for graph neural network
        hiddens, sent_rep = text_rep
        batch_size = sent_rep.size(0)

        max_node_num = max(batch_data["node_num"])
        text_lengths = batch_data["text_lengths"]
        linear_out_features_list = []

        node_tensors = []

        kg_nodes = batch_data["kg_nodes"]
        sp_nodes = batch_data["sp_nodes"]
        for i in range(batch_size):
            profile_span = batch_data["profile_span"][i]

            num_node = batch_data["node_num"][i]
            node_flag = np.array([0] * num_node)
            node_tensors.append([None] * num_node)

            # add utterance representation
            utter_spans = profile_span["utter"]
            st, ed = utter_spans[0]
            assert len(utter_spans) == 1 and ed - st >= 1, f"Utterance span '{utter_spans}' is not correct."
            node_tensors[i][st] = sent_rep[i]
            if ed - st > 1:
                node_tensors[i][st + 1:ed] = hiddens[i][:text_lengths[i]]
            node_flag[st:ed] += 1

            if "kg" in profile_span:
                # encode kg nodes
                if len(kg_nodes[i]) == 1:
                    # [num_kg_nodes, MEM_TOKEN_SIZE]
                    kg_inputs = kg_nodes[i][0]
                    kg_hiddens = self.node_combined_encoder.embedding(kg_inputs)
                    kg_hiddens.masked_fill_(kg_inputs.unsqueeze(-1) == 0, 0)
                    kg_hiddens = kg_hiddens.sum(dim=-2) / kg_inputs.count_nonzero(dim=-1).unsqueeze(-1)
                else:
                    # [num_kg_nodes, hidden_dim]
                    kg_hiddens = self.get_node_representations(kg_nodes[i])
                count = 0
                for st, ed in profile_span["kg"]:
                    node_tensors[i][st:ed] = kg_hiddens[count:count + ed - st]
                    node_flag[st:ed] += 1
                    count += ed - st

            # encode special nodes, e.g. up, ca
            if len(sp_nodes[i]) == 1:
                # [num_sp_nodes, MEM_TOKEN_SIZE]
                sp_inputs = sp_nodes[i][0]
                sp_hiddens = self.node_combined_encoder.embedding(sp_inputs)
                sp_hiddens.masked_fill_(sp_inputs.unsqueeze(-1) == 0, 0)
                sp_hiddens = sp_hiddens.sum(dim=-2) / sp_inputs.count_nonzero(dim=-1).unsqueeze(-1)
            else:
                # [num_sp_nodes, hidden_dim]
                sp_hiddens = self.get_node_representations(sp_nodes[i])
            count = 0
            for st, ed in profile_span["up"] + profile_span["ca"]:
                node_tensors[i][st:ed] = sp_hiddens[count:count + ed - st]
                node_flag[st:ed] += 1
                count += ed - st

            # empty node
            node_flag[-1] += 1

            assert np.all(node_flag == 1), f"Node tensors are not initialized correctly: {node_flag}"

            # dropout and then apply different linear layers for different types of nodes
            # linear_out_features = torch.zeros(max_node_num, self.args.profile_info_dim, requires_grad=True).to(sent_rep)
            linear_out_features = self.profile_graph.node_linear(
                node_tensors[i], (max_node_num, self.args.profile_info_dim), profile_span)
            # empty node
            linear_out_features[num_node - 1] = self.null_emb
            linear_out_features_list.append(linear_out_features)

        linear_out_features = torch.stack(linear_out_features_list, dim=0)

        return linear_out_features

    def forward(self, batch_data, slot_forcing=False, n_predicts=None):
        seq_lens = batch_data['text_lengths']
        # encode user utterances: [BSZ, max_seq_len, hidden_dim]
        if self.args.use_pretrained:
            hiddens, sent_rep = self.encoder(batch_data['text_tokenized'], seq_lens=seq_lens)
            hiddens = self.match_token(hiddens, batch_data["text_tokenized_spans"], seq_lens)
        else:
            hiddens, sent_rep = self.encoder(batch_data['text'], seq_lens, enforce_sorted=True, total_length=None)

        info_emb_slot = None
        node_logits_slot, node_logits_intent = None, None
        if self.args.use_info:
            node_nums = batch_data['node_num']
            # initialize node tensors for graph neural network: [batch_size, max_node_num, profile_info_dim]
            linear_out_features = self.init_node_tensors(batch_data, (hiddens, sent_rep))
            # each element is a batched dgl graph belonging to one edge type
            graph_edges = self.construct_graph(batch_data['graph_edges'], node_nums)
            # skip gnn for measuring the time cost of other components
            # self.profile_graph.graph_out_features = linear_out_features
            # GNN is not computation-time bottleneck
            self.profile_graph.load_memory(linear_out_features, graph_edges, node_nums)

            if self.args.info_utilization_way == "node_weighting":
                # when num_supporting_node == 0, only use leaf nodes to compute info_emb_intent and info_emb_slot
                # extra info for intent detection: [BSZ, 1, profile_info_dim], [BSZ, 1, num_nodes], [BSZ, 1, num_nodes]
                info_emb_intent, prob_soft_intent, prob_logits_intent = self.profile_graph(
                    self.query_projector(sent_rep).unsqueeze(1),
                    node_nums=node_nums, num_supporting_node=self.args.num_supporting_node,
                    leaf_node_ids=batch_data["leaf_node_ids"])
                # extra info for slot filling: [BSZ, SL, profile_info_dim], [BSZ, SL, num_nodes], [BSZ, SL, num_nodes]
                if self.args.sentence_level_node_weighting:
                    info_emb_slot = info_emb_intent.expand(-1, hiddens.size(1), -1)
                    prob_soft_slot = prob_soft_intent.expand(-1, hiddens.size(1), -1)
                    prob_logits_slot = prob_logits_intent.expand(-1, hiddens.size(1), -1)
                else:
                    info_emb_slot, prob_soft_slot, prob_logits_slot = self.profile_graph(
                        self.query_projector(hiddens),
                        node_nums=node_nums, num_supporting_node=self.args.num_supporting_node,
                        leaf_node_ids=batch_data["leaf_node_ids"])
                # address node logits and probs
                prob_soft_intent = prob_soft_intent.squeeze(1)
                prob_logits_intent = prob_logits_intent.squeeze(1)
                if n_predicts is None:
                    node_logits_slot = [prob_logits_slot, prob_soft_slot]
                    node_logits_intent = [prob_logits_intent, prob_soft_intent]
                else:
                    node_logits_slot = [prob_logits_slot.cpu().data, prob_soft_slot.cpu().data]
                    node_logits_intent = [prob_logits_intent.cpu().data, prob_soft_intent.cpu().data]
            elif self.args.info_utilization_way == "node_aggregation":
                info_emb_intent = self.profile_graph.graph_out_features[
                    torch.arange(sent_rep.size(0)), batch_data["text_node_ids"]]
                if self.args.enable_character_node:
                    # extract each character representation in an utterance from graph_out_features
                    info_emb_slot = torch.gather(
                        self.profile_graph.graph_out_features, 1, batch_data["char_node_ids"].unsqueeze(-1).expand(
                            -1, -1, self.profile_graph.graph_out_features.size(-1)))
                else:
                    info_emb_slot = info_emb_intent.unsqueeze(1).expand(-1, hiddens.size(1), -1)
            else:
                raise ValueError(f"Information utilization way '{self.args.info_utilization_way}' is not supported.")

            # TODO: concatenate info for slu: averaged node representations
            sent_rep = torch.cat([sent_rep, info_emb_intent.squeeze(1)], dim=-1)

        # intent detection
        pred_intent = self.intent_decoder(sent_rep)
        if not self.args.differentiable:
            _, idx_intent = pred_intent.topk(1, dim=-1)
            feed_intent = self.intent_embedding(idx_intent.squeeze(1))
        else:
            feed_intent = pred_intent
        feed_intent = feed_intent.unsqueeze(1).repeat(1, hiddens.size(1), 1)

        # slot filling
        pred_slot = self.slot_decoder(
            hiddens, seq_lens,
            extra_input=feed_intent,
            forced_input=batch_data["slots"] if slot_forcing else None,
            info_emb=info_emb_slot
        )

        if n_predicts is None:
            return F.log_softmax(pred_slot, dim=1), F.log_softmax(pred_intent, dim=1), \
                node_logits_slot, node_logits_intent
        else:
            _, slot_index = pred_slot.topk(n_predicts, dim=1)
            _, intent_index = pred_intent.topk(n_predicts, dim=1)
            return slot_index.cpu().data.numpy().tolist(), intent_index.cpu().data.numpy().tolist(), \
                node_logits_slot, node_logits_intent