# coding=utf-8
# @Author: dcteng
# @Desc: { 模块描述 }
# @Date: 2023/07/25

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import lr_scheduler
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence
from transformers import ElectraModel, ElectraConfig
from transformers import XLNetModel, XLNetConfig

import dgl
from dgl.nn.pytorch import GATConv, GATv2Conv

from utils.config import *


class ElectraEncoder(nn.Module):
    def __init__(self, dropout_rate):
        super(ElectraEncoder, self).__init__()
        model_config = ElectraConfig.from_pretrained(args.model_type_path)
        self.model = ElectraModel.from_pretrained(args.model_type_path, config=model_config)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, input, seq_lens=None):
        outputs = self.model(input, attention_mask=(input != 0).float())
        last_hidden = self.dropout(outputs.last_hidden_state)

        hiddens = last_hidden[:, 1:-1]
        sent_rep = last_hidden[:, 0]
        return hiddens, sent_rep


class XLNetEncoder(nn.Module):
    def __init__(self, dropout_rate):
        super(XLNetEncoder, self).__init__()
        model_config = XLNetConfig.from_pretrained(args.model_type_path)
        self.model = XLNetModel.from_pretrained(args.model_type_path, config=model_config)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, input, seq_lens=None):
        outputs = self.model(input, attention_mask=(input != 0).float())
        last_hidden = self.dropout(outputs.last_hidden_state)

        if seq_lens is not None:
            # XLNet regular tokenization: input_ids + sep_token + cls_token
            hiddens = last_hidden[:, :-2]
            sent_rep_list = []
            for i in range(len(seq_lens)):
                sent_rep_list.append(last_hidden[i][seq_lens[i] + 1])
            sent_rep = torch.stack(sent_rep_list)
        else:
            hiddens = last_hidden[:, 1:-1]
            sent_rep = last_hidden[:, 0]
        return hiddens, sent_rep


class CombinedEncoder(nn.Module):
    def __init__(self, num_word, word_embedding_dim, encoder_hidden_dim, attention_hidden_dim, attention_output_dim,
                 dropout_rate, n_layers=1):
        super(CombinedEncoder, self).__init__()
        self.embedding = nn.Embedding(num_word, word_embedding_dim)
        self.encoder = LSTMEncoder(word_embedding_dim, encoder_hidden_dim, dropout_rate)
        self.attention = SelfAttention(word_embedding_dim, attention_hidden_dim, attention_output_dim, dropout_rate)
        self.sent_attention = UnflatSelfAttention(encoder_hidden_dim + attention_output_dim, dropout_rate)

    def forward(self, input_ids, input_lengths, enforce_sorted=True, total_length=None):
        word_tensor = self.embedding(input_ids)
        encoder_hiddens = self.encoder(
            word_tensor, input_lengths, enforce_sorted=enforce_sorted, total_length=total_length)
        attention_hiddens = self.attention(word_tensor, input_lengths)
        cat_hiddens = torch.cat([encoder_hiddens, attention_hiddens], dim=-1)
        sent_rep = self.sent_attention(cat_hiddens, input_lengths)
        return cat_hiddens, sent_rep


class LSTMEncoder(nn.Module):
    """
    Encoder structure based on bidirectional LSTM.
    """

    def __init__(self, embedding_dim, hidden_dim, dropout_rate):
        super(LSTMEncoder, self).__init__()

        # Parameter recording.
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim // 2
        self.dropout_rate = dropout_rate

        # Network attributes.
        self.dropout_layer = nn.Dropout(self.dropout_rate)
        self.lstm_layer = nn.LSTM(
            input_size=self.embedding_dim,
            hidden_size=self.hidden_dim,
            batch_first=True,
            bidirectional=True,
            # dropout=self.dropout_rate,
            dropout=0,
            num_layers=1
        )

    def forward(self, embedded_text, seq_lens, enforce_sorted=True, total_length=None):
        """ Forward process for LSTM Encoder.
        (Passed) Test unsorted sequences for pack_padded_sequence and pad_packed_sequence

        (batch_size, max_sent_len)
        -> (batch_size, max_sent_len, word_dim)
        -> (batch_size, max_sent_len, hidden_dim)
        -> (total_word_num, hidden_dim)

        :param embedded_text: padded and embedded input text.
        :param seq_lens: is the length of original input text.
        :return: is encoded word hidden vectors.
        """

        # Padded_text should be instance of LongTensor.
        dropout_text = self.dropout_layer(embedded_text)

        # Pack and Pad process for input of variable length.
        packed_text = pack_padded_sequence(
            dropout_text, seq_lens, batch_first=True, enforce_sorted=enforce_sorted)
        lstm_hiddens, (h_last, c_last) = self.lstm_layer(packed_text)
        padded_hiddens, _ = pad_packed_sequence(lstm_hiddens, batch_first=True, total_length=total_length)
        return padded_hiddens


class QKVAttention(nn.Module):
    """
    Attention mechanism based on Query-Key-Value architecture. And
    especially, when query == key == value, it's self-attention.
    """

    def __init__(self, query_dim, key_dim, value_dim, hidden_dim, output_dim, dropout_rate):
        super(QKVAttention, self).__init__()

        # Record hyper-parameters.
        self.query_dim = query_dim
        self.key_dim = key_dim
        self.value_dim = value_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.dropout_rate = dropout_rate

        # Declare network structures.
        self.query_layer = nn.Linear(self.query_dim, self.hidden_dim)
        self.key_layer = nn.Linear(self.key_dim, self.hidden_dim)
        self.value_layer = nn.Linear(self.value_dim, self.output_dim)
        self.dropout_layer = nn.Dropout(p=self.dropout_rate)

    def forward(self, input_query, input_key, input_value, seq_lens=None):
        """ The forward propagation of attention.

        Here we require the first dimension of input key
        and value are equal.

        :param input_query: is query tensor, (n, d_q)
        :param input_key:  is key tensor, (m, d_k)
        :param input_value:  is value tensor, (m, d_v)
        :return: attention based tensor, (n, d_h)
        """

        # Linear transform to fine-tune dimension.
        linear_query = self.query_layer(input_query)
        linear_key = self.key_layer(input_key)
        linear_value = self.value_layer(input_value)

        score_tensor = torch.matmul(
            linear_query,
            linear_key.transpose(-2, -1)
        ) / math.sqrt(self.hidden_dim)

        if seq_lens is not None:
            # max_len = max(seq_lens)
            # for i, l in enumerate(seq_lens):
            #     if l < max_len:
            #         score_tensor.data[i, l:] = -1e9

            attention_mask = get_attention_mask(
                seq_lens, score_tensor.dtype, score_tensor.device, tgt_len=score_tensor.shape[1])
            assert attention_mask.shape == score_tensor.shape, f"{attention_mask.shape} != {score_tensor.shape}"
            score_tensor += attention_mask

        score_tensor = F.softmax(score_tensor, dim=-1)

        forced_tensor = torch.matmul(score_tensor, linear_value)
        forced_tensor = self.dropout_layer(forced_tensor)

        return forced_tensor


class SelfAttention(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim, dropout_rate):
        super(SelfAttention, self).__init__()

        # Record parameters.
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.dropout_rate = dropout_rate

        # Record network parameters.
        self.dropout_layer = nn.Dropout(self.dropout_rate)
        self.attention_layer = QKVAttention(
            self.input_dim, self.input_dim, self.input_dim,
            self.hidden_dim, self.output_dim, self.dropout_rate
        )

    def forward(self, input_x, seq_lens):
        dropout_x = self.dropout_layer(input_x)
        attention_x = self.attention_layer(
            dropout_x, dropout_x, dropout_x, seq_lens
        )
        return attention_x


class UnflatSelfAttention(nn.Module):
    """
    scores each element of the sequence with a linear layer and uses the normalized scores to compute a context over the sequence.
    """

    def __init__(self, d_hid, dropout=0.):
        super().__init__()
        self.scorer = nn.Linear(d_hid, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, inp, lens):
        batch_size, seq_len, d_feat = inp.size()
        inp = self.dropout(inp)
        scores = self.scorer(inp.contiguous().view(-1, d_feat)).view(batch_size, seq_len)

        # max_len = max(lens)
        # for i, l in enumerate(lens):
        #     if l < max_len:
        #         scores.data[i, l:] = -1e9
        attention_mask = get_attention_mask(lens, scores.dtype, scores.device, tgt_len=None)
        assert attention_mask.shape == scores.shape
        scores += attention_mask

        scores = F.softmax(scores, dim=1)
        context = scores.unsqueeze(2).expand_as(inp).mul(inp).sum(1)
        return context


class AttrProxy(object):

    def __init__(self, module, prefix):
        self.module = module
        self.prefix = prefix

    def __getitem__(self, i):
        return getattr(self.module, self.prefix + str(i))


class GraphPF(nn.Module):
    def __init__(self, embedding_dim, dropout,
                 edge_types, graph_hidden_size, graph_out_size, graph_heads, profile_type2dims):
        super(GraphPF, self).__init__()

        self.embedding_dim = embedding_dim
        self.dropout = dropout
        self.dropout_layer = nn.Dropout(self.dropout)

        self.HAN = HAN(edge_types, embedding_dim, graph_hidden_size, graph_out_size, graph_heads, self.dropout)

        self.sigmoid = nn.Sigmoid()

        for profile_type, (input_dim, output_dim) in profile_type2dims.items():
            node_li = nn.Linear(input_dim, output_dim)
            self.add_module(f"node_{profile_type}", node_li)
        self.node_lis = AttrProxy(self, "node_")

    def node_linear(self, feat_in_list, shapes, span: dict):
        """
                dropout and then apply different linear layers for different types of nodes
                not for batch version
                domain_span: dict. dict[key=profile_type] = [value=tuple(start_idx, end_idx)]
                node_num * hidden
        """
        feat_out = torch.zeros(shapes).to(feat_in_list[0])
        for profile_type, sp in span.items():
            node_list = []
            for st, ed in sp:
                node_list.extend(feat_in_list[st: ed])
            lin_in = torch.stack(node_list, dim=0)
            lin_out = self.node_lis[profile_type](self.dropout_layer(lin_in))
            out_bi = 0
            for st, ed in sp:
                feat_out[st: ed] = lin_out[out_bi: out_bi + ed - st]
                out_bi += ed - st
        return feat_out

    def load_memory(self, features, graphs, node_nums):
        batch_size = features.shape[0]
        node_nums = [num - 1 for num in node_nums]
        # NOTE: whether need to use features.clone()?
        self.graph_out_features = features.clone()
        linear_out_features = []
        for i in range(batch_size):
            # ignore the last empty node
            linear_out_features.append(features[i][:node_nums[i]])

        graph_out = self.HAN(graphs, torch.cat(linear_out_features), node_nums)
        begin = 0
        for i in range(batch_size):
            num_valid_node = node_nums[i]
            self.graph_out_features[i][:num_valid_node] = graph_out[begin:begin + num_valid_node]
            begin += num_valid_node
        return

    def forward(self, query_vector, node_nums, num_supporting_node=-1, leaf_node_ids=None):
        m_A = self.graph_out_features  # (B, N, D)

        # u = [query_vector]
        # if len(list(u[-1].size())) == 1:
        #     u[-1] = u[-1].unsqueeze(0)
        # u_temp = u[-1].unsqueeze(1).expand_as(m_A)  # (B, 1, D) -> (B, N, D)
        # prob_logits = torch.sum(m_A * u_temp, 2)  # (B, N, D) * (B, N, D) -> (B, N)

        # query_vector: (B, Q, D)
        assert len(list(query_vector.size())) == 3
        batch_size, query_num, _ = query_vector.size()

        prob_logits = torch.matmul(query_vector, m_A.transpose(-1, -2))  # (B, Q, D) * (B, D, N) -> (B, Q, N)

        empty_supporting_flag = None
        if num_supporting_node == -1:
            attention_mask = get_attention_mask(
                node_nums, prob_logits.dtype, prob_logits.device, tgt_len=prob_logits.shape[1])
            assert attention_mask.shape == prob_logits.shape
        elif num_supporting_node == 0:
            # only use leaf nodes to compute info_emb_intent and info_emb_slot
            assert leaf_node_ids is not None
            prob_mask = torch.ones_like(prob_logits, dtype=torch.bool)
            leaf_node_ids = leaf_node_ids.unsqueeze(1).expand(-1, query_num, -1)  # (B, Q, K)
            prob_mask.scatter_(dim=-1, index=leaf_node_ids, value=False)
            # print((prob_mask == False).sum(-1))
            # exit(0)
            attention_mask = prob_mask.to(prob_logits.dtype).masked_fill(prob_mask, torch.finfo(prob_logits.dtype).min)
        else:
            search_num = min(num_supporting_node, min(node_nums))
            _, top_pi = prob_logits.data.topk(search_num)  # [B, Q, K], [B, Q, K]
            # The last node is empty node, nodes after which should be ignored
            # for i in range(batch_size):
            #     for j in range(query_num):
            #         for k in range(search_num):
            #             if top_pi[i][j][k] < node_nums[i]:
            #                 prob_mask[i][j][top_pi[i][j][k]] = 1
            #             else:
            #                 break
            valid_node_counts = torch.LongTensor(node_nums).to(prob_logits.device) - 1
            # get top K valid nodes for each query in each batch
            _, top_pi = prob_logits.data.topk(search_num)  # [B, Q, K], [B, Q, K]
            # determine whether the top K nodes are valid
            invalid_mask = top_pi >= valid_node_counts.unsqueeze(1).unsqueeze(2)
            if remove_node_after_null:
                # get the index of the first invalid node for each query in each batch
                first_invalid_index = torch.where(invalid_mask.cumsum(dim=-1) == 1, 1, 0).argmax(dim=-1, keepdim=True)
                # generate a mask for the top K nodes, set the value of the top K nodes after the first invalid node to True
                seq_indices = torch.arange(search_num)[None, None, :].expand_as(top_pi).to(prob_logits.device)
                invalid_mask = seq_indices >= first_invalid_index

            # create full mask
            prob_mask = torch.ones_like(prob_logits, dtype=torch.bool)
            # scatter the invalid mask to the full mask
            prob_mask.scatter_(dim=-1, index=top_pi, src=invalid_mask)
            # convert the full mask to attention mask
            attention_mask = prob_mask.to(prob_logits.dtype).masked_fill(prob_mask, torch.finfo(prob_logits.dtype).min)
            empty_supporting_flag = prob_mask.all(dim=-1)  # (B, Q)

        # print(prob_logits.isnan().any())
        # print(m_A.isnan().any())

        masked_logits = prob_logits + attention_mask
        prob_soft = F.softmax(masked_logits, dim=-1)    # (B, Q, N)
        res_vector = torch.matmul(prob_soft, m_A)  # (B, Q, N) * (B, N, D) -> (B, Q, D)
        # TODO: when supporting node is empty, how to deal with it?
        if empty_supporting_flag is not None:
            res_vector[empty_supporting_flag] = 0
            # print(empty_supporting_flag)

        return res_vector, prob_soft, prob_logits


class HAN(nn.Module):
    def __init__(self, edge_types, in_size, hidden_size, out_size, graph_heads, dropout):
        super(HAN, self).__init__()

        self.layers = nn.ModuleList()
        self.layers.append(HANLayer(edge_types, in_size, hidden_size, graph_heads[0], dropout))
        for layer in range(1, len(graph_heads)):
            self.layers.append(HANLayer(edge_types, hidden_size * graph_heads[layer - 1],
                                        hidden_size, graph_heads[layer], dropout))
        self.Linear = nn.Linear(hidden_size * graph_heads[-1], out_size)

    def forward(self, homo_graphs, h, node_nums):
        for gnn in self.layers:
            h = gnn(homo_graphs, h, node_nums)

        return self.Linear(h)


class HANLayer(nn.Module):
    """
    HAN layer.

    Arguments
    ---------
    edge_types : homogeneous graphs generated from the different types of edges.
    in_size : input feature dimension
    out_size : output feature dimension
    layer_num_heads : number of attention heads
    dropout : Dropout probability

    """

    def __init__(self, edge_types, in_size, out_size, layer_num_heads, dropout):
        super(HANLayer, self).__init__()

        self.gat_layers = nn.ModuleDict()
        for edge_type in edge_types:
            self.gat_layers[edge_type] = GATv2Conv(in_size, out_size, layer_num_heads, dropout, dropout, activation=F.elu)
        self.semantic_attention = SemanticAttention(in_size=out_size * layer_num_heads)
        self.edge_types = edge_types

    def forward(self, homo_graphs, h, node_nums):
        """
        Inputs
        ------
        homo_graphs : Dict[key=edge_type] = [value=batched_homogenous_graphs]
        h : tensor
            Input features(batched)

        Outputs
        -------
        tensor
        The output feature
        """
        semantic_embeddings = []
        for edge_type, graph in homo_graphs.items():
            # [NumNodes, NumHeads, D] -> [NumNodes, NumHeads * D]
            semantic_embeddings.append(self.gat_layers[edge_type](graph, h).flatten(1))
        # [NumNodes, NumEdgeTypes, NumHeads * D]
        semantic_embeddings = torch.stack(semantic_embeddings, dim=1)

        return self.semantic_attention(semantic_embeddings, node_nums)


class SemanticAttention(nn.Module):
    def __init__(self, in_size, hidden_size=128):
        super(SemanticAttention, self).__init__()

        self.project = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False)
        )

    def forward(self, z, node_nums):
        # N = number of batched nodes, M = number of edge types
        if homogeneous_aggregation_strategy == "batch_weight":
            # TODO: Why do this?
            w = self.project(z).mean(0)  # (M, 1)
            beta = torch.softmax(w, dim=0)  # (M, 1)
            beta = beta.expand((z.shape[0],) + beta.shape)  # (N, M, 1)
        elif homogeneous_aggregation_strategy == "individual_weight":
            # TODO: why not do this?
            # z: (N, M, D)
            w = self.project(z)     # (N, M, 1)
            beta = torch.softmax(w, dim=1)  # (N, M, 1)
        elif homogeneous_aggregation_strategy == "sample_weight":
            assert sum(node_nums) == z.shape[0]
            w = self.project(z)     # (N, M, 1)
            beta_list = []
            count = 0
            for i, num in enumerate(node_nums):
                # (num, M, 1) -> (M, 1) -> (num, M, 1)
                beta_list.append(torch.softmax(w[count: count + num].mean(0), dim=0).expand((num,) + w.shape[1:]))
                count += num
            beta = torch.cat(beta_list, dim=0)  # (N, M, 1)
        elif homogeneous_aggregation_strategy == "sum":
            beta = 1
        elif homogeneous_aggregation_strategy == "mean":
            beta = 1 / z.shape[1]
        else:
            raise ValueError(f"Unknown homogeneous aggregation strategy: {homogeneous_aggregation_strategy}")

        return (beta * z).sum(1)  # (N, D)


class LSTMDecoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout_rate, embedding_dim=None, extra_dim=None,
                 info_dim=None, args=None):
        super(LSTMDecoder, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.dropout_rate = dropout_rate
        self.embedding_dim = embedding_dim
        self.extra_dim = extra_dim
        self.args = args

        # If embedding_dim is not None, the output and input
        # of this structure is relevant.
        if self.embedding_dim is not None:
            self.embedding_layer = nn.Embedding(output_dim, embedding_dim)
            self.init_tensor = nn.Parameter(
                torch.randn(1, self.embedding_dim),
                requires_grad=True
            )

        # Make sure the input dimension of iterative LSTM.
        lstm_input_dim = self.input_dim
        if self.extra_dim is not None:
            lstm_input_dim += self.extra_dim
        if self.embedding_dim is not None:
            lstm_input_dim += self.embedding_dim
        if info_dim != 0:
            lstm_input_dim += info_dim

        # Network parameter definition.
        self.dropout_layer = nn.Dropout(self.dropout_rate)
        self.lstm_layer = nn.LSTM(
            input_size=lstm_input_dim,
            hidden_size=self.hidden_dim,
            batch_first=True,
            bidirectional=False,
            dropout=self.dropout_rate,
            num_layers=1
        )

        self.linear_layer = nn.Linear(
            self.hidden_dim,
            self.output_dim
        )

    def _cuda(self, x):
        if args.gpu:
            return x.cuda()
        else:
            return x

    def forward(self, encoded_hiddens, seq_lens, extra_input=None, forced_input=None, info_emb=None):
        # Concatenate information tensor if possible.
        if info_emb is not None:
            encoded_hiddens = torch.cat([encoded_hiddens, info_emb], dim=-1)
        if extra_input is not None:
            input_tensor = torch.cat([encoded_hiddens, extra_input], dim=-1)
        else:
            input_tensor = encoded_hiddens
        output_tensor_list = []
        if self.embedding_dim is not None and forced_input is not None:
            forced_tensor = self.embedding_layer(forced_input)[:, :-1]
            prev_tensor = torch.cat((self.init_tensor.unsqueeze(0).repeat(len(forced_tensor), 1, 1),
                                     forced_tensor), dim=1)
            combined_input = torch.cat([input_tensor, prev_tensor], dim=2)
            dropout_input = self.dropout_layer(combined_input)
            packed_input = pack_padded_sequence(dropout_input, seq_lens, batch_first=True)
            lstm_out, _ = self.lstm_layer(packed_input)
            lstm_out, _ = pad_packed_sequence(lstm_out, batch_first=True)
            # flatten output
            for sent_i in range(0, len(seq_lens)):
                lstm_out_i = lstm_out[sent_i][:seq_lens[sent_i]]
                linear_out = self.linear_layer(lstm_out_i)
                output_tensor_list.append(linear_out)
        else:
            prev_tensor = self.init_tensor.unsqueeze(0).repeat(len(seq_lens), 1, 1)
            last_h, last_c = None, None
            for word_i in range(seq_lens[0]):
                combined_input = torch.cat((input_tensor[:, word_i].unsqueeze(1), prev_tensor), dim=2)
                dropout_input = self.dropout_layer(combined_input)
                if last_h is None and last_c is None:
                    lstm_out, (last_h, last_c) = self.lstm_layer(dropout_input)
                else:
                    lstm_out, (last_h, last_c) = self.lstm_layer(dropout_input, (last_h, last_c))
                lstm_out = self.linear_layer(lstm_out.squeeze(1))
                output_tensor_list.append(lstm_out)

                _, index = lstm_out.topk(1, dim=1)
                prev_tensor = self.embedding_layer(index.squeeze(1)).unsqueeze(1)
            # flatten output
            output_tensor = torch.stack(output_tensor_list)
            output_tensor_list = [output_tensor[:length, i] for i, length in enumerate(seq_lens)]
        return torch.cat(output_tensor_list, dim=0)


def get_attention_mask(seq_lens, dtype, device, tgt_len=None):
    # get attention mask for padded sequence, and fill the padded part with -inf
    max_len = max(seq_lens)
    mask = torch.zeros((len(seq_lens), max_len))
    for i, l in enumerate(seq_lens):
        mask[i, :l] = 1

    if tgt_len is not None:
        # expands attention mask from (batch_size, seq_len) to (batch_size, tgt_len, seq_len)
        mask = mask.unsqueeze(1).expand(-1, tgt_len, -1).to(dtype)

    inverted_mask = 1.0 - mask

    return inverted_mask.masked_fill(inverted_mask.to(torch.bool), torch.finfo(dtype).min).to(device)