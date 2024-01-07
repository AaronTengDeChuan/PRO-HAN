# coding=utf-8
# @Author: dcteng
# @Desc: { 模块描述 }
# @Date: 2023/07/18 

import os
import json
import copy
from functools import reduce

import numpy as np
from copy import deepcopy
from collections import defaultdict, Counter

if __name__ != '__main__':
    from utils.config import *


class ProGraph(object):
    order2desc = {
        0: "第一偏好",
        1: "第二偏好",
        2: "第三偏好",
    }

    profile_types = ['utter', 'kg', 'up', 'ca']

    def __init__(self, args, pad_token, unk_token):
        # parameter for nodes in graph
        self.kg_global_node = args.kg_global_node
        self.up_global_node = args.up_global_node
        self.ca_global_node = args.ca_global_node
        self.up_preference_global_node = args.up_preference_global_node

        assert self.up_global_node or self.up_preference_global_node, "Global node(s) for UP must be set"

        # parameter for edges in graph
        self.homogeneous_graph = args.homogeneous_graph
        # when some type of edges are unavailable, ignore them
        self.edge_types = []

        self.kg_inter_field_edge = args.kg_inter_field_edge # edges among all fields in a subject
        self.kg_intra_group_edge = args.kg_intra_group_edge # edges among all subjects in a group
        self.kg_inter_group_edge = args.kg_inter_group_edge # edges among all subjects in different groups
        self.edge_types.append('kg_intra_subject_edges')
        if any([self.kg_global_node, self.kg_intra_group_edge, self.kg_inter_group_edge]):
            self.edge_types.append('kg_outer_subject_edges')

        self.up_intra_preference_edge = args.up_intra_preference_edge # egdes among all fields in a preference
        self.up_inter_preference_edge = args.up_inter_preference_edge # edges among all preferences in a user profile
        self.edge_types.append('up_edges')

        self.ca_inter_status_edge = args.ca_inter_status_edge # edges among all statuses in a context awareness
        if any([self.ca_global_node, self.ca_inter_status_edge]):
            self.edge_types.append('ca_edges')

        self.kg_up_edge = args.kg_up_edge # edges between KG and UP
        self.kg_ca_edge = args.kg_ca_edge # edges between KG and CA
        self.up_ca_edge = args.up_ca_edge # edges between UP and CA
        if any([self.kg_up_edge, self.kg_ca_edge, self.up_ca_edge]):
            self.edge_types.append('global_edges')

        self.enable_character_node = args.enable_character_node # enable character node in utterance
        self.utter_char_edge = args.utter_char_edge # edges between user utterance and all characters
        self.utter_profile_edge = args.utter_profile_edge # edges between user utterance and all global nodes
        if self.utter_profile_edge != 'none':
            self.edge_types.append('utter_profile_edges')

        if self.homogeneous_graph:
            self.edge_types = ['homogeneous']

        self.length_limit = args.max_length

        self.args = args
        self.pad_token = pad_token
        self.unk_token = unk_token

    def create_node(self, triplet, triplet_representation, truncation=True):
        '''
        create node for each triplet
        :param triplet: [subject, predicate, object]
        :return:
        '''
        if truncation:
            # truncate the length of triplet
            if triplet_representation in ['natural', 'concat']:
                count = 0
                for i in range(len(triplet) - 1, -1, -1):
                    assert count < self.length_limit, "\n".join(f"ele ({len(ele)})" for ele in triplet)
                    count += len(triplet[i])
                    if count > self.length_limit:
                        if VERBOSE: print(f"Truncate triplet: {triplet}\n->\n\t\t", end="")
                        triplet[i] = triplet[i][:self.length_limit - count]
                        if VERBOSE: print(triplet)
            else:
                for i in range(len(triplet)):
                    if len(triplet[i]) > self.length_limit:
                        if VERBOSE: print(f"Truncate triplet: {triplet}\n->\n\t\t", end="")
                        triplet[i] = triplet[i][:self.length_limit]
                        if VERBOSE: print(triplet)

        node = []
        if triplet_representation == 'natural':
            if len(triplet) == 1:
                node = triplet
            else:
                node.append(f"'{triplet[0]}'是'{triplet[2]}'的'{triplet[1]}'")
        elif triplet_representation == 'concat':
            node.append("；".join(triplet))
        elif triplet_representation in ['encode_bow', 'embed_bow']:
            node = triplet + [self.pad_token] * (MEM_TOKEN_SIZE - len(triplet))
        else:
            raise NotImplementedError

        return node

    def add_edges(self, multi_edges, edge_type, edges):
        if self.homogeneous_graph:
            edge_type = "homogeneous"
        if edge_type not in multi_edges:
            assert len(edges) == 0, f"Edges for {edge_type} is not empty, available edges: {multi_edges.keys()}"
            return
        if len(edges) == 0:
            return
        multi_edges[edge_type].extend(edges)

    def check_duplicate_edges(self, edges, nodes):
        # assume head node id is smaller than tail node id
        edges = [tuple(sorted(edge)) for edge in edges]
        edge_counter = Counter(edges)
        for edge, count in edge_counter.items():
            if count > 1:
                print(f"Find duplicate edge: '{edge}' with count '{count}'")
                print(f"\tHead node: {nodes[edge[0]]}")
                print(f"\tTail node: {nodes[edge[1]]}")

    def get_edges_numpy(self, multi_edges):
        for edge_type in multi_edges.keys():
            edge_src = []
            edge_trg = []
            edges = multi_edges[edge_type]
            # assert len(edges) > 0, f"Edges for {edge_type} is empty"
            for edge in edges:
                edge_src.append(edge[0])
                edge_trg.append(edge[1])
            multi_edges[edge_type] = (np.array(edge_src), np.array(edge_trg))

    def print_graph(self, nodes, multi_edges):
        text_lines = []
        text_lines.append("Nodes:")
        for idx, node in enumerate(nodes):
            text_lines.append(f"\t{idx}: {node}")
        text_lines.append("\n")
        for edge_type, edges in multi_edges.items():
            text_lines.append(f"Edges for {edge_type} ({len(edges[0])}):")
            try:
                for edge_src, edge_trg in zip(edges[0], edges[1]):
                    text_lines.append(
                        f"\t{edge_src} -> {edge_trg}:\t{nodes[edge_src]} -> {nodes[edge_trg]}")
            except Exception as e:
                print(e)
                print(edges)
        return text_lines

    def create_graph(self, data):
        '''
        create graph for each user utterance
        :param data:
        :return:
        '''

        nodes = []
        multi_edges = {edge_type: [] for edge_type in self.edge_types}
        # profile span for node linear projection layer
        profile_span = {profile_type: [] for profile_type in self.profile_types}

        special_node_ids = []
        # special word list
        special_words = set()
        kg_global_id, up_global_id, ca_global_id = None, None, None

        leaf_node_ids = []

        # nodes and edges for KG
        # group by subject name
        subj_groups = defaultdict(list)
        for subj in data['KG']:
            # remove entity with too many characters
            new_subj = copy.deepcopy(subj)
            for key, value in subj.items():
                if len(value) >= 40:
                    # print(f"Remove entity of {key}: {value}")
                    new_subj.pop(key)
            subj_groups[subj['subject']].append(new_subj)
        HasKG = len(subj_groups) != 0
        kg_global_node_ids = []
        kg_intra_subject_edges = []
        kg_outer_subject_edges = []

        start_node_idx = len(nodes)

        kg_node_triplet_representation = self.args.kg_node_triplet_representation
        special_node_triplet_representation = self.args.special_node_triplet_representation

        entity_as_token = kg_node_triplet_representation == "embed_bow"

        # TODO: simplify the kg nodes: one node for each subject
        if self.kg_global_node and HasKG:
            kg_global_id = len(nodes)
            # special_node_ids.append(len(nodes))
            nodes.append(self.create_node(["知识图谱"], kg_node_triplet_representation))
            special_words.add("知识图谱")
        for _, subj_list in subj_groups.items():
            group_subject_node_ids = []
            for subj in subj_list:
                subj_name = subj['subject']
                # add global node for each subject
                group_subject_node_ids.append(len(nodes))
                nodes.append(self.create_node([subj_name], kg_node_triplet_representation))
                # add nodes for each subject
                node_start_id = len(nodes)
                for key, value in subj.items():
                    if key == 'subject':
                        continue
                    # subject node <-> field node
                    kg_intra_subject_edges.append([node_start_id - 1, len(nodes)])
                    if self.kg_inter_field_edge:
                        for i in range(node_start_id, len(nodes)):
                            # field node <-> field node
                            kg_intra_subject_edges.append([i, len(nodes)])
                    nodes.append(self.create_node([value, key, subj_name], kg_node_triplet_representation))
                    if entity_as_token:
                        special_words.update([value, key, subj_name])
            if self.kg_intra_group_edge:
                for i in range(len(group_subject_node_ids)):
                    for j in range(i + 1, len(group_subject_node_ids)):
                        # edges among subject nodes with the same subject name
                        kg_outer_subject_edges.append([group_subject_node_ids[i], group_subject_node_ids[j]])
            if self.kg_inter_group_edge:
                for i in range(len(kg_global_node_ids)):
                    for j in range(len(group_subject_node_ids)):
                        # edges among subject nodes with different subject name
                        kg_outer_subject_edges.append([kg_global_node_ids[i], group_subject_node_ids[j]])
            kg_global_node_ids.extend(group_subject_node_ids)
        if self.kg_global_node and HasKG:
            for i in range(len(kg_global_node_ids)):
                # edges between kg global node and subject global nodes
                kg_outer_subject_edges.append([kg_global_id, kg_global_node_ids[i]])
        self.add_edges(multi_edges, "kg_intra_subject_edges", kg_intra_subject_edges)
        self.add_edges(multi_edges, "kg_outer_subject_edges", kg_outer_subject_edges)
        leaf_node_ids.extend(kg_global_node_ids)

        if HasKG:
            profile_span["kg"].append((start_node_idx, len(nodes)))
            start_node_idx = len(nodes)
        else:
            profile_span.pop("kg")

        # nodes and edges for UP: up_keys = ['音视频应用偏好', '出行交通工具偏好', '长途交通工具偏好', '是否有车']
        up_global_node_ids = []
        up_edges = []
        if self.up_global_node:
            up_global_id = len(nodes)
            special_node_ids.append(len(nodes))
            nodes.append(self.create_node(["用户画像"], special_node_triplet_representation))
            special_words.add("用户画像")
        node_start_id = len(nodes)
        special_words.update(list(self.order2desc.values()))
        for key in self.args.up_keys:
            special_words.update(list(data['UP'][key].keys()) + [key, "用户画像"])
            if key == '是否有车':
                if self.up_preference_global_node:
                    up_global_node_ids.append(len(nodes))
                else:
                    up_edges.append([up_global_id, len(nodes)])
                value = sorted(data['UP'][key].items(), key=lambda x: x[1], reverse=True)[0][0]
                special_node_ids.append(len(nodes))
                leaf_node_ids.append(len(nodes))
                nodes.append(self.create_node([value, key, "用户画像"], special_node_triplet_representation))
                continue
            if self.up_preference_global_node:
                up_global_node_ids.append(len(nodes))
                special_node_ids.append(len(nodes))
                nodes.append(self.create_node([key], special_node_triplet_representation))
                node_start_id = len(nodes)
            preference_node_start_id = len(nodes)
            for order, (option, prob) in enumerate(sorted(data['UP'][key].items(), key=lambda x: x[1], reverse=True)):
                # preference node <-> field node
                up_edges.append([node_start_id - 1, len(nodes)])
                if self.up_intra_preference_edge:
                    for i in range(preference_node_start_id, len(nodes)):
                        # field node <-> field node
                        up_edges.append([i, len(nodes)])
                special_node_ids.append(len(nodes))
                leaf_node_ids.append(len(nodes))
                nodes.append(self.create_node([option, self.order2desc[order], key],
                                              special_node_triplet_representation))
        for i in range(len(up_global_node_ids)):
            if self.up_global_node:
                # edges between up global node and preference global nodes
                up_edges.append([up_global_id, up_global_node_ids[i]])
            if self.up_inter_preference_edge:
                for j in range(i + 1, len(up_global_node_ids)):
                    # edges among preference global nodes
                    up_edges.append([up_global_node_ids[i], up_global_node_ids[j]])
        self.add_edges(multi_edges, "up_edges", up_edges)
        if self.up_global_node and not self.up_preference_global_node:
            up_global_node_ids.append(up_global_id)

        profile_span["up"].append((start_node_idx, len(nodes)))
        start_node_idx = len(nodes)

        # nodes and edges for CA: ca_keys = ['移动状态', '姿态识别', '地理围栏', '户外围栏']
        ca_global_node_ids = []
        ca_edges = []
        if self.ca_global_node:
            ca_global_id = len(nodes)
            special_node_ids.append(len(nodes))
            nodes.append(self.create_node(["情境感知"], special_node_triplet_representation))
            special_words.add("情境感知")
        for key in self.args.ca_keys:
            special_words.update(list(data['CA'][key].keys()) + [key, "情境感知"])
            if self.ca_global_node:
                # edges between ca global node and status global nodes
                ca_edges.append([ca_global_id, len(nodes)])
            value = sorted(data['CA'][key].items(), key=lambda x: x[1], reverse=True)[0][0]
            if self.ca_inter_status_edge:
                for i in range(len(ca_global_node_ids)):
                    # edges among status global nodes
                    ca_edges.append([ca_global_node_ids[i], len(nodes)])
            ca_global_node_ids.append(len(nodes))
            special_node_ids.append(len(nodes))
            nodes.append(self.create_node([value, key, "情境感知"], special_node_triplet_representation))
        self.add_edges(multi_edges, "ca_edges", ca_edges)
        leaf_node_ids.extend(ca_global_node_ids)

        profile_span["ca"].append((start_node_idx, len(nodes)))
        start_node_idx = len(nodes)

        # edges between all global nodes
        global_edges = []
        kg_gns = [kg_global_id] if self.kg_global_node and HasKG else kg_global_node_ids
        up_gns = [up_global_id] if self.up_global_node else up_global_node_ids
        ca_gns = [ca_global_id] if self.ca_global_node else ca_global_node_ids
        # edges between kg global nodes and up global nodes
        if self.kg_up_edge:
            for kg_gn in kg_gns:
                for up_gn in up_gns:
                    global_edges.append([kg_gn, up_gn])
        # edges between kg global nodes and ca global nodes
        if self.kg_ca_edge:
            for kg_gn in kg_gns:
                for ca_gn in ca_gns:
                    global_edges.append([kg_gn, ca_gn])
        # edges between up global nodes and ca global nodes
        if self.up_ca_edge:
            for up_gn in up_gns:
                for ca_gn in ca_gns:
                    global_edges.append([up_gn, ca_gn])
        self.add_edges(multi_edges, "global_edges", global_edges)

        # nodes and edges for user utterance
        utterance_node_ids = [len(nodes)]
        utter_profile_edges = []
        nodes.append(self.create_node([data['text']], "natural", truncation=False))
        if self.enable_character_node:
            for char in data['text']:
                utterance_node_ids.append(len(nodes))
                nodes.append(self.create_node([char], "natural", truncation=False))
            if self.utter_char_edge:
                for un in utterance_node_ids[1:]:
                    # edges between user utterance node and character nodes
                    utter_profile_edges.append([utterance_node_ids[0], un])

        if self.utter_profile_edge == "none":
            profile_node_ids = []
        elif self.utter_profile_edge == "global":
            profile_node_ids = kg_gns + up_gns + ca_gns
        elif self.utter_profile_edge == "leaf":
            profile_node_ids = leaf_node_ids
        elif self.utter_profile_edge == "both":
            profile_node_ids = set(kg_gns + up_gns + ca_gns + leaf_node_ids)
        else:
            raise NotImplementedError
        for un in utterance_node_ids:
            for gn in profile_node_ids:
                # edges between user utterance and all global nodes
                utter_profile_edges.append([gn, un])
        self.add_edges(multi_edges, "utter_profile_edges", utter_profile_edges)

        profile_span["utter"].append((start_node_idx, len(nodes)))

        # add empty node: $$$$ is NULL token
        NULL_token = "$$$$"
        special_words.add(NULL_token)
        special_node_ids.append(len(nodes))
        nodes.append(self.create_node([NULL_token], special_node_triplet_representation))

        # check same edges in different edge types
        self.check_duplicate_edges(reduce(lambda x, y: x + y, multi_edges.values()), nodes)

        # TODO: get the number of connected components in a Graph

        # convert edges to src and trg
        self.get_edges_numpy(multi_edges)

        data["graph_nodes"] = nodes
        data["graph_edges"] = multi_edges
        data['node_num'] = len(nodes)
        data['profile_span'] = profile_span
        data["kg_entity_as_token"] = entity_as_token
        data['special_node_ids'] = special_node_ids
        data['special_words'] = list(special_words)
        data["leaf_node_ids"] = leaf_node_ids + ([len(nodes) - 1] if self.args.append_null_into_leaf else [])
        data['utterance_node_ids'] = utterance_node_ids



if __name__ == "__main__":
    import argparse
    args = argparse.Namespace()
    args.max_length = 64
    args.kg_node_triplet_representation = 'encode_bow'
    args.special_node_triplet_representation = 'embed_bow'
    args.homogeneous_graph = False
    args.kg_global_node = False
    args.up_global_node = False
    args.ca_global_node = False
    args.up_preference_global_node = True
    args.kg_inter_field_edge = False
    args.kg_intra_group_edge = False
    args.kg_inter_group_edge = False
    args.up_intra_preference_edge = False
    args.up_inter_preference_edge = False
    args.ca_inter_status_edge = False
    args.kg_up_edge = False
    args.kg_ca_edge = False
    args.up_ca_edge = False
    args.enable_character_node = False
    args.utter_char_edge = False
    args.utter_profile_edge = "both"
    args.up_keys = ['音视频应用偏好', '出行交通工具偏好', '长途交通工具偏好', '是否有车']
    args.ca_keys = ['移动状态', '姿态识别', '地理围栏', '户外围栏']

    MEM_TOKEN_SIZE = 3
    VERBOSE = False

    file_path = os.path.join("data/ProSLU", "dev_rebuild.json")
    pro_graph = ProGraph(args, '<PAD>', '<UNK>')
    file_writer = open(os.path.join("data/ProSLU", "dev_rebuild_graph.txt"), 'w', encoding="utf-8")
    with open(file_path, 'r', encoding="utf-8") as fin:
        other_info, data = json.load(fin)

        for didx, (sid, sample) in enumerate(data.items()):
            if didx < 2:
                continue
            data_detail = {
                'text': sample['用户话语'],
                'slots': sample['slot'],
                'intent': sample['intent'],
                'KG': sample['KG'],
                'UP': sample['UP'],
                'CA': sample['CA']
            }
            # create graph
            pro_graph.create_graph(data_detail)
            lines = pro_graph.print_graph(data_detail['graph_nodes'], data_detail['graph_edges'])
            file_writer.write("\n".join(lines))
            file_writer.write("\nProfile Span:\n" + json.dumps(data_detail['profile_span'], indent=4))
            file_writer.write("\nSpecial Words:\n\t" + "\n\t".join(data_detail['special_words']))
            file_writer.write("\nSpecial Node Ids:\n\t" + "\n\t".join([str(i) for i in data_detail['special_node_ids']]))
            file_writer.write(f"\nUtterance Node Ids:\n\t{data_detail['utterance_node_ids']}")
            exit(0)
    '''
            用户话语节点：sentence representation

            如何表示不同信息来源的节点：
                三元组表示方法：
                    编码器：传统RNN + 自注意力机制，PLM
                    1. 转为自然语言，然后使用编码器进行编码：
                        [音乐类，第一偏好，音视频应用偏好] -> "'音乐类'是'音视频应用偏好'的'第一偏好'"
                    2. 简单连接，然后使用编码器进行编码：
                        音乐类；第一偏好；音视频应用偏好
                    3. 使用编码器分别编码三项，然后用词袋（bag-of-words）方法融合
                    4. 使用词袋方法直接表示三元组
                是否分开表示：不同的embedding矩阵 或 不同的编码器网络？
            
            简化图：
                KG内每个subject对应一个节点
                单个节点代表UP：11维特征向量经过一个线性层映射得到高维空间
                单个节点代表CA：18维特征向量经过一个线性层映射得到高维空间
            
            是否为KG, UP和CA设立额外的全局大节点？

            图结点-知识图谱 KG（包含多个主体subject，可能存在同名主体）：
                subject全局结点：    [主体，PAD，PAD]       
                subject下的子节点：   [属性值, 属性, 主体]
                KG 内部边：
                    全局节点 <-> 其下所有子节点
                    非同名的subject全局节点间全连接

            图结点-用户画像 UP：
                音视频应用偏好 全局节点：[音视频应用偏好，PAD，PAD]
                    [音乐类，第一偏好，音视频应用偏好]
                    [视频类，第二偏好，音视频应用偏好]
                    [有声读物类，第三偏好，音视频应用偏好]
                出行交通工具偏好 全局节点：[出行交通工具偏好，PAD，PAD]
                    [驾车，第一偏好，出行交通工具偏好]
                    [公交，第二偏好，出行交通工具偏好]
                    [地铁，第三偏好，出行交通工具偏好]
                长途交通工具偏好 全局节点：[长途交通工具偏好，PAD，PAD]
                    [飞机，第一偏好，长途交通工具偏好]
                    [火车，第二偏好，长途交通工具偏好]
                    [汽车，第三偏好，长途交通工具偏好]
                有车状态 全局节点：[有/没有，有否有车，用户画像]
                UP 内部边：
                    每个全局节点 <-> 其下所有子节点
                    全局节点之间全连接

            图结点-情境感知 CA：
                情境感知全局节点：[情境感知，PAD，PAD]
                    [行走/跑步/静止/汽车/地铁/高铁/飞机/未知，移动状态，情境感知]
                    [躺卧/行走/未知，姿态识别，情境感知]
                    [家/公司/国内/未知，地理围栏，情境感知]
                    [户外/室内/未知，户外围栏，情境感知]
                CA 内部边：全局节点 <-> 其下所有子节点

            外部边：
                KG所有全局节点 <-> UP所有全局节点
                KG所有全局节点 <-> CA所有全局节点
                UP所有全局节点 <-> CA所有全局节点
                用户话语节点 <-> 所有三种类型全局节点
                
            边类型：
                类型一：
                    每个subject下节点之间的全连接
                    subject全局节点 <-> 其下所有子节点
                类型二：
                    非同名的subject全局节点间全连接
                    KG全局大节点 <-> 所有subject全局节点
                类型三：
                    每个preference下节点之间全连接
                    UP全局节点 <-> 其下所有子节点
                    UP全局节点之间的全连接
                    UP全局大节点 <-> 所有preference全局节点
                类型四：
                    每个status下节点之间全连接
                    CA全局节点 <-> 其下所有子节点
                    CA全局节点之间的全连接
                    CA全局大节点 <-> 所有status全局节点
                类型五：
                    KG全局大节点 <-> UP全局大节点
                    KG全局大节点 <-> CA全局大节点
                    UP全局大节点 <-> CA全局大节点
                    KG所有全局节点 <-> UP所有全局节点
                    KG所有全局节点 <-> CA所有全局节点
                    UP所有全局节点 <-> CA所有全局节点
                类型六：
                    用户话语节点 <-> 所有三种类型全局节点
            '''
