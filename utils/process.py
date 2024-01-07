# coding=utf-8
# @Author: dcteng
# @Desc: { 模块描述 }
# @Date: 2023/07/22
import os
import time
import random
import numpy as np
import torch.autograd
from tqdm import tqdm
from utils import miulab
from utils.config import *
from utils.data_loader import _cuda

from transformers import ElectraTokenizerFast, XLNetTokenizerFast, BertTokenizerFast

tqdm_ncols = 120

class Processor(object):
    def __init__(self, dataset, model, args):
        self.dataset = dataset
        self.model = model
        self.args = args

        if self.args.gpu:
            time_start = time.time()
            self.model = self.model.cuda()
            time_con = time.time() - time_start
            mylogger.info("The model has been loaded into GPU and cost {:.6f} seconds.\n".format(time_con))

        if self.args.use_pretrained:
            mylogger.info(f"Loading pretrained {args.model_type} tokenizer from {args.model_type_path} ...")
            if self.args.model_type == 'ELECTRA':
                self.tokenizer = ElectraTokenizerFast.from_pretrained(args.model_type_path)
            elif self.args.model_type == 'XLNet':
                self.tokenizer = XLNetTokenizerFast.from_pretrained(args.model_type_path)
            else:
                raise ValueError(f"Model type '{self.args.model_type}' is not supported now.")

        self.model.define_optimizers()
        self.criterion = torch.nn.NLLLoss()

    def save_model(self, model_file_path):
        torch.save(self.model.state_dict(), model_file_path)

    def load_model(self, model_file_path):
        file_exist = os.path.exists(model_file_path)
        if file_exist and model_file_path.endswith('.pt'):
            model_state_dict = torch.load(model_file_path)
            self.model.load_state_dict(model_state_dict)
        else:
            if not file_exist:
                model_file_path = model_file_path.rsplit('.', maxsplit=1)[0] + '.pkl'
            loaded_model = torch.load(model_file_path)
            self.model.load_state_dict(loaded_model.state_dict())

    def train(self):
        best_dev_sent = 0.0
        no_improve = 0

        # debug
        debug = True
        torch.autograd.set_detect_anomaly(True)

        train_dataloader = self.dataset.get_dataloader(
            'train', getattr(self, "tokenizer", None), char_level=not self.args.word_level_pretrained)
        for epoch in range(self.args.num_epoch):
            total_slot_loss, total_intent_loss, step = 0.0, 0.0, 0
            time_start = time.time()
            phar = tqdm(train_dataloader, desc=f"Epoch {epoch}", ncols=tqdm_ncols)
            node_nums = np.array([], dtype=np.int32)

            self.model.train()
            for batch_data in phar:
                node_nums = np.append(node_nums, batch_data["node_num"])

                if debug:
                    with open('first_batch_data.txt', 'w', encoding="utf-8") as fw:
                        fw.write(str(batch_data))
                    debug = False

                for optimizer in self.model.optimizers:
                    optimizer.zero_grad()

                random_slot = random.random()
                slot_out, intent_out, node_logits_slot, node_logits_intent = self.model(
                    batch_data, slot_forcing=random_slot < self.dataset.slot_forcing_rate)
                # calculate loss
                flatten_slot_label = torch.cat(
                    [sls[:slen] for sls, slen in zip(batch_data["slots"], batch_data["text_lengths"])], dim=0)
                slot_loss = self.criterion(slot_out, flatten_slot_label)
                intent_loss = self.criterion(intent_out, batch_data["intent"])
                batch_loss = slot_loss + intent_loss
                # TODO: add node loss

                try:
                    total_slot_loss += slot_loss.cpu().item()
                    total_intent_loss += intent_loss.cpu().item()
                except AttributeError:
                    total_slot_loss += slot_loss.cpu().data.numpy()[0]
                    total_intent_loss += intent_loss.cpu().data.numpy()[0]

                step += 1
                if step % self.args.logging_steps == 0:
                    fitlog.add_loss((total_slot_loss + total_intent_loss) / step, name='loss', step=step)
                    fitlog.add_loss(total_slot_loss / step, name='slot_loss', step=step)
                    fitlog.add_loss(total_intent_loss / step, name='intent_loss', step=step)

                # set postfix and update progress bar
                phar.set_postfix({
                    "tl": f"{(total_slot_loss + total_intent_loss) / step:.4f}",
                    "il": f"{total_intent_loss / step:.4f}",
                    "sl": f"{total_slot_loss / step:.4f}",
                    "nodes": f"{np.min(node_nums):.0f}, {np.percentile(node_nums, 25):.0f}, "
                             f"{np.median(node_nums):.0f}, {np.percentile(node_nums, 75):.0f}, {np.max(node_nums):.0f}",
                })

                # backward
                self.model.backward_loss(batch_loss)

            time_con = time.time() - time_start
            mylogger.info('[Epoch {:2d}]: The total slot loss on train data is {:2.6f}, intent loss is {:2.6f}, '
                          'cost about {:2.6f} seconds.'.format(
                epoch, total_slot_loss / step, total_intent_loss / step, time_con))

            # validate and evaluate
            time_start = time.time()
            dev_metrics = self.estimate(if_dev=True)
            dev_sent_acc = dev_metrics["dev_sent_acc"]
            fitlog.add_metric({
                "dev": dev_metrics
            }, step=epoch + 1)
            # apply lr scheduler
            self.model.apply_scheduler(dev_sent_acc)

            if dev_sent_acc > best_dev_sent:
                fitlog.add_best_metric({
                    "dev": dev_metrics
                })

                no_improve = 0
                best_dev_sent = dev_sent_acc
                test_metrics = self.estimate(if_dev=False)
                fitlog.add_metric({
                    "test": test_metrics
                }, step=epoch + 1)
                fitlog.add_best_metric({
                    "test": test_metrics
                })
                mylogger.info('\nTest result: {}.'.format(
                    ", ".join([f"{k}={v:.6f}" for k, v in test_metrics.items()])))

                # save model
                model_save_dir = os.path.join(self.dataset.save_dir, "model")
                if not os.path.exists(model_save_dir):
                    os.mkdir(model_save_dir)
                self.save_model(os.path.join(model_save_dir, ModelFileName))
                torch.save(self.dataset, os.path.join(model_save_dir, 'dataset.pkl'))

                time_con = time.time() - time_start
                mylogger.info('[Epoch {:2d}]: In validation process, {}, cost about {:2.6f} seconds.\n'.format(
                    epoch, ", ".join([f"{k}={v:.6f}" for k, v in dev_metrics.items()]), time_con))
            else:
                no_improve += 1

            if self.args.early_stop and no_improve > self.args.patience:
                mylogger.info(f"No optimization for {no_improve} epochs, auto-stopping at epoch {epoch}...")
                break
    @staticmethod
    def calculate_metrics(pred_slot, real_slot, pred_intent, real_intent, prefix=""):
        slot_f1 = miulab.computeF1Score(real_slot, pred_slot)[0]
        intent_acc = Evaluator.accuracy(pred_intent, real_intent)
        sent_acc = Evaluator.semantic_acc(pred_slot, real_slot, pred_intent, real_intent)
        return {
            f"{prefix}slot_f1": slot_f1,
            f"{prefix}intent_acc": intent_acc,
            f"{prefix}sent_acc": sent_acc
        }

    def estimate(self, if_dev=True):
        """
            Estimate the performance of model on dev or test dataset.
        """
        with torch.no_grad():
            if if_dev:
                pred_slot, real_slot, pred_intent, real_intent, _, _, _ = self.prediction("dev")
            else:
                pred_slot, real_slot, pred_intent, real_intent, _, _, _ = self.prediction("test")

        metrics = self.calculate_metrics(pred_slot, real_slot, pred_intent, real_intent,
                                         prefix="dev_" if if_dev else "test_")

        return metrics

    def validate(self, model_path, dataset_path):
        """
        validation will write mistaken samples to files and make scores.
        """
        if model_path is not None:
            self.load_model(model_path)
        if dataset_path is not None:
            self.dataset = torch.load(dataset_path) if self.args.gpu \
                else torch.load(dataset_path, map_location=torch.device('cpu'))

        with torch.no_grad():
            pred_slot, real_slot, pred_intent, real_intent, sent_list, slot_nodes, intent_nodes = \
                self.prediction("test")
        metrics = self.calculate_metrics(pred_slot, real_slot, pred_intent, real_intent, prefix="")

        # To make sure the directory for save error prediction.
        mistake_dir = os.path.join(self.dataset.save_dir, "error")
        if not os.path.exists(mistake_dir):
            os.mkdir(mistake_dir)

        slot_file_path = os.path.join(mistake_dir, "slot.txt")
        intent_file_path = os.path.join(mistake_dir, "intent.txt")
        both_file_path = os.path.join(mistake_dir, "both.txt")

        # Write those sample with mistaken slot prediction.
        with open(slot_file_path, 'w') as fw:
            for w_list, r_slot_list, p_slot_list in zip(sent_list, real_slot, pred_slot):
                if r_slot_list != p_slot_list:
                    for w, r, p in zip(w_list, r_slot_list, p_slot_list):
                        fw.write(w + '\t' + r + '\t' + p + '\n')
                    fw.write('\n')

        # Write those sample with mistaken intent prediction.
        with open(intent_file_path, 'w') as fw:
            for w_list, p_intent, r_intent in zip(sent_list, pred_intent, real_intent):
                if p_intent != r_intent:
                    for w in w_list:
                        fw.write(w + '\n')
                    fw.write(r_intent + '\t' + p_intent + '\n\n')

        # Write those sample both have intent and slot errors.
        with open(both_file_path, 'w') as fw:
            for w_list, r_slot_list, p_slot_list, p_intent, r_intent in \
                    zip(sent_list, real_slot, pred_slot, pred_intent, real_intent):

                if r_slot_list != p_slot_list or r_intent != p_intent:
                    for w, r_slot, p_slot in zip(w_list, r_slot_list, p_slot_list):
                        fw.write(w + '\t' + r_slot + '\t' + p_slot + '\n')
                    fw.write(w_list + '\t' + r_intent + '\t' + p_intent + '\n\n')

        # TODO: Write all samples with supporting nodes.
        all_with_nodes_file_path = os.path.join(mistake_dir, "all_with_nodes.txt")
        if len(slot_nodes[0]) > 0:
            with open(all_with_nodes_file_path, 'w') as fw:
                for w_list, r_slot_list, p_slot_list, p_intent, r_intent, \
                        s_r_nodes, s_s_nodes, i_r_nodes, i_s_nodes in \
                        zip(sent_list, real_slot, pred_slot, pred_intent, real_intent,
                            slot_nodes[0], slot_nodes[1], intent_nodes[0], intent_nodes[1]):
                    correct_slot = r_slot_list == p_slot_list
                    correct_intent = r_intent == p_intent

                    fw.write(f"slot: {correct_slot} ; intent: {correct_intent}\n")
                    for w, r_slot, p_slot, r_nodes, s_nodes in zip(
                            w_list, r_slot_list, p_slot_list, s_r_nodes, s_s_nodes):
                        fw.write(w + '\t' + r_slot + '\t' + p_slot + '\n')
                        fw.write('\treal: ' + '; '.join(
                            [f"{node} ({pv1:.4f}, {pv2:.4f})" for node, pv1, pv2 in r_nodes]) + '\n')
                        fw.write('\tmask: ' + '; '.join(
                            [f"{node} ({pv1:.4f}, {pv2:.4f})" for node, pv1, pv2 in s_nodes]) + '\n')

                    fw.write(w_list + '\t' + r_intent + '\t' + p_intent + '\n')
                    fw.write('\treal: ' + '; '.join(
                        [f"{node} ({pv1:.4f}, {pv2:.4f})" for node, pv1, pv2 in i_r_nodes]) + '\n')
                    fw.write('\tmask: ' + '; '.join(
                        [f"{node} ({pv1:.4f}, {pv2:.4f})" for node, pv1, pv2 in i_s_nodes]) + '\n\n')
        return metrics

    def prediction(self, mode):
        self.model.eval()
        if mode in ["dev", "test"]:
            dataloader = self.dataset.get_dataloader(
                mode, getattr(self, "tokenizer", None), char_level=not self.args.word_level_pretrained,
                batch_size=self.args.batch_size)
        else:
            raise Exception("Argument error! mode belongs to {\"dev\", \"test\"}.")

        pred_slot, real_slot = [], []
        pred_intent, real_intent = [], []
        sent_list = []
        slot_nodes, intent_nodes = [[], []], [[], []]

        node_nums = np.array([], dtype=np.int32)
        phar = tqdm(dataloader, desc=f"{mode}", ncols=tqdm_ncols)
        for batch_data in phar:
            node_nums = np.append(node_nums, batch_data["node_num"])
            phar.set_postfix({
                "nodes": f"{np.min(node_nums):.0f}, {np.percentile(node_nums, 25):.0f}, "
                         f"{np.median(node_nums):.0f}, {np.percentile(node_nums, 75):.0f}, {np.max(node_nums):.0f}",
            })

            slot_idx, intent_idx, node_logits_slot, node_logits_intent = self.model(batch_data, n_predicts=1)
            nested_slot = Evaluator.nested_list([list(Evaluator.expand_list(slot_idx))], batch_data["text_lengths"])[0]
            pred_slot.extend(self.dataset.slot_alphabet.get_instance(nested_slot))
            pred_intent.extend(self.dataset.intent_alphabet.get_instance(intent_idx))

            real_slot.extend(batch_data["slots_plain"])
            real_intent.extend(list(Evaluator.expand_list(batch_data["intent_plain"])))
            sent_list.extend(batch_data["text_plain"])

            if node_logits_slot is not None:
                topk = 5
                graph_nodes = batch_data["graph_nodes"]
                num_nodes = batch_data["node_num"]
                text_lengths = batch_data["text_lengths"]
                # TODO: get top k and index the corresponding nodes
                # [B, N]
                prob_logits_intent, prob_soft_intent = node_logits_intent
                prob_intent = torch.softmax(prob_logits_intent, dim=-1)
                real_intent_nodes = get_topk_nodes(
                    prob_intent, prob_soft_intent, graph_nodes, num_nodes, text_lengths, topk=topk)
                soft_intent_nodes = get_topk_nodes(
                    prob_soft_intent, prob_intent, graph_nodes, num_nodes, text_lengths, topk=topk)
                intent_nodes[0].extend(real_intent_nodes)
                intent_nodes[1].extend(soft_intent_nodes)
                # [B, T, N]
                prob_logits_slot, prob_soft_slot = node_logits_slot
                prob_slot = torch.softmax(prob_logits_slot, dim=-1)
                real_slot_nodes = get_topk_nodes(
                    prob_slot, prob_soft_slot, graph_nodes, num_nodes, text_lengths, topk=topk)
                soft_slot_nodes = get_topk_nodes(
                    prob_soft_slot, prob_slot, graph_nodes, num_nodes, text_lengths, topk=topk)
                slot_nodes[0].extend(real_slot_nodes)
                slot_nodes[1].extend(soft_slot_nodes)

        pred_intent = [pred_intent_[0] for pred_intent_ in pred_intent]
        return pred_slot, real_slot, pred_intent, real_intent, sent_list, slot_nodes, intent_nodes


def get_topk_nodes(probs, masked_probs, graph_nodes, node_nums, text_lengths, topk=10):
    is_slot = len(probs.shape) == 3
    if not is_slot:
        probs = probs.unsqueeze(1)
        masked_probs = masked_probs.unsqueeze(1)
    top_pv, top_pi = probs.topk(topk)
    masked_top_pv = masked_probs.gather(dim=-1, index=top_pi).tolist()
    top_pv, top_pi = top_pv.tolist(), top_pi.tolist()

    batch_nodes = []
    for bi, bd in enumerate(graph_nodes):
        num_valid_node = node_nums[bi]
        sample_nodes = []
        for ci in range(text_lengths[bi] if is_slot else 1):
            char_nodes = []
            for tpi, pi in enumerate(top_pi[bi][ci]):
                if pi >= num_valid_node:
                    node_plain = None
                else:
                    try:
                        node_plain = bd[pi]
                    except IndexError:
                        print(bi, ci, pi, num_valid_node, len(bd), len(top_pv), len(top_pv[bi]), len(top_pv[bi][ci]))
                        raise IndexError
                char_nodes.append([node_plain, top_pv[bi][ci][tpi], masked_top_pv[bi][ci][tpi]])
            sample_nodes.append(char_nodes)
        batch_nodes.append(sample_nodes if is_slot else sample_nodes[0])
    return batch_nodes


class Evaluator(object):

    @staticmethod
    def semantic_acc(pred_slot, real_slot, pred_intent, real_intent):
        """
        Compute the accuracy based on the whole predictions of
        given sentence, including slot and intent.
        """

        total_count, correct_count = 0.0, 0.0
        for p_slot, r_slot, p_intent, r_intent in zip(pred_slot, real_slot, pred_intent, real_intent):

            if p_slot == r_slot and p_intent == r_intent:
                correct_count += 1.0
            total_count += 1.0

        return 1.0 * correct_count / total_count

    @staticmethod
    def accuracy(pred_list, real_list):
        """
        Get accuracy measured by predictions and ground-trues.
        """

        pred_array = np.array(list(Evaluator.expand_list(pred_list)))
        real_array = np.array(list(Evaluator.expand_list(real_list)))
        return (pred_array == real_array).sum() * 1.0 / len(pred_array)

    @staticmethod
    def expand_list(nested_list):
        for item in nested_list:
            if isinstance(item, (list, tuple)):
                for sub_item in Evaluator.expand_list(item):
                    yield sub_item
            else:
                yield item

    @staticmethod
    def nested_list(items, seq_lens):
        num_items = len(items)
        trans_items = [[] for _ in range(0, num_items)]

        count = 0
        for jdx in range(0, len(seq_lens)):
            for idx in range(0, num_items):
                trans_items[idx].append(items[idx][count:count + seq_lens[jdx]])
            count += seq_lens[jdx]

        return trans_items
