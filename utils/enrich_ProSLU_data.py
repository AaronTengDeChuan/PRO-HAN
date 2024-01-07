# coding: utf-8

import os
import re
import sys
import orjson
from collections import OrderedDict, Counter

def rebuild_json(json_file):
    UP = {
        '音视频应用偏好': ['音乐类', '视频类', '有声读物类'],
        '出行交通工具偏好': ['地铁', '公交', '驾车'],
        '长途交通工具偏好': ['火车', '飞机', '汽车'],
        '是否有车': ['是', '否']
    }
    CA = {
        '移动状态': ['行走', '跑步', '静止', '汽车', '地铁', '高铁', '飞机', '未知'],
        '姿态识别': ['躺卧', '行走', '未知'],
        '地理围栏': ['家', '公司', '国内', '未知'],
        '户外围栏': ['户外', '室内', '未知']
    }

    with open(json_file, "r", encoding="utf-8") as f:
        data = orjson.loads(f.read())

    ids = data.pop("ids")
    assert set(data.keys()) == set(ids)

    rebuilt_data = OrderedDict()

    # count the number of different columns
    columns = Counter()

    # record intent and slots in corresponding intent
    intent2slots = OrderedDict()

    for data_id, data_dict in data.items():
        # align UP / CA items
        up_dict, ca_dict = {}, {}
        up_features = data_dict.pop("UP")
        for key in UP.keys():
            up_dict[key] = dict(
                (UP[key][i], up_features[key][i] if up_features != [] else 0.0)
                for i in range(len(UP[key])))
        ca_features = data_dict.pop("CA")
        for key in CA.keys():
            ca_dict[key] = dict(
                (CA[key][i], ca_features[key][i] if ca_features != [] else 0.0)
                for i in range(len(CA[key])))

        # divide single-line KG into different items starting with "subject："
        kb = []
        kg_str = data_dict.pop("KG").strip()
        if kg_str:
            subjects = kg_str.split("；subject：")
            subjects = [subj if subj.startswith("subject：") else f"subject：{subj}" for subj in subjects]
            # divide each item into different triples
            for subject in subjects:
                items = re.split(r"，(\w+)：", '，' + subject)[1:]
                assert len(items) % 2 == 0
                kb.append(dict(zip(items[::2], items[1::2])))
                columns.update(items[::2])

            # print(len(subjects), orjson.dumps(subjects, option=orjson.OPT_INDENT_2).decode())
            # exit(0)

        rebuilt_item = OrderedDict()
        rebuilt_item["用户话语"] = data_dict.pop("用户话语")
        # check whether whitespace exists in the sentence
        if " " in rebuilt_item["用户话语"]:
            print(f"Whitespace exists in {data_id} of {json_file}: \"{rebuilt_item['用户话语']}\"")
        rebuilt_item["intent"] = data_dict.pop("intent")
        rebuilt_item["slot"] = data_dict.pop("slot")
        rebuilt_item["KG"] = kb
        rebuilt_item["UP"] = up_dict
        rebuilt_item["CA"] = ca_dict
        assert data_dict == {}
        rebuilt_data[data_id] = rebuilt_item

        intent_list, slot_label_list = zip(
            *[re.split(r'\.|-', label)[1:] if label != "O" else ['O', 'O'] for label in rebuilt_item["slot"]])
        intent_set = set(intent_list) - {'O'}
        prev_slot = None
        slot_list = []
        for cur_slot_lable in slot_label_list:
            if cur_slot_lable != prev_slot:
                if cur_slot_lable != 'O':
                    slot_list.append(cur_slot_lable)
                prev_slot = cur_slot_lable

        assert len(intent_set) == 1
        intent2slots.setdefault(rebuilt_item["intent"], Counter()).update(slot_list + ['Num_Intent'])

    # print(orjson.dumps(columns.most_common()[::-1][:20], option=orjson.OPT_INDENT_2).decode())
    # print(orjson.dumps(columns.most_common(10), option=orjson.OPT_INDENT_2).decode())

    return rebuilt_data, dict(sorted([(k, OrderedDict(v.most_common())) for k, v in intent2slots.items()], key=lambda x: x[1]['Num_Intent'], reverse=True))


if __name__ == '__main__':
    dataset_dir = "data/ProSLU"
    for split in ["train", "dev", "test"]:
        json_file = os.path.join(dataset_dir, f"{split}.json")

        rebuilt_json, intent2slots = rebuild_json(json_file)

        with open(os.path.join(dataset_dir, f"{split}_rebuild.json"), "w", encoding="utf-8") as f:
            f.write(orjson.dumps([intent2slots, rebuilt_json], option=orjson.OPT_INDENT_2).decode())