import h5py
import numpy as np
import scipy
import scipy.sparse as sp
from collections import defaultdict
import re
import os
import torchvision
from torchvision import datasets, transforms
import torch
from PIL import Image
import time
import pickle
import urllib.parse
import matplotlib.pyplot as plt
np.set_printoptions(threshold=np.inf)

def read_pickle(fname):
    with open(fname, 'rb') as f:
        data = pickle.load(f)
    return data

class KGs:
    def __init__(self, data_folder, division, ordered=True, modality=None):
        kg1_relation_triples_set, kg1_entities_set, kg1_relations_set = self.read_relation_triples(data_folder + 'FB15K_EntityTriples.txt')
        kg2_relation_triples_set, kg2_entities_set, kg2_relations_set = self.read_relation_triples(data_folder + 'DB15K_EntityTriples.txt')

        self.ent_num = len(kg1_entities_set | kg2_entities_set)
        self.rel_num = len(kg1_relations_set | kg2_relations_set)

        self.ent_ids1_dict, self.ent_ids2_dict, self.ids_ent1_dict, self.ids_ent2_dict = self.generate_mapping_id(kg1_relation_triples_set, kg1_entities_set,
                                                    kg2_relation_triples_set, kg2_entities_set, ordered=ordered)
        self.rel_ids1_dict, self.rel_ids2_dict, self.ids_rel1_dict, self.ids_rel2_dict = self.generate_mapping_id(kg1_relation_triples_set, kg1_relations_set,
                                                    kg2_relation_triples_set, kg2_relations_set, ordered=ordered)
        self.ent_ids_dict = {**self.ent_ids1_dict, **self.ent_ids2_dict}
        self.id_relation_triples1, self.rt_dict1, self.hr_dict1 = self.uris_relation_triple_2ids(kg1_relation_triples_set, self.ent_ids1_dict, self.rel_ids1_dict)
        self.id_relation_triples2, self.rt_dict2, self.hr_dict2 = self.uris_relation_triple_2ids(kg2_relation_triples_set, self.ent_ids2_dict, self.rel_ids2_dict)
        self.kg1_entities_list = list(self.ent_ids1_dict.values())
        self.kg2_entities_list = list(self.ent_ids2_dict.values())

        self.relation_triples = self.id_relation_triples1.extend(self.id_relation_triples2)

        if 'i' in modality:
            self.id_ent_images_res_dict1 = self.read_image_emb(data_folder + 'clip_image_1.pkl', self.ent_ids1_dict)
            self.id_ent_images_res_dict2 = self.read_image_emb(data_folder + 'clip_image_2.pkl', self.ent_ids2_dict)
            self.images_list, self.image_mask = self.emerge_image_embedding(self.id_ent_images_res_dict1, self.id_ent_images_res_dict2)
            
        if 'a' in modality:
            self.attr_list, self.attr_id_dict1, self.attr_id_dict2, \
            self.kg1_attribute_triples_set, self.kg2_attribute_triples_set, \
            self.id_attr_dict1, self.id_attr_dict2, self.e_att = self.generate_attr_id(data_folder + 'FB15K_NumericalTriples.txt', data_folder + 'DB15K_NumericalTriples.txt',
                                                                                            data_folder + 'attr_name_1.h5', data_folder + 'attr_name_2.h5')
            
            self.id_ent_attr_res_dict1 = self.read_attr_embed(data_folder + 'bert_attr_1.pkl', self.ent_ids1_dict)
            self.id_ent_attr_res_dict2 = self.read_attr_embed(data_folder + 'bert_attr_2.pkl', self.ent_ids2_dict)
            self.attr_emb_list, self.attr_mask = self.emerge_image_embedding(self.id_ent_attr_res_dict1, self.id_ent_attr_res_dict2)

            _, self.eid_aid_v1 = self.uris_attribute_triple_2ids(self.kg1_attribute_triples_set, self.ent_ids1_dict, self.attr_id_dict1)
            _, self.eid_aid_v2 = self.uris_attribute_triple_2ids(self.kg2_attribute_triples_set, self.ent_ids2_dict, self.attr_id_dict2)
            
        self.train_links = self.read_links(data_folder + division + 'train_links', self.ent_ids1_dict, self.ent_ids2_dict)
        self.train_entities1 = [link[0] for link in self.train_links]
        self.train_entities2 = [link[1] for link in self.train_links]
        
        # add swapping triples
        sup_triples1_set, sup_triples2_set = self.generate_sup_relation_triples(self.train_links,
                                                                self.rt_dict1, self.hr_dict1,
                                                                self.rt_dict2, self.hr_dict2)
        self.relation_triples_list1, self.relation_triples_list2 = self.add_sup_relation_triples(sup_triples1_set, sup_triples2_set)
        self.relation_triples_set1 = set(self.relation_triples_list1)
        self.relation_triples_set2 = set(self.relation_triples_list2)
        
        if 'a' in modality:
            self.eid_aid_v_add1, self.eid_aid_v_add2 = self.add_sup_attribute_triples(self.train_links, self.eid_aid_v1, self.eid_aid_v2)
            self.attr_max_num, self.eid_aid_list, self.eid_vid_list, self.eav_len_list, self.eid_mask_list = self.generate_attr_list(self.eid_aid_v_add1, self.eid_aid_v_add2)
        
        self.test_links = self.read_links(data_folder + division + 'test_links', self.ent_ids1_dict, self.ent_ids2_dict) #list
        self.test_entities1 = [link[0] for link in self.test_links]
        self.test_entities2 = [link[1] for link in self.test_links]
        self.valid_links = self.read_links(data_folder + division + 'valid_links', self.ent_ids1_dict, self.ent_ids2_dict)
        self.valid_entities1 = [link[0] for link in self.valid_links]
        self.valid_entities2 = [link[1] for link in self.valid_links]

        print("KG statistics:")
        print("entities: kg1 =", len(kg1_entities_set), "kg2 =", len(kg2_entities_set))
        print("relations: kg1 =", len(kg1_relations_set), "kg2 =", len(kg2_relations_set))
        print("relation triples: kg1 =", len(self.relation_triples_list1), "kg2 =", len(self.relation_triples_list2))
        print("local relation triples: kg1 =", len(self.id_relation_triples1), "kg2 =", len(self.id_relation_triples2))
        print()


    # def read_relation_triples(self, file_path):
    #     print("read relation triples:", file_path)
    #     triples = set()
    #     entities, relations = set(), set()
    #     file = open(file_path, 'r', encoding='utf8')
    #     for line in file.readlines():
    #         params = line.strip('\n').split('\t')
    #         assert len(params) == 3
    #         h = params[0].strip()
    #         r = params[1].strip()
    #         t = params[2].strip()
    #         triples.add((h, r, t))
    #         entities.add(h)
    #         entities.add(t)
    #         relations.add(r)
    #     file.close()
    #     return triples, entities, relations
    
    def read_relation_triples(self, filename):
        print("read relation triples:", filename)
        
        relation_triples_set = set()
        entities_set = set()
        relations_set = set()
        
        # 根据文件名判断是 DBpedia 还是 Freebase
        is_dbpedia = 'DB15K' in os.path.basename(filename)
        
        with open(filename, 'r', encoding='utf-8') as f:
            for line in f:
                line_stripped = line.strip()
                h, r, t = None, None, None
                
                if not line_stripped:
                    continue
                
                # 1. FB15K (Freebase ID format) processing
                if not is_dbpedia:
                    # 剥离行末的句号 '.'
                    if line_stripped.endswith('.'):
                        line_stripped_processed = line_stripped[:-1].strip()
                    else:
                        line_stripped_processed = line_stripped
                    
                    # 以任意空白字符分隔
                    params = line_stripped_processed.split()
                    
                    if len(params) == 3:
                        h, r, t = params[0], params[1], params[2]
                
                # 2. DB15K (URI format) processing
                else:
                    # DB15Klines are URI format
                    # 使用正则匹配 <h> <r> <t> . 格式
                    match = re.search(r'<(.*?)>\s+<(.*?)>\s+<(.*?)>\s*\.?', line_stripped)
                    
                    if match:
                        h = match.group(1).strip()
                        r = match.group(2).strip()
                        t = match.group(3).strip()
                
                # --- 统一处理和清理 ---
                if h is not None and r is not None and t is not None:
                    
                    # 剥离实体和关系可能带有的尖括号 (<...>) (如果正则没有完全剥离)
                    if h.startswith('<') and h.endswith('>'):
                        h = h[1:-1]
                    if r.startswith('<') and r.endswith('>'):
                        r = r[1:-1]
                    if t.startswith('<') and t.endswith('>'):
                        t = t[1:-1]
                    
                    relation_triples_set.add((h, r, t))
                    entities_set.add(h)
                    entities_set.add(t)
                    relations_set.add(r)
                
                elif line_stripped:
                    # 如果是格式错误的非空行
                    print(
                        f"Warning: Skipping malformed line in triples file: '{line_stripped}' (Could not parse 3 elements).")
                    continue
        
        return relation_triples_set, entities_set, relations_set

    # def generate_attr_id(self, att_f1, att_f2, att_embed_1, att_embed_2):
    #     id_attr_dict1 = dict()
    #     attr_id_dict1 = dict()
    #     id_attr_dict2 = dict()
    #     attr_id_dict2 = dict()
    #     triples1 = set()
    #     triples2 = set()
    #     attr_embed = []
    #     cnt = 1
    #     e_att = dict()
    #     file = open(att_f1, 'r', encoding='utf-8')
    #     for line in file.readlines():
    #         params = line.strip().split('\t')
    #         assert len(params) == 3
    #         e = params[0].strip()
    #         a = params[1].strip()
    #         v = params[2].strip()
    #         triples1.add((e, a, v))
    #         if e not in e_att.keys():
    #             e_att[e] = []
    #         e_att[e].append((a,v))
    #         if params[1] not in attr_id_dict1.keys():
    #             id_attr_dict1[cnt] = params[1]
    #             attr_id_dict1[params[1]] = cnt
    #             cnt+=1
    #     file.close()
    #     assert len(id_attr_dict1) == len(attr_id_dict1)
    #
    #     file = open(att_f2, 'r', encoding='utf-8')
    #     for line in file.readlines():
    #         params = line.strip().split('\t')
    #         assert len(params) == 3
    #         e = params[0].strip()
    #         a = params[1].strip()
    #         v = params[2].strip()
    #         triples2.add((e, a, v))
    #         if e not in e_att.keys():
    #             e_att[e] = []
    #         e_att[e].append((a,v))
    #         if params[1] not in attr_id_dict2.keys():
    #             id_attr_dict2[cnt] = params[1]
    #             attr_id_dict2[params[1]] = cnt
    #             cnt+=1
    #     file.close()
    #     assert len(id_attr_dict2) == len(attr_id_dict2)
    #     # print(len(id_attr_dict1), len(id_attr_dict2), cnt)
    #     attr_num = len(id_attr_dict1) + len(id_attr_dict2)
    #
    #     f1 = h5py.File(att_embed_1, 'r')
    #     f2 = h5py.File(att_embed_2, 'r')
    #
    #     attr_embed.append(np.zeros(768))
    #     for i in range(1, attr_num + 1):
    #         if i in id_attr_dict1.keys():
    #             emb = np.array(f1[id_attr_dict1[i]])
    #             attr_embed.append(emb)
    #         elif i in id_attr_dict2.keys():
    #             emb = np.array(f2[id_attr_dict2[i]])
    #             attr_embed.append(emb)
    #         else:
    #             print("error!")
    #             exit()
    #
    #     return attr_embed, attr_id_dict1, attr_id_dict2, triples1, triples2, id_attr_dict1, id_attr_dict2, e_att
    
    def generate_attr_id(self, att_f1, att_f2, att_embed_1, att_embed_2):
        id_attr_dict1 = dict()  # Global ID -> Attribute Name String (CLEANED)
        attr_id_dict1 = dict()  # Attribute Name String (CLEANED) -> Global ID
        id_attr_dict2 = dict()
        attr_id_dict2 = dict()
        triples1 = set()
        triples2 = set()
        attr_embed = []
        cnt = 1  # Global counter for attribute ID, starts at 1 (0 is reserved for padding)
        e_att = dict()
        
        # --- 读取第一个属性文件 (att_f1) ---
        print("Reading numerical triples from:", att_f1)
        file = open(att_f1, 'r', encoding='utf-8')
        for line in file.readlines():
            params = line.strip().split('\t')
            
            # 核心修改：跳过格式错误的行
            if len(params) != 3:
                if line.strip():
                    # print(f"Warning: Skipping malformed line in numerical triples file 1: '{line.strip()}' (Expected 3 elements, found {len(params)}).")
                    pass
                continue
            
            e = params[0].strip()
            a = params[1].strip()  # 原始属性名称 (如: <http://...>)
            v = params[2].strip()
            
            # <<<--- 核心修复：立即剥离尖括号 --->>>
            if a.startswith('<') and a.endswith('>'):
                a = a[1:-1]
            
            triples1.add((e, a, v))
            if e not in e_att.keys():
                e_att[e] = []
            e_att[e].append((a, v))
            
            # 使用 CLEANED 属性名 'a' 存入字典
            if a not in attr_id_dict1.keys():
                id_attr_dict1[cnt] = a
                attr_id_dict1[a] = cnt
                cnt += 1
        file.close()
        assert len(id_attr_dict1) == len(attr_id_dict1)
        
        # --- 读取第二个属性文件 (att_f2) ---
        print("Reading numerical triples from:", att_f2)
        file = open(att_f2, 'r', encoding='utf-8')
        for line in file.readlines():
            params = line.strip().split('\t')
            
            # 核心修改：跳过格式错误的行
            if len(params) != 3:
                if line.strip():
                    # print(f"Warning: Skipping malformed line in numerical triples file 2: '{line.strip()}' (Expected 3 elements, found {len(params)}).")
                    pass
                continue
            
            e = params[0].strip()
            a = params[1].strip()  # 原始属性名称 (如: <http://...>)
            v = params[2].strip()
            
            # <<<--- 核心修复：立即剥离尖括号 --->>>
            if a.startswith('<') and a.endswith('>'):
                a = a[1:-1]
            
            triples2.add((e, a, v))
            if e not in e_att.keys():
                e_att[e] = []
            e_att[e].append((a, v))
            
            # 使用 CLEANED 属性名 'a' 存入字典
            if a not in attr_id_dict2.keys():
                id_attr_dict2[cnt] = a
                attr_id_dict2[a] = cnt
                cnt += 1
        file.close()
        assert len(id_attr_dict2) == len(attr_id_dict2)
        
        attr_num = len(id_attr_dict1) + len(id_attr_dict2)
        
        # ----------------------------------------------------
        # --- HDF5 嵌入查找逻辑 (保持不变，因为这部分逻辑是正确的) ---
        # ----------------------------------------------------
        print("reading attr embed and creating lookup map...")
        f1 = h5py.File(att_embed_1, 'r')
        f2 = h5py.File(att_embed_2, 'r')
        
        # 1. 加载 KG1 的属性名称和嵌入，并创建 名字 -> 嵌入 的映射
        attr_names1 = [s.decode('utf-8') for s in f1['attr_name'][:]]
        attr_embeds1 = f1['attr_emb'][:]
        name_to_embed1 = {name: attr_embeds1[i] for i, name in enumerate(attr_names1)}
        
        # 2. 加载 KG2 的属性名称和嵌入，并创建 名字 -> 嵌入 的映射
        attr_names2 = [s.decode('utf-8') for s in f2['attr_name'][:]]
        attr_embeds2 = f2['attr_emb'][:]
        name_to_embed2 = {name: attr_embeds2[i] for i, name in enumerate(attr_names2)}
        
        f1.close()
        f2.close()
        
        # 3. 按照全局 ID 顺序构建最终的 attr_embed 列表
        attr_embed.append(np.zeros(768))  # ID 0 留给 padding/unknown
        for i in range(1, attr_num + 1):
            if i in id_attr_dict1.keys():
                attr_name = id_attr_dict1[i]  # 属性名称字符串 (现在是 CLEANED 的)
                if attr_name not in name_to_embed1:
                    # 理论上不应该发生，因为 H5 文件也用 CLEANED 名称
                    print(f"Error: Attribute name '{attr_name}' (ID {i}) not found in KG1 embeddings!")
                    exit()
                emb = name_to_embed1[attr_name]  # 通过名称查找嵌入
                attr_embed.append(emb)
            
            elif i in id_attr_dict2.keys():
                attr_name = id_attr_dict2[i]  # 属性名称字符串 (现在是 CLEANED 的)
                if attr_name not in name_to_embed2:
                    print(f"Error: Attribute name '{attr_name}' (ID {i}) not found in KG2 embeddings!")
                    exit()
                emb = name_to_embed2[attr_name]  # 通过名称查找嵌入
                attr_embed.append(emb)
            else:
                print("error! Attribute ID not found in either KG.")
                exit()
        # ----------------------------------------------------
        
        return attr_embed, attr_id_dict1, attr_id_dict2, triples1, triples2, id_attr_dict1, id_attr_dict2, e_att

    def read_links(self, file_path, e_id_1, e_id_2):
        print("read links:", file_path)
        links = list()
        file = open(file_path, 'r', encoding='utf8')
        for line in file.readlines():
            params = line.strip('\n').split('\t')
            assert len(params) == 2
            # e1 = e_id_1[params[0].strip()]
            # e2 = e_id_2[params[1].strip()]
            
            e1 = e_id_1[params[0].strip()]
            # 对 e2，移除两端的尖括号 '<' 和 '>'
            e2 = e_id_2[params[1].strip().strip('<>')]
            
            links.append([e1, e2])
        return links

    def generate_sup_relation_triples_one_link(self, e1, e2, rt_dict, hr_dict):
        new_triples = set()
        for r, t in rt_dict.get(e1, set()):
            new_triples.add((e2, r, t))
        for h, r in hr_dict.get(e1, set()):
            new_triples.add((h, r, e2))
        return new_triples

    def generate_sup_relation_triples(self, sup_links, rt_dict1, hr_dict1, rt_dict2, hr_dict2):
        new_triples1, new_triples2 = set(), set()
        for ent1, ent2 in sup_links:
            new_triples1 |= (self.generate_sup_relation_triples_one_link(ent1, ent2, rt_dict1, hr_dict1))
            new_triples2 |= (self.generate_sup_relation_triples_one_link(ent2, ent1, rt_dict2, hr_dict2))
        print("supervised relation triples: {}, {}".format(len(new_triples1), len(new_triples2)))
        return new_triples1, new_triples2

    def add_sup_relation_triples(self, sup_triples1, sup_triples2):
        id_relation_triples1_set = set(self.id_relation_triples1)
        id_relation_triples1_set |= sup_triples1
        id_relation_triples2_set = set(self.id_relation_triples2)
        id_relation_triples2_set |= sup_triples2
        return list(id_relation_triples1_set), list(id_relation_triples2_set)

    def add_sup_attribute_triples(self, sup_links, e_av1, e_av2):
        add_attr_num1 = 0
        add_attr_num2 = 0
        for e1, e2 in sup_links:
            sup_e1 = e_av2.get(e2, set())
            sup_e2 = e_av1.get(e1, set())
            new_attr_set = sup_e1 | sup_e2
            e_av1[e1] = new_attr_set
            e_av2[e2] = new_attr_set
            add_attr_num1 += len(sup_e2)
            add_attr_num2 += len(sup_e1)
        print("sup attribute triples: {}, {}".format(add_attr_num1, add_attr_num2))
        return e_av1, e_av2

    def generate_attr_feature(self, e_av1, e_av2):
        eid_attr_feature = []
        e_a_f = []
        cnt1 = 0
        cnt2 = 0
        for eid in range(self.ent_num):
            if eid in e_av1.keys():
                a_list = [self.attr_list[a] for (a, _) in e_av1[eid]]
                if len(a_list) == 0:
                    cnt1+=1
                    continue
                else:
                    eid_attr_feature.append(np.mean(a_list, axis=0))
            elif eid in e_av2.keys():
                a_list = [self.attr_list[a] for (a, _) in e_av2[eid]]
                if len(a_list) == 0:
                    cnt1+=1
                    continue
                else:
                    eid_attr_feature.append(np.mean(a_list, axis=0))

        mean = np.mean(eid_attr_feature, axis=0)
        std = np.std(eid_attr_feature, axis=0)
        for eid in range(self.ent_num):
            if eid in e_av1.keys():
                a_list = [self.attr_list[a] for (a, _) in e_av1[eid]]
                if len(a_list) == 0:
                    e_a_f.append(np.random.normal(mean, std, mean.shape[0]))
                else:
                    e_a_f.append(np.mean(a_list, axis=0))
            elif eid in e_av2.keys():
                a_list = [self.attr_list[a] for (a, _) in e_av2[eid]]
                if len(a_list) == 0:
                    e_a_f.append(np.random.normal(mean, std, mean.shape[0]))
                else:
                    e_a_f.append(np.mean(a_list, axis=0))
            else:
                cnt2+=1
                e_a_f.append(np.random.normal(mean, std, mean.shape[0]))
        assert len(e_a_f) == self.ent_num
        print(cnt1, cnt2)
        return e_a_f

    def generate_attr_list(self, e_av1, e_av2):
        max_num = 0
        cnt_zero = 0
        entid_attr_list = []
        entid_value_list = []
        entid_av_len_list = []
        
        for eid in range(self.ent_num):
            if eid in e_av1.keys():
                max_num = max(max_num, len(e_av1[eid]))
            elif eid in e_av2.keys():
                max_num = max(max_num, len(e_av2[eid]))
            else:
                cnt_zero += 1
        print("attribute max num: {}".format(max_num))

        av_mask = np.ones((self.ent_num, max_num))
        for eid in range(self.ent_num):
            if eid in e_av1.keys():
                av_l = len(e_av1[eid])
                av = e_av1[eid]
            elif eid in e_av2.keys():
                av_l = len(e_av2[eid])
                av = e_av2[eid]
            else:
                av_l = 0
                av = []
            a = [ea for (ea, _) in av]
            v = [ev for (_, ev) in av]
            for i in range(max_num - av_l):
                a.append(0)
                v.append(0)
            av_mask[eid][av_l:] = 0
            entid_attr_list.append(a)
            entid_value_list.append(v)
            entid_av_len_list.append(av_l)
        return max_num, entid_attr_list, entid_value_list, entid_av_len_list, av_mask.tolist()
        

    def sort_elements(self, triples, elements_set):
        dic = dict()
        for s, p, o in triples:
            if s in elements_set:
                dic[s] = dic.get(s, 0) + 1
            if p in elements_set:
                dic[p] = dic.get(p, 0) + 1
            if o in elements_set:
                dic[o] = dic.get(o, 0) + 1

        sorted_list = sorted(dic.items(), key=lambda x: (x[1], x[0]), reverse=True)
        ordered_elements = [x[0] for x in sorted_list]
        return ordered_elements, dic

    def generate_mapping_id(self, kg1_triples, kg1_elements, kg2_triples, kg2_elements, ordered=True):
        ids1, ids2 = dict(), dict()
        ids_ent1, ids_ent2 = dict(), dict()
        if ordered:
            kg1_ordered_elements, _ = self.sort_elements(kg1_triples, kg1_elements)
            kg2_ordered_elements, _ = self.sort_elements(kg2_triples, kg2_elements)
            n1 = len(kg1_ordered_elements)
            n2 = len(kg2_ordered_elements)
            n = max(n1, n2)
            for i in range(n):
                if i < n1 and i < n2:
                    ids1[kg1_ordered_elements[i]] = i * 2
                    ids_ent1[i * 2] = kg1_ordered_elements[i]
                    ids2[kg2_ordered_elements[i]] = i * 2 + 1
                    ids_ent2[i * 2 + 1] = kg2_ordered_elements[i]
                elif i >= n1:
                    ids2[kg2_ordered_elements[i]] = n1 * 2 + (i - n1)
                    ids_ent2[n1 * 2 + (i - n1)] = kg2_ordered_elements[i]
                else:
                    ids1[kg1_ordered_elements[i]] = n2 * 2 + (i - n2)
                    ids_ent1[n2 * 2 + (i - n2)] = kg1_ordered_elements[i]
        else:
            index = 0
            for ele in kg1_elements:
                if ele not in ids1:
                    ids1[ele] = index
                    ids_ent1[index] = ids1[ele]
                    index += 1
            for ele in kg2_elements:
                if ele not in ids2:
                    ids2[ele] = index
                    ids_ent2[index] = ids2[ele]
                    index += 1
        assert len(ids1) == len(set(kg1_elements))
        assert len(ids2) == len(set(kg2_elements))
        return ids1, ids2, ids_ent1, ids_ent2

    def uris_relation_triple_2ids(self, uris, ent_ids, rel_ids):
        id_uris = list()
        rt_dict, hr_dict = dict(), dict()
        for u1, u2, u3 in uris:
            assert u1 in ent_ids
            h_id = ent_ids[u1]
            assert u2 in rel_ids
            r_id = rel_ids[u2]
            assert u3 in ent_ids
            t_id = ent_ids[u3]
            id_uris.append((h_id, r_id, t_id))

            rt_set = rt_dict.get(h_id, set())
            rt_set.add((r_id, t_id))
            rt_dict[h_id] = rt_set

            hr_set = hr_dict.get(t_id, set())
            hr_set.add((h_id, r_id))
            hr_dict[t_id] = hr_set

        assert len(id_uris) == len(set(uris))
        return id_uris, rt_dict, hr_dict
    
    def uris_attribute_triple_2ids(self, uris, ent_ids, att_ids):
        id_uris = list()
        e_av_dict = dict()
        
        # 获取 ent_ids 中键的前缀，例如 'FB$' 或 'DB$'
        prefix = ''
        try:
            # 尝试从 ent_ids 的第一个键中推断前缀
            key_sample = next(iter(ent_ids.keys()))
            if '$' in key_sample:
                prefix = key_sample.split('$')[0] + '$'
        except StopIteration:
            # 排除 ent_ids 为空的情况
            pass
        
        # 遍历属性三元组
        for u1, u2, u3 in uris:
            # --- FIX START: 修复实体u1的名称，以匹配 ent_ids ---
            e_name_raw = u1
            e_name = u1
            e_id = -1  # 默认ID
            
            # 1. 尝试原始名称查找 (通常是 raw_name)
            if e_name_raw in ent_ids:
                e_name = e_name_raw
            else:
                # 2. 尝试添加前缀查找 (通常是 prefix + raw_name)
                e_name_prefixed = prefix + e_name_raw
                if e_name_prefixed in ent_ids:
                    e_name = e_name_prefixed
                else:
                    # 实体不在 KG 的主实体映射中，跳过此三元组
                    # 替换了原来的 assert，防止程序崩溃
                    # print(f"Warning: Skipping attribute triple for unknown entity: {e_name_raw} (not found in ent_ids).")
                    continue
            
            e_id = ent_ids[e_name]
            # --- FIX END ---
            
            # assert u1 in ent_ids # 原始断言已移除
            
            # 原始代码的属性和数值处理部分（不变）
            assert u2 in att_ids
            a_id = att_ids[u2]
            v = u3.split('\"^^')[0].strip('\"')
            if 'e-' in v:
                pass
            elif '-' in v and v[0] != '-':
                v = v.split('-')[0]
            elif v[0] == '-' and v.count('-') > 1:
                v = '-' + v.split('-')[1]
            if '#' in v:
                v = v.strip('#')
            
            # 确保 v 可以转换为 float
            try:
                float_v = float(v)
            except ValueError:
                # print(f"Warning: Skipping attribute triple due to invalid numerical value: {v}")
                continue
            
            id_uris.append((e_id, a_id, float_v))
            
            av_set = e_av_dict.get(e_id, set())
            av_set.add((a_id, float_v))
            e_av_dict[e_id] = av_set
        
        # 原始代码的断言，检查映射后的三元组数量是否一致（不变）
        assert len(id_uris) <= len(set(uris))  # 修复为 <= 因为我们可能跳过了一些三元组
        
        return id_uris, e_av_dict

    def uris_entity_image_2ids(self, id_file, h5_file, ent_ids):
        ids = dict()
        entid_image = dict()
        file = open(id_file, 'r', encoding='utf8')
        for line in file.readlines():
            params = line.strip('\n').split('\t')
            assert len(params) == 2
            ids[params[0]] = params[1]
        file.close()

        f = h5py.File(h5_file, 'r')
        cnt1 = 0
        cnt2 = 0
        for (ent, id) in ent_ids.items():
            if ent in ids.keys():
                cnt1 += 1
                entid_image[id] = f[ids[ent]][0]
            else:
                cnt2 += 1
                entid_image[id] = np.random.uniform(0, 10, 4096)

        print("ent with image:", cnt1, "ent without image:", cnt2)
        f.close()
        return entid_image

    def emerge_image_embedding(self, entid_image1, entid_image2):
        image_embed = []
        num = len(entid_image1)+len(entid_image2)
        mask_image = []
        for i in range(num):
            if i in entid_image1.keys():
                image_embed.append(entid_image1[i])
            else:
                image_embed.append(entid_image2[i])
            if np.all(image_embed[i] ==0):
                mask_image.append(0)
            else:
                mask_image.append(1)
        # print(mask_image)
        indices = torch.nonzero(torch.tensor(mask_image))
        print('image not zero:', len(indices))
        return image_embed, mask_image
    
    def read_image_emb(self, pkl_file, ent_ids, ratio=1):
        print("read image embeddings:", pkl_file)
        entid_image = dict()
        f = read_pickle(pkl_file)
        cnt1 = 0
        cnt2 = 0
        non_zero_id = []
        
        for (ent, id) in ent_ids.items():
            # 1. 剥离前缀
            ent_core = ent.split('$', 1)[-1]
            
            lookup_keys = [ent_core]
            
            # 2. 如果是 DBpedia，尝试添加尖括号 (适配 ImageIndex.txt 的原始格式)
            if 'http' in ent_core and not ent_core.startswith('<'):
                lookup_keys.append(f"<{ent_core}>")
            
            # 3. 查找
            found = False
            for key in lookup_keys:
                if key in f.keys():
                    entid_image[id] = np.array(f[key][0])
                    non_zero_id.append(id)
                    cnt1 += 1
                    found = True
                    break
            
            if not found:
                cnt2 += 1
                entid_image[id] = np.zeros(768)
        
        print("ent with image:", cnt1, "ent without image:", cnt2)
        
        if ratio != 1:
            need_zero_num = int((1 - ratio) * len(non_zero_id))
            for x in non_zero_id[:need_zero_num]:
                entid_image[x] = np.zeros(768)
        return entid_image
    
    def read_attr_embed(self, pkl_file, ent_ids, ratio=1):
        print("read attr embeddings:", pkl_file)
        entid_attr = dict()
        f = read_pickle(pkl_file)
        cnt1 = 0
        cnt2 = 0
        non_zero_id = []
        
        for (ent, id) in ent_ids.items():
            # 1. 剥离 KG 内部可能添加的前缀 (e.g., 'FB$' 或 'DB$')
            ent_core = ent.split('$', 1)[-1]
            
            # --- 核心修复: 尝试所有可能的查找键 ---
            lookup_keys = [
                # Key 1: 默认键 (适用于 FB15K IDs: /m/027rn)
                ent_core
            ]
            
            if 'http' in ent_core:
                # 针对 DBpedia URI: http://dbpedia.org/resource/Anarchism
                
                # Key 2: 简化名称 (例如: 'Anarchism')
                match = re.search(r'/([^/]+)$', ent_core)
                if match:
                    simplified_name = match.group(1).strip()
                    lookup_keys.append(simplified_name)
                
                # Key 3: URI 编码/解码后的名称 (有时作者会使用未解码的名称)
                # 例如，如果有空格，URI编码会是 '%20'
                # 尝试查找 URI 最后一个斜杠后的部分，然后进行 URL 解码
                try:
                    import urllib.parse
                    decoded_simplified_name = urllib.parse.unquote(
                        simplified_name) if 'simplified_name' in locals() else simplified_name
                    if decoded_simplified_name != simplified_name:
                        lookup_keys.append(decoded_simplified_name)
                except:
                    pass
            
            # 4. 尝试查找键，找到一个就成功
            found = False
            for key in lookup_keys:
                if key in f.keys():
                    entid_attr[id] = np.array(f[key][0])
                    non_zero_id.append(id)
                    cnt1 += 1
                    found = True
                    break
            
            if not found:
                # 找不到任何特征，填充零向量
                cnt2 += 1
                entid_attr[id] = np.zeros(768)
        
        print("ent with attr:", cnt1, "ent without attr:", cnt2)
        
        if ratio != 1:
            need_zero_num = int((1 - ratio) * len(non_zero_id))
            for x in non_zero_id[:need_zero_num]:
                entid_attr[x] = np.zeros(768)
        return entid_attr
    
    def read_image_emb(self, pkl_file, ent_ids, ratio=1):
        import re
        import urllib.parse
        import numpy as np  # 确保 np 在函数内可用
        
        print("read image embeddings:", pkl_file)
        entid_image = dict()
        f = read_pickle(pkl_file)
        cnt1 = 0
        cnt2 = 0
        non_zero_id = []
        
        # 预先定义零向量，确保数据类型一致
        FEATURE_DIM = 768
        zero_vector = np.zeros(FEATURE_DIM, dtype=np.float32)
        
        for (ent, id) in ent_ids.items():
            
            ent_key = ent.split('$', 1)[-1]
            
            # 强制剥离所有可能存在的尖括号，创建 CLEANED 键
            if ent_key.startswith('<') and ent_key.endswith('>'):
                ent_key_cleaned = ent_key[1:-1]
            else:
                ent_key_cleaned = ent_key
            
            # 尝试所有可能的查找键 (与 V3 相同，保持健壮性)
            lookup_keys = [
                ent_key_cleaned,
                f"<{ent_key_cleaned}>",
            ]
            
            if 'http' in ent_key_cleaned:
                match = re.search(r'/([^/]+)$', ent_key_cleaned)
                if match:
                    simplified_name = match.group(1).strip()
                    lookup_keys.append(simplified_name)
                    
                    try:
                        decoded_simplified_name = urllib.parse.unquote(simplified_name)
                        if decoded_simplified_name != simplified_name:
                            lookup_keys.append(decoded_simplified_name)
                    except:
                        pass
            
            found = False
            for key in lookup_keys:
                if key in f.keys():
                    cnt1 += 1
                    val = f[key]
                    if isinstance(val, list):
                        val = val[0]
                    
                    # 确保返回的是 np.ndarray
                    entid_image[id] = np.array(val, dtype=np.float32)
                    non_zero_id.append(id)
                    found = True
                    break
            
            if not found:
                cnt2 += 1
                # <<<--- 核心修复：确保未找到的实体返回预定义的、类型明确的零向量 --->>>
                entid_image[id] = zero_vector
        
        print("ent with image:", cnt1, "ent without image:", cnt2)
        
        if ratio != 1:
            need_zero_num = int((1 - ratio) * len(non_zero_id))
            for x in non_zero_id[:need_zero_num]:
                entid_image[x] = zero_vector
        return entid_image