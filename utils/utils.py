#!/usr/bin/env python
# coding: utf-8


import numpy as np
import random
from scipy.spatial.distance import cdist
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform
import matplotlib.pyplot as plt


from sklearn import manifold
from sklearn.cluster import KMeans
from sklearn import metrics
import tensorflow as tf

G_TRAIN_RATIO = 0.8
G_TEST_RATIO = 0.2
NOISE_DX = 0.1

colors = ['#FF1493', '#FF00FF', '#7B68EE', '#0000FF', '#800080', '#4B0082', '#FFB6C1', '#808000', '#DC143C', '#4169E1',
          '#00BFFF', '#5F9EA0', '#00FFFF', '#00CED1', '#2F4F4F', '#00FF7F', '#2E8B57', '#FFFF00', '#FFD700', '#FFA500',
          '#FF4500', '#000000']


# np.random.shuffle(colors)

def syc_shuffle(arr_list):
    state = np.random.get_state()
    for e in arr_list:
        np.random.shuffle(e)
        np.random.set_state(state)


def split_dataset_ex_static(view_data, label,train_ratio = -1,test_ratio = -1):
    if train_ratio >= 0:
        TRAIN_RATIO = train_ratio
    else:
        TRAIN_RATIO = G_TRAIN_RATIO

    if test_ratio >= 0:
        TEST_RATIO = test_ratio
    else:
        TEST_RATIO = G_TEST_RATIO
    length = len(view_data[0])
    # 分成10分
    train_idx = []
    test_idx = []
    block_len = int(length / 10)
    block_idx = range(0, length, block_len)

    for bi in block_idx:
        if bi + block_len < length:
            candi_idx = range(bi, bi + block_len)
        else:
            candi_idx = range(bi, length)
        train_idx.extend(candi_idx[0:int(block_len * TRAIN_RATIO)])
        test_idx.extend(candi_idx[int(block_len * (TRAIN_RATIO+TEST_RATIO)):])

    # print('test_idx_ex2')
    # print(test_idx)

    train_view_data = [[] for e in range(len(view_data))]
    train_label = label[train_idx]

    test_view_data = [[] for e in range(len(view_data))]
    test_label = label[test_idx]

    for e in range(len(view_data)):
        train_view_data[e] = view_data[e][train_idx]
        test_view_data[e] = view_data[e][test_idx]

    return train_view_data, train_label, test_view_data, test_label

def split_dataset_ex_with_label_balance_static(view_data,label,train_ratio = -1,test_ratio = -1):
    if train_ratio >= 0 :
        TRAIN_RATIO = train_ratio
    else:
        TRAIN_RATIO = G_TRAIN_RATIO

    if test_ratio >= 0 :
        TEST_RATIO = test_ratio
    else:
        TEST_RATIO = G_TEST_RATIO
    label_dict = dict()
    #max_cls = np.max(label)
    for i,l in enumerate(label):
        if label_dict.__contains__(l):
            label_dict[l].append(i)
        else:
            label_dict[l] = [i]
    #每个类别按比例划分
    cls_keys = list(label_dict.keys())
    nview = len(view_data)
    train_view_data = [[] for v in range(nview)]
    test_view_data =  [[] for v in range(nview)]
    train_label = []
    test_label = []

    label_count_dict = dict()
    for ck in cls_keys:
        num_of_cls = len(label_dict[ck])
        label_count_dict[ck] = 'count:'+str(num_of_cls)
        idx = [i for i in range(num_of_cls)]#random.sample(range(num_of_cls),k=num_of_cls)
        train_idx = np.asarray(label_dict[ck])[idx[0:int(num_of_cls*TRAIN_RATIO)]]
        test_idx = np.asarray(label_dict[ck])[idx[int(num_of_cls*TRAIN_RATIO):int(num_of_cls*(TRAIN_RATIO+TEST_RATIO))]]

        for v in range(nview):
            train_view_data[v].extend(view_data[v][train_idx])
            test_view_data[v].extend(view_data[v][test_idx])

        train_label.extend(label[train_idx])
        test_label.extend(label[test_idx])


    print(label_count_dict)

    #shuffle
    train_shuffle_list = []
    test_shuffle_list = []
    for v in range(nview):
        train_shuffle_list.append(train_view_data[v])
        test_shuffle_list.append(test_view_data[v])

    train_shuffle_list.append(train_label)
    test_shuffle_list.append(test_label)

    syc_shuffle(train_shuffle_list)
    syc_shuffle(test_shuffle_list)

    for v in range(nview):
        train_view_data[v] = np.asarray(train_view_data[v])
        test_view_data[v] = np.asarray(test_view_data[v])


    return train_view_data,np.asarray(train_label),test_view_data,np.asarray(test_label)

def split_dataset_n_shot_static(view_data,label,n_shot,):

    label_dict = dict()
    #max_cls = np.max(label)
    for i,l in enumerate(label):
        if label_dict.__contains__(l):
            label_dict[l].append(i)
        else:
            label_dict[l] = [i]
    #每个类别按比例划分
    cls_keys = list(label_dict.keys())
    nview = len(view_data)
    train_view_data = [[] for v in range(nview)]
    test_view_data =  [[] for v in range(nview)]
    train_label = []
    test_label = []

    label_count_dict = dict()
    for ck in cls_keys:
        num_of_cls = len(label_dict[ck])
        label_count_dict[ck] = 'count:'+str(num_of_cls)
        idx = [i for i in range(num_of_cls)]#random.sample(range(num_of_cls),k=num_of_cls)
        train_idx = np.asarray(label_dict[ck])[idx[0:n_shot]]
        test_idx = np.asarray(label_dict[ck])[idx[0:n_shot]]

        for v in range(nview):
            train_view_data[v].extend(view_data[v][train_idx])
            test_view_data[v].extend(view_data[v][test_idx])

        train_label.extend(label[train_idx])
        test_label.extend(label[test_idx])


    print(label_count_dict)



    for v in range(nview):
        train_view_data[v] = np.asarray(train_view_data[v])
        test_view_data[v] = np.asarray(test_view_data[v])


    return train_view_data,np.asarray(train_label),test_view_data,np.asarray(test_label)

def split_dataset_ex_with_label_balance(view_data,label,train_ratio = -1,test_ratio = -1):
    if train_ratio >= 0:
        TRAIN_RATIO = train_ratio
    else:
        TRAIN_RATIO = G_TRAIN_RATIO

    if test_ratio >= 0:
        TEST_RATIO = test_ratio
    else:
        TEST_RATIO = G_TEST_RATIO
    label_dict = dict()
    #max_cls = np.max(label)
    for i,l in enumerate(label):
        if label_dict.__contains__(l):
            label_dict[l].append(i)
        else:
            label_dict[l] = [i]
    #每个类别按比例划分
    cls_keys = list(label_dict.keys())
    nview = len(view_data)
    train_view_data = [[] for v in range(nview)]
    test_view_data =  [[] for v in range(nview)]
    train_label = []
    test_label = []

    label_count_dict = dict()
    for ck in cls_keys:
        num_of_cls = len(label_dict[ck])
        label_count_dict[ck] = 'count:'+str(num_of_cls)
        idx = random.sample(range(num_of_cls),k=num_of_cls)
        train_idx = np.asarray(label_dict[ck])[idx[0:int(num_of_cls*TRAIN_RATIO)]]
        test_idx = np.asarray(label_dict[ck])[idx[int(num_of_cls*TRAIN_RATIO):int(num_of_cls*(TRAIN_RATIO+TEST_RATIO))]]

        for v in range(nview):
            train_view_data[v].extend(view_data[v][train_idx])
            test_view_data[v].extend(view_data[v][test_idx])

        train_label.extend(label[train_idx])
        test_label.extend(label[test_idx])


    print(label_count_dict)

    #shuffle
    train_shuffle_list = []
    test_shuffle_list = []
    for v in range(nview):
        train_shuffle_list.append(train_view_data[v])
        test_shuffle_list.append(test_view_data[v])

    train_shuffle_list.append(train_label)
    test_shuffle_list.append(test_label)

    syc_shuffle(train_shuffle_list)
    syc_shuffle(test_shuffle_list)

    for v in range(nview):
        train_view_data[v] = np.asarray(train_view_data[v])
        test_view_data[v] = np.asarray(test_view_data[v])


    return train_view_data,np.asarray(train_label),test_view_data,np.asarray(test_label)





def split_dataset_ex(view_data, label,train_ratio = -1,test_ratio = -1):
    #return split_dataset_ex_static(view_data,label)
    if train_ratio > 0 :
        TRAIN_RATIO = train_ratio
    if test_ratio > 0 :
        TEST_RATIO = test_ratio
    return split_dataset_ex_with_label_balance(view_data,label,train_ratio,test_ratio)
    length = len(view_data[0])
    random_idx = random.sample(range(length), k=length)

    train_idx = random_idx[:int(length * TRAIN_RATIO)]
    test_idx = random_idx[int(length * TRAIN_RATIO):int(length * (TRAIN_RATIO+TEST_RATIO))]

    train_view_data = [[] for e in range(len(view_data))]
    train_label = label[train_idx]

    test_view_data = [[] for e in range(len(view_data))]
    test_label = label[test_idx]

    for e in range(len(view_data)):
        train_view_data[e] = view_data[e][train_idx]
        test_view_data[e] = view_data[e][test_idx]

    return train_view_data, train_label, test_view_data, test_label


def split_dataset2(view1, view2, label):
    length = len(view1)
    random_idx = random.sample(range(length), k=length)
    train_idx = random_idx[:int(length * TRAIN_RATIO)]
    test_idx = random_idx[int(length * TRAIN_RATIO):]

    train_view1 = view1[train_idx]
    train_view2 = view2[train_idx]
    train_label = label[train_idx]

    test_view1 = view1[test_idx]
    test_view2 = view2[test_idx]
    test_label = label[test_idx]

    return train_label, train_view1, train_view2, test_label, test_view1, test_view2


def split_dataset(view1, view2, label):
    length = len(view1)
    random_idx = random.sample(range(length), k=length)
    train_idx = random_idx[:int(length * TRAIN_RATIO)]
    test_idx = random_idx[int(length * TRAIN_RATIO):]

    train_view1 = view1[train_idx]
    train_view2 = view2[train_idx]
    train_label = label[train_idx]

    test_view1 = view1[test_idx]
    test_view2 = view2[test_idx]
    test_label = label[test_idx]

    return train_view1, train_view2, train_label, test_view1, test_view2, test_label


def push_query(query, url, dict):
    if query in dict:
        dict[query].append(url)
    else:
        dict[query] = [url]
    return dict


def is_same_cate(labelA, labelB):
    return labelA == labelB


def make_dict(label):
    anc_index = []
    pos_indices = []
    neg_indices = []
    num = len(label)

    for i in range(num):
        anc_index.append(i)
        candi_pos = []
        candi_neg = []
        for j in range(num):
            if is_same_cate(label[i], label[j]):
                candi_pos.append(j)
            else:
                candi_neg.append(j)
        # end
        pos_indices.append(candi_pos)
        neg_indices.append(candi_neg)

    return anc_index, pos_indices,neg_indices


def make_dict_KNN(knn_graph):
    anc_index = []
    pos_indices = []
    neg_indices = []
    num = len(knn_graph)

    for i in range(num):
        anc_index.append(i)
        candi_pos = []
        candi_neg = []
        for j in range(num):
            if knn_graph[i][j] >= 2 or knn_graph[j][i] >= 2:
                candi_pos.append(j)
            else:
                candi_neg.append(j)
        # end
        pos_indices.append(candi_pos)
        neg_indices.append(candi_neg)

    return anc_index, pos_indices, neg_indices

def make_knn_graph(data,K):
    dist_mat = squareform(pdist(data))
    knn_graph = []
    num = len(data)
    for i in range(num):
        vec = np.zeros(num)
        dst = np.argsort(dist_mat[i])[:K]
        for d in dst:
            vec[d] = 1
        knn_graph.append(vec)
    return np.asarray(knn_graph)



def generate_sample(anchor, pos, neg, view_data):
    # output: NVIEW*NUM*DIM

    nview = len(view_data)

    A_list = [[] for e in range(nview)]
    P_list = [[] for e in range(nview)]
    N_list = [[] for e in range(nview)]

    random.shuffle(anchor)

    for query in anchor:
        an_idx = query

        # print(query)
        candidate_pos_list = pos[query]
        candidate_neg_list = neg[query]

        # random.shuffle(candidate_pos_list)
        # random.shuffle(candidate_neg_list)

        #
        random_pos_idx = candidate_pos_list[np.random.choice(len(candidate_pos_list))]
        random_neg_idx = candidate_neg_list[np.random.choice(len(candidate_neg_list))]
        # ok 索引有了 取数据

        for v in range(nview):
            A_list[v].append(view_data[v][an_idx])
            P_list[v].append(view_data[v][random_pos_idx])
            N_list[v].append(view_data[v][random_neg_idx])

    return A_list, P_list, N_list

def generate_sample_and_return_onehot_label(anchor, pos, neg, view_data,onehot_label):
    # output: NVIEW*NUM*DIM

    nview = len(view_data)

    A_list = [[] for e in range(nview)]
    P_list = [[] for e in range(nview)]
    N_list = [[] for e in range(nview)]

    random.shuffle(anchor)

    A_label = []
    P_label = []
    N_label = []

    for query in anchor:
        an_idx = query
        A_label.append(onehot_label[an_idx])
        # print(query)
        candidate_pos_list = pos[query]
        candidate_neg_list = neg[query]

        # random.shuffle(candidate_pos_list)
        # random.shuffle(candidate_neg_list)

        #
        random_pos_idx = candidate_pos_list[np.random.choice(len(candidate_pos_list))]
        random_neg_idx = candidate_neg_list[np.random.choice(len(candidate_neg_list))]

        P_label.append(onehot_label[random_pos_idx])
        N_label.append(onehot_label[random_neg_idx])
        # ok 索引有了 取数据

        for v in range(nview):
            A_list[v].append(view_data[v][an_idx])
            P_list[v].append(view_data[v][random_pos_idx])
            N_list[v].append(view_data[v][random_neg_idx])

    return A_list, A_label,P_list, P_label,N_list,N_label

def generate_sample_and_return_label(anchor, pos, neg, view_data,label):
    # output: NVIEW*NUM*DIM

    nview = len(view_data)

    A_list = [[] for e in range(nview)]
    P_list = [[] for e in range(nview)]
    N_list = [[] for e in range(nview)]

    random.shuffle(anchor)

    A_label = []
    P_label = []
    N_label = []

    for query in anchor:
        an_idx = query
        A_label.append(label[an_idx])
        # print(query)
        candidate_pos_list = pos[query]
        candidate_neg_list = neg[query]

        # random.shuffle(candidate_pos_list)
        # random.shuffle(candidate_neg_list)

        #
        random_pos_idx = candidate_pos_list[np.random.choice(len(candidate_pos_list))]
        random_neg_idx = candidate_neg_list[np.random.choice(len(candidate_neg_list))]

        P_label.append(label[random_pos_idx])
        N_label.append(label[random_neg_idx])
        # ok 索引有了 取数据

        for v in range(nview):
            A_list[v].append(view_data[v][an_idx])
            P_list[v].append(view_data[v][random_pos_idx])
            N_list[v].append(view_data[v][random_neg_idx])

    return A_list, A_label,P_list, P_label,N_list,N_label

def generate_shuffle_sample(view_data):
    return view_data
    idx = random.sample(range(len(view_data[0])),k=len(view_data[0]))
    nview = len(view_data)
    rt_list = [[] for v in range(nview)]
    for i in idx:
        for v in range(nview):
            rt_list[v].append(view_data[v][i])
    return rt_list


def generate_hard_probability_sample(anchor, pos, neg, view_data, model, sess):
    nview = len(view_data)
    m_feed_dict = {}
    for v in range(nview):
        m_feed_dict[model.input_A[v]] = view_data[v]
    view_data_rep = sess.run(model.AREP, feed_dict=m_feed_dict)

    A_list = [[] for e in range(nview)]
    P_list = [[] for e in range(nview)]
    N_list = [[] for e in range(nview)]

    random.shuffle(anchor)

    for query in anchor:
        an_idx = query
        # print(query)
        candidate_pos_list = pos[query]
        candidate_neg_list = neg[query]

        candidate_pos_dist = cdist(np.asarray([view_data_rep[an_idx]]), view_data_rep[candidate_pos_list])[0]
        candidate_neg_dist = cdist(np.asarray([view_data_rep[an_idx]]), view_data_rep[candidate_neg_list])[0]
        candidata_pos_prob = candidate_pos_dist / np.sum(candidate_pos_dist)
        candidata_neg_prob = (np.sum(candidate_neg_dist) - candidate_neg_dist) / np.sum(
            np.sum(candidate_neg_dist) - candidate_neg_dist)
        # a = np.argmax(candidate_pos_dist)
        # print(a)
        #
        random_pos_idx = np.random.choice(
            candidate_pos_list)  # candidate_pos_list[np.argmax(candidate_pos_dist)]#np.random.choice(len(candidate_pos_list))
        random_neg_idx = np.random.choice(candidate_neg_list, p=candidata_neg_prob)
        # ok 索引有了 取数据

        for v in range(nview):
            A_list[v].append(view_data[v][an_idx])
            P_list[v].append(view_data[v][random_pos_idx])
            N_list[v].append(view_data[v][random_neg_idx])

    return A_list, P_list, N_list


def train_dataset_augmentation(train_view_data, train_label, num_of_class):
    # count per class
    class_count = [0 for c in range(num_of_class)]
    for c in train_label:
        class_count[c] += 1
    max_count_class = np.argmax(class_count)
    max_count = np.max(class_count)

    nview = len(train_view_data)

    class_data_list = [[] for c in range(num_of_class)]
    for c in range(num_of_class):
        class_data_list[c] = [[] for v in range(nview)]

    # 分类存放
    for idx, la in enumerate(train_label):
        for v in range(nview):
            class_data_list[la][v].append(train_view_data[v][idx])

    # 填充每类
    for c in range(num_of_class):
        num_to_append = max_count - class_count[c]
        exist_num = class_count[c]  # 已经存在的个数
        candi_idx = [e for e in range(exist_num)]
        cdid = np.random.choice(a=candi_idx, size=num_to_append)
        for cid in cdid:
            for v in range(nview):  # 每个视图都补充
                vd = class_data_list[c][v][cid]  # 随机选出
                rnoise = NOISE_DX * np.random.randn(vd.shape[0])
                vd = vd + rnoise
                class_data_list[c][v].append(vd)

    new_view_data = [[] for v in range(nview)]
    new_label = []
    # 扩充
    for c in range(num_of_class):  # 这一类的所有视图都加进去
        for v in range(nview):
            for vd in class_data_list[c][v]:  # 第c类的第v个视图
                new_view_data[v].append(vd)
        # print(len(class_data_list[c][0]))  # 这一类有多少个呢？
        for cl in range(len(class_data_list[c][0])):  # 每个视图的个数是一样的
            new_label.append(c)
    # print(len(new_label))
    for v in range(nview):
        new_view_data[v] = np.asarray(new_view_data[v])

    return new_view_data, np.asarray(new_label).astype(int)


def draw_with_tsne(rep, label, metric='euclidean',title="",last_extend=0):
    tsne = manifold.TSNE(n_components=2)#, init='pca', random_state=501)
    new_tsne_rep = tsne.fit_transform(rep)

    fig, ax = plt.subplots()

    # ax = Axes3D(fig)
    if last_extend >0:
        for rep_i, rep in enumerate(new_tsne_rep[:-last_extend]):
            x, y = rep[0], rep[1]
            ax.scatter(x, y, s=5, c=colors[label[rep_i]])
        for rep_i, rep in enumerate(new_tsne_rep[-last_extend:]):
            x, y = rep[0], rep[1]
            ax.scatter(x, y, s=20, c='#050505')
    else:
        for rep_i, rep in enumerate(new_tsne_rep[:]):
            x, y = rep[0], rep[1]
            ax.scatter(x, y, s=5, c=colors[label[rep_i]])
    plt.show()

colors_num = [c*15.0 for c in range(1,21)]

def draw_dataset(dataset_name,view_reps,view_names,labels,last_extend=0):
    nview = len(view_reps)
    plt.rcParams['figure.dpi'] = 220
    plt.rcParams['figure.figsize'] = [6.0,6.0]
    fig = plt.figure()

    nrow = int(nview / 2+0.5)
    ncol = 2
    if nview == 1:
        ncol = 1
    plt.suptitle(dataset_name)
    for v in range(nview):
        vrep = view_reps[v]
        tsne = manifold.TSNE(n_components=2)
        tsne_rep = tsne.fit_transform(vrep)
        if last_extend > 0:
            x = tsne_rep[:-last_extend,0]
            y = tsne_rep[:-last_extend,1]
        else:
            x = tsne_rep[:, 0]
            y = tsne_rep[:, 1]
        ax = fig.add_subplot(nrow*100+(ncol)*10+v+1)
        for i,sx in enumerate(x):
            ax.scatter(sx,y[i],s=1.0,c=colors[labels[i]])
        #ax.scatter(x,y,s=POINT_SIZE,c=clist)
        if last_extend > 0:
            for tr in tsne_rep[-last_extend:]:
                ax.scatter(tr[0],tr[1],s=10.0,c='#050505')
        ax.set_title(view_names[v])
    #
    plt.show()
def fast_draw_with_tsne(rep, label, metric='euclidean',title=""):
    tsne = manifold.TSNE(n_components=2, init='pca', random_state=501)
    new_tsne_rep = tsne.fit_transform(rep)
    tsne_color = [colors[l] for l in label]
    tsne_size = [5 for e in label]

    #fig, ax = plt.subplots()
    # ax = Axes3D(fig)
    plt.scatter(list(new_tsne_rep[:, 0]), list(new_tsne_rep[:, 1]), s=tsne_color, c=tsne_color)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title(title)

    plt.show()


def prin_clustering(test_rep, test_label, NUM_OF_CLASS):
    # 聚类
    km = KMeans(n_clusters=NUM_OF_CLASS)
    #km.fit_transform(test_rep)
    cls_rs = km.fit_predict(test_rep)
    # ARI
    ari = metrics.adjusted_rand_score(test_label, cls_rs)
    # AMI
    ami = metrics.adjusted_mutual_info_score(test_label, cls_rs)
    # H,C,V
    H, C, V = metrics.homogeneity_completeness_v_measure(test_label, cls_rs)
    # FMI
    fmi = metrics.fowlkes_mallows_score(test_label, cls_rs)
    # s
    # s = metrics.silhouette_score(test_label, cls_rs)
    # DBI
    # dbi = metrics.davies_bouldin_score(test_label, cls_rs)
    # nmi
    nmi = metrics.normalized_mutual_info_score(test_label,cls_rs)


    d = dict()
    d['ari'] = ari
    d['ami'] = ami
    d['nmi'] = nmi
    d['fmi'] = fmi
    d['H'] = H
    d['C'] = C
    d['V'] = V

    print('ARI:%.4f,AMI:%.4f,HCV:%.4f %.4f %.4f FMI:%.4f NMI:%.4f' % (ari, ami, H, C, V, fmi,nmi))
    return d

def get_tf_session():
    gpu_options = tf.GPUOptions(allow_growth=True)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    # 设置随机种子
    tf.random.set_random_seed(1234)
    return sess


def normalize_view_data(view_data, select_view):
    return [(view_data[s] - np.min(view_data[s])) / (np.max(view_data[s]) - np.min(view_data[s])) for s in select_view]
