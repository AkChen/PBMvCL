#!/usr/bin/env python
# coding: utf-8


import tensorflow as tf
import numpy as np
import random
from TMLNet_MultiProtoOnMAdapDIM import TMLNET
from utils import *
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.distance import cdist
from scipy.spatial.distance import pdist,squareform
from sklearn import manifold
from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import recall_score
from pylab import *
from sklearn.metrics import classification_report
from map import MAP


BATCH_SIZE = 64
TRAIN_EPOCH = 300
SAMPLE_EPOCH = 10#10
TEST_EPOCH = 2
DRAW_EPOCH = 500
TOP_K = 5
KNN_K = 50
HDIM_1 = 64
HDIM_2 = 64
CDIM_1 = 64
CDIM_2 = 64
SELECT_VIEW = [0,1,2,3,4,5]
ACC_THRESH = 0.99
# 固定划分 计算每个视图的效果 不加triplet
# hw
# v1:0.6900 0.6400
# v2:0.8150 0.7700
# v3:0.9800 0.9800
# v4:0.4 0.4
# v5:0.9725 0.9700
# v6:0.86 0.86
# fusion: 0.9750 0.9775
nview = len(SELECT_VIEW)
ALPHA = 0.3 * HDIM_2  # 太大的margin 导致每个都为正 对 负样本的距离没有约束，太小的margin会导致都为负 从而 直接loss失效。 16dim大概50.0就行
DELTA = 1.0
SIGMA = 10.0

LEARNING_RATE = 0.01 #0.003
WEIGHT_DECAY = 0.0001


#0.01 0.0004  在nus-5上weight decay 更小效果更好


DATA_DIR = '../data/'
# ten colors
colors = ['#FFB6C1','#DC143C','#FF1493','#FF00FF','#800080','#4B0082','#7B68EE','#0000FF','#4169E1','#00BFFF','#5F9EA0','#00FFFF','#00CED1','#2F4F4F','#00FF7F','#2E8B57','#FFFF00','#808000','#FFD700','#FFA500','#FF4500','#000000']
np.random.shuffle(colors)
# view2 不错
dataset = 'caltech101-20_P.npy'
data_dict = dict(np.load(DATA_DIR + dataset)[()])
train_dict = data_dict['train']
train_view_data_ = [train_dict['view_data'][s] for s in SELECT_VIEW] #分割的 实际用于的后面还要分
train_label_ = train_dict['label']
test_dict= data_dict['test']
test_view_data =  [test_dict['view_data'][s] for s in SELECT_VIEW]
test_label = test_dict['label']
view_name = [data_dict['view_name'][s] for s in SELECT_VIEW]
NUM_OF_CLASS = np.max(train_dict['label']+1)
view_dim = [data_dict['view_dim'][s] for s in SELECT_VIEW]
print(view_name)
print(view_dim)

view_data = []
view_data.extend(train_view_data_)
view_data.extend(test_view_data)

label = []
label.extend(train_label_)
label.extend(test_label)


#train_view_data, train_label, test_view_data, test_label = split_dataset_ex(view_data, label)
#将train拿出一部分用于训练
#train_view_data,train_label,_,_ = split_dataset_n_shot_static(train_view_data_,train_label_,n_shot=5)#
train_view_data,train_label,_,_ = split_dataset_ex_with_label_balance_static(train_view_data_,train_label_,train_ratio=1.0,test_ratio=0.0)


#train_anch, train_pos, train_neg = make_dict(train_label)

acc_list = []
pacc_list = []
epo_list = []
los_list = []

def main():
    sess = get_tf_session()
    net = TMLNET(nview,view_dim, hdim_1=HDIM_1, hdim_2=HDIM_2, cdim_1=CDIM_1, cdim_2=CDIM_2, alpha=ALPHA, delta=DELTA,
                 num_of_class=NUM_OF_CLASS,weight_decay=WEIGHT_DECAY, learning_rate=LEARNING_RATE)
    sess.run(tf.initialize_all_variables())
    #print(sess.run(net.bs))
    max_acc = 0.00
    max_pacc = 0.0
    max_loss = 0
    print('Training')
    for e in range(TRAIN_EPOCH):

        A_ = train_view_data
        AL_ = np.asarray(train_label)

        #sl = A_
        #sl.append(sl)
        #syc_shuffle(sl)
        #A_ = sl[sl[:-1]]
        #AL_ = sl[-1]

        train_size = len(A_[0])  # d

        m_feed_dict = dict()
        for v in range(nview):
            m_feed_dict[net.input_A[v]] = A_[v]

        m_feed_dict[net.input_L] = train_label
        loss = sess.run(net.loss, feed_dict=m_feed_dict)
        print("train epoch:%d loss:%.4f " % (e, loss))
        #print(sess.run(net.vpl_loss, feed_dict=m_feed_dict))
        #print(sess.run(net.vdce_loss, feed_dict=m_feed_dict))


        cur_index = 0
        while cur_index < train_size:
            input_A = [[] for v in range(nview)]
            if cur_index + BATCH_SIZE < train_size:
                for v in range(nview):
                    input_A[v] = A_[v][cur_index:cur_index + BATCH_SIZE]
                input_L = AL_[cur_index:cur_index + BATCH_SIZE]
            else:
                for v in range(nview):
                    input_A[v] = A_[v][cur_index:]
                input_L = AL_[cur_index:]
            cur_index += BATCH_SIZE

            m_feed_dict = {}

            for v in range(nview):
                m_feed_dict[net.input_A[v]] = input_A[v]


            m_feed_dict[net.input_L] = input_L

            sess.run(net.update, feed_dict=m_feed_dict)


        #print(view_weights)
        # 是否需要测试
        if e % TEST_EPOCH == 0:
            print('test')

            m_feed_dict = {}
            for v in range(nview):
                m_feed_dict[net.input_A[v]] = train_view_data[v]


            train_rep,train_predict_label = sess.run([net.AREP,net.predict_label], feed_dict=m_feed_dict)

            m_feed_dict = {}
            for v in range(nview):
                m_feed_dict[net.input_A[v]] = test_view_data[v]
            test_rep, predict_label, vpredict_label = sess.run([net.AREP, net.predict_label, net.vpredict_label],
                                                               feed_dict=m_feed_dict)

            # 计算两两之间的距离
            test_rep_dist = cdist(test_rep, train_rep)

            test_predict_labels = []
            for idx, dst in enumerate(test_rep_dist):
                sorted_idx = np.argsort(dst)[:TOP_K]
                # 统计TOP5最多的类别
                knn_labels = train_label[sorted_idx]
                label_counts = [0 for en in range(NUM_OF_CLASS)]
                for kl in knn_labels:
                    label_counts[kl] += 1
                mx_label = np.argmax(label_counts)
                test_predict_labels.append(mx_label)

            cls_rpt = classification_report(test_label, test_predict_labels,output_dict=True)

            #print(cls_rpt)
            #print(cls_rpt['weighted avg'])
            acc = accuracy_score(test_label, test_predict_labels)
            print('ACC:%.4f' % acc)
            predict_acc = accuracy_score(test_label, predict_label)
            #vpredict_acc = [accuracy_score(test_label, vl) for vl in vpredict_label]
            #train_acc = accuracy_score(train_label, train_predict_label)
            print('PACC:%.4f' % (predict_acc))
            #print(vpredict_acc)
            #print("TACC:%.4f" % (train_acc))
            acc_list.append(acc)
            pacc_list.append(predict_acc)
            epo_list.append(e)
            los_list.append(loss)

            # train acc

            #centers = sess.run(net.centers)
            #draw_with_tsne(centers,[e for e in range(NUM_OF_CLASS)])

            #map = MAP(src_hash=test_rep, src_label=test_label, dst_hash=train_rep, dst_label=train_label)
            #print('MAP:%.4f'% map)

            if acc > max_acc or predict_acc> max_pacc:
                if  acc > max_acc:
                    max_acc = acc
                if predict_acc> max_pacc:
                    max_pacc = predict_acc

                f = open('record.txt', 'a+')
                f.write("EPO:%d ACC:%.4f PACC:%.4f \n" % (e,acc,predict_acc))
                f.close()
                M = sess.run(net.Wr)
                #plt.matshow(M, cmap='hot')
                #plt.colorbar()
                #plt.show()


                for v in range(nview):
                    m = M[(v*64):(64+v*64),:]
                    kmeans = KMeans(n_clusters=NUM_OF_CLASS)
                    kmeans.fit(np.transpose(m))
                    km_labels = kmeans.labels_

                    new_m = []

                    for i in range(NUM_OF_CLASS):
                        for j,l in enumerate(km_labels):
                            if l == i:
                                new_m.append(m[:,j])
                    new_m = np.transpose(np.asarray(new_m))
                    plt.matshow(new_m,cmap='hot')
                    plt.colorbar()
                    plt.show()

                plt.matshow([[0,0],[0,0]], cmap='hot')
                plt.colorbar()
                plt.show()



            if e == TRAIN_EPOCH - 2 :
                m_feed_dict = {}
                for v in range(nview):
                    m_feed_dict[net.input_A[v]] = view_data[v]
                    # fusion_rep,test_view_rep,proto = sess.run([net.AREP,net.centers], feed_dict=m_feed_dict)
                view_protos =sess.run(net.vcenters)
                fusion_protos = sess.run(net.centers)

                test_view_rep, test_fusion_rep,test_fusion_r_rep = sess.run([net.Aq_sig, net.As_sig,net.Ar_rep], feed_dict=m_feed_dict)
                view_rep = []  # list(test_view_rep)
                for v in range(nview):
                    vr = list(test_view_rep[v])
                    vr.extend(view_protos[v])
                    view_rep.append(vr)
                fusion_rep = list(test_fusion_rep)
                fusion_rep.extend(fusion_protos)
                fusion_rep = np.asarray(fusion_rep)
                draw_dataset(dataset+'_view_specifc',view_rep,view_names=view_name,labels=label,last_extend=NUM_OF_CLASS)
                draw_dataset(dataset + '_fusion', [fusion_rep], view_names=['fusion'], labels=label, last_extend=NUM_OF_CLASS)
                view_rep_stack = test_fusion_r_rep#np.mean(view_rep,axis=0)
                view_rep_stack_list = list(view_rep_stack)
                view_rep_stack_list.extend(np.mean(view_protos,axis=0))
                #draw_dataset(dataset + '_view_mean', [view_rep_stack], view_names=view_name, labels=label,last_extend=NUM_OF_CLASS)

    sess.close()

    #绘图

    mpl.rcParams['font.sans-serif'] = ['SimHei']

    x_axis_data = epo_list
    y1_data = acc_list
    y2_data = pacc_list
    mv = np.max(los_list)
    y3_data = [l/mv for l in los_list]
    plt.plot(x_axis_data, y1_data, marker='o', ms=0.1, label='acc of KNN')
    plt.plot(x_axis_data, y2_data, marker='*', ms=0.1, label='acc of predict')
    plt.plot(x_axis_data, y3_data, marker='>', ms=0.1, label='train loss')
    plt.title(dataset)
    plt.legend()  # 让图例生效
    plt.show()
    # plt.savefig('demo.jpg')  # 保存该图片

    ''''
            print('accuracy:%.4f f1:%.4f precision:%.4f recall:%.4f' %(whole_acc,whole_f1,whole_precision,whole_recall))

            if whole_acc > max_acc or whole_recall > max_recall or whole_precision > max_pre or whole_f1 > max_f1:
                f = open('record.txt', 'a+')
                f.write('accuracy:%.4f f1:%.4f precision:%.4f recall:%.4f\n' %(whole_acc,whole_f1,whole_precision,whole_recall))
                f.close()
            '''





if __name__ == '__main__':
    main()

# 问题：
# 1.批量训练部分的代码这样写可以吗,现在运行在生成A_v1, N_v1, P_v1, A_v2, N_v2, P_v2会出现memeory error。
# 2.有一部分损失没有加入,可以这样直接加吗？
# loss1 = tf.reduce_sum(tf.square(tf.subtract(As_sig, Aq_v1_sig)), axis=1)
# loss2 = tf.reduce_sum(tf.square(tf.subtract(As_sig, Aq_v1_sig)), axis=1)

# loss1 = tf.reduce_sum(tf.square(tf.subtract(Ps_sig, Pq_v1_sig)), axis=1)
# loss2 = tf.reduce_sum(tf.square(tf.subtract(Ps_sig, Pq_v1_sig)), axis=1)

# loss1 = tf.reduce_sum(tf.square(tf.subtract(Ns_sig, Nq_v1_sig)), axis=1)
# loss2 = tf.reduce_sum(tf.square(tf.subtract(Ns_sig, Nq_v1_sig)), axis=1)

# 4.现在我只是学了一个表示，那比如我做分类任务的时候，只是在测试集上看分类效果吗？还是训练集测试集都比较？如何用学到的模型？
