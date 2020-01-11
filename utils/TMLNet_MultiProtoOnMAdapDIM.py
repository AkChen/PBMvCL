import tensorflow as tf
import numpy as np

SQ_RATE = 10

def distance(features, centers):
    f_2 = tf.reduce_sum(tf.pow(features, 2), axis=1, keep_dims=True)
    c_2 = tf.reduce_sum(tf.pow(centers, 2), axis=1, keep_dims=True)
    dist = f_2 - 2*tf.matmul(features, centers, transpose_b=True) + tf.transpose(c_2, perm=[1,0])
    return dist

def softmax_loss(logits, labels):
    labels = tf.to_int32(labels)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels,
        logits=logits, name='xentropy')
    return tf.reduce_mean(cross_entropy, name='xentropy_mean')

def predict(features, centers):
    dist = distance(features, centers)
    prediction = tf.argmin(dist, axis=1, name='prediction')
    return tf.cast(prediction, tf.int32)

def consis_loss(v1_centers,v2_centers):
    return tf.nn.l2_loss(v1_centers-v2_centers)

def margin_proto_loss(centers,num_of_class,margin):
    loss = 0.0
    for i in range(num_of_class-1):
        for j in range(i+1,num_of_class):
            ic = tf.gather(centers,i)
            jc = tf.gather(centers,j)
            dis = tf.reduce_sum(tf.square(ic-jc))
            loss += tf.maximum(0.0,margin-dis)
    return loss

def view_between_dis(centers,num_of_class):
    loss = 0.0
    for i in range(num_of_class - 1):
        for j in range(i + 1, num_of_class):
            ic = tf.gather(centers, i)
            jc = tf.gather(centers, j)
            dis = tf.reduce_sum(tf.square(ic - jc))
            loss += dis
    return loss

class TMLNET:
    def __init__(self, nview, vdims, hdim_1, hdim_2, cdim_1, cdim_2, alpha, delta, num_of_class,weight_decay=0.01,
                 learning_rate=0.001,beta=0.02):

        self.nview = nview
        self.vdims = vdims

        self.input_A = []
        self.input_P = []
        self.input_N = []

        self.input_L = tf.placeholder(tf.int32,[None])

        # inputs
        for nv in range(nview):
            ph_A = tf.placeholder(tf.float32, [None, vdims[nv]], 'Anc_input_v' + str(nv + 1))
            ph_P = tf.placeholder(tf.float32, [None, vdims[nv]], 'Pos_input_v' + str(nv + 1))
            ph_N = tf.placeholder(tf.float32, [None, vdims[nv]], 'Neg_input_v' + str(nv + 1))
            self.input_A.append(ph_A)
            self.input_P.append(ph_P)
            self.input_N.append(ph_N)

        # weights and biases
        # p q r s t
        self.Wp = []
        self.bp = []
        self.Wq = []
        self.bq = []
        #
        for nv in range(nview):
            Wp_ = tf.Variable(tf.random_uniform([vdims[nv], int(np.log(vdims[nv])*SQ_RATE)], -1.0, 1.0), name='Wp_v' + str(nv + 1))
            bp_ = tf.Variable(tf.random_uniform([int(np.log(vdims[nv])*SQ_RATE)], -1.0, 1.0), name='bp_v' + str(nv + 1))
            Wq_ = tf.Variable(tf.random_uniform([int(np.log(vdims[nv])*SQ_RATE), hdim_2], -1.0, 1.0), name='Wq_v' + str(nv + 1))
            bq_ = tf.Variable(tf.random_uniform([hdim_2], -1.0, 1.0), name='bq_v' + str(nv + 1))

            self.Wp.append(Wp_)
            self.bp.append(bp_)
            self.Wq.append(Wq_)
            self.bq.append(bq_)
        self.Wa = tf.Variable(tf.random_uniform([hdim_2 * nview, hdim_2 * nview], -1.0, 1.0), name='Wa')
        # then concat and project
        self.Wr = tf.Variable(tf.random_uniform([hdim_2 * nview, cdim_1], -1.0, 1.0), name='Wr')
        self.br = tf.Variable(tf.random_uniform([cdim_1], -1.0, 1.0), name='br')
        self.Ws = tf.Variable(tf.random_uniform([cdim_1, cdim_2], -1.0, 1.0), name='Ws')
        self.bs = tf.Variable(tf.random_uniform([cdim_2], -1.0, 1.0), name='bs')
        self.view_metric = []
        for v in range(nview):
            L = tf.Variable(tf.random_uniform([hdim_2, cdim_2], -1.0, 1.0), name='Wr')
            self.view_metric.append(tf.matmul(L, tf.transpose(L)))
        #self.Wr = tf.concat(self.view_metric, axis=0)
        # params
        self.params = [self.Ws,self.Wr]
        for e in self.Wp:
            self.params.append(e)
        #for e in self.bp:
           # self.params.append(e)
        for e in self.Wq:
            self.params.append(e)
       # for e in self.bq:
            #self.params.append(e)

        # connect layers
        # p q r s
        self.Ap_rep = []
        self.Ap_sig = []
        self.Aq_rep = []
        self.Aq_sig = []

        self.Pp_rep = []
        self.Pp_sig = []
        self.Pq_rep = []
        self.Pq_sig = []

        self.Np_rep = []
        self.Np_sig = []
        self.Nq_rep = []
        self.Nq_sig = []


        # p layer
        for nv in range(nview):
            Ap_rep_ = tf.nn.xw_plus_b(self.input_A[nv], self.Wp[nv], self.bp[nv])
            Ap_sig_ = tf.sigmoid(Ap_rep_)

            Pp_rep_ = tf.nn.xw_plus_b(self.input_P[nv], self.Wp[nv], self.bp[nv])
            Pp_sig_ = tf.sigmoid(Pp_rep_)

            Np_rep_ = tf.nn.xw_plus_b(self.input_N[nv], self.Wp[nv], self.bp[nv])
            Np_sig_ = tf.sigmoid(Np_rep_)

            self.Ap_rep.append(Ap_rep_)
            self.Ap_sig.append(Ap_sig_)
            self.Pp_rep.append(Pp_rep_)
            self.Pp_sig.append(Pp_sig_)
            self.Np_rep.append(Np_rep_)
            self.Np_sig.append(Np_sig_)

        # q layer
        for nv in range(nview):
            Aq_rep_ = tf.nn.xw_plus_b(self.Ap_sig[nv], self.Wq[nv], self.bq[nv])
            Aq_sig_ = tf.sigmoid(Aq_rep_)

            Pq_rep_ = tf.nn.xw_plus_b(self.Pp_sig[nv], self.Wq[nv], self.bq[nv])
            Pq_sig_ = tf.sigmoid(Pq_rep_)

            Nq_rep_ = tf.nn.xw_plus_b(self.Np_sig[nv], self.Wq[nv], self.bq[nv])
            Nq_sig_ = tf.sigmoid(Nq_rep_)

            self.Aq_rep.append(Aq_rep_)
            self.Aq_sig.append(Aq_sig_)
            self.Pq_rep.append(Pq_rep_)
            self.Pq_sig.append(Pq_sig_)
            self.Nq_rep.append(Nq_rep_)
            self.Nq_sig.append(Nq_sig_)
        # 每个视图下都有一个原型

        self.vcenters = []
        vc = tf.Variable(tf.zeros([num_of_class, hdim_2]), name='cetners_of_view' + str(0 + 1))
        for nv in range(nview):
            self.vcenters.append(vc)

        # Aq_sig 代表每个视图的表示
        self.Avd2c = []
        self.Pvd2c = []
        self.Nvd2c = []
        for nv in range(nview):
            dist_f2c = distance(self.Aq_sig[nv], self.vcenters[nv])
            self.Avd2c.append(dist_f2c)
            dist_f2c = distance(self.Pq_sig[nv], self.vcenters[nv])
            self.Pvd2c.append(dist_f2c )
            dist_f2c = distance(self.Nq_sig[nv], self.vcenters[nv])
            self.Nvd2c.append(dist_f2c)
        #

        # concat layer
        self.A_concat = tf.concat(self.Aq_sig, axis=1)  #
        #self.A_concat = tf.concat(self.Avd2c, axis=1)
        #self.A_concat = tf.nn.softmax(self.A_concat)
        #self.A_concat = tf.matmul(self.A_concat,self.Wa)
        self.P_concat = tf.concat(self.Pq_sig, axis=1)
        #self.P_concat = tf.concat(self.Pvd2c, axis=1)
        #self.P_concat = tf.nn.softmax(self.P_concat)
        #self.P_concat = tf.matmul(self.P_concat, self.Wa)
        self.N_concat = tf.concat(self.Nq_sig, axis=1)
        #self.N_concat = tf.concat(self.Nvd2c, axis=1)
        #self.N_concat = tf.nn.softmax(self.N_concat)
        #self.N_concat = tf.matmul(self.N_concat, self.Wa)

        # common
        # r layer
        self.Ar_rep = tf.nn.xw_plus_b(self.A_concat, self.Wr, self.br)
        self.Ar_sig = tf.sigmoid(self.Ar_rep)
        self.Pr_rep = tf.nn.xw_plus_b(self.P_concat, self.Wr, self.br)
        self.Pr_sig = tf.sigmoid(self.Pr_rep)
        self.Nr_rep = tf.nn.xw_plus_b(self.N_concat, self.Wr, self.br)
        self.Nr_sig = tf.sigmoid(self.Nr_rep)

        # s layer
        self.As_rep = tf.nn.xw_plus_b(self.Ar_sig, self.Ws, self.bs)
        self.As_sig = tf.sigmoid(self.As_rep)
        self.Ps_rep = tf.nn.xw_plus_b(self.Pr_sig, self.Ws, self.bs)
        self.Ps_sig = tf.sigmoid(self.Ps_rep)
        self.Ns_rep = tf.nn.xw_plus_b(self.Nr_sig, self.Ws, self.bs)
        self.Ns_sig = tf.sigmoid(self.Ns_rep)

        # M layer

        self.AM = self.As_sig#tf.matmul(self.As_sig, self.L)
        self.PM = self.Ps_sig#tf.matmul(self.Ps_sig, self.L)
        self.NM = self.Ns_sig#tf.matmul(self.Ns_sig, self.L)

        AM = self.AM
        PM = self.PM
        NM = self.NM

        # 视图loss
        self.vpl_loss = []
        self.vdce_loss = []

        # 每个视图到每个center的距离
        for nv in range(nview):
            dist_f2c = self.Avd2c[nv]
            logits = -dist_f2c/1.0
            vdce_loss = softmax_loss(logits,self.input_L)
            self.vdce_loss.append(vdce_loss)
            vbatch_centers = tf.gather(self.vcenters[nv],self.input_L)
            vpl_loss = tf.reduce_mean(tf.reduce_sum(tf.square(self.Aq_sig[nv]-vbatch_centers),axis=1))
            self.vpl_loss.append(vpl_loss)


        # 单通道 以AM为输出
        # 类原型 类原型
        self.centers = tf.Variable(tf.zeros([num_of_class, cdim_2]),name='cetners')

        # 融合类别输出
        self.dist_f2c = distance(self.AM,self.centers) # batch_szie*NUM_OF_CLASS
        logits = -self.dist_f2c / 1.0
        self.dce_loss = softmax_loss(logits,self.input_L)
        batch_centers = tf.gather(self.centers,self.input_L)
        self.pl_loss = tf.reduce_mean(tf.reduce_sum(tf.square(self.AM-batch_centers),axis=1))

        # predict
        self.predict_label = predict(self.AM,self.centers)
        self.vpredict_label = [predict(self.Aq_sig[v], self.vcenters[v]) for v in range(nview)]
        #self.num_correct = tf.cast(tf.reduce_sum(tf.equal(self.predict_label,self.input_L)),tf.float32)
        w = 0.0
        for e in self.params:
            w = w + tf.nn.l2_loss(e)
        w = w * weight_decay
        for v in range(nview):
            row_ids = []
            for i in range(64):
                row_ids.append(i + v * 64)
            rows = tf.gather(self.Wr, row_ids)
            for i in range(self.Wr.shape[1]):
                v = tf.gather(rows,i,axis=1)
                #w = w + 0.01*tf.nn.l2_loss(v)


        self.loss = w + 1.0*self.dce_loss+0.001*self.pl_loss
        #self.loss += 0.0001*margin_proto_loss(self.centers,num_of_class,alpha)
        for v in range(nview):
            self.loss += 1.0*self.vdce_loss[v] # 1.5
            self.loss += 0.001*self.vpl_loss[v] #0.0015
            #self.loss += 0.0001*margin_proto_loss(self.vcenters[v],num_of_class,alpha)

        self.margin_loss = []
        for v in range(nview):
            self.margin_loss.append(margin_proto_loss(self.vcenters[v],num_of_class,alpha))

        global_step = tf.Variable(0, trainable=False)
        lr_step = tf.train.exponential_decay(learning_rate, global_step, 100, 0.9, staircase=True)  # 学习率递减
        self.opt = tf.train.AdamOptimizer(lr_step)
        self.update = self.opt.minimize(self.loss)


        #
        # extension
        self.AREP = AM
        self.ASIG = self.As_sig
        self.PREP = PM
        self.NREP = NM
