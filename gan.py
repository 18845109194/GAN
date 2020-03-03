import numpy as np
import tensorflow as tf
import DataHundle as dh
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os


class GAN(object):
    def __init__(self, noise_dim=10, gen_hidden=[100, 100], gen_dim=784, d_hidden=[100, 100], lr=0.001, std=0.01):
        '''
        初始化GAN对象
        :param noise_dim:随机噪声维度
        :param gen_hidden: 生成器隐藏层形状
        :param gen_dim: 生成器输出维度
        :param d_hidden: 判别器隐藏层形状
        :param lr: 学习率
        :param std: 权重标准差
        '''

        self.noise_dim = noise_dim
        self.gen_hidden = gen_hidden
        self.gen_dim = gen_dim
        self.d_hidden = d_hidden
        self.lr = lr
        self.std = std
        self.d_dim = 1  # 判别器输出维度为1
        self._init_w_g()  # 初始化生成器的权重
        self._init_w_d()  # 初始化判别器权重
        # 构造生成器网络结构
        self.gen_out = self._init_gen()
        # 判别器输入生成器数据
        self.gen_logis = self._init_dicriminator(self.gen_out)
        # 构造判别器网络结构
        self.x = tf.placeholder(dtype=tf.float32, shape=[None, self.gen_dim], name="input_data")
        self.real_logis = self._init_dicriminator(self.x)
        # 初始化训练方法
        self._init_train_methods()

    def _init_w_g(self):
        '''
        初始化生成器
        :return:
        '''

        self.w_g_list=[]
        self.b_g_list=[]
        self.w_g=[]
        #初始化输入层权重
        g_w=self._init_variable(shape=[self.noise_dim,self.gen_hidden[0]],name="gen_w0")
        g_b=self._init_variable(shape=[self.gen_hidden[0]],name="gen_b0")
        self.w_g_list.append(g_w)#append(x ) 方法向列表的尾部添加一个新的元素x
        self.b_g_list.append(g_b)
        self.w_g.append(g_w)
        self.w_g.append(g_b)#感觉有问题
        #初始化生成器的隐藏层权重【100,100】
        for i in range(len(self.gen_hidden)-1):
            g_w=self._init_variable(shape=[self.gen_hidden[i],self.gen_hidden[i+1]],name="gen_w"+str(i+1))
            g_b=self._init_variable(shape=[self.gen_hidden[i+1]],name="gen_b"+str(i+1))
            self.w_g_list.append(g_w)
            self.b_g_list.append(g_b)
            self.w_g.append(g_w)
            self.w_g.append(g_b)
        #初始化生成器输出层权重
        g_w=self._init_variable(shape=[self.gen_hidden[-1],self.gen_dim],name="gen_W_out")
        g_b=self._init_variable(shape=[self.gen_dim],name="gen_b_out")
        self.w_g_list.append(g_w)
        self.b_g_list.append(g_b)
        self.w_g.append(g_w)
        self.w_g.append(g_b)


    def _init_w_d(self):
        '''
        初始化判别器权重
        :return:
        '''
        self.w_d_list=[]
        self.b_d_list=[]
        self.w_d=[]
        #初始化判别器输入层权重
        d_w=self._init_variable(shape=[self.gen_dim,self.d_hidden[0]],name="d_w0")
        d_b=self._init_variable(shape=[self.gen_hidden[0]],name="d_b0")
        self.w_d_list.append(d_w)
        self.b_d_list.append(d_b)
        self.w_d.append(d_w)
        self.w_d.append(d_b)
        #初始化判别器隐藏层权重
        for i in range(len(self.d_hidden)-1):
            d_w=self._init_variable(shape=[self.d_hidden[i],self.d_hidden[i+1]],name="d_w"+str(i+1))
            d_b=self._init_variable(shape=[self.d_hidden[i+1]],name="d_b"+str(i+1))
            self.w_d_list.append(d_w)
            self.b_d_list.append(d_b)
            self.w_d.append(d_w)
            self.w_d.append(d_b)
        #初始化判别器输出层权重

        d_w = self._init_variable(shape=[self.d_hidden[-1], self.gen_dim], name="d_w_out")
        d_b = self._init_variable(shape=[self.d_dim], name="d_b_out")
        self.w_d_list.append(d_w)
        self.b_d_list.append(d_b)
        self.w_d.append(d_w)
        self.w_d.append(d_b)

    def _init_dicriminator(self, input_op):
        '''
        初始化判别器网络结构
        网络结构：比如：[gen_dim,100]*[100,100]*[100,1]
        :param op: 输入op
        :return: 判别器op
        '''
        #构造判别器输入层
        active=tf.nn.relu(tf.matmul(input_op,self.w_d_list[0])+self.b_d_list[0])
        #构造判别器隐藏层
        for i in range(len(self.d_hidden)-1):
            active=tf.nn.relu(tf.matmul(active,self.w_d_list[i+1])+self.b_d_list[i+1])
        #构造判别器输出层
        out_logis=tf.matmul(active,self.w_d_list[-1])+self.b_d_list[-1]
        return out_logis

    def _init_gen(self):
        '''
        初始化生成器网络结构
        网络结构 比如：[noise_dim,100]*[100,100]*[100,784]
        :return:生成数据op
        '''
        self.gen_x=tf.placeholder(dtype=tf.float32,shape=[None,self.noise_dim],name="gen_x")
        #构造生成器输入层
        active=tf.nn.relu(tf.matmul(self.gen_x,self.w_g_list[0])+self.b_g_list[0])
        #构造生成器隐藏层
        for i in range(len(self.gen_hidden)-1):
            active=tf.nn.relu(tf.matmul(active,self.w_g_list[i+1])+self.b_g_list[i+1])
        #构造输出层
        out_logis=tf.matmul(active,self.w_g_list[-1])+self.b_g_list[-1]
        g_out=tf.nn.sigmoid(out_logis)
        return g_out

    def _init_variable(self, shape, name):
        '''
        初始化变量
        :param shape:变量形状
        :param name: 变量名称
        :return: 变量
        '''
        return tf.Variable(tf.truncated_normal(shape=shape,stddev=self.std),name=name)

    def train(self, data_dict, batch_size=100, itrs=1000000):
        '''
        训练模型
        :param data_dict:训练数据字典
        :param batch_size: 批量采样大小
        :param itrs: 迭代次数
        :return:
        '''
        for i in range(itrs):
            mask=np.random.choice(data_dict["train_x"].shape[0],batch_size,replace=True)
            batch_x=data_dict["train_x"][mask]
            batch_noise=self.sample_Z(m=batch_size,n=self.noise_dim)
            _,D_loss_curr=self.sess.run([self.D_trainer,self.D_loss],feed_dict={self.x:batch_x,self.gen_x:batch_noise})
            _,G_loss_curr=self.sess.run([self.G_trainer,self.G_loss],feed_dict={self.gen_x:batch_noise})
            if i % 1000==0:
                self.gen_data(save_path="out/"+str(i)+".png")
                print("迭代次数:,",i,"D_loss:",D_loss_curr,"G_loss:",G_loss_curr)
                self.save()
        self.sess.close()


    def _init_train_methods(self):
        '''
        初始化训练方法，初始化判别器，训练方法，session
        :return:
        '''
        #初始化判别器损失函数
        self.D_loss_real=tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.real_logis,labels=tf.ones_like(self.real_logis)))
        self.D_loss_fake=tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.gen_logis,labels=tf.zeros_like(self.gen_logis)))
        self.D_loss=self.D_loss_real+self.D_loss_fake
        #初始化生成器的损失值
        self.G_loss=tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.gen_logis,labels=tf.ones_like(self.gen_logis)))
        #构造训练方法
        self.D_trainer=tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.D_loss,var_list=self.w_d)
        self.G_trainer=tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.G_loss,var_list=self.w_g)
        #初始化Session
        self.sess=tf.InteractiveSession()
        self.sess.run(tf.global_variables_initializer())
        self.saver=tf.train.Saver(max_to_keep=1)


    def gen_data(self, save_path="out/test.png"):
        '''
        生成数字图像并保存
        :param save_path:保存路径
        :return:
        '''
        batch_noise=self.sample_Z(9,self.noise_dim)
        sample=self.sess.run(self.gen_out,feed_dict={self.gen_x:batch_noise})
        fig=self.plot(sample)
        if not os.path.exists("out/"):
            os.makedirs("out/")
        plt.savefig(save_path,bbox_inches='tight')
        plt.close(fig)
        return sample

    def save(self, path="model/gan/"):
        '''
        保存模型
        :param path:保存模型路径
        :return:
        '''
        self.saver.save(self.sess,save_path=path)


    def restroe(self, path="model/gan/"):
        '''
        恢复模型
        :param path:模型保存的路径
        :return:
        '''
        self.saver.restore(sess=self.sess,save_path=path)

    def sample_Z(self, m, n):
        '''
        生成随机噪声
        :param m: 生成数据量
        :param n: 随机噪声维度
        :return: numpy数组
        '''
        return np.random.uniform(-1.,1.,size=[m,n])


    def plot(self, samples):
        '''
        绘制图像
        :param sample:绘制数据（numpy类型）
        :return:

        '''
        fig=plt.figure(figsize=(3,3))
        gs=gridspec.GridSpec(3,3)
        for i,sample in enumerate(samples):
            ax=plt.subplot(gs[i])
            plt.axis("off")
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_aspect('equal')
            plt.imshow(sample.reshape(28,28),cmap='Greys_r')
        return fig
if __name__=='__main__':
    #读入数据
    data=dh.load_minst()
    #初始化GAN对象
    gan=GAN(noise_dim=10,gen_dim=784)
    gan.train(data_dict=data)
