import tensorflow as tf
import numpy as np
import time
import os
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import DataHundle as dh
class CGAN_MNIST(object):
    def __init__(self, noise_dim=10, img_h=32, img_w=32, img_c=1, lr=0.0004):
        """
        初始化CGAN对象
        :param noise_dim:随机噪声维度
        :param img_h: 图像高度
        :param img_w: 图像宽度
        :param img_c: 图像深度
        :param lr: 学习
        """
        self.noise_dim = noise_dim
        self.img_h = img_h
        self.img_w = img_w
        self.img_c = img_c
        self.lr = lr
        self.d_dim = 1
        self.label = tf.placeholder(dtype=tf.float32, shape=[None, 10])#0~9一共10个数字，所以设置为10维
        self.isTrain = tf.placeholder(dtype=tf.bool)
        self.gen_x = tf.placeholder(dtype=tf.float32, shape=[None, 1, 1, self.noise_dim])#任意行，1*1大小
        self.gen_out = self._init_generator(input=self.gen_x, label=self.label, istrain=self.isTrain)  # 生成数据
        self.gen_logis = self._init_discriminator(input=self.gen_out, label=self.label, isTrain=self.isTrain)
        self.x = tf.placeholder(dtype=tf.float32, shape=[None, self.img_w, self.img_h, self.img_c], name="input_data")
        self.real_logis = self._init_discriminator(input=self.x, label=self.label, isTrain=self.isTrain, reuse=True)
        self._init_train_methods()

    def _init_discriminator(self, input, label, isTrain=True, reuse=False):
        """
        初始化判别器
        :param input:输入数据op
        :param label: 输入数据标签
        :param isTrain: 是否训练状态
        :param reuse: 是否复用内部参数
        :return: 判断结果
        """
        with tf.variable_scope("discriminator", reuse=reuse):
            # hidden layer 1  input=[None,32,32,1]
            labels = tf.reshape(label, shape=[-1, 1, 1, 10])
            data_sum = tf.shape(input=input)[0]
            la = labels * tf.ones([data_sum, self.img_w, self.img_h, 10])
            input = tf.concat([input, la], axis=3)#用来拼接张量的函数
            conv1 = tf.layers.conv2d(input, 32, [4, 4], strides=[2, 2], padding="same")
            active1 = tf.nn.leaky_relu(conv1) #[none,16,16,32]
            # layer 2
            conv2 = tf.layers.conv2d(active1, 64, [4, 4], strides=[2, 2], padding="same")
            bn2 = tf.layers.batch_normalization(conv2,training=isTrain)#[none,8,8,64]
            active2 = tf.nn.leaky_relu(bn2)#[none,8,8,64]
            # layer 3
            conv3 = tf.layers.conv2d(active2, 128, [4, 4], strides=[2, 2], padding="same")
            bn3 = tf.layers.batch_normalization(conv3,training=isTrain)
            active3 = tf.nn.leaky_relu(bn3)#[none,4,4,128]
            # out layer
            out_logis = tf.layers.conv2d(active3,1,[4,4],strides=[1,1],padding="valid") #[none,1,1,1]
        return out_logis

    def _init_generator(self, input, label, istrain=True, resue=False):
        """
        初始化生成器
        :param input:输入噪声
        :param label: 输入数据标签
        :param istrain: 是否训练状态
        :param resue: 是否复用内部参数
        :return: 生成数据op
        """
        with tf.variable_scope("generator",reuse=resue):
            # input[none,1,1,self.noise_dim]
            labels = tf.reshape(label,shape=[-1,1,1,10])
            input = tf.concat([input,labels],axis=3)
            # hidden layer 1
            conv1 = tf.layers.conv2d_transpose(input,256,[4,4],strides=(1,1),padding="valid")
            bn1 = tf.layers.batch_normalization(conv1,training=istrain)
            active1 = tf.nn.leaky_relu(bn1) #[none,4,4,256]

            conv2 = tf.layers.conv2d_transpose(active1,128,[4,4],strides=(2,2),padding="same")
            bn2 = tf.layers.batch_normalization(conv2,training=istrain)
            active2 = tf.nn.leaky_relu(bn2) #[none,8,8,128]
            # layer 3
            conv3 = tf.layers.conv2d_transpose(active2,64,[4,4],strides=(2,2),padding='same')
            bn3 = tf.layers.batch_normalization(conv3,training=istrain)
            active3 = tf.nn.leaky_relu(bn3) # [none,16,16,64]
            # out layer
            conv4 = tf.layers.conv2d_transpose(active3,self.img_c,[4,4],strides=(2,2),padding="same")
            out = tf.nn.sigmoid(conv4)  # (0,1)        [None,32,32,1]
        return out

    def _init_train_methods(self):
        """
        初始化训练方法：梯度下降方法，Session初始化,损失函数。
        :return: NONE
        """
        self.D_loss_real =tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=self.real_logis,labels=tf.ones_like(self.real_logis))
        )
        self.D_loss_fake = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=self.gen_logis,labels=tf.zeros_like(self.gen_logis))
        )
        self.D_loss = self.D_loss_real +self.D_loss_fake
        self.G_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=self.gen_logis,labels=tf.ones_like(self.gen_logis))
        )
        total_vars = tf.trainable_variables()
        d_vars =[var for var in total_vars if var.name.startswith('discriminator')]
        g_vars = [var for var in total_vars if var.name.startswith('generator')]
        self.D_trainer = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(
            self.D_loss,var_list=d_vars
        )
        self.G_trainer = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(
            self.G_loss,var_list = g_vars
        )

        self.sess = tf.InteractiveSession()
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver(max_to_keep=1)

    def gen_data(self, label, save_path="out/CGAN_MNIST/test.png"):
        """
        生成数字图像
        :param label:数字标签
        :param save_path: 保存路劲
        :return: 图像数组 numpy
        """
        batch_noise = np.random.normal(0,1,(25,1,1,self.noise_dim))
        print(np.argmax(label,axis=1))
        samples = self.sess.run(self.gen_out,feed_dict={self.gen_x:batch_noise,self.label:label,self.isTrain:True})
        # [25,32,32,1] ---->[25,32,32]
        samples =np.reshape(samples,[-1,32,32])
        fig = self.plot(samples)
        if not os.path.exists('out/CGAN_MNIST/'):
            os.makedirs('out/CGAN_MNIST/')
        plt.savefig(save_path)
        plt.close(fig)
        return samples

    def train(self, batch_size=64, itrs=100000, save_time=1000):
        """
        训练模型方法
        :param batch_size:批量采样大小
        :param itrs: 迭代次数
        :param save_time: 保存模型周期
        :return: NONE
        """
        # 读取数据
        data = dh.load_mnist_resize()
        start_time = time.time()
        test_y = data['label'][0:25]
        for i in range(itrs):
            #随机采样数据
            mask = np.random.choice(data['data'].shape[0],batch_size,replace=True)
            batch_x = data['data'][mask]
            batch_y = data['label'][mask]
            batch_noise = np.random.normal(0,1,(batch_size,1,1,self.noise_dim))
            #训练判别器
            _,D_loss_curr =self.sess.run([self.D_trainer,self.D_loss],feed_dict={
                self.x:batch_x,self.gen_x:batch_noise,self.label:batch_y,self.isTrain:True
            })
            #训练生成器
            batch_noise = np.random.normal(0,1,[batch_size,1,1,self.noise_dim])
            _,G_loss_curr = self.sess.run([self.G_trainer,self.G_loss],feed_dict={
                self.gen_x:batch_noise,self.label:batch_y,self.isTrain:True
            })
            if i%save_time ==0:
                self.gen_data(label=test_y,save_path='out/CGAN_MNIST/'+str(i).zfill(6)+".png")
                print("i:",i," D_loss:",D_loss_curr," G_loss",G_loss_curr)
                self.save()
                end_time = time.time()
                time_loss = end_time-start_time
                print("训练时间：",int(time_loss),"秒")
                start_time =time.time()
        self.sess.close()

    def save(self, path="model/CGAN_MNIST/"):
        """
        保存模型
        :param path:保存路径
        :return: NONE
        """
        self.saver.save(self.sess,save_path=path)

    def restore(self, path='model/CGAN_MNIST/'):
        """
        恢复模型
        :param path:模型保存路径
        :return: NONE
        """
        self.saver.restore(sess=self.sess,save_path=path)

    def plot(self, smaples):
        """
        绘制5*5图片
        :param smaple: numpy数组
        :return: 绘制图像
        """
        fig = plt.figure(figsize=(5,5))
        gs = gridspec.GridSpec(5,5)
        gs.update(wspace=0.05,hspace=0.05)
        for i,smaple in enumerate(smaples):
            ax = plt.subplot(gs[i])
            plt.axis("off")
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_aspect('equal')
            plt.imshow(smaple,cmap='Greys_r')
        return fig

if __name__ == '__main__':
    gan = CGAN_MNIST()
    gan.train()
    dh.img2gif(img_path="out/CGAN_MNIST/",gif_path="out/CGAN_MNIST/")
