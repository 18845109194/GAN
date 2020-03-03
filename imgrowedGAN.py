import tensorflow as tf
import DataHundle as dh
import numpy as np
import time
import os
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

class ImporvedGAN(object):
    def __init__(self, noise_dim=100, img_h=64, img_w=64, mode="wgan"):
        """
        初始化GAN网络
        :param noise_dim:输入噪声维度
        :param img_h: 图像高度
        :param img_w: 图像宽度
        :param mode: 训练模式，可选：'wgan','lsgan','dcgan','wgan-gp'
        """
        self.noise_dim = noise_dim
        self.img_h = img_h
        self.img_w = img_w
        self.mode = mode
        self.img_c = 3
        self.fix_noise = np.random.normal(0, 1,(25, 1, 1, noise_dim))
        #placeholder()函数是在神经网络构建graph的时候在模型中的占位，此时并没有把要输入的数据传入模型，
        # 它只会分配必要的内存。等建立session，在会话中，运行模型的时候通过feed_dict()函数向占位符喂入数据。
        self.isTrain = tf.placeholder(dtype=tf.bool)  # bn算法需要
        self.gen_x = tf.placeholder(dtype=tf.float32, shape=[None, 1, 1, self.noise_dim])
        self.gen_out = self._init_generator(input=self.gen_x, isTrain=self.isTrain)
        self.gen_logis = self._init_discriminator(input=self.gen_out, isTrain=self.isTrain)#假数据放入判别器得到的结果
        self.x = tf.placeholder(dtype=tf.float32, shape=[None, self.img_w, self.img_h, self.img_c], name="input_data")#x是真实数据的值，shape中的none代表任意行
        self.real_logis = self._init_discriminator(input=self.x, isTrain=self.isTrain, reuse=True)#real_logis是真实数据放入判别器得到的结果
        self._init_train_methods()

    def _init_discriminator(self, input, isTrain=True, reuse=False):
        """
        初始化判别器
        :param input:输入数据op
        :param isTrain: 是否训练状态
        :param reuse: 是否可复用变量
        :return: 判断op
        """
        with tf.variable_scope('discriminator', reuse=reuse):
            # hidden layer 1 input =[none,64,64,3]
            conv1 = tf.layers.conv2d(input, 32, [3, 3], strides=(2, 2), padding='same')  # [none,32,32,32]，长32 宽32 32个卷积核
            bn1 = tf.layers.batch_normalization(conv1, training=isTrain)
            active1 = tf.nn.leaky_relu(bn1)  # [none,32,32,32]
            # hidden 2
            conv2 = tf.layers.conv2d(active1, 64, [3, 3], strides=(2, 2), padding='same')  # [none,16,16,64]
            bn2 = tf.layers.batch_normalization(conv2, training=isTrain)
            active2 = tf.nn.leaky_relu(bn2)  # [none,16,16,64]
            # hidden 3
            conv3 = tf.layers.conv2d(active2, 128, [3, 3], strides=(2, 2), padding="same")  # [none,8,8,128]
            bn3 = tf.layers.batch_normalization(conv3, training=isTrain)
            active3 = tf.nn.leaky_relu(bn3)  # [none,8,8,128]
            # hidden 4
            conv4 = tf.layers.conv2d(active3, 256, [3, 3], strides=(2, 2), padding="same")  # [none,4,4,256]
            bn4 = tf.layers.batch_normalization(conv4, training=isTrain)
            active4 = tf.nn.leaky_relu(bn4)  # [none,4,4,256]
            # out layer
            out_logis = tf.layers.conv2d(active4, 1, [4, 4], strides=(1, 1), padding='valid')  # [none,1,1,1]
        return out_logis

    def _init_generator(self, input, isTrain=True, reuse=False):
        """
        初始化生成器
        :param input:输入op
        :param isTrain: 是否训练状态
        :param reuse: 是否复用变量
        :return: 生成数据op
        """
        with tf.variable_scope('generator', reuse=reuse):
            # input [none,1,noise_dim]
            conv1 = tf.layers.conv2d_transpose(input, 512, [4, 4], strides=(1, 1), padding="valid")  # [none,4,4,512]
            bn1 = tf.layers.batch_normalization(conv1, training=isTrain)
            active1 = tf.nn.leaky_relu(bn1)  # [none,4,4,512]
            # deconv layer 2
            conv2 = tf.layers.conv2d_transpose(active1, 256, [3, 3], strides=(2, 2), padding="same")  # [none,8,8,256]
            bn2 = tf.layers.batch_normalization(conv2, training=isTrain)
            active2 = tf.nn.leaky_relu(bn2)  # [none,8,8,256]
            # deconv layer 3
            conv3 = tf.layers.conv2d_transpose(active2, 128, [3, 3], strides=(2, 2), padding="same")  # [none,16,16,128]
            bn3 = tf.layers.batch_normalization(conv3, training=isTrain)
            active3 = tf.nn.leaky_relu(bn3)  # [none,16,16,128]
            # deconv layer 4
            conv4 = tf.layers.conv2d_transpose(active3, 64, [3, 3], strides=(2, 2), padding="same")  # [none,32,32,64]
            bn4 = tf.layers.batch_normalization(conv4, training=isTrain)
            active4 = tf.nn.leaky_relu(bn4)  # [none,32,32,64]
            # out layer
            conv5 = tf.layers.conv2d_transpose(active4, 3, [3, 3], strides=(2, 2), padding="same")  # [none,64,64,3]
            out = tf.nn.tanh(conv5)
        return out

    def _init_dcgan_loss(self):
        # 初始化DCGAN损失函数，使用了交叉熵
        self.D_loss_real = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=self.real_logis, labels=tf.ones_like(self.real_logis)))
        self.D_loss_fake = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=self.gen_logis, labels=tf.zeros_like(self.gen_logis)))
        self.D_loss = self.D_loss_fake + self.D_loss_real
        self.G_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=self.gen_logis, labels=tf.ones_like(self.gen_logis)))

    def _init_lsgan_loss(self):
        # 初始化lsgan损失函数  均方误差损失
        self.G_loss = tf.reduce_mean((self.gen_logis - 1) ** 2)
        self.D_loss = 0.5 * (tf.reduce_mean((self.real_logis - 1) ** 2) + tf.reduce_mean((self.gen_logis - 0) ** 2))

    def _init_wgan_loss(self):
        # 初始化wgan损失函数
        self.D_loss = tf.reduce_mean(self.real_logis) - tf.reduce_mean(self.gen_logis)
        self.G_loss = tf.reduce_mean(self.gen_logis)

    def _init_wgan_gp_loss(self):
        # 初始化WGAN-gp损失函数
        # 构造梯度标准差
        tem_x = tf.reshape(self.x, [-1, self.img_w * self.img_h * self.img_c])#把四维数据转换成二维数据
        tem_gen_x = tf.reshape(self.gen_out, [-1, self.img_w * self.img_h * self.img_c])
        eps = tf.random_uniform([64, 1], minval=0., maxval=1.)#返回一个64*1的矩阵，值在0到1之间
        x_inter = eps * tem_x + (1 - eps) * tem_gen_x  # 真实数据与伪造数据平均值
        x_inter = tf.reshape(x_inter, [-1, self.img_w, self.img_h, self.img_c])
        grad = tf.gradients(self._init_discriminator(x_inter, isTrain=self.isTrain, reuse=True), [x_inter])[0]#求梯度
        grad_norm = tf.sqrt(tf.reduce_sum((grad) ** 2, axis=1))#标准差
        penalty = 10
        grad_pen = penalty * tf.reduce_mean((grad_norm - 1) ** 2)
        self.D_loss = tf.reduce_mean(self.real_logis) - tf.reduce_mean(self.gen_logis) + grad_pen
        self.G_loss = tf.reduce_mean(self.gen_logis)

    def _init_train_methods(self):
        """
        初始化训练方法：生成器与判别器损失，梯度下降方法，初始化session。
        :return: None
        """
        # 寻找生成器与判别器相关的变量
        total_vars = tf.trainable_variables()#trainable_variables可以支持传入scope，来获取指定scope中的变量集合
        d_vars = [var for var in total_vars if var.name.startswith("discriminator")]#把参数放在一起
        g_vars = [var for var in total_vars if var.name.startswith("generator")]
        if self.mode == "lsgan":
            self._init_lsgan_loss()
            self.D_trainer = tf.train.RMSPropOptimizer(learning_rate=1e-4).minimize(self.D_loss, var_list=d_vars)#RMSPropOptimizer是一种优化器
            self.G_trainer = tf.train.RMSPropOptimizer(learning_rate=1e-4).minimize(self.G_loss, var_list=g_vars)#选择learning_rate=1e-4，较好
        elif self.mode == "wgan":
            self._init_wgan_loss()
            self.clip_d = [p.assign(tf.clip_by_value(p, -0.1, 0.1)) for p in d_vars]#梯度裁剪，clip_by_value将值限制在-0.1到0.1之间
            self.D_trainer = tf.train.RMSPropOptimizer(learning_rate=5e-5).minimize(self.D_loss, var_list=d_vars)
            self.G_trainer = tf.train.RMSPropOptimizer(learning_rate=5e-5).minimize(self.G_loss, var_list=g_vars)
        elif self.mode == "wgan-gp":#改进后的wgan
            self._init_wgan_gp_loss()
            self.D_trainer = tf.train.AdamOptimizer(
                learning_rate=1e-4, beta1=0., beta2=0.9).minimize(self.D_loss, var_list=d_vars)
            self.G_trainer = tf.train.AdamOptimizer(
                learning_rate=1e-4, beta1=0., beta2=0.9).minimize(self.G_loss, var_list=g_vars)
        else:  # DCGAN
            self._init_dcgan_loss()
            self.D_trainer = tf.train.AdamOptimizer(
                learning_rate=2e-4, beta1=0.5).minimize(self.D_loss, var_list=d_vars)
            self.G_trainer = tf.train.AdamOptimizer(
                learning_rate=2e-4, beta1=0.5).minimize(self.G_loss, var_list=g_vars)
        # 初始化Session，如果使用了tf.variable就要初始化session
        self.sess = tf.InteractiveSession()
        self.sess.run(tf.global_variables_initializer())#意思是初始化全局所有变量
        self.saver = tf.train.Saver(max_to_keep=1)#保存模型

    def gen_data(self, fixed=True, save_path="out/ImprovedGAN/test.png"):
        """
        生成25张5*5的图像并保存
        :param fixed: 是否使用固定的噪声
        :param save_path: 保存路径
        :return: 保存图像numpy数组
        """
        # 生成随机噪声
        if fixed is False:
            self.fix_noise = np.random.normal(0, 1, (25, 1, 1, self.noise_dim))#生成0到1,25*1*1的噪声
        # 传入噪声生成数据
        samples = self.sess.run(self.gen_out, feed_dict={self.gen_x: self.fix_noise, self.isTrain: True})
        # samples [25,64,64,3] float -1到1之间
        # 转换到 0-1 或0-255
        samples = ((samples + 1) / 2 * 255).astype(np.uint8)
        fig = self.plot(samples)
        if not os.path.exists("out/ImprovedGAN/" + self.mode + "/"):
            os.makedirs("out/ImprovedGAN/" + self.mode + "/")
        plt.savefig(save_path, bbox_inches='tight')
        plt.close(fig)
        return samples

    def train(self, batch_size=64, itrs=100000, save_time=1000):
        """
        训练模型
        :param batch_size:采样数据量
        :param itrs: 迭代次数
        :param save_time: 保存次数
        :return: None
        """
        start_time = time.time()
        for i in range(itrs):
            if (self.mode =='dcgan') or (self.mode=='lsgan'):
                d_itrs = 1
            else:
                d_itrs = 5
            for i_d in range(d_itrs):
                # 读取真实图片
                batch_x = dh.read_img2numpy(batch_size=batch_size,img_h=64,img_w=64)
                #生成随机噪声
                batch_noise = np.random.normal(0,1,(batch_size,1,1,self.noise_dim))
                # 训练判别器
                if self.mode =="wgan":
                    _,D_loss_curr,_ = self.sess.run([self.D_trainer,self.D_loss,self.clip_d],
                                                  feed_dict={self.x:batch_x,self.gen_x:batch_noise,self.isTrain:True})
                else:
                    _,D_loss_curr = self.sess.run([self.D_trainer,self.D_loss],
                                                  feed_dict={self.x:batch_x,self.gen_x:batch_noise,self.isTrain:True})
            #训练生成器
            batch_noise = np.random.normal(0,1,(batch_size,1,1,self.noise_dim))
            _,G_loss_curr = self.sess.run([self.G_trainer,self.G_loss],
                                          feed_dict={self.gen_x:batch_noise,self.isTrain:True})

            if i %save_time ==0:
                # 生成数据
                self.gen_data(save_path="out/ImprovedGAN/"+self.mode+"/"+str(i).zfill(6)+".png")
                print("itrs:",i," D_loss:",D_loss_curr," G_loss",G_loss_curr)
                self.save()
                end_time = time.time()
                time_loss = end_time -start_time
                print("时间消耗:",int(time_loss),"秒")
                start_time = time.time()
        self.sess.close()

    def save(self, path="model/ImprovedGAN/"):
        """
        保存模型
        :param path: 模型保存路径
        :return: None
        """
        self.saver.save(sess=self.sess,save_path=path)

    def restore(self, path="model/ImprovedGAN/"):
        """
        恢复模型
        :param path:模型恢复路径
        :return: None
        """
        self.saver.restore(sess=self.sess,save_path=path)

    def plot(self, samples):
        """
        绘制图像
        :param samples: numpy数组
        :return: 绘制的图像
        """
        fig = plt.figure(figsize=(5,5))#创建画布五行五列
        gs = gridspec.GridSpec(5,5)#创建区域
        gs.update(wspace=0.05,hspace=0.05)#子图之间水平和垂直方向之间的距离
        for i,sample in enumerate(samples):
            ax = plt.subplot(gs[i])#依次画出子图
            plt.axis("off")#不显示坐标尺寸
            ax.set_xticklabels([])#坐标显示
            ax.set_yticklabels([])
            ax.set_aspect("equal")#横纵比相同
            plt.imshow(sample)
        return fig

if __name__ == '__main__':
    # 训练WGAN
    gan = ImporvedGAN(mode='wgan')
    gan.train()
    dh.img2gif(img_path="out/ImprovedGAN/wgan/",gif_path="out/ImprovedGAN/wgan/")
    # 训练WGAN-GP
    tf.reset_default_graph()#释放上一个模型，接着训练下一个模型
    gan = ImporvedGAN(mode='wgan-gp')
    gan.train()
    dh.img2gif(img_path="out/ImprovedGAN/wgan-gp/", gif_path="out/ImprovedGAN/wgan-gp/")
    # 训练LSGAN
    tf.reset_default_graph()
    gan = ImporvedGAN(mode='lsgan')
    gan.train()
    dh.img2gif(img_path="out/ImprovedGAN/lsgan/", gif_path="out/ImprovedGAN/lsgan/")
    # 训练DCGAN
    tf.reset_default_graph()
    gan = ImporvedGAN(mode='dcgan')
    gan.train()
    dh.img2gif(img_path="out/ImprovedGAN/dcgan/", gif_path="out/ImprovedGAN/dcgan/")
