import tensorflow as tf
import numpy as np
import time
import os
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import DataHundle as dh
class CGAN_img2img(object):
    def __init__(self, img_h=64, img_w=64, img_c=3, lr=0.0004):
        """
        初始化CGAN_img2img对象
        :param img_h: 图像高度
        :param img_w: 图像宽度
        :param img_c: 图像深度
        :param lr: 学习率
        """
        self.img_h =img_h
        self.im_w =img_w
        self.img_c = img_c
        self.lr = lr
        self.d_dim =1
        self.isTrain = tf.placeholder(dtype=tf.bool)
        self.gen_x = tf.placeholder(dtype=tf.float32,shape=[None,img_h,img_w,img_c])# 生成器的输入
        self.condition = tf.placeholder(dtype=tf.float32,shape=[None,img_h,img_w,img_c]) # 输入条件
        self.gen_out =self._init_generator(input=self.gen_x,istrain=self.isTrain) #生成器输出
        self.gen_logis = self._init_discriminator(input=self.gen_out,label=self.condition,isTrain=self.isTrain)
        self.x = tf.placeholder(dtype=tf.float32,shape=[None,img_w,img_h,img_c],name="input_data")
        self.real_logis =self._init_discriminator(input=self.x,label=self.condition,isTrain=True,reuse=True)
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
        with tf.variable_scope('discriminator',reuse=reuse):
            # input [none,64,64,3]  label [none,64,64,3]
            input = tf.concat([input,label],axis=3)
            conv1 = tf.layers.conv2d(input,32,[4,4],strides=(2,2),padding="same") #[none,32,32,32]
            bn1 =tf.layers.batch_normalization(conv1,training=isTrain)
            active1 = tf.nn.leaky_relu(bn1) #[none,32,32,32]
            #layer 2
            conv2 = tf.layers.conv2d(active1,64,[4,4],strides=(2,2),padding="same") #[none,16,16,64]
            bn2 = tf.layers.batch_normalization(conv2,training=isTrain)
            active2 = tf.nn.leaky_relu(bn2) # [none,16,16,64]
            # layer 3
            conv3 = tf.layers.conv2d(active2, 128, [4, 4], strides=(2, 2), padding="same")  # [none,8,8,128]
            bn3 = tf.layers.batch_normalization(conv3, training=isTrain)
            active3 = tf.nn.leaky_relu(bn3)  # [none,8,8,128]
            # layer 4
            conv4 = tf.layers.conv2d(active3, 256, [4, 4], strides=(2, 2), padding="same")  # [none,4,4,256]
            bn4 = tf.layers.batch_normalization(conv4, training=isTrain)
            active4 = tf.nn.leaky_relu(bn4)  # [none,4,4,256]
            # out layer
            conv5 =tf.layers.conv2d(active4,1,[4,4],strides=(1,1),padding="valid") #[none,1,1,1]
            out = tf.reshape(conv5,shape=[-1,1])
        return out


    def _init_generator(self, input, istrain=True, resue=False):
        """
        初始化生成器
        :param input:输入噪声
        :param istrain: 是否训练状态
        :param resue: 是否复用内部参数
        :return: 生成数据op
        """
        with tf.variable_scope("generator",reuse=resue):
            # input [none,64,64,3]
            conv1 = tf.layers.conv2d(input,64,[4,4],strides=(2,2),padding="same") #[none,32,32,64]
            bn1 = tf.layers.batch_normalization(conv1,training=istrain)
            active1 = tf.nn.leaky_relu(bn1) #[none,32,32,64]
            # conv2
            conv2 = tf.layers.conv2d(active1, 128, [4, 4], strides=(2, 2), padding="same")  # [none,16,16,128]
            bn2 = tf.layers.batch_normalization(conv2, training=istrain)
            active2 = tf.nn.leaky_relu(bn2)  # [none,16,16,128]
            # conv3
            conv3 = tf.layers.conv2d(active2, 256, [4, 4], strides=(2, 2), padding="same")  # [none,8,8,256]
            bn3 = tf.layers.batch_normalization(conv3, training=istrain)
            active3 = tf.nn.leaky_relu(bn3)  # [none,8,8,256]
            # conv4
            conv4 = tf.layers.conv2d(active3, 512, [4, 4], strides=(2, 2), padding="same")  # [none,4,4,512]
            bn4 = tf.layers.batch_normalization(conv4, training=istrain)
            active4 = tf.nn.leaky_relu(bn4)  # [none,4,4,512]
            # deconv1
            de1 = tf.layers.conv2d_transpose(active4,256,[4,4],strides=(2,2),padding="same") #[none,8,8,256]
            de_bn1 = tf.layers.batch_normalization(de1,training=istrain)
            de_active1 = tf.nn.leaky_relu(de_bn1) #[none,8,8,256]
            # deconv2
            de2 = tf.layers.conv2d_transpose(de_active1, 128, [4, 4], strides=(2, 2), padding="same")  # [none,16,16,128]
            de_bn2 = tf.layers.batch_normalization(de2, training=istrain)
            de_active2 = tf.nn.leaky_relu(de_bn2)  # [none,16,16,128]
            # deconv3
            de3 = tf.layers.conv2d_transpose(de_active2, 64, [4, 4], strides=(2, 2), padding="same")  # [none,32,32,64]
            de_bn3 = tf.layers.batch_normalization(de3, training=istrain)
            de_active3 = tf.nn.leaky_relu(de_bn3)  # [none,32,32,64]
            # deconv4
            de4 = tf.layers.conv2d_transpose(de_active3, 3, [4, 4], strides=(2, 2), padding="same")  # [none,64,64,3]
            de_bn4 = tf.layers.batch_normalization(de4, training=istrain)
            out = tf.nn.tanh(de_bn4)  # [none,64,64,3]
        return out


    def _init_train_methods(self):
        """
        初始化训练方法：梯度下降方法，Session初始化,损失函数。
        :return: NONE
        """
        # 寻找判别器与生成器训练参数
        total_vars = tf.trainable_variables()
        d_vars = [var for var in total_vars if var.name.startswith('discriminator')]
        g_vars = [var for var in total_vars if var.name.startswith('generator')]
        # 构造损失函数
        self.D_loss_real = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=self.real_logis,labels=tf.ones_like(self.real_logis))
        )
        self.D_loss_fake = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=self.gen_logis,labels=tf.zeros_like(self.gen_logis))
        )
        self.D_loss = self.D_loss_fake+self.D_loss_real

        #构造生成器损失
        gen_loss_l1 = tf.reduce_mean(tf.abs(self.gen_out-self.x))
        gen_loss_d = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=self.gen_logis,labels=tf.ones_like(self.gen_logis))
        )
        beta = 0.7
        self.G_loss = beta*gen_loss_l1 +(1-beta)*gen_loss_d
        #构造训练方法
        self.D_trainer = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(
            self.D_loss,var_list=d_vars
        )
        self.G_trainer = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(
            self.G_loss,var_list=g_vars
        )
        # 初始化session
        self.sess = tf.InteractiveSession()
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver(max_to_keep=1)


    def gen_data(self,data, label, save_path="out/CGAN_img2img/test.png"):
        """
        生成36张6*6的图片，第一列生成图片，第二列输入数据，第三列真实目标值.....
        :param label:真实图片
        :param data:输入数据
        :param save_path: 保存路劲
        :return: 图像数组 numpy
        """
        samples = self.sess.run(self.gen_out,feed_dict={
            self.gen_x:data,self.isTrain:True
        })
        # samples [none,64,64,3]  float   (-1,1)
        samples = ((samples+1)/2*255).astype(np.uint8)
        label =((label+1)/2*255).astype(np.uint8)
        data = ((data+1)/2*255).astype(np.uint8)
        fig = self.plot(samples,data,label)
        if not os.path.exists("out/CGAN_img2img/"):
            os.makedirs("out/CGAN_img2img/")
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
        start_time = time.time()
        train_x,train_y = dh.read_all_facade2numpy()
        for i in range(itrs):
            mask =np.random.choice(train_x.shape[0],batch_size,replace=True)
            batch_x = train_x[mask]
            batch_y = train_y[mask]
            _,D_loss_curr = self.sess.run([self.D_trainer,self.D_loss],feed_dict={
                self.x:batch_y,self.gen_x:batch_x,self.condition:batch_x,self.isTrain:True
            })
            #训练生成器
            mask = np.random.choice(train_x.shape[0], batch_size, replace=True)
            batch_x = train_x[mask]
            batch_y = train_y[mask]
            _,G_loss_curr = self.sess.run([self.G_trainer,self.G_loss],feed_dict={
               self.x:batch_y,self.gen_x:batch_x,self.condition:batch_x,self.isTrain:True
            })
            if i%save_time ==0:
                mask = np.random.choice(train_x.shape[0],12,replace=True)
                test_x = train_x[mask]
                test_y =train_y[mask]
                self.gen_data(data=test_x,label=test_y,save_path="out/CGAN_img2img/"+str(i).zfill(6)+".png")
                print("i:",i," D_loss:",D_loss_curr," G_loss",G_loss_curr)
                self.save()
                end_time = time.time()
                time_loss = end_time-start_time
                print("时间消耗:",int(time_loss),"秒")
                start_time = time.time()
        self.sess.close()


    def save(self, path="model/CGAN_img2img/"):
        """
        保存模型
        :param path:保存路径
        :return: NONE
        """
        self.saver.save(self.sess,save_path=path)

    def restore(self, path='model/CGAN_img2img/'):
        """
        恢复模型
        :param path:模型保存路径
        :return: NONE
        """
        self.saver.restore(sess=self.sess,save_path=path)

    def plot(self, smaples,data,label):
        """
        绘制6*6图片矩阵
        :param smaples:生成图片
        :param data: 输入图片
        :param label: 真实图片
        :return: 绘制图像
        """
        fig = plt.figure(figsize=(6,6))
        gs = gridspec.GridSpec(6,6)
        gs.update(wspace=0.05,hspace=0.05)
        for i in range(12):
            # 绘制生成图片
            ax = plt.subplot(gs[3*i])
            plt.axis("off")
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_aspect("equal")
            plt.imshow(smaples[i])
            #绘制输入图片
            ax = plt.subplot(gs[3 * i+1])
            plt.axis("off")
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_aspect("equal")
            plt.imshow(data[i])
            #绘制真实图片
            ax = plt.subplot(gs[3 * i+2])
            plt.axis("off")
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_aspect("equal")
            plt.imshow(label[i])
        return  fig


if __name__ == '__main__':
    gan = CGAN_img2img()
    gan.train()
