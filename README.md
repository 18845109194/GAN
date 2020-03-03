# GAN生成对抗网络
## 简介：
>>GAN网络提供了一种高效数据生成的深度学习方法，不需要或需要很少的标注数据，通过生成网络和判别网络之间的竞争来获得高质量的生成数据。  
GAN的判别模型：需判断输入的是真实的图像还是由模型生成的图像。判别模型一般可以通过决策函数y=f（x）或者条件概率p（y|x）来判断输入x是否属于y类。决策函数判别模型通过训练图像集（有标签）来确定特征空间中不同类别的分界线，此后，输入图像特征落在哪个类别的范围内，它就属于此类别。典型的方法有SVM、感知机、k-近邻。条件概率是一种基于统计特性的判别方法，通过训练图像（特征）获得条件概率分布，计算输入图像（特征）属于各个类别的概率，概率最大的那个类别就是该图像所属的类别。决策函数判别方法简单但是难以对付复杂的类别划分，因此常采用条件概率判别模型。  
生成模型：可以用于图像的生成，还可以与判别模型一样用于图像的分类。生成模型可以自动学习到训练图像的内部分布，能够解释给定的训练图像，同时生成新的类似于图像的样本。  
GAN工作开始时，生成模型产生一幅图像去“欺骗”判别模型，然后判别模型判断此图像是真是假，将判别结果的误差反传给这两个模型，各自改进网络参数，提高自己的生成能力和判别能力。如此反复，不断优化，两个模型的能力越来越强，最终达到平衡状态。
## 传统GAN的实现
涉及到的文件:gan.py ，DataHundle.py,MNIST数据集
>>gan的核心是训练判别器和生成器  
训练判别器:首先在真实数据集中采样数据，并标记为1。随机噪声(正太分布/高斯分布）通过生成器生成伪造数据，标记为0。两者通过判别器之后进行前向传播，然后得到损失值，进行反向传播。训练判别器时锁住生成器，生成器不进行训练。  
训练生成器:对生成器生成的数据进行采样，进行前向传播，得到损失值，进行反向传播。锁住判别器，判别器不进行训练，但是要提供反向传播梯度。   
迭代237000次后的结果如下图所示：  

![Image text](https://github.com/18845109194/my/blob/master/237000.png) 

由于1相比0~9中其他的数字较容易生成，所以可能出现生成器生成的数字全为1的情况。
## DCGAN  
涉及到的文件：dcgan.py,DataHundle.py,faces数据集  
>>使用卷积神经网络架构，对gan进行扩展。主要是将原来简单的神经网络改为利用卷积核对数据进行特征提取(多次进行卷积、归一化、激活操作)。  
faces数据集中为动漫人脸形象，其作用与gan中的mnist数据集相同。  
通过19800次迭代，结果如下：  

![Image text](https://github.com/18845109194/my/blob/master/dcgan19800.png)  

## LSGAN
涉及到的文件：imgrowedgan.py,DataHundle.py,faces数据集
>>使用线性激活函数替代sigmoid激活函数。损失函数采用均方误差损失。  

## WGAN
涉及到的文件：imgrowedgan.py,DataHundle.py,faces数据集
>>损失函数使用了梯度裁剪，强制参数在某个区间(例如:[-c，c]），如果w＞c，w＝c；如果w＜-c，w＝-c。  

## WGAN-gp
涉及到的文件：imgrowedgan.py,DataHundle.py,faces数据集
>>在wgan的基础上进行改进，解决wgan产生低质量数据或错误收敛的问题。主要使用约束惩罚即梯度标准差，只对真实数据和生成数据之间的区域的分布给予梯度约束，该区域影响生成数据向真实数据进行移动。  

## CGAN生成数字图片
涉及到的文件：CGAN_MNIST.py，DataHundle.py,MNIST数据集
>>在判别器和生成器的输入加入条件c，判别器除了判断x是不是真实图片还要判断c和x是否匹配。  

## CGAN图片生成图片
涉及到的文件：CGAN_img2img.py，DataHundle.py，facade数据集
>>将建筑物草图放入生成器中，生成的图片与真实的建筑图片进行对比，得到损失值，将误差进行反向传播。  
生成器产生的图片与真实图片一起进入判别器进行判断，得到损失值，进行反向传播。  
下图中从左到右依次为：生成结果、输入草图、真实数据；生成结果、输入草图、真实数据。

![Image text](https://github.com/18845109194/my/blob/master/cgan-img2img000000.png)

