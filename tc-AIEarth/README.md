# 前言
> nlp-paper：[NLP相关Paper笔记和代码复现](https://github.com/DengBoCong/nlp-paper)
> nlp-dialogue：[一个开源的全流程对话系统，更新中！](https://github.com/DengBoCong/nlp-dialogue)
> 说明：讲解时会对相关文章资料进行思想、结构、优缺点，内容进行提炼和记录，相关引用会标明出处，引用之处如有侵权，烦请告知删除。
> 转载请注明：DengBoCong

入坑NLP这么久了，老早就想着参加比赛锻炼自己的能力，在比赛中多学习些trick。抱着学习的心态在天池上找了一圈，看到了达摩院举办的 “AI Earth”人工智能创新挑战赛——AI助力精准气象和海洋预测大赛，突然萌生想试试前不久学习的Informer模型，正好适配长序列预测模型，嘿嘿嘿，理论应用到实践正好梭哈一把，顺便提升提升自己对模型的理解。本赛题只提供了一个Pytorch权重的ResNet预训练模型，奈何我用TensorFlow2构建的代码，所以没用上（有一些方法可以转换权重，因为忙于赛事外的其他事情，我就没有去尝试）

选手群里有好多使用纯ConV、ConV+LSTM、MLP等结构的小伙伴分数都比我高，不过我还是想以学习的心态扎根Informer结构，就没有转而使用这些结构，后面如果有大佬开源出来，也会去观摩学习的。当然啦，我感觉Informer结构肯定不止我跑出来的这么点分，可能是我复现模型是构建上会有所问题，又或者是没用上巧妙的trick来把模型数据处理的更适配，如果有想法的小伙伴可以一起讨论讨论。
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210331162013335.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0RCQ18xMjE=,size_16,color_FFFFFF,t_70)

首先呈上比赛连接：[“AI Earth”人工智能创新挑战赛——AI助力精准气象和海洋预测](https://tianchi.aliyun.com/competition/entrance/531871/introduction)
本次赛题是一个时间序列预测问题，基于历史气候观测和模式模拟数据，利用T时刻过去12个月(包含T时刻)的时空序列（气象因子），预测未来1-24个月的Nino3.4指数，如下图所示：
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210331164145681.png#pic_center)
本次比赛使用的数据包括CMIP5/6模式的历史模拟数据和美国SODA模式重建的近100多年历史观测同化数据。每个样本包含以下气象及时空变量：海表温度异常(SST)，热含量异常(T300)，纬向风异常（Ua），经向风异常（Va），数据维度为（year,month,lat,lon），对于训练数据提供对应月份的Nino3.4 index标签数据，评估指标是使用相关性系数和均方根误差进行计算的，如下：
$$Score=\frac{2}{3}\cdot accskill-RMSE$$    $$acckill=\sum_{i=1}^{24}a\cdot ln(i)\cdot cor_i$$

更详细的赛题和数据描述可以参见赛事页面说明。评估指标在论坛中有小伙伴分享了实现，我就直接用了他的实现。赛题介绍完了，现在我来讲讲Informer应用的思路，首先，不了解Informer的可以了解一下Informer，我之前有写过关于Informer论文的阅读笔记，可以参考一下：
[论文阅读笔记：Informer--效果远超Transformer的长序列预测模型](https://zhuanlan.zhihu.com/p/351321328)

简单来说，赛题就是使用12个月的数据来预测未来24个月的Nino3.4指数，这一点上和原Informer论文中使用的数据有所差异，原Informer中喂入的序列长度如下：
```
seq_len=96
label_len=48
pred_len=24
```
但是本赛题的数据是12预测24，没有宽裕的数据用来学习，所以我这里做了一下处理，实验了两种序列长度：
```
方案一                  方案二
seq_len=12            12
label_len=6            12
pred_len=30           36
```
效果上来说两种方案没有很明显的差别，不过我中间尝试了使用第二种方案进行训练，保存下来的权重使用第一种方案的模型结构进行加载，发现效果上有些许提升（嘿嘿，无厘头尝试）。

喂入模型的数据这一块我也进行了特殊的建模处理，每个样本包含以下气象及时空变量：海表温度异常(SST)，热含量异常(T300)，纬向风异常（Ua），经向风异常（Va），数据维度为（year,month,lat,lon），所以在这里我将这四个维度的数据在最后一维stack起来，合成了（year,month,lat,lon,4），为了更好的喂入模型中，我对encoder和decoder前分别加了一成卷积操作，将数据平展成（batch,month,feature），同时，通过原始月份进行标记，生成month token，做Embedding之后和混入位置信息的input进行相加，如下：

```
enc_inputs = tf.keras.Input(shape=(24, 72, 4), dtype=d_type, name="{}_enc_inputs".format(name))
dec_inputs = tf.keras.Input(shape=(24, 72, 4), dtype=d_type, name="{}_dec_inputs".format(name))
enc_month_inputs = tf.keras.Input(shape=(12,), dtype=d_type, name="{}_enc_month_inputs".format(name))
dec_month_inputs = tf.keras.Input(shape=(30,), dtype=d_type, name="{}_dec_month_inputs".format(name))
```
再做Data Embedding时，为了混入位置信息，我只是单纯的使用了position encoding，没有考虑是否合适，这里估计是使用不合理，可以进行改进或者去掉：

```
def data_embedding(embedding_dim: Any, d_type: tf.dtypes.DType = tf.float32,
                   position: Any = 5000, name: AnyStr = "data_embedding") -> tf.keras.Model:
    """ Data Embedding

    :param embedding_dim: 特征维度
    :param d_type: 运行精度
    :param position: 位置总数
    :param name: 名称
    :return: Data Embedding
    """
    inputs = tf.keras.Input(shape=(None, embedding_dim), dtype=d_type, name="{}_inputs".format(name))
    month_inputs = tf.keras.Input(shape=(None,), dtype=d_type, name="{}_month_inputs".format(name))

    # token_embedding = tf.keras.layers.Conv1D(filters=embedding_dim, kernel_size=3, padding="same")(inputs)

    pos_inputs = inputs * tf.math.sqrt(x=tf.cast(x=embedding_dim, dtype=d_type), name="{}_sqrt".format(name))
    pos_encoding = positional_embedding(position=position, d_model=embedding_dim, d_type=d_type)
    pos_embeddings = pos_inputs + pos_encoding[:, :tf.shape(pos_inputs)[1], :]

    month_embedding = tf.keras.layers.Embedding(input_dim=13, output_dim=embedding_dim)(month_inputs)

    embeddings = pos_embeddings + month_embedding  # token_embedding +

    return tf.keras.Model(inputs=[inputs, month_inputs], outputs=embeddings, name=name)
```
对于CMIP和SODA两种数据，由于SODA数真实数据，而CMIP是模拟数据，需要注意的是CMIP5里面有一块是nan的噪音数据，需要进行特殊处理，大概的位置在模式5-9和模式13-14。对于数据处理策略，我也实验了两种处理策略的模型效果，一种方案是CMIP和SODA混合进行模型学习和训练，这样方式需要把那段噪音数据直接移除，否则对模型影响还是很大的。

第二种方案是将CMIP用作预训练数据处理，做预训练的话，那一段nan噪音数据直接使用np.nan_to_num直接转换就可以了，然后将SODA作为微调数据进行训练调整，两种方式的话，线下实验效果是第二种方式更好，但是奈何参加比赛中间有一段时间忙其他事情去了，使用第二种方案的时候已经是B榜了，每天两次提交机会，然后不小心提交失败个几次，你懂得。

特殊处理就是以上这些需要说明的，剩下的就是Informer复现了，我是使用TensorFlow2进行模型构建，代码仓库如下：


# 总结
第一次参加天池比赛，学习到了很多，在选手群里和论坛中也收到了很多启发，不过奈何自己时间不充裕以及经验不足，所以成绩平平，不过相信以后会越做越好，下面我提一下模型的几点改进思路：
+ Data Embedding中如何更好的处理位置编码
+ 先对空间attn，在对时间attn
+ 特征抽取的卷积操作
+ 优化器学习率调低一点（教训），下降太快了
+ 更换MSE损失函数
+ informer中Self-Attention和ProbSparse self-attention实验
