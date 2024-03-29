# WhenAnime 以图搜番

## 简介

以图搜番，动画场景搜索引擎，基于对比学习训练动漫特征提取模型，构建自己的动漫索引库，追溯动漫截图的出现时刻，并可提供预览动图。

本项目使用 Python 开发，基于 PyTorch，前端使用 Gradio 构建，你可以将此项目部署于在线机器学习托管平台，而不需要使用任何自己的服务器资源，如魔搭 ModelScope，Huggingface 等平台。你可以在[此链接](https://modelscope.cn/studios/tuxiaobei/Search_Anime_For_XYY_S19_to_S28)尝试本项目，样例项目索引了喜羊羊与灰太狼第 19~28 季，共 600 集，每集 15 分钟。

你可以训练专属于目标动漫的检索模型，达到最出色的效果，检索模型对裁剪和颜色变换均具有优秀鲁棒性。

![image-20240214160019780](assets/image-20240214160019780.png)

## 开始

### 环境配置

需要安装 `Python3.8` 或更高版本，建议配置好 `CUDA` 、`cuDNN` 深度学习环境，参照[官网安装 PyTorch](https://pytorch.org/get-started/locally/)。

```
git clone https://github.com/tuxiaobei-scu/WhenAnime.git
cd WhenAnime
pip install -r requirements.txt
```

### 视频准备

将视频存储在 `videos` 文件夹下。

`videos` 文件夹下有若干子目录，每个子目录表示一季，每个子目录的命名格式为 `<季号>-<季名称>`，如 `25-奇妙大营救`，`-` 号可替换为 `_` 或者空格。

每个子目录下即存储对应季里的所有集，命名格式为 `<集号>-<集名称>`，如 `55-暗黑的他.mp4`，`-` 号可替换为 `_` 或者空格，后缀名可为 `mp4`、`avi` 或 `mkv`。

视频文件结构样例：

```
└─videos
    ├─24_决战次时代
    │      01-怪石现世.mp4
    │      02-消失的时光.mp4
    │
    └─25_奇妙大营救
           01 奇猫国危机.mp4
           02 落入妙狗国.mp4
```

### 数据集准备

#### 数据集帧提取

此脚本用于从视频中提取帧，并将这些帧保存到训练集和验证集文件夹中。

```
python pre_dataset.py
```

**命令行参数**

- `-video_path`: 视频文件所在的目录路径。默认为 'videos'。
- `-op`: 忽略视频头部长度的百分比。默认为 0。
- `-ed`: 忽略视频尾部长度的百分比。默认为 0。
- `-num`: 从每个视频中提取的帧数。默认为 10。
- `-dataset`: 输出文件夹的路径，用于存储提取的帧。默认为 'dataset'。
- `-split`: 验证集所占的比例。默认为 0.2。
- `-quality`: 保存的 JPEG 图片的质量，范围为 0 到 100。默认为 90。
- `-color_threshold`: 颜色标准差的阈值，用于判断纯色帧。默认为 1。
- `-target_resolution`: 目标分辨率的长边长度，可选参数。默认为 None。

#### 获取数据集均值与方差

此脚本用于计算数据集的均值和标准差，并将结果保存为 JSON 文件。

```
python get_mean_std.py
```

**命令行参数**

- `-dataset`: 数据集所在的目录路径。默认为 'dataset'。
- `-batch_size`: 每批处理的数据量。默认为 64。
- `-output`: 输出 JSON 文件的路径。默认为 'info.json'。
- `-max_samples`: 用于计算的最大样本数量。默认为 1000。

#### 生成增强的验证集图片
此脚本用于对验证集图片进行数据增强，并将增强后的图片保存到同一目录下。
```
python gen_val.py
```
**命令行参数**

- `-val_path`: 验证集图片所在的目录路径。默认为 'dataset/val'。
- `-img_size`: 随机裁剪的大小。默认为 224。
- `-crop_scale`: 随机裁剪的最小比例。默认为 0.25。
- `-flip`: 随机水平翻转的概率。默认为 0.2。
- `-color`: 随机颜色亮度调整的最大值。默认为 0.2。

### 训练

此脚本用于训练和评估模型，并支持预训练模型的加载。

```
python train.py
```

**命令行参数**

- `-dataset`: 数据集所在的目录路径。默认为 'dataset'。

- `-info_path`: 包含数据集均值和标准差的 JSON 文件路径。默认为 'info.json'。

- `-train_log`: 训练日志文件的路径。默认为 'train_log.csv'。

- `-pre_train`: 预训练权重文件的路径。默认为 None。

- `-model_path`: 保存模型的目录路径。默认为 'models'。

- `-train_batch_size`: 训练时的批量大小。默认为 256。

- `-val_batch_size`: 验证时的批量大小。默认为 256。

- `-lr`: 初始学习率。默认为 0.001。

- `-epochs`: 训练的轮数。默认为 50。

- `-t_max`: 学习率周期长度。默认为 10。

- `-eta_min`: 学习率的最小值。默认为 0.00005。

训练过程中，模型会保存在 `model_path` 指定的目录下，其中最佳模型为 `best.pth`，训练日志会保存在 `train_log` 指定的文件中。如果提供了 `pre_train` 参数，脚本会从该路径加载预练权重。训练日志文件会记录每个 epoch 的损失、准确率以及 MRR。

训练完成后，你需要选择一个最优模型（可参考训练日志，建议 `best.pth`），将其移动到项目根目录，并重命名为 `model.pth`。为节省空间，训练完成确定最终模型后，你可以删除其他模型权重文件。

**模型**

默认采用 `Resnet18` 模型，若需要更改主干模型可更改 `model.py`。

**损失函数**

使用 `MultipleNegativesRankingLoss ` 多负例排名损失，在图片检索任务中，模型的目标是学习如何将查询图片与相关的图片（正样本）区分出来，同时将查询图片与不相关的图片（负样本）区分开来。这种损失函数适用于训练用于图片检索任务的模型，特别是在训练模型以进行高效检索时表现出色。

MultipleNegativesRankingLoss 是一种损失函数，它将每个查询图片与其对应的正样本图片进行比较，并将所有其他图片作为负样本。在每次前向传播时，该损失函数计算查询图片与所有其他图片之间的相似度得分，并最小化这些得分的对数似然。这意味着，模型需要学习如何将与之相关的正样本图片排在其他不相关的负样本图片之前。

在训练过程中，随着批处理大小的增加，性能通常会提高。这是因为更大的批处理大小意味着模型可以同时看到更多的正负样本对，从而更好地学习区分它们。

可参考：[Efficient Natural Language Response Suggestion for Smart Reply, Section 4.4](https://arxiv.org/pdf/1705.00652.pdf)

**评估指标**

将验证集中所有原始图片和增强后图片分别通过模型进行特征提取，对每个增强图片进行最近邻检索，找到相似度最高的 20 个候选结果进行指标计算。

- 准确率（Accuracy）：
  - Top-1 准确率：指的是模型预测的第一选项是正确答案的概率。
  - Top-5 准确率：指的是模型预测的前五个选项中包含正确答案的概率。
  - Top-10 准确率：指的是模型预测的前十个选项中包含正确答案的概率。
  - Top-20 准确率：指的是模型预测的前二十个选项中包含正确答案的概率。
  - 这些指标通常用于评估模型在给定查询时返回最相关结果的性能。
- Mean Reciprocal Rank, MRR：
  - MRR 是一个衡量模型排序质量的指标，即第一个结果匹配，分数为 1，第二个匹配分数为 0.5，第 n 个匹配分数为 1/n，如果前 20 个结果没有匹配的图片，分数为 0。最终的分数为所有结果匹配得分的平均数。MRR 越高，表示模型在排序时越能将正确答案排在前面的能力越强。

在喜羊羊与灰太狼第 19~28 季验证集上，共 3483 个验证图片，采用默认数据增强策略（随机裁剪大小 0.25，随机翻转概率 0.2，随机颜色调整 0.2）

| 模型               | TOP-1  | TOP-5  | TOP-10 | TOP-20 | MRR    |
| ------------------ | ------ | ------ | ------ | ------ | ------ |
| 训练前（默认权重） | 0.6365 | 0.7723 | 0.8191 | 0.8573 | 0.6970 |
| 训练后             | 0.9325 | 0.9888 | 0.9945 | 0.9974 | 0.9572 |

### 特征提取与索引构建

#### 生成视频剧集 JSON 文件列表

此脚本用于生成视频剧集的 JSON 文件列表，包括每个视频的详细信息，如 ID、名称、帧率、持续时间等，支持增量处理（仅可向最后添加视频内容，不可在中间插入新的视频）。
```
python gen_conf.py
```
**命令行参数**

- `-video_path`: 视频文件所在的目录路径。默认为 'videos'。
- `-input`: 输入配置文件路径。默认为 None（如需要增量处理，则需要设置）。
- `-output_diff`: 输出差异文件路径。默认为 'conf_diff.json'。
- `-output`: 输出文件路径。默认为 'conf.json'。
- `-fps`: 图像提取的帧率。默认为 8（若设置了 input，则此项无效，将沿用之前配置文件的 fps）。
- `-op`: 片头长度（秒）。默认为 0。
- `-ed`: 片尾长度（秒）。默认为 0。
- `-full_first_ep`: 是否包含首集的片头/片尾。默认为 True。
- `-sim_threshold`: 图像提取的相似度阈值。默认为 4。

#### 提取视频帧特征
此脚本使用预训练模型从视频帧中提取特征，并将这些特征保存到 `.npz` 文件中。
```
python frame_feature.py
```
**命令行参数**
- `-model_path`: 预训练模型文件的路径。默认为 'model.pth'。
- `-conf_path`: 配置文件的路径。默认为 'conf_diff.json'。
- `-info_path`: 包含数据集均值和标准差的 JSON 文件路径。默认为 'info.json'。
- `-output`: 提取的特征的输出路径。默认为 'features'。
- `-threads`: 并行处理的工作线程数。默认为 4。

#### 构建特征索引
此脚本用于构建特征索引，以便高效地搜索和检索视频帧特征。它使用 HNSW (Hierarchical Navigable Small World) 算法，这是一种用于高维空间的数据库搜索算法，特别适用于大规模数据集。
```
python build_index.py
```
**命令行参数**
- `-features_path`: 特征文件的目录路径。默认为 'features'。
- `-conf_diff`: 包含新特征配置信息的文件路径。默认为 'conf_diff.json'。
- `-pre_index`: 之前构建的特征索引文件的路径。默认为 None（若设置则进行增量构建，此时 ef_construction、m 参数无效，dim 参数必须与之前索引一致）。
- `-dim`: 特征的维度。默认为 512。
- `-ef_construction`: 平衡索引/构建时间和索引精度的参数。默认为 512。
- `-m`: 图中的最大出边数。默认为 64（取值 0~100，越大精确度越高，同时耗时和占用内存越大）。
- `-output`: 输出特征索引文件的路径。默认为 'index.bin'。

### 前端应用

使用 Gradio 库创建的机器学习 Web 应用程序，允许用户上传图片，并检索与该图片相似的视频帧，并提供预览视频动图。

```python
python app.py
```

运行后访问 `http://127.0.0.1:7860/` 即可访问查询推理页面。

**环境变量**

- `video_path_root`: 该变量用于指定视频文件根目录的路径。这个路径应该包含所有的视频文件。如果该环境变量未设置，脚本将使用空字符串。
- `ef`: 查询时用来平衡查询时间和查询精度的参数。如果该环境变量未设置，脚本将使用默认值 512。
- `index_path`: 构建的特征索引文件的路径。如果该环境变量未设置，脚本将使用默认值 'index.bin'。
- `conf_path`: 指定包含配置信息的 JSON 文件的路径。如果该环境变量未设置，脚本将使用默认值 'conf.json'。
- `info_path`: 包含数据集均值和标准差的 JSON 文件的路径。如果该环境变量未设置，脚本将使用默认值 'info.json'。
- `model_path`: 指定特征提取模型的路径，该模型将被用于从查询图片帧中提取特征。如果该环境变量未设置，脚本将使用默认值 'model.pth'。

### 部署

若你需要将此项目托管到云端平台，如魔搭 ModelScope，Huggingface 等，你需要将以下文件上传到运行空间：

- app.py
- conf.json
- index.bin
- model.pth
- model.py
- requirements.txt

若原始视频文件过多过大，不建议将 videos 目录上传，可将其上传到网络，如对象存储，Alist 等，然后通过链接访问视频文件，`video_path_root` 环境变量设置为域名路径即可。如若域名为 `https://github.com/videos/20_跨时空救兵/58_送爸爸“回家”.mp4`，则环境变量设置为 `https://github.com/videos` 即可。

上传视频前建议先压制视频，可降低分辨率和码率。如不需要视频预览功能则可以不上传视频。

注意：Huggingface 仅允许通过标准 HTTP 和 HTTPS 端口（80 和 443）以及端口 8080 发出请求，任何发往其他端口的请求都将被阻止。

## To-do

- [ ] 实现对 OP/ED 的智能识别
- [ ] 训练通用动漫检索模型
- [ ] GPU 加速视频解码，提高特征提取效率
- [ ] 支持多部动画的检索
- [ ] 支持更多剧集视频文件结构

## 鸣谢

- 部分思路来自于 https://github.com/NitroRCr/AnimeBack
- 使用 https://github.com/nmslib/hnswlib 进行特征索引

## 许可

本项目采用 [AGPL-3.0](./LICENSE) 许可协议开源，在使用本项目的源代码时请遵守许可协议。