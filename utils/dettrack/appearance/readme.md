基于 Deep SORT 目标跟踪算法的相关实现
======================================
该程序模型主要的作用包括 3 部分：\
（1）提供一个将 yolo 格式转化为 mot 格式的功能 \
（2）提供一个将标柱的图内的标柱区域抠出来，另外保存 \
（3）训练一个用于目标跟踪算法的外观特征提取模型

# 程序文件说明
>./utils
>>（1）将 yolo 格式的标签文件转为 mot 格式文件
>>>yolo_to_mot_format.py

>>（2）将二维码从含多个二维码的图像中抠出来
>>>buckle.py

>>（3）生成 deep sort 算法所需要的数据格式
>>>yolo_to_deepsort_format.py

>./models
>>（1）自编码器特征压缩模型
>>>autoencoder.py

>>（2）Dataset
>>>QRCodeDataset.py

>根目录
>>（1）特征压缩模型的训练
>>>train_appreance_feature.py
>>> #### 给定一个包含训练图像数据的文件夹，训练特征压缩模型

>>（2）模型测试分析
>>>test_appreance_feature.py
>>> #### 给定一个包含测试图像数据的文件夹，获得经过外观特征模型重构的图像，并保存
---------------------------------------

# 程序使用流程
## （1）生成外观特征压缩模型需要的训练数据集
>utils/buckle.py
## （2）训练外观特征提取模型
>train_appreance_feature.py
## （3）对外观特征压缩模型进行测试
>test_appreance_feature.py
## （4）生成线下 Deep SORT 厕所所需要的数据格式
>deepsort_format.py
