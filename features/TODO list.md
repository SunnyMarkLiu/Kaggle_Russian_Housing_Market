待添加的特征工程
- 添加年、季度、月、星期 的平均房价（可考虑按照地区进行 groupby）(会造成信息泄漏)
- 添加年、季度、月、星期 的销量（可考虑按照地区进行 groupby）(会造成信息泄漏)
由于测试集中时间跨度较大，2015.7 至 2016.5，时间窗内统计平均房价，可能会造成训练集效果很好，但测试集效果很差，因为测试集
的这些属性为0不能进行统计，一次可以考虑统计时间窗内的平均销量属性！

- 上一个月、季度（时间窗内的）的销量
- 存在 floor > max_floor 的异常数据
- 地区的人口密度

根据相关性分析需要处理的特征：
https://www.kaggle.com/captcalculator/a-very-extensive-sberbank-exploratory-analysis
- Internal Home Features
- Demographic Features
- Education Features
- Cultural/Recreational Features
- Infrastructure Features

注意考察 train、test 的特征分布情况！