dataset\filter\commonsense\train_filter.csv是从commonsense中提取出来的长的数据，按照is_short=false提取的，is_short是原始数据的一列，提取的是train部分


dataset\filter\commonsense\train_filter_50_third_person.csv是将dataset\filter\commonsense\train_filter.csv前五十条input部分转换成了第三人称描述

agent\first_perspective_agent.py用来测第一人称视角下

agent\third_perspective_agent.py测第三人称视角下，prompt基本没变，只改成了“评价故事中的主人公”

agent\perspective_rewrite_agent.py用于转换视角描述

result.py用于将分开测的两个视角结果合并到一个文件中去

平均绝对偏移mean_abs_shift:仅仅改变叙述视角，模型对“主人公是否做错了”的评分平均会发生多大变化

方向性偏移mean_signed_shift:
    正值：第三人称更“容易被判为做错”
    负值：第三人称更“宽容”

决策翻转率flip_rate:有多少比例的样本，在视角变化后，模型的“判断结果”发生了翻转