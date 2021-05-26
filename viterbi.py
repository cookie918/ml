# -*- coding: utf-8 -*-
# description   : 维特比算法(viterbi)

import math
import collections

# 维特比算法(viterbi)
def word_segmentation(text):
    ##################################################################################
    word_dictionaries = ["经常", "有", "意见", "意", "见", "有意见", "分歧", "分", "歧"]
    probability = {"经常": 0.08, "有": 0.04, "意见": 0.08, "意": 0.01, "见": 0.005, "有意见": 0.002, "分歧": 0.04, "分": 0.02,
                   "歧": 0.005}
    probability_ln = {key: -math.log(probability[key]) for key in probability}

    # 构造图的代码并没有实现，以下只是手工建立的图，为了说明 维特比算法
    ##################################################################################

    # 有向五环图，存储的格式：key是结点名，value是一个结点的所有上一个结点（以及边上的权重）
    graph = {
        0: {0: (0, "")},
        1: {0: (20, "经")},
        2: {0: (2.52, "经常"), 1: (20, "常")},
        3: {2: (3.21, "有")},
        4: {3: (20, "意")},
        5: {2: (6.21, "有意见"), 3: (2.52, "意见"), 4: (20, "见")},
        6: {5: (3.9, "分")},
        7: {5: (3.21, "分歧"), 6: (5.29, "歧")}
    }
    # 保存结点n的f(n)以及实现f(n)的上一个结点
    f = collections.OrderedDict()

    for key, value in graph.items():
        # 如果该节点的上一个结点还没有计算f(n)，则直接比较大小；如果计算了f(n)，则需要加上f(n)
        # 从该节点的所有上一个结点中选择f(n)值最小的
        min_temp = min((pre_node_value[0], pre_node_key) if pre_node_key not in f else (
            pre_node_value[0] + f[pre_node_key][0], pre_node_key) for pre_node_key, pre_node_value in value.items())
        f[key] = min_temp

    print(f)
    # 结果：OrderedDict([
    # (0, (0, 0)), (1, (20, 0)), (2, (2.52, 0)), (3, (5.73, 2)),
    # (4, (25.73, 3)), (5, (8.25, 3)), (6, (12.15, 5)), (7, (11.46, 5))
    # ])

    # 最后一个结点7
    last = next(reversed(f))
    # 第一个结点0
    first = next(iter(f))
    # 保存路径，最后一个结点先添入
    result = [last, ]
    # 最后一个结点的所有前一个结点
    pre_last = f[last]

    # 没到达第一个结点就一直循环
    while pre_last[1] is not first:
        # 加入一个路径结点X
        result.append(pre_last[1])
        # 定位到路径结点X的上一个结点
        pre_last = f[pre_last[1]]
    # 第一个结点添入
    result.append(first)

    print(result)
    # 结果：[7, 5, 3, 2, 0]

    text_result = []
    # 找到路径上边的词
    for i, num in enumerate(result):
        if i + 1 == len(result):
            break
        # print(i, num, result[i + 1])
        word = graph[num][result[i + 1]][1]
        # print(word)
        text_result.append(word)
    # 翻转一下
    text_result.reverse()

    return "".join(word + "/" for word in text_result)


if __name__ == '__main__':

    content = "经常有意见分歧"
    word_segmentation_result = word_segmentation(content)
    print("word_segmentation_result:", word_segmentation_result)
