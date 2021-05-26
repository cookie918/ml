
# -*- coding: utf-8 -*-
# description   : 后向最大分词
#
# 后向最大匹配的后向意思是说，从后往前匹配。最大意思同样是说，我们匹配的词的长度越大越好。
# 这里，我们同样假设这个最大长度max_len = 5:
#
# 第一轮搜索：
# ①"南京市长江大桥 " 词典中没有市长江大桥这个词，匹配失败
# ②"南京市长江大桥 " 词典中没有长江大桥这个词，匹配失败
# ③"南京市长江大桥 " 词典中没有江大桥这个词，匹配失败
# ④"南京市长江大桥 " 词典中有大桥这个词，匹配成功，去除
# 句子变为：“南京市长江”
#
# 第二轮搜索：
# ①"南京市长江 " 词典中没有南京市长江这个词，匹配失败
# ②"南京市长江 " 词典中没有京市长江这个词，匹配失败
# ③"南京市长江 " 词典中没有市长江这个词，匹配失败
# ④"南京市长江 " 词典中有长江这个词，匹配成功，去除
# 句子变为：“南京市”
#
# 第三轮搜索（句子长度已不足5，将max_len改为3）：
# ①"南京市 " 词典中没有南京市这个词，匹配失败
# ②"南京市 " 词典中没有京市这个词，匹配失败
# ③"南京市 " 词典中有市这个词，匹配成功，去除
# 句子变为：“南京”
#
# 第四轮搜索：
# ①"南京 " 词典中有南京这个词，匹配成功，去除
# 句子变为：""，说明已经处理完毕
#
# 最终结果：“南京 / 市 / 长江 / 大桥”
#
# 相同的话，相同的词典，分出来的效果却不一样，导致了歧义问题。统计结果表明，单纯使用后向最大匹配算法的错误率略低于正向最大匹配算法。

dictionaries = ["南京", "市长", "大桥", "长江", "江", "市"]




# 后向最大匹配
def backward_max_matching(text, max_len=5):
    result = []
    text_ = text
    index = max_len

    while len(text_) > 0:

        if index == 0:
            print("分词失败，词典中没有这个词")
            return []
        # print(text_[-index:])
        if text_[-index:] in dictionaries:
            result.insert(0, text_[-index:])
            # result.append(text_[-index:])
            text_ = text_[:-index]
            index = 5
        else:
            index = index - 1

    return "".join(word + "/" for word in result)

if __name__ == '__main__':
    content = "南京市长江大桥"

    backward_result = backward_max_matching(content)
    print("backward_result:", backward_result)
