"""固定教学字帖：University of Utah Reading Clinic, Letter Formation Suggestions.
https://uurc.utah.edu/general/letterformation.php
不同教材可能采用不同笔顺；不自动推断任意字符。
O 按用户要求简化为四顶点方框，属于本项目的几何改写。
"""
import numpy as np

PROFILE = 'uurc-print-v1'


def standard_strokes(text):
    if text == 'H':
        return [np.array(s, dtype=float) for s in [
            [[.15,.15],[.15,.85]],  # 左竖向下
            [[.15,.5],[.85,.5]],    # 中横向右
            [[.85,.15],[.85,.85]],  # 右竖向下
        ]]
    if text == 'NAO':
        # 四个不同顶点；末尾重复起点只为闭合，仍是一笔。
        box = [[.70,.15],[.70,.85],[.93,.85],[.93,.15],[.70,.15]]
        return [np.array(s, dtype=float) for s in [
            [[.05,.15],[.05,.85]],
            [[.05,.15],[.27,.85],[.27,.15]],  # 斜下后连写右竖向上
            [[.47,.15],[.36,.85]],
            [[.47,.15],[.58,.85]],
            [[.40714285714285714,.55],[.5328571428571428,.55]],
            box,
        ]]
    raise ValueError(f'尚无经过核对的规范模板: {text!r}；当前支持 H、NAO')
