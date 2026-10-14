## 目标
处理超长session

## 方案

把session之间的传递做成一个单链表

父session需要进行处理时，会新开一个新的session，并有一个previous的指针映射到父session；

当新session需要知道父session的内容时，可以通过previous指针找到父session（允许递归）。

父session需要做一下处理，让新session可以很快定位到父session中的对应部分；

这个处理就是语义切割，就像语音处理的语义切割一样，允许区间有overlap，但每个区间需要做到较好的切分（预期通过非LLM方案），然后给每个区间上topic，有些可以没有topic，然后在previous指针上，可以得到整个父session所有区间的topic枚举值，子session可以通过枚举值快速找到对应区间原文；

对于使用过程中的tool，可以在区间的class中定义一个used_tool的attribution，记录这个区间用了什么tool（只需要tool_name），并用于区间的进一步定位。

最终返回的是一个两个api，一个是process的POST api，一个是GET的api。

本质是给session做index，让agent在避免压缩损失信息情况下，用这个方法快速找到相关的历史内容，从而做到“有效扩展上下文”的效果。