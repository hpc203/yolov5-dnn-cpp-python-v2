# yolov5-dnn-cpp-python-v2
这套程序是对上一版本的后处理模块优化，把三个尺度的输出特征图作reshape和permute维度置换后，
输出特征图的形状分别是(3x80x80, 85), (3x40x40, 85), (3x20x20, 85)，然后在行方向拼接成一个特征图。
列方向的长度保持为(num_classes+5)，这样在求最大分类置信度时，可以使用opencv内置函数minMaxLoc，
这相比于上一个版本，减少了一个for循环

由于对特征图添加了维度变换的操作，那么生成onnx文件也发生了改变，新的onnx在百度云盘下载
链接: https://pan.baidu.com/s/11uF1QeYyu3otrGbMGhw0ZQ  密码: es2w
