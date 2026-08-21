#!/usr/bin/env python
import numpy as np

class Model:
    @staticmethod
    def loadmodel(filename):
        """
        读取.SGEMS文件
        '''
        网格尺寸 x*y*z
        1 （不知道是啥）
        v （不知道是啥）
        x
        y
        z
        ......
        '''
        :param filename: 文件名
        :return: 模型数据(为三维numpy数组)
        """
        with open(filename) as f:
            shape = f.readline().rstrip('\n').rstrip(' ')   # 从文件中读取一行内容，去除末尾的换行符和空格， 长宽高
            t = f.readline().rstrip('\n').rstrip(' ')       # 读第二行
            for _ in range(int(t)):
                _ = f.readline()
            x, y, z = shape.split(' ')  # 将shape按空格分隔
            x, y, z = int(x), int(y), int(z)
            model = np.zeros([x, y, z], dtype=np.float32)   # 创建三维数组
            for k in range(z):
                for j in range(y):
                    for i in range(x):
                        model[i, j, k] = float(f.readline().rstrip('\n').strip(' '))  # 读取文件内容并赋给model
        return model
    @staticmethod
    def loadmodel_2d(filename):
        """
        读取.SGEMS文件
        '''
        网格尺寸 x*y*z
        1 （不知道是啥）
        v （不知道是啥）
        x
        y
        z
        ......
        '''
        :param filename: 文件名
        :return: 模型数据(为二维numpy数组)
        """
        with open(filename) as f:
            shape = f.readline().rstrip('\n').rstrip(' ')   # 从文件中读取一行内容，去除末尾的换行符和空格， 长宽高
            t = f.readline().rstrip('\n').rstrip(' ')       # 读第二行
            for _ in range(int(t)):
                _ = f.readline()
            x, y = shape.split(' ')  # 将shape按空格分隔
            x, y = int(x), int(y)
            model = np.zeros([x, y], dtype=np.float32)   # 创建三维数组
            for j in range(y):
                for i in range(x):
                    model[i, j] = float(f.readline().rstrip('\n').strip(' '))  # 读取文件内容并赋给model
        return model

    @staticmethod
    def clipmodel(model, n_x, n_y, n_z, length):
        """
        模型剪切函数
        :param model: 要剪切的模型
        :param n_x: 在x方向上要剪切的块数
        :param n_y: 在y方向上要剪切的块数
        :param n_z: 在z方向上要剪切的块数
        :param length:每个块的边长
        :return:
        """
        num = 0

        def savemodel(data, filename):
            """
            保存模型数据到文件
            :param data: 要保存的模型数据
            :param filename: 保存的文件名
            :return:
            """
            x, y, z = data.shape  # 获取原始模型形状
            # data = np.array(data, np.int)
            with open('temp_data_32_test/' + filename, 'w') as f:
                f.write('{} {} {}\n'.format(x, y, z))
                f.write('1\n')
                f.write('v\n')
                for k in range(z):
                    for j in range(y):
                        for i in range(x):
                            f.write('{:.6f} \n'.format(data[i, j, k]))

        x, y, z = model.shape  # 获取原始模型形状
        span_x = int((x - length) / n_x)  # 计算x方向上每个块的跨度
        span_y = int((y - length) / n_y)  # 计算y方向上每个块的跨度
        span_z = int((z - length) / n_z)  # 计算z方向上每个块的跨度
        # span_z = 1  #
        for k in range(n_z):
            for i in range(n_y):
                for j in range(n_x):
                    sx, sy, sz = i * span_x, j * span_y, k * span_z  # 计算剪切起始点的坐标
                    ex, ey, ez = sx + length, sy + length, sz + length  # 计算剪切结束点的坐标
                    clip = model[sx:ex, sy:ey, sz:ez]  # 切片
                    filename = str(num) + '.sgems'
                    savemodel(clip, filename)  # 保存剪切后的数据到文件
                    num += 1
                    print('{}.sgems have been saved!!!'.format(num - 1))

    @staticmethod
    # data 参数是什么
    def savemodel(data, filename):
        x, y, z = data.shape
        with open(filename, 'w') as f:
            f.write('{} {} {}\n'.format(x, y, z))
            f.write('1\n')
            f.write('v\n')
            for k in range(z):
                for j in range(y):
                    for i in range(x):
                        f.write('{:.6f}\n'.format(data[i, j, k]))

    # 转换成point set格式数据，将模型中的结构单独提取出来
    @staticmethod
    def savemodel_data_with_position(data, filename):
        x, y, z = data.shape
        with open(filename, 'w') as f:
            f.write('{} {} {}\n'.format(x, y, z))
            f.write('4\nx\ny\nz\nv\n')
            for k in range(z):
                for j in range(y):
                    for i in range(x):
                        if np.sum(data[i, j]) != 0.:
                            f.write('{}\t{}\t{}\t{:.2f}\n'.format(i, j, k, data[i, j, k]))

    def savemodel_data_with_position_2D(data, filename):
        x, y = data.shape
        with open(filename, 'w') as f:
            f.write('{} {}\n'.format(x, y))
            f.write('1\nv\n')
            for j in range(y):
                for i in range(x):
                    f.write('{:.2f}\n'.format(data[i, j]))

    @staticmethod
    def get_condition_data(data: np.ndarray, num_x: int, num_y: int, width: int) -> np.ndarray:
        """
        从给定的单个模型数据中提取条件数据
        :param data: 单个模型的数据
        :param num_x: 在x方向上的条件数据的数量
        :param num_y: 在y方向上的条件数据的数量
        :param width: 每个条件数据的宽度
        :return: 条件数据
        """
        x, y, z = data.shape
        data_c = np.zeros([x, y, z], dtype=np.float64)
        data_c = np.zeros((x, y, z)) * np.nan
        x_index = [int((i + 1) / num_x * x) for i in range(num_x)]
        y_index = [int((i + 1) / num_y * y) for i in range(num_y)]
        for i_x in x_index:
            for i_y in y_index:
                data_c[i_x:i_x + width, i_y:i_y + width, :] = data[i_x:i_x + width, i_y:i_y + width, :]
        return data_c

    @staticmethod
    def processdata(data: np.ndarray) -> np.ndarray:
        """
        对输入数组中的每个元素进行阈值处理，根据不同的阈值范围将其转换为不同的整数值
        :param data:三维Numpy数组
        :return:经处理后相同形状的数组
        """
        x, y, z = data.shape
        for k in range(z):
            for j in range(y):
                for i in range(x):
                    if data[i, j, k] < 0.5:
                        data[i, j, k] = 0
                    elif 0.8 <= data[i, j, k] < 1.2:
                        data[i, j, k] = 1
                    elif 1.8 <= data[i, j, k] < 2.3:
                        data[i, j, k] = 2
                    elif data[i, j, k] > 2.8:
                        data[i, j, k] = 3
        return data


def random_sample(data, num_x, num_y, index):
    # x_ = [v for v in range(40)] y_ = [v for v in range(40)] x_list = random.sample(x_, num_x) y_list =
    # random.sample(y_, num_y) x_list = np.array([2, 10, 15, 20, 25, 30, 35, 38, 2, 10, 15, 20, 25, 30, 35, 38, 30,
    # 15, 30, 8, 12, 15, 20, 24, 30, 36, 8, 12, 15, 18, 5, 3]) y_list = np.array([2, 10, 15, 20, 25, 30, 35, 38, 10,
    # 20, 10, 7, 30, 15, 20, 15, 38, 19, 5, 2, 5, 7, 2, 10, 15, 9, 18, 24, 30, 35, 25, 30]) x_list = np.array([4, 10,
    # 15, 25, 30, 35, 38]) y_list = np.array([7, 13, 15, 22, 28, 34, 36])
    lis = [4, 8, 12, 16, 22, 28, 32, 36]
    lis = [4, 10, 17, 24, 30, 36]
    lis = [4, 9, 14, 19, 24, 30, 35]
    # sample = np.zeros(shape=(40, 40, 1))
    # # x_list = np.array([5, 27, 35])
    # for x in x_list:
    #     for y in y_list:
    #         sample[x, y, :] = data[x, y, :]
    #     # sample[x:x+1] = data[x:x+1]
    # data = np.reshape(data, [-1, 1])
    sample = np.zeros_like(data)
    # for (x, y) in zip(x_list, y_list):
    for x in lis:
        for y in lis:
            sample[x, y] = data[x, y]
    sample = np.reshape(sample, [40, 40, 1])
    # expand_sample = np.zeros_like(sample)              # 把条件点改成4x4的
    # data = np.reshape(data, [40, 40])
    # for x in range(39):
    #     for y in range(39):
    #         if sample[x, y] != 0:
    #             expand_sample[x:x+2, y:y+2] = data[x:x+2, y:y+2]
    # expand_sample = np.reshape(expand_sample, [40, 40, 1])
    return sample


if __name__ == '__main__':
    # dataset_size = 1435
    # modelfile = '2drecon'
    model_dir = '2drecon/'
    # for i in range(dataset_size):
    #     model = Model.loadmodel("{}{}.sgems".format(model_dir, i))
    #     model = Model.get_condition_data(model, 5, 5, 2)
    #     Model.savemodel(model, "{}condition data/{}.sgems".format(model_dir, i))
    # modelfile = '9_24.SGEMS'
    # model = Model.loadmodel(modelfile)
    # Model.clipmodel(model, 3, 3, 116, 40)
    # Model.savemodel_data_with_position(model, '24-9.sgems')
    # model_dir = '2drecon/'
    # for t in range(10):
    #     for b in range(25):
    #         model = Model.loadmodel('{}{}_{}.sgems'.format(model_dir, t, b))
    #         model = Model.processdata(model)
    #         Model.savemodel(model, '2drecon_p/{}_{}.sgems'.format(t, b))
    # model_file = '4features.sgems'
    # model = Model.loadmodel(model_file)
    # Model.clipmodel(model, 3, 3, 116, 40)
    # modeldir = 'multi_feature_dataset2d_extend'
    # index = [v for v in range(1600)]
    # index = random.sample(index, 50)
    # for i in range(0, 2932):
    #     model = Model.loadmodel('{}/{}.sgems'.format(modeldir, i))
    #     # model = model + 4
    #     # model[model == 4] = 0
    #     # model[model == 5] = 3
    #     # model[model == 6] = 1
    #     # model[model == 7] = 2
    #     # Model.savemodel(model, '{}/{}.sgems'.format(modeldir, i))
    #     Model.savemodel(random_sample(model, 8, 8, index), '{}/condition/{}.sgems'.format(modeldir, i))
    model_dir = 'expand dataset/condition data'
    for i in range(1435):
        model = Model.loadmodel(f'{model_dir}/{i}.sgems')
        Model.savemodel_data_with_position(model, f'{model_dir}/withxyz/{i}.sgems')
