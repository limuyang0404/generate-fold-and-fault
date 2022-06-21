# coding=UTF-8
import numpy as np
import math
import pyvista as pv
# from biharmonic_spline_interpolation import BiharmonicSplineInterpolation
import matplotlib.pyplot as plt
import scipy
from random import shuffle
import matplotlib.image as image
def WaveletConvolve(wavelet, rc):
    """
    将给定子波与反射系数褶积
    :param wavelet: 地震子波，一维数组
    :param rc:反射系数，多维数组
    :return:褶积结果，多维数组
    """
    convolve_output = np.zeros(np.shape(rc))#子波卷积结果
    if len(rc.shape) == 2:#通常送入的反射系数为二维数组，即全道数×单道采样点数
        for j in range(0, rc.shape[0]):
            convolve_output[j, :] = np.convolve(rc[j, :], wavelet, mode='same')
            pass
        pass
    elif len(rc.shape) == 3:
        for i in range(rc.shape[0]):
            for j in range(rc.shape[1]):
                convolve_output[i, j, :] = np.convolve(rc[i, j, :], wavelet, mode='same')
    return convolve_output
def fault_displacement_field(x, y, lx, ly, d_max):
    """
    生成断层的位移场，用于计算后续的位移量，里面的xy坐标均为断层基准面上的坐标系坐标
    :param x: 数据体各点的x坐标，一维数组
    :param y:数据体各点的y坐标，一维数组
    :param lx:椭圆位移场在x方向的半轴长度，浮点型
    :param ly:椭圆位移场在y方向的半周长度，浮点型
    :param d_max:最大位移量，浮点型
    :return:数据体各点位移场数值
    """
    rxy = ((x / lx) ** 2 + (y / ly) ** 2) ** 0.5
    rxy[rxy > 1] = 1
    displacement_out = 2 * d_max * (1 - rxy) * ((1 + rxy) ** 2 / 4 - rxy ** 2) ** 0.5
    return displacement_out
def fault_displacement_y(displacement_field, z, fxy, reverse_drag_radius, hwfwradio):
    """
    依照 fault_displacement_field 函数生成的位移场计算y方向位移量
    :param displacement_field: 数据体各点的断层位移场数值，一维数组
    :param z:数据体各点的z坐标，一维数组
    :param fxy:数据体各点的xy坐标对应的断层基准面或其附近种子点插值得到的基准曲面的z值，一维数组
    :param reverse_drag_radius:reverse drag radius γ，浮点型
    :param hwfwradio:hanging-wall and foot-wall displacements，浮点型
    :return:数据体各点y方向位移量，一维数组
    """
    fault_displacement_y_out = displacement_field * 0#与位移场等大的0数组
    z_fxy = z - fxy
    fault_displacement_y_out[np.where((0 <= z_fxy) & (z_fxy < reverse_drag_radius))] = hwfwradio * displacement_field[
        np.where((0 <= z_fxy) & (z_fxy < reverse_drag_radius))] * alpha_function(0, z[np.where((0 <= z_fxy) & (z_fxy < reverse_drag_radius))],
                                                                 reverse_drag_radius)
    fault_displacement_y_out[np.where((-1 * reverse_drag_radius <= z_fxy) & (z_fxy < 0))] = (hwfwradio-1) * displacement_field[
        np.where((-1 * reverse_drag_radius <= z_fxy) & (z_fxy < 0))] * alpha_function(0, z[np.where((-1 * reverse_drag_radius <= z_fxy) & (z_fxy < 0))],
                                                                 reverse_drag_radius)
    return fault_displacement_y_out
def fault_displacement_z(x, y, displacement_y, fxy, fxy_example):
    """
    依照 fault_displacement_y 函数生成的位移场计算z方向位移量
    :param x: 数据体各点的x坐标，一维数组
    :param y: 数据体各点的y坐标，一维数组
    :param displacement_y:数据体各点y方向位移量，一维数组
    :param fxy:数据体各点的xy坐标对应的断层基准面或其附近种子点插值得到的基准曲面的z值，一维数组
    :param fxy_example:数据体各点的xy坐标对应的断层基准面或其附近种子点插值得到的基准曲面的插值函数
    :return:数据体各点y方向位移量，一维数组
    """
    y_edit = y + displacement_y
    fxy_edit = fxy_example(x, y_edit)
    fxy_edit = fxy_edit.reshape(fxy_edit.shape[0],)
    displacement_z = fxy_edit - fxy
    return displacement_z
def R_matrix(strike_angle, dip_angle):
    """
    将XYZ坐标系转化为断层面坐标系xyz的旋转矩阵
    :param strike_angle: 断层基准面的走向方位角，浮点型
    :param dip_angle:断层基准面的倾角，浮点型
    :return:旋转矩阵
    """
    list_matrix_value = []#依照论文给出的公式构建旋转矩阵
    list_matrix_value.append(math.sin(strike_angle))
    list_matrix_value.append(-1 * math.cos(strike_angle))
    list_matrix_value.append(0)
    list_matrix_value.append(math.cos(strike_angle) * math.cos(dip_angle))
    list_matrix_value.append(math.sin(strike_angle) * math.cos(dip_angle))
    list_matrix_value.append(math.sin(dip_angle))
    list_matrix_value.append(math.cos(strike_angle) * math.sin(dip_angle))
    list_matrix_value.append(math.sin(strike_angle) * math.sin(dip_angle))
    list_matrix_value.append(-1 * math.cos(dip_angle))
    matrix = np.mat(np.round(np.array(list_matrix_value), 4).reshape(3, 3))
    return matrix
def alpha_function(fxy, z, reverse_drag_radius):
    """
    论文中计算位移量使用的非线性标量函数
    :param fxy:数据体各点的xy坐标对应的断层基准面或其附近种子点插值得到的基准曲面的z值，一维数组
    :param z:数据体各点的z坐标，一维数组
    :param reverse_drag_radius:reverse drag radius γ，整型
    :return:非线性标量函数值，一维数组
    """
    alpha_xyz = (1 - abs(z - fxy) / reverse_drag_radius) ** 2
    return alpha_xyz
class TrainingData(object):
    """
    合成地震记录类，依照论文中的流程对层状模型添加位移量实现褶皱和断层的模拟，用于神经网络模型的训练
    """
    def __init__(self, random_seed, work_area):
        """
        合成地震记录所需参数和变量
        :param random_seed: 使用的随机数种子，整型
        :param work_area: 合成地震记录尺寸，列表或一维数组
        """
        np.random.seed(random_seed)#载入随机数种子
        self.dt = np.arange(-0.02, 0.02, 0.002)#地震子波的时间范围，[-0.02, 0.02, 0.002]
        self.fm = 30#雷克子波主频
        self.work_area = work_area
        self.wavelet = None
        self.layer, self.cls, self.X, self.Y, self.Z, self.XYZ_whole_number, self.convolve_shape = None, None, None, None, None, None, None#layer:层状模型反射系数
        #cls：层状模型中各层从上到下的次序；self.X、self.Y、self.Z：合成数据体中各点在XYZ坐标系的坐标；self.XYZ_whole_number：合成数据体中样点总数；
        # self.convolve_shape:将合成数据体排成一个二维剖面时的尺寸，使用子波褶积时使用
        self.amplitude, self.amplitude_norm = None, None#合成数据体中各样点振幅和标准化后的振幅
        self.noise = None#合成数据体添加的噪声
        self.number_of_fold, self.number_of_fault = None, None#合成数据体中添加的褶皱的高斯函数个数、合成数据体中添加的断层的个数
        self.fold_parameter = []#用于存放褶皱参数的列表
        self.fault_parameter = []#用于存放断层参数的列表
        self.a0 = 0#褶皱参数a0，作为高斯函数和的偏置添加
        self.bk, self.ck, self.dk, self.ek, self.a, self.b = None, None, None, None, None, None
        #self.bk、self.ck、self.dk、self.ek为高斯函数和的参数，self.a与self.b决定了层状模型的倾斜
        self.strike_angle, self.dip_angle, self.origin, self.fault_surface_seed, self.d_max, self.lxly, self.reverse_drag_radius, self.hw_fw_radio= None, None, None, None, None, None, None, None
        #self.strike_angle：走向方位角；self.dip_angle：倾角；self.origin：每次添加断层时坐标系xyz的原点；self.fault_surface_seed：断层基准面两侧选取的随机种子点，用于基准曲面的插值；
        #self.d_max：位移场的最大位移量；self.lxly：椭圆位移场在x、y方向的半轴长度；self.reverse_drag_radius、self.hw_fw_radio：reverse drag radius γ，hanging-wall and foot-wall displacements
        self.s1_shift, self.s2_shift = None, None#self.s1_shift：添加褶皱的位移量；self.s2_shift：添加倾斜的位移量
        self.layer_model, self.fold_model, self.fault_model, self.show_model, self.saved_model = None, None, None, None, None
        #self.layer_model：层状模型；self.fold_model：添加了倾斜与褶皱的模型；self.fault_model：添加了倾斜、褶皱和断层的模型；
        #self.show_model：pyvista显示类；self.saved_model：模型保存用变量
        self.transform_matrix, self.inverse_transform_matrix, self.X_location, self.Y_location, self.Z_location, self.XYZ_location, self.xyz_location = None, None, None, None, None, None, None
        #self.transform_matrix：从XYZ坐标系转换为xyz坐标系时使用的转换矩阵；self.inverse_transform_matrix：self.transform_matrix的逆矩阵，用于xyz坐标系转换为XYZ坐标系；
        #self.X_location、self.Y_location、self.Z_location：合成数据体各点相对于xyz坐标系原点self.origin的XYZ坐标；
        self.spine_surface, self.fxy, self.displacement_field, self.displacement_y, self.displacement_z = None, None, None, None, None
        #self.spine_surface：断层基准面两侧种子点插值出的曲面；self.fxy：断层基准面的z值；
        #self.displacement_field：椭圆位移场；self.displacement_y：xyz坐标系中y方向位移量；self.displacement_z：xyz坐标系中z方向位移量
        self.mean_value, self.standard_deviation = None, None
    def layer_strcture(self):
        """
        生成层状模型，反射系数在一定范围内随机选取
        :return:
        """
        self.layer, self.cls, self.X, self.Y, self.Z = Layerstrcture(self.work_area[0], self.work_area[1], self.work_area[2])#self.layer : r
        self.XYZ_whole_number = self.work_area[0] * self.work_area[1] * self.work_area[2]
        self.convolve_shape = [self.work_area[0] * self.work_area[1], self.work_area[2]]
        self.layer_model = [self.X, self.Y, self.Z, self.layer]
        print('layer complete!')
        self.layer_model1 = np.vstack([self.X, self.Y, self.Z, self.layer])
        print('self.layer_model1.shape:', self.layer_model1.shape)
        np.save(r"layer_generate_2.npy", self.layer_model1, )
        return
    def ricker_wavelet(self):
        """
        雷克子波
        :return:
        """
        self.wavelet = (1 - 2 * (math.pi * self.fm * self.dt) ** 2) * np.exp(-1 * (math.pi * self.fm * self.dt) ** 2)
        # print('Oh here is wavelet:\n', self.wavelet, type(self.wavelet), self.wavelet.size)
        return
    def wavelet_convolve(self):
        """
        使用雷克子波褶积反射系数
        :return:
        """
        # print('Oh here is ')
        # self.amplitude = WaveletConvolve(wavelet=self.wavelet, rc=self.layer.reshape(self.work_area[0] * self.work_area[1], self.work_area[2])).reshape(self.XYZ_whole_number,)
        self.amplitude = WaveletConvolve(wavelet=self.wavelet, rc=self.layer.reshape(self.convolve_shape)).reshape(self.XYZ_whole_number,)
        # print('Layer:\n', self.layer)
        # print('Amplitude:\n', self.amplitude)
        return
    def random_noise(self):
        """
        添加噪声
        :return:
        """
        # self.noise = np.random.normal(0, 0.01, self.amplitude.shape)
        self.noise = np.random.normal(0, 0.0000001, self.amplitude.shape)
        self.amplitude = (self.amplitude + self.noise).reshape(self.XYZ_whole_number,)
        self.mean_value = np.mean(self.amplitude)
        self.standard_deviation = np.var(self.amplitude) ** 0.5
        self.amplitude_norm = (self.amplitude - self.mean_value) / self.standard_deviation
        return
    def random_parameter(self):
        """
        依照合成数据体尺寸选取自适应的倾斜褶皱断层参数
        :return:
        """
        self.number_of_fold = np.random.randint(10, 25)
        self.bk = np.random.uniform(-0.15*self.work_area[2], 0.15*self.work_area[2], size=self.number_of_fold)
        self.ck = np.random.uniform(0, self.work_area[0], size=self.number_of_fold)
        self.dk = np.random.uniform(0, self.work_area[1], size=self.number_of_fold)
        # self.ek = np.random.uniform(0.8 * np.min([self.work_area[0], self.work_area[1]]), 1 * np.min([self.work_area[0], self.work_area[1]]), size=self.number_of_fold)
        self.ek = np.random.uniform(0.1*self.work_area[0], 0.4*self.work_area[0], size=self.number_of_fold)
        self.a = np.random.uniform(-0.1, 0.1)
        self.b = np.random.uniform(-0.1, 0.1)
        self.fold_parameter.extend([self.a0, self.bk, self.ck, self.dk, self.ek, self.a, self.b])
        self.number_of_fault = np.random.randint(2, 5)
        # self.number_of_fault = 2
        # self.strike_angle = np.random.uniform(0, math.pi * 2, size=self.number_of_fault)
        self.strike_angle = np.ones(shape=(self.number_of_fault,)) * math.pi / 2
        # self.strike_angle[0] = math.pi / 2
        self.dip_angle = np.random.uniform(math.pi*5 / 12, math.pi / 2, size=self.number_of_fault)
        # self.dip_angle[0] = math.pi / 4
        self.origin = np.array([np.random.uniform(0.35 * self.work_area[0], 0.65 * self.work_area[0], size=self.number_of_fault),
                                np.random.uniform(0.35 * self.work_area[1], 0.65 * self.work_area[1], size=self.number_of_fault),
                               np.random.uniform(0.35 * self.work_area[2], 0.65 * self.work_area[2], size=self.number_of_fault)])
        # self.fault_surface_seed = np.array([np.random.uniform(-100, 100, size=(self.number_of_fault, 40)), np.random.uniform(-100, 100, size=(self.number_of_fault, 40)), np.random.uniform(-3, 3, size=(self.number_of_fault, 40))])
        # self.d_max = np.random.uniform(0.01*self.work_area[2], 0.1*self.work_area[2], size=self.number_of_fault) * (-1) ** np.random.randint(0, 2, size=self.number_of_fault)
        self.d_max = np.random.uniform(0.05*self.work_area[2], 0.1*self.work_area[2], size=self.number_of_fault)
        self.lxly = np.array([np.random.uniform(0.15*self.work_area[2], 0.4*self.work_area[2], self.number_of_fault), np.random.uniform(0.15*self.work_area[2], 0.4*self.work_area[2], self.number_of_fault)])
        self.reverse_drag_radius = np.random.uniform(0.05*self.work_area[0], 0.1*self.work_area[0], size=self.number_of_fault)
        self.hw_fw_radio = np.random.uniform(0.4, 0.6, size=self.number_of_fault)
        self.fault_parameter.extend([self.strike_angle, self.dip_angle, self.origin, self.fault_surface_seed, self.d_max, self.lxly, self.reverse_drag_radius, self.hw_fw_radio])
        return
    def fold_generate(self):
        """
        添加倾斜和褶皱
        :return:
        """
        self.s1_shift = self.Z * 0
        for i in range(self.number_of_fold):
            print(i)
            self.s1_shift = self.s1_shift + self.bk[i] * np.exp(-1 * ((self.X - self.ck[i]) ** 2 + (self.Y - self.dk[i]) ** 2) / (2 * self.ek[i] ** 2))
            pass
        self.s1_shift = 1.5 * self.s1_shift * self.Z / self.work_area[2]
        self.s2_shift = self.Z * 0
        # self.s2_shift = self.s2_shift + self.a * self.X + self.b * self.Y + (-1 * self.X * self.work_area[0]//2 - self.Y * self.work_area[1]//2)
        self.s2_shift = self.s2_shift + self.a * self.X + self.b * self.Y - self.a * self.work_area[0]//2 - self.b * self.work_area[1]//2
        self.Z = self.Z + self.s1_shift + self.s2_shift
        self.fold_model = [self.X, self.Y, self.Z]
        self.fold_model1 = np.vstack([self.X, self.Y, self.Z, self.layer])
        print('self.model.shape:', self.fold_model1.shape)
        np.save(r"fold_generate_2.npy", self.fold_model1, )
        return
    def fault_generate(self):
        """
        添加断层
        :return:
        """
        for i in range(self.number_of_fault):
            print('The' + str(i) + 'th fault')
            self.transform_matrix = R_matrix(self.strike_angle[i], self.dip_angle[i])
            self.inverse_transform_matrix = self.transform_matrix.I
            self.X_location, self.Y_location, self.Z_location = self.X - self.origin[0, i], self.Y - self.origin[1, i], self.Z - self.origin[2, i]
            self.XYZ_location = np.mat(np.vstack([self.X_location, self.Y_location, self.Z_location]))
            self.xyz_location = self.transform_matrix * self.XYZ_location
            # self.spine_surface = BiharmonicSplineInterpolation(self.fault_surface_seed[0, i, :], self.fault_surface_seed[1, i, :], self.fault_surface_seed[2, i, :])
            # # print('check:\n', self.xyz_location[0, :].getA().shape, self.xyz_location[1, :].getA().shape)
            # self.fxy = self.spine_surface(self.xyz_location[0, :].getA().reshape(self.XYZ_whole_number,), self.xyz_location[1, :].getA().reshape(self.XYZ_whole_number,))
            # self.fxy = self.fxy.reshape(self.fxy.shape[0],)
            self.fxy = 0
            self.displacement_field = fault_displacement_field(self.xyz_location[0, :].getA().reshape(self.XYZ_whole_number,), self.xyz_location[1, :].getA().reshape(self.XYZ_whole_number,), self.lxly[0, i], self.lxly[1, i], self.d_max[i])
            # print('self.displacement_field:\n', self.displacement_field.shape, self.fxy.shape)
            self.displacement_y = fault_displacement_y(self.displacement_field, self.xyz_location[2, :].getA().reshape(self.XYZ_whole_number,), self.fxy, self.reverse_drag_radius[i],
                                                  self.hw_fw_radio[i])
            self.displacement_z = np.zeros(shape=self.displacement_y.shape)
            # self.displacement_z = fault_displacement_z(self.xyz_location.getA()[0, :], self.xyz_location.getA()[1, :], self.displacement_y, self.fxy, self.spine_surface)
            self.XYZ_location = self.inverse_transform_matrix * (self.xyz_location + np.mat(np.vstack([self.xyz_location[0, :] * 0, self.displacement_y, self.displacement_z]))) + np.mat(
                np.array([self.origin[0, i], self.origin[1, i], self.origin[2, i]]).reshape(3, 1))
            self.X, self.Y, self.Z = self.XYZ_location.getA()[0, :].reshape(self.XYZ_whole_number,), self.XYZ_location.getA()[1, :].reshape(self.XYZ_whole_number,), self.XYZ_location.getA()[2, :].reshape(self.XYZ_whole_number,)
        self.fault_model = [self.X, self.Y, self.Z]
        self.fault_model1 = np.vstack([self.X, self.Y, self.Z, self.layer])
        print('self.model.shape:', self.fault_model1.shape)
        np.save(r"fault_generate_2.npy", self.fault_model1)
        return
    def model_3d_show(self, show_mode):
        """
        使用pyvista显示合成数据体。直接在工作站上无法使用pyvista，需要使用远程桌面在工作站图形界面中使用终端运行pyvista相关函数，
        或是将相关数据传送至本地电脑上显示
        :param show_mode: 需要显示的模型的类别（Layer、Fold、Fault）
        :return:
        """
        if show_mode == 'Layer':
            self.show_model = pv.StructuredGrid(self.layer_model[0].reshape([self.work_area[0], self.work_area[1], self.work_area[2]], order = 'F'), self.layer_model[1].reshape([self.work_area[0], self.work_area[1], self.work_area[2]], order = 'F'), self.layer_model[2].reshape([self.work_area[0], self.work_area[1], self.work_area[2]], order = 'F'))
            self.show_model['R'] = self.cls
            # print('Oh here is amplitude:\n', self.amplitude.shape, type(self.amplitude), self.layer.shape, type(self.layer))
            self.show_model.plot()
        elif show_mode == 'Fold':
            # print('fold_model[0]:\n', self.fold_model[0].shape)
            # print('fold_model[1]:\n', self.fold_model[1].shape)
            # print('fold_model[2]:\n', self.fold_model[2].shape)
            self.show_model = pv.StructuredGrid(self.fold_model[0].reshape([self.work_area[0], self.work_area[1], self.work_area[2]], order = 'F'), self.fold_model[1].reshape([self.work_area[0], self.work_area[1], self.work_area[2]], order = 'F'), self.fold_model[2].reshape([self.work_area[0], self.work_area[1], self.work_area[2]], order = 'F'))
            # self.show_model = pv.StructuredGrid(self.fold_model[0], self.fold_model[1], self.fold_model[2])
            self.show_model['R'] = self.amplitude
            self.show_model.plot()
        elif show_mode == 'Fault':
            # print(r"fault's info:\n", self.strike_angle, self.dip_angle, self.d_max)
            self.show_model = pv.StructuredGrid(self.fault_model[0].reshape([self.work_area[0], self.work_area[1], self.work_area[2]], order = 'F'), self.fault_model[1].reshape([self.work_area[0], self.work_area[1], self.work_area[2]], order = 'F'), self.fault_model[2].reshape([self.work_area[0], self.work_area[1], self.work_area[2]], order = 'F'))
            # print('Oh here is amplitude:\n', self.amplitude.shape, type(self.amplitude), self.layer.shape, type(self.layer))
            # print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
            self.show_model['R'] = self.amplitude
            # print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
            # print(self.show_model)
            # self.show_model.plot()
            pv.plot(self.show_model)
            # print('%%%%%%%')
        return
    def model_save(self):
        """
        将需要保存的数据整合方便保存
        :return:
        """
        self.saved_model = np.moveaxis(np.vstack([self.fault_model[0], self.fault_model[1], self.fault_model[2], self.layer, self.amplitude_norm, self.cls, self.amplitude]), -1, 0)
        return
def Layerstrcture(X_range, Y_range, Z_range):
    """
    生成水平层状模型
    :param X_range: X方向尺寸
    :param Y_range: Y方向尺寸
    :param Z_range: Z方向尺寸
    :return:
    """
    # reflect_rate_list = [-1, -0.9, -0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    reflect_rate_list = np.arange(-1, 1.01, 0.01)#随机选取反射系数
    print('reflect_rate_list:', reflect_rate_list)
    X, Y, Z = np.meshgrid(np.arange(0, X_range, dtype='float16'), np.arange(0, Y_range, dtype='float16'), np.arange(0, Z_range, dtype='float16'))
    print('XYZ:', X.shape)
    # print('X[3, 6, 9], Y[3, 6, 9], Z[3, 6, 9]:', X[3, 6, 9], Y[3, 6, 9], Z[3, 6, 9])
    X, Y, Z = X.reshape(X_range * Y_range * Z_range), Y.reshape(X_range * Y_range * Z_range), Z.reshape(X_range * Y_range * Z_range)
    r = X * 0
    cls = X * 0 + 10
    i = np.random.randint(5, 8)
    shuffle(reflect_rate_list)
    random_para = 0
    while i <Z_range-5:
        print(i)
        # random_para = np.random.randint(0, 21)
        r[Z == i] = reflect_rate_list[random_para]
        cls[Z == i] = random_para
        i += np.random.randint(10, 20)
        random_para += 1
    return r, cls, X, Y, Z
def read_saved_model(file_path):
    """
    读取保存的模型文件
    :param file_path:模型文件路径，对应的为文本文件
    :return:
    """
    model_array = np.moveaxis(np.loadtxt(file_path), -1, 0)
    # print('model_array shape', model_array.shape)
    # print(model_array[0, :].shape)
    X = model_array[0, :]
    Y = model_array[1, :]
    Z = model_array[2, :]
    r = model_array[3, :]
    amp = model_array[4, :]
    X = X.reshape([140, 140, 140], order='F')
    Y = Y.reshape([140, 140, 140], order='F')
    Z = Z.reshape([140, 140, 140], order='F')
    p = pv.Plotter(shape=(1, 2))
    p.subplot(0, 0)
    # _ = p.add_mesh(pv.StructuredGrid(X, Y, Z))
    model_show = pv.StructuredGrid(X, Y, Z)
    model_show['R'] = r
    _ = p.add_mesh(model_show)
    # model_show.plot()
    p.subplot(0, 1)
    model_show1 = pv.StructuredGrid(X, Y, Z)
    model_show1['R'] = amp
    _ = p.add_mesh(model_show1)
    # model_show1.plot()
    p.show()
    return
def read_model(file_path):
    """
    读取保存的模型文件
    :param file_path:模型文件路径，为.npy文件
    :return:
    """
    model = np.load(file_path)
    print('!!!!')
    print(model.shape)
    X = model[0, :]
    Y = model[1, :]
    Z = model[2, :]
    r = model[3, :]
    X = X.reshape([512, 512, 1024], order='F')
    Y = Y.reshape([512, 512, 1024], order='F')
    Z = Z.reshape([512, 512, 1024], order='F')
    p = pv.Plotter(shape=(1, 1))
    p.subplot(0, 0)
    model_show = pv.StructuredGrid(X, Y, Z)
    model_show['r'] = r
    _ = p.add_mesh(model_show, interpolate_before_map=False)
    _ = p.add_axes()
    p.show()
    return
def read_grid_r(file_path):
    """
    读取规整到网格点上的模型文件
    :param file_path: 模型文件路径，对应为.npy文件
    :return:
    """
    r = np.load(file_path)
    grid = pv.UniformGrid()
    grid.dimensions = np.array((512+1, 512+1, 608+1))
    grid.origin=(0, 0, 0)
    grid.spacing=(1, 1, 1)
    grid.cell_arrays['Class'] = r[:, :, :608].flatten(order="F")
    grid.plot(show_edges=False)
    return
def read_layer(file_path):
    return
if __name__ == '__main__':
    print('hello!')
    # r, _, _, _, _ = Layerstrcture(128, 128, 128)
    # r = r.reshape(128, 128, 128)
    # plt.imshow(np.moveaxis(r[10, :, :], 0, -1), cmap=plt.cm.gray)
    # plt.show()

    # a = np.array([1, 2, 3, 4])
    # b = np.array([5, 6, 7, 8])
    # c = a * b
    # print(c)
    # _, _, _, _ = Layerstrcture(512, 512, 608)
    data = TrainingData(130, [256, 256, 256])
    data.layer_strcture()
    data.random_parameter()
    data.fold_generate()
    data.fault_generate()


    # data = np.load(r"layer_generate.npy.npy")
    # x = data[0, :]
    # y = data[1, :]
    # z = data[2, :]
    # r = data[3, :]
    # x = np.trunc(x).astype(int)
    # print('x complete!')
    # y = np.trunc(y).astype(int)
    # print('y complete!')
    # z = np.trunc(z).astype(int)
    # print('z complete!')
    #这一部分使用取整将合成数据结果规整到网格点上，已可以使用一些插值方法进行

    # # output = np.vstack([x, y, z, r])
    # # points = (x, y, z)
    # x1 = np.arange(512)
    # y1 = np.arange(512)
    # z1 = np.arange(1024)
    # x2, y2, z2 = np.meshgrid(x1, y1, z1)
    # x2 = x2.reshape(512*512*1024,)
    # y2 = y2.reshape(512*512*1024,)
    # z2 = z2.reshape(512*512*1024,)
    # point2 = (x2, y2, z2)
    # print('aba')
    # r2 = scipy.interpolate.griddata(points, r, point2, method='nearest')
    # np.save(r"grid_r.npy", output)

    # x2 = x2.reshape([512, 512, 1024], order='F')
    # y2 = y2.reshape([512, 512, 1024], order='F')
    # z2 = z2.reshape([512, 512, 1024], order='F')
    # p = pv.Plotter(shape=(1, 1))
    # p.subplot(0, 0)
    # model_show = pv.StructuredGrid(x2, y2, z2)
    # model_show['r'] = r
    # _ = p.add_mesh(model_show)
    # p.show()
    # read_grid_r(r"D:\毕业备份\res\cls_model.npy")
    # a = np.load(r"D:\毕业备份\res\grid_r1.npy")
    # img = a[105, :, :]
    # plt.imshow(img, cmap=plt.cm.rainbow)
    # plt.show()

    # data = np.load(r"fold_generate.npy")
    # data[0, :] = np.trunc(data[0, :]).astype(int)
    # data[1, :] = np.trunc(data[1, :]).astype(int)
    # data[2, :] = np.trunc(data[2, :]).astype(int)
    # print(data.shape)
    # a = np.zeros(shape=(512, 512, 608), dtype='float32')
    # for i in range(data.shape[1]):
    #     # print(data[0, i], data[1, i], data[2, i])
    #     if 0<=data[0, i]<512 and 0<=data[1, i]<512 and 0<=data[2, i]<608:
    #         a[int(data[0, i] - 0), int(data[1, i] - 0), int(data[2, i] - 0)] = data[3, i]
    #     if i%1000 == 0:
    #         print(i)
    # np.save('grid_fold.npy', a)
    # np.save('grid_r1.npy', a)
    # dt = np.arange(-0.02, 0.02, 0.002)
    # fm = 35
    # wavelet = (1 - 2 * (math.pi * fm * dt) ** 2) * np.exp(-1 * (math.pi * fm * dt) ** 2)
    # b = WaveletConvolve(wavelet, a.reshape(512*512, 1024)).reshape([512, 512, 1024])
    # np.save('grid_Amp.npy', b)

    # r = np.load(r"grid_r1.npy")
    # amp = np.load(r"grid_Amp.npy")
    # # cls = np.zeros(shape=(512, 512, 1024), dtype='float32')
    # # cls[np.where(r==0.67)] = 1
    # # print('1!')
    # # cls[np.where(r==-0.59)] = 2
    # # print('2!')
    # # cls[np.where(r==0.83)] = 3
    # # print('3!')
    # # np.save(r"cls_model.npy", cls)
    # alpha = np.zeros(shape=(512, 1024), dtype='float32')
    # alpha[np.where(r[255, :, :]!=0)] = 1
    # plt.subplot(1, 2, 1)
    # plt.imshow(np.moveaxis(r[255, :, :], 0, -1), cmap=plt.cm.gray)
    # plt.subplot(1, 2, 2)
    # plt.imshow(np.moveaxis(amp[255, :, :], 0, -1), cmap=plt.cm.gray)
    # # plt.imshow(np.moveaxis(r[255, :, :], 0, -1), cmap=plt.cm.rainbow, alpha=np.moveaxis(alpha, 0, -1))
    # plt.show()
    # data_cube = np.load(r"cls_block_model.npy")
    # data_cube = data_cube[:, :, :608]
    # plt.imshow(np.moveaxis(data_cube[:, 455, :], 0, -1), cmap=plt.cm.rainbow)
    # plt.show()
    # cls = np.load(r"cls_model.npy")
    # cls = cls[:, :, :608]
    # # for i in range(5):
    # plt.imshow(np.moveaxis(cls[255, :, :], 0, -1), cmap=plt.cm.gray)
    # # plt.axis('off')
    # plt.show()
    #     plt.imsave('inline_' + str(i*100+55) + '.png', np.moveaxis(cls[i*100+55, :, :], 0, -1))
    #     plt.imsave('xline_' + str(i*100+55) + '.png', np.moveaxis(cls[:, i*100+55, :], 0, -1))

    # cls_block = np.zeros(shape=(512, 512, 608), dtype='float32')
    # for n in range(5):
    #     cls = image.imread(r"inline_" + str(n*100+55) + ".png")
    #     print('img:', cls)
    #     # cls_block = np.zeros(shape=(512, 512, 608), dtype='float32')
    #     for i in range(608):
    #         for j in range(512):
    #             if cls[i, j, 0] == 1:
    #                 cls_block[n*100+55, j, i] = 0
    #             elif cls[i, j, 1] == 1:
    #                 cls_block[n*100+55, j, i] = 1
    #             elif cls[i, j, 2] == 1:
    #                 cls_block[n*100+55, j, i] = 2
    #             else:
    #                 cls_block[n*100+55, j, i] = 3
    #     cls2 = image.imread(r"xline_" + str(n*100+55) + ".png")
    #     for i in range(608):
    #         for j in range(512):
    #             if cls2[i, j, 0] == 1:
    #                 cls_block[j, n*100+55, i] = 0
    #             elif cls2[i, j, 1] == 1:
    #                 cls_block[j, n*100+55, i] = 1
    #             elif cls2[i, j, 2] == 1:
    #                 cls_block[j, n*100+55, i] = 2
    #             else:
    #                 cls_block[j, n*100+55, i] = 3
    # np.save(r"cls_block_model.npy", cls_block)

        # plt.imshow(np.moveaxis(cls_block[55, :, :], 0, -1), cmap=plt.cm.rainbow)
        # plt.show()

    # img = np.load(r"cls_block_model.npy")
    # plt.imshow(np.moveaxis(img[55, :, :], 0, -1), cmap=plt.cm.rainbow)
    # plt.show()

    # data = TrainingData(7914, [256, 256, 256])
    # data.layer_strcture()
    data.ricker_wavelet()
    data.wavelet_convolve()
    data.random_noise()
    # data.random_parameter()
    # print('data.number_of_fold:\n', data.number_of_fold)
    # print('data.bk:\n', data.bk)
    # print('data.ck:\n', data.ck)
    # print('data.dk:\n', data.dk)
    # print('data.ek:\n', data.ek)
    # print('data.fold_parameter:\n', data.fold_parameter)
    # print(data.fold_parameter[1])
    # print('%'*60)
    # print('&'*60)
    # print('%' * 60)
    # print('&' * 60)
    # print('self.a0, self.bk, self.ck, self.dk, self.ek, self.a, self.b\n', data.fold_parameter)
    # print('self.strike_angle, self.dip_angle, self.origin, self.fault_surface_seed, self.d_max, self.lxly, self.reverse_drag_radius, self.hw_fw_radio\n', data.fault_parameter)
    # print('self.fault_surface_seed', data.fault_surface_seed, data.fault_surface_seed.shape)
    # data.fold_generate()
    # data.fault_generate()
    # print('data.fold_model:\n')
    # data.model_3d_show(show_mode='Fault')
    data.model_save()
    # print('data.saved:', data.saved_model.shape, type(data.saved_model[0, 0]), type(data.saved_model[0, 1]), type(data.saved_model[0, 2]), type(data.saved_model[0, 3]), type(data.saved_model[0, 4]))
    np.savetxt(r'1.txt', data.saved_model)
    # # read_saved_model(r'C:\Users\Administrator\Desktop\1.txt')

