# coding=UTF-8
import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange


class FoldFault():
    def __init__(self, work_area, resolution, layer_resolution, strike_angle, dip_angle, fault_lambda, fault_gamma, dmax, x0, y0, z0, lx, ly):
        self.work_area = work_area # 生成的数据体XYZ范围
        self.resolution = resolution # 生成数据体的XYZ分辨率
        self.layer_resolution = layer_resolution # 水平层状模型的Z轴分辨率
        self.rc_cube = np.zeros(shape=(int((work_area[1]-work_area[0]) / resolution[0])+1,
                                       int((work_area[3]-work_area[2]) / resolution[1])+1,
                                       int((work_area[5]-work_area[4]) / resolution[2])+1), dtype='float32') # 反射系数
        self.rc_trace = np.zeros(shape=(int((work_area[5]-work_area[4])*2/layer_resolution)+1,), dtype='float32') # 水平层状模型
        self.index_array = np.zeros(shape=(self.rc_cube.shape[0]*self.rc_cube.shape[1]*self.rc_cube.shape[2], 7), dtype='float32')
        # 共7列，依次为最终X、Y、Z，断层位移后X、Y、Z，褶皱位移前Z
        point_index = np.where(self.rc_cube > -10) # 获取XY索引
        self.index_array[:, 0], self.index_array[:, 3] = point_index[0]*resolution[0]+work_area[0], point_index[0]*resolution[0]+work_area[0]
        self.index_array[:, 1], self.index_array[:, 4] = point_index[1]*resolution[1]+work_area[2], point_index[1]*resolution[1]+work_area[2]
        self.index_array[:, 2], self.index_array[:, 5], self.index_array[:, 6] = point_index[2]*resolution[2]+work_area[4], point_index[2]*resolution[2]+work_area[4], point_index[2]*resolution[2]+work_area[4]
        # fault parameters
        self.strike_angle, self.dip_angle = strike_angle, dip_angle
        self.x0, self.y0, self.z0 = x0, y0, z0
        self.fault_lambda, self.fault_gamma, self.dmax = fault_lambda, fault_gamma, dmax
        self.lx, self.ly = lx, ly

    def reflection(self):
        # 生成层状模型，填充预设波阻抗，假设最上一层波阻抗为1
        self.rc_trace[:20] = 1
        bottom_horizon_index = 20
        bottom_horizon = 1
        self.horizon_index = [0, 20]
        self.layer_r = [1]
        while bottom_horizon_index < self.rc_trace.shape[0]:
            new_reflection = 2 * np.random.uniform()-1 # 预设反射系数
            if new_reflection == 0:
                new_reflection = 2 * np.random.uniform() - 1
            new_layer_thick = np.random.randint(25, 40) # 随机层厚度
            new_horizon = 2*bottom_horizon/(1-new_reflection)-bottom_horizon # 由上层阻抗和反射系数计算下层阻抗
            if bottom_horizon_index+new_layer_thick < self.rc_trace.shape[0]:
                self.rc_trace[bottom_horizon_index:bottom_horizon_index+new_layer_thick] = new_horizon
                self.horizon_index.append(bottom_horizon_index+new_layer_thick)
                self.layer_r.append(new_horizon)
            else:
                self.rc_trace[bottom_horizon_index:] = new_horizon
                self.horizon_index.append(self.rc_trace.shape[0])
                self.layer_r.append(new_horizon)
            bottom_horizon_index = bottom_horizon_index + new_layer_thick
            bottom_horizon = new_horizon

    def add_fault(self):
        # X-Y-Z to x-y-z
        print('fault generating...')
        for i in range(len(self.strike_angle)):
            print(f'fault {i} generating...')
            self.R_matrix = np.array([[np.sin(self.strike_angle[i]), (-1) * np.cos(self.strike_angle[i]), 0],
                                      [np.cos(self.strike_angle[i]) * np.cos(self.dip_angle[i]),
                                       np.sin(self.strike_angle[i]) * np.cos(self.dip_angle[i]), np.sin(self.dip_angle[i])],
                                      [np.cos(self.strike_angle[i]) * np.sin(self.dip_angle[i]),
                                       np.sin(self.strike_angle[i]) * np.sin(self.dip_angle[i]),
                                       (-1) * np.cos(self.dip_angle[i])]],
                                     dtype='float32')
            # XYZ到xyz的旋转矩阵
            self.R_inv = np.moveaxis(self.R_matrix, 0, 1)
            # 逆矩阵
            origion_center = np.vstack([self.index_array[:, 3]-self.x0[i], self.index_array[:, 4]-self.y0[i], self.index_array[:, 5]-self.z0[i]])
            xyz = np.dot(self.R_matrix, origion_center)
            self.xyz_x, self.xyz_y, self.xyz_z = xyz[0, :], xyz[1, :], xyz[2, :]
            min_distance = min(max(self.xyz_x)-min(self.xyz_x), max(self.xyz_y)-min(self.xyz_y))
            self.lx[i], self.ly[i] = self.lx[i] * min_distance, self.ly[i] * min_distance
            # 椭圆位移场的长短轴
            # 上下盘分离
            self.xyz_z_up_index = np.where(self.xyz_z >= 0)
            self.xyz_z_down_index = np.where(self.xyz_z < 0)
            self.xyz_x_up, self.xyz_y_up, self.xyz_z_up = self.xyz_x[self.xyz_z_up_index], self.xyz_y[self.xyz_z_up_index], self.xyz_z[self.xyz_z_up_index]
            self.xyz_x_down, self.xyz_y_down, self.xyz_z_down = self.xyz_x[self.xyz_z_down_index], self.xyz_y[self.xyz_z_down_index], self.xyz_z[self.xyz_z_down_index]
            # -----------------------------------------------
            self.xyz_y_up_origin = self.xyz_y_up * 0 # 上盘位移前y
            self.min_distance = self.xyz_y_up * 0 + 10
            print('hanging-wall...\n')
            for j in trange(int((self.dmax[i]+1) / 0.5)):
                new_y = self.xyz_y_up - j * 0.5
                new_rxy = ((self.xyz_x_up/self.lx[i])**2+(new_y/self.ly[i])**2)**0.5
                new_rxy[np.where(new_rxy>1)] = 1
                new_alpha = (1 - np.abs(self.xyz_z_up/self.fault_gamma[i]))**2
                new_alpha[np.where(self.xyz_z_up>self.fault_gamma[i])] = 0
                new_d_z0 = 2*self.dmax[i]*(1-new_rxy)*((1+new_rxy)**2/4-new_rxy**2)**0.5
                new_y_dy = new_y + self.fault_lambda[i]*new_d_z0*new_alpha
                distance_array = (new_y_dy-self.xyz_y_up)**2
                min_distance_index = np.where(self.min_distance>distance_array)
                self.min_distance[min_distance_index] = distance_array[min_distance_index]
                self.xyz_y_up_origin[min_distance_index] = new_y[min_distance_index]
            self.xyz_y_down_origin = self.xyz_y_down * 0 # 下盘位移前y
            self.min_distance = self.xyz_y_down * 0 + 1
            print('foot-wall...\n')
            for j in trange(int((self.dmax[i]+1) / 0.5)):
                # new_y = self.xyz_y_down - self.dmax + i * 0.5
                new_y = self.xyz_y_down + j * 0.5
                new_rxy = ((self.xyz_x_down/self.lx[i])**2+(new_y/self.ly[i])**2)**0.5
                new_rxy[np.where(new_rxy>1)] = 1
                new_alpha = (1-np.abs(self.xyz_z_down/self.fault_gamma[i]))**2
                new_alpha[np.where(self.xyz_z_down<(-1*self.fault_gamma[i]))] = 0
                new_d_z0 = 2*self.dmax[i]*(1-new_rxy)*((1+new_rxy)**2/4-new_rxy**2)**0.5
                new_y_dy = new_y + (self.fault_lambda[i]-1)*new_d_z0*new_alpha
                distance_array = (new_y_dy-self.xyz_y_down)**2
                min_distance_index = np.where(self.min_distance>distance_array)
                self.min_distance[min_distance_index] = distance_array[min_distance_index]
                self.xyz_y_down_origin[min_distance_index] = new_y[min_distance_index]
            origin_y = self.xyz_y * 1
            origin_y[self.xyz_z_up_index] = self.xyz_y_up_origin
            origin_y[self.xyz_z_down_index] = self.xyz_y_down_origin
            xyz_vstack = np.vstack([self.xyz_x, origin_y, self.xyz_z])
            new_origin_center = np.dot(self.R_inv, xyz_vstack)
            self.index_array[:, 3] = new_origin_center[0, :] + self.x0[i]
            self.index_array[:, 4] = new_origin_center[1, :] + self.y0[i]
            self.index_array[:, 5] = new_origin_center[2, :] + self.z0[i]
            print('*'*50)
        # 对应回层状模型
        layer_new_index = (self.index_array[:, 5] - self.work_area[4] + (self.work_area[5]-self.work_area[4])*0.5)/self.layer_resolution
        point_r = layer_new_index * 0
        for i in range(len(self.layer_r)):
            layer_index = np.where((layer_new_index>=self.horizon_index[i]) & (layer_new_index<self.horizon_index[i+1]))
            point_r[layer_index] = self.layer_r[i]
        index1 = ((self.index_array[:, 0] - self.work_area[0]) / self.resolution[0]).astype('int')
        index2 = ((self.index_array[:, 1] - self.work_area[2]) / self.resolution[1]).astype('int')
        index3 = ((self.index_array[:, 2] - self.work_area[4]) / self.resolution[2]).astype('int')
        self.rc_cube[index1, index2, index3] = point_r
        # 切片显示
        plt.subplot(1, 2, 1)
        plt.imshow(np.moveaxis(self.rc_cube[:, 250, :], 0, 1), cmap=plt.cm.rainbow)
        plt.subplot(1, 2, 2)
        plt.imshow(np.moveaxis(self.rc_cube[200, :, :], 0, 1), cmap=plt.cm.rainbow)
        plt.show()
        np.save('rc_cube_fault.npy', self.rc_cube)

    def fold_add(self, gaussian_function_number=10):
        # fold parameters
        gaussian_shift = self.index_array[:, 6] * 0
        gaussian_X_range = [2 * self.work_area[0] - self.work_area[1], 2 * self.work_area[1] - self.work_area[0]]
        gaussian_Y_range = [2 * self.work_area[2] - self.work_area[3], 2 * self.work_area[3] - self.work_area[2]]
        gaussian_sd_range = [0.2 * min(self.work_area[1] - self.work_area[0], self.work_area[3] - self.work_area[2]),
            0.3 * min(self.work_area[1] - self.work_area[0], self.work_area[3] - self.work_area[2])]
        amplitude_range = [0.07 * (self.work_area[5]-self.work_area[4]), 0.1 * (self.work_area[5]-self.work_area[4])]
        gaussian_center_X = np.random.uniform(low=gaussian_X_range[0], high=gaussian_X_range[1],
                                              size=(gaussian_function_number,))
        gaussian_center_Y = np.random.uniform(low=gaussian_Y_range[0], high=gaussian_Y_range[1],
                                              size=(gaussian_function_number,))
        gaussian_standard_deviation = np.random.uniform(low=gaussian_sd_range[0],
                                                        high=gaussian_sd_range[1],
                                                        size=(gaussian_function_number,))
        gaussian_amplitude = np.random.uniform(low=amplitude_range[0], high=amplitude_range[1],
                                               size=(gaussian_function_number,))
        gaussian_direction = np.power(-1, np.random.randint(1000, size=(gaussian_function_number,)))
        # 位移量累加
        print('fold generating...')
        for i in trange(gaussian_function_number):
            gaussian_shift_single = gaussian_direction[i] * gaussian_amplitude[i] * np.exp(-1 * ((self.index_array[:, 3] - gaussian_center_X[i])**2 + (self.index_array[:, 4] - gaussian_center_Y[i])**2) / (2 * gaussian_standard_deviation[i] ** 2))
            gaussian_shift += gaussian_shift_single
        # 更新褶皱前位置
        self.index_array[:, 6] = self.index_array[:, 5] - (self.index_array[:, 5] - self.work_area[4] + (self.work_area[5] - self.work_area[4]) * 0.5) / ((self.work_area[5] - self.work_area[4])*2) * gaussian_shift
        # 同步至层状模型
        layer_new_index = (self.index_array[:, 6] - self.work_area[4] + (self.work_area[5]-self.work_area[4])*0.5)/self.layer_resolution
        point_r = layer_new_index * 0
        for i in range(len(self.layer_r)):
            layer_index = np.where((layer_new_index>=self.horizon_index[i]) & (layer_new_index<self.horizon_index[i+1]))
            point_r[layer_index] = self.layer_r[i]
        index1 = ((self.index_array[:, 0] - self.work_area[0])/self.resolution[0]).astype('int')
        index2 = ((self.index_array[:, 1] - self.work_area[2])/self.resolution[1]).astype('int')
        index3 = ((self.index_array[:, 2] - self.work_area[4])/self.resolution[2]).astype('int')
        self.rc_cube[index1, index2, index3] = point_r
        # 切片显示
        plt.subplot(1, 2, 1)
        plt.imshow(np.moveaxis(self.rc_cube[:, 250, :], 0, 1), cmap=plt.cm.rainbow)
        plt.subplot(1, 2, 2)
        plt.imshow(np.moveaxis(self.rc_cube[200, :, :], 0, 1), cmap=plt.cm.rainbow)
        plt.show()
        np.save('rc_cube_fold.npy', self.rc_cube)
        # 由层状模型填充的阻抗计算反射系数
        rc_cube1, rc_cube2 = self.rc_cube[:, :, :-1], self.rc_cube[:, :, 1:]
        self.rc_cube = (rc_cube2-rc_cube1)/(rc_cube2+rc_cube1)
        plt.subplot(1, 2, 1)
        plt.imshow(np.moveaxis(self.rc_cube[:, 250, :], 0, 1), cmap=plt.cm.rainbow)
        plt.subplot(1, 2, 2)
        plt.imshow(np.moveaxis(self.rc_cube[200, :, :], 0, 1), cmap=plt.cm.rainbow)
        plt.show()
        np.save('rc_cube.npy', self.rc_cube)

    def wavelet_convolve(self, wavelet_amplitude=100, wavelet_frequency=30, wavelet_dt=0.002, wavelet_length=0.1):
        # 使用雷克子波褶积
        self.seismic_amplitude = np.zeros(shape=self.rc_cube.shape, dtype='float32')
        time_array = np.arange(-wavelet_length / 2, wavelet_length / 2, wavelet_dt, dtype='float32')
        wavelet_function = wavelet_amplitude * (1 - 2 * np.pi ** 2 * wavelet_frequency ** 2 * time_array ** 2) * np.exp(-np.pi ** 2 * wavelet_frequency ** 2 * time_array ** 2)
        for i in range(self.rc_cube.shape[0]):
            for j in range(self.rc_cube.shape[1]):
                self.seismic_amplitude[i, j, :] = np.convolve(wavelet_function, self.rc_cube[i, j, :], mode='same')
        # 切片显示
        plt.subplot(1, 2, 1)
        plt.imshow(np.moveaxis(self.seismic_amplitude[:, 250, :], 0, 1), cmap=plt.cm.rainbow)
        plt.subplot(1, 2, 2)
        plt.imshow(np.moveaxis(self.seismic_amplitude[200, :, :], 0, 1), cmap=plt.cm.rainbow)
        plt.show()

    def add_noise(self, snr):
        # 添加高斯噪声
        noise = np.zeros(shape=self.seismic_amplitude.shape, dtype='float32')
        for i in range(noise.shape[2]):
            noise[:, :, i] = np.random.normal(loc=0, scale=np.exp(i/noise.shape[2]), size=(noise.shape[0], noise.shape[1]))
        for i in range(noise.shape[0]):
            for j in range(noise.shape[1]):
                new_noise_array = noise[i, j, :]
                seismic_power = np.sum(self.seismic_amplitude[i, j, :] ** 2) / noise.shape[2]
                new_noise_power = np.sum(new_noise_array ** 2) / noise.shape[2]
                noise_energy = seismic_power / np.power(10, snr/10)
                new_noise_array = np.sqrt(noise_energy / new_noise_power) * new_noise_array
                noise[i, j, :] = new_noise_array
        self.seismic_amplitude_noise = noise + self.seismic_amplitude
        # 切片显示
        plt.subplot(1, 2, 1)
        plt.imshow(np.moveaxis(self.seismic_amplitude_noise[:, 250, :], 0, 1), cmap=plt.cm.rainbow)
        plt.subplot(1, 2, 2)
        plt.imshow(np.moveaxis(self.seismic_amplitude_noise[200, :, :], 0, 1), cmap=plt.cm.rainbow)
        plt.show()

if __name__ == '__main__':
    print('hello')
    np.random.seed(20231012)
    strike_angle, dip_angle = [np.pi/4, np.pi/3], [np.pi/4, np.pi/3]
    x0, y0, z0 = [250, 750], [300, 500], [200, 500]
    fault_lambda, dmax, fault_gamma = [0.5, 0.5], [150, 220], [500, 500]
    lx, ly = [1.4, 1.2], [0.9, 0.8]
    a = FoldFault(work_area=[0, 1000, 0, 800, 0, 600], resolution=[2, 2, 2], layer_resolution=1,
                    strike_angle=strike_angle, dip_angle=dip_angle, fault_lambda=fault_lambda,
                      fault_gamma=fault_gamma, dmax=dmax, x0=x0, y0=y0, z0=z0, lx=lx, ly=ly)
    a.reflection()
    a.add_fault()
    a.fold_add(gaussian_function_number=70)
    a.wavelet_convolve()
    a.add_noise(snr=5)

