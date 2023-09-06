# coding=UTF-8
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
from PIL import Image, ImageDraw
from collections import Counter
from tqdm import trange


class SimulationDataGeneration():
    def __init__(self, work_area, resolution):
        self.X_min, self.X_max, self.Y_min, self.Y_max, self.Z_min, self.Z_max = work_area[0], work_area[1], work_area[2], work_area[3], work_area[4], work_area[5]
        self.X_resolution, self.Y_resolution, self.Z_resolution = resolution[0], resolution[1], resolution[2]
        self.X_arange = np.arange(start=self.X_min, stop=self.X_max + self.X_resolution, step=self.X_resolution, dtype='float32')
        self.Y_arange = np.arange(start=self.Y_min, stop=self.Y_max + self.Y_resolution, step=self.Y_resolution, dtype='float32')
        self.Z_arange = np.arange(start=self.Z_min, stop=self.Z_max + self.Z_resolution, step=self.Z_resolution, dtype='float32')
        self.X, self.Y, self.Z = np.meshgrid(self.X_arange, self.Y_arange, self.Z_arange, indexing='ij')
        print(f"self.X:{self.X.dtype}")
        self.X_point_number, self.Y_point_number, self.Z_point_number = len(self.X_arange), len(self.Y_arange), len(self.Z_arange)
        self.grid_point_number = self.X_point_number * self.Y_point_number * self.Z_point_number
        self.rc = np.zeros(shape=self.Z.shape, dtype='float32')
        self.rgt = np.zeros(shape=self.Z.shape, dtype='float32')
        self.fault_marker = np.zeros(shape=self.Z.shape, dtype='float32')
        self.fault_marker_dict = {}
        self.horizon_rc_list = []
        self.horizon_rgt_list = []
        self.rc_grid = []
        self.rgt_grid = []
        self.seismic_amplitude = []

    def reflection(self):
        bottom_horizon = 2
        self.horizon_rc_list = []
        self.horizon_rgt_list = []
        for i in range(self.X.shape[2]):
            if i < bottom_horizon:
                self.rgt[:, :, i] = i
                self.horizon_rc_list.append(0)
                self.horizon_rgt_list.append(i)
            elif i >= bottom_horizon:
                new_reflection = 2 * np.random.random_sample() - 1
                if new_reflection == 0:
                    new_reflection = 2 * np.random.random_sample() - 1
                self.rc[:, :, i] = new_reflection
                self.rgt[:, :, i] = i
                self.horizon_rc_list.append(new_reflection)
                self.horizon_rgt_list.append(i)
                bottom_horizon += np.random.randint(5, 10)

    def add_fold(self, gaussian_function_number=10, gaussian_X_range=None, gaussian_Y_range=None, gaussian_standard_deviation_range=None, amplitude_range=None):
        gaussian_shift = np.zeros(shape=self.Z.shape, dtype='float32')
        if gaussian_X_range is None:
            gaussian_X_range = [self.X_min+0.1*(self.X_max-self.X_min), self.X_min+0.9*(self.X_max-self.X_min)]
        if gaussian_Y_range is None:
            gaussian_Y_range = [self.Y_min+0.1*(self.Y_max-self.Y_min), self.Y_min+0.9*(self.Y_max-self.Y_min)]
        if gaussian_standard_deviation_range is None:
            gaussian_standard_deviation_range = [0.05 * min(self.X_max - self.X_min, self.Y_max - self.Y_min), 0.15 * min(self.X_max - self.X_min, self.Y_max - self.Y_min)]
        if amplitude_range is None:
            amplitude_range = [0.001 * (self.Z_max - self.Z_min), 0.07 * (self.Z_max - self.Z_min)]
        gaussian_center_X = np.random.uniform(low=gaussian_X_range[0], high=gaussian_X_range[1], size=(gaussian_function_number,))
        gaussian_center_Y = np.random.uniform(low=gaussian_Y_range[0], high=gaussian_Y_range[1], size=(gaussian_function_number,))
        gaussian_standard_deviation = np.random.uniform(low=gaussian_standard_deviation_range[0], high=gaussian_standard_deviation_range[1], size=(gaussian_function_number,))
        gaussian_amplitude = np.random.uniform(low=amplitude_range[0], high=amplitude_range[1], size=(gaussian_function_number,))
        gaussian_direction = np.power(-1, np.random.randint(1000, size=(gaussian_function_number,)))
        for i in trange(gaussian_function_number):
            gaussian_shift_single = gaussian_direction[i] * gaussian_amplitude[i] * np.exp(-1 * ((self.X - gaussian_center_X[i]) ** 2 + (self.Y - gaussian_center_Y[i]) ** 2) / (2 * gaussian_standard_deviation[i] ** 2))
            gaussian_shift += gaussian_shift_single
        self.Z = self.Z - self.Z / self.Z_max * gaussian_shift
        self.Z[self.Z > self.Z_max] = self.Z_max
        self.Z[self.Z < self.Z_min] = self.Z_min

    def add_fault(self, fault_number=3, fault_center_range=None, fault_strike_range=None, fault_dip_range=None, fault_shift_range=None,
                      fault_oval_lx_range=None, fault_oval_ly_range=None, fault_gamma_range=None, fault_lambda_range=None,
                      fault_curved_surface=False, control_point_number=20, control_point_range=None):
        if fault_center_range is None:
            fault_center_range = [self.X.min(), self.X.max(), self.Y.min(), self.Y.max(),
                                  self.Z.min()+0.2*(self.Z.max()-self.Z.min()), self.Z.max()]
        if fault_strike_range is None:
            fault_strike_range = [np.pi / 4, np.pi * 3 / 4]
        if fault_dip_range is None:
            fault_dip_range = [np.pi / 6, np.pi / 3]
        if fault_shift_range is None:
            fault_shift_range = [0.04, 0.05]
        if fault_oval_lx_range is None:
            fault_oval_lx_range = [0.1, 0.2]
        if fault_oval_ly_range is None:
            fault_oval_ly_range = [0.07, 0.1]
        if fault_gamma_range is None:
            # fault_gamma_range = [0.1, 0.3]
            fault_gamma_range = [0.1, 0.15]
        if fault_lambda_range is None:
            fault_lambda_range = [0.5, 0.6]
        fault_center_X = np.random.uniform(low=fault_center_range[0], high=fault_center_range[1], size=(fault_number,))
        fault_center_Y = np.random.uniform(low=fault_center_range[2], high=fault_center_range[3], size=(fault_number,))
        fault_center_Z = np.random.uniform(low=fault_center_range[4], high=fault_center_range[5], size=(fault_number,))
        fault_strike = np.random.uniform(low=fault_strike_range[0], high=fault_strike_range[1], size=(fault_number,))
        fault_dip = np.random.uniform(low=fault_dip_range[0], high=fault_dip_range[1], size=(fault_number,))
        for i in range(fault_number):
            print("\r The %d fault is generating..."%(i+1))
            # fault_center_X, fault_center_Y, fault_center_Z = np.random.uniform(low=fault_center_range[0], high=fault_center_range[1]), \
            #                                                  np.random.uniform(low=fault_center_range[2], high=fault_center_range[3]), \
            #                                                  np.random.uniform(low=fault_center_range[4], high=fault_center_range[5])
            # fault_strike = np.random.uniform(low=fault_strike_range[0], high=fault_strike_range[1])
            # fault_dip = np.random.uniform(low=fault_dip_range[0], high=fault_dip_range[1])
            fault_transform_matrix = np.array([[np.sin(fault_strike[i]), (-1) * np.cos(fault_strike[i]), 0],
                                               [np.cos(fault_strike[i]) * np.cos(fault_dip[i]), np.sin(fault_strike[i]) * np.cos(fault_dip[i]), np.sin(fault_dip[i])],
                                               [np.cos(fault_strike[i]) * np.sin(fault_dip[i]), np.sin(fault_strike[i]) * np.sin(fault_dip[i]), (-1) * np.cos(fault_dip[i])]], dtype='float32')
            origin_to_center = np.array([(self.X - fault_center_X[i]).reshape(self.grid_point_number, ),
                                         (self.Y - fault_center_Y[i]).reshape(self.grid_point_number, ),
                                         (self.Z - fault_center_Z[i]).reshape(self.grid_point_number, )], dtype='float32')
            transform_result = np.dot(fault_transform_matrix, origin_to_center)
            if i > 0:
                for j in range(i):
                    self.fault_marker_dict[j][0, :] = self.fault_marker_dict[j][0, :]-fault_center_X[i]
                    self.fault_marker_dict[j][1, :] = self.fault_marker_dict[j][1, :]-fault_center_Y[i]
                    self.fault_marker_dict[j][2, :] = self.fault_marker_dict[j][2, :]-fault_center_Z[i]
                    self.fault_marker_dict[j] = np.dot(fault_transform_matrix, self.fault_marker_dict[j])
            x, y, z = transform_result[0, :].reshape(self.X.shape), transform_result[1, :].reshape(self.Y.shape), transform_result[2, :].reshape(self.Z.shape)
            min_distance = min(x.max()-x.min(), y.max()-y.min())
            fault_shift_max = np.random.uniform(low=fault_shift_range[0], high=fault_shift_range[1]) * (y.max() - y.min())
            fault_lx = np.random.uniform(low=fault_oval_lx_range[0], high=fault_oval_lx_range[1]) * min_distance
            fault_ly = np.random.uniform(low=fault_oval_ly_range[0] * min_distance, high=fault_oval_ly_range[1] * min_distance)
            fault_array = oval_list(fault_lx, fault_ly)
            self.fault_marker_dict[i] = fault_array
            fault_gamma = np.random.uniform(low=fault_gamma_range[0], high=fault_gamma_range[1]) * (z.max() - z.min())
            fault_lambda = np.random.uniform(low=fault_lambda_range[0], high=fault_lambda_range[1])
            print('lx, ly, shift_max:', fault_lx, fault_ly, fault_shift_max, fault_strike[i], fault_dip[i], fault_gamma)
            print(f"lx:{fault_lx, fault_lx/min_distance}", f"ly:{fault_ly, fault_ly/min_distance}", f"shift_max:{fault_shift_max, fault_shift_max /  (y.max() - y.min())}",
                  f"strike:{fault_strike[i]}", f"dip:{fault_dip[i]}", f"gamma:{fault_gamma, fault_gamma/(z.max() - z.min())}")
            fault_oval_r = np.sqrt((x / fault_lx) ** 2 + (y / fault_ly) ** 2)
            fault_oval_r[fault_oval_r>1] = 1
            fault_oval_displacement = 2 * fault_shift_max * (1 - fault_oval_r) * np.sqrt((1 + fault_oval_r) ** 2 / 4 - fault_oval_r ** 2)
            fault_plane_surface = 0
            fault_alpha_function = (1 - np.abs(z - fault_plane_surface) / fault_gamma) ** 2
            fault_alpha_function[(z - fault_plane_surface) ** 2 > fault_gamma ** 2]= 0
            fault_displacement_x = 0
            fault_displacement_y = np.zeros(shape=y.shape, dtype='float32')
            fault_displacement_z = 0
            fault_displacement_y[z>=fault_plane_surface] = fault_lambda * fault_oval_displacement[z>=fault_plane_surface] * fault_alpha_function[z>=fault_plane_surface]
            fault_displacement_y[z<fault_plane_surface] = (fault_lambda - 1) * fault_oval_displacement[z<fault_plane_surface] * fault_alpha_function[(z<fault_plane_surface)]
            x = x + fault_displacement_x
            y = y + fault_displacement_y
            z = z + fault_displacement_z
            center_to_origin = np.array([x.reshape(self.grid_point_number, ),
                                         y.reshape(self.grid_point_number, ),
                                         z.reshape(self.grid_point_number, )], dtype='float32')
            inverse_transform_result = np.dot(np.linalg.inv(fault_transform_matrix), center_to_origin)
            if i > 0:
                for j in range(i):
                    self.fault_marker_dict[j] = np.dot(np.linalg.inv(fault_transform_matrix), self.fault_marker_dict[j])
                    self.fault_marker_dict[j][0, :] = self.fault_marker_dict[j][0, :] + fault_center_X[i]
                    self.fault_marker_dict[j][1, :] = self.fault_marker_dict[j][1, :] + fault_center_Y[i]
                    self.fault_marker_dict[j][2, :] = self.fault_marker_dict[j][2, :] + fault_center_Z[i]
            self.fault_marker_dict[i] = np.dot(np.linalg.inv(fault_transform_matrix), self.fault_marker_dict[i])
            self.X, self.Y, self.Z = (inverse_transform_result[0, :] + fault_center_X[i]).reshape(self.X.shape), \
                                     (inverse_transform_result[1, :] + fault_center_Y[i]).reshape(self.Y.shape), \
                                     (inverse_transform_result[2, :] + fault_center_Z[i]).reshape(self.Z.shape)
            self.X[self.X > self.X_max] = self.X_max
            self.X[self.X < self.X_min] = self.X_min
            self.Y[self.Y > self.Y_max] = self.Y_max
            self.Y[self.Y < self.Y_min] = self.Y_min
            self.Z[self.Z > self.Z_max] = self.Z_max
            self.Z[self.Z < self.Z_min] = self.Z_min

    def inpyvista_show(self, show_resolution=None):
        print("!!!")
        print(f"self.X.shape:{self.X.shape}")
        crop_info = {'xmin': np.round(np.max(self.X[0, :, :])), 'xmax': np.round(np.min(self.X[-1, :, :])),
                         'ymin': np.round(np.max(self.Y[:, 0, :])), 'ymax': np.round(np.min(self.Y[:, -1, :])),
                         'zmin': np.round(np.max(self.Z[:, :, 0])), 'zmax': np.round(np.min(self.Z[:, :, -1]))}
        print('origin work area:', self.X_min, self.X_max, self.Y_min, self.Y_max, self.Z_min, self.Z_max)
        print('crop work area:', crop_info)
        if show_resolution is None:
            X_resolution_new, Y_resolution_new, Z_resolution_new = self.X_resolution, self.Y_resolution, self.Z_resolution
        else:
            X_resolution_new, Y_resolution_new, Z_resolution_new = show_resolution[0], show_resolution[1], show_resolution[2]
        crop_area_index = np.where((crop_info['xmin']<=self.X) & (self.X<=crop_info['xmax']) &
                                        (crop_info['ymin']<=self.Y) & (self.Y<=crop_info['ymax']) &
                                        (crop_info['zmin']<=self.Z) & (self.Z<=crop_info['zmax']))
        print(crop_area_index, crop_area_index[0].shape)
        crop_area_x = ((crop_info['xmin'] // X_resolution_new) * X_resolution_new,
                            (crop_info['xmax'] // X_resolution_new) * X_resolution_new)
        crop_area_y = ((crop_info['ymin'] // Y_resolution_new) * Y_resolution_new,
                            (crop_info['ymax'] // Y_resolution_new) * Y_resolution_new)
        crop_area_z = ((crop_info['zmin'] // Z_resolution_new) * Z_resolution_new,
                            (crop_info['zmax'] // Z_resolution_new) * Z_resolution_new)
        X_arange_grid =  np.arange(start=crop_info['xmin'], stop=crop_info['xmax'] + X_resolution_new, step=X_resolution_new, dtype='float32')
        Y_arange_grid =  np.arange(start=crop_info['ymin'], stop=crop_info['ymax'] + Y_resolution_new, step=Y_resolution_new, dtype='float32')
        X_grid, Y_grid = np.meshgrid(X_arange_grid, Y_arange_grid, indexing='ij')
        horizon_interpolate = np.moveaxis(np.vstack([X_grid.flatten(), Y_grid.flatten()]), 0, -1)
        self.rc_grid = np.zeros(shape=(int((crop_area_x[1]-crop_area_x[0])/X_resolution_new)+1,
                                       int((crop_area_y[1]-crop_area_y[0])/Y_resolution_new)+1,
                                       int((crop_area_z[1]-crop_area_z[0])/Z_resolution_new)+1),
                                dtype='float32')
        self.rgt_grid = np.zeros(shape=(int((crop_area_x[1]-crop_area_x[0])/X_resolution_new)+1,
                                       int((crop_area_y[1]-crop_area_y[0])/Y_resolution_new)+1,
                                       int((crop_area_z[1]-crop_area_z[0])/Z_resolution_new)+1), dtype='float32')
        for i in trange(self.X.shape[2]):
            if self.rc[0, 0, i] != 0:
                horizon_control_point = np.moveaxis(np.vstack([self.X[:, :, i].flatten(), self.Y[:, :, i].flatten()]), 0, -1)
                horizon_control_value = self.Z[:, :, i].flatten()
                interpolate_result = interpolate.griddata(points=horizon_control_point, values=horizon_control_value,
                                                         xi=horizon_interpolate, method='cubic')
                for j in range(interpolate_result.shape[0]):
                    if crop_info['xmin']<=horizon_interpolate[j, 0]<=crop_info['xmax']:
                        if crop_info['ymin']<=horizon_interpolate[j, 1]<=crop_info['ymax']:
                            if crop_info['zmin']<=interpolate_result[j]<=crop_info['zmax']:
                                self.rgt_grid[int((horizon_interpolate[j, 0]-crop_area_x[0])/X_resolution_new),
                                int((horizon_interpolate[j, 1]-crop_area_y[0])/Y_resolution_new),
                                int((interpolate_result[j]-crop_area_z[0])/Z_resolution_new)] = self.rgt[0, 0, i]
        horizon_rgt_max = np.max(self.rgt_grid)
        for i in range(len(self.horizon_rgt_list)):
            if self.horizon_rgt_list[i] <= horizon_rgt_max and self.horizon_rc_list[i] != 0:
                rc_interplote_index = np.where(self.rgt_grid == self.horizon_rgt_list[i])
                # self.rc_grid[rc_interplote_index] = self.horizon_rc_list[i] + np.random.normal(loc=0, scale=0.3*np.abs(self.horizon_rc_list[i]), size=(rc_interplote_index[0].shape[0],))
                rc_interplote_abs = np.abs(self.horizon_rc_list[i])
                self.rc_grid[rc_interplote_index] = self.horizon_rc_list[i] + np.random.uniform(low=-0.05 * rc_interplote_abs, high=0.05 * rc_interplote_abs, size=(rc_interplote_index[0].shape[0],))
                print(f"rc_grid_noise:{i, np.max(self.rc_grid[rc_interplote_index]), np.min(self.rc_grid[rc_interplote_index]), self.horizon_rc_list[i]}")
        self.rc_grid[np.where(self.rgt_grid > 1)] = 1
        self.rc_grid[np.where(self.rgt_grid < -1)] = -1

    def wavelet_convolve(self, wavelet_amplitude=100, wavelet_frequency=30, wavelet_dt=0.002, wavelet_length=0.1):
        self.seismic_amplitude = np.zeros(shape=self.rc_grid.shape, dtype='float32')
        time_array = np.arange(-wavelet_length / 2, wavelet_length / 2, wavelet_dt, dtype='float32')
        wavelet_function = wavelet_amplitude * (1 - 2 * np.pi ** 2 * wavelet_frequency ** 2 * time_array ** 2) * np.exp(-np.pi ** 2 * wavelet_frequency ** 2 * time_array ** 2)
        for i in range(self.rc_grid.shape[0]):
            for j in range(self.rc_grid.shape[1]):
                self.seismic_amplitude[i, j, :] = np.convolve(wavelet_function, self.rc_grid[i, j, :], mode='same')

    def add_noise(self, snr):
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
        self.seismic_amplitude_noise_random = noise
        self.seismic_amplitude_noise = self.seismic_amplitude + self.seismic_amplitude_noise_random

    def add_noise_real(self, snr, real_noise):
        noise_real = np.load(real_noise)
        index_x = int(self.seismic_amplitude.shape[0]/2)
        index_xx = int(noise_real.shape[0]/2)
        index_y = int(self.seismic_amplitude.shape[1]/2)
        index_yy = int(noise_real.shape[1]/2)
        index_z = int(self.seismic_amplitude.shape[2]/2)
        index_zz = int(noise_real.shape[2]/2)
        noise_real = noise_real[index_xx-index_x:index_xx-index_x+self.seismic_amplitude.shape[0],
                          index_yy-index_y:index_yy-index_y+self.seismic_amplitude.shape[1],
                          index_zz-index_z:index_zz-index_z+self.seismic_amplitude.shape[2]]
        for i in trange(self.seismic_amplitude.shape[0]):
            for j in range(self.seismic_amplitude.shape[1]):
                seismic_power = np.sum(self.seismic_amplitude[i, j, :] ** 2) / self.seismic_amplitude.shape[2]
                new_noise_power = np.sum(noise_real[i, j, :] ** 2) / self.seismic_amplitude.shape[2]
                noise_energy = seismic_power / (np.power(10, (snr / 10)))
                noise_trace = np.sqrt(noise_energy / new_noise_power) * noise_real[i, j, :]
                noise_real[i, j, :] = noise_trace
        self.seismic_amplitude_noise = self.seismic_amplitude + noise_real + self.seismic_amplitude_noise_random


def oval_list(lx, ly, resolution=10):
    x_arange = np.arange(-(int(lx/resolution)+resolution), (int(lx/resolution)+resolution)+1, step=resolution)
    y_arange = np.arange(-(int(ly/resolution)+resolution), (int(ly/resolution)+resolution)+1, step=resolution)
    x_grid, y_grid = np.meshgrid(x_arange, y_arange, indexing='ij')
    x_list = []
    y_list = []
    z_list = []
    for i in range(x_grid.shape[0]):
        for j in range(x_grid.shape[1]):
            check_value = (x_grid[i, j]/lx)**2 + (y_grid[i, j]/ly)**2
            if check_value <= 1:
                x_list.append(x_grid[i, j])
                y_list.append(y_grid[i, j])
                z_list.append(0)
    location_array = np.vstack([np.array(x_list), np.array(y_list), np.array(z_list)])
    return location_array

if __name__ == '__main__':
    print('hello!')
    # np.random.seed()
    sample_a = SimulationDataGeneration([0, 10000, 0, 10000, 0, 1000], [20, 20, 2])
    sample_a.reflection()
    sample_a.add_fold(gaussian_function_number=200)
    sample_a.add_fault(fault_number=10)
    sample_a.inpyvista_show()
    np.save("rc_grid.npy", sample_a.rc_grid)
    np.save("rgt_grid.npy", sample_a.rgt_grid)
    print(f"sample_a.rc_grid.shape:{sample_a.rc_grid.shape}")
    sample_a.wavelet_convolve()
    np.save("seismic_amplitude.npy", sample_a.seismic_amplitude)
    # sample_a.add_noise(snr=2)
    sample_a.add_noise(snr=2)
    # sample_a.add_noise_real(snr=10, real_noise="noise_cube.npy")
    np.save("seismic_amplitude_noise.npy", sample_a.seismic_amplitude_noise)
