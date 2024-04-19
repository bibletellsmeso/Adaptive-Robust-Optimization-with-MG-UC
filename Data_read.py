import os
import pandas as pd
import numpy as np
import random
from PV_model import PV_forecast
import matplotlib.pyplot as plt
np.set_printoptions(precision=6, suppress=True)

# os.listdir: 전체 파일목록 가져오기, endswith(): 특정 확장자 파일인지 확인
file_list = [i for i in os.listdir('./') if i.endswith('.xls')]

for i in file_list:
	SMP = pd.read_excel('{}'.format(i), sheet_name='Sheet1')
    # 행 날리기
	SMP = SMP.drop(index=[0, 1, 26, 27, 28], axis=0)
    # label 열 날리기
	SMP = SMP.drop(labels=['육지 SMP 목록'], axis=1)
    # 7.11(월) SMP 사용 위해 나머지 날리기
	SMP = SMP.drop(labels=['Unnamed: {}'.format(i) for i in range(1, 7)], axis=1)
    # 열을 행 모습으로 Transpose
	SMP = np.array(SMP).flatten()
    # np.tile(A, reps=(a,b)): A를 행으로 a번 열로 b번 반복 쌓기
	SMP = np.tile(SMP, reps=1)
    # 24시간을 15분 단위로 쪼개서 쌓기 
	SMP = np.repeat(SMP, repeats=4, axis=0)/4

# kpx 데이터 불러오기, 클래스 내의 지역 변수들
class data_read:
	def __init__(self):
		self.Sim_time = 96
		self.N_PWL = 10
		self.RTE = 0.93

		self.kpx_PV_data = pd.read_csv('KPX_PV.csv', sep=',', names=['Source', 'Location', 'Date', 'Hour', 'Power'], dtype={'Date': str, 'Hour': str, 'Power': str}, encoding='CP949')[1:] # dtype = DataFrame
		self.kpx_PV_data = pd.DataFrame(self.kpx_PV_data, columns=['Hour', 'Power']).to_numpy(dtype=np.float32)

		self.kpx_load = pd.read_csv('KPX_Load.csv', sep=',', names=['Date', 'Load_1', 'Load_2', 'Load_3', 'Load_4', 'Load_5', 'Load_6', 'Load_7', 'Load_8', 'Load_9', 'Load_10', 'Load_11',
							      'Load_12', 'Load_13', 'Load_14', 'Load_15', 'Load_16', 'Load_17', 'Load_18', 'Load_19', 'Load_20', 'Load_21', 'Load_22', 'Load_23', 'Load_0'], encoding='CP949')[1:]
		self.kpx_load = self.kpx_load.drop(['Date'], axis=1).to_numpy(dtype=np.float32) / 1000

        # 각 시간대별로 데이터 모음(전처리)
		self.kpx_PV = []
		for i in range(24):
			self.kpx_PV.append(self.kpx_PV_data[self.kpx_PV_data[:, 0] == i, -1])
        
		self.PV_var = np.array([])
		self.load_var = np.array([])

		for i in range(24):
			self.PV_var = np.append(self.PV_var, np.nanvar(self.kpx_PV[i]/np.max(self.kpx_PV)))
			self.load_var = np.append(self.load_var, np.nanvar(self.kpx_load[:, i - 1] / np.max(self.kpx_load)))

		self.PV_pred = PV_forecast.PV_output
		self.PV_pred[self.PV_pred < 0] = 0
		self.load_pred = np.array(pd.read_csv('Load_for_scheduling.txt', names=['Load']), dtype=np.float32)[:,0]
		self.load_pred[self.load_pred < 0] = 0

		df = pd.read_csv('egg_load.csv', sep=',', names=['Date', 'Total', 'Real'], dtype={'Date': str, 'Total': str, 'Real': str}, encoding='CP949')
		self.load_egg = df.loc[range(0, 286, 3)].reset_index(drop=True)['Total'].astype(float)
		self.load_egg = self.load_egg / np.mean(self.load_egg)
		self.load_egg = self.load_egg * np.mean(self.load_pred)

		# print(np.mean(self.load_pred))
		
		# self.data = np.array(pd.concat([pd.read_csv('PV_prediction.txt', names=['PV']), pd.read_csv('Load_for_scheduling.txt', names=['Load'])], axis=1), dtype=np.float32)
		# self.PV_Gen, self.load = (max(self.data[:, 0]), max(self.data[:, 1]))
        # 신뢰구간 95% , 표본 1개, 24h을 96time slot으로 쪼개기
		self.PV_pos = self.PV_pred * 1.96 * np.sqrt(np.repeat(self.PV_var, repeats=4, axis=0))/np.sqrt(1)
		self.PV_neg = self.PV_pred * 1.96 * np.sqrt(np.repeat(self.PV_var, repeats=4, axis=0))/np.sqrt(1)
		self.load_pos = self.load_egg * 1.96 * np.sqrt(np.repeat(self.load_var, repeats=4, axis=0))/np.sqrt(1)
		self.load_neg = self.load_egg * 1.96 * np.sqrt(np.repeat(self.load_var, repeats=4, axis=0))/np.sqrt(1)

		self.PV_max = self.PV_pred + self.PV_pos
		self.PV_min = self.PV_pred - self.PV_neg
		self.load_max = self.load_egg + self.load_pos
		self.load_min = self.load_egg - self.load_neg

		# Initialize the PV_oracle array
		self.PV_oracle = np.zeros(self.Sim_time)
		self.load_oracle = np.zeros(self.Sim_time)

		# Set the initial value for PV_oracle as the first predicted value
		self.PV_oracle[0] = self.PV_pred[0] 
		self.load_oracle[0] = self.load_egg[0]

		def get_random_value(lower_bound, upper_bound):
			return random.uniform(lower_bound, upper_bound)
		
		self.trend_factor = []
		for i in range(self.Sim_time):
			self.trend = i / (self.Sim_time - 1)
			self.trend_factor.append(self.trend)

		# Calculate the error based on the time index and trend factor
		self.PV_error = self.PV_pos * self.trend_factor
		self.load_error = self.load_pos * self.trend_factor

		# Calculate the upper and lower limits for the current time index
		self.PV_upper_limit = self.PV_pred + self.PV_error
		self.PV_lower_limit = self.PV_pred - self.PV_error
		self.load_upper_limit = self.load_egg + self.load_error
		self.load_lower_limit = self.load_egg - self.load_error

		# Calculate the value for PV_oracle based on the previous time period's predicted value
		for i in range(1, self.Sim_time):
			if self.PV_oracle[i-1] < self.PV_pred[i]:
				self.PV_oracle[i] = get_random_value(lower_bound=self.PV_oracle[i-1], upper_bound=self.PV_upper_limit[i])
			else:
				self.PV_oracle[i] = get_random_value(lower_bound=self.PV_lower_limit[i], upper_bound=self.PV_oracle[i-1])
			
			if self.PV_oracle[i] >= self.PV_max[i]:
				self.PV_oracle[i] = self.PV_max[i]
			elif self.PV_oracle[i] <= self.PV_min[i]:
				self.PV_oracle[i] = self.PV_min[i]

		# Calculate the value for load_oracle based on the previous time period's predicted value
			if self.load_oracle[i-1] < self.load_egg[i]:
				self.load_oracle[i] = get_random_value(lower_bound=self.load_oracle[i-1], upper_bound=self.load_upper_limit[i])
			else:
				self.load_oracle[i] = get_random_value(lower_bound=self.load_lower_limit[i], upper_bound=self.load_oracle[i-1])

			if self.load_oracle[i] >= self.load_max[i]:
				self.load_oracle[i] = self.load_max[i]
			elif self.load_oracle[i] <= self.load_min[i]:
				self.load_oracle[i] = self.load_min[i]

		# print(self.load_oracle[0])
		# print(self.load_pred[1])
		# print(self.load_lower_limit[1])
		# print(self.load_upper_limit[1])
		# print(get_random_value(self.load_oracle[0], self.load_upper_limit[1]))

		self.worst = pd.read_csv('worst.csv', sep=',', names=['PV_worst', 'load_worst'], dtype={'PV_worst': str, 'load_worst': str}, encoding='CP949')[1:]
		self.PV_worst = pd.DataFrame(self.worst, columns=['PV_worst']).to_numpy(dtype=np.float32)
		self.load_worst = pd.DataFrame(self.worst, columns=['load_worst']).to_numpy(dtype=np.float32)

data = data_read()

