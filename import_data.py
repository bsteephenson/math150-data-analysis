import csv
import numpy as np
from numpy import genfromtxt

def import_data():
	
	with open('data.csv', 'r') as file:
		data_points = np.genfromtxt(file, delimiter = ",", dtype = float, skip_header=1, usecols = range(2, 32))
	with open('data.csv', 'r') as file:
		classes = np.genfromtxt(file, delimiter = ",", dtype = None, skip_header=1, usecols = [1])

	def class_to_int(c):
		if c == "M":
			return 1
		else:
			return -1
	classes = np.vectorize(class_to_int)(classes)

	return classes, data_points