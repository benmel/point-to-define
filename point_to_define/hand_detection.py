import cv2
import numpy as np
import image_analysis

class HandDetection:
	def __init__(self):		
		self.trained_hand = False
		self.hand_row_nw = None
		self.hand_row_se = None
		self.hand_col_nw = None
		self.hand_col_se = None
		self.hand_hist = None


	def draw_hand_rect(self, frame):
		rows,cols,_ = frame.shape
		
		self.hand_row_nw = np.array([6*rows/20,6*rows/20,6*rows/20,
														10*rows/20,10*rows/20,10*rows/20,
														14*rows/20,14*rows/20,14*rows/20])

		self.hand_col_nw = np.array([9*cols/20,10*cols/20,11*cols/20,
														9*cols/20,10*cols/20,11*cols/20,
														9*cols/20,10*cols/20,11*cols/20])

		self.hand_row_se = self.hand_row_nw + 10
		self.hand_col_se = self.hand_col_nw + 10

		size = self.hand_row_nw.size
		for i in xrange(size):
			cv2.rectangle(frame,(self.hand_col_nw[i],self.hand_row_nw[i]),(self.hand_col_se[i],self.hand_row_se[i]),
										(0,255,0),1)
		black = np.zeros(frame.shape, dtype=frame.dtype)
		frame_final = np.vstack([black, frame])
		return frame_final	


	def train_hand(self, frame):
		self.set_hand_hist(frame)
		self.trained_hand = True


	def set_hand_hist(self, frame):
		#TODO use constants, only do HSV for ROI
		hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
		roi = np.zeros([90,10,3], dtype=hsv.dtype)
		
		size = self.hand_row_nw.size
		for i in xrange(size):
			roi[i*10:i*10+10,0:10] = hsv[self.hand_row_nw[i]:self.hand_row_nw[i]+10,
															 self.hand_col_nw[i]:self.hand_col_nw[i]+10]

		self.hand_hist = cv2.calcHist([roi],[0, 1], None, [180, 256], [0, 180, 0, 256])																		
		cv2.normalize(self.hand_hist, self.hand_hist, 0, 255, cv2.NORM_MINMAX)
