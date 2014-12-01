import numpy as np
import cv2
import copy
import pdb

class FingerDetection:
	def __init__(self):
		self.finished_training = False

	def resize(self, frame):
		rows,cols,_ = frame.shape
		ratio = float(cols)/float(rows)
		new_rows = 400
		new_cols = int(ratio*new_rows)
		frame = cv2.flip(frame, 1)
		frame = cv2.resize(frame, (new_cols, new_rows))
		return frame	

	def draw_rectangle(self, frame):
		"""draw rectangles on frame"""
		rows,cols,_ = frame.shape
		self.row_nw = np.array([6*rows/20,6*rows/20,6*rows/20,
														10*rows/20,10*rows/20,10*rows/20,
														14*rows/20,14*rows/20,14*rows/20])

		self.col_nw = np.array([9*cols/20,10*cols/20,11*cols/20,
														9*cols/20,10*cols/20,11*cols/20,
														9*cols/20,10*cols/20,11*cols/20])

		self.row_se = np.zeros(self.row_nw.size, dtype=np.int)
		for i in range(self.row_nw.size):
			self.row_se[i] = self.row_nw[i] + 10

		self.col_se = np.zeros(self.col_nw.size, dtype=np.int)
		for i in range(self.col_nw.size):
			self.col_se[i] = self.col_nw[i] + 10	

		for i in range(self.row_nw.size):
			cv2.rectangle(frame,(self.col_nw[i],self.row_nw[i]),(self.col_se[i],self.row_se[i]),(0,255,0),1)

	def set_skin_hist(self, frame):
		#TODO use constants, only do HSV for ROI
		hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
		roi = np.zeros([90,10,3], dtype=hsv.dtype)
		
		for i in range(self.row_nw.size):
			roi[i*10:i*10+10,0:10] = hsv[self.row_nw[i]:self.row_nw[i]+10,
																	 self.col_nw[i]:self.col_nw[i]+10]

		self.roihist = cv2.calcHist([roi],[0, 1], None, [180, 256], [0, 180, 0, 256])																		
		cv2.normalize(self.roihist,self.roihist,0,255,cv2.NORM_MINMAX)																		

	def skin_hist_mask(self, frame):
		hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
		dst = cv2.calcBackProject([hsv],[0,1],self.roihist,[0,180,0,256],1)
		
		# disc2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
		# cv2.morphologyEx(dst,cv2.MORPH_CLOSE,disc2,dst,iterations=2)

		disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(11,11))
		cv2.filter2D(dst,-1,disc,dst)
			
		ret,thresh = cv2.threshold(dst,150,255,0)
		thresh = cv2.merge((thresh,thresh,thresh))
		
		cv2.GaussianBlur(dst,(3,3),0,dst)

		res = cv2.bitwise_and(frame,thresh)
		return res

	def find_contours(self, frame):
		gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
		ret,thresh = cv2.threshold(gray,0,255,0)
		contours, hierarchy = cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)	
		return contours

	def get_hull(self, contours):
		max_i = 0
		max_area = 0
		
		for i in range(len(contours)):
			cnt = contours[i]
			area = cv2.contourArea(cnt)
			if area > max_area:
				max_area = area
				max_i = i

		contour = contours[max_i]
		hull = cv2.convexHull(contour)
		return (hull, contour)

	def get_defects(self, contour):
		hull = cv2.convexHull(contour, returnPoints = False)
		defects = cv2.convexityDefects(contour, hull)	
		return defects

	def plot_defects(self, defects, contour, frame):
		for i in range(defects.shape[0]):
			s,e,f,d = defects[i,0]
			start = tuple(contour[s][0])
			end = tuple(contour[e][0])
			far = tuple(contour[f][0])
			cv2.line(frame,start,end,[0,255,0],2)                
			cv2.circle(frame,far,5,[0,0,255],-1)	

def main():
	fd = FingerDetection()

	camera = cv2.VideoCapture(0)
	while True:
		(grabbed, frame) = camera.read()

		frame = fd.resize(frame)
		frame_orig = copy.deepcopy(frame)

		if not fd.finished_training:
			fd.draw_rectangle(frame)

		if cv2.waitKey(1) == ord('t'):
			fd.set_skin_hist(frame_orig)
			fd.finished_training = True	

		if cv2.waitKey(1) == ord('q'):
			break
	
		if fd.finished_training:
			skin = fd.skin_hist_mask(frame)
			contours = fd.find_contours(skin)
			hull, contour = fd.get_hull(contours)
			defects = fd.get_defects(contour)

			contours_img = np.zeros(skin.shape,dtype=skin.dtype)
			hull_img = np.zeros(skin.shape,dtype=skin.dtype)
			defects_img = np.zeros(skin.shape,dtype=skin.dtype)

			cv2.drawContours(contours_img,contours,-1,(0,255,0),3)
			cv2.drawContours(hull_img,[hull],0,(0,0,255),2)
			fd.plot_defects(defects, contour, defects_img)
			
			cv2.imshow('image', np.hstack([np.vstack([skin,contours_img]),
																		 np.vstack([hull_img,defects_img])]))
		else:
			cv2.imshow('image', frame)


	camera.release()
	cv2.destroyAllWindows()				

if __name__ == '__main__':
	main()	  