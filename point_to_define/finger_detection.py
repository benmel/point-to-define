import numpy as np
import cv2
import copy
from collections import deque
from tesserwrap import Tesseract
from PIL import Image
import goslate
import pdb

class FingerDetection:
	def __init__(self):
		self.finished_training = False
		self.finished_paper = False
		self.farthest_points = deque(maxlen=30)
		self.tr = Tesseract(lang='deu')
		self.gs = goslate.Goslate()

	def resize(self, frame):
		rows,cols,_ = frame.shape
		ratio = float(cols)/float(rows)
		new_rows = 400
		new_cols = int(ratio*new_rows)
		self.row_red = float(rows)/float(new_rows)
		self.col_red = float(cols)/float(new_cols)
		frame = cv2.flip(frame, 1)
		frame = cv2.resize(frame, (new_cols, new_rows))
		return frame

	def original_point(self, point):
		x,y = point
		xo = int(x*self.col_red)
		yo = int(y*self.row_red)
		return (xo,yo)

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
			cv2.rectangle(frame,(self.col_nw[i],self.row_nw[i]),(self.col_se[i],self.row_se[i]),
										(0,255,0),1)

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

		disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(11,11))
		cv2.filter2D(dst,-1,disc,dst)
			
		ret,thresh = cv2.threshold(dst,10,255,0)
		thresh = cv2.merge((thresh,thresh,thresh))
		
		cv2.GaussianBlur(dst,(3,3),0,dst)

		res = cv2.bitwise_and(frame,thresh)
		return res

	def draw_paper(self, frame):
		rows,cols,_ = frame.shape
		
		self.row_paper_nw = rows/5
		self.row_paper_se = 4*rows/5
		self.col_paper_nw = 2*cols/5
		self.col_paper_se = 3*cols/5

		cv2.rectangle(frame,(self.col_paper_nw,self.row_paper_nw),(self.col_paper_se,self.row_paper_se),
									(0,255,0),1)

	def set_paper_hist(self, frame):
		hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
		roi = hsv[self.row_paper_nw:self.row_paper_se,self.col_paper_nw:self.col_paper_se]
		self.paperhist = cv2.calcHist([roi],[0, 1], None, [180, 256], [0, 180, 0, 256])																		
		cv2.normalize(self.paperhist,self.paperhist,0,255,cv2.NORM_MINMAX)

	def paper_hist_mask(self, frame):
		hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
		dst = cv2.calcBackProject([hsv],[0,1],self.paperhist,[0,180,0,256],1)

		disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(11,11))
		cv2.filter2D(dst,-1,disc,dst)
			
		ret,thresh = cv2.threshold(dst,10,255,0)
		thresh = cv2.merge((thresh,thresh,thresh))
		
		cv2.GaussianBlur(dst,(3,3),0,dst)
		
		res = cv2.bitwise_and(frame,thresh)
		return res	

	def find_paper_contours(self, frame):
		gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
		ret,thresh = cv2.threshold(gray,0,255,0)
		contours, hierarchy = cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)	
		return contours

	def get_paper_hull(self, contours):
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

	def get_paper(self, contour, frame):
		rect = cv2.minAreaRect(contour)
		box = cv2.cv.BoxPoints(rect)
		box = np.int0(box)

		rows,cols,_ = frame.shape
		paper_mask = np.zeros((rows,cols),dtype=np.float)
		for i in xrange(rows):
			for j in xrange(cols):
				paper_mask.itemset((i,j),cv2.pointPolygonTest(box,(j,i),False))

		paper_mask = np.int0(paper_mask)
		paper_mask[paper_mask < 0] = 0
		paper_mask[paper_mask > 0] = 255
		paper_mask = np.array(paper_mask, dtype=frame.dtype)
		paper_mask = cv2.merge((paper_mask,paper_mask,paper_mask))
		paper = cv2.bitwise_and(frame,paper_mask)
		return (paper, paper_mask)

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

	def get_centroid(self, contour):
		moments = cv2.moments(contour)
		cx = int(moments['m10']/moments['m00'])
		cy = int(moments['m01']/moments['m00'])
		return (cx, cy)

	def get_farthest_point(self, defects, contour, cx, cy):
		if len(defects) > 0:
			s = defects[:,0][:,0]
			x = np.array(contour[s][:,0][:,0],dtype=np.float)
			y = np.array(contour[s][:,0][:,1],dtype=np.float)
						
			xp = cv2.pow(cv2.subtract(x, cx),2)
			yp = cv2.pow(cv2.subtract(y, cy),2)
			dist = cv2.sqrt(cv2.add(xp,yp))

			dist_max_i = np.argmax(dist)
			farthest_defect = s[dist_max_i]
			farthest_point = tuple(contour[farthest_defect][0])
			return farthest_point

	def plot_defects(self, defects, contour, frame):
		if len(defects) > 0:
			for i in range(defects.shape[0]):
				s,e,f,d = defects[i,0]
				start = tuple(contour[s][0])
				end = tuple(contour[e][0])
				far = tuple(contour[f][0])               
				cv2.circle(frame,start,5,[0,0,255],-1)

	def record_farthest_point(self, farthest_point):
		self.farthest_points.append(farthest_point)

	def test_ocr(self, paper):
		gray = cv2.cvtColor(paper,cv2.COLOR_BGR2GRAY)
		ret,thresh = cv2.threshold(gray,100,255,0)
		img = Image.fromarray(thresh)
		self.tr.set_image(img)

	def get_word_at_point(self, point):
		self.tr.get_text()
		words = self.tr.get_words()
		for w in words:
			x_nw,y_nw,x_se,y_sw = w.box
			x,y = point
			if x > x_nw and x < x_se and y > y_nw and y < y_sw:
				return w.value

	def translate(self, word):
		translated_word = self.gs.translate(word,'en',source_language='de')
		return translated_word

def main():
	fd = FingerDetection()

	camera = cv2.VideoCapture(1)
	i = 0
	text = 'Random'

	while True:
		(grabbed, frame) = camera.read()

		frame_pre = copy.deepcopy(frame)
		frame = fd.resize(frame)
		frame_orig = copy.deepcopy(frame)

		if not fd.finished_paper:
			fd.draw_paper(frame)

		if fd.finished_paper and not fd.finished_training:
			fd.draw_rectangle(frame)

		if cv2.waitKey(1) == ord('t'):
			fd.set_skin_hist(frame_orig)
			fd.finished_training = True				

		if cv2.waitKey(1) == ord('p'):
			fd.set_paper_hist(frame_orig)
			paper = fd.paper_hist_mask(frame_orig)
			contours = fd.find_paper_contours(paper)
			hull, contour = fd.get_paper_hull(contours)
			fd.paper, fd.paper_mask = fd.get_paper(contour,frame_orig)

			paper_pre = fd.paper_hist_mask(frame_pre)
			contours_pre = fd.find_paper_contours(paper_pre)
			hull_pre, contour_pre = fd.get_paper_hull(contours_pre)
			paper_f, paper_mask_f = fd.get_paper(contour_pre,frame_pre)
			
			fd.test_ocr(paper_f)
			
			fd.finished_paper = True	

		if cv2.waitKey(1) == ord('q'):
			print fd.tr.get_text()
			break
	
		if fd.finished_training:
			skin = fd.skin_hist_mask(frame_orig)
			contours = fd.find_contours(skin)
			hull, contour = fd.get_hull(contours)
			defects = fd.get_defects(contour)
			cx, cy = fd.get_centroid(contour)
			farthest = fd.get_farthest_point(defects, contour, cx, cy)

			fd.record_farthest_point(farthest)

			i = i + 1

			if i > 30:
				point = fd.original_point(farthest)
				word = fd.get_word_at_point(point)
				if word != None:
					text = fd.translate(word)
				else:
					text = ''
				i = 0

			skin_paper = copy.deepcopy(fd.paper)
			skin_paper = cv2.flip(skin_paper, 1)
			cv2.circle(skin_paper,farthest,7,[0,0,255],-1)

			contours_img = np.zeros(skin.shape,dtype=skin.dtype)
			hull_img = np.zeros(skin.shape,dtype=skin.dtype)
			defects_img = np.zeros(skin.shape,dtype=skin.dtype)

			cv2.drawContours(contours_img,contours,-1,(0,255,0),3)
			cv2.drawContours(hull_img,[hull],0,(0,0,255),2)
			fd.plot_defects(defects, contour, defects_img)
			cv2.circle(defects_img,(cx,cy),5,[255,0,0],-1)
			cv2.circle(defects_img,farthest,5,[0,255,0],-1)

			rows,cols,_ = skin_paper.shape
			cv2.putText(skin_paper, text, (rows/2-50,50), cv2.FONT_HERSHEY_PLAIN, 4, [255,255,255], 4)

			cv2.imshow('image', np.hstack([np.vstack([skin,contours_img]),
																		 np.vstack([skin_paper,defects_img])]))
		else:
			black = np.zeros(frame.shape,dtype=frame.dtype)
			if fd.finished_paper:
				cv2.imshow('image', np.hstack([np.vstack([frame,fd.paper]),
																			 np.vstack([black,black])]))
			else:
				cv2.imshow('image', np.hstack([np.vstack([frame,black]),
																			 np.vstack([black,black])]))	

	camera.release()
	cv2.destroyAllWindows()				

if __name__ == '__main__':
	main()	  