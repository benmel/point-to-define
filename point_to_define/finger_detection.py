import cv2
import numpy as np
from tesserwrap import Tesseract
from PIL import Image
import goslate
import image_analysis
from collections import deque

class FingerDetection:
	def __init__(self):		
		self.tr = Tesseract(lang='deu')
		self.gs = goslate.Goslate()
		
		self.trained_paper = False
		self.trained_hand = False
		
		self.row_ratio = None
		self.col_ratio = None

		self.paper_row_nw = None
		self.paper_row_se = None
		self.paper_col_nw = None
		self.paper_col_se = None

		self.hand_row_nw = None
		self.hand_row_se = None
		self.hand_col_nw = None
		self.hand_col_se = None

		self.paper_hist = None
		self.hand_hist = None

		self.paper = None

		self.words = None
		self.translations = []

		self.text = ''

		self.locations = deque(maxlen=20)


	def resize(self, frame):
		rows,cols,_ = frame.shape
		
		ratio = float(cols)/float(rows)
		new_rows = 400
		new_cols = int(ratio*new_rows)
		
		self.row_ratio = float(rows)/float(new_rows)
		self.col_ratio = float(cols)/float(new_cols)
		
		resized = cv2.resize(frame, (new_cols, new_rows))	
		return resized


	def flip(self, frame):
		flipped = cv2.flip(frame, 1)
		return flipped	


	def draw_paper_rect(self, frame):
		rows,cols,_ = frame.shape
		
		self.paper_row_nw = rows/5
		self.paper_row_se = 4*rows/5
		self.paper_col_nw = 2*cols/5
		self.paper_col_se = 3*cols/5
		
		cv2.rectangle(frame,(self.paper_col_nw,self.paper_row_nw),(self.paper_col_se,self.paper_row_se),
									(0,255,0),1)


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


	def draw_final(self, frame):
		hand_masked = image_analysis.apply_hist_mask(frame, self.hand_hist)

		contours = image_analysis.contours(hand_masked)
		max_contour = image_analysis.max_contour(contours)

		hull = image_analysis.hull(max_contour)
		defects = image_analysis.defects(max_contour)
		
		cx, cy = image_analysis.centroid(max_contour)
		farthest_point = self.farthest_point(defects, max_contour, cx, cy)

		self.plot_contours(frame, contours)
		self.plot_centroid(frame, (cx,cy))
		self.plot_farthest_point(frame, farthest_point)
		self.plot_hull(frame, hull)
		# self.plot_defects(frame, defects, max_contour)

		paper_hand = self.paper.copy()
		self.plot_farthest_point(paper_hand, farthest_point)
		self.plot_word_boxes(paper_hand, self.words)

		point = self.original_point(farthest_point)
		index = self.get_word_index(point)
		if index != None:
			self.locations.append(index)
		top_index = self.most_common(self.locations)

		if top_index != None:
			word = self.translations[top_index]
			self.text = self.translate(word).encode('ascii', errors='backslashreplace')	

		self.plot_text(paper_hand, self.text)		

		frame_final = np.vstack([paper_hand, frame])
		return frame_final


	def train_paper(self, frame):
		self.set_paper_hist(frame)
		self.trained_paper = True

	
	def train_hand(self, frame):
		self.set_hand_hist(frame)
		self.trained_hand = True


	def get_paper(self, frame):
		paper_masked = image_analysis.apply_hist_mask(frame, self.paper_hist)
		contours = image_analysis.contours(paper_masked)
		max_contour = image_analysis.max_contour(contours)
		paper = image_analysis.contour_interior(frame, max_contour)
		return paper
	

	def set_paper(self, frame):
		self.paper = self.get_paper(frame)	

	
	# TODO move OCR to separate class
	def set_ocr_text(self, frame):
		paper = self.get_paper(frame)
		thresh = image_analysis.gray_threshold(paper, 100)
		paper_img = Image.fromarray(thresh)
		self.tr.set_image(paper_img)
		self.tr.get_text()
		self.words = self.tr.get_words()
		for w in self.words:
			translation = self.translate(w.value)
			self.translations.append(translation)


	def set_paper_hist(self, frame):
		hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
		roi = hsv[self.paper_row_nw:self.paper_row_se, self.paper_col_nw:self.paper_col_se]
		self.paper_hist = cv2.calcHist([roi], [0, 1], None, [180, 256], [0, 180, 0, 256])																		
		cv2.normalize(self.paper_hist, self.paper_hist, 0, 255, cv2.NORM_MINMAX)


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


	def original_point(self, point):
		x,y = point
		xo = int(x*self.col_ratio)
		yo = int(y*self.row_ratio)
		return (xo,yo)


	def new_point(self, point):
		(x,y) = point
		xn = int(x/self.col_ratio)
		yn = int(y/self.row_ratio)
		return (xn,yn)

	# TODO potentially move
	def farthest_point(self, defects, contour, cx, cy):
		if len(defects) > 0:
			s = defects[:,0][:,0]
			
			x = np.array(contour[s][:,0][:,0], dtype=np.float)
			y = np.array(contour[s][:,0][:,1], dtype=np.float)
						
			xp = cv2.pow(cv2.subtract(x, cx), 2)
			yp = cv2.pow(cv2.subtract(y, cy), 2)
			dist = cv2.sqrt(cv2.add(xp, yp))

			dist_max_i = np.argmax(dist)
			farthest_defect = s[dist_max_i]
			farthest_point = tuple(contour[farthest_defect][0])
			return farthest_point

	def plot_defects(self, frame, defects, contour):
		if len(defects) > 0:
			for i in xrange(defects.shape[0]):
				s,e,f,d = defects[i,0]
				start = tuple(contour[s][0])
				end = tuple(contour[e][0])
				far = tuple(contour[f][0])               
				cv2.circle(frame, start, 5, [255,0,255], -1)


	def plot_farthest_point(self, frame, point):
		cv2.circle(frame, point, 5, [0,0,255], -1)			

	
	def plot_centroid(self, frame, point):
		cv2.circle(frame, point, 5, [255,0,0], -1)

	
	def plot_hull(self, frame, hull):
		cv2.drawContours(frame, [hull], 0, (255,0,0), 2)	


	def plot_contours(self, frame, contours):
		cv2.drawContours(frame, contours, -1, (0,255,0), 3)				


	def plot_text(self, frame, text):
		cv2.putText(frame, text, (50,50), cv2.FONT_HERSHEY_PLAIN, 3, [255,255,255], 4)	


	def plot_word_boxes(self, frame, words):
		rows,cols,_ = frame.shape
		for w in words:
			x_nw,y_nw,x_se,y_se = w.box
			x_nw,y_nw = self.new_point((x_nw,y_nw))
			x_se,y_se = self.new_point((x_se,y_se))
			x_nw = x_nw
			x_se = x_se

			cv2.rectangle(frame,(x_nw,y_nw),(x_se,y_se),
									(0,255,255),1)

	
	def get_word_at_point(self, point):
		for i, w in enumerate(self.words):
			x_nw,y_nw,x_se,y_sw = w.box
			x,y = point
			if x > x_nw and x < x_se and y > y_nw and y < y_sw:
				return self.translations[i]


	def get_word_index(self, point):
		for i, w in enumerate(self.words):
			x_nw,y_nw,x_se,y_sw = w.box
			x,y = point
			if x > x_nw and x < x_se and y > y_nw and y < y_sw:
				return i	


	def most_common(self, listi):
		values = set(listi)
		index = None
		maxi = 0
		for i in values:
			num = listi.count(i)
			if num > maxi:
				index = i
		frequency = float(listi.count(index))/float(listi.maxlen)
		if frequency > 0.25:
			return index
		else:
			return None								


	def translate(self, word):
		translated_word = self.gs.translate(word,'en',source_language='de')
		return translated_word
