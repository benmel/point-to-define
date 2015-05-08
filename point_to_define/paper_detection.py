import cv2
import numpy as np
import os
from tesserwrap import Tesseract
from PIL import Image
import goslate
from collections import deque
import image_analysis

class PaperDetection:
	def __init__(self):		
		cwd = os.path.dirname(os.path.realpath(__file__))
		os.environ['TESSDATA_PREFIX'] = cwd
		self.tr = Tesseract(lang='deu')
		self.gs = goslate.Goslate()
		self.trained_paper = False
		self.paper_row_nw = None
		self.paper_row_se = None
		self.paper_col_nw = None
		self.paper_col_se = None
		self.paper_hist = None
		self.paper = None
		self.words = None
		self.translations = []
		self.pointed_locations = deque(maxlen=20)


	def draw_paper_rect(self, frame):
		rows,cols,_ = frame.shape
		
		self.paper_row_nw = rows/5
		self.paper_row_se = 4*rows/5
		self.paper_col_nw = 2*cols/5
		self.paper_col_se = 3*cols/5
		
		cv2.rectangle(frame,(self.paper_col_nw,self.paper_row_nw),(self.paper_col_se,self.paper_row_se),
									(0,255,0),1)
		black = np.zeros(frame.shape, dtype=frame.dtype)
		frame_final = np.vstack([frame, black])
		return frame_final


	def train_paper(self, frame):
		self.set_paper_hist(frame)
		self.trained_paper = True

	
	def get_paper(self, frame):
		paper_masked = image_analysis.apply_hist_mask(frame, self.paper_hist)
		contours = image_analysis.contours(paper_masked)
		max_contour = image_analysis.max_contour(contours)
		paper = image_analysis.contour_interior(frame, max_contour)
		return paper
	

	def set_paper(self, frame):
		self.paper = self.get_paper(frame)


	def paper_copy(self):
		paper = self.paper.copy()
		return paper		

	
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


	def translate(self, word):
		translated_word = self.gs.translate(word,'en',source_language='de')
		return translated_word


	def update_pointed_locations(self, point):	
		index = self.get_word_index(point)
		if index != None:
			self.pointed_locations.append(index)


	def get_most_common_word(self):		
		index = self.most_common_location()
		if index != None:
			word = self.translations[index].encode('ascii', errors='backslashreplace')
			return word

	
	def most_common_location(self):
		values = set(self.pointed_locations)
		index = None
		maxi = 0
		for i in values:
			num = self.pointed_locations.count(i)
			if num > maxi:
				index = i
		frequency = float(self.pointed_locations.count(index))/float(self.pointed_locations.maxlen)
		if frequency > 0.25:
			return index
		else:
			return None					
			