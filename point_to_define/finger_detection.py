import cv2
import numpy as np
import image_analysis
from collections import deque

class FingerDetection:
	def __init__(self):				
		self.row_ratio = None
		self.col_ratio = None
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


	def draw_final(self, frame, paper_detection, hand_detection):
		hand_masked = image_analysis.apply_hist_mask(frame, hand_detection.hand_hist)

		contours = image_analysis.contours(hand_masked)
		max_contour = image_analysis.max_contour(contours)

		hull = image_analysis.hull(max_contour)
		defects = image_analysis.defects(max_contour)
		
		cx, cy = image_analysis.centroid(max_contour)
		farthest_point = self.farthest_point(defects, max_contour, cx, cy)

		# self.plot_contours(frame, contours)
		self.plot_centroid(frame, (cx,cy))
		self.plot_farthest_point(frame, farthest_point)
		self.plot_hull(frame, hull)
		# self.plot_defects(frame, defects, max_contour)

		paper_hand = paper_detection.paper.copy()
		self.plot_farthest_point(paper_hand, farthest_point)
		self.plot_word_boxes(paper_hand, paper_detection.words)

		point = self.original_point(farthest_point)
		index = paper_detection.get_word_index(point)
		if index != None:
			self.locations.append(index)
		top_index = self.most_common(self.locations)

		if top_index != None:
			word = paper_detection.translations[top_index]
			self.text = paper_detection.translate(word).encode('ascii', errors='backslashreplace')	

		self.plot_text(paper_hand, self.text)		

		frame_final = np.vstack([paper_hand, frame])
		return frame_final


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
