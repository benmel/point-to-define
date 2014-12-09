import cv2
import numpy as np
import image_analysis

class DrawFrame:
	def __init__(self):				
		self.row_ratio = None
		self.col_ratio = None
		self.text = ''
		

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
		paper_hand = paper_detection.paper_copy()
		self.plot_word_boxes(paper_hand, paper_detection.words)

		contours = image_analysis.contours(hand_masked)
		if contours is not None and len(contours) > 0:
			max_contour = image_analysis.max_contour(contours)
			hull = image_analysis.hull(max_contour)
			centroid = image_analysis.centroid(max_contour)
			defects = image_analysis.defects(max_contour)

			if centroid is not None and defects is not None and len(defects) > 0:	
				farthest_point = image_analysis.farthest_point(defects, max_contour, centroid)

				if farthest_point is not None:
					self.plot_farthest_point(frame, farthest_point)
					self.plot_hull(frame, hull)
					# self.plot_contours(frame, contours)
					# self.plot_defects(frame, defects, max_contour)
					# self.plot_centroid(frame, (cx,cy))

					self.plot_farthest_point(paper_hand, farthest_point)
					point = self.original_point(farthest_point)
					paper_detection.update_pointed_locations(point)
					self.text = paper_detection.get_most_common_word()
		
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
