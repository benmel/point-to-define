import numpy as np
import cv2

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
		self.x_nw = [9*cols/20,10*cols/20,11*cols/20,
						9*cols/20,10*cols/20,11*cols/20,
						9*cols/20,10*cols/20,11*cols/20]
		self.y_nw = [6*rows/20,6*rows/20,6*rows/20,
						10*rows/20,10*rows/20,10*rows/20,
						14*rows/20,14*rows/20,14*rows/20]
		
		self.x_se = []
		for x in self.x_nw:
			self.x_se.append(x+10)

		self.y_se = []
		for y in self.y_nw:
			self.y_se.append(y+10)	

		for i in range(len(self.x_nw)):
			cv2.rectangle(frame,(self.x_nw[i],self.y_nw[i]),(self.x_se[i],self.y_se[i]),(0,255,0),1)
	
	def get_rectangle_pixels(self):
		"""select pixels in rectangles"""


def main():
	fd = FingerDetection()

	camera = cv2.VideoCapture(0)
	while True:
		(grabbed, frame) = camera.read()

		frame = fd.resize(frame)
		if not fd.finished_training:
			fd.draw_rectangle(frame)

		if cv2.waitKey(1) == ord('p'):
			# fd.get_rectangle_pixels()
			fd.finished_training = True

		if cv2.waitKey(1) == ord('q'):
			break
	
		cv2.imshow('image', frame)	

	camera.release()
	cv2.destroyAllWindows()				

if __name__ == '__main__':
	main()	  