import cv2
import numpy as np 
from finger_detection import FingerDetection
from paper_detection import PaperDetection
from hand_detection import HandDetection

def main():
	camera = cv2.VideoCapture(1)
	# fourcc = cv2.cv.CV_FOURCC('m', 'p', '4', 'v')
	# video = cv2.VideoWriter('output.mp4', fourcc, 20, (711,800))
	fd = FingerDetection()
	pd = PaperDetection()
	hd = HandDetection()

	while True:
		# get frame
		(grabbed, frame_in) = camera.read()

		# original frame
		frame_orig = frame_in.copy()

		# shrink frame
		frame = fd.resize(frame_in)

		# flipped frame to draw on
		frame_draw = fd.flip(frame)

		# click p to train paper
		if cv2.waitKey(1) == ord('p') & 0xFF:
			if not pd.trained_paper:
				# fd.train_paper(frame)
				# fd.set_paper(frame)
				# fd.set_ocr_text(frame_orig)
				pd.train_paper(frame)
				pd.set_paper(frame)
				pd.set_ocr_text(frame_orig)
		# click h to train hand
		if cv2.waitKey(1) == ord('h') & 0xFF:
			if pd.trained_paper and not hd.trained_hand:
				# fd.train_hand(frame)
				hd.train_hand(frame)
		# click q to quit 
		if cv2.waitKey(1) == ord('q') & 0xFF:
		 	break	

		# create frame depending on trained status
		black = np.zeros(frame_draw.shape, dtype=frame_draw.dtype)
		if not pd.trained_paper:
			# fd.draw_paper_rect(frame_draw)
			frame_draw = pd.draw_paper_rect(frame_draw)
			# frame_draw = np.vstack([black, frame_draw])
		elif pd.trained_paper and not hd.trained_hand:
			# fd.draw_hand_rect(frame_draw)
			frame_draw = hd.draw_hand_rect(frame_draw)
			# frame_draw = np.vstack([black, frame_draw])
		elif pd.trained_paper and hd.trained_hand:
			frame_draw = fd.draw_final(frame_draw, pd, hd)

		# record frame
		# video.write(frame_draw)	

		# display frame	
		cv2.imshow('image', frame_draw)			 	

	# cleanup
	camera.release()
	video.release()
	cv2.destroyAllWindows()				

if __name__ == '__main__':
	main()