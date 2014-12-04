import cv2
from finger_detection import FingerDetection
import pdb

def main():
	fd = FingerDetection()

	camera = cv2.VideoCapture(1)
	i = 0
	# text = 'Random'

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
			if not fd.trained_paper:
				fd.train_paper(frame)
				fd.set_paper(frame)
				fd.set_ocr_text(frame_orig)
		# click h to train hand
		if cv2.waitKey(1) == ord('h') & 0xFF:
			if fd.trained_paper and not fd.trained_hand:
				fd.train_hand(frame)
		# click q to quit 
		if cv2.waitKey(1) == ord('q') & 0xFF:
		 	break	

		# create frame depending on trained status
		if not fd.trained_paper:
			fd.draw_paper_rect(frame_draw)
		elif fd.trained_paper and not fd.trained_hand:
			fd.draw_hand_rect(frame_draw)
		elif fd.trained_paper and fd.trained_hand:
			frame_draw = fd.draw_final(frame_draw)

		# display frame	
		cv2.imshow('image', frame_draw)			 	

	# cleanup
	camera.release()
	cv2.destroyAllWindows()				

if __name__ == '__main__':
	main()