import cv2
from draw_frame import DrawFrame
from paper_detection import PaperDetection
from hand_detection import HandDetection

def loop(output_video):
	camera = cv2.VideoCapture(0)
	if output_video != None:
		fourcc = cv2.cv.CV_FOURCC('m', 'p', '4', 'v')
		video = cv2.VideoWriter(output_video, fourcc, 20, (711,800))
		record_video = True
	else:
		record_video = False	
	
	df = DrawFrame()
	pd = PaperDetection()
	hd = HandDetection()

	while True:
		# get frame
		(grabbed, frame_in) = camera.read()

		# original frame
		frame_orig = frame_in.copy()

		# shrink frame
		frame = df.resize(frame_in)

		# flipped frame to draw on
		frame_final = df.flip(frame)

		# click p to train paper
		if cv2.waitKey(1) == ord('p') & 0xFF:
			if not pd.trained_paper:
				pd.train_paper(frame)
				pd.set_paper(frame)
				pd.set_ocr_text(frame_orig)
		# click h to train hand
		if cv2.waitKey(1) == ord('h') & 0xFF:
			if pd.trained_paper and not hd.trained_hand:
				hd.train_hand(frame)
		# click q to quit 
		if cv2.waitKey(1) == ord('q') & 0xFF:
		 	break	

		# create frame depending on trained status
		if not pd.trained_paper:
			frame_final = pd.draw_paper_rect(frame_final)
		elif pd.trained_paper and not hd.trained_hand:
			frame_final = hd.draw_hand_rect(frame_final)
		elif pd.trained_paper and hd.trained_hand:
			frame_final = df.draw_final(frame_final, pd, hd)

		# record frame
		if record_video:
			video.write(frame_final)	

		# display frame	
		cv2.imshow('image', frame_final)			 	

	# cleanup
	if record_video:
		video.release()
	camera.release()
	cv2.destroyAllWindows()				
