import sys
import getopt
from point_to_define import display

def main():
	def usage():
		print "python main.py [-v <output_video>]"

	output_video = None

	"""Process command line inputs"""
	try:
		opts, args = getopt.getopt(sys.argv[1:], "hv:", ["output_video="])
	except getopt.GetoptError:
		usage()
		sys.exit(2)
	for opt, arg in opts:
		if opt in ("-h", "--help"):
			usage()
			sys.exit()	
		elif opt in ("-v", "--output_video"):
			output_video = arg
					
	display.loop(output_video)

if __name__ == '__main__':
	main()	