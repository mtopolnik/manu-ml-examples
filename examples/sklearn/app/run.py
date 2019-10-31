import logging, sys
from infer import app



def main():
	logger.warning("NOT Running in debug mode.")
	app.run(debug=DEBUG, host='0.0.0.0', port=PORT)

if __name__ == "__main__":
	main()





