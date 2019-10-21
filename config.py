## @package client
#
# Configurations fo this project
import os


HOST="0.0.0.0"
PORT=5000

## Server address
ADDR = 'http://' + HOST + ':' + str(PORT)

## path to the config image
URL_IMG = os.path.dirname(os.path.abspath(__file__)) + '/images/book.jpg'
