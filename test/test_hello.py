import pytest
from project.hello import *

def test():
	h = hello()
	assert h.text == "hello test"	