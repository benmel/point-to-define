import pytest
from point_to_define.hello import *

def test():
	h = hello()
	assert h.text == "hello test"	