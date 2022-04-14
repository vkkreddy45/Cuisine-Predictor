import prediction
import pytest

ip = ["water", "vegetable oil", "wheat", "salt"]
d='data.json'

def test_readdata():
    result = prediction.Readdata(d)
    return result
    assert result is not None

r = test_readdata()

def test_preprocessing():
    result = prediction.preprocess(r, ip)
    return result
    assert result is list and not None

def test_vector():
    result = prediction.vectorization(test_preprocessing())
    return result
    assert result>0 and not None

h,v = test_vector()

def test_model():
    result = prediction.model(h, v, r)
    return result
    assert result is dict
    
def test_toprecpi():
    c = test_model()
    result = prediction.TopnRecipe(h, v, r, 3)
    return result
    assert result>0 and not None
