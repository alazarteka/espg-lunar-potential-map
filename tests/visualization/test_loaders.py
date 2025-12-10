
import pytest
from datetime import date
from src.visualization import loaders

def test_date_range_valid():
    start = date(2023, 1, 1)
    end = date(2023, 1, 3)
    days = loaders._date_range(start, end)
    assert len(days) == 3
    assert days[0] == start
    assert days[-1] == end

def test_date_range_single_day():
    start = date(2023, 1, 1)
    days = loaders._date_range(start, start)
    assert len(days) == 1
    assert days[0] == start

def test_date_range_invalid():
    start = date(2023, 1, 2)
    end = date(2023, 1, 1)
    with pytest.raises(ValueError):
        loaders._date_range(start, end)
