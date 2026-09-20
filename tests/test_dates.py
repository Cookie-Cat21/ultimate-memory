from ultimate_memory.dates import parse_loose_date, resolve_relative_dates


def test_resolve_this_and_next_month():
    anchor = parse_loose_date("15 February 2023")
    assert anchor is not None
    assert "February 2023" in resolve_relative_dates("this month", anchor)
    assert "March 2023" in resolve_relative_dates("next month", anchor)


def test_resolve_last_named_weekday():
    anchor = parse_loose_date("20 July 2023")
    assert anchor is not None
    result = resolve_relative_dates("last Saturday", anchor)
    assert "15 July 2023" in result
