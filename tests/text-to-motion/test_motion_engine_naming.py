import pytest

from motion_engine import build_base_name


@pytest.mark.unit
def test_base_name_is_the_job_id_not_a_timestamp():
    """mcp_server.py cũ đặt tên motion_<timestamp>_<random> nên mỗi lần sinh ra key khác
    và cache KHÔNG BAO GIỜ trúng. Tên phải là job_id để URL biết trước được."""
    assert build_base_name("a3f9c21") == "a3f9c21"
    assert build_base_name("a3f9c21") == build_base_name("a3f9c21")
