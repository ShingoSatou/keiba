from scripts import extract_jvlink, extract_rt_jvlink


class _DummyRTJV:
    def JVRTOpen(self, dataspec, key):
        return -1


class _DummyJV:
    def JVOpen(self, dataspec, from_date, option, readcount, downloadcount, timestamp):
        return (-1, 0, 0, "")


def test_jv_rt_open_logs_minus_one_as_no_data(capsys):
    rc = extract_rt_jvlink.jv_rt_open_with_logging(_DummyRTJV(), "0B11", "20260208")
    captured = capsys.readouterr()

    assert rc == -1
    assert "該当データ無し" in captured.out


def test_jv_open_logs_minus_one_as_no_data(capsys):
    result = extract_jvlink.jv_open_with_logging(_DummyJV(), "RACE", "20260101000000", 1)
    captured = capsys.readouterr()

    assert result[0] == -1
    assert "該当データ無し" in captured.out
