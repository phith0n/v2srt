import json
import logging
import os
import time
from pathlib import Path

import pytest
from fakes import FakeClient

from v2srt import cli
from v2srt.config import LOG_OFF, Settings, load_config, user_config_dir
from v2srt.errors import ConfigError
from v2srt.logs import RunLog, log_files, prune, resolve_log_dir
from v2srt.subtitle import Cue
from v2srt.translator import GeminiTranslator


@pytest.fixture
def home(tmp_path, monkeypatch):
    """Isolated working directory, per-user config directory and log directory."""
    cwd = tmp_path / "work"
    cwd.mkdir()
    monkeypatch.chdir(cwd)
    monkeypatch.setenv("APPDATA", str(tmp_path / "appdata"))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    logs = tmp_path / "logs" / "v2srt" / "Logs"
    import v2srt.logs

    monkeypatch.setattr(cli, "default_log_dir", lambda: logs)
    monkeypatch.setattr(v2srt.logs, "default_log_dir", lambda: logs)
    monkeypatch.setattr(cli.tempfile, "gettempdir", lambda: str(tmp_path / "tmp"))
    for name in list(os.environ):
        if name.startswith("V2SRT_") or name in ("GEMINI_API_KEY", "GOOGLE_API_KEY"):
            monkeypatch.delenv(name)
    return {"cwd": cwd, "config": user_config_dir(), "logs": logs, "tmp": tmp_path / "tmp"}


def write_env(path: Path, **values) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f"{k}={v}\n" for k, v in values.items()), encoding="utf-8")
    return path


# ------------------------------------------------------------------ configuration


def test_only_one_config_file_is_used(home):
    write_env(home["config"] / ".env", V2SRT_WHISPER_MODEL="global.bin", V2SRT_GEMINI_API_KEY="global-key")
    write_env(home["cwd"] / ".env", V2SRT_WHISPER_MODEL="local.bin")

    config = load_config(environ={})
    assert config.env_file == (home["cwd"] / ".env").resolve()
    assert config.settings.whisper_model == "local.bin"
    assert config.settings.gemini_api_key is None  # the per-user file is not merged in

    (home["cwd"] / ".env").unlink()
    config = load_config(environ={})
    assert config.env_file == (home["config"] / ".env").resolve()
    assert config.settings.gemini_api_key == "global-key"


def test_env_file_option_wins(home, tmp_path):
    write_env(home["cwd"] / ".env", V2SRT_WHISPER_MODEL="local.bin")
    other = write_env(tmp_path / "other.env", V2SRT_WHISPER_MODEL="other.bin")
    assert load_config(other, environ={}).settings.whisper_model == "other.bin"
    with pytest.raises(ConfigError):
        load_config(tmp_path / "missing.env", environ={})


def test_precedence_and_sources(home):
    write_env(home["cwd"] / ".env", V2SRT_WHISPER_MODEL="file.bin", V2SRT_LANGUAGE="ja", V2SRT_BATCH_SIZE="20")
    environ = {"V2SRT_LANGUAGE": "en", "V2SRT_BATCH_SIZE": "", "GEMINI_API_KEY": "k"}
    config = load_config(environ=environ, overrides={"target_language": "en", "whisper_jobs": None})
    s = config.settings
    assert (s.whisper_model, s.language, s.batch_size, s.target_language) == ("file.bin", "en", 20, "en")
    assert config.sources["whisper_model"] == "config file (V2SRT_WHISPER_MODEL)"
    assert config.sources["language"] == "environment (V2SRT_LANGUAGE)"
    assert config.sources["batch_size"] == "config file (V2SRT_BATCH_SIZE)"  # empty env var counts as unset
    assert config.sources["gemini_api_key"] == "environment (GEMINI_API_KEY)"
    assert config.sources["target_language"] == "command line"
    assert "whisper_jobs" not in config.sources


def test_config_file_does_not_leak_into_environment(home):
    write_env(home["cwd"] / ".env", HTTPS_PROXY="http://example:1", V2SRT_LANGUAGE="ja")
    load_config()
    assert "HTTPS_PROXY" not in os.environ or os.environ["HTTPS_PROXY"] != "http://example:1"


def test_log_dir_setting():
    s, _ = load_config(environ={"V2SRT_LOG_DIR": "off"}).settings, None
    assert s.log_dir == LOG_OFF and resolve_log_dir(s.log_dir) is None
    assert resolve_log_dir("/some/where") == Path("/some/where")
    assert resolve_log_dir(None) is not None


# ------------------------------------------------------------------ logs


def test_run_log_writes_debug_and_prunes(tmp_path):
    directory = tmp_path / "logs"
    directory.mkdir()
    for i in range(5):
        old = directory / f"20200101-00000{i}_old.log"
        old.write_text("x")
        os.utime(old, (time.time() - 1000 + i, time.time() - 1000 + i))
    (directory / "unrelated.txt").write_text("keep me")

    run = RunLog.start(directory, "My Video: part 1.mkv", record_gemini=True)
    assert run is not None
    logging.getLogger("v2srt.test").setLevel(logging.DEBUG)
    logging.getLogger("v2srt.test").debug("hello %s", "debug")
    run.close()

    assert run.path.name.endswith("_My-Video-part-1-mkv.log")
    assert "hello debug" in run.path.read_text(encoding="utf-8")
    prune(directory, keep=3)
    assert len(log_files(directory)) == 3
    assert run.path.exists() and (directory / "unrelated.txt").exists()


def test_gemini_requests_are_recorded(tmp_path):
    run = RunLog.start(tmp_path, "clip", record_gemini=True)
    t = GeminiTranslator(Settings(gemini_api_key="k"), client=FakeClient())
    t.recorder = run.gemini
    t.translate([Cue(0, 1000, "Hello")], source_language="en")
    run.close()
    entries = [json.loads(line) for line in run.gemini.path.read_text(encoding="utf-8").splitlines()]
    assert len(entries) == 1
    entry = entries[0]
    assert entry["outcome"] == "ok" and entry["cues"] == 1
    assert '"Hello"' in entry["prompt"] and "T:Hello" in entry["answer"]
    assert entry["tokens"] == {"input": 10, "output": 5, "thinking": 1}
    assert entry["system_instruction"].startswith("You are an expert subtitle translator")


# ------------------------------------------------------------------ clean


def populate(home):
    write_env(home["config"] / ".env", V2SRT_LANGUAGE="ja")
    home["logs"].mkdir(parents=True)
    (home["logs"] / "20260101-120000_clip.log").write_text("log")
    (home["logs"] / "20260101-120000_clip.gemini.jsonl").write_text("{}")
    (home["tmp"] / "v2srt-abc123").mkdir(parents=True)
    (home["tmp"] / "v2srt-abc123" / "audio.wav").write_bytes(b"0" * 100)
    (home["tmp"] / "other-tool").mkdir()
    write_env(home["cwd"] / ".env", V2SRT_LANGUAGE="en")


def test_clean_removes_everything_but_local_env(home, capsys):
    populate(home)
    groups = cli.find_cleanable(config=True, logs=True, temp=True)
    touched = [p for g in groups for p in g.paths] + [d for g in groups for d in g.remove_if_empty]
    assert all(str(p).startswith(str(home["cwd"].parent)) for p in touched), touched
    with pytest.raises(SystemExit) as exit_info:
        cli.main(["clean", "--yes"])
    assert exit_info.value.code == 0
    assert not (home["config"] / ".env").exists() and not home["config"].exists()
    assert not home["logs"].exists()
    assert not (home["tmp"] / "v2srt-abc123").exists()
    assert (home["tmp"] / "other-tool").exists()
    assert (home["cwd"] / ".env").read_text(encoding="utf-8") == "V2SRT_LANGUAGE=en\n"


def test_clean_custom_log_dir(home, tmp_path):
    custom = tmp_path / "shared-logs"
    custom.mkdir()
    (custom / "20260101-120000_clip.log").write_text("log")
    (custom / "notes.txt").write_text("not ours")
    write_env(home["cwd"] / ".env", V2SRT_LOG_DIR=str(custom))
    with pytest.raises(SystemExit):
        cli.main(["clean", "--logs", "--yes"])
    assert not (custom / "20260101-120000_clip.log").exists()
    assert (custom / "notes.txt").exists()


def test_clean_selected_category_only(home):
    populate(home)
    with pytest.raises(SystemExit):
        cli.main(["clean", "--logs", "--yes"])
    assert not home["logs"].exists()
    assert (home["config"] / ".env").exists() and (home["tmp"] / "v2srt-abc123").exists()


def test_clean_never_deletes_env_of_current_directory(home, monkeypatch):
    populate(home)
    monkeypatch.chdir(home["config"])  # the per-user .env IS the current directory's .env here
    groups = cli.find_cleanable(config=True, logs=True, temp=True)
    assert all(home["config"] / ".env" not in g.paths for g in groups)


def test_clean_asks_before_deleting(home, monkeypatch, capsys):
    populate(home)
    monkeypatch.setattr(cli.sys.stdin, "isatty", lambda: False, raising=False)
    with pytest.raises(SystemExit) as exit_info:
        cli.main(["clean"])
    assert exit_info.value.code == 1
    assert "--yes" in capsys.readouterr().out
    assert (home["config"] / ".env").exists()

    monkeypatch.setattr(cli.sys.stdin, "isatty", lambda: True, raising=False)
    monkeypatch.setattr(cli.Confirm, "ask", lambda *a, **k: False)
    with pytest.raises(SystemExit):
        cli.main(["clean"])
    assert (home["config"] / ".env").exists() and home["logs"].exists()


def test_clean_with_nothing_to_do(home, capsys):
    with pytest.raises(SystemExit) as exit_info:
        cli.main(["clean"])
    assert exit_info.value.code == 0
    assert "Nothing to clean" in capsys.readouterr().out


def test_clean_without_input_deletes_nothing(home, monkeypatch, capsys):
    populate(home)
    monkeypatch.setattr(cli.sys.stdin, "isatty", lambda: True, raising=False)

    def no_input(*args, **kwargs):
        raise EOFError

    monkeypatch.setattr(cli.Confirm, "ask", no_input)
    with pytest.raises(SystemExit) as exit_info:
        cli.main(["clean"])
    assert exit_info.value.code == 0
    assert "Nothing was deleted" in capsys.readouterr().out
    assert (home["config"] / ".env").exists() and home["logs"].exists()
