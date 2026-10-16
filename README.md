# v2srt - 视频智能字幕生成工具

v2srt 使用 [whisper.cpp](https://github.com/ggml-org/whisper.cpp) 在本地识别视频语音，生成 SRT 字幕，并可借助 Gemini 将字幕翻译成任意语言。

## ✨ 功能特点

- **🎯 本地识别**：whisper.cpp 在本机运行，支持 GPU 加速，音频不上传到任何云服务
- **⚡ 流式翻译**：长视频按分段识别，每段识别完立即进入翻译队列，识别与翻译同时进行，无需等待全部识别结束
- **🩹 补漏识别**：whisper 在人声稀疏的片段偶尔会整段漏识别；v2srt 会找出"有人声却没有字幕"的片段单独再识别一次
- **🧵 多线程并行**：多个 Gemini 请求并发翻译；长视频自动切段后可同时运行多个 whisper-cli 进程
- **✂️ 智能分段**：长视频自动按约 20 分钟分段，切点对齐到附近的静音处，不会把一句话切成两半
- **🌐 高质量翻译**：默认使用 `gemini-3.8-flash`，结构化 JSON 输出，携带上下文翻译，自动纠正语音识别错误
- **🛡️ 健壮可靠**：失败自动重试与退避、问题句子自动隔离、密钥/模型错误提前发现；中途失败或中断也会保留已生成的部分字幕
- **🧹 清理识别噪声**：过滤 `[BLANK_AUDIO]`、`(Music)` 等标记和 whisper 的重复"幻觉"句子，修正重叠的时间轴
- **⚙️ 免长命令行**：所有路径和密钥都可写进 `.env`，日常只需 `v2srt video.mp4`
- **📝 更多玩法**：双语字幕、保留原文字幕、批量处理、翻译已有的 `.srt` 文件、选择音轨

## 📋 准备工作

| 依赖 | 说明 |
|---|---|
| [uv](https://docs.astral.sh/uv/) | Python 项目与依赖管理工具（会自动安装合适的 Python） |
| [FFmpeg](https://ffmpeg.org/download.html) | 提取与切分音频，需要 `ffmpeg` 和 `ffprobe` |
| [whisper.cpp](https://github.com/ggml-org/whisper.cpp/releases) | 使用其中的 `whisper-cli` 程序；Windows 有 NVIDIA 显卡建议下载 CUDA 版本 |
| Whisper 模型 | 从 [ggerganov/whisper.cpp](https://huggingface.co/ggerganov/whisper.cpp/tree/main) 下载，推荐 `ggml-large-v3-turbo.bin`（速度与准确率兼顾）；显存不足可选 `ggml-medium.bin` / `ggml-small.bin` |
| VAD 模型（可选，推荐） | 从 [ggml-org/whisper-vad](https://huggingface.co/ggml-org/whisper-vad/tree/main) 下载 Silero VAD 模型，可减少静音段的错误识别 |
| Gemini API 密钥（可选） | 用于翻译，在 [Google AI Studio](https://aistudio.google.com/apikey) 获取；不配置则只生成原文字幕 |

macOS 可以用 Homebrew 安装 FFmpeg 和 whisper.cpp（自带 Metal GPU 加速，命令名同样是 `whisper-cli`）：

```bash
brew install uv ffmpeg whisper-cpp
```

## 🚀 快速开始

### 1. 安装

作为命令行工具全局安装（推荐）：

```bash
uv tool install git+https://github.com/phith0n/v2srt
```

或者克隆仓库后在项目内使用：

```bash
git clone https://github.com/phith0n/v2srt && cd v2srt
uv sync
uv run v2srt --help
```

### 2. 生成配置文件

```bash
v2srt init --global   # 用户级配置，在任何目录运行 v2srt 都会读取（全局安装时推荐）
v2srt init            # 或者只在当前目录生成 .env
```

编辑生成的 `.env`，至少填写 whisper 模型路径，需要翻译时再填写 Gemini 密钥：

```dotenv
# whisper-cli 已在 PATH 中时可以不填
V2SRT_WHISPER_CLI=/path/to/whisper-cli
V2SRT_WHISPER_MODEL=/path/to/ggml-large-v3-turbo.bin
V2SRT_VAD_MODEL=/path/to/ggml-silero-v6.2.0.bin
V2SRT_GEMINI_API_KEY=your-api-key
```

路径按所用系统的习惯填写即可（Windows 下如 `D:\models\ggml-large-v3-turbo.bin`），支持 `~` 和环境变量。

### 3. 检查环境

```bash
v2srt check
```

会显示正在使用的配置文件，逐项检查 ffmpeg、whisper-cli、模型文件，实际连接 Gemini 验证密钥与模型是否可用，并标明每项配置来自配置文件、环境变量、命令行还是默认值。

### 4. 生成字幕

```bash
v2srt movie.mp4
```

运行时会实时显示进度条和已生成的字幕，完成后在视频旁生成 `movie.srt`。

## 📖 常见用法

```bash
# 指定语音语言（默认自动识别；已知语言时指定会更准确）
v2srt -l ja movie.mkv

# 翻译成其他语言（语言代码或语言名称均可）
v2srt --to zh-TW movie.mp4
v2srt --to "Brazilian Portuguese" movie.mp4

# 双语字幕（译文在上、原文在下），并另存一份原文字幕 movie.ja.srt
v2srt -l ja --bilingual --keep-source movie.mkv

# 给翻译提供背景信息，人名、术语会更准确
v2srt --context "《进击的巨人》第3季，角色：艾伦、三笠、阿尔敏" episode01.mkv

# 批量处理（Windows 下也支持通配符），跳过已有字幕的文件
v2srt --skip-existing "/path/to/videos/*.mp4"

# 翻译已有的字幕文件，输出 talk.zh-CN.srt
v2srt talk.en.srt

# 只识别不翻译
v2srt --no-translate movie.mp4

# 多音轨视频选择第 2 条音轨（从 0 开始计数）
v2srt --audio-track 1 movie.mkv

# 手动指定切分时间点（替代自动分段）
v2srt --cut 1:00:00 2:00:00 long_video.mp4
```

## ⚙️ 配置

配置的优先级从高到低为：

1. **命令行参数**：临时调整，如 `--to en`
2. **环境变量**：在终端或脚本中设置的 `V2SRT_*` 变量
3. **一个配置文件**：按以下顺序取找到的**第一个**，多个文件之间**不会合并**
   - `--env-file` 指定的文件
   - 当前目录的 `.env`
   - 用户级配置：Windows 为 `%APPDATA%\v2srt\.env`，macOS / Linux 为 `~/.config/v2srt/.env`
4. **内置默认值**

不确定某项配置从哪里来时，运行 `v2srt check` 查看。完整的带注释模板见 [.env.example](.env.example)。

| 环境变量 | 命令行参数 | 默认值 | 说明 |
|---|---|---|---|
| `V2SRT_FFMPEG` | | `ffmpeg` | ffmpeg 路径、所在目录或命令名 |
| `V2SRT_FFPROBE` | | 与 ffmpeg 同目录 | ffprobe 路径 |
| `V2SRT_WHISPER_CLI` | | `whisper-cli` | whisper-cli 路径、所在目录或命令名 |
| `V2SRT_WHISPER_MODEL` | `-m` | （必填） | whisper 模型文件 |
| `V2SRT_VAD_MODEL` | `--vad-model` / `--no-vad` | | VAD 模型文件 |
| `V2SRT_VAD_THRESHOLD` | | `0.3` | VAD 阈值，越低越容易判定为人声 |
| `V2SRT_LANGUAGE` | `-l` | `auto` | 语音语言 |
| `V2SRT_SEGMENT` | `--segment` | `20m` | 自动分段长度，`0` 表示不分段 |
| `V2SRT_CUT` | `-c` / `--cut` | | 固定切分时间点 |
| `V2SRT_WHISPER_JOBS` | `-j` | `1` | 并行的 whisper-cli 进程数 |
| `V2SRT_GAP_FILL` | `--no-gap-fill` | `true` | 补漏识别（需要 VAD 模型） |
| `V2SRT_WHISPER_THREADS` | `--threads` | whisper 默认 | 每个 whisper-cli 的线程数 |
| `V2SRT_WHISPER_PROMPT` | `--prompt` | | whisper 初始提示词（人名、术语） |
| `V2SRT_WHISPER_ARGS` | | | 原样传给 whisper-cli 的额外参数，如 `--max-len 60 --split-on-word` |
| `V2SRT_GEMINI_API_KEY` | | | Gemini 密钥（也识别 `GEMINI_API_KEY`、`GOOGLE_API_KEY`） |
| `V2SRT_GEMINI_MODEL` | `--gemini-model` | `gemini-3.8-flash` | 翻译模型 |
| `V2SRT_GEMINI_THINKING` | `--thinking` | `low` | 思考级别：`default`/`minimal`/`low`/`medium`/`high` |
| `V2SRT_TARGET_LANGUAGE` | `-t` / `--to` | `zh-CN` | 目标语言 |
| `V2SRT_TRANSLATE` | `--no-translate` | `true` | 是否翻译 |
| `V2SRT_TRANSLATE_WORKERS` | `-w` | `4` | 并发翻译请求数 |
| `V2SRT_BATCH_SIZE` | `--batch-size` | `40` | 每次翻译请求包含的字幕条数 |
| `V2SRT_CONTEXT` | `--context` | | 翻译背景信息 |
| `V2SRT_PROXY` | | | 访问 Gemini 的代理 |
| `V2SRT_GEMINI_BASE_URL` | | | 自定义 API 地址（如 API 网关） |
| `V2SRT_BILINGUAL` | `--bilingual` | `false` | 输出双语字幕 |
| `V2SRT_KEEP_SOURCE` | `--keep-source` | `false` | 同时保存原文字幕 |
| `V2SRT_LOG_DIR` | | 系统日志目录 | 日志目录，设为 `off` 关闭日志 |
| `V2SRT_LOG_GEMINI` | | `false` | 额外保存每次 Gemini 请求的完整 prompt 和回复 |

其他命令行参数：`-o` 输出路径（多个输入时为目录）、`-f` 覆盖已有字幕、`--skip-existing`、`-q` 不打印字幕内容、`-v` 调试输出、`--keep-temp` 保留临时音频。运行 `v2srt --help` 查看全部参数。

## 📜 日志

每次运行都会自动写一份详细日志，结束时终端会打印日志路径。日志内容包括：运行环境、每项配置及其来源、实际执行的 ffmpeg / whisper-cli 命令及完整输出、分段与语言检测结果、每次 Gemini 请求的耗时 / token / 重试原因，以及最终统计。API 密钥会被打码。

| 系统 | 默认日志目录 |
|---|---|
| Windows | `%LOCALAPPDATA%\v2srt\Logs` |
| macOS | `~/Library/Logs/v2srt`（也可以在"控制台"应用中查看） |
| Linux | `~/.local/state/v2srt/log` |

- 只保留最近 50 次运行的日志，更早的会自动删除。
- 设置 `V2SRT_LOG_GEMINI=true` 后，会在日志旁额外生成 `*.gemini.jsonl`，每行是一次 Gemini 请求的完整 prompt、回复和 token 用量，适合调试翻译效果。
- 遇到问题时，把日志文件发给开发者即可。

## 🧹 清理

`v2srt clean` 会删除 v2srt 在系统中保存的内容：用户级配置文件、日志、中断运行后残留的临时音频。删除前会列出具体位置和大小，并请求确认：

```bash
v2srt clean            # 清理全部（会先确认）
v2srt clean --logs     # 只清理日志；也可以用 --config、--temp
v2srt clean --yes      # 不询问直接删除
```

**当前目录下的 `.env` 永远不会被删除。** 设置了自定义 `V2SRT_LOG_DIR` 时，只删除 v2srt 生成的日志文件，目录中的其他文件不受影响。

## 🔧 工作原理

```
ffmpeg ─► 16kHz 音频 ─► 按静音点分段 ─► whisper-cli（可多进程并行，逐行流式读取）
                                              │ 每识别出一句
                                              ▼
                                   攒成小批 ─► Gemini（多线程并发）
                                              │ 翻译结果（完成顺序不定）
                                              ▼
                                   按原顺序整理 ─► 逐条写入 SRT
```

- whisper-cli 每解码出一句就会输出到 stdout，v2srt 逐行解析；一个分段（约 20 分钟）识别完成后，整段字幕按批（默认 40 条）提交翻译，同时开始识别下一段。
- **补漏识别**：开启 VAD 时，whisper.cpp 会把一个分段里的所有人声拼接后连续解码，人声稀疏时偶尔会整段漏识别。每个分段识别完后，v2srt 会找出 VAD 检测到人声、却没有任何字幕的片段，在一次 whisper-cli 调用中逐个单独识别（模型只加载一次），再合并回去。补出来的"嗯""是"这类附和、重复音节和常见套话幻觉会被丢弃。这一步只处理漏掉的片段，额外耗时通常不多。
- 每批翻译都会附带前几句原文和已有译文作为上下文，保证人名与语气前后一致。
- 翻译结果可能乱序返回，由有序汇合器按时间顺序写入文件。
- 字幕先写入 `xxx.srt.part`，全部成功后才替换为 `xxx.srt`，不会留下半成品；失败或按 Ctrl+C 中断时会保留 `.part` 文件。

**翻译出错时的处理：** 网络错误、限流、服务端错误会指数退避重试；模型漏掉的条目会单独补翻；某一批反复失败时会二分拆小，只让真正有问题的那一句保留原文；密钥无效、模型不存在、地区不支持等致命错误会在开始前就报告，运行中出现时立即停止翻译并继续输出原文字幕。

## 🌐 使用代理访问 Gemini

如果无法直接访问 Gemini API，在 `.env` 中设置代理即可：

```dotenv
V2SRT_PROXY=http://127.0.0.1:7890
```

标准的 `HTTPS_PROXY` / `ALL_PROXY` 环境变量同样生效。

## 💡 建议

- 已知视频语言时用 `-l` 指定，可以避免自动识别出错，也让翻译提示更准确。
- 推荐配置 VAD 模型，可以明显减少静音或背景音乐处的"幻觉"字幕。
- 显存充足时可以用 `-j 2` 让两个 whisper-cli 同时处理不同分段；纯 CPU 运行时通常保持 `-j 1` 并调大 `--threads`。Apple Silicon 的 Mac 上 whisper.cpp 默认使用 Metal 加速。
- 想要更短的字幕行，可以设置 `V2SRT_WHISPER_ARGS=--max-len 60 --split-on-word`。
- 思考级别默认为 `low`：`gemini-3.8-flash` 在此级别下翻译字幕通常不进行思考（思考 token 为 0），速度最快、成本最低，已能纠正大部分识别错误。更看重质量（如人名译法、长难句）时可设为 `medium`，思考 token 按输出计费，输出量大约增加 1.5～2 倍。
- Gemini API 可能产生费用，运行结束时会显示本次消耗的 token 数量。

## 🛠️ 开发

```bash
uv sync
uv run pytest
```

测试使用模拟的 whisper-cli 和 Gemini 客户端，需要本机安装 ffmpeg。

## 📄 许可证

[MIT](https://opensource.org/licenses/MIT)
