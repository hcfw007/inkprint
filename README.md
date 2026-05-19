# inkprint

从个人社媒发布内容中提取写作风格指纹，生成可供 AI 直接读取的个人语言形象。

给定一组真实的个人发文（知乎答案/文章、X 推文、GitHub commit、博客等），产出一份结构化的 `VOICE PROFILE`，让任何 AI 拿到这份 profile 就能用"你的腔调"写东西。

## 当前能力

- ✅ **知乎**：爬取本人所有回答和文章（基于 [MediaCrawler_zhihu](https://github.com/xx-hub/MediaCrawler_zhihu)）
- ⏳ X / Twitter：待接入
- ⏳ GitHub：待接入
- ⏳ 微信公众号 / 朋友圈：待评估

## 技术栈

- Python 3.13 + FastAPI + Jinja2 + SQLite（应用层）
- uv（包管理 / venv）
- Ruff（lint + format）
- MediaCrawler_zhihu（Playwright + Node.js 跑知乎签名 JS）
- OpenAI-compatible LLM API（默认 DeepSeek，可切 Anthropic / OpenAI / 本地代理）

## 前置依赖

| 依赖 | 版本 | 用途 |
|---|---|---|
| Python | ≥ 3.13 | 主应用；uv 会自动下载 |
| [uv](https://github.com/astral-sh/uv) | 最新 | 包管理 |
| Node.js | ≥ 16 | 跑知乎签名 JS（execjs 调用） |
| Google Chrome | 系统安装 | MediaCrawler 通过 CDP 连接 |
| LLM API key | — | DeepSeek / Anthropic / OpenAI 任一 |

macOS 安装：

```bash
brew install uv node
# Chrome 通过官网 / brew install --cask google-chrome
```

## 安装步骤

```bash
git clone https://github.com/hcfw007/inkprint.git
cd inkprint
./setup.sh
```

`setup.sh` 是幂等的——重复跑也安全，只会补齐缺的东西。它会：

1. 检查 uv / git / node / Chrome 是否就位
2. `uv sync` 主项目依赖
3. clone 知乎爬虫（`crawlers/zhihu/`）+ 装它自己的 venv + 拷贝 example 配置
4. clone 微博爬虫（`crawlers/weibo/`）+ 建 Python 3.11 venv + pip 装依赖 + 备份默认配置
5. 没有 `.env` 时从 `.env.example` 拷一份给你填

不需要 `playwright install`——两个爬虫都配置为通过 CDP 连系统 Chrome，省 500MB 浏览器下载。

跑完按提示去编辑 `.env`，然后启动。

## 启动

```bash
uv run uvicorn app.main:app --reload --port 8765
```

打开 http://127.0.0.1:8765/

## 使用流程

1. 新建人格（点 `新建人格`，给个名字和描述）
2. 在人格详情页绑定知乎源（填你的主页 URL，形如 `https://www.zhihu.com/people/xxx`）
3. 点「立即同步」
   - **首次**：会弹出 Chrome，显示二维码，用手机知乎 APP 扫码登录
   - 后续：cookie 已缓存，直接爬取
   - 跑完显示「新增 N · 更新 M · 未变 K · 累计 X 条」
4. 点「生成 voice profile」
   - 后端选最具代表性的 30 条样本 → 调 LLM → 写到 `profiles/{id}/<时间戳>.md`
   - 大概 10-30 秒返回
5. profile 在页面上直接渲染；点「查看历史版本」翻每次重新生成的快照
6. 点「查看样本」翻你的原文，按点赞数 / 字数排序

## 目录结构

```
inkprint/
├── app/                  主应用（路由 / 模板 / 业务逻辑）
│   ├── main.py           FastAPI 路由
│   ├── db.py             SQLite 初始化 + 简易迁移
│   ├── personas.py       人格 CRUD + 源绑定
│   ├── crawler_zhihu.py  知乎爬虫子进程封装
│   ├── samples.py        样本读取 + content_id 去重合并
│   ├── voice_profile.py  Profile 生成 + 历史版本管理
│   ├── llm.py            OpenAI-compatible 客户端
│   └── templates/        Jinja2 模板
├── crawlers/zhihu/       MediaCrawler_zhihu 本地克隆（gitignore）
├── samples/              原始爬取数据，按平台 / persona 分目录（gitignore）
├── profiles/             生成的 voice profile，按 persona 分目录（gitignore）
├── data/                 SQLite 数据库（gitignore）
└── pyproject.toml        应用层依赖 + ruff 配置
```

## 已知限制

- **同步是阻塞调用**：点完按钮等 30 秒，浏览器 tab 会一直转圈。本地工具暂时不上后台任务
- **MediaCrawler 不支持增量爬取**：每次同步都是全量重爬。inkprint 在 harvest 层做了 `content_id` 去重 + `updated_time` 合并，所以累积视角是增量的，但网络/时间成本省不掉
- **profile 索引页样本顺序可能跨同步变化**：用的是数组 index，不是稳定 ID。点进样本查看后再回来同步，原 idx 可能指向别的样本
- **没有鉴权**：默认只听 127.0.0.1，不要暴露到公网
- **代理冲突**：如果本机开着 clash/v2ray（`all_proxy=socks5://...`），MediaCrawler 调用知乎时 inkprint 会自动清掉代理环境变量；LLM 调用走 socksio 透传（clash 的 CN/海外规则路由）

## 开发

```bash
uv run ruff check . --fix    # lint + autofix
uv run ruff format .         # format
```
