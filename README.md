# inkprint

从个人社媒发布内容中提取写作风格指纹，生成可供 AI 直接读取的个人语言形象。

给定一组真实的个人发文（知乎答案/文章、X 推文、GitHub commit、博客等），产出一份结构化的 `VOICE PROFILE`，让任何 AI 拿到这份 profile 就能用"你的腔调"写东西。

## 当前能力

- ✅ **知乎**：爬取本人所有回答和文章（基于 [MediaCrawler_zhihu](https://github.com/xx-hub/MediaCrawler_zhihu)，扫码登录）
- ✅ **微博**：爬取本人所有原创发文（基于 [dataabc/weibo-crawler](https://github.com/dataabc/weibo-crawler)，浏览器登录拿 cookie + 自动过滤纯转发 / 抽奖 / 过短回应）
- ✅ **多源融合**：voice profile 跨平台均衡采样，自动区分各平台腔调
- ✅ **联网搜索写作**：compose 时 LLM 自主调用 web_search 工具核实事实（可选，需 Tavily key）
- ✅ **人工 Author 履历**：覆盖 LLM 推断的事实，防止年龄 / 职业 / 经历幻觉
- ⏳ X / Twitter：待接入
- ⏳ GitHub：待接入
- ⏳ 微信公众号 / 朋友圈：待评估

## 技术栈

- Python 3.13 + FastAPI + Jinja2 + SQLite（应用层）
- uv（包管理 / venv）
- Ruff（lint + format）
- MediaCrawler_zhihu（Playwright + Node.js 跑知乎签名 JS）
- weibo-crawler（requests，Playwright 弹 Chrome 让用户登录拿 cookie）
- OpenAI-compatible LLM API（默认 DeepSeek，可切 Anthropic / OpenAI / 本地代理）
- Tavily 搜索 API（可选，用于事实核查）

## 前置依赖

| 依赖 | 版本 | 用途 |
|---|---|---|
| Python | ≥ 3.13 主应用 / 3.11 微博爬虫 | uv 会自动下载这两个版本 |
| [uv](https://github.com/astral-sh/uv) | 最新 | 包管理 |
| Node.js | ≥ 16 | 跑知乎签名 JS（execjs 调用） |
| Google Chrome | 系统安装 | 知乎扫码 + 微博登录都通过 CDP 连接 |
| LLM API key | — | DeepSeek / Anthropic / OpenAI 任一 |
| Tavily API key | — | 可选，启用 compose 的联网事实核查 |

macOS 安装：

```bash
brew install uv node
# Chrome 通过官网 / brew install --cask google-chrome
```

## 安装步骤

```bash
git clone https://github.com/hcfw007/inkprint.git
cd inkprint

# macOS / Linux
./setup.sh

# Windows（或任何平台）
python setup.py
```

脚本是幂等的——重复跑也安全，只会补齐缺的东西。它会：

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

1. **新建人格**（点 `新建人格`，给个名字和描述）
2. **绑定数据源**（一人格一平台最多一个源）：
   - **知乎**：填主页 URL，形如 `https://www.zhihu.com/people/xxx`
   - **微博**：填 uid，纯数字（在 weibo.com 个人页 URL 里）
3. 点「立即同步」
   - **知乎首次**：弹出 Chrome → 显示二维码 → 用手机知乎 APP 扫码登录 → 自动开爬
   - **微博首次**：弹出 Chrome → 你在里面登录微博（账号/密码/扫码）→ 程序自动检测 `SCF`+`SSOLoginState` cookie 出现 → 关掉浏览器开爬
   - 后续两个平台 cookie 都缓存在 `data/cookies/` 和 `data/browser_profile/`，秒级开爬
   - 跑完显示「新增 N · 更新 M · 未变 K · 累计 X 条」
4. **填写 Author 履历**（可选但强烈推荐）：在详情页 `Author 履历` 框写客观事实（姓名 / 年龄段 / 职业 / 教育 / 关键经历），LLM 后续生成 / compose 时会强制采用，避免年龄/职业幻觉
5. **生成 voice profile**：后端跨平台均衡选 30 条样本 → 调 LLM → 写到 `profiles/{id}/<时间戳>.md`。点「查看历史版本」翻每次重新生成的快照
6. **按口吻写一篇**（首页人格卡片右侧的入口）：
   - 选 **长文**（600–2000 字，允许 `[图：xxx]` 占位）或 **短文**（≤ 140 字）
   - 默认按「主动发起内容」生成；勾上「这是回答问题」才走答题腔
   - 配了 Tavily key 时，模型会**自主调用 web_search** 核实事实再下笔
   - 结果下方有「执行轨迹」折叠面板，能看模型搜了啥 / 调了几次工具
7. **查看样本**：人格详情页能翻所有原文，按点赞数 / 字数排序

## 目录结构

```
inkprint/
├── app/                  主应用（路由 / 模板 / 业务逻辑）
│   ├── main.py           FastAPI 路由
│   ├── db.py             SQLite 初始化 + 简易迁移
│   ├── personas.py       人格 CRUD + 源绑定
│   ├── crawler_zhihu.py  知乎爬虫子进程封装
│   ├── crawler_weibo.py  微博爬虫子进程封装 + 噪声过滤
│   ├── browser_auth.py   Playwright CDP 弹 Chrome 登录拿 cookie
│   ├── samples.py        多平台样本读取 + (platform, content_id) 去重合并
│   ├── voice_profile.py  Profile 生成 + compose + 历史版本管理
│   ├── search.py         Tavily 搜索客户端（事实核查工具）
│   ├── llm.py            OpenAI-compatible 客户端 + tool-use 循环
│   └── templates/        Jinja2 模板
├── crawlers/zhihu/       MediaCrawler_zhihu 本地克隆（gitignore）
├── crawlers/weibo/       weibo-crawler 本地克隆（gitignore）
├── samples/{platform}/   原始爬取数据，按平台 / persona 分目录（gitignore）
├── profiles/{persona}/   生成的 voice profile 历史版本（gitignore）
├── data/                 SQLite 数据库 + 浏览器 profile + cookie 缓存（gitignore）
├── setup.py / setup.sh   一键 / 幂等装机脚本
└── pyproject.toml        应用层依赖 + ruff 配置
```

## 已知限制

- **同步是阻塞调用**：点完按钮等几十秒到几分钟，浏览器 tab 会一直转圈。本地工具暂时不上后台任务
- **微博 cookie 会过期**：一般几天到一个月，过期后同步报错；详情页有「重新登录」按钮，点了清缓存重新弹 Chrome 登录
- **微博触发反封禁会提前停**：默认 `max_weibo_per_session=2000` + 单条延迟，大账号可能跑一次只拿到几百条，再跑一次接续。可通过编辑 `crawlers/weibo/config.json` 调高，但有封号风险
- **两个爬虫都不支持增量爬取**：每次同步都是全量重爬。inkprint 在 harvest 层做了 `(platform, content_id)` 去重 + `updated_time` 合并，所以累积视角是增量的，但网络/时间成本省不掉
- **profile 索引页样本顺序可能跨同步变化**：用的是数组 index，不是稳定 ID。点进样本查看后再回来同步，原 idx 可能指向别的样本
- **没有鉴权**：默认只听 127.0.0.1，不要暴露到公网
- **代理冲突**：本机开着 clash/v2ray（`all_proxy=socks5://...`）时，爬虫调用知乎/微博时 inkprint 会自动清掉代理环境变量；LLM 调用默认 `trust_env=False` 走直连（CN provider 用），如果你用 Anthropic/OpenAI 从 CN 调用，把 `LLM_PROXY=socks5://127.0.0.1:7890` 填到 `.env`

## 开发

```bash
uv run ruff check . --fix    # lint + autofix
uv run ruff format .         # format
```
