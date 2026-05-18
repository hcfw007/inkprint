# inkprint

从个人社媒发布内容中提取写作风格指纹，生成可供 AI 直接读取的个人语言形象。

## 目标

给定一组真实的个人发文（知乎答案/文章、X 推文、GitHub commit、博客等），产出一份结构化的 `VOICE PROFILE`，让任何 AI 拿到这份 profile 就能用"你的腔调"写东西。

## 目录结构

```
crawlers/   各平台数据采集脚本（知乎 / X / GitHub / ...）
samples/    采集到的原始文本（gitignore，不入库）
profiles/   生成的 voice profile（结构化 markdown）
```

## 数据源路线图

- [ ] 知乎（基于 MediaCrawler_zhihu，登录态爬取个人答案 + 文章）
- [ ] X / Twitter（x-api skill，OAuth 授权拉原创推文）
- [ ] GitHub（gh CLI 拉 commit message / issue / PR 描述）
- [ ] 微信公众号 / 朋友圈（待评估，可能走手动导出）

## 当前进度

第一阶段：跑通知乎数据采集 → 生成第一版 voice profile。
