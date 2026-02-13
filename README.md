# Personal Website

## Run locally

```bash
npm run dev
```

Open `http://localhost:8080`.

## Main files

- `/Users/xizuo/xizuo_website/index.html`: 页面结构
- `/Users/xizuo/xizuo_website/styles/main.css`: 粉色动态主题、按钮弹动、响应式布局
- `/Users/xizuo/xizuo_website/scripts/app.js`: 页面逻辑、i18n、搜索、目录、公式渲染
- `/Users/xizuo/xizuo_website/scripts/markdown.js`: Markdown 渲染
- `/Users/xizuo/xizuo_website/config/languages.js`: 语言开关
- `/Users/xizuo/xizuo_website/config/i18n.js`: UI 多语言文案
- `/Users/xizuo/xizuo_website/config/site-data.js`: 主页个人信息配置
- `/Users/xizuo/xizuo_website/data/notes.js`: 笔记配置
- `/Users/xizuo/xizuo_website/data/blog-posts.js`: 博客配置

## Supported languages

当前保留：
- `ja` (默认)
- `en`
- `zh-CN`

如需隐藏某语言：在 `/Users/xizuo/xizuo_website/config/languages.js` 把该项 `enabled` 改成 `false`。

## Home profile edit

编辑 `/Users/xizuo/xizuo_website/config/site-data.js`：

- `profile.avatar`: 头像路径
- `profile.email`: 邮箱链接（mailto）
- `profile.onlyfans.href`: OnlyFans 链接
- `profile.intro`: 自我介绍多语言
- `donate.alipayQr`: 支付宝二维码路径

## Notes (multi-language content)

编辑 `/Users/xizuo/xizuo_website/data/notes.js`。

每条笔记使用 `files` 配置不同语言文件：

```js
{
  id: "my-note",
  files: {
    ja: "/notes/ja/my-note.md",
    en: "/notes/en/my-note.md",
    "zh-CN": "/notes/zh-CN/my-note.md"
  },
  title: {
    ja: "...",
    en: "...",
    "zh-CN": "..."
  }
}
```

页面会根据用户语言自动读取对应 Markdown 文件。

## Blog (multi-language content)

编辑 `/Users/xizuo/xizuo_website/data/blog-posts.js`。

你可以二选一：

1. 直接写多语言内容：
```js
content: { ja: "...", en: "...", "zh-CN": "..." }
```

2. 使用多语言 Markdown 文件：
```js
files: {
  ja: "/blog/ja/post-1.md",
  en: "/blog/en/post-1.md",
  "zh-CN": "/blog/zh-CN/post-1.md"
}
```

其中：
- `title` 可选
- `image` 可选
- `content`/`files` 二选一且内容必填

## Notes page features

- 全局搜索：搜索所有笔记内容
- 单篇搜索：仅搜索当前笔记，并高亮
- 标题目录：自动抽取 h1-h6，点击可快速跳转
- 数学公式：支持 `$...$` 和 `$$...$$`（MathJax 渲染）
