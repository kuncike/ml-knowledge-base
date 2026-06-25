# 插件与依赖清单

## 核心框架

| 包名 | 版本 | 用途 | 安装命令 |
|------|------|------|----------|
| `vue` | `^3.4.0` | 前端框架 (Composition API + SFC) | `npm install vue@^3.4.0` |
| `vue-router` | `^4.3.0` | SPA 页面路由 (HTML5 History 模式) | `npm install vue-router@^4.3.0` |
| `pinia` | `^2.1.0` | 状态管理 (目录树、展开/折叠、文档切换) | `npm install pinia@^2.1.0` |
| `vite` | `^5.4.0` | 构建工具 (开发服务器 + 生产打包) | `npm install -D vite@^5.4.0` |
| `@vitejs/plugin-vue` | `^5.0.0` | Vite 的 Vue SFC 编译插件 | `npm install -D @vitejs/plugin-vue@^5.0.0` |

## CSS / 样式

| 包名 | 版本 | 用途 | 安装命令 |
|------|------|------|----------|
| `tailwindcss` | `^3.4.0` | 原子化 CSS 框架 (莫兰迪配色主题) | `npm install -D tailwindcss@^3.4.0` |
| `@tailwindcss/typography` | `^0.5.0` | Tailwind 排版插件 (文章正文样式) | `npm install @tailwindcss/typography@^0.5.0` |
| `postcss` | `^8.4.0` | CSS 后处理器 | `npm install -D postcss@^8.4.0` |
| `autoprefixer` | `^10.4.0` | 自动添加浏览器前缀 | `npm install -D autoprefixer@^10.4.0` |

## Markdown 渲染

| 包名 | 版本 | 用途 | 安装命令 |
|------|------|------|----------|
| `markdown-it` | `^14.0.0` | Markdown → HTML 解析器 | `npm install markdown-it@^14.0.0` |
| `markdown-it-katex` | `^2.0.3` | markdown-it 的 KaTeX 插件 (渲染 LaTeX 公式) | `npm install markdown-it-katex@^2.0.3` |
| `katex` | `^0.16.11` | LaTeX 数学公式渲染引擎 (支持 `$...$` 和 `$$...$$`) | `npm install katex@^0.16.11` |
| `highlight.js` | `^11.9.0` | 代码语法高亮 | `npm install highlight.js@^11.9.0` |

## 可视化与搜索

| 包名 | 版本 | 用途 | 安装命令 |
|------|------|------|----------|
| `echarts` | `^5.5.0` | 图表库 (知识图谱力导向图 + 文章内嵌柱状图) | `npm install echarts@^5.5.0` |
| `fuse.js` | `^7.4.2` | 模糊搜索 (侧边栏内容搜索) | `npm install fuse.js@^7.4.2` |

## CDN 外部资源

以下资源通过 `<link>` 标签在 `index.html` 中加载，无需 npm 安装：

| URL | 用途 |
|-----|------|
| `https://cdn.jsdelivr.net/gh/highlightjs/cdn-release@11.9.0/build/styles/github.min.css` | highlight.js GitHub 代码高亮主题 |
| `https://fonts.googleapis.com/css2?family=Inter:...` | Inter 英文字体 |
| `https://fonts.googleapis.com/css2?family=JetBrains+Mono:...` | JetBrains Mono 等宽字体 |
| `https://fonts.googleapis.com/css2?family=Noto+Sans+SC:...` | 思源黑体 (中文) |

## 全部依赖一键安装

```bash
# 生产依赖 (dependencies)
npm install vue@^3.4.0 vue-router@^4.3.0 pinia@^2.1.0 \
  markdown-it@^14.0.0 markdown-it-katex@^2.0.3 katex@^0.16.11 \
  highlight.js@^11.9.0 echarts@^5.5.0 fuse.js@^7.4.2 \
  @tailwindcss/typography@^0.5.0

# 开发依赖 (devDependencies)
npm install -D vite@^5.4.0 @vitejs/plugin-vue@^5.0.0 \
  tailwindcss@^3.4.0 postcss@^8.4.0 autoprefixer@^10.4.0
```

或者克隆项目后直接：

```bash
npm install
```

## 配置文件一览

| 文件 | 作用 |
|------|------|
| `vite.config.js` | Vite 构建配置 (`@vitejs/plugin-vue` + `assetsInclude: ['**/*.md']`) |
| `tailwind.config.js` | Tailwind 主题 (莫兰迪配色 + `@tailwindcss/typography`) |
| `postcss.config.js` | PostCSS 插件 (`tailwindcss` + `autoprefixer`) |
| `package.json` | 依赖声明 + npm scripts (`dev` / `build` / `preview`) |

## 数据流

```
src/content/*.md ──Vite ?raw──▶ ContentReader.vue
                                      │
markdown-it + katex + hljs ◀──────────┘
        │
        ▼
  MarkdownViewer.vue ──▶ 渲染的 HTML

src/data/catalog.json ──▶ Pinia Store ──▶ Sidebar.vue (目录树)
src/data/relations.json ──▶ KnowledgeGraph/GlobalGraph/LocalGraph.vue (ECharts 图谱)

SearchBar.vue ──▶ Fuse.js 模糊搜索 ──▶ Pinia Store ──▶ router.push()
```
