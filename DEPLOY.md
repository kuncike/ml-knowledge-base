# 机器学习知识库 — 部署与配置指南

## 一、便携文件架构

```
learn/                          # 项目根目录
├── index.html                  # Vite 入口 HTML（挂载 #app）
├── package.json                # 依赖与脚本定义
├── vite.config.js              # Vite 构建配置（别名、插件）
├── tailwind.config.js          # Tailwind CSS 主题（莫兰迪配色）
├── postcss.config.js           # PostCSS 插件（Tailwind + Autoprefixer）
│
├── src/                        # 源代码（开发时编辑这里）
│   ├── main.js                 # 应用入口：创建 Vue 实例，挂载 Pinia/Router
│   ├── App.vue                 # 根组件：加载目录数据，渲染 Layout
│   │
│   ├── router/
│   │   └── index.js            # Vue Router 路由定义（/ 和 /doc/:docId）
│   │
│   ├── stores/
│   │   └── catalog.js          # Pinia Store：目录树、展开/折叠、文档切换
│   │
│   ├── views/
│   │   └── Home.vue            # 页面视图，内嵌 ContentReader 组件
│   │
│   ├── components/
│   │   ├── Layout.vue          # 整体布局（侧边栏 + 主内容区）
│   │   ├── Sidebar.vue         # 左侧目录树导航
│   │   ├── ContentReader.vue   # 核心：动态加载 .md 文件并渲染
│   │   ├── MarkdownViewer.vue  # Markdown 渲染器（markdown-it + KaTeX + highlight.js）
│   │   └── KnowledgeGraph.vue  # ECharts 知识图谱可视化
│   │
│   ├── content/                # 知识库内容（91 个 .md 文件）
│   │   ├── transformer.md      # 每篇一个算法/概念，纯 Markdown
│   │   ├── rag.md
│   │   └── ...
│   │
│   ├── data/
│   │   ├── catalog.json        # 目录树结构（嵌套节点，含 file/id/title）
│   │   └── relations.json      # 知识点关联关系（用于图谱）
│   │
│   └── assets/
│       └── main.css            # 全局样式（Tailwind 指令 + 自定义）
│
└── dist/                       # 构建产物（部署时只需要这个目录）
    ├── index.html              # 构建后的入口 HTML
    └── assets/                 # 打包后的 JS/CSS（带 hash，可长期缓存）
        ├── index-CBRrOoe6.js
        ├── index-BpMUTDLF.css
        └── ...
```

### 便携性说明

- **开发时**：只需要 `src/` 下的 `.md` 文件和 `.json` 数据文件即可修改内容。
- **部署时**：只需要 `dist/` 目录，放到任意静态文件服务器即可运行。无需 Node.js、无需数据库、无需后端。

## 二、运行哪个文件

### 开发模式

```bash
cd learn
npm install        # 首次需要安装依赖
npm run dev        # 启动 Vite 开发服务器，默认 http://localhost:5173
```

- 入口文件：`src/main.js`（由 `index.html` 通过 `<script type="module" src="/src/main.js">` 加载）
- Vite 开发服务器提供热更新（HMR），修改代码即时生效。

### 生产部署

部署只需要 `dist/` 目录，入口是 `dist/index.html`。

**方式一：本地预览构建产物**
```bash
npm run build      # 构建到 dist/
npm run preview    # 预览构建结果，默认 http://localhost:4173
```

**方式二：任意静态服务器**

将 `dist/` 目录下的所有文件上传到服务器的 Web 根目录即可。例如：

```bash
# Nginx: 将 dist/ 内容放到 /usr/share/nginx/html/
cp -r dist/* /usr/share/nginx/html/

# 或用 Python 快速启动
cd dist && python3 -m http.server 8080
```

**注意**：本项目使用 Vue Router 的 HTML5 History 模式（`createWebHistory()`），所有路由（如 `/doc/transformer`）在前端通过 JS 匹配。直接访问子路径时，服务器必须配置 **全部路径回退到 index.html**，否则会 404。

## 三、域名配置

### 场景 A：Nginx 部署

```nginx
server {
    listen 80;
    server_name your-domain.com;          # 替换为你的域名

    root /path/to/learn/dist;             # 指向 dist 目录
    index index.html;

    # 关键：SPA 路由回退 —— 所有路径都返回 index.html
    location / {
        try_files $uri $uri/ /index.html;
    }

    # 静态资源长缓存（Vite 构建产物带 hash）
    location /assets/ {
        expires 1y;
        add_header Cache-Control "public, immutable";
    }
}
```

### 场景 B：Vite 开发环境自定义域名

修改 `vite.config.js`：

```js
import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'

export default defineConfig({
  plugins: [vue()],
  server: {
    host: '0.0.0.0',     // 监听所有网络接口，允许外部访问
    port: 5173,           // 自定义端口（可选）
  },
  assetsInclude: ['**/*.md'],
  resolve: {
    alias: {
      '@': '/src',
    },
  },
})
```

### 场景 C：子路径部署（如 `https://example.com/ml-kb/`）

需要修改 `vite.config.js` 的 `base` 和路由的 `history`：

```js
// vite.config.js
export default defineConfig({
  base: '/ml-kb/',       // 静态资源前缀
  // ... 其余配置
})

// src/router/index.js
const router = createRouter({
  history: createWebHistory('/ml-kb/'),   // 路由基础路径
  routes,
})
```

### 场景 D：反向代理（如前面加 Nginx/Caddy）

Vite 开发服务器默认绑定 `localhost`，外部无法直接访问。用 Nginx 反代：

```nginx
server {
    listen 80;
    server_name dev.your-domain.com;

    location / {
        proxy_pass http://127.0.0.1:5173;    # 转发到 Vite dev server
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";  # HMR WebSocket 支持
        proxy_set_header Host $host;
    }
}
```

## 四、数据添加与修改

所有内容由三个数据层驱动，修改后 `npm run build` 重新构建即可生效。

### 数据层总览

| 层级 | 文件 | 作用 |
|------|------|------|
| 文章内容 | `src/content/*.md` | 每个知识点一篇独立的 Markdown 文件 |
| 目录导航 | `src/data/catalog.json` | 嵌套目录树，定义左侧侧边栏的层级结构 |
| 知识图谱 | `src/data/relations.json` | 节点 + 连线，定义右侧知识图谱的可视化关系 |

### 操作 A：新增一篇知识文章

**第 1 步 — 创建 .md 文件**

在 `src/content/` 下新建文件，例如 `my-topic.md`：

```markdown
# 我的主题

## 核心原理

内容使用标准 Markdown 语法。支持：
- LaTeX 公式：$E = mc^2$ 或 $$f(x) = \frac{1}{\sqrt{2\pi}} e^{-x^2/2}$$
- 代码块：```python ... ```
- 表格、列表、引用等
```

**第 2 步 — 注册到目录树**

编辑 `src/data/catalog.json`，在合适的分类下插入叶子节点。找到目标父节点的 `children` 数组，追加一条：

```json
{ "id": "my-topic", "title": "我的主题", "file": "my-topic.md" }
```

三个字段都是必填的：
- `id`：唯一标识，与文件名（不含扩展名）保持一致最安全
- `title`：侧边栏显示的中文名称
- `file`：对应 `content/` 下的文件名

> **注意**：如果想新增一个分类分组（非叶子节点），只需写 `id`、`title`、`children`，**不要**写 `file`。叶子节点则必须有 `file`。

**第 3 步（可选）— 添加到知识图谱**

编辑 `src/data/relations.json`：

在 `nodes` 数组中添加节点：
```json
{ "id": "my-topic", "name": "我的主题", "category": "traditional-ml", "symbolSize": 25 }
```

在 `links` 数组中添加关系连线：
```json
{ "source": "my-topic", "target": "linear-regression", "relation": "相关概念" }
```

字段说明：
- `id`：必须与 catalog.json 中的 id 一致
- `category`：7 个分类之一：`traditional-ml` / `neural-networks` / `cv` / `nlp` / `llm` / `aigc` / `optimization`
- `symbolSize`：节点大小（20-55），越核心越大
- `source` / `target`：连线的起点和终点节点 id
- `relation`：关系描述文字

**第 4 步 — 验证**

```bash
npm run dev
```

新文章应出现在左侧目录中，点击可正常渲染。如果添加了图谱节点，右侧图谱应显示新节点和连线。

---

### 操作 B：修改已有文章

直接编辑 `src/content/` 下对应的 `.md` 文件即可。开发模式下保存后浏览器自动刷新。

---

### 操作 C：修改目录结构/标题

编辑 `src/data/catalog.json`：

- **改标题**：修改对应节点的 `title` 字段
- **移动位置**：在 `children` 数组间移动节点对象（整个 `{}` 块）
- **新增分类**：插入分组节点（只需 `id` + `title` + `children`，不带 `file`）

---

### 操作 D：修改知识图谱关系

编辑 `src/data/relations.json`：

- **增/删节点**：修改 `nodes` 数组
- **增/删连线**：修改 `links` 数组，`source` 和 `target` 必须是已存在的节点 id
- **调整节点大小**：修改对应节点的 `symbolSize`
- **调整分类颜色**：修改 `categories` 数组中对应项的 `color`

---

### 数据一致性约束

以下规则必须遵守，否则构建或运行时可能报错：

1. `catalog.json` 中每个叶子节点（有 `file` 的）必须在 `src/content/` 下有对应的 `.md` 文件
2. `catalog.json` 中的 `file` 值与 `.md` 文件名**完全一致**（含扩展名）
3. `relations.json` 中 `nodes[].id` 应与 `catalog.json` 中对应条目的 `id` 一致
4. `relations.json` 中每个 `link` 的 `source` 和 `target` 必须在 `nodes` 数组中有对应的 `id`

---

### 数据加载原理

相关内容有助于理解数据流：

```
catalog.json  ──import──▶  Pinia Store (catalog.js)
                                │
                    ┌───────────┼───────────┐
                    ▼                       ▼
              Sidebar.vue            ContentReader.vue
           (渲染目录树)         (根据 docId import .md 文件)
                                            │
                                    ┌───────┴───────┐
                                    ▼               ▼
                            MarkdownViewer    KnowledgeGraph
                          (markdown-it 渲染)  (ECharts 力导向图)
                                                    ▲
relations.json ────import───────────────────────────┘
```

- **ContentReader.vue** 通过 `import(@/content/${docId}.md?raw)` 动态加载 Markdown — Vite 的 `?raw` 后缀将文件内容作为字符串导入
- **路由 `/:docId`** 中的 `docId` 即文件名去掉 `.md` 后缀
- **KnowledgeGraph.vue** 加载 `relations.json` 并交给 ECharts 渲染力导向图，点击节点可跳转到对应文档

## 五、快速检查清单

- [ ] `npm run build` 无报错，生成 `dist/` 目录
- [ ] `dist/index.html` 中资源引用路径正确（默认 `/assets/...`）
- [ ] 服务器正确配置 SPA 路由回退（`try_files ... /index.html`）
- [ ] 直接访问子路径（如 `/doc/transformer`）刷新不 404
- [ ] HTTPS/域名 DNS 已解析到服务器 IP
