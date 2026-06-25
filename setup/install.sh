#!/bin/bash
# ============================================================
# 机器学习知识库 — 一键安装脚本
# ============================================================
# 用法: bash setup/install.sh
# 前提: 已安装 Node.js >= 18
# ============================================================
set -e

echo "=========================================="
echo "  机器学习知识库 — 依赖安装脚本"
echo "=========================================="
echo ""

# 1. 核心框架
echo "[1/5] 安装核心框架 (Vue 3 + Vite + Router + Pinia)..."
npm install vue@^3.4.0 vue-router@^4.3.0 pinia@^2.1.0

# 2. 构建工具
echo "[2/5] 安装构建工具 (Vite + Vue 插件)..."
npm install -D vite@^5.4.0 @vitejs/plugin-vue@^5.0.0

# 3. CSS 框架 (Tailwind + 排版插件)
echo "[3/5] 安装 CSS 框架 (Tailwind CSS + Typography 插件)..."
npm install -D tailwindcss@^3.4.0 postcss@^8.4.0 autoprefixer@^10.4.0
npm install @tailwindcss/typography@^0.5.0

# 4. Markdown 渲染 (markdown-it + KaTeX 数学公式 + 代码高亮)
echo "[4/5] 安装 Markdown 渲染引擎 (markdown-it + KaTeX + highlight.js)..."
npm install markdown-it@^14.0.0 markdown-it-katex@^2.0.3 katex@^0.16.11 highlight.js@^11.9.0

# 5. 可视化 + 搜索
echo "[5/5] 安装可视化图表 + 模糊搜索 (ECharts + Fuse.js)..."
npm install echarts@^5.5.0 fuse.js@^7.4.2

echo ""
echo "=========================================="
echo "  安装完成!"
echo "=========================================="
echo ""
echo "启动开发服务器:  npm run dev"
echo "构建生产版本:    npm run build"
echo "预览构建结果:    npm run preview"
echo ""
