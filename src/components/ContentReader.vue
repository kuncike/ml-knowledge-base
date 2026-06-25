<script setup>
import { ref, watch, nextTick, onUnmounted } from 'vue'
import { useRoute } from 'vue-router'
import { useCatalogStore } from '@/stores/catalog'
import mermaid from 'mermaid'
import * as echarts from 'echarts'
import MarkdownViewer from './MarkdownViewer.vue'

mermaid.initialize({
  theme: 'neutral',
  startOnLoad: false,
})

const route = useRoute()
const store = useCatalogStore()

const markdownContent = ref('')
const loading = ref(false)
const error = ref(null)
const currentTitle = ref('')

// Vite 静态 glob 导入 — 比动态 import() 更可靠，路径相对项目根
const mdModules = import.meta.glob('/src/content/*.md', { query: '?raw', import: 'default' })

function resolveModulePath(docId) {
  return `/src/content/${docId}.md`
}

async function loadDoc(docId) {
  loading.value = true
  error.value = null
  try {
    const key = resolveModulePath(docId)
    const loader = mdModules[key]
    if (!loader) throw new Error('Not found')
    markdownContent.value = await loader()
    const info = store.flatMap[docId + '.md']
    currentTitle.value = info?.title ?? docId
    store.activateDoc(docId + '.md')
  } catch (e) {
    error.value = `未找到文档: ${docId}`
    markdownContent.value = ''
  } finally {
    loading.value = false
  }
}

function reset() {
  markdownContent.value = ''
  error.value = null
  currentTitle.value = ''
}

// 监听路由变化，加载对应文档
watch(
  () => route.params.docId,
  (docId) => {
    if (docId) {
      loadDoc(docId)
    } else {
      reset()
    }
  },
  { immediate: true }
)

// Mermaid + ECharts 图表渲染 — 文档内容切换后执行
const echartsInstances = []

function disposeECharts() {
  for (const instance of echartsInstances) {
    instance.dispose()
  }
  echartsInstances.length = 0
}

function renderECharts() {
  const containers = document.querySelectorAll('.echarts-placeholder')
  for (const container of containers) {
    const code = container.getAttribute('data-echarts-code')
    if (!code) continue
    try {
      const decoded = decodeURIComponent(escape(atob(code)))
      const option = new Function(decoded)()
      const instance = echarts.init(container)
      instance.setOption(option)
      echartsInstances.push(instance)
    } catch (e) {
      console.error('[ECharts] 图表渲染失败:', e)
      container.innerHTML = `<div class="text-sm text-red-500 p-4 border border-red-200 rounded bg-red-50">图表渲染错误: ${e.message}</div>`
    }
  }
}

function handleResize() {
  for (const instance of echartsInstances) {
    try { instance.resize() } catch (_) {}
  }
}

async function renderCharts() {
  await nextTick()
  disposeECharts()
  try {
    await mermaid.run({ querySelector: '.language-mermaid' })
  } catch (_) {
    // 解析失败不阻塞页面
  }
  renderECharts()
}

watch(markdownContent, (val) => {
  if (val) renderCharts()
})

window.addEventListener('resize', handleResize)

onUnmounted(() => {
  disposeECharts()
  window.removeEventListener('resize', handleResize)
})
</script>

<template>
  <div class="max-w-3xl mx-auto">
    <!-- 加载态 -->
    <div v-if="loading" class="flex items-center justify-center py-20">
      <div class="flex flex-col items-center gap-3">
        <div class="w-8 h-8 border-2 border-morandi-accent-green border-t-transparent rounded-full animate-spin" />
        <span class="text-sm text-morandi-text-muted">加载中...</span>
      </div>
    </div>

    <!-- 错误态 -->
    <div v-else-if="error" class="flex items-center justify-center py-20">
      <div class="text-center">
        <div class="text-4xl mb-3">🤷</div>
        <p class="text-morandi-text-secondary">{{ error }}</p>
      </div>
    </div>

    <!-- 空态 -->
    <div v-else-if="!markdownContent" class="flex items-center justify-center py-20">
      <div class="text-center max-w-md">
        <div class="text-4xl mb-4">📚</div>
        <h2 class="text-xl font-semibold text-morandi-text-primary mb-2">机器学习知识库</h2>
        <p class="text-sm text-morandi-text-secondary leading-relaxed">
          欢迎！请从左侧目录选择一个主题开始阅读。<br />
          本知识库涵盖传统机器学习、深度学习、计算机视觉、NLP、大语言模型、AIGC 及模型优化部署等七大领域。
        </p>
      </div>
    </div>

    <!-- 文档内容 -->
    <div v-else>
      <div class="mb-6 pb-4 border-b border-morandi-border-light">
        <h1 class="text-2xl font-semibold text-morandi-text-primary">{{ currentTitle }}</h1>
      </div>
      <MarkdownViewer :content="markdownContent" />
    </div>
  </div>
</template>
