<script setup>
import { ref, onMounted, onUnmounted, watch, nextTick } from 'vue'
import { useRouter } from 'vue-router'
import * as echarts from 'echarts'
import { useCatalogStore } from '@/stores/catalog'

const props = defineProps({
  activeNodeId: { type: String, default: null },
})

const emit = defineEmits(['exit'])

const router = useRouter()
const store = useCatalogStore()
const chartContainer = ref(null)
let chartInstance = null
let graphData = null

// 深色学术配色
const categoryDefs = {
  'traditional-ml':  { color: '#2c3e50', name: '传统机器学习' },
  'neural-networks': { color: '#d35400', name: '神经网络基础' },
  'cv':              { color: '#16a085', name: '计算机视觉' },
  'nlp':             { color: '#8e44ad', name: 'NLP与检索' },
  'llm':             { color: '#c0392b', name: '大语言模型' },
  'aigc':            { color: '#2980b9', name: 'AIGC与多模态' },
  'optimization':    { color: '#f39c12', name: '优化与部署' },
}

function findDocId(nodeId) {
  for (const [file, info] of Object.entries(store.flatMap)) {
    if (info.id === nodeId) return file.replace('.md', '')
  }
  return null
}

async function loadGraphData() {
  const mod = await import('@/data/relations.json')
  graphData = mod.default
}

function buildOption() {
  if (!graphData) return {}

  const categories = Object.entries(categoryDefs).map(([, v]) => ({
    name: v.name, itemStyle: { color: v.color },
  }))

  const highlightId = props.activeNodeId

  return {
    backgroundColor: '#f5f6f8',
    animationDuration: 500,
    tooltip: {
      formatter(p) {
        if (p.dataType === 'node') {
          const cat = categoryDefs[p.data.category]?.name || ''
          const hint = '<br/><span style="color:#999;font-size:11px">单击高亮 · 双击打开文档</span>'
          return `<strong>${p.name}</strong><br/>${cat}${hint}`
        }
        return `${p.data.source} → ${p.data.target}<br/>${p.data.relation || ''}`
      },
      backgroundColor: 'rgba(44,62,80,0.94)',
      borderColor: '#2c3e50',
      textStyle: { color: '#fff', fontSize: 12 },
    },
    legend: {
      bottom: 8,
      textStyle: { color: '#555', fontSize: 11 },
      data: categories.map(c => c.name),
    },
    series: [{
      type: 'graph',
      layout: 'force',
      roam: true,
      draggable: true,
      force: { repulsion: 350, gravity: 0.06, edgeLength: [100, 300], layoutAnimation: true },
      data: graphData.nodes.map(node => {
        const isActive = node.id === highlightId
        return {
          ...node,
          category: categoryDefs[node.category]?.name || node.category,
          x: isActive ? 0 : undefined,
          y: isActive ? 0 : undefined,
          symbolSize: isActive
            ? Math.min((node.symbolSize || 25) + 14, 60)
            : (node.symbolSize || 18),
          itemStyle: {
            color: isActive ? '#d35400' : undefined,
            borderColor: '#fff',
            borderWidth: isActive ? 3 : 2,
            shadowBlur: isActive ? 14 : 4,
            shadowColor: 'rgba(0,0,0,0.18)',
          },
          label: {
            show: true,
            position: 'right',
            fontSize: isActive ? 13 : 10,
            fontWeight: isActive ? 'bold' : 'normal',
            color: '#2c3e50',
            textBorderColor: '#fff',
            textBorderWidth: 3,
          },
        }
      }),
      links: graphData.links.map(link => ({
        ...link,
        lineStyle: { color: '#c8ccd0', curveness: 0.2, opacity: 0.55, width: 1.2 },
      })),
      categories,
      emphasis: {
        focus: 'adjacency',
        lineStyle: { width: 3, color: '#d35400' },
        itemStyle: { shadowBlur: 18, shadowColor: 'rgba(0,0,0,0.3)' },
        label: { fontSize: 14, fontWeight: 'bold' },
      },
      lineStyle: { color: '#c8ccd0', curveness: 0.2, opacity: 0.55, width: 1.2 },
    }],
  }
}

function highlightActiveNode() {
  if (!chartInstance || !graphData || !props.activeNodeId) return
  const idx = graphData.nodes.findIndex(n => n.id === props.activeNodeId)
  if (idx >= 0) {
    chartInstance.dispatchAction({ type: 'highlight', dataIndex: idx })
  }
}

function renderChart() {
  if (!chartContainer.value) return
  if (!chartInstance) chartInstance = echarts.init(chartContainer.value)
  chartInstance.setOption(buildOption(), true)
  highlightActiveNode()
}

let clickTimer = null

function bindChartEvents() {
  if (!chartInstance) return
  chartInstance.off('click')
  chartInstance.off('dblclick')

  chartInstance.on('click', (params) => {
    if (params.dataType !== 'node') return
    if (clickTimer) clearTimeout(clickTimer)
    clickTimer = setTimeout(() => {
      // 单击：仅更新 activeNodeId，局部图谱联动
      store.activateDoc(params.data.id)
      clickTimer = null
    }, 300)
  })

  chartInstance.on('dblclick', (params) => {
    if (clickTimer) { clearTimeout(clickTimer); clickTimer = null }
    if (params.dataType !== 'node') return
    // 双击：导航到文档 + 退出全屏
    const docId = findDocId(params.data.id)
    if (docId) {
      store.activateDoc(params.data.id)
      router.push({ name: 'doc', params: { docId } })
      emit('exit')
    }
  })
}

let resizeObserver = null

onMounted(async () => {
  await loadGraphData()
  await nextTick()
  renderChart()
  bindChartEvents()
  resizeObserver = new ResizeObserver(() => chartInstance?.resize())
  if (chartContainer.value) resizeObserver.observe(chartContainer.value)
})

onUnmounted(() => {
  resizeObserver?.disconnect()
  chartInstance?.dispose()
  chartInstance = null
})

watch(() => props.activeNodeId, () => {
  if (chartInstance && graphData) {
    chartInstance.setOption(buildOption())
    highlightActiveNode()
  }
})

watch(() => store.flatMap, (m) => {
  if (Object.keys(m).length > 0) bindChartEvents()
}, { immediate: true })
</script>

<template>
  <div class="h-full w-full flex flex-col relative">
    <!-- 顶栏 -->
    <div class="flex items-center justify-between h-10 px-4 border-b border-morandi-border-light bg-white/60 flex-shrink-0">
      <span class="text-sm font-semibold text-morandi-text-primary">全量知识图谱</span>
      <div class="flex items-center gap-3">
        <span class="text-xs text-morandi-text-muted">单击高亮 · 双击打开文档</span>
        <button
          class="flex items-center gap-1 px-3 py-1 text-xs rounded bg-morandi-accent-green text-white hover:opacity-90 transition-opacity"
          @click="emit('exit')"
        >
          <svg class="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12" />
          </svg>
          退出探索
        </button>
      </div>
    </div>
    <!-- 图谱 -->
    <div ref="chartContainer" class="flex-1 w-full min-h-0" />
  </div>
</template>
