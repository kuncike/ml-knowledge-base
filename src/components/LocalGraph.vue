<script setup>
import { ref, watch, nextTick, onUnmounted } from 'vue'
import { useRouter } from 'vue-router'
import * as echarts from 'echarts'
import { useCatalogStore } from '@/stores/catalog'

const props = defineProps({
  activeNodeId: { type: String, default: '' },
  graphData: { type: Object, required: true },
})

const emit = defineEmits(['enterExplore'])

const router = useRouter()
const store = useCatalogStore()
const chartContainer = ref(null)
const notFound = ref(false)
let chartInstance = null
let resizeObserver = null

const categoryDefs = {
  'traditional-ml':  { color: '#2c3e50', name: '传统机器学习' },
  'neural-networks': { color: '#d35400', name: '神经网络基础' },
  'cv':              { color: '#16a085', name: '计算机视觉' },
  'nlp':             { color: '#8e44ad', name: 'NLP与检索' },
  'llm':             { color: '#c0392b', name: '大语言模型' },
  'aigc':            { color: '#2980b9', name: 'AIGC与多模态' },
  'optimization':    { color: '#f39c12', name: '优化与部署' },
}

function buildAdjacency(links) {
  const adj = {}
  for (const link of links) {
    if (!adj[link.source]) adj[link.source] = new Set()
    if (!adj[link.target]) adj[link.target] = new Set()
    adj[link.source].add(link.target)
    adj[link.target].add(link.source)
  }
  return adj
}

function getNeighborhood(centerId, nodes, links, maxDepth = 2) {
  const adj = buildAdjacency(links)
  if (!adj[centerId]) {
    const node = nodes.find(n => n.id === centerId)
    return { nodes: node ? [node] : [], links: [], depths: new Map([[centerId, 0]]) }
  }
  const depths = new Map()
  const queue = [{ id: centerId, depth: 0 }]
  depths.set(centerId, 0)
  while (queue.length > 0) {
    const { id, depth } = queue.shift()
    if (depth >= maxDepth) continue
    for (const nb of (adj[id] || [])) {
      if (!depths.has(nb)) {
        depths.set(nb, depth + 1)
        queue.push({ id: nb, depth: depth + 1 })
      }
    }
  }
  const filteredNodes = nodes.filter(n => depths.has(n.id))
  const filteredLinks = links.filter(l => depths.has(l.source) && depths.has(l.target))
  return { nodes: filteredNodes, links: filteredLinks, depths }
}

function buildOption(neighborhood, centerId) {
  const categories = Object.entries(categoryDefs).map(([, v]) => ({
    name: v.name, itemStyle: { color: v.color },
  }))

  return {
    backgroundColor: '#fafafa',
    animationDuration: 400,
    tooltip: {
      formatter(p) {
        if (p.dataType === 'node') {
          const cat = categoryDefs[p.data.category]?.name || ''
          return `<strong>${p.name}</strong><br/>${cat}`
        }
        return `${p.data.source} → ${p.data.target}<br/>${p.data.relation || ''}`
      },
      backgroundColor: 'rgba(44,62,80,0.94)',
      borderColor: '#2c3e50',
      textStyle: { color: '#fff', fontSize: 12 },
    },
    series: [{
      type: 'graph',
      layout: 'force',
      roam: true,
      draggable: true,
      force: { repulsion: 160, gravity: 0.18, edgeLength: [70, 180], layoutAnimation: true },
      data: neighborhood.nodes.map(node => {
        const isCenter = node.id === centerId
        return {
          ...node,
          category: categoryDefs[node.category]?.name || node.category,
          symbolSize: isCenter ? Math.min((node.symbolSize || 25) + 10, 55) : (node.symbolSize || 18),
          itemStyle: {
            color: isCenter ? '#d35400' : undefined,
            borderColor: '#fff',
            borderWidth: isCenter ? 3 : 2,
            shadowBlur: isCenter ? 12 : 4,
            shadowColor: 'rgba(0,0,0,0.18)',
          },
          label: {
            show: isCenter || (node.symbolSize || 18) > 25,
            fontSize: isCenter ? 12 : 10,
            fontWeight: isCenter ? 'bold' : 'normal',
            color: '#2c3e50',
            textBorderColor: '#fff',
            textBorderWidth: 3,
          },
        }
      }),
      links: neighborhood.links.map(link => ({
        ...link,
        lineStyle: { color: '#bdc3c7', curveness: 0.2, opacity: 0.7, width: 1.5 },
        label: {
          show: true,
          formatter: link.relation || '',
          fontSize: 9,
          color: '#7f8c8d',
          textBorderColor: '#fff',
          textBorderWidth: 2,
        },
      })),
      categories,
      emphasis: {
        focus: 'adjacency',
        lineStyle: { width: 3, color: '#d35400' },
        itemStyle: { shadowBlur: 16, shadowColor: 'rgba(0,0,0,0.3)' },
        label: { fontSize: 13, fontWeight: 'bold' },
      },
      lineStyle: { color: '#bdc3c7', curveness: 0.2, opacity: 0.7, width: 1.5 },
    }],
  }
}

function findDocId(nodeId) {
  for (const [file, info] of Object.entries(store.flatMap)) {
    if (info.id === nodeId) return file.replace('.md', '')
  }
  return null
}

function bindChartClick() {
  if (!chartInstance) return
  chartInstance.off('click')
  chartInstance.on('click', (params) => {
    if (params.dataType === 'node') {
      const docId = findDocId(params.data.id)
      if (docId) {
        store.activateDoc(params.data.id)
        router.push({ name: 'doc', params: { docId } })
      }
    }
  })
}

function disposeChart() {
  resizeObserver?.disconnect()
  resizeObserver = null
  chartInstance?.dispose()
  chartInstance = null
}

function renderLocalGraph(nodeId) {
  // 1. Guard: empty data
  if (!props.graphData?.nodes?.length) {
    console.warn('[LocalGraph] graphData 为空或 nodes 缺失')
    return
  }

  // 2. Verify node ID exists in the graph
  const targetNode = props.graphData.nodes.find(n => n.id === nodeId)
  if (!targetNode) {
    console.warn(
      `[LocalGraph] 找不到节点 ID: "${nodeId}"。` +
      `请检查 catalog.json 和 relations.json 中的 ID 是否完全一致！`
    )
    notFound.value = true
    disposeChart()
    return
  }

  notFound.value = false

  // 3. Safe BFS 2-hop neighborhood
  let neighborhood
  try {
    neighborhood = getNeighborhood(nodeId, props.graphData.nodes, props.graphData.links)
  } catch (err) {
    console.error('[LocalGraph] BFS 过滤失败:', err)
    notFound.value = true
    return
  }

  // 4. Render chart after DOM updates
  nextTick(() => {
    if (!chartContainer.value) return
    if (!chartInstance) {
      chartInstance = echarts.init(chartContainer.value)
      bindChartClick()
      resizeObserver = new ResizeObserver(() => chartInstance?.resize())
      resizeObserver.observe(chartContainer.value)
    }
    chartInstance.setOption(buildOption(neighborhood, nodeId), true)
  })
}

watch(
  () => props.activeNodeId,
  (newId) => {
    disposeChart()
    if (!newId) {
      notFound.value = false
      return
    }
    renderLocalGraph(newId)
  },
  { immediate: true }
)

watch(() => store.flatMap, (m) => {
  if (Object.keys(m).length > 0 && chartInstance) bindChartClick()
}, { immediate: true })

onUnmounted(() => {
  disposeChart()
})
</script>

<template>
  <div class="h-full flex flex-col">
    <div class="flex items-center justify-between h-10 px-3 border-b border-morandi-border-light flex-shrink-0">
      <span class="text-xs font-semibold text-morandi-accent-blue tracking-wide">局部关系</span>
      <button
        class="flex items-center gap-1 px-2 py-1 text-xs rounded text-morandi-text-muted hover:text-morandi-accent-green hover:bg-morandi-bg-tertiary transition-colors"
        title="全屏探索"
        @click="emit('enterExplore')"
      >
        <svg class="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0zM10 7v3m0 0v3m0-3h3m-3 0H7" />
        </svg>
        <span>全屏探索</span>
      </button>
    </div>
    <div class="flex-1 min-h-0">
      <div v-if="!props.activeNodeId" class="h-full flex items-center justify-center text-xs text-morandi-text-muted px-4 text-center">
        请从左侧目录选择一个主题<br/>以查看其局部关系网络
      </div>
      <div v-else-if="notFound" class="h-full flex items-center justify-center text-xs text-morandi-text-muted px-4 text-center">
        暂无关联图谱<br/><span class="text-morandi-text-muted/60">该主题暂未收录到知识图谱中</span>
      </div>
      <div v-else ref="chartContainer" class="w-full h-full" />
    </div>
  </div>
</template>
