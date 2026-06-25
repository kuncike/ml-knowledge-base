<script setup>
import { ref, onUnmounted } from 'vue'
import { useCatalogStore } from '@/stores/catalog'
import relationsData from '@/data/relations.json'
import Sidebar from './Sidebar.vue'
import ContentReader from './ContentReader.vue'
import LocalGraph from './LocalGraph.vue'
import GlobalGraph from './GlobalGraph.vue'
import SearchBar from './SearchBar.vue'

const store = useCatalogStore()
const sidebarCollapsed = ref(false)
const rightPanelCollapsed = ref(false)
const showGlobalGraph = ref(false)
const rightPanelWidth = ref(400)
const isDragging = ref(false)

function startDrag(e) {
  e.preventDefault()
  isDragging.value = true
  document.body.style.userSelect = 'none'
  document.body.style.cursor = 'col-resize'
  document.addEventListener('mousemove', onDrag)
  document.addEventListener('mouseup', stopDrag)
}

function onDrag(e) {
  if (!isDragging.value) return
  const maxWidth = Math.min(800, window.innerWidth * 0.5)
  rightPanelWidth.value = Math.min(maxWidth, Math.max(250, window.innerWidth - e.clientX))
}

function stopDrag() {
  isDragging.value = false
  document.body.style.userSelect = ''
  document.body.style.cursor = ''
  document.removeEventListener('mousemove', onDrag)
  document.removeEventListener('mouseup', stopDrag)
}

onUnmounted(() => {
  document.removeEventListener('mousemove', onDrag)
  document.removeEventListener('mouseup', stopDrag)
})
</script>

<template>
  <div
    class="flex h-screen w-screen overflow-hidden bg-morandi-bg-primary"
    :class="{ 'select-none': isDragging }"
  >

    <!-- ========== 状态 B：全局探索模式 ========== -->
    <template v-if="showGlobalGraph">
      <div class="flex-1 flex flex-col min-w-0 overflow-hidden">
        <GlobalGraph
          :activeNodeId="store.currentNodeId"
          @exit="showGlobalGraph = false"
        />
      </div>

      <!-- 拖拽分割线 -->
      <div
        v-if="!rightPanelCollapsed"
        class="w-1 flex-shrink-0 cursor-col-resize transition-colors duration-150"
        :class="isDragging ? 'bg-morandi-accent-green' : 'bg-transparent hover:bg-gray-300'"
        @mousedown="startDrag"
      />

      <aside
        class="flex-shrink-0 border-l border-morandi-border-light bg-morandi-bg-secondary transition-all duration-300 flex flex-col"
        :style="{ width: rightPanelCollapsed ? '0px' : rightPanelWidth + 'px' }"
        :class="{ 'overflow-hidden border-0': rightPanelCollapsed }"
      >
        <LocalGraph
          :activeNodeId="store.currentNodeId"
          :graphData="relationsData"
          @enterExplore="showGlobalGraph = true"
        />
      </aside>
    </template>

    <!-- ========== 状态 A：文档阅读模式 ========== -->
    <template v-else>
      <!-- 左侧目录栏 -->
      <aside
        class="flex-shrink-0 border-r border-morandi-border-light bg-morandi-bg-secondary transition-all duration-300"
        :class="sidebarCollapsed ? 'w-0 overflow-hidden border-0' : 'w-72'"
      >
        <Sidebar />
      </aside>

      <!-- 中间内容区 -->
      <main class="flex-1 flex flex-col min-w-0 overflow-hidden">
        <header class="flex items-center justify-between h-12 px-4 border-b border-morandi-border-light bg-morandi-bg-primary flex-shrink-0">
          <div class="flex items-center gap-3">
            <button
              class="w-8 h-8 flex items-center justify-center rounded text-morandi-text-secondary hover:bg-morandi-bg-tertiary transition-colors"
              @click="sidebarCollapsed = !sidebarCollapsed"
              title="切换侧边栏"
            >
              <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 6h16M4 12h16M4 18h16" />
              </svg>
            </button>
            <SearchBar @select="showGlobalGraph = false" />
            <span class="text-sm font-medium text-morandi-text-secondary select-none hidden sm:block">机器学习知识库</span>
          </div>
          <button
            class="w-8 h-8 flex items-center justify-center rounded text-morandi-text-secondary hover:bg-morandi-bg-tertiary transition-colors"
            @click="rightPanelCollapsed = !rightPanelCollapsed"
            title="切换右侧面板"
          >
            <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <rect x="2" y="3" width="20" height="14" rx="2" />
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M8 21h8M12 17v4" />
            </svg>
          </button>
        </header>

        <section class="flex-1 overflow-y-auto px-8 py-6">
          <router-view />
        </section>
      </main>

      <!-- 拖拽分割线 -->
      <div
        v-if="!rightPanelCollapsed"
        class="w-1 flex-shrink-0 cursor-col-resize transition-colors duration-150"
        :class="isDragging ? 'bg-morandi-accent-green' : 'bg-transparent hover:bg-gray-300'"
        @mousedown="startDrag"
      />

      <!-- 右侧局部图谱 -->
      <aside
        class="flex-shrink-0 border-l border-morandi-border-light bg-morandi-bg-secondary transition-all duration-300 flex flex-col"
        :style="{ width: rightPanelCollapsed ? '0px' : rightPanelWidth + 'px' }"
        :class="{ 'overflow-hidden border-0': rightPanelCollapsed }"
      >
        <LocalGraph
          :activeNodeId="store.currentNodeId"
          :graphData="relationsData"
          @enterExplore="showGlobalGraph = true"
        />
      </aside>
    </template>

  </div>
</template>
