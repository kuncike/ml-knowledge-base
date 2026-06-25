<script setup>
import { ref, computed, watch, nextTick } from 'vue'
import { useRouter } from 'vue-router'
import { useCatalogStore } from '@/stores/catalog'
import Fuse from 'fuse.js'

const emit = defineEmits(['select'])

const router = useRouter()
const store = useCatalogStore()

const query = ref('')
const selectedIndex = ref(0)
const showDropdown = ref(false)
const inputRef = ref(null)
const dropdownRef = ref(null)

let fuse = null

const searchItems = computed(() =>
  Object.entries(store.flatMap).map(([file, info]) => ({
    file,
    id: info.id,
    title: info.title,
  }))
)

function initFuse() {
  if (searchItems.value.length === 0) return
  fuse = new Fuse(searchItems.value, {
    keys: ['title', 'id'],
    threshold: 0.4,
    distance: 100,
    minMatchCharLength: 1,
  })
}

watch(() => Object.keys(store.flatMap).length, (len) => {
  if (len > 0) initFuse()
}, { immediate: true })

const results = computed(() => {
  if (!query.value.trim() || !fuse) return []
  return fuse.search(query.value.trim()).map(r => r.item)
})

watch(results, () => { selectedIndex.value = 0 })

const dropdownStyle = computed(() => {
  if (!inputRef.value) return { display: 'none' }
  const rect = inputRef.value.getBoundingClientRect()
  return {
    position: 'fixed',
    top: (rect.bottom + 4) + 'px',
    left: rect.left + 'px',
    minWidth: Math.max(rect.width, 240) + 'px',
  }
})

function select(item) {
  store.activateDoc(item.file)
  const docId = item.file.replace('.md', '')
  router.push({ name: 'doc', params: { docId } })
  emit('select')
  clear()
}

function clear() {
  query.value = ''
  selectedIndex.value = 0
  showDropdown.value = false
}

function onInput() {
  showDropdown.value = !!query.value.trim()
}

function onFocus() {
  if (query.value.trim()) showDropdown.value = true
}

function onBlur() {
  setTimeout(() => { showDropdown.value = false }, 200)
}

function onKeydown(e) {
  if (e.key === 'Escape') {
    showDropdown.value = false
    inputRef.value?.blur()
    return
  }
  if (!showDropdown.value || results.value.length === 0) return

  if (e.key === 'ArrowDown') {
    e.preventDefault()
    selectedIndex.value = Math.min(selectedIndex.value + 1, results.value.length - 1)
  } else if (e.key === 'ArrowUp') {
    e.preventDefault()
    selectedIndex.value = Math.max(selectedIndex.value - 1, 0)
  } else if (e.key === 'Enter') {
    e.preventDefault()
    const item = results.value[selectedIndex.value]
    if (item) select(item)
  }
}

function highlightText(text) {
  if (!query.value.trim()) return text
  const escaped = query.value.trim().replace(/[.*+?^${}()|[\]\\]/g, '\\$&')
  const regex = new RegExp(`(${escaped})`, 'gi')
  return text.replace(regex, '<mark class="bg-yellow-100 text-morandi-text-primary rounded-sm">$1</mark>')
}

watch(selectedIndex, () => {
  nextTick(() => {
    const el = dropdownRef.value?.querySelector('.search-item-selected')
    el?.scrollIntoView({ block: 'nearest' })
  })
})
</script>

<template>
  <div class="relative">
    <div
      class="flex items-center gap-2 px-3 py-1.5 rounded-lg bg-morandi-bg-tertiary border border-morandi-border-light focus-within:border-morandi-accent-green focus-within:bg-white transition-colors"
    >
      <svg class="w-4 h-4 text-morandi-text-muted flex-shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
      </svg>
      <input
        ref="inputRef"
        v-model="query"
        type="text"
        placeholder="搜索文章..."
        class="bg-transparent text-sm text-morandi-text-primary placeholder-morandi-text-muted outline-none w-48"
        @input="onInput"
        @focus="onFocus"
        @blur="onBlur"
        @keydown="onKeydown"
      />
      <button
        v-if="query"
        class="text-morandi-text-muted hover:text-morandi-text-secondary flex-shrink-0"
        @click="clear"
      >
        <svg class="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12" />
        </svg>
      </button>
    </div>

    <Teleport to="body">
      <!-- Results dropdown -->
      <div
        v-if="showDropdown && results.length > 0"
        ref="dropdownRef"
        :style="dropdownStyle"
        class="z-[9999] bg-white border border-morandi-border-light rounded-lg shadow-lg max-h-64 overflow-y-auto py-1"
      >
        <div
          v-for="(item, index) in results"
          :key="item.id"
          class="px-3 py-2 text-sm cursor-pointer transition-colors"
          :class="index === selectedIndex ? 'bg-morandi-bg-tertiary text-morandi-accent-green search-item-selected' : 'text-morandi-text-primary hover:bg-morandi-bg-secondary'"
          @mousedown.prevent="select(item)"
          @mouseenter="selectedIndex = index"
        >
          <div v-html="highlightText(item.title)" />
        </div>
      </div>

      <!-- No results -->
      <div
        v-if="showDropdown && query.trim() && results.length === 0"
        :style="dropdownStyle"
        class="z-[9999] bg-white border border-morandi-border-light rounded-lg shadow-lg px-4 py-3 text-sm text-center text-morandi-text-muted"
      >
        无匹配结果
      </div>
    </Teleport>
  </div>
</template>
