import { defineStore } from 'pinia'
import { ref, computed } from 'vue'

// 扁平化目录 — 从嵌套 catalog.json 提取所有叶子节点的 file -> id/title 映射
function flattenCatalog(nodes, result = {}) {
  for (const node of nodes) {
    if (node.file) {
      result[node.file] = { id: node.id, title: node.title }
    }
    if (node.children) {
      flattenCatalog(node.children, result)
    }
  }
  return result
}

export const useCatalogStore = defineStore('catalog', () => {
  const catalogTree = ref([])         // 原始嵌套目录树
  const flatMap = ref({})             // file -> { id, title }
  const currentDocId = ref(null)      // 当前选中的文档 id
  const currentNodeId = ref(null)     // 当前选中的节点 id
  const expandedIds = ref(new Set())  // 展开的节点 id 集合

  // 加载 catalog.json
  async function loadCatalog() {
    const mod = await import('@/data/catalog.json')
    catalogTree.value = mod.catalog
    flatMap.value = flattenCatalog(mod.catalog)
  }

  // 设置当前文档
  function setCurrentDoc(docId, nodeId) {
    currentDocId.value = docId
    currentNodeId.value = nodeId
  }

  // 切换节点展开/折叠
  function toggleExpand(nodeId) {
    const s = new Set(expandedIds.value)
    if (s.has(nodeId)) {
      s.delete(nodeId)
    } else {
      s.add(nodeId)
    }
    expandedIds.value = s
  }

  // 根据 file 名查找节点 id
  function findNodeIdByFile(file) {
    return flatMap.value[file]?.id ?? null
  }

  // 查找节点的祖先 id 链（用于自动展开）
  function findAncestorIds(nodes, targetId, ancestors = []) {
    for (const node of nodes) {
      if (node.id === targetId) return ancestors
      if (node.children) {
        const found = findAncestorIds(node.children, targetId, [...ancestors, node.id])
        if (found) return found
      }
    }
    return null
  }

  // 激活某个文档 — 展开所有祖先节点
  function activateDoc(docId) {
    const info = flatMap.value[docId]
    if (!info) return
    const ancestors = findAncestorIds(catalogTree.value, info.id)
    if (ancestors) {
      const s = new Set(expandedIds.value)
      ancestors.forEach((id) => s.add(id))
      s.add(info.id)
      expandedIds.value = s
    }
    setCurrentDoc(docId, info.id)
  }

  return {
    catalogTree,
    flatMap,
    currentDocId,
    currentNodeId,
    expandedIds,
    loadCatalog,
    setCurrentDoc,
    toggleExpand,
    findNodeIdByFile,
    activateDoc,
  }
})
