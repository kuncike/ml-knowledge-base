<script setup>
import { useCatalogStore } from '@/stores/catalog'
import { useRouter } from 'vue-router'

const store = useCatalogStore()
const router = useRouter()

function handleNodeClick(node) {
  if (node.file) {
    store.activateDoc(node.file)
    router.push({ name: 'doc', params: { docId: node.file.replace('.md', '') } })
  } else {
    store.toggleExpand(node.id)
  }
}

function isExpanded(nodeId) {
  return store.expandedIds.has(nodeId)
}

function isActive(nodeId) {
  return store.currentNodeId === nodeId
}
</script>

<template>
  <nav class="h-full flex flex-col select-none">
    <div class="flex items-center h-12 px-4 border-b border-morandi-border-light flex-shrink-0">
      <span class="text-sm font-semibold text-morandi-accent-green tracking-wide">目录导航</span>
    </div>
    <div class="flex-1 overflow-y-auto py-2 px-2">
      <TreeNode
        v-for="node in store.catalogTree"
        :key="node.id"
        :node="node"
        :depth="0"
        :expanded="isExpanded(node.id)"
        :active="isActive(node.id)"
        @click="handleNodeClick(node)"
      />
    </div>
  </nav>
</template>

<script>
import { h } from 'vue'

// 递归树节点组件 — 内部定义避免命名冲突
const TreeNode = {
  name: 'TreeNode',
  props: {
    node: { type: Object, required: true },
    depth: { type: Number, default: 0 },
    expanded: { type: Boolean, default: false },
    active: { type: Boolean, default: false },
  },
  emits: ['click'],
  setup(props, { emit }) {
    const store = useCatalogStore()
    const router = useRouter()

    function handleChildClick(child) {
      if (child.file) {
        store.activateDoc(child.file)
        router.push({ name: 'doc', params: { docId: child.file.replace('.md', '') } })
      } else {
        store.toggleExpand(child.id)
      }
    }

    return () => {
      const { node, depth, expanded, active } = props
      const hasChildren = node.children && node.children.length > 0
      const isLeaf = !!node.file

      return h('div', {}, [
        h('div', {
          class: [
            'flex items-center gap-1.5 py-1.5 cursor-pointer rounded transition-colors text-sm',
            isLeaf ? 'hover:bg-morandi-bg-tertiary' : 'hover:bg-morandi-bg-tertiary',
            active ? 'bg-morandi-accent-green/15 text-morandi-accent-green font-medium' : 'text-morandi-text-primary',
          ].join(' '),
          style: { paddingLeft: `${depth * 16 + 8}px`, paddingRight: '8px' },
          onClick: () => emit('click'),
        }, [
          // 展开/折叠箭头
          hasChildren
            ? h('svg', {
                class: ['w-3.5 h-3.5 flex-shrink-0 transition-transform text-morandi-text-muted', expanded ? 'rotate-90' : ''],
                fill: 'currentColor',
                viewBox: '0 0 20 20',
              }, [
                h('path', {
                  'fill-rule': 'evenodd',
                  d: 'M7.21 14.77a.75.75 0 01.02-1.06L11.168 10 7.23 6.29a.75.75 0 111.04-1.08l4.5 4.25a.75.75 0 010 1.08l-4.5 4.25a.75.75 0 01-1.06-.02z',
                  'clip-rule': 'evenodd',
                }),
              ])
            : h('span', { class: 'w-3.5 h-3.5 flex-shrink-0' }),
          // 图标
          isLeaf
            ? h('svg', {
                class: 'w-3.5 h-3.5 flex-shrink-0 text-morandi-text-muted',
                fill: 'none',
                stroke: 'currentColor',
                viewBox: '0 0 24 24',
              }, [
                h('path', {
                  'stroke-linecap': 'round',
                  'stroke-linejoin': 'round',
                  'stroke-width': '2',
                  d: 'M9 12h6m-3-3v6m-7 4h14a2 2 0 002-2V7a2 2 0 00-2-2H5a2 2 0 00-2 2v10a2 2 0 002 2z',
                }),
              ])
            : h('svg', {
                class: 'w-3.5 h-3.5 flex-shrink-0 text-morandi-accent-green',
                fill: 'none',
                stroke: 'currentColor',
                viewBox: '0 0 24 24',
              }, [
                h('path', {
                  'stroke-linecap': 'round',
                  'stroke-linejoin': 'round',
                  'stroke-width': '2',
                  d: 'M3 7v10a2 2 0 002 2h14a2 2 0 002-2V9a2 2 0 00-2-2h-6l-2-2H5a2 2 0 00-2 2z',
                }),
              ]),
          // 标题
          h('span', { class: 'truncate' }, node.title),
        ]),
        // 子节点
        hasChildren && expanded
          ? h('div', {}, node.children.map((child) =>
              h(TreeNode, {
                key: child.id,
                node: child,
                depth: depth + 1,
                expanded: store.expandedIds.has(child.id),
                active: store.currentNodeId === child.id,
                onClick: () => handleChildClick(child),
              })
            ))
          : null,
      ])
    }
  },
}

export default { components: { TreeNode } }
</script>
