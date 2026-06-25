<script setup>
import { computed } from 'vue'
import MarkdownIt from 'markdown-it'
import markdownItKatex from 'markdown-it-katex'
import hljs from 'highlight.js'
import 'katex/dist/katex.min.css'

const props = defineProps({
  content: { type: String, required: true },
})

const md = new MarkdownIt({
  html: false,
  linkify: true,
  typographer: true,
  breaks: true,
  highlight(str, lang) {
    if (lang === 'mermaid') {
      return '<pre class="mermaid-wrapper"><code class="language-mermaid">' + str + '</code></pre>'
    }
    if (lang && hljs.getLanguage(lang)) {
      try {
        return (
          '<pre class="hljs rounded-lg p-4 overflow-x-auto text-sm leading-relaxed"><code>' +
          hljs.highlight(str, { language: lang, ignoreIllegals: true }).value +
          '</code></pre>'
        )
      } catch (_) { /* fall through */ }
    }
    return '<pre class="hljs rounded-lg p-4 overflow-x-auto text-sm leading-relaxed"><code>' + md.utils.escapeHtml(str) + '</code></pre>'
  },
})

md.use(markdownItKatex, { throwOnError: false, errorColor: '#c6b8b1' })

const defaultFence = md.renderer.rules.fence
md.renderer.rules.fence = (tokens, idx, options, env, slf) => {
  const token = tokens[idx]
  const info = token.info.trim()
  if (info === 'echarts' || info.startsWith('echarts ')) {
    const encoded = btoa(unescape(encodeURIComponent(token.content)))
    return `<div class="echarts-placeholder" style="width:100%;height:400px;margin:20px 0;" data-echarts-code="${encoded}"></div>`
  }
  return defaultFence(tokens, idx, options, env, slf)
}

const rendered = computed(() => md.render(props.content))
</script>

<template>
  <article
    class="prose prose-morandi max-w-none"
    v-html="rendered"
  />
</template>
