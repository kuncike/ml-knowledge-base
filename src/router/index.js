import { createRouter, createWebHistory } from 'vue-router'

const routes = [
  {
    path: '/',
    name: 'home',
    component: () => import('@/views/Home.vue'),
  },
  {
    // 动态路由 — docId 对应 content/ 下的 .md 文件名（不含扩展名）
    path: '/doc/:docId',
    name: 'doc',
    component: () => import('@/views/Home.vue'),
    props: true,
  },
]

const router = createRouter({
  history: createWebHistory(),
  routes,
})

export default router
