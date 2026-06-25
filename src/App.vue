<script setup>
import { onMounted } from 'vue'
import { useCatalogStore } from '@/stores/catalog'
import Layout from '@/components/Layout.vue'

const store = useCatalogStore()

onMounted(async () => {
  await store.loadCatalog()
})
</script>

<template>
  <Layout />
</template>
