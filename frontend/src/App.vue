<script setup lang="ts">
import { ref } from 'vue'
import { fetchSse } from './utils/sse'

type Role = 'user' | 'assistant'
type Message = { role: Role; content: string }

const inputMessage = ref('')
const messages = ref<Message[]>([])

const ragDocId = ref('')
const ragDocName = ref('')
const ragUploading = ref(false)
const ragUploadMsg = ref('')
const ragAsking = ref(false)
const ragAbortCtrl = ref<AbortController | null>(null)

const ragHistory = ref<{ role: 'user' | 'assistant'; content: string }[]>([])


function stopStream() {
  ragAbortCtrl.value?.abort()
  ragAbortCtrl.value = null
  ragAsking.value = false
}

async function uploadRagFile(ev: Event) {
  const input = ev.target as HTMLInputElement
  const file = input.files?.[0]
  if (!file) return

  ragUploading.value = true
  ragUploadMsg.value = ''
  try {
    const fd = new FormData()
    fd.append('file', file)
    fd.append('chunk_size', '800')
    fd.append('chunk_overlap', '120')

    const resp = await fetch('/api/rag/upload', { method: 'POST', body: fd })
    if (!resp.ok) throw new Error(await resp.text())
    const data = await resp.json() as { doc_id: string; filename?: string }
    ragDocId.value = data.doc_id
    ragDocName.value = data.filename || file.name
    ragHistory.value = []
    ragUploadMsg.value = '上传成功'
  } catch (e: any) {
    ragUploadMsg.value = `上传失败：${e?.message ?? String(e)}`
  } finally {
    ragUploading.value = false
    input.value = ''
  }
}

async function agentQueryStream() {
  const q = inputMessage.value.trim()
  if (!q || ragAsking.value) return

  messages.value.push({ role: 'user', content: q })
  inputMessage.value = ''

  const assistantIndex = messages.value.push({ role: 'assistant', content: '' }) - 1
  const appendToAssistant = (textPart: string) => {
    const current = messages.value[assistantIndex]
    if (current) {
      current.content += textPart
    }
  }

  ragAsking.value = true
  ragAbortCtrl.value = new AbortController()

  try {
    await fetchSse(
      '/api/agent/stream',
      {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query: q }),
        signal: ragAbortCtrl.value.signal,
      },
      (ev) => {
        if (ev.event === 'delta') {
          appendToAssistant(ev.data)
        } else if (ev.event === 'error') {
          appendToAssistant(`\n[ERROR] ${ev.data}`)
        } else if (ev.event === 'done') {
          ragAsking.value = false
          ragAbortCtrl.value = null
        }
      },
    )
  } catch (e: any) {
    appendToAssistant(`\n[STREAM FAILED] ${e?.message ?? String(e)}`)
    ragAsking.value = false
    ragAbortCtrl.value = null
  }
}

async function sendUnified() {
  if (ragDocId.value) {
    await ragQueryStream()
    return
  }
  await agentQueryStream()
}


async function ragQueryStream() {
  const q = inputMessage.value.trim()
  if (!q || ragAsking.value) return

  messages.value.push({ role: 'user', content: q })
  ragHistory.value.push({ role: 'user', content: q })
  inputMessage.value = '' 

  const assistantIndex = messages.value.push({ role: 'assistant', content: '' }) - 1
  const appendToAssistant = (textPart: string) => {
    const current = messages.value[assistantIndex]
    if (current) {
      current.content += textPart
    }
  }

  ragAsking.value = true
  ragAbortCtrl.value = new AbortController()

  try {
    await fetchSse(
      '/api/rag/query/stream',
      {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query: q, top_k: 5, doc_id: ragDocId.value, history: ragHistory.value }),
        signal: ragAbortCtrl.value.signal,
      },
      (ev) => {
        if (ev.event === 'delta') {
          appendToAssistant(ev.data)
        } else if (ev.event === 'error') {
          appendToAssistant(`\n[ERROR] ${ev.data}`)
        } else if (ev.event === 'done') {
          const current = messages.value[assistantIndex]
          if (current?.content) {
            ragHistory.value.push({ role: 'assistant', content: current.content })
          }
          ragAsking.value = false
          ragAbortCtrl.value = null
        }
      },
    )
  } catch (e: any) {
    appendToAssistant(`\n[STREAM FAILED] ${e?.message ?? String(e)}`)
    ragAsking.value = false
    ragAbortCtrl.value = null
  }
}

</script>

<template>
  <div style="max-width: 900px; margin: 40px auto;">
    <h2>AIGC Workbench</h2>

    <div style="border: 1px solid #ddd; border-radius: 8px; padding: 12px; min-height: 360px;">
      <div v-for="(m, idx) in messages" :key="idx" style="margin: 10px 0;">
        <div style="font-size: 12px; opacity: 0.6;">{{ m.role }}</div>
        <div style="white-space: pre-wrap;">{{ m.content }}</div>
      </div>
      <div v-if="ragAsking" style="opacity: 0.7;">assistant is typing...</div>
    </div>

    <div v-if="ragDocName" style="margin-top: 8px; font-size: 12px; opacity: 0.7;">
      当前文档：{{ ragDocName }}
    </div>
    <div style="display: flex; gap: 8px; margin-top: 12px;">
      <label style="padding: 10px 14px; border: 1px solid #ddd; border-radius: 6px; cursor: pointer;">
        +
        <input
          type="file"
          accept=".pdf,.doc,.docx,.txt,.md,image/*"
          @change="uploadRagFile"
          :disabled="ragUploading"
          style="display: none;"
        />
      </label>
      <input
        v-model="inputMessage"
        @keydown.enter="sendUnified"
        placeholder="输入一句话..."
        style="flex: 1; padding: 10px; border: 1px solid #ddd; border-radius: 6px;"
      />
      <button @click="sendUnified" :disabled="ragAsking" style="padding: 10px 14px;">
        发送
      </button>
      <button @click="stopStream" :disabled="!ragAsking" style="padding: 10px 14px;">
        停止
      </button>
    </div>
  </div>
</template>
