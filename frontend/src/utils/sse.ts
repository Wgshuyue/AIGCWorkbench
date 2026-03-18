export type SseEvent =
  | { event: 'delta'; data: string }
  | { event: 'done'; data: string }
  | { event: 'error'; data: string }
  | { event: 'meta'; data: string }

// 智能解析\n\n， 会出现当DONE之后，最后一段数据没有以 \n\n 结尾就断流了，或者是 \r\n 导致分隔没匹配到
function parseSseBlocks(buffer: string): {events: SseEvent[]; rest: string}{
    const events: SseEvent[] = []
    // SSE 事件用双换行分隔， 按块分割
    const parts = buffer.split('\n\n')
    // 最后一块可能不完整， 下一轮继续解析， 确保不会丢失跨 chunk 的 token 数据
    const rest = parts.pop() ?? ''

    console.log('拆分块---parts', parts)
    console.log('最后一不完整块--rest', rest)

    for(const block in parts){
        const lines = block.split('\n')
        console.log('行---lines', lines)
        
        let eventName = 'message'
        let dataLines: string[] = []

        /**
         *  event: delta
            data: 世界上最
            event: delta
            data: 远
            event: delta
            data: 的距离
            event: done
            data: [DONE]
         */
        for(const line of lines){
            if(line.startsWith('event:')){
                eventName = line.slice('event:'.length).trim()
            } else if (line.startsWith('data:')) {
                dataLines.push(line.slice('data:'.length).trimEnd())
            }
        }

        const data = dataLines.join('\n')
        console.log('data', data)
        if(!data) continue

        if (eventName === 'delta') events.push({ event: 'delta', data })
        else if (eventName === 'done') events.push({ event: 'done', data })
        else if (eventName === 'error') events.push({ event: 'error', data })

    }
    return {events, rest}

}

/**
 * 用正则 /\r?\n\r?\n/ 来识别块分隔符，兼容 \n\n 和 \r\n\r\
 * 
 */
function splitSseBlocks(buffer: string): { blocks: string[]; rest: string } {
  const sep = /\r?\n\r?\n/g
  const blocks: string[] = []
  let lastIndex = 0
  let match: RegExpExecArray | null

  while ((match = sep.exec(buffer)) !== null) {
    blocks.push(buffer.slice(lastIndex, match.index))
    lastIndex = match.index + match[0].length
  }

  return { blocks, rest: buffer.slice(lastIndex) }
}

function parseSseBlock(block: string): SseEvent | null {
  const lines = block.split(/\r?\n/)
  let eventName = 'message'
  const dataLines: string[] = []

  for (const line of lines) {
    if (line.startsWith('event:')) {
      eventName = line.slice('event:'.length).trim()
    } else if (line.startsWith('data:')) {
        // dataLines.push(line.slice('data:'.length).trimEnd())
        // 解决空格问题
        let v = line.slice('data:'.length)
        if (v.startsWith(' ')) v = v.slice(1)
        dataLines.push(v)
    }
  }

  const data = dataLines.join('')
  if (!data) return null

  if (eventName === 'delta') return { event: 'delta', data }
  if (eventName === 'done') return { event: 'done', data }
  if (eventName === 'error') return { event: 'error', data }
  if (eventName === 'meta') return { event: 'meta', data }
  return null
}



export async function fetchSse(url: string, options: RequestInit & {signal?: AbortSignal}, onEvent: (e: SseEvent) => void){
    const resp = await fetch(url, options)
    if (!resp.ok) {
        const text = await resp.text().catch(() => '')
        throw new Error(`HTTP ${resp.status} ${text}`)
    }
    if (!resp.body) throw new Error('ReadableStream not supported')

    // 流式读取数据
    const reader = resp.body.getReader()
    // 二进制流转字符串
    const decoder = new TextDecoder('utf-8')

    console.log('流式读取数据---reader', reader)

    let buffer = '';
    let sawDone = false

    while (true) {
        const {value, done} = await reader.read()
        if (done) {
            const flushed = splitSseBlocks(buffer + '\n\n')
             for (const b of flushed.blocks) {
                const ev = parseSseBlock(b)
                if (!ev) continue
                onEvent(ev)
                if (ev.event === 'done') sawDone = true
            }
            if (!sawDone) onEvent({ event: 'done', data: '[DONE]' })
            break
        }
        //将二进制 chunk 转成字符串, stream: true 保留未完整字符，用于下一轮拼接
        buffer += decoder.decode(value, {stream: true})
        console.log('二进制流转字符串---buffer', buffer)
        const { blocks, rest } = splitSseBlocks(buffer)
        console.log('解析拼块---blocks', blocks)
        buffer = rest

        for (const b of blocks) {
            const ev = parseSseBlock(b)
            if (!ev) continue
            onEvent(ev)
            if (ev.event === 'done') {
                sawDone = true
                try { await reader.cancel() } catch {}
                return
            }
            }

    }

}
