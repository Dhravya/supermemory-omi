import { Hono } from 'hono'
import { OpenAI } from 'openai'
import { ChromaClient } from 'chromadb'

type Env = {
  OPENAI_API_KEY: string
  CHROMA_API_KEY: string
}

// Types
interface MessageBuffer {
  messages: string[]
  triggerDetected: boolean
  triggerTime: number
  collectedQuestion: string[]
  responseSent: boolean
  partialTrigger: boolean
  partialTriggerTime: number
  lastActivity: number
}

interface BufferStore {
  [key: string]: MessageBuffer
}

interface Memory {
  id: number
  created_at: string
  started_at: string
  finished_at: string
  transcript: string
  transcript_segments: {
    text: string
    speaker: string
    speakerId: number
    is_user: boolean
    start: number
    end: number
  }[]
  photos: any[]
  structured: {
    title: string
    overview: string
    emoji: string
    category: string
    action_items: {
      description: string
      completed: boolean
    }[]
    events: any[]
  }
  apps_response: {
    app_id: string
    content: string
  }[]
  discarded: boolean
}

// Constants
const TRIGGER_PHRASES = ['forgot', 'don\'t remember']
const SIMILARITY_THRESHOLD = 0.4
const BUFFER_TIMEOUT = 5000 // 5 seconds timeout to collect full sentences
const QUESTION_COLLECTION_TIMEOUT = 2000 // 2 seconds to collect full question

class MemoryManager {
  private client: ChromaClient
  private collection?: Awaited<ReturnType<ChromaClient['createCollection']>>
  private openai: OpenAI
  private buffers: BufferStore = {}
  private env: Env

  constructor(env: Env) {
    this.env =  env
    this.openai = new OpenAI({
      apiKey: this.env.OPENAI_API_KEY
    })
    this.client = new ChromaClient({
      path: "https://api.trychroma.com:8000",
      auth: { provider: "token", credentials: this.env.CHROMA_API_KEY, tokenHeaderType: "X_CHROMA_TOKEN" },
      tenant: '68b9e836-b043-45de-b783-19888f25e82c',
      database: 'omi-hackathon'
    });
    this.initializeChroma()
  }

  private async initializeChroma() {
    this.collection = await this.client.getOrCreateCollection({
      name: 'memory_store'
    })
  }

  getBuffer(sessionId: string): MessageBuffer {
    if (!this.buffers[sessionId]) {
      this.buffers[sessionId] = {
        messages: [],
        triggerDetected: false,
        triggerTime: 0,
        collectedQuestion: [],
        responseSent: false,
        partialTrigger: false,
        partialTriggerTime: 0,
        lastActivity: Date.now()
      }
    }
    return this.buffers[sessionId]
  }

  async processBuffer(sessionId: string) {
    const buffer = this.getBuffer(sessionId)
    if (buffer.messages.length > 0) {
      const fullText = buffer.messages.join(' ')
      await this.saveMemory(fullText, sessionId)
      buffer.messages = []
    }
  }

  async saveMemory(text: string, sessionId: string) {
    const embedding = await this.getEmbedding(text)
    if (!this.collection) {
      await this.initializeChroma()
    }
    await this.collection?.add({
      ids: [Date.now().toString()],
      embeddings: [embedding],
      metadatas: [{ sessionId, timestamp: new Date().toISOString() }],
      documents: [text]
    })
  }

  async searchMemories(query: string, sessionId: string) {
    const queryEmbedding = await this.getEmbedding(query)
    if (!this.collection) {
      await this.initializeChroma()
    }
    const results = await this.collection?.query({
      queryEmbeddings: [queryEmbedding],
      nResults: 5,
      where: {
        sessionId
      }
    })
    return results
  }

  private async getEmbedding(text: string) {
    const response = await this.openai.embeddings.create({
      model: 'text-embedding-ada-002',
      input: text
    })
    return response.data[0].embedding
  }

  async generateResponse(query: string, context: string) {
    return await this.openai.chat.completions.create({
      model: 'gpt-4',
      messages: [
        {
          role: 'system',
          content: 'You are a helpful AI assistant that helps recall memories and information. Use the provided context to answer questions. your answers should be as short and concise as possible. be very friendly and talk like a teenager helping a friend out. speak in lowercase.'
        },
        {
          role: 'user',
          content: `Context: ${context}\n\nQuestion: ${query}`
        }
      ],
      temperature: 0.7,
      max_tokens: 150
    })
  }

  async processMemoryObject(memory: Memory, uid: string) {
    // Save transcript
    await this.saveMemory(memory.transcript, uid)

    // Save structured data
    const structuredText = `Title: ${memory.structured.title}\nOverview: ${memory.structured.overview}\nCategory: ${memory.structured.category}`
    await this.saveMemory(structuredText, uid)

    // Save action items
    if (memory.structured.action_items.length > 0) {
      const actionItems = memory.structured.action_items
        .map(item => item.description)
        .join('\n')
      await this.saveMemory(`Action Items:\n${actionItems}`, uid)
    }

    // Save app responses
    if (memory.apps_response.length > 0) {
      const appResponses = memory.apps_response
        .map(response => response.content)
        .join('\n')
      await this.saveMemory(appResponses, uid)
    }
  }
}

const app = new Hono<{ Bindings: Env}>()

// Logging middleware
app.use('*', async (c, next) => {
  const start = Date.now()
  const method = c.req.method
  const path = c.req.url
  
  // Log request
  console.log(`[${new Date().toISOString()}] ${method} ${path}`)
  if (method === 'POST') {
    const body = await c.req.json()
    console.log('Request body:', JSON.stringify(body))
    c.req.raw = new Request(c.req.url, {
      method: c.req.method,
      headers: c.req.raw.headers,
      body: JSON.stringify(body)
    })
  }

  await next()

  // Log response
  const duration = Date.now() - start
  console.log(`[${new Date().toISOString()}] ${method} ${path} completed in ${duration}ms`)
})

app.post('/webhook', async (c) => {
  const data = await c.req.json()
  const sessionId = data.session_id
  
  if (!sessionId) {
    return c.json({ error: 'No session ID provided' }, 400)
  }


  // ideally we need to maintain one signle state of memorymanager
  const memoryManager = new MemoryManager(c.env)

  const segments = data.segments || []
  const buffer = memoryManager.getBuffer(sessionId)
  
  for (const segment of segments) {
    const text = segment.text.toLowerCase()
    buffer.messages.push(text)
    buffer.lastActivity = Date.now()

    // If we detect a sentence ending or significant pause
    if (text.match(/[.!?]$/) || (segments.length === 1 && text.length > 50)) {
      await memoryManager.processBuffer(sessionId)
    }

    // Check if it's a memory recall question
    if (TRIGGER_PHRASES.some(phrase => text.includes(phrase))) {
      if (!buffer.triggerDetected) {
        buffer.triggerDetected = true
        buffer.triggerTime = Date.now()
        buffer.collectedQuestion = [text]
        
        // Wait for the full question
        setTimeout(async () => {
          if (buffer.triggerDetected && !buffer.responseSent) {
            const fullQuestion = buffer.collectedQuestion.join(' ')
            const searchResults = await memoryManager.searchMemories(fullQuestion, sessionId)

            // @ts-ignore
            if (searchResults?.documents?.[0] && searchResults?.distances?.[0]?.[0] <= SIMILARITY_THRESHOLD) {
              if (searchResults.documents[0].length === 0) {
                buffer.responseSent = true
                return c.json({
                  message: 'I don\'t remember anything about that. sorry!'
                })
              }
              const context = searchResults.documents[0].join('\n')
              const response = await memoryManager.generateResponse(fullQuestion, context)
              
              buffer.responseSent = true
              return c.json({
                message: response.choices[0].message.content
              })
            }
            
            buffer.triggerDetected = false
            buffer.collectedQuestion = []
          }
        }, QUESTION_COLLECTION_TIMEOUT)
      } else {
        buffer.collectedQuestion.push(text)
      }
    }
  }

  // Process any remaining buffer after a timeout
  setTimeout(async () => {
    const buffer = memoryManager.getBuffer(sessionId)
    if (Date.now() - buffer.lastActivity >= BUFFER_TIMEOUT) {
      await memoryManager.processBuffer(sessionId)
    }
  }, BUFFER_TIMEOUT)

  return c.json({ status: 'success' })
})

app.get('/webhook/setup-status', (c) => {
  return c.json({
    is_setup_completed: true
  })
})

app.post("/webhook/memory", async (c) => {
  const uid = new URL(c.req.url).searchParams.get('uid')
  if (!uid) {
    return c.json({ error: 'No user ID provided' }, 400)
  }

  const memory = await c.req.json() as Memory

  const memoryManager = new MemoryManager(c.env)
  await memoryManager.processMemoryObject(memory, uid)

  return c.json({ status: 'success' })
})

app.get("/", (c) => {
  return c.json({
    message: "Hi! Welcome to supermemory for omi",
    memoryCreationEndpoint: "/webhook/memory",
    transcriptionsEndpoint: "/webhook",
    setupStatusEndpoint: "/webhook/setup-status"
  })
})

export default {
    port: 3002,
    fetch: app.fetch
}
