import { useState, useEffect, useRef } from 'react'
import { Send, Bot, User, Loader2, Settings, ChevronDown } from 'lucide-react'
import api from '../lib/api'
import { useAuth } from '../context/AuthContext'

interface Model {
  id: string
  name: string
  base_model: string
}

interface Message {
  role: 'user' | 'assistant'
  content: string
}

export default function Inference() {
  const { workspace } = useAuth()
  const [models, setModels] = useState<Model[]>([])
  const [selectedModel, setSelectedModel] = useState<string>('')
  const [messages, setMessages] = useState<Message[]>([])
  const [input, setInput] = useState('')
  const [loading, setLoading] = useState(false)
  const [showSettings, setShowSettings] = useState(false)
  const [settings, setSettings] = useState({
    max_tokens: 256,
    temperature: 0.7,
    top_p: 0.9,
  })
  const messagesEndRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    if (workspace) {
      fetchModels()
    }
  }, [workspace])

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  const fetchModels = async () => {
    if (!workspace) return
    try {
      const res = await api.get(`/api/v1/models/?workspace_id=${workspace.id}`)
      setModels(res.data)
      if (res.data.length > 0) {
        setSelectedModel(res.data[0].id)
      }
    } catch (err) {
      console.error('Failed to fetch models:', err)
    }
  }

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!input.trim() || !selectedModel || !workspace) return

    const userMessage = input.trim()
    setInput('')
    setMessages((prev) => [...prev, { role: 'user', content: userMessage }])
    setLoading(true)

    try {
      const res = await api.post(`/api/v1/inference/predict?workspace_id=${workspace.id}`, {
        model_id: selectedModel,
        prompt: userMessage,
        ...settings,
      })
      setMessages((prev) => [...prev, { role: 'assistant', content: res.data.generated_text }])
    } catch (err) {
      console.error('Inference failed:', err)
      setMessages((prev) => [
        ...prev,
        { role: 'assistant', content: 'Sorry, an error occurred. Please try again.' },
      ])
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="h-[calc(100vh-8rem)] flex flex-col">
      {/* Header */}
      <div className="flex items-center justify-between mb-6">
        <div>
          <h1 className="text-3xl font-bold text-white">Inference</h1>
          <p className="text-gray-400 mt-1">Test your fine-tuned models</p>
        </div>
        <div className="flex items-center gap-4">
          {/* Model Selector */}
          <div className="relative">
            <select
              value={selectedModel}
              onChange={(e) => setSelectedModel(e.target.value)}
              className="appearance-none px-4 py-2 pr-10 bg-gray-800 border border-gray-700 rounded-lg focus:outline-none focus:border-orange-500 text-white"
            >
              <option value="">Select a model</option>
              {models.map((m) => (
                <option key={m.id} value={m.id}>{m.name}</option>
              ))}
            </select>
            <ChevronDown className="absolute right-3 top-1/2 -translate-y-1/2 w-4 h-4 text-gray-400 pointer-events-none" />
          </div>
          {/* Settings */}
          <button
            onClick={() => setShowSettings(!showSettings)}
            className={`p-2 rounded-lg transition-colors ${
              showSettings ? 'bg-orange-600 text-white' : 'bg-gray-800 text-gray-400 hover:text-white'
            }`}
          >
            <Settings className="w-5 h-5" />
          </button>
        </div>
      </div>

      {/* Settings Panel */}
      {showSettings && (
        <div className="mb-4 p-4 bg-gray-900 rounded-xl border border-gray-800">
          <div className="grid grid-cols-3 gap-4">
            <div>
              <label className="block text-sm text-gray-400 mb-2">Max Tokens</label>
              <input
                type="number"
                value={settings.max_tokens}
                onChange={(e) => setSettings({ ...settings, max_tokens: parseInt(e.target.value) })}
                className="w-full px-3 py-2 bg-gray-800 border border-gray-700 rounded-lg text-white text-sm"
                min={1}
                max={2048}
              />
            </div>
            <div>
              <label className="block text-sm text-gray-400 mb-2">Temperature</label>
              <input
                type="number"
                value={settings.temperature}
                onChange={(e) => setSettings({ ...settings, temperature: parseFloat(e.target.value) })}
                className="w-full px-3 py-2 bg-gray-800 border border-gray-700 rounded-lg text-white text-sm"
                min={0}
                max={2}
                step={0.1}
              />
            </div>
            <div>
              <label className="block text-sm text-gray-400 mb-2">Top P</label>
              <input
                type="number"
                value={settings.top_p}
                onChange={(e) => setSettings({ ...settings, top_p: parseFloat(e.target.value) })}
                className="w-full px-3 py-2 bg-gray-800 border border-gray-700 rounded-lg text-white text-sm"
                min={0}
                max={1}
                step={0.1}
              />
            </div>
          </div>
        </div>
      )}

      {/* Chat Area */}
      <div className="flex-1 bg-gray-900 rounded-xl border border-gray-800 flex flex-col overflow-hidden">
        {/* Messages */}
        <div className="flex-1 overflow-y-auto p-6 space-y-6">
          {messages.length === 0 ? (
            <div className="h-full flex flex-col items-center justify-center text-center">
              <Bot className="w-16 h-16 text-gray-600 mb-4" />
              <p className="text-gray-400 text-lg">Start a conversation</p>
              <p className="text-gray-500 text-sm mt-2">
                Select a model and type your prompt below
              </p>
            </div>
          ) : (
            messages.map((message, i) => (
              <div
                key={i}
                className={`flex gap-4 ${message.role === 'user' ? 'justify-end' : ''}`}
              >
                {message.role === 'assistant' && (
                  <div className="w-10 h-10 rounded-lg bg-orange-600 flex items-center justify-center flex-shrink-0">
                    <Bot className="w-6 h-6 text-white" />
                  </div>
                )}
                <div
                  className={`max-w-[70%] rounded-xl p-4 ${
                    message.role === 'user'
                      ? 'bg-orange-600 text-white'
                      : 'bg-gray-800 text-gray-100'
                  }`}
                >
                  <p className="whitespace-pre-wrap">{message.content}</p>
                </div>
                {message.role === 'user' && (
                  <div className="w-10 h-10 rounded-lg bg-gray-700 flex items-center justify-center flex-shrink-0">
                    <User className="w-6 h-6 text-gray-300" />
                  </div>
                )}
              </div>
            ))
          )}
          {loading && (
            <div className="flex gap-4">
              <div className="w-10 h-10 rounded-lg bg-orange-600 flex items-center justify-center flex-shrink-0">
                <Bot className="w-6 h-6 text-white" />
              </div>
              <div className="bg-gray-800 rounded-xl p-4">
                <Loader2 className="w-5 h-5 text-orange-400 animate-spin" />
              </div>
            </div>
          )}
          <div ref={messagesEndRef} />
        </div>

        {/* Input */}
        <form onSubmit={handleSubmit} className="p-4 border-t border-gray-800">
          <div className="flex gap-4">
            <input
              type="text"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              placeholder={selectedModel ? "Type your message..." : "Select a model first"}
              disabled={!selectedModel || loading}
              className="flex-1 px-4 py-3 bg-gray-800 border border-gray-700 rounded-lg focus:outline-none focus:border-orange-500 text-white placeholder-gray-500 disabled:opacity-50"
            />
            <button
              type="submit"
              disabled={!selectedModel || !input.trim() || loading}
              className="px-6 py-3 bg-orange-600 hover:bg-orange-700 text-white rounded-lg transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
            >
              <Send className="w-5 h-5" />
            </button>
          </div>
        </form>
      </div>
    </div>
  )
}
