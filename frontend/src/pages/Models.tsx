import { useState, useEffect } from 'react'
import { Box, Download, Trash2, Info, Loader2 } from 'lucide-react'
import api from '../lib/api'
import { useAuth } from '../context/AuthContext'

interface Model {
  id: string
  name: string
  base_model: string
  training_job_id: string
  adapter_path: string
  created_at: string
  metrics?: {
    loss: number
    accuracy?: number
  }
}

export default function Models() {
  const { workspace } = useAuth()
  const [models, setModels] = useState<Model[]>([])
  const [loading, setLoading] = useState(true)
  const [selectedModel, setSelectedModel] = useState<Model | null>(null)

  useEffect(() => {
    if (workspace) {
      fetchModels()
    } else {
      setLoading(false)
    }
  }, [workspace])

  const fetchModels = async () => {
    if (!workspace) return
    try {
      const res = await api.get(`/api/v1/models/?workspace_id=${workspace.id}`)
      setModels(res.data)
    } catch (err) {
      console.error('Failed to fetch models:', err)
    } finally {
      setLoading(false)
    }
  }

  const handleDelete = async (id: string) => {
    try {
      await api.delete(`/api/v1/models/${id}`)
      setModels(models.filter((m) => m.id !== id))
    } catch (err) {
      console.error('Failed to delete model:', err)
    }
  }

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <Loader2 className="w-8 h-8 text-orange-400 animate-spin" />
      </div>
    )
  }

  return (
    <div>
      {/* Header */}
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-white">Models</h1>
        <p className="text-gray-400 mt-2">Manage your fine-tuned LoRA adapters</p>
      </div>

      {/* Models Grid */}
      {models.length === 0 ? (
        <div className="bg-gray-900 rounded-xl p-12 border border-gray-800 text-center">
          <Box className="w-16 h-16 text-gray-600 mx-auto mb-4" />
          <p className="text-gray-400 text-lg">No models yet</p>
          <p className="text-gray-500 mt-2">Complete a training job to create a model</p>
        </div>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {models.map((model) => (
            <div
              key={model.id}
              className="bg-gray-900 rounded-xl p-6 border border-gray-800 hover:border-orange-500/50 transition-all"
            >
              <div className="flex items-start justify-between mb-4">
                <div className="p-3 rounded-lg bg-purple-500/20">
                  <Box className="w-6 h-6 text-purple-400" />
                </div>
                <div className="flex items-center gap-2">
                  <button
                    onClick={() => setSelectedModel(model)}
                    className="p-2 text-gray-400 hover:text-white transition-colors"
                    title="Info"
                  >
                    <Info className="w-4 h-4" />
                  </button>
                  <button
                    className="p-2 text-gray-400 hover:text-blue-400 transition-colors"
                    title="Download"
                  >
                    <Download className="w-4 h-4" />
                  </button>
                  <button
                    onClick={() => handleDelete(model.id)}
                    className="p-2 text-gray-400 hover:text-red-400 transition-colors"
                    title="Delete"
                  >
                    <Trash2 className="w-4 h-4" />
                  </button>
                </div>
              </div>
              <h3 className="text-lg font-semibold text-white mb-2">{model.name}</h3>
              <p className="text-sm text-gray-500 mb-4">{model.base_model}</p>
              {model.metrics && (
                <div className="flex gap-4 text-sm">
                  <div>
                    <span className="text-gray-500">Loss:</span>
                    <span className="text-white ml-1">{model.metrics.loss.toFixed(4)}</span>
                  </div>
                  {model.metrics.accuracy && (
                    <div>
                      <span className="text-gray-500">Accuracy:</span>
                      <span className="text-white ml-1">{(model.metrics.accuracy * 100).toFixed(1)}%</span>
                    </div>
                  )}
                </div>
              )}
              <p className="text-xs text-gray-600 mt-4">
                Created {new Date(model.created_at).toLocaleDateString()}
              </p>
            </div>
          ))}
        </div>
      )}

      {/* Model Details Modal */}
      {selectedModel && (
        <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
          <div className="bg-gray-900 rounded-xl p-6 border border-gray-800 w-full max-w-lg mx-4">
            <div className="flex items-center justify-between mb-6">
              <h2 className="text-xl font-semibold text-white">{selectedModel.name}</h2>
              <button
                onClick={() => setSelectedModel(null)}
                className="text-gray-400 hover:text-white"
              >
                ✕
              </button>
            </div>
            <div className="space-y-4">
              <div>
                <label className="text-sm text-gray-500">Base Model</label>
                <p className="text-white">{selectedModel.base_model}</p>
              </div>
              <div>
                <label className="text-sm text-gray-500">Adapter Path</label>
                <p className="text-white font-mono text-sm">{selectedModel.adapter_path}</p>
              </div>
              <div>
                <label className="text-sm text-gray-500">Training Job ID</label>
                <p className="text-white font-mono text-sm">{selectedModel.training_job_id}</p>
              </div>
              {selectedModel.metrics && (
                <div>
                  <label className="text-sm text-gray-500">Metrics</label>
                  <pre className="bg-gray-950 rounded-lg p-4 text-sm text-gray-300 mt-2">
                    {JSON.stringify(selectedModel.metrics, null, 2)}
                  </pre>
                </div>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
