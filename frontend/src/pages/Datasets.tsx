import { useState, useRef, useEffect } from 'react'
import { Upload, File, Trash2, Eye, Loader2, Plus } from 'lucide-react'
import api from '../lib/api'
import { useAuth } from '../context/AuthContext'

interface Dataset {
  id: string
  name: string
  filename: string
  size: number
  row_count: number
  created_at: string
}

export default function Datasets() {
  const { workspace } = useAuth()
  const [datasets, setDatasets] = useState<Dataset[]>([])
  const [uploading, setUploading] = useState(false)
  const [dragOver, setDragOver] = useState(false)
  const [loading, setLoading] = useState(true)
  const fileInputRef = useRef<HTMLInputElement>(null)

  useEffect(() => {
    if (workspace) {
      fetchDatasets()
    } else {
      setLoading(false)
    }
  }, [workspace])

  const fetchDatasets = async () => {
    if (!workspace) return
    try {
      const res = await api.get(`/api/v1/datasets/?workspace_id=${workspace.id}`)
      setDatasets(res.data)
    } catch (err) {
      console.error('Failed to fetch datasets:', err)
    } finally {
      setLoading(false)
    }
  }

  const handleUpload = async (file: File) => {
    if (!workspace) return
    setUploading(true)
    const formData = new FormData()
    formData.append('file', file)

    try {
      const res = await api.post(`/api/v1/datasets/upload?workspace_id=${workspace.id}`, formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
      })
      setDatasets([res.data, ...datasets])
    } catch (err) {
      console.error('Upload failed:', err)
    } finally {
      setUploading(false)
    }
  }

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault()
    setDragOver(false)
    const file = e.dataTransfer.files[0]
    if (file && file.name.endsWith('.jsonl')) {
      handleUpload(file)
    }
  }

  const handleDelete = async (id: string) => {
    if (!workspace) return
    try {
      await api.delete(`/api/v1/datasets/${id}?workspace_id=${workspace.id}`)
      setDatasets(datasets.filter((d) => d.id !== id))
    } catch (err) {
      console.error('Delete failed:', err)
    }
  }

  const formatSize = (bytes: number) => {
    if (bytes < 1024) return bytes + ' B'
    if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB'
    return (bytes / (1024 * 1024)).toFixed(1) + ' MB'
  }

  return (
    <div>
      {/* Header */}
      <div className="flex items-center justify-between mb-8">
        <div>
          <h1 className="text-3xl font-bold text-white">Datasets</h1>
          <p className="text-gray-400 mt-2">Upload and manage your training datasets</p>
        </div>
        <button
          onClick={() => fileInputRef.current?.click()}
          className="flex items-center gap-2 px-4 py-2 bg-orange-600 hover:bg-orange-700 text-white rounded-lg transition-colors"
        >
          <Plus className="w-5 h-5" />
          Upload Dataset
        </button>
      </div>

      {/* Upload Zone */}
      <div
        onDragOver={(e) => {
          e.preventDefault()
          setDragOver(true)
        }}
        onDragLeave={() => setDragOver(false)}
        onDrop={handleDrop}
        className={`mb-8 border-2 border-dashed rounded-xl p-8 text-center transition-colors ${
          dragOver
            ? 'border-orange-500 bg-orange-500/10'
            : 'border-gray-700 hover:border-gray-600'
        }`}
      >
        <input
          ref={fileInputRef}
          type="file"
          accept=".jsonl"
          onChange={(e) => e.target.files?.[0] && handleUpload(e.target.files[0])}
          className="hidden"
        />
        {uploading ? (
          <div className="flex flex-col items-center">
            <Loader2 className="w-12 h-12 text-orange-400 animate-spin mb-4" />
            <p className="text-gray-400">Uploading...</p>
          </div>
        ) : (
          <>
            <Upload className="w-12 h-12 text-gray-500 mx-auto mb-4" />
            <p className="text-gray-300 mb-2">Drag and drop your JSONL file here</p>
            <p className="text-gray-500 text-sm">
              or{' '}
              <button
                onClick={() => fileInputRef.current?.click()}
                className="text-orange-400 hover:text-orange-300 transition-colors"
              >
                browse files
              </button>
            </p>
          </>
        )}
      </div>

      {/* Dataset Format Example */}
      <div className="mb-8 bg-gray-900 rounded-xl p-6 border border-gray-800">
        <h3 className="text-lg font-semibold text-white mb-3">Expected Format</h3>
        <pre className="bg-gray-950 rounded-lg p-4 text-sm text-gray-300 overflow-x-auto">
{`{"instruction": "Summarize this article", "input": "Long article text...", "output": "Summary..."}
{"instruction": "Translate to French", "input": "Hello world", "output": "Bonjour le monde"}`}
        </pre>
      </div>

      {/* Datasets List */}
      <div className="bg-gray-900 rounded-xl border border-gray-800 overflow-hidden">
        <table className="w-full">
          <thead>
            <tr className="border-b border-gray-800">
              <th className="px-6 py-4 text-left text-sm font-medium text-gray-400">Name</th>
              <th className="px-6 py-4 text-left text-sm font-medium text-gray-400">Rows</th>
              <th className="px-6 py-4 text-left text-sm font-medium text-gray-400">Size</th>
              <th className="px-6 py-4 text-left text-sm font-medium text-gray-400">Created</th>
              <th className="px-6 py-4 text-right text-sm font-medium text-gray-400">Actions</th>
            </tr>
          </thead>
          <tbody>
            {datasets.length === 0 ? (
              <tr>
                <td colSpan={5} className="px-6 py-12 text-center text-gray-500">
                  <File className="w-12 h-12 mx-auto mb-4 text-gray-600" />
                  <p>No datasets uploaded yet</p>
                  <p className="text-sm mt-1">Upload a JSONL file to get started</p>
                </td>
              </tr>
            ) : (
              datasets.map((dataset) => (
                <tr key={dataset.id} className="border-b border-gray-800 last:border-0 hover:bg-gray-800/50">
                  <td className="px-6 py-4">
                    <div className="flex items-center gap-3">
                      <File className="w-5 h-5 text-blue-400" />
                      <div>
                        <p className="text-white font-medium">{dataset.name}</p>
                        <p className="text-sm text-gray-500">{dataset.filename}</p>
                      </div>
                    </div>
                  </td>
                  <td className="px-6 py-4 text-gray-300">{dataset.row_count?.toLocaleString()}</td>
                  <td className="px-6 py-4 text-gray-300">{formatSize(dataset.size)}</td>
                  <td className="px-6 py-4 text-gray-300">
                    {new Date(dataset.created_at).toLocaleDateString()}
                  </td>
                  <td className="px-6 py-4 text-right">
                    <div className="flex items-center justify-end gap-2">
                      <button className="p-2 text-gray-400 hover:text-white transition-colors" title="Preview">
                        <Eye className="w-4 h-4" />
                      </button>
                      <button
                        onClick={() => handleDelete(dataset.id)}
                        className="p-2 text-gray-400 hover:text-red-400 transition-colors"
                        title="Delete"
                      >
                        <Trash2 className="w-4 h-4" />
                      </button>
                    </div>
                  </td>
                </tr>
              ))
            )}
          </tbody>
        </table>
      </div>
    </div>
  )
}
