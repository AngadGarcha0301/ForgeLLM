import { useState, useEffect } from 'react'
import { Flame, Play, Square, Clock, CheckCircle, XCircle, Loader2 } from 'lucide-react'
import api from '../lib/api'
import { useAuth } from '../context/AuthContext'

interface TrainingJob {
  id: number
  name: string
  status: 'pending' | 'queued' | 'running' | 'completed' | 'failed'
  dataset_id: number
  base_model: string
  num_epochs?: number
  epochs?: number
  current_step?: number
  total_steps?: number
  learning_rate: number
  lora_r: number
  lora_alpha: number
  created_at: string
  progress?: number
}

interface Dataset {
  id: number
  name: string
  filename: string
}

const statusIcons = {
  pending: Clock,
  queued: Clock,
  running: Loader2,
  completed: CheckCircle,
  failed: XCircle,
}

const statusColors = {
  pending: 'text-yellow-400 bg-yellow-500/20',
  queued: 'text-yellow-400 bg-yellow-500/20',
  running: 'text-blue-400 bg-blue-500/20',
  completed: 'text-green-400 bg-green-500/20',
  failed: 'text-red-400 bg-red-500/20',
}

export default function Training() {
  const { workspace } = useAuth()
  const [jobs, setJobs] = useState<TrainingJob[]>([])
  const [datasets, setDatasets] = useState<Dataset[]>([])
  const [showNewJob, setShowNewJob] = useState(false)
  const [loading, setLoading] = useState(true)
  const [submitting, setSubmitting] = useState(false)
  const [form, setForm] = useState({
    name: '',
    dataset_id: '',
    base_model: 'mistralai/Mistral-7B-v0.1',
    epochs: 3,
    learning_rate: 0.0002,
    lora_r: 16,
    lora_alpha: 32,
  })

  useEffect(() => {
    if (workspace) {
      fetchJobs()
      fetchDatasets()
    } else {
      setLoading(false)
    }
  }, [workspace])

  // Auto-refresh jobs every 2 seconds if any job is running or queued
  useEffect(() => {
    const hasActiveJobs = jobs.some(j => j.status === 'running' || j.status === 'queued' || j.status === 'pending')
    if (!hasActiveJobs || !workspace) return

    const interval = setInterval(() => {
      fetchJobs()
    }, 2000)

    return () => clearInterval(interval)
  }, [jobs, workspace])

  const fetchJobs = async () => {
    if (!workspace) return
    try {
      const res = await api.get(`/api/v1/training/?workspace_id=${workspace.id}`)
      setJobs(res.data || [])
    } catch (err) {
      console.error('Failed to fetch jobs:', err)
      setJobs([])
    } finally {
      setLoading(false)
    }
  }

  const fetchDatasets = async () => {
    if (!workspace) return
    try {
      const res = await api.get(`/api/v1/datasets/?workspace_id=${workspace.id}`)
      setDatasets(res.data || [])
    } catch (err) {
      console.error('Failed to fetch datasets:', err)
      setDatasets([])
    }
  }

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!workspace) return
    setSubmitting(true)
    try {
      const res = await api.post('/api/v1/training/start', {
        workspace_id: workspace.id,
        dataset_id: parseInt(form.dataset_id),
        name: form.name,
        base_model: form.base_model,
        config: {
          num_epochs: form.epochs,
          learning_rate: form.learning_rate,
          lora_r: form.lora_r,
          lora_alpha: form.lora_alpha,
        }
      })
      setJobs([res.data, ...jobs])
      setShowNewJob(false)
      setForm({
        name: '',
        dataset_id: '',
        base_model: 'mistralai/Mistral-7B-v0.1',
        epochs: 3,
        learning_rate: 0.0002,
        lora_r: 16,
        lora_alpha: 32,
      })
    } catch (err) {
      console.error('Failed to create job:', err)
      alert('Failed to create training job. Please try again.')
    } finally {
      setSubmitting(false)
    }
  }

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <Loader2 className="w-8 h-8 animate-spin text-orange-500" />
      </div>
    )
  }

  return (
    <div>
      {/* Header */}
      <div className="flex items-center justify-between mb-8">
        <div>
          <h1 className="text-3xl font-bold text-white">Training</h1>
          <p className="text-gray-400 mt-2">Start and monitor LoRA fine-tuning jobs</p>
        </div>
        <button
          onClick={() => setShowNewJob(true)}
          className="flex items-center gap-2 px-4 py-2 bg-orange-600 hover:bg-orange-700 text-white rounded-lg transition-colors"
        >
          <Play className="w-5 h-5" />
          New Training Job
        </button>
      </div>

      {/* New Job Form */}
      {showNewJob && (
        <div className="mb-8 bg-gray-900 rounded-xl p-6 border border-gray-800">
          <h2 className="text-xl font-semibold text-white mb-6">Configure Training Job</h2>
          <form onSubmit={handleSubmit} className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div>
              <label className="block text-sm font-medium text-gray-400 mb-2">Job Name</label>
              <input
                type="text"
                value={form.name}
                onChange={(e) => setForm({ ...form, name: e.target.value })}
                className="w-full px-4 py-3 bg-gray-800 border border-gray-700 rounded-lg focus:outline-none focus:border-orange-500 text-white"
                placeholder="my-custom-model"
                required
              />
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-400 mb-2">Dataset</label>
              <select
                value={form.dataset_id}
                onChange={(e) => setForm({ ...form, dataset_id: e.target.value })}
                className="w-full px-4 py-3 bg-gray-800 border border-gray-700 rounded-lg focus:outline-none focus:border-orange-500 text-white"
                required
              >
                <option value="">Select a dataset</option>
                {datasets.map((d) => (
                  <option key={d.id} value={d.id}>{d.name || d.filename}</option>
                ))}
              </select>
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-400 mb-2">Base Model</label>
              <select
                value={form.base_model}
                onChange={(e) => setForm({ ...form, base_model: e.target.value })}
                className="w-full px-4 py-3 bg-gray-800 border border-gray-700 rounded-lg focus:outline-none focus:border-orange-500 text-white"
              >
                <option value="mistralai/Mistral-7B-v0.1">Mistral 7B</option>
                <option value="meta-llama/Llama-2-7b-hf">Llama 2 7B</option>
                <option value="codellama/CodeLlama-7b-hf">CodeLlama 7B</option>
              </select>
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-400 mb-2">Epochs</label>
              <input
                type="number"
                value={form.epochs}
                onChange={(e) => setForm({ ...form, epochs: parseInt(e.target.value) })}
                className="w-full px-4 py-3 bg-gray-800 border border-gray-700 rounded-lg focus:outline-none focus:border-orange-500 text-white"
                min={1}
                max={10}
              />
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-400 mb-2">Learning Rate</label>
              <input
                type="number"
                value={form.learning_rate}
                onChange={(e) => setForm({ ...form, learning_rate: parseFloat(e.target.value) })}
                className="w-full px-4 py-3 bg-gray-800 border border-gray-700 rounded-lg focus:outline-none focus:border-orange-500 text-white"
                step={0.0001}
                min={0.00001}
                max={0.01}
              />
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-400 mb-2">LoRA Rank (r)</label>
              <input
                type="number"
                value={form.lora_r}
                onChange={(e) => setForm({ ...form, lora_r: parseInt(e.target.value) })}
                className="w-full px-4 py-3 bg-gray-800 border border-gray-700 rounded-lg focus:outline-none focus:border-orange-500 text-white"
                min={4}
                max={64}
              />
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-400 mb-2">LoRA Alpha</label>
              <input
                type="number"
                value={form.lora_alpha}
                onChange={(e) => setForm({ ...form, lora_alpha: parseInt(e.target.value) })}
                className="w-full px-4 py-3 bg-gray-800 border border-gray-700 rounded-lg focus:outline-none focus:border-orange-500 text-white"
                min={8}
                max={128}
              />
            </div>
            <div className="md:col-span-2 flex justify-end gap-4">
              <button
                type="button"
                onClick={() => setShowNewJob(false)}
                className="px-4 py-2 text-gray-400 hover:text-white transition-colors"
              >
                Cancel
              </button>
              <button
                type="submit"
                disabled={submitting}
                className="flex items-center gap-2 px-6 py-2 bg-orange-600 hover:bg-orange-700 text-white rounded-lg transition-colors disabled:opacity-50"
              >
                {submitting ? <Loader2 className="w-5 h-5 animate-spin" /> : <Flame className="w-5 h-5" />}
                Start Training
              </button>
            </div>
          </form>
        </div>
      )}

      {/* Jobs List */}
      <div className="space-y-4">
        {jobs.length === 0 ? (
          <div className="bg-gray-900 rounded-xl p-12 border border-gray-800 text-center">
            <Flame className="w-16 h-16 text-gray-600 mx-auto mb-4" />
            <p className="text-gray-400 text-lg">No training jobs yet</p>
            <p className="text-gray-500 mt-2">Start your first LoRA fine-tuning job</p>
          </div>
        ) : (
          jobs.map((job) => {
            const StatusIcon = statusIcons[job.status] || Clock
            const epochs = job.num_epochs || job.epochs || 3
            return (
              <div
                key={job.id}
                className="bg-gray-900 rounded-xl p-6 border border-gray-800 hover:border-gray-700 transition-colors"
              >
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-4">
                    <div className={`p-3 rounded-lg ${statusColors[job.status] || 'text-gray-400 bg-gray-500/20'}`}>
                      <StatusIcon className={`w-6 h-6 ${job.status === 'running' ? 'animate-spin' : ''}`} />
                    </div>
                    <div>
                      <h3 className="text-lg font-semibold text-white">{job.name}</h3>
                      <p className="text-sm text-gray-500">
                        {job.base_model} • {epochs} epochs • r={job.lora_r}
                      </p>
                    </div>
                  </div>
                  <div className="text-right">
                    <span className={`inline-block px-3 py-1 rounded-full text-sm ${statusColors[job.status] || 'text-gray-400 bg-gray-500/20'}`}>
                      {job.status}
                    </span>
                    <p className="text-xs text-gray-500 mt-2">
                      {new Date(job.created_at).toLocaleString()}
                    </p>
                  </div>
                </div>
                {job.status === 'running' && job.progress !== undefined && (
                  <div className="mt-4">
                    <div className="flex justify-between text-sm text-gray-400 mb-2">
                      <span>Progress</span>
                      <span>{job.progress}%</span>
                    </div>
                    <div className="w-full bg-gray-800 rounded-full h-2">
                      <div
                        className="bg-orange-500 h-2 rounded-full transition-all"
                        style={{ width: `${job.progress}%` }}
                      />
                    </div>
                  </div>
                )}
              </div>
            )
          })
        )}
      </div>
    </div>
  )
}
