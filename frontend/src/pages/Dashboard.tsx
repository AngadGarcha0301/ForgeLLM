import { useAuth } from '../context/AuthContext'
import { Database, Flame, Box, MessageSquare, TrendingUp, Clock } from 'lucide-react'

const stats = [
  { label: 'Datasets', value: '3', icon: Database, color: 'from-blue-500 to-blue-600' },
  { label: 'Training Jobs', value: '5', icon: Flame, color: 'from-orange-500 to-red-500' },
  { label: 'Models', value: '2', icon: Box, color: 'from-purple-500 to-purple-600' },
  { label: 'Inferences', value: '124', icon: MessageSquare, color: 'from-green-500 to-green-600' },
]

const recentActivity = [
  { action: 'Model training completed', model: 'customer-support-v2', time: '2 hours ago', status: 'success' },
  { action: 'Dataset uploaded', model: 'qa-pairs.jsonl', time: '5 hours ago', status: 'info' },
  { action: 'Training started', model: 'code-assistant-v1', time: '1 day ago', status: 'pending' },
  { action: 'Inference request', model: 'customer-support-v1', time: '2 days ago', status: 'success' },
]

export default function Dashboard() {
  const { user } = useAuth()

  return (
    <div>
      {/* Header */}
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-white">
          Welcome back, <span className="gradient-text">{user?.username}</span>
        </h1>
        <p className="text-gray-400 mt-2">Here's what's happening with your models</p>
      </div>

      {/* Stats Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
        {stats.map(({ label, value, icon: Icon, color }) => (
          <div
            key={label}
            className="bg-gray-900 rounded-xl p-6 border border-gray-800 hover:border-gray-700 transition-colors"
          >
            <div className="flex items-center justify-between">
              <div>
                <p className="text-gray-400 text-sm">{label}</p>
                <p className="text-3xl font-bold text-white mt-1">{value}</p>
              </div>
              <div className={`w-12 h-12 rounded-lg bg-gradient-to-br ${color} flex items-center justify-center`}>
                <Icon className="w-6 h-6 text-white" />
              </div>
            </div>
          </div>
        ))}
      </div>

      {/* Content Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Recent Activity */}
        <div className="bg-gray-900 rounded-xl p-6 border border-gray-800">
          <div className="flex items-center gap-2 mb-6">
            <Clock className="w-5 h-5 text-orange-400" />
            <h2 className="text-xl font-semibold text-white">Recent Activity</h2>
          </div>
          <div className="space-y-4">
            {recentActivity.map((activity, i) => (
              <div key={i} className="flex items-center justify-between py-3 border-b border-gray-800 last:border-0">
                <div>
                  <p className="text-white">{activity.action}</p>
                  <p className="text-sm text-gray-500">{activity.model}</p>
                </div>
                <div className="text-right">
                  <span
                    className={`inline-block px-2 py-1 rounded-full text-xs ${
                      activity.status === 'success'
                        ? 'bg-green-500/20 text-green-400'
                        : activity.status === 'pending'
                        ? 'bg-yellow-500/20 text-yellow-400'
                        : 'bg-blue-500/20 text-blue-400'
                    }`}
                  >
                    {activity.status}
                  </span>
                  <p className="text-xs text-gray-500 mt-1">{activity.time}</p>
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Quick Actions */}
        <div className="bg-gray-900 rounded-xl p-6 border border-gray-800">
          <div className="flex items-center gap-2 mb-6">
            <TrendingUp className="w-5 h-5 text-orange-400" />
            <h2 className="text-xl font-semibold text-white">Quick Actions</h2>
          </div>
          <div className="grid grid-cols-2 gap-4">
            <button className="p-4 bg-gray-800 hover:bg-gray-750 rounded-lg border border-gray-700 hover:border-orange-500/50 transition-all group">
              <Database className="w-8 h-8 text-blue-400 mb-2 group-hover:scale-110 transition-transform" />
              <p className="text-white font-medium">Upload Dataset</p>
              <p className="text-xs text-gray-500 mt-1">JSONL format</p>
            </button>
            <button className="p-4 bg-gray-800 hover:bg-gray-750 rounded-lg border border-gray-700 hover:border-orange-500/50 transition-all group">
              <Flame className="w-8 h-8 text-orange-400 mb-2 group-hover:scale-110 transition-transform" />
              <p className="text-white font-medium">Start Training</p>
              <p className="text-xs text-gray-500 mt-1">LoRA fine-tuning</p>
            </button>
            <button className="p-4 bg-gray-800 hover:bg-gray-750 rounded-lg border border-gray-700 hover:border-orange-500/50 transition-all group">
              <Box className="w-8 h-8 text-purple-400 mb-2 group-hover:scale-110 transition-transform" />
              <p className="text-white font-medium">View Models</p>
              <p className="text-xs text-gray-500 mt-1">Manage adapters</p>
            </button>
            <button className="p-4 bg-gray-800 hover:bg-gray-750 rounded-lg border border-gray-700 hover:border-orange-500/50 transition-all group">
              <MessageSquare className="w-8 h-8 text-green-400 mb-2 group-hover:scale-110 transition-transform" />
              <p className="text-white font-medium">Run Inference</p>
              <p className="text-xs text-gray-500 mt-1">Test your model</p>
            </button>
          </div>
        </div>
      </div>
    </div>
  )
}
