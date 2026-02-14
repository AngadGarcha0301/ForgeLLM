import { createContext, useContext, useState, useEffect, ReactNode } from 'react'
import api from '../lib/api'

interface User {
  id: string
  email: string
  username: string
}

interface Workspace {
  id: number
  name: string
}

interface AuthContextType {
  user: User | null
  token: string | null
  workspace: Workspace | null
  login: (username: string, password: string) => Promise<void>
  register: (email: string, username: string, password: string) => Promise<void>
  logout: () => void
  loading: boolean
}

const AuthContext = createContext<AuthContextType | undefined>(undefined)

export function AuthProvider({ children }: { children: ReactNode }) {
  const [user, setUser] = useState<User | null>(null)
  const [token, setToken] = useState<string | null>(localStorage.getItem('token'))
  const [workspace, setWorkspace] = useState<Workspace | null>(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    if (token) {
      api.defaults.headers.common['Authorization'] = `Bearer ${token}`
      fetchUser()
    } else {
      setLoading(false)
    }
  }, [token])

  const fetchUser = async () => {
    try {
      const res = await api.get('/api/v1/auth/me')
      setUser(res.data)
      // Fetch user's default workspace
      await fetchWorkspace()
    } catch {
      logout()
    } finally {
      setLoading(false)
    }
  }

  const fetchWorkspace = async () => {
    try {
      const res = await api.get('/api/v1/workspaces/')
      if (res.data && res.data.length > 0) {
        setWorkspace(res.data[0])
      } else {
        // No workspace exists, create a default one
        const createRes = await api.post('/api/v1/workspaces/', {
          name: 'Default Workspace',
          description: 'Auto-created workspace'
        })
        setWorkspace(createRes.data)
      }
    } catch (err) {
      console.error('Failed to fetch workspace:', err)
    }
  }

  const login = async (username: string, password: string) => {
    const formData = new URLSearchParams()
    formData.append('username', username)
    formData.append('password', password)
    
    const res = await api.post('/api/v1/auth/login', formData, {
      headers: {
        'Content-Type': 'application/x-www-form-urlencoded',
      },
    })
    const { access_token } = res.data
    localStorage.setItem('token', access_token)
    api.defaults.headers.common['Authorization'] = `Bearer ${access_token}`
    setToken(access_token)
  }

  const register = async (email: string, username: string, password: string) => {
    await api.post('/api/v1/auth/register', { email, username, password })
    await login(username, password)
  }

  const logout = () => {
    localStorage.removeItem('token')
    delete api.defaults.headers.common['Authorization']
    setToken(null)
    setUser(null)
    setWorkspace(null)
  }

  return (
    <AuthContext.Provider value={{ user, token, workspace, login, register, logout, loading }}>
      {children}
    </AuthContext.Provider>
  )
}

export function useAuth() {
  const context = useContext(AuthContext)
  if (context === undefined) {
    throw new Error('useAuth must be used within an AuthProvider')
  }
  return context
}
