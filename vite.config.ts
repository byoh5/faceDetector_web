import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

function normalizeBasePath(basePath: string): string {
  if (!basePath || basePath === '/') {
    return '/'
  }

  return basePath.endsWith('/') ? basePath : `${basePath}/`
}

const githubRepository = process.env.GITHUB_REPOSITORY ?? ''
const repositoryName = githubRepository.split('/')[1] ?? ''
const isUserOrOrgPagesRepo = repositoryName.toLowerCase().endsWith('.github.io')

const fallbackBasePath = process.env.GITHUB_ACTIONS
  ? isUserOrOrgPagesRepo
    ? '/'
    : `/${repositoryName}/`
  : '/'

const resolvedBasePath = normalizeBasePath(
  process.env.VITE_BASE_PATH ?? fallbackBasePath,
)

export default defineConfig({
  base: resolvedBasePath,
  plugins: [react()],
})
