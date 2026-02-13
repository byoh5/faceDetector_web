import type { DetectionRect } from '../types'

export function clampRect(
  rect: DetectionRect,
  maxWidth: number,
  maxHeight: number,
): DetectionRect | null {
  const x = Math.max(0, Math.min(rect.x, maxWidth))
  const y = Math.max(0, Math.min(rect.y, maxHeight))
  const right = Math.max(0, Math.min(rect.x + rect.width, maxWidth))
  const bottom = Math.max(0, Math.min(rect.y + rect.height, maxHeight))
  const width = right - x
  const height = bottom - y

  if (width < 2 || height < 2) {
    return null
  }

  return {
    x,
    y,
    width,
    height,
    score: rect.score,
  }
}

export function prettyKind(kind: 'face' | 'plate' | 'manual'): string {
  if (kind === 'face') {
    return '얼굴'
  }

  if (kind === 'plate') {
    return '번호판'
  }

  return '수동'
}
