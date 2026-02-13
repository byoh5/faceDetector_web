import type { MaskRegion, MaskStyle } from '../types'

function sanitizeBounds(region: MaskRegion, maxWidth: number, maxHeight: number) {
  const x = Math.max(0, Math.min(region.x, maxWidth))
  const y = Math.max(0, Math.min(region.y, maxHeight))
  const right = Math.max(0, Math.min(region.x + region.width, maxWidth))
  const bottom = Math.max(0, Math.min(region.y + region.height, maxHeight))
  const width = right - x
  const height = bottom - y

  if (width < 2 || height < 2) {
    return null
  }

  return { x, y, width, height }
}

function drawMosaicMask(
  ctx: CanvasRenderingContext2D,
  sourceCanvas: HTMLCanvasElement,
  x: number,
  y: number,
  width: number,
  height: number,
) {
  const block = Math.max(8, Math.round(Math.min(width, height) / 10))
  const downWidth = Math.max(1, Math.round(width / block))
  const downHeight = Math.max(1, Math.round(height / block))

  const tempCanvas = document.createElement('canvas')
  tempCanvas.width = downWidth
  tempCanvas.height = downHeight

  const tempContext = tempCanvas.getContext('2d')

  if (!tempContext) {
    return
  }

  tempContext.imageSmoothingEnabled = true
  tempContext.drawImage(
    sourceCanvas,
    x,
    y,
    width,
    height,
    0,
    0,
    downWidth,
    downHeight,
  )

  ctx.save()
  ctx.imageSmoothingEnabled = false
  ctx.drawImage(tempCanvas, 0, 0, downWidth, downHeight, x, y, width, height)
  ctx.restore()
}

function drawBlurMask(
  ctx: CanvasRenderingContext2D,
  sourceCanvas: HTMLCanvasElement,
  x: number,
  y: number,
  width: number,
  height: number,
) {
  ctx.save()
  ctx.beginPath()
  ctx.rect(x, y, width, height)
  ctx.clip()
  ctx.filter = 'blur(16px)'
  ctx.drawImage(sourceCanvas, x, y, width, height, x, y, width, height)
  ctx.restore()
}

function drawBlackMask(
  ctx: CanvasRenderingContext2D,
  x: number,
  y: number,
  width: number,
  height: number,
) {
  ctx.save()
  ctx.fillStyle = '#11151b'
  ctx.fillRect(x, y, width, height)
  ctx.restore()
}

export function renderPreview(
  targetCanvas: HTMLCanvasElement,
  sourceCanvas: HTMLCanvasElement,
  regions: MaskRegion[],
  style: MaskStyle,
  previewMode: 'masked' | 'original',
): void {
  targetCanvas.width = sourceCanvas.width
  targetCanvas.height = sourceCanvas.height

  const targetContext = targetCanvas.getContext('2d')

  if (!targetContext) {
    return
  }

  targetContext.clearRect(0, 0, targetCanvas.width, targetCanvas.height)
  targetContext.drawImage(sourceCanvas, 0, 0)

  if (previewMode !== 'masked') {
    return
  }

  for (const region of regions) {
    if (!region.active) {
      continue
    }

    const bounds = sanitizeBounds(region, sourceCanvas.width, sourceCanvas.height)

    if (!bounds) {
      continue
    }

    const { x, y, width, height } = bounds

    if (style === 'mosaic') {
      drawMosaicMask(targetContext, sourceCanvas, x, y, width, height)
      continue
    }

    if (style === 'blur') {
      drawBlurMask(targetContext, sourceCanvas, x, y, width, height)
      continue
    }

    drawBlackMask(targetContext, x, y, width, height)
  }
}
