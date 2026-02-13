import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import JSZip from 'jszip'

import './App.css'
import { detectFaces, warmupFaceDetector } from './lib/faceDetection'
import { prettyKind } from './lib/geometry'
import { renderPreview } from './lib/masking'
import { detectLicensePlates, warmupPlateDetector } from './lib/plateDetection'
import type {
  DetectionRect,
  ImageMeta,
  MaskRegion,
  MaskStyle,
  PreviewMode,
  RegionKind,
} from './types'

const MIN_MANUAL_BOX_SIZE = 12
const DEFAULT_JPEG_QUALITY = 0.82
const DEFAULT_MAX_LONG_EDGE = 1920
const MOBILE_BATCH_DETECTION_MAX_LONG_EDGE = 1280
const DESKTOP_BATCH_DETECTION_MAX_LONG_EDGE = 1920
const MOBILE_BATCH_COOLDOWN_MS = 40
const ENGINE_CACHE_KEY = 'face_masker_engine_assets_v1'
const ENABLE_PLATE_DETECTION = false
const FACE_WARMUP_TIMEOUT_MS = 25_000
const PLATE_WARMUP_TIMEOUT_MS = 35_000
const PLATE_DETECTION_TIMEOUT_MS = 20_000
const THEME_STORAGE_KEY = 'face_masker_theme_v1'

type ThemeMode = 'dark' | 'light'
type PageKey = 'tool' | 'about' | 'privacy' | 'contact'
type BatchStatus = 'pending' | 'processing' | 'done' | 'failed'
type BrowserFamily =
  | 'chrome'
  | 'edge'
  | 'firefox'
  | 'safari'
  | 'samsung'
  | 'other'

interface BatchResult {
  id: string
  fileName: string
  status: BatchStatus
  faceCount: number
  plateCount: number
  outputBytes?: number
  outputWidth?: number
  outputHeight?: number
  errorMessage?: string
}

interface ExportOptions {
  optimizeData: boolean
  jpegQuality: number
  maxLongEdge: number
}

interface ExportResult {
  blob: Blob
  width: number
  height: number
}

interface DownloadGuideInfo {
  isMobile: boolean
  environmentLabel: string
  summary: string
  steps: string[]
  footnote: string
  isKakaoInApp?: boolean
}

function parsePageFromHash(hash: string): PageKey {
  const normalized = hash.replace(/^#/, '')

  if (
    normalized === 'tool' ||
    normalized === 'about' ||
    normalized === 'privacy' ||
    normalized === 'contact'
  ) {
    return normalized
  }

  return 'tool'
}

function detectBrowserFamily(ua: string): BrowserFamily {
  if (ua.includes('edg/') || ua.includes('edgios/')) {
    return 'edge'
  }

  if (ua.includes('samsungbrowser/')) {
    return 'samsung'
  }

  if (ua.includes('firefox/') || ua.includes('fxios/')) {
    return 'firefox'
  }

  if (ua.includes('chrome/') || ua.includes('crios/')) {
    return 'chrome'
  }

  if (ua.includes('safari/')) {
    return 'safari'
  }

  return 'other'
}

function buildDownloadGuide(): DownloadGuideInfo {
  if (typeof navigator === 'undefined') {
    return {
      isMobile: false,
      environmentLabel: '웹 · 브라우저 미확인',
      summary: '다운로드된 파일은 브라우저의 다운로드 목록에서 확인할 수 있습니다.',
      steps: [
        '브라우저 메뉴에서 다운로드 목록을 엽니다.',
        '가장 최근 파일 옆의 폴더 열기/표시 기능을 눌러 저장 위치로 이동합니다.',
      ],
      footnote:
        '보안 정책상 웹사이트가 사용자 PC의 다운로드 폴더를 자동으로 여는 것은 제한됩니다.',
    }
  }

  const ua = navigator.userAgent.toLowerCase()
  const platform = (navigator.platform ?? '').toLowerCase()
  const isIOS = /iphone|ipad|ipod/.test(ua)
  const isAndroid = ua.includes('android')
  const isMobile = isIOS || isAndroid
  const isMac = platform.includes('mac')
  const browserFamily = detectBrowserFamily(ua)
  const isKakaoInApp = ua.includes('kakaotalk')

  if (isKakaoInApp) {
    return {
      isMobile,
      environmentLabel: isIOS
        ? '카카오톡 인앱 브라우저(iOS)'
        : '카카오톡 인앱 브라우저(Android)',
      summary:
        '카카오톡 내부 브라우저에서는 다운로드가 실패하거나 저장 위치 확인이 불안정할 수 있습니다.',
      steps: [
        '카카오톡 화면 우측 상단 메뉴(⋮ 또는 공유 버튼)를 엽니다.',
        isIOS
          ? '외부 브라우저로 열기 후 Chrome/Edge(또는 Safari)에서 다시 저장합니다.'
          : '외부 브라우저로 열기 후 Chrome 또는 Edge에서 다시 저장합니다.',
        '외부 브라우저의 다운로드 목록에서 파일을 확인합니다.',
      ],
      footnote:
        '보안 정책과 인앱 브라우저 제한으로 인해 웹페이지에서 이 문제를 직접 우회하는 것은 어렵습니다.',
      isKakaoInApp: true,
    }
  }

  if (isMobile) {
    if (isIOS) {
      return {
        isMobile: true,
        environmentLabel: '모바일(iOS)',
        summary:
          'iPhone/iPad에서는 브라우저 다운로드 목록 또는 파일 앱(Downloads)에서 바로 확인할 수 있습니다.',
        steps: [
          '브라우저의 다운로드 메뉴(예: Safari 주소창 옆 화살표/다운로드 아이콘)를 엽니다.',
          '방금 다운로드한 파일을 터치해 미리보기로 엽니다.',
          '파일 앱 > 찾아보기 > Downloads 폴더에서 같은 파일을 다시 찾을 수 있습니다.',
        ],
        footnote:
          '웹페이지가 파일 앱을 자동으로 여는 기능은 허용되지 않습니다. 대신 공유 버튼을 이용하면 저장 위치 선택이 더 쉽습니다.',
      }
    }

    if (browserFamily === 'samsung') {
      return {
        isMobile: true,
        environmentLabel: '모바일(Android · Samsung Internet)',
        summary:
          'Samsung Internet의 다운로드 목록에서 최근 파일을 연 뒤, 내 파일 앱 Download 폴더에서 위치를 확인할 수 있습니다.',
        steps: [
          '브라우저 메뉴(≡ 또는 ⋮) > 다운로드를 엽니다.',
          '최근 파일을 눌러 열거나 길게 눌러 파일 위치 메뉴를 확인합니다.',
          '내 파일/Files 앱의 Download 폴더에서도 동일 파일을 찾을 수 있습니다.',
        ],
        footnote:
          '기기마다 메뉴 명칭이 조금 다를 수 있습니다. 일반적으로 Download 폴더가 기본 저장 위치입니다.',
      }
    }

    return {
      isMobile: true,
      environmentLabel: '모바일(Android)',
      summary:
        '모바일 브라우저에서는 다운로드 목록에서 파일을 연 뒤, Files 앱 Download 폴더로 이동해 위치를 확인합니다.',
      steps: [
        '브라우저 메뉴(⋮) > 다운로드를 엽니다.',
        '최근 다운로드 파일을 탭해 정상 저장 여부를 확인합니다.',
        'Files/내 파일 앱 > Download 폴더에서 파일 위치를 확인합니다.',
      ],
      footnote:
        '웹페이지가 직접 파일 앱 화면으로 이동시키는 것은 보안 제한으로 불가능합니다.',
    }
  }

  if (browserFamily === 'safari') {
    return {
      isMobile: false,
      environmentLabel: '웹(macOS Safari)',
      summary:
        'Safari 다운로드 목록에서 파일 오른쪽의 돋보기 아이콘을 누르면 Finder 저장 위치로 바로 이동합니다.',
      steps: [
        'Safari 우측 상단 다운로드 버튼을 눌러 다운로드 목록을 엽니다.',
        '최근 파일 오른쪽 돋보기 아이콘을 눌러 Finder에서 표시합니다.',
      ],
      footnote:
        '웹사이트가 Finder를 자동으로 여는 기능은 제공되지 않으며, 사용자가 직접 목록에서 열어야 합니다.',
    }
  }

  if (browserFamily === 'firefox') {
    return {
      isMobile: false,
      environmentLabel: '웹(Firefox)',
      summary:
        'Firefox 다운로드 목록에서 폴더 아이콘을 누르면 저장 폴더로 바로 이동할 수 있습니다.',
      steps: [
        `${isMac ? 'Command + J' : 'Ctrl + J'}로 다운로드 목록을 엽니다.`,
        '최근 파일 오른쪽 폴더 아이콘(폴더에서 보기)을 클릭합니다.',
      ],
      footnote:
        '브라우저 보안 정책상 웹페이지가 사용자 파일 탐색기를 직접 여는 동작은 차단됩니다.',
    }
  }

  if (browserFamily === 'edge') {
    return {
      isMobile: false,
      environmentLabel: '웹(Microsoft Edge)',
      summary:
        'Edge 다운로드 목록에서 폴더 아이콘(폴더에서 표시)을 누르면 저장 위치를 바로 열 수 있습니다.',
      steps: [
        `${isMac ? 'Command + Option + L' : 'Ctrl + J'}로 다운로드 허브를 엽니다.`,
        '최근 파일 항목의 폴더 아이콘을 눌러 저장 폴더를 엽니다.',
      ],
      footnote:
        '웹사이트가 다운로드 폴더를 자동으로 열 권한은 없어서, 목록에서 수동 확인이 필요합니다.',
    }
  }

  return {
    isMobile: false,
    environmentLabel:
      browserFamily === 'chrome' ? '웹(Chrome 계열)' : '웹(브라우저 미확인)',
    summary:
      'Chrome 계열 브라우저는 다운로드 목록에서 파일 옆 폴더 아이콘으로 저장 위치를 바로 열 수 있습니다.',
    steps: [
      `${isMac ? 'Command + Shift + J' : 'Ctrl + J'}로 다운로드 목록을 엽니다.`,
      '최근 파일의 폴더 아이콘(폴더에서 표시/열기)을 클릭합니다.',
    ],
    footnote:
      '보안 정책 때문에 웹페이지가 사용자 PC 파일 시스템 경로를 직접 열거나 노출할 수는 없습니다.',
  }
}

const PAGE_ITEMS: Array<{ key: PageKey; label: string; icon: string }> = [
  { key: 'tool', label: '자동 가림 툴', icon: '🧩' },
  { key: 'about', label: '사이트 소개', icon: '📘' },
  { key: 'privacy', label: '개인정보처리방침', icon: '🛡️' },
  { key: 'contact', label: '문의하기', icon: '✉️' },
]

function withTimeout<T>(
  promise: Promise<T>,
  timeoutMs: number,
  timeoutMessage: string,
): Promise<T> {
  return new Promise<T>((resolve, reject) => {
    const timeoutId = globalThis.setTimeout(() => {
      reject(new Error(timeoutMessage))
    }, timeoutMs)

    promise
      .then((result) => {
        globalThis.clearTimeout(timeoutId)
        resolve(result)
      })
      .catch((error) => {
        globalThis.clearTimeout(timeoutId)
        reject(error)
      })
  })
}

function toErrorMessage(error: unknown, fallbackMessage: string): string {
  return error instanceof Error ? error.message : fallbackMessage
}

function isImageFile(file: File): boolean {
  return file.type.startsWith('image/')
}

function stripExtension(fileName: string): string {
  return fileName.replace(/\.[^.]+$/, '') || 'masked-photo'
}

function formatBytes(bytes: number): string {
  if (bytes < 1024) {
    return `${bytes}B`
  }

  if (bytes < 1024 * 1024) {
    return `${(bytes / 1024).toFixed(1)}KB`
  }

  return `${(bytes / (1024 * 1024)).toFixed(2)}MB`
}

function createAutoRegions(
  faces: DetectionRect[],
  plates: DetectionRect[],
): MaskRegion[] {
  return [
    ...faces.map((rect, index) => ({
      id: `auto-face-${index}`,
      kind: 'face' as const,
      active: true,
      x: rect.x,
      y: rect.y,
      width: rect.width,
      height: rect.height,
      score: rect.score,
    })),
    ...plates.map((rect, index) => ({
      id: `auto-plate-${index}`,
      kind: 'plate' as const,
      active: true,
      x: rect.x,
      y: rect.y,
      width: rect.width,
      height: rect.height,
      score: rect.score,
    })),
  ]
}

async function canvasToBlob(
  canvas: HTMLCanvasElement,
  mimeType: string,
  quality?: number,
): Promise<Blob> {
  return new Promise<Blob>((resolve, reject) => {
    canvas.toBlob(
      (nextBlob) => {
        if (!nextBlob) {
          reject(new Error('이미지 변환에 실패했습니다.'))
          return
        }

        resolve(nextBlob)
      },
      mimeType,
      quality,
    )
  })
}

function calculateTargetSize(
  width: number,
  height: number,
  maxLongEdge: number,
): { width: number; height: number } {
  if (maxLongEdge <= 0) {
    return { width, height }
  }

  const longest = Math.max(width, height)

  if (longest <= maxLongEdge) {
    return { width, height }
  }

  const scale = maxLongEdge / longest

  return {
    width: Math.max(1, Math.round(width * scale)),
    height: Math.max(1, Math.round(height * scale)),
  }
}

function createBatchDetectionSource(
  sourceCanvas: HTMLCanvasElement,
  maxLongEdge: number,
): { canvas: HTMLCanvasElement; scale: number } {
  const longest = Math.max(sourceCanvas.width, sourceCanvas.height)

  if (longest <= maxLongEdge || maxLongEdge <= 0) {
    return { canvas: sourceCanvas, scale: 1 }
  }

  const scale = maxLongEdge / longest
  const detectionCanvas = document.createElement('canvas')
  detectionCanvas.width = Math.max(1, Math.round(sourceCanvas.width * scale))
  detectionCanvas.height = Math.max(1, Math.round(sourceCanvas.height * scale))

  const context = detectionCanvas.getContext('2d')

  if (!context) {
    return { canvas: sourceCanvas, scale: 1 }
  }

  context.imageSmoothingEnabled = true
  context.imageSmoothingQuality = 'high'
  context.drawImage(
    sourceCanvas,
    0,
    0,
    sourceCanvas.width,
    sourceCanvas.height,
    0,
    0,
    detectionCanvas.width,
    detectionCanvas.height,
  )

  return { canvas: detectionCanvas, scale }
}

function rescaleDetectionRects(
  rects: DetectionRect[],
  scale: number,
): DetectionRect[] {
  if (scale >= 0.999) {
    return rects
  }

  const inverse = 1 / scale

  return rects.map((rect) => ({
    ...rect,
    x: rect.x * inverse,
    y: rect.y * inverse,
    width: rect.width * inverse,
    height: rect.height * inverse,
  }))
}

async function sleep(ms: number): Promise<void> {
  await new Promise<void>((resolve) => {
    globalThis.setTimeout(resolve, ms)
  })
}

function getJpegQuality(optimizeData: boolean, jpegQuality: number): number {
  if (!optimizeData) {
    return 0.94
  }

  return Math.max(0.6, Math.min(0.92, jpegQuality))
}

async function createMaskedJpegExport(
  sourceCanvas: HTMLCanvasElement,
  regions: MaskRegion[],
  maskStyle: MaskStyle,
  options: ExportOptions,
): Promise<ExportResult> {
  const maskedCanvas = document.createElement('canvas')
  renderPreview(maskedCanvas, sourceCanvas, regions, maskStyle, 'masked')

  const targetSize = options.optimizeData
    ? calculateTargetSize(
        maskedCanvas.width,
        maskedCanvas.height,
        options.maxLongEdge,
      )
    : { width: maskedCanvas.width, height: maskedCanvas.height }

  let exportCanvas = maskedCanvas

  if (
    targetSize.width !== maskedCanvas.width ||
    targetSize.height !== maskedCanvas.height
  ) {
    const resizedCanvas = document.createElement('canvas')
    resizedCanvas.width = targetSize.width
    resizedCanvas.height = targetSize.height

    const resizedContext = resizedCanvas.getContext('2d')

    if (resizedContext) {
      resizedContext.imageSmoothingEnabled = true
      resizedContext.imageSmoothingQuality = 'high'
      resizedContext.drawImage(
        maskedCanvas,
        0,
        0,
        maskedCanvas.width,
        maskedCanvas.height,
        0,
        0,
        resizedCanvas.width,
        resizedCanvas.height,
      )
      exportCanvas = resizedCanvas
    }
  }

  const blob = await canvasToBlob(
    exportCanvas,
    'image/jpeg',
    getJpegQuality(options.optimizeData, options.jpegQuality),
  )

  return {
    blob,
    width: exportCanvas.width,
    height: exportCanvas.height,
  }
}

async function loadImageFromFile(
  file: File,
): Promise<{ image: HTMLImageElement; objectUrl: string }> {
  const objectUrl = URL.createObjectURL(file)

  const image = await new Promise<HTMLImageElement>((resolve, reject) => {
    const element = new Image()
    element.onload = () => resolve(element)
    element.onerror = () => reject(new Error('이미지를 읽지 못했습니다.'))
    element.src = objectUrl
  })

  return { image, objectUrl }
}

function createSourceCanvas(image: HTMLImageElement): HTMLCanvasElement {
  const sourceCanvas = document.createElement('canvas')
  sourceCanvas.width = image.naturalWidth
  sourceCanvas.height = image.naturalHeight

  const sourceContext = sourceCanvas.getContext('2d')

  if (sourceContext) {
    sourceContext.drawImage(image, 0, 0)
  }

  return sourceCanvas
}

function App() {
  const canvasRef = useRef<HTMLCanvasElement | null>(null)
  const overlayRef = useRef<SVGSVGElement | null>(null)
  const sourceCanvasRef = useRef<HTMLCanvasElement | null>(null)
  const sourceImageRef = useRef<HTMLImageElement | null>(null)
  const activeObjectUrlRef = useRef<string | null>(null)
  const batchInputRef = useRef<HTMLInputElement | null>(null)
  const dragStartRef = useRef<{ x: number; y: number } | null>(null)
  const idSequenceRef = useRef(1)

  const [themeMode, setThemeMode] = useState<ThemeMode>(() => {
    if (typeof window === 'undefined') {
      return 'dark'
    }

    const storedTheme = window.localStorage.getItem(THEME_STORAGE_KEY)

    if (storedTheme === 'dark' || storedTheme === 'light') {
      return storedTheme
    }

    return window.matchMedia('(prefers-color-scheme: dark)').matches
      ? 'dark'
      : 'light'
  })
  const [activePage, setActivePage] = useState<PageKey>(() => {
    if (typeof window === 'undefined') {
      return 'tool'
    }

    return parsePageFromHash(window.location.hash)
  })

  const [isPrepared, setIsPrepared] = useState(false)
  const [isPreparing, setIsPreparing] = useState(false)
  const [hasCachedEngines, setHasCachedEngines] = useState(false)
  const [isPlateDetectorReady, setIsPlateDetectorReady] = useState(false)
  const [isPlateDetectorWarming, setIsPlateDetectorWarming] = useState(false)
  const [isPlateDetectorFailed, setIsPlateDetectorFailed] = useState(false)
  const [preparationMessage, setPreparationMessage] = useState(
    ENABLE_PLATE_DETECTION
      ? '기능을 시작하면 얼굴/번호판 검출 엔진을 한 번만 내려받습니다.'
      : '기능을 시작하면 얼굴 검출 엔진을 한 번만 내려받습니다. (번호판 기능 임시 보류)',
  )

  const [imageMeta, setImageMeta] = useState<ImageMeta | null>(null)
  const [regions, setRegions] = useState<MaskRegion[]>([])
  const [maskStyle, setMaskStyle] = useState<MaskStyle>('mosaic')
  const [previewMode, setPreviewMode] = useState<PreviewMode>('masked')
  const [optimizeData, setOptimizeData] = useState(true)
  const [jpegQuality, setJpegQuality] = useState(DEFAULT_JPEG_QUALITY)
  const [maxLongEdge, setMaxLongEdge] = useState(DEFAULT_MAX_LONG_EDGE)
  const [isDragging, setIsDragging] = useState(false)
  const [isScanning, setIsScanning] = useState(false)
  const [isBatchProcessing, setIsBatchProcessing] = useState(false)
  const [drawModeEnabled, setDrawModeEnabled] = useState(false)
  const [draftRect, setDraftRect] = useState<DetectionRect | null>(null)
  const [batchResults, setBatchResults] = useState<BatchResult[]>([])
  const [isDownloadGuideOpen, setIsDownloadGuideOpen] = useState(false)
  const [statusMessage, setStatusMessage] = useState(
    ENABLE_PLATE_DETECTION
      ? '사진을 올리면 얼굴+번호판 자동 가림이 시작됩니다.'
      : '사진을 올리면 얼굴 자동 가림이 시작됩니다. (번호판 기능 임시 보류)',
  )

  const canUseShareSheet =
    typeof navigator !== 'undefined' && typeof navigator.share === 'function'

  const downloadGuide = useMemo(() => buildDownloadGuide(), [])
  const isKakaoInApp = Boolean(downloadGuide.isKakaoInApp)

  const downloadHint = useMemo(() => {
    if (downloadGuide.isKakaoInApp) {
      return '카카오톡 내부 브라우저에서는 저장이 불안정할 수 있어 외부 브라우저(Chrome/Edge)에서 여는 것을 권장합니다.'
    }

    if (downloadGuide.isMobile) {
      return canUseShareSheet
        ? '모바일에서는 다운로드 목록 또는 공유 버튼으로 저장 위치를 빠르게 확인할 수 있습니다.'
        : '모바일에서는 브라우저 다운로드 목록과 파일 앱(Download/Downloads 폴더)에서 파일 위치를 확인하세요.'
    }

    return '웹 브라우저 보안 정책상 저장 폴더를 자동으로 열 수 없어서, 다운로드 목록에서 파일 위치를 확인해야 합니다.'
  }, [canUseShareSheet, downloadGuide.isKakaoInApp, downloadGuide.isMobile])

  const copyCurrentUrl = useCallback(async () => {
    if (typeof window === 'undefined' || typeof document === 'undefined') {
      return
    }

    const currentUrl = window.location.href

    try {
      if (typeof navigator !== 'undefined' && navigator.clipboard?.writeText) {
        await navigator.clipboard.writeText(currentUrl)
      } else {
        const textarea = document.createElement('textarea')
        textarea.value = currentUrl
        textarea.setAttribute('readonly', 'true')
        textarea.style.position = 'fixed'
        textarea.style.opacity = '0'
        document.body.appendChild(textarea)
        textarea.select()
        document.execCommand('copy')
        document.body.removeChild(textarea)
      }

      setStatusMessage(
        '현재 페이지 링크를 복사했습니다. 카카오톡 메뉴에서 외부 브라우저(Chrome/Edge)로 열어 주세요.',
      )
    } catch {
      setStatusMessage(
        '링크 복사에 실패했습니다. 카카오톡 메뉴에서 직접 외부 브라우저로 열어 주세요.',
      )
    }
  }, [])

  useEffect(() => {
    if (typeof window === 'undefined') {
      return
    }

    window.localStorage.setItem(THEME_STORAGE_KEY, themeMode)
    document.documentElement.dataset.theme = themeMode
  }, [themeMode])

  useEffect(() => {
    if (typeof window === 'undefined') {
      return
    }

    const onHashChange = () => {
      setActivePage(parsePageFromHash(window.location.hash))
    }

    window.addEventListener('hashchange', onHashChange)

    return () => {
      window.removeEventListener('hashchange', onHashChange)
    }
  }, [])

  useEffect(() => {
    if (typeof window === 'undefined') {
      return
    }

    const nextHash = activePage === 'tool' ? '' : `#${activePage}`
    const nextUrl = `${window.location.pathname}${window.location.search}${nextHash}`
    window.history.replaceState(null, '', nextUrl)
  }, [activePage])

  const toggleThemeMode = useCallback(() => {
    setThemeMode((prev) => (prev === 'dark' ? 'light' : 'dark'))
  }, [])

  const goToPage = useCallback((page: PageKey) => {
    setActivePage(page)
  }, [])

  useEffect(() => {
    if (typeof window === 'undefined') {
      return
    }

    const cached = window.localStorage.getItem(ENGINE_CACHE_KEY) === '1'
    setHasCachedEngines(cached)

    if (cached) {
      setPreparationMessage(
        '이 브라우저는 엔진 다운로드 이력이 있습니다. 다시 받아야 하는 데이터가 거의 없습니다.',
      )
    }
  }, [])

  useEffect(() => {
    return () => {
      if (activeObjectUrlRef.current) {
        URL.revokeObjectURL(activeObjectUrlRef.current)
      }
    }
  }, [])

  const createRegion = useCallback(
    (kind: RegionKind, rect: DetectionRect, active = true): MaskRegion => {
      const id = `region-${idSequenceRef.current}`
      idSequenceRef.current += 1

      return {
        id,
        kind,
        active,
        x: rect.x,
        y: rect.y,
        width: rect.width,
        height: rect.height,
        score: rect.score,
      }
    },
    [],
  )

  const redrawCanvas = useCallback(() => {
    const canvas = canvasRef.current
    const sourceCanvas = sourceCanvasRef.current

    if (!canvas || !sourceCanvas) {
      return
    }

    renderPreview(canvas, sourceCanvas, regions, maskStyle, previewMode)
  }, [maskStyle, previewMode, regions])

  useEffect(() => {
    redrawCanvas()
  }, [redrawCanvas])

  useEffect(() => {
    if (activePage !== 'tool' || !imageMeta) {
      return
    }

    // The canvas element is recreated when moving between top-level pages.
    // Redraw once after returning to the tool page so the previous image stays visible.
    const animationId = window.requestAnimationFrame(() => {
      redrawCanvas()
    })

    return () => {
      window.cancelAnimationFrame(animationId)
    }
  }, [activePage, imageMeta, redrawCanvas])

  const exportOptions = useMemo<ExportOptions>(
    () => ({
      optimizeData,
      jpegQuality,
      maxLongEdge,
    }),
    [jpegQuality, maxLongEdge, optimizeData],
  )

  const prepareTransformer = useCallback(async () => {
    if (isPreparing) {
      return
    }

    if (isKakaoInApp) {
      setPreparationMessage(
        '카카오톡 내부 브라우저에서는 변환 시작을 막아두었습니다. 외부 브라우저(Chrome/Edge)에서 열어 주세요.',
      )
      return
    }

    setIsPreparing(true)
    setPreparationMessage('얼굴 엔진을 준비 중입니다. 잠시만 기다려 주세요...')

    try {
      await withTimeout(
        warmupFaceDetector(),
        FACE_WARMUP_TIMEOUT_MS,
        '얼굴 엔진 준비 시간이 초과되었습니다. 네트워크 상태를 확인한 뒤 다시 시도해 주세요.',
      )

      if (typeof window !== 'undefined') {
        window.localStorage.setItem(ENGINE_CACHE_KEY, '1')
      }

      setHasCachedEngines(true)
      setIsPlateDetectorReady(false)
      setIsPlateDetectorFailed(false)
      setIsPrepared(true)
      setPreparationMessage(
        ENABLE_PLATE_DETECTION
          ? '준비 완료. 같은 브라우저에서는 추가 다운로드 없이 바로 변환할 수 있습니다.'
          : '준비 완료. 같은 브라우저에서는 추가 다운로드 없이 바로 변환할 수 있습니다. (번호판 기능 임시 보류)',
      )
      setStatusMessage(
        ENABLE_PLATE_DETECTION
          ? '엔진 준비 완료: 얼굴 엔진은 즉시 사용 가능하며 번호판 엔진은 백그라운드에서 준비됩니다.'
          : '엔진 준비 완료: 얼굴 엔진을 사용할 수 있습니다. (번호판 기능 임시 보류)',
      )
    } catch (error) {
      const message = toErrorMessage(
        error,
        '엔진 다운로드 중 오류가 발생했습니다.',
      )
      setPreparationMessage(`준비 실패: ${message}`)
    } finally {
      setIsPreparing(false)
    }
  }, [isKakaoInApp, isPreparing])

  useEffect(() => {
    if (!ENABLE_PLATE_DETECTION) {
      return
    }

    if (
      !isPrepared ||
      isPlateDetectorReady ||
      isPlateDetectorFailed ||
      isPlateDetectorWarming
    ) {
      return
    }

    let isMounted = true
    setIsPlateDetectorWarming(true)

    void withTimeout(
      warmupPlateDetector(),
      PLATE_WARMUP_TIMEOUT_MS,
      '번호판 엔진 준비 시간이 초과되었습니다.',
    )
      .then(() => {
        if (!isMounted) {
          return
        }

        setIsPlateDetectorReady(true)
        setStatusMessage(
          '번호판 엔진 준비 완료: 얼굴+번호판 자동 스캔을 사용할 수 있습니다.',
        )
      })
      .catch((error) => {
        if (!isMounted) {
          return
        }

        const message = toErrorMessage(
          error,
          '번호판 엔진 준비 중 오류가 발생했습니다.',
        )
        setIsPlateDetectorFailed(true)
        setStatusMessage(
          `번호판 엔진 준비 실패: ${message} 얼굴 중심 모드로 계속 사용할 수 있습니다.`,
        )
      })
      .finally(() => {
        if (!isMounted) {
          return
        }
        setIsPlateDetectorWarming(false)
      })

    return () => {
      isMounted = false
    }
  }, [
    isPrepared,
    isPlateDetectorFailed,
    isPlateDetectorReady,
    isPlateDetectorWarming,
  ])

  const runAutoScan = useCallback(async () => {
    const image = sourceImageRef.current

    if (!image) {
      return
    }

    if (!isPrepared) {
      setStatusMessage('먼저 변환 준비를 완료해 주세요.')
      return
    }

    const canUsePlateDetector = ENABLE_PLATE_DETECTION && isPlateDetectorReady

    setIsScanning(true)
    setStatusMessage(
      canUsePlateDetector
        ? '자동 스캔 중... 얼굴과 번호판을 찾고 있습니다.'
        : ENABLE_PLATE_DETECTION && isPlateDetectorWarming
          ? '자동 스캔 중... 얼굴을 찾고 있습니다. (번호판 엔진 준비 중)'
          : '자동 스캔 중... 얼굴을 찾고 있습니다.',
    )

    try {
      let plateErrorMessage = ''

      const [faces, plates] = await Promise.all([
        detectFaces(image),
        canUsePlateDetector
          ? withTimeout(
              detectLicensePlates(image),
              PLATE_DETECTION_TIMEOUT_MS,
              '번호판 검출 시간이 초과되었습니다.',
            ).catch((error) => {
              plateErrorMessage = toErrorMessage(
                error,
                '번호판 검출 중 오류가 발생했습니다.',
              )
              return []
            })
          : Promise.resolve<DetectionRect[]>([]),
      ])

      const detectedRegions = [
        ...faces.map((rect) => createRegion('face', rect, true)),
        ...plates.map((rect) => createRegion('plate', rect, true)),
      ]

      setRegions(detectedRegions)
      setPreviewMode('masked')

      if (canUsePlateDetector && plateErrorMessage) {
        setIsPlateDetectorReady(false)
        setIsPlateDetectorFailed(true)
        setStatusMessage(
          `자동 스캔 완료: 얼굴 ${faces.length}개, 번호판 엔진 오류로 얼굴만 처리했습니다. (${plateErrorMessage})`,
        )
        return
      }

      if (!canUsePlateDetector) {
        setStatusMessage(
          ENABLE_PLATE_DETECTION
            ? `자동 스캔 완료: 얼굴 ${faces.length}개 (번호판 엔진 비활성화)`
            : `자동 스캔 완료: 얼굴 ${faces.length}개 (번호판 기능 임시 보류)`,
        )
        return
      }

      setStatusMessage(`자동 스캔 완료: 얼굴 ${faces.length}개, 번호판 ${plates.length}개`)
    } catch (error) {
      const message = toErrorMessage(error, '자동 스캔 중 오류가 발생했습니다.')
      setStatusMessage(`자동 스캔 실패: ${message} 수동 박스로 보정해 주세요.`)
      setRegions([])
    } finally {
      setIsScanning(false)
    }
  }, [
    createRegion,
    isPlateDetectorReady,
    isPlateDetectorWarming,
    isPrepared,
  ])

  const loadFile = useCallback(
    async (file: File) => {
      if (!isPrepared) {
        setStatusMessage('먼저 상단의 변환 준비를 완료해 주세요.')
        return
      }

      if (isBatchProcessing) {
        setStatusMessage('일괄 처리 중에는 단일 이미지를 열 수 없습니다.')
        return
      }

      if (!isImageFile(file)) {
        setStatusMessage('이미지 파일만 업로드할 수 있습니다.')
        return
      }

      setDraftRect(null)
      setDrawModeEnabled(false)
      setRegions([])

      try {
        const { image, objectUrl } = await loadImageFromFile(file)

        if (activeObjectUrlRef.current) {
          URL.revokeObjectURL(activeObjectUrlRef.current)
        }

        activeObjectUrlRef.current = objectUrl
        sourceImageRef.current = image
        sourceCanvasRef.current = createSourceCanvas(image)

        setImageMeta({
          name: file.name,
          width: image.naturalWidth,
          height: image.naturalHeight,
          sizeBytes: file.size,
        })

        setPreviewMode('masked')
        setStatusMessage('이미지를 불러왔습니다. 자동 스캔을 시작합니다.')

        await runAutoScan()
      } catch (error) {
        const message =
          error instanceof Error ? error.message : '이미지 로딩 중 오류가 발생했습니다.'
        setStatusMessage(message)
      }
    },
    [isBatchProcessing, isPrepared, runAutoScan],
  )

  const processBatchFiles = useCallback(
    async (files: File[]) => {
      if (!isPrepared) {
        setStatusMessage('먼저 변환 준비를 완료해 주세요.')
        return
      }

      const imageFiles = files.filter(isImageFile)

      if (imageFiles.length < 2) {
        setStatusMessage('일괄 처리는 이미지 2장 이상에서 동작합니다.')
        return
      }

      if (isScanning) {
        setStatusMessage('단일 이미지 스캔이 끝난 뒤 다시 시도해 주세요.')
        return
      }

      let plateDetectionEnabled = ENABLE_PLATE_DETECTION && isPlateDetectorReady
      let plateDetectionErrorMessage = ''
      const isMobileBatch = downloadGuide.isMobile
      const batchDetectionMaxLongEdge = isMobileBatch
        ? MOBILE_BATCH_DETECTION_MAX_LONG_EDGE
        : DESKTOP_BATCH_DETECTION_MAX_LONG_EDGE

      setIsBatchProcessing(true)
      setDrawModeEnabled(false)
      setDraftRect(null)

      const initialResults = imageFiles.map((file, index) => ({
        id: `batch-${Date.now()}-${index}`,
        fileName: file.name,
        status: 'pending' as const,
        faceCount: 0,
        plateCount: 0,
      }))

      const batchStartBaseMessage = plateDetectionEnabled
        ? `${imageFiles.length}장 일괄 처리를 시작합니다.`
        : ENABLE_PLATE_DETECTION
          ? `${imageFiles.length}장 일괄 처리를 시작합니다. (번호판 엔진 비활성화: 얼굴 중심 모드)`
          : `${imageFiles.length}장 일괄 처리를 시작합니다. (번호판 기능 임시 보류: 얼굴 중심 모드)`
      setBatchResults(initialResults)
      setStatusMessage(
        `${batchStartBaseMessage} 일괄 처리 중에는 미리보기 패널 대신 로그에서 진행 상황을 확인할 수 있습니다.`,
      )

      const zip = new JSZip()
      let successCount = 0
      let failureCount = 0
      let inputBytesTotal = 0
      let outputBytesTotal = 0

      try {
        for (let index = 0; index < imageFiles.length; index += 1) {
          const file = imageFiles[index]
          const resultId = initialResults[index].id

          setBatchResults((prev) =>
            prev.map((item) =>
              item.id === resultId ? { ...item, status: 'processing' } : item,
            ),
          )

          let objectUrl: string | null = null
          let sourceCanvas: HTMLCanvasElement | null = null
          let detectionCanvas: HTMLCanvasElement | null = null
          let loadedImage: HTMLImageElement | null = null

          try {
            inputBytesTotal += file.size

            const loaded = await loadImageFromFile(file)
            objectUrl = loaded.objectUrl
            loadedImage = loaded.image

            sourceCanvas = createSourceCanvas(loaded.image)
            const detectionSource = createBatchDetectionSource(
              sourceCanvas,
              batchDetectionMaxLongEdge,
            )
            detectionCanvas =
              detectionSource.canvas === sourceCanvas ? null : detectionSource.canvas
            let plateScanErrorMessage = ''

            const [faces, plates] = await Promise.all([
              detectFaces(detectionSource.canvas).then((detected) =>
                rescaleDetectionRects(detected, detectionSource.scale),
              ),
              plateDetectionEnabled
                ? withTimeout(
                    detectLicensePlates(loaded.image),
                    PLATE_DETECTION_TIMEOUT_MS,
                    '번호판 검출 시간이 초과되었습니다.',
                  ).catch((error) => {
                    plateScanErrorMessage = toErrorMessage(
                      error,
                      '번호판 검출 중 오류가 발생했습니다.',
                    )
                    return []
                  })
                : Promise.resolve<DetectionRect[]>([]),
            ])

            if (plateScanErrorMessage && plateDetectionEnabled) {
              plateDetectionEnabled = false
              plateDetectionErrorMessage = plateScanErrorMessage
            }

            const exported = await createMaskedJpegExport(
              sourceCanvas,
              createAutoRegions(faces, plates),
              maskStyle,
              exportOptions,
            )

            outputBytesTotal += exported.blob.size
            zip.file(`${stripExtension(file.name)}-masked.jpg`, exported.blob)

            successCount += 1

            setBatchResults((prev) =>
              prev.map((item) =>
                item.id === resultId
                  ? {
                      ...item,
                      status: 'done',
                      faceCount: faces.length,
                      plateCount: plates.length,
                      outputBytes: exported.blob.size,
                      outputWidth: exported.width,
                      outputHeight: exported.height,
                    }
                  : item,
              ),
            )
          } catch (error) {
            failureCount += 1
            const message = toErrorMessage(
              error,
              '이미지 처리 중 오류가 발생했습니다.',
            )

            setBatchResults((prev) =>
              prev.map((item) =>
                item.id === resultId
                  ? {
                      ...item,
                      status: 'failed',
                      errorMessage: `${message}${isMobileBatch ? ' (모바일 메모리/브라우저 제한 가능성)' : ''}`,
                    }
                  : item,
              ),
            )
          } finally {
            if (detectionCanvas) {
              detectionCanvas.width = 0
              detectionCanvas.height = 0
            }

            if (sourceCanvas) {
              sourceCanvas.width = 0
              sourceCanvas.height = 0
            }

            if (loadedImage) {
              loadedImage.src = ''
            }

            if (objectUrl) {
              URL.revokeObjectURL(objectUrl)
            }

            if (isMobileBatch) {
              await sleep(MOBILE_BATCH_COOLDOWN_MS)
            }
          }
        }

        if (successCount > 0) {
          const zipBlob = await zip.generateAsync({ type: 'blob' })
          const zipUrl = URL.createObjectURL(zipBlob)

          const anchor = document.createElement('a')
          anchor.href = zipUrl
          anchor.download = `masked-batch-${new Date()
            .toISOString()
            .replace(/[.:]/g, '-')}.zip`
          anchor.click()

          URL.revokeObjectURL(zipUrl)
          setIsDownloadGuideOpen(true)
        }

        const plateModeNote = plateDetectionEnabled
          ? ''
          : ENABLE_PLATE_DETECTION
            ? ` · 번호판 엔진 비활성화(얼굴 중심 모드${plateDetectionErrorMessage ? `: ${plateDetectionErrorMessage}` : ''})`
            : ' · 번호판 기능 임시 보류(얼굴 중심 모드)'

        if (successCount === 0) {
          setStatusMessage(`일괄 처리 실패: 생성된 이미지가 없습니다.${plateModeNote}`)
        } else if (failureCount > 0) {
          const reductionText =
            outputBytesTotal > 0 && inputBytesTotal > 0
              ? ` · 총 ${formatBytes(inputBytesTotal)} → ${formatBytes(outputBytesTotal)}`
              : ''
          setStatusMessage(
            `일괄 처리 완료: 성공 ${successCount}장 / 실패 ${failureCount}장${reductionText}${plateModeNote} (ZIP 다운로드 완료)`,
          )
        } else {
          const reductionText =
            outputBytesTotal > 0 && inputBytesTotal > 0
              ? ` (${formatBytes(inputBytesTotal)} → ${formatBytes(outputBytesTotal)})`
              : ''
          setStatusMessage(
            `일괄 처리 완료: ${successCount}장 ZIP 다운로드 완료${reductionText}${plateModeNote}`,
          )
        }
      } catch (error) {
        const message = toErrorMessage(error, '일괄 처리 중 오류가 발생했습니다.')
        setStatusMessage(`일괄 처리 실패: ${message}`)
      } finally {
        if (ENABLE_PLATE_DETECTION && isPlateDetectorReady && !plateDetectionEnabled) {
          setIsPlateDetectorReady(false)
          setIsPlateDetectorFailed(true)
        }
        setIsBatchProcessing(false)
      }
    },
    [
      downloadGuide.isMobile,
      exportOptions,
      isPlateDetectorReady,
      isPrepared,
      isScanning,
      maskStyle,
    ],
  )

  const getPointOnImage = useCallback(
    (event: React.PointerEvent<SVGSVGElement>) => {
      const svg = overlayRef.current

      if (!svg || !imageMeta) {
        return null
      }

      const rect = svg.getBoundingClientRect()

      if (rect.width === 0 || rect.height === 0) {
        return null
      }

      const x = ((event.clientX - rect.left) / rect.width) * imageMeta.width
      const y = ((event.clientY - rect.top) / rect.height) * imageMeta.height

      return {
        x: Math.max(0, Math.min(x, imageMeta.width)),
        y: Math.max(0, Math.min(y, imageMeta.height)),
      }
    },
    [imageMeta],
  )

  const onOverlayPointerDown = useCallback(
    (event: React.PointerEvent<SVGSVGElement>) => {
      if (!drawModeEnabled || !imageMeta) {
        return
      }

      const point = getPointOnImage(event)

      if (!point) {
        return
      }

      event.preventDefault()
      event.currentTarget.setPointerCapture(event.pointerId)
      dragStartRef.current = point
      setDraftRect({ x: point.x, y: point.y, width: 0, height: 0 })
    },
    [drawModeEnabled, getPointOnImage, imageMeta],
  )

  const onOverlayPointerMove = useCallback(
    (event: React.PointerEvent<SVGSVGElement>) => {
      if (!drawModeEnabled) {
        return
      }

      const start = dragStartRef.current

      if (!start) {
        return
      }

      const point = getPointOnImage(event)

      if (!point) {
        return
      }

      const nextRect = {
        x: Math.min(start.x, point.x),
        y: Math.min(start.y, point.y),
        width: Math.abs(start.x - point.x),
        height: Math.abs(start.y - point.y),
      }

      setDraftRect(nextRect)
    },
    [drawModeEnabled, getPointOnImage],
  )

  const onOverlayPointerUp = useCallback(
    (event: React.PointerEvent<SVGSVGElement>) => {
      if (!drawModeEnabled) {
        return
      }

      const start = dragStartRef.current

      if (!start) {
        return
      }

      const point = getPointOnImage(event)
      dragStartRef.current = null

      event.currentTarget.releasePointerCapture(event.pointerId)

      if (!point) {
        setDraftRect(null)
        return
      }

      const nextRect = {
        x: Math.min(start.x, point.x),
        y: Math.min(start.y, point.y),
        width: Math.abs(start.x - point.x),
        height: Math.abs(start.y - point.y),
      }

      setDraftRect(null)

      if (
        nextRect.width < MIN_MANUAL_BOX_SIZE ||
        nextRect.height < MIN_MANUAL_BOX_SIZE
      ) {
        return
      }

      setRegions((prev) => [...prev, createRegion('manual', nextRect, true)])
      setStatusMessage('수동 마스킹 박스를 추가했습니다.')
    },
    [createRegion, drawModeEnabled, getPointOnImage],
  )

  const toggleRegion = useCallback((id: string) => {
    setRegions((prev) =>
      prev.map((region) =>
        region.id === id ? { ...region, active: !region.active } : region,
      ),
    )
  }, [])

  const removeRegion = useCallback((id: string) => {
    setRegions((prev) => prev.filter((region) => region.id !== id))
  }, [])

  const onPickFile: React.ChangeEventHandler<HTMLInputElement> = useCallback(
    async (event) => {
      const files = Array.from(event.target.files ?? [])

      if (files.length === 0) {
        return
      }

      if (files.length > 1) {
        await processBatchFiles(files)
        event.target.value = ''
        return
      }

      await loadFile(files[0])
      event.target.value = ''
    },
    [loadFile, processBatchFiles],
  )

  const onPickBatchFiles: React.ChangeEventHandler<HTMLInputElement> =
    useCallback(
      async (event) => {
        const files = Array.from(event.target.files ?? [])

        if (files.length === 0) {
          return
        }

        await processBatchFiles(files)
        event.target.value = ''
      },
      [processBatchFiles],
    )

  const onDrop: React.DragEventHandler<HTMLLabelElement> = useCallback(
    async (event) => {
      event.preventDefault()
      setIsDragging(false)

      if (isBatchProcessing) {
        setStatusMessage('이미 일괄 처리 중입니다. 잠시만 기다려 주세요.')
        return
      }

      const droppedFiles = Array.from(event.dataTransfer.files ?? [])

      if (droppedFiles.length === 0) {
        return
      }

      if (droppedFiles.length > 1) {
        await processBatchFiles(droppedFiles)
        return
      }

      await loadFile(droppedFiles[0])
    },
    [isBatchProcessing, loadFile, processBatchFiles],
  )

  const downloadMaskedImage = useCallback(async () => {
    const sourceCanvas = sourceCanvasRef.current

    if (!sourceCanvas || !imageMeta) {
      return
    }

    try {
      const exported = await createMaskedJpegExport(
        sourceCanvas,
        regions,
        maskStyle,
        exportOptions,
      )

      const baseName = stripExtension(imageMeta.name)
      const downloadUrl = URL.createObjectURL(exported.blob)

      const anchor = document.createElement('a')
      anchor.download = `${baseName}-masked.jpg`
      anchor.href = downloadUrl
      anchor.click()

      URL.revokeObjectURL(downloadUrl)

      const inputBytes = imageMeta.sizeBytes ?? 0
      const outputBytes = exported.blob.size

      if (inputBytes > 0) {
        const reductionPercent = Math.max(
          0,
          Math.round((1 - outputBytes / inputBytes) * 100),
        )
        setStatusMessage(
          `가려진 JPG 다운로드 완료: ${formatBytes(inputBytes)} → ${formatBytes(outputBytes)} (${reductionPercent}% 절감), ${exported.width}x${exported.height}`,
        )
      } else {
        setStatusMessage('가려진 JPG 이미지를 다운로드했습니다.')
      }
      setIsDownloadGuideOpen(true)
    } catch {
      setStatusMessage('JPG 다운로드 파일 생성에 실패했습니다.')
    }
  }, [exportOptions, imageMeta, maskStyle, regions])

  const shareMaskedImage = useCallback(async () => {
    const sourceCanvas = sourceCanvasRef.current

    if (!sourceCanvas || !imageMeta || !canUseShareSheet) {
      return
    }

    let blob: Blob

    try {
      const exported = await createMaskedJpegExport(
        sourceCanvas,
        regions,
        maskStyle,
        exportOptions,
      )
      blob = exported.blob
    } catch {
      setStatusMessage('공유 파일 생성에 실패했습니다.')
      return
    }

    const baseName = stripExtension(imageMeta.name)
    const file = new File([blob], `${baseName}-masked.jpg`, {
      type: 'image/jpeg',
    })

    const canShareFiles =
      typeof navigator.canShare === 'function'
        ? navigator.canShare({ files: [file] })
        : false

    if (!canShareFiles) {
      setStatusMessage('현재 기기/브라우저는 파일 공유를 지원하지 않습니다.')
      return
    }

    try {
      await navigator.share({
        title: '가려진 사진',
        files: [file],
      })
    } catch (error) {
      if (error instanceof DOMException && error.name === 'AbortError') {
        return
      }

      setStatusMessage('공유 중 오류가 발생했습니다.')
    }
  }, [canUseShareSheet, exportOptions, imageMeta, maskStyle, regions])

  const activeCount = useMemo(
    () => regions.filter((region) => region.active).length,
    [regions],
  )

  const batchTotalCount = batchResults.length
  const batchSuccessCount = batchResults.filter(
    (item) => item.status === 'done',
  ).length
  const batchFailedCount = batchResults.filter(
    (item) => item.status === 'failed',
  ).length
  const batchFinishedCount = batchSuccessCount + batchFailedCount

  const hasImage = Boolean(imageMeta)

  const currentPageTitle = useMemo(() => {
    if (activePage === 'tool') {
      return '자동 가림 툴'
    }

    if (activePage === 'about') {
      return '사이트 소개'
    }

    if (activePage === 'privacy') {
      return '개인정보처리방침'
    }

    return '문의하기'
  }, [activePage])

  const renderToolPage = () => {
    if (!isPrepared) {
      return (
        <section className="app-shell intro-shell">
          <section className="hero intro-hero">
            <p className="eyebrow">모바일 우선 익명화 툴</p>
            <h1>변환 전 데이터 안내</h1>
            <p className="hero-description">
              {ENABLE_PLATE_DETECTION
                ? '기능 시작 시 얼굴/번호판 검출 엔진을 내려받습니다. 한 번 준비하면 같은 브라우저에서는 다시 다운로드 없이 바로 변환할 수 있습니다.'
                : '기능 시작 시 얼굴 검출 엔진을 내려받습니다. 한 번 준비하면 같은 브라우저에서는 다시 다운로드 없이 바로 변환할 수 있습니다. 번호판 기능은 임시 보류 상태입니다.'}
            </p>
          </section>

          <section className="consent-card">
            <p className="consent-title">기능 사용 전 확인</p>
            <ul className="consent-list">
              <li>최초 1회 데이터 사용량: 약 6~25MB</li>
              <li>엔진 다운로드 후: 같은 브라우저에서 추가 다운로드 없음</li>
              <li>사진 자체는 서버 업로드 없이 기기 내에서 처리</li>
              <li>기본 출력은 JPG + 크기 최적화(품질 82%, 긴 변 1920px)</li>
            </ul>

            <div className="consent-note">{preparationMessage}</div>

            {hasCachedEngines && (
              <p className="consent-chip">
                이 브라우저는 이전에 엔진을 내려받았습니다. 대부분 즉시 시작됩니다.
              </p>
            )}

            {isKakaoInApp && (
              <div className="preflight-inapp-card">
                <p className="preflight-inapp-title">카카오톡 내부 브라우저 감지됨</p>
                <p className="preflight-inapp-description">
                  변환을 시작하기 전에 외부 브라우저(Chrome/Edge)에서 이 페이지를 열어 주세요.
                </p>
                <ol className="preflight-inapp-list">
                  <li>카카오톡 상단 메뉴(⋮ 또는 공유)에서 외부 브라우저로 열기 선택</li>
                  <li>Chrome 또는 Edge에서 페이지를 다시 열기</li>
                  <li>외부 브라우저에서 변환 시작</li>
                </ol>
                <button
                  type="button"
                  className="inapp-copy-btn"
                  onClick={() => {
                    void copyCurrentUrl()
                  }}
                >
                  현재 링크 복사
                </button>
              </div>
            )}

            <button
              type="button"
              className="primary-btn"
              onClick={() => {
                void prepareTransformer()
              }}
              disabled={isPreparing || isKakaoInApp}
            >
              {isKakaoInApp
                ? '외부 브라우저에서 열어 주세요'
                : isPreparing
                  ? '엔진 준비 중...'
                  : '변환하러가기'}
            </button>
          </section>
        </section>
      )
    }

    return (
      <div className="app-shell">
        <section className="hero">
          <p className="eyebrow">브라우저 로컬 처리</p>
          <h1>업로드 전에 10초 익명화</h1>
          <p className="hero-description">
            {ENABLE_PLATE_DETECTION
              ? '얼굴과 번호판을 자동으로 찾고 기본값으로 모자이크 처리합니다. 틀린 박스만 빠르게 on/off 하거나 드래그로 추가하면 끝납니다.'
              : '얼굴을 자동으로 찾고 기본값으로 모자이크 처리합니다. 틀린 박스만 빠르게 on/off 하거나 드래그로 추가하면 끝납니다. 번호판 기능은 임시 보류 상태입니다.'}
          </p>
        </section>

        <section className="cache-assurance">
          {!ENABLE_PLATE_DETECTION
            ? '얼굴 엔진 준비가 끝났습니다. 번호판 기능은 임시 보류 상태이며 얼굴 중심 모드로 동작합니다.'
            : isPlateDetectorReady
              ? '엔진 준비가 끝났습니다. 같은 브라우저에서는 추가 다운로드 없이 안심하고 변환할 수 있습니다. 기본 출력은 데이터 절약 JPG 설정이 적용됩니다.'
              : isPlateDetectorWarming
                ? '얼굴 엔진 준비가 끝났습니다. 번호판 엔진은 백그라운드에서 준비 중이며 준비 전에는 얼굴 중심 모드로 동작합니다.'
                : isPlateDetectorFailed
                  ? '얼굴 엔진 준비가 끝났습니다. 번호판 엔진 준비에 실패해 얼굴 중심 모드로 동작합니다.'
                  : '얼굴 엔진 준비가 끝났습니다. 번호판 엔진은 곧 백그라운드에서 준비됩니다.'}
        </section>

        <section className="controls">
          <div className="dropzone-wrapper">
            <label
              className={`dropzone ${isDragging ? 'dragging' : ''}`}
              onDragOver={(event) => {
                event.preventDefault()
                setIsDragging(true)
              }}
              onDragLeave={() => setIsDragging(false)}
              onDrop={onDrop}
            >
              <input
                type="file"
                accept="image/*"
                multiple
                onChange={onPickFile}
                className="file-input"
              />
              <span className="dropzone-title">사진 드래그 또는 클릭 업로드</span>
              <span className="dropzone-subtitle">
                2장 이상 선택/드롭 시 JPG ZIP 일괄 처리 모드로 자동 전환됩니다.
              </span>
            </label>

            <input
              ref={batchInputRef}
              type="file"
              accept="image/*"
              multiple
              onChange={onPickBatchFiles}
              className="file-input"
            />

            <div className="toolbar">
              <button
                type="button"
                onClick={() => {
                  void runAutoScan()
                }}
                disabled={!hasImage || isScanning || isBatchProcessing}
              >
                {isScanning ? '스캔 중...' : '자동 스캔 다시 실행'}
              </button>
              <button
                type="button"
                className={drawModeEnabled ? 'active' : ''}
                disabled={!hasImage || isBatchProcessing}
                onClick={() => {
                  setDrawModeEnabled((prev) => !prev)
                  setDraftRect(null)
                }}
              >
                {drawModeEnabled ? '수동 박스 모드 종료' : '수동 박스 추가'}
              </button>
              <button
                type="button"
                disabled={!hasImage || regions.length === 0 || isBatchProcessing}
                onClick={() => setRegions([])}
              >
                박스 전체 삭제
              </button>
              <button
                type="button"
                disabled={isBatchProcessing}
                onClick={() => batchInputRef.current?.click()}
              >
                {isBatchProcessing ? '일괄 처리 중...' : '여러 장 ZIP 일괄 처리'}
              </button>
            </div>
          </div>

          <div className="picker-row">
            <div className="picker-group">
              <span className="picker-label">마스킹 스타일</span>
              <div className="segmented">
                {(['mosaic', 'blur', 'black'] as const).map((style) => (
                  <button
                    key={style}
                    type="button"
                    className={maskStyle === style ? 'active' : ''}
                    onClick={() => setMaskStyle(style)}
                    disabled={isBatchProcessing}
                  >
                    {style === 'mosaic' ? '모자이크' : style === 'blur' ? '블러' : '검은 박스'}
                  </button>
                ))}
              </div>
            </div>

            <div className="picker-group">
              <span className="picker-label">미리보기</span>
              <div className="segmented segmented-two">
                {(['masked', 'original'] as const).map((mode) => (
                  <button
                    key={mode}
                    type="button"
                    className={previewMode === mode ? 'active' : ''}
                    onClick={() => setPreviewMode(mode)}
                    disabled={!hasImage || isBatchProcessing}
                  >
                    {mode === 'masked' ? '가림본' : '원본'}
                  </button>
                ))}
              </div>
            </div>

            <div className="picker-group">
              <span className="picker-label">데이터 절약(기본값 ON)</span>
              <div className="optimize-box">
                <label className="optimize-toggle">
                  <input
                    type="checkbox"
                    checked={optimizeData}
                    onChange={(event) => setOptimizeData(event.target.checked)}
                    disabled={isBatchProcessing}
                  />
                  <span>용량 절약 모드 사용</span>
                </label>

                <label className="optimize-field">
                  JPG 품질
                  <strong>{Math.round(jpegQuality * 100)}%</strong>
                </label>
                <input
                  type="range"
                  min={60}
                  max={92}
                  step={1}
                  value={Math.round(jpegQuality * 100)}
                  onChange={(event) =>
                    setJpegQuality(Number(event.target.value) / 100)
                  }
                  disabled={!optimizeData || isBatchProcessing}
                />

                <label className="optimize-field">
                  최대 긴 변
                  <strong>{maxLongEdge}px</strong>
                </label>
                <select
                  value={maxLongEdge}
                  onChange={(event) => setMaxLongEdge(Number(event.target.value))}
                  disabled={!optimizeData || isBatchProcessing}
                >
                  <option value={1280}>1280px (초절약)</option>
                  <option value={1600}>1600px</option>
                  <option value={1920}>1920px (권장)</option>
                  <option value={2560}>2560px</option>
                </select>
              </div>
            </div>
          </div>
        </section>

        <section className="workspace">
          <div className="canvas-panel">
            <div
              className="canvas-stage"
              style={
                imageMeta
                  ? { aspectRatio: `${imageMeta.width} / ${imageMeta.height}` }
                  : undefined
              }
            >
              {imageMeta ? (
                <>
                  <canvas ref={canvasRef} className="preview-canvas" />
                  <svg
                    ref={overlayRef}
                    className={`overlay ${drawModeEnabled ? 'draw-mode' : ''}`}
                    viewBox={`0 0 ${imageMeta.width} ${imageMeta.height}`}
                    preserveAspectRatio="none"
                    onPointerDown={onOverlayPointerDown}
                    onPointerMove={onOverlayPointerMove}
                    onPointerUp={onOverlayPointerUp}
                    onPointerCancel={() => {
                      dragStartRef.current = null
                      setDraftRect(null)
                    }}
                  >
                    {regions.map((region, index) => (
                      <g
                        key={region.id}
                        onClick={(event) => {
                          event.stopPropagation()
                          if (!drawModeEnabled) {
                            toggleRegion(region.id)
                          }
                        }}
                      >
                        <rect
                          x={region.x}
                          y={region.y}
                          width={region.width}
                          height={region.height}
                          className={`region-box ${region.kind} ${
                            region.active ? 'active' : 'inactive'
                          }`}
                          vectorEffect="non-scaling-stroke"
                        />
                        <text
                          x={region.x + 4}
                          y={Math.max(12, region.y - 4)}
                          className="region-index"
                        >
                          {index + 1}
                        </text>
                      </g>
                    ))}

                    {draftRect && (
                      <rect
                        x={draftRect.x}
                        y={draftRect.y}
                        width={draftRect.width}
                        height={draftRect.height}
                        className="region-box draft"
                        vectorEffect="non-scaling-stroke"
                      />
                    )}
                  </svg>
                </>
              ) : (
                <div className="empty-state">이미지를 올리면 결과가 여기에 표시됩니다.</div>
              )}
            </div>

            <div className="action-row">
              <button
                type="button"
                onClick={() => {
                  void downloadMaskedImage()
                }}
                disabled={!hasImage}
              >
                가려진 JPG 다운로드
              </button>
              <button
                type="button"
                onClick={() => {
                  void shareMaskedImage()
                }}
                disabled={!hasImage || !canUseShareSheet}
              >
                공유
              </button>
            </div>

            <section className="download-assist">
              {downloadGuide.isKakaoInApp && (
                <div className="inapp-warning-card">
                  <p className="inapp-warning-title">카카오톡 내부 브라우저 감지됨</p>
                  <p className="inapp-warning-description">
                    이 환경에서는 파일 저장이 실패할 수 있습니다. 외부 브라우저에서 열어 주세요.
                  </p>
                  <ol className="inapp-warning-list">
                    <li>카카오톡 상단 메뉴(⋮ 또는 공유)에서 외부 브라우저로 열기를 선택</li>
                    <li>Chrome 또는 Edge에서 페이지를 다시 연 뒤 다운로드 진행</li>
                    <li>브라우저 다운로드 목록에서 저장된 파일 확인</li>
                  </ol>
                  <button
                    type="button"
                    className="inapp-copy-btn"
                    onClick={() => {
                      void copyCurrentUrl()
                    }}
                  >
                    현재 링크 복사
                  </button>
                </div>
              )}

              <div className="download-assist-head">
                <p className="download-assist-caption">{downloadGuide.environmentLabel}</p>
                <button
                  type="button"
                  className={`download-guide-toggle ${isDownloadGuideOpen ? 'active' : ''}`}
                  onClick={() => setIsDownloadGuideOpen((prev) => !prev)}
                  aria-expanded={isDownloadGuideOpen}
                >
                  {isDownloadGuideOpen ? '저장 위치 안내 닫기' : '저장 위치 안내'}
                </button>
              </div>
              <p className="download-assist-note">{downloadHint}</p>

              {isDownloadGuideOpen && (
                <div className="download-guide-card">
                  <p className="download-guide-summary">{downloadGuide.summary}</p>
                  <ol className="download-guide-list">
                    {downloadGuide.steps.map((step) => (
                      <li key={step}>{step}</li>
                    ))}
                  </ol>
                  <p className="download-guide-footnote">{downloadGuide.footnote}</p>
                </div>
              )}
            </section>
          </div>

          <aside className="region-panel">
            <div className="panel-header">
              <h2>검출 결과</h2>
              <p>
                전체 {regions.length}개 / 적용 {activeCount}개
              </p>
            </div>

            <div className="status-box">{statusMessage}</div>

            {batchTotalCount > 0 && (
              <section className="batch-box">
                <div className="batch-header">
                  <h3>일괄 처리 로그</h3>
                  <p>
                    완료 {batchFinishedCount}/{batchTotalCount} · 성공 {batchSuccessCount} ·
                    실패 {batchFailedCount}
                  </p>
                </div>
                <ul className="batch-list">
                  {batchResults.map((item) => (
                    <li key={item.id} className={`batch-item ${item.status}`}>
                      <div className="batch-item-name">{item.fileName}</div>
                      <div className="batch-item-meta">
                        {item.status === 'pending' && '대기 중'}
                        {item.status === 'processing' && '처리 중...'}
                        {item.status === 'done' &&
                          (ENABLE_PLATE_DETECTION
                            ? `얼굴 ${item.faceCount} / 번호판 ${item.plateCount} · ${item.outputWidth ?? '-'}x${item.outputHeight ?? '-'} · ${item.outputBytes ? formatBytes(item.outputBytes) : '-'}`
                            : `얼굴 ${item.faceCount} · ${item.outputWidth ?? '-'}x${item.outputHeight ?? '-'} · ${item.outputBytes ? formatBytes(item.outputBytes) : '-'}`)}
                        {item.status === 'failed' &&
                          `실패: ${item.errorMessage ?? '알 수 없는 오류'}`}
                      </div>
                    </li>
                  ))}
                </ul>
              </section>
            )}

            {regions.length > 0 ? (
              <ul className="region-list">
                {regions.map((region, index) => (
                  <li key={region.id}>
                    <label>
                      <input
                        type="checkbox"
                        checked={region.active}
                        onChange={() => toggleRegion(region.id)}
                      />
                      <span>
                        #{index + 1} {prettyKind(region.kind)}
                      </span>
                    </label>
                    <button type="button" onClick={() => removeRegion(region.id)}>
                      삭제
                    </button>
                  </li>
                ))}
              </ul>
            ) : (
              <p className="panel-empty">
                자동 스캔 결과가 없으면 수동 박스로 추가해 주세요.
              </p>
            )}
          </aside>
        </section>
      </div>
    )
  }

  const renderInformationPage = () => {
    if (activePage === 'about') {
      return (
        <section className="page-card">
          <p className="page-kicker">About</p>
          <h2>모두가 쉽게 쓰는 무료 오픈 툴을 만들고 있습니다.</h2>
          <p>
            이 사이트는 어려운 설치 과정 없이 누구나 사진을 간편하게 가릴 수 있도록 만든
            무료 오픈 웹 툴입니다. 복잡한 지식이 없어도 업로드 후 바로 익명화 작업을 시작할
            수 있도록 설계했습니다.
          </p>
          <p>
            운영자는 더 많은 사람이 부담 없이 개인정보 보호 도구를 사용할 수 있기를 바라며,
            접근성과 편의성을 우선으로 기능을 계속 개선하고 있습니다.
          </p>
        </section>
      )
    }

    if (activePage === 'privacy') {
      return (
        <section className="page-card">
          <p className="page-kicker">Privacy Policy</p>
          <h2>개인정보처리방침</h2>
          <p>
            본 서비스는 사용자의 사진과 개인정보를 서버에 저장하거나 관리하지 않는 것을
            기본 원칙으로 합니다.
          </p>
          <ul className="page-list">
            <li>업로드한 이미지는 브라우저(기기) 내부에서만 처리됩니다.</li>
            <li>회원가입, 로그인, 이름/연락처 수집 기능이 없습니다.</li>
            <li>이미지 파일은 처리 후 자동으로 사용자 환경에서만 남거나 삭제됩니다.</li>
            <li>운영자는 사용자의 이미지 원본/결과물을 별도로 보관하지 않습니다.</li>
            <li>문의 메일을 보낸 경우, 회신 목적 범위에서만 최소한의 정보가 사용됩니다.</li>
          </ul>
          <p>
            따라서 사용자는 별도 계정 생성 없이 안심하고 익명화 기능을 이용할 수 있습니다.
          </p>
        </section>
      )
    }

    return (
      <section className="page-card">
        <p className="page-kicker">Contact</p>
        <h2>문의하기</h2>
        <p>기능 제안, 버그 제보, 협업 문의는 아래 이메일로 보내주세요.</p>
        <a className="contact-link" href="mailto:andreabyfive@gmail.com">
          andreabyfive@gmail.com
        </a>
      </section>
    )
  }

  return (
    <main className="dashboard-shell">
      <aside className="dashboard-sidebar">
        <div className="sidebar-brand">
          <p className="sidebar-eyebrow">Open Tool</p>
          <h2>FaceDetector Web</h2>
        </div>

        <nav className="sidebar-nav">
          {PAGE_ITEMS.map((item) => (
            <button
              key={item.key}
              type="button"
              className={`sidebar-nav-item ${activePage === item.key ? 'active' : ''}`}
              onClick={() => goToPage(item.key)}
            >
              <span className="nav-icon">{item.icon}</span>
              <span>{item.label}</span>
            </button>
          ))}
        </nav>

        <p className="sidebar-footnote">누구나 무료로 쓸 수 있는 익명화 도구</p>
      </aside>

      <section className="dashboard-main">
        <header className="dashboard-topbar">
          <div>
            <p className="topbar-eyebrow">Personal Privacy Toolkit</p>
            <h1>{currentPageTitle}</h1>
          </div>
          <button type="button" className="theme-toggle" onClick={toggleThemeMode}>
            {themeMode === 'dark' ? '☀ 주간 모드' : '🌙 야간 모드'}
          </button>
        </header>

        <div className="dashboard-content">
          {activePage === 'tool' ? renderToolPage() : renderInformationPage()}
        </div>
      </section>
    </main>
  )
}

export default App
