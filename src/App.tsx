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
const BATCH_MAX_FILE_COUNT = 20
const BATCH_MAX_TOTAL_BYTES = 150 * 1024 * 1024
const MOBILE_BATCH_DETECTION_MAX_LONG_EDGE = 1280
const DESKTOP_BATCH_DETECTION_MAX_LONG_EDGE = 1920
const MOBILE_BATCH_COOLDOWN_MS = 40
const MOBILE_SINGLE_MODE_MESSAGE =
  '모바일은 메모리 제약으로 한 번에 한 장만 변환할 수 있습니다.'
const ENGINE_CACHE_KEY = 'face_masker_engine_assets_v1'
const ENABLE_PLATE_DETECTION = false
const FACE_WARMUP_TIMEOUT_MS = 25_000
const PLATE_WARMUP_TIMEOUT_MS = 35_000
const PLATE_DETECTION_TIMEOUT_MS = 20_000
const THEME_STORAGE_KEY = 'face_masker_theme_v1'
const DEFAULT_EDITOR_QUALITY = 0.86
const EDITOR_MIN_CROP_SIZE = 20
const DEFAULT_EDITOR_PDF_COVERAGE = 92
const EDITOR_PDF_MIN_COVERAGE = 10
const EDITOR_PDF_MIN_MARGIN_MM = 4
const A4_WIDTH_MM = 210
const A4_HEIGHT_MM = 297
const EDITOR_CUSTOM_MIN_DIMENSION = 64
const EDITOR_CUSTOM_MAX_DIMENSION = 8000
const KAKAO_ADFIT_SCRIPT_SRC = 'https://t1.daumcdn.net/kas/static/ba.min.js'
const KAKAO_ADFIT_AD_UNIT = 'DAN-Blil5UyPxKSACsgC'
const KAKAO_ADFIT_WIDTH = '320'
const KAKAO_ADFIT_HEIGHT = '100'

type ThemeMode = 'dark' | 'light'
type PageKey = 'tool' | 'editor' | 'about' | 'privacy' | 'contact'
type BatchStatus = 'pending' | 'processing' | 'done' | 'failed'
type EditorOutputFormat = 'jpeg' | 'png' | 'webp' | 'pdf'
type EditorPdfOrientation = 'auto' | 'portrait' | 'landscape'
type EditorResizeMode = 'original' | 'preset-70' | 'preset-50' | 'preset-30' | 'custom'
type EditorCustomResizeMode = 'fit-width' | 'fit-height' | 'frame'
type EditorCustomFrameFit = 'contain' | 'cover'
type EditorPdfPlacement =
  | 'top-left'
  | 'top-center'
  | 'top-right'
  | 'middle-left'
  | 'center'
  | 'middle-right'
  | 'bottom-left'
  | 'bottom-center'
  | 'bottom-right'
type BrowserFamily =
  | 'chrome'
  | 'edge'
  | 'firefox'
  | 'safari'
  | 'samsung'
  | 'other'

const EDITOR_PDF_PLACEMENT_OPTIONS: Array<{
  value: EditorPdfPlacement
  label: string
  symbol: string
}> = [
  { value: 'top-left', label: '좌상단', symbol: '↖' },
  { value: 'top-center', label: '상단 중앙', symbol: '↑' },
  { value: 'top-right', label: '우상단', symbol: '↗' },
  { value: 'middle-left', label: '좌측 중앙', symbol: '←' },
  { value: 'center', label: '정중앙', symbol: '●' },
  { value: 'middle-right', label: '우측 중앙', symbol: '→' },
  { value: 'bottom-left', label: '좌하단', symbol: '↙' },
  { value: 'bottom-center', label: '하단 중앙', symbol: '↓' },
  { value: 'bottom-right', label: '우하단', symbol: '↘' },
]

const EDITOR_RESIZE_MODE_OPTIONS: Array<{
  value: EditorResizeMode
  label: string
}> = [
  { value: 'original', label: '원본 100%' },
  { value: 'preset-70', label: '70%' },
  { value: 'preset-50', label: '50%' },
  { value: 'preset-30', label: '30%' },
  { value: 'custom', label: 'Custom' },
]

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
  outputBlob?: Blob
  previewUrl?: string
  downloadName?: string
  approvedForDownload?: boolean
  sourceFile?: File
  editableRegions?: MaskRegion[]
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

interface EditorOutputPlan {
  sourceWidth: number
  sourceHeight: number
  outputWidth: number
  outputHeight: number
  sourceSampleX: number
  sourceSampleY: number
  sourceSampleWidth: number
  sourceSampleHeight: number
  widthScalePercent: number
  heightScalePercent: number
  areaScalePercent: number
  reducedAreaPercent: number
  sampleAreaPercent: number
}

function parsePageFromHash(hash: string): PageKey {
  const normalized = hash.replace(/^#/, '')

  if (
    normalized === 'tool' ||
    normalized === 'editor' ||
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
  { key: 'tool', label: '프라이버시 마스킹', icon: '🧩' },
  { key: 'editor', label: '이미지 간편 편집', icon: '🛠️' },
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

function getEditorMimeType(format: EditorOutputFormat): string {
  if (format === 'pdf') {
    return 'application/pdf'
  }

  if (format === 'png') {
    return 'image/png'
  }

  if (format === 'webp') {
    return 'image/webp'
  }

  return 'image/jpeg'
}

function getEditorExtension(format: EditorOutputFormat): string {
  if (format === 'pdf') {
    return 'pdf'
  }

  if (format === 'png') {
    return 'png'
  }

  if (format === 'webp') {
    return 'webp'
  }

  return 'jpg'
}

function getEditorQuality(format: EditorOutputFormat, quality: number): number | undefined {
  if (format === 'png') {
    return undefined
  }

  return Math.max(0.6, Math.min(0.96, quality))
}

function getEditorFormatLabel(format: EditorOutputFormat): string {
  if (format === 'jpeg') {
    return 'JPG'
  }

  if (format === 'png') {
    return 'PNG'
  }

  if (format === 'webp') {
    return 'WEBP'
  }

  return 'PDF'
}

function resolveEditorPdfOrientation(
  orientation: EditorPdfOrientation,
  width: number,
  height: number,
): 'portrait' | 'landscape' {
  if (orientation === 'auto') {
    return width > height ? 'landscape' : 'portrait'
  }

  return orientation
}

function getEditorPdfOrientationLabel(
  orientation: EditorPdfOrientation | 'portrait' | 'landscape',
): string {
  if (orientation === 'landscape') {
    return '가로'
  }

  if (orientation === 'portrait') {
    return '세로'
  }

  return '자동'
}

function getEditorPdfPlacementLabel(placement: EditorPdfPlacement): string {
  const option = EDITOR_PDF_PLACEMENT_OPTIONS.find((item) => item.value === placement)
  return option?.label ?? '정중앙'
}

function getEditorPdfPlacementFactors(placement: EditorPdfPlacement): {
  xFactor: number
  yFactor: number
} {
  if (placement === 'top-left') {
    return { xFactor: 0, yFactor: 0 }
  }

  if (placement === 'top-center') {
    return { xFactor: 0.5, yFactor: 0 }
  }

  if (placement === 'top-right') {
    return { xFactor: 1, yFactor: 0 }
  }

  if (placement === 'middle-left') {
    return { xFactor: 0, yFactor: 0.5 }
  }

  if (placement === 'middle-right') {
    return { xFactor: 1, yFactor: 0.5 }
  }

  if (placement === 'bottom-left') {
    return { xFactor: 0, yFactor: 1 }
  }

  if (placement === 'bottom-center') {
    return { xFactor: 0.5, yFactor: 1 }
  }

  if (placement === 'bottom-right') {
    return { xFactor: 1, yFactor: 1 }
  }

  return { xFactor: 0.5, yFactor: 0.5 }
}

function clampNumber(value: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, value))
}

function sanitizeEditorDimension(
  value: number,
  fallback: number,
  max: number,
): number {
  const min = Math.min(EDITOR_CUSTOM_MIN_DIMENSION, max)

  if (!Number.isFinite(value)) {
    return clampNumber(Math.round(fallback), min, max)
  }

  return clampNumber(Math.round(value), min, max)
}

function getEditorResizeModeLabel(mode: EditorResizeMode): string {
  const option = EDITOR_RESIZE_MODE_OPTIONS.find((item) => item.value === mode)
  return option?.label ?? '원본 100%'
}

function sanitizeCropRect(
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

  return { x, y, width, height }
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

function cloneRegions(regions: MaskRegion[]): MaskRegion[] {
  return regions.map((region) => ({ ...region }))
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
  const targetSize = options.optimizeData
    ? calculateTargetSize(
        sourceCanvas.width,
        sourceCanvas.height,
        options.maxLongEdge,
      )
    : { width: sourceCanvas.width, height: sourceCanvas.height }

  const exportSourceCanvas = document.createElement('canvas')
  exportSourceCanvas.width = targetSize.width
  exportSourceCanvas.height = targetSize.height

  const exportSourceContext = exportSourceCanvas.getContext('2d')

  if (exportSourceContext) {
    exportSourceContext.imageSmoothingEnabled = true
    exportSourceContext.imageSmoothingQuality = 'high'
    exportSourceContext.drawImage(
      sourceCanvas,
      0,
      0,
      sourceCanvas.width,
      sourceCanvas.height,
      0,
      0,
      exportSourceCanvas.width,
      exportSourceCanvas.height,
    )
  }

  const scaleX = exportSourceCanvas.width / sourceCanvas.width
  const scaleY = exportSourceCanvas.height / sourceCanvas.height

  const scaledRegions =
    scaleX === 1 && scaleY === 1
      ? regions
      : regions.map((region) => ({
          ...region,
          x: region.x * scaleX,
          y: region.y * scaleY,
          width: region.width * scaleX,
          height: region.height * scaleY,
        }))

  const maskedCanvas = document.createElement('canvas')
  renderPreview(maskedCanvas, exportSourceCanvas, scaledRegions, maskStyle, 'masked')

  const blob = await canvasToBlob(
    maskedCanvas,
    'image/jpeg',
    getJpegQuality(options.optimizeData, options.jpegQuality),
  )

  exportSourceCanvas.width = 0
  exportSourceCanvas.height = 0

  return {
    blob,
    width: maskedCanvas.width,
    height: maskedCanvas.height,
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

function KakaoAdBanner({ slotKey }: { slotKey: string }) {
  const hostRef = useRef<HTMLDivElement | null>(null)
  const [isNoAd, setIsNoAd] = useState(false)

  useEffect(() => {
    if (typeof window === 'undefined' || typeof document === 'undefined') {
      return
    }

    const host = hostRef.current

    if (!host) {
      return
    }

    let isDisposed = false

    const globalWindow = window as unknown as Record<string, unknown>
    const callbackName = `__kakaoAdFitNoAd_${slotKey.replace(/[^a-z0-9_]/gi, '_')}`

    globalWindow[callbackName] = () => {
      if (!isDisposed) {
        setIsNoAd(true)
      }
    }

    host.replaceChildren()

    const ins = document.createElement('ins')
    ins.className = 'kakao_ad_area'
    ins.style.display = 'none'
    ins.setAttribute('data-ad-unit', KAKAO_ADFIT_AD_UNIT)
    ins.setAttribute('data-ad-width', KAKAO_ADFIT_WIDTH)
    ins.setAttribute('data-ad-height', KAKAO_ADFIT_HEIGHT)
    ins.setAttribute('data-ad-onfail', callbackName)
    host.appendChild(ins)

    const script = document.createElement('script')
    script.async = true
    script.type = 'text/javascript'
    script.charset = 'utf-8'
    script.src = KAKAO_ADFIT_SCRIPT_SRC
    host.appendChild(script)

    return () => {
      isDisposed = true
      delete globalWindow[callbackName]
      host.replaceChildren()
    }
  }, [slotKey])

  if (isNoAd) {
    return null
  }

  return (
    <div className="page-ad-slot" aria-label="광고">
      <div ref={hostRef} className="page-ad-slot-inner" />
    </div>
  )
}

function App() {
  const canvasRef = useRef<HTMLCanvasElement | null>(null)
  const overlayRef = useRef<SVGSVGElement | null>(null)
  const sourceCanvasRef = useRef<HTMLCanvasElement | null>(null)
  const sourceImageRef = useRef<HTMLImageElement | null>(null)
  const activeObjectUrlRef = useRef<string | null>(null)
  const batchPreviewUrlsRef = useRef<string[]>([])
  const batchInputRef = useRef<HTMLInputElement | null>(null)
  const dragStartRef = useRef<{ x: number; y: number } | null>(null)
  const idSequenceRef = useRef(1)
  const editorCanvasRef = useRef<HTMLCanvasElement | null>(null)
  const editorSourceCanvasRef = useRef<HTMLCanvasElement | null>(null)
  const editorObjectUrlRef = useRef<string | null>(null)
  const editorInputRef = useRef<HTMLInputElement | null>(null)
  const editorCropStartRef = useRef<{ x: number; y: number } | null>(null)

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
      : '기능을 시작하면 얼굴 검출 엔진을 한 번만 내려받습니다.',
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
  const [isBatchDownloadPending, setIsBatchDownloadPending] = useState(false)
  const [drawModeEnabled, setDrawModeEnabled] = useState(false)
  const [draftRect, setDraftRect] = useState<DetectionRect | null>(null)
  const [batchResults, setBatchResults] = useState<BatchResult[]>([])
  const [selectedBatchResultId, setSelectedBatchResultId] = useState<string | null>(null)
  const [editingBatchResultId, setEditingBatchResultId] = useState<string | null>(null)
  const [isDownloadGuideOpen, setIsDownloadGuideOpen] = useState(false)
  const [statusMessage, setStatusMessage] = useState(
    ENABLE_PLATE_DETECTION
      ? '사진을 올리면 얼굴+번호판 자동 가림이 시작됩니다.'
      : '사진을 올리면 얼굴 자동 가림이 시작됩니다.',
  )
  const [editorImageMeta, setEditorImageMeta] = useState<ImageMeta | null>(null)
  const [editorStatusMessage, setEditorStatusMessage] = useState(
    '이미지를 업로드하면 자르기/크기 조절/포맷 변환(JPG/PNG/WEBP/PDF) 후 다운로드할 수 있습니다.',
  )
  const [editorOutputFormat, setEditorOutputFormat] =
    useState<EditorOutputFormat>('jpeg')
  const [editorResizeMode, setEditorResizeMode] = useState<EditorResizeMode>('original')
  const [editorCustomResizeMode, setEditorCustomResizeMode] =
    useState<EditorCustomResizeMode>('fit-width')
  const [editorCustomFrameFit, setEditorCustomFrameFit] =
    useState<EditorCustomFrameFit>('contain')
  const [editorCustomWidth, setEditorCustomWidth] = useState(1280)
  const [editorCustomHeight, setEditorCustomHeight] = useState(1280)
  const [editorCustomPlacement, setEditorCustomPlacement] =
    useState<EditorPdfPlacement>('center')
  const [editorPdfOrientation, setEditorPdfOrientation] =
    useState<EditorPdfOrientation>('auto')
  const [editorPdfPlacement, setEditorPdfPlacement] =
    useState<EditorPdfPlacement>('center')
  const [editorPdfCoverage, setEditorPdfCoverage] = useState(
    DEFAULT_EDITOR_PDF_COVERAGE,
  )
  const [editorQuality, setEditorQuality] = useState(DEFAULT_EDITOR_QUALITY)
  const [editorCropRect, setEditorCropRect] = useState<DetectionRect | null>(null)
  const [editorCropDraft, setEditorCropDraft] = useState<DetectionRect | null>(null)
  const [isEditorCropMode, setIsEditorCropMode] = useState(false)
  const [isEditorDragging, setIsEditorDragging] = useState(false)
  const [isEditorExporting, setIsEditorExporting] = useState(false)

  const canUseShareSheet =
    typeof navigator !== 'undefined' && typeof navigator.share === 'function'

  const downloadGuide = useMemo(() => buildDownloadGuide(), [])
  const isMobileMode = downloadGuide.isMobile
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

  const revokeBatchPreviewUrls = useCallback(() => {
    if (batchPreviewUrlsRef.current.length === 0) {
      return
    }

    for (const previewUrl of batchPreviewUrlsRef.current) {
      URL.revokeObjectURL(previewUrl)
    }

    batchPreviewUrlsRef.current = []
  }, [])

  const registerBatchPreviewUrl = useCallback((previewUrl: string) => {
    batchPreviewUrlsRef.current.push(previewUrl)
  }, [])

  const unregisterBatchPreviewUrl = useCallback((previewUrl: string) => {
    batchPreviewUrlsRef.current = batchPreviewUrlsRef.current.filter(
      (url) => url !== previewUrl,
    )
  }, [])

  const clearBatchReviewState = useCallback(() => {
    revokeBatchPreviewUrls()
    setBatchResults([])
    setSelectedBatchResultId(null)
    setEditingBatchResultId(null)
    setIsBatchDownloadPending(false)
  }, [revokeBatchPreviewUrls])

  useEffect(() => {
    return () => {
      if (activeObjectUrlRef.current) {
        URL.revokeObjectURL(activeObjectUrlRef.current)
      }

      if (editorObjectUrlRef.current) {
        URL.revokeObjectURL(editorObjectUrlRef.current)
      }

      revokeBatchPreviewUrls()
    }
  }, [revokeBatchPreviewUrls])

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

  const redrawEditorCanvas = useCallback(() => {
    const canvas = editorCanvasRef.current
    const sourceCanvas = editorSourceCanvasRef.current

    if (!canvas || !sourceCanvas) {
      return
    }

    canvas.width = sourceCanvas.width
    canvas.height = sourceCanvas.height

    const context = canvas.getContext('2d')

    if (!context) {
      return
    }

    context.clearRect(0, 0, canvas.width, canvas.height)
    context.drawImage(sourceCanvas, 0, 0)
  }, [])

  useEffect(() => {
    redrawEditorCanvas()
  }, [editorImageMeta, redrawEditorCanvas])

  useEffect(() => {
    if (activePage !== 'editor' || !editorImageMeta) {
      return
    }

    const animationId = window.requestAnimationFrame(() => {
      redrawEditorCanvas()
    })

    return () => {
      window.cancelAnimationFrame(animationId)
    }
  }, [activePage, editorImageMeta, redrawEditorCanvas])

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
          : '준비 완료. 같은 브라우저에서는 추가 다운로드 없이 바로 변환할 수 있습니다.',
      )
      setStatusMessage(
        ENABLE_PLATE_DETECTION
          ? '엔진 준비 완료: 얼굴 엔진은 즉시 사용 가능하며 번호판 엔진은 백그라운드에서 준비됩니다.'
          : '엔진 준비 완료: 얼굴 엔진을 사용할 수 있습니다.',
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
            : `자동 스캔 완료: 얼굴 ${faces.length}개`,
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
      clearBatchReviewState()

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
    [clearBatchReviewState, isBatchProcessing, isPrepared, runAutoScan],
  )

  const processBatchFiles = useCallback(
    async (files: File[]) => {
      if (isMobileMode) {
        setStatusMessage(MOBILE_SINGLE_MODE_MESSAGE)
        return
      }

      if (!isPrepared) {
        setStatusMessage('먼저 변환 준비를 완료해 주세요.')
        return
      }

      const imageFiles = files.filter(isImageFile)

      if (imageFiles.length < 2) {
        setStatusMessage('일괄 처리는 이미지 2장 이상에서 동작합니다.')
        return
      }

      if (imageFiles.length > BATCH_MAX_FILE_COUNT) {
        setStatusMessage(
          `일괄 처리 제한: 최대 ${BATCH_MAX_FILE_COUNT}장까지 선택할 수 있습니다. 현재 ${imageFiles.length}장 선택됨`,
        )
        return
      }

      const totalBytes = imageFiles.reduce((sum, file) => sum + file.size, 0)
      if (totalBytes > BATCH_MAX_TOTAL_BYTES) {
        setStatusMessage(
          `일괄 처리 제한: 총 용량은 최대 ${formatBytes(BATCH_MAX_TOTAL_BYTES)}까지 가능합니다. 현재 ${formatBytes(totalBytes)}`,
        )
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

      clearBatchReviewState()
      if (activeObjectUrlRef.current) {
        URL.revokeObjectURL(activeObjectUrlRef.current)
        activeObjectUrlRef.current = null
      }
      sourceImageRef.current = null
      if (sourceCanvasRef.current) {
        sourceCanvasRef.current.width = 0
        sourceCanvasRef.current.height = 0
      }
      sourceCanvasRef.current = null
      setImageMeta(null)
      setRegions([])

      setIsBatchProcessing(true)
      setDrawModeEnabled(false)
      setDraftRect(null)
      setIsBatchDownloadPending(false)

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
          : `${imageFiles.length}장 일괄 처리를 시작합니다.`
      setBatchResults(initialResults)
      setStatusMessage(
        `${batchStartBaseMessage} 완료 후 썸네일에서 결과를 확인하고 최종 승인 다운로드를 진행할 수 있습니다.`,
      )

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

            const autoRegions = createAutoRegions(faces, plates)
            const exported = await createMaskedJpegExport(
              sourceCanvas,
              autoRegions,
              maskStyle,
              exportOptions,
            )
            const previewUrl = URL.createObjectURL(exported.blob)
            const downloadName = `${stripExtension(file.name)}-masked.jpg`
            registerBatchPreviewUrl(previewUrl)

            outputBytesTotal += exported.blob.size
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
                      outputBlob: exported.blob,
                      previewUrl,
                      downloadName,
                      approvedForDownload: true,
                      sourceFile: file,
                      editableRegions: cloneRegions(autoRegions),
                    }
                  : item,
              ),
            )
            setSelectedBatchResultId((prev) => prev ?? resultId)
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

        const plateModeNote = plateDetectionEnabled
          ? ''
          : ENABLE_PLATE_DETECTION
            ? ` · 번호판 엔진 비활성화(얼굴 중심 모드${plateDetectionErrorMessage ? `: ${plateDetectionErrorMessage}` : ''})`
            : ''

        if (successCount === 0) {
          setStatusMessage(`일괄 처리 실패: 생성된 이미지가 없습니다.${plateModeNote}`)
        } else if (failureCount > 0) {
          const reductionText =
            outputBytesTotal > 0 && inputBytesTotal > 0
              ? ` · 총 ${formatBytes(inputBytesTotal)} → ${formatBytes(outputBytesTotal)}`
              : ''
          setStatusMessage(
            `일괄 처리 완료: 성공 ${successCount}장 / 실패 ${failureCount}장${reductionText}${plateModeNote} · 썸네일 확인 후 최종 승인 ZIP 다운로드를 눌러 주세요.`,
          )
        } else {
          const reductionText =
            outputBytesTotal > 0 && inputBytesTotal > 0
              ? ` (${formatBytes(inputBytesTotal)} → ${formatBytes(outputBytesTotal)})`
              : ''
          setStatusMessage(
            `일괄 처리 완료: ${successCount}장 처리됨${reductionText}${plateModeNote} · 썸네일 확인 후 최종 승인 ZIP 다운로드를 눌러 주세요.`,
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
      clearBatchReviewState,
      downloadGuide.isMobile,
      exportOptions,
      isMobileMode,
      isPlateDetectorReady,
      isPrepared,
      isScanning,
      maskStyle,
      registerBatchPreviewUrl,
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
        if (isMobileMode) {
          setStatusMessage(MOBILE_SINGLE_MODE_MESSAGE)
          event.target.value = ''
          return
        }

        await processBatchFiles(files)
        event.target.value = ''
        return
      }

      await loadFile(files[0])
      event.target.value = ''
    },
    [isMobileMode, loadFile, processBatchFiles],
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
        if (isMobileMode) {
          setStatusMessage(MOBILE_SINGLE_MODE_MESSAGE)
          return
        }

        await processBatchFiles(droppedFiles)
        return
      }

      await loadFile(droppedFiles[0])
    },
    [isBatchProcessing, isMobileMode, loadFile, processBatchFiles],
  )

  const loadEditorFile = useCallback(async (file: File) => {
    if (!isImageFile(file)) {
      setEditorStatusMessage('이미지 파일만 편집할 수 있습니다.')
      return
    }

    setEditorCropDraft(null)
    setIsEditorCropMode(false)

    try {
      const { image, objectUrl } = await loadImageFromFile(file)

      if (editorObjectUrlRef.current) {
        URL.revokeObjectURL(editorObjectUrlRef.current)
      }
      if (editorSourceCanvasRef.current) {
        editorSourceCanvasRef.current.width = 0
        editorSourceCanvasRef.current.height = 0
      }

      editorObjectUrlRef.current = objectUrl
      editorSourceCanvasRef.current = createSourceCanvas(image)

      setEditorImageMeta({
        name: file.name,
        width: image.naturalWidth,
        height: image.naturalHeight,
        sizeBytes: file.size,
      })
      setEditorCropRect({
        x: 0,
        y: 0,
        width: image.naturalWidth,
        height: image.naturalHeight,
      })
      setEditorResizeMode('original')
      setEditorCustomResizeMode('fit-width')
      setEditorCustomFrameFit('contain')
      setEditorCustomPlacement('center')
      setEditorCustomWidth(image.naturalWidth)
      setEditorCustomHeight(image.naturalHeight)
      setEditorStatusMessage(
        '이미지를 불러왔습니다. 자르기/크기/포맷을 설정한 뒤 다운로드하세요.',
      )
    } catch (error) {
      const message = toErrorMessage(error, '이미지 로딩 중 오류가 발생했습니다.')
      setEditorStatusMessage(message)
    }
  }, [])

  const onPickEditorFile: React.ChangeEventHandler<HTMLInputElement> = useCallback(
    async (event) => {
      const files = Array.from(event.target.files ?? [])

      if (files.length === 0) {
        return
      }

      await loadEditorFile(files[0])

      if (files.length > 1) {
        setEditorStatusMessage(
          '이미지 간편 편집은 한 번에 한 장씩 처리합니다. 첫 번째 이미지를 불러왔습니다.',
        )
      }

      event.target.value = ''
    },
    [loadEditorFile],
  )

  const onEditorDrop: React.DragEventHandler<HTMLLabelElement> = useCallback(
    async (event) => {
      event.preventDefault()
      setIsEditorDragging(false)

      const droppedFiles = Array.from(event.dataTransfer.files ?? [])

      if (droppedFiles.length === 0) {
        return
      }

      await loadEditorFile(droppedFiles[0])

      if (droppedFiles.length > 1) {
        setEditorStatusMessage(
          '이미지 간편 편집은 한 번에 한 장씩 처리합니다. 첫 번째 이미지를 불러왔습니다.',
        )
      }
    },
    [loadEditorFile],
  )

  const getPointOnEditorImage = useCallback(
    (event: React.PointerEvent<SVGSVGElement>) => {
      if (!editorImageMeta) {
        return null
      }

      const rect = event.currentTarget.getBoundingClientRect()

      if (rect.width === 0 || rect.height === 0) {
        return null
      }

      const x = ((event.clientX - rect.left) / rect.width) * editorImageMeta.width
      const y = ((event.clientY - rect.top) / rect.height) * editorImageMeta.height

      return {
        x: Math.max(0, Math.min(x, editorImageMeta.width)),
        y: Math.max(0, Math.min(y, editorImageMeta.height)),
      }
    },
    [editorImageMeta],
  )

  const onEditorOverlayPointerDown = useCallback(
    (event: React.PointerEvent<SVGSVGElement>) => {
      if (!isEditorCropMode || !editorImageMeta) {
        return
      }

      const point = getPointOnEditorImage(event)

      if (!point) {
        return
      }

      event.preventDefault()
      event.currentTarget.setPointerCapture(event.pointerId)
      editorCropStartRef.current = point
      setEditorCropDraft({ x: point.x, y: point.y, width: 0, height: 0 })
    },
    [editorImageMeta, getPointOnEditorImage, isEditorCropMode],
  )

  const onEditorOverlayPointerMove = useCallback(
    (event: React.PointerEvent<SVGSVGElement>) => {
      if (!isEditorCropMode) {
        return
      }

      const start = editorCropStartRef.current

      if (!start) {
        return
      }

      const point = getPointOnEditorImage(event)

      if (!point) {
        return
      }

      setEditorCropDraft({
        x: Math.min(start.x, point.x),
        y: Math.min(start.y, point.y),
        width: Math.abs(start.x - point.x),
        height: Math.abs(start.y - point.y),
      })
    },
    [getPointOnEditorImage, isEditorCropMode],
  )

  const onEditorOverlayPointerUp = useCallback(
    (event: React.PointerEvent<SVGSVGElement>) => {
      if (!isEditorCropMode || !editorImageMeta) {
        return
      }

      const start = editorCropStartRef.current

      if (!start) {
        return
      }

      const point = getPointOnEditorImage(event)
      editorCropStartRef.current = null
      event.currentTarget.releasePointerCapture(event.pointerId)

      if (!point) {
        setEditorCropDraft(null)
        return
      }

      const nextRect = {
        x: Math.min(start.x, point.x),
        y: Math.min(start.y, point.y),
        width: Math.abs(start.x - point.x),
        height: Math.abs(start.y - point.y),
      }
      setEditorCropDraft(null)

      if (
        nextRect.width < EDITOR_MIN_CROP_SIZE ||
        nextRect.height < EDITOR_MIN_CROP_SIZE
      ) {
        setEditorStatusMessage(
          `자르기 영역은 최소 ${EDITOR_MIN_CROP_SIZE}px 이상으로 지정해 주세요.`,
        )
        return
      }

      const sanitized = sanitizeCropRect(
        nextRect,
        editorImageMeta.width,
        editorImageMeta.height,
      )

      if (!sanitized) {
        setEditorStatusMessage('자르기 영역을 다시 지정해 주세요.')
        return
      }

      setEditorCropRect(sanitized)
      setEditorStatusMessage(
        `자르기 영역을 적용했습니다. (${Math.round(sanitized.width)}x${Math.round(sanitized.height)})`,
      )
    },
    [editorImageMeta, getPointOnEditorImage, isEditorCropMode],
  )

  const resetEditorCrop = useCallback(() => {
    if (!editorImageMeta) {
      return
    }

    setEditorCropDraft(null)
    setEditorCropRect({
      x: 0,
      y: 0,
      width: editorImageMeta.width,
      height: editorImageMeta.height,
    })
    setEditorStatusMessage('자르기 영역을 전체 이미지로 초기화했습니다.')
  }, [editorImageMeta])

  const openBatchItemForManualEdit = useCallback(
    async (id: string) => {
      const target = batchResults.find(
        (item) => item.id === id && item.status === 'done',
      )

      if (!target || !target.sourceFile) {
        setStatusMessage('선택한 일괄 결과를 편집용 캔버스로 열 수 없습니다.')
        return
      }

      setDrawModeEnabled(false)
      setDraftRect(null)
      setPreviewMode('masked')

      try {
        const { image, objectUrl } = await loadImageFromFile(target.sourceFile)

        if (activeObjectUrlRef.current) {
          URL.revokeObjectURL(activeObjectUrlRef.current)
        }

        activeObjectUrlRef.current = objectUrl
        sourceImageRef.current = image
        sourceCanvasRef.current = createSourceCanvas(image)

        setImageMeta({
          name: target.fileName,
          width: image.naturalWidth,
          height: image.naturalHeight,
          sizeBytes: target.sourceFile.size,
        })
        setRegions(cloneRegions(target.editableRegions ?? []))
        setSelectedBatchResultId(target.id)
        setEditingBatchResultId(target.id)
        setStatusMessage(
          `일괄 수동 보정 모드: ${target.fileName}. 박스를 수정한 뒤 "현재 보정 저장"을 눌러 반영해 주세요.`,
        )
      } catch (error) {
        const message = toErrorMessage(error, '편집용 이미지 로딩 중 오류가 발생했습니다.')
        setStatusMessage(message)
      }
    },
    [batchResults],
  )

  const saveBatchItemEdits = useCallback(async () => {
    if (!editingBatchResultId) {
      setStatusMessage('먼저 썸네일에서 편집할 이미지를 열어 주세요.')
      return
    }

    const sourceCanvas = sourceCanvasRef.current

    if (!sourceCanvas) {
      setStatusMessage('편집 캔버스를 찾지 못했습니다. 이미지를 다시 열어 주세요.')
      return
    }

    const target = batchResults.find(
      (item) => item.id === editingBatchResultId && item.status === 'done',
    )

    if (!target) {
      setStatusMessage('선택한 일괄 항목을 찾지 못했습니다.')
      return
    }

    try {
      const exported = await createMaskedJpegExport(
        sourceCanvas,
        regions,
        maskStyle,
        exportOptions,
      )
      const nextPreviewUrl = URL.createObjectURL(exported.blob)

      if (target.previewUrl) {
        URL.revokeObjectURL(target.previewUrl)
        unregisterBatchPreviewUrl(target.previewUrl)
      }
      registerBatchPreviewUrl(nextPreviewUrl)

      setBatchResults((prev) =>
        prev.map((item) =>
          item.id === editingBatchResultId
            ? {
                ...item,
                outputBlob: exported.blob,
                outputBytes: exported.blob.size,
                outputWidth: exported.width,
                outputHeight: exported.height,
                previewUrl: nextPreviewUrl,
                approvedForDownload: true,
                editableRegions: cloneRegions(regions),
              }
            : item,
        ),
      )

      setStatusMessage(`보정 저장 완료: ${target.fileName}`)
    } catch (error) {
      const message = toErrorMessage(error, '보정 저장 중 오류가 발생했습니다.')
      setStatusMessage(message)
    }
  }, [
    batchResults,
    editingBatchResultId,
    exportOptions,
    maskStyle,
    regions,
    registerBatchPreviewUrl,
    unregisterBatchPreviewUrl,
  ])

  const toggleBatchApproval = useCallback((id: string) => {
    setBatchResults((prev) =>
      prev.map((item) =>
        item.id === id && item.status === 'done'
          ? { ...item, approvedForDownload: !item.approvedForDownload }
          : item,
      ),
    )
  }, [])

  const setBatchApprovalForAllDone = useCallback((approved: boolean) => {
    setBatchResults((prev) =>
      prev.map((item) =>
        item.status === 'done' ? { ...item, approvedForDownload: approved } : item,
      ),
    )
  }, [])

  const downloadApprovedBatch = useCallback(async () => {
    if (isBatchProcessing || isBatchDownloadPending) {
      return
    }

    const approvedItems = batchResults.filter(
      (item) => item.status === 'done' && item.approvedForDownload && item.outputBlob,
    )

    if (approvedItems.length === 0) {
      setStatusMessage('다운로드 승인을 받은 결과가 없습니다. 썸네일에서 확인 후 승인해 주세요.')
      return
    }

    setIsBatchDownloadPending(true)

    try {
      const zip = new JSZip()

      for (const item of approvedItems) {
        if (!item.outputBlob) {
          continue
        }

        zip.file(
          item.downloadName ?? `${stripExtension(item.fileName)}-masked.jpg`,
          item.outputBlob,
        )
      }

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
      setStatusMessage(
        `최종 승인 다운로드 완료: ${approvedItems.length}장 ZIP 저장을 시작했습니다.`,
      )
    } catch (error) {
      const message = toErrorMessage(error, '승인 ZIP 생성 중 오류가 발생했습니다.')
      setStatusMessage(`승인 ZIP 다운로드 실패: ${message}`)
    } finally {
      setIsBatchDownloadPending(false)
    }
  }, [batchResults, isBatchDownloadPending, isBatchProcessing])

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
          `프라이버시 이미지 다운로드 완료: ${formatBytes(inputBytes)} → ${formatBytes(outputBytes)} (${reductionPercent}% 절감), ${exported.width}x${exported.height}`,
        )
      } else {
        setStatusMessage('프라이버시 이미지를 다운로드했습니다.')
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

  const normalizedEditorCrop = useMemo(() => {
    if (!editorImageMeta) {
      return null
    }

    const fallback = {
      x: 0,
      y: 0,
      width: editorImageMeta.width,
      height: editorImageMeta.height,
    }

    if (!editorCropRect) {
      return fallback
    }

    return (
      sanitizeCropRect(editorCropRect, editorImageMeta.width, editorImageMeta.height) ??
      fallback
    )
  }, [editorCropRect, editorImageMeta])

  const editorDisplayCrop = useMemo(() => {
    if (!editorImageMeta) {
      return null
    }

    if (editorCropDraft) {
      return (
        sanitizeCropRect(editorCropDraft, editorImageMeta.width, editorImageMeta.height) ??
        normalizedEditorCrop
      )
    }

    return normalizedEditorCrop
  }, [editorCropDraft, editorImageMeta, normalizedEditorCrop])

  const editorCustomDimensionLimits = useMemo(() => {
    if (!normalizedEditorCrop) {
      return null
    }

    const sourceWidth = Math.max(1, Math.round(normalizedEditorCrop.width))
    const sourceHeight = Math.max(1, Math.round(normalizedEditorCrop.height))

    return {
      maxWidth: Math.max(1, Math.min(EDITOR_CUSTOM_MAX_DIMENSION, sourceWidth)),
      maxHeight: Math.max(1, Math.min(EDITOR_CUSTOM_MAX_DIMENSION, sourceHeight)),
    }
  }, [normalizedEditorCrop])

  useEffect(() => {
    if (!editorCustomDimensionLimits) {
      return
    }

    setEditorCustomWidth((prev) =>
      sanitizeEditorDimension(
        prev,
        editorCustomDimensionLimits.maxWidth,
        editorCustomDimensionLimits.maxWidth,
      ),
    )
    setEditorCustomHeight((prev) =>
      sanitizeEditorDimension(
        prev,
        editorCustomDimensionLimits.maxHeight,
        editorCustomDimensionLimits.maxHeight,
      ),
    )
  }, [editorCustomDimensionLimits])

  const editorCustomPlacementFactors = useMemo(
    () => getEditorPdfPlacementFactors(editorCustomPlacement),
    [editorCustomPlacement],
  )

  const editorOutputPlan = useMemo<EditorOutputPlan | null>(() => {
    if (!normalizedEditorCrop) {
      return null
    }

    const sourceWidth = Math.max(1, Math.round(normalizedEditorCrop.width))
    const sourceHeight = Math.max(1, Math.round(normalizedEditorCrop.height))
    const sourceArea = sourceWidth * sourceHeight
    const sourceAspect = sourceWidth / sourceHeight

    let outputWidth = sourceWidth
    let outputHeight = sourceHeight
    let sourceSampleX = 0
    let sourceSampleY = 0
    let sourceSampleWidth = sourceWidth
    let sourceSampleHeight = sourceHeight

    if (editorResizeMode === 'preset-70') {
      outputWidth = Math.max(1, Math.round(sourceWidth * 0.7))
      outputHeight = Math.max(1, Math.round(sourceHeight * 0.7))
    } else if (editorResizeMode === 'preset-50') {
      outputWidth = Math.max(1, Math.round(sourceWidth * 0.5))
      outputHeight = Math.max(1, Math.round(sourceHeight * 0.5))
    } else if (editorResizeMode === 'preset-30') {
      outputWidth = Math.max(1, Math.round(sourceWidth * 0.3))
      outputHeight = Math.max(1, Math.round(sourceHeight * 0.3))
    } else if (editorResizeMode === 'custom') {
      const maxWidth = Math.max(
        1,
        Math.min(EDITOR_CUSTOM_MAX_DIMENSION, sourceWidth),
      )
      const maxHeight = Math.max(
        1,
        Math.min(EDITOR_CUSTOM_MAX_DIMENSION, sourceHeight),
      )
      const customWidth = sanitizeEditorDimension(
        editorCustomWidth,
        sourceWidth,
        maxWidth,
      )
      const customHeight = sanitizeEditorDimension(
        editorCustomHeight,
        sourceHeight,
        maxHeight,
      )

      if (editorCustomResizeMode === 'fit-width') {
        outputWidth = customWidth
        outputHeight = Math.max(
          1,
          Math.round(sourceHeight * (customWidth / sourceWidth)),
        )
      } else if (editorCustomResizeMode === 'fit-height') {
        outputHeight = customHeight
        outputWidth = Math.max(
          1,
          Math.round(sourceWidth * (customHeight / sourceHeight)),
        )
      } else if (editorCustomFrameFit === 'cover') {
        outputWidth = customWidth
        outputHeight = customHeight

        const targetAspect = customWidth / customHeight

        if (sourceAspect > targetAspect) {
          sourceSampleHeight = sourceHeight
          sourceSampleWidth = Math.max(1, Math.round(sourceHeight * targetAspect))
        } else {
          sourceSampleWidth = sourceWidth
          sourceSampleHeight = Math.max(1, Math.round(sourceWidth / targetAspect))
        }

        sourceSampleX = Math.round(
          (sourceWidth - sourceSampleWidth) * editorCustomPlacementFactors.xFactor,
        )
        sourceSampleY = Math.round(
          (sourceHeight - sourceSampleHeight) * editorCustomPlacementFactors.yFactor,
        )
      } else {
        const fitScale = Math.min(
          customWidth / sourceWidth,
          customHeight / sourceHeight,
        )
        outputWidth = Math.max(1, Math.round(sourceWidth * fitScale))
        outputHeight = Math.max(1, Math.round(sourceHeight * fitScale))
      }
    }

    sourceSampleX = clampNumber(sourceSampleX, 0, Math.max(0, sourceWidth - sourceSampleWidth))
    sourceSampleY = clampNumber(sourceSampleY, 0, Math.max(0, sourceHeight - sourceSampleHeight))

    const widthScalePercent =
      Math.round((outputWidth / sourceWidth) * 1000) / 10
    const heightScalePercent =
      Math.round((outputHeight / sourceHeight) * 1000) / 10
    const areaScalePercent =
      Math.round(((outputWidth * outputHeight) / sourceArea) * 1000) / 10
    const reducedAreaPercent = Math.max(0, Math.round((100 - areaScalePercent) * 10) / 10)
    const sampleAreaPercent =
      Math.round(((sourceSampleWidth * sourceSampleHeight) / sourceArea) * 1000) / 10

    return {
      sourceWidth,
      sourceHeight,
      outputWidth,
      outputHeight,
      sourceSampleX,
      sourceSampleY,
      sourceSampleWidth,
      sourceSampleHeight,
      widthScalePercent,
      heightScalePercent,
      areaScalePercent,
      reducedAreaPercent,
      sampleAreaPercent,
    }
  }, [
    normalizedEditorCrop,
    editorResizeMode,
    editorCustomResizeMode,
    editorCustomFrameFit,
    editorCustomWidth,
    editorCustomHeight,
    editorCustomPlacementFactors,
  ])

  const editorOutputSize = useMemo(() => {
    if (!editorOutputPlan) {
      return null
    }

    return {
      width: editorOutputPlan.outputWidth,
      height: editorOutputPlan.outputHeight,
    }
  }, [editorOutputPlan])

  const editorResizePreviewLayout = useMemo(() => {
    if (!editorOutputPlan) {
      return null
    }

    const outputWidthPercent =
      (editorOutputPlan.outputWidth / editorOutputPlan.sourceWidth) * 100
    const outputHeightPercent =
      (editorOutputPlan.outputHeight / editorOutputPlan.sourceHeight) * 100
    const sampleLeftPercent =
      (editorOutputPlan.sourceSampleX / editorOutputPlan.sourceWidth) * 100
    const sampleTopPercent =
      (editorOutputPlan.sourceSampleY / editorOutputPlan.sourceHeight) * 100
    const sampleWidthPercent =
      (editorOutputPlan.sourceSampleWidth / editorOutputPlan.sourceWidth) * 100
    const sampleHeightPercent =
      (editorOutputPlan.sourceSampleHeight / editorOutputPlan.sourceHeight) * 100

    return {
      stageAspectRatio: `${editorOutputPlan.sourceWidth} / ${editorOutputPlan.sourceHeight}`,
      outputLeftPercent: Math.max(0, (100 - outputWidthPercent) / 2),
      outputTopPercent: Math.max(0, (100 - outputHeightPercent) / 2),
      outputWidthPercent,
      outputHeightPercent,
      sampleLeftPercent,
      sampleTopPercent,
      sampleWidthPercent,
      sampleHeightPercent,
      hasSourceCrop:
        editorOutputPlan.sourceSampleWidth < editorOutputPlan.sourceWidth ||
        editorOutputPlan.sourceSampleHeight < editorOutputPlan.sourceHeight,
    }
  }, [editorOutputPlan])

  const resolvedEditorPdfOrientation = useMemo(() => {
    if (!editorOutputSize) {
      return 'portrait'
    }

    return resolveEditorPdfOrientation(
      editorPdfOrientation,
      editorOutputSize.width,
      editorOutputSize.height,
    )
  }, [editorOutputSize, editorPdfOrientation])

  const normalizedEditorPdfCoverage = useMemo(
    () => Math.max(EDITOR_PDF_MIN_COVERAGE, Math.min(100, editorPdfCoverage)),
    [editorPdfCoverage],
  )

  const editorPdfPlacementFactors = useMemo(
    () => getEditorPdfPlacementFactors(editorPdfPlacement),
    [editorPdfPlacement],
  )

  const editorPdfPreviewLayout = useMemo(() => {
    if (!editorOutputSize) {
      return null
    }

    const safeXMm = EDITOR_PDF_MIN_MARGIN_MM
    const safeYMm = EDITOR_PDF_MIN_MARGIN_MM
    const pageWidthMm =
      resolvedEditorPdfOrientation === 'landscape' ? A4_HEIGHT_MM : A4_WIDTH_MM
    const pageHeightMm =
      resolvedEditorPdfOrientation === 'landscape' ? A4_WIDTH_MM : A4_HEIGHT_MM
    const safeWidthMm = Math.max(1, pageWidthMm - EDITOR_PDF_MIN_MARGIN_MM * 2)
    const safeHeightMm = Math.max(1, pageHeightMm - EDITOR_PDF_MIN_MARGIN_MM * 2)
    const coverageRatio = normalizedEditorPdfCoverage / 100
    const placementWidthMm = Math.max(1, safeWidthMm * coverageRatio)
    const placementHeightMm = Math.max(1, safeHeightMm * coverageRatio)
    const placementXMm =
      safeXMm + (safeWidthMm - placementWidthMm) * editorPdfPlacementFactors.xFactor
    const placementYMm =
      safeYMm + (safeHeightMm - placementHeightMm) * editorPdfPlacementFactors.yFactor
    const fitScale = Math.min(
      placementWidthMm / editorOutputSize.width,
      placementHeightMm / editorOutputSize.height,
    )
    const imageWidthMm = Math.max(1, editorOutputSize.width * fitScale)
    const imageHeightMm = Math.max(1, editorOutputSize.height * fitScale)
    const imageX =
      placementXMm + (placementWidthMm - imageWidthMm) * editorPdfPlacementFactors.xFactor
    const imageY =
      placementYMm + (placementHeightMm - imageHeightMm) * editorPdfPlacementFactors.yFactor

    return {
      pageAspectRatio: `${pageWidthMm} / ${pageHeightMm}`,
      safeLeftPercent: (safeXMm / pageWidthMm) * 100,
      safeTopPercent: (safeYMm / pageHeightMm) * 100,
      safeWidthPercent: (safeWidthMm / pageWidthMm) * 100,
      safeHeightPercent: (safeHeightMm / pageHeightMm) * 100,
      placementLeftPercent: (placementXMm / pageWidthMm) * 100,
      placementTopPercent: (placementYMm / pageHeightMm) * 100,
      placementWidthPercent: (placementWidthMm / pageWidthMm) * 100,
      placementHeightPercent: (placementHeightMm / pageHeightMm) * 100,
      imageLeftPercent: (imageX / pageWidthMm) * 100,
      imageTopPercent: (imageY / pageHeightMm) * 100,
      imageWidthPercent: (imageWidthMm / pageWidthMm) * 100,
      imageHeightPercent: (imageHeightMm / pageHeightMm) * 100,
      imageAreaPercent:
        Math.round(((imageWidthMm * imageHeightMm) / (pageWidthMm * pageHeightMm)) * 1000) /
        10,
    }
  }, [
    editorOutputSize,
    normalizedEditorPdfCoverage,
    resolvedEditorPdfOrientation,
    editorPdfPlacementFactors,
  ])

  const downloadEditedImage = useCallback(async () => {
    const sourceCanvas = editorSourceCanvasRef.current

    if (
      !sourceCanvas ||
      !editorImageMeta ||
      !normalizedEditorCrop ||
      !editorOutputSize ||
      !editorOutputPlan
    ) {
      return
    }

    setIsEditorExporting(true)

    const cropCanvas = document.createElement('canvas')
    const exportCanvas = document.createElement('canvas')

    try {
      cropCanvas.width = Math.max(1, Math.round(normalizedEditorCrop.width))
      cropCanvas.height = Math.max(1, Math.round(normalizedEditorCrop.height))

      const cropContext = cropCanvas.getContext('2d')

      if (!cropContext) {
        throw new Error('자르기 캔버스를 초기화하지 못했습니다.')
      }

      cropContext.imageSmoothingEnabled = true
      cropContext.imageSmoothingQuality = 'high'
      cropContext.drawImage(
        sourceCanvas,
        normalizedEditorCrop.x,
        normalizedEditorCrop.y,
        normalizedEditorCrop.width,
        normalizedEditorCrop.height,
        0,
        0,
        cropCanvas.width,
        cropCanvas.height,
      )

      exportCanvas.width = editorOutputPlan.outputWidth
      exportCanvas.height = editorOutputPlan.outputHeight

      const exportContext = exportCanvas.getContext('2d')

      if (!exportContext) {
        throw new Error('출력 캔버스를 초기화하지 못했습니다.')
      }

      exportContext.imageSmoothingEnabled = true
      exportContext.imageSmoothingQuality = 'high'
      exportContext.drawImage(
        cropCanvas,
        editorOutputPlan.sourceSampleX,
        editorOutputPlan.sourceSampleY,
        editorOutputPlan.sourceSampleWidth,
        editorOutputPlan.sourceSampleHeight,
        0,
        0,
        editorOutputPlan.outputWidth,
        editorOutputPlan.outputHeight,
      )

      let blob: Blob
      let extension: string

      if (editorOutputFormat === 'pdf') {
        const { jsPDF } = await import('jspdf')
        const pdf = new jsPDF({
          orientation: resolvedEditorPdfOrientation,
          unit: 'mm',
          format: 'a4',
          compress: true,
        })
        const pageWidthMm = pdf.internal.pageSize.getWidth()
        const pageHeightMm = pdf.internal.pageSize.getHeight()
        const safeXMm = EDITOR_PDF_MIN_MARGIN_MM
        const safeYMm = EDITOR_PDF_MIN_MARGIN_MM
        const normalizedCoverage = normalizedEditorPdfCoverage / 100
        const safeWidthMm = Math.max(1, pageWidthMm - EDITOR_PDF_MIN_MARGIN_MM * 2)
        const safeHeightMm = Math.max(1, pageHeightMm - EDITOR_PDF_MIN_MARGIN_MM * 2)
        const placementWidthMm = Math.max(1, safeWidthMm * normalizedCoverage)
        const placementHeightMm = Math.max(1, safeHeightMm * normalizedCoverage)
        const placementXMm =
          safeXMm + (safeWidthMm - placementWidthMm) * editorPdfPlacementFactors.xFactor
        const placementYMm =
          safeYMm + (safeHeightMm - placementHeightMm) * editorPdfPlacementFactors.yFactor
        const fitScale = Math.min(
          placementWidthMm / editorOutputSize.width,
          placementHeightMm / editorOutputSize.height,
        )
        const imageWidthMm = Math.max(1, editorOutputSize.width * fitScale)
        const imageHeightMm = Math.max(1, editorOutputSize.height * fitScale)
        const imageX =
          placementXMm +
          (placementWidthMm - imageWidthMm) * editorPdfPlacementFactors.xFactor
        const imageY =
          placementYMm +
          (placementHeightMm - imageHeightMm) * editorPdfPlacementFactors.yFactor
        const imageData = exportCanvas.toDataURL(
          'image/jpeg',
          getEditorQuality('jpeg', editorQuality),
        )
        pdf.addImage(
          imageData,
          'JPEG',
          imageX,
          imageY,
          imageWidthMm,
          imageHeightMm,
          undefined,
          'MEDIUM',
        )
        blob = new Blob([pdf.output('arraybuffer')], {
          type: getEditorMimeType(editorOutputFormat),
        })
        extension = getEditorExtension(editorOutputFormat)
      } else {
        blob = await canvasToBlob(
          exportCanvas,
          getEditorMimeType(editorOutputFormat),
          getEditorQuality(editorOutputFormat, editorQuality),
        )
        extension = getEditorExtension(editorOutputFormat)
      }

      const downloadUrl = URL.createObjectURL(blob)
      const anchor = document.createElement('a')
      anchor.download = `${stripExtension(editorImageMeta.name)}-edited.${extension}`
      anchor.href = downloadUrl
      anchor.click()
      URL.revokeObjectURL(downloadUrl)

      const inputBytes = editorImageMeta.sizeBytes ?? 0
      const outputBytes = blob.size

      if (inputBytes > 0) {
        const reductionPercent = Math.max(
          0,
          Math.round((1 - outputBytes / inputBytes) * 100),
        )
        const outputSummary =
          editorOutputFormat === 'pdf'
            ? `${editorOutputSize.width}x${editorOutputSize.height} 기반 A4 ${getEditorPdfOrientationLabel(resolvedEditorPdfOrientation)} ${getEditorPdfPlacementLabel(editorPdfPlacement)} (${normalizedEditorPdfCoverage}%)`
            : `${editorOutputSize.width}x${editorOutputSize.height} (${getEditorResizeModeLabel(editorResizeMode)})`
        setEditorStatusMessage(
          `편집 파일 다운로드 완료: ${formatBytes(inputBytes)} → ${formatBytes(outputBytes)} (${reductionPercent}% 절감), ${outputSummary}`,
        )
      } else {
        const outputSummary =
          editorOutputFormat === 'pdf'
            ? `${editorOutputSize.width}x${editorOutputSize.height} 기반 A4 ${getEditorPdfOrientationLabel(resolvedEditorPdfOrientation)} ${getEditorPdfPlacementLabel(editorPdfPlacement)} (${normalizedEditorPdfCoverage}%)`
            : `${editorOutputSize.width}x${editorOutputSize.height} (${getEditorResizeModeLabel(editorResizeMode)})`
        setEditorStatusMessage(
          `편집 파일 다운로드 완료: ${outputSummary}`,
        )
      }
    } catch (error) {
      const message = toErrorMessage(error, '편집 이미지 생성 중 오류가 발생했습니다.')
      setEditorStatusMessage(message)
    } finally {
      cropCanvas.width = 0
      cropCanvas.height = 0
      exportCanvas.width = 0
      exportCanvas.height = 0
      setIsEditorExporting(false)
    }
  }, [
    editorImageMeta,
    editorOutputFormat,
    editorOutputPlan,
    editorOutputSize,
    editorPdfPlacement,
    editorPdfPlacementFactors,
    editorResizeMode,
    resolvedEditorPdfOrientation,
    editorQuality,
    normalizedEditorPdfCoverage,
    normalizedEditorCrop,
  ])

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
  const batchReviewItems = useMemo(
    () =>
      batchResults.filter(
        (item) =>
          item.status === 'done' &&
          Boolean(item.previewUrl) &&
          Boolean(item.outputBlob),
      ),
    [batchResults],
  )
  const batchApprovedCount = batchReviewItems.filter(
    (item) => item.approvedForDownload,
  ).length
  const selectedBatchResult = useMemo(() => {
    if (batchReviewItems.length === 0) {
      return null
    }

    return (
      batchReviewItems.find((item) => item.id === selectedBatchResultId) ??
      batchReviewItems[0]
    )
  }, [batchReviewItems, selectedBatchResultId])
  const editingBatchResult = useMemo(() => {
    if (!editingBatchResultId) {
      return null
    }

    return (
      batchReviewItems.find((item) => item.id === editingBatchResultId) ?? null
    )
  }, [batchReviewItems, editingBatchResultId])

  useEffect(() => {
    if (batchReviewItems.length === 0) {
      if (selectedBatchResultId !== null) {
        setSelectedBatchResultId(null)
      }
      return
    }

    if (!batchReviewItems.some((item) => item.id === selectedBatchResultId)) {
      setSelectedBatchResultId(batchReviewItems[0].id)
    }
  }, [batchReviewItems, selectedBatchResultId])

  useEffect(() => {
    if (isBatchProcessing || batchReviewItems.length === 0 || editingBatchResultId) {
      return
    }

    void openBatchItemForManualEdit(batchReviewItems[0].id)
  }, [
    batchReviewItems,
    editingBatchResultId,
    isBatchProcessing,
    openBatchItemForManualEdit,
  ])

  const hasImage = Boolean(imageMeta)

  const currentPageTitle = useMemo(() => {
    if (activePage === 'tool') {
      return '프라이버시 마스킹'
    }

    if (activePage === 'editor') {
      return '이미지 간편 편집'
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
            <p className="eyebrow">모바일 우선 프라이버시 마스킹</p>
            <h1>변환 전 데이터 안내</h1>
            <p className="hero-description">
              {ENABLE_PLATE_DETECTION
                ? '기능 시작 시 얼굴/번호판 검출 엔진을 내려받습니다. 한 번 준비하면 같은 브라우저에서는 다시 다운로드 없이 바로 변환할 수 있습니다.'
              : '기능 시작 시 얼굴 검출 엔진을 내려받습니다. 한 번 준비하면 같은 브라우저에서는 다시 다운로드 없이 바로 변환할 수 있습니다.'}
            </p>
          </section>

          <KakaoAdBanner slotKey="tool-intro" />

          <section className="consent-card">
            <p className="consent-title">기능 사용 전 확인</p>
            <ul className="consent-list">
              <li>최초 1회 데이터 사용량: 약 6~25MB</li>
              <li>엔진 다운로드 후: 같은 브라우저에서 추가 다운로드 없음</li>
              <li>사진 자체는 서버 업로드 없이 기기 내에서 처리</li>
              <li>기본 출력은 JPG + 크기 최적화(품질 82%, 긴 변 1920px)</li>
              {isMobileMode && <li>{MOBILE_SINGLE_MODE_MESSAGE}</li>}
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
          <h1>잠깐! 업로드 전에 타인의 프라이버시를 존중해 주세요!</h1>
          <p className="hero-description">
            {ENABLE_PLATE_DETECTION
              ? '얼굴과 번호판을 자동으로 찾고 기본값으로 모자이크 처리합니다. 틀린 박스만 빠르게 on/off 하거나 드래그로 추가하면 끝납니다.'
              : '얼굴을 자동으로 찾고 기본값으로 모자이크 처리합니다. 틀린 박스만 빠르게 on/off 하거나 드래그로 추가하면 끝납니다.'}
          </p>
        </section>

        <KakaoAdBanner slotKey="tool-workspace" />

        <section className="cache-assurance">
          {!ENABLE_PLATE_DETECTION
            ? '얼굴 엔진 준비가 끝났습니다. 같은 브라우저에서는 추가 다운로드 없이 안심하고 변환할 수 있습니다. 기본 출력은 데이터 절약 JPG 설정이 적용됩니다.'
            : isPlateDetectorReady
              ? '엔진 준비가 끝났습니다. 같은 브라우저에서는 추가 다운로드 없이 안심하고 변환할 수 있습니다. 기본 출력은 데이터 절약 JPG 설정이 적용됩니다.'
              : isPlateDetectorWarming
                ? '얼굴 엔진 준비가 끝났습니다. 번호판 엔진은 백그라운드에서 준비 중이며 준비 전에는 얼굴 중심 모드로 동작합니다.'
                : isPlateDetectorFailed
                  ? '얼굴 엔진 준비가 끝났습니다. 번호판 엔진 준비에 실패해 얼굴 중심 모드로 동작합니다.'
                  : '얼굴 엔진 준비가 끝났습니다. 번호판 엔진은 곧 백그라운드에서 준비됩니다.'}
        </section>

        {isMobileMode && (
          <section className="mobile-single-notice">{MOBILE_SINGLE_MODE_MESSAGE}</section>
        )}

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
                multiple={!isMobileMode}
                onChange={onPickFile}
                className="file-input"
              />
              <span className="dropzone-title">사진 드래그 또는 클릭 업로드</span>
              <span className="dropzone-subtitle">
                {isMobileMode
                  ? '모바일에서는 한 번에 한 장만 선택할 수 있습니다.'
                  : `2장 이상 선택/드롭 시 JPG ZIP 일괄 처리 모드로 자동 전환됩니다. (최대 ${BATCH_MAX_FILE_COUNT}장, 총 ${formatBytes(BATCH_MAX_TOTAL_BYTES)})`}
              </span>
            </label>

            {!isMobileMode && (
              <input
                ref={batchInputRef}
                type="file"
                accept="image/*"
                multiple
                onChange={onPickBatchFiles}
                className="file-input"
              />
            )}

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
              {!isMobileMode && (
                <button
                  type="button"
                  disabled={isBatchProcessing}
                  onClick={() => batchInputRef.current?.click()}
                >
                  {isBatchProcessing ? '일괄 처리 중...' : '여러 장 ZIP 일괄 처리'}
                </button>
              )}
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
                프라이버시 이미지 다운로드
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

            {batchReviewItems.length > 0 && (
              <section className="batch-review">
                <div className="batch-review-head">
                  <h3>일괄 편집</h3>
                  <p>
                    검토 전용 화면 없이 썸네일을 누르면 바로 편집 캔버스로 전환됩니다.
                    수동 박스 보정 후 저장해 주세요.
                  </p>
                </div>

                {selectedBatchResult && (
                  <p className="batch-review-meta">
                    {selectedBatchResult.fileName} ·{' '}
                    {selectedBatchResult.outputWidth ?? '-'}x
                    {selectedBatchResult.outputHeight ?? '-'} ·{' '}
                    {selectedBatchResult.outputBytes
                      ? formatBytes(selectedBatchResult.outputBytes)
                      : '-'}{' '}
                    ·{' '}
                    {selectedBatchResult.approvedForDownload
                      ? '다운로드 승인됨'
                      : '다운로드 제외됨'}
                  </p>
                )}

                <div className="batch-review-controls">
                  <button
                    type="button"
                    onClick={() => {
                      if (selectedBatchResult) {
                        toggleBatchApproval(selectedBatchResult.id)
                      }
                    }}
                  >
                    {selectedBatchResult?.approvedForDownload
                      ? '선택 이미지 제외'
                      : '선택 이미지 승인'}
                  </button>
                  <button
                    type="button"
                    onClick={() => {
                      void saveBatchItemEdits()
                    }}
                    disabled={!editingBatchResult || isBatchProcessing}
                  >
                    현재 보정 저장
                  </button>
                  <button
                    type="button"
                    onClick={() => setBatchApprovalForAllDone(true)}
                  >
                    전체 승인
                  </button>
                  <button
                    type="button"
                    onClick={() => setBatchApprovalForAllDone(false)}
                  >
                    전체 제외
                  </button>
                </div>

                {editingBatchResult && (
                  <p className="batch-review-edit-note">
                    편집 중: {editingBatchResult.fileName} (수정 후 "현재 보정 저장" 필수)
                  </p>
                )}

                <div className="batch-thumb-grid">
                  {batchReviewItems.map((item) => (
                    <button
                      key={item.id}
                      type="button"
                      className={`batch-thumb ${
                        editingBatchResult?.id === item.id ? 'active' : ''
                      } ${item.approvedForDownload ? 'approved' : 'rejected'}`}
                      onClick={() => {
                        void openBatchItemForManualEdit(item.id)
                      }}
                    >
                      {item.previewUrl && (
                        <img
                          src={item.previewUrl}
                          alt={`${item.fileName} 썸네일`}
                          loading="lazy"
                        />
                      )}
                      <span className="batch-thumb-name">{item.fileName}</span>
                      <span className="batch-thumb-state">
                        {item.approvedForDownload ? '승인됨' : '제외됨'}
                      </span>
                    </button>
                  ))}
                </div>

                <button
                  type="button"
                  className="batch-final-download-btn"
                  onClick={() => {
                    void downloadApprovedBatch()
                  }}
                  disabled={
                    isBatchProcessing ||
                    isBatchDownloadPending ||
                    batchApprovedCount === 0
                  }
                >
                  {isBatchDownloadPending
                    ? 'ZIP 생성 중...'
                    : `최종 승인 ZIP 다운로드 (${batchApprovedCount}/${batchReviewItems.length})`}
                </button>
              </section>
            )}

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
                            ? `얼굴 ${item.faceCount} / 번호판 ${item.plateCount} · ${item.outputWidth ?? '-'}x${item.outputHeight ?? '-'} · ${item.outputBytes ? formatBytes(item.outputBytes) : '-'} · ${item.approvedForDownload ? '승인됨' : '제외됨'}`
                            : `얼굴 ${item.faceCount} · ${item.outputWidth ?? '-'}x${item.outputHeight ?? '-'} · ${item.outputBytes ? formatBytes(item.outputBytes) : '-'} · ${item.approvedForDownload ? '승인됨' : '제외됨'}`)}
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

  const renderEditorPage = () => {
    return (
      <div className="app-shell">
        <section className="hero">
          <p className="eyebrow">브라우저 로컬 편집</p>
          <h1>이미지 크기·용량·자르기 간편 편집</h1>
          <p className="hero-description">
            서버 업로드 없이 한 장씩 빠르게 편집할 수 있습니다. 자르기, 크기 프리셋(70/50/30),
            커스텀 리사이즈(가로/세로 기준·프레임 잘라내기), 형식 변환(JPG/PNG/WEBP/PDF)을
            적용한 뒤 바로 저장하세요. PDF는 A4 스캔 스타일로 변환됩니다.
          </p>
        </section>

        <KakaoAdBanner slotKey="editor" />

        <section className="controls">
          <div className="dropzone-wrapper">
            <label
              className={`dropzone ${isEditorDragging ? 'dragging' : ''}`}
              onDragOver={(event) => {
                event.preventDefault()
                setIsEditorDragging(true)
              }}
              onDragLeave={() => setIsEditorDragging(false)}
              onDrop={onEditorDrop}
            >
              <input
                type="file"
                accept="image/*"
                onChange={onPickEditorFile}
                className="file-input"
              />
              <span className="dropzone-title">이미지 드래그 또는 클릭 업로드</span>
              <span className="dropzone-subtitle">
                간편 편집 페이지는 한 번에 한 장씩 처리합니다.
              </span>
            </label>

            <input
              ref={editorInputRef}
              type="file"
              accept="image/*"
              onChange={onPickEditorFile}
              className="file-input"
            />

            <div className="toolbar editor-toolbar">
              <button
                type="button"
                onClick={() => editorInputRef.current?.click()}
              >
                이미지 다시 선택
              </button>
              <button
                type="button"
                className={isEditorCropMode ? 'active' : ''}
                disabled={!editorImageMeta}
                onClick={() => {
                  setIsEditorCropMode((prev) => !prev)
                  setEditorCropDraft(null)
                }}
              >
                {isEditorCropMode ? '자르기 선택 종료' : '자르기 영역 지정'}
              </button>
              <button
                type="button"
                disabled={!editorImageMeta}
                onClick={resetEditorCrop}
              >
                자르기 초기화
              </button>
            </div>
          </div>

          <div className="picker-row editor-picker-row">
            <div className="picker-group">
              <span className="picker-label">출력 옵션</span>
              <div className="optimize-box">
                <label className="optimize-field">
                  파일 형식
                  <strong>{getEditorFormatLabel(editorOutputFormat)}</strong>
                </label>
                <select
                  value={editorOutputFormat}
                  onChange={(event) =>
                    setEditorOutputFormat(event.target.value as EditorOutputFormat)
                  }
                >
                  <option value="jpeg">JPG (용량 절감)</option>
                  <option value="png">PNG (무손실)</option>
                  <option value="webp">WEBP (최신 포맷)</option>
                  <option value="pdf">PDF (A4 문서형)</option>
                </select>

                <label className="optimize-field">
                  크기 프리셋
                  <strong>{getEditorResizeModeLabel(editorResizeMode)}</strong>
                </label>
                <div className="resize-mode-grid" role="group" aria-label="이미지 크기 프리셋">
                  {EDITOR_RESIZE_MODE_OPTIONS.map((option) => (
                    <button
                      key={option.value}
                      type="button"
                      className={editorResizeMode === option.value ? 'active' : ''}
                      onClick={() => setEditorResizeMode(option.value)}
                    >
                      {option.label}
                    </button>
                  ))}
                </div>

                {editorResizeMode === 'custom' ? (
                  <>
                    <label className="optimize-field">
                      Custom 방식
                      <strong>
                        {editorCustomResizeMode === 'fit-width'
                          ? '가로 기준'
                          : editorCustomResizeMode === 'fit-height'
                            ? '세로 기준'
                            : '사용자 프레임'}
                      </strong>
                    </label>
                    <select
                      value={editorCustomResizeMode}
                      onChange={(event) =>
                        setEditorCustomResizeMode(
                          event.target.value as EditorCustomResizeMode,
                        )
                      }
                    >
                      <option value="fit-width">비율 고정 (가로 기준)</option>
                      <option value="fit-height">비율 고정 (세로 기준)</option>
                      <option value="frame">사용자 프레임 (가로x세로)</option>
                    </select>

                    {editorCustomResizeMode === 'fit-width' ? (
                      <>
                        <label className="optimize-field">
                          가로(px)
                          <strong>{editorCustomWidth}px</strong>
                        </label>
                        <input
                          type="number"
                          min={Math.min(
                            EDITOR_CUSTOM_MIN_DIMENSION,
                            editorCustomDimensionLimits?.maxWidth ??
                              EDITOR_CUSTOM_MAX_DIMENSION,
                          )}
                          max={editorCustomDimensionLimits?.maxWidth ?? EDITOR_CUSTOM_MAX_DIMENSION}
                          value={editorCustomWidth}
                          onChange={(event) => {
                            const parsed = Number.parseInt(event.target.value, 10)
                            setEditorCustomWidth(
                              Number.isFinite(parsed) ? parsed : EDITOR_CUSTOM_MIN_DIMENSION,
                            )
                          }}
                        />
                      </>
                    ) : null}

                    {editorCustomResizeMode === 'fit-height' ? (
                      <>
                        <label className="optimize-field">
                          세로(px)
                          <strong>{editorCustomHeight}px</strong>
                        </label>
                        <input
                          type="number"
                          min={Math.min(
                            EDITOR_CUSTOM_MIN_DIMENSION,
                            editorCustomDimensionLimits?.maxHeight ??
                              EDITOR_CUSTOM_MAX_DIMENSION,
                          )}
                          max={
                            editorCustomDimensionLimits?.maxHeight ??
                            EDITOR_CUSTOM_MAX_DIMENSION
                          }
                          value={editorCustomHeight}
                          onChange={(event) => {
                            const parsed = Number.parseInt(event.target.value, 10)
                            setEditorCustomHeight(
                              Number.isFinite(parsed) ? parsed : EDITOR_CUSTOM_MIN_DIMENSION,
                            )
                          }}
                        />
                      </>
                    ) : null}

                    {editorCustomResizeMode === 'frame' ? (
                      <>
                        <label className="optimize-field">
                          프레임 가로(px)
                          <strong>{editorCustomWidth}px</strong>
                        </label>
                        <input
                          type="number"
                          min={Math.min(
                            EDITOR_CUSTOM_MIN_DIMENSION,
                            editorCustomDimensionLimits?.maxWidth ??
                              EDITOR_CUSTOM_MAX_DIMENSION,
                          )}
                          max={editorCustomDimensionLimits?.maxWidth ?? EDITOR_CUSTOM_MAX_DIMENSION}
                          value={editorCustomWidth}
                          onChange={(event) => {
                            const parsed = Number.parseInt(event.target.value, 10)
                            setEditorCustomWidth(
                              Number.isFinite(parsed) ? parsed : EDITOR_CUSTOM_MIN_DIMENSION,
                            )
                          }}
                        />

                        <label className="optimize-field">
                          프레임 세로(px)
                          <strong>{editorCustomHeight}px</strong>
                        </label>
                        <input
                          type="number"
                          min={Math.min(
                            EDITOR_CUSTOM_MIN_DIMENSION,
                            editorCustomDimensionLimits?.maxHeight ??
                              EDITOR_CUSTOM_MAX_DIMENSION,
                          )}
                          max={
                            editorCustomDimensionLimits?.maxHeight ??
                            EDITOR_CUSTOM_MAX_DIMENSION
                          }
                          value={editorCustomHeight}
                          onChange={(event) => {
                            const parsed = Number.parseInt(event.target.value, 10)
                            setEditorCustomHeight(
                              Number.isFinite(parsed) ? parsed : EDITOR_CUSTOM_MIN_DIMENSION,
                            )
                          }}
                        />

                        <label className="optimize-field">
                          프레임 처리
                          <strong>
                            {editorCustomFrameFit === 'cover'
                              ? '프레임 채움(잘림)'
                              : '비율 유지(안 잘림)'}
                          </strong>
                        </label>
                        <select
                          value={editorCustomFrameFit}
                          onChange={(event) =>
                            setEditorCustomFrameFit(
                              event.target.value as EditorCustomFrameFit,
                            )
                          }
                        >
                          <option value="contain">비율 유지 (프레임 안 맞춤)</option>
                          <option value="cover">프레임 채우기 (넘치는 영역 잘림)</option>
                        </select>

                        {editorCustomFrameFit === 'cover' ? (
                          <>
                            <label className="optimize-field">
                              잘림 기준 위치
                              <strong>
                                {getEditorPdfPlacementLabel(editorCustomPlacement)}
                              </strong>
                            </label>
                            <div
                              className="pdf-position-grid"
                              role="group"
                              aria-label="잘림 기준 위치"
                            >
                              {EDITOR_PDF_PLACEMENT_OPTIONS.map((option) => (
                                <button
                                  key={`custom-${option.value}`}
                                  type="button"
                                  className={
                                    editorCustomPlacement === option.value ? 'active' : ''
                                  }
                                  onClick={() => setEditorCustomPlacement(option.value)}
                                  title={option.label}
                                  aria-label={option.label}
                                >
                                  <span aria-hidden="true">{option.symbol}</span>
                                </button>
                              ))}
                            </div>
                          </>
                        ) : null}
                      </>
                    ) : null}
                  </>
                ) : null}

                {editorOutputFormat === 'pdf' ? (
                  <>
                    <label className="optimize-field">
                      A4 방향
                      <strong>{getEditorPdfOrientationLabel(editorPdfOrientation)}</strong>
                    </label>
                    <select
                      value={editorPdfOrientation}
                      onChange={(event) =>
                        setEditorPdfOrientation(event.target.value as EditorPdfOrientation)
                      }
                    >
                      <option value="auto">자동</option>
                      <option value="portrait">세로</option>
                      <option value="landscape">가로</option>
                    </select>

                    <label className="optimize-field">
                      배치 위치
                      <strong>{getEditorPdfPlacementLabel(editorPdfPlacement)}</strong>
                    </label>
                    <div className="pdf-position-grid" role="group" aria-label="A4 배치 위치">
                      {EDITOR_PDF_PLACEMENT_OPTIONS.map((option) => (
                        <button
                          key={option.value}
                          type="button"
                          className={editorPdfPlacement === option.value ? 'active' : ''}
                          onClick={() => setEditorPdfPlacement(option.value)}
                          title={option.label}
                          aria-label={option.label}
                        >
                          <span aria-hidden="true">{option.symbol}</span>
                        </button>
                      ))}
                    </div>

                    <label className="optimize-field">
                      A4 채움 비율
                      <strong>{editorPdfCoverage}%</strong>
                    </label>
                    <input
                      type="range"
                      min={EDITOR_PDF_MIN_COVERAGE}
                      max={100}
                      step={1}
                      value={editorPdfCoverage}
                      onChange={(event) =>
                        setEditorPdfCoverage(Number(event.target.value))
                      }
                    />
                  </>
                ) : null}

                <label className="optimize-field">
                  품질
                  <strong>
                    {editorOutputFormat === 'png'
                      ? '무손실'
                      : `${Math.round(editorQuality * 100)}%`}
                  </strong>
                </label>
                <input
                  type="range"
                  min={60}
                  max={96}
                  step={1}
                  value={Math.round(editorQuality * 100)}
                  onChange={(event) =>
                    setEditorQuality(Number(event.target.value) / 100)
                  }
                  disabled={editorOutputFormat === 'png'}
                />

              </div>
            </div>
          </div>
        </section>

        <section className="workspace">
          <div className="canvas-panel">
            <div
              className="canvas-stage"
              style={
                editorImageMeta
                  ? { aspectRatio: `${editorImageMeta.width} / ${editorImageMeta.height}` }
                  : undefined
              }
            >
              {editorImageMeta ? (
                <>
                  <canvas ref={editorCanvasRef} className="preview-canvas" />
                  <svg
                    className={`overlay ${isEditorCropMode ? 'draw-mode' : ''}`}
                    viewBox={`0 0 ${editorImageMeta.width} ${editorImageMeta.height}`}
                    preserveAspectRatio="none"
                    onPointerDown={onEditorOverlayPointerDown}
                    onPointerMove={onEditorOverlayPointerMove}
                    onPointerUp={onEditorOverlayPointerUp}
                    onPointerCancel={() => {
                      editorCropStartRef.current = null
                      setEditorCropDraft(null)
                    }}
                  >
                    {editorDisplayCrop && (
                      <>
                        <rect
                          x={0}
                          y={0}
                          width={editorImageMeta.width}
                          height={editorDisplayCrop.y}
                          className="editor-crop-mask"
                        />
                        <rect
                          x={0}
                          y={editorDisplayCrop.y}
                          width={editorDisplayCrop.x}
                          height={editorDisplayCrop.height}
                          className="editor-crop-mask"
                        />
                        <rect
                          x={editorDisplayCrop.x + editorDisplayCrop.width}
                          y={editorDisplayCrop.y}
                          width={Math.max(
                            0,
                            editorImageMeta.width -
                              (editorDisplayCrop.x + editorDisplayCrop.width),
                          )}
                          height={editorDisplayCrop.height}
                          className="editor-crop-mask"
                        />
                        <rect
                          x={0}
                          y={editorDisplayCrop.y + editorDisplayCrop.height}
                          width={editorImageMeta.width}
                          height={Math.max(
                            0,
                            editorImageMeta.height -
                              (editorDisplayCrop.y + editorDisplayCrop.height),
                          )}
                          className="editor-crop-mask"
                        />
                        <rect
                          x={editorDisplayCrop.x}
                          y={editorDisplayCrop.y}
                          width={editorDisplayCrop.width}
                          height={editorDisplayCrop.height}
                          className={`editor-crop-box ${editorCropDraft ? 'draft' : ''}`}
                          vectorEffect="non-scaling-stroke"
                        />
                      </>
                    )}
                  </svg>
                </>
              ) : (
                <div className="empty-state">
                  이미지를 올리면 편집 미리보기가 여기에 표시됩니다.
                </div>
              )}
            </div>

            {editorOutputPlan && editorResizePreviewLayout ? (
              <div className="resize-preview-card">
                <div className="resize-preview-head">
                  <strong>크기 변경 미리보기</strong>
                  <span>
                    {`원본 대비 ${editorOutputPlan.areaScalePercent}% 유지 · ${editorOutputPlan.reducedAreaPercent}% 감소`}
                  </span>
                </div>
                <div
                  className="resize-preview-stage"
                  style={{ aspectRatio: editorResizePreviewLayout.stageAspectRatio }}
                >
                  <div className="resize-source-frame" />
                  {editorResizePreviewLayout.hasSourceCrop ? (
                    <div
                      className="resize-source-sample"
                      style={{
                        left: `${editorResizePreviewLayout.sampleLeftPercent}%`,
                        top: `${editorResizePreviewLayout.sampleTopPercent}%`,
                        width: `${editorResizePreviewLayout.sampleWidthPercent}%`,
                        height: `${editorResizePreviewLayout.sampleHeightPercent}%`,
                      }}
                    />
                  ) : null}
                  <div
                    className="resize-output-footprint"
                    style={{
                      left: `${editorResizePreviewLayout.outputLeftPercent}%`,
                      top: `${editorResizePreviewLayout.outputTopPercent}%`,
                      width: `${editorResizePreviewLayout.outputWidthPercent}%`,
                      height: `${editorResizePreviewLayout.outputHeightPercent}%`,
                    }}
                  >
                    <span>
                      {`${editorOutputPlan.outputWidth}x${editorOutputPlan.outputHeight}`}
                    </span>
                  </div>
                </div>
                <div className="resize-progress-track">
                  <span
                    style={{
                      width: `${Math.max(2, Math.min(100, editorOutputPlan.areaScalePercent))}%`,
                    }}
                  />
                </div>
                <p className="resize-preview-caption">
                  {`가로 ${editorOutputPlan.widthScalePercent}% · 세로 ${editorOutputPlan.heightScalePercent}%`}
                </p>
                {editorResizePreviewLayout.hasSourceCrop ? (
                  <p className="resize-preview-caption">
                    {`원본 사용 영역 ${editorOutputPlan.sampleAreaPercent}% (프레임 채우기 잘림 적용)`}
                  </p>
                ) : null}
              </div>
            ) : null}

            {editorOutputFormat === 'pdf' && editorPdfPreviewLayout ? (
              <div className="pdf-layout-card">
                <div className="pdf-layout-head">
                  <strong>A4 배치 미리보기</strong>
                  <span>
                    {`${getEditorPdfOrientationLabel(resolvedEditorPdfOrientation)} / ${getEditorPdfPlacementLabel(editorPdfPlacement)} / 채움 ${normalizedEditorPdfCoverage}%`}
                  </span>
                </div>
                <div
                  className="pdf-page-preview"
                  style={{ aspectRatio: editorPdfPreviewLayout.pageAspectRatio }}
                >
                  <div
                    className="pdf-safe-zone"
                    style={{
                      left: `${editorPdfPreviewLayout.safeLeftPercent}%`,
                      top: `${editorPdfPreviewLayout.safeTopPercent}%`,
                      width: `${editorPdfPreviewLayout.safeWidthPercent}%`,
                      height: `${editorPdfPreviewLayout.safeHeightPercent}%`,
                    }}
                  />
                  <div
                    className="pdf-placement-zone"
                    style={{
                      left: `${editorPdfPreviewLayout.placementLeftPercent}%`,
                      top: `${editorPdfPreviewLayout.placementTopPercent}%`,
                      width: `${editorPdfPreviewLayout.placementWidthPercent}%`,
                      height: `${editorPdfPreviewLayout.placementHeightPercent}%`,
                    }}
                  />
                  <div
                    className="pdf-image-footprint"
                    style={{
                      left: `${editorPdfPreviewLayout.imageLeftPercent}%`,
                      top: `${editorPdfPreviewLayout.imageTopPercent}%`,
                      width: `${editorPdfPreviewLayout.imageWidthPercent}%`,
                      height: `${editorPdfPreviewLayout.imageHeightPercent}%`,
                    }}
                  >
                    <span>
                      {`${editorOutputSize?.width ?? 0}x${editorOutputSize?.height ?? 0}`}
                    </span>
                  </div>
                </div>
                <p className="pdf-layout-caption">
                  {`실제 페이지 점유 면적 약 ${editorPdfPreviewLayout.imageAreaPercent}%`}
                </p>
              </div>
            ) : null}

            <div className="action-row">
              <button
                type="button"
                onClick={() => {
                  void downloadEditedImage()
                }}
                disabled={!editorImageMeta || isEditorExporting}
              >
                {isEditorExporting ? '편집 파일 생성 중...' : '편집 결과 다운로드'}
              </button>
            </div>
          </div>

          <aside className="region-panel editor-panel">
            <div className="panel-header">
              <h2>편집 정보</h2>
              <p>
                {editorImageMeta
                  ? `원본 ${editorImageMeta.width}x${editorImageMeta.height}`
                  : '이미지를 먼저 업로드해 주세요'}
              </p>
            </div>

            <div className="status-box">{editorStatusMessage}</div>

            {editorImageMeta && normalizedEditorCrop && editorOutputSize ? (
              <ul className="editor-info-list">
                <li>
                  <span>원본 파일</span>
                  <strong>{editorImageMeta.name}</strong>
                </li>
                <li>
                  <span>원본 용량</span>
                  <strong>{formatBytes(editorImageMeta.sizeBytes ?? 0)}</strong>
                </li>
                <li>
                  <span>자르기 영역</span>
                  <strong>
                    {Math.round(normalizedEditorCrop.width)}x
                    {Math.round(normalizedEditorCrop.height)}
                  </strong>
                </li>
                <li>
                  <span>출력 형식</span>
                  <strong>{getEditorFormatLabel(editorOutputFormat)}</strong>
                </li>
                <li>
                  <span>출력 품질</span>
                  <strong>
                    {editorOutputFormat === 'png'
                      ? '무손실'
                      : `${Math.round(editorQuality * 100)}%`}
                  </strong>
                </li>
                <li>
                  <span>크기 모드</span>
                  <strong>{getEditorResizeModeLabel(editorResizeMode)}</strong>
                </li>
                {editorOutputPlan ? (
                  <li>
                    <span>원본 대비</span>
                    <strong>
                      {`${editorOutputPlan.widthScalePercent}% x ${editorOutputPlan.heightScalePercent}%`}
                    </strong>
                  </li>
                ) : null}
                {editorOutputPlan ? (
                  <li>
                    <span>픽셀 면적</span>
                    <strong>
                      {`${editorOutputPlan.areaScalePercent}% 유지 / ${editorOutputPlan.reducedAreaPercent}% 감소`}
                    </strong>
                  </li>
                ) : null}
                {editorOutputFormat === 'pdf' ? (
                  <li>
                    <span>PDF 페이지</span>
                    <strong>
                      {`A4 ${getEditorPdfOrientationLabel(resolvedEditorPdfOrientation)} (${normalizedEditorPdfCoverage}%)`}
                    </strong>
                  </li>
                ) : null}
                {editorOutputFormat === 'pdf' ? (
                  <li>
                    <span>PDF 배치</span>
                    <strong>{getEditorPdfPlacementLabel(editorPdfPlacement)}</strong>
                  </li>
                ) : null}
                <li>
                  <span>예상 출력 해상도</span>
                  <strong>
                    {editorOutputSize.width}x{editorOutputSize.height}
                  </strong>
                </li>
              </ul>
            ) : (
              <p className="panel-empty">
                자르기 영역을 지정하고 출력 옵션을 조정해 주세요.
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
          <KakaoAdBanner slotKey="about" />
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
          <KakaoAdBanner slotKey="privacy" />
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
        <KakaoAdBanner slotKey="contact" />
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
          {activePage === 'tool'
            ? renderToolPage()
            : activePage === 'editor'
              ? renderEditorPage()
              : renderInformationPage()}
        </div>
      </section>
    </main>
  )
}

export default App
