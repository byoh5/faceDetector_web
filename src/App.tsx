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
const ENGINE_CACHE_KEY = 'face_masker_engine_assets_v1'
const ENABLE_PLATE_DETECTION = false
const FACE_WARMUP_TIMEOUT_MS = 25_000
const PLATE_WARMUP_TIMEOUT_MS = 35_000
const PLATE_DETECTION_TIMEOUT_MS = 20_000

type BatchStatus = 'pending' | 'processing' | 'done' | 'failed'

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
  const [statusMessage, setStatusMessage] = useState(
    ENABLE_PLATE_DETECTION
      ? '사진을 올리면 얼굴+번호판 자동 가림이 시작됩니다.'
      : '사진을 올리면 얼굴 자동 가림이 시작됩니다. (번호판 기능 임시 보류)',
  )

  const canUseShareSheet =
    typeof navigator !== 'undefined' && typeof navigator.share === 'function'

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
  }, [isPreparing])

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

      setBatchResults(initialResults)
      setStatusMessage(
        plateDetectionEnabled
          ? `${imageFiles.length}장 일괄 처리를 시작합니다.`
          : ENABLE_PLATE_DETECTION
            ? `${imageFiles.length}장 일괄 처리를 시작합니다. (번호판 엔진 비활성화: 얼굴 중심 모드)`
            : `${imageFiles.length}장 일괄 처리를 시작합니다. (번호판 기능 임시 보류: 얼굴 중심 모드)`,
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

          try {
            inputBytesTotal += file.size

            const loaded = await loadImageFromFile(file)
            objectUrl = loaded.objectUrl

            const sourceCanvas = createSourceCanvas(loaded.image)
            let plateScanErrorMessage = ''

            const [faces, plates] = await Promise.all([
              detectFaces(loaded.image),
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
                      errorMessage: message,
                    }
                  : item,
              ),
            )
          } finally {
            if (objectUrl) {
              URL.revokeObjectURL(objectUrl)
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
    [exportOptions, isPlateDetectorReady, isPrepared, isScanning, maskStyle],
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
      const file = event.target.files?.[0]

      if (!file) {
        return
      }

      await loadFile(file)
      event.target.value = ''
    },
    [loadFile],
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

  if (!isPrepared) {
    return (
      <main className="app-shell intro-shell">
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

          <button
            type="button"
            className="primary-btn"
            onClick={() => {
              void prepareTransformer()
            }}
            disabled={isPreparing}
          >
            {isPreparing ? '엔진 준비 중...' : '변환하러가기'}
          </button>
        </section>
      </main>
    )
  }

  return (
    <main className="app-shell">
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
              onChange={onPickFile}
              className="file-input"
            />
            <span className="dropzone-title">사진 드래그 또는 클릭 업로드</span>
            <span className="dropzone-subtitle">
              2장 이상 드롭 시 JPG ZIP 일괄 처리 모드로 자동 전환됩니다.
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
            <p className="panel-empty">자동 스캔 결과가 없으면 수동 박스로 추가해 주세요.</p>
          )}
        </aside>
      </section>
    </main>
  )
}

export default App
